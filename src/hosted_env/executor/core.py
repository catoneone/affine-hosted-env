"""worker / env-server core (design §6, §8 data plane, §9).

One rollout:
  start (seq 0): verify session token -> confirm with the gateway that the session is still valid -> fetch task
      prompt -> start container, setup -> render the task prompt (P).
  turn (seq 1, 2, ...): receive model tokens (M) -> parse tool calls, execute, render tool output (T) and return it;
      a model turn with no tool call means submit (completed); not ending on a stop token, or running out of
      steps / context / time, is recorded as unresolved (still graded); a broken container, setup or grading env
      is recorded as infra_error. At the end: grade, sign the receipt, report to the gateway first, then return to
      the miner.
Each operation runs in a thread pool and the HTTP layer long-polls; resending the same tokens for the same seq returns
the same result (safe for network retries), and non-consecutive seqs are rejected. Failures before state changes
(gateway temporarily unreachable) can be retried as-is; any exception after state changes is recorded as infra_error.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field

from nacl.signing import SigningKey, VerifyKey

from ..auth import AuthError, verify_token
from ..common import canonical, key_id, sha256_hex
from ..envs import SetupError
from ..receipt import TrajectoryHasher, sign_receipt
from ..render import TemplateRenderer
from ..providers.sandbox import DockerSandbox, ImageCache, LocalSandbox
from ..toolcall import parse_tool_calls, split_reasoning
from .gateway_client import GatewayRejected, GatewayUnavailable, WorkerGatewayClient

log = logging.getLogger('hosted_env.executor')


class WorkerError(Exception):
    def __init__(self, code: str, status: int = 400, detail: str = ''):
        super().__init__(f'{code}: {detail}' if detail else code)
        self.code, self.status, self.detail = code, status, detail


@dataclass
class EnvRuntime:
    """Runtime parameters of one env on the worker (told to the miner via /start; fixed by the opening in production)."""
    env: object
    renderer: TemplateRenderer
    max_context: int = 65536
    max_turn_tokens: int = 16384
    max_steps: int = 100
    max_seconds: int = 4 * 3600
    min_room: int = 1024             # stop when less than this much context remains (same as the EXP-014 harness)
    sandbox: str = 'docker'          # docker / local (local is for tests only)
    sandbox_opts: dict = field(default_factory=dict)


@dataclass
class _Op:
    seq: int
    digest: str
    future: cf.Future


class RolloutSession:
    def __init__(self, claims: dict, rt: EnvRuntime):
        self.claims, self.rt = claims, rt
        self.sid = claims['sid']
        self.lock = threading.Lock()
        self.hasher = TrajectoryHasher()
        self.total = 0
        self.steps = 0
        self.started = time.time()
        self.last_activity = self.started
        self.envsess = None
        self.image = None
        self.ops: dict[int, _Op] = {}
        self.last_seq = -1
        self.final: dict | None = None
        self.stats = {'tool_calls': 0, 'tool_errors': 0, 'format_errors': 0, 'tool_seconds': 0.0,
                      'm_tokens': 0, 't_tokens': 0}

    def busy(self) -> bool:
        op = self.ops.get(self.last_seq)
        return op is not None and not op.future.done()


class Worker:
    def __init__(self, *, worker_id: str, key: SigningKey, gateway: WorkerGatewayClient, gateway_pub: VerifyKey,
                 runtimes: dict[str, EnvRuntime], public_url: str, capacity: int = 16, idle_ttl: int = 15 * 60,
                 retention: int = 3600, log_dir: str | None = None, image_cache: ImageCache | None = None,
                 max_threads: int = 64, scratch: str | None = None):
        self.worker_id, self.key, self.gw, self.gateway_pub = worker_id, key, gateway, gateway_pub
        self.key_id = key_id(key.verify_key)
        self.runtimes, self.public_url, self.capacity = runtimes, public_url, capacity
        self.idle_ttl, self.retention = idle_ttl, retention
        self.images = image_cache or ImageCache()
        self.scratch = scratch
        self.pool = cf.ThreadPoolExecutor(max_threads, thread_name_prefix='rollout')
        self.sessions: dict[str, RolloutSession] = {}
        self.mu = threading.Lock()
        self._spec_cache: dict[str, tuple[float, dict]] = {}
        self._log_path = os.path.join(log_dir, f'exec-{worker_id}.jsonl') if log_dir else None
        self._log_mu = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------ background threads
    def start_background(self, heartbeat_every: float = 20.0, keeper_every: float = 60.0) -> None:
        def loop(fn, every):
            while not self._stop.wait(every):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    log.exception('background task failed')
        self.heartbeat()
        for fn, every in ((self.heartbeat, heartbeat_every), (self.keep_alive, keeper_every)):
            t = threading.Thread(target=loop, args=(fn, every), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        with self.mu:
            sessions = list(self.sessions.values())
        for s in sessions:
            if s.final is None and not s.busy():
                self._abandon(s, 'worker_shutdown')
        self.pool.shutdown(wait=False, cancel_futures=True)

    def heartbeat(self) -> None:
        self.gw.heartbeat(self.public_url, self.capacity, self.images.cached(), sorted(self.runtimes))

    def keep_alive(self) -> None:
        """Renew sessions with the gateway during long operations (pulling images, running tests); abandon sessions idle too long; drop sessions finished long ago from memory."""
        now = time.time()
        with self.mu:
            sessions = list(self.sessions.values())
        for s in sessions:
            if s.final is not None:
                if now - s.last_activity > self.retention:
                    with self.mu:
                        self.sessions.pop(s.sid, None)
            elif s.busy():
                if not self.gw.touch(s.sid):
                    log.warning('gateway no longer knows session %s', s.sid)
            elif now - s.last_activity > self.idle_ttl:
                self._abandon(s, 'idle_timeout')

    def _abandon(self, s: RolloutSession, reason: str) -> None:
        with s.lock:
            if s.final is not None:
                return
            s.final = {'status': 'ok', 'done': True, 'outcome': 'abandoned', 'detail': reason, 'receipt': None}
        self._teardown(s)
        try:
            self.gw.close(s.sid, 'abandoned', None)
        except GatewayUnavailable:
            pass

    def _teardown(self, s: RolloutSession) -> None:
        if s.envsess is not None:
            try:
                s.envsess.close()
            except Exception:  # noqa: BLE001
                log.exception('sandbox cleanup failed')
            s.envsess = None
        if s.image and s.rt.sandbox == 'docker':
            self.images.release(s.image)
            s.image = None

    def _log(self, rec: dict) -> None:
        if self._log_path:
            with self._log_mu, open(self._log_path, 'a') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # ------------------------------------------------------------ entry points (called by the HTTP layer, return a Future)
    def _claims(self, token: str) -> dict:
        try:
            claims = verify_token(self.gateway_pub, token)
        except AuthError as e:
            raise WorkerError(e.code, 401) from None
        if claims.get('worker_id') != self.worker_id:
            raise WorkerError('wrong_worker', 403)
        if claims.get('env_id') not in self.runtimes:
            raise WorkerError('unknown_env', 404)
        return claims

    def start(self, token: str) -> cf.Future:
        claims = self._claims(token)
        with self.mu:
            s = self.sessions.get(claims['sid'])
            if s is None:
                live = sum(1 for x in self.sessions.values() if x.final is None)
                if live >= self.capacity:
                    raise WorkerError('worker_full', 503)
                s = RolloutSession(claims, self.runtimes[claims['env_id']])
                self.sessions[s.sid] = s
        with s.lock:
            op = s.ops.get(0)
            if op is None or (op.future.done() and op.future.exception() is not None and s.last_seq == 0):
                op = _Op(0, '', self.pool.submit(self._do_start, s))
                s.ops[0], s.last_seq = op, 0
            s.last_activity = time.time()
            return op.future

    def turn(self, token: str, seq: int, tokens) -> cf.Future:
        claims = self._claims(token)
        with self.mu:
            s = self.sessions.get(claims['sid'])
        if s is None:
            raise WorkerError('unknown_session', 404, 'call /v1/start first')
        if type(seq) is not int:
            raise WorkerError('bad_seq', 400)
        digest = sha256_hex(canonical(tokens))
        with s.lock:
            s.last_activity = time.time()
            op = s.ops.get(seq)
            if op is not None and op.digest == digest and seq == s.last_seq:
                if not (op.future.done() and op.future.exception() is not None):
                    return op.future                      # resend: return the same result
                del s.ops[seq]                            # last attempt failed before changing state, allow retry
                s.last_seq = seq - 1
            if s.final is not None:
                raise WorkerError('rollout_done', 409)
            if seq != s.last_seq + 1:
                raise WorkerError('bad_seq', 409, f'expected seq {s.last_seq + 1}')
            prev = s.ops.get(s.last_seq)
            if prev is None or not prev.future.done() or prev.future.exception() is not None:
                raise WorkerError('previous_step_pending', 409)
            self._validate_turn(s, tokens)
            op = _Op(seq, digest, self.pool.submit(self._do_turn, s, seq, list(tokens)))
            s.ops = {k: v for k, v in s.ops.items() if k == 0}
            s.ops[seq], s.last_seq = op, seq
            return op.future

    def _validate_turn(self, s: RolloutSession, tokens) -> None:
        r, rt = s.rt.renderer, s.rt
        if not isinstance(tokens, list) or not tokens:
            raise WorkerError('bad_tokens', 400, 'tokens must be a non-empty list')
        if any(type(t) is not int or not 0 <= t < r.vocab_size for t in tokens):
            raise WorkerError('bad_tokens', 400, 'token ids out of vocabulary')
        if len(tokens) > rt.max_turn_tokens:
            raise WorkerError('turn_too_long', 400, f'max {rt.max_turn_tokens} tokens per turn')
        if s.total + len(tokens) > rt.max_context:
            raise WorkerError('context_overflow', 400, f'context would exceed {rt.max_context}')
        stops = set(r.stop_ids)
        if any(t in stops for t in tokens[:-1]):
            raise WorkerError('stop_token_inside', 400, 'sampling must stop at the first stop token')

    # ------------------------------------------------------------ operations
    def _task_spec(self, task_id: str) -> dict:
        now = time.time()
        hit = self._spec_cache.get(task_id)
        if hit and now - hit[0] < 6 * 3600:
            return hit[1]
        spec = self.gw.task_spec(task_id)
        self._spec_cache[task_id] = (now, spec)
        return spec

    def _sandbox_factory(self, rt: EnvRuntime):
        opts = dict(rt.sandbox_opts)
        if rt.sandbox == 'docker':
            return lambda image, cwd: DockerSandbox(image, cwd, **opts)
        if rt.sandbox == 'local':
            return lambda image, cwd: LocalSandbox(image, cwd, base_dir=self.scratch)
        raise ValueError(f'unknown sandbox {rt.sandbox!r}')

    def _do_start(self, s: RolloutSession) -> dict:
        c, rt = s.claims, s.rt
        if not self.gw.touch(s.sid):                       # before changing state: failures can be retried
            raise WorkerError('session_closed', 410)
        try:
            spec = self._task_spec(c['task_id'])
        except GatewayRejected:
            raise WorkerError('session_closed', 410) from None
        if spec['env_id'] != c['env_id'] or spec['task'] != c['task']:
            raise WorkerError('spec_mismatch', 500)
        t0 = time.time()
        try:
            if rt.sandbox == 'docker':
                self.images.acquire(spec['spec']['image'])
                s.image = spec['spec']['image']
            s.envsess = rt.env.open(spec['spec'], self._sandbox_factory(rt))
            s.envsess.setup()
            ids = rt.renderer.prompt(s.envsess.initial_messages(), s.envsess.tools)
            if len(ids) + rt.min_room > rt.max_context:
                raise SetupError('prompt does not fit the context budget')
        except Exception as e:  # noqa: BLE001  container / setup / render errors are all infra errors
            return self._finish_infra(s, f'setup: {type(e).__name__}: {e}'[:500])
        s.hasher.add('P', ids)
        s.total = len(ids)
        self._log({'sid': s.sid, 'event': 'start', 'task': c['task'], 'env_id': c['env_id'],
                   'prompt_tokens': len(ids), 'setup_seconds': round(time.time() - t0, 2)})
        return {'status': 'ok', 'seq': 0, 'done': False, 'prompt_tokens': ids,
                'stop_token_ids': rt.renderer.stop_ids, 'max_context': rt.max_context,
                'max_turn_tokens': rt.max_turn_tokens, 'max_steps': rt.max_steps}

    def _do_turn(self, s: RolloutSession, seq: int, tokens: list[int]) -> dict:
        rt, r = s.rt, s.rt.renderer
        if not self.gw.touch(s.sid):                       # before changing state: failures can be retried
            raise WorkerError('session_closed', 410)
        s.hasher.add('M', tokens)
        s.total += len(tokens)
        s.steps += 1
        s.stats['m_tokens'] += len(tokens)
        try:
            text = r.decode_turn(tokens)
            _, content, _ = split_reasoning(text, r.prompt_opens_think)
            _, calls, err = parse_tool_calls(content)
            if tokens[-1] not in r.stop_ids:
                return self._finish(s, seq, 'unresolved', 'response_truncated')
            if tokens[-1] != r.turn_end_id:                 # stopped on another EOS: do not continue
                return self._finish(s, seq, 'completed' if not (calls or err) else 'unresolved', 'eos')
            if not calls and not err:
                return self._finish(s, seq, 'completed', 'submitted')
            obs = []
            if err:
                s.stats['format_errors'] += 1
                obs.append({'role': 'tool', 'content': f'Error: {err}'})
            t0 = time.time()
            for call in calls:
                t1 = time.time()
                out = s.envsess.call_tool(call['name'], call['arguments'])
                s.stats['tool_calls'] += 1
                s.stats['tool_errors'] += out.startswith('Error: Unknown tool')
                self._log({'sid': s.sid, 'event': 'tool', 'step': s.steps, 'tool': call['name'],
                           'args_sha256': sha256_hex(canonical(call['arguments'])),
                           'output_sha256': sha256_hex(out.encode()), 'output_chars': len(out),
                           'seconds': round(time.time() - t1, 3)})
                obs.append({'role': 'tool', 'content': out})
            s.stats['tool_seconds'] += time.time() - t0
            delta = r.tool_delta(obs, s.envsess.tools)
            if s.steps >= rt.max_steps:
                return self._finish(s, seq, 'unresolved', 'step_limit')
            if time.time() - s.started > rt.max_seconds:
                return self._finish(s, seq, 'unresolved', 'time_limit')
            if s.total + len(delta) + rt.min_room > rt.max_context:
                return self._finish(s, seq, 'unresolved', 'context_limit')
        except Exception as e:  # noqa: BLE001  state already changed, the client must not retry; always record an infra error
            log.exception('turn failed for %s', s.sid)
            return self._finish(s, seq, 'infra_error', f'{type(e).__name__}: {e}'[:500])
        s.hasher.add('T', delta)
        s.total += len(delta)
        s.stats['t_tokens'] += len(delta)
        s.last_activity = time.time()
        return {'status': 'ok', 'seq': seq, 'done': False, 'tool_tokens': delta}

    def _finish(self, s: RolloutSession, seq: int, outcome: str, detail: str) -> dict:
        """Grade -> sign receipt -> report to gateway -> return."""
        reward, grade = 0.0, {}
        if outcome != 'infra_error':
            try:
                gi = self.gw.grade_inputs(s.sid)
                grade = s.envsess.grade(gi)
                reward = float(grade['reward'])
            except Exception as e:  # noqa: BLE001  grading env broken / task voided
                outcome, detail = 'infra_error', f'grade: {type(e).__name__}: {e}'[:500]
        c = s.claims
        receipt = sign_receipt(self.key, self.key_id, epoch=c['epoch'], hotkey=c['hotkey'], env_id=c['env_id'],
                               task=c['task'], nonce=c['nonce'], trajectory_sha256=s.hasher.hexdigest(),
                               n_tokens=s.hasher.n_tokens, outcome=outcome, reward=reward)
        self._log({'sid': s.sid, 'event': 'finish', 'task': c['task'], 'nonce': c['nonce'], 'outcome': outcome,
                   'detail': detail, 'reward': reward, 'steps': s.steps, 'n_tokens': s.hasher.n_tokens,
                   'wall_seconds': round(time.time() - s.started, 1), **s.stats,
                   **{k: v for k, v in grade.items() if k in ('verifier_returncode', 'verifier_reason', 'test_seconds')}})
        self._teardown(s)
        try:
            self.gw.close(s.sid, outcome, reward)
        except GatewayUnavailable:
            log.error('could not report close for %s', s.sid)
        res = {'status': 'ok', 'seq': seq, 'done': True, 'outcome': outcome, 'reward': reward, 'detail': detail,
               'steps': s.steps, 'receipt': receipt}
        s.final = res
        s.last_activity = time.time()
        return res

    def _finish_infra(self, s: RolloutSession, detail: str) -> dict:
        """Error before the task prompt exists: no trajectory to sign, report infra_error directly."""
        self._teardown(s)
        try:
            self.gw.close(s.sid, 'infra_error', None)
        except GatewayUnavailable:
            pass
        self._log({'sid': s.sid, 'event': 'finish', 'outcome': 'infra_error', 'detail': detail})
        res = {'status': 'ok', 'seq': 0, 'done': True, 'outcome': 'infra_error', 'reward': 0.0, 'detail': detail,
               'receipt': None}
        s.final = res
        return res
