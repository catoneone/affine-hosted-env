"""Reference implementation of the miner client (design §8, §10 "miner client").

The miner keeps sampling on its own GPU: starting from the prompt tokens the worker provides, it samples up to the
first stop token, hands those tokens to the worker, gets tool output tokens back and appends them to the context as
is before sampling again. It no longer parses tool calls or assembles tool output itself. When a group finishes, each
rollout's source segments and receipt go into the group object (`attachment`), uploaded via the existing
complete-group flow.

The sampler is pluggable: `sampler(context_ids, limits) -> new_ids`; when new_ids stops at a stop token it must include
it. limits contains max_tokens, stop_token_ids, nonce, step. The real upstream sampler (with sampling proof) plugs in
here.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import httpx
from nacl.signing import SigningKey

from .auth import MINER_DOMAIN, sign_request
from .verify import GROUP_ATTACHMENT_VERSION

Sampler = Callable[[list[int], dict], list[int]]


class ApiError(RuntimeError):
    def __init__(self, status: int, code: str, detail: str = '', retry_after: float | None = None):
        super().__init__(f'{status} {code} {detail}'.strip())
        self.status, self.code, self.detail, self.retry_after = status, code, detail, retry_after


def _raise(r: httpx.Response):
    try:
        d = r.json()
    except ValueError:
        d = {}
    ra = r.headers.get('retry-after')
    raise ApiError(r.status_code, d.get('error', ''), d.get('detail', ''), float(ra) if ra else None)


def _rate_limited(r: httpx.Response) -> bool:
    if r.status_code != 429:
        return False
    try:
        return r.json().get('error') == 'rate_limited'
    except ValueError:
        return False


def _retry_after(r: httpx.Response, cap: float = 30.0) -> float:
    try:
        return min(max(float(r.headers.get('retry-after', 1)), 0.0), cap)
    except ValueError:
        return 1.0


class GatewayClient:
    def __init__(self, base_url: str, hotkey: str, key: SigningKey, *, timeout: float = 30.0,
                 http: httpx.Client | None = None):
        self.hotkey, self.key = hotkey, key
        self.http = http or httpx.Client(base_url=base_url.rstrip('/'), timeout=timeout)

    def _post(self, path: str, payload: dict, rate_retries: int = 3) -> dict:
        """Signs and sends one request. On 429 rate_limited, waits per Retry-After and resends with a fresh
        signature (request ids are single use)."""
        body = json.dumps(payload, separators=(',', ':')).encode()
        for attempt in range(rate_retries + 1):
            h = sign_request(self.key, 'POST', path, self.hotkey, body, domain=MINER_DOMAIN)
            r = self.http.post(path, content=body, headers={**h, 'Content-Type': 'application/json'})
            if r.status_code == 200:
                return r.json()
            if not _rate_limited(r) or attempt == rate_retries:
                _raise(r)
            time.sleep(_retry_after(r))
        raise AssertionError('unreachable')

    def info(self) -> dict:
        r = self.http.get('/v1/info')
        if r.status_code != 200:
            _raise(r)
        return r.json()

    def open_task(self, epoch: str, env_id: str, *, max_wait: float = 0.0) -> dict:
        """When capacity is insufficient (429), wait per Retry-After, for at most max_wait seconds."""
        deadline = time.time() + max_wait
        while True:
            try:
                return self._post('/v1/tasks/open', {'epoch': epoch, 'env_id': env_id})
            except ApiError as e:
                if e.status != 429 or time.time() >= deadline:
                    raise
                time.sleep(min(e.retry_after or 10, max(0.0, deadline - time.time())))

    def open_sessions(self, epoch: str, task_id: str, nonces: list[int]) -> list[dict]:
        return self._post('/v1/sessions/open', {'epoch': epoch, 'task_id': task_id, 'nonces': nonces})['sessions']

    def finish_task(self, epoch: str, task_id: str) -> None:
        self._post('/v1/tasks/finish', {'epoch': epoch, 'task_id': task_id})


class WorkerClient:
    def __init__(self, base_url: str, *, timeout: float = 60.0, http: httpx.Client | None = None,
                 poll_wait: float = 20.0, max_retries: int = 5):
        self.http = http or httpx.Client(base_url=base_url.rstrip('/'), timeout=timeout)
        self.poll_wait, self.max_retries = poll_wait, max_retries

    def _call(self, path: str, token: str, payload: dict) -> dict:
        """Long poll: resend as is on pending; retry network errors with the same seq (the worker is idempotent)."""
        errors = 0
        while True:
            try:
                r = self.http.post(path, json={**payload, 'wait': self.poll_wait},
                                   headers={'Authorization': f'Bearer {token}'})
            except httpx.TransportError:
                errors += 1
                if errors > self.max_retries:
                    raise
                time.sleep(min(2 ** errors, 30))
                continue
            if r.status_code >= 500 and r.status_code != 503 and errors < self.max_retries:
                errors += 1
                time.sleep(min(2 ** errors, 30))
                continue
            if _rate_limited(r):
                time.sleep(_retry_after(r))        # same seq and tokens: the executor does not run it twice
                continue
            if r.status_code != 200:
                _raise(r)
            d = r.json()
            if d.get('status') != 'pending':
                return d

    def start(self, token: str) -> dict:
        return self._call('/v1/start', token, {})

    def turn(self, token: str, seq: int, tokens: list[int]) -> dict:
        return self._call('/v1/turn', token, {'seq': seq, 'tokens': tokens})


def run_rollout(worker: WorkerClient, token: str, sampler: Sampler, *, nonce: int = 0) -> dict:
    """Run one rollout to completion; returns {nonce, segments, receipt, outcome, reward, detail, steps}.
    outcome 'voided' means the gateway voided the whole task while this rollout ran (a sibling hit an infra
    error): there is no receipt and nothing to upload."""
    segs: list[dict] = []
    try:
        r = worker.start(token)
        if r.get('done'):                              # infra error already during setup
            return {'nonce': nonce, 'segments': [], 'receipt': r.get('receipt'), 'outcome': r['outcome'],
                    'reward': r.get('reward', 0.0), 'detail': r.get('detail'), 'steps': 0}
        ctx = list(r['prompt_tokens'])
        segs = [{'src': 'P', 'ids': list(r['prompt_tokens'])}]
        stops, max_ctx, max_turn = r['stop_token_ids'], r['max_context'], r['max_turn_tokens']
        seq = 1
        while True:
            limits = {'max_tokens': min(max_turn, max_ctx - len(ctx)), 'stop_token_ids': stops, 'nonce': nonce,
                      'step': seq}
            out = list(sampler(ctx, limits))
            r = worker.turn(token, seq, out)
            segs.append({'src': 'M', 'ids': out})
            ctx += out
            if r['done']:
                return {'nonce': nonce, 'segments': segs, 'receipt': r['receipt'], 'outcome': r['outcome'],
                        'reward': r['reward'], 'detail': r.get('detail'), 'steps': r.get('steps', seq)}
            segs.append({'src': 'T', 'ids': list(r['tool_tokens'])})
            ctx += r['tool_tokens']
            seq += 1
    except ApiError as e:
        if e.status != 410:
            raise
        return {'nonce': nonce, 'segments': segs, 'receipt': None, 'outcome': 'voided', 'reward': 0.0,
                'detail': f'{e.code} {e.detail}'.strip(), 'steps': 0}


def run_task(gateway: GatewayClient, epoch: str, env_id: str, sampler: Sampler, *, concurrency: int = 8,
             worker_factory: Callable[[str], WorkerClient] = WorkerClient, max_wait: float = 0.0) -> dict:
    """Open a task, run the whole group of nonces concurrently, finish it; returns {task, rollouts, attachment,
    voided}. attachment goes into the group object; it is None when the gateway voided the task (an infra error in
    any rollout voids the group and refunds it, so there is nothing to upload)."""
    t = gateway.open_task(epoch, env_id, max_wait=max_wait)
    results: list[dict] = []
    try:
        sessions = gateway.open_sessions(epoch, t['task_id'], list(range(t['group_size'])))
        worker = worker_factory(t['worker_url'])
        with ThreadPoolExecutor(concurrency) as ex:
            results = list(ex.map(lambda s: run_rollout(worker, s['token'], sampler, nonce=s['nonce']), sessions))
    finally:
        try:
            gateway.finish_task(epoch, t['task_id'])
        except ApiError as e:
            if e.status not in (404, 409, 410):        # an already ended (voided) task cannot be finished again
                raise
    voided = any(x['outcome'] in ('voided', 'infra_error') for x in results)
    attachment = None if voided else {
        'version': GROUP_ATTACHMENT_VERSION, 'env_id': t['env_id'], 'task': t['task'],
        'rollouts': [{'nonce': x['nonce'], 'segments': x['segments'], 'receipt': x['receipt']}
                     for x in sorted(results, key=lambda x: x['nonce'])]}
    return {'task': t, 'rollouts': results, 'attachment': attachment, 'voided': voided}


# ---------------------------------------------------------------- samplers

class ScriptedSampler:
    """Fake model for tests / smoke tests: emits prewritten reply text per (nonce, step), encodes it and appends the
    turn-end token. replies can be a list (same for all nonces) or {nonce: list}. encode / turn_end_id use the miner's
    local tokenizer."""

    def __init__(self, encode: Callable[[str], list[int]], turn_end_id: int, replies, default: str = 'done'):
        self.encode, self.turn_end_id, self.replies, self.default = encode, turn_end_id, replies, default

    def __call__(self, ctx: list[int], limits: dict) -> list[int]:
        rs = self.replies.get(limits['nonce'], []) if isinstance(self.replies, dict) else self.replies
        i = limits['step'] - 1
        text = rs[i] if i < len(rs) else self.default
        return (self.encode(text) + [self.turn_end_id])[:limits['max_tokens']]


class VLLMSampler:
    """Uses vLLM's OpenAI-compatible /v1/completions, passing the prompt as token ids and requesting token ids back.

    Relies on vLLM's `return_token_ids` response field (only in newer versions; to be confirmed in P1). Whether the
    stop token is included in the returned token_ids varies by version; if it is missing, it is appended based on
    stop_reason."""

    def __init__(self, base_url: str, model: str, *, temperature: float = 1.0, top_p: float = 1.0,
                 timeout: float = 1800.0, seed: int = 0):
        self.http = httpx.Client(base_url=base_url.rstrip('/'), timeout=timeout)
        self.model, self.temperature, self.top_p, self.seed = model, temperature, top_p, seed

    def __call__(self, ctx: list[int], limits: dict) -> list[int]:
        body = {'model': self.model, 'prompt': ctx, 'max_tokens': limits['max_tokens'],
                'temperature': self.temperature, 'top_p': self.top_p,
                'seed': self.seed * 1_000_003 + limits['nonce'] * 1009 + limits['step'],
                'stop_token_ids': limits['stop_token_ids'], 'skip_special_tokens': False,
                'include_stop_str_in_output': True, 'return_token_ids': True}
        r = self.http.post('/v1/completions', json=body)
        r.raise_for_status()
        ch = r.json()['choices'][0]
        ids = list(ch['token_ids'])
        stops = set(limits['stop_token_ids'])
        if ch.get('finish_reason') == 'stop' and (not ids or ids[-1] not in stops):
            sr = ch.get('stop_reason')
            ids.append(sr if isinstance(sr, int) else limits['stop_token_ids'][0])
        return ids
