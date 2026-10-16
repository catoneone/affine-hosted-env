"""gateway HTTP API (design §8, control plane).

Miner (hotkey-signed, domain MINER_DOMAIN):
  POST /v1/tasks/open      {epoch, env_id}            -> {task_id, env_id, task, worker_url, group_size}
  POST /v1/sessions/open   {epoch, task_id, nonces}   -> {sessions: [{nonce, session_id, token}]}
  POST /v1/tasks/finish    {epoch, task_id}           -> {ok, seal}  (gateway-signed group seal; null if voided)
worker (worker-key-signed, domain WORKER_DOMAIN):
  POST /v1/internal/heartbeat     {url, capacity, images, env_ids}
  POST /v1/internal/task-spec     {task_id}           -> {env_id, task, group_size, spec}
  POST /v1/internal/grade-inputs  {session_id}        -> grading inputs
  POST /v1/internal/touch         {session_id}        -> {active}
  POST /v1/internal/close         {session_id, outcome, reward, receipt_sha256}
Public:
  GET  /v1/info   current epoch, gateway public key, worker public keys, env parameters, roster size and chain block
Data plane (only when an in-process executor is attached; session token in `Authorization: Bearer`):
  POST /v1/start, POST /v1/turn, GET /v1/health   -- see executor/app.py
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from nacl.signing import VerifyKey
from starlette.concurrency import run_in_threadpool

from ..auth import MINER_DOMAIN, WORKER_DOMAIN, AuthError, RequestVerifier, verify_token
from ..common import key_id, verify_key_from_hex
from ..coordinator.store import CapacityError, QuotaError, Store
from ..executor.app import add_executor_routes
from ..executor.core import Worker
from .ratelimit import RateLimiter, RateLimitError

log = logging.getLogger('hosted_env.gateway')


class Roster:
    """Eligible identities, from up to two files, each re-read automatically when it changes (hot reload):
    - `path`, hand-edited: {"epoch": "...", "hotkeys": {hotkey: {"uid": int, "public_key": hex}}}. It sets the epoch,
      and its hotkeys are an allowlist (test keys, partners) on top of the chain;
    - `chain_path` (optional), written by `hosted-env roster-sync` from SN120 registrations.
    A hotkey in both keeps the chain's key: the allowlist cannot re-key a registered hotkey. A file that turns
    unreadable or malformed keeps its last good content (logged), so a bad edit never locks miners out; a missing
    chain file counts as no chain hotkeys."""

    def __init__(self, path: str, chain_path: str | None = None):
        self.path, self.chain_path = path, chain_path
        self._seen = {}                                   # path -> (mtime_ns, size) last loaded
        self.epoch, self._allow, self._chain, self._chain_meta = None, {}, {}, {}
        self.keys = {}
        self._load(strict=True)

    def _read(self, path: str, strict: bool):
        """The parsed file if it changed since the last load, else None."""
        try:
            st = os.stat(path)
        except FileNotFoundError:
            if strict and path == self.path:
                raise
            if self._seen.get(path) != 'missing':
                self._seen[path] = 'missing'
                return {'hotkeys': {}}
            return None
        sig = (st.st_mtime_ns, st.st_size)
        if self._seen.get(path) == sig:
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            keys = {hk: verify_key_from_hex(v['public_key']) for hk, v in (data.get('hotkeys') or {}).items()}
        except (OSError, ValueError, KeyError, TypeError) as e:
            if strict:
                raise
            log.error('roster %s is unreadable, keeping the last good copy: %s: %s', path, type(e).__name__, e)
            self._seen[path] = sig                         # do not re-log until it changes again
            return None
        self._seen[path] = sig
        return {**data, 'keys': keys}

    def _load(self, strict: bool = False):
        changed = False
        d = self._read(self.path, strict)
        if d is not None:
            if 'epoch' in d:
                self.epoch = d['epoch']
            self._allow, changed = d.get('keys', {}), True
        if self.chain_path:
            d = self._read(self.chain_path, False)
            if d is not None:
                self._chain = d.get('keys', {})
                self._chain_meta = {k: d.get(k) for k in ('block', 'synced_at', 'netuid')}
                changed = True
        if changed:
            for hk in set(self._allow) & set(self._chain):
                if bytes(self._allow[hk]) != bytes(self._chain[hk]):
                    log.warning('roster: allowlist key for %s differs from the chain; the chain key is used', hk)
            self.keys = {**self._allow, **self._chain}

    def current_epoch(self) -> str:
        self._load()
        return self.epoch

    def key(self, hotkey: str) -> VerifyKey | None:
        self._load()
        return self.keys.get(hotkey)

    def status(self) -> dict:
        self._load()
        out = {'allowlist_hotkeys': len(self._allow)}
        if self.chain_path:
            out.update(chain_hotkeys=len(self._chain), chain_block=self._chain_meta.get('block'),
                       chain_synced_at=self._chain_meta.get('synced_at'))
        return out


def _err(status: int, code: str, detail: str = '', headers: dict | None = None) -> JSONResponse:
    return JSONResponse({'error': code, 'detail': detail}, status_code=status, headers=headers)


def create_gateway_app(store: Store, roster: Roster, worker_keys: dict[str, VerifyKey], gateway_pub: VerifyKey,
                       reap_interval: float = 10.0, limiter: RateLimiter | None = None,
                       executor: Worker | None = None, node_source: Callable[[], list] | None = None,
                       node_sync_interval: float = 10.0, planner=None, planner_interval: float = 30.0) -> FastAPI:
    """With `executor` set, the gateway also serves the data plane (`/v1/start`, `/v1/turn`) and runs that
    executor in this process (architecture v2); executors are then never exposed directly. With `node_source`
    set (placement='nodes'), the cluster's nodes are read every `node_sync_interval` seconds into the store. With
    `planner` set, the capacity planner runs every `planner_interval` seconds; its plan is logged whenever it
    changes, and the latest round is kept in `app.state.planner_last`."""
    miner_auth = RequestVerifier(MINER_DOMAIN, roster.key, store.remember_reqid)
    worker_auth = RequestVerifier(WORKER_DOMAIN, worker_keys.get, store.remember_reqid)
    limiter = limiter or RateLimiter()

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async def reaper():
            while True:
                await asyncio.sleep(reap_interval)
                with contextlib.suppress(Exception):
                    await run_in_threadpool(store.reap)
        task = asyncio.create_task(reaper())
        tasks = [task]
        if node_source is not None:
            async def node_sync():
                while True:
                    try:
                        nodes = await run_in_threadpool(node_source)
                        await run_in_threadpool(store.sync_nodes, nodes)
                    except Exception:
                        log.exception('node sync failed')
                    await asyncio.sleep(node_sync_interval)
            tasks.append(asyncio.create_task(node_sync()))
        if planner is not None:
            async def plan_loop():
                last_sig = None
                while True:
                    try:
                        snap = await run_in_threadpool(store.capacity_snapshot)
                        results = await run_in_threadpool(planner.step, snap)
                        app.state.planner_last = {'at': time.time(), 'dry_run': planner.dry_run, 'snapshot': snap,
                                                  'results': [(a.kind, a.provider, a.machine_kind, a.machine_id,
                                                               a.reason, r) for a, r in results]}
                        sig = [(a.kind, a.provider, a.machine_kind, a.machine_id, r) for a, r in results]
                        if sig != last_sig or any(r != 'dry_run' for _, r in results):
                            log.info('capacity plan (%s): need waiting=%s used=%s nodes=%d; %s',
                                     'dry run' if planner.dry_run else 'live', snap['waiting'], snap['used'],
                                     len(snap['nodes']), '; '.join(f'{a.kind} {a.provider} {a.machine_kind or ""}'
                                                                   f'{a.machine_id} ({a.reason}) -> {r}'
                                                                   for a, r in results) or 'nothing to do')
                        last_sig = sig
                    except Exception:
                        log.exception('capacity planner round failed')
                    await asyncio.sleep(planner_interval)
            tasks.append(asyncio.create_task(plan_loop()))
        if executor is not None:
            executor.start_background()
        yield
        for t in tasks:
            t.cancel()
        if executor is not None:
            executor.stop()

    app = FastAPI(title='hosted-env gateway', lifespan=lifespan)

    if executor is not None:
        def limit_data_plane(token: str) -> None:
            try:
                hotkey = verify_token(gateway_pub, token)['hotkey']
            except AuthError:
                return                      # the executor rejects the token itself
            limiter.data(hotkey)
        add_executor_routes(app, executor, before=limit_data_plane)

    def _limited(e: RateLimitError) -> JSONResponse:
        return _err(429, e.code, e.detail, {'Retry-After': str(e.retry_after)})

    @app.middleware('http')
    async def _rate_limit_public(request: Request, call_next):
        # Internal worker routes are authenticated by worker keys and not exposed to miners.
        if not request.url.path.startswith('/v1/internal/'):
            try:
                limiter.before_auth(request.client.host if request.client else '')
            except RateLimitError as e:
                return _limited(e)
        return await call_next(request)

    async def authed(request: Request, verifier: RequestVerifier) -> tuple[str, dict]:
        body = await request.body()
        ident = await run_in_threadpool(verifier.verify, request.method, request.url.path, request.headers, body)
        if verifier is miner_auth:
            limiter.control(ident)
        try:
            data = json.loads(body or b'{}')
        except ValueError:
            raise QuotaError('bad_json', status=400) from None
        if not isinstance(data, dict):
            raise QuotaError('bad_json', status=400)
        return ident, data

    def check_epoch(data: dict) -> str:
        epoch = data.get('epoch')
        if epoch != roster.current_epoch():
            raise QuotaError('wrong_epoch', f'current epoch is {roster.current_epoch()}', status=409)
        return epoch

    @app.exception_handler(RateLimitError)
    async def _rate_limit_error(request, e: RateLimitError):
        return _limited(e)

    @app.exception_handler(AuthError)
    async def _auth_error(request, e: AuthError):
        return _err(403 if e.code == 'not_eligible' else 401, e.code)

    @app.exception_handler(QuotaError)
    async def _quota_error(request, e: QuotaError):
        headers = {'Retry-After': str(e.retry_after)} if isinstance(e, CapacityError) else None
        return _err(e.status, e.code, e.detail, headers)

    # ---------------------------------------------------------------- miner
    @app.post('/v1/tasks/open')
    async def tasks_open(request: Request):
        hotkey, data = await authed(request, miner_auth)
        epoch = check_epoch(data)
        return await run_in_threadpool(store.open_task, epoch, hotkey, str(data.get('env_id')))

    @app.post('/v1/sessions/open')
    async def sessions_open(request: Request):
        hotkey, data = await authed(request, miner_auth)
        epoch = check_epoch(data)
        out = await run_in_threadpool(store.open_sessions, epoch, hotkey, str(data.get('task_id')),
                                      data.get('nonces'))
        return {'sessions': out}

    @app.post('/v1/tasks/finish')
    async def tasks_finish(request: Request):
        hotkey, data = await authed(request, miner_auth)
        seal = await run_in_threadpool(store.finish_task, str(data.get('epoch')), hotkey, str(data.get('task_id')))
        return {'ok': True, 'seal': seal}

    # ---------------------------------------------------------------- worker
    @app.post('/v1/internal/heartbeat')
    async def heartbeat(request: Request):
        wid, d = await authed(request, worker_auth)
        await run_in_threadpool(store.heartbeat, wid, str(d['url']), int(d['capacity']), list(d.get('images', [])),
                                list(d.get('env_ids', [])))
        return {'ok': True, 'server_time': time.time()}

    @app.post('/v1/internal/task-spec')
    async def task_spec(request: Request):
        wid, d = await authed(request, worker_auth)
        return await run_in_threadpool(store.worker_task_spec, wid, str(d.get('task_id')))

    @app.post('/v1/internal/grade-inputs')
    async def grade_inputs(request: Request):
        wid, d = await authed(request, worker_auth)
        return await run_in_threadpool(store.worker_grade_inputs, wid, str(d.get('session_id')))

    @app.post('/v1/internal/touch')
    async def touch(request: Request):
        wid, d = await authed(request, worker_auth)
        return {'active': await run_in_threadpool(store.worker_touch, wid, str(d.get('session_id')))}

    @app.post('/v1/internal/close')
    async def close(request: Request):
        wid, d = await authed(request, worker_auth)
        reward, digest = d.get('reward'), d.get('receipt_sha256')
        await run_in_threadpool(store.worker_close, wid, str(d.get('session_id')), str(d.get('outcome')),
                                None if reward is None else float(reward), None,
                                None if digest is None else str(digest))
        return {'ok': True}

    # ---------------------------------------------------------------- public
    @app.get('/v1/info')
    async def info():
        return {'epoch': roster.current_epoch(), 'gateway_public_key': bytes(gateway_pub).hex(),
                'workers': {w: {'public_key': bytes(k).hex(), 'key_id': key_id(k)} for w, k in worker_keys.items()},
                'envs': {e: {'kind': env.kind, 'group_size': env.group_size, 'success_reward': env.success_reward}
                         for e, env in store.envs.items()},
                'roster': roster.status()}

    return app
