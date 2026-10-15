"""gateway HTTP API (design §8, control plane).

Miner (hotkey-signed, domain MINER_DOMAIN):
  POST /v1/tasks/open      {epoch, env_id}            -> {task_id, env_id, task, worker_url, group_size}
  POST /v1/sessions/open   {epoch, task_id, nonces}   -> {sessions: [{nonce, session_id, token}]}
  POST /v1/tasks/finish    {epoch, task_id}           -> {ok}
worker (worker-key-signed, domain WORKER_DOMAIN):
  POST /v1/internal/heartbeat     {url, capacity, images, env_ids}
  POST /v1/internal/task-spec     {task_id}           -> {env_id, task, group_size, spec}
  POST /v1/internal/grade-inputs  {session_id}        -> grading inputs
  POST /v1/internal/touch         {session_id}        -> {active}
  POST /v1/internal/close         {session_id, outcome, reward}
Public:
  GET  /v1/info   current epoch, gateway public key, worker public keys, env parameters (in production these come from the opening)
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from nacl.signing import VerifyKey
from starlette.concurrency import run_in_threadpool

from ..auth import MINER_DOMAIN, WORKER_DOMAIN, AuthError, RequestVerifier
from ..common import key_id, verify_key_from_hex
from ..coordinator.store import CapacityError, QuotaError, Store
from .ratelimit import RateLimiter, RateLimitError


class Roster:
    """Roster of eligible identities for this epoch. File format {"epoch": "...", "hotkeys": {hotkey: {"uid": int, "public_key": hex}}},
    re-read automatically when the file changes (in production: the opening's identity snapshot / upstream `registrations()`)."""

    def __init__(self, path: str):
        self.path, self._mtime, self.epoch, self.keys = path, None, None, {}

    def _load(self):
        m = os.stat(self.path).st_mtime_ns
        if m != self._mtime:
            data = json.loads(open(self.path).read())
            self.keys = {hk: verify_key_from_hex(v['public_key']) for hk, v in data['hotkeys'].items()}
            self.epoch, self._mtime = data['epoch'], m

    def current_epoch(self) -> str:
        self._load()
        return self.epoch

    def key(self, hotkey: str) -> VerifyKey | None:
        self._load()
        return self.keys.get(hotkey)


def _err(status: int, code: str, detail: str = '', headers: dict | None = None) -> JSONResponse:
    return JSONResponse({'error': code, 'detail': detail}, status_code=status, headers=headers)


def create_gateway_app(store: Store, roster: Roster, worker_keys: dict[str, VerifyKey], gateway_pub: VerifyKey,
                       reap_interval: float = 10.0, limiter: RateLimiter | None = None) -> FastAPI:
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
        yield
        task.cancel()

    app = FastAPI(title='hosted-env gateway', lifespan=lifespan)

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
        await run_in_threadpool(store.finish_task, str(data.get('epoch')), hotkey, str(data.get('task_id')))
        return {'ok': True}

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
        reward = d.get('reward')
        await run_in_threadpool(store.worker_close, wid, str(d.get('session_id')), str(d.get('outcome')),
                                None if reward is None else float(reward))
        return {'ok': True}

    # ---------------------------------------------------------------- public
    @app.get('/v1/info')
    async def info():
        return {'epoch': roster.current_epoch(), 'gateway_public_key': bytes(gateway_pub).hex(),
                'workers': {w: {'public_key': bytes(k).hex(), 'key_id': key_id(k)} for w, k in worker_keys.items()},
                'envs': {e: {'kind': env.kind, 'group_size': env.group_size, 'success_reward': env.success_reward}
                         for e, env in store.envs.items()}}

    return app
