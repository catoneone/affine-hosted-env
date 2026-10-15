"""worker HTTP API (design §8, data plane); the session token goes in `Authorization: Bearer`.

  POST /v1/start {wait?}               -> {status: pending} | {status: ok, seq: 0, prompt_tokens, stop_token_ids, ...}
  POST /v1/turn  {seq, tokens, wait?}  -> {status: pending} | {status: ok, seq, done: false, tool_tokens}
                                                           | {status: ok, seq, done: true, outcome, reward, receipt}
  GET  /v1/health

Pulling an image or running tests can take minutes, so each request waits at most `wait` seconds (<=25) and returns
pending if not done; the client resends the same seq and token to keep waiting on the same operation (it is not
executed twice).
"""
from __future__ import annotations

import asyncio
import contextlib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .core import Worker, WorkerError

MAX_WAIT = 25.0


def _bearer(request: Request) -> str:
    h = request.headers.get('authorization', '')
    if not h.lower().startswith('bearer '):
        raise WorkerError('missing_token', 401)
    return h[7:].strip()


async def _await(fut, wait: float) -> dict:
    try:
        return await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(fut)), wait)
    except asyncio.TimeoutError:
        return {'status': 'pending'}


def create_worker_app(worker: Worker, background: bool = True) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app):
        if background:
            worker.start_background()
        yield
        worker.stop()

    app = FastAPI(title='hosted-env worker', lifespan=lifespan)

    @app.exception_handler(WorkerError)
    async def _worker_error(request, e: WorkerError):
        return JSONResponse({'error': e.code, 'detail': e.detail}, status_code=e.status)

    async def body(request: Request) -> dict:
        try:
            d = await request.json()
        except ValueError:
            raise WorkerError('bad_json', 400) from None
        if not isinstance(d, dict):
            raise WorkerError('bad_json', 400)
        return d

    def wait_of(d: dict) -> float:
        try:
            return max(0.0, min(float(d.get('wait', 20)), MAX_WAIT))
        except (TypeError, ValueError):
            raise WorkerError('bad_wait', 400) from None

    @app.post('/v1/start')
    async def start(request: Request):
        token, d = _bearer(request), await body(request)
        fut = await run_in_threadpool(worker.start, token)
        return await _await(fut, wait_of(d))

    @app.post('/v1/turn')
    async def turn(request: Request):
        token, d = _bearer(request), await body(request)
        fut = await run_in_threadpool(worker.turn, token, d.get('seq'), d.get('tokens'))
        return await _await(fut, wait_of(d))

    @app.get('/v1/health')
    async def health():
        live = sum(1 for s in list(worker.sessions.values()) if s.final is None)
        return {'ok': True, 'worker_id': worker.worker_id, 'key_id': worker.key_id, 'live_sessions': live,
                'capacity': worker.capacity}

    return app
