"""Client for the worker -> gateway internal API (signed with the worker key)."""
from __future__ import annotations

import json
import time

import httpx
from nacl.signing import SigningKey

from ..auth import WORKER_DOMAIN, sign_request


class GatewayUnavailable(RuntimeError):
    pass


class GatewayRejected(RuntimeError):
    def __init__(self, status: int, code: str):
        super().__init__(f'{status} {code}')
        self.status, self.code = status, code


class WorkerGatewayClient:
    def __init__(self, base_url: str, worker_id: str, key: SigningKey, *, timeout: float = 30.0,
                 retries: int = 3, http: httpx.Client | None = None):
        self.worker_id, self.key, self.retries = worker_id, key, retries
        self.http = http or httpx.Client(base_url=base_url.rstrip('/'), timeout=timeout)

    def post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload, separators=(',', ':')).encode()
        last = None
        for attempt in range(self.retries):
            headers = sign_request(self.key, 'POST', path, self.worker_id, body, domain=WORKER_DOMAIN)
            try:
                r = self.http.post(path, content=body, headers={**headers, 'Content-Type': 'application/json'})
            except httpx.TransportError as e:
                last = e
                time.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code >= 500:
                last = RuntimeError(f'gateway {r.status_code}')
                time.sleep(0.5 * (attempt + 1))
                continue
            data = r.json()
            if r.status_code != 200:
                raise GatewayRejected(r.status_code, data.get('error', ''))
            return data
        raise GatewayUnavailable(str(last))

    def heartbeat(self, url: str, capacity: int, images: list[str], env_ids: list[str]) -> dict:
        return self.post('/v1/internal/heartbeat', {'url': url, 'capacity': capacity, 'images': images,
                                                    'env_ids': env_ids})

    def task_spec(self, task_id: str) -> dict:
        return self.post('/v1/internal/task-spec', {'task_id': task_id})

    def grade_inputs(self, session_id: str) -> dict:
        return self.post('/v1/internal/grade-inputs', {'session_id': session_id})

    def touch(self, session_id: str) -> bool:
        try:
            return bool(self.post('/v1/internal/touch', {'session_id': session_id})['active'])
        except GatewayRejected:
            return False

    def close(self, session_id: str, outcome: str, reward: float | None) -> None:
        try:
            self.post('/v1/internal/close', {'session_id': session_id, 'outcome': outcome, 'reward': reward})
        except GatewayRejected:
            pass
