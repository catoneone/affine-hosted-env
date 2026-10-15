"""boat.dev machines (REST API https://boat.dev/api/v1, docs.boat.dev, checked 2026-10-15).

A boat sandbox is a full root Ubuntu VM with Docker. `create` asks for one, waits until it is ready, uploads the
boot script and starts it detached as root; the boot script joins the k3s cluster. Facts that shape this code
(internal experiments EXP-015 / EXP-017):
- The API may ignore the requested name, so besides the `hosted-env-` name prefix this provider also treats the
  ids it created (and any `owned_ids` handed in) as its own.
- Machines may have only a public IPv6 address; `public_ipv4` is often empty.
- Snapshots do not keep firewall rules; the boot script re-applies them.
- API commands are limited to 600 s, so the boot script runs detached.
"""
from __future__ import annotations

import base64
import time
import uuid

import httpx

from .base import NAME_PREFIX, Machine, MachineError, MachineSpec, Offer

API = 'https://boat.dev/api/v1'
BOOT_PATH = '/home/user/hosted-env-boot.sh'
BOOT_LOG = '/home/user/hosted-env-boot.log'

# List prices (docs.boat.dev and our design notes, 2026-10-15). xlarge needs the $100 plan and is left out until
# its price is confirmed; pass `offers=` to override.
DEFAULT_OFFERS = (
    Offer('small', 2, 4, 12, 0.018),
    Offer('default', 4, 8, 50, 0.036),
    Offer('large', 8, 16, 125, 0.072),
)

_STATE = {'provisioned': 'pending', 'cloning': 'pending', 'starting': 'pending', 'resuming': 'pending',
          'ready': 'running', 'idle': 'running', 'running': 'running',
          'archived': 'stopped', 'stopping': 'stopped', 'stopped': 'stopped',
          'deleting': 'deleted', 'deleted': 'deleted', 'error': 'error'}


class BoatProvider:
    name = 'boat'

    def __init__(self, token: str, *, http: httpx.Client | None = None, offers=DEFAULT_OFFERS,
                 owned_ids=(), ready_timeout: float = 900, poll: float = 2.0):
        self._http = http or httpx.Client(base_url=API, timeout=60)
        self._auth = {'Authorization': f'Bearer {token}', 'Accept': 'application/json'}
        self._offers, self._owned = list(offers), set(owned_ids)
        self.ready_timeout, self.poll = ready_timeout, poll

    @classmethod
    def from_token_file(cls, path: str, **kw) -> BoatProvider:
        with open(path) as f:
            return cls(f.read().strip(), **kw)

    # ------------------------------------------------------------ http
    def _req(self, method: str, path: str, ok=(200, 201, 202), **kw) -> dict:
        headers = {**self._auth, **kw.pop('headers', {})}
        try:
            r = self._http.request(method, path, headers=headers, **kw)
        except httpx.HTTPError as e:
            raise MachineError(f'boat {method} {path}: {type(e).__name__}') from None
        if r.status_code not in ok:
            try:
                d = r.json()
            except ValueError:
                d = {}
            raise MachineError(f'boat {method} {path}: {r.status_code} {d.get("code") or d.get("type") or ""}'.strip())
        try:
            return r.json() if r.content else {}
        except ValueError:
            return {}

    def _machine(self, s: dict) -> Machine:
        offer = {o.kind: o for o in self._offers}.get(s.get('type', ''))
        ip = s.get('ip')
        return Machine(id=s['id'], provider=self.name, name=s.get('name') or '', kind=s.get('type', ''),
                       state=_STATE.get(s.get('state', ''), 'pending'),
                       public_ipv4=s.get('ipv4') or (ip if ip and ':' not in ip else None),
                       public_ipv6=ip if ip and ':' in ip else None,
                       price_per_hour=offer.price_per_hour if offer else None)

    def _ours(self, s: dict) -> bool:
        return s.get('id') in self._owned or (s.get('name') or '').startswith(NAME_PREFIX)

    # ------------------------------------------------------------ MachineProvider
    def offers(self) -> list[Offer]:
        return list(self._offers)

    def list(self) -> list[Machine]:
        out, cursor = [], None
        while True:
            d = self._req('GET', '/sandboxes', params={'cursor': cursor} if cursor else None)
            out += [self._machine(s) for s in d.get('sandboxes', []) if self._ours(s)]
            cursor = (d.get('pageInfo') or {}).get('nextCursor')
            if not cursor:
                return [m for m in out if m.state != 'deleted']

    def get(self, machine_id: str) -> Machine:
        return self._machine(self._req('GET', f'/sandboxes/{machine_id}')['sandbox'])

    def create(self, spec: MachineSpec, boot_script: str) -> Machine:
        body = {'type': spec.kind, 'name': spec.name}
        if spec.ttl_seconds:
            body['ttlSeconds'] = int(spec.ttl_seconds)
        d = self._req('POST', '/sandboxes', json=body, headers={'Idempotency-Key': uuid.uuid4().hex})
        mid = d['sandbox']['id']
        self._owned.add(mid)
        deadline = time.monotonic() + self.ready_timeout
        while True:
            m = self.get(mid)
            if m.state == 'running':
                break
            if m.state in ('error', 'deleted') or time.monotonic() > deadline:
                raise MachineError(f'boat sandbox {mid} did not become ready ({m.state})')
            time.sleep(self.poll)
        self._req('PUT', f'/sandboxes/{mid}/files',
                  json={'path': BOOT_PATH, 'content': base64.b64encode(boot_script.encode()).decode(),
                        'encoding': 'base64'})
        self._req('POST', f'/sandboxes/{mid}/commands',
                  json={'command': f'sudo bash {BOOT_PATH} > {BOOT_LOG} 2>&1', 'detached': True})
        return m

    def delete(self, machine_id: str) -> None:
        self._req('DELETE', f'/sandboxes/{machine_id}', ok=(200, 202, 204, 404),
                  headers={'X-Ascii-Confirm-Delete': machine_id})
        self._owned.discard(machine_id)
