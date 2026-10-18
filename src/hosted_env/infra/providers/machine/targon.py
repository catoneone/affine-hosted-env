"""Targon VMs (REST API https://api.targon.com/tha/v3, docs.targon.com, checked 2026-10-15).

`create` registers a VM workload, deploys it and waits until it runs, then hands the boot script to a
`boot_runner`. Targon VMs have no user-data hook, so the default runner copies the script over SSH and starts it
detached with systemd-run. Facts that shape this code (internal experiment EXP-017):
- CPU tiers use SHARED networking: requested ports are remapped to other public ports (6443/tcp became :20004,
  51820/udp became :20000), there is no hairpin route to the VM's own public address, and no IPv6. The real
  mapping only appears in the workload state once it is running; `Machine.ports` carries it, and the boot script
  must use it (for example the WireGuard listen port). PASSTHROUGH tiers map ports 1:1.
- The VMs are AMD SEV-SNP guests; gVisor runs fine in them.
- Workload names are at most 32 characters (lowercase letters, digits, hyphens).
"""
from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable

import httpx

from .base import (NAME_PREFIX, BootScript, Machine, MachineError, MachineSpec, Offer, create_then_boot,
                   resolve_boot_script)

API = 'https://api.targon.com/tha/v3'
_STATE = {'registered': 'pending', 'provisioning': 'pending', 'running': 'running', 'error': 'error',
          'suspended': 'stopped', 'stopped': 'stopped', 'deleted': 'deleted'}

BootRunner = Callable[[Machine, str], None]


def ssh_boot_runner(key_path: str, known_hosts: str, user: str = 'ubuntu', timeout: float = 120) -> BootRunner:
    """Copy the script over SSH (stdin) and start it detached as root with systemd-run."""
    def run(m: Machine, script: str) -> None:
        host, port = m.ports['ssh'].rsplit(':', 1)
        remote = ('cat > /home/ubuntu/hosted-env-boot.sh && '
                  'sudo systemd-run --unit hosted-env-boot bash /home/ubuntu/hosted-env-boot.sh')
        r = subprocess.run(['ssh', '-i', key_path, '-p', port, '-o', 'StrictHostKeyChecking=accept-new',
                            '-o', f'UserKnownHostsFile={known_hosts}', '-o', 'ConnectTimeout=20',
                            f'{user}@{host}', remote], input=script.encode(), capture_output=True, timeout=timeout)
        if r.returncode != 0:
            raise MachineError(f'boot over ssh failed on {m.id}: {r.stderr.decode("utf-8", "replace")[-300:]}')
    return run


def _endpoint(url: str) -> str:
    return url.split('://', 1)[-1]


class TargonProvider:
    name = 'targon'

    def __init__(self, token: str, *, ssh_key_uid: str, boot_runner: BootRunner, org: str | None = None,
                 image: str = 'ubuntu-24-04-lts', http: httpx.Client | None = None, ready_timeout: float = 900,
                 poll: float = 5.0, gpu: bool = False):
        self._http = http or httpx.Client(base_url=API, timeout=60)
        self._auth = {'Authorization': f'Bearer {token}', 'Accept': 'application/json'}
        self.ssh_key_uid, self.boot_runner, self.image, self.gpu = ssh_key_uid, boot_runner, image, gpu
        self.ready_timeout, self.poll = ready_timeout, poll
        self._org = org
        self._prices: dict[str, float] = {}

    @classmethod
    def from_token_file(cls, path: str, **kw) -> TargonProvider:
        with open(os.path.expanduser(path)) as f:
            return cls(f.read().strip(), **kw)

    # ------------------------------------------------------------ http
    def _req(self, method: str, path: str, ok=(200, 201, 202, 204), auth: bool = True, **kw):
        try:
            r = self._http.request(method, path, headers=self._auth if auth else {'Accept': 'application/json'}, **kw)
        except httpx.HTTPError as e:
            raise MachineError(f'targon {method} {path}: {type(e).__name__}') from None
        if r.status_code not in ok:
            try:
                d = r.json()
            except ValueError:
                d = {}
            raise MachineError(f'targon {method} {path}: {r.status_code} {d.get("reason") or ""}'.strip())
        try:
            return r.json() if r.content else {}
        except ValueError:
            return {}

    @property
    def org(self) -> str:
        if self._org is None:
            items = self._req('GET', '/orgs').get('items') or []
            if not items:
                raise MachineError('targon: no organization for this token')
            self._org = items[0]['slug']
        return self._org

    def _machine(self, w: dict, state: dict | None = None) -> Machine:
        state = state or w.get('state') or {}
        ports = {}
        for u in state.get('urls') or []:
            ports[str(u.get('port'))] = _endpoint(u.get('url', ''))
        if state.get('public_ip') and state.get('ssh_port'):
            ports['ssh'] = f'{state["public_ip"]}:{state["ssh_port"]}'
        kind = w.get('resource_name') or ''
        return Machine(id=w['uid'], provider=self.name, name=w.get('name') or '', kind=kind,
                       state=_STATE.get(state.get('status', ''), 'pending'), public_ipv4=state.get('public_ip'),
                       price_per_hour=w.get('cost_per_hour') or self._prices.get(kind), ports=ports)

    # ------------------------------------------------------------ MachineProvider
    def offers(self) -> list[Offer]:
        """Public inventory (no auth). Only non-GPU tiers unless gpu=True."""
        d = self._req('GET', '/inventory', auth=False, params={'type': 'vm'})
        out = []
        for it in d if isinstance(d, list) else d.get('items', []):    # the live API returns a bare list
            if bool(it.get('gpu')) != self.gpu or it.get('type') != 'vm':
                continue
            s = it.get('spec') or {}
            self._prices[it['name']] = float(it.get('cost_per_hour') or 0)
            out.append(Offer(it['name'], s.get('cpu_millicores', 0) / 1000, s.get('memory_mib', 0) / 1024,
                             s.get('disk_size_mib', 0) / 1024, float(it.get('cost_per_hour') or 0),
                             it.get('available'), s.get('network_mode')))
        return out

    def list(self) -> list[Machine]:
        out, cursor = [], None
        while True:
            params = {'type': 'VM', **({'cursor': cursor} if cursor else {})}
            d = self._req('GET', f'/orgs/{self.org}/workloads', params=params)
            out += [self._machine(w) for w in d.get('items', []) if (w.get('name') or '').startswith(NAME_PREFIX)]
            cursor = d.get('next_cursor')
            if not cursor:
                return [m for m in out if m.state != 'deleted']

    def get(self, machine_id: str) -> Machine:
        w = self._req('GET', f'/orgs/{self.org}/workloads/{machine_id}')
        return self._machine(w, self._req('GET', f'/orgs/{self.org}/workloads/{machine_id}/state'))

    def create(self, spec: MachineSpec, boot_script: BootScript, on_created=None) -> Machine:
        if len(spec.name) > 32:
            raise MachineError(f'targon workload names are at most 32 characters: {spec.name!r}')
        ports = [{'port': int(p.split('/')[0]), 'protocol': (p.split('/')[1] if '/' in p else 'tcp').upper()}
                 for p in spec.ports]
        body = {'type': 'VM', 'name': spec.name, 'image': self.image, 'resource_name': spec.kind,
                'ssh_keys': [self.ssh_key_uid], 'vm_config': {'hostname': spec.name}}
        if ports:
            body['ports'] = ports
        uid = self._req('POST', f'/orgs/{self.org}/workloads', json=body)['uid']
        return create_then_boot(self, uid, lambda: self._deploy_and_boot(uid, boot_script), on_created)

    def _deploy_and_boot(self, uid: str, boot_script: BootScript) -> Machine:
        self._req('POST', f'/orgs/{self.org}/workloads/{uid}/deploy')
        deadline = time.monotonic() + self.ready_timeout
        while True:
            m = self.get(uid)
            if m.state == 'running' and 'ssh' in m.ports:
                break
            if m.state in ('error', 'deleted') or time.monotonic() > deadline:
                raise MachineError(f'targon workload {uid} did not start ({m.state})')
            time.sleep(self.poll)
        self.boot_runner(m, resolve_boot_script(boot_script, m))
        return m

    def delete(self, machine_id: str) -> None:
        self._req('DELETE', f'/orgs/{self.org}/workloads/{machine_id}', ok=(200, 202, 204, 404))
