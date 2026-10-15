"""Kubernetes sandboxes (architecture v2): one Pod per rollout.

- The Pod runs the task image with `sleep infinity`, `runtimeClassName: gvisor` by default, CPU and memory limits,
  no service-account token and no service links. `ensure_network_policies` installs a default-deny policy for
  Pods labelled `hosted-env/network=none` (egress and ingress), which is how `network="none"` is enforced.
- Commands run through the Kubernetes exec API (measured in EXP-017: p50 27-42 ms in region with a persistent
  client). Each command is wrapped the same way as in DockerSandbox: coreutils `timeout` as a backstop and the
  environment repaired by ENV_PROBE (Docker Hub task images lost their ENV, e.g. GOPATH).
- Files are written by exec with base64 chunks, so no extra tooling (tar, kubectl cp) is needed in the image.

The `kubernetes` client is an optional dependency (`pip install "affine-hosted-env[k8s]"`). Tests inject a fake
API object and an `exec_fn`.
"""
from __future__ import annotations

import base64
import time
import uuid
from collections.abc import Callable

from .base import OUTPUT_CAP_BYTES, SandboxError
from .docker import ENV_PROBE

MANAGED_BY = {'app.kubernetes.io/managed-by': 'hosted-env'}
NETWORK_LABEL = 'hosted-env/network'
CONTAINER = 'task'
WRITE_CHUNK = 48 * 1024          # bytes of file content per exec when writing files (base64 grows it by 4/3)

# exec_fn(pod, argv, timeout) -> (output bytes with stdout and stderr merged, returncode, timed_out)
ExecFn = Callable[[str, list[str], float], tuple[bytes, int, bool]]


def _mem(v: str) -> str:
    """Docker-style sizes ('2g', '512m') to Kubernetes quantities ('2Gi', '512Mi')."""
    v = str(v).strip()
    units = {'k': 'Ki', 'm': 'Mi', 'g': 'Gi', 't': 'Ti'}
    return v[:-1] + units[v[-1].lower()] if v and v[-1].lower() in units else v


def pod_manifest(name: str, image: str, cwd: str, *, namespace: str, node_name: str | None = None,
                 runtime_class: str | None = 'gvisor', cpus: str = '1', memory: str = '2g',
                 network: str = 'none', labels: dict | None = None) -> dict:
    spec = {
        'restartPolicy': 'Never',
        'automountServiceAccountToken': False,
        'enableServiceLinks': False,
        'terminationGracePeriodSeconds': 0,
        'containers': [{
            'name': CONTAINER, 'image': image, 'command': ['sleep', 'infinity'], 'workingDir': cwd,
            'resources': {'limits': {'cpu': str(cpus), 'memory': _mem(memory)},
                          'requests': {'cpu': str(cpus), 'memory': _mem(memory)}},
            'securityContext': {'allowPrivilegeEscalation': False},
        }],
    }
    if runtime_class:
        spec['runtimeClassName'] = runtime_class
    if node_name:
        spec['nodeName'] = node_name              # placed by the coordinator; the scheduler is bypassed
    if network == 'none':
        spec['dnsPolicy'] = 'None'                # nothing to resolve: egress is denied anyway
        spec['dnsConfig'] = {'nameservers': ['127.0.0.1']}
    return {'apiVersion': 'v1', 'kind': 'Pod',
            'metadata': {'name': name, 'namespace': namespace,
                         'labels': {**MANAGED_BY, NETWORK_LABEL: network, **(labels or {})}},
            'spec': spec}


def network_policy_manifest(namespace: str) -> dict:
    """Deny all egress and ingress for sandbox Pods with network=none."""
    return {'apiVersion': 'networking.k8s.io/v1', 'kind': 'NetworkPolicy',
            'metadata': {'name': 'hosted-env-network-none', 'namespace': namespace, 'labels': dict(MANAGED_BY)},
            'spec': {'podSelector': {'matchLabels': {NETWORK_LABEL: 'none'}}, 'policyTypes': ['Egress', 'Ingress']}}


def ensure_network_policies(networking_api, namespace: str) -> None:
    """Create the default-deny policy if it does not exist (idempotent)."""
    body = network_policy_manifest(namespace)
    try:
        networking_api.create_namespaced_network_policy(namespace, body)
    except Exception as e:                        # kubernetes.client.ApiException; 409 = already exists
        if getattr(e, 'status', None) != 409:
            raise


def _reason(pod) -> str | None:
    """Why a Pod is not starting, if it will not start on its own (bad image, crash)."""
    for cs in (getattr(pod.status, 'container_statuses', None) or []):
        waiting = getattr(cs.state, 'waiting', None)
        if waiting and waiting.reason in ('ErrImagePull', 'ImagePullBackOff', 'InvalidImageName',
                                          'CreateContainerConfigError', 'CreateContainerError', 'CrashLoopBackOff'):
            return f'{waiting.reason}: {(waiting.message or "")[:200]}'
        if getattr(cs.state, 'terminated', None):
            return f'container terminated: {cs.state.terminated.reason}'
    return None


def _ready(pod) -> bool:
    if pod.status.phase != 'Running':
        return False
    return all(cs.ready for cs in (pod.status.container_statuses or [])) and bool(pod.status.container_statuses)


def ws_exec(api, namespace: str) -> ExecFn:
    """exec_fn backed by the Kubernetes exec API (needs the `kubernetes` package)."""
    from kubernetes.client.rest import ApiException
    from kubernetes.stream import stream

    def run(pod: str, argv: list[str], timeout: float) -> tuple[bytes, int, bool]:
        try:
            resp = stream(api.connect_get_namespaced_pod_exec, pod, namespace, container=CONTAINER, command=argv,
                          stdin=False, stdout=True, stderr=True, tty=False, _preload_content=False)
        except ApiException as e:
            if e.status == 404:
                raise SandboxError('pod is gone') from None
            raise SandboxError(f'exec failed: {e.status} {e.reason}') from None
        buf, deadline, timed_out = bytearray(), time.monotonic() + timeout, False
        try:
            while resp.is_open():
                if time.monotonic() > deadline:
                    timed_out = True
                    break
                resp.update(timeout=1)
                for chunk in (resp.read_stdout() if resp.peek_stdout() else '',
                              resp.read_stderr() if resp.peek_stderr() else ''):
                    if chunk and len(buf) < OUTPUT_CAP_BYTES:
                        buf.extend(chunk.encode('utf-8', 'replace')[:OUTPUT_CAP_BYTES - len(buf)])
            rc = -1 if timed_out else (resp.returncode if resp.returncode is not None else -1)
        finally:
            resp.close()
        return bytes(buf), rc, timed_out
    return run


class K8sSandbox:
    def __init__(self, image: str, cwd: str, *, api=None, networking_api=None, exec_fn: ExecFn | None = None,
                 namespace: str = 'hosted-env', node_name: str | None = None, runtime_class: str | None = 'gvisor',
                 cpus: str = '1', memory: str = '2g', network: str = 'none', repair_env: bool = True,
                 start_timeout: float = 600, poll: float = 0.5, labels: dict | None = None, **_ignored):
        if api is None:
            from kubernetes import client, config
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            api, networking_api = client.CoreV1Api(), networking_api or client.NetworkingV1Api()
        self.image, self.cwd, self.api, self.ns = image, cwd, api, namespace
        self.name = f'hosted-env-{uuid.uuid4().hex[:12]}'
        self.manifest = pod_manifest(self.name, image, cwd, namespace=namespace, node_name=node_name,
                                     runtime_class=runtime_class, cpus=cpus, memory=memory, network=network,
                                     labels=labels)
        self.networking_api, self.network = networking_api, network
        self.exec_fn = exec_fn or ws_exec(api, namespace)
        self.start_timeout, self.poll, self.repair_env = start_timeout, poll, repair_env
        self.started, self.has_timeout = False, False
        self.env: dict[str, str] = {}

    def start(self):
        if self.network == 'none' and self.networking_api is not None:
            ensure_network_policies(self.networking_api, self.ns)
        try:
            self.api.create_namespaced_pod(self.ns, self.manifest)
        except Exception as e:
            raise SandboxError(f'create pod failed: {getattr(e, "status", "")} {str(e)[:300]}') from None
        self.started = True
        deadline = time.monotonic() + self.start_timeout
        while True:
            try:
                pod = self.api.read_namespaced_pod(self.name, self.ns)
            except Exception as e:
                raise SandboxError(f'read pod failed: {str(e)[:300]}') from None
            if _ready(pod):
                break
            why = _reason(pod) or (f'pod {pod.status.phase}' if pod.status.phase in ('Failed', 'Succeeded') else None)
            if why:
                raise SandboxError(why)
            if time.monotonic() > deadline:
                raise SandboxError(f'pod not ready after {self.start_timeout:.0f}s ({pod.status.phase})')
            time.sleep(self.poll)
        self.has_timeout = self.execute('command -v timeout', timeout=30, raw=True)['returncode'] == 0
        if self.repair_env:
            out = self.execute(ENV_PROBE, timeout=60, raw=True)['output']
            self.env = dict(ln.split('=', 1) for ln in out.splitlines()
                            if '=' in ln and ln.split('=', 1)[0].isupper())

    def _argv(self, command: str, cwd: str | None, timeout: int, raw: bool) -> list[str]:
        inner = ['bash', '-lc', command]
        if self.has_timeout and not raw:
            inner = ['timeout', '-k', '5', str(int(timeout)), *inner]
        env = ['env', *(f'{k}={v}' for k, v in self.env.items())] if self.env else []
        # exec has no working-directory option: change directory first, then exec the real command
        return ['sh', '-c', 'cd "$1" || exit 125; shift; exec "$@"', 'sh', cwd or self.cwd, *env, *inner]

    def execute(self, command: str, cwd: str | None = None, timeout: int = 60, raw: bool = False) -> dict:
        """Returns {output, returncode, reason}; reason is ok / client_timeout."""
        if not self.started:
            raise SandboxError('sandbox not started')
        out, rc, timed_out = self.exec_fn(self.name, self._argv(command, cwd, timeout, raw), timeout + 30)
        if self.has_timeout and not raw and rc == 124:
            timed_out = True
        return {'output': out[:OUTPUT_CAP_BYTES].decode('utf-8', 'replace'), 'returncode': rc,
                'reason': 'client_timeout' if timed_out else 'ok'}

    def copy_text_to(self, text: str, remote_path: str):
        data = text.encode('utf-8')
        for i in range(0, max(len(data), 1), WRITE_CHUNK):
            b64 = base64.b64encode(data[i:i + WRITE_CHUNK]).decode()
            op = '>' if i == 0 else '>>'
            argv = ['sh', '-c', f'printf %s "$1" | base64 -d {op} "$2"', 'sh', b64, remote_path]
            out, rc, _ = self.exec_fn(self.name, argv, 120)
            if rc != 0:
                raise SandboxError(f'write {remote_path} failed: {out[-300:].decode("utf-8", "replace")}')

    def cleanup(self):
        if not self.started:
            return
        self.started = False
        try:
            self.api.delete_namespaced_pod(self.name, self.ns, grace_period_seconds=0)
        except Exception as e:
            if getattr(e, 'status', None) != 404:
                raise
