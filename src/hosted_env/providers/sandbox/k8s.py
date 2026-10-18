"""Kubernetes sandboxes (architecture v2): one Pod per rollout.

- The Pod runs the task image with `sleep infinity`, CPU, memory and ephemeral-storage limits, no service-account
  token and no service links. For `network="none"` it uses the `gvisor-nonet` RuntimeClass (runsc `--network=none`: loopback
  only), checks after start that loopback is the only interface (fail closed), and is also covered by a
  default-deny NetworkPolicy for Pods labelled `hosted-env/network=none`. The policy alone is not enough: it is
  enforced asynchronously, and on a freshly joined node the first Pod reached the internet (EXP-018).
- Commands run through the Kubernetes exec API (measured in EXP-017: p50 27-42 ms in region with a persistent
  client). Each command is wrapped the same way as in DockerSandbox: coreutils `timeout` as a backstop and the
  environment repaired by ENV_PROBE (Docker Hub task images lost their ENV, e.g. GOPATH).
- Files are written by exec with base64 chunks, so no extra tooling (tar, kubectl cp) is needed in the image.

The `kubernetes` client is an optional dependency (`pip install "affine-hosted-env[k8s]"`). Tests inject a fake
API object and an `exec_fn`.
"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
import uuid
from collections.abc import Callable

from .base import OUTPUT_CAP_BYTES, THREAD_CAPS, SandboxError, SandboxCommandError, SandboxLimitError
from .docker import ENV_PROBE

MANAGED_BY = {'app.kubernetes.io/managed-by': 'hosted-env'}
NETWORK_LABEL = 'hosted-env/network'
CONTAINER = 'task'
WRITE_CHUNK = 48 * 1024          # bytes of file content per exec when writing files (base64 grows it by 4/3)
HANDSHAKE_RETRIES, HANDSHAKE_BACKOFF = 3, 1.0     # exec WebSocket handshakes answered with 5xx (see ws_exec)
DEFAULT_DISK = '10g'             # ephemeral-storage per sandbox; coordinator/nodes.py sizes slots with the same default
KILLED_RC = (137, -1)            # SIGKILLed, or the exec stream ended without a status: check the container is alive
LIMIT_WAIT = 45.0                # seconds to wait for the Pod status to show why it died (eviction lands ~20 s late)

# exec_fn(pod, argv, timeout) -> (output bytes with stdout and stderr merged, returncode, timed_out)
ExecFn = Callable[[str, list[str], float], tuple[bytes, int, bool]]


def _mem(v: str) -> str:
    """Docker-style sizes ('2g', '512m') to Kubernetes quantities ('2Gi', '512Mi'). Only the lower-case docker
    suffixes are binary; Kubernetes ones ('2G' decimal, '2Gi') pass through, as coordinator/nodes.mem_bytes reads
    them, so slot sizing and the Pod limit agree."""
    v = str(v).strip()
    units = {'k': 'Ki', 'm': 'Mi', 'g': 'Gi', 't': 'Ti'}
    return v[:-1] + units[v[-1]] if v and v[-1] in units else v


def _positive(q) -> bool:
    """A size like '10g' / '0' / '' / None: true only for a positive amount (a 0 disk limit would evict at once)."""
    m = re.match(r'\s*([0-9]*\.?[0-9]+)', str(q or ''))
    return bool(m) and float(m.group(1)) > 0


def pod_manifest(name: str, image: str, cwd: str, *, namespace: str, node_name: str | None = None,
                 runtime_class: str | None = 'gvisor', cpus: str = '1', memory: str = '2g',
                 disk: str | None = DEFAULT_DISK, network: str = 'none', labels: dict | None = None) -> dict:
    """disk: ephemeral-storage limit (writable layer + logs). Past it the kubelet evicts this Pod alone, instead of
    the node filling up and evicting every sandbox on it (EXP-018: under gVisor the limit is enforced, a 1.5 GiB
    write into a 1Gi Pod is evicted within ~20 s). None leaves it unlimited."""
    res = {'cpu': str(cpus), 'memory': _mem(memory)}
    if _positive(disk):
        res['ephemeral-storage'] = _mem(disk)
    spec = {
        'restartPolicy': 'Never',
        'automountServiceAccountToken': False,
        'enableServiceLinks': False,
        'terminationGracePeriodSeconds': 0,
        'containers': [{
            'name': CONTAINER, 'image': image, 'command': ['sleep', 'infinity'], 'workingDir': cwd,
            'resources': {'limits': dict(res), 'requests': dict(res)},
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


def default_runtime_class(network: str) -> str:
    """gvisor-nonet runs runsc with --network=none: the sandbox gets loopback only, from its first instruction,
    without waiting for any policy controller. gvisor keeps the Pod network (for network != 'none')."""
    return 'gvisor-nonet' if network == 'none' else 'gvisor'


def parse_interfaces(proc_net_dev: str) -> list[str]:
    """Interface names from /proc/net/dev (two header lines, then 'name: counters...')."""
    return [ln.split(':', 1)[0].strip() for ln in proc_net_dev.splitlines()[2:] if ':' in ln]


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


# kubelet messages for a Pod evicted over its OWN ephemeral-storage limit (pod-level / container-level). A node-pressure
# eviction reads "The node was low on resource: ephemeral-storage" and is attributed only when it also says our
# container was using more than its request (requests equal limits, so that is over its own limit too).
_OWN_DISK_EVICTION = ('exceeds the total limit of containers', 'exceeded its local ephemeral storage limit')
_NODE_PRESSURE_USAGE = re.compile(r'Container \S+ was using (\S+), request is (\S+), has larger consumption of '
                                  r'ephemeral-storage')


def _over_request(msg: str) -> bool:
    from ...coordinator.nodes import mem_bytes
    for used, req in _NODE_PRESSURE_USAGE.findall(msg):
        try:
            if mem_bytes(used.rstrip(',.')) > mem_bytes(req.rstrip(',.')) > 0:
                return True
        except ValueError:
            continue
    return False


def limit_reason(pod) -> str | None:
    """Why the kubelet killed this Pod for a limit set on it alone, if it did: evicted for its own ephemeral-storage
    usage, or a container OOMKilled. Seen in EXP-018 on boat nodes under gVisor: a 1.5 GiB write into a 1Gi Pod was
    evicted ~15-20 s later; a 3 GiB allocation in a 512Mi Pod got the command SIGKILLed (exit 137) and the container
    OOMKilled. Requests equal limits (Guaranteed QoS) and slots are sized to the node, so an OOM kill is taken to be
    the sandbox's own."""
    st = pod.status
    msg = getattr(st, 'message', None) or ''
    if getattr(st, 'reason', None) == 'Evicted' and (any(m in msg for m in _OWN_DISK_EVICTION) or _over_request(msg)):
        return f'ephemeral-storage limit: {msg.strip()[:200]}'
    for cs in getattr(st, 'container_statuses', None) or []:
        term = getattr(cs.state, 'terminated', None) if getattr(cs, 'state', None) else None
        if term is not None and term.reason == 'OOMKilled':
            return 'memory limit: container OOMKilled'
    return None


def _parse_exit_status(raw: str) -> int:
    """A finished exec's exit code, from the error-channel status (JSON) the API server sends at close. Raises
    SandboxCommandError when the status is a Failure that carries no exit code -- the OCI runtime could not start
    the program (e.g. its executable was deleted). The kubernetes client's own `returncode` raises KeyError /
    ValueError on exactly that case, which is why we parse the status ourselves."""
    if not raw:
        return 0
    try:
        status = json.loads(raw) or {}
    except ValueError:
        return 0
    if status.get('status') == 'Success':
        return 0
    for cause in (status.get('details') or {}).get('causes') or []:
        if cause.get('reason') == 'ExitCode':
            try:
                return int(cause.get('message'))
            except (TypeError, ValueError):
                break
    raise SandboxCommandError(f'command could not start in the container: {status.get("message", "exec failed")}'[:200])


def _exit_code(resp) -> int:
    """resp's exit code, reading the exec error channel directly (not resp.returncode, which raises on a launch
    failure instead of telling it apart from a clean exit)."""
    try:
        from kubernetes.stream.ws_client import ERROR_CHANNEL
        raw = resp.read_channel(ERROR_CHANNEL)
    except SandboxCommandError:
        raise
    except Exception:  # noqa: BLE001  channel API unavailable: fall back to the client's own parse
        try:
            rc = resp.returncode
            return -1 if rc is None else int(rc)
        except Exception as e:  # noqa: BLE001  the launch-failure case the client cannot parse
            raise SandboxCommandError(f'command could not start in the container: {e}'[:200]) from None
    return _parse_exit_status(raw)


def _terminal(pod) -> bool:
    """The Pod/container has stopped: a terminated container, or a Failed/Succeeded/Unknown phase."""
    st = pod.status
    if getattr(st, 'phase', None) in ('Failed', 'Succeeded', 'Unknown'):
        return True
    for cs in getattr(st, 'container_statuses', None) or []:
        if getattr(cs, 'state', None) and getattr(cs.state, 'terminated', None):
            return True
    return False


def ws_exec(api, namespace: str) -> ExecFn:
    """exec_fn backed by the Kubernetes exec API (needs the `kubernetes` package).

    `kubernetes.stream.stream` swaps `request` on the ApiClient it is handed for the duration of the call, which is
    not thread-safe: on the client shared with ordinary calls, a concurrent Pod create / delete was sent over the
    exec WebSocket and failed (EXP-018, two rollouts at once). So exec never uses the shared client: each thread
    gets its own client built from the same configuration, used for exec only."""
    from kubernetes import client as k8s_client
    from kubernetes.client.rest import ApiException
    from kubernetes.stream import stream

    local = threading.local()

    def exec_api():
        a = getattr(local, 'api', None)
        if a is None:
            a = local.api = k8s_client.CoreV1Api(k8s_client.ApiClient(api.api_client.configuration))
        return a

    def run(pod: str, argv: list[str], timeout: float) -> tuple[bytes, int, bool]:
        # A failed WebSocket handshake means the command never started, so retrying is safe. On boat nodes the
        # API server reaches the kubelet through the node's own outbound tunnel, and about 1 in 70 rollouts saw
        # a handshake answered with 500 (EXP-018): retry those a few times before calling it an infra error.
        for attempt in range(HANDSHAKE_RETRIES + 1):
            try:
                resp = stream(exec_api().connect_get_namespaced_pod_exec, pod, namespace, container=CONTAINER,
                              command=argv, stdin=False, stdout=True, stderr=True, tty=False,
                              _preload_content=False)
                break
            except ApiException as e:
                if e.status == 404:
                    raise SandboxError('pod is gone') from None
                transient = e.status == 0 and 'Handshake status 5' in str(e.reason)
                if not transient or attempt == HANDSHAKE_RETRIES:
                    raise SandboxError(f'exec failed: {e.status} {e.reason}') from None
                time.sleep(HANDSHAKE_BACKOFF * (attempt + 1))
            except Exception as e:  # noqa: BLE001  any other client failure: the command did not run
                raise SandboxError(f'exec failed: {type(e).__name__}: {e}'[:300]) from None
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
            rc = -1 if timed_out else _exit_code(resp)
        except SandboxCommandError:
            resp.close()
            raise                                           # the program never launched: not a transport failure
        except Exception as e:  # noqa: BLE001  the websocket itself dropped mid-stream: transport, infra
            resp.close()
            raise SandboxError(f'exec stream failed: {type(e).__name__}: {e}'[:300]) from None
        resp.close()
        return bytes(buf), rc, timed_out
    return run


class K8sSandbox:
    def __init__(self, image: str, cwd: str, *, api=None, networking_api=None, exec_fn: ExecFn | None = None,
                 namespace: str = 'hosted-env', node_name: str | None = None, runtime_class: str | None = 'auto',
                 cpus: str = '1', memory: str = '2g', disk: str | None = DEFAULT_DISK, network: str = 'none',
                 repair_env: bool = True,
                 offline_check: str = 'interfaces', start_timeout: float = 600, poll: float = 0.5,
                 delete_timeout: float = 120, limit_wait: float = LIMIT_WAIT, pid1_guard: bool = True,
                 labels: dict | None = None, **_ignored):
        """runtime_class 'auto': gvisor-nonet for network='none', gvisor otherwise.
        offline_check (network='none' only): 'interfaces' refuses a sandbox that has any interface besides lo;
        'off' trusts the NetworkPolicy alone (dev clusters without gVisor). NetworkPolicy is applied either way."""
        if offline_check not in ('interfaces', 'off'):
            raise ValueError(f'offline_check must be interfaces or off: {offline_check!r}')
        if runtime_class == 'auto':
            runtime_class = default_runtime_class(network)
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
                                     runtime_class=runtime_class, cpus=cpus, memory=memory, disk=disk,
                                     network=network, labels=labels)
        self.networking_api, self.network, self.offline_check = networking_api, network, offline_check
        self.exec_fn = exec_fn or ws_exec(api, namespace)
        self.start_timeout, self.poll, self.repair_env = start_timeout, poll, repair_env
        self.delete_timeout, self.limit_wait, self.pid1_guard = delete_timeout, limit_wait, pid1_guard
        self.started, self.has_timeout, self.handed_out = False, False, False
        self.env: dict[str, str] = dict(THREAD_CAPS)

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
            why = _reason(pod) or (f'pod {pod.status.phase}' + (f' ({pod.status.reason}: {pod.status.message})'
                                                            if getattr(pod.status, 'reason', None) else '')
                                   if pod.status.phase in ('Failed', 'Succeeded') else None)
            if why:
                raise SandboxError(why)
            if time.monotonic() > deadline:
                raise SandboxError(f'pod not ready after {self.start_timeout:.0f}s ({pod.status.phase})')
            time.sleep(self.poll)
        if self.network == 'none' and self.offline_check == 'interfaces':
            self._check_offline()
        self.has_timeout = self.execute('command -v timeout', timeout=30, raw=True)['returncode'] == 0
        if self.pid1_guard:
            self._check_pid1_protected()
        if self.repair_env:
            out = self.execute(ENV_PROBE, timeout=60, raw=True)['output']
            self.env = {**THREAD_CAPS, **dict(ln.split('=', 1) for ln in out.splitlines()
                                              if '=' in ln and ln.split('=', 1)[0].isupper())}
        self.handed_out = True          # from here on its commands are the caller's: a limit kill is theirs

    def _check_pid1_protected(self) -> None:
        """Refuse the sandbox (pre-hand-out, so infra not model) unless PID 1 is protected from signals the model
        sends. The model is root in its rollout container with CAP_KILL; without SIGNAL_UNKILLABLE on the init
        process it could `kill -9 1` to kill `sleep infinity`, terminate the container and get its rollout voided
        (EXP-018 PR #15 review). With the gvisor runsc `signal-unkillable-policy=linux` config an intra-namespace
        SIGSTOP to the handler-less init is ignored; without it the init stops (state T). Probe: stop, read, cont."""
        r = self.execute("kill -STOP 1 2>/dev/null; s=$(awk '/^State:/{print $2}' /proc/1/status 2>/dev/null); "
                         "kill -CONT 1 2>/dev/null; printf '%s' \"$s\"", timeout=30, raw=True)
        state = r['output'].strip()
        if state == 'T':
            raise SandboxError('PID 1 is not signal-protected (state T after SIGSTOP): the runtime lacks '
                               'signal-unkillable-policy=linux, so the model could kill its own container; '
                               'refusing the sandbox')

    def _check_offline(self) -> None:
        """Fail closed unless the sandbox has no network interface besides loopback. NetworkPolicy enforcement is
        asynchronous (EXP-018: the first Pod on a freshly joined node could reach the internet), so a sandbox is
        only handed out when the runtime itself gave it no network (gVisor network=none)."""
        r = self.execute('cat /proc/net/dev', timeout=30, raw=True)
        ifaces = parse_interfaces(r['output']) if r['returncode'] == 0 else []
        if not ifaces or set(ifaces) != {'lo'}:
            raise SandboxError(f'sandbox is not offline (interfaces: {ifaces or "unreadable"}); use the '
                               "gvisor-nonet runtime class, or offline_check='off' to rely on NetworkPolicy alone")

    def _argv(self, command: str, cwd: str | None, timeout: int, raw: bool, login: bool = True) -> list[str]:
        inner = ['bash', '-lc', command] if login else ['bash', '--noprofile', '--norc', '-c', command]
        if self.has_timeout and not raw:
            inner = ['timeout', '-k', '5', str(int(timeout)), *inner]
        env = ['env', *(f'{k}={v}' for k, v in self.env.items())] if self.env else []
        # exec has no working-directory option: change directory first, then exec the real command
        return ['sh', '-c', 'cd "$1" || exit 125; shift; exec "$@"', 'sh', cwd or self.cwd, *env, *inner]

    def execute(self, command: str, cwd: str | None = None, timeout: int = 60, raw: bool = False,
                login: bool = True) -> dict:
        """Returns {output, returncode, reason}; reason is ok / client_timeout. Raises SandboxLimitError when the
        Pod was killed for its own memory or ephemeral-storage limit (by this command or an earlier one), plain
        SandboxError for any other exec failure."""
        if not self.started:
            raise SandboxError('sandbox not started')
        t0 = time.monotonic()
        try:
            out, rc, timed_out = self.exec_fn(self.name, self._argv(command, cwd, timeout, raw, login), timeout + 30)
        except SandboxCommandError as e:
            self._reconcile_command_error(e)    # the frame is shared with a dead/limit-killed container: check first
            raise
        except SandboxError as e:
            self._raise_if_limited(e)
            raise
        if self.has_timeout and not raw and (rc == 124 or (rc == 137 and time.monotonic() - t0 >= timeout)):
            timed_out = True                        # 137 here: `timeout -k 5` had to escalate to SIGKILL
        if rc in KILLED_RC and not timed_out and self.handed_out:
            self._check_alive()
        return {'output': out[:OUTPUT_CAP_BYTES].decode('utf-8', 'replace'), 'returncode': rc,
                'reason': 'client_timeout' if timed_out else 'ok'}

    def _check_alive(self) -> None:
        """A command came back SIGKILLed (137): that command's own exit code unless the container itself died.
        Reads the Pod status rather than exec'ing a probe binary -- the model is root in its rollout container and
        could delete whatever we probed, turning its own kill into an infra error (void)."""
        try:
            pod = self.api.read_namespaced_pod(self.name, self.ns)
        except Exception:  # noqa: BLE001  status unreadable: treat 137 as the command's own exit code
            return
        why = limit_reason(pod)
        if why:
            raise SandboxLimitError(why)
        for cs in getattr(pod.status, 'container_statuses', None) or []:
            term = getattr(cs.state, 'terminated', None) if getattr(cs, 'state', None) else None
            if term is not None:                    # the container is gone with no limit reason: an infra failure
                raise SandboxError(f'container terminated: {term.reason}')

    def _reconcile_command_error(self, err: 'SandboxCommandError') -> None:
        """An exec reported its program could not launch. The exact same error-channel frame (an InternalError with
        no exit code) is produced by a node-pressure eviction, a runsc crash, a shim/containerd restart or a
        container that already exited -- all genuine infra -- as well as by the model deleting a binary exec needs
        while its container stays up. So Pod status, not the message, decides (EXP-018 round-3 review):
          - its own limit killed it -> SandboxLimitError;
          - the container is terminated / the pod is Failed/Succeeded/Unknown / gone -> plain SandboxError (infra);
          - the container stays up for the whole limit_wait window -> the model's command: keep SandboxCommandError.
        Before hand-out a launch failure is a broken image (infra), so the SandboxCommandError is re-raised as is."""
        if not self.handed_out:
            raise err
        deadline = time.monotonic() + self.limit_wait
        while True:
            try:
                pod = self.api.read_namespaced_pod(self.name, self.ns)
            except Exception:  # noqa: BLE001  gone / unreadable: the container is not there, so this is not the model
                raise SandboxError(f'exec could not start and the pod is unreadable: {err}'[:300]) from None
            why = limit_reason(pod)
            if why:
                raise SandboxLimitError(why) from err
            if _terminal(pod):
                raise SandboxError(f'exec could not start and the container is not running: {err}'[:300]) from err
            if time.monotonic() > deadline:         # stayed up the whole window: nothing killed it but the command
                raise err
            time.sleep(self.poll)

    def _raise_if_limited(self, err: Exception) -> None:
        """After a failed exec: raise SandboxLimitError if the Pod status shows (within limit_wait, since the kubelet
        reports an eviction or OOM kill some seconds late) that its own limit killed it; otherwise return. Before
        start() hands the sandbox out only our own probes have run, so nothing is attributed then."""
        if not self.handed_out:
            return
        deadline = time.monotonic() + self.limit_wait
        while True:
            try:
                pod = self.api.read_namespaced_pod(self.name, self.ns)
            except Exception:  # noqa: BLE001  gone or unreadable: cannot be attributed to the sandbox's own limits
                return
            why = limit_reason(pod)
            if why:
                raise SandboxLimitError(why) from err
            if pod.status.phase in ('Failed', 'Succeeded') or time.monotonic() > deadline:
                return
            time.sleep(self.poll)

    def copy_text_to(self, text: str, remote_path: str):
        data = text.encode('utf-8')
        for i in range(0, max(len(data), 1), WRITE_CHUNK):
            b64 = base64.b64encode(data[i:i + WRITE_CHUNK]).decode()
            op = '>' if i == 0 else '>>'
            argv = ['sh', '-c', f'printf %s "$1" | base64 -d {op} "$2"', 'sh', b64, remote_path]
            try:
                out, rc, _ = self.exec_fn(self.name, argv, 120)
            except SandboxCommandError as e:
                self._reconcile_command_error(e)     # a dead container's exec looks the same: check Pod status
                raise
            except SandboxError as e:                    # like execute: a limit kill is attributed, not hidden
                self._raise_if_limited(e)
                raise
            if rc != 0:
                if rc in KILLED_RC and self.handed_out:
                    self._check_alive()
                raise SandboxError(f'write {remote_path} failed: {out[-300:].decode("utf-8", "replace")}')

    def cleanup(self):
        """Delete the Pod and wait until it is gone: a deleted Pod keeps its CPU and memory on the node until
        the kubelet tears it down, and the next sandbox placed on that node (the grading container, the next
        group) would be refused with OutOfmemory / OutOfcpu (EXP-018)."""
        if not self.started:
            return
        self.started = False
        try:
            self.api.delete_namespaced_pod(self.name, self.ns, grace_period_seconds=0)
        except Exception as e:
            if getattr(e, 'status', None) != 404:
                raise
        deadline = time.monotonic() + self.delete_timeout
        while time.monotonic() < deadline:
            try:
                self.api.read_namespaced_pod(self.name, self.ns)
            except Exception as e:
                if getattr(e, 'status', None) == 404:
                    return
                raise
            time.sleep(self.poll)
        raise SandboxError(f'pod {self.name} still exists {self.delete_timeout:.0f}s after delete')
