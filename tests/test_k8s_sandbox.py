"""providers/sandbox/k8s.py with a fake Kubernetes API. exec_fn runs the generated argv on this machine, so the
command wrapping (cd, env, timeout) is exercised for real; only the Pod lifecycle is faked."""
import subprocess
import threading
from types import SimpleNamespace as NS

import pytest

from hosted_env.providers.sandbox import SandboxError, SandboxLimitError
from hosted_env.providers.sandbox.k8s import (WRITE_CHUNK, K8sSandbox, ensure_network_policies, limit_reason,
                                              network_policy_manifest, pod_manifest)


class ApiErr(Exception):
    def __init__(self, status):
        super().__init__(f'status {status}')
        self.status = status


def _pod(phase='Running', ready=True, waiting=None):
    st = NS(waiting=NS(reason=waiting, message='nope') if waiting else None, terminated=None)
    return NS(status=NS(phase=phase, container_statuses=[NS(ready=ready, state=st)]))


class FakeCore:
    def __init__(self, pods=None, lingering=0):
        self.created, self.deleted, self.lingering = [], [], lingering
        self.pods = list(pods or [_pod('Pending', False), _pod()])

    def create_namespaced_pod(self, ns, body):
        self.created.append((ns, body))

    def read_namespaced_pod(self, name, ns):
        if (name, ns) in self.deleted:
            if self.lingering:                       # the kubelet is still tearing it down
                self.lingering -= 1
                return self.pods[-1]
            raise ApiErr(404)
        return self.pods.pop(0) if len(self.pods) > 1 else self.pods[0]

    def delete_namespaced_pod(self, name, ns, grace_period_seconds=None):
        if (name, ns) in self.deleted:
            raise ApiErr(404)
        self.deleted.append((name, ns))


class FakeNet:
    def __init__(self):
        self.calls = 0

    def create_namespaced_network_policy(self, ns, body):
        self.calls += 1
        if self.calls > 1:
            raise ApiErr(409)


def local_exec(pod, argv, timeout):
    try:
        r = subprocess.run(argv, capture_output=True, timeout=timeout)
        return r.stdout + r.stderr, r.returncode, False
    except subprocess.TimeoutExpired as e:
        return (e.stdout or b''), -1, True


PROC_NET_DEV = ('Inter-|   Receive                            |  Transmit\n'
                ' face |bytes    packets errs drop fifo frame compressed multicast|bytes\n')
LO_ONLY = PROC_NET_DEV + '    lo:       0       0    0    0    0     0          0         0        0\n'
WITH_ETH0 = LO_ONLY + '  eth0:    1234      10    0    0    0     0          0         0      567\n'


def offline_exec(dev=LO_ONLY, rc=0):
    """local_exec, except that the offline check sees `dev` as the sandbox's /proc/net/dev."""
    def run(pod, argv, timeout):
        if argv[-1] == 'cat /proc/net/dev':
            return dev.encode(), rc, False
        return local_exec(pod, argv, timeout)
    return run


def make(tmp_path, **kw):
    kw.setdefault('repair_env', False)
    return K8sSandbox('img:1', str(tmp_path), api=kw.pop('api', FakeCore()), networking_api=kw.pop('net', FakeNet()),
                      exec_fn=kw.pop('exec_fn', offline_exec()), poll=0, **kw)


def test_pod_manifest_defaults_and_placement():
    m = pod_manifest('p1', 'img:1', '/testbed', namespace='ns', node_name='node-a', memory='3g', cpus='2')
    spec, c = m['spec'], m['spec']['containers'][0]
    assert (spec['runtimeClassName'], spec['nodeName'], spec['automountServiceAccountToken']) == ('gvisor', 'node-a', False)
    assert spec['enableServiceLinks'] is False and spec['restartPolicy'] == 'Never' and spec['dnsPolicy'] == 'None'
    assert c['resources']['limits'] == {'cpu': '2', 'memory': '3Gi', 'ephemeral-storage': '10Gi'}
    assert c['resources']['requests'] == c['resources']['limits'] and c['workingDir'] == '/testbed'
    assert c['securityContext'] == {'allowPrivilegeEscalation': False} and c['command'] == ['sleep', 'infinity']
    assert m['metadata']['labels']['hosted-env/network'] == 'none'
    m2 = pod_manifest('p2', 'i', '/w', namespace='ns', runtime_class=None, network='bridge', disk=None)
    assert 'runtimeClassName' not in m2['spec'] and 'nodeName' not in m2['spec'] and 'dnsPolicy' not in m2['spec']
    assert 'ephemeral-storage' not in m2['spec']['containers'][0]['resources']['limits']
    m3 = pod_manifest('p3', 'i', '/w', namespace='ns', disk='512m')
    assert m3['spec']['containers'][0]['resources']['limits']['ephemeral-storage'] == '512Mi'


def test_network_policy_is_default_deny_and_idempotent():
    pol = network_policy_manifest('ns')
    assert pol['spec'] == {'podSelector': {'matchLabels': {'hosted-env/network': 'none'}},
                           'policyTypes': ['Egress', 'Ingress']}
    net = FakeNet()
    ensure_network_policies(net, 'ns')
    ensure_network_policies(net, 'ns')                          # 409 the second time is fine
    assert net.calls == 2


def test_start_execute_cleanup(tmp_path):
    api = FakeCore()
    sb = make(tmp_path, api=api)
    with pytest.raises(SandboxError, match='not started'):
        sb.execute('true')
    sb.start()
    assert api.created[0][1]['metadata']['name'] == sb.name
    r = sb.execute('pwd; echo err >&2; exit 3')
    assert r['returncode'] == 3 and r['reason'] == 'ok' and str(tmp_path) in r['output'] and 'err' in r['output']
    (tmp_path / 'sub').mkdir()
    assert sb.execute('pwd', cwd=str(tmp_path / 'sub'))['output'].strip().endswith('/sub')
    assert sb.execute('pwd', cwd='/does/not/exist')['returncode'] == 125
    sb.env = {'FOO': 'bar baz'}
    assert sb.execute('echo "$FOO"')['output'].strip() == 'bar baz'
    sb.cleanup()
    sb.cleanup()                                                # idempotent
    assert api.deleted == [(sb.name, 'hosted-env')]


def test_env_probe_runs_on_start(tmp_path):
    sb = make(tmp_path, repair_env=True)
    sb.start()
    assert isinstance(sb.env, dict)                             # whatever this host needs; parsing must not fail


def test_copy_text_multi_chunk(tmp_path):
    sb = make(tmp_path)
    sb.start()
    text = ('héllo wörld ✓\n' * (WRITE_CHUNK // 8)) + 'end'
    assert len(text.encode()) > 2 * WRITE_CHUNK
    sb.copy_text_to(text, str(tmp_path / 'f.txt'))
    assert (tmp_path / 'f.txt').read_text(encoding='utf-8') == text
    sb.copy_text_to('', str(tmp_path / 'empty.txt'))
    assert (tmp_path / 'empty.txt').read_text() == ''
    with pytest.raises(SandboxError, match='write'):
        sb.copy_text_to('x', str(tmp_path / 'missing' / 'f.txt'))


def test_bad_image_fails_fast_and_cleans_up(tmp_path):
    api = FakeCore([_pod('Pending', False, waiting='ImagePullBackOff')])
    sb = make(tmp_path, api=api)
    with pytest.raises(SandboxError, match='ImagePullBackOff'):
        sb.start()
    sb.cleanup()                                                # the executor always cleans up after a failed start
    assert api.deleted == [(sb.name, 'hosted-env')]


def test_start_timeout(tmp_path):
    sb = make(tmp_path, api=FakeCore([_pod('Pending', False)]), start_timeout=0)
    with pytest.raises(SandboxError, match='not ready'):
        sb.start()


def test_pod_gone_during_exec(tmp_path):
    def gone(pod, argv, timeout):
        raise SandboxError('pod is gone')
    sb = K8sSandbox('i', str(tmp_path), api=FakeCore(), exec_fn=gone, repair_env=False, poll=0)
    with pytest.raises(SandboxError, match='gone'):
        sb.start()                                              # the `command -v timeout` probe already fails


def _dead(reason=None, message='', terminated=None, phase='Failed'):
    st = NS(waiting=None, terminated=NS(reason=terminated) if terminated else None)
    return NS(status=NS(phase=phase, reason=reason, message=message, container_statuses=[NS(ready=False, state=st)]))


OWN_DISK = 'Pod ephemeral local storage usage exceeds the total limit of containers 1Gi. '   # as seen in EXP-018
NODE_DISK = 'The node was low on resource: ephemeral-storage. '


def test_limit_reason_only_blames_the_pods_own_limits():
    assert limit_reason(_dead('Evicted', OWN_DISK)).startswith('ephemeral-storage limit: Pod ephemeral')
    assert limit_reason(_dead('Evicted', 'Container task exceeded its local ephemeral storage limit "10Gi". '))
    assert limit_reason(_dead(terminated='OOMKilled')) == 'memory limit: container OOMKilled'
    assert limit_reason(_dead('Evicted', NODE_DISK)) is None       # node pressure: not the sandbox's doing
    assert limit_reason(_dead(terminated='Error')) is None
    assert limit_reason(_pod()) is None


def scripted(**plan):
    """offline_exec, except for commands named in plan (argv[-1] -> callable returning an exec result)."""
    base = offline_exec()

    def run(pod, argv, timeout):
        act = plan.get(argv[-1])
        return act() if act else base(pod, argv, timeout)
    return run


def raises(msg):
    def act():
        raise SandboxError(msg)
    return act


def started(tmp_path, then, **plan):
    """A started sandbox whose Pod reads as `then` from now on."""
    core = FakeCore()
    sb = make(tmp_path, api=core, exec_fn=scripted(**plan), limit_wait=5)
    sb.start()
    core.pods = [then]
    return sb


def test_exec_failure_after_an_own_limit_kill_is_a_limit_error(tmp_path):
    sb = started(tmp_path, _dead('Evicted', OWN_DISK), work=raises('exec failed: 0 Handshake status 500'))
    with pytest.raises(SandboxLimitError, match='ephemeral-storage limit'):
        sb.execute('work')
    sb = started(tmp_path, _dead('Evicted', NODE_DISK), work=raises('exec failed: 0 Handshake status 500'))
    with pytest.raises(SandboxError) as e:
        sb.execute('work')
    assert not isinstance(e.value, SandboxLimitError)             # node-wide eviction stays an infra error
    sb = started(tmp_path, _pod(), work=raises('exec failed: 500'))
    sb.limit_wait = 0
    with pytest.raises(SandboxError) as e:
        sb.execute('work')                                        # Pod still running: some other failure
    assert not isinstance(e.value, SandboxLimitError)


def test_sigkilled_command_checks_the_container(tmp_path):
    oom = _dead(terminated='OOMKilled')
    sb = started(tmp_path, oom, hog=lambda: (b'', 137, False), true=raises('exec stream failed: KeyError'))
    with pytest.raises(SandboxLimitError, match='memory limit'):
        sb.execute('hog')                                         # the container died with the command
    sb = started(tmp_path, _pod(), hog=lambda: (b'Killed', 137, False))
    assert sb.execute('hog')['returncode'] == 137                 # only the process died: an ordinary exit status


def test_executor_builds_k8s_sandboxes(tmp_path):
    from nacl.signing import SigningKey

    from hosted_env.envs import DockerTaskEnv
    from hosted_env.executor.core import EnvRuntime, Worker
    from hosted_env.render import CharRenderer
    core, net = FakeCore(), FakeNet()
    rt = EnvRuntime(env=DockerTaskEnv('e', {'group_size': 8}), renderer=CharRenderer(), sandbox='k8s',
                    sandbox_opts={'exec_fn': offline_exec(), 'node_name': 'node-a', 'repair_env': False,
                                  'poll': 0})
    w = Worker(worker_id='local', key=SigningKey.generate(), gateway=None, gateway_pub=SigningKey.generate().verify_key,
               runtimes={'e': rt}, public_url='http://x', k8s_api=(core, net))
    sb = w._sandbox_factory(rt)('img:1', str(tmp_path))
    assert isinstance(sb, K8sSandbox) and sb.manifest['spec']['nodeName'] == 'node-a'
    placed = w._sandbox_factory(rt, node='node-b')('img:1', str(tmp_path))
    assert placed.manifest['spec']['nodeName'] == 'node-b'           # the coordinator's choice wins
    sb.start()
    assert core.created and net.calls == 1 and sb.execute('echo ok')['output'].strip() == 'ok'
    sb.cleanup()
    w.pool.shutdown()


def test_offline_runtime_class_defaults(tmp_path):
    rc = lambda **kw: make(tmp_path, **kw).manifest['spec'].get('runtimeClassName')   # noqa: E731
    assert rc() == 'gvisor-nonet' and rc(network='bridge') == 'gvisor'
    assert rc(runtime_class='gvisor') == 'gvisor' and rc(runtime_class=None) is None
    with pytest.raises(ValueError):
        make(tmp_path, offline_check='maybe')


def test_offline_check_fails_closed(tmp_path):
    for dev, rc in ((WITH_ETH0, 0), ('', 1), (PROC_NET_DEV, 0)):        # extra interface, unreadable, empty
        api = FakeCore()
        sb = make(tmp_path, api=api, exec_fn=offline_exec(dev, rc))
        with pytest.raises(SandboxError, match='not offline'):
            sb.start()
        sb.cleanup()
        assert api.deleted == [(sb.name, 'hosted-env')]
    sb = make(tmp_path, exec_fn=local_exec, offline_check='off', runtime_class=None)  # policy-only dev clusters
    sb.start()
    assert sb.execute('echo ok')['output'].strip() == 'ok'
    sb2 = make(tmp_path, exec_fn=offline_exec(WITH_ETH0), network='bridge')           # not offline: not checked
    sb2.start()


def test_exec_never_uses_the_shared_client(monkeypatch):
    """kubernetes.stream.stream swaps `request` on the client it is given; on the shared client a concurrent Pod
    create / delete went out over the exec WebSocket (EXP-018). Exec must get a per-thread client of its own."""
    kubernetes = pytest.importorskip('kubernetes')
    stream_mod = pytest.importorskip('kubernetes.stream')

    from hosted_env.providers.sandbox.k8s import ws_exec
    cfg = kubernetes.client.Configuration()
    cfg.host = 'https://cp.test:6443'
    shared = kubernetes.client.CoreV1Api(kubernetes.client.ApiClient(cfg))
    used = []

    class Resp:
        returncode = 0

        def is_open(self):
            return False

        def close(self):
            pass

    def fake_stream(method, pod, ns, **kw):
        used.append(method.__self__)
        return Resp()
    monkeypatch.setattr(stream_mod, 'stream', fake_stream)
    run = ws_exec(shared, 'hosted-env')
    assert run('p', ['true'], 5) == (b'', 0, False)
    run('p', ['true'], 5)
    t = threading.Thread(target=run, args=('p', ['true'], 5))
    t.start()
    t.join()
    a, b, c = used
    assert a is b and a is not c                                         # one exec client per thread
    assert all(x is not shared and x.api_client is not shared.api_client for x in used)
    assert a.api_client.configuration is cfg                             # same cluster and credentials


def test_cleanup_waits_until_the_pod_is_gone(tmp_path):
    """A deleted Pod keeps its CPU and memory on the node until the kubelet is done with it; the next sandbox on
    that node (the grading container) was refused with OutOfmemory when cleanup returned right away (EXP-018)."""
    api = FakeCore(lingering=3)
    sb = make(tmp_path, api=api)
    sb.start()
    sb.cleanup()
    assert api.lingering == 0 and api.deleted == [(sb.name, 'hosted-env')]   # polled until 404
    stuck = FakeCore(lingering=10 ** 6)
    sb = make(tmp_path, api=stuck, delete_timeout=0.05)
    sb.start()
    with pytest.raises(SandboxError, match='still exists'):
        sb.cleanup()


def test_exec_retries_a_failed_handshake_only(monkeypatch):
    """A 5xx WebSocket handshake means the command never ran: retried. Anything else is not."""
    kubernetes = pytest.importorskip('kubernetes')
    stream_mod = pytest.importorskip('kubernetes.stream')
    from kubernetes.client.rest import ApiException

    from hosted_env.providers.sandbox import k8s
    monkeypatch.setattr(k8s, 'HANDSHAKE_BACKOFF', 0)
    cfg = kubernetes.client.Configuration()
    cfg.host = 'https://cp.test:6443'
    shared = kubernetes.client.CoreV1Api(kubernetes.client.ApiClient(cfg))
    calls = []

    class Resp:
        returncode = 0

        def is_open(self):
            return False

        def close(self):
            pass

    def flaky(n_fail, err):
        def fake(method, pod, ns, **kw):
            calls.append(1)
            if len(calls) <= n_fail:
                raise err
            return Resp()
        return fake
    handshake = ApiException(status=0, reason='Handshake status 500 Internal Server Error -+-+- {...}')
    monkeypatch.setattr(stream_mod, 'stream', flaky(2, handshake))
    assert k8s.ws_exec(shared, 'hosted-env')('p', ['true'], 5) == (b'', 0, False) and len(calls) == 3
    calls.clear()
    monkeypatch.setattr(stream_mod, 'stream', flaky(10, handshake))
    with pytest.raises(SandboxError, match='Handshake status 500'):
        k8s.ws_exec(shared, 'hosted-env')('p', ['true'], 5)
    assert len(calls) == k8s.HANDSHAKE_RETRIES + 1
    calls.clear()
    monkeypatch.setattr(stream_mod, 'stream', flaky(10, ApiException(status=403, reason='Forbidden')))
    with pytest.raises(SandboxError, match='403'):
        k8s.ws_exec(shared, 'hosted-env')('p', ['true'], 5)
    assert len(calls) == 1
