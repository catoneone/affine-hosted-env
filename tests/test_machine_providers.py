"""providers/machine: data types and the static provider."""
import pytest

from hosted_env.infra.providers.machine import Machine, MachineSpec, NotSupported, StaticProvider


def test_spec_requires_prefix():
    assert MachineSpec('large', 'hosted-env-n1').ttl_seconds is None
    with pytest.raises(ValueError, match='hosted-env-'):
        MachineSpec('large', 'someone-elses-vm')


def test_machine_state_checked():
    Machine('m1', 'boat', 'hosted-env-m1', 'running')
    with pytest.raises(ValueError):
        Machine('m1', 'boat', 'hosted-env-m1', 'exploded')


def test_static_provider_counts_but_never_creates():
    p = StaticProvider([{'name': 'gpu-box', 'public_ipv4': '203.0.113.7', 'price_per_hour': 0.5}, {'name': 'b'}])
    ms = p.list()
    assert [(m.id, m.state, m.provider, m.kind) for m in ms] == [('gpu-box', 'running', 'static', 'static'),
                                                                  ('b', 'running', 'static', 'static')]
    assert ms[0].public_ipv4 == '203.0.113.7' and p.offers() == []
    with pytest.raises(NotSupported):
        p.create(MachineSpec('x', 'hosted-env-x'), '#!/bin/sh')
    with pytest.raises(NotSupported):
        p.delete('gpu-box')


# ---------------------------------------------------------------- boat (fake HTTP)
class FakeBoat:
    """Just enough of the boat REST API. Records every request."""

    def __init__(self, ready_after=2, fail_create=False):
        import httpx
        self.reqs, self.sandboxes, self.ready_after, self.fail_create = [], {}, ready_after, fail_create
        self.http = httpx.Client(base_url='https://boat.test/api/v1', transport=httpx.MockTransport(self.handle))

    def handle(self, request):
        import json as _json

        import httpx
        self.reqs.append(request)
        p, m = request.url.path.removeprefix('/api/v1'), request.method
        assert request.headers['authorization'] == 'Bearer tok'
        if m == 'POST' and p == '/sandboxes':
            if self.fail_create:
                return httpx.Response(402, json={'code': 'payment_required'})
            sid = f'bx_{len(self.sandboxes)}'
            self.sandboxes[sid] = {'id': sid, 'name': 'Box 2026', 'type': _json.loads(request.content)['type'],
                                   'state': 'provisioned', 'ip': '2001:db8::1', 'polls': 0}
            return httpx.Response(202, json={'sandbox': self.sandboxes[sid]})
        if m == 'GET' and p == '/sandboxes':
            return httpx.Response(200, json={'sandboxes': list(self.sandboxes.values()), 'pageInfo': {}})
        sid = p.split('/')[2]
        s = self.sandboxes.get(sid)
        if s is None:
            return httpx.Response(404, json={'code': 'not_found'})
        if m == 'GET':
            s['polls'] += 1
            if s['polls'] >= self.ready_after:
                s['state'] = 'idle'
            return httpx.Response(200, json={'sandbox': s})
        if m == 'DELETE':
            assert request.headers['x-ascii-confirm-delete'] == sid
            del self.sandboxes[sid]
            return httpx.Response(202, json={'type': 'sandbox.deleting'})
        return httpx.Response(200, json={'processId': 7})                  # files / commands


def test_boat_create_runs_boot_script_and_lists_own(tmp_path):
    import base64
    import json as _json

    from hosted_env.infra.providers.machine import MachineError
    from hosted_env.infra.providers.machine.boat import BOOT_PATH, BoatProvider
    fb = FakeBoat()
    fb.sandboxes['bx_other'] = {'id': 'bx_other', 'name': 'someone else', 'type': 'large', 'state': 'idle',
                                'ip': '2001:db8::9', 'polls': 0}
    tok = tmp_path / 'boat.token'
    tok.write_text('tok\n')
    p = BoatProvider.from_token_file(str(tok), http=fb.http, poll=0)
    m = p.create(MachineSpec('large', 'hosted-env-n1', ttl_seconds=3600), '#!/bin/bash\necho boot\n')
    assert (m.state, m.kind, m.public_ipv6, m.public_ipv4, m.price_per_hour) == ('running', 'large', '2001:db8::1',
                                                                                  None, 0.072)
    post = _json.loads(fb.reqs[0].content)
    assert post == {'type': 'large', 'name': 'hosted-env-n1', 'ttlSeconds': 3600}
    assert 'idempotency-key' in fb.reqs[0].headers
    files = next(r for r in fb.reqs if r.url.path.endswith('/files'))
    assert _json.loads(files.content)['path'] == BOOT_PATH
    assert base64.b64decode(_json.loads(files.content)['content']).decode() == '#!/bin/bash\necho boot\n'
    cmd = _json.loads(next(r for r in fb.reqs if r.url.path.endswith('/commands')).content)
    assert cmd['detached'] is True and cmd['command'].startswith(f'sudo bash {BOOT_PATH}')
    assert [x.id for x in p.list()] == [m.id]                           # the API ignored our name; the id counts
    assert [o.kind for o in p.offers()] == ['small', 'default', 'large']
    p.delete(m.id)
    p.delete(m.id)                                                      # 404 = already gone
    assert p.list() == [] and 'bx_other' in fb.sandboxes
    with pytest.raises(MachineError, match='402 payment_required') as e:
        BoatProvider('tok', http=FakeBoat(fail_create=True).http).create(MachineSpec('large', 'hosted-env-x'), '')
    assert 'tok' not in str(e.value)                                    # errors never carry the token


def test_boat_create_times_out_when_never_ready():
    from hosted_env.infra.providers.machine import CreateFailed
    from hosted_env.infra.providers.machine.boat import BoatProvider
    fb, ids = FakeBoat(ready_after=10 ** 9), []
    p = BoatProvider('tok', http=fb.http, ready_timeout=0, poll=0)
    with pytest.raises(CreateFailed, match='did not become ready') as e:
        p.create(MachineSpec('small', 'hosted-env-n2'), '', on_created=ids.append)
    assert ids == [e.value.machine_id] and e.value.cleaned is True       # the id was reported before the boot
    assert p.list() == [] and fb.sandboxes == {}                         # the provider deleted what it created


def test_boat_failed_cleanup_is_reported_not_swallowed():
    import httpx

    from hosted_env.infra.providers.machine import CreateFailed
    from hosted_env.infra.providers.machine.boat import BoatProvider
    fb = FakeBoat(ready_after=10 ** 9)
    fb.sandboxes['bx_other'] = {'id': 'bx_other', 'name': 'hosted-env-cp1', 'type': 'large', 'state': 'idle',
                                'ip': '2001:db8::9', 'polls': 0}
    handle = fb.handle

    def no_delete(request):
        if request.method == 'DELETE':
            fb.reqs.append(request)
            return httpx.Response(500, json={'code': 'internal'})
        return handle(request)
    fb.http = httpx.Client(base_url='https://boat.test/api/v1', transport=httpx.MockTransport(no_delete))
    p = BoatProvider('tok', http=fb.http, ready_timeout=0, poll=0)
    with pytest.raises(CreateFailed) as e:
        p.create(MachineSpec('small', 'hosted-env-n3'), '')
    assert e.value.cleaned is False and e.value.machine_id in fb.sandboxes
    deletes = [r.url.path for r in fb.reqs if r.method == 'DELETE']
    assert deletes == [f'/api/v1/sandboxes/{e.value.machine_id}']     # only the machine it was creating


# ---------------------------------------------------------------- targon (fake HTTP)
class FakeTargon:
    def __init__(self, ready_after=2):
        import httpx
        self.reqs, self.workloads, self.ready_after = [], {}, ready_after
        self.http = httpx.Client(base_url='https://targon.test/tha/v3', transport=httpx.MockTransport(self.handle))

    def handle(self, request):
        import json as _json

        import httpx
        self.reqs.append(request)
        p, m = request.url.path.removeprefix('/tha/v3'), request.method
        if p == '/inventory':
            assert 'authorization' not in request.headers                      # public endpoint
            return httpx.Response(200, json=[                                   # live shape: a bare list
                {'name': 'cpu-small-amd', 'type': 'vm', 'gpu': False, 'cost_per_hour': 0.15, 'available': 6,
                 'spec': {'cpu_millicores': 4000, 'memory_mib': 53644, 'disk_size_mib': 393216,
                          'network_mode': 'SHARED'}},
                {'name': 'h200-small', 'type': 'vm', 'gpu': True, 'cost_per_hour': 3.59, 'available': 0, 'spec': {}},
                {'name': 'sandbox-small', 'type': 'sandbox', 'gpu': False, 'cost_per_hour': 0.045, 'spec': {}}])
        assert request.headers['authorization'] == 'Bearer tok'
        if p == '/orgs':
            return httpx.Response(200, json={'items': [{'slug': 'al-1'}]})
        assert p.startswith('/orgs/al-1/workloads')
        rest = p.removeprefix('/orgs/al-1/workloads').strip('/').split('/')
        if m == 'POST' and rest == ['']:
            b = _json.loads(request.content)
            uid = f'wrk-{len(self.workloads)}'
            self.workloads[uid] = {'uid': uid, 'name': b['name'], 'resource_name': b['resource_name'], 'body': b,
                                   'polls': 0, 'state': {'status': 'registered'}}
            return httpx.Response(200, json={'uid': uid, 'state': {'status': 'registered'}})
        if m == 'GET' and rest == ['']:
            items = [{k: v for k, v in w.items() if k != 'body'} for w in self.workloads.values()]
            return httpx.Response(200, json={'items': items, 'next_cursor': None})
        w = self.workloads.get(rest[0])
        if w is None:
            return httpx.Response(404, json={'reason': 'WORKLOAD_NOT_FOUND'})
        if m == 'POST' and rest[1:] == ['deploy']:
            w['state'] = {'status': 'provisioning', 'urls': [{'port': 6443, 'url': 'tcp://160.0.0.1:6443'}]}
            return httpx.Response(200, json=w)
        if m == 'GET' and rest[1:] == ['state']:
            w['polls'] += 1
            if w['polls'] >= self.ready_after:
                w['state'] = {'status': 'running', 'public_ip': '160.0.0.1', 'ssh_port': 20002,
                              'urls': [{'port': 6443, 'url': '160.0.0.1:20004'},
                                       {'port': 51820, 'url': '160.0.0.1:20000'}]}
            return httpx.Response(200, json=w['state'])
        if m == 'GET':
            return httpx.Response(200, json={k: v for k, v in w.items() if k != 'body'})
        if m == 'DELETE':
            del self.workloads[rest[0]]
            return httpx.Response(204)
        raise AssertionError((m, p))


def test_targon_create_maps_ports_and_runs_boot(tmp_path):
    from hosted_env.infra.providers.machine import MachineError
    from hosted_env.infra.providers.machine.targon import TargonProvider
    ft, booted = FakeTargon(), []
    p = TargonProvider('tok', ssh_key_uid='shk-1', boot_runner=lambda m, s: booted.append((m.id, s)), http=ft.http,
                       poll=0)
    [o] = p.offers()
    assert (o.kind, o.vcpus, o.price_per_hour, o.available, o.network) == ('cpu-small-amd', 4.0, 0.15, 6, 'SHARED')
    m = p.create(MachineSpec('cpu-small-amd', 'hosted-env-tg1', ports=('6443/tcp', '51820/udp')), 'echo boot')
    assert m.state == 'running' and m.public_ipv4 == '160.0.0.1' and m.price_per_hour == 0.15
    assert m.ports == {'6443': '160.0.0.1:20004', '51820': '160.0.0.1:20000', 'ssh': '160.0.0.1:20002'}
    assert booted == [(m.id, 'echo boot')]                               # boot runs only once it is running
    m2 = p.create(MachineSpec('cpu-small-amd', 'hosted-env-tg3', ports=('51820/udp',)),
                  lambda mach: f"wg-port {mach.ports['51820'].rsplit(':', 1)[1]}")
    assert booted[-1] == (m2.id, 'wg-port 20000')                       # rendered with the remapped port
    p.delete(m2.id)
    body = ft.workloads[m.id]['body']
    assert body['ports'] == [{'port': 6443, 'protocol': 'TCP'}, {'port': 51820, 'protocol': 'UDP'}]
    assert (body['type'], body['image'], body['ssh_keys'], body['vm_config']) == (
        'VM', 'ubuntu-24-04-lts', ['shk-1'], {'hostname': 'hosted-env-tg1'})
    ft.workloads['wrk-x'] = {'uid': 'wrk-x', 'name': 'not-ours', 'state': {'status': 'running'}, 'polls': 0}
    assert [x.id for x in p.list()] == [m.id]
    p.delete(m.id)
    p.delete(m.id)
    assert [x.id for x in p.list()] == [] and 'wrk-x' in ft.workloads
    with pytest.raises(MachineError, match='32 characters'):
        p.create(MachineSpec('cpu-small-amd', 'hosted-env-' + 'x' * 30), '')


def test_targon_never_running_fails_without_boot():
    from hosted_env.infra.providers.machine import CreateFailed, MachineError
    from hosted_env.infra.providers.machine.targon import TargonProvider
    booted, ids = [], []
    p = TargonProvider('tok', ssh_key_uid='k', boot_runner=lambda m, s: booted.append(m), org='al-1',
                       http=FakeTargon(ready_after=10 ** 9).http, ready_timeout=0, poll=0)
    with pytest.raises(CreateFailed, match='did not start') as e:
        p.create(MachineSpec('cpu-small-amd', 'hosted-env-tg2'), '', on_created=ids.append)
    assert ids == [e.value.machine_id] and e.value.cleaned is True
    assert booted == [] and p.list() == []                               # the provider deleted what it created

    def bad_ssh(m, s):
        raise MachineError('ssh refused')
    p2 = TargonProvider('tok', ssh_key_uid='k', boot_runner=bad_ssh, org='al-1', http=FakeTargon().http, poll=0)
    with pytest.raises(CreateFailed, match='ssh refused'):
        p2.create(MachineSpec('cpu-small-amd', 'hosted-env-tg4'), '')
    assert p2.list() == []


# ---------------------------------------------------------------- boot script
def _bash_syntax_ok(script: str) -> None:
    import subprocess
    r = subprocess.run(['bash', '-n'], input=script.encode(), capture_output=True)
    assert r.returncode == 0, r.stderr.decode()


def test_boot_script_defaults():
    from hosted_env.infra.providers.machine.bootstrap import K3S_VERSION, render_boot_script
    tok = 'K10abc::server:s3cr3t'
    s = render_boot_script(server_url='https://cp.example:6443', join_token=tok,
                           labels={'hosted-env/provider': 'boat'})
    _bash_syntax_ok(s)
    assert s.count(tok) == 1 and 'set -x' not in s and 'rm -f -- "$0"' in s.splitlines()[3]
    lines = s.splitlines()
    assert lines.index('umask 077') < lines.index('exec >>/var/log/hosted-env/boot.log 2>&1') < next(
        i for i, x in enumerate(lines) if x.startswith('umask 022'))   # root-only log; apt must read the keyring
    assert f'INSTALL_K3S_VERSION={K3S_VERSION}' in s and "K3S_URL=https://cp.example:6443" in s
    assert '--node-label hosted-env/provider=boat' in s and 'apt-get install -y -qq runsc' in s
    assert "runtimes.runsc]" in s and 'ufw allow from 10.42.0.0/16' in s and 'flannel.json' not in s
    assert "runtimes.runsc-nonet]" in s and 'ConfigPath = "/etc/hosted-env/runsc-nonet.toml"' in s
    assert 'ConfigPath = "/etc/hosted-env/runsc.toml"' in s                      # the networked runsc has a config too
    assert '[runsc_config]\n  network = "none"\n  signal-unkillable-policy = "linux"\nEOF' in s   # nonet: offline + policy
    assert '[runsc_config]\n  signal-unkillable-policy = "linux"\nEOF' in s       # runsc: PID 1 unkillable by the model
    assert s.count('signal-unkillable-policy = "linux"') == 2


def test_boot_script_remapped_wireguard_and_no_gvisor():
    import json as _json

    from hosted_env.infra.providers.machine.bootstrap import render_boot_script
    s = render_boot_script(server_url='https://160.0.0.1:20004', join_token='t', node_name='hosted-env-tg1',
                           wireguard_port=20000, wireguard_forwarded_port=51820, gvisor=False)
    _bash_syntax_ok(s)
    conf = _json.loads(s.split("flannel.json <<'EOF'\n", 1)[1].split('\nEOF', 1)[0])
    assert conf['Backend'] == {'Type': 'wireguard', 'ListenPort': 20000, 'PersistentKeepaliveInterval': 25}
    assert '--dport 51820 -j REDIRECT --to-ports 20000' in s and 'ufw allow 51820/udp' in s
    assert '--flannel-conf /etc/rancher/k3s/flannel.json' in s and '--node-name hosted-env-tg1' in s
    assert 'runsc' not in s


def test_server_boot_script_targon_shape():
    from hosted_env.infra.providers.machine.bootstrap import render_server_boot_script
    tok = 'K10srv::server:s3cr3t'
    s = render_server_boot_script(join_token=tok, public_ip='160.0.0.1', api_port=20004, hairpin=True,
                                  wireguard_port=20000, wireguard_forwarded_port=51820, node_name='hosted-env-cp1')
    _bash_syntax_ok(s)
    assert s.count(tok) == 1 and 'set -x' not in s and 'rm -f -- "$0"' in s.splitlines()[3]
    assert '--advertise-address 160.0.0.1 --advertise-port 20004 --tls-san 160.0.0.1' in s
    assert '--node-external-ip 160.0.0.1' in s and 'control-plane=true:NoSchedule' in s
    assert 'ExecStartPre=-/usr/sbin/ip addr add 160.0.0.1/32 dev lo' in s
    assert 'TCP-LISTEN:20004,bind=160.0.0.1,reuseaddr,fork TCP:127.0.0.1:6443' in s
    assert "ExecStartPre=/bin/sh -c 'iptables -t nat -C PREROUTING -p udp --dport 51820" in s
    assert s.index('hosted-env-net.service') < s.index('get.k3s.io | INSTALL')    # hairpin before the server
    assert 'ufw allow 20004/tcp' in s and 'ufw allow 6443/tcp' in s and 'ufw allow 20000/udp' in s
    assert 'kind: RuntimeClass' in s and 'handler: runsc' in s and 'name: hosted-env' in s
    assert 'name: gvisor-nonet\nhandler: runsc-nonet' in s
    assert 'K3S_URL' not in s


def test_server_boot_script_plain_and_validation():
    from hosted_env.infra.providers.machine.bootstrap import render_server_boot_script
    s = render_server_boot_script(join_token='t', public_ip='2001:db8::1', schedulable=True, gvisor=False)
    _bash_syntax_ok(s)
    assert 'socat' not in s and 'NoSchedule' not in s and 'RuntimeClass' not in s and 'flannel.json' not in s
    assert '--advertise-port 6443' in s
    with pytest.raises(ValueError):
        render_server_boot_script(join_token='t', public_ip='1.2.3.4; rm -rf ~')
    with pytest.raises(ValueError):
        render_server_boot_script(join_token='t', public_ip='1.2.3.4', namespace='Bad_NS')


def test_boot_script_rejects_injection():
    from hosted_env.infra.providers.machine.bootstrap import render_boot_script
    with pytest.raises(ValueError):
        render_boot_script(server_url='https://x; rm -rf ~', join_token='t')
    with pytest.raises(ValueError):
        render_boot_script(server_url='https://x', join_token='t', labels={'a b': 'c'})
    s = render_boot_script(server_url='https://x:6443', join_token="t'; echo pwned; '", gvisor=False)
    _bash_syntax_ok(s)
    assert "K3S_TOKEN='t'\"'\"'; echo pwned; '\"'\"''" in s            # quoted, never executed as code
