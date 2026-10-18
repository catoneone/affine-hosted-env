"""End to end: gateway and worker each run a uvicorn (127.0.0.1); the miner client opens a task, runs the
whole group of 8 and finishes it; the validator checks the attachment with verify_group. The sandbox is
LocalSandbox, the model is a scripted fake model, the tokenizer is CharRenderer."""
import copy
import json
import socket
import threading
import time

import httpx
import pytest
from conftest import FIX, make_swe_tasks
from nacl.signing import SigningKey

from hosted_env.cli import SMOKE_REPLIES
from hosted_env.client import ApiError, GatewayClient, ScriptedSampler, run_task
from hosted_env.common import key_id
from hosted_env.coordinator.store import Limits, Store
from hosted_env.envs import SweEnv
from hosted_env.executor.app import create_worker_app
from hosted_env.executor.core import EnvRuntime, Worker
from hosted_env.executor.gateway_client import WorkerGatewayClient
from hosted_env.gateway.app import Roster, create_gateway_app
from hosted_env.receipt import ReceiptError
from hosted_env.render import CharRenderer
from hosted_env.verify import verify_group

uvicorn = pytest.importorskip('uvicorn')
EPOCH = 'e000001'


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, app, port=None):
        self.port = port or free_port()
        self.url = f'http://127.0.0.1:{self.port}'
        self.srv = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=self.port, log_level='warning'))
        self.th = threading.Thread(target=self.srv.run, daemon=True)

    def __enter__(self):
        self.th.start()
        for _ in range(200):
            if self.srv.started:
                return self
            time.sleep(0.02)
        raise RuntimeError('server did not start')

    def __exit__(self, *a):
        self.srv.should_exit = True
        self.th.join(10)


@pytest.fixture
def stack(tmp_path):
    path, ids = make_swe_tasks(tmp_path, n=2)
    cfg = {'kind': 'mimo_swe', 'tasks': path, 'cleanup': False, 'grader_user': 'root', 'require_image_digest': False}
    gkey, wkey = SigningKey.generate(), SigningKey.generate()
    miners = {'5H0': SigningKey.generate(), '5H1': SigningKey.generate()}
    roster = tmp_path / 'roster.json'
    roster.write_text(json.dumps({'epoch': EPOCH, 'hotkeys': {hk: {'uid': i, 'public_key': bytes(k.verify_key).hex()}
                                                              for i, (hk, k) in enumerate(miners.items())}}))
    store = Store(str(tmp_path / 'g.db'), {'swe': SweEnv('swe', cfg)}, gkey, Limits())
    gw_app = create_gateway_app(store, Roster(str(roster)), {'w1': wkey.verify_key}, gkey.verify_key)
    with Server(gw_app) as gw:
        wport = free_port()
        rt = EnvRuntime(env=SweEnv('swe', {**cfg, 'tasks': None}), renderer=CharRenderer(), sandbox='local',
                        max_context=200_000, max_turn_tokens=20_000, max_steps=10)
        worker = Worker(worker_id='w1', key=wkey, gateway=WorkerGatewayClient(gw.url, 'w1', wkey),
                        gateway_pub=gkey.verify_key, runtimes={'swe': rt}, public_url=f'http://127.0.0.1:{wport}',
                        capacity=16, scratch=str(tmp_path), log_dir=str(tmp_path / 'logs'))
        (tmp_path / 'logs').mkdir()
        with Server(create_worker_app(worker), port=wport) as wsrv:
            yield {'gw': gw, 'worker': wsrv, 'miners': miners, 'wkey': wkey, 'store': store, 'tmp': tmp_path}


def trusted(stack):
    info = httpx.get(stack['gw'].url + '/v1/info').json()
    from hosted_env.common import verify_key_from_hex
    return info, {w['key_id']: verify_key_from_hex(w['public_key']) for w in info['workers'].values()}


def gw_key(info):
    """The key the gateway signs group seals with, as /v1/info publishes it."""
    from hosted_env.common import verify_key_from_hex
    return verify_key_from_hex(info['gateway_public_key'])


def test_group_end_to_end(stack):
    r = CharRenderer()
    gw = GatewayClient(stack['gw'].url, '5H0', stack['miners']['5H0'])
    sampler = ScriptedSampler(r.encode, r.turn_end_id, {n: [FIX, 'Fixed.'] if n % 2 == 0 else ['Nothing to do.']
                                                        for n in range(8)})
    res = run_task(gw, EPOCH, 'swe', sampler)
    outs = [(x['nonce'], x['outcome'], x['reward']) for x in res['rollouts']]
    assert outs == [(n, 'completed', 1.0 if n % 2 == 0 else 0.0) for n in range(8)]
    # this task has no expected tests: the receipt says so, and the reward rests on the exit code
    from hosted_env.common import unb64u
    assert all(json.loads(unb64u(x['receipt']['payload']))['expected_tests_sha256'] == '' for x in res['rollouts'])

    info, keys = trusted(stack)
    assert key_id(stack['wkey'].verify_key) in keys and info['envs']['swe']['group_size'] == 8
    att = json.loads(json.dumps(res['attachment']))                    # round-trip through JSON, same as reading it back after upload
    v = verify_group(att, keys, gateway_key=gw_key(info), epoch=EPOCH, hotkey='5H0', env_id='swe', task=res['task']['task'])
    assert [c['reward'] for c in v['claims']] == [1.0, 0.0] * 4
    assert all(sum(m) > 0 for m in v['loss_masks'])
    c0 = v['claims'][0]                                                # receipt v2 context, set by the worker
    assert (c0['grader'], c0['grader_user'], c0['renderer']) == ('isolated-v3', 'root', 'char/v1/enable_thinking=false')
    assert len(c0['model_patch_sha256']) == len(c0['test_patch_sha256']) == len(c0['tokenizer_sha256']) == 64
    assert c0['image'] and c0['verifier_returncode'] == 0 and c0['started_at_ms'] <= c0['issued_at_ms']
    assert len(c0['start_tree']) == 40
    assert b'a + b' in v['model_patches'][0] and v['model_patches'][1] == b''   # the fix, then the no-op rollout

    bad = copy.deepcopy(att)
    seg = bad['rollouts'][0]['segments'][2]
    assert seg['src'] == 'T'
    seg['ids'][3] = r.encode('!')[0]                                   # tamper with tool output
    with pytest.raises(ReceiptError, match='trajectory_mismatch'):
        verify_group(bad, keys, gateway_key=gw_key(info), epoch=EPOCH, hotkey='5H0', env_id='swe', task=res['task']['task'])
    with pytest.raises(ReceiptError, match='identity_mismatch'):        # someone else claims this group
        verify_group(att, keys, gateway_key=gw_key(info), epoch=EPOCH, hotkey='5H1', env_id='swe', task=res['task']['task'])

    # a qualified task is not assigned again this epoch; the remaining one goes to someone else
    gw1 = GatewayClient(stack['gw'].url, '5H1', stack['miners']['5H1'])
    t2 = gw1.open_task(EPOCH, 'swe')
    assert t2['task'] != res['task']['task']
    log = (stack['tmp'] / 'logs' / 'exec-w1.jsonl').read_text().splitlines()
    assert sum(json.loads(x)['event'] == 'finish' for x in log) == 8
    fin = [json.loads(x) for x in log if json.loads(x)['event'] == 'finish']   # how each rollout was graded
    assert all(f['grader'] == 'isolated-v3' and f['grader_user'] == 'root' for f in fin)
    # no expected tests on this task: no second-check verdict is logged; the log never carries the full revert list
    assert all('second_check' not in f and 'reverted_paths' not in f for f in fin)
def test_smoke_replies_end_to_end(stack):
    """The EXP-014 smoke script: timeout, malformed call, unknown tool, truncation, submit, over the full HTTP path."""
    r = CharRenderer()
    gw = GatewayClient(stack['gw'].url, '5H0', stack['miners']['5H0'])
    res = run_task(gw, EPOCH, 'swe', ScriptedSampler(r.encode, r.turn_end_id, SMOKE_REPLIES))
    assert {(x['outcome'], x['detail'], x['steps']) for x in res['rollouts']} == {('completed', 'submitted', 6)}
    tool_texts = [r.decode(s['ids']) for s in res['rollouts'][0]['segments'] if s['src'] == 'T']
    assert 'Command timed out after 2s' in tool_texts[1]
    assert tool_texts[2].startswith('<|im_start|>tool\nError: Found 1 <tool_call>')
    assert "Unknown tool 'python'" in tool_texts[3]
    assert '[TRUNCATION WARNING]' in tool_texts[4]
    info, keys = trusted(stack)
    with pytest.raises(ReceiptError, match='group_gate_failed'):         # all completed but none correct: group not qualified
        verify_group(res['attachment'], keys, gateway_key=gw_key(info), epoch=EPOCH, hotkey='5H0', env_id='swe',
                     task=res['task']['task'])


def test_auth_and_quota_errors_over_http(stack):
    url = stack['gw'].url
    stranger = GatewayClient(url, '5Stranger', SigningKey.generate())
    with pytest.raises(ApiError) as e:
        stranger.open_task(EPOCH, 'swe')
    assert (e.value.status, e.value.code) == (403, 'not_eligible')
    impostor = GatewayClient(url, '5H0', SigningKey.generate())        # impersonating a registered hotkey
    with pytest.raises(ApiError) as e:
        impostor.open_task(EPOCH, 'swe')
    assert (e.value.status, e.value.code) == (401, 'bad_signature')
    gw = GatewayClient(url, '5H0', stack['miners']['5H0'])
    with pytest.raises(ApiError, match='wrong_epoch'):
        gw.open_task('e000999', 'swe')
    t = gw.open_task(EPOCH, 'swe')
    gw.open_sessions(EPOCH, t['task_id'], [0])
    with pytest.raises(ApiError, match='nonce_used'):
        gw.open_sessions(EPOCH, t['task_id'], [0])
    r = httpx.post(stack['worker'].url + '/v1/start', json={}, headers={'Authorization': 'Bearer nope'})
    assert (r.status_code, r.json()['error']) == (401, 'bad_token')


# ---------------------------------------------------------------- single process (architecture v2)
def test_single_process_group_and_data_plane_limit(tmp_path):
    """gateway + coordinator + executor in one process: the miner only ever talks to the gateway, which also
    serves /v1/start and /v1/turn and rate limits them by the session token's hotkey."""
    from hosted_env.executor.gateway_client import InProcessGatewayClient
    from hosted_env.gateway.ratelimit import MemoryBuckets, Rate, RateLimiter, RateLimits

    class Counting(MemoryBuckets):
        limited = 0

        def check(self, scope, key, rate, cost=1.0):
            wait = super().check(scope, key, rate, cost)
            if wait > 0 and scope == 'hotkey_data':
                Counting.limited += 1
            return wait

    path, ids = make_swe_tasks(tmp_path, n=1)
    cfg = {'kind': 'mimo_swe', 'tasks': path, 'cleanup': False, 'grader_user': 'root', 'require_image_digest': False}
    gkey, wkey, miner = SigningKey.generate(), SigningKey.generate(), SigningKey.generate()
    roster = tmp_path / 'roster.json'
    roster.write_text(json.dumps({'epoch': EPOCH, 'hotkeys': {'5H0': {'uid': 0,
                                                                       'public_key': bytes(miner.verify_key).hex()}}}))
    store = Store(str(tmp_path / 'g.db'), {'swe': SweEnv('swe', cfg)}, gkey, Limits())
    port = free_port()
    rt = EnvRuntime(env=SweEnv('swe', {**cfg, 'tasks': None}), renderer=CharRenderer(), sandbox='local',
                    max_context=200_000, max_turn_tokens=20_000, max_steps=10)
    worker = Worker(worker_id='local', key=wkey, gateway=InProcessGatewayClient(store, 'local'),
                    gateway_pub=gkey.verify_key, runtimes={'swe': rt}, public_url=f'http://127.0.0.1:{port}',
                    capacity=16, scratch=str(tmp_path))
    limiter = RateLimiter(RateLimits(hotkey_data=Rate(10, 3)), Counting())   # small: the group must hit 429 and retry
    app = create_gateway_app(store, Roster(str(roster)), {'local': wkey.verify_key}, gkey.verify_key,
                             limiter=limiter, executor=worker)
    with Server(app, port=port) as gw:
        r = CharRenderer()
        sampler = ScriptedSampler(r.encode, r.turn_end_id, {n: [FIX, 'Fixed.'] if n % 2 == 0 else ['Nothing to do.']
                                                            for n in range(8)})
        res = run_task(GatewayClient(gw.url, '5H0', miner), EPOCH, 'swe', sampler)
        assert res['task']['worker_url'] == gw.url                  # the data plane goes through the gateway
        assert [x['outcome'] for x in res['rollouts']] == ['completed'] * 8
        assert Counting.limited > 0                                 # 429s happened and the client retried them
        info = httpx.get(gw.url + '/v1/info').json()
        from hosted_env.common import verify_key_from_hex
        keys = {w['key_id']: verify_key_from_hex(w['public_key']) for w in info['workers'].values()}
        v = verify_group(json.loads(json.dumps(res['attachment'])), keys, gateway_key=gw_key(info), epoch=EPOCH,
                         hotkey='5H0', env_id='swe', task=res['task']['task'])
        assert [c['reward'] for c in v['claims']] == [1.0, 0.0] * 4
        assert httpx.get(gw.url + '/v1/health').json()['worker_id'] == 'local'
        bad = httpx.post(gw.url + '/v1/start', json={}, headers={'Authorization': 'Bearer nope'})
        assert (bad.status_code, bad.json()['error']) == (401, 'bad_token')


def test_receipt_context_binds_expected_tests():
    """The receipt carries the digest of the expected tests the grader was handed, so an offline re-grade can
    check it used the same list; an ungraded rollout (gi == {}) carries ''."""
    from types import SimpleNamespace

    from hosted_env.executor.core import Worker
    from hosted_env.receipt import expected_tests_digest
    rend = SimpleNamespace(model_id='m', model_revision='r', identity={'tokenizer_sha256': '', 'chat_template_sha256': '',
                                                                     'renderer': 'char'})
    s = SimpleNamespace(rt=SimpleNamespace(renderer=rend, sandbox='local', sandbox_opts={}), envsess=None,
                        started=1.0)
    me = SimpleNamespace(build_version='t')
    gi = {'test_patch': 'p', 'test_command': 'c', 'report_kind': 'junit',
          'fail_to_pass': ['tests/test_a.py::t1'], 'pass_to_pass': ['tests/test_a.py::t0']}
    ctx = Worker._receipt_context(me, s, gi, {'grader': 'isolated-v3'})
    assert ctx['expected_tests_sha256'] == expected_tests_digest(['tests/test_a.py::t1'], ['tests/test_a.py::t0'],
                                                                'junit') != ''
    assert Worker._receipt_context(me, s, {}, {})['expected_tests_sha256'] == ''


def test_sandbox_limits_names_what_was_enforced():
    """The receipt's sandbox_limits fills each provider's defaults, so it names the limits actually enforced."""
    from types import SimpleNamespace

    from hosted_env.executor.core import _sandbox_limits
    from hosted_env.providers.sandbox.k8s import DEFAULT_DISK

    def rt(kind, **o):
        return SimpleNamespace(sandbox=kind, sandbox_opts=o)
    assert _sandbox_limits(rt('k8s')) == f'k8s:cpus=1;memory=2g;disk={DEFAULT_DISK}'
    assert _sandbox_limits(rt('k8s', cpus='2', memory='4g', disk='20g')) == 'k8s:cpus=2;memory=4g;disk=20g'
    assert _sandbox_limits(rt('k8s', disk='0')) == _sandbox_limits(rt('k8s', disk=None)) == 'k8s:cpus=1;memory=2g;disk=none'
    assert _sandbox_limits(rt('docker', memory='8g', disk='10g')) == 'docker:cpus=1;memory=8g;disk=unenforced'
    assert _sandbox_limits(rt('local', memory='8g')) == 'local'
