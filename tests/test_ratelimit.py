"""gateway/ratelimit.py and its wiring into the gateway app and the client."""
import json
import time

import pytest
from fastapi.testclient import TestClient
from nacl.signing import SigningKey

from hosted_env.auth import sign_request
from hosted_env.client import ApiError, GatewayClient
from hosted_env.coordinator.store import Store
from hosted_env.envs import DockerTaskEnv
from hosted_env.gateway.app import Roster, create_gateway_app
from hosted_env.gateway.ratelimit import MemoryBuckets, Rate, RateLimiter, RateLimitError, RateLimits

EPOCH = 'e000001'


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ---------------------------------------------------------------- buckets
def test_bucket_burst_then_refill():
    clk = Clock()
    b, rate = MemoryBuckets(clock=clk), Rate(per_sec=2, burst=3)
    assert [b.check('s', 'k', rate) for _ in range(3)] == [0, 0, 0]
    assert b.check('s', 'k', rate) == pytest.approx(0.5)        # empty: one token refills in 0.5 s
    assert b.check('s', 'other', rate) == 0                      # keys are independent
    clk.t += 0.5
    assert b.check('s', 'k', rate) == 0
    clk.t += 100
    assert [b.check('s', 'k', rate) for _ in range(3)] == [0, 0, 0]   # refill is capped at burst
    assert b.check('s', 'k', rate) > 0


def test_bucket_key_cap_drops_least_recent():
    b, rate = MemoryBuckets(max_keys=2, clock=Clock()), Rate(1, 1)
    b.check('s', 'a', rate)
    b.check('s', 'b', rate)
    b.check('s', 'c', rate)                                      # 'a' is dropped
    assert len(b._b) == 2 and ('s', 'a') not in b._b


def test_limits_from_config():
    lim = RateLimits.from_config({'hotkey_control': [1, 2], 'ip': False})
    assert lim.hotkey_control == Rate(1, 2) and lim.ip is None and lim.total == RateLimits().total
    with pytest.raises(ValueError):
        RateLimits.from_config({'nope': [1, 1]})
    with pytest.raises(ValueError):
        Rate(0, 1)


def test_limiter_scopes_and_retry_after():
    lim = RateLimiter(RateLimits(ip=None, total=None, hotkey_control=Rate(0.1, 1), hotkey_data=None),
                      MemoryBuckets(clock=Clock()))
    lim.before_auth('1.2.3.4')                                   # disabled scopes never limit
    lim.data('5H')
    lim.control('5H')
    with pytest.raises(RateLimitError) as e:
        lim.control('5H')
    assert e.value.retry_after == 10 and e.value.detail == 'hotkey_control'


# ---------------------------------------------------------------- gateway HTTP
@pytest.fixture
def gw(tmp_path):
    p = tmp_path / 't.jsonl'
    p.write_text(json.dumps({'id': 't0', 'image': 'img', 'cwd': '/w', 'problem_statement': 'p',
                             'test_command': 'true'}) + '\n')
    env = DockerTaskEnv('env', {'tasks': str(p), 'group_size': 8})
    miner = SigningKey.generate()
    roster = tmp_path / 'roster.json'
    roster.write_text(json.dumps({'epoch': EPOCH, 'hotkeys': {'5H0': {'uid': 0,
                                                                       'public_key': bytes(miner.verify_key).hex()}}}))
    gkey = SigningKey.generate()
    store = Store(str(tmp_path / 'g.db'), {'env': env}, gkey)

    def make(limits: RateLimits):
        app = create_gateway_app(store, Roster(str(roster)), {}, gkey.verify_key, limiter=RateLimiter(limits))
        return TestClient(app, base_url='http://testserver')
    return make, miner


def _finish(client, miner):
    body = json.dumps({'epoch': EPOCH, 'task_id': 'nope'}).encode()
    h = sign_request(miner, 'POST', '/v1/tasks/finish', '5H0', body)
    return client.post('/v1/tasks/finish', content=body, headers={**h, 'Content-Type': 'application/json'})


def test_gateway_limits_per_hotkey(gw):
    make, miner = gw
    with make(RateLimits(ip=None, total=None, hotkey_control=Rate(0.5, 2))) as c:
        assert [_finish(c, miner).status_code for _ in range(2)] == [404, 404]   # passes the limiter
        r = _finish(c, miner)
        assert r.status_code == 429 and r.json()['error'] == 'rate_limited' and r.headers['retry-after'] == '2'


def test_gateway_limits_per_ip_before_auth(gw):
    make, _ = gw
    with make(RateLimits(ip=Rate(0.1, 2), total=None)) as c:
        assert [c.get('/v1/info').status_code for _ in range(2)] == [200, 200]
        r = c.get('/v1/info')
        assert r.status_code == 429 and r.json()['detail'] == 'ip'
        assert c.post('/v1/internal/touch', json={}).status_code == 401           # internal routes skip it


def test_client_waits_and_retries_when_rate_limited(gw):
    make, miner = gw
    with make(RateLimits(ip=None, total=None, hotkey_control=Rate(2, 1))) as c:
        gc = GatewayClient('http://testserver', '5H0', miner, http=c)
        with pytest.raises(ApiError, match='unknown_task'):
            gc.finish_task(EPOCH, 'nope')
        t0 = time.time()
        with pytest.raises(ApiError, match='unknown_task'):                  # 429 first, then retried
            gc.finish_task(EPOCH, 'nope')
        assert time.time() - t0 >= 0.9
