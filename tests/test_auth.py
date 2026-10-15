"""auth.py: request signing / verification / replay protection. Ported from the identity part of EXP-015
test_gateway_auth.py."""
import pytest
from nacl.signing import SigningKey

from hosted_env.auth import (MINER_DOMAIN, WORKER_DOMAIN, AuthError, RequestVerifier, issue_token, sign_request,
                             verify_token)

NOW = 1_800_000_000.0


@pytest.fixture
def env():
    miners = {f'5Hotkey{i}': SigningKey.generate() for i in range(3)}
    seen = set()

    def remember(reqid, now):
        if reqid in seen:
            return False
        seen.add(reqid)
        return True

    v = RequestVerifier(MINER_DOMAIN, lambda hk: miners[hk].verify_key if hk in miners else None, remember)
    return v, miners


def test_valid_signature(env):
    v, miners = env
    body = b'{"task":"t1"}'
    h = sign_request(miners['5Hotkey0'], 'POST', '/open', '5Hotkey0', body, ts=int(NOW))
    assert v.verify('POST', '/open', h, body, now=NOW) == '5Hotkey0'


@pytest.mark.parametrize('mutate,code', [
    (lambda h, b: (h, b + b' '), 'bad_signature'),                          # body modified
    (lambda h, b: ({**h, 'X-Timestamp': str(int(NOW) - 61)}, b), 'stale_timestamp'),
    (lambda h, b: ({**h, 'X-Identity': '5NotRegistered'}, b), 'not_eligible'),
    (lambda h, b: ({**h, 'X-Identity': '5Hotkey1'}, b), 'bad_signature'),   # impersonate another registered hotkey
    (lambda h, b: ({k: x for k, x in h.items() if k != 'X-Signature'}, b), 'malformed_auth_headers'),
    (lambda h, b: ({**h, 'X-Request-Id': 'short'}, b), 'malformed_auth_headers'),
])
def test_rejects(env, mutate, code):
    v, miners = env
    body = b'{"task":"t1"}'
    h, b = mutate(sign_request(miners['5Hotkey0'], 'POST', '/open', '5Hotkey0', body, ts=int(NOW)), body)
    with pytest.raises(AuthError) as e:
        v.verify('POST', '/open', h, b, now=NOW)
    assert e.value.code == code


def test_path_and_domain_are_signed(env):
    v, miners = env
    h = sign_request(miners['5Hotkey0'], 'POST', '/open', '5Hotkey0', b'', ts=int(NOW))
    with pytest.raises(AuthError, match='bad_signature'):
        v.verify('POST', '/close', h, b'', now=NOW)
    h = sign_request(miners['5Hotkey0'], 'POST', '/open', '5Hotkey0', b'', ts=int(NOW), domain=WORKER_DOMAIN)
    with pytest.raises(AuthError, match='bad_signature'):                    # a worker-domain signature cannot be used as a miner request
        v.verify('POST', '/open', h, b'', now=NOW)


def test_replay_rejected(env):
    v, miners = env
    h = sign_request(miners['5Hotkey0'], 'POST', '/open', '5Hotkey0', b'x', ts=int(NOW))
    v.verify('POST', '/open', h, b'x', now=NOW)
    with pytest.raises(AuthError, match='replayed_request'):
        v.verify('POST', '/open', h, b'x', now=NOW + 1)


def test_token_tamper():
    k = SigningKey.generate()
    tok = issue_token(k, {'sid': 's', 'exp': int(NOW) + 10})
    assert verify_token(k.verify_key, tok, now=NOW)['sid'] == 's'
    p, sig = tok.split('.')
    forged = p[:-2] + ('AA' if p[-2:] != 'AA' else 'BB') + '.' + sig
    for bad in (forged, 'garbage', tok + 'x', ''):
        with pytest.raises(AuthError):
            verify_token(k.verify_key, bad, now=NOW)
