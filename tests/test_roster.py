"""chain.py (the SN120 registration rule, the synced roster file) and gateway Roster (allowlist + chain, hot reload)."""
import base64
import hashlib
import json
import os

import pytest
from nacl.signing import SigningKey

from hosted_env.chain import (OWNER, activation_message, decode_commitment, registrations, ss58_public_key,
                              sync_once, write_roster)
from hosted_env.gateway.app import Roster

B58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'


def ss58(pub: bytes, prefix: int = 42) -> str:
    body = bytes([prefix]) + pub
    raw = body + hashlib.blake2b(b'SS58PRE' + body, digest_size=64).digest()[:2]
    n, s = int.from_bytes(raw, 'big'), ''
    while n:
        n, r = divmod(n, 58)
        s = B58[r] + s
    return '1' * (len(raw) - len(raw.lstrip(b'\0'))) + s


def scale_str(text: str) -> str:
    """A commitment the way bittensor returns it: SCALE compact length + UTF-8, as a latin1 str."""
    b = text.encode()
    n = len(b)
    head = bytes([n << 2]) if n < 64 else ((n << 2) | 1).to_bytes(2, 'little')
    return (head + b).decode('latin1')


def miner():
    sk = SigningKey.generate()
    return sk, ss58(bytes(sk.verify_key))


def activation(sk, hotkey, netuid=120):
    sig = base64.urlsafe_b64encode(sk.sign(activation_message(hotkey, netuid)).signature).decode().rstrip('=')
    return scale_str(f'affine2|activate|{hotkey}|{sig}')


class FakeChain:
    def __init__(self, uids, keys, commits, block=9_000_000):
        self.block, self.calls = block, []
        self.maps = {('SubtensorModule', 'Uids'): uids, ('SubtensorModule', 'Keys'): keys,
                     ('Commitments', 'RevealedCommitments'): commits}

    def query_map(self, path, params, block=None):
        self.calls.append((path, params, block))
        return iter(self.maps[path])


def test_ss58_and_commitment_decoding():
    sk, hk = miner()
    assert ss58_public_key(hk) == bytes(sk.verify_key)
    assert len(ss58_public_key(OWNER)) == 32
    with pytest.raises(ValueError, match='checksum'):
        ss58_public_key(hk[:-1] + ('1' if hk[-1] != '1' else '2'))
    with pytest.raises(ValueError):
        ss58_public_key('5Fake0OIl')                                    # not base58
    assert decode_commitment(scale_str('short')) == 'short'
    long = 'x' * 150
    assert decode_commitment(scale_str(long)) == long
    assert decode_commitment('0x' + (bytes([5 << 2]) + b'hello').hex()) == 'hello'
    with pytest.raises(ValueError, match='mismatch'):
        decode_commitment(scale_str('abc')[:-1])


def test_registrations_follow_the_upstream_rule():
    (ska, a), (skb, b), (skc, c), (skd, d), (ske, e) = (miner() for _ in range(5))
    _, other = miner()
    uids = [(a, 1), (b, 2), (c, 3), (d, 4), (OWNER, 0)]
    keys = [(1, a), (2, b), (3, c), (4, other), (0, OWNER)]               # uid 4 belongs to someone else now
    commits = [
        (a, [(scale_str('{"model": "x/y"}'), 10), (activation(ska, a), 11)]),   # a model commitment is ignored
        (b, [(activation(ska, b), 12)]),                                         # signed by the wrong key
        (c, [(scale_str(f'affine2|activate|{a}|x'), 13)]),                       # names another hotkey
        (d, [(activation(skd, d), 14)]),                                         # UID map disagrees
        (e, [(activation(ske, e), 15)]),                                         # no UID at all
        (OWNER, [(scale_str('affine2|activate|owner|x'), 16)]),
    ]
    ch = FakeChain(uids, keys, commits)
    block, regs = registrations(ch)
    assert block == 9_000_000 and all(call[1:] == ([120], 9_000_000) for call in ch.calls)   # one fixed block
    assert regs == {a: {'uid': 1, 'public_key': bytes(ska.verify_key).hex(), 'activate_block': 11}}
    with pytest.raises(ValueError, match='ambiguous'):
        registrations(FakeChain([(a, 1), (b, 1)], keys, []))
    with pytest.raises(ValueError, match='ambiguous'):
        registrations(FakeChain(uids, [(1, a), (1, b)], []))
    assert registrations(FakeChain(uids, keys, commits), netuid=7)[1] == {}   # signed for 120, not 7


def test_write_roster_is_atomic_and_refuses_a_collapse(tmp_path):
    out = str(tmp_path / 'roster.chain.json')
    hks = {f'h{i}': {'uid': i, 'public_key': '00' * 32, 'activate_block': 1} for i in range(20)}
    assert write_roster(out, 5, hks, now=100.0) == {'block': 5, 'hotkeys': 20, 'previous': 0}
    doc = json.load(open(out))
    assert (doc['source'], doc['block'], doc['synced_at'], len(doc['hotkeys'])) == ('chain', 5, 100.0, 20)
    assert oct(os.stat(out).st_mode & 0o777) == '0o644' and os.listdir(tmp_path) == ['roster.chain.json']
    with pytest.raises(ValueError, match='drop from 20 to 3'):
        write_roster(out, 6, dict(list(hks.items())[:3]))
    assert json.load(open(out))['block'] == 5                            # the old roster stays
    assert write_roster(out, 7, dict(list(hks.items())[:12]))['hotkeys'] == 12   # a normal shrink is fine
    sk, hk = miner()
    ch = FakeChain([(hk, 9)], [(9, hk)], [(hk, [(activation(sk, hk), 3)])])
    res = sync_once(str(tmp_path / 'fresh.json'), chain=ch)
    assert res['hotkeys'] == 1 and list(json.load(open(tmp_path / 'fresh.json'))['hotkeys']) == [hk]
    with pytest.raises(ValueError, match='drop from 12 to 1'):
        sync_once(out, chain=FakeChain([(hk, 9)], [(9, hk)], [(hk, [(activation(sk, hk), 3)])]))


def _bump(path):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def test_roster_merges_allowlist_and_chain_and_hot_reloads(tmp_path):
    test_sk, chain_sk, other_sk = SigningKey.generate(), SigningKey.generate(), SigningKey.generate()
    allow, chain = tmp_path / 'roster.json', tmp_path / 'roster.chain.json'
    allow.write_text(json.dumps({'epoch': 'e1', 'hotkeys': {'5Test': {'uid': 0,
                                                                      'public_key': bytes(test_sk.verify_key).hex()}}}))
    r = Roster(str(allow), str(chain))                                   # no chain file yet: allowlist only
    assert r.current_epoch() == 'e1' and bytes(r.key('5Test')) == bytes(test_sk.verify_key) and r.key('5Reg') is None
    assert r.status() == {'allowlist_hotkeys': 1, 'chain_hotkeys': 0, 'chain_block': None, 'chain_synced_at': None}
    write_roster(str(chain), 42, {'5Reg': {'uid': 3, 'public_key': bytes(chain_sk.verify_key).hex(),
                                           'activate_block': 1}}, now=1.0)
    assert bytes(r.key('5Reg')) == bytes(chain_sk.verify_key)           # picked up without a restart
    assert r.status()['chain_block'] == 42 and r.key('5Test') is not None
    # hot-add a test key and try to re-key the registered hotkey from the allowlist
    doc = json.loads(allow.read_text())
    doc['hotkeys']['5Test2'] = {'uid': 1, 'public_key': bytes(other_sk.verify_key).hex()}
    doc['hotkeys']['5Reg'] = {'uid': 3, 'public_key': bytes(other_sk.verify_key).hex()}
    allow.write_text(json.dumps(doc))
    _bump(allow)
    assert bytes(r.key('5Test2')) == bytes(other_sk.verify_key)
    assert bytes(r.key('5Reg')) == bytes(chain_sk.verify_key)           # the chain key wins
    allow.write_text('{"epoch": "e1", "hotkeys": {broken')               # a bad edit keeps the last good copy
    _bump(allow)
    assert r.key('5Test2') is not None and r.current_epoch() == 'e1'
    chain.unlink()                                                       # sync output removed: chain hotkeys gone
    assert r.status()['chain_hotkeys'] == 0 and r.key('5Test') is not None
    assert bytes(r.key('5Reg')) == bytes(other_sk.verify_key)           # only the operator's allowlist entry is left
    with pytest.raises(FileNotFoundError):
        Roster(str(tmp_path / 'missing.json'))


def test_info_reports_the_roster(tmp_path):
    from fastapi.testclient import TestClient
    from test_nodes import node_store

    from hosted_env.gateway.app import create_gateway_app
    allow = tmp_path / 'roster.json'
    allow.write_text(json.dumps({'epoch': 'e1', 'hotkeys': {}}))
    write_roster(str(tmp_path / 'c.json'), 7, {}, now=5.0)
    app = create_gateway_app(node_store(tmp_path), Roster(str(allow), str(tmp_path / 'c.json')), {},
                             SigningKey.generate().verify_key)
    with TestClient(app) as c:
        assert c.get('/v1/info').json()['roster'] == {'allowlist_hotkeys': 0, 'chain_hotkeys': 0, 'chain_block': 7,
                                                      'chain_synced_at': 5.0}
