"""Receipts (receipt.py) and group verification (verify.py). Ported from EXP-015 test_env_receipt.py, switched to
PyNaCl."""
import copy
import json

import pytest
from nacl.signing import SigningKey

from hosted_env.common import b64u, unb64u
from hosted_env.receipt import (ReceiptError, TrajectoryHasher, loss_mask, sign_receipt, trajectory_sha256,
                                verify_receipt)
from hosted_env.verify import GROUP_ATTACHMENT_VERSION, verify_group

ID = dict(epoch='e000001', hotkey='5Hotkey0', env_id='mimo_swe', task='002119', nonce=3)
SEGS = [('P', [151644, 8948, 198]), ('M', [40, 1184, 311, 1349]), ('T', [27, 14172, 9, 29]),
        ('M', [785, 5046, 151645])]


def _sign(wk, segs=SEGS, outcome='completed', reward=1.0, kid='w1', **ident):
    d, n = trajectory_sha256(segs)
    return sign_receipt(wk, kid, trajectory_sha256=d, n_tokens=n, outcome=outcome, reward=reward,
                        issued_at=1_800_000_000, **{**ID, **ident})


@pytest.fixture
def setup():
    wk = SigningKey.generate()
    return wk, {'w1': wk.verify_key}, _sign(wk)


def test_roundtrip(setup):
    _, keys, r = setup
    c = verify_receipt(r, SEGS, keys, **ID)
    assert (c['outcome'], c['reward'], c['n_tokens']) == ('completed', 1.0, 14)


def test_dict_segments_and_loss_mask(setup):
    _, keys, r = setup
    segs = [{'src': s, 'ids': i} for s, i in SEGS]
    verify_receipt(r, segs, keys, **ID)
    assert loss_mask(segs) == [0] * 3 + [1] * 4 + [0] * 4 + [1] * 3


def test_incremental_equals_batch():
    """The hash the worker computes by appending one segment per turn equals the one the validator recomputes
    in one go."""
    h = TrajectoryHasher()
    for src, ids in SEGS:
        h.add(src, ids)
    assert h.hexdigest() == trajectory_sha256(SEGS)[0]


@pytest.mark.parametrize('i,j,val', [(2, 1, 99999),      # alter a tool-output token (sneak the answer into context)
                                     (3, 0, 1),          # alter a model-generated token
                                     (0, 2, 0)])         # alter the task prompt
def test_token_tamper(setup, i, j, val):
    _, keys, r = setup
    segs = copy.deepcopy(SEGS)
    segs[i][1][j] = val
    with pytest.raises(ReceiptError, match='trajectory_mismatch'):
        verify_receipt(r, segs, keys, **ID)


def test_source_relabel(setup):
    """Relabeling a tool-output segment as model-generated (to get the trainer to compute loss on it) is also
    caught."""
    _, keys, r = setup
    with pytest.raises(ReceiptError):
        verify_receipt(r, [SEGS[0], SEGS[1], ('M', SEGS[2][1]), SEGS[3]], keys, **ID)
    merged = [SEGS[0], ('M', SEGS[1][1] + SEGS[2][1] + SEGS[3][1])]
    with pytest.raises(ReceiptError, match='trajectory_mismatch'):
        verify_receipt(r, merged, keys, **ID)


def test_truncated_or_extended(setup):
    _, keys, r = setup
    with pytest.raises(ReceiptError, match='trajectory_mismatch'):
        verify_receipt(r, SEGS[:-1], keys, **ID)
    with pytest.raises(ReceiptError, match='trajectory_mismatch'):
        verify_receipt(r, SEGS + [('T', [1]), ('M', [2])], keys, **ID)


@pytest.mark.parametrize('field,val', [('hotkey', '5Hotkey1'), ('task', '000860'), ('nonce', 4),
                                       ('epoch', 'e000002'), ('env_id', 'other')])
def test_identity_mismatch(setup, field, val):
    """A receipt from someone else, another task, another nonce or another epoch does not pass."""
    _, keys, r = setup
    with pytest.raises(ReceiptError, match='identity_mismatch'):
        verify_receipt(r, SEGS, keys, **{**ID, field: val})


def test_reward_tamper_and_untrusted_key(setup):
    wk, keys, r = setup
    body = json.loads(unb64u(r['payload']))
    body['reward'] = 0.0
    forged = dict(r, payload=b64u(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()))
    with pytest.raises(ReceiptError, match='bad_signature'):
        verify_receipt(forged, SEGS, keys, **ID)
    rogue = SigningKey.generate()                  # receipt signed with a key the miner generated itself
    with pytest.raises(ReceiptError, match='bad_signature'):
        verify_receipt(_sign(rogue), SEGS, keys, **ID)
    with pytest.raises(ReceiptError, match='untrusted_worker_key'):
        verify_receipt(_sign(rogue, kid='w9'), SEGS, keys, **ID)


def test_malformed(setup):
    _, keys, r = setup
    with pytest.raises(ReceiptError, match='malformed_receipt'):
        verify_receipt({'payload': '!!', 'sig': ''}, SEGS, keys, **ID)
    with pytest.raises(ReceiptError, match='malformed_receipt'):
        verify_receipt({}, SEGS, keys, **ID)
    with pytest.raises(ReceiptError, match='malformed_trajectory'):
        verify_receipt(r, [('M', [1])], keys, **ID)      # does not start with the environment prompt


def test_nonce_type_strict(setup):
    """nonce 3.0 / True and the like do not count as the same nonce."""
    _, keys, r = setup
    with pytest.raises(ReceiptError, match='identity_mismatch'):
        verify_receipt(r, SEGS, keys, **{**ID, 'nonce': 3.0})


# ---------------------------------------------------------------- group
def _group(wk, outcomes):
    rolls = []
    for n, (o, rw) in enumerate(outcomes):
        segs = [('P', [1, 2]), ('M', [3, n + 10]), ('T', [5]), ('M', [6, 7])]
        rolls.append({'nonce': n, 'segments': [{'src': s, 'ids': i} for s, i in segs],
                      'receipt': _sign(wk, segs, o, rw, nonce=n)})
    return {'version': GROUP_ATTACHMENT_VERSION, 'env_id': ID['env_id'], 'task': ID['task'], 'rollouts': rolls}


GID = dict(epoch=ID['epoch'], hotkey=ID['hotkey'], env_id=ID['env_id'], task=ID['task'])


def test_group_ok_and_gate():
    wk = SigningKey.generate()
    keys = {'w1': wk.verify_key}
    ok = _group(wk, [('completed', 1.0), ('completed', 0.0)] + [('unresolved', 0.0)] * 6)
    v = verify_group(ok, keys, **GID)
    assert [c['nonce'] for c in v['claims']] == list(range(8)) and sum(v['loss_masks'][0]) == 4
    with pytest.raises(ReceiptError, match='group_gate_failed'):         # no completed-and-wrong rollout
        verify_group(_group(wk, [('completed', 1.0)] * 8), keys, **GID)
    with pytest.raises(ReceiptError, match='group_gate_failed'):         # a correct rollout that did not complete does not count
        verify_group(_group(wk, [('unresolved', 1.0), ('completed', 0.0)] * 4), keys, **GID)
    with pytest.raises(ReceiptError, match='infra_error_in_group'):
        verify_group(_group(wk, [('completed', 1.0), ('completed', 0.0), ('infra_error', 0.0)] + [('unresolved', 0)] * 5),
                     keys, **GID)


def test_group_incomplete_or_swapped():
    wk = SigningKey.generate()
    keys = {'w1': wk.verify_key}
    g = _group(wk, [('completed', 1.0), ('completed', 0.0)] + [('unresolved', 0.0)] * 6)
    short = dict(g, rollouts=g['rollouts'][:7])
    with pytest.raises(ReceiptError, match='incomplete_group'):
        verify_group(short, keys, **GID)
    swapped = copy.deepcopy(g)                               # swap the nonce labels of two rollouts
    swapped['rollouts'][0]['nonce'], swapped['rollouts'][1]['nonce'] = 1, 0
    with pytest.raises(ReceiptError, match='identity_mismatch'):
        verify_group(swapped, keys, **GID)
    with pytest.raises(ReceiptError, match='identity_mismatch'):
        verify_group(g, keys, **{**GID, 'task': 'other'})
