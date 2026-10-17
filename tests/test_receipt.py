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
CTX = {'image': 'docker.io/xiaomimimo/mimo-v2.6-rl-oss@sha256:' + 'c' * 64, 'start_tree': '9' * 40,
       'test_patch_sha256': 'a' * 64,
       'test_command_sha256': 'b' * 64, 'grader': 'isolated-v3', 'grader_parsers': 'py3.12:tomllib+json+configparser',
       'grader_user': 'nobody',
       'model_patch_sha256': 'd' * 64, 'verifier_returncode': 0, 'model_id': 'XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B',
       'model_revision': '2367e865', 'tokenizer_sha256': 'e' * 64, 'chat_template_sha256': 'f' * 64,
       'renderer': 'hf/transformers-4.57.1', 'executor_version': '0.1.0+g07c593529e5c',
       'started_at_ms': 1_800_000_000_000}


def _sign(wk, segs=SEGS, outcome='completed', reward=1.0, kid='w1', ctx=CTX, **ident):
    d, n = trajectory_sha256(segs)
    return sign_receipt(wk, kid, trajectory_sha256=d, n_tokens=n, outcome=outcome, reward=reward, context=ctx,
                        issued_at_ms=1_800_000_123_456, **{**ID, **ident})


@pytest.fixture
def setup():
    wk = SigningKey.generate()
    return wk, {'w1': wk.verify_key}, _sign(wk)


def test_roundtrip(setup):
    _, keys, r = setup
    c = verify_receipt(r, SEGS, keys, **ID)
    assert (c['outcome'], c['reward'], c['n_tokens'], c['v']) == ('completed', 1.0, 14, 2)
    assert {k: c[k] for k in CTX} == CTX and c['issued_at_ms'] == 1_800_000_123_456


def test_context_is_complete_and_typed():
    """The worker cannot sign a v2 receipt that leaves out what the reward depends on, or adds to it."""
    wk = SigningKey.generate()
    for bad in ({k: v for k, v in CTX.items() if k != 'image'}, {**CTX, 'extra': 'x'},
                {**CTX, 'verifier_returncode': '0'}, {**CTX, 'started_at_ms': True},
                {**CTX, 'tokenizer_sha256': 'E' * 64}, {**CTX, 'model_patch_sha256': 'abc'},
                {**CTX, 'start_tree': '9' * 64}):
        with pytest.raises(ValueError, match='bad receipt context'):
            _sign(wk, ctx=bad)
    _sign(wk, ctx={**CTX, 'verifier_returncode': None, 'model_patch_sha256': ''})   # an ungraded rollout


def test_expect_pins_context(setup):
    """The verifier pins what it knows independently: the task's image and test patch, its own tokenizer."""
    _, keys, r = setup
    verify_receipt(r, SEGS, keys, **ID, expect={'image': CTX['image'], 'tokenizer_sha256': 'e' * 64})
    for k, v in (('image', 'docker.io/x/y@sha256:' + '0' * 64), ('test_patch_sha256', '0' * 64),
                 ('tokenizer_sha256', '1' * 64), ('grader', 'isolated-v1')):
        with pytest.raises(ReceiptError, match='context_mismatch'):
            verify_receipt(r, SEGS, keys, **ID, expect={k: v})
    with pytest.raises(ReceiptError, match='context_mismatch'):              # type-strict: 0 is not False
        verify_receipt(r, SEGS, keys, **ID, expect={'verifier_returncode': False})
    with pytest.raises(ValueError, match='not receipt context fields'):      # a typo must not pass silently,
        verify_receipt(r, SEGS, keys, **ID, expect={'tokenizer_sha': 'e' * 64})
    with pytest.raises(ValueError, match='not receipt context fields'):      # even on a receipt that fails anyway
        verify_receipt({}, SEGS, keys, **ID, expect={'tokenizer_sha': 'e' * 64})


def test_v1_and_hand_made_payloads_are_rejected(setup):
    """A v1 payload, or a v2 payload with a context field dropped, fails even when signed by a trusted key."""
    from hosted_env.common import canonical, sign
    from hosted_env.receipt import RECEIPT_DOMAIN
    wk, keys, r = setup
    body = json.loads(unb64u(r['payload']))
    for forged, code in (({**body, 'v': 1}, 'identity_mismatch'),
                         ({k: v for k, v in body.items() if k != 'grader'}, 'malformed_receipt'),
                         ({**body, 'started_at_ms': body['issued_at_ms'] + 1}, 'malformed_receipt')):
        raw = canonical(forged)
        with pytest.raises(ReceiptError, match=code):
            verify_receipt({'payload': b64u(raw), 'sig': b64u(sign(wk, RECEIPT_DOMAIN, raw))}, SEGS, keys, **ID)


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
def _group(wk, outcomes, patch=b'diff --git a/calc.py b/calc.py\n'):
    import base64
    import hashlib
    rolls = []
    for n, (o, rw) in enumerate(outcomes):
        segs = [('P', [1, 2]), ('M', [3, n + 10]), ('T', [5]), ('M', [6, 7])]
        ctx = {**CTX, 'model_patch_sha256': hashlib.sha256(patch).hexdigest()}
        rolls.append({'nonce': n, 'segments': [{'src': s, 'ids': i} for s, i in segs],
                      'receipt': _sign(wk, segs, o, rw, nonce=n, ctx=ctx),
                      'model_patch': base64.b64encode(patch).decode()})
    return {'version': GROUP_ATTACHMENT_VERSION, 'env_id': ID['env_id'], 'task': ID['task'], 'rollouts': rolls,
            'seal': _seal([r['receipt'] for r in rolls])}


GW = SigningKey.generate()                                  # the gateway's key: signs group seals


def _seal(receipts, **kw):
    from hosted_env.receipt import receipt_digest, sign_group_seal
    args = dict(epoch=ID['epoch'], hotkey=ID['hotkey'], env_id=ID['env_id'], task=ID['task'], task_id='tk-1',
                group_size=len(receipts), opened_at_ms=1_800_000_000_000, finished_at_ms=1_800_000_200_000,
                receipts=[receipt_digest(r) for r in receipts])
    return sign_group_seal(GW, **{**args, **kw})


GID = dict(epoch=ID['epoch'], hotkey=ID['hotkey'], env_id=ID['env_id'], task=ID['task'], gateway_key=GW.verify_key)


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
    with pytest.raises(ReceiptError, match='seal_mismatch'):
        verify_group(swapped, keys, **GID)
    # even with a seal over the swapped receipts, each receipt still names its own nonce
    swapped['seal'] = _seal([r['receipt'] for r in sorted(swapped['rollouts'], key=lambda r: r['nonce'])])
    with pytest.raises(ReceiptError, match='identity_mismatch'):
        verify_group(swapped, keys, **GID)
    with pytest.raises(ReceiptError, match='identity_mismatch'):
        verify_group(g, keys, **{**GID, 'task': 'other'})


def test_group_carries_the_graded_patch():
    """The attachment's patch must be the one the receipt says was graded, so a verifier can re-grade it."""
    import base64
    wk = SigningKey.generate()
    keys = {'w1': wk.verify_key}
    outcomes = [('completed', 1.0), ('completed', 0.0)] + [('unresolved', 0.0)] * 6
    g = _group(wk, outcomes)
    v = verify_group(g, keys, **GID, expect={'image': CTX['image']})
    assert v['model_patches'][0] == b'diff --git a/calc.py b/calc.py\n'
    with pytest.raises(ReceiptError, match='context_mismatch'):
        verify_group(g, keys, **GID, expect={'test_command_sha256': '0' * 64})
    swapped = copy.deepcopy(g)                              # a better-looking patch than the one graded
    swapped['rollouts'][1]['model_patch'] = base64.b64encode(b'diff --git a/x b/x\n').decode()
    with pytest.raises(ReceiptError, match='model_patch_mismatch'):
        verify_group(swapped, keys, **GID)
    for bad in ('not base64!', 7, 'A' * (3_000_000)):
        broken = copy.deepcopy(g)
        broken['rollouts'][0]['model_patch'] = bad
        with pytest.raises(ReceiptError, match='malformed_group'):
            verify_group(broken, keys, **GID)
    with pytest.raises(ReceiptError, match='bad_attachment_version'):
        verify_group(dict(g, version='hosted-env-group-v1'), keys, **GID)


def test_group_seal_binds_the_receipts_and_the_gateway():
    """Receipts from another run of the same task and nonce (each validly signed by the worker) cannot be
    swapped in, and the seal must come from the gateway and name this group."""
    wk = SigningKey.generate()
    keys = {'w1': wk.verify_key}
    outcomes = [('completed', 1.0), ('completed', 0.0)] + [('unresolved', 0.0)] * 6
    g, other_run = _group(wk, outcomes), _group(wk, [('completed', 1.0)] * 8)
    v = verify_group(g, keys, **GID)
    assert v['seal']['finished_at_ms'] - v['seal']['opened_at_ms'] == 200_000 and v['seal']['task_id'] == 'tk-1'
    mixed = copy.deepcopy(g)                                 # a better rollout from another run, same nonce
    mixed['rollouts'][1] = other_run['rollouts'][1]
    with pytest.raises(ReceiptError, match='seal_mismatch'):
        verify_group(mixed, keys, **GID)
    receipts = [r['receipt'] for r in g['rollouts']]
    for seal, code in ((None, 'malformed_seal'), (dict(g['seal'], sig=other_run['seal']['sig']), 'bad_seal_signature'),
                       (_seal(receipts, hotkey='5Hotkey1'), 'seal_identity_mismatch'),
                       (_seal(receipts, opened_at_ms=1_900_000_000_000), 'malformed_seal')):
        with pytest.raises(ReceiptError, match=code):
            verify_group(dict(g, seal=seal), keys, **GID)
    with pytest.raises(ReceiptError, match='bad_seal_signature'):    # signed by a worker key, not the gateway's
        verify_group(g, keys, **{**GID, 'gateway_key': wk.verify_key})
