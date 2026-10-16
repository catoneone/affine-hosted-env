"""Group verification: called once per group by validator admission / audit and when the trainer reads it in
(design §9, §10).

Hosted-env attachment format inside the group object (`GROUP_ATTACHMENT_VERSION`):
    {"version": "hosted-env-group-v2", "env_id": ..., "task": ...,
     "rollouts": [{"nonce": 0, "segments": [{"src": "P", "ids": [...]}, ...], "receipt": {...},
                   "model_patch": "<base64 of the patch bytes that were graded>"}, ...]}

Rules: exactly one rollout for each nonce 0..group_size-1; every receipt must pass `verify_receipt` (with the
caller's `expect`); each model_patch must hash to its receipt's model_patch_sha256 (empty when the rollout was not
graded), so a verifier can re-grade it offline with the receipt's image and test hashes; any infra_error voids
the whole group; at least one completed rollout with reward >= success_reward and at least one completed rollout
with reward < success_reward. If any check fails the whole group scores 0 (raises ReceiptError). The sampling
proof for model tokens is not checked here; the existing flow still verifies it.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
from collections.abc import Mapping

from nacl.signing import VerifyKey

from .receipt import ReceiptError, as_segments, check_expect, loss_mask, verify_receipt

GROUP_ATTACHMENT_VERSION = 'hosted-env-group-v2'
MAX_MODEL_PATCH_BYTES = 2_000_000          # the worker never grades a larger patch (envs.swe.MAX_MODEL_PATCH_BYTES)


def _model_patch(r: Mapping, claims: Mapping) -> bytes:
    b64 = r.get('model_patch', '')
    if not isinstance(b64, str) or len(b64) > (MAX_MODEL_PATCH_BYTES + 2) // 3 * 4:
        raise ReceiptError('malformed_group')
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise ReceiptError('malformed_group') from None
    want = claims.get('model_patch_sha256', '')
    if (hashlib.sha256(raw).hexdigest() != want) if want else raw:   # '' = never graded: no patch either
        raise ReceiptError('model_patch_mismatch')      # not the patch that was graded
    return raw


def verify_group(attachment: Mapping, trusted_keys: Mapping[str, VerifyKey], *, epoch: str, hotkey: str,
                 env_id: str, task: str, group_size: int = 8, success_reward: float = 1.0,
                 expect: Mapping | None = None) -> dict:
    """On success returns {'claims': [receipt contents ordered by nonce], 'loss_masks': [...], 'segments': [...],
    'model_patches': [patch bytes ...]}. expect is passed to every verify_receipt (e.g. the task's image and
    test hashes from the verifier's own task pool, its tokenizer_sha256)."""
    check_expect(expect)
    if not isinstance(attachment, Mapping) or attachment.get('version') != GROUP_ATTACHMENT_VERSION:
        raise ReceiptError('bad_attachment_version')
    if attachment.get('env_id') != env_id or attachment.get('task') != task:
        raise ReceiptError('identity_mismatch')
    rollouts = attachment.get('rollouts')
    if not isinstance(rollouts, list):
        raise ReceiptError('malformed_group')
    by_nonce = {}
    for r in rollouts:
        n = r.get('nonce') if isinstance(r, Mapping) else None
        if type(n) is not int or n in by_nonce:
            raise ReceiptError('malformed_group')
        by_nonce[n] = r
    if sorted(by_nonce) != list(range(group_size)):
        raise ReceiptError('incomplete_group')
    claims, masks, segs, patches = [], [], [], []
    for n in range(group_size):
        r = by_nonce[n]
        try:
            s = as_segments(r['segments'])
        except (KeyError, TypeError, ValueError):
            raise ReceiptError('malformed_trajectory') from None
        claims.append(verify_receipt(r.get('receipt') or {}, s, trusted_keys, epoch=epoch, hotkey=hotkey,
                                     env_id=env_id, task=task, nonce=n, expect=expect))
        patches.append(_model_patch(r, claims[-1]))
        segs.append(s)
        masks.append(loss_mask(s))
    if any(c['outcome'] == 'infra_error' for c in claims):
        raise ReceiptError('infra_error_in_group')
    done = [c for c in claims if c['outcome'] == 'completed']
    if not (any(c['reward'] >= success_reward for c in done) and any(c['reward'] < success_reward for c in done)):
        raise ReceiptError('group_gate_failed')     # at least one completed and correct, one completed and wrong
    return {'claims': claims, 'loss_masks': masks, 'segments': segs, 'model_patches': patches}
