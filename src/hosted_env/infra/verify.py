"""Group verification: called once per group by validator admission / audit and when the trainer reads it in
(design §9, §10).

Hosted-env attachment format inside the group object (`GROUP_ATTACHMENT_VERSION`):
    {"version": "hosted-env-group-v2", "env_id": ..., "task": ..., "seal": {...},
     "rollouts": [{"nonce": 0, "segments": [{"src": "P", "ids": [...]}, ...], "receipt": {...},
                   "model_patch": "<base64 of the patch bytes that were graded>"}, ...]}

Rules: the seal is signed by the gateway key and names this group (epoch, hotkey, env, task, group size) and, by
nonce, exactly the receipts in it (receipt_digest), so rollouts from different runs cannot be assembled into one
group and the open / finish times are the gateway's; exactly one rollout for each nonce 0..group_size-1; every
receipt must pass `verify_receipt` (with the
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

from .receipt import (EnvReceiptSpec, ReceiptError, as_segments, check_expect, loss_mask, receipt_digest,
                      verify_group_seal,
                      verify_receipt)

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


def verify_group(attachment: Mapping, trusted_keys: Mapping[str, VerifyKey], *, gateway_key: VerifyKey, epoch: str,
                 hotkey: str, env_id: str, task: str, spec: EnvReceiptSpec, group_size: int = 8,
                 success_reward: float = 1.0, expect: Mapping | None = None) -> dict:
    """On success returns {'claims': [receipt contents ordered by nonce], 'loss_masks': [...], 'segments': [...],
    'model_patches': [patch bytes ...], 'seal': seal contents}. trusted_keys are the worker keys, gateway_key the
    key that signs group seals (both from the epoch's opening). spec is the receipt spec of the environment kind
    serving env_id (hosted_env.envs.receipt_spec; /v1/info names the kind). expect is passed to every
    verify_receipt (e.g. the task's image and test hashes from the verifier's own task pool, its tokenizer_sha256)."""
    check_expect(expect, spec)
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
    seal = verify_group_seal(attachment.get('seal') or {}, gateway_key, epoch=epoch, hotkey=hotkey, env_id=env_id,
                             task=task, group_size=group_size)
    for n in range(group_size):
        r = by_nonce[n].get('receipt') or {}
        if seal['receipts'][n] != receipt_digest(r):
            raise ReceiptError('seal_mismatch')         # not the receipt the worker reported for this nonce
    claims, masks, segs, patches = [], [], [], []
    for n in range(group_size):
        r = by_nonce[n]
        try:
            s = as_segments(r['segments'])
        except (KeyError, TypeError, ValueError):
            raise ReceiptError('malformed_trajectory') from None
        claims.append(verify_receipt(r.get('receipt') or {}, s, trusted_keys, epoch=epoch, hotkey=hotkey,
                                     env_id=env_id, task=task, nonce=n, spec=spec, expect=expect))
        patches.append(_model_patch(r, claims[-1]))
        segs.append(s)
        masks.append(loss_mask(s))
    if any(c['outcome'] == 'infra_error' for c in claims):
        raise ReceiptError('infra_error_in_group')
    done = [c for c in claims if c['outcome'] == 'completed']
    if not (any(c['reward'] >= success_reward for c in done) and any(c['reward'] < success_reward for c in done)):
        raise ReceiptError('group_gate_failed')     # at least one completed and correct, one completed and wrong
    return {'claims': claims, 'loss_masks': masks, 'segments': segs, 'model_patches': patches, 'seal': seal}
