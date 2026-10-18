"""Group verification: called once per group by validator admission / audit and when the trainer reads it in
(design §9, §10).

Hosted-env attachment format inside the group object (`GROUP_ATTACHMENT_VERSION`):
    {"version": "hosted-env-group-v3", "env_id": ..., "task": ..., "seal": {...},
     "rollouts": [{"nonce": 0, "segments": [{"src": "P", "ids": [...]}, ...], "receipt": {...},
                   "artifacts": {"<name>": "<base64>", ...}}, ...]}
artifacts carries what the environment declares in its EnvReceiptSpec.artifacts, each bound by a sha256 field of the
receipt (SWE: {"model_patch": <the patch bytes that were graded>} bound by model_patch_sha256). v2 had a top-level
"model_patch" instead.

Rules: the seal is signed by the gateway key and names this group (epoch, hotkey, env, task, group size) and, by
nonce, exactly the receipts in it (receipt_digest), so rollouts from different runs cannot be assembled into one
group and the open / finish times are the gateway's; exactly one rollout for each nonce 0..group_size-1; every
receipt must pass `verify_receipt` (with the environment's spec and the caller's `expect`); every rollout carries
exactly the spec's artifacts, each within its size cap and hashing to its receipt field (an empty artifact when
that field is '', e.g. the rollout was not graded), so a verifier can re-grade offline with the receipt's task
hashes; any infra_error voids the whole group; at least one completed rollout with reward >= success_reward and at
least one completed rollout with reward < success_reward. If any check fails the whole group scores 0 (raises
ReceiptError). The sampling proof for model tokens is not checked here; the existing flow still verifies it.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
from collections.abc import Mapping

from nacl.signing import VerifyKey

from .receipt import (EnvReceiptSpec, ReceiptError, as_segments, check_expect, loss_mask, receipt_digest,
                      resolve_spec, verify_group_seal, verify_receipt)

GROUP_ATTACHMENT_VERSION = 'hosted-env-group-v3'


def _artifacts(r: Mapping, claims: Mapping, spec: EnvReceiptSpec) -> dict[str, bytes]:
    """The rollout's artifacts, decoded, each checked against the receipt field that binds it."""
    arts = r.get('artifacts')
    if not isinstance(arts, Mapping) or set(arts) != set(spec.artifacts):
        raise ReceiptError('malformed_group')
    out = {}
    for name, (hash_field, max_bytes) in spec.artifacts.items():
        b64 = arts[name]
        if not isinstance(b64, str) or len(b64) > (max_bytes + 2) // 3 * 4:
            raise ReceiptError('malformed_group')
        try:
            raw = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            raise ReceiptError('malformed_group') from None
        want = claims.get(hash_field, '')
        if (hashlib.sha256(raw).hexdigest() != want) if want else raw:   # '' = never produced: no artifact either
            raise ReceiptError('artifact_mismatch')         # not what the receipt says was graded
        out[name] = raw
    return out


def verify_group(attachment: Mapping, trusted_keys: Mapping[str, VerifyKey], *, gateway_key: VerifyKey, epoch: str,
                 hotkey: str, env_id: str, task: str, spec: EnvReceiptSpec | None = None, group_size: int = 8,
                 success_reward: float = 1.0, expect: Mapping | None = None, require_gate: bool = True) -> dict:
    """On success returns {'claims': [receipt contents ordered by nonce], 'loss_masks': [...], 'segments': [...],
    'artifacts': [{name: bytes} per nonce], 'seal': seal contents}. trusted_keys are the worker keys, gateway_key
    the key that signs group seals (both from the epoch's opening). spec optionally pins the receipt spec of the
    environment kind serving env_id (hosted_env.envs.receipt_spec, the kind from the epoch's opening / the
    gateway's /v1/info); omitted, each receipt's env_spec is looked up in receipt.SPECS. Either way every receipt
    of the group must name the same env_spec and carry exactly its fields, so another environment's spec rejects
    it rather than checking less. expect is passed to every verify_receipt (e.g. the task_hash, image and test
    hashes from the verifier's own task pool, its tokenizer_sha256). require_gate=False skips only the two
    group-level rules (no infra error, one pass and one fail) -- for a re-grade, which needs every rollout's
    verified receipt and artifacts whatever the group's outcome; everything else is checked the same."""
    if spec is not None:
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
    claims, masks, segs, artifacts = [], [], [], []
    for n in range(group_size):
        r = by_nonce[n]
        try:
            s = as_segments(r['segments'])
        except (KeyError, TypeError, ValueError):
            raise ReceiptError('malformed_trajectory') from None
        claims.append(verify_receipt(r.get('receipt') or {}, s, trusted_keys, epoch=epoch, hotkey=hotkey,
                                     env_id=env_id, task=task, nonce=n, spec=spec, expect=expect))
        if claims[-1]['env_spec'] != claims[0]['env_spec']:
            # one group, one field set: with spec=None each receipt resolves its own spec, so two different
            # registered specs could otherwise share a group (test_a_group_has_one_env_spec)
            raise ReceiptError('env_spec_mismatch')
        artifacts.append(_artifacts(r, claims[-1], resolve_spec(claims[-1], spec)))
        segs.append(s)
        masks.append(loss_mask(s))
    result = {'claims': claims, 'loss_masks': masks, 'segments': segs, 'artifacts': artifacts, 'seal': seal}
    if not require_gate:
        return result
    if any(c['outcome'] == 'infra_error' for c in claims):
        raise ReceiptError('infra_error_in_group')
    done = [c for c in claims if c['outcome'] == 'completed']
    if not (any(c['reward'] >= success_reward for c in done) and any(c['reward'] < success_reward for c in done)):
        raise ReceiptError('group_gate_failed')     # at least one completed and correct, one completed and wrong
    return result
