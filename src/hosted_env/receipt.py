"""Trajectory receipts: issued by the worker, checked by the validator / trainer with the pure function
`verify_receipt` (design §9).

- The worker sees every token of a rollout: the leading task prompt (rendered by the worker, source P), the model
  tokens the miner sends each turn (source M), and the tool-output tokens the worker renders itself (source T). It
  hashes [(source, token ids)...] incrementally and, when the rollout ends, signs a receipt covering identity and
  the grading result and hands it to the miner.
- The miner puts the group's receipts into the group object and uploads it through the existing complete-group
  flow; the upload path is unchanged.
- The validator calls `verify_receipt` on each rollout: check the worker signature, check identity, recompute the
  hash from the uploaded tokens, then use the outcome / reward from the receipt. Model tokens are still verified
  with the sampling proof as before.
- The source tags double as the loss mask: only M segments are trained on (`loss_mask`).

This module depends only on PyNaCl and the standard library and can be copied upstream as-is.
"""
from __future__ import annotations

import hashlib
import json
import struct
import time
from collections.abc import Mapping, Sequence

from nacl.signing import SigningKey, VerifyKey

from .common import b64u, canonical, sign, unb64u, verify

TRAJ_DOMAIN = b'affine-hosted-env-trajectory-v1'
RECEIPT_DOMAIN = b'affine-hosted-env-receipt-v1|'
SOURCES = ('P', 'M', 'T')   # P=task prompt from the environment, M=model generated, T=tool output (environment generated)
OUTCOMES = ('completed', 'unresolved', 'infra_error')
RECEIPT_VERSION = 1


class ReceiptError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def as_segments(segments) -> list[tuple[str, list[int]]]:
    """Accept [(src, ids)] or [{'src': .., 'ids': ..}] and normalize to [(src, ids)]."""
    out = []
    for s in segments:
        if isinstance(s, Mapping):
            out.append((s['src'], s['ids']))
        else:
            src, ids = s
            out.append((src, ids))
    return out


class TrajectoryHasher:
    """Incremental per-segment hash. Each segment is encoded as: source 1 byte (ASCII) + segment length u32 +
    each token id u32 (little-endian).

    Constraints: must start with P, adjacent segments have different sources, segments are non-empty, token ids
    are u32 ints.
    """

    def __init__(self):
        self._h = hashlib.sha256(TRAJ_DOMAIN)
        self.n_tokens = 0
        self.n_segments = 0
        self._last: str | None = None

    def add(self, source: str, token_ids: Sequence[int]) -> None:
        if source not in SOURCES:
            raise ValueError('bad segment source')
        if not token_ids:
            raise ValueError('empty segment')
        if self._last is None and source != 'P':
            raise ValueError('trajectory must start with the environment prompt')
        if source == self._last:
            raise ValueError('adjacent segments must have different sources')
        if any(type(t) is not int or not 0 <= t < 2**32 for t in token_ids):
            raise ValueError('token ids must be u32 ints')
        n = len(token_ids)
        self._h.update(source.encode() + struct.pack('<I', n) + struct.pack(f'<{n}I', *token_ids))
        self.n_tokens += n
        self.n_segments += 1
        self._last = source

    def hexdigest(self) -> str:
        return self._h.copy().hexdigest()


def trajectory_sha256(segments) -> tuple[str, int]:
    h = TrajectoryHasher()
    for src, ids in as_segments(segments):
        h.add(src, ids)
    return h.hexdigest(), h.n_tokens


def loss_mask(segments) -> list[int]:
    """Per-token training mask: 1 for M segments, 0 for P and T segments."""
    return [1 if src == 'M' else 0 for src, ids in as_segments(segments) for _ in ids]


def sign_receipt(worker_key: SigningKey, key_id: str, *, epoch: str, hotkey: str, env_id: str, task: str,
                 nonce: int, trajectory_sha256: str, n_tokens: int, outcome: str, reward: float,
                 issued_at: int | None = None) -> dict:
    """Worker side: issued after the rollout ends and is graded. trajectory_sha256 / n_tokens come from the
    worker's TrajectoryHasher."""
    if outcome not in OUTCOMES:
        raise ValueError('bad outcome')
    payload = {'v': RECEIPT_VERSION, 'epoch': epoch, 'hotkey': hotkey, 'env_id': env_id, 'task': task,
               'nonce': nonce, 'trajectory_sha256': trajectory_sha256, 'n_tokens': n_tokens, 'outcome': outcome,
               'reward': float(reward), 'key_id': key_id,
               'issued_at': int(time.time()) if issued_at is None else issued_at}
    body = canonical(payload)
    return {'payload': b64u(body), 'sig': b64u(sign(worker_key, RECEIPT_DOMAIN, body))}


def read_receipt(receipt: Mapping) -> dict:
    """Decode the contents without checking the signature (for debugging)."""
    try:
        return json.loads(unb64u(receipt['payload']))
    except (KeyError, ValueError, TypeError):
        raise ReceiptError('malformed_receipt') from None


def verify_receipt(receipt: Mapping, segments, trusted_keys: Mapping[str, VerifyKey], *, epoch: str, hotkey: str,
                   env_id: str, task: str, nonce: int) -> dict:
    """Validator / trainer side. segments is the [(source, token ids)] of this rollout as uploaded by the miner;
    trusted_keys are the worker public keys {key_id: VerifyKey} from this epoch's opening. Returns the receipt
    contents (including outcome and reward) on success, otherwise raises ReceiptError."""
    try:
        body, sig = unb64u(receipt['payload']), unb64u(receipt['sig'])
        claims = json.loads(body)
        key = trusted_keys.get(claims['key_id'])
    except (KeyError, ValueError, TypeError, AttributeError):
        raise ReceiptError('malformed_receipt') from None
    if key is None:
        raise ReceiptError('untrusted_worker_key')
    if not verify(key, RECEIPT_DOMAIN, body, sig):
        raise ReceiptError('bad_signature')
    expected = {'v': RECEIPT_VERSION, 'epoch': epoch, 'hotkey': hotkey, 'env_id': env_id, 'task': task,
                'nonce': nonce}
    if any(claims.get(k) != v or type(claims.get(k)) is not type(v) for k, v in expected.items()):
        raise ReceiptError('identity_mismatch')         # another task, another nonce, someone else's receipt
    if claims.get('outcome') not in OUTCOMES:
        raise ReceiptError('malformed_receipt')
    try:
        digest, n = trajectory_sha256(segments)
    except (ValueError, TypeError, KeyError):
        raise ReceiptError('malformed_trajectory') from None
    if digest != claims.get('trajectory_sha256') or n != claims.get('n_tokens'):
        raise ReceiptError('trajectory_mismatch')       # uploaded tokens differ from what actually ran in the env
    return claims
