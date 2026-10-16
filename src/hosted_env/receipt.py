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
- v2 also binds what the reward is a function of (CONTEXT_FIELDS), so a verifier can replay the grading offline
  and tie the tokens to a tokenizer: the task (image digest, test patch, test command), the grading (grader
  version and user, the model's patch, the verifier's exit code), the model side (model id and revision as the
  worker is configured, tokenizer and chat-template hashes, renderer) and the worker build, with times in
  milliseconds. The model id is the worker's configuration, not a proof: which model produced the M tokens is
  still shown by the sampling proof, and trajectory_sha256 ties both to the same tokens.

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
RECEIPT_DOMAIN = b'affine-hosted-env-receipt-v2|'
SOURCES = ('P', 'M', 'T')   # P=task prompt from the environment, M=model generated, T=tool output (environment generated)
OUTCOMES = ('completed', 'unresolved', 'infra_error')
RECEIPT_VERSION = 2
_HEX = frozenset('0123456789abcdef')
# Field -> type. *_sha256 fields are lowercase hex SHA-256 or '' (not known: e.g. a rollout that never got graded);
# start_tree is a 40-hex git tree id or ''; verifier_returncode is None when no verifier ran.
CONTEXT_FIELDS = {
    'image': str,                   # name@sha256:... the rollout and its grading ran in
    'start_tree': str,              # git tree the model started from (after setup); the patch applies to it
    'test_patch_sha256': str,       # of the hidden test patch text (UTF-8)
    'test_command_sha256': str,     # of the test command text (UTF-8)
    'grader': str,                  # grading semantics version, e.g. isolated-v2
    'grader_user': str,             # who the hidden tests ran as: nobody or root
    'model_patch_sha256': str,      # of the model's patch bytes as captured, before the grader put protected test
                                    # infrastructure back (a re-grader applies the same rule); the attachment has it
    'verifier_returncode': (int, type(None)),
    'model_id': str,                # as configured on the worker
    'model_revision': str,
    'tokenizer_sha256': str,        # of the tokenizer files the worker rendered with (render.tokenizer_digest)
    'chat_template_sha256': str,
    'renderer': str,                # renderer kind and library version
    'executor_version': str,
    'started_at_ms': int,
}


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


def _context_problem(ctx: Mapping) -> str | None:
    if not isinstance(ctx, Mapping):
        return 'context is not a mapping'
    for k, t in CONTEXT_FIELDS.items():
        if k not in ctx:
            return f'missing {k}'
        v = ctx[k]
        if not isinstance(v, t) or isinstance(v, bool):
            return f'bad type for {k}'
        n = 64 if k.endswith('_sha256') else 40 if k == 'start_tree' else 0
        if n and v and (len(v) != n or not set(v) <= _HEX):
            return f'bad digest for {k}'
    return None


def sign_receipt(worker_key: SigningKey, key_id: str, *, epoch: str, hotkey: str, env_id: str, task: str,
                 nonce: int, trajectory_sha256: str, n_tokens: int, outcome: str, reward: float,
                 context: Mapping, issued_at_ms: int | None = None) -> dict:
    """Worker side: issued after the rollout ends and is graded. trajectory_sha256 / n_tokens come from the
    worker's TrajectoryHasher; context holds exactly the CONTEXT_FIELDS."""
    if outcome not in OUTCOMES:
        raise ValueError('bad outcome')
    problem = _context_problem(context)
    if problem or set(context) != set(CONTEXT_FIELDS):
        raise ValueError(f'bad receipt context: {problem or "unknown fields"}')
    if issued_at_ms is None:                    # never before the start, even if the wall clock stepped back
        issued_at_ms = max(int(time.time() * 1000), context['started_at_ms'])
    payload = {**context, 'v': RECEIPT_VERSION, 'epoch': epoch, 'hotkey': hotkey, 'env_id': env_id, 'task': task,
               'nonce': nonce, 'trajectory_sha256': trajectory_sha256, 'n_tokens': n_tokens, 'outcome': outcome,
               'reward': float(reward), 'key_id': key_id, 'issued_at_ms': issued_at_ms}
    body = canonical(payload)
    return {'payload': b64u(body), 'sig': b64u(sign(worker_key, RECEIPT_DOMAIN, body))}


def read_receipt(receipt: Mapping) -> dict:
    """Decode the contents without checking the signature (for debugging)."""
    try:
        return json.loads(unb64u(receipt['payload']))
    except (KeyError, ValueError, TypeError):
        raise ReceiptError('malformed_receipt') from None


def check_expect(expect: Mapping | None) -> None:
    """The verifier's own `expect` names only context fields: a typo raises ValueError up front, whatever the
    receipts look like."""
    bad = sorted(set(expect or {}) - set(CONTEXT_FIELDS))
    if bad:
        raise ValueError(f'expect: not receipt context fields: {bad}')


def verify_receipt(receipt: Mapping, segments, trusted_keys: Mapping[str, VerifyKey], *, epoch: str, hotkey: str,
                   env_id: str, task: str, nonce: int, expect: Mapping | None = None) -> dict:
    """Validator / trainer side. segments is the [(source, token ids)] of this rollout as uploaded by the miner;
    trusted_keys are the worker public keys {key_id: VerifyKey} from this epoch's opening. expect optionally pins
    context fields to the values the verifier knows (the task's image and test hashes from its own copy of the
    task pool, the tokenizer it trains with). Returns the receipt contents (including outcome and reward) on
    success, otherwise raises ReceiptError."""
    check_expect(expect)
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
    issued, started = claims.get('issued_at_ms'), claims.get('started_at_ms')
    if (claims.get('outcome') not in OUTCOMES or _context_problem(claims) or type(issued) is not int
            or started > issued):
        raise ReceiptError('malformed_receipt')
    for k, v in (expect or {}).items():
        if claims[k] != v or type(claims[k]) is not type(v):
            raise ReceiptError('context_mismatch')      # e.g. another image, test patch or tokenizer than expected
    try:
        digest, n = trajectory_sha256(segments)
    except (ValueError, TypeError, KeyError):
        raise ReceiptError('malformed_trajectory') from None
    if digest != claims.get('trajectory_sha256') or n != claims.get('n_tokens'):
        raise ReceiptError('trajectory_mismatch')       # uploaded tokens differ from what actually ran in the env
    return claims
