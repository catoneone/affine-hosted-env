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
- The receipt also binds what the reward is a function of, so a verifier can replay the grading offline and tie
  the tokens to a tokenizer. The context is flat in the signed payload and has two parts: CORE_FIELDS, the same
  for every environment (the task_hash of the task row served, the model side -- model id and revision as the
  worker is configured, tokenizer and chat-template hashes, renderer -- the sandbox limits, the worker build, the
  start time in milliseconds), and the fields the environment declares in its EnvReceiptSpec (for SWE: the task's
  image digest, test patch and command, the grader version and user, the model's patch, the verifier's exit code;
  envs.swe.receipt), named in the payload by env_spec. Signing takes that spec and verifying takes it or looks it
  up by env_spec (SPECS), so this module knows no environment. The model id is the worker's configuration, not a
  proof: which model produced the M tokens is still shown by the sampling proof, and trajectory_sha256 ties both
  to the same tokens. The wire format, versions and error codes are in docs/formats.md.

This module depends only on PyNaCl and the standard library and can be copied upstream as-is.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nacl.signing import SigningKey, VerifyKey

from .common import b64u, canonical, sign, unb64u, verify

TRAJ_DOMAIN = b'affine-hosted-env-trajectory-v1'
RECEIPT_DOMAIN = b'affine-hosted-env-receipt-v3|'
SOURCES = ('P', 'M', 'T')   # P=task prompt from the environment, M=model generated, T=tool output (environment generated)
OUTCOMES = ('completed', 'unresolved', 'infra_error')
RECEIPT_VERSION = 3
_HEX = frozenset('0123456789abcdef')
# Versioning (docs/formats.md): the payload names its environment field set in env_spec ('swe/1'). A change to an
# environment's fields is a new spec version (registered next to the old one, so old receipts still verify); only
# a change to CORE_FIELDS or PAYLOAD_KEYS bumps RECEIPT_VERSION and RECEIPT_DOMAIN. Fields are never added inside a
# version: v2 grew three fields while it ran, so older v2 receipts failed verification.
#
# Context field -> type. *_sha256 fields are lowercase hex SHA-256 or '' (not known: e.g. a rollout that never got
# graded); task_hash is always known (REQUIRED_HEX). Every field is required.
CORE_FIELDS = {
    'task_hash': str,               # task_hash() of the task-index row the gateway served for this task: verifiers'
                                    # task_key encoding, so it is the hash a verifier computes from its own copy
    'sandbox_limits': str,          # the sandbox the rollout and its grading ran in and its limits, e.g.
                                    # 'k8s:cpus=1;memory=2g;disk=10g;threads=1' (threads: the OMP / OpenBLAS /
                                    # MKL caps): hitting its own limit scores 0, so an offline re-grade on a
                                    # bigger box can differ unless it uses the same limits
    'model_id': str,                # as configured on the worker
    'model_revision': str,
    'tokenizer_sha256': str,        # of the tokenizer files the worker rendered with (render.tokenizer_digest)
    'chat_template_sha256': str,
    'renderer': str,                # renderer kind and library version
    'executor_version': str,
    'started_at_ms': int,
}
REQUIRED_HEX = {'task_hash': 64}  # core fields that hold a lowercase hex digest of this length, never ''
# signed payload keys that are not context: an environment cannot declare a field with one of these names
PAYLOAD_KEYS = frozenset({'v', 'env_spec', 'epoch', 'hotkey', 'env_id', 'task', 'nonce', 'trajectory_sha256',
                          'n_tokens', 'outcome', 'reward', 'key_id', 'issued_at_ms'})
_SPEC_ID = re.compile(r'[a-z][a-z0-9_]*/[1-9][0-9]*')


def task_hash(data: Mapping) -> str:
    """Content hash of a task: verifiers' task_key -- sha256 of json.dumps(data, sort_keys=True) with the DEFAULT
    separators and ensure_ascii -- over the task-index row as served, with None-valued keys dropped (as
    model_dump(exclude_none=True) drops them). This is the encoding of verifiers' Task.hash; it equals that hash
    when the task's data model dumps to exactly this row."""
    if not isinstance(data, Mapping):
        raise TypeError('task data must be a mapping')
    return hashlib.sha256(json.dumps({k: v for k, v in data.items() if v is not None},
                                     sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class EnvReceiptSpec:
    """What an environment adds to its receipts, declared by the environment (e.g. envs.swe.receipt.SPEC).
    id: '<name>/<version>', signed in every receipt as env_spec; any change to the rest is a new version.
    fields: context field -> type (or tuple of types), all required, signed flat next to CORE_FIELDS.
    hex_lengths: fields besides *_sha256 that hold a lowercase hex digest of that length ('' allowed).
    artifacts: attachment artifact name -> (the context field holding its sha256, max bytes)."""
    id: str
    fields: Mapping[str, Any]
    hex_lengths: Mapping[str, int] = field(default_factory=dict)
    artifacts: Mapping[str, tuple[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.id, str) or not _SPEC_ID.fullmatch(self.id):
            raise ValueError(f"spec id must look like 'name/1', not {self.id!r}")
        clash = sorted(set(self.fields) & (set(CORE_FIELDS) | PAYLOAD_KEYS))
        if clash:
            raise ValueError(f'environment receipt fields clash with core / payload keys: {clash}')
        for name, (hash_field, _) in self.artifacts.items():
            if hash_field not in self.fields or not hash_field.endswith('_sha256'):
                raise ValueError(f'artifact {name!r}: {hash_field!r} is not one of the spec\'s *_sha256 fields')

    def context_fields(self) -> dict:
        return {**CORE_FIELDS, **self.fields}


# Every receipt spec an environment has ever signed with, by id: verify_receipt looks a receipt's env_spec up here
# when the caller passes no spec. Environments register their specs where they define them (hosted_env imports
# them), and keep old versions registered so receipts signed before a field change still verify.
SPECS: dict[str, EnvReceiptSpec] = {}


def register_spec(spec: EnvReceiptSpec) -> EnvReceiptSpec:
    """Add spec to SPECS (idempotent); a different spec under a registered id is a ValueError -- an id names one
    field set forever."""
    have = SPECS.setdefault(spec.id, spec)
    if have != spec:
        raise ValueError(f'receipt spec {spec.id!r} is already registered with other fields; bump its version')
    return spec


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


def _context_problem(ctx: Mapping, spec: EnvReceiptSpec) -> str | None:
    if not isinstance(ctx, Mapping):
        return 'context is not a mapping'
    for k, t in spec.context_fields().items():
        if k not in ctx:
            return f'missing {k}'
        v = ctx[k]
        if not isinstance(v, t) or isinstance(v, bool):
            return f'bad type for {k}'
        n = 64 if k.endswith('_sha256') else REQUIRED_HEX.get(k) or spec.hex_lengths.get(k, 0)
        if (n and v or k in REQUIRED_HEX) and (len(v) != n or not set(v) <= _HEX):
            return f'bad digest for {k}'
    return None


def sign_receipt(worker_key: SigningKey, key_id: str, *, epoch: str, hotkey: str, env_id: str, task: str,
                 nonce: int, trajectory_sha256: str, n_tokens: int, outcome: str, reward: float,
                 context: Mapping, spec: EnvReceiptSpec, issued_at_ms: int | None = None) -> dict:
    """Worker side: issued after the rollout ends and is graded. trajectory_sha256 / n_tokens come from the
    worker's TrajectoryHasher; context holds exactly CORE_FIELDS plus the environment's spec.fields."""
    if outcome not in OUTCOMES:
        raise ValueError('bad outcome')
    problem = _context_problem(context, spec)
    if problem or set(context) != set(spec.context_fields()):
        raise ValueError(f'bad receipt context: {problem or "unknown fields"}')
    if issued_at_ms is None:                    # never before the start, even if the wall clock stepped back
        issued_at_ms = max(int(time.time() * 1000), context['started_at_ms'])
    payload = {**context, 'v': RECEIPT_VERSION, 'env_spec': spec.id, 'epoch': epoch, 'hotkey': hotkey,
               'env_id': env_id, 'task': task, 'nonce': nonce, 'trajectory_sha256': trajectory_sha256,
               'n_tokens': n_tokens, 'outcome': outcome, 'reward': float(reward), 'key_id': key_id,
               'issued_at_ms': issued_at_ms}
    body = canonical(payload)
    return {'payload': b64u(body), 'sig': b64u(sign(worker_key, RECEIPT_DOMAIN, body))}


def receipt_digest(receipt: Mapping) -> str:
    """SHA-256 of a receipt's signed payload bytes: how the gateway's group seal names each receipt."""
    try:
        return hashlib.sha256(unb64u(receipt['payload'])).hexdigest()
    except (KeyError, ValueError, TypeError):
        raise ReceiptError('malformed_receipt') from None


# ---------------------------------------------------------------- group seal (signed by the gateway)
# When the miner finishes a task, the gateway signs which receipts make up the group, by nonce, and when the group
# was opened and finished by its own clock. The receipt hashes come from the workers (reported when each session
# closes), never from the miner, so receipts from different runs cannot be assembled into one group, and the
# times do not rest on a worker's clock.
SEAL_DOMAIN = b'affine-hosted-env-group-seal-v1|'
SEAL_VERSION = 1


def sign_group_seal(gateway_key: SigningKey, *, epoch: str, hotkey: str, env_id: str, task: str, task_id: str,
                    group_size: int, opened_at_ms: int, finished_at_ms: int, receipts: Sequence[str | None]) -> dict:
    """receipts[n] is the receipt_digest the worker reported for nonce n, or None when none was reported."""
    payload = {'v': SEAL_VERSION, 'epoch': epoch, 'hotkey': hotkey, 'env_id': env_id, 'task': task,
               'task_id': task_id, 'group_size': group_size, 'opened_at_ms': opened_at_ms,
               'finished_at_ms': finished_at_ms, 'receipts': list(receipts)}
    body = canonical(payload)
    return {'payload': b64u(body), 'sig': b64u(sign(gateway_key, SEAL_DOMAIN, body))}


def verify_group_seal(seal: Mapping, gateway_key: VerifyKey, *, epoch: str, hotkey: str, env_id: str, task: str,
                      group_size: int) -> dict:
    """Check the gateway's signature and the group identity; returns the seal contents."""
    try:
        body, sig = unb64u(seal['payload']), unb64u(seal['sig'])
        claims = json.loads(body)
    except (KeyError, ValueError, TypeError, AttributeError):
        raise ReceiptError('malformed_seal') from None
    if not verify(gateway_key, SEAL_DOMAIN, body, sig):
        raise ReceiptError('bad_seal_signature')
    if not isinstance(claims, Mapping):
        raise ReceiptError('malformed_seal')
    expected = {'v': SEAL_VERSION, 'epoch': epoch, 'hotkey': hotkey, 'env_id': env_id, 'task': task,
                'group_size': group_size}
    if any(claims.get(k) != v or type(claims.get(k)) is not type(v) for k, v in expected.items()):
        raise ReceiptError('seal_identity_mismatch')
    opened, finished, receipts = claims.get('opened_at_ms'), claims.get('finished_at_ms'), claims.get('receipts')
    if (type(opened) is not int or type(finished) is not int or opened > finished or type(claims.get('task_id'))
            is not str or not isinstance(receipts, list) or len(receipts) != group_size
            or any(r is not None and type(r) is not str for r in receipts)):
        raise ReceiptError('malformed_seal')
    return claims


def read_receipt(receipt: Mapping) -> dict:
    """Decode the contents without checking the signature (for debugging)."""
    try:
        return json.loads(unb64u(receipt['payload']))
    except (KeyError, ValueError, TypeError):
        raise ReceiptError('malformed_receipt') from None


def check_expect(expect: Mapping | None, spec: EnvReceiptSpec) -> None:
    """The verifier's own `expect` names only context fields: a typo raises ValueError up front, whatever the
    receipts look like."""
    bad = sorted(set(expect or {}) - set(spec.context_fields()))
    if bad:
        raise ValueError(f'expect: not receipt context fields: {bad}')


def resolve_spec(claims: Mapping, spec: EnvReceiptSpec | None) -> EnvReceiptSpec:
    """The spec a receipt is checked against: the caller's (which must be the one the receipt names) or, with
    none, the registered spec of the receipt's env_spec."""
    named = claims.get('env_spec')
    if spec is not None:
        if named != spec.id:
            raise ReceiptError('env_spec_mismatch')     # signed with another field set than the caller pinned
        return spec
    found = SPECS.get(named) if type(named) is str else None
    if found is None:
        raise ReceiptError('unknown_env_spec')          # import the environment's spec module (hosted_env does)
    return found


def verify_receipt(receipt: Mapping, segments, trusted_keys: Mapping[str, VerifyKey], *, epoch: str, hotkey: str,
                   env_id: str, task: str, nonce: int, spec: EnvReceiptSpec | None = None,
                   expect: Mapping | None = None) -> dict:
    """Validator / trainer side. segments is the [(source, token ids)] of this rollout as uploaded by the miner;
    trusted_keys are the worker public keys {key_id: VerifyKey} from this epoch's opening. spec pins the receipt
    spec of the environment kind serving env_id (hosted_env.envs.receipt_spec); omitted, the receipt's own
    env_spec is looked up in SPECS (the worker's signature covers it). expect optionally pins context fields to the
    values the verifier knows (the task_hash of its own copy of the task row, the task's image and test hashes, the
    tokenizer it trains with). Returns the receipt contents (including outcome and reward) on success, otherwise
    raises ReceiptError."""
    if spec is not None:
        check_expect(expect, spec)
    try:
        body, sig = unb64u(receipt['payload']), unb64u(receipt['sig'])
        claims = json.loads(body)
        key = trusted_keys.get(claims['key_id'])
    except (KeyError, ValueError, TypeError, AttributeError):
        raise ReceiptError('malformed_receipt') from None
    if claims.get('v') != RECEIPT_VERSION or type(claims.get('v')) is not int:
        raise ReceiptError('unsupported_receipt_version')   # v2 receipts are not accepted (docs/formats.md)
    if key is None:
        raise ReceiptError('untrusted_worker_key')
    if not verify(key, RECEIPT_DOMAIN, body, sig):
        raise ReceiptError('bad_signature')
    spec = resolve_spec(claims, spec)
    check_expect(expect, spec)
    expected = {'epoch': epoch, 'hotkey': hotkey, 'env_id': env_id, 'task': task, 'nonce': nonce}
    if any(claims.get(k) != v or type(claims.get(k)) is not type(v) for k, v in expected.items()):
        raise ReceiptError('identity_mismatch')         # another task, another nonce, someone else's receipt
    issued, started = claims.get('issued_at_ms'), claims.get('started_at_ms')
    # exactly the spec's fields, as sign_receipt writes them: verifying with another environment's spec fails in
    # both directions (a narrower spec would otherwise skip the type and digest checks of the fields it lacks)
    if (claims.get('outcome') not in OUTCOMES or set(claims) != PAYLOAD_KEYS | set(spec.context_fields())
            or _context_problem(claims, spec) or type(issued) is not int or started > issued):
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
