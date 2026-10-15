"""Shared utilities: canonical JSON, base64url, Ed25519 keys.

Signing uses PyNaCl (the same library upstream affine's admission / distributed_roles use); receipt verification
depends only on it and the standard library.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b'=').decode()


def unb64u(s: str) -> bytes:
    if not isinstance(s, str):
        raise ValueError('base64url must be str')
    return base64.urlsafe_b64decode(s + '=' * (-len(s) % 4))


def canonical(value) -> bytes:
    """The JSON encoding used for all signing and hashing: sorted keys, no whitespace, NaN not allowed."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def key_id(vk: VerifyKey) -> str:
    """Short id of a public key: the first 16 hex characters of sha256(raw 32-byte public key)."""
    return sha256_hex(bytes(vk))[:16]


def verify_key_from_hex(h: str) -> VerifyKey:
    raw = bytes.fromhex(h)
    if len(raw) != 32:
        raise ValueError('Ed25519 public key must be 32 bytes')
    return VerifyKey(raw)


def public_hex(sk: SigningKey) -> str:
    return bytes(sk.verify_key).hex()


def sign(sk: SigningKey, domain: bytes, msg: bytes) -> bytes:
    return sk.sign(domain + msg).signature


def verify(vk: VerifyKey, domain: bytes, msg: bytes, sig: bytes) -> bool:
    try:
        vk.verify(domain + msg, sig)
        return True
    except (BadSignatureError, ValueError, TypeError):
        return False


def generate_key_file(path: str | os.PathLike) -> SigningKey:
    """Generate a new private key and write its 32-byte seed as hex to a file with mode 600. Refuses to overwrite
    an existing file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sk = SigningKey.generate()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(bytes(sk).hex() + '\n')
    return sk


def load_key_file(path: str | os.PathLike) -> SigningKey:
    seed = bytes.fromhex(Path(path).read_text().strip())
    if len(seed) != 32:
        raise ValueError('key file must hold a 32-byte hex seed')
    return SigningKey(seed)
