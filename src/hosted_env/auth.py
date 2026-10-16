"""Control-plane identity: request signing / verification, session tokens (design §4).

- Miners sign every gateway request with the supervisor's Ed25519 mining key; the signed content is
  `domain | method | path | identity | timestamp | request id | sha256(body)`. The gateway verifies signatures
  locally against the qualified identity roster from this epoch's opening snapshot. Workers call the gateway's
  internal endpoints with the same format but a different domain, with worker_id as the identity.
- After a session opens successfully the gateway issues a session token; the worker holds only the gateway public
  key to verify tokens.
"""
from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable, Mapping

from nacl.signing import SigningKey, VerifyKey

from .common import b64u, canonical, sha256_hex, sign, unb64u, verify

MINER_DOMAIN = 'affine-hosted-env-miner-request-v1'
WORKER_DOMAIN = 'affine-hosted-env-worker-request-v1'
TOKEN_DOMAIN = b'affine-hosted-env-session-v1|'
CLOCK_SKEW = 60          # allowed request timestamp skew (seconds)
REQID_TTL = 2 * CLOCK_SKEW

H_ID, H_TS, H_REQ, H_SIG = 'X-Identity', 'X-Timestamp', 'X-Request-Id', 'X-Signature'


class AuthError(Exception):
    """Authentication failed; code is for the client and does not leak roster details."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def request_message(domain: str, method: str, path: str, identity: str, ts: int, reqid: str, body: bytes) -> bytes:
    return '|'.join([domain, method.upper(), path, identity, str(ts), reqid, sha256_hex(body)]).encode()


def sign_request(key: SigningKey, method: str, path: str, identity: str, body: bytes, *, domain: str = MINER_DOMAIN,
                 ts: int | None = None, reqid: str | None = None) -> dict:
    """Client side: build the request headers to send."""
    ts = int(time.time()) if ts is None else ts
    reqid = reqid or secrets.token_hex(16)
    sig = key.sign(request_message(domain, method, path, identity, ts, reqid, body)).signature
    return {H_ID: identity, H_TS: str(ts), H_REQ: reqid, H_SIG: b64u(sig)}


class RequestVerifier:
    """keys(identity) returns that identity's public key or None; remember(scope, reqid, now) records the request
    id and returns False if it was seen before in that scope (replay protection). The scope is the domain and the
    signing identity: request ids are only unique per signer, so one signer can never use up another's (for
    example by copying a request id seen on the wire into a request of its own before the original arrives)."""

    def __init__(self, domain: str, keys: Callable[[str], VerifyKey | None],
                 remember: Callable[[str, str, float], bool]):
        self.domain, self.keys, self.remember = domain, keys, remember

    def verify(self, method: str, path: str, headers: Mapping, body: bytes, now: float | None = None) -> str:
        now = time.time() if now is None else now
        try:
            identity, reqid = headers[H_ID], headers[H_REQ]
            ts, sig = int(headers[H_TS]), unb64u(headers[H_SIG])
        except (KeyError, ValueError, TypeError):
            raise AuthError('malformed_auth_headers') from None
        key = self.keys(identity)
        if key is None:
            raise AuthError('not_eligible')            # not among this opening's qualified identities (or not a registered worker)
        if abs(now - ts) > CLOCK_SKEW:
            raise AuthError('stale_timestamp')
        if not (16 <= len(reqid) <= 64):
            raise AuthError('malformed_auth_headers')
        if not verify(key, b'', request_message(self.domain, method, path, identity, ts, reqid, body), sig):
            raise AuthError('bad_signature')
        # Record the reqid only after the signature checks out, so others cannot fill the table with junk requests
        if not self.remember(f'{self.domain}|{identity}', reqid, now):
            raise AuthError('replayed_request')
        return identity


# ---------------------------------------------------------------- session token (signed by gateway, verified by worker)

def issue_token(key: SigningKey, claims: dict) -> str:
    if 'exp' not in claims:
        raise ValueError('token needs exp')
    payload = canonical(claims)
    return b64u(payload) + '.' + b64u(sign(key, TOKEN_DOMAIN, payload))


def verify_token(gateway_pub: VerifyKey, token: str, now: float | None = None) -> dict:
    """Worker side: needs only the gateway public key."""
    now = time.time() if now is None else now
    try:
        p, s = token.split('.')
        payload, sig = unb64u(p), unb64u(s)
    except (ValueError, AttributeError):
        raise AuthError('bad_token') from None
    if not verify(gateway_pub, TOKEN_DOMAIN, payload, sig):
        raise AuthError('bad_token')
    try:
        claims = json.loads(payload)
        exp = claims['exp']
    except (ValueError, KeyError, TypeError):
        raise AuthError('bad_token') from None
    if exp < now:
        raise AuthError('expired_token')
    return claims
