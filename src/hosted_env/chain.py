"""SN120 identity roster from the chain: which hotkeys may mine.

The rule is upstream affine's `ChainAdapter.registrations()` (subnet/chain.py, checked at f07c54e on 2026-10-16). At
one fixed block a hotkey qualifies when
- it holds a UID on the subnet, and the UID -> hotkey and hotkey -> UID maps agree, and
- it revealed a commitment `affine2|activate|<hotkey>|<signature>` whose Ed25519 signature by that hotkey over
  `activation_message(hotkey, netuid)` verifies, and
- it is not the subnet owner.
Affine hotkeys are Ed25519 keys, so the public key that verifies miner requests is the hotkey itself.

`hosted-env roster-sync` writes the result as a roster file (`{"source": "chain", "block", "synced_at",
"hotkeys": {hotkey: {uid, public_key, activate_block}}}`) that the gateway hot-reloads next to its hand-edited
allowlist. Needs the `chain` extra (bittensor, the same SDK upstream uses) only to read the chain; the rule itself
is pure Python + PyNaCl.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time

from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

log = logging.getLogger('hosted_env.chain')

NETUID = 120
OWNER = '5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8'      # upstream subnet/chain.py OWNER
_B58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'


def activation_message(hotkey: str, netuid: int = NETUID) -> bytes:
    """Upstream subnet/chain.py `activation_message`, byte for byte."""
    body = json.dumps({'hotkey': hotkey, 'netuid': netuid}, sort_keys=True, separators=(',', ':')).encode()
    rid = hashlib.sha256(b'affine-registration-v1\0' + body).hexdigest()
    return f'affine-activate|v1|{netuid}|{hotkey}|{rid}'.encode()


def decode_commitment(value) -> str:
    """Upstream subnet/chain.py `decode_commitment`: a SCALE-length-prefixed UTF-8 commitment."""
    raw = bytes.fromhex(value[2:]) if isinstance(value, str) and value.startswith('0x') else (
        value.encode('latin1') if isinstance(value, str) else bytes(value))
    if not raw:
        raise ValueError('empty commitment')
    mode = raw[0] & 3
    offset = (1, 2, 4)[mode] if mode < 3 else 1 + (raw[0] >> 2) + 4
    if len(raw) < offset:
        raise ValueError('truncated SCALE length')
    length = int.from_bytes(raw[:offset], 'little') >> 2 if mode < 3 else int.from_bytes(raw[1:offset], 'little')
    if length != len(raw) - offset:
        raise ValueError('SCALE length mismatch')
    return raw[offset:].decode('utf8')


def ss58_public_key(address: str) -> bytes:
    """The 32-byte public key in an SS58 address (any network prefix); ValueError if malformed."""
    n = 0
    for ch in address:
        i = _B58.find(ch)
        if i < 0:
            raise ValueError('not base58')
        n = n * 58 + i
    raw = n.to_bytes((n.bit_length() + 7) // 8, 'big')
    raw = b'\0' * (len(address) - len(address.lstrip('1'))) + raw
    plen = 1 if raw and raw[0] < 64 else 2
    if len(raw) != plen + 32 + 2:
        raise ValueError('not a 32-byte SS58 address')
    body, check = raw[:-2], raw[-2:]
    if hashlib.blake2b(b'SS58PRE' + body, digest_size=64).digest()[:2] != check:
        raise ValueError('bad SS58 checksum')
    return body[plen:]


def registrations(chain, netuid: int = NETUID, owner: str = OWNER, block: int | None = None) -> tuple[int, dict]:
    """(block, {hotkey: {uid, public_key, activate_block}}) at one fixed block. `chain` is a bittensor Subtensor
    (or anything with `.block` and `.query_map((pallet, storage), params, block=)`). An ambiguous UID map raises
    instead of producing an empty or partial roster."""
    block = int(chain.block) if block is None else int(block)
    uids, keys = {}, {}
    for hotkey, uid in chain.query_map(('SubtensorModule', 'Uids'), [netuid], block=block):
        if not isinstance(hotkey, str) or type(uid) is not int or uid < 0 or hotkey in uids or uid in uids.values():
            raise ValueError('ambiguous fixed-block UID map')
        uids[hotkey] = uid
    for uid, hotkey in chain.query_map(('SubtensorModule', 'Keys'), [netuid], block=block):
        if type(uid) is not int or uid < 0 or not isinstance(hotkey, str) or uid in keys or hotkey in keys.values():
            raise ValueError('ambiguous fixed-block hotkey map')
        keys[uid] = hotkey
    registered = {}
    for hotkey, entries in chain.query_map(('Commitments', 'RevealedCommitments'), [netuid], block=block):
        if hotkey == owner or not isinstance(hotkey, str):
            continue
        for value, activation_block in entries:
            try:
                parts = decode_commitment(value).strip().split('|')
                if len(parts) != 4 or parts[:2] != ['affine2', 'activate'] or parts[2] != hotkey:
                    continue
                signature = base64.urlsafe_b64decode(parts[3] + '=' * (-len(parts[3]) % 4))
                public_key = ss58_public_key(hotkey)
                if len(signature) != 64:
                    continue
                VerifyKey(public_key).verify(activation_message(hotkey, netuid), signature)
                uid = uids.get(hotkey)
                if uid is None or keys.get(uid) != hotkey:
                    continue
                registered[hotkey] = {'uid': int(uid), 'public_key': public_key.hex(),
                                      'activate_block': int(activation_block)}
            except (ValueError, TypeError, KeyError, BadSignatureError):
                continue
    return block, registered


def write_roster(path: str, block: int, hotkeys: dict, *, netuid: int = NETUID, max_drop: float = 0.5,
                 now: float | None = None) -> dict:
    """Write the chain roster atomically (readers never see a partial file). Refuses, keeping the previous file,
    when the hotkey count falls by more than `max_drop` from a previous roster of at least 10: a broken read must
    not lock every miner out. Returns a summary."""
    prev = 0
    try:
        with open(path) as f:
            prev = len(json.load(f).get('hotkeys') or {})
    except (OSError, ValueError):
        pass
    if prev >= 10 and len(hotkeys) < prev * (1 - max_drop):
        raise ValueError(f'roster would drop from {prev} to {len(hotkeys)} hotkeys at block {block}; kept the old one')
    doc = {'source': 'chain', 'netuid': netuid, 'block': block, 'synced_at': time.time() if now is None else now,
           'hotkeys': dict(sorted(hotkeys.items()))}
    tmp = f'{path}.tmp-{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(doc, f)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
    return {'block': block, 'hotkeys': len(hotkeys), 'previous': prev}


def sync_once(out: str, *, network: str = 'finney', netuid: int = NETUID, chain=None) -> dict:
    if chain is None:
        import bittensor as bt
        chain = bt.Subtensor(network=network)
    t0 = time.monotonic()
    block, regs = registrations(chain, netuid)
    res = write_roster(out, block, regs, netuid=netuid)
    return {**res, 'seconds': round(time.monotonic() - t0, 1)}


def sync_forever(out: str, every: float, *, network: str = 'finney', netuid: int = NETUID) -> None:
    """Re-sync every `every` seconds; a failed round is logged and the previous roster stays in place."""
    chain = None
    while True:
        try:
            if chain is None:
                import bittensor as bt
                chain = bt.Subtensor(network=network)
            log.info('roster sync: %s', sync_once(out, netuid=netuid, chain=chain))
        except Exception as e:
            log.warning('roster sync failed (previous roster kept): %s: %s', type(e).__name__, e)
            chain = None                                   # reconnect next round
        time.sleep(every)
