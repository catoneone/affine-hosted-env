"""Environment registry: `kind` in the config selects the task type."""
from __future__ import annotations

from .base import DockerTaskEnv, EnvSession, GradeError, SetupError
from .swe import SweEnv

KINDS = {DockerTaskEnv.kind: DockerTaskEnv, SweEnv.kind: SweEnv}


def make_env(env_id: str, cfg: dict) -> DockerTaskEnv:
    kind = cfg.get('kind', DockerTaskEnv.kind)
    if kind not in KINDS:
        raise ValueError(f'unknown environment kind {kind!r}')
    return KINDS[kind](env_id, cfg)


def receipt_spec(kind: str):
    """The EnvReceiptSpec a verifier passes to verify_receipt / verify_group for receipts of this environment kind
    (the gateway's /v1/info names the kind of each env_id)."""
    if kind not in KINDS:
        raise ValueError(f'unknown environment kind {kind!r}')
    return KINDS[kind].RECEIPT


__all__ = ['DockerTaskEnv', 'EnvSession', 'GradeError', 'KINDS', 'SetupError', 'SweEnv', 'make_env', 'receipt_spec']
