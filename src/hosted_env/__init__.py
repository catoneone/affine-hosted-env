"""SN120 hosted tool-environment service.

Core (validators / trainers only need this layer, which depends only on PyNaCl):
    from hosted_env.receipt import verify_receipt, loss_mask, task_hash
    from hosted_env.verify import verify_group

hosted_env.receipt and hosted_env.verify are the stable import paths (docs/formats.md); they are the modules
hosted_env.infra.receipt and hosted_env.infra.verify under a second name.
"""
__version__ = '0.2.0'      # docs/formats.md: the wire formats frozen at this version (tag v0.2.0)

# the environments register the receipt specs they sign with (infra.receipt.SPECS), so a verifier that passes no
# spec= finds a receipt's env_spec; importing the registry is cheap (tests/test_envs.py keeps it light)
from . import envs as _envs  # noqa: E402,F401
