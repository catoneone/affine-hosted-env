"""Stable import path for the receipt API (docs/formats.md): this name is hosted_env.infra.receipt itself, so
`from hosted_env.receipt import verify_receipt` keeps working whatever the package layout does."""
import sys

from .infra import receipt as _receipt

sys.modules[__name__] = _receipt
