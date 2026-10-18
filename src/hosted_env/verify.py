"""Stable import path for group verification (docs/formats.md): this name is hosted_env.infra.verify itself, so
`from hosted_env.verify import verify_group` keeps working whatever the package layout does."""
import sys

from .infra import verify as _verify

sys.modules[__name__] = _verify
