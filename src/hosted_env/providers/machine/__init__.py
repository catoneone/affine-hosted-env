"""MachineProvider implementations (architecture v2): whole machines that join the cluster as sandbox nodes.

- static: machines registered by hand (count only, never created or deleted).
- boat, targon: planned (vendor APIs; creating machines costs money).
"""
from .base import NAME_PREFIX, Machine, MachineError, MachineProvider, MachineSpec, NotSupported, Offer
from .static import StaticProvider

__all__ = ['NAME_PREFIX', 'Machine', 'MachineError', 'MachineProvider', 'MachineSpec', 'NotSupported', 'Offer',
           'StaticProvider']
