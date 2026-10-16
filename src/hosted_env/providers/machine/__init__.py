"""MachineProvider implementations (architecture v2): whole machines that join the cluster as sandbox nodes.

- static: machines registered by hand (count only, never created or deleted).
- boat: boat.dev sandboxes (REST API; creating machines costs money).
- targon: Targon VMs (REST API; SHARED networking remaps ports; boot script runs over SSH).
"""
from .base import (NAME_PREFIX, BootScript, CreateFailed, Machine, MachineError, MachineProvider, MachineSpec,
                   NotSupported, Offer, resolve_boot_script)
from .boat import BoatProvider
from .static import StaticProvider
from .targon import TargonProvider, ssh_boot_runner

__all__ = ['NAME_PREFIX', 'BoatProvider', 'BootScript', 'CreateFailed', 'Machine', 'MachineError', 'MachineProvider',
           'MachineSpec', 'NotSupported', 'Offer', 'StaticProvider', 'TargonProvider', 'resolve_boot_script',
           'ssh_boot_runner']
