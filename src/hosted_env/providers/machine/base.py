"""MachineProvider (architecture v2): whole machines that boot, join the k3s cluster and become sandbox nodes.

Creating a machine costs money. Providers only do what they are told; the capacity planner decides when to call
`create` / `delete` and enforces hard caps (machine count, hourly spend) and a dry-run default.

Machines created by hosted-env carry the `hosted-env-` name prefix, and `list()` returns only those, so a
provider never touches other workloads in the same account.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

NAME_PREFIX = 'hosted-env-'
STATES = ('pending', 'running', 'stopped', 'deleted', 'error')


class MachineError(RuntimeError):
    """The provider refused or failed a request."""


class NotSupported(MachineError):
    """The provider cannot do this (for example creating machines in the static provider)."""


@dataclass(frozen=True)
class Offer:
    """A machine type the provider can create right now."""
    kind: str                       # provider-specific type, e.g. boat 'large', targon 'cpu-small-amd'
    vcpus: float
    memory_gb: float
    disk_gb: float
    price_per_hour: float
    available: int | None = None    # None = unknown / unlimited
    network: str | None = None      # provider-specific notes, e.g. targon 'SHARED' (ports remapped) or 'PASSTHROUGH'


@dataclass(frozen=True)
class MachineSpec:
    kind: str
    name: str                       # must start with NAME_PREFIX
    ttl_seconds: int | None = None  # provider-side auto-stop or auto-delete, when supported
    ports: tuple[str, ...] = ()     # inbound ports to expose, e.g. ('51820/udp',), when the provider needs it

    def __post_init__(self):
        if not self.name.startswith(NAME_PREFIX):
            raise ValueError(f'machine names must start with {NAME_PREFIX!r}: {self.name!r}')


@dataclass(frozen=True)
class Machine:
    id: str
    provider: str
    name: str
    state: str
    kind: str = ''
    public_ipv4: str | None = None
    public_ipv6: str | None = None
    price_per_hour: float | None = None
    created_at: float | None = None
    ports: dict = field(default_factory=dict)   # requested port -> public endpoint, when the provider remaps

    def __post_init__(self):
        if self.state not in STATES:
            raise ValueError(f'bad machine state {self.state!r}')


class MachineProvider(Protocol):
    name: str

    def offers(self) -> list[Offer]:
        """Machine types that can be created now, with prices."""

    def list(self) -> list[Machine]:
        """Machines this provider created for hosted-env (name prefix), in any state but deleted."""

    def create(self, spec: MachineSpec, boot_script: str) -> Machine:
        """Create and start one machine that runs `boot_script` once at first boot. Costs money."""

    def delete(self, machine_id: str) -> None:
        """Delete the machine and stop billing for it. Idempotent."""
