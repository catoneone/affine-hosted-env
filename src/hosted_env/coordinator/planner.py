"""Capacity planner (architecture v2): turns waiting demand into machines, and idle machines back into nothing.

Each round: needed slots = slots in use + one group per hotkey waiting for capacity; supply = node slots plus the
slots of machines created but not yet joined. A deficit buys the cheapest offer per slot; an empty node that stays
empty for `idle_grace` seconds (while nobody waits) is deleted. Only machines this planner created are ever
deleted (static machines and other workloads are never touched).

Spending money is guarded twice: hard caps (`max_machines`, `max_hourly_usd`, `max_create_per_round`) that the plan
never exceeds, and `dry_run=True` by default, which returns the plan without calling any provider.
"""
from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from ..providers.machine import MachineError, MachineSpec, Offer
from .nodes import cpu_cores, mem_bytes


@dataclass(frozen=True)
class PlannerLimits:
    max_machines: int = 0            # machines this planner may own at once; 0 = never create
    max_hourly_usd: float = 0.0      # summed hourly price of those machines; 0 = never create
    max_create_per_round: int = 2
    min_machines: int = 0            # never shrink below this many owned machines
    idle_grace: float = 600          # seconds a node must stay empty before it is deleted
    join_timeout: float = 1800       # a created machine that never shows up as a node is deleted after this


@dataclass(frozen=True)
class Action:
    kind: str                        # create / delete
    provider: str
    reason: str
    machine_kind: str = ''
    machine_id: str = ''
    name: str = ''
    price_per_hour: float = 0.0


@dataclass
class Owned:
    provider: str
    machine_id: str
    name: str                        # also the k3s node name (the boot script sets --node-name)
    price_per_hour: float
    slots: int
    created_at: float
    idle_since: float | None = None


@dataclass
class Planner:
    providers: dict                  # name -> MachineProvider
    limits: PlannerLimits
    boot_script: Callable            # (provider_name, MachineSpec) -> str | Callable[[Machine], str]
    cpus: str = '1'                  # one standard sandbox slot, as in [placement]
    memory: str = '2g'
    reserve_cpu: str = '500m'
    reserve_memory: str = '1Gi'
    dry_run: bool = True
    clock: Callable[[], float] = time.time
    owned: dict = field(default_factory=dict)       # name -> Owned

    def slots_of(self, o: Offer) -> int:
        cpu = (o.vcpus - cpu_cores(self.reserve_cpu)) / cpu_cores(self.cpus)
        mem = (o.memory_gb * 2 ** 30 - mem_bytes(self.reserve_memory)) / mem_bytes(self.memory)
        return max(0, math.floor(min(cpu, mem) + 1e-9))

    def _cheapest(self) -> tuple[str, Offer, int] | None:
        best = None
        for pname, p in self.providers.items():
            try:
                offers = p.offers()
            except MachineError:
                continue
            for o in offers:
                n = self.slots_of(o)
                if n <= 0 or o.available == 0 or o.price_per_hour <= 0:
                    continue
                key = (o.price_per_hour / n, -n, pname, o.kind)
                if best is None or key < best[0]:
                    best = (key, pname, o, n)
        return best and best[1:]

    def plan(self, snap: dict) -> list[Action]:
        now, lim = self.clock(), self.limits
        nodes, actions = snap['nodes'], []
        joined = {n for n in self.owned if n in nodes}
        pending = {n: o for n, o in self.owned.items() if n not in nodes}
        need = snap['used'] + snap['waiting'] * snap['group_size']
        supply = sum(v['slots'] for v in nodes.values()) + sum(o.slots for o in pending.values())

        # machines that never joined
        for name, o in pending.items():
            if now - o.created_at > lim.join_timeout:
                actions.append(Action('delete', o.provider, 'never joined the cluster', machine_id=o.machine_id,
                                      name=name))
        # grow
        if need > supply:
            best = self._cheapest()
            if best:
                pname, offer, n = best
                spend = sum(o.price_per_hour for o in self.owned.values())
                want = math.ceil((need - supply) / n)
                can = min(want, lim.max_create_per_round, lim.max_machines - len(self.owned),
                          math.floor((lim.max_hourly_usd - spend) / offer.price_per_hour + 1e-9))
                for _ in range(max(0, can)):
                    name = f'hosted-env-{pname}-{uuid.uuid4().hex[:6]}'
                    actions.append(Action('create', pname, f'need {need} slots, have {supply}',
                                          machine_kind=offer.kind, name=name, price_per_hour=offer.price_per_hour))
        # shrink: owned nodes that have been empty for idle_grace while nobody waits
        keep = len(self.owned)
        for name in sorted(joined):
            o = self.owned[name]
            empty = nodes[name]['used'] == 0 and snap['waiting'] == 0
            o.idle_since = (o.idle_since or now) if empty else None
            if empty and now - o.idle_since >= lim.idle_grace and keep > lim.min_machines:
                actions.append(Action('delete', o.provider, f'idle for {now - o.idle_since:.0f}s',
                                      machine_id=o.machine_id, name=name))
                keep -= 1
        return actions

    def apply(self, actions: list[Action]) -> list[tuple[Action, str]]:
        """Carry out a plan. In dry-run mode nothing is called and every action reports 'dry_run'."""
        results = []
        for a in actions:
            if self.dry_run:
                results.append((a, 'dry_run'))
                continue
            p = self.providers[a.provider]
            try:
                if a.kind == 'create':
                    spec = MachineSpec(a.machine_kind, a.name)
                    m = p.create(spec, self.boot_script(a.provider, spec))
                    offer = next((o for o in p.offers() if o.kind == a.machine_kind), None)
                    self.owned[a.name] = Owned(a.provider, m.id, a.name, a.price_per_hour,
                                               self.slots_of(offer) if offer else 0, self.clock())
                    results.append((a, f'created {m.id}'))
                else:
                    p.delete(a.machine_id)
                    self.owned.pop(a.name, None)
                    results.append((a, 'deleted'))
            except MachineError as e:
                results.append((a, f'failed: {e}'))
                if a.kind == 'create':
                    self._delete_orphans(a.provider)
        return results

    def _delete_orphans(self, pname: str) -> None:
        """A failed create can leave a billed machine behind; delete provider machines this planner does not
        track (providers only list hosted-env machines)."""
        p, mine = self.providers[pname], {o.machine_id for o in self.owned.values()}
        try:
            for m in p.list():
                if m.id not in mine:
                    p.delete(m.id)
        except MachineError:
            pass

    def step(self, snap: dict) -> list[tuple[Action, str]]:
        return self.apply(self.plan(snap))
