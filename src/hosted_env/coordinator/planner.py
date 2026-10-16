"""Capacity planner (architecture v2): turns waiting demand into machines, and idle machines back into nothing.

Each round: needed slots = slots in use + one group per hotkey waiting for capacity; supply = node slots plus the
slots of machines created but not yet joined. A deficit buys the cheapest offer per slot; an empty node that stays
empty for `idle_grace` seconds (while nobody waits) is deleted. Only machines this planner created are ever
deleted (static machines, the control plane and other workloads are never touched).

Every machine is kept in an inventory (the coordinator Store's `machines` table, with history):
- the planner's own machines are recorded as soon as the provider assigns an id, before the slow boot, so a crash
  mid-create cannot forget a machine that bills; a restart reloads them and can still delete them;
- a failed create is cleaned up by the provider, which only deletes the machine it was creating; if that delete
  fails too the machine is marked `orphaned` and the planner retries the delete every round;
- every `inventory_interval` seconds each provider is listed (read only, also in dry-run mode): machines with the
  `hosted-env-` prefix that the planner did not create (the control plane, leftovers from a lost database) are
  recorded as `unmanaged` and logged, but never deleted automatically.
`hosted-env machines --config <file>` prints the inventory.

Spending money is guarded twice: hard caps (`max_machines`, `max_hourly_usd`, `max_create_per_round`) that the plan
never exceeds, and `dry_run=True` by default, which returns the plan without creating or deleting anything.
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from ..providers.machine import CreateFailed, MachineError, MachineSpec, Offer
from .nodes import cpu_cores, mem_bytes

log = logging.getLogger('hosted_env.planner')


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
    slots: int = 0


@dataclass
class Owned:
    provider: str
    machine_id: str
    name: str                        # also the k3s node name (the boot script sets --node-name)
    price_per_hour: float
    slots: int
    created_at: float
    idle_since: float | None = None
    state: str = 'running'           # creating / running / orphaned, as in the inventory


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
    ledger: object = None            # Store: machines / machine_created / machine_set_state / machine_seen / ...
    inventory_interval: float = 300  # seconds between provider listings (read only)
    owned: dict = field(default_factory=dict)       # name -> Owned
    inventory_last: dict | None = field(default=None, init=False)
    _inventory_at: float = field(default=float('-inf'), init=False, repr=False)

    def __post_init__(self):
        for r in (self.ledger.machines(origin='planner') if self.ledger is not None else []):
            self.owned.setdefault(r['name'], Owned(r['provider'], r['machine_id'], r['name'], r['price_per_hour'],
                                                   r['slots'], r['created_at'], state=r['state']))

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
        joined = {n for n, o in self.owned.items() if n in nodes and o.state != 'orphaned'}
        orphaned = {n: o for n, o in self.owned.items() if o.state == 'orphaned'}
        pending = {n: o for n, o in self.owned.items() if n not in nodes and n not in orphaned}
        need = snap['used'] + snap['waiting'] * snap['group_size']
        supply = sum(v['slots'] for v in nodes.values()) + sum(o.slots for o in pending.values())

        # our failed creates the provider could not clean up: they may bill, keep trying
        for name, o in orphaned.items():
            actions.append(Action('delete', o.provider, 'create failed and the provider could not delete it',
                                  machine_id=o.machine_id, name=name))
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
                                          machine_kind=offer.kind, name=name, price_per_hour=offer.price_per_hour,
                                          slots=n))
        # shrink: owned nodes that have been empty for idle_grace while nobody waits
        keep = len(self.owned) - len(orphaned)
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
            try:
                results.append((a, self._create(a) if a.kind == 'create' else self._delete(a)))
            except MachineError as e:
                results.append((a, f'failed: {e}'))
        return results

    def _create(self, a: Action) -> str:
        p, now = self.providers[a.provider], self.clock()
        spec = MachineSpec(a.machine_kind, a.name)

        def on_created(machine_id: str) -> None:       # before the boot: a crash from here on cannot lose it
            self.owned[a.name] = Owned(a.provider, machine_id, a.name, a.price_per_hour, a.slots, now,
                                       state='creating')
            if self.ledger is not None:
                self.ledger.machine_created(a.provider, machine_id, a.name, a.machine_kind, a.price_per_hour,
                                            a.slots, now)
        try:
            m = p.create(spec, self.boot_script(a.provider, spec), on_created=on_created)
        except CreateFailed as e:
            if e.cleaned:
                self.owned.pop(a.name, None)
                self._set_state(a.provider, e.machine_id, 'deleted', f'create failed: {e}')
                return f'failed: {e}'
            self.owned.setdefault(a.name, Owned(a.provider, e.machine_id, a.name, a.price_per_hour, a.slots, now))
            self.owned[a.name].state = 'orphaned'
            self._set_state(a.provider, e.machine_id, 'orphaned', f'create failed: {e}; provider cleanup failed')
            log.warning('machine %s (%s on %s) failed to create and could not be deleted; retrying every round',
                        a.name, e.machine_id, a.provider)
            return f'failed: {e} (cleanup failed; the delete is retried)'
        if a.name not in self.owned:                   # a provider that did not report the id early
            on_created(m.id)
        o = self.owned[a.name]
        o.machine_id, o.state = m.id, 'running'
        if self.ledger is not None:
            self.ledger.machine_set_state(a.provider, m.id, 'running', self.clock())
            self.ledger.machine_seen(a.provider, m, self.clock())
        return f'created {m.id}'

    def _delete(self, a: Action) -> str:
        try:
            self.providers[a.provider].delete(a.machine_id)
        except MachineError as e:
            if self.ledger is not None:
                self.ledger.machine_note(a.provider, a.machine_id, f'delete failed: {e}')
            raise
        self.owned.pop(a.name, None)
        self._set_state(a.provider, a.machine_id, 'deleted', a.reason)
        return 'deleted'

    def _set_state(self, provider: str, machine_id: str, state: str, note: str) -> None:
        if self.ledger is not None:
            self.ledger.machine_set_state(provider, machine_id, state, self.clock(), note)

    def reconcile(self) -> dict:
        """List every provider (read only) and record what exists in the inventory. Unknown machines with the
        `hosted-env-` prefix become `unmanaged` (logged once, never deleted); machines we do not own that are no
        longer listed become `gone`. A provider whose list fails is skipped this time."""
        now = self.clock()
        report = {'at': now, 'new': [], 'gone': [], 'errors': {}}
        if self.ledger is None:
            return report
        for pname, p in self.providers.items():
            try:
                ms = p.list()
            except MachineError as e:
                report['errors'][pname] = str(e)
                log.warning('inventory: listing %s failed: %s', pname, e)
                continue
            origin = 'static' if pname == 'static' else 'unmanaged'
            for m in ms:
                if self.ledger.machine_seen(pname, m, now, origin=origin):
                    report['new'].append((pname, m.id, m.name))
                    if origin == 'unmanaged':
                        log.warning('inventory: %s %s (%s) has the hosted-env- prefix but was not created by the '
                                    'planner; counted, never deleted automatically', pname, m.id, m.name)
            for r in self.ledger.machines_missing(pname, [m.id for m in ms], now):
                report['gone'].append((pname, r['machine_id'], r['name']))
                log.info('inventory: %s %s (%s) is no longer listed; marked gone', pname, r['machine_id'], r['name'])
        self.inventory_last = report
        return report

    def step(self, snap: dict) -> list[tuple[Action, str]]:
        if self.ledger is not None and self.clock() - self._inventory_at >= self.inventory_interval:
            self._inventory_at = self.clock()
            self.reconcile()
        for name, o in self.owned.items():             # a create interrupted by a restart whose machine joined anyway
            if o.state == 'creating' and name in snap['nodes']:
                o.state = 'running'
                self._set_state(o.provider, o.machine_id, 'running', 'joined the cluster')
        return self.apply(self.plan(snap))
