"""coordinator/planner.py with fake providers, plus Store.capacity_snapshot."""
import pytest
from test_nodes import node_store

from hosted_env.coordinator.nodes import NodeInfo
from hosted_env.coordinator.planner import Planner, PlannerLimits
from hosted_env.providers.machine import Machine, MachineError, Offer


class FakeProvider:
    def __init__(self, name, offers, fail_create=False):
        self.name, self._offers, self.fail_create = name, offers, fail_create
        self.machines, self.created, self.deleted = {}, [], []

    def offers(self):
        return list(self._offers)

    def list(self):
        return list(self.machines.values())

    def create(self, spec, boot_script):
        self.created.append((spec, boot_script))
        mid = f'{self.name}-{len(self.created)}'
        self.machines[mid] = Machine(mid, self.name, spec.name, 'running', spec.kind)
        if self.fail_create:
            raise MachineError('boot failed')
        return self.machines[mid]

    def delete(self, mid):
        self.deleted.append(mid)
        self.machines.pop(mid, None)


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def snap(waiting=0, used=0, nodes=None):
    return {'waiting': waiting, 'used': used, 'nodes': nodes or {}, 'group_size': 8}


def planner(limits, providers=None, dry_run=False):
    providers = providers or {'boat': FakeProvider('boat', [Offer('large', 8, 16, 125, 0.072)]),
                              'targon': FakeProvider('targon', [Offer('cpu-small-amd', 4, 52, 384, 0.15, 6)])}
    return Planner(providers, limits, boot_script=lambda p, spec: f'boot {spec.name}', dry_run=dry_run, clock=Clock())


def test_slots_and_cheapest_offer():
    p = planner(PlannerLimits())
    assert p.slots_of(Offer('large', 8, 16, 125, 0.072)) == 7          # cpu 7.5 / memory 7.5 -> 7
    assert p.slots_of(Offer('cpu-small-amd', 4, 52, 384, 0.15)) == 3   # cpu bound
    assert p._cheapest()[:2] == ('boat', Offer('large', 8, 16, 125, 0.072))   # $0.010/slot beats $0.050/slot


def test_caps_bound_every_plan():
    assert planner(PlannerLimits()).plan(snap(waiting=10)) == []        # caps default to zero: never create
    acts = planner(PlannerLimits(max_machines=10, max_hourly_usd=10, max_create_per_round=2)).plan(snap(waiting=3))
    assert [a.kind for a in acts] == ['create', 'create'] and all(a.machine_kind == 'large' for a in acts)
    acts = planner(PlannerLimits(max_machines=10, max_hourly_usd=0.1, max_create_per_round=5)).plan(snap(waiting=3))
    assert len(acts) == 1                                               # $0.1/h buys one $0.072/h machine
    assert planner(PlannerLimits(max_machines=1, max_hourly_usd=9)).plan(snap(waiting=9))[0].name.startswith(
        'hosted-env-boat-')


def test_dry_run_calls_nothing():
    p = planner(PlannerLimits(max_machines=5, max_hourly_usd=5), dry_run=True)
    res = p.step(snap(waiting=1))
    assert res and all(r == 'dry_run' for _, r in res)
    assert p.providers['boat'].created == [] and p.owned == {}
    assert planner(PlannerLimits()).dry_run is False and Planner({}, PlannerLimits(), lambda *a: '').dry_run is True


def test_create_join_idle_delete_and_min_machines():
    p = planner(PlannerLimits(max_machines=4, max_hourly_usd=4, max_create_per_round=2, idle_grace=600,
                              min_machines=1))
    res = p.step(snap(waiting=2))
    assert [r for _, r in res] == ['created boat-1', 'created boat-2']
    names = sorted(p.owned)
    assert p.providers['boat'].created[0][1] == f'boot {p.providers["boat"].created[0][0].name}'
    joined = {n: {'slots': 7, 'used': 0} for n in names}
    assert p.step(snap(nodes=joined)) == []                             # empty, but not for idle_grace yet
    p.clock.t += 601
    res = p.step(snap(nodes=joined))
    assert [a.kind for a, _ in res] == ['delete'] and len(p.owned) == 1  # min_machines keeps one
    p.clock.t += 601
    assert p.step(snap(nodes={n: joined[n] for n in p.owned})) == []


def test_busy_or_waiting_nodes_are_kept():
    p = planner(PlannerLimits(max_machines=3, max_hourly_usd=3, idle_grace=0))
    assert len(p.step(snap(waiting=1))) == 2                            # 8 slots needed, 7 per machine
    busy, idle = sorted(p.owned)
    nodes = {busy: {'slots': 7, 'used': 8}, idle: {'slots': 7, 'used': 0}}
    acts = p.plan(snap(waiting=1, used=8, nodes=nodes))
    assert [a.kind for a in acts] == ['create']                         # someone waits: grow, delete nothing
    acts = p.plan(snap(waiting=0, used=8, nodes=nodes))
    assert [(a.kind, a.name) for a in acts] == [('delete', idle)]       # the busy node stays


def test_never_joined_is_deleted_and_failed_create_cleans_up():
    p = planner(PlannerLimits(max_machines=2, max_hourly_usd=2, max_create_per_round=1, join_timeout=100))
    p.step(snap(waiting=1))
    p.clock.t += 101
    acts = p.plan(snap(waiting=0))
    assert [(a.kind, a.reason) for a in acts] == [('delete', 'never joined the cluster')]
    bad = FakeProvider('boat', [Offer('large', 8, 16, 125, 0.072)], fail_create=True)
    p2 = planner(PlannerLimits(max_machines=2, max_hourly_usd=2, max_create_per_round=1), providers={'boat': bad})
    [(a, r)] = p2.step(snap(waiting=1))
    assert r.startswith('failed') and bad.deleted == ['boat-1'] and p2.owned == {}


def test_store_capacity_snapshot(tmp_path):
    st = node_store(tmp_path)
    from test_coordinator_store import EPOCH, NOW

    from hosted_env.coordinator.store import CapacityError
    st.sync_nodes([NodeInfo('n1', 8, ())], now=NOW)
    st.open_task(EPOCH, '5H0', 'env', now=NOW)
    with pytest.raises(CapacityError):
        st.open_task(EPOCH, '5H1', 'env', now=NOW)                       # full: 5H1 is now waiting
    s = st.capacity_snapshot(now=NOW)
    assert (s['waiting'], s['used'], s['nodes'], s['group_size']) == (1, 8, {'n1': {'slots': 8, 'used': 8}}, 8)
