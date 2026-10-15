"""coordinator/planner.py with fake providers, plus Store.capacity_snapshot."""
import pytest
from test_nodes import node_store

from hosted_env.coordinator.nodes import NodeInfo
from hosted_env.coordinator.planner import Planner, PlannerLimits
from hosted_env.providers.machine import Machine, MachineError, MachineSpec, Offer


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


def test_never_joined_is_deleted_and_failed_create_touches_nothing_else():
    p = planner(PlannerLimits(max_machines=2, max_hourly_usd=2, max_create_per_round=1, join_timeout=100))
    p.step(snap(waiting=1))
    p.clock.t += 101
    acts = p.plan(snap(waiting=0))
    assert [(a.kind, a.reason) for a in acts] == [('delete', 'never joined the cluster')]
    bad = FakeProvider('boat', [Offer('large', 8, 16, 125, 0.072)], fail_create=True)
    bad.machines['cp'] = Machine('cp', 'boat', 'hosted-env-cp1', 'running', 'large')   # e.g. the control plane
    p2 = planner(PlannerLimits(max_machines=2, max_hourly_usd=2, max_create_per_round=1), providers={'boat': bad})
    [(a, r)] = p2.step(snap(waiting=1))
    assert r.startswith('failed') and p2.owned == {}
    assert bad.deleted == [] and 'cp' in bad.machines                   # cleanup is the provider's job, not ours


def test_owned_machines_survive_a_restart(tmp_path):
    st = node_store(tmp_path)
    lim = PlannerLimits(max_machines=2, max_hourly_usd=2, max_create_per_round=1, idle_grace=0)
    provs = {'boat': FakeProvider('boat', [Offer('large', 8, 16, 125, 0.072)])}
    p = Planner(provs, lim, boot_script=lambda pn, spec: '', dry_run=False, clock=Clock(), ledger=st)
    p.step(snap(waiting=1))
    [name] = p.owned
    assert st.planner_machines()[0]['name'] == name and st.planner_machines()[0]['slots'] == 7
    p2 = Planner(provs, lim, boot_script=lambda pn, spec: '', dry_run=False, clock=Clock(), ledger=st)
    assert p2.owned[name].machine_id == 'boat-1'                        # remembered, so still deletable
    [(a, r)] = p2.step(snap(nodes={name: {'slots': 7, 'used': 0}}))
    assert (a.kind, r) == ('delete', 'deleted') and st.planner_machines() == [] and provs['boat'].machines == {}


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


def test_config_machines_defaults_are_safe(tmp_path):
    from nacl.signing import SigningKey

    from hosted_env.config import _planner, _store
    (tmp_path / 'boat.token').write_text('tok\n')
    (tmp_path / 'k3s.token').write_text('K10join::server:s3cr3t\n')
    cfg = {'db': str(tmp_path / 'g.db'), 'placement': {'kind': 'k8s', 'cpus': '2', 'memory': '4g'},
           'machines': {'server_url': 'https://160.0.0.1:20005', 'join_token_file': 'k3s.token',
                        'wireguard_port': 20000, 'providers': {'boat': {'token_file': 'boat.token'}}}}
    st = _store(cfg, tmp_path, SigningKey.generate())
    p, interval = _planner(cfg, tmp_path, st)
    assert p.dry_run is True and interval == 30 and list(p.providers) == ['boat'] and p.ledger is st
    assert (p.limits.max_machines, p.limits.max_hourly_usd) == (0, 0.0) and (p.cpus, p.memory) == ('2', '4g')
    s = p.boot_script('boat', MachineSpec('large', 'hosted-env-boat-abc123'))
    assert 'K10join::server:s3cr3t' in s and '--node-name hosted-env-boat-abc123' in s
    assert 'K3S_URL=https://160.0.0.1:20005' in s and '--node-label hosted-env/provider=boat' in s
    assert '"ListenPort": 20000' in s
    assert _planner({'db': cfg['db']}, tmp_path, st) == (None, 0)
    with pytest.raises(ValueError, match='placement'):
        _planner({**cfg, 'placement': {}}, tmp_path, st)
    with pytest.raises(ValueError, match='unknown machine provider'):
        _planner({**cfg, 'machines': {**cfg['machines'], 'providers': {'aws': {}}}}, tmp_path, st)


def test_gateway_runs_the_planner(tmp_path):
    import json
    import time

    from fastapi.testclient import TestClient
    from nacl.signing import SigningKey
    from test_coordinator_store import EPOCH

    from hosted_env.gateway.app import Roster, create_gateway_app
    roster = tmp_path / 'roster.json'
    roster.write_text(json.dumps({'epoch': EPOCH, 'hotkeys': {}}))
    st = node_store(tmp_path)
    p = planner(PlannerLimits(max_machines=1, max_hourly_usd=1), dry_run=True)
    seen = []
    step = p.step

    def counting_step(snap):
        seen.append(snap)
        if len(seen) == 1:
            raise RuntimeError('provider api down')                      # one bad round must not stop the loop
        return step(snap)
    p.step = counting_step
    app = create_gateway_app(st, Roster(str(roster)), {}, SigningKey.generate().verify_key, planner=p,
                             planner_interval=0.02)
    with TestClient(app):
        for _ in range(100):
            if getattr(app.state, 'planner_last', None):
                break
            time.sleep(0.02)
    last = app.state.planner_last
    assert len(seen) >= 2 and last['dry_run'] is True and last['results'] == []          # nobody waiting
    assert p.providers['boat'].created == []
