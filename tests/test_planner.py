"""coordinator/planner.py with fake providers, plus Store.capacity_snapshot."""
import pytest
from test_nodes import node_store

from hosted_env.coordinator.nodes import NodeInfo
from hosted_env.coordinator.planner import Planner, PlannerLimits
from hosted_env.providers.machine import CreateFailed, Machine, MachineError, MachineSpec, Offer


class FakeProvider:
    """create() follows the provider contract: report the id, boot, and on failure delete only that machine
    (cleanup_fails=True: that delete fails too and the machine stays)."""

    def __init__(self, name, offers, fail_create=False, cleanup_fails=False, during_boot=None):
        self.name, self._offers, self.fail_create, self.cleanup_fails = name, offers, fail_create, cleanup_fails
        self.during_boot = during_boot
        self.machines, self.created, self.deleted = {}, [], []

    def offers(self):
        return list(self._offers)

    def list(self):
        return list(self.machines.values())

    def create(self, spec, boot_script, on_created=None):
        self.created.append((spec, boot_script))
        mid = f'{self.name}-{len(self.created)}'
        self.machines[mid] = Machine(mid, self.name, spec.name, 'running', spec.kind)
        if on_created:
            on_created(mid)
        if self.during_boot:
            self.during_boot(mid)
        if self.fail_create:
            if not self.cleanup_fails:
                self.delete(mid)
            raise CreateFailed('boot failed', machine_id=mid, cleaned=not self.cleanup_fails)
        return self.machines[mid]

    def delete(self, mid):
        self.deleted.append(mid)
        if self.cleanup_fails and self.fail_create:
            raise MachineError('delete refused')
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
    assert bad.deleted == ['boat-1'] and 'cp' in bad.machines           # the provider deletes only its own create


def test_failed_cleanup_becomes_orphaned_and_is_retried(tmp_path):
    st = node_store(tmp_path)
    bad = FakeProvider('boat', [Offer('large', 8, 16, 125, 0.072)], fail_create=True, cleanup_fails=True)
    bad.machines['cp'] = Machine('cp', 'boat', 'hosted-env-cp1', 'running', 'large')
    lim = PlannerLimits(max_machines=1, max_hourly_usd=1, max_create_per_round=1)
    p = Planner({'boat': bad}, lim, boot_script=lambda pn, spec: '', dry_run=False, clock=Clock(), ledger=st)
    [(a, r)] = p.step(snap(waiting=1))
    assert 'cleanup failed' in r and p.owned[a.name].state == 'orphaned'
    [row] = st.machines(origin='planner')
    assert (row['machine_id'], row['state']) == ('boat-1', 'orphaned') and 'cleanup failed' in row['note']
    acts = p.plan(snap(waiting=1))                                       # max_machines is spent: no new create
    assert [(x.kind, x.machine_id) for x in acts] == [('delete', 'boat-1')]
    bad.cleanup_fails = False                                            # the provider recovers
    [(a, r)] = p.step(snap(waiting=0))
    assert r == 'deleted' and p.owned == {} and 'cp' in bad.machines
    assert st.machines(origin='planner') == []
    [hist] = [x for x in st.machines(live=False) if x['machine_id'] == 'boat-1']
    assert hist['state'] == 'deleted' and hist['ended_at'] is not None


def test_machine_is_recorded_before_boot_and_survives_a_crash(tmp_path):
    st = node_store(tmp_path)
    seen = []

    class Crash(BaseException):
        pass

    def crash(mid):                                                      # the process dies during the boot
        seen.append([(r['machine_id'], r['state']) for r in st.machines()])
        raise Crash
    prov = FakeProvider('boat', [Offer('large', 8, 16, 125, 0.072)], during_boot=crash)
    lim = PlannerLimits(max_machines=1, max_hourly_usd=1, max_create_per_round=1, join_timeout=100)
    p = Planner({'boat': prov}, lim, boot_script=lambda pn, spec: '', dry_run=False, clock=Clock(), ledger=st)
    with pytest.raises(Crash):
        p.step(snap(waiting=1))
    assert seen == [[('boat-1', 'creating')]]
    prov.during_boot = None
    clock = Clock()
    p2 = Planner({'boat': prov}, lim, boot_script=lambda pn, spec: '', dry_run=False, clock=clock, ledger=st)
    [o] = p2.owned.values()
    assert (o.machine_id, o.state) == ('boat-1', 'creating')
    assert p2.plan(snap(waiting=1)) == []                                # it counts against max_machines
    clock.t += 101
    [(a, r)] = p2.step(snap(waiting=0))
    assert (a.reason, r) == ('never joined the cluster', 'deleted') and prov.machines == {}


def test_inventory_records_unmanaged_and_gone_machines(tmp_path):
    st = node_store(tmp_path)
    prov = FakeProvider('targon', [Offer('cpu-small-amd', 4, 52, 384, 0.15)])
    prov.machines['wrk-cp'] = Machine('wrk-cp', 'targon', 'hosted-env-cp1', 'running', 'cpu-small-amd',
                                      public_ipv4='160.0.0.1', price_per_hour=0.15)
    clock = Clock()
    lim = PlannerLimits(max_machines=1, max_hourly_usd=1, max_create_per_round=1, idle_grace=0)
    p = Planner({'targon': prov}, lim, boot_script=lambda pn, spec: '', dry_run=False, clock=clock, ledger=st,
                inventory_interval=60)
    [(a, r)] = p.step(snap(waiting=1))                                   # lists first, then creates
    assert r == 'created targon-1' and p.inventory_last['new'] == [('targon', 'wrk-cp', 'hosted-env-cp1')]
    rows = {x['machine_id']: x for x in st.machines()}
    assert (rows['wrk-cp']['origin'], rows['wrk-cp']['public_ipv4']) == ('unmanaged', '160.0.0.1')
    assert (rows['targon-1']['origin'], rows['targon-1']['state']) == ('planner', 'running')
    assert 'wrk-cp' not in {o.machine_id for o in p.owned.values()}      # counted, never the planner's to delete
    p.step(snap(nodes={a.name: {'slots': 3, 'used': 3}}))
    assert p.inventory_last['at'] == 1000.0                              # not listed again before 60 s
    del prov.machines['wrk-cp']                                          # someone deleted the control plane
    hidden = prov.machines.pop('targon-1')                               # and a list misses our own machine
    clock.t += 61
    assert p.step(snap(nodes={a.name: {'slots': 3, 'used': 3}})) == []
    assert p.inventory_last['gone'] == [('targon', 'wrk-cp', 'hosted-env-cp1')]
    states = {x['machine_id']: x['state'] for x in st.machines(live=False)}
    assert states == {'wrk-cp': 'gone', 'targon-1': 'running'}           # ours is never forgotten by a list
    prov.machines['targon-1'] = hidden
    [(a2, r2)] = p.step(snap(nodes={a.name: {'slots': 3, 'used': 0}}))
    assert (a2.kind, r2) == ('delete', 'deleted')


def test_inventory_lists_even_in_dry_run_and_skips_a_failing_provider(tmp_path):
    st = node_store(tmp_path)

    class Down(FakeProvider):
        def list(self):
            raise MachineError('targon GET /orgs: ConnectError')
    up = FakeProvider('boat', [Offer('large', 8, 16, 125, 0.072)])
    up.machines['bx_9'] = Machine('bx_9', 'boat', 'hosted-env-old', 'running', 'large')
    down = Down('targon', [])
    p = Planner({'boat': up, 'targon': down}, PlannerLimits(max_machines=1, max_hourly_usd=1),
                boot_script=lambda pn, spec: '', clock=Clock(), ledger=st)
    assert p.dry_run is True
    p.step(snap(waiting=1))
    assert p.inventory_last['errors'] == {'targon': 'targon GET /orgs: ConnectError'}
    assert [x['machine_id'] for x in st.machines()] == ['bx_9'] and up.created == [] and up.deleted == []


def test_old_planner_ledger_is_migrated(tmp_path):
    import sqlite3
    db = tmp_path / 'old.db'
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE planner_machines (name TEXT PRIMARY KEY, provider TEXT NOT NULL, machine_id TEXT '
                'NOT NULL, price_per_hour REAL NOT NULL, slots INTEGER NOT NULL, created_at REAL NOT NULL)')
    con.execute("INSERT INTO planner_machines VALUES ('hosted-env-boat-a1', 'boat', 'bx_1', 0.072, 7, 500.0)")
    con.commit()
    con.close()
    from nacl.signing import SigningKey

    from hosted_env.coordinator.store import Store
    st = Store(str(db), {}, SigningKey.generate())
    [r] = st.machines()
    assert (r['name'], r['machine_id'], r['origin'], r['state'], r['slots']) == ('hosted-env-boat-a1', 'bx_1',
                                                                                'planner', 'running', 7)
    tables = {t for (t,) in st.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert 'planner_machines' not in tables
    assert Store(str(db), {}, SigningKey.generate()).machines() == st.machines()   # idempotent


def test_cli_machines_prints_the_inventory(tmp_path, capsys):
    import json

    from nacl.signing import SigningKey

    from hosted_env.cli import main
    from hosted_env.coordinator.store import Store
    st = Store(str(tmp_path / 'g.db'), {}, SigningKey.generate())
    st.machine_created('boat', 'bx_1', 'hosted-env-boat-a1', 'large', 0.072, 7, now=1000.0)
    st.machine_set_state('boat', 'bx_1', 'deleted', now=1000.0 + 7200, note='idle for 600s')
    st.machine_created('boat', 'bx_2', 'hosted-env-boat-b2', 'large', 0.072, 7, now=2000.0)
    st.machine_seen('targon', Machine('wrk-cp', 'targon', 'hosted-env-cp1', 'running', 'cpu-small-amd',
                                      public_ipv4='160.0.0.1', price_per_hour=0.15), now=3000.0)
    cfg = tmp_path / 'server.toml'
    cfg.write_text('db = "g.db"\n')
    main(['machines', '--config', str(cfg), '--json'])
    out = json.loads(capsys.readouterr().out)
    assert sorted(r['machine_id'] for r in out['machines']) == ['bx_2', 'wrk-cp']       # live only by default
    main(['machines', '--config', str(cfg), '--all', '--json'])
    rows = {r['machine_id']: r for r in json.loads(capsys.readouterr().out)['machines']}
    assert rows['bx_1']['hours'] == 2.0 and rows['bx_1']['est_cost_usd'] == 0.144
    main(['machines', '--config', str(cfg)])
    text = capsys.readouterr().out
    assert 'hosted-env-cp1' in text and 'unmanaged' in text and 'bx_1' not in text.split('\n\n')[0]
    assert 'live: 2 machines (planner 1, unmanaged 1), $0.222/h' in text
    assert 'planner, all time: 2 machines' in text
    with pytest.raises(ValueError, match='--refresh'):
        main(['machines', '--config', str(cfg), '--refresh'])


def test_owned_machines_survive_a_restart(tmp_path):
    st = node_store(tmp_path)
    lim = PlannerLimits(max_machines=2, max_hourly_usd=2, max_create_per_round=1, idle_grace=0)
    provs = {'boat': FakeProvider('boat', [Offer('large', 8, 16, 125, 0.072)])}
    p = Planner(provs, lim, boot_script=lambda pn, spec: '', dry_run=False, clock=Clock(), ledger=st)
    p.step(snap(waiting=1))
    [name] = p.owned
    [row] = st.machines(origin='planner')
    assert (row['name'], row['slots'], row['state']) == (name, 7, 'running')
    p2 = Planner(provs, lim, boot_script=lambda pn, spec: '', dry_run=False, clock=Clock(), ledger=st)
    assert p2.owned[name].machine_id == 'boat-1'                        # remembered, so still deletable
    [(a, r)] = p2.step(snap(nodes={name: {'slots': 7, 'used': 0}}))
    assert (a.kind, r) == ('delete', 'deleted') and st.machines() == [] and provs['boat'].machines == {}
    assert [x['state'] for x in st.machines(live=False)] == ['deleted']   # history is kept


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
