"""coordinator/nodes.py (capacity from node status) and Store(placement='nodes')."""
import random
from types import SimpleNamespace as NS

import pytest
from nacl.signing import SigningKey
from test_coordinator_store import BIG, EPOCH, NOW, make_env

from hosted_env.coordinator.nodes import NodeInfo, cpu_cores, image_names, k8s_nodes, mem_bytes, node_info
from hosted_env.coordinator.store import CapacityError, Limits, Store


def knode(name, cpu='8', mem='16Gi', ready='True', unschedulable=False, taints=(), images=()):
    return NS(metadata=NS(name=name),
              spec=NS(unschedulable=unschedulable, taints=[NS(key=k, effect=e) for k, e in taints]),
              status=NS(allocatable={'cpu': cpu, 'memory': mem}, conditions=[NS(type='Ready', status=ready)],
                        images=[NS(names=list(n)) for n in images]))


def test_quantities():
    assert (cpu_cores('3500m'), cpu_cores('4'), cpu_cores('0.5')) == (3.5, 4.0, 0.5)
    assert mem_bytes('16Gi') == 16 * 2 ** 30 and mem_bytes('2g') == 2 * 2 ** 30 and mem_bytes('512Mi') == 2 ** 29
    assert mem_bytes('8G') == 8 * 10 ** 9 and mem_bytes('1048576') == 2 ** 20
    assert image_names(['docker.io/x/y:t', 'docker.io/x/y@sha256:ab']) >= {'x/y:t', 'docker.io/x/y:t'}
    assert 'docker.io/x/y:t' in image_names(['x/y:t']) and image_names(['ghcr.io/a/b:1']) == {'ghcr.io/a/b:1'}


def test_node_info_slots_and_filters():
    n = node_info(knode('a', cpu='8', mem='16Gi', images=[['docker.io/x/y:t']]), cpus='1', memory='2g')
    assert n == NodeInfo('a', 7, ('docker.io/x/y:t', 'x/y:t'))          # cpu 7.5 -> 7, memory 15Gi -> 7
    assert node_info(knode('b', cpu='16', mem='8Gi'), cpus='1', memory='2g').slots == 3   # memory bound
    assert node_info(knode('c', ready='False')) is None
    assert node_info(knode('d', unschedulable=True)) is None
    assert node_info(knode('e', taints=[('gpu', 'NoSchedule')])) is None
    cp = knode('f', taints=[('node-role.kubernetes.io/control-plane', 'NoSchedule')])
    assert node_info(cp) is None and node_info(cp, include_control_plane=True) is not None
    assert node_info(knode('g', taints=[('x', 'PreferNoSchedule')])) is not None
    core = NS(list_node=lambda: NS(items=[knode('a'), knode('c', ready='False')]))
    assert [x.name for x in k8s_nodes(core)] == ['a']


def node_store(tmp_path, n=40, limits=None):
    env = make_env(tmp_path, n=n)
    st = Store(str(tmp_path / 'g.db'), {'env': env}, SigningKey.generate(),
               limits or Limits(worker_ttl=BIG, node_ttl=120), rng=random.Random(0), placement='nodes')
    st.heartbeat('local', 'http://gw', 100_000, [], ['env'], now=NOW)      # one in-process executor routes
    return st


def test_nodes_placement_capacity_and_cache(tmp_path):
    st = node_store(tmp_path)
    with pytest.raises(CapacityError, match='capacity_exhausted'):
        st.open_task(EPOCH, '5H0', 'env', now=NOW)                        # no nodes yet
    st.sync_nodes([NodeInfo('n1', 16, ()), NodeInfo('n2', 8, ('img-7',))], now=NOW)
    t = st.open_task(EPOCH, '5H0', 'env', now=NOW)
    assert t['task'] == 't007' and t['worker_url'] == 'http://gw'          # cached image wins
    assert st.worker_task_spec('local', t['task_id'])['node'] == 'n2'
    t2, t3 = st.open_task(EPOCH, '5H1', 'env', now=NOW), st.open_task(EPOCH, '5H2', 'env', now=NOW)
    assert {st.worker_task_spec('local', x['task_id'])['node'] for x in (t2, t3)} == {'n1'}   # n2 is full
    with pytest.raises(CapacityError, match='capacity_exhausted'):
        st.open_task(EPOCH, '5H3', 'env', now=NOW)                        # 16 + 8 slots, 3 groups of 8
    st.finish_task(EPOCH, '5H0', t['task_id'], now=NOW)
    assert st.worker_task_spec('local', st.open_task(EPOCH, '5H3', 'env', now=NOW)['task_id'])['node'] == 'n2'


def test_node_lost_voids_its_tasks(tmp_path):
    st = node_store(tmp_path)
    st.sync_nodes([NodeInfo('n1', 8, ()), NodeInfo('n2', 8, ())], now=NOW)
    a, b = st.open_task(EPOCH, '5H0', 'env', now=NOW), st.open_task(EPOCH, '5H1', 'env', now=NOW)
    node_a = st.worker_task_spec('local', a['task_id'])['node']
    other = 'n2' if node_a == 'n1' else 'n1'
    st.sync_nodes([NodeInfo(other, 8, ())], now=NOW + 100)                # node_a stops appearing
    st.reap(now=NOW + 150)
    states = dict(st.db.execute('SELECT task_id, end_reason FROM tasks').fetchall())
    assert states[a['task_id']] == 'node_lost' and states[b['task_id']] is None
    st.sync_nodes([NodeInfo(other, 8, ())], now=NOW + 200)
    with pytest.raises(CapacityError):                                    # the surviving node is still full
        st.open_task(EPOCH, '5H2', 'env', now=NOW + 200)


def test_workers_placement_unchanged(tmp_path):
    env = make_env(tmp_path)
    st = Store(str(tmp_path / 'w.db'), {'env': env}, SigningKey.generate(), Limits(worker_ttl=BIG))
    st.heartbeat('w1', 'http://w1', 64, [], ['env'], now=NOW)
    t = st.open_task(EPOCH, '5H0', 'env', now=NOW)
    assert st.worker_task_spec('w1', t['task_id'])['node'] is None
    with pytest.raises(ValueError):
        Store(str(tmp_path / 'x.db'), {'env': env}, SigningKey.generate(), placement='pods')


def test_gateway_node_sync_loop(tmp_path):
    import json
    import time

    from fastapi.testclient import TestClient

    from hosted_env.gateway.app import Roster, create_gateway_app
    roster = tmp_path / 'roster.json'
    roster.write_text(json.dumps({'epoch': EPOCH, 'hotkeys': {}}))
    st = node_store(tmp_path)
    calls = []

    def source():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('api down')                               # one failed read must not stop the loop
        return [NodeInfo('n1', 8, ('img-1',))]
    app = create_gateway_app(st, Roster(str(roster)), {}, SigningKey.generate().verify_key, node_source=source,
                             node_sync_interval=0.02)
    with TestClient(app) as c:
        for _ in range(100):
            if st.db.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]:
                break
            time.sleep(0.02)
        assert c.get('/v1/info').status_code == 200
    assert st.db.execute('SELECT name, slots FROM nodes').fetchall() == [('n1', 8)] and len(calls) >= 2


def test_config_placement_k8s(tmp_path):
    from hosted_env.config import _node_source, _store
    cfg = {'db': str(tmp_path / 'g.db'), 'placement': {'kind': 'k8s', 'cpus': '2'}}
    assert _store(cfg, tmp_path, SigningKey.generate()).placement == 'nodes'
    assert callable(_node_source(cfg)) and _node_source({}) is None
    assert _store({'db': str(tmp_path / 'h.db')}, tmp_path, SigningKey.generate()).placement == 'workers'
