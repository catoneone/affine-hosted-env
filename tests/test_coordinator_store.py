"""coordinator/store.py: quotas (ported from the EXP-015 prototype tests) + task assignment, global budget, fair share, voiding."""
import json
import random
import threading

import pytest
from nacl.signing import SigningKey

from hosted_env.infra.auth import AuthError, verify_token
from hosted_env.envs import DockerTaskEnv
from hosted_env.infra.coordinator.store import CapacityError, Limits, QuotaError, Store

NOW = 1_800_000_000.0
EPOCH = 'e000001'
BIG = 10 ** 9


def make_env(tmp_path, n=60, group_size=8, images=None, name='env'):
    p = tmp_path / f'{name}.jsonl'
    with open(p, 'w') as f:
        for i in range(n):
            img = images(i) if images else f'img-{i}'
            f.write(json.dumps({'id': f't{i:03d}', 'image': img, 'cwd': '/w', 'problem_statement': f'p{i}',
                                'test_command': 'true'}) + '\n')
    return DockerTaskEnv(name, {'tasks': str(p), 'group_size': group_size, 'require_image_digest': False})


def make_store(tmp_path, limits=None, envs=None, workers=(('w1', 10_000, ()),), db='g.db'):
    envs = envs or {'env': make_env(tmp_path)}
    key = SigningKey.generate()
    st = Store(str(tmp_path / db), envs, key, limits or Limits(worker_ttl=BIG), rng=random.Random(0))
    for wid, cap, imgs in workers:
        st.heartbeat(wid, f'http://{wid}', cap, list(imgs), list(envs), now=NOW)
    return st, key


@pytest.fixture
def st(tmp_path):
    return make_store(tmp_path)[0]


def open_n(st, n, hk='5H0', now=NOW, epoch=EPOCH):
    return [st.open_task(epoch, hk, 'env', now=now) for _ in range(n)]


def close_all(st, task_id, outcomes, worker='w1', now=NOW):
    ss = st.open_sessions(EPOCH, st.db.execute('SELECT hotkey FROM tasks WHERE task_id=?', (task_id,)).fetchone()[0],
                          task_id, list(range(len(outcomes))), now=now)
    for s, (o, r) in zip(ss, outcomes, strict=True):
        st.worker_close(worker, s['session_id'], o, r, now=now)
    return ss


# ---------------------------------------------------------------- task limits: 9 in flight, 20 started per epoch
def test_inflight_nine(st):
    ts = open_n(st, 9)
    with pytest.raises(QuotaError, match='task_quota_inflight'):
        st.open_task(EPOCH, '5H0', 'env', now=NOW)
    st.open_sessions(EPOCH, '5H0', ts[0]['task_id'], [0, 1], now=NOW)    # in-flight tasks can still open nonces
    st.open_task(EPOCH, '5H1', 'env', now=NOW)                          # other hotkeys and the next epoch are unaffected
    st.open_task('e000002', '5H0', 'env', now=NOW)
    st.finish_task(EPOCH, '5H0', ts[3]['task_id'], now=NOW)             # finishing one frees a slot immediately
    st.open_task(EPOCH, '5H0', 'env', now=NOW)
    with pytest.raises(QuotaError, match='task_quota_inflight'):
        st.open_task(EPOCH, '5H0', 'env', now=NOW)


def test_epoch_twenty(st):
    """Counted by "started": finishing or abandoning does not refund the slot."""
    for _ in range(20):
        t = st.open_task(EPOCH, '5H0', 'env', now=NOW)
        st.finish_task(EPOCH, '5H0', t['task_id'], now=NOW)
    with pytest.raises(QuotaError, match='task_quota_epoch'):
        st.open_task(EPOCH, '5H0', 'env', now=NOW)
    st.open_task('e000002', '5H0', 'env', now=NOW)


def test_finished_task_cannot_reopen(st):
    t = st.open_task(EPOCH, '5H0', 'env', now=NOW)
    st.open_sessions(EPOCH, '5H0', t['task_id'], [0], now=NOW)
    st.finish_task(EPOCH, '5H0', t['task_id'], now=NOW)
    with pytest.raises(QuotaError, match='task_finished'):
        st.open_sessions(EPOCH, '5H0', t['task_id'], [1], now=NOW)      # once finished, unopened nonces cannot be opened


def test_task_ends_when_all_nonces_done(st):
    """Once all 8 nonces have results the task ends without finish_task, freeing an in-flight slot."""
    others = open_n(st, 8)
    t = st.open_task(EPOCH, '5H0', 'env', now=NOW)
    ss = st.open_sessions(EPOCH, '5H0', t['task_id'], list(range(8)), now=NOW)
    for s in ss[:7]:
        st.worker_close('w1', s['session_id'], 'completed', 0.0, now=NOW)
    with pytest.raises(QuotaError, match='task_quota_inflight'):
        st.open_task(EPOCH, '5H0', 'env', now=NOW)                      # one is still running
    st.worker_close('w1', ss[7]['session_id'], 'completed', 1.0, now=NOW)
    st.open_task(EPOCH, '5H0', 'env', now=NOW)
    assert len(others) == 8


def test_task_ends_when_idle(st):
    """Only some nonces opened and all sessions closed: still in flight while recently active, ends after 15 min idle."""
    open_n(st, 8, now=NOW + 1000)
    t = st.open_task(EPOCH, '5H0', 'env', now=NOW)
    s = st.open_sessions(EPOCH, '5H0', t['task_id'], [0], now=NOW)[0]
    assert st.worker_touch('w1', s['session_id'], now=NOW + 100)
    st.worker_close('w1', s['session_id'], 'completed', 0.0, now=NOW + 100)
    with pytest.raises(QuotaError, match='task_quota_inflight'):
        st.open_task(EPOCH, '5H0', 'env', now=NOW + 999)
    st.open_task(EPOCH, '5H0', 'env', now=NOW + 1001)                   # last activity at NOW+100, idle >900 s
    with pytest.raises(QuotaError, match='task_finished'):
        st.open_sessions(EPOCH, '5H0', t['task_id'], [1], now=NOW + 1002)


def test_task_without_sessions_ends_when_idle(st):
    ts = open_n(st, 9)
    with pytest.raises(QuotaError, match='task_quota_inflight'):
        st.open_task(EPOCH, '5H0', 'env', now=NOW + 899)
    st.open_task(EPOCH, '5H0', 'env', now=NOW + 901)                    # assigned but no session ever opened
    assert len(ts) == 9


# ---------------------------------------------------------------- per-task concurrency, nonces
def test_concurrency_quota_ten_per_task(tmp_path):
    st, _ = make_store(tmp_path, envs={'env': make_env(tmp_path, group_size=16)})   # group size 16 so the cap of 10 can be reached
    t = st.open_task(EPOCH, '5H', 'env', now=NOW)
    ss = st.open_sessions(EPOCH, '5H', t['task_id'], list(range(10)), now=NOW)
    with pytest.raises(QuotaError, match='concurrency_quota'):
        st.open_sessions(EPOCH, '5H', t['task_id'], [10], now=NOW)
    st.worker_close('w1', ss[0]['session_id'], 'completed', 0.0, now=NOW)  # closing one frees one concurrency slot
    st.open_sessions(EPOCH, '5H', t['task_id'], [10], now=NOW)


def test_idle_session_expires_and_frees_slot(tmp_path):
    st, _ = make_store(tmp_path, Limits(max_concurrent=2, idle_ttl=900, worker_ttl=BIG))
    t = st.open_task(EPOCH, '5H', 'env', now=NOW)
    a, b = st.open_sessions(EPOCH, '5H', t['task_id'], [0, 1], now=NOW)
    with pytest.raises(QuotaError, match='concurrency_quota'):
        st.open_sessions(EPOCH, '5H', t['task_id'], [2], now=NOW + 10)
    assert st.worker_touch('w1', a['session_id'], now=NOW + 600)          # a is active, b never moves
    st.open_sessions(EPOCH, '5H', t['task_id'], [2], now=NOW + 901)        # b expires, freeing a slot
    with pytest.raises(QuotaError, match='concurrency_quota'):
        st.open_sessions(EPOCH, '5H', t['task_id'], [3], now=NOW + 901)
    assert not st.worker_touch('w1', b['session_id'], now=NOW + 902)       # an expired session cannot be renewed
    assert not st.worker_touch('w1', 'nonexistent', now=NOW)
    assert not st.worker_touch('w2', a['session_id'], now=NOW + 903)       # another worker cannot renew it


def test_nonce_once_and_range(st):
    t = st.open_task(EPOCH, '5H0', 'env', now=NOW)
    s = st.open_sessions(EPOCH, '5H0', t['task_id'], [3], now=NOW)[0]
    st.worker_close('w1', s['session_id'], 'completed', 0.0, now=NOW)
    with pytest.raises(QuotaError, match='nonce_used'):
        st.open_sessions(EPOCH, '5H0', t['task_id'], [3], now=NOW)          # cannot reopen even after closing (prevents cherry-picking)
    with pytest.raises(QuotaError, match='bad_nonce'):
        st.open_sessions(EPOCH, '5H0', t['task_id'], [8], now=NOW)
    with pytest.raises(QuotaError, match='bad_nonce'):
        st.open_sessions(EPOCH, '5H0', t['task_id'], [4, 4], now=NOW)
    with pytest.raises(QuotaError, match='unknown_task'):
        st.open_sessions(EPOCH, '5H1', t['task_id'], [4], now=NOW)          # someone else's task


def test_concurrent_opens_respect_quota(tmp_path):
    """20 threads (separate connections, simulating multiple gateway processes) race to open tasks; at most 9 succeed."""
    st, key = make_store(tmp_path)
    ok, errs = [], []

    def run():
        s2 = Store(str(tmp_path / 'g.db'), st.envs, key, Limits(worker_ttl=BIG))
        try:
            s2.open_task(EPOCH, '5H', 'env', now=NOW)
            ok.append(1)
        except QuotaError as e:
            errs.append(e.code)

    ts = [threading.Thread(target=run) for _ in range(20)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(ok) == 9 and errs == ['task_quota_inflight'] * 11


# ---------------------------------------------------------------- session token
def test_token_claims(tmp_path):
    st, key = make_store(tmp_path)
    t = st.open_task(EPOCH, '5H0', 'env', now=NOW)
    s = st.open_sessions(EPOCH, '5H0', t['task_id'], [0], now=NOW)[0]
    c = verify_token(key.verify_key, s['token'], now=NOW)
    assert (c['hotkey'], c['task'], c['nonce'], c['worker_id'], c['env_id']) == ('5H0', t['task'], 0, 'w1', 'env')
    with pytest.raises(AuthError, match='expired_token'):
        verify_token(key.verify_key, s['token'], now=NOW + 5 * 3600)
    with pytest.raises(AuthError, match='bad_token'):
        verify_token(SigningKey.generate().verify_key, s['token'], now=NOW)


# ---------------------------------------------------------------- task assignment
def test_dispatch_exclusive_and_pool_exhausted(tmp_path):
    st, _ = make_store(tmp_path, envs={'env': make_env(tmp_path, n=5)})
    got = [st.open_task(EPOCH, f'5H{i}', 'env', now=NOW)['task'] for i in range(5)]
    assert sorted(got) == [f't{i:03d}' for i in range(5)]                # a task goes to only one miner at a time
    with pytest.raises(QuotaError, match='pool_exhausted'):
        st.open_task(EPOCH, '5H9', 'env', now=NOW)
    st.open_task('e000002', '5H9', 'env', now=NOW)                       # reopened in the next epoch


def test_unqualified_returns_to_pool_but_not_same_hotkey(tmp_path):
    st, _ = make_store(tmp_path, envs={'env': make_env(tmp_path, n=1)})
    t = st.open_task(EPOCH, '5H0', 'env', now=NOW)
    close_all(st, t['task_id'], [('completed', 0.0)] * 8)                # all wrong: not qualified
    with pytest.raises(QuotaError, match='pool_exhausted'):
        st.open_task(EPOCH, '5H0', 'env', now=NOW)                       # the same hotkey does not get the same task again
    assert st.open_task(EPOCH, '5H1', 'env', now=NOW)['task'] == t['task']  # returned to the pool for others


def test_qualified_not_redispatched(tmp_path):
    st, _ = make_store(tmp_path, envs={'env': make_env(tmp_path, n=1)})
    t = st.open_task(EPOCH, '5H0', 'env', now=NOW)
    close_all(st, t['task_id'], [('completed', 1.0), ('completed', 0.0)] + [('unresolved', 0.0)] * 6)
    with pytest.raises(QuotaError, match='pool_exhausted'):
        st.open_task(EPOCH, '5H1', 'env', now=NOW)                       # qualified: not assigned again this epoch


def test_abandoned_returns_to_pool(tmp_path):
    st, _ = make_store(tmp_path, envs={'env': make_env(tmp_path, n=1)})
    t = st.open_task(EPOCH, '5H0', 'env', now=NOW)
    st.finish_task(EPOCH, '5H0', t['task_id'], now=NOW)
    assert st.open_task(EPOCH, '5H1', 'env', now=NOW)['task'] == t['task']


def test_infra_error_voids_and_refunds(tmp_path):
    st, _ = make_store(tmp_path, envs={'env': make_env(tmp_path, n=30)})
    for _ in range(19):
        x = st.open_task(EPOCH, '5H0', 'env', now=NOW)
        st.finish_task(EPOCH, '5H0', x['task_id'], now=NOW)
    t = st.open_task(EPOCH, '5H0', 'env', now=NOW)                        # the 20th task
    a, b = st.open_sessions(EPOCH, '5H0', t['task_id'], [0, 1], now=NOW)
    st.worker_close('w1', a['session_id'], 'infra_error', None, now=NOW)
    assert not st.worker_touch('w1', b['session_id'], now=NOW)           # the other sessions in the group are voided too
    st.open_task(EPOCH, '5H0', 'env', now=NOW)                            # a voided task does not count as "started", so one more can be opened
    with pytest.raises(QuotaError, match='task_quota_epoch'):
        st.open_task(EPOCH, '5H0', 'env', now=NOW)
    assert st.open_task(EPOCH, '5H1', 'env', now=NOW)                    # the voided task returns to the pool


def test_worker_lost_voids_tasks(tmp_path):
    st, _ = make_store(tmp_path, Limits(worker_ttl=90))
    t = st.open_task(EPOCH, '5H0', 'env', now=NOW)
    s = st.open_sessions(EPOCH, '5H0', t['task_id'], [0], now=NOW)[0]
    st.heartbeat('w2', 'http://w2', 128, [], ['env'], now=NOW + 200)      # w1 has had no heartbeat for 200 s
    st.reap(now=NOW + 200)
    assert not st.worker_touch('w1', s['session_id'], now=NOW + 200)
    for _ in range(9):                                                   # voided tasks do not hold in-flight slots
        assert st.open_task(EPOCH, '5H0', 'env', now=NOW + 200)['worker_url'] == 'http://w2'


# ---------------------------------------------------------------- global budget, fair share, cache preference
def test_global_budget(tmp_path):
    st, _ = make_store(tmp_path, Limits(max_containers=16, worker_ttl=BIG))
    open_n(st, 2)
    with pytest.raises(CapacityError, match='capacity_exhausted') as e:
        st.open_task(EPOCH, '5H1', 'env', now=NOW)
    assert e.value.status == 429


def test_worker_capacity(tmp_path):
    st, _ = make_store(tmp_path, workers=(('w1', 20, ()),))              # 8 slots reserved per task: room for only 2
    open_n(st, 2)
    with pytest.raises(CapacityError, match='capacity_exhausted'):
        st.open_task(EPOCH, '5H1', 'env', now=NOW)
    st.heartbeat('w2', 'http://w2', 8, [], ['env'], now=NOW)
    assert st.open_task(EPOCH, '5H1', 'env', now=NOW)['worker_url'] == 'http://w2'


def test_fair_share(tmp_path):
    st, _ = make_store(tmp_path, Limits(max_containers=32, worker_ttl=BIG))   # budget for 4 tasks
    a = open_n(st, 4, hk='A')
    with pytest.raises(CapacityError, match='capacity_exhausted'):
        st.open_task(EPOCH, 'B', 'env', now=NOW + 1)                      # B is waiting
    st.finish_task(EPOCH, 'A', a[0]['task_id'], now=NOW + 2)
    with pytest.raises(CapacityError, match='fair_share'):
        st.open_task(EPOCH, 'A', 'env', now=NOW + 3)                      # A holds 3 >= share 4//2, the free slot is kept for B
    st.open_task(EPOCH, 'B', 'env', now=NOW + 4)
    st.finish_task(EPOCH, 'A', a[1]['task_id'], now=NOW + 5)
    st.open_task(EPOCH, 'A', 'env', now=NOW + 6)                          # nobody is waiting, A opens as usual


def test_prefers_cached_image_and_worker(tmp_path):
    env = make_env(tmp_path, n=50)
    st, _ = make_store(tmp_path, envs={'env': env}, workers=(('w1', 64, ()), ('w2', 64, ('img-37',))))
    t = st.open_task(EPOCH, '5H0', 'env', now=NOW)
    assert (t['task'], t['worker_url']) == ('t037', 'http://w2')


def test_worker_spec_and_grade_inputs_scoped(tmp_path):
    st, _ = make_store(tmp_path, workers=(('w1', 64, ()), ('w2', 0, ())))
    t = st.open_task(EPOCH, '5H0', 'env', now=NOW)
    spec = st.worker_task_spec('w1', t['task_id'])
    assert spec['spec']['id'] == t['task'] and 'test_command' not in spec['spec']   # grading inputs are not sent with the task spec
    with pytest.raises(QuotaError, match='unknown_task'):
        st.worker_task_spec('w2', t['task_id'])
    s = st.open_sessions(EPOCH, '5H0', t['task_id'], [0], now=NOW)[0]
    assert st.worker_grade_inputs('w1', s['session_id'])['test_command'] == 'true'
    with pytest.raises(QuotaError, match='unknown_task'):
        st.worker_grade_inputs('w2', s['session_id'])


# ---------------------------------------------------------------- group seal
def test_finish_returns_a_gateway_signed_seal(tmp_path):
    """The seal lists the receipt hash each worker reported, by nonce, with the gateway's own open / finish
    times; finishing again returns the same seal; a voided task has none."""
    from hosted_env.infra.receipt import ReceiptError, verify_group_seal
    st, key = make_store(tmp_path)
    t = st.open_task(EPOCH, '5H0', 'env', now=NOW)
    ss = st.open_sessions(EPOCH, '5H0', t['task_id'], list(range(8)), now=NOW + 1)
    for s in ss[:7]:
        st.worker_close('w1', s['session_id'], 'completed', 0.0, now=NOW + 5, receipt_sha256=f'{s["nonce"]:064x}')
    with pytest.raises(QuotaError, match='bad_receipt_sha256'):
        st.worker_close('w1', ss[7]['session_id'], 'completed', 1.0, now=NOW + 6, receipt_sha256='XYZ')
    seal = st.finish_task(EPOCH, '5H0', t['task_id'], now=NOW + 9.5)
    c = verify_group_seal(seal, key.verify_key, epoch=EPOCH, hotkey='5H0', env_id='env', task=t['task'],
                          group_size=8)
    assert c['receipts'] == [f'{n:064x}' for n in range(7)] + [None]       # nonce 7 never reported one
    assert (c['opened_at_ms'], c['finished_at_ms'], c['task_id']) == (int(NOW * 1000), int((NOW + 9.5) * 1000),
                                                                      t['task_id'])
    assert st.finish_task(EPOCH, '5H0', t['task_id'], now=NOW + 99) == seal
    st.envs['env'].group_size = 3                  # a config change after the task was opened does not resize it
    t2 = st.open_task(EPOCH, '5H1', 'env', now=NOW)
    st.envs['env'].group_size = 8
    assert len(verify_group_seal(st.finish_task(EPOCH, '5H1', t2['task_id'], now=NOW + 1), key.verify_key,
                                 epoch=EPOCH, hotkey='5H1', env_id='env', task=t2['task'], group_size=3)['receipts']) == 3
    for kw, code in (({'hotkey': '5H1'}, 'seal_identity_mismatch'), ({'group_size': 3}, 'seal_identity_mismatch')):
        with pytest.raises(ReceiptError, match=code):
            verify_group_seal(seal, key.verify_key, **{**dict(epoch=EPOCH, hotkey='5H0', env_id='env',
                                                              task=t['task'], group_size=8), **kw})
    with pytest.raises(ReceiptError, match='bad_seal_signature'):              # not the gateway's key
        verify_group_seal(seal, SigningKey.generate().verify_key, epoch=EPOCH, hotkey='5H0', env_id='env',
                          task=t['task'], group_size=8)
    v = st.open_task(EPOCH, '5H0', 'env', now=NOW)
    vs = st.open_sessions(EPOCH, '5H0', v['task_id'], [0], now=NOW)
    st.worker_close('w1', vs[0]['session_id'], 'infra_error', 0.0, now=NOW)
    assert st.finish_task(EPOCH, '5H0', v['task_id'], now=NOW) is None


def test_reqids_are_scoped_and_legacy_ones_still_count(tmp_path):
    """Per signer scope; an id recorded before scopes existed (an upgraded database) blocks every scope until it
    ages out, so a request signed just before the upgrade cannot be replayed after it."""
    import sqlite3

    from hosted_env.infra.auth import REQID_TTL
    old = sqlite3.connect(tmp_path / 'g.db')                  # a database from before per-signer scopes
    old.execute('CREATE TABLE reqids (reqid TEXT PRIMARY KEY, seen_at REAL NOT NULL)')
    old.execute('INSERT INTO reqids VALUES (?, ?)', ('legacy' * 4, NOW))
    old.commit()
    old.close()
    st, _ = make_store(tmp_path)
    assert st.remember_reqid('miner|5H0', 'a' * 32, NOW) and st.remember_reqid('miner|5H1', 'a' * 32, NOW)
    assert not st.remember_reqid('miner|5H0', 'a' * 32, NOW + 1)
    assert not st.remember_reqid('miner|5H0', 'legacy' * 4, NOW + 1)
    assert st.remember_reqid('miner|5H0', 'legacy' * 4, NOW + REQID_TTL + 1)    # aged out
    assert st.remember_reqid('miner|5H0', 'a' * 32, NOW + REQID_TTL + 2)
