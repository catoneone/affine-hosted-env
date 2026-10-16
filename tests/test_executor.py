"""executor/core.py: rollout state machine, idempotent resends, validation, ways a rollout ends, receipts. In-process gateway (calls Store directly)."""
import pytest
from conftest import FIX, make_swe_tasks
from nacl.signing import SigningKey

from hosted_env.common import key_id
from hosted_env.coordinator.store import Limits, Store
from hosted_env.envs import GradeError, SweEnv
from hosted_env.executor.core import EnvRuntime, Worker, WorkerError
from hosted_env.providers.sandbox import SandboxError
from hosted_env.receipt import ReceiptError, verify_receipt
from hosted_env.render import CharRenderer

EPOCH = 'e000001'


class InProcGateway:
    def __init__(self, store, wid):
        self.st, self.wid = store, wid

    def heartbeat(self, url, capacity, images, env_ids):
        self.st.heartbeat(self.wid, url, capacity, images, env_ids)

    def task_spec(self, task_id):
        return self.st.worker_task_spec(self.wid, task_id)

    def grade_inputs(self, sid):
        return self.st.worker_grade_inputs(self.wid, sid)

    def touch(self, sid):
        return self.st.worker_touch(self.wid, sid)

    def close(self, sid, outcome, reward):
        self.st.worker_close(self.wid, sid, outcome, reward)


@pytest.fixture
def rig(tmp_path):
    path, ids = make_swe_tasks(tmp_path, n=4)
    cfg = {'kind': 'mimo_swe', 'tasks': path, 'cleanup': False, 'grader_user': 'root'}
    gkey, wkey = SigningKey.generate(), SigningKey.generate()
    store = Store(str(tmp_path / 'g.db'), {'swe': SweEnv('swe', cfg)}, gkey, Limits())
    r = CharRenderer()
    rt = EnvRuntime(env=SweEnv('swe', {**cfg, 'tasks': None}), renderer=r, sandbox='local', max_context=100_000,
                    max_turn_tokens=5000, max_steps=5)
    w = Worker(worker_id='w1', key=wkey, gateway=InProcGateway(store, 'w1'), gateway_pub=gkey.verify_key,
               runtimes={'swe': rt}, public_url='http://w1', capacity=16, scratch=str(tmp_path))
    w.heartbeat()

    def session(nonces=(0,), hotkey='5H0'):
        t = store.open_task(EPOCH, hotkey, 'swe')
        return t, store.open_sessions(EPOCH, hotkey, t['task_id'], list(nonces))

    yield w, store, r, rt, session, wkey
    w.pool.shutdown(wait=True)


def enc(r, text, stop=True):
    return r.encode(text) + ([r.turn_end_id] if stop else [])


def run(w, tok, *turns):
    """Run start and some turns; return (per-step results, source segments)."""
    res = [w.start(tok).result(30)]
    segs = [('P', res[0]['prompt_tokens'])]
    for i, ids in enumerate(turns, 1):
        res.append(w.turn(tok, i, ids).result(30))
        segs.append(('M', ids))
        if not res[-1]['done']:
            segs.append(('T', res[-1]['tool_tokens']))
    return res, segs


def check_receipt(res, segs, t, nonce, wkey):
    return verify_receipt(res['receipt'], segs, {key_id(wkey.verify_key): wkey.verify_key}, epoch=EPOCH,
                          hotkey='5H0', env_id='swe', task=t['task'], nonce=nonce)


def test_fix_and_submit(rig):
    w, store, r, rt, session, wkey = rig
    t, (s,) = session()
    res, segs = run(w, s['token'], enc(r, FIX), enc(r, 'Fixed.'))
    p = r.decode(res[0]['prompt_tokens'])
    assert 'Fix the following issue:\n\nadd() subtracts' in p and p.endswith('<|im_start|>assistant\n<think></think>')
    assert res[0]['stop_token_ids'] == r.stop_ids
    obs = r.decode(res[1]['tool_tokens'])
    assert obs.startswith('<|im_start|>tool\n') and 'return a + b' in obs and "Tool metadata: {'returncode': 0}" in obs
    assert obs.endswith('<|im_start|>assistant\n<think></think>')
    last = res[-1]
    assert (last['done'], last['outcome'], last['reward'], last['detail']) == (True, 'completed', 1.0, 'submitted')
    c = check_receipt(last, segs, t, 0, wkey)
    assert c['outcome'] == 'completed' and c['reward'] == 1.0
    assert store.db.execute('SELECT outcome, reward FROM sessions WHERE id=?', (s['session_id'],)).fetchone() == \
        ('completed', 1.0)                                              # reported to the gateway before returning to the miner
    with pytest.raises(WorkerError, match='rollout_done'):
        w.turn(s['token'], 3, enc(r, 'again'))


def test_idempotent_resend_and_seq(rig):
    w, store, r, rt, session, wkey = rig
    _, (s,) = session()
    tok = s['token']
    assert w.start(tok) is w.start(tok)                                # a resent start is the same operation
    w.start(tok).result(30)
    a = w.turn(tok, 1, enc(r, FIX))
    assert w.turn(tok, 1, enc(r, FIX)) is a                            # same seq, same tokens: same result
    a.result(30)
    with pytest.raises(WorkerError, match='bad_seq'):
        w.turn(tok, 1, enc(r, 'different'))                            # same seq, different tokens
    with pytest.raises(WorkerError, match='bad_seq'):
        w.turn(tok, 3, enc(r, 'skip'))
    assert w.turn(tok, 2, enc(r, 'ok')).result(30)['done']


def test_turn_validation(rig):
    w, store, r, rt, session, wkey = rig
    _, (s,) = session()
    tok = s['token']
    w.start(tok).result(30)
    for bad, code in [(r.encode('a') + [r.turn_end_id] + r.encode('b') + [r.turn_end_id], 'stop_token_inside'),
                      ([r.vocab_size + 1, r.turn_end_id], 'bad_tokens'), ([], 'bad_tokens'), ('x', 'bad_tokens'),
                      (enc(r, 'x' * 6000), 'turn_too_long')]:
        with pytest.raises(WorkerError, match=code):
            w.turn(tok, 1, bad)
    assert w.turn(tok, 1, enc(r, 'fine')).result(30)['outcome'] == 'completed'   # a rejected request does not advance seq


def test_truncated_and_step_limit_are_unresolved_but_graded(rig):
    w, store, r, rt, session, wkey = rig
    t, (a, b) = session(nonces=(0, 1))
    res, segs = run(w, a['token'], enc(r, FIX, stop=False))           # did not end on a stop token: truncated
    assert (res[-1]['outcome'], res[-1]['detail'], res[-1]['reward']) == ('unresolved', 'response_truncated', 0.0)
    check_receipt(res[-1], segs, t, 0, wkey)
    rt.max_steps = 1
    res, segs = run(w, b['token'], enc(r, FIX))
    assert (res[-1]['outcome'], res[-1]['detail'], res[-1]['reward']) == ('unresolved', 'step_limit', 1.0)
    assert check_receipt(res[-1], segs, t, 1, wkey)['n_tokens'] == len(segs[0][1]) + len(segs[1][1])  # no trailing T


def test_context_limit(rig):
    w, store, r, rt, session, wkey = rig
    _, (s,) = session()
    w.start(s['token']).result(30)
    rt.max_context = w.sessions[s['session_id']].total + 200 + rt.min_room   # appending the tool output exceeds it
    res = w.turn(s['token'], 1, enc(r, FIX)).result(30)
    assert (res['outcome'], res['detail']) == ('unresolved', 'context_limit')


def test_tamper_detected(rig):
    w, store, r, rt, session, wkey = rig
    t, (s,) = session()
    res, segs = run(w, s['token'], enc(r, FIX), enc(r, 'Fixed.'))
    fake = list(segs[2][1])
    fake[5] = r.encode('Z')[0]                                         # change one tool-output token
    with pytest.raises(ReceiptError, match='trajectory_mismatch'):
        check_receipt(res[-1], [segs[0], segs[1], ('T', fake), segs[3]], t, 0, wkey)


def test_setup_infra_error_voids_task(rig, monkeypatch):
    w, store, r, rt, session, wkey = rig
    t, (a, b) = session(nonces=(0, 1))

    def boom(sess):
        raise SandboxError('docker run failed')
    monkeypatch.setattr(rt.env, 'setup', boom)
    res = w.start(a['token']).result(30)
    assert (res['done'], res['outcome'], res['receipt']) == (True, 'infra_error', None)
    assert store.db.execute('SELECT state, end_reason FROM tasks WHERE task_id=?', (t['task_id'],)).fetchone() == \
        ('void', 'infra_error')                                        # one infra error -> the whole task is voided
    with pytest.raises(WorkerError, match='session_closed'):
        w.start(b['token']).result(30)                                 # other sessions in the group cannot start


def test_grade_infra_error_is_signed(rig, monkeypatch):
    w, store, r, rt, session, wkey = rig
    t, (s,) = session()

    def bad_grade(sess, gi):
        raise GradeError('apply_test_patch_failed')
    monkeypatch.setattr(rt.env, 'grade', bad_grade)
    res, segs = run(w, s['token'], enc(r, 'done'))
    assert (res[-1]['outcome'], res[-1]['reward']) == ('infra_error', 0.0)
    assert check_receipt(res[-1], segs, t, 0, wkey)['outcome'] == 'infra_error'


def test_wrong_worker_and_bad_token(rig):
    w, store, r, rt, session, wkey = rig
    _, (s,) = session()
    w.worker_id = 'w2'
    with pytest.raises(WorkerError, match='wrong_worker'):
        w.start(s['token'])
    w.worker_id = 'w1'
    with pytest.raises(WorkerError, match='bad_token'):
        w.start(s['token'][:-3] + 'AAA')


def test_capacity(rig):
    w, store, r, rt, session, wkey = rig
    w.capacity = 1
    _, (a, b) = session(nonces=(0, 1))
    w.start(a['token']).result(30)
    with pytest.raises(WorkerError, match='worker_full'):
        w.start(b['token'])
