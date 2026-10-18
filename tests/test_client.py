"""client.run_task / run_rollout with fake gateway and worker clients: a task voided mid-group."""
from hosted_env.infra.client import ApiError, ScriptedSampler, run_task
from hosted_env.infra.render import CharRenderer


class FakeGateway:
    def __init__(self, finish_error=None):
        self.finished, self.finish_error = [], finish_error

    def open_task(self, epoch, env_id, max_wait=0.0):
        return {'task_id': 't1', 'env_id': env_id, 'task': 'task-1', 'worker_url': 'http://w', 'group_size': 3}

    def open_sessions(self, epoch, task_id, nonces):
        return [{'nonce': n, 'token': f'tok{n}'} for n in nonces]

    def finish_task(self, epoch, task_id):
        self.finished.append(task_id)
        if self.finish_error:
            raise self.finish_error


class FakeWorker:
    """nonce 0 hits an infra error on its first turn; the gateway then voids the task, so the others get 410."""

    def __init__(self, url):
        pass

    def start(self, token):
        return {'done': False, 'prompt_tokens': [1, 2], 'stop_token_ids': [0], 'max_context': 100,
                'max_turn_tokens': 10}

    def turn(self, token, seq, tokens):
        if token == 'tok0':
            return {'done': True, 'receipt': {'payload': 'x', 'sig': 'y'}, 'outcome': 'infra_error', 'reward': 0.0,
                    'detail': 'grade: GradeError: boom', 'steps': 1}
        raise ApiError(410, 'session_closed')


def sampler():
    r = CharRenderer()
    return ScriptedSampler(r.encode, r.turn_end_id, ['ls'])


def test_voided_task_is_reported_not_raised():
    gw = FakeGateway(finish_error=ApiError(409, 'task_not_running'))
    res = run_task(gw, 'e1', 'swe', sampler(), worker_factory=FakeWorker)
    assert res['voided'] is True and res['attachment'] is None and gw.finished == ['t1']
    assert [(x['nonce'], x['outcome']) for x in res['rollouts']] == [(0, 'infra_error'), (1, 'voided'), (2, 'voided')]
    assert res['rollouts'][1]['receipt'] is None and res['rollouts'][1]['detail'] == 'session_closed'


def test_other_errors_still_raise():
    import pytest

    class Broken(FakeWorker):
        def turn(self, token, seq, tokens):
            raise ApiError(400, 'bad_tokens')
    with pytest.raises(ApiError, match='bad_tokens'):
        run_task(FakeGateway(), 'e1', 'swe', sampler(), worker_factory=Broken)
    with pytest.raises(ApiError, match='unknown_hotkey'):
        run_task(FakeGateway(finish_error=ApiError(403, 'unknown_hotkey')), 'e1', 'swe', sampler(),
                 worker_factory=FakeWorker)


def test_group_without_a_seal_is_not_uploaded():
    """finish refused (404/409/410) after all rollouts completed: no seal, so the group could never verify."""
    class NoSealWorker(FakeWorker):
        def turn(self, token, seq, tokens):
            return {'done': True, 'receipt': {'payload': 'x', 'sig': 'y'}, 'outcome': 'completed', 'reward': 0.0,
                    'detail': 'submitted', 'steps': 1}
    gw = FakeGateway(finish_error=ApiError(409, 'task_finished'))
    res = run_task(gw, 'e1', 'swe', sampler(), worker_factory=NoSealWorker)
    assert res['voided'] is True and res['attachment'] is None
    assert [x['outcome'] for x in res['rollouts']] == ['completed'] * 3
