"""`hosted-env mine` (cli.py cmd_mine): the reference miner stops cleanly when the epoch has nothing left for it."""
import json

import pytest
from test_e2e import EPOCH, stack  # noqa: F401  (the fixture)

import hosted_env.cli as cli
import hosted_env.infra.client as client
from hosted_env.infra.client import ApiError


def _key(tmp_path, sk):
    p = tmp_path / 'miner.key'
    p.write_text(bytes(sk).hex() + '\n')
    return str(p)


def _mine(gateway, key, out, *args, epoch=EPOCH):
    with pytest.raises(SystemExit) as e:
        cli.main(['mine', '--gateway', gateway, '--hotkey', '5H0', '--key', key, '--env', 'swe',
                  *(['--epoch', epoch] if epoch else []), '--out', str(out), *args])
    return e.value.code


def _stop_line(err):
    lines = [json.loads(x) for x in err.splitlines() if x.startswith('{')]
    assert len(lines) == 1
    return lines[0]


@pytest.mark.parametrize('tasks', ['0', '5'])
def test_mine_stops_at_pool_exhausted(stack, tmp_path, capsys, monkeypatch, tasks):  # noqa: F811
    """The pool holds 2 tasks: the miner runs both, then the gateway's 409 pool_exhausted ends it with exit 0 and
    one stop line on stderr; stdout keeps exactly one line per task."""
    monkeypatch.setattr(cli, 'SMOKE_REPLIES', ['Nothing to do.'])    # one turn per rollout keeps this quick
    out = tmp_path / 'att'
    code = _mine(stack['gw'].url, _key(tmp_path, stack['miners']['5H0']), out, '--tasks', tasks)
    cap = capsys.readouterr()
    assert code == 0
    assert 'Traceback' not in cap.err
    per_task = [json.loads(x) for x in cap.out.splitlines()]
    assert len(per_task) == 2 and all(not t['voided'] for t in per_task)
    assert len(list(out.iterdir())) == 2
    stop = _stop_line(cap.err)
    assert stop == {'stopped': 'pool_exhausted', 'status': 409, 'detail': 'no task left for this hotkey in this epoch',
                    'phase': 'open', 'tasks_done': 2, 'attachments_written': 2}


@pytest.mark.parametrize('status,code,exit_code,epoch', [
    (409, 'task_quota_epoch', 0, EPOCH),
    (409, 'wrong_epoch', 1, EPOCH),            # the epoch the operator named is not the current one: their mistake
    (409, 'wrong_epoch', 0, None),             # the epoch read from /v1/info at start has ended: nothing left in it
    (409, 'task_quota_inflight', 1, EPOCH),
    (429, 'capacity_exhausted', 1, EPOCH),
    (401, '', 1, EPOCH)])
def test_mine_refusals(tmp_path, capsys, monkeypatch, status, code, exit_code, epoch):
    """Refusals to open: the epoch quota is the other "nothing left" stop (exit 0); any other exits 1. Neither
    prints a traceback."""
    from nacl.signing import SigningKey

    def refuse(*a, **kw):
        raise ApiError(status, code, 'why')
    monkeypatch.setattr(client, 'run_task', refuse)
    monkeypatch.setattr(client.GatewayClient, 'info', lambda self: {'epoch': EPOCH})
    rc = _mine('http://127.0.0.1:9', _key(tmp_path, SigningKey.generate()), tmp_path / 'att', '--tasks', '0',
               epoch=epoch)
    cap = capsys.readouterr()
    assert rc == exit_code and cap.out == '' and 'Traceback' not in cap.err
    assert _stop_line(cap.err) == {'stopped': code or f'http_{status}', 'status': status, 'detail': 'why',
                                   'phase': 'open', 'tasks_done': 0, 'attachments_written': 0}


@pytest.mark.parametrize('code', ['worker_unavailable', 'pool_exhausted', 'wrong_epoch'])
def test_mine_error_after_the_open_exits_1(tmp_path, capsys, monkeypatch, code):
    """An error once the task had opened (the worker, or finishing the task) is a failure whatever its code, and the
    stop line names the task so it does not read as a refusal to open."""
    from nacl.signing import SigningKey

    def fail(*a, **kw):
        e = ApiError(503, code, 'why')
        e.task = 'task-7'
        raise e
    monkeypatch.setattr(client, 'run_task', fail)
    monkeypatch.setattr(client.GatewayClient, 'info', lambda self: {'epoch': EPOCH})
    rc = _mine('http://127.0.0.1:9', _key(tmp_path, SigningKey.generate()), tmp_path / 'att', epoch=None)
    cap = capsys.readouterr()
    assert rc == 1 and 'Traceback' not in cap.err
    assert _stop_line(cap.err) == {'stopped': code, 'status': 503, 'detail': 'why', 'phase': 'task',
                                   'task': 'task-7', 'tasks_done': 0, 'attachments_written': 0}


def test_mine_rejects_a_negative_task_count(tmp_path, capsys):
    from nacl.signing import SigningKey
    assert _mine('http://127.0.0.1:9', _key(tmp_path, SigningKey.generate()), tmp_path / 'att', '--tasks', '-1') == 2
    assert 'must be 0' in capsys.readouterr().err
