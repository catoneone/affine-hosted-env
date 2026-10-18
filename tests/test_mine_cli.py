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


def _mine(gateway, key, out, *args):
    with pytest.raises(SystemExit) as e:
        cli.main(['mine', '--gateway', gateway, '--hotkey', '5H0', '--key', key, '--env', 'swe', '--epoch', EPOCH,
                  '--out', str(out), *args])
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
                    'tasks_done': 2, 'attachments_written': 2}


@pytest.mark.parametrize('status,code,exit_code', [(409, 'task_quota_epoch', 0), (409, 'wrong_epoch', 1),
                                                   (429, 'capacity_exhausted', 1), (401, '', 1)])
def test_mine_refusals(tmp_path, capsys, monkeypatch, status, code, exit_code):
    """The epoch quota is the other "nothing left" refusal (exit 0); any other refusal exits 1. Neither prints a
    traceback."""
    from nacl.signing import SigningKey

    def refuse(*a, **kw):
        raise ApiError(status, code, 'why')
    monkeypatch.setattr(client, 'run_task', refuse)
    rc = _mine('http://127.0.0.1:9', _key(tmp_path, SigningKey.generate()), tmp_path / 'att', '--tasks', '0')
    cap = capsys.readouterr()
    assert rc == exit_code and cap.out == '' and 'Traceback' not in cap.err
    assert _stop_line(cap.err) == {'stopped': code or f'http_{status}', 'status': status, 'detail': 'why',
                                   'tasks_done': 0, 'attachments_written': 0}
