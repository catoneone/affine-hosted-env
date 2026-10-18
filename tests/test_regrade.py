"""`hosted-env regrade` (envs/swe/cli.py) and SweEnv.regrade: a verifier grades a captured patch again."""
import json

import pytest
from test_e2e import EPOCH, stack  # noqa: F401  (the fixture)
from test_envs import _pytest_task
from test_forgery import FIX, FORGE

from hosted_env.envs import GradeError
from hosted_env.infra.client import ScriptedSampler, run_task
from hosted_env.infra.providers.sandbox import LocalSandbox
from hosted_env.infra.render import CharRenderer


def _captured(tmp_path, write):
    """The patch a rollout produced (as the receipt binds it) and the start tree."""
    env, task, base = _pytest_task(tmp_path)
    factory = lambda image, cwd: LocalSandbox(image, cwd, base_dir=base)    # noqa: E731
    s = env.open(env.public_spec(task), factory)
    s.setup()
    write(s.sb)
    patch, problem = env.capture_patch(s)
    start = s.state['start_tree']
    s.close()
    assert not problem
    return env, task, factory, patch, start


def test_regrade_reproduces_the_grade(tmp_path):
    env, task, factory, patch, start = _captured(tmp_path, lambda sb: sb.execute(FIX, cwd='/testbed'))
    g = env.regrade(task, patch, start, factory)
    assert g['reward'] == 1.0 and g['second_check'] is True and g['canary'].startswith('ok')
    env, task, factory, patch, start = _captured(tmp_path, lambda sb: sb.copy_text_to(FORGE, '/testbed/calc.py'))
    g = env.regrade(task, patch, start, factory)                    # the canary runs in the re-grade too
    assert g['reward'] == 0.0 and g['grade_detail'].startswith('canary:')
    with pytest.raises(GradeError, match='starts from tree'):       # the image must reproduce the receipt's tree
        env.regrade(task, patch, '0' * 40, factory)
    with pytest.raises(GradeError, match='over the grader cap'):
        env.regrade(task, b'x' * 2_000_001, start, factory)


def _cli(tmp_path, stack, att, tasks):  # noqa: F811
    from hosted_env.cli import main
    (tmp_path / 'att.json').write_text(json.dumps(att))
    cfg = tmp_path / 'regrade.toml'
    cfg.write_text(f'[envs.swe]\nkind = "mimo_swe"\ntasks = "{tasks}"\nsandbox = "local"\ngrader_user = "root"\n'
                   f'require_image_digest = false\ncleanup = false\n[envs.swe.sandbox_opts]\n'
                   f'base_dir = "{tmp_path}"\n')
    with pytest.raises(SystemExit) as e:
        main(['regrade', '--attachment', str(tmp_path / 'att.json'), '--gateway', stack['gw'].url, '--tasks', tasks,
              '--config', str(cfg)])
    return e.value.code


def test_regrade_cli_end_to_end(stack, tmp_path, capsys):  # noqa: F811
    """A real group from the gateway: every graded rollout re-grades to its receipt's reward; a verifier whose task
    row differs from the one served is refused before anything runs."""
    from conftest import FIX as ROLLOUT_FIX
    r = CharRenderer()
    from hosted_env.infra.client import GatewayClient
    gw = GatewayClient(stack['gw'].url, '5H0', stack['miners']['5H0'])
    sampler = ScriptedSampler(r.encode, r.turn_end_id, {n: [ROLLOUT_FIX, 'Fixed.'] if n % 2 == 0 else ['Nothing.']
                                                        for n in range(8)})
    att = run_task(gw, EPOCH, 'swe', sampler)['attachment']
    tasks = stack['store']._env('swe').cfg['tasks']
    capsys.readouterr()
    assert _cli(tmp_path, stack, att, tasks) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith('{')]
    per = lines[:-1]
    assert [(x['nonce'], x['receipt_reward'], x['regrade_reward']) for x in per] == \
        [(n, 1.0 if n % 2 == 0 else 0.0, 1.0 if n % 2 == 0 else 0.0) for n in range(8)]
    assert lines[-1]['ok'] is True and lines[-1]['regraded'] == 8
    mine = tmp_path / 'my-tasks.jsonl'                                # the verifier's copy says something else
    rows = [json.loads(x) for x in open(tasks) if x.strip()]
    mine.write_text(''.join(json.dumps({**x, 'problem_statement': 'edited'}) + '\n' for x in rows))
    assert _cli(tmp_path, stack, att, str(mine)) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['error'] == 'task_hash_mismatch'
