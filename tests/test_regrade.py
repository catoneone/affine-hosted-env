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


# A verifier's own edit, unpublished: here it inserts a pytest.fail() call, which the task-free scan in
# test_forgery.SCANNING_FORGE does not look for
VERIFIER_EDIT = r'''
import ast

calls = []


def edit(path, source, test_id):
    calls.append((path, test_id))
    func = test_id.split('::')[-1].split('[')[0]
    lines = source.split('\n')
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(f'def {func}(') and ln.rstrip().endswith(':'):
            indent = ln[:len(ln) - len(ln.lstrip())] + '    '
            lines.insert(i + 1, indent + '__import__("pytest").fail("unexpected state")')
            out = '\n'.join(lines)
            ast.parse(out)
            return out
        if ln.lstrip().startswith(f'def {func}():'):
            head, _, body = ln.partition(':')
            lines[i] = head + ': __import__("pytest").fail("unexpected state"); ' + body.strip()
            return '\n'.join(lines)
    return None
'''


def test_a_verifier_edit_catches_a_forge_the_public_edits_miss(tmp_path, monkeypatch):
    """The receipt's canary used this repository's public edits, which a forge can be written against (here: the
    built-in edits limited to the inserted constant comparison, and the task-free scan that beats it). A verifier
    re-grading with an edit of its own that the forge has not seen scores it 0 -- a mismatch with the receipt."""
    import importlib.util

    import hosted_env.envs.swe.forgery as forgery
    from test_forgery import SCANNING_FORGE
    monkeypatch.setattr(forgery, 'mutate_python_literal', lambda *a: None)
    monkeypatch.setattr(forgery, 'negate_python', lambda *a: None)
    env, task, factory, patch, start = _captured(tmp_path,
                                                 lambda sb: sb.copy_text_to(SCANNING_FORGE, '/testbed/calc.py'))
    assert env.regrade(task, patch, start, factory)['reward'] == 1.0       # the public edit is beaten
    f = tmp_path / 'my_edit.py'
    f.write_text(VERIFIER_EDIT)
    spec = importlib.util.spec_from_file_location('my_edit', f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env.canary_edit = mod.edit
    g = env.regrade(task, patch, start, factory)
    assert g['reward'] == 0.0 and g['grade_detail'].startswith('canary:'), g
    assert ('tests/test_calc.py', 'tests/test_calc.py::test_add') in mod.calls
    env2, task2, factory2, patch2, start2 = _captured(tmp_path, lambda sb: sb.execute(FIX, cwd='/testbed'))
    env2.canary_edit = mod.edit                                              # an honest fix keeps its 1
    g = env2.regrade(task2, patch2, start2, factory2)
    assert g['reward'] == 1.0 and g['canary'].startswith('ok: 2 edited test(s) failed; consistency ok'), g


def test_go_verifier_edit_is_used_in_both_canary_runs(tmp_path, monkeypatch):
    """Review of #35: the Go path of the hook. The fake go fails a test whose file holds `if got, want := ...`."""
    import json as _json

    from test_baseline import _go_task

    from hosted_env.envs import SweEnv
    env, path, factory = _go_task(tmp_path, monkeypatch)
    row = _json.loads(path.read_text())
    row.update(report_kind='gotest', fail_to_pass=['example.com/m/x::TestAdd', 'example.com/m/y::TestMul'],
               pass_to_pass=['example.com/m/x::TestOld'])
    path.write_text(_json.dumps(row) + '\n')
    calls = []

    def edit(p, source, test_id):
        calls.append(test_id)
        func = test_id.rsplit('::', 1)[1]
        head = f'func {func}(t *testing.T) {{'
        return (source.replace(head, head + ' if got, want := 1, 2; got != want { t.Fatal() };', 1)
                if head in source else None)
    e = SweEnv('swe', dict(env.cfg))
    e.canary_edit = edit
    s = e.open(e.public_spec('gt'), factory)
    s.setup()
    for c in ('sed -i.bak "s/a - b/a + b/" x/calc.go && rm x/calc.go.bak',
              'printf "package y\\nfunc Mul(a, b int) int { return a * b }\\n" > y/mul.go'):
        assert s.sb.execute(c)['returncode'] == 0, c
    g = s.grade(e.grade_inputs('gt'))
    s.close()
    assert g['reward'] == 1.0 and g['canary'] == 'ok: 3 edited test(s) failed; consistency ok: 1 of 3 edited', g
    assert {i.rsplit('::', 1)[1] for i in calls} == {'TestAdd', 'TestMul', 'TestOld'}


def test_a_broken_verifier_edit_is_an_error_not_a_silent_0(tmp_path):
    """Review of #35: a verifier edit that returns the source unchanged, a non-str, an unparseable file, or raises
    must not grade an honest patch 0 (which reads as a forge): it is a GradeError the regrade reports."""
    env, task, factory, patch, start = _captured(tmp_path, lambda sb: sb.execute(FIX, cwd='/testbed'))

    def boom(path, source, test_id):
        raise RuntimeError('bug')
    for edit, why in ((lambda p, s, i: s, 'unchanged'), (lambda p, s, i: 42, 'returned int'),
                      (lambda p, s, i: s + '\ndef (:', 'unparseable'), (boom, 'raised')):
        env.canary_edit = edit
        with pytest.raises(GradeError, match=why):
            env.regrade(task, patch, start, factory)
    env.canary_edit = lambda p, s, i: None                                   # leaving every test out: no canary
    g = env.regrade(task, patch, start, factory)
    assert g['reward'] == 1.0 and g['canary'].startswith('skipped')


def test_load_canary_edit_takes_a_module_or_a_file(tmp_path):
    from hosted_env.envs.swe.cli import load_canary_edit
    f = tmp_path / 'e.py'
    f.write_text('def edit(path, source, test_id):\n    return None\nNOT = 1\n')
    assert load_canary_edit(f'{f}:edit')('p', 's', 't') is None
    assert load_canary_edit('hosted_env.envs.swe.forgery:negate_python').__name__ == 'negate_python'
    for bad in (str(f), f'{f}:missing', f'{f}:NOT', f'{tmp_path}/nope.py:edit', 'no.such.module:x', ':edit'):
        with pytest.raises((ImportError, OSError, AttributeError, ValueError)):
            load_canary_edit(bad)
    dc = tmp_path / 'dc.py'                                         # registered in sys.modules: dataclasses work
    dc.write_text('from dataclasses import dataclass\n@dataclass\nclass C:\n    x: int = 1\n'
                  'def edit(path, source, test_id):\n    return None if C().x else source\n')
    assert load_canary_edit(f'{dc}:edit')('p', 's', 't') is None


def _cli(tmp_path, stack, att, tasks, sandbox='local', extra=()):  # noqa: F811
    from hosted_env.cli import main
    (tmp_path / 'att.json').write_text(json.dumps(att))
    cfg = tmp_path / 'regrade.toml'
    # no grader_user: the re-grade grades as the receipts say the rollouts were graded (root in this stack)
    cfg.write_text(f'[envs.swe]\nkind = "mimo_swe"\ntasks = "{tasks}"\nsandbox = "{sandbox}"\n'
                   f'require_image_digest = false\ncleanup = false\n[envs.swe.sandbox_opts]\n'
                   f'base_dir = "{tmp_path}"\n')
    with pytest.raises(SystemExit) as e:
        main(['regrade', '--attachment', str(tmp_path / 'att.json'), '--gateway', stack['gw'].url, '--tasks', tasks,
              '--config', str(cfg), *extra])
    return e.value.code


def test_regrade_cli_end_to_end(stack, tmp_path, capsys, monkeypatch):  # noqa: F811
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
    assert lines[-1]['canary_edit'] == 'built-in'
    f = tmp_path / 'my_edit.py'                                       # the verifier's own edit, from a file
    f.write_text(VERIFIER_EDIT + '\n\ndef logged(path, source, test_id):\n    return edit(path, source, test_id)\n')
    from hosted_env.envs.swe.env import SweEnv
    seen, real = [], SweEnv.regrade
    monkeypatch.setattr(SweEnv, 'regrade', lambda self, *a, **kw: seen.append(self.canary_edit) or real(self, *a, **kw))
    assert _cli(tmp_path, stack, att, tasks, extra=('--canary-edit', f'{f}:logged')) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['canary_edit'] == f'{f}:logged'
    assert len(seen) == 8 and all(getattr(e, '__name__', '') == 'logged' for e in seen)   # this stack's task has
    seen.clear()                                                        # no expected tests, so no canary runs
    assert _cli(tmp_path, stack, att, tasks) == 0 and seen and all(e is None for e in seen)
    capsys.readouterr()
    assert _cli(tmp_path, stack, att, tasks, extra=('--canary-edit', f'{f}:missing')) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['error'] == 'bad_canary_edit'
    broken = tmp_path / 'broken_edit.py'
    broken.write_text('raise RuntimeError("at import")\n')
    assert _cli(tmp_path, stack, att, tasks, extra=('--canary-edit', f'{broken}:edit')) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['error'] == 'bad_canary_edit'
    mine = tmp_path / 'my-tasks.jsonl'                                # the verifier's copy says something else
    rows = [json.loads(x) for x in open(tasks) if x.strip()]
    mine.write_text(''.join(json.dumps({**x, 'problem_statement': 'edited'}) + '\n' for x in rows))
    assert _cli(tmp_path, stack, att, str(mine)) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['error'] == 'task_hash_mismatch'


def _group(stack, fixed):  # noqa: F811
    from conftest import FIX as ROLLOUT_FIX

    from hosted_env.infra.client import GatewayClient
    r = CharRenderer()
    gw = GatewayClient(stack['gw'].url, '5H0', stack['miners']['5H0'])
    sampler = ScriptedSampler(r.encode, r.turn_end_id, {n: [ROLLOUT_FIX, 'Fixed.'] if n in fixed else ['Nothing.']
                                                        for n in range(8)})
    return run_task(gw, EPOCH, 'swe', sampler)['attachment'], stack['store']._env('swe').cfg['tasks']


def test_regrade_grades_only_verified_patches(stack, tmp_path, capsys):  # noqa: F811
    """A group that fails only the pass/fail gate (here: every rollout fixed) is still re-graded -- but only from
    verified data: a patch swapped into its attachment is refused, never graded."""
    import base64
    att, tasks = _group(stack, fixed=set(range(8)))
    capsys.readouterr()
    assert _cli(tmp_path, stack, att, tasks) == 0                    # gate fails, every rollout still re-graded
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['regraded'] == 8
    bad = json.loads(json.dumps(att))
    bad['rollouts'][0]['artifacts']['model_patch'] = base64.b64encode(b'diff --git a/x b/x\n').decode()
    assert _cli(tmp_path, stack, bad, tasks) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['error'] == 'artifact_mismatch'


def test_regrade_refuses_other_sandbox_limits(stack, tmp_path, capsys):  # noqa: F811
    att, tasks = _group(stack, fixed={0})
    capsys.readouterr()
    assert _cli(tmp_path, stack, att, tasks, sandbox='docker') == 1  # receipts say 'local'; nothing is graded
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out['error'] == 'sandbox_limits_mismatch' and out['receipt'] == 'local'


def test_regrade_refuses_a_grader_user_typo(tmp_path):
    env, task, factory, patch, start = _captured(tmp_path, lambda sb: sb.execute(FIX, cwd='/testbed'))
    env.cfg['grader_user'] = 'noboby'
    with pytest.raises(GradeError, match='grader_user'):
        env.regrade(task, patch, start, factory)
