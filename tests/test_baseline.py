"""hosted_env.baseline: expected tests from an empty rollout graded in collect mode (LocalSandbox, no docker).
The task's scripts run this interpreter's own pytest, so the tests need nothing installed system-wide."""
import json
import sys

from conftest import _run

from hosted_env.baseline import junit_cases, pytest_selection, select, static_test_ids
from hosted_env.envs import SweEnv
from hosted_env.envs.swe import report_verdict
from hosted_env.providers.sandbox import LocalSandbox

XUNIT1 = b'''<testsuites><testsuite>
<testcase classname="" name="tests.test_new" file="tests/test_new.py"><error message="collection failure"/></testcase>
<testcase classname="tests.test_one" name="test_a" file="tests/test_one.py"><failure message="x"/></testcase>
<testcase classname="tests.test_one" name="test_p[1]" file="tests/test_one.py"/>
<testcase classname="tests.test_one" name="test_p[2]" file="tests/test_one.py"><failure/></testcase>
<testcase classname="tests.test_one.TestK" name="test_m" file="tests/test_one.py"/>
<testcase classname="tests.test_one" name="test_s" file="tests/test_one.py"><skipped/></testcase>
</testsuite></testsuites>'''


def test_junit_cases():
    cases, broken = junit_cases(XUNIT1)
    assert broken == ['tests/test_new.py']
    assert cases == {'tests/test_one.py::test_a': 'fail', 'tests/test_one.py::test_p[1]': 'pass',
                     'tests/test_one.py::test_p[2]': 'fail', 'tests/test_one.py::TestK::test_m': 'pass',
                     'tests/test_one.py::test_s': 'skip'}


def test_static_ids_and_selection():
    src = ('import pytest\nfrom calc import mul\ndef test_mul(): pass\nasync def test_amul(): pass\n'
           'def helper(): pass\nclass TestK:\n    def test_m(self): pass\n    def other(self): pass\n'
           'class Helper:\n    def test_no(self): pass\n')
    ids = static_test_ids('tests/test_new.py', src)
    assert ids == ['tests/test_new.py::test_mul', 'tests/test_new.py::test_amul', 'tests/test_new.py::TestK::test_m']
    assert static_test_ids('x.py', 'def broken(:') == []
    assert select(ids, [], []) == ids
    assert select(ids, [], ['mul and not amul']) == ['tests/test_new.py::test_mul']
    assert select(ids, [], ['TestK']) == ['tests/test_new.py::TestK::test_m']       # a class name matches too
    assert select(ids, ['tests/test_new.py::test_mul[2]'], []) == ['tests/test_new.py::test_mul']   # param instance
    assert select(ids, ['tests/test_new.py'], []) == ids                                   # the whole file


def test_pytest_selection_reads_the_inner_scripts():
    s = ('cd /testbed\nrc=0\npython -m pytest -v -rA /testbed/tests/a.py::test_x \\\n  tests/b.py::K::test_y\n'
         'pytest tests/c.py -k "nested or top" -x\nPYTEST=1 python3 -m pytest -k=foo tests\necho pytest done\n')
    nodeids, ks = pytest_selection([s], '/testbed')
    assert nodeids == ['tests/a.py::test_x', 'tests/b.py::K::test_y'] and ks == ['nested or top', 'foo']


def test_report_verdict_accepts_parametrised_instances():
    junit = (b'<testsuite><testcase classname="t.test_m" name="test_p[1]"/>'
             b'<testcase classname="t.test_m" name="test_p[2]"/><testcase classname="t.test_m" name="test_q[1]"/>'
             b'<testcase classname="t.test_m" name="test_q[2]"><failure/></testcase></testsuite>')
    assert report_verdict('junit', junit, ['t/test_m.py::test_p'], [])[0] is True
    ok, why = report_verdict('junit', junit, ['t/test_m.py::test_q'], [])      # one instance fails: the id fails
    assert not ok and 'not passing' in why
    assert not report_verdict('junit', junit, ['t/test_m.py::test_r'], [])[0]   # no instance at all: missing


def _task(tmp_path, base_extra='', calc_test=None, select='-k "add or zero or mul or old"'):
    """Base: calc.add subtracts, tests/test_old.py passes. Test patch: tests/test_calc.py (test_add fails at base,
    test_zero passes, test_p is deselected by -k) and tests/test_new.py, which imports calc.mul (not in the base,
    so the file does not collect): test_mul is selected, test_other is not. base_extra: more shell for the base
    commit; calc_test / select replace tests/test_calc.py and the pytest selection."""
    tpl = tmp_path / 'tpl'
    tpl.mkdir()
    _run('git init -q -b master && printf "def add(a, b):\\n    return a - b\\n" > calc.py && mkdir tests && '
         f'printf "def test_old(): pass\\n" > tests/test_old.py && {base_extra or "true"} && git add -A && '
         'git commit -qm base', tpl)
    tp = tmp_path / 'tp'
    tp.mkdir()
    (tp / 'tests').mkdir()
    (tp / 'tests' / 'test_calc.py').write_text(calc_test or (
        'import pytest\nfrom calc import add\ndef test_add(): assert add(2, 3) == 5\n'
        'def test_zero(): assert add(0, 0) == 0\n@pytest.mark.parametrize("x", [1])\ndef test_p(x): pass\n'))
    (tp / 'tests' / 'test_new.py').write_text('from calc import mul\ndef test_mul(): assert mul(2, 3) == 6\n'
                                              'def test_other(): pass\n')
    (tp / 'mimo_test_command.sh').write_text(
        f'cd "$(dirname "$0")" && PYTHONPATH=. {sys.executable} -m pytest -p no:cacheprovider tests {select}\n')
    test_patch = _run('git init -q && git add -A && git diff --cached --no-color', tp).stdout
    path = tmp_path / 'tasks.jsonl'
    path.write_text(json.dumps({'id': 'bt', 'image': str(tpl), 'cwd': '/testbed', 'lang': 'python',
                                'problem_statement': 'add subtracts; add mul', 'test_patch': test_patch,
                                'test_command': 'bash /testbed/mimo_test_command.sh'}) + '\n')
    env = SweEnv('swe', {'kind': 'mimo_swe', 'tasks': str(path), 'cleanup': False, 'grader_user': 'root',
                         'require_image_digest': False})
    base = tmp_path / 'sb'
    base.mkdir()
    return env, path, lambda image, cwd: LocalSandbox(image, cwd, base_dir=str(base))


def test_derive_then_grade_with_the_expected_tests(tmp_path):
    from hosted_env.baseline import derive
    env, path, factory = _task(tmp_path)
    rec = derive(env, factory, 'bt', runs=2)
    assert rec['report_kind'] == 'junit'
    assert rec['fail_to_pass'] == ['tests/test_calc.py::test_add', 'tests/test_new.py::test_mul']
    assert rec['pass_to_pass'] == ['tests/test_calc.py::test_zero', 'tests/test_old.py::test_old']
    b = rec['baseline']
    assert b['uncollectable'] == ['tests/test_new.py'] and b['static_ids'] == 1 and b['flaky'] == []
    # the derived lists drive the grader's second check
    row = json.loads(path.read_text())
    row.update(report_kind='junit', fail_to_pass=rec['fail_to_pass'], pass_to_pass=rec['pass_to_pass'])
    path.write_text(json.dumps(row) + '\n')
    env2 = SweEnv('swe', {**env.cfg})

    def grade_after(*cmds):
        s = env2.open(env2.public_spec('bt'), factory)
        s.setup()
        for c in cmds:
            assert s.sb.execute(c)['returncode'] == 0, c
        g = s.grade(env2.grade_inputs('bt'))
        s.close()
        return g
    fix = ('sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak && '
           'printf "def mul(a, b):\\n    return a * b\\n" >> calc.py')
    g = grade_after(fix)
    assert g['reward'] == 1.0 and g['second_check'] is True
    g = grade_after('sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak')   # half a fix: test_new still broken
    assert g['reward'] == 0.0


def test_repo_addopts_are_kept(tmp_path):
    """Review of PR #18: the grader must not clear the repo's own pytest addopts (EXP-018 002371 crashed at
    start-up without them). Here the base pytest.ini loads a plugin that provides the fixture the tests use:
    with `-o addopts=` injected, test_add errors at base and after the fix alike, so the fix would score 0."""
    from hosted_env.baseline import derive
    plug = ("printf 'import pytest\\n@pytest.fixture\\ndef answer():\\n    return 5\\n' > plug.py && "
            "printf '[pytest]\\naddopts = -p plug\\n' > pytest.ini")
    test = 'from calc import add\ndef test_add(answer): assert add(2, 3) == answer\n'
    env, path, factory = _task(tmp_path, base_extra=plug, calc_test=test, select='')
    rec = derive(env, factory, 'bt', runs=1)
    assert 'tests/test_calc.py::test_add' in rec['fail_to_pass'] and rec['pass_to_pass'] == ['tests/test_old.py::test_old']
    row = json.loads(path.read_text())
    row.update(report_kind='junit', fail_to_pass=rec['fail_to_pass'], pass_to_pass=rec['pass_to_pass'])
    path.write_text(json.dumps(row) + '\n')
    env2 = SweEnv('swe', {**env.cfg})
    s = env2.open(env2.public_spec('bt'), factory)
    s.setup()
    fix = ('sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak && '
           'printf "def mul(a, b):\\n    return a * b\\n" >> calc.py')
    assert s.sb.execute(fix)['returncode'] == 0
    g = s.grade(env2.grade_inputs('bt'))
    s.close()
    assert g['reward'] == 1.0 and g['second_check'] is True
