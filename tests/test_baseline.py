"""hosted_env.baseline: expected tests from an empty rollout graded in collect mode (LocalSandbox, no docker).
The task's scripts run this interpreter's own pytest, so the tests need nothing installed system-wide."""
import json
import os
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


def test_gotest_cases_top_level_and_unbuilt_packages():
    from hosted_env.baseline import gotest_cases
    ev = [{'Action': 'run', 'Package': 'm/a', 'Test': 'TestA'},
          {'Action': 'fail', 'Package': 'm/a', 'Test': 'TestA/sub'},       # subtests are not ids
          {'Action': 'fail', 'Package': 'm/a', 'Test': 'TestA'},
          {'Action': 'pass', 'Package': 'm/a', 'Test': 'TestA'},           # -count=2: a fail is kept
          {'Action': 'skip', 'Package': 'm/a', 'Test': 'TestS'},
          {'Action': 'pass', 'Package': 'm/a', 'Test': 'TestP'},
          {'Action': 'fail', 'Package': 'm/a'},
          {'Action': 'fail', 'Package': 'm/new'}]                           # did not build: no test event
    rep = ('\n'.join(json.dumps(e) for e in ev) + '\nnot json\n').encode()
    cases, unbuilt = gotest_cases(rep)
    assert cases == {'m/a::TestA': 'fail', 'm/a::TestS': 'skip', 'm/a::TestP': 'pass'}
    assert unbuilt == {'m/new'}


def test_go_run_patterns_and_selection():
    from hosted_env.baseline import go_run_patterns, go_select, static_go_test_ids
    script = ("set -e\ncd /testbed\ngo test -v -count=1 \\\n  -run 'TestFoo|TestBar/case_1' ./pkg/x  # comment\n"
              'go test ./pkg/y -run="^(TestBaz|TestQ)$"\n')
    pats = go_run_patterns([script])
    assert pats == ['TestFoo|TestBar', '^(TestBaz|TestQ)$']               # continuation joined, '/sub' dropped
    assert go_run_patterns(['go test ./...\n']) is None                    # no -run: everything selected
    src = ('package x\nimport "testing"\nfunc TestFoo(t *testing.T) {}\nfunc TestFooBar(t *testing.T) {}\n'
           'func TestQ(t *testing.T) {}\nfunc helper(t *testing.T) {}\nfunc Testify(t *testing.T) {}\n'
           'func BenchmarkX(b *testing.B) {}\n')
    names = static_go_test_ids(src)
    assert names == ['TestFoo', 'TestFooBar', 'TestQ']                     # Testify is not a test: lower case
    assert go_select(names, pats) == ['TestFoo', 'TestFooBar', 'TestQ']    # unanchored, like go test
    assert go_select(names, ['^TestFoo$']) == ['TestFoo']
    assert go_select(names, ['(unclosed']) == []                           # cannot vouch: select nothing
    assert go_select(names, None) == names


def test_go_test_commands_attribute_run_to_their_packages():
    """Each go test's -run applies to its own packages only: package x must not be given package y's pattern
    (that would expect tests the script never runs there and score an honest fix 0)."""
    from hosted_env.baseline import go_patterns_for_dir, go_test_commands
    script = ('cd /testbed\n'
              'go test -v -count=1 -timeout 600s ./pkg/x -run \'TestX1|TestX2/sub\'\n'
              'go test ./pkg/y/... -run=TestY\n'
              'cd internal && go test -vet off . -run TestI\n'
              'go test example.com/m/api\n')
    cmds = go_test_commands([('', script)], '/testbed')
    assert cmds == [([('dir', 'pkg/x')], 'TestX1|TestX2'), ([('tree', 'pkg/y')], 'TestY'),
                    ([('dir', 'internal')], 'TestI'), ([('import', 'example.com/m/api')], None)]
    assert go_patterns_for_dir(cmds, 'pkg/x') == ['TestX1|TestX2']
    assert go_patterns_for_dir(cmds, 'pkg/y/deep') == ['TestY']
    assert go_patterns_for_dir(cmds, 'internal') == ['TestI']
    assert go_patterns_for_dir(cmds, 'api') is None                     # no -run: the whole package runs
    assert go_patterns_for_dir(cmds, 'other') == []                     # no command covers it
    # ./... from the repo root covers everything; a cd the parser cannot follow adds nothing
    assert go_patterns_for_dir(go_test_commands([('', 'go test ./... -run TestA\n')], '/testbed'), 'a/b') == ['TestA']
    assert go_patterns_for_dir(go_test_commands([('', 'cd "$D"\ngo test . -run TestA\n')], '/testbed'), '') == []
    # a continuation line keeps -run with its command; a value flag's value is not a package
    cmds = go_test_commands([('', 'go test -count 1 \\\n  -tags "a b" ./p -run TestP\n')], '/testbed')
    assert cmds == [([('dir', 'p')], 'TestP')]


def test_go_test_commands_follow_the_scripts_own_directory():
    from hosted_env.baseline import go_patterns_for_dir, go_test_commands
    cmds = go_test_commands([('ci/run.sh', 'cd "$(dirname "$0")" && go test . -run TestC\n'),
                             ('', 'cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.." && go test ./x -run TestX\n')],
                            '/testbed')
    assert go_patterns_for_dir(cmds, 'ci') == ['TestC']
    assert go_patterns_for_dir(cmds, 'x') == []          # the test command's own dir is the repo root: '..' leaves it


FAKE_GO_TOOLCHAIN = r'''#!/bin/bash
# `go test` for the fake module example.com/m: package x builds at base (TestOld passes, TestAdd fails until
# Add adds); package y's new test file needs Mul, so y does not build until the fix adds it.
[ "$1" = test ] || exit 0
shift; json=0
if [ "$1" = -json ]; then json=1; shift; fi
pkg=""
for a in "$@"; do case "$a" in ./*) pkg="${a#./}";; esac; done
ev() { if [ $json = 1 ]; then printf '{"Action":"%s","Package":"example.com/m/%s","Test":"%s"}\n' "$1" "$pkg" "$2"; fi; }
if [ "$pkg" = x ]; then
  ev pass TestOld
  if grep -q 'a + b' x/calc.go; then ev pass TestAdd; exit 0; fi
  ev fail TestAdd; exit 1
fi
if [ "$pkg" = y ]; then
  if grep -q 'func Mul' y/*.go; then ev pass TestMul; ev pass TestOther; exit 0; fi
  if [ $json = 1 ]; then printf '{"Action":"fail","Package":"example.com/m/y"}\n'; fi
  exit 2
fi
exit 0
'''


def _go_task(tmp_path, monkeypatch):
    bindir = tmp_path / 'toolchain'
    bindir.mkdir()
    (bindir / 'go').write_text(FAKE_GO_TOOLCHAIN)
    (bindir / 'go').chmod(0o755)
    monkeypatch.setenv('PATH', f'{bindir}:{os.environ["PATH"]}')
    tpl = tmp_path / 'tpl'
    tpl.mkdir()
    _run('git init -q -b master && mkdir x y && '
         'printf "package x\\nfunc Add(a, b int) int { return a - b }\\n" > x/calc.go && '
         'printf "package x\\nimport \\"testing\\"\\nfunc TestOld(t *testing.T) {}\\n" > x/old_test.go && '
         'printf "package y\\n" > y/y.go && git add -A && git commit -qm base', tpl)
    tp = tmp_path / 'tp'
    (tp / 'x').mkdir(parents=True)
    (tp / 'y').mkdir()
    (tp / 'x' / 'calc_test.go').write_text('package x\nimport "testing"\n'
                                           'func TestAdd(t *testing.T) { if Add(2, 3) != 5 { t.Fatal() } }\n')
    (tp / 'y' / 'mul_test.go').write_text('package y\nimport "testing"\n'
                                          'func TestMul(t *testing.T) { if Mul(2, 3) != 6 { t.Fatal() } }\n'
                                          'func TestOther(t *testing.T) {}\n')
    (tp / 'mimo_test_command.sh').write_text('cd "$(dirname "$0")"\n'
                                             "go test -v ./x -run 'TestAdd|TestOld'; a=$?\n"
                                             "go test ./y -run '^TestMul$'; b=$?\n"
                                             '[ $a -eq 0 ] && [ $b -eq 0 ]\n')
    test_patch = _run('git init -q && git add -A && git diff --cached --no-color', tp).stdout
    path = tmp_path / 'tasks.jsonl'
    path.write_text(json.dumps({'id': 'gt', 'image': str(tpl), 'cwd': '/testbed', 'lang': 'go',
                                'problem_statement': 'Add subtracts; add Mul', 'test_patch': test_patch,
                                'test_command': 'bash /testbed/mimo_test_command.sh'}) + '\n')
    env = SweEnv('swe', {'kind': 'mimo_swe', 'tasks': str(path), 'cleanup': False, 'grader_user': 'root',
                         'require_image_digest': False})
    base = tmp_path / 'sb'
    base.mkdir()
    return env, path, lambda image, cwd: LocalSandbox(image, cwd, base_dir=str(base))


def test_derive_go_then_grade_with_the_expected_tests(tmp_path, monkeypatch):
    """Go: x::TestAdd fails at base (F2P) and x::TestOld passes (P2P); package y does not build at base, so its
    expected test comes from the test file, selected by y's own -run (TestMul, not TestOther and not x's
    pattern). The derived lists then drive the grader's second check through the go shim."""
    from hosted_env.baseline import derive
    env, path, factory = _go_task(tmp_path, monkeypatch)
    rec = derive(env, factory, 'gt', runs=2)
    assert rec['report_kind'] == 'gotest', rec
    assert rec['fail_to_pass'] == ['example.com/m/x::TestAdd', 'example.com/m/y::TestMul']
    assert rec['pass_to_pass'] == ['example.com/m/x::TestOld']
    b = rec['baseline']
    assert b['unbuilt'] == ['example.com/m/y'] and b['static_ids'] == 1 and b['flaky'] == []
    row = json.loads(path.read_text())
    row.update(report_kind='gotest', fail_to_pass=rec['fail_to_pass'], pass_to_pass=rec['pass_to_pass'])
    path.write_text(json.dumps(row) + '\n')
    env2 = SweEnv('swe', {**env.cfg})

    def grade_after(*cmds):
        s = env2.open(env2.public_spec('gt'), factory)
        s.setup()
        for c in cmds:
            assert s.sb.execute(c)['returncode'] == 0, c
        g = s.grade(env2.grade_inputs('gt'))
        s.close()
        return g
    fix_x = 'sed -i.bak "s/a - b/a + b/" x/calc.go && rm x/calc.go.bak'
    fix_y = 'printf "package y\\nfunc Mul(a, b int) int { return a * b }\\n" > y/mul.go'
    g = grade_after(fix_x, fix_y)
    assert g['reward'] == 1.0 and g['second_check'] is True, g.get('grade_detail')
    assert g['go_report_shim'].endswith('/toolchain/go')
    assert grade_after(fix_x)['reward'] == 0.0


def test_amend_js_reporter():
    from hosted_env.baseline import amend_js_reporter as a
    j = '--json --outputFile="${MIMO_REPORT:-/dev/null}"'
    assert a('cd /testbed\nnpx jest src/a.test.js --ci\n') == (f'cd /testbed\nnpx jest src/a.test.js --ci {j}\n', 'jest')
    assert a('yarn jest -t "adds two" 2>&1 | tail -50\n')[0] == f'yarn jest -t "adds two" {j} 2>&1 | tail -50\n'
    assert a('node_modules/.bin/jest x.test.ts && echo ok')[0] == f'node_modules/.bin/jest x.test.ts {j} && echo ok'
    assert a('(cd pkg && npx jest a.test.js)')[0] == f'(cd pkg && npx jest a.test.js {j})'
    assert a('npx jest "a|b.test.js" > out.log')[0] == f'npx jest "a|b.test.js" {j} > out.log'
    assert a('npx vitest run test/x.spec.ts')[1] == 'junit'
    assert '--reporter=junit --outputFile=' in a('npx vitest run test/x.spec.ts')[0]
    new, kind = a('./node_modules/.bin/mocha test/a.js --timeout 5000')
    assert kind == 'junit' and new.endswith('--timeout 5000 --reporter xunit --reporter-options '
                                            'output="${MIMO_REPORT:-/dev/null}"')
    for bad in ('npx jest a.test.js\nnpx jest b.test.js\n',          # two runs would overwrite one report
                'npx jest a.test.js --json\n',                       # already has a reporter
                'npx mocha a.js --reporter dot\n',
                "bash -c 'npx jest a.test.js'\n",                    # inside a quoted string
                'npx jest \\\n  a.test.js\n',                        # continued on the next line
                'npm test\n', 'node test/run.js\n', '# npx jest a.test.js\n'):
        assert a(bad) is None, bad
    # a comment that only mentions jest does not count as a run
    assert a('# runs jest on the fixed file\nnpx jest a.test.js\n')[1] == 'jest'


def _patch_adding(tmp_path, files: dict) -> str:
    """A test patch (git diff, as the task data has it) that adds `files` to an empty tree."""
    d = tmp_path / ('p%d' % len(list(tmp_path.iterdir())))
    d.mkdir()
    for name, text in files.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text)
    return _run('git init -q && git add -A && git diff --cached --no-color --full-index', d).stdout


def test_amend_test_patch_rebuilds_a_new_script(tmp_path):
    from hosted_env.baseline import amend_test_patch
    script = 'set -e\ncd /testbed\nnpx jest src/a.test.js --ci 2>&1 | tail -30\n'
    tp = _patch_adding(tmp_path, {'src/a.test.js': 'test("x", () => {})\n', 'mimo_test_command.sh': script})
    out = amend_test_patch(tp, 'bash /testbed/mimo_test_command.sh')
    assert out['report_kind'] == 'jest' and 'test_command' not in out
    repo = tmp_path / 'repo'
    repo.mkdir()
    _run('git init -q', repo)
    (repo / 'tp.diff').write_text(out['test_patch'])
    _run('git apply --check tp.diff && git apply tp.diff && rm tp.diff', repo)
    got = (repo / 'mimo_test_command.sh').read_text()
    assert got == script.replace('--ci ', '--ci --json --outputFile="${MIMO_REPORT:-/dev/null}" ')
    assert (repo / 'src' / 'a.test.js').read_text() == 'test("x", () => {})\n'     # other files untouched
    # the rebuilt index line carries the blob id of the amended content
    blob = _run('git hash-object mimo_test_command.sh', repo).stdout.strip()
    assert f'..{blob}' in out['test_patch']


def test_amend_test_patch_refusals(tmp_path):
    from hosted_env.baseline import amend_test_patch
    one = _patch_adding(tmp_path, {'mimo_test_command.sh': 'npx jest a.test.js\n'})
    # runs in two places (patch script + test command), or two runs in the script: not amended
    assert amend_test_patch(one, 'npx jest b.test.js') is None
    assert amend_test_patch(_patch_adding(tmp_path, {'mimo_test_command.sh': 'npx jest a\nnpx jest b\n'}), 'x') is None
    # a modification of an existing script, not a new file
    base = tmp_path / 'base'
    base.mkdir()
    _run('git init -q && printf "echo hi\\n" > run.sh && git add -A && git commit -qm b && '
         'printf "echo hi\\nnpx mocha t.js\\n" > run.sh', base)
    mod = _run('git diff --no-color', base).stdout
    assert amend_test_patch(mod, 'bash run.sh') is None
    # no runner at all: nothing to amend; a test or doc file that mentions jest is not a run
    assert amend_test_patch(_patch_adding(tmp_path, {'mimo_test_command.sh': 'node t.js\n'}), 'x') is None
    assert amend_test_patch(_patch_adding(tmp_path, {'docs.md': 'run jest here\n'}), 'npx jest t.js')['test_command']
    # the run in the outer test command alone is amended there
    assert amend_test_patch(_patch_adding(tmp_path, {'t.js': 'x\n'}), 'npx vitest run t.js')['report_kind'] == 'junit'


FAKE_NPX = r'''#!/bin/bash
# `npx jest <file> [--json --outputFile=F]`: test "adds" fails until sum.js adds, "keeps" always passes.
[ "$1" = jest ] || exit 0
out=""
for a in "$@"; do case "$a" in --outputFile=*) out="${a#--outputFile=}";; esac; done
if grep -q 'a + b' sum.js; then s=passed; rc=0; else s=failed; rc=1; fi
if [ -n "$out" ]; then
  printf '{"testResults":[{"status":"x","assertionResults":[{"fullName":"sum adds","title":"adds","status":"%s"},{"fullName":"sum keeps","title":"keeps","status":"passed"}]}]}' "$s" > "$out"
fi
exit $rc
'''


def test_derive_js_amends_the_harness_then_grades(tmp_path, monkeypatch):
    """jest: the one `npx jest` run is amended to write --json to $MIMO_REPORT; at base "sum adds" fails (F2P)
    and "sum keeps" passes (P2P); the amended test patch comes back with them, and grading the amended task
    with the expected tests scores the fix 1 and the unfixed repo 0."""
    from hosted_env.baseline import derive
    bindir = tmp_path / 'node'
    bindir.mkdir()
    (bindir / 'npx').write_text(FAKE_NPX)
    (bindir / 'npx').chmod(0o755)
    monkeypatch.setenv('PATH', f'{bindir}:{os.environ["PATH"]}')
    tpl = tmp_path / 'tpl'
    tpl.mkdir()
    _run('git init -q -b master && printf "module.exports = (a, b) => a - b\\n" > sum.js && git add -A && '
         'git commit -qm base', tpl)
    tp = tmp_path / 'tp'
    tp.mkdir()
    (tp / 'sum.test.js').write_text('test("adds", () => {})\n')
    (tp / 'mimo_test_command.sh').write_text('cd "$(dirname "$0")"\nnpx jest sum.test.js --ci\n')
    test_patch = _run('git init -q && git add -A && git diff --cached --no-color', tp).stdout
    path = tmp_path / 'tasks.jsonl'
    path.write_text(json.dumps({'id': 'jt', 'image': str(tpl), 'cwd': '/testbed', 'lang': 'javascript',
                                'problem_statement': 'sum subtracts', 'test_patch': test_patch,
                                'test_command': 'bash /testbed/mimo_test_command.sh'}) + '\n')
    cfg = {'kind': 'mimo_swe', 'tasks': str(path), 'cleanup': False, 'grader_user': 'root',
           'require_image_digest': False}
    env = SweEnv('swe', cfg)
    base = tmp_path / 'sb'
    base.mkdir()

    def factory(image, cwd):
        return LocalSandbox(image, cwd, base_dir=str(base))
    rec = derive(env, factory, 'jt', runs=2)
    assert rec['report_kind'] == 'jest', rec
    assert rec['fail_to_pass'] == ['sum adds'] and rec['pass_to_pass'] == ['sum keeps']
    assert '--json --outputFile=' in rec['test_patch'] and rec['baseline']['amended'] == ['test_patch']
    row = json.loads(path.read_text())
    row.update({k: rec[k] for k in ('report_kind', 'fail_to_pass', 'pass_to_pass', 'test_patch')})
    path.write_text(json.dumps(row) + '\n')
    env2 = SweEnv('swe', cfg)

    def grade_after(*cmds):
        s = env2.open(env2.public_spec('jt'), factory)
        s.setup()
        for c in cmds:
            assert s.sb.execute(c)['returncode'] == 0, c
        g = s.grade(env2.grade_inputs('jt'))
        s.close()
        return g
    g = grade_after('printf "module.exports = (a, b) => a + b\\n" > sum.js')
    assert g['reward'] == 1.0 and g['second_check'] is True, g.get('grade_detail')
    assert grade_after()['reward'] == 0.0
