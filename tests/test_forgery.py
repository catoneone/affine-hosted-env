"""Forgery checks of grader isolated-v5 (envs/swe/forgery.py): the patch audit and the canary re-runs."""
import pytest
from test_envs import _pytest_task

from hosted_env.envs.swe.forgery import added_lines, patch_audit_hits
from hosted_env.infra.providers.sandbox import LocalSandbox

PATCH = '''diff --git a/pkg/run.go b/pkg/run.go
index 1..2 100644
--- a/pkg/run.go
+++ b/pkg/run.go
@@ -1,2 +1,3 @@
 package pkg
+func init() { fmt.Println("--- PASS: TestAdd (0.00s)"); os.Exit(0) }
diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"
new file mode 100644
--- /dev/null
+++ "b/caf\\303\\251.py"
@@ -0,0 +1,2 @@
+++ 0 or open(os.environ['MIMO_REPORT'], 'w')
+x = 1
diff --git a/tests/test_x.py b/tests/test_x.py
--- a/tests/test_x.py
+++ b/tests/test_x.py
@@ -1 +1,2 @@
+print('<testcase')
'''


def test_added_lines_never_drop_or_misfile_a_line():
    """A quoted path still starts its own file, and an added line that begins with '++ ' (valid Python) is content
    inside a hunk, not a '+++' header."""
    got = added_lines(PATCH)
    quoted = next(k for k in got if 'caf' in k)
    assert got[quoted] == ["++ 0 or open(os.environ['MIMO_REPORT'], 'w')", 'x = 1']
    assert got['pkg/run.go'] == ['func init() { fmt.Println("--- PASS: TestAdd (0.00s)"); os.Exit(0) }']
    assert added_lines('+stray\n') == {'': ['stray']}


def test_patch_audit_hits_skip_only_reverted_files():
    hits = patch_audit_hits(PATCH, {'tests/test_x.py'})
    assert ('pkg/run.go', '--- PASS:', False) in hits
    assert any(s == 'MIMO_REPORT' and always for _, s, always in hits)        # plumbing: never excused
    assert not any(p == 'tests/test_x.py' for p, _, _ in hits)                 # reset to base anyway
    assert any(p == 'tests/test_x.py' for p, _, _ in patch_audit_hits(PATCH))


def _graded(tmp_path, edit: str, base_line: str = ''):
    env, task, base = _pytest_task(tmp_path)
    s = env.open(env.public_spec(task), lambda image, cwd: LocalSandbox(image, cwd, base_dir=base))
    s.setup()
    if base_line:              # make the "base repo" already contain the string: commit it into the image
        import subprocess
        tpl = env.public_spec(task)['image']
        subprocess.run(['bash', '-c', f'printf "%s\\n" {base_line!r} >> notes.txt && git add -A && '
                        'git -c user.email=t@t -c user.name=t commit -qm notes'], cwd=tpl, check=True)
        s.close()
        s = env.open(env.public_spec(task), lambda image, cwd: LocalSandbox(image, cwd, base_dir=base))
        s.setup()
    assert s.sb.execute(edit)['returncode'] == 0
    g = s.grade(env.grade_inputs(task))
    s.close()
    return g


FIX = 'sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak'


def test_a_fix_that_adds_framework_output_scores_0(tmp_path):
    g = _graded(tmp_path, FIX + " && printf 'MARK = \"<testcase\"\\n' >> calc.py")
    assert g['reward'] == 0.0 and g['grade_detail'].startswith('patch_audit: calc.py adds')
    assert 'verifier_returncode' not in g                                     # decided before the tests ran
    g = _graded(tmp_path, FIX + " && printf 'import os\\nP = os.environ.get(\"MIMO_REPORT\")\\n' > helper.py")
    assert g['reward'] == 0.0 and 'helper.py' in g['grade_detail']


def test_framework_output_the_base_repo_already_has_is_allowed(tmp_path):
    g = _graded(tmp_path, FIX + " && printf 'MARK = \"<testcase\"\\n' >> calc.py", base_line='docs mention <testcase')
    assert g['reward'] == 1.0 and g['second_check'] is True
    g = _graded(tmp_path, FIX + " && printf 'P = \"MIMO_REPORT\"\\n' >> calc.py", base_line='MIMO_REPORT')
    assert g['reward'] == 0.0 and 'MIMO_REPORT' in g['grade_detail']        # plumbing is never excused


def test_an_honest_fix_passes_the_audit(tmp_path):
    g = _graded(tmp_path, FIX)
    assert g['reward'] == 1.0 and g['second_check'] is True and 'grade_detail' not in g


# ---------------------------------------------------------------- canary
def test_test_ids_map_to_functions():
    from hosted_env.envs.swe.forgery import go_file_matches, go_target, pytest_target
    assert pytest_target('tests/test_a.py::TestK::test_m[1-x]') == ('tests/test_a.py', ['TestK'], 'test_m')
    assert pytest_target('tests/test_a.py::test_f') == ('tests/test_a.py', [], 'test_f')
    assert pytest_target('test_a::test_f') is None and pytest_target('a.py::') is None
    assert go_target('github.com/o/r/pkg::TestX') == ('github.com/o/r/pkg', 'TestX')
    assert go_target('pkg::TestX/sub') is None and go_target('TestX') is None
    assert go_file_matches('github.com/o/r/pkg', 'pkg/a_test.go') and go_file_matches('github.com/o/r', 'a_test.go')
    assert not go_file_matches('github.com/o/r/pkg', 'other/a_test.go') and not go_file_matches('x/pkg', 'pkg/a.go')


def test_edit_python_puts_a_failure_first():
    from hosted_env.envs.swe.forgery import edit_python
    src = ('import pytest\n\nclass TestK:\n    @pytest.mark.parametrize("x", [1])\n    def test_m(self, x):\n'
           '        """doc"""\n        assert x\n\nasync def test_a():\n\tawait f()\n')
    out = edit_python(src, ['TestK'], 'test_m', "raise AssertionError('c')")
    assert "    def test_m(self, x):\n        raise AssertionError('c')\n        \"\"\"doc\"\"\"" in out
    assert "async def test_a():\n\traise AssertionError('c')\n\tawait f()" in edit_python(src, [], 'test_a',
                                                                                          "raise AssertionError('c')")
    assert edit_python('def test_x(): assert f()\n', [], 'test_x', 'raise X') == 'def test_x(): raise X; assert f()\n'
    assert edit_python('def test_x(\n    a,\n): assert a\n', [], 'test_x', 'raise X') == \
        'def test_x(\n    a,\n): raise X; assert a\n'
    for bad in (('def test_x():\n    yield 1\n', [], 'test_x'),             # a generator
                ('def test_x():\n  a\ndef test_x():\n  b\n', [], 'test_x'),  # defined twice
                ('def test_x(:\n', [], 'test_x'),                           # does not parse
                ('import pytest\n@pytest.mark.xfail\ndef test_x():\n    a\n', [], 'test_x'),   # would report xfail
                ('class K(T):\n    @unittest.expectedFailure\n    def test_x(self):\n        a\n', ['K'], 'test_x'),
                (src, [], 'test_m'), (src, ['Nope'], 'test_m')):            # not at that level
        assert edit_python(bad[0], bad[1], bad[2], 'raise X') is None


def test_negate_python_flips_the_first_assertion_as_a_test_would():
    """v5's pytest edit (review of #34: an inserted constant comparison is found by an AST scan): the test's own
    first top-level assertion, negated in place; the result must parse to exactly that negation."""
    from hosted_env.envs.swe.forgery import negate_python
    cases = {
        'def test_x():\n    r = f()\n    assert r == 5\n    assert r\n': 'def test_x():\n    r = f()\n    assert r != 5\n',
        'def test_x(): assert add(2,3)==5\n': 'def test_x(): assert add(2,3)!=5\n',
        'def test_x():\n    assert (a) <  b, "m"\n': 'def test_x():\n    assert (a) >=  b, "m"\n',
        'def test_x():\n    assert k not  in d\n': 'def test_x():\n    assert k in d\n',
        'def test_x():\n    assert v is not None\n': 'def test_x():\n    assert v is None\n',
        'def test_x():\n    assert not (a and b)\n': 'def test_x():\n    assert (a and b)\n',
        'def test_x():\n    assert ok\n': 'def test_x():\n    assert not (ok)\n',
        'def test_x():\n    assert a < b < c\n': 'def test_x():\n    assert not (a < b < c)\n',
        'def test_x():\n    assert a or b\n': 'def test_x():\n    assert not (a or b)\n',
        'def test_x():\n    assert all(\n        y for y in z)\n': 'def test_x():\n    assert not (all(\n        y for y in z))\n',
        'class T(unittest.TestCase):\n    def test_x(self):\n        self.assertEqual(f(), 3, msg="m")\n':
            'class T(unittest.TestCase):\n    def test_x(self):\n        self.assertNotEqual(f(), 3, msg="m")\n',
        'async def test_x():\n    assert await g() == 1\n': 'async def test_x():\n    assert await g() != 1\n',
    }
    for src, want in cases.items():
        cls = ['T'] if src.startswith('class') else []
        out = negate_python(src, cls, 'test_x')
        assert out is not None and out.startswith(want.rstrip('\n')), (src, out)
    for src in ('def test_x():\n    if c:\n        return\n    assert a == b\n',   # may not reach the assert
                'def test_x():\n    with pytest.raises(E):\n        f()\n',             # no top-level assertion
                'def test_x():\n    assert (a ==\n            b)\n',                    # operator on another line
                'def test_x():\r    assert a == b\r'):                                 # lone CR: lines disagree
        assert negate_python(src, [], 'test_x') is None, src


def test_negated_assertions_fail_where_the_original_passed():
    """What the edit relies on: when the original statement held, its negation raises."""
    from hosted_env.envs.swe.forgery import negate_python
    env = {'f': lambda: 3, 'd': {'k': 1}, 'k': 'k', 'v': 0}
    for body in ('assert f() == 3', 'assert f() >= 3', 'assert k in d', 'assert v is not None', 'assert d',
                 'assert not []', 'assert f() != 4'):
        src = f'def test_x():\n    {body}\n'
        ns = dict(env)
        exec(src, ns)
        ns['test_x']()                                                      # the original holds
        ns2 = dict(env)
        exec(negate_python(src, [], 'test_x'), ns2)
        with pytest.raises(AssertionError):
            ns2['test_x']()


def test_mutate_python_literal_changes_the_expected_value_in_place(monkeypatch):
    """The least recognisable edit (review of #34 round 2): the literal the first assertion expects, changed to
    another of the same kind and style; the edited test keeps its polarity and shape and fails where the original
    passed. The result must parse to exactly that change."""
    import re as _re

    import hosted_env.envs.swe.forgery as forgery
    from hosted_env.envs.swe.forgery import mutate_python_literal
    for _ in range(100):
        for src, pat, value in (
                ('def test_x():\n    r = f()\n    assert r == 5\n', r'    assert r == (\d+)\n', 5),
                ("def test_x(): assert g() == 'abc'\n", r"def test_x\(\): assert g\(\) == '([a-z0-9]+)'\n", 'abc'),
                ('def test_x():\n    assert 2.5 == h()\n', r'    assert (\d+\.5) == h\(\)\n', 2.5),
                ('class T(unittest.TestCase):\n    def test_x(self):\n        self.assertEqual(f(), 0)\n',
                 r'        self.assertEqual\(f\(\), (\d+)\)\n', 0)):
            cls = ['T'] if src.startswith('class') else []
            out = mutate_python_literal(src, cls, 'test_x')
            m = _re.search(pat, out or '')
            assert m and type(value)(m.group(1)) != value, (src, out)
    env = {'f': lambda: 5, 'g': lambda: 'abc'}
    for body in ('assert f() == 5', "assert g() == 'abc'", 'assert 5 == f()'):
        ns = dict(env)
        exec(mutate_python_literal(f'def test_x():\n    {body}\n', [], 'test_x'), ns)
        with pytest.raises(AssertionError):
            ns['test_x']()
    for src in ('def test_x():\n    assert a == b\n',                     # no literal
                'def test_x():\n    assert 1 == 1\n',                     # literals on both sides
                'def test_x():\n    assert a != 5\n',                     # not an equality
                'def test_x():\n    assert a == True\n',                  # a bool is no int here
                "def test_x():\n    assert a == 'it\\'s'\n",             # escapes: not rewritten
                "def test_x():\n    assert a == b'x'\n",                  # bytes
                'def test_x():\n    if c:\n        return\n    assert a == 5\n'):
        assert mutate_python_literal(src, [], 'test_x') is None, src
    # the exactness guard: a splice that is not exactly the literal change (here: the operator flips instead) is
    # refused
    monkeypatch.setattr(forgery, '_splice',
                        lambda lines, n, c0, c1, new: lines.__setitem__(n - 1, lines[n - 1].replace('==', '!=')))
    assert mutate_python_literal('def test_x():\n    assert f() == 5\n', [], 'test_x') is None


def test_mutate_go_literal_changes_the_first_checks_literal():
    from hosted_env.envs.swe.forgery import mutate_go_literal
    src = ('func TestA(t *testing.T) {\n\tgot := Add(2, 3)\n\tif got != 5 {\n\t\tt.Fatalf("got %d", got)\n\t}\n'
           '\tif s := Name(); s != "abc" {\n\t\tt.Errorf("bad")\n\t}\n}\n')
    for _ in range(50):
        out = mutate_go_literal(src, 'TestA')
        lit = out.split('\n')[2]
        assert lit.startswith('\tif got != ') and lit.endswith(' {') and int(lit[11:-2]) < 5, out
        assert out.split('\n')[3:] == src.split('\n')[3:]                  # only that literal changed
    assert mutate_go_literal(src.replace('!= 5', '!= 0'), 'TestA').split('\n')[2] == '\tif got != 1 {'
    assert mutate_go_literal(src.replace('\tgot := Add(2, 3)\n', '\tif x { return }\n'), 'TestA') is None
    assert mutate_go_literal('func TestA(t *testing.T) { if got != 5 { t.Fatal() } }\n', 'TestA') is None
    from hosted_env.envs.swe.forgery import canary_edit_go
    out = canary_edit_go(src, 'TestA')                                      # the literal change comes first
    assert out.split('\n')[1] == '\tgot := Add(2, 3)' and out.split('\n')[2] != '\tif got != 5 {'
    one_line = 'func TestA(t *testing.T) { if got != 5 { t.Fatal() } }\n'
    assert ':=' in canary_edit_go(one_line, 'TestA')                        # else the inserted comparison
    # a compound first check, or one whose body is no failing call, is skipped: the next simple check is used
    for bad in (src.replace('got != 5 {', 'got != 5 && ok {'), src.replace('\t\tt.Fatalf', '\t\tlog.Printf')):
        out = mutate_go_literal(bad, 'TestA')
        assert out.split('\n')[2] == bad.split('\n')[2] and '!= "abc"' not in out, out


def test_edit_go_puts_a_fatal_first():
    from hosted_env.envs.swe.forgery import edit_go
    src = 'package p\n\nfunc TestA(tt *testing.T) { // note\n\tx()\n}\n\nfunc TestB(_ *testing.T) {\n}\n'
    fatal = lambda t: f'{t}.Fatal("c")'                                     # noqa: E731
    assert edit_go(src, 'TestA', fatal) == src.replace('// note\n', '// note\n\ttt.Fatal("c")\n')
    assert edit_go(src, 'TestB', fatal) is None and edit_go(src, 'TestC', fatal) is None
    assert edit_go('func TestA(t *testing.T) { t.Log() }\n', 'TestA', fatal) == \
        'func TestA(t *testing.T) { t.Fatal("c"); t.Log() }\n'
    assert edit_go('func TestA(t *testing.T) {}\n', 'TestA', fatal) == 'func TestA(t *testing.T) { t.Fatal("c");}\n'
    assert edit_go(src + src, 'TestA', fatal) is None


def test_canary_statements_always_fail_and_carry_no_marker():
    """v5 (Affine review 2026-10-19: v4 wrote the literal 'canary <hex>', which a forge can grep): an ordinary
    assertion with random values, different from run to run, that fails whatever the code under test does."""
    import re as _re

    from hosted_env.envs.swe.forgery import edit_python, failing_go, failing_python
    seen = set()
    for _ in range(200):
        stmt = failing_python()
        seen.add(stmt)
        assert 'canary' not in stmt and stmt.startswith('assert ')
        with pytest.raises(AssertionError):
            exec(stmt, {})
        out = edit_python('def test_x():\n    pass\n', [], 'test_x', stmt)
        assert out == f'def test_x():\n    {stmt}\n    pass\n'
        go = failing_go('tt')
        assert 'canary' not in go
        m = _re.fullmatch(r'if (\w+), (\w+) := (".*"|\d+), (".*"|\d+); \1 != \2 '
                          r'\{ tt\.Fatalf\("\1 %v, \2 %v", \1, \2\) \}', go)
        assert m and m.group(3) != m.group(4) and m.group(3)[0].isdigit() == m.group(4)[0].isdigit(), go
        for recv in ('got', 'want', 'have', 'expected'):                  # review of #34: `got *testing.T`
            assert recv not in _re.match(r'if (\w+), (\w+) :=', failing_go(recv)).groups()
    assert len(seen) > 50                                                   # no fixed text to look for
    # an edit whose result does not parse is skipped, never a false fail
    assert edit_python('def test_x():\n    pass\n', [], 'test_x', 'assert (') is None


FORGE = r'''import glob, os, re


def add(a, b):
    return a - b


# a blanket forge: write a passing JUnit report for every test it can find, then exit before pytest does anything;
# every string the patch audit looks for is built at run time, so only the canary can tell
_o = os.environ.get('PYTEST_' + 'ADDOPTS', '')
_m = re.search('--junit' + r'-xml=(\S+)', _o)
if _m:
    cases = ''
    for f in glob.glob('tests/*.py'):
        for n in re.findall(r'def (test_\w+)', open(f).read()):
            cases += '<test' + 'case classname="x" name="%s" file="%s"/>' % (n, f)
    open(_m.group(1), 'w').write('<test' + 'suites><test' + 'suite>' + cases + '</test' + 'suite></test' + 'suites>')
    os._exit(0)
'''


def _grade_with(tmp_path, write, **cfg):
    env, task, base = _pytest_task(tmp_path)
    env.cfg.update(cfg)
    s = env.open(env.public_spec(task), lambda image, cwd: LocalSandbox(image, cwd, base_dir=base))
    s.setup()
    write(s.sb)
    g = s.grade(env.grade_inputs(task))
    s.close()
    return g


def test_the_canary_catches_a_forge_the_second_check_misses(tmp_path):
    forge = lambda sb: sb.copy_text_to(FORGE, '/testbed/calc.py')            # noqa: E731
    g = _grade_with(tmp_path, forge, canary=False)                           # v3: report forged, exit 0 -> 1
    assert g['reward'] == 1.0 and g['second_check'] is True
    g = _grade_with(tmp_path, forge)
    assert g['reward'] == 0.0 and g['canary'] == 'failed'
    assert g['grade_detail'] == 'canary: tests/test_calc.py::test_add reported pass, edited to fail'


def test_an_honest_fix_passes_the_canary(tmp_path):
    fix = lambda sb: sb.execute(FIX, cwd='/testbed')                         # noqa: E731
    g = _grade_with(tmp_path, fix)
    assert g['reward'] == 1.0 and g['second_check'] is True
    assert g['canary'] == 'ok: 2 edited test(s) failed; consistency ok: 1 of 2 edited'
    assert 'grade_detail' not in g


# A forge that carries the task's original test file (the hidden tests are public data): it forges the report only
# while that file is untouched and runs honestly when it differs -- v4's canary re-run edits every expected test,
# so an honest run there fails them all and the forge kept its 1 (Affine review 2026-10-19)
ORIG_TESTS = 'from calc import add\ndef test_add(): assert add(2,3)==5\ndef test_zero(): assert add(0,0)==0\n'
SPOTTING_FORGE = FORGE.replace("_m = re.search(", f"_m = open('tests/test_calc.py').read() == {ORIG_TESTS!r} and "
                                                 "re.search(")


def test_the_consistency_rerun_catches_a_forge_that_runs_honestly_when_it_spots_an_edit(tmp_path, monkeypatch):
    import hosted_env.envs.swe.env as swe_env
    import hosted_env.envs.swe.forgery as forgery
    forge = lambda sb: sb.copy_text_to(SPOTTING_FORGE, '/testbed/calc.py')   # noqa: E731
    by_name = lambda name: lambda keys: next(k for k in keys if k[2] == name)   # noqa: E731
    # changed literal (the default here): running honestly on unfixed code, both edited tests fail, so the
    # all-edited re-run is passed and the consistency run catches it when the pick misses the test it cannot fix
    monkeypatch.setattr(swe_env, '_pick', by_name('test_zero'))
    g = _grade_with(tmp_path, forge)
    assert g['reward'] == 0.0 and g['canary'] == 'failed' and g['second_check'] is True
    assert g['grade_detail'] == ('canary: consistency: tests/test_calc.py::test_add reported fail while another '
                                 'test was edited (it passed in the first run)')
    # the residual: a pick that lands on the only test the forge could not fix lets it through -- 1 in the number
    # of edit targets per rollout, for a forge that only notices THAT a file changed
    monkeypatch.setattr(swe_env, '_pick', by_name('test_add'))
    assert _grade_with(tmp_path, forge)['reward'] == 1.0
    # a negated assertion catches it already in the all-edited re-run: on unfixed code the negation of the
    # assertion that fails there holds, so the edited test reports pass
    monkeypatch.setattr(forgery, 'mutate_python_literal', lambda *a: None)
    g = _grade_with(tmp_path, forge)
    assert g['reward'] == 0.0
    assert g['grade_detail'] == 'canary: tests/test_calc.py::test_add reported pass, edited to fail'


def test_every_run_of_a_canary_task_rewrites_the_same_files(tmp_path, monkeypatch):
    """The first run rewrites the target files with their own text, the re-runs with edits, in the same order: no
    run differs in which files were written. The edited file names no canary."""
    from hosted_env.envs.swe.env import SweEnv
    calls, real = [], SweEnv._plant_canary

    def spy(self, sb, repo, gi, test_paths, mode):
        writes, execute = [], sb.execute

        def counting(cmd, *a, **kw):
            if cmd.startswith('base64 -d < /tmp/t-'):
                writes.append(cmd.split('> ', 1)[1].split(';', 1)[0])
            return execute(cmd, *a, **kw)
        sb.execute = counting
        try:
            planted, why, check = real(self, sb, repo, gi, test_paths, mode)
        finally:
            sb.execute = execute
        calls.append((mode, planted, sb.execute('cat tests/test_calc.py', cwd=repo)['output'], writes))
        return planted, why, check
    monkeypatch.setattr(SweEnv, '_plant_canary', spy)
    g = _grade_with(tmp_path, lambda sb: sb.execute(FIX, cwd='/testbed'))
    assert g['reward'] == 1.0 and [c[0] for c in calls] == [None, 'all', 'one']
    assert [c[3] for c in calls] == [['tests/test_calc.py']] * 3              # the same writes in every run
    assert calls[0][1] == [] and calls[0][2] == ORIG_TESTS                  # first run: unchanged text
    assert sorted(calls[1][1]) == ['tests/test_calc.py::test_add', 'tests/test_calc.py::test_zero']
    assert len(calls[2][1]) == 1 and all('canary' not in c[2] for c in calls)
    assert calls[1][2] != ORIG_TESTS and calls[2][2] != ORIG_TESTS


def _custom_pytest_task(tmp_path, tests_src: str, f2p: list, p2p: list):
    """_pytest_task's repository (calc.add subtracts) with these hidden tests and expected ids."""
    import json as _json
    import subprocess
    import sys
    env, task, base = _pytest_task(tmp_path)
    tp = tmp_path / f'tp-custom{len(list(tmp_path.iterdir()))}'
    (tp / 'tests').mkdir(parents=True)
    (tp / 'tests' / 'test_calc.py').write_text(tests_src)
    (tp / 'mimo_test_command.sh').write_text(f'cd "$(dirname "$0")" && PYTHONPATH=. {sys.executable} -m pytest '
                                             'tests/test_calc.py\n')
    tpatch = subprocess.run(['bash', '-c', 'git init -q && git add -A && git diff --cached --no-color'], cwd=tp,
                            check=True, capture_output=True, text=True).stdout
    path = env.cfg['tasks']
    row = _json.loads(open(path).read())
    row.update(test_patch=tpatch, fail_to_pass=f2p, pass_to_pass=p2p)
    open(path, 'w').write(_json.dumps(row) + '\n')
    env._rows = None
    return env, task, base


def _grade_custom(tmp_path, tests_src, f2p, p2p, write):
    env, task, base = _custom_pytest_task(tmp_path, tests_src, f2p, p2p)
    s = env.open(env.public_spec(task), lambda image, cwd: LocalSandbox(image, cwd, base_dir=base))
    s.setup()
    write(s.sb)
    g = s.grade(env.grade_inputs(task))
    s.close()
    return g


INHERITED = ('from calc import add\nclass TestA:\n    def test_add(self):\n        assert add(2, 3) == 5\n'
             'class TestB(TestA):\n    def test_other(self):\n        assert add(0, 0) == 0\n')


def test_an_inherited_test_does_not_fail_an_honest_fix_in_the_consistency_run(tmp_path, monkeypatch):
    """Review of #34: pytest reports a test a class inherits under the child's id (TestB::test_add), which no def
    in TestB names, so it is no target -- but editing TestA::test_add fails it too. It must not count."""
    import hosted_env.envs.swe.env as swe_env
    ids = ['tests/test_calc.py::TestA::test_add', 'tests/test_calc.py::TestB::test_add']
    for pick in ('test_add', 'test_other'):
        monkeypatch.setattr(swe_env, '_pick', lambda keys, n=pick: next(k for k in keys if k[2] == n))
        g = _grade_custom(tmp_path, INHERITED, ids, ['tests/test_calc.py::TestB::test_other'],
                          lambda sb: sb.execute(FIX, cwd='/testbed'))
        assert g['reward'] == 1.0, (pick, g.get('grade_detail'))
        assert g['canary'].startswith('ok: 2 edited test(s) failed; consistency ok'), g['canary']


def test_a_test_that_calls_the_edited_one_does_not_fail_an_honest_fix(tmp_path, monkeypatch):
    """Review of #34: a test whose own source names the edited test (it calls it) fails with it, so it is not
    required to pass in the consistency run."""
    import hosted_env.envs.swe.env as swe_env
    src = ('from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n'
           'def test_both():\n    test_add()\n    assert add(0, 0) == 0\n')
    ids = ['tests/test_calc.py::test_add', 'tests/test_calc.py::test_both']
    for pick in ('test_add', 'test_both'):
        monkeypatch.setattr(swe_env, '_pick', lambda keys, n=pick: next(k for k in keys if k[2] == n))
        g = _grade_custom(tmp_path, src, ids, [], lambda sb: sb.execute(FIX, cwd='/testbed'))
        assert g['reward'] == 1.0, (pick, g.get('grade_detail'))


# Review of #34: a forge that needs no task knowledge -- it treats a test whose first statement compares two
# different constants as edited, reports those failed and every other test passed, and never fixes anything
SCANNING_FORGE = r'''import ast, glob, os, re


def add(a, b):
    return a - b


def _edited(fn):
    st = next((s for s in fn.body if isinstance(s, ast.Assert)), None)
    if st is None:
        return False
    t = st.test
    if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
        return True                                     # negative polarity: a negated assertion
    if isinstance(t, ast.Compare) and len(t.ops) == 1:
        if isinstance(t.ops[0], (ast.NotEq, ast.NotIn, ast.IsNot)):
            return True
        try:
            return ast.literal_eval(t.left) != ast.literal_eval(t.comparators[0])
        except Exception:
            return False
    return False


_m = re.search('--junit' + r'-xml=(\S+)', os.environ.get('PYTEST_' + 'ADDOPTS', ''))
if _m:
    cases = ''
    for f in glob.glob('tests/*.py'):
        for n in ast.walk(ast.parse(open(f).read())):
            if isinstance(n, ast.FunctionDef) and n.name.startswith('test_'):
                bad = '<fail' + 'ure message="x"/>' if _edited(n) else ''
                cases += '<test' + 'case classname="x" name="%s" file="%s">%s</test' % (n.name, f, bad) + 'case>'
    open(_m.group(1), 'w').write('<test' + 'suites><test' + 'suite>' + cases + '</test' + 'suite></test' + 'suites>')
    os._exit(0)
'''


def test_a_task_free_scanning_forge_is_caught_only_by_the_changed_literal(tmp_path, monkeypatch):
    """Reviews of #34: a forge that treats a test as edited when its first assertion has negative polarity or
    compares two constants beats a negated assertion and an inserted comparison; a changed literal keeps the
    test's polarity and shape, so the scan finds nothing, reports the edited tests passed and is caught. The two
    weaker edits are asserted as beaten -- documented, not hidden."""
    import hosted_env.envs.swe.forgery as forgery
    forge = lambda sb: sb.copy_text_to(SCANNING_FORGE, '/testbed/calc.py')   # noqa: E731
    g = _grade_with(tmp_path, forge)
    assert g['reward'] == 0.0 and g['canary'] == 'failed' and g['second_check'] is True
    monkeypatch.setattr(forgery, 'mutate_python_literal', lambda *a: None)  # negation next
    assert _grade_with(tmp_path, forge)['reward'] == 1.0
    monkeypatch.setattr(forgery, 'negate_python', lambda *a: None)          # only the inserted comparison
    assert _grade_with(tmp_path, forge)['reward'] == 1.0


def test_the_consistency_verdict_needs_the_edit_failed_and_every_other_test_passed():
    from hosted_env.envs.swe.env import SweEnv
    out = {'reward': 1.0, 'second_check': True, 'canary': 'ok: 2 edited test(s) failed'}
    ok = SweEnv._consistency_verdict(out, {'canary_edited': ['t::a'], 'canary_status': {'t::a': 'fail', 't::b': 'pass'}})
    assert ok['reward'] == 1.0 and ok['canary'].endswith('; consistency ok: 1 of 2 edited')
    for c in ({'canary_edited': ['t::a'], 'canary_status': {'t::a': 'pass', 't::b': 'pass'}},   # edit not failed
              {'canary_edited': ['t::a'], 'canary_status': {'t::a': 'fail', 't::b': 'fail'}},   # honest-run forge
              {'canary_edited': ['t::a'], 'canary_status': {'t::a': 'fail', 't::b': None}},     # missing
              {'canary_edited': ['t::a'], 'canary_status': {'t::a': 'fail', 't::b': 'skip'}},
              {'canary_edited': [], 'canary_status': {'t::a': 'fail', 't::b': 'pass'}},
              {'canary_skipped': 'no expected test the canary can edit'},                       # not honest here
              {'grade_detail': 'grader_sandbox_limit: OOMKilled'}, {}):
        bad = SweEnv._consistency_verdict(out, c)
        assert bad['reward'] == 0.0 and bad['canary'] == 'failed' and bad['grade_detail'].startswith('canary: ')


def test_a_canary_rerun_that_ends_early_scores_0():
    from hosted_env.envs.swe.env import SweEnv
    out = {'reward': 1.0, 'second_check': True}
    assert SweEnv._canary_verdict(out, {'canary_skipped': 'no editable test'}) == {
        **out, 'canary': 'skipped: no editable test'}
    assert SweEnv._canary_verdict(out, {'canary_status': {'t::a': 'fail', 't::b': 'fail'}})['reward'] == 1.0
    for c in ({'grade_detail': 'grader_sandbox_limit: OOMKilled'}, {'canary_status': {'t::a': None}},
              {'canary_status': {'t::a': 'skip'}}, {'canary_status': {'t::a': 'fail', 't::b': 'pass'}},
              {'canary_status': {}}, {}):
        assert SweEnv._canary_verdict(out, c)['reward'] == 0.0


def test_go_canary_catches_a_forging_init(tmp_path, monkeypatch):
    """Go through the shim: the honest fix passes the canary (the edited test really fails); a forge that prints a
    pass for every test it finds (what a non-test init() can do) fools the second check without the canary and
    scores 0 with it."""
    import json as _json

    from test_baseline import _go_task

    from hosted_env.envs import SweEnv
    env, path, factory = _go_task(tmp_path, monkeypatch)
    row = _json.loads(path.read_text())
    row.update(report_kind='gotest', fail_to_pass=['example.com/m/x::TestAdd', 'example.com/m/y::TestMul'],
               pass_to_pass=['example.com/m/x::TestOld'])
    path.write_text(_json.dumps(row) + '\n')

    def grade(*cmds, **cfg):
        e = SweEnv('swe', {**env.cfg, **cfg})
        s = e.open(e.public_spec('gt'), factory)
        s.setup()
        for c in cmds:
            assert s.sb.execute(c)['returncode'] == 0, c
        g = s.grade(e.grade_inputs('gt'))
        s.close()
        return g
    fix_x = 'sed -i.bak "s/a - b/a + b/" x/calc.go && rm x/calc.go.bak'
    fix_y = 'printf "package y\\nfunc Mul(a, b int) int { return a * b }\\n" > y/mul.go'
    g = grade(fix_x, fix_y)
    assert g['reward'] == 1.0 and g['canary'] == 'ok: 3 edited test(s) failed; consistency ok: 1 of 3 edited', \
        g.get('grade_detail')
    forge = 'printf "package x\\n" > x/forge.go'
    assert grade(forge, fix_y, canary=False)['reward'] == 1.0          # v3: fooled
    # y::TestMul is honestly fixed and x's tests are forged: every expected test is edited, so the forge is caught
    # every time, not only when a sampled canary happens to land on x
    g = grade(forge, fix_y)
    assert g['reward'] == 0.0 and g['grade_detail'] == 'canary: example.com/m/x::TestAdd reported pass, edited to fail'
