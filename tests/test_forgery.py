"""Forgery checks of grader isolated-v4 (envs/swe/forgery.py): the patch audit."""
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


def test_edit_go_puts_a_fatal_first():
    from hosted_env.envs.swe.forgery import edit_go
    src = 'package p\n\nfunc TestA(tt *testing.T) { // note\n\tx()\n}\n\nfunc TestB(_ *testing.T) {\n}\n'
    assert edit_go(src, 'TestA', 'canary 1') == src.replace('// note\n', '// note\n\ttt.Fatal("canary 1")\n')
    assert edit_go(src, 'TestB', 'c') is None and edit_go(src, 'TestC', 'c') is None
    assert edit_go('func TestA(t *testing.T) { t.Log() }\n', 'TestA', 'c') == \
        'func TestA(t *testing.T) { t.Fatal("c"); t.Log() }\n'
    assert edit_go('func TestA(t *testing.T) {}\n', 'TestA', 'c') == 'func TestA(t *testing.T) { t.Fatal("c");}\n'
    assert edit_go(src + src, 'TestA', 'c') is None


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
    assert g['reward'] == 1.0 and g['second_check'] is True and g['canary'] == 'ok: 2 edited test(s) failed'
    assert 'grade_detail' not in g


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
    assert g['reward'] == 1.0 and g['canary'] == 'ok: 3 edited test(s) failed', g.get('grade_detail')
    forge = 'printf "package x\\n" > x/forge.go'
    assert grade(forge, fix_y, canary=False)['reward'] == 1.0          # v3: fooled
    # y::TestMul is honestly fixed and x's tests are forged: every expected test is edited, so the forge is caught
    # every time, not only when a sampled canary happens to land on x
    g = grade(forge, fix_y)
    assert g['reward'] == 0.0 and g['grade_detail'] == 'canary: example.com/m/x::TestAdd reported pass, edited to fail'
