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
