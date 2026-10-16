"""toolcall.py, envs/: ported from EXP-014 test_harness.py. No docker needed: LocalSandbox runs in a temp
directory on the host."""
import os
import shutil

import pytest
from conftest import make_swe_tasks

from hosted_env.envs import SweEnv
from hosted_env.envs.base import BASH_TOOL, truncate_middle
from hosted_env.envs.swe import future_commits, patch_touched_files, protected_paths, reset_test_files_cmd
from hosted_env.providers.sandbox import LocalSandbox
from hosted_env.toolcall import parse_tool_calls, split_reasoning



def test_parse_single_call_qwen_style():
    c = ('I will list files.\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la\ncat a.py\n</parameter>\n'
         '<parameter=timeout>\n120\n</parameter>\n</function>\n</tool_call>')
    text, calls, err = parse_tool_calls(c)
    assert err is None and text == 'I will list files.'
    assert calls == [{'name': 'bash', 'arguments': {'command': 'ls -la\ncat a.py', 'timeout': '120'}}]


def test_parse_mimo_style_multi_and_whitespace():
    c = ('<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>'
         '<tool_call><function=bash><parameter=command>pwd</parameter></function></tool_call>')
    _, calls, err = parse_tool_calls(c)
    assert err is None and [x['arguments']['command'] for x in calls] == ['ls', 'pwd']
    cmd = "cat > f.py <<'EOF'\n    x = '<parameter=fake>'\n\nEOF"
    _, calls, _ = parse_tool_calls(f'<tool_call>\n<function=bash>\n<parameter=command>\n{cmd}\n</parameter>\n</function>\n</tool_call>')
    assert calls[0]['arguments']['command'] == cmd


def test_parse_errors_and_submit():
    _, calls, err = parse_tool_calls('<tool_call>\n<function=bash>\n<parameter=command>\nls\n</function>')
    assert calls == [] and err
    text, calls, err = parse_tool_calls('All done, the fix is in place.')
    assert calls == [] and err is None and text.startswith('All done')


def test_split_reasoning_variants():
    assert split_reasoning('plan\n</think>\n\nanswer', True) == ('plan', 'answer', True)
    assert split_reasoning('<think>plan</think>answer', False) == ('plan', 'answer', True)
    assert split_reasoning('still thinking', True) == ('still thinking', '', False)
    assert split_reasoning('just answer', False) == (None, 'just answer', True)


def test_truncate_middle():
    t = truncate_middle('a' * 30000 + 'b' * 30000)
    assert t.startswith('[TRUNCATION WARNING]') and 'OUTPUT OMITTED: 20000 characters' in t
    assert t.endswith('b' * 20000) and ('a' * 20000) in t
    assert truncate_middle('short') == 'short'


def test_tool_schema_matches_official():
    f = BASH_TOOL['function']
    assert f['name'] == 'bash' and f['parameters']['required'] == ['command']
    assert 'default 60, max 300' in f['description']


def test_reset_cmd_covers_added_and_renamed_files():
    patch = ('diff --git a/tests/test_x.py b/tests/test_x.py\nnew file mode 100644\n'
             'diff --git a/old.sh b/mimo_test_command.sh\nrename from old.sh\n')
    assert patch_touched_files(patch) == ['tests/test_x.py', 'old.sh', 'mimo_test_command.sh']
    cmd = reset_test_files_cmd(patch, 'f' * 40)
    assert 'git cat-file -e ' + 'f' * 40 in cmd and 'mimo_test_command.sh' in cmd
    assert reset_test_files_cmd('', 'f' * 40) is None


# ---------------------------------------------------------------- end to end on a local git repo

def make_task(tmp_path):
    path, ids = make_swe_tasks(tmp_path)
    return SweEnv('swe', {'kind': 'mimo_swe', 'tasks': path, 'cleanup': False}), ids[0]


def open_local(env, task, tmp_path):
    return env.open(env.public_spec(task), lambda image, cwd: LocalSandbox(image, cwd, base_dir=str(tmp_path)))


def test_setup_strips_and_grades(tmp_path):
    env, task = make_task(tmp_path)
    s = open_local(env, task, tmp_path)
    s.sb.start()
    head = s.sb.execute('git rev-parse HEAD')['output'].strip()
    assert future_commits(s.sb, '/testbed', head)                      # leak is present
    env.setup(s)
    assert s.state['git_stripped'] and future_commits(s.sb, '/testbed', s.state['base']) == []
    assert 'a + b' not in s.sb.execute('git log --all -p')['output']   # the fix commit is physically deleted
    rollout_root = s.sb.root
    assert s.grade(env.grade_inputs(task))['reward'] == 0.0           # base state grades 0
    assert not os.path.exists(rollout_root) and not os.path.exists(s.grader_sb.root)   # both containers gone
    s.close()


def graded(tmp_path, *cmds, in_place_check=False, image_prep=None):
    """One rollout: the model runs `cmds` in its container, then the isolated grader runs. With in_place_check,
    also run the hidden tests inside the model's own container first (what the old grader did) and return that
    exit code too. image_prep runs once in the image template (to build images with unusual states)."""
    tmp_path = tmp_path / f'g{len(list(tmp_path.iterdir()))}'          # each call builds its own task
    tmp_path.mkdir()
    env, task = make_task(tmp_path)
    if image_prep:
        import subprocess
        subprocess.run(['bash', '-c', image_prep], cwd=env.public_spec(task)['image'], check=True)
    s = open_local(env, task, tmp_path)
    s.setup()
    for c in cmds:
        assert s.sb.execute(c)['returncode'] == 0, c
    gi = env.grade_inputs(task)
    in_place = None
    if in_place_check:
        probe = open_local(env, task, tmp_path)                        # a copy of the model's container state
        shutil.rmtree(probe.sb.root)
        shutil.copytree(s.sb.root, probe.sb.root, symlinks=True)
        os.makedirs(probe.sb.env['HOME'], exist_ok=True)
        probe.sb.copy_text_to(gi['test_patch'], '/tmp/tp.patch')
        assert probe.sb.execute('git apply /tmp/tp.patch')['returncode'] == 0
        in_place = probe.sb.execute(gi['test_command'])['returncode']
        probe.close()
    g = s.grade(gi)
    s.close()
    return g, in_place


def test_honest_fix_scores_and_is_kept(tmp_path):
    g, _ = graded(tmp_path, 'sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak',
                  'printf "notes\\n" > NOTES.md', 'mkdir -p pkg && printf "x = 1\\n" > pkg/helper.py')
    assert g['reward'] == 1.0 and g['grader'] == 'isolated-v1' and g['reverted_paths'] == []
    assert 'a + b' in g['model_patch'] and 'pkg/helper.py' in g['model_patch']


def test_planted_grading_script_is_reset(tmp_path):
    g, _ = graded(tmp_path, 'printf "exit 0\\n" > mimo_test_command.sh')
    assert g['reward'] == 0.0


def test_sitecustomize_exploit_works_in_place_but_not_isolated(tmp_path):
    hack = 'printf "import os\\nos._exit(0)\\n" > sitecustomize.py'    # python imports it from PYTHONPATH at start-up
    g, in_place = graded(tmp_path, hack, in_place_check=True)
    assert in_place == 0                                               # the old in-container grading is fooled
    assert g['reward'] == 0.0 and g['reverted_paths'] == ['sitecustomize.py']
    g, _ = graded(tmp_path, hack, 'git add -f sitecustomize.py && printf "sitecustomize.py\\n" > .gitignore')
    assert g['reward'] == 0.0
    assert 'diff --git a/sitecustomize.py' not in g['model_patch']    # force-added but ignored: never carried over


def test_home_profile_and_path_shims_never_reach_the_grader(tmp_path):
    shim = ('mkdir -p "$HOME/bin" && printf "#!/bin/sh\\nexit 0\\n" > "$HOME/bin/python3" && '
            'chmod +x "$HOME/bin/python3" && printf "export PATH=$HOME/bin:$PATH\\n" | tee "$HOME/.profile" '
            '"$HOME/.bashrc" > /dev/null && printf "python3() { return 0; }\\n" >> "$HOME/.bashrc"')
    g, _ = graded(tmp_path, shim)
    assert g['reward'] == 0.0 and g['model_patch'] == ''               # outside the repo: not part of the patch


def test_protected_test_infrastructure_is_reset_whatever_its_name(tmp_path):
    files = ['conftest.py', 'pytest.ini', 'sub dir/conftest.py', 'pkg/zz_test.go', 'web/__mocks__/m.js',
             'web/a.spec.ts', 'package.json', 'Makefile', 'tests/test_extra.py', 'x.pth']
    cmds = [f'mkdir -p "$(dirname "{f}")" && printf "exit 0\\n" > "{f}"' for f in files]
    g, _ = graded(tmp_path, 'sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak', *cmds)
    assert g['reward'] == 1.0                                          # the fix itself is kept
    assert sorted(g['reverted_paths']) == sorted(files)


def test_patch_problems_are_the_models_own_zero(tmp_path):
    g, _ = graded(tmp_path, 'head -c 2500000 /dev/urandom | base64 > big.txt')
    assert (g['reward'], g['grade_detail'], g['model_patch']) == (0.0, 'model_patch_too_large', '')
    g, _ = graded(tmp_path, 'rm -rf .git')
    assert (g['reward'], g['grade_detail']) == (0.0, 'model_patch_capture_failed')
    g, _ = graded(tmp_path, 'sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak',
                  'head -c 3000 /dev/urandom > blob.bin')
    assert g['reward'] == 1.0 and 'GIT binary patch' in g['model_patch']   # binary files carry over


def test_protected_paths_rules():
    assert protected_paths(['src/app.py', 'src/conftest.py', 'lib/util.go', 'lib/util_test.go', 'README.md',
                            'tests/helpers.py', 'tests/test_a.py', 'a/b.test.tsx', 'jest.config.js', '.env.test',
                            'src/setupTests.ts', 'go.mod']) == [
        'src/conftest.py', 'lib/util_test.go', 'tests/test_a.py', 'a/b.test.tsx', 'jest.config.js', '.env.test',
        'src/setupTests.ts', 'go.mod']


def test_image_whose_tree_differs_from_its_head(tmp_path):
    """Real images (e.g. 001488) ship a working tree that differs from their own HEAD: deleted tracked files,
    edits, extra files. That state is the task, not the model's work: a no-op rollout yields an empty patch, and
    a fix on top of it carries over exactly."""
    dirty = ('git rm -q --cached calc.py >/dev/null; printf "def add(a, b):\\n    return a - b\\n" > calc.py; '
             'printf "keep\\n" > notes.txt; printf "x\\n" > gone.txt && git add gone.txt && '
             'git -c user.email=t@t -c user.name=t commit -qm gone && rm gone.txt')
    g, _ = graded(tmp_path, image_prep=dirty)
    assert (g['reward'], g['model_patch'], g.get('grade_detail')) == (0.0, '', None)
    g, _ = graded(tmp_path, 'sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak', image_prep=dirty)
    assert g['reward'] == 1.0


def test_patch_that_cannot_apply_is_the_models_zero(tmp_path, monkeypatch):
    from hosted_env.envs import swe
    real = swe.SweEnv.capture_patch
    monkeypatch.setattr(swe.SweEnv, 'capture_patch',
                        lambda self, s: (real(self, s)[0] + b'diff --git a/nope b/nope\n--- a/nope\n+++ b/nope\n'
                                         b'@@ -1 +1 @@\n-x\n+y\n', ''))
    g, _ = graded(tmp_path, 'sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak')
    assert g['reward'] == 0.0 and g['grade_detail'].startswith('model_patch_does_not_apply')


def test_public_spec_hides_grading(tmp_path):
    env, task = make_task(tmp_path)
    spec = env.public_spec(task)
    assert 'test_patch' not in spec and 'test_command' not in spec
    assert set(env.grade_inputs(task)) == {'test_command', 'test_patch', 'verifier_timeout'}


def test_call_tool_formats(tmp_path):
    env, task = make_task(tmp_path)
    s = open_local(env, task, tmp_path)
    s.setup()
    out = s.call_tool('bash', {'command': 'cat calc.py'})
    assert 'return a - b' in out and out.endswith("Tool metadata: {'returncode': 0}")
    out = s.call_tool('bash', {'command': 'sleep 5', 'timeout': '1'})
    assert 'Command timed out after 1s' in out and "'reason': 'client_timeout'" in out
    assert "Unknown tool 'python'" in s.call_tool('python', {'code': '1'})
    assert s.call_tool('bash', {}) == "Bash tool requires a non-empty 'command' parameter"
    s.close()


@pytest.mark.parametrize('leak', ['error'])
def test_git_leak_error_mode(tmp_path, leak):
    env, task = make_task(tmp_path)
    env.cfg['git_leak'] = leak
    s = open_local(env, task, tmp_path)
    from hosted_env.envs import SetupError
    with pytest.raises(SetupError):
        s.setup()
    s.close()


def test_local_sandbox_map_root_under_tmp(tmp_path):
    """Regression (Linux CI): the sandbox root itself lives under /tmp/, so the mapping must not re-map it."""
    base = tmp_path / 'tmp' / 'x'
    base.mkdir(parents=True)
    sb = LocalSandbox('', '/w', base_dir=str(base))
    real_root = sb.root
    sb.root = '/tmp/pytest-0/hosted-env-local-abc'          # what tempfile returns on a Linux runner
    home = sb.root + '-home/'
    assert sb._map('/w/a.py') == sb.root + '/a.py'
    assert sb._map('cat /tmp/x && cd /w') == f'cat {home}x && cd {sb.root}'
    assert sb._map('"/tmp/y" :/tmp/z') == f'"{home}y" :{home}z'
    assert sb._map('/var/tmp/k $HOME/tmp/q') == '/var/tmp/k $HOME/tmp/q'
    sb.root = real_root
    sb.cleanup()
