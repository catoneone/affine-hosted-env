"""toolcall.py, envs/: ported from EXP-014 test_harness.py. No docker needed: LocalSandbox runs in a temp
directory on the host."""
import os
import shutil

import pytest
from conftest import make_swe_tasks

from hosted_env.envs import SweEnv
from hosted_env.envs.base import BASH_TOOL, GradeError, truncate_middle
from hosted_env.envs.swe import TEST_PATCH_REMOTE, future_commits, protected_paths
from hosted_env.providers.sandbox import LocalSandbox, SandboxLimitError
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


# ---------------------------------------------------------------- end to end on a local git repo

def make_task(tmp_path):
    path, ids = make_swe_tasks(tmp_path)
    return SweEnv('swe', {'kind': 'mimo_swe', 'tasks': path, 'cleanup': False, 'grader_user': 'root',
                          'require_image_digest': False}), ids[0]


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
    assert g['reward'] == 1.0 and g['grader'] == 'isolated-v2' and g['reverted_paths'] == []
    assert g['grader_user'] == 'root' and 'lock_seconds' not in g
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


def test_protected_file_renamed_away_is_put_back(tmp_path):
    # git diff records a 100% rename (conftest.py -> notes.txt); git apply --numstat would name only notes.txt
    prep = 'printf "import os\\n# fixtures the hidden tests rely on\\n" > conftest.py'
    g, _ = graded(tmp_path, 'mv conftest.py notes.txt', 'sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak',
                  image_prep=prep)
    assert 'rename from conftest.py' in g['model_patch']
    assert g['reward'] == 1.0 and g['reverted_paths'] == ['conftest.py']


def test_crafted_patch_on_an_ignored_file_scores_0_not_infra_error(tmp_path):
    """A hunk on a git-ignored file that exists on disk applies in the worktree but not to the start tree. A real
    tree-to-tree diff cannot contain it, so the model's own git produced it: graded 0, never a voided rollout."""
    import subprocess
    env, task = make_task(tmp_path)
    subprocess.run(['bash', '-c', 'printf ".env\\n" > .gitignore && printf "SECRET=1\\n" > .env'],
                   cwd=env.public_spec(task)['image'], check=True)
    s = open_local(env, task, tmp_path)
    s.setup()
    crafted = (b'diff --git a/.env b/.env\n--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-SECRET=1\n+SECRET=2\n')
    env.capture_patch = lambda _s: (crafted, '')                      # what a replaced git binary could send
    g = s.grade(env.grade_inputs(task))
    s.close()
    assert g['reward'] == 0.0 and g['grade_detail'].startswith('model_patch_does_not_apply: ')
    assert '.env' in g['grade_detail']


def test_patch_problems_are_the_models_own_zero(tmp_path):
    g, _ = graded(tmp_path, 'head -c 2500000 /dev/urandom | base64 > big.txt')
    assert (g['reward'], g['grade_detail'], g['model_patch']) == (0.0, 'model_patch_too_large', '')
    g, _ = graded(tmp_path, 'rm -rf .git')
    assert (g['reward'], g['grade_detail']) == (0.0, 'model_patch_capture_failed')
    g, _ = graded(tmp_path, 'sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak',
                  'head -c 3000 /dev/urandom > blob.bin')
    assert g['reward'] == 1.0 and 'GIT binary patch' in g['model_patch']   # binary files carry over


class LimitedSandbox(LocalSandbox):
    """LocalSandbox that reports a limit kill (as K8sSandbox does) on one command, or on every command once dead."""
    def __init__(self, *a, fail_on=None, dead=False, **kw):
        super().__init__(*a, **kw)
        self.fail_on, self.dead = fail_on, dead

    def execute(self, command, *a, **kw):
        if self.dead or command == self.fail_on:
            raise SandboxLimitError('ephemeral-storage limit: Pod ephemeral local storage usage exceeds the total '
                                    'limit of containers 10Gi.')
        return super().execute(command, *a, **kw)


def open_limited(env, task, tmp_path, **kw):
    return env.open(env.public_spec(task), lambda image, cwd: LimitedSandbox(image, cwd, base_dir=str(tmp_path), **kw))


def test_rollout_container_killed_for_its_limit_scores_0(tmp_path):
    env, task = make_task(tmp_path)
    s = open_limited(env, task, tmp_path)
    s.setup()
    s.sb.dead = True                                                   # its commands got it evicted
    g = s.grade(env.grade_inputs(task))
    s.close()
    assert (g['reward'], g['model_patch']) == (0.0, '')
    assert g['grade_detail'].startswith('rollout_sandbox_limit: ephemeral-storage limit')


def test_tests_killed_for_the_grading_containers_limit_score_0(tmp_path):
    env, task = make_task(tmp_path)
    gi = env.grade_inputs(task)
    s = open_limited(env, task, tmp_path, fail_on=gi['test_command'])   # only the hidden tests trip the limit
    s.setup()
    assert s.sb.execute('sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak')['returncode'] == 0
    g = s.grade(gi)
    s.close()
    assert g['reward'] == 0.0 and g['grade_detail'].startswith('grader_sandbox_limit: ephemeral-storage limit')
    assert 'a + b' in g['model_patch'] and g.get('verifier_returncode') is None


class LimitAfter(LimitedSandbox):
    """Reports a limit kill on the first command containing `marker` (or the upload to `copy_kill`), and on
    everything after it."""
    def __init__(self, *a, marker=None, copy_kill=None, **kw):
        super().__init__(*a, **kw)
        self.marker, self.copy_kill = marker, copy_kill

    def execute(self, command, *a, **kw):
        if self.marker and self.marker in command:
            self.dead = True
        return super().execute(command, *a, **kw)

    def copy_text_to(self, text, remote_path):
        if self.dead or remote_path == self.copy_kill:
            self.dead = True
            raise SandboxLimitError('memory limit: container OOMKilled')
        return super().copy_text_to(text, remote_path)


def grade_with_grader(tmp_path, fix=True, **grader_kw):
    """One rollout in a plain sandbox, graded in a LimitAfter(**grader_kw) container."""
    env, task = make_task(tmp_path)
    gi, made = env.grade_inputs(task), []

    def factory(image, cwd):
        made.append(LimitAfter(image, cwd, base_dir=str(tmp_path), **(grader_kw if made else {})))
        return made[-1]
    s = env.open(env.public_spec(task), factory)
    s.setup()
    if fix:
        assert s.sb.execute('sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak')['returncode'] == 0
    try:
        return s.grade(gi)
    finally:
        s.close()


@pytest.mark.parametrize('kw', [{'marker': 'git apply --cached'}, {'marker': 'git apply --binary'},
                                {'copy_kill': TEST_PATCH_REMOTE}])
def test_limit_kill_while_the_grader_handles_the_model_patch_scores_0(tmp_path, kw):
    """A few-KB git binary delta can expand to GBs inside `git apply` (review of PR #15): from the moment model bytes
    are in the grading container, a limit kill is the model's, even before the hidden tests run."""
    g = grade_with_grader(tmp_path, **kw)
    assert g['reward'] == 0.0 and g['grade_detail'].startswith('grader_sandbox_limit: ')
    assert 'a + b' in g['model_patch']


def test_limit_kill_before_any_model_bytes_is_an_infra_error(tmp_path):
    with pytest.raises(GradeError, match='during setup'):            # empty model patch: nothing of the model's ran
        grade_with_grader(tmp_path, fix=False, copy_kill=TEST_PATCH_REMOTE)


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


def test_as_grader_drops_root_and_quotes_the_command():
    import shlex

    from hosted_env.envs.swe import GRADER_UID, as_grader
    cmd = as_grader("bash /testbed/mimo_test_command.sh; echo 'it''s' \"$HOME\"")
    argv = shlex.split(cmd)
    assert argv[:6] == ['setpriv', f'--reuid={GRADER_UID}', f'--regid={GRADER_UID}', '--clear-groups',
                        '--no-new-privs', '--']
    assert argv[6:9] == ['env', 'USER=nobody', 'LOGNAME=nobody']                    # HOME stays the image's
    assert argv[9:12] == ['bash', '--noprofile', '--norc'] and argv[12] == '-c'
    assert argv[13] == "bash /testbed/mimo_test_command.sh; echo 'it''s' \"$HOME\""   # passed through intact
    assert SweEnv('swe', {}).cfg.get('grader_user', 'nobody') == 'nobody'          # non-root is the default


class _StubRootSandbox:
    """Runs _lock_down's script with bash on the host, with chown, setpriv and id stubbed (the test is not root):
    chown only logs, setpriv drops nothing but makes the stub id report 65534, chmod logs and really runs unless
    `fake_chmod` is set. The /tmp paths the env uses are mapped under `tmp`."""

    def __init__(self, tmp, home, setpriv_rc=None, fake_chmod=False, no_setpriv=False):
        import os
        self.tmp, self.log, stub = str(tmp), tmp / 'calls.log', tmp / 'bin'
        stub.mkdir()
        scripts = {
            'chown': f'echo chown "$@" >> {self.log}',
            'chmod': f'echo chmod "$@" >> {self.log}' + ('' if fake_chmod else         # BSD chmod: no `--`
                                                          '\nm=$1; shift; [ "$1" = -- ] && shift\nexec /bin/chmod "$m" "$@"'),
            'setpriv': (f'exit {setpriv_rc}' if setpriv_rc is not None else
                        'while [ "$1" != -- ]; do shift; done; shift\nFAKE_UID=65534 exec "$@"'),
            'id': 'echo "${FAKE_UID:-0}"',
        }
        if no_setpriv:                     # a real setpriv may be on the host (Linux CI): keep only what is needed
            del scripts['setpriv']
            for tool in ('base64', 'rm', 'dirname'):
                os.symlink(shutil.which(tool), stub / tool)
        for name, body in scripts.items():
            (stub / name).write_text('#!/bin/sh\n' + body + '\n')
            (stub / name).chmod(0o755)
        self.env = {'PATH': str(stub) if no_setpriv else f'{stub}:/usr/bin:/bin', 'HOME': str(home)}
        self.log.write_text('')
        os.makedirs(home, exist_ok=True)

    def _map(self, s):
        return s.replace('/tmp/_hosted_env', self.tmp + '/_hosted_env')

    def copy_text_to(self, text, path):
        with open(self._map(path), 'w') as f:
            f.write(text)

    def execute(self, command, cwd=None, timeout=None, raw=False, login=True):
        import subprocess
        p = subprocess.run([shutil.which('bash'), '-c', self._map(command)], cwd=cwd, env=self.env, capture_output=True,
                           text=True, timeout=60)
        return {'returncode': p.returncode, 'output': p.stdout + p.stderr, 'reason': ''}

    def calls(self):
        return self.log.read_text().splitlines()


def _lock_repo(tmp_path):
    import os
    repo = tmp_path / 'work' / 'repo'
    (repo / 'tests').mkdir(parents=True)
    (repo / 'src').mkdir()
    (repo / 'tests' / 'test_a.py').write_text('def test_a(): pass\n')
    (repo / 'mimo_test_command.sh').write_text('pytest tests\n')
    (repo / 'src' / 'x.py').write_text('x = 1\n')
    return os.path.realpath(repo)


def test_lock_down_locks_the_test_files_and_proves_the_drop(tmp_path):
    import os
    repo = _lock_repo(tmp_path)
    sb = _StubRootSandbox(tmp_path, tmp_path / 'home')
    paths = [b'tests/test_a.py', b'mimo_test_command.sh', b'tests/deleted_by_the_patch.py']
    took = SweEnv('swe', {})._lock_down(sb, repo, paths)
    calls = sb.calls()
    assert isinstance(took, float)
    assert calls[0] == 'chown -R 65534:65534 .'                                     # repo, then HOME, to nobody
    assert calls[1] == f'chown -R 65534:65534 {os.path.realpath(tmp_path / "home")}'
    for c in ('chown -h 0:0 -- tests', 'chmod 1777 -- tests', 'chown -h 0:0 -- tests/test_a.py',
              'chmod a-w -- tests/test_a.py', 'chmod a-w -- mimo_test_command.sh'):
        assert c in calls
    assert calls[-2:] == ['chown 0:0 .', 'chmod 1777 .']
    assert not os.access(os.path.join(repo, 'tests', 'test_a.py'), os.W_OK)
    assert os.access(os.path.join(repo, 'src', 'x.py'), os.W_OK)                     # the rest stays writable
    assert not any('src' in c for c in calls[2:])
    assert not os.path.exists(tmp_path / '_hosted_env_lock.list')


def test_lock_down_keeps_the_repo_ancestors_inside_home_root_owned(tmp_path):
    import os
    repo = _lock_repo(tmp_path)
    sb = _StubRootSandbox(tmp_path, tmp_path)                                         # HOME is above the repo
    SweEnv('swe', {})._lock_down(sb, repo, [b'tests/test_a.py'])
    calls, work, home = sb.calls(), os.path.dirname(repo), os.path.realpath(tmp_path)
    for d in (work, home):                  # else the grader could rename the repo away and put a fake one there
        assert f'chown -h 0:0 -- {d}' in calls and f'chmod 1777 -- {d}' in calls
    assert f'chown -h 0:0 -- {os.path.dirname(home)}' not in calls


def test_lock_down_refuses_unsafe_paths_and_symlinked_dirs(tmp_path):
    import os
    repo = _lock_repo(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    os.symlink(outside, os.path.join(repo, 'linked'))
    env, sb = SweEnv('swe', {}), _StubRootSandbox(tmp_path, tmp_path / 'home')
    (outside / 'sub').mkdir()
    for bad, why in ((b'linked/test_b.py', 'symlink in a test path: linked'),
                     (b'linked/sub/test_b.py', 'symlink in a test path: linked'),     # deeper dirs are not touched
                     (b'../escape.py', 'unsafe test path'), (b'/etc/passwd', 'unsafe test path'),
                     (b'tests/../../escape.py', 'unsafe test path')):
        with pytest.raises(GradeError, match=why):
            env._lock_down(sb, repo, [b'tests/test_a.py', bad])
    assert not any(str(outside) in c or 'linked' in c for c in sb.calls())
    assert (outside / 'sub').stat().st_mode & 0o7777 != 0o1777


def test_lock_down_fails_loudly_when_the_drop_does_not_work(tmp_path):
    for case, kw, why in (('a', {'setpriv_rc': 127}, 'cannot drop privileges to 65534'),   # e.g. no CAP_SETUID
                          ('c', {'no_setpriv': True}, 'setpriv not found'),
                          ('b', {'fake_chmod': True}, 'test file still writable after the drop: tests/test_a.py')):
        (tmp_path / case).mkdir()
        repo = _lock_repo(tmp_path / case)
        with pytest.raises(GradeError, match=why):
            SweEnv('swe', {})._lock_down(_StubRootSandbox(tmp_path / case, tmp_path / case / 'home', **kw),
                                         repo, [b'tests/test_a.py'])


def test_grader_user_must_be_root_or_nobody(tmp_path):
    env, task = make_task(tmp_path)
    s = open_local(env, task, tmp_path)
    s.setup()
    env.cfg['grader_user'] = 'nobdy'
    with pytest.raises(GradeError, match='grader_user must be "root" or "nobody"'):
        s.grade(env.grade_inputs(task))
    s.close()


@pytest.mark.parametrize('shape', [
    'mkdir mimo_test_command.sh && printf "exit 0\\n" > mimo_test_command.sh/x',   # a dir at the grading script
    'mkdir -p tests/test_calc.py && printf "x\\n" > tests/test_calc.py/x',          # a dir at a hidden test
    'printf "x\\n" > tests',                                                     # a file where tests/ goes
    'mkdir elsewhere && ln -s elsewhere tests',                                   # a symlink where tests/ goes
])
def test_model_cannot_block_the_test_patch(tmp_path, shape):
    """Each of these used to make the trusted test patch fail to apply (reset_tests_failed or
    apply_test_patch_failed), an infra error the model could trigger at will to void a rollout. The task's
    test_command and its scripts have fixed, known names (every mimo task runs `bash mimo_test_command.sh`)."""
    g, _ = graded(tmp_path, shape, 'sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak')
    assert g['reward'] == 1.0


def _rich_task(tmp_path):
    """A base with existing tests, a `fixtures` file and a `docs` directory, and a test patch that modifies,
    renames and deletes test files, turns `fixtures` into a directory and `docs` into a symlink."""
    import json

    from conftest import _run
    tpl = tmp_path / 'tpl'
    (tpl / 'tests').mkdir(parents=True)
    (tpl / 'calc.py').write_text('def add(a, b):\n    return a - b\n')
    (tpl / 'fixtures').write_text('x = 1\n')
    (tpl / 'tests' / 'helper.py').write_text('OLD = 1\n')
    (tpl / 'tests' / 'obsolete.py').write_text('assert True\n')
    (tpl / 'tests' / 'test_old.py').write_text('print(0)\n')
    (tpl / 'docs').mkdir()
    (tpl / 'docs' / 'x.txt').write_text('x\n')
    _run('git init -q -b master && git add -A && git commit -qm base', tpl)
    tp = tmp_path / 'tp'
    shutil.copytree(tpl, tp, symlinks=True)
    _run('git rm -q fixtures tests/obsolete.py && git rm -rq docs && git mv tests/helper.py tests/helpers.py', tp)
    os.symlink('tests', tp / 'docs')
    (tp / 'fixtures').mkdir()
    (tp / 'fixtures' / 'data.txt').write_text('5\n')
    (tp / 'tests' / 'test_old.py').write_text(
        'import sys\nsys.path[:0] = [".", "tests"]\nfrom calc import add\nfrom helpers import OLD\n'
        'assert add(2, 3) == int(open("fixtures/data.txt").read()) and OLD == 1\nprint("ok")\n')
    (tp / 'mimo_test_command.sh').write_text('cd "$(dirname "$0")" && python3 tests/test_old.py\n')
    test_patch = _run('git add -A && git diff --cached --no-color -M', tp).stdout
    assert 'rename from tests/helper.py' in test_patch and 'deleted file mode' in test_patch
    path = tmp_path / 'tasks.jsonl'
    path.write_text(json.dumps({'id': 'rich', 'image': str(tpl), 'cwd': '/testbed', 'lang': 'python',
                                'problem_statement': 'add() subtracts', 'test_patch': test_patch,
                                'test_command': 'bash /testbed/mimo_test_command.sh'}) + '\n')
    return SweEnv('swe', {'kind': 'mimo_swe', 'tasks': str(path), 'cleanup': False, 'grader_user': 'root',
                          'require_image_digest': False}), 'rich'


@pytest.mark.parametrize('shape', [
    '',                                                                     # honest: file -> dir in the test patch
    'rm -rf tests && ln -s {canary} tests',                                 # tests/ points outside the repo
    'rm fixtures && mkdir fixtures && printf "0\\n" > fixtures/data.txt',  # pre-empts the file -> dir change
    'mkdir -p tests/helpers.py && printf "x\\n" > tests/helpers.py/x',     # a dir at the rename target
])
def test_test_patch_that_modifies_renames_deletes_and_retypes(tmp_path, shape):
    canary = tmp_path / 'canary'
    canary.mkdir()
    (canary / 'keep').write_text('keep\n')
    env, task = _rich_task(tmp_path)
    s = open_local(env, task, tmp_path)
    s.setup()
    for c in (shape.format(canary=canary), 'sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak'):
        if c:
            assert s.sb.execute(c)['returncode'] == 0, c
    g = s.grade(env.grade_inputs(task))
    s.close()
    assert g['reward'] == 1.0, g.get('test_output') or g.get('grade_detail')
    assert sorted(os.listdir(canary)) == ['keep']                           # nothing outside the repo touched


def test_lock_list_is_what_the_test_patch_leaves_behind(tmp_path):
    """Not the paths it deletes: with docs/x.txt in the list, the symlink check refused docs (a dir the test
    patch turns into a symlink) and every honest rollout of such a task was an infra error."""
    env, task = _rich_task(tmp_path)
    env.cfg['grader_user'] = 'nobody'
    locked = []
    env._lock_down = lambda sb, repo, paths: locked.extend(paths) or 0.0
    s = open_local(env, task, tmp_path)
    s.setup()
    g = s.grade(env.grade_inputs(task))             # the test command itself cannot drop privileges on this host
    s.close()
    assert g['grader_user'] == 'nobody' and g['lock_seconds'] == 0.0
    assert sorted(locked) == [b'docs', b'fixtures/data.txt', b'mimo_test_command.sh', b'tests/helpers.py',
                              b'tests/test_old.py']
