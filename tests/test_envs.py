"""toolcall.py, envs/: ported from EXP-014 test_harness.py. No docker needed: LocalSandbox runs in a temp
directory on the host."""
import os

import pytest
from conftest import make_swe_tasks

from hosted_env.envs import SweEnv
from hosted_env.envs.base import BASH_TOOL, truncate_middle
from hosted_env.envs.swe import future_commits, patch_touched_files, reset_test_files_cmd
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
    gi = env.grade_inputs(task)
    assert s.grade(gi)['reward'] == 0.0                                # base state grades 0
    s.sb.execute('sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak')
    g = s.grade(gi)
    assert g['reward'] == 1.0 and 'a + b' in g['model_patch']
    assert not os.path.exists(os.path.join(s.sb.root, 'tests', 'test_calc.py'))   # test files restored after grading
    s.sb.execute(f'git checkout -q {s.state["base"]} -- calc.py; printf "exit 0\\n" > mimo_test_command.sh')
    assert s.grade(gi)['reward'] == 0.0                                # an always-pass grading script planted by the model is removed by the reset
    root = s.sb.root
    s.close()
    assert not os.path.exists(root)


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
    home = sb.root + '/.home/'
    assert sb._map('/w/a.py') == sb.root + '/a.py'
    assert sb._map('cat /tmp/x && cd /w') == f'cat {home}x && cd {sb.root}'
    assert sb._map('"/tmp/y" :/tmp/z') == f'"{home}y" :{home}z'
    assert sb._map('/var/tmp/k $HOME/tmp/q') == '/var/tmp/k $HOME/tmp/q'
    sb.root = real_root
    sb.cleanup()
