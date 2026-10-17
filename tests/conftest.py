"""Shared test helpers: build a local "image" (the template directory LocalSandbox uses) and a task file."""
import json
import os
import subprocess

GIT_ENV = {**os.environ, 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_AUTHOR_NAME': 't',
           'GIT_AUTHOR_EMAIL': 't@t', 'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@t'}

FIX = ('<tool_call>\n<function=bash>\n<parameter=command>\n'
       "sed -i.bak 's/a - b/a + b/' /testbed/calc.py && rm /testbed/calc.py.bak && cat /testbed/calc.py\n"
       '</parameter>\n</function>\n</tool_call>')


def _run(c, cwd):
    return subprocess.run(['bash', '-c', c], cwd=cwd, check=True, capture_output=True, text=True, env=GIT_ENV)


def make_swe_tasks(tmp_path, n=1, script=None):
    """The initial commit's calc.py has a bug; refs/remotes/origin/master points at the "future" fix commit
    (simulating a class C image leak). test_patch adds hidden tests and mimo_test_command.sh (`script` replaces
    its default body). Returns (task file path, [task ids]); each task's image is the template directory."""
    tpl = tmp_path / 'tpl'
    tpl.mkdir(exist_ok=True)
    _run('git init -q -b master && printf "def add(a, b):\\n    return a - b\\n" > calc.py && git add -A && '
         'git commit -qm base && git checkout -q -b fix && sed -i.bak "s/a - b/a + b/" calc.py && rm calc.py.bak && '
         'git commit -qam fix && git update-ref refs/remotes/origin/master HEAD && git checkout -q master && '
         'git branch -D fix -q', tpl)
    tp = tmp_path / 'tp'
    tp.mkdir(exist_ok=True)
    (tp / 'mimo_test_command.sh').write_text(script or 'cd "$(dirname "$0")" && PYTHONPATH=. python3 tests/test_calc.py\n')
    test_patch = _run('git init -q && mkdir tests && '
                      'printf \'from calc import add\\nassert add(2, 3) == 5\\nprint("ok")\\n\' > tests/test_calc.py && '
                      'git add -A && git diff --cached --no-color', tp).stdout
    ids = [f'fake-{i}' for i in range(n)]
    path = tmp_path / 'tasks.jsonl'
    with open(path, 'w') as f:
        for i in ids:
            f.write(json.dumps({'id': i, 'image': str(tpl), 'cwd': '/testbed', 'lang': 'python',
                                'problem_statement': 'add() subtracts', 'test_patch': test_patch,
                                'test_command': 'bash /testbed/mimo_test_command.sh'}) + '\n')
    return str(path), ids
