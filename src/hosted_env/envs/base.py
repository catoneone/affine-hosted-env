"""Environment interface (design §1 "not tied to SWE"): an environment = a set of tasks + one Docker image per
task + task prompt + tools + grading command.

The gateway and the worker use the same class:
- Gateway side: reads the task file, assigns tasks (task_ids / image_of), sends the worker a public_spec without
  grading material when a task is opened, and sends grade_inputs only at grading time (design §6: test_patch is
  sent only for grading and deleted after use).
- Worker side: uses public_spec to start the sandbox, run setup, provide the initial messages and tools, execute
  tool calls, and finally grade with grade_inputs.

`DockerTaskEnv` is the most generic task type: a single bash tool, grading = run test_command, exit code 0 scores 1.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from ..infra.receipt import EnvReceiptSpec

# The bash tool and observation format follow Xiaomi mimoagent's bash-only config (copied verbatim by the
# EXP-014 harness, MIT)
SYSTEM_TEMPLATE = ('You are an agent, your current working directory is {cwd}.\n\n'
                   'You can use the tools available to you to interact with the computer to assist the user in completing tasks.')
INSTANCE_TEMPLATE = 'Fix the following issue:\n\n{task}'

BASH_DEFAULT_TIMEOUT = 60
BASH_MAX_TIMEOUT = 300

BASH_DESCRIPTION = f"""Executes a bash command and returns its output.

This is the only available tool. Use standard shell commands and redirections to inspect, search, create, and edit files, run tests, and perform other tasks.

- Commands start in the configured current working directory.
- Each call runs in a new subshell, so directory and environment changes do not persist. Include any required `cd` or environment assignments in each command.
- Shell utilities such as `cat`, `sed`, `awk`, `grep`, and `find`, as well as shell redirections, may be used for file operations.
- Avoid interactive commands. Use non-interactive options where supported.
- Multiple calls in one message run concurrently and can race when they touch the same file (for example one call editing a file while another runs its tests, or two calls writing the same file); concurrent writes overwrite each other without any error.
- `timeout` is in seconds: default {BASH_DEFAULT_TIMEOUT}, max {BASH_MAX_TIMEOUT}. Use a larger value for long-running builds or test suites."""

BASH_TOOL = {
    'type': 'function',
    'function': {
        'name': 'bash',
        'description': BASH_DESCRIPTION,
        'parameters': {
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command string to execute; can span multiple lines and is used exactly as provided.',
                },
                'timeout': {
                    'type': 'integer',
                    'description': (f'Optional timeout in seconds for this command (default {BASH_DEFAULT_TIMEOUT}, '
                                    f'max {BASH_MAX_TIMEOUT}). Set a higher value for long-running commands (builds, test suites).'),
                },
            },
            'required': ['command'],
        },
    },
}

MAX_OBSERVATION_CHARS = 40000

_NOTICE = ('[TRUNCATION WARNING] The tool output below is INCOMPLETE: it was {total} characters, exceeding the '
           '{max_chars}-character limit, so the MIDDLE was cut out. Only the first {head} and last {tail} characters '
           'are kept; {omitted} characters are missing at the point marked [...OUTPUT OMITTED...]. Do NOT assume the '
           'text is contiguous across that marker. If you need the omitted part, make a narrower request instead '
           '(read only the relevant line range, or filter / paginate the command output) rather than repeating the '
           'same call.\n\n')
_CUT = ('\n\n[...OUTPUT OMITTED: {omitted} characters truncated from the middle ({total} total, kept first {head} + '
        'last {tail}). The text below resumes near the END of the output...]\n\n')


def truncate_middle(text: str, max_chars: int = MAX_OBSERVATION_CHARS) -> str:
    """Official utils/truncate.py: keep half from the head and half from the tail, prepend a prominent notice,
    and mark the cut."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    f = dict(total=len(text), max_chars=max_chars, head=head, tail=tail, omitted=len(text) - head - tail)
    return _NOTICE.format(**f) + text[:head] + _CUT.format(**f) + text[-tail:]


def run_bash(sb, args: dict) -> str:
    """Official BashTool.execute + DefaultAgent._format_observation."""
    command = str(args.get('command') or '').strip()
    if not command:
        return "Bash tool requires a non-empty 'command' parameter"
    raw = args.get('timeout')
    try:
        t = int(raw) if raw is not None else BASH_DEFAULT_TIMEOUT
    except (TypeError, ValueError):
        t = BASH_DEFAULT_TIMEOUT
    t = min(t if t > 0 else BASH_DEFAULT_TIMEOUT, BASH_MAX_TIMEOUT)
    r = sb.execute(command, timeout=t)
    if r['reason'] == 'client_timeout':
        out = (f'Error: Command timed out after {t}s. Partial output below. Retry with a larger `timeout` parameter '
               f'(max {BASH_MAX_TIMEOUT}s), or use a narrower command.\n{r["output"]}')
        meta = {'returncode': r['returncode'], 'reason': 'client_timeout', 'timeout': t}
    else:
        out, meta = r['output'], {'returncode': r['returncode']}
    return truncate_middle(out) + f'\n\nTool metadata: {meta}'


class SetupError(RuntimeError):
    """The task itself is broken (image git history not truncated, etc.): recorded as an infra error, not the
    model's fault."""


class GradeError(RuntimeError):
    """The grading environment is broken (test patch does not apply, etc.): recorded as an infra error."""


SandboxFactory = Callable[[str, str], object]   # (image, cwd) -> sandbox


class EnvSession:
    """Environment state of one rollout on the worker."""

    def __init__(self, env: 'DockerTaskEnv', spec: dict, sandbox, factory: SandboxFactory | None = None):
        self.env, self.spec, self.sb = env, spec, sandbox
        self.factory = factory            # makes the fresh grading sandbox (same image, same node)
        self.grader_sb = None
        self.tools = [BASH_TOOL]
        self.state: dict = {}

    def setup(self) -> None:
        self.sb.start()
        self.env.setup(self)

    def initial_messages(self) -> list[dict]:
        return self.env.initial_messages(self.spec)

    def call_tool(self, name: str, arguments: dict) -> str:
        if name != 'bash':
            return f"Error: Unknown tool '{name}'. The only available tool is 'bash'."
        return run_bash(self.sb, arguments)

    def grade(self, grade_inputs: dict) -> dict:
        return self.env.grade(self, grade_inputs)

    def close(self) -> None:
        for sb in (self.sb, self.grader_sb):
            if sb is not None:
                sb.cleanup()


class DockerTaskEnv:
    """Generic task type. The task file is jsonl; each line has at least id, image, cwd, problem_statement,
    test_command."""

    kind = 'docker_task'

    def __init__(self, env_id: str, cfg: dict):
        self.env_id, self.cfg = env_id, cfg
        self.group_size = int(cfg.get('group_size', 8))
        self.success_reward = float(cfg.get('success_reward', 1.0))
        self.verifier_timeout = int(cfg.get('verifier_timeout', 1800))
        self._rows: dict[str, dict] | None = None

    # ------------------------------------------------ gateway side
    def _load(self) -> dict[str, dict]:
        if self._rows is None:
            path = self.cfg.get('tasks')
            if not path:
                raise RuntimeError(f'env {self.env_id}: no tasks file configured')
            rows = {}
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[str(r['id'])] = r
            if self.cfg.get('require_image_digest', True):   # on by default; false only for local dev with tags
                from .images import is_pinned
                loose = sorted(k for k, r in rows.items() if not is_pinned(r.get('image', '')))
                if loose:
                    raise RuntimeError(f'env {self.env_id}: {len(loose)} task images are not pinned by digest '
                                       f'(e.g. {loose[0]}); run `hosted-env pin-images`')
            allowed = self.cfg.get('indices')            # tasks minable this epoch (environments[].indices in the opening)
            if allowed is not None:
                rows = {k: v for k, v in rows.items() if k in set(map(str, allowed))}
            self._rows = rows
        return self._rows

    def task_ids(self) -> list[str]:
        return sorted(self._load())

    def image_of(self, task: str) -> str | None:
        return self._load()[task].get('image')

    def public_spec(self, task: str) -> dict:
        r = self._load()[task]
        return {'id': str(r['id']), 'image': r['image'], 'cwd': r['cwd'], 'lang': r.get('lang', ''),
                'problem_statement': r['problem_statement']}

    def grade_inputs(self, task: str) -> dict:
        r = self._load()[task]
        return {'test_command': r['test_command'],
                'verifier_timeout': int(r.get('verifier_timeout_sec', self.verifier_timeout))}

    # ------------------------------------------------ worker side
    def open(self, spec: dict, sandbox_factory: SandboxFactory) -> EnvSession:
        return EnvSession(self, spec, sandbox_factory(spec['image'], spec['cwd']), sandbox_factory)

    def initial_messages(self, spec: dict) -> list[dict]:
        system = self.cfg.get('system_template', SYSTEM_TEMPLATE)
        instance = self.cfg.get('instance_template', INSTANCE_TEMPLATE)
        return [{'role': 'system', 'content': system.format(cwd=spec['cwd'])},
                {'role': 'user', 'content': instance.format(task=spec['problem_statement'])}]

    def setup(self, s: EnvSession) -> None:
        pass

    def grade(self, s: EnvSession, gi: dict) -> dict:
        r = s.sb.execute(gi['test_command'], cwd=s.spec['cwd'], timeout=gi['verifier_timeout'])
        return {'reward': 1.0 if r['returncode'] == 0 else 0.0, 'verifier_returncode': r['returncode'],
                'verifier_reason': r['reason'], 'test_output': r['output'][-3000:]}

    # ------------------------------------------------ what the infrastructure signs and logs
    # RECEIPT names the context fields this kind adds to every receipt (the infrastructure adds its own core
    # fields); receipt_context fills exactly those for one rollout, from its session (None when setup never
    # produced one), the grade inputs and the grade ({} when the rollout was not graded).
    RECEIPT = EnvReceiptSpec(fields={'image': str, 'test_command_sha256': str,
                                     'verifier_returncode': (int, type(None))})

    def receipt_context(self, s: EnvSession | None, gi: dict, grade: dict) -> dict:
        rc = grade.get('verifier_returncode')
        cmd = gi.get('test_command', '')
        return {'image': str((s.spec if s is not None else {}).get('image', '')),
                'test_command_sha256': hashlib.sha256(cmd.encode()).hexdigest() if cmd else '',
                'verifier_returncode': rc if type(rc) is int else None}

    def log_fields(self, grade: dict) -> dict:
        """Grade details for the executor's per-rollout log line (small values only, never test output)."""
        return {k: grade[k] for k in ('verifier_returncode', 'verifier_reason', 'grade_detail') if k in grade}

    def artifacts(self, grade: dict) -> dict[str, str]:
        """RECEIPT.artifacts for the group attachment, base64 (each bound by its *_sha256 receipt field)."""
        return {}
