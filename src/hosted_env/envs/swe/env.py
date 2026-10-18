"""SWE task type for the `code` split of Xiaomi MiMo-V2.6-RL-oss (the environment part of the EXP-014 harness,
aligned with mimoagent opensource_code.py).

- setup: truncate git history to the base commit (class-C images carry upstream remotes/origin/*; cleaned in place
  following the official strip implementation), restore environment variables lost from the image (in
  DockerSandbox), optional anti_hack_cleanup (enabled in mini-bash.yaml).
- Grading (SweEnv.grade), in a fresh container of the same image: apply the model's diff from the start tree, put
  protected test infrastructure back, apply test_patch, lock the test files and run test_command as nobody (cfg
  grader_user = "root" opts out); exit code 0 scores 1. A test patch that does not apply, or a runtime that
  cannot drop privileges, is an infra error (GradeError), never a reward of 0.
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import shlex
import time

from ...infra.providers.sandbox.base import SandboxCommandError, SandboxLimitError
from ..base import DockerTaskEnv, EnvSession, GradeError, SetupError
from .cleanup import anti_hack_cleanup, future_commits, git_strip
from .protect import (_JS_CONFIG_RE, GRADER_PARSERS, STRUCTURED_MANIFESTS, jsts_config_refs, manifest_reverts,
                      protected_paths, shadow_added)
from .reports import _REPORT_EXT, _REPORT_PARSERS, GO_REPORT_SHIM, report_verdict
from .scripts import command_repo_paths, script_named_files, script_protected, test_patch_scripts


TEST_PATCH_REMOTE = '/tmp/_hosted_env_test.patch'
MODEL_PATCH_REMOTE = '/tmp/_hosted_env_model.patch'
RESET_LIST_REMOTE = '/tmp/_hosted_env_reset.list'
TREE_INDEX_REMOTE = '/tmp/_hosted_env_tree.index'
LOCK_LIST_REMOTE = '/tmp/_hosted_env_lock.list'
GRADER_UID = 65534                     # nobody: the hidden tests run without root (cfg grader_user = "root" to opt out)
# no_new_privs: setuid-root binaries in the image (sudo, su, ...) cannot hand root back to the code under test
DROP_PRIVS = f'setpriv --reuid={GRADER_UID} --regid={GRADER_UID} --clear-groups --no-new-privs --'
GRADER_BASH = 'bash --noprofile --norc -c'   # the drop probes run under exactly what the test command runs under
MAX_MODEL_PATCH_BYTES = 2_000_000
GRADER_VERSION = 'isolated-v3'         # v1: tests as root; v2: tests as GRADER_UID; v3: structured manifests, wider revert


def as_grader(command: str) -> str:
    """Run `command` as GRADER_UID with no supplementary groups, no way to regain privileges, and no profile or rc
    file. HOME stays the image's (its caches and toolchains: Go module cache, nvm, cargo, ...), which _lock_down
    hands to the grader."""
    return f'{DROP_PRIVS} env USER=nobody LOGNAME=nobody {GRADER_BASH} {shlex.quote(command)}'


class SweEnv(DockerTaskEnv):
    """Config options: cleanup (anti-hack cleanup, on by default), git_leak ('strip' default / 'error'),
    require_expected_tests (on by default; false only for local development and the baseline tool).

    Grading never trusts the rollout container: the model's changes are captured as a git patch, that container is
    destroyed, and the hidden tests run in a fresh container from the same image with only the patch applied,
    the test infrastructure reset (`protected_paths`), and a non-login shell (no profile or rc file is read)."""

    kind = 'mimo_swe'

    def _load(self) -> dict[str, dict]:
        """Only tasks with a second check are offered (require_expected_tests, default on). On a task graded by the
        exit code alone the model wins by making the run exit 0 without passing anything -- a Go init() that calls
        os.Exit(0), a JS module that calls process.exit(0) when the tests import it -- and no revert rule can stop
        that, since it edits the very code the fix must edit (EXP-018 cheat matrix; Affine review 2026-10-18: the
        per-test-id check gates going live). So a task needs fail_to_pass ids from `hosted-env baseline` and a
        report kind the grader parses; the rest stay out of the pool until the baseline covers them."""
        if self._rows is None:
            rows = super()._load()
            if self.cfg.get('require_expected_tests', True):
                kept = {k: r for k, r in rows.items() if r.get('fail_to_pass') and r.get('report_kind') in _REPORT_PARSERS}
                if rows and not kept:
                    raise RuntimeError(f'env {self.env_id}: none of the {len(rows)} tasks has expected tests '
                                       '(fail_to_pass + report_kind); run `hosted-env baseline --tasks-out`, or set '
                                       'require_expected_tests = false for local development')
                self._rows = kept
        return self._rows

    def grade_inputs(self, task: str) -> dict:
        r = self._load()[task]
        extra = {k: r[k] for k in ('fail_to_pass', 'pass_to_pass', 'report_kind') if r.get(k)}
        return {**super().grade_inputs(task), 'test_patch': r['test_patch'], **extra}

    def _prepare_repo(self, sb, repo: str, lang: str) -> str:
        """Bring a fresh container's repo to the base state the model starts from (also used for the grading
        container, so both start from the same tree). Returns the base commit."""
        sb.execute(f'git config --global --add safe.directory {repo}', timeout=60)
        if sb.execute('git rev-parse --git-dir', cwd=repo, timeout=60)['returncode'] != 0:
            r = sb.execute('git init -q && git add -A && git -c user.email=env@local -c user.name=env '
                           'commit -q -m baseline --allow-empty', cwd=repo, timeout=600)
            if r['returncode'] != 0:
                raise SetupError(f'git init failed: {r["output"][-300:]}')
        r = sb.execute('git rev-parse HEAD', cwd=repo, timeout=60)
        base = r['output'].strip().split()[-1] if r['output'].strip() else ''
        if r['returncode'] != 0 or len(base) != 40:
            raise SetupError(f'cannot read HEAD: {r["output"][-300:]}')
        if self.cfg.get('cleanup', True):
            anti_hack_cleanup(sb, repo, lang)
        return base

    def _tree(self, sb, repo: str, base: str) -> str:
        """Content hash of the working tree as git sees it: the base commit's tracked files plus every non-ignored
        file, read through a throwaway index so neither HEAD nor the real index changes (the model's view stays
        exactly as before). Identical trees give identical hashes, which is how the grading container proves it
        starts where the rollout started even when an image's working tree differs from its own HEAD."""
        r = sb.execute(f'export GIT_INDEX_FILE={TREE_INDEX_REMOTE}; rm -f "$GIT_INDEX_FILE"; '
                       f'git read-tree {base} && git add -A >/dev/null 2>&1 && git write-tree; '
                       'rc=$?; rm -f "$GIT_INDEX_FILE"; exit $rc', cwd=repo, timeout=600, login=False)
        tree = r['output'].strip().split()[-1] if r['output'].strip() else ''
        if r['returncode'] != 0 or len(tree) != 40:
            raise SetupError(f'cannot snapshot the working tree: {r["output"][-300:]}')
        return tree

    def setup(self, s: EnvSession) -> None:
        sb, repo = s.sb, s.spec['cwd']
        base = self._prepare_repo(sb, repo, s.spec.get('lang', ''))
        stripped = False
        if future_commits(sb, repo, base):
            if self.cfg.get('git_leak', 'strip') != 'strip':
                raise SetupError('image git history is not truncated at the base commit')
            git_strip(sb, repo, base)
            stripped = True
            if future_commits(sb, repo, base):
                raise SetupError('future commits still visible after git strip')
        s.state.update(base=base, git_stripped=stripped, start_tree=self._tree(sb, repo, base))

    def capture_patch(self, s: EnvSession) -> tuple[bytes, str]:
        """(patch bytes, problem) from the rollout container: the model's changes as a binary-safe git patch from
        the tree it started with to the tree it left (tracked and non-ignored files). It travels base64-encoded (no
        UTF-8 decoding, no stderr mixed in), with no login shell and with external diff drivers and textconv
        filters off."""
        repo = s.spec['cwd']
        try:
            end_tree = self._tree(s.sb, repo, s.state['base'])
            r = s.sb.execute(f'set -o pipefail; git diff --binary --no-color --no-ext-diff --no-textconv '
                             f'{s.state["start_tree"]} {end_tree} 2>/dev/null '
                             f'| head -c {MAX_MODEL_PATCH_BYTES + 1} | base64 | tr -d "\\n"',
                             cwd=repo, timeout=300, login=False)
        except SetupError:
            return b'', 'model_patch_capture_failed'      # e.g. the model deleted .git
        except SandboxCommandError:
            return b'', 'model_patch_capture_failed'      # the model deleted a binary exec needs (/bin/sh, git)
        except SandboxLimitError as e:                    # the model's commands blew its memory / disk limit
            return b'', f'rollout_sandbox_limit: {e}'
        try:
            data = base64.b64decode(r['output'].strip(), validate=True)
        except ValueError:
            return b'', 'model_patch_capture_failed'
        if len(data) > MAX_MODEL_PATCH_BYTES:
            return b'', 'model_patch_too_large'
        if r['returncode'] != 0:
            return b'', 'model_patch_capture_failed'
        return data, ''

    def _put_bytes(self, sb, data: bytes, path: str) -> None:
        sb.copy_text_to(base64.b64encode(data).decode(), path + '.b64')
        r = sb.execute(f'base64 -d < {path}.b64 > {path} && rm -f {path}.b64', timeout=120, login=False)
        if r['returncode'] != 0:
            raise GradeError(f'cannot write {path}: {r["output"][-200:]}')

    def _patch_paths(self, sb, repo: str, path: str) -> list[bytes]:
        """The paths a patch leaves behind, as git itself parses it (`git apply --numstat -z`, so quoted names and
        names containing ' b/' come out right). For a rename or copy git apply names only the new side, which is
        all locking needs; to know everything a patch changes, including the path a rename removes, use
        _changed_paths."""
        r = sb.execute(f'set -o pipefail; git apply --numstat -z {path} 2>/dev/null | base64 | tr -d "\\n"', cwd=repo,
                       timeout=120, login=False)
        if r['returncode'] != 0:
            raise GradeError(f'cannot list the paths of {path}: ' + r['output'][-300:])
        items, out, i = base64.b64decode(r['output'].strip(), validate=True).split(b'\0'), [], 0
        while i < len(items):
            fields = items[i].split(b'\t')
            if len(fields) == 3 and fields[2] == b'' and i + 2 < len(items):   # diff-style rename record, tolerated
                out += [items[i + 1], items[i + 2]]
                i += 3
                continue
            if len(fields) == 3 and fields[2]:
                out.append(fields[2])
            i += 1
        return out

    def _changed_paths(self, sb, repo: str, start: str, path: str,
                       keep_deleted: bool = True, added_only: bool = False) -> tuple[list[bytes] | None, str]:
        """Every path whose content or mode the patch at `path` changes relative to the tree `start`, with renames
        and copies counted as their separate deletes and adds. The patch is applied to a throwaway index holding
        `start` by the grading container's own git and compared with --no-renames, so a protected file the model
        renamed away (conftest.py -> notes.txt) is reported and put back, and force-added ignored files count.
        Returns (paths, '') or, when the patch does not apply to the start tree itself, (None, git's complaint):
        a real diff between two trees always applies there, so that is the model's doing (for example a hunk on a
        git-ignored file that exists on disk but in neither tree) and is graded 0, never an infra error.
        keep_deleted=False leaves out the paths the patch deletes (what remains after it is applied)."""
        idx = f'GIT_INDEX_FILE={TREE_INDEX_REMOTE}'
        r = sb.execute(f'rm -f {TREE_INDEX_REMOTE} && {idx} git read-tree {start}', cwd=repo, timeout=120,
                       login=False)
        if r['returncode'] != 0:
            raise GradeError('cannot load the start tree into a scratch index: ' + r['output'][-300:])
        ap = sb.execute(f'{idx} git apply --cached --binary --whitespace=nowarn {path}', cwd=repo, timeout=300,
                        login=False)
        if ap['returncode'] != 0:
            sb.execute(f'rm -f {TREE_INDEX_REMOTE}', timeout=60, login=False)
            return None, ap['output'][-300:] or f'git apply --cached exited {ap["returncode"]}'
        only = '--diff-filter=A ' if added_only else ('' if keep_deleted else '--diff-filter=d ')
        r = sb.execute(f'set -o pipefail; {idx} git diff --cached --no-renames --no-ext-diff --name-only -z {only}'
                       f'{start} '
                       f'2>/dev/null | base64 | tr -d "\\n"; rc=$?; rm -f {TREE_INDEX_REMOTE}; exit $rc', cwd=repo,
                       timeout=300, login=False)
        if r['returncode'] != 0:
            raise GradeError(f'cannot list the paths {path} changes: ' + r['output'][-300:])
        return [p for p in base64.b64decode(r['output'].strip(), validate=True).split(b'\0') if p], ''

    def _base_config_files(self, sb, repo: str, start: str) -> list[str]:
        """Test-runner / tsconfig config files present in the base tree (filtered in the shell so a huge repo does
        not blow the output cap). Their referenced files are protected whether or not the model changed the config."""
        pat = r'(^|/)(package\.json|jest\.config\.|vitest\.config\.|vite\.config\.|\.mocharc|babel\.config\.|tsconfig)'
        r = sb.execute(f'set -o pipefail; git ls-tree -r --name-only {start} 2>/dev/null | grep -E {shlex.quote(pat)}'
                       ' | head -c 200000 | base64 | tr -d "\\n"', cwd=repo, timeout=120, login=False)
        try:
            names = base64.b64decode(r['output'].strip(), validate=True).decode('utf-8', 'replace').splitlines()
        except ValueError:
            return []
        out = []
        for p in names:
            b = p.rsplit('/', 1)[-1]
            if b == 'package.json' or _JS_CONFIG_RE.match(b):
                out.append(p)
        return out[:300]

    def _importable_modules(self, sb, repo: str) -> set[str]:
        """Top-level module names importable on the image (builtins + everything on the default sys.path), used to
        decide which model-added top-level names shadow a real module. Run from `/`, never the repo, so the repo's
        own modules are not counted. Empty when no python is on the image (then only the pytest internals apply)."""
        probe = ('import sys,pkgutil,json;'
                 "m=set(sys.builtin_module_names);"
                 "m|={x.name for x in pkgutil.iter_modules() if '.' not in x.name};"
                 'print(json.dumps(sorted(m)))')
        for py in ('python3', 'python'):
            r = sb.execute(f'{py} -c {shlex.quote(probe)}', cwd='/', timeout=120, login=False)
            if r['returncode'] == 0 and r['output'].strip():
                try:
                    return set(json.loads(r['output'].strip().splitlines()[-1]))
                except ValueError:
                    pass
        return set()

    def _install_go_shim(self, sb, repo: str, rdir: str, rpath: str, res: dict) -> str:
        """Put GO_REPORT_SHIM first on PATH for the test command; returns the env prefix ('' when not installed).
        Runs as root before the tests, so the shim and its directory are root-owned and not writable by the grader
        user. The real go is what the image's own PATH resolves, and must be an absolute path outside the repo (a
        model-added `go` in the repo is never the one wrapped). When there is no such go the shim is skipped and
        recorded: the report stays empty and a task with expected tests fails the second check -- never a void,
        since a missing go is fixed by the pinned image, which the baseline ran against too."""
        r = sb.execute('command -v go || true', cwd=repo, timeout=60, login=False)
        real = (r['output'].strip().splitlines() or [''])[-1].strip()
        root = repo.rstrip('/') + '/'
        if not real.startswith('/') or real.startswith(root) or any(c in real for c in '\'"$`\\'):
            res['go_report_shim'] = 'skipped: no go outside the repo on PATH'
            return ''
        bindir = f'{rdir}-bin'
        sb.execute(f'rm -rf {bindir} && mkdir -p {bindir}', timeout=60, login=False)
        sb.copy_text_to(GO_REPORT_SHIM.replace('__REAL_GO__', shlex.quote(real)), f'{bindir}/go')
        sb.execute(f'chmod 755 {bindir} {bindir}/go', timeout=60, login=False)
        res['go_report_shim'] = real
        return f'PATH={bindir}:"$PATH" MIMO_GO_REPORT={shlex.quote(rpath)} '

    def _read_abs(self, sb, path: str) -> bytes:
        """Raw bytes of an absolute file (a structured test report in a random dir), base64 so binary is safe."""
        if not path:
            return b''
        q = shlex.quote(path)
        r = sb.execute(f'set -o pipefail; [ -f {q} ] && base64 < {q} | tr -d "\\n" || true', timeout=120, login=False)
        try:
            return base64.b64decode(r['output'].strip(), validate=True) if r['output'].strip() else b''
        except ValueError:
            return b''

    def _read_blob(self, sb, repo: str, ref: str | None, rel: str) -> bytes:
        """Raw bytes of `rel` at git ref `ref` (base tree), or from the working file when ref is None (the model's
        version, after its patch was applied). Missing on either side -> b''. base64 so binary is safe. Read in the
        executor, not parsed in the container, so a crafted manifest cannot run code during grading."""
        q = shlex.quote(rel)
        src = (f'export GIT_LITERAL_PATHSPECS=1; git cat-file -e {ref}:{q} 2>/dev/null && git show {ref}:{q}'
               if ref else f'[ -f {q} ] && cat {q}')
        r = sb.execute(f'set -o pipefail; {{ {src}; }} 2>/dev/null | base64 | tr -d "\\n"', cwd=repo, timeout=120,
                       login=False)
        try:
            return base64.b64decode(r['output'].strip(), validate=True) if r['output'].strip() else b''
        except ValueError:
            return b''

    def _reset_paths(self, sb, repo: str, start: str, paths: list[bytes]) -> None:
        """Put each path back to the start tree, or delete it when the start tree does not have it (NUL-separated,
        so any name)."""
        if not paths:
            return
        self._put_bytes(sb, b'\0'.join(paths) + b'\0', RESET_LIST_REMOTE)
        r = sb.execute(f'export GIT_LITERAL_PATHSPECS=1; while IFS= read -r -d "" f; do\n'
                       f'  if git cat-file -e {start}:"$f" 2>/dev/null; then git checkout {start} -- "$f" || exit 1\n'
                       '  else git rm -rf -q --cached --ignore-unmatch -- "$f" >/dev/null && rm -rf -- "$f"; fi\n'
                       f'done < {RESET_LIST_REMOTE}; rm -f {RESET_LIST_REMOTE}', cwd=repo, timeout=300, login=False)
        if r['returncode'] != 0:
            raise GradeError('reset_protected_failed: ' + r['output'][-300:])

    def _reset_test_paths(self, sb, repo: str, base: str, paths: list[bytes]) -> None:
        """Before the test patch goes in: every path it touches is put back to the base commit, or removed when
        the base does not have it, whatever the model left there. That includes a directory at a file's path
        (mimo_test_command.sh/x) and a leading component turned into a file or a symlink (tests -> elsewhere):
        the first such component is removed, then the path itself (rm -rf), so the trusted test patch always
        applies and the model cannot turn its own rollout into an infra error. Two passes: everything is cleared
        before anything is restored, so restoring `fixtures` (a file the test patch replaces with a directory)
        is not undone by clearing `fixtures/data.txt`. Literal pathspecs: a name is never git pathspec magic."""
        if not paths:
            return
        self._put_bytes(sb, b''.join(p + b'\0' for p in paths), RESET_LIST_REMOTE)
        r = sb.execute(f'''export GIT_LITERAL_PATHSPECS=1
while IFS= read -r -d "" f; do
  case "/$f/" in //*|*/../*) echo "unsafe test path: $f"; exit 4;; esac
  a=; rest=$f
  while :; do                                   # top-down: stop at the first component that is not a real dir
    c=${{rest%%/*}}; a=${{a:+$a/}}$c
    [ "$rest" = "$c" ] && break
    rest=${{rest#*/}}
    if [ -L "$a" ] || {{ [ -e "$a" ] && [ ! -d "$a" ]; }}; then rm -f -- "$a" || exit 1; break; fi
  done
  if [ -d "$f" ] && [ ! -L "$f" ] && [ "$(git ls-tree {base} -- "$f" | cut -c1-6)" = 160000 ]; then
    continue                                    # a submodule checkout: the test patch moves only its pointer
  fi
  rm -rf -- "$f" || exit 1
done < {RESET_LIST_REMOTE}
while IFS= read -r -d "" f; do
  if git cat-file -e {base}:"$f" 2>/dev/null; then git checkout -q {base} -- "$f" || exit 1
  else git rm -rf -q --cached --ignore-unmatch -- "$f" >/dev/null || exit 1; fi
done < {RESET_LIST_REMOTE}; rm -f {RESET_LIST_REMOTE}''', cwd=repo, timeout=300, login=False)
        if r['returncode'] != 0:
            raise GradeError('reset_tests_failed: ' + r['output'][-300:])

    def _lock_down(self, sb, repo: str, test_paths: list[bytes]) -> float:
        """Before the hidden tests run as GRADER_UID. The repo and the image's HOME become the grader's, so tests
        write caches and scratch files exactly where they would as root (pytest cache, rollup temp configs, the
        Go build cache, nvm/cargo toolchains). The test infrastructure stays root's: every file of the test patch
        (`test_paths`, as git parses the patch) is root-owned and read-only, and each directory holding one, up to
        the repo root, is root-owned with mode 1777 (like /tmp); so are the repo's ancestors inside HOME. What this
        guarantees: the grader cannot modify, replace, rename or delete the existing test files and grading scripts
        (nor the repo itself) while they run. What it does not: it can still create new files next to them (a
        conftest.py in a directory pytest has not collected yet) and change everything else in the repo, and the
        code under test can still end its own test process with exit 0. Before returning, the drop is tried for
        real, so a runtime that cannot drop privileges is a GradeError rather than a reward of 0 for every
        rollout. Returns the seconds it took. (EXP-018 parity runs: a read-only repo root broke pytest with -W
        error and rollup; a changed HOME hid the Go module cache.)"""
        t0 = time.time()
        self._put_bytes(sb, b''.join(p + b'\0' for p in test_paths), LOCK_LIST_REMOTE)
        r = sb.execute(f'''set -e
command -v setpriv >/dev/null || {{ echo "setpriv not found: cannot drop privileges"; exit 3; }}
repo=$(pwd -P)
chown -R {GRADER_UID}:{GRADER_UID} .
h="${{HOME:-/root}}"
if [ -d "$h" ]; then h=$(CDPATH= cd -- "$h" && pwd -P); else h=/; fi
if [ "$h" != / ]; then
  chown -R {GRADER_UID}:{GRADER_UID} "$h"
  d=$(dirname -- "$repo")
  while [ "$d" != / ]; do
    case "$d/" in "$h"/*) chown -h 0:0 -- "$d"; chmod 1777 -- "$d";; esac
    d=$(dirname -- "$d")
  done
fi
probe=
while IFS= read -r -d "" f; do
  case "/$f/" in //*|*/../*) echo "unsafe test path: $f"; exit 4;; esac
  d=$(dirname -- "$f")
  while [ "$d" != . ]; do                       # check the whole path first: change nothing below a symlink
    if [ -L "$d" ]; then echo "symlink in a test path: $d"; exit 4; fi
    d=$(dirname -- "$d")
  done
  d=$(dirname -- "$f")
  while [ "$d" != . ]; do
    if [ -d "$d" ]; then chown -h 0:0 -- "$d"; chmod 1777 -- "$d"; fi
    d=$(dirname -- "$d")
  done
  [ -e "$f" ] || [ -L "$f" ] || continue
  chown -h 0:0 -- "$f"
  if [ ! -L "$f" ]; then chmod a-w -- "$f"; [ -n "$probe" ] || [ ! -f "$f" ] || probe=$f; fi
done < {LOCK_LIST_REMOTE}
rm -f {LOCK_LIST_REMOTE}
chown 0:0 .; chmod 1777 .
{DROP_PRIVS} {GRADER_BASH} 'test "$(id -u)" = {GRADER_UID} && test "$(id -g)" = {GRADER_UID}' \\
  || {{ echo "cannot drop privileges to {GRADER_UID} in this runtime"; exit 3; }}
if [ -n "$probe" ]; then
  {DROP_PRIVS} {GRADER_BASH} 'test ! -w "$1"' _ "$probe" \\
    || {{ echo "test file still writable after the drop: $probe"; exit 3; }}
fi
''', cwd=repo, timeout=900, login=False)
        if r['returncode'] != 0:
            raise GradeError('cannot prepare non-root grading: ' + r['output'][-300:])
        return round(time.time() - t0, 1)

    def grade(self, s: EnvSession, gi: dict) -> dict:
        repo, grader = s.spec['cwd'], self.cfg.get('grader_user', 'nobody')
        if grader not in ('root', 'nobody'):                # a typo must not silently pick one of them
            raise GradeError(f'grader_user must be "root" or "nobody", not {grader!r}')
        patch, problem = self.capture_patch(s)
        res = {'model_patch': patch.decode('utf-8', 'replace'), 'model_patch_b64': base64.b64encode(patch).decode(),
               'model_patch_sha256': hashlib.sha256(patch).hexdigest(), 'reward': 0.0, 'grader': GRADER_VERSION,
               'grader_user': grader}
        if problem:                                         # the model's own doing (deleted .git, huge diff): 0
            return {**res, 'grade_detail': problem}
        if s.factory is None:
            raise GradeError('no sandbox factory for an isolated grading container')
        s.sb.cleanup()                                      # the rollout container is never used again
        sb = s.grader_sb = s.factory(s.spec['image'], repo)
        reverted, untrusted, t0 = [], False, None
        try:
            sb.start()
            base = self._prepare_repo(sb, repo, s.spec.get('lang', ''))
            start = self._tree(sb, repo, base)
            if start != s.state['start_tree']:
                raise GradeError(f'grading container starts from tree {start[:12]}, the rollout started from '
                                 f'{s.state["start_tree"][:12]} (non-deterministic image or setup)')
            if patch:
                self._put_bytes(sb, patch, MODEL_PATCH_REMOTE)
                # From here on the container processes bytes the model chose: a tiny git binary delta can expand
                # to gigabytes in `git apply` (review of PR #15). A limit kill from now on is the model's: 0.
                untrusted = True
                touched, why = self._changed_paths(sb, repo, start, MODEL_PATCH_REMOTE)   # first: changes nothing
                added: list[str] = []
                if touched is not None:
                    added = [p.decode('utf-8', 'surrogateescape')          # new files, read before the patch is gone
                             for p in (self._changed_paths(sb, repo, start, MODEL_PATCH_REMOTE,
                                                           added_only=True)[0] or [])]
                    ap = sb.execute(f'git apply --binary --whitespace=nowarn {MODEL_PATCH_REMOTE}', cwd=repo,
                                    timeout=300, login=False)
                    why = '' if ap['returncode'] == 0 else (ap['output'][-300:] or f'exit {ap["returncode"]}')
                sb.execute(f'rm -f {MODEL_PATCH_REMOTE}', timeout=60, login=False)
                if touched is None or why:                  # same start tree, so only the patch can be at fault
                    return {**res, 'grade_detail': 'model_patch_does_not_apply: ' + why}
                prot = protected_paths([p.decode('utf-8', 'surrogateescape') for p in touched])
                reverted = set(prot)
                for p in touched:
                    rel = p.decode('utf-8', 'surrogateescape')
                    if rel in reverted or rel.rsplit('/', 1)[-1] not in STRUCTURED_MANIFESTS:
                        continue
                    name = rel.rsplit('/', 1)[-1]
                    if manifest_reverts(name, self._read_blob(sb, repo, start, rel),
                                        self._read_blob(sb, repo, None, rel)):
                        reverted.add(rel)         # a test-relevant key changed, or it does not parse: reset to base
                reverted.update(shadow_added(added, self._importable_modules(sb, repo) if added else set()))
                cmd_dirs = command_repo_paths(gi['test_command'], repo)
                touched_set = {p.decode('utf-8', 'surrogateescape') for p in touched}
                if cmd_dirs:                       # files under a PATH dir / a script the command runs directly
                    reverted.update(f for f in touched_set if any(f == d or f.startswith(d + '/') for d in cmd_dirs))
                # the commands that really run are in the test patch's scripts (a MiMo task's outer command is only
                # `bash mimo_test_command.sh`): a repo file those scripts run, e.g. `node build/run-tests.js`
                exact, suffix, named_dirs = script_named_files(gi['test_command'], repo)
                for spath, text in test_patch_scripts(gi['test_patch']):
                    e, sx, nd = script_named_files(text, repo, spath.rpartition('/')[0])
                    exact |= e
                    suffix |= sx
                    named_dirs |= nd
                reverted.update(script_protected(touched_set, exact, suffix, named_dirs))
                cfg_refs: set = set()
                for cfgpath in self._base_config_files(sb, repo, start):
                    cdir, _, cname = cfgpath.rpartition('/')
                    cfg_refs.update(jsts_config_refs(cname, self._read_blob(sb, repo, start, cfgpath), cdir))
                reverted.update(touched_set & cfg_refs)   # a file a test-runner config loads: reset even if unchanged config
                res['grader_parsers'] = GRADER_PARSERS
                reverted = sorted(reverted)
                self._reset_paths(sb, repo, start, [p.encode('utf-8', 'surrogateescape') for p in reverted])
            sb.copy_text_to(gi['test_patch'], TEST_PATCH_REMOTE)
            test_paths = self._changed_paths(sb, repo, base, TEST_PATCH_REMOTE)[0] if base else None
            if test_paths is None:                          # does not apply to the base commit: take git's parse
                test_paths = self._patch_paths(sb, repo, TEST_PATCH_REMOTE)
            if base:
                self._reset_test_paths(sb, repo, base, test_paths)
            if grader == 'nobody':                          # lock what the patch leaves behind, not what it deletes
                test_paths = (self._changed_paths(sb, repo, base, TEST_PATCH_REMOTE, keep_deleted=False)[0]
                              if base else None)
                if test_paths is None:                      # numstat also names deleted files: rare fallback
                    test_paths = self._patch_paths(sb, repo, TEST_PATCH_REMOTE)
            ap = sb.execute(f'git apply --verbose {TEST_PATCH_REMOTE}', cwd=repo, timeout=300, login=False)
            sb.execute(f'rm -f {TEST_PATCH_REMOTE}', timeout=60, login=False)
            if ap['returncode'] != 0:
                raise GradeError('apply_test_patch_failed: ' + ap['output'][-500:])
            test_cmd = gi['test_command']
            expected = list(gi.get('fail_to_pass') or []) + list(gi.get('pass_to_pass') or [])
            report_kind = gi.get('report_kind') or ''
            collect = bool(gi.get('collect_report'))       # baseline mode (hosted-env baseline): report, no verdict
            want_report = report_kind in _REPORT_PARSERS and (expected or collect)
            rdir = rpath = pre = ''
            if want_report:
                rdir = f'/tmp/mimo-report-{secrets.token_hex(8)}'     # random: the model cannot predict / forge it
                rpath = f'{rdir}/report.{_REPORT_EXT[report_kind]}'
                sb.execute(f'rm -rf {rdir} && mkdir -p {rdir} && chmod 777 {rdir}', timeout=60, login=False)
                if report_kind == 'junit':   # make even an unmodified pytest harness emit the report
                    # xunit1 adds the `file` attribute (an unambiguous node id); the baseline also wants the
                    # tests of the files that do collect when a new test file does not. The repo's own addopts are
                    # kept: its pytest config is reset to base (protected), and clearing it changes how the tests
                    # run -- 002371's `addopts = -p no:hypothesispytest` with `filterwarnings = error` crashes
                    # pytest at start-up without it (EXP-018 baseline pilot)
                    opts = f'-o junit_family=xunit1 --junit-xml={rpath}'
                    if collect:
                        opts += ' --continue-on-collection-errors'
                    pre = f'PYTEST_ADDOPTS={shlex.quote(opts)} '
                elif report_kind == 'gotest':
                    pre = self._install_go_shim(sb, repo, rdir, rpath, res)
                pre += f'MIMO_REPORT={shlex.quote(rpath)} '           # harnesses the baseline tool amended honour this
            if collect:                                     # the test files as the hidden tests see them
                res['report_test_files'] = {p.decode('utf-8', 'surrogateescape'): self._read_blob(
                    sb, repo, None, p.decode('utf-8', 'surrogateescape')).decode('utf-8', 'replace')
                    for p in (test_paths or [])[:200] if p.endswith((b'.py', b'.go', b'.js', b'.ts', b'.tsx', b'.jsx'))}
            if grader == 'nobody':
                res['lock_seconds'] = self._lock_down(sb, repo, test_paths)
                test_cmd = as_grader(test_cmd)
            untrusted, t0 = True, time.time()               # the tests run the model's code
            r = sb.execute(pre + test_cmd, cwd=repo, timeout=gi['verifier_timeout'], login=False)
            out = {**res, 'reward': 1.0 if r['returncode'] == 0 else 0.0, 'verifier_returncode': r['returncode'],
                   'verifier_reason': r['reason'], 'test_seconds': round(time.time() - t0, 1),
                   'test_output': r['output'][-3000:], 'reverted_paths': reverted}
            if collect:
                out['report'] = self._read_abs(sb, rpath).decode('utf-8', 'replace') if want_report else ''
                return out
            if expected and report_kind in _REPORT_PARSERS:        # second check: the exact tests must be present & pass
                ok, why = report_verdict(report_kind, self._read_abs(sb, rpath),
                                         gi.get('fail_to_pass') or [], gi.get('pass_to_pass') or [])
                out['second_check'] = ok
                if not ok:
                    return {**out, 'reward': 0.0, 'grade_detail': 'second_check: ' + why}
            return out
        except SandboxLimitError as e:
            if not untrusted:                               # only our own setup had run: not the model's doing
                raise GradeError(f'grading container hit a limit during setup: {e}') from None
            return {**res, 'grade_detail': f'grader_sandbox_limit: {e}', 'reverted_paths': reverted,
                    **({'test_seconds': round(time.time() - t0, 1)} if t0 else {})}
        except SandboxCommandError as e:
            if not untrusted:                               # a broken grading image, before any model bytes: infra
                raise GradeError(f'grading container command would not start during setup: {e}') from None
            return {**res, 'grade_detail': f'grader_sandbox_command_failed: {e}', 'reverted_paths': reverted,
                    **({'test_seconds': round(time.time() - t0, 1)} if t0 else {})}
        except SetupError as e:
            raise GradeError(f'grading container setup failed: {e}') from None
        finally:
            sb.cleanup()
