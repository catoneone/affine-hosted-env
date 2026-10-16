"""SWE task type for the `code` split of Xiaomi MiMo-V2.6-RL-oss (the environment part of the EXP-014 harness,
aligned with mimoagent opensource_code.py).

- setup: truncate git history to the base commit (class-C images carry upstream remotes/origin/*; cleaned in place
  following the official strip implementation), restore environment variables lost from the image (in
  DockerSandbox), optional anti_hack_cleanup (enabled in mini-bash.yaml).
- Grading: capture the model's changes -> restore the paths touched by test_patch to the base commit -> git apply
  test_patch -> run test_command; exit code 0 scores 1; a patch that does not apply is an infra error; test files
  are restored both before and after grading.
"""
from __future__ import annotations

import base64
import re
import shlex
import time

from .base import DockerTaskEnv, EnvSession, GradeError, SetupError


def patch_touched_files(patch: str) -> list[str]:
    files = []
    for line in patch.split('\n'):
        m = re.match(r'^diff --git a/(.+?) b/(.+)$', line)
        if m:
            for fp in m.groups():
                if fp not in files:
                    files.append(fp)
    return files


def reset_test_files_cmd(test_patch: str, base: str) -> str | None:
    touched = patch_touched_files(test_patch)
    if not (touched and base):
        return None
    return ('while IFS= read -r tf; do\n'
            '  [ -z "$tf" ] && continue\n'
            f'  if git cat-file -e {base}:"$tf" 2>/dev/null; then\n'
            f'    git checkout {base} -- "$tf" 2>/dev/null || true\n'
            '  else\n'
            '    git rm -f --cached "$tf" >/dev/null 2>&1 || true\n'
            '    rm -f "$tf"\n'
            '  fi\n'
            "done <<'EOF_RESET_TEST_FILES'\n"
            + '\n'.join(touched) + '\nEOF_RESET_TEST_FILES\n')


_RESIDUE_SCRUB_GLOBAL = (
    "rm -f /tmp/fail.log /tmp/pass.log /tmp/patch.diff /tmp/test_patch.diff /tmp/*.log 2>/dev/null; "
    "rm -rf /tmp/claude-0 /tmp/claude-* /tmp/testem-* /tmp/puppeteer_dev_chrome_profile-* 2>/dev/null; "
    "rm -f /tmp/test_files.json 2>/dev/null; "
    "rm -rf /tmp/jest_* /tmp/jest-* /tmp/build /tmp/build_env 2>/dev/null; "
    "rm -rf /tmp/pytest-of-root /tmp/pytest-* /tmp/.pytest_cache /tmp/__pycache__ 2>/dev/null; "
    "rm -rf /tmp/go-build* /tmp/standards /tmp/wordpress /tmp/zig-* /tmp/testbase /tmp/testbed 2>/dev/null; "
    "rm -f /tmp/tmp*.tmp /tmp/*.bak /tmp/expect* /tmp/butwas* 2>/dev/null; "
    "rm -rf /root/.cache/go-build 2>/dev/null; "
    "rm -rf /tests /logs 2>/dev/null; "
    "find /var/log -type f -delete 2>/dev/null || true; "
    "rm -rf /var/lib/postgresql/*/*/log /var/lib/postgresql/*/*/pg_log 2>/dev/null || true; "
    "find /var/lib/mysql /var/lib/mongodb -type f -name '*.log' -delete 2>/dev/null || true; "
    "rm -rf /root/.npm/_logs /root/.babel.json /root/.pytest_cache 2>/dev/null || true; "
    "find / -maxdepth 4 -xdev -name task_description.md -path '*/.build_env/*' -delete 2>/dev/null || true; "
    "true"
)
_CLEAN_KEEP_COMMON = ['node_modules', 'bower_components', '.husky', 'vendor', 'third_party', '_deps', 'vcpkg_installed']
_CLEAN_KEEP = {'python': ['.venv', 'venv'], 'java': ['.gradle'], 'kotlin': ['.gradle'], 'scala': ['.gradle'],
               'rust': ['target'], 'swift': ['.build'], 'crystal': ['lib'], 'ruby': ['.bundle'],
               'julia': ['Manifest.toml']}
_CLEAN_KEEP_UNKNOWN = ['.venv', 'venv', '.gradle', 'target', '.build', 'lib', '.bundle', 'Manifest.toml', '_build',
                       '.stack-work', 'dist-newstyle']
_RUST_PURGE_BODY = r"""
NAMES=$(grep -hoE '^[[:space:]]*name[[:space:]]*=[[:space:]]*"[^"]+"' \
          $(find . -maxdepth 4 -name Cargo.toml 2>/dev/null) 2>/dev/null \
        | sed -E 's/.*"([^"]+)".*/\1/' | sort -u)
for n in $NAMES; do
  u=$(printf '%s' "$n" | tr '-' '_')
  for d in target/debug target/release; do
    [ -d "$d" ] || continue
    rm -f "$d/$n" "$d/lib$u.rlib" "$d/lib$u.so" "$d/lib$u.dylib" "$d/lib$u.a" 2>/dev/null || true
    rm -f "$d"/deps/lib${u}-*.rlib "$d"/deps/lib${u}-*.rmeta \
          "$d"/deps/lib${u}-*.so "$d"/deps/${u}-* "$d"/deps/${n}-* 2>/dev/null || true
  done
done
rm -rf target/*/incremental target/*/.fingerprint 2>/dev/null || true
true
"""
_GLOBAL_CACHE_SCRUB = (
    "find /root/.m2 -type f \\( -name '*-SNAPSHOT.jar' -o -name '*-SNAPSHOT-sources.jar' "
    "-o -name '*-SNAPSHOT-tests.jar' -o -name '*-SNAPSHOT-test-sources.jar' \\) -delete 2>/dev/null; "
    "rm -rf /root/.julia/compiled 2>/dev/null; "
    "rm -rf /root/.gradle/caches/build-cache-* /root/.gradle/daemon 2>/dev/null; "
    "rm -rf /root/.gradle/caches/*/scripts /root/.gradle/caches/jars-* 2>/dev/null; "
    "rm -rf /root/.cache/bazel 2>/dev/null; "
    "true"
)


def anti_hack_cleanup(sb, repo: str, lang: str):
    """Official _run_build_env_anti_hack_cleanup: all three stages are best effort and a failure does not block
    the rollout. Only run inside a container."""
    sb.execute(f'rm -rf {repo}/node_modules/.cache {repo}/node_modules/.vitest 2>/dev/null; ' + _RESIDUE_SCRUB_GLOBAL,
               timeout=120)
    keep = list(dict.fromkeys(_CLEAN_KEEP_COMMON + _CLEAN_KEEP.get(lang, _CLEAN_KEEP_UNKNOWN)))
    sb.execute('git clean -fdx ' + ' '.join(f'--exclude={d}' for d in keep), cwd=repo, timeout=300)
    if lang == 'rust':
        sb.execute(f'cd {repo}\n' + _RUST_PURGE_BODY, timeout=300)
    elif lang == 'swift':
        sb.execute(f'cd {repo} && rm -rf .build/*/build 2>/dev/null || true', timeout=120)
    if lang == 'python':
        sb.execute(f'pip install -e {repo} --no-deps --force-reinstall -q 2>/dev/null || true', timeout=300)
    sb.execute(_GLOBAL_CACHE_SCRUB, timeout=180)


def git_strip(sb, repo: str, base: str):
    """Official _strip_future_commits: detach HEAD at the base commit, delete all branches/remotes/other refs,
    delete tags that are not ancestors, expire the reflog, and force gc to physically delete unreachable objects."""
    cmds = [
        f'git checkout --detach {base} 2>/dev/null || true',
        'git for-each-ref --format="%(refname)" refs/heads | while read r; do git update-ref -d "$r" 2>/dev/null; done || true',
        'git remote | xargs -r -n1 git remote remove 2>/dev/null || true',
        'git for-each-ref --format="%(refname)" refs/replace refs/notes refs/stash refs/remotes refs/pull 2>/dev/null | while read r; do git update-ref -d "$r" 2>/dev/null; done || true',
        f'git for-each-ref --format="%(refname)" refs/tags | while read t; do if ! git merge-base --is-ancestor "$t" {base} 2>/dev/null; then git tag -d "${{t#refs/tags/}}" >/dev/null 2>&1; fi; done || true',
        'git reflog expire --expire=now --all 2>/dev/null || true',
        'git -c gc.pruneExpire=now -c gc.cruftPacks=false gc --prune=now 2>/dev/null || true',
    ]
    sb.execute('\n'.join(cmds), cwd=repo, timeout=600)


def future_commits(sb, repo: str, base: str) -> list[str]:
    r = sb.execute(f'git rev-list --all --not {base} | head -n 5', cwd=repo, timeout=120)
    return [x for x in r['output'].split() if len(x) == 40]


TEST_PATCH_REMOTE = '/tmp/_hosted_env_test.patch'
MODEL_PATCH_REMOTE = '/tmp/_hosted_env_model.patch'
RESET_LIST_REMOTE = '/tmp/_hosted_env_reset.list'
TREE_INDEX_REMOTE = '/tmp/_hosted_env_tree.index'
LOCK_LIST_REMOTE = '/tmp/_hosted_env_lock.list'
GRADER_UID = 65534                     # nobody: the hidden tests run without root (cfg grader_user = "root" to opt out)


def as_grader(command: str) -> str:
    """Run `command` as GRADER_UID with no supplementary groups and no profile or rc file. HOME stays the
    image's (its caches and toolchains: Go module cache, nvm, cargo, ...), which _lock_down hands to the grader."""
    return (f'setpriv --reuid={GRADER_UID} --regid={GRADER_UID} --clear-groups -- '
            f'env USER=nobody LOGNAME=nobody bash --noprofile --norc -c {shlex.quote(command)}')
MAX_MODEL_PATCH_BYTES = 2_000_000
GRADER_VERSION = 'isolated-v1'

# Test infrastructure the model's patch must not change: anything here that the patch touches is put back to the
# base commit in the grading container before the hidden tests run (the test_patch files are handled separately).
# These are files that decide how tests are collected, configured or reported rather than the code under test.
_PROTECTED_NAMES = {
    # python: pytest/tox/nox configuration, conftest plugins, interpreter start-up hooks
    'conftest.py', 'pytest.ini', '.pytest.ini', 'tox.ini', 'setup.cfg', 'pyproject.toml', 'noxfile.py',
    'sitecustomize.py', 'usercustomize.py',
    # go: module graph (replace directives) and workspace
    'go.mod', 'go.sum', 'go.work', 'go.work.sum',
    # js/ts: package scripts and test-runner / transpiler configuration
    'package.json', '.npmrc', '.babelrc', '.babelrc.json', '.babelrc.js',
    # generic: the grading entry points, build entry points, attributes that change how patches apply
    'mimo_test_command.sh', 'test_commands.json', 'Makefile', 'makefile', 'GNUmakefile', '.gitattributes',
    '.gitmodules',
}
_PROTECTED_RE = re.compile(
    r'(^|/)(test_[^/]*\.py|[^/]*_test\.py|[^/]*\.pth|[^/]*_test\.go'
    r'|[^/]*\.(test|spec)\.[cm]?[jt]sx?'
    r'|(jest|vitest|vite|ava|babel)\.(config|setup|workspace)\.[^/]*|setupTests\.[^/]*|karma\.conf\.[^/]*'
    r'|\.mocharc[^/]*|\.env[^/]*)$'
    r'|(^|/)(__tests__|__mocks__)/')


def protected_paths(files: list[str]) -> list[str]:
    return [f for f in files if f.rsplit('/', 1)[-1] in _PROTECTED_NAMES or _PROTECTED_RE.search(f)]


class SweEnv(DockerTaskEnv):
    """Config options: cleanup (anti-hack cleanup, on by default), git_leak ('strip' default / 'error').

    Grading never trusts the rollout container: the model's changes are captured as a git patch, that container is
    destroyed, and the hidden tests run in a fresh container from the same image with only the patch applied,
    the test infrastructure reset (`protected_paths`), and a non-login shell (no profile or rc file is read)."""

    kind = 'mimo_swe'

    def grade_inputs(self, task: str) -> dict:
        r = self._load()[task]
        return {**super().grade_inputs(task), 'test_patch': r['test_patch']}

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
        except SetupError:
            return b'', 'model_patch_capture_failed'      # e.g. the model deleted .git
        r = s.sb.execute(f'set -o pipefail; git diff --binary --no-color --no-ext-diff --no-textconv '
                         f'{s.state["start_tree"]} {end_tree} 2>/dev/null '
                         f'| head -c {MAX_MODEL_PATCH_BYTES + 1} | base64 | tr -d "\\n"',
                         cwd=repo, timeout=300, login=False)
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
        """Every path the patch touches (both sides of renames), as git itself parses it (`--numstat -z`), so
        quoted names, force-added ignored files and renames cannot hide a protected file."""
        r = sb.execute(f'set -o pipefail; git apply --numstat -z {path} | base64 | tr -d "\\n"', cwd=repo,
                       timeout=120, login=False)
        if r['returncode'] != 0:
            raise GradeError('cannot list model patch paths: ' + r['output'][-300:])
        items, out, i = base64.b64decode(r['output'].strip()).split(b'\0'), [], 0
        while i < len(items):
            fields = items[i].split(b'\t')
            if len(fields) == 3 and fields[2] == b'' and i + 2 < len(items):   # rename/copy: src and dst follow
                out += [items[i + 1], items[i + 2]]
                i += 3
                continue
            if len(fields) == 3 and fields[2]:
                out.append(fields[2])
            i += 1
        return out

    def _reset_paths(self, sb, repo: str, start: str, paths: list[bytes]) -> None:
        """Put each path back to the start tree, or delete it when the start tree does not have it (NUL-separated,
        so any name)."""
        if not paths:
            return
        self._put_bytes(sb, b'\0'.join(paths) + b'\0', RESET_LIST_REMOTE)
        r = sb.execute(f'while IFS= read -r -d "" f; do\n'
                       f'  if git cat-file -e {start}:"$f" 2>/dev/null; then git checkout {start} -- "$f" || exit 1\n'
                       '  else git rm -rf -q --cached --ignore-unmatch -- "$f" >/dev/null && rm -rf -- "$f"; fi\n'
                       f'done < {RESET_LIST_REMOTE}; rm -f {RESET_LIST_REMOTE}', cwd=repo, timeout=300, login=False)
        if r['returncode'] != 0:
            raise GradeError('reset_protected_failed: ' + r['output'][-300:])

    def _lock_down(self, sb, repo: str, test_patch: str) -> float:
        """Before the hidden tests run as GRADER_UID. The repo and the image's HOME become the grader's, so tests
        write caches and scratch files exactly where they would as root (pytest cache, rollup temp configs, the
        Go build cache, nvm/cargo toolchains). The test infrastructure stays root's: every file of the test patch
        is root-owned and read-only, and each directory holding one, up to the repo root, is root-owned with mode
        1777 (like /tmp), so the grader can create files there but cannot replace, rename or delete the tests or
        the grading script while they run. Returns the seconds it took. (EXP-018 parity runs: a read-only repo
        root broke pytest with -W error and rollup; a changed HOME hid the Go module cache.)"""
        t0 = time.time()
        files = [f.encode('utf-8', 'surrogateescape') for f in patch_touched_files(test_patch)]
        self._put_bytes(sb, b'\0'.join(files) + (b'\0' if files else b''), LOCK_LIST_REMOTE)
        r = sb.execute(f'''set -e
command -v setpriv >/dev/null || {{ echo "setpriv not found: cannot drop privileges"; exit 3; }}
chown -R {GRADER_UID}:{GRADER_UID} .
h="${{HOME:-/root}}"
if [ -d "$h" ] && [ "$h" != / ]; then chown -R {GRADER_UID}:{GRADER_UID} "$h"; fi
while IFS= read -r -d "" f; do
  [ -e "$f" ] || [ -L "$f" ] || continue
  chown -h 0:0 -- "$f"; [ -L "$f" ] || chmod a-w -- "$f"
  d=$(dirname -- "$f")
  while [ "$d" != "." ] && [ "$d" != "/" ]; do chown 0:0 -- "$d"; chmod 1777 -- "$d"; d=$(dirname -- "$d"); done
done < {LOCK_LIST_REMOTE}
rm -f {LOCK_LIST_REMOTE}
chown 0:0 .; chmod 1777 .
''', cwd=repo, timeout=900, login=False)
        if r['returncode'] != 0:
            raise GradeError('cannot prepare non-root grading: ' + r['output'][-300:])
        return round(time.time() - t0, 1)

    def grade(self, s: EnvSession, gi: dict) -> dict:
        repo = s.spec['cwd']
        patch, problem = self.capture_patch(s)
        res = {'model_patch': patch.decode('utf-8', 'replace'), 'reward': 0.0, 'grader': GRADER_VERSION}
        if problem:                                         # the model's own doing (deleted .git, huge diff): 0
            return {**res, 'grade_detail': problem}
        if s.factory is None:
            raise GradeError('no sandbox factory for an isolated grading container')
        s.sb.cleanup()                                      # the rollout container is never used again
        sb = s.grader_sb = s.factory(s.spec['image'], repo)
        try:
            sb.start()
            base = self._prepare_repo(sb, repo, s.spec.get('lang', ''))
            start = self._tree(sb, repo, base)
            if start != s.state['start_tree']:
                raise GradeError(f'grading container starts from tree {start[:12]}, the rollout started from '
                                 f'{s.state["start_tree"][:12]} (non-deterministic image or setup)')
            reverted = []
            if patch:
                self._put_bytes(sb, patch, MODEL_PATCH_REMOTE)
                touched = self._patch_paths(sb, repo, MODEL_PATCH_REMOTE)
                ap = sb.execute(f'git apply --binary --whitespace=nowarn {MODEL_PATCH_REMOTE}', cwd=repo,
                                timeout=300, login=False)
                sb.execute(f'rm -f {MODEL_PATCH_REMOTE}', timeout=60, login=False)
                if ap['returncode'] != 0:                   # same start tree, so only the patch can be at fault
                    return {**res, 'grade_detail': 'model_patch_does_not_apply: ' + ap['output'][-300:]}
                prot = protected_paths([p.decode('utf-8', 'surrogateescape') for p in touched])
                reverted = sorted(set(prot))
                self._reset_paths(sb, repo, start, [p.encode('utf-8', 'surrogateescape') for p in reverted])
            reset_cmd = reset_test_files_cmd(gi['test_patch'], base)
            if reset_cmd and sb.execute(reset_cmd, cwd=repo, timeout=300, login=False)['returncode'] != 0:
                raise GradeError('reset_tests_failed')
            sb.copy_text_to(gi['test_patch'], TEST_PATCH_REMOTE)
            ap = sb.execute(f'git apply --verbose {TEST_PATCH_REMOTE}', cwd=repo, timeout=300, login=False)
            sb.execute(f'rm -f {TEST_PATCH_REMOTE}', timeout=60, login=False)
            if ap['returncode'] != 0:
                raise GradeError('apply_test_patch_failed: ' + ap['output'][-500:])
            t0 = time.time()
            test_cmd, grader = gi['test_command'], self.cfg.get('grader_user', 'nobody')
            if grader != 'root':
                lock_s = self._lock_down(sb, repo, gi['test_patch'])
                test_cmd = as_grader(test_cmd)
                res['grader_user'], res['lock_seconds'] = GRADER_UID, lock_s
                t0 = time.time()
            r = sb.execute(test_cmd, cwd=repo, timeout=gi['verifier_timeout'], login=False)
            return {**res, 'reward': 1.0 if r['returncode'] == 0 else 0.0, 'verifier_returncode': r['returncode'],
                    'verifier_reason': r['reason'], 'test_seconds': round(time.time() - t0, 1),
                    'test_output': r['output'][-3000:], 'reverted_paths': reverted}
        except SetupError as e:
            raise GradeError(f'grading container setup failed: {e}') from None
        finally:
            sb.cleanup()
