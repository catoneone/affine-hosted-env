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

import re
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


class SweEnv(DockerTaskEnv):
    """Config options: cleanup (anti-hack cleanup, on by default), git_leak ('strip' default / 'error')."""

    kind = 'mimo_swe'

    def grade_inputs(self, task: str) -> dict:
        r = self._load()[task]
        return {**super().grade_inputs(task), 'test_patch': r['test_patch']}

    def setup(self, s: EnvSession) -> None:
        sb, repo = s.sb, s.spec['cwd']
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
        stripped = False
        if future_commits(sb, repo, base):
            if self.cfg.get('git_leak', 'strip') != 'strip':
                raise SetupError('image git history is not truncated at the base commit')
            git_strip(sb, repo, base)
            stripped = True
            if future_commits(sb, repo, base):
                raise SetupError('future commits still visible after git strip')
        if self.cfg.get('cleanup', True):
            anti_hack_cleanup(sb, repo, s.spec.get('lang', ''))
        s.state.update(base=base, git_stripped=stripped)

    def grade(self, s: EnvSession, gi: dict) -> dict:
        sb, repo, base = s.sb, s.spec['cwd'], s.state['base']
        diff = sb.execute('git add -A >/dev/null 2>&1 && git -c core.fileMode=false diff HEAD 2>/dev/null', cwd=repo,
                          timeout=120)
        res = {'model_patch': diff['output'] if diff['returncode'] == 0 else '', 'reward': 0.0}
        reset_cmd = reset_test_files_cmd(gi['test_patch'], base)
        try:
            if reset_cmd and sb.execute(reset_cmd, cwd=repo, timeout=300)['returncode'] != 0:
                raise GradeError('reset_tests_failed')
            sb.copy_text_to(gi['test_patch'], TEST_PATCH_REMOTE)
            ap = sb.execute(f'git apply --verbose {TEST_PATCH_REMOTE}', cwd=repo, timeout=300)
            sb.execute(f'rm -f {TEST_PATCH_REMOTE}', timeout=60)
            if ap['returncode'] != 0:
                raise GradeError('apply_test_patch_failed: ' + ap['output'][-500:])
            t0 = time.time()
            r = sb.execute(gi['test_command'], cwd=repo, timeout=gi['verifier_timeout'])
            return {**res, 'reward': 1.0 if r['returncode'] == 0 else 0.0, 'verifier_returncode': r['returncode'],
                    'verifier_reason': r['reason'], 'test_seconds': round(time.time() - t0, 1),
                    'test_output': r['output'][-3000:]}
        finally:
            if reset_cmd:
                sb.execute(reset_cmd, cwd=repo, timeout=300)
