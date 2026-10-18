"""Rollout-container preparation against leaks and residue: anti_hack_cleanup (build output, caches,
residue of the reference solution) and the git history strip to the base commit."""
from __future__ import annotations




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
