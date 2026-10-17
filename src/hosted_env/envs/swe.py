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
import re
import shlex
import sys
import time

from ..providers.sandbox.base import SandboxCommandError, SandboxLimitError
from .base import DockerTaskEnv, EnvSession, GradeError, SetupError

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

# Test infrastructure the model's patch must not change: anything here that the patch touches is put back to the
# base commit in the grading container before the hidden tests run (the test_patch files are handled separately).
# These are files that decide how tests are collected, configured or reported rather than the code under test.
_PROTECTED_NAMES = {
    # python: pytest/tox/nox configuration, conftest plugins, interpreter start-up hooks
    'conftest.py', 'pytest.ini', '.pytest.ini', 'tox.ini', 'noxfile.py',
    'sitecustomize.py', 'usercustomize.py',
    # go: the workspace graph (go.mod is handled structurally; go.sum is never reverted -- it can only break a build)
    'go.work', 'go.work.sum',
    # js/ts: test-runner / transpiler configuration (package.json is handled structurally)
    '.npmrc', '.yarnrc', '.yarnrc.yml', '.babelrc', '.babelrc.json', '.babelrc.js',
    # build files that are code (a fix does not need to change how the project builds)
    'setup.py', 'CMakeLists.txt', 'build.gradle', 'build.gradle.kts',
    # generic: the grading entry points, build entry points, attributes that change how patches apply
    'mimo_test_command.sh', 'test_commands.json', 'Makefile', 'makefile', 'GNUmakefile', '.gitattributes',
    '.gitmodules',
}
# a path with any of these as a DIRECTORY component is test infrastructure or a vendored dependency tree, which a
# correct fix never needs to change; the whole subtree is reset to the base commit before the hidden tests run.
_PROTECTED_DIRS = {'tests', 'test', 'testdata', '__tests__', '__mocks__',
                   'node_modules', 'vendor', 'site-packages', '.venv'}
_PROTECTED_RE = re.compile(
    r'(^|/)(test_[^/]*\.py|[^/]*_test\.py|[^/]*\.pth|[^/]*_test\.go'
    r'|[^/]*\.(test|spec)\.[cm]?[jt]sx?'
    r'|(jest|vitest|vite|ava|babel)\.(config|setup|workspace)\.[^/]*|setupTests\.[^/]*|karma\.conf\.[^/]*'
    r'|\.mocharc[^/]*|\.env[^/]*|[^/]*\.cmake)$'
    r'|(^|/)(__tests__|__mocks__)/')


def protected_paths(files: list[str]) -> list[str]:
    out = []
    for f in files:
        name = f.rsplit('/', 1)[-1]
        if (name in _PROTECTED_NAMES or _PROTECTED_RE.search(f)
                or any(seg in _PROTECTED_DIRS for seg in f.split('/')[:-1])):
            out.append(f)
    return out


# Manifest files are NOT reverted wholesale (a correct fix often must bump a dependency or a build directive);
# only the fields that decide how tests are collected / run are protected. Both the base and the model's version
# are parsed in the executor (never in the model's container), the test-relevant projection compared, and the
# file reset only if that projection changed or either version fails to parse. Parsers are pinned so an offline
# re-grade reaches the same keep/revert verdict (GRADER_PARSERS, recorded with the grade).
STRUCTURED_MANIFESTS = {'pyproject.toml', 'setup.cfg', 'package.json', 'go.mod', 'Cargo.toml', 'pom.xml'}
GRADER_PARSERS = (f'py{sys.version_info.major}.{sys.version_info.minor}:tomllib+json+configparser'
                  ';gomod=v1;cargo=tomllib;pom=xml.etree-v1')
_EMPTY_SRC = {'pyproject.toml': b'', 'setup.cfg': b'', 'package.json': b'{}', 'go.mod': b'',
              'Cargo.toml': b'', 'pom.xml': b'<project/>'}


def _pyproject_test_keys(b: bytes):
    import tomllib
    d = tomllib.loads(b.decode('utf-8'))
    eps = ((d.get('project') or {}).get('entry-points') or {}).get('pytest11')
    return ((d.get('tool') or {}).get('pytest'), d.get('build-system'), eps)


def _setupcfg_test_keys(b: bytes):
    import configparser
    cp = configparser.ConfigParser()
    cp.read_string(b.decode('utf-8'))
    pytest = dict(cp['tool:pytest']) if cp.has_section('tool:pytest') else None
    eps = dict(cp['options.entry_points']) if cp.has_section('options.entry_points') else {}
    return (pytest, eps.get('pytest11'))


def _packagejson_test_keys(b: bytes):
    import json as _json
    d = _json.loads(b.decode('utf-8'))
    return {k: d.get(k) for k in ('scripts', 'jest', 'mocha', 'ava', 'nyc', 'c8', 'babel')}


def _gomod_test_keys(b: bytes):
    """`go`, `toolchain` and every `replace` directive (single-line and block form). Arbos gates these: a fix may
    add a dependency (require), but changing the toolchain or redirecting a module can change what runs."""
    go = toolchain = None
    replaces, in_replace = [], False
    for raw in b.decode('utf-8', 'replace').splitlines():
        line = raw.split('//', 1)[0].strip()
        if not line:
            continue
        if in_replace:
            if line == ')':
                in_replace = False
            else:
                replaces.append(line)
            continue
        if line.startswith('replace (') or line == 'replace (':
            in_replace = True
        elif line.startswith('replace '):
            replaces.append(line[len('replace '):].strip())
        elif line.startswith('go '):
            go = line[3:].strip()
        elif line.startswith('toolchain '):
            toolchain = line[len('toolchain '):].strip()
    return (go, toolchain, sorted(replaces))


def _cargo_test_keys(b: bytes):
    import tomllib
    d = tomllib.loads(b.decode('utf-8'))
    tests = d.get('test')
    harness = sorted((t.get('name'), t.get('harness')) for t in tests) if isinstance(tests, list) else None
    return (d.get('patch'), d.get('replace'), (d.get('package') or {}).get('build'), harness)


def _pom_test_keys(b: bytes):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(b)

    def strip(t):
        return t.rsplit('}', 1)[-1]
    plugins, flags = [], []
    for el in root.iter():
        if strip(el.tag) == 'plugin':
            art = next((c.text for c in el if strip(c.tag) == 'artifactId'), '') or ''
            if 'surefire' in art or 'failsafe' in art:
                cfg = next((c for c in el if strip(c.tag) == 'configuration'), None)
                plugins.append(ET.tostring(cfg, encoding='unicode') if cfg is not None else '')
        if strip(el.tag) in ('skipTests', 'testFailureIgnore', 'maven.test.skip', 'skip'):
            flags.append((strip(el.tag), (el.text or '').strip()))
    return (sorted(plugins), sorted(flags))


_MANIFEST_KEYS = {'pyproject.toml': _pyproject_test_keys, 'setup.cfg': _setupcfg_test_keys,
                  'package.json': _packagejson_test_keys, 'go.mod': _gomod_test_keys,
                  'Cargo.toml': _cargo_test_keys, 'pom.xml': _pom_test_keys}


def manifest_reverts(name: str, base: bytes, cur: bytes) -> bool:
    """True if a structured manifest must be reset to base: a test-relevant key changed, or either version does
    not parse (fail closed -- a model cannot smuggle a config change through a file that only one side can read)."""
    fn = _MANIFEST_KEYS[name]
    try:
        bk = fn(base if base.strip() else _EMPTY_SRC[name])
        ck = fn(cur)
    except Exception:  # noqa: BLE001  unparseable on either side: revert
        return True
    return bk != ck


_ALWAYS_SHADOW = {'pytest', '_pytest', 'py', 'pluggy'}   # internals of the test runner itself, by name


def shadow_added(added: list[str], importable: set[str]) -> list[str]:
    """Model-ADDED paths that would shadow an importable top-level module. `python -m pytest` runs with the repo on
    sys.path, so a new root file `pytest.py` or a new root directory `_pytest/` is imported before the real one
    (verified in EXP-018). The shadow is the top-level component: a root `*.py` file by its stem, or the first
    directory of a deeper path. A name is a shadow if it is importable on the image, is a pytest internal, or
    matches `pytest_*`. Files inside an existing package import as a submodule, not a top-level shadow, and are
    left to the name / structured rules."""
    out = []
    for p in added:
        segs = p.split('/')
        if len(segs) == 1:
            if not segs[0].endswith('.py'):
                continue
            mod = segs[0][:-3]
        else:
            mod = segs[0]                                 # a new top-level directory == an importable package name
        if not mod or '.' in mod:
            continue
        if mod in importable or mod in _ALWAYS_SHADOW or mod.startswith('pytest_'):
            out.append(p)
    return out


_CMD_PATH_RE = re.compile(r'PATH=([^\s;"\']+)')
_CMD_INVOKE_RE = re.compile(r'(?:^|[\s;&|(])(?:exec\s+)?(?:bash|sh)\s+(\S+\.sh)'
                            r'|(?:^|[\s;&|(])(\./[^\s;&|)]+)')


def _repo_rel(p: str, cwd: str) -> str | None:
    """`p` as a path relative to the repo `cwd`, or None when it is outside the repo. Relative paths are taken as
    repo-relative; an absolute path is kept only if it is under cwd."""
    p = p.strip().strip('"\'')
    if not p or p in ('.', './'):
        return None
    if p.startswith('/'):
        pre = cwd.rstrip('/') + '/'
        if not p.startswith(pre):
            return None
        p = p[len(pre):]
    elif p.startswith('./'):
        p = p[2:]
    if not p or p.startswith('../') or '/../' in p:
        return None
    p = p.rstrip('/')
    if any(seg in ('.', '..', '...') for seg in p.split('/')):   # e.g. go's ./... wildcard is not a path
        return None
    return p


def command_repo_paths(test_command: str, cwd: str) -> list[str]:
    """Repo-relative directories put on PATH and scripts invoked directly by the test command: anything the model
    adds or changes under them decides what the grader runs, so they are protected like the test infrastructure.
    Absolute paths outside the repo are ignored. Heuristic over the command string (the inner script of a mimo
    task lives in the test patch and is reset separately)."""
    out = set()
    for m in _CMD_PATH_RE.findall(test_command):
        for part in m.split(':'):
            rel = _repo_rel(part, cwd)
            if rel is not None and not rel.startswith('$'):
                out.add(rel)
    for sh, dot in _CMD_INVOKE_RE.findall(test_command):
        rel = _repo_rel(sh or dot, cwd)
        if rel is not None and '$' not in rel:
            out.add(rel)
    return sorted(out)


# test-runner config keys whose values name files the runner loads before / around the tests; a model that edits
# one of those files changes what runs even when the config itself is unchanged, so the referenced files are reset.
_JS_REF_KEYS = ('setupFiles', 'setupFilesAfterEnv', 'setupFilesAfterEach', 'globalSetup', 'globalTeardown',
                'testEnvironment', 'reporters', 'transform', 'moduleNameMapper', 'setupFiles')
_JSON_CONFIGS = ('.json',)                       # jest.config.json, .mocharc.json, tsconfig*.json, ...
_JS_REF_RE = re.compile(r'''['"]((?:\./|\.\./|<rootDir>/)[^'"]+)['"]''')


def _strip_jsonc(text: str) -> str:
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r'(^|[^:])//[^\n]*', r'\1', text)       # // comments (tsconfig is JSONC), keep ://  in strings-ish
    return re.sub(r',(\s*[}\]])', r'\1', text)            # trailing commas


def _resolve_ref(ref: str, config_dir: str) -> str | None:
    if not ref.startswith(('./', '../', '<rootDir>')):   # a bare specifier (a package, 'node', 'default') is not a repo file
        return None
    ref = ref.replace('<rootDir>/', '').replace('<rootDir>', '')
    if not ref or ref.rsplit('/', 1)[-1].count('*'):      # a glob pattern, not one file
        return None
    joined = f'{config_dir}/{ref}' if config_dir and not ref.startswith('/') else ref
    parts: list[str] = []
    for seg in joined.split('/'):
        if seg in ('', '.'):
            continue
        if seg == '..':
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(seg)
    rel = '/'.join(parts)
    return rel if rel and 'node_modules' not in parts else None


def _collect_strings(v, out: set):
    if isinstance(v, str):
        out.add(v)
    elif isinstance(v, list):
        for x in v:
            _collect_strings(x, out)
    elif isinstance(v, dict):
        for x in v.values():
            _collect_strings(x, out)


def jsts_config_refs(name: str, content: bytes, config_dir: str) -> list[str]:
    """Repo-relative files a jest / vitest / mocha / babel config, or a tsconfig `paths` map, points at. These are
    reset if the model touched them, because an unchanged config can still load an editable setup/transform file
    (EXP-018 review). JSON configs are parsed; .js/.ts/.cjs/.mjs/.yml configs are scanned for quoted ./ , ../ and
    <rootDir>/ paths (best effort -- a config that computes a path at runtime is not caught, and the after-the-fact
    patch audit backs this up)."""
    raws: set = set()
    text = content.decode('utf-8', 'replace')
    is_json = name.endswith('.json') or name == 'package.json'
    if is_json:
        try:
            d = json.loads(_strip_jsonc(text))
        except ValueError:
            d = None
        if isinstance(d, dict):
            if name.startswith('tsconfig'):
                _collect_strings((d.get('compilerOptions') or {}).get('paths'), raws)
            else:
                cfg = d.get('jest') if name == 'package.json' else d
                for k in _JS_REF_KEYS:
                    _collect_strings((cfg or {}).get(k), raws)
    else:
        raws.update(_JS_REF_RE.findall(text))
    out = set()
    for r in raws:
        rel = _resolve_ref(r, config_dir)
        if rel is not None:
            out.add(rel)
    return sorted(out)


_JS_CONFIG_NAMES = ('package.json', 'jest.config.json', '.mocharc.json', 'vitest.config.json')
_JS_CONFIG_RE = re.compile(r'^(jest\.config\.|vitest\.config\.|vite\.config\.|\.mocharc|babel\.config\.|'
                           r'tsconfig[^/]*\.json$)')


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
                if cmd_dirs:                       # files under a PATH dir / a script the command runs directly
                    reverted.update(f for f in (p.decode('utf-8', 'surrogateescape') for p in touched)
                                    if any(f == d or f.startswith(d + '/') for d in cmd_dirs))
                touched_set = {p.decode('utf-8', 'surrogateescape') for p in touched}
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
            if grader == 'nobody':
                res['lock_seconds'] = self._lock_down(sb, repo, test_paths)
                test_cmd = as_grader(test_cmd)
            untrusted, t0 = True, time.time()               # the tests run the model's code
            r = sb.execute(test_cmd, cwd=repo, timeout=gi['verifier_timeout'], login=False)
            return {**res, 'reward': 1.0 if r['returncode'] == 0 else 0.0, 'verifier_returncode': r['returncode'],
                    'verifier_reason': r['reason'], 'test_seconds': round(time.time() - t0, 1),
                    'test_output': r['output'][-3000:], 'reverted_paths': reverted}
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
