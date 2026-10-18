"""Which files the model's patch may not change: put back to the base commit in the grading container
before the hidden tests run -- protected names and directories, structured manifests (only their
test-relevant fields), shadow modules, files a JS/TS test-runner config references."""
from __future__ import annotations

import json
import re
import sys



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
                   'node_modules', 'vendor', 'site-packages', '.venv',
                   # MiMo test infrastructure: the test patch's bundle unpacks to .build_env/, and pipeline C
                   # tasks keep their acceptance tests in usercase-test-coderl/ -- a file the model adds there
                   # beside ours (a helper the bundle's runner imports) is not reset by the test patch itself
                   '.build_env', 'usercase-test-coderl'}
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
