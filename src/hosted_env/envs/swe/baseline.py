"""Expected tests per task (FAIL_TO_PASS / PASS_TO_PASS) from a baseline run, for the grader's second check.

The MiMo SWE data has a test patch and a test command per task but no list of the tests a fix must make pass,
and no gold patch. The grading scripts already name the tests they select (pytest node ids, `-k`, files), so
the expected set is "the tests the script runs": an EMPTY rollout -- base commit + hidden test patch, no model
change -- is graded by the real grader (`SweEnv.grade`, same container preparation, same non-root run, same
report injection) in collect mode, and its structured report is read back:

- a selected test that FAILS at base is FAIL_TO_PASS (the fix must make it pass);
- one that PASSES at base is PASS_TO_PASS (the fix must keep it passing);
- a skip is neither (a test that also skips after a correct fix could never be required to pass);
- a test file that does not even collect at base (it imports what the fix adds) has its test functions read
  from the source and filtered by the script's own selection (node ids, `-k`), all FAIL_TO_PASS;
- a test whose status differs between runs is flaky and left out.

Two runners so far. pytest (`report_kind = "junit"`): the grader injects a JUnit report through
PYTEST_ADDOPTS. go test (`report_kind = "gotest"`): the grader's `go` shim records `go test -json` events; a
package whose new test file does not build at base gets its `func TestX` from the test patch, selected by the
-run of the go test commands that run that package. jest / mocha / vitest need the grader to produce their
reports first; a task with no supported runner gets no expected tests and stays exit-code graded.

Without a gold patch this cannot prove that every FAIL_TO_PASS test is fixable; a task no real model ever
solves is dropped later.
"""
from __future__ import annotations

import ast
import json
import re
import shlex
import time

from ...infra.receipt import task_hash
from .env import GRADER_VERSION
from .reports import _junit_results
from .scripts import (_ASSIGN_RE, _KEYWORDS, _SCRIPT_DIR_RE, _WRAPPERS, MIMO_BUNDLE, _repo_rel, _shell_commands,
                      test_patch_scripts)

BASELINE_VERSION = 'baseline-v1'
_PYTEST_RE = re.compile(r'(^|[\s/;&|(])(py\.test|pytest)(\s|$)')


def junit_cases(report: bytes) -> tuple[dict[str, str], list[str]]:
    """({node id -> 'pass'|'fail'|'skip'}, [files that failed to collect]) from a pytest JUnit report written with
    junit_family=xunit1 (the `file` attribute makes the id unambiguous: `file::Class::name`). Paths are relative
    to pytest's rootdir, which is what the grader's report parser keys on too."""
    import xml.etree.ElementTree as ET
    cases: dict[str, str] = {}
    broken: list[str] = []
    for tc in ET.fromstring(report).iter('testcase'):
        name, cls, fil = tc.get('name', ''), tc.get('classname', ''), tc.get('file', '')
        kids = {c.tag: c for c in tc}
        if not cls and fil and 'error' in kids and 'collection' in (kids['error'].get('message') or ''):
            broken.append(fil)
            continue
        if not fil.endswith('.py') or not name:
            continue
        mod = fil[:-3].replace('/', '.')
        rest = cls[len(mod) + 1:] if cls.startswith(mod + '.') else ''
        nid = '::'.join([fil, *([p for p in rest.split('.') if p]), name])
        st = 'fail' if {'failure', 'error'} & set(kids) else ('skip' if 'skipped' in kids else 'pass')
        prev = cases.get(nid)
        cases[nid] = st if prev is None else ('fail' if 'fail' in (prev, st) else 'skip' if 'skip' in (prev, st)
                                              else 'pass')
    return cases, broken


def pytest_selection(scripts: list[str], cwd: str) -> tuple[list[str], list[str]]:
    """(node ids, -k expressions) the grading scripts pass to pytest; node ids repo-relative."""
    nodeids, kexprs = [], []
    for text in scripts:
        for line in re.sub(r'\\\r?\n', ' ', text).splitlines():
            if not _PYTEST_RE.search(line):
                continue
            try:
                commands, _ = _shell_commands(line.strip())
            except ValueError:
                continue
            for _, toks in commands:
                if not any(t.rsplit('/', 1)[-1] in ('pytest', 'py.test') for t in toks):
                    continue
                args = iter(toks)
                for t in args:
                    if t == '-k':
                        kexprs.append(next(args, ''))
                    elif t.startswith('-k') and len(t) > 2:
                        kexprs.append(t[2:].lstrip('='))
                    elif '::' in t and not t.startswith('-'):
                        p, _, rest = t.partition('::')
                        rel = _repo_rel(p, cwd) if p.startswith('/') else p.removeprefix('./')
                        if rel:
                            nodeids.append(f'{rel}::{rest}')
    return nodeids, [k for k in kexprs if k]


def static_test_ids(path: str, source: str) -> list[str]:
    """Test ids pytest would collect from `source` by its default rules (functions `test*`, classes `Test*` and
    their `test*` methods), as `path::name` / `path::Class::name`. Parameters are not expanded: the grader's
    report check accepts every `name[...]` instance of an id without brackets."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test'):
            out.append(f'{path}::{node.name}')
        elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith('test'):
                    out.append(f'{path}::{node.name}::{m.name}')
    return out


def _k_match(expr: str, nid: str) -> bool:
    """pytest's -k semantics for one test id: each name in the expression is a case-insensitive substring of one
    of the item's names (path components, file, class, function). Uses pytest's own expression parser."""
    from _pytest.mark.expression import Expression
    names = [p for p in re.split(r'::|/', nid) if p]
    names += [n[:-3] for n in names if n.endswith('.py')]

    def matcher(sub: str, **_kw) -> bool:
        return any(sub.lower() in n.lower() for n in names)
    try:
        return bool(Expression.compile(expr).evaluate(matcher))
    except Exception:  # noqa: BLE001  an expression we cannot evaluate selects nothing (the id is not required)
        return False


def select(ids: list[str], nodeids: list[str], kexprs: list[str]) -> list[str]:
    """The ids the script's own selection picks, for tests we could only read from source. With node ids, an id
    is kept if a node id names it (or its file, or a parametrised instance of it); else with -k, if any -k
    expression matches; else all of them (the script runs the whole file)."""
    if nodeids:
        keep = []
        for i in ids:
            f = i.split('::', 1)[0]
            for n in nodeids:
                nb = n.split('[', 1)[0]
                if (nb == i or n == f or i.endswith('/' + nb) or nb.endswith('/' + i)
                        or (nb.split('::', 1)[0] == f and '::' not in nb)):
                    keep.append(i)
                    break
        return keep
    if kexprs:
        return [i for i in ids if any(_k_match(k, i) for k in kexprs)]
    return list(ids)


_GOTEST_RE = re.compile(r'(^|[\s/;&|(])go\s+test(\s|$)')
_GO_TESTFUNC_RE = re.compile(r'^func\s+(Test[A-Z0-9_]\w*|Test)\s*\(\s*\w+\s+\*testing\.T\s*\)', re.M)


def gotest_cases(report: bytes) -> tuple[dict[str, str], set[str]]:
    """({'<package>::<TestName>': 'pass'|'fail'|'skip'}, packages that produced no test result) from `go test
    -json` events. Only top-level tests count (a failing subtest fails its parent, and generated subtest names
    are not stable ids). A test's last pass/fail/skip event wins within one run, since `go test -count=N`
    repeats it -- except that a fail is kept, like the grader's parser. A package with a package-level fail and
    no test event did not build (typically a new test file that uses code only the fix adds)."""
    cases: dict[str, str] = {}
    pkg_fail: set[str] = set()
    pkg_tests: set[str] = set()
    for line in report.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        pkg, test, act = ev.get('Package') or ev.get('ImportPath') or '', ev.get('Test') or '', ev.get('Action')
        if act not in ('pass', 'fail', 'skip'):
            continue
        if not test:
            if act == 'fail' and pkg:
                pkg_fail.add(pkg)
            continue
        pkg_tests.add(pkg)
        if '/' in test:
            continue
        key = f'{pkg}::{test}'
        if cases.get(key) != 'fail':
            cases[key] = act
    return cases, pkg_fail - pkg_tests


def _command_words(words: list[str]) -> list[str]:
    """A shell command's words from the program on: the prefixes bash runs the program through are dropped the
    way script_named_files drops them -- compound keywords (if / then / do / ! / {), `export`, VAR=value
    assignments and wrappers with their options (exec, env, timeout 600, time, nohup ...). Without this an
    `exec go test ./x/...` (EXP-018 pilot 001712) is not seen as a go test command, so a package that does not
    build at base gets no static ids and the task derives no expected tests."""
    while words and words[0] in _KEYWORDS:
        words = words[1:]
    if words[:1] == ['export']:
        words = words[1:]
    while words and _ASSIGN_RE.match(words[0]):
        words = words[1:]
    while words and words[0] in _WRAPPERS:
        words = words[1:]
        while words and (words[0].startswith('-') or _ASSIGN_RE.match(words[0])
                         or re.fullmatch(r'\d+[smhd]?', words[0])):
            words = words[1:]
    return words


def go_test_commands(scripts: list[tuple[str, str]], cwd: str) -> list[tuple[list[tuple[str, str]], str | None]]:
    """Every `go test` command in the task's scripts, given as (repo-relative script path or '' for the test
    command itself, text), as (package specs, top-level -run pattern or None). A spec is ('tree', dir) for
    `./dir/...`, ('dir', dir) for `./dir` or `.`, ('import', path) for an import path, and ('unknown', '') when
    the directory the command runs in could not be followed (a `cd` to a variable). Dirs are repo-relative ('' is
    the repo root); `cd "$(dirname "$0")"` goes to the script's own directory. Continuation lines are joined
    first. With no package argument go test runs the package in the current directory."""
    out: list[tuple[list[tuple[str, str]], str | None]] = []
    root = cwd.rstrip('/')
    for spath, text in scripts:
        here: str | None = ''
        sdir = spath.rsplit('/', 1)[0] if '/' in spath else ''
        text = _SCRIPT_DIR_RE.sub(root + ('/' + sdir if sdir else ''), text)   # as script_named_files
        for line in re.sub(r'\\\r?\n', ' ', text).splitlines():
            try:
                commands, _ = _shell_commands(line.strip())
            except ValueError:
                continue
            for _, words in commands:
                words = _command_words(words)
                if not words:
                    continue
                if words[0] == 'cd':
                    here = _go_cd(here, words[1] if len(words) > 1 else '', root)
                    continue
                if len(words) < 2 or words[0].rsplit('/', 1)[-1] != 'go' or words[1] != 'test':
                    continue
                run, pkgs, k = None, [], 2
                while k < len(words):
                    w = words[k]
                    if w == '-args':
                        break
                    if w in ('-run', '--run') and k + 1 < len(words):
                        run, k = _top_level(words[k + 1]), k + 2
                        continue
                    if w.startswith(('-run=', '--run=')):
                        run = _top_level(w.split('=', 1)[1])
                    elif w in _GO_VALUE_FLAGS and k + 1 < len(words):
                        k += 1
                    elif not w.startswith('-') and not w.endswith('.go'):
                        pkgs.append(w)
                    k += 1
                specs = [_go_pkg_spec(p, here) for p in (pkgs or ['.'])]
                out.append((specs, run))
    return out


# go test / build flags that take a separate value (so the value is not read as a package)
_GO_VALUE_FLAGS = {'-timeout', '-count', '-tags', '-p', '-parallel', '-cpu', '-o', '-mod', '-coverprofile',
                   '-covermode', '-coverpkg', '-bench', '-benchtime', '-skip', '-shuffle', '-exec', '-ldflags',
                   '-gcflags', '-asmflags', '-modfile', '-outputdir', '-vet', '-list'}


def _go_cd(here: str | None, target: str, root: str) -> str | None:
    if here is None:
        return None
    t = target.strip('"\'')
    if not t or '$' in t or '`' in t:
        return None if t else ''
    if t.startswith('/'):
        if t.rstrip('/') == root:
            return ''
        return t[len(root) + 1:].rstrip('/') if t.startswith(root + '/') else None
    parts = [p for p in (here.split('/') if here else []) if p]
    for seg in t.split('/'):
        if seg in ('', '.'):
            continue
        if seg == '..':
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(seg)
    return '/'.join(parts)


def _go_pkg_spec(arg: str, here: str | None) -> tuple[str, str]:
    if here is None:
        return ('unknown', '')
    if arg == '.' or arg.startswith('./') or arg.startswith('../'):
        tree = arg.endswith('/...')
        rel = _go_cd(here, arg[:-4] if tree else arg, '/\0')   # resolve relative to `here` only
        if rel is None:
            return ('unknown', '')
        return ('tree' if tree else 'dir', rel)
    if arg == '...':
        return ('tree', here)
    return ('import', arg[:-4] if arg.endswith('/...') else arg)


def go_patterns_for_dir(commands: list[tuple[list[tuple[str, str]], str | None]], d: str) -> list[str] | None:
    """The -run patterns of the go test commands that cover repo dir `d`: [] when none does or when a command
    with an unresolvable directory might (so no static id is added -- fewer expected tests, never ones the
    script does not run), None when a covering command has no -run (every test in the package runs)."""
    pats: list[str] = []
    for specs, run in commands:
        covers = False
        for kind, v in specs:
            if kind == 'unknown':
                return []
            if (kind == 'dir' and v == d) or (kind == 'tree' and (v == '' or d == v or d.startswith(v + '/'))) \
                    or (kind == 'import' and d and (v == d or v.endswith('/' + d))):
                covers = True
        if covers:
            if run is None:
                return None
            pats.append(run)
    return pats


def go_run_patterns(scripts: list[str]) -> list[str] | None:
    """The top-level part of every `go test -run` pattern in the task's scripts; None when some `go test` runs
    without -run (then every test of the selected packages is expected). Backslash-continued lines are joined
    first: a `-run` on the line after `go test \\` must not read as "no -run", which would expect tests the
    script never runs and score an honest fix 0."""
    pats: list[str] = []
    for text in scripts:
        for line in re.sub(r'\\\r?\n', ' ', text).splitlines():
            if not _GOTEST_RE.search(line):
                continue
            try:
                toks = shlex.split(line, comments=True)
            except ValueError:
                toks = line.split()
            run = None
            for k, t in enumerate(toks):
                if t in ('-run', '--run') and k + 1 < len(toks):
                    run = toks[k + 1]
                elif t.startswith(('-run=', '--run=')):
                    run = t.split('=', 1)[1]
            if run is None:
                return None
            pats.append(_top_level(run))
    return pats


def _top_level(pattern: str) -> str:
    """`go test -run 'A|B/sub'` matches top-level names against the part before the first '/' outside brackets
    and parentheses (cmd/go splits the pattern by unbracketed slashes, one element per subtest level)."""
    depth = 0
    for i, c in enumerate(pattern):
        if c in '([':
            depth += 1
        elif c in ')]':
            depth = max(0, depth - 1)
        elif c == '/' and depth == 0:
            return pattern[:i]
    return pattern


def static_go_test_ids(source: str) -> list[str]:
    """Top-level test function names declared in a Go test file."""
    return sorted(set(_GO_TESTFUNC_RE.findall(source)))


def go_select(names: list[str], patterns: list[str] | None) -> list[str]:
    """Names any -run pattern matches (Go regexps: RE2 syntax, unanchored). A pattern Python cannot compile
    selects nothing, so an id we cannot vouch for is never expected. None: no -run, every name is selected."""
    if patterns is None:
        return sorted(names)
    out = set()
    for p in patterns:
        try:
            rx = re.compile(p)
        except re.error:
            continue
        out.update(n for n in names if rx.search(n))
    return sorted(out)


_JS_RUNNER_RE = re.compile(r'(?:npx\s+|pnpm\s+exec\s+|yarn\s+(?:exec\s+)?|(?:\./)?node_modules/\.bin/|(?<=[\s;&|(])|^)'
                           r'(jest|vitest|mocha)(?=\s|$)')
_JS_HAS_REPORTER_RE = re.compile(r'--json\b|--reporters?[ =]|--outputFile|--reporter-options?\b')
# what each runner is given so its results land in $MIMO_REPORT (the grader sets it; /dev/null otherwise), and the
# report kind the grader's second check then parses. mocha's xunit and vitest's junit reporters write JUnit XML.
_JS_REPORTER_FLAGS = {
    'jest': ('--json --outputFile="${MIMO_REPORT:-/dev/null}"', 'jest'),
    'vitest': ('--reporter=junit --outputFile="${MIMO_REPORT:-/dev/null}"', 'junit'),
    'mocha': ('--reporter xunit --reporter-options output="${MIMO_REPORT:-/dev/null}"', 'junit'),
}


def amend_js_reporter(text: str) -> tuple[str, str] | None:
    """Rewrite a grading script so its one jest / vitest / mocha run also writes a structured report: the
    reporter flags are appended to that command, before any pipe, list operator, redirection or closing paren.
    Returns (new text, report kind), or None when the script is not one we can amend safely -- no run, more than
    one (they would overwrite one report file), more than one runner kind, a reporter flag already given, the
    run inside a quoted string (`bash -c '...'`) or continued on the next line. The baseline then gives the task
    no expected tests and it stays exit-code graded. The amended script is part of the signed task index, so
    the grader and an offline re-grader run the same one."""
    hits = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.lstrip().startswith('#'):
            continue
        for m in _JS_RUNNER_RE.finditer(line):
            hits.append((i, m))
    if len(hits) != 1 or _JS_HAS_REPORTER_RE.search(text):
        return None
    i, m = hits[0]
    line = lines[i]
    if line.rstrip().endswith('\\'):
        return None
    quote, k = None, 0
    while k < m.start():                      # quote state at the runner: refuse a run inside a string
        c = line[k]
        if quote:
            if c == quote:
                quote = None
            elif c == '\\' and quote == '"':
                k += 1
        elif c in '\'"':
            quote = c
        elif c == '\\':
            k += 1
        k += 1
    if quote:
        return None
    end, k = len(line), m.end()
    while k < len(line):                      # end of the runner's simple command
        c = line[k]
        if quote:
            if c == quote:
                quote = None
            elif c == '\\' and quote == '"':
                k += 1
        elif c in '\'"':
            quote = c
        elif c == '\\':
            k += 1
        elif c in '|&;<>)#' or (c.isdigit() and line[k + 1:k + 2] == '>' and line[k - 1:k].isspace()):
            if c == '#' and not line[k - 1:k].isspace():
                k += 1
                continue
            end = k
            break
        k += 1
    if quote:
        return None
    flags, kind = _JS_REPORTER_FLAGS[m.group(1)]
    head, rest = line[:end].rstrip(), line[end:].lstrip()
    lines[i] = f'{head} {flags}' + ('' if not rest else rest if rest.startswith(')') else ' ' + rest)
    return '\n'.join(lines), kind


def _js_runs(text: str) -> int:
    return sum(len(_JS_RUNNER_RE.findall(line)) for line in text.split('\n') if not line.lstrip().startswith('#'))


def amend_test_patch(test_patch: str, test_command: str) -> dict | None:
    """Task-level amend_js_reporter: {'test_patch' or 'test_command': amended text, 'report_kind': kind}, or None.
    The one jest / vitest / mocha run must be in the outer test command or in a `.sh` file the test patch ADDS
    (a new-file diff, rebuilt here with its hunk length and git blob id recomputed so `git apply` takes it).
    A run in a script the patch modifies, or inside the mimo_build_env bundle, is not amended (repacking the
    tarball is not worth it for the share of tasks it covers) and neither is a task whose runs are spread over
    several places: such a task gets no expected tests."""
    import hashlib
    sections = re.split(r'(?m)^(?=diff --git )', test_patch)
    cands = []                                    # (index or 'command', text)
    if _js_runs(test_command):
        cands.append(('command', test_command))
    for k, sec in enumerate(sections):
        head = sec.split('\n', 1)[0]
        if not head.startswith('diff --git '):
            continue
        path = head.rsplit(' b/', 1)[-1]
        added = [ln[1:] for ln in sec.split('\n') if ln.startswith('+') and not ln.startswith('+++ ')]
        body = '\n'.join(added)
        if path.rsplit('/', 1)[-1] == MIMO_BUNDLE:
            if any(_js_runs(t) for p, t in test_patch_scripts(sec) if p != path):
                return None
            continue
        if path.endswith('.sh') and _js_runs(body):
            if '\n--- /dev/null\n' not in sec or len(re.findall(r'(?m)^@@ ', sec)) != 1:
                return None
            cands.append((k, body))
    if len(cands) != 1:
        return None
    where, text = cands[0]
    if where == 'command':
        res = amend_js_reporter(text)
        return None if res is None else {'test_command': res[0], 'report_kind': res[1]}
    sec = sections[where]
    no_eol = '\n\\ No newline at end of file' in sec
    res = amend_js_reporter(text)
    if res is None:
        return None
    new, kind = res
    content = new + ('' if no_eol else '\n')
    hdr = sec.split('\n@@ ', 1)[0]
    blob = hashlib.sha1(b'blob %d\0' % len(content.encode()) + content.encode()).hexdigest()
    hdr = re.sub(r'(?m)^(index 0+\.\.)[0-9a-f]+', lambda m: m.group(1) + blob[:len(m.group(0)) - len(m.group(1))], hdr)
    lines = new.split('\n')
    sections[where] = (hdr + f'\n@@ -0,0 +1,{len(lines)} @@\n' + ''.join(f'+{ln}\n' for ln in lines)
                       + ('\\ No newline at end of file\n' if no_eol else ''))
    return {'test_patch': ''.join(sections), 'report_kind': kind}


def js_cases(kind: str, report: bytes) -> tuple[dict[str, str], int]:
    """({id: status}, suites that ran no test) from a jest --json or a JUnit (mocha xunit / vitest junit) report.
    One canonical id per test, in a form the grader's report parser emits: jest `fullName` (describe blocks +
    title); JUnit `classname::name`. Two cases sharing an id keep the worst status. A jest suite that failed
    with no assertion results (its file did not load at base, e.g. a TypeScript error against code the fix
    adds) contributes no id and is only counted -- unlike pytest and Go there is no static fallback yet."""
    rank = {'pass': 0, 'skip': 1, 'fail': 2}
    out: dict[str, str] = {}
    empty = 0

    def put(key, st):
        if key and (key not in out or rank[st] > rank[out[key]]):
            out[key] = st
    if kind == 'jest':
        d = json.loads(report.decode('utf-8', 'replace'))
        for suite in d.get('testResults', []):
            ar = suite.get('assertionResults') or []
            if not ar and suite.get('status') == 'failed':
                empty += 1
            for a in ar:
                put(a.get('fullName'), {'passed': 'pass', 'failed': 'fail', 'pending': 'skip', 'skipped': 'skip',
                                        'todo': 'skip'}.get(a.get('status'), 'fail'))
        return out, empty
    import xml.etree.ElementTree as ET
    for tc in ET.fromstring(report).iter('testcase'):
        name, cls = tc.get('name', ''), tc.get('classname', '')
        kids = {c.tag for c in tc}
        put(f'{cls}::{name}' if cls else name,
            'fail' if {'failure', 'error'} & kids else 'skip' if 'skipped' in kids else 'pass')
    return out, empty


def _derive_js(env, factory, task: str, gi: dict, spec: dict, meta: dict, runs: int) -> dict:
    """jest / vitest / mocha: the task's one runner run is amended to write a structured report
    (amend_test_patch), the amended task is baselined like the others, and the amended test patch or test
    command is returned with the expected tests so the signed task index -- and so the grader -- uses it."""
    amend = amend_test_patch(gi['test_patch'], gi['test_command'])
    if amend is None:
        return {'id': task, 'report_kind': '', 'fail_to_pass': [], 'pass_to_pass': [],
                'baseline': {**meta, 'runs': 0, 'skipped': 'no amendable jest / vitest / mocha run'}}
    kind = amend['report_kind']
    changed = {k: amend[k] for k in ('test_patch', 'test_command') if k in amend}
    gi2 = {**gi, **changed}
    t0 = time.time()
    per_run: list[dict[str, str]] = []
    rcs, empty = [], 0
    for _ in range(runs):
        s = env.open(spec, factory)
        try:
            s.setup()
            g = s.grade({**gi2, 'report_kind': kind, 'collect_report': True})
        finally:
            s.close()
        if g.get('grade_detail'):
            raise RuntimeError(f'baseline grade of {task} did not run the tests: {g["grade_detail"]}')
        rcs.append(g.get('verifier_returncode'))
        rep = (g.get('report') or '').encode()
        cases, e = js_cases(kind, rep) if rep.strip() else ({}, 0)
        per_run.append(cases)
        empty = max(empty, e)
    ids = set().union(*per_run) if per_run else set()
    stable = {i: per_run[0][i] for i in ids if all(r.get(i) == per_run[0].get(i) for r in per_run)}
    f2p = sorted(i for i, st in stable.items() if st == 'fail')
    p2p = sorted(i for i, st in stable.items() if st == 'pass')
    meta.update(seconds=round(time.time() - t0, 1), returncodes=rcs, n_fail=len(f2p), n_pass=len(p2p),
                n_skip=sum(1 for st in stable.values() if st == 'skip'), flaky=sorted(ids - set(stable))[:50],
                unloaded_suites=empty, amended=sorted(changed))
    if not f2p:
        meta['warning'] = 'no failing test at base: an empty patch would pass every expected test'
    if empty:
        meta['warning_unloaded'] = f'{empty} suite(s) ran no test at base: their tests are not expected'
    ok = bool(f2p or p2p)
    return {'id': task, 'report_kind': kind if ok else '', 'fail_to_pass': f2p, 'pass_to_pass': p2p,
            **(changed if ok else {}), 'baseline': meta}


def _uses_pytest(gi: dict) -> bool:
    return any(_PYTEST_RE.search(t) for _, t in test_patch_scripts(gi['test_patch'])) or bool(
        _PYTEST_RE.search(gi['test_command']))


def _uses_gotest(gi: dict) -> bool:
    return any(_GOTEST_RE.search(t) for _, t in test_patch_scripts(gi['test_patch'])) or bool(
        _GOTEST_RE.search(gi['test_command']))


def _go_dirs_of(pkg: str, test_dirs: set[str]) -> list[str]:
    """Repo dirs of the test-patch files that belong to import path `pkg`: the dirs `pkg` ends with; the repo
    root only when no other dir matches (the module path itself is not known here)."""
    hit = sorted(d for d in test_dirs if d and (pkg == d or pkg.endswith('/' + d)))
    return hit or ([''] if '' in test_dirs else [])


def _derive_go(env, factory, task: str, gi: dict, spec: dict, meta: dict, runs: int) -> dict:
    """The go test counterpart of the pytest path in derive(): the grader's `go` shim records `go test -json`
    events for an empty rollout. A package that does not build at base (its new test file uses code only the
    fix adds) has no events, so its expected tests are the test functions declared in the test patch's files
    for that package, selected by the -run of the go test commands that run that package."""
    t0 = time.time()
    per_run: list[dict[str, str]] = []
    unbuilt: set[str] = set()
    files: dict[str, str] = {}
    rcs, shims = [], set()
    for _ in range(runs):
        s = env.open(spec, factory)
        try:
            s.setup()
            g = s.grade({**gi, 'report_kind': 'gotest', 'collect_report': True})
        finally:
            s.close()
        if g.get('grade_detail'):
            raise RuntimeError(f'baseline grade of {task} did not run the tests: {g["grade_detail"]}')
        rcs.append(g.get('verifier_returncode'))
        shims.add(str(g.get('go_report_shim', '')))
        files.update(g.get('report_test_files') or {})
        cases, bad = gotest_cases((g.get('report') or '').encode())
        per_run.append(cases)
        unbuilt |= bad
    ids = set().union(*per_run) if per_run else set()
    stable = {i: per_run[0][i] for i in ids if all(r.get(i) == per_run[0].get(i) for r in per_run)}
    flaky = sorted(ids - set(stable))
    f2p = sorted(i for i, st in stable.items() if st == 'fail')
    p2p = sorted(i for i, st in stable.items() if st == 'pass')
    commands = go_test_commands(list(test_patch_scripts(gi['test_patch'])) + [('', gi['test_command'])],
                                spec['cwd'])
    go_files = {p: c for p, c in files.items() if p.endswith('_test.go')}
    test_dirs = {p.rsplit('/', 1)[0] if '/' in p else '' for p in go_files}
    static: list[str] = []
    for pkg in sorted(unbuilt):
        for d in _go_dirs_of(pkg, test_dirs):
            names = sorted({n for p, c in go_files.items()
                            if (p.rsplit('/', 1)[0] if '/' in p else '') == d for n in static_go_test_ids(c)})
            static += [f'{pkg}::{n}' for n in go_select(names, go_patterns_for_dir(commands, d))]
    f2p = sorted(set(f2p) | set(static))
    meta.update(seconds=round(time.time() - t0, 1), returncodes=rcs, n_fail=len(f2p), n_pass=len(p2p),
                n_skip=sum(1 for st in stable.values() if st == 'skip'), flaky=flaky[:50],
                unbuilt=sorted(unbuilt), static_ids=len(static), go_report_shim=sorted(shims),
                go_test_commands=len(commands))
    if not f2p:
        meta['warning'] = 'no failing test at base: an empty patch would pass every expected test'
    return {'id': task, 'report_kind': 'gotest' if (f2p or p2p) else '', 'fail_to_pass': f2p,
            'pass_to_pass': p2p, 'baseline': meta}


def _repo_prefix(file_attrs: set[str], known: set[str]) -> str:
    """The part of pytest's rootdir-relative paths that lies above the repo, found from a test file we know the
    repo-relative path of (the test patch's files): '' when rootdir is the repo or inside it. With it stripped
    the ids are repo-relative, a form the grader's report parser generates whatever the rootdir."""
    for f in sorted(file_attrs):
        for p in known:
            if f.endswith('/' + p):
                return f[:-len(p)]
    return ''


def derive(env, factory, task: str, runs: int = 2) -> dict:
    """One task's baseline record: {'id', 'report_kind', 'fail_to_pass', 'pass_to_pass', 'baseline': {...}}.
    `env` is a SweEnv, `factory` its sandbox factory (image, cwd) -> Sandbox. Raises what the grader raises
    (GradeError, SetupError) on an infra failure, so the caller can retry or skip."""
    gi = env.grade_inputs(task)
    spec = env.public_spec(task)
    meta: dict = {'version': BASELINE_VERSION, 'grader': GRADER_VERSION, 'runs': runs}
    if not _uses_pytest(gi):
        if _uses_gotest(gi):
            return _derive_go(env, factory, task, gi, spec, meta, runs)
        if _js_runs(gi['test_command']) or any(_js_runs(t) for _, t in test_patch_scripts(gi['test_patch'])):
            return _derive_js(env, factory, task, gi, spec, meta, runs)
        return {'id': task, 'report_kind': '', 'fail_to_pass': [], 'pass_to_pass': [],
                'baseline': {**meta, 'runs': 0, 'skipped': 'not a pytest or go test task'}}
    t0 = time.time()
    per_run: list[dict[str, str]] = []
    broken: set[str] = set()
    files: dict[str, str] = {}
    rcs = []
    for _ in range(runs):
        s = env.open(spec, factory)
        try:
            s.setup()
            g = s.grade({**gi, 'report_kind': 'junit', 'collect_report': True})
        finally:
            s.close()
        if g.get('grade_detail'):                  # an empty rollout must reach the tests; anything else is ours
            raise RuntimeError(f'baseline grade of {task} did not run the tests: {g["grade_detail"]}')
        rcs.append(g.get('verifier_returncode'))
        files.update(g.get('report_test_files') or {})
        rep = (g.get('report') or '').encode()
        cases, bad = junit_cases(rep) if rep.strip() else ({}, [])
        keys = _junit_results(rep) if rep.strip() else {}
        pre = _repo_prefix({i.split('::', 1)[0] for i in cases} | set(bad), set(files))
        per_run.append({i[len(pre):] if i.startswith(pre) else i: st
                        for i, st in cases.items() if (i[len(pre):] if i.startswith(pre) else i) in keys})
        broken.update(b[len(pre):] if b.startswith(pre) else b for b in bad)
    ids = set().union(*per_run) if per_run else set()
    stable = {i: per_run[0][i] for i in ids if all(r.get(i) == per_run[0].get(i) for r in per_run)}
    flaky = sorted(ids - set(stable))
    f2p = sorted(i for i, st in stable.items() if st == 'fail')
    p2p = sorted(i for i, st in stable.items() if st == 'pass')
    # test files that do not collect at base: their tests come from the source, filtered by the selection
    nodeids, kexprs = pytest_selection([t for _, t in test_patch_scripts(gi['test_patch'])] + [gi['test_command']],
                                       spec['cwd'])
    static: list[str] = []
    for fil in sorted(broken):
        src = next((c for p, c in files.items() if p == fil or p.endswith('/' + fil)), None)
        if src is not None:
            static += select(static_test_ids(fil, src), nodeids, kexprs)
    f2p = sorted(set(f2p) | set(static))
    meta.update(seconds=round(time.time() - t0, 1), returncodes=rcs, n_fail=len(f2p), n_pass=len(p2p),
                n_skip=sum(1 for st in stable.values() if st == 'skip'), flaky=flaky[:50],
                uncollectable=sorted(broken), static_ids=len(static),
                selection={'nodeids': len(nodeids), 'k': kexprs[:5]})
    if not f2p:
        meta['warning'] = 'no failing test at base: an empty patch would pass every expected test'
    return {'id': task, 'report_kind': 'junit' if (f2p or p2p) else '', 'fail_to_pass': f2p,
            'pass_to_pass': p2p, 'baseline': meta}


# ---------------------------------------------------------------- the task index: derived fields and provenance
# What `hosted-env baseline --tasks-out` adds to a task row. The receipt's task_hash covers the row as served,
# derived fields included, so a verifier holding the same index gets the same hash.
DERIVED_FIELDS = ('fail_to_pass', 'pass_to_pass', 'report_kind', 'amended_from')
# the deterministic amendment amend_test_patch makes (reporter flags on the one jest / vitest / mocha run); a
# change to what it produces is a new rule name, so an index says which one it applied
AMEND_RULE = 'js-report-flags-v1'


def source_row(row: dict) -> dict:
    """The row without what the baseline derived: for a row the baseline never amended, the source task."""
    return {k: v for k, v in row.items() if k not in DERIVED_FIELDS}


def merge_baseline(row: dict, rec: dict) -> dict:
    """The task row with a baseline record merged in: its expected tests and, for a JS task whose grading script
    was amended to write a report, the amended test_patch / test_command -- the expected tests hold only for the
    amended harness, so it replaces the original. An amended row records amended_from: the task_hash of the
    source row it came from and AMEND_RULE, so anyone holding the source row can check (check_amendment) that
    the amendment only added reporter flags. A record without expected tests leaves the row as it was."""
    if not (rec.get('report_kind') and rec.get('fail_to_pass')):
        return row
    out = {**row, 'report_kind': rec['report_kind'], 'fail_to_pass': rec['fail_to_pass'],
           'pass_to_pass': rec['pass_to_pass']}
    amended = {k: rec[k] for k in ('test_patch', 'test_command') if rec.get(k) and rec[k] != row.get(k)}
    if amended:
        out.update(amended)
        out['amended_from'] = {'task_hash': task_hash(source_row(row)), 'rule': AMEND_RULE,
                               'fields': sorted(amended)}
    return out


def check_amendment(source: dict, served: dict) -> str | None:
    """None when served is the source task row amended exactly as AMEND_RULE does it (plus the derived expected
    tests); otherwise what differs. The amendment is deterministic, so this re-runs it on the source row and
    compares byte for byte: every field outside DERIVED_FIELDS and the amended ones must be unchanged."""
    prov = served.get('amended_from')
    if not isinstance(prov, dict):
        return 'served row is not amended'
    if prov.get('rule') != AMEND_RULE:
        return f'unknown amendment rule {prov.get("rule")!r}'
    if prov.get('task_hash') != task_hash(source_row(source)):
        return 'amended_from.task_hash is not the source row'
    redo = amend_test_patch(source.get('test_patch', ''), source.get('test_command', ''))
    if redo is None:
        return 'the source row has no amendable run'
    fields = [k for k in ('test_patch', 'test_command') if k in redo]
    if prov.get('fields') != fields:
        return f'amended fields {prov.get("fields")!r}, the rule amends {fields!r}'
    if served.get('report_kind') != redo['report_kind']:
        return 'report_kind is not the amended run\'s'
    for k in sorted(set(source_row(source)) | set(source_row(served))):
        want = redo[k] if k in fields else source.get(k)
        if served.get(k) != want:
            return f'{k} differs from the amended source'
    return None
