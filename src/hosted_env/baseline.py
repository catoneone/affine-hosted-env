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

Without a gold patch this cannot prove that every FAIL_TO_PASS test is fixable; a task no real model ever
solves is dropped later. pytest only for now (`report_kind = "junit"`); Go and jest need the grader to produce
their reports first.
"""
from __future__ import annotations

import ast
import re
import time

from .envs.swe import GRADER_VERSION, _junit_results, _repo_rel, _shell_commands, test_patch_scripts

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


def _uses_pytest(gi: dict) -> bool:
    return any(_PYTEST_RE.search(t) for _, t in test_patch_scripts(gi['test_patch'])) or bool(
        _PYTEST_RE.search(gi['test_command']))


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
        return {'id': task, 'report_kind': '', 'fail_to_pass': [], 'pass_to_pass': [],
                'baseline': {**meta, 'runs': 0, 'skipped': 'not a pytest task'}}
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
