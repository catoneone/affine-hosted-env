"""The second check: structured test reports (JUnit XML, go test -json, jest --json), the per-test-id
verdict against the expected tests, and the `go` shim that makes go test write its report."""
from __future__ import annotations

import json



_STATUS_RANK = {'pass': 0, 'skip': 1, 'fail': 2}


def _record(out: dict, key: str, status: str) -> None:
    """Record `key -> status`, keeping the WORST status when a key is seen twice (fail > skip > pass). Several id
    spellings collapse to one key and a report may carry two cases that share one, so the second check -- a forgery
    backstop -- must fail closed: a later `pass` never masks an earlier `fail`/`skip` under a shared key."""
    if key and (key not in out or _STATUS_RANK[status] > _STATUS_RANK[out[key]]):
        out[key] = status


def _junit_results(report: bytes) -> dict:
    """{test id -> 'pass'|'fail'|'skip'} from a JUnit XML report (pytest --junit-xml, jest-junit, go-junit).
    A case with a <failure>/<error> child failed; `skipped` is skip; otherwise pass. pytest roots `classname` at
    the filesystem path (e.g. tmp.abc.tests.test_calc), so for every dotted classname we register nodeids built
    from each path SUFFIX -- `tests/test_calc.py::name`, with the last segment taken as a module or (for class
    tests) as a class. Over-generating ids is lossy -- two cases can collapse to one key -- so _record keeps the
    worst status (a model-added passing case cannot mask the real expected one); the model also cannot inject a
    pytest-collectable module in the first place, since protected_paths reverts any test_*.py / *_test.py file."""
    import xml.etree.ElementTree as ET
    out: dict[str, str] = {}
    root = ET.fromstring(report)
    for tc in root.iter('testcase'):
        name, cls, fil = tc.get('name', ''), tc.get('classname', ''), tc.get('file', '')
        children = {c.tag for c in tc}
        status = 'fail' if {'failure', 'error'} & children else ('skip' if 'skipped' in children else 'pass')
        keys = {name}
        if cls:
            keys |= {f'{cls}::{name}', f'{cls}.{name}'}
        if fil:
            keys.add(f'{fil}::{name}')
        parts = cls.split('.') if cls else []
        for k in range(len(parts)):
            mod = '/'.join(parts[k:])
            keys.add(f'{mod}.py::{name}')                               # last segment is the module file
            if len(parts) - k >= 2:                                     # or the last segment is a class
                keys.add(f'{"/".join(parts[k:-1])}.py::{parts[-1]}::{name}')
        for key in keys:
            _record(out, key, status)
    return out


def _gotest_results(report: bytes) -> dict:
    """{test id -> status} from `go test -json` output (one JSON object per line, action pass/fail/skip per Test).
    _record keeps the worst status so a rerun's later pass cannot mask an earlier fail under the same id."""
    out: dict[str, str] = {}
    for line in report.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        t, act = ev.get('Test'), ev.get('Action')
        if t and act in ('pass', 'fail', 'skip'):
            _record(out, t, act)
            if ev.get('Package'):
                _record(out, f"{ev['Package']}::{t}", act)
    return out


def _jestjson_results(report: bytes) -> dict:
    """{test id -> status} from `jest --json` output (testResults[].assertionResults[].{fullName,title,status}).
    _record keeps the worst status so a later pass cannot mask an earlier fail/skip under a shared id."""
    out: dict[str, str] = {}
    d = json.loads(report.decode('utf-8', 'replace'))
    for suite in d.get('testResults', []):
        for a in suite.get('assertionResults', []):
            st = {'passed': 'pass', 'failed': 'fail', 'pending': 'skip', 'skipped': 'skip',
                  'todo': 'skip'}.get(a.get('status'), 'fail')
            for key in {a.get('fullName'), a.get('title')}:
                _record(out, key, st)
    return out


_REPORT_PARSERS = {'junit': _junit_results, 'gotest': _gotest_results, 'jest': _jestjson_results}
_REPORT_EXT = {'junit': 'xml', 'gotest': 'ndjson', 'jest': 'json'}

# `go` placed first on PATH in the grading container for a report_kind='gotest' task: `go test` runs with -json
# and its stdout (one test2json event per line) is appended to $MIMO_GO_REPORT, so every `go test` the task's own
# scripts run lands in the report without editing them. Every other go command, and `go test` in a mode with no
# test events (-c compile only, -list) or already given -json, is passed through untouched -- GOFLAGS=-json is
# not used because go env / go list / go build / go mod also take -json and scripts parse their output.
# EXP-018 go_scan.py over the 721 Go tasks: 702 run `go test`; 2 redirect its output and none pipes it into
# a parser; 4 call go by absolute path or through $GO and so get no report -- the baseline then derives no
# expected tests for them and they stay exit-code graded. stdout is buffered to a file beside the report (no
# pipefail in POSIX sh) and replayed, so the exit status is go test's own.
# The report itself is writable by the test process (its directory is the grader user's, like the JUnit one),
# so the model's code can append forged events: a forged `pass` never masks the real `fail` the shim appends
# after go test exits, because _record keeps the worst status per id. A forged report that lists the expected
# tests as passing while the real run never reached them is the known limit the patch audit backs up.
GO_REPORT_SHIM = '''#!/bin/sh
REAL=__REAL_GO__
if [ "$1" = test ] && [ -n "$MIMO_GO_REPORT" ]; then
  for a in "$@"; do
    case "$a" in
      -args) break ;;
      -c|-json|-json=*|-list|-list=*) exec "$REAL" "$@" ;;
    esac
  done
  shift
  out="$MIMO_GO_REPORT.$$.out"
  "$REAL" test -json "$@" >"$out"
  rc=$?
  cat "$out" >>"$MIMO_GO_REPORT"
  cat "$out"
  rm -f "$out"
  exit $rc
fi
exec "$REAL" "$@"
'''


def report_verdict(report_kind: str, report: bytes, fail_to_pass: list[str], pass_to_pass: list[str]) -> tuple[bool, str]:
    """The second check: every expected test id (the tests the fix must make pass, plus those that must stay
    passing) is present in the structured report AND passed; a skip never counts as a pass. Returns (ok, detail).
    This is on top of the exit code -- a model that forces exit 0 but did not actually pass the tests fails here,
    as long as the ids come from our own baseline, not the model."""
    parse = _REPORT_PARSERS.get(report_kind)
    if parse is None:
        return False, f'unknown report_kind {report_kind!r}'
    try:
        results = parse(report)
    except Exception as e:  # noqa: BLE001  a malformed / forged report is not a pass
        return False, f'report unparseable: {type(e).__name__}'
    expected = list(dict.fromkeys([*(fail_to_pass or []), *(pass_to_pass or [])]))
    if not expected:
        return False, 'no expected test ids'

    def status(t: str) -> str | None:
        """An id read statically from the test source has no parameters; pytest reports `test_x[a]`, `test_x[b]`:
        every instance must pass (worst status), and at least one must be there."""
        if t in results:
            return results[t]
        inst = [s for k, s in results.items() if k.startswith(t + '[')]
        return None if not inst else ('fail' if 'fail' in inst else 'skip' if 'skip' in inst else 'pass')
    got = {t: status(t) for t in expected}
    missing = [t for t in expected if got[t] is None]
    if missing:
        return False, f'{len(missing)} expected test(s) not in the report, e.g. {missing[0]}'
    notpass = [t for t in expected if got[t] != 'pass']
    if notpass:
        return False, f'{len(notpass)} expected test(s) not passing (e.g. {notpass[0]}={got[notpass[0]]})'
    return True, f'{len(expected)} expected tests passed'
