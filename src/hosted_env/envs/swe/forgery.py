"""Forgery checks on top of the second check (grader isolated-v4).

The hidden tests run in the same process as the code under test, so that code can always produce the test
framework's output -- print Go's `--- PASS: TestX` lines, write the JUnit / jest report, exit 0 before the tests
run. No result channel is out of its reach (Affine review 2026-10-18), so these checks do not make a report
trustworthy; they make the cheap forgeries fail and leave a verifier the same rules to re-apply:

- patch_audit: the graded patch may not add test-framework output to the code it keeps (the literal forms a forge
  prints or writes), unless the base repository already has that text (a repo that is itself a test reporter), and
  may not name our report plumbing at all.
"""
from __future__ import annotations

import re

AUDIT_RULES = 'patch-audit-v1'
# What a forge has to emit for each report kind the second check reads. Allowed when the start tree already has the
# string somewhere (git grep): then the repo deals in that format itself and an honest fix may add more of it.
FRAMEWORK_OUTPUT = ('--- PASS:', '--- FAIL:', '--- SKIP:', '=== RUN', '"Action":"pass"', '"Action": "pass"',
                    '<testcase', '<testsuite', 'numPassedTests', 'numFailedTests', 'PYTEST_ADDOPTS', 'junit_family',
                    '--junit-xml', '--junitxml')
# The grader's own report plumbing (envs/swe/env.py, reports.py): no honest fix names these
GRADER_PLUMBING = ('MIMO_REPORT', 'MIMO_GO_REPORT', 'mimo-report-')

_DIFF_HEAD = re.compile(r'^diff --git a/(.*) b/(.*)$')


def added_lines(patch: str) -> dict[str, list[str]]:
    """File (new path) -> the lines the patch adds to it. Binary hunks add no lines. Every `diff --git` header
    starts a new file; one whose path git quoted (`"b/caf\\303\\251"`) is keyed by the raw header, so its lines are
    never dropped or credited to another file."""
    out: dict[str, list[str]] = {}
    cur, in_hunk = '', False             # '' collects stray lines before any header: audited, never dropped
    for line in patch.split('\n'):
        if line.startswith('diff --git '):
            m = _DIFF_HEAD.match(line)
            cur, in_hunk = (m.group(2) if m else line), False
            out.setdefault(cur, [])
        elif line.startswith('@@'):
            in_hunk = True
        elif line.startswith('+') and (in_hunk or not line.startswith('+++ ')):
            # inside a hunk every '+' line is content: an added line that itself starts with '++ ' (valid Python:
            # `++ 0 or print(...)`) shows as '+++ ...' and is not a header there
            out.setdefault(cur, []).append(line[1:])
    return out


def patch_audit_hits(patch: str, reverted: set[str] | frozenset = frozenset()) -> list[tuple[str, str, bool]]:
    """(file, string, always_forbidden) for every audited string the patch adds to a file the grader does not put
    back (reverted: the files it reset to base). A file is skipped only when its header path is exactly a reverted
    one, so a path git quotes or escapes is still audited. always_forbidden: a grader-plumbing name, never
    excused; otherwise the hit is excused when the base repository already contains the string."""
    hits = []
    for path, lines in added_lines(patch).items():
        if path in reverted:
            continue
        text = '\n'.join(lines)
        hits += [(path, s, True) for s in GRADER_PLUMBING if s in text]
        hits += [(path, s, False) for s in FRAMEWORK_OUTPUT if s in text]
    return hits
