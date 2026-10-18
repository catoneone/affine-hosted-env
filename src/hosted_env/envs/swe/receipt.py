"""What an SWE receipt binds beyond the infrastructure's core fields (EnvReceiptSpec), and the digest of a task's
expected tests. Only the standard library and the receipt framework, so a validator can import it cheaply."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from ...infra.common import canonical
from ...infra.receipt import EnvReceiptSpec, register_spec

MAX_MODEL_PATCH_BYTES = 2_000_000       # the grader never grades a larger patch; also the attachment's cap

# id: bump the version for any change below and keep the old spec registered (infra.receipt.SPECS)
SPEC = register_spec(EnvReceiptSpec(
    id='swe/1',
    fields={
        'image': str,                   # name@sha256:... the rollout and its grading ran in
        'start_tree': str,              # git tree the model started from (after setup); the patch applies to it
        'test_patch_sha256': str,       # of the hidden test patch text (UTF-8)
        'test_command_sha256': str,     # of the test command text (UTF-8)
        'grader': str,                  # grading semantics version, e.g. isolated-v3
        'grader_parsers': str,          # pinned parsers that decided keep/revert of structured manifests (v3+)
        'expected_tests_sha256': str,   # expected_tests_digest() of the task's expected tests (fail_to_pass /
                                        # pass_to_pass / report_kind) the second check used; '' when the task has
                                        # none, so the reward rests on the verifier's exit code alone
        'grader_user': str,             # who the hidden tests ran as: nobody or root
        'model_patch_sha256': str,      # of the model's patch bytes as captured, before the grader put protected
                                        # test infrastructure back (a re-grader applies the same rule)
        'verifier_returncode': (int, type(None)),   # None when no verifier ran
    },
    hex_lengths={'start_tree': 40},     # a git tree id, or '' before setup finished
    # the attachment carries the patch that was graded, so a verifier can re-grade it offline
    artifacts={'model_patch': ('model_patch_sha256', MAX_MODEL_PATCH_BYTES)},
))


def expected_tests_digest(fail_to_pass: Sequence[str], pass_to_pass: Sequence[str], report_kind: str) -> str:
    """The receipt's expected_tests_sha256: SHA-256 of canonical JSON {fail_to_pass, pass_to_pass, report_kind},
    each id list de-duplicated and sorted (the second check's verdict does not depend on order). '' when both lists
    are empty -- the task has no expected tests and the second check did not run. A re-grader computes this from
    the signed task index entry and compares it before trusting its own verdict."""
    f2p, p2p = sorted(set(fail_to_pass or ())), sorted(set(pass_to_pass or ()))
    if not f2p and not p2p:
        return ''
    body = {'fail_to_pass': f2p, 'pass_to_pass': p2p, 'report_kind': str(report_kind or '')}
    return hashlib.sha256(canonical(body)).hexdigest()
