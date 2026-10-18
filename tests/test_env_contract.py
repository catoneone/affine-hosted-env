"""The contract between the infrastructure and an environment: the receipt fields an environment declares
(EnvReceiptSpec), what it fills for one rollout, and the executor build stamp."""
import subprocess
from pathlib import Path

import pytest
from nacl.signing import SigningKey

from hosted_env import __version__
from hosted_env.envs import DockerTaskEnv, SweEnv, receipt_spec
from hosted_env.infra.receipt import (CORE_FIELDS, EnvReceiptSpec, ReceiptError, sign_receipt, trajectory_sha256,
                                      verify_receipt)

# the flat v2 context an SWE receipt carried before the fields moved out of the infrastructure: the wire format
# must not change by moving where the schema is declared
SWE_V2_FIELDS = {'image', 'start_tree', 'test_patch_sha256', 'test_command_sha256', 'grader', 'grader_parsers',
                 'expected_tests_sha256', 'sandbox_limits', 'grader_user', 'model_patch_sha256', 'verifier_returncode',
                 'model_id', 'model_revision', 'tokenizer_sha256', 'chat_template_sha256', 'renderer',
                 'executor_version', 'started_at_ms'}
CORE = {'sandbox_limits': 'k8s:cpus=1;memory=2g;disk=10g;threads=1', 'model_id': 'm', 'model_revision': 'r',
        'tokenizer_sha256': 'e' * 64, 'chat_template_sha256': 'f' * 64, 'renderer': 'char', 'executor_version': 't',
        'started_at_ms': 1_800_000_000_000}
SEGS = [('P', [1, 2]), ('M', [3, 4])]
ID = dict(epoch='e1', hotkey='5H', env_id='swe', task='t', nonce=0)


def test_swe_receipt_fields_unchanged():
    assert set(SweEnv.RECEIPT.context_fields()) == SWE_V2_FIELDS
    assert receipt_spec('mimo_swe') is SweEnv.RECEIPT and receipt_spec('docker_task') is DockerTaskEnv.RECEIPT
    with pytest.raises(ValueError, match='unknown environment kind'):
        receipt_spec('nope')


def test_spec_cannot_shadow_core_or_payload_keys():
    for name in ('reward', 'outcome', 'nonce', 'model_id', 'started_at_ms'):
        with pytest.raises(ValueError, match='clash'):
            EnvReceiptSpec(fields={name: str})
    with pytest.raises(ValueError, match='artifact'):
        EnvReceiptSpec(fields={'x_sha256': str}, artifacts={'x': ('y_sha256', 10)})
    EnvReceiptSpec(fields={'x_sha256': str}, artifacts={'x': ('x_sha256', 10)})


def test_receipt_is_checked_against_the_environments_spec():
    """A receipt verifies only with the spec of the environment that signed it: each spec requires its own fields."""
    wk = SigningKey.generate()
    d, n = trajectory_sha256(SEGS)
    env = DockerTaskEnv('dt', {})
    ctx = {**env.receipt_context(None, {'test_command': 'pytest'}, {'verifier_returncode': 0}), **CORE}
    assert set(ctx) == set(DockerTaskEnv.RECEIPT.context_fields())
    r = sign_receipt(wk, 'w1', trajectory_sha256=d, n_tokens=n, outcome='completed', reward=1.0, context=ctx,
                     spec=DockerTaskEnv.RECEIPT, **ID)
    keys = {'w1': wk.verify_key}
    assert verify_receipt(r, SEGS, keys, spec=DockerTaskEnv.RECEIPT, **ID)['verifier_returncode'] == 0
    with pytest.raises(ReceiptError, match='malformed_receipt'):          # SWE needs start_tree, grader, ...
        verify_receipt(r, SEGS, keys, spec=SweEnv.RECEIPT, **ID)
    with pytest.raises(ValueError, match='bad receipt context'):         # and cannot be signed short of them
        sign_receipt(wk, 'w1', trajectory_sha256=d, n_tokens=n, outcome='completed', reward=1.0, context=ctx,
                     spec=SweEnv.RECEIPT, **ID)
    assert set(CORE_FIELDS) <= set(DockerTaskEnv.RECEIPT.context_fields())


def test_executor_version_names_the_checkout_commit():
    """executor_version is the package version plus this checkout's commit; the path check that decides "this is
    our own checkout" must follow the package wherever its files move (it went stale when executor/ moved under
    infra/, and every receipt then said only the bare version)."""
    from hosted_env.infra.executor.core import _build_version
    here = Path(__file__).resolve().parent
    top = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True, cwd=here)
    if top.returncode != 0 or not (Path(top.stdout.strip()) / 'src' / 'hosted_env').is_dir():
        pytest.skip('not running from a git checkout of this repository')
    v = _build_version()
    assert v.startswith(f'{__version__}+g') and len(v.split('+g')[1].split('.')[0]) == 12
