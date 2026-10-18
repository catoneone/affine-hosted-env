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

# the flat context of an SWE receipt (swe/1 under receipt v3): v2's fields, which moved out of the infrastructure
# without changing on the wire, plus v3's core task_hash
SWE_V3_FIELDS = {'image', 'start_tree', 'test_patch_sha256', 'test_command_sha256', 'grader', 'grader_parsers',
                 'expected_tests_sha256', 'sandbox_limits', 'grader_user', 'model_patch_sha256', 'verifier_returncode',
                 'model_id', 'model_revision', 'tokenizer_sha256', 'chat_template_sha256', 'renderer',
                 'executor_version', 'started_at_ms', 'task_hash'}
CORE = {'task_hash': '7' * 64, 'sandbox_limits': 'k8s:cpus=1;memory=2g;disk=10g;threads=1', 'model_id': 'm',
        'model_revision': 'r', 'tokenizer_sha256': 'e' * 64, 'chat_template_sha256': 'f' * 64, 'renderer': 'char',
        'executor_version': 't', 'started_at_ms': 1_800_000_000_000}
SEGS = [('P', [1, 2]), ('M', [3, 4])]
ID = dict(epoch='e1', hotkey='5H', env_id='swe', task='t', nonce=0)


def test_swe_receipt_fields_unchanged():
    assert set(SweEnv.RECEIPT.context_fields()) == SWE_V3_FIELDS
    assert (SweEnv.RECEIPT.id, DockerTaskEnv.RECEIPT.id) == ('swe/1', 'docker_task/1')
    assert receipt_spec('mimo_swe') is SweEnv.RECEIPT and receipt_spec('docker_task') is DockerTaskEnv.RECEIPT
    with pytest.raises(ValueError, match='unknown environment kind'):
        receipt_spec('nope')


def test_spec_cannot_shadow_core_or_payload_keys():
    for name in ('reward', 'outcome', 'nonce', 'model_id', 'started_at_ms', 'task_hash', 'env_spec'):
        with pytest.raises(ValueError, match='clash'):
            EnvReceiptSpec(id='x/1', fields={name: str})
    with pytest.raises(ValueError, match='artifact'):
        EnvReceiptSpec(id='x/1', fields={'x_sha256': str}, artifacts={'x': ('y_sha256', 10)})
    EnvReceiptSpec(id='x/1', fields={'x_sha256': str}, artifacts={'x': ('x_sha256', 10)})


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
    with pytest.raises(ReceiptError, match='env_spec_mismatch'):          # signed as docker_task/1
        verify_receipt(r, SEGS, keys, spec=SweEnv.RECEIPT, **ID)
    with pytest.raises(ValueError, match='bad receipt context'):         # and cannot be signed short of them
        sign_receipt(wk, 'w1', trajectory_sha256=d, n_tokens=n, outcome='completed', reward=1.0, context=ctx,
                     spec=SweEnv.RECEIPT, **ID)
    assert set(CORE_FIELDS) <= set(DockerTaskEnv.RECEIPT.context_fields())
    # the other direction: an SWE receipt has docker_task's fields and more, and still fails with docker_task's
    # spec -- otherwise the narrower spec would skip the type and digest checks of the SWE fields
    swe = SweEnv('swe', {})
    sctx = {**swe.receipt_context(None, {}, {}), **CORE}
    rs = sign_receipt(wk, 'w1', trajectory_sha256=d, n_tokens=n, outcome='completed', reward=0.0, context=sctx,
                      spec=SweEnv.RECEIPT, **ID)
    verify_receipt(rs, SEGS, keys, spec=SweEnv.RECEIPT, **ID)
    with pytest.raises(ReceiptError, match='env_spec_mismatch'):    # the exact field-set rule behind it is pinned by
        verify_receipt(rs, SEGS, keys, spec=DockerTaskEnv.RECEIPT, **ID)   # test_spec_is_found_from_the_receipt...
    assert swe.artifacts({}) == {'model_patch': ''} and set(swe.artifacts({})) == set(SweEnv.RECEIPT.artifacts)
    assert env.artifacts({}) == {} and not DockerTaskEnv.RECEIPT.artifacts


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


def test_the_format_doc_matches_the_code():
    """docs/formats.md is what verifiers port against: its versions, domains and field tables must be the code's."""
    import re

    from hosted_env.infra import receipt as R
    from hosted_env.infra.verify import GROUP_ATTACHMENT_VERSION
    doc = (Path(__file__).resolve().parents[1] / 'docs' / 'formats.md').read_text()
    assert f'`v: {R.RECEIPT_VERSION}`' in doc and R.RECEIPT_DOMAIN.decode().rstrip('|') in doc
    assert R.SEAL_DOMAIN.decode().rstrip('|') in doc and R.TRAJ_DOMAIN.decode() in doc
    assert GROUP_ATTACHMENT_VERSION in doc and f'version {__version__}' in doc and f'`v{__version__}`' in doc

    def table(heading):                       # field names in the first column of the first table under heading
        part = doc.split(heading, 1)[1]
        rows = part[part.index('\n|'):].split('\n\n', 1)[0]
        return {n for row in re.findall(r'(?m)^\| (.+?) \|', rows) for n in re.findall(r'`(\w+)`', row)}
    assert table('Core context fields') == set(R.CORE_FIELDS)
    assert table('### `swe/1`') == set(SweEnv.RECEIPT.fields)
    assert table('Flat signed payload') == set(R.PAYLOAD_KEYS)
    assert set(re.findall(r'`(\w+)` \(', doc.split('### `docker_task/1`', 1)[1].split('\n#', 1)[0])) == \
        set(DockerTaskEnv.RECEIPT.fields)
