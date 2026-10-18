"""SWE's own `hosted-env` subcommands, registered by hosted_env.cli through hosted_env.envs.CLI_EXTENSIONS."""
from __future__ import annotations

import json
from pathlib import Path


def _env_and_factory(config: str, env_id: str, node: str | None, **overrides):
    """(env, sandbox factory, env config) from a server / worker config, with overrides on the env's config. The
    sandbox is the config's kind: k8s (optionally on one node), docker, or local (tests)."""
    import tomllib

    from ...infra.config import _envs
    from .. import make_env
    cfgp = Path(config).resolve()
    ecfg = {**_envs(tomllib.loads(cfgp.read_text()), cfgp.parent)[env_id], **overrides}
    env = make_env(env_id, ecfg)
    opts = dict(ecfg.get('sandbox_opts') or {})
    kind = ecfg.get('sandbox', 'docker')
    if kind == 'k8s':
        from ...infra.providers.sandbox.k8s import K8sSandbox
        if node:
            opts['node_name'] = node
        factory = lambda image, cwd: K8sSandbox(image, cwd, **opts)  # noqa: E731
    elif kind == 'docker':
        from ...infra.providers.sandbox import DockerSandbox
        factory = lambda image, cwd: DockerSandbox(image, cwd, **opts)  # noqa: E731
    elif kind == 'local':
        from ...infra.providers.sandbox import LocalSandbox
        factory = lambda image, cwd: LocalSandbox(image, cwd, **opts)  # noqa: E731
    else:
        raise SystemExit(f'needs a k8s, docker or local sandbox, not {kind!r}')
    return env, factory, ecfg


def cmd_baseline(a):
    """Expected tests per task from baseline runs (envs.swe.baseline): one JSON record per task to --out, and
    with --tasks-out a copy of the task file with fail_to_pass / pass_to_pass / report_kind filled in."""
    from .baseline import derive
    # the baseline derives the expected tests, so it must see the tasks that do not have them yet
    env, factory, ecfg = _env_and_factory(a.config, a.env, a.node, require_expected_tests=False)
    ids = a.only.split(',') if a.only else env.task_ids()
    known = set(env.task_ids())
    ids = [next((t for t in known if t == i or t.endswith(i)), i) for i in ids]
    done = {}
    with open(a.out, 'a') as out:
        for task in ids:
            try:
                rec = derive(env, factory, task, runs=a.runs)
            except Exception as e:  # noqa: BLE001  one task's infra failure must not stop the batch
                rec = {'id': task, 'error': f'{type(e).__name__}: {e}'[:500]}
            done[task] = rec
            out.write(json.dumps(rec) + '\n')
            out.flush()
            print(json.dumps({k: rec.get(k) for k in ('id', 'report_kind', 'error')} |
                             {'f2p': len(rec.get('fail_to_pass') or []), 'p2p': len(rec.get('pass_to_pass') or [])}),
                  flush=True)
    if a.tasks_out:
        from .baseline import merge_baseline
        rows = [json.loads(x) for x in Path(ecfg['tasks']).read_text().splitlines() if x.strip()]
        # expected tests, and for a JS task the amended harness they hold for (with its provenance)
        rows = [merge_baseline(r, done.get(str(r['id'])) or {}) for r in rows]
        Path(a.tasks_out).write_text(''.join(json.dumps(r) + '\n' for r in rows))


def cmd_regrade(a):
    """Re-grade a group attachment as a verifier would. verify_group checks every receipt and artifact with
    expect = the task_hash and image of the verifier's OWN copy of the task row (so the patch graded is exactly the
    one the receipt binds, against the task it was served), skipping only the group-level pass/fail rules; then the
    grader version must be this code's, the grading user is the receipts' own, and the verifier's sandbox limits
    must be the receipts'. Each graded rollout's captured patch is graded again in fresh containers
    (envs.swe.env.SweEnv.regrade) and the rewards compared. One JSON line per rollout and a summary; exit 0 when
    every re-grade matches, 2 on a mismatch, 1 on a refusal (nothing is graded then)."""
    import sys
    from types import SimpleNamespace

    import httpx

    from ...infra.common import verify_key_from_hex
    from ...infra.executor.core import _sandbox_limits
    from ...infra.receipt import ReceiptError, read_receipt, task_hash
    from ...infra.verify import verify_group
    from .. import receipt_spec
    from .env import GRADER_VERSION

    def refuse(code: str, **kw):
        print(json.dumps({'ok': False, 'error': code, **kw}))
        sys.exit(1)
    try:
        att = json.loads(Path(a.attachment).read_text())
        want = set(map(int, a.nonces.split(','))) if a.nonces else None
    except (OSError, ValueError) as e:
        refuse('bad_input', detail=f'{type(e).__name__}: {e}'[:200])
    if not isinstance(att, dict) or not isinstance(att.get('rollouts'), list) or not att['rollouts']:
        refuse('malformed_group')
    try:
        first = read_receipt(att['rollouts'][0].get('receipt') or {})   # defaults only; verify_group checks it all
    except (ReceiptError, AttributeError):
        first = {}
    try:
        info = httpx.get(a.gateway.rstrip('/') + '/v1/info', timeout=30).json()
        keys = {w['key_id']: verify_key_from_hex(w['public_key']) for w in info['workers'].values()}
        gw_key = verify_key_from_hex(info['gateway_public_key'])
        einfo = info['envs'][att.get('env_id')]
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as e:
        refuse('gateway_info_unusable', detail=f'{type(e).__name__}: {e}'[:200])
    try:
        rows = {str(r['id']): r for r in (json.loads(x) for x in Path(a.tasks).read_text().splitlines() if x.strip())}
    except (OSError, ValueError, KeyError, TypeError) as e:
        refuse('bad_tasks_file', detail=f'{type(e).__name__}: {e}'[:200])
    row = rows.get(str(att.get('task')))
    if row is None:
        refuse('unknown_task', task=att.get('task'))
    mine = {'task_hash': task_hash(row), 'image': row.get('image')}
    try:
        v = verify_group(att, keys, gateway_key=gw_key, epoch=a.epoch or first.get('epoch'),
                         hotkey=a.hotkey or first.get('hotkey'), env_id=att['env_id'], task=att['task'],
                         spec=receipt_spec(einfo['kind']), group_size=einfo['group_size'],
                         success_reward=einfo['success_reward'], expect=mine, require_gate=False)
    except ReceiptError as e:
        if e.code == 'context_mismatch':                   # say which: the receipts were served another task row
            field = next((k for k, x in mine.items() if first.get(k) != x), None)
            refuse(f'{field}_mismatch' if field else e.code, receipt=first.get(field), yours=mine.get(field))
        refuse(e.code)
    claims, arts = v['claims'], v['artifacts']
    if any(c['grader'] != GRADER_VERSION for c in claims) and not a.any_grader:
        refuse('grader_mismatch', receipt=claims[0]['grader'], this_code=GRADER_VERSION)
    users = {c['grader_user'] for c in claims}
    if len(users) != 1 or not users <= {'root', 'nobody'}:
        refuse('grader_user_unusable', receipt=sorted(users))
    env, factory, ecfg = _env_and_factory(a.config, a.env or att['env_id'], a.node,
                                          tasks=str(Path(a.tasks).resolve()), require_expected_tests=False,
                                          indices=None, grader_user=users.pop())   # grade as the receipts did
    limits = _sandbox_limits(SimpleNamespace(sandbox=ecfg.get('sandbox', 'docker'),
                                             sandbox_opts=dict(ecfg.get('sandbox_opts') or {})))
    if any(c['sandbox_limits'] != limits for c in claims) and not a.any_limits:
        refuse('sandbox_limits_mismatch', receipt=claims[0]['sandbox_limits'], yours=limits)
    results = []
    for c, art in zip(claims, arts, strict=True):
        n = c['nonce']
        if (want is not None and n not in want) or c['outcome'] == 'infra_error' or not c['model_patch_sha256']:
            continue                                        # not selected, voided, or never graded
        if a.passing_only and c['reward'] < einfo['success_reward']:
            continue
        rec = {'nonce': n, 'receipt_reward': c['reward']}
        try:
            g = env.regrade(att['task'], art['model_patch'], c['start_tree'], factory)
            rec.update(regrade_reward=g['reward'], match=g['reward'] == c['reward'],
                       **{k: g[k] for k in ('second_check', 'canary', 'grade_detail') if k in g})
        except Exception as e:  # noqa: BLE001  an environment that cannot re-grade is reported, not a match
            rec.update(regrade_reward=None, match=False, error=f'{type(e).__name__}: {e}'[:300])
        results.append(rec)
        print(json.dumps(rec), flush=True)
    bad = [r['nonce'] for r in results if not r['match']]
    print(json.dumps({'ok': not bad, 'regraded': len(results), 'mismatched_nonces': bad, 'grader': GRADER_VERSION,
                      'sandbox_limits': limits}))
    sys.exit(2 if bad else 0)


def register(sub) -> None:
    rg = sub.add_parser('regrade', help='Re-grade a group attachment\'s captured patches in fresh containers and '
                                        'compare with the receipts\' rewards (a verifier\'s own check)')
    rg.add_argument('--attachment', required=True, help='the group attachment (hosted-env-group-v3 JSON)')
    rg.add_argument('--gateway', required=True, help='gateway URL, for /v1/info (worker and gateway keys)')
    rg.add_argument('--tasks', required=True, help='YOUR copy of the task index; each receipt\'s task_hash must '
                                                   'match its row')
    rg.add_argument('--config', required=True, help='a config whose [envs] section has the env (its sandbox kind '
                                                    'and limits are used)')
    rg.add_argument('--env', help='env id in --config (default: the attachment\'s)')
    rg.add_argument('--node', help='k8s node to run the sandboxes on')
    rg.add_argument('--hotkey', help='expected miner hotkey (default: the receipts\' own; pass it to catch a group '
                                     'replayed under another miner)')
    rg.add_argument('--epoch', help='expected epoch (default: the receipts\' own; pass it to catch a replay from '
                                    'another epoch)')
    rg.add_argument('--nonces', help='comma-separated nonces to re-grade (default: every graded rollout)')
    rg.add_argument('--passing-only', action='store_true', help='re-grade only rollouts the receipt scored a pass')
    rg.add_argument('--any-grader', action='store_true', help='re-grade even when the receipt names another grader '
                                                              'version than this code')
    rg.add_argument('--any-limits', action='store_true', help='re-grade even when your sandbox limits differ from '
                                                              'the receipts\' (a limit kill scores 0, so results can '
                                                              'differ)')
    rg.set_defaults(fn=cmd_regrade)
    bl = sub.add_parser('baseline', help='Derive each task\'s expected tests (fail_to_pass / pass_to_pass) by grading '
                                         'an empty rollout with the real grader (pytest tasks for now)')
    bl.add_argument('--config', required=True, help='the server / worker config whose [envs] has the env')
    bl.add_argument('--env', required=True)
    bl.add_argument('--out', required=True, help='append one JSON record per task here')
    bl.add_argument('--only', help='comma-separated task ids (or id suffixes); default every task')
    bl.add_argument('--runs', type=int, default=2, help='baseline runs per task; a test whose status differs is '
                                                       'flaky and left out')
    bl.add_argument('--node', help='k8s node to run the sandboxes on')
    bl.add_argument('--tasks-out', help='write a copy of the task file with the derived fields filled in')
    bl.set_defaults(fn=cmd_baseline)
