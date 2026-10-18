"""SWE's own `hosted-env` subcommands, registered by hosted_env.cli through hosted_env.envs.CLI_EXTENSIONS."""
from __future__ import annotations

import json
from pathlib import Path


def cmd_baseline(a):
    """Expected tests per task from baseline runs (envs.swe.baseline): one JSON record per task to --out, and
    with --tasks-out a copy of the task file with fail_to_pass / pass_to_pass / report_kind filled in."""
    import tomllib

    from ...infra.config import _envs
    from .. import make_env
    from .baseline import derive
    cfgp = Path(a.config).resolve()
    ecfg = _envs(tomllib.loads(cfgp.read_text()), cfgp.parent)[a.env]
    # the baseline derives the expected tests, so it must see the tasks that do not have them yet
    ecfg = {**ecfg, 'require_expected_tests': False}
    env = make_env(a.env, ecfg)
    opts = dict(ecfg.get('sandbox_opts') or {})
    kind = ecfg.get('sandbox', 'docker')
    if kind == 'k8s':
        from ...infra.providers.sandbox.k8s import K8sSandbox
        if a.node:
            opts['node_name'] = a.node
        factory = lambda image, cwd: K8sSandbox(image, cwd, **opts)  # noqa: E731
    elif kind == 'docker':
        from ...infra.providers.sandbox import DockerSandbox
        factory = lambda image, cwd: DockerSandbox(image, cwd, **opts)  # noqa: E731
    else:
        raise SystemExit(f'baseline needs a k8s or docker sandbox, not {kind!r}')
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


def register(sub) -> None:
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
