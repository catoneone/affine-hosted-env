"""Command line: hosted-env keygen / server / gateway / worker / mine / verify / machines / roster-sync. Services listen on
127.0.0.1 only by default."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Fake model replies for the smoke test (EXP-014 run.py): look at the repo, try a command that times out, submit a
# malformed call, call a nonexistent tool, dump a large output to trigger truncation, then finish without a tool call
SMOKE_REPLIES = [
    'Let me look.\n<tool_call>\n<function=bash>\n<parameter=command>\npwd && git status --short | head && ls | head\n</parameter>\n</function>\n</tool_call>',
    '<tool_call>\n<function=bash>\n<parameter=command>\nsleep 5; echo never\n</parameter>\n<parameter=timeout>\n2\n</parameter>\n</function>\n</tool_call>',
    '<tool_call>\n<function=bash>\n<parameter=command>\necho broken\n</function>',
    '<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call>',
    '<tool_call>\n<function=bash>\n<parameter=command>\nseq 1 30000\n</parameter>\n</function>\n</tool_call>',
    'I am done.',
]


def _serve(app, host: str, port: int):
    import logging

    import uvicorn
    # uvicorn configures only its own loggers; without this, hosted_env INFO lines (capacity plans) are dropped
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    uvicorn.run(app, host=host, port=port, log_level='info')


def cmd_keygen(a):
    from .common import generate_key_file, key_id, public_hex
    sk = generate_key_file(a.path)
    print(json.dumps({'path': str(Path(a.path).resolve()), 'public_key': public_hex(sk),
                      'key_id': key_id(sk.verify_key)}))


def cmd_server(a):
    from .config import build_server_app
    _serve(build_server_app(a.config), a.host, a.port)


def cmd_gateway(a):
    from .config import build_gateway_app
    _serve(build_gateway_app(a.config), a.host, a.port)


def cmd_worker(a):
    from .config import build_worker_app
    _serve(build_worker_app(a.config), a.host, a.port)


def cmd_mine(a):
    from .client import GatewayClient, ScriptedSampler, VLLMSampler, run_task
    from .common import load_key_file
    gw = GatewayClient(a.gateway, a.hotkey, load_key_file(a.key))
    epoch = a.epoch or gw.info()['epoch']
    if a.vllm:
        sampler = VLLMSampler(a.vllm, a.model, temperature=a.temperature, top_p=a.top_p)
    else:
        from .render import make_renderer
        r = make_renderer({'kind': 'char'} if a.tokenizer == 'char' else {'kind': 'hf', 'path': a.tokenizer})
        sampler = ScriptedSampler(r.encode, r.turn_end_id, SMOKE_REPLIES)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for _ in range(a.tasks):
        res = run_task(gw, epoch, a.env, sampler, concurrency=a.concurrency, max_wait=a.max_wait)
        t = res['task']
        if res['attachment'] is not None:                # a voided task (infra error) has nothing to upload
            (out / f'{t["env_id"]}-{t["task"]}.json').write_text(json.dumps(res['attachment']))
        print(json.dumps({'task': t['task'], 'voided': res['voided'],
                          'results': [{k: x[k] for k in ('nonce', 'outcome', 'reward', 'detail', 'steps')}
                                      for x in res['rollouts']]}), flush=True)


def cmd_verify(a):
    import httpx

    from .common import verify_key_from_hex
    from .receipt import ReceiptError
    from .verify import verify_group
    att = json.loads(Path(a.attachment).read_text())
    info = httpx.get(a.gateway.rstrip('/') + '/v1/info', timeout=30).json()
    keys = {w['key_id']: verify_key_from_hex(w['public_key']) for w in info['workers'].values()}
    env = info['envs'][att['env_id']]
    try:
        v = verify_group(att, keys, gateway_key=verify_key_from_hex(info['gateway_public_key']),
                         epoch=a.epoch or info['epoch'], hotkey=a.hotkey, env_id=att['env_id'],
                         task=att['task'], group_size=env['group_size'], success_reward=env['success_reward'])
    except ReceiptError as e:
        print(json.dumps({'ok': False, 'error': e.code}))
        sys.exit(1)
    print(json.dumps({'ok': True, 'outcomes': [(c['outcome'], c['reward']) for c in v['claims']],
                      'trainable_tokens': [sum(m) for m in v['loss_masks']]}))


def _machine_rows(rows: list[dict], now: float) -> list[dict]:
    """Add hours and an estimated cost (list price x hours since created, or since first seen for machines the
    planner did not create)."""
    out = []
    for r in rows:
        hours = max(0.0, ((r['ended_at'] or now) - r['created_at']) / 3600)
        cost = None if r['price_per_hour'] is None else hours * r['price_per_hour']
        out.append({**r, 'hours': round(hours, 3), 'est_cost_usd': None if cost is None else round(cost, 4)})
    return out


def cmd_machines(a):
    import time
    from datetime import UTC, datetime

    from .config import machine_inventory
    from .coordinator.store import LIVE_MACHINE_STATES
    rows, report = machine_inventory(a.config, refresh=a.refresh)
    now = time.time()
    rows = _machine_rows(rows, now)
    live = [r for r in rows if r['state'] in LIVE_MACHINE_STATES]
    shown = rows if a.all else live
    if a.json:
        print(json.dumps({'machines': shown, 'refresh': report}))
        return

    def ts(t):
        return '-' if t is None else datetime.fromtimestamp(t, UTC).strftime('%m-%d %H:%M')

    head = ('PROVIDER', 'NAME', 'ID', 'ORIGIN', 'STATE', 'KIND', '$/H', 'SINCE(UTC)', 'SEEN(UTC)', 'HOURS', '~COST',
            'ADDRESS', 'NOTE')
    table = [head] + [(r['provider'], r['name'], r['machine_id'], r['origin'],
                       r['state'] + (f"/{r['provider_state']}" if r['provider_state'] not in (None, 'running') else ''),
                       r['kind'], '-' if r['price_per_hour'] is None else f"{r['price_per_hour']:.3f}",
                       ts(r['created_at']), ts(r['last_seen']), f"{r['hours']:.2f}",
                       '-' if r['est_cost_usd'] is None else f"${r['est_cost_usd']:.2f}",
                       r['public_ipv4'] or r['public_ipv6'] or '-', r['note']) for r in shown]
    widths = [max(len(str(row[i])) for row in table) for i in range(len(head) - 1)]
    for row in table:
        print('  '.join(str(c).ljust(w) for c, w in zip(row, widths, strict=False)) + '  ' + str(row[-1]))
    by_origin = {}
    for r in live:
        by_origin[r['origin']] = by_origin.get(r['origin'], 0) + 1
    rate = sum(r['price_per_hour'] or 0 for r in live)
    planner = [r for r in rows if r['origin'] == 'planner']
    print(f"\nlive: {len(live)} machines ({', '.join(f'{k} {v}' for k, v in sorted(by_origin.items())) or 'none'}), "
          f'${rate:.3f}/h at list prices')
    print(f"planner, all time: {len(planner)} machines, {sum(r['hours'] for r in planner):.2f} machine-hours, "
          f"~${sum(r['est_cost_usd'] or 0 for r in planner):.2f}")
    if report is not None:
        print(f"refreshed: {len(report['new'])} new, {len(report['gone'])} gone"
              + ''.join(f'; {p} list failed: {e}' for p, e in report['errors'].items()))
    elif rows:
        print(f"last listed: {ts(max((r['last_seen'] or 0) for r in rows))} UTC (--refresh lists the providers now)")


def cmd_roster_sync(a):
    import logging

    from .chain import sync_forever, sync_once
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    if a.every:
        sync_forever(a.out, a.every, network=a.network, netuid=a.netuid)
    else:
        print(json.dumps(sync_once(a.out, network=a.network, netuid=a.netuid)))


def cmd_pin_images(a):
    from .images import pin_tasks
    src = Path(a.tasks)
    lines, stats = pin_tasks(src.read_text().splitlines())
    dst = Path(a.out) if a.out else src
    dst.write_text('\n'.join(lines) + '\n')
    print(json.dumps({**stats, 'out': str(dst)}))


def main(argv=None):
    p = argparse.ArgumentParser(prog='hosted-env')
    sub = p.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('keygen', help='Generate an Ed25519 private key file (mode 600), print the public key and key id')
    k.add_argument('path')
    k.set_defaults(fn=cmd_keygen)
    for name, fn, port in (('server', cmd_server, 8780), ('gateway', cmd_gateway, 8780), ('worker', cmd_worker, 8790)):
        s = sub.add_parser(name)
        s.add_argument('--config', required=True)
        s.add_argument('--host', default='127.0.0.1')
        s.add_argument('--port', type=int, default=port)
        s.set_defaults(fn=fn)
    m = sub.add_parser('mine', help='Reference miner: open a task, run the whole group, write the group attachment to --out')
    m.add_argument('--gateway', required=True)
    m.add_argument('--hotkey', required=True)
    m.add_argument('--key', required=True, help='mining key file (hex seed)')
    m.add_argument('--env', required=True)
    m.add_argument('--epoch')
    m.add_argument('--tasks', type=int, default=1)
    m.add_argument('--concurrency', type=int, default=8)
    m.add_argument('--max-wait', type=float, default=600.0, help='max time to wait when capacity is insufficient')
    m.add_argument('--out', default='attachments')
    m.add_argument('--tokenizer', default='char', help='tokenizer path for the smoke-test fake model (char = character tokenizer for tests)')
    m.add_argument('--vllm', help='vLLM URL; if given, use a real model')
    m.add_argument('--model', default='m')
    m.add_argument('--temperature', type=float, default=1.0)
    m.add_argument('--top-p', type=float, default=1.0)
    m.set_defaults(fn=cmd_mine)
    v = sub.add_parser('verify', help='Verify a group attachment (worker public keys fetched from gateway /v1/info)')
    v.add_argument('attachment')
    v.add_argument('--gateway', required=True)
    v.add_argument('--hotkey', required=True)
    v.add_argument('--epoch')
    v.set_defaults(fn=cmd_verify)
    mc = sub.add_parser('machines', help='Machine inventory: every machine the planner created plus unmanaged '
                                         'hosted-env- machines found at the providers, with hours and est. cost')
    mc.add_argument('--config', required=True, help='the server or gateway config (its db and [machines])')
    mc.add_argument('--all', action='store_true', help='include deleted and gone machines (history)')
    mc.add_argument('--refresh', action='store_true',
                    help='list the [machines] providers now (read only; nothing is created or deleted)')
    mc.add_argument('--json', action='store_true')
    mc.set_defaults(fn=cmd_machines)
    rs = sub.add_parser('roster-sync', help='Write the SN120 registered hotkeys (upstream registrations() rule) to a '
                                           'roster file the gateway hot-reloads (needs the chain extra)')
    rs.add_argument('--out', required=True, help='e.g. roster.chain.json, the server config\'s roster_chain')
    rs.add_argument('--network', default='finney', help='bittensor network name or ws(s):// endpoint')
    rs.add_argument('--netuid', type=int, default=120)
    rs.add_argument('--every', type=float, default=0, help='seconds between syncs; 0 = sync once and exit')
    rs.set_defaults(fn=cmd_roster_sync)
    pi = sub.add_parser('pin-images', help='Rewrite a task file so every image is name@sha256:<digest> '
                                          '(the tag is kept as image_tag); needs registry access')
    pi.add_argument('tasks')
    pi.add_argument('--out', help='write here instead of in place')
    pi.set_defaults(fn=cmd_pin_images)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
