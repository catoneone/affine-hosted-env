"""Command line: hosted-env keygen / gateway / worker / mine / verify. Services listen on 127.0.0.1 only by default."""
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
        (out / f'{t["env_id"]}-{t["task"]}.json').write_text(json.dumps(res['attachment']))
        print(json.dumps({'task': t['task'], 'results': [{k: x[k] for k in ('nonce', 'outcome', 'reward', 'detail',
                                                                           'steps')} for x in res['rollouts']]}),
              flush=True)


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
        v = verify_group(att, keys, epoch=a.epoch or info['epoch'], hotkey=a.hotkey, env_id=att['env_id'],
                         task=att['task'], group_size=env['group_size'], success_reward=env['success_reward'])
    except ReceiptError as e:
        print(json.dumps({'ok': False, 'error': e.code}))
        sys.exit(1)
    print(json.dumps({'ok': True, 'outcomes': [(c['outcome'], c['reward']) for c in v['claims']],
                      'trainable_tokens': [sum(m) for m in v['loss_masks']]}))


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
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
