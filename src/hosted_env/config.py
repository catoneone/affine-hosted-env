"""Assemble the gateway / worker from TOML config. Relative paths are resolved against the config file's directory.
See examples/ for samples."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

from .common import load_key_file, verify_key_from_hex
from .envs import make_env


def _load(path: str) -> tuple[dict, Path]:
    p = Path(path).resolve()
    return tomllib.loads(p.read_text()), p.parent


def _abs(base: Path, v: str | None) -> str | None:
    if v is None:
        return None
    return v if os.path.isabs(v) else str((base / v).resolve())


def _envs(cfg: dict, base: Path, keys=('tasks',)) -> dict:
    out = {}
    for env_id, ecfg in (cfg.get('envs') or {}).items():
        ecfg = dict(ecfg)
        for k in keys:
            if k in ecfg:
                ecfg[k] = _abs(base, ecfg[k])
        out[env_id] = ecfg
    return out


def _store(cfg: dict, base: Path, key):
    from .coordinator.store import Limits, Store
    envs = {e: make_env(e, c) for e, c in _envs(cfg, base).items()}
    placement = 'nodes' if (cfg.get('placement') or {}).get('kind') == 'k8s' else 'workers'
    return Store(_abs(base, cfg.get('db', 'gateway.db')), envs, key, Limits(**(cfg.get('limits') or {})),
                 placement=placement)


def _node_source(cfg: dict):
    """[placement] kind = "k8s": read nodes from the cluster (in-cluster config, else kubeconfig). Other keys
    (cpus, memory, reserve_cpu, reserve_memory, include_control_plane) size one standard sandbox slot."""
    pcfg = dict(cfg.get('placement') or {})
    if pcfg.pop('kind', None) != 'k8s':
        return None
    kubeconfig = pcfg.pop('kubeconfig', None)

    def read():
        from kubernetes import client, config

        from .coordinator.nodes import k8s_nodes
        if kubeconfig:
            api = client.CoreV1Api(config.new_client_from_config(config_file=kubeconfig))
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            api = client.CoreV1Api()
        return k8s_nodes(api, **pcfg)
    return read


def _limiter(cfg: dict):
    from .gateway.ratelimit import RateLimiter, RateLimits
    return RateLimiter(RateLimits.from_config(cfg.get('rate_limits')))


def build_gateway_app(path: str):
    from .gateway.app import Roster, create_gateway_app
    cfg, base = _load(path)
    key = load_key_file(_abs(base, cfg['key_file']))
    workers = {w: verify_key_from_hex(h) for w, h in (cfg.get('workers') or {}).items()}
    return create_gateway_app(_store(cfg, base, key), Roster(_abs(base, cfg['roster'])), workers, key.verify_key,
                              limiter=_limiter(cfg), node_source=_node_source(cfg))


def _runtimes(cfg: dict, base: Path) -> dict:
    from .executor.core import EnvRuntime
    from .render import make_renderer
    runtimes = {}
    for env_id, ecfg in _envs(cfg, base).items():
        rcfg = dict(ecfg.get('renderer') or {})
        if 'path' in rcfg:
            rcfg['path'] = _abs(base, rcfg['path'])
        rt_keys = ('max_context', 'max_turn_tokens', 'max_steps', 'max_seconds', 'min_room', 'sandbox')
        runtimes[env_id] = EnvRuntime(env=make_env(env_id, ecfg), renderer=make_renderer(rcfg),
                                      sandbox_opts=dict(ecfg.get('sandbox_opts') or {}),
                                      **{k: ecfg[k] for k in rt_keys if k in ecfg})
    return runtimes


def _executor(wcfg: dict, base: Path, key, gateway, gateway_pub, public_url: str, runtimes: dict):
    from .executor.core import Worker
    from .providers.sandbox import ImageCache
    log_dir = _abs(base, wcfg.get('log_dir'))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    return Worker(worker_id=wcfg['worker_id'], key=key, gateway=gateway, gateway_pub=gateway_pub,
                  runtimes=runtimes, public_url=public_url, capacity=int(wcfg.get('capacity', 16)),
                  idle_ttl=int(wcfg.get('idle_ttl', 15 * 60)), log_dir=log_dir,
                  image_cache=ImageCache(max_images=int(wcfg.get('max_cached_images', 10))),
                  scratch=_abs(base, wcfg.get('scratch')))


def build_worker(path: str):
    """Standalone executor that reaches the coordinator over HTTP."""
    from .executor.gateway_client import WorkerGatewayClient
    cfg, base = _load(path)
    key = load_key_file(_abs(base, cfg['key_file']))
    gw = WorkerGatewayClient(cfg['gateway_url'], cfg['worker_id'], key)
    return _executor(cfg, base, key, gw, verify_key_from_hex(cfg['gateway_public_key']), cfg['public_url'],
                     _runtimes(cfg, base))


def build_worker_app(path: str):
    from .executor.app import create_worker_app
    return create_worker_app(build_worker(path))


def build_server_app(path: str):
    """gateway + coordinator + executor in one process (architecture v2). The executor calls the coordinator
    directly and is reachable only through the gateway's /v1/start and /v1/turn."""
    from .executor.gateway_client import InProcessGatewayClient
    from .gateway.app import Roster, create_gateway_app
    cfg, base = _load(path)
    gkey = load_key_file(_abs(base, cfg['key_file']))
    store = _store(cfg, base, gkey)
    wcfg = {'worker_id': 'local', **(cfg.get('executor') or {})}
    wkey = load_key_file(_abs(base, wcfg['key_file']))
    if bytes(wkey) == bytes(gkey):
        raise ValueError('the executor receipt key must differ from the gateway session key')
    worker = _executor(wcfg, base, wkey, InProcessGatewayClient(store, wcfg['worker_id']), gkey.verify_key,
                       cfg['public_url'], _runtimes(cfg, base))
    return create_gateway_app(store, Roster(_abs(base, cfg['roster'])), {wcfg['worker_id']: wkey.verify_key},
                              gkey.verify_key, limiter=_limiter(cfg), executor=worker, node_source=_node_source(cfg))
