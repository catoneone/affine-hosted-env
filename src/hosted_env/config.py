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


def build_gateway_app(path: str):
    from .coordinator.store import Limits, Store
    from .gateway.app import Roster, create_gateway_app
    cfg, base = _load(path)
    key = load_key_file(_abs(base, cfg['key_file']))
    envs = {e: make_env(e, c) for e, c in _envs(cfg, base).items()}
    store = Store(_abs(base, cfg.get('db', 'gateway.db')), envs, key, Limits(**(cfg.get('limits') or {})))
    workers = {w: verify_key_from_hex(h) for w, h in (cfg.get('workers') or {}).items()}
    return create_gateway_app(store, Roster(_abs(base, cfg['roster'])), workers, key.verify_key)


def build_worker(path: str):
    from .executor.core import EnvRuntime, Worker
    from .executor.gateway_client import WorkerGatewayClient
    from .providers.sandbox import ImageCache
    from .render import make_renderer
    cfg, base = _load(path)
    key = load_key_file(_abs(base, cfg['key_file']))
    runtimes = {}
    for env_id, ecfg in _envs(cfg, base).items():
        rcfg = dict(ecfg.get('renderer') or {})
        if 'path' in rcfg:
            rcfg['path'] = _abs(base, rcfg['path'])
        rt_keys = ('max_context', 'max_turn_tokens', 'max_steps', 'max_seconds', 'min_room', 'sandbox')
        runtimes[env_id] = EnvRuntime(env=make_env(env_id, ecfg), renderer=make_renderer(rcfg),
                                      sandbox_opts=dict(ecfg.get('sandbox_opts') or {}),
                                      **{k: ecfg[k] for k in rt_keys if k in ecfg})
    gw = WorkerGatewayClient(cfg['gateway_url'], cfg['worker_id'], key)
    log_dir = _abs(base, cfg.get('log_dir'))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    return Worker(worker_id=cfg['worker_id'], key=key, gateway=gw,
                  gateway_pub=verify_key_from_hex(cfg['gateway_public_key']), runtimes=runtimes,
                  public_url=cfg['public_url'], capacity=int(cfg.get('capacity', 16)),
                  idle_ttl=int(cfg.get('idle_ttl', 15 * 60)), log_dir=log_dir,
                  image_cache=ImageCache(max_images=int(cfg.get('max_cached_images', 10))),
                  scratch=_abs(base, cfg.get('scratch')))


def build_worker_app(path: str):
    from .executor.app import create_worker_app
    return create_worker_app(build_worker(path))
