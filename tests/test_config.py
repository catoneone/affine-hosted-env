"""config.py: the configs in examples/, once filled with real values, assemble a gateway and a worker."""
import json
from pathlib import Path

from conftest import make_swe_tasks

from hosted_env.cli import main
from hosted_env.common import load_key_file, public_hex
from hosted_env.config import build_gateway_app, build_worker

EX = Path(__file__).resolve().parents[1] / 'examples'


def test_examples_build(tmp_path, capsys):
    (tmp_path / 'cfg').mkdir()
    main(['keygen', str(tmp_path / 'keys' / 'gateway.key')])
    main(['keygen', str(tmp_path / 'keys' / 'w1.key')])
    out = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert len(out) == 2 and all(len(o['public_key']) == 64 and len(o['key_id']) == 16 for o in out)
    assert (tmp_path / 'keys' / 'w1.key').stat().st_mode & 0o777 == 0o600
    gk, wk = load_key_file(tmp_path / 'keys' / 'gateway.key'), load_key_file(tmp_path / 'keys' / 'w1.key')
    path, _ = make_swe_tasks(tmp_path)
    (tmp_path / 'cfg' / 'tasks.jsonl').write_text(Path(path).read_text())
    (tmp_path / 'cfg' / 'roster.json').write_text(json.dumps({'epoch': 'e1', 'hotkeys': {}}))
    (tmp_path / 'runs').mkdir()
    g = (EX / 'gateway.toml').read_text().replace('<w1 public key hex>', public_hex(wk))
    (tmp_path / 'cfg' / 'gateway.toml').write_text(g)
    w = ((EX / 'worker.toml').read_text().replace('<gateway public key hex>', public_hex(gk))
         .replace('kind = "hf"', 'kind = "char"').replace('https://<public URL from boat host>', 'http://127.0.0.1:1'))
    (tmp_path / 'cfg' / 'worker.toml').write_text(w)

    app = build_gateway_app(str(tmp_path / 'cfg' / 'gateway.toml'))
    assert {r.path for r in app.routes} >= {'/v1/tasks/open', '/v1/sessions/open', '/v1/internal/close'}
    worker = build_worker(str(tmp_path / 'cfg' / 'worker.toml'))
    rt = worker.runtimes['mimo_swe']
    assert (rt.max_context, rt.sandbox, rt.sandbox_opts['network'], worker.capacity) == (65536, 'docker', 'none', 16)
    assert (tmp_path / 'runs' / 'worker-logs').is_dir()
    worker.pool.shutdown()
