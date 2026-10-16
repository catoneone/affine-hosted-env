# affine-hosted-env

Hosted tool-environment service for SN120. Miners keep running inference on their own GPUs, but every tool call is executed by an environment hosted by the operator; at the end of each rollout the environment signs a receipt, the miner uploads the receipt alongside its data through the existing pipeline into the bucket, and the validator and trainer check the receipt with a single pure function.

Design doc: [docs/design.md](docs/design.md) (overall design v1, 2026-10-15).
Architecture v2 (layered services on Kubernetes, in progress): [docs/architecture.md](docs/architecture.md).

## Why

In tool environments like SWE, about half of a trajectory is tool output rather than model-generated tokens, which the existing sampling proof does not cover. If the environment runs on the miner's machine, the miner can forge outputs (for example, by stuffing the answer into a fake `cat` result). Here the environment is put in the operator's hands: the worker parses tool calls from the model tokens, executes them and renders the results into tokens, and the hash of the whole trajectory is signed into the receipt. Verification does not replay, does not read the operator's storage, and does not check step by step.

## Components

| Module | Purpose | Dependencies |
|---|---|---|
| `hosted_env.receipt` | Trajectory hash, receipt issuance, `verify_receipt`, `loss_mask` | PyNaCl only; can be copied upstream as is |
| `hosted_env.verify` | `verify_group`: a group's 8 receipts + the "at least one pass and one fail" gate | Same as above |
| `hosted_env.gateway` | Public HTTP edge: hotkey signature verification, replay protection, roster checks, routing | FastAPI |
| `hosted_env.coordinator` | Quota, task assignment, global budget / fair share, sessions, session tokens, worker registration | SQLite |
| `hosted_env.executor` | Formerly the worker: renders task prompt / tool output tokens, executes, grades, signs receipts; long poll, idempotent retries | FastAPI |
| `hosted_env.providers.sandbox` | One container per rollout: `K8sSandbox` (Pod, gVisor, egress denied), `DockerSandbox` + image cache, `LocalSandbox` (tests) | `kubernetes` (extra `k8s`), Docker |
| `hosted_env.envs` | Environment interface: `docker_task` (image + task prompt + grading command), `mimo_swe` (Xiaomi MiMo SWE tasks) | |
| `hosted_env.client` | Reference miner: opens a task, runs the group concurrently, writes the attachment into the group object; pluggable sampler | httpx |

```text
miner GPU ──① tasks/open, sessions/open, tasks/finish (hotkey-signed)──▶ gateway
   │                                                                    │ assigns to worker
   └──② start / turn (session token)──▶ worker (container, execute, grade, sign receipt) ◀──┘
   │
   └──③ existing upload flow (group + 8 receipts)──▶ R2 ──④ existing audit job ──▶ validator / trainer: verify_group()
```

## Install and test

```bash
uv sync                      # Python 3.12; installs server / tokenizer / dev dependencies
uv run pytest                # No docker / GPU needed: sandbox uses LocalSandbox, model is a scripted fake model
```

Rendering tests with real tokenizers read `hf-tokenizer-mimo-9b/` and `hf-tokenizer-qwen35-9b/` under the `HOSTED_ENV_TOKENIZERS` directory, and are skipped if those are absent.

Verification only (validator / trainer): `pip install "affine-hosted-env @ git+https://github.com/catoneone/affine-hosted-env"`; the core install pulls only PyNaCl.

## Running

Single process (architecture v2): gateway, coordinator and executor in one process. Miners only talk to this
port, for both the control plane and `/v1/start` / `/v1/turn`.

```bash
hosted-env keygen keys/gateway.key        # signs session tokens
hosted-env keygen keys/executor.key       # signs receipts (must be a different key)
hosted-env server --config examples/server.toml            # default 127.0.0.1:8780
hosted-env machines --config examples/server.toml          # machine inventory: what runs where, hours, est. cost
                                                           # --all adds history, --refresh lists the providers now
```

Split mode (a separate executor process that reaches the gateway over HTTP):

```bash
hosted-env keygen keys/gateway.key        # prints the public key and key id
hosted-env keygen keys/w1.key
hosted-env gateway --config examples/gateway.toml          # default 127.0.0.1:8780
hosted-env worker  --config examples/worker.toml           # default 127.0.0.1:8790; exposed publicly via `boat host`
hosted-env mine --gateway http://127.0.0.1:8780 --hotkey <ss58> --key <mining key> --env mimo_swe \
    --tokenizer <tokenizer dir>              # fake model for smoke tests; --vllm <url> for a real model
hosted-env verify attachments/<env>-<task>.json --gateway http://127.0.0.1:8780 --hotkey <ss58>
```

See [examples/](examples/) for config examples. Services listen on 127.0.0.1 only; the gateway's miner endpoints require a hotkey signature, the worker endpoints require a worker key signature, and `/v1/info` is public and read-only.

## Protocol essentials

- Control-plane signed content: `domain | method | path | identity | timestamp | request id | sha256(body)`; timestamp within ±60 s, each request id can be used only once.
- Data plane: `POST /v1/start` returns the task prompt tokens (source P) and the stop tokens; then loop `POST /v1/turn {seq, tokens}`, where tokens are the model tokens the miner sampled up to the first stop token (source M, stop token included), and the response returns tool output tokens (source T) to be appended to the context as is. A turn in which the model calls no tool finishes the rollout.
- Append-only context, no re-rendering (token in, token out); the incremental tool output is obtained by diff-rendering with the chat template.
- Each request waits at most 25 s; if not done it returns `pending`, and resending the same seq with the same tokens keeps waiting on the same operation.
- Receipt: epoch, hotkey, env_id, task, nonce, trajectory hash, token count, outcome (completed / unresolved / infra_error), reward, worker key id, issue time, signed with the worker's Ed25519 key.
- Group attachment: `{"version": "hosted-env-group-v1", "env_id", "task", "rollouts": [{"nonce", "segments": [{"src", "ids"}], "receipt"}]}`.

## Status

P0 code (single-machine env-server + minimal miner client + fake model); all local tests pass. Not yet run on boat and not yet connected to a real model. Open items are listed in design doc §13.

## License

MIT. The system prompt, bash tool description, output truncation, anti-hack cleanup and grading flow in `hosted_env/envs/` are taken from Xiaomi's [mimoagent](https://github.com/XiaomiMiMo/mimoagent) (MIT).
