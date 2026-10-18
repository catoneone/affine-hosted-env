> This document is copied from the internal research repo's `analysis/swe-env-service-design.md` (2026-10-15 v1). EXP-0xx IDs refer to experiment notes in that internal research repo (not included here); the prototype code has since moved into this repo's `src/hosted_env/`.

# SN120 hosted tool-environment service: overall design v1

- date: 2026-10-15 | status: design final v1, not implemented (the gateway quota and receipt parts have prototypes) | based on SN120 rules at upstream `f07c54e` (2026-10-14)
- Basis: EXP-014 (internal SWE harness notes), EXP-015 (internal boat trial measurements, gateway and receipt prototypes), `analysis/swe-env-requirements.md` (internal requirements analysis: problem analysis and alternative approaches)
- In one sentence: miners keep running inference on their own GPUs, and hand every tool call to our environment hosted on boat.dev for execution; at the end of each rollout the environment signs a receipt, the miner uploads the receipt with the data into the bucket through the existing flow, and validators and the trainer check the receipt with a single pure function.

## 1. Problem, goals and scope
**Problem**: in tool environments such as SWE, about half of a trajectory is tool output, not model-generated tokens, and the existing sampling proof does not cover it. If the environment runs on the miner's machine, the miner can forge outputs (e.g. stuff the answer into a fake `cat` result), and this cannot be disproven after the fact.

**Goals**
- Every tool output is produced by the operator's environment; a validator can confirm that a rollout really ran in our environment and that its outputs were not altered.
- Verification cost is about the same as for math tasks: no replay, no reads from our storage, no step-by-step checking.
- Minimal upstream changes: upload paths, commitments, freeze and audit job dispatch all stay the same.
- Controllable cost: determined by per-hotkey quotas and a global budget.
- Not tied to SWE: any environment of the form "Docker image + task prompt + grading command" can be plugged in (decided by the user, 2026-10-15).

**Out of scope for this version**
- Miners running the environment themselves (sampled replay + deposit) and TEE (in cooperation with SN4); both approaches are in `analysis/swe-env-requirements.md` §2.1.0.
- Protection against forgery after a worker is compromised (container escape to obtain the signing key): the user decided to harden this later (§12).
- Cross-datacenter disaster recovery.

## 2. Relevant current SN120 rules (upstream `f07c54e`, read 2026-10-15)
| Rule | Content | Source |
|---|---|---|
| Identity | Requires an SN120 UID, an Ed25519 hotkey and Affine activation; qualified identities are snapshotted at each opening | `dashboard/public/llms.txt`, `subnet/chain.py` `registrations()` |
| Registration cost | UID capacity raised to 1,000, PoW registration is free | `llms.txt` |
| Group contract | One group = one task, nonces 0-7, 8 rollouts, all submitted; at least 1 finished and correct, 1 finished and wrong | `llms.txt` |
| Rewards | At most 9 qualified groups per UID per epoch; if several miners submit the same task in the same epoch, all score 0 | `llms.txt` |
| Task pool | The minable tasks of each epoch are determined only by `environments[].indices` in the signed opening; a cooling mechanism forces >=4,096 tasks to be kept (only for `affine_math`) | `llms.txt`, `ops/task_cooling.py` |
| Upload | Miners use operator-presigned, encrypted-delivered PUT links to write `private/{epoch}/commitments/{hotkey}.json`, `staging/{hotkey}/{slot}.zip`, `training/{hotkey}/{slot}.json` into the operator's R2 | `subnet/storage.py`, `complete_group_transport.py` |
| Verification | The operator's continuous audit samples groups, freezes them by ETag, computes hashes, presigns read links and dispatches signed verification jobs | `subnet/continuous_audit_service.py` |
| Cadence | Each round collects for 30 minutes; target is under 1 hour per epoch | `mining.json`, `llms.txt` |

## 3. Overall architecture

```text
Miner GPU (anywhere)                         Operator
+------------------------+ (1) open / finish  +-------------------------------+
| miner client           | -----------------> | gateway: auth + quota + tasks |
|  - sample to stop token|  (hotkey-signed)   |  + global budget + scheduling |
|  - take tool tokens    |                    +---------------+---------------+
|  - upload group+rcpts  | (2) turn (session token)           | assign
+-------+----------------+ -----------------> +---------------v---------------+   +------------------+
        |                                     | worker (boat large VM)        |<--| registry cache   |
        |                                     |  env-server + many containers |   | (EU, all images) |
        |                                     |  parse tool calls/exec/grade  |   +------------------+
        |                                     |  trajectory hash + receipt    |
        |                                     +-------------------------------+
        | (3) existing upload flow (group + 8 receipts)
        v
+------------------------+  (4) existing audit job  +-------------------------------+
| operator R2 (existing) | -----------------------> | validator / trainer           |
+------------------------+                          | sampling proof +              |
                                                    | verify_receipt()              |
                                                    +-------------------------------+
```

| Component | Responsibility | Deployment |
|---|---|---|
| gateway | Hotkey signature verification, quotas, task assignment, global budget, scheduling sessions onto workers, issuing session tokens | One small operator machine + Postgres |
| worker / env-server | Start containers, render the task prompt, parse tool calls from model tokens, execute, render tool-output tokens, grade, sign receipts | boat `large` VMs, long-lived and reused |
| registry cache | Pull-through cache of all images, about 6.7 TB of compressed layers (Xiaomi SWE) | Separate large-disk machine in the EU, cold backup on R2 |
| miner client | Sampling; sends model tokens to the worker, appends returned tokens to the context; uploads the group and receipts | The miner's GPU machine |
| validator / trainer | Existing sampling proof + one `verify_receipt` call per rollout | Existing |

## 4. Identity and quotas (gateway)
Implemented in `src/hosted_env/infra/auth.py` and `src/hosted_env/infra/coordinator/` (tests: `tests/test_auth.py`, `tests/test_coordinator_store.py`).

**Identity**
- The miner signs every control-plane request with the supervisor's Ed25519 mining key; the signed content is `domain | method | path | hotkey | timestamp | request id | sha256(request body)`.
- The gateway verifies signatures locally against the qualified-identity list of this epoch's opening snapshot (the public_key from `registrations()`), with no chain lookup per request. Timestamps are allowed +-60 s, and each request id can be used only once.
- Verified: Ed25519 signatures from bittensor `Keypair(crypto_type=0)` and `cryptography` interoperate.
- After a session opens successfully, the gateway issues a gateway-signed session token (bound to hotkey, epoch, task, nonce and expiry). The worker holds only the gateway public key and only verifies tokens; it no longer verifies hotkey signatures at each step.

**Quotas** (decided by the user, 2026-10-15, configurable per environment)
| Quota | Value | How it is counted |
|---|---|---|
| Tasks running concurrently | 9 per hotkey | From opening the first nonce until `finish_task`, or all nonces opened and no active session, or no active session and idle >15 minutes |
| Tasks started per epoch | 20 per hotkey | Counted as "started"; not refunded on submit or abandon |
| Concurrent sessions per task | 10 per (hotkey, task) | Current group size is 8, so in practice at most 8 |
| nonce | Each (hotkey, task, nonce) can be opened only once | Cannot be reopened even after close or expiry, to prevent rerunning to cherry-pick results |
| Session idle | Expires after 15 minutes | Releases the concurrency slot so a crashed client does not hold it forever |
| Global in-flight containers | Budget B (set by money) | **To be implemented**: queue when exceeded, fair share across hotkeys |

- A finished task cannot have more nonces opened in this epoch.
- Guessing a qualification rate of 40-60%, 20 tasks yield about 8-12 qualified groups, which matches "9 qualified groups rewarded per epoch"; anything beyond 9 earns no reward under current rules, and the gateway does not handle that layer.
- A global budget is required: in the worst case one hotkey holds 9 x 8 = 72 containers, and hotkeys cost almost nothing to open (UID capacity 1,000, free PoW), so 1,000 hotkeys in the worst case means 72,000 containers.

## 5. Task assignment and task pool
- **The gateway assigns tasks; miners cannot choose**: the miner requests "open a task" and the gateway decides which one. Three reasons: no task collisions within an epoch; tasks can preferentially go to workers that already cache the image; miners cannot use our environment to grind specific tasks. A miner can abandon an assigned task with `finish_task`, but it consumes one of the "20 per epoch" quota slots.
- **Every epoch opens all available tasks**, not a shrunken subset. Shrinking to a few hundred tasks would conflict with the collision rule (200 UIDs x 9 groups = up to 1,800 contributions per epoch), and would go against the protocol's intent of keeping a wide task pool.
- **Collisions**: a task is assigned to only one miner at a time; once a miner's group on it qualifies, the task is not assigned again this epoch. For groups that do not qualify, the task returns to the pool and is assigned to someone else (the sampling randomness is bound to the miner identity, so a different miner gets different results).
- **The task pool itself is small**: Xiaomi SWE has only 2,698 tasks (row count of `code.parquet`, 2026-10-15); after removing held-out tasks and tasks the 9B model solves always / never, it is likely well under 2,400 (pass rate to be measured). The per-epoch qualified-contribution ceiling for SWE is set by the task pool, so the number of rewarded groups per UID must be decided separately rather than copying math's 9.
- The opening's `environments[].indices` lists all available tasks in the existing format; "tasks assigned by the gateway" is a contract term of the new environment, and math is unchanged.
- The prototype has no task assignment or global budget yet; `open_session` currently takes the task id from the caller.

## 6. Worker and execution environment
- **Machine**: boat `large` (8 vCPU / 16 GB / 125 GB, $0.072/h), long-lived and reused; started from a named snapshot template, with env-server as a systemd service. Replace with a fresh VM after about 20 groups or 6 hours. Since workers are long-lived, boat's launch limit (600 per hour on the $2000 tier) is not a problem.
- **Containers**: one container per rollout, cgroup-limited to 1 core 2 GB (limits, not reservations), `--network none`, pids-limited, unprivileged. One machine hosts several groups, possibly from different miners; groups of the same task are placed on the same machine where possible, sharing one copy of the image.
- **How many per machine**: 8 containers without oversubscription; containers spend most of their time waiting for model generation, so an estimated 16-24 fit (to be measured: CPU share of tool execution with a real model).
- **env-server** reuses the environment part of the EXP-014 harness: `setup_task` (git history truncation, restoring environment variables missing from the image, anti_hack_cleanup), `run_bash` (`bash -lc`, default 60 s, max 300 s, output over 40,000 characters truncated in the middle), `grade`. New parts are tool-call parsing and chat template rendering (moved from the harness's model part), plus the trajectory hash and receipt.
- **Grading** follows Xiaomi: restore test files -> `git apply test_patch` -> run `test_command`; exit code 0 scores 1; a patch that fails to apply is an infra error. `test_patch` is sent by the gateway to the worker only at grading time and deleted after use.
- **External access**: env-server listens on localhost and is exposed as a public URL via `boat host` (boat proxy about 50 ms); it cannot go through boat's exec API for every step (about 1.2 s). The egress firewall allows only the registry cache and the gateway; systemd re-adds the rules at boot (boat snapshots do not preserve firewall rules).
- **Execution log**: each step records the command, output hash, text returned to the model, returncode and duration, written to the operator's log storage; used only for debugging and statistics, verification does not depend on it.
- **Which environments boat can host**: the largest size is `xlarge` (16 vCPU / 32 GB / 251 GB, requires the $100 tier); there is nothing larger, no attachable disks and no GPUs. Environments that need a GPU, or whose per-task image plus workspace exceeds about 200 GB, need other machines. Each environment gets its own network policy (SWE has no network), timeouts and quotas.

## 7. Image distribution and caching
- An image on disk is about 3x its compressed size (EXP-015 measured 3 images, 2.5-3.5x); Xiaomi SWE images have a median of about 9-10 GB, a max of about 40 GB, about 20 TB in total.
- On one machine an image is stored only once and shared by any number of containers; each container adds only its own writable layer (measured: go 172 MB, python 0.5 MB). So "how many copies of an image" means how many machines cache it, and the replica count follows demand: popular tasks are replicated to more machines, cold tasks can have 0 copies and be pulled on use.
- One `large`, after the workspace, can cache about 10 SWE images; 800 machines give about 8,000 slots, so the full 2,698 images average about 3 copies, with little margin. More environments and larger total image size mean more misses. Grouping workers by environment mitigates this.
- Images cannot be stored in boat snapshots: snapshots do not preserve registry images, and saving them as tar is no faster than pulling directly (EXP-015). A self-hosted registry cache is required, placed in the EU (ideally in OVH together with boat), about 6.7 TB of compressed layers in total. Pull speed from the cache to workers has not been measured; direct pulls from Docker Hub have a median of 23 MB/s.

## 8. Protocol for one rollout
Control plane (gateway, hotkey-signed):
1. `POST /tasks/open {epoch, env_id}` -> gateway checks quota, assigns a task, picks a worker -> `{task_id, worker_url}`.
2. `POST /sessions/open {epoch, task_id, nonces: [0..7]}` -> one `{session_id, token}` per nonce.
3. `POST /tasks/finish {epoch, task_id}` (submit or abandon).

Data plane (worker, session token):
1. `POST /start` -> the worker starts the container, runs setup, and returns the task prompt tokens (source P).
2. Loop: the miner samples from P up to the first stop token -> `POST /turn {tokens}` (source M) -> the worker parses the tool call, executes it, renders the output -> returns `{tool_tokens, done}` (source T) -> the miner appends `tool_tokens` to the context verbatim and continues sampling.
3. A turn in which the model calls no tool means submit; running out of steps, context or time is recorded as unresolved. The worker grades and returns `{outcome, reward, receipt}`.
4. A single rollout with no request for 15 minutes of idleness is closed and recorded as `idle_timeout`.

The miner uploads the group's 8 rollouts through the existing complete-group flow; the group object additionally carries the 8 receipts and each rollout's source segments.

## 9. Receipts and verification
Implemented in `src/hosted_env/infra/receipt.py` and `src/hosted_env/infra/verify.py` (tests: `tests/test_receipt.py`).

- **Trajectory hash**: the worker incrementally hashes `[(source, token ids)...]`. Each segment is encoded as 1 byte of source + segment length u32 + each token id as u32 (little-endian); it must start with P, and adjacent segments must have different sources.
- **Receipt content**: epoch, hotkey, env_id, task, nonce, trajectory hash, token count, outcome (completed / unresolved / infra_error), reward, worker key id, issue time, the environment field set (`env_spec`) and what the reward depends on (the served task row's `task_hash`, model and sandbox context, the environment's grading fields), signed with the worker's Ed25519 key. Version 3; the frozen field tables and versioning rules are in [formats.md](formats.md).
- **Worker public keys**: registered with the gateway at boot, rotated every 24 hours, and signed into each epoch's opening by the operator authority. Validators already trust the authority.
- **Verification**: one `verify_receipt` call per rollout:
  1. The signature is valid and the key is in this epoch's trusted list;
  2. The epoch, hotkey, environment, task and nonce in the receipt match this rollout;
  3. Recomputing the hash from the uploaded tokens and source segments matches the receipt;
  4. Once it passes, outcome and reward are taken from the receipt, and the "at least one pass and one finished-but-wrong" gate is also computed from receipts.

  Model tokens are verified with the sampling proof as before. If any one check fails, the group scores 0.
- **What it blocks** (covered by unit tests): altering any tool output, model or task prompt token; relabeling a tool-output segment as model-generated; truncating or appending tokens; pairing in a receipt from someone else, another task, another nonce or another epoch; altering the reward; the miner signing receipts with a self-generated key. Rollouts the miner runs on its own machine get no receipt.
- **Why step-by-step checking is no longer needed**: the commands are parsed by the worker itself from model tokens, and the tool-output tokens are generated by the worker; both are in the hash. If the miner stuffs some other tool output into its own context, the uploaded trajectory will not match the receipt hash; if it uploads the trajectory matching the receipt, the subsequent tokens were sampled under a different context, so the sampling proof fails.
- **Source segments double as the loss mask**: only M segments take part in training.

## 10. Integration with the existing flow (upstream changes)
| Location | Change |
|---|---|
| opening | New environment entry: gateway URL, worker public key list, quotas, group size, timeouts, network policy |
| group object | Each rollout additionally carries source segments and a receipt |
| group validation (admission / audit) | For the SWE environment, one `verify_receipt` call per rollout |
| trainer ingest | One `verify_receipt` call; use the source segments as the loss mask |
| miner client | On reaching a stop token, hand off to the worker, take the returned tokens and keep sampling; no longer parses tool calls or assembles tool outputs itself |
| Unchanged | Upload links, commitments, conditional writes, freeze, audit job dispatch, reward computation |

## 11. Capacity and cost (boat list prices, 2026-10-15)
200 active hotkeys, computed from concurrent in-flight containers and how many containers each `large` hosts:

| Scenario | In-flight containers | 8 per machine (no oversubscription) | 16 per machine | 24 per machine |
|---|---|---|---|---|
| Average 4 tasks running per hotkey (guessed typical value) | 6,400 | 800 machines, $58/h (about $42k/month) | 400 machines, $29/h (about $21k/month) | 267 machines, $19/h (about $14k/month) |
| All 9 tasks occupied per hotkey (quota ceiling) | 14,400 | 1,800 machines, $130/h (about $95k/month) | 900 machines, $65/h (about $47k/month) | 600 machines, $43/h (about $32k/month) |

- For comparison: 200 miner H200s cost about $600/h (lium about $3/h).
- A single boat $2000 tier allows at most 2,000 concurrent machines; beyond that, buy organization seats (limits and bonus machine hours stack, and the plan fee is fully converted into machine hours) or negotiate with boat. EU capacity must be requested in advance (otherwise `out_of_capacity` is reported).
- Other costs: gateway + Postgres, the registry cache machine (about 7-8 TB of disk, price not checked), log storage; all far smaller than the workers.
- Alternative "one `default` per rollout" (4 cores 8G, same as Xiaomi's k8s spec): the typical scenario needs 6,400 machines at $230/h, 4x the cost of 8 per `large`, and the 8 machines of a group each pull the image once. Only worth considering if grading results are sensitive to machine spec.

## 12. Security boundary (not handled in this version, hardened later)
- **Container escape**: the commands in a container are decided by the miner's model, so they are untrusted input by nature. Escaping to the VM yields the worker signing key and the ability to forge receipts. The plan is to harden the container runtime with something like gVisor; whether Xiaomi images run under gVisor and how much slower compilation gets is to be measured.
- **Trusting the operator**: validators must trust that the operator's workers do not misbehave. This is the same root of trust as today's trust in the operator's trainer and audit.
- **Task prompt leakage**: Xiaomi's `test_patch` is public data, so miners can see the tests locally; but the context is constructed by the worker and the miner cannot inject anything, so the impact is limited to local pre-screening of tasks.

## 13. Decided and open
**Decided** (2026-10-15)
- Approach 3: operator-hosted environment running on boat.dev; no TEE, replay or deposit.
- Miners keep uploading to the bucket, workers only issue receipts, verification is a pure function.
- Per hotkey: 9 tasks running concurrently, 20 started per epoch, 10 concurrent per task, each nonce opened only once.
- One `large` hosts multiple containers with cgroup core limits; images replicated on demand.
- Every epoch opens all available tasks; no restriction on environment or task type.
- Anti-forgery hardening comes later.
- Fair share: the global budget is split equally per hotkey for now; weighting (e.g. by stake or past quality) is revisited in P2.

**Open**
- Gateway assignment vs. miner choice: this document is designed around gateway assignment. If it changes to miner choice, collisions are left to current rules, but cache hit rates get worse and miners can pick tasks.
- The value of the global budget B.
- What to do after an infra error (worker crash, image pull failure): currently nonces cannot be reopened, which would unfairly penalize the miner; the leaning is to void the group, not count it toward "20 per epoch", and reassign the task.
- Whether trajectories that do not finish (unresolved) should be fully masked in training: the Xiaomi recipe does not mask (scored 0, gradients as usual), while math currently masks the whole trajectory.
- The number of rewarded groups per UID per epoch for SWE.
- Whether the 30-minute collection per round is enough to finish an SWE group, or whether a separate cadence or a cross-epoch window is needed.

## 14. Measurement basis (EXP-015, boat trial, 2026-10-15)
| Item | Result |
|---|---|
| Boot | 5-6 s (5.9 s from snapshot); Ryzen 9 9950X, OVH France |
| Grading consistency | Pilot 20 tasks matched the lium pod task by task (20/20), fake-model smoke test 3/3 |
| Image pull | Docker Hub single image median 23 MB/s (lium 14) |
| Latency | boat proxy about 50 ms; exec API about 1.2 s; one command inside a container 33-72 ms |
| Memory / CPU | Idle container about 22 MiB; grading peak python <=300 MiB, go about 1.1 GiB, go compilation saturates 4 cores |
| Image disk usage | About 3.1x compressed size; multiple containers from the same image add 0 disk, writable layer go 172 MB, python 0.5 MB |
| Not measured | Steps per rollout, group duration and CPU share of tool execution with a real model; miner-to-EU latency; registry cache to worker speed; 9B pass rate on SWE |

## 15. Phases
| Phase | Content | Pass criteria | Cost |
|---|---|---|---|
| P0 | Single-machine env-server: split the EXP-014 harness into a worker (start / turn / grading / receipt) and a minimal miner client, run on one boat `default`, driven locally by a fake model | smoke3 results match EXP-014; receipts pass `verify_receipt`; altering one token is caught | boat trial credit |
| P1 | Real model end to end: rent one GPU to run MiMo, environment on boat | Pilot 20 tasks x 8 run through; obtain steps, group duration, CPU share, latency, pass rate | About $5-10 GPU, needs approval |
| P2 | gateway (quotas, task assignment, global budget) + multiple workers + registry cache + verification integration | 50 concurrent groups stable; all verification passes; tampering is caught | About $50-100, needs approval |
| P3 | Proposal to const; small pilot with external miners | Decided by const | |

## Appendix: code
- `src/hosted_env/infra/auth.py`: hotkey signature verification, session tokens.
- `src/hosted_env/infra/gateway/`: public HTTP edge (signatures, replay protection, routing; FastAPI).
- `src/hosted_env/infra/coordinator/`: quotas, task assignment, global budget / fair share, worker registry (SQLite).
- `src/hosted_env/infra/receipt.py`: trajectory hash, receipt signing, `verify_receipt`, `loss_mask`.
- `src/hosted_env/infra/verify.py`: `verify_group`.
- `src/hosted_env/infra/executor/` (formerly `worker/`): token rendering, tool execution, grading, receipt signing.
- `src/hosted_env/infra/providers/sandbox/`: one container per rollout (Docker, local for tests).
- `src/hosted_env/envs/swe/`: Xiaomi MiMo SWE task type, ported from the internal EXP-014 harness (`env.py` SweEnv,
  `cleanup.py`, `protect.py`, `scripts.py`, `reports.py` the second check, `forgery.py` the patch audit and the
  canary re-run, `baseline.py` expected tests,
  `receipt.py` the fields its receipts bind, `cli.py` the `baseline` command). How to add another task type:
  [adding-an-environment.md](adding-an-environment.md).
- `src/hosted_env/envs/images.py`: pinning task images by digest.
- `src/hosted_env/infra/client.py`: reference miner.
- Tests: `uv run pytest`.
- The original prototypes lived in internal experiment EXP-015.
