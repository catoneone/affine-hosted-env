# Architecture v2: layered services on Kubernetes

- Date: 2026-10-15. Status: agreed direction, implementation in progress (see the migration plan below).
- Evidence: internal experiment EXP-017 (k3s on boat and Targon machines), summarized in "Measured" below.
- Supersedes the deployment parts of [design.md](design.md) §5-§7 (one long-lived worker VM per env-server). The
  protocol, receipts and quotas in design.md are unchanged.

## Layers

```text
miner GPU
   |  HTTPS: hotkey-signed control plane, session-token data plane
   v
+--------------------------- control-plane process (one process for now) ---------------------------+
| gateway      auth (hotkey signatures, replay, roster), rate limits, routing; the only public surface  |
| coordinator  epochs, task pool, assignment, quotas, global budget + fair share, sessions, placement   |
| executor     rollout state machine, token rendering, tool calls, grading, receipt signing             |
+-----------------------------------------------------------------------------------------------------+
   |  Kubernetes API (pods, exec, node status)              |  vendor APIs (create / delete machines)
   v                                                         v
k3s cluster: one sandbox Pod per rollout                MachineProvider: boat, targon, static
(gVisor runtime, egress denied, resource limits)        each machine boots and joins the cluster
```

## Infrastructure and environments
The code is split in two (since 2026-10-18):

- `hosted_env/infra/`: everything above plus the providers, receipts and verification. It runs any task type and
  knows none: no task fields, no grading rules, no anti-cheat.
- `hosted_env/envs/`: the task types (`docker_task`, `mimo_swe` in `envs/swe/`). Each owns its task pool filters,
  prompt, tools, container setup, grading and anti-cheat, the extra receipt fields it binds (`RECEIPT`, an
  `EnvReceiptSpec`), the attachment artifacts, and its own CLI commands.

The seams are small: the coordinator calls `task_ids` / `image_of` / `public_spec` / `grade_inputs`; the executor
calls `open`, the session's `setup` / `call_tool` / `grade`, then `receipt_context`, `log_fields` and
`artifacts`; `sign_receipt` takes the environment's spec and stamps its id in the receipt (`env_spec`), and
`verify_receipt` / `verify_group` find the spec by that id among the registered ones or take a pinned one
(`hosted_env.envs.receipt_spec(kind)`; formats in [formats.md](formats.md)). An exception out of `setup` or
`grade` voids the rollout, so deciding
what is the model's fault (score 0) is the environment's job. `tests/test_boundary.py` fails if infra imports a
concrete environment or names SWE concepts. How to add one: [adding-an-environment.md](adding-an-environment.md).

### Gateway
- Verifies hotkey request signatures and replay ids (`auth.py`) and checks the epoch roster.
- Roster = SN120 registrations + a hand-edited allowlist, both hot-reloaded. `hosted-env roster-sync --every 300`
  writes the chain part with upstream's `registrations()` rule (a UID on netuid 120 plus a revealed
  `affine2|activate` commitment signed by the hotkey's Ed25519 key, owner excluded); the allowlist (`roster`) sets
  the epoch and adds test keys, and cannot re-key a registered hotkey. A sync that would drop more than half the
  hotkeys is refused and the previous file kept.
- Rate limits with token buckets: per source IP before auth, per hotkey after auth, and globally. Over the limit
  returns 429 with `Retry-After`. Buckets live in process memory for now behind an interface, so a shared store
  can replace them when the gateway runs as several replicas.
- Routes the control plane (`tasks/open`, `sessions/open`, `tasks/finish`) to the coordinator and the data plane
  (`start`, `turn`) to the executor. Nothing below the gateway is reachable from the internet.

Rate limiting answers "how fast"; quotas (9 in flight, 20 per epoch, 10 concurrent per task) answer "how much"
and stay in the coordinator as business rules.

### Coordinator
- Keeps today's store logic: task pool, assignment, quotas, global budget with per-hotkey fair share (equal split
  per hotkey; weighting is a P2 decision), sessions, voiding and refunds.
- Reads capacity from Kubernetes instead of worker heartbeats: node allocatable CPU, memory and ephemeral storage
  (less a reserve for the image cache), sandbox Pods
  already placed, and the images each node has cached (`node.status.images`), so placement stays cache-aware.
- Places each sandbox Pod itself (`nodeName`). The kubelet refuses a Pod that does not fit (`OutOfcpu`,
  `OutOfmemory`) instead of queueing it, so the coordinator must not over-commit a node.
- Decides liveness with its own timers. Pod phase lags: after a node disappeared its Pods still showed `Running`
  for minutes (EXP-017).
- Later: a capacity planner turns demand into a desired node count and calls the MachineProviders.

### Executor (formerly the worker)
- Same logic as the current `worker` package: rollout state machine, append-only token rendering, tool-call
  parsing, grading, receipts, idempotent long-poll turns.
- Talks to sandboxes only through `SandboxProvider`.
- Runs in the control-plane process, so the receipt signing key never lives on a sandbox node: a container escape
  on a node cannot forge receipts.
- Must run in the same region as its nodes. Exec through the Kubernetes API costs p50 27-42 ms in region but
  184 ms across the Atlantic (EXP-017). Several regions later means one executor per region.

### Providers
Two separate interfaces, because a machine and a sandbox are different products:

| Interface | Gives | Implementations |
|---|---|---|
| `SandboxProvider` | one isolated container per rollout: `start`, `execute`, `copy_text_to`, `extend_env`, `cleanup` | `k8s` (default), `docker` (single-host dev), `local` (tests); later vendor sandboxes such as Targon Sandbox (template-based, cannot run arbitrary task images) |
| `MachineProvider` | whole machines: create, list, delete, spec, price, limits; a boot script that joins the cluster | `boat`, `targon` (VM), `static` (machines registered by hand) |

The `k8s` sandbox: one Pod per rollout, CPU, memory and ephemeral-storage limits (`disk`, default 10g), no
service-account token. The MiMo task images lost their ENV (single-layer tarballs), so the environment restores
it after every sandbox start through the generic `extend_env` (`envs.base.IMAGE_ENV_PROBE`, applied by
`DockerTaskEnv.prepare_sandbox` to the rollout and the grading container). Without that repair Go tasks cannot
find their module cache and try to download it. Every command in every sandbox also runs with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`MKL_NUM_THREADS` set to 1 (`THREAD_CAPS`): a 1-CPU sandbox still sees all host cores, so numpy-heavy parallel
tests otherwise start a thread per host core, hit the pid limit, and a correct fix grades 0 (Affine measured 22.7%
of correct solutions overall, 96% on a 32-core box). Offline sandboxes are denied the network twice: the `gvisor-nonet` RuntimeClass (runsc
`--network=none`, loopback only, checked after start; a sandbox with any other interface is refused) and a
default-deny NetworkPolicy. The policy alone is not enough: it is enforced asynchronously, and on a freshly joined
node the first Pod reached the internet (EXP-018).

A sandbox killed for its own limits (OOMKilled, or evicted for its own ephemeral-storage usage) scores the
rollout 0 (`sandbox_limit`, `rollout_sandbox_limit` or `grader_sandbox_limit`) instead of an infra error, so a
model cannot void a rollout it expects to fail by filling the memory or the disk. A node-pressure eviction or any
other sandbox failure is still an infra error. Under gVisor both limits are enforced (EXP-018): an over-limit
allocation is SIGKILLed (exit 137) and the container OOMKilled; an over-limit write is evicted ~15-20 s later.
A command that cannot be launched because the model deleted a binary exec needs (e.g. /bin/sh) is likewise the
model's own doing, not an infra error. This attribution assumes gVisor, where an overrun OOMKills the whole
container; under runc an OOM may kill only the offending process.

## Cluster
- k3s. The control plane runs on a stable dual-stack host. Avoid a boat sandbox for this role: it was IPv6-only
  for inbound traffic and gets replaced. Avoid a Targon VM on SHARED networking: its ports are remapped, it has no
  hairpin route to its own public address, and it has no IPv6.
- Nodes only need outbound access: the k3s agent tunnels kubelet traffic over its own connection.
- Pod network: flannel `wireguard-native` (encrypted, 25 s keepalive from each node), so only the control plane
  opens a UDP port.
- gVisor: `runsc` from the gVisor apt repository plus a containerd drop-in (`config-v3.toml.d/runsc.toml`) with two
  runtimes, `runsc` and `runsc-nonet` (`--network=none`), and RuntimeClasses `gvisor` and `gvisor-nonet`. Both
  runsc runtimes set `signal-unkillable-policy=linux`, so the rollout container's PID 1 (`sleep infinity`) is
  signal-protected and the model, root with CAP_KILL, cannot `kill -9 1` to kill the container and void its
  rollout. `K8sSandbox.start()` probes PID 1 (SIGSTOP, read state, SIGCONT) and refuses the sandbox if the
  protection is not in force, so a node missing the config fails closed instead of letting the void through.
- Boot scripts re-apply firewall rules on every boot, because boat snapshots do not keep them.

## Measured (EXP-017, 2026-10-15)
| Item | Result |
|---|---|
| Install | k3s v1.36.5 server Ready in 12.7 s, agent joined in about 13 s (boat `default`) |
| Node replacement | a new boat machine took 5.8 s from the warm pool and 4.5 min cold; joining took about 15 s |
| Task image Pod | 422 MB Xiaomi image pulled in 13 s, Pod Ready in 15 s; under 1 s once cached |
| Exec latency | persistent client through the API server: p50 27-42 ms (runc), +10-30 ms with gVisor, 184 ms cross-region |
| gVisor grading | Go task 002119: same test verdicts as runc, 27.5 s vs 23.9 s (+15%) |
| Isolation | default-deny NetworkPolicy blocks egress for runc and gVisor Pods, but asynchronously: on a node that had just joined, the first Pod reached the internet (EXP-018). With `gvisor-nonet` the sandbox has loopback only (`Network is unreachable` for the internet, Service and Pod CIDRs), including the first Pod on a fresh node; a Pod with any other interface is refused at start (EXP-018) |
| Offline grading | Go task 002119 under `gvisor-nonet` with the image env repair (then `ENV_PROBE`, now `IMAGE_ENV_PROBE`): 32.4 s on 2 cores, same 9 FAILs as with network, no module downloads (EXP-018) |
| Targon VM | AMD SEV-SNP guest; joins as a node; as control plane it needs two workarounds (advertised address plus local hairpin forwarder, WireGuard port equal to the public mapping) |

## Migration plan
Each step is one PR with tests green:

1. **Package split** (done). `gateway/` (HTTP edge), `coordinator/` (today's store), `executor/` (today's worker
   core), `providers/sandbox/` (today's docker and local sandboxes). Behavior unchanged. All of them now live
   under `hosted_env/infra/` (see "Infrastructure and environments").
2. **Gateway** (done). Rate limits (`gateway/ratelimit.py`) and data-plane routing to an in-process executor
   (`hosted-env server`); the split mode with a separate executor process still works.
3. **`k8s` SandboxProvider** (done). `providers/sandbox/k8s.py`; Kubernetes Python client as the `k8s` extra;
   unit tests against a fake API (`tests/test_k8s_sandbox.py`); an opt-in live test against a kubeconfig
   (`tests/test_k8s_live.py`). Select it per env with `sandbox = "k8s"`.
4. **Coordinator capacity from Kubernetes** (done). `coordinator/nodes.py` reads node allocatable resources and
   cached images; `Store(placement='nodes')` reserves each task's group on one node and voids tasks of lost nodes;
   the gateway syncs nodes every 10 s; the executor passes the node to the Pod as `nodeName`. Enabled by
   `[placement] kind = "k8s"`.
5. **MachineProviders** (done, against fakes). `providers/machine/`: interface, `static`, `boat`, `targon`, and the
   node boot script (`bootstrap.py`: firewall, gVisor, WireGuard port, pinned k3s agent). `coordinator/planner.py`
   turns waiting demand into machines and deletes idle ones, bounded by `max_machines`, `max_hourly_usd` and
   `max_create_per_round`, and runs in dry-run mode unless told otherwise. It runs inside the server process
   (`[machines]` in the config, a background round every `interval` seconds, plans logged when they change) and
   never deletes a machine it did not create.
   Machine inventory: every machine is kept in the Store's `machines` table, with history. The planner records
   its machines as soon as the provider assigns an id (before the slow boot, so a crash mid-create cannot forget a
   billed machine); a failed create is cleaned up by the provider, which deletes only the machine it was creating,
   and if that delete fails the machine is marked `orphaned` and the planner retries it every round. Every
   `inventory_interval` seconds (read only, also in dry-run) each provider is listed: `hosted-env-` machines the
   planner did not create (the control plane, leftovers) are recorded as `unmanaged` and logged, never deleted;
   ones that disappear become `gone`. A provider list that misses one of the planner's own machines never makes it
   forget the machine. `hosted-env machines --config <file> [--all] [--refresh]` prints the inventory with hours
   and an estimated cost at list prices.
6. **Live P0** on a small cluster. Needs a decision on the control-plane host.

## Open
- Which host runs the control plane (provider, region, dual-stack).
- Multi-region: one executor and node pool per region, plus routing miners to the nearest region.
- Whether grading can always run with egress denied (expected yes with the image environment repair applied;
  to be confirmed on the pilot tasks).
