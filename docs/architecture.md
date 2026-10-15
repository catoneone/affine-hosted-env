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

### Gateway
- Verifies hotkey request signatures and replay ids (`auth.py`) and checks the epoch roster.
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
- Reads capacity from Kubernetes instead of worker heartbeats: node allocatable CPU and memory, sandbox Pods
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
| `SandboxProvider` | one isolated container per rollout: `start`, `execute`, `copy_text_to`, `cleanup` | `k8s` (default), `docker` (single-host dev), `local` (tests); later vendor sandboxes such as Targon Sandbox (template-based, cannot run arbitrary task images) |
| `MachineProvider` | whole machines: create, list, delete, spec, price, limits; a boot script that joins the cluster | `boat`, `targon` (VM), `static` (machines registered by hand) |

The `k8s` sandbox: one Pod per rollout, `runtimeClassName: gvisor`, a default-deny NetworkPolicy, CPU and memory
limits, no service-account token, and the same environment repair as `DockerSandbox` (`ENV_PROBE`). Without that
repair Go tasks cannot find their module cache and try to download it.

## Cluster
- k3s. The control plane runs on a stable dual-stack host. Avoid a boat sandbox for this role: it was IPv6-only
  for inbound traffic and gets replaced. Avoid a Targon VM on SHARED networking: its ports are remapped, it has no
  hairpin route to its own public address, and it has no IPv6.
- Nodes only need outbound access: the k3s agent tunnels kubelet traffic over its own connection.
- Pod network: flannel `wireguard-native` (encrypted, 25 s keepalive from each node), so only the control plane
  opens a UDP port.
- gVisor: `runsc` from the gVisor apt repository plus a containerd drop-in (`config-v3.toml.d/runsc.toml`), and a
  `RuntimeClass` named `gvisor`.
- Boot scripts re-apply firewall rules on every boot, because boat snapshots do not keep them.

## Measured (EXP-017, 2026-10-15)
| Item | Result |
|---|---|
| Install | k3s v1.36.5 server Ready in 12.7 s, agent joined in about 13 s (boat `default`) |
| Node replacement | a new boat machine took 5.8 s from the warm pool and 4.5 min cold; joining took about 15 s |
| Task image Pod | 422 MB Xiaomi image pulled in 13 s, Pod Ready in 15 s; under 1 s once cached |
| Exec latency | persistent client through the API server: p50 27-42 ms (runc), +10-30 ms with gVisor, 184 ms cross-region |
| gVisor grading | Go task 002119: same test verdicts as runc, 27.5 s vs 23.9 s (+15%) |
| Isolation | default-deny NetworkPolicy blocks egress for runc and gVisor Pods |
| Targon VM | AMD SEV-SNP guest; joins as a node; as control plane it needs two workarounds (advertised address plus local hairpin forwarder, WireGuard port equal to the public mapping) |

## Migration plan
Each step is one PR with tests green:

1. **Package split** (done). `gateway/` (HTTP edge), `coordinator/` (today's store), `executor/` (today's worker
   core), `providers/sandbox/` (today's docker and local sandboxes). Behavior unchanged.
2. **Gateway** (done). Rate limits (`gateway/ratelimit.py`) and data-plane routing to an in-process executor
   (`hosted-env server`); the split mode with a separate executor process still works.
3. **`k8s` SandboxProvider** (done). `providers/sandbox/k8s.py`; Kubernetes Python client as the `k8s` extra;
   unit tests against a fake API (`tests/test_k8s_sandbox.py`); an opt-in live test against a kubeconfig
   (`tests/test_k8s_live.py`). Select it per env with `sandbox = "k8s"`.
4. **Coordinator capacity from Kubernetes** (done). `coordinator/nodes.py` reads node allocatable resources and
   cached images; `Store(placement='nodes')` reserves each task's group on one node and voids tasks of lost nodes;
   the gateway syncs nodes every 10 s; the executor passes the node to the Pod as `nodeName`. Enabled by
   `[placement] kind = "k8s"`.
5. **MachineProviders** (in progress). Interface, `static` and `boat` done (`providers/machine/`); next `targon`
   (vendor APIs, tested against fakes), the node boot script (k3s agent, WireGuard, gVisor, firewall), then the
   capacity planner with hard caps on machine count and hourly spend.
6. **Live P0** on a small cluster. Needs a decision on the control-plane host.

## Open
- Which host runs the control plane (provider, region, dual-stack).
- Multi-region: one executor and node pool per region, plus routing miners to the nearest region.
- Whether grading can always run with egress denied (expected yes once `ENV_PROBE` is applied; to be confirmed on
  the pilot tasks).
