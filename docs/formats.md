# Wire formats

The formats a validator or trainer reads, frozen as of package version 0.2.0 (git tag `v0.2.0`) and unchanged in
version 0.2.1 (`v0.2.1`: grader `isolated-v4`, `hosted-env regrade` with the additive
`verify_group(require_gate=False)`, the reference miner's clean stop), so `verify_receipt` / `verify_group` of
either accept receipts and attachments signed by either (`hosted-env regrade` still refuses another grader version
unless `--any-grader`). Pin a tag, not a commit. Code: `hosted_env.receipt` and `hosted_env.verify` (stable names
for `hosted_env.infra.receipt` and `hosted_env.infra.verify`), PyNaCl and the standard library only.

| Object | Version | Signed by | Domain prefix |
|---|---|---|---|
| receipt | `v: 3` | worker key | `affine-hosted-env-receipt-v3\|` |
| environment field set | `env_spec`: `swe/1`, `docker_task/1` | (inside the receipt) | |
| group seal | `v: 1` | gateway key | `affine-hosted-env-group-seal-v1\|` |
| group attachment | `version: "hosted-env-group-v3"` | (unsigned container) | |
| trajectory hash | | | `affine-hosted-env-trajectory-v1` |

Keys come from the epoch's opening / the gateway's `/v1/info`: `workers[*].{key_id, public_key}` and
`gateway_public_key` (Ed25519, hex).

## Compatibility rules

- A version never gains or loses a field. Receipt v2 did (three fields were added while it ran on the test
  deployment, so older v2 receipts failed); that is why v3 exists.
- An environment's field set changes only by a new `env_spec` version (`swe/2`). The old one stays registered, so
  receipts signed under it keep verifying.
- Only a change to the core fields or payload keys below bumps the receipt version (and its domain).
- A receipt of another version is refused with `unsupported_receipt_version`; v2 and v1 are not accepted.

## Encodings

- **Signed bytes** (receipt, seal): canonical JSON, `json.dumps(obj, sort_keys=True, separators=(',', ':'),
  allow_nan=False).encode()` (`ensure_ascii` left at its default). Signature: Ed25519 over `domain prefix +
  bytes`. On the wire, `{"payload": b64url(bytes), "sig": b64url(sig)}` (URL-safe base64, no padding). A verifier
  checks the signature over the payload bytes as received; it never re-encodes them.
- **task_hash**: verifiers' `task_key` -- `sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()` with the
  **default** separators and `ensure_ascii`, over the task-index row as the gateway serves it, with `None`-valued
  keys dropped (as `model_dump(exclude_none=True)` drops them). `hosted_env.receipt.task_hash(row)`. It equals
  verifiers' `Task.hash` when the task's data model dumps to exactly this row (TaskData's own defaults, such as
  `network_allow`, are not in our rows).
- **Trajectory hash**: `sha256(domain + for each segment: source byte ('P', 'M', 'T') + u32le(len) + u32le(id)...)`;
  starts with `P`, adjacent segments differ, no empty segment. `trajectory_sha256(segments)`.

## Receipt (v3)

Flat signed payload, exactly these keys (an extra or missing key is `malformed_receipt`):

| Key | Type | Meaning |
|---|---|---|
| `v` | int, `3` | receipt version |
| `env_spec` | str | the environment field set, e.g. `swe/1` |
| `epoch`, `hotkey`, `env_id`, `task` | str | identity; must match the caller's |
| `nonce` | int | rollout index in the group, `0..group_size-1` |
| `trajectory_sha256` / `n_tokens` | str / int | over the uploaded segments |
| `outcome` | str | `completed`, `unresolved` or `infra_error` |
| `reward` | float | the environment's grade |
| `key_id` | str | which worker key signed |
| `issued_at_ms` | int | worker clock, never before `started_at_ms` |

Core context fields, the same for every environment:

| Field | Type | Meaning |
|---|---|---|
| `task_hash` | 64 hex, never `''` | task_hash of the served task row |
| `sandbox_limits` | str | e.g. `k8s:cpus=1;memory=2g;disk=10g;threads=1`; a re-grade on other limits can differ |
| `model_id`, `model_revision` | str | as the worker is configured (not a proof; the sampling proof is) |
| `tokenizer_sha256`, `chat_template_sha256` | 64 hex or `''` | what the worker rendered with |
| `renderer` | str | renderer kind and library version |
| `executor_version` | str | package version + commit, e.g. `0.2.0+g1234567890ab` |
| `started_at_ms` | int | worker clock |

Every `*_sha256` field is lowercase hex of 64 characters or `''` (not known, e.g. never graded).

### `swe/1`

| Field | Type | Meaning |
|---|---|---|
| `image` | str | `name@sha256:...` the rollout and its grading ran in |
| `start_tree` | 40 hex or `''` | git tree the model started from |
| `test_patch_sha256` | hex | of the hidden test patch as served (an amended JS patch for an amended row) |
| `test_command_sha256` | hex | of the test command |
| `grader` | str | grading semantics, e.g. `isolated-v5` (patch audit + canary and consistency re-runs of passes; no field change) |
| `grader_parsers` | str | pinned parsers for structured manifests |
| `expected_tests_sha256` | hex or `''` | `expected_tests_digest(fail_to_pass, pass_to_pass, report_kind)`; `''` = exit code only |
| `grader_user` | str | `nobody` or `root` |
| `model_patch_sha256` | hex or `''` | of the patch as captured; the attachment carries it |
| `verifier_returncode` | int or null | null when no verifier ran |

Artifact: `model_patch` (bound by `model_patch_sha256`, at most 2,000,000 bytes).

### `docker_task/1`

`image` (str), `test_command_sha256` (hex), `verifier_returncode` (int or null). No artifacts. Exit-code graded in
the model's own container: a fixture, not something to pay rewards on.

## Group seal (v1)

Signed payload: `v` (1), `epoch`, `hotkey`, `env_id`, `task`, `task_id`, `group_size`, `opened_at_ms`,
`finished_at_ms` (gateway clock), `receipts` (list, by nonce, of `sha256(receipt payload bytes)` as the worker
reported it, or null).

## Group attachment (`hosted-env-group-v3`)

```json
{"version": "hosted-env-group-v3", "env_id": "...", "task": "...", "seal": {"payload": "...", "sig": "..."},
 "rollouts": [{"nonce": 0, "segments": [{"src": "P", "ids": [1, 2]}], "receipt": {"payload": "...", "sig": "..."},
               "artifacts": {"model_patch": "<standard base64>"}}]}
```

`artifacts` holds exactly the spec's artifacts, each hashing to its receipt field (empty when that field is `''`).

## Task rows

The receipt's `task_hash` covers every field of the served row, including what `hosted-env baseline` derived:
`fail_to_pass`, `pass_to_pass`, `report_kind`, `source_task_hash` (the `task_hash` of the source row, before the
baseline added anything), and for a JS task whose grading script was amended to write a report, the amended
`test_patch` / `test_command` plus

```json
"amended_from": {"task_hash": "<task_hash of the source row>", "rule": "js-report-flags-v1", "fields": ["test_patch"]}
```

`hosted_env.envs.swe.baseline.check_source(source_row, served_row)` is a provenance check: it returns `None` only
when the served row is the source row with just those fields added (expected tests present, `source_task_hash`
naming the source row, every other field unchanged). For an amended row it also re-runs the deterministic amendment on the source row
(`check_amendment(source_row, served_row)`) and requires exactly that change (reporter flags on the one jest /
vitest / mocha run). These are fields of the task index, not of the receipt: the receipt format is unchanged.
Re-merging a derived index re-hashes a non-amended row from its own fields; an amended row keeps the hash in
`amended_from`, so change the source tasks and derive again rather than hand-editing an amended row.

## Verifying

```python
from hosted_env.receipt import task_hash
from hosted_env.verify import verify_group

v = verify_group(attachment, trusted_keys, gateway_key=gw_key, epoch=..., hotkey=..., env_id=..., task=...,
                 group_size=8, expect={'task_hash': task_hash(my_copy_of_the_row)})
```

`spec=` is optional: omitted, each receipt's `env_spec` is looked up among the registered specs (importing
`hosted_env` registers them); passed (`hosted_env.envs.receipt_spec(kind)`), it must be the spec the receipt
names. A group has one `env_spec`.

| Error code | Meaning |
|---|---|
| `malformed_receipt` / `malformed_seal` / `malformed_group` / `malformed_trajectory` | not decodable or not the format above |
| `unsupported_receipt_version` | `v` is not 3 |
| `untrusted_worker_key` / `bad_signature` | key not in the opening / signature wrong |
| `unknown_env_spec` / `env_spec_mismatch` | env_spec not registered / not the pinned spec, or mixed in one group |
| `identity_mismatch` / `seal_identity_mismatch` | another epoch, hotkey, env, task or nonce |
| `context_mismatch` | an `expect` value differs |
| `trajectory_mismatch` | uploaded tokens are not the ones that ran |
| `bad_seal_signature` / `seal_mismatch` | seal not the gateway's / not this receipt |
| `artifact_mismatch` | an artifact is not what the receipt says was graded |
| `bad_attachment_version` / `incomplete_group` | not group-v3 / a nonce missing or repeated |
| `infra_error_in_group` / `group_gate_failed` | any infra error voids the group / not one pass and one fail |

The receipt is evidence for the verifier to re-check, not a trusted reward: the hidden tests run in the same
process as the code under test, which can always produce the test framework's output.

To re-check it, `hosted-env regrade --attachment <file> --gateway <url> --tasks <your task index> --config <cfg>`
runs `verify_group` with `expect` = the `task_hash` and `image` of your own task row and `require_gate=False`
(every receipt and artifact is checked; only the group-level pass/fail and infra-error rules are skipped, so such a
group is still re-graded), refuses a receipt whose `grader` is not this code's or whose `sandbox_limits` differ from
what your config enforces (`--any-grader`, `--any-limits` to override), grades as the receipts' `grader_user`, then
grades every graded rollout's verified `model_patch` again in fresh containers from the receipt's `start_tree` (the
same pipeline, canary included) and prints the receipt's reward next to the re-grade's. Exit 0 when all match, 2 on
a mismatch, 1 on a refusal (nothing is graded then). Pass `--hotkey` / `--epoch` to catch a group replayed under
another miner or epoch; by default they are the receipts' own.

The built-in canary edits are public, so a forge can be written against them (the module docstring of
`hosted_env/envs/swe/forgery.py` says which ones a task-free scan beats). `--canary-edit module:function` (or
`path/to/file.py:function`) re-grades with your own edit instead, in both canary runs: it is called as
`edit(path, source, test_id)` once per test function the canary would edit (`test_id` is its first expected id)
and returns the file's source with that test -- every expected case of it -- edited to fail and nothing else
changed, or `None` to leave the test out. Any other result (not a str, the source unchanged, a `.py` file that no
longer parses, an exception) is reported as that rollout's `error`, never graded. An edit you do not publish is the
one a forge has not seen; the summary line names the edit used (`canary_edit`).
