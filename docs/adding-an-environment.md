# Adding an environment

An environment is a task type: where its tasks come from, what the model sees, which tools it has, and how a
rollout is graded and protected against cheating. The infrastructure runs it without knowing what it is. This
page is the contract between the two, using the SWE environment (`hosted_env/envs/swe/`) as the worked example.

## Who owns what

| | Infrastructure (`hosted_env/infra/`) | Environment (`hosted_env/envs/<name>/`) |
|---|---|---|
| Miners | auth (hotkey signatures, replay ids, roster), rate limits, quotas, task assignment, sessions | |
| Sandboxes | one container per rollout and its placement; CPU, memory, disk, pid and network limits; thread caps; gVisor; exec | which image, which working directory; what the image needs fixed after start (`prepare_sandbox`) |
| Rollout | token rendering, tool-call parsing, step / context / time limits, the trajectory hash | the task prompt, the tools and their output, setup of the container |
| Grading | calls `grade` once, turns an exception into a voided rollout (`infra_error`) | the grade itself, and deciding which failures are the model's (score 0) and which are not (raise) |
| Anti-cheat | the model cannot reach the receipt key, the network or another rollout | everything about the task: leaked solutions, tampered tests, exit-code tricks, protected files |
| Receipts | signing, core fields (`CORE_FIELDS`), the group attachment, `verify_receipt` / `verify_group` | the extra fields its receipts bind (`RECEIPT`) and their values, attachment artifacts |

`tests/test_boundary.py` keeps the left column honest: no module under `infra/` imports anything below
`hosted_env.envs` (the registry is allowed), and infra code names none of SWE's concepts. A new environment is
added without touching `infra/`; if it seems to need an infra change, that change must be generic (as
`Sandbox.extend_env` was for the image repair).

## The class

Subclass `DockerTaskEnv` (`envs/base.py`) and override what differs. The constructor takes `(env_id, cfg)`,
where `cfg` is the env's `[envs.<env_id>]` table from the server config.

```python
class MyEnv(DockerTaskEnv):
    kind = 'my_env'          # the config's `kind`, the name in /v1/info, and how verifiers pick the receipt spec
    RECEIPT = MY_SPEC        # see "Receipts" below
```

Coordinator side (the task pool; no sandbox involved):

| Member | Used for |
|---|---|
| `task_ids()` | the pool. Filter what must not be offered in `_load` (which `task_ids`, `image_of`, `public_spec` and `grade_inputs` all read): the base drops unpinned images (`require_image_digest`) and keeps only the epoch's opening (`indices`); SWE also drops tasks without expected tests (`require_expected_tests`) |
| `image_of(task)` | cache-aware placement |
| `public_spec(task)` | what the executor gets when a rollout opens: `image`, `cwd`, the prompt material. **Nothing that grades** |
| `grade_inputs(task)` | the grading material (hidden tests, expected results), sent to the executor only when it grades |
| `group_size`, `success_reward` | the group contract; a reward at or above `success_reward` counts as a pass in the group gate (published in `/v1/info`) |

Executor side, for one rollout:

| Member | Called |
|---|---|
| `open(spec, factory)` | returns an `EnvSession` on a fresh sandbox; `factory(image, cwd)` makes more sandboxes on the same node (SWE's grading container) |
| `prepare_sandbox(sb)` | right after every sandbox the environment starts. The default runs `IMAGE_ENV_PROBE` through `sb.extend_env` (restores GOPATH, JAVA_HOME, PATH, ... that the single-layer MiMo images lost); override it for images that need something else, or set `repair_image_env = false` |
| `setup(session)` | after `prepare_sandbox`, before the model sees the container: strip history, delete residue, record the start state |
| `initial_messages(spec)`, `session.tools` | the prompt; rendered and hashed by the infrastructure |
| `session.call_tool(name, args)` | each tool call; returns the text the model sees. The default is one `bash` tool |
| `grade(session, grade_inputs)` | once, at the end. Returns a dict with at least `reward` |
| `receipt_context`, `log_fields`, `artifacts` | after grading; see "Receipts" |

## Who is to blame: score 0 or void

Any exception out of `setup` or `grade` voids the rollout (`infra_error`): the miner gets the slot back. Rewards
are an incentive, so this is where most care goes. Two rules:

1. **A model that expects to fail must not be able to void.** Whatever its commands can cause -- a full disk, a
   killed process, a deleted `git` or `/bin/sh`, a broken repo -- must come back as `reward: 0` with a reason
   (SWE: `grade_detail`), never as an exception. During turns the infrastructure already does this for the
   sandbox's own limits (`SandboxLimitError`, `SandboxCommandError` -> `unresolved`, 0). Inside `grade` the
   environment does it: SWE keeps an `untrusted` flag that turns on when the model's bytes first run in the
   grading container; before that a limit or launch failure raises `GradeError`, after it it scores 0.
2. **A correct solution must not score 0 because of us.** A test patch that does not apply, a grading image
   that cannot start, a probe that fails: raise `GradeError` (or `SetupError` during setup), so it is voided
   and visible, not silently charged to the model.

## Anti-cheat belongs to the environment

The infrastructure guarantees that tool output really came from the sandbox and that the model cannot reach the
network or the signing key. Whether the reward means "solved the task" is the environment's job. SWE's answers,
as a checklist of questions a new environment has to answer for itself:

| Question | SWE |
|---|---|
| Can the solution leak into the container? | `setup` strips git history to the base commit and deletes build output, caches and residue (`cleanup.py`) |
| Can the model edit the grader? | grading runs in a **fresh container** from the same image, with only the model's patch carried over; the hidden test patch is applied only there |
| Can the patch change what the tests mean? | protected files (test files, runner configs, shadow modules, test-relevant manifest fields) are put back first (`protect.py`) |
| Can the code tamper with the results? | the tests run as an unprivileged user (`grader_user`, default `nobody`), with the test files root-owned and read-only |
| Is "exit 0" enough? | no: a second check reads the structured test report and requires every expected test to pass (`reports.py`, expected tests from `hosted-env baseline`) |
| Can the code under test forge the report? | always, in principle: it runs in the framework's process. `forgery.py` makes the cheap forms fail: a patch audit (no added test-framework output or report plumbing in kept files) and, for pytest / go test, two canary re-runs of every pass: each expected test edited to fail, then one picked at random (it must fail, the others must still pass). The pytest edit negates the test's own first assertion; Go's (and pytest's fallback) inserts a constant comparison, which static analysis of the test source finds, and any edit is found by diffing against the public original tests. So the receipt is evidence to re-verify with an edit the verifier does not publish |
| Are the tasks the same bytes later? | images pinned by digest (`hosted-env pin-images`), so a re-grade runs what was graded |

`docker_task` (the base class) answers none of these: it grades with an exit code in the model's own container.
It is the minimal example and a test fixture, not something to pay rewards on.

## Receipts

A receipt binds the infrastructure's core fields plus the fields the environment declares:

```python
MY_SPEC = register_spec(EnvReceiptSpec(
    id='my_env/1',                                                 # signed in every receipt as env_spec
    fields={'image': str, 'answer_sha256': str, 'checker': str},   # all required, typed, signed flat
    hex_lengths={},                                                # non-*_sha256 hex fields and their length
    artifacts={'answer': ('answer_sha256', 64 * 1024)},            # attachment name -> (its sha256 field, max bytes)
))
```

- `id` is `<name>/<version>`. `register_spec` adds the spec to `receipt.SPECS`, where a verifier that passes no
  `spec=` finds it by the receipt's `env_spec`; it refuses a different spec under an id already registered.
- `fields` must not reuse a core or payload key (the constructor refuses). `*_sha256` fields must be lowercase
  hex of 64 characters or `''`.
- `receipt_context(session, gi, grade)` returns exactly those fields for one rollout, from the session (its
  `spec`, and `state` that `setup` filled, e.g. SWE's `start_tree`), the grade inputs and the grade. `session` is
  `None` and `gi` / `grade` are `{}` when the rollout never got that far, so every value needs a default.
- `artifacts(grade)` returns `{name: base64}` for each declared artifact; the miner's attachment carries them, and
  `verify_group` checks each against its sha256 field and size, then returns them. SWE ships the graded patch
  (`model_patch`), which is what makes an offline re-grade possible.
- `log_fields(grade)` picks small values for the executor's per-rollout log line (never test output).
- Verifiers find the spec by `env_spec` or call `receipt_spec(kind)` (kind from the gateway's `/v1/info`), and
  `hosted_env` imports the registry so the specs are registered: the spec must import cheaply. Keep it in its own
  module with standard-library imports only (`envs/swe/receipt.py`). A receipt checked with another environment's
  spec fails in both directions (`env_spec_mismatch`, and the exact field set behind it).
- Changing a spec's fields is a new spec version (`my_env/2`): keep the old spec registered so receipts signed
  under it still verify, and never change the fields under an existing id. Pin each version with a test
  (`test_swe_receipt_fields_unchanged`). The rules are in [formats.md](formats.md).

## Registering it

1. Put the package under `hosted_env/envs/<name>/` and add the class to `KINDS` in `hosted_env/envs/__init__.py`.
2. Environment-specific commands (`baseline` for SWE) go in the package and are added to `CLI_EXTENSIONS`; each
   entry gets the CLI's subparsers. Import heavy dependencies inside the command, not at module top:
   `import hosted_env.envs` must stay light for validators (`test_importing_the_registry_stays_light`).
3. Configure it:

   ```toml
   [envs.my_env]
   kind = "my_env"
   tasks = "my_tasks.jsonl"
   group_size = 8
   sandbox = "k8s"
   ```

   The whole table reaches the class as `cfg`; the executor-side keys (`sandbox`, `sandbox_opts`, `renderer`,
   context and step limits) are shown in `examples/server.toml`, and an environment's own keys (SWE:
   `grader_user`, `require_expected_tests`, ...) are read in its class.

## Tests to write

- Grading on `LocalSandbox` end to end: an honest solution scores 1, a no-op scores 0 (see `tests/test_envs.py`).
- Every cheat you know of scores 0, and every model-caused failure in `grade` scores 0 rather than raising.
- The receipt fields: the spec is pinned, `receipt_context` fills exactly the spec on a graded, an ungraded and a
  failed-setup rollout, and a receipt signed with it verifies (`tests/test_env_contract.py`).
- `tests/test_boundary.py` still passes without being edited.
