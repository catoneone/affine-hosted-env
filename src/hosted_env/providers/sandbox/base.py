"""What every sandbox implementation provides (architecture v2, SandboxProvider).

A sandbox is one isolated container for one rollout. The executor only uses this interface, so the same rollout
logic runs on a local directory (tests), a Docker host (dev) or a Kubernetes Pod (production).
"""
from __future__ import annotations

from typing import Protocol

OUTPUT_CAP_BYTES = 4_000_000   # collect at most 4MB of output per command and drop the rest (the env truncates by characters later)

# Every command in every sandbox (the model's tool calls and the grading run alike) gets these. A sandbox has one
# CPU but sees the host's cores, so numpy / scipy / torch start one OpenBLAS / OpenMP / MKL thread per HOST core;
# parallel tests then hit the pid limit and a correct fix grades 0 -- and since a container's own limit kill now
# scores 0 (PR #15) rather than voiding, that is a wrong 0, not a retry. Measured by Affine on Targon: 22.7% of
# correct solutions overall, 96% on a 32-core box (Arbos, 2026-10-18). A task script that sets its own value
# still wins (it runs after these are in the environment). Recorded in the receipt's sandbox_limits.
THREAD_CAPS = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


class SandboxError(RuntimeError):
    """The sandbox itself is broken (fails to start, docker errors): recorded as an infra error, not the
    model's fault."""


class SandboxLimitError(SandboxError):
    """The sandbox was killed for going over a limit set on it alone (memory: OOMKilled; ephemeral storage: evicted
    for its own usage). Only the commands run inside it could cause that, so a rollout scores 0 instead of being
    voided as an infra error; otherwise a model could void any rollout it expects to fail by filling the disk or
    the memory. A node-wide eviction (the node low on a resource) is not this: it stays a plain SandboxError."""


class SandboxCommandError(SandboxError):
    """A command could not be LAUNCHED in the container -- the OCI runtime refused to start it, e.g. its executable
    was deleted. The container and the transport are fine; only this command never ran. After the sandbox is
    handed out this is the caller's own doing (a model that is root in its rollout container can delete /bin/sh),
    so it scores the rollout 0 rather than voiding it. Before hand-out it means the image is broken (infra)."""


class Sandbox(Protocol):
    def start(self) -> None:
        """Create the container and make it ready for commands."""

    def execute(self, command: str, cwd: str | None = None, timeout: int = 60, raw: bool = False,
                login: bool = True) -> dict:
        """Run a shell command. Returns {output, returncode, reason}; reason is ok / client_timeout.
        login=True runs `bash -lc` (what the model's tool calls get); login=False runs
        `bash --noprofile --norc -c`, so no profile or rc file the model could have edited is read (grading).
        Environment variables captured from the pristine image at start are applied either way."""

    def copy_text_to(self, text: str, remote_path: str) -> None:
        """Write a text file inside the container."""

    def cleanup(self) -> None:
        """Destroy the container. Safe to call more than once."""
