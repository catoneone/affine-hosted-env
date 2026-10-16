"""What every sandbox implementation provides (architecture v2, SandboxProvider).

A sandbox is one isolated container for one rollout. The executor only uses this interface, so the same rollout
logic runs on a local directory (tests), a Docker host (dev) or a Kubernetes Pod (production).
"""
from __future__ import annotations

from typing import Protocol

OUTPUT_CAP_BYTES = 4_000_000   # collect at most 4MB of output per command and drop the rest (the env truncates by characters later)


class SandboxError(RuntimeError):
    """The sandbox itself is broken (fails to start, docker errors): recorded as an infra error, not the
    model's fault."""


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
