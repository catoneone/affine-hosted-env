"""SandboxProvider implementations (architecture v2): one isolated container per rollout.

- k8s: one Pod per rollout, gVisor runtime; offline sandboxes run with gVisor networking off (`gvisor-nonet`,
  checked after start) plus a default-deny NetworkPolicy (needs the `k8s` extra).
- docker: one container per rollout on a Docker host, plus the per-host image cache.
- local: tests and local demos only, no isolation.
"""
from .base import OUTPUT_CAP_BYTES, Sandbox, SandboxError, SandboxCommandError, SandboxLimitError
from .docker import ENV_PROBE, DockerSandbox, ImageCache
from .k8s import K8sSandbox
from .local import LocalSandbox

__all__ = ['ENV_PROBE', 'OUTPUT_CAP_BYTES', 'DockerSandbox', 'ImageCache', 'K8sSandbox', 'LocalSandbox', 'Sandbox',
           'SandboxError', 'SandboxCommandError', 'SandboxLimitError']
