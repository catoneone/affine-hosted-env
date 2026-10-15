"""SandboxProvider implementations (architecture v2): one isolated container per rollout.

- docker: one container per rollout on a Docker host, plus the per-host image cache.
- local: tests and local demos only, no isolation.
- k8s (planned): one Pod per rollout, gVisor runtime, egress denied.
"""
from .base import OUTPUT_CAP_BYTES, Sandbox, SandboxError
from .docker import ENV_PROBE, DockerSandbox, ImageCache
from .local import LocalSandbox

__all__ = ['ENV_PROBE', 'OUTPUT_CAP_BYTES', 'DockerSandbox', 'ImageCache', 'LocalSandbox', 'Sandbox', 'SandboxError']
