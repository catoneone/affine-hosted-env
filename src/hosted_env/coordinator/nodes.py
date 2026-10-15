"""Node capacity from Kubernetes (architecture v2): how many sandboxes each node can take and which task images
it already has, read from node status instead of worker heartbeats.

A node counts only if it is Ready, schedulable and has no NoSchedule/NoExecute taint (control-plane nodes are
skipped unless asked). Slots = how many standard sandboxes (cpu, memory) fit into the node's allocatable resources
after a reserve for system pods. Cached images come from `node.status.images`; note the kubelet reports at most 50
images per node by default (`nodeStatusMaxImages`), so on a node with a big cache this is a lower bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_BIN = {'Ki': 2 ** 10, 'Mi': 2 ** 20, 'Gi': 2 ** 30, 'Ti': 2 ** 40, 'Pi': 2 ** 50}
_DEC = {'k': 10 ** 3, 'M': 10 ** 6, 'G': 10 ** 9, 'T': 10 ** 12, 'P': 10 ** 15}
CONTROL_PLANE_TAINTS = ('node-role.kubernetes.io/control-plane', 'node-role.kubernetes.io/master')


@dataclass(frozen=True)
class NodeInfo:
    name: str
    slots: int
    images: tuple[str, ...] = ()


def cpu_cores(q) -> float:
    """Kubernetes CPU quantity ('4', '3500m', '0.5') to cores."""
    q = str(q).strip()
    return float(q[:-1]) / 1000 if q.endswith('m') else float(q)


def mem_bytes(q) -> int:
    """Kubernetes or Docker-style memory quantity ('16Gi', '8G', '2g', '512Mi', '1048576') to bytes."""
    q = str(q).strip()
    for suf, mul in _BIN.items():
        if q.endswith(suf):
            return int(float(q[:-2]) * mul)
    if q and q[-1] in _DEC:
        return int(float(q[:-1]) * _DEC[q[-1]])
    docker = {'m': 2 ** 20, 'g': 2 ** 30, 't': 2 ** 40}      # docker style ('512m', '2g') means binary units
    if q and q[-1] in docker:
        return int(float(q[:-1]) * docker[q[-1]])
    return int(float(q))


def image_names(names) -> set[str]:
    """Names a node reports for one image, plus the forms task files use (with and without `docker.io/`)."""
    out = set()
    for n in names or ():
        out.add(n)
        if n.startswith('docker.io/'):
            out.add(n[len('docker.io/'):])
        elif '/' in n and '.' not in n.split('/', 1)[0] and ':' not in n.split('/', 1)[0]:
            out.add('docker.io/' + n)
    return out


def _usable(node, include_control_plane: bool) -> bool:
    if getattr(node.spec, 'unschedulable', False):
        return False
    for t in getattr(node.spec, 'taints', None) or []:
        if t.effect in ('NoSchedule', 'NoExecute'):
            if include_control_plane and t.key in CONTROL_PLANE_TAINTS:
                continue
            return False
    ready = [c for c in (node.status.conditions or []) if c.type == 'Ready']
    return bool(ready) and ready[0].status == 'True'


def node_info(node, *, cpus='1', memory='2g', reserve_cpu='500m', reserve_memory='1Gi',
              include_control_plane: bool = True) -> NodeInfo | None:
    if not _usable(node, include_control_plane):
        return None
    alloc = node.status.allocatable or {}
    cpu = cpu_cores(alloc.get('cpu', '0')) - cpu_cores(reserve_cpu)
    mem = mem_bytes(alloc.get('memory', '0')) - mem_bytes(reserve_memory)
    slots = max(0, min(math.floor(cpu / cpu_cores(cpus) + 1e-9), mem // mem_bytes(memory)))
    images = set()
    for img in node.status.images or []:
        images |= image_names(img.names)
    return NodeInfo(node.metadata.name, int(slots), tuple(sorted(images)))


def k8s_nodes(core_api, **kw) -> list[NodeInfo]:
    """Usable nodes of the cluster. `kw` as in `node_info`."""
    return [n for n in (node_info(x, **kw) for x in core_api.list_node().items) if n is not None]
