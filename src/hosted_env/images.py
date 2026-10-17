"""Pin task images to content digests, so every container of a task (rollout, grading, a validator's offline
regrade) runs the exact same bytes: a tag can be moved, `name@sha256:...` cannot.

`pin_tasks` rewrites a task file's `image` from `name:tag` to `name@sha256:<digest>` (keeping the tag as
`image_tag`), asking the registry for the manifest digest with a HEAD request (Docker Registry HTTP API v2,
anonymous token for Docker Hub). Already pinned rows are left alone. Envs refuse unpinned images by default
(`require_image_digest`, default true); set it to false only for local development with tag images.
"""
from __future__ import annotations

import json
import re

import httpx

MANIFEST_TYPES = ', '.join((
    'application/vnd.oci.image.index.v1+json', 'application/vnd.docker.distribution.manifest.list.v2+json',
    'application/vnd.oci.image.manifest.v1+json', 'application/vnd.docker.distribution.manifest.v2+json'))
DIGEST_RE = re.compile(r'^sha256:[0-9a-f]{64}$')


class ImageError(RuntimeError):
    pass


def is_pinned(image: str) -> bool:
    return '@' in image and bool(DIGEST_RE.match(image.rsplit('@', 1)[1]))


def split_ref(image: str) -> tuple[str, str, str]:
    """(registry, repository, tag) for a `name[:tag]` reference; Docker Hub names are normalized."""
    name, tag = image, 'latest'
    last = image.rsplit('/', 1)[-1]
    if ':' in last:
        name, tag = image.rsplit(':', 1)
    first = name.split('/', 1)[0]
    if '/' in name and ('.' in first or ':' in first or first == 'localhost'):
        registry, repo = name.split('/', 1)
    else:
        registry, repo = 'registry-1.docker.io', name if '/' in name else f'library/{name}'
    if registry == 'docker.io':
        registry = 'registry-1.docker.io'
    return registry, repo, tag


def resolve_digest(image: str, http: httpx.Client | None = None) -> str:
    """The registry's content digest for `image` (`name:tag`)."""
    registry, repo, tag = split_ref(image)
    http = http or httpx.Client(timeout=30, follow_redirects=True)
    headers = {'Accept': MANIFEST_TYPES}
    if registry == 'registry-1.docker.io':
        t = http.get('https://auth.docker.io/token',
                     params={'service': 'registry.docker.io', 'scope': f'repository:{repo}:pull'})
        if t.status_code != 200:
            raise ImageError(f'{image}: token request failed ({t.status_code})')
        headers['Authorization'] = f'Bearer {t.json()["token"]}'
    r = http.head(f'https://{registry}/v2/{repo}/manifests/{tag}', headers=headers)
    digest = r.headers.get('docker-content-digest', '')
    if r.status_code != 200 or not DIGEST_RE.match(digest):
        raise ImageError(f'{image}: no manifest digest ({r.status_code})')
    return digest


def pinned_ref(image: str, digest: str) -> str:
    name = image.rsplit(':', 1)[0] if ':' in image.rsplit('/', 1)[-1] else image
    return f'{name}@{digest}'


def pin_tasks(lines: list[str], resolve=resolve_digest) -> tuple[list[str], dict]:
    """Rewrite task-file lines; returns (lines, {'pinned': n, 'already': n}). Each distinct tag is resolved once."""
    cache: dict[str, str] = {}
    out, stats = [], {'pinned': 0, 'already': 0}
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        image = row['image']
        if is_pinned(image):
            stats['already'] += 1
        else:
            if image not in cache:
                cache[image] = resolve(image)
            row['image_tag'], row['image'] = image, pinned_ref(image, cache[image])
            stats['pinned'] += 1
        out.append(json.dumps(row, ensure_ascii=False))
    return out, stats
