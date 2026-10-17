"""images.py: pinning task images by digest (fake registry), and envs refusing unpinned images."""
import json

import httpx
import pytest

from hosted_env.images import ImageError, is_pinned, pin_tasks, pinned_ref, resolve_digest, split_ref

D1 = 'sha256:' + 'a' * 64


def test_split_and_pinned_refs():
    assert split_ref('xiaomimimo/mimo-v2.6-rl-oss:format-code-task-002371') == (
        'registry-1.docker.io', 'xiaomimimo/mimo-v2.6-rl-oss', 'format-code-task-002371')
    assert split_ref('docker.io/library/bash:5.2.37') == ('registry-1.docker.io', 'library/bash', '5.2.37')
    assert split_ref('bash') == ('registry-1.docker.io', 'library/bash', 'latest')
    assert split_ref('ghcr.io/org/img:v1') == ('ghcr.io', 'org/img', 'v1')
    assert split_ref('localhost:5000/img:v1') == ('localhost:5000', 'img', 'v1')
    assert pinned_ref('docker.io/x/y:tag', D1) == f'docker.io/x/y@{D1}'
    assert pinned_ref('localhost:5000/img', D1) == f'localhost:5000/img@{D1}'
    assert is_pinned(f'x/y@{D1}') and not is_pinned('x/y:tag') and not is_pinned('x/y@sha256:short')


def fake_registry(digest=D1, status=200):
    seen = []

    def handle(request):
        seen.append(request)
        if request.url.host == 'auth.docker.io':
            assert request.url.params['scope'] == 'repository:x/y:pull'
            return httpx.Response(200, json={'token': 'T'})
        assert request.method == 'HEAD' and request.url.path == '/v2/x/y/manifests/tag'
        assert request.headers['authorization'] == 'Bearer T' and 'oci.image.index' in request.headers['accept']
        return httpx.Response(status, headers={'docker-content-digest': digest} if status == 200 else {})
    return httpx.Client(transport=httpx.MockTransport(handle)), seen


def test_resolve_digest_against_a_fake_registry():
    http, seen = fake_registry()
    assert resolve_digest('docker.io/x/y:tag', http=http) == D1 and len(seen) == 2
    with pytest.raises(ImageError, match='no manifest digest'):
        resolve_digest('x/y:tag', http=fake_registry(status=404)[0])
    with pytest.raises(ImageError):
        resolve_digest('x/y:tag', http=fake_registry(digest='sha256:nothex')[0])


def test_pin_tasks_resolves_each_tag_once_and_keeps_pinned_rows():
    calls = []

    def resolve(image):
        calls.append(image)
        return D1
    rows = [{'id': 1, 'image': 'x/y:tag'}, {'id': 2, 'image': 'x/y:tag'}, {'id': 3, 'image': f'x/z@{D1}'}]
    lines, stats = pin_tasks([json.dumps(r) for r in rows] + [''], resolve=resolve)
    out = [json.loads(x) for x in lines]
    assert stats == {'pinned': 2, 'already': 1} and calls == ['x/y:tag']
    assert out[0] == {'id': 1, 'image': f'x/y@{D1}', 'image_tag': 'x/y:tag'} and out[2] == rows[2]


def test_env_requires_pinned_images_by_default(tmp_path):
    from hosted_env.envs import SweEnv
    path = tmp_path / 't.jsonl'
    path.write_text(json.dumps({'id': 'a', 'image': 'x/y:tag', 'cwd': '/w', 'problem_statement': 'p',
                                'test_command': 'true', 'test_patch': ''}) + '\n')
    assert SweEnv('swe', {'tasks': str(path), 'require_image_digest': False}).task_ids() == ['a']   # explicit opt-out
    with pytest.raises(RuntimeError, match='not pinned by digest'):
        SweEnv('swe', {'tasks': str(path)}).task_ids()                                      # on by default
    with pytest.raises(RuntimeError, match='not pinned by digest'):
        SweEnv('swe', {'tasks': str(path), 'require_image_digest': True}).task_ids()
    path.write_text(path.read_text().replace('x/y:tag', f'x/y@{D1}'))
    assert SweEnv('swe', {'tasks': str(path)}).task_ids() == ['a']
    assert SweEnv('swe', {'tasks': str(path), 'require_image_digest': True}).task_ids() == ['a']
