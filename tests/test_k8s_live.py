"""Opt-in test against a real cluster. Skipped unless HOSTED_ENV_KUBECONFIG points at a kubeconfig.

  HOSTED_ENV_KUBECONFIG=~/.kube/config HOSTED_ENV_K8S_RUNTIME_CLASS=gvisor uv run pytest tests/test_k8s_live.py

HOSTED_ENV_K8S_RUNTIME_CLASS defaults to gvisor; set it to an empty string on a cluster without gVisor.
HOSTED_ENV_K8S_IMAGE defaults to python:3.12-slim (needs bash and base64). The namespace must exist.
"""
import os

import pytest

KUBECONFIG = os.environ.get('HOSTED_ENV_KUBECONFIG')
pytestmark = pytest.mark.skipif(not KUBECONFIG, reason='HOSTED_ENV_KUBECONFIG not set')


def test_live_pod_sandbox():
    kubernetes = pytest.importorskip('kubernetes')
    from hosted_env.providers.sandbox.k8s import K8sSandbox
    api_client = kubernetes.config.new_client_from_config(config_file=os.path.expanduser(KUBECONFIG))
    core, net = kubernetes.client.CoreV1Api(api_client), kubernetes.client.NetworkingV1Api(api_client)
    sb = K8sSandbox(os.environ.get('HOSTED_ENV_K8S_IMAGE', 'python:3.12-slim'), '/tmp', api=core, networking_api=net,
                    namespace=os.environ.get('HOSTED_ENV_K8S_NAMESPACE', 'default'),
                    runtime_class=os.environ.get('HOSTED_ENV_K8S_RUNTIME_CLASS', 'gvisor') or None)
    try:
        sb.start()
        assert sb.execute('echo hi')['output'].strip() == 'hi'
        sb.copy_text_to('payload\n', '/tmp/x.txt')
        assert sb.execute('cat /tmp/x.txt')['output'] == 'payload\n'
        r = sb.execute('python3 -c "import socket; socket.create_connection((\'1.1.1.1\', 443), timeout=5)"')
        assert r['returncode'] != 0                                  # egress is denied
    finally:
        sb.cleanup()
