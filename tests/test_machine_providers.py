"""providers/machine: data types and the static provider."""
import pytest

from hosted_env.providers.machine import Machine, MachineSpec, NotSupported, StaticProvider


def test_spec_requires_prefix():
    assert MachineSpec('large', 'hosted-env-n1').ttl_seconds is None
    with pytest.raises(ValueError, match='hosted-env-'):
        MachineSpec('large', 'someone-elses-vm')


def test_machine_state_checked():
    Machine('m1', 'boat', 'hosted-env-m1', 'running')
    with pytest.raises(ValueError):
        Machine('m1', 'boat', 'hosted-env-m1', 'exploded')


def test_static_provider_counts_but_never_creates():
    p = StaticProvider([{'name': 'gpu-box', 'public_ipv4': '203.0.113.7', 'price_per_hour': 0.5}, {'name': 'b'}])
    ms = p.list()
    assert [(m.id, m.state, m.provider, m.kind) for m in ms] == [('gpu-box', 'running', 'static', 'static'),
                                                                  ('b', 'running', 'static', 'static')]
    assert ms[0].public_ipv4 == '203.0.113.7' and p.offers() == []
    with pytest.raises(NotSupported):
        p.create(MachineSpec('x', 'hosted-env-x'), '#!/bin/sh')
    with pytest.raises(NotSupported):
        p.delete('gpu-box')
