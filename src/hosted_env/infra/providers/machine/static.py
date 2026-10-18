"""Static machines: hosts registered by hand that already run the k3s agent. The planner can count them but never
creates or deletes them."""
from __future__ import annotations

from .base import Machine, NotSupported, Offer


class StaticProvider:
    name = 'static'

    def __init__(self, machines: list[dict]):
        """machines: [{name, kind?, public_ipv4?, public_ipv6?, price_per_hour?}], e.g. from [[machines.static]]."""
        self._machines = [Machine(id=m['name'], provider=self.name, name=m['name'], state='running',
                                  kind=m.get('kind', 'static'), public_ipv4=m.get('public_ipv4'),
                                  public_ipv6=m.get('public_ipv6'), price_per_hour=m.get('price_per_hour'))
                          for m in machines]

    def offers(self) -> list[Offer]:
        return []

    def list(self) -> list[Machine]:
        return list(self._machines)

    def create(self, spec, boot_script: str, on_created=None) -> Machine:
        raise NotSupported('static machines are registered by hand')

    def delete(self, machine_id: str) -> None:
        raise NotSupported('static machines are registered by hand')
