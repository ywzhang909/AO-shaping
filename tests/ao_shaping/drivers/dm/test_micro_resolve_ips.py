from __future__ import annotations

import ao_shaping.drivers.dm.micro.wiring_map as wm_mod
from ao_shaping.drivers.dm.micro import DEFAULT_IPS, resolve_ips


def test_resolve_ips_with_user_list():
    assert resolve_ips(()) == list(DEFAULT_IPS)
    assert resolve_ips(("192.168.0.101",)) == ["192.168.0.101"]
    assert resolve_ips(("192.168.0.101", "192.168.0.102")) == [
        "192.168.0.101",
        "192.168.0.102",
    ]


def test_resolve_ips_fallbacks(monkeypatch):
    class DummyWM:
        unique_ips = ["192.168.0.101"]

    monkeypatch.setattr(wm_mod.WiringMap, "from_file", staticmethod(lambda _p: DummyWM()))
    assert resolve_ips(()) == ["192.168.0.101"]

    monkeypatch.setattr(wm_mod.WiringMap, "from_file", staticmethod(lambda _p: None))
    assert resolve_ips(()) == list(DEFAULT_IPS)

    def _raise(*a, **k):
        raise RuntimeError

    monkeypatch.setattr(wm_mod.WiringMap, "from_file", staticmethod(_raise))
    assert resolve_ips(()) == list(DEFAULT_IPS)

    class DummyWM2:
        unique_ips = []

    monkeypatch.setattr(wm_mod.WiringMap, "from_file", staticmethod(lambda _p: DummyWM2()))
    assert resolve_ips(()) == list(DEFAULT_IPS)
