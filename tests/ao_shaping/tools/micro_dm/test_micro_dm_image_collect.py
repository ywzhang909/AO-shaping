from __future__ import annotations

import pytest

from ao_shaping.tools.micro_dm import micro_dm_image_collect as m
from ao_shaping.tools.micro_dm.micro_dm_image_collect import _resolve_ips


def test_resolve_ips_with_user_list():
    assert _resolve_ips(()) == list(m.DEFAULT_IPS)
    assert _resolve_ips(("192.168.0.101",)) == ["192.168.0.101"]
    assert _resolve_ips(("192.168.0.101", "192.168.0.102")) == ["192.168.0.101", "192.168.0.102"]


def test_resolve_ips_fallbacks(monkeypatch):
    import ao_shaping.tools.micro_dm.micro_dm_image_collect as mod

    class DummyWM:
        unique_ips = ["192.168.0.101"]

    monkeypatch.setattr(mod.WiringMap, "from_file", staticmethod(lambda _p: DummyWM()))
    assert _resolve_ips(()) == ["192.168.0.101"]

    monkeypatch.setattr(mod.WiringMap, "from_file", staticmethod(lambda _p: None))
    assert _resolve_ips(()) == list(m.DEFAULT_IPS)

    def _raise(*a, **k):
        raise RuntimeError

    monkeypatch.setattr(mod.WiringMap, "from_file", staticmethod(_raise))
    assert _resolve_ips(()) == list(m.DEFAULT_IPS)

    class DummyWM2:
        unique_ips = []

    monkeypatch.setattr(mod.WiringMap, "from_file", staticmethod(lambda _p: DummyWM2()))
    assert _resolve_ips(()) == list(m.DEFAULT_IPS)
