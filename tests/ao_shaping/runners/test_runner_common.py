"""Tests for shared runner helpers (resolve_dm).

``resolve_dm`` now lives in :mod:`ao_shaping.drivers.dm._registry`; the
tests patch the registry module directly so they keep working whether
callers import ``resolve_dm`` from there or re-export it via
``runner_common``.
"""

from __future__ import annotations

import pytest

from ao_shaping.drivers.dm import _registry as dm_registry
from ao_shaping.runners import runner_common


class TestResolveDm:
    """DM auto-detection / explicit-type resolution shared by runners."""

    def test_resolve_dm_explicit(self, monkeypatch):
        """Explicit dm_type is lowercased and routed to create_dm with kwargs."""
        fake_dm = object()
        calls: dict = {}

        def fake_create_dm(name, **kwargs):
            calls["name"] = name
            calls["kwargs"] = kwargs
            return fake_dm

        monkeypatch.setattr(dm_registry, "create_dm", fake_create_dm)

        result = runner_common.resolve_dm("NLight", keep_when_exit=True)

        assert result is fake_dm
        assert calls["name"] == "nlight"
        assert calls["kwargs"] == {"keep_when_exit": True}

    def test_resolve_dm_auto_single(self, monkeypatch):
        """Auto-detection picks the single reachable DM type."""
        calls: dict = {}

        def fake_create_dm(name, **kwargs):
            calls["name"] = name
            return "dm-instance"

        monkeypatch.setattr(dm_registry, "list_reachable_dm_types", lambda: ["slm"])
        monkeypatch.setattr(dm_registry, "create_dm", fake_create_dm)

        result = runner_common.resolve_dm(None)

        assert result == "dm-instance"
        assert calls["name"] == "slm"

    def test_resolve_dm_no_dm_raises(self, monkeypatch):
        """No reachable DM raises RuntimeError with the shared message."""
        monkeypatch.setattr(dm_registry, "list_reachable_dm_types", lambda: [])

        with pytest.raises(RuntimeError, match="No DM reachable"):
            runner_common.resolve_dm(None)

    def test_resolve_dm_multi_raises(self, monkeypatch):
        """Multiple reachable DMs raise RuntimeError listing the candidates."""
        monkeypatch.setattr(
            dm_registry, "list_reachable_dm_types", lambda: ["nlight", "micro"]
        )

        with pytest.raises(RuntimeError, match="Multiple DMs reachable"):
            runner_common.resolve_dm(None)