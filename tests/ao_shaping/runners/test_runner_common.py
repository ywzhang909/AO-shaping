"""Tests for shared runner scaffolding in :mod:`ao_shaping.runners.runner_common`.

``resolve_dm`` lives in :mod:`ao_shaping.drivers.dm._registry`; the tests
patch the registry module directly so that the lookup of ``create_dm`` and
``list_reachable_dm_types`` is intercepted at call time regardless of how
callers import ``resolve_dm``.

``load_init_v`` / ``save_best_voltages`` / ``plot_pib_debug_grid`` are the
#91 P1 helpers shared by the four ``nlight_dm`` runners (wf / pipeline /
axis_beam / combined).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from loguru import logger as loguru_logger

from ao_shaping.drivers.dm import _registry as dm_registry
from ao_shaping.drivers.dm._registry import resolve_dm
from ao_shaping.runners import runner_common
from ao_shaping.runners.runner_common import (
    load_init_v,
    plot_pib_debug_grid,
    round_to_int,
    save_best_voltages,
    truncate_to_int,
)


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

        result = resolve_dm("NLight", keep_when_exit=True)

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

        result = resolve_dm(None)

        assert result == "dm-instance"
        assert calls["name"] == "slm"

    def test_resolve_dm_no_dm_raises(self, monkeypatch):
        """No reachable DM raises RuntimeError with the shared message."""
        monkeypatch.setattr(dm_registry, "list_reachable_dm_types", lambda: [])

        with pytest.raises(RuntimeError, match="No DM reachable"):
            resolve_dm(None)

    def test_resolve_dm_multi_raises(self, monkeypatch):
        """Multiple reachable DMs raise RuntimeError listing the candidates."""
        monkeypatch.setattr(
            dm_registry, "list_reachable_dm_types", lambda: ["nlight", "micro"]
        )

        with pytest.raises(RuntimeError, match="Multiple DMs reachable"):
            resolve_dm(None)


class _FakeRecorder:
    """Duck-typed Recorder stand-in: records every ``save_best`` call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def save_best(self, saved_dir=None, target=None, process_fn=None, **kwargs):
        self.calls.append(
            {
                "saved_dir": Path(saved_dir),
                "target": target,
                "process_fn": process_fn,
                **kwargs,
            }
        )


class TestLoadInitV:
    """``load_init_v``: exists-check + warning contract (#91 P1)."""

    def test_empty_token_returns_empty(self):
        assert load_init_v("") == []

    def test_missing_file_returns_empty_and_warns(self, tmp_path):
        records: list[str] = []
        handler_id = loguru_logger.add(
            lambda m: records.append(str(m)), level="WARNING"
        )
        try:
            result = load_init_v(str(tmp_path / "does_not_exist.txt"))
        finally:
            loguru_logger.remove(handler_id)
        assert result == []
        assert any("not exists" in m for m in records)

    def test_existing_file_loads_floats(self, tmp_path):
        path = tmp_path / "v.txt"
        np.savetxt(path, np.array([0.0, 1.0, -2.5]))
        assert load_init_v(str(path)) == [0.0, 1.0, -2.5]


class TestSaveBestVoltages:
    """``save_best_voltages``: single date-dir + target/fmt/process_fn contract."""

    def test_single_recorder_uses_round_to_int(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            runner_common, "get_date_dir_name", lambda: "20261020_000000"
        )
        rec = _FakeRecorder()

        out = save_best_voltages(rec, tmp_path)

        assert out == tmp_path / "flatten_voltages" / "20261020_000000"
        assert len(rec.calls) == 1
        call = rec.calls[0]
        assert call["saved_dir"] == out
        assert call["target"] == "_v"
        assert call["fmt"] == "%d"
        assert call["process_fn"] is round_to_int

    def test_dual_recorders_share_one_date_dir(self, tmp_path, monkeypatch):
        counter = {"n": 0}

        def fake_date() -> str:
            counter["n"] += 1
            return f"20261020_{counter['n']:06d}"

        monkeypatch.setattr(runner_common, "get_date_dir_name", fake_date)
        wf_rec, ccd_rec = _FakeRecorder(), _FakeRecorder()

        out = save_best_voltages(
            [wf_rec, ccd_rec], tmp_path, process_fn=truncate_to_int
        )

        assert counter["n"] == 1
        assert wf_rec.calls[0]["saved_dir"] == out
        assert ccd_rec.calls[0]["saved_dir"] == out
        assert wf_rec.calls[0]["process_fn"] is truncate_to_int
        assert ccd_rec.calls[0]["process_fn"] is truncate_to_int


class TestPlotPibDebugGrid:
    """``plot_pib_debug_grid``: writes a non-empty 2x2 summary PNG."""

    def test_writes_png(self, tmp_path):
        import matplotlib.pyplot as plt

        plt.switch_backend("Agg")
        out = tmp_path / "debug.png"

        plot_pib_debug_grid(
            init_img=np.random.rand(8, 8),
            best_img=np.random.rand(8, 8),
            history_values=np.linspace(0.1, 0.9, 5),
            voltages=np.linspace(-1.0, 1.0, 16),
            init_title="Init Image, pib=0.100",
            best_title="Best PIB Image, pib=0.900",
            history_title="PIB History",
            save_path=out,
        )

        assert out.exists()
        assert out.stat().st_size > 0
