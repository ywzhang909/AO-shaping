"""Offline tests for the illuminated-radius probe (bench_settings).

Two properties matter here, and both were wrong in the first version:

1. **radius = far_edge − centre**, not ``far_edge / 2``. The sweep randomises a
   half-plane starting at the panel origin, so it reports the beam's *far* edge.
   Dividing by two assumes the illumination starts at the origin. Measured
   2026-10-17 on this bench: the x profile holds rel≈1.0 out to 600 px and only
   reaches the floor near 1290, i.e. the beam spans ~660–1290 — so ``edge/2``
   reported 450 px for a beam of radius ~320.

2. **the radius is the smaller of the two axes**, because the value feeds the
   forward model's ``aperture_m`` cross-check, where an overestimate makes the
   check vacuous.

No Streamlit server and no hardware: the SLM and camera are stubbed and
``sweep_axis`` is replaced, so only the radius arithmetic is exercised.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import streamlit as st

from ao_shaping.gui.slm import bench_settings as bs


@pytest.fixture(autouse=True)
def state(monkeypatch):
    """Patch ``st.session_state`` with a plain dict (bare-mode convention)."""
    state: dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", state)
    return state


class _FakeResult:
    def __init__(self, axis: str, edge: int, fall_lo: int | None = None) -> None:
        self.axis = axis
        self.edge_px = edge
        # Default to a plausible fall: onset 300px inboard of the far edge.
        self.fall_lo_px = fall_lo if fall_lo is not None else edge - 600
        self.fall_hi_px = edge
        self.centre_px = (self.fall_lo_px + edge) / 2.0
        self.radius_px = (edge - self.fall_lo_px) // 2
        self.step_px = 120.0
        self.floor_rel = 0.5
        self.profile_rel = (1.0, 0.5)
        self.flat_peak = 80.0
        self.flat_centroid = (1332.0, 962.0)


@pytest.fixture
def stub_sweep(monkeypatch):
    """Replace ``sweep_axis`` with a fixed per-axis edge, and stub the exposure."""

    edges = {"x": 1290, "y": 930}
    fall_lo: dict[str, int | None] = {"x": None, "y": None}

    def fake_sweep(**kwargs):
        axis = kwargs["axis"]
        return _FakeResult(axis, edges[axis], fall_lo=fall_lo[axis])

    monkeypatch.setattr(
        "ao_shaping.tools.slm.panel.beam_extent.sweep_axis", fake_sweep
    )
    monkeypatch.setattr(
        bs, "probe_session_exposure_ms", lambda *a, **k: (0.12, {"ceiling_ms": 0.24})
    )
    return SimpleNamespace(edges=edges, fall_lo=fall_lo)


class TestRadiusArithmetic:
    def test_radius_comes_from_the_scans_own_fall_width(
        self, stub_sweep, state, tmp_path
    ) -> None:
        """Centre-free: the fall's width is the beam, so no centre is needed."""
        out = bs.measure_illuminated_radius(1, "daheng", 0, debug_dir=tmp_path / "p")
        assert out["radius_source"] == "scan_fall_width"
        assert out["radius_px"] == pytest.approx(300.0)  # (1290-690)//2
        assert out["radius_uncertainty_px"] == pytest.approx(120.0)

    def test_two_different_centres_give_the_same_radius(
        self, stub_sweep, state, tmp_path
    ) -> None:
        """The regression that motivated the centre-free estimator.

        Measured 2026-10-17: the same scan gave 180 px with a previous session's
        centre and 300 px with the panel's geometric one -- a 1.7x swing that had
        nothing to do with the beam.
        """
        state["slm1_bench_pupil_center"] = "960,720"
        a = bs.measure_illuminated_radius(1, "daheng", 0, debug_dir=tmp_path / "a")
        state["slm1_bench_pupil_center"] = "960,600"
        b = bs.measure_illuminated_radius(1, "daheng", 0, debug_dir=tmp_path / "b")
        assert a["radius_px"] == b["radius_px"]

    def test_the_centre_subtracted_estimate_is_reported_alongside(
        self, stub_sweep, state, tmp_path
    ) -> None:
        """Both numbers, so a disagreement is visible instead of one silently
        replacing the other."""
        state["slm1_bench_pupil_center"] = "960,720"
        out = bs.measure_illuminated_radius(1, "daheng", 0, debug_dir=tmp_path / "p")
        assert out["centre_panel_px"] == [960.0, 720.0]
        assert out["radius_x_px"] == pytest.approx(330.0)   # 1290-960
        assert out["radius_y_px"] == pytest.approx(210.0)   # 930-720
        assert out["radius_from_centre_px"] == pytest.approx(210.0)
        assert out["radius_px"] == pytest.approx(300.0)     # scan-internal

    def test_no_centre_is_fine_because_the_scan_supplies_one(
        self, stub_sweep, state, tmp_path
    ) -> None:
        """Previously this refused; now the scan carries its own centre."""
        out = bs.measure_illuminated_radius(1, "daheng", 0, debug_dir=tmp_path / "p")
        assert "error" not in out
        assert out["scan_centre_panel_px"] == [990.0, 630.0]
        assert "radius_from_centre_px" not in out

    def test_the_smaller_axis_wins(self, stub_sweep, state, tmp_path) -> None:
        """A narrower fall on one axis sets the answer."""
        stub_sweep.edges["x"], stub_sweep.edges["y"] = 1500, 900
        stub_sweep.fall_lo["x"], stub_sweep.fall_lo["y"] = 300, 600
        out = bs.measure_illuminated_radius(1, "daheng", 0, debug_dir=tmp_path / "p")
        assert out["scan_radius_x"] == pytest.approx(600.0)
        assert out["scan_radius_y"] == pytest.approx(150.0)
        assert out["radius_px"] == pytest.approx(150.0)


class TestEvidence:
    def test_profiles_are_kept_for_the_whole_fall(
        self, stub_sweep, state, tmp_path
    ) -> None:
        """The edge rule reads the whole profile, so it must be on disk."""
        state["slm1_bench_pupil_center"] = "960,720"
        target = tmp_path / "p"
        out = bs.measure_illuminated_radius(1, "daheng", 0, debug_dir=target)
        files = list(target.glob("*.json"))
        assert len(files) == 1
        import json

        assert json.loads(files[0].read_text(encoding="utf-8")) == out
        assert out["profile_x"] == [1.0, 0.5]
        assert out["floor_rel_x"] == pytest.approx(0.5)

    def test_default_evidence_dir_is_under_data_debug(self) -> None:
        parts = bs.BEAM_RADIUS_DEBUG_DIR.parts
        assert parts[0] == "data" and parts[1] == "debug"


class TestFailureHandling:
    def test_a_sweep_error_is_reported_not_raised(
        self, monkeypatch, state, tmp_path
    ) -> None:
        """The sidebar must survive a failed probe and say what went wrong."""

        def boom(**kwargs):
            raise RuntimeError("camera stopped responding")

        monkeypatch.setattr(
            "ao_shaping.tools.slm.panel.beam_extent.sweep_axis", boom
        )
        monkeypatch.setattr(
            bs, "probe_session_exposure_ms", lambda *a, **k: (0.12, {})
        )
        state["slm1_bench_pupil_center"] = "960,720"
        out: dict[str, Any] = bs.measure_illuminated_radius(
            1, "daheng", 0, debug_dir=tmp_path / "p"
        )
        assert "camera stopped responding" in out["error"]