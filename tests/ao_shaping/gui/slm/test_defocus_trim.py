"""Offline tests for the defocus overlay — the T4 send-pipeline layer.

The old "defocus trim" (a slider over an accepted GS phase, dispatched from a
fragment in the main area) is gone. Defocus is now a *global, switch-gated*
overlay owned by the driver: the sidebar block builds a panel-scale phase
(Fresnel coarse + Zernike (2,0) fine) and pushes it via ``set_phase_overlay``,
so every send path (optimizer / one-click / CSV / periodic) gets the same layer.

The load-bearing properties — all fail silently when broken:

1. **auto / manual / sentinel** — the focal is the *probed* value by default,
   the operator's manual value overrides it, the ``-9999`` sentinel means "use
   the probe", and a non-finite manual value falls back to the probe (never a
   NaN phase).
2. **pure, non-accumulating build** — ``build_defocus_overlay`` recomputes from
   the session each call; zero is exactly the zero array; the layer is centred
   on the *measured* pupil centre and scaled by the panel span (radius = span/2).
3. **switch-gated push** — ``rebuild_defocus_overlay`` sets the driver overlay to
   ``None`` when off, to the built phase when on, to ``None`` when on but all
   zero, and never touches a disconnected SLM.
4. **widget order is a hard constraint** — in the sidebar block the "use
   recommended value" button must render *before* the focal ``number_input``;
   otherwise clicking it writes a widget key that is already locked and Streamlit
   raises ``StreamlitAPIException``.

No Streamlit server, no hardware: the SLM is a stub, ``fresnel_panel`` /
``zernike_panel`` are the real (pure-numpy) kernels, and ``st`` is patched to a
plain-dict session state plus recording widget stubs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import streamlit as st

from ao_shaping.gui.slm import bench_settings as bs
from ao_shaping.gui.slm import multi_slm_controller as mc
from ao_shaping.tools.kernels.bench_kernels import (
    SLM_PANEL_H,
    SLM_PANEL_W,
    fresnel_panel,
    zernike_panel,
)

PREFIX = "slm1"
SLM_NUM = 1
PANEL = (SLM_PANEL_H, SLM_PANEL_W)


class _StubSLM:
    """Records ``set_phase_overlay`` calls; no device, no driver pipeline."""

    def __init__(self) -> None:
        self.is_open = True
        self.wavelength = None  # → build falls back to 1064 nm
        self._overlay: np.ndarray | None = None
        self._base: np.ndarray | None = None

    def set_phase_overlay(self, phase: np.ndarray | None) -> None:
        self._overlay = None if phase is None else np.asarray(phase, dtype=np.float64)

    def get_phase_overlay(self) -> np.ndarray | None:
        return None if self._overlay is None else self._overlay.copy()

    def set_base_phase(self, phase: np.ndarray | None) -> None:  # controller also pushes base
        self._base = None if phase is None else np.asarray(phase, dtype=np.uint16)

    def get_base_phase(self) -> np.ndarray | None:
        return None if self._base is None else self._base.copy()


def _fresnel(focal: float, centre=(80, 60), radius: int = 20) -> np.ndarray:
    return fresnel_panel(
        float(focal),
        radius=radius,
        pupil_center=centre,
        panel_shape=PANEL,
        wavelength_nm=1064.0,
        pixel_um=8.0,
    )


@pytest.fixture
def bench(monkeypatch) -> dict[str, Any]:
    """Patch Streamlit to a plain-dict session plus recording widget stubs."""
    h: dict[str, Any] = {
        "state": {},
        "slm": _StubSLM(),
        "rendered": [],
        "click": None,
        "successes": [],
    }
    state = h["state"]

    def _ni(label, **k):
        key = k.get("key")
        if key is not None:
            h["rendered"].append(("number_input", key))
            if key in state:
                return state[key]
            return k.get("value")
        return k.get("value")

    monkeypatch.setattr(st, "session_state", state)
    monkeypatch.setattr(
        st,
        "checkbox",
        lambda label, **k: state.get(k.get("key"), k.get("value", False)),
        raising=False,
    )
    monkeypatch.setattr(
        st,
        "button",
        lambda label, **k: (h["rendered"].append(("button", k.get("key"))), k.get("key") == h["click"] and True or False)[1],
        raising=False,
    )
    monkeypatch.setattr(st, "number_input", _ni, raising=False)
    monkeypatch.setattr(st, "caption", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(
        st, "success", lambda *a, **k: h["successes"].append(a[0]), raising=False
    )
    return h


def _render(bench) -> None:
    mc._render_defocus_overlay_block(SLM_NUM)


def _setup(bench, *, span=40.0, centre="80,60") -> None:
    state = bench["state"]
    state[PREFIX] = bench["slm"]
    state[f"{PREFIX}_bench_panel_span_px"] = span
    if centre:
        state[f"{PREFIX}_bench_pupil_center"] = centre


class TestFocalResolution:
    """Auto / manual / sentinel / non-finite → three-state focal selection."""

    def test_no_manual_value_uses_the_probe(self, bench) -> None:
        bench["state"][f"{PREFIX}_bench_focus_probe"] = {"best_focal_mm": -1680.0}
        assert bs.bench_defocus_focal(PREFIX) == -1680.0

    def test_the_auto_sentinel_uses_the_probe(self, bench) -> None:
        bench["state"][f"{PREFIX}_bench_focus_probe"] = {"best_focal_mm": -1680.0}
        bench["state"][f"{PREFIX}_defocus_focal"] = bs.DEFOCUS_FOCAL_AUTO
        assert bs.bench_defocus_focal(PREFIX) == -1680.0

    def test_a_manual_value_overrides_the_probe(self, bench) -> None:
        bench["state"][f"{PREFIX}_bench_focus_probe"] = {"best_focal_mm": -1680.0}
        bench["state"][f"{PREFIX}_defocus_focal"] = 900.0
        assert bs.bench_defocus_focal(PREFIX) == 900.0

    def test_without_a_probe_the_auto_value_is_zero(self, bench) -> None:
        assert bs.bench_defocus_focal(PREFIX) == 0.0

    def test_a_non_finite_manual_value_falls_back_to_the_probe(self, bench) -> None:
        bench["state"][f"{PREFIX}_bench_focus_probe"] = {"best_focal_mm": -1680.0}
        bench["state"][f"{PREFIX}_defocus_focal"] = float("inf")
        assert bs.bench_defocus_focal(PREFIX) == -1680.0


class TestBuildOverlay:
    """``build_defocus_overlay`` is a pure, non-accumulating, panel-scale build."""

    def test_zero_settings_give_exactly_the_zero_array(self, bench) -> None:
        _setup(bench)
        out = bs.build_defocus_overlay(PREFIX)
        assert out.shape == PANEL
        assert out.dtype == np.float64
        assert np.all(out == 0.0)

    def test_focal_produces_the_fresnel_term(self, bench) -> None:
        _setup(bench)
        bench["state"][f"{PREFIX}_defocus_focal"] = -1680.0
        out = bs.build_defocus_overlay(PREFIX)
        np.testing.assert_allclose(out, _fresnel(-1680.0), atol=1e-9)

    def test_centre_follows_the_pupil_centre(self, bench) -> None:
        _setup(bench)
        bench["state"][f"{PREFIX}_defocus_focal"] = -1680.0
        bench["state"][f"{PREFIX}_bench_pupil_center"] = "120,80"
        out = bs.build_defocus_overlay(PREFIX)
        np.testing.assert_allclose(out, _fresnel(-1680.0, centre=(120, 80)), atol=1e-9)

    def test_radius_is_half_the_span(self, bench) -> None:
        _setup(bench, span=80.0)  # radius = 40
        bench["state"][f"{PREFIX}_defocus_focal"] = -1680.0
        out = bs.build_defocus_overlay(PREFIX)
        np.testing.assert_allclose(out, _fresnel(-1680.0, radius=40), atol=1e-9)

    def test_fresnel_and_zernike_compose(self, bench) -> None:
        _setup(bench)
        bench["state"][f"{PREFIX}_defocus_focal"] = -1680.0
        bench["state"][f"{PREFIX}_defocus_amount"] = 4.0
        out = bs.build_defocus_overlay(PREFIX)
        expected = _fresnel(-1680.0) + zernike_panel(
            {(2, 0): 4.0},
            radius=20,
            pupil_center=(80, 60),
            panel_shape=PANEL,
        )
        np.testing.assert_allclose(out, expected, atol=1e-9)


class TestRebuild:
    """``rebuild_defocus_overlay`` is switch-gated and never touches a dead SLM."""

    def test_disabled_pushes_none(self, bench) -> None:
        _setup(bench)
        bench["state"][f"{PREFIX}_defocus_enabled"] = False
        bench["slm"].set_phase_overlay(np.ones(PANEL))
        bs.rebuild_defocus_overlay(PREFIX, bench["slm"])
        assert bench["slm"].get_phase_overlay() is None

    def test_enabled_pushes_the_built_overlay(self, bench) -> None:
        _setup(bench)
        bench["state"][f"{PREFIX}_defocus_enabled"] = True
        bench["state"][f"{PREFIX}_defocus_focal"] = -1680.0
        bs.rebuild_defocus_overlay(PREFIX, bench["slm"])
        np.testing.assert_allclose(
            bench["slm"].get_phase_overlay(), _fresnel(-1680.0), atol=1e-9
        )

    def test_enabled_but_all_zero_clears_the_overlay(self, bench) -> None:
        _setup(bench)
        bench["state"][f"{PREFIX}_defocus_enabled"] = True
        bench["slm"].set_phase_overlay(np.ones(PANEL))
        bs.rebuild_defocus_overlay(PREFIX, bench["slm"])
        assert bench["slm"].get_phase_overlay() is None

    def test_a_disconnected_slm_is_not_touched(self, bench) -> None:
        _setup(bench)
        bench["slm"].is_open = False
        bench["slm"].set_phase_overlay(np.ones(PANEL))
        bs.rebuild_defocus_overlay(PREFIX, bench["slm"])  # must not raise, must not clear
        assert bench["slm"].get_phase_overlay() is not None

    def test_a_centre_outside_the_panel_does_not_crash(self, bench) -> None:
        """``zernike_panel`` empty-slices an out-of-panel centre; it must not raise."""
        _setup(bench)
        bench["state"][f"{PREFIX}_defocus_enabled"] = True
        bench["state"][f"{PREFIX}_defocus_focal"] = 0.0
        bench["state"][f"{PREFIX}_defocus_amount"] = 10.0
        bench["state"][f"{PREFIX}_bench_pupil_center"] = "960,690"
        bs.rebuild_defocus_overlay(PREFIX, bench["slm"])  # must not raise


class TestSidebarBlock:
    """Widget order, the recommended-value button, and the idempotent push."""

    def test_the_recommended_value_button_renders_before_the_focal_input(self, bench) -> None:
        """The order that keeps a click from writing an already-locked key."""
        _setup(bench)
        bench["state"][f"{PREFIX}_defocus_enabled"] = True
        bench["state"][f"{PREFIX}_bench_focus_probe"] = {"best_focal_mm": -1680.0}
        _render(bench)
        idx_button = [
            i
            for i, (t, k) in enumerate(bench["rendered"])
            if t == "button" and k == f"{PREFIX}_defocus_use_reco"
        ]
        idx_focal = [i for i, (t, k) in enumerate(bench["rendered"]) if t == "number_input" and k == f"{PREFIX}_defocus_focal"]
        assert idx_button and idx_focal
        assert idx_button[0] < idx_focal[0]

    def test_clicking_recommended_writes_the_key_and_pushes(self, bench) -> None:
        _setup(bench)
        bench["state"][f"{PREFIX}_defocus_enabled"] = True
        bench["state"][f"{PREFIX}_bench_focus_probe"] = {"best_focal_mm": -1680.0}
        bench["click"] = f"{PREFIX}_defocus_use_reco"
        _render(bench)
        assert bench["state"][f"{PREFIX}_defocus_focal"] == -1680.0
        np.testing.assert_allclose(
            bench["slm"].get_phase_overlay(), _fresnel(-1680.0), atol=1e-9
        )

    def test_the_status_line_renders_regardless_of_the_switch(self, bench) -> None:
        _setup(bench)
        bench["state"][f"{PREFIX}_defocus_enabled"] = False
        _render(bench)  # must not raise; block renders a status line
        _setup(bench)
        bench["state"][f"{PREFIX}_defocus_enabled"] = True
        _render(bench)

    def test_connected_and_enabled_pushes_the_overlay_each_pass(self, bench) -> None:
        _setup(bench)
        bench["state"][f"{PREFIX}_defocus_enabled"] = True
        bench["state"][f"{PREFIX}_defocus_focal"] = -1680.0
        _render(bench)
        np.testing.assert_allclose(
            bench["slm"].get_phase_overlay(), _fresnel(-1680.0), atol=1e-9
        )

    def test_disabled_pushes_none_to_the_driver(self, bench) -> None:
        _setup(bench)
        bench["slm"].set_phase_overlay(np.ones(PANEL))
        bench["state"][f"{PREFIX}_defocus_enabled"] = False
        _render(bench)
        assert bench["slm"].get_phase_overlay() is None
