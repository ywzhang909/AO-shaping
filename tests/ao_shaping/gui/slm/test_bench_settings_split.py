"""Regression for issue #168: bench_settings render/probe functions were split.

The split itself is not the contract -- the contract is what the split must not
change:

* every ``st.<widget>`` call still carries the **same widget key**, because the
  key is the identity the sidebar and ``docs/GUI/slm/slm_gui_manual.md`` §4
  both rely on. A key that silently changes looks fine in the UI and breaks
  every saved session state and every golden.
* the extracted pure helpers still compute the same numbers, because
  ``_confirm_working_exposure`` decides whether an exposure is usable and
  ``_radius_from_sweep`` decides the aperture that feeds the forward model's
  ``aperture_m`` cross-check.

Both are checked at the source level rather than by rendering. Rendering a
Streamlit widget in bare mode is vacuous: ``st.session_state`` is not a real
session, ``st.rerun`` raises, and a widget created without a ScriptRunContext
never reaches the code path that would expose a wrong key. So the key contract
is an AST scan of the module, and the numeric contract drives the pure helpers
with synthetic frames.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ao_shaping.gui.slm import bench_settings as bs

MODULE = Path(bs.__file__)

#: Every widget key the sidebar creates, with the scope it lives in. Keys are
#: per-SLM (prefixed) except the experiment-level ones, which are global.
#: The list is the contract: adding a widget means adding its key here.
EXPERIMENT_KEYS = frozenset(
    {
        "bench_cam_type",
        "bench_cam_id",
        "bench_exposure_ms",
        "bench_cam_crop",
        "bench_cam_size",
        "bench_blaze_period_px",
        "bench_blaze_direction",
    }
)

SLM_KEYS = frozenset(
    {
        "bench_pupil_center",
        "bench_panel_span_px",
        "bench_beam_radius_px",
        "bench_geometry_source",
        "bench_geometry_file",
    }
)

#: Button keys, which are not written to session state but are still identity.
EXPERIMENT_BUTTON_KEYS = frozenset(
    {
        "detect_camera_pixel_btn",
        "detect_exposure_btn",
        "bench_probe_window_btn",
        "adopt_camera_pixel_btn",
    }
)

SLM_BUTTON_KEYS = frozenset(
    {
        "detect_beam_radius_btn",
        "detect_focus_btn",
        "bench_calibrate_btn",
        "bench_restore_btn",
    }
)


#: Streamlit widget constructors that take a ``key=``.
_WIDGET_CALLS = frozenset(
    {
        "button", "checkbox", "color_picker", "date_input", "file_uploader",
        "multiselect", "number_input", "radio", "select_slider", "selectbox",
        "slider", "text_area", "text_input", "time_input", "toggle",
    }
)


def _widget_keys() -> set[str]:
    """Every literal widget key passed to a widget constructor in this module.

    Only widget *creations* count. A ``st.session_state.get("bench_manual")``
    is a read of a key the module owns, not a widget the user interacts with,
    and mixing the two would make the contract depend on internal plumbing.
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8-sig"))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in _WIDGET_CALLS):
            continue
        for kw in node.keywords:
            if kw.arg != "key":
                continue
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                keys.add(kw.value.value)
            elif isinstance(kw.value, ast.Call):
                # ``_scoped(prefix, "bench_...")`` -- the inner literal is the
                # per-SLM key; the prefix is applied at render time.
                for arg in kw.value.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        keys.add(arg.value)
    return keys


def _scoped_key_names() -> set[str]:
    """The bare key literals passed to ``_scoped(prefix, "...")`` by a widget.

    ``_scoped`` is also used to *read* session state (e.g.
    ``_scoped(prefix, "bench_manual")``), and those reads are internal
    plumbing rather than a user-facing widget identity, so only the widget
    call sites are collected.
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8-sig"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in _WIDGET_CALLS):
            continue
        for kw in node.keywords:
            if kw.arg != "key" or not isinstance(kw.value, ast.Call):
                continue
            call = kw.value
            if not (isinstance(call.func, ast.Name) and call.func.id == "_scoped"):
                continue
            if len(call.args) == 2 and isinstance(call.args[1], ast.Constant):
                out.add(str(call.args[1].value))
    return out


class TestWidgetKeyContract:
    """The key set is the sidebar's public surface; it must not drift."""

    def test_experiment_level_keys_present(self) -> None:
        keys = _widget_keys()
        missing = EXPERIMENT_KEYS - keys
        assert not missing, f"experiment-level widget keys vanished: {sorted(missing)}"

    def test_per_slm_keys_present(self) -> None:
        scoped = _scoped_key_names()
        missing = SLM_KEYS - scoped
        assert not missing, f"per-SLM widget keys vanished: {sorted(missing)}"

    def test_button_keys_present(self) -> None:
        keys = _widget_keys()
        missing = (EXPERIMENT_BUTTON_KEYS | SLM_BUTTON_KEYS) - keys
        assert not missing, f"button keys vanished: {sorted(missing)}"

    def test_no_unexpected_keys_appeared(self) -> None:
        """A new key needs a contract entry, not just a new widget.

        Without this the "keys must not change" rule would only guard deletions;
        a typo in a key would pass every other test here.
        """
        known = (
            EXPERIMENT_KEYS
            | SLM_KEYS
            | EXPERIMENT_BUTTON_KEYS
            | SLM_BUTTON_KEYS
        )
        # Keys created by helpers outside this module's widget surface.
        allowed_extras = {
            "bench_beam_radius_probe",
            "bench_focus_probe",
            "bench_calibrated_at",
            "bench_cam_pixel_checked",
            "_cam_pixel_checked",
            "_cam_pixel_mismatch",
        }
        unknown = (_widget_keys() | _scoped_key_names()) - known - allowed_extras
        assert not unknown, (
            f"new widget keys appeared without a contract entry: {sorted(unknown)}"
        )

    def test_per_slm_keys_are_scoped(self) -> None:
        """Per-SLM keys must go through ``_scoped`` so two SLMs do not collide."""
        tree = ast.parse(MODULE.read_text(encoding="utf-8-sig"))
        bare: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr in {
                "button", "text_input", "number_input", "selectbox", "checkbox",
            }):
                continue
            for kw in node.keywords:
                if kw.arg != "key":
                    continue
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    if kw.value.value in SLM_KEYS:
                        bare.append(kw.value.value)
        assert not bare, (
            f"per-SLM keys used without _scoped(): {sorted(set(bare))}"
        )


class TestConfirmWorkingExposure:
    """``_confirm_working_exposure`` decides whether an exposure is usable."""

    def test_accepts_a_frame_below_saturation(self) -> None:
        frame = np.full((4, 4), 100, dtype=np.uint8)

        working, readback, peak, confirmations, dtype = bs._confirm_working_exposure(
            lambda: frame,
            lambda ms: ms,
            ceiling_ms=0.4,
            work_fraction=0.5,
            floor_ms=0.02,
        )
        assert working == pytest.approx(0.2)
        assert readback == pytest.approx(0.2)
        assert peak == 100.0
        assert dtype == np.dtype(np.uint8)
        assert len(confirmations) == 1
        assert confirmations[0]["peak_fraction"] == pytest.approx(100 / 255)

    def test_backs_off_proportionally_when_saturated(self) -> None:
        """A clipped frame must lower the exposure, not be returned as usable.

        The stub returns the saturated frame first and a dim one second, so
        the helper has a reason to back off.
        """
        frames = [np.full((4, 4), 255, dtype=np.uint8), np.full((4, 4), 10, dtype=np.uint8)]
        reads: list[float] = []

        def read_frame() -> np.ndarray:
            return frames[0] if len(reads) <= 1 else frames[1]

        def reset(ms: float) -> float:
            reads.append(ms)
            return ms

        working, _readback, peak, confirmations, _dtype = bs._confirm_working_exposure(
            read_frame,
            reset,
            ceiling_ms=0.4,
            work_fraction=0.5,
            floor_ms=0.02,
        )
        assert len(confirmations) == 2
        assert working < 0.2, "saturated frame must lower the working exposure"
        assert peak == 10.0

    def test_raises_when_every_attempt_is_saturated(self) -> None:
        frame = np.full((4, 4), 255, dtype=np.uint8)
        with pytest.raises(RuntimeError, match="仍然饱和"):
            bs._confirm_working_exposure(
                lambda: frame,
                lambda ms: ms,
                ceiling_ms=0.4,
                work_fraction=0.5,
                floor_ms=0.02,
            )

    def test_floor_is_respected_when_backing_off(self) -> None:
        """Backing off must not go below the floor."""
        frame = np.full((4, 4), 255, dtype=np.uint8)
        with pytest.raises(RuntimeError):
            bs._confirm_working_exposure(
                lambda: frame,
                lambda ms: ms,
                ceiling_ms=0.4,
                work_fraction=0.5,
                floor_ms=0.02,
            )


class _FakeSweepResult:
    """Stand-in for ``sweep_axis``'s result object."""

    def __init__(self, edge: float, centre: float, radius: float, step: float) -> None:
        self.edge_px = edge
        self.centre_px = centre
        self.radius_px = radius
        self.step_px = step


class TestRadiusFromSweep:
    """``_radius_from_sweep`` decides the aperture fed to the forward model."""

    def _results(self) -> dict[str, Any]:
        return {
            "x": _FakeSweepResult(edge=300.0, centre=100.0, radius=150.0, step=10.0),
            "y": _FakeSweepResult(edge=260.0, centre=120.0, radius=130.0, step=12.0),
        }

    def test_takes_the_smaller_axis_and_its_centre(self) -> None:
        out = bs._radius_from_sweep(self._results(), None)
        assert out["radius_px"] == pytest.approx(130.0)
        assert out["radius_source"] == "scan_fall_width"
        assert out["scan_centre_panel_px"] == [100.0, 120.0]

    def test_uncertainty_is_one_step_of_the_coarser_axis(self) -> None:
        out = bs._radius_from_sweep(self._results(), None)
        assert out["step_px"] == pytest.approx(12.0)
        assert out["radius_uncertainty_px"] == pytest.approx(12.0)

    def test_centre_subtracted_estimate_is_reported_alongside(self) -> None:
        """Both estimates must be present so they can be compared, not merged."""
        out = bs._radius_from_sweep(self._results(), (100.0, 120.0))
        assert out["centre_panel_px"] == [100.0, 120.0]
        assert out["radius_x_px"] == pytest.approx(200.0)
        assert out["radius_y_px"] == pytest.approx(140.0)
        assert out["radius_from_centre_px"] == pytest.approx(140.0)
        assert out["radius_px"] == pytest.approx(130.0)

    def test_no_centre_means_no_centre_subtracted_keys(self) -> None:
        out = bs._radius_from_sweep(self._results(), None)
        assert "centre_panel_px" not in out
        assert "radius_from_centre_px" not in out


class TestSplitKeptFunctionInventory:
    """The helpers the split introduced must exist and stay importable."""

    @pytest.mark.parametrize(
        "name",
        [
            "_render_camera_block",
            "_render_exposure_block",
            "_render_window_block",
            "_render_blaze_block",
            "_render_summary_block",
            "_render_slm_geometry_block",
            "_render_slm_focus_block",
            "_render_slm_calibration_block",
            "_confirm_working_exposure",
            "_radius_from_sweep",
        ],
    )
    def test_helper_exists(self, name: str) -> None:
        assert callable(getattr(bs, name))
