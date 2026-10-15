"""Tests for the optional 0-order overlay (:mod:`zero_order_overlay`).

The single most important invariant is that the overlay is emitted in **raw
unwrapped radians** -- it must accumulate across many grating periods rather than
being confined to ``[0, 2*pi)``.  ``test_phase_is_never_wrapped`` pins that; a
stray ``np.mod(..., 2*pi)`` would collapse ``np.ptp`` to <= 2*pi and fail.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import ao_shaping.utils.wavefront.zero_order_overlay as zoo
from ao_shaping.utils.slm.bench_constants import TILT_SHIFT_SCALE
from ao_shaping.utils.wavefront import matrix_utils
from ao_shaping.utils.wavefront.zero_order_overlay import (
    OverlayParams,
    ZeroOrderOverlay,
    apply_zero_order_overlay,
    camera_pixel_um_from_bench,
    grating_first_order_efficiency,
    predicted_shift_px,
)

PANEL = (1200, 1200)
_PERIOD = 64.0


def _zeros(
    shape: tuple[int, int] = PANEL, dtype: type[np.floating] = np.float64
) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


# ---------------------------------------------------------------------------
# shape / mutation / identity
# ---------------------------------------------------------------------------


def test_none_is_identity() -> None:
    phase = np.arange(PANEL[0] * PANEL[1], dtype=np.float64).reshape(PANEL) * 1e-3
    out = apply_zero_order_overlay(phase, OverlayParams(mode=ZeroOrderOverlay.NONE))
    assert np.array_equal(out, phase)


def test_apply_does_not_mutate_input() -> None:
    phase = np.linspace(0.0, 10.0, PANEL[0] * PANEL[1]).reshape(PANEL)
    before = phase.copy()
    apply_zero_order_overlay(
        phase, OverlayParams(mode=ZeroOrderOverlay.BLAZE, blaze_period_px=_PERIOD)
    )
    assert np.array_equal(phase, before), "apply_zero_order_overlay mutated its input"


def test_output_shape_and_dtype_preserved() -> None:
    for dtype in (np.float64, np.float32):
        phase = _zeros(dtype=dtype)
        out = apply_zero_order_overlay(
            phase,
            OverlayParams(mode=ZeroOrderOverlay.BLAZE, blaze_period_px=_PERIOD),
        )
        assert out.shape == phase.shape
        assert np.issubdtype(out.dtype, np.floating)
        assert out.dtype == dtype


# ---------------------------------------------------------------------------
# the raw-only contract: never wrap, never normalise
# ---------------------------------------------------------------------------


def test_phase_is_never_wrapped() -> None:
    """The overlay must accumulate across periods -- a mod-2pi would collapse it."""
    out = apply_zero_order_overlay(
        _zeros(),
        OverlayParams(
            mode=ZeroOrderOverlay.BLAZE,
            blaze_period_px=_PERIOD,
            blaze_direction="vertical",
        ),
    )
    # Accumulates far beyond a single 2*pi window -> definitely not wrapped.
    assert out.max() > 2.0 * np.pi
    span = np.ptp(out)
    assert span > 2.0 * 2.0 * np.pi
    # Expect roughly (extent/period) * 2*pi with extent ~= 1200 px.
    assert span == pytest.approx((PANEL[0] / _PERIOD) * 2.0 * np.pi, rel=0.01)


def test_blaze_ramp_period() -> None:
    out = apply_zero_order_overlay(
        _zeros(),
        OverlayParams(
            mode=ZeroOrderOverlay.BLAZE,
            blaze_period_px=_PERIOD,
            blaze_direction="vertical",
        ),
    )
    row = out[0]
    diffs = np.diff(row)
    assert diffs[0] == pytest.approx(2.0 * np.pi / _PERIOD)
    # Exactly one full wave (2*pi) across one period of samples.
    assert diffs[: int(_PERIOD)].sum() == pytest.approx(2.0 * np.pi)


def test_blaze_direction_changes_axis() -> None:
    vert = apply_zero_order_overlay(
        _zeros(),
        OverlayParams(
            mode=ZeroOrderOverlay.BLAZE,
            blaze_period_px=_PERIOD,
            blaze_direction="vertical",
        ),
    )
    horiz = apply_zero_order_overlay(
        _zeros(),
        OverlayParams(
            mode=ZeroOrderOverlay.BLAZE,
            blaze_period_px=_PERIOD,
            blaze_direction="horizontal",
        ),
    )
    assert not np.allclose(vert, horiz)
    # vertical ramps along panel x -> constant along a row
    assert np.allclose(vert[0], vert[1])
    # horizontal ramps along panel y -> constant down a column
    assert np.allclose(horiz[:, 0], horiz[:, 1])


# ---------------------------------------------------------------------------
# Fresnel on-axis vs off-axis
# ---------------------------------------------------------------------------


def test_on_axis_fresnel_has_no_lateral_tilt() -> None:
    out = apply_zero_order_overlay(
        _zeros(), OverlayParams(mode=ZeroOrderOverlay.FRESNEL_ON_AXIS)
    )
    gy, gx = np.gradient(out)
    # Radially symmetric about the centre -> the mean gradient (a net linear
    # term) is negligible next to the local gradient magnitude.
    assert abs(gx.mean()) < 0.02 * np.abs(gx).max()
    assert abs(gy.mean()) < 0.02 * np.abs(gy).max()


def test_off_axis_fresnel_has_linear_term() -> None:
    on = apply_zero_order_overlay(
        _zeros(),
        OverlayParams(
            mode=ZeroOrderOverlay.FRESNEL_ON_AXIS, fresnel_focal_mm=500.0
        ),
    )
    off = apply_zero_order_overlay(
        _zeros(),
        OverlayParams(
            mode=ZeroOrderOverlay.FRESNEL_OFF_AXIS,
            fresnel_focal_mm=500.0,
            off_axis_x_um=100.0,
            off_axis_y_um=40.0,
        ),
    )
    diff = off - on
    # Pure linear ramp -> second finite differences vanish along both axes.
    assert np.allclose(np.diff(diff, n=2, axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.diff(diff, n=2, axis=0), 0.0, atol=1e-6)
    # ...and it does have a non-zero gradient (it steers).
    assert np.abs(np.diff(diff, axis=1)).mean() > 0.0


# ---------------------------------------------------------------------------
# predicted shift
# ---------------------------------------------------------------------------


def test_predicted_shift_none_and_on_axis_are_zero() -> None:
    assert predicted_shift_px(OverlayParams(mode=ZeroOrderOverlay.NONE)) == (0.0, 0.0)
    assert predicted_shift_px(
        OverlayParams(mode=ZeroOrderOverlay.FRESNEL_ON_AXIS)
    ) == (0.0, 0.0)


def test_predicted_shift_blaze_matches_tilt_shift_scale() -> None:
    dx, dy = predicted_shift_px(
        OverlayParams(
            mode=ZeroOrderOverlay.BLAZE,
            blaze_period_px=_PERIOD,
            blaze_direction="vertical",
        )
    )
    magnitude = float(np.hypot(dx, dy))
    assert magnitude == pytest.approx(TILT_SHIFT_SCALE / _PERIOD)
    assert magnitude == pytest.approx(7400.0 / _PERIOD)


def test_predicted_shift_off_axis_sign_follows_offset() -> None:
    dx_pos, _ = predicted_shift_px(
        OverlayParams(mode=ZeroOrderOverlay.FRESNEL_OFF_AXIS, off_axis_x_um=50.0)
    )
    dx_neg, _ = predicted_shift_px(
        OverlayParams(mode=ZeroOrderOverlay.FRESNEL_OFF_AXIS, off_axis_x_um=-50.0)
    )
    assert dx_pos > 0.0
    assert dx_neg < 0.0


# ---------------------------------------------------------------------------
# efficiency advisory
# ---------------------------------------------------------------------------


def test_efficiency_matches_known_values() -> None:
    assert grating_first_order_efficiency(2) == pytest.approx(0.405, abs=1e-3)
    assert grating_first_order_efficiency(4) == pytest.approx(0.811, abs=1e-3)
    assert grating_first_order_efficiency(8) == pytest.approx(0.950, abs=1e-3)


# ---------------------------------------------------------------------------
# validation + layering
# ---------------------------------------------------------------------------


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        OverlayParams(blaze_period_px=0)
    with pytest.raises(ValueError):
        OverlayParams(fresnel_focal_mm=0)
    with pytest.raises(ValueError):
        OverlayParams(blaze_direction="diagonal")
    with pytest.raises(ValueError):
        # Deliberately invalid: cast keeps the type checker happy without a
        # `# type: ignore` (which the task forbids).
        OverlayParams(mode=cast(ZeroOrderOverlay, "bogus"))
    with pytest.raises(ValueError):
        OverlayParams(wavelength_nm=0)
    with pytest.raises(ValueError):
        OverlayParams(panel_pixel_um=0)


def test_no_forbidden_imports() -> None:
    """AST guard (mirrors tests/ao_shaping/utils/test_cli_params_leaf.py).

    The module must not reach up into ``drivers`` / ``gui`` / ``algorithm`` --
    a deferred import inside a function body still counts.
    """
    module_path = Path(zoo.__file__).resolve()
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    forbidden = ("ao_shaping.drivers", "ao_shaping.gui", "ao_shaping.algorithm")

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders.extend(a.name for a in node.names if a.name.startswith(forbidden))
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import at line {node.lineno}"
            module = node.module or ""
            if module.startswith(forbidden):
                offenders.append(module)

    assert not offenders, f"forbidden imports in zero_order_overlay: {offenders}"


# ---------------------------------------------------------------------------
# New tests for Defect 1 and Defect 2
# ---------------------------------------------------------------------------


def test_off_axis_predicted_shift_is_in_camera_px() -> None:
    params = OverlayParams(
        mode=ZeroOrderOverlay.FRESNEL_OFF_AXIS,
        off_axis_x_um=1000.0,
        fresnel_focal_mm=500.0,
        lens_focal_mm=125.0,
    )
    dx, dy = predicted_shift_px(params)
    # Compute from formula: tilt = x0/f_slm = 0.002 rad; period_m = lambda/tilt
    tilt = params.off_axis_x_um * 1e-6 / (params.fresnel_focal_mm * 1e-3)
    period_m = params.wavelength_nm * 1e-9 / tilt
    period_px = period_m / (params.panel_pixel_um * 1e-6)
    expected = TILT_SHIFT_SCALE / period_px
    assert dx == pytest.approx(expected, abs=1e-6)
    assert dy == pytest.approx(0.0, abs=1e-6)
    # Must be ~111.28, not 250.0
    assert dx == pytest.approx(7400.0 / 66.5, abs=1e-6)


def test_camera_pixel_derivation_matches_matrix_utils() -> None:
    params = OverlayParams(
        mode=ZeroOrderOverlay.FRESNEL_OFF_AXIS,
        off_axis_x_um=1000.0,
        fresnel_focal_mm=500.0,
        lens_focal_mm=125.0,
    )
    p_cam_um = camera_pixel_um_from_bench(params)
    p_cam_um_ref = matrix_utils.camera_pixel_um_from_focal_scale(
        wavelength_nm=params.wavelength_nm,
        focal_length_m=params.lens_focal_mm * 1e-3,
        slm_pixel_um=params.panel_pixel_um,
        focal_scale_px=TILT_SHIFT_SCALE,
    )
    assert p_cam_um == pytest.approx(p_cam_um_ref, abs=1e-9)


def test_off_axis_matches_blaze_equivalent_period() -> None:
    params = OverlayParams(
        mode=ZeroOrderOverlay.FRESNEL_OFF_AXIS,
        off_axis_x_um=1000.0,
        fresnel_focal_mm=500.0,
        lens_focal_mm=125.0,
    )
    dx, dy = predicted_shift_px(params)
    # Blaze-equivalent: tilt = x0/f_slm = 0.002 rad; period_m = lambda/tilt
    tilt = params.off_axis_x_um * 1e-6 / (params.fresnel_focal_mm * 1e-3)
    period_m = params.wavelength_nm * 1e-9 / tilt if tilt != 0 else float("inf")
    period_px = period_m / (params.panel_pixel_um * 1e-6)
    # Blaze "vertical" ramps along panel x, which maps to camera y (the
    # deliberate 90-degree panel<->camera swap), so the displacement lands in
    # dy_blaze. Compare magnitudes: the off-axis dx must equal the blaze
    # magnitude for the equivalent period.
    dx_blaze, dy_blaze = predicted_shift_px(
        OverlayParams(
            mode=ZeroOrderOverlay.BLAZE,
            blaze_period_px=period_px,
            blaze_direction="vertical",
        )
    )
    blaze_magnitude = float(np.hypot(dx_blaze, dy_blaze))
    assert dx == pytest.approx(blaze_magnitude, abs=1e-6)
    assert dy == pytest.approx(0.0, abs=1e-6)


def test_helper_is_reused_across_calls() -> None:
    # Clear cache to ensure clean test
    zoo._HELPER_CACHE.clear()
    try:
        phase = _zeros()
        out1 = apply_zero_order_overlay(
            phase,
            OverlayParams(mode=ZeroOrderOverlay.BLAZE, blaze_period_px=_PERIOD),
        )
        out2 = apply_zero_order_overlay(
            phase,
            OverlayParams(mode=ZeroOrderOverlay.BLAZE, blaze_period_px=_PERIOD),
        )
        # Same helper instance should be used
        assert len(zoo._HELPER_CACHE) == 1
        key = (PANEL[1], PANEL[0])  # (width, height)
        assert key in zoo._HELPER_CACHE
        # Outputs should be identical
        assert np.allclose(out1, out2)
    finally:
        zoo._HELPER_CACHE.clear()


def test_injected_helper_is_used() -> None:
    zoo._HELPER_CACHE.clear()
    try:
        phase = _zeros()
        helper = zoo.PatternHelper(resolution=(PANEL[1], PANEL[0]))
        out1 = apply_zero_order_overlay(
            phase,
            OverlayParams(mode=ZeroOrderOverlay.BLAZE, blaze_period_px=_PERIOD),
            helper=helper,
        )
        out2 = apply_zero_order_overlay(
            phase,
            OverlayParams(mode=ZeroOrderOverlay.BLAZE, blaze_period_px=_PERIOD),
            helper=helper,
        )
        # Cache should not grow when helper is injected
        assert len(zoo._HELPER_CACHE) == 0
        assert np.allclose(out1, out2)
    finally:
        zoo._HELPER_CACHE.clear()
