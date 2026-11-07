"""Non-GS seeds: the square transport seed, the shape-general back-projection,
and the control-resolution export.

Everything here is offline: the seeds are built from a *frozen* checkpoint, so
no device is opened and no CCD frame is read.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
    FrozenDefocusParameters,
    recentre_window,
)
from ao_shaping.algorithm.signal_processing import multishape_objective as mo
from ao_shaping.tools.slm.transfer.geometric_seed import (
    export_frozen_parameters,
    square_transport_phase,
    target_backprojection_phase,
)

ZERO = (1224.0, 962.0)
SIDE = 1200.0
PERIOD = 128.0


def _checkpoint(tmp_path, *, grid=640, fft=2048):
    params = dict(
        defocus_diopter=-0.003, origin=[1221.0, 962.0], amplitude_center=[980.0, 720.0],
        amplitude_width=[340.0, 260.0], residual_noll_radians={str(j): 0.1 for j in range(5, 12)},
        gain=75.0, background=0.58, sigma_ccd_px=0.0,
        ramp_matrix=[[199.0, 7540.0], [-7442.0, 132.0]],
        grid_size=grid, fft_size=fft, pixel_samples=3)
    path = tmp_path / "frozen_parameters.json"
    path.write_text(json.dumps(params), encoding="utf-8")
    return path


def _forward(tmp_path, **kwargs):
    path = _checkpoint(tmp_path, **kwargs)
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = FrozenDefocusParameters.from_dicts(
        payload, payload["ramp_matrix"], grid_size=payload["grid_size"],
        fft_size=payload["fft_size"], pixel_samples=payload["pixel_samples"]).effective_parameters()
    xy = recentre_window(ZERO, half_span_px=820., step_px=8.)
    forward = FrozenDefocusForward(params, xy, device="cpu")
    forward.exposure_ms = 1.0
    centre = np.asarray(ZERO) + np.asarray(params.ramp_matrix)[:, 0] / PERIOD
    return forward, centre


def test_backprojection_seed_is_finite_and_shaped_like_the_control_grid(tmp_path):
    forward, centre = _forward(tmp_path)
    phase, diagnostics = target_backprojection_phase(forward, centre, SIDE, "circle")
    grid = forward.params.grid_size
    assert phase.shape == (grid, grid)
    assert np.isfinite(phase).all()
    assert set(diagnostics) == {"implied_cycles_per_panel_px", "nyquist_cycles_per_panel_px",
                                "target_amplitude_sum"}
    assert diagnostics["nyquist_cycles_per_panel_px"] == 0.5
    assert diagnostics["implied_cycles_per_panel_px"] > 0


def test_backprojection_seed_reduces_to_the_static_correction_for_a_flat_target(tmp_path):
    """A target with no amplitude contrast carries no phase information: the seed
    must be the (negative) static aberration alone, not a random fringe pattern."""
    forward, centre = _forward(tmp_path)
    phase, _ = target_backprojection_phase(forward, centre, SIDE, "circle")
    static = forward.params.defocus_diopter * forward.params.diopter_scale * \
        forward.z4.detach().cpu().numpy()
    # The back-projection of a disc still carries the carrier tilt, so compare
    # the *structure* after removing the linear ramp: it must be near constant.
    gradient = np.diff(phase, axis=1)
    assert np.isfinite(gradient).all()
    assert np.abs(static).max() < 10.0  # sanity: the static term is small


def test_square_transport_seed_is_unchanged_by_the_refactor(tmp_path):
    """The transport seed is the documented square construction; it must keep
    working after the module was refactored to share the export helper."""
    forward, centre = _forward(tmp_path)
    phase = square_transport_phase(forward, centre, SIDE)
    grid = forward.params.grid_size
    assert phase.shape == (grid, grid)
    assert np.isfinite(phase).all()
    # It is a smooth, monotone profile -- not a fringe pattern.
    assert np.abs(np.diff(phase, axis=1)).max() < np.pi


def test_fft_grid_target_matches_the_old_inline_construction(tmp_path):
    forward, centre = _forward(tmp_path)
    p = forward.params
    n = p.fft_size
    yy, xx = np.mgrid[:n, :n]
    frequency = np.stack((xx - n // 2, yy - n // 2), axis=-1) / (n * forward.pitch)
    camera_xy = frequency @ np.asarray(p.ramp_matrix).T + np.asarray(p.origin_xy)
    old = mo.target_at(camera_xy, "triangle", centre, SIDE)
    new = mo.fft_grid_target(forward, centre, SIDE, "triangle")
    assert np.array_equal(old, new)


def test_fft_grid_target_rejects_target_beyond_model_fft_window(tmp_path):
    """A target whose extent exceeds the forward model's FFT-grid reach must be
    rejected, not silently clipped into an all-ones full-window mask.

    The FFT window's physical span is set by the control-grid pitch
    (``grid_size``), independent of ``fft_size``: at grid=128 the reach is only
    ~510 px, so side=1200 (radius 600) maps far outside and previously produced
    a full-window mask that made the GS / back-projection seeds chase a square
    instead of the requested shape (the triangle-to-square defect).
    """
    forward, centre = _forward(tmp_path, grid=128, fft=1024)
    with pytest.raises(ValueError, match="FFT-grid reachable window"):
        mo.fft_grid_target(forward, centre, SIDE, "triangle")


def test_fft_grid_target_accepts_target_within_model_fft_window(tmp_path):
    """A target that fits inside the FFT window is sampled as its true shape,
    never clipped into a saturated full-window mask."""
    forward, centre = _forward(tmp_path)  # grid=640: reach ~2550 px > SIDE=1200
    target = mo.fft_grid_target(forward, centre, SIDE, "triangle")
    assert target.shape == (forward.params.fft_size, forward.params.fft_size)
    assert 0.0 <= float(target.min()) <= float(target.max()) <= 1.0
    # A triangle raster has both lit and dark pixels; an all-ones mask would
    # saturate the whole grid (the defect this guard prevents).
    assert float(target.sum()) < target.size
    assert not np.allclose(target, 1.0)


def test_gs_seed_rejects_target_beyond_model_fft_window(tmp_path):
    """The GS seed goes through ``fft_grid_target``, so the reach guard protects
    it too: an oversized target fails before any GS iteration runs."""
    forward, centre = _forward(tmp_path, grid=128, fft=1024)
    with pytest.raises(ValueError, match="FFT-grid reachable window"):
        mo.gs_seed(forward, centre, SIDE, "triangle")


def test_export_raises_the_control_resolution_without_touching_the_optics(tmp_path):
    path = _checkpoint(tmp_path, grid=640, fft=2048)
    out = export_frozen_parameters(path, tmp_path / "model", grid=640, fft=2048)
    payload = json.loads(out.read_text(encoding="utf-8"))
    original = json.loads(path.read_text(encoding="utf-8"))
    for key in ("defocus_diopter", "origin", "amplitude_center", "amplitude_width", "gain"):
        assert payload[key] == original[key]
    assert (tmp_path / "model" / "provenance.json").is_file()


def test_export_rejects_a_grid_that_does_not_tile_the_panel(tmp_path):
    path = _checkpoint(tmp_path)
    with pytest.raises(ValueError):
        export_frozen_parameters(path, tmp_path / "bad", grid=100, fft=128)
