"""Tests for :func:`ml.sim2f.targets.mask_target` -- a caller's bitmap on the camera grid.

``mask_target`` is the entry point the parameterised families do not cover, so it
has to honour the same contract :func:`ml.sim2f.targets.draw_target` does -- unit
sum, anti-aliased intensity, an honest support, the feature floor -- while adding
two guarantees of its own that are the reason the tests here are shaped the way
they are:

* the bitmap's *full extent* maps onto the full frame, so a shape is positioned
  by where it sits in the bitmap and there is no silent shrink (a shape that
  would touch the border is an error, not a clipped one);
* the bitmap may be greyscale, so the intensity must carry the caller's grey
  levels rather than a binarised footprint.

Everything runs on a small grid with numpy only -- no torch, no I/O.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.sim2f.targets import mask_target

#: Source bitmap side. It is resampled onto the camera grid, so its own
#: resolution is independent of ``ccd_px``.
SRC = 128

#: Disc radius in source pixels: comfortably inside the bitmap (so the border
#: check passes) and thick enough on the 64 px grid to clear the default floor.
RADIUS = 40

#: Camera grid the disc tests draw on.
CCD = 64


def _radius(rows: int = SRC, cols: int = SRC) -> np.ndarray:
    """Distance from the bitmap centre, in source pixels.

    Args:
        rows: Bitmap height.
        cols: Bitmap width.

    Returns:
        ``(rows, cols)`` float64 radial coordinate.
    """
    grid_y, grid_x = np.mgrid[0:rows, 0:cols]
    return np.hypot(grid_x - (cols - 1) / 2.0, grid_y - (rows - 1) / 2.0)


def _disc(rows: int = SRC, cols: int = SRC, radius: int = RADIUS) -> np.ndarray:
    """A hard-edged centred disc, 1 inside and 0 outside.

    Args:
        rows: Bitmap height.
        cols: Bitmap width.
        radius: Disc radius in source pixels.

    Returns:
        ``(rows, cols)`` float64 binary bitmap.
    """
    return (_radius(rows, cols) <= radius).astype(np.float64)


def _ramp(rows: int = SRC, cols: int = SRC, radius: int = RADIUS) -> np.ndarray:
    """A greyscale disc: brightest at the centre, smoothly falling to the edge.

    Args:
        rows: Bitmap height.
        cols: Bitmap width.
        radius: Disc radius in source pixels.

    Returns:
        ``(rows, cols)`` float64 greyscale bitmap in ``[0, 1]``.
    """
    return np.clip(1.0 - _radius(rows, cols) / radius, 0.0, 1.0) ** 2


def _support_centre(intensity: np.ndarray, support: np.ndarray) -> tuple[float, float]:
    """Energy-weighted centre of the support, camera pixels.

    Args:
        intensity: ``(n, n)`` target image.
        support: ``(n, n)`` boolean support.

    Returns:
        ``(mean_row, mean_col)``.
    """
    rows, cols = np.nonzero(support)
    weights = intensity[support]
    return (
        float((rows * weights).sum() / weights.sum()),
        float((cols * weights).sum() / weights.sum()),
    )


def test_disc_bitmap_is_a_unit_sum_float32_mask_target() -> None:
    """A centred disc is a unit-sum float32 target with a bool support and kind ``mask``."""
    sample = mask_target(_disc(), CCD)
    assert float(sample.intensity.sum()) == pytest.approx(1.0, abs=1e-6)
    assert sample.intensity.dtype == np.float32
    assert sample.intensity.shape == (CCD, CCD)
    assert sample.support.shape == (CCD, CCD)
    assert sample.support.dtype == np.bool_
    assert sample.kind == "mask"


def test_support_is_exactly_the_pixels_the_image_covers() -> None:
    """Every support pixel carries energy and every pixel outside it carries none."""
    sample = mask_target(_disc(), CCD)
    assert sample.support.sum() > 0
    assert bool((sample.intensity[sample.support] > 0.0).all())
    assert float(sample.intensity[~sample.support].max()) == 0.0


def test_hard_edge_is_anti_aliased_not_nearest_neighbour() -> None:
    """A binary bitmap still gets a soft rim, including pixels above zero but below the plateau."""
    sample = mask_target(_disc(), CCD)
    peak = float(sample.intensity.max())
    rim = (sample.intensity > 0.0) & (sample.intensity < 0.9 * peak)
    assert int(rim.sum()) > 8
    assert bool((sample.intensity[rim] > 0.0).all())


def test_geometry_reports_placement_coverage_and_source_dims() -> None:
    """The geometry dict carries the placement, the coverage and the bitmap's own dims."""
    sample = mask_target(_disc(), CCD)
    for key in (
        "center_x_px",
        "center_y_px",
        "coverage_px",
        "thin_px",
        "mask_rows",
        "mask_cols",
    ):
        assert key in sample.geometry
    assert sample.geometry["coverage_px"] == float(sample.support.sum())
    assert sample.geometry["center_x_px"] == 0.0
    assert sample.geometry["center_y_px"] == 0.0
    assert sample.geometry["mask_rows"] == float(SRC)
    assert sample.geometry["mask_cols"] == float(SRC)


def test_center_px_shifts_the_support_on_the_camera_grid() -> None:
    """``center_px=(x, y)`` moves the shape by ``+x`` columns and ``+y`` rows, in camera px."""
    centred = mask_target(_disc(), CCD, center_px=(0.0, 0.0))
    shifted = mask_target(_disc(), CCD, center_px=(8.0, -6.0))
    row0, col0 = _support_centre(centred.intensity, centred.support)
    row1, col1 = _support_centre(shifted.intensity, shifted.support)
    assert row1 - row0 == pytest.approx(-6.0, abs=1.0)
    assert col1 - col0 == pytest.approx(8.0, abs=1.0)


def test_greyscale_bitmap_keeps_its_levels_and_still_sums_to_one() -> None:
    """A greyscale bitmap stays greyscale -- its bright core beats the support median by >2x."""
    sample = mask_target(_ramp(), CCD)
    on_support = sample.intensity[sample.support]
    assert float(sample.intensity.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(sample.intensity.max()) > 2.0 * float(np.median(on_support))


@pytest.mark.parametrize(
    ("mask", "ccd_px", "kwargs", "message"),
    [
        pytest.param(np.zeros((SRC, SRC)), CCD, {}, "no energy", id="all-zero"),
        pytest.param(np.zeros((8, 8, 8)), CCD, {}, "must be 2-D", id="3-d"),
        pytest.param(-_disc(), CCD, {}, "negative values", id="negative"),
        pytest.param(
            np.where(np.arange(SRC * SRC).reshape(SRC, SRC) == 5, np.nan, _disc()),
            CCD,
            {},
            "non-finite",
            id="nan",
        ),
        pytest.param(
            np.ones((CCD, CCD)), CCD, {}, "touches the frame border", id="border"
        ),
        pytest.param(_disc(), 4, {}, "ccd_px must be >= 8", id="ccd-px-too-small"),
        pytest.param(_disc(), 64.0, {}, "ccd_px must be an int", id="ccd-px-not-int"),
    ],
)
def test_rejects_bitmaps_that_cannot_be_a_target(
    mask: np.ndarray, ccd_px: int, kwargs: dict, message: str
) -> None:
    """Each unusable bitmap is rejected with a ValueError naming the reason."""
    with pytest.raises(ValueError, match=message):
        mask_target(mask, ccd_px, **kwargs)


def test_interior_bar_below_the_floor_is_rejected_then_accepted_on_request() -> None:
    """The feature floor rejects a hairline bar, and ``min_feature_px`` is honoured verbatim."""
    bar = np.zeros((256, 256))
    bar[40:216, 120:121] = 1.0
    with pytest.raises(ValueError, match="thinnest feature"):
        mask_target(bar, 256)
    accepted = mask_target(bar, 256, min_feature_px=1.0)
    assert float(accepted.intensity.sum()) == pytest.approx(1.0, abs=1e-6)
    assert accepted.geometry["mask_rows"] == 256.0
    assert accepted.geometry["mask_cols"] == 256.0


def test_non_square_bitmap_keeps_its_own_aspect() -> None:
    """A non-square bitmap is accepted, and the geometry reports the source dims in order."""
    tall = np.zeros((128, 256))
    tall[24:104, 64:192] = 1.0
    sample = mask_target(tall, CCD)
    assert sample.geometry["mask_rows"] == 128.0
    assert sample.geometry["mask_cols"] == 256.0
    assert float(sample.intensity.sum()) == pytest.approx(1.0, abs=1e-6)


def test_the_same_bitmap_gives_a_bit_identical_target() -> None:
    """Nothing in the path is random: one bitmap always maps onto one target."""
    bitmap = _ramp()
    first = mask_target(bitmap, CCD)
    second = mask_target(bitmap, CCD)
    assert np.array_equal(first.intensity, second.intensity)
    assert np.array_equal(first.support, second.support)
    assert first.geometry == second.geometry
