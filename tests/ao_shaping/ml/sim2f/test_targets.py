"""Unit tests for the target library: shapes, anti-aliasing, support, determinism.

The support mask is the part that is easy to get subtly wrong, because it feeds
the teacher's encircled-energy term: it must be the *geometric* region the shape
covers, not a threshold on the (anti-aliased) image.  These tests therefore check
that the support is non-empty, that it lies inside the frame, and that every
support pixel really carries energy -- the property a threshold would break for a
dim edge pixel.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.sim2f.config import TARGET_KINDS, Sim2fConfig, SynthConfig
from ml.sim2f.targets import (
    DEFAULT_MIN_FEATURE_PX,
    REFERENCE_CCD_PX,
    TargetSample,
    draw_target,
    thin_feature_px,
)

CCD = 64

#: The measured PSF FWHM of the calibrated twin, camera pixels -- the basis of the
#: feature floor (``report/shape_trials/sim_vs_hw/``).
PSF_FWHM_PX = 16.0


def _ncc(first: np.ndarray, second: np.ndarray) -> float:
    """Normalised cross-correlation of two mean-subtracted images.

    Args:
        first: ``(n, n)`` image.
        second: ``(n, n)`` image.

    Returns:
        Correlation in ``[-1, 1]``.
    """
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


@pytest.mark.parametrize("kind", TARGET_KINDS)
def test_every_kind_is_a_valid_unit_sum_target(kind: str) -> None:
    """Each family is finite, non-negative, unit sum, and inside the frame."""
    sample = draw_target(kind, CCD, np.random.default_rng(4))
    assert isinstance(sample, TargetSample)
    assert sample.kind == kind
    assert sample.intensity.shape == (CCD, CCD)
    assert sample.intensity.dtype == np.float32
    assert sample.support.shape == (CCD, CCD)
    assert sample.support.dtype == np.bool_

    values = sample.intensity
    assert np.isfinite(values).all()
    assert float(values.min()) >= 0.0
    assert float(values.sum()) == pytest.approx(1.0, rel=1e-5)

    # Support is inside the frame and non-empty, and it only covers pixels that
    # actually carry energy (the anti-aliased halo is allowed, so the converse is
    # not required).
    assert int(sample.support.sum()) >= 16
    assert not sample.support[0, :].any()
    assert not sample.support[-1, :].any()
    assert not sample.support[:, 0].any()
    assert not sample.support[:, -1].any()
    assert bool((values[sample.support] > 0.0).all())

    # Geometry is reported for every family, in pixels.
    assert sample.geometry
    assert all(isinstance(value, float) for value in sample.geometry.values())
    assert "size_px" in sample.geometry


@pytest.mark.parametrize("kind", TARGET_KINDS)
def test_kinds_are_deterministic_for_a_fixed_seed(kind: str) -> None:
    """The same seed gives a byte-identical sample; a different seed does not."""
    first = draw_target(kind, CCD, np.random.default_rng(17))
    second = draw_target(kind, CCD, np.random.default_rng(17))
    assert np.array_equal(first.intensity, second.intensity)
    assert np.array_equal(first.support, second.support)
    assert first.geometry == second.geometry

    third = draw_target(kind, CCD, np.random.default_rng(18))
    assert not np.array_equal(first.intensity, third.intensity)


def test_kinds_are_distinguishable() -> None:
    """No two families are the same shape at the same seed (NCC < 0.99)."""
    samples = {kind: draw_target(kind, CCD, np.random.default_rng(5)) for kind in TARGET_KINDS}
    for i, first in enumerate(TARGET_KINDS):
        for second in TARGET_KINDS[i + 1 :]:
            correlation = _ncc(samples[first].intensity, samples[second].intensity)
            assert abs(correlation) < 0.99, f"{first} vs {second} correlate at {correlation:.3f}"


def test_shapes_respect_the_size_range() -> None:
    """``size_frac`` is honoured, and the shape is shrunk to stay inside."""
    rng = np.random.default_rng(3)
    for kind in TARGET_KINDS:
        sample = draw_target(kind, CCD, rng, size_frac=(0.2, 0.3))
        size = sample.geometry["size_px"]
        assert 0.2 * CCD * 0.5 <= size <= 0.3 * CCD * 1.01
        covered = sample.support.sum()
        assert covered <= CCD * CCD * 0.6


def test_target_is_anti_aliased_not_binary() -> None:
    """A rotated/curved shape must have intermediate edge values."""
    for kind in ("rect", "disc", "ring", "line"):
        sample = draw_target(kind, CCD, np.random.default_rng(6))
        values = sample.intensity[sample.intensity > 0.0]
        assert values.size > 0
        # At least one edge pixel is neither empty nor saturated.
        assert float(values.min()) < 0.5 * float(values.max())


def test_center_offset_moves_the_shape() -> None:
    """``center_px`` shifts the shape, and the support follows it."""
    centred = draw_target("disc", CCD, np.random.default_rng(9))
    shifted = draw_target("disc", CCD, np.random.default_rng(9), center_px=(12.0, -8.0))

    def centroid(sample: TargetSample) -> np.ndarray:
        weights = sample.intensity.astype(np.float64)
        rows = np.arange(CCD) - CCD / 2.0
        return np.array(
            [
                float((weights.sum(axis=0) * rows).sum() / weights.sum()),
                float((weights.sum(axis=1) * rows).sum() / weights.sum()),
            ]
        )

    offset = centroid(shifted) - centroid(centred)
    assert offset[0] == pytest.approx(12.0, abs=0.6)
    assert offset[1] == pytest.approx(-8.0, abs=0.6)


def test_invalid_arguments_are_rejected() -> None:
    """Unknown kinds and impossible geometry fail loudly."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        draw_target("hexagon", CCD, rng)
    with pytest.raises(ValueError):
        draw_target("disc", 4, rng)
    with pytest.raises(ValueError):
        draw_target("disc", CCD, rng, size_frac=(0.5, 0.2))
    with pytest.raises(ValueError):
        draw_target("disc", CCD, rng, center_px=(100.0, 0.0))


def test_tiny_frame_still_produces_a_valid_target() -> None:
    """The smallest supported camera grid yields a usable support."""
    sample = draw_target("square", 16, np.random.default_rng(1))
    assert sample.intensity.shape == (16, 16)
    assert int(sample.support.sum()) >= 16
    assert float(sample.intensity.sum()) == pytest.approx(1.0, rel=1e-5)


def test_feature_floor_is_single_sourced_from_the_config() -> None:
    """The floor is the config's number, and it is 1.5 x the measured PSF FWHM."""
    assert DEFAULT_MIN_FEATURE_PX == SynthConfig().min_feature_px
    assert REFERENCE_CCD_PX == Sim2fConfig().ccd_px
    assert DEFAULT_MIN_FEATURE_PX == pytest.approx(1.5 * PSF_FWHM_PX)
    # A floor of 0 (or a negative one) is not a floor: the config rejects it.
    with pytest.raises(ValueError, match="min_feature_px"):
        SynthConfig(min_feature_px=0.0)


@pytest.mark.parametrize("kind", TARGET_KINDS)
def test_every_family_clears_the_feature_floor_on_the_reference_grid(kind: str) -> None:
    """Every family's thinnest feature is >= the floor at the corpus geometry.

    ``ccd_px=256`` with the configured floor is the path the corpus takes, and the
    thinnest feature is *measured back* from the reported geometry -- not compared
    against the draw range -- so a family that forgets to grow a dimension fails
    here.
    """
    floor = SynthConfig().min_feature_px
    for seed in range(8):
        sample = draw_target(
            kind, Sim2fConfig().ccd_px, np.random.default_rng(seed), min_feature_px=floor
        )
        thin = thin_feature_px(kind, sample.geometry)
        assert thin >= floor, f"{kind} seed {seed}: {thin:.2f}px < {floor}px"
        assert thin >= 1.5 * PSF_FWHM_PX
        # The reported number is the checked number, so a cache description can be
        # compared without re-deriving the geometry.
        assert sample.geometry["thin_px"] == pytest.approx(thin)
        assert sample.geometry["thin_px"] >= floor


def test_line_is_a_line_above_the_floor() -> None:
    """``line`` width sits in ``[floor, 1.6 x floor]`` and the bar stays long.

    The old draw range (1.5-3.0 px) was 0.12 x PSF FWHM: a delta function, which
    the 800-iteration teacher measurably cannot shape (L1 1.62 / NCC 0.38).
    """
    floor = SynthConfig().min_feature_px
    for seed in range(24):
        sample = draw_target(
            "line",
            Sim2fConfig().ccd_px,
            np.random.default_rng(seed),
            min_feature_px=floor,
        )
        width = sample.geometry["width_px"]
        length = sample.geometry["length_px"]
        assert floor <= width <= 1.6 * floor + 1e-9
        assert length >= 3.0 * width - 1e-9, (length, width)
        assert sample.geometry["thin_px"] == pytest.approx(width)


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("line", {"width_px": 2.0, "length_px": 100.0}),
        ("square", {"edge_px": 2.0}),
        ("rect", {"width_px": 100.0, "height_px": 2.0}),
        ("disc", {"radius_px": 1.0}),
        ("ring", {"outer_radius_px": 40.0, "inner_radius_px": 38.0}),
        ("cross", {"arm_width_px": 2.0}),
        ("two_spot", {"radius_px": 1.0}),
        ("gaussian", {"sigma_px": 0.5}),
    ],
)
def test_guard_rejects_a_sub_floor_target(kind: str, params: dict[str, float]) -> None:
    """A synthetic sub-floor geometry is refused, with the numbers in the message."""
    floor = SynthConfig().min_feature_px
    measured = thin_feature_px(kind, params)
    assert measured < floor

    with pytest.raises(ValueError) as excinfo:
        thin_feature_px(kind, params, floor_px=floor)
    message = str(excinfo.value)
    assert kind in message
    assert f"{measured:.2f}px" in message
    assert f"{floor:.2f}px" in message
    assert "PSF FWHM" in message


def test_guard_is_silent_when_no_floor_is_given() -> None:
    """``thin_feature_px`` only measures unless a floor is passed."""
    assert thin_feature_px("line", {"width_px": 2.0}) == pytest.approx(2.0)
    assert thin_feature_px("line", {"width_px": 40.0}, floor_px=24.0) == pytest.approx(40.0)


@pytest.mark.parametrize("kind", TARGET_KINDS)
def test_default_floor_is_capped_on_a_toy_grid(kind: str) -> None:
    """On grids below the reference the default floor is the same *fraction*.

    A 24 px feature cannot be a target in a 64 px window -- it would be the frame --
    so the default floor is capped at ``24 / 256`` of the window while an explicit
    floor is honoured verbatim (which is what the corpus passes).
    """
    floor = DEFAULT_MIN_FEATURE_PX / REFERENCE_CCD_PX * CCD
    sample = draw_target(kind, CCD, np.random.default_rng(2))
    assert thin_feature_px(kind, sample.geometry) >= floor
    assert sample.geometry["thin_px"] >= floor


@pytest.mark.parametrize("kind", ["line", "ring"])
def test_explicit_floor_is_not_silently_relaxed(kind: str) -> None:
    """A floor the window cannot hold is an error, not a quietly thinner target.

    ``line`` needs ``3 x floor`` of length and ``ring`` needs ``3 x floor`` of
    diameter (outer radius ``1.5 x floor``), so the reference floor of 24 px does
    not fit a 64 px frame: the draw must stop instead of shrinking the feature.
    """
    with pytest.raises(ValueError) as excinfo:
        draw_target(kind, CCD, np.random.default_rng(2), min_feature_px=24.0)
    message = str(excinfo.value)
    assert kind in message
    assert "24.00px feature floor" in message

