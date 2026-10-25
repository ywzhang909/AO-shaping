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

from ml.sim2f.config import TARGET_KINDS
from ml.sim2f.targets import TargetSample, draw_target

CCD = 64


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

