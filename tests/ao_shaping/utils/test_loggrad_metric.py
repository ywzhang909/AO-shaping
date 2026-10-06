"""Regression anchors for ``log_gradient_difference_metric``.

The metric compares two CCD frames through the *gradient of their log
intensity*, which is sensitive to speckle structure that a plain power-ratio
metric (``roi_pib`` / ``rms_pib``) is blind to: a speckled frame and a smooth
frame can enclose the same power yet have very different log gradients.

Every step of the pipeline is pinned here because each one is a design
decision rather than an implementation detail:

1. negative pixels are clipped to zero (read noise makes roughly half of a raw
   CCD frame negative and ``log`` of a negative is undefined);
2. each frame is peak-normalised **independently**, which is what makes the
   metric scale-free (invariant to exposure and laser drift);
3. both frames get the same small separable Gaussian blur so single-pixel
   speckle grains do not dominate the gradient;
4. the logarithm uses an explicit ``LOGGRAD_EPS = 1e-8`` floor - never
   ``np.finfo(np.float64).eps``, which is ~5 orders of magnitude smaller and
   would turn dark pixels into a huge spurious gradient;
5. the score is the mean absolute difference of the gradient magnitudes,
   clipped into ``[0, 1]``.

A frame with no dynamic range (constant, or entirely clipped to zero) has a
zero log gradient, so it scores 0.0 - the same answer as "identical frames".
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.utils.image.target import (
    LOGGRAD_EPS,
    LOGGRAD_SIGMA,
    log_gradient_difference_metric,
)
from ao_shaping.utils.image.target import metrics as metrics_module
from ao_shaping.utils.image.targets import LOGGRAD_EPS as LEGACY_LOGGRAD_EPS
from ao_shaping.utils.image.targets import (
    log_gradient_difference_metric as legacy_log_gradient_difference_metric,
)

FRAME = (21, 21)


def make_blob(
    cx: float = 8.0,
    cy: float = 10.0,
    spread: float = 2.0,
    peak: float = 1000.0,
    shape: tuple[int, int] = FRAME,
) -> np.ndarray:
    """A smooth non-negative Gaussian blob - the analytic "clean" far field."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return peak * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * spread**2)))


def make_speckle(seed: int = 0, shape: tuple[int, int] = FRAME) -> np.ndarray:
    """A high-frequency non-negative frame - the "wrong" far field."""
    rng = np.random.default_rng(seed)
    return rng.random(shape)


class TestExports:
    """The metric is public API: it must be importable from both surfaces."""

    def test_constants_have_the_documented_values(self) -> None:
        assert LOGGRAD_EPS == 1e-8
        assert LOGGRAD_SIGMA == 1.0

    def test_legacy_shim_re_exports_the_same_objects(self) -> None:
        # ``targets.py`` is a wildcard shim driven by the package ``__all__``,
        # so a stale ``__all__`` would silently drop the new names here.
        assert legacy_log_gradient_difference_metric is log_gradient_difference_metric
        assert LEGACY_LOGGRAD_EPS is LOGGRAD_EPS


class TestValidation:
    """Bad input must raise instead of silently returning a meaningless score."""

    def test_rejects_non_square_matching_pair(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            log_gradient_difference_metric(make_blob(), make_blob(shape=(21, 20)))

    @pytest.mark.parametrize("shape", [(21,), (4, 4, 4)])
    def test_rejects_non_2d_frames(self, shape: tuple[int, ...]) -> None:
        frame = np.ones(shape)
        with pytest.raises(ValueError, match="2D"):
            log_gradient_difference_metric(frame, frame)

    @pytest.mark.parametrize("bad", [0.0, -1e-8])
    def test_rejects_non_positive_eps(self, bad: float) -> None:
        frame = make_blob()
        with pytest.raises(ValueError, match="eps"):
            log_gradient_difference_metric(frame, frame, eps=bad)

    def test_rejects_negative_sigma(self) -> None:
        frame = make_blob()
        with pytest.raises(ValueError, match="sigma"):
            log_gradient_difference_metric(frame, frame, sigma=-1.0)

    def test_accepts_zero_sigma_as_no_blur(self) -> None:
        value = log_gradient_difference_metric(make_blob(), make_blob(), sigma=0.0)
        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_sanitises_non_finite_measurement(self, bad: float) -> None:
        """A stray bad pixel must NOT abort a closed-loop run.

        This term runs once per epoch against live camera frames, so raising
        would kill a multi-hour optimisation and leave the SLM holding an
        arbitrary phase. The repo standardises on
        ``np.where(isfinite(frame), frame, 0.0)`` in ``_prepare_frame`` and both
        ``bench_kernels`` prep kernels; the score must stay finite instead.
        """
        frame = make_blob()
        frame[3, 4] = bad
        value = log_gradient_difference_metric(frame, make_blob())
        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_sanitises_non_finite_reference(self, bad: float) -> None:
        frame = make_blob()
        frame[5, 6] = bad
        value = log_gradient_difference_metric(make_blob(), frame)
        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0

    def test_non_finite_pixel_equals_explicit_precleaning(self) -> None:
        """Sanitising must be exactly equivalent to pre-cleaning with the repo idiom.

        Note the score is *not* necessarily unchanged versus the pristine frame:
        a bad pixel is replaced by ``0.0``, not by its original value. The
        invariant is that callers who clean the frame themselves (as
        ``slm_gs_refine._prepare_frame`` does) observe no behavioural difference.
        """
        frame = make_blob()
        poisoned = frame.copy()
        poisoned[9, 11] = np.nan
        precleaned = np.where(np.isfinite(poisoned), poisoned, 0.0)
        assert log_gradient_difference_metric(poisoned, make_blob()) == pytest.approx(
            log_gradient_difference_metric(precleaned, make_blob())
        )


class TestIdenticalFrames:
    """The metric must be exactly zero when the two frames agree."""

    def test_identical_frames_score_exactly_zero(self) -> None:
        frame = make_blob()
        assert log_gradient_difference_metric(frame, frame) == 0.0

    def test_identical_speckle_scores_exactly_zero(self) -> None:
        frame = make_speckle(seed=3)
        assert log_gradient_difference_metric(frame, frame) == 0.0

    def test_two_different_constants_score_exactly_zero(self) -> None:
        # Both frames are flat, so both log fields are flat and both gradients
        # are identically zero - the difference must be 0 even though the two
        # raw levels (3.0 vs 7.0) differ.
        value = log_gradient_difference_metric(
            np.full((9, 11), 3.0), np.full((9, 11), 7.0)
        )
        assert value == 0.0

    def test_all_zero_frame_scores_exactly_zero(self) -> None:
        value = log_gradient_difference_metric(np.zeros(FRAME), make_blob())
        assert value == 0.0

    def test_fully_negative_frame_scores_exactly_zero(self) -> None:
        # Clipping annihilates the whole frame, so there is no log structure.
        value = log_gradient_difference_metric(np.full(FRAME, -4.0), make_blob())
        assert value == 0.0


class TestDegenerateShapes:
    """Frames with no gradient axis must not produce NaN."""

    @pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1)])
    def test_single_pixel_axis_scores_zero(self, shape: tuple[int, int]) -> None:
        frame = make_blob(shape=shape)
        value = log_gradient_difference_metric(frame, frame)
        assert value == 0.0


class TestNegativePixels:
    """A raw CCD frame is roughly half negative; that must not poison the log."""

    def test_negative_pixels_are_clipped_not_propagated(self) -> None:
        rng = np.random.default_rng(7)
        noisy = make_blob() + rng.normal(0.0, 3.0, FRAME)
        assert (noisy < 0.0).any(), "test fixture must actually contain negatives"

        target = make_blob(cx=13.0)
        from_noisy = log_gradient_difference_metric(noisy, target)
        from_clipped = log_gradient_difference_metric(
            np.clip(noisy, 0.0, None), target
        )
        assert from_noisy == pytest.approx(from_clipped, abs=1e-12)
        assert np.isfinite(from_noisy)

    def test_heavy_noise_stays_bounded(self) -> None:
        rng = np.random.default_rng(11)
        noisy = make_blob() + rng.normal(0.0, 40.0, FRAME)
        value = log_gradient_difference_metric(noisy, make_speckle(seed=1))
        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0


class TestScaleInvariance:
    """Independent peak normalisation is what makes the metric exposure-free."""

    def test_global_gain_on_measurement_is_ignored(self) -> None:
        meas = make_blob(cx=8.0)
        target = make_blob(cx=13.0)
        base = log_gradient_difference_metric(meas, target)
        assert log_gradient_difference_metric(meas * 7.5, target) == pytest.approx(
            base, abs=1e-12
        )

    def test_global_gain_on_reference_is_ignored(self) -> None:
        meas = make_blob(cx=8.0)
        target = make_blob(cx=13.0)
        base = log_gradient_difference_metric(meas, target)
        assert log_gradient_difference_metric(meas, target * 0.25) == pytest.approx(
            base, abs=1e-12
        )

    def test_gain_on_both_frames_is_ignored(self) -> None:
        meas = make_blob(cx=8.0)
        target = make_speckle(seed=5)
        base = log_gradient_difference_metric(meas, target)
        assert log_gradient_difference_metric(
            meas * 12.0, target * 0.5
        ) == pytest.approx(base, abs=1e-12)


class TestDiscrimination:
    """A metric that cannot separate good from bad is useless to the search."""

    def test_identical_structure_scores_lower_than_mismatched(self) -> None:
        same = log_gradient_difference_metric(make_blob(cx=10.0), make_blob(cx=10.0))
        other = log_gradient_difference_metric(
            make_blob(cx=6.0), make_blob(cx=15.0)
        )
        assert same < other

    def test_smooth_beam_scores_lower_than_speckle(self) -> None:
        smooth = make_blob(cx=10.0, spread=3.0)
        speckle = make_speckle(seed=2)
        smooth_value = log_gradient_difference_metric(smooth, smooth)
        speckle_value = log_gradient_difference_metric(smooth, speckle)
        assert smooth_value == 0.0
        assert speckle_value > 0.0

    def test_value_is_always_bounded(self) -> None:
        rng = np.random.default_rng(19)
        for seed in range(5):
            a = rng.random(FRAME) * rng.choice([1.0, 1000.0])
            b = rng.random(FRAME) * rng.choice([1.0, 1000.0])
            value = log_gradient_difference_metric(a, b)
            assert np.isfinite(value)
            assert 0.0 <= value <= 1.0

    def test_result_is_a_plain_python_float(self) -> None:
        value = log_gradient_difference_metric(make_blob(), make_speckle())
        assert isinstance(value, float)
        assert not isinstance(value, np.ndarray)


class TestSigmaIsWired:
    """``sigma`` must actually reach the blur, not be accepted and ignored."""

    def test_sigma_changes_the_score(self) -> None:
        meas = make_speckle(seed=4)
        target = make_blob(cx=10.0)
        sharp = log_gradient_difference_metric(meas, target, sigma=0.0)
        blurred = log_gradient_difference_metric(meas, target, sigma=3.0)
        assert abs(sharp - blurred) > 1e-6

    def test_heavier_blur_shrinks_the_score(self) -> None:
        # Blurring both fields toward a common constant drives their gradient
        # magnitudes together, so the difference must decrease with sigma.
        meas = make_speckle(seed=6)
        target = make_speckle(seed=7)
        sharp = log_gradient_difference_metric(meas, target, sigma=0.0)
        blurred = log_gradient_difference_metric(meas, target, sigma=4.0)
        assert blurred < sharp


class TestLayering:
    """``target`` is a leaf: the metric must not drag a heavy stack in."""

    def test_metric_module_stays_pure_numpy(self) -> None:
        source = Path(metrics_module.__file__).read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        assert not imported & {"scipy", "skimage", "cv2"}

    def test_log_floor_is_the_explicit_constant_not_machine_eps(self) -> None:
        # The spec is emphatic here: ``np.finfo(np.float64).eps`` (~2.2e-16) as a
        # log floor would turn every dark pixel into a ~36-unit gradient jump and
        # make the score a read-noise detector.
        for name in (
            "log_gradient_difference_metric",
            "_loggrad_kernel",
            "_loggrad_blur",
            "_loggrad_prepare",
        ):
            src = inspect.getsource(getattr(metrics_module, name))
            assert "np.finfo(np.float64).eps" not in src, name
        main = inspect.getsource(metrics_module.log_gradient_difference_metric)
        assert "LOGGRAD_EPS" in main or "1e-8" in main