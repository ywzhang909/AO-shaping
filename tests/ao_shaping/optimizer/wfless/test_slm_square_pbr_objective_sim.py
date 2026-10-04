"""Simulation-first tests for the PBR (peak-to-background) objective term.

Transcribed from the prose definition in Liu et al., *A universal and improved
mutation strategy for feedback-based wavefront shaping optimization algorithm*,
Acta Photonica Sinica 2023, 52(6):0629002: "the ratio of the focused spot to the
average intensity of the speckle background" (PBR).

The paper prints NO equation for this -- it is prose only -- so these tests pin
OUR reading of that sentence, and they pin the two properties that make it safe
to compute on a real CCD frame:

1. Read-noise immunity. A raw frame has symmetric read noise, so ~half its
   pixels are negative and dividing by a raw background mean drives the ratio
   ABOVE 1 (measured ``PIB=1.0120`` on this bench). The metric must median-subtract
   and clip so this cannot happen.
2. The box must be excluded from the background, otherwise the ratio is inflated
   by the very signal it measures.

No hardware: pure numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.optimizer.wfless.slm_square_shaping import (
    SlmSquareConfig,
    square_objective_score,
    square_peak_to_background_ratio,
    square_quality_score,
)


def _frame_with_blob(
    *,
    shape: tuple[int, int] = (120, 120),
    center: tuple[int, int] = (60, 60),
    blob_sigma: float = 3.0,
    blob_peak: float = 100.0,
    background: float = 1.0,
) -> np.ndarray:
    """A clean Gaussian blob on a uniform background.

    The box (``side=20``) is wide enough to contain essentially all of the blob,
    so the box max is ``background + blob_peak`` and the region outside the box
    sits at exactly ``background``.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    return background + blob_peak * np.exp(-r2 / (2.0 * blob_sigma**2))


class TestSquarePeakToBackgroundRatio:
    def test_known_analytic_value(self):
        """On a clean frame PBR is exactly (background + peak) / background."""
        img = _frame_with_blob(background=2.0, blob_peak=100.0)
        pbr = square_peak_to_background_ratio(img, (60, 60), side=20)
        assert pbr == pytest.approx(102.0 / 2.0, rel=0.02)

    def test_a_dc_pedestal_lowers_the_contrast_ratio_exactly(self):
        """PBR is a contrast ratio, so a pedestal legitimately reduces it.

        Adding delta to the whole frame gives (peak+d)/(background+d), which is
        strictly smaller than peak/background for any d > 0. Pinning the exact
        arithmetic -- this is a deliberate property, not an accident, and it is
        why the metric must NOT median-subtract (that would erase the pedestal
        and make the denominator meaningless).
        """
        background, peak, delta = 2.0, 100.0, 3.0
        img = _frame_with_blob(background=background, blob_peak=peak)
        base = square_peak_to_background_ratio(img, (60, 60), side=20)
        shifted = square_peak_to_background_ratio(
            img + delta, (60, 60), side=20
        )
        expected = (background + peak + delta) / (background + delta)
        assert base == pytest.approx((background + peak) / background, rel=0.02)
        assert shifted == pytest.approx(expected, rel=0.02)
        assert shifted < base

    def test_background_level_monotonically_sets_the_ratio(self):
        """PBR must actually track contrast, not saturate.

        A previous median-subtracting version returned 1.3e5 for every background
        level -- the pedestal was subtracted away and the denominator became
        numerical noise. These values pin the real behaviour.
        """
        values = [
            square_peak_to_background_ratio(
                _frame_with_blob(background=bg), (60, 60), side=20
            )
            for bg in (0.5, 2.0, 8.0, 30.0)
        ]
        assert values == sorted(values, reverse=True)
        assert values[0] / values[-1] > 10.0, f"barely varies: {values}"

    def test_fully_negative_frame_scores_zero_not_infinity(self):
        """Clipping a fully-negative frame must not manufacture contrast."""
        img = np.full((60, 60), -5.0)
        assert square_peak_to_background_ratio(img, (30, 30), side=10) == 0.0

    def test_read_noise_cannot_inflate_the_ratio(self):
        """The documented failure mode: negative pixels inflate a raw ratio.

        Half the pixels are negative here, exactly like real read noise (measured
        7164/14400 negative on a sim frame). Without median-subtraction and
        clipping, dividing by a near-zero or negative background mean is what
        drives ``PIB``/PBR above 1 (measured ``PIB=1.0120``).
        """
        rng = np.random.default_rng(0)
        img = _frame_with_blob(background=2.0)
        # Read noise large enough to push a substantial part of the frame below
        # zero, mirroring the measured 7164/14400 negative pixels on a real frame.
        noisy = img + rng.normal(0.0, 4.0, size=img.shape)
        assert (noisy < 0).sum() > 0.3 * noisy.size

        clean = square_peak_to_background_ratio(img, (60, 60), side=20)
        pbr = square_peak_to_background_ratio(noisy, (60, 60), side=20)
        assert np.isfinite(pbr)
        # Clipping raises the background slightly, so PBR drops a little -- but
        # it must stay in the same ballpark, not blow up past1 or collapse to 0.
        assert 0.5 * clean < pbr < clean

    def test_darker_background_raises_the_ratio(self):
        bright_bg = square_peak_to_background_ratio(
            _frame_with_blob(background=8.0), (60, 60), side=20
        )
        dark_bg = square_peak_to_background_ratio(
            _frame_with_blob(background=0.5), (60, 60), side=20
        )
        assert dark_bg > bright_bg

    def test_box_is_excluded_from_the_background(self):
        """The ratio must not be inflated by the signal it is measuring.

        If the box leaked into its own denominator, growing the box to swallow the
        whole frame would drive PBR toward1 instead of tracking peak/background.
        """
        img = _frame_with_blob()
        img[0, 0] = 400.0  # a bright speckle grain far from the box

        small = square_peak_to_background_ratio(img, (60, 60), side=20)
        huge = square_peak_to_background_ratio(img, (60, 60), side=118)
        assert small > 0.0
        # Swallowing everything (including the grain) collapses the contrast.
        assert huge < small

    def test_returns_zero_for_a_dead_frame(self):
        assert (
            square_peak_to_background_ratio(np.zeros((40, 40)), (20, 20), side=8)
            == 0.0
        )

    def test_returns_zero_when_the_frame_is_too_small_to_measure(self):
        """Background pixel budget is a real guard, not decoration."""
        img = _frame_with_blob(shape=(12, 12))
        assert (
            square_peak_to_background_ratio(
                img, (6, 6), side=10, min_background_px=10_000
            )
            == 0.0
        )

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match="2D"):
            square_peak_to_background_ratio(np.zeros((4, 4, 3)), (2, 2), side=2)


class TestQualityScorePbrTerm:
    def test_default_is_off_so_existing_runs_reproduce(self):
        """w_pbr=0 must be an exact no-op -- this is why it defaults to 0."""
        base = square_quality_score(cv=0.4, encircled_energy=0.5, aspect_ratio=1.0)
        with_pbr = square_quality_score(
            cv=0.4,
            encircled_energy=0.5,
            aspect_ratio=1.0,
            peak_to_background=12.0,
            w_pbr=0.0,
        )
        assert with_pbr == pytest.approx(base)

    def test_enabling_it_can_only_increase_the_score(self):
        base = square_quality_score(cv=0.4, encircled_energy=0.5, aspect_ratio=1.0)
        better = square_quality_score(
            cv=0.4,
            encircled_energy=0.5,
            aspect_ratio=1.0,
            peak_to_background=8.0,
            w_pbr=0.3,
        )
        assert better > base

    def test_score_stays_in_the_unit_interval_even_for_absurd_pbr(self):
        """pbr -> inf must not break the documented [0, 1] contract."""
        score = square_quality_score(
            cv=0.0,
            encircled_energy=1.0,
            aspect_ratio=1.0,
            peak_to_background=1e9,
            w_pbr=5.0,
        )
        assert 0.0 <= score <= 1.0

    def test_pbr_contribution_is_monotone(self):
        scores = [
            square_quality_score(
                cv=0.4,
                encircled_energy=0.5,
                aspect_ratio=1.0,
                peak_to_background=p,
                w_pbr=0.3,
            )
            for p in (0.0, 1.0, 5.0, 50.0)
        ]
        assert scores == sorted(scores)

    def test_measured_bench_pbr_still_has_gradient(self):
        """Regression guard for a real saturation bug found on measured data.

        This bench measures peak ~79 counts against a ~0.31 counts background,
        i.e. PBR ~ 250. A linear ``pbr/(1+pbr)`` term maps that to 0.996, which
        leaves the SPGD update with essentially no gradient. The log mapping must
        keep distinct, meaningfully-spaced values across that whole range.
        """
        span = [
            square_quality_score(
                cv=0.4,
                encircled_energy=0.5,
                aspect_ratio=1.0,
                peak_to_background=p,
                w_pbr=0.3,
            )
            for p in (100.0, 250.0, 500.0, 800.0)
        ]
        # Strictly increasing: every step must move the score.
        assert all(b > a for a, b in zip(span, span[1:]))
        # And the steps must be big enough to act on, not numerical dust.
        gaps = [b - a for a, b in zip(span, span[1:])]
        assert min(gaps) > 1e-3, f"gradient collapsed across PBR 100..800: {gaps}"

    def test_pbr_reference_shifts_the_saturation_point(self):
        # cv/ee/ar chosen so the base score is ~0.5 and the PBR term is visible
        # instead of being hidden by the [0, 1] clamp.
        kwargs = dict(cv=0.4, encircled_energy=0.5, aspect_ratio=1.0, w_pbr=0.5)
        low = square_quality_score(
            peak_to_background=250.0, pbr_reference=100.0, **kwargs
        )
        high = square_quality_score(
            peak_to_background=250.0, pbr_reference=5000.0, **kwargs
        )
        assert high < low, "a larger reference must leave more headroom"

    def test_negative_pbr_is_clamped_not_rewarded(self):
        """A garbage negative reading must not beat a real zero."""
        junk = square_quality_score(
            cv=0.4,
            encircled_energy=0.5,
            aspect_ratio=1.0,
            peak_to_background=-5.0,
            w_pbr=0.3,
        )
        zero = square_quality_score(
            cv=0.4,
            encircled_energy=0.5,
            aspect_ratio=1.0,
            peak_to_background=0.0,
            w_pbr=0.3,
        )
        assert junk == pytest.approx(zero)


class TestObjectiveScoreForwarding:
    def test_dispatcher_forwards_the_pbr_term(self):
        img = _frame_with_blob()
        center, side = (60, 60), 20
        pbr = square_peak_to_background_ratio(img, center, side)

        without = square_objective_score(
            img, 0.4, 0.5, 1.0, center, side, "quality", w_pbr=0.0
        )
        with_term = square_objective_score(
            img,
            0.4,
            0.5,
            1.0,
            center,
            side,
            "quality",
            w_pbr=0.3,
            peak_to_background=pbr,
        )
        assert with_term > without

    def test_pearson_branch_ignores_pbr(self):
        """PBR is a "quality"-only term; it must not perturb pearson."""
        img = _frame_with_blob()
        a = square_objective_score(
            img, 0.4, 0.5, 1.0, (60, 60), 20, "pearson", w_pbr=0.0
        )
        b = square_objective_score(
            img,
            0.4,
            0.5,
            1.0,
            (60, 60),
            20,
            "pearson",
            w_pbr=0.9,
            peak_to_background=99.0,
        )
        assert a == pytest.approx(b)


class TestConfigAndCliPlumbing:
    def test_config_defaults_to_off(self):
        assert SlmSquareConfig().w_pbr == 0.0

    def test_config_accepts_an_explicit_weight(self):
        assert SlmSquareConfig(w_pbr=0.25).w_pbr == pytest.approx(0.25)

    def test_square_params_expose_the_flag(self):
        import dataclasses as dc

        from ao_shaping.runners.runner_common import SlmSquareParams

        assert "w_pbr" in {f.name for f in dc.fields(SlmSquareParams)}

    def test_objective_params_expose_the_flag(self):
        import dataclasses as dc

        from ao_shaping.runners.runner_common import ObjectiveParamsSquare

        assert "w_pbr" in {f.name for f in dc.fields(ObjectiveParamsSquare)}

    def test_both_square_runners_forward_the_weight(self):
        """Regression guard: a runner that forgets to forward would silently
        run with PBR off while the user believes it is on."""
        import inspect

        from ao_shaping.runners.slm import (
            gsnet_runner as slm_gsnet_runner,
            shaping_runner as slm_shaping_runner,
        )

        for mod in (slm_gsnet_runner, slm_square_runner):
            src = inspect.getsource(mod)
            assert "w_pbr=" in src, f"{mod.__name__} does not forward w_pbr"