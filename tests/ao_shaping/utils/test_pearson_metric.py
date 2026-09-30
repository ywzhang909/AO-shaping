"""Regression anchors for ``pearson_shape_metric``.

Pins the exact current behaviour of the ``1 - Pearson`` shaping loss
(``ao_shaping.utils.image.target.metrics.pearson_shape_metric``, a NumPy port
of ``ml.gsnet.losses.ShapingLosses.shaping_loss``):

* ``loss = 1 - corr`` where both the frame and the unit-sum target are
  mean-centred over the **FULL flattened frame** (never the ROI) and the
  denominator carries a ``+ 1e-12`` epsilon;
* ``energy = sum(I[roi]) / sum(I)`` - the in-ROI intensity fraction;
* a dark / NaN / un-normalisable frame **or an off-frame target** returns the
  strong penalty sentinel ``(1e3, 0.0)``;
* non-2D input raises ``ValueError``.

Every frame below is a handful of pixels wide and hand-computable; the
headline anchor derives its expected ``1/3`` from the arithmetic in the
comment rather than from a re-implementation of the same NumPy expression.
No hardware, no network, no torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.target import pearson_shape_metric, target_shape_roi


# The penalty sentinel shared with ``rmse_shape_metric`` (metrics.py:384).
SENTINEL = (1e3, 0.0)

# 4x4 frame whose mean is exactly 1.0 (so the mean-centring is a plain
# subtraction that a human can redo on paper):
#
#       0.75  0.75  0.75  0.75
#       0.75  2.00  2.00  0.75
#       0.75  1.50  0.50  1.00
#       1.00  0.75  1.00  1.00
ANCHOR_FRAME = np.array(
    [
        [0.75, 0.75, 0.75, 0.75],
        [0.75, 2.00, 2.00, 0.75],
        [0.75, 1.50, 0.50, 1.00],
        [1.00, 0.75, 1.00, 1.00],
    ],
    dtype=np.float64,
)


def _gaussian_frame() -> np.ndarray:
    """A deterministic 20x20 'CCD frame': a Gaussian spot on a pedestal."""
    ys, xs = np.mgrid[0:20, 0:20]
    return 10.0 + 5.0 * np.exp(-(((xs - 10.0) ** 2 + (ys - 10.0) ** 2) / 18.0))


class TestHandComputedAnchor:
    """``pearson_shape_metric`` against an arithmetic worked out on paper."""

    def test_roi_geometry_precondition(self) -> None:
        # The whole anchor hinges on this ROI: a 2px square centred at (2, 2)
        # inside a 4x4 frame -> x_start = y_start = floor(2 - 2/2) = 1, so the
        # ROI is exactly rows 1:3 x cols 1:3, i.e. K = 4 of N = 16 pixels.
        roi = target_shape_roi((4, 4), (2, 2), "square", 2)
        assert roi.dtype == bool
        assert roi.sum() == 4
        assert roi[1:3, 1:3].all()
        assert not roi[0, :].any() and not roi[3, :].any()

    def test_loss_equals_one_third(self) -> None:
        # N = 16, K = 4.  target_n = roi / 4, so mean(target_n) = 4*(1/4)/16
        # = 1/16 and the mean-centred target is  t = +3/16 inside the ROI and
        # -1/16 outside.
        #
        # mean(frame) = 16/16 = 1.0 exactly, hence
        #     p = frame - 1.0 =  [-0.25 -0.25 -0.25 -0.25
        #                           -0.25  1.00  1.00 -0.25
        #                           -0.25  0.50 -0.50  0.00
        #                            0.00 -0.25  0.00  0.00]   (sums to 0)
        #     sum(p**2)   = 0.25 + 2.125 + 0.5625 + 0.0625 = 3.0
        #     P_roi       = 1.0 + 1.0 + 0.5 - 0.5       = 2.0
        #     sum(p*t)    = t_roi*P_roi + t_out*(-P_roi) = (3/16 + 1/16)*2 = 0.5
        #     sum(t**2)   = 4*(3/16)**2 + 12*(1/16)**2 = 3/16
        #     denom       = sqrt(3.0 * 3/16) + 1e-12 = 0.75 + 1e-12
        #     corr        = 0.5 / 0.75 = 2/3   ->   loss = 1 - 2/3 = 1/3
        # The 1e-12 epsilon only shrinks corr by ~8.9e-13, i.e. it *raises* the
        # loss by ~8.9e-13 - four orders of magnitude below the 1e-9 tolerance.
        loss, energy = pearson_shape_metric(ANCHOR_FRAME, (2, 2), "square", 2)
        assert loss == pytest.approx(0.3333333333333333, abs=1e-9)
        # total = 16 (mean 1.0 * 16 px), in-ROI sum = 2 + 2 + 1.5 + 0.5 = 6
        assert energy == pytest.approx(6.0 / 16.0, abs=1e-12)

    def test_anchor_depends_on_the_target_roi(self) -> None:
        """The *same* frame scores differently against a different ROI.

        Guards against the loss being cached / target-independent; the
        mean-centring scope is unchanged here (still full-frame).
        """
        loss_roi4, _ = pearson_shape_metric(ANCHOR_FRAME, (2, 2), "square", 2)
        loss_roi1, _ = pearson_shape_metric(ANCHOR_FRAME, (2, 2), "square", 1)
        # square size 1 -> single centre pixel (1, 1) = 2.0, so the correlation
        # there is different from the 2/3 of the 4-pixel ROI.
        assert loss_roi1 != pytest.approx(loss_roi4, abs=1e-9)


class TestExtremeCorrelations:
    """``loss`` at the two ends of the correlation range."""

    def test_perfect_match_is_zero_loss(self) -> None:
        roi = target_shape_roi((20, 20), (10, 10), "square", 3)
        # frame proportional to the mask -> p is a positive multiple of t
        # (p = 12 t) -> corr = 1, so loss = 1e-12/(12*sum(t**2)) ~ 4.4e-13.
        loss, energy = pearson_shape_metric(
            roi.astype(np.float64) * 3.0, (10, 10), "square", 3
        )
        assert loss == pytest.approx(0.0, abs=1e-9)
        assert energy == pytest.approx(1.0, abs=1e-12)

    def test_inverted_target_is_two_loss(self) -> None:
        roi = target_shape_roi((20, 20), (10, 10), "square", 3)
        # complement of the mask -> p = -12 t -> corr = -1 -> loss = 2
        # (the 1e-12 epsilon makes it 2 - 4.4e-13).
        frame = (1.0 - roi.astype(np.float64)) * 3.0
        loss, energy = pearson_shape_metric(frame, (10, 10), "square", 3)
        assert loss == pytest.approx(2.0, abs=1e-9)
        # every photon is outside the target box - this is the blind spot the
        # docstring warns about (pair with the ROI energy guard).
        assert energy == pytest.approx(0.0, abs=1e-12)


class TestInvariances:
    """The loss is a *correlation*: invariant to scale and to a DC offset."""

    def test_global_scale_invariance(self) -> None:
        frame = _gaussian_frame()
        base_loss, base_energy = pearson_shape_metric(frame, (10, 10), "square", 3)
        for scale in (0.25, 3.0, 137.0):
            loss, energy = pearson_shape_metric(frame * scale, (10, 10), "square", 3)
            assert loss == pytest.approx(base_loss, rel=1e-12, abs=1e-12)
            # energy is a pure ratio, so it is scale invariant too
            assert energy == pytest.approx(base_energy, rel=1e-12, abs=1e-12)

    def test_scale_invariance_holds_for_a_tiny_frame_too(self) -> None:
        # The denominator carries an *additive* 1e-12 epsilon, so it is only
        # negligible while sqrt(sum(p**2) * sum(t**2)) >> 1e-12.  Shrinking the
        # frame by 1e-4 shrinks the denominator by 1e-4 too, which makes the
        # epsilon's relative weight 1e4x larger: corr is then a few parts in
        # 1e9 below 1 instead of a few parts in 1e13.  The loss is still
        # invariant to ~1e-8, which is the true accuracy of the metric.
        frame = _gaussian_frame()
        base_loss, _ = pearson_shape_metric(frame, (10, 10), "square", 3)
        loss, _ = pearson_shape_metric(frame * 1e-4, (10, 10), "square", 3)
        assert loss == pytest.approx(base_loss, rel=1e-7, abs=1e-9)

    def test_additive_dc_offset_invariance(self) -> None:
        frame = _gaussian_frame()
        base_loss, _ = pearson_shape_metric(frame, (10, 10), "square", 3)
        for offset in (1.0, 100.0, -7.5):
            loss, _energy = pearson_shape_metric(frame + offset, (10, 10), "square", 3)
            assert loss == pytest.approx(base_loss, rel=1e-12, abs=1e-12)
        # energy is NOT offset invariant - it is a sum ratio, not a correlation.
        loss_offset, energy_offset = pearson_shape_metric(
            frame + 100.0, (10, 10), "square", 3
        )
        assert loss_offset == pytest.approx(base_loss, rel=1e-12, abs=1e-12)
        assert energy_offset < 1.0

    def test_constant_frame_has_zero_variance(self) -> None:
        # p == 0 everywhere -> sum(p**2) == 0 -> denom == 1e-12 -> corr == 0
        # -> loss == 1.0 *exactly* (metrics.py:392-394, no rounding involved).
        loss, energy = pearson_shape_metric(
            np.full((20, 20), 5.0), (10, 10), "square", 3
        )
        assert loss == 1.0
        # 9 ROI px of 5.0 out of 400 px of 5.0
        assert energy == pytest.approx(9.0 / 400.0, abs=1e-12)


class TestSentinels:
    """Frames / targets the metric refuses to score return ``(1e3, 0.0)``."""

    def test_all_zero_frame(self) -> None:
        # total = 0.0 -> `total <= 0.0` branch (metrics.py:383).
        assert pearson_shape_metric(np.zeros((20, 20)), (10, 10), "square", 3) == (
            SENTINEL
        )

    def test_frame_with_nan(self) -> None:
        # sum() is NaN -> `not np.isfinite(total)` branch (metrics.py:383).
        frame = _gaussian_frame()
        frame[0, 0] = np.nan
        assert pearson_shape_metric(frame, (10, 10), "square", 3) == SENTINEL

    def test_frame_with_inf(self) -> None:
        # sum() is +inf -> not finite -> same sentinel.
        frame = _gaussian_frame()
        frame[0, 0] = np.inf
        assert pearson_shape_metric(frame, (10, 10), "square", 3) == SENTINEL

    def test_off_frame_target(self) -> None:
        # ROI clips to nothing -> `not roi.any()` branch (metrics.py:383).
        frame = _gaussian_frame()
        assert pearson_shape_metric(frame, (1000.0, 1000.0), "square", 3) == SENTINEL
        assert pearson_shape_metric(frame, (-500.0, 8.0), "square", 3) == SENTINEL

    def test_sentinel_is_strictly_worse_than_any_real_loss(self) -> None:
        # A real 1 - corr loss lives in [0, 2]; the sentinel is 1e3, so an
        # optimiser minimising this objective can never adopt a sentinel state.
        loss, _ = pearson_shape_metric(_gaussian_frame(), (10, 10), "square", 3)
        assert 0.0 <= loss <= 2.0
        assert loss < SENTINEL[0]


class TestFullFrameVersusRoi:
    """The regression guard for the deliberate full-frame mean-centring.

    ``pearson_shape_metric`` centres the *whole flattened frame*, not the ROI.
    Two frames that are pixel-identical inside the ROI but differ in a
    halo outside it therefore MUST get different losses.  If somebody later
    "optimises" the metric to ROI-only correlation, this test fails - which is
    the whole point: the halo is real light and the loss must see it.
    """

    @staticmethod
    def _pair() -> tuple[np.ndarray, np.ndarray]:
        in_roi = np.array(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float64
        )
        dark = np.zeros((20, 20), dtype=np.float64)  # total in-ROI sum = 16
        haloed = np.full((20, 20), 0.5, dtype=np.float64)  # + 391*0.5 outside
        dark[8:11, 8:11] = in_roi
        haloed[8:11, 8:11] = in_roi
        return dark, haloed

    def test_identical_roi_content_gives_different_losses(self) -> None:
        dark, haloed = self._pair()
        roi = target_shape_roi((20, 20), (10, 10), "square", 3)
        # precondition: the two frames are *pixel-identical inside the ROI*
        assert np.array_equal(dark[roi], haloed[roi])
        assert not np.array_equal(dark, haloed)

        loss_dark, _ = pearson_shape_metric(dark, (10, 10), "square", 3)
        loss_halo, _ = pearson_shape_metric(haloed, (10, 10), "square", 3)
        assert loss_dark != loss_halo
        assert abs(loss_dark - loss_halo) > 0.01
        # the halo is light that has escaped the box, so the full-frame metric
        # must punish it
        assert loss_halo > loss_dark

    def test_roi_only_correlation_would_be_degenerate(self) -> None:
        """Why the guard is needed: ROI-only correlation is undefined here.

        The target is *uniform inside the ROI*, so restricted to the ROI the
        target vector is a constant 1/9 - zero variance. Any ROI-only Pearson
        therefore divides by a zero denominator and cannot be formed at all.
        That is exactly why the shipped metric mean-centres over the whole
        flattened frame: the ROI-restricted view carries no structure, only
        the full frame does.
        """
        dark, haloed = self._pair()
        roi = target_shape_roi((20, 20), (10, 10), "square", 3)
        target_n = roi.astype(np.float64) / float(roi.sum())
        assert float(target_n[roi].std()) == 0.0  # degenerate target vector
        assert float(target_n[~roi].std()) == 0.0  # ...and so is the complement

        # The shipped, full-frame metric is well defined - and it separates the
        # two frames even though their ROI content is identical.
        loss_dark, _ = pearson_shape_metric(dark, (10, 10), "square", 3)
        loss_halo, _ = pearson_shape_metric(haloed, (10, 10), "square", 3)
        assert np.isfinite(loss_dark) and np.isfinite(loss_halo)
        assert loss_dark != loss_halo


class TestEnergy:
    """``energy`` is the in-ROI intensity fraction of the total frame."""

    def test_known_energy_fraction(self) -> None:
        frame = np.zeros((20, 20), dtype=np.float64)
        frame[8:11, 8:11] = 1.0  # 9 ROI px, sum 9
        frame[0, 0] = 3.0  # 1 stray px outside, sum 3
        # total = 12, in-ROI = 9 -> 9/12 = 0.75
        _loss, energy = pearson_shape_metric(frame, (10, 10), "square", 3)
        assert energy == pytest.approx(0.75, rel=1e-12, abs=1e-12)

    def test_energy_is_one_when_all_light_is_in_roi(self) -> None:
        frame = np.zeros((20, 20), dtype=np.float64)
        frame[8:11, 8:11] = 2.0
        _loss, energy = pearson_shape_metric(frame, (10, 10), "square", 3)
        assert energy == pytest.approx(1.0, rel=1e-12, abs=1e-12)

    def test_energy_is_zero_when_all_light_is_outside(self) -> None:
        frame = np.zeros((20, 20), dtype=np.float64)
        frame[0, 0] = 1.0
        loss, energy = pearson_shape_metric(frame, (10, 10), "square", 3)
        assert energy == pytest.approx(0.0, abs=1e-12)
        # A lone outlier pixel is *not* the complement of the 3x3 box (that
        # would need all 391 outside pixels lit), so the correlation is only
        # weakly negative - loss must stay inside the [0, 2] range a real
        # Pearson loss can occupy.
        assert 0.0 < loss < 2.0

    def test_energy_tracks_roi_membership_not_peak(self) -> None:
        # `energy` is a *membership* ratio, not a peak measure: the same total
        # energy (9) scores 1.0 spread over the 9 ROI pixels and 0.0 when the
        # whole of it sits in a single stray pixel.
        inside = np.zeros((20, 20), dtype=np.float64)
        inside[8:11, 8:11] = 1.0  # 9 px, sum 9
        outside = np.zeros((20, 20), dtype=np.float64)
        outside[0, 0] = 9.0  # 1 px, sum 9
        _l1, e_inside = pearson_shape_metric(inside, (10, 10), "square", 3)
        _l2, e_outside = pearson_shape_metric(outside, (10, 10), "square", 3)
        assert e_inside == pytest.approx(1.0, abs=1e-12)
        assert e_outside == pytest.approx(0.0, abs=1e-12)


class TestInputValidation:
    """Only the dimensionality of ``img`` is validated."""

    def test_three_dimensional_input_raises(self) -> None:
        with pytest.raises(ValueError, match=r"img must be 2D, got 3D"):
            pearson_shape_metric(np.zeros((4, 4, 3)), (2, 2), "square", 2)

    def test_one_dimensional_input_raises(self) -> None:
        with pytest.raises(ValueError, match=r"img must be 2D, got 1D"):
            pearson_shape_metric(np.zeros(8), (2, 2), "square", 2)

    def test_zero_dimensional_input_raises(self) -> None:
        with pytest.raises(ValueError, match=r"img must be 2D, got 0D"):
            pearson_shape_metric(np.array(1.0), (2, 2), "square", 2)

    def test_integer_frames_are_accepted(self) -> None:
        # np.asarray(..., dtype=float64) upcasts uint16 camera frames.
        frame = np.zeros((20, 20), dtype=np.uint16)
        frame[8:11, 8:11] = 1000
        loss, energy = pearson_shape_metric(frame, (10, 10), "square", 3)
        assert loss == pytest.approx(0.0, abs=1e-9)
        assert energy == pytest.approx(1.0, abs=1e-12)
