"""Tests for ``far_field_size`` zero-padding in ``ZernikeCoefficientOptimizer``.

``far_field_size`` zero-pads the *pupil* before the FFT, so ``forward_intensity``
returns ``far_field_size``-square intensity while the SLM phase stays
``region``-square. The padding is not a cosmetic option: on a coarse grid the
bench's spot collapses to a sub-pixel delta and the individual Zernike
coefficients stop being *identifiable*.

Measured here on ``region=64`` with a two-mode truth (Noll 4 defocus ``+0.8``,
Noll 10 ``-0.4``), ``n_orders=4`` (15 coefficients), ``lr=0.1`` and
``frozen_modes=(1, 2, 3)``:

======================  ==========  ==================
``far_field_size``      final loss  ``|dc - dc_true|``
======================  ==========  ==================
``None`` (64)            4.84e-07    **1.31**  (wrong)
``128``                  8.51e-13    **0.0002**
``256``                  8.51e-13    **0.0002**
======================  ==========  ==================

The unpadded grid reaches a small loss while the coefficients are wrong: many
vectors produce nearly the same coarse spot, so the residual cannot separate
them. This is the concrete form of the class docstring's warning that *a low
loss alone does not prove the coefficients were found* -- a judgement made on
loss alone would prefer the unpadded run by four orders of magnitude.

The invariants locked here are the ones whose violation produced *plausible but
wrong* behaviour:

* ``_target_grid()`` must be the **far-field** grid, not the pupil grid. It used
  to return ``region``, so every ``update()`` with ``far_field_size > region``
  raised a shape mismatch -- exactly the configuration the ``far_field_size``
  docstring calls mandatory.
* Padding must not change the physics: the total intensity is conserved
  (Parseval), the spot stays centred, and the field of view is unchanged --
  only the sampling density grows with ``far_field_size / region``.
* A measured frame is peak-normalised, so the per-pixel rescaling that padding
  introduces cancels and the loss stays comparable across paddings.
* Real CCD frames are 250x248 while the model grid is square, so irregular
  measurements must be accepted and resampled.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)

REGION = 64
N_ORDERS = 4
N_COEFFICIENTS = (N_ORDERS + 1) * (N_ORDERS + 2) // 2  # 15
# Piston/tilt are frozen: the SH-style null space and the two dominant
# alignment modes would otherwise absorb the fit.
FROZEN = (1, 2, 3)
# Noll 4 = defocus (2,0), Noll 10 = (3,1). Both are well inside n_orders=4.
TRUTH = {3: 0.8, 9: -0.4}
# Matches the class default, so tests that do not fit stay on the default path.
DEFAULT_LR = 0.1
# region=64 with far_field_size in {None, 64, 128, 256}: unpadded, identical,
# and the two paddings the bench calibration actually uses.
PADDINGS = (None, 64, 128, 256)
# The real camera window on this bench (see _prepare_measurement's docstring).
IRREGULAR = (248, 250)


def _optimizer(
    *,
    far_field_size: int | None = None,
    frozen_modes: tuple[int, ...] = FROZEN,
    lr: float = DEFAULT_LR,
    n_orders: int = N_ORDERS,
    region: int = REGION,
) -> ZernikeCoefficientOptimizer:
    """A CPU-only optimizer; every test opts into what it actually varies.

    The defaults deliberately mirror the class defaults (``far_field_size=None``,
    ``lr=0.1``) so a test that does not care about padding is exercising the same
    configuration a caller gets for free.
    """
    return ZernikeCoefficientOptimizer(
        n_orders=n_orders,
        region=region,
        lr=lr,
        device="cpu",
        dtype="float64",
        far_field_size=far_field_size,
        frozen_modes=frozen_modes,
    )


def _truth() -> np.ndarray:
    """The coefficient vector the tests try to recover."""
    coefficients = np.zeros(N_COEFFICIENTS, dtype=np.float64)
    for index, value in TRUTH.items():
        coefficients[index] = value
    return coefficients


def _flat_phase() -> np.ndarray:
    """A flat SLM phase, so the pupil amplitude is the only structure."""
    return np.zeros((REGION, REGION), dtype=np.float64)


def _fit_error(coefficients: np.ndarray) -> float:
    """L2 error over the *fitted* modes (frozen modes carry no truth)."""
    return float(np.linalg.norm(coefficients[3:] - _truth()[3:]))


def _centroid_sigma_px(field: np.ndarray) -> float:
    """Intensity-weighted RMS radius in pixels -- the sampling-density probe."""
    total = field.sum()
    weights = field / total
    size = field.shape[0]
    rows, cols = np.mgrid[0:size, 0:size]
    mean_col = float((weights * cols).sum())
    mean_row = float((weights * rows).sum())
    variance = float((weights * (cols - mean_col) ** 2).sum())
    return float(np.sqrt(variance))


class TestTargetGridTracksFarField:
    """``_target_grid()`` is the grid the *prediction* lives on, not the pupil.

    This is the regression that made ``far_field_size`` unusable: the residual
    in ``_anchored_intensity_loss`` is only meaningful when the measurement has
    been resampled onto the same grid the FFT produced.
    """

    @pytest.mark.parametrize("far_field_size", PADDINGS)
    def test_target_grid_is_square_and_matches_padding(
        self, far_field_size: int | None
    ) -> None:
        """The grid is ``far_field_size``-square, falling back to ``region``."""
        optimizer = _optimizer(far_field_size=far_field_size)
        expected = REGION if far_field_size is None else far_field_size

        grid = optimizer._target_grid()

        assert grid == (expected, expected)
        assert optimizer.far_field_size == expected

    @pytest.mark.parametrize("far_field_size", PADDINGS)
    def test_forward_intensity_shape_equals_target_grid(
        self, far_field_size: int | None
    ) -> None:
        """A prediction and its loss target are always on the same grid.

        Before the fix these two disagreed for every ``far_field_size > region``
        and ``update()`` raised on the shape mismatch.
        """
        optimizer = _optimizer(far_field_size=far_field_size)

        intensity = optimizer.forward_intensity(_truth(), _flat_phase())

        assert intensity.shape == optimizer._target_grid()


class TestPaddingPreservesPhysics:
    """Padding changes the sampling density, never the underlying optics."""

    @pytest.mark.parametrize("far_field_size", PADDINGS)
    def test_total_intensity_is_conserved(self, far_field_size: int | None) -> None:
        """Zero-padding the pupil redistributes light but creates none.

        ``norm="ortho"`` plus Parseval makes ``sum(|F|^2)`` equal to
        ``sum(|u|^2)``, and padding with zeros leaves the pupil energy alone.
        The loss is a plain mean of squares, so a padding-dependent total would
        silently rescale every comparison.
        """
        reference = _optimizer().forward_intensity(_truth(), _flat_phase())
        padded = _optimizer(far_field_size=far_field_size).forward_intensity(
            _truth(), _flat_phase()
        )

        assert padded.sum() == pytest.approx(reference.sum(), rel=1e-12)
        assert padded.sum() > 0.0

    @pytest.mark.parametrize("far_field_size", PADDINGS)
    def test_peak_stays_at_the_exact_grid_centre(
        self, far_field_size: int | None
    ) -> None:
        """An off-centre peak would move the ROI the runner freezes on hardware.

        The pupil is centred inside the padded array, so the zero order must
        land on the exact middle pixel of *every* grid size -- not merely near
        the middle.
        """
        intensity = _optimizer(far_field_size=far_field_size).forward_intensity(
            _truth(), _flat_phase()
        )
        side = intensity.shape[0]

        row, col = np.unravel_index(int(np.argmax(intensity)), intensity.shape)

        assert (int(row), int(col)) == (side // 2, side // 2)

    def test_padding_refines_sampling_without_changing_field_of_view(self) -> None:
        """A 4x larger grid samples the same angular range 4x more finely.

        The spot therefore grows from ~2.3 px to ~10.5 px in *pixel* units
        (measured 4.52x for a 4x grid) while covering the same fraction of the
        array. This is what makes padding resolve the coefficients at all: on
        the unpadded grid the spot is barely wider than a pixel.
        """
        unpadded = _optimizer().forward_intensity(_truth(), _flat_phase())
        padded = _optimizer(far_field_size=256).forward_intensity(
            _truth(), _flat_phase()
        )

        ratio = _centroid_sigma_px(padded) / _centroid_sigma_px(unpadded)

        assert 3.0 < ratio < 5.0

    def test_decimated_padding_reproduces_the_unpadded_profile(self) -> None:
        """Padding only rescales: the normalised profile is bit-for-bit stable.

        Every 4th sample of the 256-grid field equals the 64-grid field up to a
        constant factor, and the peak-normalised profiles are exactly equal.
        The measurement is peak-normalised before the loss, so this factor
        cancels and paddings stay comparable -- which is the precondition for
        tuning ``far_field_size`` at all.
        """
        unpadded = _optimizer().forward_intensity(_truth(), _flat_phase())
        padded = _optimizer(far_field_size=256).forward_intensity(
            _truth(), _flat_phase()
        )
        decimated = padded[::4, ::4]

        assert decimated.shape == unpadded.shape
        assert np.allclose(decimated, unpadded / 16.0, rtol=0.0, atol=1e-9)
        assert np.allclose(
            decimated / decimated.sum(),
            unpadded / unpadded.sum(),
            rtol=0.0,
            atol=1e-12,
        )


class TestIrregularMeasurementIsAccepted:
    """Real frames are 250x248; the model grid is square. Both must work."""

    def test_update_accepts_an_irregular_measurement(self) -> None:
        """``update()`` resamples (248, 250) onto the padded grid instead of raising.

        This is the padded path end to end: the measurement arrives on a
        non-square grid, is zoomed to ``far_field_size``, and still produces a
        finite coefficient step.
        """
        optimizer = _optimizer(far_field_size=256, lr=0.1)
        reference = optimizer.forward_intensity(_truth(), _flat_phase())
        # A real CCD window is not square and not the model grid: 248x250.
        measured = np.array(reference[: IRREGULAR[0], : IRREGULAR[1]])
        assert measured.shape == IRREGULAR

        coefficients = optimizer.update(measured, _flat_phase())

        assert optimizer.last_loss is not None
        assert optimizer.last_loss > 0.0
        assert coefficients.shape == (N_COEFFICIENTS,)
        assert np.all(np.isfinite(coefficients))

    def test_update_accepts_the_native_grid_without_resampling(self) -> None:
        """A measurement already on the target grid is used as-is.

        The unpadded path is the common case on a simulation bench, so it must
        not regress while the padded path is being fixed.
        """
        optimizer = _optimizer()
        measured = optimizer.forward_intensity(_truth(), _flat_phase())

        coefficients = optimizer.update(measured, _flat_phase())

        assert measured.shape == optimizer._target_grid() == (REGION, REGION)
        assert np.all(np.isfinite(coefficients))
        assert optimizer.last_loss is not None

    def test_non_positive_peak_is_rejected(self) -> None:
        """An all-zero frame is a dead camera, not a valid target.

        Peak-normalising it would divide by zero and produce NaN gradients that
        propagate silently into the coefficients.
        """
        optimizer = _optimizer(far_field_size=256)

        with pytest.raises(ValueError, match="i_meas must have a positive peak"):
            optimizer.update(np.zeros((256, 256), dtype=np.float64), _flat_phase())

    def test_non_finite_measurement_is_rejected(self) -> None:
        """A NaN frame must fail loudly instead of poisoning the loss."""
        optimizer = _optimizer(far_field_size=256)
        measured = np.ones((256, 256), dtype=np.float64)
        measured[0, 0] = np.nan

        with pytest.raises(ValueError, match="i_meas must be finite"):
            optimizer.update(measured, _flat_phase())

    def test_non_2d_measurement_is_rejected(self) -> None:
        """A 1D frame carries no 2D far field to compare against."""
        optimizer = _optimizer(far_field_size=256)

        with pytest.raises(ValueError, match=r"i_meas must be 2D, got 1D"):
            optimizer.update(np.ones(256, dtype=np.float64), _flat_phase())


class TestPaddingValidation:
    """``far_field_size`` is validated up front, before any basis is built."""

    @pytest.mark.parametrize("far_field_size", [32, 0, -8])
    def test_padding_below_region_is_rejected(self, far_field_size: int) -> None:
        """Padding below ``region`` would shrink the pupil and lose aperture.

        The error names ``region`` so the caller can see the constraint that
        failed rather than just seeing a bad number.
        """
        with pytest.raises(
            ValueError,
            match=rf"far_field_size must be None or an integer >= region \({REGION}\)",
        ):
            _optimizer(far_field_size=far_field_size)

    def test_float_padding_is_rejected(self) -> None:
        """A float would silently truncate; ``int`` is required explicitly.

        The ``cast`` is deliberate and local: the runtime guard is exactly what
        is under test, and a computed padding value would reach this call
        untyped in real use. The annotation says ``int | None``, so only the
        runtime check can catch this.
        """
        with pytest.raises(
            ValueError,
            match=rf"far_field_size must be None or an integer >= region \({REGION}\), got 128\.0",
        ):
            _optimizer(far_field_size=cast(int, 128.0))

    def test_bool_padding_is_rejected(self) -> None:
        """``True`` is an ``int`` in Python but never a meaningful grid size."""
        with pytest.raises(
            ValueError,
            match=rf"far_field_size must be None or an integer >= region \({REGION}\), got True",
        ):
            _optimizer(far_field_size=True)

    def test_none_padding_falls_back_to_region(self) -> None:
        """``None`` is the documented "no padding" spelling, not an error."""
        optimizer = _optimizer(far_field_size=None)

        assert optimizer.far_field_size == REGION
        assert optimizer._target_grid() == (REGION, REGION)

    @pytest.mark.parametrize("noll", [0, -1, N_COEFFICIENTS + 1])
    def test_frozen_modes_outside_the_noll_range_are_rejected(
        self, noll: int
    ) -> None:
        """``frozen_modes`` are 1-based Noll indices, so 0 is out of range.

        The mask is indexed ``noll - 1``, so accepting a 0 would silently write
        to ``mask[-1]`` and freeze the wrong mode. The error names the valid
        range so the caller sees which constraint failed.
        """
        with pytest.raises(
            ValueError,
            match=rf"frozen_modes entries must be Noll indices in 1\.\.{N_COEFFICIENTS}",
        ):
            _optimizer(frozen_modes=(noll,))


class TestPaddingMakesCoefficientsIdentifiable:
    """The reason ``far_field_size`` exists, pinned as a numeric claim.

    Judging a fit by loss alone would pick the *unpadded* run here: it ends at
    4.8e-07 against the padded run's 8.5e-13, yet its coefficients are off by
    1.31 while the padded run recovers them to 2e-04. A low loss on an
    undersampled grid measures how well the model reproduces a blob, not
    whether the right aberration was found.
    """

    @pytest.mark.parametrize("far_field_size", [128, 256])
    def test_padded_fit_recovers_the_truth(self, far_field_size: int) -> None:
        """With padding, the fitted coefficients match the injected truth."""
        optimizer = _optimizer(far_field_size=far_field_size, lr=0.1)
        measured = optimizer.forward_intensity(_truth(), _flat_phase())

        result = optimizer.run(measured, _flat_phase())

        assert _fit_error(result.coefficients) < 1e-2
        assert result.converged

    def test_unpadded_fit_reaches_a_small_loss_with_wrong_coefficients(self) -> None:
        """The counter-example that makes the loss number untrustworthy alone.

        This is a genuine property of the coarse grid, not a flaky threshold:
        the spot is ~2.3 px wide there, so the residual cannot separate the
        coefficients even though it keeps improving.
        """
        optimizer = _optimizer(far_field_size=None, lr=0.1)
        measured = optimizer.forward_intensity(_truth(), _flat_phase())

        result = optimizer.run(measured, _flat_phase())

        assert result.history["loss"][-1] < 1e-6
        assert _fit_error(result.coefficients) > 0.5

    @pytest.mark.parametrize("far_field_size", PADDINGS)
    def test_frozen_modes_are_never_trained(self, far_field_size: int | None) -> None:
        """Frozen modes stay exactly zero -- masking, not regularisation.

        ``frozen_modes`` builds a hard 0/1 mask, so the modes must be bit-exact
        zero after fitting rather than merely small.
        """
        optimizer = _optimizer(far_field_size=far_field_size, lr=0.1)
        measured = optimizer.forward_intensity(_truth(), _flat_phase())

        coefficients = optimizer.run(measured, _flat_phase()).coefficients

        # ``frozen_modes`` takes 1-based Noll indices and masks ``noll - 1``,
        # so Noll 1/2/3 (piston, tilt-x, tilt-y) live at array slots 0/1/2.
        frozen_slots = [noll - 1 for noll in FROZEN]
        assert np.array_equal(
            coefficients[frozen_slots], np.zeros(len(FROZEN))
        )

    def test_fit_is_deterministic_for_a_fixed_measurement(self) -> None:
        """Same truth, same padding, same coefficients -- no hidden randomness.

        ``seed`` defaults to ``None`` and the fit is a pure gradient descent, so
        any run-to-run difference would mean state leaked between runs.
        """
        measured = _optimizer(far_field_size=128).forward_intensity(
            _truth(), _flat_phase()
        )

        first = _optimizer(far_field_size=128, lr=0.1).run(measured, _flat_phase())
        second = _optimizer(far_field_size=128, lr=0.1).run(measured, _flat_phase())

        assert np.array_equal(first.coefficients, second.coefficients)