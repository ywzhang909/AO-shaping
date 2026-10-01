"""Far-field spot sampling in the 2f-Fourier SLM simulation.

The simulated CCD image is the FFT of the SLM pupil, so the far-field pixel
pitch is fixed by the transform: with no zero-padding the 0-order spot landed at
roughly **1.8 px FWHM**, i.e. barely one sample across the spot. Every
power-ratio metric then divided by a numerically unresolved peak.

Zero-padding the pupil before ``fft2`` shrinks the far-field pixel pitch by the
padding factor, which is the digital equivalent of the longer effective focal
length a real bench gets by narrowing the camera ROI. Cropping the padded result
back to a bounded window keeps the cached array (and the shot/read noise applied
to it in :meth:`far_field_noisy`) small.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem


def _fwhm_1d(profile: np.ndarray) -> float:
    """Full width at half maximum with linearly interpolated crossings."""
    peak = float(profile.max())
    if peak <= 0:
        return 0.0
    half = 0.5 * peak
    top = int(np.argmax(profile))
    lo = top
    while lo > 0 and profile[lo] > half:
        lo -= 1
    hi = top
    while hi < profile.size - 1 and profile[hi] > half:
        hi += 1
    p_lo, p_top, p_hi = float(profile[lo]), float(profile[top]), float(profile[hi])
    left = lo + (half - p_lo) * (top - lo) / (p_top - p_lo) if p_top != p_lo else float(top)
    right = hi + (p_top - half) * (hi - top) / (p_top - p_hi) if p_top != p_hi else float(top)
    return float(right - left)


def _spot_fwhm(image: np.ndarray) -> tuple[float, float]:
    """FWHM along the row and column through the brightest pixel."""
    row, col = np.unravel_index(int(np.argmax(image)), image.shape)
    return _fwhm_1d(image[row, :]), _fwhm_1d(image[:, col])


def test_unpadded_far_field_leaves_the_spot_nearly_unresolved() -> None:
    """Documents the defect the padding fixes: ~1.8 px FWHM, i.e. sub-sampled."""
    system = SimPibSystem(seed=42, far_field_padding=1)
    fwhm_x, fwhm_y = _spot_fwhm(system.far_field())
    assert min(fwhm_x, fwhm_y) < 3.0, (
        f"padding=1 should stay unresolved, got {fwhm_x:.2f}x{fwhm_y:.2f} px"
    )


def test_default_padding_oversamples_the_spot() -> None:
    """The default must resolve the spot across several pixels."""
    fwhm_x, fwhm_y = _spot_fwhm(SimPibSystem(seed=42).far_field())
    assert min(fwhm_x, fwhm_y) >= 4.0, (
        f"spot must span >=4 px to be sampled, got {fwhm_x:.2f}x{fwhm_y:.2f}"
    )


def test_spot_width_matches_the_analytic_sampling_law() -> None:
    """Spot width must be ``2*sqrt(2 ln2) * N / (2*pi*w0)`` per unit of padding.

    This is the quantity the padding exists to control, so assert it against the
    closed form rather than a hand-tuned ratio. A pixel of tolerance covers the
    grid quantisation of a ~2-7 px FWHM measured on discrete samples.
    """
    from ao_shaping.drivers.sim.slm_pib_sim import BEAM_W0, SLM_W

    expected_per_pad = 2.0 * np.sqrt(2.0 * np.log(2.0)) * SLM_W / (2.0 * np.pi * BEAM_W0)
    assert expected_per_pad == pytest.approx(1.80, abs=0.01), (
        "analytic constant changed; re-derive it"
    )

    for factor in (1, 2, 4):
        image = SimPibSystem(seed=42, far_field_padding=factor).far_field()
        fwhm_x, fwhm_y = _spot_fwhm(image)
        expected = expected_per_pad * factor
        assert abs(fwhm_x - expected) <= 1.0, (
            f"padding={factor}: FWHM_x {fwhm_x:.2f} px vs analytic {expected:.2f}"
        )
        assert abs(fwhm_y - expected) <= 1.0, (
            f"padding={factor}: FWHM_y {fwhm_y:.2f} px vs analytic {expected:.2f}"
        )


def test_spot_width_increases_monotonically_with_padding() -> None:
    widths = []
    for factor in (1, 2, 4):
        image = SimPibSystem(seed=42, far_field_padding=factor).far_field()
        widths.append(min(_spot_fwhm(image)))
    assert widths[0] < widths[1] < widths[2], f"widths did not scale: {widths}"


def test_far_field_is_cropped_to_a_bounded_window() -> None:
    """The padded result must not be retained in full (memory + noise cost)."""
    system = SimPibSystem(seed=42, far_field_padding=4, far_field_window=512)
    image = system.far_field()
    assert image.shape == (512, 512)
    assert np.all(np.isfinite(image))
    assert image.max() > 0


def test_window_larger_than_the_padded_array_is_clamped_not_crashed() -> None:
    system = SimPibSystem(seed=42, far_field_padding=1, far_field_window=4096)
    image = system.far_field()
    assert image.shape[0] <= 1920 and image.shape[1] <= 1920


def test_spot_stays_centred_after_cropping() -> None:
    """The 0-order must remain at the frame centre so the crop keeps it."""
    image = SimPibSystem(seed=42, far_field_padding=4, far_field_window=512).far_field()
    row, col = np.unravel_index(int(np.argmax(image)), image.shape)
    assert abs(row - image.shape[0] // 2) <= 1
    assert abs(col - image.shape[1] // 2) <= 1


def test_padding_does_not_change_the_peak_normalisation() -> None:
    for factor in (1, 4):
        image = SimPibSystem(seed=42, far_field_padding=factor).far_field()
        assert np.isclose(image.max(), 100.0), "far field is peak-normalised to 100"


def test_noise_is_applied_over_the_cropped_window() -> None:
    """A bounded crop must keep ``far_field_noisy`` cheap and same-shaped."""
    system = SimPibSystem(seed=42, far_field_padding=4, far_field_window=512)
    noisy = system.far_field_noisy()
    assert noisy.shape == system.far_field().shape
    assert np.all(np.isfinite(noisy))


def test_disturbance_still_bites_after_padding() -> None:
    """Padding must not wash out the disturbance's effect on the far field."""
    from ao_shaping.drivers.sim.disturbance import DisturbanceConfig, SimDisturbance

    clean = SimPibSystem(seed=42).far_field()
    dirty = SimPibSystem(
        seed=42,
        disturbance=SimDisturbance(
            DisturbanceConfig(mode="static", cn2=2e-13), (1200, 1920)
        ),
    ).far_field()
    assert not np.array_equal(clean, dirty)
    corr = float(np.corrcoef(clean.ravel(), dirty.ravel())[0, 1])
    assert corr < 0.999, f"disturbance must still change the spot, corr={corr}"