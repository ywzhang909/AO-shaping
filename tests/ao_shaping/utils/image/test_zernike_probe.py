"""HMWS five-point Zernike probe: pure function + end-to-end 2f twin bench tests.

Tests the probe-phase template on :class:`~ao_shaping.utils.wavefront.pattern_helper.PatternHelper`
and the CCD-focal-plane coefficient estimator :mod:`ao_shaping.utils.image.zernike_probe`,
the two pieces added for the "探测相位补充到 Pattern Helper" + "CCD 画面解析 Zernike
相差通用算法" (Pathak & Jomy 2026, APL, ds-AO) task.

Everything is a **pure simulation**: the end-to-end case synthesises the five
focal-plane frames with the CPU ``Sim2f`` digital twin (a small 64x64 grid), never
an SLM / CCD / laser.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.zernike_probe import (
    CALIBRATION_CONSTANT,
    FivePointIntensities,
    ProbeResult,
    estimate_zernike_probe,
    five_point_estimate,
    probe_peak_intensity,
)
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper


# ---------------------------------------------------------------------------
# I. Pure function: five_point_estimate against a synthetic quadratic model
# ---------------------------------------------------------------------------
def _quadratic_intensities(
    a: float, b: float, c: float, d: float
) -> FivePointIntensities:
    """I(x) = d - c*x**2 sampled at the five probe total-aberrations (paper model).

    ``x`` is the total aberration on the axis: the true coefficient ``a`` plus the
    probe offset.  The five samples are ``x = a, a-b, a+b, a-2b, a+2b`` (the PSF
    peak is maximal when the total aberration vanishes, i.e. ``x = 0``).
    """

    def itotal(x: float) -> float:
        return d - c * x * x

    return FivePointIntensities(
        ia=itotal(a),
        im=itotal(a - b),
        ip=itotal(a + b),
        imm=itotal(a - 2 * b),
        ipp=itotal(a + 2 * b),
    )


def test_five_point_estimate_recovers_coefficient() -> None:
    """Under ``I(x) = d - c x^2`` the paper formula recovers ``a`` to ~0.996."""
    b = 0.3
    for a in (0.1, -0.25, 0.5, -0.05):
        x = _quadratic_intensities(a, b, c=1.0, d=5.0)
        est = five_point_estimate(x, b)
        assert est == pytest.approx(a, abs=0.01)


def test_five_point_estimate_bias_invariant() -> None:
    """For a fixed true ``a``, the estimator is (nearly) independent of ``b``."""
    a = 0.2
    for b in (0.1, 0.25, 0.5, 1.0):
        x = _quadratic_intensities(a, b, c=1.0, d=10.0)
        assert five_point_estimate(x, b) == pytest.approx(a, abs=0.01)


def test_five_point_degenerate_returns_nan() -> None:
    """A mode whose probe leaves the peak unchanged gives a degenerate estimate."""
    x = FivePointIntensities(ia=1.0, im=1.0, ip=1.0, imm=1.0, ipp=1.0)
    assert np.isnan(five_point_estimate(x, 0.3))


def test_five_point_rejects_nonpositive_bias() -> None:
    with pytest.raises(ValueError):
        five_point_estimate(_quadratic_intensities(0.1, 0.3, 1.0, 1.0), 0.0)
    with pytest.raises(ValueError):
        five_point_estimate(_quadratic_intensities(0.1, 0.3, 1.0, 1.0), -0.2)


def test_calibration_constant_matches_formula() -> None:
    """The 0.83 factor is the paper's empirical constant used in the formula."""
    assert CALIBRATION_CONSTANT == pytest.approx(0.83)


def test_peak_intensity_takes_full_array_max() -> None:
    frame = np.zeros((8, 8))
    frame[3, 4] = 42.5
    assert probe_peak_intensity(frame) == pytest.approx(42.5)


def test_peak_intensity_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        probe_peak_intensity(np.zeros((2, 2, 2)))


# ---------------------------------------------------------------------------
# II. estimate_zernike_probe framing / Noll handling
# ---------------------------------------------------------------------------
def _flat_five_frames() -> dict[str, np.ndarray]:
    """Five flat (all-zero) probe frames with the canonical keys."""
    return {k: np.zeros((8, 8)) for k in ("a", "m", "p", "mm", "pp")}


def test_estimate_zernike_probe_single_mode_from_mapping() -> None:
    b = 0.3
    a = 0.2
    # peak intensities follow the quadratic model; frames carry them as their max.
    x = _quadratic_intensities(a, b, c=1.0, d=5.0)
    frames = {
        "a": _frame_with_peak(x.ia),
        "m": _frame_with_peak(x.im),
        "p": _frame_with_peak(x.ip),
        "mm": _frame_with_peak(x.imm),
        "pp": _frame_with_peak(x.ipp),
    }
    res = estimate_zernike_probe(frames, [4], b)
    assert len(res) == 1
    assert isinstance(res[0], ProbeResult)
    assert res[0].noll_index == 4  # defocus
    assert res[0].coefficient_rad == pytest.approx(a, abs=0.01)
    assert res[0].valid


def test_estimate_zernike_probe_accepts_multi_mode_list() -> None:
    b = 0.2
    res = estimate_zernike_probe(
        [
            {
                "a": _frame_with_peak(1.0),
                "m": _frame_with_peak(1.0),
                "p": _frame_with_peak(1.0),
                "mm": _frame_with_peak(1.0),
                "pp": _frame_with_peak(1.0),
            },
            {
                "a": _frame_with_peak(1.0),
                "m": _frame_with_peak(1.0),
                "p": _frame_with_peak(1.0),
                "mm": _frame_with_peak(1.0),
                "pp": _frame_with_peak(1.0),
            },
        ],
        [4, 11],
        b,
    )
    assert [r.noll_index for r in res] == [4, 11]


def test_estimate_zernike_probe_single_mapping_rejects_multi_index() -> None:
    frames = _flat_five_frames()
    with pytest.raises(ValueError):
        estimate_zernike_probe(frames, [4, 11], 0.3)


def test_estimate_zernike_probe_mismatched_group_count_raises() -> None:
    frames = _flat_five_frames()
    with pytest.raises(ValueError):
        estimate_zernike_probe([frames], [4, 11], 0.3)


def test_estimate_zernike_probe_degenerate_marks_invalid() -> None:
    frames = _flat_five_frames()
    res = estimate_zernike_probe(frames, [4], 0.3)
    assert len(res) == 1
    assert not res[0].valid
    assert np.isnan(res[0].coefficient_rad)


def _frame_with_peak(peak: float) -> np.ndarray:
    frame = np.zeros((8, 8))
    frame[4, 4] = peak
    return frame


# ---------------------------------------------------------------------------
# III. PatternHelper.generate_probe_phase template
# ---------------------------------------------------------------------------
def test_generate_probe_phase_returns_raw_unwrapped_radians() -> None:
    ph = PatternHelper(resolution=(64, 64))
    # A big coefficient -> phase far beyond pi must NOT be wrapped.
    probe = ph.generate_probe_phase(coefficients={(2, 0): 6.0}, bias_rad=0.0, radius=20)
    assert probe.dtype == np.float64
    assert np.nanmax(probe) > np.pi  # unwrapped raw radians preserved
    # aperture: center sample (defocus zero at center) small, rim large
    assert probe.max() > 3.0


def test_generate_probe_phase_zero_outside_aperture() -> None:
    ph = PatternHelper(resolution=(64, 64))
    probe = ph.generate_probe_phase(coefficients={(2, 0): 1.0}, radius=20)
    # corners outside a radius-20 disc at (32,32) must be exactly 0.
    assert probe[0, 0] == 0.0
    assert probe[0, -1] == 0.0
    # and the aperture centre is non-zero
    assert probe[32, 32] != 0.0


def test_generate_probe_phase_delegates_zernike_math() -> None:
    """The generated mode equals the canonical engine's output (unchanged by bias)."""
    from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

    ph = PatternHelper(resolution=(64, 64), bits=10)
    gen = ZernikeGenerator(resolution=(64, 64), radius=20, n_orders=6)
    ref = np.where(gen.mask.astype(bool), gen.generate_polynomial({(2, 0): 1.2}), 0.0)

    probe = ph.generate_probe_phase(coefficients={(2, 0): 1.2}, radius=20)
    np.testing.assert_allclose(probe, np.asarray(ref, dtype=np.float64), atol=1e-9)


def test_generate_probe_phase_bias_adds_constant_in_aperture() -> None:
    ph = PatternHelper(resolution=(64, 64))
    probe = ph.generate_probe_phase(coefficients={(2, 0): 1.0}, bias_rad=0.5, radius=20)
    no_bias = ph.generate_probe_phase(
        coefficients={(2, 0): 1.0}, bias_rad=0.0, radius=20
    )
    # inside the aperture they differ by exactly the bias
    mask = no_bias != 0.0  # a proxy for "inside aperture with a signal"
    # defocus has a nodal ring at r*rho=... it is zero on one circle; use edge pixels
    in_ap = no_bias != 0.0
    assert np.allclose(probe[in_ap] - no_bias[in_ap], 0.5)
    # outside the aperture both are 0
    out_ap = ~(np.abs(probe) > 0)  # truly zero outside
    assert np.all(probe[out_ap] == 0.0) and np.all(no_bias[out_ap] == 0.0)


def test_generate_probe_phase_empty_is_zeros() -> None:
    ph = PatternHelper(resolution=(16, 16))
    p = ph.generate_probe_phase()
    assert p.shape == (16, 16)
    assert np.all(p == 0.0)


# ---------------------------------------------------------------------------
# IV. End-to-end 2f twin bench (CPU): synthesize 5 frames, recover the mode
# ---------------------------------------------------------------------------
def test_end_to_end_defocus_via_sim2f_twin() -> None:
    """Recover a pure defocus coefficient from five Sim2f focal-plane frames.

    The 2f twin maps a pupil phase to the CCD-frame intensity via FFT.  A
    defocus aberration ``k * Z(2,0)`` spreads the PSF, so the on-axis (peak)
    intensity varies quadratically with ``k`` about the in-focus maximum --
    exactly the small-aberration model the five-point formula relies on.  We
    probe mode (2,0) at bias ``b`` and recover its coefficient.
    """
    from ml.sim2f.config import Sim2fConfig
    from ml.sim2f.optics import Sim2f

    sim = Sim2f(Sim2fConfig(ccd_px=64, grid=64, exposure_ms=0.0), device="cpu")
    grid = 64
    ph = PatternHelper(resolution=(grid, grid))

    # True coeff of defocus we are trying to recover.
    a_true = 0.2
    b = 0.3

    def frame_for(offset: float) -> np.ndarray:
        # pupil phase = true defocus + probe bias offset
        phase = ph.generate_probe_phase(
            coefficients={(2, 0): a_true + offset}, radius=28
        )
        p = sim.forward(np.asarray(phase, dtype=np.float32)[None, :, :])[0, 0].numpy()
        return p

    frames = {
        "a": frame_for(0.0),
        "m": frame_for(-b),
        "p": frame_for(+b),
        "mm": frame_for(-2 * b),
        "pp": frame_for(+2 * b),
    }
    res = estimate_zernike_probe(frames, [4], b)
    assert len(res) == 1
    assert res[0].valid
    # generous tolerance: the formula is an empirical quadratic approximation
    assert abs(res[0].coefficient_rad - a_true) < 0.1


def test_end_to_end_spherical_via_sim2f_twin() -> None:
    """Same probe on Noll 11 (spherical) also recovers its coefficient."""
    from ml.sim2f.config import Sim2fConfig
    from ml.sim2f.optics import Sim2f

    sim = Sim2f(Sim2fConfig(ccd_px=64, grid=64, exposure_ms=0.0), device="cpu")
    grid = 64
    ph = PatternHelper(resolution=(grid, grid))

    a_true = 0.15
    b = 0.3

    def frame_for(offset: float) -> np.ndarray:
        phase = ph.generate_probe_phase(
            coefficients={(4, 0): a_true + offset}, radius=28
        )
        p = sim.forward(np.asarray(phase, dtype=np.float32)[None, :, :])[0, 0].numpy()
        return p

    frames = {
        "a": frame_for(0.0),
        "m": frame_for(-b),
        "p": frame_for(+b),
        "mm": frame_for(-2 * b),
        "pp": frame_for(+2 * b),
    }
    res = estimate_zernike_probe(frames, [11], b)
    assert len(res) == 1 and res[0].valid
    assert abs(res[0].coefficient_rad - a_true) < 0.1
