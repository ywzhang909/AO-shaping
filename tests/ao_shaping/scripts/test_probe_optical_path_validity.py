"""Decision contracts for the optical-path validity probe.

The probe exists to answer "can the bench be calibrated right now?" *before* a
geometry sweep dies with "the spot width does not respond to any of them". The
classification that keeps it honest is the separation of two failures that look
identical in the raw numbers:

* **low_snr** -- the light is there, but the measurement cannot resolve what a
  working panel would produce (the width meter is too coarse);
* **no_response** -- the measurement is sharp enough, and the panel still does
  nothing.

Everything here is offline -- synthetic frames and synthetic state dicts in,
summary/verdict dicts out, no camera and no device.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "probe_optical_path_validity.py"


def _load_probe():
    """Load the probe by path (it lives under ``scripts/slm``, not on the path)."""
    spec = importlib.util.spec_from_file_location("probe_optical_path_validity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


def _gaussian_frame(peak: float, fwhm_px: float, seed: int) -> np.ndarray:
    """A Gaussian spot of ``peak`` counts and ``fwhm_px`` width plus shot noise."""
    shape = (300, 300)
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    sigma = fwhm_px / 2.355
    clean = peak * np.exp(
        -((rows - 150.0) ** 2 + (cols - 150.0) ** 2) / (2.0 * sigma**2)
    )
    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(0.0, np.sqrt(max(peak, 1.0)), shape)
    return np.clip(noisy, 0.0, 255.0)


def _ladder(peaks: list[tuple[float, float]]) -> list[dict]:
    """A synthetic exposure ladder: ``[(exposure_ms, peak_counts), ...]``.

    The noise floor mirrors the measured one (median 0, sigma 1 count on the
    Daheng MER2) so the threshold is ``0 + 4 * 1 = 4 counts``.
    """
    return [
        {
            "requested_ms": ms,
            "effective_ms": ms,
            "peak_raw": peak,
            "noise_median_counts": 0.0,
            "noise_sigma_counts": 1.0,
            "noise_threshold_counts": 4.0,
        }
        for ms, peak in peaks
    ]


def _state(amplitude: float, peak: float, fwhm: float, seed: int, n: int = 3) -> dict:
    """A synthetic defocus state with ``n`` noisy frames."""
    frames = [_gaussian_frame(peak, fwhm, seed + i) for i in range(n)]
    stats = [probe.frame_stats(frame) for frame in frames]
    return {
        "tag": f"{amplitude:+g}",
        "defocus_rad": amplitude,
        "peak_raw": float(np.mean([s.peak_raw for s in stats])),
        "peak_raw_per_frame": [s.peak_raw for s in stats],
        "fwhm_px": float(np.median([s.fwhm_px for s in stats])),
        "fwhm_per_frame": [s.fwhm_px for s in stats],
        "second_moment_radius_px": stats[0].second_moment_radius_px,
        "centroid_x": stats[0].centroid_x,
        "centroid_y": stats[0].centroid_y,
        "hollowness": stats[0].hollowness,
    }


#: A bench with a usable exposure window: 2..50 ms, peak 8..169 counts.
#: (The 1 ms step peaks at exactly 4 counts -- on the threshold, not above it.)
GOOD_LADDER = _ladder(
    [(0.05, 2.0), (0.2, 2.0), (1.0, 4.0), (2.0, 8.0), (5.0, 19.0), (10.0, 39.0),
     (20.0, 69.0), (50.0, 169.0)]
)


def test_ladder_reports_noise_floor_and_usable_window() -> None:
    """The usable window is the set of steps above the noise floor and below saturation."""
    summary = probe.analyse_ladder(GOOD_LADDER, 255.0)

    noise = summary["noise_floor"]
    assert noise["exposure_ms"] == pytest.approx(0.05)
    assert noise["threshold_counts"] == pytest.approx(4.0)  # median 0 + 4 * 1
    # The 1 ms step peaks at exactly 4 counts: it does not clear the threshold.
    assert summary["usable_window_ms"] == [2.0, 5.0, 10.0, 20.0, 50.0]
    # The recommended exposure is the one whose peak lands in the 100..200 band.
    assert summary["recommended_exposure"]["exposure_ms"] == pytest.approx(50.0)
    assert summary["recommended_exposure"]["in_target_band"] is True


def test_ladder_with_no_usable_step_is_reported_as_empty() -> None:
    """A ladder that never rises above the noise floor has no usable window."""
    summary = probe.analyse_ladder(_ladder([(0.05, 2.0), (0.2, 2.0), (1.0, 3.0)]), 255.0)

    assert summary["usable_window_ms"] == []
    assert summary["recommended_exposure"] is None


def test_defocus_response_is_measured_at_every_amplitude() -> None:
    """A response that only shows up at one sign of the amplitude still counts.

    The bench's own defocus offset makes +A and -A behave very differently, so
    judging a single amplitude would miss the response entirely.
    """
    states = [
        _state(0.0, 165.0, 80.0, 10),
        _state(0.5, 184.0, 75.0, 20),
        _state(1.0, 213.0, 72.0, 30),
        _state(-1.0, 130.0, 83.0, 40),
        _state(0.0, 169.0, 80.0, 50),
    ]
    response = probe.analyse_defocus_response(states, 2.4)

    assert response["responds"] is True
    assert response["peak_statistically_significant"] is True
    # The largest excursion is at +1 rad (peak rises by ~29%), not at -1 rad.
    assert response["peak_signal_at_rad"] == pytest.approx(1.0)
    assert response["peak_relative_change"] > 0.2


def test_verdict_ok_when_the_defocus_response_is_significant() -> None:
    """Signal plus a significant response => the bench can be calibrated."""
    states = [
        _state(0.0, 165.0, 80.0, 10),
        _state(1.0, 213.0, 72.0, 30),
        _state(-1.0, 130.0, 83.0, 40),
        _state(0.0, 169.0, 80.0, 50),
    ]
    summary = probe.analyse_ladder(GOOD_LADDER, 255.0)
    response = probe.analyse_defocus_response(states, 2.4)
    verdict = probe.decide(summary, response)

    assert verdict["state"] == "ok"
    assert verdict["exit_code"] == 0


def test_verdict_low_snr_when_the_response_is_drowned_in_noise() -> None:
    """A dim bench whose width meter cannot resolve anything is low_snr, not no_response.

    This is the failure the probe was written for: on 2026-10-22 the geometry
    sweep reported "the spot width does not respond" while the real cause was
    that the exposure window was 1-10 ms with peaks of 24-213 counts.
    """
    states = [
        _state(0.0, 30.0, 12.0, 10),
        _state(1.0, 30.0, 12.1, 30),
        _state(-1.0, 30.0, 12.2, 40),
        _state(0.0, 30.0, 12.0, 50),
    ]
    summary = probe.analyse_ladder(GOOD_LADDER, 255.0)
    response = probe.analyse_defocus_response(states, 2.4)
    verdict = probe.decide(summary, response)

    assert verdict["state"] == "low_snr"
    assert verdict["exit_code"] == 2
    assert response["precision_ok"] is False
    assert any("淹没在噪声里" in reason for reason in verdict["reasons"])


def test_verdict_no_response_when_the_measurement_is_sharp_but_nothing_moves() -> None:
    """A bright, well-resolved spot that ignores defocus means the panel is suspect."""
    states = [
        _state(0.0, 213.0, 12.0, 10),
        _state(1.0, 212.0, 12.1, 30),
        _state(-1.0, 212.0, 12.2, 40),
        _state(0.0, 213.0, 12.0, 50),
    ]
    summary = probe.analyse_ladder(GOOD_LADDER, 255.0)
    response = probe.analyse_defocus_response(states, 2.4)
    verdict = probe.decide(summary, response)

    assert verdict["state"] == "no_response"
    assert verdict["exit_code"] == 2
    assert response["precision_ok"] is True
    assert any("slm_diagnose" in item for item in verdict["advice"])


def test_verdict_low_snr_when_no_exposure_carries_signal() -> None:
    """No usable exposure at all is a signal problem, whatever the defocus says."""
    summary = probe.analyse_ladder(_ladder([(0.05, 2.0), (0.2, 2.0), (1.0, 3.0)]), 255.0)
    states = [_state(0.0, 3.0, 12.0, 10), _state(1.0, 3.0, 12.1, 30)]
    response = probe.analyse_defocus_response(states, 2.4)
    verdict = probe.decide(summary, response)

    assert verdict["state"] == "low_snr"
    assert verdict["exit_code"] == 2


def test_defocus_phase_is_centred_on_the_pupil_and_is_raw_radians() -> None:
    """The defocus disc follows the pupil centre, and the phase stays unwrapped."""
    phase = probe.build_defocus_phase(1.0, 450, (960, 600))

    assert phase.shape == (1200, 1920)
    assert phase.dtype == np.float64
    # Unit-RMS Noll defocus reaches +-sqrt(3) rad at the pupil edge.
    assert float(phase.max()) == pytest.approx(np.sqrt(3.0), rel=1e-3)
    assert float(phase.min()) == pytest.approx(-np.sqrt(3.0), rel=1e-3)
    rows = np.flatnonzero(phase.any(axis=1))
    cols = np.flatnonzero(phase.any(axis=0))
    assert (rows[0], rows[-1]) == (150, 1049)
    assert (cols[0], cols[-1]) == (510, 1409)

    # A pupil centre off the panel centre moves the whole disc.
    offset = probe.build_defocus_phase(1.0, 450, (960, 720))
    rows = np.flatnonzero(offset.any(axis=1))
    assert (rows[0], rows[-1]) == (270, 1170)


def test_defocus_phase_at_the_panel_centre_matches_the_canonical_call() -> None:
    """The default path is byte-identical to generating on the whole panel."""
    phase = probe.build_defocus_phase(1.0, 450, (960, 600))
    canonical = np.nan_to_num(
        probe.generate_zernike_phase(
            {4: 1.0}, resolution=(1920, 1200), n_max=4, radius=450
        )
    )

    assert np.array_equal(phase, canonical)
