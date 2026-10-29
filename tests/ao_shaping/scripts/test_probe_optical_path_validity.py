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
import json
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


# -- 光通量 (weak_path): 光被挡掉一部分时「信号还在, 但已经不是原来那束光」 --------
#
# 2026-10-29 实测把这台实验系统的光路挡掉一部分后, 三态判据仍然给 ``ok``:
# 峰值 68 counts 远高于噪声底 4 counts, 离焦响应也统计显著。唯一动了的量是
# **每毫秒峰值** —— 6.15 → 1.38 counts/ms (22%), 而总能量只掉约 30% (遮挡把光
# 散掉而不是均匀减光)。所以光通量必须单独判, 且只跟本系统自己的历史比。


def _write_history_run(
    root: Path,
    name: str,
    *,
    state: str = "ok",
    counts_per_ms: float = 6.0,
    slm_serial: str | None = "22030108",
    cam_serial: str | None = "FJB24112232",
    full_scale: float = 255.0,
) -> Path:
    """An on-disk probe run shaped the way :func:`bench_reference` reads it."""
    run = root / name / "20261029_000000Z"
    run.mkdir(parents=True)
    record = {
        "verdict": {"state": state},
        "ladder_summary": {"counts_per_ms": counts_per_ms},
        "exposure_ladder": [{"full_scale": full_scale}],
        "device": {"slm_serial": slm_serial, "cam_serial": cam_serial},
    }
    (run / "optical_path_probe.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    return run


def test_ladder_flux_slope_ignores_the_saturated_steps() -> None:
    """A saturated step is clipped at full scale: fitting it would fake an attenuation."""
    ladder = _ladder([(1.0, 6.0), (5.0, 31.0), (10.0, 61.0), (20.0, 255.0), (50.0, 255.0)])
    summary = probe.analyse_ladder(ladder, 255.0)

    assert summary["counts_per_ms_window_ms"] == [1.0, 5.0, 10.0]
    # The unsaturated steps are linear at ~6.1 counts/ms; including the clipped
    # 255-count steps would drag the slope towards zero.
    assert summary["counts_per_ms"] == pytest.approx(6.1, rel=0.03)


def test_bench_reference_is_the_median_of_ok_runs_only(tmp_path: Path) -> None:
    """A run already judged ``low_snr`` is not "what this bench normally looks like".

    Feeding it into the median would let a blocked path lower the standard until
    it judges itself normal.
    """
    for index, value in enumerate((6.0, 6.2, 6.1)):
        _write_history_run(tmp_path, f"ok{index}", counts_per_ms=value)
    _write_history_run(tmp_path, "dim", state="low_snr", counts_per_ms=0.02)

    reference = probe.bench_reference(tmp_path)

    assert reference is not None
    assert reference["n"] == 3
    assert reference["counts_per_ms"] == pytest.approx(6.1)


def test_bench_reference_refuses_fewer_than_three_ok_runs(tmp_path: Path) -> None:
    """One or two samples have no meaningful median -- better to withhold the verdict."""
    _write_history_run(tmp_path, "a", counts_per_ms=6.0)
    _write_history_run(tmp_path, "b", counts_per_ms=6.2)

    assert probe.bench_reference(tmp_path) is None


def test_bench_reference_ignores_a_different_bench(tmp_path: Path) -> None:
    """After a SLM or camera swap the old flux is another bench's flux."""
    for index in range(3):
        _write_history_run(tmp_path, f"ok{index}", counts_per_ms=6.0 + 0.1 * index)
    for index in range(3):
        _write_history_run(
            tmp_path, f"other{index}", counts_per_ms=0.5, slm_serial="99999999"
        )

    reference = probe.bench_reference(
        tmp_path, device={"slm_serial": "22030108", "cam_serial": "FJB24112232"}
    )

    assert reference is not None
    assert reference["n"] == 3
    assert reference["counts_per_ms"] == pytest.approx(6.1)


def test_bench_reference_recomputes_the_flux_from_legacy_records(tmp_path: Path) -> None:
    """Runs written before 2026-10-29 carry no ``counts_per_ms`` -- derive it.

    Without this the reference stays empty on a bench that already has history,
    so the gate would silently never fire (the first live run hit exactly this:
    it reported "history has fewer than 3 ok runs" against 19 of them).
    """
    ladder = _ladder([(1.0, 6.0), (5.0, 31.0), (10.0, 61.0)])
    probe.analyse_ladder(ladder, 255.0)  # 把 usable / saturated 标志写到档位上
    for index in range(3):
        run = _write_history_run(tmp_path, f"legacy{index}")
        record = json.loads((run / "optical_path_probe.json").read_text(encoding="utf-8"))
        record.pop("ladder_summary")
        record["exposure_ladder"] = ladder
        (run / "optical_path_probe.json").write_text(
            json.dumps(record), encoding="utf-8"
        )

    reference = probe.bench_reference(tmp_path)

    assert reference is not None
    assert reference["n"] == 3
    assert reference["counts_per_ms"] == pytest.approx(6.1, rel=0.03)


def test_a_rebaselined_baseline_replaces_the_history(tmp_path: Path) -> None:
    """After a laser-power change the history is another regime's flux.

    2026-10-29 live: the reference collapsed to 3.45 counts/ms (only four stale
    2026-10-22 records matched the current SLM serial) while the bench measured
    28.22 -- "739%", still ``ok``, i.e. the gate stopped discriminating at all.
    """
    for index, value in enumerate((6.0, 6.2, 6.1)):
        _write_history_run(tmp_path, f"ok{index}", counts_per_ms=value)
    probe._save_reference_baseline(
        tmp_path / probe.REFERENCE_BASELINE_FILE,
        counts_per_ms=28.22,
        device={"slm_serial": "22030108", "cam_serial": "FJB24112232"},
        full_scale=4095.0,
    )

    reference = probe.bench_reference(
        tmp_path, device={"slm_serial": "22030108", "cam_serial": "FJB24112232"}
    )

    assert reference is not None
    assert reference["source"] == "rebaseline"
    assert reference["counts_per_ms"] == pytest.approx(28.22)


def test_a_rebaselined_baseline_does_not_survive_a_bench_swap(tmp_path: Path) -> None:
    """A baseline recorded on another SLM says nothing about this one."""
    for index, value in enumerate((6.0, 6.2, 6.1)):
        _write_history_run(
            tmp_path, f"ok{index}", counts_per_ms=value, slm_serial="23020027"
        )
    probe._save_reference_baseline(
        tmp_path / probe.REFERENCE_BASELINE_FILE,
        counts_per_ms=28.22,
        device={"slm_serial": "22030108", "cam_serial": "FJB24112232"},
        full_scale=4095.0,
    )

    reference = probe.bench_reference(
        tmp_path, device={"slm_serial": "23020027", "cam_serial": "FJB24112232"}
    )

    assert reference is not None
    assert reference["source"] == "history"


def test_the_reference_says_why_records_were_left_out(tmp_path: Path) -> None:
    """A silently thinned pool is how the gate went blind on 2026-10-29."""
    for index in range(3):
        _write_history_run(tmp_path, f"ok{index}", counts_per_ms=6.0 + 0.1 * index)
    for index in range(2):
        _write_history_run(
            tmp_path, f"other{index}", counts_per_ms=0.5, slm_serial="99999999"
        )
    _write_history_run(tmp_path, "dim", state="low_snr", counts_per_ms=0.02)

    reference = probe.bench_reference(
        tmp_path, device={"slm_serial": "22030108", "cam_serial": "FJB24112232"}
    )

    assert reference is not None
    assert reference["n"] == 3
    assert reference["skipped"]["serial_mismatch"] == 2
    assert reference["skipped"]["not_ok"] == 1


def _ok_inputs() -> tuple[dict, dict]:
    """The ladder + defocus response that the three-state rule judges ``ok``."""
    states = [
        _state(0.0, 165.0, 80.0, 10),
        _state(1.0, 213.0, 72.0, 30),
        _state(-1.0, 130.0, 83.0, 40),
        _state(0.0, 169.0, 80.0, 50),
    ]
    return (
        probe.analyse_ladder(GOOD_LADDER, 255.0),
        probe.analyse_defocus_response(states, 2.4),
    )


def test_verdict_weak_path_when_the_flux_is_far_below_the_reference() -> None:
    """Signal + response, but a fifth of the usual flux: the path is obstructed."""
    summary, response = _ok_inputs()
    flux = summary["counts_per_ms"]
    throughput = {
        "counts_per_ms": flux,
        "reference_counts_per_ms": flux / 0.22,
        "reference_source": "history",
        "reference_n": 19,
        "ratio": 0.22,
        "min_ratio": probe.DEFAULT_MIN_THROUGHPUT_RATIO,
    }

    verdict = probe.decide(summary, response, throughput=throughput)

    assert verdict["state"] == "weak_path"
    assert verdict["exit_code"] == 2
    assert any("光通量只有参考的 22%" in reason for reason in verdict["reasons"])
    # The obstruction is the thing to fix first -- raising the exposure only
    # restores the peak, not the light that never arrived.
    assert "遮挡" in verdict["advice"][0]
    assert not any("建议用曝光" in item for item in verdict["advice"])


def test_verdict_stays_ok_when_the_flux_matches_the_reference() -> None:
    """Ordinary run-to-run drift must not trip the gate."""
    summary, response = _ok_inputs()
    flux = summary["counts_per_ms"]
    throughput = {
        "counts_per_ms": flux,
        "reference_counts_per_ms": flux / 0.95,
        "reference_source": "history",
        "reference_n": 19,
        "ratio": 0.95,
        "min_ratio": probe.DEFAULT_MIN_THROUGHPUT_RATIO,
    }

    verdict = probe.decide(summary, response, throughput=throughput)

    assert verdict["state"] == "ok"
    assert verdict["exit_code"] == 0


def test_verdict_without_a_reference_is_unchanged() -> None:
    """``--no-reference`` (or too little history) falls back to the three states."""
    summary, response = _ok_inputs()

    verdict = probe.decide(summary, response, throughput=None)

    assert verdict["state"] == "ok"
    assert verdict["exit_code"] == 0


def test_weak_path_never_masks_a_harder_failure() -> None:
    """A panel that does not modulate stays ``no_response``, however dim the path."""
    states = [
        _state(0.0, 213.0, 12.0, 10),
        _state(1.0, 212.0, 12.1, 30),
        _state(-1.0, 212.0, 12.2, 40),
        _state(0.0, 213.0, 12.0, 50),
    ]
    summary = probe.analyse_ladder(GOOD_LADDER, 255.0)
    response = probe.analyse_defocus_response(states, 2.4)
    throughput = {
        "counts_per_ms": summary["counts_per_ms"],
        "reference_counts_per_ms": summary["counts_per_ms"] / 0.1,
        "reference_source": "history",
        "reference_n": 19,
        "ratio": 0.1,
        "min_ratio": probe.DEFAULT_MIN_THROUGHPUT_RATIO,
    }

    verdict = probe.decide(summary, response, throughput=throughput)

    assert verdict["state"] == "no_response"
    assert verdict["exit_code"] == 2
