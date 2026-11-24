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

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.slm.panel.focus_probe import FocusPoint, FocusProbeResult

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
    assert any("slm.panel.diagnose" in item for item in verdict["advice"])


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


# ---------------------------------------------------------------------------
# Bit depth / full scale: the S0 precondition is Mono12 (4095), not Mono8 (255)
# ---------------------------------------------------------------------------
#: The S0 run at ``data/debug/hw_window_20261118/optical_path_probe/s0/
#: 20261118_112316Z/optical_path_record.json`` recorded 8 bit / full_scale 255 even
#: though the window plan requires ``set_bit_depth(12)``. Two separate causes:
#: ``create_camera`` never passed ``bit_depth`` (Daheng stayed on its default
#: Mono8), and the ladder derived ``full_scale`` from ``np.iinfo(dtype).max`` --
#: MONO12 frames are ``uint16``, so that yields 65535 instead of 4095. 8 bit makes
#: 1 count ~7% of full scale, which invalidates any uniformity conclusion.


class _FakeMono12Camera:
    """A camera that honours the bit-depth contract: 12 bit, uint16 frames, 4095."""

    supported_bit_depths = (8, 12, 16)

    def __init__(self, *, bit_depth: int = 12, peak_counts: float = 3000.0) -> None:
        self.bit_depth = bit_depth
        self.peak_counts = peak_counts
        self.exposure_time_ms = 1.0
        self.full_scale_reads: list[float] = []

    # -- exposure contract (common.set/get_camera_exposure_ms) --------------
    def get_exposure_time_ms(self) -> float:
        return self.exposure_time_ms

    def reset_exposure_time(self, time_ms: float) -> float:
        self.exposure_time_ms = float(time_ms)
        return self.exposure_time_ms

    # -- full-scale contract: bit depth decides, the frame dtype does not ---
    def effective_full_scale(self, img: np.ndarray) -> float:
        assert np.asarray(img).dtype == np.uint16  # MONO12 rides in a uint16 container
        self.full_scale_reads.append(1.0)
        return float((1 << int(self.bit_depth)) - 1)

    # -- capture ------------------------------------------------------------
    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = False) -> np.ndarray:
        shape = (120, 120)
        rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
        sigma = 6.0
        spot = self.peak_counts * np.exp(
            -((rows - 60.0) ** 2 + (cols - 60.0) ** 2) / (2.0 * sigma**2)
        )
        return spot.astype(np.uint16)


def test_ladder_records_the_camera_full_scale_not_the_frame_dtype(tmp_path: Path) -> None:
    """MONO12 frames are uint16: full_scale must be 4095, never 65535 or 255."""
    camera = _FakeMono12Camera()

    ladder = probe.measure_exposure_ladder(camera, [1.0, 5.0], 2, tmp_path)

    assert [row["full_scale"] for row in ladder] == [4095.0, 4095.0]
    assert camera.full_scale_reads, "ladder must ask the camera, not np.iinfo(dtype)"
    assert np.asarray(camera.get_numpy_image()).dtype == np.uint16

    # Downstream saturation now uses the same scale: a 3000-count peak on a
    # 4095 scale is 73% -- not saturated. On the old 255 literal it looked blown out.
    summary = probe.analyse_ladder(ladder, 255.0)
    assert ladder[0]["saturated"] is False
    assert summary["saturation_counts"] == pytest.approx(0.95 * 4095.0)


def test_bit_depth_readback_mismatch_raises() -> None:
    """A camera that silently stayed on Mono8 must stop the probe, not skew it."""
    camera = _FakeMono12Camera(bit_depth=8)

    with pytest.raises(RuntimeError) as excinfo:
        probe._verify_camera_bit_depth(camera, 12)

    message = str(excinfo.value)
    assert "12 bit" in message and "8 bit" in message


def test_bit_depth_verification_accepts_matching_readback() -> None:
    camera = _FakeMono12Camera(bit_depth=12)

    assert probe._verify_camera_bit_depth(camera, 12) == 12


def test_bit_depth_verification_skips_backends_without_the_attribute() -> None:
    """ffmpeg / image_folder / sim have no bit-depth truth: dtype fallback is correct."""

    class _NoBitDepthCamera:
        pass

    assert probe._verify_camera_bit_depth(_NoBitDepthCamera(), 12) is None


def test_bit_depth_flag_defaults_to_the_s0_mono12_precondition() -> None:
    parser = probe.build_parser()

    assert parser.parse_args([]).bit_depth == 12
    assert parser.parse_args(["--bit-depth", "12"]).bit_depth == 12
    assert parser.parse_args(["--bit-depth", "16"]).bit_depth == 16


# ---------------------------------------------------------------------------
# Fresnel D 标定段 (--focus-calibration, 默认关闭) —— 全部离线假设备
# ---------------------------------------------------------------------------


class _FakeFocusSlm:
    """只满足记录段只读自检; 相位下发被假 sweep 截获, 不会碰到设备。"""

    def get_device_info(self) -> dict[str, object]:
        return {"driveboard_id": "fake-board"}

    def drive_board_status(self) -> None:
        raise RuntimeError("fake slm: no drive status")

    def get_wavelength_info(self) -> tuple[int, int]:
        return 1064, 993


class _FakeFocusCamera:
    bit_depth = 12
    pixel_size_um = 3.45
    model_name = "fake-cam"

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        return np.zeros((8, 8), dtype=np.uint16)


def _focus_config_from_args(args: argparse.Namespace | None = None) -> object:
    """把 CLI 命名空间里的标定项搬进 ProbeConfig (测的是契约, 不是整套 run_probe)。"""
    if args is None:
        args = probe.build_parser().parse_args([])
    return probe.ProbeConfig(
        exposures_ms=[0.05, 1.0, 10.0],
        focus_calibration=bool(args.focus_calibration),
        focus_coarse_points=int(args.focus_coarse_points),
        focus_diopter_range=float(args.focus_diopter_range),
        focus_refine_points=int(args.focus_refine_points),
        focus_frames=int(args.focus_frames),
        focus_floor_ms=args.focus_floor_ms,
        focus_ceiling_ms=args.focus_ceiling_ms,
        focus_exposure_mode=str(
            getattr(args, "focus_exposure_mode", "adapt")
        ),
        focus_fixed_exposure_ms=getattr(args, "focus_fixed_exposure_ms", None),
        focus_centre_px=args.focus_centre_px,
        focus_radius_px=args.focus_radius_px,
    )


def _focus_point(
    diopter: float = -0.42,
    peak_per_ms: float = 600.0,
    exposure_ms: float = 2.0,
) -> FocusPoint:
    return FocusPoint(
        diopter_m1=diopter,
        focal_mm=1000.0 / diopter,
        exposure_ms=exposure_ms,
        peak=1200.0,
        peak_fraction=1200.0 / 4095.0,
        peak_per_ms=peak_per_ms,
        fwhm_px=15.0,
        hollowness=0.1,
        status="ok",
    )


def _focus_result(
    *,
    diopter: float = -0.42,
    linearity: float | None = 1.02,
    range_m1: tuple[float, float] = (-1.5, 1.5),
    path_blocked: bool = False,
    exposure_mode: str = "adapt",
    repeat_ratio: float | None = None,
    exposure_ms: float = 2.0,
    profile: list[tuple[float, float]] | None = None,
) -> FocusProbeResult:
    """造一个 ``FocusProbeResult`` 夹具。

    默认剖面是**过 D2 的五点单峰**: 最优点在 ``diopter`` (``peak_per_ms``=600),
    左右各 2 点逐点递减 100 —— 判定要读最优点两侧的邻居, 单点结果会被
    :func:`_evaluate_focus_result` 判 ``ambiguous``。``profile`` 给
    ``(diopter_m1, peak_per_ms)`` 序列时逐点使用 (最优点按 ``peak_per_ms`` 取),
    供 D2 的反例 / 边界用例。
    """
    if profile is None:
        step = 0.0833
        profile = [
            (diopter - 2.0 * step, 400.0),
            (diopter - step, 500.0),
            (diopter, 600.0),
            (diopter + step, 500.0),
            (diopter + 2.0 * step, 400.0),
        ]
    points = [
        _focus_point(d, peak_per_ms=ppm, exposure_ms=exposure_ms)
        for d, ppm in profile
    ]
    best = max(points, key=lambda p: p.peak_per_ms)
    return FocusProbeResult(
        points=tuple(points),
        best=best,
        best_focal_mm=best.focal_mm,
        best_diopter_m1=best.diopter_m1,
        gain_vs_flat=3.0,
        linearity_ratio=linearity,
        range_m1=range_m1,
        exposure_bounds_ms=(0.02, 10.0),
        notes=(),
        path_blocked=path_blocked,
        blockage_reason="" if not path_blocked else "no local structure",
        exposure_mode=exposure_mode,
        fixed_exposure_ms=(exposure_ms if exposure_mode == "fixed" else None),
        repeat_ratio=repeat_ratio,
    )


def _forbid_call(*args: object, **kwargs: object) -> None:
    raise AssertionError("设备无关的标定调用不该在开关关闭/前置失败时发生")


def _record_with_focus(evidence: dict[str, object] | None = None) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if evidence is not None:
        kwargs["focus_calibration"] = evidence
    return probe._record_inputs(
        run_id="test_run",
        config_id="S0",
        probe_report=Path("optical_path_probe.json"),
        config=probe.ProbeConfig(),
        slm=_FakeFocusSlm(),
        camera=_FakeFocusCamera(),
        device_info={"cam_serial": "fake-cam-01"},
        exposure_used_ms=1.0,
        full_scale=4095.0,
        ladder_summary={"saturation_counts": 3890.25},
        response={"baseline_peak_counts": 120.0},
        verdict={"state": "ok"},
        acquired_at="2026-11-24T00:00:00+00:00",
        **kwargs,
    )


def test_focus_calibration_default_off_leaves_record_and_devices_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probe, "sweep_focus", _forbid_call)
    monkeypatch.setattr(probe, "slm_self_check", _forbid_call)

    config = _focus_config_from_args()
    assert config.focus_calibration is False
    assert probe._run_slm_self_check_for_focus(config, _FakeFocusSlm()) is None
    assert (
        probe._maybe_run_focus_calibration(
            slm=_FakeFocusSlm(),
            camera=_FakeFocusCamera(),
            config=config,
            start_exposure_ms=1.0,
            out_dir=Path("unused"),
            run_id="off",
        )
        is None
    )

    record = _record_with_focus()
    assert record["optical_path"]["fresnel_compensation_m_inv"] is None
    assert "fresnel_calibration" not in record["optical_path"]
    assert record["reasons"]["optical_path.fresnel_compensation_m_inv"] == (
        "探针不做 Fresnel 补偿标定 (compensation 由焦面标定给出)"
    )


def test_focus_calibration_ok_writes_measured_d_and_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_sweep(**kwargs: object) -> FocusProbeResult:
        captured.update(kwargs)
        return _focus_result()

    monkeypatch.setattr(probe, "sweep_focus", fake_sweep)
    self_check = {
        "grayscale": 0.0,
        "memory_slot": None,
        "displayed_phase_source": "none",
    }
    monkeypatch.setattr(probe, "slm_self_check", lambda slm: dict(self_check))

    config = _focus_config_from_args(
        probe.build_parser().parse_args(["--focus-calibration"])
    )
    config.pupil_center = (960, 600)
    slm = _FakeFocusSlm()
    assert probe._run_slm_self_check_for_focus(config, slm) == self_check
    evidence = probe._maybe_run_focus_calibration(
        slm=slm,
        camera=_FakeFocusCamera(),
        config=config,
        start_exposure_ms=1.0,
        out_dir=tmp_path,
        run_id="ok_run",
        self_check=self_check,
    )
    assert evidence is not None
    assert evidence["status"] == "ok"
    assert evidence["accepted"] is True
    assert evidence["best_diopter_m1"] == -0.42
    assert evidence["peak_per_ms"] == 600.0
    assert evidence["exposure_ms"] == 2.0
    assert evidence["slm_self_check"] == self_check
    assert evidence["centre_px"] == [960.0, 600.0]
    assert evidence["centre_source"] == "config_pupil_center"
    assert evidence["requested"]["max_extends"] == 1
    assert captured["max_extends"] == 1
    assert captured["centre_px"] == (960.0, 600.0)
    assert captured["radius_px"] == 450.0
    # D2 判定要读最优点两侧的邻居, 证据必须逐点留档 (默认五点剖面全部落盘)。
    assert len(evidence["points"]) == 5
    assert evidence["points"][2]["peak_per_ms"] == 600.0

    saved = json.loads(
        (tmp_path / "focus_calibration.json").read_text(encoding="utf-8")
    )
    assert saved == evidence
    assert (tmp_path / "focus_calibration.json").read_bytes()[:3] != b"\xef\xbb\xbf"

    record = _record_with_focus(evidence)
    assert record["optical_path"]["fresnel_compensation_m_inv"] == -0.42
    assert "optical_path.fresnel_compensation_m_inv" not in record["reasons"]
    assert record["optical_path"]["fresnel_calibration"] == evidence

    # The canonical record builder filters its fixed-schema sections, so verify
    # the probe writer restores its documented additive evidence field on disk.
    probe._PENDING_RECORD = {"out_dir": tmp_path, "inputs": record}
    written = probe._write_pending_record()
    assert written == tmp_path / "optical_path_record.json"
    saved_record = json.loads(written.read_text(encoding="utf-8"))
    assert saved_record["optical_path"]["fresnel_calibration"] == evidence
    assert saved_record["optical_path"]["fresnel_compensation_m_inv"] == -0.42


def test_focus_calibration_non_ok_keeps_d_null_and_writes_failure_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        probe, "sweep_focus", lambda **kwargs: _focus_result(linearity=0.5)
    )
    monkeypatch.setattr(probe, "slm_self_check", lambda slm: {"grayscale": 0.0})
    config = _focus_config_from_args(
        probe.build_parser().parse_args(["--focus-calibration"])
    )
    evidence = probe._maybe_run_focus_calibration(
        slm=_FakeFocusSlm(),
        camera=_FakeFocusCamera(),
        config=config,
        start_exposure_ms=1.0,
        out_dir=tmp_path,
        run_id="bad_linearity",
        self_check={"grayscale": 0.0},
    )
    assert evidence is not None
    assert evidence["status"] == "linearity_failed"
    assert evidence["accepted"] is False
    json.loads((tmp_path / "focus_calibration.json").read_text(encoding="utf-8"))

    record = _record_with_focus(evidence)
    assert record["optical_path"]["fresnel_compensation_m_inv"] is None
    reason = record["reasons"]["optical_path.fresnel_compensation_m_inv"]
    assert "status=linearity_failed" in reason
    assert "曝光线性度" in reason
    assert "D 保持 null" in reason
    # 标定真跑过并失败时, 不能停在那句「探针不做 Fresnel 补偿标定」的旧话上。
    assert "探针不做" not in reason


def test_focus_calibration_out_of_range_after_one_extension_never_writes_d(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        probe,
        "sweep_focus",
        lambda **kwargs: _focus_result(diopter=-3.0, range_m1=(-3.0, 1.5)),
    )
    config = _focus_config_from_args(
        probe.build_parser().parse_args(["--focus-calibration"])
    )
    evidence = probe._maybe_run_focus_calibration(
        slm=_FakeFocusSlm(),
        camera=_FakeFocusCamera(),
        config=config,
        start_exposure_ms=1.0,
        out_dir=tmp_path,
        run_id="edge",
        self_check={"grayscale": 0.0},
    )
    assert evidence is not None
    assert evidence["status"] == "out_of_range"
    record = _record_with_focus(evidence)
    assert record["optical_path"]["fresnel_compensation_m_inv"] is None
    reason = record["reasons"]["optical_path.fresnel_compensation_m_inv"]
    assert "扫描边界" in reason
    assert "status=out_of_range" in reason
    assert "D 保持 null" in reason
    assert "探针不做" not in reason


def test_slm_self_check_failure_aborts_only_focus_segment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(probe, "sweep_focus", _forbid_call)

    def failing_check(slm: object) -> dict[str, object]:
        raise RuntimeError("SLM grayscale not 0 after open (read 512)")

    monkeypatch.setattr(probe, "slm_self_check", failing_check)
    config = _focus_config_from_args(
        probe.build_parser().parse_args(["--focus-calibration"])
    )
    failure = probe._run_slm_self_check_for_focus(config, _FakeFocusSlm())
    assert isinstance(failure, str)
    assert "grayscale not 0" in failure

    evidence = probe._maybe_run_focus_calibration(
        slm=_FakeFocusSlm(),
        camera=_FakeFocusCamera(),
        config=config,
        start_exposure_ms=1.0,
        out_dir=tmp_path,
        run_id="self_check_failed",
        self_check=failure,
    )
    assert evidence is not None
    assert evidence["status"] == "slm_self_check_failed"
    assert evidence["slm_self_check"] == failure
    saved = json.loads(
        (tmp_path / "focus_calibration.json").read_text(encoding="utf-8")
    )
    assert saved["slm_self_check"] == failure

    record = _record_with_focus(evidence)
    assert record["optical_path"]["fresnel_compensation_m_inv"] is None
    reason = record["reasons"]["optical_path.fresnel_compensation_m_inv"]
    assert "SLM 相位自检失败" in reason and "需人工检查面板状态后重排" in reason
    assert "grayscale not 0" in reason
    assert "status=slm_self_check_failed" in reason
    assert "D 保持 null" in reason
    # 自检失败也是「标定跑了但没成」, 不能留着「探针不做标定」的旧 reason。
    assert "探针不做" not in reason


def test_focus_centre_and_radius_source_and_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        probe,
        "sweep_focus",
        lambda **kwargs: captured.update(kwargs) or _focus_result(),
    )
    monkeypatch.setattr(probe, "slm_self_check", lambda slm: {"grayscale": 0.0})

    explicit = probe.build_parser().parse_args(
        [
            "--focus-calibration",
            "--focus-centre-px",
            "1234,567",
            "--focus-radius-px",
            "300",
        ]
    )
    config = _focus_config_from_args(explicit)
    assert config.focus_centre_px == (1234.0, 567.0)
    evidence = probe._maybe_run_focus_calibration(
        slm=_FakeFocusSlm(),
        camera=_FakeFocusCamera(),
        config=config,
        start_exposure_ms=1.0,
        out_dir=tmp_path,
        run_id="explicit",
        self_check={"grayscale": 0.0},
    )
    assert evidence is not None
    assert evidence["centre_px"] == [1234.0, 567.0]
    assert evidence["centre_source"] == "cli"
    assert captured["centre_px"] == (1234.0, 567.0)
    assert captured["radius_px"] == 300.0

    # 无 CLI 值但有本次 S0 的光瞳中心 (panel px) ⇒ 直接复用, 与离焦相位同口径。
    from_config = probe.build_parser().parse_args(["--focus-calibration"])
    config = _focus_config_from_args(from_config)
    config.pupil_center = (960, 600)
    evidence = probe._maybe_run_focus_calibration(
        slm=_FakeFocusSlm(),
        camera=_FakeFocusCamera(),
        config=config,
        start_exposure_ms=1.0,
        out_dir=tmp_path,
        run_id="from_config",
        self_check={"grayscale": 0.0},
    )
    assert evidence is not None
    assert evidence["centre_px"] == [960.0, 600.0]
    assert evidence["centre_source"] == "config_pupil_center"

    # 两者都没有时回落到 fresnel_panel 的缺省光瞳中心 (panel px), 不报错。
    config.pupil_center = None
    evidence = probe._maybe_run_focus_calibration(
        slm=_FakeFocusSlm(),
        camera=_FakeFocusCamera(),
        config=config,
        start_exposure_ms=1.0,
        out_dir=tmp_path,
        run_id="default",
        self_check={"grayscale": 0.0},
    )
    assert evidence is not None
    assert evidence["centre_px"] == [960.0, 600.0]
    assert evidence["centre_source"] == "default_beam_center"
    assert captured["radius_px"] == 450.0


def test_focus_cli_defaults_and_explicit_values() -> None:
    parser = probe.build_parser()
    args = parser.parse_args([])
    assert args.focus_calibration is False
    assert args.focus_coarse_points == 13
    assert args.focus_diopter_range == 1.5
    assert args.focus_refine_points == 7
    assert args.focus_frames == 3
    assert args.focus_floor_ms is None
    assert args.focus_ceiling_ms is None
    assert args.focus_centre_px is None
    assert args.focus_radius_px is None

    args = parser.parse_args(
        [
            "--focus-calibration",
            "--focus-coarse-points",
            "9",
            "--focus-diopter-range",
            "2.5",
            "--focus-refine-points",
            "5",
            "--focus-frames",
            "4",
            "--focus-floor-ms",
            "0.05",
            "--focus-ceiling-ms",
            "20",
            "--focus-centre-px",
            "1300,980",
            "--focus-radius-px",
            "300",
        ]
    )
    assert args.focus_calibration is True
    assert args.focus_coarse_points == 9
    assert args.focus_diopter_range == 2.5
    assert args.focus_refine_points == 5
    assert args.focus_frames == 4
    assert args.focus_floor_ms == 0.05
    assert args.focus_ceiling_ms == 20.0
    assert args.focus_centre_px == (1300.0, 980.0)
    assert args.focus_radius_px == 300.0
    assert probe.S0_BIT_DEPTH == 12


def test_bit_depth_flag_rejects_unsupported_settings() -> None:
    parser = probe.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--bit-depth", "10"])


# ---------------------------------------------------------------------------
# Absolute-count thresholds must ride the effective full scale
# ---------------------------------------------------------------------------
#: ``RECOMMENDED_PEAK_RANGE`` / ``RECOMMENDED_PEAK_TARGET`` were calibrated when
#: counts meant 0-255 gray. Left unscaled they target 2.4-4.9% of a Mono12 frame,
#: so the probe would recommend an exposure ~16x darker than intended. Likewise the
#: plot drew its saturation line at a literal ``0.95 * 255``, 242 counts, on a
#: 4095-count axis.


def _raw_ladder(
    rows: list[tuple[float, float]], full_scale: float
) -> list[dict]:
    """A synthetic ladder whose records carry an explicit (bit-depth) full scale."""
    return [
        {
            "requested_ms": ms,
            "effective_ms": ms,
            "peak_raw": peak,
            "noise_median_counts": 0.0,
            "noise_sigma_counts": 1.0,
            "noise_threshold_counts": 4.0,
            "full_scale": full_scale,
        }
        for ms, peak in rows
    ]


#: The S0 rerun's shape: Mono12 frames (uint16, 4095 max), 0.05 ms baseline ~187
#: raw counts, and the useful step peaking at ~3000 raw counts (73% of full scale).
MONO12_LADDER = _raw_ladder([(0.05, 187.0), (1.0, 1200.0), (5.0, 3000.0)], 4095.0)


def test_mono12_recommended_band_is_scaled_to_raw_counts() -> None:
    """100-200 gray on a 4095 scale is 1605.9-3211.8 raw counts, and 3000 is inside."""
    summary = probe.analyse_ladder(MONO12_LADDER, 255.0)  # the literal is overridden

    recommended = summary["recommended_exposure"]
    assert recommended is not None
    assert recommended["exposure_ms"] == pytest.approx(5.0)
    assert recommended["peak_counts"] == pytest.approx(3000.0)  # raw, for the raw axis
    assert recommended["target_band_counts"] == pytest.approx(
        [1605.8823529411764, 3211.764705882353]
    )
    assert summary["target_band_counts"] == pytest.approx(
        [1605.8823529411764, 3211.764705882353]
    )
    assert recommended["target_band_gray"] == pytest.approx([100.0, 200.0])
    assert recommended["in_target_band"] is True
    # 3000 * 255 / 4095 -- the same reading in the gray view recorded in old runs.
    assert recommended["peak_gray"] == pytest.approx(186.81318681318683)


def test_same_counts_at_8bit_keep_the_gray_band() -> None:
    """Scaling is driven by full_scale: at 255 the band is still [100, 200]."""
    ladder = _raw_ladder([(1.0, 1200.0), (5.0, 3000.0)], 255.0)

    summary = probe.analyse_ladder(ladder, 255.0)

    assert summary["target_band_counts"] == pytest.approx([100.0, 200.0])
    # A 3000-count peak is far past a 255 scale: every step is saturated, so there
    # is no recommendation to pick a band for.
    assert all(row["saturated"] for row in ladder)
    assert summary["recommended_exposure"] is None


def test_plot_draws_the_saturation_line_at_the_measured_full_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Mono12 plot's saturation line is 0.95*4095 = 3890.25, not 242."""
    summary = probe.analyse_ladder(MONO12_LADDER, 255.0)
    response = probe.analyse_defocus_response(
        [
            _state(0.0, 250.0, 12.0, 10),
            _state(1.0, 200.0, 12.6, 20),
            _state(-1.0, 200.0, 12.7, 30),
        ],
        2.4,
    )

    closed: list = []
    monkeypatch.setattr(probe.plt, "close", closed.append)  # keep the figure alive
    out_path = tmp_path / "optical_path_probe.png"
    probe.plot_results(MONO12_LADDER, summary, response, out_path)

    assert out_path.exists() and closed, "plot_results must save its figure"
    ax = closed[0].axes[0]
    saturation = float(summary["saturation_counts"])
    assert saturation == pytest.approx(3890.25)  # 0.95 * 4095

    at_saturation = [
        line
        for line in ax.get_lines()
        if np.allclose(np.asarray(line.get_ydata(), dtype=float), saturation)
    ]
    assert at_saturation, "no horizontal line at the measured saturation level"
    assert np.asarray(at_saturation[0].get_ydata(), dtype=float).tolist() == pytest.approx(
        [saturation, saturation]
    )
    assert "3890" in at_saturation[0].get_label()
    # The 8-bit literal must be gone.
    assert not any(
        np.allclose(np.asarray(line.get_ydata(), dtype=float), 0.95 * 255.0)
        for line in ax.get_lines()
    )

    monkeypatch.undo()
    probe.plt.close(closed[0])


# ---------------------------------------------------------------------------
# 单曝光模式的判定契约 (fixed 用同曝光复测替代 D3 的跨曝光线性度)
# ---------------------------------------------------------------------------


def test_fixed_mode_with_a_repeat_ratio_in_band_passes() -> None:
    result = _focus_result(exposure_mode="fixed", repeat_ratio=1.00, linearity=None)
    assert probe._evaluate_focus_result(result) == ("ok", None)


def test_fixed_mode_reports_a_non_repeatable_peak() -> None:
    result = _focus_result(exposure_mode="fixed", repeat_ratio=1.40, linearity=None)
    status, reason = probe._evaluate_focus_result(result)
    assert status == "repeatability_failed"
    assert reason is not None and "同曝光复测比" in reason


def test_fixed_mode_without_a_repeat_point_is_inconclusive() -> None:
    result = _focus_result(exposure_mode="fixed", repeat_ratio=None, linearity=None)
    status, reason = probe._evaluate_focus_result(result)
    assert status == "repeatability_unavailable"
    assert reason is not None and "同曝光复测点" in reason


def test_adapt_mode_still_gates_on_exposure_linearity() -> None:
    """adapt 的判定一个字都不能变: 线性度越界仍然判 linearity_failed。"""
    result = _focus_result(linearity=1.45)
    assert probe._evaluate_focus_result(result)[0] == "linearity_failed"


# ---------------------------------------------------------------------------
# D2 单峰: 最优点两侧 peak/ms 必须各单调下降 2 点 (预注册 §10.4 第 405 行)
# ---------------------------------------------------------------------------


def test_d2_rejects_a_single_fixed_peak() -> None:
    """真实 R4 (20261124_102701Z) 的原样反例: 左侧第二点反而抬高。

    那次 ``focus_calibration.json`` 报了 ``status=ok, accepted=true``, D 写成了
    -0.5833, 而两侧 peak/ms 并不各单调下降 —— 峰形有两个候选, 不许写 D。
    """
    result = _focus_result(
        diopter=-0.5833,
        profile=[
            (-0.75, 23476.7),
            (-0.6667, 23226.7),
            (-0.5833, 24326.7),
            (-0.5, 24093.3),
            (-0.4167, 24210.0),
        ],
    )
    status, reason = probe._evaluate_focus_result(result)
    assert status == "ambiguous"
    assert reason is not None
    assert "未各单调下降" in reason and "峰形不唯一" in reason
    # 理由里要能看出是哪三点点坏了单调性。
    assert "左: -0.750→23477, -0.667→23227, 最优→24327" in reason


def test_d2_passes_a_clean_single_peak_and_falls_through_to_the_mode_gate() -> None:
    """单峰成立时 D2 放行, 判定继续走 D3 的模式分支 (两种模式都一样)。"""
    single_peak = [
        (-0.75, 23800.0),
        (-0.5833, 24000.0),
        (-0.42, 24326.7),
        (-0.25, 24100.0),
        (-0.0833, 23900.0),
    ]
    adapt = _focus_result(diopter=-0.42, profile=single_peak)
    assert probe._evaluate_focus_result(adapt) == ("ok", None)

    fixed = _focus_result(
        diopter=-0.42,
        profile=single_peak,
        exposure_mode="fixed",
        repeat_ratio=1.00,
        linearity=None,
    )
    assert probe._evaluate_focus_result(fixed) == ("ok", None)


def test_d2_rejects_a_profile_with_too_few_neighbours() -> None:
    """最优点的某一侧不足 2 个可用点 ⇒ 单峰无从确认, 仍是 ``ambiguous``。"""
    result = _focus_result(
        diopter=-0.5,
        profile=[
            (-0.5833, 24000.0),
            (-0.5, 24326.7),
            (-0.4167, 24210.0),
        ],
    )
    status, reason = probe._evaluate_focus_result(result)
    assert status == "ambiguous"
    assert reason is not None and "采样点不足 2 个" in reason


def test_a_preregistered_fixed_exposure_overrides_the_ladder_recommendation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_sweep(**kwargs: object) -> FocusProbeResult:
        captured.update(kwargs)
        return _focus_result(
            exposure_mode="fixed",
            repeat_ratio=1.0,
            linearity=None,
            exposure_ms=float(kwargs["start_exposure_ms"]),
        )

    monkeypatch.setattr(probe, "sweep_focus", fake_sweep)
    config = _focus_config_from_args(
        probe.build_parser().parse_args(
            [
                "--focus-calibration",
                "--focus-exposure-mode",
                "fixed",
                "--focus-fixed-exposure-ms",
                "0.1",
            ]
        )
    )
    evidence = probe._maybe_run_focus_calibration(
        slm=_FakeFocusSlm(),
        camera=_FakeFocusCamera(),
        config=config,
        start_exposure_ms=1.0,
        out_dir=tmp_path,
        run_id="fixed_override",
        self_check={
            "grayscale": 0.0,
            "memory_slot": None,
            "displayed_phase_source": "none",
        },
    )
    assert evidence is not None
    assert evidence["status"] == "ok"
    assert evidence["requested"]["fixed_exposure_ms_requested"] == 0.1
    assert evidence["requested"]["exposure_mode"] == "fixed"
    # start_exposure_ms 记录的是实际用的曝光 (预注册值覆盖了阶梯推荐值 1.0)。
    assert evidence["requested"]["start_exposure_ms"] == 0.1
    assert captured["start_exposure_ms"] == 0.1
    assert captured["exposure_mode"] == "fixed"
    assert {p["exposure_ms"] for p in evidence["points"]} == {0.1}
