"""Scoring / geometry contracts for the model-vs-hardware fidelity analysis.

Everything here is offline: no run directory, no device, no writes to the repo.
Synthetic Gaussian frames are pushed through the very same primitives the script
uses, so the invariants below are the ones the report leans on:

* the linear correlation the script reports is the plain ``np.corrcoef`` of the
  flattened intensities (it must reproduce ``comparison.best.intensity_correlation``);
* NRMSE is peak-normalised, i.e. invariant to the model/counts unit conversion;
* the scale scan's ``s = 1`` is the *unscaled* baseline and its peak ``s*`` recovers
  a planted geometric scale mismatch (the K-calibration signature);
* the verdict bands and the "nearly identical" extra gates behave as documented.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.slm.analyze_model_hw_fidelity import (
    CORR_HIGH,
    CORR_MID,
    FWHM_RATIO_TOL,
    PEAK_OFFSET_TOL_SAMPLES,
    SCALE_MAX_REQUESTED,
    apply_convention,
    assert_no_drivers,
    dominant_discrepancy,
    encircled_radii,
    focus_compensation_flags,
    intensity_centroid_rms,
    mirror_compare_farfields,
    nearly_identical,
    nrmse,
    pair_history,
    pearson,
    read_history,
    resolve_config,
    scale_scan,
    scale_scan_reading,
    verdict_band,
)

# Odd size so the optical axis lands on an exact pixel.
SIZE = 129
CENTRE = (SIZE - 1) / 2.0
SIGMA_MODEL = 9.0
SCALE_MISMATCH = 3.0


def _gaussian(sigma: float, peak: float = 1.0, centre: float = CENTRE) -> np.ndarray:
    """A centred 2-D Gaussian on the ``SIZE x SIZE`` grid."""
    axis = np.arange(SIZE, dtype=np.float64) - centre
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return peak * np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))


def _model_frame() -> np.ndarray:
    return _gaussian(SIGMA_MODEL, peak=1.0e-3)


def _measured_frame(scale: float = SCALE_MISMATCH, peak: float = 1.3e2) -> np.ndarray:
    """The same spot observed ``scale`` times narrower, in CCD counts."""
    return _gaussian(SIGMA_MODEL / scale, peak=peak)


# ---------------------------------------------------------------------------
# correlation / error primitives
# ---------------------------------------------------------------------------
def test_pearson_is_self_consistent_and_degenerate_safe() -> None:
    model = _model_frame()
    assert pearson(model, model) == pytest.approx(1.0, abs=1e-12)
    assert pearson(model, 7.0 * model + 1.0) == pytest.approx(1.0, abs=1e-12)
    assert math.isnan(pearson(np.ones((4, 4)), np.ones((4, 4))))
    assert math.isnan(pearson(np.ones((4, 4)), np.ones((3, 3))))


def test_nrmse_is_invariant_to_the_unit_conversion() -> None:
    model = _model_frame()
    # Same shape, wildly different amplitude (normalised intensity vs counts):
    # after peak-normalisation the error must vanish.
    assert nrmse(model, model * 1.3e5) == pytest.approx(0.0, abs=1e-12)
    # A genuinely different shape must give a positive, finite error.
    value = nrmse(model, _measured_frame())
    assert math.isfinite(value) and value > 0.0
    # Explicit global scales must equal normalising by hand.
    manual = nrmse(model / model.max(), _measured_frame() / _measured_frame().max())
    assert nrmse(
        model,
        _measured_frame(),
        scale_ref=model.max(),
        scale_test=_measured_frame().max(),
    ) == pytest.approx(manual, abs=1e-12)


def test_apply_convention_matches_its_documented_order() -> None:
    # Order is transpose -> flip rows -> flip cols (same as the reference impl).
    frame = np.arange(SIZE * SIZE, dtype=np.float64).reshape(SIZE, SIZE)
    n = SIZE
    assert np.array_equal(apply_convention(frame, True, False, False), frame.T)
    assert np.array_equal(
        apply_convention(frame, False, True, False), frame[::-1, :]
    )
    assert np.array_equal(
        apply_convention(frame, False, False, True), frame[:, ::-1]
    )
    combined = apply_convention(frame, True, True, False)
    assert combined[3, 7] == frame[7, n - 1 - 3]
    # Every convention is a pixel permutation: values are preserved as a multiset.
    for transpose in (False, True):
        for flip_rows in (False, True):
            for flip_cols in (False, True):
                out = apply_convention(frame, transpose, flip_rows, flip_cols)
                assert np.array_equal(np.sort(out.ravel()), np.sort(frame.ravel()))


def test_centroid_of_a_centred_gaussian_is_the_optical_axis() -> None:
    cx, cy, rms = intensity_centroid_rms(_model_frame())
    assert cx == pytest.approx(CENTRE, abs=1e-6)
    assert cy == pytest.approx(CENTRE, abs=1e-6)
    assert rms > 0.0


def test_encircled_radii_are_monotone() -> None:
    radii = encircled_radii(_model_frame())
    assert radii["r50"] < radii["r80"] < radii["r90"]
    assert radii["r50"] > 0.0


# ---------------------------------------------------------------------------
# the compare kernel and the planted-scale scale scan
# ---------------------------------------------------------------------------
def test_mirror_compare_reproduces_pearson_and_the_planted_fwhm_ratio() -> None:
    model = _model_frame()
    measured = _measured_frame()
    result = mirror_compare_farfields(model, measured)
    assert result["intensity_correlation"] == pytest.approx(
        pearson(model, measured), abs=1e-12
    )
    ratio = result["model_fwhm_px"] / result["measured_fwhm_px"]
    assert ratio == pytest.approx(SCALE_MISMATCH, rel=0.15)
    assert result["peak_offset_samples"] == [0.0, 0.0]


def test_scale_scan_recovers_a_planted_geometric_scale() -> None:
    model = _model_frame()
    measured = _measured_frame()
    scales = np.arange(0.40, 4.001, 0.05)
    scan = scale_scan(model, measured, scales, fwhm_ratio=SCALE_MISMATCH)

    # s = 1 must be exactly the unscaled baseline correlation.
    assert scan["corr_at_1"] == pytest.approx(pearson(model, measured), abs=1e-12)
    assert scan["corr_full_grid_unaligned"] == pytest.approx(
        scan["corr_at_1"], abs=1e-12
    )
    # The peak must land on the planted scale (within two scan steps: bilinear
    # zoom costs a little resolution) and beat the baseline.
    assert scan["s_star"] == pytest.approx(SCALE_MISMATCH, abs=2 * 0.05 + 1e-9)
    assert scan["corr_at_s_star"] > scan["corr_at_1"] + 0.2
    # ... while the task-requested window [0.40, 2.50] can only reach its edge.
    assert scan["s_star_within_requested_range"] == pytest.approx(
        SCALE_MAX_REQUESTED, abs=1e-9
    )
    assert not scan["boundary_hit"]
    reading = scale_scan_reading(scan)
    assert "几何尺度" in reading


def test_scale_scan_is_flat_when_there_is_no_scale_mismatch() -> None:
    model = _model_frame()
    measured = model * 1.3e5  # same shape, counts amplitude
    scales = np.arange(0.40, 4.001, 0.05)
    scan = scale_scan(model, measured, scales, fwhm_ratio=1.0)
    assert scan["s_star"] == pytest.approx(1.0, abs=0.05)
    assert scan["corr_at_s_star"] == pytest.approx(1.0, abs=1e-6)
    assert scan["corr_at_1"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------
def test_verdict_band_thresholds() -> None:
    assert verdict_band(CORR_HIGH) == "高度一致 (强度相关系数 ≥ 0.8)"
    assert verdict_band(CORR_MID) == "中度一致 (0.4 ≤ 强度相关系数 < 0.8)"
    assert verdict_band(0.1) == "不一致 (强度相关系数 < 0.4)"
    assert "无法判定" in verdict_band(None)
    assert "无法判定" in verdict_band(float("nan"))


def test_nearly_identical_requires_all_three_gates() -> None:
    ok, reasons = nearly_identical(0.9, 1.0, 1.0)
    assert ok and reasons == []
    # A scale mismatch alone is enough to fail, even with a good correlation.
    ok, reasons = nearly_identical(0.9, SCALE_MISMATCH, 0.0)
    assert not ok and len(reasons) == 1 and "FWHM" in reasons[0]
    # So is a peak offset beyond the tolerance.
    ok, reasons = nearly_identical(0.9, 1.0, PEAK_OFFSET_TOL_SAMPLES + 0.1)
    assert not ok and len(reasons) == 1 and "峰值偏移" in reasons[0]
    # And so is a mid-band correlation.
    ok, reasons = nearly_identical(0.6, 1.0, 0.0)
    assert not ok and len(reasons) == 1 and "相关系数" in reasons[0]


def test_dominant_discrepancy_names_the_scale_mismatch_first() -> None:
    text = dominant_discrepancy(0.6, SCALE_MISMATCH, 8.5)
    assert "几何尺度" in text
    # With a matched scale but a large offset, the offset wins.
    text = dominant_discrepancy(0.95, 1.0, 20.0)
    assert "峰值偏移" in text


def test_focus_compensation_flags() -> None:
    assert focus_compensation_flags(float("inf"))["focus_compensation_effective_zero"]
    assert focus_compensation_flags(-4.33e4)["focus_compensation_effective_zero"]
    assert not focus_compensation_flags(1601.0)["focus_compensation_effective_zero"]


def test_assert_no_drivers_passes_in_a_clean_process() -> None:
    """真起子进程验证"干净进程"契约 (issue #205 第 132 批)。

    进程内调用会被 CI 分片污染: 同 shard 先跑的测试 import 过任何
    ``ao_shaping.drivers.*`` 后, ``sys.modules`` 检查必然误报 (CI 上 4 个
    scripts 测试同 shard 时红)。本测试的名字就是 clean process —— 语义是
    "导入分析脚本本身不拖进驱动", 只有独立解释器才能证明它。
    """
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    code = (
        "from scripts.slm.analyze_model_hw_fidelity import assert_no_drivers; "
        "assert_no_drivers()"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# history pairing and config resolution (tmp files only)
# ---------------------------------------------------------------------------
def test_pair_history_matches_by_iteration_label(tmp_path: Path) -> None:
    model_csv = tmp_path / "history.csv"
    hw_csv = tmp_path / "history_hw.csv"
    model_csv.write_text(
        "iteration,loss,phase_rms_rad\n0,1.0e-3,0.5\n1,7.0e-4,0.4\n",
        encoding="utf-8",
    )
    hw_csv.write_text(
        "iteration,loss,phase_rms_rad,measured_peak,exposure_ms\n"
        "0,2.0e-3,0.6,139.5,7.4\n1,1.5e-3,0.45,137.1,7.4\n",
        encoding="utf-8",
    )
    pairing = pair_history(read_history(model_csv), read_history(hw_csv))
    rows = pairing["rows"]
    assert [row["iteration"] for row in rows] == [0, 1]
    assert rows[0]["model_loss_same_iter"] == pytest.approx(1.0e-3)
    assert rows[0]["hw_loss_measured"] == pytest.approx(2.0e-3)
    assert rows[0]["loss_ratio_hw_over_model"] == pytest.approx(2.0)
    assert rows[1]["hw_measured_peak"] == pytest.approx(137.1)
    # The trend sentence must describe the *paired* span, not history.csv's ends.
    assert "配对迭代 0 → 1" in pairing["statement"]
    assert "模型损失下降**没有**伴随实测峰值改善" in pairing["statement"]


def test_read_history_survives_a_utf8_bom(tmp_path: Path) -> None:
    # The run directories write history CSVs with a BOM; the first header field
    # must still parse, or block C's pairing silently returns all-None rows.
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "iteration,loss,phase_rms_rad\n1,7.0e-4,0.4\n", encoding="utf-8-sig"
    )
    rows = read_history(csv_path)
    assert list(rows[0].keys()) == ["iteration", "loss", "phase_rms_rad"]
    assert rows[0]["iteration"] == pytest.approx(1.0)
    assert rows[0]["loss"] == pytest.approx(7.0e-4)


def test_resolve_config_prefers_the_run_dir_then_reports_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_x"
    run_dir.mkdir()
    config, source = resolve_config(run_dir, {})
    assert config is None and "未找到" in source

    (run_dir / "run_config.json").write_text('{"iters": 20}', encoding="utf-8")
    config, source = resolve_config(run_dir, {})
    assert config == {"iters": 20} and "运行目录" in source
