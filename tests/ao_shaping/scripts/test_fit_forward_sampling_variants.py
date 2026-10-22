"""Offline tests for the physical forward-model sampling variant sweep.

Nothing here opens a device: the datasets are synthetic directories with a
manifest and a run configuration, and the observation loader of the other work
stream is replaced by a deterministic stand-in.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.slm import fit_forward_sampling_variants as variants

ROOT = Path(__file__).resolve().parents[3]
RAMP_MATRIX = np.array([[120.0, 7400.0], [-7500.0, 80.0]])
CENTER_U = np.array([1360.0, 1020.0])
CENTER_C = np.array([1424.0, 972.0])


def _spot(point):
    return {"peak": 100.0, "fwhm_px": 14.0,
            "centroid_x": float(point[0]), "centroid_y": float(point[1])}


def _manifest(probe_labels):
    """A manifest with the ramp geometry the loader of the other work stream needs."""
    rows = []
    for index, (branch, center) in enumerate((("U", CENTER_U), ("C", CENTER_C))):
        rows.append(dict(index=index, file=f"sample_{index:04d}.npz", label="reference_start",
                         branch=branch, family="reference", split="calibration",
                         exposure_ms=10.0, settled={"settled": True},
                         spot=_spot(center)))
    index = 2
    for axis in ("x", "y"):
        for period in (480, 960):
            for sign in (-1, 1):
                column = 0 if axis == "x" else 1
                point = CENTER_U + sign * RAMP_MATRIX[:, column] / period
                rows.append(dict(index=index, file=f"sample_{index:04d}.npz",
                                 label=f"ramp_{axis}_{period}_{sign}", branch="C", family="ramp",
                                 split="calibration", exposure_ms=10.0, settled={"settled": True},
                                 spot=_spot(point)))
                index += 1
    for label, split in probe_labels:
        for branch, center in (("U", CENTER_U), ("C", CENTER_C)):
            rows.append(dict(index=index, file=f"sample_{index:04d}.npz", label=label,
                             branch=branch, family="zernike", split=split, exposure_ms=10.0,
                             settled={"settled": True}, spot=_spot(center)))
            index += 1
    return rows


def _write_dataset(directory: Path, probe_labels, seed: int = 3):
    directory.mkdir(parents=True, exist_ok=True)
    rows = _manifest(probe_labels)
    (directory / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (directory / "run_config.json").write_text(json.dumps({
        "best_diopter_m1": 0.5,
        "settings": {"center_x": 960, "center_y": 600},
    }), encoding="utf-8")
    return rows


def _fake_observations(rows, grid_size, rng):
    """Deterministic stand-in for ``load_observations``."""
    selected = [row for row in rows if row["family"] != "ramp"]
    phases = rng.normal(0.0, 0.3, (len(selected), grid_size, grid_size)).astype(np.float32)
    yy, xx = np.mgrid[-120:121:2, -120:121:2]
    camera_xy = np.stack((xx + round(CENTER_U[0]), yy + round(CENTER_U[1])), axis=-1)
    images = np.zeros((len(selected), camera_xy.shape[0], camera_xy.shape[1]), dtype=np.float32)
    for index in range(len(selected)):
        column, line = 40 + 8 * index, 50
        images[index, line - 6:line + 6, column - 6:column + 6] = 1.0
    references = {row["branch"]: row for row in rows if row["label"] == "reference_start"}
    return selected, phases, images, camera_xy, references


@pytest.fixture()
def synthetic_dataset(tmp_path, monkeypatch):
    """Two synthetic datasets whose observation loader is replaced."""
    probes = [("probe_000", "train"), ("probe_001", "test")]
    first = tmp_path / "ds1"
    second = tmp_path / "ds2"
    _write_dataset(first, probes)
    _write_dataset(second, probes, seed=7)

    def fake_load_observations(directory, camera_center=None, *, grid_size=128):
        rows = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
        rng = np.random.default_rng(11)
        selected, phases, images, camera_xy, references = _fake_observations(rows, grid_size, rng)
        return selected, phases, images, camera_xy, references

    fit_module = variants.load_workstream_module(
        variants.FIT_MODULE, "scripts/slm/fit_defocus_forward_dataset.py")
    monkeypatch.setattr(fit_module, "load_observations", fake_load_observations)
    return first, second


def test_stage_a_grid_covers_fft_size_and_pixel_samples():
    grid = variants.stage_a_variants(128)
    assert len(grid) == 9
    assert sorted({v.fft_size for v in grid}) == [256, 512, 1024]
    assert sorted({v.pixel_samples for v in grid}) == [1, 2, 3]
    assert all(v.learn_amplitude is False and v.learn_residual is False for v in grid)
    assert all(v.fft_size == factor * 128 for v in grid
               for factor in (2, 4, 8) if v.fft_size == factor * 128)


def test_stage_b_grid_covers_both_capacity_flags():
    grid = variants.stage_b_variants(variants.Variant("A", "A", 128, 512, 1, False, False))
    assert len(grid) == 4
    assert {(v.learn_amplitude, v.learn_residual) for v in grid} == {
        (False, False), (True, False), (False, True), (True, True)}
    assert all(v.fft_size == 512 and v.pixel_samples == 1 for v in grid)


def test_metrics_report_cv_center_and_width_errors():
    xy = np.stack(np.meshgrid(np.arange(41) * 2.0, np.arange(41) * 2.0, indexing="xy"), axis=-1)
    measured = np.zeros((41, 41))
    measured[18:23, 18:23] = 1.0
    predicted = np.zeros((41, 41))
    predicted[20:27, 20:27] = 1.0
    references = {"U": {"spot": {"centroid_x": 40.0, "centroid_y": 40.0}},
                  "C": {"spot": {"centroid_x": 40.0, "centroid_y": 40.0}}}
    row = {"file": "s.npz", "label": "probe", "branch": "U", "family": "zernike", "split": "test"}
    result = variants.row_metrics(row, measured, predicted, xy, references)
    box = np.all(np.abs(xy - np.array([40.0, 40.0])) <= 30, axis=-1)
    expected_cv = measured[box].std() / measured[box].mean()
    assert result["measured_cv"] == pytest.approx(expected_cv)
    assert result["predicted_cv"] != pytest.approx(expected_cv)
    assert result["cv_abs_error"] == pytest.approx(abs(result["measured_cv"] - result["predicted_cv"]))
    # The predicted block is two samples further out on both axes: 2 CCD px each.
    assert result["center_error_px"] == pytest.approx(np.hypot(6.0, 6.0), rel=1e-6)
    assert result["fwhm_abs_error_px"] == pytest.approx(4.0, rel=1e-6)
    # Half-maximum masks: 5x5 and 7x7 blocks overlapping on 3x3 samples.
    assert result["iou"] == pytest.approx(9.0 / 65.0, rel=1e-6)
    assert result["d80_abs_error_px"] >= 0.0


def test_aggregate_splits_train_test_and_branches():
    rows = []
    for split in ("train", "test"):
        for branch in ("U", "C"):
            for _ in range(2):
                rows.append({"split": split, "branch": branch, "family": "zernike",
                             "corr": 0.9, "cv_abs_error": 0.02, "ee_roi_abs_error": 0.01,
                             "center_error_px": 0.5, "fwhm_abs_error_px": 1.0,
                             "fwhm_relative_error": 0.1, "d50_abs_error_px": 1.0,
                             "d80_abs_error_px": 2.0, "d80_relative_error": 0.05,
                             "rms_relative_error": 0.02, "iou": 0.8})
    rows.append({"split": "test", "branch": "U", "family": "reference", "corr": 0.1,
                 "cv_abs_error": 9.0, "ee_roi_abs_error": 9.0, "center_error_px": 9.0,
                 "fwhm_abs_error_px": 9.0, "fwhm_relative_error": 9.0, "d50_abs_error_px": 9.0,
                 "d80_abs_error_px": 9.0, "d80_relative_error": 9.0, "rms_relative_error": 9.0,
                 "iou": 0.0})
    metrics = variants.aggregate(rows)
    assert set(metrics) == {"train_U", "train_C", "test_U", "test_C"}
    assert metrics["test_U"]["count"] == 2
    assert metrics["test_U"]["cv_mae"] == pytest.approx(0.02)
    assert metrics["test_U"]["center_mae_px"] == pytest.approx(0.5)


def _fake_result(variant, train_cv, test_cv, corr=0.9):
    return {"dataset": "ds1", "variant": variant.as_dict(),
            "train_metrics": {"train_U": {"cv_mae": train_cv}, "train_C": {"cv_mae": train_cv}},
            "test_metrics": {"test_U": {"mean_corr": corr, "median_corr": corr, "min_corr": corr,
                                        "cv_mae": test_cv, "ee_mae": 0.01,
                                        "center_mae_px": 0.5, "d80_relative_mae": 0.02},
                             "test_C": {"mean_corr": corr, "median_corr": corr, "min_corr": corr,
                                        "cv_mae": test_cv, "ee_mae": 0.01,
                                        "center_mae_px": 0.5, "d80_relative_mae": 0.02}}}


def test_selection_uses_the_training_split_only():
    good_train = variants.Variant("A", "good", 128, 512, 1, False, False)
    good_test = variants.Variant("A", "good_test", 128, 1024, 1, False, False)
    results = [_fake_result(good_train, train_cv=0.01, test_cv=0.4),
               _fake_result(good_test, train_cv=0.2, test_cv=0.001)]
    winner = min(results, key=variants.selection_score)
    assert winner["variant"]["label"] == "good"
    assert variants.best_sampling_label(results, "ds1") == good_train.key


def test_sampling_convergence_reports_the_worst_relative_change():
    variants_a = variants.stage_a_variants(128)
    results = []
    for index, variant in enumerate(variants_a):
        results.append(_fake_result(variant, train_cv=0.01, test_cv=0.01 + 0.001 * index))
    convergence = variants.sampling_convergence(results, "ds1")
    assert convergence["chains"]["fft_size"]
    assert convergence["chains"]["pixel_samples"]
    assert convergence["worst_relative_change"] >= 0.0
    assert convergence["converged"] == bool(convergence["worst_relative_change"] <= 0.05)


def test_pairing_is_flagged_when_the_test_sets_differ():
    report_one = {"name": "ds1", "test_probe_labels": ["U:probe_001"], "paired_with_primary": None}
    report_two = {"name": "ds2", "test_probe_labels": ["U:probe_999"], "paired_with_primary": None}
    reports = [report_one, report_two]
    for report in reports:
        report["paired_with_primary"] = ("是" if report["test_probe_labels"]
                                         == reports[0]["test_probe_labels"] else "否")
    assert reports[0]["paired_with_primary"] == "是"
    assert reports[1]["paired_with_primary"] == "否"


def test_missing_workstream_module_fails_loudly():
    with pytest.raises(SystemExit) as error:
        variants.load_workstream_module("scripts.slm.does_not_exist", "scripts/slm/nope.py")
    assert "另一会话的在制品" in str(error.value)


def test_end_to_end_variant_grid_writes_the_expected_artifacts(tmp_path, synthetic_dataset):
    first, second = synthetic_dataset
    output = tmp_path / "variants"
    code = variants.main([
        "--dataset", f"ds1={first}", "--dataset", f"ds2={second}",
        "--output", str(output), "--iterations", "2", "--stage", "a", "--no-cross-check",
    ])
    assert code == 0
    for name in ("variants.json", "variants.csv", "summary.md", "sampling_grid.png"):
        assert (output / name).is_file()
    payload = json.loads((output / "variants.json").read_text(encoding="utf-8"))
    keys = {entry["variant"]["key"] for entry in payload["results"]}
    assert len(keys) == 9
    assert {entry["dataset"] for entry in payload["results"]} == {"ds1", "ds2"}
    assert payload["pairing"]["identical_test_sets"] is True
    for entry in payload["results"]:
        for metrics in entry["test_metrics"].values():
            assert np.isfinite(metrics["cv_mae"])
            assert 0.0 <= metrics["iou_mean"] <= 1.0
    csv_lines = (output / "variants.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].startswith("dataset,variant,variant_key,stage")
    assert len(csv_lines) > 1
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "阶段 A" in summary
    assert "停止条件对照" in summary


def test_end_to_end_flags_unpaired_datasets(tmp_path, synthetic_dataset):
    first, _ = synthetic_dataset
    other = tmp_path / "ds3"
    _write_dataset(other, [("probe_000", "train"), ("probe_777", "test")])
    output = tmp_path / "variants_unpaired"
    variants.main([
        "--dataset", f"ds1={first}", "--dataset", f"ds3={other}",
        "--output", str(output), "--iterations", "2", "--stage", "a", "--no-cross-check",
    ])
    payload = json.loads((output / "variants.json").read_text(encoding="utf-8"))
    assert payload["pairing"]["identical_test_sets"] is False
    assert payload["pairing"]["per_dataset"]["ds3"] == "否"
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "配对性不成立" in summary


def test_render_only_rebuilds_the_summary_from_stored_results(tmp_path, synthetic_dataset):
    first, _ = synthetic_dataset
    output = tmp_path / "variants"
    variants.main([
        "--dataset", f"ds1={first}", "--output", str(output),
        "--iterations", "2", "--stage", "a", "--no-cross-check",
    ])
    before = (output / "summary.md").read_text(encoding="utf-8")
    rendered = tmp_path / "rendered"
    code = variants.main(["--output", str(rendered),
                          "--render-only", str(output / "variants.json")])
    assert code == 0
    assert (rendered / "summary.md").read_text(encoding="utf-8") == before
    assert (rendered / "sampling_grid.png").is_file()
