"""Contract tests for the offline physical-residual fitting entrypoint.

No physical data and no hardware are touched here.  The tests pin the three
properties the entrypoint must not get wrong:

* :class:`MeasuredBench` is an *exact* command lookup that refuses unknown
  commands and unsupported exposures, and never mutates its input;
* the train/holdout split is the calibration's fixed split (disjoint, sorted,
  identical to the literal reference recipe);
* the small calibrated synthetic smoke run produces the documented objective
  and artifact contract (finite metrics, no NaN in JSON, delta/coeffs in NPZ,
  evidence flags set).
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

from ml.sim2f.calibrate import BenchSample, BenchSession, CalibrationResult
from ml.sim2f.config import OpticsErrors, Sim2fConfig
from ml.sim2f.optics import Sim2f


def _load_script(name: str = "sim2f_physical_residual_under_test"):
    path = ROOT / "scripts" / "ml" / "sim2f_physical_residual.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


def _small_bench(grid: int = 8, ccd: int = 4, count: int = 4):
    rng = np.random.default_rng(3)
    phases = rng.normal(0.0, 0.1, size=(count, grid, grid)).astype(np.float32)
    phases[0] = 0.0
    frames = rng.normal(10.0, 1.0, size=(count, ccd, ccd)).astype(np.float32)
    return phases, frames


def test_measured_bench_exact_lookup_and_shape():
    phases, frames = _small_bench()
    bench = script.MeasuredBench(phases, frames)

    out = bench.forward(np.stack([phases[2], phases[0]]))
    assert out.shape == (2, 1, bench.ccd_px, bench.ccd_px)
    assert torch.equal(out[:, 0], torch.from_numpy(frames[[2, 0]]))

    single = bench.forward(phases[1])
    assert single.shape == (1, 1, bench.ccd_px, bench.ccd_px)
    assert torch.equal(single[0, 0], torch.from_numpy(frames[1]))


def test_measured_bench_unknown_phase_raises():
    phases, frames = _small_bench()
    bench = script.MeasuredBench(phases, frames)
    with pytest.raises(ValueError, match="unknown command phase"):
        bench.forward(phases[1] + np.float32(1e-3))


def test_measured_bench_shape_validation_and_read_only_input():
    phases, frames = _small_bench()
    bench = script.MeasuredBench(phases, frames)
    before = phases.copy()
    bench.forward(phases)
    assert np.array_equal(phases, before)

    with pytest.raises(ValueError, match="phase must be"):
        bench.forward(phases[:, :-1, :])
    with pytest.raises(ValueError, match="phase must be"):
        bench.forward(phases[None])


def test_measured_bench_exposure_is_unsupported():
    phases, frames = _small_bench()
    bench = script.MeasuredBench(phases, frames)
    with pytest.raises(ValueError, match="exposure"):
        bench.forward(phases[0], exposure_ms=0.15)


def test_measured_bench_duplicate_command_uses_first_frame_and_reports_counts():
    phases, frames = _small_bench(count=3)
    duplicated = np.stack([phases[0], phases[1], phases[0]])
    repeated_frames = np.stack([frames[0], frames[1], frames[2] + 100.0])
    bench = script.MeasuredBench(duplicated, repeated_frames)

    assert bench.frame_count == 3
    assert bench.unique_count == 2
    assert bench.duplicate_group_sizes == [2]
    out = bench.forward(phases[0])
    assert torch.equal(out[0, 0], torch.from_numpy(repeated_frames[0]))


def _reference_split(count: int) -> tuple[np.ndarray, np.ndarray]:
    """The literal recipe copied from the calibration stage."""
    n_holdout = int(round(0.25 * count))
    n_holdout = max(0, min(n_holdout, count - 1))
    order = np.random.default_rng(0).permutation(count)
    return np.sort(order[n_holdout:]), np.sort(order[:n_holdout])


@pytest.mark.parametrize("count", [4, 12, 100, 173])
def test_split_matches_calibration_recipe_and_is_disjoint(count: int):
    train, holdout = script.split_indices(count)
    expected_train, expected_holdout = _reference_split(count)
    assert np.array_equal(train, expected_train)
    assert np.array_equal(holdout, expected_holdout)
    assert train.size + holdout.size == count
    assert set(train).isdisjoint(set(holdout))
    assert np.array_equal(np.union1d(train, holdout), np.arange(count))
    assert np.all(np.diff(train) > 0)
    assert np.all(np.diff(holdout) > 0)


def test_split_is_not_tied_to_the_fit_seed():
    params = inspect.signature(script.split_indices).parameters
    assert list(params) == ["count"]
    assert (script.SPLIT_SEED, script.HOLDOUT_FRAC) == (0, 0.25)
    train, holdout = script.split_indices(40)
    expected_train, expected_holdout = _reference_split(40)
    assert np.array_equal(train, expected_train)
    assert np.array_equal(holdout, expected_holdout)


def _synthetic_pair():
    """A small ideal model plus a non-ideal device, both real Sim2f objects."""
    cfg = Sim2fConfig()
    model = Sim2f(cfg, device="cpu")
    device_cfg = replace(
        cfg,
        errors=OpticsErrors(slm_zernike=((4, 0.20), (6, -0.12))),
    )
    device = Sim2f(device_cfg, device="cpu")

    rng = np.random.default_rng(11)
    basis = model.zernike_basis(6).numpy()
    phases = np.zeros((12, cfg.grid, cfg.grid), dtype=np.float32)
    for index in range(1, 12):
        coeffs = rng.normal(0.0, 0.08, size=basis.shape[0])
        phases[index] = np.tensordot(coeffs, basis, axes=1).astype(np.float32)

    with torch.no_grad():
        frames = (
            device.forward(torch.from_numpy(phases), detector=False)
            .squeeze(1)
            .numpy()
            .astype(np.float32)
        )
    return cfg, model, phases, frames


def test_synthetic_smoke_objective_and_artifact_contract(tmp_path):
    cfg, model, phases, frames = _synthetic_pair()
    calibration = CalibrationResult(
        ok=True,
        reason=None,
        sim_config=cfg,
        aberration=np.zeros((cfg.grid, cfg.grid), dtype=np.float32),
        illumination=model.nominal_illumination[0, 0].numpy().astype(np.float32),
    )
    session = BenchSession(
        root=tmp_path,
        samples=tuple(
            BenchSample(phase=phases[i], frame=frames[i], label=f"s{i}")
            for i in range(phases.shape[0])
        ),
        sim_config=cfg,
        backend="sim",
    )

    report, arrays = script.run_fit(
        session,
        calibration,
        epochs=2,
        lr=0.05,
        seed=0,
        modes="data_only,pinn",
        window_px=128,
        n_modes=8,
        device="cpu",
    )

    split = report["split"]
    assert split["n"] == 12
    assert split["n_holdout"] == 3
    assert split["n_train"] == 9
    assert set(split["train"]).isdisjoint(split["holdout"])
    assert np.array_equal(arrays["train_indices"], np.asarray(split["train"]))
    assert np.array_equal(arrays["holdout_indices"], np.asarray(split["holdout"]))

    assert report["shaping"]["used"] is False
    assert report["hardware_commands"] == []
    assert report["evidence_status"] == "offline_physical_reference_only"
    assert report["acceptance_status"] == "not_ccd_acceptance"
    assert report["exposure"]["supported"] is False

    assert report["arms"]["data_only"]["weights"]["support"] == 0.0
    assert report["arms"]["data_only"]["physics"] is False
    assert report["arms"]["pinn"]["physics"] is True
    assert report["arms"]["pinn"]["weights"]["support"] > 0.0

    for arm in ("data_only", "pinn"):
        entry = report["arms"][arm]
        for split_name in ("train", "holdout", "leak_free_holdout"):
            for stage in ("start", "end"):
                assert set(entry[stage][split_name]) == {
                    "l1",
                    "ncc",
                    "gain",
                    "l1_raw",
                    "ncc_raw",
                }
        assert entry["ok"] is True, entry["reason"]
        assert entry["delta_rms_rad"] >= 0.0
        assert arrays[f"{arm}_delta"].shape == (cfg.grid, cfg.grid)
        assert np.isfinite(arrays[f"{arm}_delta"]).all()
        assert arrays[f"{arm}_coeffs"].shape == (5,)  # Noll 4..8
        assert np.isfinite(arrays[f"{arm}_coeffs"]).all()

    out = tmp_path / "artifacts"
    json_path, npz_path = script.write_artifacts(out, report, arrays)
    text = json_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    payload = json.loads(text)
    assert payload["evidence_status"] == "offline_physical_reference_only"

    loaded = np.load(npz_path)
    for key in (
        "train_indices",
        "holdout_indices",
        "data_only_delta",
        "data_only_coeffs",
        "pinn_delta",
        "pinn_coeffs",
    ):
        assert key in loaded

    # A second write to the same directory is refused: artifacts are new-only.
    with pytest.raises(FileExistsError):
        script.write_artifacts(out, report, arrays)


def test_calibrate_matched_score_equals_the_calibration_reference():
    """The reported L1/NCC must be comparable with calibrate_summary.json.

    ``calibrate._scores`` is the scorer that produced the frozen calibration's
    numbers, so the entrypoint's metric is pinned to it numerically instead of
    being re-derived from memory.
    """
    from ml.sim2f.calibrate import _scores

    generator = torch.Generator().manual_seed(11)
    truth = torch.rand((3, 1, 12, 12), generator=generator) * 40.0 + 2.0
    predicted = truth * 0.4 + torch.rand((3, 1, 12, 12), generator=generator) * 5.0

    l1, ncc, gain = script._calibrate_matched(predicted, truth)
    ref_l1, ref_ncc, ref_gain = _scores(predicted, truth, fit_gain=True)

    assert torch.allclose(l1, ref_l1.reshape(-1), atol=1e-6)
    assert torch.allclose(ncc, ref_ncc.reshape(-1), atol=1e-6)
    assert torch.allclose(gain, ref_gain.reshape(-1), atol=1e-6)


def test_a_uniform_pedestal_does_not_change_the_prepared_metric():
    """The stored frames carry a constant +1 count pedestal; prepare_frame drops it."""
    generator = torch.Generator().manual_seed(5)
    truth = torch.rand((2, 1, 16, 16), generator=generator) * 30.0 + 2.0
    predicted = truth.clone()

    base_l1, base_ncc, _ = script._calibrate_matched(predicted, truth)
    bumped_l1, bumped_ncc, _ = script._calibrate_matched(predicted, truth + 7.0)

    assert torch.allclose(base_l1, bumped_l1, atol=1e-5)
    assert torch.allclose(base_ncc, bumped_ncc, atol=1e-5)
    assert float(base_l1.mean()) < 1e-5
