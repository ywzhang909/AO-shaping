"""Pretraining library: device-free import, artifact round-trip, grouped isolation."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer.pretrain import (
    NOMINAL_CENTER,
    NOMINAL_DEFOCUS,
    NOMINAL_WIDTH,
    PARAM_NAMES,
    combo_grid,
    combo_target,
    grouped_folds,
    load_bench,
    nominal_parameters,
    pretrain,
    train_estimator,
    truth_parameters,
)

TINY = dict(combos=6, probes_per_combo=4, fft_size=64, grid_size=32)


def _write_bench_fixture(path):
    """A distinctive bench-shaped ``frozen_parameters.json`` (values != old-bench defaults)."""
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusParameters,
    )

    bench = FrozenDefocusParameters.from_dicts(
        dict(defocus_diopter=0.0815, origin=[1352.5, 635.5],
             amplitude_center=[949.0, 582.75], amplitude_width=[586.0, 388.5],
             residual_noll_radians={"5": 0.01, "6": -0.02, "7": 0.0, "8": 0.0,
                                    "9": 0.0, "10": 0.0, "11": 0.03},
             gain=7.5, background=0.215, sigma_ccd_px=0.0),
        [[-192.8, -8263.6], [-8428.1, 23.5]],
        fft_size=1024, pixel_samples=3, grid_size=128)
    payload = bench.to_dict()
    payload["ramp_matrix"] = [list(row) for row in bench.ramp_matrix]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return bench, path


def test_default_nominal_keeps_historical_placeholders():
    """bench=None must keep the shipped NOMINAL_* values and the (historical) fft//2 origin."""
    nominal = nominal_parameters(fft_size=64, pixel_samples=1, grid_size=32)
    assert nominal.defocus_diopter == NOMINAL_DEFOCUS
    assert nominal.amplitude_center_xy == NOMINAL_CENTER
    assert nominal.amplitude_width_xy == NOMINAL_WIDTH
    assert nominal.origin_xy == (32.0, 32.0)
    assert nominal.ramp_matrix == ((0.0, 5000.0), (-5000.0, 0.0))
    assert nominal.fft_size == 64 and nominal.grid_size == 32


def test_bench_injection_takes_geometry_origin_and_ramp_from_bench(tmp_path):
    bench, path = _write_bench_fixture(tmp_path / "frozen_parameters.json")

    loaded = load_bench(path)
    assert loaded == bench
    nominal = nominal_parameters(bench=path)

    # Frame fix: origin is the bench camera-pixel intercept, not the FFT index fft//2.
    assert nominal.origin_xy == pytest.approx(bench.origin_xy)
    assert nominal.origin_xy != (512.0, 512.0)
    assert nominal.defocus_diopter == pytest.approx(bench.defocus_diopter)
    assert nominal.amplitude_center_xy == pytest.approx(bench.amplitude_center_xy)
    assert nominal.amplitude_width_xy == pytest.approx(bench.amplitude_width_xy)
    assert nominal.ramp_matrix == bench.ramp_matrix
    assert nominal.residual_noll_radians == bench.residual_noll_radians
    assert nominal.gain == pytest.approx(bench.gain)
    assert nominal.background == pytest.approx(bench.background)
    # Geometry comes from the bench when the caller does not override it.
    assert (nominal.fft_size, nominal.grid_size, nominal.pixel_samples) == (1024, 128, 3)
    # Explicit caller overrides still win.
    assert nominal_parameters(bench=path, fft_size=256).fft_size == 256


def test_truth_and_target_share_the_bench_nominal(tmp_path):
    """Truth = nominal + offset for every field, and the target is exactly that offset."""
    bench, path = _write_bench_fixture(tmp_path / "frozen_parameters.json")
    combo = dict(combo=0, defocus=0.06, shift=(12.0, -8.0), width_scale=1.2, aberration=0.05)

    truth = truth_parameters(combo, bench=path)
    nominal = nominal_parameters(bench=path)
    assert truth.defocus_diopter == pytest.approx(nominal.defocus_diopter + 0.06)
    assert truth.amplitude_center_xy == pytest.approx(
        (nominal.amplitude_center_xy[0] + 12.0, nominal.amplitude_center_xy[1] - 8.0))
    assert truth.amplitude_width_xy == pytest.approx(
        (nominal.amplitude_width_xy[0] * 1.2, nominal.amplitude_width_xy[1] * 1.2))
    for j in range(5, 12):
        assert truth.residual_noll_radians[j] == pytest.approx(
            nominal.residual_noll_radians[j] + 0.05)
    assert combo_target(combo).tolist() == pytest.approx([0.06, 12.0, -8.0, np.log(1.2), 0.05])


def test_pretrain_imports_without_drivers_or_torch():
    """Must be a subprocess: this pytest process may already hold device modules."""
    code = (
        "import sys\n"
        "import ao_shaping.tools.slm.transfer.pretrain as module\n"
        "leaked = sorted(n for n in sys.modules if n.startswith('ao_shaping.drivers'))\n"
        "assert not leaked, leaked\n"
        "assert not [n for n in sys.modules if n.split('.')[0] == 'torch'], 'torch imported'\n"
        "assert hasattr(module, 'pretrain') and hasattr(module, 'pair_features')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_tiny_pretrain_writes_reloadable_frozen_parameters(tmp_path):
    out = tmp_path / "pretrain"
    result = pretrain(out, **TINY)

    for name in ("estimator.npz", "frozen_parameters.json", "report.json"):
        assert (out / name).is_file(), name

    payload = json.loads((out / "frozen_parameters.json").read_text(encoding="utf-8"))
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusParameters,
        load_frozen_parameters,
    )

    reloaded = FrozenDefocusParameters.from_dicts(
        payload, payload["ramp_matrix"], grid_size=payload["grid_size"],
        fft_size=payload["fft_size"], pixel_samples=payload["pixel_samples"])
    assert reloaded.fft_size == TINY["fft_size"]
    assert reloaded.grid_size == TINY["grid_size"]
    assert reloaded.pixel_samples == 1
    assert reloaded.defocus_diopter == pytest.approx(payload["defocus_diopter"])
    assert tuple(reloaded.amplitude_center_xy) == tuple(payload["amplitude_center"])
    assert sorted(reloaded.residual_noll_radians) == list(range(5, 12))

    # The same file is what `transfer run --checkpoint` consumes.
    assert load_frozen_parameters(out / "frozen_parameters.json") == reloaded
    assert payload["pretrain"]["params"] == list(PARAM_NAMES)
    assert len(payload["pretrain"]["mean_delta"]) == len(PARAM_NAMES)

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["combinations"] == TINY["combos"]
    assert report["folds_scored"] == TINY["combos"]
    assert report["setup"]["fft_size"] == TINY["fft_size"]
    assert set(report["metrics"]) == set(PARAM_NAMES)
    assert "0.01" in report["alpha_sweep"] and "1" in report["alpha_sweep"]

    estimator = np.load(out / "estimator.npz")
    assert estimator["coef"].shape == (16, len(PARAM_NAMES))
    assert [str(v) for v in estimator["params"]] == list(PARAM_NAMES)
    assert result["combinations"] == TINY["combos"]

    with pytest.raises(FileExistsError):
        pretrain(out, **TINY)


def test_grouped_folds_hold_out_whole_combinations():
    combo_ids = np.array([0, 0, 1, 1, 2, 2])
    folds = grouped_folds(combo_ids)
    assert [held for held, _, _ in folds] == [0, 1, 2]
    for held, train, test in folds:
        assert set(combo_ids[test]) == {held}
        assert held not in set(combo_ids[train])
        assert set(combo_ids[train]) == {0, 1, 2} - {held}
        assert int(train.sum()) == 4 and int(test.sum()) == 2
        assert not (train & test).any()
    with pytest.raises(ValueError, match="three parameter combinations"):
        grouped_folds([0, 0, 1, 1])


def test_train_estimator_scores_every_combination_once():
    rng = np.random.default_rng(7)
    combo_ids = np.repeat(np.arange(6), 4)
    library = dict(features=rng.normal(size=(24, 6)),
                   targets=rng.normal(size=(24, len(PARAM_NAMES))), combo_ids=combo_ids)
    model = train_estimator(library, alpha=0.01)
    assert model["combinations"] == 6
    assert model["scored"] == 24
    assert len(model["folds"]) == 6
    assert {fold["combo"] for fold in model["folds"]} == set(range(6))
    for fold in model["folds"]:
        # Held-out combo never trains the fold that scores it.
        assert fold["train_samples"] == 20
        assert fold["test_samples"] == 4
        assert set(fold["mae"]) == set(PARAM_NAMES)
    assert set(model["metrics"]) == set(PARAM_NAMES)
    assert model["coef"].shape == (6, len(PARAM_NAMES))


def test_combo_grid_is_deterministic_and_limited():
    full = combo_grid()
    assert len(full) == 5 * 3 * 3 * 2
    assert combo_grid() == full
    assert [row["combo"] for row in combo_grid(6)] == list(range(6))
    assert combo_grid(6) == full[:6]
