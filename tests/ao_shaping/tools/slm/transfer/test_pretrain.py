"""Pretraining library: device-free import, artifact round-trip, grouped isolation."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer.pretrain import (
    PARAM_NAMES,
    combo_grid,
    grouped_folds,
    pretrain,
    train_estimator,
)

TINY = dict(combos=6, probes_per_combo=4, fft_size=64, grid_size=32)


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
