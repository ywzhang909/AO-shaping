"""Selection must depend on model disagreement and retain command diversity."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _experiment():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "transfer_uncertainty_experiment", root / "scripts/slm/transfer_uncertainty_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uncertainty_is_gain_invariant_and_detects_shape_disagreement():
    experiment = _experiment()
    image = np.array([[1., 2.], [3., 4.]])
    predictions = np.stack([np.stack([image, image]),
                            np.stack([image * 7, image[::-1]])])
    scores = experiment.disagreement(predictions)
    assert scores[0] < 1e-12
    assert scores[1] > .1


def test_selection_rejects_wrapped_duplicate_commands():
    experiment = _experiment()
    phases = np.stack([np.zeros((8, 8)), np.full((8, 8), 2 * np.pi),
                       np.indices((8, 8))[0] * .4])
    assert experiment.select_diverse(np.array([3., 2., 1.]), phases, 2) == [0, 2]
