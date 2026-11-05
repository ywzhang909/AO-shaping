"""Tests for the offline transfer uncertainty diagnostics (device-free)."""
from __future__ import annotations

import subprocess
import sys
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer.uncertainty import (
    disagreement,
    evaluate,
    normalized,
    select_diverse,
)


def test_normalized_is_gain_invariant() -> None:
    image = np.array([[1., 2.], [3., 4.]])
    assert normalized(image * 7) == pytest.approx(normalized(image))


def test_disagreement_is_gain_invariant_and_detects_shape_disagreement() -> None:
    image = np.array([[1., 2.], [3., 4.]])
    predictions = np.stack([np.stack([image, image]),
                            np.stack([image * 7, image[::-1]])])
    scores = disagreement(predictions)
    assert scores[0] < 1e-12
    assert scores[1] > .1


def test_disagreement_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        disagreement(np.zeros((2, 3, 4)))  # wrong dimensionality
    with pytest.raises(ValueError):
        disagreement(np.full((1, 2, 3, 4), -1.))  # needs >=2 members
    with pytest.raises(ValueError):
        disagreement(np.full((2, 2, 3, 4), np.nan))  # non-finite


def test_select_diverse_rejects_wrapped_duplicates() -> None:
    phases = np.stack([np.zeros((8, 8)), np.full((8, 8), 2 * np.pi),
                       np.indices((8, 8))[0] * .4])
    assert select_diverse(np.array([3., 2., 1.]), phases, 2) == [0, 2]


def test_select_diverse_validates_inputs() -> None:
    phases = np.zeros((3, 8, 8))
    with pytest.raises(ValueError):
        select_diverse(np.array([1., 2.]), phases, 2)  # score/phase mismatch
    with pytest.raises(ValueError):
        select_diverse(np.array([1., 2., 3.]), phases, 0)  # count out of range


def test_evaluate_reports_paired_shape_error() -> None:
    prediction = np.stack([np.array([[1., 2.], [3., 4.]]),
                           np.array([[4., 3.], [2., 1.]])])
    result = evaluate(prediction, prediction.copy())
    assert result["mean_ncc"] == pytest.approx(1.0)
    assert result["mean_shape_relative_l2"] == pytest.approx(0.0, abs=1e-9)
    assert len(result["shape_relative_l2"]) == 2


def test_main_does_not_import_device_modules(tmp_path) -> None:
    """The CLI entry must rank/evaluate without loading any camera/SLM SDK."""
    root = Path(__file__).resolve().parents[5]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    npz = tmp_path / "_uncertainty_cli_dummy.npz"
    np.savez(npz, predictions=np.stack([np.stack([np.array([[1., 2.], [3., 4.]])])
                                        for _ in range(2)]),
             phases=np.zeros((1, 4, 4)))
    probe = subprocess.run(
        [sys.executable, "-m", "ao_shaping.tools.slm.transfer.uncertainty",
         "rank", "--input", str(npz), "--count", "1",
         "--out", str(npz.with_suffix(".json"))],
        capture_output=True, text=True, env=env, check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert "ao_shaping.drivers" not in probe.stderr
    result = json.loads(npz.with_suffix(".json").read_text(encoding="utf-8"))
    assert result["selected"] == [0] and result["calibrated"] is False
    npz.unlink()
    npz.with_suffix(".json").unlink()


def test_ranking_module_and_cli_do_not_import_driver_sdks() -> None:
    """Clean-subprocess check that drivers stay out of sys.modules."""
    root = Path(__file__).resolve().parents[5]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    code = (
        "import sys, json;\n"
        "from ao_shaping.tools.slm.transfer import uncertainty;\n"
        "leaked=[n for n in sys.modules if n.startswith('ao_shaping.drivers')];\n"
        "assert not leaked, leaked"
    )
    probe = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, env=env)
    assert probe.returncode == 0, probe.stderr
