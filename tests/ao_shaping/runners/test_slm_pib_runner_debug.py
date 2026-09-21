"""Tests for ``slm_pib_runner``: debug artifacts + CLI options (no hardware).

The debug figure is exercised through ``_save_debug_artifacts`` with a synthetic
:class:`Recorder`, so the whole image-output path is covered offline.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from ao_shaping.runners.slm_pib_runner import (
    HeuristicParams,
    ObjectiveParams,
    SlmParams,
    _save_debug_artifacts,
    run,
)
from ao_shaping.utils.io.file import Recorder


def _make_recorder(objective: str, mode: str, n: int = 3) -> Recorder:
    rec = Recorder(mark=objective, mode=mode)
    for i in range(n):
        img = np.full((16, 16), 10 * (i + 1), dtype=np.uint8)
        rec.append(
            {
                "J": 0.1 * i,
                objective: 0.1 * (i + 1),
                "_p%": 0.5,
                "_max_r": 5.0,
                "_c": np.linspace(-1.0, 1.0, 6),
                "_img": img,
                "_diff": 0.0,
                "lr": 0.1,
                "r": 5.0,
                "delta": 0.1,
                "_epoch": i,
                "exp_t": 10.0,
                "max_brt": float(img.max()),
                "_grad": np.zeros(6),
            }
        )
    return rec


@pytest.mark.parametrize(
    "objective,mode", [("pib", "max"), ("radiu", "min"), ("avg_radiu", "max")]
)
def test_debug_artifacts_written_for_every_objective(tmp_path, objective, mode):
    rec = _make_recorder(objective, mode)

    png = _save_debug_artifacts(
        rec,
        ObjectiveParams(name=objective),
        SlmParams(),
        ObjectiveParams(name=objective),
        str(tmp_path),
    )

    assert png.suffix == ".png"
    assert png.exists() and png.stat().st_size > 0
    assert png.with_suffix(".pkl").exists()
    assert png.with_suffix(".json").exists()


def test_debug_artifact_json_round_trips_config(tmp_path):
    rec = _make_recorder("pib", "max")
    png = _save_debug_artifacts(
        rec,
        ObjectiveParams(name="pib"),
        SlmParams(),
        HeuristicParams(algorithm="ga"),
        str(tmp_path),
    )

    payload = json.loads(png.with_suffix(".json").read_text(encoding="utf8"))
    assert payload == {"algorithm": "ga"}


def test_cli_exposes_debug_and_heuristic_options():
    result = CliRunner().invoke(run, ["heuristic", "--help"])

    assert result.exit_code == 0, result.output
    for opt in ("--debug", "--algorithm", "--pop_size", "--cam_type"):
        assert opt in result.output
    for algo in ("spgd", "ga", "pso", "sa", "hc", "rs", "cem", "de"):
        assert algo in result.output


def test_algorithm_choices_match_optimizer():
    from ao_shaping.optimizer.wfless.slm_zernike_pib import ALGORITHM_CHOICES

    assert set(ALGORITHM_CHOICES) == {"spgd", "ga", "pso", "sa", "hc", "rs", "cem", "de"}


def test_main_group_exposes_debug_flag():
    from ao_shaping.main import cli

    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--debug" in result.output
