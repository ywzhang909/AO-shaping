"""Every SLM-phase optimizer runner must expose the heuristic-search options.

No hardware: only the click surfaces are inspected.
"""

from __future__ import annotations

import importlib

import pytest
from click.testing import CliRunner

from ao_shaping.algorithm.heuristic.search import heuristic_algorithm_choices

EXPECTED_ALGORITHMS = ("spgd", "ga", "pso", "sa", "hc", "rs", "cem", "de")

# runner module path -> display name
SLM_PHASE_RUNNERS = [
    ("ao_shaping.runners.slm_pib_runner", "slm-pib"),
    ("ao_shaping.runners.slm.rms_zernike_runner", "rms-zernike"),
    ("ao_shaping.runners.greedy_zernike_runner", "greedy-zernike"),
]


def test_helper_choices_match_expected_set():
    assert heuristic_algorithm_choices() == EXPECTED_ALGORITHMS


@pytest.mark.parametrize("module_path,name", SLM_PHASE_RUNNERS)
def test_runner_exposes_algorithm_and_pop_size(module_path, name):
    module = importlib.import_module(module_path)
    result = CliRunner().invoke(module.run, ["--help"])

    assert result.exit_code == 0, f"{name}: {result.output}"
    assert "--algorithm" in result.output, name
    assert "--pop_size" in result.output, name
    for algo in EXPECTED_ALGORITHMS:
        assert algo in result.output, f"{name} missing {algo}"
