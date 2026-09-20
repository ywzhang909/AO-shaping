"""Offline tests for the SLM-Zernike heuristic wiring (no hardware).

The shared driver itself is covered by
``tests/ao_shaping/algorithm/test_heuristic_search.py``; here we only pin the
integration points of ``optimize_slm_zernike_pib`` after the refactor onto it.
"""

from __future__ import annotations

import inspect

import pytest

from ao_shaping.algorithm.heuristic.search import heuristic_algorithm_choices
from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    ALGORITHM_CHOICES,
    optimize_slm_zernike_pib,
)


def test_algorithm_choices_come_from_the_shared_driver():
    assert ALGORITHM_CHOICES == heuristic_algorithm_choices()
    assert ALGORITHM_CHOICES[0] == "spgd"
    assert set(ALGORITHM_CHOICES[1:]) == {"ga", "pso", "sa", "hc", "rs", "cem", "de"}


def test_optimizer_exposes_algorithm_and_pop_size():
    params = inspect.signature(optimize_slm_zernike_pib).parameters

    assert "algorithm" in params
    assert params["algorithm"].default == "spgd"
    assert "pop_size" in params
    assert params["pop_size"].default is None


def test_unknown_algorithm_rejected_before_touching_hardware():
    # Validation runs before the camera/SLM context managers, so this stays
    # offline: no device is opened.
    with pytest.raises(ValueError, match="algorithm must be one of"):
        optimize_slm_zernike_pib(center="shape", epochs=1, algorithm="bogus")
