"""Tests for the single-source-of-truth objective search direction.

``ao_shaping.utils.objective_direction`` owns the one vocabulary that maps
an objective name to the direction the search ascends (``"max"``) or descends
(``"min"``). Every optimizer and the PIB report script now import it instead
of keeping a private copy, so these tests pin the vocabulary that the whole
closed loop relies on.
"""
from __future__ import annotations

import pytest

from ao_shaping.utils.objective_direction import (
    MAXIMIZE_OBJECTIVES,
    objective_direction,
)


@pytest.mark.parametrize(
    "objective",
    ["pib", "avg_radiu", "shape", "roi_pib", "rms_pib", "quality"],
)
def test_maximize_objectives_ascend(objective: str) -> None:
    assert objective in MAXIMIZE_OBJECTIVES
    assert objective_direction(objective) == "max"


@pytest.mark.parametrize(
    "objective",
    ["radiu", "rmse", "rmse_out", "pearson"],
)
def test_minimize_objectives_descend(objective: str) -> None:
    assert objective_direction(objective) == "min"


@pytest.mark.parametrize("objective", ["", "not_an_objective", "radius"])
def test_unknown_defaults_to_min(objective: str) -> None:
    # Historical fallback: anything not in the vocabulary descends. This keeps
    # a misspelled or brand-new objective from silently flipping a search
    # from descending to ascending.
    assert objective_direction(objective) == "min"


@pytest.mark.parametrize("objective,expected", [("PIB", "max"), ("Shape", "max"), ("RMS_PIB", "max")])
def test_comparison_is_case_insensitive(objective: str, expected: str) -> None:
    # Convenience for report scripts; production validators already lowercase.
    assert objective_direction(objective) == expected


def test_return_type_is_plain_string() -> None:
    # Call sites compare with ``== "max"`` and pass the value straight into
    # ``Recorder(mode=...)`` / ``Mode.from_string``; it must be a plain str.
    value = objective_direction("pib")
    assert isinstance(value, str)
    assert value == "max"
