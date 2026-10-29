"""Tests for the single-source-of-truth objective search direction.

``ao_shaping.utils.objective_direction`` owns the one vocabulary that maps
an objective name to the direction the search ascends (``"max"``) or descends
(``"min"``). Every optimizer and the PIB report script now import it instead
of keeping a private copy, so these tests pin the vocabulary that the whole
closed loop relies on.

``utils/image/target/objective.py`` used to carry a fourth copy (without
``quality``) and its own ``RecordMode``; both are gone, and the absence
assertion at the bottom is what keeps them from drifting back.
"""
from __future__ import annotations

import pytest

from ao_shaping.utils.io.record_mode import RecordMode
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


@pytest.mark.parametrize(
    "objective",
    [*MAXIMIZE_OBJECTIVES, "radiu", "rmse", "not_an_objective"],
)
def test_the_wire_value_coerces_back_to_the_recorder_mode(objective: str) -> None:
    # ``objective_direction`` returns the wire string, but the recorder's search
    # direction is a ``RecordMode``. Coercion must round-trip through the enum,
    # otherwise the sign of the gradient and the recorder's best-pick would be
    # decided by two different vocabularies again (issue #130).
    mode = RecordMode.coerce(objective_direction(objective))
    assert isinstance(mode, RecordMode)
    assert mode == objective_direction(objective)
    assert str(mode) == objective_direction(objective)
    assert mode.maximizing is (objective in MAXIMIZE_OBJECTIVES)
    # Historical sign convention: maximise -> -1 (negate the descent step).
    assert mode.sign == (-1.0 if objective in MAXIMIZE_OBJECTIVES else 1.0)


def test_target_package_carries_no_copy_of_the_vocabulary() -> None:
    # ``utils/image/target/objective.py`` kept its own tuple + functions (and its
    # own ``RecordMode``) until issue #130, and the ``target`` package re-exported
    # them. Both copies are gone: the vocabulary lives only in
    # ``ao_shaping.utils.objective_direction``, and any name reappearing in the
    # legacy module is the first step of a fourth copy growing back.
    from ao_shaping.utils.image.target import objective as objective_module

    for name in ("MAXIMIZE_OBJECTIVES", "objective_direction"):
        assert not hasattr(objective_module, name), (
            f"{name} is back in utils/image/target/objective.py; the single source "
            "is ao_shaping.utils.objective_direction"
        )
