"""Staged objective weights and per-block weight overrides (issue #213 item 4c).

No hardware: everything here is the pure weight arithmetic plus one
``ShapeObjective`` evaluation on a synthetic image.
"""
from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.multishape_objective import (
    DEFAULT_WEIGHTS, ShapeObjective,
)
from ao_shaping.algorithm.signal_processing.weight_schedule import (
    PHASES, PHASE_SCALE, phase_for, staged_weights,
)


def _target(size: int = 32, box: int = 8) -> np.ndarray:
    target = np.zeros((size, size), dtype=np.float32)
    lo = size//2 - box//2
    target[lo:lo + box, lo:lo + box] = 1.0
    return target


# --- phase assignment -------------------------------------------------------

def test_thirds_split_the_stage_by_position():
    blocks = 9
    got = [phase_for(block, blocks) for block in range(blocks)]
    assert got == ["energy"]*3 + ["shape"]*3 + ["cv2"]*3


def test_short_stage_collapses_to_the_middle_phase():
    # A 1- or 2-block stage cannot be split into thirds; it gets the middle term
    # rather than an arbitrary sliver of the first or last phase.
    assert [phase_for(block, 1) for block in range(1)] == ["shape"]
    assert [phase_for(block, 2) for block in range(2)] == ["shape", "shape"]


def test_phase_names_match_the_scale_table():
    assert set(PHASES) == set(PHASE_SCALE)


# --- weight arithmetic ------------------------------------------------------

def test_staged_weights_scale_only_the_staged_keys():
    base = DEFAULT_WEIGHTS["shape"]
    early = staged_weights("shape", 0, 9)
    assert early["energy"] == pytest.approx(base["energy"]*PHASE_SCALE["energy"]["energy"])
    assert early["shape"] == pytest.approx(base["shape"]*PHASE_SCALE["energy"]["shape"])
    assert early["cv2"] == pytest.approx(base["cv2"]*PHASE_SCALE["energy"]["cv2"])
    # smooth is not staged: it must stay comparable across phases.
    assert early["smooth"] == pytest.approx(base["smooth"])


def test_staged_weights_leave_formation_only_terms_at_base():
    base = DEFAULT_WEIGHTS["formation"]
    late = staged_weights("formation", 8, 9)
    assert late["cv2"] == pytest.approx(base["cv2"]*PHASE_SCALE["cv2"]["cv2"])
    for key in ("partition", "peripheral", "floor", "boundary"):
        assert late[key] == pytest.approx(base[key])


def test_cv_mode_has_nothing_to_stage_and_says_so_by_returning_base():
    # Documented no-op: the CV-only loss has no energy/shape term. The CLI
    # rejects this combination instead of running an un-staged schedule, so this
    # test pins the library behaviour that rejection relies on.
    assert staged_weights("cv", 0, 9) == pytest.approx(DEFAULT_WEIGHTS["cv"])
    assert staged_weights("cv", 8, 9) == pytest.approx(DEFAULT_WEIGHTS["cv"])


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown objective mode"):
        staged_weights("square", 0, 9)


def test_staged_weights_can_build_on_an_explicit_base():
    base = dict(DEFAULT_WEIGHTS["shape"], energy=10.0)
    assert staged_weights("shape", 0, 9, base=base)["energy"] == pytest.approx(
        10.0*PHASE_SCALE["energy"]["energy"])


# --- ShapeObjective wiring --------------------------------------------------

def test_set_weights_rejects_unknown_keys():
    objective = ShapeObjective(_target())
    with pytest.raises(ValueError, match="unknown weight"):
        objective.set_weights({"cv2 ": 1.0})


def test_set_weights_fills_the_rest_from_defaults():
    objective = ShapeObjective(_target())
    objective.set_weights({"energy": 5.0})
    assert objective.weights["energy"] == pytest.approx(5.0)
    assert objective.weights["cv2"] == pytest.approx(DEFAULT_WEIGHTS["shape"]["cv2"])


@pytest.mark.parametrize("kwargs, mode", [
    ({"cv_only": True}, "cv"),
    ({}, "shape"),
    ({"formation": True}, "formation"),
])
def test_default_weights_reproduce_the_historical_loss(kwargs, mode):
    """The named dict must be byte-identical in effect to the inline constants.

    Historical expressions: ``cv2 + .001*smooth`` (cv), ``cv2 + 2*(1-energy) +
    shape_loss + .001*smooth`` (shape), and the formation sum.
    """
    import torch

    objective = ShapeObjective(_target(), **kwargs)
    assert objective.mode == mode
    rng = np.random.default_rng(7)
    image = np.abs(rng.normal(size=(32, 32))).astype(np.float32)
    delta = np.zeros((32, 32), dtype=np.float32)

    loss, parts = objective.evaluate(torch.as_tensor(image), torch.as_tensor(delta))
    w = DEFAULT_WEIGHTS[mode]
    if mode == "cv":
        expected = parts["l_cv"] + .001*0.0
    elif mode == "shape":
        expected = (parts["l_cv"] + 2*(1 - parts["ee_roi"]) + parts["shape_loss"])
    else:
        expected = (.05*parts["l_cv"] + 8*(1 - parts["ee_roi"]) + parts["shape_loss"]
                    + .5*parts["l_partition"] + .5*parts["l_peripheral"]
                    + 1.0*parts["l_floor"] + .3*parts["l_boundary"])
    assert float(loss) == pytest.approx(expected, rel=1e-5)
    assert w["smooth"] == pytest.approx(.001)


def test_staged_weights_change_the_loss_of_the_same_frame():
    """A schedule that cannot move the loss would be a silent no-op."""
    import torch

    rng = np.random.default_rng(11)
    image = np.abs(rng.normal(size=(32, 32))).astype(np.float32)
    delta = np.zeros((32, 32), dtype=np.float32)
    tensor = torch.as_tensor(image)

    objective = ShapeObjective(_target())
    constant, _ = objective.evaluate(tensor, torch.as_tensor(delta))
    objective.set_weights(staged_weights("shape", 0, 9))
    staged, _ = objective.evaluate(tensor, torch.as_tensor(delta))
    assert float(staged) != pytest.approx(float(constant), rel=1e-6)
