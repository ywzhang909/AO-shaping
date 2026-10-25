"""Tests for the teacher: it must actually solve the shaping problem.

Two things are checked beyond "the loss went down":

* the reported metrics are **bit-identical** to
  :func:`ml.sim2f.losses.objective_terms` recomputed on the returned frame. The
  teacher used to restate that formula locally and the two drifted (the teacher
  scored raw frames, the trainer prepared ones), which made the report's
  amortisation gap partly an artefact of two different objectives;
* the encircled energy can only *improve* when the flat spot misses the target.
  On this bench the flat spot is compact and centred, so for a centred target its
  energy is already ~99% inside the support and shaping can only lower it (it
  fills the target). The EE term is therefore exercised with an **off-centre**
  target, which is also the case the closed loop cares about.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.sim2f.config import Sim2fConfig, TeacherConfig
from ml.sim2f.losses import objective_terms
from ml.sim2f.optics import Sim2f
from ml.sim2f.targets import draw_target
from ml.sim2f.teacher import Teacher, TeacherResult

#: 64-sample pupil, 4.4 um camera pixel: the *same* 11.2 camera pixels per
#: diffraction scale as the reference bench, so the shaping problem has the
#: difficulty the real bench has (with a 2.2 um pixel and this small grid the
#: target would be ~1 PSF wide and the objective could not fall below 60%).
SHAPE = Sim2fConfig(grid=64, region_span_px=900.0, ccd_px=64, ccd_pixel_um=4.4)
#: Enough iterations for the reference difficulty; the shipped default is 800.
ITERS = 200


def _flat_losses(
    sim: Sim2f, target: torch.Tensor, support: torch.Tensor, cfg: TeacherConfig
) -> tuple[float, float, float, float]:
    """Score the flat-phase frame with the teacher's own objective.

    Args:
        sim: Bench.
        target: ``(1, ccd_px, ccd_px)`` target.
        support: Matching support.
        cfg: Objective contract (weights, metric, ``prepare``).

    Returns:
        ``(loss, l1, ee, ncc)`` for the flat phase.
    """
    flat = sim.forward(np.zeros((1, sim.grid, sim.grid), dtype=np.float32))
    loss, l1, ee, ncc = objective_terms(
        flat,
        target,
        support,
        w_ee=cfg.w_ee,
        metric=cfg.metric,
        prepare=cfg.prepare,
    )
    return float(loss), float(l1), float(ee), float(ncc)


@pytest.mark.parametrize("kind", ["square", "disc"])
def test_objective_beats_the_flat_phase(kind: str) -> None:
    """The label is much better than doing nothing (measured ratios 0.31 / 0.34)."""
    sim = Sim2f(SHAPE, device="cpu")
    cfg = TeacherConfig(iters=ITERS)
    sample = draw_target(kind, SHAPE.ccd_px, np.random.default_rng(7))
    target = torch.as_tensor(sample.intensity)[None]
    support = torch.as_tensor(sample.support)[None].float()

    flat_loss, flat_l1, flat_ee, flat_ncc = _flat_losses(sim, target, support, cfg)
    result = Teacher(sim, cfg).optimise(target, support=support)

    assert result.loss < 0.6 * flat_loss
    assert result.l1 < flat_l1
    assert result.ncc > flat_ncc
    # A teacher that found the target should be structurally similar to it.
    assert result.ncc > 0.9


def test_encircled_energy_improves_when_the_flat_spot_misses_the_target() -> None:
    """An off-centre target: the flat spot starts outside it, the label does not."""
    sim = Sim2f(SHAPE, device="cpu")
    cfg = TeacherConfig(iters=ITERS)
    sample = draw_target(
        "disc",
        SHAPE.ccd_px,
        np.random.default_rng(3),
        size_frac=(0.2, 0.3),
        center_px=(10.0, 0.0),
    )
    target = torch.as_tensor(sample.intensity)[None]
    support = torch.as_tensor(sample.support)[None].float()

    _, _, flat_ee, _ = _flat_losses(sim, target, support, cfg)
    result = Teacher(sim, cfg).optimise(target, support=support)

    assert flat_ee < 0.8, "the flat spot must start mostly outside this target"
    assert result.ee > flat_ee
    assert result.ee > 0.7


def test_reported_metrics_match_the_shared_objective() -> None:
    """``teacher_metrics`` is reproducible from ``objective_terms`` alone.

    This is the regression test for the duplicated-objective bug: the teacher's
    numbers must be the ones the trainer and evaluator recompute, not a second
    formula that happens to be close.
    """
    sim = Sim2f(SHAPE, device="cpu")
    cfg = TeacherConfig(iters=20)
    sample = draw_target("square", SHAPE.ccd_px, np.random.default_rng(11))
    target = torch.as_tensor(sample.intensity)[None]
    support = torch.as_tensor(sample.support)[None].float()

    result = Teacher(sim, cfg).optimise(target, support=support)

    frame = torch.as_tensor(result.intensity)[None, None]
    loss, l1, ee, ncc = objective_terms(
        frame, target, support, w_ee=cfg.w_ee, metric=cfg.metric, prepare=cfg.prepare
    )
    assert result.loss == pytest.approx(float(loss), rel=1e-6)
    assert result.l1 == pytest.approx(float(l1), rel=1e-6)
    assert result.ee == pytest.approx(float(ee), rel=1e-6)
    assert result.ncc == pytest.approx(float(ncc), rel=1e-6)


def test_result_arrays_have_the_documented_shape_and_dtype() -> None:
    """The label is float32 on the pupil grid; the frame is float32 on the camera grid."""
    sim = Sim2f(SHAPE, device="cpu")
    cfg = TeacherConfig(iters=10)
    sample = draw_target("disc", SHAPE.ccd_px, np.random.default_rng(2))
    result = Teacher(sim, cfg).optimise(
        torch.as_tensor(sample.intensity)[None],
        support=torch.as_tensor(sample.support)[None],
    )
    assert isinstance(result, TeacherResult)
    assert result.phase.shape == (SHAPE.grid, SHAPE.grid)
    assert result.phase.dtype == np.float32
    assert result.intensity.shape == (SHAPE.ccd_px, SHAPE.ccd_px)
    assert result.intensity.dtype == np.float32
    assert np.isfinite(result.phase).all()
    assert np.isfinite(result.intensity).all()
    # The phase is raw radians and deliberately *not* wrapped into [0, 2*pi).
    assert float(np.abs(result.phase).max()) > 0.0
    assert len(result.history) == cfg.iters
    assert result.iters == cfg.iters
    assert result.wall_s > 0.0


def test_history_starts_high_and_ends_low() -> None:
    """The objective decreases overall, and ``history[-1]`` is the label's loss."""
    sim = Sim2f(SHAPE, device="cpu")
    cfg = TeacherConfig(iters=30)
    sample = draw_target("square", SHAPE.ccd_px, np.random.default_rng(4))
    result = Teacher(sim, cfg).optimise(
        torch.as_tensor(sample.intensity)[None],
        support=torch.as_tensor(sample.support)[None],
    )
    assert len(result.history) == cfg.iters
    assert result.history[-1] <= result.history[0]
    assert result.history[-1] == pytest.approx(result.loss, rel=1e-6)
    # Adam on a smooth objective should not wander: the final quarter of the
    # history must be below the first quarter.
    quarter = max(1, cfg.iters // 4)
    assert min(result.history[-quarter:]) < result.history[0]


def test_teacher_is_deterministic() -> None:
    """No stochastic step: the same call returns bit-identical arrays."""
    sim = Sim2f(SHAPE, device="cpu")
    cfg = TeacherConfig(iters=15)
    sample = draw_target("rect", SHAPE.ccd_px, np.random.default_rng(6))
    target = torch.as_tensor(sample.intensity)[None]
    support = torch.as_tensor(sample.support)[None]
    first = Teacher(sim, cfg).optimise(target, support=support, seed=123)
    second = Teacher(sim, cfg).optimise(target, support=support, seed=123)
    assert np.array_equal(first.phase, second.phase)
    assert np.array_equal(first.intensity, second.intensity)
    assert first.history == second.history


def test_bad_targets_are_rejected() -> None:
    """Wrong grid, wrong rank, or an empty target fail loudly."""
    sim = Sim2f(SHAPE, device="cpu")
    cfg = TeacherConfig(iters=2)
    teacher = Teacher(sim, cfg)
    with pytest.raises(ValueError):
        teacher.optimise(np.zeros((SHAPE.ccd_px + 2, SHAPE.ccd_px)))
    with pytest.raises(ValueError):
        teacher.optimise(np.zeros((1, 1, 1, SHAPE.ccd_px, SHAPE.ccd_px)))
    with pytest.raises(ValueError):
        teacher.optimise(np.zeros((SHAPE.ccd_px, SHAPE.ccd_px)))


def test_support_defaults_to_the_whole_frame() -> None:
    """Without a support, ``EE`` is 1 and the objective is the shape term alone."""
    sim = Sim2f(SHAPE, device="cpu")
    cfg = TeacherConfig(iters=5)
    sample = draw_target("gaussian", SHAPE.ccd_px, np.random.default_rng(8))
    result = Teacher(sim, cfg).optimise(torch.as_tensor(sample.intensity)[None])
    assert result.ee == pytest.approx(1.0, rel=1e-6)
