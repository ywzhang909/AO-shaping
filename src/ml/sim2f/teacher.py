"""Differentiable phase optimisation on the ideal bench -- the dataset's labels.

Why the labels come from an optimiser, not from a closed form
=============================================================
``target -> phase`` is a phase-retrieval problem: many phases produce the same
target, and no closed form exists.  The teacher solves it once per target, slowly
and well, and the U-Net is then trained to reproduce that answer in one forward
pass.  The report's headline number is exactly the gap between the two (how much
of the iterative optimum a single pass recovers).

Why the objective has two parts
===============================
``L_shape`` compares the *normalised distributions* (where the energy goes) and
``w_ee * (1 - EE)`` rewards keeping the energy inside the target support.  Either
one alone is degenerate, and this repository measured both failures:

* ``L_shape`` alone is scale-free -- it happily empties the target, because an
  all-zero frame has a small L1 distance once both sides are unit-normalised;
* the encircled energy alone is maximised by a delta function, which is what the
  ``-CV``-only square-shaping run did on hardware (EE collapsed to 0.002).

Why the teacher runs on the aberration-free bench
=================================================
The aberration is handled downstream by the Zernike probe fit and the closed-loop
refit (:mod:`ml.sim2f.fit`), which is what lets *one* network serve every bench
state.  A teacher that had to model the aberration would make the labels depend
on the bench, and the U-Net would have to be retrained for every optical path.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

from ml.sim2f.config import TeacherConfig
from ml.sim2f.losses import objective_terms
from ml.sim2f.optics import Sim2f

__all__ = ["Teacher", "TeacherResult"]


@dataclass(frozen=True)
class TeacherResult:
    """One teacher optimum: the label and the evidence it is a good one.

    Attributes:
        phase: ``(grid, grid)`` float32 **raw unwrapped radians**.
        intensity: ``(ccd_px, ccd_px)`` float32 model frame at the solution (the
            ideal-bench forward pass of ``phase``, not the target).
        loss: Final objective, ``l1 + w_ee * (1 - ee) + w_tv * tv``.
        l1: Final ``L_shape`` alone.
        ee: Final encircled energy -- the fraction of predicted energy inside the
            target support.
        ncc: Normalised cross-correlation of ``intensity`` and ``target``, both
            mean-subtracted. Reported because L1 is dominated by the residual
            energy outside the support while NCC tracks the *structure*.
        history: Objective per iteration; the last entry is re-evaluated at the
            final phase, so ``history[-1]`` is the loss of the returned label.
        iters: Optimiser steps taken.
        wall_s: Wall clock of :meth:`Teacher.optimise`.
    """

    phase: np.ndarray
    intensity: np.ndarray
    loss: float
    l1: float
    ee: float
    ncc: float
    history: list[float]
    iters: int
    wall_s: float


class Teacher:
    """Adam on a free per-pixel phase, against one target distribution.

    The optimiser is deliberately plain: no momentum restarts, no line search, no
    multi-start.  The point of the teacher is to be *the* reference optimum that a
    single network forward pass is compared against, so it must be simple enough
    to be reproducible and the same for every target.

    Attributes:
        sim: The bench every forward pass goes through.  The teacher never builds
            its own propagator (rule 1).
        cfg: Objective and budget contract.
    """

    def __init__(self, sim: Sim2f, cfg: TeacherConfig) -> None:
        """Store the bench and the budget.

        Args:
            sim: Bench to optimise against. Callers that want the ideal labels
                pass ``Sim2f(Sim2fConfig())``; a calibrated bench is also valid,
                in which case the labels carry the bench's optical errors.
            cfg: Objective weights, iteration budget and learning-rate schedule.
        """
        self.sim: Sim2f = sim
        self.cfg: TeacherConfig = cfg

    def optimise(
        self,
        target: torch.Tensor | np.ndarray,
        *,
        support: torch.Tensor | np.ndarray | None = None,
        seed: int | None = None,
    ) -> TeacherResult:
        """Optimise a phase that shapes the flat spot into ``target``.

        Args:
            target: ``(ccd_px, ccd_px)``, ``(1, ccd_px, ccd_px)`` or
                ``(1, 1, ccd_px, ccd_px)`` non-negative target. Normalised
                internally to unit sum.
            support: Optional binary support, same spatial shape as ``target``.
                Defaults to the whole frame, in which case ``EE`` is trivially 1
                and the objective reduces to ``L_shape``.
            seed: Accepted for interface symmetry and logged, but unused: there
                is no stochastic step (the initial phase is flat and Adam is
                deterministic), so two calls with the same arguments return
                bit-identical results.

        Returns:
            The :class:`TeacherResult` for this target.

        Raises:
            ValueError: If ``target`` / ``support`` do not match the camera grid,
                or ``target`` has no energy.
        """
        started = time.perf_counter()
        grid = self.sim.grid
        cfg = self.cfg

        wanted = self._as_image(target, "target")
        if float(wanted.sum()) <= 0.0:
            raise ValueError("target has no energy (sum is zero)")
        weights = (
            torch.ones_like(wanted)
            if support is None
            else self._as_image(support, "support").clamp(0.0, 1.0)
        )

        phase = torch.zeros(grid, grid, dtype=self.sim.dtype, device=self.sim.device)
        phase.requires_grad_(True)
        optimiser = torch.optim.Adam([phase], lr=float(cfg.lr))

        history: list[float] = []
        initial_loss = math.nan
        for iteration in range(int(cfg.iters)):
            progress = iteration / max(1, int(cfg.iters) - 1)
            floor = float(cfg.lr_final_frac)
            lr = float(cfg.lr) * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
            for group in optimiser.param_groups:
                group["lr"] = lr

            loss, _, _ = self._objective(phase, wanted, weights)
            history.append(float(loss.detach()))
            if iteration == 0:
                initial_loss = history[0]
            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

        # Re-evaluate at the *returned* phase so history[-1] and the reported
        # metrics describe the label, not the second-to-last iterate.
        loss, parts, ncc = self._objective(phase, wanted, weights)
        history[-1] = float(loss.detach())

        with torch.no_grad():
            frame = self.sim.forward(phase.unsqueeze(0), detector=False)[0, 0]

        wall = time.perf_counter() - started
        if history[-1] > initial_loss:
            logger.warning(
                "teacher loss rose: {:.4f} -> {:.4f} (iters={}); the returned phase "
                "is the last iterate, not the best one",
                initial_loss,
                history[-1],
                cfg.iters,
            )
        logger.debug(
            "teacher done: loss={:.4f} l1={:.4f} ee={:.4f} ncc={:.4f} iters={} "
            "wall={:.2f}s seed={}",
            history[-1],
            parts["l1"],
            parts["ee"],
            ncc,
            cfg.iters,
            wall,
            seed,
        )
        return TeacherResult(
            phase=phase.detach().to("cpu").numpy().astype(np.float32),
            intensity=frame.detach().to("cpu").numpy().astype(np.float32),
            loss=float(history[-1]),
            l1=float(parts["l1"]),
            ee=float(parts["ee"]),
            ncc=float(ncc),
            history=history,
            iters=int(cfg.iters),
            wall_s=float(wall),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _objective(
        self,
        phase: torch.Tensor,
        wanted: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], float]:
        """Compute the objective at ``phase``, through ``losses.objective_terms``.

        There is exactly **one** implementation of ``L_shape`` / ``EE`` / ``NCC``
        in this package (:func:`ml.sim2f.losses.objective_terms`), shared with the
        physics loss, the trainer and the evaluator.  An earlier version of the
        teacher restated the formula here, and the two drifted: the teacher's copy
        scored raw frames while the trainer scored *prepared* ones, which put the
        teacher's ``teacher_metrics`` 0.7% (ideal) to 2.8% (``cross``/``rect``)
        below what the evaluator recomputed for the very same phase.

        Args:
            phase: ``(grid, grid)`` free parameter (differentiable).
            wanted: ``(1, 1, ccd_px, ccd_px)`` target (normalised inside).
            weights: ``(1, 1, ccd_px, ccd_px)`` support weights.

        Returns:
            ``(loss, parts, ncc)`` where ``parts`` holds the detached ``l1`` /
            ``ee`` scalars and ``ncc`` is the cross-correlation of the same pair
            of normalised frames.
        """
        cfg = self.cfg
        frame = self.sim.forward(phase.unsqueeze(0), detector=False)
        loss, l1, ee, ncc = objective_terms(
            frame,
            wanted,
            weights,
            w_ee=cfg.w_ee,
            metric=cfg.metric,
            prepare=cfg.prepare,
        )
        if cfg.w_tv > 0.0:
            tv = (phase[:, 1:] - phase[:, :-1]).abs().sum() + (
                phase[1:, :] - phase[:-1, :]
            ).abs().sum()
            loss = loss + float(cfg.w_tv) * tv
        return loss, {"l1": float(l1.detach()), "ee": float(ee.detach())}, float(
            ncc.detach()
        )

    def _as_image(
        self, value: torch.Tensor | np.ndarray, name: str
    ) -> torch.Tensor:
        """Coerce an image argument to ``(1, 1, ccd_px, ccd_px)``.

        Args:
            value: ``(ccd, ccd)``, ``(1, ccd, ccd)``, ``(1, 1, ccd, ccd)`` array
                or tensor.
            name: Argument name, for the error message.

        Returns:
            ``(1, 1, ccd_px, ccd_px)`` tensor in the bench dtype/device.

        Raises:
            ValueError: If the spatial shape is not the camera grid.
        """
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.dim() == 2:
            tensor = tensor.view(1, 1, *tensor.shape)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.dim() != 4 or tensor.shape[1] != 1:
            raise ValueError(
                f"{name} must be (ccd_px, ccd_px) or (1, 1, ccd_px, ccd_px), got "
                f"{tuple(tensor.shape)}"
            )
        if tensor.shape[-2:] != (self.sim.ccd_px, self.sim.ccd_px):
            raise ValueError(
                f"{name} spatial size {tuple(tensor.shape[-2:])} != camera grid "
                f"{(self.sim.ccd_px, self.sim.ccd_px)}"
            )
        return tensor.to(device=self.sim.device, dtype=self.sim.dtype)
