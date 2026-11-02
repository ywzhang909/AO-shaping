"""Differentiable **local** fine-tuning of an SLM phase on a fitted bench.

What this stage is
==================
:mod:`ml.sim2f.teacher` optimises a free per-pixel phase **from scratch**, on the
ideal bench, and is what labels the corpus.  The closed loop needs something
else: a *bounded* update of a phase that already exists -- a network forward
pass, a teacher label, or last round's accepted phase -- against the **fitted**
bench, i.e. the bench carrying the aberration the probe/refit stages estimated
(:mod:`ml.sim2f.fit`).

The three differences are the point:

1. **It starts from a phase.**  The refined phase is ``phi_init + delta``, and
   only ``delta`` is optimised, so the starting phase's global structure (which
   is what the deep model or the teacher paid for) survives.
2. **The update is bounded.**  ``RefineConfig.max_delta_rad`` clamps ``delta``
   and ``RefineConfig.update_sigma_px`` blurs it, because the gradient is only
   trusted as far as the forward model is: on a mis-specified model an
   unbounded per-pixel Adam step lowers the *predicted* objective by exploiting
   model error and raises the measured one.  That is exactly the failure the
   hardware session recorded (``runs/sim2f_bench/session_hw/session.json``:
   ``ee_net 0.249 < ee_flat 0.352``, ``wins: 0``).
3. **It is accountable.**  The update is accepted only if the objective
   improves by ``RefineConfig.min_gain``; otherwise :meth:`PhaseRefiner.finetune`
   returns the input phase unchanged with ``accepted=False``, so a caller can
   log "the model could not improve this shape" instead of shipping a worse
   phase to the device.

What it is not
==============
It is not a second optimiser for the labels.  The objective is
:func:`ml.sim2f.losses.objective_terms` -- the *same* function the teacher, the
physics loss and the evaluator call -- and every forward pass goes through
:class:`ml.sim2f.optics.Sim2f` (package rule 1: one FFT).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

from ml.sim2f.config import RefineConfig
from ml.sim2f.losses import objective_terms
from ml.sim2f.optics import Sim2f

__all__ = ["PhaseRefiner", "RefineResult", "finetune_phase"]


@dataclass(frozen=True)
class RefineResult:
    """Outcome of one fine-tuning call.

    Attributes:
        phase: Refined phase, ``(grid, grid)`` radians (float32, CPU).  Equal to
            :attr:`phase_init` when :attr:`accepted` is ``False``.
        phase_init: The phase the call started from, same shape/dtype.
        intensity: Predicted camera frame at :attr:`phase``,
            ``(ccd_px, ccd_px)`` float32 (pure intensity, no detector).
        objective: Objective at :attr:`phase` (lower is better).
        objective_init: Objective at :attr:`phase_init`.
        l1: Sum-L1 between the predicted and target distributions at
            :attr:`phase`.
        l1_init: The same at :attr:`phase_init`.
        ee: Encircled energy inside the target support at :attr:`phase`.
        ee_init: The same at :attr:`phase_init`.
        ncc: Normalised cross-correlation at :attr:`phase`.
        ncc_init: The same at :attr:`phase_init`.
        accepted: Whether the refined phase beat the initial one by
            ``RefineConfig.min_gain``.
        gain: Relative objective improvement, ``(objective_init - objective) /
            |objective_init|`` -- negative when the update made things worse.
        delta_rms_rad: RMS of the accepted update, radians.
        delta_max_rad: Largest absolute update, radians (never above
            ``max_delta_rad`` by construction).
        iters: Adam steps run.
        history: Objective per step.
        wall_s: Wall clock seconds.
    """

    phase: np.ndarray
    phase_init: np.ndarray
    intensity: np.ndarray
    objective: float
    objective_init: float
    l1: float
    l1_init: float
    ee: float
    ee_init: float
    ncc: float
    ncc_init: float
    accepted: bool
    gain: float
    delta_rms_rad: float
    delta_max_rad: float
    iters: int
    history: list[float]
    wall_s: float

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable summary (arrays dropped; the caller saves those)."""
        return {
            "objective": float(self.objective),
            "objective_init": float(self.objective_init),
            "l1": float(self.l1),
            "l1_init": float(self.l1_init),
            "ee": float(self.ee),
            "ee_init": float(self.ee_init),
            "ncc": float(self.ncc),
            "ncc_init": float(self.ncc_init),
            "accepted": bool(self.accepted),
            "gain": float(self.gain),
            "delta_rms_rad": float(self.delta_rms_rad),
            "delta_max_rad": float(self.delta_max_rad),
            "iters": int(self.iters),
            "wall_s": float(self.wall_s),
        }


def _as_phase(value: torch.Tensor | np.ndarray, sim: Sim2f, name: str) -> torch.Tensor:
    """Coerce a phase argument to ``(grid, grid)`` on the bench.

    Args:
        value: ``(grid, grid)`` or ``(1, grid, grid)`` radians.
        sim: Bench, for the grid size, dtype and device.
        name: Argument name, for the error message.

    Returns:
        ``(grid, grid)`` tensor in the bench dtype/device.

    Raises:
        ValueError: If the spatial shape is not the pupil grid, or the batch
            dimension is larger than one (this stage refines one phase).
    """
    tensor = torch.as_tensor(np.asarray(value) if isinstance(value, np.ndarray) else value)
    tensor = tensor.to(device=sim.device, dtype=sim.dtype)
    if tensor.ndim == 3:
        if int(tensor.shape[0]) != 1:
            raise ValueError(
                f"{name}: fine-tuning refines one phase at a time, got batch {tensor.shape[0]}"
            )
        tensor = tensor[0]
    if tensor.ndim != 2 or tuple(tensor.shape) != (sim.grid, sim.grid):
        raise ValueError(
            f"{name}: expected ({sim.grid}, {sim.grid}), got {tuple(tensor.shape)}"
        )
    return tensor


def _as_image(value: torch.Tensor | np.ndarray, sim: Sim2f, name: str) -> torch.Tensor:
    """Coerce an image argument to ``(1, 1, ccd_px, ccd_px)`` on the bench.

    Args:
        value: ``(ccd, ccd)``, ``(1, ccd, ccd)`` or ``(1, 1, ccd, ccd)`` array.
        sim: Bench, for the camera size, dtype and device.
        name: Argument name, for the error message.

    Returns:
        ``(1, 1, ccd_px, ccd_px)`` tensor in the bench dtype/device.

    Raises:
        ValueError: If the spatial shape is not the camera grid.
    """
    tensor = torch.as_tensor(np.asarray(value) if isinstance(value, np.ndarray) else value)
    tensor = tensor.to(device=sim.device, dtype=sim.dtype)
    while tensor.ndim < 4:
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape[-2:]) != (sim.ccd_px, sim.ccd_px):
        raise ValueError(
            f"{name}: expected ({sim.ccd_px}, {sim.ccd_px}), got {tuple(tensor.shape[-2:])}"
        )
    if tensor.shape[0] != 1 or tensor.shape[1] != 1:
        raise ValueError(f"{name}: expected a single frame, got {tuple(tensor.shape)}")
    return tensor


def _blur_kernel(sigma_px: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Separable Gaussian kernel for a pupil-plane update blur.

    Args:
        sigma_px: Sigma in pupil pixels.  Callers must pass ``> 0``.
        dtype: Kernel dtype.
        device: Kernel device.

    Returns:
        ``(1, 1, k, 1)`` and ``(1, 1, 1, k)``-usable 1-D kernel of odd length
        ``k = 2 * ceil(3 * sigma) + 1``, summing to one.
    """
    radius = int(math.ceil(3.0 * float(sigma_px)))
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(coords**2) / (2.0 * float(sigma_px) ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1, 1)


class PhaseRefiner:
    """Bounded Adam on a phase **update**, against one target on a fitted bench.

    The optimiser is deliberately the plainest thing that can carry a gradient
    through :class:`~ml.sim2f.optics.Sim2f`: no momentum restarts, no line
    search, no multi-start.  What makes it *fine*-tuning rather than a second
    teacher is the parameterisation (an update with a trust region, not a free
    phase) and the acceptance test, both of which live in :class:`RefineConfig`.

    Attributes:
        sim: The bench every forward pass goes through.  Callers pass a bench
            carrying the fitted aberration -- the refiner never fits anything.
        cfg: Budget, objective weights and trust region.
    """

    def __init__(self, sim: Sim2f, cfg: RefineConfig | None = None) -> None:
        """Store the bench and the contract.

        Args:
            sim: Bench to optimise against (usually the fitted/calibrated one).
            cfg: Objective weights, budget and trust region.  Defaults to
                :class:`RefineConfig`.
        """
        self.sim: Sim2f = sim
        self.cfg: RefineConfig = cfg if cfg is not None else RefineConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def finetune(
        self,
        phase_init: torch.Tensor | np.ndarray,
        target: torch.Tensor | np.ndarray,
        *,
        support: torch.Tensor | np.ndarray | None = None,
        aberration: torch.Tensor | np.ndarray | None = None,
        illumination: torch.Tensor | np.ndarray | None = None,
    ) -> RefineResult:
        """Refine ``phase_init`` into ``target`` on this bench.

        Args:
            phase_init: ``(grid, grid)`` radians -- a network output, a teacher
                label, or last round's accepted phase.
            target: ``(ccd_px, ccd_px)`` non-negative target; normalised
                internally to unit sum.
            support: Optional binary support, same spatial shape as ``target``.
                Defaults to the whole frame, in which case ``EE`` is trivially 1.
            aberration: Optional ``(grid, grid)`` aberration phase the bench adds
                (the probe/refit estimate).  Pass ``None`` to refine on the
                bench as configured.
            illumination: Optional ``(grid, grid)`` entrance-beam amplitude
                override (a calibrated beam).

        Returns:
            The :class:`RefineResult`.  ``accepted=False`` means the update did
            not pay for itself and :attr:`RefineResult.phase` is the input.

        Raises:
            ValueError: If ``target`` / ``support`` / ``phase_init`` do not match
                the bench grids, or ``target`` has no energy.
        """
        started = time.perf_counter()
        cfg = self.cfg
        sim = self.sim

        start = _as_phase(phase_init, sim, "phase_init")
        wanted = _as_image(target, sim, "target")
        if float(wanted.sum()) <= 0.0:
            raise ValueError("target has no energy (sum is zero)")
        weights = (
            torch.ones_like(wanted)
            if support is None
            else _as_image(support, sim, "support").clamp(0.0, 1.0)
        )
        sigma = float(cfg.update_sigma_px)
        kernel = _blur_kernel(sigma, sim.dtype, sim.device) if sigma > 0.0 else None

        delta = torch.zeros_like(start, requires_grad=True)
        optimiser = torch.optim.Adam([delta], lr=float(cfg.lr))

        # -- objective at the starting phase (the acceptance baseline) -------
        # Evaluated through ``_forward_terms``, i.e. with the same composed
        # objective the loop minimises (TV included): the acceptance test and the
        # reported numbers must describe the same objective the optimiser saw.
        with torch.no_grad():
            base_frame, base_parts = self._forward_terms(
                start, wanted, weights, aberration, illumination
            )
        objective_init = float(base_parts[0])
        base_l1, base_ee, base_ncc = (
            float(base_parts[1]),
            float(base_parts[2]),
            float(base_parts[3]),
        )

        history: list[float] = []
        best_objective = objective_init
        best_delta = torch.zeros_like(start)
        best_frame = base_frame[0, 0]
        best_parts = (base_l1, base_ee, base_ncc)

        for iteration in range(int(cfg.iters)):
            progress = iteration / max(1, int(cfg.iters) - 1)
            floor = float(cfg.lr_final_frac)
            lr = float(cfg.lr) * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
            for group in optimiser.param_groups:
                group["lr"] = lr

            update = self._trust_region(delta, kernel)
            loss, l1, ee, ncc = self._terms(
                start + update, wanted, weights, aberration, illumination
            )
            value = float(loss.detach())
            history.append(value)
            if cfg.keep_best and value < best_objective:
                best_objective = value
                best_delta = update.detach().clone()
                best_parts = (float(l1.detach()), float(ee.detach()), float(ncc.detach()))
            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

        # Re-evaluate at the *returned* phase: with keep_best the returned
        # update is an earlier iterate, so history[-1] must not describe it.
        with torch.no_grad():
            if not cfg.keep_best:
                best_delta = self._trust_region(delta, kernel).detach().clone()
            refined = start + best_delta
            frame, parts = self._forward_terms(
                refined, wanted, weights, aberration, illumination
            )
            best_objective = float(parts[0])
            best_parts = (float(parts[1]), float(parts[2]), float(parts[3]))
            best_frame = frame[0, 0]

        gain = (objective_init - best_objective) / max(abs(objective_init), 1e-12)
        accepted = bool(gain >= float(cfg.min_gain))

        if accepted:
            delta_flat = best_delta.detach()
            phase_out = start + delta_flat
        else:
            # A rejected update must also reject the numbers that came with it:
            # the frame and the metrics have to describe the phase actually
            # returned, otherwise a caller tabulating this round scores the
            # device against a candidate that was never applied.
            with torch.no_grad():
                frame, parts = self._forward_terms(
                    start, wanted, weights, aberration, illumination
                )
            best_objective, best_frame = float(parts[0]), frame[0, 0]
            best_parts = (float(parts[1]), float(parts[2]), float(parts[3]))
            delta_flat = torch.zeros_like(start)
            phase_out = start
            logger.info(
                "refine rejected: objective {:.4f} -> {:.4f} (gain {:.4f} < min_gain {:.4f}); "
                "returning the initial phase",
                objective_init,
                best_objective,
                gain,
                float(cfg.min_gain),
            )

        wall = time.perf_counter() - started
        logger.debug(
            "refine done: objective {:.4f} -> {:.4f} l1 {:.4f} -> {:.4f} ee {:.4f} -> {:.4f} "
            "delta_rms={:.4f} rad iters={} wall={:.2f}s accepted={}",
            objective_init,
            best_objective,
            float(base_l1),
            best_parts[0],
            float(base_ee),
            best_parts[1],
            float(delta_flat.pow(2).mean().sqrt()) if accepted else 0.0,
            cfg.iters,
            wall,
            accepted,
        )

        start_np = start.detach().to("cpu").numpy().astype(np.float32)
        return RefineResult(
            phase=phase_out.detach().to("cpu").numpy().astype(np.float32),
            phase_init=start_np,
            intensity=best_frame.detach().to("cpu").numpy().astype(np.float32),
            objective=float(best_objective),
            objective_init=objective_init,
            l1=float(best_parts[0]),
            l1_init=float(base_l1),
            ee=float(best_parts[1]),
            ee_init=float(base_ee),
            ncc=float(best_parts[2]),
            ncc_init=float(base_ncc),
            accepted=accepted,
            gain=float(gain),
            delta_rms_rad=float(delta_flat.pow(2).mean().sqrt()),
            delta_max_rad=float(delta_flat.abs().max()),
            iters=int(cfg.iters),
            history=history,
            wall_s=float(wall),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _trust_region(
        self, delta: torch.Tensor, kernel: torch.Tensor | None
    ) -> torch.Tensor:
        """Map the free parameter to the bounded, optionally smoothed update.

        Args:
            delta: Free ``(grid, grid)`` parameter (differentiable).
            kernel: Separable Gaussian kernel, or ``None`` for no blur.

        Returns:
            The update to add to the initial phase, same shape as ``delta``.
        """
        cfg = self.cfg
        update = delta
        if kernel is not None:
            radius_px = kernel.numel() // 2
            vertical = torch.nn.functional.pad(
                update[None, None], (0, 0, radius_px, radius_px), mode="replicate"
            )
            update = torch.nn.functional.conv2d(vertical, kernel)[0, 0]
            horizontal = torch.nn.functional.pad(
                update[None, None], (radius_px, radius_px, 0, 0), mode="replicate"
            )
            update = torch.nn.functional.conv2d(horizontal, kernel.transpose(-1, -2))[0, 0]
        radius = float(cfg.max_delta_rad)
        if str(cfg.trust_region) == "tanh":
            return radius * torch.tanh(update / radius)
        return update.clamp(-radius, radius)

    def _terms(
        self,
        phase: torch.Tensor,
        wanted: torch.Tensor,
        weights: torch.Tensor,
        aberration: torch.Tensor | np.ndarray | None,
        illumination: torch.Tensor | np.ndarray | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Objective at ``phase``, through :func:`~ml.sim2f.losses.objective_terms`.

        Args:
            phase: ``(grid, grid)`` phase, differentiable.
            wanted: ``(1, 1, ccd_px, ccd_px)`` target.
            weights: ``(1, 1, ccd_px, ccd_px)`` support weights.
            aberration: Optional ``(grid, grid)`` aberration phase for the bench.
            illumination: Optional ``(grid, grid)`` amplitude override.

        Returns:
            ``(loss, l1, ee, ncc)`` -- the composed objective plus the three
            diagnostics, all differentiable.
        """
        cfg = self.cfg
        frame = self.sim.forward(
            phase.unsqueeze(0),
            aberration=aberration,
            illumination=illumination,
            detector=False,
        )
        loss, l1, ee, ncc = objective_terms(
            frame,
            wanted,
            weights,
            w_ee=cfg.w_ee,
            metric=cfg.metric,
            prepare=cfg.prepare,
        )
        return self._compose(loss, phase), l1, ee, ncc

    def _compose(self, loss: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Add the phasor-TV penalty to the shaping objective.

        One implementation on purpose: the optimiser, the acceptance test and the
        reported numbers must all compare *the same* objective, and a TV weight
        that only existed inside the training loop would make the acceptance test
        score an update against a target it never minimised.

        Args:
            loss: Shaping objective from
                :func:`~ml.sim2f.losses.objective_terms`.
            phase: ``(grid, grid)`` phase the loss was evaluated at.

        Returns:
            The composed objective, differentiable in ``phase``.
        """
        if float(self.cfg.w_tv) <= 0.0:
            return loss
        tv = (phase[:, 1:] - phase[:, :-1]).abs().sum() + (
            phase[1:, :] - phase[:-1, :]
        ).abs().sum()
        return loss + float(self.cfg.w_tv) * tv

    def _forward_terms(
        self,
        phase: torch.Tensor,
        wanted: torch.Tensor,
        weights: torch.Tensor,
        aberration: torch.Tensor | np.ndarray | None,
        illumination: torch.Tensor | np.ndarray | None,
    ) -> tuple[torch.Tensor, tuple[float, float, float, float]]:
        """Evaluate the returned phase once, without a graph.

        Args:
            phase: ``(grid, grid)`` phase to report on.
            wanted: ``(1, 1, ccd_px, ccd_px)`` target.
            weights: ``(1, 1, ccd_px, ccd_px)`` support weights.
            aberration: Optional ``(grid, grid)`` aberration phase.
            illumination: Optional ``(grid, grid)`` amplitude override.

        Returns:
            ``(frame, (objective, l1, ee, ncc))`` with ``frame`` the
            ``(1, 1, ccd_px, ccd_px)`` predicted intensity.
        """
        frame = self.sim.forward(
            phase.unsqueeze(0),
            aberration=aberration,
            illumination=illumination,
            detector=False,
        )
        loss, l1, ee, ncc = objective_terms(
            frame,
            wanted,
            weights,
            w_ee=self.cfg.w_ee,
            metric=self.cfg.metric,
            prepare=self.cfg.prepare,
        )
        composed = self._compose(loss, phase)
        return frame, (
            float(composed.detach()),
            float(l1.detach()),
            float(ee.detach()),
            float(ncc.detach()),
        )


def finetune_phase(
    sim: Sim2f,
    phase_init: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    support: torch.Tensor | np.ndarray | None = None,
    aberration: torch.Tensor | np.ndarray | None = None,
    illumination: torch.Tensor | np.ndarray | None = None,
    cfg: RefineConfig | None = None,
) -> RefineResult:
    """One-shot convenience wrapper around :class:`PhaseRefiner`.

    Args:
        sim: Bench to refine against (usually the fitted one).
        phase_init: ``(grid, grid)`` starting phase, radians.
        target: ``(ccd_px, ccd_px)`` target distribution.
        support: Optional binary support for ``EE``.
        aberration: Optional ``(grid, grid)`` aberration phase for the bench.
        illumination: Optional ``(grid, grid)`` amplitude override.
        cfg: Contract; defaults to :class:`RefineConfig`.

    Returns:
        The :class:`RefineResult`.
    """
    return PhaseRefiner(sim, cfg).finetune(
        phase_init,
        target,
        support=support,
        aberration=aberration,
        illumination=illumination,
    )
