"""Iterative arbitrary-shape shaping on a measured bench.

This module drives the "deep model + physical device" closed loop the bench
exists for.  One round is three steps:

1. **Refit** -- the aberration estimate is updated against the last *measured*
   frame (:func:`ml.sim2f.fit.refine`), so the model tracks the bench's slow
   drift instead of trusting a one-off probe.
2. **Refine** -- the current phase is *locally* fine-tuned through the fitted
   bench (:class:`ml.sim2f.finetune.PhaseRefiner`): a bounded update inside the
   model's trusted neighbourhood, not a free re-optimisation.
3. **Accept on the device** -- the candidate is measured and kept only if the
   *measured* objective improves by ``ShapeLoopConfig.min_gain``.  Steps 1-2
   trust a model that is only partly right; the hardware session recorded what
   an unguarded model-optimal phase does (``net`` lost to ``flat``,
   ``runs/sim2f_bench/session_hw/session.json``), and the measured acceptance
   test is the guard.

Offline, the "device" is a bench object too -- a
:class:`ml.sim2f.calibrate.CalibratedBench` fitted to recorded hardware
frames, or a synthetic ground-truth bench -- so the loop code never knows the
difference, which is exactly what keeps the hardware path identical.  An
honest trial never lets ``device`` and ``model`` be the same object.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from ml.sim2f.config import AberrationConfig, ShapeLoopConfig
from ml.sim2f.evaluate import _row as _score_row
from ml.sim2f.finetune import PhaseRefiner
from ml.sim2f.fit import FitConfig, refine
from ml.sim2f.targets import TargetSample

__all__ = [
    "ArmResult",
    "RoundRecord",
    "ShapeTrialResult",
    "run_shape_trial",
]

logger = logging.getLogger(__name__)

#: A measurement: ``(grid, grid)`` commanded radians -> ``(ccd_px, ccd_px)``
#: or ``(1, 1, ccd_px, ccd_px)`` camera frame, tensor or array.  This is the
#: shape ``HwInstrument.capture`` has, so the same driver runs on hardware.
Measure = Callable[[torch.Tensor], torch.Tensor | np.ndarray]

#: Guard for the acceptance-gain denominator.
_EPS = 1e-12


@dataclass(frozen=True)
class RoundRecord:
    """One closed-loop round, with both gates' verdicts.

    Attributes:
        round: 1-based round index.
        accepted_model: Whether the refiner's own (model-side) gate passed.
            When ``False`` the candidate equals the current phase and no device
            measurement is taken.
        accepted_device: Whether the candidate improved the measured objective
            by ``min_gain`` and became the new current phase.
        gain_device: Relative measured-objective improvement of the candidate
            (``0`` when no candidate was measured).
        objective: Measured objective at the end of the round (the current
            phase's, whatever was accepted).
        l1: Measured sum-L1 at the end of the round.
        ee: Measured encircled energy at the end of the round.
        ncc: Measured NCC at the end of the round.
        cv_in_support: Measured in-support uniformity at the end of the round.
        delta_rms_rad: RMS of the accepted phase update (``0`` on rejection).
        fit_ok: Whether the aberration refit was usable.
        fit_residual_start: Refit residual at the incoming coefficients.
        fit_residual: Refit residual at the returned coefficients.
        coeff_rms_rad: RMS of the aberration estimate after the refit.
        phase: The command phase this round's metrics were measured on,
            ``(grid, grid)`` float32 radians (the accepted candidate, or the
            unchanged current phase when the round was rejected).  Kept for
            replaying the optimisation as a sequence of camera frames; it is
            excluded from equality/repr so records stay comparable and cheap
            to log.
    """

    round: int
    accepted_model: bool
    accepted_device: bool
    gain_device: float
    objective: float
    l1: float
    ee: float
    ncc: float
    cv_in_support: float
    delta_rms_rad: float
    fit_ok: bool
    fit_residual_start: float
    fit_residual: float
    coeff_rms_rad: float
    phase: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ArmResult:
    """One arm's outcome.

    Attributes:
        arm: ``"flat"``, ``"net_open"`` or ``"net_finetune"`` (``"opt_finetune"``
            when no network phase was supplied).
        start: Metric row of the first measurement (the arm's open-loop state).
        final: Metric row of the last accepted state (equals ``start`` for the
            static arms).
        rounds: Per-round records; empty for the static arms.
        phase: Final command phase, ``(grid, grid)`` float32 radians.
    """

    arm: str
    start: dict[str, float]
    final: dict[str, float]
    rounds: tuple[RoundRecord, ...]
    phase: np.ndarray


@dataclass(frozen=True)
class ShapeTrialResult:
    """One target, every arm.

    Attributes:
        target_kind: The target's ``kind`` (``"mask"`` for arbitrary bitmaps).
        geometry: The target's drawn geometry, for the report.
        config: The contract the trial ran with.
        arms: One entry per arm that ran.
    """

    target_kind: str
    geometry: dict[str, float]
    config: ShapeLoopConfig
    arms: tuple[ArmResult, ...]

    def arm(self, name: str) -> ArmResult:
        """Look up one arm's result.

        Args:
            name: The arm name.

        Returns:
            The :class:`ArmResult`.

        Raises:
            KeyError: If the arm did not run.
        """
        for arm in self.arms:
            if arm.arm == name:
                return arm
        raise KeyError(f"arm {name!r} did not run; ran: {[a.arm for a in self.arms]}")


def _as_command(value: torch.Tensor | np.ndarray, bench: object) -> torch.Tensor:
    """Coerce a phase to ``(grid, grid)`` on the bench's device.

    Args:
        value: ``(grid, grid)`` or ``(1, grid, grid)`` radians.
        bench: Supplies ``grid``, ``dtype``, ``device``.

    Returns:
        The command tensor.

    Raises:
        ValueError: If the map does not match the bench grid.
    """
    grid = int(bench.grid)  # type: ignore[attr-defined]
    dtype = bench.dtype  # type: ignore[attr-defined]
    device = torch.device(bench.device)  # type: ignore[attr-defined]
    tensor = torch.as_tensor(
        value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value,
        dtype=dtype,
        device=device,
    ).squeeze(0)
    if tuple(tensor.shape) != (grid, grid):
        raise ValueError(f"phase must be ({grid}, {grid}), got {tuple(tensor.shape)}")
    return tensor.contiguous()


def _as_measured(
    frame: torch.Tensor | np.ndarray, bench: object
) -> torch.Tensor:
    """Coerce a measurement to ``(1, 1, ccd_px, ccd_px)`` on the bench.

    Args:
        frame: ``(ccd_px, ccd_px)`` or ``(1, 1, ccd_px, ccd_px)`` camera frame.
        bench: Supplies ``ccd_px``, ``dtype``, ``device``.

    Returns:
        The frame tensor.

    Raises:
        ValueError: If the frame does not match the camera window.
    """
    ccd_px = int(bench.ccd_px)  # type: ignore[attr-defined]
    dtype = bench.dtype  # type: ignore[attr-defined]
    device = torch.device(bench.device)  # type: ignore[attr-defined]
    tensor = torch.as_tensor(
        frame.detach().cpu().numpy() if isinstance(frame, torch.Tensor) else frame,
        dtype=dtype,
        device=device,
    )
    if tensor.dim() == 2:
        tensor = tensor.view(1, 1, ccd_px, ccd_px)
    if tuple(tensor.shape) != (1, 1, ccd_px, ccd_px):
        raise ValueError(
            f"measurement must be (1, 1, {ccd_px}, {ccd_px}), got {tuple(tensor.shape)}"
        )
    return tensor.contiguous()


def run_shape_trial(
    device: object,
    model: object,
    target: TargetSample,
    *,
    net_phase: torch.Tensor | np.ndarray | None = None,
    measure: Measure | None = None,
    config: ShapeLoopConfig | None = None,
) -> ShapeTrialResult:
    """Run the shaping arms for one arbitrary target, with closed-loop rounds.

    Arms: ``flat`` (zero phase) and, when ``net_phase`` is given, ``net_open``
    (the network's answer, measured once) are the static baselines;
    ``net_finetune`` starts from the network's answer and runs the
    refit/refine/accept rounds (``opt_finetune`` starts from flat when no
    network is involved).

    Args:
        device: The measured bench -- hardware instrument, calibrated
            surrogate or synthetic truth.  Needs ``grid`` / ``ccd_px`` /
            ``dtype`` / ``device`` and, unless ``measure`` is given,
            ``forward(phase, detector=False)``.
        model: The bench the refit and the refiner trust.  Needs the full
            bench protocol (:func:`ml.sim2f.fit.refine`'s).
        target: The requested far-field distribution.
        net_phase: Optional ``(grid, grid)`` open-loop phase (e.g. the U-Net's
            answer).
        measure: Optional measurement callable; defaults to
            ``device.forward(phase, detector=False)`` (deterministic surrogate).
        config: The loop contract; defaults to :class:`ShapeLoopConfig`.

    Returns:
        The :class:`ShapeTrialResult`.

    Raises:
        ValueError: If the benches' geometries disagree or the target does not
            match the camera window.
    """
    cfg = ShapeLoopConfig() if config is None else config
    grid = int(model.grid)  # type: ignore[attr-defined]
    ccd_px = int(model.ccd_px)  # type: ignore[attr-defined]
    if int(device.grid) != grid or int(device.ccd_px) != ccd_px:  # type: ignore[attr-defined]
        raise ValueError(
            f"device {(int(device.grid), int(device.ccd_px))} and model "  # type: ignore[attr-defined]
            f"{(grid, ccd_px)} geometries disagree"
        )
    if tuple(target.intensity.shape) != (ccd_px, ccd_px):
        raise ValueError(
            f"target intensity is {tuple(target.intensity.shape)}, "
            f"camera window is {(ccd_px, ccd_px)}"
        )

    dtype = model.dtype  # type: ignore[attr-defined]
    device_t = torch.device(model.device)  # type: ignore[attr-defined]

    if measure is None:

        def measure(phase: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return device.forward(  # type: ignore[attr-defined]
                    phase.unsqueeze(0), detector=False
                )

    def measure_row(command: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        frame = _as_measured(measure(command), model)
        row = _score_row(
            frame,
            target_t,
            support_t,
            w_ee=float(cfg.w_ee),
            metric=str(cfg.metric),
            prepare=True,
        )
        return frame, row

    target_t = torch.as_tensor(
        np.asarray(target.intensity), dtype=dtype, device=device_t
    ).view(1, 1, ccd_px, ccd_px)
    support_t = torch.as_tensor(
        np.asarray(target.support, dtype=np.float32), dtype=dtype, device=device_t
    ).view(1, 1, ccd_px, ccd_px)

    ab_cfg = AberrationConfig(
        n_modes=int(cfg.n_modes),
        fit_frozen=tuple(int(noll) for noll in cfg.fit_frozen),
        fit_iters=int(cfg.fit_iters),
        fit_lr=float(cfg.fit_lr),
    )
    fit_cfg = FitConfig.from_aberration(ab_cfg, trust_rad=float(cfg.trust_rad))
    refiner = PhaseRefiner(model, cfg.refine)

    def static_arm(name: str, command: torch.Tensor) -> ArmResult:
        """Measure one fixed phase once."""
        _, row = measure_row(command)
        return ArmResult(
            arm=name,
            start=row,
            final=row,
            rounds=(),
            phase=command.detach().cpu().numpy().astype(np.float32),
        )

    def closed_arm(name: str, start: torch.Tensor) -> ArmResult:
        """The refit / refine / accept loop from one starting phase."""
        phase = start
        frame, row = measure_row(phase)
        start_row = dict(row)
        coeffs = np.zeros(int(cfg.n_modes), dtype=np.float64)
        records: list[RoundRecord] = []
        for index in range(1, int(cfg.rounds) + 1):
            fit_result = refine(
                model,
                phase,
                frame,
                coeffs,
                cfg=ab_cfg,
                fit_cfg=fit_cfg,
            )
            coeffs = fit_result.coeffs
            residual_map = (
                model.aberration(coeffs[None].astype(np.float32))[0]  # type: ignore[attr-defined]
                if fit_result.ok and bool(np.any(coeffs))
                else None
            )
            refined = refiner.finetune(
                phase,
                target.intensity,
                support=target.support,
                aberration=residual_map,
            )
            gain_device = 0.0
            accepted_device = False
            if refined.accepted:
                candidate = _as_command(refined.phase, model)
                candidate_frame, candidate_row = measure_row(candidate)
                gain_device = (row["objective"] - candidate_row["objective"]) / max(
                    abs(row["objective"]), _EPS
                )
                if gain_device >= float(cfg.min_gain):
                    accepted_device = True
                    phase, frame, row = candidate, candidate_frame, candidate_row
                else:
                    logger.debug(
                        "round {} rejected on the device: gain {:.4f} < min_gain",
                        index,
                        gain_device,
                    )
            records.append(
                RoundRecord(
                    round=index,
                    accepted_model=bool(refined.accepted),
                    accepted_device=accepted_device,
                    gain_device=float(gain_device),
                    objective=float(row["objective"]),
                    l1=float(row["l1"]),
                    ee=float(row["ee"]),
                    ncc=float(row["ncc"]),
                    cv_in_support=float(row["cv_in_support"]),
                    delta_rms_rad=(
                        float(refined.delta_rms_rad) if accepted_device else 0.0
                    ),
                    fit_ok=bool(fit_result.ok),
                    fit_residual_start=float(fit_result.residual_start),
                    fit_residual=float(fit_result.residual),
                    coeff_rms_rad=float(np.sqrt(np.mean(coeffs**2))),
                    phase=phase.detach().cpu().numpy().astype(np.float32),
                )
            )
        return ArmResult(
            arm=name,
            start=start_row,
            final=row,
            rounds=tuple(records),
            phase=phase.detach().cpu().numpy().astype(np.float32),
        )

    arms: list[ArmResult] = []
    flat = torch.zeros(grid, grid, dtype=dtype, device=device_t)
    arms.append(static_arm("flat", flat))
    start = flat
    name = "opt_finetune"
    if net_phase is not None:
        net_command = _as_command(net_phase, model)
        arms.append(static_arm("net_open", net_command))
        start = net_command
        name = "net_finetune"
    arms.append(closed_arm(name, start))

    return ShapeTrialResult(
        target_kind=str(target.kind),
        geometry=dict(target.geometry),
        config=cfg,
        arms=tuple(arms),
    )
