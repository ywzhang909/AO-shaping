"""Differentiable (backpropagation) beam shaping — optimizer-level wrapper.

This module hosts the pib-style top-level optimizer function
:func:`optimize_beam_shaping`, which mirrors the structure of
:func:`ao_shaping.optimizer.wfless.pib.optimize_pib`: upfront validation and
RNG seeding, a tqdm progress bar, per-step history recorded through a
:class:`~ao_shaping.utils.Recorder`, best-phase/best-loss tracking and a
summary log line.

The optimization math itself lives in the original stateful class
:class:`ao_shaping.algorithm.differentiable_beam.DifferentiableBeamOptimizer`
(``__init__`` validation + one-step ``update()``); this function constructs
that class and drives ``update()`` in a loop. The class remains the sole
public API within the ``algorithm`` package; the full-loop convenience lives
here in the optimizer layer.

Two forward-propagation modes are supported:

* **Simulation mode** (default, ``slm=None`` and ``ccd=None``): the model's
  own far-field intensity is the forward pass — ``update()`` is called with
  no ``measured_intensity`` and the loss is the simulated far-field loss.
* **Hardware-closed-loop mode** (``slm`` and/or ``ccd`` provided): the
  forward pass goes through the REAL hardware chain. Each iteration captures
  a CCD image of the currently applied SLM phase, resizes it to the target
  grid, and calls ``update(measured_intensity=measured)`` so the backprop
  step is anchored on the measured image (the model Jacobian is weighted by
  the loss gradient evaluated at the measured image). The updated phase is
  then displayed on the SLM for the next iteration. The missing device is
  auto-created when only one is provided; auto-created devices are opened
  and closed by this function (user-provided devices keep their lifecycle).

The backprop/Gerchberg-Saxton phase-retrieval pipeline is imported from the
optimizer package::

    from ao_shaping.optimizer import optimize_beam_shaping

Note:
    Unlike :mod:`ao_shaping.optimizer.wfless.differentiable_shaping`, the
    underlying module imports ``torch`` at module top level (torch is a
    hard dependency of the backprop path).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tqdm
from loguru import logger

from ao_shaping.algorithm.signal_processing.differentiable_beam import (
    DifferentiableBeamOptimizer,
    differentiable_far_field,
    far_field_intensity,
)
from ao_shaping.utils import Recorder
from ao_shaping.utils.hardware_utils import call_with_timeout
from ao_shaping.utils.slm.phase_display import DEFAULT_WAVELENGTH
from ao_shaping.utils.image.targets import crop_resize_to_grid

# Hardware drivers are imported guarded: the SDKs are not installed on every
# machine, and this module must stay importable without them. The auto-create
# helpers raise a clear RuntimeError when the driver is unavailable.
try:
    from ao_shaping.drivers import MIICamera
except Exception:  # pragma: no cover - hardware SDK not installed
    MIICamera = None  # type: ignore[assignment]

try:
    from ao_shaping.drivers.slm.santec import Santec
except Exception:  # pragma: no cover - hardware SDK not installed
    Santec = None  # type: ignore[assignment]

# Watchdog timeout for hardware SDK calls (SLM open, CCD capture) that can
# hang forever with no Python-visible timeout (see diff_beam_runner).
_CAPTURE_TIMEOUT_S = 30.0


def _display_phase(slm: Any, phase_rad: np.ndarray, wait_time_s: float) -> None:
    """Display a radian phase pattern on the SLM.

    Hardware SLMs expose ``create_phase_from_array`` (device-authentic 2π
    conversion + correction/LUT) and ``display_data`` (which rotates memory
    slots internally, satisfying the no-same-slot-twice rule). Mock SLMs
    (tests) only expose ``write_phase`` and take radians directly.
    """
    if hasattr(slm, "display_phase"):
        slm.display_phase(phase_rad, wait_time_s=wait_time_s)
    else:
        slm.write_phase(phase_rad)


def _capture_frame(ccd: Any, discard_count: int) -> np.ndarray:
    """Discard stale frames, then capture one CCD frame (watchdog-bounded).

    The first ``discard_count`` frames are thrown away so the SLM has settled
    and the SDK's internal frame buffer is flushed; the next frame is the
    measurement. Every SDK call is bounded by ``call_with_timeout`` because
    the native wait is not Python-visible (see diff_beam_runner).
    """
    for _ in range(max(0, int(discard_count))):
        call_with_timeout(
            lambda: ccd.get_numpy_image(n_sample=1, skip_first=True),
            _CAPTURE_TIMEOUT_S,
            "CCD 丢弃帧",
        )
    frame = call_with_timeout(
        lambda: ccd.get_numpy_image(n_sample=1, skip_first=True),
        _CAPTURE_TIMEOUT_S,
        "CCD 采集",
    )
    return np.asarray(frame, dtype=np.float32)


def _auto_create_slm() -> Any:
    """Create and open a default Santec SLM200 (memory mode only)."""
    if Santec is None:
        raise RuntimeError("SLM driver not available. Install Santec SLM SDK.")
    slm = Santec(slm_number=1, wavelength=int(DEFAULT_WAVELENGTH * 1e9))
    call_with_timeout(slm.open, _CAPTURE_TIMEOUT_S, "SLM open")
    return slm


def _auto_create_ccd(cam_id: int, exposure_time_ms: float | None) -> Any:
    """Create and open the default CCD (Daheng, else MiiCam)."""
    if MIICamera is None:
        raise RuntimeError("CCD driver not available. Install Daheng/MiiCam SDK.")
    ccd = MIICamera(
        cam_id=cam_id,
        exposure_time_ms=exposure_time_ms if exposure_time_ms is not None else 0.0,
        skip_sampling=False,
    )
    call_with_timeout(ccd.open, _CAPTURE_TIMEOUT_S, "CCD open")
    return ccd


def optimize_beam_shaping(
    target_intensity: np.ndarray,
    source_amplitude: np.ndarray | None = None,
    lr: float = 0.01,
    init_phase: np.ndarray | None = None,
    device: str | None = None,
    seed: int | None = None,
    epochs: int = 500,
    early_stop_patience: int = 0,
    early_stop_delta: float = 1e-5,
    log_every: int = 50,
    progress: bool = True,
    slm: Any | None = None,
    ccd: Any | None = None,
    cam_id: int = 0,
    exposure_time_ms: float | None = None,
    wait_time_s: float = 0.3,
    discard_count: int = 3,
) -> Recorder:
    """Optimize an SLM phase map to match a target far-field intensity.

    Pib-style top-level optimizer for the differentiable (backprop) beam
    shaping algorithm. Constructs the original
    :class:`~ao_shaping.algorithm.differentiable_beam.DifferentiableBeamOptimizer`
    and drives its :meth:`update` in a loop, recording per-step history
    through a :class:`~ao_shaping.utils.Recorder` (mark ``"loss"``, mode
    ``"min"``).

    When ``slm`` and ``ccd`` are both ``None`` the loop is a pure simulation
    (the model's own far field is the forward pass). When at least one device
    is provided the loop is a hardware closed loop: each iteration captures
    the CCD image of the currently applied SLM phase, resizes it to the
    target grid, performs one measurement-anchored backprop step, and
    displays the updated phase on the SLM. The missing device is auto-created
    (and closed on exit); user-provided devices keep their lifecycle.

    Args:
        target_intensity: 2D target far-field *intensity* map
            (non-negative). It is peak-normalized to a maximum of 1.0
            before comparison, so the absolute scale of the target does
            not matter.
        source_amplitude: 2D source-plane (SLM) illumination amplitude.
            If ``None``, a uniform amplitude of ones (same shape as the
            target) is used.
        lr: Adam learning rate.
        init_phase: Initial phase (radians). If ``None``, a small random
            phase is drawn (seeded by ``seed`` for reproducibility).
        device: Torch device string (``"cuda"``, ``"cpu"``). If ``None``,
            CUDA is used when available, else CPU.
        seed: Optional RNG seed for the random initial phase.
        epochs: Number of Adam optimization steps.
        early_stop_patience: If > 0, stop when the loss has not improved
            by more than ``early_stop_delta`` for this many steps.
        early_stop_delta: Minimum improvement required to reset patience.
        log_every: Log a progress line every this many steps (0 to
            disable).
        progress: Whether to show a tqdm progress bar.
        slm: SLM device for hardware-closed-loop mode. If ``None`` while
            ``ccd`` is provided, a default Santec SLM200 is auto-created
            (and closed on exit).
        ccd: CCD camera for hardware-closed-loop mode. If ``None`` while
            ``slm`` is provided, the default camera (Daheng, else MiiCam)
            is auto-created (and closed on exit).
        cam_id: Camera device ID used when auto-creating ``ccd``.
        exposure_time_ms: Camera exposure in milliseconds used when
            auto-creating ``ccd`` (``None`` keeps the driver default).
        wait_time_s: Settle time after displaying a phase on the SLM
            (hardware mode).
        discard_count: Number of CCD frames to discard before each capture
            (hardware mode), letting the SLM settle and flushing stale
            frames.

    Returns:
        A :class:`~ao_shaping.utils.Recorder` with mark ``"loss"``. Each
        record holds ``loss``, ``lr``, ``_epoch``, ``best_loss``,
        ``best_phase`` (radians, the phase that produced ``best_loss``),
        ``converged`` and ``device``. In hardware mode each record also
        holds ``measured_image`` (the resized measured CCD frame that
        produced the recorded loss). The best phase is available via
        ``recorder.get_best_iter()["best_phase"]``.

    Raises:
        ValueError: If ``progress`` is not a bool, or the target is not 2D,
            contains negative values, or the source amplitude / initial
            phase shape does not match the target shape.
        RuntimeError: In hardware mode, if the required driver SDK is not
            installed.
    """
    # Upfront validation and normalization (mirrors optimize_pib).
    epochs = int(epochs)
    early_stop_patience = int(early_stop_patience)
    early_stop_delta = abs(float(early_stop_delta))
    log_every = int(log_every)
    if not isinstance(progress, bool):
        raise ValueError(f"progress must be a bool, got {type(progress).__name__}")

    # Seed the RNG for reproducibility (mirrors optimize_pib). The Adam
    # loop itself is deterministic; the RNG is kept for parity with the
    # reference structure.
    rng = np.random.default_rng(seed)

    optimizer = DifferentiableBeamOptimizer(
        target_intensity=target_intensity,
        source_amplitude=source_amplitude,
        lr=lr,
        init_phase=init_phase,
        device=device,
        seed=seed,
    )

    logger.info(
        "Starting differentiable beam optimization: epochs={}, lr={}, device={}",
        epochs,
        lr,
        optimizer.device,
    )

    recorder = Recorder(mark="loss", mode="min")
    best_loss = float("inf")
    best_phase: np.ndarray | None = None
    steps_without_improvement = 0
    converged = False

    # Hardware-closed-loop mode: forward propagation goes through the real
    # SLM -> CCD chain. When at least one device is provided, the missing
    # one is auto-created (and auto-created devices are closed on exit).
    hardware_mode = slm is not None or ccd is not None

    if not hardware_mode:
        # --- Simulation mode: the model's own far field is the forward pass.
        with tqdm.tqdm(
            total=epochs,
            desc=f"iter {epochs}",
            dynamic_ncols=True,
            disable=not progress,
        ) as bar:
            for step in range(epochs):
                optimizer.update()

                loss_val = optimizer.loss_history[-1]

                # Track the best loss/phase (delta-threshold improvement).
                if loss_val < best_loss - early_stop_delta:
                    best_loss = loss_val
                    best_phase = optimizer.current_phase
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                # Early stopping based on improvement.
                if (
                    early_stop_patience > 0
                    and steps_without_improvement >= early_stop_patience
                ):
                    converged = True

                recorder.append(
                    {
                        "loss": loss_val,
                        "lr": lr,
                        "_epoch": step,
                        "best_loss": best_loss,
                        "best_phase": (
                            best_phase.copy() if best_phase is not None else None
                        ),
                        "converged": converged,
                        "device": optimizer.device,
                    }
                )

                if log_every and (step == 0 or (step + 1) % log_every == 0):
                    logger.debug("Step {}/{} loss={:.6f}", step + 1, epochs, loss_val)

                bar.set_postfix({"loss": loss_val, "lr": lr, "best": best_loss})
                bar.update(1)

                if converged:
                    logger.info(
                        "Early stopping at step {} (no improvement for {} steps)",
                        step + 1,
                        early_stop_patience,
                    )
                    break

        final_loss = recorder.last["loss"] if recorder.history else float("nan")

        logger.info(
            "Differentiable beam optimization finished: steps={}, final_loss={:.6f}, "
            "best_loss={}, converged={}",
            len(recorder.history),
            final_loss,
            f"{best_loss:.6f}" if recorder.history else "n/a",
            converged,
        )

        return recorder

    # --- Hardware-closed-loop mode.
    owns_slm = slm is None
    owns_ccd = ccd is None
    slm_dev = slm if slm is not None else _auto_create_slm()
    ccd_dev = ccd if ccd is not None else _auto_create_ccd(cam_id, exposure_time_ms)

    target_h = int(optimizer.phase_tensor.shape[0])
    target_w = int(optimizer.phase_tensor.shape[1])

    try:
        # Apply the optimizer's initial phase once so the SLM state matches
        # the optimizer state before the first capture (this also covers
        # "apply init_phase once if provided").
        applied_phase = optimizer.current_phase
        _display_phase(slm_dev, applied_phase, wait_time_s)

        with tqdm.tqdm(
            total=epochs,
            desc=f"iter {epochs}",
            dynamic_ncols=True,
            disable=not progress,
        ) as bar:
            for step in range(epochs):
                # 1. Capture the CCD image of the CURRENT applied phase.
                frame = _capture_frame(ccd_dev, discard_count)
                measured = crop_resize_to_grid(frame, grid_h=target_h, grid_w=target_w)

                # 2. One measurement-anchored backprop step. The recorded
                #    loss is the MEASURED loss (evaluated at the CCD image).
                optimizer.update(measured_intensity=measured)
                loss_val = optimizer.loss_history[-1]

                # 3. Track the best loss/phase on the APPLIED phase — the one
                #    that was on the SLM during the capture that produced the
                #    measured loss (not the post-update phase).
                if loss_val < best_loss - early_stop_delta:
                    best_loss = loss_val
                    best_phase = applied_phase.copy()
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                # Early stopping based on improvement.
                if (
                    early_stop_patience > 0
                    and steps_without_improvement >= early_stop_patience
                ):
                    converged = True

                recorder.append(
                    {
                        "loss": loss_val,
                        "lr": lr,
                        "_epoch": step,
                        "best_loss": best_loss,
                        "best_phase": (
                            best_phase.copy() if best_phase is not None else None
                        ),
                        "converged": converged,
                        "device": optimizer.device,
                        "measured_image": measured.copy(),
                    }
                )

                if log_every and (step == 0 or (step + 1) % log_every == 0):
                    logger.debug("Step {}/{} loss={:.6f}", step + 1, epochs, loss_val)

                bar.set_postfix({"loss": loss_val, "lr": lr, "best": best_loss})
                bar.update(1)

                if converged:
                    logger.info(
                        "Early stopping at step {} (no improvement for {} steps)",
                        step + 1,
                        early_stop_patience,
                    )
                    break

                # 4. Apply the new phase for the next iteration.
                applied_phase = optimizer.current_phase
                _display_phase(slm_dev, applied_phase, wait_time_s)
    finally:
        if owns_slm:
            slm_dev.close()
        if owns_ccd:
            ccd_dev.close()

    final_loss = recorder.last["loss"] if recorder.history else float("nan")

    logger.info(
        "Differentiable beam optimization finished: steps={}, final_loss={:.6f}, "
        "best_loss={}, converged={}",
        len(recorder.history),
        final_loss,
        f"{best_loss:.6f}" if recorder.history else "n/a",
        converged,
    )

    return recorder


__all__ = [
    "optimize_beam_shaping",
    "DifferentiableBeamOptimizer",
    "differentiable_far_field",
    "far_field_intensity",
]
