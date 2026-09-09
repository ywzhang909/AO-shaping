"""Differentiable (backpropagation) beam shaping via an FFT far-field model.

This module implements the "backprop" algorithm for SLM phase retrieval.
Instead of the alternating-projection Gerchberg-Saxton loop, it treats the
phase pattern ``φ`` as a *learnable* tensor and differentiates a
far-field intensity loss with respect to ``φ`` through a differentiable
angular-spectrum model (a Fourier transform), updating ``φ`` with an Adam
optimizer.

Model
-----
At the SLM (source) plane the complex field is

    E_near = A * exp(i * φ)

where ``A`` is the (known, typically uniform) illumination amplitude and
``φ`` is the phase we optimize. The far field is modeled as the Fourier
transform

    E_far = fftshift(fft2(E_near))

and the far-field intensity as ``I_far = |E_far|²``. The loss is the MSE
between ``I_far`` and the (normalized) target intensity. Gradients flow from
the loss through the FFT to ``φ``.

``φ`` is kept un-wrapped (no mod-2π) during optimization for numerical
stability; wrapping to the SLM's grayscale range happens only at
conversion time (see ``beam_shaping_utils.phase_to_slm_grayscale``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

import torch

from ao_shaping.algorithm.beam_shaping_utils import DEFAULT_WAVELENGTH


@dataclass
class BeamOptimizeResult:
    """Result container for the differentiable beam-shaping optimization.

    Attributes:
        phase: Final phase pattern for the SLM in radians (un-wrapped).
        loss_history: Loss value per optimization step.
        final_loss: The loss at the last step.
        converged: Whether the early-stopping criterion was met.
        steps: Number of optimization steps actually performed.
        device: The torch device the optimization ran on.
    """

    phase: np.ndarray
    loss_history: list[float]
    final_loss: float
    converged: bool
    steps: int
    device: str = field(default="cpu")


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a real 2D numpy array to a float32 torch tensor on ``device``."""
    arr = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def differentiable_far_field(
    amplitude: np.ndarray,
    phase: torch.Tensor,
) -> torch.Tensor:
    """Differentiable far-field amplitude from a source amplitude and phase.

    Args:
        amplitude: Real 2D source-plane amplitude (numpy, e.g. uniform 1s).
        phase: 2D phase tensor in radians (must be a leaf or graph-attached
            tensor; the gradient is computed with respect to it).

    Returns:
        2D complex tensor: the far-field complex field
        ``fftshift(fft2(A * exp(i * phase)))``.

    Note:
        ``fftshift`` and the complex exponential are differentiable, so the
        returned field carries a grad_fn linking back to ``phase``.
    """
    amp = _to_tensor(amplitude, phase.device)
    complex_field = amp * torch.exp(1j * phase)
    fft = torch.fft.fft2(complex_field)
    return torch.fft.fftshift(fft)


def far_field_intensity(
    amplitude: np.ndarray,
    phase: torch.Tensor,
) -> torch.Tensor:
    """Differentiable far-field *intensity* ``|E_far|²``.

    Args:
        amplitude: Real 2D source-plane amplitude (numpy).
        phase: 2D phase tensor in radians.

    Returns:
        2D float tensor of shape ``(H, W)`` holding the far-field intensity.
    """
    e_far = differentiable_far_field(amplitude, phase)
    return (e_far.real**2 + e_far.imag**2)


def differentiable_beam_optimize(
    target_intensity: np.ndarray,
    source_amplitude: np.ndarray | None = None,
    lr: float = 0.01,
    epochs: int = 500,
    init_phase: np.ndarray | None = None,
    device: str | None = None,
    early_stop_patience: int = 0,
    early_stop_delta: float = 1e-5,
    log_every: int = 50,
    seed: int | None = None,
) -> BeamOptimizeResult:
    """Optimize an SLM phase map to match a target far-field intensity.

    Uses PyTorch autograd with the Adam optimizer. The loss is the MSE
    between the (energy-normalized) far-field intensity produced by the
    current phase and the normalized target.

    Args:
        target_intensity: 2D target far-field *intensity* map (non-negative).
            It is normalized to unit sum before comparison, so the absolute
            scale of the target does not matter.
        source_amplitude: 2D source-plane (SLM) illumination amplitude.
            If ``None``, a uniform amplitude of ones (same shape as the
            target) is used.
        lr: Adam learning rate.
        epochs: Number of Adam optimization steps.
        init_phase: Initial phase (radians). If ``None``, a small random
            phase is drawn (seeded by ``seed`` for reproducibility).
        device: Torch device string (``"cuda"``, ``"cpu"``). If ``None``,
            CUDA is used when available, else CPU.
        early_stop_patience: If > 0, stop when the loss has not improved by
            more than ``early_stop_delta`` for this many steps.
        early_stop_delta: Minimum improvement required to reset patience.
        log_every: Log a progress line every this many steps (0 to disable).
        seed: Optional RNG seed for the random initial phase.

    Returns:
        A :class:`BeamOptimizeResult` with the final phase (radians), the
        per-step loss history, the final loss, and convergence info.

    Raises:
        ValueError: If the target is not 2D or contains negative values.
    """
    target = np.asarray(target_intensity)
    if target.ndim != 2:
        raise ValueError(f"target_intensity must be 2D, got {target.ndim}D")
    if np.any(target < 0):
        raise ValueError("target_intensity must be non-negative")

    # Peak-normalize the target so absolute scale is irrelevant.
    # (Peak normalization is standard in phase retrieval; energy normalization
    # underflows in float32 because the FFT concentrates energy in a few pixels.)
    t = target.astype(np.float32)
    t_max = float(t.max())
    if t_max > 0:
        t = t / t_max

    if source_amplitude is None:
        source_amplitude = np.ones_like(t)
    else:
        source_amplitude = np.asarray(source_amplitude, dtype=np.float32)
        if source_amplitude.shape != t.shape:
            raise ValueError(
                f"source_amplitude shape {source_amplitude.shape} "
                f"must match target shape {t.shape}"
            )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Seed the RNG for a reproducible random initialization.
    if seed is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        init_phase_np = np.random.default_rng(seed).random(
            t.shape, dtype=np.float32
        ) * (2 * np.pi)
    else:
        init_phase_np = np.random.random(t.shape).astype(np.float32) * (2 * np.pi)

    if init_phase is not None:
        init_phase_np = np.asarray(init_phase, dtype=np.float32)

    phase = torch.from_numpy(init_phase_np).to(dev)
    phase.requires_grad_(True)

    target_t = torch.from_numpy(t).to(dev)
    opt = torch.optim.Adam([phase], lr=lr)

    loss_history: list[float] = []
    best_loss = float("inf")
    steps_without_improvement = 0
    converged = False

    logger.info(
        "Starting differentiable beam optimization: epochs={}, lr={}, device={}",
        epochs, lr, dev,
    )

    for step in range(epochs):
        opt.zero_grad(set_to_none=True)
        i_far = far_field_intensity(source_amplitude, phase)
        # Match the target's peak normalization (target is already max=1.0).
        # Peak-normalizing in float32 avoids the underflow that energy-
        # normalization (sum) suffers from when the FFT concentrates energy.
        i_norm = i_far / (i_far.max() + 1e-8)
        loss = torch.mean((i_norm - target_t) ** 2)
        loss.backward()
        opt.step()

        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if log_every and (step == 0 or (step + 1) % log_every == 0):
            logger.debug(
                "Step {}/{} loss={:.6f}", step + 1, epochs, loss_val
            )

        # Early stopping based on improvement.
        if early_stop_patience > 0:
            if loss_val < best_loss - early_stop_delta:
                best_loss = loss_val
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                if steps_without_improvement >= early_stop_patience:
                    logger.info(
                        "Early stopping at step {} (no improvement for {} steps)",
                        step + 1, early_stop_patience,
                    )
                    converged = True
                    break

    final_phase_np = phase.detach().cpu().numpy()
    final_loss = loss_history[-1] if loss_history else float("nan")

    logger.info(
        "Differentiable beam optimization finished: steps={}, final_loss={:.6f}, converged={}",
        len(loss_history), final_loss, converged,
    )

    return BeamOptimizeResult(
        phase=final_phase_np,
        loss_history=loss_history,
        final_loss=final_loss,
        converged=converged,
        steps=len(loss_history),
        device=str(dev),
    )
