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

The optimization loop is exposed as a stateful class
(:class:`DifferentiableBeamOptimizer`) with ``__init__`` (validation + state)
and ``update()`` (one step). The full loop with logging, a progress bar,
per-step history and best-phase tracking lives in the optimizer layer as
``ao_shaping.optimizer.wfless.differentiable_beam.optimize_beam_shaping``
(pib-style top-level function that drives ``update()``). The legacy one-shot
function ``differentiable_beam_optimize`` was removed; the class is the sole
public API within the algorithm package.
"""

from __future__ import annotations

import numpy as np
import torch

from ao_shaping.algorithm.signal_processing.iterative_base import IterativeOptimizer
from ao_shaping.utils.slm_utils import DEFAULT_WAVELENGTH


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
    return e_far.real**2 + e_far.imag**2


class DifferentiableBeamOptimizer(IterativeOptimizer):
    """Optimize an SLM phase map to match a target far-field intensity.

    Uses PyTorch autograd with the Adam optimizer. The loss is the MSE
    between the (peak-normalized) far-field intensity produced by the
    current phase and the normalized target.

    The optimizer is stateful: construct it once, then call :meth:`update`
    repeatedly for manual control. The full loop (logging, progress bar,
    early stopping, best-phase tracking) is provided by the optimizer-layer
    function ``ao_shaping.optimizer.wfless.differentiable_beam.optimize_beam_shaping``.

    Attributes:
        step: Number of optimization steps performed so far.
        device: The torch device the optimization runs on.
        phase_tensor: The live phase parameter tensor (requires_grad).
        current_phase: Detached numpy copy of the current phase (radians).
        loss_history: Loss value per optimization step.
        last_loss: The loss at the last step, or ``None`` before the first
            update.
        converged: Whether the early-stopping criterion was met.
    """

    def __init__(
        self,
        target_intensity: np.ndarray,
        source_amplitude: np.ndarray | None = None,
        lr: float = 0.01,
        init_phase: np.ndarray | None = None,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the optimizer and validate all inputs.

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

        Raises:
            ValueError: If the target is not 2D, contains negative values,
                or the source amplitude / initial phase shape does not match
                the target shape.
        """
        target = np.asarray(target_intensity)
        if target.ndim != 2:
            raise ValueError(f"target_intensity must be 2D, got {target.ndim}D")
        if np.any(target < 0):
            raise ValueError("target_intensity must be non-negative")

        # Peak-normalize the target so absolute scale is irrelevant.
        # (Peak normalization is standard in phase retrieval; energy
        # normalization underflows in float32 because the FFT concentrates
        # energy in a few pixels.)
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
            rng = np.random.default_rng(seed)
            init_phase_np = rng.random(t.shape, dtype=np.float32) * (2 * np.pi)
        else:
            init_phase_np = np.random.random(t.shape).astype(np.float32) * (2 * np.pi)

        if init_phase is not None:
            init_phase_np = np.asarray(init_phase, dtype=np.float32)
            if init_phase_np.shape != t.shape:
                raise ValueError(
                    f"init_phase shape {init_phase_np.shape} "
                    f"must match target shape {t.shape}"
                )

        phase = torch.from_numpy(init_phase_np).to(dev)
        phase.requires_grad_(True)

        self._target_t = torch.from_numpy(t).to(dev)
        self._source_amplitude = source_amplitude
        self._phase = phase
        self._opt = torch.optim.Adam([phase], lr=lr)
        self._lr = lr
        self._dev = dev
        self._seed = seed
        # Initialize the base iteration bookkeeping.  ``max_iterations=1``
        # keeps the base's ``is_converged`` (budget-exhausted) check inert:
        # the real convergence is the class's own early-stopping flag
        # ``_converged`` (see the ``converged`` property), so the base
        # ``run()``/``is_converged`` budget is a no-op safeguard, not the
        # primary stopping condition.
        super().__init__(max_iterations=1)
        self._step = 0
        self._loss_history: list[float] = []
        self._converged = False

    @property
    def step(self) -> int:
        """Number of optimization steps performed so far."""
        return self._step

    @property
    def device(self) -> str:
        """The torch device the optimization runs on."""
        return str(self._dev)

    @property
    def phase_tensor(self) -> torch.Tensor:
        """The live phase parameter tensor (requires_grad)."""
        return self._phase

    @property
    def current_phase(self) -> np.ndarray:
        """Detached numpy copy of the current phase (radians)."""
        return self._phase.detach().cpu().numpy()

    @property
    def loss_history(self) -> list[float]:
        """Loss value per optimization step."""
        return self._loss_history

    @property
    def last_loss(self) -> float | None:
        """The loss at the last step, or ``None`` before the first update."""
        return self._loss_history[-1] if self._loss_history else None

    @property
    def converged(self) -> bool:
        """Whether the early-stopping criterion was met."""
        return self._converged

    def _intensity_loss(self, i_tensor: torch.Tensor) -> torch.Tensor:
        """MSE between a peak-normalized intensity map and the target.

        The loss convention used by the whole class: peak-normalize the
        intensity to a maximum of 1.0 (matching the normalized target) and
        return the mean squared difference. Peak normalization in float32
        avoids the underflow that energy normalization (sum) suffers from
        when the FFT concentrates energy in a few pixels.

        Args:
            i_tensor: 2D intensity tensor of shape ``(H, W)``.

        Returns:
            Scalar MSE loss tensor (differentiable w.r.t. ``i_tensor``).
        """
        i_norm = i_tensor / (i_tensor.max() + 1e-8)
        return torch.mean((i_norm - self._target_t) ** 2)

    def loss_value(self) -> torch.Tensor:
        """Compute the current MSE loss tensor (no backward).

        Returns:
            The MSE between the peak-normalized far-field intensity of the
            current phase and the normalized target.
        """
        i_far = far_field_intensity(self._source_amplitude, self._phase)
        return self._intensity_loss(i_far)

    def update(
        self, measured_intensity: np.ndarray | None = None
    ) -> np.ndarray:
        """Perform one backpropagation step and return the next phase.

        With ``measured_intensity=None`` (the default) the step is the
        classic simulation step: the loss is computed from the model's own
        far-field intensity of the current phase and backpropagated through
        the phase tensor.

        With a ``measured_intensity`` array the step is *measurement-anchored*
        (hardware closed loop): the loss is evaluated at the MEASURED far-field
        image, its gradient w.r.t. the intensity (``∂ℓ/∂I``) is computed at
        that measured image, and that gradient is used as the upstream
        gradient for the model's far-field intensity of the current phase —
        i.e. the model Jacobian ``∂I_sim/∂φ`` is weighted by ``∂ℓ/∂I``
        evaluated at the measured image. The phase is then updated with the
        internal Adam step. The recorded loss is the MEASURED loss.

        If the optimizer has already converged, this is a no-op that
        returns the current phase without stepping.

        Args:
            measured_intensity: Optional 2D measured far-field intensity map
                (e.g. a CCD frame resized to the target grid). Peak-normalized
                before comparison, matching the target's normalization.

        Returns:
            The updated phase as a detached numpy array (radians).

        Raises:
            ValueError: If ``measured_intensity`` is not 2D or its shape does
                not match the target shape.
        """
        if self._converged:
            return self.current_phase
        self._opt.zero_grad(set_to_none=True)

        if measured_intensity is None:
            loss = self.loss_value()
            loss.backward()
            self._opt.step()
            self._step += 1
            self._loss_history.append(float(loss.detach().cpu()))
            self._record(float(loss.detach().cpu()))
            return self.current_phase

        # Measurement-anchored step: evaluate the loss at the measured image.
        measured = np.asarray(measured_intensity, dtype=np.float32)
        if measured.ndim != 2:
            raise ValueError(
                f"measured_intensity must be 2D, got {measured.ndim}D"
            )
        if measured.shape != tuple(self._target_t.shape):
            raise ValueError(
                f"measured_intensity shape {measured.shape} must match "
                f"target shape {tuple(self._target_t.shape)}"
            )
        i_meas = torch.from_numpy(measured).to(self._dev)
        i_meas.requires_grad_(True)
        loss_meas = self._intensity_loss(i_meas)
        loss_meas.backward()  # -> i_meas.grad = ∂ℓ/∂I evaluated at the measured image

        # Weight the model Jacobian ∂I_sim/∂φ by the measured-image loss
        # gradient: the phase gradient is anchored on the real hardware image.
        i_sim = far_field_intensity(self._source_amplitude, self._phase)
        i_sim.backward(gradient=i_meas.grad)

        self._opt.step()
        self._step += 1
        self._loss_history.append(float(loss_meas.detach().cpu()))
        self._record(float(loss_meas.detach().cpu()))
        return self.current_phase