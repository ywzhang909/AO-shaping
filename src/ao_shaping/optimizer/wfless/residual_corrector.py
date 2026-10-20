"""Zero-initialised low-capacity residual corrector for the model-in-loop forward model.

This module adds an optional residual term ``R(phase, I_phys)`` on top of the
Step A physical forward model used by the model-in-loop probe-scoring path
(:mod:`ao_shaping.optimizer.wfless.model_in_loop_shaping`).  The corrector is
ZERO-INITIALISED (final 1x1 conv weight and bias are exactly zero), so a freshly
constructed :class:`CorrectedStepAOptimizer` reproduces the wrapped optimizer's
:meth:`forward_intensity` BIT-IDENTICALLY until a corrector fit deliberately
perturbs it.  That invariant is the acceptance regression test for this module.

Capacity is kept far below the 121 177-parameter U-Net that was shown to be
useless in this role (Pearson 0.018 on 44 phases): the default corrector has
65 parameters (asserted < 5000 by the test suite).

Note:
    ``torch`` is imported at module top level, following the convention of the
    other optimizer-layer modules that define ``nn.Module`` subclasses (e.g.
    :mod:`ao_shaping.optimizer.rl.sac_common`).  PyTorch is the optional ``ml``
    dependency group (``uv sync --group ml``).  Nothing in this module touches
    hardware.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ao_shaping.optimizer.wfless.model_in_loop_shaping import _StepAOptimizer

__all__ = ["ResidualCorrector", "CorrectedStepAOptimizer"]


class ResidualCorrector(nn.Module):
    """``R(phase, I_phys) -> residual intensity`` of the same shape.

    Architecture: 3x3 average-pool (stride 1, padding 1) -> 1x1 conv
    ``2 -> hidden`` -> ReLU -> 1x1 conv ``hidden -> 1`` on a ``(B, 2, H, W)``
    input (channel concat of ``phase`` and ``i_phys``).  The LAST conv's weight
    and bias are initialised with :func:`torch.nn.init.zeros_`, so the output
    is EXACTLY zero at initialisation and ``I_phys + R(...) == I_phys`` holds
    bit-identically.

    Args:
        hidden: Width of the single 1x1 convolution. Must be >= 1.

    Raises:
        ValueError: If ``hidden`` is smaller than 1.
    """

    def __init__(self, *, hidden: int = 16) -> None:
        super().__init__()
        hidden = int(hidden)
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        self.hidden = hidden
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.project = nn.Conv2d(2, hidden, kernel_size=1)
        self.activate = nn.ReLU()
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, phase: torch.Tensor, i_phys: torch.Tensor) -> torch.Tensor:
        """Evaluate the residual for a batch of (phase, I_phys) feature maps.

        Args:
            phase: ``(B, 1, H, W)`` SLM phase in radians (float or float32).
            i_phys: ``(B, 1, H, W)`` peak-normalised physical intensity.

        Returns:
            ``(B, 1, H, W)`` float32 residual intensity. Exactly zero while
            the head conv is at its zero initialisation.

        Raises:
            ValueError: If the inputs are not 4D with 1 channel and matching
                spatial shapes.
        """
        if phase.ndim != 4 or i_phys.ndim != 4:
            raise ValueError(
                f"phase and i_phys must be 4D (B, 1, H, W), got "
                f"{phase.ndim}D and {i_phys.ndim}D"
            )
        if phase.shape[1] != 1 or i_phys.shape[1] != 1:
            raise ValueError(
                f"inputs must have a single channel, got {phase.shape[1]} and "
                f"{i_phys.shape[1]}"
            )
        if phase.shape[2:] != i_phys.shape[2:]:
            raise ValueError(
                f"spatial shapes must match, got {tuple(phase.shape[2:])} and "
                f"{tuple(i_phys.shape[2:])}"
            )
        x = torch.cat([phase.to(torch.float32), i_phys.to(torch.float32)], dim=1)
        x = self.pool(x)
        return self.head(self.activate(self.project(x)))

    def count_parameters(self) -> int:
        """Return the number of trainable parameters.

        Returns:
            ``sum(p.numel() for p in self.parameters())`` — 65 for the
            default ``hidden=16`` configuration.
        """
        return sum(p.numel() for p in self.parameters())


class CorrectedStepAOptimizer:
    """Wrap a ``_StepAOptimizer``-compatible optimizer, adding the residual term.

    ``forward_intensity`` returns ``I_phys + R(phase, I_phys)`` where ``I_phys``
    comes from the wrapped optimizer's uncorrected forward.  The class satisfies
    the :class:`ao_shaping.optimizer.wfless.model_in_loop_shaping._StepAOptimizer`
    protocol (``coefficients`` / ``last_loss`` / ``converged`` / ``update``) by
    delegating to the inner optimizer.

    Corrector lifecycle (deliberate design, documented contract):

    * :meth:`update` only drives the inner optimizer — the corrector is FROZEN
      during per-iteration fitting.  It never changes the corrector's
      parameters.
    * The corrector is refined ONCE PER SESSION through the explicit, optional
      :meth:`fit_corrector`, which is not called by :meth:`update`.

    Bit-identity guarantee: :meth:`forward_intensity` computes the residual
    under ``torch.no_grad()``; when the residual is EXACTLY zero (the state at
    zero initialisation, or before :meth:`fit_corrector` has ever moved the
    head conv), the inner optimizer's float64 array is returned UNCHANGED — no
    float32 round-trip is applied — so the corrected and uncorrected forwards
    are ``np.array_equal``.  Only a genuinely non-zero residual is added, and
    then it is widened to the inner array's dtype first.

    Args:
        inner: The wrapped Step A optimizer (any object satisfying the
            ``_StepAOptimizer`` protocol plus a ``forward_intensity`` method).
        corrector: Optional pre-built :class:`ResidualCorrector`; a fresh
            (zero-initialised) one is constructed when omitted.
    """

    def __init__(
        self,
        inner: _StepAOptimizer,
        corrector: ResidualCorrector | None = None,
    ) -> None:
        self._inner = inner
        self._corrector = corrector if corrector is not None else ResidualCorrector()

    @property
    def inner(self) -> _StepAOptimizer:
        """The wrapped optimizer."""
        return self._inner

    @property
    def corrector(self) -> ResidualCorrector:
        """The residual corrector applied in the probe-scoring path."""
        return self._corrector

    @property
    def coefficients(self) -> npt.NDArray[np.floating]:
        """The inner optimizer's current coefficient vector, radians."""
        return self._inner.coefficients

    @property
    def last_loss(self) -> float | None:
        """Inner loss at the last step, or ``None`` before the first one."""
        return self._inner.last_loss

    @property
    def converged(self) -> bool:
        """Whether the inner optimizer's plateau detector has latched."""
        return self._inner.converged

    def forward_intensity(
        self,
        coefficients: npt.NDArray[np.floating],
        phase_slm: npt.NDArray[np.floating],
        source_amplitude: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Uncorrected far field plus the residual ``R(phase, I_phys)``.

        Args:
            coefficients: 1D Zernike coefficient vector, radians.
            phase_slm: 2D raw (unwrapped) SLM probe phase, radians, on the
                pupil grid.
            source_amplitude: Optional 2D amplitude override passed straight
                to the inner optimizer.

        Returns:
            The inner ``forward_intensity`` result (same shape and dtype) when
            the residual is exactly zero — bit-identical to the uncorrected
            forward — otherwise ``inner + residual`` with the float32 residual
            widened to the inner dtype.
        """
        i_phys = self._inner.forward_intensity(
            coefficients, phase_slm, source_amplitude
        )
        residual = self._residual_for(phase_slm, i_phys)
        if residual is None:
            return i_phys
        return i_phys + residual.astype(i_phys.dtype, copy=False)

    def update(
        self, i_meas: npt.NDArray[np.floating], phase_slm: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Take one inner fitting step; the corrector stays FROZEN.

        Delegates directly to ``inner.update``.  The corrector's parameters
        are never touched here — refinement happens once per session via
        :meth:`fit_corrector`, keeping the probe-scoring residual stable while
        coefficients are fitted.

        Args:
            i_meas: 2D measured far-field intensity.
            phase_slm: 2D raw (unwrapped) SLM probe phase, radians.

        Returns:
            The inner optimizer's updated coefficient vector (numpy, radians).
        """
        return self._inner.update(i_meas, phase_slm)

    def fit_corrector(
        self,
        frames: Sequence[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]],
        *,
        epochs: int = 1,
        lr: float = 1e-3,
    ) -> None:
        """Refine the corrector ONCE PER SESSION against probe frames.

        Optional entry point — deliberately never called by :meth:`update`.
        For each ``(i_meas, phase_slm)`` frame it regresses the residual so
        that ``I_phys + R(phase, I_phys)`` approaches the measured frame at the
        inner optimizer's current coefficients.  Both frames are peak-normalised
        (divided by their peak) so frame scale never enters the residual; a
        constant-zero residual keeps the peak-normalised physical intensity.

        Args:
            frames: Sequence of ``(i_meas, phase_slm)`` pairs.
            epochs: Number of passes over ``frames``. Must be >= 1.
            lr: Adam learning rate for the corrector parameters.

        Raises:
            ValueError: If ``frames`` is empty, or ``epochs`` / ``lr`` invalid.
        """
        if not frames:
            raise ValueError("frames must contain at least one (i_meas, phase_slm) pair")
        epochs = int(epochs)
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        lr = float(lr)
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError(f"lr must be a positive finite number, got {lr}")

        optimizer = torch.optim.Adam(self._corrector.parameters(), lr=lr)
        for _ in range(epochs):
            for i_meas, phase_slm in frames:
                i_phys = self._inner.forward_intensity(self.coefficients, phase_slm)
                phase_t, i_phys_t = self._feature_tensors(phase_slm, i_phys)
                target_t = self._peak_normalised_tensor(i_meas, i_phys_t.shape[-2:])
                with torch.enable_grad():
                    residual = self._corrector(phase_t, i_phys_t)
                    loss = torch.mean((i_phys_t + residual - target_t) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    def _residual_for(
        self,
        phase_slm: npt.NDArray[np.floating],
        i_phys: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating] | None:
        """Compute the float32 residual grid, or ``None`` if it is exactly zero.

        Args:
            phase_slm: 2D raw SLM probe phase, radians.
            i_phys: 2D uncorrected far-field intensity from the inner model.

        Returns:
            ``(H, W)`` float32 residual array, or ``None`` when every element
            is exactly zero — the caller then returns the inner array untouched
            to preserve bit-identity.
        """
        phase_t, i_phys_t = self._feature_tensors(phase_slm, i_phys)
        with torch.no_grad():
            residual = self._corrector(phase_t, i_phys_t)[0, 0]
        if not torch.any(residual):
            return None
        return residual.cpu().numpy()

    @staticmethod
    def _feature_tensors(
        phase_slm: npt.NDArray[np.floating],
        i_phys: npt.NDArray[np.floating],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build aligned ``(1, 1, H, W)`` float32 feature tensors.

        When the probe phase grid is smaller than the far-field grid (the
        inner model zero-pads before the FFT), the phase is upsampled by exact
        integer block repetition so both features share the scoring grid.

        Args:
            phase_slm: 2D phase array, radians.
            i_phys: 2D far-field intensity array.

        Returns:
            ``(phase_t, i_phys_t)`` float32 tensors of shape ``(1, 1, H, W)``
            with identical spatial shape.

        Raises:
            ValueError: If inputs are not 2D finite, or the grids are not
                integer multiples of each other.
        """
        phase = np.asarray(phase_slm, dtype=np.float64)
        intensity = np.asarray(i_phys, dtype=np.float64)
        if phase.ndim != 2 or intensity.ndim != 2:
            raise ValueError(
                f"phase_slm and i_phys must be 2D, got {phase.ndim}D and "
                f"{intensity.ndim}D"
            )
        if not np.all(np.isfinite(phase)) or not np.all(np.isfinite(intensity)):
            raise ValueError("phase_slm and i_phys must be finite")
        phase_t = torch.from_numpy(
            np.ascontiguousarray(phase, dtype=np.float32)
        ).view(1, 1, *phase.shape)
        i_phys_t = torch.from_numpy(
            np.ascontiguousarray(intensity, dtype=np.float32)
        ).view(1, 1, *intensity.shape)
        if phase_t.shape[-2:] != i_phys_t.shape[-2:]:
            h_out, w_out = i_phys_t.shape[-2:]
            h_in, w_in = phase_t.shape[-2:]
            if h_out % h_in or w_out % w_in:
                raise ValueError(
                    f"phase grid {phase.shape} cannot be upsampled to the "
                    f"far-field grid {intensity.shape} by integer factors"
                )
            phase_t = phase_t.repeat_interleave(h_out // h_in, dim=-2)
            phase_t = phase_t.repeat_interleave(w_out // w_in, dim=-1)
        return phase_t, i_phys_t

    @staticmethod
    def _peak_normalised_tensor(
        values: npt.NDArray[np.floating], grid: tuple[int, int]
    ) -> torch.Tensor:
        """Peak-normalise a 2D frame to float32 ``(1, 1, *grid)``.

        Args:
            values: 2D measured intensity array.
            grid: Target spatial shape; the frame must already match it.

        Returns:
            The frame divided by its peak as a float32 tensor (zeros stay
            zero when the peak is not positive).

        Raises:
            ValueError: If the array is not 2D finite or its shape is not
                ``grid``.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 2 or not np.all(np.isfinite(arr)):
            raise ValueError("measured frames must be 2D and finite")
        if tuple(arr.shape) != tuple(grid):
            raise ValueError(
                f"frame shape {arr.shape} must match the far-field grid {grid}"
            )
        peak = float(arr.max()) if arr.size else 0.0
        if np.isfinite(peak) and peak > 0.0:
            arr = arr / peak
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).view(
            1, 1, *arr.shape
        )
