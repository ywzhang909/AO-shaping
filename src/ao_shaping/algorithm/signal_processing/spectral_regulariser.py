"""Band-limit ("MTF-matched") regularisation for the closed-loop phase increment.

**The failure this targets.** The frozen forward model predicts the far field
from the *control grid* as a continuous field. The SLM renders the same grid as
a piecewise-constant staircase on the panel (``capture.to_panel`` repeats each
control pixel by the panel pitch, 3 px at ``--control-grid 640``). The staircase
adds harmonics the model never sees, and those harmonics diffract out of the
target box as speckle. Phase content the model rewards but the device turns into
scatter is unrecoverable: no amount of closed-loop iteration removes it.

**What is penalised.** The *out-of-band fraction* of the increment's spectral
power:

    penalty = Σ |FFT(delta)|² · (1 - mask) / Σ |FFT(delta)|²

The fraction is scale-free, so it does not fight the RMS cap or the step size;
it only decides *where* the increment is allowed to put energy.

**Why the mask is a parameter and not a constant.** A true MTF-matched mask needs
a measured device MTF, which this bench does not have yet (the issue that
proposed this term says so explicitly). The default is therefore a conservative
*band limit*, not a claimed MTF: it keeps the lower 70% of the control-grid
band and rolls off to zero at the control-grid Nyquist. Once a measured MTF
exists, pass it through ``cutoff``/``rolloff``; the formula stays the same.

Nothing here imports a device: it is numpy/torch array math on the phase grid.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_NYQUIST_FRACTION",
    "DEFAULT_ROLLOFF",
    "control_grid_cutoff",
    "mtf_matched_mask",
    "spectral_penalty",
]

#: Default cutoff as a fraction of the control-grid Nyquist (0.5 cycles/px).
#: 0.7 keeps structure coarser than ~2.9 control px and leaves the target box's
#: own bandwidth (its first Fourier zero sits near ``2/side``) untouched.
DEFAULT_NYQUIST_FRACTION = 0.7
#: Width of the cosine roll-off, in cycles per control pixel.
DEFAULT_ROLLOFF = 0.1


def control_grid_cutoff(grid_size: int, *, nyquist_fraction: float = DEFAULT_NYQUIST_FRACTION) -> float:
    """Cutoff in cycles per control pixel for a square ``grid_size`` grid."""
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    if not 0 < nyquist_fraction <= 1:
        raise ValueError("nyquist_fraction must be in (0, 1]")
    return 0.5 * float(nyquist_fraction)


def mtf_matched_mask(size: int, *, cutoff: float, rolloff: float = DEFAULT_ROLLOFF) -> np.ndarray:
    """Radial passband mask, ``1`` inside ``cutoff`` and ``0`` beyond the roll-off.

    Frequencies are in cycles per control pixel, ``|f| <= 0.5``. The transition
    is a raised cosine over ``[cutoff, cutoff + rolloff]`` so a hard edge does
    not ring in the phase.
    """
    if size < 2:
        raise ValueError("size must be >= 2")
    if not 0 < cutoff < 0.5:
        raise ValueError("cutoff must be in (0, 0.5) cycles/control-px")
    if rolloff < 0:
        raise ValueError("rolloff must be non-negative")
    if cutoff + rolloff > 0.5 + 1e-12:
        raise ValueError("cutoff + rolloff must not exceed the 0.5 cycles/control-px Nyquist")
    freq = np.fft.fftfreq(size)
    fy, fx = np.meshgrid(freq, freq, indexing="ij")
    radius = np.hypot(fx, fy)
    outer = cutoff + rolloff
    mask = np.ones_like(radius, dtype=np.float64)
    if rolloff <= 0:
        mask[radius > cutoff] = 0.0
        return mask.astype(np.float32)
    transition = (radius > cutoff) & (radius < outer)
    mask[transition] = 0.5 * (1.0 + np.cos(np.pi * (radius[transition] - cutoff) / rolloff))
    mask[radius >= outer] = 0.0
    return mask.astype(np.float32)


def spectral_penalty(delta_phase, mask) -> "object":
    """Out-of-band power fraction of ``delta_phase`` as a differentiable scalar.

    ``delta_phase`` may be a torch tensor (it is, inside the optimizer) or any
    array-like. The returned value requires grad when the input does, so it can
    be added to a loss and back-propagated.
    """
    import torch

    phase = delta_phase if torch.is_tensor(delta_phase) else torch.as_tensor(delta_phase)
    band = torch.as_tensor(np.asarray(mask), dtype=phase.dtype, device=phase.device)
    if band.shape != phase.shape:
        raise ValueError(f"mask shape {tuple(band.shape)} != phase shape {tuple(phase.shape)}")
    spectrum = torch.fft.fft2(phase)
    power = spectrum.real ** 2 + spectrum.imag ** 2
    total = power.sum().clamp_min(1e-12)
    return (power * (1.0 - band)).sum() / total
