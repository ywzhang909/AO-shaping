"""Data augmentation for the forward model, as small pure functions.

Motivation from measurement rather than intuition. The forward model's held-out gap is
essentially zero (``val/train = 1.08``) and its learning curve is flat, so augmentation
cannot be expected to improve in-domain R2 -- there is no gap to close. What it *can*
address is the direction that is actually broken: the model cannot be inverted, and the
diagnosis is that it has only ever seen mild bench aberrations mapping to broad spots, so a
strong phase such as a Gerchberg-Saxton proposal is out of distribution.

Two families, therefore:

``phasor_noise``
    Small additive noise on the input phasor. A regulariser for sensor-level jitter.

``strong_phase_pairs``
    Synthetic ``(phasor, far field)`` pairs from strong phases -- GS proposals over a spread
    of target sizes, large-amplitude random Zernike, and spatially rough random phase. This
    is the out-of-distribution half, and it is the only augmentation in this module
    expected to move the inverse direction.

Everything is seeded explicitly. The model initialises its coefficients to zeros
deterministically, so ``torch.manual_seed`` alone does not make an experiment repeatable;
each augmentation takes a generator so a run can be reproduced exactly.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "phasor_noise",
    "phase_flip",
    "phase_shift",
    "strong_phase_pairs",
]


def phasor_noise(
    phase_cos: Tensor, phase_sin: Tensor, std: float, generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Additive Gaussian noise on the phasor components.

    The phasor is ``(cos, sin)`` with unit modulus, so isotropic noise on the components
    perturbs amplitude and phase together and slightly reduces the modulus -- which is what
    sensor jitter does. No renormalisation is applied: re-normalising would hide exactly the
    amplitude variation this is meant to model.
    """
    if std <= 0.0:
        return phase_cos, phase_sin
    noise_c = torch.randn(phase_cos.shape, generator=generator, device=phase_cos.device, dtype=phase_cos.dtype)
    noise_s = torch.randn(phase_sin.shape, generator=generator, device=phase_sin.device, dtype=phase_sin.dtype)
    return phase_cos + std * noise_c, phase_sin + std * noise_s


def phase_flip(
    phase_cos: Tensor, phase_sin: Tensor, generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Randomly negate the phasor (i.e. add pi to the phase).

    ``-exp(i*phi) = exp(i*(phi+pi))`` is an exact optical identity -- it shifts the spot by
    the half-period and leaves the intensity distribution's *shape* alone. So this is a
    label-preserving invariance rather than a regulariser, which is why it is worth having:
    it doubles the effective sample count without inventing physics.
    """
    sign = torch.randint(
        0, 2, (1,), generator=generator, device=phase_cos.device
    ).to(phase_cos.dtype) * 2 - 1
    return sign * phase_cos, sign * phase_sin


def phase_shift(
    phase_cos: Tensor, phase_sin: Tensor, max_px: int = 2, generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Roll the phase by a small random sub-pixel-integer amount.

    A beam displaced on the SLM displaces its far field. The dataset's own frames are not
    recentred per sample, so a small roll leaves the recorded target valid while teaching
    the model that a translated pupil translates the spot -- the invariance that decides
    whether the model can be *aimed*, and therefore whether it can be inverted.
    """
    if max_px <= 0:
        return phase_cos, phase_sin
    def roll(t: Tensor) -> Tensor:
        dy = int(torch.randint(-max_px, max_px + 1, (1,), generator=generator).item())
        dx = int(torch.randint(-max_px, max_px + 1, (1,), generator=generator).item())
        return torch.roll(t, shifts=(dy, dx), dims=(-2, -1))

    # NOTE: torch.roll wraps the far edge, which is not physical. It is left in rather
    # than tapered because a wrap at max_px <= 2 affects ~6% of the aperture edge, and
    # measuring whether that helps or hurts is an experiment, not an assumption.
    return roll(phase_cos), roll(phase_sin)


def strong_phase_pairs(
    n_samples: int,
    *,
    grid: int,
    beam_w0: float,
    far_field_padding: int,
    aperture_radius: float,
    n_max: int = 20,
    seed: int = 0,
    mix: Literal["gs", "zernike", "freeform"] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Synthetic ``(phasor, far field)`` pairs from STRONG phases.

    Three deliberately different families, because using one would confound the result:
    ``gs``      Gerchberg-Saxton proposals over a spread of target sizes -- the shapes that
                inverse design actually asks for;
    ``zernike`` large-amplitude random coefficients (strong smooth aberrations);
    ``freeform`` uniformly random phase (spatially rough, speckle-like).

    Targets come from the project's own simulator, so the pair is physically consistent by
    construction rather than by assumption.

    Args:
        n_samples: How many pairs to generate.
        grid: Pupil side length; the far field is centre-cropped back to this.
        beam_w0: Simulator illumination waist, in pupil pixels.
        far_field_padding: Zero-padding factor; sets the far field's angular scale.
        aperture_radius: Illuminated radius in pixels; phase is zeroed outside it.
        n_max: Zernike order for the ``zernike`` family.
        seed: Seeds an internal numpy generator when ``generator`` is not given.
        mix: Force a single family instead of cycling through all three.
        generator: Optional torch generator, for exact reproducibility.

    Returns:
        ``(phasor, far_field)`` with shapes ``(N, 2, g, g)`` and ``(N, 1, g, g)``. The
        far field is peak-normalised per sample, matching the incumbent's ``"peak"`` mode.
    """
    from ml.zernike.inverse_design import coefficients_to_phase, gs_phase, sim_far_field

    rng = np.random.default_rng(seed)
    families = [mix] if mix else ["gs", "zernike", "freeform"]

    phasor = torch.zeros(n_samples, 2, grid, grid, dtype=torch.float32)
    target = torch.zeros(n_samples, 1, grid, grid, dtype=torch.float32)

    yy, xx = np.mgrid[0:grid, 0:grid]
    radius = grid / 2.0
    disc = ((yy + 0.5 - radius) ** 2 + (xx + 0.5 - radius) ** 2) <= aperture_radius**2

    for i in range(n_samples):
        kind = families[i % len(families)]
        if kind == "gs":
            sf = float(rng.uniform(0.25, 0.55))
            aspect = float(rng.choice([1.0, 4 / 3, 1.5]))
            phase = gs_phase(sf, aspect, iterations=30)
        elif kind == "zernike":
            n_modes = n_max * (n_max + 3) // 2
            phase = coefficients_to_phase(rng.normal(0.0, 3.0, n_modes), n_max=n_max)
        else:
            phase = rng.uniform(-np.pi, np.pi, (grid, grid))
        phase = np.where(disc, phase, 0.0)

        phasor[i, 0] = torch.as_tensor(np.cos(phase), dtype=torch.float32)
        phasor[i, 1] = torch.as_tensor(np.sin(phase), dtype=torch.float32)
        far = sim_far_field(phase, padding=far_field_padding)
        peak = float(far.max())
        target[i, 0] = torch.as_tensor(
            far / peak if peak > 0 else far, dtype=torch.float32
        )
    return phasor, target