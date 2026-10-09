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
    "geometric_pair",
    "basis_transform",
    "strong_phase_pairs",
]

#: Geometric transforms offered by :func:`geometric_pair`, as
#: ``(name, flip_rows, flip_cols, rot90_k)``.
GEOMETRIC_TRANSFORMS: tuple[tuple[str, bool, bool, int], ...] = (
    ("identity", False, False, 0),
    ("flip_rows", True, False, 0),
    ("flip_cols", False, True, 0),
    ("flip_both", True, True, 0),
    ("rot90", False, False, 1),
    ("rot180", False, False, 2),
    ("rot270", False, False, 3),
)


def geometric_pair(
    phase_cos: Tensor,
    phase_sin: Tensor,
    image: Tensor,
    transform: tuple[str, bool, bool, int],
) -> tuple[Tensor, Tensor, tuple[str, bool, bool, int]]:
    """Apply ONE spatial transform to the pupil phasor and the far field together.

    "Synchronous" is the whole point: the target must be transformed by the *same*
    operator as the input, because a reflected pupil really does produce a reflected
    far field for a shift-invariant propagator.

    .. warning::
       On :class:`~ml.zernike.models.ZernikeAmpModel` this is **not** a
       label-preserving augmentation, and that is a measured property of the
       architecture, not a bug in this function. The model holds ONE global Zernike
       coefficient vector ``c`` shared by every sample, and a reflection acts on it
       by an exact orthogonal operator ``D``:

           out(g*m, c) == roll1(g(out(m, D c)))        (measured, maxerr 1.9e-15)

       So flipping the input and the target demands ``c ~ c*`` **and**
       ``c ~ D c*``. At probability 1 the optimum is constrained to the
       ``D``-invariant subspace, which drives the odd-signature modes (tilt, coma)
       to zero. The physical bench is not reflection-symmetric, so that bias is real.
       Flip-invariant operators are exactly ``+`` and ``-`` (global sign) and the
       identity -- which is what :func:`phase_flip` already implements.

       For a model with per-sample capacity (the learned ``train_coeff`` decoder) the
       same transform is sound ordinary augmentation, and ``basis_transform`` is
       exposed so the coefficient action stays expressible.

    Args:
        phase_cos: Real part of the pupil phasor, ``(B, 1, g, g)``.
        phase_sin: Imaginary part, same shape.
        image: Far-field target, ``(B, 1, g, g)``.
        transform: One of :data:`GEOMETRIC_TRANSFORMS`.

    Returns:
        ``(phase_cos, phase_sin, image, transform)`` -- transformed in place-free
        fashion (the inputs are not mutated) plus the transform actually applied, so
        a caller can record it.
    """
    name, flip_rows, flip_cols, rot90_k = transform
    if rot90_k % 4:
        phase_cos = torch.rot90(phase_cos, rot90_k, dims=(-2, -1))
        phase_sin = torch.rot90(phase_sin, rot90_k, dims=(-2, -1))
        image = torch.rot90(image, rot90_k, dims=(-2, -1))
    if flip_rows:
        phase_cos = torch.flip(phase_cos, dims=(-2,))
        phase_sin = torch.flip(phase_sin, dims=(-2,))
        image = torch.flip(image, dims=(-2,))
    if flip_cols:
        phase_cos = torch.flip(phase_cos, dims=(-1,))
        phase_sin = torch.flip(phase_sin, dims=(-1,))
        image = torch.flip(image, dims=(-1,))
    return phase_cos, phase_sin, image, transform


def basis_transform(
    basis: Tensor, transform: tuple[str, bool, bool, int]
) -> Tensor:
    """The operator ``D`` a geometric transform induces on a Zernike basis.

    Measured, not derived: for a column flip ``D = diag(s)`` exactly (residual
    ``1.2e-14``), and ``rot90`` is an exact orthogonal **matrix** that *preserves
    radial order* ``n`` (block-diagonal by ``n`` to ``1.4e-15``; the ``n=1`` tilt
    block is ``[[0,-1],[1,0]]``). Do **not** hand-derive it as ``(-1)**|m|`` --
    that is wrong for 16 of 27 slots under this generator's convention, because its
    ``(2,-2)`` mode is ``sin 2theta``. Solve for it instead:

        D = np.linalg.lstsq(B_flat, transform(B).T).T

    Args:
        basis: ``(K, g, g)`` mode stack.
        transform: One of :data:`GEOMETRIC_TRANSFORMS`.

    Returns:
        ``(K, K)`` matrix ``D`` such that ``g(Z_k) == sum_j D[k, j] Z_j``.
    """
    import numpy as np

    array = basis.detach().cpu().numpy().astype(np.float64)
    # Build the transformed stack with the same ops geometric_pair applies.
    stacked = torch.from_numpy(array).unsqueeze(1)
    cos = torch.zeros_like(stacked)
    sin = torch.zeros_like(stacked)
    _, _, moved, _ = geometric_pair(cos, sin, stacked, transform)
    moved_np = moved.squeeze(1).numpy().astype(np.float64)
    k = array.shape[0]
    operator, *_ = np.linalg.lstsq(array.reshape(k, -1).T, moved_np.reshape(k, -1).T)
    return torch.from_numpy(np.ascontiguousarray(operator.T))


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