"""Low-dimensional Zernike-block adaptation tier (issue #229 step 4).

The existing ``configure_adaptation`` has two tiers only:

- ``alignment`` — ``origin`` + ``center`` (geometry/illumination drift);
- ``full`` — every parameter, including all seven residual Noll modes.

When transfer leaves only a *few* modal mismatches, ``full`` over-fits:
eleven-plus parameters against a handful of directional samples, with the
parameters compensating each other (the identifiability check demanded by
#221 stage D). ``alignment`` ignores aberrations entirely. This module adds
the missing middle tier, ``zernike_block``: release the residual modes that
the structured-probe identifiability gate (``structured_probes.
cross_response_matrix`` + ``direction_agreement``) selected, and only those,
plus the illumination/origin pair, under a block-level L2 trust region of
0.5 rad.

Regression safety
-----------------
``alignment`` and ``full`` behave bit-for-bit as before (the original
``configure_adaptation`` is untouched); the block clamp applies *only* when
the caller opts into this tier.
"""
from __future__ import annotations

import numpy as np

#: Block-level trust region (rad): L2 norm of the released residual
#: coefficient vector. #221 step A measured this magnitude dominating
#: generalization on this bench.
BLOCK_L2_CAP_RAD = 0.5

#: Identifiability thresholds (plan §4.3): per-mode direction-agreement
#: floor and cross-response matrix quality gates.
DIRECTION_AGREEMENT_FLOOR = 0.5
CROSS_RATIO_FLOOR = 3.0
CONDITION_NUMBER_CEIL = 50.0


def _torch():
    import torch

    return torch


def configure_zernike_block(
    model,
    modes: list[int],
    *,
    release_amplitude: bool = True,
    release_defocus: bool = False,
) -> None:
    """Release a *subset* of residual Noll modes plus geometry/illumination.

    ``model`` follows the ``PhysicalForwardFit`` surface: ``d``, ``origin``,
    ``center``, ``log_width`` parameters and a single 7-d ``residual``
    parameter indexed by ``model.residual_modes``. ``modes`` is the set of
    Noll indices that passed the identifiability gate.

    Modes outside ``modes`` are kept stationary by masking the residual
    gradient after each backward pass (see :func:`mask_residual_gradient`),
    so the shared 7-d tensor can carry one optimizer group.
    """
    if not modes:
        raise ValueError("zernike_block requires a nonempty selected-mode set")
    unknown = [m for m in modes if m not in model.residual_modes]
    if unknown:
        raise ValueError(f"modes {unknown} not in residual_modes {model.residual_modes}")
    model.d.requires_grad_(release_defocus)
    model.log_width.requires_grad_(release_amplitude)
    model.center.requires_grad_(True)
    model.origin.requires_grad_(True)
    # All-or-nothing on the shared tensor: the gradient mask (below) makes
    # unselected modes stationary even though the tensor carries a gradient.
    model.residual.requires_grad_(True)


def residual_grad_mask(model, modes: list[int]) -> np.ndarray:
    """Bool mask over ``model.residual`` indexing the released modes."""
    mask = np.zeros(len(model.residual_modes), dtype=bool)
    order = list(model.residual_modes)
    for mode in modes:
        mask[order.index(mode)] = True
    return mask


def clamp_residual_block(model, cap_rad: float = BLOCK_L2_CAP_RAD) -> float:
    """L2-project the residual vector so the block stays under ``cap_rad``.

    Applied inside ``torch.no_grad()`` after the optimizer step. Callers of
    the historical tiers never call this, so their numerics are untouched.
    Returns the pre-clamp L2 norm (for logging).
    """
    torch = _torch()
    norm = float(model.residual.detach().norm())
    if norm > cap_rad:
        with torch.no_grad():
            model.residual.mul_(cap_rad / norm)
    return norm


def mask_residual_gradient(model, mask: np.ndarray) -> None:
    """Zero the gradient of unselected residual modes (call after backward)."""
    torch = _torch()
    if model.residual.grad is None:
        return
    with torch.no_grad():
        m = torch.as_tensor(mask, device=model.residual.device)
        model.residual.grad[~m] = 0.0


def masked_fit_step(model, mask: np.ndarray) -> None:
    """Optimizer step for the shared residual group with the mask applied.

    ``model.optimizer`` may contain several groups (d / origin / center /
    log_width / residual); the mask only affects the residual group, which
    ``configure_zernike_block`` leaves on the residual's historical lr 0.03.
    """
    for group in model.optimizer.param_groups:
        if any(p is model.residual for p in group["params"]):
            mask_residual_gradient(model, mask)
    model.optimizer.step()


def identifiability_gate(
    matrix: np.ndarray,
    correlations: dict[int, float],
    *,
    direction_floor: float = DIRECTION_AGREEMENT_FLOOR,
    cross_ratio_floor: float = CROSS_RATIO_FLOOR,
    condition_ceil: float = CONDITION_NUMBER_CEIL,
) -> dict:
    """Per-mode + matrix-level identifiability verdict (plan §4.3, §6.4).

    ``matrix`` is the cross-response matrix of :func:`structured_probes.
    cross_response_matrix` (rows/columns in sorted mode order) and
    ``correlations`` maps mode -> |Pearson r| between the model's finite-
    difference direction and the measured +A difference. A mode passes the
    per-mode gate when its |r| >= ``direction_floor``. The matrix gate
    (diagonal/off-diagonal norm ratio and condition number) must pass for the
    *selected* sub-matrix before the set is usable as a ``zernike_block``
    tier.

    Returns ``{"selected": [...], "rejected": {...}, "matrix_ok": bool,
    "diag_off_ratio": float, "condition_number": float}``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    modes = sorted(correlations)
    if matrix.shape != (len(modes), len(modes)):
        raise ValueError("matrix must be square and match the number of correlations")
    per_mode = {m: (abs(float(r)) >= direction_floor) for m, r in correlations.items()}
    selected = [m for m in modes if per_mode[m]]
    rejected = {m: abs(float(r)) for m, r in correlations.items() if not per_mode[m]}
    if len(selected) < 2:
        return dict(selected=selected, rejected=rejected, matrix_ok=False,
                    diag_off_ratio=float("nan"), condition_number=float("nan"))
    order = {m: i for i, m in enumerate(modes)}
    idx = [order[m] for m in selected]
    sub = np.array([[matrix[i, j] for j in idx] for i in idx], dtype=np.float64)
    diag_norm = float(np.linalg.norm(np.diag(sub)))
    off = sub - np.diag(np.diag(sub))
    off_norm = float(np.linalg.norm(off))
    ratio = diag_norm / off_norm if off_norm > 1e-12 else float("inf")
    cond = float(np.linalg.cond(sub))
    ok = (ratio >= cross_ratio_floor) and (cond <= condition_ceil)
    return dict(selected=selected, rejected=rejected, matrix_ok=bool(ok),
                diag_off_ratio=ratio, condition_number=cond)
