"""Structured single-mode Zernike active probes (issue #229 step 3).

Why this module exists
----------------------
``probes.py::probe_grid`` draws *random* phase families (mixed 7-mode Zernike /
smooth / bandlimited). That is the wrong sample design for a low-dimensional
modal parameter space: single-modal responses get drowned by cross-modal
coupling, the Jacobian rows are dense, and ``direction_agreement`` cannot be
attributed to a specific mode (#221 measured exactly that: direction-cosine
median 0.0288 on random phases, 96/96 guard blocks ``contradict``).

This module emits **one mode at a time** with a sign-paired amplitude ladder,
which is the identifiability design of the transfer plan
(``report/shape_trials/physical_refine/forward_model_transfer_plan_20261104.md``
§4.3): the per-mode difference ``I(+A) - I(0)`` is the model's finite-difference
Jacobian row for that mode, and the M×M matrix of such rows is the cross-
response matrix used to select the ``zernike_block`` fit tier.

Canonical entry point for phase generation is
``ao_shaping.utils.wavefront.zernike_utils.generate_zernike_phase`` (raw
radians). The only ``mod 2π`` in the whole pipeline lives in
``Santec.create_phase_from_array``; this module never wraps.
"""
from __future__ import annotations

import numpy as np

from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

#: Default amplitude ladder (rad). 0.02 for high-sensitivity linear-region
#: calibration, 0.10 to give low-sensitivity high-order modes signal.
AMPLITUDE_LADDER: tuple[float, ...] = (0.02, 0.05, 0.10)

#: High-order modal family: Z5..Z11. Piston/tip/tilt are geometry (separate
#: ramp probe) and Noll 4 defocus is the independent scalar ``d`` (pre-flight
#: gate) — none of them spend this budget.
DEFAULT_MODES: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11)


def structured_probe_plan(
    modes: tuple[int, ...] = DEFAULT_MODES,
    amplitudes: tuple[float, ...] = AMPLITUDE_LADDER,
    repeats: int = 3,
    holdout_mode: int | None = 9,
    holdout_amplitude_index: int | None = None,
) -> list[dict]:
    """Enumerate the single-mode ±Zernike probe grid.

    Returns one row per (mode, amplitude, sign, repeat) with a ``split`` of
    ``train`` / ``validation`` / ``holdout`` following the plan:

    - the *highest* amplitude level of every mode is ``validation`` (whole
      level, not a scatter of points);
    - ``holdout_mode`` (default Noll 9) is ``holdout`` for all its points
      (an entire mode that never enters training);
    - holdout wins over validation when both apply.

    Each ABBA point is 4 frames, so the frame budget is
    ``len(plan) * 4 + 8`` (8 = two ABBA flat-field references).
    """
    if holdout_amplitude_index is None:
        holdout_amplitude_index = len(amplitudes) - 1
    if not modes:
        raise ValueError("at least one mode required")
    if not amplitudes:
        raise ValueError("at least one amplitude required")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if holdout_mode is not None and holdout_mode not in modes:
        raise ValueError(f"holdout mode {holdout_mode} not in {modes}")

    plan: list[dict] = []
    for mode in modes:
        for k, amplitude in enumerate(amplitudes):
            for sign in (1, -1):
                for repeat in range(repeats):
                    if mode == holdout_mode:
                        split = "holdout"
                    elif k == holdout_amplitude_index:
                        split = "validation"
                    else:
                        split = "train"
                    plan.append(dict(
                        mode=int(mode),
                        amplitude=float(amplitude),
                        sign=int(sign),
                        repeat=int(repeat),
                        split=split,
                        label=f"zn{mode:02d}_{sign:+d}{abs(amplitude):.2f}_r{repeat}",
                    ))
    return plan


def structured_phase(row: dict, size: int) -> np.ndarray:
    """Control-grid phase for one plan row (raw radians, never wrapped)."""
    coefficients = {row["mode"]: float(row["sign"]) * float(row["amplitude"])}
    phase = generate_zernike_phase(coefficients, (size, size))
    return np.nan_to_num(np.asarray(phase, dtype=np.float32), nan=0.0)


def flat_field_plan(n_repeats: int = 2) -> list[dict]:
    """ABBA flat-field references: 2 × (A B B A) = 8 frames total."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    plan: list[dict] = []
    for repeat in range(n_repeats):
        for position in range(4):
            plan.append(dict(label=f"flat_ref_r{repeat}_pos{position}", split="train"))
    return plan


def frame_budget(
    modes: tuple[int, ...] = DEFAULT_MODES,
    amplitudes: tuple[float, ...] = AMPLITUDE_LADDER,
    repeats: int = 3,
    flat_abba: int = 2,
) -> int:
    """Total CCD frames for the structured campaign (ABBA = 4 frames/point)."""
    probes = len(modes) * len(amplitudes) * 2 * repeats
    return probes * 4 + flat_abba * 4


def random_probe_budget_match(
    n_modes: int,
    n_frames: int,
    size: int,
    seed: int,
) -> list[np.ndarray]:
    """Frame-budget-matched random control (issue #229 core comparison).

    The revision of #229's comparison design: instead of splitting the budget
    three ways (1:1:1 across the zernike/smooth/bandlimited families), the
    random arm gets **one random 7-mode Zernike combination per modal
    direction** — ``n_modes`` phases with the same *total frame count* as the
    structured arm. Both arms then run the same exposure/ABBA/verdict chain.
    """
    from scipy.ndimage import zoom

    if n_frames % n_modes:
        raise ValueError("n_frames must be divisible by n_modes")
    per_mode = n_frames // n_modes
    rng = np.random.default_rng(seed)
    phases: list[np.ndarray] = []
    for index in range(n_modes):
        coefficients = {j: float(rng.normal(0, 0.6)) for j in range(4, 12)}
        phase = generate_zernike_phase(coefficients, (size, size))
        coarse = zoom(rng.normal(0, 1, (24, 24)), size / 24, order=1)
        coarse *= 1.5 / max(float(coarse.std()), 1e-9)
        # One mixed random combination per mode index, repeated to fill the
        # frame budget (repeats do not create new information — that is the
        # point of the comparison).
        base = 0.5 * (
            np.nan_to_num(phase, nan=0.0) + coarse
        ).astype(np.float32)
        for _ in range(per_mode):
            phases.append(base + rng.normal(0, 0.02, (size, size)).astype(np.float32))
    return phases


def cross_response_matrix(
    base: np.ndarray,
    modal: dict[int, np.ndarray],
    amplitude_scale: float = 1.0,
) -> tuple[np.ndarray, list[int]]:
    """M×M cross-response matrix from per-mode +A difference images.

    ``base`` is the flat-phase intensity and ``modal`` maps mode -> ``I(+A)``
    on the identical grid. Diagonal entries are the per-mode main responses
    (normalized by their own RMS); off-diagonal entries are cross-leakage
    normalized the same way, so the diagonal/off-diagonal norm ratio and the
    condition number are scale-free.

    Returns ``(matrix, mode_order)``.
    """
    modes = sorted(modal)
    n = len(modes)
    if n == 0:
        raise ValueError("no modal responses supplied")
    rows = []
    for mode in modes:
        delta = (np.asarray(modal[mode], dtype=np.float64)
                 - np.asarray(base, dtype=np.float64)).ravel()
        rows.append(delta)
    stack = np.stack(rows)
    norms = np.linalg.norm(stack, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    stacked = stack / norms * float(amplitude_scale)
    corr = (stacked @ stacked.T)
    # Symmetrize; diagonal is 1 by construction, off-diagonal is the cosine of
    # the angle between two modal response directions (cross-leakage).
    matrix = 0.5 * (corr + corr.T)
    return matrix, modes
