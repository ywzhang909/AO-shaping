"""Shared SLM+Zernike closed-loop primitives.

``slm_zernike_pib`` and ``slm_zernike_shaping`` are two *target families* of one
hardware closed loop (Zernike coefficients -> SLM phase -> CCD measurement), and
they shipped byte-identical copies of five helpers. Those copies are what
``TODO.md:98`` calls a 同源副本, and ``docs/dev/todo_archive.md`` §5.2 records the
price: R-1~R-4 and R-9~R-11 each had to be fixed on *both* sides.

What lives here is the *how we drive the hardware* layer only. The objective
maths stays per-family -- ``ShapingObjective`` for the PIB family,
``square_quality_score`` for square shaping -- because those differ by
construction and collapsing them would only trade duplication for conditionals.

Why every constant is an explicit argument: the two consumers **disagree** on
the aperture (``slm_zernike_pib`` uses 300 px, ``slm_zernike_shaping`` uses
``min(PANEL_RES)/2`` = 600). A shared module-level default would silently re-forge
that 600 -> 300 and change the square/GS phase the bench actually produces. Same
reason the memory-slot counter is an explicit :class:`SlotRotator` rather than a
module global: sharing one counter across two importers in a single process would
interleave their slot sequences.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ao_shaping.drivers.slm.santec import MEMORY_MODE_INTERNAL
from ao_shaping.optimizer.wfless._convergence_diagnostics import (
    MIN_HISTORY,
    convergence_lr_delta_factors,
)
from ao_shaping.utils.wavefront.zernike_utils import parse_zernike_coefficients

if TYPE_CHECKING:
    from ao_shaping.runners.runner_common import CameraParamsPib, SlmParamsPib
    from ao_shaping.utils.slm.pattern_helper import PatternHelper

#: Memory-slot range used for phase writes. Never repeat a slot consecutively:
#: ``display_memory`` on the slot already shown is a firmware no-op.
SLOT_MIN, SLOT_MAX = 2, 125


class SlotRotator:
    """Hands out a distinct SLM memory slot on every call.

    Writing the *same* slot twice is a firmware no-op -- the LCOS panel does not
    refresh -- so consecutive phase writes must rotate. Each consumer owns its
    own instance so two importers in one process cannot interleave sequences.
    """

    def __init__(self, slot_min: int = SLOT_MIN, slot_max: int = SLOT_MAX) -> None:
        if slot_min > slot_max:
            raise ValueError(f"slot_min {slot_min} > slot_max {slot_max}")
        self.slot_min = slot_min
        self.slot_max = slot_max
        self._slot = slot_min - 1

    def next(self) -> int:
        """Advance and return the next slot, wrapping to ``slot_min`` at the top."""
        self._slot = (
            self.slot_min if self._slot >= self.slot_max else self._slot + 1
        )
        return self._slot


def display_rotating(slm: Any, gray: np.ndarray, rotator: SlotRotator) -> int:
    """Display ``gray`` on a **rotating explicit memory slot**; return that slot.

    ``slm.display_data(gray)`` (default slot handling) produced NO optical effect
    on this bench: verified on BOTH cameras, a 5 rad tilt did not move the spot
    and a uniform 0 vs 900 changed nothing. ``slm.display_data(gray,
    memory_number=<distinct slot>, memory_mode=MEMORY_MODE_INTERNAL)`` -- the
    mechanism ``tools/slm/slm_diagnose.py`` uses -- does refresh the LCOS.
    """
    slot = rotator.next()
    slm.display_data(
        gray,
        memory_number=slot,
        memory_mode=MEMORY_MODE_INTERNAL,
    )
    return slot


def zernike_to_phase(
    coeffs: np.ndarray,
    n_max: int,
    pattern_helper: PatternHelper,
    radius: float | None = None,
    *,
    default_radius: float,
) -> np.ndarray:
    """Convert a flat Noll-order Zernike coefficient array to a
    canonical (n, m)→amplitude dict via `parse_zernike_coefficients`
    (skips |amp| < 1e-15 and n > n_max — both contribute zero phase).

    ``default_radius`` is keyword-only and mandatory: the two consumers pass
    different apertures, so there is no safe shared default.
    """
    coeffs_dict: dict[tuple[int, int], float] = parse_zernike_coefficients(
        coeffs, n_max=n_max
    )
    return pattern_helper.generate_zernike_polynomial(
        n_max=n_max,
        coefficients=coeffs_dict,
        radius=default_radius if radius is None else radius,
    )


def learning_schedule(
    power_radius: float,
    ideal_r: float,
    gradient_history: list[float] | None = None,
    pib_history: list[float] | None = None,
    epoch: int = 0,
    *,
    max_lr: float,
    max_delta: float,
) -> tuple[float, float]:
    """Dynamic learning rate scheduler based on power radius and convergence state.

    Args:
        power_radius: Current power radius.
        ideal_r: Ideal spot radius (the segment boundaries are multiples of it).
        gradient_history: Recent gradient magnitude history for convergence detection.
        pib_history: Recent PIB value history for convergence detection.
        epoch: Current iteration number.
        max_lr: Hard ceiling on the learning rate.
        max_delta: Hard ceiling on the perturbation amplitude.

    Returns:
        (lr, delta): Learning rate and perturbation amplitude.
    """
    # Base parameters: segmented by power_radius
    if power_radius <= ideal_r:
        base_lr, base_delta = 1.5, 1
    elif power_radius <= 2 * ideal_r:
        base_lr, base_delta = 2, 2
    elif power_radius <= 3 * ideal_r:
        base_lr, base_delta = 2.5, 3
    elif power_radius <= 4 * ideal_r:
        base_lr, base_delta = 3, 4
    elif power_radius <= 5 * ideal_r:
        base_lr, base_delta = 4.5, 5
    else:
        base_lr, base_delta = 6, 5

    # If no convergence history, return base parameters directly (still capped)
    if (
        gradient_history is None
        or pib_history is None
        or len(gradient_history) < MIN_HISTORY
    ):
        return (
            min(base_lr, max_lr),
            min(base_delta, max_delta),
        )

    # Convergence state detection (shared: _convergence_diagnostics;
    # PIB higher-is-better, stagnant needs low std), then cap by max_lr/max_delta
    lr_factor, delta_factor = convergence_lr_delta_factors(
        gradient_history,
        pib_history,
        epoch,
        higher_is_better=True,
        stagnant_needs_low_std=True,
    )

    final_lr = min(base_lr * lr_factor, max_lr)
    final_delta = min(base_delta * delta_factor, max_delta)

    return final_lr, final_delta


def default_camera() -> CameraParamsPib:
    """Create the canonical camera/objective group without a module-level cycle."""
    from ao_shaping.runners.runner_common import CameraParamsPib

    return CameraParamsPib()


def default_slm() -> SlmParamsPib:
    """Create the canonical SLM group without a module-level cycle."""
    from ao_shaping.runners.runner_common import SlmParamsPib

    return SlmParamsPib()
