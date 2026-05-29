"""Greedy optimization using SLM with Zernike coefficient control.

This module implements a greedy local search optimizer that:
1. Initializes N random starting positions and picks the best one
2. At each iteration, samples n random perturbation directions
3. Evaluates all n perturbations + current position
4. Moves to the best candidate
5. Repeats until convergence or max iterations

Key features:
- ZernikeSLM for phase modulation via Zernike polynomial coefficients
- ThorlabWFS for wavefront sensing and RMS measurement
- Multi-start greedy search with directional sampling
- Early stopping based on RMS threshold

Example:
    >>> from ao_shaping.optimizer.wf.greedy_zernike import optimizer_greedy
    >>> recorder = optimizer_greedy(
    ...     epochs=2000,
    ...     n_init=10,
    ...     n_directions=5,
    ...     n_max=4,
    ...     wavelength=1064,
    ... )
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tqdm

from ao_shaping.drivers import MlaRes, ThorlabWFS
from ao_shaping.drivers.slm import ZernikeSLM
from ao_shaping.utils import Recorder, logger
from ao_shaping.utils.matrix_utils import calc_n_zernike_terms

# SLM parameters
SLM_WAVELENGTH_DEFAULT = 532  # nm
SLM_SHIFT_X_DEFAULT = 0  # pixels
SLM_SHIFT_Y_DEFAULT = 0  # pixels

# Zernike coefficient bounds
ZERNIKE_MIN = -50.0  # wavelengths
ZERNIKE_MAX = 50.0  # wavelengths


def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index to (n, m) Zernike order (hardcoded convention).

    Uses a hardcoded lookup table (Noll indices 1-15 only).
    NOTE: This convention DIFFERS from the aotools-based implementation
    in `utils/zernike_calc.py`. The canonical implementation is in
    `utils/zernike_calc.noll_to_nm()`.

    Args:
        j: Noll index (1-based), valid range 1-15.

    Returns:
        Tuple of (n, m) radial and azimuthal orders.
    """
    # Noll sequence for Zernike polynomials
    noll_sequence = [
        (0, 0),   # 1: piston
        (1, -1),  # 2: tilt x
        (1, 1),   # 3: tilt y
        (2, -2),  # 4: oblique astigmatism
        (2, 0),   # 5: defocus
        (2, 2),   # 6: oblique astigmatism
        (3, -3),  # 7: vertical trefoil
        (3, -1),  # 8: vertical coma
        (3, 1),   # 9: horizontal coma
        (3, 3),   # 10: horizontal trefoil
        (4, -4),  # 11: quadrafoil
        (4, -2),  # 12: oblique trefoil
        (4, 0),   # 13: primary spherical
        (4, 2),   # 14: oblique trefoil
        (4, 4),   # 15: quadrafoil
    ]
    if j < 1 or j > len(noll_sequence):
        raise ValueError(f"Noll index {j} out of valid range (1-{len(noll_sequence)})")
    return noll_sequence[j - 1]


def _zernike_indices(n_max: int) -> list[tuple[int, int]]:
    """Return list of (n, m) pairs for all valid Zernike modes up to n_max.

    Args:
        n_max: Maximum Zernike radial order.

    Returns:
        List of (n, m) tuples in Noll order.
    """
    n_terms = calc_n_zernike_terms(n_max)
    modes = []
    for j in range(1, n_terms + 1):
        n, m = noll_to_nm(j)
        if n <= n_max:
            modes.append((n, m))
    return modes


def _generate_random_direction(param_dim: int, perturb_mask: np.ndarray | None = None) -> np.ndarray:
    """Generate a random perturbation direction.

    Args:
        param_dim: Dimension of parameter vector.
        perturb_mask: Optional mask for which parameters to perturb.

    Returns:
        Random direction vector of shape (param_dim,).
    """
    direction = np.random.binomial(1, 0.5, (param_dim,)).astype(float) * 2.0 - 1.0
    if perturb_mask is not None:
        direction = direction * perturb_mask
    return direction


def optimizer_greedy(
    epochs: int,
    n_init: int = 10,
    n_directions: int = 5,
    perturbation_scale: float = 5.0,
    init_z: Sequence[float | int] | dict[tuple[int, int], float] | None = None,
    pupil_center: tuple[float, float] = (0, 0),
    pupil_diameter: float = 4.6,
    early_stop_threshold: float = 0.12,
    wavelength: int = SLM_WAVELENGTH_DEFAULT,
    shift_x: int = SLM_SHIFT_X_DEFAULT,
    shift_y: int = SLM_SHIFT_Y_DEFAULT,
    n_max: int = 4,
    wfs_res: MlaRes = MlaRes.Res1024,
    remove_tilt: bool = False,
    slm_number: int = 1,
    slm_wavelength: int | None = None,
) -> Recorder:
    """Optimize wavefront RMS using Greedy Local Search with SLM Zernike control.

    This function implements a greedy optimizer that:
    1. Initializes N random starting positions and picks the best one
    2. At each iteration, samples n random perturbation directions
    3. Evaluates all n perturbations + current position (n+1 candidates)
    4. Moves to the best candidate
    5. Repeats until convergence or max epochs

    Args:
        epochs: Maximum number of optimization iterations.
        n_init: Number of random initial positions to sample.
        n_directions: Number of random perturbation directions per iteration.
        perturbation_scale: Scale factor for perturbation magnitudes.
        init_z: Initial Zernike coefficients. Can be:
            - dict: {(n, m): value} form
            - Sequence: Noll-ordered coefficients
            - None: starts from random initialization
        pupil_center: WFS pupil center (x, y).
        pupil_diameter: WFS pupil diameter.
        early_stop_threshold: Stop if RMS drops below this threshold.
        wavelength: SLM wavelength in nm.
        shift_x: SLM X shift in pixels.
        shift_y: SLM Y shift in pixels.
        n_max: Maximum Zernike radial order.
        wfs_res: WFS resolution.
        remove_tilt: Remove tilt in WFS wavefront measurement.
        slm_number: SLM device number (1-8).
        slm_wavelength: Override SLM wavelength (deprecated, use wavelength).

    Returns:
        Recorder: Optimization history with RMS and coefficients.
    """
    epochs = int(epochs)

    # Calculate number of Zernike terms
    n_zernike = calc_n_zernike_terms(n_max)
    zernike_modes = _zernike_indices(n_max)

    recorder = Recorder(mark='greedy_zernike', mode='min')

    # Create perturb mask (exclude piston/constant term)
    perturb_mask = np.ones(n_zernike, dtype=np.float64)
    if n_zernike > 0:
        perturb_mask[0] = 0

    with (
        ZernikeSLM(
            slm_number=slm_number,
            wavelength=slm_wavelength,
            n_max=n_max,
            shift_x=shift_x,
            shift_y=shift_y,
        ) as slm,
        ThorlabWFS(
            wfs_res,
            use_custom_ref=False,
            high_speed=True,
            pupil_diameter=pupil_diameter,
            pupil_center=pupil_center,
        ) as wfs,
    ):
        # Initialize Zernike coefficients
        if init_z is None or (
            isinstance(init_z, (list, tuple, np.ndarray)) and len(init_z) == 0
        ):
            _init_c = np.zeros(n_zernike, dtype=np.float64)
        elif isinstance(init_z, dict):
            _init_c = np.zeros(n_zernike, dtype=np.float64)
            for (n, m), amp in init_z.items():
                if (n, m) in zernike_modes:
                    idx = zernike_modes.index((n, m))
                    _init_c[idx] = amp
        else:
            _init_c = np.array(init_z, dtype=np.float64)
            if len(_init_c) < n_zernike:
                padded = np.zeros(n_zernike, dtype=np.float64)
                padded[: len(_init_c)] = _init_c
                _init_c = padded
            elif len(_init_c) > n_zernike:
                _init_c = _init_c[:n_zernike]

        def calc_wavefront():
            wfs.take_image(5)
            wf, statics = wfs.get_wavefront(cancel_tile=remove_tilt)
            return wf, statics

        def evaluate_coefficients(coeffs: np.ndarray) -> float:
            """Evaluate RMS for given Zernike coefficients."""
            slm.send_zernike(coeffs)
            _, statics = calc_wavefront()
            return float(statics.get('rms', np.inf))

        # Phase 1: Initialize N random positions and pick the best
        logger.info(
            f"Phase 1: Initializing {n_init} random positions "
            f"to find best starting point"
        )

        init_positions = []
        init_rmss = []

        # Generate n_init random initial positions
        for i in range(n_init):
            random_c = np.random.uniform(
                ZERNIKE_MIN, ZERNIKE_MAX, n_zernike
            ).astype(np.float64)
            init_positions.append(random_c)
            rms = evaluate_coefficients(random_c)
            init_rmss.append(rms)
            logger.debug(f"  Init position {i+1}/{n_init}: RMS = {rms:.4f}")

        # Find best initial position
        best_init_idx = np.argmin(init_rmss)
        best_init_rms = init_rmss[best_init_idx]
        current_c = init_positions[best_init_idx].copy()
        current_rms = best_init_rms

        logger.info(
            f"Best initial position: idx={best_init_idx+1}, "
            f"RMS={best_init_rms:.4f}"
        )

        # Evaluate initial state for recording
        wf, statics = calc_wavefront()

        recorder.append(
            {
                "rms": current_rms,
                "_c": current_c.copy(),
                "_epoch": 0,
                "_phase": "init",
                "_init_rms": init_rmss,
                "_wavefront": wf[np.newaxis, ...],
                "_statics": statics,
            }
        )

        # Phase 2: Greedy local search from best starting point
        logger.info(
            f"Phase 2: Greedy local search with {n_directions} "
            f"directions per iteration, max epochs={epochs}"
        )

        best_rms = current_rms
        best_c = current_c.copy()
        no_improvement_count = 0
        patience = 50  # Stop if no improvement for 50 epochs

        with tqdm.tqdm(
            total=epochs,
            desc=f"Greedy-Zernike RMS={current_rms:.4f}",
            dynamic_ncols=True,
        ) as bar:
            for epoch in range(1, epochs + 1):
                # Generate n_directions random perturbation directions
                candidates = [current_c.copy()]
                candidate_rmss = [current_rms]

                for _ in range(n_directions):
                    direction = _generate_random_direction(n_zernike, perturb_mask)
                    # Random magnitude between 0.5*scale and 1.5*scale
                    magnitude = perturbation_scale * np.random.uniform(0.5, 1.5)
                    perturbed = current_c + direction * magnitude
                    perturbed = np.clip(perturbed, ZERNIKE_MIN, ZERNIKE_MAX)

                    rms = evaluate_coefficients(perturbed)
                    candidates.append(perturbed)
                    candidate_rmss.append(rms)

                # Pick the best candidate (greedy: take best among all)
                best_candidate_idx = np.argmin(candidate_rmss)
                new_c = candidates[best_candidate_idx].copy()
                new_rms = candidate_rmss[best_candidate_idx]

                # Check for improvement
                if new_rms < current_rms:
                    current_c = new_c
                    current_rms = new_rms
                    no_improvement_count = 0

                    # Update global best
                    if current_rms < best_rms:
                        best_rms = current_rms
                        best_c = current_c.copy()
                else:
                    no_improvement_count += 1

                # Record current state
                wf, statics = calc_wavefront()
                recorder.append(
                    {
                        "rms": current_rms,
                        "_c": current_c.copy(),
                        "_epoch": epoch,
                        "_phase": "search",
                        "_candidate_rmss": candidate_rmss,
                        "_best_candidate_idx": best_candidate_idx,
                        "_no_improvement": no_improvement_count,
                        "_wavefront": wf[np.newaxis, ...],
                        "_statics": statics,
                        "best_rms": best_rms,
                    }
                )

                bar.set_postfix({
                    "rms": f"{current_rms:.4f}",
                    "best": f"{best_rms:.4f}",
                    "stall": no_improvement_count,
                })

                # Early stopping: RMS threshold
                if best_rms < early_stop_threshold:
                    logger.info(
                        f"Early stop at epoch {epoch} "
                        f"with best RMS={best_rms:.4f}"
                    )
                    break

                # Early stopping: patience (no improvement)
                if no_improvement_count >= patience:
                    logger.info(
                        f"Early stop at epoch {epoch} "
                        f"due to no improvement for {patience} epochs"
                    )
                    break

                bar.update(1)

        # Restore best coefficients on exit
        best_c, _ = recorder.get_best_target('_c')
        if best_c is not None:
            slm.send_zernike(best_c)
            logger.info(
                f"Greedy optimization complete. Restored best coefficients, "
                f"RMS: {recorder.get_best_target('rms')}"
            )

        return recorder