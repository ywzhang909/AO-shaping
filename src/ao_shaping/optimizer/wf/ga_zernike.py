"""Genetic Algorithm optimization using SLM with Zernike coefficient control.

This module uses a Genetic Algorithm to optimize Zernike coefficients
displayed on an SLM, minimizing the wavefront RMS measured by a WFS.

Key features:
- ZernikeSLM for phase modulation via Zernike polynomial coefficients
- Thorlab_WFS for wavefront sensing and RMS measurement
- Genetic Algorithm with tournament selection, crossover, and mutation
- Elitism to preserve top-performing individuals
- Early stopping based on RMS threshold

Example:
    >>> from ao_shaping.optimizer.wf.ga_zernike import optimizer_ga
    >>> recorder = optimizer_ga(
    ...     n_generations=2000,
    ...     population_size=50,
    ...     n_max=4,
    ...     wavelength=1064,
    ... )
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tqdm

from ao_shaping.drivers import MlaRes, Thorlab_WFS
from ao_shaping.drivers.slm import ZernikeSLM
from ao_shaping.utils import logger, Recorder
from ao_shaping.utils.matrix_utils import calc_n_zernike_terms

# SLM parameters
SLM_WAVELENGTH_DEFAULT = 532  # nm
SLM_SHIFT_X_DEFAULT = 0  # pixels
SLM_SHIFT_Y_DEFAULT = 0  # pixels

# Zernike coefficient bounds
ZERNIKE_MIN = -50.0  # wavelengths
ZERNIKE_MAX = 50.0  # wavelengths


def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index to (n, m) Zernike order.

    Args:
        j: Noll index (1-based).

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


def _tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    tournament_size: int,
) -> np.ndarray:
    """Select an individual using tournament selection.

    Args:
        population: Array of shape (pop_size, n_zernike) containing the population.
        fitness: Array of shape (pop_size,) containing fitness values (lower is better).
        tournament_size: Number of individuals to compete in the tournament.

    Returns:
        Selected individual as array of shape (n_zernike,).
    """
    pop_size = len(population)
    contestants = np.random.choice(pop_size, tournament_size, replace=False)
    # For minimization, select the one with minimum fitness
    best_idx = contestants[np.argmin(fitness[contestants])]
    return population[best_idx].copy()


def _blend_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend crossover (BLX-alpha) for two parents.

    Creates two children by interpolating/extrapolating between parents
    within a range expanded by factor alpha.

    Args:
        parent1: First parent array of shape (n_zernike,).
        parent2: Second parent array of shape (n_zernike,).
        alpha: Expansion factor for the search range.

    Returns:
        Tuple of two children arrays.
    """
    n = len(parent1)
    # Calculate the range between parents
    c_min = np.minimum(parent1, parent2)
    c_max = np.maximum(parent1, parent2)
    I = c_max - c_min

    # Expand the range
    lower = c_min - alpha * I
    upper = c_max + alpha * I

    # Generate children uniformly in the expanded range
    child1 = lower + np.random.rand(n) * (upper - lower)
    child2 = lower + np.random.rand(n) * (upper - lower)

    return child1, child2


def _gaussian_mutation(
    individual: np.ndarray,
    mutation_rate: float,
    sigma: float = 5.0,
    bounds: tuple[float, float] = (ZERNIKE_MIN, ZERNIKE_MAX),
) -> np.ndarray:
    """Apply Gaussian mutation to an individual.

    Each gene has a probability of being mutated according to mutation_rate.
    Mutated genes are perturbed by a Gaussian with mean 0 and given sigma.
    Values are clipped to the specified bounds.

    Args:
        individual: Individual to mutate, shape (n_zernike,).
        mutation_rate: Probability of mutating each gene.
        sigma: Standard deviation of the Gaussian perturbation.
        bounds: Tuple of (min, max) bounds for clipping.

    Returns:
        Mutated individual.
    """
    mutated = individual.copy()
    mask = np.random.rand(len(individual)) < mutation_rate
    if np.any(mask):
        noise = np.random.normal(0, sigma, int(np.sum(mask)))
        mutated[mask] += noise
    return np.clip(mutated, bounds[0], bounds[1])


def optimizer_ga(
    n_generations: int,
    population_size: int = 50,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.15,
    tournament_size: int = 3,
    elite_count: int = 2,
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
    """Optimize wavefront RMS using Genetic Algorithm with SLM Zernike control.

    This function uses a Genetic Algorithm to optimize Zernike coefficients
    displayed on an SLM, minimizing the wavefront RMS measured by a WFS.

    Args:
        n_generations: Number of GA generations to run.
        population_size: Number of individuals in the population.
        crossover_prob: Probability of crossover between parents.
        mutation_prob: Probability of mutating each gene.
        tournament_size: Number of individuals in tournament selection.
        elite_count: Number of top individuals to preserve each generation.
        init_z: Initial Zernike coefficients. Can be:
            - dict: {(n, m): value} form
            - Sequence: Noll-ordered coefficients
            - None: starts from zeros
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
    n_generations = int(n_generations)

    # Calculate number of Zernike terms
    n_zernike = calc_n_zernike_terms(n_max)
    zernike_modes = _zernike_indices(n_max)

        recorder = Recorder(mark='rms', mode='min')

    with (
        ZernikeSLM(
            slm_number=slm_number,
            wavelength=slm_wavelength,
            n_max=n_max,
            shift_x=shift_x,
            shift_y=shift_y,
        ) as slm,
        Thorlab_WFS(
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

        # Initialize SLM with initial coefficients
        init_phase = slm.send_zernike(_init_c)

        def calc_wavefront():
            wfs.take_image(5)
            wf, statics = wfs.get_wavefront(cancel_tile=remove_tilt)
            return wf, statics

        # Evaluate initial state
        wf, statics = calc_wavefront()
        initial_rms = statics.get('rms', np.inf)

        logger.info(
            f"Initial RMS: {initial_rms:.4f}, starting GA with "
            f"population_size={population_size}, n_generations={n_generations}"
        )

        # Initialize population around the initial coefficients
        population = np.random.uniform(
            ZERNIKE_MIN, ZERNIKE_MAX, (population_size, n_zernike)
        )
        # Set first population member to initial coefficients
        population[0] = _init_c.copy()

        # Evaluate initial population fitness
        fitness = np.full(population_size, np.inf)

        def evaluate_individual(individual: np.ndarray) -> float:
            """Evaluate an individual's fitness (RMS - lower is better)."""
            slm.send_zernike(individual)
            _, statics = calc_wavefront()
            return float(statics.get('rms', np.inf))

        # Evaluate entire initial population
        for i in range(population_size):
            fitness[i] = evaluate_individual(population[i])

        # Record initial state
        best_idx = np.argmin(fitness)
        best_rms = fitness[best_idx]
        best_c = population[best_idx].copy()

        recorder.append(
            {
                "rms": best_rms,
                "_c": best_c.copy(),
                "_generation": 0,
                "_population": population.copy(),
                "_fitness": fitness.copy(),
                "_wavefront": wf[np.newaxis, ...],
                "_statics": statics,
            }
        )

        with tqdm.tqdm(
            total=n_generations,
            desc=f"GA-Zernike RMS={best_rms:.4f}",
            dynamic_ncols=True,
        ) as bar:
            for generation in range(1, n_generations + 1):
                # Create new population
                new_population = []

                # Elitism: preserve top elite_count individuals
                elite_indices = np.argsort(fitness)[:elite_count]
                for idx in elite_indices:
                    new_population.append(population[idx].copy())

                # Generate offspring through selection, crossover, and mutation
                while len(new_population) < population_size:
                    # Tournament selection for parents
                    parent1 = _tournament_selection(
                        population, fitness, tournament_size
                    )
                    parent2 = _tournament_selection(
                        population, fitness, tournament_size
                    )

                    # Crossover
                    if np.random.rand() < crossover_prob:
                        child1, child2 = _blend_crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()

                    # Mutation
                    child1 = _gaussian_mutation(child1, mutation_prob)
                    child2 = _gaussian_mutation(child2, mutation_prob)

                    new_population.append(child1)
                    if len(new_population) < population_size:
                        new_population.append(child2)

                # Trim to exact population size
                population = np.array(new_population[:population_size])

                # Evaluate new population
                for i in range(population_size):
                    fitness[i] = evaluate_individual(population[i])

                # Find best individual
                best_idx = np.argmin(fitness)
                current_best_rms = fitness[best_idx]
                current_best_c = population[best_idx].copy()

                # Update global best
                if current_best_rms < best_rms:
                    best_rms = current_best_rms
                    best_c = current_best_c.copy()

                # Record this generation - evaluate best individual for recording
                slm.send_zernike(current_best_c)
                wf, statics = calc_wavefront()

                recorder.append(
                    {
                        "rms": current_best_rms,
                        "_c": current_best_c.copy(),
                        "_generation": generation,
                        "_population": population.copy(),
                        "_fitness": fitness.copy(),
                        "_wavefront": wf[np.newaxis, ...],
                        "_statics": statics,
                        "best_rms": best_rms,
                    }
                )

                bar.set_postfix({
                    "rms": f"{current_best_rms:.4f}",
                    "best": f"{best_rms:.4f}",
                    "gen": generation,
                })

                # Early stopping
                if best_rms < early_stop_threshold:
                    logger.info(
                        f"Early stop at generation {generation} "
                        f"with best RMS={best_rms:.4f}"
                    )
                    break

                bar.update(1)

        # Restore best coefficients on exit
        best_c, _ = recorder.get_best_target('_c')
        if best_c is not None:
            slm.send_zernike(best_c)
            logger.info(
                f"GA complete. Restored best coefficients, "
                f"RMS: {recorder.get_best_target('rms')}"
            )

        return recorder
