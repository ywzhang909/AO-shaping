"""
Tabu Search Algorithm Module

This module provides a general-purpose implementation of tabu-based adaptive neighborhood
search for optimization problems. It is designed to be domain-agnostic and can be used
with any optimization problem that requires escaping local optima.

The module includes:
- TabuMemory: Short-term memory to avoid revisiting explored candidates
- AdaptiveSearchState: Dynamic radius adjustment for neighborhood exploration
- Candidate generators: Methods for generating search candidates
- TabuSearchRunner: High-level orchestration class

Usage:
    from ao_shaping.algorithm.tabu_search import TabuSearchRunner, TabuMemory, AdaptiveSearchState

    # Create components
    tabu_memory = TabuMemory(capacity=128, quantization=2.0)
    search_state = AdaptiveSearchState(radius=2.0, min_radius=0.5, max_radius=12.0,
                                       expand_ratio=1.4, shrink_ratio=0.75, improvement_tol=1e-4)

    # Create runner
    runner = TabuSearchRunner(
        tabu_memory=tabu_memory,
        search_state=search_state,
        candidate_generator=my_candidate_generator,
        safety_check=my_safety_check,
    )

    # Run search
    result = runner.run_search(anchor_v, anchor_value, evaluate_candidate)

Author: AO-Shaping Development Team
Created: 2026-03-26
"""

from collections import deque
from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np


# =============================================================================
# Core Data Structures
# =============================================================================


@dataclass
class TabuMemory:
    """Short-term tabu memory for avoiding re-exploration of suboptimal candidates.

    This class implements a queue-based tabu memory with quantized voltage keys.
    It prevents the algorithm from revisiting recently explored candidates by storing
    their quantized representations in both a queue (for FIFO eviction) and a
    set (for O(1) lookup).

    The quantization allows for flexible granularity in the tabu memory. A larger
    quantization value means more candidates will be considered "the same" and thus
    marked as tabu.

    Attributes:
        capacity: Maximum number of candidates to store in tabu memory.
        quantization: Step size for quantizing candidate keys. Larger values
                    mean coarser discretization.

    Example:
        >>> tabu = TabuMemory(capacity=128, quantization=2.0)
        >>> voltages = np.array([10.5, 20.3, 15.7])
        >>> tabu.add(voltages)
        >>> tabu.contains(voltages)
        True
    """

    capacity: int
    quantization: float
    _queue: deque[tuple[int, ...]] = field(init=False, default_factory=deque)
    _keys: set[tuple[int, ...]] = field(init=False, default_factory=set)

    def make_key(self, voltages: np.ndarray) -> tuple[int, ...]:
        """Quantize voltage array into integer key for tabu storage.

        This method converts a voltage array into a tuple of integers by:
        1. Converting to float64 for precision
        2. Dividing by quantization scale
        3. Rounding to nearest integer

        Args:
            voltages: The voltage array to quantize.

        Returns:
            Tuple of integers representing the quantized voltage profile.
        """
        scale = max(float(self.quantization), 1e-6)
        return tuple(
            np.round(np.asarray(voltages, dtype=np.float64) / scale).astype(int)
        )

    def contains(self, voltages: np.ndarray) -> bool:
        """Check if a voltage profile is in tabu memory.

        Args:
            voltages: The voltage array to check.

        Returns:
            True if the quantized voltages are in tabu memory, False otherwise.
            Returns False if capacity is <= 0 (tabu disabled).
        """
        if self.capacity <= 0:
            return False
        return self.make_key(voltages) in self._keys

    def add(self, voltages: np.ndarray) -> None:
        """Add a voltage profile to tabu memory.

        If the key already exists or capacity is <= 0, this method does nothing.
        When capacity is exceeded, the oldest entry is evicted (FIFO).

        Args:
            voltages: The voltage array to add to tabu memory.
        """
        if self.capacity <= 0:
            return
        key = self.make_key(voltages)
        if key in self._keys:
            return
        self._queue.append(key)
        self._keys.add(key)
        while len(self._queue) > self.capacity:
            expired = self._queue.popleft()
            self._keys.discard(expired)


@dataclass
class AdaptiveSearchState:
    """State management for adaptive neighborhood search radius.

    This class manages the dynamic adjustment of the search radius based on
    whether recent search iterations have produced improvements. It implements
    an adaptive strategy that:
    - Shrinks the radius when improvements are found (exploitation)
    - Expands the radius when no improvement is found (exploration)

    The radius is clipped between min_radius and max_radius to prevent
    degenerate behavior.

    Attributes:
        radius: Current search radius.
        min_radius: Minimum allowed search radius.
        max_radius: Maximum allowed search radius.
        expand_ratio: Multiplier for radius expansion when no improvement.
        shrink_ratio: Multiplier for radius shrinking when improved.
        improvement_tol: Tolerance for considering an improvement significant.

    Example:
        >>> state = AdaptiveSearchState(radius=2.0, min_radius=0.5, max_radius=12.0,
        ...                            expand_ratio=1.4, shrink_ratio=0.75, improvement_tol=1e-4)
        >>> state.update_radius(improved=True)  # Shrink
        1.5
        >>> state.update_radius(improved=False)  # Expand
        2.1
    """

    radius: float
    min_radius: float
    max_radius: float
    expand_ratio: float
    shrink_ratio: float
    improvement_tol: float

    def update_radius(self, improved: bool) -> float:
        """Update the search radius based on improvement status.

        Args:
            improved: Whether the last search iteration found improvement.

        Returns:
            The updated radius value (clamped to [min_radius, max_radius]).
        """
        if improved:
            next_radius = self.radius * self.shrink_ratio
        else:
            next_radius = self.radius * self.expand_ratio
        self.radius = float(np.clip(next_radius, self.min_radius, self.max_radius))
        return self.radius


# =============================================================================
# Candidate Generation
# =============================================================================


def generate_search_candidates(
    anchor_v: np.ndarray,
    radius_scale: float,
    n_samples: int,
    active_mask: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Generate mixed dense/sparse perturbations around an anchor point.

    This function generates candidate solutions by perturbing an anchor voltage
    vector. It uses a mixed strategy:
    - Half of candidates: Gaussian perturbation (dense exploration)
    - Half of candidates: Sparse uniform perturbation (sparse exploration)

    The sparse perturbation activates only ~35% of dimensions with random
    magnitudes, providing a different exploration pattern than dense methods.

    Args:
        anchor_v: The anchor voltage vector to perturb.
        radius_scale: Standard deviation for Gaussian perturbations, scale for
                      uniform perturbations.
        n_samples: Number of candidate perturbations to generate.
        active_mask: Binary mask indicating which dimensions to perturb.
                    If None, all dimensions are active.
        rng: Random number generator. If None, uses default_rng.

    Returns:
        List of candidate voltage vectors.

    Example:
        >>> anchor = np.zeros(64)
        >>> candidates = generate_search_candidates(anchor, 2.0, 8, rng=np.random.default_rng())
        >>> len(candidates)
        8
    """
    if rng is None:
        rng = np.random.default_rng()

    candidates: list[np.ndarray] = []

    # Apply active mask if provided
    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=np.float64)
    else:
        mask = np.ones_like(anchor_v, dtype=np.float64)

    radius_scale = max(float(radius_scale), 1e-6)

    for sample_id in range(max(int(n_samples), 1)):
        # Alternate between dense (Gaussian) and sparse (uniform) perturbations
        if sample_id % 2 == 0:
            # Dense perturbation: Gaussian noise
            perturbation = rng.normal(0.0, radius_scale, size=anchor_v.shape)
        else:
            # Sparse perturbation: random signs + magnitudes + sparse activation
            signs = (
                rng.binomial(1, 0.5, size=anchor_v.shape).astype(np.float64) * 2.0 - 1.0
            )
            magnitudes = rng.uniform(
                radius_scale * 0.35, radius_scale, size=anchor_v.shape
            )
            sparse_mask = rng.binomial(1, 0.35, size=anchor_v.shape).astype(np.float64)
            perturbation = signs * magnitudes * sparse_mask

        # Apply mask and add to anchor
        candidates.append(anchor_v + perturbation * mask)

    return candidates


def should_trigger_search(
    epoch: int,
    enabled: bool,
    warmup: int,
    interval: int,
    patience: int,
    last_best_epoch: int,
) -> bool:
    """Determine if adaptive search should be triggered at current epoch.

    This function checks multiple conditions to determine whether to trigger
    the tabu search:
    1. Search must be enabled
    2. Must have passed warmup period
    3. Must be at correct interval (not every epoch)
    4. Must have exceeded patience threshold since last improvement

    Args:
        epoch: Current optimization epoch.
        enabled: Whether adaptive search is enabled.
        warmup: Minimum epochs before first search.
        interval: Epochs between search triggers.
        patience: Epochs without improvement before triggering search.
        last_best_epoch: Epoch of last improvement.

    Returns:
        True if search should be triggered, False otherwise.

    Example:
        >>> should_trigger_search(epoch=500, enabled=True, warmup=200,
        ...                       interval=120, patience=100, last_best_epoch=350)
        True
    """
    if not enabled:
        return False
    if epoch < max(int(warmup), 1):
        return False
    if interval <= 0 or epoch % int(interval) != 0:
        return False
    return (epoch - last_best_epoch) >= max(int(patience), 0)


# =============================================================================
# Tabu Search Runner
# =============================================================================


class TabuSearchRunner:
    """Orchestrates tabu-based adaptive neighborhood search.

    This class provides a high-level interface for running tabu search with
    adaptive neighborhood exploration. It integrates:
    - Tabu memory for avoiding re-exploration
    - Adaptive radius management
    - Candidate generation
    - Safety checks
    - Candidate evaluation

    The runner is designed to be domain-agnostic by accepting callback functions
    for problem-specific operations (candidate generation, safety checks, evaluation).

    Attributes:
        tabu_memory: Tabu memory instance for tracking explored candidates.
        search_state: Adaptive search state for radius management.
        candidate_generator: Function to generate candidate solutions.
        safety_check: Optional function to validate candidate safety.
        clip_bounds: Optional (min, max) tuple for voltage clipping.

    Example:
        >>> def evaluate(voltages):
        ...     # Your objective function here
        ...     return {"value": objective_value, "other": data}
        >>>
        >>> runner = TabuSearchRunner(
        ...     tabu_memory=TabuMemory(128, 2.0),
        ...     search_state=AdaptiveSearchState(2.0, 0.5, 12.0, 1.4, 0.75, 1e-4),
        ...     candidate_generator=generate_search_candidates,
        ...     safety_check=lambda v: np.all(np.abs(v) < 100),
        ...     clip_bounds=(-50, 50),
        ... )
        >>> result = runner.run_search(anchor, anchor_value, evaluate)
    """

    def __init__(
        self,
        tabu_memory: TabuMemory,
        search_state: AdaptiveSearchState,
        candidate_generator: Callable[
            [np.ndarray, float, int, np.ndarray | None, np.random.Generator],
            list[np.ndarray],
        ]
        | None = None,
        safety_check: Callable[[np.ndarray], bool] | None = None,
        clip_bounds: tuple[float, float] | None = None,
    ):
        """Initialize the TabuSearchRunner.

        Args:
            tabu_memory: Tabu memory instance.
            search_state: Adaptive search state instance.
            candidate_generator: Function to generate candidates.
                                If None, uses default generate_search_candidates.
            safety_check: Optional function to validate candidate safety.
                         If None, all candidates are considered safe.
            clip_bounds: Optional (min, max) tuple for voltage clipping.
        """
        self.tabu_memory = tabu_memory
        self.search_state = search_state
        self.candidate_generator = candidate_generator or generate_search_candidates
        self.safety_check = safety_check
        self.clip_bounds = clip_bounds

    def run_search(
        self,
        anchor_v: np.ndarray,
        anchor_objective: float,
        evaluate_candidate: Callable[[np.ndarray], dict],
        objective_key: str = "value",
        improvement_tol: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict | None:
        """Run one iteration of tabu search.

        This method:
        1. Generates candidate solutions around anchor
        2. Filters out tabu and unsafe candidates
        3. Evaluates valid candidates
        4. Selects best improving candidate
        5. Updates tabu memory and search radius
        6. Returns result or None if no progress

        Args:
            anchor_v: Current best voltage vector.
            anchor_objective: Objective value at anchor.
            evaluate_candidate: Function that evaluates a candidate and returns
                              a dict with at least 'objective_key' and 'value' keys.
            objective_key: Key name for objective value in evaluation dict.
            improvement_tol: Minimum improvement required. If None, uses
                           search_state.improvement_tol.
            rng: Random number generator. If None, uses default_rng.

        Returns:
            None if no candidates were evaluated (all rejected or empty).
            Dict with:
                - accepted: bool - Whether a candidate was accepted
                - voltages: np.ndarray - Best candidate voltages
                - value: float - Objective value at best candidate
                - tabu_hits: int - Number of candidates skipped due to tabu
                - safe_rejects: int - Number of candidates rejected for safety
                - evaluated: int - Number of candidates evaluated
                - radius: float - Current search radius
                - anchor: str - Source of anchor ('best' or 'current')
        """
        if rng is None:
            rng = np.random.default_rng()

        # Use provided tolerance or default from search state
        tol = (
            improvement_tol
            if improvement_tol is not None
            else self.search_state.improvement_tol
        )

        # Generate candidates - use positional args for compatibility
        if self.candidate_generator == generate_search_candidates:
            candidates = self.candidate_generator(
                anchor_v,
                self.search_state.radius,
                8,
                None,
                rng,
            )
        else:
            # Custom generator - try with kwargs
            try:
                candidates = self.candidate_generator(
                    anchor_v=anchor_v,
                    radius_scale=self.search_state.radius,
                    n_samples=8,
                    active_mask=None,
                    rng=rng,
                )
            except TypeError:
                # Fallback: positional args
                candidates = self.candidate_generator(
                    anchor_v,
                    self.search_state.radius,
                    8,
                    None,
                    rng,
                )

        best_candidate: dict | None = None
        tabu_hits = 0
        safe_rejects = 0
        evaluated = 0

        for candidate in candidates:
            # Apply clipping if bounds specified
            if self.clip_bounds is not None:
                candidate = np.clip(candidate, self.clip_bounds[0], self.clip_bounds[1])

            # Check tabu
            if self.tabu_memory.contains(candidate):
                tabu_hits += 1
                continue

            # Check safety
            if self.safety_check is not None and not self.safety_check(candidate):
                safe_rejects += 1
                self.tabu_memory.add(candidate)
                continue

            # Evaluate candidate
            candidate_eval = evaluate_candidate(candidate)
            evaluated += 1

            # Check if this is an improvement
            candidate_value = candidate_eval.get(
                objective_key, candidate_eval.get("value", 0)
            )
            improved = candidate_value > anchor_objective + tol

            if improved and (
                best_candidate is None
                or candidate_value
                > best_candidate.get(objective_key, best_candidate.get("value", 0))
            ):
                best_candidate = {
                    "voltages": candidate.copy(),
                    **candidate_eval,
                    objective_key: candidate_value,
                }
            else:
                self.tabu_memory.add(candidate)

        # Handle no valid candidates
        if best_candidate is None:
            self.search_state.update_radius(improved=False)
            return {
                "accepted": False,
                "tabu_hits": tabu_hits,
                "safe_rejects": safe_rejects,
                "evaluated": evaluated,
                "radius": self.search_state.radius,
                "anchor": "best",
            }

        # Accept best candidate
        self.tabu_memory.add(anchor_v)
        self.search_state.update_radius(improved=True)

        best_candidate.update(
            {
                "accepted": True,
                "tabu_hits": tabu_hits,
                "safe_rejects": safe_rejects,
                "evaluated": evaluated,
                "radius": self.search_state.radius,
                "anchor": "best",
            }
        )

        return best_candidate


# =============================================================================
# Factory Functions
# =============================================================================


def create_tabu_search_runner(
    capacity: int = 128,
    quantization: float = 2.0,
    initial_radius: float = 2.0,
    min_radius: float = 0.5,
    max_radius: float = 12.0,
    expand_ratio: float = 1.4,
    shrink_ratio: float = 0.75,
    improvement_tol: float = 1e-4,
    candidate_generator: Callable | None = None,
    safety_check: Callable[[np.ndarray], bool] | None = None,
    clip_bounds: tuple[float, float] | None = None,
) -> TabuSearchRunner:
    """Factory function to create a TabuSearchRunner with default parameters.

    This is a convenience function that creates all required components
    with sensible defaults.

    Args:
        capacity: Tabu memory capacity.
        quantization: Tabu memory quantization step.
        initial_radius: Initial search radius.
        min_radius: Minimum search radius.
        max_radius: Maximum search radius.
        expand_ratio: Radius expansion ratio.
        shrink_ratio: Radius shrinking ratio.
        improvement_tol: Improvement tolerance.
        candidate_generator: Custom candidate generator or None for default.
        safety_check: Custom safety check or None for no check.
        clip_bounds: Voltage clipping bounds or None.

    Returns:
        Configured TabuSearchRunner instance.

    Example:
        >>> runner = create_tabu_search_runner(
        ...     capacity=128,
        ...     initial_radius=2.0,
        ...     safety_check=lambda v: np.all(np.abs(v) < 100),
        ... )
    """
    tabu_memory = TabuMemory(capacity=capacity, quantization=quantization)
    search_state = AdaptiveSearchState(
        radius=initial_radius,
        min_radius=min_radius,
        max_radius=max_radius,
        expand_ratio=expand_ratio,
        shrink_ratio=shrink_ratio,
        improvement_tol=improvement_tol,
    )

    return TabuSearchRunner(
        tabu_memory=tabu_memory,
        search_state=search_state,
        candidate_generator=candidate_generator,
        safety_check=safety_check,
        clip_bounds=clip_bounds,
    )
