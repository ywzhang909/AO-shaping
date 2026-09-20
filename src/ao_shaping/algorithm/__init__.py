try:
    from ao_shaping.algorithm.adam_cython import (
        Base,
        SGD,
        Adam,
        AdamW,
        AdaMOD,
        learning_schedule,
    )

    CYTHON_AVAILABLE = True
except ImportError:
    from ao_shaping.algorithm.gradient.adam import (
        Base,
        SGD,
        Adam,
        AdamW,
        AdaMOD,
        learning_schedule,
    )

    CYTHON_AVAILABLE = False

try:
    from ao_shaping.algorithm.target_func_cython import ImageTargetFunc

    CYTHON_TARGET_FUNC_AVAILABLE = True
except ImportError:
    from ao_shaping.algorithm.goal_functions.target_func import ImageTargetFunc

    CYTHON_TARGET_FUNC_AVAILABLE = False

from ao_shaping.algorithm.tabu.tabu_search import (
    TabuMemory,
    AdaptiveSearchState,
    generate_search_candidates,
    should_trigger_search,
    TabuSearchRunner,
    create_tabu_search_runner,
)

from ao_shaping.algorithm.signal_processing.gerchberg_saxton import (
    gerchberg_saxton,
    adaptive_gerchberg_saxton,
    angular_spectrum_propagate,
    calculate_reconstruction_error,
    GSResult,
)

from ao_shaping.algorithm.signal_processing.controller import (
    ControlLaw,
    LoopConfig,
    HardwareConfig,
)

from ao_shaping.algorithm.signal_processing.phase_wrap import (
    PhaseWrapOptimizer,
    SLMPhaseController,
)

from ao_shaping.algorithm.signal_processing.iterative_base import IterativeOptimizer

from ao_shaping.algorithm.heuristic.ga import (
    GeneticAlgorithm,
    GAParams,
    tournament_selection,
    blend_crossover,
    gaussian_mutation,
    minimize_ga,
)

from ao_shaping.algorithm.heuristic.pso import (
    ParticleSwarmOptimizer,
    PSOParams,
    minimize_pso,
)

from ao_shaping.algorithm.heuristic.simulated_annealing import (
    SimulatedAnnealing,
    SAParams,
    TempSchedule,
    minimize_sa,
)

from ao_shaping.algorithm.heuristic.heuristic_base import (
    HeuristicOptimizer,
    OptimizerConfig,
    OptimizerType,
)

from ao_shaping.algorithm.heuristic.search import (
    HEURISTIC_ALGORITHM_MAP,
    HeuristicSearchResult,
    SearchAborted,
    create_heuristic,
    heuristic_algorithm_choices,
    run_heuristic_search,
)

from ao_shaping.algorithm.heuristic.hill_climbing import (
    HillClimbing,
    HCConfig,
)

from ao_shaping.algorithm.heuristic.random_search import RandomSearch

from ao_shaping.algorithm.heuristic.cross_entropy import (
    CrossEntropyMethod,
    CEMConfig,
)

from ao_shaping.algorithm.heuristic.differential_evolution import (
    DifferentialEvolution,
    DEConfig,
)

from ao_shaping.algorithm.signal_processing.differentiable_beam import (
    DifferentiableBeamOptimizer,
    differentiable_far_field,
    far_field_intensity,
)

__all__ = [
    "Base",
    "SGD",
    "Adam",
    "AdamW",
    "AdaMOD",
    "learning_schedule",
    "ImageTargetFunc",
    "TabuMemory",
    "AdaptiveSearchState",
    "generate_search_candidates",
    "should_trigger_search",
    "TabuSearchRunner",
    "create_tabu_search_runner",
    "gerchberg_saxton",
    "adaptive_gerchberg_saxton",
    "angular_spectrum_propagate",
    "calculate_reconstruction_error",
    "GSResult",
    "ControlLaw",
    "LoopConfig",
    "HardwareConfig",
    "PhaseWrapOptimizer",
    "SLMPhaseController",
    "IterativeOptimizer",
    "GeneticAlgorithm",
    "GAParams",
    "tournament_selection",
    "blend_crossover",
    "gaussian_mutation",
    "minimize_ga",
    "ParticleSwarmOptimizer",
    "PSOParams",
    "minimize_pso",
    "SimulatedAnnealing",
    "SAParams",
    "TempSchedule",
    "minimize_sa",
    "HeuristicOptimizer",
    "OptimizerConfig",
    "OptimizerType",
    "HEURISTIC_ALGORITHM_MAP",
    "HeuristicSearchResult",
    "SearchAborted",
    "create_heuristic",
    "heuristic_algorithm_choices",
    "run_heuristic_search",
    "HillClimbing",
    "HCConfig",
    "RandomSearch",
    "CrossEntropyMethod",
    "CEMConfig",
    "DifferentialEvolution",
    "DEConfig",
    "DifferentiableBeamOptimizer",
    "differentiable_far_field",
    "far_field_intensity",
]


def is_cython_available():
    return CYTHON_AVAILABLE


def is_target_func_cython_available():
    return CYTHON_TARGET_FUNC_AVAILABLE
