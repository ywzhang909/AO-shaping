# 尝试导入 Cython 优化版本，如果失败则回退到纯 Python 版本
try:
    from .adam_cython import Base, SGD, Adam, AdamW, AdaMOD, learning_schedule

    CYTHON_AVAILABLE = True
except ImportError:
    from .adam import Base, SGD, Adam, AdamW, AdaMOD, learning_schedule

    CYTHON_AVAILABLE = False

try:
    from .target_func_cython import ImageTargetFunc

    CYTHON_TARGET_FUNC_AVAILABLE = True
except ImportError:
    from .target_func import ImageTargetFunc

    CYTHON_TARGET_FUNC_AVAILABLE = False

# 导入 Tabu Search 模块
from .tabu_search import (
    TabuMemory,
    AdaptiveSearchState,
    generate_search_candidates,
    should_trigger_search,
    TabuSearchRunner,
    create_tabu_search_runner,
)

# 导入 Gerchberg-Saxton 模块
from .gerchberg_saxton import (
    gerchberg_saxton,
    adaptive_gerchberg_saxton,
    angular_spectrum_propagate,
    calculate_reconstruction_error,
    GSResult,
)

# 导入闭环控制器模块
from .controller import (
    ControlLaw,
    LoopConfig,
    HardwareConfig,
)

# 导入相位包裹优化模块
from .phase_wrap import (
    PhaseWrapOptimizer,
    SLMPhaseController,
)

# 导入启发式优化算法模块
from .ga import (
    GeneticAlgorithm,
    GAParams,
    tournament_selection,
    blend_crossover,
    gaussian_mutation,
    minimize_ga,
)

from .pso import (
    ParticleSwarmOptimizer,
    PSOParams,
    minimize_pso,
)

from .simulated_annealing import (
    SimulatedAnnealing,
    SAParams,
    TempSchedule,
    minimize_sa,
)

from .heuristic_base import (
    HeuristicOptimizer,
    OptimizerConfig,
    OptimizerType,
)

from .hill_climbing import (
    HillClimbing,
    HCConfig,
)

from .random_search import RandomSearch

from .cross_entropy import (
    CrossEntropyMethod,
    CEMConfig,
)

from .differential_evolution import (
    DifferentialEvolution,
    DEConfig,
)

# 导出所有类和函数
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
    "HillClimbing",
    "HCConfig",
    "RandomSearch",
    "CrossEntropyMethod",
    "CEMConfig",
    "DifferentialEvolution",
    "DEConfig",
]


# 提供检查 Cython 是否可用的函数
def is_cython_available():
    return CYTHON_AVAILABLE


def is_target_func_cython_available():
    return CYTHON_TARGET_FUNC_AVAILABLE
