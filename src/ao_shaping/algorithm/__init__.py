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
]


# 提供检查 Cython 是否可用的函数
def is_cython_available():
    return CYTHON_AVAILABLE


def is_target_func_cython_available():
    return CYTHON_TARGET_FUNC_AVAILABLE
