"""OOPAO-backend probe shared by the two ``scripts/oopao/`` report generators.

Only the genuinely identical parts live here. ``assert_oopao_usable`` and
``collect_provenance`` are **deliberately NOT** shared: the two generators
self-test different things (``activate_arm`` vs the ``backend_arm`` context
managers, single availability probe vs a dual ``_oopao_enabled()`` check) and
record different provenance fields (7 recorded-value fields vs 3). Reading the
bodies showed an AST "same name" signal of 10 functions / 947 lines was a false
lead -- six of those names were generic boilerplate and the substantive ones
genuinely diverge. Forcing them together would weaken one report's self-check to
save a few lines.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["OopaoBackendUnavailableError", "oopao_package_dir"]


class OopaoBackendUnavailableError(RuntimeError):
    """OOPAO 后端不可用或路由与预期不符 —— 必须大声中止, 不得静默产出重复结果。"""


def oopao_package_dir() -> str:
    """已导入的 OOPAO 包目录; 未导入时返回 ``unknown``。

    An editable install resolves to ``libs/OOPAO/OOPAO``, so the report can show
    which tree was actually exercised rather than a hand-copied path.
    """
    module = sys.modules.get("OOPAO")
    paths = list(getattr(module, "__path__", []) or []) if module else []
    if not paths:
        return "unknown"
    return str(Path(paths[0]).resolve())
