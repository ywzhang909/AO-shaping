"""高斯核单一实现守卫（issue #199 R4）。

R4 清理前仓库有 3 份逐字节相同的 ``_gaussian_kernel_1d``（square_shaping、
exposure_fusion、target/metrics 的 ``_loggrad_kernel`` 内联实现）。canonical
是 :func:`ao_shaping.utils.image.exposure_fusion._gaussian_kernel_1d`，
其余位置必须 import 而非重新实现。

本守卫双向钉住：
* **禁重复**：``src/`` 中名字包含 ``gaussian_kernel_1d`` 的函数定义只能有
  一处，且必须在 ``exposure_fusion.py``；
* **禁旁引**：任何 import ``_gaussian_kernel_1d`` 的模块都必须从
  ``exposure_fusion`` 导入（不存在"第二个家"）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.utils.image.target import metrics as target_metrics

SRC_ROOT = Path(__import__("ao_shaping").__file__).resolve().parents[0]

#: The single implementation.
CANONICAL_MODULE = "utils/image/exposure_fusion.py"
KERNEL_NAME = "_gaussian_kernel_1d"


def _iter_module_trees():
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path, path.relative_to(SRC_ROOT).as_posix(), ast.parse(
            path.read_text(encoding="utf-8")
        )


def _kernel_definitions() -> list[str]:
    """Every function definition whose name contains ``gaussian_kernel_1d``."""
    found: list[str] = []
    for _path, rel, tree in _iter_module_trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if "gaussian_kernel_1d" in node.name:
                    found.append(f"{rel}:{node.lineno} def {node.name}")
    return found


def _kernel_import_sources() -> dict[str, str]:
    """``module -> imported-from module`` for every ``_gaussian_kernel_1d`` import."""
    sources: dict[str, str] = {}
    for _path, rel, tree in _iter_module_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == KERNEL_NAME:
                        sources[rel] = node.module or ""
    return sources


def test_exactly_one_gaussian_kernel_implementation() -> None:
    """唯一实现：src/ 中只有 exposure_fusion.py 定义高斯核。"""
    definitions = _kernel_definitions()
    assert len(definitions) == 1, (
        f"expected exactly one {KERNEL_NAME} definition in {CANONICAL_MODULE}, "
        f"got: {definitions}"
    )
    assert definitions[0].startswith(CANONICAL_MODULE), definitions[0]


def test_every_kernel_import_comes_from_the_canonical_module() -> None:
    """所有 import 都来自 canonical 模块——不存在第二个家。"""
    sources = _kernel_import_sources()
    assert sources, f"no module imports {KERNEL_NAME}; the dedup lost its consumers?"
    wrong = {
        f"{rel} <- {module}"
        for rel, module in sources.items()
        if module != "ao_shaping.utils.image.exposure_fusion"
    }
    assert not wrong, f"{KERNEL_NAME} imported from a non-canonical module: {wrong}"


def test_the_two_historical_consumers_delegate_to_the_canonical_kernel() -> None:
    """两个历史消费者（square_shaping / target/metrics）确实委托 canonical。"""
    import ao_shaping.algorithm.signal_processing.square_shaping as square_shaping
    from ao_shaping.utils.image.exposure_fusion import _gaussian_kernel_1d

    assert square_shaping._gaussian_kernel_1d is _gaussian_kernel_1d
    assert target_metrics._gaussian_kernel_1d is _gaussian_kernel_1d
    # sigma == 0 is the documented identity kernel; both paths must agree.
    import numpy as np

    kernel = _gaussian_kernel_1d(0.0)
    assert kernel.size == 1 and kernel[0] == pytest.approx(1.0)
    assert np.isclose(square_shaping._gaussian_kernel_1d(1.5).sum(), 1.0)
