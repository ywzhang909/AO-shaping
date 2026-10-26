"""物理预测核心单一导入守卫（issue #199 R2+P5）。

R2/P5 把两份物理预测（``differentiable_defocus_uniformity`` 与
``physical_forward_fit``）的公共核心抽到
:mod:`ao_shaping.algorithm.signal_processing.defocus_forward_core`。本守卫
钉住这个结构不再漂移：

* **禁第二份实现**：core 的专属符号（``defocus_phase`` / ``panel_grid`` /
  ``gaussian_amplitude`` / ``subpixel_offsets`` / ``camera_sample_grids`` /
  ``sample_intensity`` / ``irradiance_scale``）在整个 ``src/`` 只能定义于
  core 模块；
* **``diopter_scale``** 允许两处：core 的定义 + ddu 的 ``@property`` 委托
  （Step 4 为保留公开 API 刻意保留的薄委托，函数体必须是单行
  ``return diopter_scale(...)``）；
* **正向共享**：两个前向模块都必须 ``from ... defocus_forward_core
  import`` 至少一个 core 公开符号。

范围外记录：``utils/wavefront/fraunhofer.py::focal_intensity`` 是同名的
**不同职责**原语（纯 Fraunhofer 传播 ``|F|²``，无离焦/ramp/插值重采样），
不在 R2 的两份之列，故 ``focal_intensity`` 不进专属符号集；波长字面量
``1.064e-06`` 在全 src 有 20+ 处历史散布（ml/sim/display 等），超 #199
范围，由 Step 4 的孪生逐位测试与 golden 覆盖其行为风险。
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__import__("ao_shaping").__file__).resolve().parents[0]

CORE_MODULE = "algorithm/signal_processing/defocus_forward_core.py"
CORE_DOTTED = "ao_shaping.algorithm.signal_processing.defocus_forward_core"

#: Symbols that must be defined exactly once, inside the core module.
EXCLUSIVE_SYMBOLS = frozenset(
    {
        "defocus_phase",
        "panel_grid",
        "gaussian_amplitude",
        "subpixel_offsets",
        "camera_sample_grids",
        "sample_intensity",
        "irradiance_scale",
    }
)

#: The two physical-forward consumers that must import from the core.
FORWARD_MODULES = (
    "algorithm/signal_processing/differentiable_defocus_uniformity.py",
    "algorithm/signal_processing/physical_forward_fit.py",
)


def _iter_module_trees():
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path, path.relative_to(SRC_ROOT).as_posix(), ast.parse(
            path.read_text(encoding="utf-8")
        )


def _definitions_of(name: str) -> list[tuple[str, int, ast.FunctionDef | ast.AsyncFunctionDef]]:
    found = []
    for _path, rel, tree in _iter_module_trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                found.append((rel, node.lineno, node))
    return found


def _is_property_delegation(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """``@property`` whose body is a single ``return <same-name>(...)`` call."""
    if not any(
        isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list
    ):
        return False
    body = node.body
    # Strip a leading docstring: it is an Expr(Constant(str)) statement, not logic.
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    call = body[0].value
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == node.name
    )


def test_core_symbols_are_defined_only_in_the_core_module() -> None:
    """core 专属符号全 src 唯一定义，且落在 core 模块。"""
    for symbol in sorted(EXCLUSIVE_SYMBOLS):
        definitions = _definitions_of(symbol)
        locations = [f"{rel}:{lineno}" for rel, lineno, _ in definitions]
        assert len(definitions) == 1, (
            f"{symbol} must be defined exactly once in {CORE_MODULE}, got: {locations}"
        )
        assert locations[0].startswith(CORE_MODULE), (
            f"{symbol} defined outside the core module: {locations[0]}"
        )


def test_diopter_scale_is_core_definition_plus_one_property_delegation() -> None:
    """diopter_scale：core 一处定义 + ddu 一处 property 委托，无第三份。"""
    definitions = _definitions_of("diopter_scale")
    locations = [(rel, node) for rel, _lineno, node in definitions]
    assert len(definitions) == 2, (
        "diopter_scale must have exactly two definitions (core + the ddu property "
        f"delegation), got: {[f'{rel}:{node.lineno}' for rel, node in locations]}"
    )
    core = [n for rel, n in locations if rel == CORE_MODULE]
    delegates = [n for rel, n in locations if rel != CORE_MODULE]
    assert len(core) == 1, "the core module must define diopter_scale"
    assert len(delegates) == 1 and _is_property_delegation(delegates[0]), (
        "the non-core diopter_scale must be a single-return @property delegation"
    )


def test_both_forwards_import_from_the_core() -> None:
    """两个前向模块都从 core import（正向共享断言）。"""
    for rel in FORWARD_MODULES:
        path = SRC_ROOT / rel
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == CORE_DOTTED:
                imported.update(alias.name for alias in node.names)
        assert imported, f"{rel} does not import anything from {CORE_DOTTED}"
        assert imported & EXCLUSIVE_SYMBOLS or "diopter_scale" in imported, (
            f"{rel} imports from the core but none of the shared symbols: {sorted(imported)}"
        )
