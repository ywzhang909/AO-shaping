"""形状词表单一来源守卫（issue #199 P1）。

词表 canonical 是
:data:`ao_shaping.utils.image.target.metrics.TARGET_SHAPE_CHOICES`，
领域子集（``POLYGON_SHAPE_CHOICES`` / ``DIFFERENTIABLE_SHAPE_CHOICES``）
是它的**视图**。issue #199 P1 清理了 6 处散落副本；本守卫防止再次漂移。

判据（AST 扫描 ``src/`` 全部 .py）：纯字符串 List/Tuple/Set 字面量，
成员全部来自词表、数量 >= 3，且成员集合**恰好等于**某个已登记视图
（全域或两个子集）时判违规 —— 这是"词表副本"的精确形态。

明确**不**算副本（豁免）：
* canonical 定义模块本身；
* ``Literal[...]`` 类型别名（Subscript 的 slice，如 ``TargetShape`` /
  ``ShapeName`` 与函数参数注解）——类型镜像是文档，不是运行时词表；
* ``in`` / ``not in`` 比较的操作数（``if shape in {"square", ...}`` 式
  分派表）；
* 收窄支持集（如 benchmark 套件只跑三种形状、wrapper 只接受四种）——
  那是功能限制，不等于任何登记视图，不抓。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ao_shaping.utils.image.target.metrics import (
    DIFFERENTIABLE_SHAPE_CHOICES,
    POLYGON_SHAPE_CHOICES,
    TARGET_SHAPE_CHOICES,
)

SRC_ROOT = Path(__import__("ao_shaping").__file__).resolve().parents[0]

#: The canonical definition module -- the one place allowed to spell the
#: vocabulary out.
CANONICAL_MODULE = "utils/image/target/metrics.py"

#: Registered views of the canonical vocabulary. A literal whose member set
#: equals one of these is a copy (drift); anything else is a narrowed set.
REGISTERED_VIEWS = (
    frozenset(TARGET_SHAPE_CHOICES),
    frozenset(POLYGON_SHAPE_CHOICES),
    frozenset(DIFFERENTIABLE_SHAPE_CHOICES),
)


def _iter_literal_nodes(tree: ast.AST):
    """Yield ``(node, parent)`` for every List/Tuple/Set literal."""
    stack: list[tuple[ast.AST, ast.AST | None]] = [(tree, None)]
    while stack:
        node, parent = stack.pop()
        for child in ast.iter_child_nodes(node):
            stack.append((child, node))
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            yield node, parent


def _is_literal_alias_slice(parent: ast.AST | None) -> bool:
    """True when the literal is the slice of ``Literal[...]`` / ``typing.Literal[...]``."""
    if not isinstance(parent, ast.Subscript):
        return False
    value = parent.value
    return (isinstance(value, ast.Name) and value.id == "Literal") or (
        isinstance(value, ast.Attribute) and value.attr == "Literal"
    )


def _is_comparison_operand(parent: ast.AST | None) -> bool:
    """True when the literal is an operand of an ``in`` / ``not in`` comparison."""
    return isinstance(parent, ast.Compare) and any(
        isinstance(op, (ast.In, ast.NotIn)) for op in parent.ops
    )


def _pure_string_members(node: ast.AST) -> frozenset[str] | None:
    """String members of a pure string literal, or None if any element is not a str constant."""
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)) or not node.elts:
        return None
    members: set[str] = set()
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            members.add(elt.value)
        else:
            return None
    return frozenset(members)


def _vocabulary_copies() -> list[str]:
    """Every inline copy of a registered vocabulary view under ``src/``."""
    offenders: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(SRC_ROOT).as_posix()
        if rel == CANONICAL_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node, parent in _iter_literal_nodes(tree):
            if _is_literal_alias_slice(parent) or _is_comparison_operand(parent):
                continue
            members = _pure_string_members(node)
            if members is None or len(members) < 3:
                continue
            if not members <= frozenset(TARGET_SHAPE_CHOICES):
                continue
            if members in REGISTERED_VIEWS:
                offenders.append(f"{rel}:{node.lineno} {sorted(members)}")
    return offenders


def test_no_inline_shape_vocabulary_copy_under_src() -> None:
    """src/ 中不存在任何已登记词表视图的内联副本。"""
    offenders = _vocabulary_copies()
    assert not offenders, (
        "inline copies of a registered shape-vocabulary view found; import the "
        f"view from ao_shaping.utils.image.target.metrics instead: {offenders}"
    )


def test_subset_views_are_views_of_the_canonical_vocabulary() -> None:
    """两个子集确实是 canonical 的子集视图（防视图自身漂移）。"""
    assert set(POLYGON_SHAPE_CHOICES) == {"square", "rectangle", "triangle", "star"}
    assert set(DIFFERENTIABLE_SHAPE_CHOICES) == {
        "square",
        "rectangle",
        "triangle",
        "star",
        "circle",
        "gaussian",
        "spot",
    }
    assert set(POLYGON_SHAPE_CHOICES) <= set(TARGET_SHAPE_CHOICES)
    assert set(DIFFERENTIABLE_SHAPE_CHOICES) <= set(TARGET_SHAPE_CHOICES)


@pytest.mark.parametrize("view", ["POLYGON_SHAPE_CHOICES", "DIFFERENTIABLE_SHAPE_CHOICES"])
def test_views_keep_canonical_order(view: str) -> None:
    """视图保持 canonical 的相对顺序（docstring 承诺 "order preserved"）。"""
    canonical = list(TARGET_SHAPE_CHOICES)
    members = list(
        POLYGON_SHAPE_CHOICES if view == "POLYGON_SHAPE_CHOICES" else DIFFERENTIABLE_SHAPE_CHOICES
    )
    assert members == [s for s in canonical if s in set(members)]
