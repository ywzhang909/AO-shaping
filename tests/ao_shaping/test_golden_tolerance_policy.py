"""#251 P1-B — 浮点 golden 序列必须走标定容差, 禁止逐位 `==`。

惯例出自 #205 (「漂移 ×10、变异 ÷100」) 并在 ci169 再度应验:
`test_none_schedule_reproduces_pre_change_trajectory` 用 `round(x, 6)` 逐位钉
轨迹, 跨 torch 构建 1 ULP 漂移直接把 CI 打红。惯例躺在评论里就还会复发 ——
本守卫把它变成静态约束:

* 长度 ≥9、多数元素是浮点字面量的 List/Tuple (golden 序列), 或
  `[round(...) for ...]` 推导式, 出现在 `==`/`!=` 一侧时,
  该比较必须整体套上容差比较器 (`pytest.approx` / `np.allclose` /
  `np.testing.assert_allclose` / `isclose`);
* 纯整数/布尔/字符串序列不受管 (离散量逐位相等是设计, 不是漂移);
* golden 名字只作为**输入**喂给函数 (如舍入样本表) 不受管 —— 只有出现在
  比较的一侧才算 golden 断言。

豁免 `TOLERANCE_EXEMPT` 逐条带理由, 只减不增。
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_TESTS = _ROOT / "tests" / "ao_shaping"

_TOLERANT_FUNCS = {"approx", "allclose", "assert_allclose", "isclose", "assert_near"}

#: (相对 tests/ao_shaping 的路径, 规则描述) -> 理由。只减不增。
TOLERANCE_EXEMPT: dict[tuple[str, str], str] = {
    (
        "utils/test_auto_camera_finders.py",
        "config-ladder",
    ): "曝光探针梯是**配置常量** (字面量原样透传给 reset_window, 无任何算术) —— "
    "逐位相等是设计; 跨构建漂移只发生在计算所得序列",
}


def _is_floatish_golden(node: ast.AST) -> bool:
    """≥9 个常量、≥60% 为 float 的 list/tuple 字面量。"""
    if not isinstance(node, ast.List | ast.Tuple):
        return False
    elts = node.elts
    if len(elts) < 9:
        return False
    consts = [e.value for e in elts if isinstance(e, ast.Constant)]
    if len(consts) < len(elts):  # 含非常量元素 (如 Name) 的表交给 golden-name 规则
        return False
    floats = sum(1 for v in consts if isinstance(v, float))
    return floats >= max(2, int(len(consts) * 0.6))


def _has_round_comprehension(node: ast.AST) -> bool:
    if not isinstance(node, ast.ListComp):
        return False
    return any(
        isinstance(c, ast.Call) and getattr(c.func, "id", "") == "round"
        for c in ast.walk(node)
    )


def _mentions_direct(node: ast.AST, names: set[str]) -> bool:
    """golden 名是否**直接**是被比较操作数 (不再下探任何 Call 的参数子树)。

    ``round_to_int(SAMPLE) == [ints]`` 里 SAMPLE 是变换的输入而非比较对象,
    逐位相等是对整数输出的设计 —— 不算 golden 断言; ``== (self.REF,)`` 才算。
    """
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Call):
            continue
        if isinstance(n, ast.Name) and n.id in names:
            return True
        if isinstance(n, ast.Attribute) and n.attr in names:
            return True
        stack.extend(ast.iter_child_nodes(n))
    return False


def _golden_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for st in body:
            if isinstance(st, ast.Assign) and isinstance(st.value, ast.List | ast.Tuple):
                vals = st.value.elts
                consts = [e.value for e in vals if isinstance(e, ast.Constant)]
                floats = sum(1 for v in consts if isinstance(v, float))
                if len(vals) >= 9 and floats >= max(2, int(len(vals) * 0.6)):
                    out.update(t.id for t in st.targets if isinstance(t, ast.Name))
    return out


def _compare_is_tolerant(cmp_node: ast.AST) -> bool:
    for sub in ast.walk(cmp_node):
        if isinstance(sub, ast.Call):
            name = getattr(sub.func, "attr", None) or getattr(sub.func, "id", None)
            if name in _TOLERANT_FUNCS:
                return True
    return False


def _violations_in(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    goldens = _golden_names(tree)
    bad: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, ast.Eq | ast.NotEq) for op in node.ops):
            continue
        sides = [node.left, *[c for c in node.comparators]]
        lit = any(_is_floatish_golden(s) for s in sides)
        direct = any(_mentions_direct(s, goldens) for s in sides)
        roundcomp = any(_has_round_comprehension(s) for s in sides)
        # round 推导式单独不构成漂移通道 (确定性纯 Python 算术逐位安全);
        # ci169 的模式是 round-comp **对齐 golden 序列** —— 组合才算。
        golden_side = lit or direct or (roundcomp and (lit or direct))
        if golden_side and not _compare_is_tolerant(node):
            bad.append((node.lineno, "bare ==/!= against a float golden sequence"))
    return bad


def test_float_golden_comparisons_use_calibrated_tolerance() -> None:
    offenders: list[str] = []
    for f in sorted(_TESTS.rglob("test_*.py")):
        rel = f.relative_to(_TESTS).as_posix()
        for lineno, why in _violations_in(f.read_text(encoding="utf-8")):
            key = (rel, f"line~{lineno}")
            if any(rel == k[0] for k in TOLERANCE_EXEMPT):
                continue
            offenders.append(f"{rel}:{lineno}: {why}")
    assert not offenders, (
        "浮点 golden 逐位相等断言 (#251 P1-B) —— 跨构建 ~1 ULP 漂移必红; "
        "改 `pytest.approx(..., abs=漂移×10)` 或 `np.testing.assert_allclose`: "
        f"{offenders}"
    )


def test_the_guard_itself_catches_bare_equality() -> None:
    """负样本 (验收): round()+== 打红; approx 放行; 输入表不误伤。"""
    bare = (
        "REF = (2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9)\n\n\n"
        "def test_x():\n"
        "    assert [round(v, 6) for v in hist] == REF\n"
    )
    assert _violations_in(bare), "负样本没被抓住 —— 守卫失效"
    tolerant = (
        "REF = (2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9)\n\n\n"
        "def test_x():\n"
        "    assert hist == pytest.approx(REF, abs=5e-6)\n"
    )
    assert not _violations_in(tolerant)
    input_only = (
        "SAMPLE = (2.7, 99.5, -2.7, 1.5, 0.5, 8.9, 3.2, 7.7, 4.4, 6.1)\n\n\n"
        "def test_x():\n"
        "    assert round_to_int(SAMPLE) == [3, 99, -2, 1, 0, 8, 3, 7, 4, 6]\n"
    )
    assert not _violations_in(input_only), "输入表+整数期望被误伤"
    short_round = (
        "def test_x():\n"
        "    assert [round(v, 1) for v in times] == [2.0, 4.0]\n"
    )
    assert not _violations_in(short_round), "短确定性 round 表被误伤"
