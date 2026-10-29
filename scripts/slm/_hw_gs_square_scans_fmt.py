"""GS 方扫报告的**文本与格式化小工具**。

从 ``generate_hw_gs_square_scans_report.py`` 抽出的一层叶子：只依赖标准库与
:mod:`_hw_gs_square_scans_model`，不含任何业务逻辑。

为什么单独一层：``_rel``（路径转仓库相对）、``_figname``（图名加戳）、
``_row_at``（按浮点键回查 CSV 行）这类函数被**每一个**下游模块用到。把它们
留在 facade 里，任何一次搬移都会把它们整包拉走，于是每个新模块都带着一份
重复的依赖尾巴；单独成层之后，下游只需要 import 这一个名字明确的模块。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence


# 与 facade 同在 scripts/slm/ 下，parents[2] 算出的仓库根一致。
# 用**逐个 insert** 而不是 for 循环：scripts/ 那层的守卫
# (tests/ao_shaping/scripts/test_script_file_form_imports.py) 靠解析
# `sys.path.insert` 的字面参数判断「这个脚本能不能以文件方式启动」，
# for 循环里的循环变量它解析不了（会整文件跳过，等于没检查）。
# src 在 scripts 之前：scripts/ml 会遮蔽 src/ml。
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from scripts.slm._hw_gs_square_scans_model import (
    Cell,
    FIG_TAG,
    ROOT,
    ROW_MATCH_ATOL,
    Run,
    VARIABLE,
)

def cell_of(row: dict[str, str], key: str, nd: int = 4, suffix: str = "") -> Cell:
    """从 CSV 的一行 (原文字典) 取一个单元。键缺失即报错, 绝不静默补 0。"""
    if key not in row:
        raise KeyError(f"CSV 列 '{key}' 不存在; 实际表头按实际表头解析, 见 load_stage")
    text = row[key]
    try:
        value = float(text)
    except ValueError as exc:  # pragma: no cover - 产物损坏时才会走到
        raise ValueError(f"列 '{key}' 的值 {text!r} 不是数字") from exc
    return Cell(value=value, nd=nd, suffix=suffix)

def derived(value: float, nd: int = 4, suffix: str = "") -> Cell:
    """现场推导出来的数 (差值、比值、评分分解项…)。"""
    return Cell(value=float(value), nd=nd, suffix=suffix)

def _cell_short(cell: dict[str, Any]) -> str:
    """表格/图注里的短标签: ``side=200 px`` / ``n_max=30`` / ``P=32 px``。"""
    stage = cell["stage"]
    value = float(cell["value"][VARIABLE[stage]])
    if stage == "sides":
        return f"side={value:.0f} px"
    if stage == "zernike":
        return f"n_max={value:.0f}"
    return f"P={value:.0f} px"

def _idx_of(values: Sequence[float], target: float) -> int:
    for i, v in enumerate(values):
        if abs(float(v) - float(target)) < ROW_MATCH_ATOL:
            return i
    raise ValueError(f"{target} 不在 {list(values)} 里")

def _row_at(rows: Sequence[dict[str, str]], key: str, value: float) -> dict[str, str]:
    """在 CSV 原文行里按某个数值列取一行 (取不到就报错, 不静默返回空)。"""
    for row in rows:
        if abs(float(row[key]) - float(value)) < ROW_MATCH_ATOL:
            return row
    raise KeyError(f"没有 {key} = {value} 的行")

def _join_x(xs: Sequence[float], indices: Sequence[int]) -> str:
    return " / ".join(f"{xs[i]:.0f}" for i in indices) if indices else "无"

def _pct(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:.2f}%" if whole else "-"

def _count_stage(run: Run, stage: str) -> int:
    return sum(1 for ev in run.settle_events if ev.stage == stage)

def _figname(stem: str) -> str:
    return f"{stem}{FIG_TAG}.png"

def _rel(path: Path | None) -> str:
    if path is None:
        return "(未提供)"
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()
