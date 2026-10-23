"""方向检验判决的回归测试。

存在理由（2026-10-23）：判决曾只查"两向超 3σ 且异号"，于是**方向反了但幅度够**
（Δ+ +0.08 / Δ− −0.08）的记录会被判通过；同一批提交还一度在报告里宣称判据已收紧，
而代码没改（报告领先于代码）。本测试锁住语义：

    verdict = (Δ+ < −3σ) and (Δ− > +3σ)

即 B+ 必须让实测 CV **下降**、B− 必须让 CV **上升**。
"""

from __future__ import annotations

import pytest


def verdict(delta_plus: float, delta_minus: float, resolution: float) -> bool:
    """与 `direction_test_after_optics_change.py` 内一致的判决函数（语义副本）。"""
    return bool(delta_plus < -resolution and delta_minus > resolution)


@pytest.mark.parametrize(
    "delta_plus, delta_minus, resolution, expected",
    [
        (-0.1135, +0.1342, 0.0568, True),    # 1000 px 实测：方向对且都超 3σ
        (-0.0148, +0.0115, 0.0165, False),   # 60 px 实测：方向对但幅度不足
        (-0.0186, -0.0034, 0.0165, False),   # 平场：Δ− 为负（方向错）
        (+0.0800, -0.0800, 0.0165, False),   # 异号但方向反了 ⇒ 必须不通过
        (+0.0800, +0.0800, 0.0165, False),   # 同号正
        (-0.0568, +0.0568, 0.0568, False),   # 恰好等于阈值（严格不等号）
        (-0.0569, +0.0569, 0.0568, True),    # 刚过阈值
    ],
)
def test_verdict_requires_direction_and_significance(delta_plus, delta_minus, resolution, expected):
    assert verdict(delta_plus, delta_minus, resolution) is expected


def test_script_implements_the_same_rule():
    """源码必须真的按方向判，而不是只查异号。"""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[3] / "scripts/slm/direction_test_after_optics_change.py").read_text(
        encoding="utf-8")
    assert "dp < -resolution and dm > resolution" in source, "判决未收紧到方向性检查"
    assert "mask_offset_px" in source, "缺少掩膜一致性检查"
    assert "np.sign" not in source.split("record[\"verdict\"]")[1][:400], "仍在使用异号判据"
