"""campaign 阶段名 → 中文（桌面逐块页的段名映射，无 Qt）。

原「计划预览」（4 形状 × 4 阶段矩阵 + 段级成本表）已按用户要求移除；
这里只保留 ⑥「整轮逐块」页仍在用的阶段名映射。矩阵 / 成本估算不再在
GUI 里预先摆出来，需要时直接看逐块页的实跑产物。
"""

from __future__ import annotations

__all__ = ["CAMPAIGN_STAGE_LABELS"]

#: campaign 阶段名 → 中文（键集与 ``stages.STAGE_RECIPES`` 的 ``name`` 一致）。
#:
#: 与 ``params.STAGE_LABELS`` 是两套：那个表讲**单会话**的阶段（采集 / 拟合 /
#: 迁移…），这里讲 campaign 的整形阶段配方。同名键（``transport_cv`` 等）不会同时
#: 出现在两张表里，所以不合并。
CAMPAIGN_STAGE_LABELS: dict[str, str] = {
    "transport_cv": "传输相位 + CV",
    "mono12_cv": "12-bit 连拍 + CV",
    "mono12_continue": "12-bit 续训",
    "mono12_dense": "12-bit 密集（宽脊模型）",
}
