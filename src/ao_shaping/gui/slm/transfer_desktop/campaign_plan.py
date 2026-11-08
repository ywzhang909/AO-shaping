"""campaign 计划预览的纯数据层：4 形状 × 4 阶段矩阵 + 段级成本（无 Qt、不抛）。

为什么单独一个模块
------------------
`transfer_desktop/page.py` 要展示的两块内容（campaign 矩阵、段级成本表）在
#242 里都要求「离线可预览」：不开设备、没有 Qt 事件循环也要能算出同样的数。
把这些数算在 Qt 页面里，就只能靠离屏 Qt 才测得动；算在这里，纯函数直接单测，
页面只负责摆放。

口径来源（不要另立一套）
------------------------
- 矩阵/块数/迭代数：``tools/slm/transfer/campaign.campaign_matrix`` +
  ``matrix_totals``（4 形状 → 16 段 / 112 块）；
- 墙钟与落盘：``campaign.estimate_seconds`` / ``estimate_disk_gb``（块数 × 24 s/块
  与每阶段静态基线，capture / fit / 重放不计入）；
- 实测段成本：``tools/slm/transfer/cost_table.read_campaign_costs``（段目录
  ``cost.json``，按 ``<形状>/<阶段>`` 归并）+ ``estimate_campaign`` 的历史每块秒。

本模块**不 import** Qt / Streamlit / 任何设备层，也不改 ``campaign`` / ``cost_table``
的判定（#242 边界：非 GUI 层只读消费）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ao_shaping.gui.slm.transfer_desktop import params
from ao_shaping.tools.slm.transfer.campaign import (
    SECONDS_PER_BLOCK,
    campaign_matrix,
    estimate_disk_gb,
    estimate_seconds,
    matrix_totals,
)
from ao_shaping.tools.slm.transfer.cost_table import (
    estimate_campaign,
    read_campaign_costs,
)

__all__ = [
    "CAMPAIGN_STAGE_LABELS",
    "PLAN_SHAPES",
    "CampaignSummary",
    "CostRow",
    "MatrixCell",
    "campaign_summary",
    "cost_caption",
    "cost_rows",
    "matrix_cells",
]

#: campaign 矩阵覆盖的形状：与页面左上「目标形状」四个单选同一份词表。
PLAN_SHAPES: tuple[str, ...] = tuple(params.SHAPE_CHOICES)

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


@dataclass(frozen=True)
class MatrixCell:
    """矩阵里的一格 = 一个形状的一个阶段。

    Attributes:
        shape: 形状键（``square`` …）。
        shape_label: 形状的中文名（与页面单选同一份 ``params.SHAPE_LABELS``）。
        stage: 阶段名（``stages.STAGE_RECIPES`` 的 ``name``）。
        stage_label: 阶段中文名；未知阶段退回阶段名本身（新增阶段不会炸表）。
        blocks: 该阶段块数（一块 = 一次 A1(+B) 采集）。
        bit_depth: 该阶段相机位深。
        updates: 每块优化迭代数。
        highlight: 是否是「本会话正在跑的那个形状」——页面用它标出所在行。
    """

    shape: str
    shape_label: str
    stage: str
    stage_label: str
    blocks: int
    bit_depth: int
    updates: int
    highlight: bool = False


@dataclass(frozen=True)
class CampaignSummary:
    """一整轮 4 形状 campaign 的规模与成本估算（矩阵合计 + 墙钟 + 落盘）。"""

    segments: int
    blocks: int
    updates: int
    seconds_low: float
    seconds_expected: float
    seconds_high: float
    disk_gb: float

    @property
    def text(self) -> str:
        """一行可显示的中文摘要（含口径），供矩阵预览的合计行使用。"""
        return (
            f"合计 {self.segments} 段 / {self.blocks} 块 / {self.updates} updates"
            f" · 整形墙钟 ≈{self.seconds_expected / 60:.0f} min"
            f"（{self.seconds_low / 60:.0f}–{self.seconds_high / 60:.0f} min）"
            f" · 预计落盘 ≈{self.disk_gb:.1f} GB"
            f" · 口径：{self.blocks} 块 × {SECONDS_PER_BLOCK / 60:g} min/块与每阶段"
            "静态基线，capture / fit / 重放不计入"
        )


@dataclass(frozen=True)
class CostRow:
    """段级成本表的一行：计划 vs 实测。

    Attributes:
        shape: 形状键。
        shape_label: 形状中文名。
        stage: 阶段名。
        stage_label: 阶段中文名。
        planned_blocks: 计划块数（矩阵给出）。
        expected_seconds: 计划秒数 = 计划块数 × :data:`SECONDS_PER_BLOCK`。
        measured_seconds: 段目录 ``cost.json`` 的实测秒数；没有实测时为 ``None``
            （**不是** 0 —— 「没测过」和「测出来是 0 秒」是两件事）。
        measured_blocks: 实测块数；没有实测或该段只有段首行时为 ``None``。
        delta_text: 实测相对计划的偏差文案（``"快 12%"`` / ``"慢 8%"`` / ``"—"``）。
    """

    shape: str
    shape_label: str
    stage: str
    stage_label: str
    planned_blocks: int
    expected_seconds: float
    measured_seconds: float | None
    measured_blocks: int | None
    delta_text: str


def _shape_label(shape: str) -> str:
    return params.SHAPE_LABELS.get(shape, shape)


def _stage_label(stage: str) -> str:
    return CAMPAIGN_STAGE_LABELS.get(stage, stage)


def matrix_cells(current_shape: str | None = None) -> tuple[MatrixCell, ...]:
    """4 形状 × 4 阶段的 campaign 矩阵（16 格），``current_shape`` 的行被标出。"""
    cells: list[MatrixCell] = []
    for row in campaign_matrix(PLAN_SHAPES):
        cells.append(
            MatrixCell(
                shape=row.shape,
                shape_label=_shape_label(row.shape),
                stage=row.stage,
                stage_label=_stage_label(row.stage),
                blocks=row.blocks,
                bit_depth=row.bit_depth,
                updates=row.updates,
                highlight=current_shape is not None and row.shape == current_shape,
            )
        )
    return tuple(cells)


def campaign_summary() -> CampaignSummary:
    """矩阵合计 + 墙钟三档 + 落盘估算（全部来自 ``campaign``，不重算物理）。"""
    rows = campaign_matrix(PLAN_SHAPES)
    totals = matrix_totals(rows)
    low, expected, high = estimate_seconds(rows)
    return CampaignSummary(
        segments=int(totals["segments"]),
        blocks=int(totals["blocks"]),
        updates=int(totals["updates"]),
        seconds_low=float(low),
        seconds_expected=float(expected),
        seconds_high=float(high),
        disk_gb=float(estimate_disk_gb(rows)),
    )


def _measured(costs: dict[str, dict[str, Any]], shape: str, stage: str) -> dict[str, Any]:
    return costs.get(f"{shape}/{stage}") or {}


def _delta_text(expected: float, measured: float | None) -> str:
    """实测相对计划的偏差文案；读数不可用时是破折号而不是 0 %。"""
    if measured is None or expected <= 0:
        return "—"
    ratio = measured / expected - 1.0
    if abs(ratio) < 0.005:
        return "≈计划"
    percent = abs(ratio) * 100.0
    return f"{'快' if ratio < 0 else '慢'} {percent:.0f}%"


def cost_rows(campaign_dir: str | Path | None = None) -> tuple[CostRow, ...]:
    """段级成本表：每个矩阵格一行，``campaign_dir`` 给了就把实测列填上。

    没有 ``campaign_dir``（或目录里还没有 ``cost.json``）时实测列全为 ``None``，
    表里只剩计划列 —— 这正是「计划页离线可预览」的形态，而不是印一张看起来免费的
    0 表（口径见 ``cost_table`` 模块 docstring）。
    """
    try:
        costs = read_campaign_costs(campaign_dir) if campaign_dir is not None else {}
    except (OSError, TypeError, ValueError):
        # 显示路径兜底：坏目录 / 怪类型只让实测列空着，计划列照印。
        costs = {}

    rows: list[CostRow] = []
    for cell in matrix_cells():
        expected = cell.blocks * SECONDS_PER_BLOCK
        measured = _measured(costs, cell.shape, cell.stage)
        seconds = measured.get("seconds")
        blocks = measured.get("blocks")
        measured_seconds = float(seconds) if isinstance(seconds, (int, float)) else None
        measured_blocks = (
            int(blocks) if isinstance(blocks, int) and blocks > 0 else None
        )
        rows.append(
            CostRow(
                shape=cell.shape,
                shape_label=cell.shape_label,
                stage=cell.stage,
                stage_label=cell.stage_label,
                planned_blocks=cell.blocks,
                expected_seconds=expected,
                measured_seconds=measured_seconds,
                measured_blocks=measured_blocks,
                delta_text=_delta_text(expected, measured_seconds),
            )
        )
    return tuple(rows)


def cost_caption(campaign_dir: str | Path | None = None) -> str:
    """成本表脚下的一行口径：历史每块秒基准 → 本计划的外推区间。

    没有历史时说「首次运行」，不印 0；``campaign_dir`` 不可读时同样退化成这一句
    （显示路径不抛）。
    """
    try:
        costs = read_campaign_costs(campaign_dir) if campaign_dir is not None else {}
        planned = sum(cell.blocks for cell in matrix_cells())
        estimate = estimate_campaign(costs, planned_blocks=planned)
        basis = estimate.get("basis_seconds_per_block") or {}
        mid = basis.get("mid")
        if mid is None:
            return (
                f"首次运行：{campaign_dir or '当前'} 下还没有可用的段级成本，"
                f"实测列需跑完一段后才有数（计划列按 {SECONDS_PER_BLOCK / 60:g} min/块）"
            )
        low, high = basis.get("low"), basis.get("high")
        return (
            f"历史基准 {float(mid):.1f} s/块"
            f"（最快 {float(low):.1f} / 最慢 {float(high):.1f}，{int(basis.get('rows') or 0)} 段实测）"
            f" → 本计划 {planned} 块 ≈{float(estimate['mid']) / 60:.0f} min"
            f"（{float(estimate['low']) / 60:.0f}–{float(estimate['high']) / 60:.0f} min）"
        )
    except (OSError, TypeError, ValueError):
        # 显示路径兜底：读不成历史就说这一句，不抛给 Qt 事件循环。
        return "成本基准读取失败：本表只剩计划列（不影响运行）"
