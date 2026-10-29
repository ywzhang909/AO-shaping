"""campaign 输出目录的纯读取器：``journal.json`` + 逐段 ``manifest.json`` → 进度表。

为什么需要它
------------
``slm-transfer-shapes`` 把一次多形状 × 四阶段 campaign 写成
``<out>/journal.json``（阶段 start/complete/reused）与
``<out>/shape/<shape>/<stage>/{manifest,summary,config}.json``（逐块读数）。
GUI 要显示"跑到第几段第几块、每段首末 CV、段内峰值"，只能在**跑到一半的目录**上读
—— 这正是它最容易写错的地方：最后一块的 ``manifest.json`` 可能正在写、
``summary.json`` 可能整个不存在（早期会话没有它）、目录可能根本还没建。

因此本模块的契约是：**不抛异常**。缺文件、半截 JSON、未知字段一律降级成
:data:`MISSING`（``"—"``），因为没有一行能读而让整个页面报错，比显示"未知"糟糕得多。

字段口径来自写侧代码，不是猜的
--------------------------------
- 每块一行 ``label``：``validate.run`` 用 ``capture(label, ...)`` 写出
  ``<shape>_<block>_A1`` / ``<shape>_<block>_B``；``transport_cv`` 只有 A1
  （``StageRecipe.ccd_check=False``，按模型损失验收）。
- 段的段首/段末：``<shape>_seed`` 与最后一次 ``<shape>_final`` —— 这两行是**同一相位
  的同/跨位深重测**（报告 §3.2 表注），所以"段内变化"只能由它们相减得到。
  ``<shape>_final_adaptive`` 与 ``<shape>_seed_replay`` 是**另外两次**采集，不属于段末。
- 每块的接受判定在 ``summary.json`` 里（``accepted`` / ``acceptance`` / ``accept_metric``
  / ``baseline`` / ``candidate``），**不在** ``manifest.json`` 里。
- 段状态在 ``journal.json`` 里，阶段键是 ``"<shape>/<stage>"``（``journal.record``）。
  状态**原样透传**，不校验白名单：``running`` / ``complete`` / ``reused`` / 协作停止
  新增的 ``stopped`` / 将来任何别的字符串，读到的就是读到的。只有 :data:`COMPLETE_STATUSES`
  里的值算"这段做完了"，``stopped`` 与未知状态一律不算（半途停下的段不能计入进度）。

两个峰值列，别混
----------------
``StageRow.peak_full`` 对**全部 manifest 行**（含 ``seed`` / ``seed_replay`` /
``_final``，不只是块行）取全幅 ``peak`` 的最大值 —— 这正是报告 §3.2「段内峰值」列的
口径（``multishape_scale_live_20261028.md`` 该表表注：「``段内峰值`` = 该阶段
``manifest.json`` **全部行**的最大全幅峰值」）。判**饱和余量**必须用全幅峰值对
``full_scale`` 比，所以那一列只能由它回答。
``StageRow.peak_inside`` 取**核心峰值** ``core_peak`` 的段内最大值，回答的是另一个
问题："目标框里有多亮"。两者都留，是因为一个数答不了两个问题 —— 用核心峰值判削顶会
漏掉核心外的零级，用全幅峰值判"框内亮度"则会被零级带偏。逐块的饱和计数另由
``BlockRow.saturated_px`` 承载。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ao_shaping.tools.slm.transfer.stages import STAGE_RECIPES, StageRecipe
from ao_shaping.utils.image.target import TRANSFER_SHAPE_CHOICES

__all__ = [
    "COMPLETE_STATUSES",
    "MISSING",
    "BlockRow",
    "CampaignProgress",
    "StageRow",
    "collect_campaign_progress",
]

#: 缺失/不可读字段的占位符。用 em dash 而不是 ``None``，因为 GUI 表格要直接印出来，
#: 且 ``None`` 与"这一行确实是 0"在显示上不可区分。
MISSING: str = "—"

#: 视为"这段已经完成"的 journal 状态。``reused`` 是 ``--resume`` 复用已有段时的写法
#: （``run_stages`` 里 ``journal.record(f"{shape}/{name}", "reused")``），跳过不等于没做完。
#: ``stopped``（协作式段间停止）与任何未知状态**不在**此列：停下的段还有块没跑，算完成
#: 会让进度条骗人。
COMPLETE_STATUSES: tuple[str, ...] = ("complete", "reused")

#: ``<shape>_<block>_A1`` / ``<shape>_<block>_B``。块号只在这里出现，
#: 不能按 manifest 行数推块数（每块 1 或 2 行）。
_BLOCK_LABEL = re.compile(r"^(?P<shape>.+)_(?P<block>\d+)_(?P<phase>A1|B)$")


@dataclass(frozen=True)
class BlockRow:
    """一个 CCD 块（一次 A1(+B) 采集）的读数。

    Attributes:
        shape: 形状名。
        stage: 阶段名。
        index: 块号（0 起）。
        cv: 该块**候选**读数的 CV（有 B 帧取 B，否则取 A1）；不可读为 :data:`MISSING`。
        saturated_px: 该块最后一帧的全幅饱和像素数。
        core_peak: 该块最后一帧的核心峰值（counts）。
        exposure_ms: 该块的曝光（ms）。
        accepted: ``summary.json`` 的接受判定；无 ``summary.json`` 为 :data:`MISSING`。
        note: 一句中文说明（A1→B 的 CV 变化 + 验收模式），可直接显示。
    """

    shape: str
    stage: str
    index: int
    cv: float | str
    saturated_px: int | str
    core_peak: float | str
    exposure_ms: float | str
    accepted: bool | str
    note: str


@dataclass(frozen=True)
class StageRow:
    """一个阶段（形状 × 阶段）的进度。

    Attributes:
        shape: 形状名。
        stage: 阶段名。
        status: journal 里的状态字符串**原样**（``complete`` / ``reused`` / ``running`` /
            ``stopped`` / 任何未知值，不校验）；目录在但 journal 缺该段为 :data:`MISSING`。
        blocks: 逐块读数（按块号升序）。只含**实际存在**的块 —— 段中途停下或缺 manifest
            时不会补造块行。
        seed_cv: 段首 CV（``<shape>_seed`` 行；缺失时退到第一块）。
        final_cv: 段末 CV（最后一次 ``<shape>_final`` 行）。
        peak_inside: 段内**核心**峰值最大值（``core_peak``），答"目标框里多亮"。
        peak_full: 段内**全幅**峰值最大值（全部 manifest 行的 ``peak``，含 seed /
            ``seed_replay`` / final），与报告 §3.2「段内峰值」列同口径，用来对
            ``full_scale`` 判饱和余量。没有任何一行带 ``peak``（旧 manifest）时为
            ``None``，**不是** ``0.0``。
        exposure_ms: 该段的曝光（段首行，缺失时退到第一块）。
    """

    shape: str
    stage: str
    status: str
    blocks: tuple[BlockRow, ...]
    seed_cv: float | str
    final_cv: float | str
    peak_inside: float | str
    peak_full: float | None
    exposure_ms: float | str


@dataclass(frozen=True)
class CampaignProgress:
    """一个 campaign 输出目录的只读快照。

    Attributes:
        out: 输出目录。
        stages: 逐段进度（形状按 :data:`TRANSFER_SHAPE_CHOICES` 的顺序，阶段按
            ``STAGE_RECIPES`` 的顺序）。
        journal: ``journal.json`` 的原始条目（缺失时为空元组）—— GUI 的"事件流"直接用。
    """

    out: Path
    stages: tuple[StageRow, ...]
    journal: tuple[dict[str, Any], ...]

    @property
    def totals(self) -> dict[str, int]:
        """合计：``segments_total/segments_done/blocks_total/blocks_done``。

        ``*_total`` 是**本目录里已出现的段**的计划量（块数按 ``STAGE_RECIPES``），
        所以跑到一半时 ``blocks_done < blocks_total`` 就是真实进度；要显示"整份
        campaign 的 16 段"请用
        :func:`ao_shaping.tools.slm.transfer.campaign.campaign_matrix` 的计划数。
        """
        segments_done = sum(1 for stage in self.stages if stage.status in COMPLETE_STATUSES)
        blocks_done = sum(len(stage.blocks) for stage in self.stages)
        blocks_total = sum(_planned_blocks(stage) for stage in self.stages)
        return {
            "segments_total": len(self.stages),
            "segments_done": segments_done,
            "blocks_total": blocks_total,
            "blocks_done": blocks_done,
        }


def collect_campaign_progress(out_dir: str | Path) -> CampaignProgress:
    """读一个 campaign 输出目录，返回逐段进度快照。

    只读文件，不开设备、不 import streamlit/torch。任何缺失或损坏的产物都降级成
    :data:`MISSING`/空元组（**从不抛异常**）：计划页会在"目录刚建、``journal.json``
    还没写"的时候被打开。

    Args:
        out_dir: ``--out`` 指向的目录。

    Returns:
        :class:`CampaignProgress`。目录不存在时返回空快照（``stages=()``）。
    """
    root = Path(out_dir)
    journal = _journal_entries(root)
    stages: list[StageRow] = []

    shapes_root = root / "shape"
    present = _shape_names(shapes_root)
    for shape in present:
        for recipe in STAGE_RECIPES:
            stage_dir = shapes_root / shape / recipe.name
            status = _stage_status(journal, shape, recipe, stage_dir)
            if status is None:
                continue
            stages.append(_stage_row(shape, recipe.name, status, stage_dir))

    return CampaignProgress(out=root, stages=tuple(stages), journal=journal)


def _planned_blocks(stage: StageRow) -> int:
    """该段的计划块数（``STAGE_RECIPES``）；未知阶段名退到"已读到的块数"。"""
    planned = next((recipe.blocks for recipe in STAGE_RECIPES if recipe.name == stage.stage),
                   len(stage.blocks))
    return max(planned, len(stage.blocks))


def _shape_names(shapes_root: Path) -> tuple[str, ...]:
    """``shape/`` 下的形状目录名，按词表顺序排；词表外的按字母序排在后面。"""
    try:
        names = {entry.name for entry in shapes_root.iterdir() if entry.is_dir()}
    except OSError:
        return ()
    known = [shape for shape in TRANSFER_SHAPE_CHOICES if shape in names]
    unknown = sorted(names.difference(TRANSFER_SHAPE_CHOICES))
    return tuple(known + unknown)


def _journal_entries(root: Path) -> tuple[dict[str, Any], ...]:
    """``journal.json`` 的条目（非列表或不可读时为空元组）。"""
    payload = _read_json(root / "journal.json")
    if not isinstance(payload, list):
        return ()
    return tuple(entry for entry in payload if isinstance(entry, dict))


def _stage_status(
    journal: tuple[dict[str, Any], ...], shape: str, recipe: StageRecipe, stage_dir: Path,
) -> str | None:
    """阶段状态：journal 为主口径，``manifest.json`` 只在 journal 缺该段时兜底。

    兜底是为了 ``--model-dir`` 复用或手工拷进来的目录：它们没有本会话的 journal 行，
    但 ``shape/<shape>/<stage>/manifest.json`` 是有的。

    journal 里读到的状态**原样**返回，不校验白名单 —— 包括协作停止写的 ``stopped``
    与将来任何新状态；只有"这段算不算做完"由 :data:`COMPLETE_STATUSES` 判。
    """
    key = f"{shape}/{recipe.name}"
    status = None
    for entry in journal:
        if entry.get("stage") == key and isinstance(entry.get("status"), str):
            status = entry["status"]
    if status is not None:
        return status
    rows = _manifest_rows(stage_dir)
    if rows is None:
        # 目录在但还没有 manifest（正在跑第一块，或写了一半）：显示"—"而不是整段消失。
        return MISSING if stage_dir.is_dir() else None
    block_count = len({_block_index(row, shape) for row in rows
                       if _block_index(row, shape) is not None})
    return "complete" if block_count >= recipe.blocks else "partial"


def _stage_row(shape: str, stage: str, status: str, stage_dir: Path) -> StageRow:
    """组装一个 :class:`StageRow`（逐块读数 + 段首/段末/峰值/曝光）。"""
    rows = _manifest_rows(stage_dir) or []
    summary = _summary_by_block(stage_dir)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        index = _block_index(row, shape)
        if index is not None:
            grouped.setdefault(index, []).append(row)

    blocks = tuple(
        _block_row(shape, stage, index, grouped[index], summary.get(index))
        for index in sorted(grouped)
    )

    seed_row = _row_with_label(rows, f"{shape}_seed")
    final_row = _row_with_label(rows, f"{shape}_final")
    if final_row is None:  # 兼容将来改名，但绝不把 *_final_adaptive 当段末
        final_row = next((row for row in rows
                          if isinstance(row.get("label"), str)
                          and row["label"].endswith("_final")), None)

    seed_cv = _cv(seed_row) if seed_row is not None else (blocks[0].cv if blocks else MISSING)
    final_cv = _cv(final_row) if final_row is not None else (
        blocks[-1].cv if blocks else MISSING)
    core_peaks = [_number(row.get("core_peak")) for row in rows]
    core_peaks = [peak for peak in core_peaks if isinstance(peak, float)]
    # 全幅峰值取**全部行**（seed / seed_replay / final_adaptive / final 都算），
    # 这是报告 §3.2「段内峰值」列的口径；旧 manifest 没有 ``peak`` 字段 ⇒ None。
    full_peaks = [_number(row.get("peak")) for row in rows]
    full_peaks = [peak for peak in full_peaks if isinstance(peak, float)]
    exposure_source = seed_row if seed_row is not None else (rows[0] if rows else None)
    exposure = _number(exposure_source.get("exposure_ms")) if exposure_source else MISSING

    return StageRow(
        shape=shape,
        stage=stage,
        status=status,
        blocks=blocks,
        seed_cv=seed_cv,
        final_cv=final_cv,
        peak_inside=max(core_peaks) if core_peaks else MISSING,
        peak_full=max(full_peaks) if full_peaks else None,
        exposure_ms=exposure,
    )


def _block_row(
    shape: str, stage: str, index: int, rows: list[dict[str, Any]],
    summary: dict[str, Any] | None,
) -> BlockRow:
    """一个块的读数：``cv`` 取**候选**帧（有 B 取 B），``accepted`` 取 summary。"""
    first, last = rows[0], rows[-1]
    candidate = _cv(last)
    baseline = _cv(first)
    if len(rows) > 1 and isinstance(baseline, float) and isinstance(candidate, float):
        note = f"A1 {baseline:.4f} → B {candidate:.4f}"
    elif isinstance(candidate, float):
        note = f"仅 A1 {candidate:.4f}（该段按模型损失验收，不采 B 帧）"
    else:
        note = "CV 不可读"
    if summary is not None:
        acceptance = summary.get("acceptance")
        metric = summary.get("accept_metric")
        if isinstance(acceptance, str) and isinstance(metric, str):
            note = f"{note} · {acceptance}/{metric}"
        elif isinstance(acceptance, str):
            note = f"{note} · {acceptance}"
    else:
        note = f"{note} · 无 summary.json"

    accepted: bool | str = MISSING
    if summary is not None and isinstance(summary.get("accepted"), bool):
        accepted = summary["accepted"]

    saturated = _integer(last.get("saturation_pixels"))
    return BlockRow(
        shape=shape,
        stage=stage,
        index=index,
        cv=candidate,
        saturated_px=saturated,
        core_peak=_number(last.get("core_peak")),
        exposure_ms=_number(last.get("exposure_ms")),
        accepted=accepted,
        note=note,
    )


def _manifest_rows(stage_dir: Path) -> list[dict[str, Any]] | None:
    """``manifest.json`` 的行列表；不存在或不是列表时返回 ``None``。"""
    payload = _read_json(stage_dir / "manifest.json")
    if not isinstance(payload, list):
        return None
    return [row for row in payload if isinstance(row, dict)]


def _summary_by_block(stage_dir: Path) -> dict[int, dict[str, Any]]:
    """``summary.json`` 按块号索引的判定条目（缺失/损坏时为空字典）。"""
    payload = _read_json(stage_dir / "summary.json")
    if not isinstance(payload, list):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for entry in payload:
        if isinstance(entry, dict) and isinstance(entry.get("block"), int):
            result[entry["block"]] = entry
    return result


def _block_index(row: dict[str, Any], shape: str) -> int | None:
    """从 ``label`` 取块号；不是本形状的块行时返回 ``None``。"""
    label = row.get("label")
    if not isinstance(label, str):
        return None
    match = _BLOCK_LABEL.match(label)
    if match is None or match.group("shape") != shape:
        return None
    return int(match.group("block"))


def _row_with_label(rows: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    """按 ``label`` 精确取行（段首/段末是**精确**标签，不做前缀匹配）。"""
    return next((row for row in rows if row.get("label") == label), None)


def _cv(row: dict[str, Any] | None) -> float | str:
    """一行里的 ``metrics.cv``。"""
    if row is None:
        return MISSING
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return MISSING
    return _number(metrics.get("cv"))


def _number(value: Any) -> float | str:
    """数值字段：可转成有限浮点时返回 ``float``，否则 :data:`MISSING`。"""
    if isinstance(value, bool) or value is None:
        return MISSING
    try:
        number = float(value)
    except (TypeError, ValueError):
        return MISSING
    return number if math.isfinite(number) else MISSING


def _integer(value: Any) -> int | str:
    """整数计数字段（饱和像素）。"""
    if isinstance(value, bool):
        return MISSING
    try:
        return int(value)
    except (TypeError, ValueError):
        return MISSING


def _read_json(path: Path) -> Any:
    """读 JSON；文件不存在、半截、编码坏掉都返回 ``None``（**从不抛**）。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
