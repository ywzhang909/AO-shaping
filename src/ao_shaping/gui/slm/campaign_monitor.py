"""campaign 运行监控：逐块 CV 轨迹、段间暂停闸、历史成本、统一进度字幕。

定位
----
``multishape_campaign_ui`` 是操作面，本模块是它**读数与措辞**的那一半：页面只留
三行调用（轨迹 / 闸 / 成本），口径与文案都在这里，于是"进度怎么写"只有一处实现。
放在单独模块而不是塞进页面，理由和 :mod:`gui.slm.transfer_progress` 相同：

1. 数据层（:func:`collect_campaign_traces` / :func:`progress_caption`）**不 import
   streamlit**，没有服务器也能单测；
2. 页面已经越过规模线，凡是能搬走的都该搬走（见
   ``scripts/check_module_size.py`` 的 ratchet）。

数据从哪来
----------
只读两份产物，**不重算任何指标**：

- ``<out>/journal.json`` —— 段的开始/结束（``transfer_shapes_runner._Journal`` 每写
  一行带 :func:`ao_shaping.tools.slm.transfer.capture.clock` 的
  ``timestamp_utc`` + ``monotonic_ns``，所以"已用多久"是实测墙钟，不是页面自己计的时）；
- ``<out>/shape/<形状>/<阶段>/manifest.json`` + ``summary.json`` —— 逐块读数，由
  :func:`~ao_shaping.tools.slm.transfer.campaign_progress.collect_campaign_progress`
  统一读取（它是 campaign 口径的唯一读取器，本模块只消费它的 :class:`StageRow` /
  :class:`BlockRow`）。**不画原始帧、不画逐像素数据**：轨迹图只画每块一个 CV 标量。

降级契约（为什么这里会有"尚未就绪"的说明）
------------------------------------------
段间闸与成本表各自属于
:mod:`~ao_shaping.tools.slm.transfer.campaign_control` 与
:mod:`~ao_shaping.tools.slm.transfer.cost_table`。它们与本模块并行落地，所以这里**惰性
import**、缺失时降级成一行说明 —— 绝不在本模块里再写一份"暂停请求文件长什么样"或
"历史成本怎么算"的第二实现：两份实现迟早给出两个数，而操作员只会信他先看到的那个。

成本表模块的调用约定（对不上就只剩一行提示，不会让页面报错）：
:func:`read_campaign_costs` 收 campaign 目录；:func:`estimate_campaign` 收成本行与
计划块数，返回带 ``low`` / ``mid`` / ``high``（秒）的映射。集成时若实际签名不同，
改的是 :func:`_cost_table` / :func:`_estimate_numbers` 这两处薄适配，不是页面。

键前缀
------
三个渲染函数都收 ``key_prefix``：本页的轨迹图、闸按钮、成本表在 session state /
DOM 里都带前缀，同一次会话里出现第二份（例如并排两个 campaign 目录）也不会撞键。
``st.line_chart``（streamlit 1.56）没有 ``key`` 参数，所以轨迹图外包一层
``st.container(key=...)`` 取得同一份唯一性。
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ao_shaping.gui.slm.transfer_progress import (
    SHAPE_LABELS,
    STAGE_LABELS,
    collect_progress,
    render_progress,
)
from ao_shaping.tools.slm.transfer.campaign import campaign_matrix, matrix_totals
from ao_shaping.tools.slm.transfer.campaign_progress import (
    COMPLETE_STATUSES,
    MISSING,
    collect_campaign_progress,
)
from ao_shaping.tools.slm.transfer.stages import STAGE_RECIPES

__all__ = [
    "collect_campaign_traces",
    "progress_caption",
    "render_campaign_traces",
    "render_cost_panel",
    "render_stage_gate",
    "journal_plan_shapes",
    "plan_shapes",
    "stage_table_rows",
    "block_table_rows",
    "render_progress_panel",
]

#: 段间闸的一句共同说明：停止点在哪、之后怎么接着跑。运行中与暂停请求两种状态下都要
#: 出现 —— 「停」和「续」是一件事的两半，分开放会让操作员以为停下就得重跑。
_GATE_CAPTION = (
    "停止点是**段边界**：当前段剩下的块照常跑完，journal 记 ``stopped``"
    "（不算完成，进度条不会把它算进去）。接着跑用 ⑥ 的「从断点续跑」："
    "已完成的段记 ``reused``，不会重做。"
)

#: 成本表的四列（顺序即显示顺序）。数值列缺失给 ``None`` 而不是 ``"—"``：
#: 同一列混排字符串会让 arrow 序列化失败，整张表从界面上消失（见
#: ``test_multishape_campaign_ui.py::test_progress_tables_survive_missing_values``）。
_COST_COLUMNS = ("段", "实测秒", "块数", "每块秒")


# --------------------------------------------------------------------------- #
# 数据层：逐块轨迹
# --------------------------------------------------------------------------- #
def collect_campaign_traces(out_dir: Path | str) -> list[dict[str, Any]]:
    """campaign 输出目录 → 逐块 CV 轨迹行（每块一行，跨形状/阶段按计划编号）。

    ``block`` 是**段序·块序**：每个形状从第 1 段的第 1 块起连续编号，段与段之间不
    重新计数（``transport_cv`` 4 块 ⇒ 下一段从 4 起）。这样一条折线跨段连续，段的
    交界落在整数档位上，读图的人能看出"这一段的改善是在第几块发生的"。

    ``cv`` / ``peak`` / ``exposure_ms`` 缺读数时是 ``None`` 而不是 :data:`MISSING`
    （``"—"``）：这三个数是拿去画图的，``"—"`` 在图上没有位置，``None`` 才是"这一点
    不可画"。``peak`` 取该块最后一帧的**核心峰值**（目标框内亮度），与
    :attr:`StageRow.peak_full`（全幅峰值，判饱和余量用）不是同一个口径。

    Args:
        out_dir: campaign 的 ``--out`` 目录（不存在时返回空列表，不抛异常）。

    Returns:
        行字典列表，键为 ``stage`` / ``shape`` / ``block`` / ``cv`` / ``peak`` /
        ``exposure_ms``。顺序按形状、阶段、块号（即读取器的顺序）。
    """
    progress = collect_campaign_progress(out_dir)
    offsets = _stage_offsets(progress)
    rows: list[dict[str, Any]] = []
    for stage in progress.stages:
        first = offsets.get((stage.shape, stage.stage), 0)
        for block in stage.blocks:
            rows.append({
                "stage": stage.stage,
                "shape": stage.shape,
                "block": first + block.index,
                "cv": _number(block.cv),
                "peak": _number(block.core_peak),
                "exposure_ms": _number(block.exposure_ms),
            })
    return rows


def _stage_offsets(progress: Any) -> dict[tuple[str, str], int]:
    """每个 (形状, 阶段) 的**段首块序**：前面各阶段的计划块数累加。

    用计划块数（:data:`STAGE_RECIPES`）而不是已读到的块数：某一段中途停下时，已读到
    4 块 ≠ 计划 12 块，拿前者累加会让后面所有段整体左移，两张图对不上同一条 x 轴。
    """
    planned = {recipe.name: recipe.blocks for recipe in STAGE_RECIPES}
    offsets: dict[tuple[str, str], int] = {}
    running: dict[str, int] = {}
    for stage in getattr(progress, "stages", ()) or ():
        key = (str(stage.shape), str(stage.stage))
        offsets[key] = running.get(key[0], 0)
        width = max(planned.get(key[1], 0), len(stage.blocks))
        running[key[0]] = offsets[key] + width
    return offsets


def _number(value: Any) -> float | None:
    """数字 → ``float``；``"—"`` / ``None`` / ``bool`` / 非有限值 → ``None``。

    ``bool`` 要单独挡掉：``True`` 是 ``int`` 的子类，漏过去就会在"验收"这种列上
    画成 1.0。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


# --------------------------------------------------------------------------- #
# 数据层：统一进度字幕
# --------------------------------------------------------------------------- #
def progress_caption(progress: Any, *, elapsed_s: float | None = None,
                     planned_shapes: Sequence[str] | None = None) -> str:
    """全页唯一的进度措辞：``当前段 {i}/{n} · 块 {k}/{m} · 已用 {hh:mm:ss}``。

    页面里凡是"跑到第几段第几块"的行都走这一句：两份手写格式迟早漂移成
    「段 4/16」与「第 4 段 / 共 16 段」，而操作员会以为它们在说两件事。

    分母取**计划**：``planned_shapes`` → journal 的 ``plan.shapes`` →
    目录里出现过的形状，再经 :func:`campaign_matrix` 换成段/块总数。不能拿
    :attr:`CampaignProgress.totals` 的 ``*_total`` 当分母 —— 它只数本目录里**已经
    出现**的段，跑到一半会诚实地写成「4/4」（issue #215 的现场）。

    ``planned_shapes`` 是对外契约之外的一个可选补充：页面在 journal 还没有
    ``plan`` 条目（新目录，第一行还没写）时知道"这次按哪几个形状跑"，那份信息只
    存在于页面设置里。不给就退到 journal / 目录，绝不猜。

    Args:
        progress: :class:`~ao_shaping.tools.slm.transfer.campaign_progress.CampaignProgress`
            （任何带 ``stages`` / ``journal`` / ``totals`` 的对象都行）。
        elapsed_s: 已用秒数；``None`` 时由 journal 的首末 ``timestamp_utc`` 现算。
        planned_shapes: 计划形状（可选，见上）。

    Returns:
        一句中国话，格式固定为上面的三段。
    """
    done_segments, total_segments, done_blocks, total_blocks = _caption_totals(
        progress, planned_shapes)
    seconds = elapsed_s if elapsed_s is not None else _journal_elapsed(progress)
    return (f"当前段 {done_segments}/{total_segments} · 块 {done_blocks}/{total_blocks}"
            f" · 已用 {_clock_text(seconds)}")


def _caption_totals(progress: Any,
                    planned_shapes: Sequence[str] | None) -> tuple[int, int, int, int]:
    """``(段完成, 段计划, 块完成, 块计划)``，计划侧失败时退到读取器自己的合计。"""
    totals = getattr(progress, "totals", None) or {}
    done_segments = int(totals.get("segments_done", 0))
    done_blocks = int(totals.get("blocks_done", 0))
    total_segments = int(totals.get("segments_total", 0))
    total_blocks = int(totals.get("blocks_total", 0))

    shapes = _planned_shapes(progress, planned_shapes)
    if shapes:
        try:
            plan = matrix_totals(campaign_matrix(list(shapes)))
        except ValueError as exc:  # 形状名非法/重复：退回读取器合计，不换掉这一行
            logger.warning("进度字幕的计划分母不可用（形状 {}）：{}", list(shapes), exc)
        else:
            total_segments, total_blocks = int(plan["segments"]), int(plan["blocks"])
    return done_segments, total_segments, done_blocks, total_blocks


def _planned_shapes(progress: Any, given: Sequence[str] | None) -> tuple[str, ...]:
    """本次 campaign **计划**跑哪些形状：入参 → journal 的 ``plan`` 条目 → 目录里出现过的。"""
    if given:
        return tuple(str(name) for name in given)
    for entry in reversed(tuple(getattr(progress, "journal", ()) or ())):
        plan = entry.get("plan") if isinstance(entry, Mapping) else None
        if isinstance(plan, Mapping) and plan.get("shapes"):
            return tuple(str(name) for name in plan["shapes"])
    return tuple(dict.fromkeys(str(stage.shape)
                               for stage in (getattr(progress, "stages", ()) or ())))


def _journal_elapsed(progress: Any) -> float | None:
    """journal 首末条目的墙钟差（秒）。

    用 ``timestamp_utc`` 而不是 ``monotonic_ns``：``monotonic`` 的零点随重启变化，
    而一份 campaign 跨过一次断电/重启就要接着跑（``--resume``），相减会得到负数或
    一个荒唐的小时数。``monotonic`` 只在读不到墙钟时兜底。
    """
    entries = tuple(getattr(progress, "journal", ()) or ())
    stamps: list[datetime] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        raw = entry.get("timestamp_utc")
        if not isinstance(raw, str):
            continue
        try:
            stamps.append(datetime.fromisoformat(raw))
        except ValueError:
            continue
    if len(stamps) >= 2 and stamps[-1] >= stamps[0]:
        return (stamps[-1] - stamps[0]).total_seconds()

    ticks = [entry.get("monotonic_ns") for entry in entries if isinstance(entry, Mapping)]
    ticks = [tick for tick in ticks if isinstance(tick, int)]
    if len(ticks) >= 2 and ticks[-1] >= ticks[0]:
        return (ticks[-1] - ticks[0]) / 1e9
    return None


def _clock_text(seconds: float | None) -> str:
    """秒 → ``hh:mm:ss``（``None``/负数 → ``00:00:00``；小时不折天，跑 30 小时也读得懂）。"""
    if seconds is None or math.isnan(seconds):
        return "00:00:00"
    total = max(0, int(round(seconds)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# --------------------------------------------------------------------------- #
# 渲染：逐块 CV 轨迹
# --------------------------------------------------------------------------- #
def render_campaign_traces(out_dir: Path | str, *, key_prefix: str = "cm") -> None:
    """一张折线：CV × 段序·块序，每个形状一条线（高度 240 px）。

    只画标量 CV，不画 CCD 帧、不画逐像素数据：这张图回答的是"优化还在往下走吗"，
    往回翻原始帧是另一个工具（``⑦ 历史会话`` 里按段看）。

    没有一条 CV 可画时印「尚无逐块数据」而不是空坐标轴 —— 空图会被读成"没有变化"。
    """
    import streamlit as st

    frame = _trace_frame(collect_campaign_traces(out_dir))
    if frame is None:
        st.caption("尚无逐块数据")
        return
    with st.container(key=f"{key_prefix}_traces"):
        st.markdown("**逐块 CV**")
        st.line_chart(frame, height=240)
    st.caption("横轴 = 段序·块序（每个形状从 0 起按**计划**块数连续编号，段与段之间不重新"
               "计数）；纵轴 = CV，**越低越好**。缺 CV 的块不画点（不补 0）；曝光与饱和"
               "见下方逐块表。")


def _trace_frame(rows: Sequence[Mapping[str, Any]]):
    """轨迹行 → 宽表（index = 段序·块序，一列一个形状）。一条可画的都没有时 ``None``。"""
    traceable = [row for row in rows if row.get("cv") is not None]
    if not traceable:
        return None
    import pandas as pd

    frame = pd.DataFrame([{
        "块序": int(row["block"]),
        "形状": SHAPE_LABELS.get(str(row["shape"]), str(row["shape"])),
        "CV": float(row["cv"]),
    } for row in traceable])
    # 同一 (块序, 形状) 出现两次只可能是目录被拷过/重跑过：留后写的那条（读取器的行序
    # 就是文件顺序），否则 pivot 会抛重复索引。
    frame = frame.drop_duplicates(subset=["块序", "形状"], keep="last")
    wide = frame.pivot(index="块序", columns="形状", values="CV").sort_index()
    wide.index.name = "段序·块序"
    wide.columns.name = None
    return wide


# --------------------------------------------------------------------------- #
# 渲染：段间暂停闸
# --------------------------------------------------------------------------- #
def render_stage_gate(out_dir: Path | str, *, key_prefix: str = "cm",
                      running: bool = False) -> None:
    """段边界的协作式暂停：请求 / 取消请求，两种状态一个按钮两种文案。

    停止点与 ⑥ 的「停止（本段结束后）」**完全相同** —— 都是段边界（本段剩下的块照常
    跑完，段末的 ``_final`` 重测与验收都还在），因为 runner 的 ``should_stop`` 本来就是
    段粒度（``execute_campaign``：单块是原子事务）。两者差的只是**谁读得到这个请求**：
    那个按钮写主线程内存里的 ``stop_event``，页面一关就没了；本闸把请求写进目录，换页面 /
    重开浏览器 / 另一台机器上都还在。所以这里绝不另造一个"块边界停止"，那会是同一件事
    的两个不同停止点。

    ``running=False`` 时按钮禁用并说明原因：这个闸只在有 campaign 在跑的时候有意义，
    让它在空闲时也能点，等于给操作员一个"点了但不会有任何事发生"的按钮。
    """
    import streamlit as st

    st.markdown("**段间暂停**")
    control = _campaign_control()
    if control is None:
        st.button("在本段结束后暂停", key=f"{key_prefix}_stage_pause", disabled=True)
        st.caption("段间暂停依赖 campaign 控制模块（尚未就绪）：运行中请用"
                   "「停止（本段结束后）」，停止点相同。")
        if not running:
            st.caption("当前没有 campaign 在跑：这个闸只在运行中有意义。")
        return

    requested = bool(control.stage_stop_requested(out_dir))
    if not running:
        st.button("在本段结束后暂停", key=f"{key_prefix}_stage_pause", disabled=True)
        if requested:
            st.caption("目录里还留着一个暂停请求：下一次「开始」会在第一段跑完后停下"
                       "（用「从断点续跑」接着跑）。")
        else:
            st.caption("当前没有 campaign 在跑：这个闸只在运行中有意义。")
        return

    if requested:
        if st.button("取消暂停请求", key=f"{key_prefix}_stage_cancel"):
            control.clear_stage_stop(out_dir)
        st.caption("已请求暂停：本段跑完就停。" + _GATE_CAPTION)
    else:
        if st.button("在本段结束后暂停", key=f"{key_prefix}_stage_pause"):
            control.request_stage_stop(out_dir)
        st.caption(_GATE_CAPTION)


def _campaign_control():
    """惰性取段间控制模块；未就绪或缺函数时返回 ``None``（不在这里另写一份状态文件）。"""
    try:
        from ao_shaping.tools.slm.transfer import campaign_control
    except ImportError:
        logger.debug("campaign_control 尚未就绪：段间暂停闸降级为一行说明")
        return None
    missing = [name for name in ("stage_stop_requested", "request_stage_stop",
                                 "clear_stage_stop")
               if not hasattr(campaign_control, name)]
    if missing:
        logger.warning("campaign_control 缺少 {}：段间暂停闸降级为一行说明", missing)
        return None
    return campaign_control


# --------------------------------------------------------------------------- #
# 渲染：历史成本
# --------------------------------------------------------------------------- #
def render_cost_panel(out_dir: Path | str, *, planned_blocks: int | None = None,
                      key_prefix: str = "cm") -> None:
    """历史成本表（段 / 实测秒 / 块数 / 每块秒）+ 一行按计划块数的估算。

    读的是**已落盘的**逐段实测耗时，所以第一次跑必然是空的 —— 那时印
    「首次运行，无历史成本」，而不是印一张全 0 的表让人以为"跑到这儿是免费的"。
    """
    import streamlit as st

    st.markdown("**成本**")
    table = _cost_table()
    if table is None:
        st.caption("成本统计模块尚未就绪：这里不给历史成本，也不猜一个数。")
        return

    payload = table.read_campaign_costs(out_dir)
    rows = _cost_rows(payload)
    if not rows:
        st.caption("首次运行，无历史成本")
        return
    st.dataframe(rows, hide_index=True, use_container_width=True,
                 key=f"{key_prefix}_cost")
    estimate = _estimate_caption(table, payload, planned_blocks)
    if estimate:
        st.caption(estimate)


def _cost_table():
    """惰性取成本表模块；未就绪或缺函数时返回 ``None``。"""
    try:
        from ao_shaping.tools.slm.transfer import cost_table
    except ImportError:
        logger.debug("cost_table 尚未就绪：成本面板降级为一行说明")
        return None
    missing = [name for name in ("read_campaign_costs", "estimate_campaign")
               if not hasattr(cost_table, name)]
    if missing:
        logger.warning("cost_table 缺少 {}：成本面板降级为一行说明", missing)
        return None
    return cost_table


def _cost_rows(payload: Any) -> list[dict[str, Any]]:
    """成本模块的返回 → 四列中文表（数值缺失给 ``None``，不给 ``"—"``）。

    接受两种形状：``{段名: {...}}`` 的映射，或 ``[{"stage": ..., ...}, ...]`` 的序列。
    两种都要认，是因为"按段一份"和"按段一行"在成本模块那边都说得通，而这里不该
    因此多一份实现。
    """
    rows: list[dict[str, Any]] = []
    for name, stats in _cost_entries(payload):
        seconds = _first_number(stats, "seconds", "elapsed_s", "duration_s", "wall_s")
        blocks = _first_number(stats, "blocks", "n_blocks", "block_count")
        per_block = _first_number(stats, "seconds_per_block", "per_block_s", "block_seconds")
        if per_block is None and seconds is not None and blocks:
            per_block = seconds / blocks
        rows.append({
            "段": _stage_text(name),
            "实测秒": seconds,
            "块数": int(blocks) if blocks is not None else None,
            "每块秒": per_block,
        })
    return rows


def _cost_entries(payload: Any) -> list[tuple[str, Mapping[str, Any]]]:
    """``payload`` → ``(段名, 该段的数字)`` 列表（认不出的条目跳过，不抛）。"""
    if isinstance(payload, Mapping):
        return [(str(name), stats) for name, stats in payload.items()
                if isinstance(stats, Mapping)]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        entries: list[tuple[str, Mapping[str, Any]]] = []
        for position, item in enumerate(payload):
            if not isinstance(item, Mapping):
                continue
            name = item.get("stage") or item.get("name") or f"第 {position + 1} 段"
            entries.append((str(name), item))
        return entries
    return []


def _first_number(stats: Mapping[str, Any], *names: str) -> float | None:
    """按候选键名取第一个可用的数字（``0`` 是有效值，不能被 ``or`` 吃掉）。"""
    for name in names:
        number = _number(stats.get(name))
        if number is not None:
            return number
    return None


def _stage_text(name: str) -> str:
    """段名 → 「形状/阶段」的中文标签（形状与阶段都查同一份词表，未知原名照印）。"""
    shape, _, stage = name.partition("/")
    if not stage:
        return SHAPE_LABELS.get(name, name)
    return (f"{SHAPE_LABELS.get(shape, shape)}/"
            f"{STAGE_LABELS.get(stage, stage)}")


def _estimate_caption(table: Any, payload: Any,
                      planned_blocks: int | None) -> str:
    """一行估算：「预计 112 块 ≈ 1:02:00–2:10:00（中位 1:36:00）」。

    传入的是 :func:`read_campaign_costs` 的**原样返回**，不是上面那张中文表：成本模块
    认自己的键（``seconds`` / ``blocks``），把 ``实测秒`` / ``块数`` 喂回去它只会算出全 0，
    于是「有历史」也会印成「预计 ≈ 0」。

    没有计划块数就没有估算 —— 「预计 0 块」比不写更糟。成本模块的返回认不出三条线时
    返回空串（页面少一行，不印半个数）。
    """
    if not planned_blocks:
        return ""
    try:
        raw = _call_estimate(table.estimate_campaign, payload, planned_blocks)
    except Exception as exc:  # 成本模块的口径不该拖垮整页：算不出来就只说一句"没有"
        logger.warning("estimate_campaign 调用失败：{}", exc)
        return ""
    lines = _estimate_numbers(raw)
    if lines is None:
        logger.warning("estimate_campaign 的返回认不出 low/mid/high：{}", raw)
        return ""
    low, mid, high = lines
    return (f"预计 {planned_blocks} 块 ≈ {_clock_text(low)}–{_clock_text(high)}"
            f"（中位 {_clock_text(mid)}）")


def _call_estimate(function: Any, payload: Any, planned_blocks: int) -> Any:
    """按函数**实际收几个位置参数**决定传不传计划块数（并行落地的接口不全同名同序）。"""
    positional = [param for param in inspect.signature(function).parameters.values()
                  if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)]
    if len(positional) >= 2:
        return function(payload, planned_blocks)
    return function(payload)


def _estimate_numbers(raw: Any) -> tuple[float, float, float] | None:
    """三种常见返回形状 → ``(low, mid, high)``：映射（含多种键名）/ 三元组 / 单个中位数。"""
    if isinstance(raw, Mapping):
        low = _first_number(raw, "low", "seconds_low", "min_seconds", "optimistic")
        mid = _first_number(raw, "mid", "seconds_mid", "median", "expected")
        high = _first_number(raw, "high", "seconds_high", "max_seconds", "pessimistic")
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 3:
        low, mid, high = (_number(value) for value in raw[:3])  # type: ignore[misc]
    else:
        low = mid = high = _number(raw)
    if mid is None:
        return None
    return (mid if low is None else low, mid, mid if high is None else high)


def _numeric(value: Any) -> Any:
    """表格里的缺值：``"—"``（:data:`MISSING`）→ ``None``。

    ``st.dataframe`` 走 arrow 序列化，同一列里既有 ``0.6574`` 又有 ``"—"`` 时会退化成
    object 列，arrow 接着按 double 转换并抛 ``ArrowInvalid`` —— 整张表**不显示**，栈只进
    日志。触发条件恰好是「某段还没有块读数」（刚开始 / 被 ``stopped`` / 读到半截产物），
    也就是操作员最需要看进度的那一刻。缺值一律给 ``None``：arrow 用 NaN 表示，界面空白。
    """
    return None if value == MISSING else value


def _render_block_line(stage, planned: int | None = None) -> str:
    """当前段的一行逐块 CV 摘要（与报告 §3.2 同口径：CV 必须带曝光与饱和一起读）。

    ``planned`` 是**该阶段的计划块数**（``campaign_matrix`` 里那一行）。不能拿
    ``len(stage.blocks)`` 当分母：那是「已写下的块」，于是这一行会永远显示
    ``本段块 4/4``，把「跑到第 4 块（共 12 块）」读成「这一阶段做完了」。

    计数器写「本段块」而不是「块」：campaign 的总进度由
    :func:`campaign_monitor.progress_caption` 统一说（``块 28/112``），这一行说的是
    **这一段自己**的块序，两个数混在一句里就会有人拿 4/12 去对 28/112。
    """
    blocks = stage.blocks
    if not blocks:
        return f"{stage.shape}/{stage.stage}：还没有块读数"
    last = blocks[-1]
    first_cv = blocks[0].cv
    total = len(blocks) if planned is None else planned
    return (f"{stage.shape}/{stage.stage} 本段块 {len(blocks)}/{total}"
            f" · 首块 CV {first_cv} → 末块 CV {last.cv}"
            f" · 曝光 {last.exposure_ms} ms · 饱和 {last.saturated_px} px"
            f" · 验收 {last.accepted}")


def journal_plan_shapes(progress) -> tuple[str, ...]:
    """这份目录**计划**跑哪些形状：journal 的 ``plan`` 条目（runner 开跑第一行就写）。

    目录里只留下已经跑过的形状，``collect_campaign_progress`` 也只看得到那些；所以
    「计划分母」必须来自别处 —— 开跑时的 ``plan.shapes`` 就是那份计划。没有 plan 条目
    （更早的目录）时返回空元组，由调用方决定退到哪。
    """
    for entry in reversed(progress.journal):
        plan = entry.get("plan") if isinstance(entry, dict) else None
        if isinstance(plan, dict) and plan.get("shapes"):
            return tuple(str(name) for name in plan["shapes"])
    return ()


def plan_shapes(progress) -> tuple[str, ...]:
    """历史浏览时的计划形状：journal 的 ``plan`` 条目，退到目录里出现过的形状。

    ⑦ 没有 ``settings`` 可问，只能这样；⑥ 用的是页面当前选的形状（见
    :func:`render_progress_panel` 的优先级）。
    """
    return journal_plan_shapes(progress) or tuple(
        dict.fromkeys(stage.shape for stage in progress.stages))


def stage_table_rows(progress) -> list[dict[str, Any]]:
    """段状态表的行。

    抽成函数是为了能单测「缺值不给 ``"—"``」这条：``st.dataframe`` 的 arrow 序列化不接受
    同一列混排 float 与字符串（见 :func:`_numeric`），而这条错误只会打到日志、界面直接少
    一张表。
    """
    return [{"形状": SHAPE_LABELS.get(stage.shape, stage.shape), "阶段": stage.stage,
             "状态": stage.status, "块": len(stage.blocks),
             "段首 CV": _numeric(stage.seed_cv), "段末 CV": _numeric(stage.final_cv),
             "核心峰值": _numeric(stage.peak_inside),
             "曝光 ms": _numeric(stage.exposure_ms)}
            for stage in progress.stages]


def block_table_rows(stage) -> list[dict[str, Any]]:
    """当前段的逐块表：``验收`` 来自 ``summary.json``（读取器已 join），其余照抄读数。"""
    return [{"块": block.index, "CV": _numeric(block.cv),
             "饱和 px": _numeric(block.saturated_px),
             "核心峰值": _numeric(block.core_peak),
             "曝光 ms": _numeric(block.exposure_ms),
             "验收": _numeric(block.accepted), "说明": block.note}
            for block in stage.blocks]


def render_progress_panel(out_dir: Path, *, running: bool,
                          shapes: "tuple[str, ...] | list[str] | None" = None,
                          key_prefix: str = "cm", stage_gate: bool = True) -> None:
    """逐段/逐块进度：进度条 + 段状态表 + 当前段逐块 + CV 轨迹 + 成本 + journal 流水。

    **分母来自计划，不来自读取器。** ``collect_campaign_progress`` 只枚举该目录里
    *已经出现* 的段，所以一份跑到一半的 campaign 会诚实地报「4/4 段」—— 那是本页最
    容易误读的一行（issue #215 的现场就是这个）。计划段数/块数一律由
    ``campaign.campaign_matrix(shapes)`` / ``matrix_totals`` 给出，读取器只提供**分子**。

    分母的优先级：目录 journal 里的 ``plan.shapes``（那是这份目录**实际按什么计划**跑的，
    换过勾选之后也仍然是对的）→ 传入的 ``shapes``（页面当前选择，新目录还没有 plan 时）
    → 目录里出现过的形状。

    这一处同时是 ⑥ 运行中/空闲、⑦ 历史三处视图的唯一渲染体，所以 ``key_prefix`` 由
    调用方给：⑥ 的空闲面板与 ⑦ 可能同时在看同一个目录，不带前缀的同名控件会让 Streamlit
    直接抛 ``DuplicateElementKey``（整页变成一条异常）。``stage_gate`` 在 ⑦ 关掉 ——
    那页是离线读数，没有"本段跑完就停"这回事。
    """
    import streamlit as st
    progress = collect_campaign_progress(out_dir)
    planned = (journal_plan_shapes(progress) or tuple(shapes or ())
               or plan_shapes(progress))
    rows = campaign_matrix(list(planned))
    plan = matrix_totals(rows)
    planned_blocks = {(row.shape, row.stage): row.blocks for row in rows}
    done_blocks = progress.totals["blocks_done"]
    total_blocks = plan["blocks"] or progress.totals["blocks_total"]
    total_segments = plan["segments"] or progress.totals["segments_total"]
    st.progress(min(1.0, done_blocks / total_blocks) if total_blocks else 0.0,
                text=progress_caption(progress, planned_shapes=planned))
    seen = progress.totals["segments_total"]
    if total_segments and seen < total_segments:
        st.caption(f"本目录里目前读到 {seen} 段（计划 {total_segments} 段）：分母取计划，"
                   "所以跑到一半不会显示成「全部完成」。")
    st.dataframe(stage_table_rows(progress), hide_index=True,
                 use_container_width=True)
    current = next((stage for stage in progress.stages
                    if stage.status not in COMPLETE_STATUSES and stage.blocks), None)
    if current is not None:
        planned_for_stage = planned_blocks.get((current.shape, current.stage))
        st.markdown(f"**当前段** · {_render_block_line(current, planned_for_stage)}")
        # ``验收`` 读的是 ``summary.json`` 的接受判定 —— 读取器已经把 manifest 与
        # summary 两个文件按块号连好了，这里不再自己 join 一遍（两份 join 迟早不一致）。
        st.dataframe(block_table_rows(current), hide_index=True,
                     use_container_width=True)
    with st.expander("阶段流水（journal.json）", expanded=False):
        render_progress(collect_progress(out_dir), running=running)
    render_campaign_traces(out_dir, key_prefix=key_prefix)
    if stage_gate:
        render_stage_gate(out_dir, key_prefix=key_prefix, running=running)
    render_cost_panel(out_dir, planned_blocks=total_blocks,
                      key_prefix=key_prefix)
