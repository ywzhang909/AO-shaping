"""迁移模型闭环整形的启动前参数检查。

为什么需要它
------------
2026-10-26 实测报错 ``empty target core``（``multishape_objective.ShapeObjective``
构造时抛出）看起来像几何问题，实际是**参数问题**：载波周期 ``period`` 被填成
4 panel px，而斜坡标定只在 64 / 128 panel px 上测过。载波位移 ∝ 1/period，
于是位移被放大 16 倍（实测斜率约 7496 px/周期 → 1874 px），目标方斑整体飞出
采样窗（半跨 220 px），窗内只剩背景，``binary_erosion`` 之后核心为空。
报错发生在**第一个 CCD 分块**，而采集与拟合已经跑了几分钟。

两级检查
--------
1. :func:`check_parameters` —— 只看输入参数，**启动前**就能判（GUI 校验区、
   CLI 的 plan）。没有实测几何时也能抓出明显不合理的值：周期低于 SLM 可表示
   下限、探针数不是 6 的倍数、目标边长超出采样窗的绝对上限。
2. :func:`check_geometry` —— 采集与拟合之后、CCD 分块之前，用**实测**零点与
   拟合出的斜坡矩阵复算目标几何（与 :mod:`ao_shaping.tools.slm.transfer.validate`
   同一口径），空核心这类错误在这里被拦下。

两级都要有：只有第一级时，``period`` 与 ``side`` 的**组合**是否放得下要等到
实测斜坡才知道；只有第二级时，操作员已经等了几分钟才知道参数填错。

口径来源（不要另立一套）
------------------------
- 采样窗：``validate.run`` 用 ``recentre_window(zero_xy, half_span_px=220, step_px=2)``；
- 载波位移：``validate.run`` 用 ``ramp_matrix[:, 0] / period``；
- 目标几何：``multishape_objective.target_at`` + ``ShapeObjective``（同一份实现，
  不在这里复刻）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "CALIBRATED_PERIODS_PX",
    "HALF_SPAN_PX",
    "MIN_PERIOD_PX",
    "MIN_SIDE_PX",
    "PROBE_COUNT_MULTIPLE",
    "STEP_PX",
    "WINDOW_MARGIN_PX",
    "Problem",
    "check_geometry",
    "check_parameters",
    "format_problems",
]

#: 采样窗单边跨度（px）。与 ``validate.run`` 的 ``recentre_window(half_span_px=220)`` 一致。
HALF_SPAN_PX: float = 220.0

#: 采样窗步长（px）。与 ``recentre_window(step_px=2)`` 一致。
STEP_PX: float = 2.0

#: 目标方斑与采样窗边界之间必须留的余量（px）。留 0 会在 ``validate.run`` 里
#: 触发「目标与相机窗边界相交」，留 8 px 让「差一点放得下」也能被提前判掉。
WINDOW_MARGIN_PX: float = 8.0

#: 目标方斑边长下限（px）。更小的方斑在 2 px 采样网格上取不到内部核心。
MIN_SIDE_PX: float = 8.0

#: 载波周期下限（panel px）。SLM 像元 8 µm ⇒ 每周期至少 2 个像元才不被采样成
#: 阶梯（Nyquist）。低于此值载波本身就不该下发。
MIN_PERIOD_PX: float = 16.0

#: 斜坡标定实际测过的载波周期（panel px）。不在此集合里的周期，位移是按
#: 「位移 ∝ 1/period」外推出来的，只能算警告。
CALIBRATED_PERIODS_PX: tuple[float, ...] = (64.0, 128.0)

#: 探针数必须是它的倍数：训练/验证划分写死为 ``index % 6 >= 3``。
PROBE_COUNT_MULTIPLE: int = 6


@dataclass(frozen=True)
class Problem:
    """一条检查结论。

    Attributes:
        code: 机器可读的短码（测试与日志用）。
        message: 一句话说清「哪里不对」。
        hint: 「怎么改」—— 必须给出可执行的数字或动作。
        severity: ``error`` 阻断启动；``warning`` 只提示。
    """

    code: str
    message: str
    hint: str
    severity: str = "error"


def format_problems(problems: Sequence[Problem]) -> str:
    """把检查结论拼成一段可直接显示的中文文本。"""
    if not problems:
        return ""
    lines = []
    for problem in problems:
        mark = "✗" if problem.severity == "error" else "!"
        lines.append(f"{mark} {problem.message}\n    → {problem.hint}")
    return "\n".join(lines)


def _max_side_px(offset_px: float, half_span_px: float) -> float:
    """在给定载波位移下，采样窗还能容纳的最大目标边长（px）。"""
    return max(0.0, 2.0 * (half_span_px - WINDOW_MARGIN_PX - offset_px))


def check_parameters(
    *,
    side: float,
    period: float,
    count: int,
    iterations: int,
    blocks: int,
    updates: int,
    frames: int,
    shapes: Sequence[str],
    offset_px: float | None = None,
    half_span_px: float = HALF_SPAN_PX,
) -> list[Problem]:
    """启动前的**纯参数**检查（不需要实测几何）。

    Args:
        side: 目标方斑边长（px）。
        period: 载波周期（panel px）。
        count: 探针数。
        iterations: 拟合迭代数。
        blocks: 锚定分块数。
        updates: 每块更新数。
        frames: 每点帧数。
        shapes: 目标形状名。
        offset_px: 已知的载波位移（px）。给了才检查「位移 + 目标边长是否放得下」；
            没有实测斜坡时传 ``None``（例如首次运行）。
        half_span_px: 采样窗单边跨度（px）。

    Returns:
        检查结论列表（空列表 = 通过）。
    """
    problems: list[Problem] = []

    if not math.isfinite(period) or period <= 0:
        problems.append(
            Problem(
                "period_invalid",
                f"载波周期 {period!r} 不是正数。",
                "填标定过的 64 panel px（SLM 8 µm 像元）。",
            )
        )
    elif period < MIN_PERIOD_PX:
        problems.append(
            Problem(
                "period_below_nyquist",
                f"载波周期 {period:g} panel px 低于可表示下限 {MIN_PERIOD_PX:g} panel px"
                f"（8 µm 像元 ⇒ 每周期至少 2 个像元）。",
                f"改成 {CALIBRATED_PERIODS_PX[0]:g} panel px（标定档），"
                "并同步检查目标边长是否还放得进采样窗。",
            )
        )
    elif period not in CALIBRATED_PERIODS_PX:
        problems.append(
            Problem(
                "period_extrapolated",
                f"载波周期 {period:g} panel px 不在标定过的 "
                f"{'/'.join(f'{p:g}' for p in CALIBRATED_PERIODS_PX)} panel px 上，"
                "载波位移是按「位移 ∝ 1/周期」外推的。",
                "标定只在 64 / 128 panel px 上测过位移；用其它周期时请确认外推仍然成立。",
                severity="warning",
            )
        )

    if count < PROBE_COUNT_MULTIPLE:
        problems.append(
            Problem(
                "count_too_small",
                f"探针数 {count} 少于 {PROBE_COUNT_MULTIPLE}。",
                f"训练/验证划分写死为 index % {PROBE_COUNT_MULTIPLE} >= 3，"
                f"探针数必须是 {PROBE_COUNT_MULTIPLE} 的倍数。",
            )
        )
    elif count % PROBE_COUNT_MULTIPLE:
        problems.append(
            Problem(
                "count_not_multiple",
                f"探针数 {count} 不是 {PROBE_COUNT_MULTIPLE} 的倍数。",
                f"训练/验证划分写死为 index % {PROBE_COUNT_MULTIPLE} >= 3，"
                "非倍数会让其中一个划分只剩零星几条，拟合阶段直接拒绝。",
            )
        )

    if not math.isfinite(side) or side <= 0:
        problems.append(
            Problem(
                "side_invalid",
                f"目标边长 {side!r} 不是正数。",
                f"填 {MIN_SIDE_PX:g}–{_max_side_px(0.0, half_span_px):g} px 之间的值。",
            )
        )
    elif side < MIN_SIDE_PX:
        problems.append(
            Problem(
                "side_too_small",
                f"目标边长 {side:g} px 小于 {MIN_SIDE_PX:g} px。",
                f"更小的方斑在 {STEP_PX:g} px 采样网格上取不到内部核心（空核心）。",
            )
        )
    elif side > _max_side_px(0.0, half_span_px):
        problems.append(
            Problem(
                "side_exceeds_window",
                f"目标边长 {side:g} px 超出采样窗能容纳的 "
                f"{_max_side_px(0.0, half_span_px):g} px"
                f"（半跨 {half_span_px:g} px，两侧各留 {WINDOW_MARGIN_PX:g} px）。",
                f"把目标边长降到 {_max_side_px(0.0, half_span_px):g} px 以内。",
            )
        )

    if offset_px is not None and math.isfinite(offset_px):
        limit = _max_side_px(float(offset_px), half_span_px)
        if side > limit:
            problems.append(
                Problem(
                    "side_offset_overflow",
                    f"载波位移 {offset_px:.0f} px + 目标半宽 {side / 2:.0f} px "
                    f"超出采样窗半跨 {half_span_px:g} px。",
                    f"目标边长降到 {limit:.0f} px 以内，或把载波周期调大"
                    "（位移 ∝ 1/周期）。",
                )
            )

    for name, value in (("拟合迭代", iterations), ("锚定分块数", blocks), ("每块更新数", updates), ("每点帧数", frames)):
        if value < 1:
            problems.append(
                Problem(
                    "count_nonpositive",
                    f"{name} = {value}，必须 ≥ 1。",
                    "填 1 或更大的整数。",
                )
            )

    if not shapes:
        problems.append(
            Problem(
                "no_shapes",
                "没有选任何目标形状。",
                "至少选一个（正方形 / 长方形 / 三角形）。",
            )
        )
    return problems


def check_geometry(
    *,
    zero_xy: Sequence[float],
    ramp_matrix: np.ndarray,
    side: float,
    period: float,
    shapes: Sequence[str],
    half_span_px: float = HALF_SPAN_PX,
    step_px: float = STEP_PX,
) -> list[Problem]:
    """用**实测**零点与拟合斜坡复算目标几何（采集/拟合之后、CCD 分块之前）。

    口径与 :mod:`ao_shaping.tools.slm.transfer.validate` 完全一致：采样窗用
    ``recentre_window``，目标用 ``target_at``，核心用 ``binary_erosion``。

    Args:
        zero_xy: 本次实测零点 ``(x, y)``（相机像元）。
        ramp_matrix: 拟合出的斜坡矩阵（2×2，第一列是 x 轴斜坡的位移向量）。
        side: 目标方斑边长（px）。
        period: 载波周期（panel px）。
        shapes: 目标形状名。
        half_span_px: 采样窗单边跨度（px）。
        step_px: 采样窗步长（px）。

    Returns:
        检查结论列表（空列表 = 通过）。
    """
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        recentre_window,
    )
    from ao_shaping.algorithm.signal_processing.multishape_objective import target_at
    from scipy.ndimage import binary_erosion

    problems: list[Problem] = []
    matrix = np.asarray(ramp_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 2 or matrix.shape[1] < 1:
        return [
            Problem(
                "ramp_matrix_invalid",
                f"斜坡矩阵形状 {matrix.shape} 不合法（应为 2×N：第一列是 x 轴斜坡的位移向量）。",
                "重新拟合（fit 阶段）或换一个有实测斜坡的检查点。",
            )
        ]

    offset_vec = matrix[:, 0] / float(period)
    offset = float(np.hypot(offset_vec[0], offset_vec[1]))
    centre = np.asarray(zero_xy, dtype=float) + offset_vec
    xy = recentre_window(tuple(float(v) for v in zero_xy), half_span_px=half_span_px, step_px=step_px)

    # 零级必须留在目标框外：载波位移就是用来把它推开的。
    if side > 0 and offset < side / 2:
        problems.append(
            Problem(
                "zero_order_inside_target",
                f"载波位移 {offset:.0f} px 小于目标半宽 {side / 2:.0f} px，零级仍落在目标框内。",
                "调大载波周期（位移 ∝ 1/周期）或缩小目标边长。",
                severity="warning",
            )
        )

    for shape in shapes:
        target = target_at(xy, shape, centre, side)
        box = np.asarray(target, dtype=np.float32) > 0.5
        core = binary_erosion(box, iterations=2)
        core_px = int(core.sum())
        if not box.any():
            problems.append(
                Problem(
                    "target_outside_window",
                    f"目标方斑（{shape}）完全落在采样窗之外：载波位移 {offset:.0f} px，"
                    f"采样窗半跨只有 {half_span_px:g} px。",
                    _period_hint(offset_vec, side, half_span_px),
                )
            )
        elif not core.any():
            problems.append(
                Problem(
                    "empty_target_core",
                    f"目标方斑（{shape}）与采样窗只重叠了 {int(box.sum())} px，"
                    "腐蚀后没有内部核心（就是报错里的 empty target core）。",
                    _period_hint(offset_vec, side, half_span_px),
                )
            )
        else:
            limit = _max_side_px(offset, half_span_px)
            if side > limit:
                problems.append(
                    Problem(
                        "side_offset_overflow",
                        f"目标方斑（{shape}）占 {core_px} px 核心，但边长 {side:g} px "
                        f"加上载波位移 {offset:.0f} px 已超出采样窗余量"
                        f"（上限 {limit:.0f} px）。",
                        _period_hint(offset_vec, side, half_span_px),
                        severity="warning",
                    )
                )
    return problems


def _period_hint(offset_vec: np.ndarray, side: float, half_span_px: float) -> str:
    """给出「该把载波周期调到多少」的数值提示（优先落在标定档上）。"""
    offset = float(np.hypot(offset_vec[0], offset_vec[1]))
    room = half_span_px - WINDOW_MARGIN_PX - side / 2
    if offset <= 0 or room <= 0:
        return (
            f"目标边长 {side:g} px 本身已超出采样窗（半跨 {half_span_px:g} px，"
            f"两侧各留 {WINDOW_MARGIN_PX:g} px）；先把边长降到 "
            f"{_max_side_px(0.0, half_span_px):g} px 以内。"
        )
    period_min = max(offset / room, MIN_PERIOD_PX)
    calibrated = [p for p in CALIBRATED_PERIODS_PX if p >= period_min]
    if calibrated:
        return (
            f"把载波周期调到 {calibrated[0]:g} panel px（标定档，"
            f"位移随之降到 {offset / calibrated[0]:.0f} px）；"
            f"或把目标边长降到 ≤ {_max_side_px(offset, half_span_px):.0f} px。"
        )
    return (
        f"载波周期需 ≥ {period_min:.0f} panel px，而标定档只有 "
        f"{'/'.join(f'{p:g}' for p in CALIBRATED_PERIODS_PX)} panel px —— "
        "两者都放不下时，得重新标定斜坡或把目标边长降到 "
        f"≤ {_max_side_px(offset, half_span_px):.0f} px。"
    )
