"""迁移学习闭环整形的启动前参数检查。

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

第三级（只在有实测零点时可用）：:func:`check_zero_overlap` —— 零级剔除圆盘与
目标框的重叠面积。它是**读数口径**检查，不是"能不能跑"：200 px 档实测重叠
5.61 % 的那次仍然跑完了，但那段 CV 只能带重叠面积一起读（规划报告 §1 H3 勘误）。

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
    "ZeroOverlap",
    "check_geometry",
    "check_parameters",
    "check_zero_overlap",
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
                f"探针数 {count} 少于 {PROBE_COUNT_MULTIPLE}（count >= {PROBE_COUNT_MULTIPLE}）。",
                f"训练/验证划分写死为 index % {PROBE_COUNT_MULTIPLE} >= 3，"
                f"探针数必须是 {PROBE_COUNT_MULTIPLE} 的倍数。",
            )
        )
    elif count % PROBE_COUNT_MULTIPLE:
        problems.append(
            Problem(
                "count_not_multiple",
                f"探针数 {count} 不是 {PROBE_COUNT_MULTIPLE} 的倍数"
                f"（count 必须是 {PROBE_COUNT_MULTIPLE} 的倍数）。",
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


#: 零级剔除圆盘的默认半径（px）。与 ``--zero-radius`` 的 CLI 默认 60.0 一致 ——
#: 报告 §3.2 的 0.79 % / 3.14 % / 5.61 % 三个剔除面积都按 60 px 圆盘算的，
#: 换半径就换了口径。
DEFAULT_ZERO_RADIUS_PX: float = 60.0


@dataclass(frozen=True)
class ZeroOverlap:
    """零级剔除圆盘与目标方框的重叠（读数口径，不是"能不能跑"）。

    Attributes:
        intersects: 是否重叠（重叠面积为 0 时为 ``False``）。
        area_px: 重叠面积（px²）。
        pct: 重叠面积占目标方框面积的百分比（``100 * area_px / side²``）。
        depth_px: 圆盘压进方框的深度（px）。圆心在框内时为 0；圆心在框外时为
            ``半径 - 圆心到框的距离``（夹到 0 以上）。
        note: 一句中文说明，带数字，可直接显示在 GUI 的校验区。
    """

    intersects: bool
    area_px: float
    pct: float
    depth_px: float
    note: str


def check_zero_overlap(
    target_x: float,
    target_y: float,
    side: float,
    zero_xy: tuple[float, float] | None,
    zero_radius: float = DEFAULT_ZERO_RADIUS_PX,
) -> ZeroOverlap:
    """零级剔除圆盘 ∩ 目标方框的面积与该区域压进方框的深度。

    为什么需要它：几何守卫（:func:`check_geometry`）只检查目标框是否越出采样窗，
    **不检查零级剔除区与目标框是否重叠**，所以 2026-10-28 那次 200 px 格
    「零级与目标分离」的前提在实测载波位移下失效（规划报告 §1 H3 勘误）却不会报错。
    重叠面积必须与 CV 一起读 —— 否则「200 px 的 CV 更差」会被读成优化变差，
    而不是核心里有 5.61 % 属于零级。

    几何口径：目标框是以 ``(target_x, target_y)`` 为中心、边长 ``side`` 的轴对齐正方形；
    零级是圆心 ``zero_xy``、半径 ``zero_radius`` 的圆盘。两个输入都用**同一坐标系**
    （相机 sensor px）—— 全幅坐标与窗口局部坐标混用会把重叠面积算成 0（见 AGENTS
    反模式表里 `full-frame centre leaking into window-local metrics` 那一条）。

    面积是**解析**算的，不是网格近似：本模块此前只有方框/采样窗的检查
    （:func:`_max_side_px`、:func:`check_geometry`），没有圆弧面积助手，所以
    :func:`_quarter_disc_area` 与 :func:`_disc_box_area` 就实现在这里，不引第二套几何。
    验收：1200 → 0.79 %、600 → 3.14 %、200 → 5.61 %（圆心离框边 29.70 px、
    穿入 30.30 px），见 ``tests/ao_shaping/tools/slm/transfer/test_zero_overlap.py``。

    Args:
        target_x: 目标框中心 x（sensor px）。
        target_y: 目标框中心 y（sensor px）。
        side: 目标框边长（px），须为正。
        zero_xy: 实测零级位置 ``(x, y)``（sensor px）；``None`` 表示未测。
        zero_radius: 零级剔除圆盘半径（px）。

    Returns:
        :class:`ZeroOverlap`。``zero_xy is None`` 时返回「不相交 + 说明未测」，
        不抛异常 —— 计划页在没做光路探针时必须仍然能画出来。

    Raises:
        ValueError: ``side`` 或 ``zero_radius`` 不是正的有限数。
    """
    side_value = _positive_length(side, "目标框边长 side")
    radius = _positive_length(zero_radius, "零级圆盘半径 zero_radius")
    if zero_xy is None:
        return ZeroOverlap(
            intersects=False, area_px=0.0, pct=0.0, depth_px=0.0,
            note="零级位置未测：无法判断剔除区与目标框是否重叠，运行前请做光路探针"
                 "（scripts/slm/probe_optical_path_validity.py）",
        )

    zx, zy = (float(v) for v in zero_xy)
    tx, ty = float(target_x), float(target_y)
    x0, x1 = tx - side_value / 2.0, tx + side_value / 2.0
    y0, y1 = ty - side_value / 2.0, ty + side_value / 2.0

    area = _disc_box_area(zx, zy, radius, x0, x1, y0, y1)
    # 浮点残差：完全相切/完全不相交时面积可能是 1e-12 级的负零或正零。
    area = max(0.0, min(area, math.pi * radius * radius, side_value * side_value))
    pct = 100.0 * area / (side_value * side_value)

    distance = _distance_to_box(zx, zy, x0, x1, y0, y1)
    nearest_edge = _nearest_edge(zx, zy, x0, x1, y0, y1)
    depth = 0.0 if distance <= 0.0 else max(0.0, radius - distance)
    intersects = area > 0.0

    if not intersects:
        note = (f"零级圆盘（半径 {radius:g} px）与目标框不相交："
                f"圆心离框 {distance:.2f} px ≥ 半径 {radius:g} px，剔除面积为 0")
    elif distance <= 0.0 and radius <= nearest_edge:
        note = (f"零级圆盘完全落在目标框内：剔除面积 {pct:.2f}%"
                f"（π·{radius:g}² / {side_value:g}²）")
    elif distance <= 0.0:
        note = (f"零级圆盘被目标框裁掉一部分：剔除面积 {pct:.2f}%"
                f"（半径 {radius:g} px 大于圆心到最近框边 {nearest_edge:.2f} px —— "
                "方框比圆盘小，核心指标会低估零级能量）")
    else:
        note = (f"零级圆盘压进目标框 {depth:.2f} px、占目标面积 {pct:.2f}%："
                f"这部分能量落在核心内，CV 必须与该剔除面积一起读"
                f"（圆心在框外 {distance:.2f} px，半径 {radius:g} px）")

    return ZeroOverlap(intersects=intersects, area_px=area, pct=pct, depth_px=depth, note=note)


def _positive_length(value: float, name: str) -> float:
    """把一个长度参数校验成正的有限数（px）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是数字（px），收到 {value!r}") from None
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} 必须是正的有限数（px），收到 {value!r}")
    return number


def _arc_primitive(t: float, radius: float) -> float:
    """``I(t) = ∫_0^t sqrt(r² − x²) dx``（``t`` 夹到 ``[0, r]``）。

    圆盘面积的**唯一**解析原语：四分之一圆面积就是 ``I(r) = πr²/4``，
    圆弧段面积是 ``I`` 的差。写成原语而不是逐处写扇形公式，是因为夹角、
    弦长、区域三种写法在边界（``t = r``、``t = 0``）上各自有特例，容易分叉。
    """
    t = min(max(t, 0.0), radius)
    rest = max(radius * radius - t * t, 0.0)
    return 0.5 * (t * math.sqrt(rest) + radius * radius * math.asin(t / radius))


def _quarter_disc_area(a: float, b: float, radius: float) -> float:
    """``圆盘 ∩ [0,a]×[0,b]`` 的面积（圆心在原点，``a``/``b`` 为到圆心的距离）。

    这是"一个象限里的累积分布"：对 ``[a0,a1]×[b0,b1]`` 用四项容斥即可得到
    任意轴对齐矩形的重叠面积（见 :func:`_disc_box_area`），不必给圆-矩形裁剪写
    第二种分情形实现。

    分支来自 ``min(b, sqrt(r² − x²))`` 的换元点 ``x = sqrt(r² − b²)``：
    ``b ≥ r`` 时被积函数就是 ``sqrt(r² − x²)``（整条弦都在窗口内），
    否则先是一段常数 ``b``、之后才是弦。
    """
    if a <= 0.0 or b <= 0.0 or radius <= 0.0:
        return 0.0
    limit = min(a, radius)
    if b >= radius:
        return _arc_primitive(limit, radius)
    switch = math.sqrt(max(radius * radius - b * b, 0.0))
    edge = min(limit, switch)
    return b * edge + _arc_primitive(limit, radius) - _arc_primitive(edge, radius)


def _split_at(value0: float, value1: float, cut: float) -> tuple[tuple[float, float], ...]:
    """把区间 ``[value0, value1]`` 在 ``cut`` 处切开（若 ``cut`` 落在区间内部）。"""
    low, high = (value0, value1) if value0 <= value1 else (value1, value0)
    if high <= cut or low >= cut:
        return ((low, high),)
    return ((low, cut), (cut, high))


def _disc_box_area(
    centre_x: float, centre_y: float, radius: float,
    x0: float, x1: float, y0: float, y1: float,
) -> float:
    """圆盘 ∩ 轴对齐矩形的面积（解析，非网格近似）。

    先把矩形沿圆心的两条轴线切成至多 4 块，使每块完全落在一个象限内；每块把坐标
    折叠成到圆心的距离后，用 :func:`_quarter_disc_area` 的四项容斥求面积。
    这样既不需要"圆是否完全在矩形内"之类的特例，也不会在矩形跨越圆心时算错。
    """
    area = 0.0
    for xa, xb in _split_at(x0, x1, centre_x):
        ax0, ax1 = sorted((abs(xa - centre_x), abs(xb - centre_x)))
        for ya, yb in _split_at(y0, y1, centre_y):
            ay0, ay1 = sorted((abs(ya - centre_y), abs(yb - centre_y)))
            area += (
                _quarter_disc_area(ax1, ay1, radius)
                - _quarter_disc_area(ax0, ay1, radius)
                - _quarter_disc_area(ax1, ay0, radius)
                + _quarter_disc_area(ax0, ay0, radius)
            )
    return area


def _distance_to_box(x: float, y: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """点到轴对齐矩形的距离（点在矩形内时为 0）。"""
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)


def _nearest_edge(x: float, y: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """点在矩形内时到最近边的距离（点在矩形外时按 :func:`_distance_to_box` 口径返回 0）。"""
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return 0.0
    return min(x - x0, x1 - x, y - y0, y1 - y)
