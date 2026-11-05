"""采集阶段的**几何预检**：不等拟合就把「倾斜相位 + 目标边长」的越界拦下来。

为什么提前
----------
几何守卫 :func:`~ao_shaping.tools.slm.transfer.preflight.check_geometry` 需要两样
东西: **实测零点** 与 **拟合斜坡矩阵**。两者其实都在采集阶段就有了 ——
零点来自 ``reference_start`` 帧, 斜坡来自 ``ramp_{axis}_{period}_{sign}`` 探针对
—— 但守卫原本挂在 ``fit`` 之后 (``workflow.py``), 于是「倾斜周期选得太小 /
目标边长太大」要等**整轮采集 + 整轮拟合**跑完才暴露 (2026-10-26 的
"empty target core" 就是这么浪费了一整轮采集)。

本模块把同一套判据前移到采集阶段: 用**同一批** manifest 行算出斜坡矩阵
(:func:`~ao_shaping.tools.slm.transfer.fit.geometry` 的同一实现), 再调**同一个**
:func:`~ao_shaping.tools.slm.transfer.preflight.check_geometry`。所以提前判与
拟合后判**结论一致** —— 提前判不是「另一套近似」, 是同一把尺子提前量。

与拟合后判的关系
----------------
拟合后的守卫仍然保留 (它用的是**拟合**写出的 ``ramp_matrix``, 且是最后一道
关卡)。提前判是**补充**: 提前失败时, 采集与拟合都还没跑, 轮询立刻拿到带数字的
提示 (该调周期还是该调边长), 而不是先烧掉一轮采集。

真实限制: 倾斜周期不是标定档
----------------------------
斜坡探针只采 ``ramp_periods`` (默认 64 / 128), 而用户的 ``period`` 是自由参数。
斜坡矩阵给出的是**每周期**的位移 (``(spot(+)-spot(-)) * period / 2``), 因此
``ramp_matrix[:, 0] / period`` 是按**比例外推**到用户周期的位移: 周期越小位移
越大 (周期 4 px 时位移可达 1865 px)。这与拟合后判用的是同一个外推, 所以两者
依然一致 —— 但结论本身依赖「周期越小位移越大」这个线性假设, 它在远离标定档时
外推误差变大。因此提前判在**周期远离标定档**时额外给一条 warning, 说明结论是
外推来的。

只读契约
--------
本模块**不打开任何设备**, 只读采集阶段已写下的 manifest 行。它没有副作用, 纯函数。

术语
----
「阈值」= 验收判据 (见 ``docs/terminology_glossary.md`` T-73)。光学孔径的
「口径 (aperture)」不受此约束。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from .preflight import Problem

__all__ = [
    "EarlyGeometryResult",
    "check_early_geometry",
    "format_early_geometry",
]


@dataclass(frozen=True)
class EarlyGeometryResult:
    """采集阶段几何预检的结论。

    Attributes:
        problems: 检查结论列表 (空 = 通过)。error 级阻断, warning 级只提示。
        ramp_matrix: 由采集阶段 ramp 行拟合出的斜坡矩阵 (2×N); 拟合不出来时为
            ``None``。
        displacement_px: 用户周期下的载波位移模长 (px); 无斜坡时为 ``None``。
        zero_xy: 实测零点 (相机 px); 没有有效 ``reference_start`` 时为 ``None``。
        checked: 是否真的做了检查 (``False`` = 条件不足, 未判)。
    """

    problems: tuple["Problem", ...]
    ramp_matrix: np.ndarray | None
    displacement_px: float | None
    zero_xy: tuple[float, float] | None
    checked: bool

    @property
    def ok(self) -> bool:
        """没有 error 级结论 = 可以继续。"""
        return not any(problem.severity == "error" for problem in self.problems)

    @property
    def errors(self) -> tuple["Problem", ...]:
        """阻断级结论。"""
        return tuple(p for p in self.problems if p.severity == "error")


def _usable_rows(rows: Sequence[dict]) -> list[dict]:
    """过滤出可用的行 —— 与 :func:`fit.load_dataset` 用的是同一个 ``valid``。

    ``fit.valid`` 判的是「这一帧曝光稳、峰值没打满」; 斜坡拟合与零点定位都必须
    只看这样的行, 否则提前判会与拟合后判用不同的行集, 结论就对不上了。
    """
    from . import fit

    return [row for row in rows if fit.valid(row)]


def _zero_from_rows(rows: Sequence[dict]) -> tuple[float, float] | None:
    """从 ``reference_start`` 行取实测零点 (与 ``workflow`` 同源)。"""
    from . import fit

    for row in rows:
        if row.get("label") == "reference_start":
            spot = fit.spot(row)
            return float(spot[0]), float(spot[1])
    return None


def check_early_geometry(
    rows: Sequence[dict],
    *,
    side: float,
    period: float,
    shapes: Sequence[str],
    half_span_px: float,
    step_px: float = 2.0,
    tilt: bool = True,
    ramp_periods: Sequence[float] = (64, 128),
) -> EarlyGeometryResult:
    """用采集阶段的行提前判「倾斜 + 边长」是否越界。

    这是 :func:`~ao_shaping.tools.slm.transfer.preflight.check_geometry` 的
    **提前版**: 同一把尺子, 只是尺子的数据来源从「拟合写出的斜坡矩阵」换成
    「同一批采集行现拟合的斜坡矩阵」。

    Args:
        rows: 采集阶段的 manifest 行 (``capture`` 写出的那种 dict)。
        side: 目标方斑边长 (px)。
        period: 倾斜相位周期 (panel px)。
        shapes: 目标形状名。
        half_span_px: 采样窗半跨 (px)。
        step_px: 采样窗步长 (px)。
        tilt: 是否启用倾斜相位。关闭时没有载波位移, 越界判据失去对象, 直接通过。
        ramp_periods: 采集时实际采过的斜坡周期。用于判断用户周期离标定档有多远
            —— 离得远时结论是**外推**来的, 额外给一条 warning。

    Returns:
        :class:`EarlyGeometryResult`。条件不足 (没有有效斜坡行 / 没有有效零点)
        时 ``checked=False`` 且无 error —— 「没条件判」不等于「判为不合格」。
    """
    from .preflight import Problem, check_geometry

    if not tilt:
        # 无载波位移: 越界判据失去对象, 与拟合后判一致地直接通过。
        return EarlyGeometryResult(problems=(), ramp_matrix=None,
                                   displacement_px=None, zero_xy=None, checked=True)

    usable = _usable_rows(rows)
    zero_xy = _zero_from_rows(usable)
    if zero_xy is None:
        logger.warning(
            "early geometry check skipped: no valid reference_start row in the capture"
        )
        return EarlyGeometryResult(problems=(), ramp_matrix=None,
                                   displacement_px=None, zero_xy=None, checked=False)

    # 与 fit.geometry 同一实现 —— 同一批行 ⇒ 同一个矩阵 ⇒ 与拟合后判结论一致。
    from . import fit

    try:
        matrix, _spread = fit.geometry(usable)
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        logger.warning("early geometry check skipped: ramp geometry unavailable ({!r})", exc)
        return EarlyGeometryResult(problems=(), ramp_matrix=None,
                                   displacement_px=None, zero_xy=zero_xy, checked=False)

    offset_vec = np.asarray(matrix, dtype=float)[:, 0] / float(period)
    displacement = float(np.hypot(offset_vec[0], offset_vec[1]))

    problems = list(
        check_geometry(
            zero_xy=zero_xy,
            ramp_matrix=matrix,
            side=float(side),
            period=float(period),
            shapes=tuple(shapes),
            half_span_px=float(half_span_px),
            step_px=float(step_px),
            tilt=True,
        )
    )

    # 标定档外推提醒: 斜坡只采了 ramp_periods, 用户周期不在其中时结论来自线性
    # 外推, 离得越远越不可靠。这只是提醒, 不阻断 (与拟合后判同一把尺子)。
    known = [float(p) for p in ramp_periods]
    if known and not any(abs(float(period) - p) < 1e-9 for p in known):
        nearest = min(known, key=lambda p: abs(float(period) - p))
        problems.append(
            Problem(
                "period_outside_calibration",
                f"倾斜周期 {float(period):g} 不在标定档 {known} 内，"
                f"载波位移 {displacement:.0f} px 是按比例外推来的"
                f"（最近的标定档是 {nearest:g}）。",
                "想拿到实测位移就把倾斜周期设成标定档之一；否则外推误差会随距离变大。",
                severity="warning",
            )
        )

    return EarlyGeometryResult(
        problems=tuple(problems),
        ramp_matrix=np.asarray(matrix, dtype=float),
        displacement_px=displacement,
        zero_xy=zero_xy,
        checked=True,
    )


def format_early_geometry(result: EarlyGeometryResult) -> str:
    """把结论拼成可直接显示的中文文本 (与 :func:`preflight.format_problems` 同风格)。"""
    from .preflight import format_problems

    if not result.checked:
        return ""
    text = format_problems(result.problems)
    if result.displacement_px is not None:
        text = (
            f"实测载波位移 {result.displacement_px:.0f} px"
            f"（倾斜周期下的目标位移）。\n{text}"
        ).strip()
    return text
