"""光路无光的早判与专用退出码: 把「没有光」从「曝光不收敛」里拆出来。

为什么需要它
------------
2026-10-28 的 multishape campaign 里, 光路失效那一段是 ``adapt_exposure`` 连跑
**48 s** 打不开曝光窗口、然后抛 ``RuntimeError("exposure did not converge")``
(``report/shape_trials/multishape_scale/multishape_scale_wallclock_20261028.md``
§9 第 1 条)。那条报错**不区分**两件完全不同的事:

* 光路没有光 (遮挡 / 激光没出光 / 光路漂走) —— 再等也不会好, 该立刻停, 人去查光路;
* 曝光参数越界 (会话上限 ``AO_DAHENG_MAX_EXPOSURE_MS`` 太小 / 相机范围不够) ——
  是参数问题, 换个曝光区间就能继续。

两者在旧实现里都表现为 ``status != ok``, 于是 12 档阶梯 (每档 0.6 s settle +
3 次读出 + 每次落盘 npz) 全部跑完才报错, 48 s 换来一句无法诊断的话。

这个模块只补两件事:

1. :class:`LightGuard` —— 挂在 :func:`ao_shaping.tools.slm.panel.focus_probe.adapt_exposure`
   的 ``on_step`` 上, 在阶梯**进行中**判出「这一路没有可用光」;
2. :func:`verify_light` —— 会话/阶段启动前的一次便宜检查 (少数几帧), 在
   探针/拟合/整形这些分钟级阶段之前就把坏光路挡掉。

两者都抛 :class:`OpticalPathInvalid` (``exit_code = 3``), 由 runner 与 CLI 顶层
映射成退出码, **不打印 traceback** —— 这是实验系统上预期会发生、可诊断的状态,
不是代码缺陷。

判据
----
逐档比峰值与**该帧自己的噪声底** ``median + 4·1.4826·MAD`` (canonical 在
:func:`ao_shaping.utils.image.exposure_control.frame_noise_floor`):

1. 任一路峰值越过噪声底 ⇒ 有光, 立刻放行, 阶梯走原来的收敛路径 (一行不改);
2. 连续两档都在噪声底以下, **且峰值涨不到「跟着曝光走一半」应有的涨幅** ⇒
   峰值不跟曝光走 = 没有光, 立刻停;
3. 走到阶梯最后一档仍在噪声底以下 ⇒ 走到最长曝光都没有可用光, 立刻停。

为什么第 2 条用「一半」而不是「完全不涨」: 8 bit 帧在几个 counts 上是量化的,
暗帧的 (去毛刺后) 峰值会在 0-4 counts 之间跳动; 只要曝光涨 4 倍而峰值只涨 1-2
counts, 就已经可以排除「线性响应」了, 而真信号哪怕只有 2 counts, 4 倍曝光后会到
8 counts。用一半的斜率就是给量化留的余量。

**不判「弱光」**: 2026-10-29 实测光路被挡掉一部分时仍有 1.38 counts/ms
(参考 6.15 counts/ms, 22%), 曝光能把它抬进可用带 —— 那是
``scripts/slm/probe_optical_path_validity.py`` 的 ``weak_path`` 态管的, 不在本模块
职责内。本模块只判「抬不起来」。

两种「抬不起来」分开说, 因为处置完全不同: 峰值**完全不随曝光走** ⇒ 遮挡 / 激光
未出光 (:data:`CAUSE_BLOCKED`); 峰值还在随曝光涨、只是整个阶梯都没到噪声底 ⇒
曝光上限不足 (:data:`CAUSE_EXPOSURE_CEILING`, 去查 ``AO_DAHENG_MAX_EXPOSURE_MS``
或相机范围)。
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.tools.slm.panel.focus_probe import ExposureStep, robust_peak
from ao_shaping.utils.image.exposure_control import FrameNoiseFloor, frame_noise_floor

#: 光路无光的专用退出码。与 ``1`` (普通异常) 和 ``2`` (探针判 non-ok) 分开,
#: 免得 lane 脚本把「光路没光」当成「参数没调好」。``3`` 在本仓库已有先例:
#: ``tools/slm/panel/bench_fingerprint.py`` 用它表示「光路已变更」。
EXIT_OPTICAL_PATH_INVALID = 3

#: 峰值不跟曝光走的两条门限: 曝光至少要涨 2 倍才谈得上「涨了没涨」, 而峰值涨幅
#: 不到线性预测的一半就判「不跟曝光走」。
MIN_EXPOSURE_SPAN = 2.0
LINEAR_GROWTH_FRACTION = 0.5

#: 量化下限: 相邻两档峰值的差不到 1 count 就是没涨 (8 bit 帧的最小步进)。
MIN_PEAK_RISE_COUNTS = 1.0

#: 启动前检查用的曝光档 (ms)。乘 8 递增, 覆盖 2f 系统实测的两个极端: 亮态
#: 370 counts/ms (0.05 ms 就有 ~18 counts) 与 2026-10-22 那次低光通量
#: (1 ms 只有 4-5 counts, 需要到 ms 量级才离得开噪声底)。
PRECHECK_EXPOSURES_MS: tuple[float, ...] = (0.05, 0.4, 3.2)

#: ``cause`` 的机器可读取值。
CAUSE_BLOCKED = "blocked"
CAUSE_EXPOSURE_CEILING = "exposure_ceiling"

_CAUSE_TEXT = {
    CAUSE_BLOCKED: "遮挡或激光未出光 —— 峰值不随曝光上升, 加曝光补不回来",
    CAUSE_EXPOSURE_CEILING: (
        "曝光上限不足 —— 峰值还在随曝光上升却始终没到噪声底, 会话上限"
        " AO_DAHENG_MAX_EXPOSURE_MS 或相机范围把阶梯截短了"
    ),
}

_REASON_TEXT = {
    "flat": "连续两档峰值不随曝光上升",
    "exhausted": "走到阶梯最长曝光仍无可测光",
    "precheck": "启动前检查的每一个曝光档都没有光",
}


@dataclass(frozen=True)
class LadderReading:
    """一档曝光 + 它的峰值 + 该帧自己的噪声底。"""

    index: int
    exposure_ms: float
    peak_counts: float
    noise: FrameNoiseFloor

    def as_dict(self) -> dict[str, Any]:
        return {"index": int(self.index), "exposure_ms": float(self.exposure_ms),
                "peak_counts": float(self.peak_counts),
                "noise_median_counts": float(self.noise.median_counts),
                "noise_sigma_counts": float(self.noise.sigma_counts),
                "noise_floor_counts": float(self.noise.threshold_counts)}


class OpticalPathInvalid(RuntimeError):
    """光路在这一路没有可用光: 预期状态, 不是缺陷。

    与普通 ``RuntimeError`` 的三点区别, 都是为了让一次失败**可诊断**:

    * 带实测数字 —— 各档曝光、峰值、该帧噪声底、counts/ms, 而不是一句
      「不收敛」;
    * 带单一最可能原因 (:attr:`cause`), 因为它决定处置方向 (查光路 vs 调曝光);
    * 固定 :attr:`exit_code` = ``3``, 让 lane 脚本 / 运维不必读日志就能分流。

    Attributes:
        reason: ``flat`` (峰值不跟曝光走) / ``exhausted`` (阶梯走完仍无光) /
            ``precheck`` (启动前检查没有光)。
        cause: ``blocked`` / ``exposure_ceiling``, 见 :data:`_CAUSE_TEXT`。
        readings: 每档的 :meth:`LadderReading.as_dict`。
        what: 被检查的对象 (探针标签 / 阶段名), 便于在 journal 里定位。
    """

    exit_code = EXIT_OPTICAL_PATH_INVALID

    def __init__(self, *, reason: str, cause: str, readings: Sequence[LadderReading],
                 budget: int, what: str = "光路") -> None:
        self.reason = str(reason)
        self.cause = str(cause)
        self.readings = tuple(reading.as_dict() for reading in readings)
        self.budget = int(budget)
        self.what = str(what)
        self.exit_code = EXIT_OPTICAL_PATH_INVALID
        super().__init__(self.report())

    @property
    def peak_counts(self) -> float:
        return max((row["peak_counts"] for row in self.readings), default=0.0)

    @property
    def noise_floor_counts(self) -> float:
        return float(self.readings[-1]["noise_floor_counts"]) if self.readings else 0.0

    @property
    def exposures_ms(self) -> tuple[float, ...]:
        return tuple(row["exposure_ms"] for row in self.readings)

    @property
    def counts_per_ms(self) -> float:
        """最后一档的峰值率 —— 与探针的 ``counts_per_ms`` 同一口径, 可直接对比。"""
        if not self.readings or self.readings[-1]["exposure_ms"] <= 0:
            return 0.0
        return float(self.readings[-1]["peak_counts"] / self.readings[-1]["exposure_ms"])

    def as_dict(self) -> dict[str, Any]:
        return {"error": type(self).__name__, "exit_code": self.exit_code,
                "reason": self.reason, "cause": self.cause, "what": self.what,
                "peak_counts": self.peak_counts,
                "noise_floor_counts": self.noise_floor_counts,
                "exposures_ms": list(self.exposures_ms),
                "counts_per_ms": self.counts_per_ms,
                "readings": list(self.readings)}

    def report(self) -> str:
        """一行人能读懂的诊断: 结论 + 实测数字 + 单一最可能原因 + 处置。"""
        stop = int(self.readings[-1]["index"]) if self.readings else 0
        exposures = self.exposures_ms
        span = (f"{exposures[0]:g}→{exposures[-1]:g} ms" if len(exposures) > 1
                else f"{exposures[0]:g} ms" if exposures else "未测到曝光")
        per_rung = "; ".join(f"{row['exposure_ms']:g} ms→{row['peak_counts']:.1f} counts"
                             for row in self.readings)
        lines = [
            f"光路无光 ({self.what}): 第 {stop}/{self.budget} 档中止"
            f" —— {_REASON_TEXT.get(self.reason, self.reason)}",
            f"  实测: {span}, 峰值最大 {self.peak_counts:.1f} counts ≤ 噪声底 "
            f"{self.noise_floor_counts:.1f} counts, {self.counts_per_ms:.3f} counts/ms",
            f"  各档: {per_rung}" if per_rung else "  各档: (无读数)",
            f"  最可能原因: {_CAUSE_TEXT.get(self.cause, self.cause)}",
            "  处置: 先查光阑/挡片/激光出光与光路准直; 光路通后再跑 "
            "scripts/slm/probe_optical_path_validity.py (它还会判 weak_path)",
        ]
        return "\n".join(lines)


def _cause_from(readings: Sequence[LadderReading]) -> str:
    """整段读数里峰值到底涨没涨 —— 决定处置方向 (查光路 vs 调曝光上限)。

    涨了 (超过一个噪声 sigma) 却始终没到噪声底 ⇒ 阶梯/会话上限把阶梯截短了;
    完全没涨 ⇒ 加曝光也补不回来, 是光路的问题。
    """
    if len(readings) < 2:
        return CAUSE_BLOCKED
    first, last = readings[0], readings[-1]
    rose = last.peak_counts - first.peak_counts > first.noise.sigma_counts
    return CAUSE_EXPOSURE_CEILING if rose else CAUSE_BLOCKED


def _not_rising(previous: LadderReading, last: LadderReading, *,
                min_span: float = MIN_EXPOSURE_SPAN) -> bool:
    """相邻两档 (都已在噪声底之下) 的峰值涨得够不够「跟曝光走」。

    ``previous`` 与 ``last`` 必须都是暗读数, 所以峰值只有几个 counts: 判据是
    「涨幅达不到线性预测的一半」, 用一半是因为 8 bit 量化的台阶就有一个 count。
    """
    if previous.exposure_ms <= 0:
        return False
    span = float(last.exposure_ms) / float(previous.exposure_ms)
    if span < float(min_span):
        return False  # 曝光几乎没动, 谈不上「涨没涨」
    predicted_rise = float(previous.peak_counts) * (span - 1.0)
    needed = max(LINEAR_GROWTH_FRACTION * predicted_rise, MIN_PEAK_RISE_COUNTS)
    return float(last.peak_counts) - float(previous.peak_counts) < needed


class LightGuard:
    """在一次曝光阶梯里**边测边判**「这一路没有可用光」。

    用法::

        guard = LightGuard(label=f"probe {label}", max_steps=12)
        setting = adapt_exposure(..., on_step=guard.observe)

    有光时它什么都不做 (阶梯照旧收敛); 没有光时它在第 2-3 档抛
    :class:`OpticalPathInvalid`, 把 48 s 的等待换成几秒。``observe`` 之后仍可读
    :attr:`seen_light` / :attr:`readings` 作为证据。

    Attributes:
        label: 被检查对象的标签, 写进异常里。
        max_steps: 阶梯总档数 (与传给 ``adapt_exposure`` 的 ``max_steps`` 一致)。
        readings: 已读各档。
        seen_light: 是否有一档的峰值越过噪声底。一旦为 ``True`` 就**永不**再抛
            「无光」: 那之后变暗是曝光被调回头, 不是光路没了。
    """

    def __init__(self, *, label: str = "光路", max_steps: int = 6) -> None:
        self.label = str(label)
        self.max_steps = max(int(max_steps), 1)
        self.readings: list[LadderReading] = []
        self.seen_light = False
        self.peak_at_light_counts = 0.0

    # -- 观察 ---------------------------------------------------------------
    def observe(self, step: ExposureStep) -> None:
        """``adapt_exposure`` 的 ``on_step`` 回调: 抛错即中止阶梯。"""
        reading = LadderReading(index=int(step.index), exposure_ms=float(step.exposure_ms),
                                peak_counts=float(step.peak_counts), noise=step.noise)
        self.readings.append(reading)
        if reading.peak_counts > float(step.noise.threshold_counts):
            self.seen_light = True
            self.peak_at_light_counts = max(self.peak_at_light_counts, reading.peak_counts)
            return
        if self.seen_light:
            return
        if step.is_last:
            raise self._verdict("exhausted")
        if len(self.readings) >= 2 and _not_rising(self.readings[-2], self.readings[-1]):
            raise self._verdict("flat")

    # -- 判定 ---------------------------------------------------------------
    def _cause(self) -> str:
        """整段阶梯里峰值到底涨没涨 —— 决定处置方向。"""
        return _cause_from(self.readings)

    def _verdict(self, reason: str) -> OpticalPathInvalid:
        verdict = OpticalPathInvalid(reason=reason, cause=self._cause(),
                                     readings=tuple(self.readings),
                                     budget=self.max_steps, what=self.label)
        logger.warning(verdict.report())
        return verdict


def _rungs(floor_ms: float, ceiling_ms: float,
           exposures_ms: Sequence[float]) -> list[float]:
    """把标称档位夹进相机的实际范围, 去掉重复与倒序。"""
    low, high = float(floor_ms), float(ceiling_ms)
    if high < low:
        high = low
    out: list[float] = []
    for value in exposures_ms:
        clamped = min(max(float(value), low), high)
        if not out or clamped > out[-1]:
            out.append(clamped)
    return out


def _applied_ms(returned: Any, requested: float) -> float:
    """接受「设回读」两种契约: 设为实际值就用它, 返回 ``None`` 就退回请求值。

    硬件驱动 (``DahengCamera.reset_exposure_time``) 会返回夹过的实际值, 但
    仿真/图像序列后端允许直接返回 ``None`` —— 那些后端没有可夹的量程, 请求值
    就是生效值。把 ``None`` 当成错误会让光路检查在仿真路径上直接崩掉。
    """
    return float(returned) if isinstance(returned, (int, float)) else float(requested)


def verify_light(read_frame: Callable[[], np.ndarray],
                 set_exposure_ms: Callable[[float], float], *,
                 floor_ms: float, ceiling_ms: float,
                 frames: int = 2,
                 label: str = "启动前光路检查",
                 exposures_ms: Sequence[float] = PRECHECK_EXPOSURES_MS) -> dict[str, Any]:
    """少数几帧回答「这一路现在到底有没有光」。

    为什么不是整个探针: ``scripts/slm/probe_optical_path_validity.py`` 还要判
    离焦响应与 ``weak_path``, 那是分钟级的、要在**几何标定之前**跑的权威判定;
    这里只是让一次坏光路在进入分钟级阶段之前就停下, 所以只测「有没有光」, 曝光
    档取 0.05 / 0.4 / 3.2 ms (相差 8 倍, 覆盖亮态与 2026-10-22 那种低光通量),
    第一档就看得见光时只读一两帧。

    档位是固定的, 不是从「会话曝光」来的: 会话曝光本身要靠阶梯才发现, 而检查的
    目的正是避免在没有光的时候去跑那个阶梯。

    Args:
        read_frame: 取一帧 (与生产路径同一个读出)。
        set_exposure_ms: 设曝光并返回实际生效值 (相机会夹到自己的范围)。
        floor_ms / ceiling_ms: 本次会话的曝光范围 (上限已含
            ``AO_DAHENG_MAX_EXPOSURE_MS`` 的人为上限)。
        frames: 每档平均几帧 (默认 2, 单帧读出噪声的峰值不足以冒充光)。
        label: 写进异常的对象名。
        exposures_ms: 标称档位。

    Returns:
        ``{"ok": True, "exposure_ms", "peak_counts", "noise_floor_counts",
        "counts_per_ms", "readings"}``。

    Raises:
        OpticalPathInvalid: 每一档都只有噪声底。
    """
    rungs = _rungs(floor_ms, ceiling_ms, exposures_ms)
    readings: list[LadderReading] = []
    for index, exposure in enumerate(rungs, start=1):
        applied = _applied_ms(set_exposure_ms(exposure), exposure)
        frames_read = [np.asarray(read_frame(), dtype=np.float64)
                       for _ in range(max(int(frames), 1))]
        frame = frames_read[0] if len(frames_read) == 1 else np.mean(frames_read, axis=0)
        reading = LadderReading(index=index, exposure_ms=applied,
                                peak_counts=robust_peak(frame),
                                noise=frame_noise_floor(frame))
        readings.append(reading)
        if reading.peak_counts > reading.noise.threshold_counts:
            verdict = dict(ok=True, exposure_ms=reading.exposure_ms,
                           peak_counts=reading.peak_counts,
                           noise_floor_counts=reading.noise.threshold_counts,
                           counts_per_ms=(reading.peak_counts / reading.exposure_ms
                                          if reading.exposure_ms > 0 else 0.0),
                           readings=[row.as_dict() for row in readings])
            logger.info("光路有光 ({}): {:.1f} ms 峰值 {:.1f} counts > 噪声底 {:.1f} counts",
                        label, reading.exposure_ms, reading.peak_counts,
                        reading.noise.threshold_counts)
            return verdict
    verdict = OpticalPathInvalid(reason="precheck", cause=_cause_from(readings),
                                 readings=tuple(readings), budget=len(rungs), what=label)
    logger.warning(verdict.report())
    raise verdict
