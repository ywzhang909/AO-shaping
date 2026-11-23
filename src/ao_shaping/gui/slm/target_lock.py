"""固定目标锁定监督器：无硬件的状态机核心。

与 ``dynamic_shaping.py``（开环时间轴播放）和 ``closed_loop_shaping.py``（迁移
学习闭环）隔离：本模块只做**一次运行、一个固定目标**下的监督判断，不生成相位、
不调用 SLM/CCD，也不读写任何 UI 状态。

为什么是「固定目标 + 绝对坐标」
--------------------------------
旧的「起点 → 目标」动画语义把每一帧都当成过渡中的一步，终点由时间轴保证。
本模块服务的是收敛/维持语义：目标图在 ``arm()`` 时固定，所有评分都在**同一张
CCD 像素网格**上直接比较，既不裁剪、不缩放，也不把实测光斑重新居中。目标质心
同样在 ``arm()`` 时从目标图算一次，之后不再变化；因此「形状对但整体偏了」会被
绝对位置项罚掉，而不是被重新居中掩盖。

一轮的硬性配对
--------------
集成方每写一次 SLM，就必须调用 :meth:`TargetLockSupervisor.expect_write` 登记
**单调递增**的写序号与写入时刻；随后送来的 :class:`MeasuredFrame` 必须同时满足：

* ``write_seq`` 等于最近登记的写序号；
* ``captured_at_s`` 严格晚于登记时刻；
* ``capture_seq`` 单调递增（重复帧不算新测量）。

任一不满足都只算「门禁未通过」，不进入评分，也不会被适配数据吸收。

决策是事件，不是命令
--------------------
监督器通过注入的 :class:`LockCallbacks` 发出至多**每帧一个**决策：

* ``PROPOSE``：候选在噪声内，建议维持当前最佳写序号；
* ``ACCEPT``：候选把评分改善到超过 ``improvement_epsilon``，更新最佳；
* ``ROLLBACK``：候选比最佳差到超过 ``rollback_epsilon``，建议回滚到最佳；
* ``FAULT``：连续无效帧、长期无改善或来回振荡，状态转为 ``FROZEN``。

进入 ``FROZEN`` 时只发一次 ``FAULT``；之后 ``observe`` 返回 ``None``，直到调用方
显式 ``reset()``。核心不替集成方执行回滚/写相位，所有动作都由回调之外的一侧
完成，因此这里不存在自主 SLM 调用。
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "CalibrationSample",
    "FixedTarget",
    "LockAction",
    "LockCallbacks",
    "LockConfig",
    "LockDecision",
    "LockState",
    "MeasuredFrame",
    "TargetLockSupervisor",
    "calibrate_lock_thresholds",
    "calibration_stats_from_frames",
    "target_on_frame",
]


class LockState(str, Enum):
    """监督器状态；只由 :class:`TargetLockSupervisor` 自己写。"""

    IDLE = "idle"
    CONVERGING = "converging"
    MAINTAINING = "maintaining"
    FROZEN = "frozen"


class LockAction(str, Enum):
    """一帧最多产生一个决策事件。"""

    PROPOSE = "propose"
    ACCEPT = "accept"
    ROLLBACK = "rollback"
    FAULT = "fault"


@dataclass(frozen=True)
class LockConfig:
    """门禁与冻结阈值。默认值面向 8/12-bit 相机 + 单一固定目标。"""

    # 帧质量门禁
    saturation_counts: float = 250.0
    max_saturated_fraction: float = 0.01
    min_peak_counts: float = 32.0
    min_total_counts: float = 1.0
    max_centroid_error_px: float = 40.0
    # 决策余量（评分越低越好）
    improvement_epsilon: float = 0.01
    rollback_epsilon: float = 0.03
    # 收敛/冻结
    converge_patience: int = 3
    max_invalid_frames: int = 3
    max_non_improving: int = 6
    oscillation_window: int = 6
    oscillation_amplitude: float = 0.05
    oscillation_min_flips: int = 3
    # 只用于诊断的评分历史长度
    history_window: int = 32

    def __post_init__(self) -> None:
        if self.max_saturated_fraction < 0.0:
            raise ValueError("max_saturated_fraction 不能为负")
        if self.min_peak_counts <= 0.0 or self.min_total_counts <= 0.0:
            raise ValueError("min_peak_counts / min_total_counts 必须为正")
        if self.max_centroid_error_px < 0.0:
            raise ValueError("max_centroid_error_px 不能为负")
        if self.improvement_epsilon < 0.0 or self.rollback_epsilon < 0.0:
            raise ValueError("改进/回滚余量不能为负")
        if self.converge_patience < 1:
            raise ValueError("converge_patience 至少为 1")
        if self.max_invalid_frames < 1 or self.max_non_improving < 1:
            raise ValueError("冻结阈值至少为 1")
        if self.oscillation_window < 3:
            raise ValueError("oscillation_window 至少为 3")
        if self.oscillation_amplitude < 0.0:
            raise ValueError("oscillation_amplitude 不能为负")
        if self.oscillation_min_flips < 1:
            raise ValueError("oscillation_min_flips 至少为 1")
        if self.history_window < 1:
            raise ValueError("history_window 至少为 1")


def _as_target_image(value: np.ndarray, name: str) -> np.ndarray:
    image = np.array(value, dtype=np.float64, copy=True)
    if image.ndim != 2 or image.size == 0:
        raise ValueError(f"{name} 必须是二维非空图像，实际 {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{name} 含非有限值")
    if float(image.min()) < 0.0:
        raise ValueError(f"{name} 含负强度")
    if float(image.mean()) <= 0.0:
        raise ValueError(f"{name} 的平均强度必须为正")
    image.setflags(write=False)
    return image


@dataclass(frozen=True)
class FixedTarget:
    """一次运行固定不变的目标强度图，坐标就是 CCD 像素坐标。

    目标图与后续每一帧必须形状完全一致；监督器不会对它做居中、裁剪或缩放。
    """

    image: np.ndarray
    name: str = "target"

    def __post_init__(self) -> None:
        object.__setattr__(self, "image", _as_target_image(self.image, "target"))


@dataclass(frozen=True)
class MeasuredFrame:
    """一次 CCD 测量及其来源配对信息。

    Args:
        image: 完整 CCD 帧（二维）。门禁失败也应作为对象送入，由监督器给出原因。
        write_seq: 产生这一帧时 SLM 已写入的序号；必须与 ``expect_write`` 登记值
            完全相等。
        capture_seq: 采集侧单调递增序号；用于识别重复/回放帧。
        captured_at_s: 采集时刻（秒）。必须严格晚于对应写入时刻。
        exposure_ms: 采集曝光，仅随决策记录，不改变评分。
    """

    image: np.ndarray
    write_seq: int
    capture_seq: int
    captured_at_s: float
    exposure_ms: float = 0.0


@dataclass(frozen=True)
class LockDecision:
    """监督器对一个新鲜、已配对帧的判断。"""

    action: LockAction
    state: LockState
    reason: str
    write_seq: int | None
    capture_seq: int | None
    score: float | None
    best_score: float | None
    best_write_seq: int | None
    centroid_xy: tuple[float, float] | None
    centroid_error_px: float | None
    metrics: Mapping[str, float] = field(default_factory=dict)
    gate_reasons: tuple[str, ...] = ()

    def describe(self) -> str:
        """一句中文摘要，便于日志与 UI 复用；不做任何格式化假设。"""
        if self.action is LockAction.FAULT:
            return f"冻结：{self.reason}（{self.state.value}）"
        score = "—" if self.score is None else f"{self.score:.4f}"
        best = "—" if self.best_score is None else f"{self.best_score:.4f}"
        return f"{self.action.value}：{self.reason}，score={score}，best={best}"


@dataclass(frozen=True)
class LockCallbacks:
    """注入式决策出口。所有字段可省略；监督器不持有任何设备句柄。"""

    on_propose: Callable[[LockDecision], None] | None = None
    on_accept: Callable[[LockDecision], None] | None = None
    on_rollback: Callable[[LockDecision], None] | None = None
    on_fault: Callable[[LockDecision], None] | None = None

    def emit(self, decision: LockDecision) -> None:
        callback = {
            LockAction.PROPOSE: self.on_propose,
            LockAction.ACCEPT: self.on_accept,
            LockAction.ROLLBACK: self.on_rollback,
            LockAction.FAULT: self.on_fault,
        }[decision.action]
        if callback is not None:
            callback(decision)


def _centroid(image: np.ndarray) -> tuple[float, float] | None:
    """绝对坐标下的强度加权质心；空帧返回 ``None``。"""
    total = float(image.sum())
    if not math.isfinite(total) or total <= 0.0:
        return None
    ys, xs = np.indices(image.shape, dtype=np.float64)
    return float((image * xs).sum() / total), float((image * ys).sum() / total)


def _shape_metrics(measured: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """同一固定网格上的形状比较；均值归一化后才可比。

    评分 ``score = (1 - correlation) + relative_l2``，越低越好。与
    ``closed_loop_shaping.compare_to_target`` 的公式一致，但这里不依赖 transfer
    拟合栈，保持核心只需 numpy。
    """
    measured_n = measured / float(measured.mean())
    target_n = target / float(target.mean())
    mv = (measured_n - measured_n.mean()).ravel()
    tv = (target_n - target_n.mean()).ravel()
    denom = float(np.linalg.norm(mv) * np.linalg.norm(tv))
    correlation = float(np.dot(mv, tv) / denom) if denom > 0.0 else 0.0
    relative_l2 = float(
        np.linalg.norm(measured_n - target_n) / max(np.linalg.norm(target_n), 1e-12)
    )
    return {
        "correlation": correlation,
        "relative_l2": relative_l2,
        "score": (1.0 - correlation) + relative_l2,
    }


class TargetLockSupervisor:
    """固定目标锁定的监督状态机（无硬件、无自主动作）。

    典型用法::

        supervisor = TargetLockSupervisor(LockConfig(), LockCallbacks(...))
        supervisor.arm(FixedTarget(target_image))
        supervisor.expect_write(1, written_at_s=t0)
        decision = supervisor.observe(frame)   # 至多一个决策

    ``observe`` 对无效帧返回 ``None``；具体原因保存在
    :attr:`last_gate_reasons`，并在触发冻结的那次 ``FAULT`` 决策里原样带出。
    """

    def __init__(
        self,
        config: LockConfig | None = None,
        callbacks: LockCallbacks | None = None,
    ) -> None:
        self._config = config or LockConfig()
        self._callbacks = callbacks or LockCallbacks()
        self._state = LockState.IDLE
        self._target: FixedTarget | None = None
        self._target_centroid: tuple[float, float] | None = None
        self._expected_write_seq: int | None = None
        self._expected_written_at_s: float | None = None
        self._last_capture_seq: int | None = None
        self._best_score: float | None = None
        self._best_write_seq: int | None = None
        self._invalid_streak = 0
        self._stalled_streak = 0
        self._scores: deque[float] = deque(maxlen=self._config.history_window)
        self._last_gate_reasons: tuple[str, ...] = ()
        self._fault_count = 0

    # ------------------------------------------------------------------ 状态
    @property
    def config(self) -> LockConfig:
        return self._config

    @property
    def state(self) -> LockState:
        return self._state

    @property
    def target(self) -> FixedTarget | None:
        return self._target

    @property
    def target_centroid(self) -> tuple[float, float] | None:
        """目标质心在 ``arm()`` 时固定，之后不随实测帧变化。"""
        return self._target_centroid

    @property
    def best_score(self) -> float | None:
        return self._best_score

    @property
    def best_write_seq(self) -> int | None:
        return self._best_write_seq

    @property
    def pending_write_seq(self) -> int | None:
        return self._expected_write_seq

    @property
    def last_gate_reasons(self) -> tuple[str, ...]:
        return self._last_gate_reasons

    @property
    def fault_count(self) -> int:
        return self._fault_count

    @property
    def score_history(self) -> tuple[float, ...]:
        return tuple(self._scores)

    # ------------------------------------------------------------------ 生命周期
    def arm(self, target: FixedTarget) -> None:
        """固定目标并进入 ``CONVERGING``；同时清空上一轮的全部历史。"""
        centroid = _centroid(target.image)
        if centroid is None:
            raise ValueError("目标图没有可计算的质心")
        self._reset_round()
        self._target = target
        self._target_centroid = centroid
        self._state = LockState.CONVERGING

    def reset(self) -> None:
        """回到 ``IDLE``，丢弃目标、最佳值与全部计数。"""
        self._reset_round()
        self._target = None
        self._target_centroid = None
        self._state = LockState.IDLE

    def _reset_round(self) -> None:
        self._expected_write_seq = None
        self._expected_written_at_s = None
        self._last_capture_seq = None
        self._best_score = None
        self._best_write_seq = None
        self._invalid_streak = 0
        self._stalled_streak = 0
        self._scores.clear()
        self._last_gate_reasons = ()

    def expect_write(self, write_seq: int, *, written_at_s: float) -> None:
        """登记一次 SLM 写入的序号与时刻。

        序号必须严格单调递增：同一序号的重复登记会被视为集成错误，因为无法
        区分「重新写同一帧」和「旧帧回放」。冻结状态下拒绝登记。
        """
        if self._state is LockState.IDLE:
            raise RuntimeError("先 arm() 目标再登记写入")
        if self._state is LockState.FROZEN:
            raise RuntimeError("监督器已冻结，reset() 后才能登记新写入")
        if not isinstance(write_seq, int) or write_seq < 0:
            raise ValueError("write_seq 必须是非负整数")
        if self._expected_write_seq is not None and write_seq <= self._expected_write_seq:
            raise ValueError(
                f"write_seq 必须严格递增：收到 {write_seq}，已登记 {self._expected_write_seq}"
            )
        if not math.isfinite(written_at_s):
            raise ValueError("written_at_s 必须是有限值")
        self._expected_write_seq = write_seq
        self._expected_written_at_s = float(written_at_s)

    # ------------------------------------------------------------------ 观测
    def observe(self, frame: MeasuredFrame) -> LockDecision | None:
        """校验并评分一帧；至多发出一个决策，无效帧返回 ``None``。"""
        if self._state is LockState.IDLE or self._target is None:
            raise RuntimeError("先 arm() 目标再 observe()")
        if self._state is LockState.FROZEN:
            return None

        image = np.asarray(frame.image, dtype=np.float64)
        gate = self._check_pairing(frame)
        gate.extend(self._check_image(image))

        if gate:
            self._last_gate_reasons = tuple(gate)
            self._invalid_streak += 1
            if self._invalid_streak >= self._config.max_invalid_frames:
                return self._freeze(
                    "repeated_invalid_frames",
                    frame=frame,
                    score=None,
                    metrics={},
                    gate_reasons=tuple(gate),
                )
            return None

        self._invalid_streak = 0
        self._last_gate_reasons = ()
        target = self._target.image
        centroid = _centroid(image)
        assert centroid is not None  # 已通过低信号门禁
        target_centroid = self._target_centroid
        assert target_centroid is not None
        centroid_error = math.hypot(
            centroid[0] - target_centroid[0],
            centroid[1] - target_centroid[1],
        )

        metrics = _shape_metrics(image, target)
        metrics["centroid_x"] = centroid[0]
        metrics["centroid_y"] = centroid[1]
        metrics["centroid_error_px"] = centroid_error
        score = metrics["score"]

        if centroid_error > self._config.max_centroid_error_px:
            # 形状分数仍随决策带出，但这一帧不能改变最佳值。
            self._last_gate_reasons = ("location_mismatch",)
            self._invalid_streak += 1
            if self._invalid_streak >= self._config.max_invalid_frames:
                return self._freeze(
                    "repeated_invalid_frames",
                    frame=frame,
                    score=score,
                    metrics=metrics,
                    gate_reasons=("location_mismatch",),
                )
            return None

        self._scores.append(score)
        return self._score_frame(
            frame=frame,
            score=score,
            metrics=metrics,
            centroid=centroid,
            centroid_error=centroid_error,
        )

    def _check_pairing(self, frame: MeasuredFrame) -> list[str]:
        reasons: list[str] = []
        if self._expected_write_seq is None:
            reasons.append("no_pending_write")
        else:
            if frame.write_seq != self._expected_write_seq:
                reasons.append("unpaired_frame")
            if (
                self._expected_written_at_s is not None
                and not frame.captured_at_s > self._expected_written_at_s
            ):
                reasons.append("old_frame")
        if self._last_capture_seq is not None and frame.capture_seq <= self._last_capture_seq:
            reasons.append("duplicate_frame")
        self._last_capture_seq = (
            frame.capture_seq
            if self._last_capture_seq is None
            else max(self._last_capture_seq, frame.capture_seq)
        )
        return reasons

    def _check_image(self, image: np.ndarray) -> list[str]:
        assert self._target is not None
        if image.ndim != 2 or image.shape != self._target.image.shape:
            return ["invalid_shape"]
        if not np.all(np.isfinite(image)):
            return ["not_finite"]
        if float(image.min()) < 0.0:
            return ["negative_intensity"]

        saturated_fraction = float(
            np.mean(image >= self._config.saturation_counts)
        )
        if saturated_fraction > self._config.max_saturated_fraction:
            return ["overexposed"]

        peak = float(image.max())
        if peak < self._config.min_peak_counts or float(image.sum()) < self._config.min_total_counts:
            return ["low_signal"]
        if float(image.std()) <= 0.0:
            return ["flat_frame"]
        return []

    def _score_frame(
        self,
        *,
        frame: MeasuredFrame,
        score: float,
        metrics: Mapping[str, float],
        centroid: tuple[float, float],
        centroid_error: float,
    ) -> LockDecision:
        best = self._best_score
        if best is None:
            self._best_score = score
            self._best_write_seq = frame.write_seq
            self._stalled_streak = 0
            return self._emit(
                LockAction.ACCEPT,
                "baseline_established",
                frame=frame,
                score=score,
                metrics=metrics,
                centroid=centroid,
                centroid_error=centroid_error,
            )

        improvement = best - score
        if improvement >= self._config.improvement_epsilon:
            self._best_score = score
            self._best_write_seq = frame.write_seq
            self._stalled_streak = 0
            return self._emit(
                LockAction.ACCEPT,
                "improved",
                frame=frame,
                score=score,
                metrics=metrics,
                centroid=centroid,
                centroid_error=centroid_error,
            )

        self._stalled_streak += 1
        degraded = score - best >= self._config.rollback_epsilon

        if self._oscillating():
            return self._freeze(
                "oscillating_corrections",
                frame=frame,
                score=score,
                metrics=metrics,
                centroid=centroid,
                centroid_error=centroid_error,
            )
        if degraded:
            # 维持期也要防持续漂移：连续回滚到上限仍会冻结。
            if self._stalled_streak >= self._config.max_non_improving:
                return self._freeze(
                    "non_improving_corrections",
                    frame=frame,
                    score=score,
                    metrics=metrics,
                    centroid=centroid,
                    centroid_error=centroid_error,
                )
            return self._emit(
                LockAction.ROLLBACK,
                "degraded",
                frame=frame,
                score=score,
                metrics=metrics,
                centroid=centroid,
                centroid_error=centroid_error,
            )
        if self._state is LockState.MAINTAINING:
            # 维持期「没有改进」是正常状态，不累计为故障。
            self._stalled_streak = 0
            return self._emit(
                LockAction.PROPOSE,
                "maintaining",
                frame=frame,
                score=score,
                metrics=metrics,
                centroid=centroid,
                centroid_error=centroid_error,
            )
        if self._stalled_streak >= self._config.max_non_improving:
            return self._freeze(
                "non_improving_corrections",
                frame=frame,
                score=score,
                metrics=metrics,
                centroid=centroid,
                centroid_error=centroid_error,
            )
        if (
            self._state is LockState.CONVERGING
            and self._stalled_streak >= self._config.converge_patience
            and self._stable()
        ):
            self._state = LockState.MAINTAINING
            self._stalled_streak = 0
            return self._emit(
                LockAction.PROPOSE,
                "maintaining",
                frame=frame,
                score=score,
                metrics=metrics,
                centroid=centroid,
                centroid_error=centroid_error,
            )
        return self._emit(
            LockAction.PROPOSE,
            "within_noise",
            frame=frame,
            score=score,
            metrics=metrics,
            centroid=centroid,
            centroid_error=centroid_error,
        )

    def _stable(self) -> bool:
        """最近若干分差在振荡幅度以内，才算「稳定地停在最佳附近」。"""
        window = list(self._scores)[-max(2, self._config.converge_patience + 1) :]
        if len(window) < 2:
            return True
        return (max(window) - min(window)) < self._config.oscillation_amplitude

    def _oscillating(self) -> bool:
        window = list(self._scores)[-self._config.oscillation_window :]
        if len(window) < self._config.oscillation_window:
            return False
        if (max(window) - min(window)) < self._config.oscillation_amplitude:
            return False
        flips = 0
        for before, middle, after in zip(window, window[1:], window[2:]):
            if (middle - before) * (after - middle) < 0.0:
                flips += 1
        return flips >= self._config.oscillation_min_flips

    def _freeze(
        self,
        reason: str,
        *,
        frame: MeasuredFrame,
        score: float | None,
        metrics: Mapping[str, float],
        centroid: tuple[float, float] | None = None,
        centroid_error: float | None = None,
        gate_reasons: tuple[str, ...] = (),
    ) -> LockDecision:
        self._state = LockState.FROZEN
        self._fault_count += 1
        return self._emit(
            LockAction.FAULT,
            reason,
            frame=frame,
            score=score,
            metrics=metrics,
            centroid=centroid,
            centroid_error=centroid_error,
            gate_reasons=gate_reasons,
        )

    def _emit(
        self,
        action: LockAction,
        reason: str,
        *,
        frame: MeasuredFrame,
        score: float | None,
        metrics: Mapping[str, float],
        centroid: tuple[float, float] | None,
        centroid_error: float | None,
        gate_reasons: tuple[str, ...] = (),
    ) -> LockDecision:
        decision = LockDecision(
            action=action,
            state=self._state,
            reason=reason,
            write_seq=frame.write_seq,
            capture_seq=frame.capture_seq,
            score=score,
            best_score=self._best_score,
            best_write_seq=self._best_write_seq,
            centroid_xy=centroid,
            centroid_error_px=centroid_error,
            metrics=dict(metrics),
            gate_reasons=gate_reasons,
        )
        self._callbacks.emit(decision)
        return decision

    def snapshot(self) -> dict[str, Any]:
        """只读状态快照，供日志/UI 轮询；不触发任何决策。"""
        return {
            "state": self._state.value,
            "best_score": self._best_score,
            "best_write_seq": self._best_write_seq,
            "pending_write_seq": self._expected_write_seq,
            "invalid_streak": self._invalid_streak,
            "stalled_streak": self._stalled_streak,
            "fault_count": self._fault_count,
            "last_gate_reasons": list(self._last_gate_reasons),
            "score_history": list(self._scores),
        }


# --------------------------------------------------------------------------- #
# LockConfig 阈值标定
# --------------------------------------------------------------------------- #
#
# ``LockConfig`` 的默认值只是"8/12-bit 相机"的起点，不是任何光路的保证。这一节把
# 一段实测帧变成可审计的 ``CalibrationSample``，再按固定的、写明的倍数推成
# 本次运行的阈值。故意不做"自动最优"：倍数都是保守的噪声地板倍数，操作员能在
# 页面上看到每个推导值和输入值，标定结论对错可以回头用同一段帧复核。
#
# 这里只依赖 numpy 与标准库，与监督器一样不碰设备、不碰 UI。


@dataclass(frozen=True)
class CalibrationSample:
    """一段实测帧的统计量；计数量纲是 CCD counts，重复性是逐帧标准差。"""

    dark_mean: float
    dark_sigma: float
    peak_counts: float
    total_counts: float
    saturated_fraction: float
    full_scale: float
    centroid_sigma_px: float
    score_sigma: float
    pixel_count: int

    def __post_init__(self) -> None:
        values = (
            self.dark_mean,
            self.dark_sigma,
            self.peak_counts,
            self.total_counts,
            self.saturated_fraction,
            self.full_scale,
            self.centroid_sigma_px,
            self.score_sigma,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("标定统计量必须全部有限")
        if self.dark_sigma < 0.0 or self.centroid_sigma_px < 0.0 or self.score_sigma < 0.0:
            raise ValueError("噪声/重复性标准差不能为负")
        if not 0.0 <= self.saturated_fraction <= 1.0:
            raise ValueError("saturated_fraction 必须在 [0, 1]")
        if self.full_scale <= 0.0 or self.peak_counts <= 0.0:
            raise ValueError("full_scale 与 peak_counts 必须为正")
        if self.pixel_count < 1:
            raise ValueError("pixel_count 至少为 1")


def target_on_frame(
    xy: np.ndarray,
    target: np.ndarray,
    frame_shape: tuple[int, int] | tuple[int, ...],
) -> np.ndarray:
    """把分析网格上的目标图铺到整帧坐标系，供整帧统计使用。

    ``calibration_stats_from_frames`` 要求目标与帧同形，而硬件的
    ``CapturedFrame.frame`` 是整幅 CCD，闭环目标只定义在分析窗口上。这里按
    ``xy`` 的**绝对像素坐标**把窗口目标写回整帧零场：窗口内是目标，窗口外是 0。
    暗场掩膜因此落在窗口外的背景上，不会被窗口内的信号污染。

    ``xy`` 越界（窗口超出相机幅面）或网格步长为 0 都会抛 ``ValueError`` ——
    那是操作员给错中心/窗口，不该在统计里被悄悄裁掉。
    """
    grid = np.asarray(xy, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if grid.ndim != 3 or grid.shape[-1] != 2:
        raise ValueError(f"xy 必须是 (H, W, 2) 的像素坐标，实际 {grid.shape}")
    if tgt.shape != grid.shape[:2]:
        raise ValueError(f"目标图形状 {tgt.shape} 与网格 {grid.shape[:2]} 不一致")
    if len(frame_shape) < 2:
        raise ValueError(f"帧形状至少需要两维，实际 {frame_shape}")
    rows, cols = int(frame_shape[0]), int(frame_shape[1])

    yi = np.rint(grid[..., 1]).astype(np.int64)
    xi = np.rint(grid[..., 0]).astype(np.int64)
    if yi.min() < 0 or xi.min() < 0 or yi.max() >= rows or xi.max() >= cols:
        raise ValueError(
            f"分析窗口超出帧范围 {rows}x{cols}："
            f"x [{int(xi.min())}, {int(xi.max())}]，y [{int(yi.min())}, {int(yi.max())}]"
        )
    out = np.zeros((rows, cols), dtype=np.float64)
    out[yi, xi] = tgt
    return out


def calibration_stats_from_frames(
    frames: Iterable[np.ndarray],
    *,
    target: np.ndarray,
    full_scale: float | None = None,
    dark_mask: np.ndarray | None = None,
) -> CalibrationSample:
    """从若干帧同形 CCD 图与固定目标图算 :class:`CalibrationSample`。

    ``dark_mask`` 给背景区（默认取目标图里强度为 0 的像素，全亮目标退到最外圈）；
    ``full_scale`` 用相机的饱和 counts，缺省时按峰值 70% 满量程保守估计 —— 后者只
    用于标定页的手动录入回填，真实设备采集会带准确的 ``CapturedFrame.full_scale``。
    """
    images = [np.asarray(frame, dtype=np.float64) for frame in frames]
    if not images:
        raise ValueError("标定至少需要一帧")
    shape = images[0].shape
    if len(shape) != 2:
        raise ValueError(f"标定帧必须是二维图，实际 {shape}")
    tgt = np.asarray(target, dtype=np.float64)
    if tgt.shape != shape:
        raise ValueError(f"目标图形状 {tgt.shape} 与标定帧 {shape} 不一致")

    if dark_mask is None:
        mask = tgt <= 0.0
        if not mask.any():
            mask = np.zeros(shape, dtype=bool)
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, -1] = True
    else:
        mask = np.asarray(dark_mask, dtype=bool)
        if mask.shape != shape or not mask.any():
            raise ValueError("dark_mask 必须是与帧同形且非空的二维布尔图")

    peak = max(float(image.max()) for image in images)
    scale = float(full_scale) if full_scale is not None else max(peak / 0.7, 1.0)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("full_scale 必须为正的有限值")

    dark = np.concatenate([image[mask] for image in images])
    dark_mean = float(dark.mean())
    dark_sigma = float(dark.std())
    total = float(np.median([float(image.sum()) for image in images]))
    saturated = float(np.mean([float(np.mean(image >= scale * 0.99)) for image in images]))

    centroids = [point for point in (_centroid(image) for image in images) if point is not None]
    if len(centroids) >= 2:
        spread = np.std(np.asarray(centroids, dtype=np.float64), axis=0)
        centroid_sigma = float(np.mean(spread))
    else:
        centroid_sigma = 0.0

    scores = [_shape_metrics(image, tgt)["score"] for image in images]
    score_sigma = float(np.std(np.asarray(scores, dtype=np.float64))) if len(scores) >= 2 else 0.0

    return CalibrationSample(
        dark_mean=dark_mean,
        dark_sigma=dark_sigma,
        peak_counts=peak,
        total_counts=total,
        saturated_fraction=saturated,
        full_scale=scale,
        centroid_sigma_px=centroid_sigma,
        score_sigma=score_sigma,
        pixel_count=int(shape[0] * shape[1]),
    )


def calibrate_lock_thresholds(
    sample: CalibrationSample,
    *,
    position_tolerance_ccd_px: float,
    grid_spacing_px: float,
) -> LockConfig:
    """按实测噪声地板推导本次运行的 :class:`LockConfig`。

    固定倍数（写在代码里，便于审计）：饱和阈值取满量程的 98% 且至少高于暗场 3σ；
    最小峰值为暗场均值 + 20σ；最小总量为标定总量的 2%；改进/回滚/振荡余量为评分
    重复性的 5×/15×/5×。位置容差仍由操作员给定，这里只把它从 CCD 像素换成网格单位。
    """
    if not np.isfinite(grid_spacing_px) or grid_spacing_px <= 0.0:
        raise ValueError("grid_spacing_px 必须为正的有限值")
    if not np.isfinite(position_tolerance_ccd_px) or position_tolerance_ccd_px <= 0.0:
        raise ValueError("position_tolerance_ccd_px 必须为正的有限值")

    scale = max(sample.full_scale, 1.0)
    saturation_counts = max(
        0.98 * scale,
        sample.dark_mean + 3.0 * sample.dark_sigma,
        1.0,
    )
    max_saturated_fraction = float(
        np.clip(sample.saturated_fraction * 3.0, 0.0005, 0.05)
    )
    min_peak_counts = max(sample.dark_mean + 20.0 * sample.dark_sigma, 1.0)
    min_total_counts = max(0.02 * sample.total_counts, 1.0)

    score_noise = max(sample.score_sigma, 1e-4)
    improvement_epsilon = max(5.0 * score_noise, 1e-4)
    rollback_epsilon = max(15.0 * score_noise, 2.0 * improvement_epsilon)

    return LockConfig(
        saturation_counts=float(saturation_counts),
        max_saturated_fraction=max_saturated_fraction,
        min_peak_counts=float(min_peak_counts),
        min_total_counts=float(min_total_counts),
        max_centroid_error_px=float(position_tolerance_ccd_px) / float(grid_spacing_px),
        improvement_epsilon=float(improvement_epsilon),
        rollback_epsilon=float(rollback_epsilon),
        oscillation_amplitude=max(5.0 * score_noise, 0.01),
    )
