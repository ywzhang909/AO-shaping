"""相位变化速度计算的**纯算法层** (无设备依赖).

本模块只处理两件事:

1. :func:`far_field_metric` —— 从一帧远场 (Fourier 面) 图像里取 0 级光斑的
   强度指标. 0 级位置用帧内全局 ``argmax`` (仓库固化事实: 2f 光路的光轴
   不在相机几何中心), 也允许调用方传入已标定的 ``zero_center``.
2. :func:`analyze_step` —— 从一条标量指标时间序列里提取响应时间 / 稳定时间.

⚠️ **量纲与物理含义**: 输入序列的采样率是**相机帧率**, 所以这里算出来的
``t_response_ms`` / ``t_settle_ms`` 是 **SLM + CCD 组合系统** 的响应, 不是
孤立的 SLM 上升时间. 相机曝光 + 读出本身就有 1 帧以上的延迟, 因此
``t=0`` 定义为"写入新相位后的第一帧", 真实 SLM 翻转可能早就发生. 该限制
在返回值的 ``notes`` 里明确写出, 调用方必须原样上报.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["PhaseStepResult", "analyze_step", "far_field_metric"]


#: 无阶跃判据: 幅度小于 ``_ZERO_AMPLITUDE_REL * max(1, |before|, |after|)``
#: 时视为"没有可测的跃迁", 避免噪声被当成阶跃.
_ZERO_AMPLITUDE_REL = 1e-6


@dataclass
class PhaseStepResult:
    """一次相位阶跃的时序分析结果.

    Attributes:
        t_response_ms: 首次达到 ``response_fraction`` 处的时间 (ms); 无跃迁为 ``None``.
        t_settle_ms: 从该帧起全部后续采样都落在稳定带内的时间 (ms); 未稳定为 ``None``.
        frames_to_settle: 对应 :attr:`t_settle_ms` 的帧序号 (0 基); 未稳定为 ``None``.
        fps: 计算所用的采样帧率 (帧/秒).
        step_metric_before: 阶跃前基准电平 (前 ``baseline_window`` 帧均值).
        step_metric_after: 阶跃后终值电平 (末 ``settle_window`` 帧均值).
        trace: 原始指标序列 (float), 与采样序号一一对应.
        notes: 方法与限制说明 (含"受相机帧率限制"的免责声明).
    """

    t_response_ms: float | None
    t_settle_ms: float | None
    frames_to_settle: int | None
    fps: float
    step_metric_before: float
    step_metric_after: float
    trace: list[float]
    notes: list[str] = field(default_factory=list)


def far_field_metric(
    frame: np.ndarray,
    *,
    zero_center: tuple[int, int] | None = None,
    subtract_background: bool = True,
) -> float:
    """取远场 0 级光斑的强度指标.

    0 级默认由帧内全局 ``argmax`` 定位 (2f 光路中光轴不在帧中心, 严禁用
    几何中心代替). 返回值为该像素的亮度; ``subtract_background=True`` 时
    减去全帧中位数作为加性背景估计.

    Args:
        frame: 单帧二维灰度图 (uint8 / uint16 / float 均可).
        zero_center: 可选的 ``(row, col)`` 0 级中心; ``None`` 则自动 argmax.
        subtract_background: 是否扣除全帧中位数背景.

    Returns:
        float: 0 级强度指标 (已扣背景时可能为负, 例如背景比峰值还亮的坏帧).
    """
    arr = np.asarray(frame)
    if arr.ndim != 2:
        raise ValueError(f"far_field_metric expects a 2-D frame, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("far_field_metric got an empty frame")

    if zero_center is None:
        cy, cx = (int(v) for v in np.unravel_index(int(np.argmax(arr)), arr.shape))
    else:
        cy, cx = int(zero_center[0]), int(zero_center[1])
        cy = min(max(cy, 0), arr.shape[0] - 1)
        cx = min(max(cx, 0), arr.shape[1] - 1)

    peak = float(arr[cy, cx])
    if subtract_background:
        peak -= float(np.median(arr))
    return peak


def analyze_step(
    metric_trace: list[float] | np.ndarray,
    *,
    fps: float,
    settle_fraction: float = 0.1,
    baseline_window: int = 3,
    settle_window: int = 3,
    response_fraction: float = 0.9,
) -> PhaseStepResult:
    """从相位阶跃后的标量指标序列提取响应时间与稳定时间.

    基准电平取前 ``baseline_window`` 帧均值, 终值电平取末 ``settle_window``
    帧均值, 跃迁幅度 ``amplitude = after - before``:

    * **响应时间**: 首个满足 ``sign * (m - before) >= response_fraction * |amplitude|``
      的采样序号 ``i``, ``t = 1000 * i / fps`` (升/降沿都支持).
    * **稳定时间**: 首个满足"从它开始**所有**剩余采样都在
      ``|m - after| <= settle_fraction * |amplitude|`` 带内"的序号.
      永远找不到 ⇒ ``None`` (噪声或持续漂移).

    无跃迁 (幅度约等于 0) 时返回两个时间均为 ``None`` 的结果并附 note,
    不抛异常. 采样间隔由 ``fps`` 换算, 必须是正数.

    Args:
        metric_trace: 相位切换**之后**的逐帧指标 (0 基序号 0 = 切换后首帧).
        fps: 采样帧率 (帧/秒), 通常是实测帧间隔的倒数.
        settle_fraction: 稳定带相对跃迁幅度的比例 (默认 10%).
        baseline_window: 基准电平使用的起始帧数.
        settle_window: 终值电平使用的末尾帧数.
        response_fraction: 响应判据相对跃迁幅度的比例 (默认 90%).

    Returns:
        PhaseStepResult: 含响应时间 / 稳定时间 / 帧数 / 方法与限制说明.
    """
    trace = [float(v) for v in metric_trace]
    fps = float(fps)
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    if not 0.0 < settle_fraction < 1.0:
        raise ValueError(f"settle_fraction must be in (0, 1), got {settle_fraction!r}")
    if not 0.0 < response_fraction <= 1.0:
        raise ValueError(f"response_fraction must be in (0, 1], got {response_fraction!r}")

    notes = [
        "t=0 定义为写入新相位后的第一帧; 相机曝光/读出的延迟已包含在结果中.",
        "测得的是 SLM+CCD 组合系统响应, 受相机帧率限制, 不是孤立 SLM 上升时间.",
        (
            f"基准=前 {baseline_window} 帧均值, 终值=末 {settle_window} 帧均值; "
            f"响应阈={response_fraction:.0%}, 稳定带=±{settle_fraction:.0%} 幅度."
        ),
    ]

    n = len(trace)
    if n == 0:
        notes.append("空序列, 无法分析.")
        return PhaseStepResult(None, None, None, fps, 0.0, 0.0, trace, notes)

    bw = max(1, min(int(baseline_window), n))
    sw = max(1, min(int(settle_window), n))
    before = float(np.mean(trace[:bw]))
    after = float(np.mean(trace[-sw:]))
    amplitude = after - before
    scale = max(1.0, abs(before), abs(after))

    if abs(amplitude) <= _ZERO_AMPLITUDE_REL * scale:
        notes.append(
            "未测到可分辨的阶跃 (前后电平差在数值噪声内); "
            "时间与稳定帧数置 None. 若是平场 piston 阶跃属预期现象, 建议改用 blaze 图案."
        )
        return PhaseStepResult(None, None, None, fps, before, after, trace, notes)

    sign = 1.0 if amplitude > 0.0 else -1.0
    target_offset = response_fraction * abs(amplitude)
    response_idx: int | None = None
    for idx, value in enumerate(trace):
        if sign * (value - before) >= target_offset:
            response_idx = idx
            break

    band = settle_fraction * abs(amplitude)
    settle_idx: int | None = None
    for idx in range(n):
        if all(abs(value - after) <= band for value in trace[idx:]):
            settle_idx = idx
            break

    dt_ms = 1000.0 / fps
    t_response_ms = None if response_idx is None else response_idx * dt_ms
    t_settle_ms = None if settle_idx is None else settle_idx * dt_ms
    if settle_idx is None:
        notes.append("序列结束前未进入稳定带, 稳定时间置 None.")

    return PhaseStepResult(
        t_response_ms=t_response_ms,
        t_settle_ms=t_settle_ms,
        frames_to_settle=settle_idx,
        fps=fps,
        step_metric_before=before,
        step_metric_after=after,
        trace=trace,
        notes=notes,
    )
