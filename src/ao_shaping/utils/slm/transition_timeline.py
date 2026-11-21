"""动态整形过渡时间轴的纯函数层（Qt 无关）。

为什么单独成模块
----------------
这套数学原本住在 :mod:`ao_shaping.gui.slm.dynamic_shaping`，但那个模块在
**模块级**引入 PySide6（``DynamicShapingWorker`` 继承 ``QThread``）。headless
采集脚本（``scripts/slm/dynamic_shaping_capture_20261120.py``）需要同一套
「灰度域插值 + 墙钟时间轴」口径，却既不该拖起 Qt、也不该自带一份副本 ——
两份常量（``_LCOS_FULL_FLIP_S`` / 环绕周期 1024）迟早会漂移。

因此这里只放**不碰设备、不碰 Qt** 的部分：常量、缓动、:class:`TransitionPlan`、
:class:`TransitionEstimate`、灰度最短环绕混合与估算。``dynamic_shaping`` 从本模块
re-export 原名（公共接口与 ``__all__`` 不变），GUI 与 headless 共用唯一实现。

口径与已知坑见 :mod:`ao_shaping.gui.slm.dynamic_shaping` 模块 docstring：
端点走完整 ``create_phase_from_array`` 管线各一次，中间帧只在最终灰度码上按
最短环绕路径线性混合；时间轴以墙钟为准，硬件慢时跳过中间帧。

本模块的 import 只允许 ``numpy`` 与标准库：``tests/ao_shaping/test_conventions.py``
的 utils/ 分层门禁禁止上层（drivers / gui / algorithm …）在模块作用域被拉进来。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "EASING_LABELS",
    "MAX_DWELL_MS",
    "MAX_GRAY_CODE",
    "TransitionEstimate",
    "TransitionPlan",
    "as_panel_array",
    "blend_gray",
    "ease",
    "estimate_transition",
    "gray_to_phase",
    "pixel_flip_wait_s",
    "wrapped_gray_delta",
]


#: 10-bit 面板的满码；1024 会回绕成 0，绝不能作为可取值写入。
MAX_GRAY_CODE = 1023
#: 灰度码的环绕周期（0…1023 共 1024 个码）。
#: 驱动 ``create_phase_from_array`` 结尾按 ``mod max_grayscale(1023)`` 收码，
#: 因此端点码实际落在 0…1022、永不取 1023；但面板是 10-bit，物理上 1023≡0，
#: 环绕周期仍为 1024。端点逐码精确，中间帧经 :func:`blend_gray` 走 1024 周期，
#: 在 0/1022 边界上取的是物理最短路径（经 1023），与驱动端点不冲突。
_GRAY_PERIOD = 1024
#: LCOS 0→2π 满幅翻转的最大耗时（秒）。逐帧翻转等待按码变化量线性估算，
#: 与驱动 :meth:`Santec._estimate_pixel_flip_wait` 同一口径。
_LCOS_FULL_FLIP_S = 0.2
#: 「每步停留」的上限（毫秒）。超过这个量级已不是过渡，而是在逐帧定格观察。
MAX_DWELL_MS = 2000.0

#: 缓动曲线：内部键 → 界面文案。``smooth`` 是 3t²-2t³（S 形），起止都平缓，
#: 在远场上看不出「突然起步 / 突然刹车」。
EASING_LABELS: dict[str, str] = {
    "smooth": "平滑 S 形（推荐）",
    "linear": "线性",
    "ease_in": "缓入",
    "ease_out": "缓出",
}


def ease(t: float, kind: str = "smooth") -> float:
    """把线性进度 0…1 映射成缓动进度。越界自动钳位。"""
    t = min(1.0, max(0.0, float(t)))
    if kind == "linear":
        return t
    if kind == "smooth":
        return t * t * (3.0 - 2.0 * t)
    if kind == "ease_in":
        return t * t
    if kind == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    raise ValueError(f"未知缓动曲线: {kind!r}（可选 {sorted(EASING_LABELS)}）")


@dataclass(frozen=True)
class TransitionPlan:
    """一次过渡的时间轴参数（与硬件无关，可先算清再决定是否下发）。

    Attributes:
        duration_s: 过渡时长（秒）。时间轴以墙钟为准，不是帧数。
        fps: 目标帧率（帧/秒）。硬件翻转等待更久时会自动降速，最终以实际为准。
        easing: :data:`EASING_LABELS` 的键。
        dwell_ms: 每步停留（毫秒）。0 = 只按像素翻转时间自动；>0 时执行器取
            ``max(dwell_ms, 翻转时间)``，永远不允许短于物理翻转。
    """

    duration_s: float
    fps: int = 10
    easing: str = "smooth"
    dwell_ms: float = 0.0

    MIN_DURATION_S: float = 0.5
    MAX_DURATION_S: float = 600.0
    MIN_FPS: int = 1
    MAX_FPS: int = 30

    def __post_init__(self) -> None:
        if not (self.MIN_DURATION_S <= self.duration_s <= self.MAX_DURATION_S):
            raise ValueError(
                f"过渡时长须在 {self.MIN_DURATION_S}–{self.MAX_DURATION_S} s，"
                f"收到 {self.duration_s}"
            )
        if not (self.MIN_FPS <= int(self.fps) <= self.MAX_FPS):
            raise ValueError(
                f"帧率须在 {self.MIN_FPS}–{self.MAX_FPS} 帧/秒，收到 {self.fps}"
            )
        if self.easing not in EASING_LABELS:
            raise ValueError(f"未知缓动曲线: {self.easing!r}")
        if not (0.0 <= float(self.dwell_ms) <= MAX_DWELL_MS):
            raise ValueError(
                f"每步停留须在 0–{MAX_DWELL_MS:.0f} ms（0 = 自动），收到 {self.dwell_ms}"
            )

    @property
    def dwell_s(self) -> float:
        """用户设定的每步停留（秒）；0 表示只按像素翻转时间自动。"""
        return max(0.0, float(self.dwell_ms)) / 1000.0

    @property
    def frame_count(self) -> int:
        """含首尾端点的帧数；至少 2 帧（起点 + 目标）。"""
        return max(2, int(round(self.duration_s * int(self.fps))) + 1)

    @property
    def frame_interval_s(self) -> float:
        """相邻帧的名义间隔。最后一帧与第一帧之间恰好铺满 ``duration_s``。"""
        return self.duration_s / (self.frame_count - 1)

    def progress_of_frame(self, index: int) -> float:
        """帧号（0 基）→ 线性进度 0…1。"""
        index = min(self.frame_count - 1, max(0, int(index)))
        return index / (self.frame_count - 1)


@dataclass(frozen=True)
class TransitionEstimate:
    """下发前的速率估算（用于界面明确「实际会多快」）。

    Attributes:
        frames: 帧数。
        interval_s: 名义帧间隔。
        max_step_gray: 相邻帧的最大灰度码变化（全画面）。
        flip_wait_s: 按该最大变化量估算的 LCOS 翻转等待（单帧）。
        dwell_s: 实际执行的每步停留 = ``max(flip_wait_s, 用户设定)``。
        dwell_clamped: 用户设定的停留低于物理翻转时间，已被抬到 ``flip_wait_s``。
        expected_duration_s: 计入停留后的预计实际时长。
        limited_by_panel: 每步停留是否比设定帧间隔更慢（会拉长实际时长）。
    """

    frames: int
    interval_s: float
    max_step_gray: float
    flip_wait_s: float
    expected_duration_s: float
    limited_by_panel: bool
    dwell_s: float = 0.0
    dwell_clamped: bool = False


def pixel_flip_wait_s(max_step_gray: float) -> float:
    """按单帧最大灰度变化估算 LCOS 像素翻转等待（秒），上限满幅翻转时间。"""
    step = max(0.0, float(max_step_gray))
    return min(_LCOS_FULL_FLIP_S, step / MAX_GRAY_CODE * _LCOS_FULL_FLIP_S)


def as_panel_array(arr: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim != 2:
        raise ValueError(f"{name} 必须是二维面板数组，收到 shape={out.shape}")
    return out


def wrapped_gray_delta(start: np.ndarray, target: np.ndarray) -> np.ndarray:
    """目标相对起点的最短环绕码差，范围 -512…511。

    直接相减在 0/1023 边界上会得到 ±1023 的假跳变，使中间帧出现整周相位翻转；
    取模到半周期内后，每个像素在过渡中最多走半周期。
    """
    start = as_panel_array(start, "起点灰度")
    target = as_panel_array(target, "目标灰度")
    if start.shape != target.shape:
        raise ValueError(f"起止灰度尺寸不一致: {start.shape} vs {target.shape}")
    delta = (
        target.astype(np.int32) - start.astype(np.int32) + _GRAY_PERIOD // 2
    ) % _GRAY_PERIOD - _GRAY_PERIOD // 2
    return delta.astype(np.int16)


def blend_gray(start: np.ndarray, target: np.ndarray, progress: float) -> np.ndarray:
    """按进度混合两幅最终灰度（逐像素最短环绕路径），返回 uint16 面板码。

    取模保证结果恒在 ``0…1023``（10-bit）；1024 会被固件回绕成 0，绝不外送。
    """
    t = min(1.0, max(0.0, float(progress)))
    start = as_panel_array(start, "起点灰度")
    if t <= 0.0:
        return start.astype(np.uint16, copy=True)
    delta = wrapped_gray_delta(start, target)
    frame = (start.astype(np.int32) + np.rint(delta.astype(np.float64) * t)) % _GRAY_PERIOD
    return frame.astype(np.uint16)


def estimate_transition(
    plan: TransitionPlan,
    start_gray: np.ndarray,
    target_gray: np.ndarray,
) -> TransitionEstimate:
    """估算帧数、单帧最大码变化与实际时长。

    驱动对每帧的翻转等待按「最大码变化 / 满量程」线性估算（上限 200 ms）。
    若该等待（或用户设定的每步停留）已超过名义帧间隔，实际过渡必然比设定更慢
    —— 界面必须把这件事说清楚，而不是让用户以为「设了 6 秒就是 6 秒」。
    """
    delta = wrapped_gray_delta(start_gray, target_gray)
    frames = plan.frame_count
    max_delta = float(np.abs(delta).max()) if delta.size else 0.0
    per_frame = max_delta / max(1, frames - 1)
    flip_wait = pixel_flip_wait_s(per_frame)
    user_dwell = plan.dwell_s
    dwell = max(flip_wait, user_dwell)
    limited = dwell > plan.frame_interval_s * 1.05
    # 首帧没有上一帧基准，执行器保守按满幅翻转停留（200 ms）。
    expected = (frames - 1) * max(plan.frame_interval_s, dwell) + _LCOS_FULL_FLIP_S
    return TransitionEstimate(
        frames=frames,
        interval_s=plan.frame_interval_s,
        max_step_gray=per_frame,
        flip_wait_s=flip_wait,
        expected_duration_s=expected,
        limited_by_panel=limited,
        dwell_s=dwell,
        dwell_clamped=user_dwell > 0.0 and user_dwell < flip_wait - 1e-9,
    )


def gray_to_phase(gray: np.ndarray) -> np.ndarray:
    """最终灰度码 → 用于远场示意的弧度（``code / 1023 · 2π``）。

    只用于预览：真实下发不经此反变换，避免把「最终码」重新当成原始相位。
    """
    return np.asarray(gray, dtype=np.float32) / MAX_GRAY_CODE * (2.0 * np.pi)
