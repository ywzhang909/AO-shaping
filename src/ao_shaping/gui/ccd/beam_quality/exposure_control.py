"""自动曝光与多帧平均采样 (纯 numpy, 便于离线测试)。

两件事各自独立, 都由界面调用:

* **自动曝光**: 取一帧的 ``最大值 / 中位数 / 平均值`` 之一作为反馈量, 与目标灰度
  比较后按倍率调整曝光时间 —— 在曝光时间是"亮度线性旋钮"的前提下, 目标/当前 就是
  一步到位的倍率, 因此单次最多动 ``MAX_STEP_RATIO`` 倍防止过冲; 进入
  ``±TOLERANCE`` 容差带即停手, 避免在目标附近来回抖。
* **多帧平均**: 保留最近 1~10 帧做累积平均 (``FrameAverage``), 降噪后再交给
  特征计算; 1 帧时原样透传, 不做任何拷贝。

统计量的"采样"含义是整帧统计 (含背景), 不做阈值门限 —— 自动曝光要守恒的是
画面整体亮度, 而特征计算的包围椭圆另有自己的半高阈值口径。
"""

from __future__ import annotations

from collections import deque

import numpy as np

#: (显示名, 内部键) —— 自动曝光的反馈统计量
AUTO_STATISTICS: tuple[tuple[str, str], ...] = (
    ("最大值", "max"),
    ("中位数", "median"),
    ("平均值", "mean"),
)
#: 内部键顺序, 与界面上下拉框的 index 一致
AUTO_STATISTIC_KEYS: tuple[str, ...] = tuple(key for _label, key in AUTO_STATISTICS)

#: 每种统计量的默认目标灰度 (MONO8 的 0-255 量纲)
DEFAULT_TARGETS: dict[str, float] = {"max": 200.0, "median": 128.0, "mean": 100.0}

#: 目标 ±3% 视为收敛
TOLERANCE = 0.03
#: 单次调整最多 4 倍 (曝光时间是线性旋钮, 但相机响应与饱和区并非线性)
MAX_STEP_RATIO = 4.0
#: 相对改动小于 1% 不值得再动一次相机 (MiiCam 改曝光有 ~1s 的拉流开销)
MIN_REL_STEP = 0.01


def frame_statistic(img: np.ndarray, kind: str = "max") -> float:
    """按 ``kind`` (``max`` / ``median`` / ``mean``) 取整帧统计量。

    单位与输入一致; 多帧平均后的 float32 画面会得到小数灰度。
    """
    arr = np.asarray(img)
    if arr.size == 0:
        raise ValueError("空画面没有统计量")
    key = str(kind).lower()
    if key == "max":
        return float(arr.max())
    if key == "median":
        return float(np.median(arr))
    if key == "mean":
        return float(arr.mean())
    raise ValueError(f"未知的统计量: {kind!r}")


def is_converged(current: float, target: float, tolerance: float = TOLERANCE) -> bool:
    """当前统计量是否已落在目标容差带内。"""
    if not np.isfinite(current) or not np.isfinite(target) or target <= 0:
        return False
    return abs(current - target) <= tolerance * abs(target)


def next_exposure(
    current: float,
    target: float,
    exposure_ms: float,
    *,
    min_ms: float,
    max_ms: float,
    tolerance: float = TOLERANCE,
    max_step_ratio: float = MAX_STEP_RATIO,
) -> float | None:
    """算下一次曝光 (ms); 已收敛 / 已触限 / 输入无效时返回 ``None`` (不动作)。

    倍率 = 目标/当前, 单次限幅到 ``1/max_step_ratio``~``max_step_ratio``,
    再夹到 ``[min_ms, max_ms]``; 若夹完几乎没变 (相对 <``MIN_REL_STEP``),
    说明已经贴到极限, 返回 ``None`` 让界面提示"已到曝光极限"。
    """
    values = (current, target, exposure_ms, min_ms, max_ms)
    if not all(np.isfinite(v) for v in values):
        return None
    if current <= 0 or target <= 0 or exposure_ms <= 0 or max_ms <= min_ms:
        return None
    if is_converged(current, target, tolerance):
        return None
    ratio = float(target) / float(current)
    ratio = min(max_step_ratio, max(1.0 / max_step_ratio, ratio))
    wanted = float(np.clip(exposure_ms * ratio, min_ms, max_ms))
    if abs(wanted - exposure_ms) < MIN_REL_STEP * max(exposure_ms, 1e-9):
        return None
    return wanted


class FrameAverage:
    """滚动 N 帧平均 (``size=1`` 时原样透传)。

    累加和常驻一份 float32 缓冲, 每来一帧只做一次加法/减法, 不保留 N 份副本,
    因此 10 帧模式下的常驻内存 ≈ 帧数 × 单帧大小 + 1 份累加和。
    """

    def __init__(self, size: int = 1) -> None:
        self._size = max(1, int(size))
        self._frames: deque[np.ndarray] = deque(maxlen=self._size)
        self._acc: np.ndarray | None = None

    @property
    def size(self) -> int:
        return self._size

    @property
    def filled(self) -> int:
        """当前参与平均的帧数。"""
        return len(self._frames)

    def reset(self) -> None:
        self._frames.clear()
        self._acc = None

    def push(self, frame: np.ndarray) -> np.ndarray:
        """推入最新帧, 返回最近 ``size`` 帧的平均画面。"""
        arr = np.asarray(frame)
        if self._size <= 1:
            return arr
        if self._acc is None or self._acc.shape != arr.shape:
            # 分辨率变了 (换相机 / binning 开关): 丢掉旧历史重建累加和
            self._frames.clear()
            self._acc = np.zeros(arr.shape, dtype=np.float32)
        elif len(self._frames) == self._size:
            # deque 到容量后 append 会挤掉最旧一帧, 先从累加和里减掉它
            self._acc -= self._frames[0]
        self._frames.append(arr)
        np.add(self._acc, arr, out=self._acc)
        return self._acc / len(self._frames)
