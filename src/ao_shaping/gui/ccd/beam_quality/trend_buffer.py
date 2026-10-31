"""光束质量趋势的环形缓冲 (纯 python + deque, 无 Qt 依赖)。

记录每次采样时刻与 :meth:`FrameMetrics.as_scalars` 给出的标量字段,
供趋势折线图与 CSV 导出使用; 采样与绘图完全解耦, 因此可以离线单测。
"""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

# 与 analyzer.FrameMetrics.as_scalars() 的键一致
SCALAR_KEYS: tuple[str, ...] = ("max", "mean", "std", "cx", "cy", "sat_ratio")

CSV_HEADER: tuple[str, ...] = ("t_iso", "t_rel_s", *SCALAR_KEYS)


@dataclass(frozen=True, slots=True)
class TrendSample:
    """一次采样的标量快照。``timestamp`` 为 monotonic 或 epoch 秒, 只要求单调。"""

    timestamp: float
    max: float
    mean: float
    std: float
    cx: float
    cy: float
    sat_ratio: float

    def as_dict(self) -> dict[str, float]:
        return {key: float(getattr(self, key)) for key in SCALAR_KEYS}


def _coerce_scalars(values: Any) -> dict[str, float]:
    """接受 dict 或带 ``as_scalars()`` 的对象 (如 FrameMetrics)。"""
    source = values.as_scalars() if hasattr(values, "as_scalars") else values
    missing = [key for key in SCALAR_KEYS if key not in source]
    if missing:
        raise KeyError(f"趋势采样缺少字段: {missing}")
    return {key: float(source[key]) for key in SCALAR_KEYS}


class TrendBuffer:
    """定长环形缓冲, 记录时间戳 + 标量指标。

    典型用法 (UI 线程)::

        buffer.append(time.monotonic(), metrics)
        recent = buffer.records(window_s=30.0)

    暂停 (``paused = True``) 时 :meth:`append` 直接丢弃新样本, 已有数据保留,
    因此图表继续显示历史曲线, 只是不再增长。
    """

    def __init__(self, maxlen: int = 3600) -> None:
        if maxlen <= 0:
            raise ValueError(f"maxlen 必须为正, 实际 {maxlen}")
        self._samples: deque[TrendSample] = deque(maxlen=maxlen)
        self._paused = False
        self._t0: float | None = None

    # ------------------------------------------------------------------ 状态
    @property
    def paused(self) -> bool:
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        self._paused = bool(value)

    def set_paused(self, paused: bool) -> bool:
        """设置暂停状态并返回设置后的值。"""
        self.paused = paused
        return self._paused

    @property
    def t0(self) -> float | None:
        """首样本时间戳 (相对时间的原点)。"""
        return self._t0

    @property
    def latest(self) -> TrendSample | None:
        return self._samples[-1] if self._samples else None

    @property
    def maxlen(self) -> int:
        return self._samples.maxlen or 0

    def __len__(self) -> int:
        return len(self._samples)

    # ---------------------------------------------------------------- 写入
    def append(self, timestamp: float, values: Any) -> TrendSample | None:
        """追加一条采样; 暂停时返回 ``None`` 且不记录。

        ``values`` 可以是 ``FrameMetrics`` 或含 ``max/mean/std/cx/cy/sat_ratio``
        的映射。
        """
        if self._paused:
            return None
        scalars = _coerce_scalars(values)
        sample = TrendSample(timestamp=float(timestamp), **scalars)
        if self._t0 is None:
            self._t0 = sample.timestamp
        self._samples.append(sample)
        return sample

    def clear(self) -> None:
        """清空全部样本, 相对时间原点一并复位。"""
        self._samples.clear()
        self._t0 = None

    # ---------------------------------------------------------------- 读取
    def records(self, window_s: float | None = None) -> list[TrendSample]:
        """按时间窗取样本, 时间原点为**最新**样本 (只看最近 window_s 秒)。

        ``window_s=None`` 表示全部。从尾部倒着收集, 命中窗口边界即停止,
        因此代价与返回条数成正比。
        """
        if window_s is None or not self._samples:
            return list(self._samples)
        if window_s <= 0:
            return []
        last = self._samples[-1].timestamp
        cut = last - float(window_s)
        picked: list[TrendSample] = []
        for sample in reversed(self._samples):
            if sample.timestamp < cut:
                break
            picked.append(sample)
        picked.reverse()
        return picked

    def relative_time(self, sample: TrendSample) -> float:
        """样本相对首个样本的秒数。"""
        if self._t0 is None:
            return 0.0
        return sample.timestamp - self._t0

    def iter_scalars(self, key: str) -> Iterable[float]:
        """按字段迭代全部样本 (方便画图前做统计)。"""
        if key not in SCALAR_KEYS:
            raise KeyError(f"未知字段 {key!r}, 可选: {SCALAR_KEYS}")
        return (float(getattr(sample, key)) for sample in self._samples)

    # ---------------------------------------------------------------- 导出
    def export_csv(self, path: str | Path) -> Path:
        """导出为 CSV (列: t_iso, t_rel_s, max, mean, std, cx, cy, sat_ratio)。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_HEADER)
            for sample in self._samples:
                writer.writerow(
                    [
                        datetime.fromtimestamp(sample.timestamp).isoformat(
                            timespec="milliseconds"
                        ),
                        f"{self.relative_time(sample):.3f}",
                        f"{sample.max:.3f}",
                        f"{sample.mean:.3f}",
                        f"{sample.std:.3f}",
                        f"{sample.cx:.2f}",
                        f"{sample.cy:.2f}",
                        f"{sample.sat_ratio:.6f}",
                    ]
                )
        logger.info("趋势 CSV 已导出: {} ({} 条)", target, len(self._samples))
        return target
