"""Fixed-exposure paired decisions with an explicit measured acceptance metric."""
from __future__ import annotations

import math
from collections.abc import Sequence


def abba_schedule(pairs: Sequence[tuple[str, str]]) -> list[str]:
    """把若干 (A, B) 对展开成逐对连续的 ``A-B-B-A`` 采集序列。

    每对占用 4 个连续帧位: 两次 A 夹住两次 B, 于是同一对内部的漂移可以用
    :func:`paired_decision` 的 ``drift`` 项度量。序列是纯标签表, 与设备无关。

    Args:
        pairs: ``(a_label, b_label)`` 对列表。

    Returns:
        长度 ``4 * len(pairs)`` 的标签列表, 每对形如 ``[a, b, b, a]``。
    """
    order: list[str] = []
    for a, b in pairs:
        order.extend([a, b, b, a])
    return order


def pair_window(order: Sequence[str], a: str, b: str) -> tuple[int, int, int, int]:
    """在 ``order`` 里定位某一对的连续 ``[a, b, b, a]`` 窗口下标。

    Args:
        order: :func:`abba_schedule` 产出的采集顺序。
        a: 基线标签。
        b: 候选标签。

    Returns:
        ``(i_a1, i_b1, i_b2, i_a2)`` 四个帧位下标, 与 :func:`paired_decision`
        的四个位置参数一一对应。

    Raises:
        RuntimeError: 顺序里找不到该对的连续窗口。
    """
    needle = [a, b, b, a]
    for i in range(len(order) - 3):
        if list(order[i : i + 4]) == needle:
            return (i, i + 1, i + 2, i + 3)
    raise RuntimeError(f"ABBA window {a}/{b} not found in schedule {list(order)}")


def paired_decision(a1, b1, b2, a2, *, metric="loss",
                    rel_tol: float = 0.0, abs_tol: float = 0.0):
    if metric not in {"loss", "cv"}:
        raise ValueError("acceptance metric must be loss or cv")
    values = [float(row["metrics"]["cv"] if metric == "cv" else row["loss"])
              for row in (a1, b1, b2, a2)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite paired metric")
    if rel_tol < 0 or abs_tol < 0:
        raise ValueError("tolerances must be non-negative")
    baseline = .5*(values[0]+values[3])
    candidate = .5*(values[1]+values[2])
    drift = abs(values[3]-values[0])
    tolerance = max(abs(baseline) * float(rel_tol), float(abs_tol))
    return dict(accept_metric=metric, baseline=baseline, candidate=candidate,
                drift=drift, tolerance=tolerance,
                accepted=candidate < baseline - drift - tolerance)
