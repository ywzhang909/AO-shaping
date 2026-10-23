"""载波位移的**实测**判定：从两帧（无载波 / 有载波）量出整形光斑移动了多少、往哪走。

为什么必须实测而不能算：面板上的斜坡方向与 CCD 的平移方向之间隔着一层**安装姿态**。
本实验系统的 SLM 是**竖向安装**的，面板 x 向斜坡在 CCD 上表现为 **y 向**位移；
若按 `K/周期` 只算标量、再假定"x 向斜坡 ⇒ CCD x 向位移"，就会得到方向完全错的读数
（2026-10-23 就因此把 `at_plus/at_minus` 沿 x 摆了两个框，量到的是背景不对称）。

物理基础：零级（未被调制的光）在两帧里位置相同；只有被调制的那部分随载波平移。
所以差分图 `carrier - best` 的**正峰**是新位置、**负峰**是原位置。

本模块纯 numpy，供 runner（运行时就地判定）与离线报告共用，是位移判定的唯一实现。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _centroid_in_window(image: npt.NDArray[np.floating], win: int) -> tuple[float, float]:
    """先粗定位全局峰，再在局部窗口内做强度加权质心（比裸 argmax 抗噪）。"""
    row, col = np.unravel_index(int(np.argmax(image)), image.shape)
    half = max(2, int(win) // 2)
    row_slice = slice(max(0, row - half), min(image.shape[0], row + half + 1))
    col_slice = slice(max(0, col - half), min(image.shape[1], col + half + 1))
    block = np.clip(image[row_slice, col_slice].astype(float) - float(image[row_slice, col_slice].min()), 0, None)
    total = float(block.sum())
    if total <= 0:
        return float(col), float(row)
    yy, xx = np.mgrid[: block.shape[0], : block.shape[1]]
    return (float((xx * block).sum() / total) + col_slice.start,
            float((yy * block).sum() / total) + row_slice.start)


def measure_displacement(
    best: npt.NDArray[np.floating],
    carrier: npt.NDArray[np.floating],
    *,
    win: int = 60,
    min_positive_fraction: float = 0.02,
) -> dict | None:
    """量出载波把整形光斑移动到哪里。

    Args:
        best: 无载波的帧（最优相位）。
        carrier: 同相位 + 载波的帧。
        win: 局部质心窗口边长（相机像元）。
        min_positive_fraction: 正差分总量相对全帧总量的下限；低于此判定"没有可辨识的位移"
            （例如载波帧饱和、或载波把光斑移出视场），返回 ``None`` 而不是硬给一个方向。

    Returns:
        ``{"old_xy", "new_xy", "dx", "dy", "magnitude", "dominant_axis"}``（列序 x, y），
        或 ``None``。
    """
    if best.shape != carrier.shape:
        raise ValueError(f"两帧尺寸不一致: {best.shape} vs {carrier.shape}")
    diff = carrier.astype(float) - best.astype(float)
    positive = np.clip(diff, 0, None)
    negative = np.clip(-diff, 0, None)
    total = float(np.abs(best).sum()) + 1e-9
    if float(positive.sum()) / total < float(min_positive_fraction):
        return None

    old_x, old_y = _centroid_in_window(negative, win)
    new_x, new_y = _centroid_in_window(positive, win)
    dx, dy = new_x - old_x, new_y - old_y
    return {
        "old_xy": [round(old_x, 1), round(old_y, 1)],
        "new_xy": [round(new_x, 1), round(new_y, 1)],
        "dx": round(dx, 1),
        "dy": round(dy, 1),
        "magnitude": round(float(np.hypot(dx, dy)), 1),
        "dominant_axis": "x" if abs(dx) >= abs(dy) else "y",
    }


def shift_axis_label(displacement: dict) -> str:
    """把位移说成人话：'−y（面板 x 向斜坡 + SLM 竖向安装）' 这种形式只给方向，不复述安装理由。"""
    dx, dy = float(displacement["dx"]), float(displacement["dy"])
    if displacement["dominant_axis"] == "x":
        return f"{'+' if dx >= 0 else '−'}x 方向 {abs(dx):.1f} px（另一轴 {dy:+.1f} px）"
    return f"{'+' if dy >= 0 else '−'}y 方向 {abs(dy):.1f} px（另一轴 {dx:+.1f} px）"
