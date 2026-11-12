"""棋盘格 (checkerboard) 0 级残留测量 —— 灰度→相位 LUT 前置修正的采集原语 (#235)。

为什么单独成模块
----------------
棋盘格标定路径 (``scripts/slm/calib_slm_lut_checkerboard.py``) 与其单元尺寸诊断
(原 ``scripts/slm/diag_checker_cells_sweep.py``) 共用同一套图案生成 + 同曝光局部
比值测量。两者此前靠**按路径 importlib 加载对方**来复用 (诊断脚本要先把标定脚本
塞进 ``sys.modules`` 才能取到 ``_checker_gray``)，这条耦合既脆又让"单点真源"名不
副实。本模块把那部分与设备无关的原语提出来：

* :func:`checkerboard_gray` —— 交替 ``active_gray`` / ``0`` 的**灰度**图 (标定用)；
* :func:`checkerboard_phase` —— 同一图案的**弧度**孪生 (遮挡探针 / 刀口用)；
* :func:`scan_checker_retained` —— 灰度扫描循环：每个灰度点前重测一次平坦参考，
  用同曝光、同窗口的**局部**比值抵消激光漂移；
* :func:`window_power` / :func:`fwhm_geometric` —— 0 级窗口功率与窗口定径。

物理预测模型 (``retained = cos²(Δ/2)``) 与反演数学不在这里，它们属于
:data:`ao_shaping.utils.slm.slm_lut` (``invert_checkerboard_scan``) —— 本模块只负
责**测出** ``retained(g)``。

判据 / 注意
-----------
* **标定必须在 LUT 关闭状态下做**：图案走灰度直写，绕开驱动 LUT；若先 ``load_lut``
  会二次校正，测到的曲线是校正后残差，标定作废。
* 主动格子 2π 时 ``retained≈1``、π 时 ``retained≈0`` → 内部最小值存在，反演才成立。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

#: 与 #235 / 探针一致的窗口倍数: 6x 几何 FWHM 是它的 0 级参考窗口。
REFERENCE_WINDOW_FACTOR = 6.0
#: 每个图案重复取帧数 (平均以压噪声)。
DEFAULT_FRAMES = 4
#: 平坦 / 棋盘格 0 级窗口功率比的下界; 低于此说明光太弱, 反演不可信。
MIN_FLAT_POWER = 100.0
#: 棋盘格单元默认边长 (面板 px); 需让棋盘一级衍射落出 0 级窗口。
DEFAULT_CELLS = 24


def window_power(frame: np.ndarray, cy: int, cx: int, side: int) -> float:
    """以 ``(cy, cx)`` 为中心、边长 ``side`` 的方窗内总功率 (越界裁剪)。"""
    half = side // 2
    y0, y1 = max(0, cy - half), min(frame.shape[0], cy + half + 1)
    x0, x1 = max(0, cx - half), min(frame.shape[1], cx + half + 1)
    return float(frame[y0:y1, x0:x1].sum())


def fwhm_geometric(frame: np.ndarray, cy: int, cx: int) -> float:
    """亮斑 FWHM 的几何平均 (行/列剖面的算术平均), 用于定窗口大小。"""

    def one(profile: np.ndarray) -> float:
        peak = float(profile.max())
        if peak <= 0:
            return 0.0
        above = np.flatnonzero(profile >= 0.5 * peak)
        return float(above[-1] - above[0] + 1) if above.size else 0.0

    return 0.5 * (one(frame[cy, :]) + one(frame[:, cx]))


def window_side_for(frame: np.ndarray, cy: int, cx: int) -> int:
    """按 ``REFERENCE_WINDOW_FACTOR`` x 几何 FWHM 定 0 级参考窗口边长。"""
    fwhm = fwhm_geometric(frame, cy, cx)
    return int(round(REFERENCE_WINDOW_FACTOR * max(fwhm, 5.0)))


def checkerboard_gray(
    cells: int, active_gray: int, panel_shape: tuple[int, int]
) -> np.ndarray:
    """棋盘格**灰度图** (uint16): 交替单元取 ``active_gray`` / ``0``。

    直接产灰度 (而非弧度), 绕开 ``phase_to_gray`` 的 ``mod 2π`` —— 标定要测的就是
    "主动格子放在哪个灰度 → 实际相位延迟多少", 灰度是唯一自变量。

    Args:
        cells: 单元边长 (面板 px)。
        active_gray: 主动格子的灰度值。
        panel_shape: ``(height, width)`` 面板分辨率。

    Returns:
        uint16 数组, 形状 ``panel_shape``。
    """
    yy, xx = np.mgrid[0 : panel_shape[0], 0 : panel_shape[1]]
    on = ((xx // cells + yy // cells) % 2).astype(np.uint16)
    return (on * int(active_gray)).astype(np.uint16)


def checkerboard_phase(
    cells: int, depth_rad: float, panel_shape: tuple[int, int]
) -> np.ndarray:
    """:func:`checkerboard_gray` 的**弧度**孪生: 交替单元取 ``depth_rad`` / ``0``。

    给需要走 ``phase_to_gray`` 正常显示链的调用方 (遮挡深度探针 / 刀口) 用; 标定
    本身要用灰度直写版, 因为灰度才是自变量。

    Args:
        cells: 单元边长 (面板 px)。
        depth_rad: 主动格子的相位深度 (rad)。
        panel_shape: ``(height, width)`` 面板分辨率。

    Returns:
        float32 数组, 形状 ``panel_shape``, 单位弧度。
    """
    yy, xx = np.mgrid[0 : panel_shape[0], 0 : panel_shape[1]]
    return (((xx // cells + yy // cells) % 2) * depth_rad).astype(np.float32)


def gray_scan(gray_step: int, max_gray: int) -> np.ndarray:
    """灰度扫描格点: ``0, step, 2*step, ...`` 并保证 ``max_gray`` 本身在列。"""
    g = np.arange(0, max_gray + 1, gray_step, dtype=int)
    if g[-1] != max_gray:
        g = np.append(g, max_gray)
    return g


def capture_mean(cam: Any, frames: int) -> np.ndarray:
    """取 ``frames`` 帧的平均 (原始帧 -> float64)。与探针同一相机取帧入口。"""
    stack = [
        np.asarray(cam.get_numpy_image(1, True), dtype=np.float64)
        for _ in range(frames)
    ]
    return np.mean(np.stack(stack, axis=0), axis=0)


@dataclass
class CheckerScan:
    """一次棋盘格灰度扫描的结果 (灰度点等长)。

    Attributes:
        gray: 扫描的灰度格点。
        retained: 每个灰度点的 0 级残留比 ``p_checker / p_flat``。
        flat_power: 每个灰度点**前**重测的平坦窗口功率 (同曝光局部比值)。
        rows: 逐点明细 (gray / p_flat / p_checker / retained), 供 summary.json。
        valley_index: ``retained`` 最小值的下标 (π 交叉 / 谷值点)。
        valley_frame: 谷值点的棋盘格**原始**帧 (未扣背景), 供配图与复核。
    """

    gray: np.ndarray
    retained: np.ndarray
    flat_power: np.ndarray
    rows: list[dict[str, Any]] = field(default_factory=list)
    valley_index: int = 0
    valley_frame: np.ndarray | None = None

    @property
    def valley_gray(self) -> int:
        """谷值 (π 交叉) 所在灰度。"""
        return int(self.gray[self.valley_index])

    @property
    def retained_min(self) -> float:
        """谷值残留比 —— 判"棋盘格能不能挡光"的唯一数字。"""
        return float(self.retained[self.valley_index])


def scan_checker_retained(
    slm: Any,
    cam: Any,
    *,
    cells: int,
    gray_step: int,
    max_gray: int,
    frames: int,
    window_side: int,
    center: tuple[int, int],
    display: Callable[[np.ndarray], None] | None = None,
    background: Callable[[np.ndarray], np.ndarray] | None = None,
    save_frames: Callable[[int, np.ndarray], None] | None = None,
    progress_every: int = 8,
) -> CheckerScan:
    """扫灰度 ``0..max_gray``，测棋盘格的 0 级残留比 ``retained(g)``。

    逐点次序与旧标定脚本逐字一致 (这是**行为契约**，报告产物依赖它):

    1. 显示全 0 灰度，取**一**帧，重测平坦窗口功率 ``p_flat`` (同曝光、同窗口)。
    2. 显示 ``checkerboard_gray(cells, g)``，取 ``frames`` 帧平均，落盘原始帧，
       扣背景后测窗口功率 ``p_checker``。
    3. ``retained[i] = p_checker / p_flat``；``p_flat <= 0`` 直接报错退出。

    每点前重测平坦是**局部比值**：激光漂移同时进分子分母，于是抵消。全局归一做不
    到这一点。

    Args:
        slm: 已 open 的 SLM 驱动 (取面板分辨率与 2π 灰度需要它)。
        cam: 已 open 的相机驱动。
        cells: 棋盘格单元边长 (面板 px)。
        gray_step: 灰度步长。
        max_gray: 灰度上限 (设备 2π 灰度, 或显式覆盖)。
        frames: 每个棋盘格点的平均帧数 (平坦参考恒为 1 帧)。
        window_side: 0 级窗口边长 (px)，由 :func:`window_side_for` 依平坦帧定。
        center: 0 级中心 ``(cy, cx)``。
        display: ``gray_img -> None``，上屏回调；缺省用 ``SlotRotator`` + 0.3s 稳定。
        background: ``raw -> work``，背景扣除回调；缺省减中位数。
        save_frames: ``(gray, raw_frame) -> None``，逐点落盘回调；None 则不落盘。
        progress_every: 每几点打印一次 (另恒打印最后一点)。

    Returns:
        :class:`CheckerScan`。

    Raises:
        ValueError: ``slm`` 无面板分辨率 (未 open?)。
        RuntimeError: 某个灰度点的平坦窗口功率 <= 0 (光断 / 曝光错档)。
    """
    panel_shape = _panel_shape(slm)
    cy, cx = int(center[0]), int(center[1])
    if display is None:
        display = _default_display(slm, settle_s=0.3)
    if background is None:
        background = _median_background

    grays = gray_scan(gray_step, max_gray)
    retained = np.zeros(grays.size, dtype=float)
    flat_series = np.zeros(grays.size, dtype=float)
    rows: list[dict[str, Any]] = []
    valley_frame: np.ndarray | None = None

    flat_gray = np.zeros(panel_shape, dtype=np.uint16)
    for i, g in enumerate(grays):
        display(flat_gray)
        f_work = background(capture_mean(cam, 1))
        p_flat = window_power(f_work, cy, cx, window_side)

        display(checkerboard_gray(cells, int(g), panel_shape))
        c_raw = capture_mean(cam, frames)
        if save_frames is not None:
            save_frames(int(g), c_raw)
        p_check = window_power(background(c_raw), cy, cx, window_side)

        if p_flat <= 0:
            raise RuntimeError(f"flat power zero at g={int(g)}")
        retained[i] = p_check / p_flat
        flat_series[i] = p_flat
        rows.append(
            {
                "gray": int(g),
                "p_flat": float(p_flat),
                "p_checker": float(p_check),
                "retained": float(retained[i]),
            }
        )
        if valley_frame is None or retained[i] == float(np.min(retained[: i + 1])):
            valley_frame = c_raw
        if i % progress_every == 0 or i == grays.size - 1:
            logger.info("g={} retained={:.4f}", int(g), retained[i])

    scan = CheckerScan(
        gray=grays,
        retained=retained,
        flat_power=flat_series,
        rows=rows,
        valley_index=int(np.argmin(retained)),
        valley_frame=valley_frame,
    )
    logger.info(
        "checker scan cells={} retained_min={:.4f} @g={}",
        cells,
        scan.retained_min,
        scan.valley_gray,
    )
    return scan


def _panel_shape(slm: Any) -> tuple[int, int]:
    """面板 ``(height, width)``；``Panel_Res`` 是 ``(width, height)``，故交换。"""
    res = getattr(slm, "Panel_Res", None)
    if res is None:
        raise ValueError("slm has no Panel_Res (open it first)")
    return int(res[1]), int(res[0])


def _median_background(raw: np.ndarray) -> np.ndarray:
    """缺省背景扣除: 减中位数 (无暗场时的稳健估计)。"""
    return raw - float(np.median(raw))


def _default_display(slm: Any, *, settle_s: float) -> Callable[[np.ndarray], None]:
    """用 ``SlotRotator`` 轮播槽位上屏灰度图 (与标定脚本同一显示通道)。"""
    from ao_shaping.drivers.slm.santec import SlotRotator

    rotator = SlotRotator(slm)

    def display(gray_img: np.ndarray) -> None:
        rotator.display_data(slm, gray_img, float(settle_s))

    return display
