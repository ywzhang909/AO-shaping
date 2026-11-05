"""迁移学习闭环页的「读取零级」：开一次相机 → 找 0 级 → 算出目标边长上限。

为什么需要它
------------
「目标边长」的物理上限只有一个依据：0 级光斑到**最近帧边**的距离。帧边之内
才是相机真正看得见的区域，目标方斑越大越接近帧边，一旦超出就是「目标比读出
窗口还大」—— 采样窗无论怎么设都放不下。所以上限不该写死成一个常数，而该在
真实光轴上量出来：:func:`~ao_shaping.utils.image.frame_analysis.max_target_side_at_zero_order`
已经把「到最近帧边的距离 - 留白」折算成边长，本模块只负责把它喂上真实一帧。

只读契约
--------
:func:`read_zero_order` 除了这次临时打开的相机的曝光，**不写任何持久状态、
不碰 SLM**：不开会话、不建目录、不下发相位。相机用完立刻关掉（context
manager），会话拿到的设备状态与点击之前一致。

不 import Qt / Streamlit：本模块是「中立层」，Qt 页把它交给后台线程调用
（与 :mod:`~ao_shaping.gui.slm.transfer_desktop.hardware_check` 同一分工）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from ao_shaping.utils.image.frame_analysis import max_target_side_at_zero_order
from ao_shaping.utils.image.hardware_utils import (
    call_with_timeout,
    find_exposure_ms,
    find_zero_order_center,
)

__all__ = ["ZeroOrderReading", "read_zero_order"]

#: 打开相机时的初始曝光 (ms)。只要能读出光斑即可 —— 紧随其后的
#: :func:`~ao_shaping.utils.image.hardware_utils.find_exposure_ms` 会把它调到
#: 峰值落在 [100, 245] 的那一档 (大恒 12-bit 上通常是 0.5 ms 量级)。
_INITIAL_EXPOSURE_MS = 0.05

#: 裸 SDK 调用的看门狗 (s)。打开相机 (GigE 枚举/连接) 与末帧抓取是仅有的两处
#: 未经 :func:`~ao_shaping.utils.image.hardware_utils.call_with_timeout` 保护的
#: 调用 —— 设备不可达时它们可无限挂起, 页面会停在「正在读取 0 级…」; 超时后
#: 走既有异常路径恢复界面 (见 :func:`read_zero_order` 内注释)。
_SDK_CALL_TIMEOUT_S = 20.0

#: 目标边长读数下限 (px)。比这还小，说明 0 级贴在帧边（或帧整体全暗），
#: 拿它当上限只会把表单锁死 —— 那是光路 / 曝光问题，要报出来让人查，
#: 不能静默写一个 0 进控件。
_MIN_SIDE_PX = 8

#: 全暗判据：峰值低于它就是没有光（读出噪声不会到 1 ADU）。
_DARK_PEAK = 1.0


@dataclass(frozen=True)
class ZeroOrderReading:
    """一次「读取零级」的结论（纯数据，可跨线程传递）。

    Attributes:
        center: 0 级中心 ``(x, y)``（相机 px，帧最亮点阈值）。
        max_side_px: 目标边长上限（px）—— 0 级到最近帧边、扣留白后折成的边长。
        exposure_ms: 本次使用的曝光时间（ms）。
        peak: 用于定量的一帧的峰值强度 (ADU)。
        frame_shape: 帧尺寸 ``(高, 宽)``。
    """

    center: tuple[int, int]
    max_side_px: int
    exposure_ms: float
    peak: float
    frame_shape: tuple[int, int]

    def summary(self) -> str:
        """一行中文结论（直接写进状态行）。"""
        height, width = self.frame_shape
        x, y = self.center
        return (
            f"0 级 ({x}, {y})；曝光 {self.exposure_ms:.2f} ms；"
            f"目标边长上限 {self.max_side_px} px（帧 {width}×{height}）"
        )


def _as_frame(image: object) -> np.ndarray:
    """相机读出 → 2D 浮点帧（单通道若带长度 1 的维，直接挤掉）。"""
    data = np.asarray(image, dtype=np.float64)
    if data.ndim == 3 and (data.shape[0] == 1 or data.shape[-1] == 1):
        data = data.squeeze()
    if data.ndim != 2:
        raise ValueError(f"相机返回的不是单通道 2D 帧（shape={data.shape}）")
    return data


def read_zero_order(camera: str, cam_id: int) -> ZeroOrderReading:
    """只读地量一次 0 级位置与目标边长上限。

    流程固定：准备 SDK → 以 0.05 ms 打开相机 → 升序探针定曝光 →
    多帧中位数定 0 级 → 再抓一帧算上限 → **关相机**。

    曝光与中心都由 ``utils.image`` 的现成实现给出，本函数不另写公式；
    相机生命周期交给 ``create_camera`` 的 context manager。

    Args:
        camera: 相机后端名（``params.CAMERA_CHOICES`` 之一，实测走 ``daheng`` / ``miicam``）。
        cam_id: 相机序号（大恒驱动按下标取设备，用 0）。

    Returns:
        本次读数。

    Raises:
        ValueError: 帧全暗（峰值 < 1）或边长读数 < 8 px —— 都是「光路 / 曝光
            不对」的操作性结论，页面会原样显示中文原因。
    """
    # 只在调用时 import 驱动: 模块级 import 会拉起 gxipy 等 SDK, 页面冷启动
    # 不该为一次可能的读数先付这笔代价 (见 drivers/AGENTS.md 的惰性加载契约)。
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.tools.slm.transfer.hardware import prepare_camera_sdk

    prepare_camera_sdk(camera)
    # 打开/关闭相机与末帧抓取都是 SDK 调用: 大恒是 GigE (IP 192.168.0.11),
    # 设备掉线时 open/读帧可无限挂起。find_exposure_ms / find_zero_order_center
    # 内部经 _grab_frame 已有 10 s/帧 保护, 这里补上仅剩的两处裸调用。
    # call_with_timeout 超时后抛 TimeoutError (子类 OSError→Exception),
    # 由 _ZeroOrderTask.run 的 except 捕获并送回 _on_zero_done 错误路径。
    cam_cm = create_camera(camera, cam_id, exposure_time_ms=_INITIAL_EXPOSURE_MS)
    cam = call_with_timeout(cam_cm.__enter__, _SDK_CALL_TIMEOUT_S, "打开相机")
    try:
        exposure_ms = float(find_exposure_ms(cam))
        center = find_zero_order_center(cam)
        frame = _as_frame(
            call_with_timeout(
                lambda: cam.get_numpy_image(n_sample=1, skip_first=False),
                _SDK_CALL_TIMEOUT_S,
                "抓取 0 级帧",
            )
        )
    finally:
        # 只有真正打开过才关; 打开超时 (设备不可达) 时 __enter__ 未成功, 不再关。
        if cam is not None:
            call_with_timeout(
                lambda: cam_cm.__exit__(None, None, None),
                _SDK_CALL_TIMEOUT_S,
                "关闭相机",
            )

    peak = float(np.max(frame)) if frame.size else 0.0
    if not np.isfinite(peak) or peak < _DARK_PEAK:
        raise ValueError(
            f"相机全暗（峰值 {peak:g}）：0 级没有进画面。"
            "检查光路 / 相机盖子 / 曝光上限后重试"
        )

    max_side_px = int(max_target_side_at_zero_order(frame, center))
    if max_side_px < _MIN_SIDE_PX:
        raise ValueError(
            f"0 级 ({center[0]}, {center[1]}) 到帧边太近，目标边长上限只有 "
            f"{max_side_px} px（< {_MIN_SIDE_PX} px）。"
            "把 0 级挪回画面中间，或换更大的读出窗口后重试"
        )
    height, width = int(frame.shape[0]), int(frame.shape[1])
    reading = ZeroOrderReading(
        center=(int(center[0]), int(center[1])),
        max_side_px=max_side_px,
        exposure_ms=exposure_ms,
        peak=peak,
        frame_shape=(height, width),
    )
    logger.info("读取零级完成: {}", reading.summary())
    return reading
