"""Explicit hardware-runtime setup; nothing is loaded at module import."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from loguru import logger


def prepare_camera_sdk(camera):
    """Expose the bundled Daheng binding only when a hardware action is requested."""
    if camera != "daheng":
        return
    import importlib.util
    import os
    from pathlib import Path
    import sys

    if importlib.util.find_spec("gxipy") is None:
        libs = Path(os.environ.get("AO_SHAPING_SDK_PATH", Path(__file__).resolve().parents[5] / "libs"))
        if not (libs / "gxipy/__init__.py").is_file():
            raise ImportError("gxipy unavailable; install the vendor binding or set AO_SHAPING_SDK_PATH to its parent directory")
        sys.path.insert(0, str(libs))


def enumerate_camera():
    """Read camera discovery without opening a stream or loading an SLM phase."""
    from .capture import clock

    prepare_camera_sdk("daheng")
    import gxipy as gx
    count, devices = gx.DeviceManager().update_device_list()
    return dict(camera="daheng", count=count, devices=devices, **clock())


class DeviceOpenError(RuntimeError):
    """设备硬件打开最终失败（重试耗尽）。

    Attributes:
        exit_code: 进 CLI/runner 的退出码, 固定 4。
        device: 未能打开的设备实例。
        attempts: 实际尝试次数。
    """

    exit_code = 4

    def __init__(self, message: str, device: Any, attempts: int):
        super().__init__(message)
        self.device = device
        self.attempts = attempts


# 配置/缺 SDK 类异常 —— 重试无意义, 立即抛出。
_NON_RETRYABLE = (ImportError, ValueError, TypeError)


@contextmanager
def opened_device(device, *, attempts: int = 3, delay_seconds: float = 0.5):
    """打开单个已构造但未打开的设备, 失败重试, 退出时关闭。

    只对**硬件打开失败**重试; ``ImportError``/``ValueError``/``TypeError`` 等配置
    或 SDK 缺失异常立即抛出, 不做无意义重试。驱动侧 ``CameraError``/``SantecError``
    都是 ``Exception`` 而非 ``RuntimeError``, 这里显式当作可重试硬件错误。

    Args:
        device: 已构造但未打开的设备实例 (实现 ``open()``/``close()``)。
        attempts: 上界总尝试次数 (含首次), 必须 >= 1。
        delay_seconds: 每次失败后到下次尝试前的间隔。

    Yields:
        打开成功的设备实例。

    Raises:
        ValueError: ``attempts`` 非法。
        DeviceOpenError: 硬件打开始终失败 (带 ``exit_code=4`` 等价信息)。
    """
    if not isinstance(attempts, int) or attempts < 1:
        raise ValueError("attempts must be a positive int")

    name = getattr(device, "__class__", None).__name__
    opened = False
    exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            device.open()
        except _NON_RETRYABLE:
            # 配置/SDK 缺失异常立即 re-raise, 保留其 cause, 不做无意义重试。
            raise
        except Exception as e:  # policy: 硬件打开失败重试, 耗尽后由调用方抛 DeviceOpenError
            opened = False
            exc = e
            logger.warning(
                f"打开设备 {name} 第 {attempt}/{attempts} 次失败: "
                f"{type(e).__name__}: {e}"
            )
            best_effort_close(device)
            if attempt < attempts:
                time.sleep(delay_seconds)
            continue
        opened = True
        break

    if not opened:
        # 所有尝试均失败
        raise DeviceOpenError(
            f"打开设备 {name} 失败: 已尝试 {attempts} 次仍失败",
            device,
            attempts,
        ) from exc

    try:
        yield device
    finally:
        best_effort_close(device)


def best_effort_close(device) -> None:
    """静默尝试关闭设备, 失败仅记录日志。"""
    close = getattr(device, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception as e:  # policy: 关闭失败仅记日志, 不掩盖原始异常
        logger.warning(f"关闭设备时出错: {type(e).__name__}: {e}")
