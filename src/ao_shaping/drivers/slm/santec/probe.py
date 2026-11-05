"""Santec SLM 设备探针（公共层）.

把原来只在 Streamlit 多 SLM 控制器里使用的单设备探测下沉到这里，供 Qt 桌面页
等无 UI 的后台路径调用。探测只做 "打开 → 读序列号 → 关闭"，不改变设备状态。
"""

from __future__ import annotations

import ctypes
import threading
from typing import Any

from loguru import logger

SLM_PROBE_TIMEOUT_S = 3.0


def probe_slm(slm_num: int, timeout_s: float = SLM_PROBE_TIMEOUT_S) -> dict[str, Any] | None:
    """Probe a single SLM; return its device dict, or ``None`` when no device
    is attached (including the SDK being unavailable on a non-Windows host).
    An already-open device reports ``in_use=True``. Runs in a daemon thread
    with a timeout so a hung SDK call cannot block the caller (the
    Santec SDK has historically hung on ``SLM_Ctrl_ReadSD`` after rapid
    open/close cycles).
    """
    try:
        import ao_shaping.drivers.slm.santec._slm_win as slm_sdk  # Windows-only
    except (ImportError, OSError) as e:  # non-Windows host: degrade to "no device"
        logger.debug(f"Santec SLM SDK unavailable, skipping SLM probe: {e}")
        return None

    result: list[dict[str, Any] | None] = [None]

    def worker() -> None:
        try:
            ret = slm_sdk.SLM_Ctrl_Open(slm_num)
            if ret == 0:
                # Device is free — read its serial then release it.
                device_id = ctypes.create_string_buffer(256)
                ret2 = slm_sdk.SLM_Ctrl_ReadSD(slm_num, device_id)
                serial = device_id.value.decode("utf-8").strip() if ret2 == 0 else None
                result[0] = {
                    "slm_number": slm_num,
                    "serial": serial,
                    "connected": False,
                    "in_use": False,
                    "status": "未连接",
                }
            elif ret > 0:
                # Positive return codes (SLM_NG=1, SLM_IS_BUSY=2, …) mean the
                # SDK recognised a device at this index but could not open it
                # — almost always because another instance already holds it.
                # Negative codes (SLM_NOT_OPEN_USB=-200,
                # FT_DEVICE_NOT_FOUND=-10002, …) mean no device is attached.
                logger.debug(f"SLM #{slm_num} 已被占用 (返回码 {ret})")
                result[0] = {
                    "slm_number": slm_num,
                    "serial": None,
                    "connected": False,
                    "in_use": True,
                    "status": "使用中",
                }
            else:
                logger.debug(f"SLM #{slm_num} 无设备 (返回码 {ret})")
        except Exception:  # policy: ctypes SDK open can raise on an invalid handle; a probe that cannot open reports not-responding and still reaches the finally-close, so one dead controller cannot abort scanning the rest
            logger.debug(f"扫描 SLM #{slm_num} 时发生异常", exc_info=True)
        finally:
            try:
                slm_sdk.SLM_Ctrl_Close(slm_num)
            except Exception:  # policy: the handle may never have opened, so this close is best-effort cleanup and its failure carries no information
                pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        logger.warning(f"SLM #{slm_num} 扫描超时 ({timeout_s}s)，跳过（设备未响应或被占用）")
    return result[0]