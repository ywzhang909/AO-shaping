"""设备自检 / 0级探测 / 自动 CCD 类型选择。

本模块是**探测层**：给定已打开的相机 (与可选的 SLM), 做三件事, 全部
best-effort (任何失败返回 ``{"ok": False, "error": ...}``, 绝不冒泡):

1. :func:`detect_ccd_type` —— 自动 CCD 类型选择: 读相机格式/位深/型号/像元/
   分辨率, 组装成 ``CCD profile`` 字典, 供后续所有标度 (像元、位深、满量程)
   使用。相机不支持某项时该项填 ``None``, 不抛。
2. :func:`device_selfcheck` —— 设备自检: 传感器健康 (能读到非空帧) /
   光路完整性 (峰值 > 帧噪声底, 即有光) / 初始聚焦 (零级可定位)。
3. :func:`zero_order_probe` —— 0级探测: 在 SLM 平场下读若干帧, 建立零级
   基线强度 (峰值 / 均值 / 重心), 供后续整形段对照。

:func:`probe_environment` 把三者聚合成一个字典, 供 "加载 torch 环境 / 打开
设备" 后调用, 结果通过 :func:`live_frame.frame_payload` 的 ``selfcheck`` /
``zero_order`` / ``ccd_profile`` 字段展示。

Qt-free by construction: 本模块在测量线程被调用, 不碰任何 GUI 工具包。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from .hardware import prepare_camera_sdk  # noqa: F401 - 兼容性保留
from .light_guard import OpticalPathInvalid  # noqa: F401
from .optical_probe import read_bit_depth, read_camera_format

__all__ = [
    "detect_ccd_type",
    "device_selfcheck",
    "probe_environment",
    "zero_order_probe",
]


def _safe(call, default=None):
    """容错调用: 任何异常都吞掉, 返回 default。"""
    try:
        return call()
    except Exception as exc:  # policy: _safe 的契约就是吞掉单步失败并返回 default，探测绝不冒泡
        logger.warning("device probe step failed: {}", exc)
        return default


def detect_ccd_type(
    cam: Any,
    *,
    cam_id: int = 0,
    cam_type: str = "daheng",
) -> dict[str, Any]:
    """自动 CCD 类型选择: 读相机格式/位深/型号/像元/分辨率 → profile dict。

    任一项读不到时填 ``None``, 不抛。``ok`` 表示至少读到一帧 (传感器在线)。
    """
    profile: dict[str, Any] = {
        "cam_id": int(cam_id),
        "cam_type": str(cam_type),
        "ok": False,
    }

    fmt = _safe(lambda: read_camera_format(cam), None)
    profile["format"] = fmt

    bd = _safe(lambda: read_bit_depth(cam), None)
    profile["bit_depth"] = bd

    px = _safe(lambda: float(getattr(cam, "pixel_size_um", 0) or 0) or None, None)
    profile["pixel_size_um"] = px

    model = _safe(
        lambda: str(cam.cam.DeviceModelName.get())
        if hasattr(cam, "cam") and hasattr(cam.cam, "DeviceModelName")
        else None,
        None,
    )
    profile["model"] = model

    frame = _safe(lambda: np.asarray(cam.get_numpy_image(n_sample=1, skip_first=True)), None)
    if frame is not None and frame.size > 0:
        profile["shape"] = tuple(int(x) for x in frame.shape)
        profile["ok"] = True
    else:
        profile["shape"] = None

    # 组装 profile 名: 格式 + 位深 + 分辨率, 便于日志/展示。
    parts = [str(fmt or "unknown")]
    if bd:
        parts.append(f"{int(bd)}bit")
    if profile.get("shape"):
        parts.append(f"{profile['shape'][1]}x{profile['shape'][0]}")
    profile["profile"] = "_".join(parts)
    return profile


def _noise_floor(frame: np.ndarray) -> float:
    """稳健噪声底: median + 4 * 1.4826 * MAD (与 exposure_control 同约定)。"""
    flat = np.asarray(frame).ravel().astype(np.float64)
    if flat.size == 0:
        return 0.0
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med)))
    return med + 4.0 * 1.4826 * mad


def device_selfcheck(cam, *, n_frame: int = 3) -> dict[str, Any]:
    """设备自检: 传感器健康 / 光路完整性 / 初始聚焦。

    读 ``n_frame`` 帧, 判:
    * sensor_ok: 能读到非空帧, 且帧形状一致;
    * light_ok: 峰值 > 帧噪声底 (有光, 非全黑);
    * focus_ok: 零级 (全局 argmax) 可定位 (峰值有限且 > 噪声底);
    * ok: 三项都真。
    """
    result: dict[str, Any] = {
        "ok": False,
        "sensor_ok": False,
        "light_ok": False,
        "focus_ok": False,
        "frames": 0,
        "peak": 0.0,
        "noise_floor": 0.0,
        "centroid": None,
    }
    try:
        frames = []
        for _ in range(max(1, int(n_frame))):
            f = np.asarray(cam.get_numpy_image(n_sample=1, skip_first=True))
            if f.ndim == 3:
                f = f[..., 0]
            frames.append(f)
        result["frames"] = len(frames)
        if not frames:
            return result
        # 传感器健康: 非空 + 形状一致
        shapes = {tuple(f.shape) for f in frames}
        result["sensor_ok"] = all(f.size > 0 for f in frames) and len(shapes) == 1
        last = frames[-1]
        peak = float(np.nanmax(last)) if last.size else 0.0
        nf = _noise_floor(last)
        result["peak"] = peak
        result["noise_floor"] = nf
        result["light_ok"] = bool(peak > nf)
        # 初始聚焦: 零级可定位
        if last.size > 0 and np.isfinite(peak) and peak > nf:
            idx = np.unravel_index(int(np.argmax(last)), last.shape)
            result["centroid"] = (int(idx[1]), int(idx[0]))  # (x, y)
            result["focus_ok"] = True
        result["ok"] = bool(result["sensor_ok"] and result["light_ok"] and result["focus_ok"])
    except Exception as exc:  # policy: 自检聚合出口，SDK 失败写入 result["error"] 而非抛出
        logger.warning("device selfcheck failed: {}", exc)
        result["error"] = str(exc)
    return result


def zero_order_probe(
    cam: Any,
    slm: Any = None,
    *,
    n_frame: int = 3,
    settle_s: float = 0.6,
) -> dict[str, Any]:
    """0级探测: 在 SLM 平场下读若干帧, 建立零级基线强度。

    若提供 ``slm``, 先 ``set_phase_overlay(None)`` (平场) 并等待 ``settle_s``;
    否则假设当前 SLM 状态已是平场 (调用方保证)。

    返回: peak / mean / centroid / counts_per_ms (若能读曝光) / ok。
    """
    result: dict[str, Any] = {
        "ok": False,
        "peak": 0.0,
        "mean": 0.0,
        "centroid": None,
        "counts_per_ms": None,
        "frames": 0,
    }
    try:
        if slm is not None:
            try:
                slm.set_phase_overlay(None)
            except Exception as exc:  # policy: set_phase_overlay 是可选能力，缺失时按「无叠加」继续
                logger.warning("zero_order_probe: set_phase_overlay failed: {}", exc)
            time.sleep(float(settle_s))

        frames = []
        for _ in range(max(1, int(n_frame))):
            f = np.asarray(cam.get_numpy_image(n_sample=1, skip_first=True))
            if f.ndim == 3:
                f = f[..., 0]
            frames.append(f)
        result["frames"] = len(frames)
        if not frames:
            return result
        last = frames[-1]
        peak = float(np.nanmax(last)) if last.size else 0.0
        mean = float(np.nanmean(last)) if last.size else 0.0
        result["peak"] = peak
        result["mean"] = mean
        if last.size > 0 and np.isfinite(peak):
            idx = np.unravel_index(int(np.argmax(last)), last.shape)
            result["centroid"] = (int(idx[1]), int(idx[0]))  # (x, y)
        # counts_per_ms: 用峰值 / 曝光 (ms), 曝光读不到则 None
        try:
            exp_ms = float(cam.get_exposure_time_ms() if hasattr(cam, "get_exposure_time_ms") else 0.0)
        except Exception:  # policy: get_exposure_time_ms 可选，读不到则 counts_per_ms 记 None
            exp_ms = 0.0
        if exp_ms and exp_ms > 0:
            result["counts_per_ms"] = peak / exp_ms
        result["ok"] = bool(last.size > 0 and np.isfinite(peak))
    except Exception as exc:  # policy: 零级探针出口，任何失败写入 result 并保持返回结构
        logger.warning("zero order probe failed: {}", exc)
        result["error"] = str(exc)
    return result


def probe_environment(
    cam: Any,
    slm: Any = None,
    *,
    cam_id: int = 0,
    cam_type: str = "daheng",
    n_frame: int = 3,
) -> dict[str, Any]:
    """聚合三者: CCD 类型 + 设备自检 + 0级探测, 供 torch 环境加载后调用。"""
    return {
        "ccd_profile": detect_ccd_type(cam, cam_id=cam_id, cam_type=cam_type),
        "selfcheck": device_selfcheck(cam, n_frame=n_frame),
        "zero_order": zero_order_probe(cam, slm, n_frame=n_frame),
    }
