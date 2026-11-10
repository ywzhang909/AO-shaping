"""自动重用历史模型：在从头采集之前，先用少数几帧探测当前光路，
如果存在兼容性足够的历史检查点，就自动复用它，跳过采集与拟合。

为什么需要自动重用
------------------
``slm-transfer-shapes`` 默认 ``model_dir=None`` = 每次从头采集 + 拟合，约 20 min
设备时间。历史会话已经留下了带光路指纹的检查点，但复用需要人工指定 ``--model-dir``。

本模块在采集之前插入一个轻量探针：开一次相机读几帧，算出当前光路的指纹，
扫描历史检查点，按 :data:`~ao_shaping.tools.slm.transfer.optical_session.AUTO_APPLY_THRESHOLD`
自动选择兼容性最高的那个。分数不够就什么都不做，继续走从头采集路径。

探针只读 ``frames`` 帧，不做任何判定 —— 数据交给 :func:`auto_select_checkpoint`。
任何一步失败（相机打不开 / 读不到帧 / 扫描不到检查点）都只记日志，不抛异常：
自动重用是优化，不是硬依赖。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.tools.slm.transfer.optical_session import (
    AUTO_APPLY_THRESHOLD,
    compatibility_score,
    compatibility_verdict,
)
from ao_shaping.utils.io import paths

__all__ = [
    "auto_select_checkpoint",
    "probe_for_fingerprint",
]


def probe_for_fingerprint(
    *,
    camera: str,
    cam_id: int,
    exposure_ms: float = 5.0,
    frames: int = 3,
    label: str = "auto-reuse probe",
    on_frame=None,
) -> dict[str, Any]:
    """开一次相机，读少数几帧，返回可用于指纹比较的探针数据。

    只读 ``frames`` 帧取平均，计算零级位置 (亮度加权质心)、峰值计数、
    满量程与相机格式。不写任何文件，不做任何判定。

    Args:
        camera: ``daheng`` / ``sim`` 等相机类型。
        cam_id: 相机下标。
        exposure_ms: 探针曝光 (ms)。
        frames: 读几帧取平均 (默认 3)。
        label: 写进日志的标签。
        on_frame: 可选的逐帧回调 (与 ``frame_payload`` 配合可做实时预览)。

    Returns:
        ``{"zero_xy": [x, y], "peak_counts": float, "full_scale": float,
        "exposure_ms": float, "camera_format": str | None,
        "pixel_size_um": float | None, "optics_validity": str}``

        ``zero_xy`` 是传感器像素坐标；``full_scale`` 优先从相机位深推断
        (2^bit_depth − 1)，读不到时退回实测峰值；``optics_validity`` 为
        ``"ok"`` 表示探针成功读到画面，失败路径不返回。
    """
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.ccd.pixel_size import camera_pixel_size_um_for
    from ao_shaping.tools.slm.transfer.hardware import opened_device, prepare_camera_sdk
    from ao_shaping.tools.slm.transfer.live_frame import frame_payload, notify_frame
    from ao_shaping.tools.slm.transfer.optical_probe import read_camera_format

    prepare_camera_sdk(camera)
    with opened_device(
        create_camera(camera, cam_id, exposure_time_ms=exposure_ms)
    ) as cam:
        try:
            cam.set_exposure_ms(exposure_ms)
        except Exception as exc:  # policy: 相机不支持 set_exposure_ms 属常规机型差异，沿用默认曝光继续探测
            logger.debug(f"auto-reuse probe: set_exposure_ms 失败 (继续用默认): {exc!r}")

        reads: list[np.ndarray] = []
        for _ in range(max(int(frames), 1)):
            raw = np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False))
            notify_frame(
                on_frame, frame_payload(frame=raw, label=label, phase="capture")
            )
            reads.append(raw.astype(np.float64))

        frame = reads[0] if len(reads) == 1 else np.mean(reads, axis=0)

        # 零级位置：亮度加权质心 (忽略暗像素，避免噪声底把质心拉偏)
        threshold = max(float(np.percentile(frame, 90)), 1.0)
        mask = frame > threshold
        if mask.any():
            ys, xs = np.nonzero(mask)
            zero_x = float(np.mean(xs))
            zero_y = float(np.mean(ys))
        else:
            # 整幅只有噪声：质心放画面中心 (不代表光路位置，仅作为占位)
            zero_x = float(frame.shape[1] / 2)
            zero_y = float(frame.shape[0] / 2)

        peak = float(np.max(frame))

        # 满量程：优先从相机位深推断，读不到时退回实测峰值
        bit_depth = getattr(cam, "bit_depth", None)
        if isinstance(bit_depth, (int, float)) and bit_depth > 0:
            full_scale = float(2 ** int(bit_depth) - 1)
        else:
            full_scale = peak

        camera_format = read_camera_format(cam)
        pixel_size = camera_pixel_size_um_for(camera, cam_id, strict=False)

        return {
            "zero_xy": [zero_x, zero_y],
            "peak_counts": peak,
            "full_scale": full_scale,
            "exposure_ms": float(exposure_ms),
            "camera_format": camera_format,
            "pixel_size_um": pixel_size,
            "optics_validity": "ok",
        }


def auto_select_checkpoint(
    current: Mapping[str, Any],
    *,
    root: str | Path | None = None,
    min_score: float = AUTO_APPLY_THRESHOLD,
) -> dict[str, Any] | None:
    """扫描历史检查点，按兼容性评分排序，返回最佳候选。

    Args:
        current: 当前光路指纹 (由 :func:`probe_for_fingerprint` 与 session 参数
            合并而成，或直接由 ``compute_optical_fingerprint`` 产出)。
        root: 扫描根目录；默认 ``data/debug/slm_transfer``。
        min_score: 自动应用的下限 (默认 :data:`AUTO_APPLY_THRESHOLD`)。

    Returns:
        ``{"path": str, "score": float, "verdict": str}`` 或 ``None``。
        ``path`` 是检查点所在目录 (可直接传给 ``--model-dir``)；
        ``verdict`` 是 :func:`compatibility_verdict` 的中文结论。
        没有兼容候选 (无检查点 / 分数不足 / 全部 stale) 时返回 ``None``。
    """
    from ao_shaping.tools.slm.transfer.checkpoint_registry import (
        scan_checkpoint_index,
    )
    from ao_shaping.tools.slm.transfer.optical_session import (
        compute_optical_fingerprint,
    )

    base = (
        Path(root)
        if root is not None
        else paths.data_subdir("debug", "slm_transfer")
    )
    if not base.is_dir():
        logger.info("auto-reuse: 扫描根目录不存在，跳过")
        return None

    candidates = scan_checkpoint_index(base)
    if not candidates:
        logger.info("auto-reuse: 没有历史检查点，跳过")
        return None

    best: dict[str, Any] | None = None
    for entry in candidates:
        fingerprint = entry.get("fingerprint")
        if not isinstance(fingerprint, Mapping) or not fingerprint:
            # 索引里没有内联指纹 → 从会话目录现场算一份
            session = entry.get("session") or entry.get("path")
            if not session:
                continue
            fingerprint = compute_optical_fingerprint(session)
            if not fingerprint:
                continue
        if entry.get("stale"):
            continue
        score, _dims = compatibility_score(current, fingerprint)
        if best is None or score > best["score"]:
            best = {
                "path": str(entry.get("path", "")),
                "score": score,
                "verdict": compatibility_verdict(score),
            }

    if best is None:
        logger.info("auto-reuse: 所有候选均不可用，跳过")
        return None

    if best["score"] < min_score:
        logger.info(
            f"auto-reuse: 最佳候选 {best['path']} 分数 {best['score']:.3f} "
            f"< {min_score:.3f}，从头采集"
        )
        return None

    logger.info(
        f"auto-reuse: 选中 {best['path']} "
        f"(兼容分数 {best['score']:.3f} = {best['verdict']})"
    )
    return best
