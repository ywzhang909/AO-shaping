"""Detector full-scale and saturation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger


#: Full-scale detector value, in the 0-255 gray units the whole repo uses.
#: Only consulted for **non-integer** dtypes (see :func:`full_scale`).
DETECTOR_FULL_SCALE = 255.0


def full_scale(img: np.ndarray, camera: Any | None = None) -> float:
    """Full-scale (saturation) value of a camera frame, in 0-255 gray units.

    Resolution order:

    1. Explicit annotation on the frame (``AveragedFrame.full_scale``) — kept as
       the top priority; an explicit annotation beats dtype (and beats the
       camera), because it is the saturation value written down at capture
       time.
    2. ``camera`` is given **and** can resolve an effective bit depth —
       ``camera.effective_full_scale(img)`` (duck-typed: camera shims without
       that method are skipped).
    3. Integer dtype's ``np.iinfo(...).max`` (uint8 -> 255, uint16 -> 65535).
    4. :data:`DETECTOR_FULL_SCALE` (255.0) for non-integer frames. Other float
       frames are read off the *same* detectors in the *same* units: the debug
       corpus holds uint8 / float32 / float64 frames whose per-frame maxima are
       all <= 255, so 255 - not 1.0 - is the ceiling. Treating a float frame as
       already normalised would compress it by up to 255x.

    Why dtype is only the *fallback* and never enough on its own: some backends'
    frame dtype does not match the effective bit depth — Daheng MONO12 frames
    are carried in a ``uint16`` container (true full scale 4095, not 65535),
    so ``np.iinfo(uint16).max`` silently mis-scales every saturation guard.
    dtype cannot distinguish "truly 16-bit" from "high bit depth packed into
    a uint16 container"; a camera's read-back bit depth must get the chance
    to speak (issue #246).

    Args:
        img: A camera frame.
        camera: An opened camera driver (or any object exposing
            ``effective_full_scale``); ``None`` keeps the dtype-based fallback.

    Returns:
        The value at or above which the frame counts as saturated.
    """
    if hasattr(img, "full_scale"):
        return float(img.full_scale)
    if camera is not None:
        resolve = getattr(camera, "effective_full_scale", None)
        if callable(resolve):
            return float(resolve(img))
    if np.issubdtype(img.dtype, np.integer):
        return float(np.iinfo(img.dtype).max)
    return DETECTOR_FULL_SCALE


#: 模块级别名: ``normalize_peak_to_gray`` 的参数 ``full_scale`` 会遮蔽同名函数,
#: 需要调用原函数时用这个别名。
_full_scale_of = full_scale


def normalize_peak_to_gray(img: np.ndarray, full_scale: float | None = None) -> float:
    """Peak of ``img`` expressed on the **0-255 grayscale** scale.

    The auto-exposure contract (``target_max`` / ``tolerance``) is defined in
    grayscale counts, but a frame's raw peak depends on the sensor's bit depth:
    8/12/16-bit frames cap at 255/4095/65535, and multi-frame averages come back
    as ``float32`` whose ceiling must be read off :class:`AveragedFrame`'s
    ``full_scale`` (never off ``img.dtype``). Without that rescaling a 16-bit
    frame always reads "peak >> target", so the proportional loop would drive the
    exposure straight to its lower bound.

    ``full_scale`` is 255 for 8-bit and for float frames already in gray units,
    so those are returned untouched — only wider integer formats are rescaled.

    Args:
        img: A camera frame.
        full_scale: 显式指定满量程 (相机计数的饱和值)。默认按
            :func:`full_scale` 推导; 当后端的帧 dtype 与生效位深不符时
            (大恒 MONO12 帧 dtype 是 ``uint16``, 真实满量程 4095 而非
            65535), 后端必须显式传入生效位深的满量程, 否则饱和帧会被
            折算成「过暗」, 自动曝光把曝光推到上限。

    Returns:
        The frame maximum on the 0-255 scale.
    """
    peak = float(np.max(img))
    scale = float(full_scale) if full_scale is not None else float(_full_scale_of(img))
    return peak * 255.0 / scale if scale > 255.0 else peak


def is_saturated(img: np.ndarray, camera: Any | None = None) -> bool:
    """True when ``img``'s peak reaches full scale (see :func:`full_scale`).

    Single source of truth for the saturation guard. Call sites used to hardcode
    ``255`` - and did not even agree with each other: the heuristic branches
    tested ``>= 255`` while the SPGD branches tested ``== 255`` (so a frame at
    250 was "saturated" to one and not the other).

    Pass ``camera`` when one is at hand: the effective bit depth (device
    read-back) then decides the ceiling instead of the frame dtype, which is
    wrong for backends that pack high bit depths into ``uint16`` containers
    (issue #246).
    """
    return float(np.max(img)) >= full_scale(img, camera)


def resample_on_saturation(
    img: np.ndarray,
    cam: Any,
    exposure_time_ms: float = 0.0,
    target_max_brightness: float = 0.0,
    auto_exposure_fn: Any | None = None,
    saturation_threshold: float | None = None,
    n_sample: int = 1,
) -> np.ndarray:
    """Re-capture at a lower exposure if the image is saturated.

    Mirrors the guard found in every SPGD loop: when auto-exposure is active
    (``exposure_time_ms == 0``) and the peak reaches ``saturation_threshold``,
    re-auto-expose to ``target_max_brightness`` (falling back to the original
    image data if auto-exposure is unavailable or the target is also 0).

    Args:
        img: The image to check (uint8 or float).
        cam: An opened camera driver.
        exposure_time_ms: Current fixed exposure (0 = auto mode, eligible for re-exposure).
        target_max_brightness: Auto-exposure target peak (0-255; 0 = use 220).
        auto_exposure_fn: Override for the auto-exposure implementation.
        saturation_threshold: Peak value considered saturated. ``None`` (the
            default) derives it from the frame via :func:`full_scale` **with
            ``cam``** (effective bit depth wins over dtype, per issue #246),
            which is correct for every bit depth; pass a number only to
            override (e.g. to trip early).
        n_sample: Frames averaged per re-capture.

    Returns:
        The (possibly resampled) image.
    """
    from ao_shaping.drivers.ccd.common import auto_exposure

    if exposure_time_ms > 0:
        return img
    if saturation_threshold is None:
        saturation_threshold = full_scale(img, cam)
    if float(np.max(img)) < saturation_threshold:
        return img
    target = target_max_brightness if target_max_brightness > 0 else 220.0
    fn = auto_exposure_fn if auto_exposure_fn is not None else auto_exposure
    logger.info(
        "image saturated (peak={:.0f}); auto-re-exposing to target {:.0f}",
        float(np.max(img)),
        target,
    )
    return np.asarray(fn(cam, target, n_sample=max(1, n_sample)))
