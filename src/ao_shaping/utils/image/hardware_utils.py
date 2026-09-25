"""Hardware helpers for SLM+CCD closed-loop runners.

Migrated from :mod:`ao_shaping.algorithm.beam_shaping_utils`; the old module
re-exports these names for backward compatibility.

Contains:
- ``capture_amplitude``: CCD 远场采集 → float32 振幅。
- ``call_with_timeout``: 硬件 SDK 调用 (可能挂起) 的看门狗超时。
- Auto-exposure helpers (纯目标计算 + 相机应用)。
- Frame recording (``frames/`` 目录 + PNG + JSONL 元数据)。

``utils`` is a leaf layer: hardware classes are only referenced under
``TYPE_CHECKING`` (or duck-typed via ``Any``), never imported at runtime.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from loguru import logger

from ao_shaping.utils.image.beam_metrics import median_zero_order_center
from ao_shaping.utils.image.spots_calc import centroid

if TYPE_CHECKING:  # pragma: no cover
    from ao_shaping.drivers.ccd import BaseCamera

__all__ = [
    "capture_amplitude",
    "init_frame_recording",
    "save_frame_png",
    "record_frame",
    "auto_exposure_target_ms",
    "auto_exposure_possible",
    "apply_auto_exposure",
    "find_exposure_ms",
    "find_zero_order_center",
    "log_center_brightness",
    "resolve_spot_center",
    "auto_find_exposure_and_center",
    "call_with_timeout",
    "open_camera",
]


def capture_amplitude(
    camera: "BaseCamera",
    center: Sequence[int] | None = None,
    size: Sequence[int] | None = None,
    n_sample: int = 1,
) -> np.ndarray:
    """从 CCD 采集远场强度并返回振幅。

    若提供了 ``center``/``size``, 先将相机窗口重置到该区域 (与目标图案
    的足迹匹配)。采集到的强度 (uint16) 转换为浮点振幅 ``sqrt(I)`` 并
    归一化到 ``[0, 1]``。

    Args:
        camera: 已打开的 CCD 相机对象 (须暴露 ``get_numpy_image``, 可选
            ``reset_window``)。
        center: 窗口中心 ``(cx, cy)``, 单位像素 (None 表示保持原样)。
        size: 窗口大小 ``(h, w)``, 单位像素 (None 表示保持原样)。
        n_sample: 平均的帧数。

    Returns:
        归一化到 ``[0, 1]`` 的 Float32 二维振幅数组。
    """
    if center is not None and size is not None:
        try:
            # 序列可能以列表形式传入; 驱动期望 2 元组。
            camera.reset_window(
                (int(center[0]), int(center[1])),
                (int(size[0]), int(size[1])),
            )
        except Exception:  # pragma: no cover - driver may not support
            logger.debug("Camera does not support reset_window; using full frame")

    img = camera.get_numpy_image(n_sample=n_sample, skip_first=True)
    intensity = np.asarray(img, dtype=np.float32)
    intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    amp = np.sqrt(intensity)
    amax = amp.max()
    if amax > 0:
        amp = amp / amax
    return amp.astype(np.float32)


# ---------------------------------------------------------------------------
# Frame recording (shared by diff_beam_runner and diff_shaping_runner)
# ---------------------------------------------------------------------------
_frames_dir: Path | None = None
_frame_counter: int = 0


def init_frame_recording(out_dir: Path) -> None:
    """在 ``out_dir`` 下创建 ``frames/`` 目录并重置计数器。

    Args:
        out_dir: 将包含 ``frames/`` 子目录的结果目录。
    """
    global _frames_dir, _frame_counter
    _frames_dir = Path(out_dir) / "frames"
    _frames_dir.mkdir(parents=True, exist_ok=True)
    _frame_counter = 0


def save_frame_png(frame: np.ndarray, path: Path, title: str) -> None:
    """将一帧 CCD 图像渲染为 PNG (inferno 色图 + 色条)。

    Args:
        frame: 待渲染的二维数组。
        path: 输出 PNG 路径。
        title: 图像标题。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(np.asarray(frame), cmap="inferno")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
    except Exception:  # pragma: no cover - plotting must never break the loop
        logger.warning("PNG 保存失败: {}", path.name)


def record_frame(
    raw: np.ndarray,
    phase_desc: str,
    exposure_ms: float,
    *,
    include_spot: bool = False,
) -> dict:
    """保存一帧原始 CCD 图像及一行 JSONL 元数据; 记录逐帧统计。

    同时返回元数据字典, 供调用方复用已记录的统计值。

    Args:
        raw: 原始 CCD 帧数组。
        phase_desc: 产生该帧的相位描述。
        exposure_ms: 曝光时间 (毫秒)。
        include_spot: 为 True 时, 额外将真实 0 级光斑位置 (``argmax``)
            记录为 ``meta["spot"]`` 并输出日志。在 2f 光路上强度质心并非
            光斑位置, 弥漫的杂散光晕会将其拉偏 150-450 px (2026-09-10
            实测)。

    Returns:
        该帧的元数据字典。
    """
    global _frame_counter
    _frame_counter += 1
    idx = _frame_counter
    frame = np.asarray(raw, dtype=np.float32)
    peak = float(frame.max()) if frame.size else 0.0
    total = float(frame.sum()) if frame.size else 0.0
    if frame.size:
        cx, cy = centroid(frame, return_float=True)
    else:
        cx = cy = 0.0
    meta = {
        "frame": idx,
        "phase": phase_desc,
        "exposure_ms": exposure_ms,
        "peak": peak,
        "sum": total,
    }
    if include_spot:
        if frame.size:
            sy, sx = np.unravel_index(np.argmax(frame), frame.shape)
        else:
            sy = sx = 0
        meta["spot"] = [int(sy), int(sx)]
    meta["centroid"] = [float(cy), float(cx)]
    meta["timestamp"] = datetime.now().isoformat(timespec="seconds")
    frames_dir = _frames_dir
    if frames_dir is not None:
        np.save(frames_dir / f"frame_{idx:05d}.npy", np.asarray(raw))
        save_frame_png(
            np.asarray(raw),
            frames_dir / f"frame_{idx:05d}.png",
            f"frame {idx:04d} - {phase_desc} (peak={peak:.0f})",
        )
        with open(frames_dir / "frame_meta.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    if include_spot:
        logger.info(
            "帧 {:04d}: phase={} peak={:.1f} sum={:.0f} spot=({}, {}) centroid=({:.1f}, {:.1f})",
            idx,
            phase_desc,
            peak,
            total,
            sy,
            sx,
            cy,
            cx,
        )
    else:
        logger.info(
            "帧 {:04d}: phase={} peak={:.1f} sum={:.0f} centroid=({:.1f}, {:.1f})",
            idx,
            phase_desc,
            peak,
            total,
            cy,
            cx,
        )
    return meta


# ---------------------------------------------------------------------------
# Auto-exposure helpers (shared by diff_beam_runner and diff_shaping_runner)
# ---------------------------------------------------------------------------
def auto_exposure_target_ms(
    current_ms: float,
    peak: float,
    target_brightness: float = 180.0,
    tol: float = 0.2,
    *,
    min_ms: float = 0.02,
    max_ms: float = 1000.0,
    sat_floor: float = 245.0,
    max_boost: float = 4.0,
    max_cut: float = 0.25,
) -> float:
    """计算将 ``peak`` 驱动到 ``target ± tol`` 的下一次曝光时间 (毫秒)。

    纯函数, 无硬件访问。调用方通过 ``camera.reset_exposure_time()`` 应用
    结果。

    Args:
        current_ms: 当前曝光时间 (毫秒)。
        peak: 实测峰值亮度。
        target_brightness: 目标峰值亮度 (默认 180)。
        tol: 相对容差带 (默认 0.2)。
        min_ms: 曝光时间下限。
        max_ms: 曝光时间上限。
        sat_floor: 8 位 CCD 的硬饱和保护阈值。
        max_boost: 最大提升倍数。
        max_cut: 最大削减倍数。

    Returns:
        建议的曝光时间 (毫秒)。
    """
    low, high = target_brightness * (1.0 - tol), target_brightness * (1.0 + tol)
    if low <= peak <= high:
        return float(current_ms)
    if peak >= sat_floor:
        scale = max(target_brightness / peak, 0.1)
    elif peak < low:
        scale = min(target_brightness / peak, max_boost)
    else:  # peak > high
        scale = max(target_brightness / peak, max_cut)
    return float(np.clip(current_ms * scale, min_ms, max_ms))


def auto_exposure_possible(camera: Any) -> bool:
    """若相机支持运行时曝光调整则返回 True。"""
    return callable(getattr(camera, "reset_exposure_time", None))


def apply_auto_exposure(
    camera: Any,
    peak: float,
    current_ms: float,
    target_brightness: float,
    tol: float,
) -> tuple[float, bool]:
    """将相机曝光调整到目标峰值区间。

    Args:
        camera: 具有 ``reset_exposure_time`` 的相机对象。
        peak: 当前实测峰值。
        current_ms: 当前曝光时间。
        target_brightness: 目标峰值亮度。
        tol: 容差带。

    Returns:
        ``(actual_exposure_ms, changed)`` 元组。
    """
    next_ms = auto_exposure_target_ms(current_ms, peak, target_brightness, tol)
    if abs(next_ms - current_ms) < 1e-9:
        return current_ms, False
    actual_ms = float(camera.reset_exposure_time(next_ms))
    logger.info(
        "自动曝光: peak={:.1f} -> 曝光 {:.3f} -> {:.3f} ms",
        peak,
        current_ms,
        actual_ms,
    )
    return actual_ms, True


# ---------------------------------------------------------------------------
# Auto-find exposure / 0-order centre (deterministic probes, shared by any
# runner that needs a safe fixed exposure + centred ROI before optimisation)
# ---------------------------------------------------------------------------
_DEFAULT_EXPOSURE_PROBES_MS: tuple[float, ...] = (
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
)


def _grab_frame(camera: Any, timeout_s: float = 10.0) -> np.ndarray:
    """带看门狗超时地抓取单帧图像。

    SLM/相机的 SDK 原生等待不受 Python 侧超时保护 (见 AGENTS.md 挂起记录),
    统一走 :func:`call_with_timeout`。

    Args:
        camera: 已打开的相机 (暴露 ``get_numpy_image``)。
        timeout_s: 看门狗超时 (秒)。

    Returns:
        二维强度数组 (float64)。
    """
    frame = call_with_timeout(
        lambda: camera.get_numpy_image(n_sample=1, skip_first=True),
        timeout_s,
        "相机抓帧",
    )
    return np.asarray(frame, dtype=np.float64)


def find_exposure_ms(
    camera: Any,
    target_peak: float = 160.0,
    probe_exposures_ms: Sequence[float] | None = None,
    peak_floor: float = 100.0,
    peak_ceiling: float = 245.0,
    timeout_s: float = 10.0,
) -> float:
    """用确定性升序探针找到安全的固定曝光时间 (毫秒)。

    与驱动 ``auto_exposure`` 比例收敛环不同, 本函数做少量固定探针扫描,
    返回**首个**落入 ``[peak_floor, peak_ceiling]`` 的探针曝光 —— 即满足
    亮度要求的最低曝光, 为整形调制保留余量 (硬件实测亮度在多次运行间有
    约 2x 漂移, 靠最低安全曝光 + 顶部分布留头防止饱和)。

    Args:
        camera: 已打开的相机 (须暴露 ``reset_exposure_time`` 与
            ``get_numpy_image``)。
        target_peak: 目标峰值亮度 (外推方向所用)。
        probe_exposures_ms: 升序探针曝光列表 (毫秒)。默认
            ``(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)``。
            默认列表**不含** 0.02ms —— 该曝光下常只有热像素可见
            (硬件实测 0.02ms 峰值 87 ≈ 纯噪声, 真实光斑仅 ~4), 会误导
            选择。
        peak_floor: 目标峰值下界 (含)。
        peak_ceiling: 目标峰值上界 (含)。
        timeout_s: 每帧抓取超时 (秒)。

    Returns:
        选定曝光时间 (毫秒)。相机保持在该曝光; 若相机不支持改曝光则
        抛出 ``ValueError``。

    Raises:
        ValueError: 相机不暴露 ``reset_exposure_time``; 或所有探针帧全暗。
    """
    if not callable(getattr(camera, "reset_exposure_time", None)):
        raise ValueError(
            "camera must expose reset_exposure_time(ms) for auto exposure finding"
        )
    probes = (
        list(probe_exposures_ms)
        if probe_exposures_ms
        else list(_DEFAULT_EXPOSURE_PROBES_MS)
    )
    if not probes:
        raise ValueError("probe_exposures_ms must be a non-empty sequence")
    probes = sorted(float(x) for x in probes)

    peaks: list[float] = []
    log_pairs: list[str] = []
    for ms in probes:
        actual_ms = float(camera.reset_exposure_time(ms))
        frame = _grab_frame(camera, timeout_s=timeout_s)
        peak = float(frame.max())
        peaks.append(peak)
        log_pairs.append(f"{actual_ms:.3f}ms->{peak:.0f}")
    logger.info("曝光探针: {}", " ".join(log_pairs))

    # 热像素守卫: 某探针峰值 > ceiling 而下一探针峰值 < floor 在物理上不
    # 可能 (真实光束曝光更高只会更亮) —— 视为热像素帧, 剔除该探针。
    guarded_peaks: list[tuple[float, float]] = []  # (exposure, peak)
    for i, (ms, peak) in enumerate(zip(probes, peaks)):
        if peak > peak_ceiling and i + 1 < len(peaks) and peaks[i + 1] < peak_floor:
            logger.warning("探针 {:.3f}ms 峰值 {:.0f} 判为热像素, 已剔除", ms, peak)
            continue
        guarded_peaks.append((ms, peak))
    if not guarded_peaks:
        raise ValueError("所有曝光探针均被判为热像素/无效帧")

    # 1) 首个落在目标区间的探针 -> 最低安全曝光。
    for ms, peak in guarded_peaks:
        if peak_floor <= peak <= peak_ceiling:
            logger.info("选定曝光: {:.3f} ms (peak={:.0f})", ms, peak)
            return float(ms)

    # 2) 从未发生过饱和的探针 (全部欠曝) -> 从最后探针线性外推。
    saturated = any(peak > peak_ceiling for _, peak in guarded_peaks)
    if not saturated:
        last_ms, last_peak = guarded_peaks[-1]
        if float(last_peak) <= 0.0:
            raise ValueError("所有曝光探针均全暗 (无光?) — 无法自动找曝光")
        est_ms = last_ms * (target_peak / float(last_peak))
        logger.warning(
            "所有探针欠曝, 从 {:.3f}ms(peak={:.0f}) 线性外推到 {:.3f} ms",
            last_ms,
            last_peak,
            est_ms,
        )
        apply_ms = min(est_ms, 1000.0)
        camera.reset_exposure_time(apply_ms)
        return float(apply_ms)

    # 3) 最低探针即饱和 -> 从最小探针线性下探 (该点最接近未饱和)。
    min_ms, min_peak = guarded_peaks[0]
    est_ms = min_ms * (target_peak / float(min_peak))
    apply_ms = max(min(est_ms, min_ms), 0.02)
    logger.warning(
        "最低探针 {:.3f}ms 已饱和(peak={:.0f}), 线性下探到 {:.3f} ms",
        min_ms,
        min_peak,
        apply_ms,
    )
    camera.reset_exposure_time(apply_ms)
    return float(apply_ms)


def find_zero_order_center(
    camera: Any,
    n_frames: int = 5,
    timeout_s: float = 10.0,
) -> tuple[int, int]:
    """在相机当前曝光下, 用多帧全局 argmax 中位数找 0 级光斑中心。

    2f Fourier 光路上 0 级 = 帧全局最大; 多帧中位数对偶发热像素帧鲁棒。
    相机须已处于可用曝光 (本函数不改曝光)。

    Args:
        camera: 已打开的相机 (暴露 ``get_numpy_image``)。
        n_frames: 采集帧数 (默认 5)。
        timeout_s: 每帧抓取超时 (秒)。

    Returns:
        ``(x, y)`` 像素坐标。
    """
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    frames = [_grab_frame(camera, timeout_s=timeout_s) for _ in range(n_frames)]
    center = median_zero_order_center(frames, refine=True)
    logger.info("0 级中心 ({} 帧中位数): ({}, {})", n_frames, center[0], center[1])
    return center


def auto_find_exposure_and_center(
    camera: Any,
    *,
    find_exposure: bool = True,
    find_center: bool = True,
    target_peak: float = 160.0,
    probe_exposures_ms: Sequence[float] | None = None,
    peak_floor: float = 100.0,
    peak_ceiling: float = 245.0,
    n_frames: int = 5,
    timeout_s: float = 10.0,
) -> tuple[float | None, tuple[int, int] | None]:
    """组合: 自动找曝光时间 + 0 级中心 (各自可独立开关)。

    先定曝光 (中心与曝光无关, 全局 argmax 跨曝光稳定), 再在选定曝光下
    定中心。

    Args:
        camera: 已打开的相机。
        find_exposure: 若为 False, 跳过曝光探测并保持相机当前曝光。
        find_center: 若为 False, 跳过中心探测。
        其余参数: 透传给 :func:`find_exposure_ms` /
            :func:`find_zero_order_center`。

    Returns:
        ``(exposure_ms, center)``; 被跳过的探测对应位置为 ``None``。
    """
    exposure: float | None = None
    center: tuple[int, int] | None = None
    if find_exposure:
        exposure = find_exposure_ms(
            camera,
            target_peak=target_peak,
            probe_exposures_ms=probe_exposures_ms,
            peak_floor=peak_floor,
            peak_ceiling=peak_ceiling,
            timeout_s=timeout_s,
        )
    if find_center:
        center = find_zero_order_center(camera, n_frames=n_frames, timeout_s=timeout_s)
    return exposure, center


def log_center_brightness(
    img: np.ndarray,
    center: tuple[int | float, int | float],
    exposure_ms: float,
) -> None:
    """Log the centroid-pixel brightness and frame peak at the current exposure.

    Shared log line format used by every wfless optimizer's initial centre
    acquisition so the hardware diagnostics agree across runners.

    Args:
        img: The captured image.
        center: ``(x, y)`` pixel coordinates.
        exposure_ms: Current camera exposure in ms.
    """
    _cy, _cx = int(center[1]), int(center[0])
    logger.info(
        "Centroid brightness: {}@{}, Max brightness: {:.0f} @ {:.3f}ms",
        float(img[_cy, _cx]),
        center,
        float(np.max(img)),
        exposure_ms,
    )


def resolve_spot_center(
    camera: Any,
    img: np.ndarray,
    center: Any = None,
    detect_fn: Callable[[np.ndarray], tuple[int | float, int | float]] | None = None,
    n_sample: int = 10,
    recapture: bool = True,
) -> tuple[int, int]:
    """Resolve a centre spec (``None`` / ``str`` / ``tuple``) into pixel coordinates.

    Consolidates the init-image centre resolution that was inlined in
    ``slm_zernike_pib.py``, ``pib.py``, ``combined_optimizer.py`` and
    ``slm_square_shaping.py``:

    * ``None`` — auto-detect via ``detect_fn`` on *this* frame.
    * ``"mass"`` / ``"max"`` / ``"shape"`` — re-capture a fresh frame (unless
      ``recapture`` is False) and dispatch by name:

      ``"mass"``  → full-image centroid (:func:`centroid`);
      ``"max"``   → global argmax;
      ``"shape"`` → :func:`smart_zero_order_center` (argmax-anchored, flat-core
      aware).

    * ``(x, y)`` tuple — returned as-is (operator-supplied fixed centre).

    Args:
        camera: An opened camera (used only for re-capture on ``str`` modes).
        img: The initial probe image (used for ``None`` / ``tuple`` modes).
        center: The centre spec (``None``, one of the mode strings, or a
            ``(x, y)`` tuple).
        detect_fn: Custom auto-detect function; defaults to
            :func:`smart_zero_order_center`.
        n_sample: Frame sample count for re-capture.
        recapture: When False, use ``img`` for ``str`` modes instead of
            re-capturing (useful when the caller knows the image is fresh).

    Returns:
        Integer ``(x, y)`` pixel coordinates.
    """
    from ao_shaping.utils.image.beam_metrics import smart_zero_order_center

    if detect_fn is None:
        detect_fn = smart_zero_order_center

    if center is None:
        center = detect_fn(img)
    elif isinstance(center, str):
        use_img = camera.get_numpy_image(n_sample) if recapture else img
        if center == "mass":
            center = centroid(use_img)
        elif center == "max":
            center = np.unravel_index(np.argmax(use_img), use_img.shape)[::-1]
        elif center == "shape":
            center = detect_fn(use_img)
        else:
            raise ValueError(f"known center: {center}")
    return (int(round(center[0])), int(round(center[1])))


# ---------------------------------------------------------------------------
# Hardware timeout helper
# ---------------------------------------------------------------------------
def call_with_timeout(fn: Any, timeout_s: float, desc: str) -> Any:
    """在守护线程中运行 ``fn`` 并施加看门狗超时。

    硬件 SDK 调用可能无限挂起; 本函数限制等待时间, 若调用未及时返回则
    抛出 ``TimeoutError``。

    Args:
        fn: 要运行的可调用对象。
        timeout_s: 超时时间 (秒)。
        desc: 用于错误消息的描述。

    Returns:
        ``fn`` 的返回值。

    Raises:
        TimeoutError: 当 ``fn`` 未在 ``timeout_s`` 内完成时。
    """
    result: list[Any] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(fn())
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"{desc} 超时 ({timeout_s}s)")
    if error:
        raise error[0]
    return result[0] if result else None


# ---------------------------------------------------------------------------
# Camera factory (shared by gs_square_runner and diff_shaping_runner)
# ---------------------------------------------------------------------------
def open_camera(
    camera_type: str,
    cam_id: int,
    exposure_ms: float,
    bit_depth: int = 8,
) -> Any:
    """按类型打开 CCD 相机并返回已连接的实例。

    相机驱动仅在函数内部延迟导入 (``utils`` 是叶子层, 不在模块顶层导入
    ``drivers``)。``daheng`` 使用 :class:`DahengCamera`, ``miicam``
    使用 :class:`MIICamera`; 驱动缺失或初始化失败时记录日志并
    重新抛出原异常。

    Args:
        camera_type: 相机类型, ``"daheng"`` 或 ``"miicam"``。
        cam_id: 相机设备 ID。
        exposure_ms: 曝光时间 (毫秒)。
        bit_depth: MiiCam 输出位深 (仅 ``miicam`` 使用, 默认 8)。

    Returns:
        已打开的相机实例。

    Raises:
        ValueError: ``camera_type`` 不是 ``"daheng"``/``"miicam"``。
        ImportError: 对应相机驱动不可用。
        Exception: 相机初始化失败。
    """
    if camera_type == "daheng":
        try:
            from ao_shaping.drivers.ccd.daheng import DahengCamera

            cam = DahengCamera(cam_id=cam_id, exposure_time_ms=exposure_ms)
            cam.open()
            return cam
        except ImportError as e:
            logger.warning("Daheng相机不可用: {}", e)
            raise
        except Exception as e:
            logger.error("Daheng相机初始化失败: {}", e)
            raise
    if camera_type == "miicam":
        try:
            from ao_shaping.drivers.ccd.miicam.driver import MIICamera

            cam = MIICamera(
                cam_id=cam_id,
                exposure_time_ms=exposure_ms,
                bit_depth=bit_depth,
            )
            cam.open()
            return cam
        except ImportError as e:
            logger.warning("MiiCam相机不可用: {}", e)
            raise
        except Exception as e:
            logger.error("MiiCam相机初始化失败: {}", e)
            raise
    raise ValueError(f"Unknown camera type: {camera_type}")
