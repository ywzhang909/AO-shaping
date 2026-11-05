"""光路复用探针 (optical-path reuse probe)。

这是一个**便宜探针**, 不是新的优化器: 在 ``reuse_latest`` 开启时、重建采集与
拟合之前, 打开相机 + SLM, 量三个「当前光路条件」:

1. 零点位置 ``zero_xy`` (0 级 = 帧全局最大, 用 :func:`capture.locate`);
2. 相机的可用曝光区间 ``exposure_range_ms``;
3. 相机的取图格式 (``camera_format``, 如 ``"MONO8"``, 及位深 ``bit_depth``)。

然后与保存的 ``latest`` 模型 (``data/slm_models/<serial>/latest``) 里记录的
同一批次条件做比对 (:func:`same_path`)。一致 → 光路可复用 → workflow 跳过
采集+拟合, 直接复用 latest 冻结参数跑 CCD 锚定块; 不一致 → 走正常全流程。

判定是**纯函数**, 任何一步读不到/解不开都只返回不一致 (或 ``None``), 绝不抛
异常。本模块顶层**不 import 任何设备/SDK**; 设备在 :func:`measure_probe` 内部
惰性打开。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


#: 零点漂移容忍 (像素)。探针复用是「便宜比对」, 几像素内的零点平移认为光路没变。
DEFAULT_TOL_ZERO_PX = 5.0
#: 曝光区间上界放宽倍数 (下界通常触及传感器最小档, 不稳定, 不参与判比)。
#: 曝光是试出来的自适应结果, 允许 1.5 倍以内的浮动。
DEFAULT_TOL_EXPOSURE_FACTOR = 1.5
#: latest 目录里记录探针条件/格式的文件名。
FLAT_FROZEN_SIDECAR = "_optical_probe.json"


@dataclass(frozen=True)
class OpticalProbe:
    """一次光路探针的量测结果。"""

    #: 0 级光斑的 (x, y) 像素坐标 (帧内全局最大背后的局部矩, 同 capture.locate)。
    zero_xy: tuple[float, float]
    #: 相机曝光区间 ``(min_ms, max_ms)``。
    exposure_range_ms: tuple[float, float]
    #: 相机取图格式名 (如 ``"MONO8"``); 读不到为 None。
    camera_format: str | None = None
    #: 像素位深 (8/12/16); 读不到为 None。
    bit_depth: int | None = None
    #: 本次探针用的 SLM 序列号 (与 capture.py 同一口径: 打开后读 ``serial_number``)。
    #: 读不到为 None —— 调用方据此认为「没有可比的 latest 模型」。
    slm_serial: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_probe(
    *,
    camera: str = "daheng",
    cam_id: int = 0,
    slm_number: int = 1,
) -> OpticalProbe:
    """打开相机 + SLM, 取一帧平场, 量一次光路探针。

    Args:
        camera: 相机后端名 (传给 ``create_camera``)。
        cam_id: 相机 id。
        slm_number: SLM 编号。

    Returns:
        :class:`OpticalProbe`。

    Raises:
        DeviceOpenError / ImportError: 设备打开失败或 SDK 缺失时抛出 (硬件层语义)。
    """
    from .hardware import opened_device, prepare_camera_sdk
    from ao_shaping.drivers.ccd.common import (
        create_camera,
        get_camera_exposure_range,
    )
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    prepare_camera_sdk(camera)
    cam = create_camera(camera, cam_id, exposure_time_ms=0.02)

    # 全函数最长的设备占用: 相机 + SLM, 退出即释放。探针只读一帧、不下发图案,
    # 但为与采集段一致 (零点装在「平场面」上) 还是保持零相位覆盖层。
    with opened_device(cam) as cam_handle:
        with opened_device(Santec(slm_number=slm_number, wavelength=1064, video_mode=0)) as slm:
            # 序列号: 打开后才有 (与 capture.py 同一口径); 读不到保持 None,
            # 不报错 (仿真面板没有该属性)。
            slm_serial = None
            try:
                slm_serial = getattr(slm, "serial_number", None)
            except Exception as exc:  # pragma: no cover - 设备属性读取失败
                logger.warning(f"光路探针: 无法读取 SLM 序列号: {exc!r}")
            slm.set_phase_overlay(None)
            slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
            floor, ceiling = get_camera_exposure_range(cam_handle)
            limit = configured_daheng_exposure_limit() if camera == "daheng" else 0
            if limit > 0:
                ceiling = min(ceiling, limit)
            raw = cam_handle.get_numpy_image(n_sample=1, skip_first=False)
            frame = np.asarray(raw, dtype=np.float64)
            zero = locate_zero(frame)
            exposure_range = (floor, ceiling)
            camera_format = read_camera_format(cam_handle)
            bit_depth = read_bit_depth(cam_handle)
            # 复位平场, 与采集段 finally 一致
            slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
    probe = OpticalProbe(
        zero_xy=zero,
        exposure_range_ms=exposure_range,
        camera_format=camera_format,
        bit_depth=bit_depth,
        slm_serial=None if slm_serial is None else str(slm_serial),
    )
    logger.info("光路探针: zero={} exposure={} format={} bit={}", probe.zero_xy,
                probe.exposure_range_ms, probe.camera_format, probe.bit_depth)
    return probe


def locate_zero(frame: Any) -> tuple[float, float]:
    """定位 0 级光斑: 帧全局最大 + 局部矩 (同 capture.locate 口径)。

    纯函数, 不打开设备; 供 :func:`measure_probe` 与纯测试直接使用。
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    smooth = gaussian_filter(np.asarray(frame, dtype=float), 3)
    row, col = np.unravel_index(smooth.argmax(), smooth.shape)
    y0, y1 = max(0, row - 30), min(frame.shape[0], row + 31)
    x0, x1 = max(0, col - 30), min(frame.shape[1], col + 31)
    weights = np.maximum(np.asarray(frame, dtype=float)[y0:y1, x0:x1]
                         - np.median(np.asarray(frame, dtype=float)), 0)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    if float(weights.sum()) <= 0:
        return 0.0, 0.0
    return float((xx * weights).sum() / weights.sum()), float(
        (yy * weights).sum() / weights.sum()
    )


def read_camera_format(cam: Any) -> str | None:
    """读相机当前取图格式名 (如 ``"MONO8"``); 读不到返回 None。"""
    try:
        pf = cam.cam.PixelFormat
        return str(pf.get())
    except Exception as exc:  # 格式读不到不是光路判定错误
        logger.warning(f"无法读相机格式: {exc!r}")
        return None


def read_bit_depth(cam: Any) -> int | None:
    """读相机位深 (优先驱动缓存 ``bit_depth``, 再尝试 gxipy PixelSize)。"""
    bd = getattr(cam, "bit_depth", None)
    if isinstance(bd, (int, float)) and bd > 0:
        return int(bd)
    try:
        pixel_size = cam.cam.PixelSize.get()
        if pixel_size is not None:
            # GenICam PixelSize 即位深 (8/12/16); 驱动缓存缺失时才回退到这里。
            depth = int(pixel_size)
            if depth > 0:
                return depth
    except Exception as exc:  # pragma: no cover - 设备回退路径
        logger.warning(f"无法读相机位深: {exc!r}")
    return None


def _frozen_camera_format(frozen: dict[str, Any]) -> str | None:
    """从冻结参数里取相机格式 (latest 目录 / 冻结模型可能存 ``camera_format``)。"""
    value = frozen.get("camera_format")
    if value is None and frozen.get("probe", {}):
        value = frozen["probe"].get("camera_format")
    return str(value) if value else None


def probe_from_latest(latest_dir: Path | str) -> OpticalProbe | None:
    """从 ``latest`` 目录重建上一次会话记录的探针条件。

    读 ``frozen_parameters.json`` 的 ``origin`` (零点), 并从 ``_optical_probe.json``
    侧车读曝光区间 / 取图格式 / 位深 (侧车缺失时退回 ``provenance.json``)。任何一步
    读不到/解不开 → 返回 None (表示没有可比的参照, 调用方按「不一致」回退)。

    Args:
        latest_dir: latest 目录 (含 ``frozen_parameters.json``)。

    Returns:
        该会话的 :class:`OpticalProbe`, 或 None。
    """
    latest_dir = Path(latest_dir)
    frozen_path = latest_dir / "frozen_parameters.json"
    if not frozen_path.is_file():
        return None
    try:
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(f"latest 探针对照读不出 frozen_parameters.json: {exc!r}")
        return None
    origin = frozen.get("origin")
    if not isinstance(origin, (list, tuple)) or len(origin) != 2:
        logger.warning("latest frozen_parameters.json 缺有效 origin, 无零点头")
        return None
    try:
        zero = (float(origin[0]), float(origin[1]))
    except (TypeError, ValueError):
        return None

    # 侧车 (``_optical_probe.json``) 是完整的参照: 曝光区间 + 取图格式 + 位深。
    # 侧车缺失时退回冻结参数里可能带的字段, 再缺就留 None —— 缺项在
    # :func:`same_path` 里视作「不可比」, 于是走正常全流程。
    sidecar = _read_sidecar(latest_dir)
    camera_format = sidecar.get("camera_format") or _frozen_camera_format(frozen)
    bit_depth = _as_int(sidecar.get("bit_depth")) or _as_int(frozen.get("bit_depth"))
    exposure_range = _as_range(sidecar.get("exposure_range_ms"))
    if exposure_range is None:
        exposure_range = _read_exposure_range(latest_dir)
    return OpticalProbe(
        zero_xy=zero,
        exposure_range_ms=exposure_range,
        camera_format=camera_format,
        bit_depth=bit_depth,
    )


def _read_sidecar(latest_dir: Path) -> dict[str, Any]:
    """读探针参照侧车; 读不到或坏了就返回空字典。"""
    sidecar = Path(latest_dir) / FLAT_FROZEN_SIDECAR
    if not sidecar.is_file():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(f"latest 探针侧车读不出: {exc!r}")
        return {}
    return data if isinstance(data, dict) else {}


def _as_int(value: Any) -> int | None:
    """把可转的数转成 int; 不可转返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_range(value: Any) -> tuple[float, float] | None:
    """把 ``[lo, hi]`` 形状的值转成 ``(float, float)``; 不可用返回 None。"""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _read_exposure_range(latest_dir: Path) -> tuple[float, float]:
    """从 latest 目录取曝光区间; 无可读来源时回落到默认传感器区间。"""
    for candidate in ("provenance.json", "result.json"):
        path = Path(latest_dir) / candidate
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        rng = _as_range(data.get("exposure_range_ms"))
        if rng is None:
            rng = _as_range((data.get("capture") or {}).get("exposure_range"))
        if rng is not None:
            return rng
    # 无可读曝光区间 → 用默认传感器区间 (与 get_camera_exposure_range 兜底一致)。
    return (0.011, 10_000.0)


def same_path(
    current: OpticalProbe,
    reference: OpticalProbe | None,
    *,
    tol_zero_px: float = DEFAULT_TOL_ZERO_PX,
    tol_exposure_factor: float = DEFAULT_TOL_EXPOSURE_FACTOR,
) -> bool:
    """纯函数判比: 当前探针 vs latest 参照是否同一条光路。

    任何一步配不上 (含 ``reference=None`` 或缺曝光/格式) → 返回 False, **绝不
    抛异常**。判据:

    - ``zero_xy`` 各轴漂移 <= ``tol_zero_px``;
    - 曝光上界在当前区间 ``[参考/因子, 参考*因子]`` 内 (上界稳; 下界不稳不判);
    - ``camera_format`` 完全相等 (None 之一视作不可比 → False);
    - ``bit_depth`` 完全相等 (None 之一视作不可比 → False)。
    """
    if reference is None:
        return False
    # 零点
    try:
        dz = (abs(current.zero_xy[0] - reference.zero_xy[0]),
              abs(current.zero_xy[1] - reference.zero_xy[1]))
    except (TypeError, IndexError):
        return False
    if dz[0] > tol_zero_px or dz[1] > tol_zero_px:
        return False
    # 曝光上界 (下界通常触及传感器最小档, 不稳定, 不参与判比)
    ref_hi = reference.exposure_range_ms[1] if reference.exposure_range_ms else 0.0
    cur_hi = current.exposure_range_ms[1] if current.exposure_range_ms else 0.0
    if ref_hi > 0 and not (ref_hi / tol_exposure_factor <= cur_hi <= ref_hi * tol_exposure_factor):
        return False
    # 格式与位深: 都必须非空且完全一致
    if (current.camera_format is None or reference.camera_format is None
            or current.camera_format != reference.camera_format):
        return False
    if (current.bit_depth is None or reference.bit_depth is None
            or current.bit_depth != reference.bit_depth):
        return False
    return True


def format_probe_verdict(
    verdict: bool,
    *,
    current: OpticalProbe | None = None,
    reference: OpticalProbe | None = None,
) -> str:
    """把光路复用判决格式化成一行人类可读的文案。纯函数, 不读文件。"""
    if verdict:
        return "光路复用: 与 latest 模型同一条光路, 跳过采集+拟合直接复用"
    if reference is None:
        return "光路复用: 无 latest 参照, 走正常全流程"
    if current is None:
        return "光路复用: 当前探针缺失, 走正常全流程"
    return (
        "光路复用: 光路已变 (zero=({:.1f},{:.1f}) vs {:.1f},{:.1f}, "
        "exposure_up={} vs {}), 走正常全流程"
    ).format(
        current.zero_xy[0], current.zero_xy[1],
        reference.zero_xy[0], reference.zero_xy[1],
        current.exposure_range_ms[1] if current.exposure_range_ms else float("nan"),
        reference.exposure_range_ms[1] if reference.exposure_range_ms else float("nan"),
    )


__all__ = [
    "DEFAULT_TOL_ZERO_PX",
    "DEFAULT_TOL_EXPOSURE_FACTOR",
    "FLAT_FROZEN_SIDECAR",
    "OpticalProbe",
    "measure_probe",
    "locate_zero",
    "read_camera_format",
    "read_bit_depth",
    "probe_from_latest",
    "same_path",
    "format_probe_verdict",
]