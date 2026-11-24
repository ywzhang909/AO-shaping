"""光路复用门：只省全阶梯，不省现场安全底线。

当 ``data/debug/optical_path_probe/`` 里已有最近一次 ``verdict=ok`` 的探针记录时，
本脚本用一帧**当前**平场 + 单曝光饱和校验 + 驱动板 ID 核对来决定是否可以跳过
Step 3 的 8 档曝光 / 4 档离焦全阶梯。任何一步不满足都 fail-closed：结果写
``reuse_gate=failed``，退出码 3，调用方必须改跑
``scripts/slm/probe_optical_path_validity.py`` 的完整 Step 3–4。

复用门**不**替代标准光路记录：当天 Fresnel D 与 aperture 仍按
``libs/skills/ao-optical-path-calibration/SKILL.md`` 手工补录；本脚本只把 2π 灰度
现场读数与参照 D 的来源写进结果，绝不凭记忆造一个数。

用法::

    python scripts/slm/optical_path_reuse_gate.py \
        --out-root data/debug/optical_path_probe --run-dir data/runs/my_run \
        --wait-until-idle 600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for _candidate in (ROOT, ROOT / "src", ROOT / "libs"):
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from ao_shaping.drivers.ccd.common import (
    create_camera,
    get_camera_exposure_ms,
    get_camera_exposure_range,
    set_camera_exposure_ms,
)
from ao_shaping.drivers.ccd.daheng._bit_depth import bit_depth_of_format
from ao_shaping.drivers.slm.factory import require_drive_board_entry
from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.tools.kernels import device_mutex
from ao_shaping.tools.kernels.device_mutex import DeviceLock
from ao_shaping.tools.slm.transfer.hardware import opened_device, prepare_camera_sdk
from ao_shaping.tools.slm.transfer.optical_probe import (
    FLAT_FRAME_FILE,
    OpticalProbe,
    assess_reuse,
    locate_zero,
    mono_bit_depth_of_probe,
    read_bit_depth,
    read_camera_format,
    save_flat_frame,
)
from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

#: 探针 / 标准记录的固定文件名。
PROBE_FILE = "optical_path_probe.json"
RECORD_FILE = "optical_path_record.json"
GATE_FILE = "optical_path_reuse_gate.json"
GATE_SCHEMA = "optical_path_reuse_gate/1"

#: 本脚本自己写的结果 schema 版本与退出码。
EXIT_PASSED = 0
EXIT_FAILED = 3

#: 与 ``probe_optical_path_validity.SATURATION_FRACTION`` 同口径；单曝光饱和校验
#: 还额外要求峰值 <= 0.7 x 满量程（SKILL Step 4 的留裕度口径）。
DEFAULT_SATURATION_FRACTION = 0.95
DEFAULT_PEAK_FRACTION = 0.7

#: 参考平场在探针目录里的名字（``defocus_sweep[tag=flat_repeat].npz``）。
FLAT_REPEAT_TAG = "flat_repeat"


@dataclass(frozen=True)
class ReferenceProbe:
    """一次 ``verdict=ok`` 参照记录里，复用门实际要用到的东西。"""

    source_dir: Path
    probe: OpticalProbe
    frame: np.ndarray | None
    record: dict[str, Any]
    record_path: Path
    exposure_ms: float
    drive_board_id: str | None
    two_pi_grayscale: int | None
    fresnel_compensation_m_inv: float | None
    date: str | None
    config_id: str | None


def _read_json(path: Path) -> dict[str, Any] | None:
    """JSON 读不出来返回 None（缺记录是 fail-closed 的一条理由，不是异常）。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(f"复用门: 读不了 {path}: {exc!r}")
        return None
    return value if isinstance(value, dict) else None


def _probe_sort_stamp(doc: dict[str, Any], path: Path) -> float:
    """探针的排序时刻（epoch 秒）；有 ISO-8601 ``created_utc`` 就用它。

    必须归一成数字：旧探针可能缺 ``created_utc`` 只能退回 mtime，若让 ISO 字符串
    与 mtime 浮点数直接比字典序，``"2026-..."`` 恒大于 ``"1732..."``，会把一份更旧
    但带时间戳的探针当成"最新"，让复用门拿错参照。
    """
    raw = doc.get("created_utc")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:  # 时间戳坏了，退回 mtime
            pass
    return path.stat().st_mtime


def find_latest_ok_probe(out_root: Path | str) -> Path | None:
    """在 ``out_root`` 下找最近一次 ``verdict.state == "ok"`` 的探针 JSON。

    排序优先用记录里的 ``created_utc``，缺字段/坏字段才退回文件 mtime。只找探针
    文件；配套的标准记录 / 平场帧由 :func:`load_reference` 逐项检查，缺一即 fail-closed。
    """
    root = Path(out_root)
    if not root.is_dir():
        return None
    found: list[tuple[float, Path]] = []
    for path in root.rglob(PROBE_FILE):
        doc = _read_json(path)
        if doc is None or (doc.get("verdict") or {}).get("state") != "ok":
            continue
        found.append((_probe_sort_stamp(doc, path), path))
    if not found:
        return None
    found.sort(key=lambda item: (item[0], str(item[1])))
    return found[-1][1]


def normalize_camera_format(value: Any, bit_depth: int | None = None) -> str | None:
    """把设备回读的 ``PixelFormat`` 与配置位深归一到同一个 ``MONOx`` 名字。

    探针 JSON 不存 ``camera_format``，参照侧只能从 ``config.bit_depth`` 推；
    设备侧 ``read_camera_format()`` 的 gxipy 回读却可能是 ``"Mono12"``。两边
    直接比字符串会假装换了相机，所以统一走驱动自己的位深映射。
    """
    bits = bit_depth_of_format(value)
    if bits is None and isinstance(bit_depth, (int, float)) and bit_depth > 0:
        bits = int(bit_depth)
    if bits in (8, 12, 16):
        return f"MONO{bits}"
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return str(value).strip().upper()


def load_flat_frame_any(path: Path) -> np.ndarray | None:
    """读平场 npz；兼容单帧 ``frame`` 与探针的 5 帧栈 ``frames``（取均值）。"""
    try:
        with np.load(path, allow_pickle=False) as payload:
            if "frame" in payload.files:
                frame = np.asarray(payload["frame"], dtype=np.float64)
            elif "frames" in payload.files:
                frame = np.asarray(payload["frames"], dtype=np.float64).mean(axis=0)
            else:
                return None
    except (OSError, KeyError, ValueError) as exc:
        logger.warning(f"复用门: 平场帧读不出 ({path}): {exc!r}")
        return None
    return frame if frame.ndim == 2 and frame.size > 0 else None


def _flat_repeat_entry(doc: dict[str, Any]) -> dict[str, Any] | None:
    for entry in doc.get("defocus_sweep") or []:
        if isinstance(entry, dict) and entry.get("tag") == FLAT_REPEAT_TAG:
            return entry
    return None


def load_reference(probe_path: Path | str) -> ReferenceProbe:
    """把参照探针 JSON + 标准记录 + 平场帧组装成 :class:`ReferenceProbe`。

    Raises:
        ValueError: 探针不是 ok / 缺平场证据 / 缺标准记录 / 关键字段不可用。
    """
    path = Path(probe_path)
    doc = _read_json(path)
    if doc is None:
        raise ValueError(f"参照探针读不出: {path}")
    if (doc.get("verdict") or {}).get("state") != "ok":
        raise ValueError(f"参照探针不是 verdict=ok: {path}")

    source_dir = path.parent
    record_path = source_dir / RECORD_FILE
    record = _read_json(record_path)
    if record is None:
        raise ValueError(f"参照缺标准记录 {RECORD_FILE}: {source_dir}")
    optical = record.get("optical_path") or {}
    if optical.get("probe_status") not in (None, "ok"):
        raise ValueError(f"参照标准记录的 probe_status 不是 ok: {record_path}")

    config = doc.get("config") or {}
    device = doc.get("device") or {}
    bits_raw = device.get("cam_bit_depth") or config.get("bit_depth")
    bits = int(bits_raw) if isinstance(bits_raw, (int, float)) and bits_raw > 0 else None
    camera_format = normalize_camera_format(None, bits)

    flat_entry = _flat_repeat_entry(doc)
    flat_name = str(flat_entry.get("npz")) if flat_entry and flat_entry.get("npz") else None
    if not flat_name:
        raise ValueError(f"参照缺 {FLAT_REPEAT_TAG} 平场帧证据: {path}")
    frame = load_flat_frame_any(source_dir / flat_name)
    if frame is None:
        raise ValueError(f"参照平场帧不可用: {source_dir / flat_name}")

    if flat_entry and flat_entry.get("centroid_x") is not None and flat_entry.get("centroid_y") is not None:
        zero_xy = (float(flat_entry["centroid_x"]), float(flat_entry["centroid_y"]))
    else:
        zero_xy = locate_zero(frame)

    exposure_range = device.get("cam_exposure_range_ms") or [0.011, 10_000.0]
    exposure_ms = config.get("defocus_exposure_ms")
    if exposure_ms is None:
        recommended = (doc.get("ladder_summary") or {}).get("recommended_exposure") or {}
        exposure_ms = recommended.get("exposure_ms")
    if not isinstance(exposure_ms, (int, float)) or exposure_ms <= 0:
        raise ValueError(f"参照没有可用候选曝光: {path}")

    slm_info = (record.get("devices") or {}).get("slm") or {}
    return ReferenceProbe(
        source_dir=source_dir,
        probe=OpticalProbe(
            zero_xy=zero_xy,
            exposure_range_ms=(float(exposure_range[0]), float(exposure_range[1])),
            camera_format=camera_format,
            bit_depth=bits,
            slm_serial=str(device["slm_serial"]) if device.get("slm_serial") else None,
            flat_frame=flat_name,
        ),
        frame=frame,
        record=record,
        record_path=record_path,
        exposure_ms=float(exposure_ms),
        drive_board_id=str(slm_info["drive_board_id"]) if slm_info.get("drive_board_id") else None,
        two_pi_grayscale=slm_info.get("two_pi_grayscale"),
        fresnel_compensation_m_inv=optical.get("fresnel_compensation_m_inv"),
        date=record.get("date"),
        config_id=optical.get("config_id"),
    )


def check_exposure_frame(
    frame: Any,
    full_scale: float,
    *,
    peak_fraction: float = DEFAULT_PEAK_FRACTION,
    saturation_fraction: float = DEFAULT_SATURATION_FRACTION,
) -> dict[str, Any]:
    """单曝光饱和校验：峰值留 0.7x 满量程裕度，且 0 个饱和像素。

    ``full_scale`` 必须由相机实例的 ``effective_full_scale()`` 给出（issue #246：
    MONO12 帧装在 uint16 容器里，按 dtype 推会得到 65535）。
    """
    array = np.asarray(frame)
    if array.ndim != 2 or array.size == 0:
        return {"passed": False, "reason": "曝光校验帧不是非空 2D 图像"}
    scale = float(full_scale)
    if not np.isfinite(scale) or scale <= 0:
        return {"passed": False, "reason": f"满量程不可用: {full_scale!r}"}
    peak = float(np.max(array))
    peak_limit = peak_fraction * scale
    sat_threshold = saturation_fraction * scale
    saturated_pixels = int(np.count_nonzero(array >= sat_threshold))
    passed = peak <= peak_limit and saturated_pixels == 0
    return {
        "passed": passed,
        "full_scale": scale,
        "peak_counts": peak,
        "peak_limit_counts": peak_limit,
        "peak_fraction": peak_fraction,
        "saturation_threshold_counts": sat_threshold,
        "saturation_fraction": saturation_fraction,
        "saturated_pixels": saturated_pixels,
        "reason": (
            "单曝光校验通过"
            if passed
            else f"峰值 {peak:.1f} > {peak_limit:.1f} 或饱和像素 {saturated_pixels} != 0"
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _acquire_device_lock(args: argparse.Namespace) -> DeviceLock | None:
    """复用门也要持锁：开 SLM/相机期间不能让第二个会话拿走句柄。"""
    verdict = device_mutex.wait_until_idle(
        float(args.wait_until_idle),
        slm_number=int(args.slm_number),
        cam_id=int(args.cam_id),
        cam_type=str(args.cam_type),
        exposure_ms=float(getattr(args, "probe_exposure_ms", 0.05)),
        own_paths=[Path(__file__).resolve()],
    )
    if not verdict.ok:
        raise RuntimeError(f"设备不可用 ({verdict.status}/{verdict.kind}): {verdict.reason}")
    lock = verdict.lock if verdict.lock is not None else DeviceLock.default()
    if not lock.is_locked():
        lock.acquire(timeout=0.0)
    return lock


def run_reuse_gate(args: argparse.Namespace) -> dict[str, Any]:
    """跑一次复用门；结果始终写 JSON，失败时 ``reuse_gate=failed``。"""
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": GATE_SCHEMA,
        "created_utc": _utc_now(),
        "reuse_gate": "failed",
        "reasons": [],
        "reference": None,
        "criteria": {},
        "flat_frame_comparison": {},
        "exposure_check": {},
        "on_site": {},
        "devices": {},
        "remedy": "改跑 scripts/slm/probe_optical_path_validity.py 的完整 Step 3–4",
    }

    def fail(reason: str) -> None:
        logger.warning(f"复用门失败: {reason}")
        result["reasons"].append(reason)

    lock: DeviceLock | None = None
    try:
        probe_path = find_latest_ok_probe(args.out_root)
        if probe_path is None:
            fail(f"{args.out_root} 下没有 verdict=ok 的参照探针")
        else:
            reference = load_reference(probe_path)
            result["reference"] = {
                "probe_json": str(probe_path),
                "record_json": str(reference.record_path),
                "date": reference.date,
                "config_id": reference.config_id,
            }
            result["reference_exposure_ms"] = reference.exposure_ms
            logger.info(f"复用门参照: {probe_path}")

            if reference.date != datetime.now(timezone.utc).date().isoformat():
                fail(f"参照记录日期 {reference.date} 不是当天，不能当本次光路证据")

            lock = _acquire_device_lock(args)
            prepare_camera_sdk(str(args.cam_type))
            cam = create_camera(
                str(args.cam_type),
                int(args.cam_id),
                exposure_time_ms=reference.exposure_ms,
            )
            with opened_device(cam) as cam_handle:
                with opened_device(
                    Santec(
                        slm_number=int(args.slm_number),
                        wavelength=int(args.wavelength),
                        video_mode=0,
                    )
                ) as slm:
                    # 驱动板必须现场应答，且 ID 与参照一致；读不到就 fail-closed。
                    board_id = require_drive_board_entry(slm)
                    result["devices"] = {
                        "slm_number": int(args.slm_number),
                        "cam_type": str(args.cam_type),
                        "cam_id": int(args.cam_id),
                        "drive_board_id": board_id,
                    }
                    if not board_id:
                        fail("驱动板 ID 读不到（不能排除选板 ID 静默回落）")
                    elif reference.drive_board_id and board_id != reference.drive_board_id:
                        fail(f"驱动板 ID 不一致: 当前 {board_id} vs 参照 {reference.drive_board_id}")

                    # 2π 灰度是设备当天读数，不抄参照、不写死 993/998。
                    two_pi: int | None = None
                    try:
                        _wavelength_readback, two_pi_readback = slm.get_wavelength_info()
                        two_pi = int(two_pi_readback)
                    except Exception as exc:  # pragma: no cover - 设备读数失败
                        fail(f"现场 2π 灰度读不到: {exc!r}")

                    slm_info = (reference.record.get("devices") or {}).get("slm") or {}
                    result["on_site"] = {
                        "recorded_utc": _utc_now(),
                        "two_pi_grayscale": two_pi,
                        "two_pi_source": "slm.get_wavelength_info()[1] 现场读数",
                        "fresnel_compensation_m_inv": reference.fresnel_compensation_m_inv,
                        "fresnel_source": "参照标准记录（当天 D 仍按标准记录手工补录）",
                        "fresnel_manual_required": reference.fresnel_compensation_m_inv is None,
                        "reference_slm_serial": reference.probe.slm_serial,
                        "current_slm_serial": (
                            None
                            if getattr(slm, "serial_number", None) is None
                            else str(slm.serial_number)
                        ),
                        "option_board_id": slm_info.get("option_board_id"),
                    }

                    floor, ceiling = get_camera_exposure_range(cam_handle)
                    if str(args.cam_type) == "daheng":
                        limit = configured_daheng_exposure_limit()
                        if limit > 0:
                            ceiling = min(ceiling, limit)
                    effective_ms = set_camera_exposure_ms(cam_handle, reference.exposure_ms)
                    actual_ms = get_camera_exposure_ms(cam_handle)
                    result["exposure"] = {
                        "requested_ms": reference.exposure_ms,
                        "effective_ms": effective_ms,
                        "actual_readback_ms": actual_ms,
                        "range_ms": [floor, ceiling],
                    }

                    # 一帧**当前**平场：零相位覆盖层 + 与采集段同一口径。
                    slm.set_phase_overlay(None)
                    try:
                        slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
                        raw = cam_handle.get_numpy_image(n_sample=1, skip_first=False)
                        frame = np.asarray(raw, dtype=np.float64)
                        exposure_check = check_exposure_frame(
                            frame,
                            float(cam_handle.effective_full_scale(raw)),
                            peak_fraction=float(args.peak_fraction),
                            saturation_fraction=float(args.saturation_fraction),
                        )
                        result["exposure_check"] = exposure_check
                        if not exposure_check["passed"]:
                            fail(exposure_check["reason"])

                        zero = locate_zero(frame)
                        saved_name = save_flat_frame(
                            run_dir,
                            frame,
                            spot_xy=zero,
                            exposure_ms=actual_ms,
                            slm_serial=(
                                None
                                if getattr(slm, "serial_number", None) is None
                                else str(slm.serial_number)
                            ),
                        )
                        result["current_flat_frame"] = (
                            str(run_dir / FLAT_FRAME_FILE) if saved_name else None
                        )
                    finally:
                        # 复位平场，与 measure_probe / 采集段 finally 一致。
                        slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))

                    raw_format = read_camera_format(cam_handle)
                    current_bits = read_bit_depth(cam_handle)
                    current = OpticalProbe(
                        zero_xy=zero,
                        exposure_range_ms=(float(floor), float(ceiling)),
                        camera_format=normalize_camera_format(raw_format, current_bits),
                        bit_depth=int(current_bits) if isinstance(current_bits, (int, float)) else None,
                        slm_serial=(
                            None
                            if getattr(slm, "serial_number", None) is None
                            else str(slm.serial_number)
                        ),
                        flat_frame=saved_name,
                    )
                    current_compare_bits = mono_bit_depth_of_probe(current)
                    reference_compare_bits = mono_bit_depth_of_probe(reference.probe)
                    result["flat_frame_comparison"] = {
                        "policy": (
                            "effective_full_scale"
                            if current_compare_bits is not None
                            and reference_compare_bits is not None
                            else "unavailable"
                        ),
                        "current_camera_format": current.camera_format,
                        "reference_camera_format": reference.probe.camera_format,
                        "current_bit_depth": current.bit_depth,
                        "reference_bit_depth": reference.probe.bit_depth,
                        "current_full_scale": (
                            (1 << current_compare_bits) - 1
                            if current_compare_bits is not None
                            else None
                        ),
                        "reference_full_scale": (
                            (1 << reference_compare_bits) - 1
                            if reference_compare_bits is not None
                            else None
                        ),
                        "cross_bit_depth": bool(
                            current_compare_bits is not None
                            and reference_compare_bits is not None
                            and current_compare_bits != reference_compare_bits
                        ),
                    }
                    decision = assess_reuse(
                        current,
                        reference.probe,
                        current_frame=frame,
                        reference_frame=reference.frame,
                    )
                    result["criteria"] = decision.to_dict()
                    if not decision.same:
                        fail("复用判据不一致: " + "; ".join(decision.criteria.values()))
                    if two_pi is None:
                        fail("现场 2π 灰度缺失，不能确认当天波长口径")

                    if not result["reasons"]:
                        result["reuse_gate"] = "passed"
                        logger.info("复用门通过：跳过 Step 3 全阶梯，仍须现场补录 Fresnel D / aperture")
    except Exception as exc:  # fail-closed：任何异常都退回完整 Step 3–4
        fail(f"{type(exc).__name__}: {exc}")
    finally:
        if lock is not None:
            try:
                lock.release()
                logger.info("复用门: 设备锁已释放（相机 + SLM 已关闭）")
            except Exception as exc:  # pragma: no cover - 释放失败只告警
                logger.warning(f"复用门: 设备锁释放失败: {exc!r}")

    result["closed_normally"] = True
    target = run_dir / GATE_FILE
    target.write_text(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info(f"复用门结果: {result['reuse_gate']} → {target}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", default="data/debug/optical_path_probe", help="探针记录根目录")
    parser.add_argument("--run-dir", default=None, help="本次门控结果目录；缺省在 out-root 下新建时间戳目录")
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--wavelength", type=int, default=1064)
    parser.add_argument("--cam-type", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--wait-until-idle", type=float, default=0.0)
    parser.add_argument("--peak-fraction", type=float, default=DEFAULT_PEAK_FRACTION)
    parser.add_argument("--saturation-fraction", type=float, default=DEFAULT_SATURATION_FRACTION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_dir = str(Path(args.out_root) / f"reuse_gate_{stamp}")
    result = run_reuse_gate(args)
    return EXIT_PASSED if result["reuse_gate"] == "passed" else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
