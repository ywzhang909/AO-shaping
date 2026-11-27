#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一缺数据采集：单次实验窗口合并 issue_validity_verdict §4 五项缺数据。

为什么合并成一次任务
------------------
``docs/dev/issue_validity_verdict_20261127.md`` §4 列出的五项缺数据采集，分别阻塞
#247 / #238 / #236 / #234 / #156 / #233。逐项分开上实验意味着多次开/关 SLM、多次
抢台面锁、多次重建光路状态 —— 而 §2 已证实面板轴路由跨会话不稳，**每次开关设备
都可能改变面板旋转/寻址态**，使分次采集的帧不可直接对照。因此把它们合并成**一次
实验任务**：一个台面锁、一次 SLM 打开、一次 CCD 打开、一个全新输出目录、一份
journal、一份 release 记录，五个 stage 按固定顺序在同一设备态内连续采集，使各帧
共享同一面板态，可离线直接对照。

五个 stage（顺序固定，逐 stage 独立 fail-closed）
-----------------------------------------------
A. ``panel_state_dump``        —— #247 根因：只读面板旋转/寻址态 dump（不调制面板）。
B. ``dark_signal_pair``        —— #238/#236 强度门：同曝光熄灯暗场帧 + 信号帧。
C. ``direction_geometry_probe``—— #234/#236 方向门：与训练方向独立的 ramp/2D 探针。
D. ``paired_closed_loop``      —— #156：受控/随机/离线/仅预训练配对帧 + SNR/EKF 同步日志。
E. ``cdtwin_twin_set``         —— #233：CD-Twin 闭环孪生实测帧 + 同步日志。

单个 stage 失败不影响其它 stage：该 stage 状态记为 ``error``/``no_signal``/
``saturated``，缺失数据一律写 ``null``，**绝不**用几何中心或重复帧冒充测量值。

口径（禁止重写公式，全部复用冻结源）
-----------------------------------
* 灰度 wrap 唯一入口：``issue247_geometry_capture_20261120.phase_to_gray``；2π 灰度
  标度取 S1 ``PATTERN_SPECS`` 中 flat 记录的冻结值（与 S1/x/y 光栅同源）。
* 平场：``flat_phase`` → ``phase_to_gray``；光栅/2D 探针：``grating_phase`` /
  ``blaze2d_phase``（均 raw 弧度，经同一 wrap 点）。
* 无信号/饱和门：S1 ``flat_signal_measurement`` + ``SATURATION_FRACTION``；阈值
  直接取自 S1 模块常量，本文件不新造数值。
* 面板形状 / CCD 全幅形状 / 位深 / 满量程：与 x/y 光栅采集脚本同口径。

fail-closed
-----------
无信号 / 饱和 / 帧形状不符 / 灰度完整性校验失败时：该 stage 状态与退出码为非成功，
缺失质心、pitch、SNR 一律写 ``null``。``--execute`` 退出码：0 = 全部 stage ok；
1 = 任一 stage 非 ok（error/no_signal/saturated/dark_not_dark）；2 = 等不到台面锁。

前置门控：B 要求操作员熄灯并加 ``--confirm-lights-off``；D / E 分别要求
``--confirm-magnification-alignment`` / ``--confirm-g1-g3-g4``；默认记 ``blocked``
且不放行硬件操作。这样一次窗口内既能就绪 A/C 的实测，也不会在未熄灯时把有信号的
帧冒充暗场，或在倍率口径 / G1+G3+G4 未闭合时伪造闭环配对或孪生数据。
``--stages`` 可按子集重跑（如熄灯后仅 ``--stages B``），不再重复开/关其它 stage。

硬件纪律：没有 ``--execute`` 时只打印确定性 plan，**不 import 驱动、不打开任何设备**；
台面锁、SLM/CCD 驱动只在 ``execute`` 内惰性导入/获取，``finally`` 保证释放并写
``released.json`` 与 journal。不操作激光；暗场帧要求操作员在采集前确认熄灯，脚本
先检测暗场帧 ``has_signal``，为真即 ``dark_not_dark`` 并拒绝作为强度基线。
不写共享 report/provenance 文件。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "libs", ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.slm.capture_s1_mapping_disambiguation_20261124 import (
    BLAZE2D_PERIOD_X_PX,
    BLAZE2D_PERIOD_Y_PX,
    GRATING_PERIOD_PX,
    NATIVE_CCD_SHAPE,
    PATTERN_SPECS,
    SATURATION_FRACTION,
    blaze2d_phase,
    digest,
    flat_signal_measurement,
    utc_now,
    write_json,
)
from scripts.slm.issue247_geometry_capture_20261120 import (
    flat_phase,
    grating_phase,
    phase_to_gray,
)

PLAN_SCHEMA = "unified-missing-data-capture-plan/v1"
JOURNAL_SCHEMA = "unified-missing-data-capture-journal/v1"
RELEASE_SCHEMA = "unified-missing-data-capture-released/v1"
SUMMARY_SCHEMA = "unified-missing-data-capture-summary/v1"

#: 面板形状 (rows, cols)；驱动 Panel_Res 为 (cols, rows)，与 x/y 光栅脚本同口径。
DEFAULT_PANEL_SHAPE = (1200, 1920)
BIT_DEPTH_REQUIRED = 12
FULL_SCALE_REQUIRED = float((1 << BIT_DEPTH_REQUIRED) - 1)
DEFAULT_EXPOSURE_MS = 0.087
DEFAULT_WAIT_IDLE_S = 600.0
DEFAULT_SLM_NUMBER = 1
DEFAULT_CAM_TYPE = "daheng"
DEFAULT_CAM_ID = 0
DEFAULT_FRAMES = 2
DEFAULT_SETTLE_S = 0.8
LOCK_POLL_S = 1.0

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_LOCK = 2

#: 五个 stage 的固定执行顺序。
STAGE_ORDER: tuple[str, ...] = ("A", "B", "C", "D", "E")

STAGE_LABELS: dict[str, str] = {
    "A": "panel_state_dump",
    "B": "dark_signal_pair",
    "C": "direction_geometry_probe",
    "D": "paired_closed_loop",
    "E": "cdtwin_twin_set",
}

#: §4 前置：D 依赖倍率口径对齐，E 依赖 G1+G3+G4（G4 又依赖 #238 暗场）。
#: 前置未确认时对应 stage 记 ``blocked``、不放行硬件操作，绝不伪造闭环数据。
#: B 的熄灯确认是硬门槛：实测未熄灯时暗场帧峰值 2265、has_signal=true，
#: 该帧不是暗场，不得作为 #238/#236 的强度基线。
STAGE_PREREQUISITES: dict[str, str] = {
    "A": "光路预检 ok",
    "B": "操作员已熄灯/遮挡光路（同一曝光）+ 暗场 hook（#254 T5）",
    "C": "G1 映射闭合",
    "D": "倍率口径对齐（catA/catB/C06）",
    "E": "G1 + G3 + G4（G4 依赖 #238 暗场）",
}

#: B/D/E 的显式放行开关（操作员确认前置后手动加）。
STAGE_GATE_FLAGS: dict[str, str] = {
    "B": "confirm_lights_off",
    "D": "confirm_magnification_alignment",
    "E": "confirm_g1_g3_g4",
}

#: stage C 的方向几何探针（与训练方向独立：行向 / 列向 / 2D）。
#: 全部几何常量取自 S1 冻结规格：grating axis=1 为面板 x（列向）、axis=0 为面板 y（行向）。
PROBE_NAMES: tuple[str, ...] = ("ramp_x", "ramp_y", "blaze2d")
PROBE_PERIOD_PX = int(GRATING_PERIOD_PX)
BLAZE2D_PROBE_PERIODS_PX = (int(BLAZE2D_PERIOD_X_PX), int(BLAZE2D_PERIOD_Y_PX))

#: stage D 的四个配对变体：受控 / 随机 / 离线 / 仅预训练。
PAIR_ARMS: tuple[str, ...] = ("controlled", "random", "offline", "pretrained")
ARM_PATTERNS: dict[str, str] = {
    "controlled": "ramp_x",
    "random": "ramp_y",
    "offline": "blaze2d",
    "pretrained": "flat",
}

#: stage E 的闭环孪生序列（flat → ramp_x → blaze2d → flat）。
TWIN_SEQUENCE: tuple[str, ...] = ("flat", "ramp_x", "blaze2d", "flat")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #

def _journal(handle: Any, event: str, **fields: Any) -> None:
    """Append one journal line (jsonl); never raises on serialization."""
    record = {"event": event, "ts_utc": utc_now(), **fields}
    try:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()
    except (TypeError, ValueError):
        record["serialization_error"] = "journal field not serializable"
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _jsonable(value: Any) -> Any:
    """Convert numpy/nested values to JSON-serializable form."""
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _two_pi_gray_scale() -> int:
    """冻结 2π 灰度标度：取 S1 flat 记录，与 S1 / x/y 光栅同源。"""
    flat = next(spec for spec in PATTERN_SPECS if spec["kind"] == "flat")
    return int(flat["two_pi_gray_scale"])


def _flat_gray(shape: tuple[int, int]) -> np.ndarray:
    """平场 uint16 灰度（经唯一 wrap 点 ``phase_to_gray``）。"""
    return phase_to_gray(flat_phase(shape), _two_pi_gray_scale())


def _probe_gray(name: str, shape: tuple[int, int]) -> np.ndarray:
    """方向几何探针的 uint16 灰度（行向 / 列向 ramp / 2D blaze）。"""
    scale = _two_pi_gray_scale()
    if name == "ramp_x":
        phase = grating_phase(shape, period_px=PROBE_PERIOD_PX, axis=1)
    elif name == "ramp_y":
        phase = grating_phase(shape, period_px=PROBE_PERIOD_PX, axis=0)
    elif name == "blaze2d":
        phase = blaze2d_phase(
            shape,
            period_x_px=float(BLAZE2D_PROBE_PERIODS_PX[0]),
            period_y_px=float(BLAZE2D_PROBE_PERIODS_PX[1]),
            sign_x=1,
            sign_y=1,
        )
    else:
        raise ValueError(f"unknown probe name {name!r}")
    return phase_to_gray(phase, scale)


def _arm_gray(arm: str, shape: tuple[int, int]) -> np.ndarray:
    """stage D 配对变体的灰度图案。"""
    if arm == "pretrained":
        return _flat_gray(shape)
    return _probe_gray(ARM_PATTERNS[arm], shape)


def _twin_gray(step_name: str, shape: tuple[int, int]) -> np.ndarray:
    """stage E 孪生序列的灰度图案。"""
    if step_name == "flat":
        return _flat_gray(shape)
    return _probe_gray(step_name, shape)


def _gate_frame(frame: np.ndarray, full_scale: float) -> dict[str, Any]:
    """Raw-only 门：S1 ``flat_signal_measurement`` + 饱和判据，不新造阈值。"""
    measured = flat_signal_measurement(frame)
    peak = int(frame.max()) if frame.size else 0
    threshold = float(SATURATION_FRACTION) * float(full_scale)
    saturated = bool(peak >= threshold)
    #: 直接复用 S1 ``has_signal``（阈值常量同源于 S1，本文件不重算）。
    has_signal = bool(measured["has_signal"])
    return {
        "raw_peak_counts": peak,
        "raw_peak_snr": float(measured["peak_snr"]),
        "raw_median_counts": float(measured["median_counts"]),
        "raw_sigma_counts": float(measured["sigma_counts"]),
        "signal_pixels": int(measured["signal_pixels"]),
        "signal_weight_counts": float(measured["signal_weight_counts"]),
        "saturated": saturated,
        "has_signal": has_signal,
    }


def _capture_frame(cam: Any, settle_s: float) -> np.ndarray:
    """Capture one full-frame image after a settle wait; validate shape/dtype."""
    time.sleep(float(settle_s))
    frame = np.asarray(cam.get_numpy_image(1, True)).copy()
    if frame.shape != NATIVE_CCD_SHAPE:
        raise RuntimeError(f"expected full-frame {NATIVE_CCD_SHAPE}, got {frame.shape}")
    if frame.dtype.kind not in "ui":
        raise RuntimeError(f"expected integer CCD data, got dtype {frame.dtype!r}")
    return frame


def _save_frame(raw_dir: Path, name: str, frame: np.ndarray) -> str:
    """Save one raw frame; return the file name (deterministic, no clock)."""
    path = raw_dir / name
    np.save(path, frame)
    return path.name


def _panel_shape_from(slm: Any) -> tuple[int, int]:
    """Read panel shape (rows, cols) from the SLM driver."""
    panel = getattr(slm, "Panel_Res", None)
    if panel is None:
        return DEFAULT_PANEL_SHAPE
    cols, rows = int(panel[0]), int(panel[1])
    return (rows, cols)


def acquire_device_lock(wait_s: float) -> Any | None:
    """Acquire the bench DeviceLock, polling until deadline; ``None`` on timeout."""
    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    lock = DeviceLock.default()
    deadline = time.monotonic() + max(0.0, float(wait_s))
    while True:
        if lock.acquire(timeout=0.0):
            return lock
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(LOCK_POLL_S, remaining))


def require_drive_board_entry(slm: Any) -> str | None:
    """Acquisition-entry drive-board guard (duck-typed no-op for sim panels)."""
    from ao_shaping.drivers.slm.factory import require_drive_board_entry as guard

    return guard(slm)


def _open_devices(args: argparse.Namespace) -> tuple[Any, Any, float, float, str | None]:
    """Lazily open SLM + CCD once; return (slm, cam, exposure_actual, full_scale, drive_board)."""
    from ao_shaping.drivers.ccd.common import (
        create_camera,
        get_camera_exposure_ms,
        set_camera_exposure_ms,
    )
    from ao_shaping.drivers.slm.santec import Santec

    slm = Santec(slm_number=int(args.slm_number), video_mode=0)
    slm.open()
    drive_board = require_drive_board_entry(slm)
    panel_shape = _panel_shape_from(slm)
    if panel_shape != DEFAULT_PANEL_SHAPE:
        raise RuntimeError(f"expected panel {DEFAULT_PANEL_SHAPE}, got {panel_shape}")
    slm.set_phase_overlay(None)

    cam = create_camera(
        str(args.cam_type),
        int(args.cam_id),
        exposure_time_ms=float(args.exposure_ms),
        bit_depth=BIT_DEPTH_REQUIRED,
    )
    cam.open()
    bit_depth = int(getattr(cam, "bit_depth", BIT_DEPTH_REQUIRED))
    if bit_depth != BIT_DEPTH_REQUIRED:
        raise RuntimeError(f"camera bit-depth must be {BIT_DEPTH_REQUIRED}, got {bit_depth!r}")
    set_camera_exposure_ms(cam, float(args.exposure_ms))
    exposure_actual = float(get_camera_exposure_ms(cam))
    full_scale = float(cam.effective_full_scale(np.zeros((1,), dtype=np.uint16)))
    if full_scale != FULL_SCALE_REQUIRED:
        raise RuntimeError(f"camera full scale {full_scale} != {FULL_SCALE_REQUIRED}")
    return slm, cam, exposure_actual, full_scale, drive_board


# --------------------------------------------------------------------------- #
# Stage A —— 面板旋转 / 寻址态 dump（#247 根因）
# --------------------------------------------------------------------------- #

def stage_a_panel_state(
    slm: Any, out: Path, journal: Any, common: dict[str, Any]
) -> dict[str, Any]:
    """只读面板旋转/寻址态；不调制面板。逐项防御性读取，缺失记 null。"""
    stage_dir = out / "stage_a_panel_state"
    stage_dir.mkdir(parents=True, exist_ok=True)

    reads: dict[str, Any] = {}
    driver_methods: list[tuple[str, str]] = [
        ("wavelength_info", "get_wavelength_info"),
        ("displayed_memory_number", "get_displayed_memory_number"),
        ("current_grayscale", "get_current_grayscale"),
        ("serial_number", "get_serial_number"),
        ("display_name", "get_display_name"),
        ("version", "get_version"),
        ("device_info", "get_device_info"),
    ]
    for label, method_name in driver_methods:
        fn = getattr(slm, method_name, None)
        if fn is None or not callable(fn):
            reads[label] = {"ok": False, "value": None, "error": "method not available"}
            continue
        try:
            reads[label] = {"ok": True, "value": _jsonable(fn()), "error": None}
        except Exception as exc:
            reads[label] = {
                "ok": False,
                "value": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    # SDK 级只读寄存器：ReadVI 的返回值只是 SLM_STATUS（SLM_OK=0），
    # 真实读数经 ctypes.byref 出参取回。ReadT 的真实出参是 c_uint32（驱动
    # 属性 temperature 内部即如此），且 binding argtypes 声明 LPDWORD，
    # 直接传 c_float 会抛 ArgumentError —— 因此温度统一走驱动的只读属性。
    sdk_reads: dict[str, Any] = {}
    temperature = getattr(slm, "temperature", None)
    if temperature is None:
        sdk_reads["temperature"] = {
            "ok": False, "value": None, "error": "property not available"
        }
    else:
        try:
            sdk_reads["temperature"] = {
                "ok": True, "value": _jsonable(temperature), "error": None
            }
        except Exception as exc:
            sdk_reads["temperature"] = {
                "ok": False,
                "value": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    sdk_specs = (("video_mode", "SLM_Ctrl_ReadVI", (ctypes.c_uint32,)),)
    for label, binding, out_types in sdk_specs:
        lib = getattr(slm, "_slm", None)
        fn = getattr(lib, binding, None) if lib is not None else None
        if fn is None:
            sdk_reads[label] = {"ok": False, "value": None, "error": "SDK binding not available"}
            continue
        try:
            out_values = [cls() for cls in out_types]
            status = int(fn(int(slm.slm_number), *(ctypes.byref(v) for v in out_values)))
            scalars = [v.value for v in out_values]
            sdk_reads[label] = {
                "ok": status == 0,
                "status": status,
                "value": _jsonable(scalars[0] if len(scalars) == 1 else scalars)
                if status == 0
                else None,
                "error": None if status == 0 else "SDK status != 0",
            }
        except Exception as exc:
            sdk_reads[label] = {
                "ok": False,
                "value": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    record: dict[str, Any] = {
        "stage": "A",
        "name": STAGE_LABELS["A"],
        "blocking_issues": ["#247"],
        "status": "ok",
        "error": None,
        "panel_shape_rows_cols": list(common.get("panel_shape", DEFAULT_PANEL_SHAPE)),
        "driver_reads": reads,
        "sdk_reads": sdk_reads,
        "note": (
            "只读面板旋转/寻址态，未调制面板；与 §2 三源轴路由对照，定位同 gray 不同轴根因。"
        ),
    }
    write_json(stage_dir / "panel_state.json", record)
    _journal(journal, "stage_a_panel_state", status="ok", reads_ok=sum(
        1 for v in reads.values() if v.get("ok")
    ))
    return record


# --------------------------------------------------------------------------- #
# Stage B —— 同曝光熄灯暗场 + 信号帧（#238 / #236 强度门）
# --------------------------------------------------------------------------- #

def stage_b_dark_signal(
    slm: Any,
    cam: Any,
    out: Path,
    journal: Any,
    settle_s: float,
    full_scale: float,
    exposure_actual: float,
    frame_mode: str = "both",
) -> dict[str, Any]:
    """按 ``frame_mode`` 采集暗场帧 / 信号帧；暗场与信号可分两个窗口采集。

    熄灯与光路开关只能由操作员执行。实测（2026-11-27 1412Z）已证明 ``both`` 不能
    自动成立：同一窗口连采两帧，两帧都带信号（暗场峰值 2265、has_signal=true），
    因为脚本无法在中途关灯或开光。因此正式口径是 ``dark``（熄灯窗口）与 ``signal``
    （开光窗口）各采一次，靠 ``pair_contract`` 的曝光与 flat 灰度 SHA-256 做离线配对；
    ``both`` 仅保留作同窗口哨兵，任一门未过即 fail-closed。
    """
    stage_dir = out / "stage_b_dark_signal"
    raw_dir = stage_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if frame_mode not in ("both", "dark", "signal"):
        raise ValueError(f"未知 --b-frame: {frame_mode!r}")

    gray = _flat_gray(DEFAULT_PANEL_SHAPE)
    gray_sha = digest(gray)
    slm.display_data(gray)

    recorded: dict[str, Any] = {}
    if frame_mode in ("both", "dark"):
        dark = _capture_frame(cam, settle_s)
        dark_gate = _gate_frame(dark, full_scale)
        recorded["dark"] = {
            "file": _save_frame(raw_dir, "00_dark.npy", dark),
            "gate": dark_gate,
        }
    if frame_mode in ("both", "signal"):
        signal = _capture_frame(cam, settle_s)
        signal_gate = _gate_frame(signal, full_scale)
        recorded["signal"] = {
            "file": _save_frame(raw_dir, "01_signal.npy", signal),
            "gate": signal_gate,
        }

    # 门控优先级固定：暗场不暗 > 饱和 > 无信号；只检查本次实际采到的帧。
    status = "ok"
    if recorded.get("dark", {}).get("gate", {}).get("has_signal"):
        # 暗场帧仍有信号 ⇒ 熄灯/遮挡未落实，该帧不可作强度基线。
        status = "dark_not_dark"
    elif any(frame["gate"]["saturated"] for frame in recorded.values()):
        status = "saturated"
    elif "signal" in recorded and not recorded["signal"]["gate"]["has_signal"]:
        status = "no_signal"

    record: dict[str, Any] = {
        "stage": "B",
        "name": STAGE_LABELS["B"],
        "blocking_issues": ["#238", "#236"],
        "status": status,
        "error": None if status == "ok" else f"stage B status={status}（暗场/饱和/信号门未通过）",
        "frame_mode": frame_mode,
        "exposure_ms_actual": exposure_actual,
        "flat_gray_sha256": gray_sha,
        "pair_contract": {
            "frames_expected": ["dark", "signal"],
            "requires_equal_exposure_ms_actual": True,
            "requires_equal_flat_gray_sha256": True,
        },
        "frames": recorded,
        "note": (
            "同曝光暗场 + 信号帧。暗场帧须在熄灯/遮挡窗口采、信号帧须在开光窗口采；"
            "两帧曝光与 flat 灰度 SHA-256 必须一致才可配对。暗场均值作强度门基线，"
            "不在本轮改判 #238/#236 判据。"
        ),
    }
    write_json(stage_dir / "dark_signal_meta.json", record)
    _journal(journal, "stage_b_dark_signal", status=status,
             frame_mode=frame_mode,
             dark_peak=recorded.get("dark", {}).get("gate", {}).get("raw_peak_counts"),
             signal_peak=recorded.get("signal", {}).get("gate", {}).get("raw_peak_counts"))
    return record


# --------------------------------------------------------------------------- #
# Stage C —— 独立方向几何探针（#234 / #236 方向门）
# --------------------------------------------------------------------------- #

def stage_c_direction_probe(
    slm: Any,
    cam: Any,
    out: Path,
    journal: Any,
    settle_s: float,
    full_scale: float,
    exposure_actual: float,
    frames: int,
) -> dict[str, Any]:
    """与训练方向独立的 ramp_x / ramp_y / blaze2d 探针，每探针采 ``frames`` 帧。"""
    stage_dir = out / "stage_c_direction_probe"
    raw_dir = stage_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    probes: dict[str, Any] = {}
    any_bad = False
    for name in PROBE_NAMES:
        gray = _probe_gray(name, DEFAULT_PANEL_SHAPE)
        gray_sha = digest(gray)
        slm.display_data(gray)
        frames_record: list[dict[str, Any]] = []
        for frame_index in range(int(frames)):
            frame = _capture_frame(cam, settle_s)
            gate = _gate_frame(frame, full_scale)
            fname = f"{frame_index:02d}_{name}.npy"
            file_name = _save_frame(raw_dir, fname, frame)
            frames_record.append({"file": file_name, "gate": gate})
            if gate["saturated"] or not gate["has_signal"]:
                any_bad = True
        probes[name] = {"gray_sha256": gray_sha, "frames": frames_record}

    status = "ok"
    if any_bad:
        status = "saturated" if any(
            f["gate"]["saturated"] for p in probes.values() for f in p["frames"]
        ) else "no_signal"

    record: dict[str, Any] = {
        "stage": "C",
        "name": STAGE_LABELS["C"],
        "blocking_issues": ["#234", "#236"],
        "status": status,
        "error": None,
        "exposure_ms_actual": exposure_actual,
        "probe_period_px": int(PROBE_PERIOD_PX),
        "blaze2d_probe_periods_px": list(BLAZE2D_PROBE_PERIODS_PX),
        "probes": probes,
        "note": (
            "面板 x（列向）/ y（行向）ramp 与 2D blaze 探针，独立于训练方向；用于离线对照面板实际几何"
            "映射，不改判 #234 判据。"
        ),
    }
    write_json(stage_dir / "probe_meta.json", record)
    _journal(journal, "stage_c_direction_probe", status=status,
             probes=list(PROBE_NAMES), frames=int(frames))
    return record


# --------------------------------------------------------------------------- #
# Stage D —— 固定 controller 配对闭环（#156）
# --------------------------------------------------------------------------- #

def stage_d_paired_closed_loop(
    slm: Any,
    cam: Any,
    out: Path,
    journal: Any,
    settle_s: float,
    full_scale: float,
    exposure_actual: float,
    frames: int,
) -> dict[str, Any]:
    """受控 / 随机 / 离线 / 仅预训练四变体配对帧 + SNR/EKF 同步日志（jsonl）。"""
    stage_dir = out / "stage_d_paired_closed_loop"
    raw_dir = stage_dir / "raw"
    sync_path = stage_dir / "sync_log.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)

    sync_lines: list[dict[str, Any]] = []
    sync_seq = 0
    arms: dict[str, Any] = {}
    any_bad = False

    for arm in PAIR_ARMS:
        gray = _arm_gray(arm, DEFAULT_PANEL_SHAPE)
        gray_sha = digest(gray)
        frames_record: list[dict[str, Any]] = []
        for step in range(int(frames)):
            slm.display_data(gray)
            frame = _capture_frame(cam, settle_s)
            gate = _gate_frame(frame, full_scale)
            fname = f"{step:02d}_{arm}.npy"
            file_name = _save_frame(raw_dir, fname, frame)
            frames_record.append({"step": step, "file": file_name, "gate": gate})
            sync_entry = {
                "seq": sync_seq,
                "arm": arm,
                "step": step,
                "pattern": ARM_PATTERNS[arm],
                "gray_sha256": gray_sha,
                "frame_file": file_name,
                "snr": gate["raw_peak_snr"],
                "peak_counts": gate["raw_peak_counts"],
                "saturated": gate["saturated"],
                "has_signal": gate["has_signal"],
                "ts_utc": utc_now(),
            }
            sync_lines.append(sync_entry)
            sync_seq += 1
            if gate["saturated"] or not gate["has_signal"]:
                any_bad = True
        arms[arm] = {"gray_sha256": gray_sha, "frames": frames_record}

    with sync_path.open("w", encoding="utf-8") as handle:
        for line in sync_lines:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")

    status = "ok"
    if any_bad:
        status = "saturated" if any(
            f["gate"]["saturated"] for a in arms.values() for f in a["frames"]
        ) else "no_signal"

    record: dict[str, Any] = {
        "stage": "D",
        "name": STAGE_LABELS["D"],
        "blocking_issues": ["#156"],
        "status": status,
        "error": None,
        "exposure_ms_actual": exposure_actual,
        "arms": arms,
        "sync_log": sync_path.name,
        "sync_entries": len(sync_lines),
        "note": (
            "受控/随机/离线/仅预训练四变体配对帧 + 同步日志；配对结构离线分析，本轮仅采集"
            "原始帧与 SNR/EKF 同步记录，不改判 #156 判据。"
        ),
    }
    write_json(stage_dir / "paired_meta.json", record)
    _journal(journal, "stage_d_paired_closed_loop", status=status,
             arms=list(PAIR_ARMS), sync_entries=len(sync_lines))
    return record


# --------------------------------------------------------------------------- #
# Stage E —— CD-Twin 闭环孪生实测集（#233）
# --------------------------------------------------------------------------- #

def stage_e_cdtwin_twin_set(
    slm: Any,
    cam: Any,
    out: Path,
    journal: Any,
    settle_s: float,
    full_scale: float,
    exposure_actual: float,
) -> dict[str, Any]:
    """CD-Twin 闭环孪生实测帧 + 同步日志（jsonl），固定孪生序列。"""
    stage_dir = out / "stage_e_cdtwin_twin_set"
    raw_dir = stage_dir / "raw"
    sync_path = stage_dir / "sync_log.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)

    sync_lines: list[dict[str, Any]] = []
    sync_seq = 0
    steps: list[dict[str, Any]] = []
    any_bad = False

    for step_index, step_name in enumerate(TWIN_SEQUENCE):
        gray = _twin_gray(step_name, DEFAULT_PANEL_SHAPE)
        gray_sha = digest(gray)
        slm.display_data(gray)
        frame = _capture_frame(cam, settle_s)
        gate = _gate_frame(frame, full_scale)
        fname = f"{step_index:02d}_{step_name}.npy"
        file_name = _save_frame(raw_dir, fname, frame)
        steps.append({
            "step": step_index,
            "pattern": step_name,
            "gray_sha256": gray_sha,
            "file": file_name,
            "gate": gate,
        })
        sync_entry = {
            "seq": sync_seq,
            "step": step_index,
            "pattern": step_name,
            "gray_sha256": gray_sha,
            "frame_file": file_name,
            "snr": gate["raw_peak_snr"],
            "peak_counts": gate["raw_peak_counts"],
            "saturated": gate["saturated"],
            "has_signal": gate["has_signal"],
            "ts_utc": utc_now(),
        }
        sync_lines.append(sync_entry)
        sync_seq += 1
        if gate["saturated"] or not gate["has_signal"]:
            any_bad = True

    with sync_path.open("w", encoding="utf-8") as handle:
        for line in sync_lines:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")

    status = "ok"
    if any_bad:
        status = "saturated" if any(s["gate"]["saturated"] for s in steps) else "no_signal"

    record: dict[str, Any] = {
        "stage": "E",
        "name": STAGE_LABELS["E"],
        "blocking_issues": ["#233"],
        "status": status,
        "error": None,
        "exposure_ms_actual": exposure_actual,
        "twin_sequence": list(TWIN_SEQUENCE),
        "steps": steps,
        "sync_log": sync_path.name,
        "sync_entries": len(sync_lines),
        "note": (
            "CD-Twin 闭环孪生实测帧 + 同步日志；孪生结构离线分析，本轮仅采集原始帧与"
            "同步记录，不改判 #233 判据。"
        ),
    }
    write_json(stage_dir / "twin_meta.json", record)
    _journal(journal, "stage_e_cdtwin_twin_set", status=status,
             steps=list(TWIN_SEQUENCE), sync_entries=len(sync_lines))
    return record


# --------------------------------------------------------------------------- #
# Plan / execute / main
# --------------------------------------------------------------------------- #

def selected_stages(args: Any) -> tuple[str, ...]:
    """按 ``--stages`` 过滤，顺序恒定；未指定或空串 ⇒ 全部 stage。未知字母报错。"""
    raw = getattr(args, "stages", "") or ""
    if isinstance(raw, str):
        letters = tuple(ch for ch in raw.upper() if not ch.isspace())
    else:
        letters = tuple(str(ch).upper() for ch in raw)
    unknown = sorted(set(letters) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"未知 stage: {unknown}")
    return tuple(letter for letter in STAGE_ORDER if not letters or letter in letters)


def gate_block_reason(letter: str, flags: Any) -> str | None:
    """前置未确认时返回阻塞原因，否则返回 ``None``（纯函数，供 plan 与测试复用）。

    B 的放行取决于 ``--b-frame``：``dark`` 要求已熄灯/遮挡，``signal`` 要求已打开
    光路；``both`` 沿用熄灯确认，但它在同一窗口内同时采暗场与信号，实测必然失败，
    只作哨兵。
    """
    if letter == "B":
        mode = str(getattr(flags, "b_frame", "both"))
        if mode == "signal":
            if bool(getattr(flags, "confirm_lights_on", False)):
                return None
            return "操作员已打开光路（stage B 信号窗口）"
        if bool(getattr(flags, "confirm_lights_off", False)):
            return None
        return STAGE_PREREQUISITES[letter]
    gate_flag = STAGE_GATE_FLAGS.get(letter)
    if gate_flag is None:
        return None
    if bool(getattr(flags, gate_flag, False)):
        return None
    return STAGE_PREREQUISITES[letter]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Deterministic plan: no clock, no hardware, no volatile fields."""
    stage_specs = [
        {
            "stage": letter,
            "name": STAGE_LABELS[letter],
            "blocking_issues": _blocking_issues(letter),
        }
        for letter in STAGE_ORDER
    ]
    return {
        "schema": PLAN_SCHEMA,
        "out": Path(args.out).as_posix(),
        "slm_number": int(args.slm_number),
        "cam_type": str(args.cam_type),
        "cam_id": int(args.cam_id),
        "exposure_ms": float(args.exposure_ms),
        "frames": int(args.frames),
        "b_frame": str(args.b_frame),
        "settle_s": float(args.settle_s),
        "wait_until_idle_s": float(args.wait_until_idle),
        "panel_shape_rows_cols": list(DEFAULT_PANEL_SHAPE),
        "native_ccd_shape_rows_cols": list(NATIVE_CCD_SHAPE),
        "bit_depth": int(BIT_DEPTH_REQUIRED),
        "full_scale": float(FULL_SCALE_REQUIRED),
        "two_pi_gray_scale": _two_pi_gray_scale(),
        "stage_order": list(STAGE_ORDER),
        "stages_selected": list(selected_stages(args)),
        "stage_prerequisites": dict(STAGE_PREREQUISITES),
        "gated_stages": list(STAGE_GATE_FLAGS),
        "probe_names": list(PROBE_NAMES),
        "pair_arms": list(PAIR_ARMS),
        "twin_sequence": list(TWIN_SEQUENCE),
        "stages": stage_specs,
        "exit_codes": {
            "ok": EXIT_OK,
            "error": EXIT_ERROR,
            "no_lock": EXIT_NO_LOCK,
        },
    }


def _blocking_issues(letter: str) -> list[str]:
    """每 stage 阻塞的 issue（用于 plan 可读性，与 §4 一致）。"""
    mapping: dict[str, list[str]] = {
        "A": ["#247"],
        "B": ["#238", "#236"],
        "C": ["#234", "#236"],
        "D": ["#156"],
        "E": ["#233"],
    }
    return mapping.get(letter, [])


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    """Acquire lock, open devices once, run all five stages, release in finally."""
    out: Path = args.out
    lock = acquire_device_lock(float(args.wait_until_idle))
    if lock is None:
        logger.error("could not acquire bench device lock within {}s", args.wait_until_idle)
        return EXIT_NO_LOCK

    slm: Any = None
    cam: Any = None
    journal: Any = None
    release: dict[str, Any] = {
        "schema": RELEASE_SCHEMA,
        "slm_closed": False,
        "camera_closed": False,
        "lock_released": False,
        "acquired_at_utc": utc_now(),
        "finished_at_utc": None,
        "released_at_utc": None,
        "elapsed_s": None,
        "status": "error",
        "error": None,
    }
    stage_records: list[dict[str, Any]] = []
    blocked_stages: list[str] = []
    started = time.monotonic()
    overall_status = "ok"

    try:
        if out.exists():
            raise RuntimeError(f"--execute 要求全新输出目录，{out} 已存在")
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "plan.json", plan)
        journal = (out / "capture_journal.jsonl").open("a", encoding="utf-8")
        _journal(journal, "lock_acquired", wait_until_idle_s=float(args.wait_until_idle))

        slm, cam, exposure_actual, full_scale, drive_board = _open_devices(args)
        _journal(
            journal,
            "devices_ready",
            slm_number=int(args.slm_number),
            drive_board_id=drive_board,
            panel_shape_rows_cols=list(DEFAULT_PANEL_SHAPE),
            cam_type=str(args.cam_type),
            cam_id=int(args.cam_id),
            exposure_ms_actual=exposure_actual,
            bit_depth=BIT_DEPTH_REQUIRED,
            full_scale=float(full_scale),
        )

        common = {
            "panel_shape": DEFAULT_PANEL_SHAPE,
            "exposure_actual": exposure_actual,
            "full_scale": full_scale,
            "drive_board": drive_board,
        }

        def _run(letter: str) -> dict[str, Any]:
            block_reason = gate_block_reason(letter, args)
            if block_reason is not None:
                return {
                    "stage": letter,
                    "name": STAGE_LABELS[letter],
                    "status": "blocked",
                    "prerequisite": block_reason,
                    "error": f"前置未确认：{block_reason}",
                    "artifacts": None,
                }
            try:
                if letter == "A":
                    return stage_a_panel_state(slm, out, journal, common)
                if letter == "B":
                    return stage_b_dark_signal(
                        slm, cam, out, journal, float(args.settle_s),
                        full_scale, exposure_actual, str(args.b_frame),
                    )
                if letter == "C":
                    return stage_c_direction_probe(
                        slm, cam, out, journal, float(args.settle_s),
                        full_scale, exposure_actual, int(args.frames),
                    )
                if letter == "D":
                    return stage_d_paired_closed_loop(
                        slm, cam, out, journal, float(args.settle_s),
                        full_scale, exposure_actual, int(args.frames),
                    )
                if letter == "E":
                    return stage_e_cdtwin_twin_set(
                        slm, cam, out, journal, float(args.settle_s),
                        full_scale, exposure_actual,
                    )
                raise RuntimeError(f"unknown stage {letter!r}")
            except Exception as exc:
                logger.exception("stage {} failed", letter)
                return {
                    "stage": letter,
                    "name": STAGE_LABELS.get(letter, letter),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "artifacts": None,
                }

        for letter in selected_stages(args):
            record = _run(letter)
            stage_records.append(record)
            if record.get("status") == "blocked":
                blocked_stages.append(letter)
            elif record.get("status") != "ok":
                overall_status = record.get("status", "error")

    except Exception as exc:
        overall_status = "error"
        release["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("unified capture aborted")
    finally:
        # 关闭设备 + 释放锁（无论成功失败）。未打开的设备一律记 False。
        for label, device in (("slm", slm), ("camera", cam)):
            if device is None:
                continue
            try:
                device.close()
                release[f"{label}_closed"] = True
            except Exception as exc:
                logger.warning("{} close failed: {}", label, exc)
                release[f"{label}_closed"] = False
        try:
            if lock is not None:
                lock.release()
                release["lock_released"] = True
        except Exception as exc:
            logger.warning("lock release failed: {}", exc)
            release["lock_released"] = False

        release["finished_at_utc"] = utc_now()
        release["released_at_utc"] = utc_now()
        release["elapsed_s"] = round(time.monotonic() - started, 3)
        release["status"] = overall_status
        write_json(out / "released.json", release)
        if journal is not None:
            _journal(journal, "released", status=overall_status)
            journal.close()
        # 汇总：逐 stage 状态 + 整体状态。
        summary = {
            "schema": SUMMARY_SCHEMA,
            "out": out.as_posix(),
            "overall_status": overall_status,
            "blocked_stages": blocked_stages,
            "stages": stage_records,
            "release": {
                "slm_closed": release["slm_closed"],
                "camera_closed": release["camera_closed"],
                "lock_released": release["lock_released"],
                "elapsed_s": release["elapsed_s"],
            },
        }
        write_json(out / "summary.json", summary)

    return EXIT_OK if overall_status == "ok" else EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统一缺数据采集：单次实验窗口合并 §4 五项缺数据。",
    )
    parser.add_argument("--out", required=True, type=Path, help="全新输出目录")
    parser.add_argument("--execute", action="store_true", help="打开硬件执行（默认只打印 plan）")
    parser.add_argument("--stages", default="ABCDE",
                        help="本次执行的 stage 子集，如 ABC 或 B（默认 ABCDE）")
    parser.add_argument("--slm-number", type=int, default=DEFAULT_SLM_NUMBER)
    parser.add_argument("--cam-type", default=DEFAULT_CAM_TYPE)
    parser.add_argument("--cam-id", type=int, default=DEFAULT_CAM_ID)
    parser.add_argument("--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help="stage C/D 每探针/每变体帧数")
    parser.add_argument(
        "--b-frame",
        choices=("both", "dark", "signal"),
        default="both",
        help="stage B 采哪一帧：dark=熄灯/遮挡窗口，signal=开光窗口，both=同窗口哨兵",
    )
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S,
                        help="SLM 显示后采集前等待秒数")
    parser.add_argument("--wait-until-idle", type=float, default=DEFAULT_WAIT_IDLE_S,
                        help="等台面锁的最长秒数")
    parser.add_argument(
        "--confirm-lights-off",
        action="store_true",
        help="确认激光已关闭/光路已遮挡，放行 stage B 的 dark 窗口（默认 blocked）",
    )
    parser.add_argument(
        "--confirm-lights-on",
        action="store_true",
        help="确认光路已打开，放行 stage B 的 signal 窗口（默认 blocked）",
    )
    parser.add_argument(
        "--confirm-magnification-alignment",
        action="store_true",
        help="确认 catA/catB/C06 倍率口径已对齐，放行 stage D（默认 blocked）",
    )
    parser.add_argument(
        "--confirm-g1-g3-g4",
        action="store_true",
        help="确认 G1+G3+G4 前置已完成，放行 stage E（默认 blocked）",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    plan = build_plan(args)

    if not args.execute:
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / "plan.json", plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("[dry-run] plan 已写入；未打开任何设备（无 --execute）。")
        print("[dry-run] 运行硬件：加 --execute 并在全新输出目录上执行。")
        return EXIT_OK

    return execute(args, plan)


if __name__ == "__main__":
    sys.exit(main())
