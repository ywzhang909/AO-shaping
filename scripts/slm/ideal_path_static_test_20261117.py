#!/usr/bin/env python
"""理想光路静态整形实验测试 (2026-11-17).

用**静态整形页同源**的 ``generate_phase_radians``（纯 numpy，模型前向）生成相位，
经 Santec SLM 下发，Daheng CCD **全幅**采集实测远场，逐图案记录。

观测来源标记为 ``model_forward``（未观测，理想光路假设）—— 与静态整形页在无 CCD
时的实际行为一致。本测试的目的正是检验这一假设：模型前向假设此前已被实测证伪
（模型预期方形平顶，实测为点状聚焦），故每帧都显式标注 ``observed=false``，
不允许静默当作已观测光路。

数据落盘 ``data/debug/ideal_path_static_test/<ts>/``，含：
- ``phase_<label>.png``   相位（弧度，包裹到 [0,2π) 显示）
- ``sent_<label>.png``    实际下发到 SLM 的灰度（``create_phase_from_array`` 返回值）
- ``ccd_<label>.png``     CCD 实测远场（归一化显示，原始值存 NPZ）
- ``compare_<label>.png`` 三联对比（相位 / 下发灰度 / CCD 实测）
- ``pattern_<label>.npz`` phase + sent_gray + ccd 原始数组
- ``overview.png``        全部图案 CCD 实测总览
- ``optical_path_record.json`` 标准光路记录（ao-optical-path-record/v1）
- ``summary.json``        逐图案元数据与峰值指标

用法::

    .venv\\Scripts\\python.exe scripts/slm/ideal_path_static_test_20261117.py \\
        --session ideal_path_static_test --wait-until-idle 900
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ao_shaping.drivers.ccd.common import create_camera
from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.gui.slm.pattern_specs import generate_phase_radians
from ao_shaping.utils.io.optical_path_record import build_optical_path_record
from ao_shaping.utils.slm.gray_scale import configured_max_grayscale
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper
from ao_shaping.tools.kernels import device_mutex

# 面板分辨率与静态整形页同源 (_DEFAULT_PANEL_RES)，位深同源 (_DEFAULT_BITS)。
PANEL_RES: tuple[int, int] = (1920, 1200)  # (width, height)
PANEL_BITS: int = 10
SLM_PANEL_W, SLM_PANEL_H = PANEL_RES

# 测试图案集：(pattern_type, params, 输出标签, 说明)。
# params 一律用 UI 量纲（generate_phase_radians 内部按 _UNIT_SCALE 换算）：
#   period/lens_size/order 为像素或整数；phase_range 以 π 为单位；
#   focal_length 为 mm、wavelength 为 nm、pixel_size 为 um。
PATTERNS: list[tuple[str, dict[str, Any], str, str]] = [
    (
        "linear_grating",
        {"period": 40, "phase_range": 2.0},
        "linear_grating_p40",
        "线性闪耀光栅：模型预期离散衍射级",
    ),
    (
        "checkerboard",
        {"period": 80},
        "checkerboard_p80",
        "棋盘格：模型预期二维四重衍射",
    ),
    (
        "lens",
        {"focal_length": 125.0, "wavelength": 1064.0, "pixel_size": 8.0},
        "lens_f125",
        "透镜聚焦：模型预期方形平顶，历史实测为点状聚焦（假设被证伪的对照）",
    ),
    (
        "circular_grating",
        {"radius": 400.0, "phase_range": 2.0},
        "circular_grating_r400",
        "圆形光栅：模型预期同心环级",
    ),
    (
        "dammann",
        {"order": 3, "fill_factor": 0.5},
        "dammann_o3",
        "达曼光栅：模型预期均匀多级",
    ),
]


def _save_png(arr: np.ndarray, path: Path, title: str, *, vmax: float | None = None) -> None:
    """把数组存成灰度 PNG（matplotlib，与探测脚本同款）。"""
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(arr, cmap="gray", aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if vmax is not None and vmax > 0:
        im.set_clim(0.0, float(vmax))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _display_norm(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """把采集帧归一化到 0-255 用于显示，返回 (显示图, 原始峰值)。"""
    peak = float(np.max(arr)) if arr.size else 0.0
    if peak <= 0:
        return np.zeros_like(arr, dtype=np.uint8), peak
    return (arr.astype(np.float64) / peak * 255.0).astype(np.uint8), peak


def _metrics(arr: np.ndarray) -> dict[str, Any]:
    """记录 CCD 帧的简单峰值指标（原始 counts，不折算灰度）。"""
    if arr.size == 0:
        return {"peak": 0, "peak_loc": [None, None], "mean": 0.0}
    peak = float(np.max(arr))
    idx = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return {
        "peak": peak,
        "peak_loc": [int(idx[1]), int(idx[0])],
        "mean": float(np.mean(arr)),
    }


def _collect_device_info(slm: Any, camera: Any) -> dict[str, Any]:
    """收集设备标识（序列号等），失败不致命（与探测脚本同口径）。"""
    info: dict[str, Any] = {}
    try:
        info["slm_serial"] = slm.serial_number
    except Exception:  # 设备信息拿不到不影响测试
        info["slm_serial"] = None
    try:
        info["slm_video_mode"] = int(getattr(slm, "video_mode", 0))
    except Exception:
        info["slm_video_mode"] = None
    try:
        info["slm_wavelength_nm"] = getattr(slm, "wavelength", None)
    except Exception:
        info["slm_wavelength_nm"] = None
    for attr in ("serial_number", "serial", "sn"):
        value = getattr(camera, attr, None)
        if isinstance(value, str) and value:
            info["cam_serial"] = value
            break
    else:
        info["cam_serial"] = None
    try:
        info["cam_pixel_size_um"] = float(getattr(camera, "pixel_size_um", 0.0) or 0.0)
    except Exception:
        info["cam_pixel_size_um"] = None
    try:
        info["cam_bit_depth"] = getattr(camera, "bit_depth", None)
    except Exception:
        info["cam_bit_depth"] = None
    return info


def _run(args: argparse.Namespace) -> int:
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ideal_path] 输出目录: {out_dir}")
    print("[ideal_path] 观测来源: model_forward (未观测, 理想光路假设)")

    # 设备互斥：持锁到采集结束（SLM USB 句柄独占，只等空闲就放锁会被中途抢走）。
    verdict = device_mutex.wait_until_idle(
        float(args.wait_until_idle),
        slm_number=int(args.slm_number),
        cam_id=int(args.cam_id),
        cam_type=str(args.cam_type),
        exposure_ms=float(args.exposure_ms),
    )
    if verdict.lock is None:
        print(f"[ideal_path] 设备等不来: {verdict.reason}", file=sys.stderr)
        return 2
    lock = verdict.lock
    acquired_at = now.isoformat(timespec="seconds")
    holder = lock.holder()
    holder_desc = f"pid={holder.pid} {holder.cmdline}" if holder else "(持有者元数据未读到)"
    print(f"[ideal_path] 已取得设备锁 ({holder_desc})")

    helper = PatternHelper(PANEL_RES, bits=PANEL_BITS)
    results: list[dict[str, Any]] = []
    frames_grid: list[np.ndarray] = []
    frames_labels: list[str] = []

    try:
        with Santec(
            slm_number=int(args.slm_number),
            wavelength=int(args.slm_wavelength),
            video_mode=0,  # 恒用内存模式：DVI 模式 open() 会挂死
        ) as slm:
            with create_camera(
                str(args.cam_type),
                cam_id=int(args.cam_id),
                exposure_time_ms=float(args.exposure_ms),
            ) as camera:
                device_info = _collect_device_info(slm, camera)
                print(
                    f"[ideal_path] 设备就绪: SLM #{args.slm_number} "
                    f"(sn={device_info.get('slm_serial')}), "
                    f"相机 {args.cam_type} #{args.cam_id} "
                    f"(sn={device_info.get('cam_serial')}), "
                    f"曝光 {args.exposure_ms} ms"
                )

                for pattern_type, params, label, note in PATTERNS:
                    print(f"[ideal_path] 图案 {label} ({pattern_type}) ...")
                    phase = generate_phase_radians(helper, pattern_type, params)
                    phase = np.nan_to_num(
                        np.asarray(phase, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
                    )

                    # 下发链路与静态整形页 _SendTask 完全同源。
                    slm.set_phase_overlay(None)
                    sent_gray = slm.create_phase_from_array(phase)
                    slm.display_data(sent_gray)

                    # SLM 更新 + 光路 settle 后再采集。
                    time.sleep(float(args.settle_s))
                    frame = np.asarray(camera.get_numpy_image(n_sample=1))

                    # 保存：相位（包裹显示）、下发灰度、CCD 实测。
                    phase_show = np.mod(phase, 2.0 * np.pi)
                    _save_png(phase_show, out_dir / f"phase_{label}.png",
                              f"phase {label} (rad, wrapped)")
                    _save_png(
                        np.asarray(sent_gray), out_dir / f"sent_{label}.png",
                        f"sent gray {label}",
                        vmax=float(configured_max_grayscale()),
                    )
                    frame_show, peak = _display_norm(frame)
                    _save_png(frame_show, out_dir / f"ccd_{label}.png",
                              f"CCD measured {label}")

                    np.savez_compressed(
                        out_dir / f"pattern_{label}.npz",
                        phase=phase,
                        sent_gray=np.asarray(sent_gray),
                        ccd=frame,
                    )

                    # 三联对比图。
                    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))
                    for ax, arr, t in zip(
                        axes,
                        (phase_show, np.asarray(sent_gray), frame_show),
                        ("phase (rad)", "sent gray", "CCD measured"),
                    ):
                        im = ax.imshow(arr, cmap="gray", aspect="equal")
                        ax.set_title(f"{t} · {label}", fontsize=9)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.suptitle(note, fontsize=10)
                    fig.tight_layout()
                    fig.savefig(out_dir / f"compare_{label}.png", dpi=140)
                    plt.close(fig)

                    m = _metrics(frame)
                    results.append(
                        {
                            "label": label,
                            "pattern_type": pattern_type,
                            "params": params,
                            "note": note,
                            "exposure_ms": float(args.exposure_ms),
                            "ccd_shape": [int(frame.shape[1]), int(frame.shape[0])],
                            "ccd_peak_counts": m["peak"],
                            "ccd_peak_loc": m["peak_loc"],
                            "ccd_mean_counts": m["mean"],
                            "optical_path_source": "model_forward",
                            "observed": False,
                            "files": {
                                "phase": f"phase_{label}.png",
                                "sent": f"sent_{label}.png",
                                "ccd": f"ccd_{label}.png",
                                "compare": f"compare_{label}.png",
                                "npz": f"pattern_{label}.npz",
                            },
                        }
                    )
                    frames_grid.append(frame_show)
                    frames_labels.append(label)
                    print(
                        f"[ideal_path]   peak={m['peak']:.0f} "
                        f"loc={m['peak_loc']} mean={m['mean']:.1f}"
                    )

                # CCD 实测总览（供报告与用户手册直接引用）。
                if frames_grid:
                    fig, axes = plt.subplots(
                        1, len(frames_grid), figsize=(4.0 * len(frames_grid), 4.2)
                    )
                    if len(frames_grid) == 1:
                        axes = [axes]
                    for ax, arr, lab in zip(axes, frames_grid, frames_labels):
                        ax.imshow(arr, cmap="gray", aspect="equal")
                        ax.set_title(lab, fontsize=9)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    fig.suptitle("CCD measured far-field (ideal / model-forward path)", fontsize=11)
                    fig.tight_layout()
                    fig.savefig(out_dir / "overview.png", dpi=140)
                    plt.close(fig)

        released_at = datetime.now().isoformat(timespec="seconds")
        print("[ideal_path] 设备已正常释放")
    finally:
        lock.release()

    # 标准光路记录 (ao-optical-path-record/v1)。
    record = build_optical_path_record(
        run_id=f"{stamp}_ideal_path_static_test",
        date=now.strftime("%Y-%m-%d"),
        operator_note=(
            "理想光路静态整形实验测试：generate_phase_radians 模型前向生成相位，"
            "SLM 下发 + CCD 全幅采集。观测来源 model_forward（未观测），"
            "用于检验模型前向假设（历史实测已证伪：点状聚焦 vs 模型方形平顶）。"
        ),
        devices={
            "slm": {
                "model": "Santec SLM-200",
                "slm_number": int(args.slm_number),
                "wavelength_nm": int(args.slm_wavelength),
                "two_pi_grayscale": int(configured_max_grayscale()),
                "panel_res": [SLM_PANEL_W, SLM_PANEL_H],
                "open_mode": "memory (video_mode=0)",
            },
            "camera": {
                "model": "Daheng MER2-507-23GM",
                "bit_depth": device_info.get("cam_bit_depth"),
                "exposure_used_ms": float(args.exposure_ms),
                "pixel_size_um": device_info.get("cam_pixel_size_um"),
                "averaging_frames": 1,
            },
        },
        optical_path={
            "config_id": f"ideal_path_static_{args.cam_type}_{args.slm_wavelength}nm",
            "aperture": "full-frame (no physical ROI; algorithm crop in post)",
            "probe_status": "not_probed_this_run (reuse probe_optical_path_validity output)",
        },
        exposure_check={
            "peak_counts_at_used_exposure": max(
                (r["ccd_peak_counts"] for r in results), default=0.0
            ),
            "saturated": None,
        },
        occupancy={
            "acquired_at": acquired_at,
            "released_at": released_at,
            "closed_normally": True,
        },
        reasons={
            "devices.slm.drive_board_id": "本测试不单独读驱动板 ID（见探测脚本记录）",
            "devices.camera.serial": "相机序列号经 _collect_device_info 采集",
            "exposure_check.saturated": "未做饱和判定（固定曝光，记录原始峰值供复盘）",
        },
    )
    (out_dir / "optical_path_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "run_id": f"{stamp}_ideal_path_static_test",
        "date": now.strftime("%Y-%m-%d"),
        "optical_path_source": "model_forward",
        "observed": False,
        "panel_res": [SLM_PANEL_W, SLM_PANEL_H],
        "panel_bits": PANEL_BITS,
        "exposure_ms": float(args.exposure_ms),
        "settle_s": float(args.settle_s),
        "device_info": device_info,
        "patterns": results,
        "note": (
            "模型前向假设此前已被实测证伪：模型预期方形平顶，实测为点状聚焦。"
            "本批数据用于复核该结论并在报告/手册中附实测图。"
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ideal_path] 完成: {out_dir}")
    print(f"[ideal_path] 记录: {out_dir / 'optical_path_record.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="ideal_path_static_test")
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--slm-wavelength", type=int, default=1064)
    parser.add_argument("--cam-type", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    # 探测 (probe_optical_path_validity) 实测: 0.05 ms 峰值 188 未饱和,
    # 0.2 ms 起即饱和 (255)。故默认用 0.05 ms, 避免全幅饱和丢形态。
    parser.add_argument("--exposure-ms", type=float, default=0.05)
    parser.add_argument("--settle-s", type=float, default=0.3)
    parser.add_argument("--wait-until-idle", type=float, default=900.0)
    parser.add_argument(
        "--out-root",
        default=str(Path("data/debug/ideal_path_static_test")),
        help="输出根目录（下按时间戳建子目录）",
    )
    return _run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
