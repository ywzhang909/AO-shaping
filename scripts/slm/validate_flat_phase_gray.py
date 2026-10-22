"""
Validation script for flat-phase (平场) gray-level fix, plus gray→amplitude
coupling measurement (灰度→振幅耦合曲线, 供 fit_slm_amplitude_coupling.py 消费).

Usage (on hardware):
    python scripts/slm/validate_flat_phase_gray.py                     # quick 5-point test (MiiCam, 默认)
    python scripts/slm/validate_flat_phase_gray.py --scan --gray-step 30  # full-range scan
    # Daheng 实验系统: 采耦合曲线并写 CSV (周期 = 一个 2π, 覆盖 0..993)
    python scripts/slm/validate_flat_phase_gray.py --cam-type daheng --cam-id 0 \
        --scan --gray-step 30 --exposure-ms 0.8 --csv data/slm_gray_response_daheng.csv

CSV 列名与 ``ao_shaping/tools/slm/calibration/gray_response.py`` 写出的格式兼容
(含 ``gray_value`` / ``max_brightness`` / ``sum_brightness`` / ``slm_max_gray``),
可直接喂给 ``scripts/slm/fit_slm_amplitude_coupling.py --csv <该文件>``。
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
from loguru import logger

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# 相机 / SLM 驱动一律在 run() 内延迟导入: 无 SDK 的机器上本模块必须仍可
# import (仓库约定)。模块顶层只允许 stdlib / numpy / click / loguru。

# CSV 列 = gray_response.py 的必需列 (gray_value / max_brightness / min_brightness /
# sum_brightness / slm_max_gray / wavelength_nm / exposure_ms / timestamp) 的超集;
# bucket_peak / bucket_sum 是冻结 0 级 ROI 内的峰值/桶能量 (物理正确的耦合观测量)。
CSV_FIELDNAMES = [
    "gray_value",
    "max_brightness",
    "min_brightness",
    "sum_brightness",
    "bucket_peak",
    "bucket_sum",
    "zero_order_x",
    "zero_order_y",
    "frame_dtype",
    "frame_shape",
    "slm_max_gray",
    "wavelength_nm",
    "exposure_ms",
    "cam_type",
    "cam_id",
    "slm_serial",
    "timestamp",
]


def _flat_phase(slm: Any, gray_value: int) -> np.ndarray:
    height, width = slm.Panel_Res[1], slm.Panel_Res[0]
    return np.full((height, width), gray_value, dtype=np.uint16)


def _grab_frame(camera: Any) -> np.ndarray:
    """discard 由调用方负责; 这里取一帧并转 float64 (桶求和需要浮点累加)。"""
    return np.asarray(camera.get_numpy_image(n_sample=1), dtype=np.float64)


def _zero_order_roi(
    frame: np.ndarray, radius: int
) -> tuple[int, int, int, int, int, int]:
    """``(px, py, y0, y1, x0, x1)`` —— 去噪平场帧上按 argmax 定位 0 级并给出 ROI。

    0 级 = 帧内全局最大 (2f 傅里叶光路的光轴落点), 不是相机几何中心。
    定位前先减中值并截断负值 (读噪声对称, 不减会让 argmax 落在噪声上)。
    """
    denoised = np.clip(frame - np.median(frame), 0.0, None)
    py, px = np.unravel_index(int(np.argmax(denoised)), denoised.shape)
    y0, y1 = max(py - radius, 0), min(py + radius + 1, frame.shape[0])
    x0, x1 = max(px - radius, 0), min(px + radius + 1, frame.shape[1])
    return int(px), int(py), int(y0), int(y1), int(x0), int(x1)


def _capture_brightness(camera: Any, discard: int, roi: tuple[int, int, int, int, int, int] | None) -> dict:
    frame = None
    for _ in range(max(discard, 0) + 1):
        frame = _grab_frame(camera)
    assert frame is not None
    raw = np.asarray(frame)
    stats = {
        "max": int(frame.max()),
        "min": int(frame.min()),
        "mean": float(frame.mean()),
        "sum": float(frame.sum()),
        "frame_dtype": str(raw.dtype),
        "frame_shape": "x".join(str(dim) for dim in raw.shape),
    }
    if roi is not None:
        px, py, y0, y1, x0, x1 = roi
        stats["bucket_peak"] = float(frame[py, px])
        stats["bucket_sum"] = float(frame[y0:y1, x0:x1].sum())
    return stats


@click.command()
@click.option("--slm-number", type=click.IntRange(1, 8), default=1, show_default=True)
@click.option(
    "--cam-type",
    type=click.Choice(["miicam", "daheng"]),
    default="miicam",
    show_default=True,
    help="相机类型 (daheng 走 open_camera 工厂)",
)
@click.option("--miicam-id", type=click.IntRange(0), default=0, show_default=True)
@click.option(
    "--cam-id",
    type=click.IntRange(0),
    default=None,
    help="相机 ID (--cam-type daheng 时使用; miicam 仍用 --miicam-id)",
)
@click.option("--wavelength", type=click.IntRange(450, 1600), default=1064, show_default=True)
@click.option("--exposure-ms", type=float, default=3.0, show_default=True)
@click.option("--wait-time-s", type=float, default=0.3, show_default=True)
@click.option("--discard-count", type=int, default=3, show_default=True)
@click.option("--bit-depth", type=click.Choice(["8", "16"]), default="8", show_default=True)
@click.option("--scan", is_flag=True, default=False, help="Scan full range [0..2π_gray]")
@click.option("--gray-step", type=int, default=50, help="Step for scan mode")
@click.option("--roi-radius", type=int, default=30, show_default=True, help="0 级 ROI 半径 px")
@click.option(
    "--metric",
    type=click.Choice(["bucket", "sum", "max"]),
    default="bucket",
    show_default=True,
    help="汇总 rel_spread 所用观测量 (bucket = 冻结 0 级桶能量, 物理正确)",
)
@click.option("--csv", "csv_path", type=Path, default=None, help="把测得曲线写成 CSV (供 fit_slm_amplitude_coupling.py)")
def run(
    slm_number: int,
    cam_type: str,
    miicam_id: int,
    cam_id: int | None,
    wavelength: int,
    exposure_ms: float,
    wait_time_s: float,
    discard_count: int,
    bit_depth: int,
    scan: bool,
    gray_step: int,
    roi_radius: int,
    metric: str,
    csv_path: Path | None,
) -> None:
    """Test flat-phase gray values and verify camera sees different brightness."""

    # 延迟导入: 无 SDK 机器上本模块仍可 import (仓库约定)。
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.utils.image.hardware_utils import open_camera

    # --miicam-id 保持原语义; --cam-id 显式给出时优先 (两种相机通用)。
    device_id = cam_id if cam_id is not None else miicam_id

    with Santec(
        slm_number=slm_number,
        wavelength=wavelength,
        video_mode=0,
    ) as slm:
        camera = open_camera(
            cam_type, device_id, exposure_ms, bit_depth=int(bit_depth)
        )
        try:
            wl, g_2pi = slm.get_wavelength_info()
            g_pi = g_2pi // 2
            g_pi2 = g_2pi // 4
            g_3pi2 = 3 * g_2pi // 4
            try:
                serial = slm.get_serial_number()
            except Exception as exc:  # noqa: BLE001 - 序列号只是元数据, 读不到不致命
                serial = None
                logger.warning(f"读取 SLM 序列号失败: {exc}")
            max_gray = int(slm.MAX_GRAYSCALE_VALUE)  # 10-bit 满量程 (1023), 非 2π 灰度数

            if scan:
                gray_values = list(range(0, g_2pi + 1, gray_step))
                if gray_values[-1] != g_2pi:
                    gray_values.append(g_2pi)
                mode_str = f"scan (step={gray_step}, {len(gray_values)} pts)"
            else:
                gray_values = [0, g_pi2, g_pi, g_3pi2, g_2pi, 1023]
                mode_str = f"quick: 0, {g_pi2}(π/2), {g_pi}(π), {g_3pi2}(3π/2), {g_2pi}(2π), 1023"

            logger.info(f"SLM: wl={wl}nm, 2π_gray={g_2pi}, max_gray={max_gray}, serial={serial}")
            logger.info(f"Camera: type={cam_type} id={device_id}")
            logger.info(f"Mode: {mode_str} | exp={exposure_ms}ms wait={wait_time_s}s discard={discard_count}")
            logger.info(f"{'gray':>6} {'max':>6} {'min':>6} {'mean':>10} {'sum':>14} {'bucket':>14}")
            logger.info("-" * 70)

            # 0 级 ROI 只在 gray=0 的平场帧上定位一次, 整条扫描期间**冻结**。
            # 逐灰度重新 argmax 会跟着光斑游走, 桶能量随 ROI 移动而非随振幅变化,
            # 会把耦合曲线伪造出来 (仓库反模式: 每轮重新定位目标 ROI)。
            roi: tuple[int, int, int, int, int, int] | None = None
            results = []
            rows: list[dict] = []
            csv_file = None
            writer = None
            if csv_path is not None:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                csv_file = csv_path.open("w", newline="", encoding="utf-8")
                csv_file.write(
                    f"# SLM Serial: {serial or 'unknown'}  |  Started: "
                    f"{datetime.now().isoformat(timespec='seconds')}  |  Wavelength: {wl}nm"
                    f"  |  2pi Gray: {g_2pi}  |  Max Gray: {max_gray}"
                    f"  |  Exposure: {exposure_ms}ms  |  Cam: {cam_type}#{device_id}\n"
                )
                writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()

            try:
                for gv in gray_values:
                    slm.display_data(_flat_phase(slm, gv), wait_time_s)
                    if roi is None:
                        # 平场 (gray=0) 帧: 减中值去读噪声后 argmax 定位 0 级一次。
                        probe = _grab_frame(camera)
                        for _ in range(max(discard_count, 0)):
                            probe = _grab_frame(camera)
                        roi = _zero_order_roi(probe, roi_radius)
                        logger.info(
                            f"0 级 ROI 定位一次并冻结: ({roi[0]},{roi[1]}) radius={roi_radius}"
                        )
                    stats = _capture_brightness(camera, discard=discard_count, roi=roi)
                    results.append((gv, stats))
                    logger.info(
                        f"{gv:>6} {stats['max']:>6} {stats['min']:>6} "
                        f"{stats['mean']:>10.3f} {stats['sum']:>14.0f} "
                        f"{stats.get('bucket_sum', float('nan')):>14.1f}"
                    )
                    if writer is not None:
                        row = {
                            "gray_value": gv,
                            "max_brightness": stats["max"],
                            "min_brightness": stats["min"],
                            "sum_brightness": f"{stats['sum']:.6f}",
                            "bucket_peak": f"{stats['bucket_peak']:.6f}",
                            "bucket_sum": f"{stats['bucket_sum']:.6f}",
                            "zero_order_x": roi[0],
                            "zero_order_y": roi[1],
                            "frame_dtype": stats["frame_dtype"],
                            "frame_shape": stats["frame_shape"],
                            "slm_max_gray": max_gray,
                            "wavelength_nm": wl,
                            "exposure_ms": f"{exposure_ms:.6f}",
                            "cam_type": cam_type,
                            "cam_id": device_id,
                            "slm_serial": serial or "",
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        }
                        writer.writerow(row)
                        rows.append(row)
            finally:
                if csv_file is not None:
                    csv_file.close()

            logger.info("-" * 70)
            distinct = len(set(s["max"] for _, s in results))
            if distinct >= 2:
                logger.info(f"✅ {distinct} distinct brightness levels detected.")
                if distinct >= 5:
                    logger.info("✅ Excellent dynamic range across phase.")
            else:
                logger.warning("❌ All identical. Try --exposure-ms 30 --wait-time-s 1.0 --discard-count 5")

            # ── 中文汇总: 灰度范围/步长/点数 + 所选取观测量实测 rel_spread ──
            series = {
                name: np.asarray([s[key] for _, s in results], dtype=float)
                for name, key in (
                    ("bucket", "bucket_sum"),
                    ("sum", "sum"),
                    ("max", "max"),
                )
            }
            mean_sel = float(series[metric].mean())
            rel_spread_sel = (
                (float(series[metric].max()) - float(series[metric].min())) / mean_sel
                if mean_sel != 0.0
                else float("nan")
            )
            logger.info(
                f"汇总: 灰度范围 {gray_values[0]}..{gray_values[-1]}, 步长 {gray_step}, "
                f"采样点 {len(gray_values)}"
            )
            logger.info(
                f"实测 rel_spread = (max-min)/mean: bucket = "
                f"{(series['bucket'].max() - series['bucket'].min()) / (series['bucket'].mean() or 1.0):.4f}, "
                f"sum = {(series['sum'].max() - series['sum'].min()) / (series['sum'].mean() or 1.0):.4f}, "
                f"max = {(series['max'].max() - series['max'].min()) / (series['max'].mean() or 1.0):.4f} "
                f"(所选取观测量 --metric {metric}: {rel_spread_sel:.4f}; 判据 > 0.15 即有调制)"
            )
            if csv_path is not None:
                logger.info(f"CSV 已写出: {csv_path} ({len(rows)} 行)")
                logger.info(f"  peak (0 级峰值) 序列: {[float(s['bucket_peak']) for _, s in results]}")
                logger.info(f"  bucket (0 级桶能量) 序列: {[float(s['bucket_sum']) for _, s in results]}")
                logger.info(
                    "  离线拟合: python scripts/slm/fit_slm_amplitude_coupling.py "
                    f"--csv {csv_path} --metric sum --levels {g_2pi}"
                )
        finally:
            camera.close()


if __name__ == "__main__":
    run()
