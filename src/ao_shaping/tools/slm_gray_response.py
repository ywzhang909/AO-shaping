"""SLM灰度-相机最大亮度响应采集工具。"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import click
import numpy as np
from loguru import logger

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ao_shaping.drivers.ccd.miicam_driver import CameraStreamManager
from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200

FIELDNAMES = [
    "gray_value",
    "max_brightness",
    "min_brightness",
    "mean_brightness",
    "frame_dtype",
    "frame_shape",
    "slm_max_gray",
    "wavelength_nm",
    "exposure_ms",
    "miicam_id",
    "slm_number",
    "timestamp",
]


def _gray_values(gray_step: int, max_gray: int) -> list[int]:
    values = list(range(0, max_gray + 1, gray_step))
    if values[-1] != max_gray:
        values.append(max_gray)
    return values


def _flat_phase(slm: SantecSLM200, gray_value: int) -> np.ndarray:
    height, width = slm.Panel_Res[1], slm.Panel_Res[0]
    return np.full((height, width), gray_value, dtype=np.uint16)


def _capture_brightness(camera: CameraStreamManager, n_sample: int, skip_first: bool) -> tuple[int, int, float, str, str]:
    frame = camera.get_numpy_image(n_sample=n_sample, skip_first=skip_first)
    shape = "x".join(str(dim) for dim in frame.shape)
    return int(frame.max()), int(frame.min()), float(frame.mean()), str(frame.dtype), shape


def acquire_gray_response(
    gray_step: int,
    exposure_ms: float,
    csv_path: Path,
    slm_number: int,
    miicam_id: int,
    wavelength: int,
    memory_slot: int,
    wait_time_s: float,
    n_sample: int,
    skip_first: bool,
    bit_depth: int,
) -> Path:
    csv_path = Path(csv_path)
    if csv_path.parent != Path(""):
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    with SantecSLM200(
        slm_number=slm_number,
        wavelength=wavelength,
        video_mode=0,
    ) as slm, CameraStreamManager(
        cam_id=miicam_id,
        exposure_time_ms=exposure_ms,
        bit_depth=bit_depth,
    ) as camera:
        wavelength_nm, max_gray = slm.get_wavelength_info()
        gray_values = _gray_values(gray_step, max_gray)

        logger.info(
            f"SLM #{slm_number}: wavelength={wavelength_nm}nm, "
            f"max_gray={max_gray}, gray_step={gray_step}, samples={len(gray_values)}"
        )
        logger.info(f"MiiCam ID={miicam_id}, exposure={exposure_ms}ms")
        logger.info(f"CSV output: {csv_path}")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

            for index, gray_value in enumerate(gray_values, start=1):
                phase = _flat_phase(slm, gray_value)
                slm.write_phase(phase, memory_number=memory_slot)
                slm.display_memory(memory_slot)

                if wait_time_s > 0:
                    time.sleep(wait_time_s)

                max_brightness, min_brightness, mean_brightness, frame_dtype, frame_shape = (
                    _capture_brightness(camera, n_sample=n_sample, skip_first=skip_first)
                )

                writer.writerow(
                    {
                        "gray_value": gray_value,
                        "max_brightness": max_brightness,
                        "min_brightness": min_brightness,
                        "mean_brightness": f"{mean_brightness:.6f}",
                        "frame_dtype": frame_dtype,
                        "frame_shape": frame_shape,
                        "slm_max_gray": max_gray,
                        "wavelength_nm": wavelength_nm,
                        "exposure_ms": f"{exposure_ms:.6f}",
                        "miicam_id": miicam_id,
                        "slm_number": slm_number,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                )

                logger.info(
                    f"[{index}/{len(gray_values)}] gray={gray_value}, "
                    f"max={max_brightness}, min={min_brightness}, "
                    f"mean={mean_brightness:.3f}, dtype={frame_dtype}, shape={frame_shape}"
                )

    return csv_path


@click.command()
@click.option(
    "-N",
    "--gray-step",
    "gray_step",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="灰度采样间隔 N",
)
@click.option(
    "--miicam-exposure-ms",
    "--exposure-ms",
    "exposure_ms",
    type=click.FloatRange(min=0.011, max=10000.0),
    default=20.0,
    show_default=True,
    help="MiiCam 曝光时间 ms",
)
@click.option(
    "--csv-path",
    "--csv",
    "-o",
    "csv_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/slm_gray_response.csv"),
    show_default=True,
    help="CSV 输出文件路径",
)
@click.option("--slm-number", type=click.IntRange(min=1, max=8), default=1, show_default=True, help="SLM 设备编号")
@click.option("--miicam-id", type=click.IntRange(min=0), default=0, show_default=True, help="MiiCam 相机 ID")
@click.option("--wavelength", type=click.IntRange(min=450, max=1600), default=1064, show_default=True, help="SLM 工作波长 nm")
@click.option("--memory-slot", type=click.IntRange(min=1, max=128), default=1, show_default=True, help="SLM 内存槽编号")
@click.option("--wait-time-s", type=click.FloatRange(min=0.0), default=0.3, show_default=True, help="SLM 下发后等待时间 s")
@click.option("--n-sample", type=click.IntRange(min=1), default=1, show_default=True, help="每点相机平均帧数")
@click.option("--skip-first/--no-skip-first", default=True, show_default=True, help="是否跳过首帧")
@click.option("--bit-depth", type=click.Choice(["8", "16"]), default="8", show_default=True, help="MiiCam 输出位深")
def run(
    gray_step: int,
    exposure_ms: float,
    csv_path: Path,
    slm_number: int,
    miicam_id: int,
    wavelength: int,
    memory_slot: int,
    wait_time_s: float,
    n_sample: int,
    skip_first: bool,
    bit_depth: int,
) -> None:
    """以灰度间隔 N 扫描 SLM 平相位，并用 MiiCam 记录最大亮度。"""
    output_path = acquire_gray_response(
        gray_step=gray_step,
        exposure_ms=exposure_ms,
        csv_path=csv_path,
        slm_number=slm_number,
        miicam_id=miicam_id,
        wavelength=wavelength,
        memory_slot=memory_slot,
        wait_time_s=wait_time_s,
        n_sample=n_sample,
        skip_first=skip_first,
        bit_depth=int(bit_depth),
    )
    logger.info(f"采集完成: {output_path}")


if __name__ == "__main__":
    run()
