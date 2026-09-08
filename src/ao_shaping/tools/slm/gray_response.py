"""SLM灰度-相机最大亮度响应采集工具。

可生成灰度和最大亮度的响应曲线并保存为图片。
"""

from __future__ import annotations

import csv
import itertools
import sys
import time
from datetime import datetime
from pathlib import Path

import click
import numpy as np
from loguru import logger

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless use
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ao_shaping.drivers.ccd.miicam.driver import CameraStreamManager
from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200

FIELDNAMES = [
    "gray_value",
    "max_brightness",
    "min_brightness",
    "sum_brightness",
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


def _display_rotate_slot(
    slm: SantecSLM200,
    phase: np.ndarray,
    memory_slot: int,
    wait_time_s: float,
) -> None:
    """Write phase to a memory slot and display it.

    ``memory_slot`` should be rotated (not the same as the previous call)
    because writing + displaying the **same** slot twice in a row is a no-op
    on Santec SLM firmware — the device does not refresh the LCOS panel
    when ``display_memory(slot)`` is called for the slot already being
    displayed.
    """
    slm.write_phase(phase, memory_number=memory_slot)
    slm.display_memory(memory_slot)
    time.sleep(wait_time_s)


def _capture_brightness(
    camera: CameraStreamManager,
    n_sample: int,
    skip_first: bool,
    discard_count: int = 0,
) -> np.ndarray:
    """Capture a camera frame, optionally discarding initial frames.

    When *skip_first* is True and *n_sample* == 1, at least one frame is
    taken and thrown away so the returned image is guaranteed *fresh* (not
    from the previous SLM state).  *discard_count* can be increased if the
    camera frame buffer is known to lag.
    """
    discard = 1 if skip_first and n_sample == 1 else 0
    for _ in range(discard + discard_count):
        camera.get_numpy_image(n_sample=1)  # discarded
    return camera.get_numpy_image(n_sample=n_sample, skip_first=skip_first)


def _sin_model(x: np.ndarray, offset: float, amplitude: float, period: float, phase: float) -> np.ndarray:
    """Sinusoidal model for amplitude coupling:  y = offset + amplitude * sin(2πx/period + phase)"""
    return offset + amplitude * np.sin(2 * np.pi * x / period + phase)


def _fit_and_plot(
    gray_values: list[int],
    max_brightness: list[int],
    max_gray: int,
    csv_path: Path,
    exposure_ms: float,
    serial_number: str | None = None,
    timestamp: str | None = None,
) -> None:
    """Fit a sine curve and save a scatter + fit plot alongside the CSV."""
    if not HAS_PLOT:
        logger.warning("matplotlib/scipy not available, skipping plot generation")
        return

    x = np.array(gray_values, dtype=float)
    y = np.array(max_brightness, dtype=float)

    # Initial guess: offset=mean(y), amplitude=half_range, period=max_gray, phase=0
    p0 = [float(y.mean()), float(y.max() - y.min()) / 2, float(max_gray), 0.0]
    try:
        popt, _ = curve_fit(_sin_model, x, y, p0=p0, maxfev=5000)
        offset_fit, amp_fit, period_fit, phase_fit = popt
        x_smooth = np.linspace(0, max_gray, 500)
        y_smooth = _sin_model(x_smooth, *popt)
        fit_label = f"sin fit: period={period_fit:.1f}"
    except Exception as exc:
        logger.warning(f"Sin fit failed: {exc}, skipping fit line")
        popt = None
        x_smooth = np.linspace(0, max_gray, 500)
        y_smooth = None
        fit_label = None

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # Scatter plot of measured data
    ax.scatter(x, y, color="royalblue", s=30, zorder=3, label="Measured")

    # Sine fit curve
    if y_smooth is not None:
        ax.plot(x_smooth, y_smooth, color="crimson", linewidth=2, zorder=2, label=fit_label)

        # Annotate fit parameters
        text = (
            f"$\\mathrm{{offset}} = {offset_fit:.1f}$\n"
            f"$\\mathrm{{amplitude}} = {amp_fit:.1f}$\n"
            f"$\\mathrm{{period}} = {period_fit:.1f}$  (2π gray = {max_gray})\n"
            f"$\\mathrm{{phase}} = {phase_fit:.3f}$"
        )
        ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    # Labels and styling
    ax.set_xlabel("SLM Gray Value", fontsize=12)
    ax.set_ylabel("Max Brightness (ADU)", fontsize=12)
    title_parts = [f"SLM Amplitude Coupling Response  (exp={exposure_ms:.1f}ms)"]
    if serial_number:
        title_parts.append(f"SLM={serial_number}")
    if timestamp:
        title_parts.append(timestamp)
    ax.set_title("  |  ".join(title_parts), fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_gray)

    png_path = csv_path.with_suffix(".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Plot saved: {png_path}")


def acquire_gray_response(
    gray_step: int,
    exposure_ms: float,
    csv_path: Path,
    slm_number: int,
    miicam_id: int,
    wavelength: int,
    wait_time_s: float,
    n_sample: int,
    skip_first: bool,
    bit_depth: int,
    discard_count: int = 3,
) -> Path:
    csv_path = Path(csv_path)
    if csv_path.parent != Path(""):
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    with SantecSLM200(
        slm_number=slm_number,
        video_mode=0,
    ) as slm, CameraStreamManager(
        cam_id=miicam_id,
        exposure_time_ms=exposure_ms,
        bit_depth=bit_depth,
    ) as camera:
        # 只读当前波长，不触发 set_wavelength() → 避免 SLM 固件的 2π 校准
        wavelength_nm = slm.wavelength  # _setup_wavelength 已在 open() 中读取
        max_gray = slm.MAX_GRAYSCALE_VALUE  # 硬件最大灰度值 (1023)，非 2π 锁定值
        gray_values = _gray_values(gray_step, max_gray)

        logger.info(
            f"SLM #{slm_number}: wavelength={wavelength_nm}nm, "
            f"max_gray={max_gray}, gray_step={gray_step}, samples={len(gray_values)}"
        )
        logger.info(f"MiiCam ID={miicam_id}, exposure={exposure_ms}ms")
        logger.info(f"CSV output: {csv_path}")

        # Use list to accumulate data for plotting
        recorded_gray: list[int] = []
        recorded_max: list[int] = []

        start_time = datetime.now()
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            # Metadata comment row for quick visual inspection
            f.write(
                f"# SLM Serial: {slm._serial_number or 'unknown'}"
                f"  |  Started: {start_time.isoformat(timespec='seconds')}"
                f"  |  Wavelength: {wavelength_nm}nm"
                f"  |  Max Gray: {max_gray}"
                f"  |  Exposure: {exposure_ms}ms\n"
            )
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

            # Rotate through memory slots 3,4,5 so that consecutive writes
            # NEVER target the same slot.  Santec SLM firmware treats
            # display_memory(slot) as a no-op when the slot is already
            # being displayed, so reusing the same slot back-to-back
            # causes the LCOS panel to not refresh.
            _slot_cycle = itertools.cycle([3, 4, 5])
            for index, gray_value in enumerate(gray_values, start=1):
                phase = _flat_phase(slm, gray_value)
                slot = next(_slot_cycle)
                _display_rotate_slot(slm, phase, slot, wait_time_s)

                frame = _capture_brightness(camera, n_sample=n_sample, skip_first=skip_first, discard_count=discard_count)
                max_brightness, min_brightness, sum_brightness, frame_dtype, frame_shape = (
                    int(frame.max()),
                    int(frame.min()),
                    float(frame.sum()),
                    str(frame.dtype),
                    "x".join(str(dim) for dim in frame.shape),
                )
                writer.writerow(
                    {
                        "gray_value": gray_value,
                        "max_brightness": max_brightness,
                        "min_brightness": min_brightness,
                        "sum_brightness": f"{sum_brightness:.6f}",
                        "frame_dtype": frame_dtype,
                        "frame_shape": frame_shape,
                        "slm_max_gray": max_gray,
                        "wavelength_nm": wavelength_nm,
                        "exposure_ms": f"{exposure_ms:.6f}",
                        "miicam_id": miicam_id,
                        "slm_number": slm._serial_number,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                )

                recorded_gray.append(gray_value)
                recorded_max.append(max_brightness)

                logger.info(
                    f"[{index}/{len(gray_values)}] gray={gray_value}, "
                    f"max={max_brightness}, min={min_brightness}, "
                    f"sum={sum_brightness:.3f}, dtype={frame_dtype}, shape={frame_shape}"
                )

    # Generate plot with metadata after devices are closed
    _fit_and_plot(
        recorded_gray,
        recorded_max,
        max_gray,
        csv_path,
        exposure_ms,
        serial_number=slm._serial_number,
        timestamp=datetime.now().isoformat(timespec="seconds"),
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
    default=0.8,
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
@click.option("--wait-time-s", type=click.FloatRange(min=0.0), default=0.3, show_default=True, help="SLM 下发后等待时间 s")
@click.option("--n-sample", type=click.IntRange(min=1), default=1, show_default=True, help="每点相机平均帧数")
@click.option("--skip-first/--no-skip-first", default=True, show_default=True, help="是否跳过首帧")
@click.option("--discard-count", type=click.IntRange(min=0), default=1, show_default=True, help="采集前额外丢弃帧数 (防相机帧缓存滞后)")
@click.option("--bit-depth", type=click.Choice(["8", "16"]), default="8", show_default=True, help="MiiCam 输出位深")
def run(
    gray_step: int,
    exposure_ms: float,
    csv_path: Path,
    slm_number: int,
    miicam_id: int,
    wavelength: int,
    wait_time_s: float,
    n_sample: int,
    skip_first: bool,
    discard_count: int,
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
        wait_time_s=wait_time_s,
        n_sample=n_sample,
        skip_first=skip_first,
        discard_count=discard_count,
        bit_depth=int(bit_depth),
    )
    logger.info(f"采集完成: {output_path}")


if __name__ == "__main__":
    run()
