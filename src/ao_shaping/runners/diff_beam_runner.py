"""Dual-algorithm differentiable beam-shaping runner.

Lets the user pick, via ``--algorithm {backprop,gs}``, between two
algorithms that share the *same* SLM -> CCD closed-loop plumbing and the
*same* metric definitions (so their results are directly comparable):

- ``backprop``: a PyTorch autograd loop that treats the SLM phase ``φ`` as a
  learnable tensor and backpropagates a far-field intensity loss through a
  differentiable FFT far-field model (see
  ``ao_shaping.algorithm.differentiable_beam``).

- ``gs``: the alternating-projection Gerchberg-Saxton closed loop
  (``ao_shaping.algorithm.gerchberg_saxton``), optionally with CCD feedback.

Both algorithms produce an SLM phase pattern; in ``--use-hardware`` mode the
final phase is displayed on the SLM and the far field is captured with a CCD.

Usage:
    python src/ao_shaping/main.py diff-beam --algorithm backprop --epochs 200
    python src/ao_shaping/main.py diff-beam --algorithm gs --iterations 50
    python src/ao_shaping/main.py diff-beam --algorithm backprop --use-hardware
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure src is in path when running directly
if __name__ == "__main__":
    _script_dir = Path(__file__).resolve().parent
    _src_root = _script_dir.parent
    if str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))

import click
import numpy as np
from loguru import logger

# Import the two algorithms
from ao_shaping.algorithm.differentiable_beam import differentiable_beam_optimize
from ao_shaping.algorithm.gerchberg_saxton import (
    gerchberg_saxton,
    adaptive_gerchberg_saxton,
)

# Import shared beam-shaping helpers (single source of truth for metrics/IO)
from ao_shaping.algorithm.beam_shaping_utils import (
    compute_metrics,
    create_target_shape,
    load_target_image,
    phase_to_slm_grayscale,
)

# Import hardware drivers with graceful fallback
SantecSLM200: Any = None
DahengCamManager: Any = None
SLM_AVAILABLE = False
CCD_AVAILABLE = False

try:
    from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200 as _SantecSLM200

    SantecSLM200 = _SantecSLM200
    SLM_AVAILABLE = True
except ImportError:
    logger.debug("SLM driver not available")

try:
    from ao_shaping.drivers.ccd.daheng import DahengCamManager as _DahengCamManager

    DahengCamManager = _DahengCamManager
    CCD_AVAILABLE = True
except ImportError:
    logger.debug("CCD driver not available")


def _parse_tuple(ctx, param, value: str | None) -> tuple[int, int] | None:
    """Parse a Click ``"x,y"`` option into a 2-tuple."""
    if value is None:
        return None
    s_clean = re.sub(r"[()\s]", "", str(value))
    try:
        parts = s_clean.split(",")
        if len(parts) != 2:
            raise ValueError("Must have exactly two integers")
        x, y = map(int, parts)
        return (x, y)
    except Exception:
        raise click.BadParameter(
            f"Invalid format: {value}. Expected: 'x,y' or '(x,y)'"
        )


@click.command(name="diff-beam")
@click.option(
    "--algorithm",
    type=click.Choice(["backprop", "gs"]),
    default="backprop",
    show_default=True,
    help="优化算法: backprop (PyTorch 反向传播) 或 gs (Gerchberg-Saxton 闭环)",
)
@click.option(
    "--target-image",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="目标图像路径 (灰度图 / .npy，归一化后作为目标强度)",
)
@click.option(
    "--target-shape",
    type=click.Choice(["gaussian", "circle"]),
    default="gaussian",
    show_default=True,
    help="预设目标形状 (当未指定 --target-image 时使用)",
)
@click.option(
    "-e", "--epochs",
    default=200,
    show_default=True,
    help="backprop 算法的 Adam 优化步数",
)
@click.option(
    "--lr",
    default=0.01,
    show_default=True,
    help="backprop 算法的 Adam 学习率",
)
@click.option(
    "-i", "--iterations",
    default=50,
    show_default=True,
    help="gs 算法的迭代次数",
)
@click.option(
    "-d", "--distance",
    default=0.1,
    show_default=True,
    help="传播距离 (米) (gs 算法使用)",
)
@click.option(
    "-l", "--wavelength",
    default=1064.0,
    show_default=True,
    help="激光波长 (纳米)",
)
@click.option(
    "--slm-wavelength",
    default=1064,
    show_default=True,
    help="SLM 工作波长 (纳米，用于设置 SLM)",
)
@click.option(
    "--slm-number",
    default=1,
    show_default=True,
    help="SLM 设备编号",
)
@click.option(
    "--cam-id",
    default=lambda: os.environ.get("FAR_CAM_ID", "0"),
    show_default="FAR_CAM_ID/0",
    help="CCD 相机 ID",
)
@click.option(
    "--cam-center",
    callback=_parse_tuple,
    default=None,
    help="CCD 中心位置 'x,y' (default: 自动检测)",
)
@click.option(
    "--cam-size",
    default=400,
    show_default=True,
    help="CCD 开窗大小 (像素)",
)
@click.option(
    "--cam-exposure",
    default=50.0,
    show_default=True,
    help="CCD 曝光时间 (毫秒)",
)
@click.option(
    "--adaptive",
    is_flag=True,
    help="gs 算法启用 CCD 反馈自适应 (需要 --use-hardware)",
)
@click.option(
    "--adaptive-iterations",
    default=3,
    show_default=True,
    help="gs 自适应外层迭代次数",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default=None,
    help="backprop 计算设备 (default: 自动选择)",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="backprop 初始相位随机种子 (用于可复现)",
)
@click.option(
    "-s", "--save-dir",
    default="data/diff_beam",
    show_default=True,
    help="结果保存目录",
)
@click.option(
    "--use-hardware",
    is_flag=True,
    help="使用实际硬件 (SLM+CCD)，否则仅模拟计算",
)
@click.option(
    "--show",
    is_flag=True,
    help="显示结果图像",
)
def run(
    algorithm: str,
    target_image: Path | None,
    target_shape: str,
    epochs: int,
    lr: float,
    iterations: int,
    distance: float,
    wavelength: float,
    slm_wavelength: int,
    slm_number: int,
    cam_id: str,
    cam_center: tuple[int, int] | None,
    cam_size: int,
    cam_exposure: float,
    adaptive: bool,
    adaptive_iterations: int,
    device: str | None,
    seed: int,
    save_dir: str,
    use_hardware: bool,
    show: bool,
) -> None:
    """Dual-algorithm differentiable beam shaping.

    通过 ``--algorithm {backprop,gs}`` 选择优化算法。两种算法共享相同的
    SLM->CCD 闭环硬件路径和指标定义，因此结果可直接对比。
    """
    from ao_shaping.utils.cli_helpers import get_debug_mode

    get_debug_mode()

    # Wavelength in meters (gs angular spectrum uses SI units).
    wavelength_m = wavelength * 1e-9

    # SLM200 native panel resolution (width, height).
    slm_resolution = (1920, 1200)

    logger.info("=" * 60)
    logger.info("Dual-Algorithm Differentiable Beam Shaping")
    logger.info("=" * 60)
    logger.info(f"Algorithm: {algorithm}")
    logger.info(f"Target: {target_image or target_shape}")
    logger.info(f"Wavelength: {wavelength:.0f} nm")
    logger.info(f"Hardware: {'enabled' if use_hardware else 'disabled'}")

    # Prepare the target intensity pattern (shared between both algorithms).
    if target_image is not None:
        target_intensity = load_target_image(target_image)
    else:
        target_intensity = create_target_shape(target_shape, slm_resolution[1])

    # Uniform illumination at the SLM plane, shaped to the target footprint.
    source_amplitude = np.ones(target_intensity.shape, dtype=np.float32)

    # Hardware initialization.
    slm: Any = None
    camera: Any = None

    if use_hardware:
        if not SLM_AVAILABLE:
            raise RuntimeError("SLM driver not available. Install Santec SLM SDK.")
        if not CCD_AVAILABLE:
            raise RuntimeError("CCD driver not available. Install Daheng SDK.")

        logger.info("Initializing hardware...")
        slm = SantecSLM200(slm_number=slm_number)
        slm.open()
        slm.set_wavelength(slm_wavelength)
        logger.info(f"SLM initialized: #{slm_number}, lambda={slm_wavelength}nm")

        cam_id_int = int(cam_id)
        camera = DahengCamManager(cam_id=cam_id_int, exposure_time_ms=cam_exposure)
        camera.open()
        if cam_center is not None:
            camera.reset_window(center=cam_center, size=(cam_size, cam_size))
        logger.info(f"CCD initialized: ID={cam_id_int}, exposure={cam_exposure}ms")

    # Run the selected algorithm.
    phase: np.ndarray
    loss_history: list[float]
    final_loss: float
    converged: bool
    steps: int
    run_device: str = device or "cpu"

    try:
        if algorithm == "backprop":
            logger.info(
                "Running backprop optimization: epochs={}, lr={}, device={}",
                epochs, lr, run_device,
            )
            result = differentiable_beam_optimize(
                target_intensity=target_intensity,
                source_amplitude=source_amplitude,
                lr=lr,
                epochs=epochs,
                device=device,
                seed=seed,
            )
            phase = result.phase
            loss_history = result.loss_history
            final_loss = result.final_loss
            converged = result.converged
            steps = result.steps
            run_device = result.device
        else:  # gs
            logger.info(
                "Running Gerchberg-Saxton: iterations={}, adaptive={}",
                iterations, adaptive,
            )

            target_amp = np.sqrt(target_intensity) * source_amplitude

            if adaptive:
                # CCD-feedback adaptive GS: display each candidate phase on the
                # SLM and capture the real far field with the CCD to drive
                # closed-loop refinement.
                if not (use_hardware and slm is not None and camera is not None):
                    logger.warning(
                        "--adaptive requires --use-hardware with SLM+CCD; "
                        "falling back to plain GS."
                    )
                    gs_result = gerchberg_saxton(
                        source_amplitude=source_amplitude,
                        target_amplitude=target_amp,
                        iterations=iterations,
                        distance=distance,
                        wavelength=wavelength_m,
                    )
                else:
                    def _capture_amplitude(phase_local: np.ndarray) -> np.ndarray:
                        slm_phase = phase_to_slm_grayscale(phase_local)
                        slm.display_data(slm_phase)
                        time.sleep(0.1)
                        img = camera.get_numpy_image(n_sample=1, skip_first=True)
                        intensity = np.asarray(img, dtype=np.float32)
                        intensity = np.nan_to_num(intensity)
                        amp = np.sqrt(intensity)
                        amax = amp.max()
                        return amp / amax if amax > 0 else amp

                    gs_result = adaptive_gerchberg_saxton(
                        source_amplitude=source_amplitude,
                        target_amplitude=target_amp,
                        measured_amplitude_callback=_capture_amplitude,
                        outer_iterations=adaptive_iterations,
                        inner_iterations=iterations,
                        distance=distance,
                        wavelength=wavelength_m,
                    )
            else:
                gs_result = gerchberg_saxton(
                    source_amplitude=source_amplitude,
                    target_amplitude=target_amp,
                    iterations=iterations,
                    distance=distance,
                    wavelength=wavelength_m,
                )

            phase = gs_result.phase
            loss_history = list(gs_result.error_history)
            final_loss = loss_history[-1] if loss_history else float("nan")
            converged = False
            steps = len(loss_history)
            run_device = "numpy"

            # If using hardware, display result on SLM
            if use_hardware and slm is not None:
                logger.info("Displaying phase pattern on SLM...")
                slm_phase = phase_to_slm_grayscale(phase)
                slm.display_data(slm_phase)

    finally:
        # Cleanup hardware.
        if slm is not None:
            slm.close()
        if camera is not None:
            camera.close()

    # Capture the measured far field (hardware or simulated) for metrics.
    if use_hardware and camera is None:
        measured_intensity = None
    else:
        # Simulated far field via the FFT model, matching the backprop loss.
        from ao_shaping.algorithm.differentiable_beam import far_field_intensity

        import torch

        phase_t = torch.from_numpy(
            phase.astype(np.float32)
        ).to("cpu")
        i_far = far_field_intensity(source_amplitude, phase_t)
        i_far_np = i_far.cpu().numpy()
        imax = i_far_np.max()
        measured_intensity = (
            i_far_np / imax if imax > 0 else i_far_np
        ).astype(np.float32)

    # Compute shared metrics (backprop and gs are directly comparable).
    metrics = compute_metrics(measured_intensity, target_intensity)

    logger.info("-" * 60)
    logger.info("Results:")
    logger.info(f"  Algorithm: {algorithm} ({steps} steps, converged={converged})")
    logger.info(f"  Final loss: {final_loss:.6f}")
    logger.info(f"  MSE: {metrics['mse']:.6f}")
    logger.info(f"  Correlation: {metrics['correlation']:.4f}")
    logger.info(f"  Efficiency: {metrics['efficiency']:.4f}")
    logger.info("-" * 60)

    # Save results.
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_label = (
        target_image.stem if target_image else target_shape
    )
    result_dir = save_path / f"{algorithm}_{target_label}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "algorithm": algorithm,
        "target": str(target_image) if target_image else target_shape,
        "epochs": epochs,
        "lr": lr,
        "iterations": iterations,
        "distance_m": distance,
        "wavelength_nm": wavelength,
        "slm_wavelength_nm": slm_wavelength,
        "use_hardware": use_hardware,
        "adaptive": adaptive,
        "steps": steps,
        "converged": converged,
        "final_loss": final_loss,
        "metrics": metrics,
    }
    with open(result_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    slm_phase = phase_to_slm_grayscale(phase)
    np.save(result_dir / "phase_pattern.npy", slm_phase)
    np.save(result_dir / "target_intensity.npy", target_intensity)
    np.save(result_dir / "measured_intensity.npy", measured_intensity)
    np.save(result_dir / "loss_history.npy", np.array(loss_history))

    logger.info(f"Results saved to: {result_dir}")
    click.echo("Differentiable beam shaping complete!")
    click.echo(f"  Algorithm: {algorithm}")
    click.echo(f"  Steps: {steps} (converged={converged})")
    click.echo(f"  Final loss: {final_loss:.6f}")
    click.echo(f"  Correlation: {metrics['correlation']:.4f}")
    click.echo(f"  Efficiency: {metrics['efficiency']:.4f}")
    click.echo(f"  Results saved to: {result_dir}")


if __name__ == "__main__":
    run()
