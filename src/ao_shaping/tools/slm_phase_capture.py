"""SLM随机相位采集工具

采集SLM添加的随机相位（湍流屏或Zernike系数生成）+ Daheng相机画面 + MiiCam相机画面。

用法:
    python slm_phase_capture.py --mode turbulence --samples 10 --output data/slm_capture
    python slm_phase_capture.py --mode zernike --n-max 4 --samples 5
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import torch

from loguru import logger

from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.drivers.slm.slm_pattern_helper import PatternHelper


def _get_daheng_camera(cam_id: int, exposure_ms: int):
    """Import and create Daheng camera instance."""
    try:
        from ao_shaping.drivers.ccd.daheng import DahengCamManager

        cam = DahengCamManager(cam_id=cam_id, exposure_time_ms=exposure_ms)
        # cam.reset_window()
        cam.open()
        return cam
    except ImportError as e:
        logger.warning(f"Daheng相机不可用: {e}")
        raise
    except Exception as e:
        logger.error(f"Daheng相机初始化失败: {e}")
        raise


def _get_miicam_camera(cam_id: int, exposure_ms: int, bit_depth: int = 8):
    """Import and create MiiCam camera instance."""
    try:
        from ao_shaping.drivers.ccd.miicam_driver import CameraStreamManager

        cam = CameraStreamManager(
            cam_id=cam_id, exposure_time_ms=exposure_ms, bit_depth=bit_depth
        )
        cam.open()
        # cam.reset_window()
        return cam
    except ImportError as e:
        logger.warning(f"MiiCam相机不可用: {e}")
        raise
    except Exception as e:
        logger.error(f"MiiCam相机初始化失败: {e}")
        raise


def generate_random_turbulence_phase(
    pattern_helper: PatternHelper,
    cn2: float = 1e-14,
    length: float = 1000.0,
    pixel_size_um: float = 8.0,
    wavelength_nm: int = 1064,
) -> np.ndarray:
    """Generate a random turbulence phase screen.

    Each call produces a different random screen due to Kolmogorov spectrum
    with random phase components.
    """
    return pattern_helper.generate_turbulence_screen(
        Cn2=cn2,
        L=length,
        wavelength=wavelength_nm * 1e-9,
        pixel_size=pixel_size_um * 1e-6,
    )


def generate_random_zernike_coeffs(
    n_max: int = 4,
    radius: float | None = None,
    max_coeff: float = 1.0,
) -> dict:
    """Generate a random Zernike phase pattern with random coefficients.

    Random coefficients are drawn uniformly from [-max_coeff, max_coeff]
    for all valid (n, m) pairs up to n_max (except piston which is fixed at 1.0).
    """
    coefficients: dict[tuple[int, int], float] = {}
    for n in range(n_max + 1):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2 == 0:
                if n == 0 and m == 0:
                    coefficients[(n, m)] = 1.0  # piston
                else:
                    coefficients[(n, m)] = np.random.uniform(-max_coeff, max_coeff)

    kwargs = {"n_max": n_max, "coefficients": coefficients}
    if radius is not None:
        kwargs["radius"] = radius
    return kwargs


def capture_camera_frame(
    camera, cam_name: str, n_sample: int = 1, skip_first: bool = True
) -> np.ndarray | None:
    """Safely capture a frame from a camera, returning None on failure.

    Warns if max brightness is saturated (255 for 8-bit, 65535 for 16-bit)
    or below threshold.
    """
    try:
        frame = camera.get_numpy_image(n_sample=n_sample, skip_first=skip_first)
        max_val = int(frame.max())
        min_val = int(frame.min())
        mean_val = float(frame.mean())

        # Determine saturation threshold based on dtype
        if frame.dtype == np.uint16:
            saturation_threshold = 65535
            dark_threshold = 100
        else:
            saturation_threshold = 255
            dark_threshold = 10

        if max_val >= saturation_threshold:
            logger.warning(
                f"{cam_name} 画面过曝! max={max_val}, mean={mean_val:.1f}, min={min_val}. "
                f"建议降低曝光时间或增益。"
            )
        elif max_val < dark_threshold:
            logger.warning(
                f"{cam_name} 画面过暗! max={max_val}, mean={mean_val:.1f}, min={min_val}. "
                f"建议增加曝光时间或增益。"
            )
        else:
            logger.info(
                f"{cam_name} 采集成功: shape={frame.shape}, dtype={frame.dtype}, "
                f"max={max_val}, mean={mean_val:.1f}, min={min_val}"
            )
        return frame
    except Exception as e:
        logger.warning(f"{cam_name} 采集失败: {e}")
        return None


def save_capture(
    output_dir: Path,
    sample_idx: int,
    phase_gray: np.ndarray,
    daheng_frame: np.ndarray | None,
    miicam_frame: np.ndarray | None,
    metadata: dict,
    save_pytorch: bool = False,
) -> dict:
    """Save phase pattern and camera frames to disk.

    When save_pytorch=True (default), also saves a consolidated .pt file
    containing all tensors ready for PyTorch DataLoader consumption.
    """
    sample_dir = output_dir / f"sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Save phase pattern as CSV
    phase_csv = sample_dir / "phase.csv"
    np.savetxt(phase_csv, phase_gray, fmt="%d", delimiter=",")
    saved_files["phase"] = str(phase_csv)

    # Save phase preview as PNG (normalized to 0-255)
    phase_preview = (
        phase_gray.astype(np.float32) / max(SantecSLM200.MAX_GRAYSCALE_VALUE, 1) * 255
    ).astype(np.uint8)
    phase_png = sample_dir / "phase_preview.png"
    from PIL import Image

    Image.fromarray(phase_preview).save(phase_png)
    saved_files["phase_preview"] = str(phase_png)

    # Save Daheng camera frame
    if daheng_frame is not None:
        daheng_npy = sample_dir / "daheng_frame.npy"
        np.save(daheng_npy, daheng_frame)
        saved_files["daheng_frame"] = str(daheng_npy)

        # Also save as PNG if 2D
        if daheng_frame.ndim == 2:
            daheng_png = sample_dir / "daheng_frame.png"
            img_data = np.clip(daheng_frame, 0, 255).astype(np.uint8)
            Image.fromarray(img_data).save(daheng_png)
            saved_files["daheng_preview"] = str(daheng_png)

    # Save MiiCam camera frame
    if miicam_frame is not None:
        miicam_npy = sample_dir / "miicam_frame.npy"
        np.save(miicam_npy, miicam_frame)
        saved_files["miicam_frame"] = str(miicam_npy)

        if miicam_frame.ndim == 2:
            miicam_png = sample_dir / "miicam_frame.png"
            if miicam_frame.dtype == np.uint16:
                # Normalize 16-bit data to 0-255 for PNG preview
                img_data = (
                    (miicam_frame.astype(np.float32) / max(int(miicam_frame.max()), 1))
                    * 255
                ).astype(np.uint8)
            else:
                img_data = np.clip(miicam_frame, 0, 255).astype(np.uint8)
            Image.fromarray(img_data).save(miicam_png)
            saved_files["miicam_preview"] = str(miicam_png)

    # Save metadata
    meta_file = sample_dir / "metadata.json"
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    saved_files["metadata"] = str(meta_file)

    # Save consolidated PyTorch .pt file for training
    if save_pytorch:
        pt_file = sample_dir / "sample.pt"
        tensor_dict = {
            "phase": torch.from_numpy(phase_gray.astype(np.float32)),
            "phase_type": metadata.get("phase_type", "unknown"),
            "sample_idx": sample_idx,
        }
        if daheng_frame is not None:
            tensor_dict["daheng"] = torch.from_numpy(daheng_frame.astype(np.float32))
        if miicam_frame is not None:
            tensor_dict["miicam"] = torch.from_numpy(miicam_frame.astype(np.float32))
        torch.save(tensor_dict, pt_file)
        saved_files["pytorch"] = str(pt_file)

    logger.info(f"Sample {sample_idx} saved to {sample_dir}")
    return saved_files


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["turbulence", "zernike"]),
    default="turbulence",
    help="相位生成模式: turbulence=湍流相位屏, zernike=Zernike随机系数",
)
@click.option("--samples", "-n", default=10, help="采集样本数量 (default: 10)")
@click.option(
    "--output",
    "-o",
    default="data/slm_capture",
    help="输出目录 (default: data/slm_capture)",
)
@click.option("--slm-number", default=1, help="SLM设备编号 1-8 (default: 1)")
@click.option("--wavelength", default=1064, help="SLM工作波长 nm (default: 1064)")
@click.option("--memory-slot", default=1, help="SLM内存槽编号 1-128 (default: 1)")
# Turbulence parameters
@click.option(
    "--cn2", default=1e-14, type=float, help="折射率结构常数 Cn² (default: 1e-14)"
)
@click.option(
    "--length", "-L", default=1000.0, type=float, help="传播距离 L (m) (default: 1000)"
)
# Zernike parameters
@click.option("--n-max", default=10, type=int, help="Zernike最大径向阶数 (default: 10)")
@click.option(
    "--max-coeff", default=1.0, type=float, help="Zernike系数最大绝对值 (default: 1.0)"
)
@click.option(
    "--zernike-radius",
    default=None,
    type=float,
    help="Zernike孔径半径(像素), 默认短边一半",
)
# Daheng camera parameters
@click.option("--daheng-id", default=0, help="Daheng相机ID (default: 0)")
@click.option(
    "--daheng-exposure", default=20, help="Daheng相机曝光时间 ms (default: 20)"
)
@click.option("--no-daheng", is_flag=True, help="跳过Daheng相机采集")
# MiiCam camera parameters
@click.option("--miicam-id", default=0, help="MiiCam相机ID (default: 0)")
@click.option(
    "--miicam-exposure", default=1.0, help="MiiCam相机曝光时间 ms (default: 1.0)"
)
@click.option(
    "--miicam-bit-depth",
    default=8,
    type=click.IntRange(8, 16),
    help="MiiCam相机输出位深 8或16 (default: 8)",
)
@click.option("--no-miicam", is_flag=True, help="跳过MiiCam相机采集")
# Capture parameters
@click.option("--n-sample", default=1, help="每帧平均采样数 (default: 1)")
@click.option(
    "--skip-first", is_flag=True, default=True, help="跳过首帧 (default: True)"
)
@click.option("--interval", default=0.5, type=float, help="样本间隔秒 (default: 0.5)")
@click.option("--no-slm", is_flag=True, help="仅采集相机画面，不下发相位到SLM")
@click.option("--seed", default=None, type=int, help="随机种子 (default: None)")
@click.option(
    "-f",
    "--resume-from",
    default=None,
    type=str,
    help="从已有 global_metadata.json 继续采集 (路径)",
)
def run(
    mode: str,
    samples: int,
    output: str,
    slm_number: int,
    wavelength: int,
    memory_slot: int,
    cn2: float,
    length: float,
    n_max: int,
    max_coeff: float,
    zernike_radius: float | None,
    daheng_id: int,
    daheng_exposure: int,
    no_daheng: bool,
    miicam_id: int,
    miicam_exposure: int,
    miicam_bit_depth: int,
    no_miicam: bool,
    n_sample: int,
    skip_first: bool,
    interval: float,
    no_slm: bool,
    seed: int | None,
    resume_from: str | None,
):
    """SLM随机相位采集工具

    生成随机相位（湍流屏或Zernike），下发到SLM显示，同时采集Daheng和MiiCam相机画面。
    """
    import time

    if seed is not None:
        np.random.seed(seed)

    # Determine output directory and starting sample index
    base_output_dir = Path(output)
    start_idx = 0
    resumed_meta = None

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            logger.error(f"Resume file not found: {resume_path}")
            sys.exit(1)
        base_output_dir = resume_path.parent
        with open(resume_path, "r", encoding="utf-8") as f:
            resumed_meta = json.load(f)
        logger.info(f"从 {resume_path} 恢复采集")

        # Find existing sample directories to determine next index
        existing_dirs = sorted(base_output_dir.glob("sample_*"))
        if existing_dirs:
            last_idx = max(
                int(d.name.split("_")[1])
                for d in existing_dirs
                if d.name.startswith("sample_") and d.name.split("_")[1].isdigit()
            )
            start_idx = last_idx + 1
            logger.info(
                f"发现已有 {len(existing_dirs)} 个样本，从 sample_{start_idx:04d} 开始"
            )
    else:
        # Create a new batch folder with timestamp
        batch_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = base_output_dir / batch_name

    base_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {base_output_dir}")
    logger.info(f"采集模式: {mode}, 样本数: {samples}")

    # Initialize devices
    slm: SantecSLM200 | None = None
    daheng_cam = None
    miicam_cam = None

    # Initialize SLM (unless --no-slm)
    if not no_slm:
        try:
            logger.info(f"正在连接SLM #{slm_number}...")
            slm = SantecSLM200(
                slm_number=slm_number,
                wavelength=wavelength,
                video_mode=0,  # Memory mode
            )
            slm.open()
            logger.info(f"SLM #{slm_number} 已连接, 分辨率: {slm.Panel_Res}")
        except Exception as e:
            logger.error(f"SLM连接失败: {e}")
            logger.warning("将继续采集但不显示相位到SLM")
            slm = None

    # Initialize PatternHelper for phase generation
    if slm is not None:
        resolution = (slm.Panel_Res[0], slm.Panel_Res[1])
        bits = slm.Gray_Scale_bits
    else:
        resolution = (1920, 1200)  # Default SLM200 resolution
        bits = 10

    pattern_helper = PatternHelper(resolution=resolution, bits=bits)
    logger.info(f"PatternHelper: resolution={resolution}, bits={bits}")

    # Initialize Daheng camera
    if not no_daheng:
        try:
            logger.info(
                f"正在连接Daheng相机 ID={daheng_id}, 曝光={daheng_exposure}ms..."
            )
            daheng_cam = _get_daheng_camera(daheng_id, daheng_exposure)
            logger.info(
                f"Daheng相机已连接: {daheng_cam.cam_width}x{daheng_cam.cam_height}"
            )
        except Exception as e:
            logger.warning(f"Daheng相机不可用: {e}")
            daheng_cam = None

    # Initialize MiiCam camera
    if not no_miicam:
        try:
            logger.info(
                f"正在连接MiiCam相机 ID={miicam_id}, 曝光={miicam_exposure}ms..."
            )
            miicam_cam = _get_miicam_camera(
                miicam_id, miicam_exposure, miicam_bit_depth
            )
            logger.info(
                f"MiiCam相机已连接: {miicam_cam.cam_width}x{miicam_cam.cam_height}"
            )
        except Exception as e:
            logger.warning(f"MiiCam相机不可用: {e}")
            miicam_cam = None

    # Build global metadata
    global_meta = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "samples": samples,
        "slm_number": slm_number if slm else None,
        "wavelength": wavelength,
        "resolution": resolution,
        "bits": bits,
        "daheng": {
            "enabled": daheng_cam is not None,
            "cam_id": daheng_id,
            "exposure_ms": daheng_exposure,
        },
        "miicam": {
            "enabled": miicam_cam is not None,
            "cam_id": miicam_id,
            "exposure_ms": miicam_exposure,
            "bit_depth": miicam_bit_depth,
        },
    }
    if mode == "turbulence":
        global_meta["turbulence"] = {"Cn2": cn2, "L": length}
    else:
        global_meta["zernike"] = {
            "n_max": n_max,
            "max_coeff": max_coeff,
            "radius": zernike_radius,
        }

    # Merge with resumed metadata if resuming
    if resumed_meta is not None:
        # Preserve original timestamp and merge new samples count
        global_meta["timestamp"] = resumed_meta.get(
            "timestamp", global_meta["timestamp"]
        )
        global_meta["resumed_at"] = datetime.now().isoformat()
        global_meta["total_samples"] = resumed_meta.get("total_samples", 0) + samples

    # Save global metadata
    meta_path = base_output_dir / "global_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(global_meta, f, ensure_ascii=False, indent=2)

    # Main capture loop
    logger.info(f"开始采集 {samples} 个样本...")
    all_saved = []

    for i in range(samples):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"采集样本 {i + 1}/{samples}")

        # Generate random phase
        if mode == "turbulence":
            phase_gray = generate_random_turbulence_phase(
                pattern_helper,
                cn2=cn2,
                length=length,
                pixel_size_um=slm.Pixel_Size_um if slm else 8.0,
                wavelength_nm=wavelength,
            )
            phase_type = "turbulence"
            phase_params = {"Cn2": cn2, "L": length}
        else:
            kwargs = generate_random_zernike_coeffs(
                n_max=n_max,
                radius=zernike_radius,
                max_coeff=max_coeff,
            )
            phase_gray = pattern_helper.generate_zernike_polynomial(**kwargs)
            phase_type = "zernike"
            # Record actual coefficients used
            phase_params = kwargs
            phase_params['coefficients'] = list(phase_params['coefficients'].values())

        logger.info(
            f"相位生成完成: type={phase_type}, shape={phase_gray.shape}, "
            f"min={phase_gray.min()}, max={phase_gray.max()}"
        )

        # Display phase on SLM
        if slm is not None:
            try:
                current_slot = (memory_slot + i - 1) % 128 + 1
                slm.write_phase(phase_gray, memory_number=current_slot)
                slm.display_memory(current_slot)
                logger.info(f"相位已写入SLM内存槽 {current_slot} 并显示")
            except Exception as e:
                logger.warning(f"SLM相位显示失败: {e}")

        # Small delay for SLM to settle
        if slm is not None:
            time.sleep(0.3)

        # Capture camera frames
        daheng_frame = capture_camera_frame(
            daheng_cam, "Daheng", n_sample=n_sample, skip_first=skip_first
        )
        miicam_frame = capture_camera_frame(
            miicam_cam, "MiiCam", n_sample=n_sample, skip_first=skip_first
        )

        # Save all data
        sample_meta = {
            "sample_idx": i,
            "timestamp": datetime.now().isoformat(),
            "phase_type": phase_type,
            "phase_params": phase_params,
            "phase_shape": list(phase_gray.shape),
            "phase_min": int(phase_gray.min()),
            "phase_max": int(phase_gray.max()),
            "slm_memory_slot": (memory_slot + i - 1) % 128 + 1 if slm else None,
            "daheng": {
                "captured": daheng_frame is not None,
                "shape": list(daheng_frame.shape) if daheng_frame is not None else None,
                "dtype": str(daheng_frame.dtype) if daheng_frame is not None else None,
            },
            "miicam": {
                "captured": miicam_frame is not None,
                "shape": list(miicam_frame.shape) if miicam_frame is not None else None,
                "dtype": str(miicam_frame.dtype) if miicam_frame is not None else None,
            },
        }

        saved = save_capture(
            output_dir=base_output_dir,
            sample_idx=start_idx + i,
            phase_gray=phase_gray,
            daheng_frame=daheng_frame,
            miicam_frame=miicam_frame,
            metadata=sample_meta,
        )
        all_saved.append(saved)

        # Inter-sample interval
        if i < samples - 1:
            time.sleep(interval)

    # Cleanup
    logger.info(f"\n{'=' * 50}")
    logger.info("采集完成，正在清理设备...")

    if slm is not None:
        try:
            slm.close()
            logger.info("SLM已断开")
        except Exception as e:
            logger.warning(f"SLM断开失败: {e}")

    if daheng_cam is not None:
        try:
            daheng_cam.close()
            logger.info("Daheng相机已断开")
        except Exception as e:
            logger.warning(f"Daheng相机断开失败: {e}")

    if miicam_cam is not None:
        try:
            miicam_cam.close()
            logger.info("MiiCam相机已断开")
        except Exception as e:
            logger.warning(f"MiiCam相机断开失败: {e}")

    logger.info(f"全部 {samples} 个样本已保存到 {base_output_dir}")
    logger.info(f"全局元数据: {base_output_dir / 'global_metadata.json'}")


if __name__ == "__main__":
    run()
