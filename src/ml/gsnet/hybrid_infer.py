"""Offline GSNet prediction followed by pupil-confined hybrid refinement.

Run ``python -m ml.gsnet.hybrid_infer --help``. Output phases are on the model
grid in radians, not calibrated SLM panel files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ml.gsnet.hybrid_phase import (
    CameraSampling,
    HybridPhaseOptimizer,
    camera_far_field_intensity,
    far_field_intensity,
    model_grid_aperture_size_m,
)
from ml.gsnet.model import FourierGSNet
from loguru import logger


def load_gsnet_checkpoint(path: Path) -> FourierGSNet:
    """Load the checkpoint format written by :func:`ml.gsnet.train.train_gsnet`."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("checkpoint must contain GSNet config and model_state_dict")
    config = checkpoint["config"]
    model = FourierGSNet(
        num_layers=int(config["num_layers"]),
        base_channels=int(config["base_channels"]),
        shared_weights=bool(config.get("shared_weights", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_inference(
    checkpoint: Path,
    source_path: Path,
    target_path: Path,
    *,
    center_xy: tuple[int, int],
    radius_px: int,
    steps: int = 40,
    lr: float = 0.05,
    residual_grid: int = 16,
    train_residual: bool = True,
    camera_target_path: Path | None = None,
    camera_sampling: CameraSampling | None = None,
    output_root: Path | None = None,
) -> Path:
    """Persist all inputs and candidate phases from one offline refinement."""
    if output_root is None:
        output_root = Path(__file__).resolve().parents[3] / "data/debug/gsnet_hybrid_infer"
    source = np.load(source_path, allow_pickle=False)
    target = np.load(target_path, allow_pickle=False)
    if (camera_target_path is None) != (camera_sampling is None):
        raise ValueError("camera target and sampling must be supplied together")
    camera_target = (
        np.load(camera_target_path, allow_pickle=False)
        if camera_target_path is not None else None
    )
    model = load_gsnet_checkpoint(checkpoint)
    optimizer = HybridPhaseOptimizer.from_gsnet(
        model, source, target, center_xy=center_xy, radius_px=radius_px,
        n_max=8, residual_grid=residual_grid, lr=lr,
        train_residual=train_residual,
        camera_target_intensity=camera_target,
        camera_sampling=camera_sampling,
    )
    result = optimizer.run(steps)
    out_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir.mkdir(parents=True, exist_ok=False)
    np.save(out_dir / "source_intensity.npy", source)
    np.save(out_dir / "target_intensity.npy", target)
    np.save(out_dir / "gsnet_phase_rad.npy", optimizer.phase_model.base_phase.detach().numpy())
    np.save(out_dir / "best_phase_grid_rad.npy", result.phase)
    np.save(
        out_dir / "gsnet_far_field_intensity.npy",
        far_field_intensity(
            optimizer.source, optimizer.phase_model.base_phase
        ).detach().numpy(),
    )
    np.save(
        out_dir / "best_far_field_intensity.npy",
        far_field_intensity(
            optimizer.source, torch.from_numpy(result.phase)
        ).detach().numpy(),
    )
    np.save(out_dir / "best_zernike_coefficients_rad.npy", result.coefficients)
    np.save(out_dir / "best_residual_grid_rad.npy", result.residual)
    if camera_target is not None and camera_sampling is not None:
        np.save(out_dir / "camera_target_intensity.npy", camera_target)
        np.save(
            out_dir / "gsnet_camera_far_field_intensity.npy",
            camera_far_field_intensity(
                optimizer.source, optimizer.phase_model.base_phase,
                camera_target.shape, camera_sampling,
            ).detach().numpy(),
        )
        np.save(
            out_dir / "best_camera_far_field_intensity.npy",
            camera_far_field_intensity(
                optimizer.source, torch.from_numpy(result.phase),
                camera_target.shape, camera_sampling,
            ).detach().numpy(),
        )
    (out_dir / "summary.json").write_text(
        json.dumps({
            "kind": "offline_model_grid_only",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "source_path": str(source_path.resolve()),
            "target_path": str(target_path.resolve()),
            "camera_target_path": (
                str(camera_target_path.resolve()) if camera_target_path is not None else None
            ),
            "camera_sampling": asdict(camera_sampling) if camera_sampling is not None else None,
            "grid_shape": list(source.shape),
            "pupil_center_xy": center_xy,
            "pupil_radius_px": radius_px,
            "n_max": 8,
            "residual_grid": residual_grid,
            "residual_coordinates": "pupil_local",
            "train_residual": train_residual,
            "steps": steps,
            "lr": lr,
            "initial_score": result.initial_score,
            "best_score": result.best_score,
            "history": result.history,
        }, indent=2), encoding="utf-8",
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="2D SLM-plane intensity .npy")
    parser.add_argument("--target", type=Path, required=True, help="2D far-field target intensity .npy")
    parser.add_argument("--pupil-center", required=True, help="x,y on model grid")
    parser.add_argument("--pupil-radius", type=int, required=True, help="pixels on model grid")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--residual-grid", type=int, default=16)
    parser.add_argument("--no-residual", action="store_true", help="freeze the residual for ablation")
    parser.add_argument("--camera-target", type=Path, help="full CCD-grid target .npy for physical-scale scoring")
    parser.add_argument("--aperture-size-mm", type=float, help="physical width represented by the full model grid")
    parser.add_argument("--physical-pupil-radius-px", type=float, help="pupil radius on the SLM panel")
    parser.add_argument("--panel-pixel-um", type=float, help="SLM panel pixel pitch; use with physical pupil radius")
    parser.add_argument("--wavelength-nm", type=float, default=1064.0)
    parser.add_argument("--focal-length-mm", type=float)
    parser.add_argument("--camera-pixel-um", type=float)
    parser.add_argument("--camera-center", help="fixed x,y optical-axis pixel in camera target")
    parser.add_argument("--far-field-padding", type=int, default=3)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    try:
        parts = tuple(int(value) for value in args.pupil_center.split(","))
        if len(parts) != 2:
            raise ValueError("pupil center must be x,y")
        sampling = None
        if args.camera_target is None and any(
            value is not None for value in (
                args.aperture_size_mm, args.physical_pupil_radius_px,
                args.panel_pixel_um, args.focal_length_mm,
                args.camera_pixel_um, args.camera_center,
            )
        ):
            raise ValueError("camera geometry requires --camera-target")
        if args.camera_target is not None:
            if None in (args.focal_length_mm, args.camera_pixel_um, args.camera_center):
                raise ValueError("camera target requires focal length, camera pixel and camera center")
            if args.aperture_size_mm is None:
                if None in (args.physical_pupil_radius_px, args.panel_pixel_um):
                    raise ValueError("camera target requires aperture size or panel pupil radius and pixel pitch")
                grid_shape = np.load(args.source, allow_pickle=False, mmap_mode="r").shape
                if len(grid_shape) != 2 or grid_shape[0] != grid_shape[1]:
                    raise ValueError("source must be a square model grid")
                aperture_size_m = model_grid_aperture_size_m(
                    grid_shape[0], args.pupil_radius,
                    args.physical_pupil_radius_px, args.panel_pixel_um,
                )
            else:
                if args.physical_pupil_radius_px is not None or args.panel_pixel_um is not None:
                    raise ValueError("use aperture size or pupil geometry, not both")
                aperture_size_m = args.aperture_size_mm * 1e-3
            camera_center = tuple(float(value) for value in args.camera_center.split(","))
            if len(camera_center) != 2:
                raise ValueError("camera center must be x,y")
            sampling = CameraSampling(
                aperture_size_m=aperture_size_m,
                wavelength_m=args.wavelength_nm * 1e-9,
                focal_length_m=args.focal_length_mm * 1e-3,
                camera_pixel_m=args.camera_pixel_um * 1e-6,
                center_xy=camera_center,
                far_field_padding=args.far_field_padding,
            )
        out_dir = run_inference(
            args.checkpoint, args.source, args.target,
            center_xy=parts, radius_px=args.pupil_radius,
            steps=args.steps, lr=args.lr, residual_grid=args.residual_grid,
            train_residual=not args.no_residual,
            camera_target_path=args.camera_target,
            camera_sampling=sampling,
            output_root=args.output_root,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    logger.info(out_dir)


if __name__ == "__main__":
    main()
