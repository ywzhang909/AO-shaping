from __future__ import annotations

import json

import numpy as np
import torch

from ml.gsnet.hybrid_infer import load_gsnet_checkpoint, run_inference
from ml.gsnet.hybrid_phase import CameraSampling
from ml.gsnet.model import FourierGSNet


def test_load_shared_gsnet_checkpoint(tmp_path) -> None:
    model = FourierGSNet(num_layers=3, base_channels=4, shared_weights=True)
    checkpoint = tmp_path / "shared.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"num_layers": 3, "base_channels": 4, "shared_weights": True},
    }, checkpoint)
    loaded = load_gsnet_checkpoint(checkpoint)
    assert loaded.shared_weights
    assert len(loaded.layers) == 1
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_checkpoint_to_debug_artifacts_without_hardware(tmp_path) -> None:
    model = FourierGSNet(num_layers=1, base_channels=4)
    checkpoint = tmp_path / "fourier_gsnet_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"num_layers": 1, "base_channels": 4},
    }, checkpoint)
    loaded = load_gsnet_checkpoint(checkpoint)
    assert not loaded.training
    source = np.zeros((32, 32), dtype=np.float32)
    yy, xx = np.mgrid[:32, :32]
    source[(xx - 13) ** 2 + (yy - 18) ** 2 <= 9**2] = 1
    target = np.zeros_like(source)
    target[13:19, 13:19] = 1
    source_path, target_path = tmp_path / "source.npy", tmp_path / "target.npy"
    np.save(source_path, source)
    np.save(target_path, target)
    out_dir = run_inference(
        checkpoint, source_path, target_path,
        center_xy=(13, 18), radius_px=9, steps=2,
        residual_grid=6, output_root=tmp_path / "debug",
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "offline_model_grid_only"
    assert len(summary["checkpoint_sha256"]) == 64
    assert summary["pupil_center_xy"] == [13, 18]
    assert summary["residual_coordinates"] == "pupil_local"
    assert len(summary["history"]) == 3
    assert summary["best_score"] >= summary["initial_score"]
    assert np.load(out_dir / "best_phase_grid_rad.npy").shape == source.shape
    assert np.load(out_dir / "best_far_field_intensity.npy").shape == source.shape
    assert np.load(out_dir / "best_zernike_coefficients_rad.npy").shape == (44,)
    np.testing.assert_array_equal(np.load(out_dir / "source_intensity.npy"), source)
    pure_dir = run_inference(
        checkpoint, source_path, target_path,
        center_xy=(13, 18), radius_px=9, steps=1,
        residual_grid=6, train_residual=False, output_root=tmp_path / "pure_debug",
    )
    pure = json.loads((pure_dir / "summary.json").read_text(encoding="utf-8"))
    assert pure["train_residual"] is False
    assert not np.any(np.load(pure_dir / "best_residual_grid_rad.npy"))
    camera_target = np.zeros((40, 38), dtype=np.float32)
    camera_target[14:26, 13:25] = 1
    camera_path = tmp_path / "camera_target.npy"
    np.save(camera_path, camera_target)
    sampling = CameraSampling(
        aperture_size_m=7.2e-3, wavelength_m=1064e-9,
        focal_length_m=0.1224, camera_pixel_m=2.2e-6,
        center_xy=(19, 20), far_field_padding=3,
    )
    camera_dir = run_inference(
        checkpoint, source_path, target_path,
        center_xy=(13, 18), radius_px=9, steps=1,
        camera_target_path=camera_path, camera_sampling=sampling,
        output_root=tmp_path / "camera_debug",
    )
    camera_summary = json.loads((camera_dir / "summary.json").read_text(encoding="utf-8"))
    assert camera_summary["camera_sampling"]["far_field_padding"] == 3
    assert camera_summary["camera_target_path"] == str(camera_path.resolve())
    assert np.load(camera_dir / "best_camera_far_field_intensity.npy").shape == camera_target.shape
