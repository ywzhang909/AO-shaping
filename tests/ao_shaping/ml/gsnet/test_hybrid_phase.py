from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.fft import fft2, fftshift, ifftshift
from scipy.ndimage import map_coordinates

from ml.gsnet.hybrid_phase import (
    CameraSampling,
    HybridPhaseOptimizer,
    HybridPupilPhase,
    camera_far_field_intensity,
    far_field_intensity,
    model_grid_aperture_size_m,
    square_quality,
)
from ml.gsnet.model import FourierGSNet


def test_off_center_pupil_and_noll_eighth_order() -> None:
    model = HybridPupilPhase((40, 48), (16, 23), 10, n_max=8, residual_grid=6)
    assert model.coefficients.numel() == 44
    assert model.forward().shape == (40, 48)
    assert model.pupil_mask[23, 16] == 1
    assert model.pupil_mask[20, 40] == 0
    with torch.no_grad():
        model.coefficients[0] = 0.2
        model.residual.fill_(0.3)
    phase = model()
    assert torch.count_nonzero(phase[model.pupil_mask == 0]) == 0
    assert torch.isfinite(phase).all()


def test_both_components_receive_far_field_gradient() -> None:
    torch.manual_seed(3)
    model = HybridPupilPhase((32, 36), (14, 17), 9, residual_grid=6)
    source = model.pupil_mask.clone()
    target = torch.rand(32, 36)
    loss = (far_field_intensity(source, model()) / source.sum() - target).square().mean()
    loss.backward()
    assert model.coefficients.grad is not None
    assert model.residual.grad is not None
    assert torch.isfinite(model.coefficients.grad).all()
    assert torch.isfinite(model.residual.grad).all()
    assert model.coefficients.grad.abs().sum() > 0
    assert model.residual.grad.abs().sum() > 0


def test_residual_is_local_to_pupil_independent_of_panel_position() -> None:
    first = HybridPupilPhase((48, 60), (16, 23), 9, residual_grid=5)
    shifted = HybridPupilPhase((48, 60), (39, 23), 9, residual_grid=5)
    ramp = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    with torch.no_grad():
        first.residual.copy_(ramp)
        shifted.residual.copy_(ramp)
    np.testing.assert_allclose(
        first()[14:33, 7:26].detach().numpy(),
        shifted()[14:33, 30:49].detach().numpy(),
    )


def test_camera_sampling_matches_independent_numpy_fft_and_keeps_gradient() -> None:
    rng = np.random.default_rng(4)
    source = rng.uniform(0.2, 1.0, (16, 16)).astype(np.float32)
    phase = torch.tensor(rng.normal(0, 0.3, (16, 16)).astype(np.float32), requires_grad=True)
    sampling = CameraSampling(
        aperture_size_m=16 * 8e-6, wavelength_m=1064e-9,
        focal_length_m=0.1224, camera_pixel_m=2.2e-6,
        center_xy=(11, 12), far_field_padding=3,
    )
    actual = camera_far_field_intensity(torch.from_numpy(source), phase, (25, 23), sampling)
    field = np.sqrt(source) * np.exp(1j * phase.detach().numpy())
    far = np.abs(fftshift(fft2(ifftshift(np.pad(field, 16))))) ** 2
    yy, xx = np.mgrid[:25, :23]
    scale = sampling.camera_pixel_m / sampling.far_field_pixel_m
    expected = map_coordinates(
        far, [24 + (yy - 12) * scale, 24 + (xx - 11) * scale],
        order=1, mode="constant", cval=0,
    )
    np.testing.assert_allclose(
        (actual / actual.sum()).detach().numpy(), expected / expected.sum(), rtol=3e-5, atol=3e-6,
    )
    actual[10:15, 9:14].sum().backward()
    assert phase.grad is not None and torch.isfinite(phase.grad).all()
    assert phase.grad.abs().sum() > 0


def test_full_grid_width_is_derived_from_pupil_scale() -> None:
    assert model_grid_aperture_size_m(64, 20, 450, 8) == pytest.approx(11.52e-3)
    with pytest.raises(ValueError, match="positive"):
        model_grid_aperture_size_m(64, 0, 450, 8)


def test_optimizer_scores_camera_target_on_camera_grid() -> None:
    source = np.ones((32, 32), dtype=np.float32)
    network_target = np.zeros_like(source)
    network_target[12:20, 12:20] = 1
    camera_target = np.zeros((40, 38), dtype=np.float32)
    camera_target[14:26, 13:25] = 1
    sampling = CameraSampling(
        aperture_size_m=7.2e-3, wavelength_m=1064e-9,
        focal_length_m=0.1224, camera_pixel_m=2.2e-6,
        center_xy=(19, 20), far_field_padding=3,
    )
    optimizer = HybridPhaseOptimizer(
        source, network_target, np.zeros_like(source),
        center_xy=(16, 16), radius_px=9,
        camera_target_intensity=camera_target, camera_sampling=sampling,
    )
    result = optimizer.run(1)
    assert len(result.history) == 2
    assert np.isfinite(result.best_score)
    with pytest.raises(ValueError, match="supplied together"):
        HybridPhaseOptimizer(
            source, network_target, np.zeros_like(source),
            center_xy=(16, 16), radius_px=9,
            camera_target_intensity=camera_target,
        )


def test_rejects_pupil_outside_panel_and_invalid_base() -> None:
    with pytest.raises(ValueError, match="fit entirely"):
        HybridPupilPhase((32, 32), (2, 16), 10)
    with pytest.raises(ValueError, match="base_phase"):
        HybridPupilPhase((32, 32), (16, 16), 10, base_phase=np.zeros((16, 16)))


def test_optimizer_tracks_initial_and_best_phase() -> None:
    torch.manual_seed(0)
    shape = (32, 32)
    source = np.zeros(shape, dtype=np.float32)
    yy, xx = np.mgrid[:32, :32]
    source[(xx - 12) ** 2 + (yy - 17) ** 2 <= 9**2] = 1
    target = np.zeros(shape, dtype=np.float32)
    target[13:19, 13:19] = 1
    optimizer = HybridPhaseOptimizer(
        source, target, np.zeros(shape, dtype=np.float32),
        center_xy=(12, 17), radius_px=9, residual_grid=6,
    )
    result = optimizer.run(4)
    assert len(result.history) == 5
    assert result.best_score == max(row["score"] for row in result.history)
    assert result.best_score >= result.initial_score
    assert result.phase.shape == shape
    assert np.count_nonzero(result.phase[optimizer.phase_model.pupil_mask.numpy() == 0]) == 0
    assert result.coefficients.shape == (44,)


def test_quality_requires_target_and_optimizer_rejects_negative_intensity() -> None:
    with pytest.raises(ValueError, match="illuminated"):
        square_quality(torch.ones(8, 8), torch.zeros(8, 8))
    with pytest.raises(ValueError, match="non-negative"):
        HybridPhaseOptimizer(
            -np.ones((32, 32)), np.ones((32, 32)), np.zeros((32, 32)),
            center_xy=(16, 16), radius_px=9,
        )


def test_optimizer_can_use_frozen_gsnet_prediction() -> None:
    model = FourierGSNet(num_layers=1, base_channels=4)
    model.train()
    source = np.zeros((32, 32), dtype=np.float32)
    yy, xx = np.mgrid[:32, :32]
    source[(xx - 13) ** 2 + (yy - 18) ** 2 <= 9**2] = 1
    target = np.zeros_like(source)
    target[13:19, 13:19] = 1
    optimizer = HybridPhaseOptimizer.from_gsnet(
        model, source, target, center_xy=(13, 18), radius_px=9,
        residual_grid=6,
    )
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    assert optimizer.phase_model.base_phase.shape == (32, 32)
    assert np.isfinite(optimizer.run(1).best_score)


def test_standalone_update_records_baseline_and_disallows_later_run() -> None:
    shape = (32, 32)
    source = np.ones(shape, dtype=np.float32)
    target = np.zeros(shape, dtype=np.float32)
    target[14:18, 14:18] = 1
    optimizer = HybridPhaseOptimizer(
        source, target, np.zeros(shape, dtype=np.float32),
        center_xy=(16, 16), radius_px=9, residual_grid=6,
    )
    optimizer.update()
    assert [row["step"] for row in optimizer.history] == [0.0, 1.0]
    with pytest.raises(RuntimeError, match="before standalone"):
        optimizer.run(1)
