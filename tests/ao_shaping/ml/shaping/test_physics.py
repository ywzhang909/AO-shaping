"""Unit tests for the per-FOV forward physics bank.

Covers ``src/ml/shaping/physics.py``: the bucket construction (one calibrated
model per supported FOV), the padding calibration seam, the dual input form
(``(B, 2, g, g)`` phasor channels vs ``(B, g, g)`` angles), the exclusion of
uncalibrated FOVs, the peak-normalised image output, and that the pass stays
differentiable w.r.t. its input while its own parameters are frozen.

Pure ``torch`` -- no corpus, no pickles, no hardware.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:  # so annotations type-check; runtime torch is skipped below
    from torch import Tensor

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.shaping.config import UNKNOWN_FOV
from ml.shaping.physics import ForwardBank
from ml.zernike.models import (
    NO_VALID_PADDING_FOV,
    recommended_padding,
)

GRID = 32
SUPPORTED = (64, 248, 320)


def _bank() -> ForwardBank:
    return ForwardBank(grid=GRID, device="cpu")


def _angle(batch: int) -> Tensor:
    torch.manual_seed(0)
    return torch.rand(batch, GRID, GRID) * (2.0 * torch.pi)


def _as_phasor(angle: Tensor) -> Tensor:
    unit = torch.polar(torch.ones_like(angle), angle)
    return torch.cat([unit.real.unsqueeze(1), unit.imag.unsqueeze(1)], dim=1)


class TestBucketConstruction:
    def test_builds_one_model_per_supported_fov(self) -> None:
        bank = _bank()
        for fov in SUPPORTED:
            assert bank.supports(fov)

    def test_uncalibrated_fovs_are_not_supported(self) -> None:
        bank = _bank()
        assert not bank.supports(None)
        assert not bank.supports(UNKNOWN_FOV)
        for fov in NO_VALID_PADDING_FOV:
            assert not bank.supports(fov)

    def test_padding_comes_from_recommended_padding(self) -> None:
        bank = _bank()
        for fov in SUPPORTED:
            assert bank.padding_for(fov) == recommended_padding(fov)
        assert bank.padding_for(1944) == recommended_padding(1944)

    def test_non_positive_grid_rejected(self) -> None:
        with pytest.raises(ValueError):
            ForwardBank(grid=0, device="cpu")


class TestForward:
    def test_phasor_input_shape_and_no_exclusion(self) -> None:
        bank = _bank()
        phase = _as_phasor(_angle(2))
        pred, excluded = bank.forward(phase, 64)
        assert pred.shape == (2, 1, GRID, GRID)
        assert not bool(excluded.any())

    def test_angle_and_phasor_forms_agree(self) -> None:
        bank = _bank()
        angle = _angle(2)
        out_angle, _ = bank.forward(angle, 320)
        out_phasor, _ = bank.forward(_as_phasor(angle), 320)
        assert torch.allclose(out_angle, out_phasor, atol=1e-5)

    def test_supported_output_is_peak_normalised(self) -> None:
        bank = _bank()
        pred, _ = bank.forward(_as_phasor(_angle(3)), [64, 248, 320])
        assert pred.shape == (3, 1, GRID, GRID)
        assert torch.isfinite(pred).all()
        assert bool((pred >= 0).all())
        for row in range(3):
            assert torch.allclose(pred[row].amax(), torch.tensor(1.0), atol=1e-4)

    def test_uncalibrated_rows_are_zero_and_flagged(self) -> None:
        bank = _bank()
        pred, excluded = bank.forward(_as_phasor(_angle(3)), [64, 1944, UNKNOWN_FOV])
        assert excluded.tolist() == [False, True, True]
        assert torch.count_nonzero(pred[1]) == 0
        assert torch.count_nonzero(pred[2]) == 0

    def test_scalar_fov_broadcasts_over_batch(self) -> None:
        bank = _bank()
        pred, excluded = bank.forward(_as_phasor(_angle(3)), 248)
        assert pred.shape[0] == 3
        assert not bool(excluded.any())

    def test_invalid_shapes_raise(self) -> None:
        bank = _bank()
        with pytest.raises(ValueError):
            bank.forward(torch.zeros(1, 3, GRID, GRID), 64)
        with pytest.raises(ValueError):
            bank.forward(torch.zeros(1, GRID), 64)
        with pytest.raises(ValueError):
            bank.forward(_as_phasor(_angle(1)), [64, 248])


class TestAutograd:
    def test_gradient_flows_to_input_but_not_parameters(self) -> None:
        bank = _bank()
        phase = torch.zeros(1, GRID, GRID, requires_grad=True)
        pred, _ = bank.forward(phase, 64)
        pred.sum().backward()
        assert phase.grad is not None
        assert torch.isfinite(phase.grad).all()

    def test_bucket_models_are_frozen(self) -> None:
        bank = _bank()
        for model in bank._models.values():
            for param in model.parameters():
                assert not param.requires_grad
