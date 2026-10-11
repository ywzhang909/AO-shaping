"""Unit tests for the composite physics-informed shaping loss.

Covers ``src/ml/shaping/losses.py``: the returned term dict, the algebraic
decomposition ``total = w_sup*sup + w_phy*phy + w_smooth*smooth``, backward
through the physics term, graceful degradation when every row is excluded from
``L_phy`` (finite, never NaN), the piston-pinning behaviour of ``L_sup``, the
peak-normalisation invariance of ``L_phy``, the default-off smoothness term, and
device robustness when the forward bank and the network sit on different devices.

Pure ``torch`` -- no corpus, no pickles, no hardware.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:  # so annotations type-check; runtime torch is skipped below
    from torch import Tensor

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.shaping.config import ShapingConfig  # noqa: E402
from ml.shaping.losses import ShapingLoss  # noqa: E402
from ml.shaping.physics import ForwardBank  # noqa: E402

GRID = 32
BATCH = 3
FOVS = (64, 248, 320)


def _unit_phasor(batch: int, grid: int = GRID) -> tuple[Tensor, Tensor]:
    torch.manual_seed(1)
    angle = torch.rand(batch, grid, grid) * (2.0 * torch.pi)
    unit = torch.polar(torch.ones_like(angle), angle)
    return unit.real.unsqueeze(1), unit.imag.unsqueeze(1)


def _batch() -> dict[str, Tensor]:
    cos, sin = _unit_phasor(BATCH)
    torch.manual_seed(2)
    image = torch.rand(BATCH, 1, GRID, GRID) + 0.1
    return {
        "phase_cos": cos,
        "phase_sin": sin,
        "image": image,
        "fov_px": torch.tensor(FOVS, dtype=torch.long),
    }


def _make(w_smooth: float = 0.0) -> tuple[ShapingLoss, ShapingConfig]:
    config = ShapingConfig(grid=GRID, w_smooth=w_smooth)
    loss = ShapingLoss(config, ForwardBank(grid=GRID, device="cpu"))
    return loss, config


class TestComposition:
    def test_returned_keys(self) -> None:
        loss, _ = _make()
        result = loss(torch.rand(BATCH, 2, GRID, GRID), _batch())
        assert set(result) == {"total", "sup", "phy", "smooth", "physics_excluded"}

    def test_total_is_the_weighted_sum(self) -> None:
        loss, config = _make()
        result = loss(torch.rand(BATCH, 2, GRID, GRID), _batch())
        expected = (
            config.w_sup * result["sup"]
            + config.w_phy * result["phy"]
            + config.w_smooth * result["smooth"]
        )
        assert torch.allclose(result["total"], expected)

    def test_terms_are_finite_scalars(self) -> None:
        loss, _ = _make()
        result = loss(torch.rand(BATCH, 2, GRID, GRID), _batch())
        assert result["total"].ndim == 0
        assert torch.isfinite(result["total"])
        assert torch.isfinite(result["sup"])
        assert torch.isfinite(result["phy"])


class TestGradAndDegradation:
    def test_backward_flows_through_physics(self) -> None:
        loss, _ = _make()
        pred_raw = torch.rand(BATCH, 2, GRID, GRID, requires_grad=True)
        result = loss(pred_raw, _batch())
        result["total"].backward()
        assert pred_raw.grad is not None
        assert torch.isfinite(pred_raw.grad).all()

    def test_all_rows_excluded_is_finite_zero_physics(self) -> None:
        loss, _ = _make()
        batch = _batch()
        batch["fov_px"] = torch.full((BATCH,), 1944, dtype=torch.long)
        result = loss(torch.rand(BATCH, 2, GRID, GRID), batch)
        assert result["physics_excluded"] == BATCH
        assert float(result["phy"]) == 0.0
        assert torch.isfinite(result["total"])


class TestTermBehaviour:
    def test_supervised_term_pins_phase(self) -> None:
        loss, _ = _make()
        batch = _batch()
        pred_raw = torch.cat([batch["phase_cos"], batch["phase_sin"]], dim=1)
        result = loss(pred_raw, batch)
        assert float(result["sup"]) < 1e-6

    def test_physics_term_is_peak_normalisation_invariant(self) -> None:
        loss, _ = _make()
        pred_raw = torch.rand(BATCH, 2, GRID, GRID)
        batch = _batch()
        baseline = float(loss(pred_raw, batch)["phy"])
        batch["image"] = batch["image"] * 3.0
        scaled = float(loss(pred_raw, batch)["phy"])
        assert abs(baseline - scaled) < 1e-6


class TestSmoothness:
    def test_smooth_is_zero_and_skips_torchmetrics_by_default(self) -> None:
        loss, config = _make()
        assert config.w_smooth == 0.0
        result = loss(torch.rand(BATCH, 2, GRID, GRID), _batch())
        assert float(result["smooth"]) == 0.0

    def test_smooth_is_finite_when_enabled(self) -> None:
        pytest.importorskip("torchmetrics", reason="TV term needs torchmetrics")
        loss, _ = _make(w_smooth=0.05)
        result = loss(torch.rand(BATCH, 2, GRID, GRID), _batch())
        assert torch.isfinite(result["smooth"])
        assert float(result["smooth"]) > 0.0


class TestDeviceHandling:
    def test_terms_share_input_device(self) -> None:
        loss, _ = _make()
        pred_raw = torch.rand(BATCH, 2, GRID, GRID)
        result = loss(pred_raw, _batch())
        assert result["total"].device == pred_raw.device
        assert result["phy"].device == pred_raw.device
        assert result["sup"].device == pred_raw.device
        assert result["smooth"].device == pred_raw.device

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs a second device"
    )
    def test_cross_device_batch_is_handled(self) -> None:
        # The bank defaults to CUDA while the network/batch may stay on CPU.
        config = ShapingConfig(grid=GRID)
        loss = ShapingLoss(config, ForwardBank(grid=GRID, device="cuda"))
        pred_raw = torch.rand(BATCH, 2, GRID, GRID, device="cpu")
        result = loss(pred_raw, _batch())
        assert result["phy"].device.type == "cpu"
        assert result["total"].device.type == "cpu"
        assert torch.isfinite(result["total"])
