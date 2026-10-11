"""Unit tests for the phasor U-Net inverse-shaping model.

Covers ``src/ml/shaping/models.py``: the re-used U-Net body with a widened
two-channel head, the unbounded (no-sigmoid) output, the config-driven builder,
and the ``unit_phasor`` projection (unit magnitude for both a random and an
all-zero input, gradient flow, and the absence of any wrapped-angle conversion).

Pure ``torch`` -- no corpus, no pickles, no hardware.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.shaping.config import ShapingConfig  # noqa: E402
from ml.shaping.models import (  # noqa: E402
    PhasorUNet,
    build_phasor_unet,
    unit_phasor,
)


# ---------------------------------------------------------------------------
# PhasorUNet architecture
# ---------------------------------------------------------------------------
class TestPhasorUNetArchitecture:
    def test_output_shape_grid_64(self) -> None:
        net = PhasorUNet(in_channels=2, features=[8, 16, 32, 64], grid=64)
        out = net(torch.zeros(2, 2, 64, 64))
        assert out.shape == (2, 2, 64, 64)

    def test_output_shape_grid_32(self) -> None:
        net = PhasorUNet(in_channels=2, features=[8, 16], grid=32)
        out = net(torch.zeros(1, 2, 32, 32))
        assert out.shape == (1, 2, 32, 32)

    def test_head_has_two_channels(self) -> None:
        net = PhasorUNet(features=[8, 16])
        assert isinstance(net.final_conv, torch.nn.Conv2d)
        assert net.final_conv.out_channels == 2
        assert net.output_mode == "image"

    def test_image_mode_head_is_unbounded(self) -> None:
        # With a saturated head the output must be able to exceed 1.0: that is
        # exactly what distinguishes a linear head from a bounded (sigmoid) one.
        net = PhasorUNet(features=[8, 16])
        head = net.final_conv
        with torch.no_grad():
            head.weight.fill_(100.0)
            assert head.bias is not None  # bias defaults to True
            head.bias.fill_(100.0)
        out = net(torch.ones(1, 2, 32, 32))
        assert torch.isfinite(out).all()
        assert out.abs().max().item() > 1.0

    def test_features_and_head_channels_honour_explicit_args(self) -> None:
        net = PhasorUNet(in_channels=2, features=[8, 16], grid=32)
        assert net.features == [8, 16]
        assert net.final_conv.in_channels == 8
        assert net.grid == 32

    def test_build_from_config_matches_explicit(self) -> None:
        cfg = ShapingConfig(features=(8, 16), grid=32, in_channels=2)
        from_cfg = build_phasor_unet(config=cfg)
        explicit = build_phasor_unet(in_channels=2, features=[8, 16])

        assert from_cfg.features == explicit.features == [8, 16]
        assert from_cfg.final_conv.in_channels == explicit.final_conv.in_channels == 8
        assert from_cfg.final_conv.out_channels == 2
        assert from_cfg.grid == 32
        assert from_cfg.output_mode == "image"

        cfg_shapes = {k: tuple(v.shape) for k, v in from_cfg.state_dict().items()}
        exp_shapes = {k: tuple(v.shape) for k, v in explicit.state_dict().items()}
        assert cfg_shapes == exp_shapes

    def test_rejects_degenerate_architecture(self) -> None:
        with pytest.raises(ValueError):
            PhasorUNet(in_channels=0)
        with pytest.raises(ValueError):
            PhasorUNet(features=[8])

    def test_deterministic_under_seed(self) -> None:
        torch.manual_seed(1234)
        net_a = PhasorUNet(features=[8, 16]).eval()
        torch.manual_seed(1234)
        net_b = PhasorUNet(features=[8, 16]).eval()
        x = torch.randn(1, 2, 32, 32)
        with torch.no_grad():
            assert torch.equal(net_a(x), net_b(x))


# ---------------------------------------------------------------------------
# unit_phasor
# ---------------------------------------------------------------------------
class TestUnitPhasor:
    def test_unit_magnitude_for_random_input(self) -> None:
        raw = torch.randn(3, 2, 10, 10)
        out = unit_phasor(raw)
        assert out.shape == raw.shape
        mag = torch.hypot(out[:, 0:1], out[:, 1:2])
        assert torch.allclose(mag, torch.ones_like(mag), atol=1e-5)

    def test_zero_input_is_a_valid_phasor(self) -> None:
        raw = torch.zeros(2, 2, 8, 8)
        out = unit_phasor(raw)
        mag = torch.hypot(out[:, 0:1], out[:, 1:2])
        assert torch.allclose(mag, torch.ones_like(mag), atol=1e-5)
        # Dark cells are parked at (1, 0), not (0, 0).
        assert torch.allclose(out[:, 0], torch.ones_like(out[:, 0]))
        assert torch.allclose(out[:, 1], torch.zeros_like(out[:, 1]))

    def test_is_differentiable(self) -> None:
        raw = torch.randn(2, 2, 8, 8, requires_grad=True)
        loss = unit_phasor(raw).sum()
        loss.backward()
        assert raw.grad is not None
        assert torch.isfinite(raw.grad).all()
        assert raw.grad.abs().sum().item() > 0.0

    def test_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError):
            unit_phasor(torch.zeros(2, 3, 4, 4))
        with pytest.raises(ValueError):
            unit_phasor(torch.zeros(4))

    def test_no_wrapped_angle_conversion(self) -> None:
        module_src = Path(inspect.getfile(unit_phasor)).read_text(encoding="utf-8")
        assert "atan2" not in module_src
        assert "torch.angle" not in module_src

        func_src = inspect.getsource(unit_phasor)
        assert ".detach()" not in func_src
        assert "torch.no_grad" not in func_src
        assert ".item()" not in func_src
