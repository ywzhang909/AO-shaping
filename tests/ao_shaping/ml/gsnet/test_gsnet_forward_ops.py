"""Tests for the pluggable forward operators (src/ml/gsnet/forward_ops.py).

The central contract is BIT-IDENTITY: ``FourierGSNet(..., forward_op=None)``
must produce exactly (``torch.equal``, not ``allclose``) the same output as
``forward_op=IdealFFTOperator(out_size, convention="legacy")``, so wiring an
operator in never changes the default path.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any

import numpy as np
import pytest
import torch

from ao_shaping.utils.wavefront.fraunhofer import focal_field
from ml.gsnet.forward_ops import (
    BenchTwinOperator,
    ForwardOperator,
    IdealFFTOperator,
    make_forward_operator,
)
from ml.gsnet.model import FFTLayer, FourierGSNet

def _fixed_inputs(batch: int = 2, side: int = 32, seed: int = 11) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    source = torch.rand(batch, 1, side, side)
    target = torch.zeros(batch, 1, side, side)
    target[:, :, side // 4: 3 * side // 4, side // 4: 3 * side // 4] = 1.0
    return source, target


class TestBitIdentity:
    """forward_op=None must be EXACTLY forward_op=legacy IdealFFTOperator."""

    def test_fftlayer_none_vs_legacy_operator_exact(self):
        side = 32
        source, target = _fixed_inputs(batch=1, side=side, seed=3)
        source_amp = torch.sqrt(source.clamp_min(0.0) + 1e-12)
        target_amp = torch.sqrt(target.clamp_min(0.0) + 1e-12)
        phase = torch.randn(1, 1, side, side)

        torch.manual_seed(99)
        layer_none = FFTLayer(base_channels=8)
        torch.manual_seed(99)
        layer_legacy = FFTLayer(base_channels=8, forward_op=IdealFFTOperator(side))
        layer_none.eval()
        layer_legacy.eval()

        with torch.no_grad():
            out_none = layer_none(phase, source_amp, target_amp)
            out_legacy = layer_legacy(phase, source_amp, target_amp)
        assert torch.equal(out_none, out_legacy)

    def test_network_none_vs_legacy_operator_exact(self):
        side = 32
        source, target = _fixed_inputs(batch=2, side=side, seed=7)

        torch.manual_seed(1234)
        model_none = FourierGSNet(num_layers=2, base_channels=8, shared_weights=True)
        torch.manual_seed(1234)
        model_legacy = FourierGSNet(
            num_layers=2, base_channels=8, shared_weights=True,
            forward_op=IdealFFTOperator(side, convention="legacy"),
        )
        model_none.eval()
        model_legacy.eval()

        with torch.no_grad():
            phase_none = model_none(source, target)
            phase_legacy = model_legacy(source, target)
        assert torch.equal(phase_none, phase_legacy)
        assert torch.equal(phase_none, model_none(source, target))

    def test_warm_start_legacy_matches_default_init(self):
        """Legacy warm_start == the historical angle(ifft2(ifftshift(target)))."""
        _, target = _fixed_inputs(batch=1, side=16, seed=5)
        target_amp = torch.sqrt(target.clamp_min(0.0) + 1e-12)
        op = IdealFFTOperator(16, convention="legacy")
        expected = torch.angle(
            torch.fft.ifft2(torch.fft.ifftshift(target_amp, dim=(-2, -1)))
        )
        assert torch.equal(op.warm_start(target_amp), expected)


class TestIdealFFTConventions:
    def test_legacy_forward_matches_handwritten(self):
        side = 16
        op = IdealFFTOperator(side, convention="legacy")
        amp = torch.rand(1, 1, side, side)
        phase = torch.randn(1, 1, side, side)
        expected = torch.fft.fftshift(
            torch.fft.fft2(amp * torch.exp(1j * phase)), dim=(-2, -1)
        )
        assert torch.equal(op.forward(amp, phase), expected)

    def test_canonical_forward_matches_focal_field(self):
        amp = torch.rand(2, 1, 16, 16)
        phase = torch.randn(2, 1, 16, 16)
        field = amp * torch.exp(1j * phase)
        op = IdealFFTOperator(64, convention="canonical")
        assert torch.equal(op.forward(amp, phase), focal_field(field, 64))

    def test_conventions_differ_by_checkerboard_sign(self):
        """canonical = legacy * (-1)^(k1+k2) / n on the centred even grid."""
        side = 16
        amp = torch.ones(1, 1, side, side)
        phase = torch.randn(1, 1, side, side)
        legacy = IdealFFTOperator(side, convention="legacy").forward(amp, phase)
        canonical = IdealFFTOperator(side, convention="canonical").forward(amp, phase)
        assert not torch.equal(legacy, canonical)
        ii, jj = torch.meshgrid(
            torch.arange(side), torch.arange(side), indexing="ij"
        )
        checkerboard = ((-1.0) ** (ii + jj)).to(legacy.dtype)
        scaled = legacy * checkerboard / side
        assert torch.allclose(canonical, scaled, atol=1e-5)

    def test_legacy_back_is_inverse_pair(self):
        side = 16
        op = IdealFFTOperator(side, convention="legacy")
        amp = torch.rand(1, 1, side, side)
        phase = torch.randn(1, 1, side, side)
        far = op.forward(amp, phase)
        back = op.back_propagate(far)
        expected = amp * torch.exp(1j * phase)
        assert torch.allclose(back, expected, atol=1e-4)

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError):
            IdealFFTOperator(16, convention="angular-spectrum")

    def test_nonpositive_out_size_raises(self):
        with pytest.raises(ValueError):
            IdealFFTOperator(0)

    def test_legacy_back_propagate_matches_handwritten(self):
        side = 16
        op = IdealFFTOperator(side, convention="legacy")
        far = torch.randn(1, 1, side, side, dtype=torch.complex64) + 1j * torch.randn(1, 1, side, side)
        expected = torch.fft.ifft2(torch.fft.ifftshift(far, dim=(-2, -1)))
        assert torch.equal(op.back_propagate(far), expected)

    def test_canonical_back_is_the_exact_inverse_pair(self):
        """canonical back = ifft2(ifftshift(F)) * S，是 canonical forward 的精确逆。

        canonical forward 做了 ``norm="ortho"``（除以网格边长 S），所以逆步要乘回来。
        """
        op = IdealFFTOperator(64, convention="canonical")
        far = torch.ones(1, 1, 64, 64, dtype=torch.complex64)
        back = op.back_propagate(far)
        expected = torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.ifftshift(far, dim=(-2, -1)), dim=(-2, -1), norm="ortho"
            ),
            dim=(-2, -1),
        )
        assert torch.allclose(back, expected)
        assert back.shape == (1, 1, 64, 64)


class TestOversampledGrids:
    """远场网格比瞳面网格大时（2f 台架真正需要的采样）。

    这是 issue #165 的架构前提：同尺寸 FFT 的衍射极限光斑是亚像素的、图样混叠，
    网络内部必须能看到**被解析开**的远场。
    """

    def test_pupil_side_defaults_to_out_size(self):
        op = IdealFFTOperator(64)
        assert op.pupil_side == 64
        assert op.far_side == 64

    def test_forward_pads_pupil_to_far_grid(self):
        op = IdealFFTOperator(256, convention="canonical", pupil_side=64)
        amp = torch.rand(1, 1, 64, 64)
        phase = torch.randn(1, 1, 64, 64)
        far = op.forward(amp, phase)
        assert far.shape == (1, 1, 256, 256)
        # 与规范居中 FFT 逐位相同
        expected = focal_field(amp * torch.exp(1j * phase), 256)
        assert torch.equal(far, expected)

    def test_back_propagate_crops_to_pupil_grid(self):
        op = IdealFFTOperator(256, convention="canonical", pupil_side=64)
        far = torch.randn(1, 1, 256, 256, dtype=torch.complex64)
        back = op.back_propagate(far)
        assert back.shape == (1, 1, 64, 64)

    def test_back_propagate_is_the_inverse_pair(self):
        """forward -> back 必须还原瞳面场（对规范约定，误差 < 1e-4）。"""
        op = IdealFFTOperator(256, convention="canonical", pupil_side=64)
        amp = torch.rand(1, 1, 64, 64)
        phase = torch.randn(1, 1, 64, 64)
        back = op.back_propagate(op.forward(amp, phase))
        expected = amp * torch.exp(1j * phase)
        assert torch.allclose(back, expected, atol=1e-4)

    def test_warm_start_returns_pupil_grid_phase(self):
        op = IdealFFTOperator(256, convention="canonical", pupil_side=64)
        target = torch.rand(1, 1, 256, 256)
        phase = op.warm_start(target)
        assert phase.shape == (1, 1, 64, 64)
        assert phase.dtype == torch.float32

    def test_same_grid_stays_bit_identical(self):
        """pupil_side == out_size 时，旧行为逐位不变。"""
        op = IdealFFTOperator(64, convention="legacy")
        far = torch.randn(1, 1, 64, 64, dtype=torch.complex64)
        expected = torch.fft.ifft2(torch.fft.ifftshift(far, dim=(-2, -1)))
        assert torch.equal(op.back_propagate(far), expected)

    def test_network_accepts_pupil_and_far_field_grids(self):
        """source 在瞳面网格、target 在远场网格时，网络能跑且输出仍是瞳面相位。"""
        op = IdealFFTOperator(256, convention="canonical", pupil_side=64)
        model = FourierGSNet(num_layers=2, base_channels=8, shared_weights=True, forward_op=op)
        source = torch.rand(2, 1, 64, 64)
        target = torch.rand(2, 1, 256, 256)
        with torch.no_grad():
            phase = model(source, target)
        assert phase.shape == (2, 1, 64, 64)
        assert torch.isfinite(phase).all()

    def test_network_rejects_mismatched_grids(self):
        op = IdealFFTOperator(256, convention="canonical", pupil_side=64)
        model = FourierGSNet(num_layers=1, base_channels=8, forward_op=op)
        with pytest.raises(ValueError):
            model(torch.rand(1, 1, 32, 32), torch.rand(1, 1, 256, 256))
        with pytest.raises(ValueError):
            model(torch.rand(1, 1, 64, 64), torch.rand(1, 1, 128, 128))

    def test_legacy_path_still_requires_matching_shapes(self):
        model = FourierGSNet(num_layers=1, base_channels=8)
        with pytest.raises(ValueError):
            model(torch.rand(1, 1, 64, 64), torch.rand(1, 1, 256, 256))


class TestWarmStart:
    @pytest.mark.parametrize("convention", ["legacy", "canonical"])
    def test_shape_dtype_and_range(self, convention):
        op = IdealFFTOperator(32, convention=convention)
        target_amp = torch.rand(2, 1, 32, 32)
        phase = op.warm_start(target_amp)
        assert tuple(phase.shape) == (2, 1, 32, 32)
        assert phase.dtype == torch.float32
        assert not phase.is_complex()
        assert bool((phase.abs() <= math.pi + 1e-6).all())
        assert bool(torch.isfinite(phase).all())


class TestMakeForwardOperator:
    def test_none_returns_none(self):
        assert make_forward_operator(None) is None

    def test_ideal_legacy(self):
        op = make_forward_operator("ideal-legacy", out_size=48)
        assert isinstance(op, IdealFFTOperator)
        assert op.convention == "legacy"
        assert op.out_size == 48

    def test_ideal_canonical(self):
        op = make_forward_operator("ideal-canonical", out_size=48)
        assert isinstance(op, IdealFFTOperator)
        assert op.convention == "canonical"

    def test_unknown_spec_raises(self):
        with pytest.raises(ValueError):
            make_forward_operator("fourier-optics")

    def test_bench_twin_dispatch(self):
        from ao_shaping.drivers.sim.bench_twin import BenchTwin, BenchTwinConfig

        twin = BenchTwin(BenchTwinConfig(region=32, region_span_px=900.0))
        op = make_forward_operator("bench-twin", twin=twin)
        assert isinstance(op, BenchTwinOperator)

    def test_ideal_operators_satisfy_protocol(self):
        assert isinstance(IdealFFTOperator(16), ForwardOperator)
        assert isinstance(IdealFFTOperator(16, convention="canonical"), ForwardOperator)


class TestProtocolPlumbing:
    def test_canonical_op_changes_network_output(self):
        """An operator with a different convention is a real numerical change."""
        side = 32
        source, target = _fixed_inputs(batch=1, side=side, seed=8)
        torch.manual_seed(42)
        model_none = FourierGSNet(num_layers=2, base_channels=8, shared_weights=True)
        torch.manual_seed(42)
        model_can = FourierGSNet(
            num_layers=2, base_channels=8, shared_weights=True,
            forward_op=IdealFFTOperator(side, convention="canonical"),
        )
        model_none.eval()
        model_can.eval()
        with torch.no_grad():
            p_none = model_none(source, target)
            p_can = model_can(source, target)
        assert not torch.equal(p_none, p_can)
        assert torch.isfinite(p_can).all()

    def test_op_path_is_differentiable(self):
        op = IdealFFTOperator(16, convention="canonical")
        layer = FFTLayer(base_channels=4, forward_op=op)
        phase = torch.randn(1, 1, 16, 16, requires_grad=True)
        source_amp = torch.ones(1, 1, 16, 16)
        target_amp = torch.ones(1, 1, 16, 16)
        out = layer(phase, source_amp, target_amp)
        out.sum().backward()
        assert phase.grad is not None
        assert torch.isfinite(phase.grad).all()

    def test_old_positional_signature_still_works(self):
        model = FourierGSNet(3, 8, True)
        assert model.num_layers == 3
        assert model.base_channels == 8
        assert model.shared_weights is True
        assert len(model.layers) == 1


class TestBenchTwinOperator:
    """孪生的 torch 镜像：与孪生逐位一致，且回代是精确的逆。"""

    def _twin(self, **errors: object) -> Any:
        from ao_shaping.drivers.sim.bench_twin import (
            BenchTwin,
            BenchTwinConfig,
            TwinErrorModel,
        )

        config = BenchTwinConfig(
            region=64,
            region_span_px=900.0,
            far_field_padding=8,
            beam_w0_px=250.0,
            peak_adu=100.0,
            noise_adu=0.0,
            seed=0,
            errors=TwinErrorModel(**errors),  # type: ignore[arg-type]
        )
        return BenchTwin(config)

    def test_matches_the_twin_without_errors(self) -> None:
        from ml.gsnet.forward_ops import BenchTwinOperator

        twin = self._twin()
        operator = BenchTwinOperator(twin)
        phase = np.random.default_rng(0).uniform(-np.pi, np.pi, (64, 64))
        truth = np.asarray(twin.far_field(phase), dtype=np.float64)
        far = operator.forward(None, torch.from_numpy(phase).float()[None, None])
        mine = (far.real**2 + far.imag**2)[0, 0].numpy().astype(np.float64)
        # 孪生把帧峰值归一到 peak_adu=100，算子归一的是复场（|F| 的峰值 = 1）。
        assert float(np.corrcoef(truth.ravel(), mine.ravel())[0, 1]) > 0.999
        assert float(truth.max() / mine.max()) == pytest.approx(100.0, rel=1e-3)

    def test_matches_the_twin_with_all_three_error_families(self) -> None:
        from ml.gsnet.forward_ops import BenchTwinOperator

        twin = self._twin(
            lut_gamma=1.2,
            slm_zernike={4: 0.15, 5: -0.1},
            slm_roughness_sigma_rad=0.02,
            slm_roughness_seed=3,
            defocus_m=3e-4,
            tilt_x_rad=2e-4,
            tilt_y_rad=-1e-4,
            lens_zernike={4: 0.08},
        )
        operator = BenchTwinOperator(twin)
        phase = np.random.default_rng(1).uniform(-np.pi, np.pi, (64, 64))
        truth = np.asarray(twin.far_field(phase), dtype=np.float64)
        far = operator.forward(None, torch.from_numpy(phase).float()[None, None])
        mine = (far.real**2 + far.imag**2)[0, 0].numpy().astype(np.float64)
        assert float(np.corrcoef(truth.ravel(), mine.ravel())[0, 1]) > 0.999

    def test_back_propagate_is_the_inverse_pair(self) -> None:
        """回代必须把瞳面还原 —— 少了 fftshift 就会返回全零。"""
        from ml.gsnet.forward_ops import BenchTwinOperator

        twin = self._twin()
        operator = BenchTwinOperator(twin)
        phase = np.random.default_rng(2).uniform(-np.pi, np.pi, (64, 64))
        far = operator.forward(None, torch.from_numpy(phase).float()[None, None])
        back = operator.back_propagate(far)
        assert tuple(back.shape) == (1, 1, 64, 64)
        assert float(back.abs().max()) > 0.5
        recovered = torch.angle(back)
        assert float(recovered.std()) > 0.1

    def test_warm_start_returns_a_pupil_grid_phase(self) -> None:
        from ml.gsnet.forward_ops import BenchTwinOperator

        twin = self._twin()
        operator = BenchTwinOperator(twin)
        target = torch.rand(1, 1, operator.far_side, operator.far_side)
        phase = operator.warm_start(target)
        assert tuple(phase.shape) == (1, 1, 64, 64)
        assert float(phase.std()) > 0.1
