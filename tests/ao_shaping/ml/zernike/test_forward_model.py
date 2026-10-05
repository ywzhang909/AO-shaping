"""Unit tests for the learned coefficient -> far-field forward model.

Covers `src/ml/zernike/forward_model.py`: the `peak_normalize` output contract,
the raw (anti-sigmoid) output head, the conv decoder vs the MLP baseline, the
reachability validation, and a tiny overfit smoke test.

The load-bearing test here is `TestAntiSigmoidHead`: it is the regression guard
that stops somebody from "simplifying" `ZernikeCoeffConvNet` into
`ml.phase.unet.UNetGenerator`, whose image head is literally
`self.sigmoid(self.final_conv(x))` (src/ml/phase/unet.py line 129) and therefore
cannot emit a signed image.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:  # so annotations type-check; runtime torch is skipped below
    from torch import Tensor

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

import torch.nn as nn  # noqa: E402

from ml.zernike.forward_model import (  # noqa: E402
    DEFAULT_N_COEFFS,
    ZernikeCoeffConfig,
    ZernikeCoeffConvNet,
    ZernikeCoeffMLP,
    build_forward_model,
    count_parameters,
    peak_normalize,
)

#: Coefficient scales walked by the anti-sigmoid test. ``0.0617`` is the measured
#: ``max|c|`` of the real corpus and ``0.0142`` its rms; the larger values exist
#: to drive the linear head hard negative without relying on a lucky init.
_COEFF_SCALES = (1e-3, 1.4e-2, 6.17e-2, 1.0, 50.0)


def _coeffs(batch: int = 4, seed: int = 0, scale: float = 1.0) -> Tensor:
    """Random coefficient batch at the corpus's real magnitude scale.

    Args:
        batch: Number of samples.
        seed: RNG seed, so every test is deterministic.
        scale: Multiplier on the corpus rms (0.0142 rad).

    Returns:
        ``(batch, DEFAULT_N_COEFFS)`` float32 tensor.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, DEFAULT_N_COEFFS, generator=generator) * (scale * 0.0142)


def _config(**overrides: object) -> ZernikeCoeffConfig:
    """Default config with ``overrides`` applied (the config is frozen)."""
    return dataclasses.replace(ZernikeCoeffConfig(), **overrides)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------
class TestPeakNormalize:
    """`peak_normalize` is THE output contract and must mirror `_normalize`."""

    def test_per_sample_max_is_exactly_one(self) -> None:
        frame = torch.randn(5, 1, 16, 16).abs() + 0.1
        out = peak_normalize(frame)
        maxima = out.amax(dim=(-2, -1))
        assert tuple(maxima.shape) == (5, 1)
        assert torch.allclose(maxima, torch.ones(5, 1), atol=1e-6)
        # Exactly 1.0, not merely close: the divisor is the sample's own max.
        assert torch.equal(maxima, torch.ones(5, 1))

    def test_values_stay_in_unit_interval(self) -> None:
        frame = torch.rand(3, 1, 8, 8) * 7.0 + 0.3
        out = peak_normalize(frame)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0 + 1e-6

    def test_reduction_is_per_sample_not_per_batch(self) -> None:
        """Samples must be scaled independently; a shared divisor would be a bug.

        The two samples are scaled by ~50x different amounts, so a per-batch
        normalisation would leave one of them far from 1.0.
        """
        frame = torch.stack([torch.full((1, 4, 4), 0.01), torch.full((1, 4, 4), 0.5)])
        out = peak_normalize(frame)
        assert torch.allclose(out.amax(dim=(-2, -1)).flatten(), torch.ones(2), atol=1e-6)

    @pytest.mark.parametrize("shape", [(3, 8, 8), (2, 1, 8, 8), (2, 3, 5, 5)])
    def test_accepts_3d_and_4d(self, shape: tuple[int, ...]) -> None:
        frame = torch.rand(*shape) + 0.05
        out = peak_normalize(frame)
        assert tuple(out.shape) == shape
        # The reduction always runs over the last two axes only, so channels (if
        # any) survive as a separate dimension with their own per-sample max.
        maxima = out.amax(dim=(-2, -1))
        assert torch.allclose(maxima, torch.ones_like(maxima), atol=1e-6)

    def test_all_equal_input_does_not_divide_by_zero(self) -> None:
        """A flat frame has scale 0; the eps floor must keep it finite."""
        flat = torch.zeros(2, 1, 8, 8)
        out = peak_normalize(flat)
        assert torch.isfinite(out).all()
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        negative_flat = torch.full((2, 1, 8, 8), -3.0)
        out_neg = peak_normalize(negative_flat)
        assert torch.isfinite(out_neg).all()

    def test_zero_scale_with_eps_argument(self) -> None:
        out = peak_normalize(torch.zeros(1, 1, 4, 4), eps=1e-6)
        assert torch.isfinite(out).all()
        assert torch.equal(out, torch.zeros(1, 1, 4, 4))

    def test_negative_values_keep_their_sign(self) -> None:
        """Negatives must survive the division; clamping them away is not this function.

        Sign is only preserved when the sample max is positive. When the whole
        sample is negative the max is negative, so dividing by it flips every
        value -- a documented consequence of *not* clipping.
        """
        frame = torch.tensor([[[[-1.0, 2.0], [4.0, -0.5]]]])
        out = peak_normalize(frame)
        assert out[0, 0, 0, 0].item() < 0.0
        assert out[0, 0, 0, 1].item() == pytest.approx(0.5)
        assert out[0, 0, 1, 0].item() == pytest.approx(1.0)

    def test_matches_the_physics_model_normaliser(self) -> None:
        """Byte-level parity with `ZernikeAmpModel._normalize`'s "peak" branch.

        That branch is `observable / torch.clamp(amax, min=1e-12)`, so the two
        must agree exactly on the same input -- otherwise the learned arm and
        the physics arm would be scored on different scales.
        """
        from ml.zernike.models import _EPS as physics_eps

        assert physics_eps == 1e-12
        frame = torch.randn(2, 1, 16, 16)
        reference = frame / torch.clamp(frame.amax(dim=(-2, -1), keepdim=True), min=1e-12)
        assert torch.equal(peak_normalize(frame), reference)


# ---------------------------------------------------------------------------
# Shape / dtype / device
# ---------------------------------------------------------------------------
class TestShapeContract:
    """Both arms map ``(B, 136) -> (B, 1, 64, 64)`` in float32."""

    @pytest.mark.parametrize(
        "architecture", ["conv", "mlp"], ids=["conv", "mlp"]
    )
    def test_default_config_output_shape(self, architecture: str) -> None:
        model = build_forward_model(_config(architecture=architecture))
        out = model(_coeffs(batch=4))
        assert tuple(out.shape) == (4, 1, 64, 64)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("architecture", ["conv", "mlp"], ids=["conv", "mlp"])
    def test_batch_size_is_flexible(self, architecture: str) -> None:
        model = build_forward_model(_config(architecture=architecture))
        for batch in (1, 3, 16):
            assert tuple(model(_coeffs(batch=batch)).shape) == (batch, 1, 64, 64)

    @pytest.mark.parametrize("architecture", ["conv", "mlp"], ids=["conv", "mlp"])
    def test_follows_the_module_device(self, architecture: str) -> None:
        model = build_forward_model(_config(architecture=architecture)).to("cpu")
        out = model(_coeffs(batch=2))
        assert out.device.type == "cpu"
        # And the dtype/device survive a half-precision-free round trip.
        assert out.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("architecture", ["conv", "mlp"], ids=["conv", "mlp"])
    def test_runs_on_cuda(self, architecture: str) -> None:
        model = build_forward_model(_config(architecture=architecture)).to("cuda")
        out = model(_coeffs(batch=8).to("cuda"))
        assert out.device.type == "cuda"
        assert tuple(out.shape) == (8, 1, 64, 64)
        assert out.dtype == torch.float32

    @pytest.mark.parametrize(
        "architecture", ["conv", "mlp"], ids=["conv", "mlp"]
    )
    def test_custom_grid_and_widths(self, architecture: str) -> None:
        """A smaller grid must be reachable too, not just the 64 default."""
        overrides: dict[str, object] = {"n_coeffs": 36, "grid": 16, "hidden": 64}
        if architecture == "conv":
            # 4 stages x scale 2 from 1x1 -> 16 px; the default 4x4 bottleneck
            # would overshoot to 64.
            overrides["bottleneck"] = 1
            overrides["features"] = (64, 32, 16, 8)
        model = build_forward_model(_config(architecture=architecture, **overrides))
        out = model(torch.randn(2, 36))
        assert tuple(out.shape) == (2, 1, 16, 16)

    @pytest.mark.parametrize(
        "architecture", ["conv", "mlp"], ids=["conv", "mlp"]
    )
    def test_rejects_malformed_coefficients(self, architecture: str) -> None:
        model = build_forward_model(_config(architecture=architecture))
        with pytest.raises(ValueError, match="n_coeffs"):
            model(torch.randn(2, 36))
        with pytest.raises(ValueError, match=r"\(B, n_coeffs\)"):
            model(torch.randn(2, 1, DEFAULT_N_COEFFS))
        with pytest.raises(ValueError, match="floating point"):
            model(torch.zeros(2, DEFAULT_N_COEFFS, dtype=torch.int64))
        with pytest.raises(TypeError, match="torch.Tensor"):
            model([[0.0] * DEFAULT_N_COEFFS])


# ---------------------------------------------------------------------------
# THE anti-sigmoid regression guard
# ---------------------------------------------------------------------------
class TestAntiSigmoidHead:
    """The output head is RAW. This is the load-bearing test of the module.

    `UNetGenerator`'s image head is `self.sigmoid(self.final_conv(x))`
    (src/ml/phase/unet.py line 129): bounded to (0, 1), unable to represent 0
    or 1. Reusing it here would silently saturate the dynamic range, so a raw
    signed output must be provable.
    """

    @pytest.mark.parametrize(
        ("architecture", "factory"),
        [("conv", ZernikeCoeffConvNet), ("mlp", ZernikeCoeffMLP)],
        ids=["conv", "mlp"],
    )
    def test_raw_output_contains_negative_values(
        self, architecture: str, factory: type
    ) -> None:
        model = factory(_config(architecture=architecture))
        negatives_seen = 0
        for scale in _COEFF_SCALES:
            with torch.no_grad():
                raw = model(_coeffs(batch=4, seed=1, scale=scale))
            assert raw.dtype == torch.float32
            assert torch.isfinite(raw).all()
            if bool((raw < 0).any()):
                negatives_seen += 1
        assert negatives_seen > 0, (
            "raw head never produced a negative value over 5 coefficient scales; "
            "a bounded activation (sigmoid/tanh/softmax) has crept into the head"
        )

    @pytest.mark.parametrize(
        ("architecture", "factory"),
        [("conv", ZernikeCoeffConvNet), ("mlp", ZernikeCoeffMLP)],
        ids=["conv", "mlp"],
    )
    def test_head_contains_no_activation_module(
        self, architecture: str, factory: type
    ) -> None:
        """Structural restatement: no Sigmoid/Tanh/Softmax anywhere in the graph."""
        model = factory(_config(architecture=architecture))
        forbidden = (nn.Sigmoid, nn.Tanh, nn.Softmax, nn.Hardsigmoid, nn.Hardswish)
        found = [type(m).__name__ for m in model.modules() if isinstance(m, forbidden)]
        assert found == []

    @pytest.mark.parametrize(
        ("architecture", "factory"),
        [("conv", ZernikeCoeffConvNet), ("mlp", ZernikeCoeffMLP)],
        ids=["conv", "mlp"],
    )
    def test_normalised_max_is_exactly_one_per_sample(
        self, architecture: str, factory: type
    ) -> None:
        model = factory(_config(architecture=architecture))
        with torch.no_grad():
            raw = model(_coeffs(batch=4, seed=2, scale=50.0))
        assert bool((raw < 0).any())
        normalised = peak_normalize(raw)
        maxima = normalised.amax(dim=(-2, -1))
        assert tuple(maxima.shape) == (4, 1)
        assert torch.allclose(maxima, torch.ones(4, 1), atol=1e-6)
        assert torch.isfinite(normalised).all()
        # `min >= 0` is asserted on a non-negative frame instead, because a raw
        # signed head cannot satisfy it by construction -- the model's own output
        # is not required to be a physical (non-negative) intensity. The physical
        # case is covered by TestPeakNormalize::test_values_stay_in_unit_interval.
        physical = torch.rand(4, 1, 64, 64) + 0.01
        assert peak_normalize(physical).min().item() >= 0.0

    def test_denormalized_helper_is_peak_normalised(self) -> None:
        model = ZernikeCoeffConvNet(_config())
        with torch.no_grad():
            raw = model(_coeffs(batch=3, seed=3))
            out = model.denormalized(_coeffs(batch=3, seed=3))
        assert torch.allclose(out, peak_normalize(raw), atol=1e-7)
        assert torch.allclose(out.amax(dim=(-2, -1)), torch.ones(3, 1), atol=1e-6)


# ---------------------------------------------------------------------------
# Factory + validation
# ---------------------------------------------------------------------------
class TestBuildForwardModel:
    """Dispatch and its error path."""

    def test_dispatches_to_the_right_class(self) -> None:
        assert isinstance(
            build_forward_model(_config(architecture="conv")), ZernikeCoeffConvNet
        )
        assert isinstance(
            build_forward_model(_config(architecture="mlp")), ZernikeCoeffMLP
        )

    def test_default_architecture_is_the_conv_decoder(self) -> None:
        """The conv arm is the preferred one; the MLP is only the baseline."""
        assert ZernikeCoeffConfig().architecture == "conv"
        assert isinstance(build_forward_model(ZernikeCoeffConfig()), ZernikeCoeffConvNet)

    def test_unknown_architecture_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="architecture"):
            build_forward_model(_config(architecture="unet"))  # type: ignore[arg-type]

    def test_mlp_ignores_the_conv_only_reachability_rule(self) -> None:
        """`bottleneck * 2**len(features)` is a conv-arm constraint only."""
        assert isinstance(
            build_forward_model(_config(architecture="mlp", grid=64)), ZernikeCoeffMLP
        )
        # A grid unreachable by upsampling is fine for the MLP.
        mlp = build_forward_model(
            _config(architecture="mlp", grid=30, bottleneck=4, features=(8,))
        )
        assert tuple(mlp(_coeffs(batch=1)).shape) == (1, 1, 30, 30)


class TestReachabilityValidation:
    """`grid` must be exactly `bottleneck * 2 ** len(features)`."""

    def test_unreachable_grid_message_is_actionable(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            ZernikeCoeffConvNet(_config(grid=63))
        message = str(excinfo.value)
        # The actual product that was computed...
        assert "64" in message
        assert "2**4" in message or "2^4" in message
        # ...and the requirement, with a concrete fix.
        assert "bottleneck" in message
        assert "len(features)" in message
        assert "== grid" in message

    @pytest.mark.parametrize(
        ("bottleneck", "n_features", "grid"),
        [(4, 4, 64), (1, 4, 16), (8, 3, 64), (16, 2, 64), (32, 1, 64), (2, 5, 64)],
    )
    def test_reachable_combinations_construct(
        self, bottleneck: int, n_features: int, grid: int
    ) -> None:
        widths = tuple(2 ** (2 * i + 2) for i in range(n_features))
        model = ZernikeCoeffConvNet(
            _config(bottleneck=bottleneck, features=widths, grid=grid)
        )
        assert tuple(model(_coeffs(batch=1)).shape) == (1, 1, grid, grid)

    @pytest.mark.parametrize(
        ("bottleneck", "n_features", "grid"),
        [(4, 4, 63), (4, 4, 128), (3, 4, 64), (5, 4, 64)],
    )
    def test_unreachable_combinations_raise(
        self, bottleneck: int, n_features: int, grid: int
    ) -> None:
        widths = tuple(2 ** (2 * i + 2) for i in range(n_features))
        with pytest.raises(ValueError, match="unreachable"):
            ZernikeCoeffConvNet(
                _config(bottleneck=bottleneck, features=widths, grid=grid)
            )

    def test_rejects_empty_features_and_bad_widths(self) -> None:
        with pytest.raises(ValueError, match="features"):
            ZernikeCoeffConvNet(_config(features=()))
        with pytest.raises(ValueError, match="features"):
            ZernikeCoeffConvNet(_config(features=(8, 0, 4, 2)))

    def test_rejects_non_positive_geometry(self) -> None:
        with pytest.raises(ValueError, match="n_coeffs"):
            ZernikeCoeffConvNet(_config(n_coeffs=0))
        with pytest.raises(ValueError, match="grid"):
            ZernikeCoeffConvNet(_config(grid=0))
        with pytest.raises(ValueError, match="latent_channels"):
            ZernikeCoeffConvNet(_config(latent_channels=0))
        # The positivity check must fire *before* the reachability identity:
        # `bottleneck=0` yields `reachable == 0`, which can never equal the
        # already-validated `grid >= 1`, so the reverse order would report a
        # misleading "grid unreachable" error instead of naming the field.
        with pytest.raises(ValueError, match=r"bottleneck must be >= 1"):
            ZernikeCoeffConvNet(_config(bottleneck=0))
        with pytest.raises(ValueError, match="norm_layer"):
            ZernikeCoeffConvNet(_config(norm_layer="instance"))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="hidden"):
            ZernikeCoeffMLP(_config(hidden=0))


# ---------------------------------------------------------------------------
# Coefficient contract
# ---------------------------------------------------------------------------
class TestCoefficientContract:
    """Every input dimension must be wired to the output; no silent no-ops."""

    @pytest.mark.parametrize("index", [0, 1, 5, 40, DEFAULT_N_COEFFS - 1])
    def test_single_coefficient_change_moves_the_output(self, index: int) -> None:
        """Changing exactly one coefficient must change the image."""
        torch.manual_seed(7)
        model = ZernikeCoeffConvNet(_config())
        base = _coeffs(batch=1, seed=4, scale=10.0)
        bumped = base.clone()
        bumped[0, index] += 0.5
        with torch.no_grad():
            difference = (model(bumped) - model(base)).abs().max()
        assert difference.item() > 0.0, (
            f"coefficient index {index} is a dead input dimension"
        )

    def test_every_coefficient_is_wired(self) -> None:
        """Sweep all 136 inputs: none may be a dead dimension.

        The corpus pads up to 136 from observed lengths 15/36/78, so a genuinely
        unused slot would silently waste a third of the input width.
        """
        torch.manual_seed(11)
        model = ZernikeCoeffConvNet(_config())
        base = _coeffs(batch=1, seed=5, scale=10.0)
        with torch.no_grad():
            reference = model(base)
            dead = []
            for index in range(DEFAULT_N_COEFFS):
                bumped = base.clone()
                bumped[0, index] += 0.5
                if torch.equal(model(bumped), reference):
                    dead.append(index)
        assert dead == []

    def test_piston_only_vector_is_finite_and_non_constant(self) -> None:
        """All zeros except index 0 (Noll 1 == piston).

        Piston is a *physical* far-field no-op, but the learned map is a general
        function, so it must still produce a finite, spatially varying image --
        a constant image would mean the coefficient bridge is disconnected.
        """
        torch.manual_seed(3)
        for factory in (ZernikeCoeffConvNet, ZernikeCoeffMLP):
            model = factory(_config())
            piston = torch.zeros(2, DEFAULT_N_COEFFS)
            piston[:, 0] = 0.5
            with torch.no_grad():
                out = model(piston)
            assert torch.isfinite(out).all()
            assert out.std(dim=(-2, -1)).max().item() > 0.0

    def test_zero_vector_is_finite(self) -> None:
        torch.manual_seed(5)
        for factory in (ZernikeCoeffConvNet, ZernikeCoeffMLP):
            model = factory(_config())
            with torch.no_grad():
                out = model(torch.zeros(2, DEFAULT_N_COEFFS))
            assert torch.isfinite(out).all()

    def test_gradients_reach_the_coefficient_bridge(self) -> None:
        """A dead bridge would leave the first Linear's weight grad None or all-zero."""
        torch.manual_seed(9)
        for factory in (ZernikeCoeffConvNet, ZernikeCoeffMLP):
            model = factory(_config())
            peak_normalize(model(_coeffs(batch=2, seed=6))).pow(2).mean().backward()
            # The coefficient bridge is the first Linear in both architectures:
            # `coeff_proj` on the conv arm, `net[0]` on the MLP arm.
            bridge = model.net[0] if isinstance(model, ZernikeCoeffMLP) else model.coeff_proj
            assert isinstance(bridge, nn.Linear)
            grad = bridge.weight.grad
            assert grad is not None
            assert torch.isfinite(grad).all()
            assert torch.count_nonzero(grad) > 0


# ---------------------------------------------------------------------------
# Norm layer
# ---------------------------------------------------------------------------
class TestNormLayer:
    """GroupNorm is the default; BatchNorm is a documented anti-default."""

    @pytest.mark.parametrize("norm_layer", ["group", "batch", "none"])
    def test_all_three_construct_and_forward(self, norm_layer: str) -> None:
        model = ZernikeCoeffConvNet(_config(norm_layer=norm_layer))
        out = model(_coeffs(batch=2, seed=8))
        assert tuple(out.shape) == (2, 1, 64, 64)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("norm_layer", ["group", "none"])
    def test_no_batchnorm_when_group_or_none(self, norm_layer: str) -> None:
        """Guards the decision documented in the module docstring.

        `unet.DoubleConv` hardcodes `nn.BatchNorm2d`; its running statistics
        couple eval() behaviour to training history, which is wrong for a
        per-frame peak-normalised 64x64 regression target.
        """
        model = ZernikeCoeffConvNet(_config(norm_layer=norm_layer))
        assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())

    def test_batch_norm_is_present_only_when_asked_for(self) -> None:
        model = ZernikeCoeffConvNet(_config(norm_layer="batch"))
        assert any(isinstance(m, nn.BatchNorm2d) for m in model.modules())

    def test_group_norm_is_the_default(self) -> None:
        assert ZernikeCoeffConfig().norm_layer == "group"
        model = ZernikeCoeffConvNet(_config())
        assert any(isinstance(m, nn.GroupNorm) for m in model.modules())
        assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())

    def test_indivisible_width_falls_back_to_gcd_instead_of_raising(self) -> None:
        """A width that `norm_groups` does not divide must degrade, not explode.

        `GroupNorm` raises a deep, unhelpful error for an invalid group count, so
        the block reduces to the greatest common divisor and logs instead.
        """
        # 24 is not divisible by 8's coprime partner 5 -> gcd = 1.
        model = ZernikeCoeffConvNet(_config(features=(24, 24, 24, 24), norm_groups=5))
        out = model(_coeffs(batch=2, seed=10))
        assert tuple(out.shape) == (2, 1, 64, 64)
        assert torch.isfinite(out).all()

    def test_norm_groups_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="norm_groups"):
            ZernikeCoeffConvNet(_config(norm_groups=0))


# ---------------------------------------------------------------------------
# Reproducibility + size
# ---------------------------------------------------------------------------
class TestDeterminism:
    """Same seed -> bit-for-bit identical outputs on CPU."""

    def test_same_seed_gives_identical_conv_output(self) -> None:
        torch.manual_seed(1234)
        first = ZernikeCoeffConvNet(_config())
        torch.manual_seed(1234)
        second = ZernikeCoeffConvNet(_config())
        assert count_parameters(first) == count_parameters(second)
        coeffs = _coeffs(batch=4, seed=21)
        with torch.no_grad():
            assert torch.equal(first(coeffs), second(coeffs))

    def test_same_seed_gives_identical_mlp_output(self) -> None:
        torch.manual_seed(4321)
        first = ZernikeCoeffMLP(_config())
        torch.manual_seed(4321)
        second = ZernikeCoeffMLP(_config())
        coeffs = _coeffs(batch=4, seed=22)
        with torch.no_grad():
            assert torch.equal(first(coeffs), second(coeffs))

    def test_repeated_calls_are_stable(self) -> None:
        torch.manual_seed(99)
        model = ZernikeCoeffConvNet(_config()).eval()
        coeffs = _coeffs(batch=2, seed=23)
        with torch.no_grad():
            assert torch.equal(model(coeffs), model(coeffs))

    def test_different_seeds_give_different_models(self) -> None:
        torch.manual_seed(1)
        first = ZernikeCoeffConvNet(_config())
        torch.manual_seed(2)
        second = ZernikeCoeffConvNet(_config())
        coeffs = _coeffs(batch=2, seed=24)
        with torch.no_grad():
            assert not torch.equal(first(coeffs), second(coeffs))


class TestCountParameters:
    """Report the parameter budget: the point of the conv-vs-MLP comparison."""

    def test_counts_are_positive_and_exact(self) -> None:
        conv = ZernikeCoeffConvNet(_config())
        mlp = ZernikeCoeffMLP(_config())
        for model in (conv, mlp):
            count = count_parameters(model)
            assert isinstance(count, int)
            assert count > 0
            expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert count == expected

    def test_conv_parameter_budget_is_reported(self) -> None:
        """Printed so the conv/MLP ratio lands in the test log, not in a comment."""
        conv = count_parameters(ZernikeCoeffConvNet(_config()))
        mlp = count_parameters(ZernikeCoeffMLP(_config()))
        print(f"\nZernikeCoeffConvNet params = {conv:,}")
        print(f"ZernikeCoeffMLP   params = {mlp:,}")
        print(f"MLP / conv ratio         = {mlp / conv:.3f}")
        assert conv > 0 and mlp > 0

    def test_frozen_parameters_are_excluded(self) -> None:
        model = ZernikeCoeffConvNet(_config())
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        assert count_parameters(model) == 0


# ---------------------------------------------------------------------------
# Learnability
# ---------------------------------------------------------------------------
class TestOverfitSmoke:
    """The model must be able to fit something -- gradient plumbing works."""

    #: The smoke uses a *narrow* copy of the production decoder (same topology:
    #: 4 stages of scale-2 nearest upsample + conv, same 136->64x64 mapping, same
    #: GroupNorm), only with fewer channels. The full 256-wide stack costs 43 s
    #: for 200 CPU steps; this configuration exercises exactly the same
    #: gradient path -- coeff_proj, every upsample block, the head -- in a few
    #: seconds, which keeps the smoke inside the ~20 s budget.
    _SMOKE_OVERRIDES = {"latent_channels": 64, "features": (64, 32, 16, 8), "hidden": 64}

    def test_tiny_overfit_reduces_mse(self) -> None:
        torch.manual_seed(0)
        model = ZernikeCoeffConvNet(_config(**self._SMOKE_OVERRIDES))
        generator = torch.Generator().manual_seed(31)

        coeffs = torch.randn(32, DEFAULT_N_COEFFS, generator=generator) * 0.0142
        # Stand-in for peak-normalised CCD frames.
        targets = peak_normalize(torch.rand(32, 1, 64, 64, generator=generator))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with torch.no_grad():
            initial = torch.nn.functional.mse_loss(
                peak_normalize(model(coeffs)), targets
            ).item()

        for _ in range(200):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(
                peak_normalize(model(coeffs)), targets
            )
            loss.backward()
            optimizer.step()

        final = loss.item()
        print(f"\noverfit smoke: initial MSE={initial:.6e} final MSE={final:.6e}")
        assert final < initial, f"no learning happened: {initial:.6e} -> {final:.6e}"
        # Must be a real reduction, not float noise settling.
        assert final < 0.5 * initial

    def test_production_width_takes_one_step(self) -> None:
        """The default-width model must at least be trainable, cheaply."""
        torch.manual_seed(0)
        model = ZernikeCoeffConvNet(_config())
        generator = torch.Generator().manual_seed(33)
        coeffs = torch.randn(8, DEFAULT_N_COEFFS, generator=generator) * 0.0142
        targets = peak_normalize(torch.rand(8, 1, 64, 64, generator=generator))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with torch.no_grad():
            initial = torch.nn.functional.mse_loss(
                peak_normalize(model(coeffs)), targets
            ).item()
        loss = torch.nn.functional.mse_loss(peak_normalize(model(coeffs)), targets)
        loss.backward()
        grad = model.coeff_proj.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0
        optimizer.step()
        with torch.no_grad():
            after = torch.nn.functional.mse_loss(
                peak_normalize(model(coeffs)), targets
            ).item()
        assert after <= initial + 1e-6

    def test_mlp_also_learns(self) -> None:
        torch.manual_seed(0)
        model = ZernikeCoeffMLP(_config())
        generator = torch.Generator().manual_seed(32)
        coeffs = torch.randn(16, DEFAULT_N_COEFFS, generator=generator) * 0.0142
        targets = peak_normalize(torch.rand(16, 1, 64, 64, generator=generator))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with torch.no_grad():
            initial = torch.nn.functional.mse_loss(
                peak_normalize(model(coeffs)), targets
            ).item()
        for _ in range(200):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(
                peak_normalize(model(coeffs)), targets
            )
            loss.backward()
            optimizer.step()
        assert loss.item() < initial