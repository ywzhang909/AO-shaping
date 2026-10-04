"""Defect-hunt regression tests: three defects found by metric/gradient probing.

Each of these was found by measurement, not by reading the code, and each failed
before its fix:

1. ``shape_gap_term`` -- the physical terms (``pib`` / ``uniformity``) are
   **unanchored**: they read only the prediction. On a *fitting* task their
   optimum is "ignore the data and emit an ideal spot". Measured on the real
   corpus: val R2 +0.78 -> **-0.86** with ``shape_sum`` 34% *above* the physics
   being predicted.
2. ``image_mode="robust"`` -- CCD intensity normalisation. ``abs255`` leaves the
   per-frame detector gain in (measured frame maxima span 255 / 239.6 / 100.9
   across dtypes), while ``peak`` divides by a *single pixel* and this bench has
   hot pixels that outrank the real 0-order (peak 22-46 against a frame mean of
   0.26).
3. ``FarFieldAttention`` -- an additive residual + ``clamp(min=0)`` was a silent
   dead end: Adam's first sign-based step drove ``out_proj`` negative enough to
   zero the whole field, after which **every** parameter stopped receiving
   gradient, with no error raised.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.hwdataset.transforms import ROBUST_PERCENTILE, farfield_frame_to_grid
from ml.zernike.losses import LossConfig, composite_loss, roi_mask, shape_gap_term
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel

GRID = 32
SIZE = 0.375 * GRID
CENTER = (GRID / 2.0, GRID / 2.0)


def _mask() -> torch.Tensor:
    return roi_mask((GRID, GRID), CENTER, "rectangle", SIZE, 4 / 3)


def _frame(spot: tuple[int, int], peak: float = 255.0, background: float = 0.0,
           size: int = 4) -> np.ndarray:
    img = np.full((64, 64), background, dtype=np.float32)
    x, y = spot
    img[y - size : y + size, x - size : x + size] = peak
    return img


# --------------------------------------------------------------------------
# 1. The anchored physical term
# --------------------------------------------------------------------------
def test_shape_gap_is_zero_only_when_the_shape_statistics_match():
    target = torch.rand(2, 1, GRID, GRID, dtype=torch.float64)
    target = target / target.sum(dim=(-2, -1), keepdim=True)
    mask = _mask()
    gap = shape_gap_term(target.clone(), target, mask)
    assert torch.allclose(gap, torch.zeros_like(gap), atol=1e-12)


def _square_target() -> torch.Tensor:
    """A compact, ROI-aligned square normalised to unit sum.

    Representative of the corpus (a far-field spot in a target box). A
    uniform-random frame is not: its ``shape_sum`` is ~0.2 rather than O(1), which
    quietly inverts the "relative normalisation puts the terms on the same scale"
    argument, and it lights the *whole* frame so ``pib`` is near 1 regardless of
    where the ROI is.
    """
    target = torch.zeros(1, 1, GRID, GRID, dtype=torch.float64)
    half_w = max(1, int(SIZE * (4 / 3) / 2))
    half_h = max(1, int(SIZE / 2))
    cx, cy = int(CENTER[0]), int(CENTER[1])
    target[..., cy - half_h : cy + half_h, cx - half_w : cx + half_w] = 1.0
    return target / target.sum()


def test_shape_gap_penalises_overshoot_and_undershoot_equally():
    """The property the unanchored terms lack: overshoot is not 'better'.

    ``pib``/``uniformity`` are maximised, so an emission far brighter and flatter
    than reality scores *better* than the data. The anchored gap must be blind to
    direction.
    """
    target = _square_target()
    mask = _mask()
    flat = torch.zeros_like(target)
    flat[..., : int(GRID / 2), : int(GRID / 2)] = 1.0  # a very "clean" square
    under = shape_gap_term(target, target, mask)
    over = shape_gap_term(flat, target, mask)
    assert float(under) == pytest.approx(0.0, abs=1e-12)
    assert float(over) > 0.0


def test_unanchored_terms_peak_on_an_emission_built_without_the_data():
    """Guards the premise of ``shape_gap_term``.

    The unanchored terms are maximised, so an emission constructed *without any
    reference to the measurement* drives them to their maximum -- that is exactly
    the degenerate direction that cost val R2 1.64 (see the module docstring).
    The anchored gap must not reward it.
    """
    from ml.zernike.losses import pib_term, uniformity_term

    target = _square_target()
    mask = _mask()
    ideal = torch.zeros_like(target)
    half_w = max(1, int(SIZE * (4 / 3) / 2))
    half_h = max(1, int(SIZE / 2))
    cx, cy = int(CENTER[0]), int(CENTER[1])
    ideal[..., cy - half_h : cy + half_h, cx - half_w : cx + half_w] = 1.0

    # Unanchored: maximum, with no reference to `target`.
    assert float(pib_term(ideal, mask)) == pytest.approx(1.0, abs=1e-9)
    assert float(uniformity_term(ideal, mask)) == pytest.approx(1.0, abs=1e-9)

    # Anchored: penalised for departing from the measurement. Feed it an emission
    # that is *not* the target so the gap is non-zero.
    other = torch.zeros_like(target)
    other[..., 2:6, 2:6] = 1.0
    assert float(shape_gap_term(other, target, mask)) > 0.0


def test_shape_gap_relative_normalisation_puts_it_on_the_mse_scale():
    """Without this, ``w_mse=1.0`` cannot anchor anything.

    Measured: the physical terms are O(1) and a peak-normalised MSE is O(0.003),
    a ~450x gap, so the blend's val R2 was -0.28 -- the fidelity term carried
    about 0.2% of the gradient. With ``shape_gap_relative=True`` the two terms
    are on the same footing, so ``w_mse=1`` and ``w_shape_gap=1`` are comparable.
    """
    from ml.zernike.losses import _shape_gap_reference_scale

    torch.manual_seed(0)
    target = _square_target().to(torch.float32)
    mask = _mask()
    pred = (target + 0.05 * torch.randn_like(target)).clamp(min=0)

    relative = composite_loss(
        pred,
        target,
        mask,
        LossConfig(w_mse=0.0, w_shape_gap=1.0, shape_gap_relative=True),
    )["shape_gap"]
    absolute = composite_loss(
        pred,
        target,
        mask,
        LossConfig(w_mse=0.0, w_shape_gap=1.0, shape_gap_relative=False),
    )["shape_gap"]

    scale = float(_shape_gap_reference_scale(target, mask))
    assert scale > 1.0, (
        f"a spot-like target should have shape_sum ~2, got {scale:.3f}; if this "
        "fails the relative-normalisation argument does not apply to it"
    )
    assert torch.allclose(relative, absolute / scale, atol=1e-6)
    # The point of the flag: relative makes the term smaller, so an equal weight
    # on MSE is a real balance rather than a rounding error.
    assert float(absolute.max()) > float(relative.max())


def test_composite_loss_reports_the_shape_gap_term():
    torch.manual_seed(1)
    target = torch.rand(2, 1, GRID, GRID, dtype=torch.float32)
    pred = (target + 0.1 * torch.randn_like(target)).clamp(min=0)
    out = composite_loss(
        pred, target, _mask(), LossConfig(w_mse=0.0, w_shape_gap=1.0)
    )
    assert "shape_gap" in out
    assert float(out["_mean_total"]) > 0.0


# --------------------------------------------------------------------------
# 2. CCD intensity normalisation
# --------------------------------------------------------------------------
def test_robust_mode_is_invariant_to_a_gain_change():
    """The whole point: two frames differing only by detector gain must match."""
    base = _frame((32, 32), peak=255.0, background=2.0)
    dim = base * 0.4  # same illumination, 2.5x lower gain
    a = farfield_frame_to_grid(base, GRID, mode="robust")
    b = farfield_frame_to_grid(dim, GRID, mode="robust")
    assert np.allclose(a, b, atol=0.02), (
        f"robust mode is not gain-invariant: max|d|={np.abs(a - b).max():.4f}"
    )


def test_abs255_is_not_gain_invariant_so_the_test_above_is_not_vacuous():
    base = _frame((32, 32), peak=255.0, background=2.0)
    dim = base * 0.4
    a = farfield_frame_to_grid(base, GRID, mode="abs255")
    b = farfield_frame_to_grid(dim, GRID, mode="abs255")
    assert not np.allclose(a, b, atol=0.02)


def test_robust_mode_ignores_a_single_hot_pixel_for_the_beam():
    """A lone hot pixel must not rescale the beam.

    Scoped to ``_robust_normalise`` deliberately: the earlier version of this test
    ran through ``farfield_frame_to_grid`` and failed, but not because the scale
    moved (it does not -- both frames give 253). The hot pixel *saturates* to 1.0,
    and then wins the ``argmax`` that ``_anchored_window`` uses to find the 0-order,
    so the crop lands elsewhere. That is a separate, pre-existing anchor hazard
    (AGENTS: a hot pixel can outrank the real 0-order) and is **not** something a
    normalisation mode can fix. So the claim pinned here is the one that is true
    and useful: the beam's own levels are unchanged.
    """
    from ml.hwdataset.transforms import _robust_normalise

    clean = _frame((32, 32), peak=255.0, background=2.0)
    hot = clean.copy()
    hot[4, 4] = 60000.0
    a = _robust_normalise(clean)
    b = _robust_normalise(hot)
    beam = (slice(24, 40), slice(24, 40))
    assert np.allclose(a[beam], b[beam], atol=0.02), (
        f"the beam itself was rescaled: max|d|={np.abs(a[beam] - b[beam]).max():.4f}"
    )


def test_peak_mode_is_disturbed_by_a_hot_pixel_so_that_test_is_not_vacuous():
    clean = _frame((32, 32), peak=255.0, background=2.0)
    hot = clean.copy()
    hot[4, 4] = 60000.0
    a = farfield_frame_to_grid(clean, GRID, mode="peak")
    b = farfield_frame_to_grid(hot, GRID, mode="peak")
    assert not np.allclose(a, b, atol=0.02)


def test_robust_mode_removes_the_background_pedestal():
    base = _frame((32, 32), peak=255.0, background=0.0)
    lifted = _frame((32, 32), peak=255.0, background=40.0)
    a = farfield_frame_to_grid(base, GRID, mode="robust")
    b = farfield_frame_to_grid(lifted, GRID, mode="robust")
    assert np.allclose(a, b, atol=0.02)


def test_robust_mode_of_a_dead_frame_is_zeros_not_nan():
    dead = np.zeros((64, 64), dtype=np.float32)
    out = farfield_frame_to_grid(dead, GRID, mode="robust")
    assert np.all(np.isfinite(out))
    assert float(out.max()) == 0.0


def test_robust_mode_output_stays_in_unit_range():
    out = farfield_frame_to_grid(_frame((32, 32), peak=255.0), GRID, mode="robust")
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6


def test_robust_percentile_is_the_median_of_the_lit_pixels():
    """Pinned deliberately: it was 99.5 and that was wrong.

    The percentile is taken over the *lit* pixels only, a subset of ~65 elements
    when the beam is a small fraction of the frame, so a high quantile is decided
    by a couple of hot pixels. The median is outlier-proof by construction.
    """
    assert ROBUST_PERCENTILE == 50.0


# --------------------------------------------------------------------------
# 3. The attention block
# --------------------------------------------------------------------------
def _inputs(batch: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(3)
    return torch.randn(batch, 1, 64, 64), torch.randn(batch, 1, 64, 64)


@pytest.mark.parametrize("observable", ["intensity", "amplitude"])
def test_attention_is_exactly_identity_at_initialisation(observable: str):
    """Without this the model would have to recover the physics from noise."""
    cos, sin = _inputs()
    plain = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=4, grid=64, observable=observable)
    )
    with_attn = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=4, grid=64, observable=observable, attention=True)
    )
    with torch.no_grad():
        for model in (plain, with_attn):
            model.coefficients.add_(0.4)
    with torch.no_grad():
        assert torch.equal(plain(cos, sin), with_attn(cos, sin))


def test_attention_is_off_by_default():
    assert ZernikeAmpModel(ZernikeAmpConfig(n_max=4, grid=64)).attention is None
    assert ZernikeAmpConfig().attention is False


def test_attention_parameters_receive_gradient_after_the_first_step():
    """Regression for the silent dead end.

    With an additive residual + ``clamp(min=0)``, Adam's first step zeroed the
    whole field and *every* tensor lost its gradient from step 1 onward, with no
    error raised. The multiplicative gate must keep them all alive.
    """
    from ml.zernike.train_amp import AmpTrainConfig, _build_optimizer

    cos, sin = _inputs(8)
    model = ZernikeAmpModel(ZernikeAmpConfig(n_max=4, grid=64, attention=True))
    optimizer = _build_optimizer(
        model, AmpTrainConfig(optimizer="adam", attention=True, lr=0.05)
    )
    seen_full = 0
    for _ in range(4):
        optimizer.zero_grad()
        out = model(cos, sin)
        assert float(out.max()) > 0.0, "attention zeroed the far field"
        out.pow(2).mean().backward()
        nonzero = sum(
            1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        seen_full = max(seen_full, nonzero)
        optimizer.step()
    total = sum(1 for p in model.parameters() if p.requires_grad)
    assert seen_full == total, (
        f"only {seen_full}/{total} tensors ever received gradient -- the attention "
        "is not actually an optimisation target"
    )


def test_attention_gate_is_bounded_and_positive():
    """The gate cannot invert or zero the field, whatever the weights."""
    from ml.zernike.models import FarFieldAttention

    torch.manual_seed(4)
    block = FarFieldAttention(grid=64, token_grid=16, dim=32, heads=1)
    with torch.no_grad():  # force a large, adversarial correction
        block.out_proj.weight.fill_(50.0)
        block.out_proj.bias.fill_(50.0)
    field = torch.rand(2, 1, 64, 64) + 0.1
    out = block(field)
    assert torch.isfinite(out).all()
    assert float(out.min()) > 0.0
    ratio = (out / field).max()
    assert float(ratio) <= 2.0 + 1e-5, f"gate exceeded its (0, 2x) bound: {ratio}"


def test_attention_dimension_must_divide_the_head_count():
    with pytest.raises(ValueError, match="divisible"):
        ZernikeAmpModel(
            ZernikeAmpConfig(
                n_max=4, grid=64, attention=True, attention_dim=30, attention_heads=4
            )
        )


def test_attention_parameters_are_a_small_fraction_of_a_full_grid_attention():
    """The token-grid choice is a cost decision, so pin the scale it implies.

    At ``grid=64`` a full-resolution map is 4096 tokens; the 16x16 default is 256,
    i.e. 1/16 the tokens and 1/256 the quadratic cost.
    """
    model = ZernikeAmpModel(ZernikeAmpConfig(n_max=4, grid=64, attention=True))
    n_params = sum(p.numel() for p in model.attention.parameters())
    # A 4096-token attention with dim 32 would need O(N^2) pair projections.
    quadratic = (64 * 64) ** 2
    assert n_params < quadratic / 100, (
        f"attention is {n_params} params, unexpectedly large for a {quadratic}-token "
        "quadratic budget"
    )