"""Tests for :mod:`ml.sim2f.losses` -- piston invariance, the objective, metrics.

The bench is a **hand-written stub** here on purpose: these are unit tests of
the loss arithmetic, and they must run and stay fast while ``ml.sim2f.optics``
is being written by a parallel change.  The one integration test against the
real bench is skipped cleanly when that module is absent.

Nothing here asserts that energy is conserved or that a frame has no background:
the bench this package scores is the repository's calibrated 2f twin, which
carries a zero-order background term, dark FPN and read noise.  The tests assert
the opposite -- that the prepared objective is *invariant* to a pedestal.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.sim2f.losses import (
    as_frame,
    forward_image,
    from_phasor,
    objective_terms,
    physics_loss,
    piston_free_phasor_corr,
    piston_free_phasor_l1,
    prepare_frame,
    shaping_metrics,
    to_phasor,
    tv_phasor,
)

_SIZE = 32


def _disc_target(size: int = _SIZE, pad: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """A square target patch and its support, both unit-ish.

    Args:
        size: Grid side.
        pad: Half-width of the patch.

    Returns:
        ``(target, support)``, each ``(1, 1, size, size)``.
    """
    target = torch.zeros(1, 1, size, size)
    centre = size // 2
    target[0, 0, centre - pad : centre + pad, centre - pad : centre + pad] = 1.0
    return target, (target > 0).to(torch.float32)


class FixedFrame(torch.nn.Module):
    """A forward model that returns a stored frame, ignoring its input.

    Used where the *arithmetic* of the objective is under test: with a known
    frame the expected loss can be computed by hand.
    """

    def __init__(self, frame: torch.Tensor) -> None:
        """Store the frame.

        Args:
            frame: ``(1, 1, H, W)`` or ``(B, 1, H, W)`` intensity.
        """
        super().__init__()
        self.register_buffer("frame", frame)

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        """Return the stored frame, repeated to the input's batch size.

        Args:
            phase: ``(B, grid, grid)`` phase (unused).

        Returns:
            ``(B, 1, H, W)``.
        """
        return self.frame.expand(phase.shape[0], -1, -1, -1)


class ToyBench(torch.nn.Module):
    """Differentiable stand-in for the 2f twin: ``I = |FFT2(exp(i*phase))|^2``.

    A real (if minuscule) optical model, so gradients flow through a genuine
    propagation.  It deliberately carries **no** detector model: the twin's
    pedestal is added by the tests that care about it, explicitly.
    """

    def __init__(self, background: float = 0.0) -> None:
        """Build the (stateless) bench.

        Args:
            background: Constant added to every pixel, standing in for the
                detector's dark level plus the 0-order pedestal.
        """
        super().__init__()
        self.background = float(background)

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        """Propagate a phase map.

        Args:
            phase: ``(B, G, G)`` or ``(B, 1, G, G)`` radians.

        Returns:
            ``(B, 1, G, G)`` intensity.
        """
        if phase.ndim == 4:
            phase = phase[:, 0]
        field = torch.exp(1j * phase.to(torch.complex64))
        image = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1)).abs().pow(2)
        return image[:, None] + self.background


def _smooth_phase(size: int, seed: int, amplitude: float = 1.0) -> torch.Tensor:
    """A structured phase map: box-smoothed noise scaled to ``amplitude`` radians.

    A *flat* phase is a degenerate test input here: it puts all the light in one
    pixel, and the unit-sum normalisation of that frame saturates
    (``p_peak = 1`` with ``sum == peak``), so the objective's gradient is exactly
    zero for reasons that have nothing to do with the loss.  Every test that
    needs a gradient therefore starts from a structured field.

    Args:
        size: Grid side.
        seed: RNG seed.
        amplitude: Peak phase amplitude, radians.

    Returns:
        ``(size, size)`` radians.
    """
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(1, 1, size, size, generator=generator)
    padded = torch.nn.functional.pad(noise, (4, 4, 4, 4), mode="replicate")
    smooth = torch.nn.functional.avg_pool2d(padded, kernel_size=9, stride=1)
    return (amplitude * smooth[0, 0]).detach()


# ----------------------------------------------------------------------
# Phasor representation
# ----------------------------------------------------------------------
def test_phasor_roundtrip() -> None:
    """``from_phasor(to_phasor(phi)) == phi`` modulo the 2*pi wrap."""
    phase = torch.linspace(-3.0, 3.0, _SIZE * _SIZE).reshape(1, _SIZE, _SIZE)
    phasor = to_phasor(phase)
    assert phasor.shape == (1, 2, _SIZE, _SIZE)
    assert torch.allclose(from_phasor(phasor), phase, atol=1e-5)
    with pytest.raises(ValueError, match="from_phasor"):
        from_phasor(torch.rand(1, 3, _SIZE, _SIZE))


def test_piston_free_l1_is_invariant_to_global_phase() -> None:
    """Rotating the prediction by any per-sample global phase changes nothing."""
    generator = torch.Generator().manual_seed(0)
    pred = torch.nn.functional.normalize(
        torch.rand(3, 2, _SIZE, _SIZE, generator=generator), dim=1
    )
    gt = torch.nn.functional.normalize(
        torch.rand(3, 2, _SIZE, _SIZE, generator=generator), dim=1
    )
    mask = (torch.rand(1, 1, _SIZE, _SIZE, generator=generator) > 0.3).to(torch.float32)

    baseline = piston_free_phasor_l1(pred, gt, mask)
    theta = torch.tensor([0.0, 1.1, -2.7])
    cos, sin = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]
    rotated = torch.stack(
        [pred[:, 0] * cos - pred[:, 1] * sin, pred[:, 0] * sin + pred[:, 1] * cos], dim=1
    )

    assert not torch.allclose(rotated, pred)
    assert torch.allclose(
        piston_free_phasor_l1(rotated, gt, mask), baseline, rtol=1e-5, atol=1e-6
    )


def test_masked_l1_ignores_dead_cells() -> None:
    """Cells outside the mask cannot influence the supervised loss."""
    gt = torch.zeros(1, 2, _SIZE, _SIZE)
    gt[:, 0] = 1.0
    pred = gt.clone()
    mask = torch.zeros(1, 1, _SIZE, _SIZE)
    mask[:, :, : _SIZE // 2] = 1.0

    assert float(piston_free_phasor_l1(pred, gt, mask)) == pytest.approx(0.0)
    pred[:, :, _SIZE // 2 :] = -1.0  # only dead cells change
    assert float(piston_free_phasor_l1(pred, gt, mask)) == pytest.approx(0.0, abs=1e-7)
    assert float(piston_free_phasor_l1(pred, gt)) > 0.1


def test_phasor_corr_ignores_piston_but_not_shape() -> None:
    """Correlation is 1 under a global rotation and lower for a wrong phase."""
    phase = torch.rand(2, _SIZE, _SIZE) * 6.0
    gt = to_phasor(phase)
    assert piston_free_phasor_corr(gt, gt) == pytest.approx(1.0, abs=1e-5)
    rotated = to_phasor(phase + 1.234)
    assert piston_free_phasor_corr(rotated, gt) == pytest.approx(1.0, abs=1e-5)
    different = to_phasor(torch.rand(2, _SIZE, _SIZE) * 6.0)
    assert piston_free_phasor_corr(different, gt) < 0.9


def test_tv_of_a_constant_phasor_is_zero() -> None:
    """A spatially constant phasor has no total variation."""
    constant = torch.zeros(1, 2, _SIZE, _SIZE)
    constant[:, 0] = 1.0
    assert float(tv_phasor(constant)) == pytest.approx(0.0)
    assert float(tv_phasor(to_phasor(torch.rand(1, _SIZE, _SIZE) * 3.0))) > 0.0


# ----------------------------------------------------------------------
# The teacher's objective
# ----------------------------------------------------------------------
def test_objective_matches_the_hand_computed_teacher_formula() -> None:
    """``L = L1(p, t) + w_ee * (1 - EE)`` on unit-sum frames, to 1e-5."""
    target, support = _disc_target()
    image = torch.zeros_like(target)
    image[0, 0, 10:14, 10:14] = 2.0
    image[0, 0, 20:24, 20:24] = 1.0
    w_ee = 0.25
    metric = "l1"

    loss, l1, ee, ncc = objective_terms(image, target, support, w_ee=w_ee, metric=metric)
    bench = FixedFrame(image)
    physics, terms = physics_loss(
        bench,
        to_phasor(torch.zeros(1, _SIZE, _SIZE)),
        target,
        support,
        w_ee=w_ee,
        metric=metric,
    )

    # Hand computation, in float64 so the tolerance is about the formula.
    # `L_shape` is a *sum* over the frame (total-variation distance); see the
    # scale rationale in `objective_terms`.
    p = image.double() / image.double().sum()
    t = target.double() / target.double().sum()
    expected_l1 = float((p - t).abs().sum())
    expected_ee = float((p * support.double()).sum())
    expected_loss = expected_l1 + w_ee * (1.0 - expected_ee)
    centred_p = p - p.mean()
    centred_t = t - t.mean()
    expected_ncc = float(
        (centred_p * centred_t).sum()
        / math.sqrt(float(centred_p.pow(2).sum()) * float(centred_t.pow(2).sum()))
    )

    assert float(l1) == pytest.approx(expected_l1, rel=1e-5)
    assert float(ee) == pytest.approx(expected_ee, rel=1e-5)
    assert float(loss) == pytest.approx(expected_loss, rel=1e-5)
    assert float(ncc) == pytest.approx(expected_ncc, rel=1e-4)
    assert float(physics) == pytest.approx(expected_loss, rel=1e-5)
    assert terms["loss"] == pytest.approx(expected_loss, rel=1e-5)
    assert terms["l1"] == pytest.approx(expected_l1, rel=1e-5)
    assert terms["ee"] == pytest.approx(expected_ee, rel=1e-5)


def test_objective_is_differentiable_through_the_forward_model() -> None:
    """The physics leg backpropagates into the phasor."""
    bench = ToyBench()
    target_image = bench(_smooth_phase(_SIZE, 1, amplitude=0.6)[None])[0]
    target = target_image / target_image.sum()
    support = (target >= 0.15 * target.max()).to(torch.float32)

    phase = _smooth_phase(_SIZE, 2, amplitude=0.25)[None].requires_grad_(True)
    phasor = to_phasor(phase)

    loss, terms = physics_loss(bench, phasor, target, support, w_ee=0.25)
    assert set(terms) == {"l1", "ee", "ncc", "loss"}
    loss.backward()

    assert phase.grad is not None
    assert torch.isfinite(phase.grad).all()
    assert float(phase.grad.abs().max()) > 0.0


def test_prepare_removes_a_constant_pedestal() -> None:
    """A frame that is ``target * k + c`` scores exactly like the target."""
    target, support = _disc_target()
    noisy = target * 100.0 + 7.0
    loss, l1, ee, _ncc = objective_terms(
        noisy, target, support, w_ee=0.25, prepare=True
    )
    assert float(l1) == pytest.approx(0.0, abs=1e-6)
    assert float(ee) == pytest.approx(1.0, abs=1e-6)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)

    raw_loss, raw_l1, raw_ee, _ = objective_terms(
        noisy, target, support, w_ee=0.25, prepare=False
    )
    assert float(raw_l1) > 1e-2
    assert float(raw_ee) < 0.99
    assert float(raw_loss) > float(loss)


def test_prepare_frame_subtracts_median_before_clipping() -> None:
    """The order matters: clipping first would rectify the noise into a pedestal."""
    generator = torch.Generator().manual_seed(3)
    frame = torch.full((1, 1, _SIZE, _SIZE), 10.0) + torch.randn(
        (1, 1, _SIZE, _SIZE), generator=generator
    ) * 0.5

    prepared = prepare_frame(frame)
    assert float(prepared.median()) == pytest.approx(0.0, abs=0.05)
    assert float(prepared.min()) == 0.0
    # Clipping first would keep ~10 counts per pixel as a pedestal.
    clip_first = frame.clamp_min(0.0)
    assert float(clip_first.mean()) > float(prepared.mean()) + 1.0


def test_frame_median_matches_torch_median_exactly() -> None:
    """The pedestal is ``torch.median``'s value, lower-middle for an even count.

    ``torch.median(dim=...)`` has no deterministic CUDA kernel, so
    :func:`ml.sim2f.losses._frame_median` computes the same element with
    ``kthvalue``.  "The same element" is the whole contract: the lower of the two
    middles for an even pixel count (the *mean* of the middles, which
    ``torch.quantile(0.5)`` returns, is a different statistic and would shift the
    pedestal on every even-sized frame -- i.e. always).
    """
    from ml.sim2f.losses import _frame_median

    generator = torch.Generator().manual_seed(5)
    frames = {
        "even": torch.rand(3, 1, 4, 4, generator=generator),
        "odd": torch.rand(3, 1, 3, 5, generator=generator),
        "single_pixel": torch.rand(2, 1, 1, 1, generator=generator),
        "ties": torch.zeros(2, 1, 2, 3),
    }
    for name, frame in frames.items():
        expected = frame.flatten(1).median(dim=1).values[:, None, None, None]
        sorted_middle = frame.flatten(1).sort(dim=1).values[
            :, (frame[0].numel() - 1) // 2
        ][:, None, None, None]
        got = _frame_median(frame)
        assert torch.equal(got, expected), name
        assert torch.equal(got, sorted_middle), name


def test_prepare_frame_is_silent_under_deterministic_algorithms() -> None:
    """Preparing a frame must not break ``--cudnn-deterministic``.

    ``torch.median`` warns (and, with ``warn_only=False``, raises) on CUDA, which
    would silently void the bit-identical-rerun promise of a training run.  The
    check runs on whatever device is available, so on a CUDA machine it guards
    the real defect and on CPU it guards the switch itself.
    """
    import warnings

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        frame = torch.rand(4, 1, 16, 16, device=device) * 100.0 + 5.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prepare_frame(frame)
            if device == "cuda":
                torch.cuda.synchronize()
        offenders = [str(item.message) for item in caught if "deterministic" in str(item.message)]
        assert offenders == []
    finally:
        torch.use_deterministic_algorithms(previous)


def test_support_term_penalises_spraying_energy() -> None:
    """``w_ee * (1 - EE)`` grows when energy leaves the support."""
    target, support = _disc_target()
    inside = target.clone()
    outside = torch.zeros_like(target)
    outside[0, 0, 1:5, 1:5] = 1.0

    loss_in, _, _, _ = objective_terms(inside, target, support, w_ee=1.0)
    loss_out, _, _, _ = objective_terms(outside, target, support, w_ee=1.0)
    assert float(loss_out) > float(loss_in)


# ----------------------------------------------------------------------
# Shaping metrics
# ----------------------------------------------------------------------
def test_metrics_of_a_perfect_prediction() -> None:
    """A perfect prediction gives ``l1 = 0``, ``ncc = 1``, ``iou50 = 1``."""
    target, support = _disc_target()
    metrics = shaping_metrics(target, target, support)

    assert metrics["l1"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["ncc"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["iou50"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["ee"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["cv_in_support"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["peak_ratio"] == pytest.approx(1.0, abs=1e-6)


def test_metrics_are_invariant_to_pedestal_and_brightness() -> None:
    """Scaled-and-offset copies of a frame score the same prepared metrics."""
    target, support = _disc_target()
    predicted = target.clone()
    predicted[0, 0, 5:8, 5:8] = 0.3
    predicted = predicted / predicted.sum()

    base = shaping_metrics(predicted, target, support)
    scaled = shaping_metrics(predicted * 250.0 + 12.0, target, support)
    for key in ("l1", "ncc", "ee", "iou50"):
        assert scaled[key] == pytest.approx(base[key], abs=1e-5), key


def test_cv_is_measured_inside_the_support_only() -> None:
    """A bright uniform support over a dark field has CV 0, not a bounding-box CV."""
    size = _SIZE
    support = torch.zeros(1, 1, size, size)
    support[0, 0, 8:24, 8:24] = 1.0
    predicted = support.clone()
    predicted[0, 0, 0, 0] = 5.0  # a stray hot pixel far outside the support

    metrics = shaping_metrics(predicted, support, support)
    assert metrics["cv_in_support"] == pytest.approx(0.0, abs=1e-5)

    uneven = support.clone()
    uneven[0, 0, 8:16, 8:16] *= 4.0
    assert shaping_metrics(uneven, support, support)["cv_in_support"] > 0.5


def test_metrics_are_finite_for_a_dark_frame() -> None:
    """An all-zero frame yields finite metrics instead of NaN."""
    target, support = _disc_target()
    metrics = shaping_metrics(torch.zeros_like(target), target, support)
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["ee"] == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# Integration with the real bench (skipped until optics.py exists)
# ----------------------------------------------------------------------
def test_physics_loss_with_the_real_twin() -> None:
    """The loss runs end-to-end on ``optics.Sim2f`` and the 0-order is handled."""
    optics = pytest.importorskip(
        "ml.sim2f.optics", reason="optics.py is written by a parallel change"
    )
    from dataclasses import fields, replace

    from ml.sim2f.config import OpticsErrors, Sim2fConfig

    size = 64
    known = {item.name for item in fields(Sim2fConfig)}
    overrides: dict[str, object] = {
        "grid": size,
        "ccd_px": size,
        # The calibrated twin's 0-order: 10 % of the light goes into a coherent
        # pupil mix, so `I.sum()` is not the modulated energy.  This is the term
        # `prepare_frame` exists for.
        "errors": OpticsErrors(zero_order_eta=0.1),
    }
    sim_cfg = replace(
        Sim2fConfig(), **{key: value for key, value in overrides.items() if key in known}
    )
    bench = optics.Sim2f(sim_cfg, device="cpu")
    target_phase = _smooth_phase(size, 11, amplitude=1.0)[None]
    teacher = as_frame(bench.forward(target_phase))
    target = teacher / teacher.sum().clamp_min(1e-8)
    support = (target >= 0.15 * target.max()).to(torch.float32)

    phasor = to_phasor(_smooth_phase(size, 12, amplitude=0.4)[None])
    loss, terms = physics_loss(bench, phasor, target, support, w_ee=0.25)
    raw_loss, _raw_l1, _raw_ee, _raw_ncc = objective_terms(
        forward_image(bench, from_phasor(phasor)), target, support,
        w_ee=0.25, prepare=False,
    )

    assert math.isfinite(float(loss))
    assert math.isfinite(terms["l1"])
    assert 0.0 <= terms["ee"] <= 1.0 + 1e-4
    # With a 0-order pedestal the raw and prepared objectives disagree, which is
    # exactly why the switch exists: a report must say which one it quotes.
    assert abs(float(loss) - float(raw_loss)) > 1e-4

