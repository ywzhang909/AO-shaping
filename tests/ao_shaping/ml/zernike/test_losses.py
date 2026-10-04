"""Acceptance gate for :mod:`ml.zernike.losses`.

Two things must hold, and neither is optional:

1. **Agreement.** Each torch term must reproduce the numpy metric it ports
   (:func:`roi_pib_metric`, :func:`rms_pib_terms`, :func:`roi_energy_loss` in
   ``ao_shaping.utils.image.target.metrics``) to float tolerance. If it does
   not, the loss is quietly optimising something the hardware optimiser never
   reports — the "second copy that drifts" failure mode. The ROI mask itself is
   built by calling the canonical builder, so this comparison is a genuine
   end-to-end check of the port rather than a tautology.
2. **Gradients.** Every term must produce a finite, non-zero gradient w.r.t.
   the model coefficients, otherwise it is a reported metric masquerading as a
   loss.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ao_shaping.utils.image.target.metrics import (
    roi_energy_loss as np_roi_energy_loss,
)
from ao_shaping.utils.image.target.metrics import (
    roi_pib_metric,
    rms_pib_terms,
)
from ml.zernike.losses import (
    EPS,
    LossConfig,
    composite_loss,
    pib_term,
    poisson_nll,
    roi_energy_loss,
    roi_mask,
    uniformity_term,
)

H, W = 64, 64
CENTER = (W / 2.0, H / 2.0)


def _frame(seed: int = 0) -> np.ndarray:
    """A frame with a bright blob on a dimmer background, uint8-like scale."""
    rng = np.random.default_rng(seed)
    img = rng.normal(20.0, 4.0, size=(H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.exp(-(((xx - CENTER[0]) ** 2 + (yy - CENTER[1]) ** 2) / (2 * 9.0**2)))
    return img + 180.0 * blob


def _mask(size: float = 24.0) -> torch.Tensor:
    return roi_mask((H, W), CENTER, "rectangle", size, 4.0 / 3.0)


# --- 1. agreement with the numpy originals -------------------------------
def test_pib_term_matches_roi_pib_metric() -> None:
    img = _frame()
    tensor = torch.as_tensor(img, dtype=torch.float64)[None, None]
    score, energy = roi_pib_metric(img, CENTER, "rectangle", 24.0, 4.0 / 3.0)
    assert float(pib_term(tensor, _mask())[0]) == pytest.approx(score, abs=1e-9)
    assert float(pib_term(tensor, _mask())[0]) == pytest.approx(energy, abs=1e-9)


def test_uniformity_term_matches_rms_pib_terms() -> None:
    img = _frame(1)
    tensor = torch.as_tensor(img, dtype=torch.float64)[None, None]
    _, rms_np = rms_pib_terms(img, CENTER, "rectangle", 24.0, 4.0 / 3.0)
    got = float(uniformity_term(tensor, _mask())[0])
    assert got == pytest.approx(rms_np, abs=1e-9)
    assert 0.0 <= got <= 1.0


def test_roi_energy_loss_matches_numpy() -> None:
    img = _frame(2)
    tensor = torch.as_tensor(img, dtype=torch.float64)[None, None]
    got = float(pib_term(tensor, _mask())[0])
    ref = np_roi_energy_loss(1.0, got)
    # float64 explicitly: torch.tensor defaults to float32, which silently
    # truncates `got` and made this compare a float32 against a float64.
    ours = roi_energy_loss(torch.tensor([got], dtype=torch.float64), 1.0)
    assert float(ours[0]) == pytest.approx(ref, abs=1e-12)


def test_roi_mask_is_the_canonical_mask() -> None:
    """The mask is a thin wrapper, so it must be bit-identical."""
    from ao_shaping.utils.image.target.metrics import target_shape_roi

    expected = target_shape_roi((H, W), CENTER, "rectangle", 24.0, 4.0 / 3.0)
    assert np.array_equal(_mask().numpy(), expected)


def test_both_terms_are_in_the_unit_interval() -> None:
    img = _frame(3)
    t = torch.as_tensor(img, dtype=torch.float64)[None, None]
    assert 0.0 <= float(pib_term(t, _mask())[0]) <= 1.0
    assert 0.0 <= float(uniformity_term(t, _mask())[0]) <= 1.0


# --- 2. gradients flow ----------------------------------------------------
@pytest.mark.parametrize(
    ("cfg", "key"),
    [
        (LossConfig(w_mse=1.0), "mse"),
        (LossConfig(w_mse=0.0, w_pib=1.0), "pib"),
        (LossConfig(w_mse=0.0, w_uniformity=1.0), "uniformity"),
        (LossConfig(w_mse=0.0, w_poisson=1.0, normalization="none"), "poisson_nll"),
    ],
)
def test_every_term_is_differentiable(cfg: LossConfig, key: str) -> None:
    t = torch.as_tensor(_frame(4), dtype=torch.float64)[None, None].clone()
    t.requires_grad_(True)
    target = torch.as_tensor(_frame(5), dtype=torch.float64)[None, None]

    out = composite_loss(t, target, _mask(), cfg)
    assert key in out, f"{key} not reported; got {sorted(out)}"
    out["_mean_total"].backward()

    assert t.grad is not None
    assert torch.isfinite(t.grad).all()
    assert float(t.grad.abs().sum()) > 0.0, f"{key} produced a zero gradient"


def test_uniformity_gradient_is_nonzero_on_a_realistic_roi() -> None:
    """Guard against a term that only 'works' on synthetic noise."""
    img = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.mgrid[0:H, 0:W]
    img += 50.0 * np.exp(-(((xx - 20) ** 2 + (yy - 20) ** 2) / (2 * 8.0**2)))
    img += 50.0 * np.exp(-(((xx - 44) ** 2 + (yy - 44) ** 2) / (2 * 8.0**2)))
    t = torch.as_tensor(img)[None, None].clone().requires_grad_(True)
    out = composite_loss(
        t, torch.zeros_like(t), _mask(), LossConfig(w_mse=0.0, w_uniformity=1.0)
    )
    out["_mean_total"].backward()
    assert float(t.grad.abs().sum()) > 0.0


# --- 3. Poisson NLL specifics -------------------------------------------
def test_poisson_nll_matches_the_paper_formula() -> None:
    pred = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    target = torch.tensor([[1.5, 0.0, 3.0]], dtype=torch.float64)
    expected = (1.0 - 1.5 * np.log(1.0)) + (2.0 - 0.0) + (3.0 - 3.0 * np.log(3.0))
    assert float(poisson_nll(pred, target)[0]) == pytest.approx(expected, abs=1e-12)


def test_poisson_nll_tolerates_exact_zeros() -> None:
    """A far field has exact-zero background; unguarded log would give -inf."""
    pred = torch.zeros(1, 1, 4, 4, dtype=torch.float64, requires_grad=True)
    target = torch.ones(1, 1, 4, 4, dtype=torch.float64)
    loss = poisson_nll(pred, target)[0]
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_poisson_nll_penalises_stray_light_in_the_dark() -> None:
    """A pixel the target wants dark must cost exactly the stray intensity."""
    target = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
    dark = torch.tensor([[[[0.0, 0.0]]]], dtype=torch.float64)
    stray = torch.tensor([[[[0.0, 5.0]]]], dtype=torch.float64)
    assert float(poisson_nll(stray, target)[0]) == pytest.approx(5.0, abs=1e-6)
    assert float(poisson_nll(dark, target)[0]) == pytest.approx(0.0, abs=1e-6)


def test_poisson_rejects_peak_normalisation() -> None:
    """The guard that stops a meaningless loss being configured silently."""
    with pytest.raises(ValueError, match="normalization='none'"):
        LossConfig(w_poisson=1.0, normalization="peak")


# --- 4. shape / config validation ----------------------------------------
def test_spatial_shape_mismatch_is_rejected() -> None:
    t = torch.zeros(1, 1, H, W)
    with pytest.raises(ValueError, match="does not match mask"):
        pib_term(t, roi_mask((32, 32), (16.0, 16.0), "rectangle", 8.0, 4.0 / 3.0))


def test_all_zero_weights_is_rejected() -> None:
    t = torch.zeros(1, 1, H, W)
    with pytest.raises(ValueError, match="no non-zero weight"):
        composite_loss(t, t, _mask(), LossConfig(w_mse=0.0))


def test_composite_reports_every_configured_term() -> None:
    t = torch.as_tensor(_frame(6), dtype=torch.float64)[None, None]
    cfg = LossConfig(w_mse=1.0, w_pib=1.0, w_uniformity=1.0)
    out = composite_loss(t, t, _mask(), cfg)
    assert {"mse", "pib", "uniformity", "total", "_mean_total"} <= set(out)


def test_empty_roi_is_rejected_rather_than_dividing_by_zero() -> None:
    with pytest.raises(ValueError, match="is empty"):
        roi_mask((H, W), (-50.0, -50.0), "rectangle", 20.0, 4.0 / 3.0)


def test_eps_is_a_positive_floor() -> None:
    assert EPS > 0.0
