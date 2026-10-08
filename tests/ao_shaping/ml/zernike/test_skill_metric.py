"""The constant-predictor floor and the `skill` scoreboard.

Locks the property that motivated them: on a near-static target, R^2 is dominated
by the target's own variance and says almost nothing about model quality, while
`skill = 1 - mse_model / mse_const` does. See
`report/zernike_r2_baseline/report.md` for the measurement that forced the
metric change.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.zernike.metrics import batch_image_metrics, constant_baseline_metrics


def _frames(n: int = 16, grid: int = 8, seed: int = 0) -> torch.Tensor:
    """Frames with a strong invariant component plus a small varying one.

    That ratio is the whole point: it reproduces the corpus, where a constant
    predictor reaches R^2 ~ 0.91.
    """
    g = torch.Generator().manual_seed(seed)
    static = torch.zeros(grid, grid)
    static[grid // 2, grid // 2] = 1.0            # invariant 0-order
    base = torch.rand(1, 1, grid, grid, generator=g) * 0.02   # tiny speckle
    return (static + base).expand(n, 1, grid, grid).contiguous()


class TestConstantBaseline:
    def test_a_constant_predictor_already_scores_a_high_r2(self) -> None:
        """The trap this metric exists to defuse."""
        target = _frames()
        constant = target.mean(dim=0, keepdim=True).expand_as(target)
        r2 = batch_image_metrics(constant, target)["r2"]
        assert r2 > 0.8, (
            "fixture no longer reproduces the corpus regime: a constant predictor "
            f"scored R2={r2:.3f}, so this test would no longer be guarding anything"
        )

    def test_skill_of_the_optimal_constant_is_zero(self) -> None:
        target = _frames()
        constant = target.mean(dim=0, keepdim=True).expand_as(target)
        assert constant_baseline_metrics(constant, target)["skill"] == pytest.approx(
            0.0, abs=1e-6
        )

    def test_perfect_prediction_scores_one(self) -> None:
        target = _frames()
        out = constant_baseline_metrics(target.clone(), target)
        assert out["skill"] == pytest.approx(1.0, abs=1e-6)
        assert out["mse_const"] > 0.0

    def test_skill_is_negative_when_worse_than_the_constant(self) -> None:
        target = _frames()
        # Scale the *mean* away from the target: a systematically wrong image.
        bad = target.mean(dim=0, keepdim=True) * 1.5
        assert constant_baseline_metrics(bad, target)["skill"] < 0.0

    def test_skill_orders_two_models_the_same_way_mse_does(self) -> None:
        """Skill is a monotone rescaling of MSE for one fixed baseline."""
        target = _frames()
        near = target + 0.01
        far = target + 0.10
        s_near = constant_baseline_metrics(near, target)["skill"]
        s_far = constant_baseline_metrics(far, target)["skill"]
        assert s_near > s_far

    def test_high_r2_is_not_evidence_of_a_good_model(self) -> None:
        """The reason both numbers are reported.

        On this corpus a constant predictor reaches R^2 ~ 0.91, so two models can
        be separated by a fraction of an R^2 point while one of them does nothing
        at all. Here the constant scores R^2 ~ 0.91 with skill exactly 0, and a
        perfect prediction scores R^2 = 1 -- an R^2 gap of ~0.09 that says a model
        went from "nothing" to "perfect". Compressing that gap is what `skill`
        does, and why a model table read on R^2 alone is unreadable.
        """
        target = _frames()
        constant = target.mean(dim=0, keepdim=True).expand_as(target)
        r2_const = batch_image_metrics(constant, target)["r2"]
        r2_perfect = batch_image_metrics(target.clone(), target)["r2"]
        skill_const = constant_baseline_metrics(constant, target)["skill"]
        skill_perfect = constant_baseline_metrics(target.clone(), target)["skill"]

        # R^2 spans [~0.91, 1.0] -- a range that looks like "everything is fine".
        assert r2_const > 0.8 and r2_perfect == pytest.approx(1.0, abs=1e-6)
        # skill spans [0, 1] -- "nothing" to "perfect", with no inflated floor.
        assert skill_const == pytest.approx(0.0, abs=1e-6)
        assert skill_perfect == pytest.approx(1.0, abs=1e-6)

    def test_explicit_baseline_overrides_the_split_mean(self) -> None:
        target = _frames()
        weak = torch.zeros_like(target)          # a much worse-than-mean constant
        with_default = constant_baseline_metrics(target.clone(), target)
        with_zero = constant_baseline_metrics(target.clone(), target, baseline=weak)
        # A zero baseline has larger error, so the same perfect prediction scores
        # *less* skill against it -- i.e. the baseline choice is not cosmetic.
        assert with_zero["mse_const"] > with_default["mse_const"]

    def test_returns_the_floor_so_the_record_is_self_describing(self) -> None:
        target = _frames()
        out = constant_baseline_metrics(target.clone(), target)
        assert out["var"] == pytest.approx(float(torch.var(target)), rel=1e-5)
        assert out["r2_const"] == pytest.approx(
            1.0 - out["mse_const"] / out["var"], rel=1e-5
        )

    def test_shape_mismatch_is_rejected(self) -> None:
        with pytest.raises(RuntimeError):
            constant_baseline_metrics(_frames(4), _frames(5))
