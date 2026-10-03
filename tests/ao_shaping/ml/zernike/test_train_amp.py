"""Unit tests for the training driver in `src/ml/zernike/train_amp.py`.

Focus is on the pure helpers -- the metric definitions, which are the part that
can silently lie. The end-to-end training run needs the 65 GB corpus and is
exercised manually, not here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402
from ml.zernike.train_amp import (  # noqa: E402
    AmpTrainConfig,
    grad_statistics,
    regression_perplexity,
    render_comparison,
)


def _model(n_max: int = 4, grid: int = 8) -> ZernikeAmpModel:
    return ZernikeAmpModel(ZernikeAmpConfig(n_max=n_max, grid=grid))


class TestRegressionPerplexity:
    """The metric is exp(NMSE); these pin that definition exactly."""

    def test_perfect_fit_is_one(self) -> None:
        assert regression_perplexity(0.0, 4.0) == pytest.approx(1.0)

    def test_equals_exp_of_nmse(self) -> None:
        assert regression_perplexity(0.5, 2.0) == pytest.approx(math.exp(0.25))

    def test_monotonic_in_mse(self) -> None:
        values = [regression_perplexity(m, 1.0) for m in (0.0, 0.1, 0.5, 1.0)]
        assert values == sorted(values)

    def test_degenerate_inputs_are_nan_not_a_crash(self) -> None:
        # A constant target has zero variance, so NMSE is undefined; NaN is the
        # honest answer and must not raise.
        assert math.isnan(regression_perplexity(0.1, 0.0))
        assert math.isnan(regression_perplexity(float("nan"), 1.0))

    def test_does_not_overflow(self) -> None:
        assert math.isfinite(regression_perplexity(1e9, 1e-9))


class TestGradStatistics:
    """Per-mode gradient norms are how a dead direction is detected."""

    def test_reports_nothing_before_backward(self) -> None:
        model = _model()
        stats = grad_statistics(model)
        assert stats == {"total": 0.0, "max": 0.0, "min": 0.0, "dead": 0.0}

    def test_counts_dead_modes(self) -> None:
        model = _model(n_max=4)
        grad = torch.zeros(model.K)
        grad[0] = 1.0  # only one mode alive
        model.coefficients.grad = grad
        stats = grad_statistics(model)
        assert stats["max"] == pytest.approx(1.0)
        assert stats["dead"] == model.K - 1
        assert stats["total"] == pytest.approx(1.0)

    def test_total_is_the_l2_norm(self) -> None:
        model = _model()
        model.coefficients.grad = torch.full((model.K,), 3.0)
        stats = grad_statistics(model)
        assert stats["total"] == pytest.approx(3.0 * math.sqrt(model.K))
        assert stats["max"] == pytest.approx(3.0)
        assert stats["dead"] == 0


class TestRenderComparison:
    """The figure is a required deliverable, so at least prove it writes."""

    def test_writes_a_png(self, tmp_path) -> None:
        path = tmp_path / "cmp.png"
        grid = 16
        rng = np.random.default_rng(0)
        render_comparison(
            rng.random((grid, grid)),
            rng.random((grid, grid)),
            rng.standard_normal((grid, grid)),
            path,
            "unit test",
        )
        assert path.is_file() and path.stat().st_size > 0


class TestConfigDefaults:
    """CLI defaults are derived from ZernikeAmpConfig, which must not drift.

    A literal ``--far-field-padding default=1`` once silently overrode the
    calibrated default of 10 and made every run start at R^2 = -0.38.
    """

    def test_parser_padding_matches_the_model_default(self) -> None:
        from ml.zernike.train_amp import _build_parser

        args = _build_parser().parse_args([])
        assert args.far_field_padding == ZernikeAmpConfig().far_field_padding
        assert args.center_crop is ZernikeAmpConfig().center_crop
        assert args.n_max == ZernikeAmpConfig().n_max
        assert args.grid == ZernikeAmpConfig().grid

    def test_no_center_crop_flag_flips_it(self) -> None:
        from ml.zernike.train_amp import _build_parser

        args = _build_parser().parse_args(["--no-center-crop"])
        assert args.center_crop is False

    def test_train_config_defaults_are_usable(self) -> None:
        cfg = AmpTrainConfig()
        assert cfg.far_field_padding == ZernikeAmpConfig().far_field_padding
        assert cfg.observable in ("amplitude", "intensity")
        assert cfg.grid == ZernikeAmpConfig().grid