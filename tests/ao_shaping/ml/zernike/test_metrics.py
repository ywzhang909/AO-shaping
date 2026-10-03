"""Unit tests for `src/ml/zernike/metrics.py`.

These pin the *definitions*, because a metric that silently computes something
other than what its name says is worse than no metric at all.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.zernike.metrics import (  # noqa: E402
    available_perceptual_metrics,
    batch_image_metrics,
    per_sample_beam_metrics,
    psnr,
    summarise_beam_metrics,
)


def _field(grid: int = 32, seed: int = 0) -> np.ndarray:
    """A centred Gaussian-ish spot, the shape these metrics are meant for."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:grid, 0:grid]
    centre = (grid - 1) / 2
    spot = np.exp(-((x - centre) ** 2 + (y - centre) ** 2) / (2 * 3.0**2))
    return spot + 0.01 * rng.random((grid, grid))


class TestBatchImageMetrics:
    def test_identical_images_are_perfect(self) -> None:
        x = torch.rand(3, 1, 16, 16)
        m = batch_image_metrics(x, x)
        assert m["mse"] == pytest.approx(0.0, abs=1e-12)
        assert m["mae"] == pytest.approx(0.0, abs=1e-12)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert m["ssim"] == pytest.approx(1.0, abs=1e-6)
        assert m["r2"] == pytest.approx(1.0, abs=1e-6)

    def test_zeros_against_data_is_the_worst_case(self) -> None:
        x = torch.rand(2, 1, 16, 16)
        m = batch_image_metrics(torch.zeros_like(x), x)
        # SSIM of an all-zero image against data is ~0 but not exactly 0.
        assert m["ssim"] == pytest.approx(0.0, abs=1e-4)
        assert m["r2"] < 0.0
        assert m["mse"] > 0.0

    def test_nrmse_is_rmse_over_target_range(self) -> None:
        pred = torch.zeros(1, 1, 8, 8)
        target = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 8, 8)
        m = batch_image_metrics(pred, target)
        assert m["nrmse"] == pytest.approx(m["rmse"] / 1.0, rel=1e-5)

    def test_mae_is_the_mean_absolute_error(self) -> None:
        # 8x8, not 4x4: torchmetrics' SSIM needs its 5px window to fit.
        pred = torch.zeros(1, 1, 8, 8)
        target = torch.full((1, 1, 8, 8), 0.5)
        m = batch_image_metrics(pred, target)
        assert m["mae"] == pytest.approx(0.5)
        assert m["mse"] == pytest.approx(0.25)

    def test_psnr_is_ten_log10_one_over_mse(self) -> None:
        pred = torch.zeros(1, 1, 8, 8)
        target = torch.full((1, 1, 8, 8), 0.1)
        m = batch_image_metrics(pred, target)
        assert m["psnr"] == pytest.approx(10.0 * math.log10(1.0 / m["mse"]), rel=1e-5)

    def test_psnr_helper_matches_the_identity(self) -> None:
        x = torch.rand(2, 1, 8, 8)
        assert float(psnr(x, x).mean()) > 100.0

    def test_metrics_worsen_monotonically_with_noise(self) -> None:
        target = torch.rand(1, 1, 32, 32, generator=torch.Generator().manual_seed(0))
        scores = [
            batch_image_metrics(
                (target + level * torch.rand(1, 1, 32, 32)).clamp(0, 1), target
            )["ssim"]
            for level in (0.0, 0.05, 0.2)
        ]
        assert scores[0] > scores[1] > scores[2]


class TestPerSampleBeamMetrics:
    def test_identical_patterns_score_perfectly(self) -> None:
        spot = _field()
        m = per_sample_beam_metrics(spot, spot)
        assert m["correlation"] == pytest.approx(1.0)
        assert m["efficiency"] == pytest.approx(1.0)
        assert m["centroid_offset_px"] == pytest.approx(0.0, abs=1e-6)
        assert m["spot_diameter_ratio"] == pytest.approx(1.0)
        assert m["peak_ratio"] == pytest.approx(1.0)

    def test_a_shifted_spot_moves_the_centroid(self) -> None:
        spot = _field(grid=32)
        shifted = np.roll(spot, shift=(5, 0), axis=0)
        m = per_sample_beam_metrics(shifted, spot)
        assert m["centroid_offset_px"] > 3.0
        # Width is unchanged by a shift -- that is the point of separating the two.
        assert m["spot_diameter_ratio"] == pytest.approx(1.0, rel=0.05)

    def test_a_wider_spot_changes_the_diameter_ratio(self) -> None:
        """No noise floor: a uniform pedestal would dominate the 90% EE radius
        and mask the width difference entirely."""
        grid = 48
        y, x = np.mgrid[0:grid, 0:grid]
        centre = (grid - 1) / 2
        narrow = np.exp(-((x - centre) ** 2 + (y - centre) ** 2) / (2 * 2.0**2))
        wide = np.exp(-((x - centre) ** 2 + (y - centre) ** 2) / (2 * 6.0**2))
        m = per_sample_beam_metrics(wide, narrow)
        assert m["spot_diameter_ratio"] > 1.5

    def test_scale_invariance(self) -> None:
        """The SLM+CCD chain has unknown absolute gain, so gain must not matter."""
        spot = _field()
        m = per_sample_beam_metrics(spot * 37.0, spot)
        assert m["efficiency"] == pytest.approx(1.0)
        assert m["correlation"] == pytest.approx(1.0)

    def test_zero_prediction_is_reported_not_crashed(self) -> None:
        m = per_sample_beam_metrics(np.zeros((32, 32)), _field())
        assert m["peak_ratio"] == 0.0
        assert m["efficiency"] == 0.0

    def test_summarise_averages_keys(self) -> None:
        rows = [
            per_sample_beam_metrics(_field(seed=0), _field(seed=0)),
            per_sample_beam_metrics(_field(seed=1), _field(seed=1)),
        ]
        summary = summarise_beam_metrics(rows)
        assert set(summary) == set(rows[0])
        assert all(math.isfinite(v) for v in summary.values())

    def test_summarise_empty_is_empty(self) -> None:
        assert summarise_beam_metrics([]) == {}


def test_perceptual_metric_availability_is_reported_honestly() -> None:
    """Whatever it returns must be a tuple of names that really import."""
    available = available_perceptual_metrics()
    assert isinstance(available, tuple)
    for name in available:
        assert isinstance(name, str) and name