"""Light smoke tests for ml.zernike_prediction.plots (no data dependency).

Only synthetic coefficient arrays are used; every figure is written to a
pytest ``tmp_path`` and asserted to exist (and be non-empty / PNG).
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from ml.zernike_prediction import plots

# 65 non-piston Zernike coefficients (n_max=10, piston at index 0 = 1.0)
N_COEFFS = 65


def _synthetic_pair(n_samples: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pred/true (n, 65) arrays with a strong linear correlation."""
    rng = np.random.default_rng(seed)
    true = rng.normal(0.0, 0.3, size=(n_samples, N_COEFFS))
    pred = true + rng.normal(0.0, 0.05, size=(n_samples, N_COEFFS))
    return pred, true


def _small_coeffs(seed: int = 1) -> np.ndarray:
    """One (65,) coefficient vector with a few nonzero terms."""
    rng = np.random.default_rng(seed)
    coeffs = np.zeros(N_COEFFS, dtype=np.float64)
    coeffs[1:6] = rng.normal(0.0, 0.2, size=5)  # low-order terms only
    return coeffs


def test_scatter_creates_file(tmp_path) -> None:
    pred, true = _synthetic_pair()
    out = plots.predict_true_scatter(pred, true, tmp_path / "scatter.png")
    assert out.is_file()
    assert out.stat().st_size > 10_000
    assert out.suffix == ".png"


def test_bar_creates_file(tmp_path) -> None:
    pred, true = _synthetic_pair()
    order_path = plots.per_order_mae_bar(pred, true, tmp_path / "order_mae.png")
    coeff_path = plots.per_coeff_mae_bar(pred, true, tmp_path / "coeff_mae.png")
    for p in (order_path, coeff_path):
        assert p.is_file()
        assert p.stat().st_size > 0
        assert p.suffix == ".png"


def test_phase_grid_creates_file(tmp_path) -> None:
    coeffs = [_small_coeffs(1), _small_coeffs(2)]
    labels = ["pred", "true"]
    out = plots.phase_grid(coeffs, labels, tmp_path / "grid.png", size=(96, 96))
    assert out.is_file()
    assert out.stat().st_size > 0
    assert out.suffix == ".png"


def test_phase_error_map_and_all_plots(tmp_path) -> None:
    pred, true = _synthetic_pair()
    err_path = plots.phase_error_map(
        pred[0], true[0], tmp_path / "phase_error.png", size=(96, 96)
    )
    assert err_path.is_file()

    out_dir = tmp_path / "all"
    result = plots.all_plots(pred, true, out_dir, prefix="eval", phase_size=(96, 96))
    assert isinstance(result, dict)
    assert len(result) >= 4
    for name, p in result.items():
        assert p.is_file(), f"missing file for {name}"
        assert p.stat().st_size > 0


def test_coefficients_66_accepted(tmp_path) -> None:
    """Both 65- and 66-length (piston-first) coefficient vectors must work."""
    c65 = _small_coeffs(3)
    c66 = np.concatenate([[1.0], c65])  # piston at index 0

    grid_out = plots.phase_grid(
        [c65, c66], ["65", "66"], tmp_path / "grid66.png", size=(96, 96)
    )
    assert grid_out.is_file()

    err_out = plots.phase_error_map(c66, c65, tmp_path / "err66.png", size=(96, 96))
    assert err_out.is_file()


def test_raises_on_shape_mismatch(tmp_path) -> None:
    pred, _ = _synthetic_pair()
    true_bad = np.zeros((30, 60), dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        plots.predict_true_scatter(pred, true_bad, tmp_path / "bad.png")


def test_aggregate_backend_set() -> None:
    assert matplotlib.get_backend() == "Agg"
