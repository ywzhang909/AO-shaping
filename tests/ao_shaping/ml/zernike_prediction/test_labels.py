"""Tests for ml.zernike_prediction.labels (T4: wrapped-phase Zernike recovery).

The gradient-domain IRLS recovery is validated against:
1. Real 0414 samples with known metadata coefficients (golden anchor).
2. Synthetic samples (tmp_path) generated with the capture-exact phase_gen.

Conventions under test:
- ``recover_coefficients`` returns (66,) float64, piston index 0 = 1.0.
- ``recover_run`` writes (65,) float32 sidecar ``labels.npy`` per sample.
- ``acceptance_check`` gates on phase_rmse_rad < 0.1 and rel_l2 < 0.05.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.zernike_prediction.labels import (
    acceptance_check,
    fit_error_metrics,
    recover_coefficients,
    recover_run,
)
from ml.zernike_prediction.phase_gen import coefficients_to_wrapped_gray

_REAL_ROOT = Path("data/slm_dual_spot")


def _require_real_data() -> None:
    if not (_REAL_ROOT / "20260414_171241").is_dir():
        pytest.skip("requires slm_dual_spot data at data/slm_dual_spot")


# ---------------------------------------------------------------------------
# 1. Golden anchor: recover a labeled 0414 sample near-exactly
# ---------------------------------------------------------------------------


def test_recover_0414_sample_exact() -> None:
    _require_real_data()
    sd = _REAL_ROOT / "20260414_171241" / "sample_0000"
    meta = json.loads((sd / "metadata.json").read_text())
    true = np.asarray(meta["phase_params"]["coefficients"], dtype=np.float64)

    coeffs = recover_coefficients(sd, refine=False)
    assert coeffs.shape == (66,)
    assert coeffs[0] == pytest.approx(1.0, abs=1e-9)  # piston restored

    err = np.abs(coeffs - true)
    err[0] = 0.0  # piston is invisible to the wrapped image
    assert err.max() < 0.15  # target tolerance (typical: ~5e-5)


def test_recover_0414_batch_acceptance() -> None:
    _require_real_data()
    run = _REAL_ROOT / "20260414_171241"
    passed = 0
    for i in (1, 5, 17, 50):
        sd = run / f"sample_{i:04d}"
        meta = json.loads((sd / "metadata.json").read_text())
        true = np.asarray(meta["phase_params"]["coefficients"], dtype=np.float64)
        coeffs = recover_coefficients(sd, refine=False)
        err = np.abs(coeffs - true)
        err[0] = 0.0
        if (err[1:] < 0.15).mean() > 0.95:
            passed += 1
    assert passed >= 3  # >=75% of samples recover within tolerance


# ---------------------------------------------------------------------------
# 2. Synthetic recovery roundtrip (tmp_path, no data dependency)
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_sample(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """One synthetic sample: random 65 non-piston coeffs -> wrapped gray files.

    Uses 512x512 (radius 256) so the Zernike basis is adequately sampled —
    smaller grids undersample the high-order gradient columns and degrade
    recovery (real data is 1200x1920, radius 600).
    """
    rng = np.random.default_rng(7)
    coeffs = np.zeros(66)
    coeffs[0] = 1.0
    nz = rng.choice(np.arange(1, 66), size=12, replace=False)
    coeffs[nz] = rng.uniform(-3.0, 3.0, size=12)

    height, width = 512, 512
    radius = min(height, width) // 2
    gray = coefficients_to_wrapped_gray(coeffs, 10, height, width, radius)
    sd = tmp_path / "run_a" / "sample_0000"
    sd.mkdir(parents=True)
    np.savetxt(sd / "phase.csv", gray, fmt="%d", delimiter=",")
    meta = {
        "phase_params": {"n_max": 10, "max_coeff": 3.0, "coefficients": []},
    }
    (sd / "metadata.json").write_text(json.dumps(meta))
    return sd, coeffs


def test_recover_synthetic_roundtrip(synthetic_sample: tuple[Path, np.ndarray]) -> None:
    sd, true = synthetic_sample
    coeffs = recover_coefficients(sd, refine=False)
    err = np.abs(coeffs - true)
    err[0] = 0.0
    assert err.max() < 0.15


def test_recover_run_writes_sidecars(tmp_path: Path, synthetic_sample: tuple[Path, np.ndarray]) -> None:
    sd, true = synthetic_sample
    summary = recover_run(tmp_path, "run_a", max_samples=1)
    assert summary["recovered"] == 1
    assert summary["failed"] == []
    sidecar = sd / "labels.npy"
    assert sidecar.exists()
    labels = np.load(sidecar)
    assert labels.shape == (65,)
    assert labels.dtype == np.float32
    assert np.abs(labels - true[1:].astype(np.float32)).max() < 0.15


def test_recover_run_skips_existing(tmp_path: Path, synthetic_sample: tuple[Path, np.ndarray]) -> None:
    recover_run(tmp_path, "run_a", max_samples=1)
    summary = recover_run(tmp_path, "run_a", max_samples=1)
    assert summary["recovered"] == 0
    assert summary["skipped"] == 1


# ---------------------------------------------------------------------------
# 3. Error metrics and acceptance gates
# ---------------------------------------------------------------------------


def test_fit_error_metrics_perfect(tmp_path: Path, synthetic_sample: tuple[Path, np.ndarray]) -> None:
    sd, true = synthetic_sample
    coeffs = recover_coefficients(sd, refine=False)
    gray = np.loadtxt(sd / "phase.csv", delimiter=",").astype(np.uint16)
    m = fit_error_metrics(gray, coeffs)
    assert m["mae_gray"] < 1.0  # reconstruction within 1 gray level on average
    assert m["rel_l2"] < 0.01
    assert acceptance_check(m, 3.0)


def test_acceptance_gates_fail_on_bad_coeffs(tmp_path: Path, synthetic_sample: tuple[Path, np.ndarray]) -> None:
    sd, true = synthetic_sample
    gray = np.loadtxt(sd / "phase.csv", delimiter=",").astype(np.uint16)
    bad = true.copy()
    bad[1:] += 1.0  # grossly wrong
    m = fit_error_metrics(gray, bad)
    assert not acceptance_check(m, 3.0)


# ---------------------------------------------------------------------------
# 4. Error handling
# ---------------------------------------------------------------------------


def test_recover_errors_handled(tmp_path: Path, synthetic_sample: tuple[Path, np.ndarray]) -> None:
    sd, _ = synthetic_sample
    (sd / "phase.csv").unlink()  # break the sample
    summary = recover_run(tmp_path, "run_a", max_samples=1)
    assert summary["recovered"] == 0
    assert len(summary["failed"]) == 1
    assert summary["failed"][0][0] == "sample_0000"
