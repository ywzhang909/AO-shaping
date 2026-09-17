"""Regression tests for differentiable beam shaping quality and default-weight correctness.

Promoted from the diagnostic scripts that located the default-weight bug (old
defaults lr=1e-2, w=[.4,.4,.1,.1] collapsed encircled energy to ~0.07) into
permanent pytest coverage.  Every class locks in a verified operating point so
future regressions are caught immediately.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module when torch is absent (optional dependency).
torch = pytest.importorskip("torch")

from ao_shaping.algorithm.differentiable_shaping import (  # noqa: E402
    angular_spectrum_propagate_torch,
    create_target_mask,
    train_beam_shaping,
)
from ao_shaping.algorithm.gerchberg_saxton import (  # noqa: E402
    angular_spectrum_propagate as angular_spectrum_propagate_numpy,
)

# Small grid for fast tests.
GRID = (128, 128)
SIZE = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _cv_ee(intensity: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Coefficient of variation and encircled energy inside the target mask."""
    mask = target > 0
    vals = intensity[mask]
    cv = float(vals.std() / (vals.mean() + 1e-12))
    ee = float((intensity * mask).sum() / (intensity.sum() + 1e-12))
    return cv, ee


# ============================ ASM parity ============================

class TestAsmParity:
    """Verify torch ASM matches the numpy reference implementation."""

    def test_torch_matches_numpy(self):
        rng = np.random.default_rng(0)
        field_np = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
        out_numpy = angular_spectrum_propagate_numpy(field_np, 8e-6, 0.1, 1064e-9)
        field_torch = torch.from_numpy(field_np).to(
            dtype=torch.complex128, device="cpu",
        )
        out_torch = angular_spectrum_propagate_torch(field_torch, 8e-6, 0.1, 1064e-9)
        np.testing.assert_allclose(
            np.asarray(out_torch.cpu()), out_numpy, atol=1e-6,
            err_msg="Torch ASM diverges from numpy reference",
        )


# ============================ Square quality ============================

class TestSquareQuality:
    """Verify convergence for square targets under winning configs."""

    def _run(self, **kw):
        target = create_target_mask("square", GRID, SIZE)
        defaults = dict(
            iterations=600,
            lr=3e-2,
            w_uniformity=0.4,
            w_efficiency=0.6,
            w_zero_order=0.0,
            w_smoothness=0.0,
            seed=0,
            device=DEVICE,
        )
        defaults.update(kw)
        return train_beam_shaping(target, GRID, **defaults)

    def test_adam_winning_config_converges(self):
        result = self._run(propagation="fft", optimizer="adam")
        cv, ee = _cv_ee(result.simulated_intensity, result.target_intensity)
        assert result.loss_history[-1] <= result.loss_history[0] + 1e-9, (
            f"Loss did not decrease: {result.loss_history[0]:.4f} -> {result.loss_history[-1]:.4f}"
        )
        assert cv < 0.3, f"CV={cv:.3f}"
        assert ee > 0.5, f"EE={ee:.3f}"

    def test_asm_winning_config_converges(self):
        result = self._run(propagation="asm", optimizer="adam")
        cv, ee = _cv_ee(result.simulated_intensity, result.target_intensity)
        assert result.loss_history[-1] <= result.loss_history[0] + 1e-9, (
            f"Loss did not decrease: {result.loss_history[0]:.4f} -> {result.loss_history[-1]:.4f}"
        )
        assert cv < 0.5, f"CV={cv:.3f}"
        assert ee > 0.4, f"EE={ee:.3f}"


# ============================ Spot quality ============================

class TestSpotQuality:
    """Verify convergence for a circular (spot) target."""

    def _run(self, **kw):
        # Build a circular mask directly (radius 28 px on 128x128 grid).
        yy, xx = np.mgrid[:128, :128]
        r = np.hypot(xx - 63.5, yy - 63.5)
        mask = (r <= 28).astype(np.float64)
        defaults = dict(
            iterations=600,
            lr=3e-2,
            w_uniformity=0.4,
            w_efficiency=0.6,
            w_zero_order=0.0,
            w_smoothness=0.0,
            seed=0,
            device=DEVICE,
        )
        defaults.update(kw)
        return train_beam_shaping(mask, GRID, **defaults)

    def test_spot_converges(self):
        result = self._run(propagation="fft", optimizer="adam")
        cv, ee = _cv_ee(result.simulated_intensity, result.target_intensity)
        assert result.loss_history[-1] <= result.loss_history[0] + 1e-9, (
            f"Loss did not decrease: {result.loss_history[0]:.4f} -> {result.loss_history[-1]:.4f}"
        )
        assert cv < 0.3, f"CV={cv:.3f}"
        assert ee > 0.5, f"EE={ee:.3f}"


# ============================ Weight regression guard ============================

class TestWeightRegression:
    """Guard against reversion of the default-weight fix.

    The old defaults (lr=1e-2, w_uniformity=0.4, w_efficiency=0.4,
    w_zero_order=0.1, w_smoothness=0.1) collapsed encircled energy to
    ~0.07 because the zero-order penalty pushed energy out of the centred
    target.  The new defaults (lr=3e-2, w=[.4,.6,0,0]) achieve EE~0.84.

    Measured on 256x256: ee_old ~ 0.07, ee_new ~ 0.84 (gap ~0.77).
    Thresholds below use ≥2x margin on the 128x128 grid.
    """

    def _run(self, **kw):
        target = create_target_mask("square", GRID, SIZE)
        defaults = dict(
            iterations=300,
            seed=0,
            device=DEVICE,
        )
        defaults.update(kw)
        return train_beam_shaping(target, GRID, **defaults)

    def test_new_defaults_outperform_old_defaults(self):
        result_old = self._run(
            lr=1e-2,
            w_uniformity=0.4,
            w_efficiency=0.4,
            w_zero_order=0.1,
            w_smoothness=0.1,
        )
        result_new = self._run(
            lr=3e-2,
            w_uniformity=0.4,
            w_efficiency=0.6,
            w_zero_order=0.0,
            w_smoothness=0.0,
        )
        _, ee_old = _cv_ee(result_old.simulated_intensity, result_old.target_intensity)
        _, ee_new = _cv_ee(result_new.simulated_intensity, result_new.target_intensity)
        # Old config was broken — EE far below usable quality.
        assert ee_old < 0.3, f"Old defaults unexpectedly good: EE={ee_old:.3f}"
        # New config must clearly beat old.
        assert ee_new > ee_old, (
            f"New defaults not better: ee_new={ee_new:.3f} vs ee_old={ee_old:.3f}"
        )
        assert ee_new - ee_old > 0.1, (
            f"Gap too small: delta={ee_new - ee_old:.3f}"
        )


# ============================ L-BFGS quality ============================

class TestLbfgsQuality:
    """Verify L-BFGS converges well on a square target."""

    def _run(self, **kw):
        target = create_target_mask("square", GRID, SIZE)
        defaults = dict(
            lr=1.0,
            w_uniformity=0.4,
            w_efficiency=0.6,
            w_zero_order=0.0,
            w_smoothness=0.0,
            seed=0,
            device=DEVICE,
        )
        defaults.update(kw)
        return train_beam_shaping(target, GRID, **defaults)

    def test_lbfgs_flat_top(self):
        result = self._run(propagation="fft", optimizer="lbfgs", iterations=30)
        cv, ee = _cv_ee(result.simulated_intensity, result.target_intensity)
        assert result.loss_history[-1] <= result.loss_history[0] + 1e-9, (
            f"Loss did not decrease: {result.loss_history[0]:.4f} -> {result.loss_history[-1]:.4f}"
        )
        assert cv < 0.3, f"CV={cv:.3f}"
        assert ee > 0.4, f"EE={ee:.3f}"


# ============================ Determinism ============================

# NOTE: TestDeterministic is intentionally omitted — the sibling file
# test_differentiable_shaping.py already contains test_deterministic_with_seed
# with an equivalent assertion pattern (two runs, seed=42, allclose on phase
# and loss_history).  Duplicating it here would be redundant.
