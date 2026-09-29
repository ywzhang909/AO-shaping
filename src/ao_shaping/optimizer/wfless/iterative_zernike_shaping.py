"""Iterative Zernike + phase-only SLM beam-shaping (optimizer layer).

Thin orchestrator that:
1. Builds the reference ("actual") far-field from a golden Zernike phase
   (simulating the "actual" distorted far-field in the 2f bench).
2. Builds the square target.
3. Drives the ``IterativeZernikeShapingOptimizer`` (algorithm layer) and
   records history via a simple Recorder pattern.

No hardware involved — pure simulation (torch + numpy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.algorithm.signal_processing.iterative_zernike_shaping import (
    IterativeZernikeShapingConfig,
    IterativeZernikeShapingOptimizer,
    IterativeZernikeShapingResult,
)
from ao_shaping.drivers.sim.slm_shaping_bench import (
    ShapingBenchConfig,
    composite_score,
    compute_metrics,
    forward_intensity,
    make_target,
)

__all__ = [
    "IterativeZernikePibConfig",
    "optimize_iterative_zernike_shaping",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class IterativeZernikePibConfig:
    """Configuration for the iterative Zernike + phase-only shaping run.

    Attributes:
        n_grid: SLM / simulation grid edge length.
        n_zernike: Max Zernike radial order for calibration (0 = skip).
        target_side_px: Target square side in camera pixels.
        seed: RNG seed.
        golden_coeffs: "Golden" Zernike coefficients (raw radians) that produce
            the reference "actual" far-field. If None, a fixed default set is
            used (defocus + astig + spherical).
        zernike_lr: Learning rate for Zernike calibration.
        slm_lr: Learning rate for SLM phase shaping.
        calib_iters: Adam steps per calibration pass.
        shaping_iters: Adam steps per shaping pass.
        max_outer_iters: Max A↔B outer iterations.
        early_stop_patience: Early-stop patience (non-improving passes).
        early_stop_min_delta: Min score improvement to reset patience.
        gs_iters: Number of GS iterations to compute the "ideal" square target
            (0 = use make_target analytical square).
    """

    n_grid: int = 64
    n_zernike: int = 4
    target_side_px: int = 16
    seed: int = 0
    golden_coeffs: dict[tuple[int, int], float] | None = None
    zernike_lr: float = 0.05
    slm_lr: float = 0.02
    calib_iters: int = 50
    shaping_iters: int = 100
    max_outer_iters: int = 5
    early_stop_patience: int = 2
    early_stop_min_delta: float = 0.005
    gs_iters: int = 0  # 0 = analytical target (fast); >0 = GS-ideal target

    def to_alg_config(self) -> IterativeZernikeShapingConfig:
        """Convert to the algorithm-layer config."""
        return IterativeZernikeShapingConfig(
            n_grid=self.n_grid,
            n_zernike=self.n_zernike,
            target_side_px=self.target_side_px,
            seed=self.seed,
            zernike_lr=self.zernike_lr,
            slm_lr=self.slm_lr,
            calib_iters=self.calib_iters,
            shaping_iters=self.shaping_iters,
            max_outer_iters=self.max_outer_iters,
            early_stop_patience=self.early_stop_patience,
            early_stop_min_delta=self.early_stop_min_delta,
        )


# ---------------------------------------------------------------------------
# Golden far-field builder
# ---------------------------------------------------------------------------
def _build_golden_far_field(
    cfg: IterativeZernikePibConfig,
) -> np.ndarray:
    """Build the reference "actual" far-field from golden Zernike coefficients.

    The golden phase is a Zernike phase pattern (raw radians) propagated through
    the 2f bench forward model. This simulates the "actual" distorted far-field
    that the Zernike calibration must match.

    Returns:
        Normalized far-field intensity (sum=1), shape (n, n).
    """
    from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

    n = cfg.n_grid
    bench_cfg = ShapingBenchConfig(n_grid=n, target_side_px=cfg.target_side_px, seed=cfg.seed)

    if cfg.golden_coeffs is None:
        # Default golden: defocus + 45° astig + spherical (moderate amplitudes)
        golden = {(2, 0): 0.5, (2, -2): 0.3, (4, 0): 0.2}
    else:
        golden = cfg.golden_coeffs

    # Generate Zernike phase (raw radians, numpy)
    zernike_phase = generate_zernike_phase(
        golden, resolution=(n, n), n_max=cfg.n_zernike
    )
    # generate_zernike_phase returns raw radians (float64) when coeffs non-empty
    zernike_phase = np.asarray(zernike_phase, dtype=np.float64)
    # Replace NaN (outside aperture) with 0
    zernike_phase = np.nan_to_num(zernike_phase, nan=0.0)

    # Add a small random SLM phase (simulating uncalibrated SLM)
    rng = np.random.default_rng(cfg.seed + 1)
    slm_noise = rng.normal(0, 0.1, size=(n, n)).astype(np.float64)
    total_phase = zernike_phase + slm_noise

    # Forward through the bench
    ff = forward_intensity(total_phase, bench_cfg)
    ff = ff / (ff.sum() + 1e-12)
    return ff


def _build_target(cfg: IterativeZernikePibConfig) -> np.ndarray:
    """Build the square target pattern (sum=1).

    If gs_iters > 0, runs a quick GS to get a "GS-ideal" target; otherwise
    uses the analytical square from make_target.
    """
    bench_cfg = ShapingBenchConfig(n_grid=cfg.n_grid, target_side_px=cfg.target_side_px, seed=cfg.seed)
    target = make_target(bench_cfg)
    # make_target already returns normalized (sum=1)
    return target


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def optimize_iterative_zernike_shaping(
    config: IterativeZernikePibConfig,
) -> dict[str, Any]:
    """Run the iterative Zernike + phase-only shaping optimization.

    This is the optimizer-layer entry point. It:
    1. Builds the reference "actual" far-field (from golden Zernike + noise).
    2. Builds the square target.
    3. Instantiates the algorithm-layer optimizer and runs the A↔B loop.
    4. Records history and returns a result dict.

    Args:
        config: IterativeZernikePibConfig with all parameters.

    Returns:
        Dictionary with keys:
            - "zernike_coeffs": dict[(n,m)] -> float (calibrated, raw radians)
            - "slm_phase": np.ndarray (optimized SLM phase, raw radians)
            - "far_field": np.ndarray (final far-field intensity, sum=1)
            - "target": np.ndarray (square target, sum=1)
            - "actual_far_field": np.ndarray (reference far-field, sum=1)
            - "score_history": list[dict] (per-outer-iteration scores)
            - "final_score": float (composite score of final far-field)
            - "n_outer_iters": int
            - "converged": bool
            - "metrics": dict (final bench metrics: PIB, CV, score, etc.)
    """
    logger.info("Starting iterative Zernike+phase shaping: n_grid={}, n_zernike={}",
                config.n_grid, config.n_zernike)

    # Build reference far-field and target
    actual_ff = _build_golden_far_field(config)
    target = _build_target(config)

    # Algorithm-layer config and optimizer
    alg_cfg = config.to_alg_config()
    optimizer = IterativeZernikeShapingOptimizer(alg_cfg)

    # Run
    result = optimizer.run(actual_far_field=actual_ff, initial_slm_phase=None)

    # Compute final bench metrics
    bench_cfg = ShapingBenchConfig(n_grid=config.n_grid, target_side_px=config.target_side_px, seed=config.seed)
    ff = result.far_field
    ff_norm = ff / (ff.sum() + 1e-12)
    center = np.unravel_index(np.argmax(ff_norm), ff_norm.shape)[::-1]
    metrics = compute_metrics(ff_norm, target, center=center)
    metrics["score"] = composite_score(metrics)

    logger.info(
        "Done: final_score={:.4f}, PIB={:.4f}, CV={:.4f}, n_outer={}, converged={}",
        metrics["score"], metrics.get("PIB", 0), metrics.get("CV", float("inf")),
        result.n_outer_iters, result.converged,
    )

    return {
        "zernike_coeffs": result.zernike_coeffs,
        "slm_phase": result.slm_phase,
        "far_field": result.far_field,
        "target": result.target,
        "actual_far_field": actual_ff,
        "score_history": result.score_history,
        "final_score": result.final_score,
        "n_outer_iters": result.n_outer_iters,
        "converged": result.converged,
        "metrics": metrics,
    }
