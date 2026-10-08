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
    compute_bench_metrics,
    forward_intensity,
    gs_shape,
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
        n_zernike: Max Zernike radial order for calibration. Default 0 (skip):
            the Zernike calibration pass is a documented negative result on this
            model -- its leaky low-order estimate is redundant with the free-form
            stage and degrades the result (0.810 with vs 0.849 without). Set > 0
            to run the ablation.
        target_side_px: Target square side in far-field (padded) camera pixels.
        seed: RNG seed.
        golden_coeffs: "Golden" Zernike coefficients (raw radians; coefficient
            is the RMS phase of the mode, so ``a/2*pi`` is the RMS in waves)
            that produce the reference "actual" far-field. If None, a fixed
            default set is used (defocus + astig + spherical at 0.15/0.10/0.10
            waves RMS => Strehl ~ 0.19, a clearly aberrated focus).
        zernike_lr: Learning rate for Zernike calibration. Keep small
            (1e-3..1e-2); larger values diverge to non-finite coefficients.
        slm_lr: Learning rate for SLM phase shaping.
        calib_iters: Adam steps per calibration pass.
        shaping_iters: Adam steps per shaping pass.
        max_outer_iters: Max A↔B outer iterations.
        early_stop_patience: Early-stop patience (non-improving passes).
        early_stop_min_delta: Min score improvement to reset patience.
        gs_warm_start: Warm-start Stage B from a Gerchberg-Saxton phase. Without
            it the free-form Adam pass starts from zeros and only ties GS; with
            it the differentiable refinement of the GS solution beats GS.
        gs_iters: GS iterations used to build the warm start.
        far_field_padding: Zero-padding factor for the Fraunhofer FFT (far-field
            grid = n_grid * far_field_padding). A same-size FFT undersamples the
            focal plane at ~1.1 px per waist radius and aliases into a lattice.
    """

    n_grid: int = 64
    n_zernike: int = 0
    target_side_px: int = 43
    seed: int = 0
    golden_coeffs: dict[tuple[int, int], float] | None = None
    zernike_lr: float = 0.005
    slm_lr: float = 0.02
    calib_iters: int = 50
    shaping_iters: int = 100
    max_outer_iters: int = 5
    early_stop_patience: int = 2
    early_stop_min_delta: float = 0.005
    gs_warm_start: bool = True
    gs_iters: int = 200
    far_field_padding: int = 8

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
            far_field_padding=self.far_field_padding,
        )


# ---------------------------------------------------------------------------
# Golden far-field builder
# ---------------------------------------------------------------------------
def _build_golden_far_field(
    cfg: IterativeZernikePibConfig,
) -> np.ndarray:
    """Build the reference "actual" far-field from golden Zernike coefficients.

    The golden phase is a Zernike phase pattern (raw radians) propagated through
    the 2f bench forward model with a **flat** SLM phase. This simulates the
    "actual" distorted far-field that the Zernike calibration must match; the
    flat SLM phase keeps the calibration well-posed (adding free-form SLM noise
    would make the reference unmatchable by any Zernike set).

    Returns:
        Normalized far-field intensity (sum=1) on the far-field (padded) grid.
    """
    from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

    n = cfg.n_grid
    bench_cfg = ShapingBenchConfig(
        n_grid=n,
        target_side_px=cfg.target_side_px,
        far_field_padding=cfg.far_field_padding,
        seed=cfg.seed,
    )

    if cfg.golden_coeffs is None:
        # Defocus + 45° astig + spherical at 0.15/0.10/0.10 waves RMS
        # (coefficient = RMS phase in rad, so waves = a / 2*pi).
        golden = {(2, 0): 0.94, (2, -2): 0.63, (4, 0): 0.63}
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

    ff = forward_intensity(zernike_phase, bench_cfg)
    ff = ff / (ff.sum() + 1e-12)
    return ff


def _build_target(cfg: IterativeZernikePibConfig) -> np.ndarray:
    """Build the square target pattern (sum=1).

    If gs_iters > 0, runs a quick GS to get a "GS-ideal" target; otherwise
    uses the analytical square from make_target.
    """
    bench_cfg = ShapingBenchConfig(
        n_grid=cfg.n_grid,
        target_side_px=cfg.target_side_px,
        far_field_padding=cfg.far_field_padding,
        seed=cfg.seed,
    )
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
    1. Builds the reference "actual" far-field (from golden Zernike).
    2. Builds the square target.
    3. Optionally computes a GS warm start for Stage B.
    4. Instantiates the algorithm-layer optimizer and runs the A↔B loop.
    5. Records history and returns a result dict.

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

    bench_cfg = ShapingBenchConfig(
        n_grid=config.n_grid,
        target_side_px=config.target_side_px,
        far_field_padding=config.far_field_padding,
        seed=config.seed,
    )

    warm_start: np.ndarray | None = None
    if config.gs_warm_start:
        warm_start = gs_shape(bench_cfg, n_iters=config.gs_iters, seed=config.seed).phase

    # Algorithm-layer config and optimizer
    alg_cfg = config.to_alg_config()
    optimizer = IterativeZernikeShapingOptimizer(alg_cfg)

    # Run
    result = optimizer.run(
        actual_far_field=actual_ff, initial_slm_phase=warm_start
    )

    ff = result.far_field
    ff_norm = ff / (ff.sum() + 1e-12)
    metrics = compute_bench_metrics(
        ff_norm,
        target,
        zero_order_margin_px=bench_cfg.zero_order_margin_px,
    )
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
