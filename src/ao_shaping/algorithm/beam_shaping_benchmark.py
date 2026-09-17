"""Simulation benchmark for SLM far-field beam-shaping algorithms (Unit B).

Compares the three far-field shaping strategies on a *pure simulation*
bundle — no SLM / CCD / DM hardware required:

* ``"gs"`` — :func:`~ao_shaping.algorithm.gerchberg_saxton.gerchberg_saxton`
* ``"backprop"`` — :func:`~ao_shaping.algorithm.differentiable_shaping.train_beam_shaping`
* ``"spgd-sim"`` — a compact self-contained SPGD loop over the same
  Fraunhofer (FFT) forward model used by the other two, so all three
  optimisers see an identical propagation physics.

Every algorithm produces an SLM phase map which is then turned back into
a simulated far-field intensity with the **same** FFT propagator, so the
benchmark is a fair head-to-head: the only difference between the columns
is the optimisation strategy, never the forward model.

Exposure/brightness invariant metrics follow the project rules — all
intensity comparisons are normalised so absolute scale does not matter.

Public API:
    - :func:`run_benchmark` (one algorithm × one target shape → result dict)
    - :func:`run_benchmark_suite` (grid over algorithms × shapes →
      ``(list[dict], DataFrame)``)
    - :func:`measure_shaped_area` / :func:`check_area_requirement`
      (fixed-shape area gate)
    - GIF + metrics CSV/MD writers (PIL / stdlib csv)

This module targets Python 3.12+, is hardware-free and fully offline;
tests live in ``tests/ao_shaping/algorithm/``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image

# Reuse the canonical algorithm + metrics implementations so the benchmark
# measures exactly what the runners ship, not a re-implementation.
from ao_shaping.utils.beam_metrics import compute_shaping_metrics
from ao_shaping.utils.targets import create_target_shape
from ao_shaping.optimizer import train_beam_shaping
from ao_shaping.algorithm.gerchberg_saxton import gerchberg_saxton

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
DEFAULT_GRID: tuple[int, int] = (128, 128)
DEFAULT_CELL_SPACING: float = 8e-6
DEFAULT_DISTANCE: float = 0.1
DEFAULT_WAVELENGTH: float = 1064e-9
DEFAULT_TARGET_AREA: int = 256
DEFAULT_ITERATIONS: int = 100
DEFAULT_MAX_FRAMES: int = 40

_SHAPES: set[str] = {"square", "circle", "gaussian"}
_ALGORITHMS: set[str] = {"gs", "backprop", "spgd-sim"}

# 公开别名 (shaping_runner CLI 引用): 已排序的元组, 保证确定性输出顺序.
SUITE_SHAPES: tuple[str, ...] = tuple(sorted(_SHAPES))
SUITE_ALGORITHMS: tuple[str, ...] = tuple(sorted(_ALGORITHMS))


# ---------------------------------------------------------------------------
# Forward model (shared by all algorithms)
# ---------------------------------------------------------------------------
def _propagate_far_field(
    phase: np.ndarray,
    *,
    cell_spacing: float = DEFAULT_CELL_SPACING,
    distance: float = DEFAULT_DISTANCE,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> np.ndarray:
    """Fraunhofer far-field intensity of ``exp(i*phase)`` (FFT focal model).

    The SLM plane field ``exp(j*phi)`` (uniform illumination, plateau flat
    phase = 0-order at center) is propagated to the far field with a single
    FFT — the same focal-plane model used by ``gerchberg_saxton(...,
    propagation="fft")``. Returns the normalised *intensity*
    ``|FFT(exp(j*phi))|^2`` so all algorithms are compared on the same
    physics.

    Args:
        phase: 2D phase map in radians.
        cell_spacing: Pixel pitch (m). Unused by FFT but kept for symmetry.
        distance: Propagation distance (m). Unused by FFT but kept for
            symmetry with the ASM path.
        wavelength: Wavelength (m). Unused by FFT but kept for symmetry.

    Returns:
        Float64 2D far-field intensity, normalised to sum == 1.
    """
    field = np.exp(1j * np.asarray(phase, dtype=np.float64))
    ff = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    total = float(ff.sum())
    if total > 0:
        ff = ff / total
    return ff


# ---------------------------------------------------------------------------
# Area helpers (fixed-shape gate)
# ---------------------------------------------------------------------------
def measure_shaped_area(intensity: np.ndarray, threshold_ratio: float = 0.5) -> int:
    """Count pixels whose intensity is at least ``threshold_ratio × peak``.

    This is the project-standard "shaped area" definition used by the
    square/diff runners: the bright region is the set of pixels above half
    of the frame's *peak* intensity (threshold is relative, never absolute,
    which keeps the metric exposure-invariant).

    Args:
        intensity: 2D intensity map.
        threshold_ratio: Fraction of the peak intensity defining "bright".
            Must be in ``(0, 1]``.

    Returns:
        Number of bright pixels (``0`` for an empty/zero frame).

    Raises:
        ValueError: If ``threshold_ratio`` is not in ``(0, 1]``.
    """
    if not 0.0 < float(threshold_ratio) <= 1.0:
        raise ValueError(f"threshold_ratio must be in (0, 1], got {threshold_ratio}")
    arr = np.asarray(intensity, dtype=np.float64)
    peak = float(np.max(arr)) if arr.size else 0.0
    if peak <= 0:
        return 0
    return int(np.count_nonzero(arr >= threshold_ratio * peak))


def check_area_requirement(
    measured: int,
    requested: int,
    tolerance: float = 0.20,
) -> dict[str, Any]:
    """Check whether a measured shaped area meets a requested area.

    The requirement is satisfied when ``measured >= (1 - tolerance) *
    requested`` (we tolerate undershoot but never consider an oversized
    pattern a failure — the gate is "did we fill the target box").

    Args:
        measured: Measured bright-pixel count.
        requested: Requested target-box pixel count.
        tolerance: Allowed relative undershoot ``(0, 1)``.

    Returns:
        Dict with ``"met"`` (bool), ``"measured_area"``, ``"requested_area"``,
        ``"tolerance"``, ``"fill_ratio"`` (``measured / requested``) and
        ``"shortfall"`` (``max(0, requested - measured)``).

    Raises:
        ValueError: If either area is negative or ``tolerance`` is out of
            ``(0, 1)``.
    """
    if measured < 0 or requested < 0:
        raise ValueError(f"Areas must be non-negative, got {measured}, {requested}")
    if not 0.0 < float(tolerance) < 1.0:
        raise ValueError(f"tolerance must be in (0, 1), got {tolerance}")
    fill_ratio = (float(measured) / float(requested)) if requested > 0 else 1.0
    required_min = (1.0 - float(tolerance)) * float(requested)
    return {
        "met": bool(float(measured) >= required_min),
        "measured_area": int(measured),
        "requested_area": int(requested),
        "tolerance": float(tolerance),
        "fill_ratio": float(fill_ratio),
        "shortfall": int(max(0, requested - measured)),
    }


# ---------------------------------------------------------------------------
# Target generation
# ---------------------------------------------------------------------------
def create_benchmark_target(
    shape: str,
    grid_size: tuple[int, int],
    *,
    target_area: int = DEFAULT_TARGET_AREA,
    aspect_ratio: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a normalised target shape for a benchmark run.

    Wraps :func:`~ao_shaping.utils.targets.create_target_shape`
    so the benchmark uses the exact same target factory as the runners.

    Args:
        shape: ``"square"``, ``"circle"`` or ``"gaussian"``.
        grid_size: ``(height, width)`` output grid.
        target_area: Requested bright pixel count. For ``"square"`` this
            fixes the side via ``side = round(sqrt(area))``; for ``"circle"``
            it sizes the radius so the **filled** circle area approximates the
            request; for ``"gaussian"`` it is ignored (sigma from default).
        aspect_ratio: Width:height for ``"square"``; ``>1`` makes a
            rectangle (long axis horizontal). Ignored otherwise.

    Returns:
        ``(target_intensity, info)`` where ``target_intensity`` is the
        normalised 2D intensity in ``[0, 1]`` and ``info`` contains
        ``"requested_area"``, ``"shape"``, ``"grid_size"`` and ``"side_px"``
        (square side length in grid pixels).

    Raises:
        ValueError: If ``shape`` is not supported.
    """
    if shape not in _SHAPES:
        raise ValueError(f"Unsupported shape {shape!r}; choose from {sorted(_SHAPES)}")

    if shape == "square":
        side = max(1, int(round(float(target_area) ** 0.5)))
        # For aspect_ratio > 1 the long side is horizontal: side × ratio.
        long_side = max(side, int(round(side * float(aspect_ratio))))
        target = create_target_shape(
            "rectangle" if aspect_ratio > 1.0 else "square",
            grid_size,
            side=long_side if aspect_ratio > 1.0 else side,
            aspect_ratio=float(aspect_ratio),
        )
        info_side = side
    elif shape == "circle":
        target = create_target_shape("circle", grid_size, radius_ratio=0.3)
        info_side = 0
    else:  # gaussian
        target = create_target_shape("gaussian", grid_size, radius_ratio=0.3)
        info_side = 0

    total = float(target.sum())
    if total > 0:
        target = target.astype(np.float64) / total

    requested = measure_shaped_area(target, threshold_ratio=0.5)
    info = {
        "shape": shape,
        "grid_size": grid_size,
        "requested_area": requested,
        "side_px": info_side,
    }
    return target, info


# ---------------------------------------------------------------------------
# Per-algorithm execution
# ---------------------------------------------------------------------------
def _run_gerchberg_saxton(
    target_intensity: np.ndarray,
    grid_size: tuple[int, int],
    iterations: int,
    seed: int,
) -> np.ndarray:
    """GS phase retrieval; return the phase map (radians)."""
    target_amp = np.sqrt(np.maximum(target_intensity, 0.0))
    result = gerchberg_saxton(
        source_amplitude=np.ones(grid_size, dtype=np.float64),
        target_amplitude=target_amp,
        iterations=int(iterations),
        propagation="fft",
    )
    logger.debug(
        f"GS done: {result.iterations} iters, converged={result.converged}, "
        f"final error={result.error_history[-1]:.4f}"
    )
    return np.asarray(result.phase, dtype=np.float64)


def _run_backprop(
    target_intensity: np.ndarray,
    grid_size: tuple[int, int],
    iterations: int,
    seed: int,
    device: str | None,
) -> np.ndarray:
    """Differentiable gradient-descent shaping; return the phase map."""
    result = train_beam_shaping(
        target=target_intensity,
        grid_size=grid_size,
        propagation="fft",
        optimizer="adam",
        iterations=int(iterations),
        lr=3e-2,
        w_uniformity=0.4,
        w_efficiency=0.6,
        device=device,
        seed=seed,
    )
    logger.debug(
        f"Backprop done: {getattr(result, 'iterations', '?')} iters, "
        f"converged={getattr(result, 'converged', '?')}"
    )
    return np.asarray(result.phase, dtype=np.float64)


def _run_spgd_sim(
    target_intensity: np.ndarray,
    grid_size: tuple[int, int],
    iterations: int,
    seed: int,
) -> np.ndarray:
    """Self-contained SPGD loop over the same FFT forward model.

    SPGD (Stochastic Parallel Gradient Descent) optimises the SLM phase map
    directly against the target intensity using the same
    :func:`_propagate_far_field` model as GS/backprop. The cost combines
    uniformity (CV inside the bright mask) and encircled energy, mirroring
    the project's shaping objective, so the comparison is apples-to-apples.
    """
    rng = np.random.default_rng(seed)
    phase = rng.normal(0.0, 0.05, size=grid_size).astype(np.float64)
    mask = target_intensity > 0.5 * float(np.max(target_intensity))

    best_cost = np.inf
    best_phase = phase.copy()
    for it in range(int(iterations)):
        # Random perturbation (same statistics each iteration).
        delta = rng.normal(0.0, 0.08, size=grid_size)
        plus = _propagate_far_field(phase + delta)
        minus = _propagate_far_field(phase - delta)
        cost_plus, cost_minus = ( _cost(plus, mask), _cost(minus, mask))
        grad = (cost_plus - cost_minus) / (2.0 * 0.08)
        lr = 0.10 / (1.0 + 0.01 * it)
        phase = phase - lr * grad * delta
        c = _cost(_propagate_far_field(phase), mask)
        if c < best_cost:
            best_cost, best_phase = c, phase.copy()

    logger.debug(f"SPGD-sim done: {iterations} iters, best cost={best_cost:.4f}")
    return best_phase


def _cost(intensity: np.ndarray, mask: np.ndarray) -> float:
    """Shaping cost from uniformity CV + encircled-energy shortfall."""
    metrics = compute_shaping_metrics(intensity, mask)
    cv = float(metrics.get("uniformity_cv", 0.0))
    ee = float(metrics.get("encircled_energy", 0.0))
    return cv + (1.0 - ee)


# ---------------------------------------------------------------------------
# Public benchmark entry points
# ---------------------------------------------------------------------------
def run_benchmark(
    algorithm: str,
    shape: str,
    grid_size: tuple[int, int] = DEFAULT_GRID,
    target_area: int = DEFAULT_TARGET_AREA,
    aspect_ratio: float = 1.0,
    iterations: int | None = None,
    seed: int = 42,
    max_frames: int = DEFAULT_MAX_FRAMES,
    device: str | None = None,
    output_dir: str | Path | None = None,
    make_gif: bool = True,
) -> dict[str, Any]:
    """Run one shaping algorithm on one simulated target shape.

    Args:
        algorithm: ``"gs"``, ``"backprop"`` or ``"spgd-sim"``.
        shape: ``"square"``, ``"circle"`` or ``"gaussian"``.
        grid_size: ``(height, width)`` grid.
        target_area: Requested target-box pixel count (square side derives
            from its square root).
        aspect_ratio: Width:height for a square target (``>1`` → rectangle).
        iterations: Optimisation iterations; default 100.
        seed: Random seed (reproducible runs).
        max_frames: Length cap for the recorded evolution (GIF frame budget).
        device: Backprop compute device (``"cuda"``/``"cpu"``/None=auto).
        output_dir: If given, writes metrics CSV/MD + GIF here.
        make_gif: Render an evolution GIF when ``output_dir`` is set.

    Returns:
        Dict with algorithm identifier, target/requested area, simulated
        intensity, shaped-area measurement, area-requirement check, shaping
        metrics (``uniformity_cv``, ``encircled_energy``,
        ``uniformity_cv``), and timing.
    """
    if algorithm not in _ALGORITHMS:
        raise ValueError(f"Unsupported algorithm {algorithm!r}; choose {sorted(_ALGORITHMS)}")
    if shape not in _SHAPES:
        raise ValueError(f"Unsupported shape {shape!r}; choose {sorted(_SHAPES)}")

    import time

    iterations = int(iterations) if iterations is not None else DEFAULT_ITERATIONS

    # Normalise a scalar grid_size (``32``) to a 2D grid (``(32, 32)``) at the
    # single authoritative entry point. All downstream consumers
    # (``create_benchmark_target``, ``gerchberg_saxton``/``backprop``/
    # ``spgd-sim`` phase runners) require 2D arrays; a scalar would otherwise
    # collapse ``np.ones(grid_size)`` to a 1D source amplitude and raise
    # ``ValueError: Input amplitudes must be 2D arrays``.
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    target_intensity, info = create_benchmark_target(
        shape, grid_size, target_area=target_area, aspect_ratio=aspect_ratio
    )
    requested = int(info["requested_area"])

    t0 = time.perf_counter()
    if algorithm == "gs":
        phase = _run_gerchberg_saxton(target_intensity, grid_size, iterations, seed)
    elif algorithm == "backprop":
        phase = _run_backprop(target_intensity, grid_size, iterations, seed, device)
    else:
        phase = _run_spgd_sim(target_intensity, grid_size, iterations, seed)
    elapsed = time.perf_counter() - t0

    simulated = _propagate_far_field(phase)
    measured_area = measure_shaped_area(simulated)
    area_check = check_area_requirement(measured_area, requested)

    # Normalise target for the (symmetric) metric call.
    target_norm = target_intensity / float(target_intensity.sum()) if target_intensity.sum() > 0 else target_intensity
    mask = target_norm > 0.5 * float(np.max(target_norm))
    metrics = compute_shaping_metrics(simulated, mask)

    result: dict[str, Any] = {
        "algorithm": algorithm,
        "shape": shape,
        "grid_size": grid_size,
        "target_area": target_area,
        "aspect_ratio": float(aspect_ratio),
        "iterations": iterations,
        "seed": seed,
        "requested_area": requested,
        "measured_area": measured_area,
        "area_met": area_check["met"],
        "fill_ratio": area_check["fill_ratio"],
        "uniformity_cv": float(metrics.get("uniformity_cv", 0.0)),
        "encircled_energy": float(metrics.get("encircled_energy", 0.0)),
        "intensity_peak": float(metrics.get("peak", 0.0)),
        "elapsed_s": float(elapsed),
        "simulated": simulated,
        "target": target_norm,
        "phase": phase,
    }

    if output_dir is not None and make_gif:
        _write_artifacts(result, output_dir)

    return result


def run_benchmark_suite(
    algorithms: list[str] | tuple[str, ...] | None = None,
    shapes: list[str] | tuple[str, ...] | None = None,
    *,
    grid_size: tuple[int, int] = DEFAULT_GRID,
    target_area: int = DEFAULT_TARGET_AREA,
    aspect_ratio: float = 1.0,
    iterations: int | None = None,
    seed: int = 42,
    max_frames: int = DEFAULT_MAX_FRAMES,
    device: str | None = None,
    output_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Run all algorithms × all selected shapes (exhaustive grid).

    Args:
        algorithms: Subset of ``{"gs","backprop","spgd-sim"}``;
            all three when *None*.
        shapes: Subset of ``{"square","circle","gaussian"}``;
            all three when *None*.
        grid_size: Grid dimensions.
        target_area: Requested target-box area (pixels).
        aspect_ratio: Square aspect ratio.
        iterations: Optimisation iterations.
        seed: Random seed.
        max_frames: GIF frame budget (per cell).
        device: Backprop device.
        output_dir: If set, writes a combined metrics CSV/MD table.

    Returns:
        ``(rows, dataframe)`` where each row is the scalar
        :func:`run_benchmark` result (phase/simulated arrays stripped) and
        ``dataframe`` is the tabular view with one row per cell.
    """
    algos = list(algorithms or sorted(_ALGORITHMS))
    shps = list(shapes or sorted(_SHAPES))

    rows: list[dict[str, Any]] = []
    for alg in algos:
        for shp in shps:
            logger.info(f"Benchmark: algorithm={alg} shape={shp}")
            result = run_benchmark(
                algorithm=alg,
                shape=shp,
                grid_size=grid_size,
                target_area=target_area,
                aspect_ratio=aspect_ratio,
                iterations=iterations,
                seed=seed,
                max_frames=max_frames,
                device=device,
                output_dir=output_dir,
                make_gif=False,
            )
            rows.append(result)

    df = _to_dataframe(rows)

    if output_dir is not None:
        _write_table(df, output_dir)

    return rows, df


# ---------------------------------------------------------------------------
# Serialization helpers (DF / CSV / MD / GIF)
# ---------------------------------------------------------------------------
_HPRINT_KEYS: tuple[str, ...] = (
    "algorithm",
    "shape",
    "requested_area",
    "measured_area",
    "area_met",
    "fill_ratio",
    "uniformity_cv",
    "encircled_energy",
    "elapsed_s",
)


def _to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Project rows on the scalar (non-array) fields into a DataFrame."""
    scalar_rows = []
    for row in rows:
        scalar_rows.append({k: row[k] for k in _HPRINT_KEYS if k in row})
    df = pd.DataFrame(scalar_rows)
    if not df.empty:
        df = df.sort_values(["algorithm", "shape"]).reset_index(drop=True)
    return df


def _write_table(df: pd.DataFrame, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "beam_shaping_benchmark_metrics.csv", index=False)

    lines = [
        "# Beam-Shaping Benchmark (simulation)",
        "",
        f"- Grid: {df['shape'].count() if not df.empty else 0} cells",
        "",
        "| algorithm | shape | requested | measured | area_met | fill_ratio | "
        "uniformity_cv | encircled_energy | elapsed_s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['algorithm']} | {row['shape']} | {row['requested_area']} "
            f"| {row['measured_area']} | {row['area_met']} "
            f"| {row['fill_ratio']:.3f} | {row['uniformity_cv']:.3f} "
            f"| {row['encircled_energy']:.3f} | {row['elapsed_s']:.3f} |"
        )
    lines.append("")
    (out / "beam_shaping_benchmark_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Benchmark tables written to {out}")


def _write_artifacts(result: dict[str, Any], output_dir: str | Path) -> None:
    """Write per-run GIF (PIL) + metrics CSV/MD for a single result."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CSV row
    header = list(_HPRINT_KEYS)
    with (out / f"{result['algorithm']}_{result['shape']}_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerow([result.get(k, "") for k in header])

    # MD
    lines = [
        f"# Benchmark: {result['algorithm']} × {result['shape']}",
        "",
        f"- requested_area: {result['requested_area']}",
        f"- measured_area: {result['measured_area']}  (met: {result['area_met']}, "
        f"fill_ratio: {result['fill_ratio']:.3f})",
        f"- uniformity_cv: {result['uniformity_cv']:.4f}",
        f"- encircled_energy: {result['encircled_energy']:.4f}",
        f"- elapsed_s: {result['elapsed_s']:.3f}",
        "",
    ]
    (out / f"{result['algorithm']}_{result['shape']}_metrics.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # GIF: target → simulated (fixed frame budget)
    target = np.asarray(result["target"])
    simulated = np.asarray(result["simulated"])
    if target.ndim == 2 and simulated.ndim == 2 and target.shape == simulated.shape:
        frames = _build_gif_frames(target, simulated, max_frames=DEFAULT_MAX_FRAMES)
        gif_path = out / f"{result['algorithm']}_{result['shape']}_evolution.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=max(50, int(1000 * min(2.0, max(0.05, result["elapsed_s"])))),
            loop=0,
        )
        logger.debug(f"Wrote GIF {gif_path}")


def _build_gif_frames(
    target: np.ndarray,
    simulated: np.ndarray,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[Image.Image]:
    """Render a small PIL frame sequence: target → simulated stacked.

    Uses a perceptually-scaled grayscale palette so intensity dynamics are
    visible in a low-bit GIF. Returns at least one frame.

    Args:
        target: Normalised target intensity ``(H, W)``.
        simulated: Normalised simulated intensity ``(H, W)``.
        max_frames: Maximum pragmatic frame count (GIF doesn't benefit
            from >40).

    Returns:
        List of PIL ``Image`` (mode ``"P"``, 8-bit palette).
    """
    n_frames = max(1, min(max_frames, 24))
    stack = np.stack([target, simulated], axis=-1)  # (H, W, 2)
    stack = stack / max(float(stack.max()), 1e-12)
    frames: list[Image.Image] = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        frame = stack[..., 0] * (1 - t) + stack[..., 1] * t
        gray = (255 * frame).astype(np.uint8)
        frames.append(Image.fromarray(gray, mode="L").convert("P"))
    return frames


__all__ = [
    "run_benchmark",
    "run_benchmark_suite",
    "measure_shaped_area",
    "check_area_requirement",
    "create_benchmark_target",
]
