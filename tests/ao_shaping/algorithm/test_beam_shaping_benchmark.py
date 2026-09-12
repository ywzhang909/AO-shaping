"""Tests for the pure-simulation beam-shaping benchmark core (Unit B).

Covers :mod:`ao_shaping.algorithm.beam_shaping_benchmark` — the hardware-free
benchmark that compares GS / backprop / spgd-sim across square / circle /
gaussian targets and reports shaped-area compliance, uniformity CV, encircled
energy, GIF(PIL) output and metrics CSV/MD.

These tests are simulation-only (no SLM / CCD / DM needed) and kept small so
the whole suite runs in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ao_shaping.algorithm.beam_shaping_benchmark import (
    check_area_requirement,
    create_benchmark_target,
    measure_shaped_area,
    run_benchmark,
    run_benchmark_suite,
)

# Small grid + few iterations to keep the module fast.
GRID = (64, 64)
ITER = 5

_ALL_ALGOS = {"gs", "backprop", "spgd-sim"}
_ALL_SHAPES = {"square", "circle", "gaussian"}


# ===========================================================================
# Target generation
# ===========================================================================
class TestCreateBenchmarkTarget:
    """Every shape must yield a normalised target box with the right area."""

    @pytest.mark.parametrize("shape", ["square", "circle", "gaussian"])
    def test_shapes_return_target_and_info(self, shape: str):
        target, info = create_benchmark_target(
            shape, GRID, target_area=256, aspect_ratio=1.0
        )
        assert target.shape == GRID
        assert set(info) >= {"grid_size", "requested_area"}
        assert 0.0 <= float(target.min()) <= float(target.max()) <= 1.0

    def test_normalised(self):
        target, _ = create_benchmark_target("square", GRID, target_area=256, aspect_ratio=1.0)
        assert np.isclose(target.sum(), 1.0, atol=1e-3)

    def test_requested_area_reported(self):
        # Requested box is realised as a square of side ~sqrt(area).
        _, info = create_benchmark_target("square", GRID, target_area=100, aspect_ratio=1.0)
        side = round(round(info["requested_area"]) ** 0.5)
        assert side == 10


# ===========================================================================
# Shaped-area measurement + requirement check
# ===========================================================================
class TestMeasureShapedArea:
    def test_counts_bright_pixels(self):
        img = np.zeros(GRID, dtype=np.float64)
        img[8:16, 24:32] = 1.0  # 8x8 = 64 bright pixels at full intensity
        assert measure_shaped_area(img, threshold_ratio=0.5) == 64

    def test_half_threshold(self):
        # Explicit 2D blocks so the expected pixel counts are unambiguous.
        img = np.zeros(GRID, dtype=np.float64)
        img[0:2, 0:2] = 1.0   # 2x2 = 4 px at peak
        img[3:5, 0:2] = 0.5   # 2x2 = 4 px at threshold -> counted
        img[6:8, 0:2] = 0.49  # 2x2 = 4 px below threshold -> excluded
        assert measure_shaped_area(img, threshold_ratio=0.5) == 4 + 4

    def test_empty_frame_counts_zero(self):
        assert measure_shaped_area(np.zeros(GRID, dtype=np.float64)) == 0

    def test_rejects_bad_threshold(self):
        with pytest.raises(ValueError):
            measure_shaped_area(np.ones(GRID, dtype=np.float64), threshold_ratio=0.0)
        with pytest.raises(ValueError):
            measure_shaped_area(np.ones(GRID, dtype=np.float64), threshold_ratio=1.5)


class TestCheckAreaRequirement:
    def test_met_when_full(self):
        res = check_area_requirement(measured=100, requested=100, tolerance=0.20)
        assert res["met"] is True
        assert res["fill_ratio"] == 1.0
        assert res["shortfall"] == 0

    def test_met_within_tolerance(self):
        # 90% of 100 with tolerance 0.20 -> allowed (>= 80).
        res = check_area_requirement(measured=90, requested=100, tolerance=0.20)
        assert res["met"] is True

    def test_not_met_when_undershoot(self):
        res = check_area_requirement(measured=50, requested=100, tolerance=0.20)
        assert res["met"] is False
        assert res["shortfall"] == 50

    def test_exact_boundary_counts_as_met(self):
        res = check_area_requirement(measured=80, requested=100, tolerance=0.20)
        assert res["met"] is True

    def test_rejects_bad_tolerance(self):
        with pytest.raises(ValueError):
            check_area_requirement(100, 100, tolerance=1.0)
        with pytest.raises(ValueError):
            check_area_requirement(100, 100, tolerance=0.0)

    def test_rejects_negative_areas(self):
        with pytest.raises(ValueError):
            check_area_requirement(measured=-1, requested=100)
        with pytest.raises(ValueError):
            check_area_requirement(measured=100, requested=-1)


# ===========================================================================
# Single algorithm run
# ===========================================================================
class TestRunBenchmark:
    def _run(self, algorithm: str, shape: str, **kw) -> dict:
        defaults = dict(
            grid_size=GRID,
            target_area=256,
            aspect_ratio=1.0,
            iterations=ITER,
            seed=0,
            max_frames=2,
        )
        defaults.update(kw)
        return run_benchmark(algorithm, shape, **defaults)

    @pytest.mark.parametrize("algorithm", sorted(_ALL_ALGOS))
    @pytest.mark.parametrize("shape", ["square", "circle", "gaussian"])
    def test_runs_for_all_algo_shape(self, algorithm: str, shape: str):
        result = self._run(algorithm, shape)
        assert result["algorithm"] == algorithm
        assert result["shape"] == shape

    def test_result_has_all_metric_keys(self):
        result = self._run("gs", "square")
        for key in (
            "algorithm",
            "shape",
            "requested_area",
            "measured_area",
            "area_met",
            "fill_ratio",
            "uniformity_cv",
            "encircled_energy",
            "elapsed_s",
        ):
            assert key in result
        assert isinstance(result["area_met"], bool)
        assert result["elapsed_s"] > 0.0

    def test_simulated_and_phase_present(self):
        result = self._run("gs", "square")
        assert result["simulated"].shape == GRID
        assert result["phase"].shape == GRID

    def test_rejects_unknown_algorithm(self):
        with pytest.raises(ValueError):
            self._run("nope", "square")

    def test_rejects_unknown_shape(self):
        with pytest.raises(ValueError):
            self._run("gs", "triangle")


# ===========================================================================
# Full suite (grid over algorithms x shapes)
# ===========================================================================
class TestRunBenchmarkSuite:
    def test_all_cells(self):
        rows, df = run_benchmark_suite(
            grid_size=GRID,
            target_area=256,
            aspect_ratio=1.0,
            iterations=ITER,
            seed=0,
            max_frames=2,
        )
        assert len(rows) == len(_ALL_ALGOS) * len(_ALL_SHAPES)
        assert len(df) == len(rows)

    def test_dataframe_columns(self):
        _, df = run_benchmark_suite(
            grid_size=GRID,
            target_area=256,
            aspect_ratio=1.0,
            iterations=ITER,
            seed=0,
            max_frames=2,
        )
        assert set(df.columns) == {
            "algorithm",
            "shape",
            "requested_area",
            "measured_area",
            "area_met",
            "fill_ratio",
            "uniformity_cv",
            "encircled_energy",
            "elapsed_s",
        }

    def test_subset_algorithms_and_shapes(self):
        rows, df = run_benchmark_suite(
            algorithms=["gs"],
            shapes=["square"],
            grid_size=GRID,
            target_area=256,
            aspect_ratio=1.0,
            iterations=ITER,
            seed=0,
            max_frames=2,
        )
        assert len(rows) == 1
        assert rows[0]["algorithm"] == "gs"
        assert rows[0]["shape"] == "square"
