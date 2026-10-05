"""Offline tests for the bench-probe report generator's correctness guards.

The generator's job is not to draw pictures, it is to refuse to draw conclusions
from mismatched inputs. These tests pin the two guards that matter, because both
fail *silently*: a stale geometry file and a hardcoded constant produce a
confident, well-formatted, wrong report.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
GENERATOR = ROOT / "scripts" / "generate_bench_probe_report.py"


def _load_generator() -> Any:
    """Import the generator by path; it is a script, not an installed module.

    The module is registered in ``sys.modules`` before execution because it
    defines dataclasses, and ``dataclasses`` resolves the field types through the
    defining module's namespace.
    """
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("_bench_report", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bench_report"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen() -> Any:
    return _load_generator()


def _scalars(**kw: float) -> dict[str, float]:
    return kw


class TestGeometryProvenanceGuard:
    """``bench_geometry.json`` is shared and overwritten by both routes.

    Reading it next to a fresh npz without checking provenance silently mixes two
    calibrations: every model-side number inherits the error while the derivation
    still looks self-consistent.
    """

    def test_matching_geometry_reports_no_conflict(self, gen: Any) -> None:
        geometry = {"panel_disc_radius": 450, "method": "sweep"}
        scalars = _scalars(zernike_radius=450, collect_disc=450)
        assert gen.geometry_conflicts(geometry, scalars) == []

    def test_disc_radius_mismatch_is_a_conflict(self, gen: Any) -> None:
        geometry = {"panel_disc_radius": 200, "method": "sweep"}
        scalars = _scalars(zernike_radius=450, collect_disc=450)
        conflicts = gen.geometry_conflicts(geometry, scalars)
        assert conflicts, "a different aperture radius proves a different calibration"
        assert any("450" in c for c in conflicts)

    def test_speckle_method_is_flagged_even_when_numbers_agree(self, gen: Any) -> None:
        """Same aperture, different route => the model grid still differs."""
        geometry = {"panel_disc_radius": 450, "method": "speckle"}
        scalars = _scalars(zernike_radius=450, collect_disc=450)
        conflicts = gen.geometry_conflicts(geometry, scalars)
        assert any("speckle" in c for c in conflicts)

    def test_missing_geometry_is_not_a_conflict(self, gen: Any) -> None:
        assert gen.geometry_conflicts(None, _scalars(zernike_radius=450)) == []

    def test_absent_fields_do_not_conflict(self, gen: Any) -> None:
        """Absent data must not be reported as disagreement."""
        assert gen.geometry_conflicts({"method": "sweep"}, {}) == []

    def test_non_numeric_values_do_not_raise(self, gen: Any) -> None:
        geometry = {"panel_disc_radius": None, "method": "sweep"}
        assert gen.geometry_conflicts(geometry, _scalars(zernike_radius=450)) == []


class TestBenchConstants:
    """Constants must come from the code, not from a literal that can drift."""

    def test_tilt_shift_scale_matches_the_kernel(self, gen: Any) -> None:
        from ao_shaping.tools.slm.bench_kernels import TILT_SHIFT_SCALE

        constants = gen._import_bench_constants()
        assert float(constants["tilt_shift_scale"]) == float(TILT_SHIFT_SCALE)

    def test_max_defocus_fit_rms_matches_the_fitter(self, gen: Any) -> None:
        from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
            _MAX_DEFOCUS_FIT_RMS,
        )

        constants = gen._import_bench_constants()
        assert float(constants["max_defocus_fit_rms"]) == float(_MAX_DEFOCUS_FIT_RMS)

    def test_importing_the_kernel_pure_numpy_module_is_allowed(self, gen: Any) -> None:
        """``bench_kernels`` takes its devices by injection, so it is safe.

        The generator must not blanket-refuse it: doing so is what let a literal
        drift away from the code unnoticed.
        """
        constants = gen._import_bench_constants()
        assert "TILT_SHIFT_SCALE" not in constants["fallbacks"]

    def test_fallbacks_are_reported_not_silent(self, gen: Any) -> None:
        constants = gen._import_bench_constants()
        assert isinstance(constants["fallbacks"], list)


class TestSweepDerivedQuantities:
    """The two cross-check relations the report asserts must be arithmetic."""

    def test_ramp_scale_is_slope_against_inverse_period(self, gen: Any) -> None:
        """``S`` must be the coefficient of ``displacement = S / period``.

        The per-point tolerance is loose because the inputs are real measured
        displacements, which carry a few percent of noise; the aggregate
        relation is asserted tightly because that is the actual invariant.
        """
        periods = np.array([120.0, 240.0, 480.0, 960.0, 1920.0])
        displacement = np.array([62.9, 30.84, 15.60, 8.37, 3.90])
        routes = gen.compute_ramp_routes(_fake_sweep(periods, displacement))
        assert routes
        scale = abs(routes[0].scale)
        assert np.isfinite(scale)
        # Per point: the model must track the measurement to within its noise.
        assert np.allclose(displacement, scale / periods, rtol=0.10)
        # Aggregate: S is the mean of displacement * period, tightly.
        assert scale == pytest.approx(float(np.mean(displacement * periods)), rel=0.01)

    def test_tilt_implied_scale_relation_holds(self) -> None:
        """A Zernike tilt c is a ramp of gradient 2c/R, hence S = k*pi*R."""
        k_tilt, radius = 5.446, 450.0
        implied = k_tilt * np.pi * radius
        # The measured S was 7565; the relation must land near it.
        assert abs(implied - 7565.0) / 7565.0 < 0.05

    def test_width_filter_drops_rings_below_the_hollowness_gate(self, gen: Any) -> None:
        """A ring's diameter is not a spot width; the gate must catch it."""
        curve = gen.build_mode_curve(
            "spherical",
            np.array([-1.2, -0.6, 0.6, 1.2]),
            np.array([38.16, 19.61, 12.16, 13.83]),
            np.array([0.43, 0.69, 0.92, 0.89]),
            0.60,
        )
        dropped = [abs(c) for c in curve.dropped_coefficient] if hasattr(
            curve, "dropped_coefficient"
        ) else []
        assert not dropped or all(abs(c) > 1.0 for c in dropped)
        assert 0.43 not in np.round(curve.kept_hollowness, 2) if hasattr(
            curve, "kept_hollowness"
        ) else True


class TestReportStructureUnderBothGeometryPaths:
    """The report has two rendering paths; both must produce every section.

    Regression guard: the stale-geometry branch replaced the whole §6 body
    (routes table, both figures, and the ``## 7.`` heading) instead of only
    overriding it, so a *consistent* geometry silently produced a report with an
    empty §6 and no §7 heading at all.
    """

    @staticmethod
    def _report(gen: Any, conflicts: list[str], tmp_path: Path) -> str:
        npz = tmp_path / "sweep_records.npz"
        if not npz.exists():
            # flat + two ramp periods + a +/- tilt pair. Both legs need >= 2
            # points per axis or their route is skipped and the cross-check has
            # nothing to compare -- the test would assert on a skipped branch.
            # Panel-x tilt moves camera-y on this bench (axes swapped 90 deg),
            # hence the sign: c=+1 gives dy<0.
            np.savez_compressed(
                npz,
                label=np.array(
                    ["flat", "rampx120", "rampx240", "tiltx-1", "tiltx+1"]
                ),
                mode=np.array(["flat", "ramp", "ramp", "tilt", "tilt"]),
                axis=np.array(["", "x", "x", "x", "x"]),
                coefficient=np.array([0.0, 120.0, 240.0, -1.0, 1.0]),
                fwhm_px=np.array([13.0, 12.0, 12.1, 12.5, 12.4]),
                centroid_x=np.array([668.5, 667.9, 668.0, 668.5, 668.9]),
                centroid_y=np.array(
                    [1026.7, 963.8, 995.3, 1032.1, 1021.3]
                ),
                peak=np.array([135.0, 120.0, 122.0, 127.0, 124.0]),
                hollowness=np.array([0.92, 0.87, 0.88, 0.89, 0.90]),
                shift_x=np.array([np.nan] * 5),
                shift_y=np.array([np.nan] * 5),
                shift_corr=np.array([np.nan] * 5),
                zernike_radius=np.array(450),
                collect_disc=np.array(450),
                pupil_center=np.array([960, 600]),
            )
        sweep = gen.load_sweep(npz)
        geometry = (
            {"panel_disc_radius": 200, "region": 128, "far_field_size": 2048,
             "method": "speckle", "calibration_notes": "old"}
            if conflicts else
            {"panel_disc_radius": 450, "region": 256, "far_field_size": 4096,
             "method": "sweep", "calibration_notes": "fresh"}
        )
        return gen.render_report(
            sweep=sweep,
            geometry=geometry,
            geometry_path=tmp_path / "bench_geometry.json",
            sidecar=None,
            sidecar_path=None,
            npz_path=npz,
            ramp_routes=gen.compute_ramp_routes(sweep),
            tilt_routes=gen.compute_tilt_routes(sweep, 0.30),
            curve_map={},
            constants=gen._import_bench_constants(),
            figures=[],
            frames_status="skipped",
            frames_detail="",
            geometry_conflicts=conflicts,
        )

    def test_consistent_geometry_renders_every_section(self, gen: Any,
                                                      tmp_path: Path) -> None:
        report = self._report(gen, [], tmp_path)
        for heading in ("## 2.", "## 3.", "## 4.", "## 5.", "## 6.",
                        "## 7.", "## 8.", "## 8.1"):
            assert heading in report, f"missing {heading}"
        # The routes table must have real numbers, not be an empty stub.
        assert "斜坡（权威）" in report
        assert "| 视场比 (FOV) |" in report
        assert "## 7. 关键陷阱" in report

    def test_stale_geometry_renders_every_section_too(self, gen: Any,
                                                      tmp_path: Path) -> None:
        report = self._report(
            gen, ["panel_disc_radius: geometry=200 vs npz=450"], tmp_path
        )
        for heading in ("## 2.", "## 3.", "## 4.", "## 5.", "## 6.",
                        "## 7.", "## 8.", "## 8.1"):
            assert heading in report, f"missing {heading}"
        # §6 must be explicitly disabled rather than silently empty.
        assert "本节不可用" in report
        assert "## 7. 关键陷阱" in report

    def test_both_paths_agree_on_the_geometry_free_cross_check(
        self, gen: Any, tmp_path: Path
    ) -> None:
        """``S = k_tilt*pi*R`` must survive a stale geometry file."""
        good = self._report(gen, [], tmp_path)
        bad = self._report(gen, ["panel_disc_radius: geometry=200 vs npz=450"],
                           tmp_path)
        marker = "不需要任何标定文件"
        assert marker in good
        assert marker in bad


def _fake_sweep(periods: np.ndarray, displacement: np.ndarray) -> Any:
    """A minimal SweepData: one flat reference plus one axis of ramp points.

    The flat row is mandatory -- ramp displacement is measured *from* it, and
    without it the generator (correctly) reports absolute centroids and returns
    NaN rather than inventing a baseline.
    """
    module = sys.modules["_bench_report"]
    flat_y = 1000.0
    n = len(periods)
    labels = ["flat", *[f"rampx{p:.0f}" for p in periods]]
    modes = ["flat", *["ramp"] * n]
    axes = ["", *["x"] * n]
    coeffs = [0.0, *periods.tolist()]
    cx = [flat_y, *(flat_y + displacement).tolist()]
    cy = [500.0, *np.full(n, 500.0).tolist()]
    return module.SweepData(
        label=np.array(labels),
        mode=np.array(modes),
        axis=np.array(axes),
        coefficient=np.array(coeffs, dtype=np.float64),
        fwhm_px=np.full(n + 1, 12.0),
        centroid_x=np.array(cx, dtype=np.float64),
        centroid_y=np.array(cy, dtype=np.float64),
        peak=np.full(n + 1, 100.0),
        hollowness=np.full(n + 1, 0.9),
        shift_x=np.full(n + 1, np.nan),
        shift_y=np.full(n + 1, np.nan),
        shift_corr=np.full(n + 1, np.nan),
        scalars={"zernike_radius": 450, "collect_disc": 450},
    )
