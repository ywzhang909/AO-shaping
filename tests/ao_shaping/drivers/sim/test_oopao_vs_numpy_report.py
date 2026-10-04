"""Tests for the OOPAO vs legacy-numpy comparison report generator.

Covers ``scripts/generate_oopao_vs_numpy_report.py``, which compares the two
phase-screen / focal-plane backends across an aberration x turbulence matrix and
writes ``report/oopao_vs_numpy/{report.md,summary.csv,figures/}``.

Most of these tests are pure logic (no OOPAO needed). The routing tests skip
when ``oopao_backend._oopao_available()`` is False.

What is pinned here:

* **The ``energy_frac`` exclusion** -- ``energy_frac`` goes through
  ``propagate()``, the one place the two kernels legitimately differ, so it is
  deliberately kept out of the cn2=0 equality set. Adding it back would make
  the control permanently fail.
* **Fail-fast** -- ``assert_oopao_usable()`` must abort rather than emit a
  two-arm-numpy report wearing an "oopao" label.
* **Arm switching** -- ``activate_arm`` clears the ``_get_backend`` lru_cache
  (its cache key omits the arm flag, so a stale entry silently degrades the
  oopao arm back to numpy) and rejects unknown arm names.
* **Matrix sanity** -- the Cn2 ladder is strictly increasing and contains a
  cn2=0 control; every equality metric has a CSV column and a display label.
* **Metric math** -- ``energy_fraction`` on identical fields is 1.0, and
  degenerate/empty inputs return 0.0 rather than raising.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

import numpy as np

oopao_backend = pytest.importorskip("ao_shaping.drivers.sim.oopao_backend")

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "generate_oopao_vs_numpy_report.py"


def _load_script() -> ModuleType:
    """Import the generator by path (``scripts/`` is not a package)."""
    if not SCRIPT.is_file():
        pytest.skip(f"generator not present at {SCRIPT}")
    spec = importlib.util.spec_from_file_location("_oopao_vs_numpy_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report = _load_script()

requires_oopao = pytest.mark.skipif(
    not bool(oopao_backend._oopao_available()),
    reason="OOPAO library not available in this environment",
)


@pytest.fixture(autouse=True)
def _restore_backend_env():
    """Never leak ``AO_OOPAO_BACKEND`` into other tests."""
    previous = os.environ.get("AO_OOPAO_BACKEND")
    yield
    if previous is None:
        os.environ.pop("AO_OOPAO_BACKEND", None)
    else:
        os.environ["AO_OOPAO_BACKEND"] = previous


class TestArmInvariantMetricSet:
    """``energy_frac`` must stay out of the cn2=0 equality set."""

    def test_energy_frac_excluded_from_invariants(self):
        assert "energy_frac" not in report.ARM_INVARIANT_METRICS

    def test_invariants_cover_phase_and_focal_metrics(self):
        expected = {"phase_std_rad", "phase_rms_rad", "strehl", "fwhm_px", "ee_r4"}
        assert set(report.ARM_INVARIANT_METRICS) == expected

    def test_invariants_are_csv_columns(self):
        """Every equality metric must actually be written to summary.csv."""
        assert set(report.ARM_INVARIANT_METRICS) <= set(report.CSV_COLUMNS)

    def test_every_metric_has_a_label(self):
        labelled = {metric for metric, _ in report.METRIC_LABELS}
        assert set(report.ARM_INVARIANT_METRICS) <= labelled
        assert "energy_frac" in labelled

    def test_metric_labels_are_unique(self):
        metrics = [metric for metric, _ in report.METRIC_LABELS]
        assert len(metrics) == len(set(metrics))


class TestMatrixDefinition:
    """The scenario matrix must keep a cn2=0 control and a rising ladder."""

    def test_turbulence_ladder_strictly_increases(self):
        cn2_values = [case.cn2 for case in report.TURBULENCES]
        assert cn2_values == sorted(cn2_values)
        assert len(set(cn2_values)) == len(cn2_values)

    def test_zero_cn2_control_present(self):
        """Without cn2=0 there is no 'pure aberration' column to compare against."""
        assert any(case.cn2 <= 0.0 for case in report.TURBULENCES)

    def test_aberration_names_unique(self):
        names = [case.name for case in report.ABERRATIONS]
        assert len(names) == len(set(names))

    def test_default_aberrations_exclude_spherical(self):
        """The CLI default is a strict subset -- 'spherical' is opt-in."""
        assert "spherical" in {case.name for case in report.ABERRATIONS}
        parsed = report.parse_args([])
        assert "spherical" not in parsed.aberrations.split(",")

    def test_cn2_label_formats_zero_without_exponent(self):
        zero = next(case for case in report.TURBULENCES if case.cn2 <= 0.0)
        assert zero.cn2_label == "0"
        nonzero = next(case for case in report.TURBULENCES if case.cn2 > 0.0)
        assert "e" in nonzero.cn2_label


class TestEnergyFraction:
    """``energy_fraction`` is the ASM energy-conservation ratio."""

    def test_identical_fields_preserve_energy(self):
        field = np.ones((8, 8), dtype=np.complex128)
        assert report.energy_fraction(field, field) == pytest.approx(1.0)

    def test_halved_amplitude_is_quarter_energy(self):
        field = np.ones((8, 8), dtype=np.complex128)
        assert report.energy_fraction(field, field * 0.5) == pytest.approx(0.25)

    def test_zero_input_returns_zero(self):
        """Degenerate input must return 0.0, not divide by zero."""
        field = np.zeros((4, 4), dtype=np.complex128)
        assert report.energy_fraction(field, np.ones((4, 4))) == 0.0

    def test_output_only_energy_matters(self):
        inp = np.ones((4, 4), dtype=np.complex128)
        out = np.full((4, 4), 2.0, dtype=np.complex128)
        assert report.energy_fraction(inp, out) == pytest.approx(4.0)


class TestMaskedMetrics:
    """``rms_in_mask`` / ``std_in_mask`` must tolerate an empty pupil."""

    def test_rms_empty_mask_is_zero(self):
        phase = np.ones((4, 4))
        mask = np.zeros((4, 4), dtype=bool)
        assert report.rms_in_mask(phase, mask) == 0.0

    def test_std_empty_mask_is_zero(self):
        phase = np.ones((4, 4))
        mask = np.zeros((4, 4), dtype=bool)
        assert report.std_in_mask(phase, mask) == 0.0

    def test_rms_is_measured_about_zero_not_the_mean(self):
        """Wavefront phase RMS is about zero, unlike std_in_mask (about the mean).

        A uniform 0.7 rad offset is a real wavefront error, so its RMS must be
        0.7 -- only the standard deviation of the same field is 0.
        """
        phase = np.full((4, 4), 0.7)
        mask = np.ones((4, 4), dtype=bool)
        assert report.rms_in_mask(phase, mask) == pytest.approx(0.7)
        assert report.std_in_mask(phase, mask) == pytest.approx(0.0)

    def test_pupil_mask_is_boolean_and_correct_shape(self):
        mask = report.pupil_mask(16)
        assert mask.shape == (16, 16)
        assert mask.dtype == bool
        assert mask.any()
        assert not mask.all()


class TestActivateArm:
    """``activate_arm`` must switch, clear the cache, and reject bad names."""

    def test_unknown_arm_raises(self):
        with pytest.raises(ValueError):
            report.activate_arm("not-an-arm", seed=0)

    def test_numpy_arm_disables_backend(self):
        from ao_shaping.drivers.sim import beam_backend as bb

        report.activate_arm(report.ARM_NUMPY, seed=0)
        assert bb._oopao_enabled() is False

    @requires_oopao
    def test_oopao_arm_enables_backend(self):
        from ao_shaping.drivers.sim import beam_backend as bb

        report.activate_arm(report.ARM_OOPAO, seed=0)
        assert bb._oopao_enabled() is True
        report.activate_arm(report.ARM_NUMPY, seed=0)

    @requires_oopao
    def test_switching_arms_does_not_reuse_cached_backend(self):
        """A stale lru_cache entry silently degrades the oopao arm to numpy."""
        report.activate_arm(report.ARM_OOPAO, seed=0)
        oopao_backend._get_backend.cache_clear()
        report.activate_arm(report.ARM_NUMPY, seed=0)
        assert oopao_backend._get_backend.cache_info().currsize == 0


class TestFailFast:
    """The script must abort rather than fake an oopao arm."""

    def test_raises_when_oopao_unavailable(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(report.oopao_backend, "_oopao_available", lambda: False)
        with pytest.raises(report.OopaoBackendUnavailableError):
            report.assert_oopao_usable()

    @requires_oopao
    def test_passes_when_oopao_available(self):
        report.assert_oopao_usable()
