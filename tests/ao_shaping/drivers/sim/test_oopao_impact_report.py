"""Tests for the OOPAO backend impact report generator.

Covers the offline report script ``scripts/generate_oopao_impact_report.py``,
which drives ``SimTurbulenceAOEnv`` under both AO backends and writes
``report/oopao_impact/{report.md,summary.csv,figures/}``.

OOPAO is optional: tests that need it are skipped when
``oopao_backend._oopao_available()`` is False, so a machine without OOPAO still
runs the pure-logic tests. The end-to-end run is marked ``slow`` (exclude with
``-m 'not slow'``).

What is pinned here:

* **Routing** -- ``backend_arm`` really flips ``_oopao_enabled()`` and clears the
  ``oopao_backend._get_backend`` lru_cache on enter *and* exit.
* **cn2=0 control** -- both arms must be bit-identical, because
  ``turbulence_phase`` short-circuits to a zero screen *before* the backend
  switch.
* **Anti-vacuity** -- the guards raise when the arms do not actually differ, so a
  silent two-arm-numpy regression cannot slip through.
* **Ratio analysis** -- the ``disturbance_rms`` oopao/numpy ratio is constant
  across turbulence levels (a multiplicative calibration offset, not noise), and
  the constancy verdict is *computed* from a tolerance rather than asserted. A
  single level must not be able to claim constancy.
* **Metric identity** -- ``init_rms`` and ``disturbance_rms`` are different
  quantities; a stronger screen does not imply a larger ``init_rms``.
* **Dead config** -- ``slm_shaping_bench`` ignores ``cn2`` entirely.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

oopao_backend = pytest.importorskip("ao_shaping.drivers.sim.oopao_backend")

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "generate_oopao_impact_report.py"


def _load_script() -> ModuleType:
    """Import the generator by path (``scripts/`` is not a package)."""
    if not SCRIPT.is_file():
        pytest.skip(f"generator not present at {SCRIPT}")
    spec = importlib.util.spec_from_file_location("_oopao_impact_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


impact = _load_script()

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


def _row(**overrides):
    """One synthetic cell."""
    fields = dict(
        mode=impact.MODE_OPEN,
        cn2=5e-15,
        arm=impact.ARM_NUMPY,
        n_grid=32,
        seed=42,
        steps=6,
        init_strehl=0.90,
        final_strehl=0.80,
        best_strehl=0.95,
        init_pib=100.0,
        final_pib=90.0,
        best_pib=110.0,
        init_rms=1.0,
        final_rms=0.5,
        disturbance_rms=1.0,
        trace=[0.90, 0.85, 0.80],
        oopao_enabled=False,
    )
    fields.update(overrides)
    return impact.EpisodeResult(**fields)


def _matrix(cn2_values, *, numpy_values=None, oopao_values=None) -> list:
    """Full (mode x cn2 x arm) result set -- the guards require every cell.

    ``numpy_values`` / ``oopao_values`` are ``field -> value`` dicts applied to
    that arm's rows. Passing the *same* dict to both arms makes every cell
    degenerate, which is what the negative tests need.
    """
    numpy_values = numpy_values or {}
    oopao_values = oopao_values or {}
    rows = []
    for mode in impact.MODES:
        for cn2 in cn2_values:
            for arm in (impact.ARM_NUMPY, impact.ARM_OOPAO):
                overrides = dict(
                    numpy_values if arm == impact.ARM_NUMPY else oopao_values
                )
                rows.append(
                    _row(
                        mode=mode,
                        cn2=cn2,
                        arm=arm,
                        oopao_enabled=arm == impact.ARM_OOPAO,
                        **overrides,
                    )
                )
    return rows


class TestBackendArm:
    """``backend_arm`` must genuinely switch the backend and clear the cache."""

    def test_numpy_arm_reports_disabled(self):
        from ao_shaping.drivers.sim import beam_backend as bb

        with impact.backend_arm(impact.ARM_NUMPY):
            assert bb._oopao_enabled() is False

    @requires_oopao
    def test_oopao_arm_reports_enabled(self):
        from ao_shaping.drivers.sim import beam_backend as bb

        with impact.backend_arm(impact.ARM_OOPAO):
            assert bb._oopao_enabled() is True

    @requires_oopao
    def test_cache_is_cleared_on_exit(self):
        """Leaving the arm must not leak a cached OOPAO Atmosphere instance."""
        with impact.backend_arm(impact.ARM_OOPAO):
            pass
        oopao_backend._get_backend.cache_clear()
        assert oopao_backend._get_backend.cache_info().currsize == 0


class TestVerifyGuards:
    """The report's self-checks must actually reject bad data."""

    def test_routing_mismatch_raises(self):
        rows = [_row(arm=impact.ARM_OOPAO, oopao_enabled=False)]
        with pytest.raises(RuntimeError, match="路由"):
            impact.verify_backend_routing(rows)

    def test_routing_consistent_passes(self):
        rows = [
            _row(arm=impact.ARM_NUMPY, oopao_enabled=False),
            _row(arm=impact.ARM_OOPAO, oopao_enabled=True),
        ]
        impact.verify_backend_routing(rows)

    def test_anti_vacuity_raises_when_arms_identical(self):
        """Identical arms at cn2>0 mean the switch did nothing -> must raise."""
        rows = _matrix([5e-15])
        with pytest.raises(RuntimeError, match="反空洞"):
            impact.verify_anti_vacuity(rows, [5e-15])

    def test_anti_vacuity_passes_when_compared_metrics_differ(self):
        # The guard compares (init_strehl, best_strehl, init_pib, init_rms) --
        # not disturbance_rms, despite that being the report's headline metric.
        rows = _matrix([5e-15], oopao_values=dict(init_strehl=0.50))
        assert impact.verify_anti_vacuity(rows, [5e-15]) == len(impact.MODES)

    def test_anti_vacuity_ignores_disturbance_only_difference(self):
        """A screen-only difference does NOT satisfy the guard's metric set."""
        rows = _matrix([5e-15], oopao_values=dict(disturbance_rms=5.0))
        with pytest.raises(RuntimeError):
            impact.verify_anti_vacuity(rows, [5e-15])

    def test_anti_vacuity_ignores_zero_cn2(self):
        """cn2=0 arms are identical by design and must not satisfy the guard."""
        rows = _matrix([0.0])
        with pytest.raises(RuntimeError):
            impact.verify_anti_vacuity(rows, [0.0])

    def test_control_passes_on_identical_zero_cn2_arms(self):
        # Both arms must get the SAME values -- a cn2=0 control is only
        # meaningful when the two arms are genuinely indistinguishable.
        zero = dict(
            init_strehl=1.0,
            final_strehl=1.0,
            best_strehl=1.0,
            init_pib=100.0,
            final_pib=100.0,
            best_pib=100.0,
            init_rms=0.0,
            final_rms=0.0,
            disturbance_rms=0.0,
        )
        rows = _matrix([0.0], numpy_values=zero, oopao_values=dict(zero))
        assert impact.verify_control(rows, [0.0]) is True

    def test_control_fails_when_zero_cn2_arms_differ(self):
        """cn2=0 must short-circuit to a zero screen; drift means a regression."""
        rows = _matrix([0.0], oopao_values=dict(disturbance_rms=0.5))
        assert impact.verify_control(rows, [0.0]) is False

    def test_control_skipped_when_zero_cn2_absent(self):
        rows = _matrix([5e-15], oopao_values=dict(disturbance_rms=5.0))
        assert impact.verify_control(rows, [5e-15]) is True


class TestRatioAnalysis:
    """The constancy analysis is computed, and honest about thin data."""

    def test_ratio_computed_from_arms(self):
        rows = _matrix([5e-15], oopao_values=dict(disturbance_rms=10.0))
        assert impact.disturbance_ratio(rows, impact.MODE_OPEN, 5e-15) == pytest.approx(
            10.0
        )

    def test_ratio_undefined_at_zero_cn2(self):
        """cn2<=0 -> zero screen -> undefined (must not divide by zero)."""
        rows = _matrix([0.0])
        assert impact.disturbance_ratio(rows, impact.MODE_OPEN, 0.0) is None

    def test_ratio_none_when_arm_missing(self):
        rows = [_row(arm=impact.ARM_NUMPY, cn2=5e-15, disturbance_rms=2.0)]
        assert impact.disturbance_ratio(rows, impact.MODE_OPEN, 5e-15) is None

    def test_profiles_cover_every_mode_and_skip_zero_cn2(self):
        rows = _matrix(
            [0.0, 5e-15, 5e-14],
            numpy_values=dict(disturbance_rms=1.0),
            oopao_values=dict(disturbance_rms=5.0),
        )
        profiles = impact.ratio_profiles(rows, [0.0, 5e-15, 5e-14])
        assert {p.mode for p in profiles} == set(impact.MODES)
        for profile in profiles:
            assert profile.cn2_levels == [5e-15, 5e-14]
            assert profile.ratios == pytest.approx([5.0, 5.0])

    def test_constant_ratio_is_flagged_constant(self):
        rows = _matrix(
            [5e-15, 5e-14],
            numpy_values=dict(disturbance_rms=1.0),
            oopao_values=dict(disturbance_rms=5.0),
        )
        profile = impact.ratio_profiles(rows, [5e-15, 5e-14])[0]
        assert profile.is_constancy_claimable is True
        assert profile.is_constant is True
        assert profile.spread_rel == pytest.approx(0.0, abs=1e-9)

    def test_single_level_cannot_claim_constancy(self):
        """One cn2>0 level gives no ladder -> must refuse to claim constancy."""
        rows = _matrix(
            [5e-15],
            numpy_values=dict(disturbance_rms=1.0),
            oopao_values=dict(disturbance_rms=5.0),
        )
        profile = impact.ratio_profiles(rows, [5e-15])[0]
        assert profile.is_constancy_claimable is False
        assert profile.is_constant is False

    def test_drifting_ratio_is_not_constant(self):
        """A drifting ratio must be detected, not rubber-stamped as constant."""
        rows = []
        for mode in impact.MODES:
            for cn2, ratio in ((5e-15, 5.0), (5e-14, 9.0)):
                rows.append(
                    _row(
                        mode=mode,
                        cn2=cn2,
                        arm=impact.ARM_NUMPY,
                        disturbance_rms=1.0,
                    )
                )
                rows.append(
                    _row(
                        mode=mode,
                        cn2=cn2,
                        arm=impact.ARM_OOPAO,
                        oopao_enabled=True,
                        disturbance_rms=ratio,
                    )
                )
        profile = impact.ratio_profiles(rows, [5e-15, 5e-14])[0]
        assert profile.is_constancy_claimable is True
        assert profile.is_constant is False

    def test_tolerance_is_tight(self):
        """Tolerance must stay far below the drift scale it is meant to catch."""
        assert 0 < impact.RATIO_CONSTANT_TOL_REL < 1e-2


class TestMetricIdentity:
    """``init_rms`` and ``disturbance_rms`` must never be conflated."""

    def test_inversion_detected(self):
        """oopao screen stronger but init_rms lower -> the warning case."""
        rows = _matrix(
            [5e-14],
            numpy_values=dict(init_rms=2.16, disturbance_rms=4.06),
            oopao_values=dict(init_rms=1.80, disturbance_rms=22.06),
        )
        found = impact._init_rms_inversion(rows, [5e-14])
        assert found is not None
        _, numpy_row, oopao_row = found
        assert numpy_row.init_rms > oopao_row.init_rms
        assert oopao_row.disturbance_rms > numpy_row.disturbance_rms

    def test_no_inversion_when_monotonic(self):
        rows = _matrix(
            [5e-15],
            numpy_values=dict(init_rms=1.0, disturbance_rms=1.0),
            oopao_values=dict(init_rms=5.0, disturbance_rms=5.0),
        )
        assert impact._init_rms_inversion(rows, [5e-15]) is None

    def test_strehl_gap_ignores_non_positive_gap(self):
        """Never claim 'oopao higher' while citing a smaller number."""
        rows = _matrix(
            [5e-15],
            numpy_values=dict(init_strehl=0.50),
            oopao_values=dict(init_strehl=0.40),
        )
        assert impact._max_init_strehl_gap(rows, [5e-15]) is None

    def test_strehl_gap_reports_positive_gap(self):
        rows = _matrix(
            [5e-14],
            numpy_values=dict(init_strehl=0.3276),
            oopao_values=dict(init_strehl=0.4650),
        )
        found = impact._max_init_strehl_gap(rows, [5e-14])
        assert found is not None
        _, numpy_row, oopao_row = found
        assert oopao_row.init_strehl > numpy_row.init_strehl

    def test_rms_reduction_zero_without_turbulence(self):
        """cn2=0 leaves nothing to correct; must not emit a bogus huge ratio."""
        assert _row(init_rms=0.0, final_rms=0.0).rms_reduction_rel == 0.0


class TestDeadConfig:
    """``slm_shaping_bench.cn2`` is dead config -- it must stay that way."""

    def test_forward_intensity_ignores_cn2(self):
        probe = impact.probe_slm_shaping_bench_dead_cn2()
        assert probe["max_abs_diff"] == pytest.approx(0.0)
        assert probe["byte_identical"] is True


@pytest.mark.slow
@requires_oopao
class TestEndToEnd:
    """Real episodes through the routed environment, then a full report run."""

    def test_guards_hold_on_live_data(self):
        cn2_values = [0.0, 5e-15]
        results = [
            impact.run_episode(
                mode=mode,
                cn2=cn2,
                arm=arm,
                n_grid=32,
                seed=42,
                steps=6,
            )
            for cn2 in cn2_values
            for mode in impact.MODES
            for arm in (impact.ARM_NUMPY, impact.ARM_OOPAO)
        ]
        impact.verify_backend_routing(results)
        assert impact.verify_control(results, cn2_values) is True
        assert impact.verify_anti_vacuity(results, cn2_values) >= 1

        zero = {r.arm: r.disturbance_rms for r in results if r.cn2 == 0.0}
        assert zero[impact.ARM_NUMPY] == zero[impact.ARM_OOPAO] == 0.0

    def test_full_report_run(self, tmp_path: Path):
        out = tmp_path / "impact"
        exit_code = impact.main(
            [
                "--n-grid",
                "32",
                "--cn2",
                "0,5e-15",
                "--steps",
                "6",
                "--out-dir",
                str(out),
            ]
        )
        assert exit_code == 0
        assert (out / "report.md").is_file()
        assert (out / "summary.csv").is_file()
        assert list((out / "figures").glob("*.png"))

        report = (out / "report.md").read_text(encoding="utf-8")
        # The documented claims must survive into the rendered report.
        assert "4.3" in report
        assert "非可比性" in report
        # Every linked figure must exist on disk.
        for line in report.splitlines():
            if "![" in line and "](" in line:
                target = line.split("](", 1)[1].split(")", 1)[0]
                assert (out / target).is_file(), f"missing figure: {target}"
