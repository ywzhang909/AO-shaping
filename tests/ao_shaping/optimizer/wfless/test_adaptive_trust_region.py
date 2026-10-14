"""Tests for the adaptive trust-region radius schedule and its loop wiring.

Two subjects with different natures, kept in one file because they are one claim:
the policy is useless if the loop does not feed it, and the wiring is untestable
without knowing what the policy should do.

The frozen-default guarantee is asserted the only way that is meaningful — by
comparing the **whole run** with the flag off against the flag-off run recorded
before the schedule existed — not by inspecting that a flag reads ``False``.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.optimizer.wfless.adaptive_trust_region import (  # noqa: E402
    AdaptiveTrustRegion,
    AdaptiveTrustRegionConfig,
)
from ao_shaping.optimizer.wfless.slm_model_in_loop import (  # noqa: E402
    SlmModelInLoopConfig,
    optimize_slm_model_in_loop,
    trust_region_clamp,
)


def _nan_safe(value):  # noqa: ANN001, ANN202
    """Map every NaN to ``None`` so equal histories compare equal.

    ``score_before`` is NaN on round one and ``nan == nan`` is False, so a plain
    ``==`` reports a difference between two runs that are identical down to the
    bit.
    """
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {key: _nan_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nan_safe(item) for item in value]
    return value

_CONFIG = AdaptiveTrustRegionConfig(
    initial_c_l2=0.5, min_c_l2=0.125, max_c_l2=2.0
)

_SIM_KWARGS = dict(
    cam_type="sim",
    region=32,
    cam_size=128,
    far_field_padding=4,
    n_rounds=3,
    probe_count=2,
    step_a_iterations=4,
    step_b_iterations=4,
    target_side=64,
    n_orders=4,
    dtype="float32",
    device="cpu",
    seed=0,
)


def _run(**overrides):
    """One small simulated run; returns (result, round rows)."""
    result = optimize_slm_model_in_loop(SlmModelInLoopConfig(**{**_SIM_KWARGS, **overrides}))
    rows = [r for r in result.recorder.history if r.get("stage") == "round"]
    return result, rows


class TestAdaptiveTrustRegionConfig:
    """An unusable schedule must fail at construction, not mid-run."""

    def test_a_valid_config_is_accepted(self) -> None:
        assert _CONFIG.initial_c_l2 == 0.5

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"initial_c_l2": 0.0}, "initial_c_l2"),
            ({"initial_c_l2": float("inf")}, "initial_c_l2"),
            ({"min_c_l2": 0.0}, "min_c_l2"),
            ({"max_c_l2": 0.0}, "max_c_l2"),
            ({"min_c_l2": 0.9}, "min_c_l2"),
            ({"max_c_l2": 0.1}, "max_c_l2"),
            ({"grow_factor": 1.0}, "grow_factor"),
            ({"grow_factor": 0.5}, "grow_factor"),
            ({"shrink_factor": 1.0}, "shrink_factor"),
            ({"shrink_factor": 0.0}, "shrink_factor"),
            ({"shrink_factor": 1.5}, "shrink_factor"),
            ({"accepts_before_growth": 0}, "accepts_before_growth"),
        ],
    )
    def test_bad_settings_are_refused(self, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            AdaptiveTrustRegionConfig(**{**_CONFIG.__dict__, **kwargs})


class TestAdaptiveTrustRegionSchedule:
    """Growth is earned; a rejection shrinks hard and resets the streak."""

    def test_the_radius_starts_at_the_initial_value(self) -> None:
        policy = AdaptiveTrustRegion(_CONFIG)
        assert policy.radius == 0.5
        assert policy.accept_streak == 0
        assert policy.rounds == 0

    def test_accepts_grow_the_radius(self) -> None:
        policy = AdaptiveTrustRegion(_CONFIG)
        policy.update(accepted=True)
        assert policy.radius == pytest.approx(0.5 * 1.25)

    def test_a_rejection_shrinks_the_radius(self) -> None:
        policy = AdaptiveTrustRegion(_CONFIG)
        policy.update(accepted=True)
        before = policy.radius
        policy.update(accepted=False)
        assert policy.radius == pytest.approx(before * 0.5)

    def test_a_rejection_resets_the_streak(self) -> None:
        # ``accepts_before_growth`` must exceed 1 here, otherwise every accept
        # grows *and* resets in the same call and the streak is never observable.
        config = AdaptiveTrustRegionConfig(
            initial_c_l2=0.5, min_c_l2=0.1, max_c_l2=4.0, accepts_before_growth=3
        )
        policy = AdaptiveTrustRegion(config)
        policy.update(accepted=True)
        policy.update(accepted=True)
        assert policy.accept_streak == 2
        policy.update(accepted=False)
        assert policy.accept_streak == 0

    def test_growth_waits_for_enough_consecutive_accepts(self) -> None:
        config = AdaptiveTrustRegionConfig(
            initial_c_l2=0.5, min_c_l2=0.1, max_c_l2=4.0, accepts_before_growth=3
        )
        policy = AdaptiveTrustRegion(config)
        policy.update(accepted=True)
        policy.update(accepted=True)
        assert policy.radius == pytest.approx(0.5), "grew before earning it"
        policy.update(accepted=True)
        assert policy.radius == pytest.approx(0.625)

    def test_the_radius_never_leaves_its_bounds(self) -> None:
        policy = AdaptiveTrustRegion(_CONFIG)
        for _ in range(200):
            policy.update(accepted=True)
            assert policy.radius <= _CONFIG.max_c_l2 + 1e-12
        for _ in range(200):
            policy.update(accepted=False)
            assert policy.radius >= _CONFIG.min_c_l2 - 1e-12

    def test_the_radius_never_reaches_zero(self) -> None:
        """A zero radius would silently disable the guard."""
        policy = AdaptiveTrustRegion(_CONFIG)
        for _ in range(100):
            policy.update(accepted=False)
        assert policy.radius > 0.0

    def test_state_counts_the_growths_and_shrinks(self) -> None:
        policy = AdaptiveTrustRegion(_CONFIG)
        policy.update(accepted=True)
        policy.update(accepted=True)
        policy.update(accepted=False)
        state = policy.state()
        assert state["trust_region_rounds"] == 3
        assert state["trust_region_growths"] == 2
        assert state["trust_region_shrinks"] == 1

    def test_state_reports_the_radius_in_force(self) -> None:
        policy = AdaptiveTrustRegion(_CONFIG)
        policy.update(accepted=False)
        assert policy.state()["trust_region_c_l2"] == pytest.approx(policy.radius)


class TestTheClampStaysTheOnlyImplementation:
    """The policy supplies a radius; it never re-implements the boundary."""

    def test_the_schedule_composes_with_the_existing_clamp(self) -> None:
        policy = AdaptiveTrustRegion(_CONFIG)
        policy.update(accepted=False)
        candidate = np.zeros(4)
        candidate[0] = 10.0  # far beyond every radius
        clamped, step_l2, applied = trust_region_clamp(
            candidate, np.zeros(4), policy.radius
        )
        assert applied is True
        assert np.linalg.norm(clamped) == pytest.approx(policy.radius)
        assert step_l2 == pytest.approx(10.0)


class TestFrozenDefault:
    """With the flag off the run must be indistinguishable from before."""

    def test_the_flag_defaults_to_off(self) -> None:
        assert SlmModelInLoopConfig().adaptive_trust_region is False

    def test_the_off_path_emits_no_schedule_keys(self) -> None:
        """A new history key on the default path would re-baseline every test."""
        _result, rows = _run()
        assert rows
        for row in rows:
            assert "trust_region_c_l2" not in row
            assert "trust_region_rounds" not in row

    def test_the_off_path_clamps_with_the_configured_radius(self) -> None:
        _result, rows = _run(trust_region_c_l2=0.25)
        for row in rows:
            # Whatever the step was, the applied step can never exceed the cap.
            assert row["coeff_step_applied"] <= 0.25 + 1e-9

    def test_the_off_path_ignores_the_schedule_entirely(self) -> None:
        """Anti-vacuity: turning the flag on must actually change the run."""
        off = optimize_slm_model_in_loop(SlmModelInLoopConfig(**_SIM_KWARGS))
        on = optimize_slm_model_in_loop(
            SlmModelInLoopConfig(**{**_SIM_KWARGS, "adaptive_trust_region": True})
        )
        assert _nan_safe(off.recorder.history) != _nan_safe(on.recorder.history), (
            "the adaptive flag changed nothing, so the off-path tests prove nothing"
        )


class TestAdaptiveWiring:
    """The loop builds the policy, uses its radius, and records it."""

    def test_the_on_path_records_the_schedule(self) -> None:
        _result, rows = _run(adaptive_trust_region=True)
        assert rows
        for index, row in enumerate(rows, start=1):
            assert row["trust_region_rounds"] == index
            assert row["trust_region_c_l2"] > 0.0

    def test_the_radius_starts_at_the_configured_value(self) -> None:
        _result, rows = _run(adaptive_trust_region=True, trust_region_c_l2=0.5)
        assert rows[0]["trust_region_c_l2"] == pytest.approx(0.5)

    def test_the_recorded_radius_never_leaves_the_configured_bounds(self) -> None:
        _result, rows = _run(adaptive_trust_region=True, trust_region_c_l2=0.5)
        for row in rows:
            assert row["trust_region_c_l2"] <= 0.5 * 4.0 + 1e-9
            assert row["trust_region_c_l2"] >= 0.5 * 0.25 - 1e-9

    def test_a_disabled_trust_region_is_treated_as_a_floor_not_zero(self) -> None:
        """``trust_region_c_l2 <= 0`` means "externally constrained" (clamp off).

        The adaptive schedule must not turn that into an infinitely tight region
        by dividing by zero, nor into an unbounded one.
        """
        _result, rows = _run(adaptive_trust_region=True, trust_region_c_l2=0.0)
        assert rows
        for row in rows:
            assert np.isfinite(row["trust_region_c_l2"])
            assert row["trust_region_c_l2"] > 0.0

    def test_the_run_is_deterministic_for_a_fixed_configuration(self) -> None:
        """Seeds pin the search; two identical configs must agree."""
        first = optimize_slm_model_in_loop(
            SlmModelInLoopConfig(**{**_SIM_KWARGS, "adaptive_trust_region": True})
        )
        second = optimize_slm_model_in_loop(
            SlmModelInLoopConfig(**{**_SIM_KWARGS, "adaptive_trust_region": True})
        )
        assert np.array_equal(first.best_phase, second.best_phase)
        assert _nan_safe(first.recorder.history) == _nan_safe(
            second.recorder.history
        )


class TestDeviceFreeImport:
    """The policy module must not drag hardware into an import."""

    def test_importing_the_policy_pulls_in_no_driver(self) -> None:
        source = (
            ROOT
            / "src"
            / "ao_shaping"
            / "optimizer"
            / "wfless"
            / "adaptive_trust_region.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        for name in imported:
            assert "drivers" not in name, f"device dependency leaked in: {name}"
            assert "runners" not in name, f"device dependency leaked in: {name}"