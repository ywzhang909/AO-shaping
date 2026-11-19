"""Wiring tests for uncertainty-sampled Step A probes (R-82 / R-86).

The selector's own maths (predictive variance, normalised ascent, covariance
update, reverse-validation collapse) is unit-tested in
``test_uncertainty_probe_selector.py``. These tests pin only the *integration*
into the model-in-loop runner, which is where a refactor can silently break:

* the two toggles default off and the default run never touches the selector,
  so the historical fixed-random probe path is byte-for-byte unchanged;
* ``uncertainty_probes=True`` replaces the fixed Step A probes (R-82);
* the selected probes are reproducible for a fixed seed (the seed invariant);
* ``uncertainty_on_rejection=True`` switches a round to uncertainty probes only
  once a rejection has happened (R-86), and leaves them off otherwise.

Runs offline on the simulated backend with ``device="cpu"``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    AcceptanceVerdict,
    ModelInLoopStatus,
    SlmModelInLoopConfig,
    optimize_slm_model_in_loop,
)

#: Tiny budget: two rounds so "per round" is observable, small enough to stay fast.
_SIM_KWARGS = dict(
    cam_type="sim",
    region=32,
    cam_size=128,
    far_field_padding=4,
    n_rounds=2,
    probe_count=2,
    step_a_iterations=4,
    step_b_iterations=4,
    target_side=64,
    n_orders=4,
    dtype="float32",
    device="cpu",
    seed=0,
)


def _config(**overrides) -> SlmModelInLoopConfig:
    return SlmModelInLoopConfig(**{**_SIM_KWARGS, **overrides})


def _always_reject(*_args, **_kwargs) -> AcceptanceVerdict:
    return AcceptanceVerdict(
        accepted=False, loss_worsened=True, score_degraded=False, reason="forced"
    )


class _ProbeRecorder:
    """Wrap the two probe sources so a test can see which path each round took."""

    def __init__(self, mod) -> None:
        self.selector_calls: list[list[np.ndarray]] = []
        self.fixed_seeds: list[int] = []
        self._original_selector = mod._select_uncertainty_probes
        self._original_probe = mod._probe_phase

    def selector(self, optimizer, config, seeds, *, prev_probes=None):
        probes = self._original_selector(
            optimizer, config, seeds, prev_probes=prev_probes
        )
        self.selector_calls.append(probes)
        return probes

    def probe(self, region, spread, seed, **kwargs):
        self.fixed_seeds.append(seed)
        return self._original_probe(region, spread, seed, **kwargs)

    def install(self, monkeypatch, mod) -> None:
        monkeypatch.setattr(mod, "_select_uncertainty_probes", self.selector)
        monkeypatch.setattr(mod, "_probe_phase", self.probe)


class TestTogglesDefaultOff:
    """The feature is opt-in; the historical path must be untouched by default."""

    def test_both_toggles_default_off(self) -> None:
        config = SlmModelInLoopConfig()
        assert config.uncertainty_probes is False
        assert config.uncertainty_on_rejection is False

    def test_the_default_run_never_calls_the_selector(self, monkeypatch) -> None:
        from ao_shaping.optimizer.wfless import slm_model_in_loop as mod

        recorder = _ProbeRecorder(mod)
        recorder.install(monkeypatch, mod)
        optimize_slm_model_in_loop(_config())
        assert recorder.selector_calls == []
        # The default run may reject a round and escalate probe_count, so the
        # fixed-probe path runs at least n_rounds * probe_count times.
        assert len(recorder.fixed_seeds) >= (
            _SIM_KWARGS["n_rounds"] * _SIM_KWARGS["probe_count"]
        )

    def test_the_default_run_is_reproducible(self) -> None:
        first = optimize_slm_model_in_loop(_config())
        second = optimize_slm_model_in_loop(_config())
        assert first.status == second.status
        assert np.array_equal(first.best_phase, second.best_phase)


class TestUncertaintyProbesToggle:
    """R-82: ``uncertainty_probes=True`` swaps the fixed Step A probes."""

    def test_the_toggle_replaces_the_fixed_probes(self, monkeypatch) -> None:
        from ao_shaping.optimizer.wfless import slm_model_in_loop as mod

        recorder = _ProbeRecorder(mod)
        recorder.install(monkeypatch, mod)
        result = optimize_slm_model_in_loop(_config(uncertainty_probes=True))

        assert len(recorder.selector_calls) == _SIM_KWARGS["n_rounds"]
        assert recorder.fixed_seeds == []
        assert result.status is ModelInLoopStatus.COMPLETED

    def test_each_round_gets_one_probe_per_train_seed(self, monkeypatch) -> None:
        from ao_shaping.optimizer.wfless import slm_model_in_loop as mod

        recorder = _ProbeRecorder(mod)
        recorder.install(monkeypatch, mod)
        optimize_slm_model_in_loop(
            _config(uncertainty_probes=True, probe_count=3)
        )
        region = _SIM_KWARGS["region"]
        for probes in recorder.selector_calls:
            assert len(probes) == 3
            for probe in probes:
                assert probe.shape == (region, region)
                assert np.all(np.isfinite(probe))
            assert not np.array_equal(probes[0], probes[1])
            assert not np.array_equal(probes[1], probes[2])

    def test_the_probes_are_reproducible_for_a_fixed_seed(self, monkeypatch) -> None:
        from ao_shaping.optimizer.wfless import slm_model_in_loop as mod

        recorder = _ProbeRecorder(mod)
        recorder.install(monkeypatch, mod)
        optimize_slm_model_in_loop(_config(uncertainty_probes=True))
        first_round = recorder.selector_calls[0]

        recorder2 = _ProbeRecorder(mod)
        recorder2.install(monkeypatch, mod)
        optimize_slm_model_in_loop(_config(uncertainty_probes=True))
        second_round = recorder2.selector_calls[0]

        for a, b in zip(first_round, second_round, strict=True):
            assert np.array_equal(a, b)


class TestRejectionEscalation:
    """R-86: a rejected round escalates to uncertainty probes, only when armed."""

    def test_rejection_escalates_only_after_the_first_round(self, monkeypatch) -> None:
        from ao_shaping.optimizer.wfless import slm_model_in_loop as mod

        recorder = _ProbeRecorder(mod)
        recorder.install(monkeypatch, mod)
        monkeypatch.setattr(mod, "acceptance_verdict", _always_reject)

        optimize_slm_model_in_loop(
            _config(
                uncertainty_on_rejection=True,
                n_rounds=3,
                max_rejection_streak=5,
            )
        )
        # round 0 starts at streak 0 -> fixed probes (probe_count=2)
        # rounds 1 and 2 start at streak > 0 -> uncertainty probes
        assert len(recorder.selector_calls) == 2
        assert len(recorder.fixed_seeds) == _SIM_KWARGS["probe_count"]

    def test_without_the_toggle_a_rejection_keeps_fixed_probes(self, monkeypatch) -> None:
        from ao_shaping.optimizer.wfless import slm_model_in_loop as mod

        recorder = _ProbeRecorder(mod)
        recorder.install(monkeypatch, mod)
        monkeypatch.setattr(mod, "acceptance_verdict", _always_reject)

        optimize_slm_model_in_loop(
            _config(uncertainty_on_rejection=False, n_rounds=3, max_rejection_streak=5)
        )
        assert recorder.selector_calls == []
        # probe_count escalates 2 -> 4 -> 6 across the three rejected rounds
        assert len(recorder.fixed_seeds) == 2 + 4 + 6
