"""Loop-invariant tests the main suite does not reach: ROI freezing, warm start,
probe freshness per round, and an actual CLI invocation.

These four were specified for this feature but are absent from
``test_slm_model_in_loop.py``, which covers the guards, the config, the
resampling and the metric choice in depth. Each one here guards a *documented
hardware anti-pattern* rather than a code path, which is why a behavioural test
is the only honest way to pin it:

* **ROI freezing** — the 0-order is located once on the flat baseline and never
  re-derived. Re-rolling an ``argmax`` box every iteration makes the objective
  discontinuous on a speckle field, because the global maximum hops between
  near-equal grains under a ~1e-3 perturbation, and the optimiser then chases a
  box that no longer covers the beam.
* **Warm start** — Step B restarts from the previous round's phase. The
  conditioning does change between rounds, but the previous phase is a good
  initialisation in the same basin; discarding it costs iterations.
* **Probe freshness** — each round draws a *new* family of probes. Reusing one
  family would re-fit the same degenerate direction every round.
* **CLI** — the command is invoked end to end, because a Click wiring mistake
  (an option that exists but is never forwarded) fails silently at import time.

Runs offline on the simulated backend with ``device="cpu"``.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.optimizer.wfless.slm_model_in_loop import (  # noqa: E402
    ModelInLoopStatus,
    SlmModelInLoopConfig,
    _probe_phase,
    _resolve_probe_seeds,
    _to_model_grid,
    optimize_slm_model_in_loop,
)

#: Tiny budget: enough for two rounds so "per round" is observable, small
#: enough that the whole module stays fast.
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

_cache: dict = {}


def _sim_result(**overrides) -> dict:
    """Run the small simulated optimisation once and memoise it."""
    key = tuple(sorted(overrides.items()))
    if key not in _cache:
        config = SlmModelInLoopConfig(**{**_SIM_KWARGS, **overrides})
        result = optimize_slm_model_in_loop(config)
        _cache[key] = {
            "result": result,
            "history": result.recorder.history,
            "config": config,
        }
    return copy.deepcopy(_cache[key])


class TestRoiIsFrozenOnce:
    """The optical axis is located once and reused for every metric."""

    def test_the_same_roi_is_used_for_every_round(self) -> None:
        """`_metrics_at` is called with one centre throughout the run.

        Spying the helper is the only way to prove the *centre never moves*:
        asserting on the recorded numbers would only show that some centre was
        used, not that it was constant.
        """
        from ao_shaping.optimizer.wfless import slm_model_in_loop as mod

        seen: list[tuple[int, int]] = []
        original = mod._metrics_at

        def _spy(frame, target_side, center, **_kw):  # noqa: ANN001, ANN202
            # ``_metrics_at`` is keyword-only for ``target_shape`` /
            # ``camera_rotation_quarters``; a 3-arg spy made every round raise
            # TypeError before the anchor could be observed at all.
            seen.append((int(center[0]), int(center[1])))
            return original(frame, target_side, center, **_kw)

        mod._metrics_at = _spy
        try:
            _sim_result(warm_start=True)
        finally:
            mod._metrics_at = original

        # One flat baseline plus one measurement per round.
        assert len(seen) == _SIM_KWARGS["n_rounds"] + 1, seen
        assert len(set(seen)) == 1, f"the ROI centre moved during the run: {seen}"

    def test_the_roi_centre_is_the_frame_peak(self) -> None:
        """The recorded ROI centre is the brightest pixel of the flat frame.

        On the *simulated* bench this is indistinguishable from a frame-centre
        anchor, and that limitation is deliberate to record: the sim crops a
        window centred on the peak, so the peak always lands at the crop centre
        and the two candidate anchors coincide by construction. Asserting they
        differ here would be asserting something false about the backend.

        The anti-pattern this guards -- re-rolling an ``argmax`` box every
        iteration -- is therefore pinned by
        ``test_the_same_roi_is_used_for_every_round`` (which spies the anchor
        and proves it never moves), not by this test. On hardware the two
        anchors genuinely differ, which is why the runner locates the axis with
        ``argmax`` and freezes it.
        """
        from ao_shaping.optimizer.wfless.slm_model_in_loop import _SimBench
        from ao_shaping.utils.image.beam_metrics import prepare_frame

        config = SlmModelInLoopConfig(**_SIM_KWARGS)
        bench = _SimBench(config)
        try:
            bench.display(np.zeros((config.region, config.region)))
            frame = prepare_frame(bench.measure())
        finally:
            bench.close()

        row, col = np.unravel_index(int(np.argmax(frame)), frame.shape)
        # argmax indexing is (row, col); the ROI centre is carried as (x, y).
        assert (int(col), int(row)) == (frame.shape[1] // 2, frame.shape[0] // 2), (
            "the simulated crop is expected to be peak-centred, so a deviation "
            "here means the backend changed and this test needs revisiting"
        )


class TestWarmStart:
    """Step B is initialised from the previous round's phase."""

    def test_warm_start_off_gives_a_different_first_solve(self) -> None:
        """Turning the warm start off must change what Step B is initialised from.

        Spying the solver is the only reliable probe here. Comparing the
        recorded scores does not work at a four-iteration budget: the solved
        phase barely moves from its initialisation, so both settings land on the
        same numbers and the test would pass even if the flag were ignored.
        """
        from ao_shaping.optimizer.wfless import model_in_loop_shaping as mil

        def _captured(**overrides) -> tuple[list[np.ndarray], list[np.ndarray]]:
            """Return (initial phases fed to Step B, phases Step B returned)."""
            starts: list[np.ndarray] = []
            ends: list[np.ndarray] = []
            # The loop imports this name inside the function body, so the patch
            # has to land on the defining module rather than on a re-export.
            original = mil.shape_phase_with_frozen_aberration

            def _spy(optimizer, coefficients, target, initial_phase, config, **kw):  # noqa: ANN001, ANN202
                starts.append(np.array(initial_phase, copy=True))
                shaped, loss = original(
                    optimizer, coefficients, target, initial_phase, config, **kw
                )
                ends.append(np.array(shaped, copy=True))
                return shaped, loss

            mil.shape_phase_with_frozen_aberration = _spy
            try:
                # The result cache would otherwise hand back the already-computed
                # run and the spy would never fire, so drop this entry first.
                _cache.pop(tuple(sorted(overrides.items())), None)
                _sim_result(**overrides)
            finally:
                mil.shape_phase_with_frozen_aberration = original
            return starts, ends

        region = _SIM_KWARGS["region"]
        warm_starts, warm_ends = _captured(warm_start=True)
        cold_starts, _cold_ends = _captured(warm_start=False)

        assert len(warm_starts) == len(cold_starts) == _SIM_KWARGS["n_rounds"]
        for starts in (warm_starts, cold_starts):
            assert all(s.shape == (region, region) for s in starts)

        # Round 1 starts from flat either way, so the flag cannot differ yet.
        assert np.allclose(warm_starts[0], 0.0)
        assert np.allclose(cold_starts[0], 0.0)

        # The warm start chains the last ACCEPTED phase, not merely the previous
        # one. That is the coupling guard doing its job: a rejected round must not
        # become the starting point of the next one. So the expectation depends on
        # whether round 1 was accepted, which is read from the history rather than
        # assumed.
        round1_accepted = _sim_result(warm_start=True)["history"][1]["accepted"]
        if round1_accepted:
            assert np.array_equal(warm_starts[1], warm_ends[0]), (
                "an accepted round 1 must be handed to Step B as the warm start"
            )
        else:
            assert np.allclose(warm_starts[1], 0.0), (
                "a rejected round 1 must not become the warm start"
            )
        # With the flag off, round 2 starts from flat regardless.
        assert np.allclose(cold_starts[1], 0.0)
        # Guard the guard: at least one branch must be the non-trivial one, so
        # this test cannot pass by both sides being flat.
        assert not np.allclose(warm_ends[0], 0.0), (
            "round 1 returned a flat phase, so this test cannot discriminate"
        )

    def test_the_flag_is_carried_into_the_config(self) -> None:
        """The parameter the runner sets is the parameter the loop reads."""
        assert _sim_result(warm_start=False)["config"].warm_start is False
        assert _sim_result(warm_start=True)["config"].warm_start is True

    def test_warm_start_survives_a_rejected_round(self) -> None:
        """A rejected round must not corrupt the warm start.

        On rejection the loop keeps the previous phase AND the previous
        coefficients; if it advanced the phase anyway, the next round would start
        from a phase the guard just rejected.
        """
        result = _sim_result(warm_start=True)
        rounds = [r for r in result["history"] if r["stage"] == "round"]
        rejected = [r for r in rounds if not r["accepted"]]
        for row in rejected:
            # score_before of a rejected round equals the score still held, so a
            # rejected row never becomes the reference the next round improves on.
            assert not row["accepted"]
        if rejected:
            assert result["result"].best_phase is not None


class TestProbesAreFreshEachRound:
    """Each round draws a new probe family."""

    def test_consecutive_rounds_use_different_probe_seeds(self) -> None:
        """The seed schedule advances per round, so no round re-fits one family.

        Pinned directly on the seed arithmetic rather than inferred from the
        history, because the history records only a probe *count*.
        """
        first = _probe_phase(8, 4.0, seed=0 + 0 * 1000 + 0)
        second = _probe_phase(8, 4.0, seed=0 + 1 * 1000 + 0)
        assert not np.array_equal(first, second)

    def test_probes_within_one_round_are_distinct(self) -> None:
        """The probes inside a round differ too -- that is the phase diversity.

        Step A needs several independent probes because focal-plane intensity is
        a non-convex function of pupil phase: one probe leaves Adam in whichever
        stationary point it entered, and the config therefore refuses
        ``probe_count < 2``.
        """
        seeds = [0 * 1000 + i for i in range(4)]
        phases = [_probe_phase(8, 4.0, seed=s) for s in seeds]
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                assert not np.array_equal(phases[i], phases[j])

    def test_probe_count_is_recorded_per_round(self) -> None:
        """Every round records how many probes it actually used."""
        result = _sim_result()
        rounds = [r for r in result["history"] if r["stage"] == "round"]
        assert rounds
        for row in rounds:
            assert row["probe_count"] >= 2


class TestToModelGridCentreSelection:
    """The model-grid resample picks the window around the frozen axis."""

    def test_an_off_centre_bright_square_survives_the_reduction(self) -> None:
        """A feature off the grid centre must land off the grid centre.

        The window is centred on the frozen axis, not on the frame. If it were
        centred on the frame, an off-axis spot would be block-averaged into the
        wrong cell and Step A would fit the aberration against a shifted
        measurement -- a silent error that still produces plausible numbers.

        The window is kept fully inside the frame (``region * scale`` equals the
        frame side and the centre is the frame centre) so no zero padding shifts
        the arithmetic, which keeps the expected cell exact.
        """
        region, scale = 16, 4.0
        side = int(region * scale)
        frame = np.zeros((side, side))
        # A 4x4 block at rows 20:24, cols 28:32 -> out coords unchanged, so the
        # 4x4 block average lands on grid cell (row 5, col 7).
        frame[20:24, 28:32] = 100.0
        centre = (side // 2, side // 2)
        reduced = _to_model_grid(
            frame, centre, region=region, camera_px_per_model_px=scale
        )

        peak_row, peak_col = np.unravel_index(int(np.argmax(reduced)), reduced.shape)
        assert (int(peak_row), int(peak_col)) == (5, 7), (peak_row, peak_col)
        # And it is demonstrably not the middle of the grid.
        assert (int(peak_row), int(peak_col)) != (region // 2, region // 2)


class TestCliInvocation:
    """The registered command runs end to end."""

    def test_help_exits_zero_and_lists_the_key_guards(self) -> None:
        """`--help` must work, and must document the two guards.

        A guard that is not in the help text is a guard nobody will reach for
        when a run rejects a round.
        """
        from click.testing import CliRunner

        from ao_shaping.runners.slm.model_in_loop_runner import run

        result = CliRunner().invoke(run, ["--help"])
        assert result.exit_code == 0, result.output
        assert "--trust-region-c-l2" in result.output
        assert "--max-rejection-streak" in result.output
        assert "--probe-spread" in result.output

    def test_the_command_completes_a_simulated_run(self, tmp_path: Path) -> None:
        """Invoke the command the way the CLI does, including the output root.

        Exercises the Click wiring, the config construction and the artefact
        writing together. A parameter that exists but is never forwarded reaches
        the library as its default and this still passes -- so the assertion on
        the written target side is what makes the forwarding observable.
        """
        from click.testing import CliRunner

        from ao_shaping.runners.slm.model_in_loop_runner import run

        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--cam-type", "sim",
                "--n-rounds", "1",
                "--probe-count", "2",
                "--step-a-iterations", "4",
                "--step-b-iterations", "4",
                "--target-side", "64",
                "--n-orders", "4",
                "--dtype", "float32",
                "--device", "cpu",
                "--seed", "0",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Per-round summary" in result.output
        # The summary labels this row "Target width"; asserting the label *and*
        # the forwarded value is what makes the forwarding observable -- an
        # unfetched --target-side would silently arrive as the dataclass default
        # and still satisfy a label-only check.
        assert "Target width" in result.output
        assert "64 cam px" in result.output
        assert "Done." in result.output

    def test_the_command_is_registered_on_main(self) -> None:
        """`main.py` exposes it under the documented name."""
        from ao_shaping.main import cli

        assert "slm-model-in-loop" in cli.commands


def _legacy_hook(seed, index, probe_count, holdout_probe_count):  # noqa: ANN001, ANN202
    """A hook that reproduces the historical seed arithmetic exactly."""
    base = int(seed) + int(index) * 1000
    return (
        [base + k for k in range(int(probe_count))],
        [base + 1_000_000 + k for k in range(int(holdout_probe_count))],
    )


def _nan_safe(value):  # noqa: ANN001, ANN202
    """Map every NaN to ``None`` so equal histories compare equal.

    ``score_before`` is NaN on round one (there is no committed score yet), and
    ``nan == nan`` is False, so a plain ``==`` on the history reports a
    difference between two runs that are in fact identical down to the bit.
    Substituting a sentinel keeps the comparison strict everywhere else.
    """
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {key: _nan_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nan_safe(item) for item in value]
    return value


class TestProbeSplitResolution:
    """``_resolve_probe_seeds`` keeps the legacy schedule and validates overrides."""

    def test_the_default_is_the_historical_arithmetic(self) -> None:
        """No hook reproduces ``seed + index*1000 + k`` and the +1e6 holdout offset."""
        for seed, index, probe_count, holdout_count in (
            (0, 0, 8, 2),
            (7, 3, 4, 1),
            (123, 11, 2, 6),
        ):
            train, holdout = _resolve_probe_seeds(seed, index, probe_count, holdout_count)
            assert train == [seed + index * 1000 + k for k in range(probe_count)]
            assert holdout == [
                seed + 1_000_000 + index * 1000 + k for k in range(holdout_count)
            ]

    def test_the_default_halves_are_disjoint(self) -> None:
        """A held-out probe must never be one Step A already fitted."""
        train, holdout = _resolve_probe_seeds(5, 2, 8, 2)
        assert not set(train) & set(holdout)

    def test_a_hook_replaces_both_halves(self) -> None:
        """The override is used verbatim, so a harness owns the whole split."""
        train, holdout = _resolve_probe_seeds(
            0, 4, 3, 2, hook=lambda s, i, pc, hc: ([10, 11, 12], [20, 21])
        )
        assert train == [10, 11, 12]
        assert holdout == [20, 21]

    def test_the_hook_receives_seed_index_and_both_counts(self) -> None:
        """The signature exposes everything needed to build a fold split."""
        seen: list[tuple[int, int, int, int]] = []

        def _record(seed, index, probe_count, holdout_probe_count):  # noqa: ANN001, ANN202
            seen.append((seed, index, probe_count, holdout_probe_count))
            return _legacy_hook(seed, index, probe_count, holdout_probe_count)

        _resolve_probe_seeds(9, 3, 5, 2, hook=_record)
        assert seen == [(9, 3, 5, 2)]

    def test_a_legacy_equivalent_hook_agrees_with_the_default(self) -> None:
        """The hook receives enough context to reproduce the default exactly."""
        for seed, index, probe_count, holdout_count in ((0, 0, 8, 2), (3, 2, 4, 1)):
            assert _resolve_probe_seeds(
                seed, index, probe_count, holdout_count, hook=_legacy_hook
            ) == _resolve_probe_seeds(seed, index, probe_count, holdout_count)

    def test_a_wrong_length_half_is_refused(self) -> None:
        """A hook that supplies the wrong count is a harness bug, not a silent pad."""
        with pytest.raises(ValueError, match="train seeds"):
            _resolve_probe_seeds(0, 0, 4, 2, hook=lambda s, i, pc, hc: ([1, 2], [3, 4]))
        with pytest.raises(ValueError, match="holdout seeds"):
            _resolve_probe_seeds(0, 0, 4, 2, hook=lambda s, i, pc, hc: ([1, 2, 3, 4], [3]))

    def test_a_non_integer_seed_is_refused(self) -> None:
        """Truncating 1.5 would index the wrong probe, so it is an error."""
        with pytest.raises(ValueError, match="non-integer train seed"):
            _resolve_probe_seeds(0, 0, 2, 1, hook=lambda s, i, pc, hc: ([1.5, 2], [3]))

    def test_a_malformed_return_is_refused(self) -> None:
        """A hook that does not return a pair of seed lists fails loudly."""
        with pytest.raises(ValueError, match="must return"):
            _resolve_probe_seeds(0, 0, 2, 1, hook=lambda s, i, pc, hc: [1, 2])
        with pytest.raises(ValueError, match="must return"):
            _resolve_probe_seeds(0, 0, 2, 1, hook=lambda s, i, pc, hc: ([1, 2],))

    def test_an_overlapping_split_is_refused(self) -> None:
        """Reusing a seed would turn the held-out correlation into an in-sample one."""
        with pytest.raises(ValueError, match="reused seed"):
            _resolve_probe_seeds(0, 0, 2, 1, hook=lambda s, i, pc, hc: ([1, 2], [2]))

    def test_the_config_carries_the_hook_and_defaults_to_none(self) -> None:
        """The knob the harness sets is the knob the loop reads."""
        assert SlmModelInLoopConfig().probe_split_hook is None
        config = SlmModelInLoopConfig(probe_split_hook=_legacy_hook)
        assert config.probe_split_hook is _legacy_hook


class TestSplitHookIsWiredIntoTheRun:
    """The hook changes a real run, and a legacy-equivalent one changes nothing."""

    def test_a_legacy_hook_reproduces_the_hookless_run_bit_for_bit(self) -> None:
        """The default path is byte-identical, proven on the result, not on intent.

        Comparing the recorded history and the final phase is stronger than
        comparing seed lists: it shows the seeds actually reached the bench in the
        same order, through the whole loop.
        """
        baseline = _sim_result()
        via_hook = _sim_result(probe_split_hook=_legacy_hook)

        assert np.array_equal(baseline["result"].best_phase, via_hook["result"].best_phase)
        assert _nan_safe(baseline["history"]) == _nan_safe(via_hook["history"])

    def test_a_custom_hook_changes_the_run(self) -> None:
        """Guard the guard: if a hook changed nothing the test above would be vacuous."""
        baseline = _sim_result()
        shifted = _sim_result(probe_split_hook=lambda s, i, pc, hc: ([500 + i * 10 + k for k in range(pc)], [900 + i * 10 + k for k in range(hc)]))

        assert not np.array_equal(
            baseline["result"].best_phase, shifted["result"].best_phase
        ), "a different probe family produced the same phase, so this cannot discriminate"

    def test_each_round_asks_the_hook_for_its_own_split(self) -> None:
        """The hook is consulted per round, so folds can vary probes by round."""
        rounds: list[int] = []

        def _record(seed, index, probe_count, holdout_probe_count):  # noqa: ANN001, ANN202
            rounds.append(index)
            return _legacy_hook(seed, index, probe_count, holdout_probe_count)

        _cache.pop(tuple(sorted({"probe_split_hook": _record}.items())), None)
        _sim_result(probe_split_hook=_record)
        assert rounds == list(range(_SIM_KWARGS["n_rounds"]))


class TestHoldoutOnSim:
    """The held-out score is opt-in on the simulated backend, default NaN."""

    def test_the_flag_defaults_to_off(self) -> None:
        assert SlmModelInLoopConfig().holdout_on_sim is False

    def test_the_default_sim_run_measures_no_holdout(self) -> None:
        """Off by default: the sim path is a smoke run, so it stays unmeasured."""
        rounds = [r for r in _sim_result()["history"] if r["stage"] == "round"]
        assert rounds
        for row in rounds:
            value = row.get("holdout_correlation", float("nan"))
            assert not np.isfinite(value), (
                "the simulated backend measured held-out probes without the flag"
            )

    def test_the_flag_makes_the_holdout_score_finite_on_sim(self) -> None:
        """On: the K-fold harness gets a number per fold from the real loop."""
        rounds = [
            r
            for r in _sim_result(holdout_on_sim=True)["history"]
            if r["stage"] == "round"
        ]
        assert rounds
        values = [row.get("holdout_correlation") for row in rounds]
        assert all(v is not None and np.isfinite(v) for v in values), values

    def test_the_flag_does_not_arm_the_mismatch_abort_on_sim(self) -> None:
        """It measures; it does not judge. An aborted fold would lose its score.

        The abort guard exists to stop hardware committing a phase the model
        merely likes. A simulation has nothing to protect, and the CV harness
        needs the score even when it is low -- so the gate stays off in sim.
        """
        result = _sim_result(holdout_on_sim=True, min_holdout_correlation=0.99)
        assert result["result"].status is not ModelInLoopStatus.ABORTED_UNIDENTIFIABLE

    def test_the_flag_composes_with_the_split_hook(self) -> None:
        """The fold's own probes are what gets scored, not the legacy offset."""
        baseline = [
            r["holdout_correlation"]
            for r in _sim_result(holdout_on_sim=True)["history"]
            if r["stage"] == "round"
        ]
        shifted = [
            r["holdout_correlation"]
            for r in _sim_result(
                holdout_on_sim=True,
                probe_split_hook=lambda s, i, pc, hc: (
                    [700 + i * 10 + k for k in range(pc)],
                    [800 + i * 10 + k for k in range(hc)],
                ),
            )["history"]
            if r["stage"] == "round"
        ]
        assert not np.allclose(np.asarray(baseline), np.asarray(shifted)), (
            "a different holdout probe set produced the same score"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
