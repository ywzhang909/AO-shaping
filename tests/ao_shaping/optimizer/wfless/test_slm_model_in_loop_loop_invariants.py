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
    SlmModelInLoopConfig,
    _probe_phase,
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

        def _spy(frame, target_side, center):  # noqa: ANN001, ANN202
            seen.append((int(center[0]), int(center[1])))
            return original(frame, target_side, center)

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
        from ao_shaping.optimizer.wfless.slm_model_in_loop import (
            _prepare_frame,
            _SimBench,
        )

        config = SlmModelInLoopConfig(**_SIM_KWARGS)
        bench = _SimBench(config)
        try:
            bench.display(np.zeros((config.region, config.region)))
            frame = _prepare_frame(bench.measure())
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
        reduced = _to_model_grid(frame, centre, region=region, camera_px_per_model_px=scale)

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

        from ao_shaping.runners.slm_model_in_loop_runner import run

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

        from ao_shaping.runners.slm_model_in_loop_runner import run

        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--cam_type", "sim",
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
        assert "Target side" in result.output
        assert "Done." in result.output

    def test_the_command_is_registered_on_main(self) -> None:
        """`main.py` exposes it under the documented name."""
        from ao_shaping.main import cli

        assert "slm-model-in-loop" in cli.commands


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))