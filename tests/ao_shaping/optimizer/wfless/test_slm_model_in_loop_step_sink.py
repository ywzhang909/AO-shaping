"""Contract tests for the per-measurement ``step_sink`` debug hook.

WHAT is pinned
--------------
``SlmModelInLoopConfig.step_sink`` exists because ``callback`` cannot do the job
the ``--debug`` per-step artefact dump needs:

* ``callback`` fires **once per round**, at the very END of the round body, and
  three ``break`` statements precede it (rejection-streak abort, early stop,
  display-closed). So it is blind to exactly the rounds a debug dump exists for:
  the one that got rolled back and the one that aborted the run.
* ``callback`` also cannot see the probe loop's per-probe arrays at all -- it is
  called with the round row, which is scalars-only by contract.

``step_sink`` therefore fires once per **MEASUREMENT** and before any ``break``,
with a closed four-value ``stage`` vocabulary and a fixed eight-key payload. The
runner-side consumer keys off those exact strings, so they are asserted literally
rather than derived.

The payload's ``raw_peak`` / ``raw_p999`` / ``raw_mean`` exist to expose detector
saturation, which the *prepared* frame cannot show: ``_prepare_frame`` subtracts
the median and clips at zero, so it destroys the very ceiling those three scalars
are there to reveal. That is checked exactly, by having the test bench keep every
raw ``measure()`` return and matching them positionally against the sink calls --
``optimize_slm_model_in_loop`` contains exactly four ``bench.measure()`` sites, so
the two sequences are 1:1.

HOW the calibration loop is reached offline
-------------------------------------------
``_calibrate_geometry`` short-circuits on ``isinstance(bench, _SimBench)``, so the
``calib-probe`` loop is structurally unreachable from ``cam_type="sim"`` alone.
:class:`_CalibratedSimBench` delegates to a real ``_SimBench`` (so every number
still comes from the canonical simulated physics) while *not* being one, which is
the only way to exercise that branch without a SLM and a CCD. The speckle
correlation solve itself is stubbed -- it is a geometry estimator, not part of the
hook contract, and its verdict would otherwise gate the run on a bench that does
not exist.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from ao_shaping.optimizer.wfless import model_in_loop_shaping as mils
from ao_shaping.optimizer.wfless import slm_model_in_loop as mod
from ao_shaping.optimizer.wfless.slm_gs_refine import _prepare_frame
from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    ModelInLoopStatus,
    SlmModelInLoopConfig,
    _SimBench,
    optimize_slm_model_in_loop,
)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

#: The tiny simulated run, mirroring ``_SIM_KWARGS`` in ``test_slm_model_in_loop``
#: so both files measure the same physics. Both rounds are ACCEPTED on this
#: configuration (verified: status ``completed``, ``probe_count`` stays 2), which is
#: what makes the exact call sequence below well defined -- a single rejection
#: would escalate ``probe_count`` and change the number of ``stepA-probe`` calls.
_TINY_SIM: dict[str, Any] = {
    "n_rounds": 2,
    "seed": 0,
    "device": "cpu",
    "dtype": "float64",
    "target_side": 24,
    "probe_count": 2,
    "n_calibration_probes": 2,
    "step_a_iterations": 3,
    "step_b_iterations": 3,
    "region": 32,
    "cam_type": "sim",
    "cam_size": 128,
    "far_field_padding": 1,
    "panel_span_px": 8.0,
    "settle_discard": 0,
    "n_eval_frames": 1,
    "max_rejection_streak": 3,
    "sim_aberration": {},
}

#: Exactly the four strings the runner-side consumer switches on.
_EXPECTED_STAGES = frozenset({"flat", "calib-probe", "stepA-probe", "stepB-shaped"})

#: Exactly the eight keys the payload contract specifies.
_EXPECTED_PAYLOAD_KEYS = frozenset(
    {
        "phase_rad",
        "frame",
        "raw_peak",
        "raw_p999",
        "raw_mean",
        "round",
        "probe_index",
        "probe_count",
    }
)

#: The literal stage order the hook must produce for one 2-round / 2-probe run.
_EXPECTED_SEQUENCE = [
    "flat",
    "calib-probe",
    "calib-probe",
    "stepA-probe",
    "stepA-probe",
    "stepB-shaped",
    "stepA-probe",
    "stepA-probe",
    "stepB-shaped",
]


class SinkLog:
    """Collects ``(stage, payload)`` pairs instead of persisting anything."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, stage: str, payload: dict[str, Any]) -> None:
        self.calls.append((stage, payload))

    @property
    def stages(self) -> list[str]:
        return [stage for stage, _ in self.calls]

    def payloads_for(self, stage: str) -> list[dict[str, Any]]:
        return [payload for name, payload in self.calls if name == stage]


class _CalibratedSimBench:
    """A ``_SimBench`` that is *not* a ``_SimBench``, so calibration really runs.

    ``_calibrate_geometry`` returns early on ``isinstance(bench, _SimBench)``, which
    is why this delegates rather than subclasses: a subclass would trip the same
    short-circuit and the ``calib-probe`` loop would stay unreachable offline.

    Every measurement still comes from the canonical simulated forward model, so
    the frames the hook reports are the ones a sim run really measured. Each RAW
    return is also retained, which is what lets
    :class:`TestRawScalarsAreReadFromTheRawFrame` recompute the saturation scalars
    from the pre-``_prepare_frame`` data instead of trusting them.
    """

    def __init__(self, config: SlmModelInLoopConfig) -> None:
        self._inner = _SimBench(config)
        self.raw_frames: list[np.ndarray] = []

    def display(self, phase_model: np.ndarray) -> None:
        self._inner.display(phase_model)

    def measure(self) -> np.ndarray:
        raw = self._inner.measure()
        self.raw_frames.append(np.array(raw, dtype=np.float64, copy=True))
        return raw

    def close(self) -> None:
        self._inner.close()

    def true_geometry(self) -> Any:
        return self._inner.true_geometry()


def _use_calibrated_sim_bench(monkeypatch: pytest.MonkeyPatch) -> list[_CalibratedSimBench]:
    """Route the loop through the *hardware* calibration branch, offline.

    ``_open_bench`` and ``calibrate_bench_geometry`` are both resolved at call
    time (the latter through a function-local import), so patching the module
    attributes is enough -- no source edit, no device.
    """
    opened: list[_CalibratedSimBench] = []

    def _fake_open(config: SlmModelInLoopConfig) -> _CalibratedSimBench:
        bench = _CalibratedSimBench(config)
        opened.append(bench)
        return bench

    def _fake_solve(records, **kwargs) -> Any:  # noqa: ANN001, ANN003, ANN202
        # The simulated bench knows its geometry by construction, so the solve is
        # a no-op returning the same answer ``_SimBench.true_geometry`` would.
        # Correlation is 1.0, so the bake-off gate always passes.
        return opened[-1].true_geometry()

    monkeypatch.setattr(mod, "_open_bench", _fake_open)
    monkeypatch.setattr(mils, "calibrate_bench_geometry", _fake_solve)
    return opened


_run_cache: dict[tuple, dict] = {}


def _key(overrides: dict) -> tuple:
    return tuple(sorted((k, repr(v)) for k, v in overrides.items()))


def _memo(key: tuple, build) -> dict:  # noqa: ANN001, ANN202
    if key not in _run_cache:
        _run_cache[key] = build()
    return copy.deepcopy(_run_cache[key])


def _execute(kwargs: dict[str, Any], sink: SinkLog | None, opened: list | None) -> dict:
    """Run once, then memoise and hand back a deep copy.

    Each simulated round costs a couple of seconds and a dozen assertions want the
    same one, so a run is shared -- exactly the trade the existing tiny-sim harness
    makes. The deep copy stops one test from perturbing the next.
    """
    extra = {} if sink is None else {"step_sink": sink}
    result = optimize_slm_model_in_loop(SlmModelInLoopConfig(**kwargs, **extra))
    return {
        "result": result,
        "history": result.recorder.history,
        "sink": sink,
        "raw_frames": None if opened is None else opened[-1].raw_frames,
    }


def _run_calibrated(monkeypatch: pytest.MonkeyPatch, overrides: dict | None = None) -> dict:
    """Tiny sim with the calibration loop live, plus every RAW frame it measured."""
    overrides = dict(overrides or {})

    def _build() -> dict:
        opened = _use_calibrated_sim_bench(monkeypatch)
        return _execute({**_TINY_SIM, **overrides}, SinkLog(), opened)

    return _memo(("calibrated", _key(overrides)), _build)


def _run_plain(overrides: dict | None = None) -> dict:
    """Tiny sim on the stock ``cam_type="sim"`` path, with a sink attached."""
    overrides = dict(overrides or {})
    return _memo(
        ("plain", _key(overrides)),
        lambda: _execute({**_TINY_SIM, **overrides}, SinkLog(), None),
    )


def _run_without_sink(overrides: dict | None = None) -> dict:
    """The identical run with NO hook attached -- the recorder-row baseline."""
    overrides = dict(overrides or {})
    return _memo(
        ("nosink", _key(overrides)),
        lambda: _execute({**_TINY_SIM, **overrides}, None, None),
    )


# ---------------------------------------------------------------------------
# 1. The default
# ---------------------------------------------------------------------------


class TestSinkIsOptional:
    """``step_sink`` defaults to ``None`` and the loop is unaffected by that."""

    def test_default_is_none(self) -> None:
        """A config that never mentions the hook must not carry a callable."""
        assert SlmModelInLoopConfig().step_sink is None

    def test_it_is_not_rendered_in_the_repr(self) -> None:
        """Same as ``callback``: a closure must not leak into provenance logs."""
        assert "step_sink" not in repr(SlmModelInLoopConfig(step_sink=SinkLog()))

    def test_sink_not_called_when_unset(self) -> None:
        """A run with no hook completes, and no call site explodes.

        There is nothing to count, so the observable is the *absence of failure*:
        all four guarded sites are no-ops and the run still reaches a terminal
        status with the full history.
        """
        payload = _run_without_sink()
        assert payload["result"].status == ModelInLoopStatus.COMPLETED
        assert payload["sink"] is None
        assert len(payload["history"]) == _TINY_SIM["n_rounds"] + 1


# ---------------------------------------------------------------------------
# 2-5. The call contract on a run that reaches all four sites
# ---------------------------------------------------------------------------


class TestStepSinkCallContract:
    """One memoised run, asserted from every angle the runner consumer needs."""

    @pytest.fixture(scope="class")
    def run(self) -> dict:
        monkeypatch = pytest.MonkeyPatch()
        try:
            return _run_calibrated(monkeypatch)
        finally:
            monkeypatch.undo()

    def test_sink_fires_once_per_measurement(self, run: dict) -> None:
        """One flat baseline, ``n_calibration_probes``, then per round (probes+1).

        ``1 + 2 + 2 * (2 + 1) == 9``. The count is the point: a hook firing per
        *round* would give 3 and would have nothing to say about the probes.
        """
        assert len(run["sink"].calls) == 9
        assert len(run["sink"].stages) == 1 + 2 + 2 * (2 + 1)

    def test_stage_sequence_is_exact_and_ordered(self, run: dict) -> None:
        """The literal sequence. Nothing about this order is incidental: flat is
        measured before the geometry solve, calibration before the loop, and each
        round measures its probes before it shapes."""
        assert run["sink"].stages == _EXPECTED_SEQUENCE

    def test_stage_vocabulary_is_closed(self, run: dict) -> None:
        """Exactly four names; a typo at one site must fail here, not in a report."""
        assert set(run["sink"].stages) == _EXPECTED_STAGES

    def test_payload_shapes_and_dtypes(self, run: dict) -> None:
        """``phase_rad`` is the MODEL grid, ``frame`` is the 2-D camera frame.

        ``phase_rad`` is the model-grid phase in raw unwrapped radians -- never the
        panel array, which is a different coordinate frame entirely (on this bench
        the axes are swapped and the scales differ by >10x).
        """
        region = int(_TINY_SIM["region"])
        for stage, payload in run["sink"].calls:
            assert set(payload) == _EXPECTED_PAYLOAD_KEYS, stage
            phase = payload["phase_rad"]
            assert phase.shape == (region, region), stage
            assert phase.dtype == np.float64, stage
            assert np.isfinite(phase).all(), stage
            frame = payload["frame"]
            assert frame.ndim == 2, stage
            assert frame.shape[0] == frame.shape[1], stage
            assert np.isfinite(frame).all(), stage

    def test_frame_is_the_prepared_frame_not_the_raw_one(self, run: dict) -> None:
        """Background removal has happened, so nothing may be negative.

        This is what distinguishes ``frame`` from the raw ``bench.measure()``
        return: a raw CCD frame carries symmetric read noise and is roughly half
        negative.
        """
        for stage, payload in run["sink"].calls:
            assert float(payload["frame"].min()) >= 0.0, stage

    def test_probe_and_round_indexing(self, run: dict) -> None:
        """``round`` / ``probe_index`` / ``probe_count`` mean exactly what they say.

        Pre-loop stages are round 0; probe ordinals run 0..n-1 inside their own
        loop; ``probe_count`` is a denominator for a progress readout, so it is
        supplied only where one exists.
        """
        sink: SinkLog = run["sink"]
        assert len(sink.payloads_for("flat")) == 1
        assert len(sink.payloads_for("calib-probe")) == 2
        assert len(sink.payloads_for("stepA-probe")) == 4
        assert len(sink.payloads_for("stepB-shaped")) == 2

        flat = sink.payloads_for("flat")[0]
        assert (flat["round"], flat["probe_index"], flat["probe_count"]) == (0, None, None)

        calib = sink.payloads_for("calib-probe")
        assert [p["probe_index"] for p in calib] == [0, 1]
        assert all(p["round"] == 0 for p in calib)
        assert all(p["probe_count"] == int(_TINY_SIM["n_calibration_probes"]) for p in calib)

        probes = sink.payloads_for("stepA-probe")
        assert [(p["round"], p["probe_index"]) for p in probes] == [
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
        ]
        assert all(p["probe_count"] == int(_TINY_SIM["probe_count"]) for p in probes)

        shaped = sink.payloads_for("stepB-shaped")
        assert [p["round"] for p in shaped] == [1, 2]
        assert all(p["probe_index"] is None and p["probe_count"] is None for p in shaped)

    def test_raw_saturation_scalars_are_finite_and_ordered(self, run: dict) -> None:
        """``raw_peak >= raw_p999 >= raw_mean``, finite, and plain ``float``.

        The ordering holds for any real frame, so it is a cheap check that the
        three scalars came from ONE array rather than being mixed across
        measurements. ``type is float`` is pinned because a ``numpy.float64``
        would survive arithmetic but break a JSON/CSV sink in the consumer.
        """
        for stage, payload in run["sink"].calls:
            for key in ("raw_peak", "raw_p999", "raw_mean"):
                value = payload[key]
                assert type(value) is float, f"{stage}/{key} is {type(value).__name__}"
                assert np.isfinite(value), f"{stage}/{key}"
            assert payload["raw_peak"] >= payload["raw_p999"] >= payload["raw_mean"], stage


# ---------------------------------------------------------------------------
# 5b. The raw scalars, verified against the RAW frames themselves
# ---------------------------------------------------------------------------


class TestRawScalarsAreReadFromTheRawFrame:
    """The saturation scalars must be pre-``_prepare_frame``, checked exactly.

    ``optimize_slm_model_in_loop`` contains exactly four ``bench.measure()`` call
    sites, so the raw frames the test bench collected are positionally 1:1 with
    the sink calls. That makes the requirement checkable rather than merely
    asserted: the reported scalars are recomputed from the untouched raw array.
    """

    @pytest.fixture(scope="class")
    def run(self) -> dict:
        monkeypatch = pytest.MonkeyPatch()
        try:
            return _run_calibrated(monkeypatch)
        finally:
            monkeypatch.undo()

    def test_raw_frames_line_up_one_to_one_with_the_calls(self, run: dict) -> None:
        """Guard for the pairing below: if these diverge the rest is vacuous."""
        assert len(run["raw_frames"]) == len(run["sink"].calls) == 9

    def test_scalars_match_the_raw_frames_exactly(self, run: dict) -> None:
        """``peak``/``p99``/``mean`` recomputed from the raw array must agree."""
        for (stage, payload), raw in zip(run["sink"].calls, run["raw_frames"]):
            assert payload["raw_peak"] == float(raw.max()), stage
            assert payload["raw_p999"] == float(np.percentile(raw, 99)), stage
            assert payload["raw_mean"] == float(raw.mean()), stage

    def test_frame_matches_prepare_frame_of_the_raw_frame(self, run: dict) -> None:
        """``frame`` IS the prepared array, so the two payloads cannot be swapped."""
        for (stage, payload), raw in zip(run["sink"].calls, run["raw_frames"]):
            np.testing.assert_array_equal(payload["frame"], _prepare_frame(raw), err_msg=stage)

    def test_the_scalars_are_not_the_prepared_frames_statistics(self, run: dict) -> None:
        """A real detector has a dark background, so this differs measurably.

        The simulated bench clips at zero, so its raw median is non-negative and
        subtraction moves the mean down by a strictly positive amount. If the
        implementation ever read the scalars from the prepared frame, this fails.
        """
        for (stage, payload), raw in zip(run["sink"].calls, run["raw_frames"]):
            assert float(np.median(raw)) > 0.0, stage
            assert payload["raw_mean"] > float(_prepare_frame(raw).mean()), stage


# ---------------------------------------------------------------------------
# 6. The regression guard: fires before any ``break``
# ---------------------------------------------------------------------------


class TestSinkFiresOnRejectedAndAbortingRounds:
    """The reason ``step_sink`` exists at all, pinned as a behaviour.

    ``callback`` cannot cover this: it is the last statement of the round body,
    and the rejection-streak abort breaks out of the loop before reaching it. So
    the round that got rolled back -- and the round that ended the run -- would
    leave no trace in a ``--debug`` dump built on ``callback``.

    ``acceptance_score_eps`` is set absurdly large so every round from the second
    onward is rejected on the score axis, and ``max_rejection_streak=1`` makes the
    first rejection abort. Neither field is validated, so this is a legal config,
    and it drives the run down the exact ``break`` this test cares about.
    """

    @pytest.fixture(scope="class")
    def run(self) -> dict:
        monkeypatch = pytest.MonkeyPatch()
        try:
            return _run_calibrated(
                monkeypatch,
                {
                    "n_rounds": 3,
                    "acceptance_score_eps": 1e9,
                    "max_rejection_streak": 1,
                },
            )
        finally:
            monkeypatch.undo()

    def test_the_run_really_aborted_early(self, run: dict) -> None:
        """Guards the guard: without this, the assertions below would be vacuous."""
        result = run["result"]
        assert result.status == ModelInLoopStatus.ABORTED_REJECTION_STREAK
        rounds = run["history"][1:]
        assert 1 <= len(rounds) < 3, [row["round"] for row in rounds]
        assert rounds[-1]["accepted"] is False

    def test_stepB_shaped_still_fired_for_the_aborting_round(self, run: dict) -> None:
        """Every recorded round has a shaped dump -- including the aborting one.

        This is the strongest available form of the claim: the number of
        ``stepB-shaped`` calls equals the number of recorded round rows, and the
        last call is the last round. If the hook were ever placed after the
        ``break``, this count would be one short.
        """
        rounds = run["history"][1:]
        shaped = run["sink"].payloads_for("stepB-shaped")
        assert len(shaped) == len(rounds)
        assert shaped[-1]["round"] == rounds[-1]["round"]

    def test_the_rejected_round_is_diagnosable(self, run: dict) -> None:
        """Its phase and frame are in the dump, so the failure can be investigated.

        A rejection means the phase was NOT committed, so without this dump the
        one phase a human most wants to look at is exactly the one the run threw
        away.
        """
        rounds = run["history"][1:]
        shaped = run["sink"].payloads_for("stepB-shaped")
        rejected = [p for p, row in zip(shaped, rounds) if not row["accepted"]]
        assert rejected, "the aborting round produced no shaped dump"
        for payload in rejected:
            assert float(np.abs(payload["phase_rad"]).max()) > 0.0
            assert float(payload["frame"].max()) > 0.0


# ---------------------------------------------------------------------------
# 7. Exceptions propagate
# ---------------------------------------------------------------------------


class TestSinkExceptionPropagates:
    """A raising sink must not be swallowed.

    Silently absorbing it would be the worst outcome available: the run would keep
    spending hardware minutes while writing no artefacts, and the operator would
    only find out from an empty dump directory.
    """

    def test_an_exception_from_inside_the_round_loop_surfaces(self, monkeypatch) -> None:
        seen: list[str] = []

        def _boom(stage: str, payload: dict[str, Any]) -> None:
            seen.append(stage)
            if stage == "stepA-probe":
                raise RuntimeError("sink exploded at stepA-probe")

        _use_calibrated_sim_bench(monkeypatch)
        with pytest.raises(RuntimeError, match="sink exploded at stepA-probe"):
            optimize_slm_model_in_loop(
                SlmModelInLoopConfig(**{**_TINY_SIM, "n_rounds": 1}, step_sink=_boom)
            )
        # It fired for the baseline and both calibration probes first, then died.
        assert seen == ["flat", "calib-probe", "calib-probe", "stepA-probe"]


# ---------------------------------------------------------------------------
# 8. The hook must not perturb the recorder contract
# ---------------------------------------------------------------------------


class TestRecorderContractIsUntouched:
    """Recorder rows stay scalars-only and unchanged when a sink is attached.

    The rows are pinned elsewhere (``test_slm_model_in_loop``'s ``TestRowSchema``)
    to an exact key set with no arrays. Rather than duplicate those literals, this
    asserts the differential form, which is strictly stronger: the same run with
    and without the hook must produce the *same rows*, so nothing was added,
    renamed, dropped, or perturbed numerically. Duplicating a key-set literal into
    a second file would only create a second thing to keep in sync.
    """

    @pytest.fixture(scope="class")
    def with_sink(self) -> dict:
        return _run_plain()

    @pytest.fixture(scope="class")
    def without_sink(self) -> dict:
        return _run_without_sink()

    def test_rows_are_identical_with_and_without_the_hook(
        self, with_sink: dict, without_sink: dict
    ) -> None:
        """Not just the same keys -- the same values, to the bit."""
        assert with_sink["history"] == without_sink["history"]

    def test_rows_stay_flat_scalars_only(self, with_sink: dict) -> None:
        """Every value must still survive a CSV/DataFrame round trip."""
        for row in with_sink["history"]:
            for key, value in row.items():
                assert isinstance(
                    value, (bool, int, float, str)
                ), f"{key} is {type(value).__name__}, not a scalar"

    def test_no_payload_array_leaked_into_a_row(self, with_sink: dict) -> None:
        """Explicitly: no ``_phase`` / ``_img`` / ``_c``, and no payload key.

        Recorder rows are the CSV surface; the hook's arrays belong to the
        per-measurement dump only.
        """
        forbidden = {"_phase", "_img", "_c", "phase_rad", "frame"} | {
            "raw_peak",
            "raw_p999",
            "raw_mean",
            "probe_index",
        }
        for row in with_sink["history"]:
            assert not (set(row) & forbidden), sorted(set(row) & forbidden)

    def test_the_hook_actually_ran_on_that_same_run(self, with_sink: dict) -> None:
        """Otherwise the three tests above would pass on a hook that never fires."""
        assert with_sink["sink"] is not None
        assert len(with_sink["sink"].calls) > 0


# ---------------------------------------------------------------------------
# 9. The copy semantics the sequencing relies on
# ---------------------------------------------------------------------------


class TestPrepareFrameCopySemantics:
    """Why reading the raw scalars after ``_prepare_frame`` is sound -- and pinned.

    ``_prepare_frame`` builds its result with ``np.where`` + ``np.clip(clean -
    median, 0, None)``, so it **returns a copy and does not mutate its input**.
    That is the fact the four call sites' sequencing leans on. They snapshot the
    scalars before calling it anyway -- belt and braces, because the failure mode
    (silently reading post-subtraction data, which still looks like a healthy
    frame) is silent and trivially reintroduced.
    """

    def test_prepare_frame_does_not_mutate_its_input(self) -> None:
        """The behavioural fact, asserted directly rather than inferred."""
        raw = np.array([[-3.0, 4.0], [10.0, 1.0]], dtype=np.float64)
        before = raw.copy()
        _prepare_frame(raw)
        np.testing.assert_array_equal(raw, before)

    def test_the_helper_returns_plain_floats(self) -> None:
        """``numpy.float64`` would serialise in a report but break CSV/JSON sinks."""
        raw = np.array([[0.0, 1.0], [2.0, 8.0]], dtype=np.float64)
        scalars = mod._raw_frame_scalars(raw)
        assert isinstance(scalars, tuple)
        peak, p999, mean = scalars
        assert all(type(v) is float for v in (peak, p999, mean))
        assert peak == 8.0
        assert mean == pytest.approx(2.75)
        assert 2.75 <= p999 <= 8.0

    def test_the_helper_reads_the_frame_it_is_given(self) -> None:
        """A constant shift of the same frame must shift all three scalars by it."""
        raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        pairs = zip(mod._raw_frame_scalars(raw), mod._raw_frame_scalars(raw + 100.0))
        for low, high in pairs:
            assert high == pytest.approx(low + 100.0)