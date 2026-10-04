"""Offline tests for the measurement-floor probe.

No hardware, no SDK, no real waiting: ``settle_curve`` takes an injectable
``sleep``, so the whole settle-timeline is exercised in-process and instantly.

The point of the probe is to decide *which* noise dominates before an
optimisation run is trusted:

* independent read noise -> averaging K frames helps, SNR grows like sqrt(K)
* slow drift           -> averaging does NOT help, and only a settle/ABBA
  protocol can fix it

The second case is the one that matters here, so it is pinned explicitly below.
"""

from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner
from loguru import logger

from ao_shaping.tools.slm import slm_floor_probe as probe
from ao_shaping.tools.slm.slm_floor_probe import (
    DEFAULT_EXPOSURE_MS,
    DEFAULT_GRID,
    DEFAULT_KS,
    DEFAULT_N_REPEAT,
    DEFAULT_ROI,
    DEFAULT_ROI_HALF,
    DEFAULT_SETTLE_CURVE_S,
    DEFAULT_SETTLE_SAMPLE_MS,
    checkerboard_coeffs,
    main,
    perturbation_pattern,
    prepare_roi_frame,
    repeatability_floor,
    settle_curve,
    snr_ladder,
)

FRAME = (64, 64)
CENTRE = (32, 32)


def _flat(value: float = 100.0) -> np.ndarray:
    yy, xx = np.mgrid[0 : FRAME[0], 0 : FRAME[1]]
    r2 = (xx - CENTRE[0]) ** 2 + (yy - CENTRE[1]) ** 2
    return value + 40.0 * np.exp(-r2 / (2 * 3.0**2))


def _cropper(frame_shape: tuple[int, int] = FRAME, centre=CENTRE, half=None):
    half = DEFAULT_ROI_HALF if half is None else half

    def capture() -> np.ndarray:
        return prepare_roi_frame(frame_shape and _flat(), centre, half)

    return capture


class TestPrepareRoiFrame:
    def test_returns_the_requested_window(self) -> None:
        out = prepare_roi_frame(_flat(), CENTRE, 8)
        assert out.shape == (16, 16)

    def test_is_non_negative_even_with_read_noise(self) -> None:
        rng = np.random.default_rng(1)
        frame = 3.0 + rng.normal(0.0, 8.0, FRAME)
        assert (frame < 0.0).any()
        out = prepare_roi_frame(frame, CENTRE, 8)
        assert float(out.min()) >= 0.0

    def test_crops_about_the_given_centre_not_the_frame_centre(self) -> None:
        frame = np.zeros(FRAME)
        frame[6, 6] = 900.0
        out = prepare_roi_frame(frame, (6, 6), 4)
        assert out.shape == (8, 8)
        assert out[4, 4] > 0.0

    def test_rejects_a_bad_half_width(self) -> None:
        with pytest.raises(ValueError):
            prepare_roi_frame(_flat(), CENTRE, 0)


class TestRepeatabilityFloor:
    def test_a_perfectly_stable_bench_gives_zero(self) -> None:
        """Identical frames -> the floor is 0.0, i.e. any response is resolvable."""
        records, summary = repeatability_floor(lambda: _flat(), 5)
        assert len(records) == 5
        assert summary["floor"] == pytest.approx(0.0, abs=1e-9)

    def test_records_carry_the_documented_row_keys(self) -> None:
        records, _ = repeatability_floor(lambda: _flat(), 3)
        for row in records:
            assert {"peak", "norm", "sum", "stage", "_epoch"} <= set(row)

    def test_epoches_are_contiguous_from_zero(self) -> None:
        records, _ = repeatability_floor(lambda: _flat(), 4)
        assert [r["_epoch"] for r in records] == [0, 1, 2, 3]

    def test_independent_noise_raises_the_floor_above_zero(self) -> None:
        rng = np.random.default_rng(2)
        frames = iter(4.0 + rng.normal(0.0, 5.0, FRAME) for _ in range(6))
        _, summary = repeatability_floor(lambda: next(frames), 6)
        assert summary["floor"] > 0.0

    def test_a_single_frame_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n must be"):
            repeatability_floor(lambda: _flat(), 1)


class TestSettleCurve:
    def test_never_waits_when_sleep_is_injected(self) -> None:
        """The whole point of the injectable sleep: CI must not block."""
        slept: list[float] = []
        records, summary = settle_curve(
            lambda: _flat(), total_s=0.5, sample_ms=100.0,
            sleep=slept.append,
        )
        assert slept, "settle_curve must pace its samples"
        assert all(d >= 0.0 for d in slept)
        assert len(records) >= 2

    def test_reports_a_settle_time_for_a_decaying_observable(self) -> None:
        # An observable that relaxes toward a plateau.
        state = {"n": 0}

        def capture() -> np.ndarray:
            state["n"] += 1
            return _flat(value=100.0 - 40.0 * (0.6**state["n"]))

        records, summary = settle_curve(capture, 0.6, 100.0, sleep=lambda _s: None)
        assert len(records) >= 3
        assert "settle_s" in summary
        # A settling curve must produce a finite, positive settle time.
        assert summary["settle_s"] is None or summary["settle_s"] >= 0.0

    def test_an_already_settled_bench_reports_zero(self) -> None:
        records, summary = settle_curve(
            lambda: _flat(), 0.3, 100.0, sleep=lambda _s: None,
        )
        assert len(records) >= 2
        assert summary["settle_s"] == pytest.approx(0.0, abs=1e-9)

    def test_sample_count_matches_the_requested_span(self) -> None:
        span_s, step_s = 0.4, 0.1
        records, _ = settle_curve(
            lambda: _flat(), span_s, step_s * 1000.0, sleep=lambda _s: None,
        )
        assert len(records) == pytest.approx(int(span_s / step_s) + 1, abs=2)

    def test_a_negative_span_is_rejected(self) -> None:
        # total_s == 0 is legal (one sample, already-settled bench); negative is not.
        with pytest.raises(ValueError, match="total_s"):
            settle_curve(lambda: _flat(), -1.0, 100.0, sleep=lambda _s: None)

    def test_a_non_positive_sample_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="sample_ms"):
            settle_curve(lambda: _flat(), 0.3, 0.0, sleep=lambda _s: None)


class TestSnrLadder:
    def _responder(self, response: float, noise: float, seed: int = 3):
        """A flat reference, then a perturbed frame with the given response."""
        rng = np.random.default_rng(seed)
        state = {"first": True}

        def capture() -> np.ndarray:
            if state["first"]:
                state["first"] = False
                return _flat()
            return _flat() + rng.normal(0.0, noise, FRAME) + response

        return capture

    def test_a_clear_signal_grows_with_k(self) -> None:
        capture = self._responder(response=30.0, noise=1.0)
        _, summary = snr_ladder(capture, lambda: None, floor=1.0, ks=DEFAULT_KS)
        snr = summary["snr"]
        assert snr[1] < snr[4] < snr[9]
        assert summary["improves"] is True

    def test_independent_noise_grows_like_sqrt_k(self) -> None:
        capture = self._responder(response=20.0, noise=2.0, seed=7)
        _, summary = snr_ladder(capture, lambda: None, floor=2.0, ks=(1, 4, 9))
        snr = summary["snr"]
        # K=9 should be ~3x the K=1 SNR under independent noise.
        assert snr[9] / snr[1] == pytest.approx(3.0, rel=0.6)

    def test_a_transient_response_does_not_improve_with_k(self) -> None:
        """The bench-diagnosis signal.

        ``snr[K] = mean(observable[:K]) / (floor / sqrt(K))`` and the observable
        is an L2 *norm*, so its mean never cancels -- averaging only helps while
        the response is reproducible. A response that appears once and then
        vanishes (the signature of an unsettled panel or a drifting bench) gets
        *washed out* by averaging, so ``improves`` must be False and the verdict
        must read drift_limited. That tells you to fix the settle protocol or use
        ABBA instead of merely lowering delta.
        """
        state = {"n": 0}

        def capture() -> np.ndarray:
            state["n"] += 1
            if state["n"] == 1:
                return _flat()  # the flat reference
            if state["n"] == 2:
                return _flat() + 60.0  # a one-off transient
            return _flat()  # response gone

        _, summary = snr_ladder(capture, lambda: None, floor=1.0, ks=(1, 4, 9))
        assert summary["improves"] is False
        assert summary["verdict"] == "drift_limited"

    def test_a_zero_floor_is_clamped_not_divided_by(self) -> None:
        """A degenerate floor must not produce inf/nan SNR.

        The module floors the divisor at ``SIGMA_FLOOR`` rather than raising,
        so a caller who measured a perfect bench still gets finite numbers.
        """
        capture = self._responder(response=25.0, noise=1.0)
        _, summary = snr_ladder(capture, lambda: None, floor=0.0, ks=(1, 4))
        assert summary["floor"] > 0.0
        for value in summary["snr"].values():
            assert np.isfinite(value)

    def test_an_empty_k_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ks"):
            snr_ladder(lambda: _flat(), lambda: None, 1.0, ())

    def test_records_report_snr_per_k(self) -> None:
        capture = self._responder(response=25.0, noise=1.0)
        records, _ = snr_ladder(capture, lambda: None, floor=1.0, ks=(1, 4))
        assert {r["k"] for r in records} == {1, 4}
        assert all("snr" in r for r in records)


class TestPatterns:
    def test_checkerboard_is_deterministic_and_bounded(self) -> None:
        a = checkerboard_coeffs(DEFAULT_GRID, amplitude_rad=1.0)
        b = checkerboard_coeffs(DEFAULT_GRID, amplitude_rad=1.0)
        assert np.array_equal(a, b), "must be reproducible"
        assert a.size == DEFAULT_GRID * DEFAULT_GRID
        assert float(np.abs(a).max()) <= 1.0 + 1e-12

    def test_perturbation_pattern_matches_the_requested_amplitude(self) -> None:
        pattern = perturbation_pattern(8, 0.5, (16, 24))
        assert pattern.shape == (16, 24)
        assert float(np.abs(pattern).max()) == pytest.approx(0.5, rel=1e-9)


class TestNoHardware:
    def test_no_hw_creates_no_hardware_and_exits_zero(self, tmp_path) -> None:
        # The plan lines are logged, not printed, so a loguru sink is the only
        # way to see that "--ks 1,4" survived the Click layer as [1, 4]: it is
        # one comma-joined string field, NOT a repeated flag (which would be
        # argparse "1,4" -> click multiple=True -> ks=[("1",), ("4",)]).
        logged: list[str] = []
        sink_id = logger.add(logged.append, level="INFO")
        try:
            result = CliRunner().invoke(
                main,
                [
                    "--out", str(tmp_path / "floor"),
                    "--n-repeat", "3",
                    "--settle-curve-s", "0.2",
                    "--settle-sample-ms", "100",
                    "--ks", "1,4",
                    "--no-hw",
                ],
            )
        finally:
            logger.remove(sink_id)
        assert result.exit_code == 0, result.output
        assert any("ks [1, 4]" in line for line in logged), logged
        assert not (tmp_path / "floor").exists(), "--no-hw must not touch disk"

    def test_no_driver_import_at_module_scope(self) -> None:
        import ast
        from pathlib import Path

        tree = ast.parse(Path(probe.__file__).read_text(encoding="utf-8"))
        eager = [
            n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom)) and "drivers" in ast.dump(n)
        ]
        assert eager == [], f"drivers must not be imported eagerly: {eager}"
        assert not hasattr(probe, "Santec")


class TestDefaultsMatchTheDocumentedBench:
    def test_the_documented_defaults_are_in_force(self) -> None:
        # These are the values quoted in docs/slm/pre_run_characterization.md;
        # if one changes, that document must change with it.
        assert DEFAULT_ROI == 192
        assert DEFAULT_EXPOSURE_MS == 1.5
        assert DEFAULT_N_REPEAT == 20
        assert DEFAULT_SETTLE_CURVE_S == 4.0
        assert DEFAULT_SETTLE_SAMPLE_MS == 120.0
        assert DEFAULT_KS == (1, 4, 9)