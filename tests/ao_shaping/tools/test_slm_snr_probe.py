"""Offline tests for the device-agnostic SNR probe (no hardware required).

The point of :mod:`ao_shaping.tools.slm.slm_snr_probe` is that the measurement
logic is separable from any device: these tests drive it with **fakes** and
assert the statistics, so the bench-facing code has real coverage even while
the instruments are offline.

The fakes are deliberately *not* mocks of a driver — they only implement
``get_numpy_image`` and ``display_data``, i.e. the duck-type the tool requires.
That is what proves the tool is hardware-independent.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.tools.slm.slm_snr_probe import (
    DEFAULT_DELTAS,
    SIGMA_FLOOR,
    SNR_STRONG,
    SNR_USABLE,
    SnrSweepResult,
    abba_signal,
    measure_noise_floor,
    snr_sweep,
    snr_verdict,
)

# ---------------------------------------------------------------------------
# Fakes: minimal duck-typed devices
# ---------------------------------------------------------------------------


class FakeSLM:
    """Records the grayscale it is handed; never touches hardware."""

    def __init__(self) -> None:
        self.writes: list[np.ndarray] = []

    def display_data(self, gray: np.ndarray) -> None:
        self.writes.append(np.asarray(gray).copy())


class FakeCamera:
    """Gaussian spot whose amplitude responds to the last written phase.

    ``response`` is the caller's hook: given the SLM's latest write it returns
    the frame to serve. The default returns a fixed frame, which is enough for
    the noise-floor statistics.
    """

    def __init__(self, shape: tuple[int, int] = (64, 64), response=None) -> None:
        self.shape = shape
        self._response = response
        self._fixed = self._base_frame()
        self.n_calls = 0

    def _base_frame(self) -> np.ndarray:
        yy, xx = np.mgrid[0 : self.shape[0], 0 : self.shape[1]]
        cy = (self.shape[0] - 1) / 2.0
        cx = (self.shape[1] - 1) / 2.0
        frame = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 4.0**2)))
        return (frame * 100.0).astype(np.float64)

    def get_numpy_image(self, n_sample: int = 1, **_: object) -> np.ndarray:
        self.n_calls += 1
        if self._response is None:
            return self._fixed.copy()
        return np.asarray(self._response(), dtype=np.float64)


# ---------------------------------------------------------------------------
# snr_verdict
# ---------------------------------------------------------------------------


class TestSnrVerdict:
    @pytest.mark.parametrize(
        ("snr", "expected"),
        [
            (5.0, "strong"),
            (SNR_STRONG, "strong"),
            (2.5, "usable"),
            (SNR_USABLE, "usable"),
            (1.0, "unusable"),
            (0.0, "unusable"),
            # A non-finite SNR means the *measurement* failed, not that the
            # bench is excellent — so it must not be reported as strong.
            (float("nan"), "unusable"),
            (float("inf"), "unusable"),
        ],
    )
    def test_thresholds(self, snr: float, expected: str) -> None:
        assert snr_verdict(snr) == expected

    def test_thresholds_are_ordered(self) -> None:
        assert SNR_STRONG > SNR_USABLE > 0.0


# ---------------------------------------------------------------------------
# measure_noise_floor
# ---------------------------------------------------------------------------


class TestMeasureNoiseFloor:
    def test_zero_noise_gives_zero_sigma(self) -> None:
        sigma, values = measure_noise_floor(lambda: 1.25, n_frames=10)

        assert sigma == 0.0
        assert values == [1.25] * 10

    def test_sigma_matches_numpy(self) -> None:
        series = iter([0.1, 0.2, 0.15, 0.05, 0.3])
        sigma, values = measure_noise_floor(lambda: next(series), n_frames=5)

        assert values == [0.1, 0.2, 0.15, 0.05, 0.3]
        assert sigma == pytest.approx(float(np.std(values)))

    def test_drift_raises_sigma(self) -> None:
        counter = iter(range(20))
        drifting, _ = measure_noise_floor(
            lambda: float(next(counter)) * 1e-3, n_frames=20
        )
        steady, _ = measure_noise_floor(lambda: 0.5, n_frames=20)

        assert drifting > steady

    def test_rejects_single_frame(self) -> None:
        with pytest.raises(ValueError, match="n_frames"):
            measure_noise_floor(lambda: 1.0, n_frames=1)

    def test_calls_score_exactly_n_frames_times(self) -> None:
        cam = FakeCamera()
        calls = {"n": 0}

        def score() -> float:
            calls["n"] += 1
            return float(np.asarray(cam.get_numpy_image(1)).sum())

        measure_noise_floor(score, n_frames=7)

        assert calls["n"] == 7


# ---------------------------------------------------------------------------
# abba_signal
# ---------------------------------------------------------------------------


class TestAbbaSignal:
    def test_returns_zero_for_a_constant_score(self) -> None:
        assert abba_signal(
            lambda: 0.5,
            lambda a: None,
            lambda: None,
            0.01,
        ) == 0.0

    def test_detects_a_linear_response(self) -> None:
        state = {"a": 0.0}

        def score() -> float:
            return state["a"] * 2.0

        sig = abba_signal(
            score,
            lambda a: state.__setitem__("a", a),
            lambda: state.__setitem__("a", 0.0),
            0.5,
        )

        # score(+0.5) = +1.0, score(-0.5) = -1.0 -> |1.0 - (-1.0)| = 2.0
        assert sig == pytest.approx(2.0)

    def test_linear_drift_cancels(self) -> None:
        """The ``+ - - +`` palindrome must cancel a locally linear drift.

        With the superficially similar ``+ - + -`` ordering a drift of ``c*t``
        leaves a residual of ``c`` per pair — the same order as the signal —
        so the ordering is not cosmetic.
        """
        state = {"a": 0.0}
        ticks = iter(np.arange(16, dtype=float))  # 4 reps x 4 signs

        def score() -> float:
            return state["a"] * 2.0 + 1e-3 * next(ticks)

        sig = abba_signal(
            score,
            lambda a: state.__setitem__("a", a),
            lambda: state.__setitem__("a", 0.0),
            0.5,
            pairs=4,
        )

        assert sig == pytest.approx(2.0, abs=1e-9)

    def test_abab_ordering_would_not_cancel(self) -> None:
        """Guard the regression: the palindrome is what buys the cancellation."""
        # Reproduce the palindrome's means and an ABAB variant by hand.
        drift = np.arange(4, dtype=float)
        plus_pal, minus_pal = (drift[0], drift[3]), (drift[1], drift[2])
        plus_abab, minus_abab = (drift[0], drift[2]), (drift[1], drift[3])

        pal = abs(np.mean(plus_pal) - np.mean(minus_pal))
        abab = abs(np.mean(plus_abab) - np.mean(minus_abab))

        assert pal == pytest.approx(0.0)
        assert abab == pytest.approx(1.0)
        assert abab > pal

    def test_phase_write_count(self) -> None:
        writes: list[float] = []
        abba_signal(
            lambda: 0.0,
            lambda a: writes.append(a),
            lambda: None,
            0.1,
            pairs=2,
        )

        # 4 sign writes per repetition x 2 repetitions, palindrome order
        assert len(writes) == 8
        assert writes[:4] == [0.1, -0.1, -0.1, 0.1]

    def test_rejects_zero_pairs(self) -> None:
        with pytest.raises(ValueError, match="pairs"):
            abba_signal(lambda: 0.0, lambda a: None, lambda: None, 0.1, pairs=0)


# ---------------------------------------------------------------------------
# snr_sweep (end-to-end on fakes)
# ---------------------------------------------------------------------------


class TestSnrSweep:
    def test_runs_without_any_driver(self) -> None:
        cam, slm = FakeCamera(), FakeSLM()

        result = snr_sweep(
            cam, slm, n_max=4, n_frames=6, pairs=1, deltas=(0.01,)
        )

        assert isinstance(result, SnrSweepResult)
        assert result.sigma >= 0.0
        assert len(result.noise_values) == 6

    def test_writes_flat_phase_to_slm(self) -> None:
        cam, slm = FakeCamera(), FakeSLM()

        snr_sweep(cam, slm, n_max=4, n_frames=5, pairs=1, deltas=(0.01,))

        assert slm.writes, "the tool must drive the SLM through display_data"
        assert all(w.shape == (1200, 1920) for w in slm.writes)

    def test_leaves_bench_flat(self) -> None:
        """The last write must be the reference state, not a perturbation."""
        cam, slm = FakeCamera(), FakeSLM()

        snr_sweep(cam, slm, n_max=4, n_frames=5, pairs=1, deltas=(0.01, 0.1))

        assert np.allclose(slm.writes[-1], 0.0)

    def test_dof_count_excludes_piston(self) -> None:
        cam, slm = FakeCamera(), FakeSLM()

        result = snr_sweep(
            cam, slm, n_max=4, n_frames=5, pairs=1, deltas=(0.01,)
        )

        # n_max=4 -> 15 Zernike terms, minus piston (0,0) -> 14
        assert result.n_dof == 14

    def test_dof_count_scales_with_n_max(self) -> None:
        cam, slm = FakeCamera(), FakeSLM()

        small = snr_sweep(cam, slm, n_max=2, n_frames=5, pairs=1, deltas=(0.01,))
        large = snr_sweep(cam, slm, n_max=6, n_frames=5, pairs=1, deltas=(0.01,))

        assert large.n_dof > small.n_dof

    def test_zero_sigma_yields_zero_snr(self) -> None:
        """A static bench must report "no information", not an infinite SNR."""
        cam, slm = FakeCamera(), FakeSLM()  # fixed frame -> sigma ~ float epsilon

        result = snr_sweep(
            cam, slm, n_max=4, n_frames=5, pairs=1, deltas=(0.01,)
        )

        assert result.sigma <= SIGMA_FLOOR
        assert all(v == 0.0 for v in result.multi_snrs.values())
        assert all(v == "unusable" for v in result.verdicts.values())
        assert result.usable_deltas() == []

    def test_usable_deltas_filters_on_multi_mode(self) -> None:
        """The single probe must not be able to smuggle a delta through."""
        result = SnrSweepResult(
            center=(0.0, 0.0),
            sigma=1.0,
            single_signals={"0.001": 5.0, "0.1": 0.5},
            multi_signals={"0.001": 0.9, "0.1": 4.0},
            n_dof=10,
        )

        assert result.single_snrs["0.001"] == 5.0
        assert result.multi_snrs["0.001"] == 0.9
        assert result.usable_deltas() == [0.1]
        assert result.usable_deltas(SNR_STRONG) == [0.1]

    def test_to_dict_is_json_serialisable(self) -> None:
        import json

        cam, slm = FakeCamera(), FakeSLM()
        result = snr_sweep(
            cam, slm, n_max=3, n_frames=5, pairs=1, deltas=(0.01, 0.1)
        )

        payload = json.loads(json.dumps(result.to_dict()))

        assert set(payload) >= {
            "center", "sigma", "single_snrs", "multi_snrs", "verdicts",
            "n_dof", "thresholds",
        }

    def test_can_disable_single_probe(self) -> None:
        cam, slm = FakeCamera(), FakeSLM()

        result = snr_sweep(
            cam, slm, n_max=4, n_frames=5, pairs=1, deltas=(0.01,),
            measure_single=False,
        )

        assert result.single_signals == {}
        assert result.multi_signals  # multi still ran

    def test_seed_makes_multi_pattern_reproducible(self) -> None:
        cam_a, slm_a = FakeCamera(), FakeSLM()
        cam_b, slm_b = FakeCamera(), FakeSLM()

        a = snr_sweep(cam_a, slm_a, n_max=4, n_frames=5, pairs=1,
                      deltas=(0.01,), seed=7)
        b = snr_sweep(cam_b, slm_b, n_max=4, n_frames=5, pairs=1,
                      deltas=(0.01,), seed=7)

        assert np.array_equal(slm_a.writes, slm_b.writes)
        assert a.multi_signals == b.multi_signals

    def test_rejects_nonpositive_deltas(self) -> None:
        with pytest.raises(ValueError, match="deltas"):
            snr_sweep(FakeCamera(), FakeSLM(), n_max=4, deltas=(0.0, -1.0))

    def test_rejects_negative_n_max(self) -> None:
        with pytest.raises(ValueError, match="n_max"):
            snr_sweep(FakeCamera(), FakeSLM(), n_max=-1)

    def test_default_deltas_span_the_measured_range(self) -> None:
        assert min(DEFAULT_DELTAS) < 0.001
        assert max(DEFAULT_DELTAS) >= 0.1
