"""Characterization net for ``optimizer.wf.greedy_zernike.optimizer_greedy``.

The greedy Zernike optimizer had no unit-level test: the only existing coverage
is ``tests/ao_shaping/runners/test_greedy_zernike_runner.py``, which mocks
``optimizer_greedy`` out entirely.  So nothing pinned what the optimizer itself
records, how it selects candidates or which coefficients it restores.

Everything here drives the **real** optimizer against fake devices (a fake
``ZernikeSLM`` and a fake WFS whose RMS is a known function of the coefficients),
so the assertions are about observable behaviour:

* the epoch-0 ``init`` record and its ``_init_rms`` list,
* one ``search`` record per epoch, each carrying the candidate RMS list,
* the greedy invariant (RMS never increases; the recorded best tracks the min),
* threshold / patience early stop,
* the coefficients restored on exit,
* the black-box heuristic arm returning the recorder instead of the greedy arm.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

import ao_shaping.optimizer.wf.greedy_zernike as mod
from ao_shaping.utils.io.recorder_keys import RECORD_COEFFS, RECORD_EPOCH, RECORD_PHASE


class _Bench:
    """Shared state between the fake SLM and the fake WFS."""

    def __init__(self) -> None:
        self.sent: list[np.ndarray] = []
        self.wavefront_calls = 0
        self.take_image_calls = 0

    @property
    def last(self) -> np.ndarray:
        assert self.sent, "optimizer never sent coefficients to the SLM"
        return self.sent[-1]

    def rms(self) -> float:
        coeffs = self.last.astype(np.float64)
        # Quadratic cost: strictly minimised at the all-zero coefficient vector,
        # so a greedy step that scales the vector down always looks better.
        return float(np.sqrt(np.mean(coeffs**2)))


class _FakeSlm:
    def __init__(self, bench: _Bench) -> None:
        self._bench = bench

    def __enter__(self) -> _FakeSlm:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def send_zernike(self, coeffs: np.ndarray) -> np.ndarray:
        arr = np.asarray(coeffs, dtype=np.float64).copy()
        self._bench.sent.append(arr)
        return np.zeros((2, 2))


class _FakeWfs:
    def __init__(self, bench: _Bench) -> None:
        self._bench = bench

    def __enter__(self) -> _FakeWfs:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def take_image(self, n: int) -> None:
        self._bench.take_image_calls += 1

    def get_wavefront(self, cancel_tile: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        self._bench.wavefront_calls += 1
        wf = np.zeros((8, 8), dtype=np.float64)
        return wf, {"rms": self._bench.rms(), "cancel_tile": cancel_tile}


@pytest.fixture(name="bench")
def fixture_bench(monkeypatch: pytest.MonkeyPatch) -> _Bench:
    bench = _Bench()

    @contextmanager
    def _slm(*args: object, **kwargs: object):
        yield _FakeSlm(bench)

    @contextmanager
    def _wfs(*args: object, **kwargs: object):
        yield _FakeWfs(bench)

    monkeypatch.setattr(mod, "ZernikeSLM", _slm)
    monkeypatch.setattr(mod, "resolve_wfs", _wfs)
    np.random.seed(1234)
    return bench


class TestGreedyPhaseOne:
    def test_epoch_zero_record_lists_every_init_position(self, bench: _Bench) -> None:
        recorder = mod.optimizer_greedy(epochs=2, n_init=3, n_directions=1, n_max=2)

        first = recorder.history[0]
        assert first[RECORD_PHASE] == "init"
        assert first[RECORD_EPOCH] == 0
        assert len(first["_init_rms"]) == 3
        assert first["rms"] == min(first["_init_rms"])

    def test_best_init_position_is_the_one_that_is_measured(self, bench: _Bench) -> None:
        recorder = mod.optimizer_greedy(epochs=1, n_init=4, n_directions=1, n_max=2)

        first = recorder.history[0]
        best_index = int(np.argmin(first["_init_rms"]))
        # init positions are drawn in order, so ``sent[best_index]`` is that one
        np.testing.assert_array_equal(first[RECORD_COEFFS], bench.sent[best_index])


class TestGreedySearch:
    def test_one_search_record_per_epoch_with_candidates(self, bench: _Bench) -> None:
        recorder = mod.optimizer_greedy(
            epochs=5,
            n_init=2,
            n_directions=3,
            n_max=2,
            early_stop_threshold=0.0,
        )

        search = [r for r in recorder.history if r[RECORD_PHASE] == "search"]
        assert len(search) == 5
        assert [r[RECORD_EPOCH] for r in search] == [1, 2, 3, 4, 5]
        for row in search:
            assert len(row["_candidate_rmss"]) == 4  # current + 3 directions
            assert row["rms"] == min(row["_candidate_rmss"])

    def test_greedy_rms_never_increases(self, bench: _Bench) -> None:
        recorder = mod.optimizer_greedy(
            epochs=6,
            n_init=2,
            n_directions=2,
            n_max=2,
            early_stop_threshold=0.0,
        )

        rmss = [r["rms"] for r in recorder.history]
        assert all(later <= earlier for earlier, later in zip(rmss, rmss[1:]))

    def test_best_coefficients_are_restored_on_exit(self, bench: _Bench) -> None:
        recorder = mod.optimizer_greedy(
            epochs=4,
            n_init=2,
            n_directions=2,
            n_max=2,
            early_stop_threshold=0.0,
        )

        expected_best, _ = recorder.get_best_target(RECORD_COEFFS)
        np.testing.assert_array_equal(bench.last, expected_best)


class TestEarlyStop:
    def test_threshold_stops_after_the_first_epoch(self, bench: _Bench) -> None:
        recorder = mod.optimizer_greedy(
            epochs=100,
            n_init=2,
            n_directions=1,
            n_max=2,
            early_stop_threshold=1e9,
        )

        search = [r for r in recorder.history if r[RECORD_PHASE] == "search"]
        assert len(search) == 1

    def test_patience_stops_a_stalled_search(self, bench: _Bench) -> None:
        # The all-zero coefficient vector is the optimum of the fake bench, so a
        # search started at the origin can never improve: patience must fire.
        recorder = mod.optimizer_greedy(
            epochs=200,
            n_init=1,
            n_directions=1,
            n_max=2,
            init_z=[0.0, 0.0, 0.0],
            early_stop_threshold=0.0,
        )

        search = [r for r in recorder.history if r[RECORD_PHASE] == "search"]
        assert 1 <= len(search) < 200
        assert max(r["_no_improvement"] for r in search) >= 50


class TestHeuristicArm:
    def test_heuristic_algorithm_returns_recorder_and_restores_best(self, bench: _Bench) -> None:
        recorder = mod.optimizer_greedy(
            epochs=12,
            n_init=2,
            n_directions=1,
            n_max=2,
            algorithm="hc",
        )

        phases = {r[RECORD_PHASE] for r in recorder.history}
        assert "heuristic" in phases
        assert "search" not in phases
        expected_best, _ = recorder.get_best_target(RECORD_COEFFS)
        np.testing.assert_array_equal(bench.last, expected_best)

    def test_unknown_algorithm_is_rejected_before_touching_devices(self, bench: _Bench) -> None:
        with pytest.raises(ValueError, match="algorithm must be one of"):
            mod.optimizer_greedy(epochs=1, algorithm="not-an-algorithm")

        assert bench.sent == []
