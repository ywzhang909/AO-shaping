"""Offline tests for the shared heuristic-search driver.

No hardware: the objective is a synthetic numpy function.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.algorithm.heuristic.search import (
    HEURISTIC_ALGORITHM_MAP,
    SearchAborted,
    create_heuristic,
    heuristic_algorithm_choices,
    run_heuristic_search,
)


def test_algorithm_choices_order_and_content():
    choices = heuristic_algorithm_choices()
    assert choices[0] == "spgd"
    assert set(choices[1:]) == {"ga", "pso", "sa", "hc", "rs", "cem", "de"}
    assert set(HEURISTIC_ALGORITHM_MAP) == set(choices[1:])
    assert heuristic_algorithm_choices(include_spgd=False)[0] == "ga"


@pytest.mark.parametrize("algorithm", sorted(HEURISTIC_ALGORITHM_MAP))
def test_all_algorithms_minimise(algorithm):
    def objective(x: np.ndarray) -> float:
        return float(np.sum((x - 0.5) ** 2))

    res = run_heuristic_search(
        algorithm,
        objective,
        dim=3,
        iterations=6,
        bounds=(-2.0, 2.0),
        x0=np.zeros(3),
        seed=0,
        pop_size=8,
    )

    assert res.best_x.shape == (3,)
    assert np.isfinite(res.best_value)
    assert np.all(np.abs(res.best_x) <= 2.0 + 1e-9)
    assert res.evaluations == len(res.history) >= 1
    assert res.best_value == pytest.approx(min(res.history))


@pytest.mark.parametrize("algorithm", ["ga", "pso", "cem", "de", "hc"])
def test_maximize_ascends_a_concave_objective(algorithm):
    def objective(x: np.ndarray) -> float:
        return float(-np.sum((x - 0.5) ** 2))

    res = run_heuristic_search(
        algorithm,
        objective,
        dim=2,
        iterations=40,
        bounds=(-2.0, 2.0),
        x0=np.zeros(2),
        maximize=True,
        seed=1,
        pop_size=20,
    )

    assert res.best_value == pytest.approx(max(res.history))
    assert res.best_value > objective(np.zeros(2))  # improved on the start point


def test_on_evaluate_called_for_every_evaluation_with_raw_values():
    seen: list[tuple[int, float]] = []

    def objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    res = run_heuristic_search(
        "rs",
        objective,
        dim=2,
        iterations=5,
        bounds=(-1.0, 1.0),
        seed=3,
        on_evaluate=lambda x, v, i: seen.append((i, v)),
    )

    assert [i for i, _ in seen] == list(range(1, res.evaluations + 1))
    assert [v for _, v in seen] == res.history


def test_should_stop_aborts_and_returns_best_so_far():
    calls = {"n": 0}
    history: list[float] = []

    def objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    res = run_heuristic_search(
        "rs",
        objective,
        dim=2,
        iterations=1000,
        bounds=(-1.0, 1.0),
        seed=4,
        on_evaluate=lambda x, v, i: history.append(v),
        # Abort once the predicate has been consulted 8 times; the evaluation that
        # triggered the stop is still recorded (best-so-far includes it).
        should_stop=lambda: calls.__setitem__("n", calls["n"] + 1) or calls["n"] > 7,
    )

    assert res.evaluations == calls["n"] == 8
    assert res.evaluations < 1000  # aborted early, did not run to completion
    assert len(history) == res.evaluations
    assert res.best_value == pytest.approx(min(history))


def test_objective_only_receives_bounds_clipped_vectors():
    out_of_bounds = {"seen": False}

    def objective(x: np.ndarray) -> float:
        if np.any(x < -1.0) or np.any(x > 1.0):
            out_of_bounds["seen"] = True
        return float(np.sum(x**2))

    run_heuristic_search(
        "ga",
        objective,
        dim=4,
        iterations=5,
        bounds=(-1.0, 1.0),
        seed=5,
        pop_size=10,
    )

    assert out_of_bounds["seen"] is False


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError, match="Unknown heuristic algorithm"):
        run_heuristic_search(
            "does-not-exist",
            lambda x: 0.0,
            dim=2,
            iterations=1,
            bounds=(-1.0, 1.0),
        )


def test_create_heuristic_population_size_is_accepted():
    opt = create_heuristic(
        HEURISTIC_ALGORITHM_MAP["ga"], dim=3, iterations=2, bounds=(-1, 1), pop_size=7
    )
    assert opt.params.pop_size == 7


def test_search_aborted_is_a_runtime_error():
    assert issubclass(SearchAborted, RuntimeError)
