from __future__ import annotations

import pytest

from ao_shaping.optimizer.wfless.bayes_hyperparam import (
    SearchParam,
    bayesian_search_hyperparams,
)


def test_bayesian_search_custom_objective() -> None:
    """Custom objective should be maximized by Bayesian search wrapper."""

    def objective(params: dict[str, float]) -> float:
        x = params["x"]
        y = params["y"]
        return -((x - 0.4) ** 2 + (y - 0.7) ** 2)

    result = bayesian_search_hyperparams(
        mode="sim",
        search_params=[
            SearchParam("x", 0.0, 1.0),
            SearchParam("y", 0.0, 1.0),
        ],
        n_calls=12,
        random_state=7,
        objective_fn=objective,
    )

    assert result.best_score > -0.05
    assert abs(result.best_params["x"] - 0.4) < 0.2
    assert abs(result.best_params["y"] - 0.7) < 0.2
    assert len(result.history) == 12


def test_bayesian_search_requires_params() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        bayesian_search_hyperparams(mode="sim", search_params=[])
