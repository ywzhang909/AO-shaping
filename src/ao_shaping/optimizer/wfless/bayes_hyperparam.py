from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from collections.abc import Callable

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from ao_shaping.optimizer.wfless.pib import optimize_pib
from ao_shaping.optimizer.wfless.sim_spgd import optimize_spgd, optimize_spgd_zernike


Mode = Literal["physical", "sim"]


@dataclass
class SearchParam:
    name: str
    low: float
    high: float


@dataclass
class BayesSearchResult:
    best_params: dict[str, float]
    best_score: float
    history: list[dict[str, float]]
    raw_result: Any


def _build_dimensions(search_params: list[SearchParam]) -> list[Real]:
    return [Real(item.low, item.high, name=item.name) for item in search_params]


def _extract_metric(recorder: Any, metric_key: str = "pib") -> float:
    best_iter, _ = recorder.get_best_iter()
    return float(best_iter[metric_key])


def _run_physical(params: dict[str, float], fixed_kwargs: dict[str, Any]) -> float:
    recorder = optimize_pib(
        center=fixed_kwargs.get("center", None),
        epochs=fixed_kwargs.get("epochs", 20),
        lr=float(params.get("lr", 1.0)),
        delta=float(params.get("delta", 1.0)),
        exposure_time_ms=fixed_kwargs.get("exposure_time_ms", 80),
        cam_id=fixed_kwargs.get("cam_id", 0),
        show=False,
        init_v=fixed_kwargs.get("init_v", []),
    )
    return _extract_metric(recorder, fixed_kwargs.get("metric_key", "pib"))


def _run_sim(params: dict[str, float], fixed_kwargs: dict[str, Any]) -> float:
    sim_type = fixed_kwargs.get("sim_type", "spgd_zernike")
    common = {
        "epochs": fixed_kwargs.get("epochs", 40),
        "r_bucket": fixed_kwargs.get("r_bucket", 0),
        "delta": float(params.get("delta", fixed_kwargs.get("delta", 0.08))),
        "gamma": float(params.get("gamma", fixed_kwargs.get("gamma", 1e-3))),
        "n_grid": fixed_kwargs.get("n_grid", 64),
        "Cn2": fixed_kwargs.get("Cn2", 1e-9),
        "optimizer_type": fixed_kwargs.get("optimizer_type", "spgd"),
        "seed": fixed_kwargs.get("seed", 42),
        "use_momentum": fixed_kwargs.get("use_momentum", False),
    }
    if "beta1" in params:
        common["beta1"] = float(params["beta1"])
    if "beta2" in params:
        common["beta2"] = float(params["beta2"])
    if "beta3" in params:
        common["beta3"] = float(params["beta3"])

    if sim_type == "spgd_zernike":
        recorder = optimize_spgd_zernike(n_max=fixed_kwargs.get("n_max", 5), **common)
    else:
        recorder = optimize_spgd(dm_actuators=fixed_kwargs.get("dm_actuators", 8), **common)
    return _extract_metric(recorder, fixed_kwargs.get("metric_key", "pib"))


def bayesian_search_hyperparams(
    mode: Mode,
    search_params: list[SearchParam],
    n_calls: int = 20,
    random_state: int = 42,
    objective_fn: Callable[[dict[str, float]], float] | None = None,
    **fixed_kwargs: Any,
) -> BayesSearchResult:
    """Run Bayesian hyperparameter search for physical or simulation optimizer flows.

    If `objective_fn` is provided, it is used directly and should return a score to maximize.
    Otherwise built-in runners for physical (`optimize_pib`) and simulation
    (`optimize_spgd` / `optimize_spgd_zernike`) are used.
    """

    if not search_params:
        raise ValueError("search_params must not be empty")

    dimensions = _build_dimensions(search_params)
    history: list[dict[str, float]] = []

    @use_named_args(dimensions)
    def objective(**kwargs: float) -> float:
        params = {k: float(v) for k, v in kwargs.items()}
        if objective_fn is not None:
            score = float(objective_fn(params))
        elif mode == "physical":
            score = _run_physical(params, fixed_kwargs)
        elif mode == "sim":
            score = _run_sim(params, fixed_kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        payload = dict(params)
        payload["score"] = score
        history.append(payload)
        return -score

    result = gp_minimize(
        objective,
        dimensions=dimensions,
        n_calls=n_calls,
        n_initial_points=min(10, n_calls),
        random_state=random_state,
    )

    best_params = {
        item.name: float(value)
        for item, value in zip(search_params, result.x)
    }

    return BayesSearchResult(
        best_params=best_params,
        best_score=float(-result.fun),
        history=history,
        raw_result=result,
    )
