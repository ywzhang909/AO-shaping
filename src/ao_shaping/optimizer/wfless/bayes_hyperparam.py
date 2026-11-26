from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from collections.abc import Callable

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import numpy as np
from loguru import logger

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
        for item, value in zip(search_params, result.x, strict=True)
    }

    return BayesSearchResult(
        best_params=best_params,
        best_score=float(-result.fun),
        history=history,
        raw_result=result,
    )


# ---------------------------------------------------------------------------
# lr/delta 选型（#138 第 2 项，第 187 批）：原 ``bayes_opt_lr_delta.py`` 并入。
# 评分统一为「越大越好」的 PIB；原 ``objective_function`` 的「返回 -PIB / 失败
# 返回 1e6」是最小化方向的同值写法（送入 ``gp_minimize`` 的目标逐点相同：
# 引擎取 ``-score`` 后失败点仍是 +1e6，``n_initial_points`` 在 skopt 内部
# 钳到 ``min(10, n_calls)``，与旧 ``10`` 等价，``random_state=42`` 同值）。
# 原模块 import 期的 ``warnings.filterwarnings('ignore')`` 全局副作用不再
# 保留——它只影响告警显示，不影响任何数值。
# ---------------------------------------------------------------------------

WORST_LR_DELTA_SCORE = -1.0e6

#: 原 bayes_opt_lr_delta 的 optimize_pib 固定形状参数（与引擎 _run_physical
#: 的默认值**不同**：shrink_ratio 0.9 / cam_size 200 / dm_neibor_diff 200）。
_LR_DELTA_PIB_KWARGS: dict[str, Any] = {
    "shrink_iter": 0,
    "shrink_ratio": 0.9,
    "cam_size": 200,
    "dm_unit_mask": None,
    "dm_neibor_diff": 200,
    "dm_max_voltage": None,
    "dm_min_voltage": None,
}


def _auto_center_from_camera(cam_id: int, exposure_time_ms: float):
    from ao_shaping.drivers import MIICamera
    from ao_shaping.utils.image.spots_calc import centroid

    with MIICamera(cam_id=cam_id, exposure_time_ms=exposure_time_ms) as cam:
        img = cam.get_numpy_image(10)
    h, w = img.shape
    return centroid(np.where(img > np.max(img[: max(int(h // 50), 2), : max(int(w // 50), 2)]), 1, 0))


def lr_delta_score(
    lr: float,
    delta: float,
    *,
    center: Any = None,
    epochs: int = 100,
    exposure_time_ms: float = 80,
    cam_id: int = 0,
) -> float:
    """一次硬件 lr/delta 评估，返回 PIB（越大越好）。

    ``center=None`` 时与原实现一致：每次评估都开相机取左上角亮区质心。
    """
    try:
        if center is None:
            center = _auto_center_from_camera(cam_id, exposure_time_ms)
        recorder = optimize_pib(
            center=center,
            epochs=epochs,
            delta=abs(delta),
            lr=lr,
            exposure_time_ms=exposure_time_ms,
            cam_id=cam_id,
            show=False,
            init_v=[],
            **_LR_DELTA_PIB_KWARGS,
        )
        best_iter, _ = recorder.get_best_iter()
        return float(best_iter["pib"])
    # policy: a failed hardware measurement is reported to the optimiser as its
    # worst score so the search keeps going; the failure itself is logged
    except Exception as exc:
        logger.error(f"lr/delta 评估失败: {exc}")
        return WORST_LR_DELTA_SCORE


def bayesian_optimize_lr_delta(
    n_calls: int = 30,
    lr_bounds: tuple[float, float] = (0.1, 5.0),
    delta_bounds: tuple[float, float] = (0.1, 5.0),
    center: Any = None,
    epochs: int = 100,
    exposure_time_ms: float = 80,
    cam_id: int = 0,
) -> dict[str, Any]:
    """贝叶斯搜索 (lr, delta)，走共享 ``bayesian_search_hyperparams`` 引擎。

    返回键与原模块一致：``best_params`` / ``best_score`` / ``optimization_result``。
    """

    def _score(params: dict[str, float]) -> float:
        return lr_delta_score(
            params["lr"],
            params["delta"],
            center=center,
            epochs=epochs,
            exposure_time_ms=exposure_time_ms,
            cam_id=cam_id,
        )

    result = bayesian_search_hyperparams(
        mode="physical",
        search_params=[
            SearchParam("lr", float(lr_bounds[0]), float(lr_bounds[1])),
            SearchParam("delta", float(delta_bounds[0]), float(delta_bounds[1])),
        ],
        n_calls=n_calls,
        objective_fn=_score,
    )
    return {
        "best_params": result.best_params,
        "best_score": result.best_score,
        "optimization_result": result.raw_result,
    }


def grid_search_lr_delta(
    lr_values: tuple[float, ...] = (0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
    delta_values: tuple[float, ...] = (0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
    center: Any = None,
    epochs: int = 100,
    exposure_time_ms: float = 80,
    cam_id: int = 0,
) -> dict[str, Any]:
    """网格搜索 (lr, delta)；单点失败记为 ``-inf`` 并继续（比较价值在全表）。"""
    best_score = -np.inf
    best_params: dict[str, Any] = {"lr": None, "delta": None}
    results: list[dict[str, Any]] = []
    total = len(lr_values) * len(delta_values)
    current = 0

    for lr in lr_values:
        for delta in delta_values:
            current += 1
            logger.debug(f"测试参数组合 {current}/{total}: lr={lr}, delta={delta}")
            try:
                score = lr_delta_score(
                    lr,
                    delta,
                    center=center,
                    epochs=epochs,
                    exposure_time_ms=exposure_time_ms,
                    cam_id=cam_id,
                )
                results.append({"lr": lr, "delta": delta, "score": score})
                if score > best_score:
                    best_score = score
                    best_params = {"lr": lr, "delta": delta}
                logger.debug(f"  得分: {score:.4f}")
            # policy: one failing (lr, delta) point is recorded with its error and
            # the sweep continues -- the grid's value is the comparison across points
            except Exception as exc:
                logger.error(f"  错误: {exc}")
                results.append({"lr": lr, "delta": delta, "score": -np.inf, "error": str(exc)})

    return {"best_params": best_params, "best_score": best_score, "all_results": results}
