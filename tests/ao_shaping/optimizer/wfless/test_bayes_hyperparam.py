from __future__ import annotations

import pytest

# skopt 是 bayes_hyperparam 的重型可选依赖；环境未安装时跳过全部用例
pytest.importorskip("skopt")

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


# ---------------------------------------------------------------------------
# lr/delta 物理选型入口（第 187 批 #138 并入；等价性以桩驱动，不碰硬件）
# ---------------------------------------------------------------------------
class _FakeRecorder:
    def __init__(self, pib: float) -> None:
        self._pib = pib

    def get_best_iter(self):
        return {"pib": self._pib}, (0, 0)


def test_lr_delta_score_uses_legacy_pib_kwargs(monkeypatch) -> None:
    from ao_shaping.optimizer.wfless import bayes_hyperparam as bh

    seen: dict = {}

    def fake_optimize_pib(**kwargs):
        seen.update(kwargs)
        return _FakeRecorder(0.83)

    monkeypatch.setattr(bh, "optimize_pib", fake_optimize_pib)
    score = bh.lr_delta_score(1.5, -0.4, center=(5, 6))
    assert score == 0.83
    assert seen["delta"] == 0.4  # 原实现 abs(delta)
    assert seen["shrink_ratio"] == 0.9 and seen["cam_size"] == 200
    assert seen["dm_neibor_diff"] == 200 and seen["show"] is False


def test_lr_delta_score_failure_returns_worst_without_raising(monkeypatch) -> None:
    from ao_shaping.optimizer.wfless import bayes_hyperparam as bh

    def boom(**kwargs):
        raise RuntimeError("camera offline")

    monkeypatch.setattr(bh, "optimize_pib", boom)
    assert bh.lr_delta_score(1.0, 1.0, center=(0, 0)) == bh.WORST_LR_DELTA_SCORE


def test_bayesian_optimize_lr_delta_contract(monkeypatch) -> None:
    from ao_shaping.optimizer.wfless import bayes_hyperparam as bh

    calls: list[tuple[float, float]] = []

    def fake_score(lr, delta, **_kw):
        calls.append((lr, delta))
        return 1.0 - (lr - 2.0) ** 2 - (delta - 3.0) ** 2

    monkeypatch.setattr(bh, "lr_delta_score", fake_score)
    result = bh.bayesian_optimize_lr_delta(n_calls=14, epochs=1)
    assert set(result) == {"best_params", "best_score", "optimization_result"}
    assert set(result["best_params"]) == {"lr", "delta"}
    best = max(
        1.0 - (lr - 2.0) ** 2 - (d - 3.0) ** 2 for lr, d in calls
    )
    assert result["best_score"] == pytest.approx(best)


def test_grid_search_records_error_rows_and_best(monkeypatch) -> None:
    from ao_shaping.optimizer.wfless import bayes_hyperparam as bh

    def fake_score(lr, delta, **_kw):
        if (lr, delta) == (2.0, 1.0):
            raise RuntimeError("point offline")
        return lr + delta

    monkeypatch.setattr(bh, "lr_delta_score", fake_score)
    out = bh.grid_search_lr_delta(lr_values=(1.0, 2.0), delta_values=(1.0, 2.0))
    assert out["best_params"] == {"lr": 2.0, "delta": 2.0}
    assert out["best_score"] == 4.0
    errors = [r for r in out["all_results"] if "error" in r]
    assert len(errors) == 1 and errors[0]["score"] == float("-inf")
