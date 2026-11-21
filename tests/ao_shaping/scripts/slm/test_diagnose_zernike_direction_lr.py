"""Offline tests for the Zernike-direction / residual-lr diagnostic script.

All data is synthetic (no hardware, no real run): we build small parameter histories
with a consistent or zigzag update direction, exercise the pure metric functions and
the CLI ``run`` orchestration, and verify JSON output.
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "slm" / "diagnose_zernike_direction_lr.py"


def _load_script():
    """Load the diagnostic script by path (it lives under ``scripts/slm``)."""
    spec = importlib.util.spec_from_file_location(
        "diagnose_zernike_direction_lr", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


diag = _load_script()

#: Seven residual_noll modes, matching the real ``_flatten_parameters`` key set.
NOLLS = ("5", "6", "7", "8", "9", "10", "11")


def _entry(Z: np.ndarray, loss: float, legacy: bool = True) -> dict:
    """Build one history entry from a residual vector ``Z`` and a scalar loss."""
    if legacy:
        params: dict[str, Any] = {
            "residual_noll_radians": {noll: float(v) for noll, v in zip(NOLLS, Z)}
        }
    else:
        params = {f"residual_noll_{noll}": float(v) for noll, v in zip(NOLLS, Z)}
    return {"loss": float(loss), "parameters": params}


# ---------------------------------------------------------------------------
# extract_residual
# ---------------------------------------------------------------------------
def test_extract_residual_legacy_dict_sorted_by_noll() -> None:
    """Legacy nested dict is flattened with numeric (not lexical) Noll ordering."""
    Z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    # 输入字典故意乱序，验证按数值排序。
    params = {"residual_noll_radians": {"11": 0.7, "5": 0.1, "9": 0.5, "6": 0.2,
                                        "8": 0.4, "10": 0.6, "7": 0.3}}
    out = diag.extract_residual({"parameters": params})
    np.testing.assert_allclose(out, Z)


def test_extract_residual_flat_keys_preferred() -> None:
    """New-style flat ``residual_noll_*`` keys take priority over the legacy dict."""
    params: dict[str, Any] = {
        "residual_noll_radians": {"5": 9.9, "6": 9.9, "7": 9.9, "8": 9.9,
                                  "9": 9.9, "10": 9.9, "11": 9.9},
        "residual_noll_5": 1.0,
        "residual_noll_6": 2.0,
        "residual_noll_7": 3.0,
    }
    out = diag.extract_residual({"parameters": params})
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_extract_residual_missing_keys_raises() -> None:
    """Absence of both sources raises ``ValueError``."""
    for entry in (
        {},
        {"parameters": {}},
        {"parameters": {"gain": 1.0}},
    ):
        try:
            diag.extract_residual(entry)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {entry!r}")


# ---------------------------------------------------------------------------
# direction_cosines
# ---------------------------------------------------------------------------
def test_direction_cosines_straight_sequence_near_one() -> None:
    """Collinear same-direction increments give cosine ~= 1."""
    dZ = np.tile(np.array([1.0, 2.0, -0.5]), (5, 1))
    cos = diag.direction_cosines(dZ)
    assert cos.shape == (4,)
    np.testing.assert_allclose(cos, np.ones(4), atol=1e-12)


def test_direction_cosines_zigzag_lower() -> None:
    """Alternating increments give cosine ~= -1, clearly below the straight case."""
    dZ = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    cos = diag.direction_cosines(dZ)
    np.testing.assert_allclose(cos, -np.ones(3), atol=1e-12)
    assert float(np.mean(cos)) < 0.0


# ---------------------------------------------------------------------------
# analyze_trajectory
# ---------------------------------------------------------------------------
def test_analyze_trajectory_straight_progress() -> None:
    """20 rounds advancing along a fixed direction with descending loss."""
    n_iter = 20
    direction = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.1])
    history = [
        _entry((k + 1) * direction, loss=10.0 / (k + 1))
        for k in range(n_iter)
    ]
    a = diag.analyze_trajectory(history)

    assert a["n_iters"] == n_iter
    np.testing.assert_allclose(a["cos_mean"], 1.0, atol=1e-6)
    np.testing.assert_allclose(a["cos_last10"], 1.0, atol=1e-6)
    assert a["loss_first"] > a["loss_last"]
    assert a["loss_drop_mean"] > 0.0
    assert a["consistent_active_frac"] > 0.9
    assert a["noise_floor_from"] is None


# ---------------------------------------------------------------------------
# aggregate + make_recommendation
# ---------------------------------------------------------------------------
def _analysis(cos_last10: float, rho_samples: float) -> dict:
    """Build a synthetic per-run analysis dict with given last-10 cosine."""
    steps = np.linspace(0.01, 0.05, 6)
    drops = steps * rho_samples  # 正相关时 drop 随 step 增大
    return {
        "n_iters": 40,
        "loss_first": 1.0,
        "loss_last": 0.1,
        "cos_mean": cos_last10,
        "cos_last10": cos_last10,
        "step_norm_mean": 0.03,
        "step_norm_std": 0.001,
        "loss_drop_mean": 0.01,
        "consistent_active_frac": 0.8,
        "noise_floor_from": None,
        "cos_values": [cos_last10] * 30,
        "consistent_step_norms": steps.tolist(),
        "consistent_loss_drops": drops.tolist(),
    }


def test_aggregate_positive_correlation_recommends_scaling() -> None:
    """Direction-consistent + positive spearman yields the 'safe to scale' text."""
    analyses = [_analysis(0.98, 100.0) for _ in range(4)]
    metrics = diag.aggregate(analyses)

    assert metrics["n_runs"] == 4
    assert metrics["frac_runs_cos_gt_0_9"] == 1.0
    assert metrics["step_vs_lossdrop_spearman"] is not None
    assert metrics["step_vs_lossdrop_spearman"] > 0.3
    assert "可安全放大" in metrics["recommendation"]


def test_make_recommendation_weak_correlation_and_inconsistent() -> None:
    """Weak correlation -> A/B; low consistency -> no prescription."""
    weak = {
        "frac_runs_cos_gt_0_9": 1.0,
        "step_vs_lossdrop_spearman": 0.1,
    }
    assert "A/B" in diag.make_recommendation(weak)
    inconsistent = {
        "frac_runs_cos_gt_0_9": 0.2,
        "step_vs_lossdrop_spearman": 0.8,
    }
    assert "不建议" in diag.make_recommendation(inconsistent)


# ---------------------------------------------------------------------------
# CLI run
# ---------------------------------------------------------------------------
def test_run_writes_json_with_recommendation(tmp_path: Path) -> None:
    """run() reads a synthetic iterations.pkl and writes a JSON with recommendation."""
    run_dir = tmp_path / "run0" / "fit"
    run_dir.mkdir(parents=True)

    direction = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.1])
    history = [
        _entry((k + 1) * direction, loss=10.0 / (k + 1))
        for k in range(35)
    ]
    with (run_dir / "iterations.pkl").open("wb") as fh:
        pickle.dump(history, fh)

    out = tmp_path / "summary.json"
    args = diag._build_parser().parse_args(
        ["--data-dir", str(tmp_path), "--min-iters", "10", "--out", str(out)]
    )
    result = diag.run(args)

    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "recommendation" in payload["aggregate"]
    assert payload["aggregate"]["n_runs"] == 1
    assert len(payload["analyses"]) == 1
    assert result["aggregate"]["recommendation"] == payload["aggregate"]["recommendation"]
