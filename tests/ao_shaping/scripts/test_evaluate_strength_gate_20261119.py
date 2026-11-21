# -*- coding: utf-8 -*-
"""强度门 runner 的判决聚合：不可评估行不得被"删掉后放行"。

预注册 §3.4 / 计划 §6.5-6.6 规定：只要有一行 holdout 不可评估（暗场缺失、
曝光簿记不符、饱和、空 mask），该 split 就是 ``not_assessable``，整体门也不得
因为另一个 split 通过而判 pass。这里只测判决聚合，不跑前向模型、不碰硬件。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "slm" / "evaluate_strength_gate_20261119.py"


def _load():
    spec = importlib.util.spec_from_file_location("evaluate_strength_gate_20261119", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load()


def _row(label: str, rel_l2, reason: str | None = None, *, split: str = "test") -> dict:
    return {
        "label": label,
        "split": split,
        "rel_l2_gate": rel_l2,
        "n_pixels_gate": 0 if rel_l2 is None else 100,
        "rel_l2_fullwindow_unmasked": 0.5 if rel_l2 is None else rel_l2,
        "not_assessable_reason": reason,
    }


def test_all_rows_assessable_and_below_delta_passes() -> None:
    rows = [_row("a", 0.05), _row("b", 0.10)]
    summary = runner._pass_summary(rows, "test")
    assert summary["verdict"] == "pass"
    assert summary["n_assessable_rows"] == 2
    assert summary["rel_l2_gate"] == 0.10


def test_all_rows_assessable_but_one_over_delta_fails() -> None:
    rows = [_row("a", 0.05), _row("b", 0.31)]
    summary = runner._pass_summary(rows, "test")
    assert summary["verdict"] == "fail"
    assert summary["rel_l2_gate_worst_label"] == "b"


def test_single_unassessable_row_forces_split_not_assessable() -> None:
    rows = [
        _row("a", 0.05),
        _row("b", 0.06),
        _row("c", None, "no dark at 0.051 ms"),
    ]
    summary = runner._pass_summary(rows, "test")
    assert summary["verdict"] == "not_assessable"
    assert summary["n_assessable_rows"] == 2
    assert summary["n_rows"] == 3
    assert summary["unassessable_labels"] == ["c"]
    assert "no dark at 0.051 ms" in summary["reason"]
    # 诊断值仍在，但不产生 pass/fail
    assert summary["rel_l2_gate"] == 0.06


def test_no_assessable_rows_is_not_assessable() -> None:
    rows = [_row("a", None, "no dark"), _row("b", None, "saturated")]
    summary = runner._pass_summary(rows, "test")
    assert summary["verdict"] == "not_assessable"
    assert summary["rel_l2_gate"] is None


def test_overall_needs_a_dark_source() -> None:
    passes = [{"verdict": "pass"}]
    assert runner._overall_verdict(passes, dark_supplied=False, dark_error=None) == "not_assessable"
    assert runner._overall_verdict(passes, dark_supplied=True, dark_error="boom") == "not_assessable"


def test_overall_not_assessable_split_blocks_the_gate() -> None:
    passes = [{"verdict": "pass"}, {"verdict": "not_assessable"}]
    assert runner._overall_verdict(passes, dark_supplied=True, dark_error=None) == "not_assessable"


def test_overall_fail_dominates_pass_and_pass_needs_all() -> None:
    assert runner._overall_verdict(
        [{"verdict": "pass"}, {"verdict": "fail"}], dark_supplied=True, dark_error=None
    ) == "fail"
    assert runner._overall_verdict(
        [{"verdict": "pass"}, {"verdict": "pass"}], dark_supplied=True, dark_error=None
    ) == "pass"
    assert runner._overall_verdict([], dark_supplied=True, dark_error=None) == "not_assessable"
