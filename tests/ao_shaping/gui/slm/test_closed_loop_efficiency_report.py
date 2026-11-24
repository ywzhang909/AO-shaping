"""闭环完成态的能量利用率报告 (2026-11-25)。

锁三件事:
1. 能量利用率直接复用最优轮已落账的 ``window_fraction``, 不重算、不采集;
2. SLM 一级衍射效率从探针 ``summary.json`` 的 ``lobe.ratios[blaze_order]``
   读出 (与诊断页同一数据源), 缺键/非正报错不静默;
3. 对比值 ``relative_to_slm`` = 利用率 / 衍射效率, 不设硬判定。
"""
from __future__ import annotations

import json

import pytest

from ao_shaping.gui.slm.closed_loop_shaping import (
    FrameGate,
    LoopState,
    LoopSummary,
    RoundRecord,
    load_slm_diffraction_efficiency,
    shaping_efficiency_report,
)


def _gate() -> FrameGate:
    return FrameGate(ok=True, reasons=(), peak=100.0, full_scale=255.0,
                     saturation_pixels=0, centroid_xy=(10.0, 10.0), exposure_ms=0.1)


def _round(index: int, score: float | None, fraction: float | None) -> RoundRecord:
    return RoundRecord(round_index=index, phase_id=f"p{index}", frame_id=index,
                       exposure_ms=0.1, gate=_gate(),
                       measured_score=score, window_fraction=fraction)


def _summary(rounds, best_score=None) -> LoopSummary:
    return LoopSummary(state=LoopState.STOPPED, rounds=list(rounds), best_score=best_score)


def _probe_json(tmp_path, *, efficiency=0.39775069961204507):
    payload = {
        "probe": "diffraction_efficiency",
        "timestamp_utc": "2026-11-23T04:53:11.928665+00:00",
        "config": {"gray_depth": 993},
        "lobe": {
            "ratios": {"zero": 1.0 - efficiency, "plus_one": 0.0, "minus_one": efficiency},
            "orders": {"blaze_order": "minus_one"},
        },
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 探针 summary.json 读取
# ---------------------------------------------------------------------------

def test_load_slm_diffraction_efficiency_reads_the_blaze_order(tmp_path):
    """闪耀方向是 minus_one 时, 效率取 minus_one 级, 不是零级也不是 ±1 之和。"""
    info = load_slm_diffraction_efficiency(_probe_json(tmp_path))
    assert info["efficiency"] == pytest.approx(0.39775069961204507)
    assert info["blaze_order"] == "minus_one"
    assert info["gray_depth"] == 993
    assert info["timestamp_utc"] == "2026-11-23T04:53:11.928665+00:00"


def test_load_slm_diffraction_efficiency_rejects_zero_efficiency(tmp_path):
    """零级全收 (灰度 0 之类) 时效率为 0: 直接报错, 不拿 0 去做除数。"""
    with pytest.raises(ValueError, match="衍射效率非正或无效"):
        load_slm_diffraction_efficiency(_probe_json(tmp_path, efficiency=0.0))


def test_load_slm_diffraction_efficiency_missing_keys_raise(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"probe": "diffraction_efficiency"}), encoding="utf-8")
    with pytest.raises(KeyError):
        load_slm_diffraction_efficiency(path)


# ---------------------------------------------------------------------------
# 完成态报告: 复用最优轮的 window_fraction
# ---------------------------------------------------------------------------

def test_report_uses_the_best_score_round():
    """best_score 命中的轮次就是"当前整形结果" (best_phase 对应的那一轮)。"""
    summary = _summary(
        [_round(0, 1.2, 0.10), _round(1, 0.8, 0.30), _round(2, 0.9, 0.25)],
        best_score=0.8,
    )
    report = shaping_efficiency_report(summary)
    assert report["round_index"] == 1
    assert report["window_fraction"] == pytest.approx(0.30)
    assert report["relative_to_slm"] is None


def test_report_falls_back_to_lowest_scored_round_without_best_score():
    summary = _summary([_round(0, 1.2, 0.10), _round(1, 0.8, 0.30)])
    report = shaping_efficiency_report(summary)
    assert report["round_index"] == 1
    assert report["window_fraction"] == pytest.approx(0.30)


def test_report_skips_rounds_without_a_fraction():
    """门禁失败等没有 window_fraction 的轮次不参与最优轮选择。"""
    summary = _summary([_round(0, 0.5, None), _round(1, 0.8, 0.30)], best_score=0.5)
    report = shaping_efficiency_report(summary)
    assert report["round_index"] == 1
    assert report["window_fraction"] == pytest.approx(0.30)


def test_report_empty_when_no_usable_round():
    report = shaping_efficiency_report(_summary([_round(0, None, None)]))
    assert report["round_index"] is None
    assert report["window_fraction"] is None
    assert report["relative_to_slm"] is None


# ---------------------------------------------------------------------------
# 与 SLM 衍射效率的对比
# ---------------------------------------------------------------------------

def test_report_compares_utilization_against_slm_efficiency():
    summary = _summary([_round(1, 0.8, 0.30)], best_score=0.8)
    report = shaping_efficiency_report(summary, slm_diffraction_efficiency=0.3977507)
    assert report["slm_diffraction_efficiency"] == pytest.approx(0.3977507)
    assert report["relative_to_slm"] == pytest.approx(0.30 / 0.3977507)


def test_report_allows_relative_above_one():
    """目标压零级/漏光更少的构型可以超过一级衍射效率: 只报数, 不封顶不判定。"""
    summary = _summary([_round(1, 0.8, 0.60)], best_score=0.8)
    report = shaping_efficiency_report(summary, slm_diffraction_efficiency=0.40)
    assert report["relative_to_slm"] == pytest.approx(1.5)


def test_report_rejects_non_positive_slm_efficiency():
    summary = _summary([_round(1, 0.8, 0.30)], best_score=0.8)
    with pytest.raises(ValueError, match="slm_diffraction_efficiency"):
        shaping_efficiency_report(summary, slm_diffraction_efficiency=0.0)


def test_report_without_utilization_has_no_relative():
    report = shaping_efficiency_report(_summary([]), slm_diffraction_efficiency=0.4)
    assert report["window_fraction"] is None
    assert report["relative_to_slm"] is None
