"""闭环完成态的能量利用率报告 (2026-11-25, 2026-11-26 修正效率口径)。

锁三件事:
1. 能量利用率直接复用最优轮已落账的 ``window_fraction``, 不重算、不采集;
2. SLM 衍射效率按「从平相位 0 级转走的能量」定义: η = E_blaze / F0,
   F0 = 探针 ``relative_efficiency.baseline_zero_window_counts``;
   ``lobe.ratios[blaze_order]`` 的分母是闪耀后三级之和, 只作诊断;
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
    flat_zero_transfer_fraction,
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


def _probe_json(tmp_path, *, efficiency=0.19274475503017363, schema="new"):
    """F0 = 1000 counts, 闪耀级能量 = efficiency * F0。"""
    flat_zero = 1000.0
    payload = {
        "probe": "diffraction_efficiency",
        "timestamp_utc": "2026-11-23T04:53:11.928665+00:00",
        "config": {"gray_depth": 993},
        "lobe": {
            "energies": {
                "zero": 500.0,
                "plus_one": 0.0,
                "minus_one": efficiency * flat_zero,
            },
            "ratios": {"zero": 0.6, "plus_one": 0.0, "minus_one": 0.4},
        },
        "relative_efficiency": {
            "baseline_zero_window_counts": flat_zero,
        },
    }
    if schema == "new":
        payload["relative_efficiency"]["blaze_order"] = "minus_one"
    elif schema == "old":
        payload["lobe"]["orders"] = {"blaze_order": "minus_one"}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# η = E / F0 公式
# ---------------------------------------------------------------------------

def test_flat_zero_transfer_fraction_allows_negative_shaping_numerator():
    """η_s = (S - S0) / F0 可以是负的 (0 级泄漏比可调制分量多)。"""
    assert flat_zero_transfer_fraction(-66.0, 650.0) == pytest.approx(-0.1015384615)
    assert flat_zero_transfer_fraction(350.9, 649.7) == pytest.approx(0.5400954287)


@pytest.mark.parametrize("numerator,denominator", [(1.0, 0.0), (1.0, -3.0), (float("nan"), 1.0)])
def test_flat_zero_transfer_fraction_rejects_invalid_baseline(numerator, denominator):
    with pytest.raises(ValueError, match="平相位 0 级基准"):
        flat_zero_transfer_fraction(numerator, denominator)


# ---------------------------------------------------------------------------
# 探针 summary.json 读取
# ---------------------------------------------------------------------------

def test_load_slm_diffraction_efficiency_reads_the_blaze_order(tmp_path):
    """闪耀方向是 minus_one 时, 效率取 minus_one 级 / F0, 不是 lobe 三级占比。"""
    info = load_slm_diffraction_efficiency(_probe_json(tmp_path))
    assert info["efficiency"] == pytest.approx(0.19274475503017363)
    assert info["lobe_share"] == pytest.approx(0.4)  # 仅诊断, 不参与效率
    assert info["blaze_energy"] == pytest.approx(192.74475503017363)
    assert info["flat_zero_energy"] == pytest.approx(1000.0)
    assert info["blaze_order"] == "minus_one"
    assert info["gray_depth"] == 993
    assert info["timestamp_utc"] == "2026-11-23T04:53:11.928665+00:00"


def test_load_slm_diffraction_efficiency_reads_legacy_blaze_order(tmp_path):
    """旧探针把闪耀级放在 lobe.orders 下, 也照读。"""
    info = load_slm_diffraction_efficiency(_probe_json(tmp_path, schema="old"))
    assert info["blaze_order"] == "minus_one"


def test_load_slm_diffraction_efficiency_prefers_probe_reported_share(tmp_path):
    """探针自己算的 blaze_energy_share 就是该公式, 存在时直接采信。"""
    path = _probe_json(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["relative_efficiency"]["blaze_energy_share"] = 0.25
    path.write_text(json.dumps(payload), encoding="utf-8")
    info = load_slm_diffraction_efficiency(path)
    assert info["efficiency"] == pytest.approx(0.25)
    assert info["blaze_energy"] == pytest.approx(192.74475503017363)


def test_load_slm_diffraction_efficiency_requires_the_flat_zero_baseline(tmp_path):
    """没有平相位 0 级基准就算不出新定义的效率: 报错, 不回退到三窗分母。"""
    path = _probe_json(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["relative_efficiency"]["baseline_zero_window_counts"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KeyError, match="baseline_zero_window_counts"):
        load_slm_diffraction_efficiency(path)


def test_load_slm_diffraction_efficiency_picks_stronger_first_order(tmp_path):
    """两版 schema 都没标闪耀级时, 取 ±1 中较强者。"""
    path = _probe_json(tmp_path, schema="none")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["lobe"]["energies"]["plus_one"] = 250.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    info = load_slm_diffraction_efficiency(path)
    assert info["blaze_order"] == "plus_one"


def test_load_slm_diffraction_efficiency_rejects_above_one(tmp_path):
    """闪耀级能量超过 F0 说明窗口口径不一致, 不报 > 100% 的假效率。"""
    with pytest.raises(ValueError, match="衍射效率非正或无效"):
        load_slm_diffraction_efficiency(_probe_json(tmp_path, efficiency=1.2))


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
