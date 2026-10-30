"""检查卡的判定层：六项结论、红黄绿分工、以及"没测就说不知道"。

这张卡片存在的理由写在 ``precheck`` 模块头：那次 16 段 / 112 块里，至少有三个问题
开跑前就能看出来（零级剔除盘压进目标框 5.61 %、落盘量与磁盘余量、复用旧模型）。
所以这里测的不是"卡片画得出来"，而是**判定本身**：

* 每项的结论必须与它自己引用的判据算得一致（几何 pct 用
  ``preflight.check_zero_overlap`` 复算，不是硬编码一个百分比）；
* 缺输入时必须是 ``unknown``，不许静默降级成 ``pass``；
* ``stop`` 只留给结构性问题，``warn`` 才需要人勾选。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from ao_shaping.gui.slm import precheck
from ao_shaping.tools.slm.transfer.campaign import (
    campaign_matrix,
    derive_settings,
    estimate_disk_gb,
    estimate_seconds,
)
from ao_shaping.tools.slm.transfer.cost_table import record_stage_cost
from ao_shaping.tools.slm.transfer.preflight import check_zero_overlap

SHAPES = ("square", "triangle", "star", "circle")

#: 四形状一份 campaign 的派生参数集（纯计算，不读盘、不开设备）。
SETTINGS = derive_settings(SHAPES, 1200.0)


def _card(*, out_dir: Path, zero_xy=None, model_dir=None, campaigns=(), target_xy=None,
          cam_size=None, budget_hours=precheck.BUDGET_HOURS,
          settings=SETTINGS) -> precheck.CheckCard:
    return precheck.build_check_card(
        settings=settings, out_dir=out_dir, model_dir=model_dir, zero_xy=zero_xy,
        recent_campaigns=campaigns, target_xy=target_xy, cam_size=cam_size,
        budget_hours=budget_hours)


def _by_key(card: precheck.CheckCard) -> dict[str, precheck.Check]:
    return {check.key: check for check in card.checks}


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    """一个**不存在**的落盘目录：磁盘预算要能沿着父目录上溯到真实盘符。"""
    return tmp_path / "campaign" / "s1200"


# --------------------------------------------------------------------------- #
# 卡片骨架
# --------------------------------------------------------------------------- #
def test_card_has_the_six_checks_in_a_fixed_order(out_dir: Path) -> None:
    """顺序固定：几何 → 磁盘 → 时间 → 模型 → 饱和 → 曝光（表格按它印）。"""
    card = _card(out_dir=out_dir)
    assert [check.key for check in card.checks] == [
        "geometry", "disk", "time", "model", "saturation", "exposure"]
    assert all(check.label and check.detail for check in card.checks)


def test_every_check_carries_its_evidence(out_dir: Path) -> None:
    """没有出处的结论不许上卡片 —— 否则操作员无法判断"这个绿勾凭什么"。"""
    for check in _card(out_dir=out_dir).checks:
        assert check.evidence.strip(), check.key
        assert len(check.evidence) > 30, check.key


def test_status_is_the_worst_check(out_dir: Path) -> None:
    """``stop`` > ``warn`` > ``unknown`` > ``pass``：卡片只报最坏那一项。"""
    card = _card(out_dir=out_dir, zero_xy=(1000.0, 1000.0), target_xy=(1000.0, 1000.0),
                 budget_hours=0.01, settings=derive_settings(SHAPES, 200.0))
    assert "stop" in {check.status for check in card.checks}
    assert card.status == "stop"
    assert card.blocking is True


# --------------------------------------------------------------------------- #
# ① 几何自洽
# --------------------------------------------------------------------------- #
def test_geometry_is_unknown_without_a_measured_zero_order(out_dir: Path) -> None:
    """零级没测过 → ``unknown`` + 指路，绝不拿同心假设算出 28 % 当真。"""
    check = _by_key(_card(out_dir=out_dir, target_xy=(1000.0, 1000.0)))["geometry"]
    assert check.status == "unknown"
    assert "零级位置未测" in check.detail
    assert "probe_optical_path_validity" in check.detail


def test_geometry_is_unknown_without_a_target_center(out_dir: Path) -> None:
    """零级有、目标框中心没有（无历史 geometry_plan.json）→ 也报 ``unknown``。"""
    check = _by_key(_card(out_dir=out_dir, zero_xy=(1000.0, 1000.0)))["geometry"]
    assert check.status == "unknown"
    assert "目标框位置未知" in check.detail


def test_geometry_reports_the_number_the_canonical_overlap_computes(out_dir: Path) -> None:
    """卡片上的百分比必须与 ``preflight.check_zero_overlap`` 逐位一致。"""
    zero_radius = float(SETTINGS.settings["zero_radius"].value)
    zero = (1000.0, 1000.0)
    target = (1240.0, 1000.0)
    expected = check_zero_overlap(target[0], target[1], float(SETTINGS.side), zero,
                                  zero_radius=zero_radius)
    check = _by_key(_card(out_dir=out_dir, zero_xy=zero, target_xy=target))["geometry"]
    assert f"{expected.pct:.2f} %" in check.detail
    band = "stop" if expected.pct > 15.0 else "warn" if expected.pct > 5.0 else "pass"
    assert check.status == band


def test_geometry_stops_on_the_concentric_28_percent_case(out_dir: Path) -> None:
    """同心假设在 200 px 档是 28 %：那正是这条检查要拦下的形态（>15 % → stop）。"""
    settings = derive_settings(SHAPES, 200.0)
    zero = (1000.0, 1000.0)
    card = precheck.build_check_card(
        settings=settings, out_dir=out_dir, model_dir=None, zero_xy=zero, target_xy=zero)
    check = _by_key(card)["geometry"]
    assert check.status == "stop"
    assert "换尺寸或挪目标框" in check.detail


def test_geometry_stops_when_the_target_does_not_fit_the_window(out_dir: Path) -> None:
    """目标半边长 + 余量 > 采样窗半跨时 ``validate.recentre_window`` 会**裁掉**目标。"""
    settings = derive_settings(SHAPES, 1200.0, overrides={"window_span": 100.0})
    card = precheck.build_check_card(
        settings=settings, out_dir=out_dir, model_dir=None, zero_xy=(1000.0, 1000.0),
        target_xy=(1000.0, 1000.0))
    check = _by_key(card)["geometry"]
    assert check.status == "stop"
    assert "采样窗半跨" in check.detail


def test_geometry_stops_when_the_target_exceeds_the_camera_window(out_dir: Path) -> None:
    settings = derive_settings(SHAPES, 600.0, overrides={"window_span": 900.0})
    card = precheck.build_check_card(
        settings=settings, out_dir=out_dir, model_dir=None, zero_xy=(1000.0, 1000.0),
        target_xy=(1000.0, 1000.0), cam_size=400)
    check = _by_key(card)["geometry"]
    assert check.status == "stop"
    assert "相机窗口" in check.detail


def test_geometry_uses_the_historical_target_cell_of_the_same_side(tmp_path: Path) -> None:
    """没有显式 target_xy 时取历史 ``geometry_plan.json`` 里**同尺寸**的规划中心。"""
    campaign = tmp_path / "stamp" / "s1200"
    campaign.mkdir(parents=True)
    (campaign / "geometry_plan.json").write_text(json.dumps({
        "zero_xy": [1000.0, 1000.0],
        "cells": [
            {"side": 600.0, "target_x": [50.0, 50.0], "target_y": [50.0, 50.0]},
            {"side": 1200.0, "target_x": [1300.0, 1300.0], "target_y": [1000.0, 1000.0]},
        ],
    }), encoding="utf-8")
    zero_radius = float(SETTINGS.settings["zero_radius"].value)
    expected = check_zero_overlap(1300.0, 1000.0, 1200.0, (1000.0, 1000.0),
                                  zero_radius=zero_radius)
    card = _card(out_dir=tmp_path / "out", zero_xy=(1000.0, 1000.0),
                 campaigns=[campaign])
    assert f"{expected.pct:.2f} %" in _by_key(card)["geometry"].detail


# --------------------------------------------------------------------------- #
# ② 磁盘预算
# --------------------------------------------------------------------------- #
def test_disk_passes_with_enough_free_space(out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda _path: SimpleNamespace(total=10 ** 12, used=0, free=10 ** 12))
    check = _by_key(_card(out_dir=out_dir))["disk"]
    assert check.status == "pass"
    assert f"{estimate_disk_gb(campaign_matrix(SHAPES)):.1f} GB" in check.detail


def test_disk_stops_when_the_drive_would_fill(out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda _path: SimpleNamespace(total=10 ** 9, used=10 ** 9, free=10 ** 9))
    check = _by_key(_card(out_dir=out_dir))["disk"]
    assert check.status == "stop"
    assert "余量不足" in check.detail


def test_disk_is_unknown_when_the_usage_cannot_be_read(out_dir: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_path: str) -> None:
        raise OSError("no such drive")

    monkeypatch.setattr(shutil, "disk_usage", boom)
    check = _by_key(_card(out_dir=out_dir))["disk"]
    assert check.status == "unknown"
    assert "手工确认" in check.detail


# --------------------------------------------------------------------------- #
# ③ 时间预算
# --------------------------------------------------------------------------- #
def test_time_passes_under_the_default_budget(out_dir: Path) -> None:
    check = _by_key(_card(out_dir=out_dir))["time"]
    low, expected, high = estimate_seconds(campaign_matrix(SHAPES))
    assert check.status == "pass"
    assert "常数估算" in check.detail
    assert f"≈{expected / 60:.0f} min" in check.detail
    assert f"区间 {low / 60:.0f}–{high / 60:.0f} min" in check.detail


def test_time_warns_over_budget_but_never_stops(out_dir: Path) -> None:
    """超出预算只 ``warn``：这条问的是"今天还来不来得及"，不是"跑得对不对"。"""
    check = _by_key(_card(out_dir=out_dir, budget_hours=0.1))["time"]
    assert check.status == "warn"
    assert "超出预算" in check.detail


def _write_cost(campaign: Path, shape: str, stage: str, *, seconds: float,
                blocks: int) -> None:
    """用真实写入口落一份段成本（不手搓 JSON —— 键名漂移要被这里抓住）。"""
    record_stage_cost(campaign / "shape" / shape / stage, stage=stage, shape=shape,
                      seconds=seconds, blocks=blocks)


def test_time_prefers_measured_costs_over_the_constant(out_dir: Path,
                                                       tmp_path: Path) -> None:
    """有历史 ``cost.json`` 时按实测外推，并把常数估算放在旁边做对照。"""
    campaign = tmp_path / "20261028" / "s1200"
    _write_cost(campaign, "square", "validate", seconds=160.0, blocks=8)   # 20 s/块
    _write_cost(campaign, "square", "dense", seconds=360.0, blocks=12)     # 30 s/块
    check = _by_key(_card(out_dir=out_dir, campaigns=[campaign]))["time"]
    # 中位 25 s/块 × 112 块 = 2800 s ≈ 47 min；常数 2688 s → 实测比常数 +4 %。
    assert check.status == "pass"
    assert "按实测" in check.detail
    assert "2 段实测" in check.detail
    assert "每块 25.0 s" in check.detail
    assert "常数估算" in check.detail
    assert "差 +4 %" in check.detail
    assert "实测更慢" in check.evidence


def test_time_falls_back_to_the_constant_and_says_why(out_dir: Path,
                                                      tmp_path: Path) -> None:
    """坏 JSON / 缺字段不许抛：回落常数，并把"为什么回落"写进判据。"""
    campaign = tmp_path / "stamp" / "s1200"
    half_written = campaign / "shape" / "square" / "validate"
    half_written.mkdir(parents=True)
    (half_written / "cost.json").write_text("{not json", encoding="utf-8")
    no_fields = campaign / "shape" / "star" / "validate"
    no_fields.mkdir(parents=True)
    (no_fields / "cost.json").write_text(
        json.dumps({"stage": "validate", "shape": "star"}), encoding="utf-8")
    check = _by_key(_card(out_dir=out_dir, campaigns=[campaign]))["time"]
    assert check.status == "pass"
    assert check.detail.startswith("常数估算")
    assert "常数估算而不是实测" in check.evidence
    assert "1 份历史 campaign" in check.evidence
    assert "没有一行同时写着" in check.evidence


def test_time_constant_estimate_scales_with_the_planned_blocks(out_dir: Path) -> None:
    """判据里的块数与秒数按当前计划生成，不再写死「112 块 → 2688 s」。"""
    settings = derive_settings(("square",), 1200.0)
    check = _by_key(_card(out_dir=out_dir, settings=settings))["time"]
    assert "本页 28 块 → 672 s" in check.evidence
    # 依据句保留：常数与实测的既有偏差仍是事实陈述，不因改动删掉。
    assert "112 块 → 2688 s" in check.evidence
    assert "2675.0 s" in check.evidence


# --------------------------------------------------------------------------- #
# ④ 模型新鲜度
# --------------------------------------------------------------------------- #
def _write_model(root: Path, correlations: list[float]) -> Path:
    (root / "fit").mkdir(parents=True)
    (root / "fit" / "result.json").write_text(json.dumps({
        "iteration": 3,
        "loss": 0.01,
        "timestamp_utc": "2026-10-28T09:00:00Z",
        "metrics": [{"correlation": value, "relative_l2": 0.1} for value in correlations],
    }), encoding="utf-8")
    return root


def test_model_is_unknown_without_a_directory(out_dir: Path) -> None:
    check = _by_key(_card(out_dir=out_dir))["model"]
    assert check.status == "unknown"
    assert "从零重训" in check.detail


def test_model_is_unknown_when_result_json_is_missing(out_dir: Path, tmp_path: Path) -> None:
    """只有 ``capture/`` 没有 ``fit/result.json`` 的目录**不是**一份可用模型。"""
    check = _by_key(_card(out_dir=out_dir, model_dir=tmp_path / "half"))["model"]
    assert check.status == "unknown"
    assert "fit/result.json" in check.detail or "fit" in check.detail


def test_model_warns_below_the_proposed_quality_line(out_dir: Path, tmp_path: Path) -> None:
    check = _by_key(_card(out_dir=out_dir,
                          model_dir=_write_model(tmp_path / "m", [0.5, 0.6])))["model"]
    assert check.status == "warn"
    assert f"{precheck.MODEL_QUALITY_PROPOSAL:g}" in check.detail
    assert "阈值待实测" in check.detail


def test_model_passes_above_the_proposed_quality_line(out_dir: Path, tmp_path: Path) -> None:
    check = _by_key(_card(out_dir=out_dir,
                          model_dir=_write_model(tmp_path / "m", [0.93, 0.91])))["model"]
    assert check.status == "pass"
    assert "0.920" in check.detail


# --------------------------------------------------------------------------- #
# ⑤ 饱和预估 / ⑥ 曝光收敛史
# --------------------------------------------------------------------------- #
def _write_manifest(campaign: Path, *, peak: float, full_scale: float = 4095.0) -> None:
    stage = campaign / "shape" / "square" / "1_geometric"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "manifest.json").write_text(json.dumps(
        [{"index": 1, "peak": peak, "full_scale": full_scale, "cv": 0.2}]),
        encoding="utf-8")


def test_saturation_is_unknown_without_history(out_dir: Path) -> None:
    check = _by_key(_card(out_dir=out_dir))["saturation"]
    assert check.status == "unknown"
    assert "无历史峰值" in check.detail


def test_saturation_warns_when_the_peak_sits_on_full_scale(out_dir: Path,
                                                          tmp_path: Path) -> None:
    campaign = tmp_path / "stamp" / "s1200"
    campaign.mkdir(parents=True)
    (campaign / "journal.json").write_text("[]", encoding="utf-8")
    _write_manifest(campaign, peak=4090.0)
    check = _by_key(_card(out_dir=out_dir, campaigns=[campaign]))["saturation"]
    assert check.status == "warn"
    assert f"{precheck.SATURATION_WARN_FRACTION:.0%}" in check.evidence
    assert "削顶" in check.detail


def test_saturation_passes_with_headroom(out_dir: Path, tmp_path: Path) -> None:
    campaign = tmp_path / "stamp" / "s1200"
    campaign.mkdir(parents=True)
    (campaign / "journal.json").write_text("[]", encoding="utf-8")
    _write_manifest(campaign, peak=1000.0)
    assert _by_key(_card(out_dir=out_dir, campaigns=[campaign]))["saturation"].status == "pass"


def test_exposure_is_unknown_without_history(out_dir: Path) -> None:
    check = _by_key(_card(out_dir=out_dir))["exposure"]
    assert check.status == "unknown"
    assert "无曝光收敛记录可查" in check.detail


def test_exposure_warns_once_a_journal_recorded_a_non_convergence(out_dir: Path,
                                                                  tmp_path: Path) -> None:
    campaign = tmp_path / "stamp" / "s1200"
    campaign.mkdir(parents=True)
    (campaign / "journal.json").write_text(
        json.dumps([{"stage": "replay", "error": "exposure did not converge"}]),
        encoding="utf-8")
    check = _by_key(_card(out_dir=out_dir, campaigns=[campaign]))["exposure"]
    assert check.status == "warn"
    assert "重测曝光阶梯" in check.detail


def test_exposure_passes_when_no_journal_mentions_it(out_dir: Path, tmp_path: Path) -> None:
    campaign = tmp_path / "stamp" / "s1200"
    campaign.mkdir(parents=True)
    (campaign / "journal.json").write_text(json.dumps([{"stage": "replay"}]), encoding="utf-8")
    check = _by_key(_card(out_dir=out_dir, campaigns=[campaign]))["exposure"]
    assert check.status == "pass"
    assert "没有不收敛记录" in check.detail


# --------------------------------------------------------------------------- #
# 勾选门禁
# --------------------------------------------------------------------------- #
def test_warnings_need_an_acknowledgement_each(out_dir: Path, tmp_path: Path) -> None:
    """红停项禁用按钮；黄项每一条都要人勾 —— 缺一条就不许开跑。"""
    campaign = tmp_path / "stamp" / "s1200"
    campaign.mkdir(parents=True)
    (campaign / "journal.json").write_text("[]", encoding="utf-8")
    _write_manifest(campaign, peak=4090.0)          # → 饱和 warn
    card = _card(out_dir=out_dir, campaigns=[campaign], budget_hours=0.1)  # → 时间 warn
    assert {check.key for check in card.warnings} == {"time", "saturation"}
    assert card.acknowledged({}) is False
    assert card.acknowledged({"time": True}) is False
    assert card.acknowledged({"time": True, "saturation": True}) is True


def test_unknown_checks_do_not_require_an_acknowledgement(out_dir: Path) -> None:
    """"不知道"不是"已知风险"：不阻断、也不静默算通过（它仍是 unknown 一档）。"""
    card = _card(out_dir=out_dir)
    assert card.warnings == ()
    assert card.blocking is False
    assert card.acknowledged({}) is True
    assert card.status == "unknown"
