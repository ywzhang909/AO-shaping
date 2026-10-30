"""光路指纹：采集口径、逐维度打分与加载推荐 (免设备)。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ao_shaping.tools.slm.transfer.optical_session import (
    AUTO_APPLY_THRESHOLD,
    COMPAT_THRESHOLD,
    DIMENSION_WEIGHTS,
    FINGERPRINT_NAME,
    compatibility_score,
    compatibility_verdict,
    compute_optical_fingerprint,
    load_fingerprint,
    recommend_load,
    save_fingerprint,
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _session(tmp_path: Path) -> Path:
    """一条"跑完的"会话目录：每个数据来源各留一份最小产物。"""
    root = tmp_path / "session"
    _write(root / "journal.json", [
        {"stage": "plan", "status": "complete",
         "timestamp_utc": "2026-01-01T00:00:00+00:00",
         "plan": {"window_span": 820.0, "window_step": 4.0, "zero_radius": 0.0,
                  "period": 128.0, "control_grid": [640, 640], "fft_size": 1024}},
        {"stage": "geometry", "status": "complete", "zero_xy": [101.5, 208.25]},
    ])
    _write(root / "shape" / "square" / "capture" / "config.json", {"pixel_size_um": 3.45})
    _write(root / "shape" / "square" / "capture" / "manifest.json",
           [{"peak": 900.0, "full_scale": 4095.0, "exposure_ms": 0.6},
            {"peak": 1500.0, "full_scale": 4095.0, "exposure_ms": 0.6}])
    _write(root / "fit" / "provenance.json",
           {"grid_size": [320, 320], "fft_size": 1024, "origin": [50.0, 60.0],
            "ramp_matrix": [[1, 2], [3, 4]]})
    _write(root / "camera_probe" / "formats.json",
           {"observations": [{"format": "Mono8", "exposure_ms": 0.5},
                             {"format": "Mono12", "exposure_ms": 0.6,
                              "pixel_size_um": 3.45}]})
    _write(root / "optical_path_probe.json", {"verdict": {"state": "ok"}})
    return root


def test_fingerprint_collects_every_session_artifact(tmp_path):
    root = _session(tmp_path)
    fingerprint = compute_optical_fingerprint(root)

    assert fingerprint["created_at"] == "2026-01-01T00:00:00+00:00"
    assert fingerprint["window_span"] == 820.0
    assert fingerprint["window_step"] == 4.0
    assert fingerprint["zero_radius"] == 0.0
    assert fingerprint["period"] == 128.0
    assert fingerprint["zero_xy"] == [101.5, 208.25]
    # 最亮那一行给出峰值 / 满量程 / 曝光, 而不是第一行.
    assert fingerprint["peak_counts"] == 1500.0
    assert fingerprint["full_scale"] == 4095.0
    assert fingerprint["exposure_ms"] == 0.6
    assert fingerprint["camera_format"] == "Mono12"
    assert fingerprint["pixel_size_um"] == 3.45
    # plan 的 control_grid 先被读到, provenance 的 grid_size 不覆盖它.
    assert fingerprint["control_grid"] == [640, 640]
    assert fingerprint["fft_size"] == 1024
    assert fingerprint["optics_validity"] == "ok"
    assert fingerprint["origin_px"] == [50.0, 60.0]
    expected = hashlib.sha256(
        json.dumps([[1, 2], [3, 4]], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert fingerprint["ramp_matrix_sha256"] == expected


def test_a_missing_directory_leaves_the_fields_empty(tmp_path):
    fingerprint = compute_optical_fingerprint(tmp_path / "never_written")

    assert fingerprint["window_span"] is None
    assert fingerprint["zero_xy"] is None
    assert fingerprint["peak_counts"] is None
    assert fingerprint["optics_validity"] is None
    # created_at 总是有值: 它是"这份指纹是什么时候被读出来的", 不是会话时间.
    assert fingerprint["created_at"]


def test_overrides_win_over_the_collected_values(tmp_path):
    root = _session(tmp_path)
    fingerprint = compute_optical_fingerprint(root, exposure_ms=1.25, zero_xy=(7.0, 8.0))

    assert fingerprint["exposure_ms"] == 1.25
    assert fingerprint["zero_xy"] == [7.0, 8.0]
    # 没被覆盖的字段照旧来自产物.
    assert fingerprint["window_span"] == 820.0


def test_fingerprint_round_trips_through_the_session_directory(tmp_path):
    fingerprint = compute_optical_fingerprint(_session(tmp_path))
    path = save_fingerprint(fingerprint, tmp_path)

    assert path == tmp_path / FINGERPRINT_NAME
    assert load_fingerprint(tmp_path) == fingerprint
    assert load_fingerprint(path) == fingerprint
    assert load_fingerprint(tmp_path / "absent.json") is None


def test_identical_fingerprints_score_one(tmp_path):
    fingerprint = compute_optical_fingerprint(_session(tmp_path))
    score, dimensions = compatibility_score(fingerprint, fingerprint)

    assert score == pytest.approx(1.0)
    assert {item["name"] for item in dimensions} == set(DIMENSION_WEIGHTS)
    assert all(item["value"] == 1.0 for item in dimensions)


def test_missing_fields_are_neutral_rather_than_incompatible():
    score, dimensions = compatibility_score({}, {})

    assert score == pytest.approx(0.5)
    assert all(item["value"] == 0.5 for item in dimensions)
    assert all("字段缺失" in item["reason"] for item in dimensions)


def test_an_exposure_mismatch_lowers_the_weighted_score(tmp_path):
    fingerprint = compute_optical_fingerprint(_session(tmp_path))
    brighter = dict(fingerprint, exposure_ms=1.2)
    score, dimensions = compatibility_score(fingerprint, brighter)

    exposure = next(item for item in dimensions if item["name"] == "曝光")
    assert exposure["value"] == pytest.approx(0.5)
    # 曝光权重 0.20, 其余维度相同 ⇒ 1 - 0.20 * 0.5
    assert score == pytest.approx(0.9)


def test_recommend_load_picks_the_compatible_candidate(tmp_path):
    fingerprint = compute_optical_fingerprint(_session(tmp_path))
    result = recommend_load(fingerprint, [
        {"file": "a/frozen_parameters.json", "fingerprint": dict(fingerprint)},
        {"file": "b/frozen_parameters.json", "fingerprint": dict(fingerprint)},
    ])

    assert result["threshold"] == COMPAT_THRESHOLD
    assert result["recommended"] is not None
    assert result["recommended"]["score"] == pytest.approx(1.0)
    assert result["recommended"]["rejected"] is False
    assert [row["score"] for row in result["ranked"]] == [1.0, 1.0]


def test_hard_rejections_stay_in_the_ranking_with_reasons(tmp_path):
    fingerprint = compute_optical_fingerprint(_session(tmp_path))
    result = recommend_load(fingerprint, [
        {"file": "bad_optics", "fingerprint": dict(fingerprint, optics_validity="low_snr")},
        {"file": "stale", "fingerprint": dict(fingerprint), "stale": True},
        {"file": "moved", "fingerprint": dict(fingerprint, zero_xy=[1001.5, 208.25])},
        {"file": "fine", "fingerprint": dict(fingerprint)},
    ])

    rejected = {row["candidate"]["file"]: row for row in result["ranked"] if row["rejected"]}
    assert set(rejected) == {"bad_optics", "stale", "moved"}
    assert any("光路探针" in reason for reason in rejected["bad_optics"]["reasons"])
    assert any("stale" in reason for reason in rejected["stale"]["reasons"])
    assert any("零级位移" in reason for reason in rejected["moved"]["reasons"])
    assert result["recommended"]["candidate"]["file"] == "fine"


def test_a_candidate_below_the_threshold_is_ranked_but_not_recommended(tmp_path):
    fingerprint = compute_optical_fingerprint(_session(tmp_path))
    result = recommend_load(fingerprint,
                            [{"file": "dimmer", "fingerprint": dict(fingerprint,
                                                                    exposure_ms=1.2)}],
                            min_score=0.95)

    row = result["ranked"][0]
    assert row["rejected"] is False
    assert row["score"] == pytest.approx(0.9)
    assert any("低于阈值" in reason for reason in row["reasons"])
    assert result["recommended"] is None


def test_a_candidate_without_a_fingerprint_is_rejected():
    result = recommend_load({}, [{"file": "legacy"}])

    row = result["ranked"][0]
    assert row["rejected"] is True
    assert row["score"] == 0.0
    assert any("缺少光路指纹" in reason for reason in row["reasons"])


def test_without_a_current_fingerprint_nothing_is_scored(tmp_path):
    fingerprint = compute_optical_fingerprint(_session(tmp_path))
    result = recommend_load(None, [{"file": "any", "fingerprint": fingerprint}])

    row = result["ranked"][0]
    assert row["score"] == 0.0
    assert row["rejected"] is False
    assert row["dimensions"] == []
    assert any("未提供当前光路指纹" in reason for reason in row["reasons"])
    assert result["recommended"] is None


def test_the_verdict_splits_the_two_thresholds():
    """两档阈值：可用下限 0.60 与自动应用下限 0.80，三档结论由同一函数给出。"""
    assert COMPAT_THRESHOLD < AUTO_APPLY_THRESHOLD

    assert compatibility_verdict(0.85) == "推荐"
    assert compatibility_verdict(0.70) == "谨慎"
    assert compatibility_verdict(0.50) == "不推荐"
    # 阈值本身落在各自档内的一侧（含边界）。
    assert compatibility_verdict(AUTO_APPLY_THRESHOLD) == "推荐"
    assert compatibility_verdict(COMPAT_THRESHOLD) == "谨慎"
