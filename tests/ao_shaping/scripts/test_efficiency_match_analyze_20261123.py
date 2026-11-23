"""``scripts/slm/efficiency_match_analyze_20261123.py`` 的离线测试。

只测纯逻辑，不打开硬件、不读真设备数据：
- 四个合成场景（匹配 / 不匹配 / blaze 泄漏超预期 / 缺暗帧）的 verdict 与 reason；
- 关键数值（eta_s / eta_d / d）与合成真值在容差内；
- bootstrap 固定种子可复现、换种子会变；
- ``write_outputs`` 默认拒绝覆盖，``--overwrite`` 放行；
- 大质心位移只记 ``drift_warning``，不作废；
- CLI：``--dir`` 缺失报错、正常路径写产物并返回 0。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/slm/efficiency_match_analyze_20261123.py"
)


def _load():
    """按路径加载脚本模块。"""
    spec = importlib.util.spec_from_file_location(
        "efficiency_match_analyze_20261123_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


em = _load()


def _dataset(tmp_path: Path, scenario: str) -> dict:
    """在临时目录里合成一个采集目录并返回真值。"""
    return em.build_synthetic_dataset(tmp_path / scenario, scenario)


def _analyze(tmp_path: Path, scenario: str, **kwargs) -> dict:
    """合成 + 分析一步到位。"""
    root = tmp_path / scenario
    em.build_synthetic_dataset(root, scenario)
    return em.analyze_dir(root, **kwargs)


# ---------------------------------------------------------------------------
# 四个预注册场景
# ---------------------------------------------------------------------------
def test_matched_scenario() -> None:
    """eta_s≈eta_d≈0.4 时判 matched，数值在容差内。"""
    result = em.run_self_test()["matched"]
    assert result["verdict"] == "matched"
    rounds = result["stats"]["rounds"]
    assert len(rounds) >= 2
    for row in rounds:
        assert abs(row["eta_s"] - 0.4) <= 0.02
        assert abs(row["eta_d"] - 0.4) <= 0.02
    assert abs(result["stats"]["mean_d"]) <= 0.05
    assert result["stats"]["ci95_high"] <= 0.08
    assert (result["stats"]["sd_eta_s"] or 0.0) <= 0.03
    assert (result["stats"]["sd_eta_d"] or 0.0) <= 0.03


def test_mismatched_scenario() -> None:
    """差 0.2 时判 mismatched，reason 带出两条越界。"""
    result = em.run_self_test()["mismatched"]
    assert result["verdict"] == "mismatched"
    stats = result["stats"]
    assert abs(stats["mean_d"] - 0.2) <= 0.03
    assert "abs(mean_d)" in result["reason"]
    assert "ci95_high" in result["reason"]
    for row in stats["rounds"]:
        assert abs(row["eta_s"] - 0.4) <= 0.02
        assert abs(row["eta_d"] - 0.2) <= 0.02


def test_asymmetric_scenario_leakage_diagnostic_only() -> None:
    """±1 级严重不对称（泄漏超预期）只算诊断，不判 undeterminable。"""
    result = em.run_self_test()["asymmetric"]
    assert result["verdict"] == "matched"
    assert result["reason"] == "preregistered_bounds_all_met"
    assert "pm1_order_asymmetry_exceeds_limit" not in result["reasons"]
    diag = [
        row["blaze_leakage_diagnostic"]
        for row in result["frames"]
        if row["role"] == "blaze" and row["blaze_leakage_diagnostic"]
    ]
    assert diag, "blaze 泄漏诊断字段必须存在"
    assert all(d["within_expectation"] is False for d in diag)
    assert all(d["gm1_over_g1"] > d["expected_gm1_over_g1_max"] for d in diag)
    # 主测量本身仍然算出来了（0.4/0.4），不被泄漏诊断影响。
    for row in result["stats"]["rounds"]:
        assert abs(row["eta_d"] - 0.4) <= 0.02


def test_missing_dark_scenario_is_undeterminable() -> None:
    """dark_source=missing 且无暗帧 -> undeterminable，不做任何光度学。"""
    result = em.run_self_test()["missing_dark"]
    assert result["verdict"] == "undeterminable"
    assert result["reason"] == "missing_same_exposure_dark"
    assert result["dark"]["source"] == "missing"
    assert result["stats"]["mean_d"] is None
    assert result["frames"] == []


# ---------------------------------------------------------------------------
# 数值口径
# ---------------------------------------------------------------------------
def test_eta_values_match_targets(tmp_path: Path) -> None:
    """效率真值 vs 分析值（含 S0 扣除与完整 F0 分母）。"""
    info = _dataset(tmp_path, "matched")
    result = em.analyze_dir(tmp_path / "matched")
    row = next(r for r in result["frames"] if r["role"] == "shaping")
    # eta_s = (S - S0) / F0，合成时 S 的 patch 分量就是 eta_s_target * F0。
    assert abs(row["eta_s"] - info["eta_s_target"]) <= 0.02
    blaze = next(r for r in result["frames"] if r["role"] == "blaze")
    assert abs(blaze["eta_d"] - info["eta_d_target"]) <= 0.02
    # 原始 S/F0 只是对照，但也被报出来。
    assert row["raw_s_over_f0"] is not None


def test_f0_and_s0_photometry(tmp_path: Path) -> None:
    """F0 = R0 圆盘暗减积分均值；S0 = 同方窗积分均值。"""
    info = _dataset(tmp_path, "matched")
    result = em.analyze_dir(tmp_path / "matched")
    photo = result["photometry"]
    assert abs(photo["f0_mean"] - info["f0"]) / info["f0"] <= 0.01
    assert abs(photo["s0_mean"] - info["s0"]) / info["s0"] <= 0.01
    # 三类 flat 帧都在逐帧表里。
    roles = {r["role"] for r in result["frames"]}
    assert {"flat_pre", "flat_mid", "flat_post"} <= roles


def test_orders_integrated_at_ring(tmp_path: Path) -> None:
    """+1 级峰落在 SEP 环带内，-1 在反方向。"""
    result = _analyze(tmp_path, "matched")
    th = result["thresholds"]
    for row in result["frames"]:
        if row["role"] != "blaze":
            continue
        center = row["reference_center"]
        for key in ("plus1_peak", "minus1_peak"):
            peak = row[key]
            assert peak is not None
            dist = np.hypot(peak[0] - center[0], peak[1] - center[1])
            assert th["separation_px"] - th["separation_tolerance_px"] <= dist
            assert dist <= th["separation_px"] + th["separation_tolerance_px"]
        assert row["g1"] > 0 and row["gm1"] > 0
        diag = row["blaze_leakage_diagnostic"]
        assert diag is not None
        assert diag["gm1_over_g1"] <= th["blaze_leakage_expected"]
        assert diag["within_expectation"] is True


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_seed_reproducible(tmp_path: Path) -> None:
    """同种子同结果；种子写进结果。"""
    first = _analyze(tmp_path, "matched", bootstrap=2000, seed=0)
    again = _analyze(tmp_path, "matched", bootstrap=2000, seed=0)
    assert first["stats"]["ci95_low"] == again["stats"]["ci95_low"]
    assert first["stats"]["ci95_high"] == again["stats"]["ci95_high"]
    assert first["stats"]["mean_d"] == again["stats"]["mean_d"]
    assert first["bootstrap"]["seed"] == 0
    assert first["bootstrap"]["iterations"] == 2000


def test_bootstrap_seed_enters_resampling() -> None:
    """种子确实进入重采样：轮数足够多时换种子 CI 会动。

    合成目录只有 4 个重复轮，bootstrap 均值只有 35 个离散取值，分位数
    对种子不敏感（这本身是合理的统计行为）；这里用 30 轮构造数据直接
    调 ``bootstrap_difference`` 验证种子改变了重采样。
    """
    data_rng = np.random.default_rng(123)
    rounds = [
        {"pass": 1, "d": float(v)}
        for v in data_rng.uniform(-0.05, 0.05, size=30)
    ]
    first = em.bootstrap_difference(rounds, 2000, 0)
    other = em.bootstrap_difference(rounds, 2000, 7)
    # 均值来自数据本身，与种子无关。
    assert first["mean_d"] == other["mean_d"]
    assert first["n_rounds"] == 30
    assert first["seed"] == 0 and other["seed"] == 7
    assert first["ci95_low"] < first["mean_d"] < first["ci95_high"]
    # 区间有限且换了种子后至少一个端点不同。
    assert other["ci95_low"] < other["mean_d"] < other["ci95_high"]
    moved = (
        other["ci95_low"] != first["ci95_low"]
        or other["ci95_high"] != first["ci95_high"]
    )
    assert moved


# ---------------------------------------------------------------------------
# 门控行为
# ---------------------------------------------------------------------------
def test_drift_warning_recorded_not_fatal(tmp_path: Path) -> None:
    """flat 质心位移 > 2 px -> 只记告警 + verdict 说明，不作废。"""
    root = tmp_path / "drifty"
    em.build_synthetic_dataset(root, "matched")
    # 把两张 flat 的光斑挪 5 px（mid 与 post）。
    for name in root.glob("raw/*_flat_mid.npy"):
        frame = np.asarray(np.load(name), dtype=np.float64)
        np.save(name, np.roll(frame, 5, axis=0).astype(np.uint16))
    result = em.analyze_dir(root)
    drift = result["drift"]
    assert drift["drift_warnings"], "质心位移应触发 drift_warning"
    assert drift["drift_rounds"] <= 1
    assert result["gates"]["drift_rounds"] is False
    assert "drift_warning" in result["reason"]
    # F0 相对变化超过 5% 的轮才算漂移轮。
    assert result["verdict"] in ("matched", "mismatched", "undeterminable")


def test_saturation_gate(tmp_path: Path) -> None:
    """积分帧饱和（全幅打满）-> undeterminable，首要理由是饱和门。"""
    root = tmp_path / "saturated"
    em.build_synthetic_dataset(root, "matched")
    target = next(root.glob("raw/*_shaping_target.npy"))
    frame = np.load(target)
    np.save(target, np.full(frame.shape, 4095, np.uint16))  # 全幅打满
    result = em.analyze_dir(root)
    assert result["gates"]["saturated_frame"] is True
    assert result["verdict"] == "undeterminable"
    # 全幅打满同时会让效率越界，所以 reasons 可能带第二条；首要理由必须是饱和。
    assert result["reason"] == "integration_frame_saturated"
    assert result["reasons"][0] == "integration_frame_saturated"


def test_hash_mismatch_gate(tmp_path: Path) -> None:
    """journal 哈希与 phases/hashes.json 不一致 -> undeterminable。"""
    root = tmp_path / "badhash"
    em.build_synthetic_dataset(root, "matched")
    journal = root / "journal.jsonl"
    rows = [json.loads(line) for line in journal.read_text("utf-8").splitlines()]
    rows[0]["phase_hash"] = "sha256:wrong"
    journal.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    result = em.analyze_dir(root)
    assert result["gates"]["hash_or_device_mismatch"] is True
    assert result["reason"] == "phase_hash_or_device_identity_mismatch"


def test_out_of_range_efficiency_gate(tmp_path: Path) -> None:
    """效率越界（<0 或 >1）-> undeterminable，不做物理解释。"""
    root = tmp_path / "oor"
    em.build_synthetic_dataset(root, "mismatched")
    # 把 F0 压小到让 eta_s > 1：所有 flat 只剩暗底。
    for name in root.glob("raw/*flat*.npy"):
        frame = np.asarray(np.load(name), dtype=np.float64)
        np.save(name, (frame * 0.05).astype(np.uint16))
    result = em.analyze_dir(root)
    assert result["gates"]["energy_out_of_range"] is True
    assert result["reason"] == "out_of_range_energy_check_background_or_saturation"


# ---------------------------------------------------------------------------
# 产物与 CLI
# ---------------------------------------------------------------------------
def test_write_outputs_refuses_overwrite(tmp_path: Path) -> None:
    """默认拒绝覆盖已有产物；--overwrite 放行。"""
    root = tmp_path / "write"
    em.build_synthetic_dataset(root, "matched")
    result = em.analyze_dir(root)
    analysis, report = em.write_outputs(root, result, overwrite=False)
    assert analysis.is_file() and report.is_file()
    try:
        em.write_outputs(root, result, overwrite=False)
        raise AssertionError("应拒绝覆盖")
    except FileExistsError:
        pass
    em.write_outputs(root, result, overwrite=True)


def test_report_content(tmp_path: Path) -> None:
    """EM_REPORT.md 有中文各节 + 不改标定的声明。"""
    root = tmp_path / "report"
    em.build_synthetic_dataset(root, "matched")
    result = em.analyze_dir(root)
    report = em.render_report(result)
    for section in ("摘要", "口径", "逐帧表", "逐 pass", "判据表", "结论"):
        assert section in report
    assert "不改标定、不解禁任何门控" in report
    assert result["schema"] == "efficiency_match_analysis_v1"


def test_cli_requires_dir() -> None:
    """--dir 缺失时 argparse 报错退出码 2。"""
    try:
        em.main([])
        raise AssertionError("应报错")
    except SystemExit as exc:
        assert exc.code == 2


def test_cli_writes_outputs(tmp_path: Path, capsys) -> None:
    """正常路径返回 0 并写两个产物。"""
    root = tmp_path / "cli"
    em.build_synthetic_dataset(root, "matched")
    code = em.main(["--dir", str(root)])
    assert code == 0
    assert (root / "analysis.json").is_file()
    assert (root / "EM_REPORT.md").is_file()
    out = capsys.readouterr().out
    assert "verdict=matched" in out


def test_cli_refuses_second_run(tmp_path: Path) -> None:
    """重复跑同一个目录且不加 --overwrite -> 退出码 3。"""
    root = tmp_path / "cli2"
    em.build_synthetic_dataset(root, "matched")
    assert em.main(["--dir", str(root)]) == 0
    assert em.main(["--dir", str(root)]) == 3
    assert em.main(["--dir", str(root), "--overwrite"]) == 0


def test_analysis_json_serializable(tmp_path: Path) -> None:
    """产物 JSON 可序列化且含种子/阈值快照。"""
    result = _analyze(tmp_path, "matched")
    text = json.dumps(result, ensure_ascii=False)
    loaded = json.loads(text)
    assert loaded["bootstrap"]["seed"] == 0
    assert loaded["thresholds"]["r0_px"] == 90.0
    assert loaded["thresholds"]["match_abs_d"] == 0.05
    assert loaded["thresholds"]["roi_frozen_source"] == "frozen_mask"


def test_thresholds_from_config_override() -> None:
    """frozen_mask 优先于内置默认值。"""
    config = {
        "frozen_mask": {"r0": 120.0, "side": 80.0, "sep": 500.0},
        "sep": 999.0,  # 顶层应被 frozen_mask 压过
    }
    th = em.thresholds_from_config(config)
    assert th.r0 == 120.0
    assert th.side == 80.0
    assert th.sep == 500.0
    # 缺项回默认。
    assert th.match_abs_d == 0.05
    assert th.blaze_leakage_expected == 0.05


def test_thresholds_from_config_roi_frozen() -> None:
    """采集端 ``roi_frozen`` 块显式映射到分析口径，不回退过期计划默认值。"""
    config = {
        "roi_frozen": {
            "r0_px": 48.0,
            "side_px": 61.2,
            "separation_seed_px": 59.1,
            "separation_search_px": 25.0,
            "r1_px": 60.0,
            "zero_order_exclusion_px": 48.0,
            "saturation_fraction": 0.9,
            "blaze_period_panel_px": 128.0,
            "blaze_axis": "panel_x",
            "geometry": {"k_mean": 7563.9, "mirror": False},
        }
    }
    th = em.thresholds_from_config(config)
    assert th.r0 == 48.0
    assert th.side == 61.2
    assert th.sep == 59.1
    assert th.tol == 25.0
    assert th.r1 == 60.0
    assert th.zero_exclusion == 48.0
    assert th.saturation_fraction == 0.9
    # 过期计划默认值（sep=471、r0=90）不得回流。
    assert th.sep != 471.0 and th.r0 != 90.0
    assert th.roi_frozen_source == "roi_frozen"
    assert th.roi_frozen_audit["blaze_period_panel_px"] == 128.0
    assert th.roi_frozen_audit["blaze_axis"] == "panel_x"
    assert th.roi_frozen_audit["geometry"]["k_mean"] == 7563.9


def test_frames_built_from_journal(tmp_path: Path) -> None:
    """journal 驱动帧列表：16 帧、两遍、角色齐全。"""
    root = tmp_path / "layout"
    em.build_synthetic_dataset(root, "matched")
    config = em.read_json(root / "config.json")
    journal = em.read_journal(root / "journal.jsonl")
    frames = em.build_frames(root, journal, config)
    assert len(frames) == 16
    assert {f.pass_index for f in frames} == {1, 2}
    roles = [f.role for f in frames if f.pass_index == 1]
    assert roles == [
        "flat_pre",
        "blaze",
        "shaping",
        "dark",
        "flat_mid",
        "blaze",
        "shaping",
        "flat_post",
    ]
    # blaze/shaping 各遍出现两次，ordinal 区分。
    blazes = [f for f in frames if f.role == "blaze" and f.pass_index == 1]
    assert sorted(f.ordinal for f in blazes) == [0, 1]


# ---------------------------------------------------------------------------
# 真实采集端命名：``{frame_index:04d}_{frame_id}.npy``（2026-11-23 05:07 产物）
# ---------------------------------------------------------------------------
def _real_naming_dataset(tmp_path: Path, scenario: str = "matched") -> Path:
    """合成后再改写成真实采集端命名，返回目录。

    与真实产物的差异对齐点：帧行只有 ``id`` + ``frame_index``（没有
    ``raw_file``/``file``/``path``/``npy``）；dark / dark_r2 共享一份
    ``raw/dark_library_0.04ms.npy``；journal 末尾混入 3 行 cooldown（未知 id）。
    """
    root = tmp_path / "real_naming"
    em.build_synthetic_dataset(root, scenario)
    raw = root / "raw"
    journal_path = root / "journal.jsonl"
    entries = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    shared_dark = raw / "dark_library_0.04ms.npy"
    rewritten: list[dict] = []
    for index, entry in enumerate(entries):
        frame_id = entry["id"]
        source = raw / entry["raw_file"]
        if frame_id.removesuffix("_r2") == "dark":
            # 相机暗帧库命中后复制同一份，两个 dark 槽位没有独立帧。
            if not shared_dark.is_file():
                shared_dark.write_bytes(source.read_bytes())
            source.unlink()
        else:
            source.rename(raw / f"{index:04d}_{frame_id}.npy")
        rewritten.append(
            {
                "id": frame_id,
                "frame_index": index,
                "timestamp_utc": entry["timestamp"],
                "exposure_ms": entry["exposure_ms"],
                "saturated_pixels": entry["saturated_count"],
                "phase_hash": entry["phase_hash"],
            }
        )
    for offset, cooled in enumerate(("dark", "flat_post", "dark_r2"), start=1):
        rewritten.append(
            {
                "id": f"cooldown_after_{cooled}",
                "event": "cooldown",
                "timestamp_utc": 1_700_000_000.0 + 1000.0 + offset,
            }
        )
    journal_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in rewritten) + "\n",
        encoding="utf-8",
    )
    return root


def test_resolve_raw_path_uses_frame_index(tmp_path: Path) -> None:
    """同 id 两遍：frame_index 各自解析到自己的文件，而不是 glob 猜一个。"""
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("0001_blaze_grating.npy", "0005_blaze_grating.npy"):
        (raw / name).write_bytes(b"x")
    first = {"id": "blaze_grating", "frame_index": 1}
    fifth = {"id": "blaze_grating", "frame_index": 5}
    assert (
        em._resolve_raw_path(raw, first, "blaze_grating").name
        == "0001_blaze_grating.npy"
    )
    assert (
        em._resolve_raw_path(raw, fifth, "blaze_grating").name
        == "0005_blaze_grating.npy"
    )


def test_resolve_raw_path_prefixed_fallback_direction(tmp_path: Path) -> None:
    """无 frame_index 时，兜底 glob 的序号在 id 之前（原来写反永远不命中）。"""
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("0003_flat_post.npy", "0007_flat_post.npy"):
        (raw / name).write_bytes(b"x")
    resolved = em._resolve_raw_path(raw, {"id": "flat_post"}, "flat_post")
    assert resolved is not None
    assert resolved.name == "0003_flat_post.npy"


def test_real_capture_naming_resolves_every_frame(tmp_path: Path) -> None:
    """真实命名下 14 个图像帧全部解析，dark_r2 回退到共享暗帧库且只算 1 张。"""
    root = _real_naming_dataset(tmp_path)
    result = em.analyze_dir(root)

    rows = result["frames"]
    assert len(rows) == 16
    image_rows = [row for row in rows if row["role"] != "dark"]
    assert len(image_rows) == 14
    # 两遍同 id 的 blaze 各自落到自己的文件（序号区分），不共用一份。
    blaze_files = [row["file"] for row in rows if row["role"] == "blaze"]
    assert len(blaze_files) == 4
    assert len(set(blaze_files)) == 4
    assert {
        name.split("_", 1)[1] for name in blaze_files
    } == {"blaze_grating.npy", "blaze_grating_r2.npy"}
    # dark / dark_r2 都解析到共享暗帧库。
    dark_rows = [row for row in rows if row["role"] == "dark"]
    assert len(dark_rows) == 2
    assert {row["file"] for row in dark_rows} == {"dark_library_0.04ms.npy"}
    assert result["dark"]["source"] == "capture"
    assert result["dark"]["frames"] == ["dark_library_0.04ms.npy"]
    assert "1/1" in result["dark"]["note"]
    # 全链路仍然算得出效率，不是「跳过帧」的退化结果。
    assert result["verdict"] == "matched"
    assert result["gates"]["plus1_not_locatable"] is False
    assert all(row["g1"] is not None for row in rows if row["role"] == "blaze")
