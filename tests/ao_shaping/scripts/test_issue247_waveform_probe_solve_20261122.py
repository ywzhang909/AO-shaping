"""Issue #247 波形判别离线分析器的离线验证（无设备）。

覆盖：

1. ``--help`` 列出 ``--dir`` / ``--self-test`` / ``--overwrite``；
2. ``--self-test`` 无需设备即退出 0，且四种合成场景各得预期结论（含 held-out 反转）；
3. import 脚本不会加载任何相机/SLM 驱动；
4. 合成帧下逐门槛实测值正确（波形门、轴门、标度门、符号门、patch 门）；
5. 二值 grating 必须**同时**报告 ±1 两侧，不允许只挑一侧；
6. flat 漂移、饱和分别把结论压到 ``invalid_drift_or_saturation``；
7. 产物写盘、默认拒绝覆盖、绝不产出 ``issue247_axis_map*.json`` / ``*.npz``；
8. 缺失图样时相关门槛记为 skipped，不得被当成通过；
9. 脚本与测试文件无 BOM、以 LF 结尾；
10. role 解析（缺省/合法/非法）与旧 15 图样 JSON 的 gates/verdict 不变；
11. G1–G4 的 pass/fail/skipped、grating 双峰必须用 axis_angle_deg；
12. G1/G2 同时 pass 时 attribution_label 必须归因单族路径；
13. held-out 轴向准确率、逐族门槛与符号判定。

本测试只使用 ``tmp_path`` 与内存合成帧，不读 ``data/``、不碰设备。
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = importlib.import_module("scripts.slm.issue247_waveform_probe_solve_20261122")

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "scripts" / "slm" / "issue247_waveform_probe_solve_20261122.py"
TEST_PATH = ROOT / "tests" / "ao_shaping" / "scripts" / "test_issue247_waveform_probe_solve_20261122.py"

SCENES = (
    ("consistent", "waveform_independent_map_ok"),
    ("grating_rotated", "binary_grating_specific"),
    ("flat_drift", "invalid_drift_or_saturation"),
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )


def _analyze(scene: str, thresholds: dict[str, float] | None = None):
    capture = SCRIPT.build_self_test_capture(scene)
    return SCRIPT.analyze(capture, dict(SCRIPT.DEFAULT_THRESHOLDS) if thresholds is None else thresholds)


def _write_capture(target: Path, capture: SCRIPT.Capture) -> None:
    """按采集脚本的命名把合成帧与 plan.json 写到磁盘。

    ``summary.json`` 按实验系统 schema 写满量程（采集脚本会写 ``full_scale``），
    这样默认路径与真采集一致；需要测推断分支时由用例自行删掉它。
    """
    frames_dir = target / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for spec in capture.catalog:
        frame = capture.means[spec.pattern_id]
        for index in range(4):
            np.save(frames_dir / f"{index:04d}_{spec.pattern_id}.npy", frame)
    plan = {k: v for k, v in capture.plan.items() if k != "_capture_dir"}
    (target / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (target / "summary.json").write_text(
        json.dumps({"full_scale": 4095}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


# ---------------------------------------------------------------------------
# 1. CLI 形状
# ---------------------------------------------------------------------------
def test_help_lists_required_flags():
    result = _run("--help")
    assert result.returncode == 0, result.stderr
    for flag in ("--dir", "--self-test", "--overwrite"):
        assert flag in result.stdout


def test_dir_is_required_without_self_test():
    result = _run()
    assert result.returncode != 0
    assert "--dir" in result.stderr


def test_missing_plan_returns_two(tmp_path: Path):
    result = _run("--dir", str(tmp_path))
    assert result.returncode == 2
    assert "plan.json" in result.stderr


# ---------------------------------------------------------------------------
# 2. 自检
# ---------------------------------------------------------------------------
def test_self_test_exits_zero_and_reports_all_scenes():
    result = _run("--self-test")
    assert result.returncode == 0, result.stderr
    assert "all invariants passed" in result.stdout
    for scene, verdict in SCENES:
        assert f"scene={scene}" in result.stdout
        assert f"verdict={verdict}" in result.stdout
    assert "scene=held_out_axis_flipped" in result.stdout
    assert "held_out_gate_pass=False" in result.stdout


# ---------------------------------------------------------------------------
# 3. 驱动隔离
# ---------------------------------------------------------------------------
def test_importing_script_does_not_load_drivers():
    probe = (
        "import sys; import scripts.slm.issue247_waveform_probe_solve_20261122; "
        "tokens = ('daheng', 'gxipy', 'santec', '_slm_win'); "
        "print(sorted(n for n in sys.modules if any(t in n for t in tokens)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout


# ---------------------------------------------------------------------------
# 4. 合成场景的门槛实测值
# ---------------------------------------------------------------------------
def test_consistent_scene_gates_are_sane():
    analysis = _analyze("consistent")
    assert analysis["verdict"] == "waveform_independent_map_ok"
    assert analysis["failed_gates"] == []

    waveform_x = analysis["gates"]["waveform_gate"]["axes"]["x"]
    waveform_y = analysis["gates"]["waveform_gate"]["axes"]["y"]
    assert waveform_x["status"] == "independent"
    assert waveform_x["measured_diff_deg"] <= 2.0
    assert waveform_y["status"] == "independent"

    blaze = analysis["gates"]["axis_gate"]["blaze"]
    assert blaze["pass"] is True
    assert abs(blaze["measured_diff_deg"] - 90.0) < 1e-3
    assert abs(blaze["measured_diff_deg"] - 88.544) <= 2.0

    tilt = analysis["gates"]["axis_gate"]["tilt"]
    assert tilt["pass"] is True
    assert abs(tilt["measured_diff_deg"] - 90.0) < 1e-3

    for axis in ("x", "y"):
        scale = analysis["gates"]["scale_gate"]["axes"][axis]
        assert scale["pass"] is True
        assert abs(scale["measured_ratio"] - 2.0) < 1e-3

    sign_pairs = {(pair["family"], pair["axis"]) for pair in analysis["gates"]["sign_gate"]["pairs"]}
    assert {("blaze", "x"), ("blaze", "y")} <= sign_pairs
    for pair in analysis["gates"]["sign_gate"]["pairs"]:
        assert pair["pass"] is True
        assert abs(pair["measured_angle_deg"] - 180.0) < 1e-3

    patch_pairs = analysis["gates"]["patch_gate"]["pairs"]
    assert len(patch_pairs) == 2
    assert all(pair["pass"] is True for pair in patch_pairs)


def test_grating_reports_both_sides():
    analysis = _analyze("consistent")
    record = analysis["patterns"]["grating_x_p128"]
    assert len(record["lobes"]) == 2
    assert record["counter_lobe_missing"] is False
    angles = sorted(lobe["angle_deg"] for lobe in record["lobes"])
    assert abs(angles[0]) < 1e-3
    assert abs(angles[1] - 180.0) < 1e-3
    # 两侧位移大小必须一致，不得只报符合预期的那一侧。
    assert abs(record["lobes"][0]["dist"] - record["lobes"][1]["dist"]) < 1e-3


def test_grating_rotated_is_waveform_dependent():
    analysis = _analyze("grating_rotated")
    assert analysis["verdict"] == "binary_grating_specific"
    for axis in ("x", "y"):
        waveform = analysis["gates"]["waveform_gate"]["axes"][axis]
        assert waveform["status"] == "dependent"
    # 只二值 grating 异常时，blaze 自身门仍须通过。
    assert analysis["gates"]["axis_gate"]["blaze_pass"] is True
    assert analysis["gates"]["axis_gate"]["tilt_pass"] is True
    assert analysis["gates"]["sign_gate"]["pass"] is True


def test_flat_drift_scene_is_invalid():
    analysis = _analyze("flat_drift")
    assert analysis["verdict"] == "invalid_drift_or_saturation"
    assert analysis["gates"]["drift_gate"]["pass"] is False
    assert analysis["flat_reference"]["invalid_drift"] is True
    assert abs(analysis["flat_reference"]["max_drift_px"] - 5.0) < 0.5


def test_saturation_scene_is_invalid():
    capture = SCRIPT.build_self_test_capture("consistent")
    capture.frame_stats["blaze_x_plus"]["saturated"] = True
    analysis = SCRIPT.analyze(capture, dict(SCRIPT.DEFAULT_THRESHOLDS))
    assert analysis["verdict"] == "invalid_drift_or_saturation"
    assert analysis["gates"]["saturation_gate"]["pass"] is False
    assert analysis["gates"]["saturation_gate"]["saturated_patterns"] == ["blaze_x_plus"]


def test_missing_patterns_leave_gates_skipped_not_passed():
    # 真实路径：plan 声明了 blaze_y_minus，但帧没采到（capture.missing 记录）。
    capture = SCRIPT.build_self_test_capture("consistent")
    capture.means.pop("blaze_y_minus")
    capture.frame_stats.pop("blaze_y_minus")
    capture.missing = list(capture.missing) + ["blaze_y_minus"]
    analysis = SCRIPT.analyze(capture, dict(SCRIPT.DEFAULT_THRESHOLDS))
    # blaze_y 的 +/− 对缺了一半：这一对不许被当成通过，整门记 skipped。
    assert analysis["gates"]["sign_gate"]["pass"] is None
    assert len(analysis["gates"]["sign_gate"]["pairs"]) == 1
    assert analysis["gates"]["sign_gate"]["pairs"][0]["axis"] == "x"
    assert analysis["verdict"] == "map_not_reproduced"
    assert any("blaze_y_minus" in note for note in analysis["notes"])


def test_plan_threshold_override_is_snapshotted():
    plan = SCRIPT.build_self_test_capture("consistent").plan
    plan["analysis_thresholds"] = {"axis_separation_tol_deg": 1.0, "drift_max_px": 1.0}
    thresholds, overrides = SCRIPT.resolve_thresholds(plan)
    assert thresholds["axis_separation_tol_deg"] == 1.0
    assert thresholds["drift_max_px"] == 1.0
    assert set(overrides) == {"axis_separation_tol_deg", "drift_max_px"}
    # 默认值不受影响
    assert thresholds["sign_angle_tol_deg"] == SCRIPT.DEFAULT_THRESHOLDS["sign_angle_tol_deg"]


def test_classify_family_beats_misleading_kind_field():
    # 采集计划把 blaze_x_plus 的 kind 写成 "tilt"（同一构造函数），族必须由 id 决定。
    assert SCRIPT.classify_family("blaze_x_plus", "tilt") == "blaze"
    assert SCRIPT.classify_family("tilt_x_plus", "tilt") == "tilt"
    assert SCRIPT.classify_family("blaze_patch_x", "blaze_patch") == "blaze_patch"
    assert SCRIPT.classify_family("grating_x_p128", "grating") == "grating"


def test_axial_helpers():
    assert abs(SCRIPT.axial_mean_deg([0.0, 180.0])) < 1e-9
    assert abs(SCRIPT.axial_mean_deg([90.0, -90.0]) - 90.0) < 1e-9
    assert SCRIPT.axial_diff_deg(0.0, 90.0) == 90.0
    assert SCRIPT.axial_diff_deg(88.544, 90.0) < 2.0
    assert abs(SCRIPT.angle_between_deg((1.0, 0.0), (-1.0, 0.0)) - 180.0) < 1e-9
    assert abs(SCRIPT.angle_deg(1.0, 0.0) - 90.0) < 1e-9
    assert abs(SCRIPT.angle_deg(0.0, 1.0)) < 1e-9


# ---------------------------------------------------------------------------
# 5. 写盘、覆盖保护与边界
# ---------------------------------------------------------------------------
def test_write_artifacts_and_overwrite_guard(tmp_path: Path, capsys):
    capture = SCRIPT.build_self_test_capture("consistent")
    _write_capture(tmp_path, capture)

    exit_code = SCRIPT.main(["--dir", str(tmp_path)])
    assert exit_code == 0

    out_json = tmp_path / SCRIPT.ANALYSIS_JSON_NAME
    out_report = tmp_path / SCRIPT.REPORT_MD_NAME
    assert out_json.is_file()
    assert out_report.is_file()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "issue247_waveform_probe_analysis/1"
    assert payload["verdict"] == "waveform_independent_map_ok"
    assert set(payload["gates"]) >= {
        "drift_gate",
        "saturation_gate",
        "waveform_gate",
        "axis_gate",
        "scale_gate",
        "sign_gate",
        "patch_gate",
    }
    assert payload["full_scale"]["assumed_full_scale"] is False

    report = out_report.read_text(encoding="utf-8")
    assert "## 摘要" in report
    assert "## 逐图样测量" in report
    assert "## 逐门槛判定" in report
    assert "本结果不自动改映射、不写 slm_geometry_calib.npz。" in report
    assert "waveform_independent_map_ok" in report

    # 默认拒绝覆盖。
    second = SCRIPT.main(["--dir", str(tmp_path)])
    assert second == 3
    assert "--overwrite" in capsys.readouterr().err

    # 允许覆盖时可重跑。
    assert SCRIPT.main(["--dir", str(tmp_path), "--overwrite"]) == 0

    # 绝不产出冻结映射相关文件。
    assert not list(tmp_path.glob("issue247_axis_map*.json"))
    assert not list(tmp_path.glob("*.npz"))
    assert not list(tmp_path.glob("slm_geometry_calib*"))


def test_assumed_full_scale_is_annotated(tmp_path: Path, capsys):
    capture = SCRIPT.build_self_test_capture("consistent")
    _write_capture(tmp_path, capture)
    # 去掉任何满量程声明，验证会标注 assumed_full_scale=true。
    (tmp_path / "summary.json").unlink()
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["declared_constants"] = {}
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    assert SCRIPT.main(["--dir", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / SCRIPT.ANALYSIS_JSON_NAME).read_text(encoding="utf-8"))
    assert payload["full_scale"]["assumed_full_scale"] is True
    assert payload["full_scale"]["source"] == "inferred_from_frame_max"
    assert any("assumed_full_scale=true" in note for note in payload["notes"])
    # 满量程是推断的，饱和判据不可用，不得误触发。
    assert payload["gates"]["saturation_gate"]["pass"] is True
    assert payload["verdict"] == "waveform_independent_map_ok"


# ---------------------------------------------------------------------------
# 6. 文件形态
# ---------------------------------------------------------------------------
def test_source_files_have_no_bom_and_lf_endings():
    for path in (SCRIPT_PATH, TEST_PATH):
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), path
        assert b"\r\n" not in raw, path


# ---------------------------------------------------------------------------
# 7. 归因门与 held-out 独立验证（预注册 §3.2 / §3.3）
# ---------------------------------------------------------------------------
def _fit_parts(capture, analysis):
    fit_ids = {spec.pattern_id for spec in capture.catalog if spec.role == "fit"}
    fit_catalog = [spec for spec in capture.catalog if spec.role == "fit"]
    fit_records = {
        pid: record for pid, record in analysis["patterns"].items() if pid in fit_ids
    }
    return fit_catalog, fit_records


def _held_parts(capture, analysis):
    held_ids = {spec.pattern_id for spec in capture.catalog if spec.role == "held_out"}
    held_catalog = [spec for spec in capture.catalog if spec.role == "held_out"]
    held_records = {
        pid: record for pid, record in analysis["patterns"].items() if pid in held_ids
    }
    return held_catalog, held_records


def test_role_parsing_default_legal_and_invalid():
    plan = {
        "patterns": [
            {"id": "grating_x_p128", "kind": "grating"},
            {"id": "tilt_x_plus", "kind": "tilt", "role": "held_out"},
            {"id": "patch_x_plus_p128", "kind": "patch", "role": "fit"},
        ]
    }
    specs = SCRIPT.build_specs(plan)
    assert [spec.role for spec in specs] == ["fit", "held_out", "fit"]
    plan["patterns"][1]["role"] = "validation"
    with pytest.raises(ValueError, match="role"):
        SCRIPT.build_specs(plan)


def test_old_15_pattern_json_keeps_gates_and_verdict():
    """旧窗口没有 role 字段；新增产物不得改动旧 gates/verdict 的口径。"""
    old_dir = ROOT / "data" / "debug" / "hw_window_20261123_issue247_waveform_probe"
    if not (old_dir / SCRIPT.ANALYSIS_JSON_NAME).exists():
        pytest.skip(f"旧窗口产物不在本机 (硬件调试数据未入库): {old_dir}")
    payload = json.loads(
        (old_dir / SCRIPT.ANALYSIS_JSON_NAME).read_text(encoding="utf-8")
    )
    plan = json.loads((old_dir / "plan.json").read_text(encoding="utf-8"))
    specs = SCRIPT.build_specs(plan)
    assert len(specs) == 15
    assert all(spec.role == "fit" for spec in specs)
    drift = dict(payload["flat_reference"])
    drift["pass"] = drift["invalid_drift"] is False
    drift["pairwise_px"] = drift["pairwise_drift_px"]
    gates = SCRIPT._build_gates(specs, payload["patterns"], drift, payload["thresholds"])
    assert gates == payload["gates"]
    verdict = SCRIPT._decide_verdict(gates)
    assert verdict == payload["verdict"] == "patch_position_specific"

    attribution = SCRIPT._build_attribution_gates(specs, payload["patterns"])
    for key in (
        "G1_waveform_dependency",
        "G2_single_family_anomaly",
        "G3_axis_separation",
        "G4_sign_consistency",
        "attribution_label",
        "anomalous_family",
    ):
        assert key in attribution


def test_attribution_gates_pass_fail_and_skipped():
    consistent = _analyze("consistent")
    gates = consistent["attribution_gates"]
    # 三族在同面板轴上同轴：G1 fail（波形无关），G2 fail（无单族异常）。
    assert gates["G1_waveform_dependency"]["pass"] is False
    assert gates["G2_single_family_anomaly"]["pass"] is False
    assert gates["G3_axis_separation"]["pass"] is True
    # 旧自检场景的三族里没有成对的 ±，G4 必须 skipped 而不是 pass。
    assert gates["G4_sign_consistency"]["pass"] is None
    assert gates["attribution_label"] == "not_assessable"

    capture = SCRIPT.build_self_test_capture("consistent")
    capture.means.pop("blaze_patch_x")
    capture.frame_stats.pop("blaze_patch_x")
    capture.missing.append("blaze_patch_x")
    skipped = SCRIPT.analyze(capture, dict(SCRIPT.DEFAULT_THRESHOLDS))["attribution_gates"]
    assert skipped["G1_waveform_dependency"]["pass"] is None
    assert skipped["G2_single_family_anomaly"]["pass"] is None
    assert skipped["G3_axis_separation"]["families"]["blaze_patch"]["pass"] is None


def test_g1_and_g2_both_pass_prefers_single_family_attribution():
    analysis = _analyze("grating_rotated")
    gates = analysis["attribution_gates"]
    assert gates["G1_waveform_dependency"]["pass"] is True
    assert gates["G2_single_family_anomaly"]["pass"] is True
    assert gates["G2_single_family_anomaly"]["anomalous_family"] == "grating"
    assert gates["attribution_label"] == "该族生成或下发路径问题"
    assert gates["anomalous_family"] == "grating"
    assert gates["g1_role"] == "descriptive_only"
    # 逐门值必须在报告里同时列出，单一 label 不得盖住其他 fail。
    report = SCRIPT.build_report(
        analysis, SCRIPT.build_self_test_capture("grating_rotated").catalog
    )
    assert "## 归因门与 held-out 独立验证" in report
    assert "G2_single_family_anomaly > G1_waveform_dependency" in report
    assert "该族生成或下发路径问题" in report
    for gate in ("G1 波形依赖[x]", "G2 单族异常[grating]", "G3 轴分离[grating]"):
        assert gate in report


def test_grating_attribution_uses_axis_angle_deg_not_principal():
    capture = SCRIPT.build_self_test_capture("grating_rotated")
    analysis = SCRIPT.analyze(capture, dict(SCRIPT.DEFAULT_THRESHOLDS))
    record = dict(analysis["patterns"]["grating_x_p128"])
    assert len(record["lobes"]) == 2
    expected = float(record["axis_angle_deg"])
    record["principal"] = {**record["principal"], "angle_deg": 0.0}
    records = {
        pid: record if pid == "grating_x_p128" else value
        for pid, value in analysis["patterns"].items()
    }
    fit_catalog, _ = _fit_parts(capture, analysis)
    gates = SCRIPT._build_attribution_gates(fit_catalog, records)
    measured = gates["G1_waveform_dependency"]["axes"]["x"]["angles_deg"]["grating"]
    assert abs(abs(measured) - 90.0) < 1e-3
    assert abs(abs(expected) - 90.0) < 1e-3
    assert record["principal"]["angle_deg"] == 0.0


def test_g3_axis_routing_failure_and_g4_sign_failure():
    capture = SCRIPT.build_self_test_capture("held_out_axis_flipped")
    analysis = SCRIPT.analyze(capture, dict(SCRIPT.DEFAULT_THRESHOLDS))
    fit_catalog, fit_records = _fit_parts(capture, analysis)

    routing_records = dict(fit_records)
    routing_records["tilt_y_plus"] = {
        **fit_records["tilt_y_plus"],
        "axis_angle_deg": 0.0,
    }
    routing_records["tilt_y_minus"] = {
        **fit_records["tilt_y_minus"],
        "axis_angle_deg": 180.0,
    }
    routing = SCRIPT._build_attribution_gates(fit_catalog, routing_records)
    tilt = routing["G3_axis_separation"]["families"]["tilt"]
    assert tilt["axis_routing_failed"] is True
    assert routing["G3_axis_separation"]["pass"] is False
    assert "tilt" in routing["G3_axis_separation"]["axis_routing_failed_families"]

    sign_records = dict(fit_records)
    plus_angle = float(fit_records["tilt_x_plus"]["axis_angle_deg"])
    sign_records["tilt_x_minus"] = {
        **fit_records["tilt_x_minus"],
        "axis_angle_deg": plus_angle,
    }
    sign_gates = SCRIPT._build_attribution_gates(fit_catalog, sign_records)
    pairs = {
        (pair["family"], pair["axis"]): pair
        for pair in sign_gates["G4_sign_consistency"]["pairs"]
    }
    assert pairs[("tilt", "x")]["pass"] is False
    assert sign_gates["G4_sign_consistency"]["pass"] is False


def test_held_out_validation_rejects_axis_flip_and_accepts_match():
    capture = SCRIPT.build_self_test_capture("held_out_axis_flipped")
    analysis = SCRIPT.analyze(capture, dict(SCRIPT.DEFAULT_THRESHOLDS))
    validation = analysis["held_out_validation"]
    predictions = analysis["held_out_predictions"]
    assert validation["gate_pass"] is False
    assert validation["held_out_axis_accuracy"] == 0.0
    assert validation["axis_total_count"] == 10
    assert len(predictions) == 10
    assert all(not item["gate_pass"] for item in validation["per_family"].values())
    assert "grating_x_p192" in validation["held_out_ids"]
    assert "grating_x_p128" in validation["fit_ids"]
    assert all(pred["status"] == "assessed" for pred in predictions)
    assert all(pred["prediction_rule"] == "G3_all_pass" for pred in predictions)
    report = SCRIPT.build_report(analysis, capture.catalog)
    assert "held-out 预测与实测" in report
    assert "held_out_axis_accuracy = 0/10" in report
    assert "旧 `gates` / `verdict` 仅为兼容保存" in report

    # 把 held-out 的轴向角改成 fit 预测值：必须全过（符号仍用原始 lobes，不影响轴向门）。
    fit_catalog, fit_records = _fit_parts(capture, analysis)
    held_catalog, held_records = _held_parts(capture, analysis)
    fit_angles = SCRIPT._attribution_axis_angles(fit_catalog, fit_records)
    matched = dict(held_records)
    for spec in held_catalog:
        if spec.family == "flat":
            continue
        matched[spec.pattern_id] = {
            **matched[spec.pattern_id],
            "axis_angle_deg": fit_angles[spec.axis][spec.family]["angle_deg"],
        }
    ok_validation, ok_predictions = SCRIPT._build_held_out_validation(
        fit_catalog, fit_records, held_catalog, matched, analysis["attribution_gates"]
    )
    assert ok_validation["gate_pass"] is True
    assert ok_validation["held_out_axis_accuracy"] == 1.0
    assert all(item["gate_pass"] for item in ok_validation["per_family"].values())
    assert all(pred["axis_correct"] is True for pred in ok_predictions)


def test_measured_sign_uses_single_lobe_projection():
    def sign(record: dict, pattern_id: str, axis: str):
        return SCRIPT._measured_sign({pattern_id: record}, pattern_id, axis)

    single = {"lobes": [{"dr": 0.0, "dc": 96.0}]}
    assert sign(single, "grating_x_p192", "x") == 1
    single = {"lobes": [{"dr": 0.0, "dc": -96.0}]}
    assert sign(single, "tilt_x_minus_p192", "x") == -1
    single = {"lobes": [{"dr": 96.0, "dc": 0.0}]}
    assert sign(single, "tilt_y_plus_p192", "y") == 1
    double = {"lobes": [{"dr": 0.0, "dc": 96.0}, {"dr": 0.0, "dc": -96.0}]}
    assert sign(double, "grating_x_p192", "x") is None


def test_held_out_predictions_and_validation_written_to_json(tmp_path: Path):
    capture = SCRIPT.build_self_test_capture("held_out_axis_flipped")
    _write_capture(tmp_path, capture)
    assert SCRIPT.main(["--dir", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / SCRIPT.ANALYSIS_JSON_NAME).read_text(encoding="utf-8"))
    for key in (
        "attribution_gates",
        "held_out_validation",
        "held_out_predictions",
        "role_by_pattern",
    ):
        assert key in payload
    assert payload["role_by_pattern"]["grating_x_p192"] == "held_out"
    assert payload["role_by_pattern"]["grating_x_p128"] == "fit"
    assert payload["held_out_validation"]["gate_pass"] is False
