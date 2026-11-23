"""Issue #247 波形判别离线分析器的离线验证（无设备）。

覆盖：

1. ``--help`` 列出 ``--dir`` / ``--self-test`` / ``--overwrite``；
2. ``--self-test`` 无需设备即退出 0，且三种合成场景各得预期结论；
3. import 脚本不会加载任何相机/SLM 驱动；
4. 合成帧下逐门槛实测值正确（波形门、轴门、标度门、符号门、patch 门）；
5. 二值 grating 必须**同时**报告 ±1 两侧，不允许只挑一侧；
6. flat 漂移、饱和分别把结论压到 ``invalid_drift_or_saturation``；
7. 产物写盘、默认拒绝覆盖、绝不产出 ``issue247_axis_map*.json`` / ``*.npz``；
8. 缺失图样时相关门槛记为 skipped，不得被当成通过；
9. 脚本与测试文件无 BOM、以 LF 结尾。

本测试只使用 ``tmp_path`` 与内存合成帧，不读 ``data/``、不碰设备。
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

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

    ``summary.json`` 按真机 schema 写满量程（采集脚本会写 ``full_scale``），
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
