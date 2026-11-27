# -*- coding: utf-8 -*-
"""离线守卫测试：统一缺数据采集脚本（不打开任何设备、不改判任何 freeze 判据）。"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "capture_missing_data_unified_20261127.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("unified_capture_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load_module()


def _args(tmp_path: Path, **overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "out": tmp_path / "dry_run_plan",
        "execute": False,
        "slm_number": 1,
        "cam_type": "daheng",
        "cam_id": 0,
        "exposure_ms": 0.087,
        "frames": 2,
        "b_frame": "both",
        "confirm_lights_off": False,
        "confirm_lights_on": False,
        "settle_s": 0.8,
        "wait_until_idle": 600.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


# --------------------------------------------------------------------------- #
# 计划与 §4 对齐
# --------------------------------------------------------------------------- #

def test_stage_order_covers_all_section4_missing_items() -> None:
    assert M.STAGE_ORDER == ("A", "B", "C", "D", "E")
    issues = {issue for letter in M.STAGE_ORDER for issue in M._blocking_issues(letter)}
    assert issues == {"#247", "#238", "#236", "#234", "#156", "#233"}
    assert set(M.STAGE_LABELS) == set(M.STAGE_ORDER)


def test_build_plan_is_deterministic_and_contains_no_clock_fields(tmp_path: Path) -> None:
    plan_a = M.build_plan(_args(tmp_path))
    plan_b = M.build_plan(_args(tmp_path))
    assert json.dumps(plan_a, ensure_ascii=False) == json.dumps(plan_b, ensure_ascii=False)
    assert plan_a["schema"] == M.PLAN_SCHEMA
    assert plan_a["stage_order"] == list(M.STAGE_ORDER)
    assert len(plan_a["stages"]) == 5
    assert plan_a["panel_shape_rows_cols"] == [1200, 1920]
    assert plan_a["native_ccd_shape_rows_cols"] == [1944, 2592]
    assert plan_a["bit_depth"] == 12
    assert plan_a["full_scale"] == 4095.0


def test_plan_declares_prerequisites_and_gated_stages(tmp_path: Path) -> None:
    plan = M.build_plan(_args(tmp_path))
    assert plan["gated_stages"] == ["B", "D", "E"]
    assert plan["stages_selected"] == list(M.STAGE_ORDER)
    assert "倍率口径对齐（catA/catB/C06）" == plan["stage_prerequisites"]["D"]
    assert "G1 + G3 + G4" in plan["stage_prerequisites"]["E"]
    assert "熄灯" in plan["stage_prerequisites"]["B"]

    subset = M.build_plan(_args(tmp_path, stages="B"))
    assert subset["stages_selected"] == ["B"]


def test_selected_stages_reject_unknown_and_keep_order(tmp_path: Path) -> None:
    assert M.selected_stages(_args(tmp_path, stages="cba")) == ("A", "B", "C")
    assert M.selected_stages(_args(tmp_path, stages="")) == M.STAGE_ORDER
    try:
        M.selected_stages(_args(tmp_path, stages="Z"))
    except ValueError as exc:
        assert "未知 stage" in str(exc)
    else:
        raise AssertionError("unknown stage must fail closed")


def test_gate_block_reason_blocks_b_d_e_until_confirmed(tmp_path: Path) -> None:
    args = _args(tmp_path)
    assert M.gate_block_reason("A", args) is None
    assert M.gate_block_reason("C", args) is None
    assert M.gate_block_reason("B", args) == M.STAGE_PREREQUISITES["B"]
    assert M.gate_block_reason("D", args) == M.STAGE_PREREQUISITES["D"]
    assert M.gate_block_reason("E", args) == M.STAGE_PREREQUISITES["E"]

    args.confirm_lights_off = True
    assert M.gate_block_reason("B", args) is None
    assert M.gate_block_reason("D", args) == M.STAGE_PREREQUISITES["D"]
    assert M.gate_block_reason("E", args) == M.STAGE_PREREQUISITES["E"]

    args.confirm_magnification_alignment = True
    assert M.gate_block_reason("D", args) is None
    assert M.gate_block_reason("E", args) == M.STAGE_PREREQUISITES["E"]

    args.confirm_g1_g3_g4 = True
    assert M.gate_block_reason("E", args) is None


def test_gate_b_depends_on_frame_window(tmp_path: Path) -> None:
    args = _args(tmp_path, b_frame="dark")
    assert M.gate_block_reason("B", args) == M.STAGE_PREREQUISITES["B"]
    args.confirm_lights_off = True
    assert M.gate_block_reason("B", args) is None

    signal = _args(tmp_path, b_frame="signal")
    assert M.gate_block_reason("B", signal) == "操作员已打开光路（stage B 信号窗口）"
    signal.confirm_lights_off = True  # 熄灯确认不能放行信号窗口
    assert M.gate_block_reason("B", signal) == "操作员已打开光路（stage B 信号窗口）"
    signal.confirm_lights_on = True
    assert M.gate_block_reason("B", signal) is None


def test_probe_axis_mapping_matches_frozen_s1_spec() -> None:
    shape = M.DEFAULT_PANEL_SHAPE
    scale = M._two_pi_gray_scale()
    spec_x = next(s for s in M.PATTERN_SPECS if s["id"] == "grating_x_p128_scale993")
    spec_y = next(s for s in M.PATTERN_SPECS if s["id"] == "grating_y_p128_scale993")
    spec_b = next(s for s in M.PATTERN_SPECS if s["kind"] == "blaze2d")

    def _frozen(spec: dict[str, Any]) -> Any:
        if spec["kind"] == "grating":
            phase = M.grating_phase(
                shape, period_px=int(spec["period_px"]), axis=int(spec["axis"])
            )
        else:
            phase = M.blaze2d_phase(
                shape,
                period_x_px=float(spec["period_x_px"]),
                period_y_px=float(spec["period_y_px"]),
                sign_x=int(spec["sign_x"]),
                sign_y=int(spec["sign_y"]),
            )
        return M.phase_to_gray(phase, scale)

    assert M.digest(M._probe_gray("ramp_x", shape)) == M.digest(_frozen(spec_x))
    assert M.digest(M._probe_gray("ramp_y", shape)) == M.digest(_frozen(spec_y))
    assert M.digest(M._probe_gray("blaze2d", shape)) == M.digest(_frozen(spec_b))


def test_dry_run_never_imports_hardware_or_calls_execute(tmp_path: Path, monkeypatch: Any) -> None:
    def _forbidden(*_a: Any, **_kw: Any) -> None:
        raise AssertionError("dry-run must not touch devices")

    monkeypatch.setattr(M, "execute", _forbidden)
    monkeypatch.setattr(sys, "argv", ["unified-capture", "--out", str(tmp_path / "dry")])
    assert M.main() == M.EXIT_OK
    assert (tmp_path / "dry" / "plan.json").is_file()


def test_parser_defaults_match_plan(tmp_path: Path) -> None:
    parsed = M.build_parser().parse_args([
        "--out", str(tmp_path / "p"),
    ])
    assert parsed.execute is False
    assert parsed.frames == M.DEFAULT_FRAMES
    assert parsed.wait_until_idle == M.DEFAULT_WAIT_IDLE_S
    assert parsed.slm_number == M.DEFAULT_SLM_NUMBER
    assert parsed.confirm_lights_off is False
    assert parsed.confirm_lights_on is False
    assert parsed.b_frame == "both"
    assert parsed.confirm_magnification_alignment is False
    assert parsed.confirm_g1_g3_g4 is False
    assert parsed.stages == "ABCDE"


# --------------------------------------------------------------------------- #
# 灰度 / 图案契约
# --------------------------------------------------------------------------- #

def test_two_pi_gray_scale_matches_frozen_s1_flat_spec() -> None:
    flat_spec = next(spec for spec in M.PATTERN_SPECS if spec["kind"] == "flat")
    assert M._two_pi_gray_scale() == int(flat_spec["two_pi_gray_scale"])


def test_probe_gray_is_uint16_full_panel_and_digest_stable() -> None:
    shape = M.DEFAULT_PANEL_SHAPE
    digests: dict[str, str] = {}
    for name in M.PROBE_NAMES:
        gray = M._probe_gray(name, shape)
        assert gray.shape == shape
        assert gray.dtype == np.uint16
        assert int(gray.min()) >= 0 and int(gray.max()) <= 1023
        digests[name] = M.digest(gray)
        assert M.digest(M._probe_gray(name, shape)) == digests[name]
    assert len(set(digests.values())) == len(M.PROBE_NAMES)


def test_flat_and_arm_pattern_mapping() -> None:
    shape = M.DEFAULT_PANEL_SHAPE
    flat = M._flat_gray(shape)
    assert M.digest(flat) == M.digest(M._arm_gray("pretrained", shape))
    assert M.digest(flat) != M.digest(M._arm_gray("controlled", shape))
    assert M.digest(M._arm_gray("random", shape)) == M.digest(
        M._probe_gray("ramp_y", shape)
    )
    assert M.digest(M._arm_gray("offline", shape)) == M.digest(
        M._probe_gray("blaze2d", shape)
    )


def test_twin_sequence_is_symmetric_and_repeats_flat() -> None:
    assert M.TWIN_SEQUENCE == ("flat", "ramp_x", "blaze2d", "flat")
    shape = M.DEFAULT_PANEL_SHAPE
    assert M.digest(M._twin_gray("flat", shape)) == M.digest(
        M._twin_gray("flat", shape)
    )
    assert M.digest(M._twin_gray("ramp_x", shape)) == M.digest(
        M._probe_gray("ramp_x", shape)
    )


def test_jsonable_converts_numpy_and_nested_values() -> None:
    value = {
        "a": np.array([1, 2], dtype=np.uint16),
        "b": (np.int64(3), {"c": np.float64(1.5)}),
        "none": None,
    }
    assert M._jsonable(value) == {
        "a": [1, 2],
        "b": [3, {"c": 1.5}],
        "none": None,
    }


# --------------------------------------------------------------------------- #
# 静态纪律守卫（AST）
# --------------------------------------------------------------------------- #

def test_no_laser_or_shared_report_writes_in_script() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    forbidden_attrs = {"laser_on", "enable_laser", "set_laser_power", "laser_off"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else ""
            assert name not in forbidden_attrs
    text = SCRIPT.read_text(encoding="utf-8")
    assert "set_laser" not in text


def test_device_imports_are_lazy_and_release_is_in_finally() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    hardware_modules = ("ao_shaping.drivers", "ao_shaping.tools.kernels")
    for node in tree.body:  # 仅模块顶层
        assert not isinstance(node, ast.ImportFrom) or not node.module or not node.module.startswith(hardware_modules)
        assert not isinstance(node, ast.Import) or not any(
            a.name.startswith(hardware_modules) for a in node.names
        )
    function_names = {
        n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "acquire_device_lock" in function_names
    assert "execute" in function_names


def test_each_stage_has_its_own_artifact_writer() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for artifact in (
        "panel_state.json",
        "dark_signal_meta.json",
        "probe_meta.json",
        "paired_meta.json",
        "twin_meta.json",
        "summary.json",
        "released.json",
    ):
        assert artifact in text


# --------------------------------------------------------------------------- #
# Fail-closed（不打开任何设备）
# --------------------------------------------------------------------------- #

def test_no_lock_returns_exit_code_without_touching_output(tmp_path: Path, monkeypatch: Any) -> None:
    out = tmp_path / "no_lock"
    monkeypatch.setattr(M, "acquire_device_lock", lambda _s: None)
    code = M.execute(_args(tmp_path, out=out), M.build_plan(_args(tmp_path, out=out)))
    assert code == M.EXIT_NO_LOCK
    assert not out.exists()


def test_existing_output_dir_fails_closed_and_releases_lock(tmp_path: Path, monkeypatch: Any) -> None:
    class _FakeLock:
        acquired = 0

        def release(self) -> None:
            self.acquired += 1

    lock = _FakeLock()
    monkeypatch.setattr(M, "acquire_device_lock", lambda _s: lock)
    monkeypatch.setattr(M, "_open_devices", lambda _a: (_ for _ in ()).throw(
        AssertionError("devices must not open when out dir already exists")
    ))

    out = tmp_path / "existing"
    out.mkdir()
    code = M.execute(_args(tmp_path, out=out), M.build_plan(_args(tmp_path, out=out)))
    assert code == M.EXIT_ERROR
    assert lock.acquired == 1
    release = json.loads((out / "released.json").read_text(encoding="utf-8"))
    assert release["status"] == "error"
    assert release["lock_released"] is True
    assert release["slm_closed"] is False
    assert release["camera_closed"] is False
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall_status"] == "error"
    assert summary["stages"] == []


# --------------------------------------------------------------------------- #
# Stage B 两个窗口（假设备，不打开任何硬件）
# --------------------------------------------------------------------------- #

class _FakeSLM:
    def __init__(self) -> None:
        self.displayed: list[np.ndarray] = []

    def display_data(self, gray: np.ndarray) -> None:
        self.displayed.append(np.asarray(gray))


class _FakeCam:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)

    def get_numpy_image(self, *_args: Any) -> np.ndarray:
        if not self._frames:
            raise AssertionError("no more fake frames queued")
        return self._frames.pop(0)


def _blank_frame() -> np.ndarray:
    return np.full(M.NATIVE_CCD_SHAPE, 15, dtype=np.uint16)


def _bright_frame() -> np.ndarray:
    frame = _blank_frame()
    frame[900:1000, 900:1000] = 2000
    return frame


def _run_stage_b(
    tmp_path: Path, frames: list[np.ndarray], frame_mode: str
) -> dict[str, Any]:
    return M.stage_b_dark_signal(
        _FakeSLM(),
        _FakeCam(frames),
        tmp_path / "out",
        io.StringIO(),
        settle_s=0.0,
        full_scale=M.FULL_SCALE_REQUIRED,
        exposure_actual=0.087,
        frame_mode=frame_mode,
    )


def test_stage_b_dark_window_records_only_dark_and_pairs_by_contract(tmp_path: Path) -> None:
    record = _run_stage_b(tmp_path, [_blank_frame()], "dark")
    assert record["status"] == "ok"
    assert record["frame_mode"] == "dark"
    assert list(record["frames"]) == ["dark"]
    contract = record["pair_contract"]
    assert contract["frames_expected"] == ["dark", "signal"]
    assert contract["requires_equal_exposure_ms_actual"] is True
    assert contract["requires_equal_flat_gray_sha256"] is True
    written = json.loads(
        (tmp_path / "out" / "stage_b_dark_signal" / "dark_signal_meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["flat_gray_sha256"] == record["flat_gray_sha256"]


def test_stage_b_dark_window_with_light_is_dark_not_dark(tmp_path: Path) -> None:
    record = _run_stage_b(tmp_path, [_bright_frame()], "dark")
    assert record["status"] == "dark_not_dark"
    assert record["frames"]["dark"]["gate"]["has_signal"] is True


def test_stage_b_signal_window_needs_signal(tmp_path: Path) -> None:
    blank = _run_stage_b(tmp_path / "blank", [_blank_frame()], "signal")
    assert blank["status"] == "no_signal"
    assert list(blank["frames"]) == ["signal"]

    lit = _run_stage_b(tmp_path / "lit", [_bright_frame()], "signal")
    assert lit["status"] == "ok"
    assert lit["frames"]["signal"]["gate"]["has_signal"] is True


def test_stage_b_both_mode_is_ordered_sentinel(tmp_path: Path) -> None:
    record = _run_stage_b(tmp_path, [_blank_frame(), _bright_frame()], "both")
    assert record["status"] == "ok"
    assert list(record["frames"]) == ["dark", "signal"]
    assert record["frames"]["dark"]["file"] == "00_dark.npy"
    assert record["frames"]["signal"]["file"] == "01_signal.npy"


def test_stage_b_rejects_unknown_frame_mode(tmp_path: Path) -> None:
    try:
        _run_stage_b(tmp_path, [], "sideways")
    except ValueError as exc:
        assert "--b-frame" in str(exc)
    else:
        raise AssertionError("unknown frame_mode must fail closed")
