# -*- coding: utf-8 -*-
"""``scripts/slm/efficiency_match_capture_20261123.py`` 的离线测试（不碰硬件）。

覆盖冻结采集端的核心保证：
1. 16 帧矩阵: §4 表 ×2 轮遍历的 id 顺序 / 每相位计数 / 轮次行号;
2. blaze 相位: 只沿声明轴变化, 单一连续 2π/period ramp, 哈希稳定;
3. 整形相位: shape 必须等于 Panel_Res, 违规路径硬失败并同时报告两个 shape;
4. 哈希: 数组内容 tobytes 的 SHA256 口径, 同参数稳定、异内容相异;
5. 暗帧: missing 是显式降级 (ok_dark_missing), 绝不伪造;
6. config: roi_frozen / geometry_source 冻结块完整落盘;
7. plan-only: 模块加载与 ``--dry-run-plan`` 不 import 任何设备驱动。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/slm/efficiency_match_capture_20261123.py"
)
ROOT = SCRIPT.parents[2]

#: 冻结的 16 帧采集顺序 (§4 表 ×2 轮, 第二轮 id 加 _r2 后缀)。
EXPECTED_IDS = [
    "flat_pre",
    "blaze_grating",
    "shaping_target",
    "dark",
    "flat_mid",
    "blaze_grating",
    "shaping_target",
    "flat_post",
    "flat_pre_r2",
    "blaze_grating_r2",
    "shaping_target_r2",
    "dark_r2",
    "flat_mid_r2",
    "blaze_grating_r2",
    "shaping_target_r2",
    "flat_post_r2",
]


def _load():
    spec = importlib.util.spec_from_file_location(
        "efficiency_match_capture_20261123_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


capture = _load()


def _ns() -> argparse.Namespace:
    """冻结窗口参数 (与脚本默认值一致) 的 Namespace 替身。"""
    return argparse.Namespace(
        exposure_ms=0.05,
        settle_s=0.8,
        cooldown_s=20.0,
        saturation_fraction=0.9,
        blaze_period_px=128.0,
        blaze_axis=1,
        out=None,
    )


# --------------------------------------------------------------- 16 帧矩阵


def test_matrix_order_ids_counts_and_rounds() -> None:
    """§4 表 ×2 遍历: id 顺序、每相位计数、轮次与行号全部冻结。"""
    frames = capture.build_matrix()
    assert [f["id"] for f in frames] == EXPECTED_IDS
    counts: dict[str, int] = {}
    for frame in frames:
        counts[frame["phase_id"]] = counts.get(frame["phase_id"], 0) + 1
    assert counts == {"flat": 6, "blaze": 4, "shaping": 4, "dark": 2}
    assert [f["round"] for f in frames[:8]] == [1] * 8
    assert [f["round"] for f in frames[8:]] == [2] * 8
    assert [f["row"] for f in frames] == list(range(1, 9)) * 2
    # 每帧的 phase_id 与 §4 表对应 (第二轮同类别同 id)。
    assert [f["phase_id"] for f in frames] == [
        row[1] for row in capture.MATRIX_ROWS * 2
    ]
    assert all(f["purpose"] for f in frames)


def test_matrix_interpretation_is_frozen() -> None:
    """矩阵解释字符串必须显式声明 16 帧 ≠ "每相位 4 帧"。"""
    expected = (
        "§4 table原文 ×2 traversal → 16 frames; differs from "
        "'4 frames per phase': flat 6 / blaze 4 / shaping 4 / dark 2"
    )
    assert capture.MATRIX_INTERPRETATION == expected


# --------------------------------------------------------------- blaze 相位


def test_blaze_phase_varies_only_along_declared_axis() -> None:
    """blaze ramp 沿声明轴变化, 另一轴恒为 0; 周期边界增量为 2π。"""
    rows, cols = 6, 9
    phase_x = capture.blaze_phase((rows, cols), period_px=4.0, axis=1)
    # axis=1 (panel_x): 每行相同, 列方向 ramp。
    assert np.array_equal(phase_x[0, :], phase_x[rows - 1, :])
    assert float(np.ptp(phase_x[:, 0])) == 0.0
    assert abs(float(phase_x[0, 4]) - float(phase_x[0, 0]) - capture.TWO_PI) < 1e-12
    phase_y = capture.blaze_phase((rows, cols), period_px=4.0, axis=0)
    # axis=0 (panel_y): 每列相同, 行方向 ramp。
    assert np.array_equal(phase_y[:, 0], phase_y[:, cols - 1])
    assert float(np.ptp(phase_y[0, :])) == 0.0
    assert abs(float(phase_y[4, 0]) - float(phase_y[0, 0]) - capture.TWO_PI) < 1e-12


def test_blaze_phase_rejects_bad_axis_and_period() -> None:
    """非法 axis / 非正 period 直接拒绝, 不静默降级。"""
    with pytest.raises(ValueError):
        capture.blaze_phase((6, 6), axis=2)
    with pytest.raises(ValueError):
        capture.blaze_phase((6, 6), period_px=0.0)


def test_blaze_gray_offline_is_uint16_scaled() -> None:
    """相位→灰度走 phase_to_slm_grayscale 标度, 输出 uint16 且 <= G_REF。"""
    phase = capture.blaze_phase((6, 12), period_px=4.0, axis=1)
    gray = capture.phase_gray_offline(phase)
    assert gray.dtype == np.uint16
    assert int(gray.max()) <= capture.G_REF


# --------------------------------------------------------------- 哈希稳定性


def test_digest_is_stable_and_content_sensitive() -> None:
    """数组内容 tobytes 的 SHA256: 同参数稳定, 异内容相异。"""
    phase = capture.blaze_phase((8, 16), period_px=8.0, axis=1)
    assert capture.digest(phase) == capture.digest(
        capture.blaze_phase((8, 16), period_px=8.0, axis=1)
    )
    assert capture.digest(phase) != capture.digest(
        capture.blaze_phase((8, 16), period_px=16.0, axis=1)
    )
    flat = capture.flat_phase((8, 16))
    assert capture.digest(flat) == capture.digest(
        np.zeros((8, 16), dtype=np.float64)
    )
    assert capture.digest(flat) != capture.digest(phase)


def test_flat_phase_is_raw_zeros() -> None:
    """flat 相位走 raw 全零路径 (非 create_phase_from_array 包装)。"""
    flat = capture.flat_phase((4, 7))
    assert flat.dtype == np.float64
    assert flat.shape == (4, 7)
    assert float(flat.max()) == 0.0


# --------------------------------------------------------------- 整形相位校验


def test_shaping_phase_shape_validation_failure_path() -> None:
    """shape 不符 / dtype 不符 / 文件缺失都是硬失败, 信息里同时给出两个 shape。"""
    panel = (1200, 1920)
    wrong = np.zeros((1200, 1921), dtype=np.float32)
    with pytest.raises(ValueError) as excinfo:
        capture.validate_shaping_phase(wrong, panel, "test.npy")
    message = str(excinfo.value)
    assert "(1200, 1921)" in message
    assert "(1200, 1920)" in message
    with pytest.raises(ValueError):
        capture.validate_shaping_phase(
            np.zeros((1200, 1920), dtype=np.uint16), panel, "test.npy"
        )
    ok = capture.validate_shaping_phase(
        np.zeros((4, 5), dtype=np.float64), (4, 5), "test.npy"
    )
    assert ok.dtype == np.float32 and ok.shape == (4, 5)


def test_load_shaping_phase_missing_and_mismatched_files(tmp_path: Path) -> None:
    """文件缺失 / shape 与 Panel_Res 不符都硬失败, 绝不静默接受。"""
    with pytest.raises(FileNotFoundError):
        capture.load_shaping_phase(tmp_path / "nope.npy", (4, 5))
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((5, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        capture.load_shaping_phase(bad, (4, 5))
    good = tmp_path / "good.npy"
    np.save(good, np.zeros((4, 5), dtype=np.float32))
    arr = capture.load_shaping_phase(good, (4, 5))
    assert arr.shape == (4, 5) and arr.dtype == np.float32


# --------------------------------------------------------------- 暗帧语义


def test_dark_missing_status_and_summary_semantics() -> None:
    """暗帧缺失是显式降级 (ok_dark_missing), 库/用户提供则是 ok。"""
    assert capture.final_status("missing") == "ok_dark_missing"
    assert capture.final_status("camera_library") == "ok"
    assert capture.final_status("provided") == "ok"

    frames = capture.build_matrix()
    missing = capture.build_summary(
        frame_records=frames,
        saturation_count=0,
        peak_counts={},
        dark_source="missing",
        elapsed_s=1.5,
    )
    assert missing["status"] == "ok_dark_missing"
    assert missing["dark_source"] == "missing"
    assert missing["n_frames"] == 16
    assert missing["matrix_interpretation"] == capture.MATRIX_INTERPRETATION

    ok = capture.build_summary(
        frame_records=frames,
        saturation_count=2,
        peak_counts={"flat_pre": 2941.0},
        dark_source="camera_library",
        elapsed_s=9.5,
    )
    assert ok["status"] == "ok"
    assert ok["saturation_count"] == 2


def test_exposure_from_name_parsing() -> None:
    """文件名里的 ms 曝光数提取: 带 ms 与裸数字都认, 提不出返回 None。"""
    assert capture.exposure_from_name("dark-exp-0.05ms") == pytest.approx(0.05)
    assert capture.exposure_from_name("dark_5ms") == pytest.approx(5.0)
    assert capture.exposure_from_name("dark_5") == pytest.approx(5.0)
    assert capture.exposure_from_name("no_number_here") is None


def test_resolve_dark_from_dir_same_exposure_only(tmp_path: Path) -> None:
    """用户提供的暗帧目录: 只有同曝光 (tol 内) 才采用, 否则 None。"""
    dark_dir = tmp_path / "darks"
    dark_dir.mkdir()
    np.save(dark_dir / "dark-exp-0.05ms.npy", np.zeros((4, 4), dtype=np.float32))
    hit = capture.resolve_dark_from_dir(dark_dir, 0.05)
    assert hit is not None
    assert hit["source"] == "provided"
    assert hit["exposure_ms"] == pytest.approx(0.05)
    assert hit["mean"].shape == (4, 4)
    # 不同曝光严格不套用。
    assert capture.resolve_dark_from_dir(dark_dir, 0.5) is None
    # 目录不存在 → None (走 missing 降级, 不报错)。
    assert capture.resolve_dark_from_dir(tmp_path / "absent", 0.05) is None


def test_load_dark_array_npy_and_npz(tmp_path: Path) -> None:
    """暗帧 .npy 裸数组 / .npz 含 mean 都读; 非二维硬失败。"""
    npy = tmp_path / "dark.npy"
    np.save(npy, np.zeros((4, 6), dtype=np.float32))
    assert capture.load_dark_array(npy).shape == (4, 6)
    npz = tmp_path / "dark.npz"
    np.savez(npz, mean=np.ones((4, 6), dtype=np.float32))
    assert capture.load_dark_array(npz).shape == (4, 6)
    flat = tmp_path / "flat.npy"
    np.save(flat, np.zeros(24, dtype=np.float32))
    with pytest.raises(ValueError):
        capture.load_dark_array(flat)


# --------------------------------------------------------------- config 冻结块


def test_config_roi_frozen_and_geometry_source_complete() -> None:
    """config.json 的 roi_frozen / geometry_source 冻结块完整且逐值相等。"""
    cfg = capture.build_config(
        _ns(),
        panel_shape=(1200, 1920),
        shaping_phase_path=capture.SHAPING_PHASE_PATH,
        shaping_phase_sha=capture.SHAPING_PHASE_SHA256,
        shaping_phase_shape=(1200, 1920),
        device=None,
        dark_source="missing",
        out_dir=Path("out"),
    )
    roi = cfg["roi_frozen"]
    assert set(capture.ROI_KEYS) <= set(roi)
    assert roi["r0_px"] == 48.0
    assert roi["side_px"] == 61.2
    assert roi["separation_seed_px"] == 59.1
    assert roi["separation_search_px"] == 25.0
    assert roi["r1_px"] == 60.0
    assert roi["zero_order_exclusion_px"] == 48.0
    assert roi["saturation_fraction"] == 0.9
    assert roi["blaze_period_panel_px"] == 128.0
    assert roi["blaze_axis"] == "panel_x"
    assert roi["geometry"] == capture.ROI_FROZEN["geometry"]

    geo = cfg["geometry_source"]
    assert set(capture.GEOMETRY_KEYS) <= set(geo)
    assert geo["kx"] == 7551.141568012036
    assert geo["ky"] == 7576.669164768378
    assert geo["k_mean"] == 7563.905366390207
    assert geo["rotation_deg"] == 0.16752399978553
    assert geo["mirror"] is False
    assert geo["center_cycx"] == [1104.8427044240864, 1224.4160271278206]
    assert geo["fwhm0"] == 16.0
    assert geo["measured_utc"] == "20261117_022701"
    assert (
        geo["source"]
        == "data/debug/differentiable_opt/20261117_022658/run_config.json"
    )

    # 冻结窗口其余字段。
    assert cfg["exposure"]["requested_ms"] == 0.05
    assert cfg["exposure"]["all_frames_same_exposure"] is True
    assert cfg["frames_per_phase"] == {"flat": 6, "blaze": 4, "shaping": 4, "dark": 2}
    assert cfg["shaping_phase"]["sha256"] == capture.SHAPING_PHASE_SHA256
    assert cfg["shaping_phase"]["compensation_focal_mm"] is None
    assert "无菲涅尔补偿" in cfg["shaping_phase"]["compensation_note"]
    assert cfg["blaze"]["kind"] == "continuous_ramp_not_binary"
    assert cfg["plan"]["matrix"] == EXPECTED_IDS


def test_build_plan_16_ids_and_interpretation() -> None:
    """plan 骨架: 16 帧 id 顺序冻结, 解释字符串存在, 冻结相位哈希随文件。"""
    plan = capture.build_plan(_ns(), capture.SHAPING_PHASE_PATH)
    assert [f["id"] for f in plan["matrix"]] == EXPECTED_IDS
    assert plan["matrix_interpretation"] == capture.MATRIX_INTERPRETATION
    assert "×2 traversal" in plan["matrix_interpretation"]
    if capture.SHAPING_PHASE_PATH.is_file():
        assert plan["config"]["shaping_phase"]["sha256"] == (
            capture.SHAPING_PHASE_SHA256
        )


def test_write_release_files_double_write_identical(tmp_path: Path) -> None:
    """released.json 与 release.json 双写, 内容逐字节一致。"""
    out = tmp_path / "out"
    out.mkdir()
    release = {
        "slm_flat_gray0": "ok",
        "slm_closed": True,
        "camera_closed": True,
        "lock_released": True,
        "closed_normally": True,
        "acquired_at_utc": "2026-11-23T00:00:00+00:00",
        "error": None,
    }
    paths = capture.write_release_files(out, release)
    assert {p.name for p in paths} == {"released.json", "release.json"}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == capture.RELEASE_SCHEMA
        assert payload["closed_normally"] is True
    assert (out / "released.json").read_bytes() == (out / "release.json").read_bytes()


# --------------------------------------------------------------- plan-only 安全


def test_module_load_does_not_import_device_drivers() -> None:
    """plan-only: 加载脚本模块不得 import 任何 ao_shaping.drivers 驱动。"""
    code = (
        "import importlib.util, sys;"
        f"spec = importlib.util.spec_from_file_location('eff_under_test', {str(SCRIPT)!r});"
        "mod = importlib.util.module_from_spec(spec);"
        "sys.modules[spec.name] = mod;"
        "spec.loader.exec_module(mod);"
        "bad = [k for k in sys.modules if k.startswith('ao_shaping.drivers')];"
        "assert not bad, bad"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr


def test_dry_run_plan_prints_frozen_16_ids() -> None:
    """--dry-run-plan: 不建目录不碰设备, 打印 16 帧 id 冻结顺序。"""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run-plan"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert [f["id"] for f in payload["matrix"]] == EXPECTED_IDS
    assert payload["matrix_interpretation"] == capture.MATRIX_INTERPRETATION
