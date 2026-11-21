"""``scripts/slm/target_phase_matrix_20261120.py`` 的离线测试（不碰硬件）。

覆盖五条核心保证：

1. **离线**：加载脚本不得 import 任何设备驱动（``ao_shaping.drivers`` 一个都不许进
   ``sys.modules``），也不打开 SLM / CCD / 激光；
2. **尺寸语义**：冻结矩阵的 ``size_px`` 对每种形状的含义（边长 / 外接圆直径 /
   长边 / 外径）与 ``patterns.create_target_shape`` 的参数一一对应，光栅化面积
   能和解析面积对账；
3. **代表性**：冻结模型的 ``grid_size=128`` / ``fft_size=512`` 具体覆盖数字
   （FFT 窗 ±253.35 px、居中上限边长 490.7 px、分辨元 3.96 px），200 px 可表示、
   500 px 不可表示，而 GS 网格（±1900 px）连 1500 px 都放得下；
4. **manifest 契约**：``sha256{target,mask,phase}`` 与文件内容一致、顶层记录
   checkpoint/脚本 SHA256 与 ``git_commit``、``carrier="none"``、model 相位标
   ``model_phase_diagnostic_only``；重复运行幂等（全 ``skipped``）；
5. **不伪造**：不可表示时 ``status="skipped_unrepresentable"`` 且不写相位文件。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/slm/target_phase_matrix_20261120.py"
CHECKPOINT = ROOT / "data/debug/slm_transfer/p3_fit_20261120/frozen_parameters.json"


def _load():
    """按文件路径加载脚本模块（``scripts/`` 不是可导入的包）。"""
    spec = importlib.util.spec_from_file_location(
        "target_phase_matrix_20261120_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


matrix = _load()


@pytest.fixture(scope="module")
def params():
    """P3 裸 checkpoint（双资格门 fail，只作诊断）。"""
    if not CHECKPOINT.is_file():
        pytest.skip(f"冻结 checkpoint 缺失: {CHECKPOINT}")
    return matrix.load_frozen_parameters(CHECKPOINT)


def _cli(out: Path, *extra: str) -> int:
    """以固定 CLI 参数跑一次 ``main``（checkpoint 用绝对路径，不依赖 cwd）。"""
    return matrix.main(
        ["--out", str(out), "--checkpoint", str(CHECKPOINT), *extra]
    )


def _read_manifest(out: Path) -> dict:
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------- 离线保证
def test_loading_script_imports_no_hardware_driver() -> None:
    """脚本加载后 ``ao_shaping.drivers`` 一个子模块都不在 ``sys.modules`` 里。"""
    code = "\n".join(
        [
            "import importlib.util, sys",
            f"spec = importlib.util.spec_from_file_location('m', r'{SCRIPT}')",
            "module = importlib.util.module_from_spec(spec)",
            "sys.modules['m'] = module",
            "spec.loader.exec_module(module)",
            "bad = sorted(k for k in sys.modules if k.startswith('ao_shaping.drivers'))",
            "print('IMPORTED_DRIVERS=' + repr(bad))",
            "raise SystemExit(1 if bad else 0)",
        ]
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert done.returncode == 0, done.stdout + done.stderr


# --------------------------------------------------------------------- 尺寸语义
def test_size_semantics_are_the_frozen_ones() -> None:
    """冻结矩阵的尺寸语义：边长 / 外接圆直径 / 长边=size / 外径=size。"""
    assert matrix.SHAPES == (
        "square",
        "circle",
        "triangle",
        "star",
        "rectangle",
        "annular",
    )
    assert matrix.SIZES_PX == (200, 500, 1000, 1500)
    assert matrix.shape_kwargs("square", 500) == {"side": 500.0}

    rect = matrix.shape_kwargs("rectangle", 500)
    assert rect == {"side": 250.0, "aspect_ratio": 2.0}

    ring = matrix.shape_kwargs("annular", 500)
    assert ring["outer_radius"] == 250.0
    assert ring["inner_radius"] == pytest.approx(150.0)

    circle = matrix.shape_kwargs("circle", 500)
    assert circle["radius_ratio"] == pytest.approx(500.0 / matrix.CCD_H)
    # create_target_shape 的半径 = radius_ratio * min(h, w) / 2 ⇒ 直径 = size
    assert circle["radius_ratio"] * matrix.CCD_H / 2.0 == pytest.approx(250.0)


def test_rasterised_targets_match_analytic_area() -> None:
    """光栅化在 CCD 原生网格上，面积与解析值一致（<1%），矩形长边 = size。"""
    centre = (matrix.CCD_W / 2.0, matrix.CCD_H / 2.0)
    target, mask = matrix.build_target("circle", 200, centre)
    assert target.shape == (matrix.CCD_H, matrix.CCD_W)
    assert target.dtype == np.float32
    assert set(np.unique(target)) == {0.0, 1.0}
    assert int(mask.sum()) == pytest.approx(matrix.analytic_area("circle", 200), rel=0.01)
    rows, cols = np.nonzero(mask)
    assert int(cols.max() - cols.min() + 1) == pytest.approx(200, abs=3)
    assert int(rows.max() - rows.min() + 1) == pytest.approx(200, abs=3)

    _rect_target, rect_mask = matrix.build_target("rectangle", 200, centre)
    rrows, rcols = np.nonzero(rect_mask)
    assert int(rcols.max() - rcols.min() + 1) == pytest.approx(200, abs=3)
    assert int(rrows.max() - rrows.min() + 1) == pytest.approx(100, abs=3)


# --------------------------------------------------------------------- 代表性数字
def test_model_grid_numbers_and_representability(params) -> None:
    """冻结模型的覆盖半径 / 上限边长 / 分辨元必须是具体数字，且逐尺寸判定正确。"""
    report = matrix.model_grid_report(params)
    assert report["grid_size"] == 128
    assert report["fft_size"] == 512
    assert report["zero_pad_factor"] == 4
    assert report["control_cell_panel_px"] == pytest.approx(15.0)
    assert report["window_half_span_px"] == pytest.approx(253.35, abs=0.05)
    assert report["usable_half_span_px"] == pytest.approx(245.35, abs=0.05)
    assert report["max_target_side_px"] == pytest.approx(490.70, abs=0.05)
    assert report["resolution_px_per_dof"] == pytest.approx(3.9586, abs=0.001)

    centre = matrix.resolve_centre(params)
    ok = matrix.representability("square", 200, None, params, centre, method="model")
    assert ok["representable"] is True
    assert ok["window_headroom_px"] == pytest.approx(145.35, abs=0.05)

    bad = matrix.representability("square", 500, None, params, centre, method="model")
    assert bad["representable"] is False
    assert bad["window_ok"] is False
    assert "490.7" in bad["reason"]

    # GS 网格覆盖整块面板：1500 px 目标的半跨 750 px 仍在 ±1900 px 内。
    gs = matrix.representability("square", 1500, None, params, centre, method="gs")
    assert gs["representable"] is True
    assert gs["usable_half_span_px"] == pytest.approx(1892.14, abs=0.05)
    assert gs["elements_across_long"] == pytest.approx(1500 / 6.3338, rel=0.01)


# --------------------------------------------------------------------- dry-run
def test_dry_run_lists_48_entries_and_writes_nothing(tmp_path, capsys) -> None:
    """冻结矩阵 48 条：24 条 GS 全生成；model 仅 200 px 的 6 条生成。"""
    out = tmp_path / "dry"
    assert matrix.main(["--out", str(out), "--checkpoint", str(CHECKPOINT), "--dry-run"]) == 0
    assert not out.exists()
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["entries"]) == 48
    gs_rows = [row for row in plan["entries"] if row["method"] == "gs"]
    model_rows = [row for row in plan["entries"] if row["method"] == "model"]
    assert len(gs_rows) == 24 and len(model_rows) == 24
    assert all(row["action"] == "generate" for row in gs_rows)
    skipped = {row["size_px"] for row in model_rows if row["action"] != "generate"}
    assert skipped == {500, 1000, 1500}
    assert plan["gs_grid"] == [600, 960]


# --------------------------------------------------------------------- manifest
def test_run_writes_manifest_sha256_and_is_idempotent(tmp_path) -> None:
    """实跑 circle@200 的 GS + model：manifest 契约成立，且第二次全部 skipped。"""
    pytest.importorskip("torch")
    out = tmp_path / "out"
    argv = [
        "--shapes", "circle",
        "--sizes", "200",
        "--methods", "both",
        "--gs-iterations", "3",
        "--model-iterations", "2",
        "--device", "cpu",
    ]
    assert _cli(out, *argv) == 0
    payload = _read_manifest(out)
    assert payload["carrier"] == "none"
    assert payload["tilt_applied"] is False
    assert payload["checkpoint_sha256"] == hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    assert payload["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert payload["git_commit"] is None or len(payload["git_commit"]) == 40
    assert payload["gs_iterations"] == 3
    assert payload["model_iterations"] == 2
    assert payload["seed"] == 0
    assert payload["zero_order"]["core_radius_px"] > 0
    assert payload["summary"]["total"] == 2

    entries = payload["entries"]
    assert {entry["method"] for entry in entries} == {"gs", "model"}
    for entry in entries:
        assert entry["status"] in {"written", "skipped", "overwritten"}
        assert entry["representable"] is True
        assert entry["size_semantics"]
        assert entry["created_utc"].endswith("Z")
        assert set(entry["sha256"]) == {"target", "mask", "phase"}
        for key in ("target", "mask", "phase"):
            digest = hashlib.sha256((out / entry[f"{key}_path"]).read_bytes()).hexdigest()
            assert digest == entry["sha256"][key]

    model_entry = next(entry for entry in entries if entry["method"] == "model")
    assert model_entry["model_phase_diagnostic_only"] is True
    assert model_entry["acceptance"] == "model_only"
    assert model_entry["grid"] == [128, 128]
    assert model_entry["diagnostics"]["model_target_pixels_in_window"] > 0
    assert model_entry["diagnostics"]["phase_wrapped"] is False

    gs_entry = next(entry for entry in entries if entry["method"] == "gs")
    assert gs_entry["model_phase_diagnostic_only"] is False
    assert gs_entry["acceptance"] == "simulation_only"
    assert gs_entry["grid"] == [1200, 1920]
    assert gs_entry["diagnostics"]["gs_iterations"] == 3
    assert gs_entry["diagnostics"]["phase_wrapped"] is True

    phase_path = out / gs_entry["phase_path"]
    mtime_before = phase_path.stat().st_mtime_ns
    assert _cli(out, *argv) == 0
    again = _read_manifest(out)
    assert again["summary"]["skipped"] == 2
    assert again["summary"]["written"] == 0
    assert again["summary"]["overwritten"] == 0
    assert again["summary"]["failed"] == 0
    assert phase_path.stat().st_mtime_ns == mtime_before
    assert {entry["sha256"]["phase"] for entry in again["entries"]} == {
        entry["sha256"]["phase"] for entry in entries
    }


def test_unrepresentable_model_phase_is_not_faked(tmp_path) -> None:
    """500 px 在冻结模型网格外：记 skipped_unrepresentable，不写相位。"""
    out = tmp_path / "out"
    assert _cli(
        out, "--shapes", "square", "--sizes", "500", "--methods", "model"
    ) == 0
    payload = _read_manifest(out)
    entry = payload["entries"][0]
    assert entry["status"] == "skipped_unrepresentable"
    assert entry["representable"] is False
    assert entry["model_phase_diagnostic_only"] is True
    assert entry["sha256"]["phase"] is None
    assert entry["phase_path"] is None
    assert "490.7" in entry["notes"]
    assert payload["summary"]["skipped_unrepresentable"] == 1
    assert payload["summary"]["representable_false"] == ["square@500"]
    phases = out / "phases"
    assert not phases.exists() or not list(phases.glob("*.npy"))
