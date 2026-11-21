"""多端点批次计划 / 原始帧堆叠的离线测试（不碰硬件）。

覆盖：
1. --plan 校验：重复/非法 label、越界灰度、缺失文件 → plan-only 阶段 SystemExit；
2. 多 transition plan-only：打印 N 条摘要、不建目录、不 import 驱动；
3. stack 写入：假相机/假 SLM 跑 2 个 transition，检查容量、元数据、run_summary；
4. 串行语义：create_camera / SLM open 各一次，采样线程按 transition 重建；
5. 失败路径：第 2 个 transition 抛错 + abort → 第 3 个不执行、设备仍释放。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/slm/dynamic_shaping_capture_20261120.py"
)


def _load(name: str = "dynamic_shaping_capture_plan_under_test"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


capture = _load()
PANEL = (8, 6)  # W, H


def _make_plan(
    tmp_path: Path,
    *,
    n: int = 2,
    gray_out_of_range: bool = False,
    duplicate_label: bool = False,
    bad_label: bool = False,
    missing_file: bool = False,
) -> tuple[Path, dict[str, Any]]:
    p, q = PANEL
    rng = np.random.default_rng(1234)
    phase = rng.normal(size=(q, p))
    phase_path = tmp_path / "phase.npy"
    np.save(phase_path, phase)
    gray = np.zeros((q, p), dtype=np.uint16)
    if gray_out_of_range:
        gray[0, 0] = 2000
    gray_path = tmp_path / "gray.npy"
    np.save(gray_path, gray)
    mask = np.zeros((q, p), dtype=bool)
    mask[2:4, 2:4] = True
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)

    transitions = []
    for i in range(n):
        label = "dup" if duplicate_label else (f"bad label {i}" if bad_label else f"tr_{i}")
        target_path = str(phase_path) if not missing_file or i == 0 else "does_not_exist.npy"
        transitions.append({
            "label": label,
            "group": {"shape": "star", "size_px": 200 + i, "method": "gs"},
            "start": {"kind": "flat", "gray": 0},
            "target": {"kind": "phase", "path": target_path},
            "target_mask": str(mask_path),
        })
    payload = {
        "schema": "speed_decouple_plan/1",
        "panel": [p, q],
        "duration_s": 0.5,
        "fps": 10,
        "easing": "smooth",
        "measure": {"strategy": "every", "every_n_frames": 1},
        "exposure_ms": 0.087,
        "bit_depth": 12,
        "transitions": transitions,
    }
    if gray_out_of_range:
        payload["transitions"][0]["target"] = {"kind": "gray", "path": str(gray_path)}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


# ----------------------------------------------------------------------- 校验


@pytest.mark.parametrize(
    "kwarg", ["duplicate_label", "bad_label", "gray_out_of_range", "missing_file"]
)
def test_plan_validation_errors_at_plan_only(tmp_path: Path, kwarg: str) -> None:
    path, _ = _make_plan(tmp_path, n=3, **{kwarg: True})
    args = capture.parse_args([
        "--out", str(tmp_path / "out"), "--plan", str(path),
    ])
    # load_plan 在 plan-only 阶段（execute=False）前被调用 → SystemExit。
    with pytest.raises(SystemExit):
        capture.main([
            "--out", str(tmp_path / "out"), "--plan", str(path),
        ])


def test_plan_only_prints_all_transitions_no_dir_no_driver(tmp_path: Path, capsys) -> None:
    path, _ = _make_plan(tmp_path, n=3)
    out = tmp_path / "out"
    assert capture.main([
        "--out", str(out), "--plan", str(path),
    ]) == 0
    text = capsys.readouterr().out
    for label in ("tr_0", "tr_1", "tr_2"):
        assert label in text
    assert "transitions=3" in text
    assert "total_frames=" in text
    assert not out.exists()

    # 干净解释器里 sys.modules 不得出现 ao_shaping.drivers。
    script = (
        "import importlib.util, sys, json; "
        f"spec = importlib.util.spec_from_file_location('x', r'{SCRIPT}'); "
        "m = importlib.util.module_from_spec(spec); sys.modules['x'] = m; "
        "spec.loader.exec_module(m); "
        "assert not any(k.startswith('ao_shaping.drivers') for k in sys.modules); "
        f"assert m.main(['--out', {str(out)!r}, '--plan', {str(path)!r}]) == 0; "
        "assert not any(k.startswith('ao_shaping.drivers') for k in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        cwd=str(tmp_path), check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not out.exists()


# ----------------------------------------------------------------------- 硬件假件


class FakeCamera:
    camera_shape = (PANEL[1], PANEL[0])

    def __init__(self, shape: tuple[int, int] | None = None) -> None:
        self.shape = shape or self.camera_shape
        self.opened = False
        self.closed = False
        self.n = 0

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        self.n += 1
        time.sleep(0.002)
        return np.full(self.shape, 100 + self.n, dtype=np.uint16)

    def effective_full_scale(self, img: np.ndarray) -> float:
        return 4095.0


class FakeSLM:
    def __init__(self, *args, **kwargs) -> None:
        self.Panel_Res = PANEL  # (W, H)；脚本会倒序成 (H, W) 面板
        self.opened = False
        self.closed = False
        self.calls = 0

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def create_phase_from_array(self, phase: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.asarray(
            (np.asarray(phase) * 10) % 1023, dtype=np.uint16
        )


class FakeRotator:
    def __init__(self, slm: FakeSLM) -> None:
        self.slm = slm

    def display_data(self, slm: Any, gray: np.ndarray, wait_time_s: float = 0) -> int:
        return 1


def _install_fakes(monkeypatch, cam: FakeCamera, slm: FakeSLM) -> dict[str, int]:
    calls = {"create_camera": 0}

    def create_camera(camera, cam_id, **kwargs):
        calls["create_camera"] += 1
        return cam

    def get_camera_exposure_ms(cam):
        return 0.087

    monkeypatch.setitem(sys.modules, "ao_shaping", importlib.import_module("ao_shaping"))
    import types

    ccd_common = types.ModuleType("ao_shaping.drivers.ccd.common")
    ccd_common.create_camera = create_camera
    ccd_common.get_camera_exposure_ms = get_camera_exposure_ms
    slm_mod = types.ModuleType("ao_shaping.drivers.slm.santec")
    slm_mod.Santec = lambda *a, **k: slm
    slm_mod.SlotRotator = FakeRotator
    logger_mod = types.ModuleType("loguru")
    logger_mod.logger = type("L", (), {"warning": staticmethod(lambda *a, **k: None)})()
    monkeypatch.setitem(sys.modules, "ao_shaping.drivers.ccd.common", ccd_common)
    monkeypatch.setitem(sys.modules, "ao_shaping.drivers.slm.santec", slm_mod)
    monkeypatch.setitem(sys.modules, "loguru", logger_mod)
    # _wait_and_lock 必须被禁掉（否则 import device_mutex）。
    monkeypatch.setattr(
        capture, "_wait_and_lock",
        lambda args: (None, {"status": "test-stub"}, lambda: None),
    )
    return calls


def _run_batch(
    tmp_path: Path, monkeypatch, *, path: Path, on_error: str = "abort"
) -> tuple[int, FakeCamera, FakeSLM, dict[str, int]]:
    cam, slm = FakeCamera(), FakeSLM()
    calls = _install_fakes(monkeypatch, cam, slm)
    out = tmp_path / "out"
    code = capture.main([
        "--out", str(out), "--plan", str(path), "--execute",
        "--raw-frames", "stack", "--on-error", on_error,
        "--inter-transition-settle-s", "0",
        "--wait-until-idle", "0",
    ])
    return code, cam, slm, calls


# ----------------------------------------------------------------------- 批量执行


def test_batch_stack_and_serial_semantics(tmp_path: Path, monkeypatch) -> None:
    path, _ = _make_plan(tmp_path, n=2)
    code, cam, slm, calls = _run_batch(tmp_path, monkeypatch, path=path)
    assert code == 0
    # 串行语义：只开一次设备。
    assert calls["create_camera"] == 1
    assert cam.opened and cam.closed
    assert slm.opened and slm.closed
    out = tmp_path / "out"
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 2
    assert all(r["status"] == "ok" for r in summary)
    assert all(r["frames_dispatched"] == 6 for r in summary)  # 0.5s * 10fps + 1

    for label in ("tr_0", "tr_1"):
        tr_dir = out / label
        frames = np.load(tr_dir / "frames.npy", mmap_mode="r")
        assert frames.dtype == np.uint16
        assert frames.shape == (7, 6, 8)  # ceil(0.5*10)+2
        meta = json.loads((tr_dir / "frames_meta.json").read_text(encoding="utf-8"))
        assert meta["capacity"] == 7
        assert meta["n_measured"] >= 1
        assert meta["n_measured"] <= meta["capacity"]
        assert meta["shape"] == [6, 8]
        assert meta["dtype"] == "uint16"
        assert meta["bit_depth"] == 12
        assert meta["exposure_ms"] == 0.087
        assert meta["raw_dtype_max"] == 65535
        assert meta["effective_full_scale"] == 4095.0  # issue #246
        assert meta["full_scale_source"] == "camera.effective_full_scale"
        assert "achieved_measure_fps" in meta
        assert meta["requested_measure_every"] == 1
        assert meta["slm_frames_dispatched"] == 6
        assert "measurements_per_slm_frame" in meta
        for i, rec in enumerate(meta["records"]):
            assert rec["i"] == i
            assert rec["slm_frame_index"] >= 0
            assert rec["t_rel_s"] >= 0
            assert rec["monotonic_ns"] > 0
            assert rec["peak"] > 0
            "saturated_pixels" in rec
            "uniformity" in rec
        trans = json.loads((tr_dir / "transition.json").read_text(encoding="utf-8"))
        assert trans["label"] == label
        assert "sha256" in trans["target"]
        assert len(trans["target"]["sha256"]) == 64
        assert "sha256" in trans["target_mask"]

    # frames.jsonl / device_log / run_plan 存在。
    assert (out / "frames.jsonl").exists()
    dev_log = json.loads((out / "device_log.json").read_text(encoding="utf-8"))
    assert any(e["action"] == "close" for e in dev_log)
    assert any(e["action"] == "transition_begin" for e in dev_log)
    run_plan = json.loads((out / "run_plan.json").read_text(encoding="utf-8"))
    assert run_plan["schema"] == "speed_decouple_plan/1"
    assert all(
        "path" in t["target"] and "sha256" in t["target"]
        for t in run_plan["transitions"]
    )

    # 采样线程按 transition 重建：slm_frame_index 从 0 重新计数（每条 <=5）。
    for label in ("tr_0", "tr_1"):
        meta = json.loads(
            (out / label / "frames_meta.json").read_text(encoding="utf-8")
        )
        assert all(rec["slm_frame_index"] <= 5 for rec in meta["records"])


def test_stack_raw_frames_keep_camera_native_full_frame(
    tmp_path: Path, monkeypatch
) -> None:
    """回归（实验失败）：raw 帧按 CCD 原生全幅落盘，绝不按 SLM 面板尺寸预分配。

    实验 CCD 返回 1944x2592 全幅（冻结计划禁止物理开窗），面板却是 (1200,1920)。
    旧实现用面板尺寸预分配 memmap，``on_frame`` 首帧即抛 shape 不一致；异常发生在
    daemon 采样线程里被吞掉，主流程照常结束但 **零帧落盘**（所有 transition 目录失效）。
    这里用与面板不同的相机幅面跑真实 execute 路径，必须全幅写入且无异常。
    """
    ccd_shape = (1944, 2592)
    assert ccd_shape != (PANEL[1], PANEL[0])  # 与面板 (H,W) 明确不同

    path, payload = _make_plan(tmp_path, n=1)
    # target_mask 是面板（显示）空间的量，load_plan 要求它与 panel 同形；本用例只关心
    # raw 帧幅面，故不传 mask（真实失败的那次采集同样没有 mask）。
    payload["transitions"][0]["target_mask"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    cam, slm = FakeCamera(shape=ccd_shape), FakeSLM()
    _install_fakes(monkeypatch, cam, slm)
    out = tmp_path / "out"
    code = capture.main([
        "--out", str(out), "--plan", str(path), "--execute",
        "--raw-frames", "stack", "--on-error", "abort",
        "--inter-transition-settle-s", "0",
        "--wait-until-idle", "0",
    ])
    assert code == 0

    tr_dir = out / "tr_0"
    meta = json.loads((tr_dir / "frames_meta.json").read_text(encoding="utf-8"))
    assert meta["n_measured"] >= 1  # 旧实现此处为 0（异常被采样线程吞掉）
    assert meta["n_measured"] <= meta["capacity"]
    assert meta["shape"] == list(ccd_shape)
    assert len(meta["records"]) == meta["n_measured"]

    frames = np.load(tr_dir / "frames.npy", mmap_mode="r")
    assert frames.dtype == np.uint16
    # 首帧定型原生幅面：capacity 与 CCD 全幅，不裁剪、不降采样。
    assert frames.shape == (meta["capacity"], *ccd_shape)
    for i in range(meta["n_measured"]):
        assert frames[i].shape == ccd_shape
        assert int(frames[i].max()) > 0


def test_failure_abort_releases_devices(tmp_path: Path, monkeypatch) -> None:
    path, _ = _make_plan(tmp_path, n=3)
    cam, slm = FakeCamera(), FakeSLM()
    calls = _install_fakes(monkeypatch, cam, slm)
    out = tmp_path / "out"
    # 第 2 个 transition 的 display 抛错更可靠：按 display 调用计数。
    display_count = {"n": 0}

    class FailingRotator(FakeRotator):
        def display_data(self, slm, gray, wait_time_s=0):
            display_count["n"] += 1
            if 6 < display_count["n"] <= 12:
                raise RuntimeError("simulated SLM fault")
            return 1

    slm_mod = sys.modules["ao_shaping.drivers.slm.santec"]
    slm_mod.SlotRotator = FailingRotator
    code = capture.main([
        "--out", str(out), "--plan", str(path), "--execute",
        "--raw-frames", "stack", "--on-error", "abort",
        "--inter-transition-settle-s", "0", "--wait-until-idle", "0",
    ])
    assert code == 1
    assert cam.closed and slm.closed  # 设备仍释放
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 3
    assert summary[0]["status"] == "ok"
    assert summary[1]["status"] == "failed"
    assert "simulated SLM fault" in summary[1]["error"]
    assert summary[2]["status"] == "skipped"  # 第 3 个未执行
    assert not (out / "tr_2").exists()
    assert calls["create_camera"] == 1
