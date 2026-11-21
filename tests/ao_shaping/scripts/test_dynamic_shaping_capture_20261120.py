"""``scripts/slm/dynamic_shaping_capture_20261120.py`` 的离线测试（不碰硬件）。

覆盖四条核心保证：
1. 端点逐码精确：首帧 = 起点灰度、末帧 = 终点灰度（走最短环绕路径）；
2. 最短环绕：每帧码变化不超过半周期 512，且 0/1023 边界走相邻短路径；
3. 速度解耦：CCD 采样在独立线程按自身节拍运行，不阻塞 SLM 帧派发；
4. plan-only 不 import 任何设备驱动、不创建目录。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/slm/dynamic_shaping_capture_20261120.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "dynamic_shaping_capture_20261120_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


capture = _load()


class FakeClock:
    """确定性的墙钟替身：sleep 直接推进时间，派发瞬时完成。"""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


def _small_panel() -> tuple[int, int]:
    return (8, 6)


# --------------------------------------------------------------- 端点 / 短路径


def test_dispatch_endpoints_are_exact_and_shortest_wrapped() -> None:
    """首帧逐码 = 起点、末帧逐码 = 终点；中途最大码变化 <= 512。"""
    start = np.array(
        [[0, 100, 512, 900, 1, 0, 1023, 300]] * 6, dtype=np.uint16
    )
    target = np.array(
        [[1023, 900, 100, 10, 1022, 512, 0, 301]] * 6, dtype=np.uint16
    )
    plan = capture.build_plan(
        capture.parse_args(["--out", "x", "--duration-s", "1", "--fps", "10"])
    )
    shown: list[np.ndarray] = []

    def display(index: int, gray: np.ndarray) -> dict[str, int]:
        shown.append(gray.copy())
        return {"index": index}

    clock = FakeClock()
    state = capture.SharedState()
    sink = capture.RecordSink(None)
    frames = capture.dispatch_frames(
        plan, start, target, display=display, state=state, sink=sink,
        sleeper=clock.sleep, clock_fn=clock,
    )
    try:
        assert len(frames) == plan.frame_count
        np.testing.assert_array_equal(shown[0], start)
        np.testing.assert_array_equal(shown[-1], target)
        # 0 ↔ 1023 走物理相邻短路径（码差 1），绝不出现 ±1023 的整周跳变。
        assert frames[0]["max_code_change"] <= 512
        assert max(f["max_code_change"] for f in frames) <= 512
        assert all(f["progress_t"] <= 1.0 for f in frames)
        assert state.done and state.index == len(frames) - 1
    finally:
        sink.close()


def test_plan_only_summary_basis_and_frames() -> None:
    """无端点数组时摘要走 full-scale-bound 且帧数正确；给数组时给精确值。"""
    plan = capture.build_plan(
        capture.parse_args(["--out", "x", "--duration-s", "2", "--fps", "10"])
    )
    args = capture.parse_args(["--out", "x"])
    bound = capture.summarize(args, plan, None, None)
    assert bound["frames"] == 21
    assert bound["basis"].startswith("full-scale-bound")

    p, q = _small_panel()
    start = np.zeros((q, p), dtype=np.uint16)
    target = np.full((q, p), 200, dtype=np.uint16)
    exact = capture.summarize(args, plan, start, target)
    assert exact["basis"].startswith("exact")
    assert exact["max_code_change_per_frame"] == pytest.approx(200 / 20)


# --------------------------------------------------------------- 速度解耦


def test_sampler_runs_concurrently_with_dispatch() -> None:
    """CCD 采样线程与派发方并行：grab 阻塞时派发/主线程仍能推进。

    若采样不是独立线程（而是被同步调用），主线程会卡在 get_numpy_image 里，
    永远走不到 ``release.set()`` → ``release.wait`` 超时 → 断言失败。
    """
    p, q = _small_panel()
    release = threading.Event()
    grabbed = threading.Event()

    class BlockingCCD:
        def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
            grabbed.set()
            assert release.wait(timeout=3.0), "CCD grab 阻塞了派发线程（未解耦）"
            return np.full((q, p), 10, dtype=np.uint8)

    state = capture.SharedState()
    sink = capture.RecordSink(None)
    log = capture.DeviceLog(None)
    sampler = capture.CCDSampler(
        BlockingCCD(), state, strategy="every", sink=sink, device_log=log,
        every_n=1, stop=threading.Event(),
    )
    try:
        sampler.start()
        # 模拟派发主线程继续推进帧号（不调用采样器的任何方法、也不 join）。
        for i in range(4):
            state.index = i
            state.t = i / 3.0
        assert grabbed.wait(timeout=2.0), "采样线程从未抓取"
        state.done = True
        release.set()
        sampler.join(timeout=3.0)
        assert not sampler.is_alive()
        measures = [r for r in sink.records if r.get("event") == "measure"]
        assert measures, "没有产生测量记录"
        assert measures[0]["peak"] == 10.0
    finally:
        release.set()
        sink.close()


def test_dispatch_never_calls_phase_pipeline_or_camera() -> None:
    """派发路径只接收最终灰度 + display 回调，不碰相位管线、不抓相机。"""
    p, q = _small_panel()
    start = np.zeros((q, p), dtype=np.uint16)
    target = np.full((q, p), 50, dtype=np.uint16)
    plan = capture.build_plan(
        capture.parse_args(["--out", "x", "--duration-s", "1", "--fps", "5"])
    )
    calls: list[int] = []

    def display(index: int, gray: np.ndarray) -> dict[str, int]:
        calls.append(index)
        return {}

    clock = FakeClock()
    sink = capture.RecordSink(None)
    capture.dispatch_frames(
        plan, start, target, display=display, state=capture.SharedState(), sink=sink,
        sleeper=clock.sleep, clock_fn=clock,
    )
    try:
        assert calls == list(range(len(calls)))
        assert len(calls) == plan.frame_count
    finally:
        sink.close()


# --------------------------------------------------------------- plan-only 无驱动


def test_plan_only_no_driver_import_and_no_dir(tmp_path: Path) -> None:
    """plan-only 在干净解释器里运行：不 import 驱动、不创建输出目录。"""
    out = tmp_path / "must_not_exist"
    script = f"""
import sys, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("cap_uut", {str(SCRIPT)!r})
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
rc = mod.main(["--out", {str(out)!r}, "--duration-s", "1", "--fps", "5"])
assert rc == 0
for name in ("ao_shaping.drivers.slm.santec.driver",
             "ao_shaping.drivers.ccd.daheng.driver"):
    assert name not in sys.modules, name
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    assert "PLAN ONLY" in proc.stdout
    assert not out.exists()
