"""``scripts/experiments/_window.py`` 的锁归属两模式: 纯离线验证。

背景 (2026-11-19 ``issue238_binary_phase_20261119/run3_period_sweep``): 窗口脚本
``run_issue238_period_sweep_20261119.py`` 自己持有 bench ``DeviceLock``, 又把子段
argv 固定成 ``--wait-until-idle 0``。探针脚本**总是** ``acquire``, 0 只是超时上限,
于是 ``period32`` 被父进程的锁挡在门外 (``exit_code=2``, 见 ``segments/period32.log``)。

本文件用**假 DeviceLock + 假子段** (不碰设备、不跑真探针) 钉住修复契约:

1. 默认 ``lock_mode="window"``: 父窗口持锁到结束, 子段内再 acquire 必然失败 (互锁证据),
   transfer 窗口的 argv/模式不变;
2. ``lock_mode="segment"``: 窗口**不** acquire, 每段脚本自取自放 —— 不会父子互锁;
3. 任一段非零退出 → 立即停止, 后面的段不跑;
4. 段间冷却按 ``--cooldown-s`` 逐段调用 (段数 - 1 次);
5. 每段结束锁状态 + 最终锁证据落盘; 段后锁仍被别人占着时**只记录并停止**, 窗口
   ``release`` 次数为 0 (绝不释放别人的锁);
6. dry-run 不建目录、不 import 设备模块, 并把真实 ``--wait-until-idle`` 转发给子段。
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import subprocess
import sys
import types
from pathlib import Path



ROOT = Path(__file__).resolve().parents[3]


WINDOW_SCRIPT = ROOT / "scripts" / "experiments" / "_window.py"
PERIOD_SWEEP_SCRIPT = (
    ROOT / "scripts" / "experiments" / "run_issue238_period_sweep_20261119.py"
)
TRANSFER_WINDOW_SCRIPT = (
    ROOT / "scripts" / "experiments" / "run_transfer_window_20261119.py"
)


def _load_window_module():
    """按路径加载 ``_window`` (``scripts/experiments`` 不是包)。"""
    spec = importlib.util.spec_from_file_location(
        "experiment_window_under_test", WINDOW_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


window = _load_window_module()


# ---------------------------------------------------------------------------
# 假 DeviceLock / 假 device_mutex (进程内, 不碰文件、不碰驱动)
# ---------------------------------------------------------------------------
class FakeHolder:
    """``DeviceLock.holder()`` 的返回对象 (只要有这几个属性)。"""

    def __init__(self, pid: int = 4321, cmdline: str = "fake-segment") -> None:
        self.pid = pid
        self.ppid = 0
        self.cmdline = cmdline
        self.acquired_at = "2026-11-19T00:00:00+00:00"


class FakeLock:
    """进程内假锁: 记录 acquire/release 次数, 从不开锁文件。"""

    def __init__(self) -> None:
        self.holder_obj: FakeHolder | None = None
        self.acquire_timeouts: list[float] = []
        self.release_calls = 0
        self.fail_acquire = False

    def acquire(self, timeout: float = 0.0) -> bool:
        self.acquire_timeouts.append(timeout)
        if self.fail_acquire or self.holder_obj is not None:
            return False
        self.holder_obj = FakeHolder()
        return True

    def holder(self) -> FakeHolder | None:
        return self.holder_obj

    def release(self) -> None:
        if self.holder_obj is None:
            raise RuntimeError("release() on a lock nobody holds")
        self.release_calls += 1
        self.holder_obj = None

    def is_locked(self) -> bool:
        return self.holder_obj is not None


def _install_fake_mutex(monkeypatch, lock: FakeLock) -> types.ModuleType:
    """把假锁塞进 ``sys.modules``; monkeypatch 负责还原。"""

    class _Verdict:
        ok = True
        status = "idle"
        reason = ""
        reason_text = ""

    fake = types.ModuleType("ao_shaping.tools.kernels.device_mutex")
    fake.DeviceLock = type(  # type: ignore[attr-defined]
        "DeviceLock", (), {"default": staticmethod(lambda label="bench": lock)}
    )
    fake.wait_until_idle = lambda seconds, **kwargs: _Verdict()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    return fake


# ---------------------------------------------------------------------------
# 假子段 (monkeypatch ``run_segment``): 记录调用顺序与段内看到的锁状态
# ---------------------------------------------------------------------------
class FakeRunner:
    def __init__(self, lock: FakeLock, *, fail_on: str | None = None,
                 hold_lock_after: str | None = None) -> None:
        self.lock = lock
        self.fail_on = fail_on
        self.hold_lock_after = hold_lock_after
        self.calls: list[str] = []
        self.lock_held_on_entry: list[bool] = []
        self.child_acquire_ok: list[bool] = []

    def __call__(self, seg, *, window, timeout_s):
        self.calls.append(seg.name)
        held = self.lock.is_locked()
        self.lock_held_on_entry.append(held)
        # 子脚本会自己 acquire; 父窗口持锁时这里必然 False (就是 run3 的死法)。
        child_acquired = self.lock.acquire(timeout=0.0)
        self.child_acquire_ok.append(child_acquired)
        if child_acquired and seg.name != self.hold_lock_after:
            self.lock.release()  # 只放自己拿到的锁; 父窗口的锁不碰
        status = "failed" if seg.name == self.fail_on else "ok"
        return {
            "name": seg.name,
            "purpose": seg.purpose,
            "argv": list(seg.argv),
            "out": str(seg.out),
            "log": str(window / "segments" / f"{seg.name}.log"),
            "started_at": "2026-11-19T00:00:00+00:00",
            "finished_at": "2026-11-19T00:00:01+00:00",
            "timeout_s": timeout_s,
            "timed_out": False,
            "exit_code": 0 if status == "ok" else 2,
            "duration_s": 1.0,
            "out_exists": False,
            "out_files": [],
            "status": status,
        }


def _segments(tmp_path: Path, count: int = 2):
    return [
        window.Segment(
            name=f"seg{i}",
            argv=(sys.executable, "-c", "pass"),
            out=tmp_path / f"out{i}",
            purpose=f"fake segment {i}",
        )
        for i in range(count)
    ]


def _run(tmp_path: Path, segments, monkeypatch, runner, **kwargs):
    monkeypatch.setattr(window, "run_segment", runner)
    params = dict(
        window=tmp_path / "window",
        cooldown_s=0.0,
        wait_s=30.0,
        timeout_s=10.0,
        slm_number=1,
        cam_id=0,
        execute=True,
    )
    params.update(kwargs)
    return window.run_window(segments, **params)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. 默认模式 = window, transfer 窗口行为不变
# ---------------------------------------------------------------------------
def test_default_lock_mode_is_window():
    signature = inspect.signature(window.run_window)
    assert signature.parameters["lock_mode"].default == "window"
    assert window.LOCK_MODES == ("window", "segment")


def test_window_mode_parent_holds_lock_so_child_acquire_fails(tmp_path, monkeypatch):
    lock = FakeLock()
    _install_fake_mutex(monkeypatch, lock)
    runner = FakeRunner(lock)

    rc = _run(tmp_path, _segments(tmp_path, 2), monkeypatch, runner,
              lock_mode="window")

    assert rc == 0
    # 父窗口恰好 acquire 一次, 子段看到锁已被父进程持有 ⇒ 子段 acquire 全部失败。
    assert len(lock.acquire_timeouts) == 3  # 1 次父窗口 + 2 次假子段尝试
    assert runner.lock_held_on_entry == [True, True]
    assert runner.child_acquire_ok == [False, False]
    assert lock.release_calls == 1  # 只有父窗口自己的 release
    payload = _read_json(tmp_path / "window" / "window.json")
    assert payload["lock_mode"] == "window"
    assert payload["released"]["lock_released"] is True
    assert _read_json(tmp_path / "window" / "released.json")["lock_released"] is True


def test_transfer_window_dry_run_still_uses_window_mode(tmp_path):
    target = tmp_path / "transfer_window"
    result = subprocess.run(
        [
            sys.executable,
            str(TRANSFER_WINDOW_SCRIPT),
            "--window",
            str(target),
            "--segments",
            "k1",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "window —" in result.stdout  # 计划行: 设备锁 : window — ...
    assert not target.exists()


# ---------------------------------------------------------------------------
# 2. segment 模式: 窗口不拿锁, 子段自取自放 (不会父子互锁)
# ---------------------------------------------------------------------------
def test_segment_mode_window_never_locks_and_children_manage_their_own(
    tmp_path, monkeypatch
):
    lock = FakeLock()
    _install_fake_mutex(monkeypatch, lock)
    runner = FakeRunner(lock)

    rc = _run(tmp_path, _segments(tmp_path, 2), monkeypatch, runner,
              lock_mode="segment", wait_s=900.0)

    assert rc == 0
    assert runner.lock_held_on_entry == [False, False]
    assert runner.child_acquire_ok == [True, True]
    assert lock.release_calls == 2  # 每段自己释放
    assert len(lock.acquire_timeouts) == 2  # 窗口一次都没 acquire
    window_payload = _read_json(tmp_path / "window" / "window.json")
    assert window_payload["lock_mode"] == "segment"
    assert window_payload["device_lock"]["status"] == "segment-managed"
    assert [s["status"] for s in window_payload["segment_lock_states"]] == [
        "free",
        "free",
    ]
    released = _read_json(tmp_path / "window" / "released.json")
    assert released["released_by_window"] is False
    assert released["window_held_lock"] is False
    assert released["lock_released"] is True
    assert window_payload["all_ok"] is True
    # 每段自己的证据文件也带 lock_after (段结束时的锁状态)。
    assert _read_json(tmp_path / "window" / "segments" / "seg0.json")["lock_after"][
        "status"
    ] == "free"


# ---------------------------------------------------------------------------
# 3. 失败即停: 后续段不跑
# ---------------------------------------------------------------------------
def test_segment_failure_stops_window(tmp_path, monkeypatch):
    lock = FakeLock()
    _install_fake_mutex(monkeypatch, lock)
    runner = FakeRunner(lock, fail_on="seg1")

    rc = _run(tmp_path, _segments(tmp_path, 3), monkeypatch, runner,
              lock_mode="segment")

    assert rc == 1
    assert runner.calls == ["seg0", "seg1"]
    payload = _read_json(tmp_path / "window" / "window.json")
    assert payload["stopped_after"] == "seg1"
    assert payload["stop_reason"] == "segment-failed"
    assert len(payload["cooldowns"]) == 1
    assert payload["all_ok"] is False


# ---------------------------------------------------------------------------
# 4. 段间冷却: 每两段之间恰好一次, 秒数原样
# ---------------------------------------------------------------------------
def test_cooldown_called_once_between_each_pair(tmp_path, monkeypatch):
    lock = FakeLock()
    _install_fake_mutex(monkeypatch, lock)
    runner = FakeRunner(lock)

    calls: list[tuple[float, str]] = []
    real_cooldown = window.cooldown

    def spy(seconds, *, after, lock):
        calls.append((seconds, after))
        return real_cooldown(0.0, after=after, lock=lock)

    monkeypatch.setattr(window, "cooldown", spy)
    rc = _run(tmp_path, _segments(tmp_path, 3), monkeypatch, runner,
              lock_mode="segment", cooldown_s=123.0)

    assert rc == 0
    assert calls == [(123.0, "seg0"), (123.0, "seg1")]


def test_real_cooldown_records_elapsed(tmp_path, monkeypatch):
    lock = FakeLock()
    _install_fake_mutex(monkeypatch, lock)
    runner = FakeRunner(lock)

    rc = _run(tmp_path, _segments(tmp_path, 2), monkeypatch, runner,
              lock_mode="segment", cooldown_s=0.05)

    assert rc == 0
    cool = _read_json(tmp_path / "window" / "window.json")["cooldowns"][0]
    assert cool["skipped"] is False
    assert cool["elapsed_s"] >= 0.0


# ---------------------------------------------------------------------------
# 5. 最终锁证据: 锁仍被别人占着 ⇒ 记录 + 停止, 绝不代解锁
# ---------------------------------------------------------------------------
def test_foreign_lock_after_segment_is_recorded_never_released(
    tmp_path, monkeypatch
):
    lock = FakeLock()
    _install_fake_mutex(monkeypatch, lock)
    runner = FakeRunner(lock, hold_lock_after="seg0")

    rc = _run(tmp_path, _segments(tmp_path, 3), monkeypatch, runner,
              lock_mode="segment")

    assert rc == 1
    assert runner.calls == ["seg0"]
    assert lock.release_calls == 0  # 窗口没有释放任何锁
    payload = _read_json(tmp_path / "window" / "window.json")
    assert payload["stopped_after"] == "seg0"
    assert payload["stop_reason"] == "lock-not-released"
    state = payload["segment_lock_states"][0]
    assert state["status"] == "held"
    assert state["holder"]["pid"] == 4321
    released = _read_json(tmp_path / "window" / "released.json")
    assert released["lock_released"] is False
    assert released["final_state"]["status"] == "held"
    assert released["released_by_window"] is False
    assert payload["all_ok"] is False


def test_unknown_lock_mode_refused_without_creating_dirs(tmp_path):
    rc = window.run_window(
        _segments(tmp_path, 1),
        window=tmp_path / "window",
        cooldown_s=0.0,
        wait_s=1.0,
        timeout_s=1.0,
        slm_number=1,
        cam_id=0,
        execute=True,
        lock_mode="bogus",
    )
    assert rc == 2
    assert not (tmp_path / "window").exists()


# ---------------------------------------------------------------------------
# 6. dry-run: 不建目录、不 import 设备模块、转发真实 wait
# ---------------------------------------------------------------------------
_DRY_RUN_CODE = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("iss238_dry", r"{script}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
rc = module.main(["--run-dir", r"{target}", "--wait-until-idle", "900"])
print("RC", rc)
print("DEVICE_IMPORTED", "ao_shaping.tools.kernels.device_mutex" in sys.modules)
"""


def test_period_sweep_dry_run_is_side_effect_free_and_forwards_wait(tmp_path):
    target = tmp_path / "run3_period_sweep"
    code = _DRY_RUN_CODE.format(script=str(PERIOD_SWEEP_SCRIPT), target=str(target))
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "RC 0" in result.stdout
    assert "DEVICE_IMPORTED False" in result.stdout
    assert not target.exists()

    # 计划里写明 segment 模式, 子段 argv 转发真实等待值 (不再是硬编码 0)。
    assert "segment —" in result.stdout
    assert "--wait-until-idle 900" in result.stdout
    assert "--wait-until-idle 0" not in result.stdout


def test_period_sweep_segments_carry_real_wait_and_segment_mode(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "issue238_period_sweep_under_test", PERIOD_SWEEP_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    parser_args = [
        "--run-dir",
        str(tmp_path / "window"),
        "--periods-px",
        "32",
        "48",
        "--wait-until-idle",
        "4321",
    ]
    # build_segments 只需要 Namespace; 用其 main 的 parser 复用默认值。
    namespace = argparse.Namespace(
        run_dir=str(tmp_path / "window"),
        periods_px=[32, 48],
        frames_per_level=20,
        dense_lo=None,
        dense_hi=None,
        dense_step=None,
        checker_frames=None,
        settle_s=None,
        exposure_bracket=[0.02, 0.04, 0.08, 0.16],
        wait_until_idle=4321.0,
        slm_number=1,
        cam_id=0,
    )
    segments = module.build_segments(namespace, tmp_path / "window")
    windows = [
        seg.argv[seg.argv.index("--wait-until-idle") + 1] for seg in segments
    ]
    assert windows == ["4321.0", "4321.0"]
    assert module.LOCK_MODE_SEGMENT == "segment"

    # main() 的 dry-run 即便参数解析路径不同, 也不该硬编码 0。
    result = subprocess.run(
        [sys.executable, str(PERIOD_SWEEP_SCRIPT), *parser_args],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "window").exists()
