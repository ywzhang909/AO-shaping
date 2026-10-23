"""设备互斥的三个调用方: 迁移后的契约 (不碰硬件)。

迁移清单上的三条 (``probe_optical_path_validity`` / ``model_in_loop_hw_ab`` /
``capture_prior_phase_dataset``) 都必须:

* 用**权威模块**的规范等待 (:func:`ao_shaping.tools.kernels.device_mutex.wait_until_idle`),
  而不是自己再写一份「探一次 / 猜一个占用者」的循环;
* 把设备锁**拿在手里**直到采集结束 (只等到「空闲」就撒手, 另一个会话会在中途
  抢走句柄);
* 在 ``finally`` 里放锁 —— 抛异常也不许把设备锁住在自己进程里;
* 认出 **SDK 陈旧状态** (锁空 + ``-10003``) 并给出**物理断电**的处置, 而不是空等。

``capture_prior_phase_dataset`` 还多一条: 它**以前没有等待能力** (撞上 -10003
就崩), 现在有 ``--wait-until-idle``。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.kernels.device_mutex import (
    KIND_SDK_STALE,
    KIND_SLM_BUSY,
    DeviceLock,
    WaitVerdict,
)

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, relative: str):
    """按路径加载一个脚本模块 (``scripts/`` 是脚本目录, 不是包)。"""
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe_script = _load("probe_optical_path_validity_lock", "scripts/slm/probe_optical_path_validity.py")
campaign = _load("model_in_loop_hw_ab_lock", "scripts/slm/model_in_loop_hw_ab.py")
capture = _load("capture_prior_phase_dataset_lock", "scripts/slm/capture_prior_phase_dataset.py")


# ---------------------------------------------------------------------------
# 假的等待结论 (真设备一律不碰)
# ---------------------------------------------------------------------------
def _ok_verdict(lock: DeviceLock, *, waited_s: float = 1.0) -> WaitVerdict:
    return WaitVerdict(
        ok=True,
        status="idle",
        exit_code=0,
        waited_s=waited_s,
        lock=lock,
    )


def _stale_verdict() -> WaitVerdict:
    return WaitVerdict(
        ok=False,
        status="stale",
        exit_code=1,
        reason="SDK 报设备被占, 但设备锁是空的 (没有任何活着的持有者) —— 陈旧句柄",
        detail="SDK 报设备被占, 但设备锁是空的 (没有任何活着的持有者) —— 陈旧句柄",
        remedy="给 SLM 控制器物理断电重置 (断电几秒再上电) —— 排队等不来",
        kind=KIND_SDK_STALE,
        code=-10003,
    )


def _busy_verdict(kind: str, detail: str) -> WaitVerdict:
    return WaitVerdict(
        ok=False,
        status="timeout",
        exit_code=1,
        reason=detail,
        detail=detail,
        kind=kind,
    )


# ---------------------------------------------------------------------------
# capture_prior_phase_dataset: 新增 --wait-until-idle
# ---------------------------------------------------------------------------
def _write_focus(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "focus": {"best_focal_mm": 1783.5, "best_diopter_m1": 0.56},
        "settings": {
            "center_x": 960,
            "center_y": 600,
            "radius": 450,
            "wavelength": 1064,
            "slm_number": 1,
            "cam_type": "daheng",
            "cam_id": 0,
        },
    }
    (directory / "focus.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return directory


def _write_candidates(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "candidates").mkdir(exist_ok=True)
    name = "main00.npy"
    np.save(directory / "candidates" / name,
            np.random.default_rng(0).normal(0, 1.5, (128, 128)))
    payload = {
        "candidates": [
            {
                "index": 0,
                "region": 128,
                "phase_file": f"candidates/{name}",
                "score": 0.9,
                "source": "test",
            }
        ],
        "secondary_groups": [],
    }
    (directory / "candidates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return directory


@pytest.fixture
def capture_workspace(tmp_path):
    return _write_focus(tmp_path / "focus"), _write_candidates(tmp_path / "audit")


def _capture_argv(tmp_path, workspace, extra=()) -> list[str]:
    focus, candidates = workspace
    return [
        "capture_prior_phase_dataset.py",
        "--output", str(tmp_path / "out"),
        "--focus-directory", str(focus),
        "--candidates", str(candidates),
        "--count", "6",
        "--region", "128",
        *extra,
    ]


class TestCaptureWaitUntilIdle:
    """``capture_prior_phase_dataset`` 以前**没有**等待能力, 现在有。"""

    def test_the_flag_exists_and_defaults_to_no_waiting(self):
        parser_args = [
            "--output", "o", "--focus-directory", "f", "--candidates", "c"
        ]
        args = _parse_capture(parser_args)
        assert args.wait_until_idle == 0.0

    def test_zero_budget_keeps_the_old_behaviour(self, tmp_path, capture_workspace):
        """0 = 不等: 不探针、不拿锁, 直接开设备 (旧行为)。"""
        args = _parse_capture(_capture_argv(tmp_path, capture_workspace))
        plan: dict = {}

        lock = capture.acquire_device_lock(args, {"slm_number": 1, "cam_id": 0,
                                                  "cam_type": "daheng"}, plan)

        assert lock is None
        assert plan["device_wait"]["status"] == "skipped"

    def test_the_plan_records_the_wait(self, tmp_path, capture_workspace, capsys):
        """计划里要留下等待的预算 (报告要能回答「当时等了多久」)。"""
        args = _parse_capture(
            _capture_argv(tmp_path, capture_workspace, ["--wait-until-idle", "120"])
        )
        assert args.wait_until_idle == 120.0

    def test_an_ok_verdict_yields_a_held_lock(self, tmp_path, monkeypatch):
        args = _parse_capture([
            "--output", "o", "--focus-directory", "f", "--candidates", "c",
            "--wait-until-idle", "120",
        ])
        fake_lock = DeviceLock(tmp_path / "bench.lock")
        monkeypatch.setattr(
            capture.device_mutex, "wait_until_idle", lambda *a, **kw: _ok_verdict(fake_lock)
        )
        plan: dict = {}

        lock = capture.acquire_device_lock(
            args, {"slm_number": 1, "cam_id": 0, "cam_type": "daheng"}, plan
        )
        try:
            assert lock is fake_lock
            assert lock.is_locked(), "拿到的锁必须是**已经持有**的"
            assert plan["device_wait"]["status"] == "idle"
            assert plan["device_wait"]["budget_s"] == 120.0
        finally:
            lock.release()

    def test_a_stale_verdict_refuses_to_wait(self, monkeypatch):
        """锁空 + -10003 ⇒ SDK 陈旧状态 ⇒ 提示物理断电, 不空等。"""
        args = _parse_capture([
            "--output", "o", "--focus-directory", "f", "--candidates", "c",
            "--wait-until-idle", "1800",
        ])
        monkeypatch.setattr(
            capture.device_mutex, "wait_until_idle", lambda *a, **kw: _stale_verdict()
        )
        plan: dict = {}

        with pytest.raises(SystemExit) as excinfo:
            capture.acquire_device_lock(
                args, {"slm_number": 1, "cam_id": 0, "cam_type": "daheng"}, plan
            )

        message = str(excinfo.value)
        assert "物理断电" in message
        assert plan["device_wait"]["status"] == "stale"

    def test_a_timeout_tells_the_operator_to_queue(self, monkeypatch):
        args = _parse_capture([
            "--output", "o", "--focus-directory", "f", "--candidates", "c",
            "--wait-until-idle", "30",
        ])
        monkeypatch.setattr(
            capture.device_mutex,
            "wait_until_idle",
            lambda *a, **kw: _busy_verdict(KIND_SLM_BUSY, "别的会话在用设备"),
        )
        with pytest.raises(SystemExit) as excinfo:
            capture.acquire_device_lock(
                args, {"slm_number": 1, "cam_id": 0, "cam_type": "daheng"}, {}
            )
        assert "--wait-until-idle" in str(excinfo.value)

    def test_execute_with_a_stale_bench_stops_before_opening_anything(
        self, tmp_path, capture_workspace, monkeypatch
    ):
        """端到端: 设备陈旧时连设备都不开 (只落 run_config 就退)。"""
        monkeypatch.setattr(
            capture.device_mutex, "wait_until_idle", lambda *a, **kw: _stale_verdict()
        )
        opened: list[str] = []

        def _boom(*args, **kwargs):
            opened.append("device")
            raise AssertionError("设备陈旧时不许开设备")

        monkeypatch.setattr("ao_shaping.drivers.ccd.common.create_camera", _boom)
        monkeypatch.setattr("ao_shaping.drivers.slm.santec.Santec", _boom)

        argv = _capture_argv(
            tmp_path, capture_workspace, ["--execute", "--wait-until-idle", "60"]
        )
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as excinfo:
            capture.main()
        assert "物理断电" in str(excinfo.value)
        assert opened == []
        config = json.loads((tmp_path / "out" / "run_config.json").read_text(encoding="utf-8"))
        assert config["device_wait"]["status"] == "stale"


def _parse_capture(argv: list[str]):
    """解析 ``capture_prior_phase_dataset`` 的参数 (用脚本自己的 parser)。

    ``main`` 走 ``sys.argv``, 那份列表里带脚本名; 单独写的参数列表则不带 ⇒
    第一个不是选项的词当脚本名去掉。
    """
    if argv and not argv[0].startswith("-"):
        argv = argv[1:]
    return capture.build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# probe_optical_path_validity
# ---------------------------------------------------------------------------
class TestOpticalPathProbeLock:
    def test_acquire_devices_returns_a_held_lock(self, tmp_path, monkeypatch):
        fake_lock = DeviceLock(tmp_path / "bench.lock")
        monkeypatch.setattr(
            probe_script.device_mutex, "wait_until_idle", lambda *a, **kw: _ok_verdict(fake_lock)
        )
        args = probe_script.build_parser().parse_args([])

        lock = probe_script.acquire_devices(args)
        try:
            assert lock is fake_lock
            assert lock.is_locked()
        finally:
            lock.release()

    def test_a_stale_bench_is_named_not_waited_out(self, monkeypatch):
        monkeypatch.setattr(
            probe_script.device_mutex, "wait_until_idle", lambda *a, **kw: _stale_verdict()
        )
        args = probe_script.build_parser().parse_args(["--wait-until-idle", "1800"])

        with pytest.raises(SystemExit) as excinfo:
            probe_script.acquire_devices(args)
        assert "物理断电" in str(excinfo.value)

    def test_the_wait_is_asked_for_the_same_device_as_the_probe(self, tmp_path, monkeypatch):
        seen: dict = {}

        def _fake(budget, **kwargs):
            seen["budget"] = budget
            seen.update(kwargs)
            return _ok_verdict(DeviceLock(tmp_path / "bench.lock"), waited_s=0.0)

        monkeypatch.setattr(probe_script.device_mutex, "wait_until_idle", _fake)
        args = probe_script.build_parser().parse_args(
            ["--wait-until-idle", "42", "--slm-number", "1", "--cam-id", "0"]
        )
        lock = probe_script.acquire_devices(args)
        lock.release()

        assert seen["budget"] == 42.0
        assert seen["slm_number"] == 1
        assert seen["cam_id"] == 0
        assert str(Path(probe_script.__file__).resolve()) in [str(p) for p in seen["own_paths"]]

    def test_main_releases_the_lock_when_the_probe_body_raises(self, monkeypatch, tmp_path):
        lock = DeviceLock(tmp_path / "bench.lock")

        def _acquire(args):  # noqa: ARG001
            assert lock.acquire(timeout=0.0)
            return lock

        def _boom(args):  # noqa: ARG001
            assert lock.is_locked(), "采集主体跑的时候锁必须还在手里"
            raise RuntimeError("probe body failed")

        monkeypatch.setattr(probe_script, "acquire_devices", _acquire)
        monkeypatch.setattr(probe_script, "run_probe", _boom)

        with pytest.raises(RuntimeError):
            probe_script.main([])
        assert lock.is_locked() is False, "抛异常也必须放锁"


# ---------------------------------------------------------------------------
# model_in_loop_hw_ab
# ---------------------------------------------------------------------------
class TestCampaignLock:
    def test_zero_budget_disables_waiting(self):
        args = campaign.parse_args(["--wait-until-idle", "0"])
        lock, record = campaign.acquire_device_lock(args)
        assert lock is None
        assert record == "skipped"

    def test_an_ok_verdict_yields_a_held_lock(self, tmp_path, monkeypatch):
        fake_lock = DeviceLock(tmp_path / "bench.lock")
        monkeypatch.setattr(
            campaign.device_mutex,
            "wait_until_idle",
            lambda *a, **kw: _ok_verdict(fake_lock),
        )
        args = campaign.parse_args(["--wait-until-idle", "600"])

        lock, record = campaign.acquire_device_lock(args)
        try:
            assert lock is fake_lock
            assert lock.is_locked()
            assert record == "idle after waiting"
        finally:
            lock.release()

    def test_a_stale_verdict_is_refused_with_the_power_cycle_advice(self, monkeypatch):
        monkeypatch.setattr(
            campaign.device_mutex, "wait_until_idle", lambda *a, **kw: _stale_verdict()
        )
        args = campaign.parse_args(["--wait-until-idle", "600"])

        with pytest.raises(RuntimeError) as excinfo:
            campaign.acquire_device_lock(args)
        assert "物理断电" in str(excinfo.value)

    def test_the_campaign_releases_the_lock_after_it_finishes(self, tmp_path, monkeypatch):
        fake_lock = DeviceLock(tmp_path / "bench.lock")
        monkeypatch.setattr(
            campaign.device_mutex,
            "wait_until_idle",
            lambda *a, **kw: _ok_verdict(fake_lock),
        )
        seen: dict = {}

        def _fake_campaign(args, plan, session_dir, manifest):  # noqa: ARG001
            seen["held"] = fake_lock.is_locked()
            return 0

        monkeypatch.setattr(campaign, "run_campaign", _fake_campaign)

        rc = campaign.main(["--out-root", str(tmp_path / "campaign"),
                            "--wait-until-idle", "600", "--stage", "light"])

        assert rc == 0
        assert seen["held"] is True, "整轮跑的时候锁必须在手里"
        assert fake_lock.is_locked() is False, "跑完必须放锁"

    def test_a_stale_bench_marks_the_campaign_device_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            campaign.device_mutex, "wait_until_idle", lambda *a, **kw: _stale_verdict()
        )

        rc = campaign.main(["--out-root", str(tmp_path / "campaign"),
                            "--wait-until-idle", "600", "--stage", "light"])

        assert rc == 2
        manifest_path = next((tmp_path / "campaign").glob("*/manifest.json"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["campaign_state"] == "device_unavailable"
        assert any("物理断电" in note for note in manifest["notes"])
