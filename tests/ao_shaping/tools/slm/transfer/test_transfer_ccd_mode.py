"""``ccd_mode`` 三档 (full / light / none) 的计划、CLI 接线与免设备行为。

三件事:

1. ``plan()`` 的阶段表逐档固定: ``full`` 一字不变, ``light`` 去掉实测模型块,
   ``none`` 用纯 SLM 上屏替换全部 CCD 阶段;
2. CLI ``--ccd-mode`` / ``--open-loop-dwell-s`` 真的进了 ``TransferSession``;
3. ``light`` 每轮只跑一次锚定块 (``updates=1``, ``abba=True``), ``none`` 先离线
   合成再只开 SLM 上屏 (相机全程关闭, 锁与句柄都在 ``finally`` 里收尾)。

``light``/``none`` 缺 dataset 或 checkpoint 时必须当场失败, 不允许悄悄退化成
采集 + 拟合。
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer import open_loop
from ao_shaping.tools.slm.transfer import workflow as workflow_module
from ao_shaping.tools.slm.transfer.workflow import (
    TransferSession,
    main,
    run_session,
)


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def _recorded_model(tmp_path):
    """一个"已录数据集 + 冻结模型"的最小现场 (只要求路径存在)。"""
    dataset = tmp_path / "recorded"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("[]", encoding="utf-8")
    checkpoint = tmp_path / "frozen_parameters.json"
    checkpoint.write_text(
        json.dumps(
            dict(
                ramp_matrix=[[1.0, 0.0], [0.0, 1.0]],
                grid_size=8,
                fft_size=16,
                pixel_samples=3,
            )
        ),
        encoding="utf-8",
    )
    return dataset, checkpoint


def _session(tmp_path, name, **over):
    dataset, checkpoint = _recorded_model(tmp_path)
    return TransferSession(
        tmp_path / name,
        dataset=dataset,
        checkpoint=checkpoint,
        shapes=("square",),
        count=6,
        frames=3,
        iterations=1,
        blocks=1,
        updates=1,
        **over,
    )


class _FakeSLM:
    """只记录生命周期与上屏数据, 不碰任何设备。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.opened = False
        self.closed = False
        self.panels: list[np.ndarray] = []

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def create_phase_from_array(self, phase):
        return np.asarray(phase)

    def display_data(self, panel):
        assert self.opened and not self.closed
        self.panels.append(np.asarray(panel))


class _FakeLock:
    def __init__(self):
        self.acquisitions: list[float] = []
        self.releases = 0

    def acquire(self, timeout=None):
        self.acquisitions.append(timeout)
        return True

    def release(self):
        self.releases += 1


def _patch_session(monkeypatch, seen, *, slm=None, lock=None, display_fails=False):
    """把 run_session 的所有设备/重活替换成记录器, 保留真实控制流。"""
    from ao_shaping.tools.slm.transfer import (
        artifacts,
        capture,
        direction,
        fit,
        latest_model,
        summary,
        validate,
    )

    def fake_fit(args):
        seen.setdefault("fit", []).append(args)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "frozen_parameters.json").write_text(
            json.dumps(dict(grid_size=8, fft_size=16, pixel_samples=3)),
            encoding="utf-8",
        )

    def fake_synthesize(config, *, checkpoint, zero, reference, out, device):
        seen["synthesize"] = dict(checkpoint=checkpoint, out=out, device=device)
        for shape in config.shapes:
            np.savez_compressed(
                out / f"{shape}_offline_seed.npz",
                optimized_phase=np.full((8, 8), 0.25, np.float32),
            )

    monkeypatch.setattr(fit, "fit", fake_fit)
    monkeypatch.setattr(
        workflow_module, "_round_geometry_preflight",
        lambda config, *, dataset, checkpoint: (
            [0.0, 0.0],
            checkpoint.parent / "reference_start.npz",
        ),
    )
    monkeypatch.setattr(workflow_module, "_write_reference_probe", lambda *a, **k: None)
    monkeypatch.setattr(workflow_module, "_offline_synthesize", fake_synthesize)
    monkeypatch.setattr(latest_model, "save_latest_model", lambda *a, **k: None)
    monkeypatch.setattr(latest_model, "serial_from_capture", lambda out: None)
    monkeypatch.setattr(
        capture, "capture", lambda *a, **k: pytest.fail("capture must not run")
    )
    monkeypatch.setattr(
        capture,
        "precheck_light",
        lambda **k: seen.setdefault("light_check", []).append(k) or dict(status="ok"),
    )
    monkeypatch.setattr(
        validate, "run", lambda args: seen.setdefault("validate", []).append(args) or {}
    )
    monkeypatch.setattr(artifacts, "audit_run", lambda run: dict(status="passed"))
    monkeypatch.setattr(
        summary,
        "summarize",
        lambda run, out: out.mkdir(parents=True, exist_ok=True)
        or (out / "summary.json").write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(direction, "analyze", lambda run, out: None)

    # open-loop 的两个注入点: 默认工厂与默认锁。
    fake_slm = slm or _FakeSLM()
    calls: list[dict] = []

    def factory(**kwargs):
        calls.append(kwargs)
        if display_fails:
            fake_slm.display_data = lambda panel: (_ for _ in ()).throw(
                RuntimeError("panel write failed")
            )
        return fake_slm

    events: dict = dict(factory_calls=calls, slm=fake_slm)
    if lock is not None:
        events["lock"] = lock
    monkeypatch.setattr(open_loop, "_default_slm_factory", factory)
    monkeypatch.setattr(open_loop, "_default_lock", lambda: lock or _FakeLock())
    monkeypatch.setattr(time, "sleep", lambda seconds: seen.setdefault("sleep", []).append(seconds))
    seen["open_loop"] = events
    return events


def _journal(out):
    return [
        entry["stage"]
        for entry in json.loads((out / "journal.json").read_text(encoding="utf-8"))
    ]


# ---------------------------------------------------------------------------
# 1. plan(): 三档阶段表
# ---------------------------------------------------------------------------


FULL_STAGES = [
    "capture",
    "nominal_initialization",
    "fit",
    "CCD_model_blocks",
    "exposure_response_capture",
    "exposure_response_analysis",
    "blur_calibration",
    "CCD_anchored_blocks",
    "raw_data_audit",
    "summary",
]


def test_plan_full_is_the_historical_plan(tmp_path):
    plan = TransferSession(tmp_path / "full", reuse_latest=False).plan()
    assert plan["stages"] == FULL_STAGES
    # ``full`` 的 settings/resolved 一字不变: 两个新字段都不出现。
    assert "ccd_mode" not in plan["settings"]
    assert "open_loop_dwell_s" not in plan["settings"]
    assert "ccd_mode" not in plan["resolved"]
    assert plan["resolved"]["acceptance"] == "model_only"


def test_plan_light_drops_the_measured_model_stage(tmp_path):
    dataset, checkpoint = _recorded_model(tmp_path)
    config = TransferSession(
        tmp_path / "light", dataset=dataset, checkpoint=checkpoint, ccd_mode="light"
    )
    plan = config.plan()
    assert plan["stages"] == [
        "reuse_capture",
        "fit",
        "CCD_anchored_blocks",
        "raw_data_audit",
        "summary",
    ]
    assert "CCD_model_blocks" not in plan["stages"]
    assert plan["settings"]["ccd_mode"] == "light"
    assert plan["resolved"]["ccd_mode"] == "light"
    # light 仍是实测验收: 每块单次锚点 + 终检 A–B–B–A, 不是 open_loop。
    assert plan["resolved"]["acceptance"] == "paired_abba"


def test_plan_none_replaces_every_ccd_stage_with_the_open_loop_display(tmp_path):
    dataset, checkpoint = _recorded_model(tmp_path)
    config = TransferSession(
        tmp_path / "none",
        dataset=dataset,
        checkpoint=checkpoint,
        ccd_mode="none",
        open_loop_dwell_s=0.5,
    )
    plan = config.plan()
    assert plan["stages"] == [
        "reuse_capture",
        "fit",
        "open_loop_slm_display",
        "summary",
    ]
    for dropped in (
        "CCD_model_blocks",
        "CCD_anchored_blocks",
        "raw_data_audit",
        "exposure_response_capture",
    ):
        assert dropped not in plan["stages"]
    assert plan["settings"]["ccd_mode"] == "none"
    assert plan["settings"]["open_loop_dwell_s"] == pytest.approx(0.5)
    assert plan["resolved"]["ccd_mode"] == "none"
    assert plan["resolved"]["acceptance"] == "open_loop"


def test_illegal_ccd_mode_is_rejected_before_anything_runs(tmp_path):
    config = TransferSession(tmp_path / "bad", ccd_mode="partial")
    with pytest.raises(ValueError, match="ccd_mode"):
        config.plan()
    with pytest.raises(ValueError, match="ccd_mode"):
        run_session(config)
    assert not (tmp_path / "bad").exists()


# ---------------------------------------------------------------------------
# 2. CLI 接线
# ---------------------------------------------------------------------------


def test_cli_forwards_ccd_mode_and_dwell_into_the_session(tmp_path, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        workflow_module, "run_session", lambda config, offline=False: seen.update(config=config)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "transfer",
            "run",
            "--execute",
            "--out",
            str(tmp_path / "cli"),
            "--ccd-mode",
            "none",
            "--open-loop-dwell-s",
            "1.5",
        ],
    )
    main()
    config = seen["config"]
    assert isinstance(config, TransferSession)
    assert config.ccd_mode == "none"
    assert config.open_loop_dwell_s == pytest.approx(1.5)


def test_cli_default_is_full(tmp_path, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        workflow_module, "run_session", lambda config, offline=False: seen.update(config=config)
    )
    monkeypatch.setattr(
        sys, "argv", ["transfer", "run", "--execute", "--out", str(tmp_path / "cli")]
    )
    main()
    assert seen["config"].ccd_mode == "full"


def test_cli_rejects_an_unknown_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["transfer", "plan", "--out", str(tmp_path / "cli"), "--ccd-mode", "half"]
    )
    with pytest.raises(SystemExit):
        main()


# ---------------------------------------------------------------------------
# 3. light: 只跑一次锚定块, updates=1 / abba=True
# ---------------------------------------------------------------------------


def test_light_runs_one_single_update_anchored_pass(tmp_path, monkeypatch):
    seen: dict = {}
    _patch_session(monkeypatch, seen)
    config = _session(tmp_path, "light", ccd_mode="light")
    run_session(config)

    assert _journal(config.out) == [
        "light_check",
        "fit",
        "CCD_anchored_blocks",
        "raw_data_audit",
        "summary",
    ]
    # 没有采集、没有实测模型块。
    assert "capture" not in seen
    calls = seen["validate"]
    assert len(calls) == 1
    (anchored,) = calls
    assert anchored.mode == "anchored"
    assert anchored.execute is True
    assert anchored.updates == 1
    assert anchored.abba is True
    assert anchored.blocks == config.blocks
    assert all(call.mode != "model" for call in calls)
    # 锚定块的输出目录与历史一致。
    assert anchored.out == config.out / "anchored"


def test_light_never_degrades_to_capture_without_a_checkpoint(tmp_path, monkeypatch):
    seen: dict = {}
    _patch_session(monkeypatch, seen)
    dataset, _checkpoint = _recorded_model(tmp_path)
    config = TransferSession(
        tmp_path / "light-missing",
        dataset=dataset,
        checkpoint=None,
        ccd_mode="light",
        shapes=("square",),
    )
    with pytest.raises(ValueError, match="checkpoint"):
        run_session(config)
    assert not config.out.exists()
    assert "validate" not in seen


def test_none_never_degrades_to_capture_without_a_dataset(tmp_path, monkeypatch):
    seen: dict = {}
    _patch_session(monkeypatch, seen)
    _dataset, checkpoint = _recorded_model(tmp_path)
    config = TransferSession(
        tmp_path / "none-missing",
        dataset=None,
        checkpoint=checkpoint,
        ccd_mode="none",
        shapes=("square",),
    )
    with pytest.raises(ValueError, match="dataset"):
        run_session(config)
    assert not config.out.exists()
    assert "validate" not in seen


# ---------------------------------------------------------------------------
# 4. none: 离线合成 + 纯 SLM 上屏
# ---------------------------------------------------------------------------


def test_none_synthesizes_then_displays_without_a_camera(tmp_path, monkeypatch):
    seen: dict = {}
    lock = _FakeLock()
    events = _patch_session(monkeypatch, seen, lock=lock)
    dataset, checkpoint = _recorded_model(tmp_path)
    config = TransferSession(
        tmp_path / "open-loop",
        dataset=dataset,
        checkpoint=checkpoint,
        ccd_mode="none",
        open_loop_dwell_s=0.25,
        shapes=("square", "rectangle"),
        count=6,
        frames=3,
        iterations=1,
    )
    run_session(config)

    # 相机一次都没开: 采集/轻检/实测块全部缺席。
    assert "capture" not in seen
    assert "light_check" not in seen
    assert "validate" not in seen
    assert _journal(config.out) == ["fit", "open_loop_slm_display", "summary"]

    # 先离线合成, 再上屏。
    assert seen["synthesize"]["out"] == config.out

    # SLM 只构造一次, 打开一次, 关闭一次, 上屏顺序 == shapes。
    assert len(events["factory_calls"]) == 1
    assert events["factory_calls"][0]["slm_number"] == config.slm_number
    slm = events["slm"]
    assert slm.opened is True
    assert slm.closed is True
    assert len(slm.panels) == 2
    assert seen["sleep"] == [pytest.approx(0.25), pytest.approx(0.25)]

    # 锁在开 SLM 之前拿, 在关 SLM 之后放。
    assert lock.acquisitions == [1800]
    assert lock.releases == 1

    payload = json.loads(
        (config.out / "open_loop_display.json").read_text(encoding="utf-8")
    )
    assert payload["mode"] == "open_loop"
    assert payload["ccd_mode"] == "none"
    assert payload["acceptance"] == "open_loop"
    assert payload["shapes"] == ["square", "rectangle"]
    assert [entry["shape"] for entry in payload["entries"]] == ["square", "rectangle"]
    assert all(entry["dwell_s"] == pytest.approx(0.25) for entry in payload["entries"])
    assert all("timestamp_utc" in entry for entry in payload["entries"])

    summary = json.loads(
        (config.out / "summary" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["acceptance"] == "open_loop"
    assert summary["ccd_mode"] == "none"

    session = json.loads((config.out / "session.json").read_text(encoding="utf-8"))
    assert session["resolved"]["ccd_mode"] == "none"
    assert session["resolved"]["acceptance"] == "open_loop"
    assert "CCD_anchored_blocks" not in session["stages"]


def test_none_closes_the_slm_and_releases_the_lock_on_failure(tmp_path, monkeypatch):
    seen: dict = {}
    lock = _FakeLock()
    events = _patch_session(monkeypatch, seen, lock=lock, display_fails=True)
    dataset, checkpoint = _recorded_model(tmp_path)
    config = TransferSession(
        tmp_path / "open-loop-fail",
        dataset=dataset,
        checkpoint=checkpoint,
        ccd_mode="none",
        shapes=("square",),
        count=6,
        frames=3,
        iterations=1,
    )
    with pytest.raises(RuntimeError, match="panel write failed"):
        run_session(config)

    assert events["slm"].closed is True
    assert lock.releases == 1
    # 上屏失败就没有记录文件, 但阶段本身已被记成 failed。
    journal = json.loads((config.out / "journal.json").read_text(encoding="utf-8"))
    failed = [entry for entry in journal if entry["status"] == "failed"]
    assert [entry["stage"] for entry in failed] == ["open_loop_slm_display"]


def test_open_loop_rejects_a_non_positive_dwell(tmp_path):
    with pytest.raises(ValueError, match="dwell_s"):
        open_loop.display_phases(
            ("square",),
            source=tmp_path,
            out_path=tmp_path / "open_loop_display.json",
            dwell_s=0.0,
            slm_factory=lambda **k: _FakeSLM(),
            lock=_FakeLock(),
            sleep=lambda seconds: None,
        )
