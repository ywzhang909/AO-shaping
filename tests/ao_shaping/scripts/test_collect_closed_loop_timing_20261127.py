# -*- coding: utf-8 -*-
"""离线门禁: GUI 闭环逐阶段耗时采集器（§5 第 1 项数据缺口）。

覆盖:
- ``TimingBench`` 只包装不改变调用顺序，逐次记录 apply_phase/capture 耗时;
- ``Timeline.stage_timings`` 按事件对拆出 fit/adapt/candidate;
- ``_round_rows`` 把帧采样与阶段计时按 phase_id 合并，baseline 阶段为 null;
- plan-only 不创建输出目录、不导入设备模块;
- 适配参数缺失在占设备之前就被拦下。

不打开设备、不加锁、不写仓库内文件。
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPT = importlib.import_module("scripts.slm.collect_closed_loop_timing_20261127")


class _InnerBench:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def open(self) -> None:
        self.calls.append("open")

    def describe(self) -> dict[str, object]:
        return {"name": "fake"}

    def apply_phase(self, phase: object) -> str:
        self.calls.append("apply_phase")
        return "ok"

    def capture(self) -> object:
        self.calls.append("capture")
        return "frame"

    def close(self) -> None:
        self.calls.append("close")


def test_timing_bench_wraps_without_reordering() -> None:
    inner = _InnerBench()
    bench = SCRIPT.TimingBench(inner)
    bench.open()
    assert bench.apply_phase(object()) == "ok"
    assert bench.capture() == "frame"
    assert bench.describe() == {"name": "fake"}
    bench.close()

    assert inner.calls == ["open", "apply_phase", "capture", "close"]
    assert bench.apply_calls == 1 and bench.capture_calls == 1
    assert bench.last_apply_ms is not None and bench.last_apply_ms >= 0.0
    assert bench.last_capture_ms is not None and bench.last_capture_ms >= 0.0


def test_stage_timings_from_event_pairs() -> None:
    timeline = SCRIPT.Timeline()
    timeline.events = [
        {"t_ms": 10.0, "kind": "state", "state": "updating", "note": "适配重拟合"},
        {"t_ms": 30.0, "kind": "adaptation", "state": None, "note": None},
        {"t_ms": 40.0, "kind": "state", "state": "updating", "note": "生成第 1 轮候选"},
        {"t_ms": 90.0, "kind": "state", "state": "measuring", "note": "下发第 1 轮候选"},
        {"t_ms": 95.0, "kind": "state", "state": "updating", "note": "适配重拟合"},
        {"t_ms": 105.0, "kind": "adaptation", "state": None, "note": None},
        {"t_ms": 110.0, "kind": "state", "state": "updating", "note": "生成第 2 轮候选"},
        {"t_ms": 150.0, "kind": "state", "state": "measuring", "note": "下发第 2 轮候选"},
    ]
    stages = timeline.stage_timings()
    assert stages["r01"] == {"fit_ms": 20.0, "adapt_ms": 10.0, "candidate_ms": 50.0}
    assert stages["r02"] == {"fit_ms": 10.0, "adapt_ms": 5.0, "candidate_ms": 40.0}
    assert "__refit" not in stages


def test_round_rows_join_stages_and_frames() -> None:
    timeline = SCRIPT.Timeline()
    timeline.events = [
        {"t_ms": 0.0, "kind": "state", "state": None, "note": "适配重拟合"},
        {"t_ms": 5.0, "kind": "adaptation", "state": None, "note": None},
        {"t_ms": 9.0, "kind": "state", "state": None, "note": "生成第 1 轮候选"},
        {"t_ms": 20.0, "kind": "state", "state": None, "note": "下发第 1 轮候选"},
    ]
    timeline._frames["r01"] = {"t_ms": 20.0, "frame_id": 7, "monotonic_ns": 700}

    def record(phase_id: str, index: int) -> SimpleNamespace:
        return SimpleNamespace(
            round_index=index,
            phase_id=phase_id,
            frame_id=7,
            gate=SimpleNamespace(ok=True, reasons=[]),
            decision="accept",
            adaptation="promoted",
            measured_score=1.1,
            predicted_score=1.2,
            note="",
        )

    summary = SimpleNamespace(rounds=[record("r00", 0), record("r01", 1)])
    bench = SCRIPT.TimingBench(_InnerBench())
    bench.last_apply_ms = 3.0
    bench.last_capture_ms = 4.0

    rows = SCRIPT._round_rows(summary.rounds, timeline, bench, "2026-11-27T00:00:00+00:00")
    assert len(rows) == 2
    baseline, first = rows
    assert baseline["phase_id"] == "r00"
    assert baseline["fit_ms"] is None and baseline["capture_ms"] is None
    assert first["phase_id"] == "r01"
    assert first["fit_ms"] == 5.0
    assert first["adapt_ms"] == 4.0
    assert first["candidate_ms"] == 11.0
    assert first["apply_phase_ms"] == 3.0
    assert first["ccd_capture_ms"] == 4.0
    assert first["capture_ms"] == 7.0
    json.dumps(rows, ensure_ascii=False)


def test_plan_mode_does_not_create_out(tmp_path: Path) -> None:
    out = tmp_path / "must_not_exist"
    assert SCRIPT.main(["--out", str(out)]) == 0
    assert not out.exists()


def test_gate_fail_round_is_persisted(tmp_path: Path) -> None:
    """门禁失败轮不进 summary.rounds，也必须从 round 事件兜底落盘。"""

    class DarkBench(_SpotBench):
        def capture(self) -> object:
            import numpy as np

            from ao_shaping.gui.slm.closed_loop_shaping import CapturedFrame

            self.frame_id += 1
            return CapturedFrame(
                frame=np.zeros((1024, 1024), dtype=np.float64),
                frame_id=self.frame_id,
                exposure_ms=20.0,
                full_scale=255.0,
                monotonic_ns=self.frame_id,
            )

    out = tmp_path / "dark"
    exit_code = SCRIPT.main(
        [
            "--out",
            str(out),
            "--execute",
            "--no-adapt",
            "--wait-until-idle",
            "0",
            "--grid-size",
            "8",
            # 合成帧曝光 20 ms，显式匹配，确保 gate 失败来自 low_signal 而非曝光失配
            "--exposure-ms",
            "20",
        ],
        bench_factory=DarkBench,
    )
    assert exit_code == 4

    rows = [
        json.loads(line)
        for line in (out / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["phase_id"] == "r00"
    assert rows[0]["decision"] == "baseline_fail"
    assert rows[0]["gate_ok"] is False
    assert rows[0]["ccd_capture_ms"] is not None


def _adapter_args(tmp_path: Path, dataset: Path) -> object:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}", encoding="utf-8")
    return SCRIPT.build_parser().parse_args(
        [
            "--out",
            str(tmp_path / "run"),
            "--execute",
            "--checkpoint",
            str(checkpoint),
            "--session-dir",
            str(tmp_path / "session"),
            "--source-dataset",
            str(dataset),
        ]
    )


def _write_manifest(dataset: Path, labels: list[str]) -> Path:
    dataset.mkdir(parents=True, exist_ok=True)
    rows = [{"label": label, "file": f"{label}.npz"} for label in labels]
    manifest = dataset / "manifest.json"
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    return manifest


def test_adapter_inputs_validated_before_devices(tmp_path: Path) -> None:
    # source_dataset 必须是目录（seed_from_source 读 <dir>/manifest.json）
    assert SCRIPT.adapter_inputs_error(_adapter_args(tmp_path, tmp_path / "missing")) is not None

    # 目录存在但缺 manifest.json
    empty = tmp_path / "empty"
    empty.mkdir()
    assert SCRIPT.adapter_inputs_error(_adapter_args(tmp_path, empty)) is not None

    # 有 manifest 但缺 reference_start
    no_ref = _write_manifest(
        tmp_path / "no_ref", ["ramp_x_128_1", "ramp_x_128_-1", "ramp_y_128_1", "ramp_y_128_-1"]
    )
    assert SCRIPT.adapter_inputs_error(_adapter_args(tmp_path, no_ref.parent)) is not None

    # 有 reference_start 但缺负 ramp（fit.geometry 会在占设备后抛，须提前挡住）
    no_negative = _write_manifest(
        tmp_path / "no_negative", ["reference_start", "ramp_x_128_1", "ramp_y_128_1"]
    )
    assert SCRIPT.adapter_inputs_error(_adapter_args(tmp_path, no_negative.parent)) is not None

    # 完整合法目录应通过离线校验
    full = _write_manifest(
        tmp_path / "full",
        [
            "reference_start",
            "reference_end",
            "ramp_x_128_1",
            "ramp_x_128_-1",
            "ramp_y_128_1",
            "ramp_y_128_-1",
        ],
    )
    assert SCRIPT.adapter_inputs_error(_adapter_args(tmp_path, full.parent)) is None


def _spot_frame() -> object:
    import numpy as np

    frame = np.zeros((1024, 1024), dtype=np.float64)
    yy, xx = np.indices(frame.shape, dtype=np.float64)
    frame += 100.0 * np.exp(-(((yy - 500.0) ** 2 + (xx - 500.0) ** 2) / 32.0))
    return frame


class _SpotBench:
    """过门禁的假设备：写入/取帧都返回同一稳定光斑。"""

    def __init__(self) -> None:
        self.frame_id = 0

    def open(self) -> None:
        return None

    def describe(self) -> dict[str, object]:
        return {"name": "fake-spot"}

    def apply_phase(self, phase: object) -> str:
        return "ok"

    def capture(self) -> object:
        import numpy as np

        from ao_shaping.gui.slm.closed_loop_shaping import CapturedFrame

        self.frame_id += 1
        return CapturedFrame(
            frame=np.asarray(_spot_frame()),
            frame_id=self.frame_id,
            exposure_ms=20.0,
            full_scale=255.0,
            monotonic_ns=self.frame_id,
        )

    def close(self) -> None:
        return None


class _FakeForward:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape

    def predict(self, phase: object, exposure_ms: float | None = None) -> object:
        import torch

        tensor = phase if torch.is_tensor(phase) else torch.as_tensor(phase)
        return torch.ones(self.shape, dtype=tensor.dtype, device=tensor.device) * (
            0.5 + 0.01 * tensor.mean()
        )


class _FakeAdapter:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape

    def ready(self) -> bool:
        return True

    def refit(self) -> object:
        return SimpleNamespace(promoted=True, status="promoted")

    def forward(self, grid: object) -> object:
        return _FakeForward(self.shape)


def test_execute_with_fake_devices_writes_multiple_rounds(tmp_path: Path) -> None:
    xy = SCRIPT.analysis_grid((500.0, 500.0), 4)
    shape = (int(xy.shape[0]), int(xy.shape[1]))
    out = tmp_path / "run"
    exit_code = SCRIPT.main(
        [
            "--out",
            str(out),
            "--execute",
            "--wait-until-idle",
            "0",
            "--max-rounds",
            "3",
            "--step-iterations",
            "2",
            "--grid-size",
            "8",
            # 合成光斑造在 (500,500)、帧曝光 20 ms，显式覆盖生产默认质心/曝光，保持测试自洽
            "--centre-x",
            "500",
            "--centre-y",
            "500",
            "--exposure-ms",
            "20",
        ],
        bench_factory=_SpotBench,
        adapter_factory=lambda _grid: _FakeAdapter(shape),
    )
    assert exit_code == 0

    rows = [
        json.loads(line)
        for line in (out / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # baseline + 3 轮
    assert len(rows) == 4
    assert [row["phase_id"] for row in rows] == ["r00", "r01", "r02", "r03"]
    assert all(row["schema"] == SCRIPT.ROUNDS_SCHEMA for row in rows)
    assert all(row["gate_ok"] for row in rows)
    assert rows[0]["fit_ms"] is None
    assert all(
        rows[index]["fit_ms"] is not None
        and rows[index]["adapt_ms"] is not None
        and rows[index]["candidate_ms"] is not None
        and rows[index]["capture_ms"] is not None
        for index in (1, 2, 3)
    )

    session = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert session["mode"] == "execute"
    assert session["rounds_recorded"] == 4
    assert session["state"].upper() == "STOPPED"
    assert "最大轮数" in session["reason"]
    assert any(item["note"] == "适配重拟合" for item in session["timeline"])


def test_production_defaults_match_probe_geometry() -> None:
    """生产默认值必须落在探针实测光斑上，且曝光不顶满量程。"""
    args = SCRIPT.build_parser().parse_args(["--out", "x"])
    # 与 18:50 光路探针质心 (1228,1107) / frozen origin (1229.20,1106.23) 一致
    assert args.centre_x == 1228.0
    assert args.centre_y == 1107.0
    # 8-bit Mono8：探针 0.05 ms 峰值约 117，远低于 0.7×255 饱和线
    assert args.exposure_ms == 0.05
    # 分析窗口必须真的落在光斑上（原始帧坐标，无缩放）
    xy = SCRIPT.analysis_grid((args.centre_x, args.centre_y), args.span)
    assert int(round(xy[..., 0].mean())) == 1228
    assert int(round(xy[..., 1].mean())) == 1107


def test_start_phase_aligns_to_checkpoint_grid_size(tmp_path: Path) -> None:
    """start_phase 必须从 checkpoint 的 grid_size 对齐，否则候选生成 shape 失配。

    2026-11-27 上机失败根因：--grid-size 默认 128，但 checkpoint grid_size=640，
    forward.predict 收到 128×128 phase 在 dimension 2 崩 "128 vs 640"。
    """
    ckpt = tmp_path / "frozen_parameters.json"
    ckpt.write_text(json.dumps({"grid_size": 640}), encoding="utf-8")

    assert SCRIPT.checkpoint_grid_size(str(ckpt)) == 640
    # 目录形式（checkpoint 传目录，内部拼 frozen_parameters.json）也要识别
    assert SCRIPT.checkpoint_grid_size(str(tmp_path)) == 640
    assert SCRIPT.checkpoint_grid_size("") is None
    assert SCRIPT.checkpoint_grid_size(str(tmp_path / "missing.json")) is None

    args = SCRIPT.build_parser().parse_args(
        ["--out", "x", "--grid-size", "128", "--checkpoint", str(ckpt)]
    )
    config, _xy = SCRIPT.build_config(args)
    assert config.start_phase.shape == (640, 640)
    plan = SCRIPT.plan_doc(args, config)
    assert plan["config"]["grid_size"] == 640
    assert plan["config"]["grid_size_cli"] == 128
    # 无 checkpoint 时回退到 --grid-size，保持离线/无适配语义
    args2 = SCRIPT.build_parser().parse_args(["--out", "x", "--grid-size", "8"])
    config2, _ = SCRIPT.build_config(args2)
    assert config2.start_phase.shape == (8, 8)


def test_plan_doc_records_cpu_adapter_device() -> None:
    """生产 adapter 必须显式 CPU，避免与 refine_phase 的 CPU init 混设备。"""
    args = SCRIPT.build_parser().parse_args(["--out", "x"])
    assert SCRIPT.plan_doc(args, SCRIPT.build_config(args)[0])["adapter"]["device"] == "cpu"


_REAL_CKPT = (
    SCRIPT.ROOT / "data/debug/slm_transfer/ab_session_20261124/refit_forward/frozen_parameters.json"
)
_REAL_SOURCE = SCRIPT.ROOT / "data/debug/slm_transfer/ab_session_20261124/capture"


@pytest.mark.skipif(
    not (_REAL_CKPT.is_file() and (_REAL_SOURCE / "manifest.json").is_file()),
    reason="真实 transfer 检查点/标定数据集不在本机",
)
def test_e2e_real_adapter_cpu_advances_multiple_rounds(tmp_path: Path) -> None:
    """真实 adapter(device=cpu) + 真实 forward 能跨过多轮候选，不再 cuda/cpu 失配。

    2026-11-27 第二次上机失败根因：forward 在 cuda、refine_phase 的 init 在 CPU，
    第一轮候选报 "Expected all tensors on the same device, but found cuda:0 and cpu"。
    """
    from ao_shaping.gui.slm.closed_loop_shaping import OnlineTransferAdapter

    ckpt, source = str(_REAL_CKPT), str(_REAL_SOURCE)
    session_dir = tmp_path / "session"

    def real_adapter(grid: object) -> Any:
        return OnlineTransferAdapter(
            checkpoint=ckpt, session_dir=session_dir, span=4,
            source_dataset=source, device="cpu",
        )

    out = tmp_path / "run"
    exit_code = SCRIPT.main(
        [
            "--out", str(out), "--execute", "--wait-until-idle", "0",
            "--max-rounds", "2", "--step-iterations", "2",
            "--centre-x", "500", "--centre-y", "500", "--exposure-ms", "20",
            "--checkpoint", ckpt, "--session-dir", str(session_dir),
            "--source-dataset", source,
        ],
        bench_factory=_SpotBench,
        adapter_factory=real_adapter,
    )
    assert exit_code == 0
    session = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert session["state"].upper() == "STOPPED"
    assert session["rounds_recorded"] == 3  # baseline + 2 轮
    rows = [json.loads(line) for line in (out / "rounds.jsonl").read_text().splitlines()]
    assert all(row["candidate_ms"] is not None for row in rows[1:])


def test_plan_doc_records_refit_every() -> None:
    """--refit-every 必须在 plan-only 可见；默认 2 保持现状语义。"""
    args = SCRIPT.build_parser().parse_args(["--out", "x"])
    plan = SCRIPT.plan_doc(args, SCRIPT.build_config(args)[0])
    assert plan["config"]["refit_every"] == 2

    args1 = SCRIPT.build_parser().parse_args(["--out", "x", "--refit-every", "1"])
    plan1 = SCRIPT.plan_doc(args1, SCRIPT.build_config(args1)[0])
    assert plan1["config"]["refit_every"] == 1


def test_register_rejected_samples_flag_plumbs_to_config() -> None:
    """默认关闭；开启后须同时进 plan 记录与 ClosedLoopConfig。"""
    args = SCRIPT.build_parser().parse_args(["--out", "x"])
    plan = SCRIPT.plan_doc(args, SCRIPT.build_config(args)[0])
    assert plan["config"]["register_rejected_samples"] is False
    assert SCRIPT.build_config(args)[0].register_rejected_samples is False

    args1 = SCRIPT.build_parser().parse_args(
        ["--out", "x", "--register-rejected-samples"]
    )
    plan1 = SCRIPT.plan_doc(args1, SCRIPT.build_config(args1)[0])
    assert plan1["config"]["register_rejected_samples"] is True
    assert SCRIPT.build_config(args1)[0].register_rejected_samples is True


def test_adapter_factory_applies_refit_every(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--refit-every 1 必须传到在线适配器的 policy（首个接受轮次后立即重拟合）。"""
    xy = SCRIPT.analysis_grid((500.0, 500.0), 4)
    shape = (int(xy.shape[0]), int(xy.shape[1]))
    captured: dict[str, Any] = {}

    class RecorderAdapter:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def ready(self) -> bool:
            return True

        def refit(self) -> object:
            return SimpleNamespace(promoted=True, status="promoted")

        def forward(self, grid: object) -> object:
            return _FakeForward(shape)

    # 拦截生产默认 adapter_factory 里的 OnlineTransferAdapter 构造，捕获 policy。
    # 输入校验有独立测试，这里只关心 policy 传递，故跳过（无需真实 checkpoint）。
    monkeypatch.setattr(SCRIPT, "OnlineTransferAdapter", RecorderAdapter)
    monkeypatch.setattr(SCRIPT, "adapter_inputs_error", lambda _args: None)
    exit_code = SCRIPT.main(
        [
            "--out", str(tmp_path / "run"), "--execute", "--wait-until-idle", "0",
            "--max-rounds", "1", "--step-iterations", "2",
            "--grid-size", "8", "--centre-x", "500", "--centre-y", "500",
            "--exposure-ms", "20", "--refit-every", "1",
            # 生产 adapter_factory 会先做输入校验，给非空占位即可（适配器已 fake）
            "--checkpoint", "dummy.json", "--session-dir", str(tmp_path / "session"),
            "--source-dataset", "dummy",
        ],
        bench_factory=_SpotBench,
    )
    assert exit_code == 0
    assert "policy" in captured
    assert captured["policy"].refit_every == 1
