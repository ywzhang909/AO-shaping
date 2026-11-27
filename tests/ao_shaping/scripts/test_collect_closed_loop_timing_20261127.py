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


def test_adapter_inputs_validated_before_devices(tmp_path: Path) -> None:
    args = SCRIPT.build_parser().parse_args(
        [
            "--out",
            str(tmp_path / "run"),
            "--execute",
            "--checkpoint",
            str(tmp_path / "missing.json"),
            "--source-dataset",
            str(tmp_path / "missing.npz"),
        ]
    )
    assert SCRIPT.adapter_inputs_error(args) is not None


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
