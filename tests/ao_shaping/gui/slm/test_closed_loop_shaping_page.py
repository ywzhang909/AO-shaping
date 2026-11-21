"""实测闭环页的离屏行为测试（不打开 SLM / CCD / 激光）。

核心逻辑（门禁、接受/回滚、适配）已在 ``test_closed_loop_shaping`` 里逐条钉住；
这里只验页面这层容易出错的三件事：

* 页面构造**不碰设备**，也不 import torch —— 冷启动切页不该拉起任何驱动；
* 没在确认框里点「是」就**不占用设备**；
* 确认后整场闭环跑在 worker 线程里，实测结果渲染到轮次表，结束时把设备关掉；
* transfer 适配默认关闭，开启后三件套只在占用设备前做**只读**校验。
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from ao_shaping.gui.slm.closed_loop_shaping import (
    CapturedFrame,
    LoopState,
    analysis_grid,
    target_field,
)

_SENSOR = 40
_SPAN = 4
_CENTRE = (10.0, 10.0)


@pytest.fixture(scope="module")
def app() -> QApplication:
    """离屏 QApplication（一个进程只能有一个）。"""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


def _target() -> np.ndarray:
    return target_field(analysis_grid(_CENTRE, _SPAN), half_width=2.0, half_height=2.0)


def _good_frame() -> np.ndarray:
    """窗口内采样点上的目标形状（峰值 66，远低于饱和门限）。"""
    frame = np.zeros((_SENSOR, _SENSOR), dtype=np.float64)
    grid = analysis_grid(_CENTRE, _SPAN)
    frame[grid[..., 1].astype(int), grid[..., 0].astype(int)] = 60.0 * _target() + 6.0
    return frame


class _Bench:
    """假光路：每次下发后都返回一张合格的实测帧。"""

    def __init__(self) -> None:
        self.applied: list[np.ndarray] = []
        self.closed = False
        self.opened = 0
        self._frame_id = 0

    def open(self) -> None:
        self.opened += 1

    def describe(self) -> dict:
        return {"name": "fake-bench", "sensor": _SENSOR}

    def apply_phase(self, phase: np.ndarray) -> str:
        self.applied.append(np.array(phase, dtype=np.float64, copy=True))
        return f"phase-{len(self.applied)}"

    def capture(self) -> CapturedFrame:
        self._frame_id += 1
        return CapturedFrame(
            frame=_good_frame(),
            frame_id=self._frame_id,
            exposure_ms=1.0,
            full_scale=255.0,
            monotonic_ns=time.monotonic_ns(),
        )

    def close(self) -> None:
        self.closed = True


class _Forward:
    """把控制网格池化到分析网格的替身正向模型（与核心测试同一手法）。"""

    def __init__(self, shape: tuple[int, int]) -> None:
        self._shape = shape

    def predict(self, phase, exposure_ms=None):
        import torch
        import torch.nn.functional as F

        p = phase if torch.is_tensor(phase) else torch.as_tensor(np.asarray(phase))
        if p.ndim == 2:
            p = p[None]
        pooled = F.adaptive_avg_pool2d(p, self._shape)
        return 0.5 + 0.5 * torch.tanh(pooled)


def _make_page(**kwargs):
    from ao_shaping.gui.slm.closed_loop_shaping_page import ClosedLoopShapingPage

    page = ClosedLoopShapingPage(**kwargs)
    page._span.setValue(_SPAN)
    page._centre_x.setValue(_CENTRE[0])
    page._centre_y.setValue(_CENTRE[1])
    page._half_width.setValue(2.0)
    page._half_height.setValue(2.0)
    page._grid_size.setValue(8)
    page._exposure_ms.setValue(1.0)
    return page


def test_page_construction_touches_no_devices(app) -> None:
    """构造页面只搭控件：工厂一次都不该被调用，也没有起线程。"""
    calls: list[str] = []
    page = _make_page(
        bench_factory=lambda: calls.append("bench"),
        forward_factory=lambda xy: calls.append("forward"),
        adapter_factory=lambda xy: calls.append("adapter"),
    )
    try:
        assert calls == []
        assert page._thread is None
        assert page._table.rowCount() == 0
    finally:
        page.close()
        app.processEvents()


def test_start_without_confirmation_does_not_open_devices(app, monkeypatch) -> None:
    """确认框点「否」= 什么都不做：不开设备、不起线程。"""
    opened: list[str] = []
    page = _make_page(bench_factory=lambda: opened.append("bench"))
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
    )
    try:
        page.on_start()
        assert opened == []
        assert page._thread is None
        assert "取消" in page._log.toPlainText()
    finally:
        page.close()
        app.processEvents()


def _wait_until(app: QApplication, predicate, *, timeout_s: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _drive_confirming(
    app: QApplication, page, *, timeout_s: float = 180.0, stop_after: int | None = None
) -> bool:
    """代替操作员：每当闭环停下来等确认就点「接受候选」，直到线程结束。

    ``confirm`` 策略下每一轮都停一次，只接受一次会让线程永远等第二轮的决定。
    """
    deadline = time.monotonic() + timeout_s
    accepted = 0
    while time.monotonic() < deadline:
        app.processEvents()
        thread = page._thread
        if thread is None:
            return True
        if stop_after is not None and accepted >= stop_after:
            return False
        if thread.runner.state is LoopState.AWAITING_DECISION:
            page.on_accept()
            accepted += 1
        time.sleep(0.01)
    return False


def test_confirmed_run_renders_measured_rounds_and_closes_devices(
    app, monkeypatch
) -> None:
    """人工确认路径：实测轮次进表、人工接受生效、结束时设备被关闭。"""
    bench = _Bench()
    page = _make_page(
        bench_factory=lambda: bench,
        forward_factory=lambda xy: _Forward(xy.shape[:2]),
    )
    page._rounds.setValue(2)
    page._auto_accept.setChecked(False)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    try:
        page.on_start()
        assert page._thread is not None

        # 等到第一次"待确认"，此时必须出现实测分数而不是预测分数。
        assert _wait_until(app, lambda: bool(page._decision_hint.text()))
        assert "实测分数" in page._decision_hint.text()

        assert _drive_confirming(app, page)
        assert page._table.rowCount() >= 2
        decisions = [
            page._table.item(row, 6).text() for row in range(page._table.rowCount())
        ]
        assert decisions[0] == "起点"
        assert "人工接受" in decisions
        # 实测分数列必须是数值，不能是"—"（起点与候选都是实测过的）。
        assert all(
            page._table.item(row, 2).text() != "—"
            for row in range(page._table.rowCount())
        )
        # 页面接线必须让 worker 打开设备一次：否则真实的 HardwareBench 会在
        # 第一次 describe/下发时抛 "open() must be called first"。
        assert bench.opened == 1
        assert bench.closed is True
        assert "已停止" in page._state_label.text()
    finally:
        page.close()
        app.processEvents()


# ---------------------------------------------------------------------------
# transfer 在线适配的接线（路径校验 / 默认关闭 / 真实 adapter）
# ---------------------------------------------------------------------------


def _write_calibration_dataset(root, *, reference: bool = True) -> None:
    """写一个最小合法采集数据集：reference_start + x/y 各一对 ±ramp。

    载荷内容无关紧要（``seed_from_source`` 只复制文件），但文件必须存在，
    校验会逐个 stat。
    """
    import json

    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if reference:
        rows.append(
            {"file": "ref.npz", "label": "reference_start", "spot_xy": [10.0, 10.0]}
        )
    for axis in ("x", "y"):
        for sign in ("1", "-1"):
            rows.append({"file": f"ramp_{axis}_{sign}.npz", "label": f"ramp_{axis}_2_{sign}"})
    for row in rows:
        (root / row["file"]).write_bytes(b"npz-placeholder")
    (root / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")


def _write_checkpoint(root) -> Path:
    import json

    root.mkdir(parents=True, exist_ok=True)
    path = root / "frozen_parameters.json"
    path.write_text(json.dumps({"parameters": {}, "ramp_matrix": []}), encoding="utf-8")
    return path


def test_adaptation_is_off_by_default(app, tmp_path) -> None:
    """适配默认关闭：不勾选时不做任何路径校验，也不构造 adapter。"""
    page = _make_page()
    try:
        assert page._adapt.isChecked() is False
        assert "未启用" in page._adapt_hint.text()
        assert page._adaptation_input_error() is not None  # 空路径必然不合格
    finally:
        page.close()
        app.processEvents()


def test_adaptation_rejects_missing_reference_start(app, tmp_path) -> None:
    """缺 reference_start 时明确报原点无从确定，而不是等到 worker 里才炸。"""
    page = _make_page()
    source = tmp_path / "source"
    _write_calibration_dataset(source, reference=False)
    page._adapt.setChecked(True)
    page._checkpoint_edit.setText(str(_write_checkpoint(tmp_path / "ckpt")))
    page._session_dir_edit.setText(str(tmp_path / "session"))
    page._source_dataset_edit.setText(str(source))
    try:
        problem = page._adaptation_input_error()
        assert problem is not None
        assert "reference_start" in problem
    finally:
        page.close()
        app.processEvents()


def test_adaptation_rejects_missing_ramp_pair(app, tmp_path) -> None:
    """每个轴必须同 period 的 +1/-1 成对；缺负 ramp 要指名到轴。"""
    import json

    page = _make_page()
    source = tmp_path / "source"
    _write_calibration_dataset(source)
    rows = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in rows if row["label"] != "ramp_y_2_-1"]
    (source / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    page._checkpoint_edit.setText(str(_write_checkpoint(tmp_path / "ckpt")))
    page._session_dir_edit.setText(str(tmp_path / "session"))
    page._source_dataset_edit.setText(str(source))
    try:
        problem = page._adaptation_input_error()
        assert problem is not None
        assert "y 轴" in problem
    finally:
        page.close()
        app.processEvents()


def test_adaptation_rejects_session_dir_equal_to_source_dataset(app, tmp_path) -> None:
    """会话目录不能等于标定源：新样本会混进标定数据。"""
    page = _make_page()
    source = tmp_path / "source"
    _write_calibration_dataset(source)
    page._checkpoint_edit.setText(str(_write_checkpoint(tmp_path / "ckpt")))
    page._session_dir_edit.setText(str(source))
    page._source_dataset_edit.setText(str(source))
    try:
        problem = page._adaptation_input_error()
        assert problem is not None
        assert "相同" in problem
    finally:
        page.close()
        app.processEvents()


def test_adaptation_accepts_a_replayable_calibrated_dataset(app, tmp_path) -> None:
    """齐全的检查点 + 标定数据集 + 独立会话目录应通过只读校验。"""
    page = _make_page()
    source = tmp_path / "source"
    _write_calibration_dataset(source)
    page._checkpoint_edit.setText(str(_write_checkpoint(tmp_path / "ckpt")))
    page._session_dir_edit.setText(str(tmp_path / "session"))
    page._source_dataset_edit.setText(str(source))
    try:
        assert page._adaptation_input_error() is None
        # 校验必须是只读的：通过校验不等于已经建目录或复制标定行。
        assert not (tmp_path / "session").exists()
    finally:
        page.close()
        app.processEvents()


def test_start_refuses_before_devices_when_adaptation_paths_are_bad(
    app, monkeypatch
) -> None:
    """勾了适配但路径不合格：连确认框都不该弹，更不该打开设备。"""
    opened: list[str] = []
    warnings: list[str] = []
    questions: list[str] = []
    page = _make_page(bench_factory=lambda: opened.append("bench"))
    page._adapt.setChecked(True)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: questions.append("asked")
    )
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: warnings.append(a[2] if len(a) > 2 else "")
    )
    try:
        page.on_start()
        assert opened == []
        assert questions == []
        assert page._thread is None
        assert warnings and "检查点" in warnings[0]
    finally:
        page.close()
        app.processEvents()


def test_default_adapter_builds_from_paths_without_torch_or_drivers(
    app, tmp_path
) -> None:
    """没有注入工厂时，页面用三件套构造真实 adapter 并完成开场标定。

    构造只做 JSON/文件操作；torch 与硬件驱动都不该因此被导入。
    """
    import sys

    from ao_shaping.gui.slm.closed_loop_shaping import OnlineTransferAdapter

    page = _make_page()
    source = tmp_path / "source"
    _write_calibration_dataset(source)
    page._checkpoint_edit.setText(str(_write_checkpoint(tmp_path / "ckpt")))
    page._session_dir_edit.setText(str(tmp_path / "session"))
    page._source_dataset_edit.setText(str(source))
    torch_was_loaded = "torch" in sys.modules
    drivers_was_loaded = "ao_shaping.drivers" in sys.modules
    try:
        adapter = page._make_adapter(analysis_grid(_CENTRE, _SPAN))
        assert isinstance(adapter, OnlineTransferAdapter)
        assert adapter.seeded is True
        assert adapter.seed_error is None
        assert adapter.span == _SPAN
        # 只比较构造前后：同进程里别的页面可能早已导入这些模块。
        assert ("torch" in sys.modules) is torch_was_loaded
        assert ("ao_shaping.drivers" in sys.modules) is drivers_was_loaded
    finally:
        page.close()
        app.processEvents()


def test_stop_request_ends_the_loop_and_reports_reason(app, monkeypatch) -> None:
    """停止按钮走 runner.request_stop：结束后页面回到空闲控件状态。"""
    bench = _Bench()
    page = _make_page(
        bench_factory=lambda: bench,
        forward_factory=lambda xy: _Forward(xy.shape[:2]),
    )
    page._rounds.setValue(50)
    page._auto_accept.setChecked(True)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    try:
        page.on_start()
        assert _wait_until(app, lambda: page._table.rowCount() >= 1)
        page.on_stop()
        assert _wait_until(app, lambda: page._thread is None, timeout_s=120.0)
        assert page._start_button.isEnabled() is True
        assert page._stop_button.isEnabled() is False
        assert bench.closed is True
    finally:
        page.close()
        app.processEvents()
