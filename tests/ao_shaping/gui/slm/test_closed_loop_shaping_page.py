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
    LoopEvent,
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
    """把控制网格池化到分析网格的替身物理预测模型（与核心测试同一手法）。"""

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


def _select_mode(page, mode: str) -> None:
    """按数据值选运行方式（hold / optimize），而不是猜下拉框顺序。"""
    index = page._mode_combo.findData(mode)
    assert index >= 0, f"未知运行方式：{mode}"
    page._mode_combo.setCurrentIndex(index)


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
    # 这条测试守的是旧的人工确认流程，必须显式切到「逐轮优化」。
    _select_mode(page, "optimize")
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


# ---------------------------------------------------------------------------
# 运行方式：目标保持（默认） / 逐轮优化（旧流程）
# ---------------------------------------------------------------------------


def _start_and_grab_config(app, page, monkeypatch):
    """在确认框点「是」后开跑，返回这一刻的不可变配置。"""
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    page.on_start()
    assert page._thread is not None
    return page._thread.runner.config


def _stop_and_wait(app, page) -> None:
    page.on_stop()
    assert _wait_until(app, lambda: page._thread is None, timeout_s=120.0)


def test_hold_is_the_default_run_mode(app, monkeypatch) -> None:
    """默认就是「目标保持」：配置带 mode=hold 且强制自动接受。"""
    page = _make_page(
        bench_factory=_Bench,
        forward_factory=lambda xy: _Forward(xy.shape[:2]),
    )
    try:
        assert page._run_mode() == "hold"
        assert page._rounds_cap_label.text() == "运行轮数上限"
        assert page._rounds.maximum() == 500
        assert page._rounds.value() == 120
        # 保持模式无人值守：逐轮确认控件不该出现。
        assert page._accept_button.isHidden() is True
        assert page._reject_button.isHidden() is True

        config = _start_and_grab_config(app, page, monkeypatch)
        assert config.mode == "hold"
        assert config.candidate_policy == "auto"
        assert config.hold_target_ccd_px == pytest.approx(5.0)
    finally:
        if page._thread is not None:
            _stop_and_wait(app, page)
        page.close()
        app.processEvents()


def test_switching_to_optimize_restores_the_confirm_flow(app, monkeypatch) -> None:
    """切回逐轮优化：旧标签/旧轮数上限/人工确认控件都回来。"""
    page = _make_page(
        bench_factory=_Bench,
        forward_factory=lambda xy: _Forward(xy.shape[:2]),
    )
    try:
        _select_mode(page, "optimize")
        assert page._rounds_cap_label.text() == "最大轮数"
        assert page._rounds.maximum() == 200
        assert page._rounds.value() == 8
        assert page._accept_button.isHidden() is False
        assert page._reject_button.isHidden() is False

        config = _start_and_grab_config(app, page, monkeypatch)
        assert config.mode == "optimize"
        assert config.candidate_policy == "confirm"
        # 跑起来后人工接受按钮可用（旧行为不变）。
        assert page._accept_button.isEnabled() is True
    finally:
        if page._thread is not None:
            _stop_and_wait(app, page)
        page.close()
        app.processEvents()


def test_hold_position_tolerance_reaches_the_config(app, monkeypatch) -> None:
    """「位置容差（CCD px）」控件必须一路传到 hold_target_ccd_px。"""
    page = _make_page(
        bench_factory=_Bench,
        forward_factory=lambda xy: _Forward(xy.shape[:2]),
    )
    try:
        page._hold_tolerance.setValue(3.5)
        config = _start_and_grab_config(app, page, monkeypatch)
        assert config.mode == "hold"
        assert config.hold_target_ccd_px == pytest.approx(3.5)
    finally:
        if page._thread is not None:
            _stop_and_wait(app, page)
        page.close()
        app.processEvents()


def test_target_widgets_are_frozen_while_a_run_is_in_flight(app, monkeypatch) -> None:
    """目标在本次运行中不变：中心与尺寸控件开跑后必须锁住。"""
    page = _make_page(
        bench_factory=_Bench,
        forward_factory=lambda xy: _Forward(xy.shape[:2]),
    )
    try:
        assert page._centre_x.isEnabled() is True
        _start_and_grab_config(app, page, monkeypatch)
        for widget in (
            page._centre_x, page._centre_y, page._half_width, page._half_height,
            page._span, page._mode_combo,
        ):
            assert widget.isEnabled() is False
        _stop_and_wait(app, page)
        assert page._centre_x.isEnabled() is True
    finally:
        if page._thread is not None:
            _stop_and_wait(app, page)
        page.close()
        app.processEvents()


def test_lock_events_drive_the_phase_line_and_a_visible_fault(app) -> None:
    """收敛中 → 维持中 → 已冻结；冻结原因留在正文红色横幅里。"""
    page = _make_page()
    try:
        page._on_event(
            LoopEvent(
                "lock",
                payload={
                    "round_index": 3,
                    "lock_state": "converging",
                    "action": "propose",
                    "score": 0.42,
                    "best_score": 0.40,
                    "centroid_error_ccd_px": 6.0,
                    "target_centroid_xy": [10.0, 10.0],
                    "centroid_xy": [12.0, 11.0],
                },
            )
        )
        assert "收敛中" in page._phase_label.text()
        assert "3/120" in page._phase_label.text()

        page._on_event(
            LoopEvent(
                "lock",
                payload={
                    "round_index": 4,
                    "lock_state": "maintaining",
                    "action": "accept",
                    "score": 0.41,
                    "best_score": 0.41,
                    "centroid_error_ccd_px": 0.8,
                    "target_centroid_xy": [10.0, 10.0],
                    "centroid_xy": [10.2, 10.1],
                },
            )
        )
        assert "维持中" in page._phase_label.text()
        assert "4/120" in page._phase_label.text()

        # 不认识的 kind 只应被忽略，不能把页面弄崩。
        page._on_event(LoopEvent("mystery", payload={"x": 1}))

        page._on_event(
            LoopEvent(
                "lock",
                payload={
                    "round_index": 5,
                    "lock_state": "frozen",
                    "action": "fault",
                    "score": 0.30,
                    "best_score": 0.41,
                    "centroid_error_ccd_px": 7.5,
                    "target_centroid_xy": [10.0, 10.0],
                    "centroid_xy": [17.5, 10.0],
                },
            )
        )
        assert "已冻结" in page._phase_label.text()
        assert page._fault_banner.isHidden() is False
        assert "7.5" in page._fault_banner.text()
        # 红色横幅来自内联样式：原因必须在正文可见，而不只在弹窗里。
        assert "c0392b" in page._fault_banner.styleSheet()
    finally:
        page.close()
        app.processEvents()


def test_dynamic_page_requests_the_target_hold_page(app) -> None:
    """开环动态页的按钮只发信号；切页由外壳负责。"""
    from ao_shaping.gui.slm.dynamic_shaping_page import DynamicShapingPage

    page = DynamicShapingPage()
    hits: list[str] = []
    page.request_target_hold.connect(lambda: hits.append("hold"))
    try:
        page._target_hold_button.click()
        assert hits == ["hold"]
    finally:
        page.close()
        page._pool.waitForDone(5000)
        app.processEvents()


# --------------------------------------------------------------------------- #
# LockConfig 标定面板：预览、过期失效、显式应用与设备确认
# --------------------------------------------------------------------------- #


def test_lock_calibration_panel_preview_and_apply_gate(app) -> None:
    page = _make_page()
    xy = analysis_grid(_CENTRE, _SPAN)
    try:
        assert page._cal_apply.isChecked() is False
        assert page._lock_config_for_run(xy) is None

        page._cal_dark_mean.setValue(1.0)
        page._cal_dark_sigma.setValue(4.0)
        page._cal_score_sigma.setValue(0.04)
        assert page.on_calibrate() is True
        assert page._cal_recommended is not None
        assert "饱和阈值" in page._cal_preview.text()
        assert page._cal_recommended.min_peak_counts == pytest.approx(81.0)

        # 任何输入变化都让建议值过期，不能把旧阈值悄悄用到新配置上。
        page._cal_peak.setValue(300.0)
        assert page._cal_recommended is None
        assert "尚未计算" in page._cal_preview.text()

        assert page.on_calibrate() is True
        page._cal_apply.setChecked(True)
        cfg = page._lock_config_for_run(xy)
        assert cfg is page._cal_recommended
        assert cfg is not None
        assert cfg.max_centroid_error_px == pytest.approx(5.0 / 2.0)
    finally:
        page.close()


def test_capture_calibration_requires_device_confirmation(app, monkeypatch) -> None:
    calls: list[str] = []
    page = _make_page(bench_factory=lambda: calls.append("bench"))
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.No,
    )
    try:
        page.on_capture_calibration()
        assert calls == []
        assert page._cal_thread is None
        assert page._cal_capture_button.isEnabled()
    finally:
        page.close()


def test_calibration_done_fills_inputs_and_refreshes_preview(app) -> None:
    from ao_shaping.gui.slm.target_lock import CalibrationSample

    page = _make_page()
    try:
        sample = CalibrationSample(
            dark_mean=1.0,
            dark_sigma=4.0,
            peak_counts=220.0,
            total_counts=12345.0,
            saturated_fraction=0.02,
            full_scale=255.0,
            centroid_sigma_px=0.3,
            score_sigma=0.04,
            pixel_count=81,
        )
        page._on_calibration_done(sample)

        assert page._cal_dark_sigma.value() == pytest.approx(4.0)
        assert page._cal_saturated.value() == pytest.approx(2.0)
        assert page._cal_recommended is not None
        assert page._cal_thread is None
        assert page._cal_capture_button.isEnabled()
    finally:
        page.close()
