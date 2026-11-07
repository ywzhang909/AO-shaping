"""迁移学习闭环页「光路复用」的离屏交互契约。

光路复用 = 用一次廉价探针量当前光路条件并与最近一份 latest 模型比对；一致则跳过
采集与拟合，直接把那次 CCD 对照重做一遍。页面端只做三件事:

* 一个开关「光路复用」, 默认关（= 当前全流程重新采集）, 值原样进表单快照, 进而随
  :func:`build_session` 作为 native 字段进会话;
* 运行期间随其余必设参数一起锁住;
* 结算时若会话复用过, 状态行补一行「光路复用：一致 / 已变化」。

这里不该也不需要碰任何硬件: 全部离屏、参数校验换成常量、会话 tasks 用桩。
"""

from __future__ import annotations

import json
import os
import types
from datetime import datetime

# 必须在任何 QApplication 之前设置: CI / 无显示环境走离屏后端。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from ao_shaping.gui.slm.transfer_desktop import page as page_mod
from ao_shaping.gui.slm.transfer_desktop import params


class _StubSignal:
    def connect(self, *args, **kwargs):
        pass


class _StubTask(page_mod.QRunnable):
    """接住 ``_start_run`` 建的任务, 不真正开会话 (不碰硬件)。"""

    def __init__(self, values, out, should_stop=None):
        super().__init__()
        self.values = values
        self.out = out
        self.should_stop = should_stop
        self.signals = types.SimpleNamespace(
            stage=_StubSignal(),
            frame=_StubSignal(),
            finished=_StubSignal(),
            failed=_StubSignal(),
        )

    def run(self):
        pass


@pytest.fixture(scope="module")
def app() -> QApplication:
    """离屏 QApplication (一个进程只能有一个)。"""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture()
def page(app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path):
    """离屏页面; 校验换成常量, tasks 换成桩, 免得拉进 torch / 硬件。"""
    monkeypatch.setattr(page_mod.params, "validate_form", lambda values: ([], []))
    monkeypatch.setattr(page_mod.params, "plan_session_dir", lambda: tmp_path / "s")
    monkeypatch.setattr(page_mod, "_SessionTask", _StubTask)
    widget = page_mod.TransferPage()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


# --- 构造回归 --------------------------------------------------------------


def test_page_constructs_without_attribute_error(app: QApplication, monkeypatch, tmp_path) -> None:
    """回归: 控件创建途中的 setValue 不得触发 _revalidate 读到未建好的控件。"""
    monkeypatch.setattr(page_mod.params, "validate_form", lambda values: ([], []))
    monkeypatch.setattr(page_mod.params, "plan_session_dir", lambda: tmp_path / "s")
    monkeypatch.setattr(page_mod, "_SessionTask", _StubTask)
    widget = page_mod.TransferPage()
    try:
        assert widget.form_values().acceptance_upper == params.ACCEPTANCE_UPPER_DEFAULT
        assert widget.form_values().reuse_latest is False
    finally:
        widget.close()
        app.processEvents()


# --- 光路复用控件 ---------------------------------------------------------


def test_reuse_control_exists_and_defaults_off(page) -> None:
    """开关存在、默认关 (= 当前全流程重新采集)。"""
    assert hasattr(page, "_reuse_latest")
    assert page._reuse_latest.isChecked() is False


def test_form_values_reflect_the_reuse_toggle(page) -> None:
    """控件的勾选状态原样进表单快照 (= params 层建会话要读的字段)。"""
    page._reuse_latest.setChecked(True)
    assert page.form_values().reuse_latest is True

    page._reuse_latest.setChecked(False)
    assert page.form_values().reuse_latest is False


def test_reuse_defaults_match_params_layer(page) -> None:
    """页面控件默认值就是 params 层的默认值 (单一口径, 不另立一套)。"""
    values = params.TransferFormValues()
    assert page._reuse_latest.isChecked() == bool(values.reuse_latest)


def test_reuse_control_is_in_the_runtime_lock_set(page, monkeypatch) -> None:
    """运行期间复选框随必设参数一起锁住 (本次用的是启动时的快照)。"""
    assert page._reuse_latest.isEnabled() is True
    monkeypatch.setattr(page._pool, "start", lambda task: None)
    monkeypatch.setattr(page._poll, "start", lambda *args: None)
    page._start_run()
    assert page._reuse_latest.isEnabled() is False
    page._on_failed("cleanup")
    assert page._reuse_latest.isEnabled() is True


# --- 光路复用结论的显示 ---------------------------------------------------


def _write_reused_session(tmp_path, same_path):
    """造一个写有 ``optical_probe.json`` 的会话, 供显示测试。"""
    session_dir = tmp_path / "s"
    session_dir.mkdir(exist_ok=True)
    (session_dir / "optical_probe.json").write_text(
        json.dumps(
            {
                "serial": "sn-1",
                "latest": str(session_dir / "latest"),
                "same_path": bool(same_path),
                "verdict": "same" if same_path else "different",
            }
        ),
        encoding="utf-8",
    )
    return session_dir


def test_reuse_verdict_line_skips_when_not_run(tmp_path) -> None:
    session_dir = tmp_path / "s"
    session_dir.mkdir()
    # 没写 optical_probe.json = 本场没走光路复用 → 空串。
    assert page_mod.reuse_latest_verdict_line(session_dir) == ""


def test_reuse_verdict_line_shows_skipped_reason(tmp_path) -> None:
    """复用被跳过 (未开启/探针失败/无 latest) → 说明「为什么没走复用」。"""
    session_dir = tmp_path / "s"
    session_dir.mkdir()
    (session_dir / "optical_probe.json").write_text(
        json.dumps({"skipped": True, "reason": "光路复用未开启"}),
        encoding="utf-8",
    )
    line = page_mod.reuse_latest_verdict_line(session_dir)
    assert line == "光路复用：跳过（光路复用未开启） → 走正常全流程"


def test_reuse_verdict_line_shows_skipped_without_reason(tmp_path) -> None:
    """复用被跳过但缺 reason → 无原因的兜底文案, 绝不抛。"""
    session_dir = tmp_path / "s"
    session_dir.mkdir()
    (session_dir / "optical_probe.json").write_text(
        json.dumps({"skipped": True}), encoding="utf-8",
    )
    line = page_mod.reuse_latest_verdict_line(session_dir)
    assert line == "光路复用：跳过 → 走正常全流程"


def test_reuse_verdict_line_shows_same_when_reused(tmp_path) -> None:
    session_dir = _write_reused_session(tmp_path, same_path=True)
    line = page_mod.reuse_latest_verdict_line(session_dir)
    assert "光路复用" in line
    assert "一致" in line


def test_reuse_verdict_line_shows_different_when_changed(tmp_path) -> None:
    session_dir = _write_reused_session(tmp_path, same_path=False)
    line = page_mod.reuse_latest_verdict_line(session_dir)
    assert "光路复用" in line
    assert "未采用" in line
    assert "走正常全流程" in line


def test_reuse_verdict_line_lists_detailed_reasons(tmp_path) -> None:
    """未通过时列出每一条不通过的原因 (零点 / 曝光 / 格式 / 位深)。"""
    session_dir = tmp_path / "s"
    session_dir.mkdir()
    cur = {
        "zero_xy": [100.0, 200.0],
        "exposure_range_ms": [0.02, 5.0],
        "camera_format": "MONO8",
        "bit_depth": 8,
        "slm_serial": "sn-1",
    }
    ref = {
        "zero_xy": [100.0, 210.0],  # 零点漂移 10 px
        "exposure_range_ms": [0.02, 8.0],  # 曝光上界超容忍
        "camera_format": "MONO12",  # 格式不同
        "bit_depth": 12,  # 位深不同
        "slm_serial": "sn-1",
    }
    (session_dir / "optical_probe.json").write_text(
        json.dumps(
            {"serial": "sn-1", "latest": "x", "same_path": False,
             "current": cur, "reference": ref, "verdict": "drift"}
        ),
        encoding="utf-8",
    )
    line = page_mod.reuse_latest_verdict_line(session_dir)
    assert "光路复用" in line
    assert "未采用" in line
    assert "零点漂移" in line
    assert "曝光上界" in line
    assert "取图格式" in line
    assert "位深" in line


def test_reuse_verdict_line_never_raises_on_partial_artifacts(tmp_path) -> None:
    session_dir = tmp_path / "s"
    session_dir.mkdir()
    # 文件在但是坏 JSON → 不报错, 返回空串 (没值可显示)。
    (session_dir / "optical_probe.json").write_text("{not json", encoding="utf-8")
    assert page_mod.reuse_latest_verdict_line(session_dir) == ""


def test_on_finished_appends_reuse_verdict(page, tmp_path, monkeypatch) -> None:
    """会话写完 ``optical_probe.json`` 结算时, 状态行补一行光路复用结论。"""
    session_dir = tmp_path / "s"
    session_dir.mkdir()
    (session_dir / "optical_probe.json").write_text(
        json.dumps({"serial": "sn-1", "latest": "x", "same_path": True}),
        encoding="utf-8",
    )
    # 直接调 _on_finished: 它内部先 _settle 复位状态, 再补一行光路复用结论。
    page._on_finished(str(session_dir))
    assert "光路复用" in page._status.text()
    page._on_failed("cleanup")


# --- 所用模型权重的保存日期 -----------------------------------------------


def _write_weight(path, stamp: float):
    """写一个假权重文件并把 mtime 钉到 ``stamp``。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    os.utime(path, (stamp, stamp))
    return path


def test_model_weight_line_skips_without_artifacts(tmp_path) -> None:
    session_dir = tmp_path / "s"
    session_dir.mkdir()
    assert page_mod.model_weight_saved_line(session_dir) == ""
    assert page_mod.model_weight_saved_line(tmp_path / "nope") == ""


def test_model_weight_line_uses_newest_fit_weight(tmp_path) -> None:
    """本场拟合产物: 取 ``model*`` / ``fit*`` 里 mtime 最新的冻结参数。"""
    session_dir = tmp_path / "s"
    _write_weight(session_dir / "fit" / "frozen_parameters.json", 1_700_000_000)
    newest = _write_weight(
        session_dir / "model" / "frozen_parameters.json", 1_700_003_600
    )
    line = page_mod.model_weight_saved_line(session_dir)
    assert "model/frozen_parameters.json" in line
    assert "保存于" in line
    stamp = datetime.fromtimestamp(newest.stat().st_mtime).strftime("%Y-%m-%d")
    assert stamp in line


def test_model_weight_line_prefers_reused_latest(tmp_path) -> None:
    """复用光路 = 本场没拟合新模型: 显示 models_root 里那份 latest 的 mtime。"""
    session_dir = tmp_path / "s"
    latest = _write_weight(
        tmp_path / "models" / "sn-1" / "latest" / "frozen_parameters.json",
        1_700_010_000,
    ).parent
    _write_weight(session_dir / "fit" / "frozen_parameters.json", 1_700_000_000)
    (session_dir / "optical_probe.json").write_text(
        json.dumps({"serial": "sn-1", "latest": str(latest), "same_path": True}),
        encoding="utf-8",
    )
    line = page_mod.model_weight_saved_line(session_dir)
    assert "sn-1/latest/frozen_parameters.json" in line
    stamp = datetime.fromtimestamp(1_700_010_000).strftime("%Y-%m-%d")
    assert stamp in line


def test_model_weight_line_hits_deep_subdirectory(tmp_path) -> None:
    """产物在更深的子目录 (如 ``model*/run/``) 也能命中, 不再空串。"""
    session_dir = tmp_path / "s"
    _write_weight(
        session_dir / "model" / "run1" / "frozen_parameters.json", 1_700_000_000
    )
    line = page_mod.model_weight_saved_line(session_dir)
    assert "model/run1/frozen_parameters.json" in line
    assert "保存于" in line


def test_model_weight_line_reports_corrupt_weight(tmp_path) -> None:
    """权重文件在但不可解析 → 一行「模型加载失败: …损坏或不可解析」。"""
    session_dir = tmp_path / "s"
    weight = session_dir / "model" / "frozen_parameters.json"
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_text("{not json", encoding="utf-8")
    line = page_mod.model_weight_saved_line(session_dir)
    assert line.startswith("模型加载失败：")
    assert "损坏或不可解析" in line
    assert "model/frozen_parameters.json" in line


def test_on_finished_appends_weight_saved_date(page, tmp_path) -> None:
    """会话完成状态行补一行「模型权重 · 保存于 …」。"""
    session_dir = tmp_path / "s"
    _write_weight(session_dir / "fit" / "frozen_parameters.json", 1_700_000_000)
    page._on_finished(str(session_dir))
    assert "模型权重" in page._status.text()
    assert "保存于" in page._status.text()
    page._on_failed("cleanup")