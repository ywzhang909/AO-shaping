"""左栏「高级参数」折叠的离屏契约。

折叠是**展示层**: ① 设备里的相机 / CCD cam_id / SLM 编号、② 里的验收阈值循环
与光路复用收进「高级参数」, 默认收起; 但控件照常构建、照常进 ``form_values()``、
照常参与 ``_revalidate`` 与 ``_set_form_enabled``。这里钉住:

* 默认收起, 运行键照常可用, 首次校验照常跑过;
* 折起来的控件依旧供值、依旧被校验 (红字照样进 ③, 不自动展开);
* 展开 / 收起只翻 ``setVisible``, 不重跑相机枚举 / 设备自检;
* 运行中置灰、程序式 setter 都不因折叠而变。

离屏跑法: ``QT_QPA_PLATFORM=offscreen``; 页面一律 ``auto_detect_ccd=False``。
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QCheckBox

from ao_shaping.gui.slm.transfer_desktop import page as page_mod


@pytest.fixture(scope="module")
def app() -> QApplication:
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture()
def page(app: QApplication, monkeypatch: pytest.MonkeyPatch):
    """离屏页面; 校验换成常量, 免得首次校验拉进 torch。"""
    monkeypatch.setattr(page_mod.params, "validate_form", lambda values: ([], []))
    widget = page_mod.TransferPage(auto_detect_ccd=False)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def test_default_state_collapsed_and_behaviour_unchanged(page) -> None:
    """默认收起: 两组容器都不可见、勾选行都没勾, 运行键照常可用。"""
    for toggle, container in (
        (page.advanced_toggle(), page.advanced_container()),
        (page.device_advanced_toggle(), page.device_advanced_container()),
    ):
        assert isinstance(toggle, QCheckBox)
        assert toggle.text() == "高级参数"
        assert toggle.isChecked() is False
        assert "收起不改变任何行为" in toggle.toolTip()
        assert container.isVisible() is False

    # 折起来的控件确实在容器里 (不是被摘掉了)。
    assert page.advanced_container().isAncestorOf(page._accept)
    assert page.advanced_container().isAncestorOf(page._reuse_latest)
    assert page.device_advanced_container().isAncestorOf(page._camera)
    assert page.device_advanced_container().isAncestorOf(page._slm_number)

    # 首次校验照常跑过, 默认值下运行键可点 (与折叠前一致)。
    assert page._last_errors == []
    assert "参数自检通过" in page._checks.text()
    assert page._run_btn.isEnabled() is True


def test_folded_widgets_still_feed_form_values(page) -> None:
    """折起来的控件照样供值: 改完表单快照立刻反映, 容器不因此展开。"""
    page._accept.setChecked(True)
    page._accept_lower.setValue(0.9)
    page._accept_upper.setValue(0.2)
    page._slm_number.setValue(3)

    values = page.form_values()
    assert values.acceptance_enabled is True
    assert values.acceptance_lower == pytest.approx(0.9)
    assert values.acceptance_upper == pytest.approx(0.2)
    assert values.slm_number == 3

    assert page.advanced_container().isVisible() is False
    assert page.device_advanced_container().isVisible() is False


def test_validation_errors_from_folded_widgets_reach_checks(
    page, monkeypatch: pytest.MonkeyPatch
) -> None:
    """③ 校验照常看到折起来的控件: 红字进 ③, 不自动展开, 运行键被拦。"""
    monkeypatch.setattr(
        page_mod.params,
        "validate_form",
        lambda values: (["验收阈值下界必须在 0 到 1 之间"], []),
    )
    page._revalidate()

    assert "验收阈值" in page._checks.text()
    assert page.advanced_container().isVisible() is False
    assert page._run_btn.isEnabled() is False


def test_toggle_shows_and_hides(page, app: QApplication) -> None:
    """勾选行只翻 ``setVisible``: 两组各展各收, 互不影响。"""
    page.advanced_toggle().setChecked(True)
    app.processEvents()
    assert page.advanced_container().isVisible() is True
    assert page.device_advanced_container().isVisible() is False

    page.device_advanced_toggle().setChecked(True)
    app.processEvents()
    assert page.device_advanced_container().isVisible() is True

    page.advanced_toggle().setChecked(False)
    page.device_advanced_toggle().setChecked(False)
    app.processEvents()
    assert page.advanced_container().isVisible() is False
    assert page.device_advanced_container().isVisible() is False


def test_set_form_enabled_false_greys_folded_widgets(page) -> None:
    """运行中置灰不因折叠而漏掉: 折起来的控件同样 ``isEnabled()`` 为假。"""
    assert page.advanced_container().isVisible() is False

    page._set_form_enabled(False)
    assert page._accept.isEnabled() is False
    assert page._reuse_latest.isEnabled() is False
    assert page._camera.isEnabled() is False
    assert page._cam_id.isEnabled() is False

    page._set_form_enabled(True)
    assert page._accept.isEnabled() is True
    assert page._reuse_latest.isEnabled() is True
    assert page._camera.isEnabled() is True


def test_programmatic_setters_work_while_folded(page) -> None:
    """程序式 setter 与直接改值都不受折叠影响, 也不会把容器翻出来。"""
    page.set_shapes(["rectangle"])
    page.set_period(128.0)
    page.set_camera("sim")
    page._side.setValue(120.0)

    values = page.form_values()
    assert values.shapes == ("rectangle",)
    assert values.period == pytest.approx(128.0)
    assert values.camera == "sim"
    assert values.side == pytest.approx(120.0)

    assert page.advanced_container().isVisible() is False
    assert page.device_advanced_container().isVisible() is False


def test_toggling_does_not_touch_devices(page, app: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    """展开 / 收起纯展示: 不重跑相机枚举, 不进自检态。"""
    calls: list[bool] = []
    monkeypatch.setattr(page, "_start_ccd_detection", lambda: calls.append(True))

    for toggle in (page.advanced_toggle(), page.device_advanced_toggle()):
        toggle.setChecked(True)
        app.processEvents()
        toggle.setChecked(False)
        app.processEvents()

    assert calls == []
    assert page._checking is False
    assert page.ccd_probe_summary() == ""
