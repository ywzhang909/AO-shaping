"""迁移学习闭环页 ⑦ 计划预览的离屏契约（矩阵 / 合计 / 段级成本）。

要求来自 issue #242 的第 1、3 项: 开跑前先在右栏把「一场 campaign 有多大」摆出来,
再拿历史目录对一下实测偏差。这里钉住的是**显示层**的约定:

* 4 形状 × 4 阶段的矩阵全在, ``▸`` 跟着 ② 的目标单选走 (换形状只是重上色);
* 合计行就是数据层的原文, 页面不另算一套;
* 成本表 16 行, 没实测的格子写「—」而不是 0;
* 目录空 / 不存在 / 不是目录 / 读盘炸了, 都只退成计划列, 不弹错误框、不抛异常;
* 读取实测只在点按钮 / 回车 / 程序设值时发生, 每个按键都扫目录没有意义;
* 读成本不碰设备 (不构造任何设备任务); 正文可收起, 收起再展开内容不丢。

离屏跑法: ``QT_QPA_PLATFORM=offscreen``; 页面一律 ``auto_detect_ccd=False``。
"""

from __future__ import annotations

import os

# 必须在任何 QApplication 之前设置: CI / 无显示环境走离屏后端。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ao_shaping.gui.slm.transfer_desktop import campaign_plan
from ao_shaping.gui.slm.transfer_desktop import page as page_mod
from ao_shaping.gui.slm.transfer_desktop import params
from ao_shaping.gui.slm.transfer_desktop.campaign_plan import (
    PLAN_SHAPES,
    campaign_summary,
)
from ao_shaping.gui.slm.transfer_progress import MISSING
from ao_shaping.tools.slm.transfer.cost_table import record_stage_cost

_COLUMNS = ("形状", "阶段", "计划块数", "计划秒", "实测秒", "实测块数", "偏差")


@pytest.fixture(scope="module")
def app() -> QApplication:
    """离屏 QApplication (一个进程只能有一个)。"""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture()
def page(app: QApplication, monkeypatch: pytest.MonkeyPatch):
    """离屏页面; 校验换成常量, 免得每次实例化都 import torch 跑一遍 preflight。"""
    monkeypatch.setattr(page_mod.params, "validate_form", lambda values: ([], []))
    widget = page_mod.TransferPage(auto_detect_ccd=False)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def _matrix_lines(page) -> list[str]:
    return page.campaign_matrix_text().splitlines()


def _line_for(page, shape: str) -> str:
    label = params.SHAPE_LABELS[shape]
    return next(line for line in _matrix_lines(page) if label in line)


# --- 矩阵 -------------------------------------------------------------------


def test_the_matrix_shows_all_four_shapes_by_four_stages(page) -> None:
    lines = _matrix_lines(page)

    assert len(lines) == 1 + len(PLAN_SHAPES)
    assert lines[0].startswith("形状")
    for shape in PLAN_SHAPES:
        assert params.SHAPE_LABELS[shape] in _line_for(page, shape)
    # 每行四格: 行标题 + 4 个阶段格。
    assert all(len(line.split(" | ")) == 5 for line in lines)


def test_the_arrow_follows_the_shape_radio_both_ways(page) -> None:
    assert _line_for(page, "square").startswith("▸ ")

    page.set_shapes(("star",))
    assert _line_for(page, "star").startswith("▸ ")
    assert not _line_for(page, "square").startswith("▸ ")

    page.set_shapes(("square",))
    assert _line_for(page, "square").startswith("▸ ")
    assert not _line_for(page, "star").startswith("▸ ")


def test_the_totals_line_is_the_data_layer_text(page) -> None:
    assert page.plan_summary_text() == campaign_summary().text


# --- 段级成本表 -------------------------------------------------------------


def test_the_cost_table_starts_plan_only_with_dashes_for_measured(page) -> None:
    table = page.cost_table()
    planned = campaign_plan.cost_rows()

    assert table.columnCount() == len(_COLUMNS)
    assert tuple(
        table.horizontalHeaderItem(column).text()
        for column in range(table.columnCount())
    ) == _COLUMNS
    assert table.rowCount() == 16
    for row, cost in enumerate(planned):
        assert table.item(row, 0).text() == cost.shape_label
        assert table.item(row, 1).text() == cost.stage_label
        assert table.item(row, 2).text() == str(cost.planned_blocks)
        assert table.item(row, 3).text() == f"{cost.expected_seconds:.0f} s"
        # 没有历史时实测三列都是破折号, 绝不是「0 s / 0 块 / 0%」。
        assert table.item(row, 4).text() == MISSING
        assert table.item(row, 5).text() == MISSING
        assert table.item(row, 6).text() == MISSING
    assert page.campaign_dir() is None
    assert "首次运行" in page.cost_caption_text()


def test_a_recorded_campaign_fills_the_measured_columns(
    page, tmp_path: Path
) -> None:
    root = tmp_path / "campaign"
    record_stage_cost(
        root / "square" / "transport_cv",
        stage="transport_cv",
        shape="square",
        seconds=120.0,
        blocks=4,
    )

    page.set_campaign_dir(root)

    table = page.cost_table()
    assert page.campaign_dir() == root
    assert table.rowCount() == 16
    assert table.item(0, 0).text() == params.SHAPE_LABELS["square"]
    assert table.item(0, 4).text() == "120.0 s"
    assert table.item(0, 5).text() == "4"
    assert table.item(0, 6).text() == campaign_plan.cost_rows(root)[0].delta_text
    assert table.item(0, 6).text() != MISSING
    assert "历史基准" in page.cost_caption_text()
    # 其余 15 段仍没有实测, 保持破折号。
    assert table.item(1, 4).text() == MISSING


def test_an_unreadable_directory_degrades_to_plan_only(page, tmp_path: Path) -> None:
    missing = tmp_path / "no-such-campaign"

    page.set_campaign_dir(missing)

    table = page.cost_table()
    assert page.campaign_dir() is None
    assert "目录不可读" in page._cost_caption.text()
    assert "首次运行" in page.cost_caption_text()
    assert table.rowCount() == 16
    assert table.item(0, 4).text() == MISSING
    assert table.item(0, 6).text() == MISSING


def test_a_corrupt_data_layer_never_raises(page, tmp_path: Path, monkeypatch) -> None:
    real_rows = campaign_plan.cost_rows

    def boom_rows(directory=None):
        if directory is not None:
            raise OSError("坏目录")
        return real_rows(directory)

    monkeypatch.setattr(campaign_plan, "cost_rows", boom_rows)
    monkeypatch.setattr(
        campaign_plan,
        "cost_caption",
        lambda directory=None: (_ for _ in ()).throw(OSError("坏数据")),
    )

    page.set_campaign_dir(tmp_path)

    # 兜底仍是 16 行计划值 (表格不整块消失), 只在口径行写一句读取失败。
    assert page.campaign_dir() is None
    assert "读取失败" in page._cost_caption.text()
    assert page.cost_table().rowCount() == 16
    assert page.cost_table().item(0, 4).text() == MISSING


def test_a_corrupt_matrix_degrades_to_a_visible_failure_line(page, monkeypatch) -> None:
    monkeypatch.setattr(
        campaign_plan,
        "matrix_cells",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("坏矩阵")),
    )

    # 换形状会走 _sync_shape_widgets → 重画矩阵, 正好是失败路径。
    page.set_shapes(("star",))

    assert page.campaign_matrix_text() == "读取失败"
    # 计划预览在「计划预览」页签里; 切过去才可见。
    page._right_tabs.setCurrentIndex(1)
    assert page._plan_matrix_error.isVisible()


# --- 交互 / 不碰设备 --------------------------------------------------------


def test_the_body_collapses_and_expands_without_losing_content(page) -> None:
    before = page.campaign_matrix_text()
    page._right_tabs.setCurrentIndex(1)

    page._plan_toggle.setChecked(False)
    assert not page._plan_body.isVisible()
    assert page.campaign_matrix_text() == before

    page._plan_toggle.setChecked(True)
    assert page._plan_body.isVisible()
    assert page.campaign_matrix_text() == before


def test_reading_costs_never_builds_a_device_task(
    page, tmp_path: Path, monkeypatch
) -> None:
    class _Boom:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("读取实测不该构造设备任务")

    for name in ("CcdDetectTask", "HardwareCheckTask", "_ZeroOrderTask", "_SessionTask"):
        monkeypatch.setattr(page_mod, name, _Boom)

    page.set_campaign_dir(tmp_path)
    QTest.mouseClick(page._cost_btn, Qt.MouseButton.LeftButton)
    QTest.keyClick(page._cost_dir, Qt.Key.Key_Return)

    assert page.cost_table().rowCount() == 16


def test_typing_in_the_directory_box_does_not_read_the_disk(page, monkeypatch) -> None:
    calls: list[object] = []
    real_rows = campaign_plan.cost_rows

    def counting_rows(directory=None):
        calls.append(directory)
        return real_rows(directory)

    monkeypatch.setattr(campaign_plan, "cost_rows", counting_rows)
    page._cost_dir.clear()

    QTest.keyClicks(page._cost_dir, "campaign")
    assert calls == []

    QTest.mouseClick(page._cost_btn, Qt.MouseButton.LeftButton)
    assert len(calls) == 1
