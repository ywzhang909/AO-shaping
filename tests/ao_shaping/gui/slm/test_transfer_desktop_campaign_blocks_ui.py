"""⑥ 「campaign 逐块」页的离屏契约。

这一页只读 ⑦ 填的 campaign 目录: 段一行、块作子行, 缺值印「—」, 不做好坏判断。
这里钉住:

* 没有目录 / 不存在 / 是文件 / 没有段级产物 → 一行降级说明, 树为空, 不抛异常;
* 有段有块 → 页头计数与 ``.totals`` 一致, 块列与 ``BlockRow`` 字段一致;
* 饱和像素 > 0 的块 **不** 比 0 的块多任何颜色 (GUI 不设阈值, G3 才管);
* 验收 ✓ / ✗ / — 照搬数据层判定;
* 「读取实测」与「刷新」都会重读, 换主题只重上色不重读。

离屏跑法: ``QT_QPA_PLATFORM=offscreen``; 页面一律 ``auto_detect_ccd=False``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from ao_shaping.gui.slm.transfer_desktop import page as page_mod
from ao_shaping.tools.slm.transfer import campaign_progress as cp

MISSING = cp.MISSING


def _row(label: str, cv: float, *, core_peak: float, peak: float | None, saturated: int) -> dict:
    row = {
        "label": label,
        "file": f"{label}.npz",
        "metrics": {"cv": cv, "mean": 1.0, "ee_roi": 0.4, "target_peak": 1.0},
        "core_peak": core_peak,
        "saturation_pixels": saturated,
        "exposure_ms": 135.196,
        "bit_depth": 12,
        "full_scale": 4095.0,
    }
    if peak is not None:
        row["peak"] = peak
    return row


def _write_campaign(root: Path, *, blocks: int = 2, saturated_first: int = 0) -> Path:
    """合成一个 1 段 × ``blocks`` 块的 campaign 目录 (沿用 test_campaign_progress 的布局)。

    journal 写 ``square/seed`` 为 complete, 其余阶段不写 → 数据层只看见这一段。
    """
    # 阶段名必须是 STAGE_RECIPES 里真有的 (mono12_cv), 否则数据层不认。
    stage_dir = root / "shape" / "square" / "mono12_cv"
    stage_dir.mkdir(parents=True, exist_ok=True)
    rows = [_row("square_seed", 0.80, core_peak=102.0, peak=380.0, saturated=0)]
    for index in range(blocks):
        saturated = saturated_first if index == 0 else 0
        rows.append(_row(f"square_{index}_A1", 0.90 - 0.01 * index, core_peak=200.0 + index,
                         peak=900.0, saturated=saturated))
    rows.append(_row("square_final", 0.70, core_peak=230.0, peak=3600.0, saturated=0))
    (stage_dir / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    summary = [
        {"shape": "square", "block": index, "accepted": index == 0,
         "acceptance": "sequential_ccd", "accept_metric": "cv",
         "baseline": 0.9, "candidate": 0.85}
        for index in range(blocks)
    ]
    (stage_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    # journal.json 是裸列表 (与 test_campaign_progress._write_journal 一致)。
    (root / "journal.json").write_text(
        json.dumps([{"stage": "square/mono12_cv", "status": "complete"}]),
        encoding="utf-8",
    )
    return root


@pytest.fixture(scope="module")
def app() -> QApplication:
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture()
def make_page(app: QApplication, monkeypatch: pytest.MonkeyPatch):
    created: list[page_mod.TransferPage] = []
    monkeypatch.setattr(page_mod.params, "validate_form", lambda values: ([], []))

    def _make() -> page_mod.TransferPage:
        widget = page_mod.TransferPage(auto_detect_ccd=False)
        widget.show()
        app.processEvents()
        created.append(widget)
        return widget

    yield _make
    for widget in created:
        widget.close()
    app.processEvents()


def _tree_top_rows(page: page_mod.TransferPage) -> list:
    tree = page.campaign_tree()
    return [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]


# --- 降级 -------------------------------------------------------------------


def test_no_dir_shows_the_hint_and_an_empty_tree(make_page) -> None:
    page = make_page()

    assert page.campaign_tree().topLevelItemCount() == 0
    assert page.campaign_status_text() == "在 ⑦ 填整轮目录后显示逐块读数"
    assert page.campaign_header_text() == ""


@pytest.mark.parametrize("kind", ["missing", "file", "empty"])
def test_unusable_dirs_degrade_to_one_line_without_raising(make_page, tmp_path, kind) -> None:
    if kind == "missing":
        target = tmp_path / "nope"
    elif kind == "file":
        target = tmp_path / "a_file.txt"
        target.write_text("x", encoding="utf-8")
    else:
        target = tmp_path / "empty"
        target.mkdir()
    page = make_page()

    page.set_campaign_dir(target)

    assert page.campaign_tree().topLevelItemCount() == 0
    assert page.campaign_header_text() == ""
    assert page.campaign_status_text()  # 一行说明, 非空


# --- 有数据 -----------------------------------------------------------------


def test_header_counts_match_the_data_layer_totals(make_page, tmp_path) -> None:
    campaign = _write_campaign(tmp_path / "camp", blocks=2)
    page = make_page()

    page.set_campaign_dir(campaign)

    totals = cp.collect_campaign_progress(campaign).totals
    header = page.campaign_header_text()
    assert f"{totals['segments_total']} 段" in header
    assert f"已完成 {totals['segments_done']} 段" in header
    assert f"{totals['blocks_total']} 块" in header
    assert f"已到 {totals['blocks_done']} 块" in header
    assert page.campaign_status_text() == ""


def test_one_stage_row_with_its_block_children(make_page, tmp_path) -> None:
    campaign = _write_campaign(tmp_path / "camp", blocks=2)
    page = make_page()

    page.set_campaign_dir(campaign)

    tops = _tree_top_rows(page)
    assert len(tops) == 1
    top = tops[0]
    assert top.text(1) == "完成"
    assert top.childCount() == 2
    first, second = top.child(0), top.child(1)
    assert first.text(0) == "0"
    assert second.text(0) == "1"
    stage = cp.collect_campaign_progress(campaign).stages[0]
    block = stage.blocks[0]
    assert first.text(1) == f"{block.cv:.4f}"
    assert first.text(2) == str(block.saturated_px)
    assert first.text(3) == f"{block.core_peak:.0f}"
    assert first.text(4) == f"{block.exposure_ms:.1f}"


def test_missing_block_values_print_a_dash_never_zero(make_page, tmp_path) -> None:
    campaign = _write_campaign(tmp_path / "camp", blocks=1)
    # 删掉 summary.json → 验收判定变 MISSING; 块号仍在。
    (campaign / "shape" / "square" / "mono12_cv" / "summary.json").unlink()
    page = make_page()

    page.set_campaign_dir(campaign)

    child = _tree_top_rows(page)[0].child(0)
    assert child.text(5) == MISSING


def test_saturated_block_is_printed_but_not_coloured(make_page, tmp_path) -> None:
    # 块 0 饱和 (7 px) 且 accepted=True; 块 1 不饱和 (0 px) 且 accepted=False。
    # 验收列本来就按判定上色 (✓ 绿 / ✗ 红), 所以两块的验收列颜色不同是正常的 ——
    # 这里只比「饱和 → 颜色」: 把验收列排除在外, 其余 6 列必须一致。
    campaign = _write_campaign(tmp_path / "camp", blocks=2, saturated_first=7)
    page = make_page()

    page.set_campaign_dir(campaign)

    top = _tree_top_rows(page)[0]
    saturated, clean = top.child(0), top.child(1)
    assert saturated.text(2) == "7"
    assert clean.text(2) == "0"
    for column in (0, 1, 2, 3, 4, 6):
        assert saturated.foreground(column).color() == clean.foreground(column).color()


def test_verdict_colour_follows_accepted_not_saturation(make_page, tmp_path) -> None:
    # 两块饱和数不同、判定相同 → 验收列颜色相同: 颜色只跟判定走。
    campaign = _write_campaign(tmp_path / "camp", blocks=2, saturated_first=7)
    page = make_page()

    page.set_campaign_dir(campaign)

    top = _tree_top_rows(page)[0]
    # 块 0 accepted=True(饱和 7), 块 1 accepted=False(饱和 0); 颜色分别是 ok / error。
    assert top.child(0).foreground(5).color() != top.child(1).foreground(5).color()
    assert top.child(0).foreground(5).color().name() == page_mod.QColor(page._colors["ok"]).name()
    assert top.child(1).foreground(5).color().name() == page_mod.QColor(page._colors["error"]).name()


@pytest.mark.parametrize(("accepted", "mark"), [(True, "✓"), (False, "✗"), (MISSING, "—")])
def test_accept_column_renders_verdict_marks(accepted, mark) -> None:
    assert page_mod._accept_mark(accepted)[0] == mark


def test_accept_column_uses_the_data_layer_verdict(make_page, tmp_path) -> None:
    campaign = _write_campaign(tmp_path / "camp", blocks=2)
    page = make_page()

    page.set_campaign_dir(campaign)

    top = _tree_top_rows(page)[0]
    # _write_campaign 让块 0 accepted=True, 块 1 accepted=False
    assert top.child(0).text(5) == "✓"
    assert top.child(1).text(5) == "✗"


def test_all_block_columns_visible_with_the_accept_column_towards_the_right(make_page, tmp_path) -> None:
    # 树只有 4 列而块行写 7 格 → 「验收 / 说明」被静默裁掉; 这里钉住 7 列都在。
    campaign = _write_campaign(tmp_path / "camp", blocks=1)
    page = make_page()

    page.set_campaign_dir(campaign)

    tree = page.campaign_tree()
    assert tree.columnCount() == 7
    headers = [tree.headerItem().text(column) for column in range(7)]
    # 段义 / 块义分两行, 两行都得在 —— 少一行就有一级行的数据落在别人的表头下。
    assert headers[0] == "形状 / 阶段\n块号"
    assert headers[1] == "状态\nCV"
    assert headers[2] == "seed CV → final CV\n饱和像素"
    assert headers[3] == "段内峰值\n核心峰值"
    assert headers[4] == "曝光 ms"
    assert headers[5] == "验收"
    assert headers[6] == "说明"
    # 段行与块行的曝光在同一列 (段行取 StageRow.exposure_ms)。
    stage = cp.collect_campaign_progress(campaign).stages[0]
    top = _tree_top_rows(page)[0]
    expected = stage.exposure_ms if isinstance(stage.exposure_ms, str) else f"{stage.exposure_ms:.1f}"
    assert top.text(4) == expected
    assert top.child(0).text(4) == f"{stage.blocks[0].exposure_ms:.1f}"


def test_unknown_stage_status_is_printed_raw(make_page, tmp_path) -> None:
    campaign = _write_campaign(tmp_path / "camp", blocks=1)
    (campaign / "journal.json").write_text(
        json.dumps([{"stage": "square/mono12_cv", "status": "exploded"}]), encoding="utf-8"
    )
    page = make_page()

    page.set_campaign_dir(campaign)

    assert _tree_top_rows(page)[0].text(1) == "exploded"


# --- 刷新 / 读取实测 / 主题 ------------------------------------------------


def test_cost_read_button_also_fills_the_campaign_tree(make_page, tmp_path) -> None:
    campaign = _write_campaign(tmp_path / "camp", blocks=1)
    page = make_page()
    assert page.campaign_tree().topLevelItemCount() == 0

    page._cost_dir.setText(str(campaign))
    page._cost_btn.click()

    assert page.campaign_dir() == campaign
    assert page.campaign_tree().topLevelItemCount() == 1


def test_refresh_button_rereads_the_watched_dir(make_page, tmp_path) -> None:
    campaign = _write_campaign(tmp_path / "camp", blocks=1)
    page = make_page()
    page.set_campaign_dir(campaign)
    assert _tree_top_rows(page)[0].child(0).text(1) == "0.9000"

    rows_path = campaign / "shape" / "square" / "mono12_cv" / "manifest.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    for row in rows:
        if row["label"] == "square_0_A1":
            row["metrics"]["cv"] = 0.1234
    rows_path.write_text(json.dumps(rows), encoding="utf-8")

    page._campaign_refresh.click()

    assert _tree_top_rows(page)[0].child(0).text(1) == "0.1234"


def test_theme_repaint_keeps_the_tree_without_rereading(make_page, tmp_path, monkeypatch) -> None:
    campaign = _write_campaign(tmp_path / "camp", blocks=1)
    page = make_page()
    page.set_campaign_dir(campaign)
    before = page.campaign_tree().topLevelItemCount()

    calls = {"n": 0}
    real = cp.collect_campaign_progress

    def counting(path):
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(page_mod.campaign_progress, "collect_campaign_progress", counting)

    page._apply_theme()

    assert calls["n"] == 0
    assert page.campaign_tree().topLevelItemCount() == before
    assert _tree_top_rows(page)[0].child(0).text(1) == f"{cp.collect_campaign_progress(campaign).stages[0].blocks[0].cv:.4f}"
