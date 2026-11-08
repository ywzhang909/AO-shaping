"""campaign 计划预览的数据层：矩阵格、合计、段级成本（纯离线，无 Qt / 无设备）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ao_shaping.gui.slm.transfer_desktop import campaign_plan
from ao_shaping.gui.slm.transfer_desktop.campaign_plan import (
    CAMPAIGN_STAGE_LABELS,
    PLAN_SHAPES,
    campaign_summary,
    cost_caption,
    cost_rows,
    matrix_cells,
)
from ao_shaping.tools.slm.transfer.campaign import (
    SECONDS_PER_BLOCK,
    campaign_matrix,
    estimate_disk_gb,
    estimate_seconds,
    matrix_totals,
)
from ao_shaping.tools.slm.transfer.cost_table import record_stage_cost
from ao_shaping.tools.slm.transfer.stages import STAGE_RECIPES


def _write_cost(root: Path, *, shape: str, stage: str, seconds: float, blocks: int) -> None:
    record_stage_cost(
        root / shape / stage, stage=stage, shape=shape, seconds=seconds, blocks=blocks
    )


# --- 矩阵 -------------------------------------------------------------------


def test_the_matrix_covers_four_shapes_by_four_stages() -> None:
    cells = matrix_cells()

    assert PLAN_SHAPES == ("square", "rectangle", "triangle", "star")
    assert len(cells) == 16
    assert [cell.shape for cell in cells] == [
        shape for shape in PLAN_SHAPES for _ in STAGE_RECIPES
    ]
    assert sum(cell.blocks for cell in cells) == 112


def test_the_matrix_does_not_restate_the_campaign_numbers() -> None:
    """每一格都是 ``campaign_matrix`` 的原值 —— 数据层不另立一套阈值。"""
    expected = campaign_matrix(PLAN_SHAPES)

    for cell, row in zip(matrix_cells(), expected):
        assert (cell.shape, cell.stage) == (row.shape, row.stage)
        assert (cell.blocks, cell.bit_depth, cell.updates) == (
            row.blocks,
            row.bit_depth,
            row.updates,
        )


def test_the_selected_shape_is_the_only_highlighted_row() -> None:
    cells = matrix_cells(current_shape="triangle")

    highlighted = {cell.shape for cell in cells if cell.highlight}
    assert highlighted == {"triangle"}
    assert sum(cell.highlight for cell in cells) == len(STAGE_RECIPES)


def test_no_selection_highlights_nothing() -> None:
    assert not any(cell.highlight for cell in matrix_cells())


def test_every_stage_recipe_has_a_chinese_label() -> None:
    """键集钉住 ``STAGE_RECIPES``：新增阶段而漏加标签会被抓。"""
    assert set(CAMPAIGN_STAGE_LABELS) == {recipe.name for recipe in STAGE_RECIPES}


def test_an_unknown_stage_falls_back_to_its_name() -> None:
    assert campaign_plan._stage_label("brand_new_stage") == "brand_new_stage"


# --- 合计 -------------------------------------------------------------------


def test_the_summary_is_the_campaign_estimate() -> None:
    summary = campaign_summary()
    rows = campaign_matrix(PLAN_SHAPES)
    totals = matrix_totals(rows)
    low, expected, high = estimate_seconds(rows)

    assert (summary.segments, summary.blocks, summary.updates) == (
        totals["segments"],
        totals["blocks"],
        totals["updates"],
    )
    assert summary.seconds_expected == pytest.approx(expected)
    assert (summary.seconds_low, summary.seconds_high) == pytest.approx((low, high))
    assert summary.disk_gb == pytest.approx(estimate_disk_gb(rows))


def test_the_summary_text_carries_scale_time_disk_and_the_basis() -> None:
    text = campaign_summary().text

    assert "16 段 / 112 块" in text
    assert "45 min" in text
    assert f"{estimate_disk_gb(campaign_matrix(PLAN_SHAPES)):.1f} GB" in text
    assert "capture / fit / 重放不计入" in text


# --- 段级成本 ---------------------------------------------------------------


def test_without_a_campaign_dir_the_table_is_plan_only() -> None:
    rows = cost_rows()

    assert len(rows) == 16
    assert all(row.measured_seconds is None for row in rows)
    assert all(row.measured_blocks is None for row in rows)
    assert all(row.delta_text == "—" for row in rows)
    assert all(row.expected_seconds == row.planned_blocks * SECONDS_PER_BLOCK for row in rows)


def test_measured_seconds_fill_the_segment_they_belong_to(tmp_path: Path) -> None:
    _write_cost(tmp_path, shape="square", stage="mono12_cv", seconds=160.0, blocks=8)

    rows = {(row.shape, row.stage): row for row in cost_rows(tmp_path)}
    measured = rows[("square", "mono12_cv")]

    assert measured.measured_seconds == pytest.approx(160.0)
    assert measured.measured_blocks == 8
    # 计划 8 块 × 24 s = 192 s，实测 160 s → 快 17 %。
    assert measured.delta_text == "快 17%"
    # 没测过的段保持 None，不补 0。
    assert rows[("star", "mono12_dense")].measured_seconds is None


def test_a_slower_segment_reads_as_slow(tmp_path: Path) -> None:
    _write_cost(tmp_path, shape="star", stage="mono12_dense", seconds=144.0, blocks=4)

    row = next(r for r in cost_rows(tmp_path) if r.shape == "star" and r.stage == "mono12_dense")
    assert row.delta_text == "慢 50%"


def test_a_stage_without_blocks_cannot_report_per_block_numbers(tmp_path: Path) -> None:
    """只有段首行（blocks=0）时实测秒数在、实测块数缺 —— 不编一个 0 块。"""
    _write_cost(tmp_path, shape="square", stage="transport_cv", seconds=12.0, blocks=0)

    row = next(r for r in cost_rows(tmp_path) if r.stage == "transport_cv")
    assert row.measured_seconds == pytest.approx(12.0)
    assert row.measured_blocks is None


def test_the_caption_says_first_run_without_history() -> None:
    assert "首次运行" in cost_caption()
    assert "首次运行" in cost_caption(Path("does/not/exist"))


def test_the_caption_extrapolates_from_the_historical_per_block_seconds(
    tmp_path: Path,
) -> None:
    _write_cost(tmp_path, shape="square", stage="mono12_cv", seconds=160.0, blocks=8)

    caption = cost_caption(tmp_path)

    assert "20.0 s/块" in caption
    assert "本计划 112 块" in caption
    assert "≈37 min" in caption


def test_the_helpers_never_raise_on_a_broken_directory(tmp_path: Path) -> None:
    """给一个**文件**当 campaign 目录：成本表退化，不抛。"""
    broken = tmp_path / "not_a_dir"
    broken.write_text("{}", encoding="utf-8")

    rows = cost_rows(broken)

    assert len(rows) == 16
    assert all(row.measured_seconds is None for row in rows)
    assert isinstance(cost_caption(broken), str)
