"""逐段实测成本：写盘、扫回归并与估算包络 (免设备)。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ao_shaping.tools.slm.transfer.cost_table import (
    COST_FILE,
    estimate_campaign,
    read_campaign_costs,
    record_stage_cost,
)


def test_a_measured_stage_cost_is_written_into_its_stage_directory(tmp_path):
    stage_dir = tmp_path / "shape" / "square" / "capture"
    path = record_stage_cost(stage_dir, stage="capture", shape="square",
                             seconds=72.0, blocks=12)

    assert path == stage_dir / COST_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stage"] == "capture"
    assert payload["shape"] == "square"
    assert payload["seconds"] == 72.0
    assert payload["blocks"] == 12
    assert payload["seconds_per_block"] == pytest.approx(6.0)
    assert payload["source"] == "measured"
    assert payload["timestamp_utc"]


def test_zero_blocks_leaves_the_per_block_rate_empty(tmp_path):
    """只有段首那一行时没有可比速率 —— 记 ``None``, 不当成免费的 0 秒。"""
    path = record_stage_cost(tmp_path / "capture", stage="capture", shape="square",
                             seconds=5.0, blocks=0)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["blocks"] == 0
    assert payload["seconds_per_block"] is None


def test_a_rerun_overwrites_the_stage_cost(tmp_path):
    stage_dir = tmp_path / "capture"
    record_stage_cost(stage_dir, stage="capture", shape="square", seconds=10.0, blocks=2)
    record_stage_cost(stage_dir, stage="capture", shape="square", seconds=20.0, blocks=2)

    payload = json.loads((stage_dir / COST_FILE).read_text(encoding="utf-8"))
    assert payload["seconds"] == 20.0
    assert payload["seconds_per_block"] == pytest.approx(10.0)


@pytest.mark.parametrize("seconds", [-1.0, float("nan"), float("inf"), None, "abc"])
def test_a_non_measured_elapsed_time_is_refused(tmp_path, seconds):
    with pytest.raises(ValueError):
        record_stage_cost(tmp_path, stage="capture", shape="square", seconds=seconds)


def test_the_campaign_table_merges_every_stage_cost(tmp_path):
    record_stage_cost(tmp_path / "shape" / "square" / "capture",
                      stage="capture", shape="square", seconds=60.0, blocks=12)
    record_stage_cost(tmp_path / "shape" / "star" / "refine",
                      stage="refine", shape="star", seconds=30.0, blocks=6)

    table = read_campaign_costs(tmp_path)

    assert set(table) == {"square/capture", "star/refine"}
    assert table["square/capture"]["seconds"] == 60.0
    assert table["square/capture"]["path"].endswith(COST_FILE)
    assert table["star/refine"]["blocks"] == 6


def test_an_absent_or_broken_cost_row_is_skipped(tmp_path):
    assert read_campaign_costs(tmp_path / "never_written") == {}
    broken = tmp_path / "shape" / "square" / "capture" / COST_FILE
    broken.parent.mkdir(parents=True)
    broken.write_text("{ not json", encoding="utf-8")
    record_stage_cost(tmp_path / "shape" / "star" / "refine",
                      stage="refine", shape="star", seconds=12.0, blocks=3)

    table = read_campaign_costs(tmp_path)

    assert set(table) == {"star/refine"}


def test_estimates_bracket_the_historical_rate():
    rows = [{"seconds_per_block": 5.0, "blocks": 10, "seconds": 50.0},
            {"seconds_per_block": 7.5, "blocks": 10, "seconds": 75.0},
            {"seconds_per_block": 6.0, "blocks": 10, "seconds": 60.0}]

    estimate = estimate_campaign(rows, planned_blocks=20)

    assert estimate["low"] == pytest.approx(100.0)
    assert estimate["mid"] == pytest.approx(120.0)
    assert estimate["high"] == pytest.approx(150.0)
    assert estimate["seconds_low"] == pytest.approx(100.0)
    assert estimate["basis_blocks"] == 20
    assert estimate["basis_seconds_per_block"]["rows"] == 3
    assert estimate["basis_seconds_per_block"]["mid"] == pytest.approx(6.0)


def test_the_per_block_rate_is_derived_when_the_row_has_only_totals():
    estimate = estimate_campaign([{"seconds": 90.0, "blocks": 15}], planned_blocks=2)

    assert estimate["basis_seconds_per_block"]["mid"] == pytest.approx(6.0)
    assert estimate["mid"] == pytest.approx(12.0)


def test_rows_without_a_block_count_do_not_become_rates():
    estimate = estimate_campaign([{"seconds": 90.0, "blocks": 0}], planned_blocks=4)

    assert estimate["low"] == 0.0
    assert estimate["mid"] == 0.0
    assert estimate["high"] == 0.0
    assert estimate["basis_seconds_per_block"]["rows"] == 0
    assert estimate["basis_seconds_per_block"]["mid"] is None


def test_no_history_is_reported_as_no_history():
    estimate = estimate_campaign({}, planned_blocks=8)

    assert estimate["mid"] == 0.0
    assert estimate["basis_blocks"] == 8
    assert estimate["basis_seconds_per_block"] == {"low": None, "mid": None,
                                                  "high": None, "rows": 0}


def test_the_estimate_consumes_the_read_back_table(tmp_path):
    record_stage_cost(tmp_path / "shape" / "square" / "capture",
                      stage="capture", shape="square", seconds=60.0, blocks=12)

    estimate = estimate_campaign(read_campaign_costs(tmp_path), planned_blocks=24)

    assert estimate["basis_seconds_per_block"]["rows"] == 1
    assert estimate["mid"] == pytest.approx(120.0)


def test_a_planned_block_count_of_zero_yields_no_total():
    estimate = estimate_campaign([{"seconds": 6.0, "blocks": 2}], planned_blocks=0)

    assert estimate["basis_blocks"] == 0
    assert (estimate["low"], estimate["mid"], estimate["high"]) == (0.0, 0.0, 0.0)
    assert estimate["basis_seconds_per_block"]["rows"] == 1


def test_the_cost_file_name_is_a_single_constant():
    assert COST_FILE == "cost.json"
    assert Path(COST_FILE).suffix == ".json"
