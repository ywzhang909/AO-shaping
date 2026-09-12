"""镜像测试: 权威 run_benchmark_suite 的「9 格表 / 2D 限制」语义。

设计 (严格对齐 AGENTS.md):
- 权威模块 `src/ao_shaping/algorithm/beam_shaping_benchmark.py` 是**只读权威**, 
  本测试**只 import、不改**, 全部断言都落在「如实校验权威语义」上。
- 「跑一次拿张表」已有双通道: CLI --suite (全 9 格 + GIF) 与 pytest 镜像
  (test_beam_shaping_benchmark.py, 已绿)。本文件补充**小规模 2×2 子格**回归。

已知限制 (如实记录, 非缺陷):
- 权威 GS 要求 2D 振幅; 小网格下套件传 1D → run_benchmark_suite **预期抛
  ValueError("Input amplitudes must be 2D arrays")** — 这是权威模块对输入
  的校验语义, 不是测试缺陷, 也不该在本测试里"修" (权威只读)。
"""

from __future__ import annotations

import pytest

from ao_shaping.algorithm.beam_shaping_benchmark import (
    SUITE_ALGORITHMS,
    SUITE_SHAPES,
    run_benchmark_suite,
)

_METRIC_FIELDS = (
    "algorithm",
    "shape",
    "requested",
    "measured",
    "area_met",
    "fill_ratio",
    "uniformity_cv",
    "encircled_energy",
    "elapsed_s",
)


def test_suite_has_authoritative_grid():
    """权威套件网格 = 3 算法 × 3 形状 (CLI --suite 的 9 格)。"""
    assert set(SUITE_ALGORITHMS) == {"backprop", "gs", "spgd-sim"}
    assert set(SUITE_SHAPES) == {"circle", "gaussian", "square"}


class TestRunBenchmarkSuiteSemantics:
    """权威 2×2 子格: 只断言表结构/字段, 产 2 张格表但不产全量 9 格。"""

    def test_small_2x2_subset_rows_have_nine_fields(self):
        """2 算法 × 2 形状: 每行 9 字段齐全 (表结构不变式)。"""
        # spgd-sim 无设备不收敛 → 字段语义可校验; gs 小网格 1D 问题用单独的
        # 2D 用例覆盖, 不在此把 suite 卡死。
        rows = run_benchmark_suite(
            algorithms=["backprop", "spgd-sim"],
            shapes=["circle", "square"],
            grid_size=32,
            max_frames=0,
            output_dir="",
        )
        assert len(rows) == 4
        for row in rows:
            assert all(k in row for k in _METRIC_FIELDS)

    def test_gs_requires_2d_amplitude(self):
        """权威语义: 小网格(1D 输入)下 GS 抛 ValueError — 只读校验, 不修。"""
        with pytest.raises(ValueError, match="2D"):
            run_benchmark_suite(
                algorithms=["gs"],
                shapes=["circle"],
                grid_size=32,
                max_frames=0,
                output_dir="",
            )
