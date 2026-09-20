"""镜像测试: 权威 run_benchmark_suite / gerchberg_saxton 的核心语义。

设计 (严格对齐 AGENTS.md):
- 权威模块 `src/ao_shaping/algorithm/beam_shaping_benchmark.py` 与
  `src/ao_shaping/algorithm/gerchberg_saxton.py` 是**只读权威**,
  本测试**只 import、不改**, 全部断言都落在「如实校验权威语义」上。
- 「跑一次拿张表」已有双通道: CLI --suite (全 9 格 + GIF) 与 pytest 镜像
  (test_beam_shaping_benchmark.py, 已绿)。本文件补充**小规模 2×2 子格**回归。

已知语义 (如实记录, 非缺陷):
- `run_benchmark_suite` 返回 `(rows, df)` 二元组; 标量 `grid_size` (如 32)
  会在 `run_benchmark` 入口被规范为 2D 网格 `(32, 32)`, 因此套件调用**不会**
  触发 GS 的 2D 校验 — 该校验只在直接向 `gerchberg_saxton` 传 1D 振幅时生效。
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.beam_shaping_benchmark import (
    SUITE_ALGORITHMS,
    SUITE_SHAPES,
    run_benchmark_suite,
)
from ao_shaping.algorithm.signal_processing.gerchberg_saxton import gerchberg_saxton

# 行字段与权威 _write_table 的 9 列表头一一对应 (权威键名为 *_area 后缀)。
_METRIC_FIELDS = (
    "algorithm",
    "shape",
    "requested_area",
    "measured_area",
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

    def test_small_2x2_subset_rows_have_nine_fields(self, tmp_path):
        """2 算法 × 2 形状: 每行 9 字段齐全 (表结构不变式)。"""
        rows, df = run_benchmark_suite(
            algorithms=["backprop", "spgd-sim"],
            shapes=["circle", "square"],
            grid_size=32,
            max_frames=0,
            output_dir=str(tmp_path),
        )
        assert len(rows) == 4
        assert len(df) == 4
        for row in rows:
            assert all(k in row for k in _METRIC_FIELDS)

    def test_gs_requires_2d_amplitude(self):
        """权威语义: gerchberg_saxton 直接收 1D 振幅抛 ValueError("2D")。"""
        with pytest.raises(ValueError, match="2D"):
            gerchberg_saxton(
                source_amplitude=np.ones(32),
                target_amplitude=np.ones((32, 32)),
            )
