#!/usr/bin/env python3
"""Bayesian lr/delta optimization demo (extracted from
``ao_shaping.optimizer.wfless.bayes_opt_lr_delta`` — see issue #90).

Runs ``bayesian_optimize_lr_delta`` and ``grid_search_lr_delta`` on real
hardware (requires a camera); shrunk to 50 epochs for a quick pass.
"""

from __future__ import annotations

import sys
from loguru import logger
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ao_shaping.optimizer.wfless.bayes_opt_lr_delta import (
    bayesian_optimize_lr_delta,
    grid_search_lr_delta,
)


def main() -> None:
    # 示例用法
    logger.info("开始贝叶斯优化...")

    # 贝叶斯优化
    bayes_result = bayesian_optimize_lr_delta(
        n_calls=20,
        lr_bounds=(0.1, 5.0),
        delta_bounds=(0.1, 5.0),
        epochs=50,  # 减少迭代次数以加快测试
        exposure_time_ms=60
    )

    logger.info("贝叶斯优化结果:")
    logger.info(f"最优学习率: {bayes_result['best_params']['lr']:.4f}")
    logger.info(f"最优delta: {bayes_result['best_params']['delta']:.4f}")
    logger.info(f"最优PIB值: {bayes_result['best_score']:.4f}")

    logger.info("\n" + "="*50 + "\n")

    # 网格搜索
    logger.info("开始网格搜索...")
    grid_result = grid_search_lr_delta(
        lr_values=[0.5, 1.0, 1.5, 2.0, 2.5],
        delta_values=[0.5, 1.0, 1.5, 2.0, 2.5],
        epochs=50,  # 减少迭代次数以加快测试
        exposure_time_ms=60
    )

    logger.info("网格搜索结果:")
    logger.info(f"最优学习率: {grid_result['best_params']['lr']}")
    logger.info(f"最优delta: {grid_result['best_params']['delta']}")
    logger.info(f"最优PIB值: {grid_result['best_score']:.4f}")


if __name__ == "__main__":
    main()
