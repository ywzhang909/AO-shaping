"""AO-Shaping CLI主入口

提供统一的命令行界面，集成所有优化器和工具。

Usage:
    python main.py [OPTIONS] COMMAND [ARGS]...

Commands:
    wf              波前RMS优化器
    pib             轴向光束PIB优化器
    pipeline        串行WF→PIB流水线优化器
    gs              Gerchberg-Saxton全息图生成器
    zernike-matrix  Zernike响应矩阵校准
    rms-zernike     Zernike RMS优化器
    ga-zernike      GA Zernike优化器
    greedy-zernike  贪婪局部搜索Zernike优化器

Examples:
    python main.py --debug wf --epochs 10000
    python main.py --debug pib --cam_id 1 --epochs 5000
    python main.py --debug gs --target-shape gaussian --use-hardware
    python main.py --debug zernike-matrix --n-max 10
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 确保src在路径中
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import click
from loguru import logger

# Import all runners from runners package
from ao_shaping.runners import (
    wf_run,
    pib_run,
    pipeline_run,
    gs_run,
    zernike_matrix_run,
    zernike_closed_loop_run,
    rms_zernike_run,
    ga_zernike_run,
    greedy_zernike_run,
    dm_matrix_run,
    combined_run,
)


from ao_shaping.utils.cli_helpers import get_debug_mode
from ao_shaping.profiler import maybe_profile


@click.group()
@click.option("--dir", default="data", help="数据保存根目录 (default: data)")
@click.pass_context
def cli(ctx: click.Context, dir: str):
    """AO-Shaping自适应光学整形系统 CLI

    提供波前优化、光束整形、全息图生成等功能。

    全局选项:
        --dir    数据保存根目录
        DEBUG    环境变量控制调试模式 (export DEBUG=1 或 DEBUG=true)
    """
    _debug = get_debug_mode()
    if _debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        logger.debug("Debug mode enabled via DEBUG env var - DEBUG level logging active")
        logger.debug("Debug mode enabled")
    ctx.ensure_object(dict)
    ctx.obj["dir"] = dir
    ctx.obj["debug"] = _debug

# Register subcommands
cli.add_command(wf_run, name="wf")
cli.add_command(pib_run, name="pib")
cli.add_command(pipeline_run, name="pipeline")
cli.add_command(gs_run, name="gs")
cli.add_command(zernike_matrix_run, name="zernike-matrix")
cli.add_command(rms_zernike_run, name="rms-zernike")
cli.add_command(ga_zernike_run, name="ga-zernike")
cli.add_command(greedy_zernike_run, name="greedy-zernike")
cli.add_command(zernike_closed_loop_run, name="closed-loop")
cli.add_command(dm_matrix_run, name="dm-matrix")
cli.add_command(combined_run, name="combined")


# Entry point
if __name__ == "__main__":
    with maybe_profile():
        try:
            cli()
        except Exception as e:
            logger.error(f"CLI error: {e}")
            raise
