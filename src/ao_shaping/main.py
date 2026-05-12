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

Examples:
    python main.py --debug wf --epochs 10000
    python main.py --debug pib --cam_id 1 --epochs 5000
    python main.py --debug gs --target-shape gaussian --use-hardware
    python main.py --debug zernike-matrix --n-max 10
"""

from __future__ import annotations

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
)


@click.group()
@click.option("--debug", is_flag=True, default=False, help="启用全局调试模式 (loguru显示DEBUG级别)")
@click.option("--dir", default="data", help="数据保存根目录 (default: data)")
@click.pass_context
def cli(ctx: click.Context, debug: bool, dir: str):
    """AO-Shaping自适应光学整形系统 CLI
    
    提供波前优化、光束整形、全息图生成等功能。
    
    全局选项:
        --debug  启用调试模式，loguru显示DEBUG级别日志
        --dir    数据保存根目录
    """
    # 确保context对象存在
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["dir"] = dir

    # 配置loguru日志级别
    # 默认情况下loguru不显示DEBUG级别(只显示INFO及以上)
    # 启用--debug时设置级别为DEBUG
    if debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        logger.debug("Debug mode enabled - DEBUG level logging active")
    # else: 保持默认配置，不显示DEBUG


# Register subcommands
cli.add_command(wf_run, name="wf")
cli.add_command(pib_run, name="pib")
cli.add_command(pipeline_run, name="pipeline")
cli.add_command(gs_run, name="gs")
cli.add_command(zernike_matrix_run, name="zernike-matrix")
cli.add_command(rms_zernike_run, name="rms-zernike")
cli.add_command(ga_zernike_run, name="ga-zernike")
cli.add_command(zernike_closed_loop_run, name="closed-loop")


# Entry point
if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        logger.error(f"CLI error: {e}")
        raise
