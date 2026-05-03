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

Examples:
    python main.py wf --epochs 10000 --debug
    python main.py pib --cam_id 1 --epochs 5000
    python main.py gs --target-shape gaussian --use-hardware
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

# Import all runners
from ao_shaping import wf_runner
from ao_shaping import axis_beam_runner
from ao_shaping import pipeline_runner
from ao_shaping import gs_hologram_runner


@click.group()
@click.option("--debug", is_flag=True, help="启用全局调试模式")
@click.option("--dir", default="data", help="数据保存根目录 (default: data)")
@click.pass_context
def cli(ctx: click.Context, debug: bool, dir: str):
    """AO-Shaping自适应光学整形系统 CLI
    
    提供波前优化、光束整形、全息图生成等功能。
    """
    # 确保context对象存在
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["dir"] = dir
    
    if debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        logger.debug("Debug mode enabled")


# Register subcommands
cli.add_command(wf_runner.run, name="wf")
cli.add_command(axis_beam_runner.run, name="pib")
cli.add_command(pipeline_runner.run, name="pipeline")
cli.add_command(gs_hologram_runner.run, name="gs")


# Entry point
if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        logger.error(f"CLI error: {e}")
        raise
