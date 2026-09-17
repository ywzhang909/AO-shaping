"""光束整形基准 CLI runner (shaping_run)。

以权威的 :mod:`ao_shaping.algorithm.beam_shaping_benchmark` 公共 API 为准，
本文件只做 Click CLI 包装与注册，不含任何算法/指标逻辑（权威模块不改）。

用法示例::

    python -m ao_shaping.runners.shaping_runner \\
        --algorithm gs --shape square --iterations 200 --seed 42
    python -m ao_shaping.runners.shaping_runner \\
        --suite --algorithms gs,spgd --shapes square,circle --iterations 40
    # 经 runners hub:
    python -m ao_shaping.runners <subcommand> ...
"""

from __future__ import annotations

from pathlib import Path

import click

from ao_shaping.algorithm.beam_shaping_benchmark import (
    DEFAULT_GRID,
    DEFAULT_MAX_FRAMES,
    DEFAULT_TARGET_AREA,
    SUITE_ALGORITHMS,
    SUITE_SHAPES,
    run_benchmark,
    run_benchmark_suite,
)
from ao_shaping.utils.cli_helpers import setup_coredumpy


def _split_csv(ctx: click.Context, param: click.Parameter, value: str | None):
    """将逗号分隔的字符串转为 list; 空/None 返回 None (表示"全部")。"""
    if value is None:
        return None
    items = [s.strip() for s in value.split(",") if s.strip()]
    return items or None


@click.command(name="shaping")
@click.option("--algorithm", type=str, default="gs", show_default=True, help="算法: gs / spgd / backprop / 可微")
@click.option("--shape", type=str, default="square", show_default=True, help="目标形状: square / rectangle / circle / pentagon / gaussian")
@click.option("--grid-size", type=int, default=DEFAULT_GRID[0], show_default=False, help="网格边长 (N×N, default: 128)")
@click.option("--target-area", type=int, default=DEFAULT_TARGET_AREA, show_default=True, help="目标面积 (像素)")
@click.option("--aspect-ratio", type=float, default=1.0, show_default=True, help="长宽比 (仅 rectangle/square)")
@click.option("--iterations", type=int, default=None, help="迭代次数 (default: 算法默认)")
@click.option("--seed", type=int, default=42, show_default=True, help="随机种子")
@click.option("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, show_default=True, help="GIF 最大帧数")
@click.option("--device", type=str, default=None, help="计算设备 (default: auto)")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None, help="输出目录 (default: 自动日期目录)")
@click.option("--gif/--no-gif", default=True, show_default=True, help="是否生成演化 GIF (单点 benchmark)")
@click.option("--suite", is_flag=True, default=False, help="运行 run_benchmark_suite (算法×形状网格)")
@click.option("--algorithms", type=str, default=None, callback=_split_csv, help="suite 算法列表, 逗号分隔 (default: 全部)")
@click.option("--shapes", type=str, default=None, callback=_split_csv, help="suite 形状列表, 逗号分隔 (default: 全部)")
def run(
    algorithm: str,
    shape: str,
    grid_size: int,
    target_area: int,
    aspect_ratio: float,
    iterations: int | None,
    seed: int,
    max_frames: int,
    device: str | None,
    output_dir: str | None,
    gif: bool,
    suite: bool,
    algorithms: list[str] | None,
    shapes: list[str] | None,
):
    """光束整形基准 CLI (包装权威 beam_shaping_benchmark)。

    --suite 时调用 run_benchmark_suite (无 make_gif); 否则调用 run_benchmark。
    """
    setup_coredumpy()

    grid = (grid_size, grid_size)

    if suite:
        rows, df = run_benchmark_suite(
            algorithms=algorithms or list(SUITE_ALGORITHMS),
            shapes=shapes or list(SUITE_SHAPES),
            grid_size=grid,
            target_area=target_area,
            aspect_ratio=aspect_ratio,
            iterations=iterations,
            seed=seed,
            max_frames=max_frames,
            device=device,
            output_dir=output_dir,
        )
        click.echo(f"\n🏁 suite 完成: {len(rows)} 行")
        if not df.empty:
            click.echo(df.to_string(index=False))
        return rows, df if False else None  # click runner 不需要返回值; 仅便于测试

    result = run_benchmark(
        algorithm,
        shape,
        grid_size=grid,
        target_area=target_area,
        aspect_ratio=aspect_ratio,
        iterations=iterations,
        seed=seed,
        max_frames=max_frames,
        device=device,
        output_dir=output_dir,
        make_gif=gif,
    )
    click.echo(f"✅ {result.get('algorithm')} × {result.get('shape')}: "
               f"面积 {result.get('requested_area')} → {result.get('measured_area')} "
               f"(fill_ratio={result.get('fill_ratio', 0):.3f}, "
               f"CV={result.get('uniformity_cv', 0):.3f})")
    return result


if __name__ == "__main__":
    run()
