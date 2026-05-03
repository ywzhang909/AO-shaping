"""GA Zernike优化器CLI命令.

使用遗传算法(Genetic Algorithm)优化Zernike系数,
通过SLM显示并使用WFS测量波前RMS进行优化.
"""

from __future__ import annotations

import click

from ao_shaping.drivers import MlaRes
from ao_shaping.optimizer.wf.ga_zernike import optimizer_ga
from ao_shaping.utils.cli_helpers import setup_coredumpy


@click.command(name="ga-zernike")
@click.option(
    "--population-size",
    type=int,
    default=50,
    help="种群大小 (default: 50)",
)
@click.option(
    "--n-generations",
    type=int,
    default=2000,
    help="GA迭代代数 (default: 2000)",
)
@click.option(
    "--crossover-prob",
    type=float,
    default=0.7,
    help="交叉概率 (default: 0.7)",
)
@click.option(
    "--mutation-prob",
    type=float,
    default=0.15,
    help="变异概率 (default: 0.15)",
)
@click.option(
    "--tournament-size",
    type=int,
    default=3,
    help="锦标赛选择大小 (default: 3)",
)
@click.option(
    "--elite-count",
    type=int,
    default=2,
    help="精英个体数量 (default: 2)",
)
@click.option(
    "-n",
    "--n-max",
    type=int,
    default=4,
    help="最大Zernike径向阶数 (default: 4)",
)
@click.option(
    "-w",
    "--wavelength",
    type=int,
    default=532,
    help="SLM波长 (nm) (default: 532)",
)
@click.option(
    "--wfs-res",
    type=int,
    default=1024,
    help="WFS分辨率 (default: 1024)",
)
@click.option(
    "--pupil-diameter",
    type=float,
    default=4.6,
    help="WFS瞳孔直径 (default: 4.6)",
)
@click.option(
    "--early-stop-threshold",
    type=float,
    default=0.01,
    help="早停RMS阈值 (default: 0.01)",
)
@click.option(
    "--slm-number",
    type=int,
    default=1,
    help="SLM设备编号 (default: 1)",
)
@click.option(
    "--remove-tilt",
    is_flag=True,
    help="去除波前倾斜 (default: False)",
)
@click.option(
    "--shift-x",
    type=int,
    default=0,
    help="SLM X方向偏移 (pixels) (default: 0)",
)
@click.option(
    "--shift-y",
    type=int,
    default=0,
    help="SLM Y方向偏移 (pixels) (default: 0)",
)
def run(
    population_size: int,
    n_generations: int,
    crossover_prob: float,
    mutation_prob: float,
    tournament_size: int,
    elite_count: int,
    n_max: int,
    wavelength: int,
    wfs_res: int,
    pupil_diameter: float,
    early_stop_threshold: float,
    slm_number: int,
    remove_tilt: bool,
    shift_x: int,
    shift_y: int,
) -> None:
    """使用遗传算法优化Zernike系数进行波前校正."""
    # Convert wfs_res from int to MlaRes
    wfs_res_enum = MlaRes.from_str(str(wfs_res))

    click.echo(f"GA-Zernike优化参数:")
    click.echo(f"  种群大小: {population_size}")
    click.echo(f"  迭代代数: {n_generations}")
    click.echo(f"  交叉概率: {crossover_prob}")
    click.echo(f"  变异概率: {mutation_prob}")
    click.echo(f"  锦标赛大小: {tournament_size}")
    click.echo(f"  精英数量: {elite_count}")
    click.echo(f"  最大Zernike阶数: {n_max}")
    click.echo(f"  波长: {wavelength} nm")
    click.echo(f"  WFS分辨率: {wfs_res_enum}")
    click.echo(f"  瞳孔直径: {pupil_diameter}")
    click.echo(f"  早停阈值: {early_stop_threshold}")
    click.echo(f"  SLM编号: {slm_number}")
    click.echo(f"  去除倾斜: {remove_tilt}")
    click.echo(f"  X偏移: {shift_x}")
    click.echo(f"  Y偏移: {shift_y}")

    recorder = optimizer_ga(
        n_generations=n_generations,
        population_size=population_size,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        tournament_size=tournament_size,
        elite_count=elite_count,
        n_max=n_max,
        wavelength=wavelength,
        wfs_res=wfs_res_enum,
        pupil_diameter=pupil_diameter,
        early_stop_threshold=early_stop_threshold,
        slm_number=slm_number,
        remove_tilt=remove_tilt,
        shift_x=shift_x,
        shift_y=shift_y,
    )

    # Extract results
    best_rms = recorder.get_best_target("rms")
    best_gen = recorder.last.get("_generation", 0)
    best_zernike = recorder.get_best_target("_c")

    click.echo(f"\nGA-Zernike优化完成!")
    click.echo(f"  最佳RMS: {best_rms:.4f}")
    click.echo(f"  最佳代数: {best_gen}")
    click.echo(f"  Zernike系数: {best_zernike}")


if __name__ == "__main__":
    setup_coredumpy()
    run()