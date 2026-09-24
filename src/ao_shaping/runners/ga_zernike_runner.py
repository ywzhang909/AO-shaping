"""GA Zernike优化器CLI命令.

使用遗传算法(Genetic Algorithm)优化Zernike系数,
通过SLM显示并使用WFS测量波前RMS进行优化.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.drivers import MlaRes
from ao_shaping.optimizer.wf.ga_zernike import optimizer_ga
from ao_shaping.runners.runner_common import GaZernikeParams, with_params
from ao_shaping.utils.image.display import plot_funcs
from ao_shaping.utils.io.cli_helpers import (
    setup_coredumpy,
    get_date_dir_name,
    get_debug_mode,
)
from ao_shaping.utils.io.file import (
    build_debug_save_paths,
    save_optimization_debug_artifacts,
)


@click.command(name="ga-zernike")
@with_params(GaZernikeParams, kw_name="params")
def run(params: GaZernikeParams) -> None:
    """使用遗传算法优化Zernike系数进行波前校正."""
    debug = get_debug_mode()

    # Convert wfs_res from int to MlaRes
    wfs_res_enum = MlaRes.from_str(str(params.wfs_res))

    click.echo("GA-Zernike优化参数:")
    click.echo(f"  种群大小: {params.population_size}")
    click.echo(f"  迭代代数: {params.n_generations}")
    click.echo(f"  交叉概率: {params.crossover_prob}")
    click.echo(f"  变异概率: {params.mutation_prob}")
    click.echo(f"  锦标赛大小: {params.tournament_size}")
    click.echo(f"  精英数量: {params.elite_count}")
    click.echo(f"  最大Zernike阶数: {params.n_max}")
    click.echo(f"  波长: {params.wavelength} nm")
    click.echo(f"  WFS分辨率: {wfs_res_enum}")
    click.echo(f"  瞳孔直径: {params.pupil_diameter}")
    click.echo(f"  瞳孔中心: {params.pupil_center}")
    click.echo(f"  早停阈值: {params.early_stop_threshold}")
    click.echo(f"  SLM编号: {params.slm_number}")
    click.echo(f"  去除倾斜: {params.remove_tilt}")
    click.echo(f"  X偏移: {params.shift_x}")
    click.echo(f"  Y偏移: {params.shift_y}")

    recorder = optimizer_ga(
        n_generations=params.n_generations,
        population_size=params.population_size,
        crossover_prob=params.crossover_prob,
        mutation_prob=params.mutation_prob,
        tournament_size=params.tournament_size,
        elite_count=params.elite_count,
        n_max=params.n_max,
        wavelength=params.wavelength,
        wfs_res=wfs_res_enum,
        pupil_diameter=params.pupil_diameter,
        pupil_center=params.pupil_center,
        early_stop_threshold=params.early_stop_threshold,
        slm_number=params.slm_number,
        remove_tilt=params.remove_tilt,
        shift_x=params.shift_x,
        shift_y=params.shift_y,
    )

    # Extract results
    min_iter, (min_gen, min_rms) = recorder.get_best_iter()
    best_zernike = min_iter["_c"]

    root_dir = Path(params.dir)

    if debug or params.show:
        save_dir, saved_file = build_debug_save_paths(root_dir, "ga_zernike")

        save_optimization_debug_artifacts(
            records=recorder,
            save_dir=save_dir,
            saved_file_name=saved_file,
            min_epoch=min_gen,
            min_metric=min_rms,
            best_coeff_key="_c",
            init_wavefront=recorder.first["_wavefront"][0],
            opt_wavefront=min_iter["_wavefront"][0],
            init_title="初始波前",
            opt_title="最优波前",
            plot_params_note=f"Min RMS: {min_rms:.3f} @ gen {min_gen}",
        )

    flatten_dir = root_dir / "flatten_zernike" / get_date_dir_name()
    recorder.save_best(
        saved_dir=flatten_dir, target="_c", process_fn=np.round, fmt="%.6f"
    )

    click.echo("\nGA-Zernike优化完成!")
    click.echo(f"  最佳RMS: {min_rms:.4f} @ generation {min_gen}")


if __name__ == "__main__":
    setup_coredumpy()
    run()