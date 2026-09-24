import click
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.algorithm.heuristic.search import heuristic_algorithm_choices
from ao_shaping.optimizer.wf.greedy_zernike import optimizer_greedy
from ao_shaping.utils.image.display import plot_funcs
from ao_shaping.utils.wavefront.matrix_utils import calc_n_zernike_terms
from ao_shaping.utils.io.cli_helpers import (
    parse_tuple,
    setup_coredumpy,
    get_date_dir_name,
    resolve_debug,
)
from ao_shaping.drivers import MlaRes
from ao_shaping.runners.runner_common import (
    build_debug_save_paths,
    save_optimization_debug_artifacts,
    wfs_options,
    zernike_slm_options,
)


@click.command()
@click.option("-d", "--dir", default="data", help="数据保存根目录 (default: data)")
@click.option("-e", "--epochs", default=2000, help="优化迭代次数 (default: 2000)")
@click.option("-n", "--n-max", default=4, help="Zernike最大阶数 (default: 4)")
@wfs_options
@click.option(
    "-t", "--early_stop_threshold", default=0.12, help="早停阈值 (default: 0.12)"
)
@zernike_slm_options
@click.option(
    "--show", is_flag=True, help="显示远场光斑CCD图像和优化历史 (default: False)"
)
@click.option("--n-init", default=10, help="初始随机位置数量 (default: 10)")
@click.option("--n-directions", default=5, help="每次迭代的随机方向数量 (default: 5)")
@click.option(
    "--perturbation-scale", default=5.0, help="扰动幅度缩放因子 (default: 5.0)"
)
@click.option(
    "--algorithm",
    type=click.Choice(list(heuristic_algorithm_choices()), case_sensitive=False),
    default="spgd",
    show_default=True,
    help="搜索算法: spgd (贪婪局部搜索) 或启发式 (ga/pso/sa/hc/rs/cem/de)",
)
@click.option(
    "--pop_size",
    type=int,
    default=None,
    help="种群规模 (ga/pso/cem/de 使用; 默认取算法默认值)",
)
@click.option(
    "--debug",
    "debug_flag",
    is_flag=True,
    default=None,
    help="启用调试模式: 保存 pkl/json 与汇总图",
)
@click.pass_context
def run(
    ctx,
    dir,
    epochs,
    n_max,
    wfs_res,
    pupil_diameter,
    pupil_center,
    early_stop_threshold,
    wavelength,
    shift_x,
    shift_y,
    slm_number,
    remove_tilt,
    show,
    n_init,
    n_directions,
    perturbation_scale,
    algorithm,
    pop_size,
    debug_flag,
):
    """Zernike波前优化器 - 贪婪局部搜索 / 启发式搜索

    通过SLM进行波前校正，最小化WFS测量的波前RMS值。
    ``--algorithm spgd`` 使用贪婪局部搜索:
    1. 随机初始化N个位置，选取最优作为起始点
    2. 每次迭代采样n个随机扰动方向
    3. 评估所有候选(当前位置+n个扰动)，选择最优
    其他取值 (ga/pso/sa/hc/rs/cem/de) 使用 algorithm/heuristic 的启发式搜索。

    调试模式: ``main.py --debug greedy-zernike`` / 本命令 ``--debug`` / ``DEBUG=1``。
    """
    debug = resolve_debug(ctx, debug_flag)

    records = optimizer_greedy(
        epochs=epochs,
        n_init=n_init,
        n_directions=n_directions,
        perturbation_scale=perturbation_scale,
        init_z=None,
        pupil_center=pupil_center,
        pupil_diameter=pupil_diameter,
        early_stop_threshold=early_stop_threshold,
        wavelength=wavelength,
        shift_x=shift_x,
        shift_y=shift_y,
        n_max=n_max,
        wfs_res=MlaRes.from_str(wfs_res),
        remove_tilt=remove_tilt,
        slm_number=slm_number,
        algorithm=algorithm,
        pop_size=pop_size,
    )
    root_dir = Path(dir)

    min_iter, (min_epoch, min_rms) = records.get_best_iter()

    if debug:
        save_dir, saved_file = build_debug_save_paths(root_dir, "greedy_zernike")

        save_optimization_debug_artifacts(
            records=records,
            save_dir=save_dir,
            saved_file_name=saved_file,
            min_epoch=min_epoch,
            min_metric=min_rms,
            best_coeff_key="_c",
            init_wavefront=records.first["_wavefront"][0],
            opt_wavefront=min_iter["_wavefront"][1],
            init_title="初始波前",
            opt_title="最优波前",
            plot_params_note=f"Min RMS: {min_rms:.3f} @ epoch {min_epoch}",
        )

    save_dir = root_dir / "flatten_zernike" / get_date_dir_name()
    records.save_best(saved_dir=save_dir, target="_c", process_fn=np.round, fmt="%.6f")

    click.echo(f"贪婪优化完成，最优RMS值: {min_rms:.4f} @ epoch {min_epoch}")


if __name__ == "__main__":
    setup_coredumpy()
    run()
