import click
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.optimizer.wf.greedy_zernike import optimizer_greedy
from ao_shaping.utils.display import plot_funcs
from ao_shaping.utils.matrix_utils import calc_n_zernike_terms
from ao_shaping.utils.cli_helpers import (
    parse_tuple,
    setup_coredumpy,
    get_date_dir_name,
    get_debug_mode,
)
from ao_shaping.drivers import MlaRes
from ao_shaping.runners.runner_common import (
    build_debug_save_paths,
    save_optimization_debug_artifacts,
)


@click.command()
@click.option("-d", "--dir", default="data", help="数据保存根目录 (default: data)")
@click.option("-e", "--epochs", default=2000, help="优化迭代次数 (default: 2000)")
@click.option("-n", "--n-max", default=4, help="Zernike最大阶数 (default: 4)")
@click.option("-r", "--wfs_res", default="1024", help="WFS分辨率 (default: 1024)")
@click.option("-p", "--pupil_diameter", default=2.7, help="瞳孔直径 (default: 2.7)")
@click.option(
    "-c",
    "--pupil_center",
    callback=parse_tuple,
    default="(0,0)",
    help="瞳孔中心坐标 (default: (0,0))",
)
@click.option(
    "-t", "--early_stop_threshold", default=0.12, help="早停阈值 (default: 0.12)"
)
@click.option("--wavelength", default=532, help="SLM波长 (nm, default: 532)")
@click.option("--shift-x", default=0, help="SLM X方向平移 (像素, default: 0)")
@click.option("--shift-y", default=0, help="SLM Y方向平移 (像素, default: 0)")
@click.option("--slm-number", default=1, help="SLM设备编号 (default: 1)")
@click.option("--remove-tilt", is_flag=True, help="移除波前测量中的倾斜项")
@click.option(
    "--show", is_flag=True, help="显示远场光斑CCD图像和优化历史 (default: False)"
)
@click.option("--n-init", default=10, help="初始随机位置数量 (default: 10)")
@click.option("--n-directions", default=5, help="每次迭代的随机方向数量 (default: 5)")
@click.option(
    "--perturbation-scale", default=5.0, help="扰动幅度缩放因子 (default: 5.0)"
)
def run(
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
):
    """Zernike波前优化器 - 贪婪局部搜索算法

    使用贪婪局部搜索算法通过SLM进行波前校正，最小化WFS测量的波前RMS值。
    算法流程:
    1. 随机初始化N个位置，选取最优作为起始点
    2. 每次迭代采样n个随机扰动方向
    3. 评估所有候选(当前位置+n个扰动)，选择最优
    DEBUG环境变量控制调试模式。
    """
    debug = get_debug_mode()

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
