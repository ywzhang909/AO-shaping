from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from ao_shaping.drivers import MlaRes
from ao_shaping.optimizer.wf.greedy_zernike import optimizer_greedy
from ao_shaping.runners.runner_common import (
    GreedyZernikeParams,
    WfsParams,
    ZernikeSlmParams,
    with_params,
)
from ao_shaping.utils.io.cli_helpers import (
    get_date_dir_name,
    resolve_debug,
    setup_coredumpy,
)
from ao_shaping.utils.io.file import (
    build_debug_save_paths,
    save_optimization_debug_artifacts,
)


@click.command()
@click.pass_context
@with_params(GreedyZernikeParams, kw_name="params")
@with_params(WfsParams, kw_name="wfs")
@with_params(ZernikeSlmParams, kw_name="slm")
def run(
    ctx: click.Context,
    params: GreedyZernikeParams,
    wfs: WfsParams,
    slm: ZernikeSlmParams,
) -> None:
    """Zernike波前优化器 - 贪婪局部搜索 / 启发式搜索

    通过SLM进行波前校正，最小化WFS测量的波前RMS值。
    ``--algorithm spgd`` 使用贪婪局部搜索:
    1. 随机初始化N个位置，选取最优作为起始点
    2. 每次迭代采样n个随机扰动方向
    3. 评估所有候选(当前位置+n个扰动)，选择最优
    其他取值 (ga/pso/sa/hc/rs/cem/de) 使用 algorithm/heuristic 的启发式搜索。

    调试模式: ``main.py --debug greedy-zernike`` / 本命令 ``--debug`` / ``DEBUG=1``。
    """
    debug = resolve_debug(ctx, params.debug)

    records = optimizer_greedy(
        epochs=params.epochs,
        n_init=params.n_init,
        n_directions=params.n_directions,
        perturbation_scale=params.perturbation_scale,
        init_z=None,
        pupil_center=wfs.pupil_center,
        pupil_diameter=wfs.pupil_diameter,
        early_stop_threshold=params.early_stop_threshold,
        wavelength=slm.wavelength,
        shift_x=slm.shift_x,
        shift_y=slm.shift_y,
        n_max=params.n_max,
        wfs_res=MlaRes.from_str(wfs.wfs_res),
        remove_tilt=wfs.remove_tilt,
        slm_number=slm.slm_number,
        algorithm=params.algorithm,
        pop_size=params.pop_size,
    )
    root_dir = Path(params.dir)

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