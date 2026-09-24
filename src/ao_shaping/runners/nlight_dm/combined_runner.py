from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import click
import numpy as np

from ao_shaping.optimizer.combined_optimizer import optimize_pib
from ao_shaping.utils.io.cli_helpers import (
    resolve_debug,
    setup_coredumpy,
    get_date_dir_name,
)
from ao_shaping.utils.io.file import gen_file_path_uuid, gen_date_dir, logger
from ao_shaping.utils.image.display import plot_funcs
from ao_shaping.drivers.dm._registry import resolve_dm
from ao_shaping.runners.runner_common import CombinedRunnerParams, with_params

import matplotlib.pyplot as plt


@click.command()
@click.pass_context
@with_params(CombinedRunnerParams, kw_name="params")
def run(ctx: click.Context, params: CombinedRunnerParams) -> None:
    """AdaMOD 综合PIB优化器

    使用AdaMOD优化器进行桶内功率(PIB)优化，支持自适应桶半径收缩。

    调试模式: ``main.py --debug combined`` / 本命令 ``--debug`` / 环境变量 ``DEBUG=1``
    任一开启即可输出 pkl/json 与汇总图片。
    """
    debug = resolve_debug(ctx, params.debug_flag)

    if params.load_file and Path(params.load_file).exists():
        init_v = np.loadtxt(params.load_file).tolist()
    else:
        init_v = []

    config = {
        "root_dir": params.root_dir,
        "load_file": params.load_file,
        "cam_id": params.cam_id,
        "center": params.center,
        "exposure_time_ms": params.exposure_time_ms,
        "epochs": params.epochs,
        "r_bucket": params.r_bucket,
        "delta": params.delta,
        "lr": params.lr,
        "shrink_iter": params.shrink_iter,
        "shrink_ratio": params.shrink_ratio,
        "cam_size": params.cam_size,
        "target_max_brightness": params.target_max_brightness,
        "debug": debug,
        "show": params.show,
    }
    logger.info(config)

    dm = resolve_dm(params.dm_type, keep_when_exit=True, max_neibor_diff=200)

    res_list = optimize_pib(
        dm=dm,
        center=params.center,
        epochs=params.epochs,
        r_bucket=params.r_bucket,
        delta=params.delta,
        lr=params.lr,
        exposure_time_ms=params.exposure_time_ms,
        shrink_iter=params.shrink_iter,
        shrink_ratio=params.shrink_ratio,
        cam_id=cast(int, params.cam_id),
        show=params.show,
        init_v=init_v,
        cam_size=params.cam_size,
        target_max_brightness=params.target_max_brightness,
    )

    saved_dir = f"{params.root_dir}/flatten_voltages/{get_date_dir_name()}"
    res_list.save_best(
        saved_dir, target="_v", process_fn=lambda x: np.around(x).astype(int), fmt="%d"
    )
    best_iter, (max_j_id, max_j) = res_list.get_best_iter()
    last_V = best_iter["_v"]

    if debug:
        save_dir = gen_date_dir(f"{params.root_dir}/combined")
        saved_file_name = gen_file_path_uuid(save_dir, "pkl")
        res_list.save_dataframe(saved_file_name.with_suffix(".pkl"), compression="zip")

        with saved_file_name.with_suffix(".json").open("w", encoding="utf8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        plot_funcs["img"](
            res_list.first["_img"],
            ax[0, 0],
            f"Init Image, pib={res_list.first['pib']:.3f}",
        )
        axim = plot_funcs["img"](
            best_iter["_img"], ax[0, 1], f"Best PIB Image, pib={max_j:.3f}"
        )
        fig.colorbar(axim, ax=[ax[0, 0], ax[0, 1]], orientation="horizontal")
        plot_funcs["pib_history"](res_list.dataframe["pib"], ax[1, 0])
        plot_funcs["voltages"](last_V, ax[1, 1], "Best Voltages")
        plt.savefig(saved_file_name.with_suffix(".png"))
        plt.close()

    objective_name = "PIB"
    click.echo(
        f"Combined PIB optimization complete, best {objective_name}: {max_j:.4f} @ epoch {max_j_id}"
    )


if __name__ == "__main__":
    setup_coredumpy()
    run()
