from __future__ import annotations

import click
from pathlib import Path
from typing import Literal, cast

import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.utils import gen_date_dir, gen_file_path_uuid, logger
from ao_shaping.optimizer.wf.rms import optimizer_rms_dm
from ao_shaping.utils.image.display import plot_funcs
from ao_shaping.utils.io.cli_helpers import (
    setup_coredumpy,
    get_date_dir_name,
    get_debug_mode,
)
from ao_shaping.drivers.dm._registry import resolve_dm
from ao_shaping.runners.runner_common import WfRunnerParams, with_params


@click.command()
@click.pass_context
@with_params(WfRunnerParams, kw_name="params")
def run(ctx: click.Context, params: WfRunnerParams) -> None:
    """波前优化器

    DEBUG环境变量控制调试模式。
    """
    debug = get_debug_mode()

    dm = resolve_dm(params.dm_type)
    try:
        dm.open()
        init_v = [0 for _ in range(dm.DM_NUM)]
        records = optimizer_rms_dm(
            init_v=init_v,
            epochs=params.epochs,
            wfs_res=cast(Literal["512", "768"], params.wfs_res),
            pupil_diameter=params.pupil_diameter,
            pupil_center=cast(tuple[float, float], params.pupil_center),
            early_stop_threshold=params.early_stop_threshold,
            dm=dm,
        )
    finally:
        dm.close()

    root_dir = Path(params.dir)

    min_iter, (min_epoch, min_rms) = records.get_best_iter()

    if debug:
        save_dir = gen_date_dir(root_dir / "wf")
        saved_file_name = gen_file_path_uuid(save_dir, "pkl")

        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        rms_values = records.get_sublist()
        plot_funcs["rms_history"](rms_values, ax[0, 0], min_epoch, min_rms)
        plot_funcs["voltages"](
            min_iter["_v"], ax[0, 1], f"Min J: {min_rms:.3f} @ epoch {min_epoch}"
        )
        im = plot_funcs["wavefront"](
            records.first["_wavefront"][0], ax[1, 0], "init wavefront"
        )
        plt.colorbar(im, ax=ax[1, 0], orientation="horizontal")
        im = plot_funcs["wavefront"](
            min_iter["_wavefront"][1], ax[1, 1], "opt wavefront"
        )
        plt.colorbar(im, ax=ax[1, 1], orientation="horizontal")

        plt.savefig(saved_file_name.with_suffix(".png"))
        plt.close()
        records.save_dataframe(saved_file_name.with_suffix(".zip"), compression="zip")

    save_dir = root_dir / "flatten_voltages" / get_date_dir_name()
    records.save_best(saved_dir=save_dir, target="_v", process_fn=np.round, fmt="%d")

    click.echo(f"波前优化完成，最优RMS值: {min_rms:.4f} @ epoch {min_epoch}")


if __name__ == "__main__":
    setup_coredumpy()
    run()
