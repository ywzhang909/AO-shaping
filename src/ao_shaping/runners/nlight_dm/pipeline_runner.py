from __future__ import annotations

import os
from typing import Literal, cast

import click
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.utils import gen_date_dir, gen_file_path_uuid, logger
from ao_shaping.utils.image.display import plot_funcs
from ao_shaping.optimizer.wf.rms import optimizer_rms_dm
from ao_shaping.optimizer.wfless.pib import optimize_pib
from ao_shaping.utils.io.cli_helpers import (
    setup_coredumpy,
    get_date_dir_name,
    get_debug_mode,
)
from ao_shaping.drivers.dm._registry import resolve_dm
from ao_shaping.runners.runner_common import PipelineRunnerParams, with_params


@click.command()
@click.pass_context
@with_params(PipelineRunnerParams, kw_name="params")
def run(ctx: click.Context, params: PipelineRunnerParams) -> None:
    """串行优化器（先波前优化，再轴向光束优化）

    DEBUG环境变量控制调试模式。
    """
    debug = get_debug_mode()

    dm = resolve_dm(params.dm_type)

    if params.load_file:
        last_v = np.loadtxt(params.load_file)
        init_v = last_v.tolist()
    else:
        init_v = []

    try:
        dm.open()

        wf_records = optimizer_rms_dm(
            init_v=init_v,
            pupil_diameter=params.pupil_diameter,
            wfs_res=cast(Literal["512", "768"], params.wfs_res),
            early_stop_threshold=params.rms_threshold,
            epochs=params.wf_epochs,
            dm=dm,
        )

        min_iter, (min_epoch, min_rms) = wf_records.get_best_iter()
        logger.info("WF优化完成，最佳迭代：{}，RMS：{:.4f}", min_epoch, min_rms)

        if debug:
            init_wf, min_wf = (
                wf_records.first["_wavefront"][0],
                min_iter["_wavefront"][0],
            )

        dm_available = np.ones(dm.DM_NUM, dtype=bool)
        dm_available[0] = False
        if params.dm_unit_mask == "inner":
            dm_available[21:] = False
        elif params.dm_unit_mask == "outer":
            dm_available[:39] = False

        ccd_records = optimize_pib(
            cam_id=cast(int, params.cam_id),
            center="mass",
            exposure_time_ms=params.exposure_time_ms,
            cam_size=params.cam_size,
            dm_unit_mask=dm_available,
            target_max_brightness=0,
            epochs=params.epochs,
            lr=0.9,
            delta=0.9,
            shrink_iter=20,
            shrink_ratio=0.8,
            r_bucket=int(os.environ.get("IDEAL_SPOT_RADIUS", 6)),
            init_v=min_iter["_v"],
            show=False,
            dm=dm,
        )

        max_pid_iter, (max_epoch, max_pib) = ccd_records.get_best_iter()
        last_V = max_pid_iter["_v"]

        save_dir = Path(params.dir) / "flatten_voltages" / get_date_dir_name()

        def np_array_to_int(arr):
            return arr.astype(int)

        wf_records.save_best(
            saved_dir=save_dir, target="_v", process_fn=np_array_to_int, fmt="%d"
        )
        ccd_records.save_best(
            saved_dir=save_dir, target="_v", process_fn=np_array_to_int, fmt="%d"
        )

        if debug:
            fig, ax = plt.subplots(2, 4, figsize=(12, 8))
            rms_values = wf_records.get_sublist()
            plot_funcs["rms_history"](rms_values, ax[0, 0], min_epoch, min_rms)
            pib_values = ccd_records.get_sublist()[1:]
            plot_funcs["pib_history"](pib_values, ax[1, 0])
            axim = plot_funcs["wavefront"](init_wf, ax[0, 1], "Init WF")
            plt.colorbar(
                axim, ax=ax[0, 1], orientation="vertical", fraction=0.046, pad=0.04
            )
            axim = plot_funcs["wavefront"](min_wf, ax[1, 1], "Best WF")
            plt.colorbar(axim, ax=ax[1, 1], orientation="vertical")
            axim1 = plot_funcs["img"](ccd_records.first["_img"], ax[0, 2], "Init CCD")
            axim2 = plot_funcs["img"](max_pid_iter["_img"], ax[1, 2], "Best CCD")
            plt.colorbar(
                axim2, ax=ax[1, 2], fraction=0.046, pad=0.04, orientation="vertical"
            )
            plot_funcs["voltage_comparison"](
                init_v, last_V, ax[0, 3], "Voltage Comparison"
            )
            voltages = np.array(
                wf_records.get_sublist("_v") + ccd_records.get_sublist("_v")[1:]
            )
            plot_funcs["voltage_heatmap"](voltages, ax[1, 3], "Voltage History")

            plt.tight_layout()
            save_dir = gen_date_dir(f"{params.dir}/pipeline")
            saved_file_name = gen_file_path_uuid(save_dir)
            wf_records.save_dataframe(
                saved_file_name.with_suffix(".wfs.pkl"), compression="zip"
            )
            ccd_records.save_dataframe(
                saved_file_name.with_suffix(".ccd.pkl"), compression="zip"
            )
            plt.savefig(saved_file_name.with_suffix(".png"))

            with open(saved_file_name.with_suffix(".json"), "w", encoding="utf8") as f:
                import json

                json.dump(
                    {
                        "dir": params.dir,
                        "load_file": params.load_file,
                        "epochs": params.epochs,
                        "wfs_res": params.wfs_res,
                        "pupil_diameter": params.pupil_diameter,
                        "cam_id": params.cam_id,
                        "exposure_time_ms": params.exposure_time_ms,
                        "cam_size": params.cam_size,
                        "rms_threshold": params.rms_threshold,
                        "debug": debug,
                    },
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

        click.echo(
            f"组合优化完成，最优RMS值: {min_rms:.4f} @ epoch {min_epoch}, 最优PIB值: {max_pib:.4f} @ epoch {max_epoch}"
        )

    finally:
        dm.close()


if __name__ == "__main__":
    setup_coredumpy()
    run()
