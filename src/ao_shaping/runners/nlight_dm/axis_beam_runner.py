import json
from pathlib import Path
from typing import cast

import click
import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.optimizer.wfless.pib import optimize_pib
from ao_shaping.algorithm.goal_functions.target_func import ImageTargetFunc
from ao_shaping.utils.io.file import (
    gen_file_path_uuid,
    gen_date_dir,
    get_init_V_by_rms,
    logger,
)
from ao_shaping.display.image_display import plot_funcs
from ao_shaping.utils.io.cli_helpers import (
    setup_coredumpy,
    get_date_dir_name,
)
from ao_shaping import config as ao_config
from ao_shaping.utils.io.cli_helpers import resolve_debug
from ao_shaping.drivers.dm._registry import resolve_dm
from ao_shaping.runners.runner_common import (
    DmDeviceParams,
    PibAlgorithmParams,
    PibOpticalParams,
    PibRunParams,
    with_params,
)


@click.command()
@click.pass_context
@with_params(DmDeviceParams, kw_name="device")
@with_params(PibRunParams, kw_name="run")
@with_params(PibOpticalParams, kw_name="optical")
@with_params(PibAlgorithmParams, kw_name="algorithm")
def run(
    ctx: click.Context,
    device: DmDeviceParams,
    run: PibRunParams,
    optical: PibOpticalParams,
    algorithm: PibAlgorithmParams,
) -> None:
    """轴向光束优化器

    调试模式: ``main.py --debug pib`` / 本命令 ``--debug`` / 环境变量 ``DEBUG=1``
    任一开启即可输出 pkl/json 与汇总图片。
    """
    debug = resolve_debug(ctx, run.debug_flag)

    if run.load_file.lower() == "rms":
        init_v = get_init_V_by_rms()
    elif Path(run.load_file).exists():
        last_v = np.loadtxt(run.load_file)
        init_v = last_v.tolist()
    else:
        logger.warning(f"load_file {run.load_file} not exists")
        init_v = []

    config = {
        "root_dir": run.root_dir,
        "load_file": run.load_file,
        "cam_id": optical.cam_id,
        "center": optical.center,
        "exposure_time_ms": optical.exposure_time_ms,
        "target_max_brightness": optical.target_max_brightness,
        "epochs": run.epochs,
        "r_bucket": optical.r_bucket,
        "delta": algorithm.delta,
        "lr": algorithm.lr,
        "weight_decay": algorithm.weight_decay,
        "optimizer_type": algorithm.optimizer_type,
        "shrink_iter": algorithm.shrink_iter,
        "shrink_ratio": algorithm.shrink_ratio,
        "enable_adaptive_search": algorithm.enable_adaptive_search,
        "search_interval": algorithm.search_interval,
        "search_warmup": algorithm.search_warmup,
        "search_patience": algorithm.search_patience,
        "search_samples": algorithm.search_samples,
        "search_radius": algorithm.search_radius,
        "tabu_memory_size": algorithm.tabu_memory_size,
        "cam_size": optical.cam_size,
        "objective": algorithm.objective,
        "debug": debug,
        "show": run.show,
    }
    logger.info(config)

    dm = resolve_dm(device.dm_type, dm_neibor_diff=300)

    dm_unit_mask = np.ones(ao_config.DM_N_ACTUATORS, dtype=bool)
    dm_unit_mask[0] = False
    res_list = optimize_pib(
        dm=dm,
        center=optical.center,
        r_bucket=optical.r_bucket,
        epochs=run.epochs,
        delta=algorithm.delta,
        lr=algorithm.lr,
        exposure_time_ms=optical.exposure_time_ms,
        cam_type=optical.cam_type,
        shrink_iter=algorithm.shrink_iter,
        shrink_ratio=algorithm.shrink_ratio,
        cam_id=cast(int, optical.cam_id),
        show=run.show,
        init_v=init_v,
        cam_size=optical.cam_size,
        target_max_brightness=optical.target_max_brightness,
        dm_unit_mask=dm_unit_mask,
        dm_neibor_diff=300,
        optimizer_type=algorithm.optimizer_type,
        enable_adaptive_search=algorithm.enable_adaptive_search,
        search_interval=algorithm.search_interval,
        search_warmup=algorithm.search_warmup,
        search_patience=algorithm.search_patience,
        search_samples=algorithm.search_samples,
        search_radius=algorithm.search_radius,
        tabu_memory_size=algorithm.tabu_memory_size,
        weight_decay=algorithm.weight_decay,
        objective=algorithm.objective,
    )
    res_df = res_list.dataframe

    saved_dir = f"{run.root_dir}/flatten_voltages/{get_date_dir_name()}"
    res_list.save_best(
        saved_dir, target="_v", process_fn=lambda x: np.around(x).astype(int), fmt="%d"
    )
    best_iter, (max_j_id, max_j) = res_list.get_best_iter()
    if debug:
        save_dir = gen_date_dir(f"{run.root_dir}/wf-less")
        saved_file_name = gen_file_path_uuid(save_dir, "pkl")

        def _calc_second_moment_radius(row: dict) -> float | None:
            img = row.get("_img")
            if img is None:
                return None
            target = ImageTargetFunc.build_from_init_image(img)
            return target.second_moment_radius(img)

        res_list.postprocess_feature(
            "second_moment_radius",
            _calc_second_moment_radius,
            column="_second_moment_radius",
        )

        res_df = res_list.dataframe
        res_df.to_pickle(saved_file_name, compression="zip")
        with saved_file_name.with_suffix(".json").open("w", encoding="utf8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        valid_radii = res_df.dropna(subset=["_second_moment_radius"])
        if not valid_radii.empty:
            min_radius_idx = valid_radii["_second_moment_radius"].idxmin()
            min_radius = valid_radii.loc[min_radius_idx, "_second_moment_radius"]
            logger.info(
                f"Best second moment radius: {min_radius:.3f} pixels @ epoch {min_radius_idx}"
            )
        else:
            min_radius_idx = max_j_id
            min_radius = None
            logger.warning(
                "All second moment calculations failed, falling back to PIB best"
            )

        # The metric column is the objective itself (pib/radiu/avg_radiu); the old
        # hardcoded "pib" column raised KeyError for the other objectives.
        first_val = float(res_df.iloc[0][algorithm.objective])
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        plot_funcs["img"](
            res_df.iloc[0]["_img"],
            ax[0, 0],
            f"Init Image, {algorithm.objective}={first_val:.3f}",
        )
        axim = plot_funcs["img"](
            res_df.iloc[max_j_id]["_img"],
            ax[0, 1],
            f"Best {algorithm.objective} Image, {algorithm.objective}={max_j:.3f}",
        )
        fig.colorbar(axim, ax=[ax[0, 0], ax[0, 1]], orientation="horizontal")
        plot_funcs["pib_history"](
            res_df[algorithm.objective], ax[1, 0], title=f"{algorithm.objective} History"
        )
        plot_funcs["voltages"](best_iter["_v"], ax[1, 1], "Best Voltages")

        plt.savefig(saved_file_name.with_suffix(".png"))
        plt.close()

        if not valid_radii.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(
                valid_radii.index,
                valid_radii["_second_moment_radius"],
                marker=".",
                markersize=2,
                alpha=0.5,
            )
            ax2.axvline(
                x=min_radius_idx,
                color="r",
                linestyle="--",
                label=f"Min radius @ {min_radius_idx}",
            )
            ax2.axhline(
                y=min_radius,
                color="g",
                linestyle="--",
                label=f"Min radius = {min_radius:.3f}",
            )
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Second Moment Radius (pixels)")
            ax2.set_title("Second Moment Radius vs Epoch")
            ax2.legend()
            ax2.grid(alpha=0.3)
            plt.savefig(
                saved_file_name.with_name(
                    saved_file_name.stem + "_second_moment_radius.png"
                )
            )
            plt.close()

        if not valid_radii.empty:
            best_img = res_df.loc[min_radius_idx, "_img"]
            best_v = res_df.loc[min_radius_idx, "_v"]
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            plot_funcs["img"](
                best_img,
                ax3,
                f"Best Focus (Second Moment Radius={min_radius:.3f}px) @ epoch {min_radius_idx}",
            )
            plt.savefig(
                saved_file_name.with_name(saved_file_name.stem + "_best_focus.png")
            )
            plt.close()

            best_focus_v_file = saved_file_name.with_name(
                saved_file_name.stem + "_best_focus_v.txt"
            )
            np.savetxt(best_focus_v_file, np.around(best_v).astype(int), fmt="%d")
            logger.info(f"Best focus voltages saved to {best_focus_v_file}")

    objective_names = {
        "pib": "PIB",
        "radiu": "半径",
        "avg_radiu": "平均半径",
    }
    objective_name = objective_names.get(algorithm.objective, algorithm.objective)

    click.echo(
        f"轴向光束优化完成，最优{objective_name}值: {max_j:.4f} @ epoch {max_j_id}"
    )


if __name__ == "__main__":
    setup_coredumpy()
    run()
