from __future__ import annotations

import json

import click

from ao_shaping.optimizer.combined_optimizer import optimize_pib
from ao_shaping.utils.io.cli_helpers import (
    resolve_debug,
    setup_coredumpy,
)
from ao_shaping.utils.io.file import gen_file_path_uuid, gen_date_dir, logger
from ao_shaping.drivers.dm._registry import resolve_dm
from ao_shaping.runners.runner_common import (
    CameraParams,
    CombinedAlgorithmParams,
    CombinedRunParams,
    load_init_v,
    plot_pib_debug_grid,
    save_best_voltages,
)
from ao_shaping.utils.cli.params import with_params
from ao_shaping.utils.cli.exposure import exposure_ms_legacy, resolve_exposure
from ao_shaping.utils.io.recorder_keys import (
    RECORD_IMAGE,
    RECORD_VOLTAGES,
)


@click.command()
@click.pass_context
@with_params(CombinedAlgorithmParams, kw_name="algorithm")
@with_params(CombinedRunParams, kw_name="run")
@with_params(CameraParams, kw_name="camera")
def run(ctx: click.Context, algorithm: CombinedAlgorithmParams, run: CombinedRunParams, camera: CameraParams) -> None:
    """AdaMOD 综合PIB优化器

    使用AdaMOD优化器进行桶内功率(PIB)优化，支持自适应桶半径收缩。

    调试模式: ``main.py --debug combined`` / 本命令 ``--debug`` / 环境变量 ``DEBUG=1``
    任一开启即可输出 pkl/json 与汇总图片。
    """
    debug = resolve_debug(ctx, run.debug_flag)
    center = camera.center if camera.center is not None else "mass"
# CLI token -> the legacy float the optimizer has always taken ('auto' == 0).
    exposure_time_ms = exposure_ms_legacy(
        resolve_exposure(
            camera.exposure_time_ms, param_hint="--exposure-time-ms"
        )
    )

    init_v = load_init_v(run.load_file)

    config = {
        "root_dir": run.root_dir,
        "load_file": run.load_file,
        "cam_id": camera.cam_id,
        "center": center,
        "exposure_time_ms": exposure_time_ms,
        "epochs": run.epochs,
        "r_bucket": algorithm.r_bucket,
        "delta": algorithm.delta,
        "lr": algorithm.lr,
        "shrink_iter": algorithm.shrink_iter,
        "shrink_ratio": algorithm.shrink_ratio,
        "cam_size": camera.cam_size,
        "target_max_brightness": algorithm.target_max_brightness,
        "debug": debug,
        "show": run.show,
    }
    logger.info(config)

    dm = resolve_dm(algorithm.dm_type, keep_when_exit=True, max_neibor_diff=200)

    res_list = optimize_pib(
        dm=dm,
        center=center,
        epochs=run.epochs,
        r_bucket=algorithm.r_bucket,
        delta=algorithm.delta,
        lr=algorithm.lr,
        exposure_time_ms=exposure_time_ms,
        cam_type=camera.cam_type,
        shrink_iter=algorithm.shrink_iter,
        shrink_ratio=algorithm.shrink_ratio,
        cam_id=camera.cam_id,
        show=run.show,
        init_v=init_v,
        cam_size=camera.cam_size,
        target_max_brightness=algorithm.target_max_brightness,
    )

    save_best_voltages(res_list, run.root_dir)
    best_iter, (max_j_id, max_j) = res_list.get_best_iter()
    last_V = best_iter[RECORD_VOLTAGES]

    if debug:
        save_dir = gen_date_dir(f"{run.root_dir}/combined")
        saved_file_name = gen_file_path_uuid(save_dir, "pkl")
        res_list.save_dataframe(saved_file_name.with_suffix(".pkl"), compression="zip")

        with saved_file_name.with_suffix(".json").open("w", encoding="utf8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        plot_pib_debug_grid(
            init_img=res_list.first[RECORD_IMAGE],
            best_img=best_iter[RECORD_IMAGE],
            history_values=res_list.dataframe["pib"],
            voltages=last_V,
            init_title=f"Init Image, pib={res_list.first['pib']:.3f}",
            best_title=f"Best PIB Image, pib={max_j:.3f}",
            history_title="PIB History",
            save_path=saved_file_name.with_suffix(".png"),
        )

    objective_name = "PIB"
    click.echo(
        f"Combined PIB optimization complete, best {objective_name}: {max_j:.4f} @ epoch {max_j_id}"
    )


if __name__ == "__main__":
    setup_coredumpy()
    run()
