from __future__ import annotations

import os
import json
from pathlib import Path

import click
import numpy as np

from ao_shaping.optimizer.combined_optimizer import optimize_pib
from ao_shaping.utils.cli_helpers import get_debug_mode, setup_coredumpy, get_date_dir_name
from ao_shaping.utils.file import gen_file_path_uuid, gen_date_dir, logger
from ao_shaping.utils.display import plot_funcs
from ao_shaping.drivers.dm import create_dm, list_reachable_dm_types

import matplotlib.pyplot as plt


DM_TYPES = list_reachable_dm_types()


def _create_dm(dm_type: str, **kwargs):
    return create_dm(dm_type, **kwargs)


@click.command()
@click.option("-d", "--root_dir", default="data", help="数据保存根目录 (default: data)")
@click.option("-f", "--load_file", default=None, help="加载初始电压文件 (default: None)")
@click.option("--cam_id", default=lambda: os.environ.get("FAR_CAM_ID", "0"), help="远场光斑CCD设备ID (default: Far_CAM_ID/0)")
@click.option("-c", "--center", default="mass", help="场光斑CCD中心位置 (example: 665,403)")
@click.option("-t", "--exposure_time_ms", default=80, help="远场光斑CCD曝光时间 (毫秒) (default: 80)")
@click.option("-e", "--epochs", default=4_000, help="优化迭代次数 (default: 4000)")
@click.option("-r", "--r_bucket", default=0, help="半径桶大小 (default: 0, 环围半径自动调整)")
@click.option("--delta", default=1.0, help="优化步长 (default: 1)")
@click.option("--lr", default=0.0, help="优化学习率 (default: 0.0, 动态学习率衰减)")
@click.option("--shrink_iter", default=0, help="收缩半径桶的迭代间隔 (default: 0, 不收缩)")
@click.option("--shrink_ratio", default=0.9, help="收缩半径桶比例 (default: 0.9)")
@click.option("-s", "--cam_size", default=250, help="相机开窗大小 (default: 250)")
@click.option("-b", "--target_max_brightness", default=40, help="目标最大亮度值 (default: 40)")
@click.option("--show", is_flag=True, help="显示远场光斑CCD图像和优化历史 (default: False)")
@click.option("--dm_type", type=click.Choice(DM_TYPES, case_sensitive=False), default=None,
              help="变形镜类型 (default: auto-detect)")
def run(
    root_dir: str,
    load_file: str | None,
    cam_id: int,
    center: str,
    exposure_time_ms: float,
    epochs: int,
    r_bucket: float,
    delta: float,
    lr: float,
    shrink_iter: int,
    shrink_ratio: float,
    cam_size: int,
    target_max_brightness: float,
    show: bool,
    dm_type: str | None,
):
    """AdaMOD 综合PIB优化器

    使用AdaMOD优化器进行桶内功率(PIB)优化，支持自适应桶半径收缩。
    DEBUG环境变量控制调试模式。
    """
    debug = get_debug_mode()

    if load_file and Path(load_file).exists():
        init_v = np.loadtxt(load_file).tolist()
    else:
        init_v = []

    config = {
        "root_dir": root_dir,
        "load_file": load_file,
        "cam_id": cam_id,
        "center": center,
        "exposure_time_ms": exposure_time_ms,
        "epochs": epochs,
        "r_bucket": r_bucket,
        "delta": delta,
        "lr": lr,
        "shrink_iter": shrink_iter,
        "shrink_ratio": shrink_ratio,
        "cam_size": cam_size,
        "target_max_brightness": target_max_brightness,
        "debug": debug,
        "show": show,
    }
    logger.info(config)

    if dm_type is not None:
        dm_type = dm_type.lower()
        logger.info(f"Using specified DM type: {dm_type}")
    else:
        reachable = DM_TYPES
        if len(reachable) == 1:
            dm_type = reachable[0]
            logger.info(f"Auto-detected reachable DM: {dm_type}")
        elif len(reachable) == 0:
            raise RuntimeError("No DM reachable. Specify --dm_type explicitly or connect a DM.")
        else:
            raise RuntimeError(
                f"Multiple DMs reachable ({', '.join(reachable)}). "
                f"Specify --dm_type explicitly to choose one."
            )

    dm = _create_dm(dm_type, keep_when_exit=True, max_neibor_diff=200)

    res_list = optimize_pib(
        dm=dm,
        center=center,
        epochs=epochs,
        r_bucket=r_bucket,
        delta=delta,
        lr=lr,
        exposure_time_ms=exposure_time_ms,
        shrink_iter=shrink_iter,
        shrink_ratio=shrink_ratio,
        cam_id=cam_id,
        show=show,
        init_v=init_v,
        cam_size=cam_size,
        target_max_brightness=target_max_brightness,
    )

    saved_dir = f"{root_dir}/flatten_voltages/{get_date_dir_name()}"
    res_list.save_best(
        saved_dir, target="_v", process_fn=lambda x: np.around(x).astype(int), fmt="%d"
    )
    best_iter, (max_j_id, max_j) = res_list.get_best_iter()
    last_V = best_iter["_v"]

    if debug:
        save_dir = gen_date_dir(f"{root_dir}/combined")
        saved_file_name = gen_file_path_uuid(save_dir, "pkl")
        res_list.save_dataframe(saved_file_name.with_suffix(".pkl"), compression="zip")

        with saved_file_name.with_suffix(".json").open("w", encoding="utf8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        plot_funcs["img"](res_list.first["_img"], ax[0, 0], f"Init Image, pib={res_list.first['pib']:.3f}")
        axim = plot_funcs["img"](best_iter["_img"], ax[0, 1], f"Best PIB Image, pib={max_j:.3f}")
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
