import os
import json
import re
from datetime import datetime
from pathlib import Path
import sys

import click
import coredumpy

import numpy as np
import matplotlib.pyplot as plt

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ao_shaping.optimizer.wfless.pib import optimize_pib
from ao_shaping.algorithm.target_func import ImageTargetFunc
from ao_shaping.utils.file import (
    gen_file_path_uuid,
    gen_date_dir,
    get_init_V_by_rms,
    logger,
)
from ao_shaping.utils.display import plot_funcs


def parse_tuple(ctx, param, value):
    """解析元组格式的参数，支持 'x,y' 或 '(x,y)' 格式"""
    if value is None:
        return None
    if value.lower() in ["mass", "max", "shape"]:
        return value.lower()
    # 移除空格和括号
    s_clean = re.sub(r"[()\s]", "", str(value))
    try:
        parts = s_clean.split(",")
        if len(parts) != 2:
            raise ValueError("Must have exactly two integers")
        x, y = map(int, parts)
        return (x, y)
    except Exception as e:
        raise click.BadParameter(
            f"Invalid center format: {value}. Expected formats: 'x,y' or '(x,y)'"
        )


@click.command()
@click.option("-d", "--root_dir", default="data", help="数据保存根目录 (default: data)")
@click.option(
    "-f",
    "--load_file",
    default="rms",
    help="加载优化结果文件 (default: None), 若为'rms',则使用RMS优化结果初始化",
)
@click.option(
    "--cam_id",
    default=lambda: os.environ.get("FAR_CAM_ID", "0"),
    help="远场光斑CCD设备ID (default: Far_Cam_ID/0)",
)
@click.option(
    "-c",
    "--center",
    callback=parse_tuple,
    default="mass",
    help="场光斑CCD中心位置 (example: 665,403)",
)
@click.option(
    "-t",
    "--exposure_time_ms",
    default=60,
    help="远场光斑CCD曝光时间 (毫秒) (default: 60)",
)
@click.option("-e", "--epochs", default=4_000, help="优化迭代次数 (default: 4000)")
@click.option(
    "-r",
    "--r_bucket",
    default=0,
    help="半径桶大小 (default: 0,环围半径)。若设置为0,则根据功率半径自动调整。",
)
@click.option("--delta", default=2.0, help="优化步长 (default: 2)")
@click.option(
    "--lr", default=0.0, help="优化学习率 (default: 0.0,表示基于环围半径动态学习率衰减)"
)
@click.option("--weight_decay", default=0.0, help="权重衰减 (default: 0.0)")
@click.option(
    "--optimizer_type",
    type=click.Choice(
        ["adam", "adamw", "adamod", "sgd", "muno", "munow"], case_sensitive=False
    ),
    default="adamod",
    show_default=True,
    help="梯度阶段使用的优化器类型",
)
@click.option(
    "--shrink_iter",
    default=200,
    help="优化迭代次数后收缩半径桶和步长 (default: 200)。若设置为0，则不进行收缩。",
)
@click.option("--shrink_ratio", default=0.8, help="收缩半径桶和步长比例 (default: 0.8)")
@click.option(
    "--enable_adaptive_search", is_flag=True, help="启用局部最优后的自适应邻域搜索"
)
@click.option(
    "--search_interval", default=120, show_default=True, help="邻域搜索触发间隔"
)
@click.option(
    "--search_warmup", default=200, show_default=True, help="邻域搜索启动前的最小迭代数"
)
@click.option(
    "--search_patience",
    default=100,
    show_default=True,
    help="最佳 PIB 无提升时触发搜索的等待轮数",
)
@click.option(
    "--search_samples",
    default=8,
    show_default=True,
    help="每次邻域搜索评估的候选解数量",
)
@click.option(
    "--search_radius",
    default=None,
    type=float,
    help="邻域搜索初始半径，默认跟随 delta 自适应",
)
@click.option(
    "--tabu_memory_size", default=128, show_default=True, help="禁忌记忆表容量"
)
@click.option("-s", "--cam_size", default=200, help="相机开窗大小 (default: 200*200)")
@click.option(
    "-b",
    "--target_max_brightness",
    default=90,
    help="目标最大亮度值 (default: 90), 若为0则不自动调整曝光时间",
)
@click.option(
    "-o",
    "--objective",
    type=click.Choice(["pib", "radiu", "avg_radiu"]),
    default="pib",
    show_default=True,
    help="优化目标函数: pib(最大化PIB), radiu(最小化半径), avg_radiu(最大化平均半径)",
)
@click.option("--debug", is_flag=True, help="是否开启调试模式 (default: False)")
@click.option(
    "--show", is_flag=True, help="显示远场光斑CCD图像和优化历史 (default: False)"
)
def run(
    root_dir,
    load_file,
    cam_id,
    center,
    exposure_time_ms,
    epochs,
    r_bucket,
    delta,
    lr,
    weight_decay,
    optimizer_type,
    shrink_iter,
    shrink_ratio,
    enable_adaptive_search,
    search_interval,
    search_warmup,
    search_patience,
    search_samples,
    search_radius,
    tabu_memory_size,
    cam_size,
    target_max_brightness,
    objective,
    debug,
    show,
):
    """轴向光束优化器"""

    # 处理初始电压
    if load_file.lower() == "rms":
        init_v = get_init_V_by_rms()
    elif Path(load_file).exists():
        last_v = np.loadtxt(load_file)
        init_v = last_v.tolist()
    else:
        logger.warning(f"load_file {load_file} not exists")
        init_v = []

    config = {
        "root_dir": root_dir,
        "load_file": load_file,
        "cam_id": cam_id,
        "center": center,
        "exposure_time_ms": exposure_time_ms,
        "target_max_brightness": target_max_brightness,
        "epochs": epochs,
        "r_bucket": r_bucket,
        "delta": delta,
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer_type": optimizer_type,
        "shrink_iter": shrink_iter,
        "shrink_ratio": shrink_ratio,
        "enable_adaptive_search": enable_adaptive_search,
        "search_interval": search_interval,
        "search_warmup": search_warmup,
        "search_patience": search_patience,
        "search_samples": search_samples,
        "search_radius": search_radius,
        "tabu_memory_size": tabu_memory_size,
        "cam_size": cam_size,
        "objective": objective,
        "debug": debug,
        "show": show,
    }
    logger.info(config)

    dm_unit_mask = np.ones(64, dtype=bool)
    dm_unit_mask[0] = False
    # dm_unit_mask[38:] = False
    res_list = optimize_pib(
        center=center,
        r_bucket=r_bucket,
        epochs=epochs,
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
        dm_unit_mask=dm_unit_mask,
        dm_neibor_diff=300,
        optimizer_type=optimizer_type,
        enable_adaptive_search=enable_adaptive_search,
        search_interval=search_interval,
        search_warmup=search_warmup,
        search_patience=search_patience,
        search_samples=search_samples,
        search_radius=search_radius,
        tabu_memory_size=tabu_memory_size,
        weight_decay=weight_decay,
        objective=objective,
    )
    # 保存结果
    res_df = res_list.dataframe

    saved_dir = f"{root_dir}/flatten_voltages/{datetime.now().strftime('%Y%m%d')}"
    res_list.save_best(
        saved_dir, target="_v", process_fn=lambda x: np.around(x).astype(int), fmt="%d"
    )
    best_iter, (max_j_id, max_j) = res_list.get_best_iter()
    if debug:
        save_dir = gen_date_dir(f"{root_dir}/wf-less")
        saved_file_name = gen_file_path_uuid(save_dir, "pkl")

        # Register second moment radius as a postprocess feature
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

        # Re-fetch dataframe with postprocess columns applied
        res_df = res_list.dataframe
        res_df.to_pickle(saved_file_name, compression="zip")
        with saved_file_name.with_suffix(".json").open("w", encoding="utf8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        # Find the image with minimum second moment radius (best focus)
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

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        # init image
        plot_funcs["img"](
            res_df.iloc[0]["_img"],
            ax[0, 0],
            f"Init Image, pib={res_df.iloc[0]['pib']:.3f}",
        )
        # best PIB image
        axim = plot_funcs["img"](
            res_df.iloc[max_j_id]["_img"], ax[0, 1], f"Best PIB Image, pib={max_j:.3f}"
        )
        cbar = fig.colorbar(axim, ax=[ax[0, 0], ax[0, 1]], orientation="horizontal")
        # pib history
        plot_funcs["pib_history"](res_df["pib"], ax[1, 0])
        # best voltages plot bar
        plot_funcs["voltages"](best_iter["_v"], ax[1, 1], "Best Voltages")

        plt.savefig(saved_file_name.with_suffix(".png"))
        plt.close()

        # Save additional plot: second moment radius vs epoch
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

        # Save the best image (minimum second moment radius) as a separate file
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

            # Save the best focus voltages
            best_focus_v_file = saved_file_name.with_name(
                saved_file_name.stem + "_best_focus_v.txt"
            )
            np.savetxt(best_focus_v_file, np.around(best_v).astype(int), fmt="%d")
            logger.info(f"Best focus voltages saved to {best_focus_v_file}")

    # 优化目标名称映射
    objective_names = {
        "pib": "PIB",
        "radiu": "半径",
        "avg_radiu": "平均半径",
    }
    objective_name = objective_names.get(objective, objective)

    click.echo(
        f"轴向光束优化完成，最优{objective_name}值: {max_j:.4f} @ epoch {max_j_id}"
    )


if __name__ == "__main__":
    try:
        coredumpy.patch_except(directory="logs/debug/error")
    except Exception:
        logger.error("coredumpy初始化失败")

    run()
