"""SLM Zernike PIB 优化器 Runner

通过 SLM 加载 Zernike 相位, 以 CCD 测量的 PIB (桶内功率) 为目标进行优化 (SPGD), 实现远场光斑优化。
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from ao_shaping.config import DEVICES
from ao_shaping.optimizer.wfless.slm_zernike_pib import optimize_slm_zernike_pib
from ao_shaping.utils.cli_helpers import (
    get_date_dir_name,
    get_debug_mode,
    parse_tuple,
    setup_coredumpy,
)
from ao_shaping.utils.display import plot_funcs
from ao_shaping.utils.file import gen_date_dir, gen_file_path_uuid, logger
from ao_shaping.utils.zernike_calc import calc_n_zernike_terms


@click.command()
@click.option("-d", "--root_dir", default="data", help="数据保存根目录 (default: data)")
@click.option(
    "-f",
    "--load_file",
    default="",
    help="初始 Zernike 系数文件路径 (每行一个, Noll 顺序) (default: 空, 从零开始)",
)
@click.option(
    "--cam_id",
    type=int,
    default=lambda: DEVICES.far_cam_id,
    help="远场光斑CCD设备ID (default: Far_Cam_ID/0)",
)
@click.option(
    "-c",
    "--center",
    callback=parse_tuple,
    default="shape",
    help="远场光斑CCD中心位置 (example: 665,403)",
)
@click.option(
    "-t",
    "--exposure_time_ms",
    default=0.0,
    help="远场光斑CCD曝光时间 (毫秒) (default: auto)",
)
@click.option("-e", "--epochs", default=2000, help="优化迭代次数 (default: 2000)")
@click.option(
    "-r",
    "--r_bucket",
    default=0,
    help="半径桶大小 (default: 0, 环围半径自动调整)",
)
@click.option("--delta", default=0.1, help="优化步长 (default: 0.1)")
@click.option(
    "--lr",
    default=0.0,
    help="优化学习率 (default: 0.0, 表示基于环围半径动态学习率衰减)",
)
@click.option("-n", "--n_max", default=4, help="Zernike 最大径向阶数 (default: 4)")
@click.option("--slm_number", default=1, help="SLM 设备编号 (default: 1)")
@click.option("--slm_wavelength", default=1064, help="SLM 工作波长 nm (default: 1064)")
@click.option(
    "--zernike_radius",
    type=float,
    default=300.0,
    show_default=True,
    help="Zernike 相位孔径半径 (px)。必须匹配光束在 SLM 上的实际半径 (本台 ~300);"
    "取默认值(短边半长 600)会让相位在光束范围内近乎常数 → 矫正静默失效。",
)
@click.option(
    "--shift_x", type=int, default=0, show_default=True, help="SLM 相位 X 平移 (px)"
)
@click.option(
    "--shift_y", type=int, default=0, show_default=True, help="SLM 相位 Y 平移 (px)"
)
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
    default=0,
    help="优化迭代次数后收缩半径桶和步长 (default: 0, 不收缩)",
)
@click.option("--shrink_ratio", default=0.9, help="收缩半径桶和步长比例 (default: 0.9)")
@click.option("-s", "--cam_size", default=250, help="相机开窗大小 (default: 250)")
@click.option(
    "-b",
    "--target_max_brightness",
    default=40,
    help="目标最大亮度值 (default: 40), 若为0则不自动调整曝光时间",
)
@click.option(
    "-o",
    "--objective",
    type=click.Choice(["pib", "radiu", "avg_radiu"]),
    default="pib",
    show_default=True,
    help="优化目标函数: pib(最大化PIB), radiu(最小化半径), avg_radiu(最大化平均半径)",
)
@click.option(
    "--show",
    is_flag=True,
    help="启用 pygame 实时可视化显示 (相位、CCD图像、Zernike系数、PIB曲线) (default: False)",
)
@click.option("--seed", type=int, default=None, help="随机种子 (default: None)")
@click.option(
    "--disturb-scale",
    type=float,
    default=0.0,
    help="扰动幅度 (弧度/系数, default: 0.0, >0 时在初始系数上叠加随机扰动)",
)
@click.option(
    "--disturb-seed",
    type=int,
    default=None,
    help="扰动随机种子 (与 --seed 独立)",
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
    n_max,
    slm_number,
    slm_wavelength,
    zernike_radius,
    shift_x,
    shift_y,
    optimizer_type,
    shrink_iter,
    shrink_ratio,
    cam_size,
    target_max_brightness,
    objective,
    show,
    seed,
    disturb_scale,
    disturb_seed,
):
    """SLM Zernike PIB 优化器

    通过 SLM 加载 Zernike 相位, 以 CCD 测量的 PIB (桶内功率) 为目标进行优化 (SPGD)。
    使用 --show 启用 pygame 实时可视化显示 (相位图案、远场光斑、Zernike系数柱状图、PIB收敛曲线)。
    DEBUG 环境变量控制调试模式。
    """
    debug = get_debug_mode()

    # 加载初始 Zernike 系数
    if load_file and Path(load_file).exists():
        loaded_coeffs = np.loadtxt(load_file).tolist()
        logger.info(
            f"Loaded Zernike coefficients from {load_file}: {len(loaded_coeffs)} terms"
        )
    elif load_file:
        logger.warning(f"load_file {load_file} not exists, starting from zeros")
        loaded_coeffs = []
    else:
        loaded_coeffs = []

    # 扰动注入逻辑
    nk = calc_n_zernike_terms(n_max)
    if disturb_scale > 0:
        rng = np.random.default_rng(disturb_seed)
        disturb = rng.uniform(-disturb_scale, disturb_scale, nk)
        base = np.asarray(loaded_coeffs, dtype=np.float64)
        if base.size == 0:
            init_c = disturb
        elif base.size == nk:
            init_c = base + disturb
        else:
            pad = np.zeros(nk, dtype=np.float64)
            pad[: min(base.size, nk)] = base[:nk]
            init_c = pad + disturb
        init_c = np.asarray(init_c, dtype=np.float64).tolist()
        logger.info(
            "Injecting random Zernike disturbance: scale={}, seed={}, nk={}",
            disturb_scale,
            disturb_seed,
            nk,
        )
    else:
        init_c = loaded_coeffs

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
        "n_max": n_max,
        "slm_number": slm_number,
        "slm_wavelength": slm_wavelength,
        "zernike_radius": zernike_radius,
        "shift_x": shift_x,
        "shift_y": shift_y,
        "optimizer_type": optimizer_type,
        "shrink_iter": shrink_iter,
        "shrink_ratio": shrink_ratio,
        "cam_size": cam_size,
        "objective": objective,
        "debug": debug,
        "show": show,
        "seed": seed,
        "disturb_scale": disturb_scale,
        "disturb_seed": disturb_seed,
    }
    logger.info(config)

    saved_dir = f"{root_dir}/slm_zernike_pib/{get_date_dir_name()}"

    res_list = optimize_slm_zernike_pib(
        center=center,
        epochs=epochs,
        n_max=n_max,
        r_bucket=r_bucket,
        delta=delta,
        lr=lr,
        exposure_time_ms=exposure_time_ms,
        shrink_iter=shrink_iter,
        shrink_ratio=shrink_ratio,
        cam_id=cam_id,
        show=show,
        display=show,
        init_c=init_c,
        cam_size=cam_size,
        target_max_brightness=target_max_brightness,
        slm_number=slm_number,
        slm_wavelength=slm_wavelength,
        zernike_radius=zernike_radius,
        shift_x=shift_x,
        shift_y=shift_y,
        optimizer_type=optimizer_type,
        random_seed=seed,
        objective=objective,
    )
    res_df = res_list.dataframe

    res_list.save_best(
        saved_dir, target="_c", process_fn=lambda x: np.round(x, 6), fmt="%.6f"
    )

    # 保存扰动系数
    if disturb_scale > 0:
        disturb_file = Path(saved_dir) / "disturb_c.txt"
        np.savetxt(disturb_file, disturb, fmt="%.6f")
        logger.info(f"Disturbance coefficients saved to {disturb_file}")

    best_iter, (max_j_id, max_j) = res_list.get_best_iter()

    if debug:
        save_dir = gen_date_dir(f"{root_dir}/slm-zernike-pib")
        saved_file_name = gen_file_path_uuid(save_dir, "pkl")

        res_df.to_pickle(saved_file_name, compression="zip")
        with saved_file_name.with_suffix(".json").open("w", encoding="utf8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        plt.switch_backend("Agg")

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        plot_funcs["img"](
            res_df.iloc[0]["_img"],
            ax[0, 0],
            f"Init Image, pib={res_df.iloc[0]['pib']:.3f}",
        )
        axim = plot_funcs["img"](
            res_df.iloc[max_j_id]["_img"],
            ax[0, 1],
            f"Best PIB Image, pib={max_j:.3f}",
        )
        fig.colorbar(axim, ax=[ax[0, 0], ax[0, 1]], orientation="horizontal")
        plot_funcs["pib_history"](res_df["pib"], ax[1, 0])

        # Zernike 系数柱状图
        best_c = best_iter["_c"]
        ax[1, 1].bar(range(len(best_c)), best_c)
        ax[1, 1].set_xlabel("Zernike Index (Noll)")
        ax[1, 1].set_ylabel("Coefficient")
        ax[1, 1].set_title("Best Zernike Coeffs")
        ax[1, 1].set_xticks(range(len(best_c)))

        plt.savefig(saved_file_name.with_suffix(".png"))
        plt.close()
        logger.info(f"Debug plot saved to {saved_file_name.with_suffix('.png')}")

    objective_names = {
        "pib": "PIB",
        "radiu": "半径",
        "avg_radiu": "平均半径",
    }
    objective_name = objective_names.get(objective, objective)

    click.echo(
        f"SLM Zernike PIB 优化完成，最优{objective_name}值: {max_j:.4f} @ epoch {max_j_id}"
    )


if __name__ == "__main__":
    setup_coredumpy()
    run()
