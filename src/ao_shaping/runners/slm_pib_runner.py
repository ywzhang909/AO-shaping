"""SLM Zernike PIB 优化器 Runner

通过 SLM 加载 Zernike 相位, 以 CCD 测量的 PIB (桶内功率) 为目标进行优化 (SPGD), 实现远场光斑优化。
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.drivers.ccd.common import list_camera_types
from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    ALGORITHM_CHOICES,
    TARGET_SHAPE_CHOICES,
    optimize_slm_zernike_pib,
)
from ao_shaping.utils.io.file import gen_file_path_uuid, gen_date_dir, logger
from ao_shaping.utils.image.display import plot_funcs
from ao_shaping.utils.io.cli_helpers import (
    parse_tuple,
    setup_coredumpy,
    get_date_dir_name,
    resolve_debug,
)
from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms
from ao_shaping.config import DEVICES


def _save_debug_artifacts(
    res_list, objective: str, config: dict, root_dir: str
) -> Path:
    """Dump the full recorder + a 4-panel summary figure (debug mode).

    Mirrors ``axis_beam_runner``'s debug output: raw pickle, run-config JSON and a
    summary PNG (init image, best image, objective history, best Zernike
    coefficients). The metric column is ``objective`` itself (``pib`` / ``radiu``
    / ``avg_radiu``) — the old hardcoded ``"pib"`` column raised ``KeyError`` for
    the other objectives.

    Returns:
        Path of the written summary PNG.
    """
    save_dir = gen_date_dir(f"{root_dir}/slm-zernike-pib")
    saved_file_name = gen_file_path_uuid(save_dir, "pkl")
    res_df = res_list.dataframe

    res_df.to_pickle(saved_file_name, compression="zip")
    with saved_file_name.with_suffix(".json").open("w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    plt.switch_backend("Agg")
    best_iter, (best_id, best_val) = res_list.get_best_iter()
    first_val = float(res_df.iloc[0][objective])

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    plot_funcs["img"](
        res_df.iloc[0]["_img"], ax[0, 0], f"Init Image, {objective}={first_val:.3f}"
    )
    axim = plot_funcs["img"](
        res_df.iloc[best_id]["_img"],
        ax[0, 1],
        f"Best {objective} Image, {objective}={best_val:.3f}",
    )
    fig.colorbar(axim, ax=[ax[0, 0], ax[0, 1]], orientation="horizontal")
    plot_funcs["pib_history"](res_df[objective], ax[1, 0], title=f"{objective} History")

    best_c = np.asarray(best_iter["_c"])
    ax[1, 1].bar(range(len(best_c)), best_c)
    ax[1, 1].set_xlabel("Zernike Index (Noll)")
    ax[1, 1].set_ylabel("Coefficient")
    ax[1, 1].set_title("Best Zernike Coeffs")
    ax[1, 1].set_xticks(range(len(best_c)))

    png_file = saved_file_name.with_suffix(".png")
    plt.savefig(png_file)
    plt.close(fig)
    logger.info(
        "Debug artifacts saved: {} (+ .pkl/.json in {})",
        png_file,
        saved_file_name.parent,
    )
    return png_file


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
    "--cam_type",
    type=click.Choice(list_camera_types(), case_sensitive=False),
    default="daheng",
    show_default=True,
    help="相机后端类型 (经 create_camera 注册表解析; daheng/miicam 等)",
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
    help="远场光斑CCD曝光时间 (毫秒); 0 = 按 -b 目标亮度自动曝光 (default: 0.0)",
)
@click.option("-e", "--epochs", default=2000, help="优化迭代次数 (default: 2000)")
@click.option(
    "-r",
    "--r_bucket",
    default=0,
    help="半径桶大小 (default: 0, 环围半径自动调整)",
)
@click.option(
    "--delta",
    default=0.2,
    help="优化步长 (default: 0.2; 0.1 时扰动量 SNR≈0.7 淹没在噪声里, 见 measure_shape_sensitivity.py)",
)
@click.option(
    "--max-energy-loss",
    default=0.6,
    show_default=True,
    help="安全保护: ROI 内能量损失超过该比例 (0~1) 即放弃该评估; 0 = 关闭",
)
@click.option("--w-uniformity", default=2.0, show_default=True, help="整形: 不均匀度权重")
@click.option("--w-peak", default=0.5, show_default=True, help="整形: 峰值因子权重")
@click.option(
    "--w-displacement",
    default=0.0,
    show_default=True,
    help="整形: 质心偏移权重 (ROI 固定时为 0)",
)
@click.option(
    "--log-uniformity/--no-log-uniformity",
    default=False,
    show_default=True,
    help="整形: 用 log1p(u) 替代 u/(1+u) 以放大近收敛梯度",
)
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
    help="梯度阶段使用的优化器类型 (仅 spgd 生效)",
)
@click.option(
    "--algorithm",
    type=click.Choice(list(ALGORITHM_CHOICES), case_sensitive=False),
    default="spgd",
    show_default=True,
    help="搜索算法: spgd (梯度) 或启发式 (ga/pso/sa/hc/rs/cem/de)",
)
@click.option(
    "--pop_size",
    type=int,
    default=None,
    help="种群规模 (ga/pso/cem/de 使用; 默认取算法默认值)",
)
@click.option(
    "--shrink_iter",
    default=0,
    help="优化迭代次数后收缩半径桶和步长 (default: 0, 不收缩)",
)
@click.option("--shrink_ratio", default=0.9, help="收缩半径桶和步长比例 (default: 0.9)")
@click.option("-s", "--cam_size", default=320, help="相机开窗大小 (default: 320; 须 > 目标长边)")
@click.option(
    "-b",
    "--target_max_brightness",
    default=40,
    help="自动曝光目标最大亮度 (0-255, default: 40); 0 = 不自动调整曝光 "
    "(与 -t 0 一起用时保持当前曝光)",
)
@click.option(
    "-o",
    "--objective",
    type=click.Choice(["shape", "roi_pib", "pib", "radiu", "avg_radiu"]),
    default="shape",
    show_default=True,
    help="优化目标: shape(整形为长方形, 默认), roi_pib(最大化目标ROI内亮度), "
    "pib(最大化桶内功率), radiu(最小化半径), avg_radiu(最大化平均半径)",
)
@click.option(
    "--shape-schedule/--no-shape-schedule",
    default=False,
    show_default=True,
    help="整形 coarse→fine 调度 (实验性: 跨 stage 分数不可比, 会让 gain 归零, 默认关闭)",
)
@click.option(
    "--target-shape",
    type=click.Choice(TARGET_SHAPE_CHOICES, case_sensitive=False),
    default=None,
    help="动态ROI目标形状 (default: rectangle — objective=shape 时的默认)",
)
@click.option(
    "--target-size",
    type=float,
    default=None,
    help="目标ROI尺寸(px)：矩形短边/圆直径/方形边长；默认按初始99%能量半径推导",
)
@click.option(
    "--target-aspect-ratio",
    type=float,
    default=4.0 / 3.0,
    show_default=True,
    help="矩形目标ROI的宽:高 (default: 4:3)",
)
@click.option(
    "--target-center-smooth",
    type=int,
    default=3,
    show_default=True,
    help="动态ROI质心滑动平均帧数 (default: 3)",
)
@click.option(
    "--show", is_flag=True, help="显示远场光斑CCD图像和优化历史 (default: False)"
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
@click.option(
    "--debug",
    "debug_flag",
    is_flag=True,
    default=None,
    help="启用调试模式: 保存 pkl/json 与汇总图 (初始/最优光斑, 目标曲线, 最优 Zernike 系数)",
)
@click.pass_context
def run(
    ctx,
    root_dir,
    load_file,
    cam_id,
    cam_type,
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
    algorithm,
    pop_size,
    shrink_iter,
    shrink_ratio,
    cam_size,
    shape_schedule,
    max_energy_loss,
    w_uniformity,
    w_peak,
    w_displacement,
    log_uniformity,
    target_max_brightness,
    objective,
    target_shape,
    target_size,
    target_aspect_ratio,
    target_center_smooth,
    show,
    seed,
    disturb_scale,
    disturb_seed,
    debug_flag,
):
    """SLM Zernike PIB 优化器

    通过 SLM 加载 Zernike 相位, 以 CCD 测量的 PIB (桶内功率) 或半径为目标进行优化。
    搜索算法可选 spgd (梯度) 或启发式 (ga/pso/sa/hc/rs/cem/de)。

    调试模式: ``main.py --debug slm-pib`` / 本命令 ``--debug`` / 环境变量 ``DEBUG=1``
    任一开启即可输出 pkl/json 与汇总图片。
    """
    debug = resolve_debug(ctx, debug_flag)

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
        "cam_type": cam_type,
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
        "algorithm": algorithm,
        "pop_size": pop_size,
        "shrink_iter": shrink_iter,
        "shrink_ratio": shrink_ratio,
        "cam_size": cam_size,
        "objective": objective,
        "target_shape": target_shape,
        "target_size": target_size,
        "target_aspect_ratio": target_aspect_ratio,
        "target_center_smooth": target_center_smooth,
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
        cam_type=cam_type,
        show=show,
        init_c=init_c,
        cam_size=cam_size,
        shape_schedule=shape_schedule,
        max_roi_energy_loss=max_energy_loss,
        w_uniformity=w_uniformity,
        w_peak=w_peak,
        w_displacement=w_displacement,
        log_uniformity=log_uniformity,
        target_max_brightness=target_max_brightness,
        slm_number=slm_number,
        slm_wavelength=slm_wavelength,
        zernike_radius=zernike_radius,
        shift_x=shift_x,
        shift_y=shift_y,
        optimizer_type=optimizer_type,
        algorithm=algorithm,
        pop_size=pop_size,
        random_seed=seed,
        objective=objective,
        target_shape=target_shape,
        target_size=target_size,
        target_aspect_ratio=target_aspect_ratio,
        target_center_smooth=target_center_smooth,
    )
    res_list.save_best(
        saved_dir, target="_c", process_fn=lambda x: np.round(x, 6), fmt="%.6f"
    )

    # 保存扰动系数
    if disturb_scale > 0:
        disturb_file = Path(saved_dir) / "disturb_c.txt"
        np.savetxt(disturb_file, disturb, fmt="%.6f")
        logger.info(f"Disturbance coefficients saved to {disturb_file}")

    _, (max_j_id, max_j) = res_list.get_best_iter()

    if debug:
        _save_debug_artifacts(res_list, objective, config, root_dir)

    objective_names = {
        "pib": "PIB",
        "radiu": "半径",
        "avg_radiu": "平均半径",
        "shape": "动态ROI整形",
        "roi_pib": "目标ROI内亮度",
    }
    objective_name = objective_names.get(objective, objective)

    click.echo(
        f"SLM Zernike PIB 优化完成，最优{objective_name}值: {max_j:.4f} @ epoch {max_j_id}"
    )


if __name__ == "__main__":
    setup_coredumpy()
    run()
