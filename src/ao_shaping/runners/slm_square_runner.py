"""Click CLI runner for SLM square beam uniformity optimization.

SPGD optimization of Zernike coefficients on an SLM to produce a uniform
square intensity pattern, using camera feedback.

==================== 使用方法 ====================

CLI 两种入口等价:
    python src/ao_shaping/main.py spgd-square [OPTIONS]
    python -m ao_shaping.runners.slm_square_runner [OPTIONS]

常用示例:
    # 默认配置 (SLM #1, 相机 0, n_max=4, 2000 轮)
    python src/ao_shaping/main.py spgd-square -e 2000 -n 4

    # 指定初始 Zernike 系数 (Noll 索引 dict)。zernike 基只优化
    # Defocus (2,0) [Noll 4] 与 Spherical (4,0) [Noll 11], 其余系数忽略:
    python src/ao_shaping/main.py spgd-square --init-coeffs '{"4":1.0,"11":0.5}'

    # 指定目标方形平均亮度 (能量守恒自动推导边长; 与 --target-side 互斥)
    python src/ao_shaping/main.py spgd-square -e 500 --target-mean-brightness 80

    # Noll 序扁平数组 (索引 0 = Noll 1 = 0): 只取活动模式条目,
    # 索引 3 = Noll 4 = defocus (2,0), 索引 10 = Noll 11 = spherical (4,0)
    python src/ao_shaping/main.py spgd-square --init-coeffs '[0,0,0,1.0,0,0,0,0,0,0,0.5]'

配套模块:
- 核心算法: `ao_shaping.optimizer.wfless.slm_square_shaping.optimize_slm_square`
- Zernike 工具: `ao_shaping.utils.zernike_utils` (系数解析/相位生成, 含 Noll 约定说明)

结果输出 (默认 data/slm_square/<日期>/):
- 最优迭代 CSV (recorder)
- 最优 Zernike 系数 CSV (Noll 序)
- 最优远场图 PNG (--save-best-image)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
from loguru import logger

from ao_shaping.utils.cli_helpers import get_debug_mode, get_date_dir_name


@click.command()
@click.option(
    "-e", "--epochs", default=2000, type=int, help="优化迭代次数 (default: 2000)"
)
@click.option(
    "-n", "--n-max", default=4, type=int, help="Zernike最大径向阶数 (default: 4)"
)
@click.option(
    "-c",
    "--center",
    default="shape",
    type=str,
    help="光斑中心检测: shape(智能argmax锚定,默认)/centroid_thresh(亮度重心)"
    "/max(峰值位置)/mass(质心,易被杂散光拉偏)/'x,y'(固定坐标) (default: shape)",
)
@click.option(
    "--target-side",
    default=0,
    type=int,
    help="目标方形边长(像素), 0=自动; 与 --target-mean-brightness 互斥 (default: 0)",
)
@click.option(
    "--target-mean-brightness",
    default=0.0,
    type=float,
    help="目标方形平均亮度(灰度), >0 时由总亮度能量守恒自动推导边长; "
    "与 --target-side 互斥 (default: 0 = 不启用)",
)
@click.option(
    "--side-factor", default=1.5, type=float, help="自动边长倍率 (default: 1.5)"
)
@click.option("-d", "--delta", default=0.1, type=float, help="扰动幅度 (default: 0.1)")
@click.option("--lr", default=0.0, type=float, help="学习率, 0=自动 (default: 0)")
@click.option(
    "-t", "--exposure-ms", default=80.0, type=float, help="相机曝光时间ms (default: 80)"
)
@click.option("--cam-id", default=0, type=int, help="相机设备ID (default: 0)")
@click.option(
    "-s", "--cam-size", default=300, type=int, help="相机开窗大小 (default: 300)"
)
@click.option("--slm-number", default=1, type=int, help="SLM设备编号 (default: 1)")
@click.option(
    "--slm-wavelength", default=1064, type=int, help="SLM波长nm (default: 1064)"
)
@click.option(
    "--optimizer",
    default="adamod",
    type=str,
    help="优化器: adam/adamod/sgd/muno (default: adamod)",
)
@click.option("--seed", default=None, type=int, help="随机种子 (default: None)")
@click.option("--show", is_flag=True, default=False, help="显示中间图像")
@click.option(
    "--target-brightness", default=200, type=int, help="目标最大亮度 (default: 200)"
)
@click.option(
    "--w-uniformity", default=0.4, type=float, help="均匀性权重 (default: 0.4)"
)
@click.option(
    "--w-efficiency", default=0.6, type=float, help="能量效率权重 (default: 0.6)"
)
@click.option("--w-aspect", default=0.0, type=float, help="宽高比权重 (default: 0.0)")
@click.option(
    "--basis",
    default="zernike",
    type=click.Choice(["freeform", "zernike"]),
    help="相位参数化: zernike(默认, 与GUI一致: radius=600 + defocus(2,0) + spherical(4,0))/freeform(自由相位, 可合成方形)",
)
@click.option(
    "--phase-grid",
    default=24,
    type=int,
    help="freeform 相位网格边长 (dim=grid²) (default: 24)",
)
@click.option(
    "--zernike-radius",
    default=600,
    type=int,
    help="Zernike 孔径半径(px), 默认 600 = SLM 面板短边一半 (与GUI一致)",
)
@click.option(
    "--zernike-mask",
    default=None,
    type=str,
    help="0/1 binary mask for Zernike modes (comma-separated), e.g. '0,0,0,1,0,0,0,0,0,0,0,1' for defocus+spherical only. Noll 1-3 forced to 0. Overrides --basis zernike defaults.",
)
@click.option(
    "--rotation-search",
    "rotation_search_deg",
    default=0.0,
    type=float,
    help="SLM↔相机相对旋转搜索范围(度, 0~360; 0=关闭旋转校正)。>0 时旋转角作为额外 SPGD 自由度在 ±range/2 内搜索",
)
@click.option(
    "--init-defocus",
    default=1.0,
    type=float,
    help="初始 Defocus (2,0) 系数 (default: 1.0)",
)
@click.option(
    "--init-spherical",
    default=0.5,
    type=float,
    help="初始 Spherical (4,0) 系数 (default: 0.5)",
)
@click.option(
    "--init-coeffs",
    default=None,
    type=str,
    help='初始Zernike系数JSON (Noll 索引 dict 或 Noll 序数组); zernike 基只优化 Defocus(2,0)[Noll 4] 与 Spherical(4,0)[Noll 11], e.g. \'{"4":1.0,"11":0.5}\'',
)
@click.option("--save-best-image", is_flag=True, default=False, help="保存最优图像")
@click.pass_context
def run(
    ctx: click.Context,
    epochs: int,
    n_max: int,
    center: str,
    target_side: int,
    target_mean_brightness: float,
    side_factor: float,
    delta: float,
    lr: float,
    exposure_ms: float,
    cam_id: int,
    cam_size: int,
    slm_number: int,
    slm_wavelength: int,
    optimizer: str,
    seed: int | None,
    show: bool,
    target_brightness: int,
    w_uniformity: float,
    w_efficiency: float,
    w_aspect: float,
    basis: str,
    phase_grid: int,
    zernike_radius: int,
    zernike_mask: str | None,
    rotation_search_deg: float,
    init_defocus: float,
    init_spherical: float,
    init_coeffs: str | None,
    save_best_image: bool,
):
    """SLM方形光斑整形优化器

    使用SPGD算法优化SLM上的Zernike系数，通过相机反馈产生均匀方形远场光斑。

    优化目标: 最小化目标方形区域内光强的变异系数(CV = std/mean)，同时可选
    加权环围能量(EE)和宽高比(AR)指标。

    DEBUG环境变量控制调试模式。
    """
    debug = get_debug_mode()
    root_dir = (
        Path(ctx.parent.obj.get("dir", "data"))
        if ctx.parent is not None and ctx.parent.obj is not None
        else Path("data")  # python -m 独立运行时无 main 组上下文
    )

    # 目标方形参数二选一: 边长(px) 或 平均亮度
    if target_side > 0 and target_mean_brightness > 0:
        raise click.UsageError(
            "--target-side 与 --target-mean-brightness 互斥: "
            "方形边长(px) 与 方形平均亮度只能二选一"
        )

    # Parse center
    center_arg: tuple[int, int] | str | None = center
    try:
        parts = center.split(",")
        if len(parts) == 2:
            center_arg = (int(parts[0]), int(parts[1]))
    except (ValueError, AttributeError):
        pass

    # Parse init coefficients
    init_c: list[float] | np.ndarray | None = None
    if init_coeffs is not None:
        try:
            parsed = json.loads(init_coeffs)
            if isinstance(parsed, dict):
                # Noll index dict format, convert to list
                from ao_shaping.utils.zernike_utils import parse_zernike_coefficients

                coeffs_dict = parse_zernike_coefficients(parsed, n_max=n_max)
                nk_terms = (n_max + 1) * (n_max + 2) // 2
                init_c = np.zeros(nk_terms, dtype=np.float64)
                from ao_shaping.utils.zernike_calc import noll_to_nm

                for j_idx in range(nk_terms):
                    n, m = noll_to_nm(j_idx + 1)
                    if (n, m) in coeffs_dict:
                        init_c[j_idx] = coeffs_dict[(n, m)]
            elif isinstance(parsed, list):
                init_c = [float(v) for v in parsed]
        except (json.JSONDecodeError, ValueError) as e:
            click.echo(f"Error parsing --init-coeffs: {e}", err=True)
            sys.exit(1)
    elif basis == "zernike":
        # 默认初始系数 = GUI 同款: Defocus(2,0) + Spherical(4,0)
        # 与 multi_slm_controller.py 的 Zernike 分支一致 (radius=600)
        nk_terms = (n_max + 1) * (n_max + 2) // 2
        init_c = np.zeros(nk_terms, dtype=np.float64)
        from ao_shaping.utils.zernike_calc import noll_to_nm

        for j_idx in range(nk_terms):
            n, m = noll_to_nm(j_idx + 1)
            if (n, m) == (2, 0):
                init_c[j_idx] = init_defocus
            elif (n, m) == (4, 0):
                init_c[j_idx] = init_spherical

    # Parse zernike_mask
    zernike_mask_arr = None
    if zernike_mask is not None:
        zernike_mask_arr = np.array(
            [int(x.strip()) for x in zernike_mask.split(",")],
            dtype=int,
        )

    if basis == "zernike" and init_c is not None:
        from ao_shaping.optimizer.wfless.slm_square_shaping import (
            ZERNIKE_ACTIVE_MODES,
            _zernike_indices,
        )

        _noll = _zernike_indices(n_max)
        if zernike_mask_arr is not None:
            _mask = np.asarray(zernike_mask_arr, dtype=int).ravel()
            nk = len(_noll)
            if _mask.size < nk:
                _mask = np.pad(_mask, (0, nk - _mask.size), constant_values=0)
            elif _mask.size > nk:
                _mask = _mask[:nk]
            _mask[0] = _mask[1] = _mask[2] = 0
            active_pos = [i for i in range(nk) if _mask[i] == 1]
            active_modes_str = [_noll[i] for i in active_pos]
        else:
            active_pos = [i for i, m in enumerate(_noll) if m in ZERNIKE_ACTIVE_MODES]
            active_modes_str = ZERNIKE_ACTIVE_MODES
        dropped = [i for i, v in enumerate(init_c) if v != 0 and i not in active_pos]
        if dropped:
            click.echo(
                f"警告: zernike 基活动模式 {active_modes_str} "
                f"(Noll 索引 {active_pos}); 非活动系数被忽略: {dropped}"
            )

    click.echo("=" * 60)
    click.echo("SLM Square Beam Uniformity Optimization (SPGD)")
    click.echo("=" * 60)
    click.echo(f"Zernike order: {n_max} ({(n_max + 1) * (n_max + 2) // 2} terms)")
    if target_mean_brightness > 0:
        click.echo(f"Target: mean brightness = {target_mean_brightness} (auto side)")
    else:
        click.echo(f"Target side: {'auto' if target_side <= 0 else target_side} px")
    click.echo(f"Optimizer: {optimizer}")
    click.echo(
        f"Basis: {basis}"
        + (
            f" (grid={phase_grid})"
            if basis == "freeform"
            else f" (radius={zernike_radius})"
        )
    )
    if basis == "zernike" and init_coeffs is None:
        click.echo(
            f"Init Zernike: Defocus(2,0)={init_defocus}, Spherical(4,0)={init_spherical}"
        )
    if basis == "zernike" and zernike_mask_arr is not None:
        click.echo(f"Zernike mask: {zernike_mask_arr.tolist()}")
    if rotation_search_deg > 0:
        click.echo(f"Rotation search: ±{rotation_search_deg / 2.0:.1f}° (SPGD extra DOF)")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"SLM: #{slm_number} @ {slm_wavelength}nm")
    click.echo(f"Camera: ID={cam_id}, size={cam_size}")
    click.echo(f"Weights: CV={w_uniformity}, EE={w_efficiency}, AR={w_aspect}")
    click.echo("=" * 60)

    from ao_shaping.optimizer.wfless.slm_square_shaping import optimize_slm_square
    from ao_shaping.utils.file import gen_date_dir, gen_date_str

    recorder = optimize_slm_square(
        center=center_arg,
        epochs=epochs,
        n_max=n_max,
        target_side=target_side,
        target_mean_brightness=target_mean_brightness,
        side_factor=side_factor,
        delta=delta,
        lr=lr,
        exposure_time_ms=exposure_ms,
        cam_id=cam_id,
        show=show,
        init_c=init_c,
        cam_size=cam_size,
        target_max_brightness=target_brightness,
        slm_number=slm_number,
        slm_wavelength=slm_wavelength,
        optimizer_type=optimizer,
        random_seed=seed,
        w_uniformity=w_uniformity,
        w_efficiency=w_efficiency,
        w_aspect=w_aspect,
        basis=basis,
        phase_grid=phase_grid,
        zernike_radius=zernike_radius,
        zernike_mask=zernike_mask_arr,
        rotation_search_deg=rotation_search_deg,
    )

    # Print results summary
    best_iter, (best_epoch, best_val) = recorder.get_best_iter()
    click.echo("\n" + "=" * 60)
    click.echo("Optimization Complete")
    click.echo("=" * 60)
    click.echo(f"Best quality: {best_val:.4f} @ epoch {best_epoch}")
    click.echo(f"  CV: {best_iter.get('cv', 'N/A')}")
    click.echo(f"  EE: {best_iter.get('ee', 'N/A')}")
    click.echo(f"  AR: {best_iter.get('ar', 'N/A')}")
    click.echo(f"  Side: {best_iter.get('side', 'N/A')} px")

    # Save results
    save_dir = gen_date_dir(root_dir / "slm_square")
    csv_file = save_dir / f"slm_square_{gen_date_str()}.csv"
    recorder.save_dataframe(csv_file)
    click.echo(f"Results saved: {csv_file}")

    # Save best coefficients
    best_c = best_iter.get("_c")
    if best_c is not None:
        coeffs_file = save_dir / f"best_coefficients.csv"
        np.savetxt(coeffs_file, best_c, fmt="%.6f")
        click.echo(f"Best coefficients saved: {coeffs_file}")

    # Save best image
    if save_best_image:
        best_img = best_iter.get("_img")
        if best_img is not None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(best_img, cmap="gray")
            ax.set_title(
                f"Best square (CV={best_iter.get('cv', 0):.4f}, "
                f"EE={best_iter.get('ee', 0):.4f})"
            )
            ax.scatter(
                best_iter.get("side", 0) // 2,
                best_iter.get("side", 0) // 2,
                c="red",
                s=20,
                marker="+",
            )
            img_file = save_dir / f"best_square.png"
            plt.savefig(img_file, dpi=150, bbox_inches="tight")
            plt.close()
            click.echo(f"Best image saved: {img_file}")

    if debug:
        click.echo(f"\nDebug data saved to: {save_dir}")


if __name__ == "__main__":
    run()
