from time import sleep

import click
from typing import Literal
from pathlib import Path

import numpy as np
from loguru import logger

from ao_shaping.optimizer.wf.zernike_response_matrix import (
    calibrate_zernike_response_matrix,
    save_zernike_response_matrix,
    DEFAULT_N_MAX,
    DEFAULT_MAGNITUDE,
    DEFAULT_N_AVERAGES,
    DEFAULT_N_CYCLES,
    DEFAULT_WAIT_TIME,
)
from ao_shaping.drivers.slm import ZernikeSLM
from ao_shaping.drivers.wfs.thorlab_wfs import WFSManager, MlaRes
from ao_shaping.utils.matrix_utils import calc_n_zernike_terms
from ao_shaping.utils.display import ZernikeCalibrationDisplay
from ao_shaping.utils.cli_helpers import parse_tuple, setup_coredumpy, get_timestamp_str
from ao_shaping.utils.wfs_utils import flatten_slopes
from ao_shaping.utils.wfs_utils import DitheredReference
from ao_shaping.algorithm.controller import AOClosedLoop, ControlLaw, LoopConfig, HardwareConfig


@click.command('zernike-matrix')
@click.pass_context
@click.option('--n-max', default=10, help='Zernike最大阶数')
@click.option('--magnitude', default=0.5, help='扰动幅度 (波长, 0=自动优化)')
@click.option('--n-averages', 'n_averages', default=3, help='每次WFS读取次数 (M)')
@click.option('--n-cycles', 'n_cycles', default=1, help='正负交替循环次数 (N)')
@click.option('--wait', 'wait_time', default=0.2, help='等待时间 (秒)')
@click.option('--output', 'output_path', default='data/zernike_response_matrix', help='输出文件路径')
@click.option('--slm-number', 'slm_number', default=1, help='SLM设备编号')
@click.option('--shift-x', 'shift_x', type=int, default=0, help='SLM X方向平移像素 (正=右, 负=左)')
@click.option('--shift-y', 'shift_y', type=int, default=0, help='SLM Y方向平移像素 (正=下, 负=上)')
@click.option('--wavelength', default=1064, help='工作波长 (nm)')
@click.option('--mla-index', 'mla_index', type=click.Choice(['512', '540', '600', '768', '1280']), default='512', help='MLA分辨率 (512, 540, 600, 768, 1280)')
@click.option('--exp-time', 'exp_time', type=float, default=0.0, help='曝光时间 (ms, 0=自动)')
@click.option('--auto-exposure/--no-auto-exposure', 'auto_exposure', default=True, help='启用WFS自动曝光 (默认开启)')
@click.option('--high-speed', 'high_speed', is_flag=True, default=False, help='启用高速模式')
@click.option('--use-custom-ref', 'use_custom_ref', is_flag=False, default=False, help='使用自定义参考文件')
@click.option('--pupil-diameter', 'pupil_diameter', type=float, default=2.0, help='瞳孔直径 (mm)')
@click.option('--pupil-center', callback=parse_tuple, default="(0,0)", help='瞳孔中心坐标 (默认: (0,0))')
@click.option('--no-inverses', 'compute_inverses', default=True, flag_value=False, help='不计算逆矩阵')
@click.option('--excluded-tip-tilt', 'excluded_tip_tilt', default=False, flag_value=True, help='排除tip/tilt模式 (Z2, Z3)')
@click.option('--cancel-tile', 'cancel_tile', is_flag=True, default=False, help='测量时去除WFS的tip/tilt (对应Thorlabs的cancel_tile功能)')
@click.option('--display/--no-display', default=False, help='显示实时pygame显示')
@click.option('--debug', 'debug', is_flag=True, default=None, help='启用调试模式 (保存原始测量数据)')
@click.option('--auto-optimize/--no-auto-optimize', 'auto_optimize_amplitude', default=True, help='自动优化扰动幅度 (magnitude=0时)')
@click.option('--optimize-n-avg', 'optimize_n_avg', default=10, help='幅度优化时的WFS读取次数')
@click.option('--n-magnitudes', 'n_magnitudes', default=0, help='自动生成N个不同扰动幅度并分别保存 (0=禁用)')
@click.option('--dither-amp', 'dither_amp', default=0.0, help='亚波长抖动幅度 [λ], 0=禁用 (建议0.02-0.05)')
@click.option('--correction-csv', 'correction_csv_path', default=None, help='误差矫正CSV文件路径 (如 libs/SLM_DLL_ver.2.51/Wavefront_correction_Data/Wavefront_correction_Data_240236000006(520nm).csv)')
def run(
    ctx: click.Context,
    n_max: int,
    magnitude: float,
    n_averages: int,
    n_cycles: int,
    wait_time: float,
    output_path: str,
    slm_number: int,
    shift_x: int,
    shift_y: int,
    wavelength: int,
    mla_index: Literal['512', '540', '600', '768', '1280'],
    exp_time: float,
    auto_exposure: bool,
    high_speed: bool,
    use_custom_ref: bool,
    pupil_diameter: float,
    pupil_center: tuple,
    compute_inverses: bool,
    excluded_tip_tilt: bool,
    cancel_tile: bool,
    display: bool,
    debug: bool | None,
    auto_optimize_amplitude: bool,
    optimize_n_avg: int,
    n_magnitudes: int,
    dither_amp: float,
    correction_csv_path: str | None,
    excluded_piston: bool = True
):
    """获取Zernike响应矩阵

    支持 N 次正负交替循环测量 + M 次 WFS 读取取平均 + 方差跟踪 + 逆矩阵计算。

    多幅度模式 (--n-magnitudes N):
        自动生成N个不同扰动幅度 (0.1到0.8λ)，分别标定并保存。

    调试模式 (--debug):
        保存每次测量的原始数据:
        - SLM相位图 (灰度值, 已应用shift)
        - WFS deviation数据
        - WFS Zernike系数 (averaging前)
    """
    n_max = n_max or DEFAULT_N_MAX
    magnitude = magnitude or DEFAULT_MAGNITUDE
    n_averages = n_averages or DEFAULT_N_AVERAGES
    n_cycles = n_cycles or DEFAULT_N_CYCLES
    wait_time = wait_time or DEFAULT_WAIT_TIME

    # Auto-enable cancel_tile when excluded_tip_tilt is set
    if excluded_tip_tilt:
        cancel_tile = True
        click.echo("Note: --excluded-tip-tilt enabled, auto-setting --cancel-tile")

    # Determine debug flag: use explicit value, or inherit from parent context
    if debug is None:
        debug = ctx.parent.obj.get("debug", False) if ctx.parent and ctx.parent.obj else False

    # Handle auto_exposure: when enabled, set exp_time to 0.0 to trigger auto-exposure
    effective_exp_time = 0.0 if auto_exposure else exp_time

    # Convert mla_index string to MlaRes enum
    mla_index_enum = MlaRes.from_str(mla_index)

    # Debug data saving: create callback if debug mode is enabled
    debug_data_callback = None
    debug_data_dir = None
    if debug:
        debug_data_dir = Path(output_path) / f"debug_{get_timestamp_str()}"
        debug_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug mode enabled, saving raw data to: {debug_data_dir}")

        def debug_callback(
            mode_index: int,
            cycle: int,
            sample: int,
            slm_phase: np.ndarray,
            shift_x: int,
            shift_y: int,
            deviation_x: np.ndarray,
            deviation_y: np.ndarray,
            zernike_coeffs: np.ndarray,
            is_plus: bool,
        ) -> None:
            """Save debug data for each measurement."""
            sign_str = "plus" if is_plus else "minus"
            mode_dir = debug_data_dir / f"mode_{mode_index:03d}" / f"cycle_{cycle}" / sign_str
            mode_dir.mkdir(parents=True, exist_ok=True)

            np.save(mode_dir / f"sample_{sample:03d}_slm_phase.npy", slm_phase)

            if deviation_x is not None and len(deviation_x) > 0:
                np.save(mode_dir / f"sample_{sample:03d}_deviation_x.npy", deviation_x)
                np.save(mode_dir / f"sample_{sample:03d}_deviation_y.npy", deviation_y)

            if zernike_coeffs is not None and len(zernike_coeffs) > 0:
                np.save(mode_dir / f"sample_{sample:03d}_zernike_coeffs.npy", zernike_coeffs)

            import json
            meta = {
                "mode_index": mode_index,
                "cycle": cycle,
                "sample": sample,
                "shift_x": shift_x,
                "shift_y": shift_y,
                "is_plus": is_plus,
            }
            with open(mode_dir / f"sample_{sample:03d}_meta.json", "w") as f:
                json.dump(meta, f)

        debug_data_callback = debug_callback

    # Calculate n_slm_terms and n_wfs_terms before calibration
    n_remove = (1 if excluded_piston else 0) + (2 if excluded_tip_tilt else 0)
    n_slm_terms = calc_n_zernike_terms(n_max) - n_remove
    n_wfs_terms = calc_n_zernike_terms(n_max) - n_remove

    # Conditionally create display
    ui_display = None
    if display:
        ui_display = ZernikeCalibrationDisplay(n_wfs_terms=n_wfs_terms, n_slm_terms=n_slm_terms)

    try:
        with ZernikeSLM(slm_number=slm_number, wavelength=wavelength, n_max=n_max, shift_x=shift_x, shift_y=shift_y, correction_csv_path=correction_csv_path) as zslm:
            with WFSManager(
                mla_index=mla_index_enum,
                exp_time=effective_exp_time,
                high_speed=high_speed,
                use_custom_ref=use_custom_ref,
                pupil_diameter=pupil_diameter,
                pupil_center=pupil_center,
            ) as wfs:
                dither_diagnostics = None
                if dither_amp > 0:
                    click.echo(f"Dithered reference: amp={dither_amp}λ, n={n_averages}")
                    dither = DitheredReference(
                        slm=zslm,
                        dither_amp=dither_amp,
                        n_dither=n_averages,
                        wait_time=wait_time,
                    )
                    _, dither_diagnostics = dither.measure(wfs, n_averages=n_averages)
                    click.echo(f"Dithered ref SNR: {dither_diagnostics['snr_db']:.1f} dB")

                if not use_custom_ref:
                    # 测量响应矩阵需要在平面下测量防止wfs质心偏移过大
                    zslm.set_flat()
                    sleep(0.5)
                    wfs.save_user_ref()
                    wfs.load_user_ref()

                    logger.debug(f'自动标定参考波前:{wfs.get_wavefront(cancel_tile)[1]["rms"]}')


                magnitudes_to_run = []
                if n_magnitudes > 0:
                    magnitudes_to_run = np.linspace(0.1, 0.8, n_magnitudes).round(3).tolist()
                    click.echo(f"Multi-magnitude mode: running {n_magnitudes} calibrations with magnitudes {magnitudes_to_run}")
                elif magnitude == 0:
                    magnitudes_to_run = [None]
                else:
                    magnitudes_to_run = [magnitude]

                results = []
                for mag in magnitudes_to_run:
                    effective_magnitude = mag
                    if mag is None:
                        effective_magnitude = 0.0

                    mag_suffix = f"_mag{mag}" if mag is not None else "_auto"
                    mag_output_path = f"{output_path}{mag_suffix}"

                    click.echo(f"\n=== Calibration run: magnitude={effective_magnitude} ===")

                    result = calibrate_zernike_response_matrix(
                        zslm=zslm,
                        wfs=wfs,
                        n_max=n_max,
                        magnitude=effective_magnitude,
                        n_averages=n_averages,
                        n_cycles=n_cycles,
                        wait_time=wait_time,
                        excluded_piston=excluded_piston,
                        excluded_tip_tilt=excluded_tip_tilt,
                        compute_inverses=compute_inverses,
                        display=ui_display,
                        verbose=True,
                        debug_data_callback=debug_data_callback,
                        auto_optimize_amplitude=auto_optimize_amplitude,
                        optimize_n_avg=optimize_n_avg,
                        cancel_tile=cancel_tile,
                    )

                    # 保存硬件配置快照 (用于闭环优化恢复)
                    result.device_config = {
                        "slm_number": slm_number,
                        "wavelength": wavelength,
                        "n_max": n_max,
                        "shift_x": shift_x,
                        "shift_y": shift_y,
                        "correction_csv_path": correction_csv_path or "",
                        "mla_index": int(mla_index_enum),
                        "exposure_time": effective_exp_time,
                        "high_speed": high_speed,
                        "use_custom_ref": use_custom_ref,
                        "pupil_center": list(pupil_center),
                        "pupil_diameter": pupil_diameter,
                    }
                    save_zernike_response_matrix(result, mag_output_path, include_inverses=compute_inverses)
                    click.echo(f"Saved: {mag_output_path}.h5")
                    results.append((mag, result))

                if len(results) == 1:
                    result = results[0][1]
                    click.echo(f"\n响应矩阵已保存到: {output_path}")
                else:
                    click.echo(f"\n多幅度标定完成: {len(results)} 组")
                    for mag, res in results:
                        click.echo(f"  mag={mag}: mean_var={res.mean_variance:.6f}, shape={res.matrix.shape}")

                click.echo(f"矩阵形状: {result.matrix.shape}")
                click.echo(f"平均方差: {result.mean_variance:.6f}")
                click.echo(f"最大方差: {result.max_variance:.6f}")
                click.echo(f"排除piston: {result.excluded_piston}, 排除tip/tilt: {result.excluded_tip_tilt}")
                if result.condition_number is not None:
                    click.echo(f"条件数: {result.condition_number:.2e}")
                if debug_data_dir is not None:
                    click.echo(f"调试数据已保存到: {debug_data_dir}")
    finally:
        if ui_display is not None:
            ui_display.close()


# ==================== 独立辅助函数 ====================


def from_response_matrix(result, config: LoopConfig) -> AOClosedLoop:
    """从ZernikeResponseMatrixResult加载响应矩阵并构造纯控制器

    优先使用 deviation_response_matrix (斜率空间) 进行控制,
    若不可用则回退到 Zernike矩阵+pinv (模态空间)。

    Args:
        result: 已加载的 ZernikeResponseMatrixResult 对象
        config: 环路配置

    Returns:
        配置好的 AOClosedLoop 实例 (不含硬件依赖)
    """
    n_max = result.n_max
    excluded_piston = result.excluded_piston
    excluded_tip_tilt = result.excluded_tip_tilt

    if result.deviation_response_matrix is not None and result.subaperture_mask is not None:
        dev_mat = result.deviation_response_matrix
        mask = result.subaperture_mask.flatten()
        mask_2n = np.concatenate([mask, mask])
        valid_rows = np.where(mask_2n)[0]

        D_slopes = dev_mat[valid_rows]
        D_pinv_slopes = np.linalg.pinv(D_slopes)
        s_ref = np.zeros(D_slopes.shape[0])

        logger.info(
            f"使用斜率空间响应矩阵: D shape={D_slopes.shape}, "
            f"有效测量={D_slopes.shape[0]}"
        )
        return AOClosedLoop(
            D=D_slopes,
            D_pinv=D_pinv_slopes,
            s_ref=s_ref,
            mask_indices=valid_rows,
            config=config,
            excluded_piston=excluded_piston,
            excluded_tip_tilt=excluded_tip_tilt,
        )
    elif result.pinv_matrix is not None:
        D = result.matrix
        D_pinv = result.pinv_matrix
        s_ref = np.zeros(D_pinv.shape[1])

        logger.info(
            f"使用模态空间响应矩阵: D shape={D.shape}, "
            f"pinv shape={D_pinv.shape}"
        )
        return AOClosedLoop(
            D=D,
            D_pinv=D_pinv,
            s_ref=s_ref,
            mask_indices=np.arange(D.shape[0]),
            config=config,
            excluded_piston=excluded_piston,
            excluded_tip_tilt=excluded_tip_tilt,
        )
    else:
        raise ValueError(
            "响应矩阵缺少 deviation_response_matrix 或 pinv_matrix, "
            "无法构建控制器。请确保校准时 --no-inverses 未启用且产生了斜率响应。"
        )


def _measure_wfs_step(
    wfs,
    mask_indices: np.ndarray,
    s_ref: np.ndarray,
    D_pinv: np.ndarray,
    cancel_tile: bool = False,
) -> tuple[np.ndarray, float]:
    """单次WFS测量: 获取斜率残差并估计RMS

    Args:
        wfs: WFSManager 实例
        mask_indices: 有效子孔径索引
        s_ref: 参考斜率 [n_meas]
        D_pinv: 响应矩阵伪逆 [n_modes, n_meas]
        cancel_tile: 测量时是否去除WFS tip/tilt

    Returns:
        tuple: (delta_s, rms)
            - delta_s: 斜率残差 [n_meas]
            - rms: 估计的Zernike系数RMS [λ]
    """
    dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=cancel_tile)
    s = flatten_slopes(dev_x, dev_y)
    delta_s = s[mask_indices] - s_ref
    a_est = D_pinv @ delta_s
    rms = float(np.sqrt(np.sum(a_est**2)))
    return delta_s, rms


def _expand_to_noll(
    u: np.ndarray,
    n_max: int,
    excluded_piston: bool = True,
    excluded_tip_tilt: bool = False,
) -> np.ndarray:
    """将控制器系数扩展为完整Noll顺序 (补零排除的模式)

    Args:
        u: 控制器系数 [n_modes] (不含piston/tip-tilt)
        n_max: Zernike最大阶数
        excluded_piston: 是否排除了piston模式
        excluded_tip_tilt: 是否排除了tip/tilt模式

    Returns:
        完整Noll顺序系数 [calc_n_zernike_terms(n_max)]
    """
    n_total = calc_n_zernike_terms(n_max)
    full = np.zeros(n_total, dtype=np.float64)
    n_pad = 0
    if excluded_piston:
        n_pad += 1
    if excluded_tip_tilt:
        n_pad += 2
    full[n_pad:n_pad + len(u)] = u
    return full


@click.command("closed-loop")
@click.pass_context
@click.option("--load-file", "load_file", required=True, help="已保存的响应矩阵 .h5 文件路径")
@click.option("--output", "output_path", default=None, help="结果保存路径 (默认: 在load-file同目录生成)")
@click.option("--control-law", "control_law",
              type=click.Choice(["pid", "leaky", "qg", "lqg", "mpc", "adaptive"]),
              default="leaky", help="控制律 (默认: leaky)")
@click.option("--gain", default=None, type=float, help="控制增益覆盖 (控制律依赖)")
@click.option("--leak", default=None, type=float, help="泄漏因子覆盖")
@click.option("--kp", default=None, type=float, help="PID比例增益")
@click.option("--ki", default=None, type=float, help="PID积分增益")
@click.option("--kd", default=None, type=float, help="PID微分增益")
@click.option("--dt", default=0.067, type=float, help="采样周期 [s] (默认: 0.067)")
@click.option("--rms-target", "rms_target", default=0.05, type=float, help="目标RMS [λ] (默认: 0.05)")
@click.option("--max-iter", "max_iter", default=100, type=int, help="最大迭代次数 (默认: 100)")
@click.option("--delay-steps", "delay_steps", default=1, type=int, help="延时补偿步数 (默认: 1)")
@click.option("--cancel-tile/--no-cancel-tile", "cancel_tile", default=False, help="测量时去除WFS tip/tilt")
@click.option("--display/--no-display", default=False, help="显示实时pygame显示")
@click.option("--debug", is_flag=True, default=False, help="启用调试模式")
def closed_loop_run(
    ctx: click.Context,
    load_file: str,
    output_path: str | None,
    control_law: str,
    gain: float | None,
    leak: float | None,
    kp: float | None,
    ki: float | None,
    kd: float | None,
    dt: float,
    rms_target: float,
    max_iter: int,
    delay_steps: int,
    cancel_tile: bool,
    display: bool,
    debug: bool,
):
    """基于响应矩阵的闭环波前优化

    载入已保存的Zernike响应矩阵 (.h5), 恢复硬件参数,
    使用指定控制律进行闭环优化, 保存优化结果。
    """
    from pathlib import Path
    import json
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import datetime
    from loguru import logger

    from ao_shaping.optimizer.wf.zernike_response_matrix import load_zernike_response_matrix
    from ao_shaping.drivers.slm import ZernikeSLM
    from ao_shaping.drivers.wfs.thorlab_wfs import WFSManager, MlaRes
    from ao_shaping.utils.cli_helpers import get_timestamp_str

    # 加载响应矩阵
    click.echo(f"加载响应矩阵: {load_file}")
    result = load_zernike_response_matrix(load_file)
    click.echo(f"  矩阵形状: {result.matrix.shape}")
    click.echo(f"  n_max={result.n_max}, excluded_piston={result.excluded_piston}, excluded_tip_tilt={result.excluded_tip_tilt}")

    # 提取硬件配置
    device_cfg = HardwareConfig()
    if result.device_config is not None:
        device_cfg = HardwareConfig.from_dict(result.device_config)
        click.echo(f"已恢复硬件配置: SLM#{device_cfg.slm_number}, WL={device_cfg.wavelength}nm, "
                    f"shift=({device_cfg.shift_x},{device_cfg.shift_y}), "
                    f"MLA={device_cfg.mla_index}, exp={device_cfg.exposure_time}ms")
    else:
        click.echo("警告: 响应矩阵中未找到硬件配置, 将使用默认参数")
        click.echo("建议重新标定以保存完整硬件参数, 或手动指定设备参数")

    # 构建控制配置
    n_modes = result.n_slm_terms
    Q_diag = np.ones(n_modes)
    loop_cfg = LoopConfig(
        n_modes=n_modes,
        dt=dt,
        Kp=kp if kp is not None else 0.5,
        Ki=ki if ki is not None else 0.3,
        Kd=kd if kd is not None else 0.05,
        leak=leak if leak is not None else 0.97,
        Q_diag=Q_diag,
        R_scalar=0.1,
        delay_steps=delay_steps,
        rms_target=rms_target,
        max_iter=max_iter,
        cancel_tile=cancel_tile,
    )
    if gain is not None:
        loop_cfg.gain_schedule = [(0, max_iter, gain, loop_cfg.leak)]

    click.echo(f"\n闭环配置:")
    click.echo(f"  控制律: {control_law}, DT={dt}s")
    click.echo(f"  目标RMS: {rms_target}λ, 最大迭代: {max_iter}")

    # 解析控制律枚举
    law_map = {
        "pid": ControlLaw.PID,
        "leaky": ControlLaw.LEAKY_INTEGRATOR,
        "qg": ControlLaw.QUADRATIC_GAUSSIAN,
        "lqg": ControlLaw.LQG,
        "mpc": ControlLaw.PREDICTIVE,
        "adaptive": ControlLaw.ADAPTIVE_GAIN,
    }
    law = law_map[control_law]

    # 设置输出路径
    load_path = Path(load_file)
    if output_path is None:
        output_path = str(load_path.parent / f"closed_loop_{load_path.stem}_{get_timestamp_str()}")

    # 初始化硬件
    ui_display = None
    try:
        click.echo("\n初始化SLM...")
        with ZernikeSLM(
            slm_number=device_cfg.slm_number,
            wavelength=device_cfg.wavelength,
            n_max=result.n_max,
            shift_x=device_cfg.shift_x,
            shift_y=device_cfg.shift_y,
            correction_csv_path=(
                device_cfg.correction_csv_path
                if device_cfg.correction_csv_path
                else None
            ),
        ) as zslm:
            click.echo(f"  SLM已初始化: shift=({device_cfg.shift_x}, {device_cfg.shift_y})")

            click.echo("初始化WFS...")
            with WFSManager(
                mla_index=MlaRes(device_cfg.mla_index),
                exp_time=device_cfg.exposure_time,
                high_speed=device_cfg.high_speed,
                use_custom_ref=device_cfg.use_custom_ref,
                pupil_diameter=device_cfg.pupil_diameter,
                pupil_center=tuple(device_cfg.pupil_center),
            ) as wfs:
                click.echo(f"  WFS已初始化: MLA={device_cfg.mla_index}, "
                           f"exp={device_cfg.exposure_time}ms")

                # 设置SLM为平面, 更新WFS参考
                click.echo("\n设置初始参考 (SLM平场)...")
                zslm.set_flat()
                sleep(0.5)
                wfs.save_user_ref()
                wfs.load_user_ref()

                # 构造纯控制器 (不含硬件依赖)
                click.echo("构建闭环控制器...")
                loop = from_response_matrix(
                    result=result,
                    config=loop_cfg,
                )

                # 创建测量和执行回调 (捕获硬件实例)
                measure_func = lambda: _measure_wfs_step(
                    wfs=wfs,
                    mask_indices=loop.mask_indices,
                    s_ref=loop.s_ref,
                    D_pinv=loop.D_pinv,
                    cancel_tile=cancel_tile,
                )
                def _apply(u: np.ndarray) -> None:
                    u_full = _expand_to_noll(
                        u=u,
                        n_max=result.n_max,
                        excluded_piston=result.excluded_piston,
                        excluded_tip_tilt=result.excluded_tip_tilt,
                    )
                    zslm.send_zernike(u_full)

                # 运行闭环
                click.echo(f"\n启动闭环优化 ({control_law})...")
                loop_result = loop.run(
                    measure_func=measure_func,
                    apply_func=_apply,
                    control_law=law,
                )

                # 保存结果
                output_dir = Path(output_path)
                output_dir.mkdir(parents=True, exist_ok=True)

                # 保存系数和RMS历史
                np.savez(
                    output_dir / "history.npz",
                    a_history=loop_result["a_history"],
                    rms_history=loop_result["rms_history"],
                    s_history=loop_result["s_history"],
                    u_history=loop_result["u_history"],
                    final_coefficients=loop_result["final_coefficients"],
                )

                # 保存最终系数 (文本格式, 便于加载到其他工具)
                np.savetxt(
                    output_dir / "final_coefficients.txt",
                    loop_result["final_coefficients"],
                    fmt="%.8f",
                    header=f"Closed-loop final Zernike coefficients (noll order, n_modes={n_modes})",
                )

                # 保存结果元数据
                meta = {
                    "load_file": load_file,
                    "control_law": control_law,
                    "n_iter": loop_result["n_iter"],
                    "rms_initial": float(loop_result["rms_initial"]),
                    "rms_final": float(loop_result["rms_final"]),
                    "improvement_db": float(loop_result["improvement_db"]),
                    "converged": bool(loop_result["converged"]),
                    "diverged": bool(loop_result["diverged"]),
                    "timestamp": datetime.now().isoformat(),
                    "device_config": device_cfg.to_dict(),
                    "loop_config": {
                        "n_modes": n_modes,
                        "dt": dt,
                        "Kp": loop_cfg.Kp,
                        "Ki": loop_cfg.Ki,
                        "Kd": loop_cfg.Kd,
                        "leak": loop_cfg.leak,
                        "rms_target": rms_target,
                        "max_iter": max_iter,
                        "delay_steps": delay_steps,
                        "cancel_tile": cancel_tile,
                    },
                }
                with open(output_dir / "meta.json", "w") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

                # 生成收敛曲线
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                iterations = np.arange(len(loop_result["rms_history"]))
                axes[0].plot(iterations, loop_result["rms_history"], "b-")
                axes[0].axhline(y=rms_target, color="r", linestyle="--", alpha=0.5, label=f"target={rms_target}λ")
                axes[0].set_xlabel("Iteration")
                axes[0].set_ylabel("RMS [λ]")
                axes[0].set_title(f"Convergence ({control_law})")
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()

                if len(loop_result["a_history"]) > 0:
                    coeffs = np.array(loop_result["a_history"])
                    for i in range(min(5, coeffs.shape[1])):
                        axes[1].plot(iterations, coeffs[:, i], label=f"Z{i+2}")
                    axes[1].set_xlabel("Iteration")
                    axes[1].set_ylabel("Coefficient [λ]")
                    axes[1].set_title("First 5 Zernike Modes")
                    axes[1].grid(True, alpha=0.3)
                    axes[1].legend()

                fig.tight_layout()
                fig.savefig(output_dir / "convergence.png", dpi=150)
                plt.close(fig)

                click.echo(f"\n结果已保存到: {output_dir}")
                click.echo(f"  RMS: {loop_result['rms_initial']:.4f}λ → {loop_result['rms_final']:.4f}λ")
                click.echo(f"  改善: {loop_result['improvement_db']:.1f} dB")
                click.echo(f"  迭代: {loop_result['n_iter']}, 收敛: {loop_result['converged']}")

    finally:
        if ui_display is not None:
            ui_display.close()


if __name__ == "__main__":
    setup_coredumpy()
    run()
