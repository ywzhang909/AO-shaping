"""Hadamard响应矩阵校准Runner

使用HadamardDM逐一加载各阶Hadamard相位模式，测量对应的Thorlab WFS响应，
建立Hadamard模式命令到WFS响应的响应矩阵。

支持:
- N次正负交替循环测量
- M次WFS读取取平均
- 方差计算作为稳定性指标
- 逆矩阵计算

Example:
    python -m ao_shaping.runners.hadamard_matrix_runner --mode-order 8 --n-averages 5
"""

from __future__ import annotations

import click
from pathlib import Path
from time import sleep

import numpy as np
from loguru import logger

from ao_shaping.drivers.dm.hadamard_dm import HadamardDM
from ao_shaping.drivers.wfs.thorlab_wfs import WFSManager, MlaRes
from ao_shaping.utils.cli_helpers import parse_tuple, setup_coredumpy, get_timestamp_str
from ao_shaping.utils.wfs_utils import flatten_slopes
from ao_shaping.utils.hadamard_calc import calc_n_hadamard_modes


DEFAULT_MODE_ORDER = 8
DEFAULT_MAGNITUDE = 0.5
DEFAULT_N_AVERAGES = 10
DEFAULT_N_CYCLES = 1
DEFAULT_WAIT_TIME = 0.1


@click.command('hadamard-matrix')
@click.pass_context
@click.option('--mode-order', default=8, help='Hadamard矩阵阶数 (2的幂次, 默认8)')
@click.option('--magnitude', default=0.5, help='扰动幅度 (波长)')
@click.option('--n-averages', 'n_averages', default=10, help='每次WFS读取次数 (M)')
@click.option('--n-cycles', 'n_cycles', default=1, help='正负交替循环次数 (N)')
@click.option('--wait', 'wait_time', default=0.1, help='等待时间 (秒)')
@click.option('--output', 'output_path', default='data/hadamard_response_matrix', help='输出文件路径')
@click.option('--resolution', 'resolution', default='1920,1080', help='SLM分辨率 (宽,高)')
@click.option('--wavelength', default=1064, help='工作波长 (nm)')
@click.option('--mla-index', 'mla_index', type=click.Choice(['512', '540', '600', '768', '1280']), default='512', help='MLA分辨率')
@click.option('--exp-time', 'exp_time', type=float, default=0.0, help='曝光时间 (ms, 0=自动)')
@click.option('--auto-exposure/--no-auto-exposure', 'auto_exposure', default=True, help='启用WFS自动曝光')
@click.option('--high-speed', 'high_speed', is_flag=True, default=False, help='启用高速模式')
@click.option('--use-custom-ref', 'use_custom_ref', is_flag=False, default=False, help='使用自定义参考文件')
@click.option('--pupil-diameter', 'pupil_diameter', type=float, default=2.0, help='瞳孔直径 (mm)')
@click.option('--pupil-center', callback=parse_tuple, default="(0,0)", help='瞳孔中心坐标')
@click.option('--no-inverses', 'compute_inverses', default=True, flag_value=False, help='不计算逆矩阵')
@click.option('--display/--no-display', default=False, help='显示实时pygame显示')
@click.option('--debug', 'debug', is_flag=True, default=None, help='启用调试模式')
def run(
    ctx: click.Context,
    mode_order: int,
    magnitude: float,
    n_averages: int,
    n_cycles: int,
    wait_time: float,
    output_path: str,
    resolution: str,
    wavelength: int,
    mla_index: str,
    exp_time: float,
    auto_exposure: bool,
    high_speed: bool,
    use_custom_ref: bool,
    pupil_diameter: float,
    pupil_center: tuple,
    compute_inverses: bool,
    display: bool,
    debug: bool | None,
):
    """获取Hadamard响应矩阵

    支持 N 次正负交替循环测量 + M 次 WFS 读取取平均 + 方差跟踪 + 逆矩阵计算。

    Hadamard模式数量 = mode_order²，例如:
    - mode_order=8: 64个模式
    - mode_order=16: 256个模式
    - mode_order=32: 1024个模式
    """
    res_width, res_height = map(int, resolution.split(','))

    n_modes = calc_n_hadamard_modes(mode_order)
    logger.info(f"Hadamard响应矩阵校准: mode_order={mode_order}, n_modes={n_modes}")

    if debug is None:
        debug = ctx.parent.obj.get("debug", False) if ctx.parent and ctx.parent.obj else False

    effective_exp_time = 0.0 if auto_exposure else exp_time
    mla_index_enum = MlaRes.from_str(mla_index)

    try:
        hdm = HadamardDM(
            mode_order=mode_order,
            resolution=(res_width, res_height),
            bits=10,
            mask_type='circular',
        )
        logger.info(f"HadamardDM initialized: {hdm.DM_NUM} modes")

        with WFSManager(
            mla_index=mla_index_enum,
            exp_time=effective_exp_time,
            high_speed=high_speed,
            use_custom_ref=use_custom_ref,
            pupil_diameter=pupil_diameter,
            pupil_center=pupil_center,
        ) as wfs:
            logger.info(f"WFS initialized: MLA={mla_index}")

            flat_phase = np.zeros(n_modes)
            flat_pattern = hdm.generate_phase_2pi(flat_phase)
            logger.debug("Set flat reference on WFS")

            if not use_custom_ref:
                wfs.save_user_ref()
                wfs.load_user_ref()

            dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=False)
            s_flat = flatten_slopes(dev_x, dev_y)
            n_measurements = len(s_flat)

            response_matrix = np.zeros((n_measurements, n_modes))
            variance_matrix = np.zeros((n_measurements, n_modes))

            logger.info(f"响应矩阵大小: ({n_measurements}, {n_modes})")

            for mode_idx in range(n_modes):
                logger.debug(f"校准模式 {mode_idx + 1}/{n_modes}")

                coeffs_plus = np.zeros(n_modes)
                coeffs_plus[mode_idx] = magnitude

                coeffs_minus = np.zeros(n_modes)
                coeffs_minus[mode_idx] = -magnitude

                s_plus_all = []
                for _ in range(n_averages):
                    pattern_plus = hdm.generate_phase_2pi(coeffs_plus)
                    sleep(wait_time)
                    dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=False)
                    s_plus_all.append(flatten_slopes(dev_x, dev_y))

                s_plus_mean = np.mean(s_plus_all, axis=0)
                s_plus_var = np.var(s_plus_all, axis=0)

                s_minus_all = []
                for _ in range(n_averages):
                    pattern_minus = hdm.generate_phase_2pi(coeffs_minus)
                    sleep(wait_time)
                    dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=False)
                    s_minus_all.append(flatten_slopes(dev_x, dev_y))

                s_minus_mean = np.mean(s_minus_all, axis=0)
                s_minus_var = np.var(s_minus_all, axis=0)

                if n_cycles > 1:
                    pass
                else:
                    response_matrix[:, mode_idx] = (s_plus_mean - s_minus_mean) / (2 * magnitude)
                    variance_matrix[:, mode_idx] = (s_plus_var + s_minus_var) / (4 * magnitude ** 2)

                if (mode_idx + 1) % 10 == 0:
                    logger.info(f"进度: {mode_idx + 1}/{n_modes} 模式")

            pinv_matrix = None
            lstsq_matrix = None

            if compute_inverses:
                logger.info("计算伪逆矩阵...")
                pinv_matrix = np.linalg.pinv(response_matrix)
                lstsq_matrix = np.linalg.lstsq(response_matrix, np.eye(n_measurements), rcond=None)[0]

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            save_dict = {
                'response_matrix': response_matrix,
                'variance_matrix': variance_matrix,
                'mode_order': mode_order,
                'n_modes': n_modes,
                'n_measurements': n_measurements,
                'magnitude': magnitude,
                'n_averages': n_averages,
            }
            if pinv_matrix is not None:
                save_dict['pinv_matrix'] = pinv_matrix
            if lstsq_matrix is not None:
                save_dict['lstsq_matrix'] = lstsq_matrix

            np.savez(output_file, **save_dict)

            logger.info(f"响应矩阵已保存到: {output_path}.npz")
            logger.info(f"  矩阵形状: {response_matrix.shape}")
            logger.info(f"  平均方差: {np.mean(variance_matrix):.6f}")

    except Exception as e:
        logger.error(f"校准失败: {e}")
        raise


if __name__ == "__main__":
    setup_coredumpy()
    run()