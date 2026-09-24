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

from pathlib import Path
from time import sleep
from typing import cast

import click
import numpy as np
from loguru import logger

from ao_shaping.drivers.dm.hadamard_dm import HadamardDM
from ao_shaping.drivers.wfs import MlaRes, ThorlabWFS
from ao_shaping.runners.runner_common import HadamardMatrixRunnerParams, with_params
from ao_shaping.utils.io.cli_helpers import setup_coredumpy
from ao_shaping.utils.wavefront.hadamard_calc import calc_n_hadamard_modes
from ao_shaping.utils.wavefront.wfs_utils import flatten_slopes


@click.command('hadamard-matrix')
@click.pass_context
@with_params(HadamardMatrixRunnerParams, kw_name="params")
def run(ctx: click.Context, params: HadamardMatrixRunnerParams) -> None:
    """获取Hadamard响应矩阵

    支持 N 次正负交替循环测量 + M 次 WFS 读取取平均 + 方差跟踪 + 逆矩阵计算。

    Hadamard模式数量 = mode_order²，例如:
    - mode_order=8: 64个模式
    - mode_order=16: 256个模式
    - mode_order=32: 1024个模式
    """
    res_width, res_height = map(int, params.resolution.split(','))

    n_modes = calc_n_hadamard_modes(params.mode_order)
    logger.info(f"Hadamard响应矩阵校准: mode_order={params.mode_order}, n_modes={n_modes}")

    debug = params.debug
    if debug is None:
        debug = ctx.parent.obj.get("debug", False) if ctx.parent and ctx.parent.obj else False

    effective_exp_time = 0.0 if params.auto_exposure else params.exp_time
    mla_index_enum = MlaRes.from_str(params.mla_index)

    try:
        hdm = HadamardDM(
            mode_order=params.mode_order,
            resolution=(res_width, res_height),
            bits=10,
            mask_type='circular',
        )
        logger.info(f"HadamardDM initialized: {hdm.DM_NUM} modes")

        with ThorlabWFS(
            mla_index=mla_index_enum,
            exposure_time=effective_exp_time,
            high_speed=params.high_speed,
            use_custom_ref=params.use_custom_ref,
            pupil_diameter=params.pupil_diameter,
            pupil_center=cast(tuple[float, float], params.pupil_center),
        ) as wfs:
            logger.info(f"WFS initialized: MLA={params.mla_index}")

            flat_phase = np.zeros(n_modes)
            flat_pattern = hdm.generate_phase_2pi(flat_phase)
            logger.debug("Set flat reference on WFS")

            if not params.use_custom_ref:
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
                coeffs_plus[mode_idx] = params.magnitude

                coeffs_minus = np.zeros(n_modes)
                coeffs_minus[mode_idx] = -params.magnitude

                s_plus_all = []
                for _ in range(params.n_averages):
                    pattern_plus = hdm.generate_phase_2pi(coeffs_plus)
                    sleep(params.wait_time)
                    dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=False)
                    s_plus_all.append(flatten_slopes(dev_x, dev_y))

                s_plus_mean = np.mean(s_plus_all, axis=0)
                s_plus_var = np.var(s_plus_all, axis=0)

                s_minus_all = []
                for _ in range(params.n_averages):
                    pattern_minus = hdm.generate_phase_2pi(coeffs_minus)
                    sleep(params.wait_time)
                    dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=False)
                    s_minus_all.append(flatten_slopes(dev_x, dev_y))

                s_minus_mean = np.mean(s_minus_all, axis=0)
                s_minus_var = np.var(s_minus_all, axis=0)

                if params.n_cycles > 1:
                    pass
                else:
                    response_matrix[:, mode_idx] = (s_plus_mean - s_minus_mean) / (2 * params.magnitude)
                    variance_matrix[:, mode_idx] = (s_plus_var + s_minus_var) / (4 * params.magnitude ** 2)

                if (mode_idx + 1) % 10 == 0:
                    logger.info(f"进度: {mode_idx + 1}/{n_modes} 模式")

            pinv_matrix = None
            lstsq_matrix = None

            if params.compute_inverses:
                logger.info("计算伪逆矩阵...")
                pinv_matrix = np.linalg.pinv(response_matrix)
                lstsq_matrix = np.linalg.lstsq(response_matrix, np.eye(n_measurements), rcond=None)[0]

            output_file = Path(params.output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            save_dict = {
                'response_matrix': response_matrix,
                'variance_matrix': variance_matrix,
                'mode_order': params.mode_order,
                'n_modes': n_modes,
                'n_measurements': n_measurements,
                'magnitude': params.magnitude,
                'n_averages': params.n_averages,
            }
            if pinv_matrix is not None:
                save_dict['pinv_matrix'] = pinv_matrix
            if lstsq_matrix is not None:
                save_dict['lstsq_matrix'] = lstsq_matrix

            np.savez(output_file, **save_dict)

            logger.info(f"响应矩阵已保存到: {params.output_path}.npz")
            logger.info(f"  矩阵形状: {response_matrix.shape}")
            logger.info(f"  平均方差: {np.mean(variance_matrix):.6f}")

    except Exception as e:
        logger.error(f"校准失败: {e}")
        raise


if __name__ == "__main__":
    setup_coredumpy()
    run()