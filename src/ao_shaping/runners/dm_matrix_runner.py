"""DM actuator response matrix calibration CLI runner.

Measures the DM-to-WFS response matrix by poking each actuator with
push-pull voltage perturbations and recording WFS subaperture slope deviations.

Usage:
    python -m ao_shaping.runners.dm_matrix_runner [OPTIONS]

Or via main CLI:
    python -m ao_shaping.main dm-matrix [OPTIONS]
"""

from __future__ import annotations

from pathlib import Path
from time import sleep
from typing import Literal

import click
import numpy as np
from loguru import logger

from ao_shaping.drivers.dm.NLight import NLight
from ao_shaping.drivers.wfs import MlaRes, ThorlabWFS
from ao_shaping.optimizer.wf.dm_response_matrix import (
    DEFAULT_DISTURB_VOLTAGE,
    DEFAULT_N_AVERAGES,
    DEFAULT_N_CYCLES,
    DEFAULT_WAIT_TIME,
    calibrate_dm_response_matrix,
    save_dm_response_matrix,
)
from ao_shaping.utils.cli_helpers import (
    get_debug_mode,
    get_timestamp_str,
    parse_tuple,
    setup_coredumpy,
)
from ao_shaping.utils.wfs_utils import make_actuator_debug_callback


@click.command("dm-matrix")
@click.pass_context
@click.option("--voltage", "disturb_voltage", default=DEFAULT_DISTURB_VOLTAGE, type=float,
              help=f"扰动电压 (0=自动优化, 默认: {DEFAULT_DISTURB_VOLTAGE})")
@click.option("--n-averages", "n_averages", default=DEFAULT_N_AVERAGES, type=int,
              help=f"每次WFS读取次数 M (默认: {DEFAULT_N_AVERAGES})")
@click.option("--n-cycles", "n_cycles", default=DEFAULT_N_CYCLES, type=int,
              help=f"正负交替循环次数 N (默认: {DEFAULT_N_CYCLES})")
@click.option("--wait", "wait_time", default=DEFAULT_WAIT_TIME, type=float,
              help=f"电压施加后等待时间 (秒, 默认: {DEFAULT_WAIT_TIME})")
@click.option("--output", "output_path", default="data/dm_response_matrix", type=str,
              help="输出文件路径 (默认: data/dm_response_matrix)")
@click.option("--dm-unit-mask", "dm_unit_mask_str", default=None, type=str,
              help="DM单元掩码 (逗号分隔的0/1列表, 默认: 全部有效, actuator 0禁用)")
@click.option("--mla-index", "mla_index",
              type=click.Choice(["512", "540", "600", "768", "1280"]), default="512",
              help="MLA分辨率 (默认: 512)")
@click.option("--exp-time", "exp_time", type=float, default=0.0,
              help="WFS曝光时间 (ms, 0=自动)")
@click.option("--auto-exposure/--no-auto-exposure", "auto_exposure", default=True,
              help="启用WFS自动曝光 (默认开启)")
@click.option("--high-speed", "high_speed", is_flag=True, default=False,
              help="启用高速模式")
@click.option("--use-custom-ref", "use_custom_ref", is_flag=True, default=False,
              help="使用自定义参考文件")
@click.option("--pupil-diameter", "pupil_diameter", type=float, default=2.0,
              help="瞳孔直径 (mm, 默认: 2.0)")
@click.option("--pupil-center", callback=parse_tuple, default="(0,0)",
              help="瞳孔中心坐标 (默认: (0,0))")
@click.option("--no-inverses", "compute_inverses", default=True, flag_value=False,
              help="不计算逆矩阵")
@click.option("--cancel-tile", "cancel_tile", is_flag=True, default=False,
              help="测量时去除WFS的tip/tilt")
@click.option("--auto-optimize/--no-auto-optimize", "auto_optimize_voltage", default=True,
              help="自动优化每路扰动电压 (voltage=0时, 默认开启)")
@click.option("--optimize-n-avg", "optimize_n_avg", default=10, type=int,
              help="电压优化时的WFS读取次数 (默认: 10)")
@click.option("--display/--no-display", default=False,
              help="显示实时pygame显示 (暂未实现)")
@click.option("--debug", "debug", is_flag=True, default=None,
              help="启用调试模式 (保存原始测量数据)")
def run(
    ctx: click.Context,
    disturb_voltage: float,
    n_averages: int,
    n_cycles: int,
    wait_time: float,
    output_path: str,
    dm_unit_mask_str: str | None,
    mla_index: Literal["512", "540", "600", "768", "1280"],
    exp_time: float,
    auto_exposure: bool,
    high_speed: bool,
    use_custom_ref: bool,
    pupil_diameter: float,
    pupil_center: tuple,
    compute_inverses: bool,
    cancel_tile: bool,
    auto_optimize_voltage: bool,
    optimize_n_avg: int,
    display: bool,
    debug: bool | None,
):
    """获取DM变形镜响应矩阵

    依次对每个DM单元施加正负电压扰动, 记录WFS子孔径斜率偏差,
    构建 DM -> WFS 的响应矩阵。

    支持 N 次正负交替循环测量 + M 次 WFS 读取取平均 +
    方差跟踪 + 逆矩阵计算 + 自动电压优化。

    调试模式 (--debug):
        保存每次测量的原始WFS deviation数据。
    """
    if debug is None:
        debug = get_debug_mode()

    # Resolve WFS exposure time
    effective_exp_time = 0.0 if auto_exposure else exp_time

    # Convert mla_index string to MlaRes enum
    mla_index_enum = MlaRes.from_str(mla_index)

    # Parse dm_unit_mask if provided
    dm_unit_mask: np.ndarray | None = None
    if dm_unit_mask_str is not None:
        try:
            vals = [int(x.strip()) for x in dm_unit_mask_str.split(",")]
            dm_unit_mask = np.array(vals, dtype=bool)
        except (ValueError, IndexError):
            raise click.BadParameter(
                "dm_unit_mask must be comma-separated 0/1 values, "
                f"got: {dm_unit_mask_str}"
            )

    # Setup debug callback
    debug_data_callback = None
    debug_data_dir = None
    if debug:
        debug_data_dir = Path(output_path) / f"debug_{get_timestamp_str()}"
        debug_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug mode enabled, saving raw data to: {debug_data_dir}")

        debug_data_callback = make_actuator_debug_callback(debug_data_dir)

    # Warn about display mode (not yet implemented for DM calibration)
    if display:
        click.echo("Note: --display mode is not yet implemented for DM calibration.")

    try:
        with NLight() as dm:
            with ThorlabWFS(
                mla_index=mla_index_enum,
                exposure_time=effective_exp_time,
                high_speed=high_speed,
                use_custom_ref=use_custom_ref,
                pupil_diameter=pupil_diameter,
                pupil_center=pupil_center,
            ) as wfs:
                # Set flat DM and refresh WFS reference
                click.echo("Setting flat DM and refreshing WFS reference...")
                voltages_zero = np.zeros(dm.DM_NUM, dtype=np.float64)
                dm.send_voltages(voltages_zero, wait_time_s=wait_time)
                sleep(0.3)

                if not use_custom_ref:
                    wfs.save_user_ref()
                    wfs.load_user_ref()
                    logger.debug("WFS reference updated to flat DM.")

                # Run calibration
                click.echo("\n=== Starting DM response matrix calibration ===")
                click.echo(f"  Actuators: {dm.DM_NUM} total")
                click.echo(f"  Voltage: {disturb_voltage}" +
                           (" (auto-optimize)" if disturb_voltage == 0 else ""))
                click.echo(f"  Averages: {n_averages}, Cycles: {n_cycles}")
                click.echo(f"  Inverses: {'yes' if compute_inverses else 'no'}")
                click.echo(f"  Cancel tile: {cancel_tile}")
                click.echo(f"  Auto-optimize voltage: {auto_optimize_voltage}")

                result = calibrate_dm_response_matrix(
                    dm=dm,
                    wfs=wfs,
                    disturb_voltage=disturb_voltage,
                    n_averages=n_averages,
                    n_cycles=n_cycles,
                    wait_time=wait_time,
                    compute_inverses=compute_inverses,
                    verbose=True,
                    dm_unit_mask=dm_unit_mask,
                    cancel_tile=cancel_tile,
                    auto_optimize_voltage=auto_optimize_voltage,
                    optimize_n_avg=optimize_n_avg,
                    debug_data_callback=debug_data_callback,
                )

                # Build device config snapshot
                result.device_config = {
                    "mla_index": int(mla_index_enum),
                    "exposure_time": effective_exp_time,
                    "high_speed": high_speed,
                    "use_custom_ref": use_custom_ref,
                    "pupil_center": list(pupil_center),
                    "pupil_diameter": pupil_diameter,
                }

                # Save
                save_dm_response_matrix(
                    result, output_path, include_inverses=compute_inverses
                )
                click.echo(f"\n响应矩阵已保存到: {output_path}.h5")
                click.echo(f"  矩阵形状: {result.matrix.shape}")
                click.echo(f"  有效单元数: {result.n_actuators_valid}")
                click.echo(f"  斜率维数: {result.n_slopes}")
                click.echo(f"  平均方差: {result.mean_variance:.6e}")
                click.echo(f"  最大方差: {result.max_variance:.6e}")
                if result.condition_number is not None:
                    click.echo(f"  条件数: {result.condition_number:.2e}")
                if debug_data_dir is not None:
                    click.echo(f"  调试数据已保存到: {debug_data_dir}")

    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise


if __name__ == "__main__":
    setup_coredumpy()
    run()
