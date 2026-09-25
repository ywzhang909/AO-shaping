"""DM actuator response matrix calibration CLI runner.

Measures the DM-to-WFS response matrix by poking each actuator with
push-pull voltage perturbations and recording WFS subaperture slope deviations.

Usage:
    python -m ao_shaping.runners.dm_matrix_runner [OPTIONS]

Or via main CLI:
    python -m ao_shaping.main dm-matrix [OPTIONS]

TODO (实机待测试清单, 完整清单见 run() docstring):
    ⚠️ 2026-09-17: sequential/hadamard 双模式标定内核为新增代码, 已通过离线
    仿真测试 (122 passed, 1 hardware-skip) 与合成数据冒烟, 但**尚未上设备
    实测**。DM/WFS 当前不可达 (nlight/micro is_reachable()=False), 待测项
    逐条列于 `run()` docstring 的 TODO 清单。
"""

from __future__ import annotations

from pathlib import Path
from time import sleep
from typing import Literal, cast

import click
import numpy as np
from loguru import logger

from ao_shaping.drivers.dm._registry import resolve_dm
from ao_shaping.drivers.wfs import MlaRes, ThorlabWFS
from ao_shaping.optimizer.wf.dm_response_matrix import (
    calibrate_dm_response_matrix,
    save_dm_response_matrix,
)
from ao_shaping.runners.runner_common import (
    DmMatrixRunnerParams,
    ThorlabWfsDriverParams,
    WfsParams,
    with_params,
)
from ao_shaping.utils.io.cli_helpers import (
    get_debug_mode,
    get_timestamp_str,
    setup_coredumpy,
)
from ao_shaping.utils.wavefront.wfs_utils import make_actuator_debug_callback


@click.command("dm-matrix")
@click.pass_context
@with_params(DmMatrixRunnerParams, kw_name="params")
@with_params(WfsParams, kw_name="wfs")
@with_params(ThorlabWfsDriverParams, kw_name="wfs_drv")
def run(
    ctx: click.Context,
    params: DmMatrixRunnerParams,
    wfs: WfsParams,
    wfs_drv: ThorlabWfsDriverParams,
) -> None:
    """获取DM变形镜响应矩阵

    依次对每个DM单元施加正负电压扰动, 记录WFS子孔径斜率偏差,
    构建 DM -> WFS 的响应矩阵。

    支持 N 次正负交替循环测量 + M 次 WFS 读取取平均 +
    方差跟踪 + 逆矩阵计算 + 自动电压优化。

    调试模式 (--debug):
        保存每次测量的原始WFS deviation数据。

    TODO (实机待测试清单 —— sequential/hadamard 双模式尚未上设备实测):
        [ ] sequential 模式真实 DM+WFS n=5 重跑 (n-averages 10) + 自动电压优化:
            矩阵形状/条件数与仿真一致, 平均方差正常 (无死执行器误报)
        [ ] hadamard 模式自动阶数实测: 与 sequential 矩阵一致性 (相关性/条件数
            同量级), 测量时间显著减少 (所有有效单元同时扰动)
        [ ] hadamard 显式阶数: --hadamard-order 指定 ≥有效单元数的最小 2 的幂
            时正常; 非合法值应报清晰错误
        [ ] 矫正闭环验证: 以标定矩阵 (pinv_matrix) 做波前矫正, RMS 明显改善
        [ ] report3: scripts/generate_dm_response_matrix_report.py 以真实 h5
            生成报告 (图/表齐全), 结果写入 docs/slm 日报
    """
    debug = params.debug
    if debug is None:
        debug = get_debug_mode()

    # Resolve WFS exposure time
    effective_exp_time = 0.0 if wfs_drv.auto_exposure else wfs_drv.exp_time

    # Convert mla_index string to MlaRes enum
    mla_index_enum = MlaRes.from_str(wfs_drv.mla_index)

    # Parse dm_unit_mask if provided
    dm_unit_mask: np.ndarray | None = None
    if params.dm_unit_mask_str is not None:
        try:
            vals = [int(x.strip()) for x in params.dm_unit_mask_str.split(",")]
            dm_unit_mask = np.array(vals, dtype=bool)
        except (ValueError, IndexError):
            raise click.BadParameter(
                "dm_unit_mask must be comma-separated 0/1 values, "
                f"got: {params.dm_unit_mask_str}"
            )

    # Setup debug callback
    debug_data_callback = None
    debug_data_dir = None
    if debug:
        debug_data_dir = Path(params.output_path) / f"debug_{get_timestamp_str()}"
        debug_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug mode enabled, saving raw data to: {debug_data_dir}")

        debug_data_callback = make_actuator_debug_callback(debug_data_dir)

    # Warn about display mode (not yet implemented for DM calibration)
    if params.display:
        click.echo("Note: --display mode is not yet implemented for DM calibration.")

    # DM selection — delegated to drivers.dm._registry.resolve_dm so the logic
    # stays in sync with the wf / pipeline / pib / combined runners.
    dm = resolve_dm(params.dm_type)
    try:
        dm.open()
        with ThorlabWFS(
            mla_index=mla_index_enum,
            exposure_time=effective_exp_time,
            high_speed=wfs_drv.high_speed,
            use_custom_ref=wfs_drv.use_custom_ref,
            pupil_diameter=wfs.pupil_diameter,
            pupil_center=cast(tuple[float, float], wfs.pupil_center),
        ) as wfs_dev:
            # Set flat DM and refresh WFS reference
            click.echo("Setting flat DM and refreshing WFS reference...")
            voltages_zero = np.zeros(dm.DM_NUM, dtype=np.float64)
            dm.send_voltages(voltages_zero, wait_time_s=params.wait_time)
            sleep(0.3)

            if not wfs_drv.use_custom_ref:
                wfs_dev.save_user_ref()
                wfs_dev.load_user_ref()
                logger.debug("WFS reference updated to flat DM.")

            # Run calibration
            click.echo("\n=== Starting DM response matrix calibration ===")
            click.echo(f"  Actuators: {dm.DM_NUM} total")
            click.echo(
                f"  Voltage: {params.disturb_voltage}"
                + (" (auto-optimize)" if params.disturb_voltage == 0 else "")
            )
            click.echo(f"  Averages: {params.n_averages}, Cycles: {params.n_cycles}")
            click.echo(f"  Inverses: {'yes' if params.compute_inverses else 'no'}")
            click.echo(f"  Cancel tile: {params.cancel_tile}")
            click.echo(f"  Auto-optimize voltage: {params.auto_optimize_voltage}")

            result = calibrate_dm_response_matrix(
                dm=dm,
                wfs=wfs_dev,
                disturb_voltage=params.disturb_voltage,
                n_averages=params.n_averages,
                n_cycles=params.n_cycles,
                wait_time=params.wait_time,
                compute_inverses=params.compute_inverses,
                verbose=True,
                dm_unit_mask=dm_unit_mask,
                cancel_tile=params.cancel_tile,
                auto_optimize_voltage=params.auto_optimize_voltage,
                optimize_n_avg=params.optimize_n_avg,
                debug_data_callback=debug_data_callback,
                mode=cast(Literal["sequential", "hadamard"], params.mode),
                hadamard_order=params.hadamard_order,
            )

            # Build device config snapshot
            result.device_config = {
                "mla_index": int(mla_index_enum),
                "exposure_time": effective_exp_time,
                "high_speed": wfs_drv.high_speed,
                "use_custom_ref": wfs_drv.use_custom_ref,
                "pupil_center": list(wfs.pupil_center),
                "pupil_diameter": wfs.pupil_diameter,
                "dm_type": type(dm).__name__,
                "dm_num": dm.DM_NUM if hasattr(dm, "DM_NUM") else 64,
            }

            # Save
            save_dm_response_matrix(
                result, params.output_path, include_inverses=params.compute_inverses
            )
            click.echo(f"\n响应矩阵已保存到: {params.output_path}.h5")
            click.echo(f"  矩阵形状: {result.matrix.shape}")
            click.echo(f"  有效单元数: {result.n_actuators_valid}")
            click.echo(f"  斜率维数: {result.n_slopes}")
            click.echo(f"  平均方差: {result.mean_variance:.6e}")
            click.echo(f"  最大方差: {result.max_variance:.6e}")
            click.echo(f"  校准模式: {params.mode}")
            if result.hadamard_order is not None:
                click.echo(f"  哈达玛阶数: {result.hadamard_order}")
            if result.condition_number is not None:
                click.echo(f"  条件数: {result.condition_number:.2e}")
            if debug_data_dir is not None:
                click.echo(f"  调试数据已保存到: {debug_data_dir}")

    except Exception as e:
        logger.error("Calibration failed: {}", e)
        raise
    finally:
        dm.close()


if __name__ == "__main__":
    setup_coredumpy()
    run()