"""Response-matrix calibration CLI runners (DM voltage and Hadamard patterns).

Two commands, one module:

* :func:`run` → ``dm-matrix`` — pokes every DM actuator with push/pull voltage
  perturbations and records WFS subaperture slope deviations. Delegates the
  measurement kernel to
  :func:`ao_shaping.optimizer.wf.dm_response_matrix.calibrate_dm_response_matrix`
  and saves HDF5.
* :func:`hadamard_matrix_run` → ``hadamard-matrix`` — loads successive
  Hadamard patterns through :class:`~ao_shaping.drivers.dm.hadamard_dm.HadamardDM`,
  measures the same slope response, saves ``.npz``.

They were separate modules until 2026-10-04. The merge is real code sharing, not
merely co-location: both open the WFS through an identical
``resolve_wfs(...)`` block, derive exposure and MLA resolution identically, and
refresh the WFS reference the same way. That block is now :func:`_open_wfs`.

The measurement kernels stay separate on purpose. ``dm-matrix`` is a per-actuator
sweep with optional automatic voltage optimisation and a shared ``.h5`` writer;
``hadamard-matrix`` drives a Hadamard coefficient vector and writes ``.npz``.
Note they do NOT share a calibration path: ``dm-matrix`` also accepts
``--mode hadamard``, which is a different implementation again inside the
optimizer layer.

Usage::

    python -m ao_shaping.runners.matrix_runner [opts]                 -> dm-matrix
    python -m ao_shaping.runners.matrix_runner hadamard-matrix [opts] -> hadamard-matrix

Or via the unified CLI::

    python -m ao_shaping.main dm-matrix --voltage 0.2 --n-averages 5
    python -m ao_shaping.main hadamard-matrix --mode-order 16
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import sleep
from typing import Literal, NoReturn, cast

import click
import numpy as np
from loguru import logger

from ao_shaping.drivers.dm._registry import resolve_dm
from ao_shaping.drivers.dm.hadamard_dm import HadamardDM
from ao_shaping.drivers.wfs import MlaRes
from ao_shaping.drivers.wfs._registry import resolve_wfs
from ao_shaping.optimizer.wf.dm_response_matrix import (
    calibrate_dm_response_matrix,
    save_dm_response_matrix,
)
from ao_shaping.runners.runner_common import (
    DmMatrixRunnerParams,
    HadamardMatrixRunnerParams,
    ThorlabWfsDriverParams,
    WfsParams,
)
from ao_shaping.utils.cli.params import with_params
from ao_shaping.utils.io.cli_helpers import (
    get_debug_mode,
    get_timestamp_str,
    setup_coredumpy,
)
from ao_shaping.utils.wavefront.hadamard_calc import calc_n_hadamard_modes
from ao_shaping.utils.wavefront.wfs_utils import (
    flatten_slopes,
    make_actuator_debug_callback,
)

#: Selectors accepted by :func:`main` for `python -m` dispatch.
COMMAND_SELECTORS = ("dm-matrix", "hadamard-matrix")


@contextmanager
def _open_wfs(
    wfs: WfsParams,
    wfs_drv: ThorlabWfsDriverParams,
) -> Iterator[tuple[object, float, MlaRes]]:
    """Open the WFS with the settings shared by both calibration commands.

    Yields:
        ``(wfs_dev, effective_exp_time, mla_index_enum)``. The last two are
        returned because callers need them after the context exits to record a
        device-config snapshot.
    """
    effective_exp_time = 0.0 if wfs_drv.auto_exposure else wfs_drv.exp_time
    mla_index_enum = MlaRes.from_str(wfs_drv.mla_index)
    with resolve_wfs(
        wfs.wfs_type,
        mla_index=mla_index_enum,
        exposure_time=effective_exp_time,
        high_speed=wfs_drv.high_speed,
        use_custom_ref=wfs_drv.use_custom_ref,
        pupil_diameter=wfs.pupil_diameter,
        pupil_center=cast(tuple[float, float], wfs.pupil_center),
    ) as wfs_dev:
        logger.info(f"WFS initialized: MLA={wfs_drv.mla_index}")
        yield wfs_dev, effective_exp_time, mla_index_enum


def _refresh_wfs_reference(wfs_dev: object, use_custom_ref: bool) -> None:
    """Re-take the WFS reference unless the user supplied their own."""
    if not use_custom_ref:
        wfs_dev.save_user_ref()
        wfs_dev.load_user_ref()


# --------------------------------------------------------------------------- #
# dm-matrix
# --------------------------------------------------------------------------- #


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
        with _open_wfs(wfs, wfs_drv) as (wfs_dev, effective_exp_time, mla_index_enum):
            # Set flat DM and refresh WFS reference
            click.echo("Setting flat DM and refreshing WFS reference...")
            voltages_zero = np.zeros(dm.DM_NUM, dtype=np.float64)
            dm.send_voltages(voltages_zero, wait_time_s=params.wait_time)
            sleep(0.3)

            _refresh_wfs_reference(wfs_dev, wfs_drv.use_custom_ref)
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


# --------------------------------------------------------------------------- #
# hadamard-matrix
# --------------------------------------------------------------------------- #


@click.command("hadamard-matrix")
@click.pass_context
@with_params(HadamardMatrixRunnerParams, kw_name="params")
@with_params(WfsParams, kw_name="wfs")
@with_params(ThorlabWfsDriverParams, kw_name="wfs_drv")
def hadamard_matrix_run(
    ctx: click.Context,
    params: HadamardMatrixRunnerParams,
    wfs: WfsParams,
    wfs_drv: ThorlabWfsDriverParams,
) -> None:
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

    try:
        hdm = HadamardDM(
            mode_order=params.mode_order,
            resolution=(res_width, res_height),
            bits=10,
            mask_type='circular',
        )
        logger.info(f"HadamardDM initialized: {hdm.DM_NUM} modes")

        with _open_wfs(wfs, wfs_drv) as (wfs_dev, _effective_exp_time, _mla_index_enum):
            logger.debug("Set flat reference on WFS")

            _refresh_wfs_reference(wfs_dev, wfs_drv.use_custom_ref)

            dev_x, dev_y = wfs_dev.get_spot_deviation(cancel_tile=False)
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
                    hdm.generate_phase_2pi(coeffs_plus)
                    sleep(params.wait_time)
                    dev_x, dev_y = wfs_dev.get_spot_deviation(cancel_tile=False)
                    s_plus_all.append(flatten_slopes(dev_x, dev_y))

                s_plus_mean = np.mean(s_plus_all, axis=0)
                s_plus_var = np.var(s_plus_all, axis=0)

                s_minus_all = []
                for _ in range(params.n_averages):
                    hdm.generate_phase_2pi(coeffs_minus)
                    sleep(params.wait_time)
                    dev_x, dev_y = wfs_dev.get_spot_deviation(cancel_tile=False)
                    s_minus_all.append(flatten_slopes(dev_x, dev_y))

                s_minus_mean = np.mean(s_minus_all, axis=0)
                s_minus_var = np.var(s_minus_all, axis=0)

                # NOTE: pre-existing behaviour, preserved verbatim. When
                # n_cycles > 1 this branch stores NOTHING, so the column stays
                # zero and the saved matrix is all zeros for that mode. Only
                # n_cycles == 1 actually writes a column. Left as-is because
                # fixing it changes measurement output, which is outside the
                # scope of a module merge -- but it should not be mistaken for
                # a working multi-cycle mode.
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


# --------------------------------------------------------------------------- #
# `python -m` entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> NoReturn:
    """Dispatch to a command by selector for `python -m` execution.

    With no selector the module runs ``dm-matrix``, preserving the shortest
    invocation. ``hadamard-matrix`` must be named explicitly.

    Raises:
        SystemExit: Always; the chosen Click command decides the code.
    """
    args = list(sys.argv[1:] if argv is None else argv)

    command: click.Command = run
    label = "dm-matrix"
    if args and args[0] in COMMAND_SELECTORS:
        label = args[0]
        args = args[1:]
        command = hadamard_matrix_run

    try:
        command(args)
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"❌ 运行时错误: {e}")
        logger.exception("{} runner failed", label)
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    setup_coredumpy()
    main()
