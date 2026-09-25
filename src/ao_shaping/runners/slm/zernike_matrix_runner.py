"""Zernike响应矩阵标定与闭环控制 Runner

包含两个 CLI 入口:
  - run()          : Zernike响应矩阵标定
  - closed_loop_run(): 基于响应矩阵的闭环波前优化

=== run() 执行顺序 ===

  1. _normalize_run_options       — 归一化CLI参数, 计算 n_slm_terms/n_wfs_terms
  2. _setup_debug_callback        — 条件创建调试数据保存回调
  3. ZernikeCalibrationDisplay     — 条件创建pygame实时显示
  4. Santec / PatternHelper / ThorlabWFS — 设备初始化
     (2026-09-16 重写: 原 ZernikeSLM 链路实机原生崩溃 0xC0000005/0xC000041C,
      改用与 tools/slm/slm_zernike_response.py 相同的 Santec 直控链路)
  5. pupil 启发式                 — CLI 默认 pupil (2.0mm/(0,0)) 视为"未指定"
     -> optimize_pupil() 自动获取并**写回** (硬编码 pupil 会污染 WFS_ZernikeLsf 拟合)
  6. collect_device_info          — 完整 SLM/WFS 设备参数 (随 h5 device_config 落盘)
  7. _capture_init_state          — 捕获出厂/用户参考状态 (display_data flat)
    -> _capture_wfs_full_state    —   take_image -> get_spot_deviation -> get_zernike -> get_wavefront
  8. _save_init_state_hdf5        — 保存初始状态到HDF5
  9. _compute_calibration_magnitudes — 计算扰动幅度列表 (**弧度**, 默认 3.0 rad ≈0.48λ)
  10. [循环: 每个幅度]
    a. calibrate_response_matrix_pushpull — 推拉标定 (Santec + PatternHelper, -A/+A,
        um_to_waves→λ, wfs_validity 门控, 逐点剔除, 统一半径)
    b. 内联 device_config                 — 附加硬件配置快照
    c. save_zernike_response_matrix        — 保存标定结果HDF5
    d. _save_wavefront_log_hdf5            — 保存全过程波前跟踪数据
  11. _print_calibration_summary      — 打印结果摘要

  ✅ 标定/矫正链路 (tools/slm/slm_zernike_common.py 共享):
    - 单位: WFS 系数 µm → um_to_waves → λ; 施加相位 = 系数(λ)×2π = 弧度
      (2026-09-16 修复两个单位 bug: µm/λ 混用放大 1.88×, λ/rad 混用缩小 6.28×)
    - ZernikeDM.generate_phase 已移除 min-max 归一化, 系数即弧度 (×1 与 ×4 PV 比 = 4.0)
    - DLL Zernike 排序 = 顺序 m 枚举 (非标准 Noll 1976): [5](2,0) [9](3,1) [13](4,0)
    - pupil 约定: 必须 optimize_pupil() 并写回; ThorlabWFS.__init__ 的 pupil 会覆盖配置实测值

=== closed_loop_run() 执行顺序 ===

  1. load_zernike_response_matrix  — 加载 .h5 响应矩阵
  2. HardwareConfig.from_dict      — 恢复硬件配置
  3. LoopConfig                    — 构建控制配置
  4. ZernikeSLM / ThorlabWFS       — 设备初始化
  5. from_response_matrix          — 构建 AOClosedLoop 控制器
    -> 斜率空间 (deviation_response_matrix) 或 模态空间 (pinv_matrix)
  6. measure_func  (_measure_wfs_step):
    -> get_spot_deviation -> flatten_slopes -> delta_s -> D_pinv@delta_s -> RMS
  7. apply_func  (_expand_to_noll):
    -> 控制器系数补零 -> 完整Noll顺序 -> zslm.send_zernike
  8. loop.run(control_law)         — 迭代闭环优化
  9. 保存 history.npz / final_coefficients.txt / meta.json / convergence.png

TODO (实机待测试清单, 完整清单见 run() docstring):
    ⚠️ 2026-09-16 重写后的标定内核 (λ/λ 单位 + n_max 透传) 已按已验证的
    tools/slm/slm_zernike_response.py 对齐, 并新增「测完矩阵后自动离线矫正测试 +
    施加相位经 export_correction_csv 导出」。以上改动**尚未上设备实测**,
    待测项逐条列于 `run()` docstring 的 TODO 清单。
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Annotated, Any

import click
import h5py
import numpy as np
from loguru import logger

from ao_shaping.algorithm.signal_processing.controller import (
    ControlLaw,
    HardwareConfig,
    LoopConfig,
)
from ao_shaping.drivers.slm import Santec, ZernikeSLM
from ao_shaping.drivers.wfs import MlaRes, ThorlabWFS
from ao_shaping.optimizer.wf.zernike_response_matrix import (
    DEFAULT_N_AVERAGES,
    DEFAULT_N_CYCLES,
    DEFAULT_N_MAX,
    DEFAULT_WAIT_TIME,
    ZernikeResponseMatrixResult,
    save_zernike_response_matrix,
)
from ao_shaping.optimizer.wf.closed_loop import AOClosedLoop
from ao_shaping.runners.runner_common import (
    ThorlabWfsDriverParams,
    WfsParams,
    ZernikeSlmParams,
    option,
    with_params,
)
from ao_shaping.tools.slm.slm_zernike_common import (
    DLL_ZERNIKE_ORDER,
    WFS_ZERNIKE_ORDER,
    collect_device_info,
    effective_cond,
    export_correction_csv,
    flat_gray,
    make_phase,
    measure_wavefront,
    measure_zernike,
    nm_of,
    safe_pinv,
    show_phase,
    um_to_waves,
    wfs_validity,
)
from ao_shaping.tools.slm.slm_scan_analysis import outlier_mask
from ao_shaping.utils.io.cli_helpers import (
    get_timestamp_str,
    setup_coredumpy,
)
from ao_shaping.utils.image.display import ZernikeCalibrationDisplay
from ao_shaping.utils.wavefront.matrix_utils import calc_n_zernike_terms
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper
from ao_shaping.utils.wavefront.wfs_utils import (
    flatten_slopes,
    make_mode_debug_callback,
)

# ==================== 数据类 ====================


@dataclass
class WfsStateSnapshot:
    """WFS完整测量状态快照 (deviations, Zernike系数, 波前)"""

    dev_x: np.ndarray
    dev_y: np.ndarray
    zernike_coeffs: np.ndarray
    wavefront: np.ndarray
    rms: float


# ==================== 辅助函数 (从原run()中提取) ====================


def _normalize_run_options(
    n_max: int,
    magnitude: float,
    n_averages: int,
    n_cycles: int,
    wait_time: float,
    output_path: str,
    mla_index: str,
    auto_exposure: bool,
    exp_time: float,
    excluded_piston: bool,
    excluded_tip_tilt: bool,
    cancel_tile: bool,
    debug: bool | None,
    ctx: click.Context | None = None,
) -> dict:
    """归一化CLI选项, 返回配置字典"""
    n_max = n_max or DEFAULT_N_MAX
    magnitude = (
        magnitude if magnitude is not None else DEFAULT_MAGNITUDE_RAD
    )  # None → 默认; 0 → 保留给 _compute_... 的自动分支
    n_averages = n_averages or DEFAULT_N_AVERAGES
    n_cycles = n_cycles or DEFAULT_N_CYCLES
    wait_time = wait_time or DEFAULT_WAIT_TIME

    if excluded_tip_tilt:
        cancel_tile = True
        click.echo("Note: --excluded-tip-tilt enabled, auto-setting --cancel-tile")

    if debug is None:
        debug = (
            ctx.parent.obj.get("debug", False)
            if ctx and ctx.parent and ctx.parent.obj
            else False
        )

    effective_exp_time = 0.0 if auto_exposure else exp_time
    mla_index_enum = MlaRes.from_str(mla_index)

    n_remove = (1 if excluded_piston else 0) + (2 if excluded_tip_tilt else 0)
    n_slm_terms = calc_n_zernike_terms(n_max) - n_remove
    # WFS 侧通道数 = Noll 总数 (按 WFS 测量阶数 WFS_ZERNIKE_ORDER=10 → 66, 与 n_max
    # 无关; 矩阵行固定 66, 见 calibrate_response_matrix_pushpull L655)。WFS 不从读数
    # 里移除 piston (仅离线矫正时置零), 因此**不减 n_remove** —— 曾镜像 SLM 侧
    # 写成 -n_remove 导致 65≠66, 同 L643 根因; 也曾按 n_max 计 (n_max<10 时 <66)。
    n_wfs_terms = calc_n_zernike_terms(WFS_ZERNIKE_ORDER)

    return {
        "n_max": n_max,
        "magnitude": magnitude,
        "n_averages": n_averages,
        "n_cycles": n_cycles,
        "wait_time": wait_time,
        "output_path": output_path,
        "effective_exp_time": effective_exp_time,
        "mla_index_enum": mla_index_enum,
        "cancel_tile": cancel_tile,
        "debug": debug,
        "n_remove": n_remove,
        "n_slm_terms": n_slm_terms,
        "n_wfs_terms": n_wfs_terms,
    }


def _setup_debug_callback(
    output_path: str,
    debug: bool,
) -> tuple[Callable | None, Path | None]:
    """创建调试模式的数据保存回调

    Args:
        output_path: 输出路径
        debug: 是否启用调试模式

    Returns:
        (debug_data_callback, debug_data_dir) 或 (None, None)
    """
    if not debug:
        return None, None

    debug_data_dir = Path(output_path) / f"debug_{get_timestamp_str()}"
    debug_data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Debug mode enabled, saving raw data to: {debug_data_dir}")

    def get_arrays(
        slm_phase, shift_x, shift_y, deviation_x, deviation_y, zernike_coeffs
    ):
        return {
            "slm_phase": slm_phase,
            "deviation_x": deviation_x
            if deviation_x is not None and len(deviation_x) > 0
            else None,
            "deviation_y": deviation_y
            if deviation_y is not None and len(deviation_y) > 0
            else None,
            "zernike_coeffs": zernike_coeffs
            if zernike_coeffs is not None and len(zernike_coeffs) > 0
            else None,
        }

    def get_meta(mode_index, cycle, sample, shift_x, shift_y):
        return {
            "shift_x_rad": float(shift_x * np.deg2rad(1)),
            "shift_y_rad": float(shift_y * np.deg2rad(1)),
        }

    debug_data_callback = make_mode_debug_callback(
        debug_data_dir,
        get_arrays=get_arrays,
        get_meta=get_meta,
    )
    return debug_data_callback, debug_data_dir


def _compute_calibration_magnitudes(
    magnitude: float,
    n_magnitudes: int,
) -> list[float | None]:
    """计算需要标定的扰动幅度列表

    Args:
        magnitude: 用户指定的扰动幅度
        n_magnitudes: 自动生成幅度数量 (0=禁用)

    Returns:
        幅度列表, None 表示自动优化
    """
    if n_magnitudes > 0:
        # 多幅度扫描 (单位: **弧度**; 2026-09 重写后 magnitude 语义为弧度)
        mags = np.linspace(2.0, 5.0, n_magnitudes).round(3).tolist()
        click.echo(
            f"Multi-magnitude mode: running {n_magnitudes} calibrations "
            f"with magnitudes {mags}"
        )
        return mags
    elif magnitude == 0:
        return [None]
    else:
        return [magnitude]


# ==================== 新增: WFS状态捕获 ====================


def _capture_wfs_full_state(
    wfs: ThorlabWFS,
    cancel_tile: bool = False,
    zernike_order: int = 10,
) -> WfsStateSnapshot:
    """捕获WFS完整状态: deviations, Zernike系数, 2D波前, RMS

    单帧一致性契约: 一次 `take_image()` 采集后, deviation/Zernike/波前
    三个读数都基于同一帧 (同源数据, 读取结果相同属预期行为——这是闭环
    控制所需的"冻结快照"语义, 不是故障)。若某指标需要独立的新测量,
    调用方必须在两次读数之间再次 `take_image()`。

    Args:
        wfs: ThorlabWFS实例
        cancel_tile: 是否去除WFS tip/tilt
        zernike_order: Zernike阶数

    Returns:
        WfsStateSnapshot 包含所有测量数据
    """
    wfs.take_image()
    dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=cancel_tile)
    # 注: 三个读数同源于上面一次 take_image 的同一帧 (冻结快照, 见 docstring)
    zernike_coeffs = wfs.get_zernike(zernike_order=zernike_order)
    wf_2d, stats = wfs.get_wavefront(cancel_tile=cancel_tile)
    rms = float(stats.get("rms", np.nan)) if stats else np.nan
    return WfsStateSnapshot(
        dev_x=dev_x,
        dev_y=dev_y,
        zernike_coeffs=zernike_coeffs,
        wavefront=wf_2d,
        rms=rms,
    )


def _capture_init_state(
    slm: Santec,
    wfs: ThorlabWFS,
    cancel_tile: bool = False,
    zernike_order: int = 10,
    wait_time: float = 0.5,
) -> tuple[WfsStateSnapshot, WfsStateSnapshot] | None:
    """在全0控制量(SLM平面)下捕获WFS初始状态

    步骤:
        1. 设置SLM平面 (纯平灰度, 直接发 uint16 — 严禁走 create_phase_from_array)
        2. 捕获出厂参考状态 (save_user_ref前)
        3. 保存并加载用户参考
        4. 捕获用户参考状态 (load_user_ref后)

    Args:
        slm: Santec 实例 (2026-09-16 重写: 原为 ZernikeSLM, 改为已验证的 Santec 直控链路)
        wfs: ThorlabWFS实例
        cancel_tile: 是否去除WFS tip/tilt
        zernike_order: Zernike阶数
        wait_time: 等待时间(秒)

    Returns:
        (factory_ref_state, user_ref_state) 或 None (失败时)
    """
    try:
        click.echo("正在捕获WFS初始状态 (SLM平面)...")

        # 1. 设置SLM平面 (纯平灰度直发)
        flat = np.full((slm.Panel_Res[1], slm.Panel_Res[0]), 0, dtype=np.uint16)
        slm.display_data(flat, wait_time_s=wait_time)
        sleep(wait_time)

        # 2. 捕获出厂参考状态
        click.echo("  出厂参考 (save_user_ref前)...")
        factory_state = _capture_wfs_full_state(
            wfs,
            cancel_tile=cancel_tile,
            zernike_order=zernike_order,
        )
        click.echo(
            f"    偏差: {factory_state.dev_x.shape}, "
            f"Zernike: {len(factory_state.zernike_coeffs)}, "
            f"RMS: {factory_state.rms:.4f}"
        )

        # 3. 保存并加载用户参考
        click.echo("  保存用户参考...")
        wfs.save_user_ref()
        wfs.load_user_ref()
        sleep(wait_time)

        # 4. 捕获用户参考状态
        click.echo("  用户参考 (load_user_ref后)...")
        user_state = _capture_wfs_full_state(
            wfs,
            cancel_tile=cancel_tile,
            zernike_order=zernike_order,
        )
        click.echo(
            f"    偏差: {user_state.dev_x.shape}, "
            f"Zernike: {len(user_state.zernike_coeffs)}, "
            f"RMS: {user_state.rms:.4f}"
        )

        logger.info(
            f"Initial state captured: factory RMS={factory_state.rms:.4f}, "
            f"user RMS={user_state.rms:.4f}"
        )
        return factory_state, user_state

    except Exception as e:
        logger.error(f"Failed to capture initial state: {e}")
        return None


def _save_init_state_hdf5(
    init_state: tuple[WfsStateSnapshot, WfsStateSnapshot] | None,
    output_path: str,
) -> None:
    """保存初始化状态到HDF5文件

    文件: {output_path}_init_state.h5
    包含两组: factory_ref/ 和 user_ref/,
    每组有 deviation_x/y, zernike_coeffs, wavefront, rms属性

    Args:
        init_state: _capture_init_state 的返回值, None则跳过
        output_path: 输出基础路径
    """
    if init_state is None:
        return

    factory_state, user_state = init_state
    save_path = Path(f"{output_path}_init_state.h5")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(save_path, "w") as f:
        f.attrs["description"] = "Initial WFS state at flat SLM (all-zero control)"

        for group_name, state in [
            ("factory_ref", factory_state),
            ("user_ref", user_state),
        ]:
            grp = f.create_group(group_name)
            grp.create_dataset("deviation_x", data=state.dev_x)
            grp.create_dataset("deviation_y", data=state.dev_y)
            grp.create_dataset("zernike_coeffs", data=state.zernike_coeffs)
            grp.create_dataset("wavefront", data=state.wavefront)
            grp.attrs["rms"] = state.rms

    logger.info(f"Initial state saved: {save_path}")


# ==================== 新增: 全过程波前跟踪 ====================


def _make_wavefront_tracking_callback(
    wfs: ThorlabWFS,
    cancel_tile: bool = False,
    zernike_order: int = 10,
) -> tuple[Callable[[int, int, np.ndarray, np.ndarray], None], list[WfsStateSnapshot]]:
    """创建校准过程中的波前跟踪回调

    在校准完每个模式后捕获当前WFS完整状态 (SLM处于平面),
    累积到返回的列表中供后续保存.

    Args:
        wfs: ThorlabWFS实例
        cancel_tile: 是否去除WFS tip/tilt
        zernike_order: Zernike阶数

    Returns:
        (callback_function, records_list)
            - callback: 兼容 calibrate_zernike_response_matrix 的 callback 参数
            - records_list: 校准完成后包含每个模式的 WfsStateSnapshot
    """
    records: list[WfsStateSnapshot] = []

    def _callback(
        mode_index: int, n_total: int, mean_resp: np.ndarray, var_resp: np.ndarray
    ) -> None:
        """在每个模式校准后记录当前波前"""
        try:
            state = _capture_wfs_full_state(
                wfs,
                cancel_tile=cancel_tile,
                zernike_order=zernike_order,
            )
            records.append(state)
        except Exception as e:
            logger.warning(f"Failed to capture wavefront after mode {mode_index}: {e}")

    return _callback, records


def _save_wavefront_log_hdf5(
    records: list[WfsStateSnapshot],
    output_path: str,
) -> None:
    """保存全过程波前跟踪数据到HDF5

    文件: {output_path}_wf_log.h5
    每个模式对应 mode_{idx:03d} 组,
    包含 deviation_x/y, zernike_coeffs, wavefront, rms属性

    Args:
        records: _make_wavefront_tracking_callback 返回的记录列表
        output_path: 输出基础路径
    """
    if not records:
        return

    save_path = Path(f"{output_path}_wf_log.h5")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(save_path, "w") as f:
        f.attrs["description"] = "Wavefront history during Zernike calibration"
        f.attrs["n_records"] = len(records)

        for i, rec in enumerate(records):
            grp = f.create_group(f"mode_{i:03d}")
            grp.create_dataset("deviation_x", data=rec.dev_x)
            grp.create_dataset("deviation_y", data=rec.dev_y)
            grp.create_dataset("zernike_coeffs", data=rec.zernike_coeffs)
            grp.create_dataset("wavefront", data=rec.wavefront)
            grp.attrs["rms"] = rec.rms

    logger.info(f"Wavefront history saved: {save_path} ({len(records)} records)")


# ==================== 辅助函数 (从原run()中提取) ====================


def _attach_device_config(
    result: ZernikeResponseMatrixResult,
    slm_number: int,
    wavelength: int,
    n_max: int,
    shift_x: int,
    shift_y: int,
    correction_csv_path: str | None,
    mla_index_enum: MlaRes,
    effective_exp_time: float,
    high_speed: bool,
    use_custom_ref: bool,
    pupil_center: tuple,
    pupil_diameter: float,
) -> ZernikeResponseMatrixResult:
    """在标定结果上附加硬件配置快照

    Args:
        result: 标定结果 (会原地修改)
        以及其他硬件参数

    Returns:
        修改后的 result
    """
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
    return result


def _print_calibration_summary(
    results: list[tuple[float | None, ZernikeResponseMatrixResult]],
    output_path: str,
    debug_data_dir: Path | None,
) -> None:
    """打印标定结果摘要

    Args:
        results: (magnitude, result) 列表
        output_path: 输出基础路径
        debug_data_dir: 调试数据目录 (None表示未启用调试)
    """
    if len(results) == 1:
        click.echo(f"\n响应矩阵已保存到: {output_path}")
    else:
        click.echo(f"\n多幅度标定完成: {len(results)} 组")
        for mag, res in results:
            click.echo(
                f"  mag={mag}: mean_var={res.mean_variance:.6f}, "
                f"shape={res.matrix.shape}"
            )

    last_result = results[-1][1] if results else None
    if last_result is not None:
        click.echo(f"矩阵形状: {last_result.matrix.shape}")
        click.echo(f"平均方差: {last_result.mean_variance:.6f}")
        click.echo(f"最大方差: {last_result.max_variance:.6f}")
        click.echo(
            f"排除piston: {last_result.excluded_piston}, "
            f"排除tip/tilt: {last_result.excluded_tip_tilt}"
        )
        if last_result.condition_number is not None:
            click.echo(f"条件数: {last_result.condition_number:.2e}")

    if debug_data_dir is not None:
        click.echo(f"调试数据已保存到: {debug_data_dir}")


# ==================== 标定内核 (基于 slm_zernike_response 的已验证链路) ====================

#: 默认扰动幅度 (**弧度**) —— 2026-09-16 重写后 magnitude 语义由"归一化任意单位"
#: 改为"弧度": `ZernikeDM.generate_phase` 已移除 min-max 归一化, 系数即弧度。
#: 3.0 rad ≈ 0.48λ, 在 SNR 与线性度之间取平衡。
DEFAULT_MAGNITUDE_RAD = 3.0

#: 默认 Zernike 归一化半径 (px)。实测光束在 SLM 上半径 ≈200px → 取 ≈1.5× 即 300px,
#: 既保证光束落在图案内, 又避免 R≈光束半径时大振幅击穿 WFS 拟合。
DEFAULT_ZERNIKE_RADIUS_PX = 300.0


def _mode_ids(
    n_max: int, excluded_piston: bool = True, excluded_tip_tilt: bool = False
) -> list[int]:
    """DLL 顺序 m 枚举下要标定的 SLM 模式索引 (1-based).

    ⚠️ matrix_utils.calc_n_zernike_terms 返回 **Noll 总数** (66 @ n=10, 含 piston,
    **不含** DLL index-0 占位) —— 恰好是 DLL 有效索引上限。此前误减 1 → ids 2..65,
    漏掉 Noll 66 (n=10 最后一个模式)。与 slm_zernike_response.py L321-323 对齐:
    1..66 → 去 piston → **2..66** (65 个模式)。
    """
    total = calc_n_zernike_terms(n_max)  # 66 = Noll 总数 = DLL 索引上限
    ids = [i for i in range(1, total + 1)]
    if excluded_piston:
        ids = [i for i in ids if i != 1]
    if excluded_tip_tilt:
        ids = [i for i in ids if i not in (2, 3)]
    return ids


def _apply_mode_rad(
    slm: Santec,
    ph: PatternHelper,
    nm: tuple[int, int],
    amp_rad: float,
    radius: float,
    settle: float,
    n_max: int = 4,
) -> np.ndarray:
    """加载单个 Zernike 模式 (系数单位 = **弧度**) 并等待像素翻转.

    ⚠️ 单位约定: `PatternHelper.generate_zernike_polynomial` 与 (2026-09-16 修复后的)
    `ZernikeDM.generate_phase` 的系数单位都是**弧度**, 且保留绝对幅度 —— 系数 ×4
    得到 ×4 的相位 PV。**不要**再传"波长"或归一化后的值。

    Args:
        n_max: 相位生成阶数。**必须与标定一致并透传** —— 曾硬编码 4,
            导致 n≥5 的模式被截断成低阶混叠 (2026-09-16 按已验证脚本修复)。
    """
    phase = make_phase(ph, {nm: float(amp_rad)}, float(radius), n_max=n_max)
    show_phase(slm, phase, settle)
    return phase


def _apply_coeffs_rad(
    slm: Santec,
    ph: PatternHelper,
    coeffs_rad: dict[tuple[int, int], float],
    radius: float,
    settle: float,
    n_max: int = 4,
) -> np.ndarray:
    """加载多模式 Zernike 系数组合 (系数单位 = 弧度).

    Args:
        n_max: 相位生成阶数, 必须与标定时一致 (同 `_apply_mode_rad`)。
    """
    phase = make_phase(ph, coeffs_rad, float(radius), n_max=n_max)
    show_phase(slm, phase, settle)
    return phase


def calibrate_response_matrix_pushpull(
    slm: Santec,
    wfs: ThorlabWFS,
    ph: PatternHelper,
    *,
    n_max: int,
    magnitude_rad: float,
    zernike_radius: float,
    n_averages: int,
    n_cycles: int,
    settle: float,
    cancel_tile: bool = False,
    outlier_factor: float = 3.0,
    excluded_piston: bool = True,
    excluded_tip_tilt: bool = False,
    debug_cb: Callable | None = None,
    progress_cb: Callable | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    list["WfsStateSnapshot"],
]:
    """推拉法 Zernike 响应矩阵标定 (内核与 `tools/slm/slm_zernike_response.py` 一致).

    对每个 SLM Zernike 模式施加 ±A 扰动, 测 WFS Zernike 读数增量:
        ``matrix[wfs_coeff, slm_mode] = Δ(WFS) / Δ(SLM 幅度)``,  单位 **λ/λ**

    与旧实现 (`calibrate_zernike_response_matrix` + `ZernikeSLM`) 的差异:
      - 走 **Santec + PatternHelper** 直控链路 (已验证; 旧链路在实机出现原生崩溃)
      - 单位统一 **µm → λ** (`um_to_waves`), 与闭环矫正的 `w` 同单位
      - 内置 **光斑有效比门控** (`wfs_validity`) + **逐点幅度合理性剔除**
      - 矩阵**统一使用同一 Zernike 半径** (矫正相位按单一 R 生成, 否则归一化不匹配)

    Args:
        magnitude_rad: 扰动幅度, **单位弧度** (建议 2~5 rad ≈ 0.3~0.8λ)
        zernike_radius: Zernike 归一化半径 px (建议 ≈1.5×光束半径)
        settle: 相位下发后的额外等待 s (像素翻转估算之外)

    Returns:
        (matrix, variance, deviation_matrix, subaperture_mask, wf_records)
        其中 matrix 形状 (66, n_modes), 单位 λ/λ; deviation_matrix 为
        (2*n_spots, n_modes) 的展平 [dev_x; dev_y] 响应 (供斜率空间闭环)。
    """
    modes = _mode_ids(n_max, excluded_piston, excluded_tip_tilt)
    # WFS 通道数 = Noll 总数 (66 @ order 10) — get_zernike 返回 67 长数组 (index-0
    # 占位, DLL 填 coeff[1..66]), 取 [1:] 即 66 通道。matrix_utils 版不含占位,
    # 直接用 (不再 -1; 曾误减 → 65 行 vs 66 通道广播失败 L728)。
    n_wfs = calc_n_zernike_terms(WFS_ZERNIKE_ORDER)  # 66

    # 推拉单位幅度 (**λ**): 矩阵列 = Δ(WFS λ) / Δ(SLM 幅度 λ) → **λ/λ**。
    # ⚠️ 与已验证脚本 tools/slm/slm_zernike_response.py 完全一致:
    #   amp_waves = magnitude_rad/(2π), 除的是 amp_waves 而非 magnitude_rad。
    #   (曾误除 magnitude_rad → λ/rad, 差 2π 因子 → 反解系数被放大 6.28×)
    amp_waves = magnitude_rad / (2.0 * np.pi)

    matrix = np.zeros((n_wfs, len(modes)), dtype=np.float64)
    variance = np.zeros_like(matrix)
    dev_cols: list[np.ndarray] = []
    wf_records: list[WfsStateSnapshot] = []

    # 子孔径掩膜 (供斜率空间闭环; 失败不影响模态空间)
    mask = None
    try:
        mask, _ = wfs.build_subaperture_mask(
            n_avg=min(n_averages, 5), threshold_ratio=0.3, edge_clip=1
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"build_subaperture_mask 失败 (跳过斜率空间): {e}")

    for col, midx in enumerate(modes):
        nm = nm_of(midx)
        vecs: list[np.ndarray] = []
        devs: list[np.ndarray] = []
        for cyc in range(max(1, n_cycles)):
            readings: dict[int, np.ndarray | None] = {}
            devs_sign: dict[int, np.ndarray | None] = {}
            for sign in (+1, -1):
                _apply_mode_rad(
                    slm,
                    ph,
                    nm,
                    sign * magnitude_rad,
                    zernike_radius,
                    settle,
                    n_max=n_max,
                )

                # 光斑有效比门控: 大振幅下 DLL 拟合会崩溃 (|resp| 可暴涨到 10³)
                val = wfs_validity(wfs)
                if not val["ok"]:
                    logger.warning(
                        "[{}] {} cyc{} sign{} 光斑有效比 {:.2f} < 阈值 → 剔除",
                        midx,
                        nm,
                        cyc,
                        sign,
                        val["valid_ratio"],
                    )
                    readings[sign] = None
                    devs_sign[sign] = None
                    continue

                z = measure_zernike(wfs, n_avg=n_averages, order=WFS_ZERNIKE_ORDER)
                readings[sign] = z
                try:
                    dx, dy = wfs.get_spot_deviation(cancel_tile=cancel_tile)
                    devs_sign[sign] = np.concatenate([dx.ravel(), dy.ravel()])
                except Exception as e:  # noqa: BLE001
                    logger.debug("get_spot_deviation 失败: {}", e)
                    devs_sign[sign] = None

                if debug_cb is not None and z is not None:
                    debug_cb(
                        mode_index=midx,
                        cycle=cyc,
                        sample=0,
                        slm_phase=slm.get_displayed_phase()[0],
                        shift_x=slm.shift_x,
                        shift_y=slm.shift_y,
                        deviation_x=dx if devs_sign[sign] is not None else None,
                        deviation_y=dy if devs_sign[sign] is not None else None,
                        zernike_coeffs=z,
                        is_plus=(sign > 0),
                    )

            zp, zn = readings.get(+1), readings.get(-1)
            if zp is not None and zn is not None:
                # µm → λ (um_to_waves), 再除以幅度(**λ** = rad/(2π)) → **λ/λ** 每单位响应。
                # 与 slm_zernike_response.py 逐字一致; 曾除以 magnitude_rad (rad) → λ/rad。
                vecs.append(um_to_waves((zp - zn) / 2.0) / amp_waves)
                dp, dn = devs_sign.get(+1), devs_sign.get(-1)
                if dp is not None and dn is not None:
                    devs.append((dp - dn) / 2.0 / amp_waves)

        if not vecs:
            click.echo(f"[WARN] 模式 [{midx}] {nm} 无有效响应, 该列置零")
            continue

        arr = np.array(vecs)[:, 1:]  # 去 index 0 (piston 占位)
        # 逐点幅度合理性剔除 (抓不到拟合崩溃时兜底)
        norms = np.array([float(np.linalg.norm(v)) for v in arr])
        keep = outlier_mask(norms, outlier_factor)
        if not keep.all():
            logger.warning(
                "[{}] 逐点剔除 {}/{} (|resp|={})",
                midx,
                int((~keep).sum()),
                len(norms),
                np.round(norms, 3).tolist(),
            )
            arr = arr[keep]
        if arr.size == 0:
            continue

        matrix[:, col] = np.median(arr, axis=0)
        variance[:, col] = np.var(arr, axis=0) if len(arr) > 1 else 0.0
        if devs:
            dev_cols.append(np.median(np.array(devs), axis=0))

        top = np.argsort(np.abs(matrix[:, col]))[::-1][:3]
        click.echo(
            f"[{col + 1:2d}/{len(modes)}] [{midx:2d}] {str(nm):9s} "
            f"|resp|={np.linalg.norm(matrix[:, col]):7.3f}  主导: "
            + ", ".join(f"[{int(k) + 1}]{matrix[k, col]:+.3f}" for k in top)
        )

        # 每模式后记录波前 (供 wf_log 产物)
        try:
            wf_records.append(
                _capture_wfs_full_state(
                    wfs, cancel_tile=cancel_tile, zernike_order=n_max
                )
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("模式 {} 后波前捕获失败: {}", midx, e)

        if progress_cb is not None:
            try:
                progress_cb(col + 1, len(modes), matrix[:, col], variance[:, col])
            except Exception as e:  # noqa: BLE001
                logger.debug("progress_cb 失败: {}", e)

    dev_mat = np.column_stack(dev_cols) if dev_cols else None
    return matrix, variance, dev_mat, mask, wf_records


# ==================== 离线矫正测试 ====================


def offline_correction_test(
    slm: Santec,
    wfs: ThorlabWFS,
    ph: PatternHelper,
    matrix: np.ndarray,
    modes: list[int],
    *,
    n_max: int,
    zernike_radius: float,
    n_avg: int,
    settle: float,
    device_info: dict,
    matrix_path: str,
    correction_csv_path: str | None = None,
    amplitude_rad: float | None = None,
    out_dir: str = "data/slm_corrections",
    limit_amp: float = 3.0,
) -> dict[str, Any]:
    """测完响应矩阵后自动执行的**离线矫正测试** + 施加矫正相位导出.

    与 `tools/slm/slm_zernike_correction.py` 的 `closed_loop_correct` 同语义,
    但为**单次反解** (不迭代), 且在结束时强制把施加的矫正相位经
    `export_correction_csv` 落盘 (文件名含 序列号/波长/shift/半径/时间 + sidecar).

    流程:
      1. 恢复 WFS **内部参考** (`set_ref_plane(custom=False)`) —— 参考已按用户要求
         在标定前用 SLM 纯 0 相位建过自定义参考; 矫正必须基于内部参考读数
      2. 平场读基线像差 ``w0`` (µm → λ, 67→66, **piston 置零**)
      3. ``c = safe_pinv(matrix) @ w0`` (λ), 截断到 ±limit_amp (λ)
      4. 系数 ×2π → 弧度 → `make_phase` (n_max 与标定一致) → `show_phase`
      5. 复测残差 ``w1`` + 波前 RMS; **模型自检** ``||w0 + M·c|| vs ||w1||`` (比 0.7~1.4)
      6. `export_correction_csv` 保存施加的矫正相位 (sidecar 键已 str 化, 防 json 报错)
      7. 恢复平场, 交还干净状态

    Args:
        matrix: 标定响应矩阵 (λ/λ, 行 0..65 ↔ DLL [1..66])
        modes: 标定模式 DLL 索引列表 (与 matrix 列一一对应)
        n_max: 相位生成阶数 (**必须与标定一致**; 曾硬编码 4 → n≥5 截断)
        zernike_radius: Zernike 归一化半径 px (**必须与标定一致**)
        device_info: `collect_device_info` 结果 (随 sidecar 落盘)
        matrix_path: 标定矩阵 h5 路径 (随 sidecar 记录)
        correction_csv_path: SLM 误差矫正 CSV (仅记录)
        amplitude_rad: 标定扰动幅度 rad (仅记录)
        out_dir: 矫正相位 CSV 输出目录
        limit_amp: 反解系数幅度上限, 单位 **λ** (同 closed_loop_correct 默认 3.0)

    Returns:
        dict: before/after RMS、|w|、模型自检 ratio、反解系数 (λ)、导出路径。
        系数全零时跳过加载与导出。
    """
    import time as _time

    wfs.set_ref_plane(custom=False)
    click.echo(f"\n[离线矫正测试] 恢复内部参考: use_custom_ref={wfs.use_custom_ref}")

    pinv = safe_pinv(matrix)

    # --- 基线: 平场 → 内部参考下的当前像差 ---
    slm.display_data(flat_gray(), wait_time_s=0.5)
    _time.sleep(0.4)
    z0 = measure_zernike(wfs, n_avg=max(n_avg, 3), order=WFS_ZERNIKE_ORDER)
    wf0 = measure_wavefront(wfs, max(n_avg, 3))
    if z0 is None or wf0 is None:
        raise RuntimeError("内部参考下基线测量失败")
    w0 = um_to_waves(z0[1:])
    w0[0] = 0.0  # piston: 参考面整体偏置, 不可矫正且污染反解
    rms0 = float(wf0[1]["rms"])
    click.echo(
        f"  矫正前: RMS={rms0:.4f}λ, PV={wf0[1]['diff']:.4f}λ, "
        f"|w|={np.linalg.norm(w0):.4f}λ (piston 已置零)"
    )

    # --- 反解 (λ) → 截断 → ×2π 弧度 (make_phase 收弧度) ---
    c_lam = np.clip(pinv @ w0, -limit_amp, limit_amp)
    coeffs_lam = {
        nm: float(c_lam[i])
        for i, nm in enumerate(map(nm_of, modes))
        if abs(c_lam[i]) > 1e-4
    }
    n_pos = int((c_lam > 1e-4).sum())
    n_neg = int((c_lam < -1e-4).sum())
    click.echo(
        f"  反解系数 (λ): 正 {n_pos} / 负 {n_neg} / 零 "
        f"{len(modes) - n_pos - n_neg}  |  "
        + ", ".join(
            f"[{modes[i]}]{c_lam[i]:+.3f}"
            for i in range(len(modes))
            if abs(c_lam[i]) > 1e-3
        )
    )

    summary: dict[str, Any] = {
        "before_rms": rms0,
        "before_z_norm": float(np.linalg.norm(w0)),
        "coeffs_lam": coeffs_lam,
    }

    if not coeffs_lam:
        click.echo("  [WARN] 反解系数全为 0 → 跳过加载与导出")
        slm.display_data(flat_gray(), wait_time_s=0.5)
        return summary

    coeffs_rad = {nm: v * 2.0 * np.pi for nm, v in coeffs_lam.items()}
    # ⚠️ ×2π: make_phase 生成的是**弧度**相位; 曾漏换算 → 施加相位缩小 2π = 6.28×
    phase_corr = _apply_coeffs_rad(
        slm, ph, coeffs_rad, zernike_radius, settle, n_max=n_max
    )
    _time.sleep(0.3)

    # --- 复测残差 ---
    z1 = measure_zernike(wfs, n_avg=max(n_avg, 3), order=WFS_ZERNIKE_ORDER)
    wf1 = measure_wavefront(wfs, max(n_avg, 3))
    if z1 is None or wf1 is None:
        raise RuntimeError("矫正后测量失败")
    w1 = um_to_waves(z1[1:])
    rms1 = float(wf1[1]["rms"])

    # --- 模型自检: 预测 ||w0 + M·c|| 必须与实测 ||w1|| 一致 (偏离 → 单位/符号错误) ---
    pred_norm = float(np.linalg.norm(w0 + matrix @ c_lam))
    meas_norm = float(np.linalg.norm(w1))
    ratio = pred_norm / meas_norm if meas_norm > 0 else float("nan")
    flag = (
        "  <== 模型/实测不符, 检查单位/符号/半径!" if not (0.7 <= ratio <= 1.4) else ""
    )
    click.echo(
        f"  矫正后: RMS={rms1:.4f}λ, PV={wf1[1]['diff']:.4f}λ, "
        f"|w|={meas_norm:.4f}λ  (改善 {100 * (1 - rms1 / rms0):.1f}%)"
    )
    click.echo(
        f"  模型自检: 预测|w|={pred_norm:.4f}λ vs 实测 {meas_norm:.4f}λ "
        f"(比 {ratio:.2f}){flag}"
    )
    if flag:
        logger.warning(
            "模型自检失败: 预测 |w|={:.4f} vs 实测 {:.4f} (比 {:.2f})",
            pred_norm,
            meas_norm,
            ratio,
        )

    summary.update(
        {
            "after_rms": rms1,
            "after_z_norm": float(np.linalg.norm(w1)),
            "after_pv": float(wf1[1]["diff"]),
            "improvement_pct": 100 * (1 - rms1 / rms0),
            "model_ratio": ratio,
            "pred_z_norm": pred_norm,
            "n_modes_valid": int((np.abs(c_lam) > 1e-4).sum()),
            "w_before": w0.tolist(),
            "w_after": w1.tolist(),
        }
    )

    # --- 导出施加的矫正相位 (驱动自带 Santec.save_phase_to_csv) ---
    dev = device_info or {}
    meta = {
        "purpose": "Zernike 模式法波前矫正相位 (SLM) — zernike-matrix 离线矫正测试",
        "device": dev,
        "slm_serial": (dev.get("slm") or {}).get("serial_number"),
        "wavelength_nm": (dev.get("slm") or {}).get("wavelength_nm"),
        "shift_x": (dev.get("slm") or {}).get("shift_x"),
        "shift_y": (dev.get("slm") or {}).get("shift_y"),
        "zernike_radius_px": float(zernike_radius),
        "n_max": int(n_max),
        "magnitude_rad": amplitude_rad,
        "matrix_h5": matrix_path,
        "correction_csv_path": correction_csv_path or "",
        "reference": "internal (restored after calibration)",
        "coefficients_lam": {str(k): float(v) for k, v in coeffs_lam.items()},
        "coefficients_rad": {str(k): float(v) for k, v in coeffs_rad.items()},
        "zernike_ordering": (
            "DLL 顺序 m 枚举 (m=-n..+n), 非标准 Noll 1976: "
            "[5](2,0)defocus [9](3,1)coma [13](4,0)spherical"
        ),
        "units": (
            "coefficients_lam 单位 λ (matrix λ/λ, w λ); coefficients_rad 单位 rad "
            "(= lam × 2π, make_phase 输入)"
        ),
        "offline_test": {
            k: v for k, v in summary.items() if k not in ("w_before", "w_after")
        },
        "reproduce": (
            "读取本 CSV 数据区为**弧度** → create_phase_from_array(); "
            "勿走 load_gray_from_csv/csv_to_phase (只接受灰度)"
        ),
    }
    csv_p, json_p = export_correction_csv(
        slm, phase_corr, meta, out_dir=out_dir, prefix="slm_corr"
    )
    click.echo(f"  [OK] 矫正相位已导出: {csv_p}")
    click.echo(f"  [OK]   元数据 sidecar: {json_p}")
    summary["export"] = {"csv": str(csv_p), "json": str(json_p)}

    # 恢复平场, 交还干净状态
    slm.display_data(flat_gray(), wait_time_s=0.5)
    return summary


# ==================== CLI: zernike-matrix ====================


@dataclass
class ZernikeMatrixParams:
    """zernike-matrix 命令的 CLI 参数 (dataclass-click 转换, 2026-09)."""

    n_max: Annotated[int, option("--n-max", help="Zernike最大阶数")] = 10
    magnitude: Annotated[
        float,
        option(
            "--magnitude",
            show_default=True,
            help="扰动幅度 (**弧度**, 0=自动优化; 重写后系数即弧度, 建议 2~5 rad ≈0.3~0.8λ)",
        ),
    ] = DEFAULT_MAGNITUDE_RAD
    zernike_radius: Annotated[
        float,
        option(
            "--zernike-radius",
            type=float,
            show_default=True,
            help="Zernike 归一化半径 px (建议 ≈1.5×光束半径; 闭环矫正必须用同一值)",
        ),
    ] = DEFAULT_ZERNIKE_RADIUS_PX
    n_averages: Annotated[int, option("--n-averages", help="每次WFS读取次数 (M)")] = 3
    n_cycles: Annotated[int, option("--n-cycles", help="正负交替循环次数 (N)")] = 1
    wait_time: Annotated[float, option("--wait", help="等待时间 (秒)")] = 0.2
    output_path: Annotated[
        str, option("--output", help="输出文件路径")
    ] = "data/zernike_response_matrix"
    compute_inverses: Annotated[
        bool, option("--no-inverses", flag_value=False, help="不计算逆矩阵")
    ] = True
    excluded_tip_tilt: Annotated[
        bool,
        option("--excluded-tip-tilt", flag_value=True, help="排除tip/tilt模式 (Z2, Z3)"),
    ] = False
    cancel_tile: Annotated[
        bool,
        option(
            "--cancel-tile",
            is_flag=True,
            help="测量时去除WFS的tip/tilt (对应Thorlabs的cancel_tile功能)",
        ),
    ] = False
    display: Annotated[
        bool, option("--display/--no-display", help="显示实时pygame显示")
    ] = False
    debug: Annotated[
        bool | None,
        option("--debug", is_flag=True, help="启用调试模式 (保存原始测量数据)"),
    ] = None
    auto_optimize_amplitude: Annotated[
        bool,
        option("--auto-optimize/--no-auto-optimize", help="自动优化扰动幅度 (magnitude=0时)"),
    ] = True
    optimize_n_avg: Annotated[
        int, option("--optimize-n-avg", help="幅度优化时的WFS读取次数")
    ] = 10
    n_magnitudes: Annotated[
        int, option("--n-magnitudes", help="自动生成N个不同扰动幅度并分别保存 (0=禁用)")
    ] = 0
    dither_amp: Annotated[
        float, option("--dither-amp", help="亚波长抖动幅度 [λ], 0=禁用 (建议0.02-0.05)")
    ] = 0.0
    correction_csv_path: Annotated[
        str | None,
        option(
            "--correction-csv",
            help="误差矫正CSV文件路径 (如 libs/SLM_DLL_ver.2.51/Wavefront_correction_Data/Wavefront_correction_Data_240236000006(520nm).csv)",
        ),
    ] = None
    # 非 CLI 字段: 函数局部常量 (保持原语义, 不生成 --excluded-piston 选项)
    excluded_piston: bool = True


@click.command("zernike-matrix")
@click.pass_context
@with_params(ZernikeMatrixParams, kw_name="params")
@with_params(ZernikeSlmParams, kw_name="slm_params")
@with_params(ThorlabWfsDriverParams, kw_name="wfs_driver")
@with_params(WfsParams, kw_name="wfs_params")
def run(
    ctx: click.Context,
    params: ZernikeMatrixParams,
    slm_params: ZernikeSlmParams,
    wfs_driver: ThorlabWfsDriverParams,
    wfs_params: WfsParams,
) -> None:
    """获取Zernike响应矩阵

    支持 N 次正负交替循环测量 + M 次 WFS 读取取平均 + 方差跟踪 + 逆矩阵计算。

    多幅度模式 (--n-magnitudes N):
        自动生成N个不同扰动幅度 (2~5 rad)，分别标定并保存。

    调试模式 (--debug):
        保存每次测量的原始数据:
        - SLM相位图 (灰度值, 已应用shift)
        - WFS deviation数据
        - WFS Zernike系数 (averaging前)

    标定内核 (2026-09-16 按已验证的 tools/slm/slm_zernike_response.py 重写):
        - 直控链路: Santec + PatternHelper (与原 ZernikeSLM 链路不同; 旧链路实机原生崩溃)
        - 矩阵单位 **λ/λ**: 列 = Δ(WFS λ) / Δ(SLM 幅度 λ); 幅度 λ = rad/(2π)
        - make_phase **透传 n_max** (不再硬编码 4 → n≥5 模式不再被截断)
        - 测完矩阵后自动执行 offline_correction_test():
            恢复内部参考 → 平场基线 → pinv 反解当前像差 (λ) → ×2π 弧度加载
            → 复测残差 + 模型自检 (‖w0+M·c‖ vs ‖w1‖, 比 0.7~1.4)
            → export_correction_csv 保存施加的矫正相位
              (文件名含 序列号/波长/shift/半径/时间 + sidecar JSON)

    TODO (实机待测试清单 —— 重写后尚未上设备实测):
        [ ] 重标响应矩阵, 检查主导系数: [5](2,0)defocus [9](3,1)coma [13](4,0)spherical
            应与理论形状对应; n≥5 模式不再被截断 (旧 bug: 硬编码 n_max=4)
        [ ] 矩阵量纲: cond_eff 应仍 ≈8 量级; 反解系数幅值与推拉幅度 (λ) 同量级
        [ ] 离线矫正测试: RMS 应有明显改善 (>20%); 模型自检比 ∈ 0.7~1.4
            (偏离即 单位 λ/rad 或 zernike_radius 与标定不一致)
        [ ] export_correction_csv 产物: 数据区为**弧度**; create_phase_from_array 重载
            应还原矫正相位 (勿走 load_gray_from_csv/csv_to_phase 灰度读取管线)
        [ ] closed-loop 用新矩阵: 收敛行为与离线反解一致, 控制系数无 6.28× 缩放
        [ ] --n-magnitudes N: 各幅度结果应线性 (系数 / 幅度 ≈ 常数)
    """
    # === 1. 归一化选项 ===
    norm = _normalize_run_options(
        n_max=params.n_max,
        magnitude=params.magnitude,
        n_averages=params.n_averages,
        n_cycles=params.n_cycles,
        wait_time=params.wait_time,
        output_path=params.output_path,
        mla_index=wfs_driver.mla_index,
        auto_exposure=wfs_driver.auto_exposure,
        exp_time=wfs_driver.exp_time,
        excluded_piston=params.excluded_piston,
        excluded_tip_tilt=params.excluded_tip_tilt,
        cancel_tile=params.cancel_tile,
        debug=params.debug,
        ctx=ctx,
    )

    # === 2. 设置调试回调 ===
    debug_cb, debug_dir = _setup_debug_callback(params.output_path, norm["debug"])

    # === 3. 条件创建显示 ===
    ui_display = None
    if params.display:
        ui_display = ZernikeCalibrationDisplay(
            n_wfs_terms=norm["n_wfs_terms"],
            n_slm_terms=norm["n_slm_terms"],
        )

    try:
        # 2026-09-16 重写: 原 ZernikeSLM 链路在实机出现原生崩溃 (0xC0000005/0xC000041C),
        # 改用与 tools/slm/slm_zernike_response.py 相同的 **Santec + PatternHelper** 直控链路。
        slm = Santec(
            slm_number=slm_params.slm_number,
            wavelength=slm_params.wavelength,
            video_mode=0,
            correction_csv_path=params.correction_csv_path,
        )
        wfs = ThorlabWFS(
            mla_index=norm["mla_index_enum"],
            exposure_time=norm["effective_exp_time"],
            high_speed=wfs_driver.high_speed,
            use_custom_ref=wfs_driver.use_custom_ref,
        )
        slm.open()
        wfs.open()
        ph = PatternHelper(resolution=(slm.Panel_Res[0], slm.Panel_Res[1]))

        # 平移: CLI 非默认值时覆盖, 否则沿用设备配置 (避免把标定好的 shift 覆盖成 0)
        # ⚠️ --shift-x/--shift-y 默认 None → 先归一为 0 再判跳过, 防止 int(None) 崩溃
        shift_cli = (
            int(slm_params.shift_x) if slm_params.shift_x is not None else 0,
            int(slm_params.shift_y) if slm_params.shift_y is not None else 0,
        )
        if shift_cli != (0, 0):
            slm.set_shift(*shift_cli)
        click.echo(
            f"[OK] SLM #{slm._serial_number} {slm.wavelength}nm "
            f"2π={slm._max_gray} shift=({slm.shift_x},{slm.shift_y})"
        )

        # pupil: 未显式指定 (默认 2.0mm / (0,0)) 时用 optimize_pupil 自动获取并**写回**。
        # ⚠️ 2026-09 教训: 硬编码 pupil 会污染 WFS_ZernikeLsf 拟合 (假 tip/tilt 达 4.6~12.8λ);
        #    且 ThorlabWFS.__init__ 传入的 pupil **会覆盖配置文件中的实测值** (L440-453)。
        pupil_is_default = float(wfs_params.pupil_diameter) == 2.7 and tuple(
            wfs_params.pupil_center
        ) == (
            0,
            0,
        )
        if pupil_is_default:
            wfs.take_image(n_sample=1, dynamicNoiseCut=True)
            cx, cy, dx, dy = wfs.pupil = wfs.optimize_pupil()
            click.echo(
                f"[OK] pupil auto (optimize_pupil): "
                f"center=({cx:.3f},{cy:.3f})mm d=({dx:.3f},{dy:.3f})mm"
            )
        else:
            wfs.pupil = (
                float(wfs_params.pupil_center[0]),
                float(wfs_params.pupil_center[1]),
                float(wfs_params.pupil_diameter),
                float(wfs_params.pupil_diameter),
            )
            click.echo(f"[OK] pupil (CLI): {wfs.pupil}")

        # 完整设备参数 (随 h5 的 device_config 落盘, 供复现与审计)
        device_info = collect_device_info(slm, wfs, slm_params.slm_number)
        _ds = device_info.get("slm") or {}
        _dw = device_info.get("wfs") or {}
        click.echo(
            f"[INFO] 设备: SLM 温度={_ds.get('temperature_c')}°C "
            f"版本={_ds.get('version')} 矫正={_ds.get('correction_enabled')} | "
            f"WFS 曝光={_dw.get('exposure_time_ms')}ms MLA={_dw.get('mla_name')} "
            f"{_dw.get('num_spots_x')}x{_dw.get('num_spots_y')}"
        )

        # === 4. 抖动参考 (可选) ===
        if params.dither_amp > 0:
            click.echo(
                "[WARN] --dither-amp 原依赖 ZernikeSLM 链路; 重写后的 Santec 直控"
                "链路暂不支持该功能, 已跳过 (推拉标定本身已含参考设置)"
            )

        # === 5. 初始化状态捕获 + 参考设置 ===
        if not wfs_driver.use_custom_ref:
            init_state = _capture_init_state(
                slm,
                wfs,
                cancel_tile=norm["cancel_tile"],
                zernike_order=norm["n_max"],
                wait_time=norm["wait_time"],
            )
            _save_init_state_hdf5(init_state, params.output_path)

        # === 6. 计算标定幅度列表 (单位: **弧度**) ===
        magnitudes = _compute_calibration_magnitudes(
            norm["magnitude"], params.n_magnitudes
        )

        # === 7. 标定循环 ===
        mode_ids = _mode_ids(
            norm["n_max"], params.excluded_piston, params.excluded_tip_tilt
        )
        results: list[tuple[float | None, ZernikeResponseMatrixResult]] = []
        # 跟踪最后一次标定, 供步骤 8 的离线矫正测试 (用最后矩阵; 测试会恢复 WFS
        # 内部参考, 若放在循环内会破坏后续标定的自定义参考基准)
        last_cal: tuple[float, np.ndarray, str] | None = None

        def _display_cb(i: int, n: int, resp: np.ndarray, var: np.ndarray) -> None:
            """适配 ZernikeCalibrationDisplay.update 的进度回调 (失败不中断标定)."""
            if ui_display is None:
                return
            try:
                ui_display.update(
                    mode_index=i - 1,
                    mode_name=f"mode{i}",
                    response_col=resp,
                    variance_col=var,
                    current_cycle=0,
                    total_cycles=norm["n_cycles"],
                    mean_variance=float(np.mean(var)),
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("ui_display.update 失败: {}", e)

        for mag in magnitudes:
            mag_rad = float(mag if mag is not None else DEFAULT_MAGNITUDE_RAD)
            mag_suffix = f"_mag{mag}" if mag is not None else "_auto"
            mag_output_path = f"{params.output_path}{mag_suffix}"
            click.echo(
                f"\n=== Calibration run: magnitude={mag_rad} rad "
                f"({mag_rad / (2 * np.pi):.3f}λ) ==="
            )

            matrix, variance, dev_mat, mask, wf_records = (
                calibrate_response_matrix_pushpull(
                    slm,
                    wfs,
                    ph,
                    n_max=norm["n_max"],
                    magnitude_rad=mag_rad,
                    zernike_radius=float(params.zernike_radius),
                    n_averages=norm["n_averages"],
                    n_cycles=norm["n_cycles"],
                    settle=norm["wait_time"],
                    cancel_tile=norm["cancel_tile"],
                    excluded_piston=params.excluded_piston,
                    excluded_tip_tilt=params.excluded_tip_tilt,
                    debug_cb=debug_cb,
                    progress_cb=_display_cb,
                )
            )

            result = ZernikeResponseMatrixResult(
                matrix=matrix,
                variance_matrix=variance,
                deviation_response_matrix=dev_mat,
                subaperture_mask=mask,
                n_max=norm["n_max"],
                # 单位: **λ** (waves) —— 与已验证脚本 slm_zernike_response.py 一致;
                # verify_response_matrix / GUI 均按 "λ" 显示。曾误存弧度 (差 2π)。
                magnitude=mag_rad / (2.0 * np.pi),
                wavelength_nm=int(slm_params.wavelength),
                n_averages=norm["n_averages"],
                n_cycles=norm["n_cycles"],
                timestamp=get_timestamp_str(),
                excluded_piston=params.excluded_piston,
                excluded_tip_tilt=params.excluded_tip_tilt,
            )
            if params.compute_inverses:
                try:
                    result.pinv_matrix = safe_pinv(matrix)
                    result.lstsq_matrix = result.pinv_matrix
                except np.linalg.LinAlgError as e:
                    logger.warning(f"逆矩阵计算失败: {e}")

            # 附加硬件配置快照 (含完整 SLM/WFS 设备参数)
            result.device_config = {
                "device": device_info,
                "slm_number": slm_params.slm_number,
                "wavelength_nm": int(slm_params.wavelength),
                "shift_x": int(slm.shift_x),
                "shift_y": int(slm.shift_y),
                "pupil": list(wfs.pupil),
                "mla_index": str(norm["mla_index_enum"]),
                "exposure_ms": norm["effective_exp_time"],
                "high_speed": bool(wfs_driver.high_speed),
                "use_custom_ref": bool(wfs_driver.use_custom_ref),
                "correction_csv_path": params.correction_csv_path,
                "zernike_radius_px": float(params.zernike_radius),
                "magnitude_rad": mag_rad,
                "amplitude_waves": mag_rad / (2.0 * np.pi),
                "slm_mode_ids_dll": mode_ids,
                "zernike_ordering": (
                    "DLL 顺序 m 枚举 (m=-n..+n), 非标准 Noll 1976: "
                    "[5](2,0)defocus [9](3,1)coma [13](4,0)spherical"
                ),
                "matrix_layout": (
                    "matrix[wfs_coeff_index, slm_mode_index]; 行 0..65 ↔ DLL [1..66]"
                ),
                "units": "λ/λ (WFS 系数 µm 经 um_to_waves 换算; 与矫正 w 同单位)",
                "method": (
                    "push-pull ±A (Santec + PatternHelper, 与 "
                    "tools/slm/slm_zernike_response.py 同链路)"
                ),
                "condition_number_effective": effective_cond(matrix),
            }

            save_zernike_response_matrix(
                result, mag_output_path, include_inverses=params.compute_inverses
            )
            click.echo(
                f"Saved: {mag_output_path}.h5  (cond_eff={effective_cond(matrix):.2f})"
            )
            _save_wavefront_log_hdf5(wf_records, mag_output_path)
            results.append((mag, result))
            last_cal = (mag_rad, matrix, f"{mag_output_path}.h5")

        # === 8. 离线矫正测试 (测完响应矩阵后自动执行) ===
        #     恢复内部参考 → 平场基线 → pinv 反解当前像差 → ×2π 弧度加载
        #     → 复测残差 + 模型自检 → export_correction_csv 保存施加的矫正相位。
        #     ⚠️ 需要实机验证: 基线 RMS→残差 RMS 应有明显改善且模型自检比 ∈ 0.7~1.4;
        #     失败多半是单位 (λ/rad) 或 zernike_radius 与标定不一致。
        if last_cal is not None:
            try:
                last_mag_rad, last_matrix, last_matrix_path = last_cal
                offline_summary = offline_correction_test(
                    slm,
                    wfs,
                    ph,
                    last_matrix,
                    mode_ids,
                    n_max=norm["n_max"],
                    zernike_radius=float(params.zernike_radius),
                    n_avg=norm["n_averages"],
                    settle=norm["wait_time"],
                    device_info=device_info,
                    matrix_path=last_matrix_path,
                    correction_csv_path=params.correction_csv_path,
                    amplitude_rad=last_mag_rad,
                )
                if offline_summary.get("export"):
                    click.echo(
                        f"离线矫正: RMS {offline_summary['before_rms']:.4f}λ → "
                        f"{offline_summary['after_rms']:.4f}λ "
                        f"({offline_summary['improvement_pct']:.1f}%)"
                    )
            except Exception as e:  # noqa: BLE001 - 矫正测试失败不使标定整体失败
                logger.error("离线矫正测试失败: {}", e)
                click.echo(f"[WARN] 离线矫正测试失败: {type(e).__name__}: {e}")

        # === 9. 打印结果摘要 ===
        _print_calibration_summary(results, params.output_path, debug_dir)
    finally:
        for dev in (wfs, slm):
            try:
                dev.close()
            except Exception:  # noqa: BLE001
                pass
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

    if (
        result.deviation_response_matrix is not None
        and result.subaperture_mask is not None
    ):
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
            f"使用模态空间响应矩阵: D shape={D.shape}, pinv shape={D_pinv.shape}"
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
        wfs: ThorlabWFS 实例
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
    full[n_pad : n_pad + len(u)] = u
    return full


@dataclass
class ClosedLoopParams:
    """closed-loop 命令的 CLI 参数 (dataclass-click 转换, 2026-09)."""

    # required=True, 无默认值 → 必须放在首位 (dataclass 非默认字段在前)
    load_file: Annotated[
        str, option("--load-file", required=True, help="已保存的响应矩阵 .h5 文件路径")
    ]
    output_path: Annotated[
        str | None,
        option("--output", help="结果保存路径 (默认: 在load-file同目录生成)"),
    ] = None
    control_law: Annotated[
        str,
        option(
            "--control-law",
            type=click.Choice(["pid", "leaky", "qg", "lqg", "mpc", "adaptive"]),
            help="控制律 (默认: leaky)",
        ),
    ] = "leaky"
    gain: Annotated[
        float | None, option("--gain", type=float, help="控制增益覆盖 (控制律依赖)")
    ] = None
    leak: Annotated[
        float | None, option("--leak", type=float, help="泄漏因子覆盖")
    ] = None
    kp: Annotated[float | None, option("--kp", type=float, help="PID比例增益")] = None
    ki: Annotated[float | None, option("--ki", type=float, help="PID积分增益")] = None
    kd: Annotated[float | None, option("--kd", type=float, help="PID微分增益")] = None
    dt: Annotated[
        float, option("--dt", type=float, help="采样周期 [s] (默认: 0.067)")
    ] = 0.067
    rms_target: Annotated[
        float, option("--rms-target", type=float, help="目标RMS [λ] (默认: 0.05)")
    ] = 0.05
    max_iter: Annotated[
        int, option("--max-iter", type=int, help="最大迭代次数 (默认: 100)")
    ] = 100
    delay_steps: Annotated[
        int, option("--delay-steps", type=int, help="延时补偿步数 (默认: 1)")
    ] = 1
    cancel_tile: Annotated[
        bool,
        option("--cancel-tile/--no-cancel-tile", help="测量时去除WFS tip/tilt"),
    ] = False
    display: Annotated[
        bool, option("--display/--no-display", help="显示实时pygame显示")
    ] = False
    debug: Annotated[
        bool, option("--debug", is_flag=True, help="启用调试模式")
    ] = False


@click.command("closed-loop")
@click.pass_context
@with_params(ClosedLoopParams, kw_name="params")
def closed_loop_run(ctx: click.Context, params: ClosedLoopParams) -> None:
    """基于响应矩阵的闭环波前优化

    载入已保存的Zernike响应矩阵 (.h5), 恢复硬件参数,
    使用指定控制律进行闭环优化, 保存优化结果。
    """
    import json
    from pathlib import Path

    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    from datetime import datetime

    import matplotlib.pyplot as plt
    from loguru import logger

    from ao_shaping.drivers.slm import ZernikeSLM
    from ao_shaping.drivers.wfs import MlaRes, ThorlabWFS
    from ao_shaping.optimizer.wf.zernike_response_matrix import (
        load_zernike_response_matrix,
    )
    from ao_shaping.utils.io.cli_helpers import get_timestamp_str

    # 加载响应矩阵
    click.echo(f"加载响应矩阵: {params.load_file}")
    result = load_zernike_response_matrix(params.load_file)
    click.echo(f"  矩阵形状: {result.matrix.shape}")
    click.echo(
        f"  n_max={result.n_max}, excluded_piston={result.excluded_piston}, excluded_tip_tilt={result.excluded_tip_tilt}"
    )

    # 提取硬件配置
    device_cfg = HardwareConfig()
    if result.device_config is not None:
        device_cfg = HardwareConfig.from_dict(result.device_config)
        click.echo(
            f"已恢复硬件配置: SLM#{device_cfg.slm_number}, WL={device_cfg.wavelength}nm, "
            f"shift=({device_cfg.shift_x},{device_cfg.shift_y}), "
            f"MLA={device_cfg.mla_index}, exp={device_cfg.exposure_time}ms"
        )
    else:
        click.echo("警告: 响应矩阵中未找到硬件配置, 将使用默认参数")
        click.echo("建议重新标定以保存完整硬件参数, 或手动指定设备参数")

    # 构建控制配置
    n_modes = result.n_slm_terms
    Q_diag = np.ones(n_modes)
    loop_cfg = LoopConfig(
        n_modes=n_modes,
        dt=params.dt,
        Kp=params.kp if params.kp is not None else 0.5,
        Ki=params.ki if params.ki is not None else 0.3,
        Kd=params.kd if params.kd is not None else 0.05,
        leak=params.leak if params.leak is not None else 0.97,
        Q_diag=Q_diag,
        R_scalar=0.1,
        delay_steps=params.delay_steps,
        rms_target=params.rms_target,
        max_iter=params.max_iter,
        cancel_tile=params.cancel_tile,
    )
    if params.gain is not None:
        loop_cfg.gain_schedule = [(0, params.max_iter, params.gain, loop_cfg.leak)]

    click.echo("\n闭环配置:")
    click.echo(f"  控制律: {params.control_law}, DT={params.dt}s")
    click.echo(f"  目标RMS: {params.rms_target}λ, 最大迭代: {params.max_iter}")

    # 解析控制律枚举
    law_map = {
        "pid": ControlLaw.PID,
        "leaky": ControlLaw.LEAKY_INTEGRATOR,
        "qg": ControlLaw.QUADRATIC_GAUSSIAN,
        "lqg": ControlLaw.LQG,
        "mpc": ControlLaw.PREDICTIVE,
        "adaptive": ControlLaw.ADAPTIVE_GAIN,
    }
    law = law_map[params.control_law]

    # 设置输出路径
    load_path = Path(params.load_file)
    output_path = params.output_path
    if output_path is None:
        output_path = str(
            load_path.parent / f"closed_loop_{load_path.stem}_{get_timestamp_str()}"
        )

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
            click.echo(
                f"  SLM已初始化: shift=({device_cfg.shift_x}, {device_cfg.shift_y})"
            )

            click.echo("初始化WFS...")
            with ThorlabWFS(
                mla_index=MlaRes(device_cfg.mla_index),
                exposure_time=device_cfg.exposure_time,
                high_speed=device_cfg.high_speed,
                use_custom_ref=device_cfg.use_custom_ref,
                pupil_diameter=device_cfg.pupil_diameter,
                pupil_center=tuple(device_cfg.pupil_center),
            ) as wfs:
                click.echo(
                    f"  WFS已初始化: MLA={device_cfg.mla_index}, "
                    f"exp={device_cfg.exposure_time}ms"
                )

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
                    cancel_tile=params.cancel_tile,
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
                click.echo(f"\n启动闭环优化 ({params.control_law})...")
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
                    "load_file": params.load_file,
                    "control_law": params.control_law,
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
                        "dt": params.dt,
                        "Kp": loop_cfg.Kp,
                        "Ki": loop_cfg.Ki,
                        "Kd": loop_cfg.Kd,
                        "leak": loop_cfg.leak,
                        "rms_target": params.rms_target,
                        "max_iter": params.max_iter,
                        "delay_steps": params.delay_steps,
                        "cancel_tile": params.cancel_tile,
                    },
                }
                with open(output_dir / "meta.json", "w") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

                # 生成收敛曲线
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                iterations = np.arange(len(loop_result["rms_history"]))
                axes[0].plot(iterations, loop_result["rms_history"], "b-")
                axes[0].axhline(
                    y=params.rms_target,
                    color="r",
                    linestyle="--",
                    alpha=0.5,
                    label=f"target={params.rms_target}λ",
                )
                axes[0].set_xlabel("Iteration")
                axes[0].set_ylabel("RMS [λ]")
                axes[0].set_title(f"Convergence ({params.control_law})")
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()

                if len(loop_result["a_history"]) > 0:
                    coeffs = np.array(loop_result["a_history"])
                    for i in range(min(5, coeffs.shape[1])):
                        axes[1].plot(iterations, coeffs[:, i], label=f"Z{i + 2}")
                    axes[1].set_xlabel("Iteration")
                    axes[1].set_ylabel("Coefficient [λ]")
                    axes[1].set_title("First 5 Zernike Modes")
                    axes[1].grid(True, alpha=0.3)
                    axes[1].legend()

                fig.tight_layout()
                fig.savefig(output_dir / "convergence.png", dpi=150)
                plt.close(fig)

                click.echo(f"\n结果已保存到: {output_dir}")
                click.echo(
                    f"  RMS: {loop_result['rms_initial']:.4f}λ → {loop_result['rms_final']:.4f}λ"
                )
                click.echo(f"  改善: {loop_result['improvement_db']:.1f} dB")
                click.echo(
                    f"  迭代: {loop_result['n_iter']}, 收敛: {loop_result['converged']}"
                )

    finally:
        if ui_display is not None:
            ui_display.close()


if __name__ == "__main__":
    setup_coredumpy()
    run()
