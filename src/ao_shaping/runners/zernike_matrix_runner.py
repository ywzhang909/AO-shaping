"""Zernike响应矩阵标定与闭环控制 Runner

包含两个 CLI 入口:
  - run()          : Zernike响应矩阵标定
  - closed_loop_run(): 基于响应矩阵的闭环波前优化

=== run() 执行顺序 ===

  1. _normalize_run_options       — 归一化CLI参数, 计算 n_slm_terms/n_wfs_terms
  2. _setup_debug_callback        — 条件创建调试数据保存回调
  3. ZernikeCalibrationDisplay     — 条件创建pygame实时显示
  4. ZernikeSLM / ThorlabWFS       — 设备初始化 (上下文管理器)
  5. DitheredReference.measure     — 可选: 亚波长抖动参考
  6. _capture_init_state           — 捕获出厂/用户参考状态
    -> _capture_wfs_full_state     —   take_image -> get_spot_deviation -> get_zernike -> get_wavefront
  7. _save_init_state_hdf5         — 保存初始状态到HDF5
  8. _compute_calibration_magnitudes — 计算扰动幅度列表
  9. [循环: 每个幅度]
    a. _make_wavefront_tracking_callback — 创建波前跟踪回调
    b. calibrate_zernike_response_matrix  — 执行标定
    c. _attach_device_config              — 附加硬件配置快照
    d. save_zernike_response_matrix        — 保存标定结果HDF5
    e. _save_wavefront_log_hdf5            — 保存全过程波前跟踪数据
  10. _print_calibration_summary      — 打印结果摘要

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
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Literal

import click
import h5py
import numpy as np
from loguru import logger

from ao_shaping.algorithm.controller import ControlLaw, HardwareConfig, LoopConfig
from ao_shaping.drivers.slm import Santec, ZernikeSLM
from ao_shaping.drivers.wfs import MlaRes, ThorlabWFS
from ao_shaping.optimizer.wf.zernike_response_matrix import (
    DEFAULT_MAGNITUDE,
    DEFAULT_N_AVERAGES,
    DEFAULT_N_CYCLES,
    DEFAULT_N_MAX,
    DEFAULT_WAIT_TIME,
    ZernikeResponseMatrixResult,
    calibrate_zernike_response_matrix,
    save_zernike_response_matrix,
)
from ao_shaping.runners.closed_loop import AOClosedLoop
from ao_shaping.tools.slm.slm_zernike_common import (
    DLL_ZERNIKE_ORDER,
    WFS_ZERNIKE_ORDER,
    collect_device_info,
    effective_cond,
    make_phase,
    measure_wavefront,
    measure_zernike,
    nm_of,
    safe_pinv,
    show_phase,
    um_to_waves,
    wfs_validity,
)
from ao_shaping.utils.cli_helpers import get_timestamp_str, parse_tuple, setup_coredumpy
from ao_shaping.utils.display import ZernikeCalibrationDisplay
from ao_shaping.utils.matrix_utils import calc_n_zernike_terms
from ao_shaping.utils.pattern_helper import PatternHelper
from ao_shaping.utils.wfs_utils import (
    DitheredReference,
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
    magnitude = magnitude or DEFAULT_MAGNITUDE
    n_averages = n_averages or DEFAULT_N_AVERAGES
    n_cycles = n_cycles or DEFAULT_N_CYCLES
    wait_time = wait_time or DEFAULT_WAIT_TIME

    if excluded_tip_tilt:
        cancel_tile = True
        click.echo("Note: --excluded-tip-tilt enabled, auto-setting --cancel-tile")

    if debug is None:
        debug = ctx.parent.obj.get("debug", False) if ctx and ctx.parent and ctx.parent.obj else False

    effective_exp_time = 0.0 if auto_exposure else exp_time
    mla_index_enum = MlaRes.from_str(mla_index)

    n_remove = (1 if excluded_piston else 0) + (2 if excluded_tip_tilt else 0)
    n_slm_terms = calc_n_zernike_terms(n_max) - n_remove
    n_wfs_terms = calc_n_zernike_terms(n_max) - n_remove

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

    def get_arrays(slm_phase, shift_x, shift_y,
                   deviation_x, deviation_y, zernike_coeffs):
        return {
            "slm_phase": slm_phase,
            "deviation_x": deviation_x
                           if deviation_x is not None and len(deviation_x) > 0
                           else None,
            "deviation_y": deviation_y
                           if deviation_y is not None and len(deviation_y) > 0
                           else None,
            "zernike_coeffs": zernike_coeffs
                              if zernike_coeffs is not None
                              and len(zernike_coeffs) > 0
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
        mags = np.linspace(0.1, 0.8, n_magnitudes).round(3).tolist()
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

    Args:
        wfs: ThorlabWFS实例
        cancel_tile: 是否去除WFS tip/tilt
        zernike_order: Zernike阶数

    Returns:
        WfsStateSnapshot 包含所有测量数据
    """
    wfs.take_image()
    dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=cancel_tile)
    # FIXME:读取的数据似乎一样
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
            wfs, cancel_tile=cancel_tile, zernike_order=zernike_order,
        )
        click.echo(f"    偏差: {factory_state.dev_x.shape}, "
                   f"Zernike: {len(factory_state.zernike_coeffs)}, "
                   f"RMS: {factory_state.rms:.4f}")

        # 3. 保存并加载用户参考
        click.echo("  保存用户参考...")
        wfs.save_user_ref()
        wfs.load_user_ref()
        sleep(wait_time)

        # 4. 捕获用户参考状态
        click.echo("  用户参考 (load_user_ref后)...")
        user_state = _capture_wfs_full_state(
            wfs, cancel_tile=cancel_tile, zernike_order=zernike_order,
        )
        click.echo(f"    偏差: {user_state.dev_x.shape}, "
                   f"Zernike: {len(user_state.zernike_coeffs)}, "
                   f"RMS: {user_state.rms:.4f}")

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

        for group_name, state in [("factory_ref", factory_state), ("user_ref", user_state)]:
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

    def _callback(mode_index: int, n_total: int,
                  mean_resp: np.ndarray, var_resp: np.ndarray) -> None:
        """在每个模式校准后记录当前波前"""
        try:
            state = _capture_wfs_full_state(
                wfs, cancel_tile=cancel_tile, zernike_order=zernike_order,
            )
            records.append(state)
        except Exception as e:
            logger.warning(
                f"Failed to capture wavefront after mode {mode_index}: {e}"
            )

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
            click.echo(f"  mag={mag}: mean_var={res.mean_variance:.6f}, "
                       f"shape={res.matrix.shape}")

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


def _mode_ids(n_max: int, excluded_piston: bool = True,
              excluded_tip_tilt: bool = False) -> list[int]:
    """DLL 顺序 m 枚举下要标定的 SLM 模式索引 (1-based)."""
    total = calc_n_zernike_terms(n_max) - 1          # 去掉 DLL[0] 占位
    ids = [i for i in range(1, total + 1)]
    if excluded_piston:
        ids = [i for i in ids if i != 1]
    if excluded_tip_tilt:
        ids = [i for i in ids if i not in (2, 3)]
    return ids


def _apply_mode_rad(slm: Santec, ph: PatternHelper, nm: tuple[int, int],
                    amp_rad: float, radius: float, settle: float) -> np.ndarray:
    """加载单个 Zernike 模式 (系数单位 = **弧度**) 并等待像素翻转.

    ⚠️ 单位约定: `PatternHelper.generate_zernike_polynomial` 与 (2026-09-16 修复后的)
    `ZernikeDM.generate_phase` 的系数单位都是**弧度**, 且保留绝对幅度 —— 系数 ×4
    得到 ×4 的相位 PV。**不要**再传"波长"或归一化后的值。
    """
    phase = make_phase(ph, {nm: float(amp_rad)}, float(radius), n_max=4)
    show_phase(slm, phase, settle)
    return phase


def _apply_coeffs_rad(slm: Santec, ph: PatternHelper,
                      coeffs_rad: dict[tuple[int, int], float],
                      radius: float, settle: float) -> np.ndarray:
    """加载多模式 Zernike 系数组合 (系数单位 = 弧度)."""
    phase = make_phase(ph, coeffs_rad, float(radius), n_max=4)
    show_phase(slm, phase, settle)
    return phase


def calibrate_response_matrix_pushpull(
    slm: Santec, wfs: ThorlabWFS, ph: PatternHelper,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None,
           list["WfsStateSnapshot"]]:
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
    n_wfs = calc_n_zernike_terms(WFS_ZERNIKE_ORDER) - 1      # 66

    matrix = np.zeros((n_wfs, len(modes)), dtype=np.float64)
    variance = np.zeros_like(matrix)
    dev_cols: list[np.ndarray] = []
    wf_records: list[WfsStateSnapshot] = []

    # 子孔径掩膜 (供斜率空间闭环; 失败不影响模态空间)
    mask = None
    try:
        mask, _ = wfs.build_subaperture_mask(n_avg=min(n_averages, 5),
                                             threshold_ratio=0.3, edge_clip=1)
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
                _apply_mode_rad(slm, ph, nm, sign * magnitude_rad,
                                zernike_radius, settle)

                # 光斑有效比门控: 大振幅下 DLL 拟合会崩溃 (|resp| 可暴涨到 10³)
                val = wfs_validity(wfs)
                if not val["ok"]:
                    logger.warning("[{}] {} cyc{} sign{} 光斑有效比 {:.2f} < 阈值 → 剔除",
                                   midx, nm, cyc, sign, val["valid_ratio"])
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
                    debug_cb(mode_index=midx, cycle=cyc, sample=0,
                             slm_phase=slm.get_displayed_phase()[0],
                             shift_x=slm.shift_x, shift_y=slm.shift_y,
                             deviation_x=dx if devs_sign[sign] is not None else None,
                             deviation_y=dy if devs_sign[sign] is not None else None,
                             zernike_coeffs=z, is_plus=(sign > 0))

            zp, zn = readings.get(+1), readings.get(-1)
            if zp is not None and zn is not None:
                # µm → λ, 再除以幅度(rad) → λ/(rad·mode) 的每单位响应
                vecs.append(um_to_waves((zp - zn) / 2.0) / magnitude_rad)
                dp, dn = devs_sign.get(+1), devs_sign.get(-1)
                if dp is not None and dn is not None:
                    devs.append((dp - dn) / 2.0 / magnitude_rad)

        if not vecs:
            click.echo(f"[WARN] 模式 [{midx}] {nm} 无有效响应, 该列置零")
            continue

        arr = np.array(vecs)[:, 1:]                     # 去 index 0 (piston 占位)
        # 逐点幅度合理性剔除 (抓不到拟合崩溃时兜底)
        norms = np.array([float(np.linalg.norm(v)) for v in arr])
        med = float(np.median(norms))
        if med > 0:
            keep = norms <= outlier_factor * med
            if not keep.all():
                logger.warning("[{}] 逐点剔除 {}/{} (|resp|={})",
                               midx, int((~keep).sum()), len(norms),
                               np.round(norms, 3).tolist())
                arr = arr[keep]
        if arr.size == 0:
            continue

        matrix[:, col] = np.median(arr, axis=0)
        variance[:, col] = np.var(arr, axis=0) if len(arr) > 1 else 0.0
        if devs:
            dev_cols.append(np.median(np.array(devs), axis=0))

        top = np.argsort(np.abs(matrix[:, col]))[::-1][:3]
        click.echo(f"[{col + 1:2d}/{len(modes)}] [{midx:2d}] {str(nm):9s} "
                   f"|resp|={np.linalg.norm(matrix[:, col]):7.3f}  主导: "
                   + ", ".join(f"[{int(k) + 1}]{matrix[k, col]:+.3f}" for k in top))

        # 每模式后记录波前 (供 wf_log 产物)
        try:
            wf_records.append(_capture_wfs_full_state(
                wfs, cancel_tile=cancel_tile, zernike_order=n_max))
        except Exception as e:  # noqa: BLE001
            logger.warning("模式 {} 后波前捕获失败: {}", midx, e)

        if progress_cb is not None:
            try:
                progress_cb(col + 1, len(modes), matrix[:, col], variance[:, col])
            except Exception as e:  # noqa: BLE001
                logger.debug("progress_cb 失败: {}", e)

    dev_mat = (np.column_stack(dev_cols) if dev_cols else None)
    return matrix, variance, dev_mat, mask, wf_records


# ==================== CLI: zernike-matrix ====================


@click.command('zernike-matrix')
@click.pass_context
@click.option('--n-max', default=10, help='Zernike最大阶数')
@click.option('--magnitude', default=DEFAULT_MAGNITUDE_RAD, show_default=True, help='扰动幅度 (**弧度**, 0=自动优化; 重写后系数即弧度, 建议 2~5 rad ≈0.3~0.8λ)')
@click.option('--zernike-radius', 'zernike_radius', type=float, default=DEFAULT_ZERNIKE_RADIUS_PX, show_default=True, help='Zernike 归一化半径 px (建议 ≈1.5×光束半径; 闭环矫正必须用同一值)')
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
    zernike_radius: float,
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
    # === 1. 归一化选项 ===
    params = _normalize_run_options(
        n_max, magnitude, n_averages, n_cycles, wait_time, output_path,
        mla_index, auto_exposure, exp_time, excluded_piston,
        excluded_tip_tilt, cancel_tile, debug, ctx,
    )

    # === 2. 设置调试回调 ===
    debug_cb, debug_dir = _setup_debug_callback(output_path, params["debug"])

    # === 3. 条件创建显示 ===
    ui_display = None
    if display:
        ui_display = ZernikeCalibrationDisplay(
            n_wfs_terms=params["n_wfs_terms"],
            n_slm_terms=params["n_slm_terms"],
        )

    try:
        # 2026-09-16 重写: 原 ZernikeSLM 链路在实机出现原生崩溃 (0xC0000005/0xC000041C),
        # 改用与 tools/slm/slm_zernike_response.py 相同的 **Santec + PatternHelper** 直控链路。
        slm = Santec(slm_number=slm_number, wavelength=wavelength,
                     video_mode=0, correction_csv_path=correction_csv_path)
        wfs = ThorlabWFS(
            mla_index=params["mla_index_enum"],
            exposure_time=params["effective_exp_time"],
            high_speed=high_speed,
            use_custom_ref=use_custom_ref,
        )
        slm.open()
        wfs.open()
        ph = PatternHelper(resolution=(slm.Panel_Res[0], slm.Panel_Res[1]))

        # 平移: CLI 非默认值时覆盖, 否则沿用设备配置 (避免把标定好的 shift 覆盖成 0)
        if (int(shift_x), int(shift_y)) != (0, 0):
            slm.set_shift(int(shift_x), int(shift_y))
        click.echo(f"[OK] SLM #{slm._serial_number} {slm.wavelength}nm "
                   f"2π={slm._max_gray} shift=({slm.shift_x},{slm.shift_y})")

        # pupil: 未显式指定 (默认 2.0mm / (0,0)) 时用 optimize_pupil 自动获取并**写回**。
        # ⚠️ 2026-09 教训: 硬编码 pupil 会污染 WFS_ZernikeLsf 拟合 (假 tip/tilt 达 4.6~12.8λ);
        #    且 ThorlabWFS.__init__ 传入的 pupil **会覆盖配置文件中的实测值** (L440-453)。
        pupil_is_default = (float(pupil_diameter) == 2.0
                            and tuple(pupil_center) == (0, 0))
        if pupil_is_default:
            wfs.take_image(n_sample=1, dynamicNoiseCut=True)
            cx, cy, dx, dy = wfs.pupil = wfs.optimize_pupil()
            click.echo(f"[OK] pupil auto (optimize_pupil): "
                       f"center=({cx:.3f},{cy:.3f})mm d=({dx:.3f},{dy:.3f})mm")
        else:
            wfs.pupil = (float(pupil_center[0]), float(pupil_center[1]),
                         float(pupil_diameter), float(pupil_diameter))
            click.echo(f"[OK] pupil (CLI): {wfs.pupil}")

        # 完整设备参数 (随 h5 的 device_config 落盘, 供复现与审计)
        device_info = collect_device_info(slm, wfs, slm_number)
        _ds = device_info.get("slm") or {}
        _dw = device_info.get("wfs") or {}
        click.echo(f"[INFO] 设备: SLM 温度={_ds.get('temperature_c')}°C "
                   f"版本={_ds.get('version')} 矫正={_ds.get('correction_enabled')} | "
                   f"WFS 曝光={_dw.get('exposure_time_ms')}ms MLA={_dw.get('mla_name')} "
                   f"{_dw.get('num_spots_x')}x{_dw.get('num_spots_y')}")

        # === 4. 抖动参考 (可选) ===
        if dither_amp > 0:
            click.echo("[WARN] --dither-amp 原依赖 ZernikeSLM 链路; 重写后的 Santec 直控"
                       "链路暂不支持该功能, 已跳过 (推拉标定本身已含参考设置)")

        # === 5. 初始化状态捕获 + 参考设置 ===
        if not use_custom_ref:
            init_state = _capture_init_state(
                slm, wfs,
                cancel_tile=params["cancel_tile"],
                zernike_order=params["n_max"],
                wait_time=params["wait_time"],
            )
            _save_init_state_hdf5(init_state, output_path)

        # === 6. 计算标定幅度列表 (单位: **弧度**) ===
        magnitudes = _compute_calibration_magnitudes(params["magnitude"], n_magnitudes)

        # === 7. 标定循环 ===
        mode_ids = _mode_ids(params["n_max"], excluded_piston, excluded_tip_tilt)
        results: list[tuple[float | None, ZernikeResponseMatrixResult]] = []

        def _display_cb(i: int, n: int, resp: np.ndarray, var: np.ndarray) -> None:
            """适配 ZernikeCalibrationDisplay.update 的进度回调 (失败不中断标定)."""
            if ui_display is None:
                return
            try:
                ui_display.update(
                    mode_index=i - 1, mode_name=f"mode{i}", response_col=resp,
                    variance_col=var, current_cycle=0,
                    total_cycles=params["n_cycles"], mean_variance=float(np.mean(var)),
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("ui_display.update 失败: {}", e)

        for mag in magnitudes:
            mag_rad = float(mag if mag is not None else DEFAULT_MAGNITUDE_RAD)
            mag_suffix = f"_mag{mag}" if mag is not None else "_auto"
            mag_output_path = f"{output_path}{mag_suffix}"
            click.echo(f"\n=== Calibration run: magnitude={mag_rad} rad "
                       f"({mag_rad / (2 * np.pi):.3f}λ) ===")

            matrix, variance, dev_mat, mask, wf_records = (
                calibrate_response_matrix_pushpull(
                    slm, wfs, ph,
                    n_max=params["n_max"],
                    magnitude_rad=mag_rad,
                    zernike_radius=float(zernike_radius),
                    n_averages=params["n_averages"],
                    n_cycles=params["n_cycles"],
                    settle=params["wait_time"],
                    cancel_tile=params["cancel_tile"],
                    excluded_piston=excluded_piston,
                    excluded_tip_tilt=excluded_tip_tilt,
                    debug_cb=debug_cb,
                    progress_cb=_display_cb,
                )
            )

            result = ZernikeResponseMatrixResult(
                matrix=matrix,
                variance_matrix=variance,
                deviation_response_matrix=dev_mat,
                subaperture_mask=mask,
                n_max=params["n_max"],
                magnitude=mag_rad,                 # 单位: 弧度 (系数即弧度)
                wavelength_nm=int(wavelength),
                n_averages=params["n_averages"],
                n_cycles=params["n_cycles"],
                timestamp=get_timestamp_str(),
                excluded_piston=excluded_piston,
                excluded_tip_tilt=excluded_tip_tilt,
            )
            if compute_inverses:
                try:
                    result.pinv_matrix = safe_pinv(matrix)
                    result.lstsq_matrix = result.pinv_matrix
                except np.linalg.LinAlgError as e:
                    logger.warning(f"逆矩阵计算失败: {e}")

            # 附加硬件配置快照 (含完整 SLM/WFS 设备参数)
            result.device_config = {
                "device": device_info,
                "slm_number": slm_number,
                "wavelength_nm": int(wavelength),
                "shift_x": int(slm.shift_x), "shift_y": int(slm.shift_y),
                "pupil": list(wfs.pupil),
                "mla_index": str(params["mla_index_enum"]),
                "exposure_ms": params["effective_exp_time"],
                "high_speed": bool(high_speed),
                "use_custom_ref": bool(use_custom_ref),
                "correction_csv_path": correction_csv_path,
                "zernike_radius_px": float(zernike_radius),
                "magnitude_rad": mag_rad,
                "slm_mode_ids_dll": mode_ids,
                "zernike_ordering": ("DLL 顺序 m 枚举 (m=-n..+n), 非标准 Noll 1976: "
                                     "[5](2,0)defocus [9](3,1)coma [13](4,0)spherical"),
                "matrix_layout": ("matrix[wfs_coeff_index, slm_mode_index]; "
                                  "行 0..65 ↔ DLL [1..66]"),
                "units": "λ/λ (WFS 系数 µm 经 um_to_waves 换算; 与矫正 w 同单位)",
                "method": ("push-pull ±A (Santec + PatternHelper, 与 "
                           "tools/slm/slm_zernike_response.py 同链路)"),
                "condition_number_effective": effective_cond(matrix),
            }

            save_zernike_response_matrix(result, mag_output_path,
                                         include_inverses=compute_inverses)
            click.echo(f"Saved: {mag_output_path}.h5  "
                       f"(cond_eff={effective_cond(matrix):.2f})")
            _save_wavefront_log_hdf5(wf_records, mag_output_path)
            results.append((mag, result))

        # === 8. 打印结果摘要 ===
        _print_calibration_summary(results, output_path, debug_dir)
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
    import json
    from pathlib import Path

    import matplotlib
    import numpy as np
    matplotlib.use("Agg")
    from datetime import datetime

    import matplotlib.pyplot as plt
    from loguru import logger

    from ao_shaping.drivers.slm import ZernikeSLM
    from ao_shaping.drivers.wfs.ThorlabWFS import MlaRes, ThorlabWFS
    from ao_shaping.optimizer.wf.zernike_response_matrix import (
        load_zernike_response_matrix,
    )
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

    click.echo("\n闭环配置:")
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
            with ThorlabWFS(
                mla_index=MlaRes(device_cfg.mla_index),
                exposure_time=device_cfg.exposure_time,
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
