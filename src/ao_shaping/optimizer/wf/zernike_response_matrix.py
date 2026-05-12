"""Zernike响应矩阵校准模块

使用ZernikeSLM逐一加载各阶Zernike相位，测量对应的Thorlab WFS Zernike系数，
建立SLM Zernike命令到WFS响应的响应矩阵。

支持:
- N次正负交替循环测量
- M次WFS读取取平均
- 方差计算作为稳定性指标
- SVD伪逆和最小二乘逆计算
- 可视化报告生成

Example:
    >>> from ao_shaping.optimizer.wf.zernike_response_matrix import calibrate_zernike_response_matrix
    >>> from ao_shaping.drivers.slm import ZernikeSLM
    >>> from ao_shaping.drivers.wfs import ThorlabWFS
    >>>
    >>> with ZernikeSLM(slm_number=1, wavelength=1064, n_max=10) as zslm:
    ...     with ThorlabWFS() as wfs:
    ...         result = calibrate_zernike_response_matrix(zslm, wfs, n_max=10, n_cycles=3, n_averages=5)
    ...         print(f"Matrix shape: {result.matrix.shape}")
"""

from __future__ import annotations

import json
import h5py

import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np
from loguru import logger
from tqdm import tqdm

from ao_shaping.utils.matrix_utils import (
    calc_n_zernike_terms,
    compute_lstsq,
    compute_pinv,
)
from ao_shaping.utils.wfs_utils import flatten_slopes
from ao_shaping.utils.display import ZernikeCalibrationDisplay


if TYPE_CHECKING:
    from ao_shaping.drivers.slm.zernike_slm import ZernikeSLM
    from ao_shaping.drivers.wfs.thorlab_wfs import WFSManager
    from ao_shaping.utils.display import ZernikeCalibrationDisplay


# 默认参数
DEFAULT_N_MAX = 10
DEFAULT_MAGNITUDE = 0.5  # 波长单位
DEFAULT_N_AVERAGES = 20  # 每次WFS读取次数
DEFAULT_N_CYCLES = 1  # 正负交替循环次数
DEFAULT_WAIT_TIME = 0.1  # 秒


@dataclass
class ZernikeResponseMatrixResult:
    """Zernike响应矩阵校准结果"""

    matrix: np.ndarray  # 响应矩阵 (n_wfs_terms, n_slm_terms)
    variance_matrix: np.ndarray  # 方差矩阵 (n_wfs_terms, n_slm_terms)
    deviation_response_matrix: np.ndarray | None = None  # 子孔径斜率响应矩阵 (2*n_spots, n_slm_terms), 每列包含展平后的[dev_x; dev_y]
    subaperture_mask: np.ndarray | None = None  # 有效子孔径掩膜 2D bool数组
    n_max: int = 10  # Zernike最大阶数
    magnitude: float = 0.5  # 校准时使用的幅度 (波长)
    wavelength_nm: int = 1064  # 工作波长 (nm)
    n_averages: int = 20  # 每次WFS读取次数 (M)
    n_cycles: int = 1  # 正负交替循环次数 (N)
    timestamp: str = ""  # 时间戳
    excluded_piston: bool = True  # 是否排除piston
    excluded_tip_tilt: bool = False  # 是否排除tip/tilt

    # 硬件配置快照 (用于闭环优化恢复)
    device_config: dict | None = None  # 包含 shift_x/y, mla_index, exposure_time, pupil 等

    # 可选的逆矩阵
    pinv_matrix: np.ndarray | None = None
    lstsq_matrix: np.ndarray | None = None
    amplitude_optimization: dict | None = None  # 幅度优化结果: {mode_idx: {test_amps, responses, linearity, optimal}}

    @property
    def n_wfs_terms(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_slm_terms(self) -> int:
        return self.matrix.shape[1]

    @property
    def slm_noll_terms(self) -> int:
        """SLM侧包含的Noll项数 (排除piston/tip-tilt)"""
        n_remove = (1 if self.excluded_piston else 0) + (2 if self.excluded_tip_tilt else 0)
        return calc_n_zernike_terms(self.n_max) - n_remove

    @property
    def mean_variance(self) -> float:
        """平均方差"""
        return float(np.mean(self.variance_matrix))

    @property
    def max_variance(self) -> float:
        """最大方差"""
        return float(np.max(self.variance_matrix))

    @property
    def condition_number(self) -> float | None:
        """矩阵条件数 (如果已计算逆矩阵)"""
        if self.pinv_matrix is not None:
            return float(np.linalg.cond(self.matrix))
        return None

    def to_dict(self) -> dict:
        """转换为字典用于JSON序列化"""
        d = asdict(self)
        d["matrix"] = self.matrix.tolist()
        d["variance_matrix"] = self.variance_matrix.tolist()
        if self.deviation_response_matrix is not None:
            d["deviation_response_matrix"] = self.deviation_response_matrix.tolist()
        if self.subaperture_mask is not None:
            d["subaperture_mask"] = self.subaperture_mask.tolist()
        if self.pinv_matrix is not None:
            d["pinv_matrix"] = self.pinv_matrix.tolist()
        if self.lstsq_matrix is not None:
            d["lstsq_matrix"] = self.lstsq_matrix.tolist()
        if self.amplitude_optimization is not None:
            d["amplitude_optimization"] = self.amplitude_optimization
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ZernikeResponseMatrixResult:
        """从字典加载"""
        d = d.copy()
        d["matrix"] = np.array(d["matrix"])
        d["variance_matrix"] = np.array(d["variance_matrix"])
        if "deviation_response_matrix" in d and d["deviation_response_matrix"] is not None:
            d["deviation_response_matrix"] = np.array(d["deviation_response_matrix"])
        else:
            d["deviation_response_matrix"] = None
        if "subaperture_mask" in d and d["subaperture_mask"] is not None:
            d["subaperture_mask"] = np.array(d["subaperture_mask"])
        else:
            d["subaperture_mask"] = None
        if "pinv_matrix" in d and d["pinv_matrix"] is not None:
            d["pinv_matrix"] = np.array(d["pinv_matrix"])
        else:
            d["pinv_matrix"] = None
        if "lstsq_matrix" in d and d["lstsq_matrix"] is not None:
            d["lstsq_matrix"] = np.array(d["lstsq_matrix"])
        else:
            d["lstsq_matrix"] = None
        if "amplitude_optimization" in d and d["amplitude_optimization"] is not None:
            d["amplitude_optimization"] = d["amplitude_optimization"]
        else:
            d["amplitude_optimization"] = None
        return cls(**d)


def set_slm_flat(slm) -> None:
    """设置SLM为平相位"""
    zero_phase = np.zeros((slm.Panel_Res[1], slm.Panel_Res[0]), dtype=np.uint16)
    slm.display_data(zero_phase)


def _optimize_perturbation_amplitude(
    zslm: ZernikeSLM,
    wfs: WFSManager,
    mode_idx: int,
    test_amps: np.ndarray,
    n_avg: int = 20,
    zernike_order: int = 10,
    cancel_tile: bool = False,
) -> tuple[float, dict]:
    """Find optimal perturbation amplitude for a Zernike mode.

    Uses linearity criterion: find amplitude where response magnitude vs
    perturbation ratio is most stable (minimum second derivative).

    Args:
        zslm: ZernikeSLM instance
        wfs: Thorlab WFS instance
        mode_idx: Mode index (0-based, excluding piston/tip-tilt)
        test_amps: Array of amplitudes to test
        n_avg: Number of WFS readings per amplitude
        zernike_order: WFS Zernike order

    Returns:
        tuple: (optimal_amplitude, diagnostics_dict)
            - optimal_amplitude: Best amplitude found
            - diagnostics: dict with test_amps, responses, linearity scores, best_idx
    """
    noll_offset = 3  # exclude piston + tip/tilt
    n_full = wfs.calc_n_zernike_terms(zernike_order)
    coeffs = np.zeros(n_full, dtype=np.float64)
    coeffs[mode_idx + noll_offset] = 1.0

    responses = []
    for a in test_amps:
        set_slm_flat(zslm._slm)
        time.sleep(0.05)

        s_pos = measure_zernike_mode_response(
            zslm, wfs, coeffs * a, a, n_averages=n_avg, n_cycles=1,
            excluded_piston=True, excluded_tip_tilt=True, zernike_order=zernike_order,
            mode_index=mode_idx, cancel_tile=cancel_tile,
        )[0]

        set_slm_flat(zslm._slm)
        time.sleep(0.05)

        s_neg = measure_zernike_mode_response(
            zslm, wfs, coeffs * (-a), a, n_averages=n_avg, n_cycles=1,
            excluded_piston=True, excluded_tip_tilt=True, zernike_order=zernike_order,
            mode_index=mode_idx, cancel_tile=cancel_tile,
        )[0]

        resp = np.linalg.norm(s_pos - s_neg) / (2 * a)
        responses.append(resp)

    responses = np.array(responses)

    linearity = []
    for i in range(1, len(test_amps) - 1):
        k1 = (responses[i] - responses[i - 1]) / (test_amps[i] - test_amps[i - 1])
        k2 = (responses[i + 1] - responses[i]) / (test_amps[i + 1] - test_amps[i])
        linearity.append(abs(k1 - k2))

    best_idx = np.argmin(linearity) + 1
    optimal = float(test_amps[best_idx])

    diagnostics = {
        "test_amps": test_amps.copy(),
        "responses": responses.copy(),
        "linearity": np.array(linearity),
        "best_idx": int(best_idx),
        "optimal_amplitude": optimal,
    }

    logger.info(
        f"Optimized amplitude for mode {mode_idx}: {optimal:.4f}λ "
        f"(test range: [{test_amps[0]:.2f}, {test_amps[-1]:.2f}])"
    )

    return optimal, diagnostics


def measure_zernike_mode_response(
    zslm: ZernikeSLM,
    wfs: WFSManager,
    zernike_coeffs: np.ndarray,
    magnitude: float,
    n_averages: int = DEFAULT_N_AVERAGES,
    n_cycles: int = DEFAULT_N_CYCLES,
    wait_time: float = DEFAULT_WAIT_TIME,
    excluded_piston: bool = True,
    excluded_tip_tilt: bool = False,
    zernike_order: int = DEFAULT_N_MAX,
    mode_index: int = -1,
    debug_data_callback: Callable | None = None,
    cancel_tile: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """测量单个Zernike模式的响应 (多次循环)

    使用正负扰动测量N次循环，以消除偏置和系统误差:
    response = (response_plus - response_minus) / (2 * coeff_value)

    Args:
        zslm: ZernikeSLM实例
        wfs: Thorlab WFS实例
        coefficients: Zernike系数数组 (numpy格式，包含或不包含piston，根据noll顺序排列)
        coeff_value: 扰动系数值 (用于归一化)
        n_averages: 每次WFS读取次数 (M)
        n_cycles: 正负交替循环次数 (N)
        wait_time: 等待时间 (秒)
        excluded_piston: 是否排除piston (Z1)
        excluded_tip_tilt: 是否排除tip/tilt (Z2, Z3)
        zernike_order: WFS Zernike拟合阶数 (最大10)
        debug_data_callback: 调试数据回调，签名为:
            callback(mode_index: int, cycle: int, sample: int, 
                     slm_phase: np.ndarray, shift_x: int, shift_y: int,
                     deviation_x: np.ndarray, deviation_y: np.ndarray,
                     zernike_coeffs: np.ndarray, is_plus: bool)
            仅在debug模式且回调非None时调用

    Returns:
        tuple: (mean_response, variance_response, mean_deviation, variance_deviation)
            - mean_response: 平均Zernike响应向量
            - variance_response: Zernike响应方差向量
            - mean_deviation: 平均子孔径斜率响应 (flattened concat of dev_x and dev_y)
            - variance_deviation: 子孔径斜率响应方差
    """
    def measure_once(coeffs: np.ndarray, mode_idx: int, cycle: int, is_plus: bool) -> tuple[np.ndarray, np.ndarray]:
        """单次测量 (M次WFS读取取平均)"""
        phase_gray = zslm.send_zernike(coeffs)
        time.sleep(wait_time)

        responses = []
        deviations = []
        for sample_idx in range(n_averages):
            zernike_coeffs = wfs.get_zernike(zernike_order=zernike_order)
            responses.append(zernike_coeffs)

            dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=cancel_tile)
            deviations.append((dev_x, dev_y))

            if debug_data_callback is not None:
                debug_data_callback(
                    mode_index=mode_idx,
                    cycle=cycle,
                    sample=sample_idx,
                    slm_phase=phase_gray,
                    shift_x=zslm.shift_x,
                    shift_y=zslm.shift_y,
                    deviation_x=dev_x,
                    deviation_y=dev_y,
                    zernike_coeffs=zernike_coeffs.copy() if zernike_coeffs is not None else np.array([]),
                    is_plus=is_plus,
                )

        if len(responses) > 5:
            arr = np.array(responses)
            sorted_arr = np.sort(arr, axis=0)
            mean_resp = np.mean(sorted_arr[1:-1], axis=0)
        else:
            mean_resp = np.mean(responses, axis=0)

        dev_x_arr = np.array([d[0] for d in deviations])
        dev_y_arr = np.array([d[1] for d in deviations])
        if len(deviations) > 5:
            sorted_x = np.sort(dev_x_arr, axis=0)
            sorted_y = np.sort(dev_y_arr, axis=0)
            mean_dev_x = np.mean(sorted_x[1:-1], axis=0)
            mean_dev_y = np.mean(sorted_y[1:-1], axis=0)
        else:
            mean_dev_x = np.mean(dev_x_arr, axis=0)
            mean_dev_y = np.mean(dev_y_arr, axis=0)

        mean_dev = flatten_slopes(mean_dev_x, mean_dev_y)
        return mean_resp, mean_dev

    all_responses = []
    all_deviations = []

    for cycle in range(n_cycles):
        response_plus, dev_plus = measure_once(+zernike_coeffs, mode_index, cycle, True)
        set_slm_flat(zslm._slm)
        time.sleep(wait_time)

        response_minus, dev_minus = measure_once(-zernike_coeffs, mode_index, cycle, False)
        set_slm_flat(zslm._slm)
        time.sleep(wait_time)

        response = (response_plus - response_minus) / (2 * magnitude)
        deviation = (dev_plus - dev_minus) / (2 * magnitude)

        start_idx = 0
        if excluded_piston:
            start_idx += 1
        if excluded_tip_tilt:
            start_idx += 2

        all_responses.append(response[start_idx:])
        all_deviations.append(deviation)

    all_responses = np.array(all_responses)
    all_deviations = np.array(all_deviations)

    mean_response = np.mean(all_responses, axis=0)
    variance_response = np.var(all_responses, axis=0)
    mean_deviation = np.mean(all_deviations, axis=0)
    variance_deviation = np.var(all_deviations, axis=0)

    return mean_response, variance_response, mean_deviation, variance_deviation

def calibrate_zernike_response_matrix(
    zslm: ZernikeSLM,
    wfs: WFSManager,
    n_max: int = DEFAULT_N_MAX,
    magnitude: float | None = None,
    n_cycles: int = DEFAULT_N_CYCLES,
    n_averages: int = DEFAULT_N_AVERAGES,
    wait_time: float = DEFAULT_WAIT_TIME,
    excluded_piston: bool = True,
    excluded_tip_tilt: bool = False,
    compute_inverses: bool = True,
    verbose: bool = True,
    display: ZernikeCalibrationDisplay | None = None,
    callback: Callable[[int, int, np.ndarray, np.ndarray], None] | None = None,
    debug_data_callback: Callable | None = None,
    subaperture_mask: np.ndarray | None = None,
    mask_n_avg: int = 30,
    mask_threshold_ratio: float = 0.3,
    mask_edge_clip: int = 1,
    auto_optimize_amplitude: bool = True,
    optimize_n_avg: int = 10,
    cancel_tile: bool = False,
) -> ZernikeResponseMatrixResult:
    """Calibrate Zernike response matrix.

    Args:
        zslm: ZernikeSLM instance
        wfs: Thorlab WFS instance
        n_max: Zernike maximum order
        magnitude: Perturbation magnitude per mode (in wavelength). If None or 0, auto-optimizes.
        n_cycles: Number of positive/negative perturbation cycles (N)
        n_averages: Number of WFS readings per measurement (M)
        wait_time: Wait time after applying phase (seconds)
        excluded_piston: Exclude piston (Z1)
        excluded_tip_tilt: Exclude tip/tilt (Z2, Z3)
        compute_inverses: Compute inverse matrices
        verbose: Show progress bar
        display: Optional ZernikeCalibrationDisplay for real-time visualization
        callback: Optional callback after each mode measurement.
        debug_data_callback: Debug data callback for raw measurement storage.
        subaperture_mask: Pre-computed valid subaperture mask. If None, builds automatically.
        mask_n_avg: Frames to average for mask building (if auto-built)
        mask_threshold_ratio: Intensity threshold ratio for mask (if auto-built)
        mask_edge_clip: Edge clip for mask (if auto-built)
        auto_optimize_amplitude: If True and magnitude is None/0, auto-optimize per mode.
        optimize_n_avg: WFS readings per amplitude during optimization.
        cancel_tile: If True, remove tip/tilt from WFS measurements.

    Returns:
        ZernikeResponseMatrixResult with response matrices and metadata
    """
    n_remove = (1 if excluded_piston else 0) + (2 if excluded_tip_tilt else 0)
    n_slm_terms = calc_n_zernike_terms(n_max) - n_remove
    n_wfs_terms = wfs.calc_n_zernike_terms(DEFAULT_N_MAX) - n_remove

    amplitude_optimization = None
    if magnitude is None or magnitude == 0:
        if auto_optimize_amplitude:
            logger.info("Auto-optimizing perturbation amplitude...")
            test_amps = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            amplitude_optimization = {}
            for mode_idx in range(n_slm_terms):
                opt_amp, diagnostics = _optimize_perturbation_amplitude(
                    zslm, wfs, mode_idx, test_amps, n_avg=optimize_n_avg, zernike_order=n_max,
                    cancel_tile=cancel_tile,
                )
                amplitude_optimization[mode_idx] = diagnostics
            magnitude = float(np.mean([v["optimal_amplitude"] for v in amplitude_optimization.values()]))
            logger.info(f"Using average optimal amplitude: {magnitude:.3%}λ")
        else:
            magnitude = DEFAULT_MAGNITUDE

    if subaperture_mask is None:
        logger.info("Building subaperture mask automatically...")
        subaperture_mask, _ = wfs.build_subaperture_mask(
            n_avg=mask_n_avg,
            threshold_ratio=mask_threshold_ratio,
            edge_clip=mask_edge_clip,
        )

    valid_count = np.sum(subaperture_mask)
    total_count = subaperture_mask.size
    logger.info(
        f"Starting Zernike response matrix calibration: n_max={n_max}, "
        f"SLM terms={n_slm_terms}, WFS terms={n_wfs_terms}, "
        f"valid subapertures={valid_count}/{total_count} ({valid_count/total_count*100:.1f}%), "
        f"magnitude={magnitude}λ, cycles={n_cycles}, averages={n_averages}"
    )

    response_matrix = np.zeros((n_wfs_terms, n_slm_terms), dtype=np.float64)
    variance_matrix = np.zeros((n_wfs_terms, n_slm_terms), dtype=np.float64)
    deviation_response_matrix = np.zeros((2 * wfs.num_spots_x * wfs.num_spots_y, n_slm_terms), dtype=np.float64)

    set_slm_flat(zslm._slm)
    time.sleep(wait_time)

    # Initialize display window if provided
    if display is not None:
        display.init_window()

    mode_indices = range(n_slm_terms)

    # Determine if we should use tqdm (only if no callback and verbose=True)
    use_tqdm = (callback is None) and verbose
    if use_tqdm:
        mode_indices = tqdm(mode_indices, desc="校准进度")

    for i in mode_indices:
        # Create coefficient array for this mode
        # coeffs_full is in full Zernike format (size calc_n_zernike_terms(n_max))
        # index 0 = piston (Z1), index 1 = tip (Z2), index 2 = tilt (Z3), ...
        # When excluded_piston=True, mode i starts from Noll index i+2
        # When excluded_piston=True AND excluded_tip_tilt=True, mode i starts from Noll index i+4
        n_full = wfs.calc_n_zernike_terms(DEFAULT_N_MAX)
        coeffs_full = np.zeros(n_full, dtype=np.float64)
        coeff_value = magnitude

        # Calculate which Noll index this mode corresponds to
        if excluded_piston and excluded_tip_tilt:
            # Skip piston (1) + tip/tilt (2) = 3 terms
            noll_offset = 3
        elif excluded_piston:
            # Skip piston (1) = 1 term
            noll_offset = 1
        else:
            noll_offset = 0

        coeffs_full[i + noll_offset] = coeff_value

        mean_resp, var_resp, mean_dev, var_dev = measure_zernike_mode_response(
            zslm=zslm,
            wfs=wfs,
            zernike_coeffs=coeffs_full,
            magnitude=coeff_value,
            n_averages=n_averages,
            n_cycles=n_cycles,
            wait_time=wait_time,
            excluded_piston=excluded_piston,
            excluded_tip_tilt=excluded_tip_tilt,
            zernike_order=DEFAULT_N_MAX,
            mode_index=i,
            debug_data_callback=debug_data_callback,
            cancel_tile=cancel_tile,
        )
        logger.debug(f'iter {i} rms = {np.sqrt(np.mean(mean_resp ** 2))}')

        response_matrix[:, i] = mean_resp
        variance_matrix[:, i] = var_resp
        deviation_response_matrix[:, i] = mean_dev

        # Call callback if provided (after each mode measurement)
        if callback is not None:
            callback(i, n_slm_terms, mean_resp, var_resp)

        # Update display if provided
        if display is not None:
            noll_index = i + noll_offset + 1  # Noll index is 1-based
            mode_name = f"Z{noll_index}"
            continue_flag = display.update(
                mode_index=i,
                mode_name=mode_name,
                response_col=mean_resp,
                variance_col=var_resp,
                current_cycle=n_cycles,
                total_cycles=n_cycles,
                mean_variance=float(np.mean(var_resp)),
            )
            if not continue_flag:
                logger.warning("用户关闭了显示窗口，校准终止")
                break

    # 计算逆矩阵 (可选)
    pinv_matrix = None
    lstsq_matrix = None

    if compute_inverses:
        logger.info("计算SVD伪逆矩阵...")
        pinv_matrix = compute_pinv(response_matrix)

        logger.info("计算最小二乘逆矩阵...")
        lstsq_matrix = compute_lstsq(response_matrix)

    result = ZernikeResponseMatrixResult(
        matrix=response_matrix,
        variance_matrix=variance_matrix,
        deviation_response_matrix=deviation_response_matrix,
        subaperture_mask=subaperture_mask,
        n_max=n_max,
        magnitude=magnitude,
        wavelength_nm=zslm.wavelength,
        n_averages=n_averages,
        n_cycles=n_cycles,
        timestamp=datetime.now().isoformat(),
        excluded_piston=excluded_piston,
        excluded_tip_tilt=excluded_tip_tilt,
        pinv_matrix=pinv_matrix,
        lstsq_matrix=lstsq_matrix,
        amplitude_optimization=amplitude_optimization,
    )

    logger.info(
        f"Calibration complete: matrix shape={result.matrix.shape}, "
        f"valid_subapertures={np.sum(subaperture_mask)}/{subaperture_mask.size}, "
        f"mean_variance={result.mean_variance:.6f}, "
        f"timestamp={result.timestamp}"
    )

    # Close display window if provided
    if display is not None:
        display.close()

    return result


def save_zernike_response_matrix(
    result: ZernikeResponseMatrixResult,
    path: str | Path,
    include_inverses: bool = True,
) -> None:
    """Save response matrix to HDF5 file with full metadata.

    Args:
        result: Calibration result
        path: Save path (with .h5 extension or as base path)
        include_inverses: Whether to include inverse matrices
    """
    path = Path(path)
    if path.suffix != ".h5":
        path = path.with_suffix(".h5")
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset("matrix", data=result.matrix)
        f.create_dataset("variance_matrix", data=result.variance_matrix)

        if result.deviation_response_matrix is not None:
            f.create_dataset("deviation_response_matrix", data=result.deviation_response_matrix)

        if result.subaperture_mask is not None:
            f.create_dataset("subaperture_mask", data=result.subaperture_mask)

        if include_inverses and result.pinv_matrix is not None:
            f.create_dataset("pinv_matrix", data=result.pinv_matrix)

        if include_inverses and result.lstsq_matrix is not None:
            f.create_dataset("lstsq_matrix", data=result.lstsq_matrix)

        meta = f.create_group("metadata")
        meta.attrs["n_max"] = result.n_max
        meta.attrs["magnitude"] = result.magnitude
        meta.attrs["wavelength_nm"] = result.wavelength_nm
        meta.attrs["n_averages"] = result.n_averages
        meta.attrs["n_cycles"] = result.n_cycles
        meta.attrs["timestamp"] = result.timestamp
        meta.attrs["excluded_piston"] = result.excluded_piston
        meta.attrs["excluded_tip_tilt"] = result.excluded_tip_tilt
        meta.attrs["mean_variance"] = result.mean_variance
        meta.attrs["max_variance"] = result.max_variance
        meta.attrs["condition_number"] = result.condition_number if result.condition_number is not None else -1

        if result.amplitude_optimization is not None:
            opt_grp = f.create_group("amplitude_optimization")
            for mode_idx, diag in result.amplitude_optimization.items():
                mode_grp = opt_grp.create_group(f"mode_{mode_idx}")
                mode_grp.create_dataset("test_amps", data=diag["test_amps"])
                mode_grp.create_dataset("responses", data=diag["responses"])
                mode_grp.create_dataset("linearity", data=diag["linearity"])
                mode_grp.attrs["best_idx"] = diag["best_idx"]
                mode_grp.attrs["optimal_amplitude"] = diag["optimal_amplitude"]

        if result.device_config is not None:
            meta.attrs["device_config"] = json.dumps(result.device_config)

    logger.info(f"Response matrix saved to: {path}")


def load_zernike_response_matrix(path: str | Path) -> ZernikeResponseMatrixResult:
    """Load response matrix from HDF5 file.

    Args:
        path: File path (.h5 extension or base)

    Returns:
        Calibration result
    """
    path = Path(path)
    if path.suffix != ".h5":
        path = path.with_suffix(".h5")

    with h5py.File(path, "r") as f:
        matrix = f["matrix"][:]
        variance_matrix = f["variance_matrix"][:]
        deviation_matrix = f["deviation_response_matrix"][:] if "deviation_response_matrix" in f else None
        subaperture_mask = f["subaperture_mask"][:] if "subaperture_mask" in f else None
        pinv_matrix = f["pinv_matrix"][:] if "pinv_matrix" in f else None
        lstsq_matrix = f["lstsq_matrix"][:] if "lstsq_matrix" in f else None

        meta = f["metadata"]
        n_max = int(meta.attrs["n_max"])
        magnitude = float(meta.attrs["magnitude"])
        wavelength_nm = int(meta.attrs["wavelength_nm"])
        n_averages = int(meta.attrs["n_averages"])
        n_cycles = int(meta.attrs["n_cycles"])
        timestamp = str(meta.attrs["timestamp"])
        excluded_piston = bool(meta.attrs["excluded_piston"])
        excluded_tip_tilt = bool(meta.attrs["excluded_tip_tilt"])
        mean_variance = float(meta.attrs["mean_variance"])
        max_variance = float(meta.attrs["max_variance"])
        condition_number = meta.attrs["condition_number"]
        if condition_number == -1:
            condition_number = None

        amplitude_optimization = None
        if "amplitude_optimization" in f:
            amplitude_optimization = {}
            opt_grp = f["amplitude_optimization"]
            for mode_key in opt_grp.keys():
                mode_idx = int(mode_key.split("_")[1])
                mode_grp = opt_grp[mode_key]
                amplitude_optimization[mode_idx] = {
                    "test_amps": mode_grp["test_amps"][:],
                    "responses": mode_grp["responses"][:],
                    "linearity": mode_grp["linearity"][:],
                    "best_idx": int(mode_grp.attrs["best_idx"]),
                    "optimal_amplitude": float(mode_grp.attrs["optimal_amplitude"]),
                }

        device_config = None
        if "device_config" in meta.attrs:
            try:
                device_config = json.loads(meta.attrs["device_config"])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse device_config: {e}")

        return ZernikeResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance_matrix,
            deviation_response_matrix=deviation_matrix,
            subaperture_mask=subaperture_mask,
            n_max=n_max,
            magnitude=magnitude,
            wavelength_nm=wavelength_nm,
            n_averages=n_averages,
            n_cycles=n_cycles,
            timestamp=timestamp,
            excluded_piston=excluded_piston,
            excluded_tip_tilt=excluded_tip_tilt,
            pinv_matrix=pinv_matrix,
            lstsq_matrix=lstsq_matrix,
            amplitude_optimization=amplitude_optimization,
            device_config=device_config,
        )


def plot_response_matrix(
    result: ZernikeResponseMatrixResult,
    output_dir: str | Path,
) -> None:
    """绘制响应矩阵和方差矩阵可视化

    Args:
        result: 校准结果
        output_dir: 输出目录
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 响应矩阵热图
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(result.matrix, aspect="auto", cmap="RdBu_r")
    ax.set_xlabel("SLM Zernike Mode Index")
    ax.set_ylabel("WFS Zernike Mode Index")
    ax.set_title(f"Response Matrix (n_max={result.n_max})")
    fig.colorbar(im, ax=ax, label="Response")
    fig.tight_layout()
    fig.savefig(output_dir / "response_heatmap.png", dpi=150)
    plt.close(fig)

    # 2. 方差矩阵热图
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(result.variance_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("SLM Zernike Mode Index")
    ax.set_ylabel("WFS Zernike Mode Index")
    ax.set_title(f"Variance Matrix (mean={result.mean_variance:.6f})")
    fig.colorbar(im, ax=ax, label="Variance")
    fig.tight_layout()
    fig.savefig(output_dir / "variance_heatmap.png", dpi=150)
    plt.close(fig)

    # 3. 每列平均方差 (稳定性指标)
    fig, ax = plt.subplots(figsize=(12, 5))
    col_mean_var = np.mean(result.variance_matrix, axis=0)
    ax.bar(range(len(col_mean_var)), col_mean_var)
    ax.set_xlabel("SLM Zernike Mode Index")
    ax.set_ylabel("Mean Variance")
    ax.set_title("Measurement Stability per Mode")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "variance_per_mode.png", dpi=150)
    plt.close(fig)

    # 4. SVD奇异值 (如果已计算逆矩阵)
    if result.pinv_matrix is not None:
        _, s, _ = np.linalg.svd(result.matrix)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(s, "o-")
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Singular Value")
        ax.set_title(f"SVD Singular Values (condition={result.condition_number:.2e})")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "singular_values.png", dpi=150)
        plt.close(fig)

    logger.info(f"可视化图表已保存到: {output_dir}")


# 导出主要函数
__all__ = [
    "DEFAULT_MAGNITUDE",
    "DEFAULT_N_AVERAGES",
    "DEFAULT_N_CYCLES",
    "DEFAULT_N_MAX",
    "DEFAULT_WAIT_TIME",
    "ZernikeResponseMatrixResult",
    "calibrate_zernike_response_matrix",
    "compute_lstsq",
    "compute_pinv",
    "load_zernike_response_matrix",
    "measure_zernike_mode_response",
    "plot_response_matrix",
    "save_zernike_response_matrix",
]
