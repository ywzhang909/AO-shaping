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
    n_max: int = 10  # Zernike最大阶数
    magnitude: float = 0.5  # 校准时使用的幅度 (波长)
    wavelength_nm: int = 1064  # 工作波长 (nm)
    n_averages: int = 20  # 每次WFS读取次数 (M)
    n_cycles: int = 1  # 正负交替循环次数 (N)
    timestamp: str = ""  # 时间戳
    excluded_piston: bool = True  # 是否排除piston
    excluded_tip_tilt: bool = False  # 是否排除tip/tilt

    # 可选的逆矩阵
    pinv_matrix: np.ndarray | None = None
    lstsq_matrix: np.ndarray | None = None

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
        if self.pinv_matrix is not None:
            d["pinv_matrix"] = self.pinv_matrix.tolist()
        if self.lstsq_matrix is not None:
            d["lstsq_matrix"] = self.lstsq_matrix.tolist()
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
        if "pinv_matrix" in d and d["pinv_matrix"] is not None:
            d["pinv_matrix"] = np.array(d["pinv_matrix"])
        else:
            d["pinv_matrix"] = None
        if "lstsq_matrix" in d and d["lstsq_matrix"] is not None:
            d["lstsq_matrix"] = np.array(d["lstsq_matrix"])
        else:
            d["lstsq_matrix"] = None
        return cls(**d)


def set_slm_flat(slm) -> None:
    """设置SLM为平相位"""
    zero_phase = np.zeros((slm.Panel_Res[1], slm.Panel_Res[0]), dtype=np.uint16)
    slm.display_data(zero_phase)


def measure_zernike_mode_response(
    zslm: ZernikeSLM,
    wfs: WFSManager,
    coefficients: np.ndarray,
    coeff_value: float,
    n_averages: int = DEFAULT_N_AVERAGES,
    n_cycles: int = DEFAULT_N_CYCLES,
    wait_time: float = DEFAULT_WAIT_TIME,
    excluded_piston: bool = True,
    excluded_tip_tilt: bool = False,
    zernike_order: int = 10,
    mode_index: int = 0,
    debug_data_callback: Callable | None = None,
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

            dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=False)
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

        mean_dev = np.concatenate([mean_dev_x.flatten(), mean_dev_y.flatten()])
        return mean_resp, mean_dev

    all_responses = []
    all_deviations = []

    for cycle in range(n_cycles):
        response_plus, dev_plus = measure_once(+coefficients, mode_index, cycle, True)
        set_slm_flat(zslm._slm)
        time.sleep(wait_time)

        response_minus, dev_minus = measure_once(-coefficients, mode_index, cycle, False)
        set_slm_flat(zslm._slm)
        time.sleep(wait_time)

        response = (response_plus - response_minus) / (2 * coeff_value)
        deviation = (dev_plus - dev_minus) / (2 * coeff_value)

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
    magnitude: float = DEFAULT_MAGNITUDE,
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
) -> ZernikeResponseMatrixResult:
    """校准Zernike响应矩阵 (增强版)

    逐一加载各阶Zernike模式，测量对应的WFS响应，构建响应矩阵。

    Args:
        zslm: ZernikeSLM实例
        wfs: Thorlab WFS实例
        n_max: Zernike最大阶数
        magnitude: 每个模式的扰动幅度 (波长)
        n_cycles: 正负交替循环次数 (N)
        n_averages: 每次WFS读取次数 (M)
        wait_time: 每次施加相位后的等待时间 (秒)
        excluded_piston: 是否排除piston (Z1)
        excluded_tip_tilt: 是否排除tip/tilt (Z2, Z3)
        compute_inverses: 是否计算逆矩阵
        verbose: 是否显示进度条
        display: 可选的ZernikeCalibrationDisplay实例，用于实时可视化
        callback: 可选的回调函数，每次模式测量后调用。
            签名: callback(mode_index, total_modes, response_col, variance_col)
            - mode_index: 当前模式索引 (0-based)
            - total_modes: 总模式数 (n_slm_terms)
            - response_col: 响应向量 (mean_resp)
            - variance_col: 方差向量 (var_resp)
            如果提供callback，将跳过tqdm进度条。
        debug_data_callback: 调试数据回调，用于保存每次测量的原始数据。
            签名为: callback(mode_index, cycle, sample, slm_phase, shift_x, shift_y,
                           deviation_x, deviation_y, zernike_coeffs, is_plus)
            - mode_index: SLM Zernike模式索引
            - cycle: 当前循环次数 (0 to n_cycles-1)
            - sample: 当前采样索引 (0 to n_averages-1)
            - slm_phase: SLM灰度相位图 (uint16, with shift applied)
            - shift_x, shift_y: SLM平移值
            - deviation_x, deviation_y: WFS spot deviation数组
            - zernike_coeffs: WFS Zernike系数 (原始, averaging前)
            - is_plus: 是否是正向扰动 (+coefficients)
            仅在debug模式且回调非None时调用。

    Returns:
        ZernikeResponseMatrixResult对象，包含响应矩阵、方差和逆矩阵
    """
    n_remove = (1 if excluded_piston else 0) + (2 if excluded_tip_tilt else 0)
    n_slm_terms = calc_n_zernike_terms(n_max) - n_remove
    n_wfs_terms = wfs.calc_n_zernike_terms(n_max) - n_remove

    logger.info(
        f"开始Zernike响应矩阵校准: n_max={n_max}, "
        f"SLM terms={n_slm_terms}, WFS terms={n_wfs_terms}, "
        f"magnitude={magnitude}λ, cycles={n_cycles}, averages={n_averages}, "
        f"excluded_piston={excluded_piston}, excluded_tip_tilt={excluded_tip_tilt}"
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
        n_full = wfs.calc_n_zernike_terms(n_max)
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
            coefficients=coeffs_full,
            coeff_value=coeff_value,
            n_averages=n_averages,
            n_cycles=n_cycles,
            wait_time=wait_time,
            excluded_piston=excluded_piston,
            excluded_tip_tilt=excluded_tip_tilt,
            zernike_order=n_max,
            mode_index=i,
            debug_data_callback=debug_data_callback,
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
    )

    logger.info(
        f"校准完成: matrix shape={result.matrix.shape}, "
        f"mean_variance={result.mean_variance:.6f}, "
        f"max_variance={result.max_variance:.6f}, "
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
    """保存响应矩阵到文件 (增强版)

    Args:
        result: 校准结果
        path: 保存路径 (不含扩展名)
        include_inverses: 是否包含逆矩阵
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 保存响应矩阵
    np.save(path.with_suffix(".response.npy"), result.matrix)

    # 保存方差矩阵
    np.save(path.with_suffix(".variance.npy"), result.variance_matrix)

    # 保存子孔径斜率响应矩阵
    if result.deviation_response_matrix is not None:
        np.save(path.with_suffix(".deviation.npy"), result.deviation_response_matrix)

    # 保存逆矩阵 (可选)
    if include_inverses and result.pinv_matrix is not None:
        np.save(path.with_suffix(".pinv.npy"), result.pinv_matrix)

    if include_inverses and result.lstsq_matrix is not None:
        np.save(path.with_suffix(".lstsq.npy"), result.lstsq_matrix)

    # 保存元数据
    metadata = {
        "n_max": result.n_max,
        "magnitude": result.magnitude,
        "wavelength_nm": result.wavelength_nm,
        "n_averages": result.n_averages,
        "n_cycles": result.n_cycles,
        "timestamp": result.timestamp,
        "excluded_piston": result.excluded_piston,
        "excluded_tip_tilt": result.excluded_tip_tilt,
        "matrix_shape": result.matrix.shape,
        "variance_shape": result.variance_matrix.shape,
        "mean_variance": result.mean_variance,
        "max_variance": result.max_variance,
        "condition_number": result.condition_number,
        "has_pinv": result.pinv_matrix is not None,
        "has_lstsq": result.lstsq_matrix is not None,
        "has_deviation": result.deviation_response_matrix is not None,
        "deviation_shape": result.deviation_response_matrix.shape if result.deviation_response_matrix is not None else None,
    }

    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"响应矩阵已保存到: {path.parent}")


def load_zernike_response_matrix(path: str | Path) -> ZernikeResponseMatrixResult:
    """从文件加载响应矩阵 (增强版)

    Args:
        path: 文件路径 (不含扩展名)

    Returns:
        校准结果
    """
    path = Path(path)

    # 加载响应矩阵
    matrix = np.load(path.with_suffix(".response.npy"))

    # 加载方差矩阵
    variance_matrix = np.load(path.with_suffix(".variance.npy"))

    # 加载子孔径斜率响应矩阵 (如果存在)
    dev_path = path.with_suffix(".deviation.npy")
    deviation_matrix = np.load(dev_path) if dev_path.exists() else None

    # 加载逆矩阵 (如果存在)
    pinv_path = path.with_suffix(".pinv.npy")
    pinv_matrix = np.load(pinv_path) if pinv_path.exists() else None

    lstsq_path = path.with_suffix(".lstsq.npy")
    lstsq_matrix = np.load(lstsq_path) if lstsq_path.exists() else None

    # 加载元数据
    with open(path.with_suffix(".json"), encoding="utf-8") as f:
        metadata = json.load(f)

    return ZernikeResponseMatrixResult(
        matrix=matrix,
        variance_matrix=variance_matrix,
        deviation_response_matrix=deviation_matrix,
        n_max=metadata["n_max"],
        magnitude=metadata["magnitude"],
        wavelength_nm=metadata["wavelength_nm"],
        n_averages=metadata["n_averages"],
        n_cycles=metadata["n_cycles"],
        timestamp=metadata["timestamp"],
        excluded_piston=metadata.get("excluded_piston", True),
        excluded_tip_tilt=metadata.get("excluded_tip_tilt", False),
        pinv_matrix=pinv_matrix,
        lstsq_matrix=lstsq_matrix,
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
