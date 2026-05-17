"""RMS optimization using SLM with Zernike coefficient control.

This module replaces DM voltage control with SLM phase modulation via Zernike
polynomial coefficients. The optimization perturbs Zernike coefficients,
displays the resulting phase pattern on the SLM, and measures the wavefront
RMS metric from the WFS.

Key changes from rms_by_zernike.py:
- NlightDM → ZernikeSLM
- Voltage vectors → Zernike coefficient vectors
- dm.send_voltages(v) → slm.send_zernike(c)
- DM neighbor safety checks → Zernike coefficient clipping
- dm.DM_Num → calc_n_zernike_terms(n_max)

Example:
    >>> from ao_shaping.optimizer.wf.rms_by_zernike import optimizer_rms
    >>> recorder = optimizer_rms(
    ...     epochs=2000,
    ...     n_max=4,
    ...     wavelength=1064,
    ... )
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import tqdm
import numpy as np

from ao_shaping.drivers import MlaRes, Thorlab_WFS
from ao_shaping.drivers.slm import ZernikeSLM
from ao_shaping.algorithm.adam import AdaMOD
from ao_shaping.utils import logger, Recorder
from ao_shaping.utils.matrix_utils import calc_n_zernike_terms


def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index to (n, m) Zernike order (hardcoded convention).

    Uses a hardcoded lookup table (Noll indices 1-15 only).
    NOTE: This convention DIFFERS from the aotools-based implementation
    in `utils/zernike_calc.py`. The canonical implementation is in
    `utils/zernike_calc.noll_to_nm()`.

    Args:
        j: Noll index (1-based), valid range 1-15.

    Returns:
        Tuple of (n, m) radial and azimuthal orders.
    """
    # Noll sequence for Zernike polynomials
    noll_sequence = [
        (0, 0),   # 1: piston
        (1, -1),  # 2: tilt x
        (1, 1),   # 3: tilt y
        (2, -2),  # 4: oblique astigmatism
        (2, 0),   # 5: defocus
        (2, 2),   # 6: oblique astigmatism
        (3, -3),  # 7: vertical trefoil
        (3, -1),  # 8: vertical coma
        (3, 1),   # 9: horizontal coma
        (3, 3),   # 10: horizontal trefoil
        (4, -4),  # 11: quadrafoil
        (4, -2),  # 12: oblique trefoil
        (4, 0),   # 13: primary spherical
        (4, 2),   # 14: oblique trefoil
        (4, 4),   # 15: quadrafoil
    ]
    if j < 1 or j > len(noll_sequence):
        raise ValueError(f"Noll index {j} out of valid range (1-{len(noll_sequence)})")
    return noll_sequence[j - 1]

# SLM parameters
SLM_WAVELENGTH_DEFAULT = 532  # nm
SLM_SHIFT_X_DEFAULT = 0  # pixels
SLM_SHIFT_Y_DEFAULT = 0  # pixels

# Zernike coefficient bounds
ZERNIKE_MIN = -500  # wavelengths
ZERNIKE_MAX = 500  # wavelengths

# Debug mode control via environment variable
DEBUG_MODE = os.environ.get("RMS_ZERNIKE_DEBUG", "0") == "1"

# Zernike n项扰动权重数组 - n越小影响越小
# 默认使用 n/n_max 归一化权重，可通过环境变量覆盖
# 环境变量格式: "1.0,0.8,0.6,0.4,0.2" (逗号分隔的权重值)
_ZERNIKE_PERTURB_WEIGHTS_DEFAULT = None  # 将在运行时根据n_max动态生成


def _get_perturb_weights(n_zernike: int) -> np.ndarray:
    """获取扰动权重数组

    使用环境变量 RMS_ZERNIKE_WEIGHTS 覆盖默认权重。
    环境变量格式: "w0,w1,w2,..." (逗号分隔)

    Args:
        n_zernike: Zernike项数量

    Returns:
        权重数组，形状为 (n_zernike,)
    """
    env_weights = os.environ.get("RMS_ZERNIKE_WEIGHTS", "")
    if env_weights:
        try:
            weights = np.array([float(x.strip()) for x in env_weights.split(",")], dtype=np.float64)
            if len(weights) == n_zernike:
                logger.debug(f"Using custom Zernike weights from env: {weights}")
                return weights
            elif len(weights) > n_zernike:
                logger.debug(f"Using first {n_zernike} custom weights: {weights[:n_zernike]}")
                return weights[:n_zernike]
            else:
                logger.warning(f"Custom weights count ({len(weights)}) != n_zernike ({n_zernike}), using default")
        except ValueError as e:
            logger.warning(f"Failed to parse RMS_ZERNIKE_WEIGHTS: {e}")

    # 默认权重: 基于n项的归一化权重 (n+1)/n_max
    # 生成zernike_modes对应的权重
    modes = _zernike_indices_from_n(n_zernike)
    n_values = np.array([n for n, m in modes], dtype=np.float64)
    n_max = np.max(n_values) if len(n_values) > 0 else 1
    weights = (n_values + 1) / (n_max + 1)  # 归一化到 (0, 1]
    weights[0] = 0.1  # piston mode权重最小
    if DEBUG_MODE:
        logger.debug(f"Default Zernike perturb weights: {weights}")
    return weights


def _zernike_indices_from_n(n_zernike: int) -> list[tuple[int, int]]:
    """根据Zernike项数量生成(n,m)模式列表

    Args:
        n_zernike: Zernike项数量 (Noll顺序)

    Returns:
        (n,m)元组列表
    """
    modes = []
    for j in range(1, n_zernike + 1):
        n, m = noll_to_nm(j)
        modes.append((n, m))
    return modes


def _zernike_indices(n_max: int) -> list[tuple[int, int]]:
    """Return list of (n, m) pairs for all valid Zernike modes up to n_max.

    Args:
        n_max: Maximum Zernike radial order.

    Returns:
        List of (n, m) tuples in Noll order.
    """
    n_terms = calc_n_zernike_terms(n_max)
    modes = []
    for j in range(1, n_terms + 1):
        n, m = noll_to_nm(j)
        if n <= n_max:
            modes.append((n, m))
    return modes


def optimizer_rms(
    epochs: int,
    init_z: Sequence[float | int] | dict[tuple[int, int], float] | None = None,
    lr:float = 0.01,
    delta: float = 0.0,
    early_stop_threshold: float = 0.12,
    # Zernike/SLM parameters
    wavelength: int = SLM_WAVELENGTH_DEFAULT,
    shift_x: int = SLM_SHIFT_X_DEFAULT,
    shift_y: int = SLM_SHIFT_Y_DEFAULT,
    n_max: int = 4,
    # WFS parameters
    wfs_res: MlaRes = MlaRes.Res1024,
    remove_tilt: bool = False,
    wfs_exposure_time: float = 0.2,
    pupil_center: tuple[float, float] = (0, 0),
    pupil_diameter: float = 4.6,
    # Other parameters
    slm_number: int = 1,
    slm_wavelength: int | None = None,
    slm_wait_time: float = 0.2,
    n_init_positions: int = 0,
    init_range: float = 20.0,
):
    """Optimize wavefront RMS using SLM with Zernike coefficient control.

    This function uses the AdaMOD algorithm to optimize Zernike coefficients
    displayed on an SLM, minimizing the wavefront RMS measured by a WFS.

    Args:
        epochs: Number of optimization iterations.
        init_z: Initial Zernike coefficients. Can be:
            - dict: {(n, m): value} form
            - np.ndarray: Noll-ordered coefficients
            - None: starts from zeros
        pupil_center: WFS pupil center (x, y).
        pupil_diameter: WFS pupil diameter.
        early_stop_threshold: Stop if RMS drops below this threshold.
        wavelength: SLM wavelength in nm.
        shift_x: SLM X shift in pixels.
        shift_y: SLM Y shift in pixels.
        n_max: Maximum Zernike radial order.
        wfs_res: WFS resolution.
        remove_tilt: Remove tilt in WFS wavefront measurement.
        slm_number: SLM device number (1-8).
        slm_wavelength: Override SLM wavelength (deprecated, use wavelength).
        n_init_positions: Number of random initial positions to try (0 = disabled).
        init_range: Range for random initialization.

    Returns:
        Recorder: Optimization history with RMS and coefficients.
    """
    epochs = int(epochs)

    # Calculate number of Zernike terms
    n_zernike = calc_n_zernike_terms(n_max)
    zernike_modes = _zernike_indices(n_max)

    recorder = Recorder(mark='rms', mode='min')

    with (
        ZernikeSLM(
            slm_number=slm_number,
            wavelength=slm_wavelength,
            n_max=n_max,
            shift_x=shift_x,
            shift_y=shift_y,
        ) as slm,
        Thorlab_WFS(
            wfs_res,
            exposure_time=wfs_exposure_time,
            use_custom_ref=False,
            high_speed=True,
            pupil_diameter=pupil_diameter,
            pupil_center=pupil_center,
        ) as wfs,
    ):
        # Initialize Zernike coefficients
        if init_z is None or (
            isinstance(init_z, (list, tuple, np.ndarray)) and len(init_z) == 0
        ):
            _init_c = np.zeros(n_zernike, dtype=np.float64)
        elif isinstance(init_z, dict):
            _init_c = np.zeros(n_zernike, dtype=np.float64)
            for (n, m), amp in init_z.items():
                # Find index in modes list
                if (n, m) in zernike_modes:
                    idx = zernike_modes.index((n, m))
                    _init_c[idx] = amp
        else:
            _init_c = np.array(init_z, dtype=np.float64)
            if len(_init_c) < n_zernike:
                padded = np.zeros(n_zernike, dtype=np.float64)
                padded[: len(_init_c)] = _init_c
                _init_c = padded
            elif len(_init_c) > n_zernike:
                _init_c = _init_c[:n_zernike]

        # Multi-start optimization: try multiple random positions
        if n_init_positions > 0:
            logger.info(f"Multi-start: testing {n_init_positions} random positions...")
            best_init_c = _init_c.copy()
            slm.send_zernike(_init_c)
            wfs.take_image(3)
            wf, statics = wfs.get_wavefront(cancel_tile=remove_tilt)
            best_rms = statics.get('rms', np.inf)
            
            for i in range(n_init_positions):
                test_c = np.random.uniform(-init_range, init_range, size=n_zernike)
                test_c[0] = 0
                test_c = np.clip(test_c, ZERNIKE_MIN, ZERNIKE_MAX)
                
                slm.send_zernike(test_c)
                wfs.take_image(3)
                wf, statics = wfs.get_wavefront(cancel_tile=remove_tilt)
                test_rms = statics.get('rms', np.inf)
                
                logger.debug(f"  Position {i+1}/{n_init_positions}: RMS={test_rms:.4f}")
                
                if test_rms < best_rms:
                    best_rms = test_rms
                    best_init_c = test_c.copy()
            
            _init_c = best_init_c
            logger.info(f"Multi-start: best position RMS={best_rms}")
            slm.send_zernike(_init_c)

        # Initialize SLM with initial coefficients
        init_phase = slm.send_zernike(_init_c, 1.0)

        def calc_j():
            wfs.take_image(10)
            wf, statics = wfs.get_wavefront(cancel_tile=remove_tilt)
            extra = {}
            if DEBUG_MODE:
                spots_intensities, (cx, cy) = wfs.get_spots_statics()
                dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=remove_tilt)
                extra = {
                    "_intensity": spots_intensities,
                    "_dev_x": dev_x,
                    "_dev_y": dev_y,
                }
            return wf, statics, extra

        wf, statics, init_extra = calc_j()
        rms = statics.get('rms', np.inf)
        optimizer = AdaMOD(dim=n_zernike, lr=lr, beta3=0.9999)

        logger.info(
            f"Initial RMS: {statics['rms']:.4f}, weight_rms: {rms:.4f}"
        )

        init_record = {
            "rms": statics['rms'],
            "_c": _init_c,
            "_diff": 0,
            "_gamma": lr,
            "delta": delta,
            "_epoch": 0,
            "_wavefront": wf[np.newaxis, ...],
            "_phase": init_phase[np.newaxis, ...],
            "_statics": statics,
        }
        if DEBUG_MODE:
            init_record.update(init_extra)
        recorder.append(init_record)

        perturb_weights = _get_perturb_weights(n_zernike)
        if DEBUG_MODE:
            logger.debug(f"Zernike perturb weights shape: {perturb_weights.shape}")

        with tqdm.tqdm(
            total=epochs,
            desc=f"RMS {statics['rms']:.3f} iter {epochs}",
            dynamic_ncols=True,
        ) as bar:
            for epoch in range(1, epochs + 1):
                disturb_c = np.random.binomial(1, 0.5, (n_zernike,)).astype(float) * 2.0 - 1.0
                disturb_c = disturb_c * delta * perturb_weights
                if len(disturb_c) > 0:
                    disturb_c[0] = 0

                # Positive perturbation
                _pos_c = np.clip(_init_c + disturb_c, ZERNIKE_MIN, ZERNIKE_MAX)
                pos_phase = slm.send_zernike(_pos_c, slm_wait_time)
                pos_wf, pos_statics, pos_extra = calc_j()
                pos_j = pos_statics['rms']

                # Negative perturbation
                _neg_c = np.clip(_init_c - disturb_c, ZERNIKE_MIN, ZERNIKE_MAX)
                neg_phase = slm.send_zernike(_neg_c, slm_wait_time)
                neg_wf, neg_statics, neg_extra = calc_j()
                neg_j = neg_statics['rms']

                diff = pos_statics['rms'] - neg_statics['rms']
                gradient = diff * disturb_c

                # Update using AdaMOD
                avg_j = (pos_j + neg_j) / 2
                # optimizer.lr = lr
                update = optimizer.update(gradient)

                # Clip update to Zernike bounds (no neighbor safety check for SLM)
                _to_update_c = np.clip(_init_c - update, ZERNIKE_MIN, ZERNIKE_MAX)
                _init_c = _to_update_c

                log = {
                    "rms": avg_j,
                    "_diff": diff,
                    "_gamma": optimizer.lr,
                    "delta": delta,
                    "_epoch": epoch,
                    "_c": _init_c,
                    "_pos_c": _pos_c,
                    "_neg_c": _neg_c,
                    "_wavefront": np.stack([pos_wf, neg_wf]),
                    "_phase": np.stack([pos_phase, neg_phase]),
                    "_statics": {"pos": pos_statics, "neg": neg_statics},
                }
                if DEBUG_MODE:
                    log.update({
                        "_pos_intensity": pos_extra.get("_intensity"),
                        "_neg_intensity": neg_extra.get("_intensity"),
                        "_pos_dev_x": pos_extra.get("_dev_x"),
                        "_neg_dev_x": neg_extra.get("_dev_x"),
                        "_pos_dev_y": pos_extra.get("_dev_y"),
                        "_neg_dev_y": neg_extra.get("_dev_y"),
                    })
                recorder.append(log)
                bar.set_postfix(recorder.last_info_dict)

                if avg_j < early_stop_threshold:
                    logger.info(f"Early stop at epoch {epoch} with rms={avg_j:.4f}")
                    break

                bar.update(1)

        # Restore best coefficients on exit
        best_c, _ = recorder.get_best_target('_c')
        if best_c is not None:
            slm.send_zernike(best_c)
            logger.info(f"Restored best coefficients, RMS: {recorder.get_best_target('rms')}")

        return recorder


