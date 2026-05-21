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
from ao_shaping.algorithm.adam import AdaMOD, AdamW
from ao_shaping.utils import logger, Recorder
from ao_shaping.utils.matrix_utils import calc_n_zernike_terms


# =============================================================================
# Scheduling utilities for LR and delta
# =============================================================================


def cosine_annealing_lr(
    epoch: int,
    T_max: int,
    lr_max: float,
    lr_min: float = 1e-6,
) -> float:
    """Cosine annealing learning rate schedule.
    
    Args:
        epoch: Current epoch (0-indexed).
        T_max: Total number of epochs.
        lr_max: Maximum learning rate.
        lr_min: Minimum learning rate.
        
    Returns:
        Current learning rate.
    """
    if T_max <= 0:
        return lr_max
    # Cosine annealing: starts at lr_max, ends at lr_min
    return lr_min + (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2


def cosine_annealing_delta(
    epoch: int,
    T_max: int,
    delta_max: float,
    delta_min: float = 1e-7,
) -> float:
    """Cosine annealing delta (perturbation amplitude) schedule.
    
    Args:
        epoch: Current epoch (0-indexed).
        T_max: Total number of epochs.
        delta_max: Maximum delta.
        delta_min: Minimum delta.
        
    Returns:
        Current delta.
    """
    if T_max <= 0:
        return delta_max
    return delta_min + (delta_max - delta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2


def exponential_decay_lr(
    epoch: int,
    T_max: int,
    lr_max: float,
    lr_min: float = 1e-6,
    decay_factor: float = 0.01,
) -> float:
    """Exponential decay learning rate schedule.
    
    Args:
        epoch: Current epoch (0-indexed).
        T_max: Total number of epochs.
        lr_max: Maximum learning rate.
        lr_min: Minimum learning rate.
        decay_factor: Decay factor (how much to decay over T_max).
        
    Returns:
        Current learning rate.
    """
    if T_max <= 0:
        return lr_max
    # Exponential decay: lr = lr_min + (lr_max - lr_min) * exp(-decay_factor * epoch / T_max)
    progress = epoch / T_max
    return lr_min + (lr_max - lr_min) * np.exp(decay_factor * progress * np.log(lr_max / (lr_min + 1e-10)))


def linear_decay_lr(
    epoch: int,
    T_max: int,
    lr_max: float,
    lr_min: float = 1e-6,
) -> float:
    """Linear decay learning rate schedule.
    
    Args:
        epoch: Current epoch (0-indexed).
        T_max: Total number of epochs.
        lr_max: Maximum learning rate.
        lr_min: Minimum learning rate.
        
    Returns:
        Current learning rate.
    """
    if T_max <= 0:
        return lr_max
    progress = min(epoch / T_max, 1.0)
    return lr_min + (lr_max - lr_min) * (1 - progress)


def get_lr_schedule(
    schedule_type: str,
    epoch: int,
    T_max: int,
    lr_max: float,
    lr_min: float = 1e-6,
) -> float:
    """Get learning rate based on schedule type.
    
    Args:
        schedule_type: Schedule type ("static", "cosine", "exp", "linear").
        epoch: Current epoch.
        T_max: Total epochs.
        lr_max: Maximum LR.
        lr_min: Minimum LR.
        
    Returns:
        Current LR value.
    """
    if schedule_type == "static":
        return lr_max
    elif schedule_type == "cosine":
        return cosine_annealing_lr(epoch, T_max, lr_max, lr_min)
    elif schedule_type == "exp":
        return exponential_decay_lr(epoch, T_max, lr_max, lr_min)
    elif schedule_type == "linear":
        return linear_decay_lr(epoch, T_max, lr_max, lr_min)
    else:
        logger.warning(f"Unknown schedule type '{schedule_type}', using static")
        return lr_max


def get_delta_schedule(
    schedule_type: str,
    epoch: int,
    T_max: int,
    delta_max: float,
    delta_min: float = 1e-7,
) -> float:
    """Get delta (perturbation amplitude) based on schedule type.
    
    Args:
        schedule_type: Schedule type ("static", "cosine", "exp", "linear").
        epoch: Current epoch.
        T_max: Total epochs.
        delta_max: Maximum delta.
        delta_min: Minimum delta.
        
    Returns:
        Current delta value.
    """
    if schedule_type == "static":
        return delta_max
    elif schedule_type == "cosine":
        return cosine_annealing_delta(epoch, T_max, delta_max, delta_min)
    elif schedule_type == "exp":
        return exponential_decay_lr(epoch, T_max, delta_max, delta_min)
    elif schedule_type == "linear":
        return linear_decay_lr(epoch, T_max, delta_max, delta_min)
    else:
        logger.warning(f"Unknown schedule type '{schedule_type}', using static")
        return delta_max


# =============================================================================
# Early stopping utilities
# =============================================================================


def early_stopping_check(
    rms_history: list[float],
    window: int = 10,
    min_epochs: int = 0,
    patience: int = 20,
    improvement_threshold: float = 1e-4,
) -> tuple[bool, float]:
    """Check if optimization should stop based on sliding window validation.
    
    Args:
        rms_history: List of RMS values over epochs.
        window: Sliding window size for validation.
        min_epochs: Minimum epochs before early stopping can trigger.
        patience: Number of consecutive non-improvement windows before stopping.
        improvement_threshold: Minimum improvement to count as improvement.
        
    Returns:
        Tuple of (should_stop, window_mean_rms).
    """
    n_epochs = len(rms_history)
    
    if n_epochs < min_epochs + window:
        return False, float('inf')
    
    # Calculate recent window mean
    recent_window = rms_history[-window:]
    window_mean = sum(recent_window) / window
    
    # Check if any previous window was significantly better
    best_window_rms = float('inf')
    consecutive_no_improve = 0
    
    for i in range(min_epochs, n_epochs - window + 1):
        past_window = rms_history[i:i + window]
        past_mean = sum(past_window) / window
        
        if past_mean < best_window_rms - improvement_threshold:
            best_window_rms = past_mean
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
    
    # Stop if too many consecutive non-improvement windows
    should_stop = consecutive_no_improve >= patience
    
    return should_stop, window_mean


# =============================================================================
# Mini-batch SPGD gradient estimation
# =============================================================================


def compute_mini_batch_gradient(
    current_c: np.ndarray,
    slm: ZernikeSLM,
    wfs: Thorlab_WFS,
    delta: float,
    perturb_weights: np.ndarray,
    n_zernike: int,
    n_batches: int = 1,
    slm_wait_time: float = 0.2,
    remove_tilt: bool = False,
    zernike_min: float = -500.0,
    zernike_max: float = 500.0,
) -> tuple[np.ndarray, dict]:
    """Compute mini-batch SPGD gradient by averaging multiple perturbations.
    
    Args:
        current_c: Current Zernike coefficients.
        slm: SLM device.
        wfs: WFS device.
        delta: Perturbation amplitude.
        perturb_weights: Zernike perturbation weights.
        n_zernike: Number of Zernike terms.
        n_batches: Number of perturbations to average (1 = no averaging).
        slm_wait_time: Wait time after SLM update.
        remove_tilt: Whether to remove tilt in WFS measurement.
        zernike_min: Minimum Zernike coefficient.
        zernike_max: Maximum Zernike coefficient.
        
    Returns:
        Tuple of (averaged_gradient, info_dict with details).
    """
    if n_batches <= 0:
        n_batches = 1
    
    gradients = []
    pos_rms_list = []
    neg_rms_list = []
    
    for batch_idx in range(n_batches):
        # Generate random perturbation direction
        disturb_c = np.random.binomial(1, 0.5, (n_zernike,)).astype(float) * 2.0 - 1.0
        disturb_c = disturb_c * delta * perturb_weights
        # Zero piston mode
        if len(disturb_c) > 0:
            disturb_c[0] = 0
        
        # Positive perturbation
        pos_c = np.clip(current_c + disturb_c, zernike_min, zernike_max)
        slm.send_zernike(pos_c, slm_wait_time)
        wfs.take_image(3)
        _, pos_statics = wfs.get_wavefront(cancel_tile=remove_tilt)
        pos_rms = pos_statics.get('rms', np.inf)
        
        # Negative perturbation
        neg_c = np.clip(current_c - disturb_c, zernike_min, zernike_max)
        slm.send_zernike(neg_c, slm_wait_time)
        wfs.take_image(3)
        _, neg_statics = wfs.get_wavefront(cancel_tile=remove_tilt)
        neg_rms = neg_statics.get('rms', np.inf)
        
        # Compute gradient for this batch
        diff = pos_rms - neg_rms
        gradient = diff * disturb_c
        gradients.append(gradient)
        pos_rms_list.append(pos_rms)
        neg_rms_list.append(neg_rms)
    
    # Average gradients
    avg_gradient = np.mean(gradients, axis=0)
    
    info = {
        'n_batches': n_batches,
        'avg_pos_rms': np.mean(pos_rms_list),
        'avg_neg_rms': np.mean(neg_rms_list),
        'pos_rms_std': np.std(pos_rms_list) if len(pos_rms_list) > 1 else 0.0,
        'neg_rms_std': np.std(neg_rms_list) if len(neg_rms_list) > 1 else 0.0,
    }
    
    return avg_gradient, info


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
    lr: float = 0.01,
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
    # NEW: Learning rate scheduling
    lr_schedule: str = "static",
    lr_min: float = 1e-6,
    # NEW: Delta scheduling
    delta_schedule: str = "static",
    delta_min: float = 1e-7,
    # NEW: Optimizer selection
    optimizer_type: str = "adamod",
    beta1: float = 0.95,
    weight_decay: float = 1e-2,
    # NEW: Mini-batch SPGD
    mini_batch: int = 1,
    # NEW: Gradient clipping
    gradient_clip: float = 0.0,
    # NEW: Stagnation detection and restart
    stagnation_patience: int = 30,
    stagnation_delta_boost: float = 1.5,
    freeze_high_order_threshold: float | None = None,
    # NEW: Early stopping with sliding window
    early_stop_window: int = 0,
    early_stop_min_epochs: int = 0,
    early_stop_patience: int = 0,
    # NEW: WFS frame averaging
    n_frames: int = 10,
):
    """Optimize wavefront RMS using SLM with Zernike coefficient control.

    This function uses AdaMOD or AdamW algorithm to optimize Zernike coefficients
    displayed on an SLM, minimizing the wavefront RMS measured by a WFS.

    Args:
        epochs: Number of optimization iterations.
        init_z: Initial Zernike coefficients. Can be:
            - dict: {(n, m): value} form
            - np.ndarray: Noll-ordered coefficients
            - None: starts from zeros
        lr: Initial learning rate.
        delta: Initial perturbation delta.
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
        lr_schedule: Learning rate schedule type ("static", "cosine", "exp", "linear").
        lr_min: Minimum learning rate for scheduling.
        delta_schedule: Delta schedule type ("static", "cosine", "exp", "linear").
        delta_min: Minimum delta for scheduling.
        optimizer_type: Optimizer type ("adamod" or "adamw").
        beta1: Adam beta1 parameter for momentum.
        weight_decay: AdamW weight decay parameter.
        mini_batch: Number of SPGD perturbations to average per gradient estimate.
        gradient_clip: Maximum absolute gradient value (0 to disable).
        stagnation_patience: Epochs without improvement before delta boost.
        stagnation_delta_boost: Multiplier for delta boost on stagnation.
        freeze_high_order_threshold: RMS threshold to freeze high-order modes.
        early_stop_window: Sliding window size for early stopping validation.
        early_stop_min_epochs: Minimum epochs before early stopping can trigger.
        early_stop_patience: Consecutive non-improvement windows before stopping.
        n_frames: Number of WFS images to average per measurement.

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
            wfs.take_image(n_frames)
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

        if optimizer_type == "adamw":
            optimizer = AdamW(dim=n_zernike, lr=lr, beta1=beta1, beta2=0.99, weight_decay=weight_decay)
        else:
            optimizer = AdaMOD(dim=n_zernike, lr=lr, beta1=beta1, beta3=0.9995)

        best_c = _init_c.copy()
        best_rms = rms
        stagnation_counter = 0
        rms_history = [rms]
        freeze_mask = np.zeros(n_zernike, dtype=bool)

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
                current_epoch = epoch - 1

                current_lr = get_lr_schedule(lr_schedule, current_epoch, epochs, lr, lr_min)
                current_delta = get_delta_schedule(delta_schedule, current_epoch, epochs, delta, delta_min)
                optimizer.lr = current_lr

                effective_weights = perturb_weights * (~freeze_mask).astype(float)

                if mini_batch > 1:
                    gradient, mb_info = compute_mini_batch_gradient(
                        _init_c, slm, wfs, current_delta, effective_weights,
                        n_zernike, mini_batch, slm_wait_time, remove_tilt,
                        ZERNIKE_MIN, ZERNIKE_MAX
                    )
                    pos_j = mb_info['avg_pos_rms']
                    neg_j = mb_info['avg_neg_rms']
                    _pos_c = _init_c
                    _neg_c = _init_c
                    pos_wf = np.zeros((64, 64))
                    neg_wf = np.zeros((64, 64))
                    pos_phase = np.zeros((512, 512))
                    neg_phase = np.zeros((512, 512))
                    diff = pos_j - neg_j
                    pos_statics = {"rms": pos_j}
                    neg_statics = {"rms": neg_j}
                    pos_extra = {}
                    neg_extra = {}
                else:
                    disturb_c = np.random.binomial(1, 0.5, (n_zernike,)).astype(float) * 2.0 - 1.0
                    disturb_c = disturb_c * current_delta * effective_weights
                    if len(disturb_c) > 0:
                        disturb_c[0] = 0

                    _pos_c = np.clip(_init_c + disturb_c, ZERNIKE_MIN, ZERNIKE_MAX)
                    pos_phase = slm.send_zernike(_pos_c, slm_wait_time)
                    pos_wf, pos_statics, pos_extra = calc_j()
                    pos_j = pos_statics['rms']

                    _neg_c = np.clip(_init_c - disturb_c, ZERNIKE_MIN, ZERNIKE_MAX)
                    neg_phase = slm.send_zernike(_neg_c, slm_wait_time)
                    neg_wf, neg_statics, neg_extra = calc_j()
                    neg_j = neg_statics['rms']

                    diff = pos_statics['rms'] - neg_statics['rms']
                    gradient = diff * disturb_c

                if gradient_clip > 0:
                    gradient = np.clip(gradient, -gradient_clip, gradient_clip)

                avg_j = (pos_j + neg_j) / 2
                update = optimizer.update(gradient)
                _to_update_c = np.clip(_init_c - update, ZERNIKE_MIN, ZERNIKE_MAX)
                _init_c = _to_update_c

                if avg_j < best_rms:
                    best_rms = avg_j
                    best_c = _init_c.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                if stagnation_counter >= stagnation_patience and stagnation_patience > 0:
                    logger.info(f"Stagnation detected at epoch {epoch}, boosting delta by {stagnation_delta_boost}x")
                    current_delta = current_delta * stagnation_delta_boost
                    stagnation_counter = 0
                    if freeze_high_order_threshold is not None and best_rms > freeze_high_order_threshold:
                        high_order_start = calc_n_zernike_terms(2)
                        freeze_mask[high_order_start:] = True
                        logger.info(f"Freezing high-order Zernike modes (indices >= {high_order_start})")

                rms_history.append(avg_j)
                if early_stop_patience > 0 and early_stop_window > 0:
                    should_stop, window_mean = early_stopping_check(
                        rms_history, early_stop_window, early_stop_min_epochs, early_stop_patience
                    )
                    if should_stop:
                        logger.info(f"Early stop at epoch {epoch} with window_mean={window_mean:.4f}")
                        _init_c = best_c
                        break

                log = {
                    "rms": avg_j,
                    "_diff": diff,
                    "_gamma": optimizer.lr,
                    "delta": current_delta,
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
                    _init_c = best_c
                    break

                bar.update(1)

        # Restore best coefficients on exit
        best_c, _ = recorder.get_best_target('_c')
        if best_c is not None:
            slm.send_zernike(best_c)
            logger.info(f"Restored best coefficients, RMS: {recorder.get_best_target('rms')}")

        return recorder


