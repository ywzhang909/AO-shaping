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

from typing import Sequence

import tqdm
import numpy as np

from ao_shaping.drivers import MlaRes, Thorlab_WFS
from ao_shaping.drivers.slm import ZernikeSLM
from ao_shaping.algorithm.adam import AdaMOD
from ao_shaping.utils import logger, Recorder
from ao_shaping.utils.matrix_utils import calc_n_zernike_terms


def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index to (n, m) Zernike order.
    
    Args:
        j: Noll index (1-based).
    
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
SLM_WAVELENGTH_DEFAULT = 1064  # nm
SLM_SHIFT_X_DEFAULT = 0  # pixels
SLM_SHIFT_Y_DEFAULT = 0  # pixels

# Zernike coefficient bounds
ZERNIKE_MIN = -5.0  # wavelengths
ZERNIKE_MAX = 5.0  # wavelengths


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


def schedule_lr_delta(rms: float) -> tuple[float, float]:
    """Schedule learning rate and perturbation delta based on wavefront RMS.
    
    Args:
        rms: Current wavefront RMS value.
    
    Returns:
        tuple: (learning_rate, delta) for perturbation amplitude.
    """
    if rms > 0.3:
        return 2, 3
    elif rms > 0.25:
        return 1.5, 2
    elif rms > 0.2:
        return 1.1, 1.2
    elif rms > 0.15:
        return 1, 1
    elif rms > 0.11:
        return 0.9, 0.9
    elif rms > 0.08:
        return 0.8, 0.8
    else:
        return 0.7, 0.7


def optimizer_rms(
    epochs: int,
    init_z: Sequence[float | int] | dict[tuple[int, int], float] | None = None,
    pupil_center: tuple[float, float] = (0, 0),
    pupil_diameter: float = 2.24,
    early_stop_threshold: float = 0.12,
    # Zernike/SLM parameters
    wavelength: int = SLM_WAVELENGTH_DEFAULT,
    shift_x: int = SLM_SHIFT_X_DEFAULT,
    shift_y: int = SLM_SHIFT_Y_DEFAULT,
    n_max: int = 4,
    # WFS parameters
    wfs_res: MlaRes = MlaRes.Res768,
    remove_tilt: bool = False,
    # Other parameters
    slm_number: int = 1,
    slm_wavelength: int | None = None,
) -> Recorder:
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
    
    Returns:
        Recorder: Optimization history with RMS and coefficients.
    """
    epochs = int(epochs)
    
    # Use wavelength parameter if slm_wavelength not provided
    real_wavelength = slm_wavelength if slm_wavelength is not None else wavelength
    
    # Calculate number of Zernike terms
    n_zernike = calc_n_zernike_terms(n_max)
    zernike_modes = _zernike_indices(n_max)
    
    recorder = Recorder(mark='rms', mode='min')
    
    with (
        ZernikeSLM(
            slm_number=slm_number,
            wavelength=real_wavelength,
            n_max=n_max,
            shift_x=shift_x,
            shift_y=shift_y,
        ) as slm,
        Thorlab_WFS(
            wfs_res,
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
        
        # Initialize SLM with initial coefficients
        init_phase = slm.send_zernike(_init_c, display=True)
        
        def calc_j():
            wfs.take_image(5)
            wf, statics = wfs.get_wavefront(cancel_tile=remove_tilt)
            return wf, statics
        
        wf, statics = calc_j()
        # Support both 'weighted_rms' and 'wighted_rms' (historical typo)
        weight_rms = statics.get('weighted_rms') or statics.get('wighted_rms', statics['rms'])
        lr, delta = schedule_lr_delta(weight_rms)
        optimizer = AdaMOD(dim=n_zernike, lr=lr, beta3=0.9999)
        
        logger.info(
            f"Initial RMS: {statics['rms']:.4f}, weight_rms: {weight_rms:.4f}"
        )
        
        recorder.append(
            {
                "rms": statics['rms'],
                "_c": _init_c,
                "_diff": 0,
                "_gamma": lr,
                "delta": delta,
                "_epoch": 0,
                "_wavefront": wf[np.newaxis, ...],
                "_statics": statics,
            }
        )
        
        with tqdm.tqdm(
            total=epochs,
            desc=f"RMS {statics['rms']:.3f} iter {epochs}",
            dynamic_ncols=True,
        ) as bar:
            for epoch in range(1, epochs + 1):
                # Generate random perturbation (±1 pattern)
                disturb_c = np.random.binomial(1, 0.5, (n_zernike,)).astype(float) * 2.0 - 1.0
                disturb_c = disturb_c * delta
                # Set first mode (piston) to zero
                if len(disturb_c) > 0:
                    disturb_c[0] = 0
                
                # Positive perturbation
                _pos_c = np.clip(_init_c + disturb_c, ZERNIKE_MIN, ZERNIKE_MAX)
                slm.send_zernike(_pos_c, display=True)
                pos_wf, pos_statics = calc_j()
                pos_j = pos_statics['rms']
                
                # Negative perturbation
                _neg_c = np.clip(_init_c - disturb_c, ZERNIKE_MIN, ZERNIKE_MAX)
                slm.send_zernike(_neg_c, display=True)
                neg_wf, neg_statics = calc_j()
                neg_j = neg_statics['rms']
                
                diff = pos_statics['rms'] - neg_statics['rms']
                gradient = -diff * disturb_c
                
                # Update using AdaMOD
                avg_j = (pos_j + neg_j) / 2
                lr, delta = schedule_lr_delta(avg_j)
                optimizer.lr = lr
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
                    "_statics": {"pos": pos_statics, "neg": neg_statics},
                }
                recorder.append(log)
                bar.set_postfix(recorder.last_info_dict)
                
                if avg_j < early_stop_threshold:
                    logger.info(f"Early stop at epoch {epoch} with rms={avg_j:.4f}")
                    break
                
                bar.update(1)
        
        # Restore best coefficients on exit
        best_c, _ = recorder.get_best_target('_c')
        if best_c is not None:
            slm.send_zernike(best_c, display=True)
            logger.info(f"Restored best coefficients, RMS: {recorder.get_best_target('rms')}")
        
        return recorder


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RMS optimization using SLM with Zernike coefficients"
    )
    parser.add_argument("-e", "--epochs", type=int, default=2000, help="Number of iterations")
    parser.add_argument("-n", "--n_max", type=int, default=4, help="Max Zernike radial order")
    parser.add_argument("-w", "--wavelength", type=int, default=1064, help="SLM wavelength (nm)")
    parser.add_argument("--wfs_res", type=int, default=768, help="WFS resolution")
    parser.add_argument(
        "--pupil_diameter", type=float, default=2.24, help="WFS pupil diameter"
    )
    parser.add_argument(
        "--early_stop_threshold", type=float, default=0.12, help="Early stop threshold"
    )
    parser.add_argument("--slm_number", type=int, default=1, help="SLM device number")
    parser.add_argument("--remove_tilt", action="store_true", help="Remove tilt in WFS")
    parser.add_argument("--shift_x", type=int, default=0, help="SLM X shift")
    parser.add_argument("--shift_y", type=int, default=0, help="SLM Y shift")
    
    args = parser.parse_args()
    
    # Convert WFS resolution to MlaRes
    wfs_res = MlaRes.from_str(args.wfs_res)
    
    recorder = optimizer_rms(
        epochs=args.epochs,
        n_max=args.n_max,
        wavelength=args.wavelength,
        wfs_res=wfs_res,
        pupil_diameter=args.pupil_diameter,
        early_stop_threshold=args.early_stop_threshold,
        slm_number=args.slm_number,
        remove_tilt=args.remove_tilt,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
    )
    
    best_epoch, (best_rms, _) = recorder.get_best_iter()
    logger.info(f"Optimization complete. Best RMS: {best_rms:.4f} @ epoch {best_epoch}")