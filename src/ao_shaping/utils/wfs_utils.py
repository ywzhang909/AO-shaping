"""WFS utility functions for wavefront sensor operations."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.ndimage import gaussian_filter
from loguru import logger

if TYPE_CHECKING:
    from ao_shaping.drivers.wfs.thorlab_wfs import WFSManager
    from ao_shaping.drivers.slm import ZernikeSLM


def flatten_slopes(dev_x: np.ndarray, dev_y: np.ndarray) -> np.ndarray:
    """Flatten x/y deviation arrays into single slope vector.

    Args:
        dev_x: X deviation array of shape (nx, ny)
        dev_y: Y deviation array of shape (nx, ny)

    Returns:
        Concatenated flatten [dev_x; dev_y] vector
    """
    return np.concatenate([dev_x.flatten(), dev_y.flatten()])


def unflatten_slopes(slopes: np.ndarray, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Unflatten slope vector back to x/y deviation arrays.

    Args:
        slopes: Flattened slope vector
        nx: Number of spots in x
        ny: Number of spots in y

    Returns:
        tuple: (dev_x, dev_y) arrays each shape (nx, ny)
    """
    n = nx * ny
    return slopes[:n].reshape(nx, ny), slopes[n:].reshape(nx, ny)


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> tuple[float, float]:
    """Compute SNR in linear and dB scale.

    Args:
        signal: Signal vector
        noise: Noise vector (same shape)

    Returns:
        tuple: (snr_linear, snr_db)
    """
    snr_linear = float(np.linalg.norm(signal) / (np.linalg.norm(noise) + 1e-10))
    snr_db = 20 * np.log10(snr_linear + 1e-10)
    return snr_linear, snr_db


class DitheredReference:
    """Dithered reference measurement for SLM phase averaging.

    Applies sub-wavelength random phase dithering to average out
    pixelation steps and liquid crystal local relaxation errors.
    """

    def __init__(
        self,
        slm,
        dither_amp: float = 0.03,
        n_dither: int = 30,
        wait_time: float = 0.05,
    ):
        """Initialize dithered reference.

        Args:
            slm: ZernikeSLM instance
            dither_amp: Dithering amplitude in wavelength units (0.02-0.05 typical)
            n_dither: Number of dithering samples to average
            wait_time: Wait time after loading phase (seconds)
        """
        self.slm = slm
        self._slm = slm._slm
        self.dither_amp = dither_amp
        self.n_dither = n_dither
        self.wait_time = wait_time

    def measure(
        self,
        wfs: WFSManager,
        base_phase: np.ndarray | None = None,
        n_averages: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Measure reference slopes with dithering average.

        Args:
            wfs: WFS instance with get_spot_deviation method
            base_phase: Base phase to add dither to (None = zero flat)
            n_averages: Number of samples (uses self.n_dither if None)

        Returns:
            tuple: (s_ref, diagnostics)
                - s_ref: Median reference slopes
                - diagnostics: dict with snr, std, n_samples
        """
        if base_phase is None:
            h, w = self._slm.Panel_Res[1], self._slm.Panel_Res[0]
            base_phase = np.zeros((h, w), dtype=np.float64)

        n_samples = n_averages if n_averages is not None else self.n_dither
        slopes_list = []
        for _ in range(n_samples):
            noise = np.random.randn(*base_phase.shape)
            noise = gaussian_filter(noise, sigma=20)
            noise = noise / np.std(noise) * self.dither_amp * 2 * np.pi

            phase_rad = (base_phase + noise) % (2 * np.pi)
            phase_gray = self._slm._phase_to_gray(phase_rad)
            self._slm.display_data(phase_gray)
            time.sleep(self.wait_time)

            dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=False)
            s = flatten_slopes(dev_x, dev_y)
            slopes_list.append(s)

        slopes_arr = np.array(slopes_list)
        s_ref = np.median(slopes_arr, axis=0)
        std = np.std(slopes_arr, axis=0)

        snr_linear, snr_db = compute_snr(s_ref, std)

        diagnostics = {
            "n_samples": n_samples,
            "dither_amp": self.dither_amp,
            "snr_linear": snr_linear,
            "snr_db": snr_db,
            "std_mean": float(np.mean(std)),
        }

        logger.info(f"Dithered reference SNR: {snr_db:.1f} dB (n={n_samples})")

        return s_ref, diagnostics


# ==================== Debug-data saving helpers ====================


def save_debug_data(
    base_dir: Path,
    idx: int,
    cycle: int,
    sample: int,
    arrays: dict[str, np.ndarray | None],
    meta: dict,
) -> None:
    """Save per-measurement debug data; encompasses directory creation, numpy
    saves, and a metadata sidecar (JSON).

    Arrays whose value is ``None`` are silently skipped so callers can build
    a dict unconditionally without ``if`` guards.

    Args:
        base_dir: Root directory for all debug data (created if absent).
        idx: Zero-based mode or actuator index (embedded in the path).
        cycle: Zero-based cycle repetition index.
        sample: Zero-based sample index within the cycle.
        arrays: Mapping from filename-suffix (no extension) to array value.
            ``None`` values are skipped.
        meta: Arbitrary metadata dict written as ``meta.json`` alongside the
            arrays.
    """
    idx_dir = base_dir / f"mode_{idx:03d}"
    run_dir = idx_dir / f"cycle_{cycle}"
    sign_dir = run_dir / meta["_sign"]
    sign_dir.mkdir(parents=True, exist_ok=True)

    for name, arr in arrays.items():
        if arr is not None:
            np.save(sign_dir / f"sample_{sample:03d}_{name}.npy", arr)

    full_meta = {k: v for k, v in meta.items() if not k.startswith("_")}
    with open(sign_dir / f"sample_{sample:03d}_meta.json", "w") as f:
        json.dump(full_meta, f, default=str)


def make_mode_debug_callback(
    debug_data_dir: Path,
    *,
    get_arrays: Callable[..., dict[str, np.ndarray | None]],
    get_meta: Callable[..., dict],
) -> Callable:
    """Factory that builds the inner ``debug_callback`` for Zernike-like
    mode-sweep calibrations.

    Compared to a raw closure, this halves the amount of duplicated filesystem
    code and makes the two matrix-runner branches (Zernike / DM) differ only in
    *what they save* (the ``get_arrays`` / ``get_meta`` callables) rather than
    *how they save it*.

    Args:
        debug_data_dir: Root debug-save directory.
        get_arrays: Callable with signature
            ``(slm_phase, shift, dev_x, dev_y, zernike) -> dict[str, ndarray|None]``.
        get_meta: Callable with signature
            ``(mode_index, cycle, sample, shift, is_plus) -> dict``.
            Keys starting with ``_`` are treated as internal-only and stripped
            before writing the JSON sidecar.

    Returns:
        A ``debug_callback`` function matching the Zernike matrix-runner
        signature.
    """
    def debug_callback(
        mode_index: int,
        cycle: int,
        sample: int,
        slm_phase: np.ndarray,
        shift_x: int,
        shift_y: int,
        deviation_x: np.ndarray,
        deviation_y: np.ndarray,
        zernike_coeffs: np.ndarray,
        is_plus: bool,
    ) -> None:
        save_debug_data(
            base_dir=debug_data_dir,
            idx=mode_index,
            cycle=cycle,
            sample=sample,
            arrays=get_arrays(slm_phase, shift_x, shift_y,
                               deviation_x, deviation_y, zernike_coeffs),
            meta={
                "_sign": "plus" if is_plus else "minus",
                "mode_index": mode_index,
                "cycle": cycle,
                "sample": sample,
                "shift_x": shift_x,
                "shift_y": shift_y,
                "is_plus": is_plus,
                **get_meta(mode_index, cycle, sample, shift_x, shift_y),
            },
        )
    return debug_callback


def make_actuator_debug_callback(
    debug_data_dir: Path,
) -> Callable:
    """Factory that builds the inner ``debug_callback`` for DM actuator-sweep
    calibrations.

    The DM variant saves per-actuator deviation arrays and the applied voltage.
    The directory layout mirrors :func:`make_mode_debug_callback` (``idx``
    becomes the actuator number and ``sign`` is ``plus`` / ``minus``).

    Args:
        debug_data_dir: Root debug-save directory.

    Returns:
        A ``debug_callback`` function matching the *DM matrix-runner*
        signature.
    """
    def debug_callback(
        actuator_idx: int,
        cycle: int,
        sample: int,
        deviation_x: np.ndarray,
        deviation_y: np.ndarray,
        voltage: float,
        is_plus: bool,
    ) -> None:
        save_debug_data(
            base_dir=debug_data_dir,
            idx=actuator_idx,
            cycle=cycle,
            sample=sample,
            arrays={
                "deviation_x": deviation_x,
                "deviation_y": deviation_y,
            },
            meta={
                "_sign": "plus" if is_plus else "minus",
                "actuator_idx": actuator_idx,
                "cycle": cycle,
                "sample": sample,
                "voltage": voltage,
                "is_plus": is_plus,
            },
        )
    return debug_callback
