"""WFS utility functions for wavefront sensor operations."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter
from loguru import logger

if TYPE_CHECKING:
    from ao_shaping.drivers.wfs.thorlab_wfs import WFSManager


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