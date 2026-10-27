"""Gaussian illumination amplitude residual (report 20261027 §4 item 3).

The illumination amplitude is currently a single Gaussian.  Real illumination
has slow spatial variations (beam profile, vignetting, static phase).  The
report suggests expressing the amplitude as

    A = A0 * exp(r)          or
    A = softplus(logA0 + r)

where ``r`` is a low-frequency residual (few B-spline or Fourier modes, or a
small MLP with SiLU hidden layer and tanh-bounded output).  The residual
must not depend on the input phase (otherwise it absorbs propagation errors).

This module provides:
- ``AmplitudeResidual``: a low-frequency residual on a coarse grid, upsampled
  to the panel grid via bilinear interpolation.
- ``amplitude_with_residual``: combine base Gaussian with residual.

Pure torch/numpy, no hardware or tools imports.
"""
from __future__ import annotations

import numpy as np


class AmplitudeResidual:
    """Low-frequency amplitude residual on a coarse grid.

    Parameters
    ----------
    grid_size : int
        Panel grid size (e.g. 128).
    n_modes : int
        Number of Fourier modes per axis (residual has (2*n_modes+1)^2 coeffs).
    """

    def __init__(self, grid_size=128, n_modes=3):
        self.grid_size = int(grid_size)
        self.n_modes = int(n_modes)
        n_coeff = (2 * n_modes + 1) ** 2
        self.coeffs = np.zeros(n_coeff, dtype=np.float64)

    def evaluate(self, panel_x, panel_y):
        """Evaluate residual r(x,y) on the panel grid.

        Returns a (grid_size, grid_size) array.
        """
        # Build Fourier basis on the panel grid.
        # Normalise coordinates to [-1, 1].
        x = (panel_x - panel_x.mean()) / (panel_x.max() - panel_x.min() + 1e-9)
        y = (panel_y - panel_y.mean()) / (panel_y.max() - panel_y.min() + 1e-9)
        r = np.zeros_like(x, dtype=np.float64)
        idx = 0
        for ky in range(-self.n_modes, self.n_modes + 1):
            for kx in range(-self.n_modes, self.n_modes + 1):
                r = r + self.coeffs[idx] * np.cos(np.pi * (kx * x + ky * y))
                idx += 1
        return r

    def amplitude(self, base_amplitude, mode="exp"):
        """Combine base Gaussian amplitude with residual.

        mode='exp': A = base * exp(r)  (non-negative, multiplicative)
        mode='softplus': A = softplus(log(base) + r)  (additive in log space)
        """
        r = self.evaluate  # placeholder; caller passes evaluated r
        if mode == "exp":
            return base_amplitude * np.exp(r)
        elif mode == "softplus":
            import numpy as np
            log_base = np.log(np.clip(base_amplitude, 1e-12, None))
            # softplus(x) = log(1 + exp(x))
            x = log_base + r
            return np.log1p(np.exp(np.clip(x, None, 20)))  # stable softplus
        else:
            raise ValueError(f"unknown mode: {mode}")

    def set_coeffs(self, coeffs):
        self.coeffs = np.asarray(coeffs, dtype=np.float64).ravel()
        if self.coeffs.shape[0] != (2 * self.n_modes + 1) ** 2:
            raise ValueError(f"expected {(2*self.n_modes+1)**2} coeffs")


def amplitude_with_residual(base_amplitude, residual_r, mode="exp"):
    """Combine base Gaussian amplitude with a pre-evaluated residual.

    base_amplitude : array-like, shape (H, W)
    residual_r : array-like, shape (H, W)
    mode : 'exp' or 'softplus'
    """
    base = np.asarray(base_amplitude, dtype=np.float64)
    r = np.asarray(residual_r, dtype=np.float64)
    if mode == "exp":
        return base * np.exp(r)
    elif mode == "softplus":
        log_base = np.log(np.clip(base, 1e-12, None))
        x = log_base + r
        return np.log1p(np.exp(np.clip(x, None, 20)))
    else:
        raise ValueError(f"unknown mode: {mode}")
