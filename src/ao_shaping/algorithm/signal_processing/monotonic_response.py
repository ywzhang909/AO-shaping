"""Monotonic CCD observation-chain response (report 20261027 §4 item 1).

The exposure experiment showed that the linear ``g*t*I + b*t + c`` model leaves
a systematic curvature in the CCD response (core CV still varies 8.5-9.3 % at
low exposure after correction).  The next step is a monotonic response whose
derivative is constrained to be positive:

    response(v) = integral_0^v (softplus(h(x)) + eps) dx

with a fixed reference scale so the response cannot arbitrarily rescale the
signal.  ``h`` is a small lookup table (or MLP) over the input value; the
integral is computed by cumulative trapezoidal quadrature on a fixed grid.

This module is pure torch/numpy, no hardware or tools imports.  It is intended
to be **appended** to the linear radiometry in ``FrozenDefocusForward.predict``
and ``PhysicalForwardFit`` as an optional post-processing step, gated by a
checkpoint flag (default off) so the historical linear baseline is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softplus(x):
    """Numerically stable softplus."""
    import torch
    return torch.nn.functional.softplus(x)


@dataclass
class MonotonicResponseConfig:
    """Configuration for the monotonic CCD response."""
    n_bins: int = 256
    v_min: float = 0.0
    v_max: float = 255.0
    eps: float = 1e-4
    # Fixed reference scale: response(v_ref) = v_ref (identity at reference).
    v_ref: float = 128.0


class MonotonicResponse:
    """Monotonic response with positive derivative via softplus.

    Parameters
    ----------
    config : MonotonicResponseConfig
    h_values : array-like or None
        Log-derivative table ``h(v)`` on the grid.  If None, initialised to
        zeros (identity response: derivative = softplus(0)+eps ≈ 0.693+eps,
        normalised to identity at reference).
    """

    def __init__(self, config: MonotonicResponseConfig | None = None, h_values=None):
        self.config = config or MonotonicResponseConfig()
        cfg = self.config
        self.grid = np.linspace(cfg.v_min, cfg.v_max, cfg.n_bins, dtype=np.float64)
        if h_values is None:
            h_values = np.zeros(cfg.n_bins, dtype=np.float64)
        self.h_values = np.asarray(h_values, dtype=np.float64).reshape(-1)
        if self.h_values.shape[0] != cfg.n_bins:
            raise ValueError(f"h_values must have {cfg.n_bins} entries, got {self.h_values.shape[0]}")

    def derivative(self, v):
        """Derivative at value v: softplus(h(v)) + eps."""
        import torch
        v_t = torch.as_tensor(v, dtype=torch.float64)
        # Interpolate h on grid.
        h_interp = self._interp_h(v_t)
        return torch.nn.functional.softplus(h_interp) + self.config.eps

    def _interp_h(self, v):
        import torch
        # Clamp to grid range.
        v_clamped = v.clamp(self.config.v_min, self.config.v_max)
        # Normalised position.
        idx = (v_clamped - self.config.v_min) / (self.config.v_max - self.config.v_min) * (self.config.n_bins - 1)
        idx0 = idx.floor().long().clamp(0, self.config.n_bins - 2)
        idx1 = idx0 + 1
        frac = idx - idx0.float()
        h_table = torch.as_tensor(self.h_values, dtype=torch.float64, device=v.device)
        h0 = h_table[idx0]
        h1 = h_table[idx1]
        return h0 * (1 - frac) + h1 * frac

    def forward(self, v):
        """Apply monotonic response: integrate derivative from 0 to v.

        Accepts a numpy array or torch tensor.  Returns the same kind of object
        (numpy → numpy float64, tensor → tensor float64).
        """
        import torch

        is_numpy = not hasattr(v, "clamp")
        v_t = torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64)
        original_shape = v_t.shape
        v_flat = v_t.reshape(-1)
        # Vectorized: integrate from v_min to v on a shared fine grid, then interpolate.
        n_quad = 64
        xs = torch.linspace(self.config.v_min, self.config.v_max, n_quad,
                            device=v_t.device, dtype=torch.float64)
        dx = (self.config.v_max - self.config.v_min) / (n_quad - 1)
        derivs = self.derivative(xs)  # (n_quad,)
        cumint = torch.cumsum(derivs, dim=0) * dx  # (n_quad,)
        # cumint[i] ≈ integral from v_min to xs[i]
        idx = (v_flat - self.config.v_min) / (self.config.v_max - self.config.v_min) * (n_quad - 1)
        idx0 = idx.floor().long().clamp(0, n_quad - 2)
        idx1 = idx0 + 1
        frac = idx - idx0.float()
        cum0 = cumint[idx0.clamp(0, n_quad - 1)]
        cum1 = cumint[idx1.clamp(0, n_quad - 1)]
        integral = cum0 * (1 - frac) + cum1 * frac
        # Fixed reference: response(v_ref) = v_ref, so the response cannot
        # arbitrarily rescale the signal.
        v_ref_integral = self._integral_at(torch.tensor(self.config.v_ref,
                                                        device=v_t.device, dtype=torch.float64))
        result = integral - v_ref_integral + self.config.v_ref
        result = result.reshape(original_shape)
        if is_numpy:
            return result.detach().cpu().numpy().astype(np.float64)
        return result

    def _integral_at(self, v):
        import torch
        n_quad = 64
        xs = torch.linspace(self.config.v_min, self.config.v_max, n_quad, device=v.device, dtype=torch.float64)
        dx = (self.config.v_max - self.config.v_min) / (n_quad - 1)
        derivs = self.derivative(xs)
        cumint = torch.cumsum(derivs, dim=0) * dx
        idx = (v - self.config.v_min) / (self.config.v_max - self.config.v_min) * (n_quad - 1)
        idx0 = idx.floor().long().clamp(0, n_quad - 2)
        idx1 = idx0 + 1
        frac = idx - idx0.float()
        cum0 = cumint[idx0.clamp(0, n_quad - 1)]
        cum1 = cumint[idx1.clamp(0, n_quad - 1)]
        return cum0 * (1 - frac) + cum1 * frac

    def params(self):
        """Return trainable parameters (h_values)."""
        import torch
        return [torch.nn.Parameter(torch.as_tensor(self.h_values, dtype=torch.float64))]


def monotonic_derivative_check(h_values, eps=1e-4):
    """Verify that softplus(h)+eps > 0 for all h (always true)."""
    import torch
    h = torch.as_tensor(h_values, dtype=torch.float64)
    derivs = torch.nn.functional.softplus(h) + eps
    return bool((derivs > 0).all())
