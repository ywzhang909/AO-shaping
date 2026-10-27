"""SLM command → effective phase nonlinearity (report 20261027 §4 item 2).

The SLM's phase response is not perfectly linear: LUT nonlinearity, panel
static phase, and quantization all distort the commanded phase.  A simple,
well-understood correction is a bounded periodic residual:

    phi_eff = phi + a * sin(phi) + b * sin(2*phi)

with the constraint that the derivative stays positive:

    d(phi_eff)/d(phi) = 1 + a * cos(phi) + 2*b * cos(2*phi) > 0

This is enforced by clamping a, b such that |a| + 2*|b| < 1.

This module is pure torch/numpy, no hardware or tools imports.  It is intended
to be applied inside ``defocus_phase`` (or as a post-processing step on the
panel phase) when a checkpoint flag enables it (default off).
"""
from __future__ import annotations

import math

import numpy as np


def slm_phase_nonlinearity(phi, a=0.0, b=0.0):
    """Apply bounded periodic residual to phase (radians).

    phi_eff = phi + a*sin(phi) + b*sin(2*phi)

    Parameters
    ----------
    phi : array-like or torch.Tensor
        Input phase in radians.
    a, b : float
        Residual coefficients.  Should satisfy |a| + 2*|b| < 1 for monotonicity.

    Returns
    -------
    Same type as input.
    """
    # Detect torch tensor.
    try:
        import torch
        if isinstance(phi, torch.Tensor):
            return phi + a * torch.sin(phi) + b * torch.sin(2.0 * phi)
    except ImportError:
        pass
    phi = np.asarray(phi, dtype=np.float64)
    return phi + a * np.sin(phi) + b * np.sin(2.0 * phi)


def check_monotonicity(a, b):
    """Check that the residual preserves monotonicity: |a| + 2*|b| < 1."""
    return abs(a) + 2.0 * abs(b) < 1.0


def clamp_coefficients(a, b, margin=0.9):
    """Clamp a, b to satisfy monotonicity with a safety margin."""
    scale = margin / max(abs(a) + 2.0 * abs(b), 1e-12)
    if scale < 1.0:
        a = a * scale
        b = b * scale
    return float(a), float(b)


def derivative(phi, a=0.0, b=0.0):
    """Derivative of the nonlinearity: 1 + a*cos(phi) + 2*b*cos(2*phi)."""
    try:
        import torch
        if isinstance(phi, torch.Tensor):
            return 1.0 + a * torch.cos(phi) + 2.0 * b * torch.cos(2.0 * phi)
    except ImportError:
        pass
    phi = np.asarray(phi, dtype=np.float64)
    return 1.0 + a * np.cos(phi) + 2.0 * b * np.cos(2.0 * phi)
