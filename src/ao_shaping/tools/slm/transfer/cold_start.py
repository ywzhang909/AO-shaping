"""Untrained shaping seeds: no phase retrieval or historical checkpoints."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def cold_phase(forward, centre, *, kind="plane", seed=102626):
    """Add a measured steering carrier to a flat or random shaping phase.

    The carrier only places the requested signal away from zero order. No
    fitted static aberration is subtracted and no target amplitude is retrieved.
    """
    if kind not in {"plane", "random"}:
        raise ValueError("cold phase must be plane or random")
    p = forward.params
    yy, xx = np.mgrid[:p.grid_size, :p.grid_size] * forward.pitch
    q = np.linalg.solve(np.asarray(p.ramp_matrix), np.asarray(centre)-p.origin_xy)
    carrier = 2*np.pi*(q[0]*xx+q[1]*yy)
    shaping = np.zeros_like(carrier)
    if kind == "random":
        shaping = gaussian_filter(np.random.default_rng(seed).normal(size=carrier.shape), 2.)
        shaping -= shaping.mean()
        shaping /= max(float(shaping.std()), 1e-9)
    phase = np.asarray(carrier+shaping, dtype=np.float32)
    return phase, dict(kind=kind, random_seed=seed, gs_calls=0,
                      historical_phase=False, static_correction=False,
                      shaping_rms_rad=float(shaping.std()), carrier_cycles_per_pixel=q.tolist())
