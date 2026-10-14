"""Settle / drift / linearity characterisation criteria.

Pure numpy over plain sequences: no device, no panel geometry, no image I/O.
Moved verbatim from ``tools/kernels/bench_kernels.py`` (R-49 / #65). These are
the stability-judgement kernels the bench probes (``ccd.floor_probe``,
``ccd.drift_probe``) use to decide whether a measurement settled, whether
averaging K frames actually helps, and whether an exposure ladder stayed
linear.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def settle_time_s(
    deltas: Sequence[float],
    times_s: Sequence[float],
    *,
    frac: float = 0.10,
    run: int = 3,
) -> float | None:
    """First time the settle curve stays within ``frac`` of its final value.

    Replaces "sleep a fixed duration and hope". A fixed wait is invalid on this
    bench because the driver's flip-time estimate under-reports (it reports
    0.0 ms for two phases with similar grey statistics), so the same ramp read
    43.2 px FWHM immediately and 12.8 px three seconds later.

    Args:
        deltas: Observable (e.g. a norm) at each sample.
        times_s: Sample times, seconds.
        frac: Tolerance as a fraction of the final value.
        run: Number of consecutive samples that must all be within tolerance.

    Returns:
        The settle time, or ``None`` if it never settles.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    d = np.asarray(list(deltas), dtype=np.float64)
    t = np.asarray(list(times_s), dtype=np.float64)
    if d.shape != t.shape:
        raise ValueError("deltas and times_s must have the same length")
    if d.size == 0:
        return None
    tol = abs(float(frac)) * abs(float(d[-1]))
    if tol <= 0.0:
        # A zero-valued curve is trivially settled from the first sample.
        return float(t[0]) if abs(float(d[0])) <= 0.0 else None
    for i in range(d.size):
        window = d[i : i + int(run)]
        if window.size < int(run):
            break
        if bool(np.all(np.abs(window - d[-1]) <= tol)):
            return float(t[i])
    return None


def snr_vs_averages(
    signal_norms: Sequence[float],
    floor: float,
    ks: Sequence[int],
) -> dict[int, float]:
    """SNR at K-frame averages against a measured noise floor.

    Useful as a *diagnostic*: if averaging K frames does not improve the SNR,
    the residual is not independent read noise but drift, and averaging longer
    will not help. That distinction decides whether a bench needs a better
    exposure or a better settle protocol.

    Args:
        signal_norms: Observable for each of K identical repeats.
        floor: Single-frame noise floor (same observable, same units).
        ks: Averaging factors to report.

    Returns:
        ``{K: snr}``.

    Raises:
        ValueError: If ``floor`` is not positive.
    """
    f = float(floor)
    if not np.isfinite(f) or f <= 0.0:
        raise ValueError(f"floor must be finite and positive, got {floor!r}")
    x = np.asarray(list(signal_norms), dtype=np.float64)
    out: dict[int, float] = {}
    for k in ks:
        kk = int(k)
        if kk < 1:
            raise ValueError(f"K must be >= 1, got {k!r}")
        mean = float(x[:kk].mean()) if x.size >= kk else float(x.mean())
        out[kk] = mean / (f / np.sqrt(kk))
    return out


def exposure_monotonicity(
    exposures_ms: Sequence[float],
    peaks: Sequence[float],
    *,
    rel_tol: float = 0.02,
    saturation_level: float | None = None,
) -> dict[str, object]:
    """Check that peak brightness rises monotonically with exposure.

    Monotonicity is the cheap way to prove the camera is not being pushed past
    its linear range and that the laser is not drifting across the bracket.

    Args:
        exposures_ms: Exposure settings, ascending.
        peaks: Measured peak per exposure.
        rel_tol: Allowed relative shortfall on each step.
        saturation_level: Detector full-scale value; flags saturation when a
            peak reaches it.

    Returns:
        Dict with ``verdict``, ``ratios``, ``expected`` and ``saturated``.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    e = np.asarray(list(exposures_ms), dtype=np.float64)
    p = np.asarray(list(peaks), dtype=np.float64)
    if e.shape != p.shape:
        raise ValueError("exposures_ms and peaks must have the same length")
    if e.size < 2:
        return {
            "verdict": "insufficient_data",
            "ratios": [],
            "expected": [],
            "saturated": False,
        }

    with np.errstate(divide="ignore", invalid="ignore"):
        expected = e[1:] / e[:-1]
        ratios = p[1:] / p[:-1]
    ok = ratios >= (expected * (1.0 - float(rel_tol)))
    saturated = False
    if saturation_level is not None:
        saturated = bool(np.any(p >= float(saturation_level)))
    return {
        "verdict": "monotonic"
        if bool(np.all(ok)) and not saturated
        else "saturated"
        if saturated
        else "non_monotonic",
        "ratios": [float(v) for v in ratios],
        "expected": [float(v) for v in expected],
        "saturated": saturated,
    }
