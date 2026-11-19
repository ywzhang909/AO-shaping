"""Issue #238: offline fit of the binary phase-grating sweep -> physical 2*pi gray.

Reads a run directory written by
``scripts/slm/issue238_gray_phase_probes.py --mode binary`` and fits the exact
binary-grating Fourier model to the measured 0 / +-1 order powers:

    eta_0(phi) = d^2 + (1-d)^2 + 2 d (1-d) cos(phi)
    eta_1(phi) = (4/pi^2) sin^2(phi/2) sin^2(pi d)          (per +-1 order)
    phi(g)     = 2*pi * (g / g2pi) ** gamma

with ``(g2pi, gamma)`` free and ``duty`` optionally free.

Why this breaks the degeneracy of the blaze fit
-----------------------------------------------
The blaze (sawtooth) far field depends on the *shape* of the within-period ramp,
so the fitted 2*pi gray trades off against gamma and against a visibility factor.
A binary 0/g grating is a two-level grating whose far field is the exact Fourier
result above: it depends on the phase depth only through ``cos(phi)`` and
``sin^2(phi/2)``.  Both are 2*pi-periodic, so the same model holds for g beyond
g2pi (no branch ambiguity) and the absolute phase at every gray is pinned by the
*ratio* of the 0-order to the +-1 order power -- amplitude free, gamma free.

No devices, no network, no checkpoint.  Writes ``fit.json`` next to the run and
prints a markdown summary.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

#: Integration half-width (px) around each order centre.
APERTURE_R = 30
#: Half-width (px) of the box searched for the 0-order peak on the flat frame.
ZERO_SEARCH_R = 220
#: Aperture radius (px) blanked out before looking for the +-1 orders.
ZERO_BLANK_R = 120
#: Tolerance (px) between a detected order and the prior geometry from the v1 run.
GEOMETRY_TOL_PX = 60


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def binary_eta(phi: np.ndarray, duty: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Exact 0-order and per-+-1-order efficiencies of a binary phase grating."""
    d = float(duty)
    phi = np.asarray(phi, dtype=float)
    eta0 = d**2 + (1.0 - d) ** 2 + 2.0 * d * (1.0 - d) * np.cos(phi)
    eta1 = (4.0 / math.pi**2) * np.sin(phi / 2.0) ** 2 * math.sin(math.pi * d) ** 2
    return np.clip(eta0, 0.0, None), np.clip(eta1, 0.0, None)


def model_fractions(
    gray: np.ndarray, g2pi: float, gamma: float, duty: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Model fractions of the measured (eta0 + 2*eta1) sum for each gray."""
    gray = np.asarray(gray, dtype=float)
    phi = np.zeros_like(gray)
    nz = gray > 0
    phi[nz] = 2.0 * math.pi * np.power(gray[nz] / float(g2pi), float(gamma))
    eta0, eta1 = binary_eta(phi, duty)
    total = eta0 + 2.0 * eta1
    return eta0 / total, eta1 / total


def fit_g2pi(
    gray: np.ndarray,
    e0n: np.ndarray,
    e1n: np.ndarray,
    *,
    duty: float = 0.5,
    x0: tuple[float, float] = (950.0, 0.8),
) -> dict[str, Any]:
    """Least-squares fit of ``(g2pi, gamma)`` directly in the measured e-domain."""
    from scipy.optimize import least_squares

    gray = np.asarray(gray, dtype=float)
    e0n = np.asarray(e0n, dtype=float)
    e1n = np.asarray(e1n, dtype=float)
    keep = gray > 0

    def resid(p: np.ndarray) -> np.ndarray:
        p0, p1 = model_fractions(gray[keep], p[0], p[1], duty)
        return np.concatenate([p0 - e0n[keep], p1 - e1n[keep]])

    sol = least_squares(resid, list(x0), bounds=([400.0, 0.2], [1400.0, 2.5]))
    r = resid(sol.x)
    dof = max(len(r) - 2, 1)
    rms = float(np.sqrt(np.sum(r**2) / dof))
    # Linearised 1-sigma covariance from the Jacobian.
    try:
        jtj = sol.jac.T @ sol.jac
        cov = np.linalg.inv(jtj) * (float(np.sum(r**2)) / dof)
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        sd = np.array([np.nan, np.nan])
    return {
        "g2pi": float(sol.x[0]),
        "gamma": float(sol.x[1]),
        "g2pi_sd": float(sd[0]),
        "gamma_sd": float(sd[1]),
        "rms": rms,
        "n_points": int(keep.sum()),
        "duty": float(duty),
        "success": bool(sol.success),
    }


def g2pi_direct_fold(
    gray: np.ndarray, e0n: np.ndarray, e1n: np.ndarray
) -> dict[str, Any]:
    """gamma-independent estimator: the fold-back zero of the recovered phase.

    For a 50% duty binary grating the exact Fourier result gives

        tan^2(phi/2) = (pi^2 / 4) * (eta_1 / eta_0)

    so ``phi(g) = 2*atan((pi/2)*sqrt(e1n/e0n))`` is recovered point by point with
    NO amplitude and NO gamma assumption.  It rises to pi at the eta_1 peak and
    folds back to 0 at phi = 2*pi, i.e. exactly at g = g2pi.  Parabolic
    interpolation of the (V-shaped) minimum in the high-gray dense band gives
    g2pi directly.  This is the handle that breaks the blaze-fit degeneracy.
    """
    gray = np.asarray(gray, dtype=float)
    e0n = np.asarray(e0n, dtype=float)
    e1n = np.asarray(e1n, dtype=float)
    ratio = np.clip(e1n / np.clip(e0n, 1e-9, None), 0.0, None)
    fold = 2.0 * np.arctan((math.pi / 2.0) * np.sqrt(ratio))

    # The eta_1 peak is where fold is maximal; the fold-back zero is at high gray.
    k_peak = int(np.argmax(e1n))
    hi = np.arange(len(gray)) > k_peak
    if hi.sum() < 3:
        return {"available": False, "reason": "too few points past eta1 peak"}
    idx = np.where(hi)[0]
    k_min_local = int(np.argmin(fold[idx]))
    k = int(idx[k_min_local])
    if k == 0 or k == len(gray) - 1:
        # Minimum at the band edge: report the edge, flag it as a bound.
        return {
            "available": True,
            "g2pi": float(gray[k]),
            "fold_min": float(fold[k]),
            "at_edge": True,
            "k_peak_gray": float(gray[k_peak]),
        }
    # Parabolic vertex through (g[k-1],fold[k-1]),(g[k],fold[k]),(g[k+1],fold[k+1]).
    g0, g1, g2 = gray[k - 1], gray[k], gray[k + 1]
    f0, f1, f2 = fold[k - 1], fold[k], fold[k + 1]
    coeffs = np.polyfit([g0, g1, g2], [f0, f1, f2], 2)
    a = float(coeffs[0])
    vertex = float(-coeffs[1] / (2.0 * a)) if abs(a) > 1e-12 else float(g1)
    vertex = float(np.clip(vertex, g0, g2))
    return {
        "available": True,
        "g2pi": vertex,
        "fold_min": float(f1),
        "at_edge": False,
        "k_peak_gray": float(gray[k_peak]),
        "bracket": [float(g0), float(g2)],
    }


def fit_g2pi_free_duty(
    gray: np.ndarray,
    e0n: np.ndarray,
    e1n: np.ndarray,
    *,
    x0: tuple[float, float, float] = (950.0, 0.8, 0.5),
) -> dict[str, Any]:
    """Fit ``(g2pi, gamma, duty)`` to check the as-written 50% duty assumption."""
    from scipy.optimize import least_squares

    gray = np.asarray(gray, dtype=float)
    keep = gray > 0

    def resid(p: np.ndarray) -> np.ndarray:
        p0, p1 = model_fractions(gray[keep], p[0], p[1], p[2])
        return np.concatenate(
            [p0 - np.asarray(e0n)[keep], p1 - np.asarray(e1n)[keep]]
        )

    sol = least_squares(
        resid,
        list(x0),
        bounds=([400.0, 0.2, 0.25], [1400.0, 2.5, 0.75]),
    )
    r = resid(sol.x)
    return {
        "g2pi": float(sol.x[0]),
        "gamma": float(sol.x[1]),
        "duty": float(sol.x[2]),
        "rms": float(np.sqrt(np.sum(r**2) / max(len(r) - 3, 1))),
        "success": bool(sol.success),
    }


# ---------------------------------------------------------------------------
# Frame I/O and order photometry
# ---------------------------------------------------------------------------
def read_rows(run: Path) -> list[dict[str, Any]]:
    path = run / "frames.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_frame(run: Path, row: dict[str, Any]) -> np.ndarray:
    with np.load(run / row["frame_file"]) as data:
        return np.asarray(data["frame"], dtype=np.float32)


def _box(frame: np.ndarray, cy: int, cx: int, r: int) -> float:
    h, w = frame.shape
    y0, y1 = max(cy - r, 0), min(cy + r + 1, h)
    x0, x1 = max(cx - r, 0), min(cx + r + 1, w)
    return float(frame[y0:y1, x0:x1].sum())


def _smoothed(frame: np.ndarray, k: int = 9) -> np.ndarray:
    """Box-smoothed frame (spot ~30 px) so hot pixels cannot fake an order."""
    from scipy.ndimage import uniform_filter

    return uniform_filter(frame, size=k, mode="constant", cval=0.0)


def detect_orders(run: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Locate the 0 order (flat frame) and the +-1 orders (strongest sweep frame).

    Uses a smoothed probe, excludes a border band, and places the second first
    order by symmetry about the 0 order (valid because the binary grating is a
    symmetric two-level pattern -> equal +1 and -1).
    """
    ref = [r for r in rows if r.get("group") == "reference"]
    if not ref:
        raise RuntimeError("no reference flat frame in the run")
    flat = load_frame(run, ref[0])
    bg = float(np.median(flat))
    flat_s = _smoothed(flat)
    peak_flat = np.unravel_index(int(np.argmax(flat_s)), flat_s.shape)
    cy, cx = int(peak_flat[0]), int(peak_flat[1])

    # Probe = level with most power outside the 0-order box (near eta_1 peak).
    best: tuple[float, dict[str, Any]] | None = None
    seen: set[int] = set()
    for row in rows:
        if row.get("group") != "sweep":
            continue
        gray = int(row["gray"])
        if gray in seen:
            continue
        seen.add(gray)
        frame = load_frame(run, row)
        outside = float(frame.sum()) - _box(frame, cy, cx, ZERO_BLANK_R)
        if best is None or outside > best[0]:
            best = (outside, row)
    if best is None:
        raise RuntimeError("no sweep frames in the run")
    probe = load_frame(run, best[1])
    sm = _smoothed(probe)

    h, w = sm.shape
    border = 80
    search = np.zeros_like(sm)
    search[border : h - border, border : w - border] = sm[
        border : h - border, border : w - border
    ]
    # Blank out the 0-order region.
    search[max(cy - ZERO_BLANK_R, 0) : cy + ZERO_BLANK_R, :] = 0.0

    # First-order spot = brightest surviving blob (smoothed -> robust to hot px).
    p1 = np.unravel_index(int(np.argmax(search)), search.shape)
    a = (int(p1[0]), int(p1[1]))
    # Second first order by symmetry about the 0 order (binary = symmetric).
    mirror = (2 * cy - a[0], 2 * cx - a[1])
    if 0 <= mirror[0] < h and 0 <= mirror[1] < w:
        b = mirror
        # Verify there is real power at the mirror location.
        if sm[b[0], b[1]] < 0.25 * sm[a[0], a[1]]:
            # Fallback: independent brightest blob away from a and zero.
            search2 = search.copy()
            search2[max(a[0] - ZERO_BLANK_R, 0) : a[0] + ZERO_BLANK_R, :] = 0.0
            p2 = np.unravel_index(int(np.argmax(search2)), search2.shape)
            b = (int(p2[0]), int(p2[1]))
    else:
        b = a

    return {
        "background": bg,
        "zero": {"row": cy, "col": cx, "peak_flat": float(flat[peak_flat])},
        "plus1": {"row": a[0], "col": a[1], "peak": float(sm[a])},
        "minus1": {"row": b[0], "col": b[1], "peak": float(sm[b])},
        "probe_gray": int(best[1]["gray"]),
        "separation_px": int(
            abs(a[0] - b[0]) + abs(a[1] - b[1])
        ),
    }


def level_photometry(
    run: Path,
    rows: list[dict[str, Any]],
    orders: dict[str, Any],
    aperture_r: int = APERTURE_R,
) -> dict[str, Any]:
    """Per-gray, per-frame background-subtracted 0 / +-1 order powers."""
    bg = float(orders["background"])
    cy, cx = orders["zero"]["row"], orders["zero"]["col"]
    py, px = orders["plus1"]["row"], orders["plus1"]["col"]
    my, mx = orders["minus1"]["row"], orders["minus1"]["col"]

    grays: list[int] = []
    per_frame: dict[int, list[tuple[float, float, float]]] = {}
    for row in rows:
        if row.get("group") != "sweep":
            continue
        gray = int(row["gray"])
        frame = load_frame(run, row)
        n_pix = float((2 * aperture_r + 1) ** 2)
        p0 = _box(frame, cy, cx, aperture_r) - bg * n_pix
        pp = _box(frame, py, px, aperture_r) - bg * n_pix
        pm = _box(frame, my, mx, aperture_r) - bg * n_pix
        if gray not in per_frame:
            per_frame[gray] = []
            grays.append(gray)
        per_frame[gray].append((max(p0, 0.0), max(pp, 0.0), max(pm, 0.0)))

    grays.sort()
    order = [g for g in grays]
    stacked = np.array([per_frame[g] for g in order], dtype=float)  # (G, N, 3)
    return {"grays": order, "frames": stacked}


def fractions(powers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised 0 / +-1 fractions from (..., 3) powers."""
    total = powers[..., 0] + powers[..., 1] + powers[..., 2]
    total = np.clip(total, 1e-9, None)
    return powers[..., 0] / total, (powers[..., 1] + powers[..., 2]) / 2.0 / total


def _parabolic_vertex(x: np.ndarray, y: np.ndarray, index: int) -> float:
    """Three-point local quadratic vertex, clamped to its sampled bracket."""
    if index <= 0 or index >= len(x) - 1:
        return float(x[index])
    coeffs = np.polyfit(x[index - 1 : index + 2], y[index - 1 : index + 2], 2)
    if abs(float(coeffs[0])) < 1e-12:
        return float(x[index])
    return float(np.clip(-coeffs[1] / (2.0 * coeffs[0]), x[index - 1], x[index + 1]))


def landmark_estimates(
    gray: np.ndarray, e1: np.ndarray, powers: np.ndarray, *, dense_lo: int = 864
) -> dict[str, Any]:
    """Extract gamma-free first-order minima and conditional peak/fold landmarks.

    The order-specific minima expose systematic asymmetry hidden by averaging.
    The reported gamma is conditional on interpreting the e1 maximum/minimum as
    the physical pi/2pi landmarks; it is not an independent absolute phase readout.
    """
    gray = np.asarray(gray, dtype=float)
    e1 = np.asarray(e1, dtype=float)
    powers = np.asarray(powers, dtype=float)
    high = np.flatnonzero(gray >= dense_lo)
    if high.size < 3:
        return {"available": False, "reason": "too few high-gray points"}

    peak_i = int(np.argmax(e1))
    peak_gray = _parabolic_vertex(gray, -e1, int(np.argmin(-e1)))
    # Use the averaged-order direct fold as the central phase landmark.
    e0, _ = fractions(powers.mean(axis=1))
    fold = g2pi_direct_fold(gray, e0, e1)
    order_results: dict[str, Any] = {}
    vertices: list[float] = []
    for name, channel in (("plus1", 1), ("minus1", 2)):
        level_power = powers[:, :, channel].mean(axis=1)
        local_i = int(high[np.argmin(level_power[high])])
        vertex = _parabolic_vertex(gray, level_power, local_i)
        vertices.append(vertex)
        order_results[name] = {
            "sampled_min_gray": float(gray[local_i]),
            "vertex_gray": vertex,
            "min_power": float(level_power[local_i]),
        }

    if fold.get("available"):
        g2pi = float(fold["g2pi"])
        gamma = (
            float(math.log(2.0) / math.log(g2pi / peak_gray))
            if g2pi > peak_gray > 0
            else float("nan")
        )
        gamma_range = [
            float(math.log(2.0) / math.log(max(vertices) / peak_gray)),
            float(math.log(2.0) / math.log(min(vertices) / peak_gray)),
        ]
    else:
        g2pi = float("nan")
        gamma = float("nan")
        gamma_range = [float("nan"), float("nan")]

    return {
        "available": bool(fold.get("available")),
        "peak_pi_gray": peak_gray,
        "fold_2pi_gray": g2pi,
        "conditional_gamma": gamma,
        "conditional_gamma_systematic_range": gamma_range,
        "individual_order_minima": order_results,
        "individual_order_vertex_mean": float(np.mean(vertices)),
        "systematic_envelope": [float(min(vertices)), float(max(vertices))],
        "interpretation": "conditional landmark estimate; +/-1 asymmetry is systematic",
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyse(run: Path, bootstrap: int = 200, seed: int = 20261119) -> dict[str, Any]:
    rows = read_rows(run)
    orders = detect_orders(run, rows)
    photo = level_photometry(run, rows, orders)
    gray = np.array(photo["grays"], dtype=float)
    frames = photo["frames"]  # (G, N, 3)

    e0_all, e1_all = fractions(frames)
    e0n = e0_all.mean(axis=1)
    e1n = e1_all.mean(axis=1)
    # Standard error of the mean over frames, as a noise scale for the fit.
    sem0 = e0_all.std(axis=1, ddof=1) / math.sqrt(e0_all.shape[1])
    sem1 = e1_all.std(axis=1, ddof=1) / math.sqrt(e1_all.shape[1])

    fold = g2pi_direct_fold(gray, e0n, e1n)
    landmarks = landmark_estimates(gray, e1n, frames)
    fit = fit_g2pi(gray, e0n, e1n)
    fit_free = fit_g2pi_free_duty(
        gray, e0n, e1n, x0=(fit["g2pi"], fit["gamma"], 0.5)
    )

    rng = np.random.default_rng(seed)
    fit_boot: list[float] = []
    fold_boot: list[float] = []
    n_frames = e0_all.shape[1]
    for _ in range(bootstrap):
        idx = rng.integers(0, n_frames, size=n_frames)
        b0 = e0_all[:, idx].mean(axis=1)
        b1 = e1_all[:, idx].mean(axis=1)
        try:
            fit_boot.append(fit_g2pi(gray, b0, b1)["g2pi"])
            bfold = g2pi_direct_fold(gray, b0, b1)
            if bfold.get("available"):
                fold_boot.append(float(bfold["g2pi"]))
        except Exception:  # pragma: no cover - numerical guard
            continue
    fit_boot_arr = np.array(fit_boot, dtype=float)
    fold_boot_arr = np.array(fold_boot, dtype=float)
    fit_ci = (
        [float(np.percentile(fit_boot_arr, 2.5)), float(np.percentile(fit_boot_arr, 97.5))]
        if fit_boot_arr.size >= 20
        else [float("nan"), float("nan")]
    )
    fold_ci = (
        [float(np.percentile(fold_boot_arr, 2.5)), float(np.percentile(fold_boot_arr, 97.5))]
        if fold_boot_arr.size >= 20
        else [float("nan"), float("nan")]
    )

    # Residuals of the best fit, per gray (diagnostic for saturation/wrap issues).
    p0, p1 = model_fractions(gray, fit["g2pi"], fit["gamma"])
    resid0 = e0n - p0
    resid1 = e1n - p1

    # Duty-cycle sensitivity: refit assuming a range of duty values.
    duty_scan = {
        f"{d:.2f}": fit_g2pi(gray, e0n, e1n, duty=d, x0=(fit["g2pi"], fit["gamma"]))["g2pi"]
        for d in (0.44, 0.47, 0.50, 0.53, 0.56)
    }

    # Drift monitor: post-sweep flat vs the reference flat (same flat pattern).
    drift = _drift(run, rows, orders)

    # Where does the measured eta1 peak sit, and what does it imply?
    k = int(np.argmax(e1n))
    implied_peak_phase = float(2.0 * math.pi * (gray[k] / fit["g2pi"]) ** fit["gamma"])

    return {
        "run": str(run),
        "orders": orders,
        "n_gray_levels": int(len(gray)),
        "frames_per_level": int(n_frames),
        "gray_range": [float(gray.min()), float(gray.max())],
        "fit": fit,
        "fold": fold,
        "landmarks": landmarks,
        "fit_free": fit_free,
        "fit_ci95_bootstrap": fit_ci,
        "fold_ci95_bootstrap": fold_ci,
        "fit_boot_n": int(fit_boot_arr.size),
        "fold_boot_n": int(fold_boot_arr.size),
        "noise_sem_median": {"e0": float(np.median(sem0)), "e1": float(np.median(sem1))},
        "residual_rms": {
            "e0": float(np.sqrt(np.mean(resid0**2))),
            "e1": float(np.sqrt(np.mean(resid1**2))),
            "max_abs_e0": float(np.max(np.abs(resid0))),
            "max_abs_e1": float(np.max(np.abs(resid1))),
        },
        "duty_scan_g2pi": duty_scan,
        "eta1_peak": {
            "gray": float(gray[k]),
            "value": float(e1n[k]),
            "implied_phase_over_2pi": implied_peak_phase / (2.0 * math.pi),
        },
        "drift": drift,
        "curve": {
            "gray": gray.astype(int).tolist(),
            "e0": e0n.tolist(),
            "e1": e1n.tolist(),
            "model_e0": p0.tolist(),
            "model_e1": p1.tolist(),
        },
        "device_two_pi_gray_readback": _device_readback(run),
    }


def _drift(run: Path, rows: list[dict[str, Any]], orders: dict[str, Any]) -> dict[str, Any]:
    ref = [r for r in rows if r.get("group") == "reference"]
    drift_rows = [r for r in rows if r.get("group") == "drift"]
    if not ref or not drift_rows:
        return {"available": False}
    bg = float(orders["background"])
    cy, cx = orders["zero"]["row"], orders["zero"]["col"]
    n_pix = float((2 * APERTURE_R + 1) ** 2)

    def power(row: dict[str, Any]) -> float:
        frame = load_frame(run, row)
        return _box(frame, cy, cx, APERTURE_R) - bg * n_pix

    a = power(ref[0])
    b = float(np.mean([power(r) for r in drift_rows]))
    return {
        "available": True,
        "reference_power": a,
        "post_sweep_power": b,
        "relative_change": (b - a) / a if a > 0 else float("nan"),
    }


def _device_readback(run: Path) -> Any:
    path = run / "summary.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle).get("device_two_pi_gray_readback")


def _markdown(result: dict[str, Any]) -> str:
    fit = result["fit"]
    land = result["landmarks"]
    fit_ci = result["fit_ci95_bootstrap"]
    fold_ci = result["fold_ci95_bootstrap"]
    avail = bool(land.get("available"))
    fold_avail = bool(result["fold"].get("available"))

    if fold_avail:
        fold_cell = (
            f"**{result['fold']['g2pi']:.1f}** (frame bootstrap 95% CI "
            f"[{fold_ci[0]:.0f}, {fold_ci[1]:.0f}], n={result['fold_boot_n']})"
        )
    else:
        fold_cell = "unavailable"

    if avail:
        peak_cell = f"{land['peak_pi_gray']:.1f}"
        gam_cell = (
            f"{land['conditional_gamma']:.3f} (systematic range "
            f"[{land['conditional_gamma_systematic_range'][0]:.3f}, "
            f"{land['conditional_gamma_systematic_range'][1]:.3f}])"
        )
        plus_cell = f"{land['individual_order_minima']['plus1']['vertex_gray']:.1f}"
        minus_cell = f"{land['individual_order_minima']['minus1']['vertex_gray']:.1f}"
        env_cell = (
            f"[{land['systematic_envelope'][0]:.0f}, "
            f"{land['systematic_envelope'][1]:.0f}]"
        )
    else:
        peak_cell = gam_cell = plus_cell = minus_cell = env_cell = "n/a"

    lines = [
        "# Issue #238 - binary phase-grating fit",
        "",
        f"- run: `{result['run']}`",
        f"- levels: {result['n_gray_levels']} (gray {result['gray_range'][0]:.0f}"
        f"..{result['gray_range'][1]:.0f}), frames/level: {result['frames_per_level']}",
        f"- 0 order @ (row {result['orders']['zero']['row']}, "
        f"col {result['orders']['zero']['col']}); "
        f"+-1 separation {result['orders']['separation_px']} px",
        "",
        "> **Primary estimate**: the gamma-independent fold / landmark estimators.",
        "> The ideal full-curve e-domain fit is misspecified (see residual rms) and is",
        "> reported as diagnostic only, not as the calibration value.",
        "",
        "| quantity | value |",
        "| --- | --- |",
        f"| gamma-independent fold g2pi (primary) | {fold_cell} |",
        f"| landmark peak (phi=pi) gray | {peak_cell} |",
        f"| conditional gamma | {gam_cell} |",
        f"| +1 order vertex gray | {plus_cell} |",
        f"| -1 order vertex gray | {minus_cell} |",
        f"| +/-1 systematic envelope | {env_cell} |",
        f"| ideal e-domain fit (diagnostic, NOT adopted) | {fit['g2pi']:.1f} "
        f"(rms e1 {result['residual_rms']['e1']:.3f}) |",
        f"| ideal-fit bootstrap 95% CI | [{fit_ci[0]:.0f}, {fit_ci[1]:.0f}] |",
        f"| fitted gamma (ideal model) | {fit['gamma']:.3f} +- {fit['gamma_sd']:.3f} |",
        f"| residual rms (e0 / e1) | {result['residual_rms']['e0']:.4f} / "
        f"{result['residual_rms']['e1']:.4f} |",
        f"| free-duty fit | g2pi {result['fit_free']['g2pi']:.1f}, "
        f"gamma {result['fit_free']['gamma']:.3f}, "
        f"duty {result['fit_free']['duty']:.3f} "
        f"(rms {result['fit_free']['rms']:.4f}) |",
        f"| eta1 peak at gray | {result['eta1_peak']['gray']:.0f} "
        f"(phase/2pi = {result['eta1_peak']['implied_phase_over_2pi']:.3f}) |",
        f"| device readback 2*pi gray | {result['device_two_pi_gray_readback']} |",
    ]
    if result["drift"]["available"]:
        lines.append(
            f"| beam drift over sweep | "
            f"{100.0 * result['drift']['relative_change']:+.2f}% |"
        )
    lines += [
        "",
        "Duty-cycle sensitivity (assumed duty -> fitted 2*pi gray):",
        "",
        "| duty | 2*pi gray |",
        "| --- | --- |",
    ]
    for duty, value in result["duty_scan_g2pi"].items():
        lines.append(f"| {duty} | {value:.1f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline binary phase-grating fit for an issue #238 run directory "
            "(no devices touched)."
        )
    )
    parser.add_argument("--run", type=Path, required=True, help="run directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="fit.json path (default: <run>/fit.json; refuses to overwrite)",
    )
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument(
        "--json-only", action="store_true", help="print JSON instead of markdown"
    )
    args = parser.parse_args(argv)

    if not args.run.is_dir():
        print(f"ERROR: not a directory: {args.run}", file=sys.stderr)
        return 2
    out = args.out or (args.run / "fit.json")
    if out.exists():
        print(f"ERROR: refusing to overwrite {out}", file=sys.stderr)
        return 2

    result = analyse(args.run, bootstrap=max(int(args.bootstrap), 0), seed=args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    summary_path = out.with_suffix(".md")
    summary_path.write_text(_markdown(result), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
          if args.json_only else _markdown(result))
    print(f"wrote {out} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
