"""Shared scan-analysis helpers for SLM Zernike scan reports (pure numpy + stdlib).

Phase-1 extraction of the scan-analysis helpers shared by the Zernike report
scripts and the shift-calibration tool, so the logic can be reused and
unit-tested without hardware, matplotlib or any driver import.

Public symbols
--------------
- ``LINEARITY_AMPS``    — amplitude grid used by the linearity report
- ``outlier_mask``      — median-based outlier rejection mask
- ``clamp_shift``       — round + clip a shift to ±limit
- ``parabolic_min``     — 3-point parabolic interpolation of a minimum
- ``latest_match``      — newest file matching a glob pattern
- ``group_raw_scan``    — nest raw scan records into (dll_index, radius) → amp → sign → array
- ``analyze_linearity`` — per-(mode, radius) proportionality verdicts

Sources (behaviour preserved byte-for-byte)
-------------------------------------------
- ``scripts/generate_zernike_linearity_report.py``  (AMPS, _latest, load_groups, analyze)
- ``scripts/generate_zernike_response_matrix_report.py``  (_latest, _raw_groups)
- ``src/ao_shaping/tools/slm/calibration.py``  (_clamp, parabolic_min)
"""
from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

#: Amplitude grid (rad) used by the Zernike linearity report
#: (``scripts/generate_zernike_linearity_report.py``, ``AMPS``).
LINEARITY_AMPS: tuple[float, float, float] = (2.0, 5.0, 10.0)


def outlier_mask(arr: np.ndarray, factor: float) -> np.ndarray:
    """Boolean mask keeping values ``<= factor * median(arr)``.

    Reproduces the ``med > 0`` guard used at the outlier-rejection call sites
    (``slm_zernike_common.diagnose_beam_radius`` and
    ``slm_zernike_correction``): when the median is ``<= 0`` the source skips
    the filter entirely, i.e. keeps every point → an all-True mask.

    Callers pass non-negative magnitude arrays (no ``abs()`` inside).

    Args:
        arr: Non-negative magnitude array to filter.
        factor: Outlier threshold as a multiple of the median.

    Returns:
        Boolean mask of the same shape as ``arr``.
    """
    med = float(np.median(arr))
    if med <= 0:
        return np.ones(arr.shape, dtype=bool)
    return arr <= factor * med


def clamp_shift(value: float, limit: int) -> int:
    """Round ``value`` and clip to ``[-limit, limit]``.

    Verbatim from ``_clamp`` in ``calibration.py`` (keeps the defocus disc
    inside the SLM panel).

    Args:
        value: Raw shift to clamp.
        limit: Absolute shift limit.

    Returns:
        The rounded, clipped integer shift.
    """
    return int(np.clip(round(value), -limit, limit))


def parabolic_min(pts: list[tuple[float, float]]) -> float | None:
    """三点抛物线插值细化最小值位置 (``pts`` 需按 x 升序).

    Verbatim from ``calibration.py``. Guards: fewer than 3 points → None;
    minimum at an edge → that edge's x; degenerate denominator
    (``|denom| < 1e-12``) or near-zero curvature (``|a| < 1e-12``) → the middle
    point's x; a fitted vertex outside the sampled x-range → the middle point's x.

    Args:
        pts: ``(x, y)`` samples sorted by x ascending.

    Returns:
        The interpolated minimum x, or ``None`` with fewer than 3 points.
    """
    if len(pts) < 3:
        return None
    i = int(np.argmin([p[1] for p in pts]))
    if i == 0 or i == len(pts) - 1:
        return pts[i][0]
    (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < 1e-12:
        return x1
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    if abs(a) < 1e-12:
        return x1
    xv = -b / (2 * a)
    return float(xv) if min(x0, x2) <= xv <= max(x0, x2) else x1


def latest_match(pattern: str) -> Path | None:
    """Newest file matching a glob ``pattern``, or ``None`` if no match.

    Verbatim from ``_latest`` in both report scripts — the two sources are
    identical (``generate_zernike_linearity_report.py`` and
    ``generate_zernike_response_matrix_report.py``), so no divergence to note.

    Args:
        pattern: Glob pattern (e.g. ``"data/zernike_correction/raw_scan_*.json"``).

    Returns:
        The lexicographically-last matching path, or ``None``.
    """
    hits = sorted(glob.glob(pattern))
    return Path(hits[-1]) if hits else None


def group_raw_scan(raw: list[dict], to_waves: bool) -> dict:
    """Nest raw scan records into ``(dll_index, radius) → amp → sign → np.ndarray``.

    Merge of the two source loaders:
    - ``load_groups`` (``generate_zernike_linearity_report.py``): divides the
      ``readout_um`` by 0.532 (µm → waves) — reproduced with ``to_waves=True``.
    - ``_raw_groups`` (``generate_zernike_response_matrix_report.py``): keeps
      the raw µm values — reproduced with ``to_waves=False``.

    Args:
        raw: List of raw scan records, each with ``dll_index`` (int),
            ``radius``, ``amp_rad``, ``sign`` (int) and ``readout_um`` (list).
        to_waves: If True, divide ``readout_um`` by 0.532 (µm → λ).

    Returns:
        Nested dict ``{(dll_index: int, radius: float): {amp_rad: float:
        {sign: int: np.ndarray[float64]}}}``.
    """
    g: dict[tuple[int, float], dict[float, dict[int, np.ndarray]]] = defaultdict(
        lambda: defaultdict(dict))
    for s in raw:
        val = np.asarray(s["readout_um"], dtype=float)
        if to_waves:
            val = val / 0.532     # → λ
        g[(s["dll_index"], float(s["radius"]))][float(s["amp_rad"])][int(s["sign"])] = val
    return g


def analyze_linearity(
    groups: dict, amps: tuple[float, ...] = LINEARITY_AMPS
) -> list[dict]:
    """Per-(mode, radius) proportionality verdicts for a Zernike scan.

    Verbatim from ``analyze`` in ``generate_zernike_linearity_report.py``
    (the module-level ``AMPS`` is parameterised as ``amps``). Consumes the
    ``groups`` structure produced by ``group_raw_scan(..., to_waves=True)``.

    For every (mode, radius) with all ``amps`` present as ± sign pairs:
    ``diag = (z₊[m] − z₋[m])/2`` is checked for proportionality to the
    amplitude A via an origin-fit R² and the CV of ``diag/A``; the residual
    baseline ``|z₊ + z₋|/2`` sets the SNR floor.

    Verdicts (exact strings, ``**`` markers kept verbatim):
    - ``"成比例"``            — R² > 0.98, CV < 0.15, SNR ≥ 1.5
    - ``"成比例 (弱耦合)"``    — R² > 0.98, CV < 0.15, SNR < 1.5
    - ``"噪声受限"``          — not proportional and SNR < 1.5
    - ``"**不成比例**"``       — not proportional and SNR ≥ 1.5

    A (mode, radius) is skipped when any amplitude is missing a ± sign pair
    (``ok=False``) or fewer than 3 amplitudes were collected.

    Args:
        groups: ``{(dll_index, radius): {amp: {sign: np.ndarray}}}`` in waves.
        amps: Amplitude grid to evaluate (default ``LINEARITY_AMPS``).

    Returns:
        List of row dicts with keys ``m, R, diag, vec, base, k, r2, cv,
        ratio_10_2, snr, verdict``.
    """
    rows: list[dict] = []
    for (m, R) in sorted(groups):
        d = groups[(m, R)]
        diag, vec, base = [], [], []
        ok = True
        for a in amps:
            zp, zn = d.get(a, {}).get(1), d.get(-a, {}).get(-1)
            if zp is None or zn is None:
                ok = False
                break
            diag.append(float((zp[m] - zn[m]) / 2.0))
            vec.append(float(np.linalg.norm(zp[1:] - zn[1:]) / 2.0))
            base.append(float(np.linalg.norm(zp[1:] + zn[1:]) / 2.0))
        if not ok or len(diag) < 3:
            continue
        diag_a, vec_a, base_a = map(np.array, (diag, vec, base))
        A = np.array(amps)
        # 过原点线性拟合 diag = k·A
        k = float(np.sum(A * diag_a) / np.sum(A * A))
        pred = k * A
        ss_res = float(np.sum((diag_a - pred) ** 2))
        ss_tot = float(np.sum((diag_a - diag_a.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ratios = diag_a / A
        cv = (float(np.std(ratios) / abs(np.mean(ratios)))
              if np.mean(ratios) != 0 else float("inf"))
        ratio_10_2 = (float(diag_a[2] / diag_a[0]) if diag_a[0] != 0 else np.nan)
        floor = float(np.median(base_a))
        snr = float(abs(diag_a[2]) / floor) if floor > 0 else float("inf")
        # 判定: 主判据用**统计上稳健**的 R² + CV (三点拟合已含点间散布);
        # `A10/A2` 在 A=2 时受基线噪声影响极大 (误差 ±0.1λ → 比值 ±2.4 不确定),
        # 故仅作展示, 不作硬判据。SNR 用于标注"弱耦合"而非判失败。
        if r2 > 0.98 and cv < 0.15:
            verdict = "成比例" if snr >= 1.5 else "成比例 (弱耦合)"
        elif snr < 1.5:
            verdict = "噪声受限"
        else:
            verdict = "**不成比例**"
        rows.append({"m": m, "R": R, "diag": diag_a.tolist(), "vec": vec_a.tolist(),
                     "base": floor, "k": k, "r2": float(r2), "cv": cv,
                     "ratio_10_2": ratio_10_2, "snr": snr, "verdict": verdict})
    return rows