"""Shared scan/sweep analysis helpers for SLM Zernike scans (merged).

Merged from:
- :mod:`ao_shaping.tools.slm.slm_scan_analysis` (223 lines, 7 public symbols)
- :mod:`ao_shaping.tools.slm.delta_explorer` (363 lines, 12 public symbols)
- :mod:`ao_shaping.tools.slm.slm_snr_probe` (427 lines, 12 public symbols)

Phase-1 extraction of the scan-analysis helpers shared by the Zernike report
scripts and the shift-calibration tool, so the logic can be reused and
unit-tested without hardware, matplotlib or any driver import.

**Pure analysis + math, no device construction and no CLI.** Every entry point
takes already-opened device *instances* (anything with the ``BaseCamera`` /
``Santec`` duck-type) or plain trajectories, so the same code serves the Daheng
bench, a MiiCam, a simulation camera, or a fake in a unit test. Nothing here
imports a driver or reads ``.env``.

Public symbols (from slm_scan_analysis)
---------------------------------------
- ``LINEARITY_AMPS``    — amplitude grid used by the linearity report
- ``outlier_mask``      — median-based outlier rejection mask
- ``clamp_shift``       — round + clip a shift to ±limit
- ``parabolic_min``     — 3-point parabolic interpolation of a minimum
- ``latest_match``      — newest file matching a glob pattern
- ``group_raw_scan``    — nest raw scan records into (dll_index, radius) → amp → sign → array
- ``analyze_linearity`` — per-(mode, radius) proportionality verdicts

Public symbols (from delta_explorer)
-----------------------------------
- ``DeltaScanResult`` / ``analyze_delta_scan`` — rank candidate SPGD ``delta``
  values by ``frac_decreasing`` and ``late_gain``
- ``explore_delta`` / ``pick_best_delta``      — driver over the above

Public symbols (from slm_snr_probe)
-----------------------------------
- ``snr_verdict``        — threshold → strong / usable / unusable
- ``measure_noise_floor``— σ of a score at a fixed state
- ``abba_signal``        — drift-cancelled ΔJ for one perturbation direction
- ``snr_sweep``          — noise floor + per-delta SNR, single- and multi-mode
- ``SnrSweepResult``     — dataclass returned by :func:`snr_sweep`

Typical use (SNR half; the scan/delta halves take arrays)::

    with create_camera("daheng", cam_id=0, exposure_time_ms=1.2) as cam, \\
         Santec(slm_number=1, wavelength=1064) as slm:
        result = snr_sweep(cam, slm, n_max=9, radius=480.0)
        print(result.verdicts)
"""
from __future__ import annotations

import glob
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ao_shaping.utils.image.beam_metrics import zero_order_center
from ao_shaping.utils.image.targets import roi_pib_metric, target_shape_roi
from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale
from ao_shaping.utils.wavefront.zernike_calc import zernike_modes
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

# === SECTION: slm_scan_analysis (from slm_scan_analysis.py) ===
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


# === SECTION: delta_explorer (from delta_explorer.py) ===
FRAC_DECIDING_BAND = 0.05

#: Minimum number of scored samples before a verdict is meaningful. A run whose
#: accepted updates are this rare tells us nothing about convergence.
MIN_SAMPLES = 10

#: Below this the ``late_gain`` is indistinguishable from run-to-run scatter.
#: Calibrated on the bench where converged candidates scored >10 % and coin
#: flips landed within +-10 %.
LATE_GAIN_FLOOR = 10.0

#: Sentinel J used by the energy guard for an abandoned evaluation. Rows at or
#: beyond this magnitude are "rejected", not scores.
GUARD_SENTINEL = 100.0


@dataclass
class TraceStats:
    """Convergence statistics for one candidate ``delta``."""

    delta: float
    n_samples: int
    #: Fraction of steps that improve (0.5 == random walk).
    frac_decreasing: float
    #: Improvement (%) between first-third and last-third means.
    late_gain_pct: float
    #: First and last sample (reported for context only, NOT a verdict).
    first: float
    final: float
    best: float
    #: Objective range covered by the trace (a proxy for how much it moved).
    span_pct: float
    #: Fraction of rows rejected by the energy guard.
    guard_fraction: float = 0.0
    #: Rows whose update was actually adopted, when the recorder carries gates.
    accepted: int | None = None
    total_rows: int | None = None

    @property
    def converged(self) -> bool:
        """True only when the trace descends consistently *and* by a margin."""
        return (
            self.n_samples >= MIN_SAMPLES
            and self.frac_decreasing > 0.5 + FRAC_DECIDING_BAND
            and self.late_gain_pct >= LATE_GAIN_FLOOR
        )

    @property
    def random_walk(self) -> bool:
        return abs(self.frac_decreasing - 0.5) <= FRAC_DECIDING_BAND

    def verdict(self) -> str:
        if self.n_samples < MIN_SAMPLES:
            return "too-few-samples"
        if self.converged:
            return "converged"
        if self.random_walk:
            return "random-walk"
        if self.late_gain_pct >= LATE_GAIN_FLOOR:
            return "noisy-but-rising"
        return "not-converging"


@dataclass
class DeltaScanResult:
    """Outcome of :func:`explore_delta`."""

    stats: list[TraceStats] = field(default_factory=list)

    @property
    def converged(self) -> list[TraceStats]:
        return sorted(
            (s for s in self.stats if s.converged),
            key=lambda s: -s.late_gain_pct,
        )

    @property
    def recommended(self) -> float | None:
        """Best converging ``delta``, or ``None`` if nothing converged.

        Returning ``None`` is the point: it refuses to crown a random walk.
        """
        best = self.converged
        return best[0].delta if best else None

    def table(self) -> str:
        """Markdown table of every candidate."""
        head = (
            "| delta | verdict | dec | late % | span % | guard % | accepted |"
            " first | final |"
        )
        sep = "|---|---|---|---|---|---|---|---|---|"
        lines = [head, sep]
        for s in sorted(self.stats, key=lambda s: s.delta):
            acc = "-" if s.accepted is None else f"{s.accepted}/{s.total_rows}"
            lines.append(
                f"| {s.delta:g} | {s.verdict()} | {s.frac_decreasing:.2f} | "
                f"{s.late_gain_pct:+.1f} | {s.span_pct:.1f} | "
                f"{100 * s.guard_fraction:.1f} | {acc} | "
                f"{s.first:.4f} | {s.final:.4f} |"
            )
        return "\n".join(lines)


def verdict_for(stats: TraceStats) -> str:
    """Standalone verdict for one :class:`TraceStats`."""
    return stats.verdict()


def _as_rows(result: Any) -> list[dict[str, Any]]:
    """Normalise a recorder / dict / list-of-dicts into a list of row dicts."""
    if result is None:
        return []
    if hasattr(result, "history"):  # ao_shaping Recorder
        return [dict(r) for r in result.history]
    if isinstance(result, dict):
        # {epoch: row} as written by save_recorder_debug_artifacts
        return [dict(v) for v in result.values() if isinstance(v, dict)]
    if isinstance(result, (list, tuple)):
        return [dict(r) for r in result]
    raise TypeError(
        "run_one must return a Recorder, a {epoch: row} dict or a list of dicts; "
        f"got {type(result).__name__}"
    )


def _pick_objective_key(
    rows: Sequence[dict[str, Any]], requested: str | None
) -> str | None:
    if requested is not None:
        return requested if requested in rows[0] else None
    for candidate in ("pearson", "shape", "roi_pib", "rms_pib", "pib", "rmse"):
        if candidate in rows[0]:
            return candidate
    return "J" if "J" in rows[0] else None


def analyse_trace(
    values: Sequence[float],
    *,
    delta: float = float("nan"),
    lower_is_better: bool | None = None,
    guard_rows: int = 0,
    total_rows: int | None = None,
    accepted: int | None = None,
) -> TraceStats:
    """Convergence statistics for one objective series.

    Args:
        values: Objective per epoch, in run order.
        delta: The perturbation amplitude this trace was produced with.
        lower_is_better: Polarity. Inferred from ``delta``'s companion when
            ``None`` via :func:`_infer_polarity` if the key name is known, else
            assumed higher-is-better.
        guard_rows: Rows discarded because the energy guard rejected them.
        total_rows: Rows recorded before guard filtering.
        accepted: Rows whose update was adopted (from the gate column).

    Returns:
        A populated :class:`TraceStats`.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        return TraceStats(
            delta=delta, n_samples=0, frac_decreasing=0.0, late_gain_pct=0.0,
            first=float("nan"), final=float("nan"), best=float("nan"),
            span_pct=0.0, guard_fraction=1.0 if total_rows else 0.0,
            accepted=accepted, total_rows=total_rows,
        )

    if lower_is_better is None:
        lower_is_better = False
    # Orient so that IMPROVING IS ALWAYS AN INCREASE, regardless of polarity.
    # For a loss, improvement is a decrease, so negate it; for a score the raw
    # values already increase on improvement. Getting this backwards silently
    # reports frac_decreasing ~0.15 for a clean descent.
    sign = -1.0 if lower_is_better else 1.0
    work = sign * arr

    steps = np.diff(work)
    frac_improving = float(np.mean(steps > 0)) if steps.size else 0.0

    third = max(arr.size // 3, 1)
    head = float(np.mean(work[:third]))
    tail = float(np.mean(work[-third:]))
    denom = abs(head) if abs(head) > 1e-12 else 1.0
    late = 100.0 * (tail - head) / denom

    # ``best`` is reported in the objective's own units, not the oriented ones.
    best = float(arr.min() if lower_is_better else arr.max())
    span = 100.0 * float(np.max(arr) - np.min(arr)) / denom

    return TraceStats(
        delta=delta,
        n_samples=int(arr.size),
        frac_decreasing=frac_improving,
        late_gain_pct=late,
        first=float(arr[0]),
        final=float(arr[-1]),
        best=best,
        span_pct=span,
        guard_fraction=(guard_rows / total_rows) if total_rows else 0.0,
        accepted=accepted,
        total_rows=total_rows,
    )


def _infer_polarity(key: str | None) -> bool:
    """Pearson is a loss; every other objective in this project is a score."""
    return (key or "").lower() == "pearson"


def explore_delta(
    run_one: Callable[[float], Any],
    deltas: Sequence[float],
    *,
    objective_key: str | None = None,
    lower_is_better: bool | None = None,
    progress: Callable[[float, int, int], None] | None = None,
) -> DeltaScanResult:
    """Run one optimisation per candidate ``delta`` and rank by convergence.

    Args:
        run_one: ``delta -> recorder``. Performs one optimisation and returns
            its recorder (or a ``{epoch: row}`` dict, or a list of rows).
            Raising is treated as a failed candidate and recorded as such.
        deltas: Candidate perturbation amplitudes to try.
        objective_key: Objective column to score (default: auto-detect,
            preferring ``pearson``/``shape``/... then ``J``).
        lower_is_better: Override the polarity inference.
        progress: Optional ``(delta, index, total)`` callback.

    Returns:
        A :class:`DeltaScanResult`. Check :attr:`DeltaScanResult.recommended` —
        it is ``None`` when no candidate actually converged.
    """
    deltas = [float(d) for d in deltas]
    if not deltas:
        raise ValueError("deltas must not be empty")

    out: list[TraceStats] = []
    for i, delta in enumerate(deltas, start=1):
        if progress is not None:
            progress(delta, i, len(deltas))
        try:
            raw = run_one(delta)
        except Exception:  # a failed candidate must not abort the scan
            out.append(
                TraceStats(
                    delta=delta, n_samples=0, frac_decreasing=0.0,
                    late_gain_pct=0.0, first=float("nan"), final=float("nan"),
                    best=float("nan"), span_pct=0.0, guard_fraction=1.0,
                )
            )
            continue

        rows = _as_rows(raw)
        if not rows:
            out.append(
                TraceStats(
                    delta=delta, n_samples=0, frac_decreasing=0.0,
                    late_gain_pct=0.0, first=float("nan"), final=float("nan"),
                    best=float("nan"), span_pct=0.0, guard_fraction=1.0,
                )
            )
            continue

        key = _pick_objective_key(rows, objective_key)
        if key is None:
            continue

        guarded = [r for r in rows if abs(float(r.get("J", 0.0))) > GUARD_SENTINEL]
        scored = [r for r in rows if abs(float(r.get("J", 0.0))) <= GUARD_SENTINEL]
        accepted = None
        if any("_gate" in r for r in rows):
            accepted = sum(1 for r in rows if str(r.get("_gate")) == "applied")

        vals = [float(r[key]) for r in scored if key in r]
        if not vals:
            continue

        out.append(
            analyse_trace(
                vals,
                delta=delta,
                lower_is_better=(
                    _infer_polarity(key)
                    if lower_is_better is None
                    else lower_is_better
                ),
                guard_rows=len(guarded),
                total_rows=len(rows),
                accepted=accepted,
            )
        )

    return DeltaScanResult(stats=out)


# === SECTION: slm_snr_probe (from slm_snr_probe.py) ===
SNR_STRONG = 3.0
SNR_USABLE = 2.0

#: σ at or below this is treated as *no measurable noise* (a perfectly static
#: bench, or a synthetic camera). Without the floor, ``signal / sigma`` would
#: divide by ~1e-16 and report an absurd SNR from float round-off. Such a
#: measurement carries no information about a real bench.
SIGMA_FLOOR = 1e-12

#: Default perturbation amplitudes (rad) probed when the caller does not pass any.
DEFAULT_DELTAS: tuple[float, ...] = (0.0005, 0.001, 0.01, 0.05, 0.1, 0.2)

#: Reference Zernike mode for the single-mode probe: defocus (2, 0).
REFERENCE_MODE: tuple[int, int] = (2, 0)


def snr_verdict(snr: float) -> str:
    """Classify an SNR into ``strong`` / ``usable`` / ``unusable``."""
    if not np.isfinite(snr):
        return "unusable"
    if snr >= SNR_STRONG:
        return "strong"
    if snr >= SNR_USABLE:
        return "usable"
    return "unusable"


def measure_noise_floor(
    score: Callable[[], float],
    n_frames: int = 15,
) -> tuple[float, list[float]]:
    """Standard deviation of ``score`` at a *fixed* device state.

    The caller is responsible for leaving the devices untouched between calls
    (e.g. a flat phase already written). This measures the combined shot noise
    **and** slow drift, which is what a single SPGD step actually sees.

    Args:
        score: Zero-arg callable returning one scalar objective value.
        n_frames: Number of consecutive evaluations.

    Returns:
        ``(sigma, values)`` — the sample standard deviation and the raw series.
    """
    if n_frames < 2:
        raise ValueError(f"n_frames must be >= 2, got {n_frames}")
    values = [float(score()) for _ in range(n_frames)]
    return float(np.std(values)), values


def abba_signal(
    score: Callable[[], float],
    write_phase: Callable[[float], None],
    write_reference: Callable[[], None],
    amplitude: float,
    pairs: int = 3,
) -> float:
    """Drift-cancelled ``ΔJ`` for one perturbation amplitude.

    Each repetition uses the palindrome ``+ - - +``. That ordering is what makes
    the drift cancellation work: the two ``+`` samples sit at the outer time
    positions and the two ``-`` samples at the inner ones, so for *any* drift
    that is locally linear in time the two means carry an identical drift term
    and it drops out of ``|mean(+) - mean(-)|``.

    The superficially similar ``+ - + -`` ordering does **not** cancel: with a
    drift of ``c·t`` it leaves a residual of ``c`` per pair, which on this bench
    is the same order as the signal being measured. Frame averaging cannot fix
    this either, because the dominant residual is slow intensity drift rather
    than per-frame shot noise.

    Args:
        score: Zero-arg callable returning one scalar objective value.
        write_phase: ``amplitude -> None``; writes that coefficient value.
        write_reference: ``-> None``; restores the flat/reference state.
        amplitude: Perturbation magnitude (rad).
        pairs: Number of repetitions to average.

    Returns:
        Mean ``|mean(+) - mean(-)|`` over the repetitions.
    """
    if pairs < 1:
        raise ValueError(f"pairs must be >= 1, got {pairs}")
    # Palindrome, not ABAB: see the docstring.
    signs = (1.0, -1.0, -1.0, 1.0)
    diffs: list[float] = []
    for _ in range(pairs):
        plus: list[float] = []
        minus: list[float] = []
        for sign in signs:
            write_phase(sign * amplitude)
            (plus if sign > 0 else minus).append(float(score()))
            write_reference()
        diffs.append(abs(float(np.mean(plus)) - float(np.mean(minus))))
    return float(np.mean(diffs))


@dataclass
class SnrSweepResult:
    """Outcome of :func:`snr_sweep`."""

    #: 0-order centre (x, y) the ROI was anchored on, window-local.
    center: tuple[float, float]
    #: σ of the objective at a fixed state.
    sigma: float
    #: Raw noise-floor series.
    noise_values: list[float] = field(default_factory=list)
    #: ``str(amplitude) -> ΔJ`` for the single-mode probe.
    single_signals: dict[str, float] = field(default_factory=dict)
    #: ``str(amplitude) -> ΔJ`` for the multi-mode (SPGD-style) probe.
    multi_signals: dict[str, float] = field(default_factory=dict)
    #: Number of perturbed DOF in the multi-mode probe.
    n_dof: int = 0
    seed: int | None = None
    #: Fraction of the frame's light inside the scored ROI. A very small value
    #: means the ROI geometry does not match the bench (typically the camera was
    #: never windowed around the spot) and the SNRs above are not meaningful.
    roi_fraction: float = 0.0

    @property
    def _denom(self) -> float:
        """σ with a floor, so a perfectly static bench cannot yield inf SNR."""
        return self.sigma if self.sigma > SIGMA_FLOOR else 0.0

    @property
    def single_snrs(self) -> dict[str, float]:
        return {
            k: (v / self._denom if self._denom else 0.0)
            for k, v in self.single_signals.items()
        }

    @property
    def multi_snrs(self) -> dict[str, float]:
        return {
            k: (v / self._denom if self._denom else 0.0)
            for k, v in self.multi_signals.items()
        }

    @property
    def verdicts(self) -> dict[str, str]:
        """Multi-mode (SPGD-relevant) verdict per amplitude."""
        return {k: snr_verdict(v) for k, v in self.multi_snrs.items()}

    def usable_deltas(self, minimum: float = SNR_USABLE) -> list[float]:
        """Amplitudes whose *multi-mode* SNR clears ``minimum``.

        This is the list to feed to ``--delta``: the single-mode column is a
        trap, since it over-reports what SPGD can actually resolve.
        """
        return sorted(
            float(k) for k, v in self.multi_snrs.items() if v >= minimum
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary (used by the report generator)."""
        return {
            "center": [float(self.center[0]), float(self.center[1])],
            "sigma": self.sigma,
            "noise_mean": float(np.mean(self.noise_values)) if self.noise_values else 0.0,
            "n_dof": self.n_dof,
            "seed": self.seed,
            "roi_fraction": self.roi_fraction,
            "single_signals": self.single_signals,
            "multi_signals": self.multi_signals,
            "single_snrs": self.single_snrs,
            "multi_snrs": self.multi_snrs,
            "verdicts": self.verdicts,
            "thresholds": {"strong": SNR_STRONG, "usable": SNR_USABLE},
            "sigma_floor": SIGMA_FLOOR,
        }


def _phase_to_gray(phase: np.ndarray, slm: Any) -> np.ndarray:
    """Radian phase -> uint16 grayscale, tolerating a minimal SLM stub.

    Prefers the driver's own pipeline (``create_phase_from_array``: mod 2π +
    wavefront correction + LUT + shift, with the device's wavelength-dependent
    2π gray) so the measurement sees exactly what the optimizer would send.
    An instance that only implements ``display_data`` (a test double, or a
    reduced driver) falls back to the pure-math conversion rather than raising —
    that is what keeps this module usable without hardware.
    """
    if hasattr(slm, "create_phase_from_array"):
        return phase_to_slm_grayscale(phase, slm=slm)
    return phase_to_slm_grayscale(phase, slm=None)


def _write_zernike(
    slm: Any,
    amps: Mapping[Any, float],
    n_max: int,
    resolution: tuple[int, int],
    radius: float | None,
) -> None:
    """Write a Zernike phase (raw radians -> grayscale) through the driver."""
    # generate_zernike_phase takes a concrete dict of {(n, m) | noll | name: amp};
    # normalise at the boundary so callers may pass any mapping.
    coefficients: dict[Any, float] = {k: float(v) for k, v in amps.items()}
    phase = generate_zernike_phase(
        coefficients, resolution=resolution, n_max=n_max, radius=radius
    )
    # generate_zernike_phase returns NaN outside its inscribed aperture; the
    # optimizer's aperture-limited phase is flat there.
    phase = np.nan_to_num(phase, nan=0.0)
    slm.display_data(_phase_to_gray(phase, slm))


def snr_sweep(
    cam: Any,
    slm: Any,
    *,
    n_max: int = 9,
    radius: float | None = 480.0,
    resolution: tuple[int, int] = (1920, 1200),
    target_shape: str = "square",
    target_size: float = 50.0,
    deltas: Sequence[float] = DEFAULT_DELTAS,
    n_frames: int = 15,
    pairs: int = 3,
    seed: int | None = 0,
    measure_single: bool = True,
    measure_multi: bool = True,
    score_fn: Callable[[np.ndarray], float] | None = None,
    window: tuple[Any, Any] | None = None,
) -> SnrSweepResult:
    """Measure the objective's noise floor and per-amplitude SNR on a live bench.

    Both device arguments are **already-open instances** — this function never
    constructs, opens or closes a device, and never reads a camera type, so it
    is equally valid for Daheng, MiiCam, a simulation camera or a test double.

    Two probes are run per amplitude:

    * **single-mode** — only :data:`REFERENCE_MODE` (defocus) is perturbed.
      This is what a naive sensitivity check measures.
    * **multi-mode** — every non-piston mode is perturbed by a random ±1
      pattern, which is what SPGD actually does.

    The gap between the two is the practical dilution factor; judge
    ``--delta`` by the multi-mode column.

    Args:
        cam: Open camera exposing ``get_numpy_image(n_sample=...)``.
        slm: Open SLM exposing ``display_data(gray)``.
        n_max: Max Zernike radial order (sets the DOF count).
        radius: Zernike aperture radius (px) on the SLM panel.
        resolution: SLM panel resolution as ``(width, height)``.
        target_shape: ROI shape for the built-in score.
        target_size: ROI size (px) for the built-in score.
        deltas: Perturbation amplitudes to probe (rad).
        n_frames: Frames for the noise floor.
        pairs: ABBA repetitions per amplitude.
        seed: RNG seed for the ±1 pattern (``None`` -> nondeterministic).
        measure_single: Run the single-mode probe.
        measure_multi: Run the multi-mode probe.
        score_fn: ``frame -> float`` overriding the built-in ``roi_pib`` score.

            **Pass the objective you actually care about.** The built-in score
            anchors a fixed ``target_size`` box on the flat-state 0-order, which
            is only meaningful once the caller has windowed the camera around the
            spot. ``create_camera(..., cam_size=N)`` does *not* window: the driver
            only calls ``reset_window`` when asked, so on a full 2592x1944 frame
            a 50 px box can hold <1 % of the light and the measured SNR collapses
            to ~1 for every amplitude. Bench-verified: that misconfiguration
            reported ``usable_deltas() == []`` while the optimizer at the same
            delta reached +18.8 % sustained improvement. When ``score_fn`` is
            given, ``target_shape``/``target_size`` are ignored.
        window: Optional ``(size, center)`` forwarded to ``cam.reset_window``
            before measuring, so the probe can window the bench itself. The
            caller still owns device configuration; this is a convenience.

    Returns:
        A populated :class:`SnrSweepResult`.
    """
    if n_max < 0:
        raise ValueError(f"n_max must be >= 0, got {n_max}")
    if deltas and all(d <= 0 for d in deltas):
        raise ValueError(f"deltas must contain a positive amplitude, got {deltas!r}")

    if window is not None and hasattr(cam, "reset_window"):
        cam.reset_window(*window)

    def write(amps: dict[tuple[int, int], float]) -> None:
        _write_zernike(slm, amps, n_max, resolution, radius)

    def write_flat() -> None:
        write({})

    # Piston (0,0) is a constant phase offset the detector cannot see; including
    # it would waste perturbation budget and dilute the multi-mode signal.
    modes = [(n, m) for n, m in zernike_modes(n_max) if (n, m) != (0, 0)]
    pattern = None
    if measure_multi and modes:
        rng = np.random.default_rng(seed)
        pattern = rng.choice([-1.0, 1.0], size=len(modes))

    # Anchor the ROI once on the flat state. Re-deriving it per frame is fine
    # (measured cost: 1.1x more noise) but a fixed anchor is what the optimizer
    # uses, so measure what the optimizer sees.
    write_flat()
    first = np.asarray(cam.get_numpy_image(1), dtype=np.float64)
    cx, cy = (float(v) for v in zero_order_center(first))
    center: tuple[float, float] = (cx, cy)

    def score() -> float:
        frame = np.asarray(cam.get_numpy_image(1), dtype=np.float64)
        if score_fn is not None:
            return float(score_fn(frame))
        return float(
            roi_pib_metric(
                frame, center, target_shape, target_size
            )[0]
        )

    sigma, noise_values = measure_noise_floor(score, n_frames=n_frames)

    single: dict[str, float] = {}
    multi: dict[str, float] = {}
    for delta in deltas:
        key = str(delta)
        if measure_single:
            single[key] = abba_signal(
                score,
                lambda a, _d=delta: write({REFERENCE_MODE: a}),
                write_flat,
                delta,
                pairs=pairs,
            )
        if measure_multi and pattern is not None:
            multi[key] = abba_signal(
                score,
                lambda a, _d=delta, _p=pattern: write(
                    {k: a * _d * float(v) for k, v in zip(modes, _p, strict=True)}
                ),
                write_flat,
                delta,
                pairs=pairs,
            )
    write_flat()  # leave the bench flat, not on a perturbation

    # Diagnostic: what fraction of the frame's light does the scored ROI actually
    # hold? A tiny value means the geometry is wrong (e.g. the camera was never
    # windowed around the spot) and every SNR below is meaningless, however
    # plausible the numbers look. This is the check that would have caught the
    # misconfiguration documented on ``score_fn``.
    roi_fraction = 0.0
    if score_fn is None:
        probe = np.asarray(cam.get_numpy_image(1), dtype=np.float64)
        total = float(probe.sum())
        if total > 0:
            box = target_shape_roi(
                probe.shape, center, target_shape, target_size
            )
            roi_fraction = float(probe[box].sum()) / total

    return SnrSweepResult(
        center=center,
        sigma=sigma,
        noise_values=noise_values,
        single_signals=single,
        multi_signals=multi,
        n_dof=len(modes),
        seed=seed,
        roi_fraction=roi_fraction,
    )
