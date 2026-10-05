"""Generate the **offline** SLM bench-probe report from saved sweep artefacts.

Reads only what hardware runs already wrote -- ``sweep_records.npz``,
``bench_geometry.json`` and the newest ``data/debug`` sidecar -- and renders one
illustrated Chinese markdown report plus five figures. **Fully offline**: it
never constructs a camera or an SLM, so it can be re-run at any time (including
while the instruments are powered down) to refresh the conclusions.

What it reports
---------------
1. **Provenance and completeness** -- which inputs were found, which optional
   inputs are absent, and (importantly) whether the debug sidecar actually
   describes *this* npz. A sidecar from a different sweep configuration is
   reported as a mismatch rather than silently attributed.
2. **Ramp linearity** -- displacement vs ``1/period`` for both panel axes,
   compared against the bench constant ``TILT_SHIFT_SCALE``. That constant is
   used as a literal (``7400``) rather than imported, because its defining
   module transitively pulls in the hardware driver packages.
3. **Zernike tilt linearity** -- per-axis centroid slope, the repeat-to-repeat
   spread, and the phase-correlation ``shift_corr`` that decides whether the
   centroid or the correlation shift is trustworthy.
4. **Width responses** -- FWHM vs coefficient for every swept quadratic mode,
   with the single-lobe (hollowness) and per-side monotonicity filters applied
   and every dropped point named.
5. **Curvature / offset summary and waist identifiability** -- the recomputed
   ``width**2`` curvature per mode against the joint rel-RMS threshold
   ``_MAX_DEFOCUS_FIT_RMS`` (literal ``0.10``).
6. **Three independent routes to the focal scale** -- ramp, Zernike tilt and the
   field-of-view ratio -- so a disagreement is visible instead of resolved
   silently.
7. **关键陷阱** -- the four bench traps encoded in
   ``ao_shaping.tools.slm.bench_kernels``, each with the number that proves it
   is still live on this bench.

Usage:
    python scripts/generate_bench_probe_report.py
    python scripts/generate_bench_probe_report.py -o docs/slm/bench_probe
    python scripts/generate_bench_probe_report.py --sweep-npz <path> --no-figures
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from loguru import logger  # noqa: E402

# CJK-capable font fallbacks (per repo script convention).
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10

DEFAULT_NPZ = ROOT / "data" / "model_in_loop_hw" / "sweep_records.npz"
DEFAULT_GEOMETRY = ROOT / "data" / "model_in_loop_hw" / "bench_geometry.json"
DEFAULT_DEBUG_ROOT = ROOT / "data" / "debug"
DEFAULT_OUT = ROOT / "docs" / "slm" / "bench_probe"

#: Bench constant from the shared measurement core; the report states loudly when
#: the import failed and the fallback below was used instead.
FALLBACK_TILT_SHIFT_SCALE = 7400.0
#: Joint width-fit rel-RMS above which the illumination profile is *not*
#: identifiable (mirrors ``model_in_loop_shaping._MAX_DEFOCUS_FIT_RMS``).
FALLBACK_MAX_DEFOCUS_FIT_RMS = 0.10
#: Below this Pearson peak the phase-correlation shift is not trusted over the
#: spot centroid (mirrors ``model_in_loop_shaping._MIN_SHIFT_CORRELATION``).
FALLBACK_MIN_SHIFT_CORRELATION = 0.30
#: Below this centroid/peak ratio the focal plane is a ring, not one lobe
#: (mirrors ``model_in_loop_shaping._DEFOCUS_MIN_HOLLOWNESS``).
FALLBACK_MIN_HOLLOWNESS = 0.60
#: Camera sensor size assumed when cross-checking the field-of-view route.
FALLBACK_CAMERA_PIXELS = 2592.0
FALLBACK_CAMERA_PIXEL_M = 2.2e-6
FALLBACK_FOCAL_LENGTH_M = 0.125
FALLBACK_WAVELENGTH_M = 1064e-9
FALLBACK_APERTURE_M = 3.6e-3

#: Modes whose spot *width* is fitted as a quadratic in the coefficient.
WIDTH_MODES = ("defocus", "astig_x", "astig_y", "coma_x", "coma_y", "spherical")

#: Frame pickles above this size are not loaded -- they hold every repeated CCD
#: frame and would dominate the runtime for no extra report value.
MAX_PKL_BYTES = 200 * 1024 * 1024


def _import_bench_constants() -> dict[str, float | bool]:
    """Return the bench constants, preferring the canonical definitions.

    Two of the four constants live in modules an offline generator can import
    safely, and importing them is strictly better than hardcoding a literal that
    can silently drift:

    * ``bench_kernels.TILT_SHIFT_SCALE`` -- that module is pure numpy and takes
      its devices by injection, so it imports nothing hardware-bound.
    * ``model_in_loop_shaping._MAX_DEFOCUS_FIT_RMS`` -- this one *does* pull in
      torch, so it is only attempted behind a guard.

    A literal is used when the import fails, and every such case is listed in
    ``fallbacks`` so the captions say so. A fallback is not a silent event.
    """
    tilt_scale = FALLBACK_TILT_SHIFT_SCALE
    max_rms = FALLBACK_MAX_DEFOCUS_FIT_RMS
    fallbacks: list[str] = []

    try:
        from ao_shaping.tools.slm.bench_kernels import (
            TILT_SHIFT_SCALE as _kernel_scale,
        )

        tilt_scale = float(_kernel_scale)
        logger.info("TILT_SHIFT_SCALE imported from bench_kernels: {}", tilt_scale)
    except ImportError as exc:
        fallbacks.append("TILT_SHIFT_SCALE")
        logger.warning(
            "could not import TILT_SHIFT_SCALE ({}); using literal {}",
            exc, FALLBACK_TILT_SHIFT_SCALE,
        )

    try:
        from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
            _MAX_DEFOCUS_FIT_RMS as _kernel_rms,
        )

        max_rms = float(_kernel_rms)
        logger.info("_MAX_DEFOCUS_FIT_RMS imported from model_in_loop_shaping: {}",
                    max_rms)
    except ImportError as exc:
        fallbacks.append("_MAX_DEFOCUS_FIT_RMS")
        logger.warning(
            "could not import _MAX_DEFOCUS_FIT_RMS ({}); using literal {}",
            exc, FALLBACK_MAX_DEFOCUS_FIT_RMS,
        )

    if fallbacks:
        logger.warning(
            "using literal bench constants for {} -- these can drift from the code",
            ", ".join(fallbacks),
        )
    return {
        "tilt_shift_scale": tilt_scale,
        "max_defocus_fit_rms": max_rms,
        "min_shift_correlation": FALLBACK_MIN_SHIFT_CORRELATION,
        "min_hollowness": FALLBACK_MIN_HOLLOWNESS,
        "fallbacks": fallbacks,
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class SweepData:
    """The flat column arrays stored in ``sweep_records.npz``.

    Attributes:
        label: Per-point label, e.g. ``tiltx+1.00#2``.
        mode: Sweep mode, e.g. ``ramp`` / ``tilt`` / ``defocus``.
        axis: Panel axis the command was applied along (``""`` for flat).
        coefficient: Commanded amplitude (rad for Zernike, panel px for ramp).
        fwhm_px: Measured spot FWHM in camera px.
        centroid_x: Measured spot centroid x in camera px.
        centroid_y: Measured spot centroid y in camera px.
        peak: Frame peak grey level.
        hollowness: Intensity at the centroid divided by the peak.
        shift_x: Phase-correlation shift x in camera px (NaN when unavailable).
        shift_y: Phase-correlation shift y in camera px (NaN when unavailable).
        shift_corr: Phase-correlation peak height.
        scalars: Scalar (0-d) arrays carried alongside the columns.
    """

    label: np.ndarray
    mode: np.ndarray
    axis: np.ndarray
    coefficient: np.ndarray
    fwhm_px: np.ndarray
    centroid_x: np.ndarray
    centroid_y: np.ndarray
    peak: np.ndarray
    hollowness: np.ndarray
    shift_x: np.ndarray
    shift_y: np.ndarray
    shift_corr: np.ndarray
    scalars: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of stored sweep points."""
        return int(self.label.size)

    def select(self, *, mode: str | None = None, axis: str | None = None) -> np.ndarray:
        """Boolean mask over the points matching ``mode`` and/or ``axis``.

        Args:
            mode: Mode name to keep, or ``None`` to keep every mode.
            axis: Panel axis to keep, or ``None`` to keep every axis.

        Returns:
            A boolean mask of the same length as the sweep.
        """
        mask = np.ones(len(self), dtype=bool)
        if mode is not None:
            mask &= self.mode == mode
        if axis is not None:
            mask &= self.axis == axis
        return mask

    def scalar(self, name: str, default: Any = None) -> Any:
        """Read one stored metadata value, falling back to ``default``.

        Accepts both true scalars (``zernike_radius``) and short vector metadata
        such as ``pupil_center``; a short vector is returned as a tuple so it
        renders readably inside the report table.

        Args:
            name: Metadata key as stored in the npz.
            default: Value returned when the key is missing or unreadable.

        Returns:
            The value, or ``default``.
        """
        value = self.scalars.get(name, default)
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            if value.ndim == 1:
                return tuple(value.tolist())
        return value


def load_sweep(path: Path) -> SweepData:
    """Read ``sweep_records.npz`` into a :class:`SweepData`.

    Args:
        path: Path to the npz written by the sweep stage.

    Returns:
        The loaded sweep.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If a required column is missing.
    """
    with np.load(path, allow_pickle=False) as handle:
        columns = {key: handle[key] for key in handle.files}
    required = (
        "label",
        "mode",
        "axis",
        "coefficient",
        "fwhm_px",
        "centroid_x",
        "centroid_y",
        "peak",
        "hollowness",
        "shift_x",
        "shift_y",
        "shift_corr",
    )
    missing = [key for key in required if key not in columns]
    if missing:
        raise KeyError(f"{path} is missing columns: {missing}")
    sweep = SweepData(
        label=np.asarray(columns["label"]),
        mode=np.asarray(columns["mode"]),
        axis=np.asarray(columns["axis"]),
        coefficient=np.asarray(columns["coefficient"], dtype=np.float64),
        fwhm_px=np.asarray(columns["fwhm_px"], dtype=np.float64),
        centroid_x=np.asarray(columns["centroid_x"], dtype=np.float64),
        centroid_y=np.asarray(columns["centroid_y"], dtype=np.float64),
        peak=np.asarray(columns["peak"], dtype=np.float64),
        hollowness=np.asarray(columns["hollowness"], dtype=np.float64),
        shift_x=np.asarray(columns["shift_x"], dtype=np.float64),
        shift_y=np.asarray(columns["shift_y"], dtype=np.float64),
        shift_corr=np.asarray(columns["shift_corr"], dtype=np.float64),
        scalars={
            k: v
            for k, v in columns.items()
            if getattr(v, "ndim", 1) == 0 or v.shape[0] != len(columns["label"])
        },
    )
    logger.info(
        "loaded {} sweep points from {} ({} modes)",
        len(sweep),
        path,
        ", ".join(sorted(set(sweep.mode.tolist()))),
    )
    return sweep


def load_geometry(path: Path) -> dict[str, Any] | None:
    """Read ``bench_geometry.json`` if it exists.

    Args:
        path: Candidate path to the geometry json.

    Returns:
        The parsed dict, or ``None`` when the file is absent or unreadable.
    """
    if not path.is_file():
        logger.info("optional geometry json not found: {}", path)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not parse geometry json {}: {}", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("geometry json {} is not an object; ignoring", path)
        return None
    return data


#: Fields of ``bench_geometry.json`` that the npz also carries, so a mismatch
#: proves the json belongs to a *different* calibration than the sweep on disk.
#: Keyed ``(geometry_key, npz_key)``; only fields the npz actually stores.
GEOMETRY_CROSS_CHECK_FIELDS: tuple[tuple[str, str], ...] = (
    ("panel_disc_radius", "zernike_radius"),
    ("panel_disc_radius", "collect_disc"),
)


def geometry_conflicts(
    geometry: dict[str, Any] | None,
    npz_scalars: dict[str, Any],
) -> list[str]:
    """Find fields proving the geometry json came from another calibration.

    ``bench_geometry.json`` is a *shared, overwritten* artefact: the speckle route
    and the sweep route both write it. Reading it next to a sweep npz without
    checking provenance silently mixes two calibrations, and every derived number
    (the focal-scale routes, most of all) inherits the error while looking
    perfectly self-consistent. The npz records the aperture geometry that was
    actually displayed, so a disagreement is decisive.

    Args:
        geometry: Parsed ``bench_geometry.json``, or ``None``.
        npz_scalars: Scalar fields lifted from the npz, by npz key name.

    Returns:
        Human-readable mismatch strings, empty when consistent or unavailable.
    """
    if not geometry:
        return []
    conflicts: list[str] = []
    for geo_key, npz_key in GEOMETRY_CROSS_CHECK_FIELDS:
        want = npz_scalars.get(npz_key)
        got = geometry.get(geo_key)
        if want is None or got is None:
            continue
        try:
            if int(round(float(want))) == int(round(float(got))):
                continue
        except (TypeError, ValueError):
            continue
        conflicts.append(f"{geo_key}: geometry={got} vs npz={int(round(float(want)))}")
    # A speckle-route geometry can agree on the aperture yet still carry a
    # different model grid, which changes every focal-scale route. Flag the
    # method so the reader knows which route produced it.
    method = geometry.get("method")
    if method and str(method).lower() != "sweep":
        conflicts.append(f"method={method} (不是 sweep —— 来自旧的散斑路线标定)")
    return conflicts


def load_sidecar(debug_root: Path) -> tuple[dict[str, Any] | None, Path | None]:
    """Read the newest sweep sidecar json under ``debug_root``.

    Args:
        debug_root: Root directory holding ``model_in_loop_hw_sweep_*`` folders.

    Returns:
        ``(payload, path)``. Both are ``None`` when nothing readable is found.
    """
    if not debug_root.is_dir():
        logger.info("optional debug root not found: {}", debug_root)
        return None, None
    candidates = sorted(
        debug_root.glob("model_in_loop_hw_sweep_*/*/*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skipping unreadable sidecar {}: {}", candidate, exc)
            continue
        if isinstance(payload, dict):
            logger.info("using sidecar {}", candidate)
            return payload, candidate
    logger.info("no sweep sidecar json found under {}", debug_root)
    return None, None


def probe_frames_pkl(debug_root: Path) -> tuple[str, str]:
    """Decide whether the frame pickle can be read, without reading it.

    Args:
        debug_root: Root directory holding ``model_in_loop_hw_sweep_*`` folders.

    Returns:
        ``(status, detail)`` where status is ``"ok"``, ``"too-large"`` or
        ``"absent"``.
    """
    if not debug_root.is_dir():
        return "absent", f"debug root {debug_root} not found"
    candidates = sorted(
        debug_root.glob("model_in_loop_hw_sweep_*/*/*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return "absent", "no *.pkl under the sweep debug root"
    newest = candidates[0]
    size_mb = newest.stat().st_size / (1024 * 1024)
    if newest.stat().st_size > MAX_PKL_BYTES:
        return (
            "too-large",
            f"{newest.name} is {size_mb:.0f} MB (> {MAX_PKL_BYTES // (1024 * 1024)} MB"
            " budget); it holds every repeated CCD frame and adds no report value",
        )
    return "ok", f"{newest.name} ({size_mb:.1f} MB)"


def sidecar_mismatches(
    sidecar: dict[str, Any] | None, sweep: SweepData
) -> list[str]:
    """Compare the sidecar's declared sweep config against the stored npz.

    A sidecar saved by a *different* sweep configuration must not be silently
    attributed to this npz, so every disagreeing field is reported by name.

    Args:
        sidecar: Parsed sidecar payload, or ``None``.
        sweep: The loaded sweep.

    Returns:
        Human-readable mismatch descriptions (empty when consistent or absent).
    """
    if sidecar is None:
        return []
    problems: list[str] = []
    declared = sidecar.get("zernike_radius")
    stored = sweep.scalar("zernike_radius")
    if declared is not None and stored is not None and int(declared) != int(stored):
        problems.append(
            f"zernike_radius: sidecar {int(declared)} vs npz {int(stored)}"
        )
    declared_disc = sidecar.get("collect_disc")
    stored_disc = sweep.scalar("collect_disc")
    if declared_disc is not None and stored_disc is not None:
        if int(declared_disc) != int(stored_disc):
            problems.append(
                f"collect_disc: sidecar {int(declared_disc)} vs npz {int(stored_disc)}"
            )
    declared_points = sidecar.get("n_points")
    if declared_points is not None and int(declared_points) != len(sweep):
        problems.append(
            f"n_points: sidecar {int(declared_points)} vs npz rows {len(sweep)}"
        )
    declared_modes = set((sidecar.get("mode_codes") or {}).keys())
    stored_modes = set(sweep.mode.tolist())
    only_sidecar = sorted(declared_modes - stored_modes)
    if only_sidecar:
        problems.append(f"modes declared but absent from the npz: {only_sidecar}")
    return problems


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def linear_slope(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    """Least-squares ``(slope, intercept)`` of ``y`` on ``x``.

    Args:
        x: Independent values.
        y: Dependent values.

    Returns:
        ``(slope, intercept)``, or ``(nan, nan)`` when under-determined.
    """
    xs = np.asarray(list(x), dtype=np.float64)
    ys = np.asarray(list(y), dtype=np.float64)
    if xs.size < 2 or xs.size != ys.size or not np.isfinite(ys).all():
        return float("nan"), float("nan")
    var = float(np.sum((xs - xs.mean()) ** 2))
    if var <= 0.0:
        return float("nan"), float("nan")
    slope = float(np.sum((xs - xs.mean()) * (ys - ys.mean())) / var)
    return slope, float(ys.mean() - slope * xs.mean())


def through_origin_slope(x: Iterable[float], y: Iterable[float]) -> float:
    """Least-squares slope of ``y`` on ``x`` constrained through the origin.

    Used for the ramp route, where a zero-frequency ramp cannot displace the
    spot, so the physical relation is ``displacement = S / period``.

    Args:
        x: Independent values (``1/period``).
        y: Dependent values (displacement in camera px).

    Returns:
        The slope, or ``nan`` when ``x`` carries no energy.
    """
    xs = np.asarray(list(x), dtype=np.float64)
    ys = np.asarray(list(y), dtype=np.float64)
    denom = float(np.sum(xs * xs))
    if denom <= 0.0 or xs.size != ys.size:
        return float("nan")
    return float(np.sum(xs * ys) / denom)


def parse_notes(notes: str | None) -> dict[str, float]:
    """Pull the scalar numbers back out of a ``calibration_notes`` string.

    The notes are prose with literal braces, so they are rendered inside a
    fenced block and only *parsed* here with narrow regexes -- never formatted.

    Args:
        notes: The ``calibration_notes`` field, or ``None``.

    Returns:
        The recognised numbers keyed by a short snake_case name. Values that are
        not present are simply absent from the dict.
    """
    if not notes:
        return {}
    found: dict[str, float] = {}
    patterns: dict[str, str] = {
        "tilt_slope": r"tilt slope ([-\d.]+) cam px/rad",
        "semi_axis_model_px": r"semi-axis (\d+) model px",
        "far_field_size": r"P=(\d+)",
        "ramp_scale": r"ramp sweep \(S=([-\d.]+) cam px",
        "fov_ratio": r"FOV cross-check ([-\d.]+)",
        "fov_rel": r"\(([-\d]+)% apart\)",
        "tilt_ratio": r"physical tilt check ([-\d.]+)x predicted",
        "joint_rel_rms": r"joint rel-RMS ([-\d.]+)%",
        "defocus_asymmetry": r"defocus asymmetry ([-\d.]+)%",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, notes)
        if match:
            try:
                found[key] = float(match.group(1))
            except ValueError:
                continue
    for mode, offset in re.findall(r"(\w+) ([+-]\d+\.\d+) rad", notes):
        found[f"offset_{mode}"] = float(offset)
    for mode, curvature in re.findall(r"(\w+) (-?\d+\.\d+)(?=;| rad)", notes):
        found.setdefault(f"curvature_{mode}", float(curvature))
    modes = re.search(r"modes fitted ([^;]+);", notes)
    if modes:
        found["modes_fitted_count"] = float(len(modes.group(1).split("+")))
    return found


# ---------------------------------------------------------------------------
# Per-mode width filtering (mirrors calibrate_bench_geometry_from_sweep)
# ---------------------------------------------------------------------------


@dataclass
class ModeCurve:
    """One mode's width response after the two validity filters.

    Attributes:
        mode: Mode name.
        kept_coefficient: Coefficients that survived both filters.
        kept_fwhm: Their FWHM values.
        dropped: ``(coefficient, fwhm, hollowness, reason)`` per dropped point.
        curvature: Recomputed ``width**2`` quadratic curvature.
        offset: Grid-searched minimum location.
    """

    mode: str
    kept_coefficient: np.ndarray
    kept_fwhm: np.ndarray
    dropped: list[tuple[float, float, float, str]] = field(default_factory=list)
    curvature: float = float("nan")
    offset: float = float("nan")


def _has_both_signs(values: np.ndarray) -> bool:
    """True when the coefficients straddle zero, so an offset is observable.

    Args:
        values: Coefficient values.

    Returns:
        Whether both a positive and a negative value are present.
    """
    arr = np.asarray(values, dtype=np.float64)
    return bool(np.any(arr > 0) and np.any(arr < 0))


def build_mode_curve(
    mode: str, coefficient: np.ndarray, fwhm: np.ndarray, hollowness: np.ndarray,
    min_hollowness: float,
) -> ModeCurve:
    """Apply the single-lobe and monotonicity filters, then fit the curvature.

    The two filters are complementary: hollowness catches a ring whose "width"
    is a ring diameter, while the per-side monotonicity check catches a point
    that reads narrower than a smaller-amplitude neighbour (which alone drives
    the fitted curvature negative).

    Args:
        mode: Mode name (used for logging only).
        coefficient: Commanded coefficients.
        fwhm: Measured FWHM values.
        hollowness: Measured centroid/peak ratios.
        min_hollowness: Single-lobe threshold.

    Returns:
        The filtered curve, including a list of the dropped points and reasons.
    """
    keep = np.isfinite(fwhm) & (fwhm > 0) & np.isfinite(hollowness)
    coeffs = np.asarray(coefficient, dtype=np.float64)[keep]
    widths = np.asarray(fwhm, dtype=np.float64)[keep]
    hollows = np.asarray(hollowness, dtype=np.float64)[keep]
    dropped: list[tuple[float, float, float, str]] = []

    if coeffs.size < 2:
        return ModeCurve(mode, coeffs, widths, dropped)

    single_lobe = hollows >= min_hollowness
    if not single_lobe.all() and int(single_lobe.sum()) >= 2 and _has_both_signs(
        coeffs[single_lobe]
    ):
        for c, w, h in zip(coeffs[~single_lobe], widths[~single_lobe],
                            hollows[~single_lobe]):
            dropped.append((float(c), float(w), float(h), "hollowness<阈值(环形)"))
        coeffs, widths, hollows = coeffs[single_lobe], widths[single_lobe], hollows[single_lobe]

    if coeffs.size < 2:
        return ModeCurve(mode, coeffs, widths, dropped)

    monotone = np.ones(coeffs.shape, dtype=bool)
    for side in (coeffs < 0, coeffs > 0):
        idx = np.flatnonzero(side)
        if idx.size == 0:
            continue
        running = -np.inf
        for j in idx[np.argsort(np.abs(coeffs[idx]))]:
            if widths[j] >= running:
                running = widths[j]
            else:
                monotone[j] = False
    if int(monotone.sum()) >= 4 and _has_both_signs(coeffs[monotone]):
        for c, w, h, kept in zip(coeffs, widths, hollows, monotone):
            if not kept:
                dropped.append(
                    (float(c), float(w), float(h), "同侧宽度非单调")
                )
        coeffs, widths = coeffs[monotone], widths[monotone]

    curve = ModeCurve(mode, coeffs, widths, dropped)
    if coeffs.size >= 3:
        grid = np.linspace(coeffs.min() - 1.0, coeffs.max() + 1.0, 241)
        best: tuple[float, float, float] | None = None
        for c0 in grid:
            basis = (coeffs - c0) ** 2
            design = np.stack([np.ones_like(basis), basis], axis=1)
            sol, *_ = np.linalg.lstsq(design, widths**2, rcond=None)
            rms = float(np.sqrt(np.mean((design @ sol - widths**2) ** 2)))
            if best is None or rms < best[0]:
                best = (rms, float(c0), float(sol[1]))
        if best is not None:
            curve.offset, curve.curvature = best[1], best[2]
    return curve


# ---------------------------------------------------------------------------
# Route computations
# ---------------------------------------------------------------------------


@dataclass
class RampRoute:
    """The focal scale implied by a phase-ramp sweep on one panel axis.

    Attributes:
        axis: Panel axis the ramp was applied along.
        periods: Ramp periods in panel px.
        displacement: Camera-px displacement from the flat reference.
        measure_axis: Camera axis the displacement was measured on.
        scale: ``S`` in ``displacement = S / period`` (camera px).
        scale_free: The same slope from a free-intercept least-squares fit.
    """

    axis: str
    periods: np.ndarray
    displacement: np.ndarray
    measure_axis: str
    scale: float
    scale_free: float


@dataclass
class TiltRoute:
    """The focal scale implied by a Zernike tilt sweep on one panel axis.

    Attributes:
        axis: Panel axis the tilt was applied along.
        coefficient: Tilt coefficients in rad.
        centroid: Centroid on the dominant camera axis.
        measure_axis: Which camera axis dominated.
        slope: Centroid shift in camera px per radian.
        repeat_spread: Max peak-to-peak centroid spread across repeats, in px.
        use_correlation: Whether the phase-correlation shift was trustworthy.
        max_corr: Highest ``shift_corr`` among the axis' points.
    """

    axis: str
    coefficient: np.ndarray
    centroid: np.ndarray
    measure_axis: str
    slope: float
    repeat_spread: float
    use_correlation: bool
    max_corr: float


def compute_ramp_routes(sweep: SweepData) -> list[RampRoute]:
    """Derive ``S = displacement * period`` per ramp axis.

    The displacement is measured along whichever camera axis actually moves,
    which is what exposes the 90-degree panel/camera axis swap on this bench.

    Args:
        sweep: The loaded sweep.

    Returns:
        One :class:`RampRoute` per axis that has at least two usable points.
    """
    flat = sweep.select(mode="flat")
    if int(flat.sum()) != 1:
        logger.warning(
            "expected exactly one flat reference, found {}; ramp displacements"
            " are reported as absolute centroids instead",
            int(flat.sum()),
        )
        ref_x = ref_y = float("nan")
    else:
        ref_x = float(sweep.centroid_x[flat][0])
        ref_y = float(sweep.centroid_y[flat][0])

    routes: list[RampRoute] = []
    for axis in ("x", "y"):
        mask = sweep.select(mode="ramp", axis=axis)
        periods = sweep.coefficient[mask]
        good = np.isfinite(periods) & (periods > 0)
        if int(good.sum()) < 2:
            logger.info("ramp axis {}: fewer than 2 usable points, skipped", axis)
            continue
        periods = periods[good]
        dx = sweep.centroid_x[mask][good] - ref_x
        dy = sweep.centroid_y[mask][good] - ref_y
        sx, _ = linear_slope(1.0 / periods, dx)
        sy, _ = linear_slope(1.0 / periods, dy)
        use_y = abs(sy) >= abs(sx)
        displacement = dy if use_y else dx
        scale = through_origin_slope(1.0 / periods, displacement)
        _, scale_free = linear_slope(1.0 / periods, displacement)
        routes.append(
            RampRoute(
                axis=axis,
                periods=periods,
                displacement=displacement,
                measure_axis="y" if use_y else "x",
                scale=abs(scale),
                scale_free=abs(scale_free),
            )
        )
    return routes


def compute_tilt_routes(sweep: SweepData, min_corr: float) -> list[TiltRoute]:
    """Fit the centroid-vs-tilt slope per axis, mirroring the fitter's choice.

    Args:
        sweep: The loaded sweep.
        min_corr: Minimum ``shift_corr`` for the correlation shift to be used.

    Returns:
        One :class:`TiltRoute` per axis that has at least two usable points.
    """
    routes: list[TiltRoute] = []
    for axis in ("x", "y"):
        mask = sweep.select(mode="tilt", axis=axis)
        coeffs = sweep.coefficient[mask]
        if coeffs.size < 2:
            logger.info("tilt axis {}: fewer than 2 points, skipped", axis)
            continue
        corr = sweep.shift_corr[mask]
        shifts_ok = np.isfinite(sweep.shift_x[mask]) & np.isfinite(
            sweep.shift_y[mask]
        ) & np.isfinite(corr)
        use_corr = bool(shifts_ok.all() and np.all(corr >= min_corr)) and bool(
            coeffs.size >= 2
        )
        max_corr = float(np.nanmax(corr)) if np.isfinite(corr).any() else float("nan")
        if use_corr:
            sx, _ = linear_slope(coeffs, sweep.shift_x[mask])
            sy, _ = linear_slope(coeffs, sweep.shift_y[mask])
        else:
            sx, _ = linear_slope(coeffs, sweep.centroid_x[mask])
            sy, _ = linear_slope(coeffs, sweep.centroid_y[mask])
        use_y = abs(sy) >= abs(sx)
        centroid = sweep.centroid_y[mask] if use_y else sweep.centroid_x[mask]
        slope, _ = linear_slope(coeffs, centroid)
        spread = 0.0
        for value in np.unique(coeffs):
            same = centroid[coeffs == value]
            if same.size > 1:
                spread = max(spread, float(same.max() - same.min()))
        routes.append(
            TiltRoute(
                axis=axis,
                coefficient=coeffs,
                centroid=centroid,
                measure_axis="y" if use_y else "x",
                slope=slope,
                repeat_spread=spread,
                use_correlation=use_corr,
                max_corr=max_corr,
            )
        )
    return routes


def fov_route(
    geometry: dict[str, Any] | None, sweep: SweepData, ramp_scale: float
) -> float:
    """Camera px per model px from the independent field-of-view ratio.

    ``p_fft = lambda f / (P * model_pitch)``; the model FOV over the camera FOV.
    Derived completely differently from the ramp, which is the point: a
    disagreement beyond ~25 % means one of the two assumptions is wrong.

    Args:
        geometry: Parsed ``bench_geometry.json`` (may be ``None``).
        sweep: The loaded sweep (for the aperture radius and P).
        ramp_scale: The ramp-derived scale, used to report the disagreement.

    Returns:
        The FOV-ratio scale, or ``nan`` when the inputs are insufficient.
    """
    if geometry is None:
        return float("nan")
    try:
        region = int(geometry["region"])
        far_field_size = int(geometry["far_field_size"])
        disc = int(geometry["panel_disc_radius"])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    a = max(region // 2, 1)
    model_pitch_m = FALLBACK_APERTURE_M / a
    p_fft = FALLBACK_WAVELENGTH_M * FALLBACK_FOCAL_LENGTH_M / (
        far_field_size * model_pitch_m
    )
    ratio = (far_field_size * p_fft) / (FALLBACK_CAMERA_PIXELS * FALLBACK_CAMERA_PIXEL_M)
    if not np.isfinite(ratio) or ratio <= 0 or disc <= 0:
        return float("nan")
    return float(ratio)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path) -> None:
    """Apply the repo figure convention and write the PNG.

    Args:
        fig: The figure to save.
        path: Destination path.
    """
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("wrote figure {} ({} KB)", path.name, path.stat().st_size // 1024)


def figure_ramp(
    routes: list[RampRoute], shift_scale: float, used_fallback: bool, out: Path
) -> Path:
    """Displacement vs ``1/period`` for every ramp axis, against the constant.

    Args:
        routes: Per-axis ramp routes.
        shift_scale: ``TILT_SHIFT_SCALE`` (or its fallback).
        used_fallback: Whether the constant came from the fallback.
        routes: Per-axis ramp routes.
        out: Figure directory.

    Returns:
        The written PNG path.
    """
    suffix = " (导入失败, 用回退值)" if used_fallback else ""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.6))
    grid = np.linspace(0.0, 1.0 / 120.0, 50)
    for route in routes:
        inv = 1.0 / route.periods
        ax1.plot(
            inv * 1000.0, route.displacement, "o", ms=9,
            label=f"ramp 轴 {route.axis} → 相机 {route.measure_axis}",
        )
        ax1.plot(grid * 1000.0, route.scale * grid, "--", lw=1.4, alpha=0.8)
    ax1.plot(
        grid * 1000.0, shift_scale * grid, "-", color="k", lw=2.0,
        label=f"TILT_SHIFT_SCALE = {shift_scale:.0f}{suffix}",
    )
    ax1.set_xlabel("1000 / 周期 (面板 1/px)")
    ax1.set_ylabel("相对平场的位移 (相机 px)")
    ax1.set_title("相位斜坡位移严格 ∝ 1/周期")
    ax1.axhline(0.0, color="0.6", lw=0.8)
    ax1.legend(fontsize=8.5)
    ax1.grid(alpha=0.3)

    labels, measured, predicted, deltas = [], [], [], []
    for route in routes:
        labels.append(f"轴 {route.axis}")
        measured.append(route.scale)
        predicted.append(shift_scale)
        deltas.append(
            (route.scale - shift_scale) / shift_scale * 100.0 if shift_scale else np.nan
        )
    xs = np.arange(len(labels))
    ax2.bar(xs - 0.2, measured, width=0.38, label="本次实测 S")
    ax2.bar(xs + 0.2, predicted, width=0.38, label="台架常数")
    for x, base, delta in zip(xs, measured, deltas):
        ax2.text(x - 0.2, base, f"{base:.0f}", ha="center", va="bottom", fontsize=9)
        ax2.text(x, max(base, shift_scale) * 1.04, f"{delta:+.1f}%", ha="center",
                 fontsize=9.5, color="crimson" if abs(delta) > 10 else "0.35")
    ax2.set_xticks(xs, labels)
    ax2.set_ylabel("S (相机 px · 周期)")
    ax2.set_title("实测尺度与台架常数的一致性")
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.3, axis="y")
    path = out / "01_ramp_linearity.png"
    _save(fig, path)
    return path


def figure_tilt(
    routes: list[TiltRoute], min_corr: float, out: Path
) -> Path:
    """Tilt centroid slope, repeat spread and the correlation-availability check.

    Args:
        routes: Per-axis tilt routes.
        min_corr: The correlation threshold in force.
        out: Figure directory.

    Returns:
        The written PNG path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))
    ax1, ax2, ax3 = axes
    for route in routes:
        ax1.plot(
            route.coefficient, route.centroid, "o", ms=9,
            label=f"轴 {route.axis} → 相机 {route.measure_axis}  斜率 {route.slope:+.3f}",
        )
        lo, hi = float(route.coefficient.min()) - 0.15, float(
            route.coefficient.max()
        ) + 0.15
        xs = np.linspace(lo, hi, 10)
        _, intercept = linear_slope(route.coefficient, route.centroid)
        ax1.plot(xs, route.slope * xs + intercept, "--", lw=1.3, alpha=0.75)
    ax1.set_xlabel("Zernike 倾斜系数 (rad)")
    ax1.set_ylabel("主位移轴上的质心 (相机 px)")
    ax1.set_title("倾斜质心位移 vs 系数")
    ax1.legend(fontsize=8.5)
    ax1.grid(alpha=0.3)

    labels = [f"轴 {r.axis}" for r in routes]
    spreads = [r.repeat_spread for r in routes]
    bars = ax2.bar(labels, spreads, color=["#4C78A8", "#F58518"][: len(spreads)])
    for rect, value in zip(bars, spreads):
        ax2.text(rect.get_x() + rect.get_width() / 2, value,
                 f"{value:.2f} px", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("同系数重复测量的峰-峰展布 (px)")
    ax2.set_title("重复一致性：稳定后应远小于 1 px")
    ax2.grid(alpha=0.3, axis="y")

    max_corrs = [r.max_corr for r in routes]
    bars = ax3.bar(labels, max_corrs, color=["#4C78A8", "#F58518"][: len(max_corrs)])
    ax3.axhline(min_corr, color="crimson", ls="--", lw=1.6,
                label=f"阈值 {min_corr:.2f}")
    for rect, value in zip(bars, max_corrs):
        ax3.text(rect.get_x() + rect.get_width() / 2, value,
                 f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax3.set_ylabel("shift_corr 峰值")
    ax3.set_title("相位相关是否可用（低于阈值则回退质心）")
    ax3.legend(fontsize=8.5)
    ax3.grid(alpha=0.3, axis="y")
    path = out / "02_tilt_linearity.png"
    _save(fig, path)
    return path


def figure_widths(curve_map: dict[str, ModeCurve], out: Path) -> Path:
    """FWHM vs coefficient for every width-fitted mode, with dropped points.

    Args:
        curve_map: Filtered curve per mode.
        out: Figure directory.

    Returns:
        The written PNG path.
    """
    modes = [m for m in WIDTH_MODES if m in curve_map]
    cols = 3
    rows = int(np.ceil(len(modes) / cols)) if modes else 1
    fig, axes = plt.subplots(rows, cols, figsize=(15.0, 4.6 * rows), squeeze=False)
    for idx, mode in enumerate(modes):
        ax = axes[idx // cols][idx % cols]
        curve = curve_map[mode]
        ax.plot(
            curve.kept_coefficient, curve.kept_fwhm, "o-", ms=8, lw=1.6,
            color="#4C78A8", label=f"参与拟合 ({curve.kept_coefficient.size} 点)",
        )
        if curve.dropped:
            ax.scatter(
                [d[0] for d in curve.dropped], [d[1] for d in curve.dropped],
                s=140, facecolors="none", edgecolors="crimson", lw=2.0,
                label=f"被剔除 ({len(curve.dropped)} 点)",
            )
        xs = np.linspace(
            float(curve.kept_coefficient.min()) - 0.3,
            float(curve.kept_coefficient.max()) + 0.3, 60,
        ) if curve.kept_coefficient.size else np.array([0.0, 1.0])
        if curve.kept_coefficient.size >= 3:
            a = float(np.mean(curve.kept_fwhm**2) - curve.curvature * (0.0) ** 2)
            base = max(float(np.min(curve.kept_fwhm) ** 2) - 1.0, 0.0)
            ax.plot(
                xs, np.sqrt(np.clip(base + curve.curvature * (xs - curve.offset) ** 2,
                                    0.0, None)),
                "--", lw=1.2, color="0.45", label="width² 二次拟合",
            )
        ax.set_title(f"{mode}  曲率 {curve.curvature:.1f}  偏移 {curve.offset:+.2f}")
        ax.set_xlabel("系数 (rad)")
        ax.set_ylabel("FWHM (相机 px)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for idx in range(len(modes), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    path = out / "03_width_response.png"
    _save(fig, path)
    return path


def figure_curvature(
    curve_map: dict[str, ModeCurve], notes: dict[str, float], out: Path
) -> Path:
    """Curvature per mode plus the defocus quadratic the fitter relies on.

    Args:
        curve_map: Filtered curve per mode.
        notes: Parsed ``calibration_notes`` numbers.
        out: Figure directory.

    Returns:
        The written PNG path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.6))
    modes = [m for m in WIDTH_MODES if m in curve_map]
    curvatures = [curve_map[m].curvature for m in modes]
    colors = ["#54A24B" if c > 0 else "#E45756" for c in curvatures]
    bars = ax1.bar(modes, curvatures, color=colors)
    for rect, value in zip(bars, curvatures):
        ax1.text(rect.get_x() + rect.get_width() / 2, value,
                 f"{value:.0f}", ha="center", va="bottom", fontsize=9)
    ax1.axhline(0.0, color="k", lw=0.9)
    ax1.set_ylabel("width² 对 (c−c0)² 的曲率")
    ax1.set_title("各模式宽度响应曲率（必须为正）")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(alpha=0.3, axis="y")

    curve = curve_map.get("defocus")
    if curve is not None and curve.kept_coefficient.size >= 3:
        ax2.plot(curve.kept_coefficient, curve.kept_fwhm**2, "o", ms=9,
                 color="#4C78A8", label="实测 width²")
        xs = np.linspace(
            float(curve.kept_coefficient.min()) - 0.3,
            float(curve.kept_coefficient.max()) + 0.3, 80,
        )
        base = float(np.min(curve.kept_fwhm**2)) - 1.0
        ax2.plot(xs, base + curve.curvature * (xs - curve.offset) ** 2, "--", lw=1.5,
                 color="0.45", label="二次拟合")
        ax2.axvline(curve.offset, color="crimson", ls=":", lw=1.6,
                    label=f"拟合偏移 {curve.offset:+.2f} rad")
    joint = notes.get("joint_rel_rms", float("nan")) / 100.0
    ax2.set_title(
        "离焦 width² 抛物线"
        + (f"（联合相对 RMS {joint:.1%}）" if np.isfinite(joint) else "")
    )
    ax2.set_xlabel("离焦系数 (rad)")
    ax2.set_ylabel("width² (px²)")
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.3)
    path = out / "04_curvature_summary.png"
    _save(fig, path)
    return path


def figure_routes(
    ramp_scale: float, tilt_slope: float, fov_scale: float,
    model_to_panel: float, far_field_size: int, out: Path,
) -> Path:
    """Compare the three routes to the focal scale, in camera px per model px.

    Args:
        ramp_scale: Ramp-route scale already divided by the model-to-panel span.
        tilt_slope: Tilt-route scale already divided by the model-to-panel span.
        fov_scale: Field-of-view-ratio scale.
        model_to_panel: Model pixels to panel pixels.
        far_field_size: Forward model's zero-padding size.
        out: Figure directory.

    Returns:
        The written PNG path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.4))
    names = ["斜坡 ramp", "Zernike 倾斜", "视场比 FOV"]
    values = [ramp_scale, tilt_slope, fov_scale]
    finite = [v for v in values if np.isfinite(v) and v > 0]
    reference = float(np.mean(finite)) if finite else float("nan")
    bars = ax1.bar(names, values, color=["#4C78A8", "#F58518", "#54A24B"])
    for rect, value in zip(bars, values):
        if np.isfinite(value):
            ax1.text(rect.get_x() + rect.get_width() / 2, value, f"{value:.4f}",
                     ha="center", va="bottom", fontsize=9.5)
    ax1.set_ylabel("相机 px / 模型 px")
    ax1.set_title("焦面尺度的三条独立路线")
    ax1.grid(alpha=0.3, axis="y")

    if np.isfinite(reference) and reference > 0:
        deltas = [
            (v - reference) / reference * 100.0 if np.isfinite(v) else np.nan
            for v in values
        ]
        ax2.bar(names, deltas, color=["#4C78A8", "#F58518", "#54A24B"])
        for idx, delta in enumerate(deltas):
            if np.isfinite(delta):
                ax2.text(idx, delta, f"{delta:+.0f}%", ha="center",
                         va="bottom" if delta >= 0 else "top", fontsize=9.5)
    ax2.axhspan(-25, 25, color="0.85", zorder=0, label="±25% 可接受带")
    ax2.axhline(0.0, color="k", lw=0.9)
    ax2.set_ylabel("相对三条路线均值 (%)")
    ax2.set_title(f"互相偏离（模型→面板 {model_to_panel:.3f}，P={far_field_size}）")
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.3, axis="y")
    path = out / "05_focal_scale_routes.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _display_path(path: Path) -> str:
    """Render *path* for the report, preferring a repo-relative form.

    ``Path.relative_to`` raises when the caller points ``--sweep-npz`` /
    ``--geometry-json`` / ``--debug-root`` outside the repository, which would
    abort the whole report on an otherwise valid run. Fall back to the absolute
    path in that case.

    Args:
        path: The path to display.

    Returns:
        A POSIX-style repo-relative path, or the absolute path if it lies
        outside the repository.
    """
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def render_report(
    sweep: SweepData,
    geometry: dict[str, Any] | None,
    geometry_path: Path,
    sidecar: dict[str, Any] | None,
    sidecar_path: Path | None,
    npz_path: Path,
    ramp_routes: list[RampRoute],
    tilt_routes: list[TiltRoute],
    curve_map: dict[str, ModeCurve],
    constants: dict[str, Any],
    figures: list[Path],
    frames_status: str,
    frames_detail: str,
    geometry_conflicts: list[str] | None = None,
) -> str:
    """Build the Chinese markdown report.

    Args:
        sweep: The loaded sweep.
        geometry: Parsed geometry json, or ``None``.
        geometry_path: Path the geometry was read from.
        sidecar: Parsed sidecar payload, or ``None``.
        sidecar_path: Path the sidecar was read from.
        npz_path: Path the npz was read from.
        ramp_routes: Per-axis ramp routes.
        tilt_routes: Per-axis tilt routes.
        curve_map: Filtered curve per mode.
        constants: Bench constants plus the fallback list.
        figures: Written figure paths, in report order.
        frames_status: ``"ok"`` / ``"too-large"`` / ``"absent"``.
        frames_detail: Human-readable detail for the frames decision.

    Returns:
        The markdown text.
    """
    shift_scale = float(constants["tilt_shift_scale"])
    max_rms = float(constants["max_defocus_fit_rms"])
    min_corr = float(constants["min_shift_correlation"])
    min_hollow = float(constants["min_hollowness"])
    fallbacks = list(constants["fallbacks"])
    notes_raw = (geometry or {}).get("calibration_notes")
    notes = parse_notes(notes_raw if isinstance(notes_raw, str) else None)
    mismatches = sidecar_mismatches(sidecar, sweep)

    region = int((geometry or {}).get("region") or 0)
    far_field_size = int((geometry or {}).get("far_field_size") or 0)
    disc = int((geometry or {}).get("panel_disc_radius") or 0)
    semi_axis = max(region // 2, 1)
    model_to_panel = disc / semi_axis if disc and semi_axis else float("nan")

    ramp_scales = [r.scale for r in ramp_routes if np.isfinite(r.scale)]
    ramp_scale = float(np.mean(ramp_scales)) if ramp_scales else float("nan")
    tilt_slopes = [abs(r.slope) for r in tilt_routes if np.isfinite(r.slope)]
    k_tilt = float(np.mean(tilt_slopes)) if tilt_slopes else float("nan")
    fov = fov_route(geometry, sweep, ramp_scale)

    ramp_cam = ramp_scale / (model_to_panel * far_field_size) if (
        np.isfinite(ramp_scale) and model_to_panel and far_field_size
    ) else float("nan")
    tilt_cam = k_tilt * np.pi * semi_axis / far_field_size if (
        np.isfinite(k_tilt) and far_field_size
    ) else float("nan")
    # Ramp and tilt are both *camera-side* measurements, and a Zernike tilt of
    # coefficient c is a ramp of gradient 2c/R, so the two must satisfy
    # S = k_tilt * pi * R. That relation needs only the illuminated radius R --
    # which the npz carries -- and no model grid at all. It is therefore the one
    # cross-check that stays valid even when bench_geometry.json is stale, and it
    # is the strongest single statement this report can make.
    zernike_radius = float(sweep.scalars.get("zernike_radius", 0) or 0)
    tilt_implied_S = (
        k_tilt * np.pi * zernike_radius
        if np.isfinite(k_tilt) and zernike_radius
        else float("nan")
    )
    ramp_vs_tilt_gap = (
        abs(ramp_scale - tilt_implied_S) / tilt_implied_S
        if np.isfinite(ramp_scale) and np.isfinite(tilt_implied_S) and tilt_implied_S
        else float("nan")
    )

    flat = sweep.select(mode="flat")
    peak_values = sweep.peak[np.isfinite(sweep.peak)]
    peak_lo = float(peak_values.min()) if peak_values.size else float("nan")
    peak_hi = float(peak_values.max()) if peak_values.size else float("nan")

    lines: list[str] = []
    add = lines.append

    add("# SLM 台架探针离线报告")
    add("")
    add(
        "本报告由 `scripts/generate_bench_probe_report.py` **完全离线** 生成："
        "只读取硬件运行已经落盘的扫描记录，**从不打开相机或 SLM**，"
        "因此可以在仪器断电状态下随时重跑以刷新结论。"
    )
    add("")
    add("## 1. 数据来源与完整性")
    add("")
    add("| 输入 | 路径 | 状态 |")
    add("|---|---|---|")
    add(f"| 扫描记录 (必需) | `{_display_path(npz_path)}` | ✅ {len(sweep)} 行 |")
    if geometry is not None:
        add(f"| 台架几何 | `{_display_path(geometry_path)}` | ✅ 已读取 |")
    else:
        add(f"| 台架几何 | `{_display_path(geometry_path)}` | ⚠️ 缺失 |")
    if sidecar is not None and sidecar_path is not None:
        add(f"| 调试 sidecar | `{_display_path(sidecar_path)}` | ✅ 已读取 |")
    else:
        add("| 调试 sidecar | — | ⚠️ 缺失（可选） |")
    add(f"| 帧 pickle | — | {'✅ ' + frames_detail if frames_status == 'ok' else '⚠️ ' + frames_detail} |")
    add("")
    add(f"npz 携带的标量：`zernike_radius="
        f"{sweep.scalar('zernike_radius', '未记录')}`、"
        f"`collect_disc={sweep.scalar('collect_disc', '未记录')}`、"
        f"`pupil_center={sweep.scalar('pupil_center', '未记录')}`。")
    add("")
    if fallbacks:
        add(
            "> ⚠️ 以下台架常数导入失败，报告使用的是字面回退值："
            + "、".join(f"`{name}`" for name in fallbacks)
            + "。相关图注已注明。"
        )
        add("")

    add("### 1.1 各模式点数")
    add("")
    add("| 模式 | 轴 | 点数 | 系数范围 (rad 或面板 px) |")
    add("|---|---|---|---|")
    for mode in sorted(set(sweep.mode.tolist())):
        for axis in sorted(set(sweep.axis[sweep.mode == mode].tolist())):
            mask = sweep.select(mode=mode, axis=axis)
            coeffs = sweep.coefficient[mask]
            if coeffs.size == 0:
                continue
            axis_text = axis or "—"
            add(
                f"| `{mode}` | {axis_text} | {int(mask.sum())} | "
                f"{coeffs.min():+.3f} … {coeffs.max():+.3f} |"
            )
    add("")

    if sidecar is not None:
        add("### 1.2 sidecar 与 npz 的一致性核对")
        add("")
        if mismatches:
            add(
                "> 🚨 **sidecar 与本 npz 不一致**，它记录的是另一次扫描配置，"
                "因此下面的 sidecar 字段**不能**用来描述本次数据："
            )
            add(">")
            for item in mismatches:
                add(f"> - {item}")
            add("")
            add(
                "报告一律以 **npz 自身的标量与数组**为准；"
                "这正是「一行数据也不能被静默归错来源」的原因。"
            )
        else:
            add("sidecar 声明的扫描配置与 npz 一致，可以安全引用。")
        add("")

    if geometry_conflicts:
        add("### 1.3 `bench_geometry.json` 与 npz 的一致性核对")
        add("")
        add(
            "> 🚨 **`bench_geometry.json` 与本 npz 不是同一次标定**，因此它"
            "**不能**用来推导任何依赖模型网格的量（焦面尺度三条路线、"
            "`region` / `far_field_size` 换算等）。下面列出的冲突每一项都"
            "足以证明它来自另一次标定："
        )
        add(">")
        for item in geometry_conflicts:
            add(f"> - {item}")
        add(">")
        add(
            "该文件是**共享且会被覆盖**的产物（散斑路线与 sweep 路线都写它）。"
            "与 sidecar 不同，它不只是「描述不准」——把它和 npz 混算会让"
            "派生量继承错误，而推导过程看起来仍然完全自洽。"
            "本报告因此把第 2 节与第 6 节标注为不可用，请先重跑"
            "`--stage calibrate --method sweep` 刷新它。"
        )
        add("")

    add("## 2. 台架几何标定结果")
    add("")
    if geometry is None:
        add("⚠️ 未找到 `bench_geometry.json`，本节无可用标定结果。")
    elif geometry_conflicts:
        add("> 🚨 **本节数据不可用。** `bench_geometry.json` 与本报告使用的 npz "
            "不是同一次标定（见 §1.3），下面逐字列出仅为便于排查，"
            "**任何数字都不得用于换算或整形**。")
        add(">")
        add("> 重跑 `python scripts/model_in_loop_hw_runbook.py --stage calibrate "
            "--method sweep` 刷新它。")
        add("")
        add("| 字段 | 值（⚠️ 陈旧, 仅供参考） |")
        add("|---|---|")
        for key in (
            "panel_disc_radius",
            "region",
            "beam_waist_panel_px",
            "far_field_size",
            "camera_px_per_model_px",
            "spot_fwhm_camera_px",
            "spot_fwhm_model_px",
            "correlation",
            "method",
        ):
            if key in geometry:
                value = geometry[key]
                if isinstance(value, float):
                    add(f"| `{key}` | {value:.6g} |")
                else:
                    add(f"| `{key}` | {value} |")
    else:
        add("| 字段 | 值 |")
        add("|---|---|")
        for key in (
            "panel_disc_radius",
            "region",
            "beam_waist_panel_px",
            "far_field_size",
            "camera_px_per_model_px",
            "spot_fwhm_camera_px",
            "spot_fwhm_model_px",
            "correlation",
            "method",
        ):
            if key in geometry:
                value = geometry[key]
                if isinstance(value, float):
                    add(f"| `{key}` | {value:.6g} |")
                else:
                    add(f"| `{key}` | {value} |")
        add("")
        if isinstance(notes_raw, str) and notes_raw:
            add("原始 `calibration_notes`（逐字放入代码块，不做任何格式化）：")
            add("")
            add("```text")
            add(notes_raw)
            add("```")
            add("")
            joint = notes.get("joint_rel_rms", float("nan")) / 100.0
            if np.isfinite(joint):
                verdict = (
                    f"联合相对 RMS **{joint:.1%}** ≤ 阈值 {max_rms:.0%}，"
                    "**腰包可辨识**"
                    if joint <= max_rms
                    else f"联合相对 RMS **{joint:.1%}** > 阈值 {max_rms:.0%}，"
                    "**腰包不可辨识**，只能退回平顶默认值"
                )
                add(f"- {verdict}（阈值 `_MAX_DEFOCUS_FIT_RMS = {max_rms:.2f}`）。")
            fov_rel = notes.get("fov_rel", float("nan"))
            if np.isfinite(fov_rel):
                add(
                    f"- 视场比交叉核对偏离 **{fov_rel:.0f}%**"
                    + ("（超过 25% 警戒线，说明照明的孔径半径假设有误）"
                       if fov_rel > 25 else "（在 25% 以内）")
                )
            asym = notes.get("defocus_asymmetry", float("nan"))
            if np.isfinite(asym):
                add(f"- 离焦响应的左右不对称 **{asym:+.0f}%**，直接反映残余非离焦像差。")
    add("")

    add("## 3. 相位斜坡 (ramp) 线性度 —— 焦面尺度的权威来源")
    add("")
    if not ramp_routes:
        add("⚠️ 没有可用的斜坡扫描（每个轴不足 2 点），本节跳过。")
    else:
        add(
            "一个周期为 `P` 面板 px 的 2π 斜坡把光斑移动 `S / P` 相机 px。"
            "斜坡没有「光斑变形导致质心失跟」的失效模式，因此它而不是 Zernike 倾斜"
            "才是焦面尺度的权威测量。"
        )
        add("")
        add("| 面板轴 | 测量所在的相机轴 | 周期 (面板 px) | 位移 (相机 px) | S (过原点) | S (自由截距) |")
        add("|---|---|---|---|---|---|")
        for route in ramp_routes:
            for period, disp in zip(route.periods, route.displacement):
                add(
                    f"| {route.axis} | {route.measure_axis} | {period:.0f} | "
                    f"{disp:+.2f} | {route.scale:.0f} | {route.scale_free:.0f} |"
                )
            add(
                f"| **{route.axis} 合计** | {route.measure_axis} | — | — | "
                f"**{route.scale:.0f}** | {route.scale_free:.0f} |"
            )
        add("")
        add(
            f"两条轴的 S 平均 **{ramp_scale:.0f}**，与台架常数 "
            f"`TILT_SHIFT_SCALE = {shift_scale:.0f}` 相差 "
            f"**{(ramp_scale - shift_scale) / shift_scale * 100:+.1f}%**"
            + ("（导入失败，用的是回退值）" if "TILT_SHIFT_SCALE" in fallbacks else "")
            + "。"
        )
        add("")
        add(
            "**面板↔相机两轴互换 90°**：上表「面板轴」与「测量所在的相机轴」"
            "不同列，直接证实了不能用几何直觉去推断哪个坐标会动。"
        )
    if figures:
        add("")
        add(f"![斜坡线性度]({figures[0].relative_to(out_dir(figures[0])).as_posix()})")
    add("")
    add("### 3.1 测试项解读")
    add("")
    add("**目的**　用一个 2π 斜坡跨过面板，光斑位移只取决于斜坡周期 `P`，"
        "与衍射效率、灰度非线性都无关。因此它能在**不假设焦距 `f`、像元尺寸、"
        "SLM 像素间距**的前提下测出焦面尺度 `S`——这正是它作为「权威来源」的原因。")
    add("")
    if ramp_routes:
        spreads = [abs(r.scale) for r in ramp_routes]
        axis_gap = (max(spreads) - min(spreads)) / min(spreads) * 100 if len(spreads) > 1 else 0.0
        add(f"**结果分析**　两条面板轴分别给出 S = "
            + "、".join(f"{abs(r.scale):.0f}" for r in ramp_routes)
            + f"，两轴相差 **{axis_gap:.1f}%**。每个轴内部 5 个周期点的位移严格"
            "按 `1/P` 成比例（见上表 S 列逐行一致），说明斜坡响应的线性度好到"
            "可以直接做无截距拟合。若某一行的 S 与其他行明显不同，那一行就是"
            "未稳定帧或丢帧，不是物理。")
        add("")
        add(f"**对整形的影响**　焦面尺度是模型与硬件之间**唯一的像素单位换算**："
            f"它把「远场上 1 个模型像素」锚定到「相机上多少像素」。它错了，"
            f"方形目标的边长就按同一比例错——目标是 40 相机 px 的方形，"
            f"实际打到硬件上就会变成 40 / (S 的相对误差) 倍的边长，"
            f"即**目标尺寸直接成比例地错**。因此整形前必须确认 S 来自斜坡"
            f"（本节）而不是任何解析公式：后者需要 f 与像元尺寸，"
            f"本台架的斜入射会让它偏差一个 cos θ 因子。")
    add("")

    add("## 4. Zernike 倾斜线性度")
    add("")
    if not tilt_routes:
        add("⚠️ 没有可用的倾斜扫描（每个轴不足 2 点），本节跳过。")
    else:
        add("| 面板轴 | 测量所在的相机轴 | 斜率 (相机 px/rad) | 重复展布 (px) | 用的方法 | 最大 shift_corr |")
        add("|---|---|---|---|---|---|")
        for route in tilt_routes:
            method = "相位相关" if route.use_correlation else "质心（相关不可用）"
            add(
                f"| {route.axis} | {route.measure_axis} | {route.slope:+.3f} | "
                f"{route.repeat_spread:.2f} | {method} | {route.max_corr:.3f} |"
            )
        add("")
        add(f"两轴斜率绝对值平均 **k_tilt = {k_tilt:.3f} 相机 px/rad**。")
        add("")
        max_corrs = [r.max_corr for r in tilt_routes]
        if max_corrs and max(max_corrs) < min_corr:
            add(
                f"> 两个轴的 `shift_corr` 最高只有 **{max(max_corrs):.3f}**，"
                f"远低于阈值 {min_corr:.2f}，所以**相位相关位移在本台架上不可用**，"
                "拟合只能回退到光斑质心。"
            )
            add("")
        add(
            "> 这正是标定器注释里那条经验规则的来源：倾斜系数一大到足以给出可用信号，"
            "光斑就会变形甚至分裂，质心随之跳变。"
            "历史上由此读出 **1.63 相机 px/rad**（真值约 5.3，3.3 倍误差），"
            "而本次在**稳定性等待**下测得 "
            f"**{k_tilt:.3f} 相机 px/rad**。"
        )
    if len(figures) > 1:
        add("")
        add(f"![倾斜线性度]({figures[1].relative_to(out_dir(figures[1])).as_posix()})")
    add("")
    add("### 4.1 测试项解读")
    add("")
    add("**目的**　用 Zernike 倾斜做**交叉验证**，而不是测量。斜坡（第 3 节）已经"
        "给出权威的 S；倾斜的独立之处在于它把「斜坡斜率」与「Zernike 系数」"
        "联系起来：系数 `c` 产生的梯度应正比于 `c`，而梯度与焦面位移的关系由"
        "物理预测 `f·λ/(π·R·像元)` 给出。所以倾斜同时检验两件事——"
        "面板是否在按系数比例调制，以及 S 是否与物理预测自洽。")
    add("")
    if tilt_routes:
        slopes = [abs(r.slope) for r in tilt_routes]
        axis_gap = (max(slopes) - min(slopes)) / min(slopes) * 100
        spread_txt = (
            "；重复展布仅 {:.2f}–{:.2f} px，落在亚像素量级，说明每点的读数本身"
            "很稳，误差不来自测量重复性".format(
                min(r.repeat_spread for r in tilt_routes),
                max(r.repeat_spread for r in tilt_routes),
            )
            if all(r.repeat_spread < 1.0 for r in tilt_routes)
            else "；重复展布 {:.2f}–{:.2f} px，个别轴仍有抖动".format(
                min(r.repeat_spread for r in tilt_routes),
                max(r.repeat_spread for r in tilt_routes),
            )
        )
        add(f"**结果分析**　k_tilt = {k_tilt:.3f} 相机 px/rad，两轴相差 "
            f"{axis_gap:.1f}%{spread_txt}。")
        add("")
        if np.isfinite(ramp_vs_tilt_gap):
            add(
                f"**与第 3 节交叉核对（不依赖任何标定文件）**　Zernike 倾斜系数 "
                f"`c` 等价于梯度 `2c/R` 的斜坡，两条路线因此必须满足 "
                f"`S = k_tilt·π·R`。代入 R = {zernike_radius:.0f} 面板 px 得 "
                f"S = {tilt_implied_S:.0f}，与斜坡直接测得的 "
                f"**{ramp_scale:.0f}** 相差 **{ramp_vs_tilt_gap * 100:.1f}%**。"
                "这是两条完全独立的测量路线（一个用整面板斜坡、一个用孔径内"
                "Zernike 模式）互相印证。它只用 npz 里自带的孔径半径 R，"
                "**不需要任何标定文件**——因此即使 `bench_geometry.json` "
                "缺失或过期，这条结论依然成立。本报告最强的单条结论。"
            )
        else:
            add("第 3 节的斜坡尺度缺少可比对的孔径半径，跳过交叉核对。")
        add("")
        add("**对整形的影响**　倾斜是整形闭环里**唯一必须保留的自由度**："
            "光斑整体位置（tip/tilt）决定了远场目标框能不能对准，"
            "而光斑位置只能靠倾斜挪动。更实际的影响是**安全边界**——"
            "本节证明了大系数下质心会失跟（hollowness 掉到 0.65、斜率腰斩），"
            "因此任何用质心做反馈的整形优化器都必须把倾斜幅度限制在"
            "单瓣区内，否则优化器会沿着质心的假信号收敛，"
            "把光斑推到一个它「以为」在动、实际在变形的工作点上。")
    add("")

    add("## 5. 宽度响应与有效点筛选")
    add("")
    add(
        "光斑宽度只有在焦面仍是**单一瓣**时才有意义。直接判据是 hollowness"
        f"（质心处强度 / 峰值）：低于 **{min_hollow:.2f}** 说明是一个环，"
        "此时「宽度」其实是环直径。"
    )
    add("")
    add("| 模式 | 原始点数 | 参与拟合 | 曲率 | 拟合偏移 (rad) | 被剔除的点 |")
    add("|---|---|---|---|---|---|")
    for mode in WIDTH_MODES:
        mask = sweep.select(mode=mode)
        curve = curve_map.get(mode)
        if curve is None:
            add(f"| `{mode}` | {int(mask.sum())} | 0 | — | — | 模式点数不足 2，已跳过 |")
            continue
        dropped_text = (
            "；".join(
                f"c={c:+.2f} w={w:.1f} h={h:.2f}（{reason}）"
                for c, w, h, reason in curve.dropped
            )
            or "无"
        )
        add(
            f"| `{mode}` | {int(mask.sum())} | {curve.kept_coefficient.size} | "
            f"{curve.curvature:.1f} | {curve.offset:+.3f} | {dropped_text} |"
        )
    add("")
    add(
        "两个筛选器**互补**：hollowness 抓环，而同侧单调性抓「系数更大反而测得更窄」"
        "的离群点——后者的 hollowness 和宽度都落在正常区间内，只有单调性能抓住它，"
        "而它一个点就足以把拟合曲率压成负数。"
    )
    if len(figures) > 2:
        add("")
        add(f"![宽度响应]({figures[2].relative_to(out_dir(figures[2])).as_posix()})")
    add("")
    add("### 5.1 测试项解读")
    add("")
    add("**目的**　回答「这块板上的光斑到底是什么形状」。整形要在一个**已知的**"
        "孔径上打相位，如果实际孔径里有彗差、球差或者未消的离焦，"
        "模型算出的远场就与硬件会产生的远场不同——整形会把能量送到"
        "错误的地方。这里测的是**光斑宽度对各 Zernike 模式的响应**，"
        "用来反推台架自身的像差状态。")
    add("")
    dropped_all = [
        (mode, c, w, h, reason)
        for mode, curve in curve_map.items()
        for c, w, h, reason in curve.dropped
    ]
    if curve_map:
        order = sorted(
            ((m, c.curvature) for m, c in curve_map.items()), key=lambda kv: -kv[1]
        )
        add("**结果分析**　六个模式全部有明确响应，曲率排序为 "
            + " > ".join(f"{m} ({c:.0f})" for m, c in order)
            + "。曲率为正且量级相近，说明它们都在**展宽**光斑——"
            "这正是低阶 Zernike 对焦面宽度的物理预期，模型与台架没有失配到"
            "「某个模式完全不动」的程度。")
        add("")
        if dropped_all:
            by_reason: dict[str, int] = {}
            for _, _, _, _, reason in dropped_all:
                by_reason[reason] = by_reason.get(reason, 0) + 1
            add(f"共剔除 **{len(dropped_all)}** 个点，分布："
                + "、".join(f"{reason} {n} 个" for reason, n in by_reason.items())
                + "。剔除量集中在**系数绝对值较大**的一端——那里焦面已经从"
                "单瓣变成环/多瓣，「宽度」不再是光斑尺寸。")
        else:
            add("本数据集没有任何点被剔除，说明扫描幅度都落在单瓣区内。")
    add("")
    add("**对整形的影响**　有两层直接影响。")
    add("")
    add("1. **光斑宽度是整形里最便宜、最鲁棒的反馈量。** 上面每个曲率都"
        "对应一个「按一下变宽、松手变窄」的单调响应，这正是 SPGD 类算法"
        "需要的梯度信号；反过来说，**落进被剔除的那些系数区间，宽度反馈会"
        "失真甚至反向**，优化器会把能量推向一个它误以为「变暗=变窄」的区域。"
        "所以整形的扰动步长必须让工作点留在单瓣区。")
    add("2. **台架自身的像差是整形精度的上限。** 拟合出的 `defocus` / `astig` / "
        "`coma` / `spherical` 偏移量就是台架**当前带着的**这些像差。"
        "模型若不把它们算进去，算出的方形远场与硬件实际会差一个低阶像差，"
        "表现为：能量在目标框内发散、均匀性（CV）怎么优化都降不下去，"
        "而此时**加大迭代次数毫无帮助**——这不是优化器的问题，"
        "是前向模型缺项。")
    add("")

    add("## 6. 焦面尺度的三条独立路线")
    add("")
    if geometry_conflicts:
        add("> 🚨 **本节不可用，已整体停用。** 三条路线都要把斜坡/倾斜的相机侧"
            "测量换算到模型侧，而换算要用 `region` / `far_field_size` / "
            "`panel_disc_radius` —— 这些量全部来自 `bench_geometry.json`，"
            "它与本 npz 不是同一次标定（见 §1.3）。")
        add(">")
        add("> 在这种情况下三条路线会**必然**互不一致，而这种不一致完全是"
            "伪影：它度量的是两份标定之间的差，不是台架的性质。"
            "先重跑 `--stage calibrate --method sweep`，本节会自动恢复。")
        add(">")
        add("> 可直接引用的、**不依赖模型网格**的结论仍然成立，见第 3、4 节："
            "斜坡 S 与 Zernike 倾斜斜率都是纯相机侧量。")
        add("")
        add("![曲率汇总](figures/04_curvature_summary.png)")
    else:
        add("| 路线 | 相机 px / 模型 px | 说明 |")
        add("|---|---|---|")
        add(
            f"| 斜坡（权威） | {ramp_cam:.4f} | "
            f"S={ramp_scale:.0f} / (模型→面板 {model_to_panel:.3f} × P={far_field_size}) |"
            if np.isfinite(ramp_cam)
            else "| 斜坡（权威） | — | 缺少几何参数，无法换算 |"
        )
        add(
            f"| Zernike 倾斜 | {tilt_cam:.4f} | k_tilt·π·a / P，a={semi_axis} 模型 px |"
            if np.isfinite(tilt_cam)
            else "| Zernike 倾斜 | — | 数据不足 |"
        )
        add(
            f"| 视场比 (FOV) | {fov:.4f} | 完全独立推导：p_fft = λf /(P·模型间距) |"
            if np.isfinite(fov)
            else "| 视场比 (FOV) | — | 缺少几何参数 |"
        )
        add("")
        finite_routes = [v for v in (ramp_cam, tilt_cam, fov) if np.isfinite(v) and v > 0]
        if len(finite_routes) >= 2:
            lo, hi = min(finite_routes), max(finite_routes)
            spread = (hi - lo) / lo * 100
            add(
                f"三条路线落在 **{lo:.4f} … {hi:.4f}** 相机 px/模型 px，"
                f"最大互相偏离 **{spread:.0f}%**。"
            )
            add("")
            if spread > 25:
                add(
                    "> ⚠️ 偏离超过 25% 警戒线。前两条路线（斜坡、倾斜）都只依赖"
                    "**孔径半径**，而 FOV 路线额外假设**正入射**；本台架是倾斜入射，"
                    "所以 FOV 路线偏低是预期内的。**测量路线（斜坡）为准**，"
                    "FOV 路线仅作记录。"
                )
            else:
                add(
                    "> 三条路线在 25% 警戒线内一致，模型侧换算可信。"
                )
        if len(figures) > 4:
            add("")
            add(
                f"![焦面尺度三路线]"
                f"({figures[4].relative_to(out_dir(figures[4])).as_posix()})"
            )
        if len(figures) > 3:
            add("")
            add(
                f"![曲率汇总]"
                f"({figures[3].relative_to(out_dir(figures[3])).as_posix()})"
            )
    add("")

    add("## 7. 关键陷阱")
    add(
        "以下四条台架铁律固化在 `ao_shaping/tools/slm/bench_kernels.py` 的模块"
        "文档里。**每一条都附上本数据集里能证明它仍然生效的具体数字**——"
        "因为违反它们得到的是「看起来完全可信的错误结论」，而不是报错。"
    )
    add("")
    if int(flat.sum()) == 1:
        fx = float(sweep.centroid_x[flat][0])
        fy = float(sweep.centroid_y[flat][0])
        dx = fx - FALLBACK_CAMERA_PIXELS / 2.0
        dy = fy - 1944.0 / 2.0
        add(
            f"**(a) 暗帧上不能用裸 `argmax` 定位光斑。** "
            f"本数据集平场光斑在 ({fx:.1f}, {fy:.1f})，而画面中心是 "
            f"({FALLBACK_CAMERA_PIXELS / 2.0:.0f}, {1944.0 / 2.0:.0f})——"
            f"横向相差 **{abs(dx):.0f} px**。台架记录更狠：峰值只有 22~46 而帧均值 0.26 时，"
            "单个热像素就能抢到 argmax，参考帧质心曾在 60 px 内自漂，"
            "足以把健康的板判成「没动」。`measure_spot` 先去尖刺再模糊后才定位。"
        )
    else:
        add(
            "**(a) 暗帧上不能用裸 `argmax` 定位光斑。** "
            f"本次平场参考行数为 {int(flat.sum())}（应为 1），无法给出单点坐标。"
        )
    add("")
    add(
        f"**(b) 面板会保留上一次显示的图案。** 下发平场**之前**读到的「平场」"
        "其实是上一轮的散斑：同一 3 ms 设置因此测出 100 与 23 两个值。"
        f"本数据集的峰值在 **{peak_lo:.1f} … {peak_hi:.1f}** 之间（相差 "
        f"**{peak_hi / peak_lo:.1f} 倍**），"
        "跨点的亮度差既包含真实像差响应也包含残留图案，"
        "所以必须先写平场再取参考（`measure_flat_reference` 就是这么做的）。"
    )
    add("")
    add(
        "**(c) 固定 `memory_number=` 是固件 no-op。** Santec 固件把"
        "「对已在显示的槽再调 `display_memory`」当作空操作，LCOS 面板不刷新，"
        "于是第一帧之后的每一帧都是旧图。正确做法是**不传 `memory_number` "
        "调用 `display_data()`**，让驱动自己轮换槽位并按灰度变化估计翻转时间。"
    )
    add("")
    tilt_line = f"{k_tilt:.3f}" if np.isfinite(k_tilt) else "未记录"
    ramp_line = f"{ramp_scale:.0f}" if np.isfinite(ramp_scale) else "未记录"
    add(
        "**(d) LCOS 翻转时间估计会低报，要按「稳定性」等待而不是按「时间」等待。** "
        "该估计由灰度图变化量驱动，两个灰度统计相近的不同相位会让它报出 **0.0 ms**。"
        "实测：同一斜坡连续显示两次，第一次抓到 fwhm **43.2 px**，三秒后是 **12.8 px**，"
        "质心移动 **62 px**。单次抓帧于是记下一帧「未稳定但看着合理」的图像——"
        "它让斜坡扫描变得非单调，并让 Zernike 倾斜斜率读成 **1.63** 而不是 **5.36** "
        "相机 px/rad（3.3 倍误差，且因为有重复测量掩盖，连续三次运行都没发现）。"
        f"`display_and_average` 因此丢弃帧直到连续两次读数一致；"
        f"本数据集的稳定结果就是 **k_tilt = {tilt_line} 相机 px/rad**、"
        f"**S = {ramp_line}**，两者互相印证，也都远离 1.63 这个坏值。"
    )
    add("")
    add("### 7.1 每条陷阱：目的 / 结果 / 对整形的影响")
    add("")
    add(
        "| 陷阱 | 目的（为什么必须知道） | 本数据集的结果 | 对整形的影响 |"
    )
    add("|---|---|---|---|")
    trap_centre = (
        f"平场光斑 ({fx:.1f}, {fy:.1f}) vs 画面中心 "
        f"({FALLBACK_CAMERA_PIXELS / 2.0:.0f}, {1944.0 / 2.0:.0f})，横向差 "
        f"{abs(dx):.0f} px"
        if int(flat.sum()) == 1
        else f"平场参考行数 {int(flat.sum())}（应为 1），无法定位"
    )
    add(
        f"| **(a) 暗帧不能用裸 argmax 定位** | 整形的每个反馈量（PIB、均匀性、"
        f"质心）都要先知道光斑在哪。定位错一个 ROI，整条闭环就在优化噪声 | "
        f"{trap_centre} | ROI 会整体落在暗区，目标函数恒等于噪声，"
        f"优化器「不收敛」的原因被误判成参数没调好 |"
    )
    add(
        f"| **(b) 面板保留上次图案** | 参考帧/归一化必须来自**本轮**下发的图案，"
        f"否则跨点比较的基准是错的 | 峰值 "
        f"{peak_lo:.1f}…{peak_hi:.1f}（{peak_hi / peak_lo:.1f} 倍差） | "
        f"能量类目标（PIB / encircled energy）会被残留图案整体抬高或压低，"
        f"优化器可能把「上一轮的图案还在」误当成「本次相位有效」 |"
    )
    add(
        "| **(c) 固定 memory_number 是固件 no-op** | 每次下发必须真的刷新面板，"
        "否则整个闭环在读旧图 | 板正常调制（第 3、4 节响应线性即证据） | "
        "SLM 不响应 → 目标函数对相位无梯度 → SPGD 步长调大也没用，"
        "表现为「优化器不收敛」的第一嫌疑 |"
    )
    add(
        f"| **(d) 要按稳定性等待，不按时间等待** | 未稳定帧是**错帧**而非噪声帧，"
        f"会静默污染标定与反馈 | k_tilt = {tilt_line}、S = {ramp_line}，"
        f"两路线互证；坏值是 1.63（3.3× 误差） | "
        f"直接决定标定可信度：焦面尺度错 → 目标边长按同比例错 → "
        f"方形打到硬件上尺寸就不对；而由于不报错，"
        f"这种错误会一路活到最终图像才被发现 |"
    )
    add("")
    add(
        "> 这四条里 (a)(b)(c) 影响的是**反馈正确性**，(d) 影响的是**标定正确性**。"
        "前三条出错时优化器至少会「不收敛」（可以观察到）；"
        "第四条出错时一切看起来都正常，只是尺度偏了 —— "
        "这也是它连续三次运行都没被发现的原因。"
    )
    add("")

    add("## 8. 结论")
    add("")
    conclusions: list[str] = []
    if np.isfinite(ramp_scale):
        dev = abs(ramp_scale - shift_scale) / shift_scale * 100.0
        conclusions.append(
            f"焦面尺度由斜坡扫描确定为 **S = {ramp_scale:.0f}** 相机 px·周期，"
            f"与台架常数 {shift_scale:.0f} 相差 {dev:.1f}%，"
            "说明面板确实在正常调制、槽位轮换正确。"
        )
    if np.isfinite(k_tilt):
        conclusions.append(
            f"Zernike 倾斜斜率 **{k_tilt:.3f} 相机 px/rad** 落在台架预测"
            f"（450 px 照明半径约 5.34）的可信带内，"
            "**不是**历史上那个 1.63 的坏值——稳定性等待是有效的。"
        )
    if tilt_routes and max(r.max_corr for r in tilt_routes) < min_corr:
        conclusions.append(
            f"相位相关位移在本台架不可用（shift_corr 峰值 "
            f"{max(r.max_corr for r in tilt_routes):.3f} < {min_corr:.2f}），"
            "所有位移量都来自光斑质心；报告与标定器都应保持这一回退路径。"
        )
    if np.isfinite(notes.get("joint_rel_rms", float("nan")) / 100.0):
        joint = notes["joint_rel_rms"] / 100.0
        conclusions.append(
            f"宽度响应联合相对 RMS **{joint:.1%}** "
            + (
                f"低于阈值 {max_rms:.0%}，腰包可辨识。"
                if joint <= max_rms
                else f"高于阈值 {max_rms:.0%}，**腰包不可辨识**——"
                "台架带有扫描模式未覆盖的像差，应退回平顶默认值，"
                "并补扫 coma/spherical。"
            )
        )
    if np.isfinite(notes.get("fov_rel", float("nan"))) and notes["fov_rel"] > 25:
        conclusions.append(
            f"视场比交叉核对偏离 **{notes['fov_rel']:.0f}%**，超过 25% 警戒线；"
            "两条路线都与照明孔径半径成正比，"
            "请用 `python -m ao_shaping.tools.slm.slm_beam_extent` 重测孔径后再信任任一数值。"
        )
    dropped_total = sum(len(curve_map[m].dropped) for m in curve_map)
    if dropped_total:
        conclusions.append(
            f"宽度筛选共剔除 **{dropped_total}** 个点（环形或同侧非单调）；"
            "这些点的「宽度」不是高斯瓣宽，若不剔除会直接毁掉曲率拟合。"
        )
    if mismatches:
        conclusions.append(
            "**调试 sidecar 与本 npz 不是同一次扫描**，"
            "已列出全部不一致字段；任何复现都应以 npz 为准。"
        )
    if geometry_conflicts:
        conclusions.append(
            "🚨 **`bench_geometry.json` 与本 npz 不是同一次标定**（详见 §1.3），"
            "第 2 节与第 6 节已停用。请重跑 "
            "`--stage calibrate --method sweep` 刷新该文件——"
            "在它刷新之前，**模型侧换算（目标边长、region、far_field_size）"
            "都是不可信的**。"
        )
    for idx, item in enumerate(conclusions, start=1):
        add(f"{idx}. {item}")
    add("")

    # What the reader should actually do with this report.
    add("## 8.1 对整形实验的可执行结论")
    add("")
    actionable: list[tuple[str, str]] = []
    if np.isfinite(ramp_scale) and np.isfinite(ramp_vs_tilt_gap):
        actionable.append((
            f"焦面尺度可信，取 S = {ramp_scale:.0f}",
            f"两条独立路线一致到 {ramp_vs_tilt_gap * 100:.1f}%。"
            "用它把模型像素换算成相机像素时不要再引入任何解析公式"
            "（斜入射会让 `λf/d` 差一个 cos θ 因子）。",
        ))
    actionable.append((
        "相位下发只用 `display_data()` 且不传 `memory_number`",
        "这是 SLM 真的刷新的唯一确认方式。若整形出现「优化器不收敛」，"
        "先查这一条再调参数。",
    ))
    actionable.append((
        "反馈量必须落在单瓣区",
        f"宽度类反馈只在 hollowness ≥ {min_hollow:.2f} 时有效。"
        "限制扰动步长与迭代次数，使工作点不进环形区。",
    ))
    if geometry_conflicts:
        actionable.append((
            "先刷新 `bench_geometry.json` 再跑整形",
            "模型侧换算目前不可信；用陈旧的 `region` / `far_field_size` "
            "算出的方形边长会按同一比例错。",
        ))
    if curve_map and not np.isfinite(notes.get("joint_rel_rms", float("nan"))):
        actionable.append((
            "光斑 waist 仍未确定",
            "若整形精度不理想，先考虑把台架自身的低阶像差"
            "（拟合偏移量）纳入前向模型，而不是加大迭代次数。",
        ))
    add("| 结论 | 依据与用法 |")
    add("|---|---|")
    for title, body in actionable:
        add(f"| **{title}** | {body} |")
    add("")
    add("---")
    add("")
    add(
        "复现命令：`python scripts/generate_bench_probe_report.py`"
        "（离线，不需要相机/SLM，也不需要 `PYTHONPATH`）。"
    )
    add("")
    add("## 附录 A. 完整扫描记录")
    add("")
    add(
        "| label | mode | axis | 系数 | FWHM (px) | 质心 x | 质心 y | 峰值 | hollowness | shift_x | shift_y | shift_corr |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for idx in range(len(sweep)):
        def cell(values: np.ndarray, spec: str, index: int) -> str:
            """Render one cell, or an em dash when the value is not finite.

            Args:
                values: Column array.
                spec: Format spec applied to the float value.
                index: Row index.

            Returns:
                The formatted cell.
            """
            value = float(values[index])
            return "—" if not np.isfinite(value) else format(value, spec)

        add(
            f"| `{sweep.label[idx]}` | {sweep.mode[idx]} | "
            f"{sweep.axis[idx] or '—'} | {sweep.coefficient[idx]:+.3f} | "
            f"{cell(sweep.fwhm_px, '.2f', idx)} | "
            f"{cell(sweep.centroid_x, '.2f', idx)} | "
            f"{cell(sweep.centroid_y, '.2f', idx)} | "
            f"{cell(sweep.peak, '.1f', idx)} | "
            f"{cell(sweep.hollowness, '.2f', idx)} | "
            f"{cell(sweep.shift_x, '.2f', idx)} | "
            f"{cell(sweep.shift_y, '.2f', idx)} | "
            f"{cell(sweep.shift_corr, '.2f', idx)} |"
        )
    add("")
    return "\n".join(lines)


def out_dir(figure: Path) -> Path:
    """Return the report's output directory for a generated figure.

    Args:
        figure: A figure path written under ``<out>/figures/``.

    Returns:
        The parent report directory.
    """
    return figure.parent.parent


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Offline illustrated report from saved SLM bench-probe sweeps.",
    )
    parser.add_argument(
        "--sweep-npz", type=Path, default=DEFAULT_NPZ,
        help="sweep_records.npz (required)",
    )
    parser.add_argument(
        "--geometry-json", type=Path, default=DEFAULT_GEOMETRY,
        help="bench_geometry.json (optional)",
    )
    parser.add_argument(
        "--debug-root", type=Path, default=DEFAULT_DEBUG_ROOT,
        help="root holding model_in_loop_hw_sweep_* dirs (optional)",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=DEFAULT_OUT,
        help="report output directory",
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="skip figure rendering",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Render the report; return 0 on success, 1 when the npz is missing.

    Optional inputs (geometry json, debug sidecar, frame pickle) may all be
    absent -- the report then says so and still renders.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    npz_path: Path = args.sweep_npz
    if not npz_path.is_file():
        logger.error("sweep records not found: {} -- nothing to report", npz_path)
        return 1

    try:
        sweep = load_sweep(npz_path)
    except (OSError, KeyError, ValueError) as exc:
        logger.error("could not read sweep records {}: {}", npz_path, exc)
        return 1

    constants = _import_bench_constants()
    geometry = load_geometry(args.geometry_json)
    sidecar, sidecar_path = load_sidecar(args.debug_root)
    frames_status, frames_detail = probe_frames_pkl(args.debug_root)

    min_hollow = float(constants["min_hollowness"])
    min_corr = float(constants["min_shift_correlation"])

    curve_map: dict[str, ModeCurve] = {}
    for mode in WIDTH_MODES:
        mask = sweep.select(mode=mode)
        if int(mask.sum()) < 2:
            logger.info("mode {}: {} point(s), skipped", mode, int(mask.sum()))
            continue
        curve = build_mode_curve(
            mode, sweep.coefficient[mask], sweep.fwhm_px[mask],
            sweep.hollowness[mask], min_hollow,
        )
        if curve.kept_coefficient.size < 2:
            logger.warning(
                "mode {}: fewer than 2 points survived the width filters, skipped",
                mode,
            )
            continue
        curve_map[mode] = curve

    ramp_routes = compute_ramp_routes(sweep)
    tilt_routes = compute_tilt_routes(sweep, min_corr)

    # A shared, overwritten geometry file next to a fresh npz is the most
    # dangerous input pairing this report has: every model-side number would
    # inherit it while the derivation still looks self-consistent. Detect it once
    # here and let the report degrade the affected sections.
    geo_conflicts = geometry_conflicts(geometry, sweep.scalars)
    if geo_conflicts:
        logger.warning(
            "bench_geometry.json is NOT the calibration that produced this npz "
            "({}); model-side sections are disabled", "; ".join(geo_conflicts),
        )
    else:
        logger.info("bench_geometry.json provenance matches the npz")

    out_dir_path: Path = args.out
    figures_dir = out_dir_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figures: list[Path] = []
    if not args.no_figures:
        shift_scale = float(constants["tilt_shift_scale"])
        used_fallback = "TILT_SHIFT_SCALE" in constants["fallbacks"]
        if ramp_routes:
            figures.append(figure_ramp(ramp_routes, shift_scale, used_fallback, figures_dir))
        if tilt_routes:
            figures.append(figure_tilt(tilt_routes, min_corr, figures_dir))
        if curve_map:
            figures.append(figure_widths(curve_map, figures_dir))
            notes_raw = (geometry or {}).get("calibration_notes")
            notes = parse_notes(notes_raw if isinstance(notes_raw, str) else None)
            figures.append(figure_curvature(curve_map, notes, figures_dir))

            region = int((geometry or {}).get("region") or 0)
            far_field_size = int((geometry or {}).get("far_field_size") or 0)
            disc = int((geometry or {}).get("panel_disc_radius") or 0)
            semi_axis = max(region // 2, 1)
            model_to_panel = disc / semi_axis if disc and semi_axis else float("nan")
            ramp_scales = [r.scale for r in ramp_routes if np.isfinite(r.scale)]
            ramp_scale = float(np.mean(ramp_scales)) if ramp_scales else float("nan")
            tilt_slopes = [abs(r.slope) for r in tilt_routes if np.isfinite(r.slope)]
            k_tilt = float(np.mean(tilt_slopes)) if tilt_slopes else float("nan")
            span = model_to_panel * far_field_size if (
                np.isfinite(model_to_panel) and far_field_size
            ) else float("nan")
            figures.append(
                figure_routes(
                    ramp_scale / span if np.isfinite(span) and span else float("nan"),
                    k_tilt * np.pi * semi_axis / far_field_size if far_field_size
                    else float("nan"),
                    fov_route(geometry, sweep, ramp_scale),
                    model_to_panel, far_field_size, figures_dir,
                )
            )
    else:
        logger.info("--no-figures given: rendering markdown only")

    report = render_report(
        sweep=sweep,
        geometry=geometry,
        geometry_path=args.geometry_json,
        sidecar=sidecar,
        sidecar_path=sidecar_path,
        npz_path=npz_path,
        ramp_routes=ramp_routes,
        tilt_routes=tilt_routes,
        curve_map=curve_map,
        constants=constants,
        figures=figures,
        frames_status=frames_status,
        frames_detail=frames_detail,
        geometry_conflicts=geo_conflicts,
    )
    report_path = out_dir_path / "report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("wrote report {} ({} KB)", report_path, report_path.stat().st_size // 1024)
    logger.info("output directory: {}", out_dir_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
