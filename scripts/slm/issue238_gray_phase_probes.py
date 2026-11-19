"""Issue #238: raw Mono12 gray -> phase probes (offline fit) + panel crosstalk.

This script captures a *model-free* dataset on the bench (Santec SLM-200 #1
@1064 nm + Daheng MER2-507-23GM) so that the true gray -> phase response and the
panel's spatial crosstalk can be fitted **offline**.  No WFS, no interferometer,
no correction application, no frozen model checkpoint is involved.

What is acquired (all frames are the FULL native 1944x2592 Mono12 frame, stored
as uint16 npz; nothing is cropped):

* an exposure bracket (>= 4 steps) on the flat / zero-order (brightest) pattern,
  auto-scaled so the measured peak stays <= 0.9 x 4095;
* a blazed-grating gray sweep ``g = 0, 16, ..., 1008, 1023`` (65 points), period
  16 px along panel x, nominal depth ``phi = 2*pi*g/G_ref`` with ``G_ref = 1023``;
* a flat (all-zero gray) reference frame;
* checkerboard period 16 px at g = 480 / 496 / 992 / 1023 (20 frames each) and
  one g = 993 check frame (matches the 2026-11-18b convention);
* 3x3 and 5x5 center-cell crosstalk grids (center g = 993 grating, neighbors 0
  or 993), one frame per condition.

``G_ref = 1023`` is the *encoding* reference used to build the patterns.  The
true physical 2*pi gray is deliberately **pending offline fit**: issue #238
historically assumed 993 while the 2026-11-18 ``optical_path_record.json``
reports ``two_pi_grayscale = 1023``.  Both values are recorded in the summary;
no runtime decision depends on either one.

Device discipline (docs/dev/hardware_discipline.md):

* WITHOUT ``--execute`` the script NEVER imports a camera/SLM driver and NEVER
  opens a device.  It only prints the deterministic plan as JSON to stdout and
  does NOT create ``--out``.
* WITH ``--execute`` it acquires ``DeviceLock.default()`` first (waiting up to
  ``--wait-until-idle`` seconds), then creates ``--out`` with
  ``exist_ok=False``.  Any failure path releases the lock.  ``finally`` resets
  the SLM to flat gray 0, closes both devices, releases the lock and writes
  ``release.json`` + ``device_log.json``.
* Mono12 is a hard requirement: the effective bit depth read back after open
  must be 12, the frame dtype must be uint16 and the effective full scale must
  be 4095.  A mismatch raises immediately -- there is no silent downgrade.

Usage::

    python scripts/slm/issue238_gray_phase_probes.py --out <fresh-dir>            # plan only
    python scripts/slm/issue238_gray_phase_probes.py --execute --out <fresh-dir>  # hardware
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

#: SLM-200 panel default; overwritten by the driver's actual Panel_Res at execute time.
DEFAULT_PANEL_SHAPE = (1200, 1920)
#: Native, never-windowed Daheng frame shape.
NATIVE_CCD_SHAPE = (1944, 2592)
#: 10-bit panel code space; 1024 wraps to 0 on this device.
MAX_GRAYSCALE = 1023
#: Encoding reference for the nominal blaze depth (NOT the physical 2*pi gray).
G_REF = 1023
#: Historical (issue #238) assumption for the physical 2*pi gray.
TWO_PI_GRAY_ASSUMED = 993
#: Monotonic JSON-safe note carried into plan / summary.
TWO_PI_GRAY_NOTE = (
    "true 2*pi gray is PENDING OFFLINE FIT; G_ref=1023 is only the encoding "
    f"reference used to build patterns; issue #238 historically assumed "
    f"{TWO_PI_GRAY_ASSUMED} while the 2026-11-18 optical_path_record.json "
    "reports two_pi_grayscale=1023"
)
#: Default exposure bracket, milliseconds (brightest / flat pattern).
DEFAULT_EXPOSURE_BRACKET = (0.02, 0.04, 0.08, 0.16)
#: Peak target for the auto-scaled bracket: 0.9 x 4095.
TARGET_PEAK_FRACTION = 0.9
#: A frame at/above this fraction of full scale counts as saturated.
SATURATION_FRACTION = 0.9
#: Preferred peak ceiling for the *chosen* exposure.  Leaves headroom for
#: laser warm-up / hot pixels between the bracket and the real capture; without
#: it the bracket picks the step closest to saturation (0.9) and the very next
#: flat frame can tip 6 pixels over the line (seen 2026-11-19: 0.16 ms bracket
#: peak 3519 -> flat 3712, saturated).
PREFERRED_PEAK_FRACTION = 0.7
#: Default blaze / checkerboard period (px).
DEFAULT_PERIOD_PX = 16
#: Default gray sweep step.
DEFAULT_SWEEP_STEP = 16
#: Checkerboard repeat count (2026-11-18b convention).
CHECKER_FRAMES = 20
#: Checkerboard gray levels.
CHECKER_GRAYS = (480, 496, 992, 1023)
CHECKER_CHECK_GRAY = 993
#: Crosstalk grid sizes and center gray.
CROSSTALK_GRIDS = (3, 5)
CROSSTALK_CENTER_GRAY = 993
CROSSTALK_NEIGHBOR_GRAYS = (0, 993)
CROSSTALK_CELL_PX = 128
CROSSTALK_GAP_PX = 64
#: Device-lock polling interval (seconds).
LOCK_POLL_S = 1.0


# ---------------------------------------------------------------------------
# Pure pattern helpers (no hardware, no I/O)
# ---------------------------------------------------------------------------
def gray_sweep(step: int = DEFAULT_SWEEP_STEP) -> tuple[int, ...]:
    """Return the sweep levels ``0, step, ..., 1023`` with 1023 always included.

    For the default ``step=16`` this is ``range(0, 1024, 16)`` (64 levels) plus
    the endpoint 1023 -> exactly 65 points.  The endpoint is deduplicated for
    steps that already divide 1023.
    """
    step = int(step)
    if step <= 0:
        raise ValueError("--sweep-step must be positive")
    levels = list(range(0, MAX_GRAYSCALE + 1, step))
    if MAX_GRAYSCALE not in levels:
        levels.append(MAX_GRAYSCALE)
    return tuple(int(v) for v in levels)


def binary_sweep_levels(
    coarse_step: int = 32,
    dense_lo: int = 864,
    dense_hi: int = MAX_GRAYSCALE,
    dense_step: int = 8,
) -> tuple[int, ...]:
    """Union of a coarse full sweep and a dense band near the expected 2*pi gray.

    The coarse grid (default step 32) pins the rising branch and hence ``gamma``;
    the dense band (default 864..1023 step 8) brackets the second ``eta_1`` zero
    (``phi = 2*pi``) so the physical 2*pi gray is localised to a few gray levels.
    Returns a sorted, de-duplicated tuple that always includes 0 and 1023.
    """
    coarse_step = int(coarse_step)
    dense_step = int(dense_step)
    if coarse_step <= 0 or dense_step <= 0:
        raise ValueError("sweep steps must be positive")
    levels: set[int] = set(range(0, MAX_GRAYSCALE + 1, coarse_step))
    lo = max(0, int(dense_lo))
    hi = min(MAX_GRAYSCALE, int(dense_hi))
    if hi >= lo:
        levels.update(range(lo, hi + 1, dense_step))
    levels.add(MAX_GRAYSCALE)
    return tuple(sorted(levels))


def choose_exposure(
    usable: list[dict[str, Any]], full_scale: float
) -> dict[str, Any]:
    """Pick the brightest unsaturated bracket row under the headroom ceiling.

    ``usable`` rows are the bracket steps that measured zero saturated pixels.
    Among them we prefer the brightest whose peak stays at or below
    ``PREFERRED_PEAK_FRACTION * full_scale`` so a small laser warm-up between
    the bracket and the real capture cannot tip the flat frame over the
    saturation line.  If every usable row is already above that ceiling (dim
    scene scaling, or a bracket that only has bright steps) we fall back to the
    brightest usable row so the run still proceeds at its best SNR.
    """
    if not usable:
        raise ValueError("choose_exposure requires at least one usable row")
    preferred = PREFERRED_PEAK_FRACTION * float(full_scale)
    within = [r for r in usable if float(r["peak"]) <= preferred]
    return max(within if within else usable, key=lambda r: r["peak"])


def _clamp_gray(array: np.ndarray) -> tuple[np.ndarray, str | None]:
    """Hard-clamp panel grays to 1023; return the array and an optional warning."""
    arr = np.asarray(array)
    if arr.size and int(arr.max()) > MAX_GRAYSCALE:
        warning = (
            f"pattern pixels reached {int(arr.max())} > {MAX_GRAYSCALE}; "
            f"clamped to {MAX_GRAYSCALE}"
        )
        return np.clip(arr, 0, MAX_GRAYSCALE).astype(np.uint16), warning
    return arr.astype(np.uint16), None


def flat_gray(shape: tuple[int, int], gray: int) -> np.ndarray:
    """Uniform gray panel image."""
    return np.full((int(shape[0]), int(shape[1])), int(gray), dtype=np.uint16)


def blaze_gray(
    shape: tuple[int, int],
    *,
    gray: int,
    period_px: int,
    axis: int = 1,
) -> np.ndarray:
    """Sawtooth blaze with peak gray ``g`` and nominal depth ``2*pi*g/G_ref``.

    The ramp covers ``0..g`` inclusively within each period so ``array.max() ==
    g`` for in-range levels.  ``axis=1`` ramps along panel x (columns).
    """
    period = int(period_px)
    if period < 2:
        raise ValueError("blaze period_px must be >= 2")
    ramp = np.rint(
        float(gray) * np.arange(period, dtype=np.float64) / float(period - 1)
    )
    if axis == 1:
        line = np.tile(ramp, int(np.ceil(shape[1] / period)))[: shape[1]]
        array = np.broadcast_to(line, (int(shape[0]), int(shape[1]))).copy()
    elif axis == 0:
        line = np.tile(ramp, int(np.ceil(shape[0] / period)))[: shape[0]]
        array = np.broadcast_to(line[:, None], (int(shape[0]), int(shape[1]))).copy()
    else:
        raise ValueError("axis must be 0 or 1")
    clamped, _ = _clamp_gray(array)
    return clamped


def checkerboard_gray(
    shape: tuple[int, int], *, gray: int, period_px: int
) -> np.ndarray:
    """Checkerboard of ``0`` / ``gray`` blocks; full period is ``period_px``.

    ``period_px=16`` -> 8 px cells (2026-11-18b convention).
    """
    cell = int(period_px) // 2
    if cell < 1:
        raise ValueError("checkerboard period_px must be >= 2")
    rows = np.arange(int(shape[0])) // cell
    cols = np.arange(int(shape[1])) // cell
    parity = (rows[:, None] + cols[None, :]) % 2
    array = (parity * int(gray)).astype(np.uint16)
    clamped, _ = _clamp_gray(array)
    return clamped


def crosstalk_gray(
    shape: tuple[int, int],
    *,
    gray: int,
    neighbor_gray: int,
    grid_n: int,
    cell_px: int = CROSSTALK_CELL_PX,
    gap_px: int = CROSSTALK_GAP_PX,
    period_px: int = DEFAULT_PERIOD_PX,
) -> np.ndarray:
    """Centered ``grid_n x grid_n`` cell grid for panel crosstalk diagnosis.

    The center cell holds a ``g=993`` blaze grating; the other cells are flat
    ``neighbor_gray`` (0 or 993).  Everything outside the grid is 0.
    """
    grid_n = int(grid_n)
    if grid_n % 2 != 1 or grid_n < 3:
        raise ValueError("grid_n must be an odd integer >= 3")
    cell, gap = int(cell_px), int(gap_px)
    side = grid_n * cell + (grid_n - 1) * gap
    h, w = int(shape[0]), int(shape[1])
    if side > min(h, w):
        raise ValueError(f"crosstalk grid side {side} exceeds panel {shape}")
    array = np.zeros((h, w), dtype=np.uint16)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    k = grid_n // 2
    center_patch = blaze_gray((cell, cell), gray=gray, period_px=period_px, axis=1)
    for row in range(grid_n):
        for col in range(grid_n):
            py = y0 + row * (cell + gap)
            px = x0 + col * (cell + gap)
            patch = center_patch if (row == k and col == k) else np.full(
                (cell, cell), int(neighbor_gray), dtype=np.uint16
            )
            array[py : py + cell, px : px + cell] = patch
    clamped, _ = _clamp_gray(array)
    return clamped


def binary_gray(
    shape: tuple[int, int],
    *,
    gray: int,
    period_px: int,
    duty: float = 0.5,
    axis: int = 1,
) -> np.ndarray:
    """Binary phase grating: a two-level square wave between gray ``0`` and ``gray``.

    Unlike the sawtooth ``blaze_gray`` (asymmetric -> only +1 order), this is a
    *symmetric* two-phase grating.  For phase depth ``phi = phi(gray)`` its far
    field obeys the exact Fourier result, independent of the within-period ramp
    shape:

        eta_0 = d^2 + (1-d)^2 + 2 d (1-d) cos(phi)
        eta_1(+/-) = (4/pi^2) sin^2(phi/2) sin^2(pi d)     (equal +1 and -1)

    For ``duty d = 0.5`` this reduces to ``eta_0 = cos^2(phi/2)`` and
    ``eta_1 = (4/pi^2) sin^2(phi/2)``.  Measuring ``eta_0`` and ``eta_1`` together
    therefore recovers the *absolute* phase ``phi(g)`` point by point, which is the
    model-free handle that pins the physical 2*pi gray (issue #238) and breaks the
    ``gamma``-vs-2pi-gray degeneracy of the blaze fit.

    ``duty`` is the fraction of each period held at ``gray`` (0.5 = square wave).
    ``axis=1`` varies along panel x (columns).
    """
    period = int(period_px)
    if period < 2:
        raise ValueError("binary period_px must be >= 2")
    duty = float(duty)
    if not (0.0 < duty < 1.0):
        raise ValueError("binary duty must be in (0, 1)")
    hi = int(min(max(int(gray), 0), MAX_GRAYSCALE))
    cut = int(round(period * duty))
    cell = np.zeros(period, dtype=np.uint16)
    cell[:cut] = hi
    if axis == 1:
        line = np.tile(cell, int(np.ceil(shape[1] / period)))[: shape[1]]
        array = np.broadcast_to(line, (int(shape[0]), int(shape[1]))).copy()
    elif axis == 0:
        line = np.tile(cell, int(np.ceil(shape[0] / period)))[: shape[0]]
        array = np.broadcast_to(line[:, None], (int(shape[0]), int(shape[1]))).copy()
    else:
        raise ValueError("binary axis must be 0 or 1")
    return array.astype(np.uint16)


def pattern_array(item: dict, shape: tuple[int, int]) -> np.ndarray:
    """Render a plan item into raw uint16 panel grays."""
    kind = item["kind"]
    if kind == "flat":
        return flat_gray(shape, item["gray"])
    if kind == "blaze":
        return blaze_gray(
            shape,
            gray=item["gray"],
            period_px=item["period_px"],
            axis=1,
        )
    if kind == "checkerboard":
        return checkerboard_gray(
            shape, gray=item["gray"], period_px=item["period_px"]
        )
    if kind == "crosstalk":
        return crosstalk_gray(
            shape,
            gray=item["gray"],
            neighbor_gray=item["neighbor_gray"],
            grid_n=item["grid_n"],
            cell_px=item.get("cell_px", CROSSTALK_CELL_PX),
            gap_px=item.get("gap_px", CROSSTALK_GAP_PX),
            period_px=item.get("period_px", DEFAULT_PERIOD_PX),
        )
    if kind == "binary":
        return binary_gray(
            shape,
            gray=item["gray"],
            period_px=item["period_px"],
            duty=item.get("duty", 0.5),
            axis=int(item.get("axis", 1)),
        )
    raise ValueError(f"unknown pattern kind {kind!r}")


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Deterministic plan (pure)
# ---------------------------------------------------------------------------
def _materialize_add(patterns: list[dict[str, Any]]):
    """Return an ``add(**kwargs)`` closure that appends a finalised pattern item."""

    def add(**kwargs: Any) -> None:
        index = len(patterns)
        item = dict(index=index, label=f"{index:04d}_{kwargs['kind']}", **kwargs)
        array = pattern_array(item, DEFAULT_PANEL_SHAPE)
        requested = max(
            int(item.get("gray", 0)),
            int(item.get("neighbor_gray", 0)),
        )
        item["panel_gray_max"] = int(array.max())
        item["clamp_warning"] = (
            f"requested gray {requested} > {MAX_GRAYSCALE}"
            if requested > MAX_GRAYSCALE
            else None
        )
        item["panel_gray_sha256"] = _digest(array)
        item.setdefault("warnings", [])
        if item["clamp_warning"]:
            item["warnings"].append(item["clamp_warning"])
        patterns.append(item)

    return add


def _build_binary_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Focused binary phase-grating plan (breaks the gamma<->2*pi-gray degeneracy).

    Only a flat reference, an exposure bracket, the binary sweep (multi-frame
    averaged per level) and a post-sweep drift monitor.  Reuses the same capture
    harness as the blaze plan via the mode-agnostic ``_run``.
    """
    period = int(getattr(args, "period_px", DEFAULT_PERIOD_PX))
    if period < 2:
        raise ValueError("period_px >= 2 required")
    frames_per_level = int(getattr(args, "frames_per_level", 16))
    if frames_per_level < 1:
        raise ValueError("frames_per_level >= 1 required")
    dense_step = int(getattr(args, "dense_step", 8))
    dense_lo = int(getattr(args, "dense_lo", 864))
    dense_hi = int(getattr(args, "dense_hi", MAX_GRAYSCALE))
    levels = binary_sweep_levels(
        coarse_step=32,
        dense_lo=dense_lo,
        dense_hi=dense_hi,
        dense_step=dense_step,
    )

    patterns: list[dict[str, Any]] = []
    add = _materialize_add(patterns)

    # Flat reference + bracket (flat / zero order is the brightest pattern).
    add(kind="flat", group="reference", gray=0, frames=1, params={})
    add(
        kind="flat",
        group="bracket",
        gray=0,
        frames=len(args.exposure_bracket),
        params={"note": "flat/zero-order = brightest pattern for exposure bracket"},
    )

    # Binary phase grating sweep (symmetric 0/g square wave), averaged per level.
    for g in levels:
        add(
            kind="binary",
            group="sweep",
            gray=int(g),
            period_px=period,
            duty=0.5,
            axis=1,
            frames=frames_per_level,
            params={"note": "binary 0/g grating, 50% duty, x-period", "duty": 0.5},
        )

    # Post-sweep flat monitor: detects beam drift over the run.
    add(
        kind="flat",
        group="drift",
        gray=0,
        frames=4,
        params={"note": "post-sweep drift monitor (compare flat power to reference)"},
    )

    n_frames = sum(int(p["frames"]) for p in patterns)
    bracket = tuple(float(v) for v in args.exposure_bracket)
    return {
        "schema": "issue238-gray-phase-probes/v2-binary",
        "mode": "binary",
        "task": "binary phase-grating sweep to pin physical 2*pi gray (offline fit)",
        "g_ref": G_REF,
        "two_pi_gray_assumed_issue238": TWO_PI_GRAY_ASSUMED,
        "two_pi_gray_true": "pending offline fit (binary grating)",
        "binary_period_px": period,
        "binary_duty": 0.5,
        "binary_sweep": list(levels),
        "n_sweep": len(levels),
        "frames_per_level": frames_per_level,
        "dense_region": {"lo": dense_lo, "hi": dense_hi, "step": dense_step},
        "exposure_bracket_ms": list(bracket),
        "n_patterns": len(patterns),
        "n_frames_expected": n_frames,
        "native_ccd_shape": list(NATIVE_CCD_SHAPE),
        "panel_shape_default": list(DEFAULT_PANEL_SHAPE),
        "bit_depth_required": 12,
        "full_scale_required": 4095,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "patterns": patterns,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build the full acquisition plan; never touches devices or the filesystem."""
    if getattr(args, "mode", "blaze") == "binary":
        return _build_binary_plan(args)
    sweep = gray_sweep(args.sweep_step)
    period = int(getattr(args, "period_px", DEFAULT_PERIOD_PX))
    checker_frames = int(getattr(args, "checker_frames", CHECKER_FRAMES))
    if period < 2 or checker_frames < 1:
        raise ValueError("period_px >= 2 and checker_frames >= 1 required")

    patterns: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        index = len(patterns)
        item = dict(index=index, label=f"{index:04d}_{kwargs['kind']}", **kwargs)
        array = pattern_array(item, DEFAULT_PANEL_SHAPE)
        requested = max(
            int(item.get("gray", 0)),
            int(item.get("neighbor_gray", 0)),
        )
        item["panel_gray_max"] = int(array.max())
        item["clamp_warning"] = (
            f"requested gray {requested} > {MAX_GRAYSCALE}"
            if requested > MAX_GRAYSCALE
            else None
        )
        item["panel_gray_sha256"] = _digest(array)
        item.setdefault("warnings", [])
        if item["clamp_warning"]:
            item["warnings"].append(item["clamp_warning"])
        patterns.append(item)

    # Flat reference + bracket pattern (flat / zero order is the brightest).
    add(kind="flat", group="reference", gray=0, frames=1, params={})
    add(
        kind="flat",
        group="bracket",
        gray=0,
        frames=len(args.exposure_bracket),
        params={"note": "flat/zero-order = brightest pattern for exposure bracket"},
    )

    # Gray sweep: blazed grating, period 16 px, direction x.
    for g in sweep:
        add(
            kind="blaze",
            group="sweep",
            gray=int(g),
            period_px=period,
            direction="x",
            frames=1,
            params={
                "nominal_phase_depth_rad": 2.0 * np.pi * float(g) / float(G_REF),
                "g_ref": G_REF,
            },
        )

    # Static-component diagnosis: checkerboard repeats.
    for g in CHECKER_GRAYS:
        add(
            kind="checkerboard",
            group="checker",
            gray=int(g),
            period_px=period,
            direction="xy",
            frames=checker_frames,
            params={"cell_px": period // 2, "convention": "2026-11-18b"},
        )
    add(
        kind="checkerboard",
        group="checker_check",
        gray=CHECKER_CHECK_GRAY,
        period_px=period,
        direction="xy",
        frames=1,
        params={"cell_px": period // 2, "note": "g=993 check frame"},
    )

    # Crosstalk: 3x3 and 5x5, center 993 grating, neighbors 0 or 993.
    for grid_n in CROSSTALK_GRIDS:
        for neighbor in CROSSTALK_NEIGHBOR_GRAYS:
            add(
                kind="crosstalk",
                group="crosstalk",
                gray=CROSSTALK_CENTER_GRAY,
                neighbor_gray=int(neighbor),
                grid_n=int(grid_n),
                cell_px=CROSSTALK_CELL_PX,
                gap_px=CROSSTALK_GAP_PX,
                period_px=period,
                frames=1,
                params={
                    "center": f"blaze g={CROSSTALK_CENTER_GRAY}",
                    "neighbors": f"flat gray={neighbor}",
                },
            )

    n_frames = sum(int(p["frames"]) for p in patterns)
    bracket = tuple(float(v) for v in args.exposure_bracket)
    return {
        "schema": "issue238-gray-phase-probes/v1",
        "task": "raw Mono12 SLM gray->phase + panel crosstalk probes (offline fit)",
        "g_ref": G_REF,
        "g_ref_note": TWO_PI_GRAY_NOTE,
        "two_pi_gray_assumed_issue238": TWO_PI_GRAY_ASSUMED,
        "two_pi_gray_true": "pending offline fit",
        "gray_sweep": list(sweep),
        "n_sweep": len(sweep),
        "sweep_step": int(args.sweep_step),
        "blaze_period_px": period,
        "blaze_direction": "x",
        "exposure_bracket_ms": list(bracket),
        "checker_frames_per_level": checker_frames,
        "checker_grays": list(CHECKER_GRAYS) + [CHECKER_CHECK_GRAY],
        "crosstalk_grids": list(CROSSTALK_GRIDS),
        "crosstalk_center_gray": CROSSTALK_CENTER_GRAY,
        "crosstalk_neighbor_grays": list(CROSSTALK_NEIGHBOR_GRAYS),
        "n_patterns": len(patterns),
        "n_frames_expected": n_frames,
        "native_ccd_shape": list(NATIVE_CCD_SHAPE),
        "panel_shape_default": list(DEFAULT_PANEL_SHAPE),
        "bit_depth_required": 12,
        "full_scale_required": 4095,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "patterns": patterns,
    }


# ---------------------------------------------------------------------------
# Small helpers shared by the hardware path
# ---------------------------------------------------------------------------
def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def clock() -> dict:
    return dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        monotonic_ns=time.monotonic_ns(),
    )


def acquire_device_lock(wait_s: float) -> Any | None:
    """Acquire ``DeviceLock.default()``, polling for up to ``wait_s`` seconds.

    This is an explicit loop on ``acquire`` (never an immediate ``timeout=0``
    failure when the operator asked for a wait).  The occupied holder is printed
    to stderr for the bench operator.  Returns ``None`` when the lock stays busy.
    """
    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    lock = DeviceLock.default()
    deadline = time.monotonic() + max(0.0, float(wait_s))
    reported = False
    while True:
        if lock.acquire(timeout=0.0):
            return lock
        if not reported:
            holder = lock.holder()
            print(
                f"bench device lock occupied by {holder if holder else 'unknown'}; "
                f"waiting up to {wait_s:.1f}s",
                file=sys.stderr,
            )
            reported = True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(LOCK_POLL_S, remaining))


def _read_best_effort(cam: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        value = getattr(cam, name, None)
        if value is not None:
            return value
    sdk_cam = getattr(cam, "cam", None)
    for name in names:
        value = getattr(sdk_cam, name, None) if sdk_cam is not None else None
        if value is not None:
            return value
    return None


def _stats(frame: np.ndarray, full_scale: float) -> dict[str, Any]:
    threshold = SATURATION_FRACTION * full_scale
    return {
        "peak": float(frame.max()),
        "saturated_pixels": int((frame >= threshold).sum()),
        "saturation_threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# Hardware execution
# ---------------------------------------------------------------------------
def _run(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    """Execute the acquisition.  Device imports live here, never at module scope."""
    t_start_monotonic = time.monotonic()
    t_start_utc = datetime.now(timezone.utc).isoformat(timespec="microseconds")

    # 1) Lock FIRST, so an existing --out never reaches any driver.
    lock = acquire_device_lock(args.wait_until_idle)
    if lock is None:
        print("ERROR: bench device lock stayed busy; refusing to start", file=sys.stderr)
        return 2

    out = args.out
    release: dict[str, Any] = {
        "slm_flat_gray0": "not-attempted",
        "slm_closed": "not-attempted",
        "cam_closed": "not-attempted",
        "lock_released": "not-attempted",
    }
    device_log: dict[str, Any] = {
        "acquired_at_utc": t_start_utc,
        "acquired_monotonic_ns": time.monotonic_ns(),
        "out": str(out),
        "occupied_devices": ["slm", "camera"],
        "exposure_table": [],
        "events": [],
    }
    created_out = False
    slm: Any = None
    cam: Any = None
    frames_jsonl: Any = None
    frame_records: list[dict[str, Any]] = []
    bit_depth_check: dict[str, Any] = {"status": "not-attempted"}
    exposure_rows: list[dict[str, Any]] = []
    chosen_exposure_ms: float | None = None
    device_two_pi_gray: int | None = None
    error: str | None = None

    try:
        # 2) Create --out inside try (exist_ok=False); FileExistsError -> return 2.
        out.mkdir(parents=True, exist_ok=False)
        created_out = True
        (out / "frames").mkdir(parents=True, exist_ok=False)
        write_json(out / "plan.json", plan)

        # 3) ONLY now are drivers imported / devices opened.
        from ao_shaping.drivers.ccd.common import (
            create_camera,
            get_camera_exposure_ms,
            get_camera_exposure_range,
            set_camera_exposure_ms,
        )
        from ao_shaping.drivers.slm.factory import require_drive_board_entry
        from ao_shaping.drivers.slm.santec import Santec
        from ao_shaping.tools.slm.transfer.hardware import prepare_camera_sdk
        from ao_shaping.utils.io.optical_path_record import build_optical_path_record

        prepare_camera_sdk("daheng")
        slm = Santec(slm_number=args.slm_number, video_mode=0)
        slm.open()
        require_drive_board_entry(slm)
        panel_shape = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        print(f"SLM #{args.slm_number} open (drive board OK), panel={panel_shape}")

        wavelength_nm, device_two_pi_gray = slm.get_wavelength_info()
        print(
            f"SLM wavelength readback {wavelength_nm} nm, "
            f"physical 2pi gray {device_two_pi_gray} "
            "(G_ref=1023 remains the encoding reference; true 2pi pending offline fit)"
        )
        slm.set_phase_overlay(None)

        cam = create_camera(
            "daheng",
            args.cam_id,
            exposure_time_ms=float(args.exposure_bracket[0]),
            bit_depth=12,
        )
        cam.open()
        set_camera_exposure_ms(cam, float(args.exposure_bracket[0]))

        # Mono12 / full-scale assertions -- fail immediately, never downgrade.
        bit_depth_readback = cam.bit_depth
        bit_depth_check = {
            "status": "ok",
            "requested": 12,
            "readback": bit_depth_readback,
        }
        if bit_depth_readback != 12:
            bit_depth_check["status"] = "failed"
            raise RuntimeError(
                f"Mono12 bit depth readback is {bit_depth_readback!r}, expected 12"
            )
        cam_serial = _read_best_effort(cam, ("DeviceSerialNumber", "serial_number"))

        frames_jsonl = (out / "frames.jsonl").open("w", encoding="utf-8")

        def save_frame(meta: dict[str, Any], frame: np.ndarray) -> dict[str, Any]:
            if frame.shape != NATIVE_CCD_SHAPE or frame.dtype != np.uint16:
                raise RuntimeError(
                    f"expected uint16 frame {NATIVE_CCD_SHAPE}, got "
                    f"{frame.dtype} {frame.shape}"
                )
            full_scale = float(cam.effective_full_scale(frame))
            if full_scale != float(plan["full_scale_required"]):
                raise RuntimeError(
                    f"Mono12 full scale readback {full_scale}, expected 4095"
                )
            stamp = clock()
            row = dict(meta)
            row.update(
                frame_file=f"frames/{meta['index']:04d}_{meta['label']}.npz",
                frame_shape=list(NATIVE_CCD_SHAPE),
                frame_dtype=str(frame.dtype),
                full_scale=full_scale,
                **stamp,
            )
            row.update(_stats(frame, full_scale))
            np.savez_compressed(
                out / row["frame_file"],
                frame=frame,
                kind=np.array(row["kind"]),
                gray=np.array(int(row.get("gray", 0))),
                period_px=np.array(float(row.get("period_px", 0))),
                exposure_ms=np.array(float(row["exposure_ms"])),
                timestamp_utc=np.array(stamp["timestamp_utc"]),
                meta_json=np.array(json.dumps(row, ensure_ascii=False)),
            )
            frames_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            frames_jsonl.flush()
            frame_records.append(row)
            return row

        def capture_one(meta: dict[str, Any]) -> dict[str, Any]:
            time.sleep(float(args.settle_s))
            raw = cam.get_numpy_image(1, True)
            frame = np.asarray(raw, dtype=np.uint16)
            return save_frame(meta, frame)

        def set_exposure(ms: float) -> float:
            lo, hi = get_camera_exposure_range(cam)
            clamped = float(np.clip(float(ms), float(lo), float(hi)))
            set_camera_exposure_ms(cam, clamped)
            return float(get_camera_exposure_ms(cam))

        def display(item: dict[str, Any]) -> np.ndarray:
            gray_array = pattern_array(item, panel_shape)
            slm.display_data(gray_array)
            return gray_array

        # ---- exposure bracket on the brightest (flat / zero-order) pattern ----
        bracket_item = next(p for p in plan["patterns"] if p["group"] == "bracket")
        display(bracket_item)
        for step_index, requested in enumerate(args.exposure_bracket):
            readback = set_exposure(float(requested))
            row = capture_one(
                {
                    "index": len(frame_records),
                    "label": f"bracket_{step_index:02d}",
                    "kind": "flat",
                    "group": "bracket",
                    "gray": 0,
                    "period_px": 0,
                    "exposure_ms": readback,
                    "requested_exposure_ms": float(requested),
                }
            )
            exposure_rows.append(
                {
                    "pass": 1,
                    "requested_ms": float(requested),
                    "readback_ms": readback,
                    "peak": row["peak"],
                    "saturated_pixels": row["saturated_pixels"],
                    "usable": row["saturated_pixels"] == 0,
                }
            )

        usable = [r for r in exposure_rows if r["usable"]]
        if not usable:
            # Pass 2: rescale from the dimmest measured step.
            dimmest = min(exposure_rows, key=lambda r: r["readback_ms"])
            target = TARGET_PEAK_FRACTION * float(plan["full_scale_required"])
            scale = min(0.5, target / max(float(dimmest["peak"]), 1.0))
            base = float(dimmest["readback_ms"])
            for factor in (1.0, 0.25):
                requested = base * scale * factor
                readback = set_exposure(requested)
                row = capture_one(
                    {
                        "index": len(frame_records),
                        "label": f"bracket_pass2_{factor:g}",
                        "kind": "flat",
                        "group": "bracket",
                        "gray": 0,
                        "period_px": 0,
                        "exposure_ms": readback,
                        "requested_exposure_ms": requested,
                    }
                )
                exposure_rows.append(
                    {
                        "pass": 2,
                        "requested_ms": requested,
                        "readback_ms": readback,
                        "peak": row["peak"],
                        "saturated_pixels": row["saturated_pixels"],
                        "usable": row["saturated_pixels"] == 0,
                    }
                )
            usable = [r for r in exposure_rows if r["usable"]]
        if not usable:
            raise RuntimeError(
                "no unsaturated exposure after two bracket passes; "
                "reduce illumination / adjust --exposure-bracket"
            )
        preferred = PREFERRED_PEAK_FRACTION * float(plan["full_scale_required"])
        chosen = choose_exposure(usable, plan["full_scale_required"])
        chosen_exposure_ms = float(chosen["readback_ms"])
        print(
            f"exposure chosen: {chosen_exposure_ms:.5f} ms "
            f"(peak {chosen['peak']:.0f}, saturated {chosen['saturated_pixels']}, "
            f"preferred ceiling {preferred:.0f})"
        )

        # ---- full plan at the chosen exposure ----
        set_exposure(chosen_exposure_ms)
        for item in plan["patterns"]:
            if item["group"] == "bracket":
                continue
            gray_array = display(item)
            for repeat in range(int(item["frames"])):
                row = capture_one(
                    {
                        "index": len(frame_records),
                        "label": f"{item['label']}_r{repeat:02d}",
                        "kind": item["kind"],
                        "group": item["group"],
                        "gray": item["gray"],
                        "period_px": item.get("period_px", 0),
                        "neighbor_gray": item.get("neighbor_gray"),
                        "grid_n": item.get("grid_n"),
                        "pattern_params": item.get("params", {}),
                        "panel_gray_max": int(gray_array.max()),
                        "panel_gray_sha256": item["panel_gray_sha256"],
                        "warnings": item.get("warnings", []),
                        "exposure_ms": chosen_exposure_ms,
                    }
                )
                if row["saturated_pixels"]:
                    raise RuntimeError(
                        f"{row['label']} saturated {row['saturated_pixels']} pixels "
                        f"at chosen exposure {chosen_exposure_ms} ms"
                    )

        # ---- summaries -----------------------------------------------------
        exposure_stats = {
            "n_passes": sorted({r["pass"] for r in exposure_rows}),
            "n_steps": len(exposure_rows),
            "n_usable": len(usable),
            "chosen_readback_ms": chosen_exposure_ms,
            "rows": exposure_rows,
            "analysis_policy": "only unsaturated bracket steps are usable for offline fit",
        }
        mode = plan.get("mode", "blaze")
        summary = {
            "schema": plan["schema"],
            "mode": mode,
            "status": "ok",
            "frame_count": len(frame_records),
            "n_patterns": plan["n_patterns"],
            "exposure_bracket": exposure_stats,
            "bit_depth_assertion": bit_depth_check,
            "g_ref": G_REF,
            "two_pi_gray_assumed_issue238": TWO_PI_GRAY_ASSUMED,
            "device_two_pi_gray_readback": device_two_pi_gray,
            "two_pi_gray_true": plan.get("two_pi_gray_true", "pending offline fit"),
            "two_pi_gray_note": TWO_PI_GRAY_NOTE,
            "saturated_frames": sum(1 for r in frame_records if r["saturated_pixels"]),
            "clamp_warnings": [
                {"label": r["label"], "warnings": r["warnings"]}
                for r in frame_records
                if r.get("warnings")
            ],
            "elapsed_s": time.monotonic() - t_start_monotonic,
            **clock(),
        }
        write_json(out / "summary.json", summary)

        record = build_optical_path_record(
            run_id=f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_issue238_{mode}",
            date=f"{datetime.now(timezone.utc):%Y-%m-%d}",
            operator_note=(
                f"Issue #238 ({mode}): raw Mono12 SLM probes; no WFS/interferometer; "
                "G_ref=1023 (true 2pi pending offline fit)"
            ),
            devices={
                "slm": {
                    "model": "Santec SLM-200",
                    "slm_number": args.slm_number,
                    "wavelength_nm": wavelength_nm,
                    "two_pi_grayscale": device_two_pi_gray,
                    "panel_res": list(panel_shape),
                    "open_mode": "memory",
                },
                "camera": {
                    "model": "Daheng MER2-507-23GM",
                    "serial": cam_serial,
                    "bit_depth": bit_depth_readback,
                    "full_scale": 4095,
                    "exposure_used_ms": chosen_exposure_ms,
                    "averaging_frames": 1,
                    "discard_frames_after_exposure_change": 3,
                },
            },
            optical_path={
                "probe_status": "raw-gray probes (offline fit pending)",
            },
            exposure_check={
                "peak_counts_at_used_exposure": chosen["peak"],
                "saturation_threshold_counts": 0.9 * 4095.0,
                "saturated": False,
            },
            occupancy={
                "acquired_at": t_start_utc,
                "closed_normally": True,
            },
        )
        write_json(out / "optical_path_record.json", record)
        device_log["exposure_table"] = exposure_rows
        print(f"issue238: {len(frame_records)} frames -> {out}")
        return 0

    except FileExistsError:
        error = f"--out already exists: {out}"
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    # Bench scripts catch broadly on purpose: report the error, then always release.
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        # 4) Always release everything, on every exception path.
        if slm is not None:
            try:
                slm.set_grayscale(0)
                release["slm_flat_gray0"] = "ok"
            except Exception as exc:
                release["slm_flat_gray0"] = f"failed: {exc}"
            try:
                slm.close()
                release["slm_closed"] = "ok"
            except Exception as exc:
                release["slm_closed"] = f"failed: {exc}"
        if cam is not None:
            try:
                cam.close()
                release["cam_closed"] = "ok"
            except Exception as exc:
                release["cam_closed"] = f"failed: {exc}"
        if frames_jsonl is not None:
            try:
                frames_jsonl.close()
            except Exception:
                pass
        try:
            lock.release()
            release["lock_released"] = "ok"
        except Exception as exc:
            release["lock_released"] = f"failed: {exc}"

        release["released_at_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="microseconds"
        )
        release["elapsed_s"] = time.monotonic() - t_start_monotonic
        release["error"] = error
        device_log["released_at_utc"] = release["released_at_utc"]
        device_log["release"] = release
        device_log["error"] = error
        print(
            f"released: SLM gray0={release['slm_flat_gray0']}, "
            f"SLM close={release['slm_closed']}, cam close={release['cam_closed']}, "
            f"lock={release['lock_released']}, elapsed={release['elapsed_s']:.1f}s"
        )
        if created_out:
            try:
                write_json(out / "release.json", release)
                write_json(out / "device_log.json", device_log)
            except Exception as exc:
                print(f"could not write release/device_log: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Issue #238: capture raw full-frame Mono12 SLM gray->phase probes "
            "and panel crosstalk (offline fit; no WFS/interferometer). "
            "Without --execute only prints the plan; no devices, no --out."
        )
    )
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    parser.add_argument(
        "--exposure-bracket",
        type=float,
        nargs="+",
        default=list(DEFAULT_EXPOSURE_BRACKET),
        metavar="MS",
        help="exposure bracket in ms (>= 4 steps; default: 0.02 0.04 0.08 0.16)",
    )
    parser.add_argument(
        "--sweep-step",
        type=int,
        default=DEFAULT_SWEEP_STEP,
        help="gray sweep step (default 16 -> 65 points 0..1023)",
    )
    parser.add_argument(
        "--mode",
        choices=("blaze", "binary"),
        default="blaze",
        help=(
            "blaze = v1 sawtooth sweep + crosstalk (default); "
            "binary = symmetric binary phase-grating sweep to pin the 2*pi gray"
        ),
    )
    parser.add_argument(
        "--frames-per-level",
        type=int,
        default=16,
        help="[binary] frames averaged per gray level (default 16)",
    )
    parser.add_argument(
        "--dense-lo",
        type=int,
        default=864,
        help="[binary] dense-band low gray (default 864)",
    )
    parser.add_argument(
        "--dense-hi",
        type=int,
        default=MAX_GRAYSCALE,
        help="[binary] dense-band high gray (default 1023)",
    )
    parser.add_argument(
        "--dense-step",
        type=int,
        default=8,
        help="[binary] dense-band gray step (default 8)",
    )
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=600.0,
        help="seconds to wait for the bench device lock (default 600)",
    )
    parser.add_argument(
        "--execute", action="store_true", help="open hardware and acquire"
    )
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument(
        "--period-px",
        type=int,
        default=DEFAULT_PERIOD_PX,
        help="blaze / checkerboard period in panel px (default 16)",
    )
    parser.add_argument(
        "--checker-frames",
        type=int,
        default=CHECKER_FRAMES,
        help="frames per checkerboard gray level (default 20)",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.3,
        help="settle time after each pattern write (default 0.3)",
    )
    args = parser.parse_args(argv)

    if len(args.exposure_bracket) < 4:
        parser.error("--exposure-bracket requires at least 4 steps")
    if any(v <= 0 for v in args.exposure_bracket):
        parser.error("--exposure-bracket values must be positive")
    if args.sweep_step <= 0 or args.period_px < 2 or args.checker_frames < 1:
        parser.error("--sweep-step>0, --period-px>=2, --checker-frames>=1 required")
    if args.wait_until_idle < 0 or args.settle_s < 0:
        parser.error("--wait-until-idle>=0 and --settle-s>=0 required")

    try:
        plan = build_plan(args)
    except ValueError as exc:
        parser.error(f"invalid plan parameters: {exc}")

    if not args.execute:
        # Plan mode: print JSON only.  Neither drivers nor --out are touched.
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        print("skip: requires --execute", file=sys.stderr)
        return 0

    return _run(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
