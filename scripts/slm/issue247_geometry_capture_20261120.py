"""Issue #247: geometry-capture probes for panel<->CCD scale, chirality and tilt sign.

Why this script exists
----------------------
The measured panel->CCD rotation +88.544 deg (SD 0.088 deg) is reproducible but
lives in a 180-deg-symmetric family (a global sign flip gives -91.457 deg); the
branch is currently pinned only by a convention string.  ``slm_geometry_calib.npz``
also has no panel/CCD pixel pitch and no focal magnification, so no physical
mapping can be frozen from existing data.

This script captures a *descriptive*, model-free dataset that gives a future
offline fit the inputs it needs:

* binary 0/pi gratings along both panel axes at periods 128 and 256 px -> panel
  period projected to CCD gives per-axis scale (and a 2-period consistency check);
* a **blazed spot ladder** along each panel axis -> breaks the 180-deg ambiguity.
  The CCD sits at the Fourier plane, so a target's panel *position* only adds an
  invisible linear phase; chirality must therefore live in the far-field
  *intensity* arrangement.  Two/three 600x600 blaze patches with deliberately
  unequal ramps (2*pi per 96 px and 2*pi per 32 px, i.e. orders +1 and +3 of the
  96 px fundamental) all on one side give the ordered far-field ladder
  ``[0, +1, +3]``.  Its rot180 image is ``[-3, -1, 0]`` (mirrored to the other
  side of the zero order), so one capture decides the branch.  Mirror and
  rotation stay separable because every ladder spot lies on a single axis.  A
  plain symmetric grating pair (+/-1 orders, both sides) is kept as the scale
  reference;
* +/- tilt along both panel axes measured in the same session -> re-checks tilt
  sign / handedness relative to whichever branch is later chosen.  The tilt is a
  genuine prismatic ramp (``2*pi`` every ``--tilt-period-px``, default 128 px as
  in ``issue247_capture_tilt_20261117.py``), not a small phase amplitude: a
  0.5 rad peak phase would deflect by < 1 CCD pixel and be unmeasurable.
* a **full-panel 2D diagonal blaze** -> a *single* off-axis far-field spot at
  ``(fy, fx) = (sign_y/period_y_px, sign_x/period_x_px)``.  One frame therefore
  encodes both panel axes at once, which is the decisive test for a 2x2
  panel->CCD map: a physically impossible (non-invertible) map must leave a
  non-trivial residual.  Two period pairs (128/192 and 192/128) are used because
  sign flips alone are linearly dependent (flipping both signs gives ``-M v``),
  so a degenerate design set could hide an inconsistent map.

Nothing here fits a rotation, applies a threshold, or writes a calibration
constant.  Every measurement written to ``summary.json`` is explicitly
**descriptive only**.  Pitch / magnification values that cannot be resolved from
device drivers are written as ``null`` with an explicit ``missing`` reason; they
are never hard-coded or guessed.  ``data/slm_geometry_calib.npz`` is not read or
modified.

Device discipline (docs/dev/hardware_discipline.md, docs/dev/hw_plan_consolidated_20261118.md):

* WITHOUT ``--execute`` (the default) the script only builds and prints the
  deterministic plan as JSON.  It creates no directory, imports no camera/SLM
  driver and touches no device.  ``--self-test`` likewise needs no device.
* WITH ``--execute`` it acquires ``DeviceLock.default()`` first (waiting up to
  ``--wait-until-idle`` seconds), creates one fresh ``--out`` with
  ``exist_ok=False``, opens SLM and CCD exactly once, and uses ``finally`` so
  ``cam.close -> slm.close -> lock.release`` always run.  A fixed exposure is
  used for the whole run; any saturating frame aborts with a recorded reason and
  a normal release.

Usage::

    python scripts/slm/issue247_geometry_capture_20261120.py --self-test
    python scripts/slm/issue247_geometry_capture_20261120.py --out <fresh-dir>            # plan only
    python scripts/slm/issue247_geometry_capture_20261120.py --execute --out <fresh-dir>  # hardware
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
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Declared constants (no device needed).  Physical pitches are NEVER hardcoded.
# ---------------------------------------------------------------------------
PLAN_SCHEMA = "issue247-geometry-capture/v2"
SUMMARY_SCHEMA = "issue247-geometry-capture-summary/v1"
RECORD_SCHEMA = "ao-optical-path-record/v1"

DEFAULT_PANEL_SHAPE = (1200, 1920)  # (rows, cols); overwritten by Panel_Res at execute
NATIVE_CCD_SHAPE = (1944, 2592)  # full native Daheng frame, never windowed
G_REF = 1023  # rad->gray ENCODING reference (NOT the physical 2*pi gray)
TWO_PI = 2.0 * np.pi
PI_GRAY = int(np.rint(G_REF / 2.0))  # 0/pi binary grating -> grays {0, 512}

GRATING_PERIODS_PX: tuple[int, ...] = (128, 256)
GRATING_FRAMES = 4
FLAT_FRAMES = 4  # >= 2

# ---------------------------------------------------------------------------
# Tilt: a genuine prismatic ramp, NOT a small phase amplitude.
# ---------------------------------------------------------------------------
#: Slope: one full 2*pi wrap every ``DEFAULT_TILT_PERIOD_PX`` panel pixels,
#: matching ``scripts/slm/issue247_capture_tilt_20261117.py`` (default 128 px).
DEFAULT_TILT_PERIOD_PX = 128.0
TILT_FRAMES = 4
#: The axis span must contain at least this many 2*pi fringes, otherwise the
#: deflected spot sits too close to the zero order to separate reliably.
TILT_MIN_FRINGES = 10.0
#: A tilt that cannot move the spot by at least this many CCD pixels under the
#: most pessimistic plausible geometry must fail ``--self-test``.
TILT_MIN_CCD_PX = 5.0
#: Documented, auditable pessimistic (pitch, focal, camera pixel) triple used by
#: the feasibility check.  Chosen as the *worst* plausible pair for this bench:
#: the largest SLM pitch and shortest focal length we would accept, with the
#: camera's fixed pixel size (Daheng MER2-507, 3.45 um).  These are bounds for a
#: pass/fail check only; they are never written as measured constants.
TILT_PESSIMISTIC_PITCH_UM = 20.0
TILT_PESSIMISTIC_FOCAL_MM = 100.0
TILT_PESSIMISTIC_CCD_PIXEL_UM = 3.45
WAVELENGTH_NM_DECLARED = 1064.0

# ---------------------------------------------------------------------------
# 180-deg branch resolver: blazed spot ladder in the Fourier-plane intensity.
# ---------------------------------------------------------------------------
LADDER_FRAMES = 4
LADDER_PATCH_SIZE_PX = 600
#: Fundamental deflection period; a patch with this ramp lands at order +1.
LADDER_BASE_PERIOD_PX = 96.0
#: Patch ramps as integer orders of the fundamental (all on the + side).
#: ``{order: period_px}`` with ``period = base // order`` gives unequal
#: deflection amplitudes: 96 px -> +1, 32 px -> +3.
LADDER_ORDERS: tuple[int, ...] = (1, 3)
#: Top-left (row, col) panel origins of the 600x600 blaze patches.
LADDER_PATCH_ORIGINS: tuple[tuple[int, int], ...] = (
    (300, 240),
    (300, 1080),
)

# ---------------------------------------------------------------------------
# 2D diagonal blaze: one full-panel linear ramp -> a SINGLE off-axis far-field
# spot, so one frame encodes both panel axes at once.
# ---------------------------------------------------------------------------
#: Reuse the grating frame count: same pattern family, same per-pattern budget.
BLAZE2D_FRAMES = GRATING_FRAMES
#: Frozen probe designs.  The two period pairs (128/192 and 192/128) are both
#: needed: sign flips alone are linearly dependent -- flipping both signs gives
#: exactly ``-M v``, i.e. the same axis directions -- so only designs whose
#: periods differ per axis give two independent directions, and only then can a
#: least-squares fit of the four predicted frequencies leave a non-trivial
#: residual (an inconsistent 2x2 map cannot hide inside a degenerate design set).
BLAZE2D_DESIGNS: tuple[dict[str, float | int], ...] = (
    {"period_x_px": 128.0, "period_y_px": 192.0, "sign_x": 1, "sign_y": 1},
    {"period_x_px": 128.0, "period_y_px": 192.0, "sign_x": 1, "sign_y": -1},
    {"period_x_px": 192.0, "period_y_px": 128.0, "sign_x": 1, "sign_y": 1},
    {"period_x_px": 192.0, "period_y_px": 128.0, "sign_x": 1, "sign_y": -1},
)

DEFAULT_EXPOSURE_MS = 0.05
DEFAULT_SATURATION_FRACTION = 0.9
DEFAULT_SETTLE_S = 0.8
DEFAULT_WAIT_IDLE_S = 1800.0
LOCK_POLL_S = 1.0
BIT_DEPTH_REQUIRED = 12
FULL_SCALE_REQUIRED = 4095

GEOMETRY_CALIB_REL = "data/slm_geometry_calib.npz"

PLAN_KEYS: tuple[str, ...] = (
    "schema",
    "task",
    "patterns",
    "pattern_ids",
    "n_patterns",
    "frames_per_pattern",
    "n_frames_expected",
    "declared_constants",
    "missing",
    "grating_periods_px",
    "grating_duty",
    "blaze2d_designs",
    "blaze2d_frames",
    "tilt",
    "ladder",
    "encoding_max_gray",
    "frame_digests",
    "timestamp_utc",
)
SUMMARY_KEYS: tuple[str, ...] = (
    "schema",
    "status",
    "descriptive_only",
    "note",
    "n_frames",
    "per_pattern",
    "frame_digests",
    "saturation_count",
    "declared_constants",
    "missing",
    "elapsed_s",
    "timestamp_utc",
)


class SaturationAbort(RuntimeError):
    """Raised when any captured frame saturates; aborts with a normal release."""


# ---------------------------------------------------------------------------
# Pure helpers (no hardware, no I/O)
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def clock() -> dict[str, Any]:
    return dict(timestamp_utc=utc_now(), monotonic_ns=time.monotonic_ns())


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def flat_phase(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros((int(shape[0]), int(shape[1])), dtype=np.float64)


def grating_phase(
    shape: tuple[int, int], *, period_px: int, axis: int
) -> np.ndarray:
    """Binary 0/pi grating, 50% duty, along panel x (axis=1) or y (axis=0)."""
    period = int(period_px)
    if period < 2 or period % 2:
        raise ValueError("grating period_px must be even and >= 2")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    n = int(shape[1]) if axis == 1 else int(shape[0])
    high = (np.arange(n) % period) < (period // 2)
    line = np.where(high, np.pi, 0.0)
    out = line.reshape((1, n) if axis == 1 else (n, 1))
    return np.broadcast_to(out, (int(shape[0]), int(shape[1]))).astype(np.float64)


def axis_length(shape: tuple[int, int], axis: int) -> int:
    return int(shape[1]) if axis == 1 else int(shape[0])


def effective_tilt_period(period_px: float, n_px: int) -> float:
    """Shorten the ramp period, if needed, to keep >= TILT_MIN_FRINGES fringes.

    The CLI default stays 128 px (matching the 2026-11-17 tilt capture); on the
    shorter panel axis (1200 rows) that would give only 9.375 fringes, so the
    period is reduced to ``n_px / TILT_MIN_FRINGES`` (120 px) for that axis.  The
    requested period is still recorded in the plan.
    """
    requested = float(period_px)
    if requested <= 0:
        raise ValueError("tilt period_px must be positive")
    limit = float(n_px) / TILT_MIN_FRINGES
    return min(requested, limit)


def tilt_phase(
    shape: tuple[int, int], *, period_px: float, axis: int, sign: int
) -> np.ndarray:
    """Genuine prismatic ramp, centred (zero piston), ``2*pi`` every ``period_px``.

    ``axis=1`` -> panel x (columns), ``axis=0`` -> panel y (rows).  The ramp is
    centred so it has zero piston and is exactly antisymmetric in radians:
    ``tilt(+1) == -tilt(-1)``.  Raw radians; ``phase_to_gray`` is the only wrap.
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    period = float(period_px)
    if period <= 0:
        raise ValueError("tilt period_px must be positive")
    height, width = int(shape[0]), int(shape[1])
    n = axis_length(shape, axis)
    coord = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    line = float(sign) * TWO_PI * coord / period
    out = line.reshape((1, width) if axis == 1 else (height, 1))
    return np.broadcast_to(out, (height, width)).astype(np.float64)


def tilt_geometry(
    shape: tuple[int, int], *, period_px: float, axis: int
) -> dict[str, Any]:
    """Derived, device-free numbers for one tilt axis (all recorded in the plan)."""
    n = axis_length(shape, axis)
    effective = effective_tilt_period(period_px, n)
    span_rad = TWO_PI * float(n - 1) / effective
    return {
        "axis": "x" if axis == 1 else "y",
        "axis_length_px": n,
        "requested_period_px": float(period_px),
        "period_px": effective,
        "phase_span_rad": span_rad,
        "fringe_count": float(n) / effective,
    }


def tilt_expected_ccd_px(
    *,
    period_px: float,
    pitch_um: float,
    focal_mm: float,
    ccd_pixel_um: float,
    order: float = 1.0,
) -> float:
    """Fourier-plane displacement ``order * lambda * f / (period * pitch)`` in px.

    ``lambda`` is ``WAVELENGTH_NM_DECLARED`` (the bench laser); all lengths are
    converted to metres.  Descriptive arithmetic only -- the caller decides
    whether the inputs are measured or pessimistic bounds.
    """
    lam_m = WAVELENGTH_NM_DECLARED * 1e-9
    focal_m = float(focal_mm) * 1e-3
    grating_m = float(period_px) * float(pitch_um) * 1e-6
    return float(order) * lam_m * focal_m / (grating_m * float(ccd_pixel_um) * 1e-6)


def raw_tilt_feasible(period_px: float, n_px: int) -> bool:
    """Audit check on a *declared* period: >= TILT_MIN_FRINGES and >= 5 CCD px.

    No clamping here, so an unphysically weak (long-period) design -- e.g. the
    old 0.5 rad peak amplitude, equivalent to ``period = 2*pi*n/1.0`` -- fails
    the check instead of being silently steepened.
    """
    period = float(period_px)
    if period <= 0:
        return False
    if float(n_px) / period < TILT_MIN_FRINGES:
        return False
    px = tilt_expected_ccd_px(
        period_px=period,
        pitch_um=TILT_PESSIMISTIC_PITCH_UM,
        focal_mm=TILT_PESSIMISTIC_FOCAL_MM,
        ccd_pixel_um=TILT_PESSIMISTIC_CCD_PIXEL_UM,
    )
    return px >= TILT_MIN_CCD_PX


def ladder_periods() -> tuple[tuple[int, float], ...]:
    """``(order, period_px)`` for every blaze patch: period = base / order."""
    return tuple(
        (int(order), float(LADDER_BASE_PERIOD_PX) / float(order))
        for order in LADDER_ORDERS
    )


def ladder_phase(
    shape: tuple[int, int], *, axis: int, sign: int = 1
) -> np.ndarray:
    """Sum of non-overlapping blaze patches; each patch ramps ``2*pi/period``.

    Only the phase is written (mod 2*pi by ``phase_to_gray``): the patch's panel
    position adds a linear phase that is invisible at the Fourier plane, so the
    far-field intensity is the ladder of patch orders, not the patch footprint.
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    height, width = int(shape[0]), int(shape[1])
    side = int(LADDER_PATCH_SIZE_PX)
    phase = np.zeros((height, width), dtype=np.float64)
    periods = ladder_periods()
    for (row, col), (_, period) in zip(LADDER_PATCH_ORIGINS, periods):
        if row + side > height or col + side > width:
            raise ValueError("ladder patch falls outside the panel")
        n = axis_length((side, side), axis)
        coord = np.arange(n, dtype=np.float64)
        ramp = float(sign) * TWO_PI * coord / float(period)
        block = np.broadcast_to(
            ramp.reshape((1, n) if axis == 1 else (n, 1)), (side, side)
        )
        phase[row : row + side, col : col + side] = block
    return phase


def ladder_spot_orders(axis: int, sign: int = 1) -> list[tuple[int, int]]:
    """Predicted far-field spots (dx, dy) in units of the fundamental deflection.

    Zero order plus every patch order, all on the ``sign`` side of one axis.
    """
    spots: list[tuple[int, int]] = [(0, 0)]
    for order, _period in ladder_periods():
        step = int(sign) * int(order)
        if axis == 1:
            spots.append((step, 0))
        else:
            spots.append((0, step))
    return spots


def spots_invariant_under_sign_flip(spots: list[tuple[int, int]]) -> bool:
    return sorted(spots) == sorted((-x, -y) for x, y in spots)


def spots_invariant_under_rot180(spots: list[tuple[int, int]]) -> bool:
    """rot180 maps (x, y) -> (-x, -y); same test, kept separate for auditability."""
    return spots_invariant_under_sign_flip(spots)


def blaze2d_freq(
    *, period_x_px: float, period_y_px: float, sign_x: int, sign_y: int
) -> dict[str, float]:
    """Predicted far-field spot frequency (cycles/panel-pixel) of one blaze2d."""
    return {
        "fy": float(sign_y) / float(period_y_px),
        "fx": float(sign_x) / float(period_x_px),
    }


def blaze2d_phase(
    shape: tuple[int, int],
    *,
    period_x_px: float,
    period_y_px: float,
    sign_x: int = 1,
    sign_y: int = 1,
) -> np.ndarray:
    """Full-panel 2D linear phase ramp (a 2D prism / blaze).

    The far field of ``exp(i*phi)`` under a pure linear ramp is the aperture
    diffraction pattern shifted to a **single** spot at frequency
    ``(fy, fx) = (sign_y / period_y_px, sign_x / period_x_px)`` -- no orders
    ladder, no on-axis zero order.  One capture therefore encodes both axes' sign
    *and* magnitude at once: with ``period_x != period_y`` the spot sits off-axis
    diagonally, so axis swap, sign flip and rot180 each send it somewhere else.
    Raw radians (zero piston not enforced); ``phase_to_gray`` is the only wrap.
    """
    if sign_x not in (-1, 1) or sign_y not in (-1, 1):
        raise ValueError("signs must be -1 or +1")
    if float(period_x_px) <= 0 or float(period_y_px) <= 0:
        raise ValueError("periods must be positive")
    height, width = int(shape[0]), int(shape[1])
    cols = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    ramp_x = float(sign_x) * TWO_PI * cols / float(period_x_px)
    ramp_y = float(sign_y) * TWO_PI * rows / float(period_y_px)
    return (ramp_y.reshape(height, 1) + ramp_x.reshape(1, width)).astype(
        np.float64
    )


def phase_to_gray(phase: np.ndarray, encoding_max_gray: int = G_REF) -> np.ndarray:
    """Rad -> uint16 panel gray using the declared rad->gray encoding scale.

    This is the single wrap point; pattern phases are always raw (unwrapped).
    """
    scale = int(encoding_max_gray)
    wrapped = np.mod(np.asarray(phase, dtype=np.float64), TWO_PI)
    gray = np.rint(wrapped * scale / TWO_PI)
    return np.mod(gray, scale + 1).astype(np.uint16)


def pattern_phase(item: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    kind = item["kind"]
    if kind == "flat":
        return flat_phase(shape)
    if kind == "grating":
        return grating_phase(
            shape, period_px=int(item["period_px"]), axis=int(item["axis"])
        )
    if kind == "tilt":
        return tilt_phase(
            shape,
            period_px=float(item["period_px"]),
            axis=int(item["axis"]),
            sign=int(item["sign"]),
        )
    if kind == "ladder":
        return ladder_phase(shape, axis=int(item["axis"]), sign=1)
    if kind == "blaze2d":
        return blaze2d_phase(
            shape,
            period_x_px=float(item["period_x_px"]),
            period_y_px=float(item["period_y_px"]),
            sign_x=int(item["sign_x"]),
            sign_y=int(item["sign_y"]),
        )
    raise ValueError(f"unknown pattern kind {kind!r}")


def pattern_array(item: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """Render a plan item into raw uint16 panel grays."""
    return phase_to_gray(pattern_phase(item, shape))


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def rot180(array: np.ndarray) -> np.ndarray:
    return np.rot90(array, 2)


# ---------------------------------------------------------------------------
# Deterministic plan (pure)
# ---------------------------------------------------------------------------
def build_patterns(only: list[str] | None = None) -> list[dict[str, Any]]:
    """All frozen probes, or just the ids named in ``only`` (approval scoping).

    An empty/None ``only`` keeps the full frozen list; naming any id that is not
    in the plan is a hard error, so a typo can never silently shrink a run.
    """
    patterns: list[dict[str, Any]] = []

    def add(pid: str, kind: str, frames: int, **params: Any) -> None:
        patterns.append(dict(id=pid, kind=kind, frames=int(frames), **params))

    add("flat", "flat", FLAT_FRAMES)
    for period in GRATING_PERIODS_PX:
        for axis, name in ((1, "x"), (0, "y")):
            add(
                f"grating_{name}_p{period}",
                "grating",
                GRATING_FRAMES,
                period_px=int(period),
                axis=axis,
            )
    for axis, name in ((1, "x"), (0, "y")):
        add(
            f"ladder_{name}",
            "ladder",
            LADDER_FRAMES,
            axis=axis,
            base_period_px=LADDER_BASE_PERIOD_PX,
            orders=list(LADDER_ORDERS),
            spots=[list(s) for s in ladder_spot_orders(axis)],
        )
    for axis, name in ((1, "x"), (0, "y")):
        geometry = tilt_geometry(
            DEFAULT_PANEL_SHAPE, period_px=DEFAULT_TILT_PERIOD_PX, axis=axis
        )
        for sign, sname in ((1, "plus"), (-1, "minus")):
            add(
                f"tilt_{name}_{sname}",
                "tilt",
                TILT_FRAMES,
                axis=axis,
                sign=sign,
                period_px=geometry["period_px"],
                requested_period_px=geometry["requested_period_px"],
            )
    # One full-panel 2D blaze per frozen design -> a single off-axis spot each.
    # ``expected_freq`` is NOT ``spots``: a blaze2d has exactly one far-field
    # spot, while ``spots`` carries (row, col) ladder orders and would be
    # meaningless (and silently misread) here.
    for design in BLAZE2D_DESIGNS:
        period_x = float(design["period_x_px"])
        period_y = float(design["period_y_px"])
        sign_x = int(design["sign_x"])
        sign_y = int(design["sign_y"])
        add(
            f"blaze2d_px{int(period_x)}_py{int(period_y)}"
            f"_sx{sign_x:+d}_sy{sign_y:+d}",
            "blaze2d",
            BLAZE2D_FRAMES,
            period_x_px=period_x,
            period_y_px=period_y,
            sign_x=sign_x,
            sign_y=sign_y,
            expected_freq=blaze2d_freq(
                period_x_px=period_x,
                period_y_px=period_y,
                sign_x=sign_x,
                sign_y=sign_y,
            ),
        )

    if only:
        wanted = {str(pid).strip() for pid in only if str(pid).strip()}
        unknown = sorted(wanted - {p["id"] for p in patterns})
        if unknown:
            raise SystemExit(
                f"--patterns: unknown pattern id(s) {unknown}; "
                f"known: {sorted(p['id'] for p in patterns)}"
            )
        patterns = [p for p in patterns if p["id"] in wanted]

    for item in patterns:
        item["phase_sha256"] = digest(pattern_phase(item, DEFAULT_PANEL_SHAPE))
        arr = pattern_array(item, DEFAULT_PANEL_SHAPE)
        item["panel_gray_levels"] = (
            [int(v) for v in np.unique(arr)]
            if item["kind"] in ("flat", "grating")
            else None
        )
    return patterns


def tilt_expected_ccd_formula() -> str:
    return (
        "ccd_px = order * lambda_nm*1e-9 * focal_mm*1e-3 / "
        "(period_px * pitch_um*1e-6 * ccd_pixel_um*1e-6); "
        f"lambda={WAVELENGTH_NM_DECLARED:g} nm; pitch/focal/ccd_pixel not "
        "resolved here -- see missing"
    )


def build_tilt_plan(period_px: float) -> dict[str, Any]:
    """Derived tilt numbers per axis plus the pessimistic feasibility check."""
    axes: dict[str, Any] = {}
    worst_px: float | None = None
    for axis, name in ((1, "x"), (0, "y")):
        geometry = tilt_geometry(
            DEFAULT_PANEL_SHAPE, period_px=period_px, axis=axis
        )
        pessimistic_px = tilt_expected_ccd_px(
            period_px=geometry["period_px"],
            pitch_um=TILT_PESSIMISTIC_PITCH_UM,
            focal_mm=TILT_PESSIMISTIC_FOCAL_MM,
            ccd_pixel_um=TILT_PESSIMISTIC_CCD_PIXEL_UM,
        )
        geometry["pessimistic_ccd_px"] = pessimistic_px
        geometry["feasible"] = raw_tilt_feasible(
            geometry["period_px"], geometry["axis_length_px"]
        )
        axes[name] = geometry
        worst_px = pessimistic_px if worst_px is None else min(worst_px, pessimistic_px)
    weak_span_rad = 1.0  # the rejected 0.5 rad peak / 1.0 rad total design
    weak_period_px = TWO_PI * DEFAULT_PANEL_SHAPE[1] / weak_span_rad
    return {
        "requested_period_px": float(period_px),
        "min_fringes": TILT_MIN_FRINGES,
        "min_ccd_px": TILT_MIN_CCD_PX,
        "pessimistic_geometry": {
            "pitch_um": TILT_PESSIMISTIC_PITCH_UM,
            "focal_mm": TILT_PESSIMISTIC_FOCAL_MM,
            "ccd_pixel_um": TILT_PESSIMISTIC_CCD_PIXEL_UM,
            "wavelength_nm": WAVELENGTH_NM_DECLARED,
        },
        "axes": axes,
        "feasible": bool(worst_px is not None and worst_px >= TILT_MIN_CCD_PX),
        "rejected_weak_design": {
            "phase_span_rad": weak_span_rad,
            "equivalent_period_px": weak_period_px,
            "feasible": raw_tilt_feasible(weak_period_px, DEFAULT_PANEL_SHAPE[1]),
        },
        "expected_ccd_px_formula": tilt_expected_ccd_formula(),
    }


def build_ladder_plan() -> dict[str, Any]:
    spots_x = ladder_spot_orders(1)
    spots_y = ladder_spot_orders(0)
    return {
        "base_period_px": LADDER_BASE_PERIOD_PX,
        "patch_size_px": LADDER_PATCH_SIZE_PX,
        "orders": list(LADDER_ORDERS),
        "periods_px": {int(o): p for o, p in ladder_periods()},
        "patch_origins": [list(o) for o in LADDER_PATCH_ORIGINS],
        "spots": {
            "x": [list(s) for s in spots_x],
            "y": [list(s) for s in spots_y],
        },
        "invariant_under_sign_flip": {
            "x": spots_invariant_under_sign_flip(spots_x),
            "y": spots_invariant_under_sign_flip(spots_y),
        },
        "invariant_under_rot180": {
            "x": spots_invariant_under_rot180(spots_x),
            "y": spots_invariant_under_rot180(spots_y),
        },
        "rationale": (
            "CCD at the Fourier plane: patch position adds only linear phase, so "
            "chirality must be carried by intensity.  Ramps 2*pi/96 px (+1) and "
            "2*pi/32 px (+3), all on the + side, give [0,+1,+3]; rot180 mirrors "
            "the ladder to [-3,-1,0], so one capture decides the branch.  All "
            "spots share one axis, keeping mirror and rotation separable."
        ),
        "expected_ccd_px_formula": (
            "ccd_px = order * lambda_nm*1e-9 * focal_mm*1e-3 / "
            "(period_px * pitch_um*1e-6 * ccd_pixel_um*1e-6); "
            f"lambda={WAVELENGTH_NM_DECLARED:g} nm; period_px from periods_px"
        ),
    }


def build_blaze2d_designs() -> list[dict[str, Any]]:
    """Self-describing record of the frozen 2D blaze probes (id + geometry)."""
    designs: list[dict[str, Any]] = []
    for design in BLAZE2D_DESIGNS:
        period_x = float(design["period_x_px"])
        period_y = float(design["period_y_px"])
        sign_x = int(design["sign_x"])
        sign_y = int(design["sign_y"])
        designs.append(
            {
                "id": (
                    f"blaze2d_px{int(period_x)}_py{int(period_y)}"
                    f"_sx{sign_x:+d}_sy{sign_y:+d}"
                ),
                "kind": "blaze2d",
                "frames": int(BLAZE2D_FRAMES),
                "period_x_px": period_x,
                "period_y_px": period_y,
                "sign_x": sign_x,
                "sign_y": sign_y,
                "expected_freq": blaze2d_freq(
                    period_x_px=period_x,
                    period_y_px=period_y,
                    sign_x=sign_x,
                    sign_y=sign_y,
                ),
            }
        )
    return designs


def build_declared_constants(
    resolved: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Physical constants: declared where known, null+`missing` where not.

    `resolved` is only ever populated from live device drivers at execute time;
    plan-only mode leaves the physical pitches null with an explicit reason.
    """
    declared: dict[str, Any] = {
        "panel_shape": list(DEFAULT_PANEL_SHAPE),
        "native_ccd_shape": list(NATIVE_CCD_SHAPE),
        "encoding_max_gray": G_REF,
        "two_pi_gray_note": (
            "G_REF=1023 is the rad->gray ENCODING reference only; the physical "
            "2*pi gray is read from the SLM at execute time and is not a pattern "
            "parameter here."
        ),
        "grating_periods_px": list(GRATING_PERIODS_PX),
        "grating_duty": 0.5,
        "tilt_period_px_default": DEFAULT_TILT_PERIOD_PX,
        "tilt_min_fringes": TILT_MIN_FRINGES,
        "tilt_min_ccd_px": TILT_MIN_CCD_PX,
        "ladder_base_period_px": LADDER_BASE_PERIOD_PX,
        "ladder_patch_size_px": LADDER_PATCH_SIZE_PX,
        "ladder_orders": list(LADDER_ORDERS),
        "blaze2d_designs": build_blaze2d_designs(),
        "bit_depth_required": BIT_DEPTH_REQUIRED,
        "full_scale_required": FULL_SCALE_REQUIRED,
        "camera_pixel_size_um": None,
        "slm_pixel_pitch_um": None,
        "slm_pixel_pitch_source": None,
        "focal_magnification": None,
    }
    missing: dict[str, str] = {
        "declared.camera_pixel_size_um": (
            "requires --execute: read from camera driver cam.pixel_size_um; "
            "never hardcoded in this script"
        ),
        "declared.slm_pixel_pitch_um": (
            "requires --execute: read from SLM driver accessor (Pitch_um / "
            "Pixel_Size_um); never hardcoded in this script"
        ),
        "declared.focal_magnification": (
            f"not present in {GEOMETRY_CALIB_REL}; not inferred or hardcoded"
        ),
        "declared.tilt_expected_ccd_px": tilt_expected_ccd_formula(),
        "declared.ladder_expected_ccd_px": (
            "symbolic: ccd_px = order * lambda_nm*1e-9 * focal_mm*1e-3 / "
            "(period_px * pitch_um*1e-6 * ccd_pixel_um*1e-6); period_px from "
            "ladder periods_px; pitch/focal/ccd_pixel unresolved"
        ),
    }
    if resolved:
        for key, value in resolved.items():
            if key in declared:
                declared[key] = value
            dotted = f"declared.{key}"
            if value is not None and dotted in missing:
                missing.pop(dotted)
    return declared, missing


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    raw_only = getattr(args, "patterns", None)
    only: list[str] | None = None
    if raw_only:
        only = [s.strip() for s in str(raw_only).split(",") if s.strip()]
    patterns = build_patterns(only)
    declared, missing = build_declared_constants()
    frames = {p["id"]: int(p["frames"]) for p in patterns}
    out = getattr(args, "out", None)
    tilt_period = float(
        getattr(args, "tilt_period_px", DEFAULT_TILT_PERIOD_PX)
    )
    return {
        "schema": PLAN_SCHEMA,
        "task": (
            "issue #247 geometry capture: per-axis panel->CCD scale (binary "
            "gratings), 180-deg disambiguation (blazed spot ladder), tilt "
            "sign/handedness (panel +/- prismatic ramps). Descriptive only; no "
            "fitting, no calibration."
        ),
        "out": None if out is None else str(out),
        "patterns": patterns,
        "pattern_ids": [p["id"] for p in patterns],
        "n_patterns": len(patterns),
        "frames_per_pattern": frames,
        "n_frames_expected": int(sum(frames.values())),
        "declared_constants": declared,
        "missing": missing,
        "grating_periods_px": list(GRATING_PERIODS_PX),
        "grating_duty": 0.5,
        "blaze2d_designs": [
            d
            for d in build_blaze2d_designs()
            if only is None or d["id"] in set(only)
        ],
        "blaze2d_frames": int(BLAZE2D_FRAMES),
        "tilt": build_tilt_plan(tilt_period),
        "ladder": build_ladder_plan(),
        "encoding_max_gray": G_REF,
        "frame_digests": [],
        "requested_exposure_ms": float(getattr(args, "exposure_ms", DEFAULT_EXPOSURE_MS)),
        "timestamp_utc": utc_now(),
    }


def summary_skeleton() -> dict[str, Any]:
    declared, missing = build_declared_constants()
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "not-executed",
        "descriptive_only": True,
        "note": (
            "All measurements here are DESCRIPTIVE only (per-pattern peak, "
            "centroid, saturation count, frame digests). No axis rotation is "
            "fitted and no threshold decision is made; offline analysis is a "
            "separate step."
        ),
        "n_frames": 0,
        "per_pattern": {},
        "frame_digests": [],
        "saturation_count": 0,
        "declared_constants": declared,
        "missing": missing,
        "elapsed_s": None,
        "timestamp_utc": utc_now(),
    }


# ---------------------------------------------------------------------------
# Offline self-test (no devices)
# ---------------------------------------------------------------------------
def run_self_test() -> int:
    """Assert plan/pattern invariants with no device and no directory creation."""
    # 1) Every grating: declared period, exact 50% duty, exactly two gray levels
    #    corresponding to 0 and pi under the declared encoding reference.
    gratings = [p for p in build_patterns() if p["kind"] == "grating"]
    assert len(gratings) == 2 * len(GRATING_PERIODS_PX)
    for item in gratings:
        arr = pattern_array(item, DEFAULT_PANEL_SHAPE)
        levels = np.unique(arr)
        assert len(levels) == 2, (item["id"], levels)
        assert int(levels[0]) == 0 and int(levels[1]) == PI_GRAY, (
            item["id"],
            levels,
        )
        period = int(item["period_px"])
        line = arr[0, :] if item["axis"] == 1 else arr[:, 0]
        assert np.array_equal(line[:-period], line[period:]), item["id"]
        assert float(np.mean(line[:period] == PI_GRAY)) == 0.5, item["id"]

    # 2) Blazed spot ladder: unequal one-sided orders, ladder not invariant under
    #    x -> -x or rot180, and every patch ramps exactly 2*pi per its period.
    ladders = [p for p in build_patterns() if p["kind"] == "ladder"]
    assert {p["id"] for p in ladders} == {"ladder_x", "ladder_y"}
    periods = [period for _order, period in ladder_periods()]
    assert len(set(periods)) == len(periods), "ladder amplitudes must be unequal"
    for item in ladders:
        axis = int(item["axis"])
        spots = [tuple(s) for s in item["spots"]]
        assert (0, 0) in spots, item["id"]
        assert len(spots) == 1 + len(LADDER_ORDERS), item["id"]
        assert not spots_invariant_under_sign_flip(spots), item["id"]
        assert not spots_invariant_under_rot180(spots), item["id"]
        # All ladder spots on one axis and one side (mirror vs rotation separable).
        on_axis = 0 if axis == 1 else 1
        off_axis = 1 - on_axis
        assert all(s[off_axis] == 0 for s in spots), item["id"]
        assert all(s[on_axis] >= 0 for s in spots), item["id"]
        assert [s[on_axis] for s in spots] == [0, *LADDER_ORDERS], item["id"]

        phase = ladder_phase(DEFAULT_PANEL_SHAPE, axis=axis)
        side = LADDER_PATCH_SIZE_PX
        for (row, col), (_order, period) in zip(LADDER_PATCH_ORIGINS, ladder_periods()):
            patch = phase[row : row + side, col : col + side]
            line = patch[0, :] if axis == 1 else patch[:, 0]
            expected = TWO_PI * (side - 1) / period
            assert abs(abs(line[-1] - line[0]) - expected) <= 5e-3 * expected, item["id"]
            steps = np.diff(line)
            assert np.allclose(steps, steps[0], rtol=0, atol=1e-9), item["id"]

    # 3) Tilt: prismatic ramp with 2*pi per period, >= 10 fringes, span matches
    #    2*pi*n_px/period within 0.5%, exact antisymmetry.  The rejected weak
    #    (0.5 rad peak) design must fail the pessimistic feasibility check.
    plan = build_plan(argparse.Namespace(out=Path("__selftest__")))
    assert plan["tilt"]["feasible"] is True
    assert plan["tilt"]["rejected_weak_design"]["feasible"] is False
    for axis, name in ((1, "x"), (0, "y")):
        geometry = plan["tilt"]["axes"][name]
        n = int(geometry["axis_length_px"])
        period = float(geometry["period_px"])
        expected_span = TWO_PI * n / period
        assert abs(geometry["phase_span_rad"] - expected_span) <= 5e-3 * expected_span
        assert geometry["fringe_count"] >= TILT_MIN_FRINGES, name
        assert geometry["pessimistic_ccd_px"] >= TILT_MIN_CCD_PX, name
        plus = tilt_phase(DEFAULT_PANEL_SHAPE, period_px=period, axis=axis, sign=+1)
        minus = tilt_phase(DEFAULT_PANEL_SHAPE, period_px=period, axis=axis, sign=-1)
        assert np.array_equal(plus, -minus), name
        assert abs(float(np.ptp(plus)) - TWO_PI * (n - 1) / period) <= (
            5e-3 * TWO_PI * (n - 1) / period
        ), name

    # 3b) 2D diagonal blaze: exact per-axis ramp slope on the RAW phase, four
    #     designs spanning two independent frequency directions, and a probe set
    #     that is distinguishable under axis swap / sign flip / rot180.
    blazes = [p for p in build_patterns() if p["kind"] == "blaze2d"]
    assert len(blazes) == len(BLAZE2D_DESIGNS) == 4, len(blazes)
    for item in blazes:
        px = float(item["period_x_px"])
        py = float(item["period_y_px"])
        raw = blaze2d_phase(
            DEFAULT_PANEL_SHAPE,
            period_x_px=px,
            period_y_px=py,
            sign_x=int(item["sign_x"]),
            sign_y=int(item["sign_y"]),
        )
        # Raw (unwrapped) phase: a full-panel linear ramp, so the per-pixel step
        # is constant and equals 2*pi/period on each axis exactly.
        row_steps = np.diff(raw[0, :])
        col_steps = np.diff(raw[:, 0])
        assert np.allclose(row_steps, row_steps[0], rtol=0, atol=1e-9), item["id"]
        assert np.allclose(col_steps, col_steps[0], rtol=0, atol=1e-9), item["id"]
        expected_fx_step = float(item["sign_x"]) * TWO_PI / px
        expected_fy_step = float(item["sign_y"]) * TWO_PI / py
        assert abs(row_steps[0] - expected_fx_step) <= 1e-9 * TWO_PI, item["id"]
        assert abs(col_steps[0] - expected_fy_step) <= 1e-9 * TWO_PI, item["id"]
        # The declared expected frequency must match the measured ramp slope.
        freq = item["expected_freq"]
        assert abs(freq["fx"] - row_steps[0] / TWO_PI) <= 1e-12, item["id"]
        assert abs(freq["fy"] - col_steps[0] / TWO_PI) <= 1e-12, item["id"]

    freq_vecs = [
        (float(p["expected_freq"]["fx"]), float(p["expected_freq"]["fy"]))
        for p in blazes
    ]
    assert len(set(freq_vecs)) == 4, freq_vecs
    # Rank 2: at least two linearly independent directions (designs span 2D).
    rank = np.linalg.matrix_rank(np.asarray(freq_vecs, dtype=np.float64))
    assert int(rank) == 2, (rank, freq_vecs)
    freq_set = set(freq_vecs)
    assert {(-fx, -fy) for fx, fy in freq_vecs} != freq_set, "sign flip must move spots"
    assert {(fy, fx) for fx, fy in freq_vecs} != freq_set, "axis swap must move spots"
    assert {(-fy, -fx) for fx, fy in freq_vecs} != freq_set, "rot180 must move spots"
    # The two period pairs must differ in BOTH components, otherwise the design
    # set would collapse onto one line in frequency space.
    pair_a = (1.0 / 128.0, 1.0 / 192.0)
    pair_b = (1.0 / 192.0, 1.0 / 128.0)
    assert pair_a[0] != pair_b[0] and pair_a[1] != pair_b[1], (pair_a, pair_b)
    assert [p["id"] for p in blazes] == [
        "blaze2d_px128_py192_sx+1_sy+1",
        "blaze2d_px128_py192_sx+1_sy-1",
        "blaze2d_px192_py128_sx+1_sy+1",
        "blaze2d_px192_py128_sx+1_sy-1",
    ]
    assert all("spots" not in p for p in blazes), "blaze2d must not reuse ladder spots"

    # 4) Plan / summary schema keys exist; frame counts satisfy the table.
    assert all(key in plan for key in PLAN_KEYS)
    assert all(key in summary_skeleton() for key in SUMMARY_KEYS)
    patterns = plan["patterns"]
    assert all(int(p["frames"]) >= 4 for p in patterns)
    assert int(next(p for p in patterns if p["id"] == "flat")["frames"]) >= 2
    expected_ids = {
        "flat",
        "grating_x_p128",
        "grating_y_p128",
        "grating_x_p256",
        "grating_y_p256",
        "ladder_x",
        "ladder_y",
        "tilt_x_plus",
        "tilt_x_minus",
        "tilt_y_plus",
        "tilt_y_minus",
        "blaze2d_px128_py192_sx+1_sy+1",
        "blaze2d_px128_py192_sx+1_sy-1",
        "blaze2d_px192_py128_sx+1_sy+1",
        "blaze2d_px192_py128_sx+1_sy-1",
    }
    assert set(plan["pattern_ids"]) == expected_ids
    assert plan["declared_constants"]["focal_magnification"] is None
    assert plan["declared_constants"]["camera_pixel_size_um"] is None
    assert plan["declared_constants"]["slm_pixel_pitch_um"] is None

    logger.info("issue247 geometry capture self-test: all invariants passed")
    return 0


# ---------------------------------------------------------------------------
# Hardware helpers (imports live inside the execute path)
# ---------------------------------------------------------------------------
def frame_stats(
    frame: np.ndarray,
    full_scale: float,
    saturation_fraction: float = DEFAULT_SATURATION_FRACTION,
) -> dict[str, Any]:
    threshold = float(saturation_fraction) * float(full_scale)
    return {
        "peak": float(np.max(frame)),
        "saturated_pixels": int(np.count_nonzero(frame >= threshold)),
        "saturation_threshold": float(threshold),
    }


def frame_centroid(frame: np.ndarray) -> dict[str, float]:
    """Descriptive intensity-weighted centroid of the full frame."""
    arr = frame.astype(np.float64)
    total = float(arr.sum())
    if total <= 0:
        return {"row": None, "col": None}  # type: ignore[dict-item]
    height, width = arr.shape
    rows = np.arange(height, dtype=np.float64)[:, None]
    cols = np.arange(width, dtype=np.float64)[None, :]
    return {
        "row": float((rows * arr).sum() / total),
        "col": float((cols * arr).sum() / total),
    }


def read_camera_pixel_size(cam: Any) -> float | None:
    try:
        value = cam.pixel_size_um
        return float(value) if value else None
    except Exception:
        return None


def read_slm_pitch(slm: Any) -> tuple[float | None, str | None]:
    """Read SLM pixel pitch from a driver accessor; never hardcode a number."""
    for name in ("Pitch_um", "Pixel_Size_um", "pixel_pitch_um", "get_pitch_um"):
        value = getattr(slm, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if value is not None:
            try:
                return float(value), name
            except (TypeError, ValueError):
                continue
    return None, None


def acquire_device_lock(wait_s: float) -> Any | None:
    """Acquire ``DeviceLock.default()``, polling for up to ``wait_s`` seconds."""
    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    lock = DeviceLock.default()
    deadline = time.monotonic() + max(0.0, float(wait_s))
    reported = False
    while True:
        if lock.acquire(timeout=0.0):
            return lock
        if not reported:
            holder = lock.holder()
            logger.warning(
                "bench device lock occupied by {}; waiting up to {:.1f}s",
                holder if holder else "unknown",
                wait_s,
            )
            reported = True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(LOCK_POLL_S, remaining))


# ---------------------------------------------------------------------------
# Hardware execution
# ---------------------------------------------------------------------------
def execute(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    out: Path = args.out
    lock = acquire_device_lock(args.wait_until_idle)
    if lock is None:
        logger.error("could not acquire bench device lock within {}s", args.wait_until_idle)
        return 2

    status = "not-started"
    error: str | None = None
    exit_code = 1
    created_out = False
    slm: Any = None
    cam: Any = None
    frame_records: list[dict[str, Any]] = []
    per_pattern: dict[str, Any] = {}
    resolved: dict[str, Any] = {}
    record_data: dict[str, Any] = {}
    acquired_at = utc_now()
    t_start = time.monotonic()

    release: dict[str, Any] = {
        "slm_flat_gray0": None,
        "slm_closed": False,
        "camera_closed": False,
        "lock_released": False,
        "closed_normally": False,
        "acquired_at_utc": acquired_at,
        "error": None,
    }

    try:
        out.mkdir(parents=True, exist_ok=False)
        created_out = True
        (out / "frames").mkdir(parents=True, exist_ok=False)
        write_json(out / "plan.json", plan)

        # Driver imports only here, after the lock is held and --out is fresh.
        from ao_shaping.drivers.ccd.common import (
            create_camera,
            get_camera_exposure_ms,
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
        wavelength_nm, two_pi_gray = slm.get_wavelength_info()
        slm.set_phase_overlay(None)

        pitch_um, pitch_source = read_slm_pitch(slm)
        if pitch_um is not None:
            resolved["slm_pixel_pitch_um"] = pitch_um
            resolved["slm_pixel_pitch_source"] = pitch_source

        cam = create_camera(
            "daheng",
            args.cam_id,
            exposure_time_ms=float(args.exposure_ms),
            bit_depth=BIT_DEPTH_REQUIRED,
        )
        cam.open()
        effective_bits = getattr(cam, "bit_depth", None)
        if effective_bits != BIT_DEPTH_REQUIRED:
            raise RuntimeError(
                f"Mono12 bit-depth readback is {effective_bits!r}, expected "
                f"{BIT_DEPTH_REQUIRED}"
            )
        requested = float(np.clip(float(args.exposure_ms), 0.001, 1000.0))
        set_camera_exposure_ms(cam, requested)
        exposure_ms = float(get_camera_exposure_ms(cam))
        cam_serial = getattr(cam, "serial_number", None) or getattr(cam, "sn", None)
        pixel_um = read_camera_pixel_size(cam)
        if pixel_um is not None:
            resolved["camera_pixel_size_um"] = pixel_um

        declared, missing = build_declared_constants(resolved)
        plan["declared_constants"] = declared
        plan["missing"] = missing

        frame_index = 0

        def save_frame(meta: dict[str, Any], frame: np.ndarray) -> dict[str, Any]:
            nonlocal frame_index
            if frame.dtype != np.uint16 or frame.shape != NATIVE_CCD_SHAPE:
                raise RuntimeError(
                    f"expected uint16 frame {NATIVE_CCD_SHAPE}, got "
                    f"{frame.dtype} {frame.shape}"
                )
            full_scale = float(cam.effective_full_scale(frame))
            if full_scale != float(FULL_SCALE_REQUIRED):
                raise RuntimeError(
                    f"Mono12 full scale readback {full_scale}, expected "
                    f"{FULL_SCALE_REQUIRED}"
                )
            stats = frame_stats(frame, full_scale, args.saturation_fraction)
            frame_path = out / "frames" / f"{frame_index:04d}_{meta['id']}.npy"
            np.save(frame_path, frame)
            row = dict(meta)
            row.update(
                frame_file=str(frame_path.relative_to(out)).replace("\\", "/"),
                frame_index=frame_index,
                sha256=digest(frame),
                exposure_ms=exposure_ms,
                full_scale=full_scale,
                **clock(),
                **stats,
                **frame_centroid(frame),
            )
            frame_records.append(row)
            frame_index += 1
            return row

        def display(item: dict[str, Any]) -> None:
            gray = pattern_array(item, panel_shape)
            slm.display_data(gray)

        def grab() -> np.ndarray:
            return np.asarray(cam.get_numpy_image(1, True), dtype=np.uint16).copy()

        # Pre-flight saturation probe on the flat pattern, same fixed exposure.
        display(next(p for p in plan["patterns"] if p["id"] == "flat"))
        time.sleep(float(args.settle_s))
        for _ in range(3):
            grab()
        probe = grab()
        probe_full_scale = float(cam.effective_full_scale(probe))
        probe_stats = frame_stats(
            probe, probe_full_scale, args.saturation_fraction
        )
        if probe_stats["saturated_pixels"]:
            raise SaturationAbort(
                "flat pre-flight frame saturated "
                f"({probe_stats['saturated_pixels']} pixels >= "
                f"{probe_stats['saturation_threshold']})"
            )

        for item in plan["patterns"]:
            display(item)
            time.sleep(float(args.settle_s))
            for _ in range(3):
                grab()
            rows: list[dict[str, Any]] = []
            for _ in range(int(item["frames"])):
                frame = grab()
                row = save_frame(
                    {"id": item["id"], "kind": item["kind"]}, frame
                )
                rows.append(row)
                if row["saturated_pixels"]:
                    raise SaturationAbort(
                        f"{item['id']} frame saturated "
                        f"({row['saturated_pixels']} pixels >= "
                        f"{row['saturation_threshold']})"
                    )
            per_pattern[item["id"]] = {
                "frames": len(rows),
                "peaks": [float(r["peak"]) for r in rows],
                "centroids": [
                    {"row": r["row"], "col": r["col"]} for r in rows
                ],
                "digests": [r["sha256"] for r in rows],
            }

        record_data = dict(
            wavelength_nm=int(wavelength_nm),
            two_pi_gray=two_pi_gray,
            panel_res=list(panel_shape),
            cam_serial=cam_serial,
            bit_depth=int(effective_bits),
            full_scale=int(FULL_SCALE_REQUIRED),
            exposure_ms=exposure_ms,
            pixel_um=pixel_um,
            probe_peak=probe_stats["peak"],
            probe_threshold=probe_stats["saturation_threshold"],
        )

        # ao-optical-path-record/v1 via the repo's pure assembly helper.
        reasons: dict[str, str] = {}
        if pixel_um is None:
            reasons["devices.camera.pixel_size_um"] = (
                "camera driver cam.pixel_size_um unavailable; not hardcoded"
            )
        record = build_optical_path_record(
            run_id=f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_issue247_geometry",
            date=f"{datetime.now(timezone.utc):%Y-%m-%d}",
            operator_note=(
                "issue #247 geometry capture: descriptive panel<->CCD probes; "
                "no rotation fit; pitches/magnification not inferred"
            ),
            devices={
                "slm": {
                    "model": "Santec SLM-200",
                    "slm_number": args.slm_number,
                    "wavelength_nm": int(wavelength_nm),
                    "two_pi_grayscale": two_pi_gray,
                    "panel_res": list(panel_shape),
                    "open_mode": "memory",
                },
                "camera": {
                    "model": "Daheng MER2-507-23GM",
                    "serial": cam_serial,
                    "bit_depth": int(effective_bits),
                    "full_scale": int(FULL_SCALE_REQUIRED),
                    "exposure_used_ms": exposure_ms,
                    "averaging_frames": 1,
                    "discard_frames_after_exposure_change": 3,
                    "pixel_size_um": pixel_um,
                },
            },
            optical_path={
                "geometry_calib_file": GEOMETRY_CALIB_REL,
                "probe_status": "geometry-capture (descriptive; offline fit pending)",
            },
            exposure_check={
                "peak_counts_at_used_exposure": probe_stats["peak"],
                "saturation_threshold_counts": probe_stats["saturation_threshold"],
                "saturated": False,
                "settle_timeout": False,
            },
            occupancy={
                "acquired_at": acquired_at,
                "closed_normally": True,
            },
            reasons=reasons,
        )
        write_json(out / "optical_path_record.json", record)

        status = "ok"
        exit_code = 0
    except FileExistsError:
        status = "out_exists"
        error = f"--out already exists: {out}"
        logger.error(error)
        exit_code = 2
    except SaturationAbort as exc:
        status = "aborted_saturation"
        error = f"SaturationAbort: {exc}"
        logger.error(error)
        exit_code = 1
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        logger.error(error)
        exit_code = 1
    finally:
        # Always release, on every path: cam.close -> slm.close -> lock.release.
        if slm is not None:
            try:
                slm.set_grayscale(0)
                release["slm_flat_gray0"] = "ok"
            except Exception as exc:
                release["slm_flat_gray0"] = f"failed: {exc}"
        if cam is not None:
            try:
                cam.close()
                release["camera_closed"] = True
            except Exception as exc:
                release["camera_closed"] = f"failed: {exc}"
        if slm is not None:
            try:
                slm.close()
                release["slm_closed"] = True
            except Exception as exc:
                release["slm_closed"] = f"failed: {exc}"
        release["lock_released"] = False
        try:
            lock.release()
            release["lock_released"] = True
        except Exception as exc:
            release["lock_release_error"] = str(exc)

        release["error"] = error
        release["status"] = status
        release["finished_utc"] = utc_now()
        release["elapsed_s"] = time.monotonic() - t_start
        release["closed_normally"] = bool(
            release["camera_closed"] is True
            and release["slm_closed"] is True
            and release["lock_released"] is True
            and error is None
        )

        if created_out:
            declared, missing = build_declared_constants(resolved)
            plan["frame_digests"] = [r["sha256"] for r in frame_records]
            plan["declared_constants"] = declared
            plan["missing"] = missing
            write_json(out / "plan.json", plan)

            summary = summary_skeleton()
            summary.update(
                status=status,
                n_frames=len(frame_records),
                per_pattern=per_pattern,
                frame_digests=[r["sha256"] for r in frame_records],
                saturation_count=sum(
                    1 for r in frame_records if r["saturated_pixels"]
                ),
                declared_constants=declared,
                missing=missing,
                elapsed_s=release["elapsed_s"],
                timestamp_utc=utc_now(),
                error=error,
                exposure_ms=record_data.get("exposure_ms"),
            )
            write_json(out / "summary.json", summary)
            write_json(out / "released.json", release)

    logger.info("issue247 geometry capture: status={} frames={} -> {}",
                status, len(frame_records), out)
    return exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Issue #247 geometry capture (plan-only by default): binary gratings "
            "for per-axis panel->CCD scale, a one-sided blazed spot ladder for "
            "180-deg branch disambiguation, and panel +/- prismatic tilts for "
            "sign/handedness. Descriptive only; no fitting, no threshold decisions."
        )
    )
    parser.add_argument("--out", type=Path, default=None, help="fresh output directory")
    parser.add_argument(
        "--execute", action="store_true", help="open hardware and acquire (only way to touch devices)"
    )
    parser.add_argument(
        "--self-test", action="store_true", help="assert offline invariants; no devices, no --out"
    )
    parser.add_argument(
        "--patterns",
        default=None,
        help=(
            "comma-separated pattern ids to run (default: the whole frozen "
            "list). Unknown ids abort. Used to keep a run inside the currently "
            "approved scope, e.g. 'flat,blaze2d_px128_py192_sx+1_sy+1,...'"
        ),
    )
    parser.add_argument(
        "--tilt-period-px", type=float, default=DEFAULT_TILT_PERIOD_PX,
        help="tilt ramp: one 2*pi wrap every this many panel px (default 128)",
    )
    parser.add_argument(
        "--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS,
        help="fixed exposure for the whole run (ms)",
    )
    parser.add_argument(
        "--settle-s", type=float, default=DEFAULT_SETTLE_S,
        help="settle time after displaying each pattern",
    )
    parser.add_argument(
        "--saturation-fraction", type=float, default=DEFAULT_SATURATION_FRACTION,
        help="abort if any frame pixel >= this fraction of full scale",
    )
    parser.add_argument(
        "--wait-until-idle", type=float, default=DEFAULT_WAIT_IDLE_S,
        help="seconds to wait for the bench device lock",
    )
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--wavelength-nm", type=int, default=1064)
    parser.add_argument("--cam-id", type=int, default=0)
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()
    if args.out is None:
        parser.error("--out is required unless --self-test is given")
    if args.exposure_ms <= 0 or args.settle_s < 0:
        parser.error("--exposure-ms>0 and --settle-s>=0 required")
    if args.tilt_period_px <= 0:
        parser.error("--tilt-period-px>0 required")
    if not 0 < args.saturation_fraction <= 1:
        parser.error("--saturation-fraction must be in (0, 1]")
    if args.wait_until_idle < 0:
        parser.error("--wait-until-idle>=0 required")

    plan = build_plan(args)
    # Plan print in main is intentionally allowed by the repo conventions.
    print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
    if not args.execute:
        print("skip: requires --execute", file=sys.stderr)
        return 0
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
