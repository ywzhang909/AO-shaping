"""Smooth-Zernike sweep probe: acquire bench response curves with a *stable* read.

This is the reusable core behind ``scripts/model_in_loop_hw_runbook.py
--stage sweep``. It was extracted because the acquisition logic -- pattern
building, the settle criterion, the per-point metrics, the Recorder bookkeeping
and the ``.npz`` round-trip -- is useful to anything that needs to characterise
this bench, not just to that one script.

What it buys over calling :mod:`ao_shaping.tools.slm.slm_bench_probe` directly:

* **One settle criterion, applied once.** :func:`display_and_average` discards
  frames until two consecutive readings agree. An unsettled frame is not a noisy
  frame, it is a *wrong* frame that still looks plausible -- see the module
  docstring of :mod:`ao_shaping.tools.slm.slm_bench_probe` for the measurement
  that cost a 3.3x error.
* **Devices are injected, not constructed.** :func:`acquire_sweep` takes an open
  ``cam``/``slm`` pair, so the whole probe runs offline against mocks in the test
  suite. Only :func:`main` touches hardware.
* **Provenance is structural.** Every point carries its mode, axis and
  coefficient, and the sweep as a whole carries the aperture geometry, so a
  stored sweep can be re-fitted later without remembering how it was taken.

Layering note: this module deliberately does **not** import
``model_in_loop_shaping``'s private ``_SWEEP_MODE_NM`` table even though the two
cover the same modes. The optimizer is a library layer that must not depend on
hardware tools, so the duplication is intentional and each table is canonical for
its own layer; :data:`SWEEP_MODE_NM` is the hardware-facing one and is what the
fitter's records are labelled with.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from loguru import logger

from ao_shaping.tools.slm.slm_bench_probe import (
    SLM_PITCH_M,
    SLM_PANEL_H,
    SLM_PANEL_W,
    core_fraction,
    display_and_average,
    measure_spot,
    ramp_panel,
    zernike_panel,
)

#: Panel raster of the Santec SLM-200 used on this bench (height, width).
DEFAULT_PANEL_SHAPE: tuple[int, int] = (SLM_PANEL_H, SLM_PANEL_W)

#: Smooth Zernike modes this probe sweeps, mapped to their ``(n, m)``.
#:
#: The cubic/quartic entries are what make the beam waist identifiable at all:
#: defocus alone cannot separate "wide beam" from "residual quadratic
#: aberration", and the quadratic pair is degenerate with that residual. They
#: respond strongly and distinctly on this bench (fitted curvatures 92-219).
SWEEP_MODE_NM: dict[str, tuple[int, int]] = {
    "defocus": (2, 0),
    "astig_x": (2, -2),
    "astig_y": (2, 2),
    "coma_x": (3, -1),
    "coma_y": (3, 1),
    "spherical": (4, 0),
}

#: ``(n, m)`` for the two panel-axis tilts, indexed by panel axis.
TILT_AXES_NM: dict[str, tuple[int, int]] = {"x": (1, 1), "y": (1, -1)}

#: ``mode`` strings the fitter understands but that are not Zernike modes.
AUX_MODES: frozenset[str] = frozenset({"flat", "ramp"})

#: Below this centroid/peak ratio the focal plane is a ring, not one lobe, and
#: its "width" is a ring diameter rather than a spot size. Mirrors
#: ``model_in_loop_shaping._DEFOCUS_MIN_HOLLOWNESS``.
SINGLE_LOBE_MIN_HOLLOWNESS: float = 0.60

#: Stable-mode integer codes for the Recorder sidecar. The standard recorder
#: serialiser coerces every scalar to ``float``, which silently destroys string
#: columns, so mode/axis travel as ints and the mapping is written to the JSON.
MODE_CODES: dict[str, int] = {
    "flat": 0, "tilt": 1, "defocus": 2, "astig_x": 3, "astig_y": 4,
    "ramp": 5, "coma_x": 6, "coma_y": 7, "spherical": 8,
}
AXIS_CODES: dict[str, int] = {"": 0, "x": 1, "y": 2}


@dataclass
class SweepPoint:
    """One requested acquisition.

    Attributes:
        mode: A key of :data:`SWEEP_MODE_NM`, or one of :data:`AUX_MODES`.
        coefficient: Radians for a Zernike mode, or the period in panel pixels
            for a ``ramp``.
        axis: Panel axis for ``tilt``/``ramp``; unused otherwise.
        label: Overrides the generated ``"<mode><coefficient>"`` tag.
    """

    mode: str
    coefficient: float = 0.0
    axis: str = ""
    label: str = ""

    def tag(self) -> str:
        """A filesystem- and column-safe identifier for this point."""
        return self.label or f"{self.mode}{self.coefficient:+.2f}"


@dataclass
class SweepPointResult:
    """A measured :class:`SweepPoint`: scalars plus the raw arrays."""

    point: SweepPoint
    peak: float
    fwhm_px: float
    centroid_x: float
    centroid_y: float
    hollowness: float
    phase_rad: np.ndarray
    frame: np.ndarray

    def is_single_lobe(self) -> bool:
        """Whether the focal plane was one spot rather than a ring."""
        return self.hollowness >= SINGLE_LOBE_MIN_HOLLOWNESS


@dataclass
class SweepResult:
    """Everything one sweep produced, in acquisition order."""

    points: list[SweepPointResult] = field(default_factory=list)
    zernike_radius: int = 0
    pupil_center: tuple[int, int] = (0, 0)
    panel_shape: tuple[int, int] = DEFAULT_PANEL_SHAPE

    def tags(self) -> list[str]:
        """Labels in acquisition order."""
        return [r.point.tag() for r in self.points]

    def as_columns(self) -> dict[str, np.ndarray]:
        """Columns for ``np.savez``: labels, modes, metrics and provenance.

        Returns:
            A dict of equal-length arrays plus the scalar aperture geometry,
            ready for :func:`np.savez_compressed`.
        """
        def col(name: str, fn: Callable[[SweepPointResult], Any]) -> np.ndarray:
            return np.array([fn(r) for r in self.points])

        return {
            "label": np.array(self.tags()),
            "mode": np.array([r.point.mode for r in self.points]),
            "axis": np.array([r.point.axis for r in self.points]),
            "coefficient": col("coefficient", lambda r: r.point.coefficient),
            "fwhm_px": col("fwhm_px", lambda r: r.fwhm_px),
            "centroid_x": col("centroid_x", lambda r: r.centroid_x),
            "centroid_y": col("centroid_y", lambda r: r.centroid_y),
            "peak": col("peak", lambda r: r.peak),
            "hollowness": col("hollowness", lambda r: r.hollowness),
            "zernike_radius": np.array(self.zernike_radius),
            "pupil_center": np.array(self.pupil_center),
        }


def point_panel_phase(
    point: SweepPoint,
    pupil_center: tuple[int, int],
    zernike_radius: int,
    panel_shape: tuple[int, int] = DEFAULT_PANEL_SHAPE,
) -> np.ndarray:
    """Build the panel phase for one sweep point.

    Args:
        point: What to display.
        pupil_center: Beam centre in panel pixels ``(x, y)``.
        zernike_radius: Aperture radius in panel pixels.
        panel_shape: ``(height, width)`` of the panel raster.

    Returns:
        Raw unwrapped radians, ``panel_shape``.

    Raises:
        ValueError: If the point's mode is not a known mode or its axis is
            missing where the mode needs one.
    """
    if point.mode == "flat":
        return np.zeros((int(panel_shape[0]), int(panel_shape[1])), dtype=np.float64)
    if point.mode == "ramp":
        if point.axis not in ("x", "y"):
            raise ValueError(f"ramp point needs axis 'x' or 'y', got {point.axis!r}")
        return ramp_panel(
            int(point.coefficient), 1 if point.axis == "x" else 0, panel_shape
        )
    if point.mode == "tilt":
        if point.axis not in TILT_AXES_NM:
            raise ValueError(f"tilt point needs axis 'x' or 'y', got {point.axis!r}")
        nm = TILT_AXES_NM[point.axis]
    elif point.mode in SWEEP_MODE_NM:
        nm = SWEEP_MODE_NM[point.mode]
    else:
        raise ValueError(
            f"unknown sweep mode {point.mode!r}; expected one of "
            f"{sorted(SWEEP_MODE_NM)}, 'tilt', or {sorted(AUX_MODES)}"
        )
    return zernike_panel({nm: float(point.coefficient)}, zernike_radius, pupil_center,
                         panel_shape)


def capture_settled(
    cam: Any,
    slm: Any,
    phase_rad: np.ndarray,
    *,
    n_frames: int = 4,
    n_discard: int = 3,
    wait_time_s: float = 0.5,
    stable_tol: float = 0.02,
    max_wait_s: float = 6.0,
) -> np.ndarray:
    """Display one phase and return the averaged far-field frame once settled.

    The single-phase entry point, for callers that build their own acquisition
    order (an ABBA interleave, a repeated push-pull) and only need the settle
    discipline and the averaging from here.

    Args:
        cam: An open camera exposing ``get_numpy_image(n_sample=...)``.
        slm: An open SLM exposing ``create_phase_from_array`` and
            ``display_data(gray, wait_time_s)``.
        phase_rad: Raw unwrapped radians, panel-shaped.
        n_frames: Frames averaged into the returned measurement.
        n_discard: Frames discarded before the stability loop starts.
        wait_time_s: Explicit LCOS settle before measuring.
        stable_tol: Relative agreement required between consecutive readings.
        max_wait_s: Cap on the wait for stability.

    Returns:
        The averaged frame.
    """
    return display_and_average(
        cam, slm, phase_rad, n_frames=n_frames, n_discard=n_discard,
        wait_time_s=wait_time_s, stable_tol=stable_tol, max_wait_s=max_wait_s,
    )


def acquire_sweep(
    cam: Any,
    slm: Any,
    points: Sequence[SweepPoint],
    *,
    pupil_center: tuple[int, int],
    zernike_radius: int,
    panel_shape: tuple[int, int] = DEFAULT_PANEL_SHAPE,
    n_frames: int = 4,
    n_discard: int = 3,
    wait_time_s: float = 0.5,
    stable_tol: float = 0.02,
    max_wait_s: float = 6.0,
    on_point: Callable[[SweepPointResult], None] | None = None,
) -> SweepResult:
    """Display each point in turn and measure the settled far field.

    The flat reference is captured on the **first** call rather than being
    requested by the caller, and always before anything else is displayed: the
    panel retains the previous run's pattern, so a "flat" read taken before any
    write is really the last frame of the last experiment.

    Args:
        cam: An open camera exposing ``get_numpy_image(n_sample=...)``.
        slm: An open SLM exposing ``create_phase_from_array`` and
            ``display_data(gray, wait_time_s)``.
        points: What to acquire, in order.
        pupil_center: Beam centre in panel pixels ``(x, y)``.
        zernike_radius: Aperture radius in panel pixels.
        panel_shape: ``(height, width)`` of the panel raster.
        n_frames: Frames averaged into the returned measurement.
        n_discard: Frames discarded before the stability loop starts.
        wait_time_s: Explicit LCOS settle before measuring.
        stable_tol: Relative agreement required between consecutive readings.
        max_wait_s: Cap on the wait for stability.
        on_point: Called with each result as it lands, for progress output.

    Returns:
        The measurements, prepended with the flat reference.
    """
    ordered = [SweepPoint("flat", 0.0, "", "flat"), *points]
    result = SweepResult(
        zernike_radius=int(zernike_radius),
        pupil_center=(int(pupil_center[0]), int(pupil_center[1])),
        panel_shape=(int(panel_shape[0]), int(panel_shape[1])),
    )
    for point in ordered:
        phase = point_panel_phase(point, result.pupil_center, zernike_radius,
                                  result.panel_shape)
        frame = capture_settled(
            cam, slm, phase, n_frames=n_frames, n_discard=n_discard,
            wait_time_s=wait_time_s, stable_tol=stable_tol, max_wait_s=max_wait_s,
        )
        spot = measure_spot(frame)
        measured = SweepPointResult(
            point=point, peak=spot.peak, fwhm_px=spot.fwhm_px,
            centroid_x=spot.centroid_x, centroid_y=spot.centroid_y,
            hollowness=spot.hollowness, phase_rad=phase, frame=frame,
        )
        result.points.append(measured)
        logger.info(
            "{:<18} {}{}", point.tag(), spot.as_row(),
            "" if measured.is_single_lobe() else "  <- not a single lobe",
        )
        if on_point is not None:
            on_point(measured)
    return result


def sweep_to_recorder_rows(result: SweepResult) -> list[dict[str, Any]]:
    """Convert a sweep into Recorder rows, raw process arrays included.

    The standard recorder serialiser coerces scalars to ``float``, which
    destroys string columns, so ``mode``/``axis``/``label`` are stored as their
    integer codes (see :data:`MODE_CODES` / :data:`AXIS_CODES`) and the mapping
    travels in the JSON sidecar. ``peak`` is the Recorder mark, so a stored sweep
    can be ranked by "sharpest focus" without re-reading the arrays.

    Args:
        result: The sweep to convert.

    Returns:
        One dict per point, each with ``_epoch``, ``_phase`` and ``_img``.
    """
    rows: list[dict[str, Any]] = []
    for index, measured in enumerate(result.points):
        point = measured.point
        rows.append({
            "peak": measured.peak,
            "_epoch": index,
            "mode_id": MODE_CODES[point.mode],
            "axis_id": AXIS_CODES[point.axis],
            "coefficient": float(point.coefficient),
            "fwhm_px": measured.fwhm_px,
            "centroid_x": measured.centroid_x,
            "centroid_y": measured.centroid_y,
            "hollowness": measured.hollowness,
            "_phase": np.asarray(measured.phase_rad, dtype=np.float32),
            "_img": np.asarray(measured.frame, dtype=np.float32),
        })
    return rows


def default_sweep_points(
    tilt: Iterable[float] = (-1.0, 1.0),
    defocus: Iterable[float] = (-4.0, -2.5, -1.5, -0.75, 0.75, 1.5, 2.5, 4.0),
    astig: Iterable[float] = (-3.0, -1.5, 1.5, 3.0),
    coma: Iterable[float] = (-1.2, -0.6, 0.6, 1.2),
    spherical: Iterable[float] = (-1.2, -0.6, 0.6, 1.2),
    ramps: Iterable[float] = (120.0, 240.0, 480.0, 960.0, 1920.0),
) -> list[SweepPoint]:
    """The standard bench sweep: ramps, tilts, then the six Zernike modes.

    Keep the tilt coefficients small. Above ~1.5 rad the spot deforms
    (hollowness falls to 0.65) and the centroid stops tracking the peak, which
    halves the measured slope; the ramp sweep is immune and is the authoritative
    focal-scale measurement, so the tilt sweep only has to corroborate it.

    Args:
        tilt: Zernike tilt coefficients in radians, applied on both axes.
        defocus: Defocus coefficients in radians.
        astig: Astigmatism coefficients in radians, applied on both axes.
        coma: Coma coefficients in radians, applied on both axes.
        spherical: Spherical coefficients in radians.
        ramps: Phase-ramp periods in panel pixels, applied on both axes.

    Returns:
        The points to hand to :func:`acquire_sweep`, in acquisition order.
    """
    points: list[SweepPoint] = []
    for axis in ("x", "y"):
        for period in ramps:
            points.append(SweepPoint("ramp", float(period), axis))
    for axis in ("x", "y"):
        for coeff in tilt:
            points.append(SweepPoint("tilt", float(coeff), axis))
    for coeff in defocus:
        points.append(SweepPoint("defocus", float(coeff)))
    for mode in ("astig_x", "astig_y"):
        for coeff in astig:
            points.append(SweepPoint(mode, float(coeff)))
    for mode in ("coma_x", "coma_y"):
        for coeff in coma:
            points.append(SweepPoint(mode, float(coeff)))
    for coeff in spherical:
        points.append(SweepPoint("spherical", float(coeff)))
    return points


def save_sweep_npz(path: Path, result: SweepResult) -> Path:
    """Write a sweep's columns to a compressed ``.npz``.

    Args:
        path: Destination file; parents are created.
        result: The sweep to persist.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result.as_columns())
    logger.info("sweep columns -> {} ({} points)", path, len(result.points))
    return path


def _parse_floats(text: str) -> list[float]:
    return [float(v) for v in str(text).split(",") if v.strip()]


def _parse_int_tuple(text: str) -> tuple[int, int]:
    """Parse an ``"x,y"`` integer pair, e.g. a panel-space pupil centre."""
    parts = [p for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected 'x,y', got {text!r}")
    return int(parts[0]), int(parts[1])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    from ao_shaping.utils.io.cli_helpers import setup_coredumpy

    setup_coredumpy()
    ap = argparse.ArgumentParser(
        description="光滑 Zernike 台架扫描探针 (需硬件: Santec SLM-200 + 远场相机)",
    )
    ap.add_argument("--out", default="data/slm_zernike_sweep",
                    help="输出目录 (默认 data/slm_zernike_sweep)")
    ap.add_argument("--slm-number", type=int, default=1)
    ap.add_argument("--slm-wavelength", type=int, default=1064)
    ap.add_argument("--cam-type", default="daheng", choices=["daheng", "miicam"])
    ap.add_argument("--cam-id", type=int, default=0)
    ap.add_argument("--exposure-ms", type=float, default=3.0)
    ap.add_argument("--pupil-center", default="960,600", type=_parse_int_tuple,
                    help="光斑中心 (面板 px x,y)")
    ap.add_argument("--zernike-radius", type=int, default=450)
    ap.add_argument("--sweep-tilt", default="-1.0,1.0")
    ap.add_argument("--sweep-defocus", default="-4.0,-2.5,-1.5,-0.75,0.75,1.5,2.5,4.0")
    ap.add_argument("--sweep-astig", default="-3.0,-1.5,1.5,3.0")
    ap.add_argument("--sweep-coma", default="-1.2,-0.6,0.6,1.2")
    ap.add_argument("--sweep-spherical", default="-1.2,-0.6,0.6,1.2")
    ap.add_argument("--sweep-ramps", default="120,240,480,960,1920")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--discard", type=int, default=3)
    ap.add_argument("--settle-s", type=float, default=0.5)
    ap.add_argument("--stable-tol", type=float, default=0.02)
    ap.add_argument("--max-wait-s", type=float, default=6.0)
    ap.add_argument("--save-frames/--no-save-frames", default=True)
    ap.add_argument("--no-hw", action="store_true",
                    help="不打开硬件, 只打印将要采集的点 (自检用)")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweep on hardware and persist it.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success.
    """
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.utils.io.file import Recorder, save_recorder_debug_artifacts

    args = _parse_args(argv)
    points = default_sweep_points(
        tilt=_parse_floats(args.sweep_tilt),
        defocus=_parse_floats(args.sweep_defocus),
        astig=_parse_floats(args.sweep_astig),
        coma=_parse_floats(args.sweep_coma),
        spherical=_parse_floats(args.sweep_spherical),
        ramps=_parse_floats(args.sweep_ramps),
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_hw:
        logger.info("--no-hw: {} 个采集点 (含前置 flat), 不打开硬件",
                    len(points) + 1)
        for point in points:
            logger.info("  {} axis={} coeff={:+.3f}", point.tag(), point.axis or "-",
                        point.coefficient)
        return 0

    with Santec(
        slm_number=args.slm_number, wavelength=args.slm_wavelength, video_mode=0
    ) as slm, create_camera(
        args.cam_type, args.cam_id, exposure_time_ms=args.exposure_ms
    ) as cam:
        cam.reset_exposure_time(float(args.exposure_ms))
        result = acquire_sweep(
            cam, slm, points,
            pupil_center=tuple(args.pupil_center),
            zernike_radius=args.zernike_radius,
            n_frames=args.frames, n_discard=args.discard,
            wait_time_s=args.settle_s, stable_tol=args.stable_tol,
            max_wait_s=args.max_wait_s,
        )

    npz_path = save_sweep_npz(out_dir / "sweep_records.npz", result)
    rings = [m for m in result.points[1:] if not m.is_single_lobe()]
    if rings:
        logger.warning(
            "{} 个采集点的远场不是单瓣 (hollowness < {:.2f}), 其宽度不可当作高斯"
            "光斑宽度使用: {}",
            len(rings), SINGLE_LOBE_MIN_HOLLOWNESS,
            ", ".join(m.point.tag() for m in rings),
        )
    logger.info("panel pitch {} m, 斜坡标定请用 ramp 点 (S = 位移 x 周期)",
                SLM_PITCH_M)

    recorder = Recorder(mark="peak", mode="min")
    for row in sweep_to_recorder_rows(result):
        recorder.append(row)
    save_recorder_debug_artifacts(
        recorder,
        _root=out_dir.parent,
        name="slm_zernike_sweep",
        extra_meta={
            "sweep_npz": str(npz_path),
            "zernike_radius": result.zernike_radius,
            "pupil_center": list(result.pupil_center),
            "panel_shape": list(result.panel_shape),
            "mode_codes": MODE_CODES,
            "axis_codes": AXIS_CODES,
            "single_lobe_min_hollowness": SINGLE_LOBE_MIN_HOLLOWNESS,
            "exposure_ms": args.exposure_ms,
            "settle_s": args.settle_s,
            "stable_tol": args.stable_tol,
            "max_wait_s": args.max_wait_s,
        },
    )
    logger.info("扫描探针完成 -> {}", out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
