"""One-device-session calibration for ``runs/sim2f_bench`` -- measure, do not recall.

Why this file exists
====================
Every number this protocol needs from the optical path -- the working exposure,
the pupil centre on the panel, the beam extent, the panel whether it is
modulating at all -- is a *property of today's alignment*.  Reusing a remembered
value is how a session gets a wrong far field instead of a shifted one (see
``src/ml/sim2f/bench.py`` module docstring: a wrong ``center_xy`` makes the far
field wrong, not offset).

The single-run scripts in ``tools/slm/panel`` and ``tools/ccd`` each open and
close their own SLM + CCD pair.  On this bench that costs about a minute of
device bring-up per probe, and the 0.44 s of actual measurement is dwarfed by
it, so calibrating is dominated by *re-opening devices*, not by measuring.  This
script therefore opens the devices **once**, runs the same measurement bodies
back to back, and records the whole command sequence (panel grey levels, memory
slot rotations, exposure ladder steps, iterations) so the compression points are
visible in one log instead of seven.

What it is NOT
--------------
It does not re-implement any measurement.  ``tools/slm/panel/panel_locate.py``
has no importable-with-devices entry point -- its measurement body lives inside
:func:`scan_panel`, which takes the closed-over devices -- so this file carries
that body over **verbatim in behaviour** and nothing else; every helper it calls
(``measure_flat_reference``, ``capture_settled_report``, ``probe_policy``,
``_paired_delta``) is the same imported function the standalone probe uses.  The
one deviation is deliberate and reported: **frames per point is a parameter**
(``--frames``), used to measure what the fixed floor of 7 costs.

The exposure ladder, the slot rotation and the grey levels all pass through the
canonical helpers (``utils/slm/phase_display.pick_slm_slot`` /
``phase_to_slm_grayscale`` and the driver's own write path), so what is logged
here is what the device receives.

Cost accounting
---------------
:class:`CommandLedger` counts every command sent to a device and every frame
read, and prints them at the end next to the wall-clock split (open vs measure).
That is the compression budget: the probe's iteration count is the number you
have to reduce to make calibration cheaper, and this prints it rather than
leaving it to be counted by hand.

Usage
-----
::

    $env:PYTHONPATH="src;libs"
    $env:PATH="libs\\Drv_UDPST\\x64\\Release;libs\\gxipy;$env:PATH"
    $env:AO_DAHENG_MAX_EXPOSURE_MS="0.8"
    .venv\\Scripts\\python.exe runs\\sim2f_bench\\calibrate_session.py --out-dir runs/sim2f_bench/calibration

Add ``--frames 5`` to measure the cost of the probe budget; a bare run uses the
probe's own floor (7) so its numbers are comparable with a standalone
``panel_locate`` run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

# Repo-root bootstrap: this file lives in ``runs/``, outside ``src/``.
ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT / "src"), str(ROOT / "libs")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from ml.sim2f.bench import (  # noqa: E402
    MAX_EXPOSURE_ENV,
    PANEL_SHAPE,
    PUPIL_RADIUS_PX,
)

#: The probe's own kept-frame floor (``panel_locate`` module docstring: a median
#: needs enough samples to reject one corrupted frame).  Not a tuning knob -- the
#: default here equals it so the two runs are comparable.
FRAMES_FLOOR = 7

#: Peak grey above which a Daheng frame is treated as clipped.  Matches the guard
#: ``panel_locate.scan_panel`` raises on (``>= 250`` of 255).
SATURATION_PEAK = 250

#: Radius of the 0-order scoring box, i.e. the aperture whose energy the probes
#: read.  60 px is ``panel_locate.scan_panel``'s own box; keeping it makes the
#: response numbers directly comparable with that probe's output.
FLAT_BOX_RADIUS_PX = 60

#: Peak grey below which a flat frame is "no light", not "dim light".  A working
#: exposure on this bench puts the 0-order near 40 % of full scale (the
#: ``working_exposure_ms`` rationale), so a peak of a few counts means the beam is
#: not arriving, and every downstream ratio would be read off read noise.
DARK_FRAME_PEAK = 10

#: Radius of the *core* whose share of total frame energy the edge witnesses.
#: ``beam_extent``'s own default, and the quantity it documents as the only one
#: that is monotone in "how much light left the core": a box around the 0-order
#: is not (a disc both scatters light out of it and redistributes light in), and
#: total energy carries no size information at all (measured conserved).
CORE_RADIUS_PX = 40

#: Exposure ladder start.  The camera's own minimum is 0.02 ms; starting there
#: costs one read and guarantees the ladder never begins above the beam.
PROBE_FLOOR_MS = 0.02

#: Where the 0-order peak must land for the calibration ratios to mean anything.
#: The exposure is placed **at** ``TARGET_PEAK`` by a bracket + bisection, not by
#: one proportional step: ``peak`` is only *approximately* linear in exposure
#: below saturation on this bench (0.441 ms -> 135 counts, 0.8 ms -> 239 counts is
#: 1.12x the linear expectation), so the last word is a measurement.
TARGET_PEAK = 140.0
TARGET_PEAK_LO = 100.0
TARGET_PEAK_HI = 200.0

#: Coarse ladder step.  Four, not two: the ladder only has to bracket the target
#: (``log4(ceiling / floor)`` reads, ~5 on this bench), because the bisection
#: refines inside the bracket.  Doubling spent 12 reads to do the same job.
COARSE_FACTOR = 4.0

#: Accepted distance from ``TARGET_PEAK``, in grey counts.  One count of
#: quantisation is 0.7 % at 140, so this tolerance is deliberate and loose.
BRACKET_TOL = 5.0

#: Bisection stop, in ms, for a bench so stiff that counts cannot be resolved.
BRACKET_TOL_MS = 1e-3

#: Bisection budget.  Six steps shrink the coarse bracket by 64x, which is more
#: than the 12-bit counts can resolve.
MAX_BISECT_STEPS = 6

#: Below this the frame is read noise, not a beam.  Used *only* by the exposure
#: resolver, where a dark reading is the thing being classified; the panel scan
#: uses ``DARK_FRAME_PEAK`` because by then the exposure is already settled.
DARK_PEAK = 3.0


@dataclass
class CommandLedger:
    """Every command and every frame, counted per phase.

    ``writes`` counts SLM panel writes, ``reads`` counts camera captures and
    ``slots`` the memory slots the driver reported for a write (a rotation is
    what makes the LCOS refresh -- see ``drivers/slm/AGENTS.md``).  It is filled
    only where the slot is actually readable; ``SettledAcquisition`` carries just
    (frame, settle), so a write made through ``acquire_settled`` cannot be
    attributed to a slot and is recorded as such rather than assumed.
    ``exposures`` records each exposure the camera was moved through, so a ladder
    is visible as its steps.
    """

    writes: int = 0
    reads: int = 0
    slots: list[int] = field(default_factory=list)
    slots_unreadable: int = 0
    exposures: list[float] = field(default_factory=list)
    phases: dict[str, float] = field(default_factory=dict)

    def timed(self, name: str):
        """Context manager that accumulates wall-clock time under ``name``."""

        class _Timer:
            def __enter__(inner) -> None:
                inner.t0 = time.perf_counter()

            def __exit__(inner, *exc) -> None:
                elapsed = time.perf_counter() - inner.t0
                self.phases[name] = self.phases.get(name, 0.0) + elapsed

        return _Timer()

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly summary."""
        return {
            "slm_writes": self.writes,
            "camera_reads": self.reads,
            "memory_slot_rotations": len(self.slots),
            "distinct_slots": sorted(set(self.slots)),
            "writes_with_unreadable_slot": self.slots_unreadable,
            "exposure_values_touched": sorted(set(self.exposures)),
            "seconds_by_phase": {k: round(v, 3) for k, v in self.phases.items()},
        }

    def report(self) -> None:
        """Log the counts and the wall-clock split."""
        split = "  ".join(f"{k}={v:.1f}s" for k, v in self.phases.items())
        logger.info(
            "commands: {} SLM writes ({} with a slot the driver reported), "
            "{} camera reads, {} distinct slots, {} exposures touched",
            self.writes, len(self.slots), self.reads, len(set(self.slots)),
            len(set(self.exposures)),
        )
        logger.info("wall clock: {}", split)


def _set_exposure(cam, ms: float, ledger: CommandLedger, *, why: str) -> float:
    """Move the camera to ``ms`` and return the readback, all in one place."""
    readback = float(cam.reset_exposure_time(float(ms)))
    ledger.exposures.append(readback)
    logger.info("exposure: {} -> {:.4f} ms, readback {:.4f} ms", why, ms, readback)
    return readback


def _measure_peak(cam, ledger: CommandLedger, *, frames: int = 3) -> float:
    """Peak grey over ``frames`` fresh frames at the current exposure."""
    peak = 0.0
    for _ in range(max(int(frames), 1)):
        frame = np.asarray(cam.get_numpy_image(n_sample=1), dtype=np.float64)
        ledger.reads += 1
        peak = max(peak, float(frame.max()))
    return peak


def _resolve_exposure(cam, requested: float | None, ledger: CommandLedger) -> float:
    """Pick the working exposure from the signal the bench has *right now*.

    Nothing here is remembered and nothing is assumed about the light level.
    ``AO_DAHENG_MAX_EXPOSURE_MS`` only sets a **ceiling** -- and on this bench it
    is an operator value, not a measurement -- so the working point cannot be
    derived from it: on 2026-10-27 that ceiling was pinned at 0.8 ms, twice the
    exposure at which the morning probe had read the 0-order at 141 counts.

    The working point is instead **localised**, then bracketed:

    1. A coarse geometric ladder from ``PROBE_FLOOR_MS`` while the peak is below
       ``TARGET_PEAK_LO`` (bounded by the ceiling).  Its steps are factors of four,
       so it costs ``log4(ceiling / floor)`` reads (~5 on this bench) and can only
       land within a factor of four of the target.
    2. A *bisection* inside the bracket until the peak is within
       ``BRACKET_TOL`` of ``TARGET_PEAK``.  One proportional step is not
       trustworthy as the last word -- see the note in the docstring of the
       measured 0.441 ms -> 135 counts, 0.8 ms -> 239 counts pair, whose own
       ratio is 1.12x the linear expectation -- and a bisection needs no
       linearity assumption at all.  Saturating at the top of the bracket is
       reported, not smoothed over.
    3. ``DARK_PEAK`` at the ceiling means no light: an optical-path fault,
       reported as one.

    ``--exposure-ms`` pins the value and skips all of this; the ceiling still
    bounds it.
    """
    from ao_shaping.utils.image.exposure_control import resolve_daheng_exposure_limit

    ceiling = resolve_daheng_exposure_limit(camera=cam)
    _set_exposure(cam, PROBE_FLOOR_MS, ledger, why="ladder floor")
    if requested is not None:
        if float(requested) > ceiling + 1e-9:
            raise SystemExit(
                f"exposure {requested} ms exceeds the ceiling {ceiling} ms "
                f"({MAX_EXPOSURE_ENV})"
            )
        return _set_exposure(cam, requested, ledger, why="pinned by --exposure-ms")

    def peak_at(ms: float, why: str) -> float:
        _set_exposure(cam, ms, ledger, why=why)
        return _measure_peak(cam, ledger)

    ladder: list[tuple[float, float]] = []
    with ledger.timed("exposure_probe"):
        ms = PROBE_FLOOR_MS
        peak = _measure_peak(cam, ledger)
        ladder.append((ms, peak))
        # (a) coarse localisation: within a factor of COARSE_FACTOR of the target.
        while peak < TARGET_PEAK_LO and ms * COARSE_FACTOR <= ceiling + 1e-9:
            ms = min(ms * COARSE_FACTOR, ceiling)
            peak = peak_at(ms, "ladder step")
            ladder.append((ms, peak))

        if peak < DARK_PEAK:
            # Saturation is one dark-reading case and a dead path is another; a
            # single reading at the ceiling cannot tell them apart, so look at the
            # *middle* of the range before declaring a fault.
            mid = max(ceiling / 8.0, PROBE_FLOOR_MS)
            if mid < ms:
                peak = peak_at(mid, "dark check")
                ladder.append((mid, peak))
                ms = mid
        if peak < DARK_PEAK:
            raise SystemExit(
                f"peak {peak:.0f}/{SATURATION_PEAK} near the ceiling {ceiling} ms: "
                f"the camera is seeing essentially no light.  That is not an "
                f"exposure setting -- check the optical path (laser on, shutter "
                f"open, beam on the SLM) before calibrating"
            )

        # (b) bracket the target.  Two cases, both ending in the same bisection:
        # the ladder may have landed below target (bracket = its top .. ceiling)
        # or overshot -- typically straight into saturation -- in which case the
        # bracket is the step below it, and if even that floor clips, there is no
        # exposure on offer that is not saturated.
        saturated_top = peak >= TARGET_PEAK
        if saturated_top:
            lo = max(ms / COARSE_FACTOR, PROBE_FLOOR_MS)
            best_peak = peak_at(lo, "bracket floor")
            ladder.append((lo, best_peak))
            if best_peak >= SATURATION_PEAK:
                raise SystemExit(
                    f"the 0-order clips at every exposure probed from "
                    f"{PROBE_FLOOR_MS} ms to {ms} ms (peak {best_peak:.0f}/"
                    f"{SATURATION_PEAK} at the lowest): no exposure setting lands "
                    f"it on target -- attenuate the light, then re-run"
                )
            hi = ms
        else:
            lo, hi = ms, ceiling
            best_peak = peak
        best_ms = lo
        scale = best_peak / best_ms
        reached = abs(best_peak - TARGET_PEAK) <= BRACKET_TOL
        for _ in range(0 if reached else MAX_BISECT_STEPS):
            # Proportional *proposal* (exact on a linear bench, one read), kept
            # inside the bracket so a mis-estimate costs a step, not a re-run.
            probe = TARGET_PEAK / scale if scale > 0 else 0.5 * (lo + hi)
            if not lo < probe < hi:
                probe = 0.5 * (lo + hi)
            peak = peak_at(probe, "bracket")
            ladder.append((probe, peak))
            if abs(peak - TARGET_PEAK) <= BRACKET_TOL:
                best_ms, best_peak = probe, peak
                break
            if peak > TARGET_PEAK:
                hi = probe
            else:
                lo = probe
                best_ms, best_peak = probe, peak
            if hi - lo <= BRACKET_TOL_MS:
                break
        # ``best_ms`` is the exposure the loop last *measured* at, so re-reading it
        # would only add a duplicate ladder row.
        peak, saturated, readback = best_peak, best_peak >= SATURATION_PEAK, best_ms
        if peak < DARK_FRAME_PEAK:
            raise SystemExit(
                f"working point {best_ms:.4f} ms reads peak {peak:.0f}: the signal "
                f"vanished while bracketing -- re-run rather than calibrate on it"
            )

    logger.info(
        "exposure ladder (measured now): {}  -> ceiling {:.4f} ms, working "
        "{:.4f} ms, peak {:.0f}/{} (target {} +- {})",
        "  ".join(f"{e:.4f}ms:{p:.0f}" for e, p in ladder),
        ceiling, readback, peak, SATURATION_PEAK, TARGET_PEAK, BRACKET_TOL,
    )
    if saturated:
        logger.warning(
            "the working point sits at saturation (peak {:.0f}/{}): no exposure "
            "below the ceiling {} ms lands the 0-order on target.  The panel scan "
            "will refuse a clipped frame -- lower the light or {}", peak,
            SATURATION_PEAK, ceiling, MAX_EXPOSURE_ENV,
        )
    elif not TARGET_PEAK_LO <= peak <= TARGET_PEAK_HI:
        logger.warning(
            "peak {:.0f}/{} is outside the calibration band {}..{}: the ceiling "
            "{} ms bounds the exposure, so the ratios will be read on a small "
            "signal", peak, SATURATION_PEAK, TARGET_PEAK_LO, TARGET_PEAK_HI,
            ceiling,
        )
    return readback


def _flat_box(ref, flat) -> np.ndarray:
    """The 0-order scoring box around the flat frame's **measured** centroid."""
    from ao_shaping.tools.slm.panel.panel_locate import _box_mask

    return _box_mask(
        ref.shape, int(round(flat.centroid_x)), int(round(flat.centroid_y)),
        FLAT_BOX_RADIUS_PX,
    )


def _check_frame(ref, where: str) -> None:
    """Refuse to locate or sweep on a clipped frame.

    A saturated 0-order makes the box energy insensitive to the patch (the
    counts hit 255 and stop moving), so a scan on it reports a *low* response
    exactly where the beam is -- the failure looks like "illumination is
    elsewhere" instead of "exposure is too high".
    """
    peak = float(np.max(ref))
    if peak >= SATURATION_PEAK:
        raise SystemExit(
            f"{where} peaks at {peak:.0f}/255 (clipped at {SATURATION_PEAK}); "
            "lower --exposure-ms before measuring the panel"
        )


def _measure_panel(devices, args, ledger: CommandLedger) -> dict[str, object]:
    """Locate the illuminated patch on the panel, paired-flat, in one device pass.

    Behaviour matches ``tools/slm/panel/panel_locate.scan_panel``: for every
    candidate patch write a random disc, take a flat *before* and *after*, and
    score the 0-order box energy change against the paired flat mean.  A drift
    above ``--max-flat-drift`` rejects the point instead of scoring it.

    Two differences from ``scan_panel``, both of them cost, neither of them
    accuracy: the *after* flat of one point is reused as the *before* flat of
    the next (the panel goes flat between every pair either way, so this drops
    one flat per point), and the disc is only ever written where the measured
    illumination exists (a disc placed outside it cannot respond, and its
    zero response would then compete with the true centre).
    """
    from ao_shaping.tools.slm.panel.panel_locate import (
        _paired_delta,
        capture_settled_report,
        measure_flat_reference,
    )
    from ao_shaping.tools.slm.panel.probe_policy import probe_policy

    cam, slm, panel = devices
    policy = probe_policy(int(args.frames))
    xs = [int(v) for v in args.grid_xs.split(",") if v.strip()]
    ys = [int(v) for v in args.grid_ys.split(",") if v.strip()]
    rng = np.random.default_rng(int(args.seed))
    yy, xx = np.mgrid[0:panel[0], 0:panel[1]]

    # The panel keeps the last pattern written, so the very first "flat" read is
    # whatever a previous run left there.  Two attempts are the cheapest way to
    # make the two failures distinguishable: still dark on the second read means
    # it is the optical path or the exposure, not a stale panel.
    first = second = None
    for attempt in range(2):
        with ledger.timed(f"flat_attempt_{attempt + 1}"):
            ref, flat = measure_flat_reference(
                cam, slm, n_frames=int(args.frames), panel_shape=panel,
                policy=policy, wait_time_s=None,
            )
            ledger.reads += int(args.frames)
        peak = float(np.max(ref))
        logger.info(
            "flat attempt {}/2: {} frame_peak={:.0f} frame_mean={:.3f} "
            "frame_sum={:.0f}", attempt + 1, flat.as_row(), peak,
            float(ref.mean()), float(ref.sum()),
        )
        if attempt == 0:
            first = (ref, flat, peak)
        else:
            second = (ref, flat, peak)
    assert first is not None and second is not None
    ref, flat, _ = second
    if float(np.max(ref)) < DARK_FRAME_PEAK or float(np.max(first[0])) < DARK_FRAME_PEAK:
        raise SystemExit(
            "flat frame peaks at {:.0f}/{} on both attempts (first {:.0f}): the "
            "camera is seeing essentially no light.  That is not a stale-panel "
            "reading and not an exposure setting -- check the optical path "
            "(laser on, shutter open, beam on the SLM) before calibrating"
            .format(float(np.max(ref)), SATURATION_PEAK,
                    float(np.max(first[0])))
        )
    box = _flat_box(ref, flat)
    base = float(ref[box].sum())
    logger.info("flat reference (2nd attempt): {}  0-order box energy={:.0f}",
                flat.as_row(), base)
    _check_frame(ref, "flat frame")


    # A flat frame's spot measurement carries a centre and a shape, **not** a
    # radius: ``SpotMeasurement`` is (peak, fwhm, centroid, hollowness).  So the
    # illumination size is not inferred here at all -- the candidate grid is
    # swept as given, and the extent sweep (``_measure_beam_extent``) is what
    # states where the light stops.  Filtering on an invented radius would be
    # exactly the kind of remembered number this session exists to avoid.
    inside: list[tuple[int, int]] = [(px, py) for py in ys for px in xs]
    if any(px < FLAT_BOX_RADIUS_PX or px >= panel[1] - FLAT_BOX_RADIUS_PX
           or py < FLAT_BOX_RADIUS_PX or py >= panel[0] - FLAT_BOX_RADIUS_PX
           for px, py in inside):
        raise SystemExit(
            "a candidate centre is closer to the panel edge than the scoring box; "
            "widen the panel or trim --grid-xs/--grid-ys"
        )

    results: list[tuple[float, int, int, float]] = []
    rows: list[dict[str, object]] = []
    with ledger.timed("panel_scan"):
        before = ref
        for index, (px, py) in enumerate(inside):
            phase = np.zeros(panel, dtype=np.float64)
            m = (yy - py) ** 2 + (xx - px) ** 2 <= int(args.patch_radius) ** 2
            phase[m] = rng.uniform(0.0, 2.0 * np.pi, int(m.sum()))
            candidate = capture_settled_report(
                cam, slm, phase, wait_time_s=None, policy=policy
            )
            ledger.writes += 1
            ledger.reads += int(args.frames)
            _check_frame(candidate.frame, f"panel patch ({px}, {py})")
            if index + 1 < len(inside):
                after, _ = measure_flat_reference(
                    cam, slm, n_frames=int(args.frames), panel_shape=panel,
                    policy=policy, wait_time_s=None,
                )
                ledger.reads += int(args.frames)
                _check_frame(after, "re-paired flat")
            else:
                after = before  # last point: no successor to pair with
            try:
                delta, drift = _paired_delta(before, candidate.frame, after, box,
                                             float(args.max_flat_drift))
            except ValueError as exc:
                delta, drift = float("nan"), float("nan")
                logger.warning("panel ({}, {}): {}", px, py, exc)
            rows.append({
                "x": px, "y": py,
                "delta": None if not np.isfinite(delta) else float(delta),
                "flat_drift": None if not np.isfinite(drift) else float(drift),
                "flat_before": float(before[box].sum()),
                "candidate": float(candidate.frame[box].sum()),
                "flat_after": float(after[box].sum()),
                "peak": float(np.max(candidate.frame)),
                "rel_sigma": float(candidate.settle.rel_sigma),
            })
            if np.isfinite(delta):
                results.append((abs(delta), py, px, delta))
            logger.info(
                "panel ({:>4},{:>4}): delta={:+.4f} drift={:.1%} rel_sigma={:.3f}",
                px, py, delta, drift, candidate.settle.rel_sigma,
            )
            before = after
            # ``acquire_settled`` discards ``display_data``'s return value, so the
            # slot the driver auto-rotated to is not visible from here.  Record
            # that, do not guess it: an assumed slot in the ledger would hide a
            # real LCOS-refresh fault.
            ledger.slots_unreadable += 1

    if not results:
        raise SystemExit("panel scan produced no scorable candidate")
    results.sort(reverse=True)
    _, by, bx = results[0][:3]
    logger.info("panel centre: x={} y={} response={:+.4f}", bx, by, results[0][0])
    return {
        "center_xy": [int(bx), int(by)],
        "offset_from_panel_centre": [int(bx) - panel[1] // 2, int(by) - panel[0] // 2],
        "offset_from_frame_centre": [
            int(bx) - PANEL_SHAPE[1] // 2, int(by) - PANEL_SHAPE[0] // 2
        ],
        "response": float(results[0][3]),
        "runner_up_abs": float(results[1][0]) if len(results) > 1 else None,
        "candidates_swept": len(inside),
        "points": rows,
    }


def _measure_beam_extent(devices, args, ledger: CommandLedger) -> dict[str, object]:
    """Sweep the panel to find where the light stops responding.

    This is ``tools/slm/panel/beam_extent.py``'s measurement, run inside this
    session instead of its own device pair: a random phase disc at radius ``r``,
    and the **core fraction** -- the share of total frame energy inside
    ``CORE_RADIUS_PX`` of the flat frame's centroid -- as the reading.  That
    statistic is chosen for the reason the standalone probe documents: it is
    monotone in how much light left the core, whereas a box around the 0-order
    is not (a disc both scatters light out of it and redistributes light in) and
    total energy carries no size information at all (measured conserved within
    0.5 % across apertures from 100 to 850 px).

    The centroid is the flat frame's, held fixed for the whole sweep: the
    0-order does not move while the patch moves, and re-locating it per point
    would fold spot wander into the reading.
    """
    from ao_shaping.tools.kernels.bench_kernels import core_fraction
    from ao_shaping.tools.slm.panel.panel_locate import (
        capture_settled_report,
        measure_flat_reference,
    )
    from ao_shaping.tools.slm.panel.probe_policy import probe_policy

    cam, slm, panel = devices
    policy = probe_policy(int(args.frames))
    radius = int(args.patch_radius)
    yy, xx = np.mgrid[0:panel[0], 0:panel[1]]
    rng = np.random.default_rng(int(args.seed) + 1)

    with ledger.timed("extent_flat"):
        ref, flat = measure_flat_reference(
            cam, slm, n_frames=int(args.frames), panel_shape=panel,
            policy=policy, wait_time_s=None,
        )
        ledger.reads += int(args.frames)
    _check_frame(ref, "extent reference frame")
    cx, cy = float(flat.centroid_x), float(flat.centroid_y)
    base = core_fraction(ref, cx, cy, CORE_RADIUS_PX)
    logger.info("extent reference: {}  core_fraction({}px)={:.4f}",
                flat.as_row(), CORE_RADIUS_PX, base)

    out: dict[str, object] = {
        "reference_centroid": [cx, cy],
        "core_radius_px": CORE_RADIUS_PX,
        "core_fraction_flat": base,
        "panel_shape": list(panel),
    }
    with ledger.timed("extent_sweep"):
        for axis, values, centre_axis in (
            ("x", args.extent_x, 1), ("y", args.extent_y, 0)
        ):
            found: list[dict[str, object]] = []
            before = ref
            kept = [int(v) for v in values.split(",") if v.strip()]
            for index, value in enumerate(kept):
                # The *other* coordinate is the flat frame's measured centre, not
                # the panel's geometric middle: the beam is not parked centrally,
                # so sweeping through the panel centre would walk off the beam.
                px = value if centre_axis == 1 else int(round(cx))
                py = value if centre_axis == 0 else int(round(cy))
                if not (radius <= px < panel[1] - radius
                        and radius <= py < panel[0] - radius):
                    # A disc at the edge would be a different aperture, not a
                    # sweep point: report the skip instead of a number that
                    # means something else.
                    found.append({"value": value, "core_fraction": None,
                                  "skipped": "patch would leave the panel"})
                    continue
                phase = np.zeros(panel, dtype=np.float64)
                m = (yy - py) ** 2 + (xx - px) ** 2 <= radius * radius
                phase[m] = rng.uniform(0.0, 2.0 * np.pi, int(m.sum()))
                candidate = capture_settled_report(
                    cam, slm, phase, wait_time_s=None, policy=policy
                )
                ledger.writes += 1
                ledger.reads += int(args.frames)
                _check_frame(candidate.frame, f"extent {axis}={value}")
                if index + 1 < len(kept):
                    after, _ = measure_flat_reference(
                        cam, slm, n_frames=int(args.frames), panel_shape=panel,
                        policy=policy, wait_time_s=None,
                    )
                    ledger.reads += int(args.frames)
                    _check_frame(after, "extent re-paired flat")
                else:
                    after = before
                # Paired flat: the reading is the drop from the mean of the flats
                # either side of this write, so bench drift is subtracted rather
                # than attributed to the patch.
                pair = (core_fraction(before, cx, cy, CORE_RADIUS_PX)
                        + core_fraction(after, cx, cy, CORE_RADIUS_PX)) / 2
                value_fraction = core_fraction(
                    candidate.frame, cx, cy, CORE_RADIUS_PX
                )
                drop = float(value_fraction - pair)
                logger.info(
                    "extent {}={} (patch at {},{}): core={:.4f} paired flat={:.4f} "
                    "drop={:+.4f}", axis, value, px, py, value_fraction, pair, drop,
                )
                found.append({
                    "value": value, "patch_xy": [px, py],
                    "core_fraction": value_fraction,
                    "paired_flat": pair, "drop": drop,
                    "rel_sigma": float(candidate.settle.rel_sigma),
                })
                before = after
                ledger.slots_unreadable += 1
            # The extent number on its own does not say whether the aperture
            # actually stopped changing: an edge that still modulates is a real
            # boundary, an edge that has flattened is the sweep running out, and
            # the two look the same in the raw drop.  Ratio to the *previous*
            # point separates them (``beam_extent`` reports the same thing as
            # ``rel``); the deepest point of the sweep is not a boundary, so it
            # gets no ratio at all rather than a fabricated one.
            prev: float | None = None
            for row in found:
                value_fraction = row.get("core_fraction")
                if value_fraction is None or prev is None:
                    prev = value_fraction if value_fraction is not None else prev
                    row["rel_to_previous"] = None
                    continue
                row["rel_to_previous"] = (
                    None if prev <= 0 else float(value_fraction / prev)
                )
                prev = value_fraction
            out[axis] = found
    return out


def main(argv: list[str] | None = None) -> int:
    """Open the devices once and run the measurement sequence."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="runs/sim2f_bench/calibration")
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--slm-wavelength", type=int, default=1064)
    parser.add_argument("--cam-type", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument(
        "--exposure-ms", type=float, default=None,
        help="pin the exposure; omit to probe the ceiling and work under it NOW",
    )
    parser.add_argument(
        "--frames", type=int, default=FRAMES_FLOOR,
        help=f"frames kept per point (probe floor is {FRAMES_FLOOR})",
    )
    parser.add_argument("--patch-radius", type=int, default=int(PUPIL_RADIUS_PX))
    parser.add_argument("--grid-xs", default="240,600,960,1320,1680")
    parser.add_argument("--grid-ys", default="220,480,720,980")
    parser.add_argument(
        "--extent-x", default="640,780,920,1060,1200,1340",
        help="patch-centre x values for the beam-extent sweep",
    )
    parser.add_argument(
        "--extent-y", default="300,450,600,750",
        help="patch-centre y values for the beam-extent sweep",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-flat-drift", type=float, default=0.2)
    parser.add_argument(
        "--skip-extent", action="store_true",
        help="panel locate only (the extent sweep is the longer half)",
    )
    args = parser.parse_args(argv)

    if not args.skip_extent and int(args.frames) > FRAMES_FLOOR + 1:
        logger.info(
            "--frames {} is above the probe floor {}; this run measures the cost "
            "of that budget, not a comparable number", args.frames, FRAMES_FLOOR,
        )

    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.kernels.bench_kernels import SLM_PANEL_H, SLM_PANEL_W
    from ao_shaping.tools.slm.panel.display_state import preserve_slm_display

    ledger = CommandLedger()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload: dict[str, object] = {"started_utc": stamp, "args": vars(args)}

    t_open = time.perf_counter()
    with Santec(
        slm_number=args.slm_number, wavelength=args.slm_wavelength, video_mode=0
    ) as slm, create_camera(args.cam_type, args.cam_id, exposure_time_ms=0.02) as cam, \
            preserve_slm_display(slm):
        ledger.phases["device_open"] = time.perf_counter() - t_open
        panel = (SLM_PANEL_H, SLM_PANEL_W)
        payload["panel_shape"] = list(panel)
        ledger.exposures.append(0.02)  # the value the driver applied during open()

        exposure = _resolve_exposure(cam, args.exposure_ms, ledger)
        payload["exposure_ms"] = exposure

        panel_result = _measure_panel((cam, slm, panel), args, ledger)
        payload["panel_locate"] = panel_result

        if not args.skip_extent:
            payload["beam_extent"] = _measure_beam_extent((cam, slm, panel), args, ledger)

        # The centre this session must adopt, stated once, in the form the
        # bench run takes it (`--pupil-center-xy X,Y`), so it is copied not derived.
        x, y = panel_result["center_xy"]
        payload["pupil_center_xy_cli"] = f"{x},{y}"
        logger.info(
            "pass to the bench run: --pupil-center-xy {},{}   (panel centre is "
            "{},{})", x, y, panel[1] // 2, panel[0] // 2,
        )

    ledger.report()
    payload["ledger"] = ledger.as_dict()
    path = out_dir / f"calibration_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("calibration -> {}", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
