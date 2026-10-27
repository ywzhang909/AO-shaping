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


@dataclass
class CommandLedger:
    """Every command and every frame, counted per phase.

    ``writes`` counts SLM panel writes, ``reads`` counts camera captures and
    ``slots`` the memory-slot rotations (a rotation is what makes the LCOS
    refresh -- see ``drivers/slm/AGENTS.md``).  ``exposures`` records each
    exposure the camera was moved through, so a ladder is visible as its steps.
    """

    writes: int = 0
    reads: int = 0
    slots: list[int] = field(default_factory=list)
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
            "exposure_values_touched": sorted(set(self.exposures)),
            "seconds_by_phase": {k: round(v, 3) for k, v in self.phases.items()},
        }

    def report(self) -> None:
        """Log the counts and the wall-clock split."""
        split = "  ".join(f"{k}={v:.1f}s" for k, v in self.phases.items())
        logger.info(
            "commands: {} SLM writes, {} camera reads, {} slot rotations "
            "({} distinct), {} exposures touched",
            self.writes, self.reads, len(self.slots), len(set(self.slots)),
            len(set(self.exposures)),
        )
        logger.info("wall clock: {}", split)


def _resolve_exposure(cam, requested: float | None, ledger: CommandLedger) -> float:
    """Set the camera exposure, measuring the ceiling when none is given.

    ``None`` means "the operator did not pin it", which is **not** the same as
    reusing a remembered value: the ceiling is probed on this path right now and
    the working point is taken from it.  An explicit number is honoured and
    still validated against the measured ceiling.
    """
    from ao_shaping.utils.image.exposure_control import (
        resolve_daheng_exposure_limit,
        working_exposure_ms,
    )

    ceiling = resolve_daheng_exposure_limit(camera=cam)
    ledger.exposures.append(float(cam.reset_exposure_time(0.02)))
    if requested is None:
        # The ladder runs inside ``resolve_daheng_exposure_limit`` (it writes the
        # flat and reads the peak itself), so the measurement that picks the
        # exposure happens *here*, now -- nothing is remembered from a past run.
        with ledger.timed("exposure_probe"):
            chosen = working_exposure_ms(ceiling)
        logger.info(
            "exposure: measured ceiling {:.4f} ms -> working {:.4f} ms "
            "(half the ceiling, by ``working_exposure_ms``)", ceiling, chosen,
        )
    else:
        chosen = float(requested)
    if chosen > ceiling + 1e-9:
        raise SystemExit(
            f"exposure {chosen} ms exceeds the measured ceiling {ceiling} ms "
            f"({MAX_EXPOSURE_ENV})"
        )
    readback = cam.reset_exposure_time(chosen)
    ledger.exposures.append(float(readback))
    logger.info(
        "exposure: ceiling {:.4f} ms (measured now), working {:.4f} ms, "
        "readback {:.4f} ms", ceiling, chosen, readback,
    )
    return float(readback)


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

    with ledger.timed("panel_flat"):
        ref, flat = measure_flat_reference(
            cam, slm, n_frames=int(args.frames), panel_shape=panel,
            policy=policy, wait_time_s=None,
        )
        ledger.reads += int(args.frames)
    box = _flat_box(ref, flat)
    base = float(ref[box].sum())
    logger.info("flat reference: {}  0-order box energy={:.0f}", flat.as_row(), base)
    _check_frame(ref, "flat frame")

    # Only candidates at least a patch radius inside the illuminated disc: a disc
    # straddling the edge writes part of its phase where there is no light, so
    # its response is a *fraction* of a real patch's and it biases the winner.
    inside: list[tuple[int, int]] = []
    skipped: list[tuple[int, int]] = []
    for py in ys:
        for px in xs:
            reach = np.hypot(px - flat.centroid_x, py - flat.centroid_y)
            (inside if reach + int(args.patch_radius) <= flat.radius_px
             else skipped).append((px, py))
    if skipped:
        # A skipped cell is not "the illumination stops here": the illumination
        # estimate itself may be wrong.  Prove it with the extent sweep before
        # believing either, and say so.
        logger.warning(
            "{} of {} candidates sit outside the measured illumination "
            "(centre {:.0f},{:.0f} radius {:.0f} px from the flat frame): {}",
            len(skipped), len(xs) * len(ys), flat.centroid_x, flat.centroid_y,
            flat.radius_px, skipped,
        )
        logger.warning(
            "that estimate is not trusted as the answer to "
            "'where is the beam' -- confirm it with the extent sweep "
            "(run without --skip-extent) before running the bench"
        )
    if not inside:
        raise SystemExit(
            "no candidate patch fits inside the measured illumination; the flat "
            "frame's centroid/radius disagree with the grid -- lower the "
            "exposure or widen --grid-xs/--grid-ys"
        )

    results: list[tuple[float, int, int, float]] = []
    rows: list[dict[str, object]] = []
    candidates = skipped
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
            ledger.slots.append(int(candidate.writes_slot))
    _ = candidates

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
        "inside_illumination": len(inside),
        "skipped_outside_illumination": skipped,
        "points": rows,
    }


def _measure_beam_extent(devices, args, ledger: CommandLedger) -> dict[str, object]:
    """Sweep the panel edge to find where the beam stops responding.

    Same question as ``tools/slm/panel/beam_extent.py`` (does a random phase disc
    at radius ``r`` still change the far field?) asked with the cheaper statistic
    this session already needs: the 0-order box energy change against a paired
    flat.  A radius past the illuminated region gives a response at the floor.

    The box here is the *patch's* own 60 px circle around the fixed 0-order
    centroid, not a fresh centroid estimate: the 0-order does not move while the
    patch moves, and re-locating it per point would fold spot wander into the
    response.
    """
    from ao_shaping.tools.slm.panel.panel_locate import _box_mask
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
    box = _box_mask(ref.shape, int(round(flat.centroid_x)),
                    int(round(flat.centroid_y)), FLAT_BOX_RADIUS_PX)
    reference = float(ref[box].sum())

    out: dict[str, object] = {
        "reference_centroid": [float(flat.centroid_x), float(flat.centroid_y)],
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
                cx = value if centre_axis == 1 else panel[1] // 2
                cy = value if centre_axis == 0 else panel[0] // 2
                if not (radius <= cx < panel[1] - radius
                        and radius <= cy < panel[0] - radius):
                    # A disc at the edge would be a different aperture, not a
                    # sweep point: report the skip instead of a number that
                    # means something else.
                    found.append({"axis": axis, "value": value, "delta": None,
                                  "skipped": "patch would leave the panel"})
                    continue
                phase = np.zeros(panel, dtype=np.float64)
                m = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
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
                delta = (float(candidate.frame[box].sum())
                         - (float(before[box].sum()) + float(after[box].sum())) / 2)
                delta /= reference if reference > 0 else float("nan")
                logger.info("extent {}={}: delta={:+.4f}", axis, value, delta)
                found.append({"axis": axis, "value": value, "delta": float(delta)})
                before = after
                ledger.slots.append(int(candidate.writes_slot))
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
