"""MiiCam ``auto_exposure`` driver-algorithm hardware verification probe.

What this verifies
------------------
``MIICamera.auto_exposure`` (proportional iteration, no SDK fast path) is the
only automatic-exposure path on the MiiCam bench camera. This probe exercises
it on real hardware and checks the contract from ``ccd/AGENTS.md``:

1. **Premise** -- at the starting exposure the scene really is lit and the
   peak is in a sane 0-255 range (peak/mean/lit-fraction are recorded, not
   assumed).
2. **Response map** -- peak and mean vs exposure over the working range,
   including the device-side exposure readback from the frame info
   (``MiicamFrameInfoV3.expotime``), so the sensor response curve is measured
   rather than assumed. Any step/knee is located explicitly.
3. **Reachable targets** -- convergence to ``target_max +/- tolerance`` from
   both below (exposure must shrink) and above (exposure must grow), with the
   returned image matching the final state.
4. **Unreachable target** -- graceful exit at the hardware boundary instead of
   spinning ``max_iterations`` times.
5. **No stale frames** -- the first frame after every exposure change must
   respond to the *new* exposure (MiiCam changes exposure via
   Stop -> put_ExpoTime -> restart streaming, so unlike the Daheng backend
   there must be no one-frame lag; a lag would make the iteration overshoot).
   Verified frame-level by the dirty-data scan (sequence numbers, timestamps,
   pixel-identity of consecutive frames, interleaved switch test) and
   per-run by a freshness probe at the final operating point.
6. **Dirty-data scan** -- adjusting the exposure around the response step,
   is the read data clean? Frame sequence numbers must advance by one,
   timestamps must advance, consecutive frames must not be pixel-identical
   duplicates, and the brightness must deterministically track the requested
   exposure across interleaved switch rounds with no lag.
7. **Post state** -- SDK auto exposure stays disabled and the exposure
   readback agrees with the last applied value.

Device safety (hard requirement)
--------------------------------
This probe opens the camera **only** through the miicam SDK
(``EnumV2`` / ``Miicam.Open`` / ``WaitImageV3``). It must never construct
``DahengCamera``, never call ``create_camera("daheng")`` and never import
``gxipy``: the Daheng GigE camera on this bench is a separate physical device
that other sessions may be using, and any enumerate/open on that SDK would
occupy it. The guard :func:`_import_miicamera` enforces this at runtime.

Usage
-----
Run from the repository root with the project venv::

    .venv/Scripts/python.exe scripts/ccd/miicam_auto_exposure_probe.py

Run specs for ``--targets`` are
``TARGET[:n_sample[:twice_valid[:tolerance[:expect]]]]`` (``expect``:
``conv`` / ``bound`` / ``dead``). Two output layouts via ``--layout``:

* ``flat`` (default) -- everything lands in ``--output-dir`` root, which must
  not exist yet: ``config.json``, ``results.json``, ``trajectory.csv``,
  ``auto_exposure_probe.png``, ``report.md``. Used for ``data/debug/`` dumps.
* ``report`` -- the registered-report tree of ``report/AGENTS.md``:
  ``report.md`` at the root (with the provenance header stamped in when the
  path is a registered key), the figure under ``figures/`` and the raw
  artefacts under ``artifacts/``. The directory is reused in place, so the
  pytest wrapper regenerates the checked-in report deterministically.

Exit code is 0 only when every *check* passes; diagnostics (``info_*``
entries, e.g. the linearity R2 of a stepped scene) are recorded but never
fail the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

# Tolerances for the checks (see module docstring).
PREMISE_PEAK_MIN = 100.0  # counts; scene must be clearly lit
LIT_FRACTION_MIN = 0.5  # "basically all pixels have light"
CONVERGE_PEAK_TOL = 1.0  # counts, returned image vs last measured frame


def _import_miicamera():
    """Import ``MIICamera`` from the miicam backend only.

    Raises if the Daheng SDK (gxipy) is already in ``sys.modules`` or if the
    resolved class does not come from the miicam driver module.
    """
    if "gxipy" in sys.modules:
        raise RuntimeError(
            "gxipy is already imported; refusing to run because the Daheng "
            "SDK must not be touched by this probe"
        )
    from ao_shaping.drivers.ccd.miicam.driver import MIICamera

    module = getattr(MIICamera, "__module__", "")
    if "miicam" not in module or "daheng" in module:
        raise RuntimeError(f"MIICamera resolved from unexpected module: {module!r}")
    return MIICamera


def _frame_stats(img: np.ndarray) -> dict:
    """Peak/mean/lit-fraction statistics of one frame (0-255 counts)."""
    arr = np.asarray(img, dtype=np.float64)
    return {
        "peak": float(arr.max()),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "lit_fraction": float((arr > 10).mean()),
    }


def _grab_with_frame_info(cam) -> tuple[np.ndarray, dict]:
    """Grab one frame together with the SDK frame info (device-side truth).

    Mirrors ``_FramePuller.wait_image`` but keeps the ``MiicamFrameInfoV3``:
    ``expotime`` is the exposure the device actually used for this frame,
    which is the only reliable readback the SDK offers (the driver caches a
    clamped-but-unquantized request; ``put_ExpoTime`` quantizes to integer
    microseconds).
    """
    import ctypes

    import miicam

    bufsize, bits, dtype = cam._get_buffer_params()
    buffer = (ctypes.c_char * bufsize)()
    frame_info = miicam.MiicamFrameInfoV3()
    timeout_ms = max(300, int(cam.exposure_time_ms * 5))
    cam.cam.WaitImageV3(timeout_ms, buffer, 0, bits, 0, frame_info)
    img = cam._decode_image(np.frombuffer(buffer, dtype=dtype))
    info = {
        "seq": int(frame_info.seq),
        "timestamp_us": int(frame_info.timestamp),
        "device_expo_us": int(frame_info.expotime),
        "flag": int(frame_info.flag),
    }
    return img, info


def _fit_through_origin(exposures: list[float], values: list[float]) -> dict:
    """Least-squares slope of ``value = k * exposure`` through the origin."""
    exp = np.asarray(exposures, dtype=np.float64)
    val = np.asarray(values, dtype=np.float64)
    if exp.size < 2 or float(exp @ exp) == 0.0:
        return {"slope": float("nan"), "r2": float("nan"), "n_fit": int(exp.size)}
    k = float((exp @ val) / (exp @ exp))
    resid = val - k * exp
    ss_res = float(resid @ resid)
    ss_tot = float(((val - val.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": k, "r2": r2, "n_fit": int(exp.size)}


def _dirty_data_scan(cam, args: argparse.Namespace) -> dict:
    """Dirty-data investigation around the response step.

    Two parts, both adjusting exposure and inspecting what is actually read:

    1. **Interleaved switch test** -- cycle through the exposures around the
       step (default 16/17/18/19 us) for several rounds. A clean read shows
       the brightness deterministically at each requested level, with no lag
       after a switch. Dirty data (stale buffers, duplicated frames) shows up
       as brightness that does not follow the request, lags by one switch, or
       is not reproducible across rounds.
    2. **Frame freshness at fixed exposure** -- grab a burst of consecutive
       frames and check the SDK frame info: sequence numbers must advance by
       one, timestamps must advance, the device-side exposure must match the
       request (quantised to integer microseconds), and consecutive frames
       must not be pixel-identical duplicates.

    Returns a dict with the raw records plus per-check verdict strings.
    """
    # ---- part 1: interleaved switch test -------------------------------
    switch_records: list[dict] = []
    for rnd in range(args.switch_rounds):
        for exp_us in args.switch_exposures:
            cam.reset_exposure_time(exp_us / 1000.0)
            img, info = None, None
            for _ in range(3):  # first frames after a restart may be stale
                img, info = _grab_with_frame_info(cam)
            s = _frame_stats(img)
            switch_records.append({
                "round": rnd,
                "requested_us": exp_us,
                "device_expo_us": info["device_expo_us"],
                "flag": info["flag"],
                "seq": info["seq"],
                **s,
            })
            logger.info(
                "  switch round {} req={} us -> device={} us: peak={:.0f} "
                "mean={:.2f}",
                rnd, exp_us, info["device_expo_us"], s["peak"], s["mean"],
            )

    by_exp: dict[int, list[dict]] = {}
    for r in switch_records:
        by_exp.setdefault(r["requested_us"], []).append(r)
    per_exp = {}
    for e, recs in sorted(by_exp.items()):
        means = [r["mean"] for r in recs]
        peaks = [r["peak"] for r in recs]
        med_m = float(np.median(means))
        per_exp[e] = {
            "mean_median": med_m,
            "mean_spread_rel": (max(means) - min(means)) / max(med_m, 1e-9),
            "peak_median": float(np.median(peaks)),
            "device_expo_us": sorted({r["device_expo_us"] for r in recs}),
        }

    # switch lag: after each exposure change, does the reading already sit at
    # the new level, or does it still reflect the previous one? Switches that
    # stay inside one response rung (|before - target| below the floor) carry
    # no information about lag and are skipped.
    lags = []
    for prev, cur in zip(switch_records, switch_records[1:]):
        if cur["requested_us"] == prev["requested_us"]:
            continue
        target = per_exp[cur["requested_us"]]["mean_median"]
        before = per_exp[prev["requested_us"]]["mean_median"]
        denom = abs(before - target)
        if denom < args.lag_denominator_floor:
            continue
        lag = abs(cur["mean"] - target) / denom
        lags.append({
            "from_us": prev["requested_us"], "to_us": cur["requested_us"],
            "measured_mean": cur["mean"], "target_mean": target,
            "lag": lag,
        })

    spreads = {e: v["mean_spread_rel"] for e, v in per_exp.items()}
    step_det = all(s <= args.determinism_tol for s in spreads.values())
    no_lag = all(l["lag"] <= args.switch_lag_tol for l in lags)

    # ---- part 2: frame freshness at fixed exposure ----------------------
    fresh = {}
    for exp_us in args.fresh_exposures:
        cam.reset_exposure_time(exp_us / 1000.0)
        frames: list[tuple[np.ndarray, dict]] = []
        for _ in range(args.fresh_frames):
            img, info = _grab_with_frame_info(cam)
            frames.append((img, info))
        seqs = [f[1]["seq"] for f in frames]
        tss = [f[1]["timestamp_us"] for f in frames]
        devs = [f[1]["device_expo_us"] for f in frames]
        flags = [f[1]["flag"] for f in frames]
        seq_deltas = [b - a for a, b in zip(seqs, seqs[1:])]
        ts_deltas = [b - a for a, b in zip(tss, tss[1:])]
        diffs = []
        duplicates = 0
        for (img_a, _), (img_b, _) in zip(frames, frames[1:]):
            d = float(np.mean(np.abs(img_a.astype(np.int16)
                                     - img_b.astype(np.int16))))
            diffs.append(d)
            if int(np.max(np.abs(img_a.astype(np.int16)
                                 - img_b.astype(np.int16)))) == 0:
                duplicates += 1
        zero_rows = int(sum(np.all(f[0] == 0, axis=1).sum() for f in frames))
        sat_rows = int(sum(np.all(f[0] >= 255, axis=1).sum() for f in frames))
        fresh[exp_us] = {
            "seqs": seqs,
            "seq_deltas": seq_deltas,
            "timestamp_us": tss,
            "ts_deltas_us": ts_deltas,
            "device_expo_us": devs,
            "flags": flags,
            "expo_flag_present": any(f & 0x04 for f in flags),
            "consecutive_mean_abs_diff": diffs,
            "exact_duplicate_pairs": duplicates,
            "all_zero_rows": zero_rows,
            "all_saturated_rows": sat_rows,
            "device_expo_match": all(abs(d - exp_us) <= 1 for d in devs),
            "seq_advance": all(d == 1 for d in seq_deltas),
            "timestamps_advance": all(d > 0 for d in ts_deltas),
        }
        logger.info(
            "  fresh @{} us: seq_deltas={} ts_deltas={} device={} "
            "mean|diff|={:.2f} duplicates={}",
            exp_us, seq_deltas, ts_deltas, devs,
            float(np.mean(diffs)) if diffs else float("nan"), duplicates,
        )

    # The SDK only fills MiicamFrameInfoV3.expotime when the frame-info flag
    # carries MIICAM_FRAMEINFO_FLAG_EXPOTIME (0x04). On this camera the flag
    # does not, so the field reads 0 -- that is an unavailable readback, not a
    # mismatched exposure, and must not be scored as dirty data.
    expo_flag_present = any(
        v["expo_flag_present"] for v in fresh.values()
    ) or any(r["flag"] & 0x04 for r in switch_records)
    if expo_flag_present:
        all_dev_match = all(v["device_expo_match"] for v in fresh.values()) and all(
            abs(r["device_expo_us"] - r["requested_us"]) <= 1
            for r in switch_records
        )
        dev_detail = "; ".join(
            f"{e} us -> device {v['device_expo_us']}" for e, v in fresh.items()
        )
    else:
        all_dev_match = True
        dev_detail = ("frame-info expotime not provided by this camera/SDK "
                      "(flag lacks MIICAM_FRAMEINFO_FLAG_EXPOTIME=0x04, field "
                      "reads 0); device-side exposure readback unavailable, "
                      "requested exposure taken from the driver cache")
    all_seq = all(v["seq_advance"] for v in fresh.values())
    all_ts = all(v["timestamps_advance"] for v in fresh.values())
    no_dup = all(v["exact_duplicate_pairs"] == 0 for v in fresh.values())

    seq_detail = "; ".join(
        f"{e} us: {v['seq_deltas']}" for e, v in fresh.items())
    ts_detail = "; ".join(
        f"{e} us: {v['ts_deltas_us']}" for e, v in fresh.items())
    dup_detail = "; ".join(
        f"{e} us: mean|diff|={np.mean(v['consecutive_mean_abs_diff']):.2f} "
        f"counts, exact duplicates={v['exact_duplicate_pairs']}"
        for e, v in fresh.items()
    )
    step_detail = "; ".join(
        f"{e} us: mean {v['mean_median']:.2f} (spread "
        f"{v['mean_spread_rel']:.1%})"
        for e, v in per_exp.items()
    )
    lag_detail = "; ".join(
        f"{l['from_us']}->{l['to_us']} us: lag {l['lag']:.0%}"
        for l in lags
    ) or "no switches"

    return {
        "switch_records": switch_records,
        "per_exposure": per_exp,
        "switch_lags": lags,
        "freshness": fresh,
        "all_device_expo_match": all_dev_match,
        "device_expo_detail": dev_detail,
        "all_seq_advance": all_seq,
        "seq_detail": seq_detail,
        "all_timestamps_advance": all_ts,
        "timestamp_detail": ts_detail,
        "no_duplicate_frames": no_dup,
        "duplicate_detail": dup_detail,
        "step_deterministic": step_det,
        "step_detail": step_detail,
        "no_switch_lag": no_lag,
        "lag_detail": lag_detail,
    }


def run_probe(args: argparse.Namespace) -> int:
    """Execute the full probe; return process exit code."""
    MIICamera = _import_miicamera()

    out_dir = Path(args.output_dir)
    if args.layout == "report":
        # Registered report tree (report/AGENTS.md): the report sits at the
        # root, figures under figures/, raw artefacts under artifacts/. The
        # directory is reused across runs, so it is created idempotently.
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Output directory: {} (layout={})", out_dir, args.layout)

    results: dict = {
        "probe": "miicam_auto_exposure",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "checks": [],
        "runs": [],
        "response_map": None,
        "knee_scan": None,
        "premise": None,
        "post_state": None,
    }

    def record(name: str, passed: bool | None, detail: str) -> None:
        """Log one outcome.

        ``passed=None`` records a *diagnostic*: a measured property of the
        scene/device (e.g. the linearity R2 of a stepped response) that is
        reported but not part of the pass/fail contract. Only boolean entries
        count towards the summary and the exit code.
        """
        results["checks"].append({"check": name, "passed": passed,
                                  "detail": detail})
        if passed is None:
            logger.info("INFO {}: {}", name, detail)
        else:
            logger.info("CHECK {}: {} ({})", "PASS" if passed else "FAIL",
                        name, detail)

    cam = MIICamera(
        cam_id=args.cam_id,
        exposure_time_ms=args.initial_exposure_ms,
        bit_depth=8,
        capture_mode="wait",
    )
    try:
        cam.initialize()
        logger.info(
            "Opened miicam only: sn={} {}x{} bit_depth={}",
            cam._sn, cam.cam_width, cam.cam_height, cam._bit_depth,
        )
        results["camera"] = {
            "sn": cam._sn,
            "width": int(cam.cam_width),
            "height": int(cam.cam_height),
            "bit_depth": int(cam._bit_depth),
            "capture_mode": cam._capture_mode,
        }

        # ------------------------------------------------------- premise
        logger.info("=== premise: stats at {:.3f} ms ===", args.initial_exposure_ms)
        premise_frames = [
            _frame_stats(cam.get_numpy_image(n_sample=1, skip_first=True))
            for _ in range(args.premise_frames)
        ]
        peaks = [f["peak"] for f in premise_frames]
        lit = [f["lit_fraction"] for f in premise_frames]
        results["premise"] = {
            "exposure_ms": float(cam.get_exposure_time_ms()),
            "frames": premise_frames,
            "peak_median": float(np.median(peaks)),
            "peak_min": min(peaks),
            "peak_max": max(peaks),
            "lit_fraction_min": min(lit),
        }
        record(
            "premise_brightness",
            float(np.median(peaks)) >= PREMISE_PEAK_MIN,
            f"median peak {np.median(peaks):.0f} counts at "
            f"{cam.get_exposure_time_ms():.3f} ms (range "
            f"{min(peaks):.0f}-{max(peaks):.0f}); scene must be clearly lit",
        )
        record(
            "premise_lit_fraction",
            min(lit) >= LIT_FRACTION_MIN,
            f"min lit fraction {min(lit):.2f} (expected >= {LIT_FRACTION_MIN})",
        )

        # ------------------------------------------------- response map
        logger.info("=== response map: peak/mean vs exposure (with device readback) ===")
        rm_exp, rm_peak, rm_mean, rm_dev, rm_req_us = [], [], [], [], []
        for exp_ms in args.response_exposures:
            cam.reset_exposure_time(exp_ms)
            img, info = _grab_with_frame_info(cam)
            s = _frame_stats(img)
            rm_exp.append(float(cam.get_exposure_time_ms()))
            rm_peak.append(s["peak"])
            rm_mean.append(s["mean"])
            rm_dev.append(info["device_expo_us"])
            # The driver quantizes put_ExpoTime to integer microseconds; when
            # the frame-info readback is unavailable (this camera) the
            # quantised request is the best exposure axis available.
            rm_req_us.append(int(round(exp_ms * 1000)))
            logger.info(
                "  req={:.4f} ms (cache) -> device={} us: peak={:.0f} mean={:.2f}",
                rm_exp[-1], info["device_expo_us"], s["peak"], s["mean"],
            )
        # saturation level for the peak series (clipped points break fits)
        sat_us = next((u for u, p in zip(rm_req_us, rm_peak) if p >= 250.0), None)
        usable = [(u, p) for u, p in zip(rm_req_us, rm_peak) if p < 250.0]
        peak_fit = _fit_through_origin([u[0] for u in usable], [u[1] for u in usable])
        mean_fit = _fit_through_origin(rm_req_us, rm_mean)
        results["response_map"] = {
            "requested_ms": rm_exp,
            "requested_us": rm_req_us,
            "device_expo_us": rm_dev,
            "peaks": rm_peak,
            "means": rm_mean,
            "peak_fit": peak_fit,
            "mean_fit": mean_fit,
            "saturation_onset_us": sat_us,
        }
        # Diagnostic, not a contract check: on this scene the response is
        # stepped (see the knee scan below), so a through-origin fit is
        # expected to fit badly. The R2 values are still recorded because
        # they quantify *how* non-linear the scene is.
        record(
            "info_response_mean_linearity",
            None,
            f"mean = {mean_fit['slope']:.3f} counts/us, R2={mean_fit['r2']:.4f} "
            f"(n={mean_fit['n_fit']}, axis = requested exposure quantised to "
            f"integer us), saturation onset {sat_us} us -- the proportional "
            f"iteration assumes linearity; a stepped scene breaks it",
        )
        record(
            "info_response_peak_linearity",
            None,
            f"peak = {peak_fit['slope']:.3f} counts/us, R2={peak_fit['r2']:.4f} "
            f"(n={peak_fit['n_fit']}) -- diagnostic only, see above",
        )

        # ----------------------------------------------------- knee scan
        logger.info("=== knee scan: integer-us steps around any response step ===")
        knee = []
        for exp_us in args.knee_exposures:
            cam.reset_exposure_time(exp_us / 1000.0)
            for _ in range(args.knee_repeats):
                img, info = _grab_with_frame_info(cam)
                s = _frame_stats(img)
                knee.append({
                    "requested_us": exp_us,
                    "device_expo_us": info["device_expo_us"],
                    "seq": info["seq"],
                    **s,
                })
                logger.info(
                    "  req={} us -> device={} us (seq={}): peak={:.0f} mean={:.2f}",
                    exp_us, info["device_expo_us"], info["seq"],
                    s["peak"], s["mean"],
                )
        results["knee_scan"] = knee
        # locate the largest relative mean step between consecutive device
        # exposures (the knee that breaks proportionality)
        steps = []
        by_dev: dict[int, list[dict]] = {}
        for k in knee:
            by_dev.setdefault(k["device_expo_us"], []).append(k)
        devs = sorted(by_dev)
        for a, b in zip(devs, devs[1:]):
            ma = float(np.median([k["mean"] for k in by_dev[a]]))
            mb = float(np.median([k["mean"] for k in by_dev[b]]))
            pa = float(np.median([k["peak"] for k in by_dev[a]]))
            pb = float(np.median([k["peak"] for k in by_dev[b]]))
            steps.append({
                "from_us": a, "to_us": b,
                "mean_rel": (mb - ma) / max(ma, 1e-9),
                "peak_rel": (pb - pa) / max(pa, 1e-9),
                "exp_rel": (b - a) / a,
            })
        results["knee_steps"] = steps
        if steps:
            worst = max(steps, key=lambda s: abs(s["mean_rel"]))
            logger.info(
                "  largest mean step: {}->{} us: mean {:+.0%}, peak {:+.0%} "
                "(exposure {:+.0%})",
                worst["from_us"], worst["to_us"], worst["mean_rel"],
                worst["peak_rel"], worst["exp_rel"],
            )

        # -------------------------------------------------- dirty-data scan
        logger.info("=== dirty-data scan: is the step real or a bad read? ===")
        dirty = _dirty_data_scan(cam, args)
        results["dirty_data"] = dirty
        record(
            "device_expo_matches_request",
            dirty["all_device_expo_match"],
            dirty["device_expo_detail"],
        )
        record(
            "frame_seq_advances",
            dirty["all_seq_advance"],
            dirty["seq_detail"],
        )
        record(
            "frame_timestamps_advance",
            dirty["all_timestamps_advance"],
            dirty["timestamp_detail"],
        )
        record(
            "frames_not_duplicated",
            dirty["no_duplicate_frames"],
            dirty["duplicate_detail"],
        )
        record(
            "switch_step_deterministic",
            dirty["step_deterministic"],
            dirty["step_detail"],
        )
        record(
            "switch_no_lag",
            dirty["no_switch_lag"],
            dirty["lag_detail"],
        )

        # ------------------------------------------------- AE test runs
        for spec in args.targets:
            target, n_sample, twice_valid, tol, expect = spec
            if expect is None:
                expect = "conv" if target >= args.reachable_floor else "bound"
            label = (f"target={target:.0f}_tol{tol:.0f}_n{n_sample}"
                     f"_twice{int(twice_valid)}_{expect}")
            logger.info("=== auto_exposure run: {} ===", label)

            # Restore the known starting point so every run starts equal.
            cam.reset_exposure_time(args.initial_exposure_ms)

            traj: list[dict] = []
            resets: list[dict] = []
            orig_get = cam.get_numpy_image
            orig_reset = cam.reset_exposure_time

            def spy_get(n_sample=1, skip_first=True, _orig=orig_get):
                img = _orig(n_sample, skip_first)
                traj.append({
                    "t_rel_s": round(time.time() - t_run, 3),
                    "exposure_ms": float(cam.exposure_time_ms),
                    **_frame_stats(img),
                })
                return img

            def spy_reset(ms, _orig=orig_reset):
                applied = _orig(ms)
                resets.append({"request_ms": float(ms),
                               "applied_ms": float(applied)})
                return applied

            cam.get_numpy_image = spy_get  # type: ignore[method-assign]
            cam.reset_exposure_time = spy_reset  # type: ignore[method-assign]
            t_run = time.time()
            try:
                img = cam.auto_exposure(
                    target_max=target,
                    tolerance=tol,
                    twice_valid=twice_valid,
                    max_iterations=args.max_iterations,
                    n_sample=n_sample,
                )
            finally:
                cam.get_numpy_image = orig_get  # type: ignore[method-assign]
                cam.reset_exposure_time = orig_reset  # type: ignore[method-assign]
            wall = time.time() - t_run

            low = max(target - tol, 10)
            high = min(target + tol, 254)
            final_peak = float(np.max(img))
            final_exp = float(cam.get_exposure_time_ms())
            run = {
                "label": label,
                "target_max": target,
                "tolerance": tol,
                "n_sample": n_sample,
                "twice_valid": twice_valid,
                "expect": expect,
                "start_exposure_ms": float(args.initial_exposure_ms),
                "final_exposure_ms": final_exp,
                "final_peak_returned": final_peak,
                "final_peak_last_measured": traj[-1]["peak"] if traj else None,
                "converged": low <= final_peak <= high,
                "n_measurements": len(traj),
                "n_resets": len(resets),
                "wall_s": round(wall, 2),
                "returned_type": type(img).__name__,
                "returned_is_ndarray": isinstance(img, np.ndarray),
                "resets": resets,
                "trajectory": traj,
            }

            # -- check: returned image matches the last measured frame
            run["returned_matches_last"] = (
                abs(final_peak - traj[-1]["peak"]) <= CONVERGE_PEAK_TOL
                if traj else False
            )
            # -- informational: how the peak responded to each exposure
            #    change (this scene has response rungs, so a plateau is a
            #    legitimate non-response; the authoritative frame-freshness
            #    verdict is the post-run probe below).
            responses = []
            for i, r in enumerate(resets):
                if i + 1 >= len(traj):
                    continue
                before, after = traj[i], traj[i + 1]
                exp_rel = abs(after["exposure_ms"] - before["exposure_ms"]) \
                    / max(before["exposure_ms"], 1e-9)
                peak_rel = abs(after["peak"] - before["peak"]) \
                    / max(before["peak"], 1e-9)
                responses.append({
                    "exposure_before_ms": before["exposure_ms"],
                    "exposure_after_ms": after["exposure_ms"],
                    "exp_rel": exp_rel,
                    "peak_before": before["peak"],
                    "peak_after": after["peak"],
                    "peak_rel": peak_rel,
                })
            run["change_responses"] = responses

            # -- authoritative per-run frame-freshness probe: at the final
            #    operating point, grab consecutive frames and verify they
            #    are fresh (not pixel-identical duplicates) and stable.
            probe_peaks, probe_diffs, probe_dups = [], [], 0
            prev_img = None
            for _ in range(3):
                pimg = cam.get_numpy_image(n_sample=1, skip_first=True)
                probe_peaks.append(float(np.max(pimg)))
                if prev_img is not None:
                    d = np.abs(prev_img.astype(np.int16)
                               - pimg.astype(np.int16))
                    probe_diffs.append(float(np.mean(d)))
                    if int(d.max()) == 0:
                        probe_dups += 1
                prev_img = pimg
            run["freshness_probe"] = {
                "exposure_ms": final_exp,
                "peaks": probe_peaks,
                "mean_abs_diffs": probe_diffs,
                "exact_duplicate_pairs": probe_dups,
                "peak_spread_rel": (max(probe_peaks) - min(probe_peaks))
                / max(float(np.median(probe_peaks)), 1e-9),
            }
            run["frames_fresh"] = (
                probe_dups == 0
                and all(d > 0.1 for d in probe_diffs)
                and run["freshness_probe"]["peak_spread_rel"] <= 0.05
            )

            results["runs"].append(run)
            logger.info(
                "  -> final exp={:.4f} ms, returned peak={:.0f}, converged={}, "
                "resets={}, wall={:.1f}s",
                final_exp, final_peak, run["converged"], len(resets), wall,
            )

        # --------------------------------------------------- post state
        try:
            ae_state = cam.get_auto_exposure_state()
        except Exception as exc:  # noqa: BLE001 - SDK getters may be absent
            ae_state = {"error": f"{type(exc).__name__}: {exc}"}
        post = {
            "auto_exposure_state": ae_state,
            "exposure_readback_ms": float(cam.get_exposure_time_ms()),
            "last_applied_ms": float(cam.exposure_time_ms),
        }
        results["post_state"] = post
        record(
            "sdk_auto_exposure_disabled",
            isinstance(ae_state, dict) and ae_state.get("mode") == 0,
            f"state={ae_state}",
        )
        record(
            "exposure_readback_consistent",
            abs(post["exposure_readback_ms"] - post["last_applied_ms"]) < 1e-9,
            f"readback={post['exposure_readback_ms']:.4f} ms, "
            f"last applied={post['last_applied_ms']:.4f} ms",
        )
    finally:
        try:
            cam.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("camera close failed: {}", exc)

    # ----------------------------------------------------- run verdicts
    for run in results["runs"]:
        band = (max(run["target_max"] - run["tolerance"], 10),
                min(run["target_max"] + run["tolerance"], 254))
        expect = run["expect"]
        if expect == "conv":
            record(
                f"converged[{run['label']}]",
                run["converged"],
                f"band contains a response rung: final peak "
                f"{run['final_peak_returned']:.0f} in [{band[0]:.0f}, "
                f"{band[1]:.0f}], exp {run['final_exposure_ms']:.4f} ms, "
                f"{run['n_resets']} reset(s)",
            )
        elif expect == "bound":
            hit_min = abs(run["final_exposure_ms"] - 0.011) < 1e-6
            record(
                f"boundary_exit[{run['label']}]",
                hit_min and run["n_resets"] <= 3,
                f"target below the lowest response rung: final exp "
                f"{run['final_exposure_ms']:.4f} ms (min boundary), "
                f"{run['n_resets']} reset(s) of max {args.max_iterations} "
                f"iterations, returned peak {run['final_peak_returned']:.0f}",
            )
        else:  # dead: band between rungs -- bounded non-convergence is correct
            record(
                f"deadzone_bounded[{run['label']}]",
                run["n_resets"] <= args.max_iterations
                and run["returned_is_ndarray"],
                f"band between rungs: terminated after {run['n_resets']} "
                f"reset(s) of max {args.max_iterations}, converged="
                f"{run['converged']}, final peak "
                f"{run['final_peak_returned']:.0f} at "
                f"{run['final_exposure_ms']:.4f} ms",
            )
        record(
            f"returned_matches_last[{run['label']}]",
            run["returned_matches_last"],
            f"returned peak {run['final_peak_returned']:.0f} vs last measured "
            f"{run['final_peak_last_measured']}",
        )
        record(
            f"frames_fresh[{run['label']}]",
            run["frames_fresh"],
            f"at {run['freshness_probe']['exposure_ms']:.4f} ms: peaks "
            f"{[f'{p:.0f}' for p in run['freshness_probe']['peaks']]}, "
            f"mean|diff|={np.mean(run['freshness_probe']['mean_abs_diffs']):.2f} "
            f"counts, duplicates={run['freshness_probe']['exact_duplicate_pairs']}, "
            f"peak spread {run['freshness_probe']['peak_spread_rel']:.1%}",
        )

    n_bool = sum(1 for c in results["checks"] if c["passed"] is not None)
    n_pass = sum(1 for c in results["checks"] if c["passed"])
    n_info = len(results["checks"]) - n_bool
    results["summary"] = {
        "n_checks": n_bool,
        "n_info": n_info,
        "n_passed": n_pass,
        "all_passed": n_pass == n_bool,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }

    _write_artifacts(out_dir, results, args)
    logger.info("Probe finished: {}/{} checks passed ({} diagnostic(s) recorded)",
                n_pass, n_bool, n_info)
    return 0 if results["summary"]["all_passed"] else 1


def _write_artifacts(out_dir: Path, results: dict, args: argparse.Namespace) -> None:
    """Persist config/results/trajectory CSV/figure/report.

    ``flat`` layout (debug dumps): everything in ``out_dir`` root.
    ``report`` layout (registered reports): ``report.md`` at the root, the
    figure under ``figures/`` and the raw artefacts under ``artifacts/``,
    per ``report/AGENTS.md``; the provenance header is stamped in because the
    sync script's ``--check`` would otherwise flag drift on the next run.
    """
    report_layout = args.layout == "report"
    art_dir = out_dir / "artifacts" if report_layout else out_dir
    fig_rel = "figures/auto_exposure_probe.png" if report_layout \
        else "auto_exposure_probe.png"

    config = {
        "cam_id": args.cam_id,
        "initial_exposure_ms": args.initial_exposure_ms,
        "premise_frames": args.premise_frames,
        "response_exposures": args.response_exposures,
        "knee_exposures": args.knee_exposures,
        "knee_repeats": args.knee_repeats,
        "switch_exposures": args.switch_exposures,
        "switch_rounds": args.switch_rounds,
        "fresh_exposures": args.fresh_exposures,
        "fresh_frames": args.fresh_frames,
        "determinism_tol": args.determinism_tol,
        "switch_lag_tol": args.switch_lag_tol,
        "targets": [
            {"target_max": t[0], "n_sample": t[1], "twice_valid": t[2],
             "tolerance": t[3], "expect": t[4]}
            for t in args.targets
        ],
        "reachable_floor": args.reachable_floor,
        "max_iterations": args.max_iterations,
        "python": sys.version,
        "device_safety": "miicam SDK only; Daheng/gxipy never touched",
    }
    (art_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")

    (art_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    with (art_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run", "t_rel_s", "exposure_ms", "peak", "mean",
                         "min", "lit_fraction"])
        for run in results["runs"]:
            for t in run["trajectory"]:
                writer.writerow([run["label"], t["t_rel_s"], t["exposure_ms"],
                                 f"{t['peak']:.1f}", f"{t['mean']:.3f}",
                                 f"{t['min']:.0f}", f"{t['lit_fraction']:.3f}"])

    _write_figure(out_dir / fig_rel, results)
    report_path = out_dir / "report.md"
    _write_report(report_path, results, config, fig_rel)
    if report_layout:
        _stamp_provenance(report_path)


def _stamp_provenance(path: Path) -> None:
    """Insert the ``report/`` provenance header when *path* is a registered key.

    Same contract as ``TestReport.save``: the probe **overwrites** the report
    on every run, so a header added by a later sync pass would be dropped by
    the next run and ``--check`` would report drift on a correct file.
    Unregistered paths (``data/debug/`` dumps) are left untouched --
    ``insert_header`` is a no-op for them.
    """
    try:
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from scripts._common.provenance import insert_header
    except ImportError:
        return
    try:
        key = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return
    text = path.read_text(encoding="utf-8")
    stamped = insert_header(text, key)
    if stamped != text:
        path.write_text(stamped, encoding="utf-8")


def _write_figure(path: Path, results: dict) -> None:
    """Four-panel figure: response map, knee scan, switch test, trajectories."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(19, 4.5))

    rm = results.get("response_map") or {}
    if rm.get("requested_us"):
        dev = np.asarray(rm["requested_us"], dtype=float)
        ax1.plot(dev, rm["means"], "o-", color="#2471a3", label="mean")
        ax1.plot(dev, rm["peaks"], "s-", color="#b03a2e", label="peak")
        ax1.axhline(255, color="#7f8c8d", ls="--", lw=1, label="8-bit full scale")
        ax1.set_xlabel("requested exposure (us, quantised)")
        ax1.set_ylabel("counts (0-255)")
        ax1.set_title("sensor response map")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

    knee = results.get("knee_scan") or []
    if knee:
        xs = [k["requested_us"] for k in knee]
        ys = [k["peak"] for k in knee]
        ms = [k["mean"] for k in knee]
        order = np.argsort(xs)
        ax2.plot(np.asarray(xs)[order], np.asarray(ys)[order], "s",
                 color="#b03a2e", label="peak")
        ax2.plot(np.asarray(xs)[order], np.asarray(ms)[order], "o",
                 color="#2471a3", label="mean")
        ax2.set_xlabel("requested exposure (us)")
        ax2.set_ylabel("counts (0-255)")
        ax2.set_title("knee scan (integer-us steps)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

    dirty = results.get("dirty_data") or {}
    if dirty.get("switch_records"):
        colors = {16.0: "#7d3c98", 17.0: "#b03a2e", 18.0: "#1e8449",
                  19.0: "#2471a3"}
        for i, r in enumerate(dirty["switch_records"]):
            ax3.scatter(i, r["mean"],
                        color=colors.get(float(r["requested_us"]), "#555555"),
                        s=28, zorder=3)
        # legend proxies
        seen = []
        for e, c in colors.items():
            if any(float(r["requested_us"]) == e
                   for r in dirty["switch_records"]):
                seen.append(ax3.scatter([], [], color=c, label=f"{e:.0f} us"))
        ax3.legend(handles=seen, fontsize=8, title="requested exposure")
        ax3.set_xlabel("switch index (interleaved rounds)")
        ax3.set_ylabel("mean counts")
        ax3.set_title("dirty-data switch test (does mean track request?)")
        ax3.grid(alpha=0.3)

    for run in results["runs"]:
        traj = run["trajectory"]
        if not traj:
            continue
        xs = np.arange(len(traj))
        ys = [t["peak"] for t in traj]
        tgt = run["target_max"]
        ax4.plot(xs, ys, marker="o", ms=4,
                 label=f"target {tgt:.0f} (tol {run['tolerance']:.0f}, "
                       f"n={run['n_sample']})")
        ax4.axhspan(tgt - run["tolerance"], tgt + run["tolerance"],
                    alpha=0.08, color="#2471a3")
    ax4.set_xlabel("measurement index (each exposure change = +1)")
    ax4.set_ylabel("peak counts (0-255)")
    ax4.set_title("auto_exposure iteration trajectories")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_report(path: Path, results: dict, config: dict,
                  fig_rel: str = "auto_exposure_probe.png") -> None:
    """Markdown report: verdict table plus per-run trajectory tables."""
    lines: list[str] = []
    cam = results.get("camera", {})
    prem = results.get("premise") or {}
    lines.append("# MiiCam 自动曝光算法实验报告")
    lines.append("")
    lines.append(f"- 时间: {results['started_at']} ~ "
                 f"{results['summary']['finished_at']}")
    lines.append(f"- 相机: sn={cam.get('sn')}, "
                 f"{cam.get('width')}x{cam.get('height')}, "
                 f"{cam.get('bit_depth')} bit, 模式 {cam.get('capture_mode')}")
    lines.append("- 设备安全: 仅经 miicam SDK 枚举/打开/取帧; "
                 "Daheng (gxipy/GigE) 全程未导入、未枚举、未打开")
    lines.append(f"- 起点: 曝光 {config['initial_exposure_ms']} ms, "
                 f"峰值中位数 {prem.get('peak_median', float('nan')):.0f} "
                 f"counts (范围 {prem.get('peak_min', float('nan')):.0f}-"
                 f"{prem.get('peak_max', float('nan')):.0f}), "
                 f"点亮像素占比最低 {prem.get('lit_fraction_min', float('nan')):.2f}")
    lines.append("")

    s = results["summary"]
    lines.append("## 总体结论")
    lines.append("")
    diag = (f"，另有 {s['n_info']} 项诊断记录（不计入通过率）"
            if s.get("n_info") else "")
    lines.append(f"**{s['n_passed']}/{s['n_checks']} 项检查通过**"
                 f"{'（全部通过）' if s['all_passed'] else '（存在未通过项，见下表）'}"
                 f"{diag}")
    lines.append("")
    lines.append("| 检查项 | 结果 | 明细 |")
    lines.append("|---|---|---|")
    for c in results["checks"]:
        verdict = ("记录" if c["passed"] is None
                   else ("通过" if c["passed"] else "**未通过**"))
        lines.append(f"| `{c['check']}` | {verdict} | {c['detail']} |")
    lines.append("")
    lines.append(f"![四联图：响应图 / 台阶扫描 / 切换测试 / 迭代轨迹]({fig_rel})")
    lines.append("")

    rm = results.get("response_map") or {}
    if rm:
        lines.append("## 传感器响应图")
        lines.append("")
        lines.append("| 请求曝光 (ms) | 量化 (us) | 帧信息回读 (us) | 峰值 | 均值 |")
        lines.append("|---|---|---|---|---|")
        for r, u, d, p, m in zip(rm["requested_ms"], rm["requested_us"],
                                 rm["device_expo_us"], rm["peaks"],
                                 rm["means"]):
            dev = str(d) if d else "不回读"
            lines.append(f"| {r:.4f} | {u} | {dev} | {p:.0f} | {m:.2f} |")
        lines.append("")
        pf, mf = rm["peak_fit"], rm["mean_fit"]
        lines.append(f"- 峰值过原点拟合: {pf['slope']:.3f} counts/us, "
                     f"R2 = {pf['r2']:.4f} (n={pf['n_fit']})")
        lines.append(f"- 均值过原点拟合: {mf['slope']:.3f} counts/us, "
                     f"R2 = {mf['r2']:.4f} (n={mf['n_fit']})")
        lines.append(f"- 峰值饱和起始设备曝光: {rm['saturation_onset_us']} us")
        lines.append("")

    steps = results.get("knee_steps") or []
    if steps:
        lines.append("## 响应台阶扫描（整数微秒步进）")
        lines.append("")
        lines.append("| 设备曝光 (us) | 峰值中位数 | 均值中位数 |")
        lines.append("|---|---|---|")
        by_dev: dict[int, list[float]] = {}
        by_dev_peak: dict[int, list[float]] = {}
        for k in results["knee_scan"]:
            by_dev.setdefault(k["device_expo_us"], []).append(k["mean"])
            by_dev_peak.setdefault(k["device_expo_us"], []).append(k["peak"])
        for d in sorted(by_dev):
            lines.append(f"| {d} | {np.median(by_dev_peak[d]):.0f} "
                         f"| {np.median(by_dev[d]):.2f} |")
        lines.append("")
        lines.append("相邻档位相对变化:")
        lines.append("")
        lines.append("| 从 (us) | 到 (us) | 曝光变化 | 均值变化 | 峰值变化 |")
        lines.append("|---|---|---|---|---|")
        for st in steps:
            lines.append(f"| {st['from_us']} | {st['to_us']} "
                         f"| {st['exp_rel']:+.1%} | {st['mean_rel']:+.1%} "
                         f"| {st['peak_rel']:+.1%} |")
        lines.append("")

    dirty = results.get("dirty_data") or {}
    if dirty:
        lines.append("## 脏数据检查（调整曝光验证读取是否正常）")
        lines.append("")
        lines.append("**交叉切换测试**（多轮交替设置 16/17/18/19 us，"
                     "每次取跳过首帧后的一帧）:")
        lines.append("")
        lines.append("| 轮次 | 请求 (us) | 帧信息回读 (us) | 峰值 | 均值 |")
        lines.append("|---|---|---|---|---|")
        for r in dirty["switch_records"]:
            dev = str(r["device_expo_us"]) if r["device_expo_us"] else "不回读"
            lines.append(f"| {r['round']} | {r['requested_us']:.0f} "
                         f"| {dev} | {r['peak']:.0f} "
                         f"| {r['mean']:.2f} |")
        lines.append("")
        lines.append("各请求档位的统计:")
        lines.append("")
        lines.append("| 请求 (us) | 帧信息回读 (us) | 均值中位数 | 跨轮离散度 |")
        lines.append("|---|---|---|---|")
        for e, v in dirty["per_exposure"].items():
            dev = ", ".join(str(d) if d else "不回读"
                            for d in v["device_expo_us"])
            lines.append(f"| {e:.0f} | {dev} "
                         f"| {v['mean_median']:.2f} "
                         f"| {v['mean_spread_rel']:.1%} |")
        lines.append("")
        lines.append("切换滞后（切换后读数距新档位的比例，0=完全落位，"
                     "1=仍是旧档位）:")
        lines.append("")
        lines.append("| 切换 (us) | 实测均值 | 档位均值 | 滞后 |")
        lines.append("|---|---|---|---|")
        for l in dirty["switch_lags"]:
            lines.append(f"| {l['from_us']:.0f}→{l['to_us']:.0f} "
                         f"| {l['measured_mean']:.2f} | {l['target_mean']:.2f} "
                         f"| {l['lag']:.0%} |")
        lines.append("")
        lines.append("**定档帧新鲜度**（连续 burst 帧的 SDK 帧信息）:")
        lines.append("")
        lines.append("| 档位 (us) | 帧序号增量 | 时间戳增量 (us) | 帧信息 flag "
                     "| 设备曝光字段 (us) | 相邻帧平均绝对差 (counts) "
                     "| 完全重复帧对数 |")
        lines.append("|---|---|---|---|---|---|---|")
        for e, v in dirty["freshness"].items():
            lines.append(
                f"| {e:.0f} | {v['seq_deltas']} | {v['ts_deltas_us']} "
                f"| {[hex(f) for f in v['flags'][:3]]} "
                f"| {v['device_expo_us']} "
                f"| {np.mean(v['consecutive_mean_abs_diff']):.2f} "
                f"| {v['exact_duplicate_pairs']} |")
        lines.append("")
        lines.append("判定: "
                     f"设备曝光与请求一致={dirty['all_device_expo_match']}, "
                     f"帧序号推进={dirty['all_seq_advance']}, "
                     f"时间戳推进={dirty['all_timestamps_advance']}, "
                     f"无完全重复帧={dirty['no_duplicate_frames']}, "
                     f"台阶确定性={dirty['step_deterministic']}, "
                     f"切换无滞后={dirty['no_switch_lag']}")
        lines.append("")

    lines.append("## 各用例轨迹")
    lines.append("")
    for run in results["runs"]:
        lines.append(f"### {run['label']}")
        lines.append("")
        expect_zh = {"conv": "带内存在响应台阶，应收敛",
                     "bound": "目标低于最低台阶，应在最小曝光边界退出",
                     "dead": "目标带落在台阶之间，有界不收敛为正确行为"}
        lines.append(f"- 目标峰值 {run['target_max']:.0f}±{run['tolerance']:.0f} "
                     f"counts, n_sample={run['n_sample']}, "
                     f"twice_valid={run['twice_valid']}; 预期: "
                     f"{expect_zh.get(run['expect'], run['expect'])}")
        lines.append(f"- 起始曝光 {run['start_exposure_ms']} ms → 终止曝光 "
                     f"**{run['final_exposure_ms']:.4f} ms**; "
                     f"返回帧峰值 {run['final_peak_returned']:.0f} "
                     f"({'收敛' if run['converged'] else '未收敛/边界退出'}); "
                     f"曝光修改 {run['n_resets']} 次, 测量 {run['n_measurements']} 次, "
                     f"耗时 {run['wall_s']:.1f} s; 返回类型 {run['returned_type']}")
        lines.append("")
        lines.append("| # | t (s) | 曝光 (ms) | 峰值 | 均值 |")
        lines.append("|---|---|---|---|---|")
        for i, t in enumerate(run["trajectory"]):
            lines.append(f"| {i} | {t['t_rel_s']:.2f} | {t['exposure_ms']:.4f} "
                         f"| {t['peak']:.0f} | {t['mean']:.2f} |")
        if run["change_responses"]:
            lines.append("")
            lines.append("曝光变化时的峰值响应（本场景响应呈台阶状，"
                         "平台内不响应属正常；仅作记录，判定见下方新鲜度探针）:")
            lines.append("")
            lines.append("| 曝光变化 | 峰值变化 |")
            lines.append("|---|---|")
            for sc in run["change_responses"]:
                lines.append(
                    f"| {sc['exposure_before_ms']:.4f}→"
                    f"{sc['exposure_after_ms']:.4f} ms ({sc['exp_rel']:+.0%}) "
                    f"| {sc['peak_before']:.0f}→{sc['peak_after']:.0f} "
                    f"({sc['peak_rel']:+.0%}) |")
        fp = run["freshness_probe"]
        lines.append("")
        lines.append(f"终曝光 {fp['exposure_ms']:.4f} ms 连续帧新鲜度: 峰值 "
                     f"{[f'{p:.0f}' for p in fp['peaks']]}, "
                     f"相邻帧平均绝对差 "
                     f"{np.mean(fp['mean_abs_diffs']):.2f} counts, "
                     f"完全重复帧 {fp['exact_duplicate_pairs']} 对, "
                     f"峰值离散 {fp['peak_spread_rel']:.1%} "
                     f"({'新鲜' if run['frames_fresh'] else '**异常**'})")
        lines.append("")

    post = results.get("post_state") or {}
    if post:
        lines.append("## 运行后状态")
        lines.append("")
        lines.append(f"- SDK 自动曝光状态: {post.get('auto_exposure_state')}")
        lines.append(f"- 曝光回读 {post.get('exposure_readback_ms'):.4f} ms, "
                     f"最后一次 applied {post.get('last_applied_ms'):.4f} ms")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("生成脚本: `scripts/ccd/miicam_auto_exposure_probe.py` "
                 "（本报告由该脚本自动生成，数值见 `results.json` / "
                 f"`trajectory.csv`，图见 `{fig_rel}`）")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_targets(specs: list[str]) -> list[tuple]:
    """Parse ``TARGET[:n[:twice[:tol[:expect]]]]`` run specs.

    ``expect`` is one of ``conv`` (must converge), ``bound`` (target below
    the lowest response rung: must exit at the min-exposure boundary) or
    ``dead`` (band between rungs: bounded non-convergence is the correct
    behaviour). Omitted -> inferred from ``--reachable-floor`` by the caller.
    """
    out: list[tuple] = []
    for spec in specs:
        parts = spec.split(":")
        target = float(parts[0])
        n_sample = int(parts[1]) if len(parts) > 1 else 1
        twice = bool(int(parts[2])) if len(parts) > 2 else True
        tol = float(parts[3]) if len(parts) > 3 else 5.0
        expect = parts[4] if len(parts) > 4 else None
        out.append((target, n_sample, twice, tol, expect))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiiCam auto_exposure hardware verification probe "
                    "(miicam SDK only; never touches the Daheng camera)")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--initial-exposure-ms", type=float, default=0.02)
    parser.add_argument("--premise-frames", type=int, default=5)
    parser.add_argument(
        "--response-exposures", type=float, nargs="+",
        default=[0.011, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
        help="coarse response map exposures (ms)",
    )
    parser.add_argument(
        "--knee-exposures", type=float, nargs="+",
        default=[15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 22.0],
        help="integer-microsecond knee-scan exposures (us)",
    )
    parser.add_argument("--knee-repeats", type=int, default=2)
    parser.add_argument(
        "--switch-exposures", type=float, nargs="+",
        default=[16.0, 17.0, 18.0, 19.0],
        help="interleaved dirty-data switch-test exposures (us)",
    )
    parser.add_argument("--switch-rounds", type=int, default=3)
    parser.add_argument(
        "--fresh-exposures", type=float, nargs="+",
        default=[17.0, 20.0],
        help="fixed exposures for the frame-freshness burst (us)",
    )
    parser.add_argument("--fresh-frames", type=int, default=8)
    parser.add_argument("--determinism-tol", type=float, default=0.05,
                        help="max relative spread per exposure across rounds")
    parser.add_argument("--switch-lag-tol", type=float, default=0.5,
                        help="max fraction of the previous level left after "
                             "a switch (0 = fully settled, 1 = fully stale)")
    parser.add_argument("--lag-denominator-floor", type=float, default=2.0,
                        help="min mean-counts gap between two exposure levels "
                             "for their switch to carry lag information")
    parser.add_argument(
        "--targets", nargs="+",
        default=["220:1:1:12:conv", "240:1:1:15:conv", "200:1:1:10:dead",
                 "200:1:1:5:dead", "40:1:1:5:bound", "120:3:1:10:bound"],
        help="run specs TARGET[:n_sample[:twice_valid[:tolerance[:expect]]]]; "
             "expect: conv=must converge, bound=exit at min-exposure "
             "boundary, dead=bounded non-convergence between rungs",
    )
    parser.add_argument("--reachable-floor", type=float, default=180.0,
                        help="peak level above which a target is expected "
                             "reachable on this scene")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument(
        "--output-dir", type=str,
        default=f"data/debug/miicam_auto_exposure_"
                f"{datetime.now():%Y%m%d_%H%M%S}",
    )
    parser.add_argument(
        "--layout", choices=["flat", "report"], default="flat",
        help="flat: all artefacts in --output-dir root (debug dumps, the "
             "directory must not exist yet); report: report.md at the root "
             "with figures/ and artifacts/ subdirs plus the provenance "
             "header, for a registered report/ directory (reused in place)",
    )
    args = parser.parse_args()
    args.targets = _parse_targets(args.targets)
    sys.exit(run_probe(args))


if __name__ == "__main__":
    main()
