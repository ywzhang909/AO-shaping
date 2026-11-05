"""Issue #230 online-refeed probe: does re-fitting on newly captured frames recover
holdout quality that a stale model loses (drift), and is it as good as a full refit?

Two-pass design on hardware. Pass 1 at t0 captures calibration + a structured
single-mode ±Zernike train pool + a holdout pool (unseen out-of-family phases).
After ``--gap-seconds`` (thermal/mechanical drift window) pass 2 re-captures the
*same* train pool and *same* holdout pool. Offline, three models answer the
refeed question on the pass-2 holdout:

- ``stale``  = fit on pass-1 train only (never refit)  -> drift loss
- ``refeed`` = fit on pass-1 + pass-2 train            -> does refeed recover it?
- ``fresh``  = fit on pass-2 train only                -> upper bound (full refit)

If drift exists, ``stale`` degrades relative to its own pass-1 holdout; refeed
benefit = ``stale`` - ``refeed``. This isolates refeed (accumulate) from full
refit (replace), and needs no production closed loop.

Reuses production ``transfer/capture.py`` for all hardware, exposure, manifest
and safe-close behaviour by patching ``capture.probes`` (same pattern as
``scripts/slm/zernike_probe_hardware.py``). Default is plan-only; ``--execute``
is required to open hardware. All device data goes to ``data/debug``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ao_shaping.tools.slm.transfer import capture
from ao_shaping.tools.slm.transfer.probes import probe_grid
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase


def build_plan(
    modes: list[int], amplitudes: list[float], n_holdout: int
) -> list[tuple[str, str, str, float | None]]:
    """Return (label, split, kind, noll) plan; kind in {zernike, holdout}.

    Train pool is single-mode ±Zernike (one mode active, rest zero), nested so
    any prefix is a balanced subset. Holdout uses the random family (out of the
    structured family) so the fit is genuinely extrapolating.
    """
    plan: list[tuple[str, str, str, float | None]] = []
    for amp in amplitudes:
        for j in modes:
            for sign in (1, -1):
                plan.append(
                    (f"ztrain_{len(plan):03d}_Z{j}_{sign * amp:+.2f}", "train", "zernike", j)
                )
    for k in range(n_holdout):
        plan.append((f"holdout_{k:03d}", "test", "holdout", None))
    return plan


def make_probes(plan: list[tuple[str, str, str, float | None]], seed: int, pass_index: int):
    """Generator that patches ``capture.probes`` for one pass.

    Pass 2 re-captures the exact same phases: for train rows the amplitude is
    recovered from the label (same as pass 1); holdout rows are re-derived from
    the same seed, so ``holdout_{k}`` pass 2 == pass 1 phase.
    """
    original = capture.probes

    def probes(size: int, count: int, seed_arg: int, periods=(64, 128)):
        # reference_start + signed ramps (drop the trailing reference_end); the
        # harness only needs one calibration, so pass 2 emits none.
        if pass_index == 0:
            yield from list(original(size, 0, seed_arg, periods))[:-1]
        zero = np.zeros((size, size), np.float32)
        holdout_index = 0
        for label, split, kind, noll in plan:
            if kind == "zernike":
                amp = float(label.rsplit("_", 1)[1])
                phase = generate_zernike_phase({int(noll): amp}, (size, size))
                phase = np.nan_to_num(np.asarray(phase, dtype=np.float32), nan=0.0)
                yield f"p{pass_index}_{label}", split, phase
            elif kind == "holdout":
                _, phase = probe_grid(1000 + holdout_index, size, seed + 5)
                yield f"p{pass_index}_{label}", split, phase
                holdout_index += 1
        # Pass 2 re-emits the trailing reference_end so the session closes cleanly.
        if pass_index == 1:
            yield "p2_reference_end", "calibration", zero

    return probes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--modes", type=int, nargs="+", default=list(range(4, 12)))
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.5])
    parser.add_argument("--n-holdout", type=int, default=6)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2301105)
    parser.add_argument("--gap-seconds", type=float, default=0.0)
    parser.add_argument("--camera", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    plan = build_plan(args.modes, args.amplitudes, args.n_holdout)
    n_train = sum(1 for p in plan if p[2] == "zernike")
    plan_doc = {
        "schema": "issue230_refeed/1",
        "issue": 230,
        "grid": args.grid,
        "modes": args.modes,
        "amplitudes_rad": args.amplitudes,
        "frames": args.frames,
        "seed": args.seed,
        "gap_seconds": args.gap_seconds,
        "n_train": n_train,
        "n_holdout": args.n_holdout,
        "passes": 2,
        "plan": [{"label": p[0], "split": p[1], "kind": p[2]} for p in plan],
    }

    if not args.execute:
        for label, split, kind, _ in plan:
            print(f"{split:12s} {kind:8s} {label}")
        print(
            f"\nPLAN ONLY: 2 passes x ({n_train} structured train + "
            f"{args.n_holdout} holdout) + calibration, gap {args.gap_seconds}s."
        )
        print("Re-run with --execute to open hardware and acquire.")
        print(
            "NOTE: --execute requires a fresh --out directory (capture uses exist_ok=False)."
        )
        return 0

    # capture() creates each pass directory itself (exist_ok=False); the parent
    # may pre-exist but must be empty.
    if args.out.exists() and any(args.out.iterdir()):
        print(f"ERROR: --out must be fresh and empty: {args.out}")
        return 2

    # capture() uses mkdir(exist_ok=False) and a fixed manifest.json, so each
    # pass needs its own fresh directory: out/pass1 and out/pass2.
    namespace = SimpleNamespace(
        out=args.out / "pass1",
        execute=True,
        camera=args.camera,
        cam_id=args.cam_id,
        slm_number=args.slm_number,
        grid=args.grid,
        ramp_periods=(64, 128),
        count=0,
        frames=args.frames,
        seed=args.seed,
        exposure_max_steps=2,
        save_dataset=True,
        device=None,
        checkpoint=None,
    )

    # Pass 1: calibration + train + holdout. capture() creates args.out itself.
    with patch.object(capture, "probes", make_probes(plan, args.seed, 0)):
        capture.capture(namespace)

    if args.gap_seconds > 0:
        print(f"[gap] waiting {args.gap_seconds:.0f}s for drift window ...")
        time.sleep(args.gap_seconds)

    # Pass 2: same train + holdout, no calibration, trailing reference_end.
    namespace.out = args.out / "pass2"
    with patch.object(capture, "probes", make_probes(plan, args.seed, 1)):
        capture.capture(namespace)

    (args.out / "probe_plan.json").write_text(
        json.dumps(plan_doc, indent=2), encoding="utf-8"
    )
    print(f"Acquisition complete -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
