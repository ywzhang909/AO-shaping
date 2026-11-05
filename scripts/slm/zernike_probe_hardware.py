"""Issue #229 hardware probe acquisition: physically-interpretable ±Zernike vs random.

One-off experiment (2026-11-04). Reuses production ``transfer/capture.py`` for all
hardware, exposure, manifest and safe-close behaviour by patching ``capture.probes``
to emit a single superset pool. The pool is captured once so the ±Zernike arm, the
random arm and the baseline can be sliced **offline at equal training-frame budgets**
without re-occupying the SLM/CCD.

Dataset layout matches ``transfer/fit.load_dataset``: calibration rows
(``reference_start`` + signed ``ramp_*`` + ``reference_end``) for geometry, then
labelled probe rows whose NPZ carries ``model_phase`` and ``raw_frames``:

- ``ztrain_{i}_Z{j}_{amp}``  split=train  single-mode Noll-j at ±amplitude (rad)
- ``rtrain_{i}_{family}``    split=train  random probes (baseline/random arm)
- ``holdout_{i}_{family}``   split=test   unseen phases, never used for fitting

Zernike rows are nested-ordered (all modes at ±0.1 first, then ±0.2) so any prefix
of length n is a balanced training subset for a sample-efficiency learning curve.
Default is plan-only; ``--execute`` is required to open hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ao_shaping.tools.slm.transfer import capture
from ao_shaping.tools.slm.transfer.probes import probe_grid
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase


def build_plan(
    modes: list[int], amplitudes: list[float], n_random: int, n_holdout: int
) -> list[tuple[str, str, str, float | None]]:
    """Return (label, split, kind, noll_or_nan) plan; kind in {zernike,random,holdout}."""
    plan: list[tuple[str, str, str, float | None]] = []
    i = 0
    for amp in amplitudes:  # nested: every mode at the small amplitude first
        for j in modes:
            for sign in (1.0, -1.0):
                plan.append(
                    (
                        f"ztrain_{i:03d}_Z{j}_{sign * amp:+.2f}",
                        "train",
                        "zernike",
                        float(j),
                    )
                )
                i += 1
    for k in range(n_random):
        plan.append((f"rtrain_{k:03d}", "train", "random", None))
    for k in range(n_holdout):
        plan.append((f"holdout_{k:03d}", "test", "holdout", None))
    return plan


def make_probes(plan, seed: int):
    original = capture.probes

    def probes(size: int, count: int, seed_arg: int, periods=(64, 128)):
        # reference_start + signed ramps (drop the trailing reference_end).
        yield from list(original(size, 0, seed_arg, periods))[:-1]
        zero = np.zeros((size, size), np.float32)
        random_index = 0
        holdout_index = 0
        for label, split, kind, noll in plan:
            if kind == "zernike":
                # label encodes Z{j} and signed amplitude; recover amplitude from label.
                amp = float(label.rsplit("_", 1)[1])
                phase = generate_zernike_phase({int(noll): amp}, (size, size))
                phase = np.nan_to_num(np.asarray(phase, dtype=np.float32), nan=0.0)
                yield label, split, phase
            elif kind == "random":
                family, phase = probe_grid(random_index, size, seed + 777)
                yield f"{label}_{family}", split, phase
                random_index += 1
            else:  # holdout
                family, phase = probe_grid(1000 + holdout_index, size, seed + 5)
                yield f"{label}_{family}", split, phase
                holdout_index += 1
        yield "reference_end", "calibration", zero

    return probes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--modes", type=int, nargs="+", default=list(range(4, 12)))
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--n-random", type=int, default=24)
    parser.add_argument("--n-holdout", type=int, default=12)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1104229)
    parser.add_argument("--camera", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan = build_plan(args.modes, args.amplitudes, args.n_random, args.n_holdout)
    plan_doc = {
        "schema": "zernike_probe_hardware/1",
        "issue": 229,
        "grid": args.grid,
        "modes": args.modes,
        "amplitudes_rad": args.amplitudes,
        "frames": args.frames,
        "seed": args.seed,
        "n_zernike": sum(1 for p in plan if p[2] == "zernike"),
        "n_random": args.n_random,
        "n_holdout": args.n_holdout,
        "plan": [{"label": p[0], "split": p[1], "kind": p[2]} for p in plan],
    }

    if not args.execute:
        for label, split, kind, _ in plan:
            print(f"{split:12s} {kind:8s} {label}")
        print(
            f"\nPLAN ONLY: {len(plan)} probes + calibration (reference_start, ramps, reference_end)."
        )
        print("Re-run with --execute to open hardware and acquire.")
        print(
            "NOTE: --execute requires a fresh --out directory (capture uses exist_ok=False)."
        )
        return

    # capture() creates args.out itself (exist_ok=False); do not pre-create it.
    namespace = SimpleNamespace(
        out=args.out,
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
    with patch.object(capture, "probes", make_probes(plan, args.seed)):
        capture.capture(namespace)
    (args.out / "probe_plan.json").write_text(
        json.dumps(plan_doc, indent=2), encoding="utf-8"
    )
    print(f"Acquisition complete -> {args.out}")


if __name__ == "__main__":
    main()
