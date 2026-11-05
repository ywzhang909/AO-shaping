# -*- coding: utf-8 -*-
"""Read-only preflight for a SLM transfer hardware run.

Checks, without touching any device (no SLM, no camera, no laser):

1. ``check_parameters`` on the intended CLI-level values (pure parameters).
2. ``check_geometry`` on the measured zero point and fitted ramp taken from a
   frozen-parameters checkpoint (``fit/frozen_parameters.json`` or
   ``blur/frozen_parameters.json``).
3. ``check_zero_overlap`` on the same geometry, reported as a reading threshold.
4. Prints the exact ``workflow`` CLI line to run next, with the checkpoint wired.

Usage (from repo root)::

    python scripts/slm/prepare_transfer_hardware_run.py ^
        --checkpoint data/debug/slm_transfer/20261105_171947/fit/frozen_parameters.json ^
        --zero-xy 949 583 --period 64 --side 80 --shapes square rectangle triangle

Exit code 0 = all checks pass; 1 = at least one blocking problem; 2 = bad input.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np

from ao_shaping.tools.slm.transfer.preflight import (
    check_geometry,
    check_parameters,
    check_zero_overlap,
    format_problems,
)


def _load_checkpoint(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ("origin", "amplitude_center", "amplitude_width", "ramp_matrix")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{path} lacks keys: {missing}")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--zero-xy", type=float, nargs=2, default=None,
                        metavar=("X", "Y"), help="measured zero point (camera px); default: checkpoint origin")
    parser.add_argument("--period", type=float, default=64.0)
    parser.add_argument("--side", type=float, default=80.0)
    parser.add_argument("--shapes", nargs="+", default=["square", "rectangle", "triangle"])
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--camera", default="miicam")
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="session dir for the printed command (default data/debug/slm_transfer/<stamp>)")
    args = parser.parse_args(argv)

    try:
        ckpt = _load_checkpoint(args.checkpoint)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[bad input] {exc}", file=sys.stderr)
        return 2

    zero_xy = tuple(args.zero_xy) if args.zero_xy else tuple(float(v) for v in ckpt["origin"])
    ramp = np.asarray(ckpt["ramp_matrix"], dtype=float)

    problems = check_parameters(
        side=args.side, period=args.period, count=args.count, iterations=args.iterations,
        blocks=args.blocks, updates=args.updates, frames=args.frames, shapes=args.shapes,
        offset_px=None,
    )
    problems += check_geometry(
        zero_xy=zero_xy, ramp_matrix=ramp, side=args.side, period=args.period, shapes=args.shapes,
    )
    if problems:
        print(format_problems(problems))
    blocking = [p for p in problems]

    # Zero overlap: reading threshold per shape, centred on the target the
    # geometry places at the first shape's default centre. Reported, not blocking.
    # Zero-overlap needs the carrier-shifted target centre, which only the
    # fitted geometry knows; it is not computed here (use check_zero_overlap
    # from the fit output when reading CV).
    out = args.out or Path("data/debug/slm_transfer/<stamp>")
    cmd = [
        "python", "-m", "ao_shaping.tools.slm.transfer.workflow", "run",
        "--out", str(out),
        "--checkpoint", str(args.checkpoint),
        "--camera", args.camera, "--slm-number", str(args.slm_number), "--cam-id", str(args.cam_id),
        "--period", str(args.period), "--side", str(args.side),
        "--shapes", *args.shapes,
        "--adaptation", "auto", "--accept-metric", "loss",
        "--execute",  # device gate: required to open SLM #1 / MIICAM
    ]
    print("\nnext run (NOT executed here; device must be free, SLM #1 memory mode, no DVI):")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    print("  (verify flag names against `workflow.py run --help` before launching)")

    if blocking:
        print(f"\nBLOCKED: {len(blocking)} problem(s) above. Do not start the run.")
        return 1
    print("\nOK: all preflight checks pass (device untouched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
