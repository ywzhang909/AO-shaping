"""Re-render the multishape-scale process animation from the finished runs.

Re-reads the capture ``.npz`` files and ``iterations.pkl`` of each run (no
device, no re-measurement) and writes the wide 4-shape x 4-stage GIF plus the
single-stage GIF next to it.

The phase panel used to be identical in every frame: the colour scale was
recomputed from each frame's own 90% quantile, so every frame got its full
range back. ``recording.render_process_gif`` now takes one scale for the whole
animation (see ``test_process_gif_uses_one_phase_scale_for_every_frame``).

Usage::

    .venv/Scripts/python.exe scripts/slm/render_multishape_scale_process_gif.py \
        --session data/debug/slm_transfer/multishape_scale_20261028/s1200 \
        --out report/shape_trials/multishape_scale/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.tools.slm.transfer import recording

SHAPES = ("square", "triangle", "star", "circle")
STAGES = ("transport_cv", "mono12_cv", "mono12_continue", "mono12_dense")


def _stage_records(session: Path, shapes, stages) -> list[dict]:
    records: list[dict] = []
    for shape in shapes:
        for stage in stages:
            run = session / "shape" / shape / stage
            if not (run / "manifest.json").is_file():
                print(f"skip (no manifest): {run}")
                continue
            stage_records = recording.block_records(run, f"{shape}/{stage}")
            print(f"{shape}/{stage}: {len(stage_records)} blocks")
            records.extend(stage_records)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", type=Path, required=True,
                        help="session directory (contains shape/<name>/<stage>/)")
    parser.add_argument("--out", type=Path, required=True, help="figure directory")
    parser.add_argument("--stamp", default="20261028", help="figure name stamp")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--sample-every", type=int, default=2,
                        help="keep one block frame out of N (size budget)")
    parser.add_argument("--single-shape", default="square")
    parser.add_argument("--single-stage", default="mono12_cv")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    records = _stage_records(args.session, SHAPES, STAGES)
    if not records:
        print("no records; nothing to render")
        return 1
    every = max(1, int(args.sample_every))
    sampled = records[::every]
    title = (f"多形状 × 四段 · {len(records)} 个已实测块 · 每 {every} 块取 1 · "
             f"{'/'.join(SHAPES)}")
    target = args.out / f"process_s1200_{args.stamp}.gif"
    recording.render_process_gif(sampled, target, fps=args.fps, title=title)
    print(f"{target} {target.stat().st_size / 1e6:.1f} MB · {len(sampled)} frames")

    single_run = args.session / "shape" / args.single_shape / args.single_stage
    if (single_run / "manifest.json").is_file():
        single = recording.block_records(single_run, f"{args.single_shape}/{args.single_stage}")
        single_target = args.out / "multishape_scale_wallclock_mono12_cv_process.gif"
        recording.render_process_gif(single, single_target, fps=args.fps,
                                     title=f"{args.single_shape}/{args.single_stage}")
        print(f"{single_target} {single_target.stat().st_size / 1e6:.1f} MB · {len(single)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
