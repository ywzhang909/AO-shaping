"""Offline GPU micro-benchmark for the transfer optimizer's K-candidate path.

Measures one ``DifferentiableUniformityOptimizer.update()`` step (model mode) on a
REAL ``FrozenDefocusForward`` (torch FFT on CUDA), across grid size x target shape
x K.  This isolates optimizer compute from SLM-write / CCD-exposure wall clock, which
dominates the end-to-end hardware loop (optimizer ~<0.5% there).

OFFLINE ONLY: opens no SLM / CCD / laser, touches no DeviceLock.  CUDA event timing
with explicit synchronize(), plus warmup to exclude kernel JIT / allocator effects.

Same measurement contract as report/shape_trials/candidate_parallelism_20261117.md
section 3, but for the transfer (``--candidates K``) path rather than the
``train_beam_shaping(n_candidates=)`` path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _make_forward(size: int, fft_size: int, device: str):
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
        FrozenDefocusParameters,
    )

    params = FrozenDefocusParameters(
        defocus_diopter=-0.1, origin_xy=(1360.0, 1020.0),
        amplitude_center_xy=(960.0, 600.0), amplitude_width_xy=(400.0, 300.0),
        residual_noll_radians={5: 0.3}, gain=25.0, background=1.0, sigma_ccd_px=2.0,
        ramp_matrix=((8000.0, 0.0), (0.0, -8000.0)),
        fft_size=fft_size, pixel_samples=1, grid_size=size,
    )
    window = 13
    offsets = np.arange(-(window - 1), window, 2.0)
    xx, yy = np.meshgrid(1360.0 + offsets, 1020.0 + offsets, indexing="xy")
    forward = FrozenDefocusForward(params, np.stack([xx, yy], axis=-1), device=device)
    return forward, window


def _make_objective(window: int, shape: str):
    """Target masks live on the CAMERA sampling grid (forward output window), not
    on the phase-control grid.  With the synthetic forward below the window is 13."""
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        UniformityObjective,
    )

    size = window
    box = np.zeros((size, size), dtype=bool)
    core = np.zeros((size, size), dtype=bool)
    if shape == "square":
        lo, hi = int(size * 0.15), int(size * 0.85)
        box[lo:hi, lo:hi] = True
        c0, c1 = int(size * 0.4), int(size * 0.6)
        core[c0:c1, c0:c1] = True
    elif shape == "star":
        yy, xx = np.mgrid[0:size, 0:size]
        cy = cx = (size - 1) / 2.0
        r = np.hypot(xx - cx, yy - cy)
        ang = np.arctan2(yy - cy, xx - cx)
        star = r < (size * 0.35 * (0.6 + 0.4 * np.cos(6 * ang)))
        box = star
        core = r < size * 0.12
    elif shape == "triangle":
        yy, xx = np.mgrid[0:size, 0:size]
        n = size - 1
        inside = (xx / n <= 1) & (yy <= xx) & (yy <= (n - xx))
        box = inside
        core = (yy > 0.45 * n) & (yy < 0.6 * n) & (abs(xx - 0.5 * n) < 0.08 * n)
    elif shape == "ring":
        yy, xx = np.mgrid[0:size, 0:size]
        r = np.hypot(xx - (size - 1) / 2, yy - (size - 1) / 2)
        box = (r > size * 0.2) & (r < size * 0.38)
        core = (r > size * 0.27) & (r < size * 0.31)
    else:
        raise ValueError(shape)
    return UniformityObjective(box, core, w_phase=0.5)


def bench_one(size: int, fft_size: int, shape: str, k: int, reps: int, warmup: int,
              device: str) -> dict:
    import torch

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        DifferentiableUniformityOptimizer,
    )

    forward, window = _make_forward(size, fft_size, device)
    objective = _make_objective(window, shape)
    seed = np.zeros((size, size), dtype=np.float32)

    times: list[float] = []
    peak_mem = 0
    for rep in range(warmup + reps):
        opt = DifferentiableUniformityOptimizer(
            forward, objective, seed, lr=0.01, step_rms_cap=0.1,
            device=device, n_candidates=k,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.update()  # model mode (no measured_intensity): forward+loss+backward+Adam+projection
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if rep >= warmup:
            times.append(dt)
            if device.startswith("cuda"):
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())
    return {
        "grid": size, "fft_size": fft_size, "shape": shape, "k": k,
        "median_s": round(statistics.median(times), 6),
        "min_s": round(min(times), 6),
        "reps": reps, "warmup": warmup,
        "peak_mem_mib": round(peak_mem / 1024**2, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="32,64,128")
    ap.add_argument("--fft-sizes", default="64,128,256",
                    help="comma list aligned with --sizes (fft>=2*grid)")
    ap.add_argument("--shapes", default="square,star,triangle,ring")
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    sizes = [int(x) for x in args.sizes.split(",")]
    ffts = [int(x) for x in args.fft_sizes.split(",")]
    if len(ffts) == 1 and len(sizes) > 1:
        ffts = ffts * len(sizes)
    if len(ffts) != len(sizes):
        raise SystemExit("--fft-sizes must align with --sizes")
    shapes = args.shapes.split(",")
    ks = [int(x) for x in args.ks.split(",")]

    rows: list[dict] = []
    for size, fft in zip(sizes, ffts):
        for shape in shapes:
            for k in ks:
                row = bench_one(size, fft, shape, k, args.reps, args.warmup, args.device)
                rows.append(row)
                print(f"grid={size:3d} fft={fft:3d} {shape:8s} K={k:2d} "
                      f"median={row['median_s']:.5f}s peak={row['peak_mem_mib']:.1f}MiB")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "measurement": "DifferentiableUniformityOptimizer.update() (model mode), "
                       "torch FFT forward+backward, cuda-synchronized wall clock",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
