"""Offline GS + Zernike-8 / pupil-residual comparison; no device imports.

Run ``python scripts/slm/benchmark_gsnet_hybrid.py``. All values are synthetic FFT
metrics and must not be read as camera or SLM performance.
"""

from __future__ import annotations

import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ml.gsnet.dataset import compute_gs_phase, make_target
from ml.gsnet.hybrid_phase import HybridPupilPhase, far_field_intensity
from ml.gsnet.losses import ShapingLosses


def run_benchmark(output_root: Path | None = None, *, objective: str = "composite") -> Path:
    if objective not in {"composite", "pearson"}:
        raise ValueError("objective must be composite or pearson")
    if output_root is None:
        output_root = Path(__file__).resolve().parents[2] / "data/debug/gsnet_hybrid"
    torch.set_num_threads(2)
    torch.manual_seed(0)
    grid, center, radius, steps, lr = 64, (27, 34), 20, 40, 0.05
    yy, xx = np.mgrid[:grid, :grid]
    aperture = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2)
    source = torch.from_numpy(
        (aperture * np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * 15**2)))
        .astype(np.float32)
    )
    target = torch.from_numpy(make_target("square", 10, grid))
    gs = compute_gs_phase(
        source.sqrt()[None, None], target.sqrt()[None, None], iterations=40
    )[0, 0].detach().numpy()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = output_root / stamp
    out_dir.mkdir(parents=True, exist_ok=False)
    np.save(out_dir / "source_intensity.npy", source.numpy())
    np.save(out_dir / "target_intensity.npy", target.numpy())
    results = {}
    for name, use_residual in (("zernike8", False), ("zernike8_residual", True)):
        model = HybridPupilPhase(
            (grid, grid), center, radius, n_max=8,
            residual_grid=16, base_phase=gs,
        )
        if not use_residual:
            model.residual.requires_grad_(False)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
        history = []
        for step in range(steps + 1):
            phase = model()
            intensity = far_field_intensity(source, phase)
            pearson_loss = ShapingLosses.shaping_loss(
                source.sqrt()[None, None], phase[None, None], target[None, None]
            )
            mask = target > 0
            roi = intensity[mask]
            pib = roi.sum() / intensity.sum()
            cv = roi.std(unbiased=False) / roi.mean()
            score = 0.5 * pib + 0.5 / (1.0 + cv)
            loss = pearson_loss if objective == "pearson" else 1.0 - score
            history.append({
                "step": step, "pearson_loss": float(pearson_loss.detach()),
                "pib": float(pib.detach()), "cv": float(cv.detach()),
                "score": float(score.detach()),
            })
            if step == steps:
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        np.save(out_dir / f"{name}_phase.npy", model().detach().numpy())
        np.save(out_dir / f"{name}_intensity.npy", intensity.detach().numpy())
        results[name] = history

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "kind": "synthetic_fft_only", "objective": objective,
                "residual_coordinates": "pupil_local",
                "grid": grid, "pupil_center_xy": center, "radius_px": radius,
                "gs_iterations": 40, "n_max": 8, "residual_grid": 16,
                "steps": steps, "lr": lr, "seed": 0, "histories": results,
            },
            indent=2,
        ), encoding="utf-8",
    )
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--objective", choices=("composite", "pearson"), default="composite")
    args = parser.parse_args()
    print(run_benchmark(objective=args.objective))
