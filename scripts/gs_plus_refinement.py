"""Does feedback refinement on top of GS beat GS alone? (the slm_gs_refine shape)

Attempt 10 showed canonical GS beats gradient inverse design from a random start by
+0.1171 +/- 0.0423 (t=+2.77) and is 4.7x more reproducible. The repo already ships a
two-stage architecture on that premise: `slm_gs_refine` runs GS pre-shaping, keeps it
only if it beats flat (bake-off), then refines with feedback (SPGD).

That premise has two halves and only one has been tested. This asks the other:
starting FROM a GS solution, does gradient refinement improve it further? If not,
the refinement stage is wasted budget and the honest architecture is GS-only.

Arms, all paired by the same random initial pupil phase so the comparison is within
one GS basin:
  gs_only      GS coefficients, no refinement
  gs_then_mse  GS coefficients + gradient steps on the model's MSE
  gs_then_phys GS coefficients + gradient steps on the physical ROI terms

The refinement is gradient-on-model because that is what is available offline; on the
bench it would be SPGD against a camera. The question asked here is only whether
refinement has anything to add to GS at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\Projects\TIFO\AO-shaping\src")))
sys.path.insert(0, str(Path(r"D:\Projects\TIFO\AO-shaping\scripts")))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ao_shaping.algorithm.signal_processing.gerchberg_saxton import gerchberg_saxton  # noqa: E402
from ao_shaping.utils.wavefront.zernike_calc import fit_zernike  # noqa: E402
from inverse_design_sim_eval import (  # noqa: E402
    ASPECT,
    GRID,
    N_MAX,
    PADDING,
    SIZE_FRAC,
    score_on_sim,
)
from ml.zernike.losses import LossConfig, composite_loss, roi_mask  # noqa: E402
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402

DRAWS = 16
REFINE_STEPS = 60
PIXEL_UM = 8.0
GS_ITERS = 40

ARMS = {
    "gs_then_mse": LossConfig(w_mse=1.0),
    "gs_then_phys": LossConfig(w_mse=0.0, w_pib=1.0, w_uniformity=1.0),
}


def square_amplitude(grid: int) -> np.ndarray:
    amp = np.zeros((grid, grid), dtype=np.float64)
    half_w = int(SIZE_FRAC * grid * ASPECT / 2)
    half_h = int(SIZE_FRAC * grid / 2)
    c = grid // 2
    amp[c - half_h : c + half_h, c - half_w : c + half_w] = 1.0
    return amp


def gs_coefficients(init_phase: np.ndarray, grid: int) -> np.ndarray:
    gs = gerchberg_saxton(
        np.ones((grid, grid), dtype=np.float64),
        square_amplitude(grid),
        iterations=GS_ITERS,
        cell_spacing=PIXEL_UM * 1e-6,
        propagation="fft",
    )
    combined = init_phase + np.asarray(gs.phase, dtype=np.float64)
    # [1:] drops piston; the model holds the K non-piston Noll modes.
    return fit_zernike(combined, n_max=N_MAX)[1:]


def refine(start: np.ndarray, weights: LossConfig, mask, target, steps: int) -> np.ndarray:
    model = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING, normalization="peak")
    )
    with torch.no_grad():
        model.coefficients.copy_(torch.as_tensor(np.asarray(start, dtype=np.float32)))
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        composite_loss(model.correction_far_field(), target, mask, weights)[
            "_mean_total"
        ].backward()
        opt.step()
    return model.coefficients_array()


def main() -> None:
    mask = roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle", SIZE_FRAC * GRID, ASPECT)
    target = torch.as_tensor(square_amplitude(GRID), dtype=torch.float32)[None, None]
    flat = score_on_sim(np.zeros(N_MAX * (N_MAX + 3) // 2), GRID, N_MAX)["shape_sum"]

    print("=" * 92)
    print("DOES REFINEMENT ADD ANYTHING TO GS?  (the slm_gs_refine two-stage shape)")
    print("=" * 92)
    print(f"draws={DRAWS} refine_steps={REFINE_STEPS} gs_iters={GS_ITERS} flat={flat:.4f}")

    rows = []
    for d in range(DRAWS):
        init = np.random.default_rng(77000 + d).normal(0.0, 0.6, (GRID, GRID))
        gs_c = gs_coefficients(init, GRID)
        row = {"draw": d, "gs_only": score_on_sim(gs_c, GRID, N_MAX)["shape_sum"]}
        for name, weights in ARMS.items():
            refined = refine(gs_c, weights, mask, target, REFINE_STEPS)
            row[name] = score_on_sim(refined, GRID, N_MAX)["shape_sum"]
        rows.append(row)

    def arr(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows])

    print(f"\n{'draw':>6}{'GS only':>11}{'GS+MSE':>11}{'GS+phys':>11}")
    for r in rows:
        print(f"{r['draw']:>6}{r['gs_only']:>11.4f}{r['gs_then_mse']:>11.4f}{r['gs_then_phys']:>11.4f}")

    print(f"\n{'':<11}{'mean':>10}{'sd':>10}{'min':>10}{'max':>10}{'vs GS':>10}{'positives':>11}")
    base = arr("gs_only")
    for name in ("gs_only", *ARMS):
        a = arr(name)
        delta = ""
        pos = ""
        if name != "gs_only":
            d = a - base
            delta = f"{d.mean():+.4f}"
            pos = f"{int((d > 0).sum())}/{len(d)}"
        print(
            f"{name:<11}{a.mean():>10.4f}{a.std():>10.4f}{a.min():>10.4f}"
            f"{a.max():>10.4f}{delta:>10}{pos:>11}"
        )

    print("\npaired vs GS only (same draw):")
    for name in ARMS:
        d = arr(name) - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        verdict = "REFINEMENT HELPS" if d.mean() > 2 * se else "no gain from refinement"
        print(
            f"  {name:<14} delta={d.mean():+.4f} +/- {se:.4f}  "
            f"positives={int((d > 0).sum())}/{len(d)}  t={d.mean() / se:+.2f}  -> {verdict}"
        )
    print(f"\nflat reference = {flat:.4f}")

    out = Path("report/loss_defects/gs_plus_refinement.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()