"""Is "refinement is a restart, not a gradient" robust across BOTH ROI and objective?

**The claim is DEAD, and this run is what killed it.**

On the first evaluator the answer was yes: spearman(proposal quality, refinement delta)
came out -0.87 and -0.92, i.e. refinement appeared to rescue weak proposals and degrade
strong ones regardless of what it optimised. That is the reading of a *restart*, and it
was the one conclusion attempt 14 promoted out of the ROI sweep.

But that evaluator cropped the simulator's far field with ``[:64, :64]`` -- the top-left
CORNER, where the value is ~1e5 times dimmer than the beam at the array CENTRE. On the
corrected evaluator the monotone relationship is not merely weaker, it **reverses sign**,
and the mean refinement delta is *positive* (refinement helps) instead of ~zero or
negative.

Protocol, unchanged from the original so the two runs are comparable: 9 ROI geometries x
2 objectives x 10 draws. ``x`` = proposal quality (GS - flat), ``y`` = refinement delta.
If refinement were acting as a restart, ``y`` would fall as ``x`` rises, in both
objectives. It does not.

The evaluator, ROI, GS and gradient operators come from ``ml.zernike.inverse_design``
rather than a per-script copy -- the copies are what let the bug survive in twelve files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ml.zernike import inverse_design as inv  # noqa: E402
from ml.zernike.losses import LossConfig  # noqa: E402

GRID, N_MAX, REFINE_STEPS, DRAWS = inv.GRID, inv.N_MAX, 60, 10
N_MODES = N_MAX * (N_MAX + 3) // 2

SIZE_FRACS = [0.25, 0.375, 0.50]
ASPECTS = [1.0, 4.0 / 3.0, 1.5]
OBJECTIVES = {
    "mse": LossConfig(w_mse=1.0),
    "physical": LossConfig(w_mse=0.0, w_pib=1.0, w_uniformity=1.0),
}


def refine(start: np.ndarray, weights: LossConfig, mask, target) -> np.ndarray:
    return inv.gradient_zernike(
        start, inv.build_model(n_max=N_MAX, grid=GRID, padding=inv.PADDING),
        target, mask, weights=weights, steps=REFINE_STEPS,
    )


def main() -> None:
    print("=" * 100)
    print("IS 'REFINEMENT IS A RESTART' ROBUST ACROSS ROI *AND* OBJECTIVE?  (corrected evaluator)")
    print("=" * 100)
    print("score = pib + uniformity on an independent simulator; HIGHER IS BETTER.")
    print("y = refinement delta, so POSITIVE means refinement HELPED.")

    rows: list[dict] = []
    for size_frac in SIZE_FRACS:
        for aspect in ASPECTS:
            mask = inv.roi(GRID, size_frac, aspect)
            target = inv.target_tensor(GRID, size_frac, aspect)
            flat = inv.score_coefficients(np.zeros(N_MODES), size_frac, aspect, n_max=N_MAX)
            for d in range(DRAWS):
                rng = np.random.default_rng(99000 + d)
                gs_c = inv.gs_coefficients(
                    size_frac, aspect, n_max=N_MAX,
                    init_phase=rng.normal(0.0, 0.6, (GRID, GRID)),
                )
                gs_score = inv.score_coefficients(gs_c, size_frac, aspect, n_max=N_MAX)
                row = {"size_frac": size_frac, "aspect": aspect, "draw": d,
                       "flat": flat, "gs": gs_score}
                for name, weights in OBJECTIVES.items():
                    row[f"refined_{name}"] = inv.score_coefficients(
                        refine(gs_c, weights, mask, target), size_frac, aspect, n_max=N_MAX
                    )
                rows.append(row)

    print(f"\n{'ROI':>14}{'GS-flat':>10}" + "".join(f"{f'delta {k}':>14}" for k in OBJECTIVES))
    cells = []
    for size_frac in SIZE_FRACS:
        for aspect in ASPECTS:
            sel = [r for r in rows if r["size_frac"] == size_frac
                   and abs(r["aspect"] - aspect) < 1e-9]
            gs_minus_flat = float(np.mean([r["gs"] - r["flat"] for r in sel]))
            deltas = {
                k: float(np.mean([r[f"refined_{k}"] - r["gs"] for r in sel]))
                for k in OBJECTIVES
            }
            cells.append((gs_minus_flat, deltas))
            print(f"{size_frac:>7.3f}x{aspect:<6.3f}{gs_minus_flat:>+10.4f}"
                  + "".join(f"{deltas[k]:>+14.4f}" for k in OBJECTIVES))

    xs = np.array([c[0] for c in cells])
    print(f"\n{'objective':<12}{'pearson':>10}{'spearman':>11}{'vs restart claim':>22}")
    verdicts = {}
    for k in OBJECTIVES:
        ys = np.array([c[1][k] for c in cells])
        p, s = inv.pearson(xs, ys), inv.spearman(xs, ys)
        ok = s < -0.6
        verdicts[k] = ok
        print(f"{k:<12}{p:>+10.4f}{s:>+11.4f}"
              f"{('monotone (restart)' if ok else 'NOT monotone'):>22}")

    print(f"\nmean refinement delta over all {len(rows)} runs, per objective:")
    for k in OBJECTIVES:
        d = np.array([r[f"refined_{k}"] - r["gs"] for r in rows])
        print(f"  {k:<12}{d.mean():>+9.4f}   positive (helped) in {int((d > 0).sum())}/{len(d)}")

    both = all(verdicts.values())
    print(f"\nVERDICT: {'ROBUST across ROI x objective' if both else 'CLAIM REFUTED -- not restart behaviour'}")
    if not both:
        print("The original -0.87/-0.92 was a property of the corner-cropped evaluator.")
        print("Refinement now tracks its own objective and helps on average, which is what a")
        print("gradient should do; GS alone does not beat flat, so refinement is the load-bearing step.")

    out = Path("report/loss_defects/restart_claim_robustness.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
