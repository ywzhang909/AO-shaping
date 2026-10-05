"""Inverse design step sweep with GENUINE random restarts.

Methodological bug found and fixed here: `ZernikeAmpModel.coefficients` is
initialised to zeros *deterministically*, so `torch.manual_seed(seed)` has no effect
on the inverse design loop -- every "seed" was byte-identical (measured std 0.0000
across 4 seeds, mean == best == worst). Any paired statistic computed over those
seeds had an effective n equal to the number of sample blocks, not the number of
rows.

Restarts are therefore drawn explicitly: `coefficients ~ N(0, sigma)` per restart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\Projects\TIFO\AO-shaping\src")))
sys.path.insert(0, str(Path(r"D:\Projects\TIFO\AO-shaping\scripts")))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from inverse_design_accuracy_ladder import make_target  # noqa: E402
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

STEP_LADDER = [20, 60, 150, 300, 600]
N_RESTARTS = 5
SIGMAS = [0.3, 1.0]
OBJECTIVES = {
    "physical": LossConfig(w_mse=0.0, w_pib=1.0, w_uniformity=1.0),
    "mse": LossConfig(w_mse=1.0),
}


def design(weights: LossConfig, start: np.ndarray, steps: int, lr: float = 0.02) -> np.ndarray:
    model = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING, normalization="peak")
    )
    with torch.no_grad():
        model.coefficients.copy_(torch.as_tensor(np.asarray(start, dtype=np.float32)))
    if steps == 0:
        return model.coefficients_array()
    mask = roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle", SIZE_FRAC * GRID, ASPECT)
    target = make_target()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        composite_loss(model.correction_far_field(), target, mask, weights)[
            "_mean_total"
        ].backward()
        opt.step()
    return model.coefficients_array()


def main() -> None:
    flat = score_on_sim(np.zeros(N_MAX * (N_MAX + 3) // 2), GRID, N_MAX)["shape_sum"]
    n_modes = N_MAX * (N_MAX + 3) // 2
    rows: list[dict] = []

    print("=" * 96)
    print("INVERSE DESIGN step sweep, GENUINE random restarts (independent sim)")
    print(f"flat = {flat:.4f};  restarts={N_RESTARTS};  starts ~ N(0, sigma)")
    print("=" * 96)

    for sigma in SIGMAS:
        starts = [np.random.default_rng(1000 + r).normal(0.0, sigma, n_modes) for r in range(N_RESTARTS)]
        for name, weights in OBJECTIVES.items():
            print(f"\nsigma={sigma}  objective={name}")
            print(f"{'steps':>7}{'mean':>10}{'best':>10}{'worst':>10}{'sd':>9}{'vs flat':>10}")
            for steps in STEP_LADDER:
                values = [
                    score_on_sim(design(weights, starts[r], steps), GRID, N_MAX)["shape_sum"]
                    for r in range(N_RESTARTS)
                ]
                for r, v in enumerate(values):
                    rows.append(
                        {
                            "sigma": sigma,
                            "objective": name,
                            "steps": steps,
                            "restart": r,
                            "sim": v,
                        }
                    )
                print(
                    f"{steps:>7}{np.mean(values):>10.4f}{max(values):>10.4f}"
                    f"{min(values):>10.4f}{np.std(values):>9.4f}"
                    f"{np.mean(values) - flat:>+10.4f}"
                )

    print("\n" + "=" * 96)
    print("paired deltas vs the previous rung (restart-matched => real progress)")
    print("=" * 96)
    for sigma in SIGMAS:
        for name in OBJECTIVES:
            print(f"  sigma={sigma}  {name}:")
            for a, b in zip([0] + STEP_LADDER, STEP_LADDER):
                base = flat if a == 0 else None
                deltas = []
                for r in range(N_RESTARTS):
                    sel = [
                        x["sim"]
                        for x in rows
                        if x["sigma"] == sigma
                        and x["objective"] == name
                        and x["steps"] == b
                        and x["restart"] == r
                    ]
                    if a == 0:
                        deltas.append(sel[0] - flat)
                    else:
                        ref = [
                            x["sim"]
                            for x in rows
                            if x["sigma"] == sigma
                            and x["objective"] == name
                            and x["steps"] == a
                            and x["restart"] == r
                        ]
                        deltas.append(sel[0] - ref[0])
                pos = sum(1 for d in deltas if d > 0)
                verdict = "REAL" if pos == len(deltas) else ("none" if pos == 0 else "mixed")
                tag = f"{a:>4} -> {b:<4}"
                print(
                    f"    {tag} delta={np.mean(deltas):+.4f} +/- {np.std(deltas):.4f} "
                    f"positives={pos}/{len(deltas)}  [{verdict}]"
                )

    print("\nbest-of-N restarts (practical recipe)")
    for sigma in SIGMAS:
        for name in OBJECTIVES:
            for steps in STEP_LADDER:
                per = {
                    r: [
                        x["sim"]
                        for x in rows
                        if x["sigma"] == sigma
                        and x["objective"] == name
                        and x["steps"] == steps
                        and x["restart"] == r
                    ][0]
                    for r in range(N_RESTARTS)
                }
                single = float(np.mean(list(per.values())))
                # Expected best-of-N for N up to 5, by bootstrap over restarts.
                rng = np.random.default_rng(7)
                draws = rng.choice(list(per.values()), size=(4000, N_RESTARTS), replace=True)
                best_of = float(np.mean(draws.max(axis=1)))
                print(
                    f"  sigma={sigma} {name:<9} steps={steps:<4} "
                    f"single={single:.4f}  best-of-{N_RESTARTS}={best_of:.4f} "
                    f"(+{best_of - single:.4f})"
                )

    out = Path("report/loss_defects/inverse_step_saturation.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()