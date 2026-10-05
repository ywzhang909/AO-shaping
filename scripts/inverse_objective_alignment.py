"""Is the inverse design objective aligned with the shaping metric?

Root cause found while debugging the under-determination test: `correction_far_field()`
depends ONLY on `self.coefficients`, and the design loss never reads the fitted
forward-model state. So inverse design is completely independent of what the model was
fitted to -- changing `n_train` or `l2` gives bit-identical inverse solutions. Only
`n_max` (the parameterisation size) matters.

That reframes attempt 7. The "model score" tested there was *literally the objective
being minimised*. So the result was: driving the design objective lower does NOT
produce better real far fields. The objective and the shaping metric are misaligned,
and the misalignment -- not the forward model's accuracy -- is what caps inverse
shaping quality.

This experiment tests that directly. Same starts, same steps, only the objective
changes; every run is scored on the independent sim. The decisive column is whether
the objective value and the sim score move together.

Also reported: `improves_objective_but_not_sim`, the count of restarts where the
objective got better while the sim score did not. That is misalignment made visible.
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

RESTARTS = 16
STEPS = 60
SIGMA = 0.6
N_MODES = N_MAX * (N_MAX + 3) // 2

OBJECTIVES = {
    "mse": LossConfig(w_mse=1.0),
    "physical": LossConfig(w_mse=0.0, w_pib=1.0, w_uniformity=1.0),
    "pib_only": LossConfig(w_mse=0.0, w_pib=1.0),
    "uni_only": LossConfig(w_mse=0.0, w_uniformity=1.0),
    "mse+physical": LossConfig(w_mse=1.0, w_pib=1.0, w_uniformity=1.0),
}


def run(start: np.ndarray, weights: LossConfig, mask, target, steps: int):
    model = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING, normalization="peak")
    )
    with torch.no_grad():
        model.coefficients.copy_(torch.as_tensor(start.astype(np.float32)))
    first = None
    for step in range(steps):
        opt = torch.optim.AdamW(model.parameters(), lr=0.02)
        opt.zero_grad()
        value = composite_loss(model.correction_far_field(), target, mask, weights)[
            "_mean_total"
        ]
        if step == 0:
            first = float(value.detach())
        value.backward()
        opt.step()
    with torch.no_grad():
        final = float(
            composite_loss(model.correction_far_field(), target, mask, weights)[
                "_mean_total"
            ]
        )
    return model.coefficients_array(), first, final


def main() -> None:
    mask = roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle", SIZE_FRAC * GRID, ASPECT)
    target = make_target()
    flat = score_on_sim(np.zeros(N_MODES), GRID, N_MAX)["shape_sum"]
    starts = [np.random.default_rng(31000 + r).normal(0.0, SIGMA, N_MODES) for r in range(RESTARTS)]

    print("=" * 100)
    print("OBJECTIVE ALIGNMENT: does optimising the objective improve the real far field?")
    print("=" * 100)
    print(f"restarts={RESTARTS} steps={STEPS} sigma={SIGMA} flat={flat:.4f}")

    rows = []
    for name, weights in OBJECTIVES.items():
        for idx, start in enumerate(starts):
            coeffs, first, final = run(start, weights, mask, target, STEPS)
            rows.append(
                {
                    "objective": name,
                    "restart": idx,
                    "obj_start": first,
                    "obj_final": final,
                    "obj_improvement": first - final,
                    "sim": score_on_sim(coeffs, GRID, N_MAX)["shape_sum"],
                }
            )

    def arr(obj: str, key: str) -> np.ndarray:
        return np.array([r[key] for r in rows if r["objective"] == obj])

    print(f"\n{'objective':<14}{'obj start':>11}{'obj final':>11}{'improved':>10}"
          f"{'sim mean':>11}{'sim sd':>9}{'vs flat':>10}")
    for name in OBJECTIVES:
        o0, o1 = arr(name, "obj_start"), arr(name, "obj_final")
        s = arr(name, "sim")
        print(
            f"{name:<14}{o0.mean():>11.5f}{o1.mean():>11.5f}{(o0 - o1).mean():>10.5f}"
            f"{s.mean():>11.4f}{s.std():>9.4f}{s.mean() - flat:>+10.4f}"
        )

    print("\npaired deltas vs `mse` (same restart = paired):")
    base = arr("mse", "sim")
    for name in OBJECTIVES:
        if name == "mse":
            continue
        d = arr(name, "sim") - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(
            f"  {name:<14} delta={d.mean():+.4f} +/- {se:.4f}  "
            f"positives={int((d > 0).sum())}/{len(d)}  t={d.mean() / se:+.2f}"
            if se
            else f"  {name:<14} delta={d.mean():+.4f}"
        )

    print("\nMISALIGNMENT: restarts where the objective improved but the sim did not")
    for name in OBJECTIVES:
        imp = arr(name, "obj_improvement")
        s = arr(name, "sim")
        median = np.argsort(imp)[len(imp) // 2]
        worse_half = s[imp <= np.median(imp)]
        better_half = s[imp > np.median(imp)]
        print(
            f"  {name:<14} objective improved in {int((imp > 0).sum())}/{len(imp)} restarts; "
            f"sim on best-objective half {better_half.mean():.4f} vs "
            f"worst-objective half {worse_half.mean():.4f} "
            f"(delta {better_half.mean() - worse_half.mean():+.4f})"
        )
    print("\n  If the objective were aligned, better-objective restarts would score higher on")
    print("  the sim. A delta near zero -- or negative -- is misalignment.")

    out = Path("report/loss_defects/objective_alignment.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()