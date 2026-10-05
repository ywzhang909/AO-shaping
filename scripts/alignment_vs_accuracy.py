"""Does gradient alignment improve with forward-model accuracy?

This resolves an apparent contradiction in this study:

* attempts 4-5: forward-model accuracy does NOT predict inverse-shaping quality
* attempts 8 & 11: the learned model's gradients are ANTI-aligned with real quality

Both used forward models fitted to a SINGLE sample -- attempt 5's worst rung
(held-out MSE 0.0031 vs 0.0008 for the best). If the anti-alignment is a *consequence*
of that poor fit rather than intrinsic to the model class, then a genuinely accurate
model should have usable gradients, and the two results reconcile: accuracy would
matter for the refinement step specifically, even though it does not matter for the
proposal step (GS).

Decisive test. For a ladder of forward models, varying n_max and training-set size,
measure held-out MSE, then ask the only question that matters for refinement:

    does refining FROM a GS start help or hurt, as a function of model accuracy?

If the delta trends toward 0 as accuracy improves, the misalignment is a fitting
problem and better forward models buy a usable refinement step. If delta stays
strongly negative, misalignment is intrinsic, forward accuracy is irrelevant to
inverse work, and attempts 4-5 and 8-11 agree after all.

Paired: the same GS draws are refined by every model in the ladder.
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
from inverse_design_accuracy_ladder import load_sample  # noqa: E402
from inverse_design_sim_eval import (  # noqa: E402
    ASPECT,
    GRID,
    N_MAX,
    PADDING,
    SIZE_FRAC,
    score_on_sim,
)
from ml.hwdataset import build_hw_index  # noqa: E402
from ml.zernike.losses import LossConfig, composite_loss, roi_mask  # noqa: E402
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402

DRAWS = 12
REFINE_STEPS = 60
FIT_STEPS = 200
FIT_LR = 0.02
PIXEL_UM = 8.0
GS_ITERS = 40

# The ladder spans a wide accuracy range: n_max sets capacity, n_train sets data.
LADDER = [
    {"n_max": 4, "n_train": 1},
    {"n_max": 4, "n_train": 8},
    {"n_max": 8, "n_train": 1},
    {"n_max": 8, "n_train": 8},
    {"n_max": 15, "n_train": 1},
    {"n_max": 15, "n_train": 8},
    {"n_max": 20, "n_train": 1},
    {"n_max": 20, "n_train": 8},
]


def square_amplitude(grid: int) -> np.ndarray:
    amp = np.zeros((grid, grid), dtype=np.float64)
    half_w = int(SIZE_FRAC * grid * ASPECT / 2)
    half_h = int(SIZE_FRAC * grid / 2)
    c = grid // 2
    amp[c - half_h : c + half_h, c - half_w : c + half_w] = 1.0
    return amp


def gs_coefficients(init_phase: np.ndarray, grid: int, n_max: int) -> np.ndarray:
    gs = gerchberg_saxton(
        np.ones((grid, grid), dtype=np.float64),
        square_amplitude(grid),
        iterations=GS_ITERS,
        cell_spacing=PIXEL_UM * 1e-6,
        propagation="fft",
    )
    combined = init_phase + np.asarray(gs.phase, dtype=np.float64)
    return fit_zernike(combined, n_max=n_max)[1:]


def fit_forward(samples, n_max: int, lr: float = FIT_LR) -> ZernikeAmpModel:
    torch.manual_seed(0)
    model = ZernikeAmpModel(
        ZernikeAmpConfig(
            n_max=n_max, grid=GRID, far_field_padding=PADDING, normalization="peak"
        )
    )
    references = [model._normalize(t.clone()) for _, _, t in samples]  # noqa: SLF001
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(FIT_STEPS):
        opt.zero_grad()
        (cos, sin, _), ref = samples[step % len(samples)], references[step % len(samples)]
        ((model(cos, sin) - ref) ** 2).mean().backward()
        opt.step()
    return model


def held_out_mse(model: ZernikeAmpModel, cos, sin, target) -> float:
    with torch.no_grad():
        ref = model._normalize(target.clone())  # noqa: SLF001
        return float(((model(cos, sin) - ref) ** 2).mean())


def refine(model: ZernikeAmpModel, start: np.ndarray, mask, target, steps: int) -> np.ndarray:
    with torch.no_grad():
        model.coefficients.copy_(torch.as_tensor(np.asarray(start, dtype=np.float32)))
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        composite_loss(model.correction_far_field(), target, mask, LossConfig(w_mse=1.0))[
            "_mean_total"
        ].backward()
        opt.step()
    return model.coefficients_array()


def main() -> None:
    index = build_hw_index(index_cache="data/hw_index_cache.json").filter(
        families=["slm_zernike_shaping"]
    )
    max_train = max(cfg["n_train"] for cfg in LADDER)
    need = DRAWS * max_train
    print(f"loading {need} real samples ...")
    pool = [load_sample(index, i) for i in range(need)]

    mask = roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle", SIZE_FRAC * GRID, ASPECT)
    target = torch.as_tensor(square_amplitude(GRID), dtype=torch.float32)[None, None]
    flat = score_on_sim(np.zeros(N_MAX * (N_MAX + 3) // 2), GRID, N_MAX)["shape_sum"]

    print("=" * 100)
    print("DOES GRADIENT ALIGNMENT IMPROVE WITH FORWARD-MODEL ACCURACY?")
    print("=" * 100)

    # GS reference per draw, per n_max (GS projection depends on n_max).
    rows: list[dict] = []
    for cfg in LADDER:
        n_max, n_train = cfg["n_max"], cfg["n_train"]
        for d in range(DRAWS):
            block = pool[d * max_train : (d + 1) * max_train]
            train, holdout = block[:n_train], block[-1]
            model = fit_forward(train, n_max)
            acc = held_out_mse(model, *holdout)

            init = np.random.default_rng(88000 + d).normal(0.0, 0.6, (GRID, GRID))
            gs_c = gs_coefficients(init, GRID, n_max)
            gs_sim = score_on_sim(gs_c, GRID, n_max)["shape_sum"]
            ref_sim = score_on_sim(refine(model, gs_c, mask, target, REFINE_STEPS), GRID, n_max)[
                "shape_sum"
            ]
            rows.append(
                {
                    "n_max": n_max,
                    "n_train": n_train,
                    "draw": d,
                    "held_out_mse": acc,
                    "gs": gs_sim,
                    "refined": ref_sim,
                }
            )

    def sel(**kw) -> list[dict]:
        return [
            r for r in rows if all(r[k] == v for k, v in kw.items())
        ]

    print(f"\n{'n_max':>6}{'n_train':>8}{'held-out MSE':>14}{'GS only':>10}"
          f"{'GS+refine':>11}{'delta':>10}{'positives':>11}")
    summary = []
    for cfg in LADDER:
        s = sel(**cfg)
        acc = float(np.mean([r["held_out_mse"] for r in s]))
        gs = np.array([r["gs"] for r in s])
        rf = np.array([r["refined"] for r in s])
        d = rf - gs
        summary.append({**cfg, "held_out_mse": acc, "gs": float(gs.mean()), "delta": float(d.mean())})
        print(
            f"{cfg['n_max']:>6}{cfg['n_train']:>8}{acc:>14.5f}{gs.mean():>10.4f}"
            f"{rf.mean():>11.4f}{d.mean():>+10.4f}{int((d > 0).sum())}/{len(d):>10}"
        )

    print(f"\nflat reference = {flat:.4f}")

    # The decision question: does the delta trend toward zero as accuracy improves?
    ordered = sorted(summary, key=lambda r: r["held_out_mse"])
    print("\ndelta ordered by forward accuracy (best model first):")
    for r in ordered:
        print(
            f"  MSE={r['held_out_mse']:.5f}  n_max={r['n_max']:>2} n_train={r['n_train']}  "
            f"delta={r['delta']:+.4f}"
        )
    best, worst = ordered[0], ordered[-1]
    print(
        f"\nmost accurate model delta={best['delta']:+.4f} vs "
        f"least accurate {worst['delta']:+.4f}"
    )
    trend = best["delta"] - worst["delta"]
    if abs(best["delta"]) < 0.05:
        verdict = "MISALIGNMENT IS A FITTING PROBLEM -- accurate models have usable gradients"
    elif trend > 0.10:
        verdict = "alignment improves with accuracy, but not enough to make refinement safe"
    else:
        verdict = "MISALIGNMENT IS INTRINSIC -- forward accuracy does not rescue refinement"
    print(f"VERDICT: {verdict}")

    out = Path("report/loss_defects/gradient_alignment_vs_accuracy.json")
    out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()