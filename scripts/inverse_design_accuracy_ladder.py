"""Does forward-model accuracy affect inverse-shaping quality? (the accuracy ladder)

Answers the question ``scripts/inverse_design_sim_eval.py`` leaves open: every run
there used a single forward-model accuracy, so accuracy was never varied.

Method
------
Ladder rung = number of AdamW steps used to fit the forward model (``fit_steps=0``
is a randomly-initialised model, i.e. the "no information" control). Accuracy is
measured as **held-out** forward MSE on a different real sample than the one fitted
-- otherwise the rungs are not ordered by generalisation and the ladder is
meaningless. Confirmed monotone: MSE falls 0.00311 -> 0.00079 across the rungs.

Each fitted model is then inverse-designed and its phase scored on the independent
sim (``SimPibSystem``). ``design_steps=0`` scores the fitted coefficient vector
*directly*, with no inverse optimisation in front of it -- that is the only place
forward accuracy cannot hide behind the optimiser.

Result (see ``report/loss_defects/PROCESS.md``)
------------------------------------------------
* A no-information start is strictly worse (0/6 paired) -- forward accuracy does
  carry usable information.
* **More accuracy is not better.** The best start is ``fit_steps=15``
  (held-out MSE 0.00124); the *most* accurate model, ``fit_steps=200``
  (MSE 0.00079, a 36% lower error) is markedly worse as a start. Held-out forward
  MSE is therefore **not** a valid model-selection criterion for inverse shaping.
* Inverse design steps add a real, monotone gain (+0.045 / +0.093 / +0.116 at
  5 / 20 / 60 steps) which reduces but does not erase the dependence on the start.

Caveats: 3 sample pairs x 2 seeds per cell; the ``fit_steps=0`` rungs share one
random init (seed 0) so their variance is understated; the evaluator is a sim, not
the bench; and the mechanism behind the non-monotonicity is a hypothesis.

Usage
-----
    python scripts/inverse_design_accuracy_ladder.py
    python scripts/inverse_design_accuracy_ladder.py --pairs 2 --seeds 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from inverse_design_sim_eval import (  # noqa: E402
    ASPECT,
    GRID,
    N_MAX,
    PADDING,
    SIZE_FRAC,
    score_on_sim,
)
from ml.hwdataset import MaterialiserConfig, build_hw_index  # noqa: E402
from ml.hwdataset.records import Materialiser  # noqa: E402
from ml.zernike.losses import LossConfig, composite_loss, roi_mask  # noqa: E402
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402

FIT_LADDER = [0, 5, 15, 40, 80, 200]
DESIGN_LADDER = [0, 5, 20, 60]


def make_target() -> torch.Tensor:
    target = torch.zeros(1, 1, GRID, GRID)
    half_w = int(SIZE_FRAC * GRID * ASPECT / 2)
    half_h = int(SIZE_FRAC * GRID / 2)
    c = GRID // 2
    target[..., c - half_h : c + half_h, c - half_w : c + half_w] = 1.0
    return target


def load_sample(index, position: int):
    """Materialise one real corpus sample as (phase_cos, phase_sin, image)."""
    usable = [r for r in index.records if r.key is not None and r.key >= 0]
    sample = Materialiser(
        config=MaterialiserConfig(grid=GRID), use_cache=True
    ).materialise(usable[position])
    return (
        torch.as_tensor(np.asarray(sample.phase_cos, dtype=np.float32))[None, None],
        torch.as_tensor(np.asarray(sample.phase_sin, dtype=np.float32))[None, None],
        torch.as_tensor(np.asarray(sample.image, dtype=np.float32))[None, None],
    )


def fit_forward(cos, sin, target, steps: int, lr: float = 0.03) -> ZernikeAmpModel:
    """Fit the forward model; ``steps=0`` returns the random init (no information)."""
    torch.manual_seed(0)
    model = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING, normalization="peak")
    )
    if steps == 0:
        return model
    reference = model._normalize(target.clone())  # noqa: SLF001
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        optimiser.zero_grad()
        ((model(cos, sin) - reference) ** 2).mean().backward()
        optimiser.step()
    return model


def held_out_mse(model: ZernikeAmpModel, cos, sin, target) -> float:
    """Forward error on a sample the model was NOT fitted on."""
    with torch.no_grad():
        reference = model._normalize(target.clone())  # noqa: SLF001
        return float(((model(cos, sin) - reference) ** 2).mean())


def inverse_design(start: np.ndarray, seed: int, steps: int, lr: float = 0.02) -> np.ndarray:
    """Inverse-design from ``start``. ``steps=0`` returns ``start`` unchanged."""
    if steps == 0:
        return np.asarray(start, dtype=np.float64)
    torch.manual_seed(seed)
    model = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING, normalization="peak")
    )
    with torch.no_grad():
        model.coefficients.copy_(torch.as_tensor(np.asarray(start, dtype=np.float32)))
    mask = roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle", SIZE_FRAC * GRID, ASPECT)
    target = make_target()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        optimiser.zero_grad()
        composite_loss(
            model.correction_far_field(), target, mask, LossConfig(w_mse=1.0)
        )["_mean_total"].backward()
        optimiser.step()
    return model.coefficients_array()


def _mean(rows: list[dict], fit_steps: int, design_steps: int, key: str) -> float:
    selected = [
        r for r in rows if r["fit_steps"] == fit_steps and r["design_steps"] == design_steps
    ]
    return float(np.mean([r[key] for r in selected]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=3, help="fit/holdout sample pairs")
    parser.add_argument("--seeds", type=int, default=2, help="inverse seeds per cell")
    parser.add_argument("--index-cache", default="data/hw_index_cache.json")
    parser.add_argument("--out", default="report/loss_defects/accuracy_design_grid.json")
    args = parser.parse_args()

    index = build_hw_index(index_cache=args.index_cache).filter(
        families=["slm_zernike_shaping"]
    )
    flat = score_on_sim(np.zeros(N_MAX * (N_MAX + 3) // 2), GRID, N_MAX)["shape_sum"]

    rows: list[dict] = []
    for pair in range(args.pairs):
        cos_a, sin_a, tgt_a = load_sample(index, 2 * pair)
        cos_b, sin_b, tgt_b = load_sample(index, 2 * pair + 1)
        for fit_steps in FIT_LADDER:
            model = fit_forward(cos_a, sin_a, tgt_a, fit_steps)
            accuracy = held_out_mse(model, cos_b, sin_b, tgt_b)
            start = model.coefficients_array()
            for design_steps in DESIGN_LADDER:
                for seed in range(args.seeds):
                    coeffs = inverse_design(start, seed, design_steps)
                    rows.append(
                        {
                            "pair": pair,
                            "fit_steps": fit_steps,
                            "design_steps": design_steps,
                            "seed": seed,
                            "held_out_mse": accuracy,
                            "flat": flat,
                            "sim_shape_sum": score_on_sim(coeffs, GRID, N_MAX)["shape_sum"],
                        }
                    )

    print("=" * 88)
    print("INVERSE DESIGN: sim shape_sum (rows = forward fit_steps, cols = design_steps)")
    print("=" * 88)
    print(f"{'fit':>6}" + "".join(f"{d:>12}" for d in DESIGN_LADDER) + f"{'held-out MSE':>15}")
    summary = []
    for fit_steps in FIT_LADDER:
        cells = "".join(
            f"{_mean(rows, fit_steps, d, 'sim_shape_sum'):>12.4f}" for d in DESIGN_LADDER
        )
        mse = _mean(rows, fit_steps, 0, "held_out_mse")
        print(f"{fit_steps:>6}{cells}{mse:>15.5f}")
        for d in DESIGN_LADDER:
            summary.append(
                {
                    "fit_steps": fit_steps,
                    "design_steps": d,
                    "held_out_mse": mse,
                    "sim_shape_sum": _mean(rows, fit_steps, d, "sim_shape_sum"),
                }
            )

    print(f"\nflat reference shape_sum = {flat:.4f}")

    starts = {s: _mean(rows, s, 0, "sim_shape_sum") for s in FIT_LADDER}
    best = max(starts, key=lambda k: starts[k])
    print(f"\nQ1 accuracy axis at design_steps=0 (paired vs best rung fit_steps={best}):")
    print("    held-out MSE falls monotonically; shape_sum does NOT.")
    for fit_steps in FIT_LADDER:
        sel = [r for r in rows if r["design_steps"] == 0 and r["fit_steps"] == fit_steps]
        ref = [r for r in rows if r["design_steps"] == 0 and r["fit_steps"] == best]
        deltas = [s["sim_shape_sum"] - b["sim_shape_sum"] for s, b in zip(sel, ref)]
        print(
            f"    fit_steps={fit_steps:<4} held-out MSE={_mean(rows, fit_steps, 0, 'held_out_mse'):.5f}"
            f"  delta={np.mean(deltas):+.4f}  positives={sum(1 for d in deltas if d > 0)}/{len(deltas)}"
        )

    print("\nQ2 design axis (paired vs design_steps=0 on the same start):")
    for design_steps in DESIGN_LADDER:
        if design_steps == 0:
            continue
        deltas = []
        for fit_steps in FIT_LADDER:
            sel = [
                r for r in rows if r["design_steps"] == design_steps and r["fit_steps"] == fit_steps
            ]
            ref = [r for r in rows if r["design_steps"] == 0 and r["fit_steps"] == fit_steps]
            deltas += [s["sim_shape_sum"] - b["sim_shape_sum"] for s, b in zip(sel, ref)]
        print(
            f"    design_steps={design_steps:<4} delta={np.mean(deltas):+.4f}  "
            f"positives={sum(1 for d in deltas if d > 0)}/{len(deltas)}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())