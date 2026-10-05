"""Is "score on the model, keep the best" actually better than random?

The single-trial version of this check said: spearman(model_loss, sim) = -0.10 (no
usable rank correlation) yet argmin(model_loss) happened to land on rank 4/24 with
regret 0.0375. That is exactly the shape of result that turns out to be luck, so the
selection rule is measured over MANY independent draws here.

Protocol: repeat `trials` times. Each trial draws `restarts` fresh coefficient
vectors, inverse-designs each for a fixed number of steps, then scores every restart
two ways -- model loss (free) and independent sim shape_sum (expensive). For each
trial record what each selection rule achieves:

    pick_model   = sim[argmin(model_loss)]      the proposed cheap recipe
    oracle       = sim[argmax(sim)]              the best achievable
    random       = mean(sim)                     expected value of picking blind
    worst        = sim[argmin(sim)]

The decision-relevant quantity is pick_model - random: if it is reliably positive,
the recipe adds value even without rank correlation, because the metric is dominated
by a few very good restarts. If it straddles zero, the model score is useless for
selection and an external scorer is required.
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

TRIALS = 30
RESTARTS = 12
STEPS = 60
SIGMA = 0.6
N_MODES = N_MAX * (N_MAX + 3) // 2


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def ranks(x: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(x)).astype(np.float64)

    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom else 0.0


def design_once(start: np.ndarray, mask, target, weights) -> tuple[np.ndarray, float]:
    model = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING, normalization="peak")
    )
    with torch.no_grad():
        model.coefficients.copy_(torch.as_tensor(start.astype(np.float32)))
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(STEPS):
        opt.zero_grad()
        composite_loss(model.correction_far_field(), target, mask, weights)[
            "_mean_total"
        ].backward()
        opt.step()
    with torch.no_grad():
        loss = float(
            composite_loss(model.correction_far_field(), target, mask, weights)["_mean_total"]
        )
    return model.coefficients_array(), loss


def main() -> None:
    weights = LossConfig(w_mse=1.0)
    mask = roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle", SIZE_FRAC * GRID, ASPECT)
    target = make_target()
    flat = score_on_sim(np.zeros(N_MODES), GRID, N_MAX)["shape_sum"]

    trials = []
    correlations = []
    for t in range(TRIALS):
        rng = np.random.default_rng(90000 + t)
        starts = rng.normal(0.0, SIGMA, (RESTARTS, N_MODES))
        losses, sims = [], []
        for s in starts:
            coeffs, loss = design_once(s, mask, target, weights)
            losses.append(loss)
            sims.append(score_on_sim(coeffs, GRID, N_MAX)["shape_sum"])
        losses = np.array(losses)
        sims = np.array(sims)
        correlations.append(spearman(losses, sims))
        trials.append(
            {
                "pick_model": float(sims[int(np.argmin(losses))]),
                "oracle": float(sims.max()),
                "random": float(sims.mean()),
                "worst": float(sims.min()),
                "spread": float(sims.max() - sims.min()),
            }
        )

    def col(key: str) -> np.ndarray:
        return np.array([x[key] for x in trials])

    pick, oracle, rand, worst, spread = (
        col("pick_model"), col("oracle"), col("random"), col("worst"), col("spread")
    )
    gain = pick - rand

    print("=" * 84)
    print(f"SELECTION RULE over {TRIALS} independent trials x {RESTARTS} restarts")
    print("=" * 84)
    print(f"flat = {flat:.4f}")
    print("\nmean outcome of each rule (mean +/- sd over trials):")
    for name, values in (
        ("pick by model score", pick),
        ("oracle (argmax sim)", oracle),
        ("random restart", rand),
        ("worst restart", worst),
    ):
        print(f"  {name:<24} {values.mean():.4f} +/- {values.std():.4f}")

    print(f"\nrestart spread within a trial: {spread.mean():.4f} +/- {spread.std():.4f}")
    print(f"spearman(model_loss, sim) per trial: {np.mean(correlations):+.4f} "
          f"+/- {np.std(correlations):.4f}  (n={TRIALS})")

    print("\nTHE DECISION QUESTION -- does picking by model score beat picking blind?")
    print(f"  pick_model - random = {gain.mean():+.4f} +/- {gain.std():.4f}")
    print(f"  positive in {int((gain > 0).sum())}/{TRIALS} trials")
    se = gain.std() / np.sqrt(TRIALS)
    print(f"  standard error = {se:.4f}  ->  t = {gain.mean() / se:+.2f}" if se else "  n/a")

    frac = ((pick - rand) / np.maximum(oracle - rand, 1e-9)).mean()
    print(f"\n  fraction of the available gain captured: {frac:.3f}")
    print(f"  regret vs oracle: {(oracle - pick).mean():.4f} +/- {(oracle - pick).std():.4f}")

    verdict = (
        "MODEL SCORE IS USEFUL for selection"
        if gain.mean() > 2 * se
        else "MODEL SCORE IS NOT USEFUL -- an external scorer is required"
    )
    print(f"\nVERDICT: {verdict}")

    out = Path("report/loss_defects/restart_selection_trials.json")
    out.write_text(
        json.dumps(
            {"trials": trials, "mean_spearman": float(np.mean(correlations))}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()