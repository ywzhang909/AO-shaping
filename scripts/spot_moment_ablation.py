"""Does the anchored spot-size term actually improve the forward fit?

The term was added on the argument that a pixel-wise MSE is dominated by the bright
core, so it barely notices a prediction that matches the peak while getting the spread
wrong. That is a hypothesis, and this repo's rule applies: measure it paired, on real
data, before believing it.

Protocol: real corpus samples, one forward model trained per configuration from the
same initialisation and the same steps, paired by sample. Reported on held-out frames.

Configurations:
  mse                 incumbent baseline
  mse + moment        the new anchored spot-size term at two weights
  mse + moment + gap  the full anchored set

The forward model is *untrained* (coefficients at their zero init) unless fitted, so
this measures what the loss does to the fit it produces -- not a claim about
attainable accuracy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\Projects\TIFO\AO-shaping\src")))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ml.hwdataset import build_hw_index  # noqa: E402
from ml.zernike.inverse_design import load_corpus_sample  # noqa: E402
from ml.zernike.losses import (  # noqa: E402
    LossConfig,
    composite_loss,
    roi_mask,
    second_moments,
    spot_moment_gap_term,
)
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402

GRID, N_MAX, PADDING = 64, 15, 12
SIZE_FRAC, ASPECT = 0.375, 4.0 / 3.0
N_SAMPLES, FIT_STEPS, LR = 12, 150, 0.02

CONFIGS = {
    "mse": LossConfig(w_mse=1.0),
    "mse+moment(0.5)": LossConfig(w_mse=1.0, w_spot_moment=0.5),
    "mse+moment(2)": LossConfig(w_mse=1.0, w_spot_moment=2.0),
    "mse+moment+gap": LossConfig(w_mse=1.0, w_spot_moment=0.5, w_shape_gap=1.0),
}


def fit(weights: LossConfig, cos, sin, target, conserve: bool) -> ZernikeAmpModel:
    torch.manual_seed(0)
    model = ZernikeAmpModel(
        ZernikeAmpConfig(
            n_max=N_MAX, grid=GRID, far_field_padding=PADDING,
            normalization="peak", conserve_energy=conserve,
        )
    )
    reference = model._normalize(target.clone())  # noqa: SLF001
    mask = roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle", SIZE_FRAC * GRID, ASPECT)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for _ in range(FIT_STEPS):
        opt.zero_grad()
        composite_loss(model(cos, sin), reference, mask, weights)["_mean_total"].backward()
        opt.step()
    return model


def score(model: ZernikeAmpModel, cos, sin, target) -> dict[str, float]:
    reference = model._normalize(target.clone())  # noqa: SLF001
    with torch.no_grad():
        pred = model(cos, sin)
        p = pred / pred.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
        t = reference / reference.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
        mse = float(((p - t) ** 2).mean())
        r2 = 1.0 - mse / float(((t - t.mean()) ** 2).mean())
        mask = roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle",
                        SIZE_FRAC * GRID, ASPECT)
        gap = float(spot_moment_gap_term(p, t, mask).mean())
        var_p = float(second_moments(p, mask)[2].mean())
        var_t = float(second_moments(t, mask)[2].mean())
    return {"r2": r2, "mse": mse, "moment_gap": gap, "var_pred": var_p, "var_true": var_t}


def main() -> None:
    index = build_hw_index(index_cache="data/hw_index_cache.json")
    rows: list[dict] = []
    print("=" * 96)
    print("ANCHORED SPOT-SIZE TERM: paired forward fit on the real corpus")
    print("=" * 96)
    print(f"samples={N_SAMPLES} steps={FIT_STEPS} lr={LR} n_max={N_MAX} grid={GRID}")
    print(f"{'config':<20}{'R2':>9}{'MSE':>10}{'moment gap':>12}{'var pred':>11}{'var true':>11}")

    for position in range(N_SAMPLES):
        cos, sin, target = load_corpus_sample(index, position, grid=GRID)
        for conserve in (False, True):
            for name, weights in CONFIGS.items():
                model = fit(weights, cos, sin, target, conserve)
                stats = score(model, cos, sin, target)
                rows.append({
                    "sample": position, "conserve_energy": conserve,
                    "config": name, **stats,
                })

    for conserve in (False, True):
        print(f"\n--- conserve_energy={conserve} ---")
        print(f"{'config':<20}{'R2':>9}{'MSE':>10}{'moment gap':>12}"
              f"{'var pred':>11}{'var true':>11}")
        for name in CONFIGS:
            sel = [r for r in rows if r["config"] == name and r["conserve_energy"] == conserve]
            print(f"{name:<20}{np.mean([r['r2'] for r in sel]):>9.4f}"
                  f"{np.mean([r['mse'] for r in sel]):>10.5f}"
                  f"{np.mean([r['moment_gap'] for r in sel]):>12.4f}"
                  f"{np.mean([r['var_pred'] for r in sel]):>11.3f}"
                  f"{np.mean([r['var_true'] for r in sel]):>11.3f}")

    print("\npaired deltas vs `mse` (same sample, same conserve flag):")
    for conserve in (False, True):
        for name in CONFIGS:
            if name == "mse":
                continue
            d = [r["r2"] for r in rows
                 if r["config"] == name and r["conserve_energy"] == conserve
                 and r["sample"] < N_SAMPLES]
            b = [r["r2"] for r in rows
                 if r["config"] == "mse" and r["conserve_energy"] == conserve
                 and r["sample"] < N_SAMPLES]
            delta = np.array(d) - np.array(b)
            se = delta.std(ddof=1) / np.sqrt(len(delta))
            print(f"  conserve={str(conserve):<5} {name:<20} dR2={delta.mean():+.4f}"
                  f" +/- {se:.4f}  positives={int((delta > 0).sum())}/{len(delta)}")

    out = Path("report/loss_defects/spot_moment_ablation.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()