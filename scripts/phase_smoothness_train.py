"""Does phase-domain smoothness regularisation help this forward model? Paired, 3 seeds.

Prediction being tested
-----------------------
**It will not, and for a reason already measured.**

The proposal argues for ``||grad phi||_1`` on the SLM phase: "limit the local rate of
change, otherwise you command modulation beyond the panel's spatial bandwidth product and
get zero-order enhancement, speckle and efficiency loss". The motivation is sound, and it
is the only item on that list this repo had not already tested.

But this model has **no overfitting gap to close**. Measured: val/train error ratio 1.08,
a flat learning curve (128 /+0.8767, 256 /+0.8850, 512 /+0.8860, 768 /+0.8851), and every
regulariser made it *worse* -- ``l2_penalty=1e-3`` -0.025, ``1e-2`` -0.253,
``weight_decay=1e-2`` -0.207, all 0/3 paired, and reducing capacity (``n_max`` 20 -> 15)
changed nothing (-0.0005).

``l2_penalty`` is *already* a coefficient-domain regulariser and it failed at every
strength. Phase smoothness is the same lever with better conditioning: L2 on Zernike
coefficients is isotropic in mode index, which is wrong, because for one radian of
coefficient the high-order modes have far steeper phase gradients than the low-order
ones. TV on the phase removes that anisotropy. So the proposal's term should be the
*better* version of a lever that is already measured to be harmful.

Therefore: expect the paired delta to be <= 0, and to get *more* negative as the weight
rises. Confirming or refuting that is the point of the run; either answer is recorded.

What would falsify it: a weight at which the paired delta is positive in >=2/3 seeds.
That would mean there *is* a generalisation gap that phase-domain structure -- rather than
capacity -- is what closes, and the premise of every ``l2``/``weight_decay`` row above
would need revisiting.

Run::

    python scripts/phase_smoothness_train.py

Writes ``report/loss_defects/phase_smoothness_train.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "report" / "loss_defects"

GRID = 64
N_MAX = 15
PADDING = 12  # the calibrated value; see report/loss_defects report section 6
EPOCHS = 50
SEEDS = (0, 1, 2)
MAX_TRAIN = 512
MAX_VAL = 128
BS = 64
LR = 0.01

#: Weight ladder. Includes 0.0 as the literal incumbent so the comparison cannot drift.
WEIGHTS = (0.0, 0.003, 0.03, 0.3)


def _build(seed: int):
    from scripts.compare_unet_baseline import _peak_normalise, _select_records, _inputs
    from ml.hwdataset import HwPhaseImageDataset, MaterialiserConfig, build_hw_index
    from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel
    from ml.zernike.train_amp import AmpTrainConfig, collect_split

    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    index = index.filter(families=["slm_zernike_shaping"])
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=GRID), use_cache=True)
    cfg = AmpTrainConfig(
        families=("slm_zernike_shaping",), n_max=N_MAX, grid=GRID, epochs=EPOCHS, lr=LR,
        batch_size=BS, max_train=MAX_TRAIN, max_val=MAX_VAL, beam_samples=32,
        use_wandb=False, save_checkpoint=False, seed=seed,
    )
    torch.manual_seed(seed)
    tr, va = _select_records(dataset, cfg)
    tr_t = collect_split(dataset, tr, torch.device("cpu"))
    va_t = collect_split(dataset, va, torch.device("cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.cat([tr_t["phase_cos"], tr_t["phase_sin"]], dim=1).to(device)
    y = _peak_normalise(tr_t["target"].clone()).to(device)
    xv = _inputs(va_t).to(device)
    yv = _peak_normalise(va_t["target"].clone()).to(device)
    return cfg, x, y, xv, yv, device


def _train_one(weight: float, seed: int, data) -> dict:
    """One paired cell: same split, same seed, same budget; only the weight changes."""
    from ml.zernike.losses import LossConfig
    from ml.zernike.metrics import batch_image_metrics
    from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel

    cfg, x, y, xv, yv, device = data
    torch.manual_seed(seed)
    model = ZernikeAmpModel(ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    weights = LossConfig(w_mse=1.0, w_phase_smooth=weight)
    n = x.shape[0]
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, BS):
            idx = perm[s : s + BS]
            opt.zero_grad(set_to_none=True)
            pred = model(x[idx, :1], x[idx, 1:])
            loss = ((pred - y[idx]) ** 2).flatten(1).mean(dim=1).mean()
            if weight:
                from ml.zernike.losses import phase_smoothness_penalty

                loss = loss + weight * phase_smoothness_penalty(model.correction_phase())
            loss.backward()
            opt.step()
    with torch.no_grad():
        pv = torch.cat([model(xv[s : s + 256, :1], xv[s : s + 256, 1:]) for s in range(0, xv.shape[0], 256)])
    img = batch_image_metrics(pv, yv)
    with torch.no_grad():
        phi = model.correction_phase()
        tv = float(
            (phi[1:, :] - phi[:-1, :]).abs().mean() + (phi[:, 1:] - phi[:, :-1]).abs().mean()
        )
    return {
        "weight": weight,
        "seed": seed,
        "r2": float(img["r2"]),
        "ssim": float(img["ssim"]),
        "mse": float(img["mse"]),
        "phase_tv_rad_per_px": tv,
        "phase_pv_rad": float(phi.amax() - phi.amin()),
        "max_abs_coeff": float(model.coefficients.abs().max()),
        "train_s": time.perf_counter() - t0,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} epochs={EPOCHS} seeds={SEEDS} weights={WEIGHTS}")

    rows: list[dict] = []
    for seed in SEEDS:
        data = _build(seed)
        print(f"  seed {seed}: split built", flush=True)
        for w in WEIGHTS:
            r = _train_one(w, seed, data)
            rows.append(r)
            print(
                f"    w={w:<6} R2={r['r2']:+.4f} SSIM={r['ssim']:.4f} "
                f"phaseTV={r['phase_tv_rad_per_px']:.4f} PV={r['phase_pv_rad']:.2f}rad "
                f"max|c|={r['max_abs_coeff']:.3f} ({r['train_s']:.0f}s)",
                flush=True,
            )

    base = {r["seed"]: r for r in rows if r["weight"] == 0.0}
    verdict: dict[str, dict[str, object]] = {}
    for w in WEIGHTS:
        if w == 0.0:
            continue
        sel = [r for r in rows if r["weight"] == w]
        d = [r["r2"] - base[r["seed"]]["r2"] for r in sel]
        ds = [r["ssim"] - base[r["seed"]]["ssim"] for r in sel]
        mean = float(np.mean(d))
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        verdict[str(w)] = {
            "d_r2_paired": mean,
            "d_r2_se": se,
            "beats_baseline": int(sum(1 for v in d if v > 0)),
            "n": len(d),
            "d_ssim_paired": float(np.mean(ds)),
            "phase_tv": float(np.mean([r["phase_tv_rad_per_px"] for r in sel])),
            "phase_pv_rad": float(np.mean([r["phase_pv_rad"] for r in sel])),
        }

    base_tv = float(np.mean([r["phase_tv_rad_per_px"] for r in base.values()]))
    base_pv = float(np.mean([r["phase_pv_rad"] for r in base.values()]))
    print("\n--- paired verdict (vs w=0, same seed) ---")
    for w, v in verdict.items():
        print(
            f"  w={w:<6} dR2={v['d_r2_paired']:+.4f} +/- {v['d_r2_se']:.4f} "
            f"({v['beats_baseline']}/{v['n']})  dSSIM={v['d_ssim_paired']:+.4f}  "
            f"phaseTV {base_tv:.4f}->{v['phase_tv']:.4f}  PV {base_pv:.2f}->{v['phase_pv_rad']:.2f}"
        )
    helped = [w for w, v in verdict.items() if v["beats_baseline"] >= 2]
    print(f"\nweights that helped in >=2/{len(SEEDS)} seeds: {helped or 'NONE'}")
    print("PREDICTION was 'none of them help' ->", "CONFIRMED" if not helped else "REFUTED")

    (OUT_DIR / "phase_smoothness_train.json").write_text(
        json.dumps(
            {
                "config": {
                    "grid": GRID, "n_max": N_MAX, "padding": PADDING, "epochs": EPOCHS,
                    "seeds": list(SEEDS), "weights": list(WEIGHTS), "max_train": MAX_TRAIN,
                    "max_val": MAX_VAL, "lr": LR, "batch_size": BS, "device": str(device),
                },
                "baseline": {"phase_tv": base_tv, "phase_pv_rad": base_pv},
                "rows": rows,
                "verdict": verdict,
                "prediction": "no weight helps in >=2/3 seeds",
                "outcome": "CONFIRMED" if not helped else "REFUTED",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_DIR / 'phase_smoothness_train.json'}")


if __name__ == "__main__":
    main()