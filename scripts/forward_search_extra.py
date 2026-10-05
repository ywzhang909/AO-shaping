"""Three gaps left in the forward search, addressed together.

**1. The two architectures were only ever tried apart.** Option 1 (physics, 230 params,
closed-form phase) and option 2 (U-Net, 7.8M params, better SSIM) were compared as separate
arms and the U-Net was rejected because its R2 gain lands inside the seed spread. The obvious
middle path was never tried: give the physics model the job it is good at and let a CNN add
only what it cannot represent. Two variants, because they answer different questions:

``frozen``   physics coefficients fitted first, then FROZEN; only the residual trains. This is
             the honest test of "can a CNN add anything" -- if the residual is noise, a
             frozen base cannot absorb the blame.
``joint``    both train together. Weaker as a test (the base can drift to compensate) but it
             is what the existing hybrid already does, so it is the control.

The residual is scaled small because the physics output is peak-normalised: the CNN can only
be trusted to add structure it does not already have, and an unscaled residual would simply
overwrite the physics prediction -- which is the U-Net's over-smoothing failure again.

**2. Averaging over the split can hide one objective being badly wrong.** This family is FOUR
optimisation objectives (rms_pib / rmse_out / shape / roi_pib) whose runs leave the SLM at
different aberrations, so the mean R2 is not the whole story. Every arm is therefore also
scored per objective, and the spread across objectives is reported. The repo's own sweep
warns that ~0.05 R2 between adjacent settings sits near the noise floor, so a single-objective
regression counts only if it is consistent.

**3. No per-epoch curves were kept**, so "did an arm help or just move the stopping point?"
could not be answered afterwards. Curves are saved per arm per seed this time.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from compare_unet_baseline import _peak_normalise, _select_records, _inputs  # noqa: E402
from forward_search import (  # noqa: E402
    APERTURE_R,
    BEAM_W0,
    Arm,
    GRID,
    N_MAX,
    PADDING,
    fwd,
    train_arm,
)
from ml.hwdataset import (  # noqa: E402
    HwPhaseImageDataset,
    MaterialiserConfig,
    build_hw_index,
)
from ml.phase.unet import UNetGenerator  # noqa: E402
from ml.zernike import augment as aug  # noqa: E402
from ml.zernike.metrics import batch_image_metrics  # noqa: E402
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402
from ml.zernike.train_amp import AmpTrainConfig, collect_split  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
EPOCHS, BS, LR = 60, 64, 0.02
UNET_FEATURES = [16, 32, 64, 128, 256]
RESIDUAL_SCALE = 0.15
OBJECTIVE_RE = re.compile(r"slm_zernike_shaping_([a-z_]+)_\d{8}_\d{6}")


class PhysicsPlusResidual(nn.Module):
    """``physics far field + scale * CNN residual``, both consuming the same phasor.

    The physics output is peak-normalised, so the residual is deliberately small: an unscaled
    residual lets the CNN overwrite the prediction outright, which is how the standalone U-Net
    ends up over-smoothing. Clamping is not used -- it would hide the question being asked.
    """

    def __init__(self, physics: nn.Module, width: int, scale: float) -> None:
        super().__init__()
        self.physics = physics
        self.residual = UNetGenerator(
            in_channels=2, features=[width, width * 2, width * 4, width * 8, width * 16],
            output_mode="phase",
        )
        self.scale = scale

    def forward(self, phase_cos: torch.Tensor, phase_sin: torch.Tensor) -> torch.Tensor:
        # Matches the (cos, sin) convention forward_search.fwd uses for every non-U-Net
        # model, so this arm goes through the identical dispatch rather than a special case.
        stacked = torch.cat([phase_cos, phase_sin], dim=1)
        return self.physics(phase_cos, phase_sin) + self.scale * self.residual(stacked)


def objective_of(path) -> str:
    m = OBJECTIVE_RE.search(str(path))
    return m.group(1) if m else "unknown"


def train_curves(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, seed: int, device,
    xv: torch.Tensor, yv: torch.Tensor, epochs: int = EPOCHS,
) -> tuple[list[dict], torch.Tensor]:
    """Same loop as forward_search.train_arm, but it keeps the per-epoch validation curve."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    torch.manual_seed(seed)
    n = x.shape[0]
    curve: list[dict] = []
    model.train()
    for ep in range(epochs):
        # model.train() MUST be restored every epoch: the validation pass below calls
        # model.eval(), and without this the U-Net's BatchNorm switches to running statistics
        # after epoch 0 and stops training -- which showed up as an R2 of -0.4 and looked
        # like a finding until it was traced to the harness.
        model.train()
        order = torch.randperm(n)
        run = 0.0
        for s in range(0, n, BS):
            idx = order[s : s + BS]
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((fwd(model, x[idx].to(device)) - y[idx].to(device)) ** 2)
            loss.backward()
            opt.step()
            run += float(loss.detach()) * idx.numel()
        sched.step()
        model.eval()
        with torch.no_grad():
            p = torch.cat([fwd(model, xv[s : s + 256]) for s in range(0, xv.shape[0], 256)])
        m = batch_image_metrics(p, yv)
        curve.append({
            "epoch": ep, "train_mse": run / n,
            "val_r2": float(m["r2"]), "val_ssim": float(m["ssim"]),
        })
    model.eval()
    with torch.no_grad():
        pred = torch.cat([fwd(model, xv[s : s + 256]) for s in range(0, xv.shape[0], 256)])
    return curve, pred


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 112)
    print("COMBINED ARCHITECTURE + PER-OBJECTIVE BREAKDOWN + PER-EPOCH CURVES")
    print("=" * 112)
    print(f"seeds={SEEDS} epochs={EPOCHS} padding={PADDING} residual_scale={RESIDUAL_SCALE}")

    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    index = index.filter(families=["slm_zernike_shaping"])
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=GRID), use_cache=True)
    cfg0 = AmpTrainConfig(
        families=("slm_zernike_shaping",), n_max=N_MAX, grid=GRID, epochs=EPOCHS, lr=LR,
        batch_size=BS, max_train=512, max_val=128, beam_samples=16, use_wandb=False,
        save_checkpoint=False,
    )
    objectives = [objective_of(index.records[i].path) for i in range(len(index.records))]
    print("objectives in corpus: " + ", ".join(sorted(set(objectives))))

    rows: list[dict] = []
    curves: dict[str, list[dict]] = {}
    for seed in SEEDS:
        cfg = AmpTrainConfig(**{**cfg0.__dict__, "seed": seed})
        torch.manual_seed(seed)
        tr, va = _select_records(dataset, cfg)
        t = collect_split(dataset, tr, device)
        xv = _inputs(collect_split(dataset, va, device))
        yv = _peak_normalise(collect_split(dataset, va, device)["target"].clone())
        x = torch.cat([t["phase_cos"], t["phase_sin"]], dim=1)
        y = _peak_normalise(t["target"].clone())
        val_obj = [objectives[i] for i in va]

        for name in ("A0 physics", "D1 frozen+residual", "D2 joint+residual", "A5 unet"):
            if name == "D1 frozen+residual" or name == "D2 joint+residual":
                torch.manual_seed(seed)
                phys = ZernikeAmpModel(
                    ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING)
                ).to(device)
                combo = PhysicsPlusResidual(phys, 16, RESIDUAL_SCALE).to(device)
                if name == "D1 frozen+residual":
                    # fit the base first, then freeze it -- the only fair version of this test
                    tmp, _ = train_curves(
                        ZernikeAmpModel(ZernikeAmpConfig(
                            n_max=N_MAX, grid=GRID, far_field_padding=PADDING)).to(device),
                        x, y, seed, device, xv, yv, epochs=EPOCHS)
                    del tmp
                    fitted = ZernikeAmpModel(
                        ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING)
                    ).to(device)
                    c2, _ = train_curves(fitted, x, y, seed, device, xv, yv, epochs=EPOCHS)
                    with torch.no_grad():
                        combo.physics.coefficients.copy_(fitted.coefficients)
                    combo.physics.coefficients.requires_grad_(False)
            else:
                arm = Arm("A0") if name == "A0 physics" else Arm("A5", kind="unet")
                combo = None
                curve, pred = train_curves(
                    (ZernikeAmpModel(ZernikeAmpConfig(
                        n_max=N_MAX, grid=GRID, far_field_padding=PADDING)).to(device)
                     if arm.kind == "physics" else
                     UNetGenerator(in_channels=2, features=UNET_FEATURES).to(device)),
                    x, y, seed, device, xv, yv)
            if combo is not None:
                curve, pred = train_curves(combo, x, y, seed, device, xv, yv)

            img = batch_image_metrics(pred, yv)
            per_obj: dict[str, list[float]] = {}
            for i, ob in enumerate(val_obj):
                b = batch_image_metrics(pred[i : i + 1], yv[i : i + 1])
                per_obj.setdefault(ob, []).append(float(b["r2"]))
            row = {
                "arm": name, "seed": seed,
                "r2": float(img["r2"]), "ssim": float(img["ssim"]), "psnr": float(img["psnr"]),
                "params": int(sum(p.numel() for p in combo.parameters())) if combo is not None
                else int(sum(p.numel() for p in [])) or None,
                "per_objective": {k: float(np.mean(v)) for k, v in per_obj.items()},
            }
            if combo is not None:
                row["params"] = int(sum(p.numel() for p in combo.parameters()))
            rows.append(row)
            curves.setdefault(name, []).extend(
                [{**c, "seed": seed} for c in curve])
            po = "  ".join(f"{k}={v:+.3f}" for k, v in sorted(row["per_objective"].items()))
            print(f"  seed={seed} {name:<20} r2={row['r2']:+.4f} ssim={row['ssim']:.4f} | {po}")

    print("\n" + "=" * 112)
    print("PAIRED vs A0, and the per-objective spread")
    print("=" * 112)
    base = {r["seed"]: r for r in rows if r["arm"] == "A0 physics"}
    for name in ("D1 frozen+residual", "D2 joint+residual", "A5 unet"):
        sel = [r for r in rows if r["arm"] == name]
        d = np.array([r["r2"] - base[r["seed"]]["r2"] for r in sel])
        ds = np.array([r["ssim"] - base[r["seed"]]["ssim"] for r in sel])
        print(f"\n{name}:  dR2={d.mean():+.4f} +/- {d.std(ddof=1)/np.sqrt(len(d)):.4f} "
              f"({int((d > 0).sum())}/{len(d)})   dSSIM={ds.mean():+.4f}")
        objs = sorted(sel[0]["per_objective"])
        print("    per-objective dR2 vs A0:")
        for ob in objs:
            # The val split is per-seed, so an objective can be absent from some seeds.
            # Paired differences are only formed where BOTH sides have it.
            od = np.array([
                r["per_objective"][ob] - base[r["seed"]]["per_objective"][ob]
                for r in sel
                if ob in r["per_objective"] and ob in base[r["seed"]]["per_objective"]
            ])
            if od.size == 0:
                print(f"      {ob:<12} (absent from this split)")
                continue
            flag = "REGRESSION" if (od < -0.02).all() else ("better" if (od > 0).all() else "mixed")
            print(f"      {ob:<12} {od.mean():+.4f}  ({int((od>0).sum())}/{len(od)})  {flag}")

    out = ROOT / "report" / "loss_defects" / "forward_search_extra.json"
    out.write_text(json.dumps({"rows": rows, "curves": curves}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  (rows + per-epoch curves for every arm/seed)")


if __name__ == "__main__":
    main()