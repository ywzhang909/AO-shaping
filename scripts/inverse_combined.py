"""Inverse phase optimisation through the winning combined architecture.

Triggered by the user's condition: "if the improvement is effective, try inverse phase
optimisation". D2 (physics + U-Net residual, trained jointly) is effective -- +0.0426 +/- 0.0071
R2 over the incumbent, **5/5 seeds paired** -- so the condition is met and this runs.

Three arms, so the architecture's contribution to the INVERSE axis is attributable:

    A0  physics alone            the incumbent
    A5  U-Net alone              the capacity extreme
    D2  physics + U-Net residual  the winner

Padding is the trap that has voided three separate results in this project, so it is stated
rather than inherited: everything -- training targets, the model and the evaluator -- runs at
padding 1, which is the scale Gerchberg-Saxton designs on. The forward numbers here are
therefore NOT comparable with the padding-12 search (R2 is lower at padding 1); the arms are
comparable with each other, which is what this script is for. The script ASSERTS that GS beats
flat before reporting anything, and refuses to emit a number otherwise.

Also recorded, because it is what a reader will ask next: whether inverse capability
improves *at all* relative to doing nothing (flat).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from compare_unet_baseline import _peak_normalise, _select_records, _inputs  # noqa: E402
from forward_search import CoeffForward, _aperture, fwd  # noqa: E402
from forward_search_extra import PhysicsPlusResidual  # noqa: E402
from ml.hwdataset import (  # noqa: E402
    HwPhaseImageDataset,
    MaterialiserConfig,
    build_hw_index,
)
from ml.phase.unet import UNetGenerator  # noqa: E402
from ml.zernike import inverse_design as inv  # noqa: E402
from ml.zernike.losses import LossConfig, composite_loss, roi_mask  # noqa: E402
from ml.zernike.metrics import batch_image_metrics  # noqa: E402
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402
from ml.zernike.train_amp import AmpTrainConfig, collect_split  # noqa: E402

GRID, N_MAX, PADDING = inv.GRID, 20, 1
SEEDS = [0, 1, 2, 3, 4]
EPOCHS, BS, LR = 60, 64, 0.02
SIZE_FRAC, ASPECT = inv.SIZE_FRAC, inv.ASPECT
UNET_FEATURES = [16, 32, 64, 128, 256]
RESIDUAL_SCALE = 0.15


def build(name: str, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    cfg = ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING)
    if name == "A0":
        return ZernikeAmpModel(cfg)
    if name == "A5":
        return UNetGenerator(in_channels=2, features=UNET_FEATURES)
    return PhysicsPlusResidual(ZernikeAmpModel(cfg), 16, RESIDUAL_SCALE)


def train(model: nn.Module, x, y, seed, device, xv, yv):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    torch.manual_seed(seed)
    n = x.shape[0]
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(n)
        for s in range(0, n, BS):
            idx = order[s : s + BS]
            opt.zero_grad(set_to_none=True)
            torch.mean((fwd(model, x[idx].to(device)) - y[idx].to(device)) ** 2).backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return torch.cat([fwd(model, xv[s : s + 256]) for s in range(0, xv.shape[0], 256)])


def inverse_score(model, basis, stacked, start, target, mask, device) -> float:
    wrapped = CoeffForward(model, basis, stacked).to(device).eval()
    theta = torch.nn.Parameter(torch.as_tensor(start, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([theta], lr=LR)
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        composite_loss(wrapped(theta), target, mask, LossConfig(w_mse=1.0))["_mean_total"].backward()
        opt.step()
    return inv.score_coefficients(
        theta.detach().cpu().numpy()[0], SIZE_FRAC, ASPECT, n_max=N_MAX, padding=PADDING
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 104)
    print("INVERSE PHASE OPTIMISATION through the winning combined architecture")
    print("=" * 104)
    print(f"seeds={SEEDS} padding={PADDING} (training, evaluator and GS target all share it)")

    gs = inv.gs_phase(SIZE_FRAC, ASPECT, iterations=40)
    flat = inv.score_phase(np.zeros((GRID, GRID)), SIZE_FRAC, ASPECT, padding=PADDING)
    gs_score = inv.score_phase(gs, SIZE_FRAC, ASPECT, padding=PADDING)
    if gs_score <= flat:
        print("STOPPING: the GS target does not beat flat, so the target scale is wrong and")
        print("every inverse number below would be meaningless.")
        raise SystemExit(1)
    print(f"references: flat={flat:.4f}  GS={gs_score:.4f}  (GS - flat = {gs_score - flat:+.4f})")

    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    index = index.filter(families=["slm_zernike_shaping"])
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=GRID), use_cache=True)
    cfg0 = AmpTrainConfig(
        families=("slm_zernike_shaping",), n_max=N_MAX, grid=GRID, epochs=EPOCHS, lr=LR,
        batch_size=BS, max_train=512, max_val=128, beam_samples=16, use_wandb=False,
        save_checkpoint=False,
    )
    basis = ZernikeAmpModel(ZernikeAmpConfig(n_max=N_MAX, grid=GRID)).basis.detach().clone().to(device)
    n_modes = basis.shape[0]
    target = inv.target_tensor(GRID, SIZE_FRAC, ASPECT).to(device)
    mask = roi_mask(
        (GRID, GRID), (GRID / 2, GRID / 2), "rectangle", SIZE_FRAC * GRID, ASPECT
    ).to(device)

    rows: list[dict] = []
    for name in ("A0", "A5", "D2"):
        for seed in SEEDS:
            cfg = AmpTrainConfig(**{**cfg0.__dict__, "seed": seed})
            torch.manual_seed(seed)
            tr, va = _select_records(dataset, cfg)
            t = collect_split(dataset, tr, device)
            v = collect_split(dataset, va, device)
            xv, yv = _inputs(v), _peak_normalise(v["target"].clone())
            x = torch.cat([t["phase_cos"], t["phase_sin"]], dim=1)
            y = _peak_normalise(t["target"].clone())
            model = build(name, seed).to(device)
            pred = train(model, x, y, seed, device, xv, yv)
            img = batch_image_metrics(pred, yv)
            stacked = isinstance(model, UNetGenerator)
            rng = np.random.default_rng(4242 + seed)
            row = {
                "arm": {"A0": "A0 physics", "A5": "A5 unet", "D2": "D2 physics+residual"}[name],
                "seed": seed, "r2": float(img["r2"]), "ssim": float(img["ssim"]),
                "inv_rand": inverse_score(
                    model, basis, stacked,
                    rng.normal(0.0, 0.6, (1, n_modes)).astype(np.float32), target, mask, device),
            }
            rows.append(row)
            print(f"  {row['arm']:<20} seed={seed} r2={row['r2']:+.4f} "
                  f"inv(rand)={row['inv_rand']:.4f}  (flat={flat:.4f})")

    print("\n" + "=" * 104)
    print(f"means over {len(SEEDS)} seeds -- flat={flat:.4f}; higher is better everywhere")
    print("=" * 104)
    print(f"{'arm':<22}{'R2':>9}{'inv(rand)':>11}{'vs flat':>10}{'d(inv) vs A0':>14}{'pos':>7}")
    base = {r["seed"]: r["inv_rand"] for r in rows if r["arm"] == "A0 physics"}
    for name in ("A0 physics", "A5 unet", "D2 physics+residual"):
        sel = [r for r in rows if r["arm"] == name]
        d = np.array([r["inv_rand"] - base[r["seed"]] for r in sel])
        m = float(np.mean([r["inv_rand"] for r in sel]))
        print(f"{name:<22}{np.mean([r['r2'] for r in sel]):>+9.4f}{m:>11.4f}"
              f"{m - flat:>+10.4f}{d.mean():>+14.4f}{int((d > 0).sum()):>4}/{len(d)}")

    out = ROOT / "report" / "loss_defects" / "inverse_combined.json"
    out.write_text(json.dumps({"flat": flat, "gs": gs_score, "padding": PADDING, "rows": rows},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()