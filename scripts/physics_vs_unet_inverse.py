"""Option 1 (physics, closed-form) vs option 2 (U-Net): forward accuracy AND invertibility.

Option 1 and option 2 are being weighed as alternatives for reducing the forward error.
Forward accuracy alone already favours the U-Net on record (+0.028 R2, +0.11 SSIM at
10-fold grouped CV), so repeating that measurement would decide nothing. The question still
open is the one the shaping workflow depends on:

    **can the model be inverted?**

Physics exposes an exact differentiable forward map, so inverse design is a chain rule away.
A U-Net is *also* differentiable in principle -- it consumes the phasor, and ``polar(1,
phase)`` is differentiable in the phase -- so "it is a neural net, therefore it cannot be
inverted" would be an assumption, not a result. This measures it.

Both arms get identical treatment:

  forward   held-out R2 / SSIM / PSNR, paired per seed
  inverse   gradient design on the Zernike coefficients FROM A RANDOM START, pushed
            through the model and scored on the corrected independent simulator
  refine    the same, started from a GS proposal

Both are scored at **padding 10**, the value both were trained at, because padding sets the
angular scale and comparing across scales is what made the earlier inverse panel meaningless
(pearson <= 0.27 at every padding tried).

Fairness detail that decides whether this measures anything: **one adapter, both arms.**
``_PhasorForward`` turns a coefficient vector into a phase via the shared canonical basis,
forms the phasor, and calls the model. So the parameterisation, the basis, the loss and the
optimiser are the same objects in both arms; the only difference is which module consumes
the phasor. Without that, one arm would be invertible-by-construction and the other would
merely be declared so.

The U-Net runs in ``eval()`` during the inverse steps: its BatchNorm would otherwise use
batch statistics, making the objective depend on which other samples happen to share the
batch -- the "gradient" would be partly an artefact of batch composition.
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

from compare_unet_baseline import (  # noqa: E402
    _inputs,
    _peak_normalise,
    _select_records,
    build_model,
    collect_split,
    fit,
    forward_for,
)
from ml.hwdataset import (  # noqa: E402
    HwPhaseImageDataset,
    MaterialiserConfig,
    build_hw_index,
)
from ml.zernike import inverse_design as inv  # noqa: E402
from ml.zernike.losses import LossConfig, composite_loss, roi_mask  # noqa: E402
from ml.zernike.metrics import batch_image_metrics  # noqa: E402
from ml.zernike.train_amp import AmpTrainConfig  # noqa: E402

SEEDS = [0, 1, 2]
EPOCHS = 60
N_MAX, GRID, PADDING = 20, inv.GRID, 10
UNET_FEATURES = [16, 32, 64, 128, 256]
SIZE_FRAC, ASPECT = inv.SIZE_FRAC, inv.ASPECT
WEIGHTS = LossConfig(w_mse=1.0)


class _PhasorForward(nn.Module):
    """``coefficients -> phase -> phasor -> model``, identical for both arms.

    Physics takes ``(cos, sin)``; the U-Net takes the stacked 2-channel pair. That
    calling-convention difference is each model's own API rather than a handicap, so it is
    absorbed here instead of giving one arm a richer path.
    """

    def __init__(self, model: nn.Module, basis: torch.Tensor, stacked: bool) -> None:
        super().__init__()
        self.model = model
        self.stacked = stacked
        self.register_buffer("basis", basis)

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        # (B, K) x (K, g, g) -> (B, 1, g, g). The channel axis is required by physics
        # (`phase_cos` is contracted as (B, 1, g, g)) and is exactly what the U-Net's
        # 2-channel input wants once cat'ed along dim=1.
        phase = torch.einsum("bk,kij->bij", coefficients, self.basis).unsqueeze(1)
        phasor = torch.polar(torch.ones_like(phase), phase)
        if self.stacked:
            return self.model(torch.cat([phasor.real, phasor.imag], dim=1))
        return self.model(phasor.real, phasor.imag)


def inverse_design(forward, start: np.ndarray, target, mask, steps: int, lr: float) -> np.ndarray:
    """Gradient design on the coefficient vector, through whatever forward map is given."""
    device = next(forward.parameters()).device
    theta = torch.nn.Parameter(torch.as_tensor(start, dtype=torch.float32, device=device)[None])
    opt = torch.optim.AdamW([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        composite_loss(forward(theta), target, mask, WEIGHTS)["_mean_total"].backward()
        opt.step()
    return theta.detach().cpu().numpy()[0]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 108)
    print("OPTION 1 (physics, closed-form) vs OPTION 2 (U-Net): forward accuracy AND invertibility")
    print("=" * 108)
    print(f"seeds={SEEDS} epochs={EPOCHS} n_max={N_MAX} grid={GRID} padding={PADDING} "
          f"unet={UNET_FEATURES} device={device}")
    print("one shared adapter: coefficients -> canonical basis -> phase -> phasor -> model")

    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    index = index.filter(families=["slm_zernike_shaping"])
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=GRID), use_cache=True)

    cfg0 = AmpTrainConfig(
        families=("slm_zernike_shaping",), n_max=N_MAX, grid=GRID, epochs=EPOCHS, lr=0.02,
        batch_size=64, max_train=512, max_val=128, beam_samples=64, use_wandb=False,
        save_checkpoint=False,
    )
    basis = build_model("physics", cfg0).basis.detach().clone().to(device)
    n_modes = basis.shape[0]

    target = inv.target_tensor(GRID, SIZE_FRAC, ASPECT).to(device)
    mask = roi_mask(
        (GRID, GRID), (GRID / 2, GRID / 2), "rectangle", SIZE_FRAC * GRID, ASPECT
    ).to(device)
    gs = inv.gs_coefficients(SIZE_FRAC, ASPECT, n_max=N_MAX).astype(np.float32)
    flat = inv.score_coefficients(
        np.zeros(n_modes), SIZE_FRAC, ASPECT, n_max=N_MAX, padding=PADDING
    )
    print(f"n_modes={n_modes}  flat reference score={flat:.4f}  (higher is better)\n")

    rows: list[dict] = []
    for seed in SEEDS:
        cfg = AmpTrainConfig(**{**cfg0.__dict__, "seed": seed})
        torch.manual_seed(seed)
        train_idx, val_idx = _select_records(dataset, cfg)
        train_t = collect_split(dataset, train_idx, device)
        val_t = collect_split(dataset, val_idx, device)
        stacked_val = _inputs(val_t)
        ref = _peak_normalise(val_t["target"].clone())

        row: dict = {"seed": seed, "flat": flat, "n_modes": n_modes}
        for name in ("physics", "unet"):
            model = build_model(name, cfg, unet_features=UNET_FEATURES).to(device)
            fwd = forward_for(name)
            model, seconds, train_mse = fit(model, train_t, cfg, device, fwd)
            model.eval()
            with torch.no_grad():
                pred = torch.cat([
                    fwd(model, stacked_val[s:s + 256])
                    for s in range(0, stacked_val.shape[0], 256)
                ])
            img = batch_image_metrics(pred, ref)
            row[f"{name}_r2"] = float(img["r2"])
            row[f"{name}_ssim"] = float(img["ssim"])
            row[f"{name}_psnr"] = float(img["psnr"])
            row[f"{name}_params"] = int(sum(p.numel() for p in model.parameters()))
            row[f"{name}_train_mse"] = float(train_mse)
            row[f"{name}_seconds"] = float(seconds)

            wrapped = _PhasorForward(model, basis, stacked=(name == "unet")).to(device).eval()
            rng = np.random.default_rng(4242 + seed)
            start = rng.normal(0.0, 0.6, n_modes).astype(np.float32)
            row[f"{name}_inv_rand"] = inv.score_coefficients(
                inverse_design(wrapped, start, target, mask, 60, 0.02),
                SIZE_FRAC, ASPECT, n_max=N_MAX, padding=PADDING,
            )
            row[f"{name}_inv_gs"] = inv.score_coefficients(
                inverse_design(wrapped, gs, target, mask, 60, 0.02),
                SIZE_FRAC, ASPECT, n_max=N_MAX, padding=PADDING,
            )
            print(
                f"  seed={seed} {name:<8} r2={row[f'{name}_r2']:+.4f} "
                f"ssim={row[f'{name}_ssim']:.4f} psnr={row[f'{name}_psnr']:5.2f} "
                f"params={row[f'{name}_params']:>10,} "
                f"inv(rand)={row[f'{name}_inv_rand']:.4f} inv(gs)={row[f'{name}_inv_gs']:.4f}"
            )
        rows.append(row)

    print("\n" + "=" * 108)
    print(f"means over {len(SEEDS)} seeds -- higher is better everywhere; flat={flat:.4f}")
    print("=" * 108)
    print(f"{'':<10}{'R2':>9}{'SSIM':>9}{'PSNR':>8}{'params':>13}"
          f"{'inv(rand)':>12}{'inv(gs)':>10}{'inv(gs)-flat':>14}")

    def mean(name: str, key: str) -> float:
        return float(np.mean([r[f"{name}_{key}"] for r in rows]))

    for name in ("physics", "unet"):
        inv_gs = mean(name, "inv_gs")
        print(f"{name:<10}{mean(name,'r2'):>+9.4f}{mean(name,'ssim'):>9.4f}"
              f"{mean(name,'psnr'):>8.2f}{mean(name,'params'):>13,.0f}"
              f"{mean(name,'inv_rand'):>12.4f}{inv_gs:>10.4f}{inv_gs - flat:>+14.4f}")

    print("\npaired, unet - physics:")
    for key, label in (("r2", "forward R2"), ("ssim", "forward SSIM"),
                       ("inv_rand", "inverse from random"), ("inv_gs", "inverse from GS")):
        d = np.array([r[f"unet_{key}"] - r[f"physics_{key}"] for r in rows])
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {label:<22} d={d.mean():+.4f} +/- {se:.4f}   "
              f"positives {int((d > 0).sum())}/{len(d)}")

    print("\nReading: judge on the inverse columns. Forward R2 says how well a model")
    print("predicts frames it was trained on; the inverse columns say whether it can")
    print("DESIGN a phase. A model that cannot be inverted cannot run the shaping loop,")
    print("however good its R2 is.")

    out = ROOT / "report" / "loss_defects" / "physics_vs_unet_inverse.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
