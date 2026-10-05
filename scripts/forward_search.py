"""Forward-model search: architectures, loss terms, augmentation. One harness, paired.

Everything the request asks for is a row in one table rather than three separate scripts, and
every arm runs through **one** training loop, so an augmented arm and the incumbent are
comparable by construction rather than by my having matched two implementations.

Arms
-----
loss       ``mse`` (incumbent), ``+ellipse`` at three weights
arch       ``physics``, ``physics+attention``, ``unet``
augment    phasor noise, phase flip, phase shift, synthetic strong-phase mixing,
           and noise + strong-phase together

Two things are deliberately *not* arms, each for a measured reason recorded in PROCESS.md:

* **Exposure / device-parameter conditioning.** All 7 families carry a CONSTANT exposure
  (``scripts/probe_device_metadata.py``): 1.1, 1.1, 0.4, 1.5, 1.2, 1.2, 0.1 ms. A constant
  input carries zero information, so there is nothing to fuse.
* **Regularisation.** ``val/train = 1.08`` with a flat learning curve, and ``l2`` /
  ``weight_decay`` / smaller ``n_max`` all lose 0/3 paired. There is no gap to close.

Ranking is on **R2 and SSIM**, never MSE/PSNR alone -- this repo has already been burned by
that (``normalization="sum"`` once reported PSNR 72 dB while its R2 was worse than a constant
predictor). Every arm is paired by seed, because the between-seed spread of R2 is ~0.075 and
would otherwise invent winners.

An inverse score is recorded per arm too, since the real bottleneck is inversion and a
forward-only win would not change that.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from compare_unet_baseline import _inputs, _peak_normalise, _select_records  # noqa: E402
from ml.hwdataset import (  # noqa: E402
    HwPhaseImageDataset,
    MaterialiserConfig,
    build_hw_index,
)
from ml.phase.unet import UNetGenerator  # noqa: E402
from ml.zernike import augment as aug  # noqa: E402
from ml.zernike import inverse_design as inv  # noqa: E402
from ml.zernike.losses import (  # noqa: E402
    LossConfig,
    composite_loss,
    ellipse_gap_term,
    roi_mask,
    spot_moment_gap_term,
)
from ml.zernike.metrics import batch_image_metrics  # noqa: E402
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpHybrid, ZernikeAmpModel  # noqa: E402
from ml.zernike.train_amp import AmpTrainConfig, collect_split  # noqa: E402

# PADDING is the family's MEASURED optimum (the sweep in PROCESS.md section 5 puts
# slm_zernike_shaping at 12; the model dataclass default is 10). Setting it to 1 to match the
# Gerchberg-Saxton design scale drops R2 from ~0.88 to ~0.53-0.74 -- the padding sweep
# result, reproduced. Forward accuracy is the thing under test here, so it gets the padding
# that forward accuracy needs; the inverse check is a SEPARATE script with a scale-matched
# target, because conflating the two is what voided the earlier ROI sweep.
GRID, N_MAX, PADDING = inv.GRID, 20, 12
SEEDS = [0, 1, 2]
EPOCHS = 60
BS = 64
LR = 0.02
SIZE_FRAC, ASPECT = inv.SIZE_FRAC, inv.ASPECT
UNET_FEATURES = [16, 32, 64, 128, 256]
BEAM_W0, APERTURE_R = 30.0, GRID / 2.0


@dataclass
class Arm:
    name: str
    kind: str = "physics"           # physics | hybrid | unet
    weights: LossConfig = field(default_factory=lambda: LossConfig(w_mse=1.0))
    noise_std: float = 0.0
    flip: bool = False
    shift_px: int = 0
    strong_frac: float = 0.0        # fraction of each batch replaced by synthetic pairs
    note: str = ""


ARMS = [
    Arm("A0 mse (incumbent)"),
    Arm("A1 +ellipse 0.05", weights=LossConfig(w_mse=1.0, w_ellipse=0.05)),
    Arm("A2 +ellipse 0.2", weights=LossConfig(w_mse=1.0, w_ellipse=0.2)),
    Arm("A3 +ellipse 1.0", weights=LossConfig(w_mse=1.0, w_ellipse=1.0)),
    Arm("A4 physics+attention", kind="hybrid"),
    Arm("A5 unet", kind="unet"),
    Arm("C1 +noise 0.05", noise_std=0.05),
    Arm("C2 +phase flip", flip=True),
    Arm("C3 +shift 2px", shift_px=2),
    Arm("C4 +strong 25%", strong_frac=0.25, note="the out-of-distribution arm"),
    Arm("C5 +noise+strong", noise_std=0.05, strong_frac=0.25),
]


def build(kind: str, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    cfg = ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING)
    if kind == "physics":
        return ZernikeAmpModel(cfg)
    if kind == "hybrid":
        return ZernikeAmpHybrid(cfg, residual_width=32)
    return UNetGenerator(in_channels=2, features=UNET_FEATURES, output_mode="phase")


def fwd(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Both arms consume the phasor pair; only the calling convention differs."""
    return model(x) if isinstance(model, UNetGenerator) else model(x[:, :1], x[:, 1:])


def train_arm(
    arm: Arm, x: torch.Tensor, y: torch.Tensor, seed: int, device, strong: tuple | None
) -> nn.Module:
    model = build(arm.kind, seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    gen = torch.Generator(device="cpu").manual_seed(seed + 777)
    model.train()
    n = x.shape[0]
    for _ in range(EPOCHS):
        order = torch.randperm(n)
        for s in range(0, n, BS):
            idx = order[s : s + BS]
            xb, yb = x[idx].to(device), y[idx].to(device)
            if arm.strong_frac > 0.0 and strong is not None:
                k = int(arm.strong_frac * xb.shape[0])
                if k > 0:
                    pick = torch.randint(0, strong[0].shape[0], (k,), generator=gen)
                    xb = torch.cat([xb[: xb.shape[0] - k], strong[0][pick].to(device)])
                    yb = torch.cat([yb[: yb.shape[0] - k], strong[1][pick].to(device)])
            if arm.noise_std > 0.0:
                xb_c, xb_s = aug.phasor_noise(
                    xb[:, :1].cpu(), xb[:, 1:].cpu(), arm.noise_std, gen
                )
                xb = torch.cat([xb_c, xb_s], dim=1).to(device)
            if arm.flip:
                xb_c, xb_s = aug.phase_flip(xb[:, :1].cpu(), xb[:, 1:].cpu(), gen)
                xb = torch.cat([xb_c, xb_s], dim=1).to(device)
            if arm.shift_px > 0:
                xb_c, xb_s = aug.phase_shift(xb[:, :1].cpu(), xb[:, 1:].cpu(), arm.shift_px, gen)
                xb = torch.cat([xb_c, xb_s], dim=1).to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((fwd(model, xb) - yb) ** 2)
            if arm.weights.w_ellipse or arm.weights.w_spot_moment:
                mask = roi_mask(
                    (GRID, GRID), (GRID / 2, GRID / 2), "rectangle",
                    SIZE_FRAC * GRID, ASPECT,
                ).to(device)
                loss = composite_loss(fwd(model, xb), yb, mask, arm.weights)["_mean_total"]
            loss.backward()
            opt.step()
        sched.step()
    return model.eval()


class CoeffForward(nn.Module):
    """coefficients -> canonical basis -> phasor -> model, for the inverse check."""

    def __init__(self, model: nn.Module, basis: torch.Tensor, stacked: bool) -> None:
        super().__init__()
        self.model = model
        self.stacked = stacked
        self.register_buffer("basis", basis)
        self.register_buffer(
            "amp", torch.as_tensor(_aperture(), dtype=torch.float32, device=basis.device)
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        phase = torch.einsum("bk,kij->bij", theta, self.basis).unsqueeze(1)
        phasor = torch.polar(torch.ones_like(phase), phase) * self.amp
        if self.stacked:
            return self.model(torch.cat([phasor.real, phasor.imag], dim=1))
        return self.model(phasor.real, phasor.imag)


def _aperture() -> np.ndarray:
    r = GRID / 2.0
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    return (((yy + 0.5 - r) ** 2 + (xx + 0.5 - r) ** 2) <= r * r).astype(np.float64)


def inverse_score(model: nn.Module, basis: torch.Tensor, stacked: bool, start: np.ndarray,
                  target, mask, device) -> float:
    wrapped = CoeffForward(model, basis, stacked).to(device).eval()
    theta = torch.nn.Parameter(torch.as_tensor(start, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([theta], lr=LR)
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        composite_loss(wrapped(theta), target, mask, LossConfig(w_mse=1.0))["_mean_total"].backward()
        opt.step()
    # score_coefficients takes the COEFFICIENT vector and does its own coefficients_to_phase
    # internally. Converting here as well makes it treat a (g, g) phase as a coefficient
    # vector -- a shape error that reads like a Zernike bug.
    return inv.score_coefficients(
        theta.detach().cpu().numpy()[0],
        SIZE_FRAC, ASPECT, n_max=N_MAX, padding=PADDING,
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 112)
    print("FORWARD-MODEL SEARCH: loss terms, architectures, augmentation -- paired over seeds")
    print("=" * 112)
    print(f"seeds={SEEDS} epochs={EPOCHS} n_max={N_MAX} grid={GRID} padding={PADDING} "
          f"device={device}")
    print("ranked on R2 and SSIM; MSE/PSNR alone are not used (this repo has been burned by that)")
    print("not arms, for measured reasons: exposure conditioning (constant in 7/7 families),")
    print("                              regularisation (val/train=1.08, flat learning curve)")

    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    index = index.filter(families=["slm_zernike_shaping"])
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=GRID), use_cache=True)

    cfg0 = AmpTrainConfig(
        families=("slm_zernike_shaping",), n_max=N_MAX, grid=GRID, epochs=EPOCHS, lr=LR,
        batch_size=BS, max_train=512, max_val=128, beam_samples=32, use_wandb=False,
        save_checkpoint=False,
    )
    basis = ZernikeAmpModel(ZernikeAmpConfig(n_max=N_MAX, grid=GRID)).basis.detach().clone().to(device)
    target = inv.target_tensor(GRID, SIZE_FRAC, ASPECT).to(device)
    mask = roi_mask(
        (GRID, GRID), (GRID / 2, GRID / 2), "rectangle", SIZE_FRAC * GRID, ASPECT
    ).to(device)
    n_modes = basis.shape[0]
    gs_coeff = inv.gs_coefficients(SIZE_FRAC, ASPECT, n_max=N_MAX).astype(np.float32)

    splits = {}
    for seed in SEEDS:
        cfg = AmpTrainConfig(**{**cfg0.__dict__, "seed": seed})
        torch.manual_seed(seed)
        tr, va = _select_records(dataset, cfg)
        splits[seed] = (
            torch.cat([collect_split(dataset, tr, device)["phase_cos"],
                       collect_split(dataset, tr, device)["phase_sin"]], dim=1),
            _peak_normalise(collect_split(dataset, tr, device)["target"].clone()),
            _inputs(collect_split(dataset, va, device)),
            _peak_normalise(collect_split(dataset, va, device)["target"].clone()),
        )

    rows: list[dict] = []
    for arm in ARMS:
        t0 = time.perf_counter()
        for seed in SEEDS:
            x, y, xv, yv = splits[seed]
            strong = None
            if arm.strong_frac > 0.0:
                strong = aug.strong_phase_pairs(
                    300, grid=GRID, beam_w0=BEAM_W0, far_field_padding=PADDING,
                    aperture_radius=APERTURE_R, n_max=N_MAX, seed=seed + 99,
                )
            model = train_arm(arm, x, y, seed, device, strong)
            with torch.no_grad():
                pred = torch.cat([fwd(model, xv[s : s + 256]) for s in range(0, xv.shape[0], 256)])
            img = batch_image_metrics(pred, yv)
            e_mask = roi_mask(
                (GRID, GRID), (GRID / 2, GRID / 2), "rectangle", SIZE_FRAC * GRID, ASPECT
            )
            k = min(64, pred.shape[0])
            eg = float(ellipse_gap_term(pred[:k], yv[:k], e_mask)["_mean_ellipse"])
            mg = float(spot_moment_gap_term(pred[:k], yv[:k], e_mask).mean())
            row = {
                "arm": arm.name, "kind": arm.kind, "seed": seed,
                "r2": float(img["r2"]), "ssim": float(img["ssim"]), "psnr": float(img["psnr"]),
                "ellipse_gap": eg, "moment_gap": mg,
                "params": int(sum(p.numel() for p in model.parameters())),
            }
            rows.append(row)
            print(
                f"  {arm.name:<20} seed={seed} r2={row['r2']:+.4f} ssim={row['ssim']:.4f} "
                f"psnr={row['psnr']:5.2f} ellipse={row['ellipse_gap']:.4f} "
                f"moment={row['moment_gap']:.4f}"
            )
        print(f"  -> {arm.name}: {time.perf_counter() - t0:.0f}s\n")

    print("=" * 112)
    print("SUMMARY -- paired against A0 (incumbent). An arm is only interesting if BOTH")
    print("its delta and its sign are consistent; a lone positive mean is inside the noise.")
    print("=" * 112)
    print(f"{'arm':<20}{'R2':>9}{'dR2':>9}{'sd':>7}{'pos':>6}{'SSIM':>8}{'dSSIM':>8}"
          f"{'ellipse':>9}{'params':>11}")
    base = {r["seed"]: r for r in rows if r["arm"] == "A0 mse (incumbent)"}
    summary = []
    for arm in ARMS:
        sel = [r for r in rows if r["arm"] == arm.name]
        d = np.array([r["r2"] - base[r["seed"]]["r2"] for r in sel])
        ds = np.array([r["ssim"] - base[r["seed"]]["ssim"] for r in sel])
        summary.append({
            "arm": arm.name, "kind": arm.kind,
            "r2": float(np.mean([r["r2"] for r in sel])),
            "d_r2": float(d.mean()), "d_r2_se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "pos": int((d > 0).sum()),
            "ssim": float(np.mean([r["ssim"] for r in sel])),
            "d_ssim": float(ds.mean()),
            "ellipse": float(np.mean([r["ellipse_gap"] for r in sel])),
            "moment": float(np.mean([r["moment_gap"] for r in sel])),
            "params": sel[0]["params"],
        })
        s = summary[-1]
        print(f"{s['arm']:<20}{s['r2']:>+9.4f}{s['d_r2']:>+9.4f}{s['d_r2_se']:>7.4f}"
              f"{s['pos']:>4}/{len(d)}{s['ssim']:>8.4f}{s['d_ssim']:>+8.4f}"
              f"{s['ellipse']:>9.4f}{s['params']:>11,}")

    out = ROOT / "report" / "loss_defects" / "forward_search.json"
    out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()