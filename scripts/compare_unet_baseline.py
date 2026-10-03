"""Head-to-head: the physics forward model vs a U-Net img2img baseline.

Both models solve the *same* task -- predict the measured far-field frame from
the commanded phase -- so the comparison is apples to apples:

* identical train/val split (same seed, same files, via the same
  ``_select_records`` the trainer uses),
* identical inputs ``(phase_cos, phase_sin)``,
* identical target (the peak-normalised ``image`` the physics model regresses),
* identical loss (MSE), optimiser (Adam), schedule (cosine) and epoch budget,
* identical metrics (``batch_image_metrics`` + ``per_sample_beam_metrics``).

Run it::

    python scripts/compare_unet_baseline.py --seeds 3

Caveat worth stating up front: these are **not the same class of model**. The
physics model has ``n_max`` trainable numbers shared by the whole corpus; the
U-Net is a per-sample function approximator with millions of parameters. The
U-Net winning on val is therefore expected and does not invalidate the physics
model -- it says the mapping is not globally low-rank. What the comparison
settles is whether the physics prior buys anything *at matched capacity*, and
where each model fails.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from ml.hwdataset import HwPhaseImageDataset, MaterialiserConfig, build_hw_index
from ml.phase.unet import UNetGenerator
from ml.zernike.metrics import batch_image_metrics, per_sample_beam_metrics, summarise_beam_metrics
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpHybrid, ZernikeAmpModel
from ml.zernike.train_amp import AmpTrainConfig, _select_records, collect_split


@dataclass
class Score:
    """One model's result on one split."""

    name: str
    params: int
    seconds: float
    mse: float
    r2: float
    psnr: float
    ssim: float
    nrmse: float
    correlation: float
    efficiency: float
    centroid_offset_px: float
    spot_diameter_ratio: float
    peak_ratio: float
    train_mse: float


def _peak_normalise(x: torch.Tensor) -> torch.Tensor:
    """Per-sample peak normalisation, matching the physics model's default."""
    return x / x.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


def _score(
    name: str,
    model: nn.Module,
    inputs: torch.Tensor,
    target: torch.Tensor,
    forward,
    seconds: float,
    train_mse: float,
    beam_samples: int,
) -> Score:
    """Evaluate a model with the shared metric set."""
    model.eval()
    with torch.no_grad():
        predictions = []
        for start in range(0, inputs.shape[0], 256):
            stop = min(start + 256, inputs.shape[0])
            predictions.append(forward(model, inputs[start:stop]))
        prediction = torch.cat(predictions)
    image = batch_image_metrics(prediction, target)
    rows = [
        per_sample_beam_metrics(
            prediction[i, 0].float().cpu().numpy(), target[i, 0].float().cpu().numpy()
        )
        for i in range(min(beam_samples, target.shape[0]))
    ]
    beam = summarise_beam_metrics(rows)
    model.train()
    return Score(
        name=name,
        params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        seconds=seconds,
        mse=image["mse"],
        r2=image["r2"],
        psnr=image["psnr"],
        ssim=image["ssim"],
        nrmse=image["nrmse"],
        correlation=beam["correlation"],
        efficiency=beam["efficiency"],
        centroid_offset_px=beam["centroid_offset_px"],
        spot_diameter_ratio=beam["spot_diameter_ratio"],
        peak_ratio=beam["peak_ratio"],
        train_mse=train_mse,
    )


def _inputs(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    """Stack the phasor pair into the 2-channel U-Net input."""
    return torch.cat([tensors["phase_cos"], tensors["phase_sin"]], dim=1)


def train_physics(train_t, val_t, cfg, device) -> Score:
    """Train the physics model exactly as ``train_amp`` would."""
    model = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=cfg.n_max, grid=cfg.grid)
    ).to(device)
    target = _peak_normalise(train_t["target"].clone())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    n = train_t["phase_cos"].shape[0]
    started = time.perf_counter()
    running = 0.0
    for _ in range(cfg.epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(
                (
                    model(train_t["phase_cos"][idx], train_t["phase_sin"][idx])
                    - target[idx]
                )
                ** 2
            )
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * idx.numel()
        scheduler.step()
    seconds = time.perf_counter() - started

    def forward(m, x):
        return m(x[:, :1], x[:, 1:])

    return _score(
        f"physics(n_max={cfg.n_max})", model, _inputs(val_t),
        _peak_normalise(val_t["target"].clone()), forward, seconds,
        running / (n * cfg.epochs), cfg.beam_samples,
    )


def train_unet(train_t, val_t, cfg, device, features: list[int]) -> Score:
    """Train the U-Net img2img baseline under the identical budget."""
    model = UNetGenerator(
        in_channels=2, features=features, output_mode="phase"
    ).to(device)
    inputs = _inputs(train_t)
    target = _peak_normalise(train_t["target"].clone())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    n = inputs.shape[0]
    started = time.perf_counter()
    running = 0.0
    for _ in range(cfg.epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(inputs[idx]) - target[idx]) ** 2)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * idx.numel()
        scheduler.step()
    seconds = time.perf_counter() - started

    def forward(m, x):
        return m(x)

    return _score(
        f"unet{features[0]}", model, _inputs(val_t),
        _peak_normalise(val_t["target"].clone()), forward, seconds,
        running / (n * cfg.epochs), cfg.beam_samples,
    )


def train_hybrid(train_t, val_t, cfg, device, width: int) -> Score:
    """Train the physics + learned-residual hybrid under the identical budget."""
    model = ZernikeAmpHybrid(
        ZernikeAmpConfig(n_max=cfg.n_max, grid=cfg.grid), residual_width=width
    ).to(device)
    inputs = _inputs(train_t)
    target = _peak_normalise(train_t["target"].clone())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    n = inputs.shape[0]
    started = time.perf_counter()
    running = 0.0
    for _ in range(cfg.epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(inputs[idx, :1], inputs[idx, 1:]) - target[idx]) ** 2)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * idx.numel()
        scheduler.step()
    seconds = time.perf_counter() - started

    def forward(m, x):
        return m(x[:, :1], x[:, 1:])

    return _score(
        f"hybrid(w={width})", model, _inputs(val_t),
        _peak_normalise(val_t["target"].clone()), forward, seconds,
        running / (n * cfg.epochs), cfg.beam_samples,
    )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--n-max", type=int, default=15)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train", type=int, default=1010)
    parser.add_argument("--max-val", type=int, default=128)
    parser.add_argument("--beam-samples", type=int, default=96)
    parser.add_argument(
        "--unet-features", type=int, nargs="+", default=[16, 32, 64, 128, 256]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--models", nargs="*", default=["physics", "hybrid", "unet"],
        help="subset of physics / hybrid / unet to run",
    )
    parser.add_argument(
        "--residual-width", type=int, nargs="+", default=[16, 32, 64]
    )
    parser.add_argument("--out", default="logs/unet_comparison.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    index = index.filter(families=["slm_zernike_shaping"])
    dataset = HwPhaseImageDataset(
        index, config=MaterialiserConfig(grid=args.grid), use_cache=True
    )

    cfg = AmpTrainConfig(
        families=("slm_zernike_shaping",), n_max=args.n_max, grid=args.grid,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        max_train=args.max_train, max_val=args.max_val,
        beam_samples=args.beam_samples, use_wandb=False, save_checkpoint=False,
    )

    rows: list[dict] = []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        cfg_seed = AmpTrainConfig(**{**cfg.__dict__, "seed": seed})
        train_idx, val_idx = _select_records(dataset, cfg_seed)
        train_t = collect_split(dataset, train_idx, device)
        val_t = collect_split(dataset, val_idx, device)
        runners = []
        if "physics" in args.models:
            runners.append(
                lambda: train_physics(train_t, val_t, cfg_seed, device)
            )
        if "hybrid" in args.models:
            for width in args.residual_width:
                runners.append(
                    lambda w=width: train_hybrid(train_t, val_t, cfg_seed, device, w)
                )
        if "unet" in args.models:
            runners.append(
                lambda: train_unet(
                    train_t, val_t, cfg_seed, device, args.unet_features
                )
            )
        for run in runners:
            score = run()
            logger.info(
                "seed {} | {:<22} params={:>9,} mse={:.5f} r2={:+.4f} "
                "psnr={:.2f}dB ssim={:.4f} d_off={:.2f}px {:.1f}s",
                seed, score.name, score.params, score.mse, score.r2,
                score.psnr, score.ssim, score.centroid_offset_px, score.seconds,
            )
            rows.append({"seed": seed, **score.__dict__})

    print(f"\n{'model':<22}{'params':>10}{'mse':>10}{'r2':>9}{'psnr':>9}{'ssim':>8}"
          f"{'corr':>7}{'d_off':>8}{'d_rat':>8}{'sec':>7}")
    print("-" * 96)
    names = sorted({r["name"] for r in rows})
    for name in names:
        group = [r for r in rows if r["name"] == name]
        pick = lambda k: float(np.mean([r[k] for r in group]))  # noqa: E731
        print(f"{name:<22}{int(pick('params')):>10,}{pick('mse'):>10.5f}"
              f"{pick('r2'):>+9.4f}{pick('psnr'):>8.2f}dB{pick('ssim'):>8.4f}"
              f"{pick('correlation'):>7.3f}{pick('centroid_offset_px'):>8.2f}"
              f"{pick('spot_diameter_ratio'):>8.3f}{pick('seconds'):>7.1f}")
    print(f"\n(mean of {args.seeds} seeds; seed-to-seed spread on r2 is ~0.13 here, "
          "so differences below that are not meaningful)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    logger.remove()
    raise SystemExit(main())