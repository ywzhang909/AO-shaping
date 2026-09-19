"""FourierGSNet beam-shaping demo & convergence simulation.

End-to-end run:

- Generates a synthetic (source, target, GS-ground-truth-phase) dataset
  on a 64x64 grid with mixed target shapes.
- Builds a paper-config FourierGSNet (10 unrolled layers, 32 base channels).
- Evaluates the *untrained* network (baseline) and the *trained* network
  on a held-out eval set, proving that training converges: far-field
  correlation / encircled energy up, phase MAE down.
- Saves the loss history plot, a before/after field comparison figure,
  and the best checkpoint under ``data/gsnet/``.

Usage::

    python -m ml.gsnet.run_demo --epochs 20 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from loguru import logger

from ml.gsnet.dataset import GSShapingDataset
from ml.gsnet.evaluate import EvalSummary, compute_sample_metrics, evaluate_model
from ml.gsnet.model import FourierGSNet, count_parameters
from ml.gsnet.train import train_gsnet

OUT_ROOT = Path("data/gsnet")


def pick_device(force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def reconstruct_far_intensity(
    model: FourierGSNet,
    source: torch.Tensor,
    target: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Single-pass far-field intensity the network would produce on an SLM."""
    model.to(device).eval()
    with torch.no_grad():
        phase = model(source.to(device), target.to(device))
        source_amp = torch.sqrt(source.to(device).clamp_min(0.0) + 1e-12)
        field = source_amp * torch.exp(1j * phase)
        far = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
    return torch.abs(far) ** 2, phase


def render_figure(
    out_path: Path,
    history: list[dict[str, float]],
    ds: GSShapingDataset,
    model_before: FourierGSNet,
    model_after: FourierGSNet,
    device: str,
    samples: tuple[int, int, int] = (0, 1, 2),
) -> None:
    """Save a 3-sample before/after comparison figure plus the loss curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(samples)
    fig, axes = plt.subplots(n, 5, figsize=(4 * 5, 4 * n))

    for row, idx in enumerate(samples):
        src, tgt, _ = ds[idx]
        far_before, phase_before = reconstruct_far_intensity(
            model_before, src.unsqueeze(0), tgt.unsqueeze(0), device
        )
        far_after, phase_after = reconstruct_far_intensity(
            model_after, src.unsqueeze(0), tgt.unsqueeze(0), device
        )

        def show(ax, img, title: str, cmap: str = "inferno") -> None:
            im = ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)

        show(axes[row, 0], src[0].cpu().numpy(), "source")
        show(axes[row, 1], tgt[0].cpu().numpy(), "target")
        show(axes[row, 2], phase_after[0, 0].cpu().numpy(), "predicted phase")
        show(axes[row, 3], far_before[0, 0].cpu().numpy(), "far-field (before)")
        show(axes[row, 4], far_after[0, 0].cpu().numpy(), "far-field (after)")

    fig.suptitle("FourierGSNet beam shaping — before vs after training", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=110)

    # Loss curve figure.
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(history) + 1)
    ax2.plot(epochs, [h["loss"] for h in history], "-o", label="total loss")
    ax2.plot(epochs, [h["phase_loss"] for h in history], "--", label="phase loss")
    ax2.plot(epochs, [h["shaping_loss"] for h in history], ":", label="shaping loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_path.with_name("train_history.png"), bbox_inches="tight", dpi=110)

    logger.info("Figures saved: {} (+ train_history.png)", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=10)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-eval", type=int, default=32)
    parser.add_argument("--gs-iterations", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w-phase", type=float, default=1.0)
    parser.add_argument("--w-shaping", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--out", type=str, default=str(OUT_ROOT))
    args = parser.parse_args()

    device = pick_device(args.force_cpu)
    logger.info("Device: {}", device)
    logger.info(
        "Config: grid={} layers={} channels={} train={} eval={} epochs={}",
        args.grid, args.num_layers, args.base_channels,
        args.n_train, args.n_eval, args.epochs,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic RNG.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Generating datasets (FFT-GS ground truth)...")
    train_ds = GSShapingDataset(
        n_samples=args.n_train, grid=args.grid, seed=args.seed,
        gs_iterations=args.gs_iterations,
    )
    eval_ds = GSShapingDataset(
        n_samples=args.n_eval, grid=args.grid, seed=1000 + args.seed,
        gs_iterations=args.gs_iterations,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch_size)

    model = FourierGSNet(
        num_layers=args.num_layers, base_channels=args.base_channels,
    )
    logger.info("Model parameters: {}", count_parameters(model))

    # Snapshot the untrained weights for the before/after figure.
    model_before = FourierGSNet(
        num_layers=args.num_layers, base_channels=args.base_channels,
    )
    model_before.load_state_dict(model.state_dict())

    # ---- Baseline (untrained) evaluation — proves training is necessary. ----
    logger.info("Evaluating UNTRAINED baseline...")
    baseline: EvalSummary = evaluate_model(model, eval_dl, device=device)
    logger.info("BASELINE  {}", baseline)

    # ---- Train. ----
    logger.info("Training FourierGSNet on {} samples, {} epochs...",
                args.n_train, args.epochs)
    result = train_gsnet(
        model, train_dl, epochs=args.epochs, lr=args.lr,
        w_phase=args.w_phase, w_shaping=args.w_shaping, device=device,
        checkpoint_dir=str(out_dir), seed=args.seed,
    )
    logger.info("Best epoch {} / best loss {:.6f} / time {:.1f}s",
                result.best_epoch, result.best_loss, result.seconds)

    # ---- Trained evaluation. ----
    logger.info("Evaluating TRAINED model...")
    trained: EvalSummary = evaluate_model(model, eval_dl, device=device)
    logger.info("TRAINED   {}", trained)

    # ---- Convergence verdict. ----
    improvements = {
        "far_correlation": trained.means["far_correlation"]
        - baseline.means["far_correlation"],
        "encircled_energy": trained.means["encircled_energy"]
        - baseline.means["encircled_energy"],
        "phase_mae_gain": baseline.means["phase_mae"]
        - trained.means["phase_mae"],
    }
    loss_dropped = result.history[-1]["loss"] < result.history[0]["loss"]
    corr_improved = improvements["far_correlation"] > 1e-3
    logger.info(
        "Convergence: loss↓={} corr↑={:+.4f} EE↑={:+.4f} phase_mae↓(gain)={:+.4f}",
        loss_dropped, improvements["far_correlation"],
        improvements["encircled_energy"], improvements["phase_mae_gain"],
    )
    converged = loss_dropped and corr_improved
    verdict = "CONVERGED" if converged else "NOT CONVERGED"
    logger.info("VERDICT: {}", verdict)

    # ---- Artifacts. ----
    render_figure(
        out_dir / "predictions.png", result.history, eval_ds,
        model_before, model, device,
    )

    # Save eval metrics as JSON.
    import json

    summary = {
        "verdict": verdict,
        "baseline": baseline.means,
        "trained": trained.means,
        "improvements": improvements,
        "best_epoch": result.best_epoch,
        "best_loss": result.best_loss,
        "seconds": result.seconds,
        "device": device,
        "args": vars(args),
    }
    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(summary, indent=2))
    logger.info("Metrics saved: {}", json_path)

    # Save one predicted phase for inspection (radians, SLM-ready).
    src, tgt, _ = eval_ds[0]
    with torch.no_grad():
        phase = model(src.unsqueeze(0).to(device), tgt.unsqueeze(0).to(device))
    np.save(out_dir / "sample_phase_rad.npy", phase[0, 0].cpu().numpy())
    logger.info("Sample phase saved: {}", out_dir / "sample_phase_rad.npy")


if __name__ == "__main__":
    main()