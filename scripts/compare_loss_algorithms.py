"""Compare training objectives and search algorithms on a common axis (offline).

Two tracks, runnable concurrently because they share no state:

**Track A -- ML (autograd).** ``train_amp`` x ``AmpTrainConfig.loss`` x torch
optimizer. Reports forward-model quality: val R2, PSNR, SSIM, and the *physical*
ROI metrics computed with the repo's own numpy implementations. This is where the
new losses in :mod:`ml.zernike.losses` actually apply -- they are torch terms
differentiated through ``ZernikeAmpModel``.

**Track B -- AO (measurement).** ``optimize_slm_zernike_pib`` x objective x search
algorithm x SPGD optimizer_type, on the 2f-Fourier sim bench. Reports the
*achieved spot* quality straight off the measured frames.

Why two tracks and not one
--------------------------
The new losses are autograd terms over a *predicted* far field; the AO
optimizer is a measurement-driven SPGD/heuristic search that never sees a torch
tensor. So they cannot be substituted into each other. What this script does
instead is measure both on the same outcome axis (the repo's canonical
``rms_pib_terms`` / ``metric_panel`` shape metrics, computed with the same numpy
code), so "which loss" and "which algorithm" are answerable, and the coupling
between them is explicit rather than assumed.

Read the results with the repo's noise rules in mind
---------------------------------------------------
* Never rank these runs on MSE/PSNR/SSIM alone -- ``normalization="sum"`` once
  scored PSNR 72 dB / SSIM 0.9996 while R2 was worse than a constant predictor.
  Judge on R2 and the physical terms.
* Seeds pin the *search*, not the measurement; the sim camera carries noise, so
  a single-seed delta is not a ranking. ``--seeds`` defaults to >1 and the
  summary reports mean +/- spread, and every configuration sees the SAME seed
  list so the comparison is paired.
* The sim bench starts near-optimal, so "improvement over the flat baseline" is
  not guaranteed and is not asserted here.

Usage (report generation belongs in ``scripts/``, per AGENTS.md)::

    python scripts/compare_loss_algorithms.py --quick
    python scripts/compare_loss_algorithms.py --out report/loss_algorithms
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")  # BEFORE pyplot
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------
# Track A: ML objective x torch optimizer
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MLAxis:
    loss: str
    optimizer: str


def _run_ml(axes: MLAxis, *, epochs: int, lr: float, grid: int, n_max: int,
            max_train: int, max_val: int, device: str, out_dir: str) -> dict:
    """One ML cell. Must be module-level: Windows spawn re-imports workers."""
    from ml.zernike.train_amp import AmpTrainConfig, train

    cfg = AmpTrainConfig(
        grid=grid,
        n_max=n_max,
        epochs=epochs,
        lr=lr,
        batch_size=min(32, max_train),
        max_train=max_train,
        max_val=max_val,
        device=device,
        optimizer=axes.optimizer,
        loss=axes.loss,
        out_dir=out_dir,
        use_wandb=False,
        save_checkpoint=False,
        seed=0,
    )
    result = train(cfg)
    # Only the forward-model fidelity is reported here. The *physical* spot
    # quality of a prediction is deliberately NOT inferred from coefficients:
    # ZernikeAmpModel predicts a far field from a (phase_cos, phase_sin) grid,
    # not from a coefficient vector, so there is no direct coefficient -> image
    # entry point. Measured spot quality is Track B's job, off real frames.
    return {
        "track": "A_ml",
        "loss": axes.loss,
        "optimizer": axes.optimizer,
        "best_epoch": result.best_epoch,
        "val_r2": result.best_val_r2,
        "val_psnr": result.best_val_psnr,
        "seconds": result.seconds,
        "max_abs_coeff": float(np.max(np.abs(result.coefficients))),
    }


# --------------------------------------------------------------------------
# Track B: AO objective x search algorithm
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AOAxis:
    objective: str
    algorithm: str
    optimizer_type: str
    seed: int


def _run_ao(axes: AOAxis, *, epochs: int, pop_size: int, cam_size: int,
            lr: float, delta: float) -> dict:
    """One AO cell against the sim bench. Paired across cells by ``seed``."""
    import importlib

    from ao_shaping.drivers.sim.sim_bench_patch import SimSLMPib, install_sim_slm
    from ao_shaping.drivers.sim.slm_pib_sim import register_sim_camera, reset_system
    from ao_shaping.runners.runner_common import (
        CameraParamsPib,
        ObjectiveTarget,
        SlmParamsPib,
    )

    engine = importlib.import_module("ao_shaping.optimizer.wfless.slm_zernike_pib")
    install_sim_slm(engine)
    engine.Santec = SimSLMPib

    register_sim_camera()
    reset_system(seed=axes.seed)

    kwargs = {"pop_size": pop_size} if axes.algorithm != "spgd" else {}
    cfg = engine.SlmZernikePibConfig(
        center="shape",
        epochs=epochs,
        algorithm=axes.algorithm,
        optimizer_type=axes.optimizer_type,
        lr=lr,
        delta=delta,
        random_seed=axes.seed,
        camera=CameraParamsPib(
            target=ObjectiveTarget(name=axes.objective, target_shape=None),
            cam_type="sim",
            cam_size=cam_size,
            exposure_time_ms=80.0,
        ),
        slm=SlmParamsPib(n_max=4),
        **kwargs,
    )
    recorder = engine.optimize_slm_zernike_pib(cfg)
    last = recorder.history[-1]
    row = {
        "track": "B_ao",
        "objective": axes.objective,
        "algorithm": axes.algorithm,
        "optimizer_type": axes.optimizer_type if axes.algorithm == "spgd" else "-",
        "seed": axes.seed,
        "epochs_recorded": len(recorder.history),
        "final_J": float(last["J"]),
    }
    for key in ("m_pib", "m_shape", "m_energy", "m_rmse", "m_ee", "m_rms_pib"):
        if key in last:
            row[key] = float(last[key])
    row["coeff_norm"] = float(np.linalg.norm(np.asarray(last["_c"], dtype=np.float64)))
    return row


# --------------------------------------------------------------------------
def _expand(spec: str, items: list[str]) -> list[str]:
    if spec == "all":
        return items
    return [spec] if spec else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="report/loss_algorithms")
    parser.add_argument("--quick", action="store_true", help="tiny smoke matrix")
    parser.add_argument("--workers", type=int, default=min(4, (os.cpu_count() or 2)))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--n-max", type=int, default=4)
    parser.add_argument("--max-train", type=int, default=64)
    parser.add_argument("--max-val", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ml-loss", default="all", help="all|mse|physical")
    parser.add_argument("--ml-optimizers", default="adam,adamw,sgd")
    parser.add_argument("--ao-objectives", default="pib,shape,rms_pib")
    parser.add_argument("--ao-algorithms", default="all")
    parser.add_argument("--ao-spgd-optimizers", default="adam,adamod,muno")
    parser.add_argument("--ao-pop", type=int, default=6)
    parser.add_argument("--cam-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    from ao_shaping.optimizer.wfless.slm_zernike_pib import (
        ALGORITHM_CHOICES,
        OPTIMIZER_MAP,
    )

    ml_losses = (
        ["mse", "physical"]
        if args.ml_loss == "all"
        else _expand(args.ml_loss, ["mse", "physical"])
    )
    ml_axes = [
        MLAxis(loss=loss, optimizer=optimizer)
        for loss, optimizer in itertools.product(
            ml_losses, args.ml_optimizers.split(",")
        )
    ]

    ao_objectives = args.ao_objectives.split(",")
    ao_algorithms = (
        sorted(ALGORITHM_CHOICES)
        if args.ao_algorithms == "all"
        else args.ao_algorithms.split(",")
    )
    seeds = list(range(args.seeds))
    ao_axes = [
        AOAxis(
            objective=obj,
            algorithm=alg,
            optimizer_type=(opt if alg == "spgd" else "-"),
            seed=seed,
        )
        for obj, alg, opt, seed in itertools.product(
            ao_objectives,
            ao_algorithms,
            args.ao_spgd_optimizers.split(","),
            seeds,
        )
    ]

    epochs = 2 if args.quick else args.epochs
    print(
        f"matrix: track A = {len(ml_axes)} cells, track B = {len(ao_axes)} cells, "
        f"epochs={epochs}, workers={args.workers}"
    )

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        if ml_axes:
            print("\n--- Track A: ML objective x torch optimizer ---")
            for axes in ml_axes:
                fut = pool.submit(
                    _run_ml,
                    axes,
                    epochs=epochs,
                    lr=args.lr,
                    grid=args.grid,
                    n_max=args.n_max,
                    max_train=args.max_train,
                    max_val=args.max_val,
                    device=args.device,
                    out_dir=str(out_dir / "ml"),
                )
                futures[fut] = ("A", axes)
        for axes in ao_axes:
            fut = pool.submit(
                _run_ao,
                axes,
                epochs=epochs,
                pop_size=args.ao_pop,
                cam_size=args.cam_size,
                lr=args.lr,
                delta=args.delta,
            )
            futures[fut] = ("B", axes)

        done = 0
        for fut in as_completed(futures):
            track, axes = futures[fut]
            done += 1
            tag = (
                f"{axes.loss}/{axes.optimizer}"
                if track == "A"
                else f"{axes.objective}/{axes.algorithm}"
                f"{'/' + axes.optimizer_type if axes.algorithm == 'spgd' else ''}"
                f"@{axes.seed}"
            )
            try:
                row = fut.result()
                rows.append(row)
                extra = ""
                if track == "A":
                    extra = f" R2={row['val_r2']:+.4f} PSNR={row['val_psnr']:.2f}"
                else:
                    extra = f" J={row['final_J']:.4g}"
                    if "m_pib" in row:
                        extra += f" m_pib={row['m_pib']:.4f}"
                print(f"  [{done}/{len(futures)}] {tag}{extra}")
            except Exception as exc:  # noqa: BLE001 - report, do not abort the matrix
                print(f"  [{done}/{len(futures)}] {tag} FAILED: "
                      f"{type(exc).__name__}: {exc}")
                rows.append({"track": "A_ml" if track == "A" else "B_ao",
                             "loss" if track == "A" else "objective":
                             getattr(axes, "loss", getattr(axes, "objective", "?")),
                             "error": f"{type(exc).__name__}: {exc}"})

    _write_csv(out_dir / "results.csv", rows)
    _write_markdown(out_dir / "report.md", rows, args, epochs)
    _plot(out_dir / "summary.png", rows)
    print(f"\nwrote {out_dir / 'results.csv'}")
    print(f"wrote {out_dir / 'report.md'}")
    print(f"wrote {out_dir / 'summary.png'}")
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float) -> str:
    return f"{value:+.4f}" if isinstance(value, (int, float)) else str(value)


def _write_markdown(path: Path, rows: list[dict], args, epochs: int) -> None:
    a_rows = [r for r in rows if r.get("track") == "A_ml" and "val_r2" in r]
    b_rows = [r for r in rows if r.get("track") == "B_ao" and "final_J" in r]
    failed = [r for r in rows if "error" in r]

    lines = [
        "# Loss x algorithm comparison (offline sim bench)",
        "",
        f"- epochs per cell: {epochs}",
        f"- paired seeds: {args.seeds} (same list for every cell)",
        f"- grid {args.grid}, n_max {args.n_max}, cam_size {args.cam_size}",
        "",
        "## How to read this",
        "",
        "Rank on **R2** and the physical ROI terms, never on MSE/PSNR/SSIM:",
        "`normalization=\"sum\"` once reported PSNR 72 dB / SSIM 0.9996 while its",
        "R2 was *worse* than a constant predictor. Single-seed deltas are inside",
        "the noise, which is why seeds are paired and spread is reported.",
        "",
        "Track A optimises a torch loss through the forward model. Track B is a",
        "measurement-driven search and never sees that loss -- they are reported",
        "separately on purpose.",
        "",
    ]

    lines += ["## Track A -- ML objective x torch optimizer", ""]
    if a_rows:
        lines += ["| loss | optimizer | val R2 | val PSNR (dB) | best epoch | max abs coeff |",
                  "|---|---|---|---|---|---|"]
        for row in sorted(a_rows, key=lambda r: (-(r.get("val_r2") or -9))):
            lines.append(
                f"| {row['loss']} | {row['optimizer']} | {_fmt(row.get('val_r2'))} | "
                f"{_fmt(row.get('val_psnr'))} | {row.get('best_epoch')} | "
                f"{_fmt(row.get('max_abs_coeff'))} |"
            )
    else:
        lines.append("_no successful Track A cell_")

    lines += ["", "## Track B -- AO objective x search algorithm (sim bench)", ""]
    if b_rows:
        lines += ["| objective | algorithm | spgd optimizer | seed | final J | m_pib | m_shape | m_ee | m_rmse | coeff norm |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for row in b_rows:
            lines.append(
                f"| {row['objective']} | {row['algorithm']} | {row['optimizer_type']} | "
                f"{row['seed']} | {_fmt(row['final_J'])} | {_fmt(row.get('m_pib'))} | "
                f"{_fmt(row.get('m_shape'))} | {_fmt(row.get('m_ee'))} | "
                f"{_fmt(row.get('m_rmse'))} | {_fmt(row.get('coeff_norm'))} |"
            )
        # Aggregate over seeds so a single noisy cell cannot look like a winner.
        lines += ["", "### Track B aggregated over seeds (mean +/- spread)", ""]
        lines += ["| objective | algorithm | spgd optimizer | n | m_pib mean | m_shape mean | m_ee mean |",
                  "|---|---|---|---|---|---|---|"]
        groups: dict[tuple, list[dict]] = {}
        for row in b_rows:
            key = (row["objective"], row["algorithm"], row["optimizer_type"])
            groups.setdefault(key, []).append(row)
        for key, items in sorted(groups.items()):
            def stat(field: str) -> str:
                vals = [float(i[field]) for i in items if field in i]
                if not vals:
                    return "-"
                spread = max(vals) - min(vals) if len(vals) > 1 else 0.0
                return f"{np.mean(vals):.4f} +/-{spread:.4f}"
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2]} | {len(items)} | "
                f"{stat('m_pib')} | {stat('m_shape')} | {stat('m_ee')} |"
            )
    else:
        lines.append("_no successful Track B cell_")

    if failed:
        lines += ["", "## Failed cells", "", "| cell | error |", "|---|---|"]
        for row in failed:
            name = row.get("loss") or row.get("objective")
            lines.append(f"| {name} | {row['error']} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(path: Path, rows: list[dict]) -> None:
    b_rows = [r for r in rows if r.get("track") == "B_ao" and "m_pib" in r]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax in axes:
        ax.grid(True, alpha=0.3)
    a_rows = [r for r in rows if r.get("track") == "A_ml" and "val_r2" in r]
    if a_rows:
        labels = [f"{r['loss']}\n{r['optimizer']}" for r in a_rows]
        axes[0].bar(labels, [r["val_r2"] for r in a_rows])
        axes[0].set_title("Track A: forward-model val R2")
        axes[0].set_ylabel("val R2")
    else:
        axes[0].set_title("Track A: no data")
    if b_rows:
        groups: dict[tuple, list[float]] = {}
        for row in b_rows:
            groups.setdefault(
                (row["objective"], row["algorithm"], row["optimizer_type"]), []
            ).append(float(row["m_pib"]))
        labels = [f"{k[0]}\n{k[1]}" for k in sorted(groups)]
        values = [np.mean(groups[k]) for k in sorted(groups)]
        axes[1].bar(labels, values)
        axes[1].set_title("Track B: achieved m_pib (sim, mean over seeds)")
        axes[1].set_ylabel("m_pib")
    else:
        axes[1].set_title("Track B: no data")
    for label in axes[1].get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())