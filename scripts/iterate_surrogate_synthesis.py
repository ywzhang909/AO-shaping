"""Iterate on the surrogate, and check whether a better surrogate gives a better phase.

The inversion run produced a result that could not be trusted: a freeform phase
reached CV 0.155 in surrogate space, but an independently trained model scored the
same phase at 0.62, and the phase was demonstrably off both models' training
manifold (27 % of its energy inside the ``n_max=15`` Zernike span versus 63 % for
the corpus). Band-limiting the phase to that span fixed the trust problem and cost
2.5x in surrogate CV (0.155 -> 0.386).

That leaves the question this script answers: **does making the surrogate better
actually make the synthesised phase better?** It is worth asking rather than
assuming, because a surrogate can improve its *average* accuracy while getting
worse exactly where the optimiser pushes it.

Every quantity here is checkable, which is the point:

* each surrogate is scored on **held-out pickles** (grouped CV, never the frame it
  was trained on), so the R^2 column is a real generalisation number;
* each synthesised phase is **Zernike band-limited to n_max=15**, i.e. inside the
  manifold every surrogate in the zoo was validated on, so its CV is comparable
  across rounds instead of being inflated by extrapolation;
* every phase is scored under **every** surrogate in the zoo, which gives a
  cross-model matrix. If the ranking of phases is stable down that matrix, the
  improvements are properties of the optics; if it inverts, they are properties of
  a particular model.

Surrogate zoo spans underfit to over-trained on the axis that matters
(``n_max`` x epochs), so "better" is a real sweep rather than a monotone knob.

Usage::

    python scripts/iterate_surrogate_synthesis.py
    python scripts/iterate_surrogate_synthesis.py --folds 3

Writes ``logs/surrogate_iteration.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from loguru import logger  # noqa: E402

from ml.hwdataset import HwPhaseImageDataset, MaterialiserConfig, build_hw_index  # noqa: E402
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402
from ml.zernike.train_amp import AmpTrainConfig, collect_split  # noqa: E402

from optimize_uniform_spot_phase import (  # noqa: E402
    box_slice,
    optimise_phase,
    predict_detached,
    score_spot,
)

# (label, n_max, epochs, lr) -- spans underfit to over-trained.
ZOO: tuple[tuple[str, int, int, float], ...] = (
    ("n4_e25", 4, 25, 0.02),
    ("n8_e25", 8, 25, 0.02),
    ("n15_e25", 15, 25, 0.02),
    ("n15_e50", 15, 50, 0.01),
    ("n15_e100", 15, 100, 0.01),
)

#: The manifold the corpus was trained on. Band-limiting the synthesised phase to
#: this is what makes cross-round CV numbers comparable at all.
SYNTH_N_MAX = 15


def train_surrogate(train_t, n_max: int, epochs: int, lr: float, device) -> ZernikeAmpModel:
    """Fit one surrogate on one fold's training split (weights stay free here)."""
    config = ZernikeAmpConfig(n_max=n_max, grid=64, observable="intensity",
                              normalization="peak", far_field_padding=10, center_crop=True)
    model = ZernikeAmpModel(config).to(device)
    target = _peak_norm(train_t["target"])
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(epochs, 1))
    n = target.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, 64):
            idx = order[start : start + 64]
            optimiser.zero_grad(set_to_none=True)
            loss = torch.mean(
                (model(train_t["phase_cos"][idx], train_t["phase_sin"][idx]) - target[idx]) ** 2
            )
            loss.backward()
            optimiser.step()
        scheduler.step()
    model.eval()
    return model


def _peak_norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def held_out_r2(model, val_t) -> float:
    """R^2 on held-out pickles, from the recorded phase pair.

    Not `predict_detached`, which synthesises a phase and expects a raw-radian
    tensor; here the phasor is already measured.
    """
    prediction = model(val_t["phase_cos"], val_t["phase_sin"])
    target = _peak_norm(val_t["target"].clone())
    ss_res = float(((prediction - target) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def synthesise(model, device, side: int, epochs: int, lr: float) -> np.ndarray:
    """Band-limited phase synthesis -- the trustworthy configuration."""
    phase, _ = optimise_phase(
        model, 64, side, epochs, lr, device, init=None,
        loss_kind="quality", mode="zernike", n_max=SYNTH_N_MAX,
    )
    return phase.cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=3, help="grouped folds per surrogate")
    parser.add_argument("--target-side", type=int, default=50)
    parser.add_argument("--synth-epochs", type=int, default=3000)
    parser.add_argument("--synth-lr", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--index-cache", default="data/hw_index_cache.json")
    parser.add_argument("--out", default="logs/surrogate_iteration.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    index = build_hw_index(index_cache=args.index_cache).filter(families=["slm_zernike_shaping"])
    records = list(index.records)
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=64), use_cache=True)

    from compare_models_cv import build_folds

    folds = build_folds(records, "file")[: args.folds]
    logger.info("zoo of {} surrogates x {} folds", len(ZOO), len(folds))

    rows: list[dict] = []
    # Every phase is re-scored under every surrogate, so the matrix is filled in
    # after all surrogates exist. The judges are the *same* fold-0 model instances
    # that produced the phases -- retraining them here would make the diagonal
    # meaningless, since a fresh draw of the same config is a different model.
    phases: dict[str, np.ndarray] = {}
    producers: dict[str, ZernikeAmpModel] = {}
    started = time.perf_counter()

    for label, n_max, epochs, lr in ZOO:
        fold_r2, fold_cv, fold_quality, fold_dof = [], [], [], []
        for fold in folds:
            train_t = collect_split(dataset, fold.train, device)
            val_t = collect_split(dataset, fold.val, device)
            model = train_surrogate(train_t, n_max, epochs, lr, device)
            fold_r2.append(held_out_r2(model, val_t))

            phase = synthesise(model, device, args.target_side, args.synth_epochs, args.synth_lr)
            if label not in phases:
                phases[label] = phase  # every fold explores the same basin
                producers[label] = model  # keep this instance as a judge
            tensor = torch.from_numpy(phase).to(device)
            spot = score_spot(predict_detached(model, tensor)[0, 0].cpu().numpy(), args.target_side)
            fold_cv.append(spot["uniformity_cv"])
            fold_quality.append(spot["quality"])
            fold_dof.append(n_max * (n_max + 3) // 2)
            logger.info(
                "{} fold {}: held-out R2={:+.4f} -> synthesised CV={:.4f} quality={:.4f}",
                label, fold.index, fold_r2[-1], fold_cv[-1], fold_quality[-1],
            )
            if label not in producers:
                del model
                del train_t, val_t
        rows.append({
            "label": label, "n_max": n_max, "epochs": epochs, "lr": lr,
            "held_out_r2": float(np.mean(fold_r2)),
            "held_out_r2_per_fold": fold_r2,
            "synth_cv": float(np.mean(fold_cv)),
            "synth_quality": float(np.mean(fold_quality)),
            "fold0_synth_cv": fold_cv[0],
            "model_dof": int(fold_dof[-1]),
        })

    # Cross-model matrix: the diagonal is each producer scoring its own phase.
    logger.info("--- cross-model scoring (rows = phase source, cols = judge) ---")
    for model in producers.values():
        model.requires_grad_(False)

    matrix: dict[str, dict[str, float]] = {}
    for source, phase in phases.items():
        tensor = torch.from_numpy(phase).to(device)
        matrix[source] = {}
        for judge, model in producers.items():
            matrix[source][judge] = score_spot(
                predict_detached(model, tensor)[0, 0].cpu().numpy(), args.target_side
            )["uniformity_cv"]

    header = "phase from".ljust(11) + "".join(f"{label:>11}" for label, *_ in ZOO)
    logger.info(header)
    for source in matrix:
        logger.info(source.ljust(11) + "".join(f"{matrix[source][j]:>11.4f}" for j, *_ in ZOO))

    report = {
        "synth_n_max": SYNTH_N_MAX,
        "target_side": args.target_side,
        "folds": len(folds),
        "synth_epochs": args.synth_epochs,
        "surrogates": rows,
        "cross_model_cv": matrix,
        "seconds": time.perf_counter() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote {}", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
