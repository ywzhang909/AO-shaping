"""Paired Santec #1 / Daheng forward prediction on saved model-in-loop frames.

Train only on older 0.4 ms runs. The four default evaluation runs use a new
probe seed; their holdout frames never enter pretraining or per-run adaptation.
The saved physical prediction and U-Net outputs are scored against the *same*
CCD model grid, with the same frozen square and exposure readback.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ao_shaping.optimizer.wfless.slm_model_in_loop import _to_model_grid
from ao_shaping.optimizer.wfless.slm_model_in_loop import _metrics_at
from ao_shaping.utils.image.beam_metrics import zero_order_center
from ml.phase.unet import UNetGenerator


DEFAULT_EVAL = ("20261013_173039", "20261013_173230",
                "20261013_174628", "20261013_174740")


@dataclass
class Sample:
    run: str
    split: str
    family: str
    index: int
    exposure_ms: float
    phase: np.ndarray
    measured: np.ndarray
    physical: np.ndarray | None
    observed_mask: np.ndarray

    @property
    def phase_key(self) -> str:
        return hashlib.sha256(np.asarray(self.phase, dtype="<f4").tobytes()).hexdigest()


def load_run(root: Path, *, exposure_ms: float | None = 0.4) -> list[Sample]:
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    geometry = json.loads((root / "bench_geometry.json").read_text(encoding="utf-8"))
    if (config.get("cam_type"), config.get("slm_number"), config.get("region")) != (
        "daheng", 1, 64
    ):
        raise ValueError(f"{root}: expected Daheng / Santec #1 / 64-pixel pupil")
    records = root / "probes"
    manifest = json.loads((records / "manifest.json").read_text(encoding="utf-8"))
    with np.load(records / "round_00_flat_00.npz") as flat:
        center = zero_order_center(flat["frame"], refine=False)
        observed_mask = _to_model_grid(
            np.ones_like(flat["frame"], dtype=np.float32), center,
            int(geometry["far_field_size"]),
            float(geometry["camera_px_per_model_px"]), 1,
        ) > 0.5
    result = []
    for row in manifest:
        actual_ms = float(row["exposure_ms"])
        if row["split"] == "flat" or not 0 < actual_ms <= 0.4:
            continue
        if exposure_ms is not None and abs(actual_ms - exposure_ms) > 1e-6:
            continue
        with np.load(records / row["file"]) as record:
            phase = np.asarray(record["phase"], dtype=np.float32).copy()
            measured = (
                np.asarray(record["model_grid"], dtype=np.float32).copy()
                if "model_grid" in record
                else _to_model_grid(
                    record["frame"], center, int(geometry["far_field_size"]),
                    float(geometry["camera_px_per_model_px"]), 1,
                ).astype(np.float32)
            )
            physical = (
                np.asarray(record["prediction"], dtype=np.float32).copy()
                if "prediction" in record else None
            )
        if phase.shape != (64, 64) or measured.shape != (192, 192):
            raise ValueError(f"{root}/{row['file']}: unexpected grid")
        result.append(Sample(
            root.name, row["split"], row.get("family") or row["split"],
            int(row["index"]), float(row["exposure_ms"]), phase, measured, physical,
            observed_mask,
        ))
    return result


def _input(
    phase: np.ndarray, exposure_ms: float = 0.4, *, exposure_channel: bool = False,
) -> torch.Tensor:
    phase = torch.from_numpy(np.asarray(phase, dtype=np.float32))[None, None]
    phasor = torch.cat((phase.cos(), phase.sin()), dim=1)
    resized = F.interpolate(phasor, size=(192, 192), mode="bilinear",
                            align_corners=False)[0]
    if exposure_channel:
        covariate = np.log10(exposure_ms / 0.4)
        resized = torch.cat((resized, torch.full_like(resized[:1], float(covariate))))
    return resized


def _target(image: np.ndarray) -> torch.Tensor:
    image = np.maximum(np.asarray(image, dtype=np.float32), 0)
    peak = max(float(image.max()), 1e-12)
    return torch.from_numpy((image / peak)[None])


def _tensors(
    samples: list[Sample], device: torch.device, *, exposure_channel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.stack([
        _input(s.phase, s.exposure_ms, exposure_channel=exposure_channel) for s in samples
    ]).to(device),
            torch.stack([_target(s.measured) for s in samples]).to(device))


def _fit(
    model: UNetGenerator, samples: list[Sample], *,
    steps: int, lr: float, device: torch.device, seed: int,
    exposure_channel: bool,
) -> None:
    if not samples:
        raise ValueError("no 0.4 ms training frames")
    x, y = _tensors(samples, device, exposure_channel=exposure_channel)
    model.train()
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        indices = rng.choice(len(samples), size=min(8, len(samples)), replace=False)
        pred = model(x[indices])
        # The 192-pixel grid is mostly dark; plain MSE otherwise rewards a
        # nearly constant frame on this 36-phase corpus.
        weight = 1.0 + 20.0 * y[indices]
        loss = torch.mean(weight * (pred - y[indices]).square())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()


def _score(
    image: np.ndarray, measured: np.ndarray, side: int, observed_mask: np.ndarray,
    *, target_shape: str = "square",
) -> dict[str, float]:
    a = np.where(observed_mask, np.maximum(np.asarray(image, dtype=np.float64), 0), 0)
    b = np.where(observed_mask, np.maximum(np.asarray(measured, dtype=np.float64), 0), 0)
    corr = float(np.corrcoef(a[observed_mask], b[observed_mask])[0, 1])
    metric = _metrics_at(a, side, (96, 96), target_shape=target_shape)
    return {
        "correlation": corr,
        "cv": float(metric["uniformity_cv"]),
        "ee": float(metric["encircled_energy"]),
    }


def run(
    data_root: Path, output_dir: Path, *,
    epochs: int = 80, adapt_steps: int = 10, seed: int = 20261013,
    mixed_exposure_covariate: bool = False,
) -> list[dict]:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_samples = {name: load_run(data_root / name) for name in DEFAULT_EVAL}
    holdouts = [s for rows in eval_samples.values() for s in rows if s.split == "holdout"]
    forbidden = {s.phase_key for s in holdouts}
    prior = []
    for root in sorted(data_root.glob("20261013_*")):
        if root.name >= DEFAULT_EVAL[0] or not (root / "run_config.json").is_file():
            continue
        for sample in load_run(
            root, exposure_ms=None if mixed_exposure_covariate else 0.4
        ):
            if sample.phase_key not in forbidden:
                prior.append(sample)
    # Repeated phase commands from paired earlier runs should not count twice.
    unique = {sample.phase_key: sample for sample in prior}
    prior = list(unique.values())
    if len(prior) < 10:
        raise ValueError("too few disjoint historical 0.4 ms phases")

    model = UNetGenerator(in_channels=3 if mixed_exposure_covariate else 2,
                          features=[8, 16, 32],
                          output_mode="phase").to(device)
    _fit(model, prior, steps=epochs * max(1, (len(prior) + 7) // 8),
         lr=1e-3, device=device, seed=seed,
         exposure_channel=mixed_exposure_covariate)
    base_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "predictions").mkdir()
    torch.save({
        "model_state": base_state, "features": [8, 16, 32], "output_mode": "phase",
        "in_channels": 3 if mixed_exposure_covariate else 2,
        "prior_runs": sorted({s.run for s in prior}), "seed": seed,
    }, output_dir / "unet_prior.pt")

    rows: list[dict] = []
    for run_name, samples in eval_samples.items():
        adaptation = [s for s in samples if s.split == "train"]
        model.load_state_dict(base_state)
        zero_predictions = {}
        with torch.no_grad():
            for sample in samples:
                if sample.split == "holdout":
                    zero_predictions[sample.index] = model(
                        _input(sample.phase, sample.exposure_ms,
                               exposure_channel=mixed_exposure_covariate)[None].to(device)
                    )[0, 0].cpu().numpy().copy()
        _fit(model, adaptation, steps=adapt_steps, lr=1e-4,
             device=device, seed=seed + int(run_name[-4:]),
             exposure_channel=mixed_exposure_covariate)
        for sample in samples:
            if sample.split != "holdout":
                continue
            if sample.physical is None:
                raise ValueError(f"{run_name}: missing paired physical prediction")
            with torch.no_grad():
                adapted = model(_input(
                    sample.phase, sample.exposure_ms,
                    exposure_channel=mixed_exposure_covariate,
                )[None].to(device))[0, 0].cpu().numpy()
            np.savez_compressed(
                output_dir / "predictions" / f"{run_name}_holdout_{sample.index:02d}.npz",
                phase=sample.phase, ccd=sample.measured, physical=sample.physical,
                unet_prior=zero_predictions[sample.index], unet_adapted=adapted,
                observed_mask=sample.observed_mask,
                exposure_ms=np.float64(sample.exposure_ms),
            )
            measured_metrics = _score(
                sample.measured, sample.measured, 36, sample.observed_mask,
            )
            for name, image in (
                ("ccd", sample.measured),
                ("physical", sample.physical),
                ("unet_prior", zero_predictions[sample.index]),
                ("unet_adapted", adapted),
            ):
                rows.append({
                    "run": run_name, "family": sample.family, "index": sample.index,
                    "exposure_ms": sample.exposure_ms, "model": name,
                    "training_probes": len(adaptation), "prior_unique_phases": len(prior),
                    **_score(image, sample.measured, 36, sample.observed_mask),
                    "ccd_cv": measured_metrics["cv"], "ccd_ee": measured_metrics["ee"],
                })
    with (output_dir / "paired_frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "config.json").write_text(json.dumps({
        "source_root": str(data_root), "evaluation_runs": DEFAULT_EVAL,
        "train_runs": sorted({s.run for s in prior}),
        "prior_unique_phases": len(prior), "seed": seed, "epochs": epochs,
        "adapt_steps": adapt_steps, "device": str(device), "target_side_model_px": 36,
        "unet_trainable_parameters": sum(p.numel() for p in model.parameters()),
        "matched_exposure_ms": 0.4,
        "mixed_exposure_covariate": mixed_exposure_covariate,
        "loss": "mean((1 + 20 * peak_normalized_target) * squared_error)",
        "note": "No evaluation phase entered pretraining; only each run's train probes entered adaptation.",
    }, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--adapt-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--mixed-exposure-covariate", action="store_true")
    args = parser.parse_args()
    rows = run(args.data_root, args.output_dir, epochs=args.epochs,
               adapt_steps=args.adapt_steps, seed=args.seed,
               mixed_exposure_covariate=args.mixed_exposure_covariate)
    for model in ("physical", "unet_prior", "unet_adapted"):
        values = [r for r in rows if r["model"] == model]
        print(model, {key: round(float(np.mean([v[key] for v in values])), 4)
                      for key in ("correlation", "cv", "ee")})


if __name__ == "__main__":
    main()
