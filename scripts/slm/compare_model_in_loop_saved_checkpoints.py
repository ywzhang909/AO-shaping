"""Evaluate frozen historical U-Net and saved physical predictions on later CCD frames.

All evaluation frames are read from disk. Only each run's two training probes may
adapt the U-Net; holdouts and shaped frames are never used for fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# `scripts.*` is imported as a package path, so the repo ROOT itself must be
# importable -- under a file-form launch (`python scripts/x/y.py`) the cwd is
# not on sys.path and only the script's own directory is.
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ml.phase.unet import UNetGenerator
from scripts.slm.compare_model_in_loop_unet import _fit, _input, _score, load_run


DEFAULT_RUNS = (
    "20261014_170820", "20261014_171043", "20261014_171219",
    "20261014_171301", "20261014_171340",
)


def check_disjoint(train_keys: set[str], eval_keys: set[str]) -> None:
    overlap = train_keys & eval_keys
    if overlap:
        raise ValueError(f"checkpoint training overlaps evaluation by {len(overlap)} phases")


def evaluate(
    data_root: Path, checkpoint: Path, output_dir: Path, *,
    runs: tuple[str, ...] = DEFAULT_RUNS, adapt_steps: int = 10,
) -> list[dict]:
    if adapt_steps < 0:
        raise ValueError("adapt_steps must be nonnegative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["output_mode"] != "phase" or saved["in_channels"] != 3:
        raise ValueError("expected historical 3-channel, sigmoid-output U-Net")
    evaluation = {name: load_run(data_root / name) for name in runs}
    train_keys = {
        sample.phase_key
        for name in saved["prior_runs"]
        for sample in load_run(data_root / name, exposure_ms=None)
    }
    eval_keys = {
        sample.phase_key for samples in evaluation.values() for sample in samples
    }
    check_disjoint(train_keys, eval_keys)
    model = UNetGenerator(
        in_channels=3, features=saved["features"], output_mode="phase"
    ).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "predictions").mkdir()
    rows: list[dict] = []
    for name, samples in evaluation.items():
        config = json.loads((data_root / name / "run_config.json").read_text(encoding="utf-8"))
        geometry = json.loads((data_root / name / "bench_geometry.json").read_text(encoding="utf-8"))
        shape = config["target_shape"]
        # The runner's 64-cell target is expanded by the FFT padding to 192.
        # Thus the saved prediction grid uses the calibrated camera px/FFT cell.
        side = round(config["target_side"] / geometry["camera_px_per_model_px"])
        training = [s for s in samples if s.split == "train"]
        assessed = [s for s in samples if s.split in {"holdout", "shaped"}]
        if len(training) != 2 or len(assessed) != 7:
            raise ValueError(f"{name}: expected 2 train and 7 evaluation frames")
        if any(s.physical is None for s in assessed):
            raise ValueError(f"{name}: missing saved physical prediction")
        if any(abs(s.exposure_ms - 0.4) > 1e-6 for s in samples):
            raise ValueError(f"{name}: exposure mismatch")
        model.load_state_dict(saved["model_state"])
        frozen = {}
        with torch.inference_mode():
            for sample in assessed:
                x = _input(sample.phase, sample.exposure_ms, exposure_channel=True)[None].to(device)
                frozen[(sample.split, sample.index)] = model(x)[0, 0].cpu().numpy().copy()
        if adapt_steps:
            _fit(model, training, steps=adapt_steps, lr=1e-4, device=device,
                 seed=saved["seed"] + int(name[-4:]), exposure_channel=True)
        for sample in assessed:
            with torch.inference_mode():
                x = _input(sample.phase, sample.exposure_ms, exposure_channel=True)[None].to(device)
                adapted = model(x)[0, 0].cpu().numpy().copy()
            prior = frozen[(sample.split, sample.index)]
            np.savez_compressed(
                output_dir / "predictions" / f"{name}_{sample.split}_{sample.index:02d}.npz",
                ccd=sample.measured, physical=sample.physical, unet_prior=prior,
                unet_adapted=adapted, observed_mask=sample.observed_mask,
                phase=sample.phase, exposure_ms=sample.exposure_ms,
            )
            ccd = _score(sample.measured, sample.measured, side,
                         sample.observed_mask, target_shape=shape)
            for kind, image in (("physical", sample.physical),
                                ("unet_prior", prior), ("unet_adapted", adapted)):
                rows.append({
                    "run": name, "shape": shape, "split": sample.split,
                    "family": sample.family, "index": sample.index,
                    "exposure_ms": sample.exposure_ms, "model": kind,
                    "target_side_model_px": side,
                    **_score(image, sample.measured, side, sample.observed_mask,
                             target_shape=shape),
                    "ccd_cv": ccd["cv"], "ccd_ee": ccd["ee"],
                })
    with (output_dir / "paired_frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "config.json").write_text(json.dumps({
        "evaluation_runs": runs, "checkpoint": str(checkpoint),
        "checkpoint_train_runs": saved["prior_runs"],
        "checkpoint_train_unique_phases": len(train_keys),
        "evaluation_unique_phases": len(eval_keys), "phase_overlap": 0,
        "adapt_steps": adapt_steps, "adaptation_frames_per_run": 2,
        "exposure_ms": 0.4, "device": str(device),
        "note": "Frozen U-Net and physical predictions share saved CCD frames. Adapted U-Net uses only same-run training probes. No U-Net phase was sent to SLM.",
    }, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--adapt-steps", type=int, default=10)
    args = parser.parse_args()
    rows = evaluate(args.data_root, args.checkpoint, args.output_dir,
                    adapt_steps=args.adapt_steps)
    for split in ("holdout", "shaped"):
        for model in ("physical", "unet_prior", "unet_adapted"):
            subset = [row for row in rows if row["split"] == split and row["model"] == model]
            print(split, model, {metric: round(float(np.mean([row[metric] for row in subset])), 4)
                                for metric in ("correlation", "cv", "ee")})


if __name__ == "__main__":
    main()
