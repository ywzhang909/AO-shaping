"""Reanalyze archived square-1000 ABBA differences without device access."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward, load_frozen_parameters,
)
from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective
from ao_shaping.tools.slm.transfer.radiometry import count_rate

ROOT = Path(__file__).resolve().parents[2]


def cosine(a, b, support):
    x, y = a[support], b[support]
    return float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "protocol", ROOT / "scripts/slm/square_1000_diff_abba_20261030.py")
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    torch.set_num_threads(4)
    manifest = json.loads((args.session / "manifest.json").read_text())
    contract = json.loads((args.session / "measurement_contract.json").read_text())
    bg = contract["background_counts"]
    rates, metrics = {}, {}
    for row in manifest:
        with np.load(args.session / row["file"]) as saved:
            xy = saved["camera_xy"]
            yy, xx = np.mgrid[:1944, :2592]
            native_masks = protocol.masks(np.stack([xx, yy], -1))
            rate = count_rate(np.maximum(saved["raw_frames"].mean(0) - bg, 0.)[
                xy[..., 1].astype(int), xy[..., 0].astype(int)], row)
            rates[row["label"]] = rate
            metrics[row["label"]] = protocol.metrics(
                saved["raw_frames"], row["exposure_ms"], bg,
                contract["coverage_threshold_counts"], native_masks,
                ShapeObjective(protocol.masks(xy)["box"].astype(np.float32), formation=True), xy)
    params = load_frozen_parameters(args.checkpoint).effective_parameters()
    forward = FrozenDefocusForward(params, xy, device="cpu")
    forward.exposure_ms = manifest[0]["exposure_ms"]
    objective = ShapeObjective(protocol.masks(xy)["box"].astype(np.float32), formation=True)
    report = {"session": str(args.session), "metric_recalculation_max_error": 0.,
              "radiometry": [], "abba": [], "proposals": []}
    for label, measured in rates.items():
        with np.load(args.session / next(r["file"] for r in manifest if r["label"] == label)) as saved:
            with torch.no_grad():
                prediction = forward.predict(saved["model_phase"])[0].numpy()
        a, b = np.polyfit(prediction.ravel(), measured.ravel(), 1)
        residual = measured - (a * prediction + b)
        report["radiometry"].append(dict(label=label, gain=a, offset=b,
            relative_l2=float(np.linalg.norm(residual) / np.linalg.norm(measured)),
            correlation=float(np.corrcoef(prediction.ravel(), measured.ravel())[0, 1])))
        for key in metrics[label]:
            if isinstance(metrics[label][key], (float, int)):
                old = next(r for r in manifest if r["label"] == label).get("metrics", {}).get(key)
                if old is None:
                    continue
                report["metric_recalculation_max_error"] = max(
                    report["metric_recalculation_max_error"], abs(old - metrics[label][key]))
    for block, tag in ((0, "gradient"), (0, "reverse"), (1, "gradient")):
        prefix = f"block{block}_{tag}_"
        if prefix + "A1" not in rates:
            continue
        measured_delta = .5 * (rates[prefix+"B1"] + rates[prefix+"B2"] - rates[prefix+"A1"] - rates[prefix+"A2"])
        with np.load(args.session / f"proposal_{block:02d}.npz") as saved:
            delta_phase = saved["delta"] if tag == "gradient" else -saved["delta"]
            anchor = saved["seed"]
        with torch.no_grad():
            p0 = forward.predict(anchor)[0].numpy()
            p1 = forward.predict(anchor + delta_phase)[0].numpy()
            loss0, terms0 = objective.evaluate(torch.from_numpy(rates[prefix+"A1"]), torch.zeros((2, 2)))
            loss1, terms1 = objective.evaluate(torch.from_numpy(p1), torch.zeros((2, 2)))
        support = np.abs(measured_delta) > .15 * np.abs(measured_delta).max()
        report["abba"].append(dict(block=block, tag=tag,
            cosine=cosine(p1-p0, measured_delta, support),
            correlation=float(np.corrcoef((p1-p0).ravel(), measured_delta.ravel())[0, 1]),
            measured_delta_rms=float(np.sqrt(np.mean(measured_delta**2))),
            predicted_delta_rms=float(np.sqrt(np.mean((p1-p0)**2))),
            measured_anchor_loss=float(loss0), predicted_candidate_loss=float(loss1),
            predicted_loss_change=float(loss1-loss0), anchor_terms=terms0,
            candidate_terms=terms1))
    for path in sorted(args.session.glob("proposal_*.json")):
        report["proposals"].append(json.loads(path.read_text()))
    out = args.session / "offline_direction_analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
