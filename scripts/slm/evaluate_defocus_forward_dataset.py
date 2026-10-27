"""Evaluate frozen physical models on new acquisitions without fitting any value."""
from __future__ import annotations

import argparse
from loguru import logger
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.slm.fit_defocus_forward_dataset import PhysicalForwardFit, load_observations


def main():
    import torch
    from loguru import logger

    logger.remove()
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--fits", nargs="+", required=True, type=Path)
    parser.add_argument("--reference-directory", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    calibration = json.loads((args.reference_directory / "manifest.json").read_text())
    refs = {r["branch"]: r for r in calibration if r["label"] == "reference_start"}
    results = {}
    for fit in args.fits:
        saved = json.loads(fit.read_text())
        rows, phases, images, xy, _ = load_observations(args.directory, saved["camera_crop_center"])
        for name, params in saved["models"].items():
            model = PhysicalForwardFit(phases, images, xy, np.asarray(saved["ramp_matrix"]), 0.,
                                       np.ones(len(rows), dtype=bool), params["amplitude_center"],
                                       learn_residual=bool(params.get("residual_noll_radians")))
            with torch.no_grad():
                model.d.fill_(params["defocus_diopter"])
                model.origin.copy_(torch.tensor(params["origin"]))
                model.log_width.copy_(torch.tensor(params["amplitude_width"]).log())
                for i, j in enumerate(model.residual_modes):
                    model.residual[i] = params["residual_noll_radians"][str(j)]
                predicted = model.prediction().numpy()*params["gain"]+params["background"]
            metrics = []
            for i, row in enumerate(rows):
                center = np.array([refs[row["branch"]]["spot"][f"centroid_{k}"] for k in ("x", "y")])
                mask = np.all(np.abs(xy-center) <= 30, axis=-1)
                m = {k: row[k] for k in ("file", "label", "branch", "family")}
                m["corr"] = float(np.corrcoef(images[i].ravel(), predicted[i].ravel())[0, 1])
                for label, a in (("measured", images[i]), ("predicted", predicted[i])):
                    values = a[mask]
                    m[label+"_cv"] = float(values.std()/max(values.mean(), 1e-9))
                    m[label+"_ee_roi"] = float(values.sum()/max(a.sum(), 1e-9))
                metrics.append(m)
            results[name] = {"source_fit": str(fit.resolve()), "rows": metrics}
            np.savez_compressed(out/f"prediction_{name}.npz", predicted=predicted, measured=images, camera_xy=xy)
            logger.info(name, [(b, np.mean([m["corr"] for m in metrics if m["branch"]==b and m["family"]!="reference"])) for b in ("U", "C")])
    (out/"evaluation.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
