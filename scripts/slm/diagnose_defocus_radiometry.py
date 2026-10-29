"""Audit exposure bias and effective blur/phase-independent light using saved data.

Reads a completed sampling audit; never recomputes or overwrites its predictions.
All choices use training samples. Effective blur and unmodulated-light mixing are
diagnostic hypotheses, not measured detector PSF or a proven zero-order fraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.slm.diagnose_defocus_sampling import metrics, response_bias, stamp
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-source", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    source_config = json.loads((args.sampling_source/"config.json").read_text())
    rows = json.loads((args.sampling_source/"records.json").read_text())
    original_rows = json.loads((Path(source_config["directory"])/"manifest.json").read_text())
    refs = {r["branch"]: r for r in original_rows if r["label"] == "reference_start"}
    with np.load(args.sampling_source/"pixel_3/arrays.npz") as arrays:
        optical, images, xy = [arrays[k] for k in ("optical", "measured", "camera_xy")]
    with np.load(args.sampling_source/"inputs_grid_512.npz") as inputs:
        train = inputs["train_mask"]
    exposure = np.array([r["exposure_ms"] for r in rows])
    bias = response_bias(args.response, xy)
    corrected = images-bias["intercept"]/exposure[:, None, None]
    np.savez_compressed(out/"response_fit.npz", **bias, camera_xy=xy)
    np.savez_compressed(out/"inputs.npz", optical=optical, raw_rate=images, corrected_rate=corrected,
                        exposure_ms=exposure, train_mask=train, camera_xy=xy)
    (out/"records.json").write_text(json.dumps(rows, indent=2))
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (Path(__file__), ROOT/"scripts/slm/diagnose_defocus_sampling.py",
                        args.sampling_source/"pixel_3/arrays.npz", args.response/"manifest.json")}
    for row in json.loads((args.response/"manifest.json").read_text()):
        p = args.response/row["file"]
        hashes[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out/"config.json").write_text(json.dumps(dict(source=str(args.sampling_source.resolve()),
                                                  response=str(args.response.resolve()),
                                                  hashes=hashes, started=stamp()), indent=2))
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (out/"diagnose_defocus_sampling.py").write_bytes((ROOT/"scripts/slm/diagnose_defocus_sampling.py").read_bytes())
    # A physical, phase-independent template: model prediction of a zero command.
    reference_index = next(i for i, r in enumerate(rows) if r["label"] == "reference_start" and r["branch"] == "U")
    template = optical[reference_index]
    np.save(out/"unmodulated_template.npy", template)
    results, history = {}, []
    for mixture in (False, True):
        for sigma in (0., 1., 2., 3., 4., 6., 8.):
            started = stamp()
            blurred = gaussian_filter(optical, sigma=(0, sigma/2, sigma/2), mode="constant")
            columns = [blurred[train].ravel()]
            if mixture:
                columns.append(np.broadcast_to(template, optical.shape)[train].ravel())
            columns.append(np.ones(blurred[train].size))
            # Per-image weighting prevents bright short exposures dominating.
            scale = np.sqrt((corrected[train]**2).mean((1, 2))).clip(1e-8)
            weights = np.broadcast_to((1/scale)[:, None, None], blurred[train].shape).ravel()
            coefficients = nnls(np.stack(columns, axis=1)*weights[:, None], corrected[train].ravel()*weights)[0]
            predicted = blurred*coefficients[0]+coefficients[-1]
            if mixture:
                predicted += template*coefficients[1]
            errors = ((predicted[train]-corrected[train])**2).mean((1, 2))/(corrected[train]**2).mean((1, 2)).clip(1e-12)
            result = dict(started=started, ended=stamp(), sigma_ccd_px=sigma,
                          mixture=mixture, coefficients=coefficients.tolist(),
                          training_score=float(errors.mean()),
                          rows=metrics(rows, corrected, predicted, xy, refs))
            key = f"{'mixture' if mixture else 'single'}_sigma_{sigma:g}"
            stage = out/key
            stage.mkdir()
            np.savez_compressed(stage/"arrays.npz", optical=blurred, measured=corrected,
                                predicted=predicted, camera_xy=xy, intercept=bias["intercept"])
            (stage/"result.json").write_text(json.dumps(result, indent=2))
            results[key] = result
            history.append(dict(iteration=len(history), **result["ended"], stage=key,
                                 training_score=result["training_score"], coefficients=coefficients.tolist()))
            with (out/"diagnostic_iterations.pkl").open("wb") as stream:
                pickle.dump(history, stream, protocol=5)
            (out/"summary.json").write_text(json.dumps(results, indent=2))
            logger.info(key, result["training_score"], coefficients.tolist())
    results["selection"] = dict(stage=min(results, key=lambda k: results[k]["training_score"]),
                                 rule="minimum training normalized MSE; never test CV")
    validation = bias["validation_mask"]
    results["response_summary"] = dict(negative_intercept_fraction=float(np.mean(bias["raw_intercept"]<0)),
                                       intercept_percentiles=np.percentile(bias["intercept"], [5, 50, 95]).tolist(),
                                       validation_rmse_counts=float(np.sqrt(np.mean((bias["predicted"][validation]-bias["frames"][validation])**2))),
                                       corrected_negative_fraction=float(np.mean(corrected<0)))
    response_rows = json.loads((args.response/"manifest.json").read_text())
    results["response_summary"]["stable_validation_by_branch"] = {
        b: sum(bool(valid) and r["branch"] == b for valid, r in zip(validation, response_rows))
        for b in ("U", "C")}
    (out/"summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
