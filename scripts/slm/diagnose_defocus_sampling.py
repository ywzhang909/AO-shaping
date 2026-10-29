"""Audit physical sampling, detector integration, radiometry and effective blur.

Never opens devices. Every stage retains its input arrays, predictions, metrics,
source hashes and precise timestamps in a new directory. Optical parameters stay
frozen; radiometric and blur selection use the original training samples only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.slm.fit_defocus_forward_dataset import PhysicalForwardFit, load_observations
from loguru import logger


def stamp():
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                monotonic_ns=time.monotonic_ns())


def fixed_panel_basis(size):
    """Sample canonical modes with the historical fit's physical centre/radius.

    The 128-grid fit centred its modes at 63.5*15=952.5 panel pixels. Sampling a
    full-resolution canonical grid shifted by seven pixels keeps that same
    physical basis when refining the numerical grid, avoiding a hidden tilt.
    """
    from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

    yy, xx = np.mgrid[:size, :size] * (1920./size)
    return {j: map_coordinates(np.nan_to_num(generate_zernike_phase({j: 1.}, (1920, 1920))),
                                [yy+7, xx+7], order=1, mode="constant").astype(np.float32)
            for j in range(4, 12)}


def restore_model(phases, images, xy, fit, params, *, fft_size, pixels, physical_basis):
    import torch

    model = PhysicalForwardFit(phases, images, xy, np.asarray(fit["ramp_matrix"]), 0.,
                               np.ones(len(phases), bool), params["amplitude_center"],
                               learn_residual=True, fft_size=fft_size, pixel_samples=pixels,
                               basis_arrays=physical_basis, batch_size=1)
    with torch.no_grad():
        model.d.fill_(params["defocus_diopter"])
        model.origin.copy_(torch.tensor(params["origin"]))
        model.log_width.copy_(torch.tensor(params["amplitude_width"]).log())
        for i, j in enumerate(model.residual_modes):
            model.residual[i] = params["residual_noll_radians"][str(j)]
    return model


def response_bias(directory, xy):
    """Shared per-pixel affine intercept, independent slopes for two fixed phases.

    The repeated half level is validation, not part of this fit. The resulting
    illuminated intercept is diagnostic and is never labelled a dark frame.
    """
    rows = json.loads((directory/"manifest.json").read_text())
    frames = []
    for row in rows:
        with np.load(directory/row["file"]) as data:
            frame = data["raw_frames"].mean(0)
            frames.append(frame[xy[..., 1].astype(int), xy[..., 0].astype(int)])
    frames = np.stack(frames)
    # Reserve any repeat of an earlier exposure at the same fixed phase.
    seen = set()
    train = []
    for row in rows:
        key = (row["branch"], row["exposure_ms"])
        train.append(key not in seen)
        seen.add(key)
    train = np.asarray(train)
    stable = np.array([r.get("settled", {}).get("settled", True) for r in rows])
    validation = ~train & stable
    train &= stable
    exposure = np.array([r["exposure_ms"] for r in rows])
    branch = np.array([r["branch"] == "U" for r in rows])
    design = np.stack((exposure*branch, exposure*(~branch), np.ones(len(rows))), axis=1)
    coefficients = np.linalg.lstsq(design[train], frames[train].reshape(train.sum(), -1), rcond=None)[0]
    raw_intercept = coefficients[-1].reshape(xy.shape[:2])
    intercept = raw_intercept.clip(0, 255)
    # Once the physical nonnegative intercept bound is applied, refit only slopes.
    slopes = np.linalg.lstsq(design[train, :2], (frames[train]-intercept).reshape(train.sum(), -1), rcond=None)[0]
    predicted = (design[:, :2]@slopes).reshape(frames.shape)+intercept
    return dict(raw_intercept=raw_intercept, intercept=intercept, frames=frames,
                predicted=predicted, exposure_ms=exposure, validation_mask=validation,
                stable_mask=stable)


def metrics(rows, images, prediction, xy, references):
    results = []
    for row, measured, predicted in zip(rows, images, prediction):
        center = [references[row["branch"]]["spot"][f"centroid_{k}"] for k in ("x", "y")]
        mask = np.all(np.abs(xy-center) <= 30, axis=-1)
        result = {k: row[k] for k in ("file", "branch", "family", "split", "label")}
        result["corr"] = float(np.corrcoef(measured.ravel(), predicted.ravel())[0, 1])
        for name, a in (("measured", measured), ("predicted", predicted)):
            values = a[mask]
            result[name+"_cv"] = float(values.std()/max(values.mean(), 1e-9))
            result[name+"_ee_roi"] = float(values.sum()/max(a.sum(), 1e-9))
            signal = np.maximum(a, 0)
            total = max(signal.sum(), 1e-9)
            centroid = np.sum(signal[..., None]*xy, axis=(0, 1))/total
            result[name+"_center"] = centroid.tolist()
            result[name+"_rms_radius"] = float(np.sqrt(np.sum(signal*np.sum((xy-centroid)**2, axis=-1))/total))
        result["center_error_px"] = float(np.linalg.norm(np.array(result["measured_center"])-result["predicted_center"]))
        result["rms_relative_error"] = float(abs(result["predicted_rms_radius"]/max(result["measured_rms_radius"], 1e-9)-1))
        results.append(result)
    return results


def fit_radiometry(prediction, images, mask):
    design = np.stack((prediction[mask].ravel(), np.ones(prediction[mask].size)), axis=1)
    gain, background = nnls(design, images[mask].ravel())[0]
    return prediction*gain+background, float(gain), float(background)


def main():
    import torch
    from loguru import logger

    logger.remove()
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--evaluation-directory", required=True, type=Path)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fit = json.loads(args.fit.read_text())
    params = fit["models"]["M4_residual"]
    reference_rows = json.loads((args.directory/"manifest.json").read_text())
    references = {r["branch"]: r for r in reference_rows if r["label"] == "reference_start"}
    config = {k: str(v.resolve()) for k, v in vars(args).items()}
    config.update(stamp())
    config["stages"] = [("legacy", 128, 512, 1, False),
                        ("physical_basis", 128, 512, 1, True),
                        ("fft_1024", 128, 1024, 1, True),
                        ("fft_2048", 128, 2048, 1, True),
                        ("grid_256", 256, 2048, 1, True),
                        ("grid_512", 512, 2048, 1, True),
                        ("fft_4096", 512, 4096, 1, True),
                        ("pixel_3", 512, 4096, 3, True)]
    config["source_hashes"] = {}
    for source in (Path(__file__), ROOT/"scripts/slm/fit_defocus_forward_dataset.py", args.fit):
        config["source_hashes"][str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
    # Preserve the exact analysis source, not only a git HEAD missing local edits.
    for source in (Path(__file__), ROOT/"scripts/slm/fit_defocus_forward_dataset.py"):
        (out/source.name).write_bytes(source.read_bytes())
    (out/"config.json").write_text(json.dumps(config, indent=2))
    summary = {}
    datasets = {}
    for size in (128, 256, 512):
        loaded = [load_observations(d, fit["camera_crop_center"], grid_size=size)
                  for d in (args.directory, args.evaluation_directory)]
        rows = loaded[0][0]+[{**r, "split": "external", "file": str(args.evaluation_directory/r["file"])} for r in loaded[1][0]]
        phases = np.concatenate([a[1] for a in loaded])
        images = np.concatenate([a[2] for a in loaded])
        xy = loaded[0][3]
        train = np.array([r["split"] != "test" and r["split"] != "external" for r in rows])
        datasets[size] = rows, phases, images, xy, train
        np.savez_compressed(out/f"inputs_grid_{size}.npz", phases=phases, measured=images,
                            camera_xy=xy, exposure_ms=np.array([r["exposure_ms"] for r in rows]), train_mask=train)
    (out/"records.json").write_text(json.dumps(datasets[128][0], indent=2))
    inputs = {}
    for d in (args.directory, args.evaluation_directory, args.response):
        for row in json.loads((d/"manifest.json").read_text()):
            source = d/row["file"]
            inputs[str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
    (out/"raw_input_hashes.json").write_text(json.dumps(inputs, indent=2))
    for name, size, fft, pixels, physical in config["stages"]:
        stage = out/name
        stage.mkdir()
        rows, phases, images, xy, train = datasets[size]
        started = stamp()
        model = restore_model(phases, images, xy, fit, params, fft_size=fft, pixels=pixels,
                              physical_basis=fixed_panel_basis(size) if physical else None)
        with torch.no_grad():
            optical = model.prediction().numpy()
        predicted = optical*params["gain"]+params["background"]
        result = dict(started=started, ended=stamp(), grid_size=size, fft_size=fft,
                      pixel_samples=pixels, radiometry="frozen", rows=metrics(rows, images, predicted, xy, references))
        np.savez_compressed(stage/"arrays.npz", optical=optical, predicted=predicted, measured=images, camera_xy=xy)
        (stage/"result.json").write_text(json.dumps(result, indent=2))
        summary[name] = result
        (out/"summary.json").write_text(json.dumps(summary, indent=2))
        logger.info(name, [(b, np.mean([abs(r['measured_cv']-r['predicted_cv']) for r in result['rows'] if r['split']=='external' and r['family']!='reference' and r['branch']==b])) for b in ('U','C')])
        del model
    # Bias is calibrated independently from fixed-phase exposure data.
    bias = response_bias(args.response, xy)
    bias_stage = out/"exposure_bias"
    bias_stage.mkdir()
    np.savez_compressed(bias_stage/"response_fit.npz", **bias, camera_xy=xy)
    exposure = np.array([r["exposure_ms"] for r in rows])
    corrected = images-bias["intercept"]/exposure[:, None, None]
    # Do not clip corrected signal: it changes CV and hides offset uncertainty.
    negatives = float(np.mean(corrected < 0))
    iterations = []
    # Effective blur is a model-discrepancy diagnostic, not a measured CCD PSF.
    for index, sigma in enumerate((0., .5, 1., 2., 3., 4., 6., 8.)):
        stage = out/f"blur_{index:02d}"
        stage.mkdir()
        started = stamp()
        blurred = gaussian_filter(optical, sigma=(0, sigma/2, sigma/2), mode="constant")
        predicted, gain, background = fit_radiometry(blurred, corrected, train)
        # Use training image errors to choose blur; no test CV tuning.
        errors = ((predicted[train]-corrected[train])**2).mean((1, 2)) / (corrected[train]**2).mean((1, 2)).clip(1e-12)
        result = dict(started=started, ended=stamp(), sigma_ccd_px=sigma, gain=gain,
                      background=background, training_score=float(errors.mean()), negative_fraction=negatives,
                      rows=metrics(rows, corrected, predicted, xy, references))
        np.savez_compressed(stage/"arrays.npz", optical=blurred, predicted=predicted,
                            measured=corrected, exposure_ms=exposure, intercept=bias["intercept"], camera_xy=xy)
        (stage/"result.json").write_text(json.dumps(result, indent=2))
        summary[f"blur_{index:02d}"] = result
        iterations.append(dict(iteration=index, **result["ended"], sigma_ccd_px=sigma,
                               training_score=result["training_score"], gain=gain, background=background))
        with (out/"diagnostic_iterations.pkl").open("wb") as stream:
            pickle.dump(iterations, stream, protocol=5)
        logger.info('blur', sigma, result['training_score'])
    winner = min((k for k in summary if k.startswith("blur_")), key=lambda k: summary[k]["training_score"])
    summary["selection"] = dict(stage=winner, rule="minimum training normalized image MSE; optical parameters frozen",
                                 bias_negative_intercept_fraction=float(np.mean(bias["raw_intercept"]<0)),
                                 note="Previously inspected external data are diagnostic; fresh hardware validation required after model selection.")
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    logger.info('selected', winner)


if __name__ == "__main__":
    main()
