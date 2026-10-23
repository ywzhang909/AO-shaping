"""Fit a compact physical model to paired CCD data, keeping test phases untouched.

Uses canonical Zernike and Fraunhofer implementations. Camera mapping is fixed by
calibration ramps. Reports predictions on the original, fixed CCD coordinate grid.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def ramp_matrix(rows: list[dict]) -> np.ndarray:
    columns = []
    for axis in ("x", "y"):
        values = []
        for period in (480, 960):
            pair = {sign: next(r for r in rows if r["label"] == f"ramp_{axis}_{period}_{sign}")
                    for sign in (-1, 1)}
            point = lambda sign, _pair=pair: np.array([_pair[sign]["spot"][f"centroid_{k}"] for k in ("x", "y")])
            values.append((point(1) - point(-1)) * period / 2)
        columns.append(np.mean(values, axis=0))
    matrix = np.stack(columns, axis=1)
    if abs(np.linalg.det(matrix)) < 1:
        raise ValueError("ramp geometry is singular")
    return matrix


from ao_shaping.algorithm.signal_processing.physical_forward_fit import PhysicalForwardFit


def load_observations(directory, camera_center=None, *, grid_size=128):
    """Read complete recorded phases and sample observations on a fixed CCD grid."""
    rows = json.loads((directory / "manifest.json").read_text())
    references = {r["branch"]: r for r in rows if r["label"] == "reference_start"}
    center = np.array([references["U"]["spot"][f"centroid_{k}"] for k in ("x", "y")]) if camera_center is None else np.asarray(camera_center)
    yy, xx = np.mgrid[-120:121:2, -120:121:2]
    camera_xy = np.stack((xx + round(center[0]), yy + round(center[1])), axis=-1)
    selected = [r for r in rows if r["family"] != "ramp" and r["settled"]["settled"]]
    phases, images = [], []
    for row in selected:
        with np.load(directory / row["file"]) as data:
            padded = np.zeros((1920, 1920), dtype=np.float32)
            padded[:1200] = data["panel_phase"]
            py, px = np.mgrid[:grid_size, :grid_size] * (1920. / grid_size)
            phases.append(map_coordinates(padded, [py, px], order=1))
            frame = data["raw_frames"].mean(0)
            images.append(frame[camera_xy[..., 1].astype(int), camera_xy[..., 0].astype(int)] / row["exposure_ms"])
    return selected, np.stack(phases), np.stack(images), camera_xy, references


def main():
    import torch
    from loguru import logger

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--residual-from", type=Path,
                        help="Fit seven residual modes starting from M3 in this JSON")
    parser.add_argument("--output", type=Path, help="New fit directory, keeping the source fit intact")
    args = parser.parse_args()
    logger.remove()
    torch.set_num_threads(4)
    out = args.directory
    destination = args.output or out
    if args.output:
        destination.mkdir(parents=True, exist_ok=False)
    elif (destination / "forward_fit.json").exists():
        parser.error("existing fit detected; supply a new --output directory")
    rows = json.loads((out / "manifest.json").read_text())
    config = json.loads((out / "run_config.json").read_text())
    matrix = ramp_matrix(rows)
    references = {r["branch"]: r for r in rows if r["label"] == "reference_start"}
    # Estimate illumination centre from chirp-induced displacement, diagnostic
    # approximation only; the amplitude-learning arm may adjust it on train data.
    shift = np.array([references["C"]["spot"][f"centroid_{k}"] - references["U"]["spot"][f"centroid_{k}"]
                      for k in ("x", "y")])
    gradient = np.linalg.solve(matrix, shift)
    q = (8e-6) ** 2 * config["best_diopter_m1"] / 1064e-9
    settings = config["settings"]
    amplitude_center = np.array([settings["center_x"], settings["center_y"]]) + gradient / q
    center = np.array([references["U"]["spot"][f"centroid_{k}"] for k in ("x", "y")])
    selected, phases, images, camera_xy, references = load_observations(out, center)
    train = np.array([r["split"] != "test" for r in selected])
    results = {"ramp_matrix": matrix.tolist(), "illumination_center_estimate": amplitude_center.tolist(),
               "camera_crop_center": center.tolist(), "sample_count": len(selected), "models": {}}
    arms = (("M0", False, 0., 0),
                                         ("M1_defocus", False, None, args.iterations),
                                         ("M3_illumination", True, None, args.iterations))
    if args.residual_from:
        if not args.output:
            parser.error("--residual-from requires a new --output directory")
        arms = (("M4_residual", True, None, args.iterations),)
    for name, learn, initial, budget in arms:
        model = PhysicalForwardFit(phases, images, camera_xy, matrix, config["best_diopter_m1"],
                                   train, amplitude_center, learn_amplitude=learn, initial_defocus=initial,
                                   learn_residual=bool(args.residual_from))
        with torch.no_grad():
            model.origin.copy_(torch.tensor(center, dtype=torch.float32))
            if args.residual_from:
                previous = json.loads(args.residual_from.read_text())["models"]["M3_illumination"]
                model.d.fill_(previous["defocus_diopter"])
                model.origin.copy_(torch.tensor(previous["origin"]))
                model.center.copy_(torch.tensor(previous["amplitude_center"]))
                model.log_width.copy_(torch.tensor(previous["amplitude_width"]).log())
        prediction = model.run(budget)
        design = np.stack((prediction[train].ravel(), np.ones(prediction[train].size)), axis=1)
        gain, background = nnls(design, images[train].ravel())[0]
        prediction = prediction * gain + background
        metrics = []
        for i, row in enumerate(selected):
            branch_center = np.array([references[row["branch"]]["spot"][f"centroid_{k}"] for k in ("x", "y")])
            mask = np.all(np.abs(camera_xy - branch_center) <= 30, axis=-1)
            optical = {}
            for label, array in (("measured", images[i]), ("predicted", prediction[i])):
                values = array[mask]
                optical[label + "_cv"] = float(values.std() / max(values.mean(), 1e-9))
                optical[label + "_ee_roi"] = float(values.sum() / max(array.sum(), 1e-9))
            metrics.append({"file": row["file"], "label": row["label"], "branch": row["branch"],
                            "split": row["split"], "family": row["family"],
                            **optical,
                            "corr": float(np.corrcoef(images[i].ravel(), prediction[i].ravel())[0, 1]),
                            "nrmse": float(np.sqrt(np.mean((images[i]-prediction[i])**2)) / max(images[i].max(), 1e-8))})
        result = {"defocus_diopter": model.d.item(), "origin": model.origin.detach().tolist(),
                  "amplitude_center": model.center.detach().tolist(), "amplitude_width": model.log_width.exp().detach().tolist(),
                  "gain": float(gain), "background": float(background), "history": model.history, "rows": metrics}
        result["residual_noll_radians"] = dict(zip(model.residual_modes, model.residual.detach().tolist()))
        with (destination / f"fit_iterations_{name}.pkl").open("wb") as stream:
            pickle.dump(model.iteration_records, stream, protocol=5)
        results["models"][name] = result
        np.savez_compressed(destination / f"prediction_{name}.npz", predicted=prediction, measured=images, camera_xy=camera_xy)
        (destination / "forward_fit.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(name, [(b, float(np.mean([r["corr"] for r in metrics if r["branch"] == b and r["split"] == "test"])))
                     for b in ("U", "C")], flush=True)


if __name__ == "__main__":
    main()
