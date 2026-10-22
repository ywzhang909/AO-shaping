"""Refit a compact physical predictor with explicit intensity and CV supervision.

Uses only original training phases for updates and checkpoint selection. Each
iteration retains predictions and all parameters; original raw data stay intact.
The optional effective Gaussian blur is a discrepancy term, not a measured PSF.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.slm.diagnose_defocus_sampling import (  # noqa: E402
    fixed_panel_basis, metrics, response_bias, restore_model, stamp,
)


def supervised_loss(predicted, measured, masks, cv_weight=1.):
    """Image NMSE plus fixed-mask CV error, retaining absolute-intensity evidence."""
    import torch

    nmse = ((predicted-measured)**2).mean((1, 2))/(measured**2).mean((1, 2)).clamp_min(1e-8)
    cvs = []
    for array in (predicted, measured):
        count = masks.sum((1, 2))
        mean = (array*masks).sum((1, 2))/count
        variance = ((array-mean[:, None, None])**2*masks).sum((1, 2))/count
        cvs.append(torch.sqrt(variance.clamp_min(1e-12))/mean.clamp_min(1e-6))
    cv_error = ((cvs[0]-cvs[1])**2).mean()
    return nmse.mean()+cv_weight*cv_error, nmse.mean(), cv_error


def main():
    import torch
    from loguru import logger

    logger.remove()
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-source", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--cv-weight", type=float, default=1.)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.sampling_source/"config.json").read_text())
    saved = json.loads(Path(config["fit"]).read_text())
    original = saved["models"]["M4_residual"]
    rows = json.loads((args.sampling_source/"records.json").read_text())
    source_rows = json.loads((Path(config["directory"])/"manifest.json").read_text())
    refs = {r["branch"]: r for r in source_rows if r["label"] == "reference_start"}
    with np.load(args.sampling_source/"inputs_grid_128.npz") as data:
        phases, raw_rate, xy, train, exposures = [data[k] for k in
                                               ("phases", "measured", "camera_xy", "train_mask", "exposure_ms")]
    bias = response_bias(args.response, xy)
    measured = raw_rate-bias["intercept"]/exposures[:, None, None]
    masks = np.stack([np.all(np.abs(xy-np.array([refs[r["branch"]]["spot"][f"centroid_{k}"] for k in ("x", "y")])) <= 30, axis=-1) for r in rows])
    np.savez_compressed(out/"inputs.npz", phases=phases, raw_rate=raw_rate, measured=measured,
                        masks=masks, exposure_ms=exposures, train_mask=train, camera_xy=xy,
                        bias=bias["intercept"])
    (out/"config.json").write_text(json.dumps(dict(source=str(args.sampling_source.resolve()),
                                                  response=str(args.response.resolve()),
                                                  iterations=args.iterations, cv_weight=args.cv_weight,
                                                  started=stamp()), indent=2))
    for source in (Path(__file__), ROOT/"scripts/slm/fit_defocus_forward_dataset.py",
                   ROOT/"scripts/slm/diagnose_defocus_sampling.py"):
        (out/source.name).write_bytes(source.read_bytes())
    (out/"records.json").write_text(json.dumps(rows, indent=2))
    model = restore_model(phases[train], measured[train], xy, saved, original, fft_size=1024,
                          pixels=3, physical_basis=fixed_panel_basis(128))
    model.center.requires_grad_(True)
    model.log_width.requires_grad_(True)
    gain_log = torch.nn.Parameter(torch.tensor(float(np.log(original["gain"]))))
    background_log = torch.nn.Parameter(torch.tensor(float(np.log(.02))))
    sigma_log = torch.nn.Parameter(torch.tensor(float(np.log(2.))))
    optimizer = torch.optim.Adam([
        {"params": [model.d], "lr": .0003}, {"params": [model.origin], "lr": .15},
        {"params": [model.center], "lr": .25}, {"params": [model.log_width], "lr": .002},
        {"params": [model.residual], "lr": .015},
        {"params": [gain_log, background_log, sigma_log], "lr": .01}])
    target = torch.tensor(measured[train], dtype=torch.float32)
    target_masks = torch.tensor(masks[train], dtype=torch.float32)
    kernel_grid = torch.arange(-12, 13, dtype=torch.float32)

    def predict():
        optical = model.prediction()
        kernel = torch.exp(-.5*(kernel_grid/(sigma_log.exp()/2))**2)
        kernel = kernel/kernel.sum()
        blurred = torch.nn.functional.conv2d(optical[:, None],
                                             (kernel[:, None]*kernel[None, :])[None, None], padding=12)[:, 0]
        return blurred*gain_log.exp()+background_log.exp()

    def parameters():
        return dict(defocus_diopter=model.d.item(), origin=model.origin.detach().tolist(),
                    amplitude_center=model.center.detach().tolist(), amplitude_width=model.log_width.exp().detach().tolist(),
                    residual_noll_radians=dict(zip(model.residual_modes, model.residual.detach().tolist())),
                    gain=gain_log.exp().item(), background=background_log.exp().item(),
                    sigma_ccd_px=sigma_log.exp().item())

    history, best_score, best_state = [], float("inf"), None
    learnables = [model.d, model.origin, model.center, model.log_width, model.residual,
                  gain_log, background_log, sigma_log]
    for iteration in range(args.iterations+1):
        started = stamp()
        optimizer.zero_grad()
        prediction = predict()
        loss, nmse, cv_error = supervised_loss(prediction, target, target_masks, args.cv_weight)
        score = float(loss.detach())
        record = dict(iteration=iteration, loss=score, nmse=float(nmse.detach()),
                      cv_mse=float(cv_error.detach()), parameters=parameters(), started=started, ended=stamp())
        np.savez_compressed(out/f"iteration_{iteration:04d}.npz", predicted=prediction.detach().numpy(),
                            parameters_json=json.dumps(record["parameters"]))
        history.append(record)
        with (out/"iterations.pkl").open("wb") as stream:
            pickle.dump(history, stream, protocol=5)
        if score < best_score:
            best_score = score
            best_state = [p.detach().clone() for p in learnables]
        if iteration % 25 == 0:
            print(iteration, score, record["nmse"], record["cv_mse"], record["parameters"]["sigma_ccd_px"], flush=True)
        if iteration == args.iterations:
            break
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.d.clamp_(-2, 2)
            model.center.clamp_(0, 1920)
            model.log_width.clamp_(np.log(80), np.log(800))
            sigma_log.clamp_(np.log(.25), np.log(8.))
            background_log.clamp_(np.log(1e-5), np.log(1.))
    with torch.no_grad():
        for p, state in zip(learnables, best_state):
            p.copy_(state)
        model.phases = torch.tensor(phases, dtype=torch.float32)
        predicted = predict().numpy()
    result = dict(parameters=parameters(), training_score=best_score,
                  rows=metrics(rows, measured, predicted, xy, refs), finished=stamp())
    np.savez_compressed(out/"prediction.npz", predicted=predicted, measured=measured, camera_xy=xy)
    (out/"result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
