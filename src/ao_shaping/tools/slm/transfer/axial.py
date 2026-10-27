"""Conditional ABCD input-plane scan on recorded phase/CCD pairs; no devices.

The fitted distance is an effective correction relative to a frozen nominal
calibration. It is not an absolute SLM-to-lens distance measurement.
"""
from __future__ import annotations

import argparse
from loguru import logger
from dataclasses import replace
import hashlib
import json
import pickle
from pathlib import Path
import time

import numpy as np
from scipy.optimize import nnls

from .capture import clock
from .device import resolve_device


def scan(args):
    import torch
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import FrozenDefocusForward, load_frozen_parameters
    torch.set_num_threads(4)
    params = load_frozen_parameters(args.checkpoint).effective_parameters()
    rows = json.loads((args.capture/"manifest.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    phases, images, hashes, exposures = [], [], [], []
    reference_xy, reference_core = None, None
    for row in rows:
        with np.load(args.capture/row["file"]) as saved:
            phase = saved["model_phase"].copy()
            xy = saved["camera_xy"][::args.sample_stride, ::args.sample_stride].copy()
            core = saved["core"][::args.sample_stride, ::args.sample_stride].copy()
            if reference_xy is None:
                reference_xy, reference_core = xy.copy(), core.copy()
            elif not np.array_equal(reference_xy, xy) or not np.array_equal(reference_core, core):
                raise ValueError("all captures require identical camera grid and core mask")
            if row["exposure_ms"] <= 0 or not core.any():
                raise ValueError("positive exposure and nonempty core required")
            sampled_raw = saved["raw_frames"][:, xy[..., 1].astype(int), xy[..., 0].astype(int)]
            if sampled_raw[:, core].max() >= row.get("full_scale", 255)*250/255:
                raise ValueError("saturated core; retain raw data and adjust exposure")
            image = saved["raw_frames"].mean(0)[xy[..., 1].astype(int), xy[..., 0].astype(int)]
        phases.append(phase)
        images.append(image/(row["exposure_ms"]*row.get("counts_scale", 1.)))
        hashes.append(hashlib.sha256(phase.tobytes()).hexdigest())
        exposures.append(row["exposure_ms"])
    unique = list(dict.fromkeys(hashes))
    if len(unique) < 3:
        raise ValueError("at least three distinct physical phases required")
    train_hashes = set(unique[::2])
    train = np.array([h in train_hashes for h in hashes])
    exposure = np.asarray(exposures)
    measured = np.stack(images)
    np.savez_compressed(args.out/"inputs.npz", measured=measured, camera_xy=xy, core=core, train=train, exposure_ms=exposure)
    config = dict(checkpoint=str(args.checkpoint.resolve()), capture=str(args.capture.resolve()),
                  sample_stride=args.sample_stride, offsets_m=args.offsets, phase_hashes=hashes,
                  train=train.tolist(), interpretation="conditional effective distance; no absolute identifiability",
                  selection="training relative L2 only", device=resolve_device(args.device), **clock())
    (args.out/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = []
    for index, offset in enumerate(args.offsets):
        started = clock()
        candidate = replace(params, slm_axial_offset_m=float(offset))
        optical = replace(candidate, gain=1., background=0., bias_counts=0.)
        forward = FrozenDefocusForward(optical, xy, device=config["device"])
        with torch.no_grad():
            prediction = np.stack([forward.predict(phase)[0].cpu().numpy() for phase in phases])
        design = np.stack([prediction[:, core], np.ones_like(prediction[:, core]),
                           np.broadcast_to(1/exposure[:, None], prediction[:, core].shape)], axis=-1)
        coeff, _ = nnls(design[train].reshape(-1, 3), measured[train][:, core].ravel())
        fitted = coeff[0]*prediction+coeff[1]+coeff[2]/exposure[:, None, None]
        observed, predicted = measured[:, core], fitted[:, core]
        p_cv, t_cv = predicted.std(1)/predicted.mean(1), observed.std(1)/observed.mean(1)
        per_frame = [dict(correlation=float(np.corrcoef(p,t)[0,1]), cv_prediction=float(cp),
                          cv_measured=float(ct), cv_abs_error=float(abs(cp-ct)),
                          relative_l2=float(np.linalg.norm(p-t)/np.linalg.norm(t)))
                     for p,t,cp,ct in zip(predicted, observed, p_cv,t_cv)]
        def aggregate(mask, records=per_frame):
            return {key: float(np.mean([r[key] for r,m in zip(records,mask) if m])) for key in records[0]}
        record = dict(offset_m=float(offset), radiometry=coeff.tolist(), train=aggregate(train),
                      validation=aggregate(~train), per_frame=per_frame, **started,
                      end_monotonic_ns=time.monotonic_ns())
        history.append(record)
        np.savez_compressed(args.out/f"offset_{index:03d}.npz", prediction=fitted, optical=prediction)
        (args.out/"scan.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        with (args.out/"iterations.pkl").open("wb") as stream:
            pickle.dump(history, stream, protocol=5)
        logger.info(json.dumps(record | dict(per_frame=len(per_frame))))
    best = min(history, key=lambda r:r["train"]["relative_l2"])
    baseline = next(r for r in history if r["offset_m"] == 0.)
    validation_gain = 1-best["validation"]["relative_l2"]/baseline["validation"]["relative_l2"]
    boundary = best["offset_m"] in (min(args.offsets), max(args.offsets))
    promoted = bool(not boundary and validation_gain > .005 and
                    best["validation"]["cv_abs_error"] < baseline["validation"]["cv_abs_error"])
    for filename, selected in (("candidate_parameters.json", best),
                               ("frozen_parameters.json", best if promoted else baseline)):
        frozen = replace(params, slm_axial_offset_m=selected["offset_m"], gain=selected["radiometry"][0],
                         background=selected["radiometry"][1], bias_counts=selected["radiometry"][2])
        (args.out/filename).write_text(json.dumps(frozen.to_dict(), indent=2), encoding="utf-8")
    (args.out/"selection.json").write_text(json.dumps(dict(candidate=best, baseline=baseline,
        boundary=boundary, validation_relative_l2_improvement=validation_gain, promoted=promoted,
        rule="training selects candidate; validation L2 improvement >0.5%, lower CV error and interior optimum required for promotion",
        interpretation="conditional offset; not absolute SLM distance; validation used for promotion"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-.2,-.15,-.1,-.05,0.,.05,.1,.15,.2])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    options = parser.parse_args()
    if options.sample_stride < 1 or not np.isfinite(options.offsets).all() or 0. not in options.offsets:
        parser.error("positive sample stride and finite offsets including zero required")
    scan(options)
