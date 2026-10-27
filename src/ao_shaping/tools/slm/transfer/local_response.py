"""Calibrate local Jacobian gain/blur from recorded signed CCD phase probes."""
from __future__ import annotations
from loguru import logger

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np


def fit_local_response(run, out, *, shape="square", sigmas=(0., 1., 2., 4., 8.)):
    """Choose on one local response pair; subsequent CCD data must validate it.

    This uses differential images, so a static additive background cancels.
    Drift and camera nonlinearity can still contaminate this pair. Gain is
    constrained positive; a reversed response cannot be fixed by flipping it.
    """
    import torch
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward, load_frozen_parameters,
    )
    from .artifacts import resolve_recorded_path
    torch.set_num_threads(4)
    run, out = Path(run), Path(out)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    rows = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    params = load_frozen_parameters(resolve_recorded_path(config["checkpoint"], relative_to=run))
    def read(label):
        row = next(r for r in rows if r["label"] == label)
        with np.load(run / row["file"]) as payload:
            xy = payload["camera_xy"].copy()
            image = payload["raw_frames"].mean(0)[xy[..., 1].astype(int), xy[..., 0].astype(int)] / (row["exposure_ms"]*row.get("counts_scale", 1.))
            return image, payload["model_phase"].copy(), xy
    plus1, plus_phase, xy = read(shape+"_0_B1")
    plus2, _, _ = read(shape+"_0_B2")
    minus, minus_phase, _ = read(shape+"_direction_minus")
    measured = (plus1+plus2)*.5-minus
    candidates = []
    for sigma in sigmas:
        forward = FrozenDefocusForward(replace(params, sigma_ccd_px=float(sigma)), xy, device="cpu")
        forward.exposure_ms = next(r["exposure_ms"] for r in rows if r["label"] == shape+"_0_B1")
        with torch.no_grad():
            predicted = (forward.predict(plus_phase)-forward.predict(minus_phase))[0].numpy()
        scale = max(0., float(np.sum(predicted*measured)/max(np.sum(predicted**2), 1e-12)))
        error = float(np.linalg.norm(scale*predicted-measured)/max(np.linalg.norm(measured), 1e-12))
        corr = float(np.corrcoef(predicted.ravel(), measured.ravel())[0, 1])
        if not np.isfinite([scale, error, corr]).all() or scale <= 0:
            continue
        candidates.append(dict(sigma_ccd_px=float(sigma), gain_ratio=scale,
                               correlation=corr, relative_l2=error))
    if not candidates:
        raise ValueError("no finite positive local response; retain the previous model")
    best = min(candidates, key=lambda row: row["relative_l2"])
    calibrated = replace(params, gain=params.gain*best["gain_ratio"], sigma_ccd_px=best["sigma_ccd_px"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "frozen_parameters.json").write_text(json.dumps(dict(**calibrated.to_dict(),
        local_response_only=True, source_model=config["checkpoint"]), indent=2), encoding="utf-8")
    result = dict(source=str(run.resolve()), shape=shape, selected=best, candidates=candidates,
                  selection="fit on one signed local probe, not independent validation",
                  assumption="shared positive response gain and shift-invariant residual PSF; static offset cancels")
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(out / "signed_response.npz", measured=measured, plus_phase=plus_phase,
                        minus_phase=minus_phase, camera_xy=xy)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shape", default="square")
    args = parser.parse_args()
    logger.info(json.dumps(fit_local_response(args.run, args.out, shape=args.shape), indent=2))


if __name__ == "__main__":
    main()
