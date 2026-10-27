"""Create a nominal physical checkpoint from a fresh signed-ramp capture.

No device imports. This is an initialization, not a calibrated model. Fit it on
phase-diverse measurements before using its gradients for physical control.
"""
from __future__ import annotations
from loguru import logger

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


from ao_shaping.tools.slm.transfer.fit import geometry, spot, valid


def bootstrap(dataset: Path, *, center=(960., 600.), width=(400., 300.), defocus=0.):
    """Use only usable current-path calibration rows, never validation images."""
    rows = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in rows if valid(row)]
    matrix, samples = geometry(rows)
    reference = next(row for row in rows if row["label"] == "reference_start")
    capture = json.loads((dataset / "config.json").read_text(encoding="utf-8"))
    grid = int(capture["grid"])
    if grid < 8 or 1920 % grid or 1200 % (1920 // grid):
        raise ValueError("capture grid must exactly tile the 1200x1920 panel")
    values = np.asarray([*center, *width, defocus], dtype=float)
    if not np.isfinite(values).all() or min(width) <= 0:
        raise ValueError("finite parameters and positive amplitude widths required")
    params = dict(defocus_diopter=float(defocus), origin=spot(reference).tolist(),
                  amplitude_center=list(center), amplitude_width=list(width),
                  residual_noll_radians={str(j): 0. for j in range(5, 12)},
                  gain=1., background=0., sigma_ccd_px=0.,
                  ramp_matrix=matrix.tolist(), grid_size=grid,
                  fft_size=max(1024, grid), pixel_samples=3)
    provenance = dict(dataset=str(dataset.resolve()), ramp_samples=samples,
                      manifest_sha256=hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
                      status="nominal initialization; not fitted or physically validated",
                      assumption="Santec 1200x1920, 8um pitch, 1064nm; Gaussian input amplitude")
    return params, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--center", type=float, nargs=2, default=(960., 600.))
    parser.add_argument("--width", type=float, nargs=2, default=(400., 300.))
    parser.add_argument("--defocus", type=float, default=0.)
    args = parser.parse_args()
    params, provenance = bootstrap(args.dataset, center=args.center, width=args.width, defocus=args.defocus)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "frozen_parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    logger.info(json.dumps(dict(checkpoint=str(args.out / "frozen_parameters.json"), **provenance), indent=2))


if __name__ == "__main__":
    main()
