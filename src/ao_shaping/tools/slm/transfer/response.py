"""Fit and independently check illuminated exposure response on a fixed CCD grid."""

from __future__ import annotations

import argparse
from loguru import logger
import json
from pathlib import Path

import numpy as np

from ao_shaping.utils.image.exposure_response import response_fits


def analyze(source: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    rows = json.loads((source / "manifest.json").read_text())
    results = {}
    for phase in dict.fromkeys(r["phase"] for r in rows):
        selected = [r for r in rows if r["phase"] == phase]
        with np.load(source / f"{phase}_probe.npz") as probe:
            xy, target = probe["camera_xy"], probe["target"]
        x, y = xy[..., 0].astype(int), xy[..., 1].astype(int)
        images, per_frame = [], []
        for row in selected:
            with np.load(source / row["file"]) as payload:
                raw = payload["raw_frames"][:, y, x].astype(float)
                per_frame.append(raw)
                images.append(raw.mean(0))
        images = np.stack(images)
        exposures = np.array([r["exposure_ms"] for r in selected])
        if phase == "flat":
            peak = np.unravel_index(images[0].argmax(), images[0].shape)
            yy, xx = np.indices(images[0].shape)
            mask = (xx - peak[1]) ** 2 + (yy - peak[0]) ** 2 <= 15**2
        else:
            mask = target > 0.5
        train, seen = [], set()
        for exposure in exposures:
            train.append(exposure not in seen)
            seen.add(exposure)
        train = np.array(train)
        design = np.stack((exposures, np.ones_like(exposures)), axis=-1)
        slope, intercept = np.linalg.lstsq(
            design[train], images[train].reshape(train.sum(), -1), rcond=None
        )[0].reshape(2, *images.shape[1:])
        predicted = exposures[:, None, None] * slope + intercept
        corrected = (images - intercept) / exposures[:, None, None]
        signals = images[:, mask].mean(1)
        reference_mask = np.isclose(exposures, exposures[0])
        reference_signals = signals[reference_mask]
        frame_metrics = [
            dict(
                index=r["index"],
                exposure_ms=float(t),
                roi_mean=float(im[mask].mean()),
                roi_cv=float(im[mask].std() / max(im[mask].mean(), 1e-9)),
                roi_rate_mean=float((im / t)[mask].mean()),
                corrected_cv=float(c[mask].std() / max(c[mask].mean(), 1e-9)),
                frame_roi_means=[float(f[mask].mean()) for f in fs],
            )
            for r, t, im, c, fs in zip(
                selected, exposures, images, corrected, per_frame
            )
        ]
        result = dict(
            levels_ms=np.unique(exposures).tolist(),
            train_indices=np.flatnonzero(train).tolist(),
            validation_indices=np.flatnonzero(~train).tolist(),
            affine_intercept_roi_percentiles=np.percentile(
                intercept[mask], [5, 50, 95]
            ).tolist(),
            validation_rmse_counts=float(
                np.sqrt(((predicted[~train] - images[~train])[:, mask] ** 2).mean())
            ),
            reference_roi_mean_span_fraction=float(
                np.ptp(reference_signals) / reference_signals.mean()
            ),
            response_candidates=response_fits(exposures, signals),
            rows=frame_metrics,
            qualification="illuminated intercept; raw signed fit; no dark-frame or phase-transfer validation",
        )
        np.savez_compressed(
            out / f"{phase}_arrays.npz",
            exposures_ms=exposures,
            camera_xy=xy,
            images=images,
            train_mask=train,
            slope=slope,
            intercept=intercept,
            predicted=predicted,
            corrected_rate=corrected,
            roi_mask=mask,
        )
        results[phase] = result
        logger.info(phasejson.dumps({k: v for k, v in result.items() if k not in {'rows', 'response_candidates'}}))
    (out / "result.json").write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.out)


if __name__ == "__main__":
    main()
