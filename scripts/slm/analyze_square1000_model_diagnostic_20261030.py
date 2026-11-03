"""Analyze the measured null/ramp/defocus/exposure diagnostic session."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import shift, uniform_filter


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.vdot(a, b).real / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-20))


def affine_fit(reference: np.ndarray, measured: np.ndarray) -> dict:
    gain, offset = np.polyfit(reference.ravel(), measured.ravel(), 1)
    residual = measured - (gain * reference + offset)
    return dict(gain=float(gain), offset=float(offset),
                relative_l2=float(np.linalg.norm(residual) / np.linalg.norm(measured)),
                correlation=float(np.corrcoef(reference.ravel(), measured.ravel())[0, 1]))


def binned_rate(raw: np.ndarray, xy: np.ndarray, exposure_ms: float,
                bin_size: int) -> np.ndarray:
    """Sample the native frame as ``bin_size``-pixel square blocks.

    A one-pixel sample aliases the speckle field: the model prediction already
    integrates subpixel offsets and a blaze kernel, while a nearest-pixel
    measurement does not.  Averaging the same block the analyzer uses for the
    prediction makes the two quantities comparable without changing the
    recorded frames.
    """
    frame = raw.mean(axis=0)
    if bin_size == 1:
        return frame[xy[..., 1].astype(int), xy[..., 0].astype(int)] / exposure_ms
    half = bin_size // 2
    offsets = np.arange(-half, bin_size - half)
    y = xy[..., 1].astype(int)[:, :, None, None] + offsets[None, None, None, :]
    x = xy[..., 0].astype(int)[:, :, None, None] + offsets[None, None, :, None]
    block = raw[:, y, x].mean(axis=(3, 4))
    return block.mean(axis=0) / exposure_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--bin-size", type=int, default=1,
                        help="square CCD block (px) used for measurement and model")
    args = parser.parse_args()
    if args.bin_size < 1:
        parser.error("bin-size must be a positive integer")
    manifest = json.loads((args.session / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((args.session / "config.json").read_text(encoding="utf-8"))
    means, rates, xy = {}, {}, None
    saturation = []
    for row in manifest:
        with np.load(args.session / row["file"]) as saved:
            raw = saved["raw_frames"].astype(np.float64)
            xy = saved["camera_xy"]
            frame = raw.mean(axis=0)
            means[row["index"]] = frame
            rates[row["index"]] = binned_rate(raw, xy, row["exposure_ms"], args.bin_size)
            saturation.append(dict(index=row["index"], maximum=float(raw.max()),
                                   saturated_pixels=int((raw >= .98 * 4095).sum()),
                                   repeat_std=float(raw.std(axis=0).mean())))
    with np.load(args.session / "predictions.npz") as saved:
        predictions = {key: saved[key] for key in saved.files}
    if args.bin_size > 1:
        predictions = {key: uniform_filter(value.astype(np.float64),
                                           size=args.bin_size, mode="nearest")
                       for key, value in predictions.items()}

    anchors = [0, 1, 4, 7, 10, 13]
    drift = {}
    for i in range(len(manifest)):
        left = max(a for a in anchors if a <= i)
        right = min(a for a in anchors if a >= i)
        weight = 0. if left == right else (i - left) / (right - left)
        drift[i] = (1 - weight) * rates[left] + weight * rates[right]
    nulls = [dict(pair=[a, b], seconds=(manifest[b]["monotonic_ns"] - manifest[a]["monotonic_ns"]) / 1e9,
                  cosine=cosine(rates[b] - rates[a], rates[b] - rates[a]),
                  difference_rms=float(np.sqrt(np.mean((rates[b] - rates[a]) ** 2))),
                  mean_ratio=float(rates[b].sum() / rates[a].sum()))
             for a, b in zip(anchors[:-1], anchors[1:])]

    groups = {"N4": (2, 3, 4), "RX": (5, 6, 7), "RY": (8, 9, 10)}
    responses = {}
    for label, (plus, minus, anchor) in groups.items():
        measured = .5 * (rates[plus] - rates[minus])
        drift_corrected = .5 * ((rates[plus] - drift[plus]) - (rates[minus] - drift[minus]))
        predicted = .5 * (predictions[label + "+"] - predictions[label + "-"])
        # Test sensor coordinate conventions without changing the measured data.
        transforms = {"identity": predicted, "transpose": predicted.T,
                      "flip_y": predicted[::-1], "flip_x": predicted[:, ::-1],
                      "rotate_180": predicted[::-1, ::-1],
                      "transpose_flip_y": predicted.T[::-1],
                      "transpose_flip_x": predicted.T[:, ::-1],
                      "transpose_rotate_180": predicted.T[::-1, ::-1]}
        alignments = {name: cosine(value, drift_corrected) for name, value in transforms.items()}
        best_shift = (0, 0)
        best_cosine = alignments["identity"]
        for dy in range(-20, 21, 2):
            for dx in range(-20, 21, 2):
                value = cosine(shift(predicted, (dy, dx), order=1, mode="nearest"), drift_corrected)
                if value > best_cosine:
                    best_cosine, best_shift = value, (dy, dx)
        even = .5 * (rates[plus] + rates[minus]) - .5 * (rates[plus - 1] + rates[anchor])
        predicted_even = .5 * (predictions[label + "+"] + predictions[label + "-"]) - predictions["A"]
        responses[label] = dict(
            measured_rms=float(np.sqrt(np.mean(measured ** 2))),
            drift_corrected_rms=float(np.sqrt(np.mean(drift_corrected ** 2))),
            predicted_rms=float(np.sqrt(np.mean(predicted ** 2))),
            cosine=cosine(predicted, measured), drift_corrected_cosine=cosine(predicted, drift_corrected),
            best_transform=max(alignments, key=alignments.get), transform_cosines=alignments,
            best_shift_yx=best_shift, best_shift_cosine=float(best_cosine),
            even_rms=float(np.sqrt(np.mean(even ** 2))),
            even_over_odd_rms=float(np.sqrt(np.mean(even ** 2)) / max(np.sqrt(np.mean(drift_corrected ** 2)), 1e-20)),
            predicted_even_over_odd_rms=float(np.sqrt(np.mean(predicted_even ** 2)) / max(np.sqrt(np.mean(predicted ** 2)), 1e-20)),
            odd_affine=affine_fit(predicted, drift_corrected),
        )
    exposure_rates = {str(row["exposure_ms"]): rates[row["index"]] for row in manifest if row["index"] in (11, 12, 13)}
    # Shared optical gain, irradiance background and electronic bias in counts.
    exposure_rows = [manifest[i] for i in (11, 12, 13)]
    exposure = np.array([row["exposure_ms"] for row in exposure_rows])
    measured_counts = np.stack([
        rates[row["index"]] * row["exposure_ms"] for row in exposure_rows
    ])
    optical = np.stack([predictions["A"]] * 3)
    design = np.stack([
        exposure[:, None, None] * optical,
        np.broadcast_to(exposure[:, None, None], measured_counts.shape),
        np.ones_like(measured_counts),
    ], -1)
    coef = np.linalg.lstsq(design.reshape(-1, 3), measured_counts.ravel(), rcond=None)[0]
    residual = measured_counts - design @ coef
    report = dict(session=str(args.session), bin_size=args.bin_size, config=config,
                  saturation=saturation,
                  anchor_absolute=affine_fit(predictions["A"], rates[13]), nulls=nulls,
                  responses=responses,
                  radiometry=dict(gain=float(coef[0]), background=float(coef[1]), bias_counts=float(coef[2]),
                                  relative_l2=float(np.linalg.norm(residual) / np.linalg.norm(measured_counts)),
                                  mean_rate_by_exposure={key: float(value.mean()) for key, value in exposure_rates.items()}))
    output = args.session / "diagnostic_analysis.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
