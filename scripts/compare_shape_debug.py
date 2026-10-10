"""Compare recorded square-shaping CCD frames with one consistent metric set.

Example:
    python scripts/compare_shape_debug.py \
        report/shape_trials/hardware_fractional_baseline_20261010 \
        report/shape_trials/hardware_gs_pad3_refine24_20261010

Only use trusted local pickle files. The center is frozen at each run's flat
frame peak, as it was during optimization; it is never recentered on the best
frame. Scores from different optimizers select the best row but are not compared.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def _pkl_path(path: Path) -> Path:
    if path.is_file() and path.suffix == ".pkl":
        return path
    matches = list(path.glob("debug/**/*.pkl"))
    if len(matches) != 1:
        raise ValueError(f"expected one debug PKL under {path}, found {len(matches)}")
    return matches[0]


def _score_key(row: dict) -> str:
    for key in ("score", "shape", "quality"):
        if key in row:
            return key
    raise ValueError("record has no score, shape, or quality column")


def _metrics(frame: np.ndarray, center: tuple[int, int], side: int, bins: int) -> dict:
    raw = np.asarray(frame, dtype=np.float64)
    height, width = raw.shape
    corner_h, corner_w = max(1, height // 10), max(1, width // 10)
    corners = np.concatenate(
        (
            raw[:corner_h, :corner_w].ravel(),
            raw[:corner_h, -corner_w:].ravel(),
            raw[-corner_h:, :corner_w].ravel(),
            raw[-corner_h:, -corner_w:].ravel(),
        )
    )
    background = float(np.median(corners))
    signal = np.clip(raw - background, 0.0, None)
    cy, cx = center
    y0, x0 = int(round(cy - side / 2)), int(round(cx - side / 2))
    y1, x1 = y0 + side, x0 + side
    if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
        raise ValueError(f"{side}px target at {center} leaves {raw.shape} frame")
    box = signal[y0:y1, x0:x1]
    total, box_total = float(signal.sum()), float(box.sum())
    if total <= 0 or box_total <= 0:
        raise ValueError(
            "frame or target box has no signal after background subtraction"
        )
    tile_energy = np.array(
        [
            part.sum()
            for row in np.array_split(box, bins)
            for part in np.array_split(row, bins, axis=1)
        ],
        dtype=np.float64,
    )
    tile_target = (
        np.array(
            [
                part.size
                for row in np.array_split(box, bins)
                for part in np.array_split(row, bins, axis=1)
            ],
            dtype=np.float64,
        )
        / box.size
    )
    return {
        "background": background,
        "pib": box_total / total,
        "cv": float(box.std() / box.mean()),
        "coverage": float(np.minimum(tile_energy / box_total, tile_target).sum()),
        "below_10pct_mean": float(np.mean(box < 0.1 * box.mean())),
        "below_50pct_mean": float(np.mean(box < 0.5 * box.mean())),
    }


def compare(path: Path, side: int = 100, bins: int = 5) -> dict:
    with _pkl_path(path).open("rb") as stream:
        rows = pickle.load(stream)  # noqa: S301 - trusted local hardware debug artifact
    if not isinstance(rows, dict) or not rows:
        raise ValueError(f"expected a nonempty row dictionary in {path}")
    flat_key = min(rows)
    score_key = _score_key(rows[flat_key])
    best_key = max(rows, key=lambda key: float(rows[key].get(score_key, -np.inf)))
    flat = np.asarray(rows[flat_key]["_img"])
    center = tuple(int(v) for v in np.unravel_index(np.nanargmax(flat), flat.shape))
    return {
        "run": str(path),
        "frame_shape": tuple(flat.shape),
        "center": center,
        "score_key": score_key,
        "best_row": best_key,
        "flat": _metrics(flat, center, side, bins),
        "best": _metrics(rows[best_key]["_img"], center, side, bins),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", nargs="+", type=Path, help="run directories or debug PKL files"
    )
    parser.add_argument(
        "--side", type=int, default=100, help="fixed target side in camera pixels"
    )
    parser.add_argument(
        "--bins", type=int, default=5, help="tiles along each target side"
    )
    args = parser.parse_args()
    if args.side < 1 or args.bins < 1:
        parser.error("--side and --bins must be positive")
    print(
        "run\tframe\tcenter_yx\tbest_row\tstage\tbackground\tpib\tcv\tcoverage\tbelow_10pct_mean\tbelow_50pct_mean"
    )
    for run in args.runs:
        result = compare(run, side=args.side, bins=args.bins)
        for stage in ("flat", "best"):
            values = result[stage]
            print(
                f"{run.name}\t{result['frame_shape']}\t{result['center']}\t"
                f"{result['best_row']}\t{stage}\t"
                + "\t".join(
                    f"{values[key]:.4f}"
                    for key in (
                        "background",
                        "pib",
                        "cv",
                        "coverage",
                        "below_10pct_mean",
                        "below_50pct_mean",
                    )
                )
            )


if __name__ == "__main__":
    main()
