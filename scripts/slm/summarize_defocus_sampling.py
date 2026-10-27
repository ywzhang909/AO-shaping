"""Render sampling/radiometry audit metrics without modifying raw evidence."""
from __future__ import annotations

import argparse
from loguru import logger
import csv
import json
from pathlib import Path

import numpy as np


def summarize(rows):
    result = {}
    for split in ("train", "test", "external"):
        for branch in ("U", "C"):
            a = [r for r in rows if r["family"] != "reference" and r["branch"] == branch
                 and r["split"] == split]
            if not a:
                continue
            corr = np.array([r["corr"] for r in a])
            cv = np.array([abs(r["measured_cv"]-r["predicted_cv"]) for r in a])
            ee = np.array([abs(r["measured_ee_roi"]-r["predicted_ee_roi"]) for r in a])
            center = np.array([r["center_error_px"] for r in a])
            width = np.array([r["rms_relative_error"] for r in a])
            result[split+"_"+branch] = dict(count=len(a), mean_corr=float(corr.mean()),
                                            median_corr=float(np.median(corr)), min_corr=float(corr.min()),
                                            cv_mae=float(cv.mean()), ee_mae=float(ee.mean()),
                                            center_mae_px=float(center.mean()), width_relative_mae=float(width.mean()),
                                            passed_cv_gate=bool(cv.mean() <= .05),
                                            passed_full_gate=bool(cv.mean() <= .05 and ee.mean() <= .05
                                                                 and center.max() <= 1 and width.max() <= .1
                                                                 and np.median(corr) >= .85 and corr.min() >= .65))
    return result


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling", required=True, type=Path)
    parser.add_argument("--radiometry", required=True, type=Path)
    parser.add_argument("--refit", type=Path)
    parser.add_argument("--repeat", type=Path,
                        help="Fixed phase/exposure repeated raw frames for measurement noise")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {}
    for group, file in (("sampling", args.sampling/"summary.json"),
                        ("radiometry", args.radiometry/"summary.json")):
        data = json.loads(file.read_text())
        result[group] = {}
        for key, stage in data.items():
            if "rows" not in stage:
                result[group][key] = stage
                continue
            result[group][key] = dict(metrics=summarize(stage["rows"]),
                                      **{k: v for k, v in stage.items() if k not in ("rows", "started", "ended")})
    if args.refit:
        data = json.loads((args.refit/"result.json").read_text())
        result["refit"] = dict(metrics=summarize(data["rows"]), parameters=data["parameters"], training_score=data["training_score"])
    if args.repeat:
        source = json.loads((args.sampling/"config.json").read_text())["directory"]
        refs = {r["branch"]: r for r in json.loads((Path(source)/"manifest.json").read_text()) if r["label"] == "reference_start"}
        with np.load(args.radiometry/"response_fit.npz") as data:
            xy, intercept = data["camera_xy"], data["intercept"]
        repeated = []
        for row in json.loads((args.repeat/"manifest.json").read_text()):
            with np.load(args.repeat/row["file"]) as data:
                raw = data["raw_frames"]
            center = [refs[row["branch"]]["spot"][f"centroid_{k}"] for k in ("x", "y")]
            mask = np.all(np.abs(xy-center) <= 30, axis=-1)
            for index, frame in enumerate([*raw, raw.mean(0)]):
                frame = frame[xy[..., 1].astype(int), xy[..., 0].astype(int)]
                m = dict(branch=row["branch"], file=row["file"], frame_index=index,
                         averaged=index == len(raw), settled=row["settled"]["settled"],
                         timestamp_utc=row["timestamp_utc"], exposure_ms=row["exposure_ms"])
                for name, signal in (("raw", frame), ("corrected", frame-intercept)):
                    values = signal[mask]
                    m[name+"_cv"] = float(values.std()/max(values.mean(), 1e-9))
                    weight = np.maximum(signal, 0)
                    center_xy = (weight[..., None]*xy).sum((0, 1))/max(weight.sum(), 1e-9)
                    m[name+"_center"] = center_xy.tolist()
                repeated.append(m)
        result["fixed_phase_repeats"] = {}
        for branch in ("U", "C"):
            a = [r for r in repeated if r["branch"] == branch and r["averaged"] and r["settled"]]
            result["fixed_phase_repeats"][branch] = dict(count=len(a))
            for name in ("raw", "corrected"):
                result["fixed_phase_repeats"][branch][name] = dict(
                    cv_values=[r[name+"_cv"] for r in a],
                    cv_mean=float(np.mean([r[name+"_cv"] for r in a])),
                    cv_std=float(np.std([r[name+"_cv"] for r in a], ddof=1)) if len(a)>1 else None,
                    center_values=[r[name+"_center"] for r in a])
    artifacts, figures = args.output/"artifacts", args.output/"figures"
    artifacts.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (artifacts/"sampling_radiometry_summary_20261022_r2.json").write_text(json.dumps(result, indent=2))
    if args.repeat:
        (artifacts/"fixed_phase_repeat_metrics_20261022_r2.json").write_text(json.dumps(repeated, indent=2))
    flat = []
    for group in ("sampling", "radiometry"):
        for key, stage in result[group].items():
            for subset, values in stage.get("metrics", {}).items():
                flat.append(dict(group=group, stage=key, subset=subset, **values))
    if args.refit:
        flat += [dict(group="refit", stage="joint_cv", subset=s, **v) for s, v in result["refit"]["metrics"].items()]
    with (artifacts/"sampling_radiometry_metrics_20261022_r2.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sampling = [(k, v) for k, v in result["sampling"].items()
                if "metrics" in v and not k.startswith("blur_")]
    for branch in ("U", "C"):
        axes[0].plot(range(len(sampling)), [v["metrics"]["external_"+branch]["cv_mae"] for _, v in sampling], "o-", label=branch)
    axes[0].set_xticks(range(len(sampling)), [k for k, _ in sampling], rotation=45, ha="right")
    axes[0].set_ylabel("CV absolute prediction error")
    axes[0].axhline(.05, color="gray", linestyle="--")
    axes[0].legend()
    for branch in ("U", "C"):
        a = [(v["sigma_ccd_px"], v["metrics"]["external_"+branch]["cv_mae"]) for k, v in result["radiometry"].items() if k.startswith("single_")]
        axes[1].plot(*zip(*a), "o-", label=branch)
    axes[1].set_xlabel("Effective blur sigma (CCD pixels)")
    axes[1].set_ylabel("CV absolute prediction error")
    axes[1].axhline(.05, color="gray", linestyle="--")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures/"sampling_radiometry_20261022_r2.png", dpi=160)
    plt.close(fig)
    logger.info(json.dumps(result.get("refit", result["radiometry"]["selection"]), indent=2))


if __name__ == "__main__":
    main()
