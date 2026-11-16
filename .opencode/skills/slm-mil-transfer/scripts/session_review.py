"""Offline MIL/transfer record review; never opens devices or reads pickle.

Metadata mode uses only the standard library. --check-raw needs numpy and the
AO-shaping package; --audit delegates to its canonical transfer audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def record_path(run, filename):
    root = Path(run).resolve()
    path = (root/filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"record escapes run directory: {filename}")
    return path


def finite_positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def raw_details(run, row):
    import numpy as np
    from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective, image_metrics

    with np.load(record_path(run, row["file"]), allow_pickle=False) as saved:
        required = {"raw_frames", "model_phase", "camera_xy", "target", "core"}
        if not required.issubset(saved.files):
            raise ValueError(f"raw schema missing {sorted(required-set(saved.files))}: {row['file']}")
        raw, phase, xy, target, core = (saved[k] for k in ("raw_frames", "model_phase", "camera_xy", "target", "core"))
        if raw.ndim != 3 or not len(raw) or xy.shape != (*target.shape, 2) or core.shape != target.shape:
            raise ValueError(f"invalid raw frame/grid shapes: {row['file']}")
        if core.dtype != np.bool_ or not core.any() or phase.ndim != 2:
            raise ValueError(f"nonempty boolean core and 2D phase required: {row['file']}")
        if not all(np.isfinite(a).all() for a in (raw, phase, xy, target)):
            raise ValueError(f"nonfinite raw arrays: {row['file']}")
        if not np.equal(xy, np.round(xy)).all():
            raise ValueError("fractional camera sampling requires its original interpolation implementation")
        x, y = xy[...,0].astype(int), xy[...,1].astype(int)
        if min(x.min(), y.min()) < 0 or x.max() >= raw.shape[2] or y.max() >= raw.shape[1]:
            raise ValueError(f"camera grid outside raw frame: {row['file']}")
        objective = ShapeObjective(target)
        objective.core = core
        metrics = image_metrics(raw.mean(0)[y,x], objective)
        stored_cv = row.get("metrics", {}).get("cv")
        if stored_cv is not None and not np.isclose(metrics["cv"], stored_cv, rtol=1e-6, atol=1e-6):
            raise ValueError(f"raw/manifest sampled CV mismatch: {row['file']}")
        full_scale = row.get("full_scale")
        if full_scale is None and isinstance(row.get("bit_depth"), int):
            full_scale = 2**row["bit_depth"]-1
        clipped = None
        if full_scale is not None:
            if not finite_positive(full_scale):
                raise ValueError("invalid full_scale")
            clipped = int((raw[:,y,x][:,core] >= full_scale*250/255).sum())
        digest = hashlib.sha256()
        for array in (xy,target,core):
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        return dict(metrics=metrics, metric_scope="saved sampled core on mean of raw frames",
                    raw_frames=len(raw), saturated_core_samples=clipped,
                    saturation_rule=">=250/255 of explicit full scale; null means unknown scale",
                    phase_sha256=hashlib.sha256(phase.tobytes()).hexdigest(),
                    grid_core_sha256=digest.hexdigest())


def load_records(run):
    rows = read_json(Path(run)/"manifest.json")
    if not isinstance(rows,list) or not rows:
        raise ValueError("manifest must be a nonempty JSON list")
    for row in rows:
        if not finite_positive(row.get("exposure_ms")):
            raise ValueError(f"finite positive exposure_ms required: {row.get('label')}")
        record_path(run,row["file"])
    return rows


def review(run, *, check_raw=False, audit=False):
    run = Path(run)
    rows = load_records(run)
    findings, captures, phase_splits = [], [], {}
    labels = Counter(r.get("label") for r in rows)
    if any(count > 1 for count in labels.values()):
        findings.append("Repeated labels exist; comparisons must disambiguate by exposure.")
    for row in rows:
        path = record_path(run,row["file"])
        if not path.is_file():
            findings.append(f"Missing raw record: {row['file']}")
            if check_raw:
                raise ValueError(f"missing raw record: {row['file']}")
        record = {key:row.get(key) for key in ("label","file","exposure_ms","bit_depth","full_scale","split","timestamp_utc")}
        record["metrics"] = row.get("metrics",{})
        clocks = row.get("frames",[])
        if not isinstance(clocks,list):
            raise ValueError("manifest frames must contain per-frame clock records")
        if not clocks:
            findings.append(f"No per-frame timestamps: {row.get('label')}")
        for clock in clocks:
            if clock.get("exposure_ms") != row["exposure_ms"]:
                findings.append(f"Frame exposure differs from capture: {row.get('label')}")
            if not clock.get("timestamp_utc") or clock.get("monotonic_ns") is None or clock.get("end_monotonic_ns") is None:
                findings.append(f"Incomplete host timestamps: {row.get('label')}")
            elif clock["end_monotonic_ns"] < clock["monotonic_ns"]:
                findings.append(f"Reversed host clock interval: {row.get('label')}")
        record["recorded_frame_clocks"] = len(clocks)
        if check_raw:
            record["raw_check"] = raw_details(run,row)
            if clocks and record["raw_check"]["raw_frames"] != len(clocks):
                findings.append(f"Raw frame count differs from timestamps: {row.get('label')}")
            if record["raw_check"]["saturated_core_samples"]:
                findings.append(f"Core saturation: {row.get('label')}")
            split = row.get("split")
            if split in {"train","test","validation"}:
                phase_splits.setdefault(record["raw_check"]["phase_sha256"],set()).add(split)
        captures.append(record)
    leakage = {key:sorted(value) for key,value in phase_splits.items() if len(value)>1}
    if leakage:
        findings.append("Identical phase arrays occur in multiple data splits; independent phase generalization is not established.")
    result = dict(run=str(run.resolve()), scope="offline records, not hardware health or release verification",
                  captures=len(rows), exposure_ms=sorted({r["exposure_ms"] for r in rows}),
                  findings=sorted(set(findings)), phase_split_overlap=leakage,
                  raw_checked=check_raw, records=captures)
    summary_path = run/"summary.json"
    if summary_path.exists():
        blocks = read_json(summary_path)
        result["optimization"] = dict(blocks=len(blocks), accepted=sum(bool(b.get("accepted")) for b in blocks),
            acceptance_modes=dict(Counter(b.get("acceptance","legacy_unspecified") for b in blocks)),
            accepted_uphill=sum(bool(b.get("accepted")) and b.get("candidate",0)>b.get("baseline",0) for b in blocks))
    release_path = run/"released.json"
    result["release_record"] = read_json(release_path) if release_path.exists() else None
    if audit:
        from ao_shaping.tools.slm.transfer.artifacts import audit_run
        result["canonical_audit"] = audit_run(run)
    return result


def compare(run, baseline_label, candidate_label, *, check_raw=False):
    if baseline_label == candidate_label:
        raise ValueError("baseline and candidate labels must differ")
    rows = load_records(run)
    groups = []
    for label in (baseline_label,candidate_label):
        group = {}
        for row in rows:
            if row.get("label") != label:
                continue
            exposure = row["exposure_ms"]
            if exposure in group:
                raise ValueError(f"ambiguous repeated label/exposure: {label}, {exposure}")
            group[exposure] = row
        if not group:
            raise ValueError(f"label not found: {label}")
        groups.append(group)
    if set(groups[0]) != set(groups[1]):
        raise ValueError("exposure sets differ; do not silently discard unmatched exposures")
    comparisons = []
    phase_by_label = {}
    for exposure in sorted(groups[0]):
        a,b = groups[0][exposure],groups[1][exposure]
        for field in ("bit_depth","full_scale","counts_scale"):
            if a.get(field) != b.get(field):
                raise ValueError(f"baseline/candidate {field} differs; compare a common camera format")
        raw = [raw_details(run,r) for r in (a,b)] if check_raw else None
        if raw and raw[0]["grid_core_sha256"] != raw[1]["grid_core_sha256"]:
            raise ValueError("baseline/candidate camera grid, target or core differs")
        if raw and any(r["saturated_core_samples"] for r in raw):
            raise ValueError("saturated comparison core; inspect original frames")
        if raw:
            for label, detail in zip((baseline_label,candidate_label),raw):
                previous = phase_by_label.setdefault(label,detail["phase_sha256"])
                if previous != detail["phase_sha256"]:
                    raise ValueError(f"phase changes across exposures for {label}; not an exact-phase replay")
        metrics = [r["metrics"] for r in raw] if raw else [a["metrics"],b["metrics"]]
        ca,cb = (m["cv"] for m in metrics)
        if not all(math.isfinite(c) and c>=0 for c in (ca,cb)):
            raise ValueError("finite nonnegative CV required")
        comparisons.append(dict(exposure_ms=exposure, baseline_file=a["file"], candidate_file=b["file"],
            baseline_cv=ca,candidate_cv=cb,delta_cv=cb-ca,
            relative_cv_reduction=(1-cb/ca) if ca>0 else None,
            baseline_mean=metrics[0].get("mean"),candidate_mean=metrics[1].get("mean"),
            raw_checks=raw))
    return dict(run=str(Path(run).resolve()), baseline=baseline_label,candidate=candidate_label,
                scope="saved sampled core; same exposure; no causal drift correction",
                raw_checked=check_raw, comparison=comparisons,
                caveat="Metadata mode does not establish identical masks, phases, bit depth, or valid raw frames." if not check_raw else
                       "Independent test status depends on whether these records were used for model/phase selection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    for name in ("review","compare"):
        command = commands.add_parser(name)
        command.add_argument("--run",type=Path,required=True,help="Directory containing manifest.json")
        command.add_argument("--check-raw",action="store_true",help="Recompute saved sampled core metrics via AO-shaping; requires numpy")
        command.add_argument("--out",type=Path,help="Optional NEW JSON output; existing files are not overwritten")
        if name == "review":
            command.add_argument("--audit",action="store_true",help="Delegate complete validate-run audit to AO-shaping; not for replay-only directories")
        else:
            command.add_argument("--baseline-label",required=True)
            command.add_argument("--candidate-label",required=True)
    args = parser.parse_args()
    try:
        if args.out and args.out.exists():
            raise ValueError(f"output already exists: {args.out}")
        if args.command == "review":
            result = review(args.run,check_raw=args.check_raw,audit=args.audit)
        else:
            result = compare(args.run,args.baseline_label,args.candidate_label,check_raw=args.check_raw)
        result["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
        text = json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
        if args.out:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            with args.out.open("x",encoding="utf-8") as stream:
                stream.write(text+"\n")
        print(text)
    except (ValueError,OSError,KeyError,ImportError) as error:
        parser.exit(2,f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
