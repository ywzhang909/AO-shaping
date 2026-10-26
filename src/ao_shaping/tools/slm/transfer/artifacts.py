"""Device-free validation and relocation of recorded transfer artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .capture import to_panel
from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective, image_metrics


def resolve_recorded_path(value, *, relative_to=None):
    """Resolve a relocated repository artifact without assuming the old drive."""
    path = Path(value)
    if path.exists():
        return path.resolve()
    roots = [Path(relative_to), *Path(relative_to).parents] if relative_to else []
    roots += [Path.cwd(), *Path.cwd().parents]
    normalized = str(value).replace("\\", "/")
    suffix = normalized[normalized.index("data/"):] if "data/" in normalized else normalized
    for root in roots:
        candidate = root / suffix
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Recorded artifact unavailable: {value}; restore its data directory")


def audit_run(run: Path):
    """Recompute metrics from raw CCD frames and verify phase/mask/clock contracts."""
    run = Path(run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    records = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if not records:
        raise ValueError("empty acquisition manifest")
    exposures, labels, hashes = {}, set(), {}
    maximum_peak, frame_count = 0., 0
    for row in records:
        label = row["label"]
        if label in labels:
            raise ValueError(f"duplicate capture label: {label}")
        labels.add(label)
        file = run / row["file"]
        with np.load(file, allow_pickle=False) as saved:
            raw, phase, xy, target, core = (saved[k] for k in ("raw_frames", "model_phase", "camera_xy", "target", "core"))
            if raw.ndim != 3 or raw.shape[0] != len(row["frames"]):
                raise ValueError(f"raw frame count differs: {label}")
            if not np.isfinite(raw).all() or not np.isfinite(phase).all():
                raise ValueError(f"nonfinite data: {label}")
            if not np.array_equal(to_panel(phase), saved["panel_phase"]):
                raise ValueError(f"display/model phase mismatch: {label}")
            if saved["panel_gray"].shape != saved["panel_phase"].shape:
                raise ValueError(f"grayscale shape mismatch: {label}")
            if xy.shape != (*target.shape, 2) or core.shape != target.shape:
                raise ValueError(f"camera/target shape mismatch: {label}")
            x, y = xy[..., 0].astype(int), xy[..., 1].astype(int)
            if x.min() < 0 or y.min() < 0 or x.max() >= raw.shape[2] or y.max() >= raw.shape[1]:
                raise ValueError(f"camera window beyond sensor: {label}")
            objective = ShapeObjective(target)
            if config.get("core_margin") is not None:
                from scipy.ndimage import binary_erosion
                objective.core = binary_erosion(objective.box, iterations=max(1, int(np.ceil(config["core_margin"]/config.get("window_step", 2.)))))
            if config.get("zero_radius", 0.) > 0:
                objective.core &= np.linalg.norm(xy - np.asarray(config["zero_xy"]), axis=-1) > config["zero_radius"]
            if not np.array_equal(objective.core, core):
                raise ValueError(f"core mask mismatch: {label}")
            measured = image_metrics(raw.mean(0)[y, x], objective)
            for key in ("cv", "mean", "ee_roi", "target_peak"):
                if not np.isclose(measured[key], row["metrics"][key], rtol=1e-6, atol=1e-6):
                    raise ValueError(f"raw/manifest {key} mismatch: {label}")
            image = raw.mean(0)[y, x]
            distribution = image / image.sum()
            loss = measured["cv"]**2 + 2*(1-measured["ee_roi"]) + target.sum()*np.square(distribution-target/target.sum()).sum()
            if config.get("objective", "shape") == "cv":
                loss = measured["cv"]**2
            elif config.get("objective", "shape") == "formation":
                loss = .05*measured["cv"]**2+8*(1-measured["ee_roi"])+target.sum()*np.square(distribution-target/target.sum()).sum()
            if not np.isclose(loss, row["loss"], rtol=2e-6, atol=2e-6):
                raise ValueError(f"raw/manifest objective mismatch: {label}")
            maximum_peak = max(maximum_peak, float(raw.max()))
            peak_to_check = raw[:, y, x][:, core].max() if config.get("meter_target", False) else raw.max()
            if peak_to_check >= row.get("full_scale", 255.)*250/255:
                raise ValueError(f"saturated measurement: {label}")
            frame_count += len(raw)
        exposure = float(row["exposure_ms"])
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError(f"invalid exposure: {label}")
        exposures.setdefault(label.split("_")[0], set()).add(exposure)
        for frame in row["frames"]:
            if not frame.get("timestamp_utc") or frame["end_monotonic_ns"] < frame["monotonic_ns"]:
                raise ValueError(f"invalid host clock: {label}")
            if frame["exposure_ms"] != exposure:
                raise ValueError(f"per-frame exposure mismatch: {label}")
        hashes[row["file"]] = hashlib.sha256(file.read_bytes()).hexdigest()
    for shape in config["shapes"]:
        if not {shape+"_seed", shape+"_final"}.issubset(labels):
            raise ValueError(f"unfinished shape: {shape}")
        if len(exposures[shape]) != 1 and config.get("exposure_policy", "per_shape") != "per_block":
            raise ValueError(f"paired exposure changed: {shape}")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if len(summary) != len(config["shapes"])*config["blocks"]:
        raise ValueError("incomplete optimization blocks")
    by_label = {r["label"]: r for r in records}
    exploration_policies = {}
    champion_labels = {}
    for block in summary:
        prefix = f"{block['shape']}_{block['block']}_"
        metric = config.get("accept_metric", "loss")
        if metric not in {"loss", "cv"}:
            raise ValueError("unknown acceptance metric")
        if block.get("acceptance") == "nonmonotone_ccd":
            from .exploration import NonmonotoneAcceptance
            shape = block["shape"]
            if shape not in exploration_policies:
                champion_labels[shape] = shape+"_seed"
                exploration_policies[shape] = NonmonotoneAcceptance(by_label[shape+"_seed"]["metrics"]["cv"],
                    temperature=config["exploration_temperature"], relative_budget=config["exploration_budget"],
                    seed=config["exploration_seed"])
            a, b = by_label[prefix+"A1"], by_label[prefix+"B"]
            if a["exposure_ms"] != b["exposure_ms"]:
                raise ValueError(f"exploration exposure mismatch: {prefix}")
            decision = exploration_policies[shape].update(a["metrics"]["cv"], b["metrics"]["cv"])
            for key, value in decision.items():
                if value != block[key]:
                    raise ValueError(f"exploration decision differs: {prefix} {key}")
            if decision["improved_best"]:
                champion_labels[shape] = prefix+"B"
            continue
        if block.get("acceptance") == "sequential_ccd":
            a, b = by_label[prefix+"A1"], by_label[prefix+"B"]
            if a["exposure_ms"] != b["exposure_ms"]:
                raise ValueError(f"sequential exposure mismatch: {prefix}")
            av = a["metrics"]["cv"] if metric == "cv" else a["loss"]
            bv = b["metrics"]["cv"] if metric == "cv" else b["loss"]
            if bool(block["accepted"]) != (bv < av):
                raise ValueError(f"sequential acceptance differs: {prefix}")
            continue
        if block.get("acceptance", "paired") == "model_only":
            # No A–B–B–A apparatus: recompute the model-only rule from the two
            # recorded losses, and refuse an artefact that also carries candidate
            # captures it claims not to have measured.
            if any(prefix+suffix in by_label for suffix in ("B1", "B2", "A2")):
                raise ValueError(f"model-only block carries paired captures: {prefix}")
            previous, last = block.get("model_loss_previous"), block.get("model_loss_last")
            if bool(block["accepted"]) != (previous is not None and last is not None and last < previous):
                raise ValueError(f"model-only acceptance differs: {prefix}")
            continue
        if len({by_label[prefix+suffix]["exposure_ms"] for suffix in ("A1", "B1", "B2", "A2")}) != 1:
            raise ValueError(f"exposure differs within paired block: {prefix}")
        losses = {suffix: (by_label[prefix+suffix]["metrics"]["cv"] if metric == "cv" else
                           by_label[prefix+suffix]["loss"]) for suffix in ("A1", "B1", "B2", "A2")}
        accepted = .5*(losses["B1"]+losses["B2"]) < .5*(losses["A1"]+losses["A2"])-abs(losses["A2"]-losses["A1"])
        if accepted != block["accepted"]:
            raise ValueError(f"acceptance decision differs: {prefix}")
    for shape, champion_label in champion_labels.items():
        with np.load(run / by_label[champion_label]["file"]) as champion:
            expected = champion["model_phase"]
            with np.load(run / by_label[shape+"_final"]["file"]) as final:
                if not np.array_equal(expected, final["model_phase"]):
                    raise ValueError(f"exploration final differs from champion: {shape}")
            if not np.array_equal(expected, np.load(run / f"{shape}_best.npy")):
                raise ValueError(f"exploration export differs from champion: {shape}")
    return dict(run=str(run.resolve()), captures=len(records), raw_frames=frame_count,
                maximum_peak=maximum_peak, shapes=config["shapes"], blocks=len(summary),
                accepted_blocks=sum(b["accepted"] for b in summary),
                acceptance_scopes=sorted({b.get("acceptance", "paired") for b in summary}),
                exposure_ms={s: sorted(e)[0] for s, e in exposures.items()},
                exposure_history_ms={s: sorted(e) for s, e in exposures.items()},
                file_sha256=hashes, status="passed", timestamp_scope="host calls, not sensor exposure triggers")
