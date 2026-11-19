# -*- coding: utf-8 -*-
"""Do the ORIGINAL-session residual correctors transfer to a NEW session (B)?

The correctors under test were trained on the original dataset's physical fit
(``fit_m3_full``, iteration 17, 56 train rows) and never saw session B:

* ``residual_operator_m3_iter17.pt`` -- ``kind=operator``, phase enters only
  through the fixed transport basis (coefficients predicted from a 16x16 pooled
  phase map plus rms/peak);
* ``residual_cnn_m3_iter17.pt`` -- ``kind=cnn``, phase is a full second input
  channel (2x2 -> 64x64 pooled).

Session B has different probe phases AND a different reference spot, so the
original ``iteration_0017.npz`` predictions are unusable. This script rebuilds
B's baseline physical prediction from B's own data (ramp matrix re-fitted from
B's signed ramp pairs, physical parameters and radiometry from the original
fit's ``frozen_parameters.json``, origin carried across sessions by the fit's
fitted offset), feeds it to both correctors exactly as training did (raw
``model_phase`` on the 128 grid, no extra normalisation), and measures a single
scalar gain ``g = <pred, meas> / <pred, pred>`` fitted on B's TRAIN rows only.

Two gates run before B is touched:

1. the rebuilt ORIGINAL prediction must match ``iteration_0017.npz``
   (``max abs dev`` in counts);
2. both correctors, rebuilt from their bundles alone, must reproduce the
   stored in-session metrics in ``residual_*_m3_iter17.json``.

If either gate fails the script stops without evaluating B. Nothing trains and
no hardware is touched.

Recalibration is a per-session scalar gain only: it is fitted on
``--recalibrate-split`` (default ``train``) and applied unchanged to every row,
test included. Test rows are never used to fit anything.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SPLIT_GROUPS = ("all", "train", "test", "calibration")
FAMILIES = ("zernike", "smooth", "bandlimited")
RECALIBRATION_SPLITS = ("train", "calibration")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "slm"

REBUILD_MAX_ABS_DEV_COUNTS = 0.05
REBUILD_MAX_RELATIVE_DEV = 1e-4
CORRECTOR_REPRODUCTION_TOL = 1e-5


# ---------------------------------------------------------------- helpers

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sibling(filename: str, module_name: str):
    """Import a sibling script by path (``scripts/`` is not a package)."""
    path = SCRIPTS_DIR / filename
    if not path.is_file():
        raise SystemExit(f"missing sibling script {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_bundle(path: Path, expected_kind: str) -> dict[str, Any]:
    """Load a ``weights_only`` corrector bundle and check its shape."""
    import torch

    if not path.is_file():
        raise SystemExit(f"missing weights bundle {path}")
    bundle = torch.load(path, weights_only=True, map_location="cpu")
    required = {"format_version", "state_dict", "kind", "best_epoch",
                "best_train_loss", "config", "fit", "iteration"}
    missing = required - set(bundle)
    if missing:
        raise SystemExit(f"{path}: bundle lacks {sorted(missing)}")
    if bundle["kind"] != expected_kind:
        raise SystemExit(f"{path}: kind {bundle['kind']!r} != {expected_kind!r}")
    return bundle


def rows_of(dataset: Path) -> tuple[list[dict], dict]:
    """Manifest rows the production loader keeps, plus the dataset config."""
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((dataset / "config.json").read_text(encoding="utf-8"))
    return manifest, config


def selected_rows(train_module, dataset: Path) -> tuple[list[dict], int]:
    """Rows in the exact order ``fit.load_dataset`` selects them."""
    manifest, config = rows_of(dataset)
    usable = [row for row in manifest if train_module.valid(row)]
    selected = [row for row in usable
                if row.get("split") in {"train", "test", "validation"}]
    return selected, int(config.get("grid", 128))


def load_phases(train_module, dataset: Path, selected: list[dict], grid: int) -> np.ndarray:
    """Stack ``model_phase`` exactly as ``train_residual_corrector.assemble`` does.

    Training fed the raw SLM control grid (no extra resize/crop/normalisation);
    every normalisation lives inside the model as a state_dict buffer, so the
    only thing to preserve here is the array and the row order.
    """
    phases = []
    for row in selected:
        path = Path(row["file"])
        if not path.is_absolute():
            path = dataset / path
        payload = np.load(path)
        try:
            phase = np.asarray(payload["model_phase"])
        finally:
            payload.close()
        if phase.shape != (grid, grid):
            raise SystemExit(f"{path.name}: model_phase {phase.shape} != {(grid, grid)}")
        phases.append(phase)
    return np.stack(phases).astype(np.float32)


def all_rows_view(new_dir: Path, workspace: Path) -> Path:
    """A view of ``new_dir`` whose calibration rows are visible to the loader.

    ``fit.load_dataset`` keeps only train/test/validation, which silently drops
    the 10 calibration rows (references + signed ramps) and would make "all
    rows" 104, not 114. The view relabels ``calibration -> validation`` and
    points ``file`` at absolute paths; nothing else changes. On the shared rows
    the rebuilt prediction and measured images are bit-identical to the direct
    build (asserted by the caller through the reproduction gate).
    """
    view = workspace / "new_session_view"
    view.mkdir(parents=True, exist_ok=True)
    manifest, _ = rows_of(new_dir)
    rewritten = []
    for row in manifest:
        entry = dict(row)
        if entry.get("split") == "calibration":
            entry["split"] = "validation"
        path = Path(entry["file"])
        entry["file"] = str(path if path.is_absolute() else (new_dir / path).resolve())
        rewritten.append(entry)
    (view / "manifest.json").write_text(
        json.dumps(rewritten, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(new_dir / "config.json", view / "config.json")
    return view


# ---------------------------------------------------------------- gates

def gate_physical_rebuild(transfer_module, original: Path, fit: Path,
                          span: int, device: str | None) -> tuple[dict, np.ndarray]:
    """Rebuilt ORIGINAL prediction vs the stored ``iteration_XXXX.npz``."""
    frozen, result = transfer_module.load_fit(fit)
    report, _ = transfer_module.validate_rebuild(original, fit, frozen, result, span, device)
    if not report.get("passed"):
        raise SystemExit(
            "physical rebuild fails the faithfulness gate: "
            f"max_abs_dev={report.get('max_abs_deviation_counts'):.3e} counts, "
            f"relative={report.get('relative_l2_deviation'):.3e} "
            f"(tolerances {REBUILD_MAX_ABS_DEV_COUNTS} / {REBUILD_MAX_RELATIVE_DEV}); "
            "refusing to evaluate the new session")
    return report, np.asarray(report.get("fitted_origin_offset"), dtype=float)


def gate_corrector_reproduction(train_module, transfer_module, original: Path,
                                fit: Path, iteration: int,
                                models: dict[str, Any],
                                stored_reports: dict[str, dict]) -> dict:
    """Both correctors must reproduce the stored in-session metrics exactly.

    This proves the bundle rebuild and the (prediction, raw model_phase) input
    pair reproduce training, which is the part most likely to drift when the
    same code is pointed at a new session.
    """
    data = train_module.assemble(original, fit, iteration)
    phases = data["phases"]
    predictions = data["predictions"]
    out: dict[str, Any] = {}
    worst = 0.0
    for name, bundle in models.items():
        model = train_module.rebuild_from_bundle(bundle)
        with torch_no_grad():
            corrected = model(predictions, phases)
        deviations: dict[str, float] = {}
        reference = stored_reports[name]["summary"]["test"]
        for family in FAMILIES:
            index = [i for i, row in enumerate(data["rows"])
                     if row["split"] == "test" and row["family"] == family]
            if not index:
                continue
            for metric in ("correlation", "relative_l2", "cv_abs_error"):
                ours = float(np.mean([train_module.row_metrics(
                    corrected[i].numpy(), data["measured"][i].numpy())[metric]
                    for i in index]))
                theirs = float(reference[family][metric][f"{name}_mean"])
                deviations[f"{family}.{metric}"] = abs(ours - theirs)
                worst = max(worst, abs(ours - theirs))
        out[name] = {"max_abs_deviation": max(deviations.values(), default=0.0),
                     "deviations": deviations,
                     "best_epoch": int(bundle["best_epoch"]),
                     "parameters": int(sum(p.numel() for p in model.parameters()
                                           if p.requires_grad))}
    out["passed"] = worst <= CORRECTOR_REPRODUCTION_TOL
    out["tolerance"] = CORRECTOR_REPRODUCTION_TOL
    if not out["passed"]:
        raise SystemExit(
            "corrector reproduction gate fails: reconstructed in-session metrics "
            f"deviate by up to {worst:.3e} (tolerance {CORRECTOR_REPRODUCTION_TOL}); "
            "the (prediction, phase) path does not match training")
    return out


class torch_no_grad:
    """``torch.no_grad`` without importing torch at module import time."""

    def __enter__(self):
        import torch
        self._context = torch.no_grad()
        return self._context.__enter__()

    def __exit__(self, *exc):
        return self._context.__exit__(*exc)


# ---------------------------------------------------------------- metrics

def measure(train_module, prediction: np.ndarray, measured: np.ndarray,
            splits: np.ndarray, families: np.ndarray) -> dict:
    """Grouped metrics: per split, overall and per probe family."""
    import torch

    def block(mask: np.ndarray) -> dict:
        count = int(mask.sum())
        if count == 0:
            return {"n_rows": 0}
        mean = train_module.mean_metrics(
            torch.from_numpy(np.ascontiguousarray(prediction[mask], dtype=np.float64)),
            torch.from_numpy(np.ascontiguousarray(measured[mask], dtype=np.float64)))
        per_row_metrics = per_row(train_module, prediction[mask], measured[mask])
        return {"n_rows": count, **{k: float(v) for k, v in mean.items()},
                "cv_prediction": float(np.mean([m["cv_prediction"]
                                                for m in per_row_metrics])),
                "cv_measured": float(np.mean([m["cv_measured"]
                                              for m in per_row_metrics]))}

    out: dict[str, Any] = {}
    for split in SPLIT_GROUPS:
        mask = np.ones(len(splits), bool) if split == "all" else splits == split
        out[split] = {"overall": block(mask),
                      "by_family": {family: block(mask & (families == family))
                                    for family in FAMILIES}}
    return out


def scalar_gain(prediction: np.ndarray, measured: np.ndarray, mask: np.ndarray) -> float:
    """Least-squares scalar ``g = <p, m> / <p, p>`` over ``mask``."""
    p = prediction[mask].astype(np.float64).ravel()
    m = measured[mask].astype(np.float64).ravel()
    denominator = float(np.dot(p, p))
    if denominator <= 0.0:
        raise SystemExit("degenerate prediction set: <p, p> <= 0")
    return float(np.dot(p, m) / denominator)


def per_row(train_module, prediction: np.ndarray, measured: np.ndarray) -> list[dict]:
    return [train_module.row_metrics(prediction[i], measured[i])
            for i in range(prediction.shape[0])]


# ---------------------------------------------------------------- reporting

def verdict_block(models: dict[str, dict]) -> dict:
    """Does each corrector beat the recalibrated baseline on unseen B test rows?"""
    base = models["baseline"]["after"]["test"]["overall"]
    base_rows = models["baseline"]["rows"]
    test_index = [i for i, row in enumerate(base_rows) if row["split"] == "test"]
    out: dict[str, Any] = {
        "reference": "baseline after recalibration",
        "baseline_test": {
            "relative_l2_before": models["baseline"]["before"]["test"]["overall"]["relative_l2"],
            "relative_l2_after": base["relative_l2"],
            "correlation_after": base["correlation"],
        },
        "models": {},
    }
    for name in ("operator", "cnn"):
        after = models[name]["after"]["test"]["overall"]
        before = models[name]["before"]["test"]["overall"]
        delta_l2 = after["relative_l2"] - base["relative_l2"]
        by_family = {}
        for family in FAMILIES:
            b = models["baseline"]["after"]["test"]["by_family"][family]
            a = models[name]["after"]["test"]["by_family"][family]
            if not a.get("n_rows") or not b.get("n_rows"):
                continue
            rows = [i for i in test_index if base_rows[i]["family"] == family]
            wins = sum(models[name]["rows"][i]["after"]["relative_l2"]
                       < base_rows[i]["after"]["relative_l2"] for i in rows)
            by_family[family] = {
                "relative_l2_delta": a["relative_l2"] - b["relative_l2"],
                "correlation_delta": a["correlation"] - b["correlation"],
                "relative_l2_wins": int(wins),
                "n_rows": len(rows),
            }
        wins_all = sum(models[name]["rows"][i]["after"]["relative_l2"]
                       < base_rows[i]["after"]["relative_l2"] for i in test_index)
        out["models"][name] = {
            "relative_l2_before_delta": before["relative_l2"]
            - models["baseline"]["before"]["test"]["overall"]["relative_l2"],
            "relative_l2_after": after["relative_l2"],
            "relative_l2_delta": delta_l2,
            "relative_l2_reduction": -delta_l2 / base["relative_l2"],
            "correlation_after": after["correlation"],
            "correlation_delta": after["correlation"] - base["correlation"],
            "relative_l2_wins": int(wins_all),
            "n_rows": len(test_index),
            "transfers": bool(delta_l2 < 0.0),
            "by_family": by_family,
        }
    return out


def shared_gain_block(measures: dict[str, dict], gain: float) -> dict:
    """Sensitivity: every model under the BASELINE's session gain.

    A corrector refines the physical prediction, so it arguably inherits the
    session radiometry rather than getting a gain of its own. This shows what
    each model does when the scale is not free to absorb its amplitude error.
    """
    base = measures["baseline"]["test"]["overall"]
    out: dict[str, Any] = {"gain": gain, "models": {}}
    for name in ("operator", "cnn"):
        entry = measures[name]["test"]["overall"]
        out["models"][name] = {
            "relative_l2": entry["relative_l2"],
            "relative_l2_delta": entry["relative_l2"] - base["relative_l2"],
            "correlation": entry["correlation"],
            "correlation_delta": entry["correlation"] - base["correlation"],
        }
    out["baseline_test"] = {"relative_l2": base["relative_l2"],
                            "correlation": base["correlation"]}
    return out


def print_summary(payload: dict, out: Path) -> None:
    checks = payload["checks"]
    print(f"gate physical rebuild : max_abs_dev={checks['physical_rebuild']['max_abs_deviation_counts']:.3e} "
          f"counts  rel={checks['physical_rebuild']['relative_l2_deviation']:.3e}  "
          f"{'PASS' if checks['physical_rebuild']['passed'] else 'FAIL'}")
    replay = checks["corrector_reproduction"]
    worst = max(replay[name]["max_abs_deviation"] for name in ("operator", "cnn"))
    print(f"gate corrector replay : max_abs_dev={worst:.3e} "
          f"{'PASS' if replay['passed'] else 'FAIL'}")
    splits = payload["dataset"]["splits"]
    print(f"session B rows      : {payload['dataset']['n_rows']}  "
          f"(train {splits.get('train', 0)}, test {splits.get('test', 0)}, "
          f"calibration {splits.get('calibration', 0)})")
    print(f"recalibration       : split={payload['recalibration']['split']} "
          f"n={payload['recalibration']['n_rows']}")
    for name, entry in payload["models"].items():
        print(f"  gain[{name:<8}] = {entry['recalibration']['gain']:.6f}")
    header = f"{'model':<9}{'split':<8}{'corr':>8}{'rl2':>8}{'cv_abs':>9}"
    print("\nAFTER recalibration (B test):")
    print(header)
    for name, entry in payload["models"].items():
        for split in ("train", "test"):
            m = entry["after"][split]["overall"]
            print(f"{name:<9}{split:<8}{m['correlation']:>8.4f}"
                  f"{m['relative_l2']:>8.4f}{m['cv_abs_error']:>9.4f}")
    print("\nAFTER recalibration, B test by family (relative_l2):")
    print(f"{'family':<14}" + "".join(f"{n:>11}" for n in payload["models"]))
    for family in FAMILIES:
        cells = "".join(
            f"{payload['models'][n]['after']['test']['by_family'][family].get('relative_l2', float('nan')):>11.4f}"
            for n in payload["models"])
        print(f"{family:<14}{cells}")
    print("\nBEFORE recalibration, B test by family (relative_l2):")
    print(f"{'family':<14}" + "".join(f"{n:>11}" for n in payload["models"]))
    for family in FAMILIES:
        cells = "".join(
            f"{payload['models'][n]['before']['test']['by_family'][family].get('relative_l2', float('nan')):>11.4f}"
            for n in payload["models"])
        print(f"{family:<14}{cells}")
    print("\nverdict (unseen B test, after gain):")
    for name, entry in payload["verdict"]["models"].items():
        print(f"  {name:<8} d(rl2)={entry['relative_l2_delta']:+.4f} "
              f"({entry['relative_l2_reduction'] * 100:+.1f}%)  "
              f"d(corr)={entry['correlation_delta']:+.4f}  "
              f"wins={entry['relative_l2_wins']}/{entry['n_rows']}  "
              f"transfers={entry['transfers']}")
        for family, cell in entry["by_family"].items():
            print(f"    {family:<12} d(rl2)={cell['relative_l2_delta']:+.4f}  "
                  f"d(corr)={cell['correlation_delta']:+.4f}  "
                  f"wins={cell['relative_l2_wins']}/{cell['n_rows']}")
    shared = payload["shared_gain_sensitivity"]
    print(f"\nsensitivity: every model under the baseline's gain {shared['gain']:.4f} "
          f"(B test relative_l2)")
    for name, entry in shared["models"].items():
        print(f"  {name:<8} rl2={entry['relative_l2']:.4f} "
              f"(d={entry['relative_l2_delta']:+.4f})  "
              f"corr={entry['correlation']:.4f} (d={entry['correlation_delta']:+.4f})")
    print(f"\nwrote {out}")


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original", required=True, type=Path,
                        help="original-session dataset directory")
    parser.add_argument("--new", required=True, type=Path,
                        help="new-session dataset directory (B)")
    parser.add_argument("--operator-weights", required=True, type=Path)
    parser.add_argument("--cnn-weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fit", default="fit_m3_full",
                        help="fit inside --original (name or path)")
    parser.add_argument("--iteration", type=int, default=17)
    parser.add_argument("--span", type=int, default=160)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--origin-mode", default="session",
                        choices=("session", "measured"),
                        help="carry the fit's origin offset across sessions (session) "
                             "or re-anchor at the new measured spot (measured)")
    parser.add_argument("--recalibrate-split", default="train",
                        choices=RECALIBRATION_SPLITS)
    args = parser.parse_args(argv)

    torch = importlib.import_module("torch")

    transfer = load_sibling("evaluate_transfer_on_new_session.py", "evaluate_transfer_on_new_session")
    train_module = load_sibling("train_residual_corrector.py", "train_residual_corrector")

    original = args.original.resolve()
    new = args.new.resolve()
    fit = transfer.resolve_fit(original, args.fit)
    if not new.is_dir():
        raise SystemExit(f"missing new-session dataset {new}")

    bundles = {
        "operator": load_bundle(args.operator_weights.resolve(), "operator"),
        "cnn": load_bundle(args.cnn_weights.resolve(), "cnn"),
    }
    for name, bundle in bundles.items():
        if Path(bundle["fit"]).resolve() != fit.resolve():
            print(f"note: {name} bundle records fit {bundle['fit']} (expected {fit})")
        if int(bundle["iteration"]) != args.iteration:
            print(f"note: {name} bundle records iteration {bundle['iteration']} "
                  f"(expected {args.iteration})")

    stored_reports: dict[str, dict] = {}
    for name, weights_path in (
            ("operator", args.operator_weights.resolve()),
            ("cnn", args.cnn_weights.resolve())):
        report_path = weights_path.with_suffix(".json")
        if not report_path.is_file():
            raise SystemExit(
                f"missing {report_path}; the corrector reproduction gate needs the "
                "in-session report written next to the weights")
        stored_reports[name] = json.loads(report_path.read_text(encoding="utf-8"))

    # --- gates: the ORIGINAL session must reproduce before B is touched
    rebuild, origin_offset = gate_physical_rebuild(
        transfer, original, fit, args.span, args.device)
    reproduction = gate_corrector_reproduction(
        train_module, transfer, original, fit, args.iteration, bundles, stored_reports)

    # --- session B: predictions from B's own data
    workspace = Path(tempfile.mkdtemp(prefix="corrector_transfer_"))
    view = all_rows_view(new, workspace)
    try:
        comps, frozen = transfer.build_transfer_model(
            view, fit, args.span, args.device, origin_mode=args.origin_mode,
            origin_offset=origin_offset)
        prediction_raw = transfer.predict_counts(comps, frozen)  # counts at 161x161
        measured = np.asarray(comps.images, dtype=np.float64)
        labels = [str(row["label"]) for row in comps.rows]
        splits = np.asarray([str(row["split"]) for row in comps.rows])
        splits = np.where(splits == "validation", "calibration", splits)
        families = np.asarray([transfer.probe_family(label) for label in labels])
        selected, grid = selected_rows(train_module, view)
        if [row["label"] for row in selected] != labels:
            raise SystemExit("row order of the B view disagrees with the manifest; "
                             "cannot align phases")
        phases = load_phases(train_module, view, selected, grid)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    if prediction_raw.shape != measured.shape:
        raise SystemExit(f"prediction {prediction_raw.shape} != measured {measured.shape}")
    if prediction_raw.shape[1] != bundles["operator"]["config"]["CAMERA_GRID"]:
        raise SystemExit(f"prediction grid {prediction_raw.shape[1:]} != "
                         f"CAMERA_GRID {bundles['operator']['config']['CAMERA_GRID']}")

    prediction32 = torch.from_numpy(prediction_raw.astype(np.float32))
    phase32 = torch.from_numpy(phases)
    outputs = {"baseline": prediction_raw.astype(np.float64)}
    with torch.no_grad():
        for name, bundle in bundles.items():
            model = train_module.rebuild_from_bundle(bundle)
            outputs[name] = model(prediction32, phase32).detach().numpy().astype(np.float64)

    # --- per-session scalar gain on B train only
    train_mask = splits == args.recalibrate_split
    if not train_mask.any():
        raise SystemExit(f"B has no {args.recalibrate_split!r} rows to recalibrate on")
    gains = {name: scalar_gain(values, measured, train_mask)
             for name, values in outputs.items()}

    models: dict[str, Any] = {}
    for name, values in outputs.items():
        after = values * gains[name]
        before_rows = per_row(train_module, values, measured)
        after_rows = per_row(train_module, after, measured)
        models[name] = {
            "recalibration": {
                "split": args.recalibrate_split,
                "n_rows": int(train_mask.sum()),
                "gain": gains[name],
                "scale_shift": gains[name] - 1.0,
                "note": "one scalar fitted on the recalibration split and applied "
                        "unchanged to every row; correlation and the CV metrics are "
                        "scale-invariant, relative_l2 (and the absolute residual) "
                        "scale with it",
            },
            "before": measure(train_module, values, measured, splits, families),
            "after": measure(train_module, after, measured, splits, families),
            "rows": [
                {"label": labels[i], "split": str(splits[i]),
                 "family": str(families[i]),
                 "before": before_rows[i], "after": after_rows[i]}
                for i in range(len(labels))
            ],
        }

    shared_measures = {
        "baseline": measure(train_module, outputs["baseline"], measured, splits, families)}
    for name in ("operator", "cnn"):
        shared_measures[name] = measure(
            train_module, outputs[name] * gains["baseline"], measured, splits, families)

    payload = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "question": "do the original-session residual correctors transfer to "
                    "session B with a per-session scalar gain only?",
        "inputs": {
            "original": str(original),
            "new": str(new),
            "fit": str(fit),
            "iteration": args.iteration,
            "span": args.span,
            "origin_mode": args.origin_mode,
            "recalibrate_split": args.recalibrate_split,
            "operator_weights": {"path": str(args.operator_weights.resolve()),
                                 "sha256": sha256_file(args.operator_weights.resolve()),
                                 "best_epoch": int(bundles["operator"]["best_epoch"]),
                                 "kind": bundles["operator"]["kind"],
                                 "parameters": sum(p.numel() for p in
                                                   train_module.rebuild_from_bundle(
                                                       bundles["operator"]).parameters()
                                                   if p.requires_grad)},
            "cnn_weights": {"path": str(args.cnn_weights.resolve()),
                            "sha256": sha256_file(args.cnn_weights.resolve()),
                            "best_epoch": int(bundles["cnn"]["best_epoch"]),
                            "kind": bundles["cnn"]["kind"],
                            "parameters": sum(p.numel() for p in
                                              train_module.rebuild_from_bundle(
                                                  bundles["cnn"]).parameters()
                                              if p.requires_grad)},
        },
        "method": {
            "prediction": "B's own physical forward: ramp matrix from B's signed "
                          "ramp pairs, physical parameters and radiometry from the "
                          "original fit's frozen_parameters.json, origin carried "
                          f"across sessions ({args.origin_mode}), residual Z5..Z11 "
                          "from the original fit",
            "phase": "B's raw model_phase (128 grid) fed unchanged, matching "
                     "train_residual_corrector.assemble; all normalisation is a "
                     "fixed buffer inside each bundle",
            "metrics": "train_residual_corrector.row_metrics / mean_metrics",
            "calibration_rows": "the production loader keeps only train/test/"
                                "validation; calibration rows were exposed by "
                                "relabelling split=calibration -> validation in a "
                                "temporary manifest view (prediction identical)",
        },
        "checks": {
            "physical_rebuild": {
                "reference": str(fit / f"iteration_{args.iteration:04d}.npz"),
                "max_abs_deviation_counts": float(rebuild["max_abs_deviation_counts"]),
                "relative_l2_deviation": float(rebuild["relative_l2_deviation"]),
                "correlation": float(rebuild["correlation"]),
                "tolerance_counts": REBUILD_MAX_ABS_DEV_COUNTS,
                "passed": bool(rebuild["passed"]),
                "fitted_origin_offset_px": origin_offset.tolist(),
            },
            "corrector_reproduction": reproduction,
        },
        "dataset": {
            "n_rows": len(labels),
            "splits": {group: int((splits == group).sum()) for group in SPLIT_GROUPS
                       if group != "all"},
            "families": {family: int((families == family).sum()) for family in FAMILIES},
            "n_family_calibration": int((families == "calibration").sum()),
            "phase_rms": float(np.sqrt((phases.astype(np.float64) ** 2).mean())),
        },
        "recalibration": {
            "formula": "g = <prediction, measured> / <prediction, prediction>",
            "split": args.recalibrate_split,
            "n_rows": int(train_mask.sum()),
            "gains": {name: float(value) for name, value in gains.items()},
        },
        "models": models,
        "verdict": verdict_block(models),
        "shared_gain_sensitivity": shared_gain_block(shared_measures, gains["baseline"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print_summary(payload, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
