# -*- coding: utf-8 -*-
"""Cross-session transfer: do ORIGINAL-session physical fits predict NEW-session rows?

The two fits under test were produced on the original dataset (``--original``):
``fit_m3_full`` (baseline) and ``fit_m3_static_global`` (corrected M1, a shared
static high-pass term). The new dataset (``--new``) has DIFFERENT probe phases
and a DIFFERENT reference spot, so the stored prediction arrays from the fit
cannot be reused. This script builds a fresh forward on the new dataset:

* the ramp matrix is re-fitted from the new dataset's signed ramp pairs,
* the physical parameters (defocus, amplitude centre/width, residual Z5..Z11,
  sigma_ccd_px, registration shift, grid/fft/pixel_samples) and the radiometry
  (gain / background / bias_counts) are the ORIGINAL fit's,
* the model origin is re-anchored to the new session's ``reference_start`` spot.

Origin placement is reported two ways, because the fit does not keep its origin
at the measured reference spot: it moves it by a fitted offset
``origin_fit - measured_reference`` (5.2 px on the baseline fit). The physically
consistent transfer carries that offset across sessions,
``origin_new = measured_new + (origin_fit - measured_old)`` (``session``); the
literal reading of "use the new measured origin" is ``measured``. Both are
reported. Nothing is trained here.

The new-phase prediction path is validated first: the same builder is applied to
the ORIGINAL dataset with the original fitted origin and ramp matrix, and the
rebuilt prediction must match the stored ``iteration_XXXX.npz`` array. If that
check fails the script stops without evaluating the new session.

Static terms (master note): the master forward (``tools.slm.transfer.fit``)
carries no static basis. A checkpoint that declares ``static_basis_k > 0`` still
loads -- the field is read defensively and reported in ``static`` -- but its
coefficients cannot be applied here, so the ``*_static_off`` variants do not
exist and the original-rebuild gate for that fit fails closed instead of
silently evaluating the new session with the wrong observation model. Only the
``static_here`` branch below (a compatibility ``comps.static_coef``) can apply a
term; on master no such attribute exists.

After the raw transfer the script fits ONE scalar gain on the new dataset's
train rows only, ``g = <pred, meas> / <pred, pred>``, applies it to ALL rows
(test included) and reports the metrics before and after. That scalar is the
per-session radiometric recalibration.

No device imports, no hardware access.

Usage:
  python3 scripts/slm/evaluate_transfer_on_new_session.py \
    --original data/debug/slm_transfer/20261118_transfer_mil_zernike_probe_v1 \
    --new data/debug/slm_transfer/extend_b_20261118 \
    --fit fit_m3_full --fit fit_m3_static_global \
    --out data/debug/slm_transfer/extend_b_20261118/eval_transfer_master_api
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

#: Camera sampling window half-width, matching the fits (161x161 window, step 2).
DEFAULT_SPAN = 160
#: Residual Zernike modes the fits optimised.
RESIDUAL_MODES = tuple(range(5, 12))
#: Splits reported for the new dataset.
SPLIT_GROUPS = ("all", "train", "test", "calibration")
#: Probe families; calibration rows (reference + signed ramps) are not probes.
FAMILIES = ("zernike", "smooth", "bandlimited")
#: Faithfulness gate for the rebuilt-original path. The rebuilt prediction is a
#: float64 forward against a stored float32 array; the observed deviation is
#: ~8e-7 relative (float32 storage of counts up to ~8e3), so the absolute bound
#: is a coarse sanity check and the relative bound is the real gate.
VALIDATION_MAX_ABS_DEV_COUNTS = 0.05
VALIDATION_MAX_RELATIVE_DEV = 1e-4
VALIDATION_MIN_CORR = 0.9999
#: Origin placements: carried fitted offset, literal measured spot, fitted spot.
ORIGIN_MODES = ("session", "measured")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe_family(label: str) -> str:
    """Family from a manifest label, as the other transfer scripts derive it.

    ``ztrain_000_Z4_+0.10`` has no family suffix but is a Zernike probe;
    ``rtrain_000_smooth`` / ``holdout_000_bandlimited`` carry it as the suffix.
    """
    if label.startswith("ztrain_"):
        return "zernike"
    if label.startswith(("rtrain_", "holdout_")):
        return label.rsplit("_", 1)[-1]
    return "calibration"


def resolve_fit(original: Path, spec: str) -> Path:
    """Accept a fit name inside ``--original`` or an explicit directory."""
    candidate = Path(spec)
    if candidate.is_dir():
        return candidate
    candidate = original / spec
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"fit directory not found: {spec!r} (also tried {candidate})")


def load_fit(fit_dir: Path) -> tuple[dict, dict]:
    """Frozen transferable parameters + the selected iteration's result entry.

    ``frozen_parameters.json`` is the fit's own export of exactly the fields a
    frozen forward needs (it includes ``ramp_matrix``, which ``result.json``
    does not). ``result.json`` carries the selected iteration and its metrics.
    """
    frozen_path = fit_dir / "frozen_parameters.json"
    result_path = fit_dir / "result.json"
    if not frozen_path.is_file():
        raise SystemExit(f"missing {frozen_path}")
    if not result_path.is_file():
        raise SystemExit(f"missing {result_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for key in ("gain", "background", "bias_counts", "defocus_diopter", "origin"):
        if key not in frozen:
            raise SystemExit(f"{frozen_path} lacks {key!r}")
    return frozen, result


def build_transfer_model(dataset: Path, fit_dir: Path, span: int, device: str | None, *,
                         origin_mode: str = "measured", origin_offset: np.ndarray | None = None,
                         static_term: bool = True):
    """Fresh forward on ``dataset`` carrying ``fit_dir``'s physical parameters.

    ``build_fit_model`` derives the measured origin and the ramp matrix from the
    dataset it is pointed at, so this is the supported way to move a fit onto a
    session with a different reference spot. The origin is then re-anchored:
    ``measured`` keeps the dataset's own reference spot, ``session`` adds the
    fit's fitted offset ``origin_fit - measured_old`` (``origin_offset``), and
    ``fitted`` restores the fitted spot verbatim (used only for validation on
    the original dataset). Nothing here trains.
    """
    import torch
    from ao_shaping.tools.slm.transfer.fit import build_fit_model

    frozen, _ = load_fit(fit_dir)
    # Master ``build_fit_model`` always warm-starts residuals at Noll modes
    # 5..11, carries no static basis and never learns sigma_ccd (it is taken
    # from the checkpoint), so the mil-only kwargs ``residual_modes`` /
    # ``static_basis_k`` / ``static_mode`` / ``learn_sigma_ccd`` are not passed.
    # The declared static basis is still read defensively so a mil-era
    # checkpoint loads; it is reported, but the master forward cannot apply it
    # (see the module docstring).
    static_k = int(frozen.get("static_basis_k", 0) or 0) if static_term else 0
    comps = build_fit_model(
        dataset,
        fit_dir,
        span=span,
        device=device,
        adaptation="full",
    )
    if tuple(getattr(comps.model, "residual_modes", ())) != RESIDUAL_MODES:
        raise SystemExit(
            f"unexpected residual modes: {getattr(comps.model, 'residual_modes', None)}; "
            f"expected {RESIDUAL_MODES}")
    if static_k:
        print(
            f"WARNING: {fit_dir} declares static_basis_k={static_k}; the master "
            "forward has no static term, so it is reported but NOT applied",
            file=sys.stderr,
        )
    if origin_mode not in ("measured", "session", "fitted"):
        raise SystemExit(f"unknown origin_mode {origin_mode!r}")
    with torch.no_grad():
        if origin_mode == "fitted":
            origin = np.asarray(frozen["origin"], dtype=float)
        elif origin_mode == "session":
            if origin_offset is None:
                raise SystemExit("origin_mode='session' needs origin_offset")
            origin = np.asarray(comps.measured_origin, dtype=float) + np.asarray(origin_offset)
        else:
            origin = None
        if origin is not None:
            comps.model.origin.copy_(torch.as_tensor(origin, device=comps.model.origin.device))
    if getattr(comps, "static_coef", None) is not None:
        coefficients = frozen.get("static_coef", frozen.get("static_gain"))
        if coefficients is None:
            raise SystemExit(f"{fit_dir} has static basis k={static_k} but no coefficients")
        coefficients = np.asarray(coefficients, dtype=np.float32)
        with torch.no_grad():
            comps.static_coef.copy_(torch.as_tensor(coefficients, device=comps.static_coef.device))
    return comps, frozen


def predict_counts(comps, frozen: dict) -> np.ndarray:
    """One forward pass in measured counts, mirroring ``fit_loss``'s post-chain.

    Order is fixed by ``fit.fit_loss``: optical prediction, CCD blur kernel,
    static term, then ``gain*optical + background + bias_counts/exposure``.
    The master ``FitComponents`` carries the frozen blaze ``kernel`` only (no
    trainable blur term) and no static basis, so the static branch runs only for
    a compatibility object that actually exposes ``static_coef``.
    The radiometry is the ORIGINAL fit's, not refitted on this session.
    """
    import torch
    from torch.nn.functional import conv2d

    with torch.no_grad():
        prediction = comps.model.prediction()
        kernel = comps.kernel
        if kernel is not None:
            prediction = conv2d(prediction[:, None], kernel, padding=12)[:, 0]
        static_coef = getattr(comps, "static_coef", None)
        if static_coef is not None:
            static_basis = comps.static_basis
            if getattr(comps, "static_mode", "global") == "global":
                prediction = prediction + torch.einsum(
                    "k,khw->hw", static_coef, static_basis)
            else:
                prediction = prediction + torch.einsum(
                    "nk,khw->nhw", static_coef, static_basis)
        optical = prediction.detach().cpu().numpy().astype(np.float64)
    exposure = np.asarray(comps.exposure_values, dtype=float)[:, None, None]
    return (optical * float(frozen["gain"]) + float(frozen["background"])
            + float(frozen["bias_counts"]) / exposure)


def drop_static_term(comps) -> None:
    """Zero the static coefficients in place (pure physical forward)."""
    if getattr(comps, "static_coef", None) is None:
        return
    import torch
    with torch.no_grad():
        comps.static_coef.zero_()


def mean_metrics(rows: list[dict], mask: np.ndarray) -> dict:
    """Mean of the per-row ``fit.metrics`` entries over ``mask``."""
    selected = [rows[i] for i in np.flatnonzero(mask)]
    if not selected:
        return {"n_rows": 0}
    out = {"n_rows": len(selected)}
    for name in ("correlation", "relative_l2", "cv_abs_error", "cv_prediction", "cv_measured"):
        out[name] = float(np.mean([row["metrics"][name] for row in selected]))
    return out


def metric_block(rows: list[dict], splits: np.ndarray, families: np.ndarray) -> dict:
    """Grouped metrics by split, by family and test-only by family."""
    by_split = {}
    for group in SPLIT_GROUPS:
        mask = np.ones(len(rows), bool) if group == "all" else splits == group
        by_split[group] = mean_metrics(rows, mask)
    is_probe = families != "calibration"
    by_family = {family: mean_metrics(rows, is_probe & (families == family))
                 for family in FAMILIES}
    test_by_family = {
        family: mean_metrics(rows, (splits == "test") & (families == family))
        for family in FAMILIES
    }
    return {"by_split": by_split, "by_family": by_family, "test_by_family": test_by_family}


def score_rows(labels, splits, families, prediction, measured) -> list[dict]:
    """Per-row metrics from the fit's own metric function."""
    from ao_shaping.tools.slm.transfer.fit import metrics

    scored = metrics(prediction, measured)
    return [
        {"label": str(label), "split": str(split), "family": str(family), "metrics": entry}
        for label, split, family, entry in zip(labels, splits, families, scored)
    ]


def scalar_gain(prediction: np.ndarray, measured: np.ndarray, mask: np.ndarray) -> float:
    """Least-squares scalar ``g = <pred, meas> / <pred, pred>`` on ``mask`` rows."""
    p = np.asarray(prediction[mask], dtype=np.float64).ravel()
    m = np.asarray(measured[mask], dtype=np.float64).ravel()
    denominator = float(np.dot(p, p))
    if denominator <= 0:
        raise SystemExit("degenerate recalibration: zero predicted energy")
    return float(np.dot(p, m) / denominator)


def validate_rebuild(original: Path, fit_dir: Path, frozen: dict, result: dict,
                     span: int, device: str | None):
    """Rebuild the ORIGINAL fit's stored prediction; this proves the path.

    Both the stored array (``iteration_XXXX.npz``) and the fit's own per-row
    ``result.json`` metrics are compared against the rebuilt forward. A model
    whose static coefficients cannot be reproduced fails here rather than
    silently evaluating the new session with the wrong observation model.
    """
    from ao_shaping.tools.slm.transfer.fit import metrics

    comps, _ = build_transfer_model(original, fit_dir, span, device, origin_mode="fitted")
    rebuilt = predict_counts(comps, frozen)
    iteration = int(result["iteration"])
    stored_path = fit_dir / f"iteration_{iteration:04d}.npz"
    if not stored_path.is_file():
        raise SystemExit(f"missing stored prediction {stored_path}")
    with np.load(stored_path) as payload:
        stored = np.asarray(payload["prediction"], dtype=np.float64)
    if stored.shape != rebuilt.shape:
        raise SystemExit(f"{stored_path} shape {stored.shape} != rebuilt {rebuilt.shape}")
    deviation = np.abs(rebuilt - stored)
    corr = float(np.corrcoef(rebuilt.ravel(), stored.ravel())[0, 1])
    rebuilt_metrics = metrics(rebuilt, comps.images)
    stored_metrics = result["metrics"]
    metric_dev = {
        name: float(np.max(np.abs(
            np.asarray([m[name] for m in rebuilt_metrics])
            - np.asarray([m[name] for m in stored_metrics]))))
        for name in ("correlation", "relative_l2", "cv_abs_error")
    }
    measured_origin = np.asarray(comps.measured_origin, dtype=float)
    fitted_origin = np.asarray(frozen["origin"], dtype=float)
    report = {
        "iteration": iteration,
        "stored_prediction": str(stored_path),
        "rows": int(stored.shape[0]),
        "max_abs_deviation_counts": float(deviation.max()),
        "relative_l2_deviation": float(np.linalg.norm(deviation.ravel())
                                       / max(float(np.linalg.norm(stored.ravel())), 1e-12)),
        "correlation": corr,
        "max_metric_deviation": metric_dev,
        "tolerance_counts": VALIDATION_MAX_ABS_DEV_COUNTS,
        "tolerance_relative_l2": VALIDATION_MAX_RELATIVE_DEV,
        "tolerance_correlation": VALIDATION_MIN_CORR,
        "measured_origin": measured_origin.tolist(),
        "fitted_origin": fitted_origin.tolist(),
        "fitted_origin_offset": (fitted_origin - measured_origin).tolist(),
        "ramp_matrix": np.asarray(comps.matrix).tolist(),
    }
    report["passed"] = bool(
        report["max_abs_deviation_counts"] <= VALIDATION_MAX_ABS_DEV_COUNTS
        and report["relative_l2_deviation"] <= VALIDATION_MAX_RELATIVE_DEV
        and corr >= VALIDATION_MIN_CORR
    )
    return report, comps


def evaluate_variant(comps, frozen: dict, measured: np.ndarray, recalibrate_split: str,
                     *, static_term: bool) -> tuple[dict, np.ndarray]:
    """Score one observation-model variant before/after the scalar gain."""
    labels = [str(row["label"]) for row in comps.rows]
    splits = np.asarray([str(row["split"]) for row in comps.rows])
    families = np.asarray([probe_family(label) for label in labels])
    raw = predict_counts(comps, frozen)
    train_mask = splits == recalibrate_split
    if not train_mask.any():
        raise SystemExit(f"new dataset has no {recalibrate_split!r} rows to recalibrate on")
    gain = scalar_gain(raw, measured, train_mask)
    before = score_rows(labels, splits, families, raw, measured)
    after = score_rows(labels, splits, families, raw * gain, measured)
    return {
        "static_term": static_term,
        "recalibration": {
            "split": recalibrate_split,
            "n_rows": int(train_mask.sum()),
            "gain": gain,
            "scale_shift": gain - 1.0,
            "note": "one scalar applied to the predicted image; corr is invariant, "
                    "relative_l2/cv_prediction scale with it",
        },
        "metrics": {
            "before": metric_block(before, splits, families),
            "after": metric_block(after, splits, families),
        },
        "rows": [
            {"label": b["label"], "split": b["split"], "family": b["family"],
             "before": b["metrics"], "after": a["metrics"]}
            for b, a in zip(before, after)
        ],
    }, raw * gain


def evaluate_fit(name: str, fit_dir: Path, new_dir: Path, span: int, device: str | None,
                 recalibrate_split: str, validation: dict):
    """Transfer one fit to the new session, with every reported variant."""
    offset = np.asarray(validation["fitted_origin_offset"], dtype=float)
    measured = None
    variants = {}
    arrays = {}
    for origin_mode in ORIGIN_MODES:
        comps, frozen = build_transfer_model(
            new_dir, fit_dir, span, device, origin_mode=origin_mode, origin_offset=offset)
        if measured is None:
            measured = np.asarray(comps.images, dtype=np.float64)
        static_here = getattr(comps, "static_coef", None) is not None
        variant, prediction = evaluate_variant(
            comps, frozen, measured, recalibrate_split, static_term=static_here)
        variants[f"origin_{origin_mode}"] = variant
        arrays[f"origin_{origin_mode}"] = prediction
        if static_here:
            drop_static_term(comps)
            static_off, static_pred = evaluate_variant(
                comps, frozen, measured, recalibrate_split, static_term=False)
            variants[f"origin_{origin_mode}_static_off"] = static_off
            arrays[f"origin_{origin_mode}_static_off"] = static_pred

    primary = variants["origin_session"]
    rows = primary["rows"]
    labels = [row["label"] for row in rows]
    splits = np.asarray([row["split"] for row in rows])
    families = np.asarray([row["family"] for row in rows])
    _, result = load_fit(fit_dir)
    comps, frozen = build_transfer_model(
        new_dir, fit_dir, span, device, origin_mode="session", origin_offset=offset)
    summary = {
        "fit": name,
        "fit_dir": str(fit_dir),
        "original_iteration": int(result["iteration"]),
        "validation": validation,
        "origin_modes": list(ORIGIN_MODES),
        "primary_variant": "origin_session",
        "transferred_parameters": {
            key: frozen.get(key)
            for key in ("defocus_diopter", "amplitude_center", "amplitude_width",
                        "residual_noll_radians", "sigma_ccd_px", "gain", "background",
                        "bias_counts", "registration_shift_ccd_px", "grid_size",
                        "fft_size", "pixel_samples")
        },
        "static": {
            "static_basis_k": int(frozen.get("static_basis_k", 0)),
            "static_mode": frozen.get("static_mode"),
            "transferred_static_coef": frozen.get("static_coef"),
        },
        "new_session_geometry": {
            "measured_origin": np.asarray(comps.measured_origin).tolist(),
            "origin_used": np.asarray(comps.model.origin.detach().cpu()).tolist(),
            "ramp_matrix": np.asarray(comps.matrix).tolist(),
            "ramp_samples": comps.ramp_samples,
            "span": int(span),
        },
        "variants": variants,
        "row_index": {
            "labels": labels,
            "split": splits.tolist(),
            "family": families.tolist(),
            "note": "row order of the prediction arrays on disk",
        },
    }
    return summary, arrays, measured, splits, families


def build_verdict(models: dict) -> dict:
    """Plain deltas: static_global minus baseline, after recalibration."""
    names = list(models)
    baseline = next((name for name in names if "static" not in name), None)
    statics = [name for name in names if "static" in name]
    if baseline is None or not statics:
        return {"baseline": baseline, "static": statics, "deltas": None,
                "note": "needs one baseline and one static fit for deltas"}

    def block(model: dict, variant: str, phase: str, key: str) -> dict:
        return model["variants"][variant]["metrics"][phase]["by_split"][key]

    def family(model: dict, variant: str, phase: str, key: str) -> dict:
        return model["variants"][variant]["metrics"][phase]["test_by_family"][key]

    verdict = {"baseline": baseline, "static": statics, "deltas": {}}
    for static_name in statics:
        deltas = {}
        for variant in ("origin_session", "origin_session_static_off"):
            if variant not in models[static_name]["variants"]:
                continue
            deltas[variant] = {}
            for phase in ("before", "after"):
                for split in ("all", "train", "test"):
                    b = block(models[baseline], "origin_session", phase, split)
                    s = block(models[static_name], variant, phase, split)
                    deltas[variant][f"{phase}.{split}"] = {
                        "relative_l2": s["relative_l2"] - b["relative_l2"],
                        "correlation": s["correlation"] - b["correlation"],
                        "cv_abs_error": s["cv_abs_error"] - b["cv_abs_error"],
                    }
                for fam in FAMILIES:
                    b = family(models[baseline], "origin_session", phase, fam)
                    s = family(models[static_name], variant, phase, fam)
                    deltas[variant][f"{phase}.test.{fam}"] = {
                        "relative_l2": s["relative_l2"] - b["relative_l2"],
                        "correlation": s["correlation"] - b["correlation"],
                    }
    verdict["deltas"] = deltas
    verdict["origin_sensitivity"] = {
        name: {
            f"{phase}.{key}": {
                "relative_l2": block(model, "origin_measured", phase, key)["relative_l2"]
                - block(model, "origin_session", phase, key)["relative_l2"],
                "correlation": block(model, "origin_measured", phase, key)["correlation"]
                - block(model, "origin_session", phase, key)["correlation"],
            }
            for phase in ("before", "after") for key in ("all", "test")
        }
        for name, model in models.items()
    }
    return verdict


def print_summary(payload: dict, out: Path) -> None:
    print(f"\nvalidation: {'PASS' if payload['validation']['all_passed'] else 'FAIL'} "
          f"(max|dp|<={payload['validation']['tolerance_counts']:g} counts)")
    for name, model in payload["models"].items():
        print(f"\n== {name}  (original it={model['original_iteration']}, "
              f"static_k={model['static']['static_basis_k']})")
        for variant in ("origin_session", "origin_measured", "origin_session_static_off"):
            if variant not in model["variants"]:
                continue
            v = model["variants"][variant]
            rec = v["recalibration"]
            print(f"  [{variant}] gain {rec['gain']:.6g} (scale shift {rec['scale_shift']:+.3f})")
            for phase in ("before", "after"):
                bb = v["metrics"][phase]
                parts = []
                for split in ("all", "train", "test"):
                    m = bb["by_split"][split]
                    parts.append(f"{split} rl2={m.get('relative_l2', float('nan')):.4f} "
                                 f"corr={m.get('correlation', float('nan')):.4f} "
                                 f"cv_abs={m.get('cv_abs_error', float('nan')):.4f}")
                print(f"    {phase:6s} " + " | ".join(parts))
                fams = bb["test_by_family"]
                print("           test fam " + " ".join(
                    f"{fam}:{fams[fam].get('relative_l2', float('nan')):.3f}"
                    for fam in FAMILIES))

    verdict = payload.get("verdict") or {}
    if verdict.get("deltas"):
        print("\n== deltas (static - baseline), after recalibration, primary origin")
        for variant, deltas in verdict["deltas"].items():
            for split in ("test", "all"):
                d = deltas[f"after.{split}"]
                print(f"  {variant} {split}: d_rl2={d['relative_l2']:+.4f} "
                      f"d_corr={d['correlation']:+.4f} d_cv_abs={d['cv_abs_error']:+.4f}")
    if verdict.get("origin_sensitivity"):
        print("\n== origin sensitivity (measured - session)")
        for name, block in verdict["origin_sensitivity"].items():
            for key in ("after.test", "after.all"):
                d = block[key]
                print(f"  {name} {key}: d_rl2={d['relative_l2']:+.4f} "
                      f"d_corr={d['correlation']:+.4f}")
    print(f"\nwrote {out / 'results.json'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate ORIGINAL-session physical fits on a NEW session "
                    "(ramp re-derived, physical params transferred, origin "
                    "re-anchored, one scalar gain recalibrated on the new train split).")
    parser.add_argument("--original", required=True, type=Path,
                        help="dataset the fits were trained on")
    parser.add_argument("--new", required=True, type=Path,
                        help="evaluation dataset (different phases and reference spot)")
    parser.add_argument("--fit", action="append", required=True,
                        help="fit directory, or a name inside --original (repeatable)")
    parser.add_argument("--out", required=True, type=Path,
                        help="output directory for results.json and prediction arrays")
    parser.add_argument("--recalibrate-split", default="train", choices=("train",),
                        help="new-dataset split used for the scalar gain (default: train)")
    parser.add_argument("--span", type=int, default=DEFAULT_SPAN,
                        help=f"camera window half-width (default: {DEFAULT_SPAN})")
    parser.add_argument("--device", default=None,
                        help="torch device for the forward ('cpu', 'cuda', or omitted for auto)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="evaluate the new session even if the original-rebuild "
                             "check fails (NOT the default: the check is the proof)")
    args = parser.parse_args(argv)

    original = args.original.resolve()
    new_dir = args.new.resolve()
    for path, flag in ((original, "--original"), (new_dir, "--new")):
        if not (path / "manifest.json").is_file():
            raise SystemExit(f"{flag} has no manifest.json: {path}")

    fit_dirs = [(spec, resolve_fit(original, spec)) for spec in args.fit]
    out = args.out.resolve()

    validation_all = {}
    for spec, fit_dir in fit_dirs:
        frozen, result = load_fit(fit_dir)
        report, _ = validate_rebuild(original, fit_dir, frozen, result, args.span, args.device)
        validation_all[fit_dir.name] = report
        print(f"[validate] {fit_dir.name}: max|dp|={report['max_abs_deviation_counts']:.3e} "
              f"counts, rel={report['relative_l2_deviation']:.2e}, "
              f"corr={report['correlation']:.8f}, "
              f"{'PASS' if report['passed'] else 'FAIL'}")
        if not report["passed"] and not args.skip_validation:
            print("[validate] rebuild of the original prediction failed; "
                  "refusing to evaluate the new session (use --skip-validation to override)",
                  file=sys.stderr)
            return 3

    models = {}
    arrays_by_fit = {}
    splits = families = None
    measured = None
    for spec, fit_dir in fit_dirs:
        name = fit_dir.name
        summary, arrays, measured, splits, families = evaluate_fit(
            name, fit_dir, new_dir, args.span, args.device,
            args.recalibrate_split, validation_all[name])
        models[name] = summary
        arrays_by_fit[name] = arrays

    payload = {
        "kind": "transfer_cross_session_evaluation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "original": {
            "dataset": str(original),
            "manifest_sha256": sha256_file(original / "manifest.json"),
        },
        "new": {
            "dataset": str(new_dir),
            "manifest_sha256": sha256_file(new_dir / "manifest.json"),
            "row_counts": {split: int((splits == split).sum()) for split in SPLIT_GROUPS[1:]},
            "family_counts": {
                family: int(((families != "calibration") & (families == family)).sum())
                for family in FAMILIES
            },
        },
        "validation": {
            "required": not args.skip_validation,
            "all_passed": all(report["passed"] for report in validation_all.values()),
            "tolerance_counts": VALIDATION_MAX_ABS_DEV_COUNTS,
            "tolerance_relative_l2": VALIDATION_MAX_RELATIVE_DEV,
            "tolerance_correlation": VALIDATION_MIN_CORR,
            "models": validation_all,
        },
        "models": models,
        "verdict": build_verdict(models),
    }

    out.mkdir(parents=True, exist_ok=True)
    for name, arrays in arrays_by_fit.items():
        for variant, prediction in arrays.items():
            np.savez_compressed(
                out / f"predictions_{name}_{variant}.npz",
                prediction=prediction.astype(np.float32),
                measured=measured.astype(np.float32),
                label=np.asarray(models[name]["row_index"]["labels"]),
                split=splits, family=families)
    (out / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(payload, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
