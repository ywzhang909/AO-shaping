# -*- coding: utf-8 -*-
"""Opt-in per-session refit of the physical forward model (offline, no device).

What it does
------------
Starts from an existing ``frozen_parameters.json`` (the transferred checkpoint)
and re-fits the physical forward on **one session's train rows only**, then
writes a loadable refit checkpoint plus the held-out metrics of both checkpoints
and the provenance that says how they were produced.

Why it exists
-------------
A per-session Gaussian refit measurably improves held-out forward prediction,
but it was only reachable through diagnostic scripts. This entry point makes it
usable by the existing checkpoint-driven pipeline without changing any default::

    python scripts/slm/refit_transfer_forward.py \
        --dataset data/debug/slm_transfer/<session> \
        --checkpoint <transfer>/frozen_parameters.json \
        --out data/debug/slm_transfer/<session>/refit_forward

    # then, unchanged:
    python -m ao_shaping.tools.slm.transfer.validate \
        --checkpoint data/debug/slm_transfer/<session>/refit_forward/frozen_parameters.json ...

Guarantees
----------
* **Train rows only.** The fit goes through
  :func:`ao_shaping.tools.slm.transfer.fit.fit` with the historical split, so
  the loss mask is exactly ``split == "train"``. Calibration rows never enter
  it, test rows are never fitted, and both facts are asserted on the assembled
  mask before anything is written (see :func:`verify_split_mask`).
* **One code path.** The checkpoint is produced by production ``fit()``, not by
  a copy of the loop, so ``frozen_parameters.json`` is bit-for-bit the artefact
  the rest of the pipeline already knows how to load.
* **Held-out scoring, once.** Base and refit checkpoints are scored on the same
  rows with the same forward (``PhysicalForwardFit.prediction`` + the frozen
  blaze ``kernel`` conv), so the delta is the refit's, not a scoring change.
* **Nothing outside ``--out``.** The dataset and the base checkpoint are only
  read, and nothing is written into ``report/``.

``--dry-run`` builds the model, evaluates one loss and stops: no optimizer step,
no directory, no checkpoint.

Exit status is non-zero only on a hard error (``SystemExit`` with a message).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

KIND = "transfer_forward_refit"
SCRIPT = Path(__file__).resolve()
FIT_MODULE = "ao_shaping.tools.slm.transfer.fit"
FITTER = "ao_shaping.algorithm.signal_processing.physical_forward_fit.PhysicalForwardFit"
FORWARD_MODULE = "ao_shaping.algorithm.signal_processing.physical_forward_fit"
RECOGNIZED_SPLITS = ("train", "test", "validation", "calibration")
SCORED_SPLITS = ("train", "test", "validation")
METRIC_KEYS = ("correlation", "relative_l2", "cv_abs_error")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def source_record(module: str) -> dict:
    """Module path + sha256 without importing it (no torch, no device code)."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - install issue
        raise SystemExit(f"cannot locate module {module}: {error}") from error
    if spec is None or spec.origin is None:  # pragma: no cover - install issue
        raise SystemExit(f"cannot locate source file for module {module}")
    path = Path(spec.origin).resolve()
    return {"module": module, "path": str(path), "sha256": sha256_file(path)}


def read_manifest(dataset: Path) -> list[dict]:
    manifest = dataset / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"dataset has no manifest.json: {dataset}")
    if not (dataset / "config.json").is_file():
        raise SystemExit(f"dataset has no config.json: {dataset}")
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"manifest is not a non-empty row list: {manifest}")
    return rows


def parse_expected_splits(text: str | None) -> dict[str, int] | None:
    """``"train=80,test=24,calibration=10"`` -> ``{"train": 80, ...}``."""
    if text is None or not str(text).strip():
        return None
    expected: dict[str, int] = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--expect-splits entry {item!r} is not NAME=COUNT")
        name, count = (part.strip() for part in item.split("=", 1))
        if name not in RECOGNIZED_SPLITS:
            raise SystemExit(
                f"--expect-splits names unknown split {name!r}; "
                f"choose from {', '.join(RECOGNIZED_SPLITS)}")
        try:
            expected[name] = int(count)
        except ValueError as error:
            raise SystemExit(f"--expect-splits count for {name!r} is not an integer: {count!r}") from error
    return expected or None


def parse_score_splits(text: str) -> list[str]:
    names = [part.strip() for part in str(text).split(",") if part.strip()]
    if not names:
        raise SystemExit("--score-splits is empty")
    for name in names:
        if name not in SCORED_SPLITS:
            raise SystemExit(
                f"--score-splits may only name {', '.join(SCORED_SPLITS)} "
                f"(calibration rows are never scored); got {name!r}")
    return names


def split_plan(rows: list[dict], expected: dict[str, int] | None = None) -> dict:
    """Count splits and refuse anything that could leak into the train mask.

    The fit mask is ``split == "train"`` over the selected rows, so the only way
    a calibration row can enter it is a mislabelled or duplicated manifest row.
    Both are refused here, before any heavy I/O.
    """
    for row in rows:
        split = row.get("split")
        if split not in RECOGNIZED_SPLITS:
            raise SystemExit(
                f"manifest row {row.get('label')!r} has split {split!r}; "
                f"expected one of {', '.join(RECOGNIZED_SPLITS)}")
    counts = dict(Counter(row["split"] for row in rows))
    train_rows = counts.get("train", 0)
    if train_rows <= 0:
        raise SystemExit("manifest has no train rows; nothing to refit on")
    calibration = counts.get("calibration", 0)
    files = Counter(row.get("file") for row in rows if row.get("file"))
    duplicated = sorted(name for name, count in files.items() if count > 1)
    if duplicated:
        raise SystemExit(
            "manifest references the same payload more than once "
            f"({duplicated[:5]}); a duplicated calibration/test row could enter the train mask")
    if expected is not None:
        for name, count in expected.items():
            actual = counts.get(name, 0)
            if actual != count:
                raise SystemExit(
                    f"--expect-splits mismatch for {name!r}: manifest has {actual}, expected {count} "
                    f"(manifest counts {counts})")
    selected = sum(counts.get(name, 0) for name in SCORED_SPLITS)
    logger.info(
        f"split counts {counts}; train={train_rows} selected(train+test+validation)={selected} "
        f"calibration={calibration} (excluded from the fit mask)")
    return {"counts": counts, "train": train_rows, "selected": selected,
            "calibration": calibration, "expected": expected}


def verify_split_mask(comps: Any, plan: dict) -> None:
    """Assert the assembled mask is train-only and calibration-free."""
    splits = [row.get("split") for row in comps.rows]
    leaked = sorted({split for split in splits if split not in SCORED_SPLITS})
    if leaked:
        raise SystemExit(f"non-train/test/validation rows reached the fit selection: {leaked}")
    if len(comps.rows) > plan["selected"]:
        raise SystemExit(
            f"assembled rows {len(comps.rows)} exceed the {plan['selected']} selected manifest rows")
    mask = [bool(flag) for flag in comps.train]
    if len(mask) != len(splits):
        raise SystemExit(f"train mask length {len(mask)} != assembled rows {len(splits)}")
    offenders = [splits[index] for index, flag in enumerate(mask) if flag
                 and splits[index] != "train"]
    if offenders:
        raise SystemExit(
            f"train mask would include non-train rows: {sorted(set(offenders))}; refusing")
    train_rows = sum(1 for split in splits if split == "train")
    if train_rows <= 0:
        raise SystemExit("no train row survived into the fit selection")
    if int(sum(mask)) != train_rows:
        raise SystemExit(
            f"train mask has {int(sum(mask))} rows but the assembled selection has "
            f"{train_rows} train rows")
    # Rows dropped by ``valid()`` change the count, so the provenance records the
    # rows the refit actually saw rather than the manifest total.
    plan["fit_train"] = train_rows
    plan["fit_rows"] = len(splits)


# ---------------------------------------------------------------------------
# model / scoring (logic mirrored from
# scripts/slm/diagnose_complex_illumination_b.py, adapted to this API)
# ---------------------------------------------------------------------------
def build_components(dataset: Path, checkpoint: Path, args: argparse.Namespace, fitmod: Any) -> Any:
    """Assemble a model at ``checkpoint``.

    ``build_fit_model`` only accepts ``alignment``/``full``; with ``--adaptation
    auto`` the scoring model uses ``full``. That choice only sets
    ``requires_grad`` flags, so the predicted images are identical either way.
    """
    logger.info(f"building fit model from {checkpoint.name} on {dataset.name}")
    return fitmod.build_fit_model(
        dataset, checkpoint, span=args.span, device=args.device,
        adaptation="full" if args.adaptation == "auto" else args.adaptation,
        amplitude_center_lr=args.amplitude_center_lr, radiometry=args.radiometry)


def optical_prediction(comps: Any) -> Any:
    """Optical irradiance after the CCD blur, before radiometry.

    Mirrors ``fit.fit_loss``: the frozen blaze ``kernel`` (when present) is
    convolved with the same zero padding the production loss uses.
    """
    import torch
    from torch.nn.functional import conv2d

    with torch.no_grad():
        prediction = comps.model.prediction()
        if comps.kernel is not None:
            prediction = conv2d(prediction[:, None], comps.kernel, padding=12)[:, 0]
    return prediction.detach().cpu().numpy()


def score_splits(comps: Any, optical: Any, fitmod: Any, names: list[str],
                 radiometry: str = "exposure") -> dict:
    """Average ``fit.metrics`` per split, with train-only radiometry.

    The same treatment is applied to the base and the refit checkpoint, so the
    comparison is like for like; the split's predicted counts borrow only train
    coefficients.
    """
    import numpy as np

    from ao_shaping.tools.slm.transfer.radiometry import fit_radiometry

    splits = np.asarray([row.get("split") for row in comps.rows], dtype=object)
    exposure = np.asarray(comps.exposure_values, dtype=float)
    measured = np.asarray(comps.images, dtype=float)
    _, radiometry_fit = fit_radiometry(optical, measured, exposure, comps.train,
                                       mode=radiometry)
    counts = (optical * radiometry_fit["gain"] + radiometry_fit["background"]
              + radiometry_fit["bias_counts"] / exposure[:, None, None])
    results: dict[str, dict] = {}
    for name in names:
        mask = splits == name
        if not mask.any():
            raise SystemExit(f"--score-splits asks for {name!r}, which has no rows in this dataset")
        per_row = fitmod.metrics(counts[mask], measured[mask])
        results[name] = {key: float(np.mean([item[key] for item in per_row]))
                         for key in METRIC_KEYS}
        results[name]["radiometry_fit"] = {
            "gain": float(radiometry_fit["gain"]),
            "background": float(radiometry_fit["background"]),
            "bias_counts": float(radiometry_fit["bias_counts"]),
        }
    return results


# ---------------------------------------------------------------------------
# the two expensive seams (kept as functions so tests can replace them)
# ---------------------------------------------------------------------------
def run_fit(args: argparse.Namespace, fitmod: Any) -> None:
    """Call production ``fit()`` with exactly the Namespace it expects."""
    namespace = argparse.Namespace(
        iterations=args.iterations,
        patience=args.patience,
        min_delta=args.min_delta,
        out=args.out,
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        span=args.span,
        adaptation=args.adaptation,
        amplitude_center_lr=args.amplitude_center_lr,
        radiometry=args.radiometry,
        record_responses=False,
        device=args.device,
    )
    fitmod.fit(namespace)


def loss_once(comps: Any, fitmod: Any, radiometry: str) -> float:
    _, entry, _ = fitmod.fit_loss(comps.model, comps, radiometry_mode=radiometry, iteration=0)
    return float(entry["loss"])


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def build_provenance(*, args: argparse.Namespace, plan: dict,
                     base_metrics: dict, refit_metrics: dict,
                     scored_split: str) -> dict:
    base_checkpoint = args.base_checkpoint
    refit_checkpoint = args.out / "frozen_parameters.json"
    beats = (refit_metrics[scored_split]["correlation"] >= base_metrics[scored_split]["correlation"]
             and refit_metrics[scored_split]["relative_l2"] <= base_metrics[scored_split]["relative_l2"])
    return {
        "kind": KIND,
        "generated": utc_now(),
        "script": str(SCRIPT),
        "script_sha256": sha256_file(SCRIPT),
        "dataset": str(args.dataset),
        "manifest_sha256": sha256_file(args.dataset / "manifest.json"),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "refit_checkpoint": str(refit_checkpoint),
        "refit_checkpoint_sha256": sha256_file(refit_checkpoint),
        "span": int(args.span),
        "adaptation": args.adaptation,
        "radiometry": args.radiometry,
        "iterations": int(args.iterations),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "device": args.device,
        "score_splits": list(args.score_splits),
        "split_counts": plan["counts"],
        "expect_splits": plan["expected"],
        "train_rows": int(plan.get("fit_train", plan["train"])),
        "fit_rows": plan.get("fit_rows"),
        "calibration_excluded": True,
        "test_scored_once": True,
        "forward_api": {
            "fitter": FITTER,
            "prediction": "PhysicalForwardFit.prediction() then conv2d(kernel, padding=12)",
            "sources": [source_record(FORWARD_MODULE), source_record(FIT_MODULE)],
        },
        "metrics": {"base": base_metrics, "refit": refit_metrics},
        # These two are recorded, not assumed: verify_split_mask() ran on both
        # assembled selections and raises SystemExit instead of returning when a
        # calibration/test row is in the mask or the train count disagrees.
        "gates": {
            "train_only_fit": True,
            "calibration_excluded": True,
            "split_counts_match": plan["expected"] is None or all(
                plan["counts"].get(name, 0) == count
                for name, count in plan["expected"].items()),
            "refit_beats_base_on_score_split": bool(beats),
            "scored_split": scored_split,
        },
    }


# ---------------------------------------------------------------------------
# flow
# ---------------------------------------------------------------------------
def validate_args(args: argparse.Namespace) -> None:
    if args.iterations < 0:
        raise SystemExit("--iterations must be >= 0")
    if args.patience <= 0:
        raise SystemExit("--patience must be > 0")
    if args.min_delta < 0:
        raise SystemExit("--min-delta must be >= 0")
    if args.span < 20 or args.span % 2:
        raise SystemExit("--span must be even and >= 20")
    if args.amplitude_center_lr is not None and args.amplitude_center_lr <= 0:
        raise SystemExit("--amplitude-center-lr must be > 0")
    if not args.dataset.is_dir():
        raise SystemExit(f"--dataset is not a directory: {args.dataset}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"--checkpoint does not exist: {args.checkpoint}")
    if not args.dry_run and args.out.exists():
        raise SystemExit(f"--out already exists; refusing to overwrite: {args.out}")


def run(args: argparse.Namespace) -> int:
    from ao_shaping.tools.slm.transfer import fit as fitmod

    args.dataset = Path(args.dataset).resolve()
    args.checkpoint = Path(args.checkpoint).resolve()
    args.out = Path(args.out).resolve()
    args.base_checkpoint = args.checkpoint
    args.score_splits = parse_score_splits(args.score_splits)
    validate_args(args)

    rows = read_manifest(args.dataset)
    plan = split_plan(rows, parse_expected_splits(args.expect_splits))

    # Built and asserted before anything is fitted: the mask the refit will use
    # is the one this model carries, so a calibration/test leak stops the run
    # here rather than after the optimizer has already touched parameters.
    base_comps = build_components(args.dataset, args.checkpoint, args, fitmod)
    verify_split_mask(base_comps, plan)

    if args.dry_run:
        loss = loss_once(base_comps, fitmod, args.radiometry)
        print(f"dry run: train rows={plan.get('fit_train', plan['train'])} loss={loss:.6f}; "
              f"no optimizer step, no checkpoint written")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_fit(args, fitmod)
    refit = args.out / "frozen_parameters.json"
    if not refit.is_file():
        raise SystemExit(f"fit finished without writing {refit}")

    base_metrics = score_splits(base_comps, optical_prediction(base_comps), fitmod,
                                args.score_splits, args.radiometry)

    refit_comps = build_components(args.dataset, refit, args, fitmod)
    verify_split_mask(refit_comps, plan)
    refit_metrics = score_splits(refit_comps, optical_prediction(refit_comps), fitmod,
                                 args.score_splits, args.radiometry)

    scored_split = args.score_splits[0]
    provenance = build_provenance(args=args, plan=plan, base_metrics=base_metrics,
                                 refit_metrics=refit_metrics, scored_split=scored_split)
    (args.out / "refit_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    base, refit_row = base_metrics[scored_split], refit_metrics[scored_split]
    print(f"refit checkpoint: {refit}")
    print(f"{scored_split}: base correlation={base['correlation']:.4f} "
          f"relative_l2={base['relative_l2']:.4f} | refit correlation={refit_row['correlation']:.4f} "
          f"relative_l2={refit_row['relative_l2']:.4f}")
    print("gates: " + json.dumps(provenance["gates"], ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="session directory with manifest.json and config.json")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="base frozen_parameters.json to warm-start from")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory for the refit checkpoint (must not exist)")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8,
                        help="early-stop patience in iterations without improvement")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="minimum loss improvement that resets the patience")
    parser.add_argument("--span", type=int, default=160,
                        help="camera window half-span in pixels (even, >= 20)")
    parser.add_argument("--adaptation", choices=("auto", "alignment", "full"), default="full",
                        help="physical adaptation rule (the diagnostic used full)")
    parser.add_argument("--radiometry", choices=("shared", "exposure"), default="exposure")
    parser.add_argument("--amplitude-center-lr", type=float, default=None,
                        help="override the illumination position learning rate")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--expect-splits", default=None,
                        help="optional NAME=COUNT list, e.g. train=80,test=24,calibration=10")
    parser.add_argument("--score-splits", default="test",
                        help="comma list of splits to score after the refit (default test)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the model and evaluate one loss only; write nothing")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
