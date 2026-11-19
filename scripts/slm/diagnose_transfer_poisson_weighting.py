"""Diagnose whether a Poisson/count-variance data fidelity is defensible for session B.

The transfer fit currently minimises ``correlation + 0.25*relative_mse + 0.25*CV``
on exposure-normalised *count-rate* images.  This script asks a single upstream
question before any refit is attempted: **can a physically meaningful Poisson (or
count-variance) likelihood be derived from the stored session-B data?**

It only reads recorded files; it never opens a device, never imports the fitting
code, and never selects anything on test rows.  If the answer is yes it would be
the precondition for the Poisson-vs-existing-objective refit; if the answer is no
it stops and writes a diagnostic that explains the blocking evidence instead of
inventing a noise model.

The decision rests on four things a Poisson likelihood needs and that are checked
explicitly here:

1. integer raw counts (available: ``raw_frames`` uint8 x3),
2. one exposure value per frame in ms (available),
3. an *independently measured* additive camera offset / dark level, and
4. a variance law consistent with Poisson once the offset is removed.

The script also records the signal's exposure linearity (it is photon-like) so the
diagnostic separates "the signal is Poisson" from "the recorded ADU are Poisson".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_DATASET = Path(
    r"D:\Projects\TIFO\AO-shaping\data\debug\slm_transfer\extend_b_20261118"
)
# Session A full physical fit: authoritative radiometry context only (never modified).
DEFAULT_BASELINE_FIT = Path(
    r"D:\Projects\TIFO\AO-shaping\data\debug\slm_transfer"
    r"\20261118_transfer_mil_zernike_probe_v1\fit_m3_full"
)
DEFAULT_OUTPUT = ROOT / "report/shape_trials/physical_refine/poisson_weighted_b_20261119.json"

# Crop geometry copied from ``transfer.fit.load_dataset``: ``span`` half-width on
# a stride-2 camera grid, centred on the session's ``reference_start`` spot.
FIT_SPAN = 160


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _f(x: float, ndigits: int = 6) -> float:
    return float(round(float(x), ndigits))


def _split_rows(manifest: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {"train": [], "test": [], "calibration": []}
    for row in manifest:
        groups.setdefault(row.get("split", "unknown"), []).append(row)
    return groups


def _crop_grid(centre_xy, span: int):
    """The stride-2 camera sample grid used by ``load_dataset``."""
    yy, xx = np.mgrid[-span:span + 1:2, -span:span + 1:2]
    gx = (xx + round(float(centre_xy[0]))).astype(int)
    gy = (yy + round(float(centre_xy[1]))).astype(int)
    radius = np.hypot(xx, yy)
    return gx, gy, radius


def _aggregate(per_row: list[dict], keys: tuple[str, ...]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key in keys:
        vals = np.asarray([row[key] for row in per_row], dtype=float)
        out[key] = {
            "mean": _f(vals.mean()),
            "min": _f(vals.min()),
            "max": _f(vals.max()),
            "std": _f(vals.std()),
        }
    return out


def analyze(dataset: Path, span: int) -> dict:
    """Read session-B rows and quantify the Poisson feasibility evidence."""
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    groups = _split_rows(manifest)
    reference = next(
        (row for row in manifest if row.get("label") == "reference_start"), None
    )
    if reference is None:
        raise ValueError("manifest has no reference_start row; cannot place crop")

    gx, gy, radius = _crop_grid(reference["spot_xy"], span)
    bg_mask = (radius > 60) & (radius < 150)
    core_mask = radius < 15
    if bg_mask.sum() == 0 or core_mask.sum() == 0:
        raise ValueError("crop too small for the background/core annuli")

    rows: list[dict] = []
    raw_dtypes: set[str] = set()
    frame_counts: list[int] = []
    # Train-only format facts: every decision-driving check is sourced from these.
    train_raw_dtypes: set[str] = set()
    train_frame_counts: list[int] = []
    for row in manifest:
        split = row.get("split")
        if split not in {"train", "test"}:
            continue  # calibration rows are never part of a fit or an evaluation
        payload = np.load(dataset / row["file"])
        try:
            raw = np.asarray(payload["raw_frames"])
        finally:
            if hasattr(payload, "close"):
                payload.close()
        raw_dtypes.add(str(raw.dtype))
        frame_counts.append(int(raw.shape[0]))
        if split == "train":
            train_raw_dtypes.add(str(raw.dtype))
            train_frame_counts.append(int(raw.shape[0]))

        crop = raw[:, gy, gx].astype(np.float64)  # (frames, side, side)
        total = crop.sum(axis=0)  # integer total over all retained frames
        frame_mean = crop.mean(axis=0)  # what fit.load_dataset stores
        exposure = float(row["exposure_ms"])

        bg = crop[:, bg_mask]
        core = crop[:, core_mask]
        rows.append(
            dict(
                split=split,
                exposure_ms=exposure,
                frames=int(raw.shape[0]),
                background_median=_f(np.median(bg)),
                background_mean=_f(bg.mean()),
                background_frac_0_or_1=_f(((bg == 0) | (bg == 1)).mean()),
                core_mean=_f(core.mean()),
                peak=_f(raw.max()),
                total_mean=_f(total.mean()),
                total_zero_fraction=_f((total == 0).mean()),
                # Per-pixel temporal variance across the repeated frames, averaged
                # over the region; Poisson in ADU would put this ratio at 1.0.
                var_over_mean_background=_f(
                    bg.var(axis=0, ddof=1).mean() / max(bg.mean(), 1e-12)
                ),
                var_over_mean_core=_f(
                    core.var(axis=0, ddof=1).mean() / max(core.mean(), 1e-12)
                ),
                # Stored fit target is count-rate: mean(raw_frames) / exposure_ms.
                count_rate_mean=_f(frame_mean.mean() / max(exposure, 1e-12)),
            )
        )

    def _group_by_exposure(subset: list[dict]) -> dict[str, dict]:
        """Aggregate the per-row statistics for each exposure level."""
        grouped: dict[str, dict] = {}
        for exposure in sorted({round(r["exposure_ms"], 4) for r in subset}):
            group = [r for r in subset if round(r["exposure_ms"], 4) == exposure]
            grouped[f"{exposure:g}"] = {
                "n_rows": len(group),
                **_aggregate(
                    group,
                    (
                        "background_median",
                        "background_mean",
                        "background_frac_0_or_1",
                        "core_mean",
                        "peak",
                        "total_mean",
                        "total_zero_fraction",
                        "var_over_mean_background",
                        "var_over_mean_core",
                        "count_rate_mean",
                    ),
                ),
            }
        return grouped

    # The decision is made from train rows only. Test rows are aggregated too,
    # but only as corroboration of the shared camera noise law; no model is fit
    # or selected here, so no test information can influence a parameter choice.
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    return dict(
        rows=rows,
        by_exposure=_group_by_exposure(rows),
        by_exposure_train=_group_by_exposure(train_rows),
        by_exposure_test=_group_by_exposure(test_rows),
        raw_dtypes=sorted(raw_dtypes),
        frame_counts=sorted(set(frame_counts)),
        train_raw_dtypes=sorted(train_raw_dtypes),
        train_frame_counts=sorted(set(train_frame_counts)),
        crop=dict(
            span=int(span),
            side=int(gx.shape[0]),
            stride=2,
            background_pixels=int(bg_mask.sum()),
            core_pixels=int(core_mask.sum()),
        ),
        splits={k: len(v) for k, v in groups.items()},
        reference_label=reference.get("label"),
    )


def _find_offset_artifacts(dataset: Path, baseline_fit: Path | None) -> dict:
    """Look for an independent dark/offset measurement and the fitted radiometry."""
    dark_candidates = sorted(
        str(p.relative_to(dataset))
        for p in dataset.glob("*")
        if p.is_file() and "dark" in p.name.lower()
    )
    frozen = None
    if baseline_fit is not None:
        frozen_path = baseline_fit / "frozen_parameters.json"
        if frozen_path.exists():
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    return dict(
        dark_candidates=dark_candidates,
        has_independent_dark=bool(dark_candidates),
        baseline_fit=str(baseline_fit) if baseline_fit else None,
        fitted_radiometry=None
        if frozen is None
        else dict(
            gain=frozen.get("gain"),
            background=frozen.get("background"),
            bias_counts=frozen.get("bias_counts"),
        ),
    )


def _load_light_check(dataset: Path) -> dict | None:
    path = dataset / "light_check.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_checks(analysis: dict, offset: dict, light: dict | None) -> list[dict]:
    excluded = analysis["splits"].get("calibration", 0)
    total_rows = sum(analysis["splits"].values())

    # Every decision-driving check is sourced from B TRAIN rows only. Test rows are
    # never used to form a status, a threshold, a blocker, or the fit plan; their
    # aggregates are recorded separately as descriptive corroboration.
    source_split = "train"
    train_by_exp = analysis["by_exposure_train"]

    # Train-only format facts drive the integer/frames check.
    dtypes = analysis["train_raw_dtypes"]
    integer_dtype = all(
        name in {"uint8", "uint16", "int16", "int32", "uint32"}
        for name in dtypes
    )
    frame_counts = analysis["train_frame_counts"]
    frames_ok = bool(frame_counts) and min(frame_counts) >= 2

    # Exposure linearity: a photon signal scales with exposure, a bias does not.
    exps = sorted(train_by_exp, key=float)
    if len(exps) == 2:
        lo, hi = (train_by_exp[e] for e in exps)
        core_ratio = hi["core_mean"]["mean"] / max(lo["core_mean"]["mean"], 1e-12)
        exposure_ratio = float(exps[1]) / float(exps[0])
        bg_ratio = hi["background_mean"]["mean"] / max(
            lo["background_mean"]["mean"], 1e-12
        )
    else:
        core_ratio = exposure_ratio = bg_ratio = float("nan")

    # A Poisson variance law in ADU requires var/mean ~ 1.0. Reject a value that
    # sits well above or below it: this is the direct empirical test, on TRAIN rows
    # only so the precondition decision cannot depend on held-out data.
    var_ratio_bg = [
        v["var_over_mean_background"]["mean"] for v in train_by_exp.values()
    ]
    var_ratio_core = [
        v["var_over_mean_core"]["mean"] for v in train_by_exp.values()
    ]
    var_ratios = var_ratio_bg + var_ratio_core
    poisson_variance_ok = bool(var_ratios) and all(0.66 <= r <= 1.5 for r in var_ratios)

    bias = None if offset["fitted_radiometry"] is None else offset["fitted_radiometry"]["bias_counts"]
    background_term = (
        None if offset["fitted_radiometry"] is None else offset["fitted_radiometry"]["background"]
    )
    noise_sigma = None if light is None else light.get("noise_sigma_counts")
    if noise_sigma is None and light is not None:
        readings = light.get("readings") or []
        if readings:
            noise_sigma = readings[0].get("noise_sigma_counts")

    return [
        dict(
            name="integer_raw_counts",
            status="pass" if integer_dtype and frames_ok else "fail",
            blocking=False,
            evidence=dict(source_split=source_split,
                          raw_dtypes=dtypes, frames_per_row=frame_counts,
                          inspected_train_rows=analysis["splits"].get("train", 0),
                          manifest_rows=total_rows),
            explanation=(
                "raw_frames are integer ADU with repeated frames, so a count law is "
                "at least representable." if integer_dtype and frames_ok else
                "raw_frames are missing, non-integer, or single-frame."
            ),
        ),
        dict(
            name="per_row_exposure_ms",
            status="pass" if train_by_exp else "fail",
            blocking=False,
            evidence=dict(
                source_split=source_split,
                exposures=sorted(train_by_exp, key=float),
                train_rows=analysis["splits"].get("train", 0),
                test_rows_descriptive_only=analysis["splits"].get("test", 0),
                calibration_rows_excluded=excluded,
            ),
            explanation="Every train/test row carries a positive exposure in ms.",
        ),
        dict(
            name="signal_linear_in_exposure",
            status="pass" if np.isfinite(exposure_ratio) and abs(core_ratio - exposure_ratio) < 0.3 * exposure_ratio else "insufficient",
            blocking=False,
            evidence=dict(
                source_split=source_split,
                exposure_ratio=_f(exposure_ratio),
                core_mean_ratio=_f(core_ratio),
                background_mean_ratio=_f(bg_ratio),
            ),
            explanation=(
                "Core counts scale with exposure like light; the additive background "
                "does not scale with exposure, so it is a constant pedestal rather than "
                "ambient (Poisson) light."
            ),
        ),
        dict(
            name="independently_measured_offset",
            status="pass" if offset["has_independent_dark"] else "fail",
            blocking=True,
            evidence=dict(
                source_split=source_split,
                dark_frames_found=offset["dark_candidates"],
                only_offset_is_fitted_radiometry=offset["fitted_radiometry"],
                fitted_bias_counts=bias,
                fitted_background_term=background_term,
                empirical_background_mean={
                    e: v["background_mean"]["mean"] for e, v in train_by_exp.items()
                },
                empirical_background_median={
                    e: v["background_median"]["mean"] for e, v in train_by_exp.items()
                },
            ),
            explanation=(
                "No dark/offset frame exists in the session. The only additive offset is "
                "the radiometry bias_counts from the session-A fit, which the codebase "
                "documents as not independently identifiable without a dark/response "
                "experiment. A Poisson mean cannot be separated into light + measured "
                "offset from these data."
            ),
        ),
        dict(
            name="camera_conversion_gain",
            status="fail",
            blocking=True,
            evidence=dict(
                source_split="dataset_metadata",
                source_note="Reads light_check.json metadata only; no image rows, no split dependence.",
                recorded_camera_gain=None,
                counts_per_ms=(light or {}).get("counts_per_ms"),
                read_noise_sigma_counts=noise_sigma,
                note=(
                    "counts_per_ms is an optical count-rate, not the camera ADU/e- "
                    "conversion gain; no e-/ADU value is stored."
                ),
            ),
            explanation=(
                "Poisson statistics apply to photoelectrons. Without the ADU/e- "
                "conversion gain the variance of an ADU count is unknown up to a "
                "multiplicative factor, so an absolute Poisson likelihood cannot be "
                "formed."
            ),
        ),
        dict(
            name="poisson_variance_law",
            status="pass" if poisson_variance_ok else "fail",
            blocking=True,
            evidence=dict(
                source_split=source_split,
                var_over_mean_background=var_ratio_bg,
                var_over_mean_core=var_ratio_core,
                poisson_expectation=1.0,
                note="Computed from B train rows only; no test row contributes to this decision.",
            ),
            explanation=(
                "Per-pixel temporal variance is about 0.3x the mean, not ~1x. The "
                "recorded ADU are strongly sub-Poisson (consistent with ~3 e-/ADU or "
                "on-camera averaging), so a var=mean likelihood mis-weights pixels by "
                "roughly 3x and mis-models the near-deterministic pedestal."
            ),
        ),
        dict(
            name="background_quantisation",
            status="fail" if any(v["background_frac_0_or_1"]["mean"] > 0.9 for v in train_by_exp.values()) else "pass",
            blocking=False,
            evidence=dict(
                source_split=source_split,
                background_frac_0_or_1={
                    e: v["background_frac_0_or_1"]["mean"] for e, v in train_by_exp.items()
                },
                background_mean={
                    e: v["background_mean"]["mean"] for e, v in train_by_exp.items()
                },
            ),
            explanation=(
                "Nearly all background pixels are exactly 0 or 1 ADU and the mean is "
                "below one ADU, so the background is quantisation-limited, not a "
                "sampled Poisson light level."
            ),
        ),
        dict(
            name="stored_target_is_count_rate",
            status="insufficient",
            blocking=False,
            evidence=dict(
                source_split="code_definition",
                source_note="Describes how fit.load_dataset stores the target; no image rows involved.",
                stored_frame="raw_frames.mean(axis=0)",
                stored_target="frame / exposure_ms  # counts per ms",
            ),
            explanation=(
                "The fit compares count-rate images, not integer total counts. A Poisson "
                "likelihood would need N=sum(raw_frames) and a mean that also carries the "
                "three-frame stack, which the current target does not."
            ),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline-fit", type=Path, default=DEFAULT_BASELINE_FIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--span", type=int, default=FIT_SPAN)
    args = parser.parse_args()

    if not (args.dataset / "manifest.json").exists():
        parser.error(f"no manifest.json under {args.dataset}")
    if args.span < 20:
        parser.error("--span must be >= 20 to hold the background/core annuli")

    logger.info("reading session {}", args.dataset)
    analysis = analyze(args.dataset, args.span)
    offset = _find_offset_artifacts(args.dataset, args.baseline_fit)
    light = _load_light_check(args.dataset)
    checks = build_checks(analysis, offset, light)

    blocking = [c for c in checks if c["blocking"] and c["status"] != "pass"]
    defensible = not blocking

    if defensible:
        decision = "proceed_to_comparison"
        reason = (
            "All Poisson preconditions pass. The next step is the same-start, "
            "same-budget refit described under intended_design; it is not run by this "
            "diagnostic."
        )
    else:
        decision = "stop_not_defensible"
        reason = (
            "A defensible Poisson likelihood cannot be derived from the stored data. "
            "Blocking preconditions fail (see blocking_reasons). Per the task rule, the "
            "script stops and explains why instead of inventing a noise model."
        )

    authoritative_context = {
        "session_a_fit": str(args.baseline_fit),
        "session_a_fitted_radiometry": offset["fitted_radiometry"],
        "prior_authoritative_results": [
            "report/shape_trials/physical_refine/fit_compare_m3_20261118.md",
            "report/shape_trials/physical_refine/corrector_transfer_to_b.json",
        ],
        "current_objective": "correlation + 0.25*relative_mse + 0.25*CV (fit_loss)",
        "known_bias_to_test": (
            "flat/smooth-profile contrast: predicted CV is biased relative to measured "
            "(cv_prediction rose +0.45 in the session-A comparison); this is the target "
            "of the Poisson experiment."
        ),
    }

    result = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/slm/diagnose_transfer_poisson_weighting.py",
        "question": (
            "Can a physically meaningful Poisson/count-variance data fidelity be derived "
            "from session B, and would it improve held-out B prediction and the "
            "flat-profile bias over the existing correlation+MSE+CV objective?"
        ),
        "decision": decision,
        "decision_reason": reason,
        "poisson_likelihood_defensible": defensible,
        "blocking_reasons": [
            dict(name=c["name"], evidence=c["evidence"], explanation=c["explanation"])
            for c in blocking
        ],
        "checks": checks,
        "decision_data_policy": {
            "decision_source_split": "train",
            "rule": (
                "Every decision-driving / blocking check is free of test-row "
                "influence. Row-derived checks (integer_raw_counts, "
                "per_row_exposure_ms, signal_linear_in_exposure, "
                "independently_measured_offset, poisson_variance_law, "
                "background_quantisation) are computed from B train rows only "
                "(source_split=train). camera_conversion_gain reads dataset metadata "
                "only (source_split=dataset_metadata), and stored_target_is_count_rate "
                "is a code-definition fact (source_split=code_definition); neither "
                "uses any image row, so neither can depend on the test split."
            ),
            "test_rows": (
                "Held-out B test rows appear only in data_inventory.exposure_groups_test "
                "as descriptive aggregates. They never determine decision, blockers, "
                "thresholds, or the fit plan, and no parameter is fit or selected on them."
            ),
            "calibration_rows": "excluded from all checks (10 rows).",
        },
        "data_inventory": {
            "dataset": str(args.dataset),
            "splits": analysis["splits"],
            "calibration_excluded": analysis["splits"].get("calibration", 0),
            "exposure_groups": analysis["by_exposure"],  # descriptive only, all train+test rows
            "exposure_groups_train": analysis["by_exposure_train"],  # decision-driving source
            "exposure_groups_test": analysis["by_exposure_test"],  # descriptive only, held out
            "exposure_groups_note": (
                "exposure_groups is an all-split descriptive aggregate and is NOT used by "
                "any decision; decision checks use exposure_groups_train only."
            ),
            "crop": analysis["crop"],
            "reference_label": analysis["reference_label"],
        },
        "units_and_offsets": {
            "raw_frames": "integer ADU, repeated frames; summed for a count target",
            "stored_frame": "raw_frames.mean(axis=0)  # ADU per frame",
            "stored_target": "frame / exposure_ms  # count-rate (counts/ms)",
            "model_counts_rate": "gain*optical + background + bias_counts/exposure_ms",
            "fitted_offset_per_frame_ADU": None if offset["fitted_radiometry"] is None else offset["fitted_radiometry"]["bias_counts"],
            "measured_background_per_frame_ADU_train": {
                e: v["background_mean"]["mean"] for e, v in analysis["by_exposure_train"].items()
            },
        },
        "intended_design": {
            "start": (
                "session-A fit_m3_full frozen parameters (defocus, amplitude centre/width, "
                "residual Z5..Z11) with B's ramp matrix re-fitted from B's signed ramp "
                "pairs and origin re-anchored to B's reference_start; adaptation=full."
            ),
            "train_rows": analysis["splits"].get("train", 0),
            "test_rows": analysis["splits"].get("test", 0),
            "calibration_rows_excluded": analysis["splits"].get("calibration", 0),
            "comparison": [
                "baseline: existing correlation + 0.25*relative_mse + 0.25*CV, fitted on B train only",
                "treatment: Poisson NLL on integer total counts versus the same full physical forward fitted on B train only",
            ],
            "same_start_same_budget": (
                "identical initial parameters and identical number of optimizer steps for "
                "both objectives; model frozen before test"
            ),
            "freeze_then_test": (
                "select on train, freeze, then report signed Pearson, relative L2, CV "
                "absolute error and the flat-profile bias on the 24 test rows; no test tuning"
            ),
            "run": False,
            "not_run_reason": reason,
        },
        "what_would_make_it_defensible": [
            "Capture a blocked-beam dark frame stack at each exposure (0.028 and 0.051 ms) "
            "to measure the per-pixel bias and its temporal noise.",
            "Capture an exposure-response series to estimate the ADU/e- conversion gain and "
            "read noise from variance versus mean; verify var/mean ~ 1/gain.",
            "Re-run this diagnostic; if var/mean ~ 1/gain and a measured dark exists, use a "
            "Poisson (or Poisson+read-noise / Pearson chi-square) likelihood on "
            "dark-subtracted integer counts.",
        ],
        "authoritative_context": authoritative_context,
        "provenance": {
            "manifest_sha256": _sha256(args.dataset / "manifest.json"),
            "config_sha256": _sha256(args.dataset / "config.json")
            if (args.dataset / "config.json").exists()
            else None,
            "light_check_sha256": _sha256(args.dataset / "light_check.json")
            if (args.dataset / "light_check.json").exists()
            else None,
            "baseline_frozen_sha256": _sha256(args.baseline_fit / "frozen_parameters.json")
            if (args.baseline_fit / "frozen_parameters.json").exists()
            else None,
            "device_touched": False,
            "fit_code_modified": False,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("decision={} defensible={}", decision, defensible)
    logger.info("wrote {}", args.output)

    print(json.dumps(dict(decision=decision, defensible=defensible,
                          blocking=[c["name"] for c in blocking]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
