"""Simulation pretraining for the low-dimensional physical forward model.

Why this exists
---------------
The transfer pipeline (:mod:`ao_shaping.tools.slm.transfer`) estimates 14 physical
scalars from a handful of real probe images: defocus, shared optical origin,
incident amplitude centre/width, residual Noll 5-11 and an NNLS gain/background.
On a *new* optical path those scalars must be recovered from very few measurements,
so a cold start can wander (see the reproduction report §12.3 step 5).

This module builds a **matched-model synthetic library** over an explicit grid of
optical-path parameters and fits a small *residual* estimator: given a probe image
and the nominal-model prediction for that same probe, predict the deviation of the
low-dimensional parameter vector from the nominal prior. The experiment then has to
transfer only this low-dimensional vector, not a large network.

What it is NOT
--------------
* Not a fidelity claim. The library uses the same frozen forward model that the
  experiment is fitted with, no camera noise, no LUT, no non-linearity, no zero-order
  light. It only measures how well the parameters are *identifiable* from these
  features under a matched propagation model.
* Not an alternative physics implementation: propagation is *only*
  :class:`FrozenDefocusForward`. There is no second FFT in this repo and none here.

Validation protocol
-------------------
Grouping unit = one optical-path parameter **combination**. Cross-validation is
leave-one-combination-out, so no probe from the evaluated combo is ever seen in
training. Random per-probe or per-pixel splits are forbidden: probes within one combo
share the same truth and would leak it.

Feature/penalty notes (measured on the 60-combination library, ``fft_size=128``,
8 probes per combo, 16 standardized descriptors)
-----------------------------------------------------------------------------------
* Every descriptor is a ratio, a correlation, a centroid difference or a log width
  ratio, so it is invariant to exposure *and* to a global throughput factor.
* The penalty is small on purpose. ``alpha=1`` shrinks the collinear centroid block
  into the noise (grouped-CV R2 0.74 for the incident shift, 0.07 for the shared
  Noll 5-11 aberration); ``alpha=0.01`` keeps it (0.93 / 0.33). ``alpha_sweep``
  records the whole grid in ``report.json``; the default came from that sweep on the
  same combinations, i.e. it is indicative, not nested-CV selected.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ao_shaping.tools.slm.transfer.capture import clock, probes

#: Axis values of the synthetic optical-path grid. Deterministic, documented, small.
DEFOCUS_AXIS = (-0.12, -0.06, 0.0, 0.06, 0.12)
SHIFT_AXIS = ((0.0, 0.0), (12.0, -8.0), (-18.0, 10.0))
WIDTH_SCALE_AXIS = (0.85, 1.0, 1.2)
ABERRATION_AXIS = (0.0, 0.05)

#: Regression targets, in the order used by the coefficient matrix.
PARAM_NAMES = ("dy_defocus", "dx_center", "dy_center", "log_width_scale", "aberration_rad")

#: Ridge penalty. 16 standardized descriptors against 59 training combinations per
#: fold: ``alpha=1`` shrinks the collinear centroid block into the noise (grouped CV
#: R2 0.74 for the incident shift, 0.07 for the shared aberration) while the
#: low-penalty end keeps it. The default sits at that end; see ``ALPHA_GRID`` /
#: :func:`alpha_sweep` and the selection caveat recorded in ``report.json``.
DEFAULT_ALPHA = 0.01

#: Candidate penalties reported by :func:`alpha_sweep` (same grouped CV, no re-split).
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0)

#: Nominal prior the residual estimator is expressed against.
NOMINAL_DEFOCUS = 0.025
NOMINAL_CENTER = (960.0, 600.0)
NOMINAL_WIDTH = (300.0, 260.0)


def combo_grid(limit: int | None = None) -> list[dict]:
    """Deterministic full-factorial grid of optical-path combinations.

    Args:
        limit: take only the first ``limit`` combinations (in the fixed order below).
            ``None`` keeps the whole grid.

    Returns:
        List of ``{"combo", "defocus", "shift", "width_scale", "aberration"}`` dicts.
    """
    combos: list[dict] = []
    for defocus in DEFOCUS_AXIS:
        for shift in SHIFT_AXIS:
            for width_scale in WIDTH_SCALE_AXIS:
                for aberration in ABERRATION_AXIS:
                    combos.append(dict(combo=len(combos), defocus=float(defocus),
                                       shift=tuple(float(v) for v in shift),
                                       width_scale=float(width_scale),
                                       aberration=float(aberration)))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        combos = combos[:limit]
    return combos


def nominal_parameters(*, fft_size: int = 128, pixel_samples: int = 1, grid_size: int = 32):
    """The prior that the residual estimator predicts *deviations from*."""
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusParameters,
    )

    half = fft_size // 2
    return FrozenDefocusParameters(
        defocus_diopter=NOMINAL_DEFOCUS, origin_xy=(float(half), float(half)),
        amplitude_center_xy=NOMINAL_CENTER, amplitude_width_xy=NOMINAL_WIDTH,
        residual_noll_radians={j: 0.0 for j in range(5, 12)},
        gain=1.0, background=0.0, sigma_ccd_px=0.0,
        ramp_matrix=((0.0, 5000.0), (-5000.0, 0.0)),
        grid_size=grid_size, fft_size=fft_size, pixel_samples=pixel_samples)


def truth_parameters(combo: dict, *, fft_size: int = 128, pixel_samples: int = 1,
                     grid_size: int = 32):
    """Build the ground-truth parameters of one library combination."""
    nominal = nominal_parameters(fft_size=fft_size, pixel_samples=pixel_samples,
                                 grid_size=grid_size)
    center = (NOMINAL_CENTER[0] + combo["shift"][0], NOMINAL_CENTER[1] + combo["shift"][1])
    width = (NOMINAL_WIDTH[0] * combo["width_scale"], NOMINAL_WIDTH[1] * combo["width_scale"])
    residual = {j: float(combo["aberration"]) for j in range(5, 12)}
    return replace(nominal, defocus_diopter=float(combo["defocus"]),
                   amplitude_center_xy=center, amplitude_width_xy=width,
                   residual_noll_radians=residual)


def combo_target(combo: dict) -> np.ndarray:
    """The regression target of a combination: deviation from the nominal prior."""
    width_scale = float(combo["width_scale"])
    if width_scale <= 0:
        raise ValueError("width_scale must be positive")
    return np.array([float(combo["defocus"]) - NOMINAL_DEFOCUS,
                     float(combo["shift"][0]), float(combo["shift"][1]),
                     float(np.log(width_scale)), float(combo["aberration"])], dtype=np.float64)


def _centroid(image: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    weights = np.clip(image * mask, 0, None)
    total = float(weights.sum())
    if total <= 0:
        return 0.5 * image.shape[1], 0.5 * image.shape[0]
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    return float((xx * weights).sum() / total), float((yy * weights).sum() / total)


def _spread(image: np.ndarray, mask: np.ndarray, centre: tuple[float, float]) -> tuple[float, float]:
    weights = np.clip(image * mask, 0, None)
    total = float(weights.sum())
    if total <= 0:
        return 1.0, 1.0
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    sx = float(np.sqrt(((xx - centre[0]) ** 2 * weights).sum() / total))
    sy = float(np.sqrt(((yy - centre[1]) ** 2 * weights).sum() / total))
    return max(sx, 1e-6), max(sy, 1e-6)


def pair_features(measured: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Eight descriptors of ``measured`` against the nominal prediction ``predicted``.

    Only the two images enter. No ground-truth parameter is touched, so the feature
    vector cannot leak the regression target.
    """
    measured = np.asarray(measured, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if measured.shape != predicted.shape:
        raise ValueError(f"image shape mismatch: {measured.shape} vs {predicted.shape}")
    peak = float(predicted.max())
    mask = predicted > 0.05 * peak if peak > 0 else np.zeros_like(predicted, dtype=bool)
    if int(mask.sum()) < 16:
        mask = predicted >= np.percentile(predicted, 90)

    energy_ratio = float(np.log((measured[mask].sum() + 1e-9) / (predicted[mask].sum() + 1e-9)))
    a, b = measured[mask], predicted[mask]
    denom = float(np.std(a) * np.std(b))
    corr = float(np.corrcoef(a, b)[0, 1]) if denom > 1e-12 else 0.0
    corr = 0.0 if not np.isfinite(corr) else corr

    cx_m, cy_m = _centroid(measured, mask)
    cx_p, cy_p = _centroid(predicted, mask)
    sx_m, sy_m = _spread(measured, mask, (cx_m, cy_m))
    sx_p, sy_p = _spread(predicted, mask, (cx_p, cy_p))
    peak_ratio = float(np.log((measured.max() + 1e-9) / (predicted.max() + 1e-9)))
    ratio = np.clip(measured[mask] / (predicted[mask] + 1e-9), 1e-3, 1e3)
    return np.array([energy_ratio, corr, cx_m - cx_p, cy_m - cy_p,
                     float(np.log(sx_m / sx_p)), float(np.log(sy_m / sy_p)),
                     peak_ratio, float(np.log(ratio).std())], dtype=np.float64)


def build_library(*, combos: int | None = 60, probes_per_combo: int = 8, fft_size: int = 128,
                  pixel_samples: int = 1, grid_size: int = 32, seed: int = 102419,
                  power: float = 1.0) -> dict:
    """Synthesize the library with the frozen forward model (matched propagation).

    Exposure cycles over 0.5/1/1.5 ms exactly like
    :func:`ao_shaping.tools.slm.transfer.simulation.generate_dataset`, and every image
    is evaluated as ``counts / exposure_ms`` before features are taken, so features are
    exposure-invariant by construction.

    ``power`` is the measured-path throughput kept for signature symmetry with
    :func:`generate_dataset`. Every descriptor in :func:`pair_features` is a ratio or a
    correlation, so a global throughput factor cancels and ``power`` is deliberately
    *not* a regression target: it carries no learnable signal whatever its value.

    Returns:
        ``{"features", "targets", "combo_ids", "combo_rows", "probes", "setup"}``.
    """
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
    )

    if probes_per_combo < 1:
        raise ValueError("probes_per_combo must be positive")
    if grid_size < 2 or fft_size < 8:
        raise ValueError("grid_size >= 2 and fft_size >= 8 required")
    rows = combo_grid(combos)
    yy, xx = np.mgrid[:fft_size, :fft_size]
    camera_xy = np.stack((xx, yy), axis=-1).astype(np.float64)

    probe_list = [entry for entry in probes(grid_size, probes_per_combo, seed)
                  if entry[0].startswith("probe_")]
    nominal = nominal_parameters(fft_size=fft_size, pixel_samples=pixel_samples,
                                 grid_size=grid_size)
    nominal_forward = FrozenDefocusForward(nominal, camera_xy, device="cpu")
    with np.errstate(all="ignore"):
        nominal_images = [nominal_forward.predict(phase).detach().numpy()[0]
                          for _, _, phase in probe_list]
    reference_peak = max(float(image.max()) for image in nominal_images)
    scale = 80.0 / reference_peak if reference_peak > 0 else 1.0

    feature_rows, target_rows, combo_ids = [], [], []
    for row in rows:
        truth = truth_parameters(row, fft_size=fft_size, pixel_samples=pixel_samples,
                                 grid_size=grid_size)
        forward = FrozenDefocusForward(truth, camera_xy, device="cpu")
        per_probe = []
        for index, (_, _, phase) in enumerate(probe_list):
            exposure = (0.5, 1.0, 1.5)[index % 3]
            with np.errstate(all="ignore"):
                image = forward.predict(phase).detach().numpy()[0]
            # counts / exposure_ms on both sides: features stay exposure-invariant.
            measured_counts = image * scale * power * exposure
            predicted_counts = nominal_images[index] * scale * power * exposure
            per_probe.append(pair_features(measured_counts / exposure,
                                           predicted_counts / exposure))
        stack = np.stack(per_probe)
        feature_rows.append(np.concatenate((stack.mean(0), stack.std(0))))
        target_rows.append(combo_target(row))
        combo_ids.append(int(row["combo"]))

    return dict(features=np.stack(feature_rows), targets=np.stack(target_rows),
                combo_ids=np.asarray(combo_ids, dtype=int), combo_rows=rows,
                probes=[label for label, _, _ in probe_list],
                setup=dict(fft_size=fft_size, pixel_samples=pixel_samples, grid_size=grid_size,
                           seed=seed, power=power, probes_per_combo=len(probe_list),
                           exposure_cycle=[0.5, 1.0, 1.5], scale=scale))


def _standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(0)
    scale = matrix.std(0)
    scale[scale < 1e-9] = 1.0
    return (matrix - mean) / scale, mean, scale


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    design = np.hstack([x, np.ones((len(x), 1))])
    gram = design.T @ design + alpha * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ y)
    return weights[:-1], weights[-1]


def grouped_folds(combo_ids) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Leave-one-combination-out fold masks, in ``sorted(unique)`` order.

    The grouping unit is the optical-path **combination**, never an individual probe
    or pixel: probes inside one combination share the same truth and would leak it.

    Args:
        combo_ids: per-row combination id (one row per library combination).

    Returns:
        ``[(held_out_combo, train_mask, test_mask), ...]`` — ``train_mask`` never
        contains ``held_out_combo`` and ``test_mask`` contains nothing else.
    """
    groups = np.asarray(combo_ids, dtype=int)
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("grouped CV needs at least three parameter combinations")
    return [(int(held), groups != held, groups == held) for held in unique]


def train_estimator(library: dict, *, alpha: float = DEFAULT_ALPHA) -> dict:
    """Fit the residual estimator and score it with leave-one-combination-out CV.

    Args:
        library: output of :func:`build_library`.
        alpha: ridge penalty (features and targets are standardized inside each fold).

    Returns:
        ``{"coef", "intercept", "feature_mean", "feature_scale", "target_mean",
        "target_scale", "metrics", "folds"}``.
    """
    features = np.asarray(library["features"], dtype=np.float64)
    targets = np.asarray(library["targets"], dtype=np.float64)
    groups = np.asarray(library["combo_ids"], dtype=int)
    unique = np.unique(groups)

    predictions = np.full_like(targets, np.nan)
    folds = []
    for held, train, test in grouped_folds(groups):
        if train.sum() < 2:
            continue
        x_train, feat_mean, feat_scale = _standardize(features[train])
        y_train, tgt_mean, tgt_scale = _standardize(targets[train])
        coef, intercept = _ridge(x_train, y_train, alpha)
        x_test = (features[test] - feat_mean) / feat_scale
        prediction = (x_test @ coef + intercept) * tgt_scale + tgt_mean
        predictions[test] = prediction
        folds.append(dict(combo=int(held), train_samples=int(train.sum()),
                          test_samples=int(test.sum()),
                          mae={name: float(np.mean(np.abs(prediction[:, i] - targets[test][:, i])))
                               for i, name in enumerate(PARAM_NAMES)}))

    scored = ~np.isnan(predictions[:, 0])
    if not scored.any():
        raise RuntimeError("grouped CV produced no held-out prediction")
    metrics = {}
    for index, name in enumerate(PARAM_NAMES):
        truth = targets[scored, index]
        pred = predictions[scored, index]
        residual = pred - truth
        ss_res = float((residual ** 2).sum())
        ss_tot = float(((truth - truth.mean()) ** 2).sum())
        metrics[name] = dict(
            mae=float(np.mean(np.abs(residual))),
            rmse=float(np.sqrt(np.mean(residual ** 2))),
            r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            bias=float(residual.mean()))

    x_all, feat_mean, feat_scale = _standardize(features)
    y_all, tgt_mean, tgt_scale = _standardize(targets)
    coef, intercept = _ridge(x_all, y_all, alpha)
    return dict(coef=coef, intercept=intercept, feature_mean=feat_mean,
                feature_scale=feat_scale, target_mean=tgt_mean, target_scale=tgt_scale,
                metrics=metrics, folds=folds, scored=int(scored.sum()),
                combinations=int(len(unique)))


def alpha_sweep(library: dict, alphas: tuple[float, ...] = ALPHA_GRID) -> dict:
    """Same grouped CV, one fit per candidate penalty — no re-split, no peeking.

    The outer folds are identical for every ``alpha`` (``grouped_folds`` is
    deterministic), so the entries are comparable row by row. They are still *not* a
    nested selection: the reported ``alpha`` was picked on the same 60 combinations it
    is scored on, which is why ``report.json`` carries this caveat explicitly.

    Returns:
        ``{str(alpha): {"mae": {...}, "r2": {...}, "worst_r2": float}}``.
    """
    sweep = {}
    for alpha in alphas:
        metrics = train_estimator(library, alpha=float(alpha))["metrics"]
        sweep[f"{float(alpha):g}"] = dict(
            mae={name: metrics[name]["mae"] for name in PARAM_NAMES},
            r2={name: metrics[name]["r2"] for name in PARAM_NAMES},
            worst_r2=float(min(metrics[name]["r2"] for name in PARAM_NAMES)))
    return sweep


def _prior_checkpoint(library: dict, model: dict, *, fft_size: int, pixel_samples: int,
                      grid_size: int) -> dict:
    """Translate the mean predicted deviation into a ``FrozenDefocusParameters`` dict."""
    prediction = (library["features"] - model["feature_mean"]) / model["feature_scale"]
    delta = (prediction @ model["coef"] + model["intercept"]) * model["target_scale"] \
        + model["target_mean"]
    mean_delta = delta.mean(0)
    base = nominal_parameters(fft_size=fft_size, pixel_samples=pixel_samples,
                              grid_size=grid_size)
    width_scale = float(np.exp(np.clip(mean_delta[3], -0.7, 0.7)))
    updated = replace(
        base,
        defocus_diopter=float(base.defocus_diopter + mean_delta[0]),
        amplitude_center_xy=(float(base.amplitude_center_xy[0] + mean_delta[1]),
                             float(base.amplitude_center_xy[1] + mean_delta[2])),
        amplitude_width_xy=(float(base.amplitude_width_xy[0] * width_scale),
                            float(base.amplitude_width_xy[1] * width_scale)),
        residual_noll_radians={j: float(mean_delta[4]) for j in range(5, 12)})
    payload = updated.to_dict()
    payload["pretrain"] = dict(note="synthetic mean deviation from the nominal prior; "
                                    "not a hardware-calibrated checkpoint",
                               mean_delta=[float(v) for v in mean_delta],
                               params=list(PARAM_NAMES))
    return payload


def pretrain(out: str | Path, *, combos: int | None = 60, probes_per_combo: int = 8,
             fft_size: int = 128, pixel_samples: int = 1, grid_size: int = 32,
             seed: int = 102419, alpha: float = DEFAULT_ALPHA, power: float = 1.0,
             keep_raw: bool = False) -> dict:
    """Build the library, fit the estimator and write the three artifacts.

    Artifacts written under ``out``:

    * ``estimator.npz`` — ridge coefficients plus feature/target standardization;
    * ``frozen_parameters.json`` — the prior initialization in
      ``FrozenDefocusParameters.to_dict()`` form, consumable by
      ``transfer run --checkpoint``;
    * ``report.json`` — grid definition, grouped metrics, per-fold errors and clocks.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    started = time.perf_counter()
    library = build_library(combos=combos, probes_per_combo=probes_per_combo,
                            fft_size=fft_size, pixel_samples=pixel_samples,
                            grid_size=grid_size, seed=seed, power=power)
    built = time.perf_counter()
    model = train_estimator(library, alpha=alpha)
    sweep = alpha_sweep(library)
    fitted = time.perf_counter()

    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "estimator.npz", coef=model["coef"],
                        intercept=model["intercept"], feature_mean=model["feature_mean"],
                        feature_scale=model["feature_scale"],
                        target_mean=model["target_mean"], target_scale=model["target_scale"],
                        params=np.asarray(PARAM_NAMES))
    prior = _prior_checkpoint(library, model, fft_size=fft_size,
                              pixel_samples=pixel_samples, grid_size=grid_size)
    (out / "frozen_parameters.json").write_text(json.dumps(prior, indent=2), encoding="utf-8")
    per_combo = int(library["features"].shape[1])
    report = dict(
        setup=library["setup"], grid=library["combo_rows"], probes=library["probes"],
        combinations=int(library["features"].shape[0]),
        samples=f"{per_combo} features per combination "
                f"(mean and std over {len(library['probes'])} probes of "
                f"{per_combo // 2} descriptors)",
        alpha=alpha, alpha_sweep=sweep,
        alpha_selection="alpha default comes from this sweep on the same combinations "
                        "(not nested CV); treat the ranking as indicative",
        metrics=model["metrics"], folds=model["folds"],
        folds_scored=model["scored"],
        timings=dict(library_s=round(built - started, 2),
                     fit_s=round(fitted - built, 2),
                     total_s=round(time.perf_counter() - started, 2)),
        caveat="Matched-model synthetic library: no camera noise, LUT, non-linearity or "
               "zero-order light. Grouped CV measures identifiability, not hardware accuracy.",
        **clock())
    if keep_raw:
        report["raw_library"] = True  # features only; raw images are not retained
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return dict(out=out, metrics=model["metrics"], timings=report["timings"],
                combinations=report["combinations"], prior=prior)
