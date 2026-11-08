"""Training-loss comparison for the defocus forward model (issue #236 groundwork).

Re-drives the repo's own fitter (``PhysicalForwardFit``) on the historical paired
U/C dataset already on disk, changing ONLY the training loss. Offline: reads
``.npz``/``.json`` under ``data/debug/slm_transfer`` and writes its results under
``data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1/``. It opens no device and
imports no ``ao_shaping.drivers`` module.

Pre-registered before the first run (recorded in ``results.json`` and
``REPORT.md``):

* Composite weights: ``0.4 * masked_mse + 0.4 * pearson + 0.2 * ee_uniformity``.
* Ranking rule, in order: median per-pair U->C direction correlation (full
  window), then mean held-out core-61 Pearson, then smaller mean FWHM error.
* Headline arms are full-batch Adam, 300 iterations, same init and same data
  split as ``fit_defocus_forward_dataset.py`` (this is the deterministic
  configuration that produced the stored ``M1_defocus`` fit). Seed spread is
  measured on a stochastic mini-batch variant, because the full-batch optimiser
  has no random source.

Usage::

    $env:PYTHONPATH="src"
    .venv\\Scripts\\python.exe scripts/slm/loss_compare_20261107.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "slm"))

import torch

from fit_defocus_forward_dataset import load_observations, ramp_matrix
from ao_shaping.algorithm.signal_processing.physical_forward_fit import (
    PhysicalForwardFit,
)

DATASET = ROOT / "data/debug/slm_transfer/defocus_forward_20261107_focus_r2"
REFERENCE_FIT = ROOT / "data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/forward_fit.json"
OLD_FIT = ROOT / "data/debug/slm_transfer/20261107_135252/fit/frozen_parameters.json"
OUT_DIR = ROOT / "data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1"

GRID = 121                      # camera crop side used by the repo loader
CORE_HALF = 30                  # repo's core box half-width (|dx|,|dy| <= 30 px)
CENTRAL = slice(30, 91)         # central 61x61 window of the uc_direction analysis
ITERATIONS = 300

#: Pre-registered composite weights (stated before any run).
COMPOSITE_WEIGHTS = {"masked_mse": 0.4, "pearson": 0.4, "ee_uniformity": 0.2}

LOSSES = (
    "pearson",
    "mse",
    "log_mse",
    "rel_l2",
    "masked_mse",
    "ee_uniformity",
    "composite",
)

PRIMARY_SIGN_MODE = "negated"
LEGACY_SIGN_MODE = "current"
SIGN_MODES = (PRIMARY_SIGN_MODE, LEGACY_SIGN_MODE)
SIGN_LABELS = {
    PRIMARY_SIGN_MODE: "negated (phase_sign=-1, primary)",
    LEGACY_SIGN_MODE: "current (phase_sign=+1, legacy/rejected)",
}

#: Losses whose value depends on the prediction's absolute scale. The other
#: losses are exactly scale-invariant, so their gain is frozen (a scale the
#: loss cannot identify must not be allowed to drift).
SCALE_SENSITIVE = frozenset({"mse", "log_mse", "rel_l2", "masked_mse", "composite"})


def pearson_rows(pred: torch.Tensor, meas: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Per-sample Pearson r over ``mask`` (or the full frame)."""
    if mask is not None:
        weight = mask.to(pred.dtype)
        count = weight.flatten(1).sum(1).clamp_min(1.)
        p = pred.flatten(1) * weight.flatten(1)
        t = meas.flatten(1) * weight.flatten(1)
        p_mean = p.sum(1, keepdim=True) / count.unsqueeze(1)
        t_mean = t.sum(1, keepdim=True) / count.unsqueeze(1)
    else:
        p, t = pred.flatten(1), meas.flatten(1)
        p_mean, t_mean = p.mean(1, keepdim=True), t.mean(1, keepdim=True)
    p_centre, t_centre = p - p_mean, t - t_mean
    return (p_centre * t_centre).sum(1) / (
        p_centre.norm(dim=1) * t_centre.norm(dim=1)
    ).clamp_min(1e-8)


def masked_stats(pred: torch.Tensor, meas: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Encircled-energy fraction and core coefficient of variation, per sample."""
    weight = mask.to(pred.dtype)
    count = weight.flatten(1).sum(1).clamp_min(1.)
    out: dict[str, torch.Tensor] = {}
    for tag, frame in (("pred", pred), ("meas", meas)):
        flat = frame.flatten(1)
        wflat = weight.flatten(1)
        total = flat.sum(1).clamp_min(1e-9)
        core_sum = (flat * wflat).sum(1)
        core_mean = core_sum / count
        core_var = ((flat * flat) * wflat).sum(1) / count - core_mean * core_mean
        out[tag + "_ee"] = core_sum / total
        out[tag + "_cv"] = core_var.clamp_min(0.).sqrt() / core_mean.clamp_min(1e-9)
    return out


class LossFit(PhysicalForwardFit):
    """``PhysicalForwardFit`` with one swappable training loss.

    The parameter set is never touched between arms: only the scalar loss differs
    (plus an optional input-phase negation used to probe the sign convention).
    The sign is pinned explicitly: when the fitter exposes ``phase_sign`` (a
    concurrent issue-#235 change), the arm passes it; otherwise the command
    phase is negated inside :meth:`prediction`. The primary arm uses sign -1;
    sign +1 is retained only as a legacy comparison. A single extra scalar
    ``log_gain`` keeps scale-sensitive losses identifiable; it is frozen for
    the scale-invariant arms, so the Pearson baseline stays bit-identical to
    the repo fitter.
    """

    #: True when the installed fitter takes ``phase_sign`` (issue #235).
    SUPPORTS_PHASE_SIGN = "phase_sign" in inspect.signature(PhysicalForwardFit.__init__).parameters

    def __init__(
        self,
        *args: Any,
        loss_name: str,
        negate_phase: bool,
        core_mask: torch.Tensor,
        loss_scale: float,
        **kwargs: Any,
    ) -> None:
        self.negate_phase = negate_phase
        self.manual_negation = negate_phase and not self.SUPPORTS_PHASE_SIGN
        if self.SUPPORTS_PHASE_SIGN:
            kwargs["phase_sign"] = -1.0 if negate_phase else 1.0
        super().__init__(*args, **kwargs)
        if loss_name not in LOSSES:
            raise ValueError(f"unknown loss {loss_name!r}")
        self.loss_name = loss_name
        self.core_mask = core_mask
        self.loss_scale = loss_scale
        self.log_gain = torch.nn.Parameter(torch.zeros((), device=self._device))
        self.log_gain.requires_grad_(loss_name in SCALE_SENSITIVE)
        self.optimizer.add_param_group({"params": [self.log_gain], "lr": 0.05})
        self.loss_history: list[float] = []

    def prediction(self) -> torch.Tensor:
        if self.manual_negation:
            saved = self.phases
            self.phases = -saved
            try:
                base = super().prediction()
            finally:
                self.phases = saved
        else:
            base = super().prediction()
        return base * torch.exp(self.log_gain)

    def loss_terms(self, pred: torch.Tensor, meas: torch.Tensor,
                   core_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        weight = core_mask.to(pred.dtype)
        p, t = pred.flatten(1), meas.flatten(1)
        wflat = weight.flatten(1)
        mask_norm = ((t * t) * wflat).sum(1).clamp_min(1e-9)
        masked_mse = (((p - t) ** 2) * wflat).sum(1) / mask_norm
        stats = masked_stats(pred, meas, core_mask)
        terms = {
            "pearson": (1. - pearson_rows(pred, meas)).mean(),
            "mse": ((p - t) ** 2).mean() / self.loss_scale,
            "log_mse": ((torch.log1p(pred.clamp_min(0.)) - torch.log1p(meas)) ** 2).mean(),
            "rel_l2": ((((p - t) ** 2).sum(1)) / ((t * t).sum(1)).clamp_min(1e-9)).mean(),
            "masked_mse": masked_mse.mean(),
            "ee_uniformity": (stats["pred_ee"] - stats["meas_ee"]).abs().mean()
            + (stats["pred_cv"] - stats["meas_cv"]).abs().mean(),
        }
        terms["composite"] = sum(
            COMPOSITE_WEIGHTS[name] * terms[name] for name in COMPOSITE_WEIGHTS
        )
        return terms

    def update(self) -> float:
        self.optimizer.zero_grad()
        pred = self.prediction()[self.train]
        meas = self.images[self.train]
        loss = self.loss_terms(pred, meas, self.core_mask[self.train])[self.loss_name]
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.d.clamp_(-2., 2.)
            self.center.clamp_(0., 1920.)
            self.log_width.clamp_(float(np.log(80)), float(np.log(800)))
            self.log_gain.clamp_(-4., 8.)
        value = float(loss.detach())
        self.loss_history.append(value)
        return value

    def run(self, iterations: int) -> np.ndarray:
        for _ in range(iterations):
            self.update()
        return self.prediction().detach().cpu().numpy()


def area_fwhm(frame: np.ndarray) -> float:
    """Orientation-free half-maximum width: 2*sqrt(area(>=0.5 peak)/pi)."""
    peak = float(frame.max())
    if peak <= 0.:
        return float("nan")
    return float(2. * np.sqrt(float((frame >= 0.5 * peak).sum()) / np.pi))


def build_data() -> dict[str, Any]:
    rows = json.loads((DATASET / "manifest.json").read_text())
    config = json.loads((DATASET / "run_config.json").read_text())
    matrix = ramp_matrix(rows)
    references = {r["branch"]: r for r in rows if r["label"] == "reference_start"}
    center = np.array([references["U"]["spot"][f"centroid_{k}"] for k in ("x", "y")])
    selected, phases, images, camera_xy, references = load_observations(DATASET, center)
    train = np.array([r["split"] != "test" for r in selected])

    # Core box masks for the loss: side 3 x measured FWHM, centred on the
    # measured spot of that sample (fallback: the repo's +-30 px core).
    masks = []
    for row in selected:
        fwhm = float(row["spot"]["fwhm_px"])
        half = max(0.5, min(CORE_HALF, 1.5 * fwhm))
        cx = float(row["spot"]["centroid_x"])
        cy = float(row["spot"]["centroid_y"])
        masks.append((np.abs(camera_xy[..., 0] - cx) <= half) & (np.abs(camera_xy[..., 1] - cy) <= half))
    core_mask = torch.tensor(np.stack(masks), dtype=torch.bool)

    # Repo-comparable core-61 mask for metrics only.
    metric_mask = []
    for row in selected:
        branch_center = np.array(
            [references[row["branch"]]["spot"][f"centroid_{k}"] for k in ("x", "y")]
        )
        metric_mask.append(np.all(np.abs(camera_xy - branch_center) <= CORE_HALF, axis=-1))
    metric_mask = np.stack(metric_mask)

    pairs = []
    by_label: dict[str, dict[str, int]] = {}
    for index, row in enumerate(selected):
        by_label.setdefault(row["label"], {})[row["branch"]] = index
    for label, branches in by_label.items():
        if "U" in branches and "C" in branches:
            order = by_label[label]
            if all(selected[order[b]]["split"] == "test" for b in ("U", "C")):
                pairs.append((label, order["U"], order["C"]))

    return {
        "rows": rows,
        "config": config,
        "matrix": matrix,
        "references": references,
        "center": center,
        "selected": selected,
        "phases": phases,
        "images": images,
        "camera_xy": camera_xy,
        "train": train,
        "core_mask": core_mask,
        "metric_mask": metric_mask,
        "pairs": pairs,
        "loss_scale": float(np.mean(images ** 2)),
    }


def fit_once(
    data: dict[str, Any],
    loss_name: str,
    sign_mode: str,
    seed: int,
    iterations: int,
    *,
    batch_size: int | None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    start = time.perf_counter()
    model = LossFit(
        data["phases"],
        data["images"],
        data["camera_xy"],
        data["matrix"],
        data["config"]["best_diopter_m1"],
        data["train"],
        np.array(json.loads(REFERENCE_FIT.read_text())["illumination_center_estimate"]),
        learn_amplitude=False,
        initial_defocus=None,
        loss_name=loss_name,
        negate_phase=sign_mode == "negated",
        core_mask=data["core_mask"],
        loss_scale=data["loss_scale"],
        batch_size=batch_size,
    )
    with torch.no_grad():
        model.origin.copy_(torch.tensor(data["center"], dtype=torch.float32))
        # Closed-form gain at the init point: a nuisance scale, not a physical
        # parameter, so every arm starts at the same least-squares scale.
        if loss_name in SCALE_SENSITIVE:
            p0 = model.prediction()[model.train]
            t0 = model.images[model.train]
            initial_gain = (p0 * t0).sum() / (p0 * p0).sum().clamp_min(1e-12)
            model.log_gain.fill_(initial_gain.log())

    if batch_size is None:
        model.run(iterations)
        raw = model.prediction().detach().cpu().numpy()
        sampled = [float(v) for v in model.loss_history]
    else:
        # Stochastic variant: seeded mini-batches, same iteration budget.
        rng = np.random.default_rng(seed)
        train_index = torch.tensor(np.flatnonzero(data["train"]))
        history = []
        for _ in range(iterations):
            take = rng.choice(train_index.numpy(), size=min(batch_size, len(train_index)), replace=False)
            model.train = torch.zeros(len(data["train"]), dtype=torch.bool)
            model.train[torch.tensor(np.sort(take))] = True
            history.append(model.update())
        model.train = torch.tensor(data["train"], dtype=torch.bool)
        raw = model.prediction().detach().cpu().numpy()
        sampled = history

    train_mask = data["train"]
    measured = data["images"]
    design = np.stack((raw[train_mask].ravel(), np.ones(raw[train_mask].size)), axis=1)
    gain, background = nnls(design, measured[train_mask].ravel())[0]
    calibrated = raw * gain + background
    elapsed = time.perf_counter() - start
    return {
        "loss": loss_name,
        "sign_mode": sign_mode,
        "seed": seed,
        "variant": "full_batch" if batch_size is None else f"minibatch{batch_size}",
        "iterations": iterations,
        "elapsed_s": elapsed,
        "parameters": {
            "defocus_diopter": float(model.d.item()),
            "origin": [float(v) for v in model.origin.detach().cpu().tolist()],
            "amplitude_center": [float(v) for v in model.center.detach().cpu().tolist()],
            "amplitude_width": [float(v) for v in model.log_width.exp().detach().cpu().tolist()],
            "log_gain": float(model.log_gain.item()),
            "gain": float(gain),
            "background": float(background),
        },
        "loss_history": [float(v) for v in model.loss_history],
        "sampled_loss_history": sampled,
        "predictions": calibrated,
        "raw_predictions": raw,
    }


def evaluate(data: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    selected = data["selected"]
    test = ~data["train"]
    measured = data["images"]
    calibrated = run["predictions"]
    core = data["metric_mask"]
    rows = []
    for index in np.flatnonzero(test):
        p, t = calibrated[index], measured[index]
        rows.append({
            "label": selected[index]["label"],
            "branch": selected[index]["branch"],
            "pearson_full": float(np.corrcoef(p.ravel(), t.ravel())[0, 1]),
            "pearson_core61": float(np.corrcoef(p[core[index]], t[core[index]])[0, 1]),
            "rel_l2": float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-9)),
            "fwhm_pred": area_fwhm(p),
            "fwhm_meas_area": area_fwhm(t),
            "fwhm_meas_manifest": float(selected[index]["spot"]["fwhm_px"]),
            "ee_roi_error": abs(
                float(p[core[index]].sum() / p.sum()) - float(t[core[index]].sum() / t.sum())
            ),
            "cv_roi_error": abs(
                float(p[core[index]].std() / p[core[index]].mean())
                - float(t[core[index]].std() / t[core[index]].mean())
            ),
        })
    for row in rows:
        row["fwhm_error"] = abs(row["fwhm_pred"] - row["fwhm_meas_area"])

    direction: dict[str, Any] = {}
    for name, window in (("full", slice(0, GRID)), ("central61", CENTRAL)):
        corrs = []
        for _label, u_index, c_index in data["pairs"]:
            dmeas = (measured[c_index] - measured[u_index])[window]
            dpred = (calibrated[c_index] - calibrated[u_index])[window]
            dmeas = dmeas - dmeas.mean()
            dpred = dpred - dpred.mean()
            corrs.append(float((dmeas * dpred).sum() / max(np.linalg.norm(dmeas) * np.linalg.norm(dpred), 1e-12)))
        corrs = np.array(corrs)
        direction[name] = {
            "n_pairs": int(corrs.size),
            "median": float(np.median(corrs)),
            "mean": float(corrs.mean()),
            "frac_above_0.5": float((corrs > 0.5).mean()),
            "per_pair": [float(v) for v in corrs],
        }

    return {
        "test_rows": rows,
        "mean": {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "pearson_full",
                "pearson_core61",
                "rel_l2",
                "fwhm_pred",
                "fwhm_meas_area",
                "fwhm_error",
                "ee_roi_error",
                "cv_roi_error",
            )
        },
        "direction": direction,
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate evaluated runs over seeds per (loss, sign, variant)."""
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for run in runs:
        buckets.setdefault((run["loss"], run["sign_mode"], run["variant"]), []).append(run)
    summary: dict[str, Any] = {}
    for (loss, sign, variant), group in buckets.items():
        metrics = group[0]["evaluation"]["mean"].keys()
        entry: dict[str, Any] = {"n_seeds": len(group), "seeds": sorted(r["seed"] for r in group)}
        entry["defocus_diopter"] = {
            "mean": float(np.mean([r["parameters"]["defocus_diopter"] for r in group])),
            "std": float(np.std([r["parameters"]["defocus_diopter"] for r in group])),
        }
        entry["metrics"] = {
            key: {
                "mean": float(np.mean([r["evaluation"]["mean"][key] for r in group])),
                "std": float(np.std([r["evaluation"]["mean"][key] for r in group])),
            }
            for key in metrics
        }
        for window in ("full", "central61"):
            entry["direction_" + window] = {
                "median_mean": float(np.mean([r["evaluation"]["direction"][window]["median"] for r in group])),
                "median_std": float(np.std([r["evaluation"]["direction"][window]["median"] for r in group])),
                "frac_above_0.5_mean": float(
                    np.mean([r["evaluation"]["direction"][window]["frac_above_0.5"] for r in group])
                ),
            }
        summary[f"{loss}|{sign}|{variant}"] = entry
    return summary


def rank(summary: dict[str, Any], sign_mode: str = PRIMARY_SIGN_MODE) -> list[dict[str, Any]]:
    """Pre-registered ordering for one sign arm: direction median, then core Pearson, then FWHM error.

    The primary arm is ``PRIMARY_SIGN_MODE`` (phase_sign=-1, the convention the
    hardware evidence of issue #235 supports); ``LEGACY_SIGN_MODE`` is ranked only
    for the labelled comparison.
    """
    rows = []
    for key, entry in summary.items():
        loss, sign, variant = key.split("|")
        if variant != "full_batch" or sign != sign_mode:
            continue
        rows.append({
            "loss": loss,
            "sign_mode": sign,
            "direction_median": entry["direction_full"]["median_mean"],
            "pearson_core61": entry["metrics"]["pearson_core61"]["mean"],
            "fwhm_error": entry["metrics"]["fwhm_error"]["mean"],
        })
    rows.sort(key=lambda r: (-r["direction_median"], -r["pearson_core61"], r["fwhm_error"]))
    for position, row in enumerate(rows, start=1):
        row["rank"] = position
    return rows


def reproduction_check(baseline_run: dict[str, Any]) -> dict[str, Any]:
    """Compare the primary (-1) Pearson arm with the archived ``M1_defocus`` fit."""
    stored = json.loads(REFERENCE_FIT.read_text())["models"]["M1_defocus"]
    stored_rows = stored["rows"]
    reproduced = baseline_run["parameters"]
    origin_delta = np.abs(np.array(reproduced["origin"]) - np.array(stored["origin"]))
    check = {
        "stored": {
            "defocus_diopter": float(stored["defocus_diopter"]),
            "origin": [float(v) for v in stored["origin"]],
            "mean_corr_all_rows": float(np.mean([row["corr"] for row in stored_rows])),
        },
        "reproduced": {
            "defocus_diopter": float(reproduced["defocus_diopter"]),
            "origin": [float(v) for v in reproduced["origin"]],
            "mean_corr_test_full": float(baseline_run["evaluation"]["mean"]["pearson_full"]),
        },
        "defocus_abs_delta": float(abs(reproduced["defocus_diopter"] - stored["defocus_diopter"])),
        "origin_max_abs_delta_px": float(origin_delta.max()),
        "tolerance": {"defocus_abs": 5e-3, "origin_px": 1.0},
    }
    check["passed"] = bool(
        check["defocus_abs_delta"] <= check["tolerance"]["defocus_abs"]
        and check["origin_max_abs_delta_px"] <= check["tolerance"]["origin_px"]
    )
    return check


def narrative(
    ranking: list[dict[str, Any]],
    summary: dict[str, Any],
    legacy_ranking: list[dict[str, Any]] | None = None,
) -> str:
    """Mechanical reading of the ranking, so the report never drifts from the data."""
    legacy_ranking = legacy_ranking or []
    if not ranking:
        return "No full-batch arm completed; nothing to recommend."
    baseline = next(
        (row for row in ranking if row["loss"] == "pearson"),
        None,
    )
    best = ranking[0]
    parts = []
    if baseline is not None:
        parts.append(
            f"Against the Pearson baseline in the primary phase_sign=-1 arm (direction median "
            f"{baseline['direction_median']:+.3f}, core-61 r {baseline['pearson_core61']:.3f}), "
            f"{best['loss']} moves the direction median to "
            f"{best['direction_median']:+.3f} and core-61 r to {best['pearson_core61']:.3f}."
        )
    legacy_best = legacy_ranking[0] if legacy_ranking else None
    if legacy_best is not None:
        parts.append(
            f"Legacy phase_sign=+1 comparison (rejected convention, not used for selection): "
            f"best {legacy_best['loss']} with direction median {legacy_best['direction_median']:+.3f}."
        )
    parts.append(
        "Pixel-fit metrics (core-61 r, FWHM error) and the direction metric do not "
        "necessarily rank the arms the same way; the ranking above follows the "
        "pre-registered direction-first rule as a proxy only, not a closed-loop loss selection."
    )
    return " ".join(parts)


def write_report(
    out_dir: Path,
    results: dict[str, Any],
    summary: dict[str, Any],
    ranking: list[dict[str, Any]],
) -> None:
    legacy_ranking = results.get("legacy_ranking", [])
    lines = [
        "# Defocus forward model: training-loss comparison (issue #236 groundwork)",
        "",
        "Offline analysis only: no SLM, camera or laser was occupied, and no",
        "`ao_shaping.drivers` module was imported. Every number comes from data",
        "already on disk under `data/debug/slm_transfer/`.",
        "",
        "**Primary sign convention: phase_sign = -1 (negate_phase=True, the \"negated\" arm).** "
        "Issue #235 hardware evidence shows the panel phase enters propagation with sign -1; "
        "with +1 all 42 single-mode Zernike screens gave negative signed direction agreement "
        "(median ~ -0.47). The phase_sign=+1 arm is kept only as a labelled legacy comparison "
        "and was NOT used for selection. The earlier recommendation (masked_mse + current) was "
        "made under the rejected convention and is superseded.",
        "",
        "The direction metric is a proxy. The U/C data is radially symmetric and insensitive to "
        "polarity, so this report does NOT select a loss for closed loop.",
        "",
        f"Generated: {results['generated_utc']}",
        f"Command: `{results['command']}`",
        "",
        "## Setup",
        "",
        f"- Dataset: `{results['dataset']['directory']}` — {results['dataset']['sample_count']} "
        f"settled samples ({results['dataset']['train']} train / {results['dataset']['test']} test), "
        f"{results['dataset']['pairs']} held-out U/C pairs.",
        f"- Fitter: repo `PhysicalForwardFit` (M1 configuration, fixed Gaussian amplitude), "
        f"{ITERATIONS} Adam iterations, same init "
        f"(d0 = {results['config']['initial_defocus']:.6f} 1/m, origin = "
        f"{results['config']['initial_origin']}).",
        "- Only the scalar training loss changes between arms. A single nuisance "
        "scalar `log_gain` (initialised in closed form at the init point) keeps "
        "scale-sensitive losses identifiable; it is frozen for the exactly "
        "scale-invariant arms (Pearson, EE/uniformity). The primary -1 Pearson arm is "
        "checked against archived `M1_defocus`; reproduction is not assumed.",
        f"- Pre-registered composite weights: {COMPOSITE_WEIGHTS}.",
        "- Pre-registered ranking (primary phase_sign=-1 arm): (1) median per-pair U→C direction "
        "correlation (full window), (2) mean held-out core-61 Pearson, (3) smaller mean FWHM error.",
        "",
    ]
    validation = results.get("validation")
    if validation:
        lines += [
            "## Fit-reproduction check",
            "",
            f"- Stored `M1_defocus` d = {validation['stored']['defocus_diopter']:.9f}; "
            f"Pearson-baseline arm d = {validation['reproduced']['defocus_diopter']:.9f}; "
            f"|Δ| = {validation['defocus_abs_delta']:.2e} "
            f"(tolerance {validation['tolerance']['defocus_abs']:g}).",
            f"- Origin max |Δ| = {validation['origin_max_abs_delta_px']:.3f} px "
            f"(tolerance {validation['tolerance']['origin_px']:g} px).",
            f"- Result: **{'PASSED' if validation['passed'] else 'FAILED'}**.",
            "",
        ]
    lines += [
        "## Results (held-out test split)",
        "",
        "| loss | sign | dir median (full) | dir frac >0.5 | core-61 r | full r | rel-L2 | "
        "FWHM err (px) | EE err | CV err | d (1/m) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, entry in summary.items():
        loss, sign, variant = key.split("|")
        if variant != "full_batch":
            continue
        d = entry["defocus_diopter"]
        m = entry["metrics"]
        lines.append(
            f"| {loss} | {SIGN_LABELS[sign]} | {entry['direction_full']['median_mean']:+.3f} ± "
            f"{entry['direction_full']['median_std']:.3f} | "
            f"{entry['direction_full']['frac_above_0.5_mean']:.2f} | "
            f"{m['pearson_core61']['mean']:.3f} | {m['pearson_full']['mean']:.3f} | "
            f"{m['rel_l2']['mean']:.3f} | {m['fwhm_error']['mean']:.2f} | "
            f"{m['ee_roi_error']['mean']:.3f} | {m['cv_roi_error']['mean']:.3f} | "
            f"{d['mean']:+.4f} ± {d['std']:.4f} |"
        )
    lines += [
        "",
        "`dir` = median over held-out pairs of the cosine between the measured and "
        "predicted C−U differential; `core-61` = repo's ±30 px core box; `rel-L2` = "
        "‖pred−meas‖/‖meas‖ after the NNLS gain; FWHM is the area-equivalent half-max "
        "width (2√(area/π)); EE/CV are measured on the ±30 px core box.",
        "",
        "## Ranking, PRIMARY phase_sign=-1 (pre-registered rule, full-batch arms)",
        "",
        "| rank | loss | sign | dir median | core-61 r | FWHM err |",
        "|---|---|---|---|---|---|",
    ]
    for row in ranking:
        lines.append(
            f"| {row['rank']} | {row['loss']} | {row['sign_mode']} | "
            f"{row['direction_median']:+.3f} | {row['pearson_core61']:.3f} | {row['fwhm_error']:.2f} |"
        )
    lines += [
        "",
        "## LEGACY comparison, phase_sign=+1 (rejected convention, not used for selection)",
        "",
        "| rank | loss | sign | dir median | core-61 r | FWHM err |",
        "|---|---|---|---|---|---|",
    ]
    for row in legacy_ranking:
        lines.append(
            f"| {row['rank']} | {row['loss']} | {row['sign_mode']} | "
            f"{row['direction_median']:+.3f} | {row['pearson_core61']:.3f} | {row['fwhm_error']:.2f} |"
        )
    best = results["best"]
    lines += [
        "",
        "## Recommendation",
        "",
        f"Best arm in the primary phase_sign=-1 ranking: **{best['loss']}** with input phase "
        f"**{best['sign_mode']}** (direction median {best['direction_median']:+.3f}, "
        f"core-61 r {best['pearson_core61']:.3f}, FWHM error {best['fwhm_error']:.2f} px). "
        f"Fit-quality/proxy ranking only: it does NOT select a loss for closed loop. "
        f"See `best_loss_parameters.json`.",
        "",
        results["narrative"],
        "",
        "## Caveats",
        "",
        "- 43 samples / 9 held-out pairs: differences below roughly 0.05 in the "
        "direction median are inside the pair-to-pair spread; ranking is indicative.",
        "- The direction metric is a proxy: U/C data is radially symmetric and insensitive "
        "to polarity. It does not select a loss for closed loop. The primary sign is fixed "
        "by independent issue #235 hardware evidence, not selected from these U/C data.",
        "- `mse` is divided by the dataset's mean squared irradiance (a positive "
        "constant): the argmin is unchanged, only conditioning improves.",
        "- The composite weights and the ranking rule were fixed before the first run.",
        "",
        "## Files",
        "",
        "- `results.json` — every run, history, per-sample metric and summary.",
        "- `best_loss_parameters.json` — parameters of the recommended arm.",
        "- `REPORT.md` — this file.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--output", "--out", dest="out", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-minibatch", action="store_true",
                        help="Skip the stochastic seed-spread variant (debug runs)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logger.add(args.out / "run.log", mode="w", encoding="utf-8")
    logger.info("Offline only; no SLM, camera or laser occupied. Primary phase_sign=-1; +1 legacy.")

    torch.set_num_threads(4)
    started = datetime.now(timezone.utc)
    data = build_data()
    data["config"]["initial_defocus"] = float(-data["config"]["best_diopter_m1"])
    torch.manual_seed(0)

    runs: list[dict[str, Any]] = []
    for sign_mode in SIGN_MODES:
        for loss_name in LOSSES:
            run = fit_once(data, loss_name, sign_mode, 0, args.iterations, batch_size=None)
            run["evaluation"] = evaluate(data, run)
            runs.append(run)
            logger.info(f"[{sign_mode}] {loss_name}: d={run['parameters']['defocus_diopter']:+.5f} "
                        f"dir={run['evaluation']['direction']['full']['median']:+.3f} "
                        f"core61={run['evaluation']['mean']['pearson_core61']:.3f} "
                        f"({run['elapsed_s']:.1f}s)")

    # Determinism proof: the full-batch optimiser is seeded but has no random
    # source, so repeats must be identical. Confirm on three arms.
    if not args.skip_minibatch:
        for loss_name in ("pearson", "composite", "masked_mse"):
            for seed in (1, 2):
                run = fit_once(data, loss_name, PRIMARY_SIGN_MODE, seed, args.iterations, batch_size=None)
                run["evaluation"] = evaluate(data, run)
                runs.append(run)
        # Stochastic spread: seeded mini-batches, same budget.
        for loss_name in ("pearson", "composite"):
            for seed in (1, 2, 3):
                run = fit_once(data, loss_name, PRIMARY_SIGN_MODE, seed, args.iterations, batch_size=8)
                run["evaluation"] = evaluate(data, run)
                runs.append(run)

    summary = summarize(runs)
    ranking = rank(summary, PRIMARY_SIGN_MODE)
    legacy_ranking = rank(summary, LEGACY_SIGN_MODE)
    baseline_run = next(
        (
            run for run in runs
            if run["loss"] == "pearson" and run["sign_mode"] == PRIMARY_SIGN_MODE
            and run["variant"] == "full_batch" and run["seed"] == 0
        ),
        None,
    )
    validation = reproduction_check(baseline_run) if baseline_run is not None else None

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run.pop("predictions")
        run.pop("raw_predictions")

    best_row = ranking[0] if ranking else None
    best_run = None
    if best_row is not None:
        best_run = next(
            r for r in runs
            if r["loss"] == best_row["loss"] and r["sign_mode"] == best_row["sign_mode"]
            and r["variant"] == "full_batch" and r["seed"] == 0
        )
    command = " ".join([sys.executable, str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]])
    results: dict[str, Any] = {
        "generated_utc": started.isoformat(timespec="seconds"),
        "command": command,
        "device_occupied": False,
        "dataset": {
            "directory": str(DATASET.relative_to(ROOT)),
            "manifest_rows": len(data["rows"]),
            "sample_count": len(data["selected"]),
            "train": int(data["train"].sum()),
            "test": int((~data["train"]).sum()),
            "pairs": len(data["pairs"]),
            "exposure_normalised": True,
            "crop": f"{GRID}x{GRID}",
        },
        "reference_fit": {
            "path": str(REFERENCE_FIT.relative_to(ROOT)),
            "defocus_diopter": 0.11386794596910477,
            "mean_corr_stored": 0.831,
            "old_optics_fit": str(OLD_FIT.relative_to(ROOT)),
        },
        "config": {
            "iterations": args.iterations,
            "losses": list(LOSSES),
            "sign_modes": list(SIGN_MODES),
            "sign_labels": SIGN_LABELS,
            "primary_phase_sign": -1,
            "legacy_phase_sign": 1,
            "optimizer": "Adam(d lr=0.01, origin lr=1.0, log_gain lr=0.05 when scale-sensitive)",
            "initial_defocus": data["config"]["initial_defocus"],
            "initial_origin": [float(v) for v in data["center"]],
            "initial_log_gain": 0.0,
            "phase_sign_parameter_available": LossFit.SUPPORTS_PHASE_SIGN,
            "loss_scale": data["loss_scale"],
            "core_box": "side 3 x measured FWHM, clamped to +-30 px",
        },
        "preregistered": {
            "composite_weights": COMPOSITE_WEIGHTS,
            "ranking_rule": [
                "median per-pair U->C direction correlation (full window, descending)",
                "mean held-out core-61 Pearson (descending)",
                "mean FWHM area error (ascending)",
            ],
            "metrics": [
                "pearson_full", "pearson_core61", "rel_l2", "fwhm_error",
                "ee_roi_error", "cv_roi_error", "direction_full", "direction_central61",
            ],
        },
        "runs": runs,
        "summary": summary,
        "validation": validation,
        "ranking": ranking,
        "legacy_ranking": legacy_ranking,
        "primary_sign_mode": PRIMARY_SIGN_MODE,
        "legacy_sign_mode": LEGACY_SIGN_MODE,
        "best": {
            "loss": best_row["loss"],
            "sign_mode": best_row["sign_mode"],
            "direction_median": best_row["direction_median"],
            "pearson_core61": best_row["pearson_core61"],
            "fwhm_error": best_row["fwhm_error"],
        } if best_row else None,
        "narrative": narrative(ranking, summary, legacy_ranking),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    if best_run is not None:
        payload = {
            "source": "loss_compare_20261107.py",
            "loss": best_run["loss"],
            "sign_mode": best_run["sign_mode"],
            "variant": best_run["variant"],
            "iterations": best_run["iterations"],
            "parameters": best_run["parameters"],
            "metrics": best_run["evaluation"]["mean"],
            "direction": best_run["evaluation"]["direction"],
        }
        (out_dir / "best_loss_parameters.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    write_report(out_dir, results, summary, ranking)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())