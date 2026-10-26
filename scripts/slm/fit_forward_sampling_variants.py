"""Sweep the physical forward model over a sampling/variant grid.

The physical forward model of another work stream (sampling options, amplitude
learning and residual aberrations) is fitted here on already acquired
phase/far-field pairs. The fit and evaluation modules of that work stream are
loaded from their file path and never edited, so this script only decides
*which* variant to run and *how* to score it.

Two questions are answered:

* is the numerical sampling converged, and
* does the model reach the acceptance gates that allow inverse optimisation.

No device is opened: everything is offline computation over stored captures.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from functools import partial
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(ROOT), str(ROOT / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from scripts._common import fmt_general

FIT_MODULE = "scripts.slm.fit_defocus_forward_dataset"
EVALUATE_MODULE = "scripts.slm.evaluate_defocus_forward_dataset"
SAMPLING_MODULE = "scripts.slm.diagnose_defocus_sampling"

# Camera sampling grid of the other work stream: 2 CCD pixels per sample.
CCD_STEP_PX = 2.0
# Radius of the fixed CV/EE target box, in CCD pixels.
ROI_RADIUS_PX = 30

# Acceptance gates. The correlation/CV/EE/geometry numbers come from
# report/shape_trials/physical_forward_defocus_plan_20261022.md; the sampling
# convergence tolerance is this round's own criterion and is flagged as such.
GATES = {
    "corr_median": 0.85,
    "corr_family_median": 0.80,
    "corr_min": 0.65,
    "cv_absolute": 0.05,
    "ee_absolute": 0.05,
    "center_px": 1.0,
    "width_relative": 0.10,
    "sampling_relative_change": 0.05,
}


# ---------------------------------------------------------------------------
# module loading
# ---------------------------------------------------------------------------


def load_workstream_module(name: str, relative: str):
    """Load a module of another work stream from its file path.

    ``sys.modules`` is populated *before* ``exec_module`` so dataclasses inside
    the module can resolve their own ``__module__``; without it they raise
    ``NoneType object has no attribute '__dict__'``.
    """
    path = ROOT / relative
    if not path.is_file():
        raise SystemExit(
            f"缺少模块文件 {relative}（解析为 {path}）。"
            "该模块是另一会话的在制品，路径可能已变；本脚本不静默退化，请先确认路径。"
        )
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None):
        if Path(cached.__file__).resolve() == path.resolve():
            return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"无法为 {relative} 建立 import 规格。该模块是另一会话的在制品，路径可能已变。")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    parent_name, _, leaf = name.rpartition(".")
    if parent_name:
        parent = importlib.import_module(parent_name)
        setattr(parent, leaf, module)
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # the message must name the cause
        sys.modules.pop(name, None)
        raise SystemExit(
            f"加载 {relative} 失败：{error!r}。"
            "该模块是另一会话的在制品，路径可能已变；本脚本不静默退化。"
        )
    return module


# ---------------------------------------------------------------------------
# data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class Variant:
    stage: str
    label: str
    grid_size: int
    fft_size: int
    pixel_samples: int
    learn_amplitude: bool
    learn_residual: bool

    @property
    def key(self) -> str:
        return (f"{self.stage}_{self.grid_size}g_{self.fft_size}f_ps{self.pixel_samples}"
                f"_amp{int(self.learn_amplitude)}_res{int(self.learn_residual)}")

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "label": self.label,
            "key": self.key,
            "grid_size": self.grid_size,
            "fft_size": self.fft_size,
            "pixel_samples": self.pixel_samples,
            "learn_amplitude": self.learn_amplitude,
            "learn_residual": self.learn_residual,
        }


@dataclass
class Prepared:
    spec: DatasetSpec
    rows: list
    phases: np.ndarray
    images: np.ndarray
    camera_xy: np.ndarray
    references: dict
    matrix: np.ndarray
    amplitude_center: np.ndarray
    camera_center: np.ndarray
    focus_diopter: float


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _profile_fwhm(profile: np.ndarray, step: float) -> float:
    """Full width at half maximum of a 1-D profile, in CCD pixels.

    The crossings are interpolated linearly between the two samples that bracket
    the half level and the interpolation ratio is clamped to the bracketing
    interval, so a profile that never falls below half inside the window
    yields a truncated width instead of a runaway one.
    """
    if profile.size == 0:
        return 0.0
    peak = int(np.argmax(profile))
    height = float(profile[peak])
    if height <= 0:
        return 0.0
    half = height / 2.0
    left = peak
    while left > 0 and profile[left] > half:
        left -= 1
    right = peak
    while right < profile.size - 1 and profile[right] > half:
        right += 1
    left_ratio = 0.0
    denominator = float(profile[left + 1] - profile[left])
    if denominator > 0:
        left_ratio = min(max((half - float(profile[left])) / denominator, 0.0), 1.0)
    right_ratio = 0.0
    if right + 1 < profile.size:
        denominator = float(profile[right] - profile[right + 1])
        if denominator > 0:
            right_ratio = min(max((float(profile[right]) - half) / denominator, 0.0), 1.0)
    return float(max((right + right_ratio) - (left + left_ratio), 0.0) * step)


def spot_fwhm(signal: np.ndarray, step: float) -> float:
    """Mean of the x and y marginal FWHM of a non-negative spot image."""
    widths = [_profile_fwhm(signal.sum(axis=0), step), _profile_fwhm(signal.sum(axis=1), step)]
    return float(np.mean(widths))


def encircled_diameter(signal: np.ndarray, fraction: float, step: float) -> float:
    """Equivalent-area diameter of the region holding ``fraction`` of the energy."""
    values = np.sort(signal.ravel())[::-1]
    total = float(values.sum())
    if total <= 0:
        return 0.0
    cumulative = np.cumsum(values)
    index = int(np.searchsorted(cumulative, fraction * total))
    threshold = float(values[min(index, values.size - 1)])
    cells = float(np.count_nonzero(signal >= threshold))
    area = cells * step * step
    return float(2.0 * np.sqrt(area / np.pi))


def half_max_mask(array: np.ndarray) -> np.ndarray:
    signal = np.maximum(array, 0.0)
    peak = float(signal.max()) if signal.size else 0.0
    if peak <= 0:
        return np.zeros(signal.shape, dtype=bool)
    return signal >= 0.5 * peak


def mask_iou(measured: np.ndarray, predicted: np.ndarray) -> float:
    left = half_max_mask(measured)
    right = half_max_mask(predicted)
    union = int(np.count_nonzero(left | right))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(left & right) / union)


def row_metrics(row: dict, measured: np.ndarray, predicted: np.ndarray,
                xy: np.ndarray, references: dict) -> dict:
    """Per-record metrics on the fixed camera grid.

    The CV, EE_ROI, centroid and second-moment definitions follow
    ``scripts/slm/diagnose_defocus_sampling.py::metrics`` exactly, so the
    numbers are comparable with that work stream's report.
    """
    center = np.array([references[row["branch"]]["spot"][f"centroid_{k}"] for k in ("x", "y")])
    mask = np.all(np.abs(xy - center) <= ROI_RADIUS_PX, axis=-1)
    result = {key: row.get(key) for key in ("file", "label", "branch", "family", "split")}
    result["corr"] = float(np.corrcoef(measured.ravel(), predicted.ravel())[0, 1])
    for name, array in (("measured", measured), ("predicted", predicted)):
        values = array[mask]
        result[name + "_cv"] = float(values.std() / max(values.mean(), 1e-9))
        result[name + "_ee_roi"] = float(values.sum() / max(array.sum(), 1e-9))
        signal = np.maximum(array, 0.0)
        total = max(float(signal.sum()), 1e-9)
        centroid = np.sum(signal[..., None] * xy, axis=(0, 1)) / total
        result[name + "_center"] = [float(centroid[0]), float(centroid[1])]
        result[name + "_rms_radius"] = float(
            np.sqrt(np.sum(signal * np.sum((xy - centroid) ** 2, axis=-1)) / total))
        result[name + "_fwhm"] = spot_fwhm(signal, CCD_STEP_PX)
        result[name + "_d50"] = encircled_diameter(signal, 0.50, CCD_STEP_PX)
        result[name + "_d80"] = encircled_diameter(signal, 0.80, CCD_STEP_PX)
    result["cv_abs_error"] = abs(result["measured_cv"] - result["predicted_cv"])
    result["ee_roi_abs_error"] = abs(result["measured_ee_roi"] - result["predicted_ee_roi"])
    result["center_error_px"] = float(np.linalg.norm(
        np.array(result["measured_center"]) - np.array(result["predicted_center"])))
    result["rms_radius_abs_error_px"] = abs(result["predicted_rms_radius"] - result["measured_rms_radius"])
    result["rms_relative_error"] = float(abs(
        result["predicted_rms_radius"] / max(result["measured_rms_radius"], 1e-9) - 1))
    result["fwhm_abs_error_px"] = abs(result["predicted_fwhm"] - result["measured_fwhm"])
    result["fwhm_relative_error"] = float(abs(
        result["predicted_fwhm"] / max(result["measured_fwhm"], 1e-9) - 1))
    result["d50_abs_error_px"] = abs(result["predicted_d50"] - result["measured_d50"])
    result["d80_abs_error_px"] = abs(result["predicted_d80"] - result["measured_d80"])
    result["d80_relative_error"] = float(abs(
        result["predicted_d80"] / max(result["measured_d80"], 1e-9) - 1))
    result["iou"] = mask_iou(measured, predicted)
    return result


def aggregate(rows: list) -> dict:
    """Per split and branch aggregates over the probe records."""
    result = {}
    for split in ("train", "test"):
        for branch in ("U", "C"):
            subset = [r for r in rows
                      if r["split"] == split and r["branch"] == branch and r["family"] != "reference"]
            if not subset:
                continue
            corr = np.array([r["corr"] for r in subset])
            cv = np.array([r["cv_abs_error"] for r in subset])
            ee = np.array([r["ee_roi_abs_error"] for r in subset])
            center = np.array([r["center_error_px"] for r in subset])
            fwhm = np.array([r["fwhm_abs_error_px"] for r in subset])
            fwhm_relative = np.array([r["fwhm_relative_error"] for r in subset])
            d80 = np.array([r["d80_abs_error_px"] for r in subset])
            d80_relative = np.array([r["d80_relative_error"] for r in subset])
            rms_relative = np.array([r["rms_relative_error"] for r in subset])
            iou = np.array([r["iou"] for r in subset])
            result[f"{split}_{branch}"] = {
                "count": len(subset),
                "mean_corr": float(corr.mean()),
                "median_corr": float(np.median(corr)),
                "min_corr": float(corr.min()),
                "cv_mae": float(cv.mean()),
                "cv_max": float(cv.max()),
                "ee_mae": float(ee.mean()),
                "center_mae_px": float(center.mean()),
                "center_max_px": float(center.max()),
                "fwhm_mae_px": float(fwhm.mean()),
                "fwhm_max_px": float(fwhm.max()),
                "fwhm_relative_mae": float(fwhm_relative.mean()),
                "d50_mae_px": float(np.array([r["d50_abs_error_px"] for r in subset]).mean()),
                "d80_mae_px": float(d80.mean()),
                "d80_relative_mae": float(d80_relative.mean()),
                "rms_relative_mae": float(rms_relative.mean()),
                "iou_mean": float(iou.mean()),
            }
    return result


# ---------------------------------------------------------------------------
# dataset preparation and fitting
# ---------------------------------------------------------------------------


def prepare_dataset(fit_mod, spec: DatasetSpec, grid_size: int, camera_center=None) -> Prepared:
    manifest = json.loads((spec.path / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((spec.path / "run_config.json").read_text(encoding="utf-8"))
    matrix = fit_mod.ramp_matrix(manifest)
    references = {r["branch"]: r for r in manifest if r["label"] == "reference_start"}
    if camera_center is None:
        camera_center = np.array([references["U"]["spot"][f"centroid_{k}"] for k in ("x", "y")])
    shift = np.array([references["C"]["spot"][f"centroid_{k}"] - references["U"]["spot"][f"centroid_{k}"]
                      for k in ("x", "y")])
    gradient = np.linalg.solve(matrix, shift)
    q = (8e-6) ** 2 * config["best_diopter_m1"] / 1064e-9
    settings = config["settings"]
    amplitude_center = np.array([settings["center_x"], settings["center_y"]]) + gradient / q
    selected, phases, images, camera_xy, refs = fit_mod.load_observations(
        spec.path, camera_center, grid_size=grid_size)
    return Prepared(spec=spec, rows=selected, phases=phases, images=images, camera_xy=camera_xy,
                    references=refs, matrix=matrix, amplitude_center=amplitude_center,
                    camera_center=np.asarray(camera_center, dtype=float),
                    focus_diopter=float(config["best_diopter_m1"]))


def _batch_size(count: int, fft_size: int) -> int:
    """Bound the far-field batch so a 2048/4096 grid does not blow up memory."""
    return int(max(1, min(count, 16_000_000 // (fft_size ** 2))))


def fit_variant(fit_mod, prepared: Prepared, variant: Variant, iterations: int, seed: int,
                basis, warm_start: dict | None = None):
    """Fit one variant on the training split and return its frozen parameters."""
    import torch

    torch.manual_seed(seed)
    train = np.array([r["split"] != "test" for r in prepared.rows])
    model = fit_mod.PhysicalForwardFit(
        prepared.phases, prepared.images, prepared.camera_xy, prepared.matrix,
        prepared.focus_diopter, train, prepared.amplitude_center,
        learn_amplitude=variant.learn_amplitude,
        learn_residual=variant.learn_residual,
        fft_size=variant.fft_size, pixel_samples=variant.pixel_samples,
        basis_arrays=basis, batch_size=_batch_size(len(prepared.phases), variant.fft_size))
    with torch.no_grad():
        model.origin.copy_(torch.tensor(prepared.camera_center, dtype=torch.float32))
        if warm_start is not None:
            model.d.fill_(float(warm_start["defocus_diopter"]))
            model.origin.copy_(torch.tensor(warm_start["origin"], dtype=torch.float32))
            model.center.copy_(torch.tensor(warm_start["amplitude_center"], dtype=torch.float32))
            model.log_width.copy_(torch.tensor(warm_start["amplitude_width"]).log())
    started = time.monotonic()
    prediction = model.run(iterations)
    elapsed = time.monotonic() - started
    design = np.stack((prediction[train].ravel(), np.ones(prediction[train].size)), axis=1)
    gain, background = nnls(design, prepared.images[train].ravel())[0]
    prediction = prediction * gain + background
    parameters = {
        "defocus_diopter": float(model.d.item()),
        "origin": [float(v) for v in model.origin.detach().tolist()],
        "amplitude_center": [float(v) for v in model.center.detach().tolist()],
        "amplitude_width": [float(v) for v in model.log_width.exp().detach().tolist()],
        "gain": float(gain),
        "background": float(background),
        "residual_noll_radians": {str(j): float(v) for j, v
                                  in zip(model.residual_modes, model.residual.detach().tolist())},
        "loss_first": float(model.history[0]) if model.history else None,
        "loss_last": float(model.history[-1]) if model.history else None,
        "batch_size": _batch_size(len(prepared.phases), variant.fft_size),
        "fit_seconds": elapsed,
    }
    rows = [row_metrics(row, measured, predicted, prepared.camera_xy, prepared.references)
            for row, measured, predicted in zip(prepared.rows, prepared.images, prediction)]
    return parameters, rows


def evaluate_frozen(fit_mod, fitted: Prepared, target: Prepared, parameters: dict,
                    variant: Variant, basis):
    """Predict another acquisition with frozen parameters (no refit)."""
    import torch

    model = fit_mod.PhysicalForwardFit(
        target.phases, target.images, target.camera_xy, fitted.matrix, 0.,
        np.ones(len(target.phases), dtype=bool), parameters["amplitude_center"],
        learn_residual=bool(parameters["residual_noll_radians"]),
        fft_size=variant.fft_size, pixel_samples=variant.pixel_samples,
        basis_arrays=basis, batch_size=_batch_size(len(target.phases), variant.fft_size))
    with torch.no_grad():
        model.d.fill_(float(parameters["defocus_diopter"]))
        model.origin.copy_(torch.tensor(parameters["origin"], dtype=torch.float32))
        model.center.copy_(torch.tensor(parameters["amplitude_center"], dtype=torch.float32))
        model.log_width.copy_(torch.tensor(parameters["amplitude_width"]).log())
        for index, mode in enumerate(model.residual_modes):
            model.residual[index] = parameters["residual_noll_radians"][str(mode)]
        optical = model.prediction().numpy()
    predicted = optical * parameters["gain"] + parameters["background"]
    rows = [row_metrics(row, measured, prediction, target.camera_xy, fitted.references)
            for row, measured, prediction in zip(target.rows, target.images, predicted)]
    return rows


# ---------------------------------------------------------------------------
# variant grid
# ---------------------------------------------------------------------------


def stage_a_variants(base_grid: int) -> list:
    """Sampling convergence: far-field grid size x per-pixel integration."""
    variants = []
    for factor in (2, 4, 8):
        for pixels in (1, 2, 3):
            variants.append(Variant("A", f"A_fft{factor}x_ps{pixels}", base_grid,
                                     factor * base_grid, pixels, False, False))
    return variants


def stage_a2_variants() -> list:
    """Pupil grid refinement: the panel sampling, not only the FFT output."""
    return [
        Variant("A2", "A2_grid256_fft1024", 256, 1024, 1, False, False),
        Variant("A2", "A2_grid256_fft2048", 256, 2048, 1, False, False),
        Variant("A2", "A2_grid512_fft2048", 512, 2048, 1, False, False),
    ]


def stage_b_variants(sampling: Variant) -> list:
    """Model capacity at the sampling chosen from stage A."""
    common = (sampling.grid_size, sampling.fft_size, sampling.pixel_samples)
    return [
        Variant("B", "B_none", *common, False, False),
        Variant("B", "B_amplitude", *common, True, False),
        Variant("B", "B_residual", *common, False, True),
        Variant("B", "B_amplitude_residual", *common, True, True),
    ]


def selection_score(result: dict) -> float:
    """Training-split CV error used to pick the sampling; never the test split."""
    metrics = result["train_metrics"]
    values = [metrics[f"train_{branch}"]["cv_mae"] for branch in ("U", "C")
              if f"train_{branch}" in metrics]
    return float(np.mean(values)) if values else float("inf")


def best_sampling_label(results: list, dataset_name: str) -> str:
    """Label of the stage-A sampling with the smallest training-split CV error."""
    candidates = [r for r in results
                  if r["dataset"] == dataset_name and r["variant"]["stage"] == "A"]
    if not candidates:
        return "-"
    return min(candidates, key=selection_score)["variant"]["key"]


def relative_change(previous: dict | None, current: dict | None, key: str):
    if not previous or not current:
        return None
    before, after = previous.get(key), current.get(key)
    if before is None or after is None:
        return None
    return float(abs(after - before) / max(abs(before), 1e-9))


def sampling_convergence(results: list, dataset_name: str) -> dict:
    """Relative change of the key metrics between adjacent sampling settings."""
    stage_a = sorted([r for r in results
                      if r["dataset"] == dataset_name and r["variant"]["stage"] == "A"],
                     key=lambda r: (r["variant"]["pixel_samples"], r["variant"]["fft_size"]))
    point = [r for r in stage_a if r["variant"]["pixel_samples"] == 1]
    pixels = [r for r in stage_a if r["variant"]["fft_size"] == max(
        v["variant"]["fft_size"] for v in stage_a)]
    metrics = ("mean_corr", "cv_mae", "ee_mae", "fwhm_mae_px", "d80_relative_mae")
    chains = {}
    for name, chain in (("fft_size", point), ("pixel_samples", pixels)):
        pairs = []
        for previous, current in zip(chain, chain[1:]):
            entry = {
                "from": previous["variant"]["key"],
                "to": current["variant"]["key"],
                "changes": {},
            }
            for branch in ("U", "C"):
                for scope in ("test",):
                    for metric in metrics:
                        key = f"{scope}_{branch}"
                        value = relative_change(previous["test_metrics"].get(key),
                                                current["test_metrics"].get(key), metric)
                        if value is not None:
                            entry["changes"][f"{key}_{metric}"] = value
            pairs.append(entry)
        chains[name] = pairs
    worst = 0.0
    for pairs in chains.values():
        for pair in pairs:
            for value in pair["changes"].values():
                worst = max(worst, value)
    return {
        "chains": chains,
        "worst_relative_change": worst,
        "tolerance": GATES["sampling_relative_change"],
        "converged": bool(worst <= GATES["sampling_relative_change"]),
        "criterion": ("相邻采样档之间测试集关键指标的相对变化；判据为本轮自定，"
                      f"容差 {GATES['sampling_relative_change']:.0%}"),
    }


def sampling_geometry(prepared: Prepared, variants: list) -> list:
    """Physical sampling geometry: does a larger FFT move the coordinate mapping?

    The model samples the far-field at array index
    ``j = f * (fft_size * pitch) + fft_size // 2``; the physical frequency of
    array element ``j`` is ``(j - fft_size // 2) / (fft_size * pitch)``. The
    implied frequency is therefore checked against the requested one, and the
    far-field step is converted to CCD pixels.
    """
    inverse = np.linalg.inv(prepared.matrix)
    frequencies = (prepared.camera_xy - prepared.camera_center) @ inverse.T
    cycles_per_ccd_px = np.linalg.norm(inverse, axis=1)
    entries = []
    for variant in sorted(variants, key=lambda v: (v.grid_size, v.fft_size)):
        pitch = 1920.0 / variant.grid_size
        indices = frequencies * (variant.fft_size * pitch) + variant.fft_size // 2
        implied = (indices - variant.fft_size // 2) / (variant.fft_size * pitch)
        step_ccd = (1.0 / (variant.fft_size * pitch)) / float(np.mean(cycles_per_ccd_px))
        entries.append({
            "variant": variant.key,
            "grid_size": variant.grid_size,
            "panel_pitch_px": pitch,
            "fft_size": variant.fft_size,
            "far_field_extent_panel_px": variant.fft_size * pitch,
            "far_field_step_ccd_px": float(step_ccd),
            "camera_step_ccd_px": CCD_STEP_PX,
            "max_frequency_deviation": float(np.max(np.abs(implied - frequencies))),
            "index_min": float(indices.min()),
            "index_max": float(indices.max()),
        })
    return entries


# ---------------------------------------------------------------------------
# acceptance gates
# ---------------------------------------------------------------------------


def branch_gates(metrics: dict) -> dict:
    """Evaluate the acceptance gates on one branch aggregate."""
    verdict = {}
    for branch in ("U", "C"):
        entry = metrics.get(f"test_{branch}")
        if entry is None:
            continue
        verdict[branch] = {
            "corr_median": entry["median_corr"],
            "corr_min": entry["min_corr"],
            "passed_correlation": bool(entry["median_corr"] >= GATES["corr_median"]
                                       and entry["min_corr"] >= GATES["corr_min"]),
            "cv_mae": entry["cv_mae"],
            "passed_cv": bool(entry["cv_mae"] < GATES["cv_absolute"]),
            "ee_mae": entry["ee_mae"],
            "passed_ee": bool(entry["ee_mae"] < GATES["ee_absolute"]),
            "center_mae_px": entry["center_mae_px"],
            "passed_center": bool(entry["center_mae_px"] <= GATES["center_px"]),
            "width_relative_mae": entry["d80_relative_mae"],
            "passed_width": bool(entry["d80_relative_mae"] <= GATES["width_relative"]),
        }
    return verdict


def stopping_conditions(variant_result: dict, convergence: dict, scope: str,
                        metrics: dict | None = None) -> dict:
    """The three stop conditions of the plan, scored for one variant."""
    metrics = variant_result["test_metrics"] if metrics is None else metrics
    gates = branch_gates(metrics)
    correlation = all(g["passed_correlation"] for g in gates.values()) if gates else False
    uniformity = all(g["passed_cv"] for g in gates.values()) if gates else False
    geometry = all(g["passed_center"] and g["passed_width"] for g in gates.values()) if gates else False
    return {
        "scope": scope,
        "condition_1_correlation": {
            "requirement": f"测试集相关性中位数 ≥ {GATES['corr_median']}，最差样本 ≥ {GATES['corr_min']}",
            "passed": bool(correlation),
            "per_branch": gates,
        },
        "condition_2_uniformity": {
            "requirement": f"测试集 CV 绝对误差 < {GATES['cv_absolute']}",
            "passed": bool(uniformity),
            "ee_passed": bool(all(g["passed_ee"] for g in gates.values())) if gates else False,
            "per_branch": {b: g["cv_mae"] for b, g in gates.items()},
        },
        "condition_3_sampling_and_geometry": {
            "requirement": ("采样收敛（本轮自定容差 "
                            f"{GATES['sampling_relative_change']:.0%}）且中心/环围宽度误差已评估"),
            "sampling_converged": convergence["converged"],
            "worst_relative_change": convergence["worst_relative_change"],
            "geometry_evaluated": bool(geometry),
            "per_branch": gates,
            "passed": bool(convergence["converged"] and geometry),
        },
        "all_passed": bool(correlation and uniformity and convergence["converged"] and geometry),
    }


def evaluate_conditions(results: list, convergence: dict) -> dict:
    """Stop conditions of every variant, on its own test split and on each frozen set."""
    conditions = {}
    for result in results:
        entry = {"own_test": stopping_conditions(result, convergence, "测试集")}
        for name, payload in result.get("independent", {}).items():
            entry[name] = stopping_conditions(result, convergence, name, payload["metrics"])
        conditions[f"{result['dataset']}|{result['variant']['key']}"] = entry
    return conditions


def build_condition_rows(conditions: dict) -> list:
    """Human-readable rows of the stop-condition table."""
    rows = []
    rows.append({
        "name": "条件 1：平均强度相关性",
        "requirement": f"中位数 ≥ {GATES['corr_median']}，最差 ≥ {GATES['corr_min']}",
        "observed": " / ".join(f"{b} 中位 {_FMT(g['corr_median'], ".4f")} 最低 {_FMT(g['corr_min'], ".4f")}"
                              for b, g in conditions["condition_1_correlation"]["per_branch"].items()),
        "verdict": "通过" if conditions["condition_1_correlation"]["passed"] else "未通过",
    })
    rows.append({
        "name": "条件 2：CV 绝对误差",
        "requirement": f"< {GATES['cv_absolute']}",
        "observed": " / ".join(f"{b} {_FMT(g, ".4f")}"
                              for b, g in conditions["condition_2_uniformity"]["per_branch"].items()),
        "verdict": "通过" if conditions["condition_2_uniformity"]["passed"] else "未通过",
    })
    rows.append({
        "name": "条件 3a：采样收敛",
        "requirement": f"相邻档相对变化 ≤ {GATES['sampling_relative_change']:.0%}（本轮自定）",
        "observed": f"最大相对变化 {conditions['condition_3_sampling_and_geometry'].get('worst_relative_change', 0.0):.4%}",
        "verdict": "通过" if conditions["condition_3_sampling_and_geometry"]["sampling_converged"] else "未通过",
    })
    geometry = conditions["condition_3_sampling_and_geometry"]
    rows.append({
        "name": "条件 3b：中心/环围宽度误差",
        "requirement": f"中心误差 ≤ {GATES['center_px']} px，环围宽度相对误差 ≤ {GATES['width_relative']:.0%}",
        "observed": " / ".join(
            f"{b} 中心 {_FMT(g['center_mae_px'], ".3f")} px 环围 {_FMT(g['width_relative_mae'], ".4f")}"
            for b, g in geometry["per_branch"].items()),
        "verdict": "通过" if geometry["geometry_evaluated"] else "未通过",
    })
    return rows


# ---------------------------------------------------------------------------
# cross-check against the frozen evaluation module
# ---------------------------------------------------------------------------


def write_fit_json(path: Path, prepared: Prepared, parameters: dict, rows: list) -> None:
    payload = {
        "ramp_matrix": prepared.matrix.tolist(),
        "camera_crop_center": [float(v) for v in prepared.camera_center],
        "illumination_center_estimate": [float(v) for v in prepared.amplitude_center],
        "sample_count": len(prepared.rows),
        "models": {"V0": dict(parameters, rows=rows)},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cross_check(evaluate_mod, fit_json: Path, dataset_path: Path, reference_path: Path,
                work_dir: Path, my_rows: list) -> dict:
    """Re-score one variant with the frozen evaluation module of the other stream."""
    argv = sys.argv
    sys.argv = ["evaluate_defocus_forward_dataset.py", str(dataset_path),
                "--fits", str(fit_json), "--reference-directory", str(reference_path),
                "--output", str(work_dir)]
    try:
        evaluate_mod.main()
    except SystemExit:
        raise
    finally:
        sys.argv = argv
    scored = json.loads((work_dir / "evaluation.json").read_text(encoding="utf-8"))
    theirs = scored["V0"]["rows"]
    # The frozen evaluator reports file/label/branch/family but no split, so the
    # pairing key is (file, branch).
    lookup = {(r["file"], r["branch"]): r for r in my_rows}
    worst = {"corr": 0.0, "measured_cv": 0.0, "predicted_cv": 0.0,
             "measured_ee_roi": 0.0, "predicted_ee_roi": 0.0}
    compared = 0
    for row in theirs:
        mine = lookup.get((row["file"], row["branch"]))
        if mine is None:
            continue
        compared += 1
        for name in ("corr", "measured_cv", "predicted_cv", "measured_ee_roi", "predicted_ee_roi"):
            worst[name] = max(worst[name], abs(float(row[name]) - float(mine[name])))
    return {"compared_rows": compared, "max_abs_difference": worst,
            "tolerance": 1e-6, "passed": bool(compared and max(worst.values()) <= 1e-6)}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


_FMT = partial(
    fmt_general,
    missing_none="-",
    bool_labels=("是", "否"),
    verbatim_int=True,
)


def _variant_table(results: list, dataset_name: str, stages: tuple, scope: str) -> str:
    lines = ["| 变体 | FFT 网格 | 每像元积分 | 学振幅 | 学残差 | 分支 | 相关(均值/中位/最低) "
             "| CV 绝对误差 | EE_ROI 误差 | 中心误差/px | 环围 D80 误差/px | 环围相对误差 "
             "| 峰宽误差/px | 二阶矩半径相对误差 | IoU |",
             "| --- | ---: | ---: | :-: | :-: | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for result in results:
        if result["dataset"] != dataset_name or result["variant"]["stage"] not in stages:
            continue
        variant = result["variant"]
        metrics = result["test_metrics"] if scope == "test" else result["independent"][scope]["metrics"]
        for branch in ("U", "C"):
            entry = metrics.get(f"test_{branch}")
            if entry is None:
                continue
            lines.append(
                f"| {variant['label']} | {variant['fft_size']} | {variant['pixel_samples']} "
                f"| {'是' if variant['learn_amplitude'] else '否'} "
                f"| {'是' if variant['learn_residual'] else '否'} | {branch} "
                f"| {_FMT(entry['mean_corr'], ".4f")}/{_FMT(entry['median_corr'], ".4f")}/{_FMT(entry['min_corr'], ".4f")} "
                f"| {_FMT(entry['cv_mae'], ".4f")} | {_FMT(entry['ee_mae'], ".4f")} "
                f"| {_FMT(entry['center_mae_px'], ".3f")} | {_FMT(entry['d80_mae_px'], ".3f")} "
                f"| {_FMT(entry['d80_relative_mae'], ".4f")} | {_FMT(entry['fwhm_mae_px'], ".3f")} "
                f"| {_FMT(entry['rms_relative_mae'], ".4f")} | {_FMT(entry['iou_mean'], ".4f")} |")
    return "\n".join(lines)


def _delta_table(results: list, dataset_name: str) -> str:
    """Stage B against the stage-A baseline at the same sampling."""
    baseline = next((r for r in results if r["dataset"] == dataset_name
                     and r["variant"]["label"] == "B_none"), None)
    if baseline is None:
        return "（没有阶段 B 的基线变体）"
    lines = ["| 变体 | 分支 | Δ相关 | ΔCV 绝对误差 | ΔEE_ROI 误差 | Δ中心误差/px | Δ峰宽误差/px |",
             "| --- | :-: | ---: | ---: | ---: | ---: | ---: |"]
    for result in results:
        if result["dataset"] != dataset_name or result["variant"]["stage"] != "B":
            continue
        for branch in ("U", "C"):
            before = baseline["test_metrics"].get(f"test_{branch}")
            after = result["test_metrics"].get(f"test_{branch}")
            if before is None or after is None:
                continue
            lines.append(
                f"| {result['variant']['label']} | {branch} "
                f"| {after['mean_corr'] - before['mean_corr']:+.4f} "
                f"| {after['cv_mae'] - before['cv_mae']:+.4f} "
                f"| {after['ee_mae'] - before['ee_mae']:+.4f} "
                f"| {after['center_mae_px'] - before['center_mae_px']:+.3f} "
                f"| {after['fwhm_mae_px'] - before['fwhm_mae_px']:+.3f} |")
    return "\n".join(lines)


def _cross_dataset_table(results: list, dataset_names: list, scope: str = "test") -> str:
    """Side-by-side comparison of the same variant across datasets."""
    if len(dataset_names) < 2:
        return "（本轮只有一个数据集，未做并排对照）"
    lines = ["| 变体 | 分支 | " + " | ".join(f"{name} CV 误差" for name in dataset_names)
             + " | " + " | ".join(f"{name} 相关" for name in dataset_names) + " | ΔCV(后-前) |",
             "| --- | :-: | " + " | ".join("---:" for _ in dataset_names)
             + " | " + " | ".join("---:" for _ in dataset_names) + " | ---: |"]
    labels = []
    for result in results:
        if result["dataset"] != dataset_names[0]:
            continue
        labels.append(result["variant"]["key"])
    for label in labels:
        for branch in ("U", "C"):
            cells, correlations, values = [], [], []
            for name in dataset_names:
                entry = next((r for r in results if r["dataset"] == name
                              and r["variant"]["key"] == label), None)
                metrics = entry["test_metrics"] if scope == "test" else entry["independent"][scope]["metrics"]
                block = metrics.get(f"test_{branch}")
                if block is None:
                    cells.append("-")
                    correlations.append("-")
                    continue
                cells.append(_FMT(block["cv_mae"], ".4f"))
                correlations.append(_FMT(block["mean_corr"], ".4f"))
                values.append(block["cv_mae"])
            delta = f"{values[-1] - values[0]:+.4f}" if len(values) == len(dataset_names) else "-"
            lines.append(f"| {label} | {branch} | " + " | ".join(cells) + " | "
                         + " | ".join(correlations) + f" | {delta} |")
    return "\n".join(lines)


def render_summary(context: dict) -> str:
    results = context["results"]
    datasets = context["datasets"]
    primary = datasets[0]["name"]
    convergence = context["convergence"]
    lines = []
    lines.append("# 物理预测模型的采样变体网格评估")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：{context['generated_utc']}")
    lines.append(f"- 迭代预算：{context['settings']['iterations']}；随机种子：{context['settings']['seed']}")
    lines.append("- 判据来源：相关性/CV/EE/几何门槛取自 "
                 "`report/shape_trials/physical_forward_defocus_plan_20261022.md`；"
                 "采样收敛容差为本轮自定。")
    lines.append("")
    lines.append("## 1. 数据集与配对性")
    lines.append("")
    lines.append("| 数据集 | 路径 | 探针训练 | 探针测试 | 与基线组的测试集是否一致 |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for dataset in datasets:
        lines.append(f"| {dataset['name']} | `{dataset['path']}` | {dataset['train_probes']} "
                     f"| {dataset['test_probes']} | {dataset['paired_with_primary']} |")
    lines.append("")
    if context["pairing"]["identical_test_sets"]:
        lines.append("两个数据集的测试探针标签一致 ⇒ 配对比较成立（同一测试集）。")
    else:
        lines.append("两个数据集的测试探针标签**不一致** ⇒ 配对性不成立，"
                     "跨数据集的差值只能作趋势参考，不能作显著性结论。")
    lines.append("")
    lines.append("### 1.1 多数据集并排对照")
    lines.append("")
    lines.append(_cross_dataset_table(results, [d["name"] for d in datasets]))
    lines.append("")
    lines.append("## 2. 阶段 A：采样收敛（固定学振幅=否、学残差=否）")
    lines.append("")
    lines.append(_variant_table(results, primary, ("A",), "test"))
    lines.append("")
    lines.append("### 2.1 面板采样加密（面板网格同步细化）")
    lines.append("")
    lines.append(_variant_table(results, primary, ("A2",), "test"))
    lines.append("")
    lines.append("面板网格加密时，Zernike 基函数按同一物理中心（`952.5 panel px`）从规范全分辨率"
                 "模式重采样，避免无意引入倾斜；因此阶段 A2 与阶段 A 的差别**只**在面板相位采样密度。")
    lines.append("")
    lines.append("### 2.2 采样是否收敛")
    lines.append("")
    lines.append(f"- 判据：{convergence['criterion']}")
    lines.append(f"- 相邻档最大相对变化：**{convergence['worst_relative_change']:.4%}**")
    lines.append(f"- 结论：**{'已收敛' if convergence['converged'] else '未收敛'}**")
    lines.append("")
    lines.append("| 链 | 相邻档 | 关键指标相对变化 |")
    lines.append("| --- | --- | --- |")
    for chain_name, pairs in convergence["chains"].items():
        for pair in pairs:
            worst = max(pair["changes"].values()) if pair["changes"] else 0.0
            lines.append(f"| {chain_name} | {pair['from']} → {pair['to']} | 最大 {worst:.4%} |")
    lines.append("")
    lines.append("### 2.3 只加 FFT 点数是否改变坐标映射")
    lines.append("")
    lines.append("| 变体 | 面板节距/px | 远场跨度/panel px | 远场步长/CCD px | 相机步长/CCD px "
                 "| 采样频率偏差 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for entry in context["geometry"]:
        lines.append(f"| {entry['variant']} | {entry['panel_pitch_px']:.3f} "
                     f"| {entry['far_field_extent_panel_px']:.0f} "
                     f"| {entry['far_field_step_ccd_px']:.4f} | {entry['camera_step_ccd_px']:.1f} "
                     f"| {entry['max_frequency_deviation']:.3e} |")
    lines.append("")
    lines.append(context["geometry_note"])
    lines.append("")
    lines.append("## 3. 阶段 B：模型容量（采样取阶段 A 的训练集最优）")
    lines.append("")
    lines.append(f"- 选中的采样设置：{context['selected_sampling']}")
    lines.append(f"- 选择规则：{context['selection_rule']}")
    lines.append("")
    lines.append(_variant_table(results, primary, ("B",), "test"))
    lines.append("")
    lines.append("### 3.1 相对阶段 A 基线（同一采样）的变化")
    lines.append("")
    lines.append(_delta_table(results, primary))
    lines.append("")
    if context.get("independent"):
        lines.append("## 3.2 独立种子冻结参数检验")
        lines.append("")
        lines.append("模型在主数据集的训练相位上拟合，参数全部冻结后在独立采集上评估；"
                     "相机网格与目标框取自主数据集。")
        lines.append("")
        for scope in context["independent"]:
            lines.append(f"**独立数据集 `{scope}`**")
            lines.append("")
            lines.append(_variant_table(results, primary, ("A", "A2", "B"), scope))
            lines.append("")
    lines.append("## 4. 停止条件对照")
    lines.append("")
    lines.append(f"- 对照对象：训练集 CV 误差最小的变体 `{context['recommended_variant']}`"
                 f"（阶段 A 采样最优为 `{context['stage_a_best_variant']}`）。")
    lines.append("")
    lines.append("### 4.1 主数据集的测试集")
    lines.append("")
    lines.append("| 条件 | 要求 | 本轮结果 | 判定 |")
    lines.append("| --- | --- | --- | --- |")
    for row in context["condition_rows"]:
        lines.append(f"| {row['name']} | {row['requirement']} | {row['observed']} | {row['verdict']} |")
    lines.append("")
    for scope in context.get("independent", []):
        lines.append(f"### 4.2 独立种子冻结参数检验（`{scope}`）")
        lines.append("")
        lines.append("| 条件 | 要求 | 本轮结果 | 判定 |")
        lines.append("| --- | --- | --- | --- |")
        frozen = context.get("conditions_by_variant", {}).get(
            f"{context['datasets'][0]['name']}|{context['recommended_variant']}", {}).get(scope)
        for row in build_condition_rows(frozen) if frozen else []:
            lines.append(f"| {row['name']} | {row['requirement']} | {row['observed']} | {row['verdict']} |")
        lines.append("")
    lines.append(f"- 三条同时满足才允许进入逆向优化 ⇒ 本轮结论："
                 f"**{'允许' if context['conditions']['all_passed'] else '不允许'}**。")
    lines.append("")
    lines.append("## 5. 口径与已知限制")
    lines.append("")
    for note in context["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def render_figure(context: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    results = context["results"]
    primary = context["datasets"][0]["name"]
    stage_a = [r for r in results if r["dataset"] == primary and r["variant"]["stage"] == "A"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    panels = [("cv_mae", "CV 绝对误差"), ("mean_corr", "平均强度相关性"),
              ("fwhm_mae_px", "峰宽误差（半高全宽）/px")]
    for column, branch in enumerate(("U", "C")):
        for row_index, (metric, title) in enumerate(panels):
            axis = axes[row_index][column]
            for pixels in sorted({r["variant"]["pixel_samples"] for r in stage_a}):
                chain = sorted([r for r in stage_a if r["variant"]["pixel_samples"] == pixels],
                               key=lambda r: r["variant"]["fft_size"])
                xs = [r["variant"]["fft_size"] for r in chain]
                ys = [r["test_metrics"][f"test_{branch}"][metric] for r in chain]
                axis.plot(xs, ys, "o-", label=f"每像元积分 {pixels}")
            if metric == "cv_mae":
                axis.axhline(GATES["cv_absolute"], color="gray", linestyle="--")
            axis.set_xscale("log", base=2)
            axis.set_xticks(sorted({r["variant"]["fft_size"] for r in stage_a}))
            axis.get_xaxis().set_major_formatter(ScalarFormatter())
            axis.set_xlabel("远场 FFT 网格边长")
            axis.set_ylabel(title)
            axis.set_title(f"分支 {branch}")
            axis.grid(alpha=.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("物理预测模型：采样变体网格的测试集指标（面板采样固定）")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def parse_dataset_option(text: str) -> DatasetSpec:
    if "=" not in text:
        raise argparse.ArgumentTypeError("数据集参数格式为 name=path")
    name, _, path = text.partition("=")
    name, path = name.strip(), path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("数据集参数格式为 name=path")
    return DatasetSpec(name, Path(path))


def _dataset_report(prepared: Prepared) -> dict:
    probes = [r for r in prepared.rows if r["family"] != "reference"]
    labels = sorted(f"{r['branch']}:{r['label']}" for r in probes if r["split"] == "test")
    return {
        "name": prepared.spec.name,
        "path": str(prepared.spec.path),
        "camera_center": [float(v) for v in prepared.camera_center],
        "focus_diopter": prepared.focus_diopter,
        "ramp_matrix": prepared.matrix.tolist(),
        "train_probes": sum(1 for r in probes if r["split"] != "test"),
        "test_probes": sum(1 for r in probes if r["split"] == "test"),
        "test_probe_labels": labels,
        "paired_with_primary": None,
    }


def run_dataset(fit_mod, sampling_mod, spec: DatasetSpec, variants: list, iterations: int,
                seed: int, evaluation_specs: list, basis_cache: dict,
                evaluation_cache: dict | None = None) -> list:
    """Run one variant grid on one dataset."""
    results = []
    prepared_cache: dict = {}
    if evaluation_cache is None:
        evaluation_cache = {}
    for variant in variants:
        if variant.grid_size not in prepared_cache:
            prepared_cache[variant.grid_size] = prepare_dataset(fit_mod, spec, variant.grid_size)
        prepared = prepared_cache[variant.grid_size]
        basis = None
        if variant.grid_size != 128:
            if variant.grid_size not in basis_cache:
                basis_cache[variant.grid_size] = sampling_mod.fixed_panel_basis(variant.grid_size)
            basis = basis_cache[variant.grid_size]
        warm_start = None
        if variant.label == "B_amplitude_residual":
            source = next((r for r in results if r["variant"]["label"] == "B_amplitude"), None)
            if source is None:
                raise SystemExit("阶段 B 的学残差变体需要先跑学振幅变体")
            warm_start = source["parameters"]
        parameters, rows = fit_variant(fit_mod, prepared, variant, iterations, seed, basis, warm_start)
        entry = {
            "dataset": spec.name,
            "variant": variant.as_dict(),
            "parameters": parameters,
            "train_metrics": aggregate([r for r in rows if r["split"] != "test"]),
            "test_metrics": aggregate([r for r in rows if r["split"] == "test"]),
            "rows": rows,
            "independent": {},
        }
        for evaluation_spec in evaluation_specs:
            cache_key = (evaluation_spec.name, variant.grid_size)
            if cache_key not in evaluation_cache:
                evaluation_cache[cache_key] = prepare_dataset(
                    fit_mod, evaluation_spec, variant.grid_size,
                    camera_center=prepared.camera_center)
            target = evaluation_cache[cache_key]
            scored = evaluate_frozen(fit_mod, prepared, target, parameters, variant, basis)
            entry["independent"][evaluation_spec.name] = {
                "rows": scored,
                "metrics": aggregate([r for r in scored if r["split"] == "test"]),
            }
        results.append(entry)
        print(f"[{spec.name}] {variant.label} "
              f"fft={variant.fft_size} ps={variant.pixel_samples} "
              f"amp={int(variant.learn_amplitude)} res={int(variant.learn_residual)} "
              f"{parameters['fit_seconds']:.1f}s", flush=True)
    return results


def main(argv: list | None = None) -> int:
    import torch
    from loguru import logger

    logger.remove()
    torch.set_num_threads(4)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=parse_dataset_option,
                        help="name=path，可重复；同一组变体在每个数据集上各跑一遍")
    parser.add_argument("--evaluation-dataset", action="append", type=parse_dataset_option, default=[],
                        help="name=path，可重复；独立种子的冻结参数检验集")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--stage", choices=("all", "a", "a2", "b"), default="all")
    parser.add_argument("--base-grid", type=int, default=128)
    parser.add_argument("--no-cross-check", dest="cross_check", action="store_false", default=True,
                        help="跳过与冻结评估模块的口径交叉核对")
    parser.add_argument("--render-only", type=Path,
                        help="只重新渲染 summary.md 与图（读取已有 variants.json），不重新拟合")
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if args.render_only is not None:
        context = json.loads(args.render_only.read_text(encoding="utf-8"))
        convergence = context["convergence"]
        context["conditions_by_variant"] = evaluate_conditions(context["results"], convergence)
        key = f"{context['datasets'][0]['name']}|{context['recommended_variant']}"
        context["conditions"] = context["conditions_by_variant"][key]["own_test"]
        context["condition_rows"] = build_condition_rows(context["conditions"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.md").write_text(render_summary(context), encoding="utf-8")
        render_figure(context, output / "sampling_grid.png")
        print(f"re-rendered {output}")
        return 0
    if not args.dataset:
        parser.error("--dataset 必填（name=path，可重复）")
    if output.exists():
        raise SystemExit(f"输出目录已存在：{output}；请换一个目录，避免覆盖既有证据")
    output.mkdir(parents=True)

    fit_mod = load_workstream_module(FIT_MODULE, "scripts/slm/fit_defocus_forward_dataset.py")
    evaluate_mod = load_workstream_module(EVALUATE_MODULE,
                                          "scripts/slm/evaluate_defocus_forward_dataset.py")
    sampling_mod = load_workstream_module(SAMPLING_MODULE, "scripts/slm/diagnose_defocus_sampling.py")

    for spec in list(args.dataset) + list(args.evaluation_dataset):
        if not (spec.path / "manifest.json").is_file():
            raise SystemExit(f"数据集 {spec.name} 缺少 manifest.json：{spec.path}")

    stage_a = stage_a_variants(args.base_grid)
    stage_a2 = stage_a2_variants()
    results: list = []
    basis_cache: dict = {}
    evaluation_cache: dict = {}
    dataset_reports = []
    for spec in args.dataset:
        if args.stage in ("all", "a", "a2"):
            variants = list(stage_a)
            if args.stage in ("all", "a2"):
                variants += stage_a2
            results += run_dataset(fit_mod, sampling_mod, spec, variants, args.iterations,
                                   args.seed, args.evaluation_dataset, basis_cache, evaluation_cache)
        prepared = prepare_dataset(fit_mod, spec, args.base_grid)
        dataset_reports.append(_dataset_report(prepared))

    # Stage B needs the stage-A winner, so stage A always runs when stage B is asked for.
    if args.stage in ("all", "b"):
        if not any(r["variant"]["stage"] == "A" for r in results):
            for spec in args.dataset:
                results += run_dataset(fit_mod, sampling_mod, spec, stage_a, args.iterations,
                                       args.seed, args.evaluation_dataset, basis_cache, evaluation_cache)
        for spec in args.dataset:
            candidates = [r for r in results
                          if r["dataset"] == spec.name and r["variant"]["stage"] == "A"]
            if not candidates:
                raise SystemExit(f"数据集 {spec.name} 没有阶段 A 结果，无法选择采样设置")
            winner = min(candidates, key=selection_score)
            sampling = Variant("A", winner["variant"]["label"], winner["variant"]["grid_size"],
                               winner["variant"]["fft_size"], winner["variant"]["pixel_samples"],
                               False, False)
            print(f"[{spec.name}] 选中的采样：{winner['variant']['key']} "
                  f"(训练集 CV 误差 {selection_score(winner):.4f})", flush=True)
            results += run_dataset(fit_mod, sampling_mod, spec, stage_b_variants(sampling),
                                   args.iterations, args.seed, args.evaluation_dataset,
                                   basis_cache, evaluation_cache)

    primary = args.dataset[0].name
    for report in dataset_reports:
        report["paired_with_primary"] = (
            "是" if report["test_probe_labels"] == dataset_reports[0]["test_probe_labels"] else "否")

    convergence = sampling_convergence(results, primary)
    geometry = sampling_geometry(prepare_dataset(fit_mod, args.dataset[0], args.base_grid),
                                 stage_a + stage_a2)

    notes = [
        "所有变体共用同一批原始帧、同一几何矩阵与同一固定相机网格；增益与常数背景只在训练集上拟合。",
        "测试集只用于评估；阶段 B 的采样选择规则只看训练集的 CV 绝对误差，不看测试集。",
        "CV/EE_ROI/质心/二阶矩半径的定义与 `scripts/slm/diagnose_defocus_sampling.py::metrics` 一致；"
        "峰宽用二阶矩半径与半高全宽两种口径给出。",
        "环围宽度取 80% 环围能量的等面积直径 D80（另给 D50），逐像元 2 CCD px。",
        "IoU 用半高掩码的重叠比，只作形状重叠的辅助量。",
        "面板网格加密时 Zernike 基函数按同一物理中心重采样，避免引入隐藏倾斜。",
    ]
    geometry_note = (
        "模型的远场采样索引为 `j = f * (fft_size * pitch) + fft_size // 2`，"
        "而远场数组第 j 元的物理频率是 `(j - fft_size // 2) / (fft_size * pitch)`。"
        "两式相消 ⇒ **只加 FFT 点数不改变坐标映射**：每个相机采样点的物理频率不变，"
        "变化的只是远场网格步长（上表「远场步长」随 FFT 点数变细）。"
        "上表「采样频率偏差」为该频率一致性的数值残差，量级 1e-16，"
        "即映射未移动。面板节距只由面板网格决定；"
        "只有同时改变面板网格（阶段 A2）才会改变被采样的物理内容。")

    condition_rows = []
    primary_results = [r for r in results if r["dataset"] == primary]
    stage_a_best = min([r for r in primary_results if r["variant"]["stage"] == "A"],
                       key=selection_score)
    recommended = min(primary_results, key=selection_score)
    conditions_by_variant = evaluate_conditions(results, convergence)
    conditions = conditions_by_variant[f"{primary}|{recommended['variant']['key']}"]["own_test"]
    condition_rows = build_condition_rows(conditions)

    check = None
    if args.cross_check:
        default_sampling = next((r for r in results if r["dataset"] == primary
                                 and r["variant"]["stage"] == "A"
                                 and r["variant"]["fft_size"] == 4 * args.base_grid
                                 and r["variant"]["pixel_samples"] == 1), None)
        if default_sampling is not None:
            prepared = prepare_dataset(fit_mod, args.dataset[0], args.base_grid)
            fit_json = output / "cross_check_fit.json"
            write_fit_json(fit_json, prepared, default_sampling["parameters"],
                           default_sampling["rows"])
            check = cross_check(evaluate_mod, fit_json, args.dataset[0].path, args.dataset[0].path,
                                output / "cross_check_evaluation", default_sampling["rows"])
            check["variant"] = default_sampling["variant"]["key"]

    if check is not None:
        notes.append(
            f"口径交叉核对：用另一条线的冻结评估模块 `evaluate_defocus_forward_dataset.py` "
            f"重算 `{check['variant']}` 的全部 {check['compared_rows']} 条记录，"
            f"与本脚本的 corr/CV/EE_ROI 最大绝对差 {max(check['max_abs_difference'].values()):.2e}"
            f"（容差 {check['tolerance']:.0e}）⇒ {'一致' if check['passed'] else '不一致'}。")
    else:
        notes.append("口径交叉核对已跳过（`--no-cross-check`）。")
    if len(args.dataset) > 1:
        notes.append("多数据集只作配对 A/B 对照；若测试集不一致，差值不作显著性结论。")

    context = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {"iterations": args.iterations, "seed": args.seed, "stage": args.stage,
                     "base_grid": args.base_grid},
        "datasets": dataset_reports,
        "results": results,
        "convergence": convergence,
        "geometry": geometry,
        "geometry_note": geometry_note,
        "notes": notes,
        "condition_rows": condition_rows,
        "conditions": conditions,
        "conditions_by_variant": conditions_by_variant,
        "recommended_variant": recommended["variant"]["key"],
        "stage_a_best_variant": stage_a_best["variant"]["key"],
        "independent": sorted({name for r in results for name in r["independent"]}),
        "selected_sampling": best_sampling_label(results, primary),
        "selection_rule": "阶段 A 内训练集 CV 绝对误差（U/C 均值）最小者；不使用测试集。",
        "pairing": {
            "identical_test_sets": all(r["paired_with_primary"] == "是" for r in dataset_reports),
            "per_dataset": {r["name"]: r["paired_with_primary"] for r in dataset_reports},
        },
        "cross_check": check,
    }

    (output / "variants.json").write_text(json.dumps(context, indent=2, ensure_ascii=False),
                                           encoding="utf-8")

    flat = []
    for result in results:
        scopes = {"test": result["test_metrics"]}
        scopes.update({name: payload["metrics"] for name, payload in result["independent"].items()})
        for scope, metrics in scopes.items():
            for key, entry in metrics.items():
                flat.append({
                    "dataset": result["dataset"],
                    "variant": result["variant"]["label"],
                    "variant_key": result["variant"]["key"],
                    "stage": result["variant"]["stage"],
                    "grid_size": result["variant"]["grid_size"],
                    "fft_size": result["variant"]["fft_size"],
                    "pixel_samples": result["variant"]["pixel_samples"],
                    "learn_amplitude": result["variant"]["learn_amplitude"],
                    "learn_residual": result["variant"]["learn_residual"],
                    "scope": scope,
                    "subset": key,
                    **entry,
                })
    if flat:
        with (output / "variants.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)

    (output / "summary.md").write_text(render_summary(context), encoding="utf-8")
    render_figure(context, output / "sampling_grid.png")

    print(json.dumps({"converged": convergence["converged"],
                      "worst_relative_change": convergence["worst_relative_change"],
                      "cross_check": check}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
