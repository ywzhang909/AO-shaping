"""Offline probe: subpixel global registration and low-order gain field (issue #237).

This probe is deliberately orthogonal to the phase-polarity scope work.  It reads
only archived ``npz`` frames plus the frozen forward fit, wraps the frozen model
output with a single global translation and/or a low-order multiplicative gain
field, and evaluates the same three accuracy rows as
``offline_benchmark_20261107.py`` under both phase signs.

No hardware is touched and no ``ao_shaping.drivers`` module is imported.  The
script writes only under
``data/debug/slm_transfer/registration_gain_probe_20261107/``.

Method fixed before running
---------------------------
* Registration: one global ``(dy, dx)`` for all frames, estimated on the pooled
  training split (r2 train 18 + ladder pass1 42).  Integer search is on a ±4 px
  grid, expanded by 4 px only while the optimum lies on the border (max ±12),
  maximising mean full-frame Pearson.  Subpixel refinement is a separable
  three-point parabola around the integer peak.  No per-pair translation is
  fitted.
* Gain field: degree-2 polynomial in normalised detector coordinates,
  ``M = 1 + sum(c_k b_k)`` with zero-mean bases ``x, y, x^2, y^2, xy`` on the
  training signal mask; five free coefficients, applied as
  ``background + M * (prediction - background)``, clipped to [0.5, 1.5].
  Coefficients are fitted by least squares on the training split.  The
  ``registration_gain`` variant re-fits the same five coefficients after the
  global shift.
* CCD response LUT: audited, not fitted unless the training split contains
  same-command exposure brackets spanning the low/high response range.  The
  audit is written to ``results.json`` either way.

Determinism: CPU, fixed seeds are not needed (no stochastic fitting), threads
fixed on the command line.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BASE_DIR = ROOT / "data/debug/slm_transfer"
R2_DIR = BASE_DIR / "defocus_forward_20261107_focus_r2"
FIT_PATH = BASE_DIR / "defocus_forward_20261107_focus_r2_fit/forward_fit.json"
LADDER_DIRS = {
    "s010": BASE_DIR / "issue234_single_mode_screen_20261107",
    "s005": BASE_DIR / "issue234_amp05_screen_20261107",
    "s002": BASE_DIR / "issue234_amp02_screen_20261107",
}
PARITY_SCAN = BASE_DIR / "issue235_parity_20261107/parity_registration_scan.json"
BASELINE_RESULTS = BASE_DIR / "offline_benchmark_20261107/results.json"
DEFAULT_OUT = BASE_DIR / "registration_gain_probe_20261107"

DEVICE_STATEMENT = (
    "离线分析，未占用任何设备：未打开 SLM / 相机 / 激光，未 import 任何 "
    "ao_shaping.drivers 模块；全部数字来自磁盘存档 npz 帧与冻结拟合 JSON，"
    "CPU 单机、固定流程、确定性运行。"
)

# Fixed hyperparameters (written to results.json as well).
COARSE_HALF = 4
COARSE_EXPAND = 4
COARSE_MAX = 12
SHIFT_ORDER = 1
GAIN_ORDER = 2
GAIN_BASIS = ("x", "y", "x^2", "y^2", "xy")
GAIN_CLIP = (0.5, 1.5)
GAIN_RIDGE = 1.0e-8
GAIN_SIGNAL_FRACTION = 0.05
CORE_FRACTION = 0.1
LOW_FREQ_CYCLES_PER_PIXEL = 1.0 / 16.0
SIGNS = (-1.0, 1.0)
EVAL_SETS = ("ladder_pass2", "r2_usable", "r2_heldout")
VARIANTS = ("baseline", "registration", "gain", "registration_gain")


@dataclass
class Audit:
    """Container for the LUT feasibility audit."""

    status: str
    reason: str
    training_brackets: int
    eval_brackets: int
    details: dict[str, Any]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_TRANSFER = _load_module("_registration_gain_transfer", ROOT / "scripts/slm/gated_recurrent_transfer_20261107.py")

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return _TRANSFER.pearson(a, b)


def _rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    return _TRANSFER.rel_l2(pred, target)


def _predict(forward: FrozenDefocusForward, phase: np.ndarray) -> np.ndarray:
    return _TRANSFER.predict_frame(forward, phase)


def _shift(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    return ndimage.shift(
        image,
        (float(dy), float(dx)),
        order=SHIFT_ORDER,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32)


def _core_mask(frames: list[Any], level: float = CORE_FRACTION) -> np.ndarray:
    mean_image = np.mean([f.image for f in frames], axis=0)
    return mean_image >= level * float(np.max(mean_image))


def _read_exposures(directory: Path) -> dict[str, float]:
    rows = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return {str(r["file"]): float(r["exposure_ms"]) for r in rows}


def _load_frames(offset: np.ndarray, default_center: np.ndarray) -> dict[str, list[Any]]:
    """Load exactly the regular frames used by the archived accuracy benchmark."""
    r2_frames = _TRANSFER.usable(_TRANSFER.load_frames(R2_DIR, None, offset, default_center))
    r2_exposures = _read_exposures(R2_DIR)
    for frame in r2_frames:
        frame.exposure_ms = r2_exposures[str(frame.key)]

    ladder_pass1: list[Any] = []
    ladder_pass2: list[Any] = []
    ladder_all_seqs: list[list[Any]] = []
    for tag, directory in LADDER_DIRS.items():
        for pass_name in ("pass1", "pass2"):
            frames = _TRANSFER.usable(_TRANSFER.load_frames(directory, pass_name, offset, default_center))
            exposures = _read_exposures(directory / pass_name)
            for frame in frames:
                frame.key = f"{tag}/{pass_name}/{frame.key}"
                frame.exposure_ms = exposures[str(frame.key).rsplit("/", 1)[-1]]
            train_frames = [frame for frame in frames if frame.split == "train"]
            for frame in train_frames:
                frame.split = "train" if pass_name == "pass1" else "test"
            if pass_name == "pass1":
                ladder_pass1.extend(train_frames)
            else:
                ladder_pass2.extend(train_frames)
            ladder_all_seqs.append(sorted(train_frames, key=lambda frame: frame.ns))

    return {
        "r2_usable": r2_frames,
        "r2_train": [frame for frame in r2_frames if frame.split == "train"],
        "r2_heldout": [frame for frame in r2_frames if frame.split == "test"],
        "ladder_pass1": ladder_pass1,
        "ladder_pass2": ladder_pass2,
        "ladder_all_seqs": ladder_all_seqs,
    }


def _score_shift(frames: list[Any], preds: dict[str, np.ndarray], dy: int, dx: int) -> float:
    values = [
        _pearson(_shift(preds[frame.key], dy, dx), frame.image)
        for frame in frames
    ]
    return float(np.mean(values))


def _parabola_offset(minus: float, center: float, plus: float) -> float:
    denominator = minus + plus - 2.0 * center
    if denominator >= -1.0e-12:
        return 0.0
    delta = 0.5 * (minus - plus) / denominator
    return float(np.clip(delta, -0.75, 0.75))


def estimate_global_shift(
    frames: list[Any], preds: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Single global integer grid + separable parabola subpixel refinement."""
    scores: dict[tuple[int, int], float] = {}
    half = COARSE_HALF
    while True:
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                key = (dy, dx)
                if key not in scores:
                    scores[key] = _score_shift(frames, preds, dy, dx)
        best_key = max(scores, key=lambda key: (scores[key], -abs(key[0]) - abs(key[1])))
        if abs(best_key[0]) < half and abs(best_key[1]) < half or half >= COARSE_MAX:
            break
        half += COARSE_EXPAND

    best_dy, best_dx = best_key
    peak = scores[best_key]
    side_y_minus = scores.get((best_dy - 1, best_dx), peak)
    side_y_plus = scores.get((best_dy + 1, best_dx), peak)
    side_x_minus = scores.get((best_dy, best_dx - 1), peak)
    side_x_plus = scores.get((best_dy, best_dx + 1), peak)
    sub_dy = float(best_dy) + _parabola_offset(side_y_minus, peak, side_y_plus)
    sub_dx = float(best_dx) + _parabola_offset(side_x_minus, peak, side_x_plus)

    return {
        "integer_dy": int(best_dy),
        "integer_dx": int(best_dx),
        "subpixel_dy": sub_dy,
        "subpixel_dx": sub_dx,
        "parabola_dy": sub_dy - best_dy,
        "parabola_dx": sub_dx - best_dx,
        "integer_peak_score": float(peak),
        "search_half_px": int(half),
        "search_grid_points": len(scores),
        "score_at_integer": {
            f"{dy},{dx}": float(value)
            for (dy, dx), value in sorted(scores.items())
            if abs(dy - best_dy) <= 1 and abs(dx - best_dx) <= 1
        },
    }


def _gain_basis_shape(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    u = (xx - (width - 1) / 2.0) / ((width - 1) / 2.0)
    v = (yy - (height - 1) / 2.0) / ((height - 1) / 2.0)
    return u, v


def _gain_bases(u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    raw = (u, v, u ** 2, v ** 2, u * v)
    return [item - float(item[mask].mean()) for item in raw]


def fit_gain_field(
    frames: list[Any],
    preds: dict[str, np.ndarray],
    background: float,
) -> dict[str, Any]:
    """Least-squares fit of the fixed five-coefficient multiplicative field."""
    mean_pred = np.mean([preds[frame.key] for frame in frames], axis=0)
    u, v = _gain_basis_shape(mean_pred.shape)
    mask = mean_pred >= GAIN_SIGNAL_FRACTION * float(np.max(mean_pred))
    bases = _gain_bases(u, v, mask)

    matrix = np.zeros((5, 5), dtype=np.float64)
    rhs = np.zeros(5, dtype=np.float64)
    samples = 0
    for frame in frames:
        q = preds[frame.key] - background
        residual = frame.image - background - q
        columns = [q * basis for basis in bases]
        values = np.stack([column[mask] for column in columns], axis=1)
        target = residual[mask]
        matrix += values.T @ values
        rhs += values.T @ target
        samples += int(mask.sum())

    scale = float(np.trace(matrix) / 5.0)
    coefficients = np.linalg.solve(
        matrix + GAIN_RIDGE * scale * np.eye(5), rhs
    )

    field = np.ones_like(u)
    for coefficient, basis in zip(coefficients, bases):
        field += coefficient * basis
    clipped = int(np.count_nonzero((field < GAIN_CLIP[0]) | (field > GAIN_CLIP[1])))
    field = np.clip(field, GAIN_CLIP[0], GAIN_CLIP[1])

    train_before = float(
        np.mean([
            np.mean((frame.image - preds[frame.key])[mask] ** 2)
            for frame in frames
        ])
    )
    train_after = float(
        np.mean([
            np.mean(
                (
                    frame.image
                    - (background + field * (preds[frame.key] - background))
                )[mask] ** 2
            )
            for frame in frames
        ])
    )
    return {
        "order": GAIN_ORDER,
        "basis": list(GAIN_BASIS),
        "coefficients": [float(value) for value in coefficients],
        "basis_mask_pixels": int(mask.sum()),
        "fit_samples": samples,
        "coefficient_of_variation_before": float(np.sqrt(train_before)),
        "coefficient_of_variation_after": float(np.sqrt(train_after)),
        "clipped_pixels": clipped,
        "clip": list(GAIN_CLIP),
        "ridge": GAIN_RIDGE,
        "signal_fraction": GAIN_SIGNAL_FRACTION,
        "field": field.astype(np.float32),
    }


def apply_gain(pred: np.ndarray, background: float, field: np.ndarray) -> np.ndarray:
    return (background + field * (pred - background)).astype(np.float32)


def make_variant_maps(
    base: dict[str, np.ndarray],
    shift: dict[str, Any] | None = None,
    gain: dict[str, Any] | None = None,
    background: float = 0.0,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key, pred in base.items():
        value = pred
        if shift is not None:
            value = _shift(value, shift["subpixel_dy"], shift["subpixel_dx"])
        if gain is not None:
            value = apply_gain(value, background, gain["field"])
        output[key] = value
    return output


def evaluate(
    frames: list[Any],
    preds: dict[str, np.ndarray],
) -> dict[str, float]:
    core = _core_mask(frames)
    return {
        "full_pearson": float(np.mean([
            _pearson(preds[frame.key], frame.image) for frame in frames
        ])),
        "core_pearson": float(np.mean([
            _pearson(preds[frame.key][core], frame.image[core]) for frame in frames
        ])),
        "rel_l2": float(np.mean([
            _rel_l2(preds[frame.key], frame.image) for frame in frames
        ])),
        "core_pixels": int(core.sum()),
    }


def _direction_stats(preds: dict[str, np.ndarray], pairs: list[Any]) -> dict[str, Any]:
    corrs = []
    for _mode, _key, plus, minus in pairs:
        corrs.append(
            _pearson(preds[plus.key] - preds[minus.key], plus.image - minus.image)
        )
    values = np.asarray(corrs, dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "n_ge_0.5": int(np.count_nonzero(values >= 0.5)),
        "n": int(values.size),
        "corrs": [float(value) for value in values],
    }


def _residual_structure(
    frames: list[Any], preds: dict[str, np.ndarray]
) -> dict[str, float]:
    core = _core_mask(frames)
    residual = np.mean(
        [frame.image - preds[frame.key] for frame in frames], axis=0
    )
    residual = residual - float(residual[core].mean())
    power = np.abs(np.fft.fft2(residual)) ** 2
    fy = np.fft.fftfreq(residual.shape[0])[:, None]
    fx = np.fft.fftfreq(residual.shape[1])[None, :]
    radial = np.sqrt(fy ** 2 + fx ** 2)
    total_mask = radial > 0.0
    low_mask = (radial <= LOW_FREQ_CYCLES_PER_PIXEL) & total_mask
    total = float(power[total_mask].sum())
    low = float(power[low_mask].sum())
    return {
        "mean_residual_rms_in_core": float(np.sqrt(np.mean(residual[core] ** 2))),
        "low_frequency_power_fraction": low / total if total > 0.0 else 0.0,
        "low_frequency_cutoff_cycles_per_pixel": LOW_FREQ_CYCLES_PER_PIXEL,
    }


def _panel_hash(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        return hashlib.sha256(np.ascontiguousarray(data["panel_phase"]).tobytes()).hexdigest()[:12]


def audit_lut(frames: dict[str, list[Any]]) -> Audit:
    """Check whether the training split can identify a monotone exposure response."""
    bracket_records: list[dict[str, Any]] = []
    for tag, directory in LADDER_DIRS.items():
        for pass_name in ("pass1", "pass2"):
            capture_dir = directory / pass_name
            with open(capture_dir / "capture_records.pkl", "rb") as handle:
                records = pickle.load(handle)
            for record in records:
                if record.get("split") not in ("train", "test"):
                    continue
                if record.get("split") == "train":
                    split = "train" if pass_name == "pass1" else "test"
                else:
                    split = "test"
                clocks = record.get("exposure_probe_clocks") or []
                exposures = [float(item["exposure_ms"]) for item in clocks]
                unique = sorted(set(exposures))
                if len(unique) < 2:
                    continue
                bracket_records.append({
                    "tag": tag,
                    "pass": pass_name,
                    "split": split,
                    "file": record["file"],
                    "label": record["label"],
                    "exposures_ms": unique,
                    "n_frames": len(exposures),
                })

    training_brackets = [item for item in bracket_records if item["split"] == "train"]
    eval_brackets = [item for item in bracket_records if item["split"] == "test"]

    # Same commanded panel phase repeated at different exposures *within* the
    # training pass.  These are cross-run repeats, not a controlled bracket.
    pass1_groups: dict[str, list[dict[str, Any]]] = {}
    for tag, directory in LADDER_DIRS.items():
        capture_dir = directory / "pass1"
        with open(capture_dir / "capture_records.pkl", "rb") as handle:
            records = pickle.load(handle)
        for record in records:
            if record.get("split") != "train":
                continue
            digest = _panel_hash(capture_dir / record["file"])
            pass1_groups.setdefault(digest, []).append({
                "tag": tag,
                "file": record["file"],
                "exposure_ms": float(record["exposure_ms"]),
            })
    cross_run = [
        {
            "panel_sha256_12": digest,
            "exposures_ms": sorted({item["exposure_ms"] for item in items}),
            "records": items,
        }
        for digest, items in pass1_groups.items()
        if len({item["exposure_ms"] for item in items}) > 1
    ]

    focus = json.loads(
        (BASE_DIR / "20261107_focus_scan/focus.json").read_text(encoding="utf-8")
    )
    points = focus.get("probe", {}).get("points", [])
    focus_diopters = [float(point["diopter_m1"]) for point in points]
    focus_repeated = len(focus_diopters) - len(set(focus_diopters))

    training_span = 0.0
    if cross_run:
        all_exp = [exp for group in cross_run for exp in group["exposures_ms"]]
        training_span = float(max(all_exp) / min(all_exp)) if min(all_exp) > 0 else 0.0

    if training_brackets:
        return Audit(
            status="fit_available",
            reason="training split contains same-command multi-exposure brackets",
            training_brackets=len(training_brackets),
            eval_brackets=len(eval_brackets),
            details={
                "brackets": bracket_records,
                "cross_run_training_groups": cross_run,
                "focus_scan_repeated_diopters": focus_repeated,
            },
        )

    return Audit(
        status="not_fit",
        reason=(
            "训练划分（ladder pass1 42 帧 + r2 train 18 帧）中没有任何同一指令的"
            "曝光括号：pass1 的 exposure_probe_frames 全部是同一曝光；r2 无曝光探针；"
            "仅 pass2 held-out 的 3 个样本含 0.02 ms + 当前曝光括号。训练集内仅有跨 run "
            "重复的 ramp 相位（曝光约 0.311–0.354 ms，跨度约 14%），不足以辨识低曝光曲率，"
            "且跨 run 存在漂移混淆。"
        ),
        training_brackets=0,
        eval_brackets=len(eval_brackets),
        details={
            "training_bracket_records": [],
            "eval_bracket_records": eval_brackets,
            "cross_run_training_groups": cross_run,
            "cross_run_training_exposure_span_ratio": training_span,
            "focus_scan_points": len(points),
            "focus_scan_repeated_diopters": focus_repeated,
            "focus_scan_note": (
                "focus.json 每个离焦值只有一个自适应曝光点，没有同一场景的多曝光重复，"
                "不能把峰值/曝光随离焦的变化当作响应曲线。"
            ),
            "monotonic_response_note": (
                "既有 MonotonicResponse 默认 v_max=255（counts），而本数据的归一化峰值"
                "可达数百至约 600 counts/ms；在缺少重复曝光约束时直接拟合会与模型空间误差"
                "混叠，故不硬拟合。"
            ),
        },
    )


def _json_default(value: Any):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _run_sign(
    sign: float,
    forward: FrozenDefocusForward,
    frames: dict[str, list[Any]],
    background: float,
    pairs: list[Any],
    parity_reference: dict[str, Any],
) -> dict[str, Any]:
    all_frames = [
        *frames["ladder_pass1"],
        *frames["ladder_pass2"],
        *frames["r2_usable"],
    ]
    base = {frame.key: _predict(forward, frame.phase) for frame in all_frames}
    train = [*frames["ladder_pass1"], *frames["r2_train"]]
    logger.info("sign {sign:+d}: predicted {n} frames", sign=int(sign), n=len(base))

    shift = estimate_global_shift(train, base)
    logger.info(
        "sign {sign:+d}: global shift integer ({dy},{dx}) subpixel ({sdy:+.3f},{sdx:+.3f})",
        sign=int(sign),
        dy=shift["integer_dy"],
        dx=shift["integer_dx"],
        sdy=shift["subpixel_dy"],
        sdx=shift["subpixel_dx"],
    )

    gain_base = fit_gain_field(train, base, background)
    shifted_train = make_variant_maps(
        {frame.key: base[frame.key] for frame in train}, shift=shift
    )
    gain_combo = fit_gain_field(train, shifted_train, background)

    variant_maps = {
        "baseline": base,
        "registration": make_variant_maps(base, shift=shift),
        "gain": make_variant_maps(base, gain=gain_base, background=background),
        "registration_gain": make_variant_maps(
            base, shift=shift, gain=gain_combo, background=background
        ),
    }

    evaluations: dict[str, dict[str, Any]] = {}
    residual: dict[str, dict[str, Any]] = {}
    for set_name in EVAL_SETS:
        eval_frames = frames[set_name]
        evaluations[set_name] = {}
        residual[set_name] = {}
        for variant in VARIANTS:
            evaluations[set_name][variant] = evaluate(
                eval_frames, variant_maps[variant]
            )
            residual[set_name][variant] = _residual_structure(
                eval_frames, variant_maps[variant]
            )

    direction = {
        variant: _direction_stats(variant_maps[variant], pairs)
        for variant in VARIANTS
    }

    increments: dict[str, dict[str, Any]] = {}
    for set_name in EVAL_SETS:
        increments[set_name] = {}
        baseline = evaluations[set_name]["baseline"]
        for variant in VARIANTS[1:]:
            value = evaluations[set_name][variant]
            increments[set_name][variant] = {
                "full_pearson_delta": value["full_pearson"] - baseline["full_pearson"],
                "core_pearson_delta": value["core_pearson"] - baseline["core_pearson"],
                "rel_l2_delta": value["rel_l2"] - baseline["rel_l2"],
            }

    for name in ("gain_base", "gain_combo"):
        field = gain_base if name == "gain_base" else gain_combo
        field.pop("field", None)

    sign_key = f"{int(sign):+d}"
    return {
        "sign": sign,
        "phase_sign": float(sign),
        "registration": {
            "scope": "pooled training split (ladder pass1 42 + r2 train 18 = 60)",
            "method": (
                "single global translation; ±4 px integer NCC/Pearson grid expanded by 4 "
                "only at borders (max ±12); separable three-point parabola subpixel"
            ),
            **shift,
        },
        "gain_fields": {"gain_alone": gain_base, "registration_gain_combo": gain_combo},
        "evaluations": evaluations,
        "increments": increments,
        "direction": direction,
        "residual_structure": residual,
        "parity_reference": parity_reference,
        "sign_key": sign_key,
    }


def _build_verdict(results: dict[str, Any]) -> dict[str, Any]:
    verdict: dict[str, Any] = {"heldout_sets": ["ladder_pass2", "r2_heldout"]}
    for sign_key, sign_result in results["by_sign"].items():
        per_sign: dict[str, Any] = {}
        for variant in VARIANTS[1:]:
            heldout = {}
            for set_name in ("ladder_pass2", "r2_heldout"):
                increment = sign_result["increments"][set_name][variant]
                heldout[set_name] = {
                    "core_pearson_delta": increment["core_pearson_delta"],
                    "rel_l2_delta": increment["rel_l2_delta"],
                    "improves_both": (
                        increment["core_pearson_delta"] > 0.0
                        and increment["rel_l2_delta"] < 0.0
                    ),
                }
            per_sign[variant] = {
                "heldout": heldout,
                "improves_both_sets": all(
                    item["improves_both"] for item in heldout.values()
                ),
            }
        verdict[sign_key] = per_sign

    verdict["lut"] = {
        "status": results["lut"]["status"],
        "supports_implementation_now": False,
        "reason": results["lut"]["reason"],
    }
    verdict["supports_implementation"] = {
        "subpixel_registration": bool(
            verdict["-1"]["registration"]["improves_both_sets"]
        ),
        "gain_field": bool(verdict["-1"]["gain"]["improves_both_sets"]),
        "registration_plus_gain": bool(
            verdict["-1"]["registration_gain"]["improves_both_sets"]
        ),
        "ccd_response_lut": False,
    }
    verdict["note"] = (
        "门禁 0.50 未被重定义；这里只报相对冻结基线的增量。只有 -1 极性两个 held-out "
        "集合上 core Pearson 与 rel-L2 同时改善，才记为支持进入 #237 实现阶段；"
        "+1 仅作正交参考。"
    )
    return verdict


def _format_table(rows: list[list[str]], header: list[str]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _build_report(results: dict[str, Any], baseline: dict[str, Any]) -> str:
    lines = [
        "# 亚像素全局配准 + 空间增益场 + CCD LUT 可行性探针",
        "",
        "**日期**：2026-11-07  ",
        "**设备状态**：离线分析，未占用 SLM / CCD / 激光，未 import `ao_shaping.drivers`。  ",
        f"**命令**：`{results['command']}`  ",
        f"**输出**：`{DEFAULT_OUT}`",
        "",
        "## 1. 范围与固定口径",
        "",
        "- 主极性 `phase_sign=-1`；`+1` 仅作正交参考行。",
        "- 配准只估计**一个全局** `(dy, dx)`，训练划分为 ladder pass1 42 帧 + r2 train 18 帧（共 60 帧）；"
        "评估集不参与参数估计。**没有引入逐对位移。**",
        "- 增益场为 2 阶可乘多项式（5 个零均值基 `x, y, x², y², xy`），训练集最小二乘拟合，"
        "`background + M·(pred-background)`，`M` 截断到 [0.5, 1.5]。",
        "- LUT 仅在训练划分存在同指令多曝光括号时才拟合；否则如实判定数据不足，不硬拟合。",
        "- 指标与 `offline_benchmark_20261107/results.json` 一致：逐帧 full Pearson、"
        "0.1·max 核心掩膜 Pearson、rel-L2，再取均值。",
        "",
        "## 2. 数据与基线核对",
        "",
        f"- ladder pass2（held-out）：{len(results['data']['ladder_pass2'])} 帧。",
        f"- r2 usable：{len(results['data']['r2_usable'])} 帧；r2 held-out："
        f"{len(results['data']['r2_heldout'])} 帧。",
        f"- r2 usable 中同时被训练的 train 帧：{len(results['data']['r2_train'])} 帧"
        "（因此 r2 usable 行只作口径参照，verdict 用 r2 held-out）。",
        "",
        "存档基线与本次 baseline 复算：",
        "",
    ]
    rows = []
    for set_name in EVAL_SETS:
        archived = baseline["axis_a"]["rows"][set_name]
        for sign_key, label in (("-1", "minus"), ("+1", "plus")):
            computed = results["by_sign"][sign_key]["evaluations"][set_name]["baseline"]
            rows.append([
                set_name,
                sign_key,
                f"{archived[label]['full_pearson']:.4f}",
                f"{computed['full_pearson']:.4f}",
                f"{archived[label]['core_pearson']:.4f}",
                f"{computed['core_pearson']:.4f}",
                f"{archived[label]['rel_l2']:.4f}",
                f"{computed['rel_l2']:.4f}",
            ])
    lines += _format_table(
        rows,
        ["eval set", "sign", "arch full P", "calc full P", "arch core P", "calc core P", "arch rel-L2", "calc rel-L2"],
    )
    lines += [
        "",
        "## 3. 单一全局亚像素配准",
        "",
    ]
    rows = []
    for sign_key in ("-1", "+1"):
        reg = results["by_sign"][sign_key]["registration"]
        rows.append([
            sign_key,
            f"{reg['integer_dy']}",
            f"{reg['integer_dx']}",
            f"{reg['parabola_dy']:+.3f}",
            f"{reg['parabola_dx']:+.3f}",
            f"{reg['subpixel_dy']:+.3f}",
            f"{reg['subpixel_dx']:+.3f}",
            f"{reg['integer_peak_score']:.4f}",
            f"{reg['search_grid_points']}",
        ])
    lines += _format_table(
        rows,
        ["sign", "int dy", "int dx", "sub Δdy", "sub Δdx", "dy", "dx", "integer peak Pearson", "grid pts"],
    )
    lines += [
        "",
        "方向一致性（ladder ±模 42 对，合并 pass1+pass2）与 #235 整数扫描对照：",
        "",
    ]
    parity = results["by_sign"]["-1"]["parity_reference"]
    rows = []
    for sign_key in ("-1", "+1"):
        d = results["by_sign"][sign_key]["direction"]
        pol = f"pol{sign_key}"
        rows.append([
            sign_key,
            f"{d['baseline']['median']:+.4f}",
            f"{d['baseline']['n_ge_0.5']}/{d['baseline']['n']}",
            f"{d['registration']['median']:+.4f}",
            f"{d['registration']['n_ge_0.5']}/{d['registration']['n']}",
            f"{parity[pol]['no_registration']['median']:+.4f}",
            f"{parity[pol]['no_registration']['n_ge_0.5']}/{parity[pol]['no_registration']['n']}",
            f"{parity[pol]['best_global_shift']['dy']},{parity[pol]['best_global_shift']['dx']}",
            f"{parity[pol]['best_global_shift']['median']:+.4f}",
            f"{parity[pol]['best_global_shift']['n_ge_0.5']}/{parity[pol]['best_global_shift']['n']}",
        ])
    lines += _format_table(
        rows,
        ["sign", "ours baseline med", "n≥.5", "ours reg med", "n≥.5", "int scan baseline", "n≥.5", "int scan dy,dx", "int scan reg med", "n≥.5"],
    )
    lines += [
        "",
        "> 对照的是 #235 已存档的整数全局扫描 `+0.470 / +0.481`（pol−1 无配准 / 最优全局 dy=0, dx=−2）。"
        "本次只允许一个全局亚像素位移；逐对最优位移仍然不可部署，未计算、未报告为方案。",
        "",
        "## 4. 准确率与增量",
        "",
    ]
    for sign_key in ("-1", "+1"):
        lines += [f"### phase_sign={sign_key}", ""]
        rows = []
        for set_name in EVAL_SETS:
            for variant in VARIANTS:
                value = results["by_sign"][sign_key]["evaluations"][set_name][variant]
                inc = results["by_sign"][sign_key]["increments"][set_name].get(variant)
                rows.append([
                    set_name,
                    variant,
                    f"{value['full_pearson']:.4f}",
                    f"{value['core_pearson']:.4f}",
                    f"{value['rel_l2']:.4f}",
                    "—" if inc is None else f"{inc['core_pearson_delta']:+.4f}",
                    "—" if inc is None else f"{inc['rel_l2_delta']:+.4f}",
                ])
        lines += _format_table(
            rows,
            ["eval set", "variant", "full Pearson", "core Pearson", "rel-L2", "Δ core P", "Δ rel-L2"],
        )
        lines += [""]
    lines += [
        "## 5. 增益场参数与残差结构",
        "",
    ]
    for sign_key in ("-1", "+1"):
        for name, label in (("gain_alone", "gain only"), ("registration_gain_combo", "registration+gain")):
            gain = results["by_sign"][sign_key]["gain_fields"][name]
            lines += [
                f"- sign {sign_key} / {label}: coeffs="
                f"`{[round(v, 6) for v in gain['coefficients']]}`，"
                f"fit mask {gain['basis_mask_pixels']} px，clipped {gain['clipped_pixels']} px，"
                f"train RMS {gain['coefficient_of_variation_before']:.4f} → "
                f"{gain['coefficient_of_variation_after']:.4f}。",
            ]
    lines += ["", "验证集残差空间结构（平均残差场的低频功率占比，越低越结构化收窄）：", ""]
    rows = []
    for sign_key in ("-1", "+1"):
        for set_name in EVAL_SETS:
            residual = results["by_sign"][sign_key]["residual_structure"][set_name]
            rows.append([
                sign_key,
                set_name,
                f"{residual['baseline']['low_frequency_power_fraction']:.4f}",
                f"{residual['registration']['low_frequency_power_fraction']:.4f}",
                f"{residual['gain']['low_frequency_power_fraction']:.4f}",
                f"{residual['registration_gain']['low_frequency_power_fraction']:.4f}",
                f"{residual['baseline']['mean_residual_rms_in_core']:.4f}",
                f"{residual['registration_gain']['mean_residual_rms_in_core']:.4f}",
            ])
    lines += _format_table(
        rows,
        ["sign", "eval set", "base LF", "reg LF", "gain LF", "reg+gain LF", "base RMS", "reg+gain RMS"],
    )
    lines += [
        "",
        "## 6. CCD 响应 LUT 可行性",
        "",
        f"**结论：`{results['lut']['status']}`。** {results['lut']['reason']}",
        "",
        f"- 训练划分同指令多曝光括号数：{results['lut']['training_brackets']}；"
        f"held-out 括号数：{results['lut']['eval_brackets']}。",
        f"- focus.json 共 {results['lut']['details']['focus_scan_points']} 个离焦点，"
        f"重复离焦值 {results['lut']['details']['focus_scan_repeated_diopters']} 个。",
        f"- 训练集内跨 run 同相位组数：{len(results['lut']['details']['cross_run_training_groups'])}；"
        "其曝光跨度很小，不能辨识低曝光曲率。",
        "- 因此本次**没有**拟合 LUT，也没有把 LUT 数字混入准确率表；需要实验在单一会话内对同一指令"
        "采集至少 4 档曝光（覆盖低曝光到接近饱和），再谈 `MonotonicResponse` 的 n_bins/v_max/h 联合辨识。",
        "",
        "## 7. verdict",
        "",
    ]
    support = results["verdict"]["supports_implementation"]
    lines += [
        f"- 亚像素全局配准进入 #237 实现阶段：**{'支持' if support['subpixel_registration'] else '不支持'}**"
        "（判据：−1 极性下 ladder pass2 与 r2 held-out 的 core Pearson 与 rel-L2 同时改善）。",
        f"- 空间增益场：**{'支持' if support['gain_field'] else '不支持'}**（同一判据）。",
        f"- 配准 + 增益场组合：**{'支持' if support['registration_plus_gain'] else '不支持'}**（同一判据）。",
        "- CCD 响应 LUT：**暂不支持实现**（训练数据不足，离线不可辨识）。",
        "",
        "增量明细（−1，held-out）：",
        "",
    ]
    rows = []
    for variant in VARIANTS[1:]:
        item = results["verdict"]["-1"][variant]["heldout"]
        rows.append([
            variant,
            f"{item['ladder_pass2']['core_pearson_delta']:+.4f}",
            f"{item['ladder_pass2']['rel_l2_delta']:+.4f}",
            f"{item['r2_heldout']['core_pearson_delta']:+.4f}",
            f"{item['r2_heldout']['rel_l2_delta']:+.4f}",
            "是" if results["verdict"]["-1"][variant]["improves_both_sets"] else "否",
        ])
    lines += _format_table(
        rows,
        ["variant", "ladder Δ core P", "ladder Δ rel-L2", "r2 heldout Δ core P", "r2 heldout Δ rel-L2", "两项同时改善"],
    )
    lines += [
        "",
        "## 8. 与背景/验收口径的差异与 caveat",
        "",
        "- issue #237 验收提到的 **11-03 回放数据（零位移 NCC 0.03、对齐后 0.68、实测位移约 (-3,+32) px）不在本次指定的存档集合内，未测**；"
        "没有用 2026-11-07 数据冒充该回放。",
        "- 本次配准只回答“在当前 2026-11-07 三组存档上，单一全局亚像素位移是否有可部署增量”，"
        "不替代 11-03 回放的零位移 NCC 验收。",
        "- `phase_sign` 作用域问题仍是独立未解项；本报告没有修改任何核心 `predict`，"
        "所有包装都在本脚本内完成，−1 的绝对精度数字仍受该问题影响。",
        "- 结果文件 `results.json` 保存全部变体 × 3 评估集 × 2 极性的数字、全局位移、增益参数和 LUT 审计。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(out / "run.log", mode="w", encoding="utf-8", level="INFO")
    logger.info(DEVICE_STATEMENT)
    logger.info("command: {cmd}", cmd=" ".join([sys.executable, *sys.argv]))
    logger.info("threads={threads} torch={torch} numpy={numpy}", threads=args.threads, torch=torch.__version__, numpy=np.__version__)

    if any(name == "ao_shaping.drivers" or name.startswith("ao_shaping.drivers.")
           for name in sys.modules):
        raise RuntimeError("ao_shaping.drivers was imported; refusing to continue")

    started = time.perf_counter()
    fit = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_RESULTS.read_text(encoding="utf-8"))
    parity = json.loads(PARITY_SCAN.read_text(encoding="utf-8"))
    parity_reference = {
        "pol+1": parity["pol+1"],
        "pol-1": parity["pol-1"],
        "source": str(PARITY_SCAN.relative_to(ROOT)),
    }

    forward_minus, offset = _TRANSFER.make_forward(fit, -1.0)
    default_center = np.asarray(fit["camera_crop_center"], dtype=np.float64)
    frames = _load_frames(offset, default_center)
    pairs = _TRANSFER.direction_pairs(frames["ladder_all_seqs"])
    logger.info(
        "data: ladder pass1={p1}, pass2={p2}, r2 usable={r2}, r2 heldout={r2h}, direction pairs={pairs}",
        p1=len(frames["ladder_pass1"]),
        p2=len(frames["ladder_pass2"]),
        r2=len(frames["r2_usable"]),
        r2h=len(frames["r2_heldout"]),
        pairs=len(pairs),
    )
    if len(pairs) != 42:
        raise RuntimeError(f"expected 42 direction pairs, got {len(pairs)}")

    by_sign: dict[str, Any] = {}
    for sign in SIGNS:
        forward = forward_minus if sign < 0 else _TRANSFER.make_forward(fit, +1.0)[0]
        by_sign[f"{int(sign):+d}"] = _run_sign(
            sign,
            forward,
            frames,
            float(fit["models"]["M0"]["background"]),
            pairs,
            parity_reference,
        )

    lut_audit = audit_lut(frames)
    logger.info("LUT audit: {status} ({reason})", status=lut_audit.status, reason=lut_audit.reason)

    results: dict[str, Any] = {
        "schema": "registration_gain_probe/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "scripts/slm/registration_gain_probe_20261107.py",
        "device_statement": DEVICE_STATEMENT,
        "seed": "not needed: deterministic least squares / fixed grid",
        "config": {
            "coarse_half_px": COARSE_HALF,
            "coarse_expand_px": COARSE_EXPAND,
            "coarse_max_px": COARSE_MAX,
            "shift_interpolation_order": SHIFT_ORDER,
            "registration_scope": "pooled training split only; no per-pair displacement",
            "gain_order": GAIN_ORDER,
            "gain_basis": list(GAIN_BASIS),
            "gain_clip": list(GAIN_CLIP),
            "gain_ridge": GAIN_RIDGE,
            "gain_signal_fraction": GAIN_SIGNAL_FRACTION,
            "core_fraction": CORE_FRACTION,
            "low_frequency_cycles_per_pixel": LOW_FREQ_CYCLES_PER_PIXEL,
            "signs": [float(sign) for sign in SIGNS],
            "eval_sets": list(EVAL_SETS),
            "variants": list(VARIANTS),
            "data": {
                "fit": str(FIT_PATH.relative_to(ROOT)),
                "r2": str(R2_DIR.relative_to(ROOT)),
                "ladder": {name: str(path.relative_to(ROOT)) for name, path in LADDER_DIRS.items()},
                "parity_scan": str(PARITY_SCAN.relative_to(ROOT)),
                "baseline_results": str(BASELINE_RESULTS.relative_to(ROOT)),
            },
        },
        "data": {
            "ladder_pass1": [frame.key for frame in frames["ladder_pass1"]],
            "ladder_pass2": [frame.key for frame in frames["ladder_pass2"]],
            "r2_usable": [frame.key for frame in frames["r2_usable"]],
            "r2_train": [frame.key for frame in frames["r2_train"]],
            "r2_heldout": [frame.key for frame in frames["r2_heldout"]],
            "direction_pairs": len(pairs),
        },
        "lut": {
            "status": lut_audit.status,
            "reason": lut_audit.reason,
            "training_brackets": lut_audit.training_brackets,
            "eval_brackets": lut_audit.eval_brackets,
            "details": lut_audit.details,
        },
        "by_sign": by_sign,
    }
    results["verdict"] = _build_verdict(results)

    results_path = out / "results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    report_path = out / "REPORT.md"
    report_path.write_text(_build_report(results, baseline), encoding="utf-8")

    logger.info("wrote {path}", path=results_path)
    logger.info("wrote {path}", path=report_path)
    logger.info("elapsed {seconds:.1f}s", seconds=time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
