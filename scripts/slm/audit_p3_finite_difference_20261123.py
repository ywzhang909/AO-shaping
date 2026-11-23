"""P3-1 offline audit: direction gate on the 78-row Zernike set + finite-difference Jacobian.

Device-free. Reads archived npz/JSON only: never opens SLM, CCD or laser, never
calls a capture script's ``--execute``, never touches DeviceLock.

Two independent questions, one script, one evidence file:

(A) **Direction gate recomputed on the large sample.** The 2026-11-20 direction
    holdout gate (``scripts/slm/direction_holdout_gate_20261119.py``) ran on
    ``p3_direction_pass1/pass2_20261120`` (Z5..Z11 x +/-0.05 rad, 2 passes) and
    returned pooled **4/14**. The same campaign also archived
    ``p3_zernike_20261120`` (78 rows: Z4..Z11 x {0.10, 0.20} rad x +/- plus a
    random-phase arm), which was never replayed through the gate. This script
    replays *every* amplitude level of that dataset through the gate protocol
    and additionally reports a **reproducibility ceiling** per pair (the best
    score any predictor could earn, given how repeatable the measured
    differential actually is).

(B) **Finite-difference Jacobian and identifiability.** The P0 numbers
    (``report/shape_trials/physical_refine/p0_results.json``:
    ``identifiability_rate.structured = 0.0``,
    ``cross_response_condition_number = 17425``) were measured on a
    device-free *twin* under an unfrozen mapping. This script recomputes the
    same criteria on the fitted P3 checkpoint and on the real measured data,
    using the criterion口径 read out of the branch ``feat/zernike-block-transfer``
    (``src/ao_shaping/tools/slm/transfer/zernike_block.py::identifiability_gate``
    and ``.../structured_probes.py::cross_response_matrix``) -- the branch is
    read with ``git show``, never checked out or merged.

Reused, never re-implemented:
    * ``scripts/slm/direction_holdout_gate_20261119.py`` -> ``pair_rows``,
      ``crop``, ``measured_centre``, ``build_forward``, ``run_grid``;
    * ``ao_shaping.tools.slm.transfer.jacobian_guard`` -> ``direction_agreement``
      and the single-source thresholds (0.50 / 0.85 / 2e-3);
    * ``ao_shaping.tools.slm.transfer.fit`` -> ``build_fit_model`` (production
      fit-geometry variant).

Usage::

    .venv/Scripts/python.exe scripts/slm/audit_p3_finite_difference_20261123.py

Outputs (both under ``data/debug/``, git-ignored):
    ``data/debug/slm_transfer/p3_gate_20261120/p3_finite_difference_audit.json``
    ``data/debug/slm_transfer/p3_gate_20261120/p3_finite_difference_audit.md``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "slm"))

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
    FrozenDefocusParameters,
)
from ao_shaping.tools.slm.transfer.jacobian_guard import (
    DEFAULT_HIGH,
    DEFAULT_LOW,
    DEFAULT_MIN_SIGNAL,
    direction_agreement,
)

import direction_holdout_gate_20261119 as gate

# ---------------------------------------------------------------------------
# Constants (all single-sourced; the provenance of each is recorded in the JSON)
# ---------------------------------------------------------------------------

#: Dataset replayed through the direction gate (78 rows).
DEFAULT_DATASET = ROOT / "data/debug/slm_transfer/p3_zernike_20261120"
#: Fitted P3 checkpoint (the one the 2026-11-20 gate replayed).
DEFAULT_CHECKPOINT = ROOT / "data/debug/slm_transfer/p3_fit_20261120/frozen_parameters.json"
#: Evidence directory (git-ignored).
DEFAULT_OUT_DIR = ROOT / "data/debug/slm_transfer/p3_gate_20261120"
#: P0 baseline quoted by the task (device-free twin, unfrozen mapping).
P0_BASELINE = ROOT / "report/shape_trials/physical_refine/p0_results.json"
#: Evidence file of the 14-pair small sample (2026-11-20 direction gate).
SMALL_SAMPLE_EVIDENCE = ROOT / "data/debug/slm_transfer/p3_gate_20261120/direction_gate.json"
#: 14-pair resolvability ceiling measured between the two real passes.
SMALL_SAMPLE_RESOLVABILITY = ROOT / "data/debug/slm_transfer/p3_gate_20261120/resolvability_audit.json"

#: ``zernike_block.identifiability_gate`` (branch feat/zernike-block-transfer):
#: ``DIRECTION_AGREEMENT_FLOOR = DEFAULT_LOW``, ``MIN_SIGNAL = DEFAULT_MIN_SIGNAL``,
#: ``CROSS_RATIO_FLOOR = 3.0``, ``CONDITION_NUMBER_CEIL = 50.0``.
CROSS_RATIO_FLOOR = 3.0
CONDITION_NUMBER_CEIL = 50.0
#: ``issue229_structured_probes_p0.py`` (same branch): a mode is identifiable when
#: its isolated response is >= 2x its cross-leakage.
PEAK_LEAKAGE_FLOOR = 2.0
#: Branch the criterion口径 was read from (read with ``git show`` only).
CRITERION_BRANCH = "feat/zernike-block-transfer"
CRITERION_SOURCES = {
    "branch": CRITERION_BRANCH,
    "per_mode_gate": "src/ao_shaping/tools/slm/transfer/zernike_block.py::identifiability_gate",
    "cross_response_matrix": (
        "src/ao_shaping/tools/slm/transfer/structured_probes.py::cross_response_matrix"
    ),
    "peak_leakage": "scripts/slm/issue229_structured_probes_p0.py::per_mode_peak_leakage",
    "read_method": "git show (branch never checked out, never merged)",
    "thresholds": {
        "direction_floor_signed": DEFAULT_LOW,
        "direction_strong": DEFAULT_HIGH,
        "min_signal": DEFAULT_MIN_SIGNAL,
        "cross_ratio_floor": CROSS_RATIO_FLOOR,
        "condition_number_ceil": CONDITION_NUMBER_CEIL,
        "peak_leakage_floor": PEAK_LEAKAGE_FLOOR,
    },
}

#: ``ztrain_000_Z4_+0.10`` -- mode, sign, amplitude.
ZERNIKE_LABEL_RE = re.compile(r"^ztrain_\d+_Z(\d+)_([+-])(\d+\.\d+)$")
#: Modal basis used for the design-space (structured vs random) comparison.
DESIGN_MODES: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 11)
#: Modes the ``zernike_block`` tier may release (Z4 is the defocus scalar d).
RESIDUAL_MODES: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """SHA256 of a file, empty string when unreadable."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r of two arrays (NaN when either has no spread)."""
    x = (np.asarray(a, dtype=np.float64) - float(np.mean(a))).ravel()
    y = (np.asarray(b, dtype=np.float64) - float(np.mean(b))).ravel()
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx < 1e-12 or ny < 1e-12:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def normalised_gram(columns: list[np.ndarray]) -> np.ndarray:
    """P0口径 cross-response matrix: rows normalised to unit norm, then Gram.

    Same construction as ``structured_probes.cross_response_matrix`` (branch
    ``feat/zernike-block-transfer``): the diagonal is 1 by construction, the
    off-diagonal entries are the cosines between two modal response
    directions (cross-leakage).
    """
    if not columns:
        return np.zeros((0, 0))
    stack = np.stack([np.asarray(c, dtype=np.float64).ravel() for c in columns])
    norms = np.linalg.norm(stack, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    scaled = stack / norms
    corr = scaled @ scaled.T
    return 0.5 * (corr + corr.T)


def diag_off_ratio(matrix: np.ndarray) -> float:
    """``||diag|| / ||off-diag||`` of the cross-response matrix (P0 gate口径)."""
    if matrix.size == 0:
        return float("nan")
    diag = float(np.linalg.norm(np.diag(matrix)))
    off = matrix - np.diag(np.diag(matrix))
    off_norm = float(np.linalg.norm(off))
    return diag / off_norm if off_norm > 1e-12 else float("inf")


def peak_leakage(columns: list[np.ndarray]) -> list[float]:
    """P0口径 per-mode peak/leakage of a modal response basis."""
    out: list[float] = []
    for index, column in enumerate(columns):
        vector = np.asarray(column, dtype=np.float64).ravel()
        peak = float(np.linalg.norm(vector))
        leak_sq = 0.0
        for other, other_column in enumerate(columns):
            if other == index:
                continue
            target = np.asarray(other_column, dtype=np.float64).ravel()
            projection = float(np.dot(vector, target) / (np.linalg.norm(target) + 1e-12))
            leak_sq += projection**2
        out.append(peak / (math.sqrt(leak_sq) + 1e-12))
    return out


def design_gram_condition(design: np.ndarray) -> float:
    """P0口径 design-space Gram condition number (``issue229_structured_probes_p0``).

    ``cond = s[0] / s[-1]`` of the singular values of ``D.T @ D`` -- the
    conditioning of the *probe design* in the modal coefficient space, not of
    the image-space response.
    """
    if design.size == 0:
        return float("nan")
    gram = design.T @ design
    gram = 0.5 * (gram + gram.T)
    values = np.linalg.svd(gram, compute_uv=False)
    return float(values[0] / max(values[-1], 1e-12))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_manifest(dataset: Path) -> list[dict]:
    """Manifest rows of a capture directory."""
    return json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))


def zernike_pairs(manifest: list[dict]) -> dict[tuple[int, float], dict[str, dict]]:
    """Index the single-mode +/- pairs by (mode, amplitude).

    ``direction_holdout_gate_20261119.pair_rows`` only takes the *first* match
    of a mode, so it cannot separate the 0.10 rad ladder from the 0.20 rad
    ladder of this dataset. Here the amplitude is part of the key.
    """
    pairs: dict[tuple[int, float], dict[str, dict]] = {}
    for row in manifest:
        match = ZERNIKE_LABEL_RE.match(str(row.get("label", "")))
        if match is None:
            continue
        mode = int(match.group(1))
        sign = match.group(2)
        amplitude = float(match.group(3))
        pairs.setdefault((mode, amplitude), {})[sign] = row
    return pairs


class RowSampler:
    """Crop one archive row to the window a forward model predicts on.

    The crop happens on the ``uint8`` frames *before* the float conversion, so
    the working set stays at a few hundred kilobytes per row instead of the
    ~120 MB a full-frame float64 copy would need. Values are identical: the
    crop is exact integer indexing and the rate normalisation is elementwise.
    """

    def __init__(self, dataset: Path, name: str) -> None:
        self.dataset = dataset
        self.name = name
        self._cache: dict[str, np.ndarray] = {}

    def _crop_frame(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def frames(self, row: dict) -> np.ndarray:
        """Rate-normalised sampled frames of one row (counts per ms)."""
        key = str(row["file"])
        if key not in self._cache:
            with np.load(self.dataset / key) as data:
                raw = data["raw_frames"]
            sampled = np.stack([self._crop_frame(raw[index]) for index in range(raw.shape[0])])
            self._cache[key] = sampled.astype(np.float64) / float(row["exposure_ms"])
        return self._cache[key]

    def mean_frame(self, row: dict) -> np.ndarray:
        """Mean of the sampled, rate-normalised frames."""
        return self.frames(row).mean(axis=0)

    def phase(self, row: dict) -> np.ndarray:
        """Control-grid phase (rad) stored with one manifest row."""
        with np.load(self.dataset / str(row["file"])) as data:
            return data["model_phase"].astype(np.float32)


class ProtocolSampler(RowSampler):
    """The direction-gate protocol window: ``gate.crop`` at the ellipse centre."""

    def __init__(self, dataset: Path, centre: tuple[float, float]) -> None:
        super().__init__(dataset, "protocol")
        self.centre = centre

    def _crop_frame(self, frame: np.ndarray) -> np.ndarray:
        return gate.crop(frame, self.centre)


class FitWindowSampler(RowSampler):
    """The production fit camera window (``fit.load_dataset``'s ``xy`` grid)."""

    def __init__(self, dataset: Path, xy: np.ndarray) -> None:
        super().__init__(dataset, "fit_window")
        self.xy = np.asarray(xy, dtype=np.int64)

    def _crop_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.xy[..., 1], self.xy[..., 0]]


def build_forward_variant(
    model: dict,
    ramp: list[list[float]],
    centre_xy: tuple[float, float],
    grid: int,
    *,
    phase_sign: float = -1.0,
    phase_sign_scope: str = "command",
    residual_noll_radians: dict[int, float] | None = None,
    camera_xy: np.ndarray | None = None,
) -> FrozenDefocusForward:
    """``gate.build_forward`` with two extra knobs (polarity, residual override).

    The protocol variant of part (A) uses ``gate.build_forward`` itself; this
    helper only serves the sensitivity variants and mirrors it field for
    field apart from ``phase_sign`` and ``residual_noll_radians``.
    """
    residual = (
        residual_noll_radians
        if residual_noll_radians is not None
        else model.get("residual_noll_radians", {})
    )
    params = FrozenDefocusParameters(
        defocus_diopter=float(model["defocus_diopter"]),
        origin_xy=(float(centre_xy[0]), float(centre_xy[1])),
        amplitude_center_xy=tuple(model["amplitude_center"]),
        amplitude_width_xy=tuple(model["amplitude_width"]),
        residual_noll_radians={int(mode): float(value) for mode, value in residual.items()},
        gain=float(model["gain"]),
        background=float(model["background"]),
        sigma_ccd_px=0.0,
        ramp_matrix=((ramp[0][0], ramp[0][1]), (ramp[1][0], ramp[1][1])),
        phase_sign=float(phase_sign),
        phase_sign_scope=str(phase_sign_scope),
        grid_size=int(grid),
    )
    if camera_xy is None:
        yy, xx = np.mgrid[: gate.CROP, : gate.CROP].astype(np.float64)
        camera_xy = np.stack(
            [xx + centre_xy[0] - gate.CROP / 2, yy + centre_xy[1] - gate.CROP / 2], axis=-1
        )
    return FrozenDefocusForward(params, camera_xy, device="cpu")


def predict(fwd: FrozenDefocusForward, phase: np.ndarray) -> np.ndarray:
    """Model far-field intensity of one control-grid phase."""
    with torch.no_grad():
        return fwd.predict(torch.as_tensor(np.asarray(phase, dtype=np.float32))).cpu().numpy()[0]


def set_residual(fwd: FrozenDefocusForward, residual: dict[int, float]) -> None:
    """Overwrite the frozen model's residual coefficients in place.

    ``FrozenDefocusForward`` keeps the residual coefficients in
    ``residual_values`` (one entry per mode of ``residual_modes``) and reads
    that tensor at every ``predict`` call, so mutating it is exactly
    equivalent to rebuilding the forward with different coefficients -- and
    avoids rebuilding the 1920x1920 Zernike bases 14 times per Jacobian.
    The equivalence is verified at runtime (see ``--check-mutation``).
    """
    with torch.no_grad():
        for index, mode in enumerate(fwd.residual_modes):
            fwd.residual_values[index] = float(residual[int(mode)])


# ---------------------------------------------------------------------------
# Part (A): direction gate on the 78-row dataset
# ---------------------------------------------------------------------------


def reproducibility_ceiling(
    frames_plus: np.ndarray, frames_minus: np.ndarray
) -> dict:
    """Best score any predictor could earn on this pair (measured side only).

    Every row archived three frames ~50 ms apart. Each frame index therefore
    gives one *pseudo pass* differential ``(+ frame k) - (- frame k)``; the
    Pearson between two pseudo passes is the repeatability of the measured
    differential, i.e. the ceiling of the direction metric. The differentials
    are cropped to the same window the gate metric uses.
    """
    n = int(min(frames_plus.shape[0], frames_minus.shape[0]))
    diffs = [frames_plus[k] - frames_minus[k] for k in range(n)]
    pairwise = [pearson(diffs[i], diffs[j]) for i in range(n) for j in range(i + 1, n)]
    finite = [value for value in pairwise if math.isfinite(value)]
    split = (
        pearson(
            frames_plus[0:2].mean(axis=0) - frames_minus[0:2].mean(axis=0),
            frames_plus[2:3].mean(axis=0) - frames_minus[2:3].mean(axis=0),
        )
        if n >= 3
        else float("nan")
    )
    return {
        "n_pseudo_passes": n,
        "pseudo_pass_pairwise_pearson": [round(value, 4) for value in pairwise],
        "ceiling_mean": round(float(np.mean(finite)), 4) if finite else None,
        "ceiling_max": round(float(np.max(finite)), 4) if finite else None,
        "ceiling_min": round(float(np.min(finite)), 4) if finite else None,
        "split_half_pearson": round(float(split), 4) if math.isfinite(split) else None,
        "note": (
            "the three frames are ~50 ms apart, so this is a frame-to-frame "
            "repeatability ceiling (an upper bound for a true multi-pass ceiling)"
        ),
    }


def evaluate_gate_variant(
    pairs: dict[tuple[int, float], dict[str, dict]],
    forward: FrozenDefocusForward,
    sampler: RowSampler,
    min_pearson: float,
) -> list[dict]:
    """Run the direction gate for one forward-model variant.

    ``sampler`` maps a manifest row to the window the forward predicts on: the
    protocol sampler crops CROP x CROP around the ellipse centre (the
    direction-gate protocol), the fit-window sampler uses the production fit
    camera grid.
    """
    rows: list[dict] = []
    for (mode, amplitude), sides in sorted(pairs.items()):
        if "-" not in sides or "+" not in sides:
            continue
        row_minus, row_plus = sides["-"], sides["+"]
        frames_minus = sampler.frames(row_minus)
        frames_plus = sampler.frames(row_plus)
        measured_minus = frames_minus.mean(axis=0)
        measured_plus = frames_plus.mean(axis=0)
        predicted_minus = predict(forward, sampler.phase(row_minus))
        predicted_plus = predict(forward, sampler.phase(row_plus))
        agreement = direction_agreement(
            predicted_minus, predicted_plus, measured_minus, measured_plus
        )
        correlation = float(agreement.correlation)
        reference = float(np.mean(np.abs(measured_minus))) + 1e-12
        signal = float(agreement.measured_rms) / reference
        rows.append(
            {
                "mode": int(mode),
                "amplitude_rad": float(amplitude),
                "pearson": correlation if math.isfinite(correlation) else None,
                "verdict": agreement.verdict,
                "passed": bool(math.isfinite(correlation) and correlation >= min_pearson),
                "relative_l2": float(agreement.relative_l2)
                if math.isfinite(agreement.relative_l2)
                else None,
                "predicted_rms": float(agreement.predicted_rms),
                "measured_rms": float(agreement.measured_rms),
                "anchor_mean_abs": round(reference, 4),
                "signal_measured_rms_over_anchor": round(signal, 6),
                "min_signal_threshold": DEFAULT_MIN_SIGNAL,
                "insufficient_signal": bool(signal < DEFAULT_MIN_SIGNAL),
                "reproducibility": reproducibility_ceiling(frames_plus, frames_minus),
            }
        )
    return rows


def summarise_gate(rows: list[dict]) -> dict:
    """Per-amplitude and pooled pass counts."""
    by_amplitude: dict[str, dict] = {}
    for row in rows:
        key = f"{row['amplitude_rad']:.2f}"
        bucket = by_amplitude.setdefault(key, {"passed": 0, "total": 0, "modes": []})
        bucket["total"] += 1
        bucket["passed"] += int(row["passed"])
        bucket["modes"].append(
            {"mode": row["mode"], "pearson": row["pearson"], "verdict": row["verdict"]}
        )
    for bucket in by_amplitude.values():
        bucket["rate"] = round(bucket["passed"] / bucket["total"], 4) if bucket["total"] else None
    passed = sum(1 for row in rows if row["passed"])
    residual_rows = [row for row in rows if int(row["mode"]) in RESIDUAL_MODES]
    residual_passed = sum(1 for row in residual_rows if row["passed"])
    return {
        "by_amplitude": by_amplitude,
        "pooled": {
            "passed": passed,
            "total": len(rows),
            "rate": round(passed / len(rows), 4) if rows else None,
        },
        "pooled_residual_modes_only": {
            "passed": residual_passed,
            "total": len(residual_rows),
            "rate": round(residual_passed / len(residual_rows), 4) if residual_rows else None,
            "note": "Z5..Z11 only (Z4 = defocus is not a zernike_block mode)",
        },
    }


def load_small_sample() -> dict:
    """The 14-pair small-sample evidence (2026-11-20 direction gate)."""
    if not SMALL_SAMPLE_EVIDENCE.exists():
        return {}
    data = json.loads(SMALL_SAMPLE_EVIDENCE.read_text(encoding="utf-8"))
    per_mode: dict[str, dict] = {}
    for pair in data.get("pairs", []):
        key = f"Z{pair['mode']}"
        per_mode.setdefault(key, {})[pair["pass"]] = {
            "pearson": pair.get("pearson"),
            "passed": pair.get("passed"),
        }
    return {
        "evidence_file": str(SMALL_SAMPLE_EVIDENCE),
        "pooled": data.get("pooled"),
        "verdict": data.get("verdict"),
        "min_pearson": data.get("min_pearson"),
        "modes": data.get("modes"),
        "per_mode_per_pass": per_mode,
    }


def load_small_sample_resolvability() -> dict:
    """Two-pass reproducibility ceiling of the 14-pair small sample."""
    if not SMALL_SAMPLE_RESOLVABILITY.exists():
        return {}
    data = json.loads(SMALL_SAMPLE_RESOLVABILITY.read_text(encoding="utf-8"))
    return {
        "evidence_file": str(SMALL_SAMPLE_RESOLVABILITY),
        "rows": [
            {
                "mode": row["mode"],
                "two_pass_ceiling": row["reproducibility_ceiling"],
                "measured_rms_over_reference": row["measured_rms_over_reference"],
                "resolvable": row["resolvable"],
            }
            for row in data.get("rows", [])
        ],
    }


# ---------------------------------------------------------------------------
# Part (B): finite-difference Jacobian and identifiability
# ---------------------------------------------------------------------------


def residual_jacobian(
    model: dict,
    ramp: list[list[float]],
    centre: tuple[float, float],
    grid: int,
    residual: dict[int, float],
    operating_points: dict[str, dict[int, np.ndarray]],
    delta_fraction: float,
    ladder_max: float,
    *,
    phase_sign: float = -1.0,
) -> dict:
    """Finite-difference Jacobian of the far field w.r.t. the residual modes.

    For every residual mode ``j`` the coefficient is perturbed by
    ``+/- delta_j`` with ``delta_j = 5% * |c_j|`` (clipped so the perturbed
    mode amplitude stays inside the amplitude ladder used by the campaign),
    and the column of the Jacobian is

        ``J[:, j] = (I(c_j + delta_j) - I(c_j - delta_j)) / (2 delta_j)``.

    ``operating_points`` maps an operating-point name to ``{mode: phase}``;
    the derivative of mode ``j`` is taken at that mode's own phase.
    """
    modes = sorted(residual)
    deltas: dict[int, dict] = {}
    for mode in modes:
        base = float(residual[mode])
        nominal = delta_fraction * abs(base)
        if abs(base) < ladder_max:
            applied = min(nominal, max(ladder_max - abs(base), 0.0))
        else:
            applied = nominal
        deltas[mode] = {
            "base_rad": base,
            "delta_nominal_rad": nominal,
            "delta_applied_rad": applied,
            "baseline_exceeds_ladder": bool(abs(base) > ladder_max),
            "perturbed_magnitude_rad": [
                round(abs(base) - applied, 6),
                round(abs(base) + applied, 6),
            ],
        }
    fwd = build_forward_variant(
        model, ramp, centre, grid, phase_sign=phase_sign, residual_noll_radians=residual
    )
    columns: dict[str, dict[int, np.ndarray]] = {name: {} for name in operating_points}
    for mode in modes:
        base = float(residual[mode])
        delta = deltas[mode]["delta_applied_rad"]
        if delta <= 0:
            for name in operating_points:
                columns[name][mode] = np.zeros((gate.CROP, gate.CROP))
            continue
        plus = dict(residual)
        plus[mode] = base + delta
        minus = dict(residual)
        minus[mode] = base - delta
        predicted_plus: dict[str, np.ndarray] = {}
        predicted_minus: dict[str, np.ndarray] = {}
        set_residual(fwd, plus)
        for name, phases in operating_points.items():
            phase = phases.get(mode)
            if phase is None:
                continue
            predicted_plus[name] = predict(fwd, phase)
        set_residual(fwd, minus)
        for name, phases in operating_points.items():
            phase = phases.get(mode)
            if phase is None:
                continue
            predicted_minus[name] = predict(fwd, phase)
        for name in operating_points:
            if name in predicted_plus and name in predicted_minus:
                columns[name][mode] = (
                    predicted_plus[name] - predicted_minus[name]
                ) / (2 * delta)
    set_residual(fwd, residual)
    return {"deltas": deltas, "columns": columns}


def verify_mutation_equivalence(
    model: dict,
    ramp: list[list[float]],
    centre: tuple[float, float],
    grid: int,
    residual: dict[int, float],
) -> dict:
    """Check that mutating ``residual_values`` equals rebuilding the forward."""
    probe = np.zeros((int(grid), int(grid)), dtype=np.float32)
    mutated = build_forward_variant(
        model, ramp, centre, grid, phase_sign=-1.0, residual_noll_radians=residual
    )
    set_residual(mutated, residual)
    reference_residual = {mode: value + 0.01 for mode, value in residual.items()}
    rebuilt = build_forward_variant(
        model, ramp, centre, grid, phase_sign=-1.0, residual_noll_radians=reference_residual
    )
    set_residual(mutated, reference_residual)
    difference = float(np.max(np.abs(predict(mutated, probe) - predict(rebuilt, probe))))
    return {"max_abs_difference": difference, "equivalent": difference == 0.0}


def jacobian_statistics(columns: dict[int, np.ndarray], modes: list[int]) -> dict:
    """Cross-response matrix, condition numbers, ratios and per-column coupling."""
    ordered = [columns[mode] for mode in modes]
    matrix = normalised_gram(ordered)
    leakage = peak_leakage(ordered)
    stack = np.stack([np.asarray(c, dtype=np.float64).ravel() for c in ordered])
    # (M, P) @ (P, M) -> (M, M): the Gram of the *columns* (J^T J in the
    # row-major layout used here). ``stack.T @ stack`` would build a (P, P)
    # matrix instead (P = 128*128 pixels).
    gram = stack @ stack.T
    per_column = []
    for index, mode in enumerate(modes):
        off = [float(matrix[index, j]) for j in range(len(modes)) if j != index]
        worst = int(np.argmax(np.abs(off))) if off else index
        per_column.append(
            {
                "mode": int(mode),
                "max_abs_offdiag_r": round(float(max(np.abs(off))), 4) if off else None,
                "worst_offdiag_mode": int(modes[worst]) if off else None,
                "worst_offdiag_r": round(float(matrix[index, worst]), 4) if off else None,
                "mean_abs_offdiag_r": round(float(np.mean(np.abs(off))), 4) if off else None,
                "peak_leakage_ratio": round(float(leakage[index]), 4),
                "coupled_abs_r_ge_0p50": bool(off and max(np.abs(off)) >= 0.5),
            }
        )
    identified = [item for item in per_column if item["peak_leakage_ratio"] > PEAK_LEAKAGE_FLOOR]
    return {
        "modes": [int(mode) for mode in modes],
        "cross_response_matrix": [[round(float(value), 4) for value in row] for row in matrix],
        "condition_number_of_column_correlation_matrix": round(float(np.linalg.cond(matrix)), 4),
        "condition_number_of_JtJ": round(float(np.linalg.cond(gram)), 4),
        "diag_off_ratio": round(float(diag_off_ratio(matrix)), 4),
        "per_column": per_column,
        "identifiability_rate_peak_leakage": round(len(identified) / len(per_column), 4)
        if per_column
        else None,
        "identifiable_modes": [item["mode"] for item in identified],
        "peak_leakage_floor": PEAK_LEAKAGE_FLOOR,
    }


def per_column_direction_gate(rows: list[dict]) -> dict:
    """Per-column signed direction gate (branch ``identifiability_gate`` semantics).

    A column is selected when its **signed** Pearson r against the model column
    is >= 0.50 and its signal floor is met; ``r <= -0.50`` is
    ``direction_reversed`` (inverted Jacobian), non-finite is
    ``insufficient_signal``.
    """
    selected: list[int] = []
    rejected: dict[int, str] = {}
    correlations: dict[int, float] = {}
    for row in rows:
        mode = int(row["mode"])
        correlation = row["pearson"]
        correlations[mode] = float(correlation) if correlation is not None else float("nan")
        if correlation is None:
            rejected[mode] = "insufficient_signal"
        elif correlation <= -DEFAULT_LOW:
            rejected[mode] = "direction_reversed"
        elif correlation < DEFAULT_LOW:
            rejected[mode] = "weak_direction"
        elif row["signal_measured_rms_over_anchor"] < DEFAULT_MIN_SIGNAL:
            rejected[mode] = "insufficient_signal"
        else:
            selected.append(mode)
    return {
        "signed_correlations": {str(mode): correlations[mode] for mode in sorted(correlations)},
        "selected": sorted(selected),
        "rejected": {str(mode): rejected[mode] for mode in sorted(rejected)},
        "passed_columns": len(selected),
        "total_columns": len(rows),
        "passed_columns_residual_modes": len(
            [mode for mode in selected if mode in RESIDUAL_MODES]
        ),
        "total_columns_residual_modes": len(
            [row for row in rows if int(row["mode"]) in RESIDUAL_MODES]
        ),
        "residual_modes_note": (
            "the zernike_block tier can only release Z5..Z11 (Z4 is the defocus "
            "scalar d), so the residual-mode counts are the ones that decide the tier"
        ),
    }


def measured_differential_columns(
    sampler: RowSampler,
    pairs: dict[tuple[int, float], dict[str, dict]],
    modes: list[int],
    amplitude: float,
) -> dict[int, np.ndarray]:
    """Measured differential columns ``I(+A) - I(-A)`` per mode."""
    columns: dict[int, np.ndarray] = {}
    for mode in modes:
        sides = pairs.get((mode, amplitude))
        if sides is None or "-" not in sides or "+" not in sides:
            continue
        minus = sampler.mean_frame(sides["-"])
        plus = sampler.mean_frame(sides["+"])
        columns[mode] = plus - minus
    return columns


def design_matrices(
    manifest: list[dict], seed: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Structured and random probe designs (coefficient space).

    Structured arm: the single-mode ``ztrain_*`` rows (one mode per row).
    Random arm: the random *Zernike* family rows (``rtrain_*_zernike`` and
    ``holdout_*_zernike``); their coefficients are recovered by regenerating
    the phase from the campaign seed and projecting it on the Noll basis.
    """
    from ao_shaping.tools.slm.transfer.probes import probe_grid
    from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

    basis = np.stack(
        [
            np.nan_to_num(
                np.asarray(generate_zernike_phase({mode: 1.0}, (128, 128)), dtype=np.float64)
            ).ravel()
            for mode in DESIGN_MODES
        ]
    )
    structured_rows = []
    for row in manifest:
        match = ZERNIKE_LABEL_RE.match(str(row.get("label", "")))
        if match is None:
            continue
        mode = int(match.group(1))
        amplitude = float(match.group(3)) * (1.0 if match.group(2) == "+" else -1.0)
        vector = np.zeros(len(DESIGN_MODES))
        vector[DESIGN_MODES.index(mode)] = amplitude
        structured_rows.append(vector)
    structured = np.array(structured_rows)

    random_rows: list[np.ndarray] = []
    labels: list[str] = []
    fit_residuals: list[float] = []
    for row in manifest:
        label = str(row.get("label", ""))
        if not (label.startswith("rtrain_") or label.startswith("holdout_")):
            continue
        if label.rsplit("_", 1)[-1] != "zernike":
            continue
        if label.startswith("rtrain_"):
            plan_index = int(label[len("rtrain_") : label.rfind("_")])
            _, phase = probe_grid(plan_index, 128, seed + 777)
        else:
            plan_index = int(label[len("holdout_") : label.rfind("_")])
            _, phase = probe_grid(1000 + plan_index, 128, seed + 5)
        # basis is (M, P) with one row per mode; the phase is the linear
        # combination of those rows, so the least-squares problem is
        # min ||basis.T @ c - phase||.
        coefficients = np.linalg.lstsq(
            basis.T, np.asarray(phase, dtype=np.float64).ravel(), rcond=None
        )[0]
        fit_residuals.append(
            float(
                np.linalg.norm(basis.T @ coefficients - phase.ravel())
                / (np.linalg.norm(phase) + 1e-12)
            )
        )
        random_rows.append(coefficients)
        labels.append(label)
    random_design = np.array(random_rows) if random_rows else np.zeros((0, len(DESIGN_MODES)))
    meta = {
        "structured_rows": int(structured.shape[0]),
        "random_rows": int(random_design.shape[0]),
        "random_labels": sorted(labels),
        "random_coefficient_fit_relative_residual_max": round(max(fit_residuals), 10)
        if fit_residuals
        else None,
        "design_modes": list(DESIGN_MODES),
        "seed": seed,
    }
    return structured, random_design, meta


def budget_matched_subset(design: np.ndarray, modes: list[int], budget: int) -> np.ndarray:
    """Pick ``budget`` design rows that stay balanced over ``modes``.

    The structured arm is nested by mode (all amplitudes of one mode first),
    so a plain prefix would leave later modes out and make the coefficient
    Gram rank-deficient *and* unbalanced (the Gram's diagonal is the sum of
    squared amplitudes per mode, so an unbalanced subset reports a spurious
    condition number). One row per mode is taken first, then the remaining
    rows are spread round-robin over the modes.
    """
    by_mode: dict[int, list[int]] = {}
    for index, row in enumerate(design):
        active = [position for position, value in enumerate(row) if value != 0.0]
        key = active[0] if active else -1
        by_mode.setdefault(key, []).append(index)
    picked: list[int] = []
    round_robin = 0
    while len(picked) < budget:
        added = False
        for position in modes:
            bucket = by_mode.get(position)
            if bucket:
                picked.append(bucket.pop(0))
                added = True
                if len(picked) >= budget:
                    break
        if not added:
            break
        round_robin += 1
        if round_robin > 4:
            break
    return design[sorted(picked[:budget])]


def design_column_correlation_condition(design: np.ndarray) -> float:
    """Scale-free conditioning of the probe design's *modal coupling*.

    Each mode's coefficient profile across the probe rows is normalised and
    the columns are correlated: for a single-mode (structured) design the
    correlation matrix is the identity (cond 1), for a random mixed-mode
    design it picks up the mode-to-mode coupling. Unlike the raw Gram this
    is immune to how many rows (and which amplitudes) each mode got, so it
    stays comparable when the two arms have different row budgets.
    """
    if design.size == 0:
        return float("nan")
    columns = design.T
    norms = np.linalg.norm(columns, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    scaled = columns / norms
    correlation = scaled @ scaled.T
    return float(np.linalg.cond(correlation))


def p0_recipe_recompute(
    model: dict,
    ramp: list[list[float]],
    centre: tuple[float, float],
    grid: int,
    modes: list[int],
) -> dict:
    """P0 recipe on this bench: residual-free model, unit-amplitude probes.

    ``issue229_structured_probes_p0.py`` built the cross-response matrix from a
    *twin* whose residual was zeroed and from unit-amplitude single-mode
    probes. This is the same construction on the fitted P3 checkpoint, so the
    difference against P0's 17425 is attributable to the geometry/window and
    to the checkpoint, not to the criterion.
    """
    from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

    zero_residual: dict[int, float] = {mode: 0.0 for mode in modes}
    fwd = build_forward_variant(
        model, ramp, centre, grid, phase_sign=-1.0, residual_noll_radians=zero_residual
    )
    base = predict(fwd, np.zeros((grid, grid)))
    columns = []
    for mode in modes:
        phase = np.nan_to_num(
            np.asarray(generate_zernike_phase({mode: 1.0}, (grid, grid)), dtype=np.float32)
        )
        columns.append(predict(fwd, phase) - base)
    matrix = normalised_gram(columns)
    leakage = peak_leakage(columns)
    identified = [index for index, value in enumerate(leakage) if value > PEAK_LEAKAGE_FLOOR]
    return {
        "recipe": "P0: residual-free model, unit-amplitude single-mode probes, normalised Gram",
        "modes": [int(mode) for mode in modes],
        "cross_response_condition_number": round(float(np.linalg.cond(matrix)), 4),
        "diag_off_ratio": round(float(diag_off_ratio(matrix)), 4),
        "identifiability_rate_peak_leakage": round(len(identified) / len(modes), 4)
        if modes
        else None,
        "per_mode_peak_leakage": {
            str(mode): round(float(leakage[index]), 4) for index, mode in enumerate(modes)
        },
        "window": [gate.CROP, gate.CROP],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, report: dict) -> None:
    """Chinese, conclusion-first Markdown summary of the audit."""
    part_a = report["part_a"]
    part_b = report["part_b"]
    lines: list[str] = []
    lines.append("# P3-1 方向门复算 + 有限差分 Jacobian / 可辨识性审计（2026-11-23）")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for line in report["headline"]:
        lines.append(f"- {line}")
    lines.append("")
    lines.append("## A. 方向门复算（78 行大样本）")
    lines.append("")
    lines.append(
        "口径：`scripts/slm/direction_holdout_gate_20261119.py` 的同一套流程"
        "（`build_forward` + `crop(CROP=128)` + `measured_centre` 椭圆中心 + "
        "`jacobian_guard.direction_agreement`），过门线 Pearson ≥ 0.50（`DEFAULT_LOW`）。"
    )
    lines.append("")
    lines.append("### A.1 逐幅度档逐 mode 明细（主变体：协议原样，phase_sign=-1）")
    lines.append("")
    lines.append(
        "| 幅度 (rad) | mode | Pearson | verdict | 过门 | 信号 rms/anchor | 重现性上限(均值) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in part_a["variants"]["gate_protocol"]["rows"]:
        ceiling = row["reproducibility"]["ceiling_mean"]
        lines.append(
            "| {amp} | Z{mode} | {p} | {v} | {ok} | {sig} | {ceil} |".format(
                amp=_fmt(row["amplitude_rad"], 2),
                mode=row["mode"],
                p=_fmt(row["pearson"], 4),
                v=row["verdict"],
                ok="PASS" if row["passed"] else "FAIL",
                sig=_fmt(row["signal_measured_rms_over_anchor"], 4),
                ceil=_fmt(ceiling, 4),
            )
        )
    lines.append("")
    lines.append("### A.2 各变体汇总与 14 对小样本的对照")
    lines.append("")
    lines.append("| 变体 | 0.10 rad 档 | 0.20 rad 档 | 合计 | 仅 Z5..Z11 |")
    lines.append("|---|---|---|---|---|")
    for name, variant in part_a["variants"].items():
        summary = variant["summary"]
        by_amp = summary["by_amplitude"]
        cells = []
        for key in sorted(by_amp):
            bucket = by_amp[key]
            cells.append(f"{bucket['passed']}/{bucket['total']}")
        while len(cells) < 2:
            cells.append("—")
        pooled = summary["pooled"]
        residual_only = summary["pooled_residual_modes_only"]
        lines.append(
            f"| {name} | {cells[0]} | {cells[1]} | {pooled['passed']}/{pooled['total']} | "
            f"{residual_only['passed']}/{residual_only['total']} |"
        )
    small = part_a["small_sample"]
    lines.append("")
    lines.append(
        f"14 对小样本（`{small.get('evidence_file', '')}`）：pooled "
        f"{small.get('pooled', {}).get('passed')}/{small.get('pooled', {}).get('n_pairs')}"
        f"，判决 {small.get('verdict')}（Z5..Z11、±0.05 rad、两 pass）。"
    )
    lines.append("")
    lines.append("### A.3 哪些 mode 贡献了差异（小样本 vs 本次）")
    lines.append("")
    lines.append("| mode | 小样本 pass1 | 小样本 pass2 | 本次 0.10 rad | 本次 0.20 rad |")
    lines.append("|---|---|---|---|---|")
    new_rows = {
        (int(row["mode"]), float(row["amplitude_rad"])): row
        for row in part_a["variants"]["gate_protocol"]["rows"]
    }
    for mode in range(5, 12):
        entry = small.get("per_mode_per_pass", {}).get(f"Z{mode}", {})
        pass1 = entry.get("pass1", {}).get("pearson")
        pass2 = entry.get("pass2", {}).get("pearson")
        row10 = new_rows.get((mode, 0.10))
        row20 = new_rows.get((mode, 0.20))
        lines.append(
            f"| Z{mode} | {_fmt(pass1, 4)} | {_fmt(pass2, 4)} | "
            f"{_fmt(row10['pearson'] if row10 else None, 4)} | "
            f"{_fmt(row20['pearson'] if row20 else None, 4)} |"
        )
    lines.append("")
    lines.append(
        "小样本里稳定过门的是 Z8（2/2）与 Z5/Z11（各 1/2）；本次 Z8 在两个幅度档都不过门、"
        "Z11 也退化为 0/2。但两者**不是同一实验**（幅度档 ±0.05 vs ±0.10/±0.20 rad、"
        "数据集与时刻不同），所以只能读作趋势，不能读作「幅度加大导致退化」的因果结论。"
    )
    lines.append("")
    lines.append("## B. 有限差分 Jacobian 与可辨识性")
    lines.append("")
    lines.append(
        "判据出处（只用 `git show` 读取，未 checkout/合并）："
        f"`{CRITERION_SOURCES['branch']}` 的 "
        f"`{CRITERION_SOURCES['per_mode_gate']}`、"
        f"`{CRITERION_SOURCES['cross_response_matrix']}`、"
        f"`{CRITERION_SOURCES['peak_leakage']}`。"
    )
    lines.append("")
    lines.append("### B.1 残差系数有限差分 Jacobian（拟合 checkpoint，Δ=5%|c|）")
    lines.append("")
    lines.append("| 工况 | 列数 | cond(列相关矩阵) | cond(JᵀJ) | 对角/非对角比 | 可辨识率 |")
    lines.append("|---|---|---|---|---|---|")
    for name, block in part_b["residual_jacobian"].items():
        for point, stats in block["per_operating_point"].items():
            lines.append(
                "| {n} / {p} | {m} | {c1} | {c2} | {r} | {ir} |".format(
                    n=name,
                    p=point,
                    m=len(stats["modes"]),
                    c1=_fmt(stats["condition_number_of_column_correlation_matrix"], 2),
                    c2=_fmt(stats["condition_number_of_JtJ"], 2),
                    r=_fmt(stats["diag_off_ratio"], 3),
                    ir=_fmt(stats["identifiability_rate_peak_leakage"], 3),
                )
            )
    lines.append("")
    lines.append("### B.1b 逐列耦合明细（fitted_residual / flat 工况）")
    lines.append("")
    lines.append("| 列 (mode) | 与其它列最大绝对相关 | 最强耦合列 | 平均绝对相关 | peak/leakage | 绝对相关≥0.5 耦合 |")
    lines.append("|---|---|---|---|---|---|")
    fitted_flat = part_b["residual_jacobian"]["fitted_residual"]["per_operating_point"]["flat"]
    for column in fitted_flat["per_column"]:
        lines.append(
            f"| Z{column['mode']} | {_fmt(column['max_abs_offdiag_r'], 4)} | "
            f"Z{column['worst_offdiag_mode']} | {_fmt(column['mean_abs_offdiag_r'], 4)} | "
            f"{_fmt(column['peak_leakage_ratio'], 3)} | "
            f"{'是' if column['coupled_abs_r_ge_0p50'] else '否'} |"
        )
    lines.append("")
    lines.append("### B.2 逐列方向门（模型 vs 实测，有符号）")
    lines.append("")
    lines.append("| 幅度 | mode | 有符号 r | 判定 | 过门列数/总列数 | 仅 Z5..Z11 |")
    lines.append("|---|---|---|---|---|---|")
    for block in part_b["probe_jacobian"]["per_amplitude"]:
        selected = {str(mode) for mode in block["selected"]}
        for mode, correlation in sorted(
            block["signed_correlations"].items(), key=lambda item: int(item[0])
        ):
            state = "PASS" if mode in selected else f"FAIL({block['rejected'].get(mode)})"
            lines.append(
                f"| {_fmt(block['amplitude_rad'], 2)} | Z{mode} | "
                f"{_fmt(correlation, 4)} | {state} | "
                f"{block['passed_columns']}/{block['total_columns']} | "
                f"{block['passed_columns_residual_modes']}/"
                f"{block['total_columns_residual_modes']} |"
            )
        gate_all = block["matrix_gate_on_all_columns"]
        lines.append(
            f"| {_fmt(block['amplitude_rad'], 2)} | 全列矩阵 | — | — | — | "
            f"对角/非对角 {gate_all['diag_off_ratio']}，cond {gate_all['condition_number']}，"
            f"matrix_ok={gate_all['matrix_ok']} |"
        )
    lines.append("")
    lines.append("### B.3 与 P0 基线对比")
    lines.append("")
    lines.append("| 量 | P0 基线 | 本次新算 | 说明 |")
    lines.append("|---|---|---|---|")
    for row in part_b["baseline_comparison"]["rows"]:
        new_value = row["new"]
        if isinstance(new_value, dict):
            new_value = ", ".join(f"{key}: {value}" for key, value in new_value.items())
        lines.append(f"| {row['quantity']} | {row['p0']} | {new_value} | {row['note']} |")
    lines.append("")
    lines.append("### B.4 structured vs random（设计空间口径）")
    lines.append("")
    svr = part_b["structured_vs_random"]
    lines.append(f"- 实测口径的 structured vs random 对比：**不支持**。{svr['reason']}")
    lines.append(
        f"- 设计空间（系数 Gram）口径：structured {svr['design_gram_condition_structured']} "
        f"（{svr['structured_rows']} 行）、random {svr['design_gram_condition_random']} "
        f"（{svr['random_rows']} 行）；同预算同子空间（{svr['budget_matched_subspace']}，"
        f"{svr['budget_matched_rows']} 行）下 structured "
        f"{svr['design_gram_condition_structured_budget_matched']} vs random "
        f"{svr['design_gram_condition_random_residual_subspace']}；"
        f"P0 同口径为 structured {svr['p0_design_condition_structured']} / "
        f"random {svr['p0_design_condition_random']}"
    )
    lines.append(
        f"- 与行数无关的**模态耦合**口径（设计矩阵列相关矩阵的 cond）：structured "
        f"{svr['design_coupling_condition_structured']} vs random "
        f"{svr['design_coupling_condition_random']}；同预算同子空间 structured "
        f"{svr['design_coupling_condition_structured_budget_matched']} vs random "
        f"{svr['design_coupling_condition_random_residual_subspace']}"
    )
    lines.append(f"- 要补的数据：{svr['needed_for_measured_comparison']}")
    lines.append("")
    lines.append("## 口径与限制")
    lines.append("")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 复现")
    lines.append("")
    lines.append("```")
    lines.append(report["command"])
    lines.append("```")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point: run the whole audit and write the evidence files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-pearson", type=float, default=DEFAULT_LOW)
    parser.add_argument("--delta-fraction", type=float, default=0.05)
    parser.add_argument("--ladder-max", type=float, default=0.20)
    args = parser.parse_args()

    started = time.time()
    dataset = args.dataset
    checkpoint = args.checkpoint
    if not (dataset / "manifest.json").exists():
        raise SystemExit(f"dataset manifest missing: {dataset}")
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint missing: {checkpoint}")

    manifest = load_manifest(dataset)
    model, ramp, checkpoint_variant = gate.load_checkpoint(checkpoint)
    grid = gate.run_grid(dataset, manifest)
    if grid is None:
        raise SystemExit("no npz carries model_phase")
    print(f"[0] manifest rows={len(manifest)} grid={grid}", flush=True)
    centre = gate.measured_centre(dataset, manifest)
    print(f"[0] ellipse centre={centre} ({time.time() - started:.1f}s)", flush=True)
    pairs = zernike_pairs(manifest)
    modes = sorted({mode for mode, _ in pairs})
    amplitudes = sorted({amplitude for _, amplitude in pairs})
    residual = {int(k): float(v) for k, v in model.get("residual_noll_radians", {}).items()}

    # ---------------- part (A) ----------------
    forward_protocol = gate.build_forward(model, ramp, centre, int(grid))
    forward_plus = build_forward_variant(model, ramp, centre, int(grid), phase_sign=+1.0)
    forward_negated = build_forward_variant(
        model,
        ramp,
        centre,
        int(grid),
        phase_sign=-1.0,
        residual_noll_radians={mode: -value for mode, value in residual.items()},
    )
    protocol_sampler = ProtocolSampler(dataset, centre)
    print(f"[A] forwards built ({time.time() - started:.1f}s)", flush=True)

    variants: dict[str, dict] = {
        "gate_protocol": {
            "description": "方向门协议原样：checkpoint ramp + 椭圆中心 + 128×128 窗口 + phase_sign=-1",
            "forward": forward_protocol,
            "sampler": protocol_sampler,
        },
        "gate_protocol_phase_sign_plus1": {
            "description": "同一模型但 phase_sign=+1（checkpoint 自带约定）",
            "forward": forward_plus,
            "sampler": protocol_sampler,
        },
        "gate_protocol_negated_residual": {
            "description": "同一模型但残差系数取反（极性假设探针，非结论）",
            "forward": forward_negated,
            "sampler": protocol_sampler,
        },
    }

    fit_geometry: dict = {}
    try:
        from ao_shaping.tools.slm.transfer import fit as fit_module

        components = fit_module.build_fit_model(
            dataset, checkpoint, span=160, device="cpu", adaptation="full"
        )
        fit_forward = build_forward_variant(
            model,
            components.matrix.tolist(),
            (float(components.measured_origin[0]), float(components.measured_origin[1])),
            int(grid),
            phase_sign=-1.0,
            camera_xy=np.asarray(components.xy, dtype=np.float64),
        )
        variants["fit_geometry"] = {
            "description": (
                "生产拟合几何：fit.build_fit_model 的实测 ramp + 实测原点 + "
                "161×161 步长 2 窗口 + phase_sign=-1"
            ),
            "forward": fit_forward,
            "sampler": FitWindowSampler(dataset, np.asarray(components.xy)),
        }
        fit_geometry = {
            "measured_origin": [float(value) for value in components.measured_origin],
            "ramp_matrix": components.matrix.tolist(),
            "window": list(components.xy.shape[:2]),
            "window_step_px": 2,
            "fit_rows": len(components.rows),
        }
        print(f"[A] fit geometry built ({time.time() - started:.1f}s)", flush=True)
    except Exception as exc:  # pragma: no cover - reported, never fatal
        fit_geometry = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"[A] fit geometry unavailable: {exc}", flush=True)

    variant_reports: dict[str, dict] = {}
    for name, spec in variants.items():
        rows = evaluate_gate_variant(
            pairs, spec["forward"], spec["sampler"], float(args.min_pearson)
        )
        variant_reports[name] = {
            "description": spec["description"],
            "rows": rows,
            "summary": summarise_gate(rows),
        }
        print(f"[A] {name} done ({time.time() - started:.1f}s)", flush=True)

    part_a = {
        "dataset": str(dataset),
        "manifest_sha256": sha256_file(dataset / "manifest.json"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_variant": checkpoint_variant,
        "checkpoint_phase_sign_field": "absent (FrozenDefocusParameters default +1.0)",
        "grid": int(grid),
        "centre": [float(centre[0]), float(centre[1])],
        "crop": gate.CROP,
        "min_pearson": float(args.min_pearson),
        "min_signal_threshold": DEFAULT_MIN_SIGNAL,
        "modes": [int(mode) for mode in modes],
        "amplitudes_rad": [float(value) for value in amplitudes],
        "n_manifest_rows": len(manifest),
        "n_pairs": sum(1 for sides in pairs.values() if "-" in sides and "+" in sides),
        "variants": variant_reports,
        "small_sample": load_small_sample(),
        "small_sample_resolvability": load_small_sample_resolvability(),
        "fit_geometry": fit_geometry,
    }

    # ---------------- part (B) ----------------
    mutation_check = verify_mutation_equivalence(model, ramp, centre, int(grid), residual)
    print(f"[B] residual-mutation equivalence: {mutation_check} ({time.time() - started:.1f}s)", flush=True)

    flat_points = {
        "flat": {mode: np.zeros((int(grid), int(grid)), dtype=np.float32) for mode in residual}
    }
    probe_points: dict[str, dict[int, np.ndarray]] = {}
    for amplitude in amplitudes:
        phases: dict[int, np.ndarray] = {}
        for mode in modes:
            sides = pairs.get((mode, amplitude))
            if sides is None or "+" not in sides:
                continue
            phases[mode] = protocol_sampler.phase(sides["+"])
        probe_points[f"probe@{amplitude:.2f}"] = phases

    residual_jacobian_report: dict[str, dict] = {}
    for variant_name, override, points in (
        ("fitted_residual", residual, {**flat_points, **probe_points}),
        (
            "negated_residual",
            {mode: -value for mode, value in residual.items()},
            flat_points,
        ),
    ):
        block = residual_jacobian(
            model,
            ramp,
            centre,
            int(grid),
            override,
            points,
            float(args.delta_fraction),
            float(args.ladder_max),
            phase_sign=-1.0,
        )
        residual_jacobian_report[variant_name] = {
            "deltas": {str(mode): value for mode, value in block["deltas"].items()},
            "per_operating_point": {
                name: jacobian_statistics(columns, sorted(residual))
                for name, columns in block["columns"].items()
            },
        }
        print(f"[B] residual Jacobian {variant_name} done ({time.time() - started:.1f}s)", flush=True)

    # per-column direction gate (model vs measured) + measured cross-response matrix
    probe_gate_per_amplitude = []
    for amplitude in amplitudes:
        rows = [
            row
            for row in variant_reports["gate_protocol"]["rows"]
            if math.isclose(row["amplitude_rad"], amplitude)
        ]
        block = per_column_direction_gate(rows)
        block["amplitude_rad"] = float(amplitude)
        columns = measured_differential_columns(protocol_sampler, pairs, modes, amplitude)
        ordered_modes = [mode for mode in modes if mode in columns]
        matrix = normalised_gram([columns[mode] for mode in ordered_modes])
        leakage = peak_leakage([columns[mode] for mode in ordered_modes])
        identified = [
            mode for index, mode in enumerate(ordered_modes) if leakage[index] > PEAK_LEAKAGE_FLOOR
        ]
        block["measured_cross_response"] = {
            "modes": [int(mode) for mode in ordered_modes],
            "condition_number_of_column_correlation_matrix": round(float(np.linalg.cond(matrix)), 4),
            "diag_off_ratio": round(float(diag_off_ratio(matrix)), 4),
            "identifiability_rate_peak_leakage": round(len(identified) / len(ordered_modes), 4)
            if ordered_modes
            else None,
            "identifiable_modes": [int(mode) for mode in identified],
            "per_mode_peak_leakage": {
                str(mode): round(float(leakage[index]), 4)
                for index, mode in enumerate(ordered_modes)
            },
        }
        selected = [int(mode) for mode in block["selected"]]
        if len(selected) >= 2:
            sub = np.array(
                [
                    [matrix[ordered_modes.index(i), ordered_modes.index(j)] for j in selected]
                    for i in selected
                ]
            )
            block["matrix_ok"] = bool(
                diag_off_ratio(sub) >= CROSS_RATIO_FLOOR
                and float(np.linalg.cond(sub)) <= CONDITION_NUMBER_CEIL
            )
            block["diag_off_ratio"] = round(float(diag_off_ratio(sub)), 4)
            block["condition_number"] = round(float(np.linalg.cond(sub)), 4)
            block["matrix_ok_on_selected_note"] = (
                "the selected sub-matrix passes only because the selected set is "
                f"{len(selected)} column(s); see matrix_gate_on_all_columns"
            )
        else:
            block["matrix_ok"] = False
            block["diag_off_ratio"] = None
            block["condition_number"] = None
        block["matrix_gate_on_all_columns"] = {
            "columns": [int(mode) for mode in ordered_modes],
            "diag_off_ratio": round(float(diag_off_ratio(matrix)), 4),
            "condition_number": round(float(np.linalg.cond(matrix)), 4),
            "cross_ratio_floor": CROSS_RATIO_FLOOR,
            "condition_number_ceil": CONDITION_NUMBER_CEIL,
            "matrix_ok": bool(
                diag_off_ratio(matrix) >= CROSS_RATIO_FLOOR
                and float(np.linalg.cond(matrix)) <= CONDITION_NUMBER_CEIL
            ),
        }
        probe_gate_per_amplitude.append(block)

    p0_recipe = p0_recipe_recompute(model, ramp, centre, int(grid), modes)
    # P0 itself used Z5..Z11 only (Z4 is defocus there too), so the same
    # restriction is applied for the baseline-comparable number.
    p0_recipe_residual = p0_recipe_recompute(
        model, ramp, centre, int(grid), [mode for mode in modes if mode in RESIDUAL_MODES]
    )
    print(f"[B] P0 recipe recompute done ({time.time() - started:.1f}s)", flush=True)

    # design-space structured vs random
    seed = 1104229
    config_path = dataset / "config.json"
    if config_path.exists():
        try:
            seed = int(json.loads(config_path.read_text(encoding="utf-8")).get("seed", seed))
        except (OSError, ValueError, TypeError):
            pass
    structured_design, random_design, design_meta = design_matrices(manifest, seed)
    structured_cond = design_gram_condition(structured_design)
    random_cond = design_gram_condition(random_design) if random_design.size else float("nan")
    # Budget-matched comparison inside the residual-modal subspace (Z5..Z11),
    # which is the subspace the zernike_block tier actually works in.
    residual_positions = [DESIGN_MODES.index(mode) for mode in RESIDUAL_MODES]
    budget = min(
        2 * len(RESIDUAL_MODES),
        random_design.shape[0],
    )
    matched = budget_matched_subset(
        structured_design[:, residual_positions], list(range(len(RESIDUAL_MODES))), budget
    )
    structured_cond_matched = design_gram_condition(matched) if matched.size else float("nan")
    random_cond_residual = (
        design_gram_condition(random_design[:, residual_positions])
        if random_design.size
        else float("nan")
    )
    coupling_structured = design_column_correlation_condition(structured_design)
    coupling_random = (
        design_column_correlation_condition(random_design) if random_design.size else float("nan")
    )
    coupling_structured_matched = design_column_correlation_condition(matched)
    coupling_random_residual = (
        design_column_correlation_condition(random_design[:, residual_positions])
        if random_design.size
        else float("nan")
    )
    structured_vs_random = {
        "measured_cross_response_supported": False,
        "reason": (
            "该数据集的随机族行（rtrain_*/holdout_*）没有 +/- 配对，无法构成差分列，"
            "因此不支持 P0 口径的**实测** cross-response structured vs random 对比；"
            "下面给出的是**设计空间**（系数 Gram）口径的对比，与 P0 的 "
            "condition_number.structured/random 同口径。"
        ),
        "structured_rows": design_meta["structured_rows"],
        "random_rows": design_meta["random_rows"],
        "random_labels": design_meta["random_labels"],
        "design_modes": design_meta["design_modes"],
        "random_coefficient_fit_relative_residual_max": design_meta[
            "random_coefficient_fit_relative_residual_max"
        ],
        "design_gram_condition_structured": round(float(structured_cond), 4),
        "design_gram_condition_random": round(float(random_cond), 4)
        if math.isfinite(random_cond)
        else None,
        "design_gram_condition_structured_budget_matched": round(float(structured_cond_matched), 4)
        if math.isfinite(structured_cond_matched)
        else None,
        "design_gram_condition_random_residual_subspace": round(float(random_cond_residual), 4)
        if math.isfinite(random_cond_residual)
        else None,
        "design_coupling_condition_structured": round(float(coupling_structured), 4),
        "design_coupling_condition_random": round(float(coupling_random), 4)
        if math.isfinite(coupling_random)
        else None,
        "design_coupling_condition_structured_budget_matched": round(
            float(coupling_structured_matched), 4
        ),
        "design_coupling_condition_random_residual_subspace": round(
            float(coupling_random_residual), 4
        )
        if math.isfinite(coupling_random_residual)
        else None,
        "design_coupling_note": (
            "the coupling condition is the scale-free modal-coupling measure "
            "(normalised columns of the design); it is the number to compare "
            "across arms with different row budgets"
        ),
        "budget_matched_rows": budget,
        "budget_matched_subspace": "Z5..Z11 (residual modes), one +/- pair per mode",
        "p0_design_condition_structured": 1.0,
        "p0_design_condition_random": 4.052105465029448,
        "needed_for_measured_comparison": (
            "需要同帧预算的 ±配对随机混模探针（每个随机混模相位各采 + 与 − 两行，"
            "并重复第二 pass 以估计重现性上限），以及同口径的模型前向；"
            "现有 rtrain_*/holdout_* 行只有单侧，无法成差分列。"
        ),
    }

    # P0 baseline comparison
    p0 = {}
    if P0_BASELINE.exists():
        p0 = json.loads(P0_BASELINE.read_text(encoding="utf-8"))
    fitted_stats = residual_jacobian_report["fitted_residual"]["per_operating_point"]["flat"]
    baseline_rows = [
        {
            "quantity": "identifiability_rate.structured（模型基 peak/leakage > 2）",
            "p0": p0.get("identifiability_rate", {}).get("structured"),
            "new": fitted_stats["identifiability_rate_peak_leakage"],
            "note": (
                "P0 为 device-free 双胞胎（GRID=32/相机 256²，注入残差，单位幅度单模探针）；"
                "本次为拟合 checkpoint（GRID=128/相机 128² 窗口）+ 5% 有限差分"
            ),
        },
        {
            "quantity": "cross_response_condition_number",
            "p0": p0.get("cross_response_condition_number"),
            "new": fitted_stats["condition_number_of_column_correlation_matrix"],
            "note": "本次口径：列相关矩阵（对角=1）的 cond；P0 为双胞胎实测列的同一构造",
        },
        {
            "quantity": "cross_response_condition_number（P0 配方在本台复算，Z5..Z11）",
            "p0": p0.get("cross_response_condition_number"),
            "new": p0_recipe_residual["cross_response_condition_number"],
            "note": (
                "残差清零 + 单位幅度单模探针 + 同一窗口 + 与 P0 相同的 Z5..Z11 模式集；"
                "与 P0 的差异只来自几何/窗口/模型"
            ),
        },
        {
            "quantity": "identifiability_rate.structured（P0 配方在本台复算，Z5..Z11）",
            "p0": p0.get("identifiability_rate", {}).get("structured"),
            "new": p0_recipe_residual["identifiability_rate_peak_leakage"],
            "note": "同上",
        },
        {
            "quantity": "identifiability_rate（实测差分列的 peak/leakage > 2）",
            "p0": None,
            "new": {
                f"{block['amplitude_rad']:.2f}": block["measured_cross_response"][
                    "identifiability_rate_peak_leakage"
                ]
                for block in probe_gate_per_amplitude
            },
            "note": "实测口径，P0 无对应量（双胞胎不是实测）",
        },
    ]

    part_b = {
        "residual_jacobian": residual_jacobian_report,
        "probe_jacobian": {"per_amplitude": probe_gate_per_amplitude},
        "p0_recipe_recompute": p0_recipe,
        "p0_recipe_recompute_residual_modes": p0_recipe_residual,
        "structured_vs_random": structured_vs_random,
        "mutation_equivalence_check": mutation_check,
        "baseline_comparison": {
            "p0_file": str(P0_BASELINE),
            "p0_values": {
                "identifiability_rate.structured": p0.get("identifiability_rate", {}).get(
                    "structured"
                ),
                "cross_response_condition_number": p0.get("cross_response_condition_number"),
                "gate.diag_off_ratio": p0.get("gate", {}).get("diag_off_ratio"),
                "gate.condition_number": p0.get("gate", {}).get("condition_number"),
                "condition_number.structured": p0.get("condition_number", {}).get("structured"),
                "condition_number.random": p0.get("condition_number", {}).get("random"),
            },
            "rows": baseline_rows,
            "caveat": (
                "P0 基线是在 #247 未冻结映射下、device-free 双胞胎上测的，"
                "与本次实测口径不可直接相减；差异不能直接归因于「可辨识性变好/变差」。"
            ),
        },
        "criteria_sources": CRITERION_SOURCES,
    }

    # ---------------- headline ----------------
    protocol_summary = variant_reports["gate_protocol"]["summary"]
    pooled = protocol_summary["pooled"]
    residual_only = protocol_summary["pooled_residual_modes_only"]
    by_amp = protocol_summary["by_amplitude"]
    amp_cells = ", ".join(
        f"{key} rad {bucket['passed']}/{bucket['total']}" for key, bucket in sorted(by_amp.items())
    )
    small = part_a["small_sample"]
    small_pooled = small.get("pooled", {})
    plus_summary = variant_reports["gate_protocol_phase_sign_plus1"]["summary"]["pooled"]
    negated_summary = variant_reports["gate_protocol_negated_residual"]["summary"]["pooled"]
    fit_summary = variant_reports["fit_geometry"]["summary"]["pooled"]
    protocol_rows = variant_reports["gate_protocol"]["rows"]
    ceilings = [
        (row["mode"], row["amplitude_rad"], row["reproducibility"]["ceiling_mean"])
        for row in protocol_rows
    ]
    resolvable = [item for item in ceilings if item[2] is not None and item[2] >= DEFAULT_LOW]
    unresolvable = [item for item in ceilings if item[2] is not None and item[2] < DEFAULT_LOW]
    low_signal = [row for row in protocol_rows if row["insufficient_signal"]]
    headline = [
        f"(A) 78 行大样本上方向门 pooled {pooled['passed']}/{pooled['total']}"
        f"（{amp_cells}）；只看 zernike_block 可释放的 Z5..Z11 是 "
        f"{residual_only['passed']}/{residual_only['total']}，"
        f"14 对小样本（Z5..Z11、±0.05 rad、两 pass）为 {small_pooled.get('passed')}/"
        f"{small_pooled.get('n_pairs')}：**过门率下降、不一致**，"
        "0.20 rad 档 0/8 是主要贡献差异的一档。",
        f"(B) 新算可辨识率（模型基 peak/leakage>2）= "
        f"{fitted_stats['identifiability_rate_peak_leakage']}，"
        f"cross-response cond = {fitted_stats['condition_number_of_column_correlation_matrix']:.1f}"
        "，P0 基线为 0.0 / 17425：**不能据此说「可辨识性变好」**"
        "（P0 是 #247 未冻结映射下的 device-free 双胞胎，几何/探针/残差处理都不同；"
        "同一 P0 配方在本台复算得 "
        f"{p0_recipe_residual['cross_response_condition_number']:.1f} / "
        f"{p0_recipe_residual['identifiability_rate_peak_leakage']}）。",
        f"口径敏感性：phase_sign=+1 变体 pooled {plus_summary['passed']}/{plus_summary['total']}、"
        f"残差取反变体 {negated_summary['passed']}/{negated_summary['total']}、"
        f"生产拟合几何变体 {fit_summary['passed']}/{fit_summary['total']}；"
        "每个 mode 的 Pearson 符号随命令极性整体翻转（模型差分反号），"
        "过门是单侧阈值，所以极性/残差约定一变计数就变。",
        "可分辨性上限不是主要瓶颈：16 对里 "
        f"{len(resolvable)} 对的帧间重现性上限 ≥0.50（上限本身够门），"
        f"只有 {len(unresolvable)} 对（"
        + "、".join(
            f"Z{item[0]}@{item[1]:.2f}({item[2]:.2f})" for item in unresolvable
        )
        + "）的上限本身就 <0.50，测量分辨率不够；"
        f"信号门（rms/anchor ≥ {DEFAULT_MIN_SIGNAL}）全部满足，"
        f"insufficient_signal 的对数 = {len(low_signal)}。",
    ]

    limitations = [
        "本审计只读归档 npz/JSON，未打开 SLM/CCD/激光，未运行任何采集脚本的 --execute。",
        "方向门与 14 对小样本不是同一实验：幅度档不同（0.10/0.20 rad vs 0.05 rad）、"
        "数据集与采样时刻不同；过门率差异只能作为趋势读，不能做显著性检验。",
        "可辨识性门的逐列 r 有两种口径：模型列 vs 实测列（有符号，direction_reversed 语义）与"
        "列 vs 其它列（交叉耦合）；本报告两者都给。",
        "残差系数有限差分的扰动步长 Δ=5%|c|；当 |c| 本身已超过所用幅度档（0.20 rad）时基线就已越档，"
        "脚本按相对 5% 扰动并在 JSON 中逐模式标记 baseline_exceeds_ladder。",
        "实测 cross-response 矩阵只能由 ±配对行构成，因此 structured vs random 的**实测**对比"
        "在本数据集上不成立；报告里给的是设计空间（系数 Gram）口径。",
        "P0 基线（0.0 / 17425）来自 device-free 双胞胎与 #247 未冻结映射，"
        "几何（GRID 32 / 相机 256²）、探针幅度（单位幅度）、残差处理都不同，"
        "不能与本次数值直接相减。",
        "残差取反变体只是极性假设的探针，不是结论；它的作用是把「方向门失败」"
        "与「失败原因」分开记录。",
        "probe@幅度 工况下各列取各自 mode 的探针相位，列之间不在同一工作点，"
        "其 cross-response 矩阵只作参考（flat 工况才是同一工作点）。",
    ]

    report = {
        "schema": "p3_finite_difference_audit/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - started, 1),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "command": ".venv/Scripts/python.exe scripts/slm/audit_p3_finite_difference_20261123.py",
        "headline": headline,
        "part_a": part_a,
        "part_b": part_b,
        "limitations": limitations,
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "p3_finite_difference_audit.json"
    md_path = out_dir / "p3_finite_difference_audit.md"
    json_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    write_markdown(md_path, report)

    for name, variant in variant_reports.items():
        summary = variant["summary"]
        cells = ", ".join(
            f"{key}: {bucket['passed']}/{bucket['total']}"
            for key, bucket in sorted(summary["by_amplitude"].items())
        )
        print(
            f"[A] {name}: pooled {summary['pooled']['passed']}/{summary['pooled']['total']} ({cells})"
        )
    for name, block in residual_jacobian_report.items():
        for point, stats in block["per_operating_point"].items():
            print(
                f"[B] residual Jacobian {name}/{point}: "
                f"cond={stats['condition_number_of_column_correlation_matrix']:.2f} "
                f"ratio={stats['diag_off_ratio']:.3f} "
                f"ident_rate={stats['identifiability_rate_peak_leakage']}"
            )
    print(
        f"[B] P0 recipe recompute cond={p0_recipe['cross_response_condition_number']:.2f} "
        f"ident_rate={p0_recipe['identifiability_rate_peak_leakage']}"
    )
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"total {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
