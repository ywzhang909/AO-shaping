"""Offline baseline-vs-variant comparison of one fitted physical forward model.

The script trains the *same* physical forward three times on one archived
capture, differing only in the panel geometry (ramp matrix):

* ``baseline`` reuses the archived ramp matrix (the geometry the capture fitted);
* ``corrected`` replaces it with a deswivel matrix built from the issue-247
  panel-to-CCD calibration: panel +x is placed on ``panel_to_ccd_deg``, panel +y
  exactly 90 degrees from it, the geometric mean column magnitude is preserved
  and the y/x scale ratio is set from the calibration;
* ``axis`` keeps the baseline panel +x direction (no derotation) but
  orthogonalizes the two panel axes and applies the same y/x scale ratio,
  isolating the axis-mismatch effect from the rotation effect.

All three arms see the same raw CCD sample grid, measured frames, train/held-out
split, warm-start checkpoint, radiometry fit and iteration budget, so the only
difference is the panel geometry. Held-out rows never enter training or model
selection.

Each arm's best snapshot is then exported to a frozen forward and used for the
same cumulative inverse uniformity optimization (identical mask, zero seed,
steps, learning rate and step-RMS cap). The optimized phases are finally
cross-evaluated through all three forwards; the same-arm cell is in-sample for
that forward and therefore optimistic.

No device is opened. Inputs are read-only and outputs go to a fresh run folder.
The archived ``result.json`` / ``iteration_NNNN.npz`` are used only as a
credibility guard, never as a training target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    DifferentiableUniformityOptimizer,
    FrozenDefocusForward,
    FrozenDefocusParameters,
    UniformityObjective,
    load_frozen_parameters,
    policy_cv,
)
from ao_shaping.algorithm.signal_processing.physical_forward_fit import (
    PhysicalForwardFit,
)
from ao_shaping.tools.slm.transfer.adaptation import configure_adaptation
from ao_shaping.tools.slm.transfer.device import DEVICE_CHOICES, resolve_device
from ao_shaping.tools.slm.transfer.fit import (
    FitComponents,
    fit_loss,
    spot,
    valid,
)
from ao_shaping.tools.slm.transfer.parallel_fit import EarlyStopper

DEFAULT_FIT_DIR = Path("data/debug/slm_transfer/20261117_202916/fit")
DEFAULT_CALIBRATION = Path(
    "data/debug/issue247_analysis_20261117/issue247_analysis.json"
)
DEFAULT_OUT_ROOT = Path("data/debug/deswivel_forward_inverse_20261118")

#: Forward-fit hyper-parameters of the archived run (``fit/result.json``).
DEFAULT_ITERATIONS = 200
DEFAULT_PATIENCE = 8
DEFAULT_MIN_DELTA = 1e-4


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def sanitize_json(value: Any) -> Any:
    """Replace non-finite floats so the report is strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    return value


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def corrected_ramp_matrix(
    baseline: npt.ArrayLike, *, angle_deg: float, scale_ratio: float
) -> npt.NDArray[np.float64]:
    """Deswivel panel->CCD matrix with exact orthogonal axes.

    Column 0 is the panel +x displacement, placed on ``angle_deg``; column 1 is
    panel +y, exactly 90 degrees later. Their product is preserved from the
    baseline matrix, so only the anisotropy (``scale_ratio`` = |col y|/|col x|)
    and the axis orientation differ from the fitted geometry.
    """
    matrix = np.asarray(baseline, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError(f"baseline ramp matrix must be 2x2, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("baseline ramp matrix must be finite")
    if not np.isfinite(scale_ratio) or scale_ratio <= 0:
        raise ValueError(f"scale_ratio must be positive, got {scale_ratio!r}")
    geometric_mean = float(np.sqrt(abs(np.linalg.det(matrix))))
    if not np.isfinite(geometric_mean) or geometric_mean <= 0:
        raise ValueError("baseline ramp matrix must be non-degenerate")
    magnitude_x = geometric_mean / np.sqrt(scale_ratio)
    magnitude_y = geometric_mean * np.sqrt(scale_ratio)
    angle = np.radians(float(angle_deg))
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    return np.array(
        [
            [magnitude_x * cosine, -magnitude_y * sine],
            [magnitude_x * sine, magnitude_y * cosine],
        ],
        dtype=np.float64,
    )


def axis_corrected_ramp_matrix(
    baseline: npt.ArrayLike, *, scale_ratio: float
) -> npt.NDArray[np.float64]:
    """Orthogonalize baseline panel axes and apply the y/x scale ratio.

    The panel +x direction is kept at the baseline matrix's own angle (no
    derotation); only the axis mismatch -- the non-orthogonality between the
    two panel axes and the y/x scale anisotropy -- is corrected. This isolatess
    the effect of axis mismatch from the effect of panel rotation.
    """
    matrix = np.asarray(baseline, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError(f"baseline ramp matrix must be 2x2, got {matrix.shape}")
    panel_x_deg = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    return corrected_ramp_matrix(matrix, angle_deg=panel_x_deg, scale_ratio=scale_ratio)


def load_calibration(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    alignment = data.get("image_alignment")
    if not isinstance(alignment, dict):
        raise ValueError(f"{path} has no image_alignment block")
    if "panel_to_ccd_deg" not in alignment:
        raise ValueError(f"{path} image_alignment lacks panel_to_ccd_deg")
    ratio = alignment.get("scale_ratio")
    if ratio is None:
        magnitude_x = float(alignment["panel_x_magnitude_px"])
        magnitude_y = float(alignment["panel_y_magnitude_px"])
        ratio = magnitude_y / magnitude_x
    return {
        "panel_to_ccd_deg": float(alignment["panel_to_ccd_deg"]),
        "scale_ratio": float(ratio),
        "orthogonality_error_deg": float(
            alignment.get("orthogonality_error_deg", np.nan)
        ),
    }


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArchivedInputs:
    phases: npt.NDArray[np.float32]
    measured: npt.NDArray[np.float64]
    camera_xy: npt.NDArray[np.int64]
    train: npt.NDArray[np.bool_]
    exposures: npt.NDArray[np.float64]
    rows: list[dict[str, Any]]
    provenance: dict[str, Any]
    provenance_path: Path
    inputs_path: Path

    @property
    def train_indices(self) -> list[int]:
        return np.flatnonzero(self.train).astype(int).tolist()

    @property
    def heldout_indices(self) -> list[int]:
        return np.flatnonzero(~self.train).astype(int).tolist()


def load_archived_inputs(fit_dir: Path) -> ArchivedInputs:
    inputs_path = fit_dir / "inputs.npz"
    provenance_path = fit_dir / "provenance.json"
    for path in (inputs_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(inputs_path) as archive:
        phases = np.asarray(archive["phases"], dtype=np.float32)
        measured = np.asarray(archive["measured"], dtype=np.float64)
        camera_xy = np.asarray(archive["camera_xy"], dtype=np.int64)
        train = np.asarray(archive["train"], dtype=bool)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    rows = list(provenance["rows"])
    exposures = np.asarray(
        [float(row["exposure_ms"]) for row in rows], dtype=np.float64
    )
    if phases.ndim != 3 or phases.shape[1] != phases.shape[2]:
        raise ValueError(f"phases must be (N, S, S), got {phases.shape}")
    if measured.shape != (len(rows), *camera_xy.shape[:2]):
        raise ValueError(
            f"measured {measured.shape} does not match {len(rows)} rows x {camera_xy.shape[:2]}"
        )
    if train.shape != (len(rows),) or train.all() or not train.any():
        raise ValueError("train mask must be boolean, non-empty and non-full")
    split_train = np.asarray([row["split"] == "train" for row in rows], dtype=bool)
    if not np.array_equal(split_train, train):
        raise ValueError("inputs.npz train mask disagrees with provenance row splits")
    return ArchivedInputs(
        phases=phases,
        measured=measured,
        camera_xy=camera_xy,
        train=train,
        exposures=exposures,
        rows=rows,
        provenance=provenance,
        provenance_path=provenance_path,
        inputs_path=inputs_path,
    )


def assemble_components(
    inputs: ArchivedInputs,
    checkpoint: Path,
    *,
    span: int,
) -> tuple[FitComponents, dict[str, Any]]:
    """Build the shared fit state straight from the archived ``inputs.npz``.

    ``build_fit_model`` is deliberately not used here: the capture ran with
    ``save_dataset=False`` (``sample_source: "memory"``), so the per-row
    ``sample_*.npz`` files the loader expects were never written. Every field the
    fit needs is already in ``inputs.npz``/``provenance.json``, except the model
    origin, which only the manifest's ``reference_start`` row carries.

    ``model``/``frozen``/``kernel`` are left empty on purpose: :func:`build_arm`
    rebuilds them per ramp geometry, so nothing is duplicated.
    """
    dataset = Path(inputs.provenance["dataset"])
    manifest_path = dataset / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_row = next(
        row for row in manifest if row["label"] == "reference_start" and valid(row)
    )
    measured_origin = spot(reference_row)
    old = load_frozen_parameters(checkpoint).effective_parameters()
    if int(old.grid_size) != int(inputs.phases.shape[1]):
        raise ValueError(
            f"checkpoint grid_size {old.grid_size} != phase size {inputs.phases.shape[1]}"
        )
    matrix = np.asarray(inputs.provenance["ramp_matrix"], dtype=np.float64)
    span_px = 2 * int(span)
    grid_span = (
        int(inputs.camera_xy[0, -1, 0] - inputs.camera_xy[0, 0, 0]),
        int(inputs.camera_xy[-1, 0, 1] - inputs.camera_xy[0, 0, 1]),
    )
    checks = {
        "phases_shape": list(inputs.phases.shape),
        "images_shape": list(inputs.measured.shape),
        "camera_xy_shape": list(inputs.camera_xy.shape),
        "camera_window_span_px": list(grid_span),
        "window_span_px": int(span),
        "window_matches_manifest": bool(grid_span == (span_px, span_px)),
        "n_train": int(inputs.train.sum()),
        "n_heldout": int((~inputs.train).sum()),
        "exposure_ms_unique": sorted({float(v) for v in inputs.exposures}),
        "measured_origin_px": [float(v) for v in measured_origin],
        "ramp_matrix": matrix.tolist(),
    }
    if not checks["window_matches_manifest"]:
        raise ValueError(f"camera grid {grid_span} != 2*span {span_px}")
    comps = FitComponents(
        model=None,
        rows=inputs.rows,
        phases=inputs.phases,
        images=inputs.measured,
        xy=inputs.camera_xy,
        matrix=matrix,
        ramp_samples=dict(inputs.provenance.get("ramp_samples", {})),
        train=inputs.train,
        frozen=None,
        old=old,
        measured_origin=measured_origin,
        kernel=None,
        exposure_values=inputs.exposures,
    )
    return comps, checks


# --------------------------------------------------------------------------- #
# forward fit, one arm
# --------------------------------------------------------------------------- #
@dataclass
class ArmFit:
    name: str
    matrix: npt.NDArray[np.float64]
    base_parameters: FrozenDefocusParameters
    frozen: FrozenDefocusParameters
    best_entry: dict[str, Any]
    best_prediction: npt.NDArray[np.float32]
    history: list[dict[str, Any]]
    stopped_iteration: int | None

    @property
    def iterations_run(self) -> int:
        return len(self.history) - 1


def build_arm(
    comps: FitComponents,
    matrix: npt.ArrayLike,
    *,
    device: str,
    adaptation: str = "full",
) -> tuple[FitComponents, FrozenDefocusParameters]:
    """Mirror ``build_fit_model`` but with a caller-supplied ramp matrix."""
    import torch

    matrix_array = np.asarray(matrix, dtype=np.float64)
    matrix_tuple = tuple(tuple(float(v) for v in row) for row in matrix_array)
    old = replace(comps.old, ramp_matrix=matrix_tuple)
    frozen = FrozenDefocusForward(old, comps.xy, device=device)
    basis = {4: frozen.z4.detach().cpu().numpy()}
    basis.update(
        {
            mode: block.detach().cpu().numpy()
            for mode, block in zip(frozen.residual_modes, frozen.residual_basis)
        }
    )
    model = PhysicalForwardFit(
        comps.phases,
        comps.images,
        comps.xy,
        matrix_array,
        0.0,
        comps.train,
        old.amplitude_center_xy,
        learn_amplitude=True,
        initial_defocus=old.defocus_diopter,
        learn_residual=True,
        fft_size=old.fft_size,
        pixel_samples=old.pixel_samples,
        basis_arrays=basis,
        batch_size=4,
        device=device,
        registration_shift_ccd_px=old.registration_shift_ccd_px,
    )
    with torch.no_grad():
        model.origin.copy_(torch.tensor(comps.measured_origin))
        model.log_width.copy_(torch.tensor(old.amplitude_width_xy).log())
        model.residual.copy_(
            torch.tensor(
                [
                    old.residual_noll_radians.get(mode, 0.0)
                    for mode in model.residual_modes
                ]
            )
        )
    configure_adaptation(model, adaptation)
    arm_comps = replace(
        comps,
        model=model,
        matrix=matrix_array,
        frozen=frozen,
        kernel=frozen._blaze_kernel,
    )
    return arm_comps, old


def train_arm(
    name: str,
    comps: FitComponents,
    matrix: npt.ArrayLike,
    *,
    device: str,
    iterations: int,
    patience: int,
    min_delta: float,
) -> ArmFit:
    """Run the canonical ``fit_loss`` + Adam loop with training-only selection."""
    arm_comps, base_parameters = build_arm(comps, matrix, device=device)
    model = arm_comps.model
    stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    best_loss = float("inf")
    best_entry: dict[str, Any] | None = None
    best_prediction: npt.NDArray[np.float32] | None = None
    history: list[dict[str, Any]] = []
    stopped_at: int | None = None
    started = time.monotonic()
    for iteration in range(iterations + 1):
        model.optimizer.zero_grad()
        loss, entry, predicted = fit_loss(model, arm_comps, "exposure", iteration)
        history.append(entry)
        if entry["loss"] < best_loss:
            best_loss = float(entry["loss"])
            best_entry = entry
            best_prediction = np.asarray(predicted, dtype=np.float32).copy()
        if stopper.update(entry["loss"]):
            stopped_at = iteration
            break
        loss.backward()
        model.optimizer.step()
        if iteration % 10 == 0 or iteration == iterations:
            logger.info(
                "{} iteration {} loss {:.6f} best {:.6f}",
                name,
                iteration,
                entry["loss"],
                best_loss,
            )
    if best_entry is None or best_prediction is None:
        raise RuntimeError(f"{name}: fit produced no finite snapshot")
    frozen = frozen_from_entry(base_parameters, best_entry, arm_comps.matrix)
    logger.info(
        "{}: best iteration {} loss {:.6f} stopped {} elapsed {:.1f}s",
        name,
        best_entry["iteration"],
        best_entry["loss"],
        stopped_at,
        time.monotonic() - started,
    )
    return ArmFit(
        name=name,
        matrix=np.asarray(arm_comps.matrix, dtype=np.float64),
        base_parameters=base_parameters,
        frozen=frozen,
        best_entry=best_entry,
        best_prediction=best_prediction,
        history=history,
        stopped_iteration=stopped_at,
    )


def frozen_from_entry(
    base: FrozenDefocusParameters,
    entry: dict[str, Any],
    matrix: npt.ArrayLike,
) -> FrozenDefocusParameters:
    parameters = base.to_dict()
    parameters.update(entry["parameters"])
    return FrozenDefocusParameters.from_dicts(
        parameters,
        np.asarray(matrix, dtype=np.float64).tolist(),
        fft_size=base.fft_size,
        pixel_samples=base.pixel_samples,
        grid_size=base.grid_size,
    )


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def row_metrics(prediction: npt.ArrayLike, measured: npt.ArrayLike) -> dict[str, float]:
    p = np.asarray(prediction, dtype=np.float64)
    t = np.asarray(measured, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"prediction {p.shape} != measured {t.shape}")
    residual = p - t
    target_norm = float(np.linalg.norm(t))
    target_mean = float(t.mean())
    sst = float(((t - target_mean) ** 2).sum())
    p_cv = float(p.std() / max(float(p.mean()), 1e-9))
    t_cv = float(t.std() / max(target_mean, 1e-9))
    if p.size > 1 and p.std() > 0 and t.std() > 0:
        correlation = float(np.corrcoef(p.ravel(), t.ravel())[0, 1])
    else:
        correlation = float("nan")
    return {
        "correlation": correlation,
        "relative_l2": float(np.linalg.norm(residual) / max(target_norm, 1e-9)),
        "normalized_rmse": float(
            np.sqrt((residual**2).mean()) / max(abs(target_mean), 1e-9)
        ),
        "r2": float(1.0 - (residual**2).sum() / sst) if sst > 0 else float("nan"),
        "cv_prediction": p_cv,
        "cv_measured": t_cv,
        "cv_abs_error": abs(p_cv - t_cv),
    }


def summarize(
    prediction: np.ndarray, measured: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    indices = np.flatnonzero(mask).astype(int)
    rows = [row_metrics(prediction[i], measured[i]) for i in indices]
    keys = rows[0].keys()
    return {
        "indices": indices.tolist(),
        "count": len(rows),
        "mean": {key: float(np.nanmean([row[key] for row in rows])) for key in keys},
        "per_row": rows,
    }


def predict_frozen(
    parameters: FrozenDefocusParameters,
    camera_xy: npt.NDArray[np.floating],
    phases: npt.NDArray[np.floating],
    exposures: npt.NDArray[np.floating],
    *,
    device: str,
    chunk: int = 6,
) -> npt.NDArray[np.float32]:
    import torch

    forward = FrozenDefocusForward(parameters, camera_xy, device=device)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(phases), chunk):
            block = torch.as_tensor(
                phases[start : start + chunk], dtype=torch.float32, device=device
            )
            predicted = forward.predict(
                block, exposure_ms=exposures[start : start + chunk]
            )
            outputs.append(predicted.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def arm_forward_report(
    arm: ArmFit, inputs: ArchivedInputs, device: str
) -> dict[str, Any]:
    frozen_prediction = predict_frozen(
        arm.frozen, inputs.camera_xy, inputs.phases, inputs.exposures, device=device
    )
    consistency = row_metrics(frozen_prediction.ravel(), arm.best_prediction.ravel())[
        "relative_l2"
    ]
    return {
        "matrix": arm.matrix.tolist(),
        "best": {
            "iteration": int(arm.best_entry["iteration"]),
            "loss": float(arm.best_entry["loss"]),
            "parameters": arm.best_entry["parameters"],
        },
        "stopped_iteration": arm.stopped_iteration,
        "iterations_run": arm.iterations_run,
        "train": summarize(arm.best_prediction, inputs.measured, inputs.train),
        "heldout": summarize(arm.best_prediction, inputs.measured, ~inputs.train),
        "frozen_vs_fit_prediction_relative_l2": consistency,
        "loss_trace": [
            {"iteration": int(entry["iteration"]), "loss": float(entry["loss"])}
            for entry in arm.history
        ],
    }


# --------------------------------------------------------------------------- #
# inverse uniformity
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InverseTarget:
    sampled: npt.NDArray[np.float64]
    mask_core: npt.NDArray[np.bool_]
    mask_box: npt.NDArray[np.bool_]
    peak: float
    probe_exposure_ms: float
    probe_spot_xy: npt.NDArray[np.float64]


def load_inverse_target(
    path: Path, camera_xy: npt.NDArray[np.integer], *, core_fraction: float
) -> InverseTarget:
    if not 0.0 < core_fraction < 1.0:
        raise ValueError(f"core_fraction must be in (0, 1), got {core_fraction!r}")
    with np.load(path) as probe:
        frame = np.asarray(probe["frame"], dtype=np.float64)
        probe_exposure = float(probe["exposure_ms"])
        spot_xy = np.asarray(probe["spot_xy"], dtype=np.float64)
    yy = camera_xy[..., 1].astype(int)
    xx = camera_xy[..., 0].astype(int)
    if (
        yy.min() < 0
        or xx.min() < 0
        or yy.max() >= frame.shape[0]
        or xx.max() >= frame.shape[1]
    ):
        raise ValueError("camera_xy falls outside the flat probe frame")
    sampled = frame[yy, xx]
    peak = float(sampled.max())
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("flat probe has no positive signal inside camera_xy")
    mask_core = sampled >= core_fraction * peak
    return InverseTarget(
        sampled=sampled,
        mask_core=mask_core,
        mask_box=np.ones_like(mask_core, dtype=bool),
        peak=peak,
        probe_exposure_ms=probe_exposure,
        probe_spot_xy=spot_xy,
    )


def objective_cv(objective: UniformityObjective, image: Any, *, device: str) -> float:
    """``sqrt`` of the fixed-mask CV² objective, matching ``policy_cv``."""
    import torch

    tensor = (
        image if torch.is_tensor(image) else torch.as_tensor(image, dtype=torch.float32)
    )
    tensor = tensor.to(device)
    if tensor.ndim == 3:
        tensor = tensor[0]
    with torch.no_grad():
        return float(torch.sqrt(objective.cv(tensor.detach())))


def run_inverse(
    parameters: FrozenDefocusParameters,
    *,
    camera_xy: npt.NDArray[np.floating],
    objective: UniformityObjective,
    exposure_ms: float,
    steps: int,
    lr: float,
    cap: float,
    device: str,
) -> dict[str, Any]:
    """Cumulative inverse optimization: one ``update`` per step, never ``accept``."""
    import torch

    if steps < 1:
        raise ValueError("steps must be >= 1")
    forward = FrozenDefocusForward(parameters, camera_xy, device=device)
    # Must be set before the optimizer builds its anchor cache (see its docstring).
    forward.exposure_ms = float(exposure_ms)
    seed = np.zeros((parameters.grid_size, parameters.grid_size), dtype=np.float32)
    with torch.no_grad():
        cv_start = objective_cv(objective, forward.predict(seed), device=device)
    optimizer = DifferentiableUniformityOptimizer(
        forward, objective, seed, lr=lr, step_rms_cap=cap, device=device
    )
    trace: list[dict[str, Any]] = [
        {
            "step": 0,
            "cv_after_step": cv_start,
            "cv_record": cv_start,
            "loss": None,
            "increment_rms": 0.0,
            "phase_rms": 0.0,
        }
    ]
    for _ in range(steps):
        optimizer.update()
        record = optimizer.loss_history[-1]
        candidate = optimizer.last_candidate_prediction
        if candidate is None:
            raise RuntimeError(
                "optimizer did not expose its post-step candidate prediction"
            )
        trace.append(
            {
                "step": int(record["step"]),
                "cv_after_step": objective_cv(objective, candidate, device=device),
                "cv_record": policy_cv(record),
                "loss": float(record["loss"]),
                "increment_rms": float(record["increment_rms"]),
                "phase_rms": float(record["phase_rms"]),
            }
        )
    phase = np.asarray(optimizer.current_phase, dtype=np.float32)
    final = trace[-1]
    with torch.no_grad():
        final_cv = objective_cv(objective, forward.predict(phase), device=device)
    return {
        "phase": phase,
        "trace": trace,
        "cv_start": cv_start,
        "cv_final": float(final["cv_after_step"]),
        "cv_improvement": float(cv_start - final["cv_after_step"]),
        "phase_rms": float(final["phase_rms"]),
        "same_arm_check_abs_diff": abs(float(final["cv_after_step"]) - final_cv),
    }


def cross_evaluate(
    phase: npt.NDArray[np.floating],
    parameters: FrozenDefocusParameters,
    *,
    camera_xy: npt.NDArray[np.floating],
    objective: UniformityObjective,
    exposure_ms: float,
    device: str,
) -> tuple[float, npt.NDArray[np.float32]]:
    import torch

    forward = FrozenDefocusForward(parameters, camera_xy, device=device)
    forward.exposure_ms = float(exposure_ms)
    with torch.no_grad():
        prediction = forward.predict(phase)
    return objective_cv(objective, prediction, device=device), (
        prediction.detach().cpu().numpy()[0].astype(np.float32)
    )


def inverse_config(
    target: InverseTarget,
    *,
    steps: int,
    lr: float,
    cap: float,
    exposure_ms: float,
    core_fraction: float,
    seed: str,
) -> dict[str, Any]:
    return {
        "steps": steps,
        "lr": lr,
        "step_rms_cap": cap,
        "accept_policy": "cumulative (accept is never called)",
        "seed": seed,
        "exposure_ms": exposure_ms,
        "core_fraction": core_fraction,
        "mask_core_pixels": int(target.mask_core.sum()),
        "mask_box_pixels": int(target.mask_box.sum()),
        "mask_core_bbox_yx": _bbox(target.mask_core),
        "mask_core_centroid_xy": _centroid(target.mask_core),
        "probe_peak_counts": target.peak,
        "probe_exposure_ms": target.probe_exposure_ms,
        "probe_spot_xy": target.probe_spot_xy.tolist(),
    }


def _bbox(mask: npt.NDArray[np.bool_]) -> list[int]:
    rows, columns = np.nonzero(mask)
    return [int(rows.min()), int(rows.max()), int(columns.min()), int(columns.max())]


def _centroid(mask: npt.NDArray[np.bool_]) -> list[float]:
    rows, columns = np.nonzero(mask)
    return [float(columns.mean()), float(rows.mean())]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-dir",
        type=Path,
        default=DEFAULT_FIT_DIR,
        help="Archived fit folder containing inputs.npz/provenance.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Capture dataset; default: provenance.json dataset",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Warm-start frozen_parameters.json; default: provenance.json checkpoint",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="issue-247 analysis JSON with image_alignment",
    )
    parser.add_argument(
        "--flat-probe",
        type=Path,
        default=None,
        help="Flat probe npz; default: <session>/capture/_optical_probe_flat.npz",
    )
    parser.add_argument(
        "--panel-to-ccd-deg",
        type=float,
        default=None,
        help="Override calibrated panel +x direction (CCD frame)",
    )
    parser.add_argument(
        "--scale-ratio",
        type=float,
        default=None,
        help="Override |panel +y column| / |panel +x column|",
    )
    parser.add_argument(
        "--span",
        type=int,
        default=None,
        help="Half window span in CCD px; default: provenance window_span_px",
    )
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Maximum optimizer steps per forward arm",
    )
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--inverse-steps", type=int, default=60)
    parser.add_argument("--inverse-lr", type=float, default=0.01)
    parser.add_argument("--inverse-cap", type=float, default=0.1)
    parser.add_argument(
        "--inverse-exposure-ms",
        type=float,
        default=None,
        help="Inverse forward exposure; default: median training exposure",
    )
    parser.add_argument(
        "--core-fraction",
        type=float,
        default=0.1,
        help="mask_core threshold as a fraction of the probe peak",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--guard-tolerance",
        type=float,
        default=1e-3,
        help="Max relative L2 between reproduced and archived best prediction",
    )
    parser.add_argument(
        "--allow-guard-mismatch",
        action="store_true",
        help="Continue when the archive guard fails (e.g. a smoke run)",
    )
    args = parser.parse_args(argv)
    if args.iterations < 0:
        parser.error("--iterations must be >= 0")
    if args.inverse_steps < 1:
        parser.error("--inverse-steps must be >= 1")
    if args.inverse_lr <= 0 or args.inverse_cap <= 0:
        parser.error("--inverse-lr and --inverse-cap must be positive")
    if not 0.0 < args.core_fraction < 1.0:
        parser.error("--core-fraction must be in (0, 1)")
    return args


def resolve_geometry(
    args: argparse.Namespace, baseline: npt.NDArray[np.float64]
) -> tuple[float, float, dict[str, Any] | None]:
    calibration: dict[str, Any] | None = None
    if args.calibration.is_file():
        calibration = load_calibration(args.calibration)
    angle = args.panel_to_ccd_deg
    ratio = args.scale_ratio
    if angle is None:
        if calibration is None:
            raise FileNotFoundError(
                f"{args.calibration} missing and --panel-to-ccd-deg not supplied"
            )
        angle = calibration["panel_to_ccd_deg"]
    if ratio is None:
        if calibration is None:
            raise FileNotFoundError(
                f"{args.calibration} missing and --scale-ratio not supplied"
            )
        ratio = calibration["scale_ratio"]
    if not np.isfinite(baseline).all():
        raise ValueError("archived ramp matrix is not finite")
    return float(angle), float(ratio), calibration


def new_run_dir(root: Path, name: str | None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = name or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    candidate = root / stamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    import torch

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    started = utc_now()
    clock_start = time.monotonic()

    fit_dir = args.fit_dir
    inputs = load_archived_inputs(fit_dir)
    dataset = args.dataset or Path(inputs.provenance["dataset"])
    checkpoint = args.checkpoint or Path(inputs.provenance["checkpoint"])
    span = (
        args.span if args.span is not None else int(inputs.provenance["window_span_px"])
    )
    flat_probe = args.flat_probe or (
        dataset.parent / "capture" / "_optical_probe_flat.npz"
    )
    for path in (dataset, checkpoint, flat_probe):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    baseline_matrix = np.asarray(inputs.provenance["ramp_matrix"], dtype=np.float64)
    angle, ratio, calibration = resolve_geometry(args, baseline_matrix)
    corrected_matrix = corrected_ramp_matrix(
        baseline_matrix, angle_deg=angle, scale_ratio=ratio
    )
    axis_matrix = axis_corrected_ramp_matrix(baseline_matrix, scale_ratio=ratio)
    logger.info(
        "device={} angle={:.6f} deg ratio={:.9f} axis_panel_x_deg={:.6f}",
        device,
        angle,
        ratio,
        float(np.degrees(np.arctan2(baseline_matrix[1, 0], baseline_matrix[0, 0]))),
    )

    base_comps, shared = assemble_components(inputs, checkpoint, span=span)

    baseline = train_arm(
        "baseline",
        base_comps,
        baseline_matrix,
        device=device,
        iterations=args.iterations,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    corrected = train_arm(
        "corrected",
        base_comps,
        corrected_matrix,
        device=device,
        iterations=args.iterations,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    axis = train_arm(
        "axis",
        base_comps,
        axis_matrix,
        device=device,
        iterations=args.iterations,
        patience=args.patience,
        min_delta=args.min_delta,
    )

    archived = json.loads((fit_dir / "result.json").read_text(encoding="utf-8"))
    archived_iteration = int(archived["iteration"])
    archived_prediction_path = fit_dir / f"iteration_{archived_iteration:04d}.npz"
    archived_prediction = None
    if archived_prediction_path.is_file():
        with np.load(archived_prediction_path) as archive:
            archived_prediction = np.asarray(archive["prediction"], dtype=np.float64)
    guard = {
        "shared_inputs": shared,
        "archive": {
            "best_iteration": archived_iteration,
            "best_loss": float(archived["loss"]),
            "stopped_iteration": inputs.provenance.get("stopped_iteration"),
            "early_stopped": inputs.provenance.get("early_stopped"),
            "prediction_file": archived_prediction_path.name
            if archived_prediction_path.is_file()
            else None,
        },
        "reproduced": {
            "best_iteration": int(baseline.best_entry["iteration"]),
            "best_loss": float(baseline.best_entry["loss"]),
            "best_iteration_match": int(baseline.best_entry["iteration"])
            == archived_iteration,
            "loss_abs_diff": abs(
                float(baseline.best_entry["loss"]) - float(archived["loss"])
            ),
        },
    }
    prediction_rel_l2: float | None = None
    if archived_prediction is not None and guard["reproduced"]["best_iteration_match"]:
        prediction_rel_l2 = float(
            np.linalg.norm(
                baseline.best_prediction.ravel() - archived_prediction.ravel()
            )
            / max(float(np.linalg.norm(archived_prediction.ravel())), 1e-9)
        )
        guard["reproduced"]["prediction_max_abs_diff"] = float(
            np.max(np.abs(baseline.best_prediction - archived_prediction))
        )
    guard["reproduced"]["prediction_relative_l2"] = prediction_rel_l2
    guard["passed"] = bool(
        guard["reproduced"]["best_iteration_match"]
        and prediction_rel_l2 is not None
        and prediction_rel_l2 <= args.guard_tolerance
    )
    if not guard["passed"] and not args.allow_guard_mismatch:
        raise RuntimeError(
            "archive guard failed: reproduced "
            f"iteration {guard['reproduced']['best_iteration']} loss "
            f"{guard['reproduced']['best_loss']:.6f}, archived {archived_iteration} / "
            f"{archived['loss']:.6f}, prediction relative L2 {prediction_rel_l2}; "
            "pass --allow-guard-mismatch only for a deliberately shortened run"
        )

    target = load_inverse_target(
        flat_probe, inputs.camera_xy, core_fraction=args.core_fraction
    )
    inverse_exposure = (
        float(args.inverse_exposure_ms)
        if args.inverse_exposure_ms is not None
        else float(np.median(inputs.exposures[inputs.train]))
    )
    objective = UniformityObjective(target.mask_box, target.mask_core)
    inverse_baseline = run_inverse(
        baseline.frozen,
        camera_xy=inputs.camera_xy,
        objective=objective,
        exposure_ms=inverse_exposure,
        steps=args.inverse_steps,
        lr=args.inverse_lr,
        cap=args.inverse_cap,
        device=device,
    )
    inverse_corrected = run_inverse(
        corrected.frozen,
        camera_xy=inputs.camera_xy,
        objective=objective,
        exposure_ms=inverse_exposure,
        steps=args.inverse_steps,
        lr=args.inverse_lr,
        cap=args.inverse_cap,
        device=device,
    )
    inverse_axis = run_inverse(
        axis.frozen,
        camera_xy=inputs.camera_xy,
        objective=objective,
        exposure_ms=inverse_exposure,
        steps=args.inverse_steps,
        lr=args.inverse_lr,
        cap=args.inverse_cap,
        device=device,
    )

    cross: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    arms_by_name = {"baseline": baseline, "corrected": corrected, "axis": axis}
    runs_by_name = {
        "baseline": inverse_baseline,
        "corrected": inverse_corrected,
        "axis": inverse_axis,
    }
    for phase_name, run in runs_by_name.items():
        cell: dict[str, Any] = {}
        for forward_name, arm in arms_by_name.items():
            cv, prediction = cross_evaluate(
                run["phase"],
                arm.frozen,
                camera_xy=inputs.camera_xy,
                objective=objective,
                exposure_ms=inverse_exposure,
                device=device,
            )
            cell[f"through_{forward_name}_forward"] = {
                "cv": cv,
                "same_arm": phase_name == forward_name,
            }
            predictions[f"{phase_name}_phase_{forward_name}_forward"] = prediction
        cross[f"{phase_name}_phase"] = cell

    run_dir = new_run_dir(args.out_root, args.run_name)
    finished = utc_now()
    report = {
        "schema": "deswivel_forward_inverse/2",
        "run": {
            "directory": str(run_dir),
            "started_utc": started,
            "finished_utc": finished,
            "elapsed_seconds": time.monotonic() - clock_start,
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "script_sha256": sha256_file(Path(__file__)),
            "device_requested": args.device,
            "device_resolved": device,
            "seed": args.seed,
        },
        "config": {
            "fit_dir": str(fit_dir),
            "dataset": str(dataset),
            "checkpoint": str(checkpoint),
            "flat_probe": str(flat_probe),
            "calibration": None if calibration is None else str(args.calibration),
            "span_px": span,
            "iterations": args.iterations,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "guard_tolerance": args.guard_tolerance,
            "baseline_matrix": baseline_matrix.tolist(),
            "corrected_matrix": corrected_matrix.tolist(),
            "axis_matrix": axis_matrix.tolist(),
            "corrected_geometry": {
                "panel_to_ccd_deg": angle,
                "scale_ratio": ratio,
                "geometric_mean_column_px": float(
                    np.sqrt(abs(np.linalg.det(baseline_matrix)))
                ),
                "column_magnitudes_px": np.linalg.norm(
                    corrected_matrix, axis=0
                ).tolist(),
                "column_angles_deg": np.degrees(
                    np.arctan2(corrected_matrix[1], corrected_matrix[0])
                ).tolist(),
                "orthogonality_error_deg": float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                float(
                                    corrected_matrix[:, 0]
                                    @ corrected_matrix[:, 1]
                                    / (
                                        np.linalg.norm(corrected_matrix[:, 0])
                                        * np.linalg.norm(corrected_matrix[:, 1])
                                    )
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                ),
                "calibration_orthogonality_error_deg": None
                if calibration is None
                else calibration["orthogonality_error_deg"],
            },
            "inverse": inverse_config(
                target,
                steps=args.inverse_steps,
                lr=args.inverse_lr,
                cap=args.inverse_cap,
                exposure_ms=inverse_exposure,
                core_fraction=args.core_fraction,
                seed="zeros",
            ),
            "data": {
                "inputs_sha256": sha256_file(inputs.inputs_path),
                "provenance_sha256": sha256_file(inputs.provenance_path),
                "result_sha256": sha256_file(fit_dir / "result.json"),
                "checkpoint_sha256": sha256_file(checkpoint),
                "flat_probe_sha256": sha256_file(flat_probe),
                "calibration_sha256": None
                if calibration is None
                else sha256_file(args.calibration),
                "phases_shape": list(inputs.phases.shape),
                "measured_shape": list(inputs.measured.shape),
                "train_indices": inputs.train_indices,
                "heldout_indices": inputs.heldout_indices,
                "exposures_ms": inputs.exposures.tolist(),
                "rows": [
                    {
                        "index": index,
                        "label": row["label"],
                        "split": row["split"],
                        "exposure_ms": float(row["exposure_ms"]),
                        "file": row.get("file"),
                    }
                    for index, row in enumerate(inputs.rows)
                ],
            },
        },
        "guard": guard,
        "forward": {
            "baseline": arm_forward_report(baseline, inputs, device),
            "corrected": arm_forward_report(corrected, inputs, device),
            "axis": arm_forward_report(axis, inputs, device),
        },
        "inverse": {
            "target": {
                "sampled_sha256": hashlib.sha256(
                    np.ascontiguousarray(target.sampled).tobytes()
                ).hexdigest(),
                "core_pixels": int(target.mask_core.sum()),
                "box_pixels": int(target.mask_box.sum()),
                "core_bbox_yx": _bbox(target.mask_core),
                "core_centroid_xy": _centroid(target.mask_core),
            },
            "baseline": {
                key: value for key, value in inverse_baseline.items() if key != "phase"
            },
            "corrected": {
                key: value for key, value in inverse_corrected.items() if key != "phase"
            },
            "axis": {
                key: value for key, value in inverse_axis.items() if key != "phase"
            },
            "comparison": {
                "cv_final_delta": inverse_corrected["cv_final"]
                - inverse_baseline["cv_final"],
                "cv_final_ratio": inverse_corrected["cv_final"]
                / max(inverse_baseline["cv_final"], 1e-12),
                "axis_cv_final_delta": inverse_axis["cv_final"]
                - inverse_baseline["cv_final"],
                "axis_cv_final_ratio": inverse_axis["cv_final"]
                / max(inverse_baseline["cv_final"], 1e-12),
            },
            "cross_evaluation": cross,
            "circularity_note": (
                "Each phase's same-arm CV is computed on its own fitted forward and is "
                "therefore in-sample; the off-diagonal cells quantify how much of the "
                "apparent gain depends on which forward is used."
            ),
        },
    }
    report["forward"]["comparison"] = {
        "train_correlation_delta": (
            report["forward"]["corrected"]["train"]["mean"]["correlation"]
            - report["forward"]["baseline"]["train"]["mean"]["correlation"]
        ),
        "train_relative_l2_delta": (
            report["forward"]["corrected"]["train"]["mean"]["relative_l2"]
            - report["forward"]["baseline"]["train"]["mean"]["relative_l2"]
        ),
        "heldout_correlation_delta": (
            report["forward"]["corrected"]["heldout"]["mean"]["correlation"]
            - report["forward"]["baseline"]["heldout"]["mean"]["correlation"]
        ),
        "heldout_relative_l2_delta": (
            report["forward"]["corrected"]["heldout"]["mean"]["relative_l2"]
            - report["forward"]["baseline"]["heldout"]["mean"]["relative_l2"]
        ),
        "heldout_r2_delta": (
            report["forward"]["corrected"]["heldout"]["mean"]["r2"]
            - report["forward"]["baseline"]["heldout"]["mean"]["r2"]
        ),
        "axis_train_correlation_delta": (
            report["forward"]["axis"]["train"]["mean"]["correlation"]
            - report["forward"]["baseline"]["train"]["mean"]["correlation"]
        ),
        "axis_heldout_correlation_delta": (
            report["forward"]["axis"]["heldout"]["mean"]["correlation"]
            - report["forward"]["baseline"]["heldout"]["mean"]["correlation"]
        ),
        "axis_heldout_r2_delta": (
            report["forward"]["axis"]["heldout"]["mean"]["r2"]
            - report["forward"]["baseline"]["heldout"]["mean"]["r2"]
        ),
        "delta_convention": "corrected/axis minus baseline; positive correlation/r2 and negative "
        "relative_l2 favor the variant arm",
    }

    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(
            sanitize_json(report),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        run_dir / "inverse.npz",
        camera_xy=inputs.camera_xy,
        train=inputs.train,
        exposures_ms=inputs.exposures,
        mask_core=target.mask_core,
        mask_box=target.mask_box,
        target_sampled=target.sampled,
        baseline_matrix=baseline_matrix,
        corrected_matrix=corrected_matrix,
        axis_matrix=axis_matrix,
        baseline_phase=inverse_baseline["phase"],
        corrected_phase=inverse_corrected["phase"],
        axis_phase=inverse_axis["phase"],
        baseline_cv_trace=np.asarray(
            [row["cv_after_step"] for row in inverse_baseline["trace"]],
            dtype=np.float64,
        ),
        corrected_cv_trace=np.asarray(
            [row["cv_after_step"] for row in inverse_corrected["trace"]],
            dtype=np.float64,
        ),
        axis_cv_trace=np.asarray(
            [row["cv_after_step"] for row in inverse_axis["trace"]], dtype=np.float64
        ),
        **predictions,
    )

    logger.info("wrote {} and inverse.npz", report_path)
    logger.info(
        "forward heldout correlation baseline {:.4f} corrected {:.4f}; "
        "inverse final CV baseline {:.4f} corrected {:.4f}",
        report["forward"]["baseline"]["heldout"]["mean"]["correlation"],
        report["forward"]["corrected"]["heldout"]["mean"]["correlation"],
        inverse_baseline["cv_final"],
        inverse_corrected["cv_final"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
