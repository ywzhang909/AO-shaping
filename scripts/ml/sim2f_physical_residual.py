"""Fit a physical-residual correction offline against recorded calibration frames.

What this is
============
An **offline reference** experiment, not a CCD acceptance test and not a closed
loop.  It reads the already-recorded ``flat`` / ``net`` frames of a physical
sim2f session, fits the existing physics-informed residual
(:func:`ml.sim2f.pinn.fit_residual`) through the existing calibrated forward
model, and writes the numbers.  No hardware is opened, no phase is commanded,
and no new propagator or neural network is introduced.

Why the split is copied literally
=================================
The calibration stage that produced ``calibration.json`` split its samples with::

    rng = np.random.default_rng(0)
    order = rng.permutation(N)
    n_holdout = round(0.25 * N)
    holdout = sort(order[:n_holdout])
    train = sort(order[n_holdout:])

That same split is reproduced here.  The residual optimiser is given **only the
train command phases**; the holdout frames never enter ``fit_residual``.  They
are used afterwards, only to score the frozen residual.  This keeps the
comparison honest against exactly the holdout the calibration itself used.

Duplicate commands (a real property of the recording, visible in the report)
mean several rows share one command phase.  :class:`MeasuredBench` resolves a
command by an exact lookup to the **first recorded frame** for that command; it
does not average, interpolate, register or fit.  The report therefore includes
the unique/duplicate counts, so the information lost by that lookup is explicit.

The two arms
============
``data_only``
    ``fit_residual(physics=False)``: window L1 plus the existing smooth (TV)
    regulariser.  This is the ablation.
``pinn``
    The constrained residual: ``support`` (aperture) + ``band`` (Zernike Noll
    4..n_modes) + the same data/smooth terms.  The ``exposure`` term is absent:
    the recording holds one exposure only, so no detector-linearity pair exists.

Both arms receive identical train phases, the same fixed base state, the same
initial residual seed and the same optimiser budget.  The base state is the
:class:`~ml.sim2f.calibrate.CalibratedBench` itself: its fitted static
aberration is the composition of the nominal bench state and
``calibration.aberration``, and its fitted illumination replaces the nominal
beam.  No additional aberration/illumination map is passed on top of that, which
would double-count the fitted state.

Output contract
===============
``physical_residual.json`` records the configuration, the exact train/holdout
indices, per-arm history, start/end train+holdout L1/NCC, and the evidence
status.  ``physical_residual.npz`` holds the fitted ``delta`` maps and Zernike
coefficients.  The JSON is written with ``allow_nan=False``; undefined metrics
are ``null``, never NaN.  A new output directory is required -- a default run
picks a fresh timestamped path, and an explicit existing ``--out`` is refused.

Example::

    python scripts/ml/sim2f_physical_residual.py \
      --session data/debug/sim2f_fov_calib \
      --calibration data/debug/sim2f_fov_calib_seq_20261102 \
      --epochs 200 --lr 0.03 --seed 0 --modes data_only,pinn
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.sim2f.calibrate import (
    BenchSession,
    CalibratedBench,
    CalibrationResult,
    load_bench_session,
    load_calibration,
)
from ml.sim2f.losses import prepare_frame
from ml.sim2f.optics import Sim2f
from ml.sim2f.pinn import DEFAULT_WEIGHTS, fit_residual
from ml.sim2f.window import CameraWindow

__all__ = [
    "ARM_MODES",
    "HOLDOUT_FRAC",
    "MeasuredBench",
    "ResidualModel",
    "SPLIT_SEED",
    "main",
    "run_fit",
    "split_indices",
    "write_artifacts",
]

#: Recorded conditions read from the session.  ``flat`` is the unshaped command
#: and ``net`` the recorded network phase; neither implies live shaping here.
SESSION_GROUPS: tuple[str, ...] = ("flat", "net")

#: The calibration split, frozen.  These match ``CalibrateConfig``'s defaults
#: (seed 0, holdout_frac 0.25) and are deliberately not tied to ``--seed``.
SPLIT_SEED = 0
HOLDOUT_FRAC = 0.25

ARM_MODES: tuple[str, ...] = ("data_only", "pinn")

EVIDENCE_STATUS = "offline_physical_reference_only"
ACCEPTANCE_STATUS = "not_ccd_acceptance"

#: Guard for the gain/prepared-frame denominators (mirrors ``calibrate._EPS``).
_EPS = 1e-12


def split_indices(count: int) -> tuple[np.ndarray, np.ndarray]:
    """The calibration's exact train/holdout split.

    The recipe is frozen: :data:`SPLIT_SEED` and :data:`HOLDOUT_FRAC` are module
    constants rather than arguments, so ``--seed`` (the residual-head seed)
    cannot change which frames are held out.

    Args:
        count: Number of recorded samples ``N``.

    Returns:
        ``(train, holdout)`` -- sorted integer index arrays.  The split is
        identical to ``ml.sim2f.calibrate.calibrate_bench`` for the same config.

    Raises:
        ValueError: If ``count`` is not positive.
    """
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    n_holdout = int(round(float(HOLDOUT_FRAC) * count))
    n_holdout = max(0, min(n_holdout, count - 1))
    order = np.random.default_rng(int(SPLIT_SEED)).permutation(count)
    holdout = np.sort(order[:n_holdout])
    train = np.sort(order[n_holdout:])
    return train, holdout


def _as_phase_array(phase: Any, dtype: np.dtype) -> np.ndarray:
    """Read-only numpy view of a phase batch, cast to the stored dtype."""
    if torch.is_tensor(phase):
        arr = phase.detach().cpu().numpy()
    else:
        arr = np.asarray(phase)
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"phase must be floating, got dtype {arr.dtype}")
    return np.ascontiguousarray(arr, dtype=dtype)


class MeasuredBench:
    """Exact command-phase lookup into recorded physical frames.

    This is the *device* side of :func:`fit_residual`: it presents already
    recorded measurements through the bench protocol.  It performs no
    propagation, interpolation, registration or fitting -- a command either has
    a recorded frame or the lookup raises.

    Duplicate commands (real recordings repeat the flat command) resolve to the
    **first recorded frame** for that command.  The duplicate counts are exposed
    so the report can state how much information that collapse discards.

    Attributes:
        grid: Pupil phase grid, pixels.
        ccd_px: Camera frame side, pixels.
        device: Torch device returned frames live on.
        dtype: Torch dtype returned frames use.
    """

    def __init__(
        self,
        phases: Any,
        frames: Any,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Build the lookup.

        Args:
            phases: ``(N, grid, grid)`` recorded command phases, radians.
            frames: ``(N, ccd_px, ccd_px)`` recorded frames.
            device: Device for returned torch frames.
            dtype: Torch dtype for returned frames.

        Raises:
            ValueError: If the arrays are not batched, square and consistent.
        """
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError(f"dtype must be a floating torch dtype, got {dtype!r}")
        phase_arr = np.asarray(phases)
        frame_arr = np.asarray(frames)
        if phase_arr.ndim != 3:
            raise ValueError(f"phases must be (N, grid, grid), got {phase_arr.shape}")
        if frame_arr.ndim != 3:
            raise ValueError(
                f"frames must be (N, ccd_px, ccd_px), got {frame_arr.shape}"
            )
        if phase_arr.shape[0] != frame_arr.shape[0]:
            raise ValueError(
                "phases and frames disagree on sample count: "
                f"{phase_arr.shape[0]} vs {frame_arr.shape[0]}"
            )
        if phase_arr.shape[1] != phase_arr.shape[2]:
            raise ValueError(f"phase grid must be square, got {phase_arr.shape[1:]}")
        if frame_arr.shape[1] != frame_arr.shape[2]:
            raise ValueError(f"frames must be square, got {frame_arr.shape[1:]}")
        if phase_arr.shape[0] < 1:
            raise ValueError("measurement set is empty")

        self.grid = int(phase_arr.shape[1])
        self.ccd_px = int(frame_arr.shape[1])
        self.device = torch.device(device)
        self.dtype = dtype
        self._phase_dtype = phase_arr.dtype
        self._phases = np.ascontiguousarray(phase_arr, dtype=self._phase_dtype)
        self._frames = np.ascontiguousarray(frame_arr, dtype=np.float32)
        self._phases.flags.writeable = False
        self._frames.flags.writeable = False

        self._lookup: dict[bytes, int] = {}
        self._duplicate_sizes: dict[bytes, int] = {}
        for index, row in enumerate(self._phases):
            key = row.tobytes()
            if key in self._lookup:
                self._duplicate_sizes[key] += 1
            else:
                self._lookup[key] = index
                self._duplicate_sizes[key] = 1

    @classmethod
    def from_session(
        cls,
        session: BenchSession,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "MeasuredBench":
        """Build from a :class:`~ml.sim2f.calibrate.BenchSession`."""
        return cls(session.phases(), session.frames(), device=device, dtype=dtype)

    @property
    def frame_count(self) -> int:
        """Recorded rows."""
        return int(self._phases.shape[0])

    @property
    def unique_count(self) -> int:
        """Distinct command phases."""
        return len(self._lookup)

    @property
    def duplicate_group_sizes(self) -> list[int]:
        """Sizes (>1) of command groups measured more than once."""
        return sorted(size for size in self._duplicate_sizes.values() if size > 1)

    @property
    def command_keys(self) -> tuple[bytes, ...]:
        """Per-row exact command identity, for duplicate-aware scoring."""
        return tuple(row.tobytes() for row in self._phases)

    def forward(
        self,
        phase: torch.Tensor | np.ndarray,
        exposure_ms: float | None = None,
    ) -> torch.Tensor:
        """Look up the recorded frame for every exact command phase.

        Args:
            phase: ``(grid, grid)`` or ``(N, grid, grid)`` command radians.
            exposure_ms: Not supported.  The recording has one exposure only,
                so there is no exposure-dependent measurement to return.

        Returns:
            ``(N, 1, ccd_px, ccd_px)`` measured frames on ``device``/``dtype``.

        Raises:
            ValueError: On an unsupported exposure, a shape mismatch, or a
                command phase that does not exactly match a recorded command.
        """
        if exposure_ms is not None:
            raise ValueError(
                "MeasuredBench has no exposure-dependent data: the recording "
                f"holds one exposure, so exposure_ms={exposure_ms!r} is "
                "unsupported and the detector-linearity term cannot be fitted"
            )
        arr = _as_phase_array(phase, self._phase_dtype)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim != 3 or arr.shape[1:] != (self.grid, self.grid):
            raise ValueError(
                f"phase must be ({self.grid}, {self.grid}) or "
                f"(N, {self.grid}, {self.grid}), got {arr.shape}"
            )

        indices: list[int] = []
        for row, command in enumerate(arr):
            index = self._lookup.get(command.tobytes())
            if index is None:
                raise ValueError(
                    f"unknown command phase in batch row {row}: no recorded "
                    f"{self.grid}x{self.grid} command matches exactly"
                )
            indices.append(index)

        selected = self._frames[np.asarray(indices, dtype=np.intp)]
        return (
            torch.from_numpy(selected)
            .unsqueeze(1)
            .to(device=self.device, dtype=self.dtype)
        )


class ResidualModel:
    """Protocol adapter around the calibrated bench for :func:`fit_residual`.

    ``fit_residual`` needs ``zernike_basis``, ``pupil_mask`` and
    ``nominal_illumination`` in addition to ``forward``.  ``CalibratedBench``
    deliberately does not expose those, so they are forwarded from the wrapped
    :class:`~ml.sim2f.optics.Sim2f`.  The propagation itself is always
    ``CalibratedBench.forward``: there is no second propagator here.

    The base state is fixed: the calibrated bench already applies the fitted
    static aberration (nominal bench state + ``calibration.aberration``) and the
    fitted illumination.  Passing those again as ``aberration=`` would
    double-count them, so this adapter binds the base and only allows an
    *additional* aberration (the fit passes none).
    """

    def __init__(self, bench: CalibratedBench, sim: Sim2f) -> None:
        """Bind a built ``CalibratedBench`` and the ``Sim2f`` it wraps."""
        grid = int(bench.grid)
        self._bench = bench
        self._sim = sim
        self.base_aberration = np.asarray(
            bench.calibration.aberration, dtype=np.float32
        )
        self.base_illumination = np.asarray(
            bench.calibration.illumination, dtype=np.float32
        )
        self._base_illumination = torch.as_tensor(
            self.base_illumination, dtype=bench.dtype, device=bench.device
        ).reshape(1, 1, grid, grid)

    @property
    def grid(self) -> int:
        return int(self._bench.grid)

    @property
    def ccd_px(self) -> int:
        return int(self._bench.ccd_px)

    @property
    def device(self) -> torch.device:
        return torch.device(self._bench.device)

    @property
    def dtype(self) -> torch.dtype:
        return self._bench.dtype

    @property
    def nominal_illumination(self) -> torch.Tensor:
        """The fixed calibrated beam, used for the phase-energy weighting."""
        return self._base_illumination

    @property
    def pupil_mask(self) -> torch.Tensor:
        return self._sim.pupil_mask

    def zernike_basis(self, count: int) -> torch.Tensor:
        return self._sim.zernike_basis(count)

    def forward(
        self,
        phase: torch.Tensor | np.ndarray,
        *,
        aberration: torch.Tensor | np.ndarray | None = None,
        illumination: torch.Tensor | np.ndarray | None = None,
        exposure_ms: float | None = None,
    ) -> torch.Tensor:
        """Calibrated prediction; extra exposure and illumination are refused."""
        if exposure_ms is not None:
            raise ValueError(
                "the calibrated residual model has no exposure-dependent "
                "measurement, so exposure_ms is unsupported"
            )
        if illumination is not None and illumination is not self._base_illumination:
            light = torch.as_tensor(illumination, dtype=self.dtype, device=self.device)
            if not torch.equal(light, self._base_illumination):
                raise ValueError(
                    "the residual model binds the calibrated illumination; an "
                    "additional amplitude would double-count it"
                )
        frame = self._bench.forward(
            phase, aberration=aberration, illumination=None, detector=False
        )
        return frame


def _arm_weights(arm: str) -> dict[str, float]:
    """Term weights for one arm; the exposure term is always off."""
    weights = dict(DEFAULT_WEIGHTS)
    weights["exposure"] = 0.0
    return weights


def _frame_l1(pred: np.ndarray, truth: np.ndarray) -> float:
    """Relative L1 of two already normalised frames."""
    total = float(truth.sum())
    return float(np.abs(pred - truth).sum() / total) if total > 0 else float("nan")


def _frame_ncc(pred: np.ndarray, truth: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation of two frames."""
    a = pred - pred.mean()
    b = truth - truth.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def _normalise(frame: np.ndarray) -> np.ndarray:
    total = float(frame.sum())
    return frame / total if total > 0 else frame


def _mean_or_none(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def _calibrate_matched(
    predicted: torch.Tensor, truth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-frame L1/NCC/gain exactly as ``calibrate._scores(fit_gain=True)``.

    The reference metric subtracts each frame's median pedestal
    (:func:`~ml.sim2f.losses.prepare_frame`) and then projects the brightness
    gain out in closed form, because absolute brightness is exposure x laser
    power x PRNU, not optical path.  Scoring raw frames instead would compare
    the camera pedestal against the model's zero background and would not be
    comparable with ``calibrate_summary.json``.

    Args:
        predicted: ``(B, 1, H, W)`` model frame.
        truth: ``(B, 1, H, W)`` recorded frame.

    Returns:
        ``(l1, ncc, gain)``, each ``(B,)``; degenerate frames give ``NaN``.
    """
    pred = prepare_frame(predicted.float())
    meas = prepare_frame(truth.float())
    gain = (pred * meas).sum(dim=(-2, -1)) / pred.pow(2).sum(dim=(-2, -1)).clamp_min(
        _EPS
    )
    difference = pred * gain.unsqueeze(-1).unsqueeze(-1) - meas
    l1 = difference.abs().sum(dim=(-2, -1)) / meas.sum(dim=(-2, -1)).clamp_min(_EPS)
    a = pred - pred.mean(dim=(-2, -1), keepdim=True)
    b = meas - meas.mean(dim=(-2, -1), keepdim=True)
    numerator = (a * b).sum(dim=(-2, -1))
    denominator = a.pow(2).sum(dim=(-2, -1)).sqrt() * b.pow(2).sum(dim=(-2, -1)).sqrt()
    return l1.reshape(-1), (numerator / denominator).reshape(-1), gain.reshape(-1)


def _score(
    model: ResidualModel,
    measured: MeasuredBench,
    window: CameraWindow,
    phases: np.ndarray,
    delta: torch.Tensor | None,
) -> dict[str, float | None]:
    """Mean L1/NCC/gain on a phase set, with or without the fitted residual.

    ``l1``/``ncc``/``gain`` are the calibration's own prepared, gain-projected
    metric.  ``l1_raw``/``ncc_raw`` keep the plain unit-sum comparison on the
    frames as stored, so the pedestal the prepared metric removes stays visible.
    """
    command = torch.as_tensor(phases, dtype=model.dtype, device=model.device)
    if command.numel() == 0:
        empty = {"l1": None, "ncc": None, "gain": None, "l1_raw": None, "ncc_raw": None}
        return dict(empty)
    with torch.no_grad():
        predicted = model.forward(command if delta is None else command + delta)
        truth = measured.forward(phases)
        predicted = window.crop(predicted)
        truth = window.crop(truth)
        l1, ncc, gain = _calibrate_matched(predicted, truth)
    pred = predicted.detach().cpu().numpy().astype(np.float64)[:, 0]
    real = truth.detach().cpu().numpy().astype(np.float64)[:, 0]
    raw_l1 = [_frame_l1(_normalise(p), _normalise(t)) for p, t in zip(pred, real)]
    raw_ncc = [_frame_ncc(_normalise(p), _normalise(t)) for p, t in zip(pred, real)]
    return {
        "l1": _mean_or_none(l1.detach().cpu().numpy()),
        "ncc": _mean_or_none(ncc.detach().cpu().numpy()),
        "gain": _mean_or_none(gain.detach().cpu().numpy()),
        "l1_raw": _mean_or_none(raw_l1),
        "ncc_raw": _mean_or_none(raw_ncc),
    }


def _session_window_px(root: Path, fallback: int) -> int:
    """Window side recorded in ``session.json``, else the camera side."""
    path = Path(root) / "session.json"
    if not path.is_file():
        return int(fallback)
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("window_px")
    if value is None:
        value = dict(payload.get("bench") or {}).get("ccd_px")
    return int(value) if value is not None else int(fallback)


def _normalise_modes(modes: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(modes, str):
        requested = [part.strip() for part in modes.split(",") if part.strip()]
    else:
        requested = [str(part).strip() for part in modes]
    if requested == ["all"]:
        return ARM_MODES
    unknown = [name for name in requested if name not in ARM_MODES]
    if unknown:
        raise ValueError(f"unknown mode(s) {unknown}; choose from {ARM_MODES} or 'all'")
    if not requested:
        raise ValueError("at least one mode is required")
    return tuple(dict.fromkeys(requested))


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy values and non-finite floats for strict JSON."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def run_fit(
    session: BenchSession,
    calibration: CalibrationResult,
    *,
    epochs: int = 200,
    lr: float = 0.03,
    seed: int = 0,
    modes: Sequence[str] | str = ARM_MODES,
    window_px: int | None = None,
    n_modes: int = 20,
    grid: int | None = None,
    device: str | torch.device = "auto",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit every requested arm and return ``(report, npz_arrays)``.

    The split is the calibration's fixed split; only train indices reach
    ``fit_residual``.  Holdout is scored afterwards with the frozen residual.

    Raises:
        ValueError: On an inconsistent grid/camera/window, or bad arguments.
    """
    mode_list = _normalise_modes(modes)
    if int(epochs) < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if int(n_modes) < 4:
        raise ValueError(f"n_modes must be >= 4, got {n_modes}")

    started = time.perf_counter()
    sim = Sim2f(calibration.sim_config, device=device)
    if grid is not None and int(grid) != int(sim.grid):
        raise ValueError(
            f"--grid={grid} conflicts with the calibration grid {sim.grid}; "
            "the calibration maps are bound to that grid"
        )
    bench = CalibratedBench(sim, calibration)
    model = ResidualModel(bench, sim)
    measured = MeasuredBench.from_session(session, device=sim.device, dtype=sim.dtype)
    if measured.grid != int(sim.grid) or measured.ccd_px != int(sim.ccd_px):
        raise ValueError(
            "recorded session geometry does not match the calibration: "
            f"session grid/ccd=({measured.grid}, {measured.ccd_px}), "
            f"calibration=({sim.grid}, {sim.ccd_px})"
        )

    if window_px is None:
        window_px = _session_window_px(session.root, int(sim.ccd_px))
    window_px = int(window_px)
    if window_px < 1 or window_px > int(sim.ccd_px):
        raise ValueError(f"window_px must be in [1, {sim.ccd_px}], got {window_px}")
    window = CameraWindow.centered(window_px, sim.ccd_px)

    phases = np.asarray(session.phases(), dtype=np.float32)
    count = int(phases.shape[0])
    train, holdout = split_indices(count)
    if train.size == 0:
        raise ValueError(f"session has {count} samples; nothing left to train on")
    train_phases = torch.as_tensor(phases[train], dtype=sim.dtype, device=sim.device)

    # The recording repeats some commands (one group is 50 rows), so the
    # row-wise split can put an identical command in both sets: a holdout frame
    # whose twin was fitted is not a generalisation test.  This subset excludes
    # exactly those rows and is reported beside the frozen split, never instead
    # of it (the calibration's own holdout has the same contamination).
    keys = measured.command_keys
    train_keys = {keys[int(index)] for index in train}
    leak_free = np.array(
        [int(index) for index in holdout if keys[int(index)] not in train_keys],
        dtype=np.int64,
    )
    leaks = int(holdout.size - leak_free.size)

    arms: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "train_indices": train.astype(np.int64),
        "holdout_indices": holdout.astype(np.int64),
        "leak_free_holdout_indices": leak_free,
    }
    for arm in mode_list:
        physics = arm == "pinn"
        fit = fit_residual(
            measured,
            model,
            train_phases,
            window=window,
            epochs=int(epochs),
            lr=float(lr),
            weights=_arm_weights(arm),
            seed=int(seed),
            physics=physics,
            aberration=None,
            illumination=model.nominal_illumination,
            exposures=None,
            n_modes=int(n_modes),
        )
        delta = torch.as_tensor(fit.delta, dtype=sim.dtype, device=sim.device)
        start = {
            "train": _score(model, measured, window, phases[train], None),
            "holdout": _score(model, measured, window, phases[holdout], None),
            "leak_free_holdout": _score(
                model, measured, window, phases[leak_free], None
            ),
        }
        end = {
            "train": _score(model, measured, window, phases[train], delta),
            "holdout": _score(model, measured, window, phases[holdout], delta),
            "leak_free_holdout": _score(
                model, measured, window, phases[leak_free], delta
            ),
        }
        arms[arm] = {
            "physics": physics,
            "weights": dict(fit.weights),
            "history": fit.history,
            "fit_l1_start": fit.l1_start,
            "fit_l1_end": fit.l1_end,
            "start": start,
            "end": end,
            "delta_rms_rad": float(np.sqrt(np.mean(np.square(fit.delta)))),
            "delta_max_abs_rad": float(np.max(np.abs(fit.delta))),
            "coefficients": int(np.size(fit.coeffs)),
            "ok": bool(fit.ok),
            "reason": fit.reason,
        }
        arrays[f"{arm}_delta"] = np.asarray(fit.delta, dtype=np.float32)
        arrays[f"{arm}_coeffs"] = np.asarray(fit.coeffs, dtype=np.float64)

    report: dict[str, Any] = {
        "schema": "sim2f_physical_residual/1",
        "evidence_status": EVIDENCE_STATUS,
        "acceptance_status": ACCEPTANCE_STATUS,
        "evidence_note": (
            "Offline fit against recorded calibration frames; model numbers are "
            "not CCD acceptance and no hardware was commanded."
        ),
        "shaping": {
            "used": False,
            "groups": list(SESSION_GROUPS),
            "note": (
                "Only already-recorded flat/net calibration frames were read. "
                "No closed-loop shaping, no new commands, no phase updates."
            ),
        },
        "hardware_commands": [],
        "exposure": {
            "supported": False,
            "term_weight": 0.0,
            "note": (
                "The recording has a single exposure; no exposure pairs exist, "
                "so the detector-linearity term is absent from both arms."
            ),
        },
        "base_state": {
            "aberration": (
                "CalibratedBench static map: nominal bench state + "
                "calibration.aberration"
            ),
            "illumination": "calibration.illumination (absolute fitted beam)",
            "additional_aberration_passed": False,
            "additional_illumination_passed": False,
        },
        "geometry": {
            "grid": int(sim.grid),
            "ccd_px": int(sim.ccd_px),
            "window_px": window_px,
            "device": str(sim.device),
            "dtype": str(sim.dtype),
        },
        "scoring": {
            "metric": "prepare_frame + closed-form gain projection",
            "reference": "ml.sim2f.calibrate._scores(fit_gain=True)",
            "fields": {
                "l1": "mean relative L1 of the gain-matched prepared frames",
                "ncc": "mean zero-mean NCC of the prepared frames",
                "gain": "mean projected per-frame brightness gain",
                "l1_raw": "relative L1 of the unit-sum frames as stored (pedestal kept)",
                "ncc_raw": "NCC of the unit-sum frames as stored (pedestal kept)",
            },
            "note": (
                "Only l1/ncc/gain are comparable with calibrate_summary.json; the "
                "*_raw fields keep the camera pedestal in view."
            ),
        },
        "split": {
            "rule": (
                "rng=default_rng(0).permutation(N); n_holdout=round(0.25*N); "
                "holdout=sort(order[:n_holdout]); train=sort(order[n_holdout:])"
            ),
            "n": count,
            "n_train": int(train.size),
            "n_holdout": int(holdout.size),
            "train": train.tolist(),
            "holdout": holdout.tolist(),
        },
        "duplicates": {
            "n_rows": measured.frame_count,
            "n_unique_commands": measured.unique_count,
            "n_rows_collapsed": measured.frame_count - measured.unique_count,
            "group_sizes": measured.duplicate_group_sizes,
            "lookup": "first recorded frame per exact command phase",
            "holdout_rows_with_train_duplicate": leaks,
            "n_leak_free_holdout": int(leak_free.size),
            "note": (
                "leak_free_holdout excludes holdout rows whose exact command also "
                "appears in train; it is a diagnostic, the frozen split stays primary"
            ),
        },
        "fit": {
            "epochs": int(epochs),
            "lr": float(lr),
            "seed": int(seed),
            "n_modes": int(n_modes),
            "modes": list(mode_list),
        },
        "arms": arms,
        "runtime_s": float(time.perf_counter() - started),
        "torch_version": torch.__version__,
    }
    return report, arrays


def default_output(session_root: str | Path) -> Path:
    """A fresh timestamped output directory for a default run."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = Path(session_root).name or "session"
    base = ROOT / "data" / "debug" / "sim2f_physical_residual" / f"{name}_{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = base.with_name(f"{base.name}_{suffix}")
    return candidate


def write_artifacts(
    out: str | Path,
    report: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> tuple[Path, Path]:
    """Write ``physical_residual.json`` and ``physical_residual.npz``.

    The output directory must not exist: this experiment never overwrites a
    previous run.

    Raises:
        FileExistsError: If ``out`` already exists.
    """
    target = Path(out)
    target.mkdir(parents=True, exist_ok=False)
    json_path = target / "physical_residual.json"
    npz_path = target / "physical_residual.npz"
    payload = json.dumps(_jsonable(dict(report)), indent=2, allow_nan=False)
    json_path.write_text(payload + "\n", encoding="utf-8")
    np.savez_compressed(npz_path, **dict(arrays))
    return json_path, npz_path


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="recorded physical session (holds session.json + process_data/)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="calibration directory (calibration.json + calibration.npz)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="new output directory; defaults to a fresh timestamped path",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--modes",
        default="data_only,pinn",
        help="comma-separated data_only,pinn or 'all'",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=None,
        help="optional check; must equal the calibration grid",
    )
    parser.add_argument(
        "--window-px",
        type=int,
        default=None,
        help="central window to score/fit; default from session.json",
    )
    parser.add_argument("--n-modes", type=int, default=20, help="highest Noll mode")
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device for the calibrated model ('auto' follows Sim2f)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI: load the recording + calibration, fit, and write new artifacts."""
    args = _parse(argv)
    session = load_bench_session(args.session, groups=SESSION_GROUPS)
    calibration = load_calibration(args.calibration)
    out = Path(args.out) if args.out is not None else default_output(args.session)
    report, arrays = run_fit(
        session,
        calibration,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        modes=args.modes,
        window_px=args.window_px,
        n_modes=args.n_modes,
        grid=args.grid,
        device=args.device,
    )
    report["inputs"] = {
        "session": str(Path(args.session)),
        "calibration": str(Path(args.calibration)),
        "groups": list(SESSION_GROUPS),
    }
    report["argv"] = list(argv if argv is not None else sys.argv[1:])
    json_path, npz_path = write_artifacts(out, report, arrays)
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")
    for arm, entry in report["arms"].items():
        print(
            f"{arm}: train L1 {entry['start']['train']['l1']} -> "
            f"{entry['end']['train']['l1']}; holdout L1 "
            f"{entry['start']['holdout']['l1']} -> "
            f"{entry['end']['holdout']['l1']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
