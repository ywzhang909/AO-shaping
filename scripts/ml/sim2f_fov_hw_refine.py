"""Resumable measured-acceptance phase refinement for a sim2f FOV session.

The source session supplies the completed plan, native zero-order metadata, and
an already commanded starting phase.  Its fitted calibration is frozen: this
script performs bounded :class:`PhaseRefiner` updates, never a probe or refit.
Every candidate is read as a native full-sensor frame and cropped offline.  A
candidate replaces the accepted phase only when the measured objective improves
strictly by more than ``--min-gain``; model metrics are recorded separately and
are never presented as CCD success.

``--dry-run`` uses ``SimInstrument`` and the nominal twin instead of loading the
fitted calibration.  It exercises the same artifacts, capture, acceptance, and
resume logic, but is only a plumbing check.  Physical outputs must be children
of ``data/debug``; dry runs may use a temporary directory.

Example::

    python scripts/ml/sim2f_fov_hw_refine.py \
      --source data/debug/sim2f_fov/session \
      --calibration data/debug/sim2f_fov/calibration \
      --out data/debug/sim2f_fov/refine_ring --target ring --rounds 4 \
      --exposure-ms 0.4

A drifted zero order is routine on site: with ``--zero-order-tolerance-px N``
(default 0) a live zero order within ``N`` pixels (Chebyshev) of the recorded
one is adopted instead of refusing the run, and the reference frame is then
recaptured at the live alignment, so a frame cropped at the old zero order is
never compared with one cropped at the new one.  ``--min-window-peak-count``
(default 0, no guard) fails any capture whose peak inside the analysis window is
below that many counts, which keeps a shape frame at the noise floor out of the
acceptance record.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.sim2f.bench import HwInstrument, SimInstrument
from ml.sim2f.calibrate import CalibratedBench, load_calibration
from ml.sim2f.config import RefineConfig, Sim2fConfig
from ml.sim2f.finetune import PhaseRefiner
from ml.sim2f.losses import objective_terms, to_phasor
from ml.sim2f.optics import Sim2f
from ml.sim2f.window import CameraWindow, WindowedBench

SUMMARY_NAME = "refine_summary.json"
SUMMARY_SCHEMA = "sim2f_fov_hw_refine_v1"
TARGETS = ("ring", "cross", "frame", "two_spot", "two_spot_wide")
NO_REFIT = (
    "The fitted calibration is frozen; no probe or refit is performed. "
    "Model metrics are predictions, never CCD acceptance."
)
DRY_RUN_WARNING = (
    "Dry-run plumbing only: SimInstrument and the nominal twin were used; "
    "the fitted calibration was not loaded and this is not real CCD success."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    data = np.ascontiguousarray(value, dtype=np.float32)
    return hashlib.sha256(data.tobytes()).hexdigest()


def _npy_sha256(value: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _load_npy(path: Path, shape: tuple[int, ...] | None = None) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{path}: expected shape {shape}, got {value.shape}")
    numeric = np.issubdtype(value.dtype, np.number) or np.issubdtype(
        value.dtype, np.bool_
    )
    if not numeric or not np.isfinite(value).all():
        raise ValueError(f"{path}: expected a finite numeric or boolean array")
    return value


def _ensure_array(path: Path, value: np.ndarray) -> None:
    if path.exists():
        if not np.array_equal(_load_npy(path), value):
            raise RuntimeError(f"refusing to overwrite different existing array: {path}")
        return
    _atomic_npy(path, value)


def _under(directory: Path, relative: str) -> Path:
    candidate = (directory / relative).resolve()
    try:
        candidate.relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes session {directory}: {relative!r}") from exc
    return candidate


def _fingerprint(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _target_mask(kind: str, ccd_px: int) -> tuple[np.ndarray, np.ndarray]:
    """The same unit-sum target and support as the source session."""
    yy, xx = np.mgrid[0:ccd_px, 0:ccd_px].astype(np.float64)
    cx = cy = (ccd_px - 1) / 2.0
    radius = np.hypot(xx - cx, yy - cy)
    target = np.zeros((ccd_px, ccd_px), dtype=np.float64)
    if kind == "ring":
        target[(radius > 0.30 * ccd_px) & (radius < 0.42 * ccd_px)] = 1.0
    elif kind == "cross":
        target[(np.abs(xx - cx) < 0.06 * ccd_px) & (radius < 0.40 * ccd_px)] = 1.0
        target[(np.abs(yy - cy) < 0.06 * ccd_px) & (radius < 0.40 * ccd_px)] = 1.0
    elif kind == "frame":
        square = np.maximum(np.abs(xx - cx), np.abs(yy - cy))
        target[(square > 0.26 * ccd_px) & (square < 0.36 * ccd_px)] = 1.0
    elif kind in ("two_spot", "two_spot_wide"):
        separation = 0.42 * ccd_px if kind == "two_spot" else 0.46 * ccd_px
        for spot_x in (cx - separation, cx + separation):
            target[np.hypot(xx - spot_x, yy - cy) < 0.07 * ccd_px] = 1.0
    else:
        raise ValueError(f"unknown target {kind!r}; choose from {TARGETS}")
    if target.sum() <= 0.0:
        raise ValueError(f"target {kind!r} is empty at ccd_px={ccd_px}")
    support = target > 0.0
    target = target / target.sum()
    return target.astype(np.float32), support


def _native_layout(
    native_shape: tuple[int, int], zero_order_xy: tuple[int, int], window_px: int
) -> dict[str, Any]:
    """Map a native frame to the model's square, then to the analysis window."""
    height, width = native_shape
    full_px = min(native_shape)
    half = full_px // 2
    col0 = max(0, min(zero_order_xy[0] - half, width - full_px))
    row0 = max(0, min(zero_order_xy[1] - half, height - full_px))
    square_center = (zero_order_xy[0] - col0, zero_order_xy[1] - row0)
    window = CameraWindow(
        center=(float(square_center[0]), float(square_center[1])),
        size=window_px,
        frame=(full_px, full_px),
    )
    return {
        "full_px": full_px,
        "native_square_offset_xy": [col0, row0],
        "square_zero_order_xy": list(square_center),
        "window": window,
    }


def _dry_run_layout(
    native_shape: tuple[int, int], zero_order_xy: tuple[int, int], window_px: int
) -> dict[str, Any]:
    """Nominal-twin layout on a physical canvas.

    The twin has no calibrated camera shift: its zero order is the centre of its
    square read.  A physical source can record the zero order off that centre.
    A nominal dry run therefore keeps the physical canvas and offline crop but
    places the twin zero at the model centre; using the physical offset here
    would compare a shifted prediction with an unshifted twin reference.
    """
    base = _native_layout(native_shape, zero_order_xy, window_px)
    full_px = int(base["full_px"])
    col0, row0 = (int(value) for value in base["native_square_offset_xy"])
    half = full_px // 2
    center = (half, half)
    return {
        **base,
        "effective_zero_order_xy": [col0 + half, row0 + half],
        "square_zero_order_xy": list(center),
        "window": CameraWindow(
            center=(float(center[0]), float(center[1])),
            size=window_px,
            frame=(full_px, full_px),
        ),
    }


def _crop_native(frame: np.ndarray, layout: dict[str, Any]) -> np.ndarray:
    col0, row0 = layout["native_square_offset_xy"]
    full_px = int(layout["full_px"])
    square = frame[row0 : row0 + full_px, col0 : col0 + full_px]
    return np.asarray(layout["window"].crop(square))


def _score(
    phase: np.ndarray,
    frame: np.ndarray,
    target: torch.Tensor,
    support: torch.Tensor,
    cfg: RefineConfig,
) -> dict[str, float]:
    """One objective implementation for model predictions and measured crops."""
    x = to_phasor(torch.as_tensor(phase, dtype=target.dtype)[None])
    image = torch.as_tensor(frame, dtype=target.dtype)[None, None]
    with torch.no_grad():
        loss, l1, ee, ncc = objective_terms(
            image, target, support, w_ee=cfg.w_ee, metric=cfg.metric, prepare=cfg.prepare
        )
        tv = (
            (x[:, :, 1:] - x[:, :, :-1]).abs().sum()
            + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().sum()
        )
        total = loss + cfg.w_tv * tv
    result = {
        "objective": float(total),
        "l1": float(l1),
        "ee": float(ee),
        "ncc": float(ncc),
        "tv": float(tv),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError(f"non-finite objective metrics: {result}")
    return result


def _predict(model: WindowedBench, phase: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        frame = model.forward(torch.as_tensor(phase, dtype=model.dtype)[None], detector=False)
    return frame[0, 0].detach().cpu().numpy()


def _validate_output(out: Path, dry_run: bool) -> Path:
    resolved = out.expanduser().resolve()
    if dry_run:
        return resolved
    root = (ROOT / "data" / "debug").resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"physical --out must be under {root}; use --dry-run for temporary outputs"
        ) from exc
    if not relative.parts:
        raise ValueError(f"physical --out must be a child of {root}, not the root itself")
    return resolved


def _load_source(source: Path, start_phase: str | None, target: str) -> dict[str, Any]:
    source = source.resolve()
    plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((source / "capture_manifest.json").read_text(encoding="utf-8"))
    if plan.get("schema") != "sim2f_fov_hw_session/plan/v1":
        raise ValueError("source is not a sim2f_fov_hw_session plan")
    if manifest.get("schema") != "sim2f_fov_hw_session/manifest/v1":
        raise ValueError("source is not a sim2f_fov_hw_session manifest")
    if manifest.get("plan_sha256") != _sha256(source / "plan.json"):
        raise ValueError("source capture manifest belongs to a different plan")
    native = tuple(int(value) for value in manifest["native_shape"])
    if len(native) != 2 or min(native) <= 0:
        raise ValueError(f"invalid source native shape: {native}")
    zero_order = tuple(int(value) for value in manifest["zero_order_xy"])
    by_id = manifest["items"]
    plan_ids = {item["id"] for item in plan["items"]}
    if set(by_id) != plan_ids:
        raise ValueError("source session is not complete: plan/manifest item IDs differ")
    plan_by_id = {item["id"]: item for item in plan["items"]}
    for item_id, item in by_id.items():
        if item.get("status") != "done":
            raise ValueError(f"source session is incomplete at {item_id}")
        frame = _under(source, item["frame"])
        panel = _under(source, item["panel"])
        phase_path = _under(source, plan_by_id[item_id]["phase"])
        valid = (
            frame.is_file()
            and panel.is_file()
            and phase_path.is_file()
            and _array_sha256(_load_npy(frame, native)) == item["frame_sha256"]
            and _array_sha256(_load_npy(panel)) == item["panel_sha256"]
            and _array_sha256(_load_npy(phase_path))
            == plan_by_id[item_id]["phase_sha256"]
        )
        if not valid:
            raise ValueError(f"source session is incomplete or corrupt at {item_id}")
    start_name = start_phase or f"phases/shape_{target}_fitted_r0.npy"
    start_path = _under(source, start_name)
    matches = [item for item in plan["items"] if _under(source, item["phase"]) == start_path]
    if len(matches) != 1:
        raise ValueError(f"starting phase is not a planned source item: {start_name}")
    plan_item = matches[0]
    if plan_item.get("kind") != "shape" or plan_item.get("target") != target:
        raise ValueError("starting phase must be a shape item for the requested target")
    source_item = {"id": plan_item["id"], **by_id[plan_item["id"]]}
    grid = int(plan["grid"])
    phase = _load_npy(start_path, (grid, grid)).astype(np.float32, copy=False)
    if _array_sha256(phase) != plan_item["phase_sha256"]:
        raise ValueError("starting phase is missing or its hash does not match the plan")
    exposure = source_item.get("exposure_ms", manifest.get("exposure_ms"))
    if exposure is not None and float(exposure) <= 0.0:
        raise ValueError("source starting item has a non-positive recorded exposure")
    return {
        "source": source,
        "plan": plan,
        "manifest": manifest,
        "plan_sha256": _sha256(source / "plan.json"),
        "manifest_sha256": _sha256(source / "capture_manifest.json"),
        "native_shape": native,
        "zero_order_xy": zero_order,
        "phase": phase,
        "phase_path": start_path,
        "phase_name": start_name,
        "phase_sha256": _sha256(start_path),
        "source_item": source_item,
        "source_exposure_ms": None if exposure is None else float(exposure),
    }


class _CanvasSimInstrument:
    """Place a square twin read into a physical sensor canvas, zero-order aligned.

    The twin has no sensor: its read is already a square ``ccd_px`` window.  A
    hardware session can be wider and its zero order need not be centred.  This
    adapter keeps the dry run on the physical session's native geometry without
    pretending the twin measured those outer pixels: it embeds one twin read at
    the recorded zero-order position and leaves the rest dark.
    """

    def __init__(
        self,
        inner: Any,
        native_shape: tuple[int, int],
        zero_order_xy: tuple[int, int],
    ) -> None:
        self.inner = inner
        self._native_shape = (int(native_shape[0]), int(native_shape[1]))
        self._zero_order_xy = (int(zero_order_xy[0]), int(zero_order_xy[1]))
        inner_native = tuple(int(value) for value in inner.native_shape)
        inner_zero = tuple(int(value) for value in inner.zero_order_xy)
        if len(inner_native) != 2 or len(inner_zero) != 2:
            raise ValueError(f"invalid twin geometry: {inner_native}, {inner_zero}")
        self._inner_native = inner_native
        offset_x = self._zero_order_xy[0] - inner_zero[0]
        offset_y = self._zero_order_xy[1] - inner_zero[1]
        self._source_y = max(0, -offset_y)
        self._source_x = max(0, -offset_x)
        self._target_y = max(0, offset_y)
        self._target_x = max(0, offset_x)
        height = min(
            inner_native[0] - self._source_y,
            self._native_shape[0] - self._target_y,
        )
        width = min(
            inner_native[1] - self._source_x,
            self._native_shape[1] - self._target_x,
        )
        if height <= 0 or width <= 0:
            raise ValueError(
                "the twin read does not overlap the physical canvas at the recorded zero order"
            )
        self._height = height
        self._width = width

    @property
    def native_shape(self) -> tuple[int, int]:
        return self._native_shape

    @property
    def zero_order_xy(self) -> tuple[int, int]:
        return self._zero_order_xy

    def enable_full_frame(self) -> tuple[int, int]:
        self.inner.enable_full_frame()
        return self._native_shape

    def capture_full_frame(self, phase: np.ndarray) -> np.ndarray:
        frame = np.asarray(self.inner.capture_full_frame(phase))
        if frame.shape != self._inner_native:
            raise ValueError(
                f"twin read shape {frame.shape} differs from {self._inner_native}"
            )
        canvas = np.zeros(self._native_shape, dtype=frame.dtype)
        canvas[
            self._target_y : self._target_y + self._height,
            self._target_x : self._target_x + self._width,
        ] = frame[
            self._source_y : self._source_y + self._height,
            self._source_x : self._source_x + self._width,
        ]
        return canvas

    def describe(self) -> dict[str, Any]:
        info = self.inner.describe()
        if not isinstance(info, dict):
            return info
        return {
            **info,
            "dry_run_canvas": {
                "native_shape": list(self._native_shape),
                "zero_order_xy": list(self._zero_order_xy),
                "twin_native_shape": list(self._inner_native),
            },
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class _LazyInstrument:
    """Open only when a capture is needed, and always close an opened device."""

    def __init__(
        self,
        *,
        dry_run: bool,
        full_cfg: Sim2fConfig,
        native_shape: tuple[int, int],
        zero_order_xy: tuple[int, int],
        exposure_ms: float,
        settle_s: float,
        cam_type: str,
        cam_id: int,
        window_px: int,
        panel_shape: tuple[int, int] | None,
        pupil_center_xy: tuple[float, float] | None,
        zero_order_tolerance_px: int = 0,
    ) -> None:
        self.dry_run = dry_run
        self.full_cfg = full_cfg
        self.expected_native = native_shape
        self.expected_zero_order = zero_order_xy
        self.zero_order_tolerance_px = int(zero_order_tolerance_px)
        self.zero_order_drift_px = 0
        self.adopted_zero_order = False
        self.exposure_ms = exposure_ms
        self.settle_s = settle_s
        self.cam_type = cam_type
        self.cam_id = cam_id
        self.window_px = window_px
        self.panel_shape = panel_shape
        self.pupil_center_xy = pupil_center_xy
        self.instrument: Any | None = None

    def open(self) -> Any:
        if self.instrument is not None:
            return self.instrument
        instrument = (
            SimInstrument(bench=Sim2f(self.full_cfg, device="cpu"), detector=False)
            if self.dry_run
            else HwInstrument(
                settle_s=self.settle_s,
                exposure_ms=self.exposure_ms,
                ccd_px=int(self.window_px),
                cam_type=self.cam_type,
                cam_id=self.cam_id,
                panel_shape=self.panel_shape,
                pupil_center_xy=self.pupil_center_xy,
                device="cpu",
            )
        )
        if self.dry_run and (
            tuple(instrument.native_shape) != self.expected_native
            or tuple(instrument.zero_order_xy) != self.expected_zero_order
        ):
            instrument = _CanvasSimInstrument(
                instrument, self.expected_native, self.expected_zero_order
            )
        self.instrument = instrument
        try:
            if not self.dry_run:
                applied = float(instrument.exposure_ms)
                if abs(applied - self.exposure_ms) > 1e-9:
                    raise RuntimeError(
                        "camera reports "
                        f"{applied:.6f} ms after the session requested "
                        f"{self.exposure_ms:.6f} ms; refusing to score a different exposure"
                    )
            native = tuple(int(value) for value in instrument.enable_full_frame())
            zero_order = tuple(int(value) for value in instrument.zero_order_xy)
            mismatch = (
                "opened instrument metadata differs from the source session: "
                f"native {native} vs {self.expected_native}, zero order "
                f"{zero_order} vs {self.expected_zero_order}"
            )
            if native != self.expected_native:
                raise RuntimeError(mismatch)
            if zero_order != self.expected_zero_order:
                drift = max(
                    abs(zero_order[0] - self.expected_zero_order[0]),
                    abs(zero_order[1] - self.expected_zero_order[1]),
                )
                adopted = (
                    not self.dry_run
                    and self.zero_order_tolerance_px > 0
                    and drift <= self.zero_order_tolerance_px
                )
                if not adopted:
                    raise RuntimeError(mismatch)
                # A drifted zero order shifts every crop, so the live alignment
                # wins and the recorded reference has to be recaptured: scoring
                # a source frame cropped at the old zero order against a
                # candidate cropped at the new one would compare two geometries.
                self.expected_zero_order = zero_order
                self.zero_order_drift_px = int(drift)
                self.adopted_zero_order = True
            return instrument
        except Exception:
            self.instrument = None
            instrument.close()
            raise

    def close(self) -> None:
        if self.instrument is not None:
            instrument = self.instrument
            self.instrument = None
            instrument.close()


def _capture_artifact(
    lazy: _LazyInstrument,
    *,
    out: Path,
    stem: str,
    phase: np.ndarray,
    exposure_ms: float,
    layout: dict[str, Any],
    min_window_peak_count: float = 0.0,
) -> dict[str, Any]:
    instrument = lazy.open()
    started = _utc_now()
    phase_path = out / "phases" / f"{stem}.npy"
    gray_path = out / "panel_gray" / f"{stem}.npy"
    frame_path = out / "frames" / f"{stem}.npy"
    _ensure_array(phase_path, phase)
    gray = instrument.sent_grayscale(phase)
    gray_relative: str | None = None
    gray_hash: str | None = None
    if gray is not None:
        _atomic_npy(gray_path, np.asarray(gray))
        gray_relative = gray_path.relative_to(out).as_posix()
        gray_hash = _sha256(gray_path)
    frame = np.asarray(instrument.capture_full_frame(phase))
    _atomic_npy(frame_path, frame)
    _load_npy(frame_path, lazy.expected_native)
    native_frame = np.asarray(frame, dtype=np.float64)
    frame_max = float(np.max(native_frame))
    window_peak = float(np.max(_crop_native(native_frame, layout)))
    if min_window_peak_count > 0.0 and window_peak < min_window_peak_count:
        raise RuntimeError(
            f"capture {stem!r} peaks at {window_peak:.6g} counts inside the "
            "analysis window, below the required "
            f"{min_window_peak_count:.6g} counts (--min-window-peak-count); the "
            "frame is at the noise floor, so no measured gain can be trusted"
        )
    return {
        "phase": phase_path.relative_to(out).as_posix(),
        "phase_sha256": _sha256(phase_path),
        "panel_gray": gray_relative,
        "panel_gray_sha256": gray_hash,
        "native_frame": frame_path.relative_to(out).as_posix(),
        "native_frame_sha256": _sha256(frame_path),
        "native_shape": list(lazy.expected_native),
        "zero_order_xy": list(lazy.expected_zero_order),
        "exposure_ms": float(exposure_ms),
        "capture": instrument.describe(),
        "frame_max": frame_max,
        "window_peak": window_peak,
        "timestamps": {"started_utc": started, "completed_utc": _utc_now()},
        "capture_method": "sequential_native_full_frame",
    }


def _check_record_metadata(
    record: dict[str, Any],
    *,
    out: Path,
    expected_phase_hash: str,
    native_shape: tuple[int, int],
    zero_order_xy: tuple[int, int],
    exposure_ms: float,
) -> None:
    if record.get("phase_sha256") != expected_phase_hash:
        raise RuntimeError("resume phase hash differs from the deterministic candidate")
    phase_path = _under(out, str(record.get("phase", "")))
    if not phase_path.is_file() or _sha256(phase_path) != expected_phase_hash:
        raise RuntimeError(f"resume phase is missing or modified: {phase_path}")
    if tuple(record.get("native_shape", ())) != native_shape:
        raise RuntimeError("resume record native shape differs from the source")
    if tuple(record.get("zero_order_xy", ())) != zero_order_xy:
        raise RuntimeError("resume record zero order differs from the source")
    if float(record.get("exposure_ms", -1.0)) != exposure_ms:
        raise RuntimeError("resume record exposure differs from this run")


def _valid_record(
    record: dict[str, Any],
    *,
    out: Path,
    expected_phase_hash: str,
    native_shape: tuple[int, int],
    zero_order_xy: tuple[int, int],
    exposure_ms: float,
) -> bool:
    try:
        _check_record_metadata(
            record,
            out=out,
            expected_phase_hash=expected_phase_hash,
            native_shape=native_shape,
            zero_order_xy=zero_order_xy,
            exposure_ms=exposure_ms,
        )
        frame_path = _under(out, record["native_frame"])
        if _sha256(frame_path) != record["native_frame_sha256"]:
            return False
        _load_npy(frame_path, native_shape)
        if record.get("panel_gray"):
            gray_path = _under(out, record["panel_gray"])
            if _sha256(gray_path) != record["panel_gray_sha256"]:
                return False
            _load_npy(gray_path)
        def finite_metrics(value: Any) -> bool:
            if isinstance(value, dict):
                return bool(value) and all(finite_metrics(item) for item in value.values())
            return bool(np.isfinite(float(value)))

        return bool(
            finite_metrics(record.get("measured"))
            and finite_metrics(record.get("predicted"))
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _reference_from_source(
    source: dict[str, Any], out: Path, exposure_ms: float
) -> dict[str, Any]:
    item = source["source_item"]
    phase_path = out / "phases" / "reference.npy"
    frame_path = out / "frames" / "reference.npy"
    _ensure_array(phase_path, source["phase"])
    _atomic_copy(_under(source["source"], item["frame"]), frame_path)
    gray_relative = item.get("panel")
    gray_hash: str | None = None
    if gray_relative:
        gray_path = out / "panel_gray" / "reference.npy"
        _atomic_copy(_under(source["source"], gray_relative), gray_path)
        gray_hash = _sha256(gray_path)
        gray_relative = gray_path.relative_to(out).as_posix()
    return {
        "id": "reference",
        "phase": phase_path.relative_to(out).as_posix(),
        "phase_sha256": _sha256(phase_path),
        "panel_gray": gray_relative,
        "panel_gray_sha256": gray_hash,
        "native_frame": frame_path.relative_to(out).as_posix(),
        "native_frame_sha256": _sha256(frame_path),
        "native_shape": list(source["native_shape"]),
        "zero_order_xy": list(source["zero_order_xy"]),
        "exposure_ms": float(exposure_ms),
        "capture_method": "reused_source_native_frame",
        "source_item_id": item["id"],
        "source_capture": {
            key: item.get(key)
            for key in ("captured_at", "exposure_ms", "zero_order_offset_px")
        },
        "timestamps": {"source_captured_at": item.get("captured_at")},
    }


def _save_state(
    out: Path,
    summary: dict[str, Any],
    accepted_phase: np.ndarray,
    rounds: list[dict[str, Any]],
    *,
    status: str,
) -> None:
    accepted_path = out / "accepted_phase.npy"
    _atomic_npy(accepted_path, accepted_phase)
    summary["rounds"] = rounds
    summary["accepted_phase"] = accepted_path.name
    summary["accepted_phase_sha256"] = _sha256(accepted_path)
    summary["accepted_rounds"] = sum(bool(record["accepted"]) for record in rounds)
    summary["status"] = status
    summary["updated_at_utc"] = _utc_now()
    _atomic_json(out / SUMMARY_NAME, summary)


def run_refinement(args: argparse.Namespace) -> dict[str, Any]:
    out = _validate_output(Path(args.out), bool(args.dry_run))
    source = _load_source(Path(args.source), args.start_phase, args.target)
    source_backend = source["manifest"].get("backend")
    expected_backend = "sim" if args.dry_run else "hw"
    physical_dry_run = bool(
        args.dry_run and args.recapture_reference and source_backend == "hw"
    )
    if source_backend != expected_backend and not physical_dry_run:
        raise ValueError(
            f"source backend must be {expected_backend!r} for this run"
            + (
                " (a hardware source is allowed only with --recapture-reference)"
                if args.dry_run
                else ""
            )
            + "; never mix simulated and hardware reference frames"
        )
    calibration = Path(args.calibration).expanduser().resolve()
    if not calibration.is_dir():
        raise ValueError(f"calibration directory does not exist: {calibration}")
    if out == source["source"]:
        raise ValueError("--out must differ from the source session")
    if int(args.rounds) < 1:
        raise ValueError("rounds must be >= 1")
    if args.exposure_ms is not None:
        exposure_ms = float(args.exposure_ms)
    elif source["source_exposure_ms"] is not None:
        exposure_ms = float(source["source_exposure_ms"])
    elif args.dry_run:
        exposure_ms = 0.5
    else:
        raise ValueError("source has no exposure; physical runs require --exposure-ms")
    if not np.isfinite(exposure_ms) or exposure_ms <= 0.0:
        raise ValueError("exposure must be finite and > 0")
    summary_path = out / SUMMARY_NAME
    if out.exists() and any(out.iterdir()) and not summary_path.is_file():
        raise RuntimeError(f"non-empty output has no {SUMMARY_NAME}; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    plan = source["plan"]
    window_px = int(plan["window_px"])
    source_zero = tuple(int(value) for value in source["zero_order_xy"])
    if physical_dry_run:
        layout = _dry_run_layout(source["native_shape"], source_zero, window_px)
        run_zero = tuple(int(value) for value in layout["effective_zero_order_xy"])
    else:
        layout = _native_layout(source["native_shape"], source_zero, window_px)
        run_zero = source_zero
    target_np, support_np = _target_mask(args.target, window_px)
    _ensure_array(out / "target.npy", target_np)
    _ensure_array(out / "support.npy", support_np)
    refine_cfg = RefineConfig(
        iters=int(args.iters),
        lr=float(args.lr),
        lr_final_frac=float(args.lr_final_frac),
        w_ee=float(args.w_ee),
        w_tv=float(args.w_tv),
        metric=args.metric,
        prepare=True,
        max_delta_rad=float(args.max_delta_rad),
        update_sigma_px=float(args.update_sigma_px),
        min_gain=float(args.model_min_gain),
    )
    grid = int(plan["grid"])
    full_px = int(layout["full_px"])
    full_cfg = replace(Sim2fConfig.calibrated(), grid=grid, ccd_px=full_px)
    full_model: Any = Sim2f(full_cfg, device="cpu")
    calibration_mode = "frozen_fitted_calibration"
    if args.dry_run:
        calibration_mode = "nominal_twin_dry_run_calibration_not_loaded"
    else:
        full_model = CalibratedBench(full_model, load_calibration(calibration))
    lazy = _LazyInstrument(
        dry_run=bool(args.dry_run),
        full_cfg=full_cfg,
        native_shape=source["native_shape"],
        zero_order_xy=run_zero,
        exposure_ms=exposure_ms,
        settle_s=float(args.settle_s),
        cam_type=args.cam_type,
        cam_id=int(args.cam_id),
        window_px=int(plan["window_px"]),
        panel_shape=None if args.panel_shape is None else tuple(args.panel_shape),
        pupil_center_xy=args.pupil_center_xy,
        zero_order_tolerance_px=int(args.zero_order_tolerance_px),
    )
    # Reconcile the live zero order before any crop geometry is frozen: an
    # adopted drift re-crops every native frame, so the analysis window and the
    # model have to be rebuilt at the live alignment.
    try:
        lazy.open()
    except Exception:
        lazy.close()
        raise
    if lazy.adopted_zero_order:
        run_zero = tuple(int(value) for value in lazy.expected_zero_order)
        layout = _native_layout(source["native_shape"], run_zero, window_px)
    model = WindowedBench(full_model, layout["window"])
    target = torch.as_tensor(target_np, dtype=model.dtype)[None, None]
    support = torch.as_tensor(support_np, dtype=model.dtype)[None, None]
    request = {
        "source": str(source["source"]),
        "plan_sha256": source["plan_sha256"],
        "manifest_sha256": source["manifest_sha256"],
        "starting_phase": source["phase_name"],
        "starting_phase_sha256": source["phase_sha256"],
        "calibration": str(calibration),
        "calibration_files": _fingerprint(calibration),
        "target": args.target,
        "rounds": int(args.rounds),
        "dry_run": bool(args.dry_run),
        "exposure_ms": exposure_ms,
        "recapture_reference": bool(args.recapture_reference),
        "source_zero_order_xy": list(source_zero),
        "effective_zero_order_xy": list(run_zero),
        "refine": refine_cfg.to_dict(),
        "measured_min_gain": float(args.min_gain),
    }
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema") != SUMMARY_SCHEMA or summary.get("request") != request:
            raise RuntimeError("resume metadata differs; choose a new --out instead of renumbering")
    else:
        summary = {
            "schema": SUMMARY_SCHEMA,
            "request": request,
            "created_at_utc": _utc_now(),
        }
    summary.update(
        {
            "acceptance": "sequential_ccd",
            "acceptance_rule": "model_accepted AND measured_gain > measured_min_gain",
            "measured_source": "native_full_frames_cropped_offline_no_device_roi",
            "native_shape": list(source["native_shape"]),
            "zero_order_xy": list(run_zero),
            "source_zero_order_xy": list(source_zero),
            "full_model_px": full_px,
            "native_square_offset_xy": layout["native_square_offset_xy"],
            "square_zero_order_xy": layout["square_zero_order_xy"],
            "analysis_window": {
                "center_xy": list(layout["window"].center),
                "size": layout["window"].size,
                "frame_wh": layout["window"].frame,
            },
            "target": "target.npy",
            "target_sum": float(target_np.sum()),
            "support": "support.npy",
            "calibration_mode": calibration_mode,
            "limitation": NO_REFIT,
            "dry_run_warning": DRY_RUN_WARNING if args.dry_run else None,
            "zero_order_recorded_xy": list(source_zero),
            "zero_order_effective_xy": list(run_zero),
            "zero_order_drift_px": int(lazy.zero_order_drift_px),
            "zero_order_tolerance_px": int(args.zero_order_tolerance_px),
            "zero_order_reference_recaptured": False,
        }
    )
    current_phase = source["phase"].copy()
    rounds: list[dict[str, Any]] = []
    old_rounds = list(summary.get("rounds", []))
    reference_phase_hash = _npy_sha256(current_phase)
    try:
        reference = summary.get("reference")
        reference_captured = False
        reference_valid = bool(
            reference
            and _valid_record(
                reference,
                out=out,
                expected_phase_hash=reference_phase_hash,
                native_shape=source["native_shape"],
                zero_order_xy=run_zero,
                exposure_ms=exposure_ms,
            )
        )
        can_reuse_source = (
            not args.recapture_reference
            and not lazy.adopted_zero_order
            and (
                args.exposure_ms is None
                or (
                    source["source_exposure_ms"] is not None
                    and exposure_ms == float(source["source_exposure_ms"])
                )
            )
        )
        if not reference_valid:
            if reference:
                _check_record_metadata(
                    reference,
                    out=out,
                    expected_phase_hash=reference_phase_hash,
                    native_shape=source["native_shape"],
                    zero_order_xy=run_zero,
                    exposure_ms=exposure_ms,
                )
            if can_reuse_source:
                reference = _reference_from_source(source, out, exposure_ms)
            else:
                reference = {
                    "id": "reference",
                    **_capture_artifact(
                        lazy,
                        out=out,
                        stem="reference",
                        phase=current_phase,
                        exposure_ms=exposure_ms,
                        layout=layout,
                        min_window_peak_count=float(args.min_window_peak_count),
                    ),
                }
                reference_captured = True
            reference_frame = _load_npy(
                _under(out, reference["native_frame"]), source["native_shape"]
            )
            reference["measured"] = _score(
                current_phase, _crop_native(reference_frame, layout), target, support, refine_cfg
            )
            reference["predicted"] = _score(
                current_phase, _predict(model, current_phase), target, support, refine_cfg
            )
        summary["reference"] = reference
        summary["zero_order_reference_recaptured"] = bool(
            lazy.adopted_zero_order and reference_captured
        )
        current_measured = dict(reference["measured"])
        if not old_rounds:
            _save_state(out, summary, current_phase, rounds, status="running")

        old_by_id: dict[str, dict[str, Any]] = {}
        for record in old_rounds:
            item_id = str(record.get("id", ""))
            if not item_id or item_id in old_by_id:
                raise RuntimeError("resume summary has missing or duplicate round IDs")
            old_by_id[item_id] = record
        if len(old_rounds) > int(args.rounds):
            raise RuntimeError("resume summary contains more rounds than requested")

        refiner = PhaseRefiner(model, refine_cfg)
        current_reference_id = "reference"
        for index in range(1, int(args.rounds) + 1):
            item_id = f"round_{index:04d}"
            result = refiner.finetune(current_phase, target, support=support)
            candidate = np.asarray(result.phase, dtype=np.float32)
            candidate_hash = _npy_sha256(candidate)
            existing = old_by_id.get(item_id)
            if existing is not None and int(existing.get("round", -1)) != index:
                raise RuntimeError(f"resume round number mismatch for {item_id}")
            valid = bool(
                existing
                and _valid_record(
                    existing,
                    out=out,
                    expected_phase_hash=candidate_hash,
                    native_shape=source["native_shape"],
                    zero_order_xy=run_zero,
                    exposure_ms=exposure_ms,
                )
            )
            if valid:
                assert existing is not None
                if existing["phase_sha256"] != candidate_hash:
                    raise RuntimeError(f"resume candidate differs for {item_id}")
                if bool(existing["model_accepted"]) != bool(result.accepted):
                    raise RuntimeError(f"resume model acceptance differs for {item_id}")
                measured_gain = float(current_measured["objective"]) - float(
                    existing["measured"]["candidate"]["objective"]
                )
                expected_accepted = bool(result.accepted) and (
                    measured_gain > float(args.min_gain)
                )
                if bool(existing["accepted"]) != expected_accepted:
                    raise RuntimeError(f"resume acceptance is inconsistent for {item_id}")
                record = existing
            else:
                if existing:
                    _check_record_metadata(
                        existing,
                        out=out,
                        expected_phase_hash=candidate_hash,
                        native_shape=source["native_shape"],
                        zero_order_xy=run_zero,
                        exposure_ms=exposure_ms,
                    )
                artifact = _capture_artifact(
                    lazy,
                    out=out,
                    stem=item_id,
                    phase=candidate,
                    exposure_ms=exposure_ms,
                    layout=layout,
                    min_window_peak_count=float(args.min_window_peak_count),
                )
                candidate_frame = _load_npy(
                    _under(out, artifact["native_frame"]), source["native_shape"]
                )
                measured = _score(
                    candidate, _crop_native(candidate_frame, layout), target, support, refine_cfg
                )
                predicted_reference = _score(
                    current_phase,
                    _predict(model, current_phase),
                    target,
                    support,
                    refine_cfg,
                )
                predicted_candidate = _score(
                    candidate,
                    np.asarray(result.intensity),
                    target,
                    support,
                    refine_cfg,
                )
                measured_gain = float(current_measured["objective"] - measured["objective"])
                accepted = bool(result.accepted) and measured_gain > float(args.min_gain)
                if not result.accepted:
                    reason = "model_rejected_bounded_update"
                elif accepted:
                    reason = (
                        f"measured_gain {measured_gain:.8g} > min_gain "
                        f"{float(args.min_gain):.8g}"
                    )
                else:
                    reason = (
                        f"measured_gain {measured_gain:.8g} <= min_gain "
                        f"{float(args.min_gain):.8g}"
                    )
                record = {
                    "id": item_id,
                    "round": index,
                    **artifact,
                    "reference_id": current_reference_id,
                    "reference_accepted_phase_sha256": hashlib.sha256(
                        current_phase.tobytes()
                    ).hexdigest(),
                    "predicted": {
                        "reference": predicted_reference,
                        "candidate": predicted_candidate,
                        "gain": float(
                            predicted_reference["objective"]
                            - predicted_candidate["objective"]
                        ),
                    },
                    "measured": {
                        "reference": dict(current_measured),
                        "candidate": measured,
                        "gain": measured_gain,
                    },
                    "model_accepted": bool(result.accepted),
                    "model_reason": (
                        "bounded_update_improved_model"
                        if result.accepted
                        else "model_gain_below_min_gain"
                    ),
                    "model_iterations": int(result.iters),
                    "model_wall_s": float(result.wall_s),
                    "accepted": accepted,
                    "status": "accepted" if accepted else "rejected",
                    "reason": reason,
                }
            if record["accepted"]:
                current_phase = candidate.copy()
                current_measured = dict(record["measured"]["candidate"])
                current_reference_id = item_id
            rounds.append(record)
            _save_state(out, summary, current_phase, rounds, status="running")
        _save_state(out, summary, current_phase, rounds, status="complete")
        return summary
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(summary_path, summary)
        raise
    finally:
        lazy.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed sim2f_fov_hw_session")
    parser.add_argument("--calibration", required=True, help="fitted calibration directory")
    parser.add_argument("--out", required=True, help="new/resumable output session")
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--start-phase", help="source-relative phase; default fitted r0")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--min-gain", type=float, default=1e-3, help="measured strict gain")
    parser.add_argument("--model-min-gain", type=float, default=0.0)
    parser.add_argument("--iters", type=int, default=RefineConfig.iters)
    parser.add_argument("--lr", type=float, default=RefineConfig.lr)
    parser.add_argument("--lr-final-frac", type=float, default=RefineConfig.lr_final_frac)
    parser.add_argument("--w-ee", type=float, default=RefineConfig.w_ee)
    parser.add_argument("--w-tv", type=float, default=RefineConfig.w_tv)
    parser.add_argument("--metric", choices=("l1", "mse"), default=RefineConfig.metric)
    parser.add_argument("--max-delta-rad", type=float, default=RefineConfig.max_delta_rad)
    parser.add_argument("--update-sigma-px", type=float, default=RefineConfig.update_sigma_px)
    parser.add_argument("--exposure-ms", type=float, help="fixed; default source exposure")
    parser.add_argument("--recapture-reference", action="store_true")
    parser.add_argument(
        "--zero-order-tolerance-px",
        type=int,
        default=0,
        help=(
            "adopt a live zero order within this many pixels (Chebyshev) of the "
            "recorded one and recapture the reference; 0 refuses any drift"
        ),
    )
    parser.add_argument(
        "--min-window-peak-count",
        type=float,
        default=0.0,
        help=(
            "fail a capture whose peak inside the analysis window is below this "
            "many counts; 0 disables the guard"
        ),
    )
    parser.add_argument("--settle-s", type=float, default=0.05)
    parser.add_argument("--cam-type", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--panel-shape", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument(
        "--pupil-center-xy", type=float, nargs=2, metavar=("X", "Y"), default=None
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if int(args.zero_order_tolerance_px) < 0:
        parser.error("--zero-order-tolerance-px must be >= 0")
    try:
        _validate_output(Path(args.out), bool(args.dry_run))
        summary = run_refinement(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve()),
                "status": summary["status"],
                "accepted_rounds": summary["accepted_rounds"],
                "acceptance": summary["acceptance"],
            },
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    main()
