"""Recorder-based process records for transfer shaping runs.

A transfer run directory is already the source of truth: every CCD frame, every
phase and every block decision is on disk. This module re-reads them and turns
a run into :class:`~ao_shaping.utils.io.file.Recorder` records -- one per CCD
block -- so the process can be stored as a table (CSV + HDF5) and rendered as an
animation **without re-running anything and without opening a device**.

The record vocabulary follows the project's single source of truth
(``utils.io.recorder_keys``): the phase goes to ``RECORD_PHASE``, the measured
far field to ``RECORD_IMAGE`` and the phase change inside the block to
``RECORD_GRAD``. Everything else is a named scalar.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from ao_shaping.utils.io.recorder_keys import (
    RECORD_EPOCH,
    RECORD_GRAD,
    RECORD_ID,
    RECORD_IMAGE,
    RECORD_PHASE,
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def crop_frame(frame: np.ndarray, camera_xy: np.ndarray, *, max_side: int = 420,
               margin: int = 24) -> np.ndarray:
    """Crop the measured frame to the optimization window and decimate it.

    The raw frame is the full 1944x2592 sensor; a GIF of that is unreadable
    and enormous. The window is taken from the run's own ``camera_xy`` (the
    sampled grid the optimizer saw), so the crop is exactly the region the
    numbers refer to. Decimation is a plain block average, so a bright speckle
    cannot disappear between pixels.
    """
    frame = np.asarray(frame, dtype=np.float64)
    x0, x1 = int(camera_xy[..., 0].min()), int(camera_xy[..., 0].max())
    y0, y1 = int(camera_xy[..., 1].min()), int(camera_xy[..., 1].max())
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(frame.shape[1] - 1, x1 + margin)
    y1 = min(frame.shape[0] - 1, y1 + margin)
    crop = frame[y0:y1 + 1, x0:x1 + 1]
    factor = max(1, int(np.ceil(max(crop.shape) / max_side)))
    if factor > 1:
        h, w = crop.shape[0] // factor * factor, crop.shape[1] // factor * factor
        cropped = crop[:h, :w]
        crop = cropped.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    return crop.astype(np.float32)


def _mean_frame(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as saved:
        return np.asarray(saved["raw_frames"]).mean(0)


def run_rows(run_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Return ``(captures, iterations, blocks)`` of one transfer run directory."""
    run_dir = Path(run_dir)
    captures = _read_json(run_dir / "manifest.json")
    blocks = _read_json(run_dir / "summary.json") if (run_dir / "summary.json").is_file() else []
    iterations: list[dict] = []
    if (run_dir / "iterations.pkl").is_file():
        with (run_dir / "iterations.pkl").open("rb") as stream:
            iterations = pickle.load(stream)
    return captures, iterations, blocks


def block_records(run_dir: Path, stage: str) -> list[dict]:
    """One record per CCD block of a run: what was measured, and what was kept."""
    run_dir = Path(run_dir)
    captures, iterations, blocks = run_rows(run_dir)
    by_label = {row["label"]: row for row in captures}
    last_phase: dict[tuple[str, int], np.ndarray] = {}
    phases: dict[tuple[str, int], list[np.ndarray]] = {}
    for entry in iterations:
        key = (entry["shape"], int(entry["block"]))
        phases.setdefault(key, []).append(np.asarray(entry["phase"]))
        last_phase[key] = np.asarray(entry["phase"])

    records: list[dict] = []
    for block in blocks:
        shape = block["shape"]
        index = int(block["block"])
        anchor = by_label.get(f"{shape}_{index}_A1")
        candidate = by_label.get(f"{shape}_{index}_B")
        if candidate is None:
            candidate = by_label.get(f"{shape}_{index}_B1")
        source = candidate or anchor
        if source is None:
            continue
        npz_path = run_dir / source["file"]
        with np.load(npz_path) as saved:
            camera_xy = np.asarray(saved["camera_xy"])
        frame = _mean_frame(npz_path)
        image = crop_frame(frame, camera_xy)
        key = (shape, index)
        phase = last_phase.get(key)
        gradient = None
        if phase is not None:
            series = phases.get(key, [])
            if len(series) >= 2:
                gradient = phase - series[0]
        anchor_metrics = (anchor or {}).get("metrics", {})
        candidate_metrics = (candidate or {}).get("metrics", {}) if candidate else {}
        anchor_cv = anchor_metrics.get("cv")
        candidate_cv = candidate_metrics.get("cv")
        record = {
            RECORD_EPOCH: len(records),
            RECORD_ID: len(records),
            "stage": stage,
            "shape": shape,
            "block": index,
            "label": source["label"],
            "exposure_ms": float(source.get("exposure_ms", 0.0)),
            "bit_depth": int(source.get("bit_depth") or 8),
            "cv_anchor": anchor_cv,
            "cv_candidate": candidate_cv,
            "cv_improvement": (None if anchor_cv is None or candidate_cv is None
                                else float(candidate_cv - anchor_cv)),
            "accepted": bool(block.get("accepted", False)),
            "acceptance": str(block.get("acceptance", "unknown")),
            "accept_metric": str(block.get("accept_metric", "")),
            "model_loss": block.get("model_loss_last"),
            "model_loss_previous": block.get("model_loss_previous"),
            "phase_lr": block.get("phase_lr"),
            "core_peak": float(source.get("core_peak", 0.0)),
            "saturation_pixels": int(source.get("saturation_pixels", 0)),
            "timestamp_utc": source.get("timestamp_utc"),
        }
        if image is not None:
            record[RECORD_IMAGE] = image
        if phase is not None:
            record[RECORD_PHASE] = phase.astype(np.float32)
        if gradient is not None:
            record[RECORD_GRAD] = gradient.astype(np.float32)
        records.append(record)
    return records


def append_run(recorder, run_dir: Path, stage: str) -> int:
    """Append one run's block records to ``recorder``; returns the count added."""
    records = block_records(run_dir, stage)
    for record in records:
        recorder.append(record)
    return len(records)


def save_records(recorder, out_dir: Path, *, mark: str = "process",
                 gif_name: str | None = None, fps: float = 2.0,
                 title: str | None = None) -> dict:
    """Write the recorder's history as CSV (+ sidecars), HDF5 and optionally a GIF.

    The GIF needs the ``Recorder.render_gif`` extension; when the installed
    ``utils.io.file`` does not have it, the table files are still written and
    the caller is told which part was skipped.
    """
    from ao_shaping.utils.io.file import save_history_hdf5

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder.save_dataframe(out_dir / f"{mark}.csv", sidecar_dir=out_dir / f"{mark}_sidecars")
    hdf5 = save_history_hdf5(recorder, out_dir / f"{mark}.h5",
                             metadata=dict(records=len(recorder.history), mark=mark))
    gif = None
    if gif_name and hasattr(recorder, "render_gif"):
        gif = recorder.render_gif(out_dir / gif_name, fps=fps, title=title)
    return dict(csv=str(out_dir / f"{mark}.csv"), hdf5=str(hdf5),
                gif=str(gif) if gif else None, records=len(recorder.history))
