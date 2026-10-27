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


def crop_frame(frame: np.ndarray, camera_xy: np.ndarray, *, target: np.ndarray | None = None,
               max_side: int = 420, margin: int = 24) -> np.ndarray:
    """Crop the measured frame to the target region and decimate it.

    The raw frame is the full 1944x2592 sensor; a GIF of that is unreadable
    and enormous. With a target raster the crop is the *shape* itself (plus a
    margin), so the spot fills the panel; without one it falls back to the
    sampled window. The crop is taken from the run's own ``camera_xy`` (the
    grid the optimizer saw), so it is exactly the region the numbers refer to.
    Decimation is a plain block average, so a bright speckle cannot
    disappear between pixels.
    """
    frame = np.asarray(frame, dtype=np.float64)
    target_array = None if target is None else np.asarray(target)
    if (target_array is not None and target_array.size
            and target_array.shape == camera_xy.shape[:2]):
        rows, columns = np.nonzero(target_array > .5)
        if rows.size:
            x0 = int(camera_xy[rows.min(), columns.min()][0])
            x1 = int(camera_xy[rows.max(), columns.max()][0])
            y0 = int(camera_xy[rows.min(), columns.min()][1])
            y1 = int(camera_xy[rows.max(), columns.max()][1])
        else:
            x0, x1, y0, y1 = 0, 0, 0, 0
    else:
        x0, x1 = int(camera_xy[..., 0].min()), int(camera_xy[..., 0].max())
        y0, y1 = int(camera_xy[..., 1].min()), int(camera_xy[..., 1].max())
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(frame.shape[1] - 1, x1 + margin)
    y1 = min(frame.shape[0] - 1, y1 + margin)
    if x1 <= x0 or y1 <= y0:
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
        try:
            with (run_dir / "iterations.pkl").open("rb") as stream:
                iterations = pickle.load(stream)
        except (EOFError, pickle.UnpicklingError):
            # The file is rewritten after every update, so a reader can catch
            # it half-written. The measurements (manifest) are still valid, so
            # the block records are kept without the phase panels.
            iterations = []
    return captures, iterations, blocks


def block_records(run_dir: Path, stage: str) -> list[dict]:
    """One record per CCD block of a run: what was measured, and what was kept."""
    run_dir = Path(run_dir)
    captures, iterations, blocks = run_rows(run_dir)
    by_label = {row["label"]: row for row in captures}
    last_phase: dict[tuple[str, int], np.ndarray] = {}
    for entry in iterations:
        key = (entry["shape"], int(entry["block"]))
        last_phase[key] = np.asarray(entry["phase"])

    records: list[dict] = []
    for block in blocks:
        shape = block["shape"]
        index = int(block["block"])
        anchor = by_label.get(f"{shape}_{index}_A1")
        candidate = by_label.get(f"{shape}_{index}_B")
        if candidate is None:
            candidate = by_label.get(f"{shape}_{index}_B1")
        if anchor is None and candidate is None:
            continue
        source = candidate or anchor
        npz_path = run_dir / source["file"]
        with np.load(npz_path) as saved:
            camera_xy = np.asarray(saved["camera_xy"])
            target = np.asarray(saved["target"]) if "target" in saved.files else None
        frame = _mean_frame(npz_path)
        image = crop_frame(frame, camera_xy, target=target)
        key = (shape, index)
        phase = last_phase.get(key)
        gradient = None
        if phase is not None:
            # The phase the block *started* from: the previous block's last
            # phase, or -- for the first block -- the phase that was displayed
            # as the seed. ``iterations.pkl`` only stores post-update phases,
            # so the seed has to come from the capture itself.
            if index == 0:
                seed_row = by_label.get(f"{shape}_seed")
                if seed_row is not None:
                    with np.load(run_dir / seed_row["file"]) as saved:
                        start_phase = np.asarray(saved["model_phase"], dtype=np.float64)
                    gradient = phase - start_phase
            else:
                start_phase = last_phase.get((shape, index - 1))
                if start_phase is not None:
                    gradient = phase - start_phase
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
    """Append one run's block records to ``recorder``; returns the count added.

    The epoch / id numbering is owned by the recorder, not by the run: two runs
    both start their own block counter at zero, and a duplicate id would make
    the HDF5 export fail on a repeated group name.
    """
    records = block_records(run_dir, stage)
    base = len(recorder.history)
    for offset, record in enumerate(records):
        index = base + offset
        record[RECORD_EPOCH] = index
        record[RECORD_ID] = index
        recorder.append(record)
    return len(records)


def _detrend(phase: np.ndarray) -> np.ndarray:
    """Remove the piston and the linear (carrier) part of a phase map.

    The displayed phase carries the carrier tilt and the device's piston; both
    are a linear ramp over the whole panel, and leaving them in makes the map
    look like white noise once it is wrapped for display.
    """
    phase = np.asarray(phase, dtype=np.float64)
    h, w = phase.shape
    y, x = np.mgrid[0:h, 0:w]
    design = np.stack((np.ones(h * w), x.ravel(), y.ravel()), axis=1)
    solution, *_ = np.linalg.lstsq(design, phase.ravel(), rcond=None)
    return (phase - (design @ solution).reshape(h, w))


def render_process_gif(records: list[dict], path: str | Path, *, fps: float = 2.0,
                       title: str | None = None) -> Path:
    """Render the CCD-block records as a readable process animation.

    Six panels per frame: the detrended pupil phase, the measured far field
    (cropped to the target), the phase change inside the block, the block's
    numbers, the measured core-CV trajectory and the model-loss trajectory.
    The generic ``Recorder.render_gif`` is not used here: it classifies the
    columns by type, which puts a phase map and a measured frame on the same
    footing and cannot remove the carrier.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    # The panel labels are Chinese; DejaVu Sans has no CJK glyphs, so the
    # default font would render them as empty boxes.
    font_path = "C:/Windows/Fonts/msyh.ttc"
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)
        family = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = family
        plt.rcParams["font.sans-serif"] = [family, "SimHei", "SimSun"]
    else:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False

    if not records:
        raise ValueError("no records to render")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = [record.get(RECORD_IMAGE) for record in records]
    # Normalise by the exposure: the blocks are metered independently, so the
    # raw counts of an early block are dimmer than a later one for reasons that
    # have nothing to do with the phase. Counts per millisecond is the
    # comparable quantity, and it is what the metrics are computed on.
    def _counts_per_ms(record, image):
        """Mono8-equivalent counts per millisecond.

        The blocks are metered independently and the bit depth changes between
        stages, so the raw counts are neither comparable nor displayable: an
        8-bit frame is capped at 255 counts while a 12-bit frame reaches 4095.
        ``counts / (exposure_ms * 2**(bit_depth - 8))`` is the same unit the
        metrics are computed on.
        """
        exposure = record.get("exposure_ms") or 0.0
        if image is None or not exposure:
            return image
        bit_depth = int(record.get("bit_depth") or 8)
        return image / (float(exposure) * float(2 ** (bit_depth - 8)))

    normalized = [_counts_per_ms(record, image)
                  for record, image in zip(records, images)]
    # A robust scale, not the maximum: the zero order saturates a few hundred
    # pixels in the middle of the target, and a scale set by it would render
    # the shaped light itself as a dark field.
    peaks = [float(np.percentile(image, 95)) for image in normalized if image is not None]
    image_vmax = max(peaks) if peaks else 1.0
    image_vmax = image_vmax if image_vmax > 0 else 1.0
    changes = [record.get(RECORD_GRAD) for record in records]
    change_scale = max((float(np.percentile(np.abs(change), 90)) for change in changes
                        if change is not None), default=1.0)
    change_scale = change_scale if change_scale > 0 else 1.0
    phases = [record.get(RECORD_PHASE) for record in records]
    detrended = [_detrend(phase) for phase in phases if phase is not None]
    phase_scale = max((float(np.percentile(np.abs(phase), 90)) for phase in detrended),
                       default=1.0)
    phase_scale = phase_scale if phase_scale > 0 else 1.0

    anchors = [record.get("cv_anchor") for record in records]
    candidates = [record.get("cv_candidate") for record in records]
    losses = [record.get("model_loss") for record in records]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    writer = PillowWriter(fps=int(fps))
    with writer.saving(fig, str(path), dpi=110):
        for index, record in enumerate(records):
            for row in axes:
                for ax in row:
                    ax.clear()

            phase = record.get(RECORD_PHASE)
            if phase is not None:
                axes[0][0].imshow(_detrend(phase), cmap="coolwarm",
                                  vmin=-phase_scale, vmax=phase_scale)
            axes[0][0].set_title("瞳面相位 (去活塞/倾斜, 90% 分位色标)", fontsize=10)
            axes[0][0].axis("off")

            image = normalized[index]
            if image is not None:
                axes[0][1].imshow(image, cmap="inferno", vmin=0, vmax=image_vmax)
            axes[0][1].set_title("实测远场 (目标区, Mono8 等效 counts/ms)", fontsize=10)
            axes[0][1].axis("off")

            change = record.get(RECORD_GRAD)
            if change is not None:
                axes[0][2].imshow(_detrend(change), cmap="coolwarm",
                                  vmin=-change_scale, vmax=change_scale)
            axes[0][2].set_title("块内相位变化 (rad, 90% 分位色标)", fontsize=10)
            axes[0][2].axis("off")

            axes[1][0].axis("off")
            lines = [f"{key}: {record.get(key)}" for key in
                     ("shape", "stage", "block", "label", "exposure_ms", "bit_depth",
                      "cv_anchor", "cv_candidate", "cv_improvement", "accepted",
                      "acceptance", "model_loss")]
            axes[1][0].text(0.02, 0.98, "\n".join(lines), transform=axes[1][0].transAxes,
                            fontsize=9, va="top", family="sans-serif")
            axes[1][0].set_title("本块读数", fontsize=10)

            xs = list(range(index + 1))
            axes[1][1].plot(xs, [value if value is not None else np.nan
                                 for value in anchors[: index + 1]],
                            marker="o", markersize=3, color="#8C8C8C", label="锚点 A1")
            axes[1][1].plot(xs, [value if value is not None else np.nan
                                 for value in candidates[: index + 1]],
                            marker="s", markersize=3, color="#C44E52", label="候选 B")
            axes[1][1].set_title("实测核心 CV", fontsize=10)
            axes[1][1].set_xlabel("块")
            axes[1][1].legend(fontsize=7)
            axes[1][1].grid(alpha=.3)

            axes[1][2].plot(xs, [value if value is not None else np.nan
                                 for value in losses[: index + 1]],
                            marker="o", markersize=3, color="#4C72B0")
            axes[1][2].set_title("模型损失", fontsize=10)
            axes[1][2].set_xlabel("块")
            axes[1][2].grid(alpha=.3)

            heading = title or "逐 CCD 块"
            axes[0][1].figure.suptitle(f"{heading} — 块 {index + 1}/{len(records)}", fontsize=12)
            writer.grab_frame()
    plt.close(fig)
    return path


def save_records(recorder, out_dir: Path, *, mark: str = "process",
                 gif_name: str | None = None, fps: float = 2.0,
                 title: str | None = None) -> dict:
    """Write the recorder's history as CSV (+ sidecars), HDF5 and the GIF."""
    from ao_shaping.utils.io.file import save_history_hdf5

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder.save_dataframe(out_dir / f"{mark}.csv", sidecar_dir=out_dir / f"{mark}_sidecars")
    hdf5 = save_history_hdf5(recorder, out_dir / f"{mark}.h5",
                             metadata=dict(records=len(recorder.history), mark=mark))
    gif = None
    if gif_name:
        gif = render_process_gif(recorder.history, out_dir / gif_name, fps=fps, title=title)
    return dict(csv=str(out_dir / f"{mark}.csv"), hdf5=str(hdf5),
                gif=str(gif) if gif else None, records=len(recorder.history))
