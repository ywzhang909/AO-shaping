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
               max_side: int = 420, margin: int = 24,
               ) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Crop the measured frame to the **sampling window** and decimate it.

    The raw frame is the full 1944x2592 sensor; a GIF of that is unreadable and
    enormous. The crop is the window the optimizer actually measured (taken from
    the run's own ``camera_xy``), not just the target box: what happens *outside*
    the target is a result too, and with a window at 1.5x the side it is the only
    place that is visible. The target raster is returned as a box **in crop
    pixel coordinates** so the renderer can outline it.

    Decimation is a plain block average, so a bright speckle cannot disappear
    between pixels.
    """
    frame = np.asarray(frame, dtype=np.float64)
    x0 = int(camera_xy[..., 0].min()) - margin
    y0 = int(camera_xy[..., 1].min()) - margin
    x1 = int(camera_xy[..., 0].max()) + margin
    y1 = int(camera_xy[..., 1].max()) + margin
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame.shape[1] - 1, x1), min(frame.shape[0] - 1, y1)
    crop = frame[y0:y1 + 1, x0:x1 + 1]
    factor = max(1, int(np.ceil(max(crop.shape) / max_side)))
    if factor > 1:
        h, w = crop.shape[0] // factor * factor, crop.shape[1] // factor * factor
        crop = crop[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    box = None
    target_array = None if target is None else np.asarray(target)
    if (target_array is not None and target_array.size
            and target_array.shape == camera_xy.shape[:2]):
        rows, columns = np.nonzero(target_array > .5)
        if rows.size:
            tx0 = int(camera_xy[rows.min(), columns.min()][0]) - x0
            ty0 = int(camera_xy[rows.min(), columns.min()][1]) - y0
            tx1 = int(camera_xy[rows.max(), columns.max()][0]) - x0
            ty1 = int(camera_xy[rows.max(), columns.max()][1]) - y0
            box = (tx0 // factor, ty0 // factor, tx1 // factor, ty1 // factor)
    return crop.astype(np.float32), box


def capture_record(row: dict, *, image: np.ndarray, camera_xy: np.ndarray,
                   target: np.ndarray | None, phase: np.ndarray,
                   previous_phase: np.ndarray | None, stage: str) -> dict:
    """One Recorder record from one capture of a ``validate.run`` block.

    Single implementation: the live run and :func:`rebuild_run` both build their
    records here, so the table, the HDF5 and the animation cannot drift apart.
    ``image`` is the already-averaged frame; ``target`` is the target raster on
    the ``camera_xy`` grid (the outline is drawn from it).
    """
    phase_array = np.asarray(phase, dtype=np.float32)
    far_field, box = crop_frame(image, camera_xy, target=target)
    return {
        "cv": float(row["metrics"]["cv"]),
        "core_peak": float(row["core_peak"]),
        "model_loss": float(row["loss"]),
        "label": row["label"],
        "stage": stage,
        "shape": str(row["label"]).split("_")[0],
        "exposure_ms": float(row["exposure_ms"]),
        "bit_depth": int(row["bit_depth"]),
        "saturation_pixels": int(row["saturation_pixels"]),
        # Target outline in crop pixels: the window is wider than the target on
        # purpose, so the animation has to say which part is the requested shape.
        "target_box": ",".join(str(int(v)) for v in box) if box else "",
        RECORD_PHASE: phase_array,
        RECORD_IMAGE: far_field,
        RECORD_GRAD: (phase_array - previous_phase if previous_phase is not None
                      else np.zeros_like(phase_array)),
    }


def rebuild_run(run_dir: Path, out_dir: Path | None = None, *, fps: float = 2.0,
                title: str | None = None) -> dict:
    """Rebuild the Recorder records of a finished ``validate.run`` directory.

    Device-free: the capture ``.npz`` files and ``manifest.json`` are the source
    of truth, so a run whose live export failed (or was made with an older
    renderer) can be re-exported without re-measuring anything.
    """
    from ao_shaping.utils.io.file import Recorder

    run_dir = Path(run_dir)
    rows = _read_json(run_dir / "manifest.json")
    recorder = Recorder(mark="cv", mode="min")
    last_phase: dict[str, np.ndarray] = {}
    for row in rows:
        with np.load(run_dir / row["file"]) as saved:
            image = np.asarray(saved["raw_frames"]).mean(0)
            camera_xy = np.asarray(saved["camera_xy"])
            target = np.asarray(saved["target"]) if "target" in saved.files else None
            phase = np.asarray(saved["model_phase"], dtype=np.float32)
        shape = str(row["label"]).split("_")[0]
        recorder.append(capture_record(row, image=image, camera_xy=camera_xy, target=target,
                                       phase=phase, previous_phase=last_phase.get(shape),
                                       stage=run_dir.name))
        last_phase[shape] = phase
    return save_records(recorder, out_dir or run_dir / "recording", mark="process",
                        gif_name="process.gif", fps=fps,
                        title=title or f"{run_dir.name} (重建记录)")


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
        image, _ = crop_frame(frame, camera_xy, target=target)
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

    The displayed phase carries the carrier tilt and the device's piston; the
    linear ramp is a *drawn* pattern, not a measurement. Keeping the fit to a
    fixed linear basis (rather than a higher-order polynomial) is deliberate: the basis is fixed,
    so two frames that differ by an aberration difference differ by the same
    amount after detrending -- a higher-order fit in the same 8-bit quantised
    phase makes the removed part jump between neighbours and buries the
    evolution in resampling noise.
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
    # The in-block change is drawn to the same colour scale as the phase itself:
    # it is a difference of two phases, so the range that fits them fits it, and
    # a second scale would only cost the reader the comparison.
    phases = [record.get(RECORD_PHASE) for record in records]
    detrended = [_detrend(phase) for phase in phases if phase is not None]
    # One colour scale for the whole animation has to hold every frame, so two
    # numbers are taken from the set of frames, never from the per-frame 90%
    # quantile: the median *offset* (a frame's own median sits wherever the
    # aberration happens to be, and a shared ``vmin=-vmax`` would clip a frame
    # whose median is not zero), and the largest *spread* (the 90% quantile of
    # ``|frame - its median|``). A per-frame scale would silently flatten the
    # evolution: every frame gets its full range back, so the panel looks the
    # same in every frame.
    phase_centre = (float(np.median([np.median(m) for m in detrended]))
                    if detrended else 0.0)
    phase_scale = max((float(np.percentile(np.abs(m - np.median(m)), 90))
                       for m in detrended), default=1.0)
    phase_scale = phase_scale if phase_scale > 0 else 1.0

    # The frame's own phase and its in-block change share one colour scale --
    # the change is a difference of two phases, so a scale that fits the phases
    # fits it. Both are taken from the whole animation, never per frame.
    phase_limits = phase_centre - phase_scale, phase_centre + phase_scale

    trajectory_keys = [key for key in ("cv_anchor", "cv_candidate")
                       if any(key in record for record in records)]
    if not trajectory_keys:
        trajectory_keys = ["cv"]
    series = {key: [record.get(key) for record in records] for key in trajectory_keys}
    losses = [record.get("model_loss") for record in records]
    trajectory_labels = {"cv": "实测核心 CV", "cv_anchor": "锚点 A1", "cv_candidate": "候选 B"}

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    writer = PillowWriter(fps=int(fps))
    writer.setup(fig, str(path), dpi=110)
    rendered = 0
    try:
        seen_change = False
        for index, record in enumerate(records):
            for row in axes:
                for ax in row:
                    ax.clear()

            phase = record.get(RECORD_PHASE)
            if phase is not None:
                axes[0][0].imshow(_detrend(phase), cmap="coolwarm",
                                  vmin=phase_limits[0], vmax=phase_limits[1])
            axes[0][0].set_title("瞳面相位 (去活塞/倾斜; 全片共用色标)", fontsize=10)
            axes[0][0].axis("off")

            image = normalized[index]
            if image is not None:
                axes[0][1].imshow(image, cmap="inferno", vmin=0, vmax=image_vmax)
            # The window is deliberately wider than the target; the outline says
            # which part of it is the requested shape, so the light outside it is
            # readable as "outside" rather than as more target.
            box = str(record.get("target_box") or "")
            if box and image is not None:
                from matplotlib.patches import Rectangle

                try:
                    x0, y0, x1, y1 = (float(value) for value in box.split(","))
                except ValueError:
                    pass
                else:
                    axes[0][1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                                   edgecolor="#FFFFFF", linewidth=1.0,
                                                   linestyle="--"))
            axes[0][1].set_title("实测远场 (整窗, Mono8 等效 counts/ms; 虚线=目标)", fontsize=10)
            axes[0][1].axis("off")

            change = record.get(RECORD_GRAD)
            if change is not None and np.abs(change).max() > 0:
                seen_change = True
                axes[0][2].imshow(_detrend(change), cmap="coolwarm",
                                  vmin=phase_limits[0], vmax=phase_limits[1])
            elif change is not None:
                # B2 replays B1's phase on purpose; an all-zero change is a
                # result (the repeat), not a rendering failure.
                axes[0][2].text(.5, .5, "与上一次相位相同\n(重复采集)", ha="center", va="center",
                                transform=axes[0][2].transAxes, fontsize=10, color="#555555")
            axes[0][2].set_title("块内相位变化 (与左图同色标)", fontsize=10)
            axes[0][2].axis("off")

            axes[1][0].axis("off")
            text_keys = ("shape", "stage", "block", "label", "exposure_ms", "bit_depth",
                         "cv", "cv_anchor", "cv_candidate", "cv_improvement",
                         "core_peak", "saturation_pixels", "accepted", "acceptance",
                         "model_loss")
            lines = [f"{key}: {record.get(key)}" for key in text_keys
                     if record.get(key) is not None]
            axes[1][0].text(0.02, 0.98, "\n".join(lines), transform=axes[1][0].transAxes,
                            fontsize=9, va="top", family="sans-serif")
            axes[1][0].set_title("本块读数", fontsize=10)

            xs = list(range(index + 1))
            for key in trajectory_keys:
                axes[1][1].plot(xs, [value if value is not None else np.nan
                                     for value in series[key][: index + 1]],
                                marker="o", markersize=3,
                                label=trajectory_labels.get(key, key))
            axes[1][1].set_title("实测核心 CV", fontsize=10)
            axes[1][1].set_xlabel("采集")
            axes[1][1].legend(fontsize=7)
            axes[1][1].grid(alpha=.3)

            axes[1][2].plot(xs, [value if value is not None else np.nan
                                 for value in losses[: index + 1]],
                            marker="o", markersize=3, color="#4C72B0")
            axes[1][2].set_title("模型损失", fontsize=10)
            axes[1][2].set_xlabel("采集")
            axes[1][2].grid(alpha=.3)

            heading = title or "逐 CCD 块"
            # The A–B–B–A block also captures a deliberate opposite-sign phase
            # to separate a model-Jacobian error from an autograd error. Its CV
            # is bad *by construction*, so the frame says so instead of looking
            # like a failed block.
            label = str(record.get("label") or "")
            if "direction" in label:
                heading = f"{heading} · 诊断探针 (反号, 非候选)"
            axes[0][1].figure.suptitle(f"{heading} — 块 {index + 1}/{len(records)}", fontsize=12)
            writer.grab_frame()
            rendered += 1
    except BaseException:
        # ``PillowWriter.saving`` calls ``finish()`` from a ``finally``, so an
        # error inside the loop used to surface as "list index out of range"
        # from finish() instead of the real cause. Setup/finish are driven by
        # hand here so the original exception survives.
        plt.close(fig)
        path.unlink(missing_ok=True)
        raise
    if rendered == 0:
        plt.close(fig)
        path.unlink(missing_ok=True)
        raise ValueError("no GIF frame was rendered")
    if not seen_change:
        # Every frame says "same as the previous phase". That is only a result
        # when the records carry no phase at all (the fallback records below);
        # if they do carry one, the renderer has nothing to show for it, and it
        # says so instead of writing a GIF whose first frame equals its last.
        plt.close(fig)
        path.unlink(missing_ok=True)
        raise ValueError("no frame carried an in-block phase change; "
                         "the phase column is missing from the records")
    writer.finish()
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
