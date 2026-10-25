"""Evaluate a trained sim2f U-Net against the teacher on the aberration-free twin.

The table this module produces is a **three-row** comparison, never a single
number:

* ``flat`` -- no phase at all, the bench's own spot.  This is the baseline any
  shaping result has to beat, and the repository has measured more than once
  that a shaping method can lose to it (Gerchberg-Saxton alone: 90/90 runs
  negative).
* ``net`` -- one forward pass of the trained network.
* ``teacher`` -- the iterative optimum the cache's labels came from, recomputed
  on this bench so the row is measured, not copied.

All three rows are scored with :func:`ml.sim2f.losses.objective_terms`, i.e. the
teacher's own ``L_shape + w_ee * (1 - EE)``, so the difference between ``net``
and ``teacher`` is the *amortisation* cost of replacing ``TeacherConfig.iters``
Adam steps with one forward pass.  That is the only claim this package makes.

Per-family rows are always reported with their sample counts.  A cross-family
mean is offered as well, but it is not a claim about any family: the target
kinds differ in how much of their energy is already concentrated, and averaging
them hides exactly the case that fails.

What is measured, and on what frame
-----------------------------------
The bench is the repository's calibrated 2f twin, so a frame carries a zero-order
background term and (when ``Sim2fConfig.exposure_ms > 0``) a dark level, PRNU and
read noise.  A ratio formed on a raw frame therefore mixes light with detector
pedestal, so every row is scored on the **prepared** frame by default: per-frame
median subtracted, negatives clipped at 0
(:func:`ml.sim2f.losses.prepare_frame`).  ``prepare=False`` reproduces the raw
numbers for the ablation; those are *not* comparable with a prepared teacher, and
``eval.json`` records which one it holds under ``"prepare"``.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import Tensor

from ml.sim2f.config import Sim2fConfig, TrainConfig
from ml.sim2f.losses import (
    as_frame,
    forward_image,
    from_phasor,
    objective_terms,
    piston_free_phasor_corr,
    shaping_metrics,
    to_phasor,
)
from ml.sim2f._device import _resolve_device
from ml.sim2f.model import build_unet, network_input
from ml.sim2f.schema import CACHE_FILE
from ml.sim2f.train import (
    _build_sim,
    load_corpus,
)

__all__ = ["EVAL_NAME", "evaluate", "format_table"]

#: Report written next to the checkpoint (or at ``--out``).
EVAL_NAME = "eval.json"

#: Columns printed, in order.  ``objective`` is the teacher's objective; the rest
#: are the camera-space metrics of :func:`ml.sim2f.losses.shaping_metrics`.
TABLE_COLUMNS: tuple[str, ...] = (
    "objective",
    "l1",
    "ncc",
    "ee",
    "cv_in_support",
    "iou50",
    "peak_ratio",
)


def _config_from_checkpoint(payload: dict[str, Any]) -> tuple[TrainConfig, Sim2fConfig, Tensor]:
    """Recover the two configs and the PSF context from a checkpoint.

    Args:
        payload: The object written by :func:`ml.sim2f.train._save_checkpoint`.

    Returns:
        ``(train_cfg, sim_cfg, context)``.

    Raises:
        KeyError: If the checkpoint lacks ``state_dict``.
        ValueError: If the configs cannot be rebuilt.
    """
    if "state_dict" not in payload:
        raise KeyError("checkpoint has no 'state_dict'")
    train_cfg = payload.get("train_config")
    if not isinstance(train_cfg, TrainConfig):
        section = payload.get("train_config_dict") or payload.get("train_config") or {}
        known = {item.name for item in fields(TrainConfig)}
        train_cfg = TrainConfig(**{k: v for k, v in section.items() if k in known})
    sim_cfg = payload.get("sim_config")
    if not isinstance(sim_cfg, Sim2fConfig):
        section = payload.get("sim_config_dict") or payload.get("sim_config") or {}
        known = {item.name for item in fields(Sim2fConfig)}
        sim_cfg = Sim2fConfig(**{k: v for k, v in section.items() if k in known})
    context = payload.get("context")
    if context is None:
        raise ValueError(
            "checkpoint has no 'context' -- retrain; the PSF-context channel is "
            "part of the model input and cannot be recomputed here without optics"
        )
    return train_cfg, sim_cfg, as_frame(context)


def _row(
    image: Tensor,
    target: Tensor,
    support: Tensor,
    *,
    w_ee: float,
    metric: str,
    prepare: bool = True,
) -> dict[str, float]:
    """Score one frame against one target: metrics plus the teacher objective.

    Args:
        image: ``(1, 1, ccd_px, ccd_px)`` predicted frame.
        target: ``(1, 1, ccd_px, ccd_px)`` requested frame.
        support: ``(1, 1, ccd_px, ccd_px)`` analytic support.
        w_ee: Teacher's encircled-energy weight.
        metric: Teacher's shape metric.
        prepare: Remove the detector pedestal before any ratio (default on; the
            bench is the calibrated twin, not an ideal chain).

    Returns:
        The :func:`~ml.sim2f.losses.shaping_metrics` columns plus
        ``objective``, which additionally carries the ``w_ee * (1 - EE)`` term
        (so it is not a monotone function of ``l1`` alone).
    """
    row = shaping_metrics(image, target, support, prepare=prepare)
    loss, _l1, _ee, _ncc = objective_terms(
        image, target, support, w_ee=w_ee, metric=metric, prepare=prepare
    )
    row["objective"] = float(loss)
    return row


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """Column-wise mean, ignoring nothing and preserving every key.

    Args:
        rows: Non-empty list of metric dicts with identical keys.

    Returns:
        The mean of each column.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("cannot average an empty list of metric rows")
    keys = list(rows[0])
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def _finite(value: Any) -> Any:
    """Replace non-finite floats with ``None`` for a standards-compliant JSON.

    Args:
        value: Any JSON-shaped object.

    Returns:
        The same structure with non-finite floats replaced by ``None``.
    """
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def format_table(rows: dict[str, dict[str, float]]) -> str:
    """Render ``{row name: columns}`` as a fixed-width text table.

    Args:
        rows: Ordered mapping of row label to its metric dict.

    Returns:
        A multi-line table, one header plus one line per row.  Missing columns
        print as ``-`` so a partial row is visible rather than silently absent.
    """
    width = max((len(name) for name in rows), default=6)
    header = " " * (width + 2) + "  ".join(f"{column:>12}" for column in TABLE_COLUMNS)
    lines = [header]
    for name, row in rows.items():
        cells = []
        for column in TABLE_COLUMNS:
            value = row.get(column)
            cells.append(f"{value:>12.4f}" if isinstance(value, float) else f"{'-':>12}")
        lines.append(f"{name:<{width}}  " + "  ".join(cells))
    return "\n".join(lines)


def evaluate(
    checkpoint: str | Path,
    *,
    cache_dir: str | Path,
    split: str = "test",
    device: str | torch.device = "auto",
    out: str | Path | None = None,
    prepare: bool = True,
) -> dict[str, Any]:
    """Score a checkpoint on one split of the cache and write ``eval.json``.

    Args:
        checkpoint: Checkpoint written by :func:`ml.sim2f.train.train`.
        cache_dir: Directory holding the corpus the checkpoint was trained on.
        split: ``"train"``, ``"val"``, ``"test"`` or ``"all"``.
        device: ``"auto"``, ``"cuda"``, ``"cpu"`` or a device.
        out: Report path.  Defaults to ``eval.json`` next to the checkpoint.
        prepare: Remove the detector pedestal (median subtraction, clip at 0)
            before any ratio.  On by default because the bench is the calibrated
            2f twin; ``False`` reproduces the raw-frame numbers for the ablation
            and is *not* comparable with a prepared teacher.

    Returns:
        The report payload (also written to disk).

    Raises:
        FileNotFoundError: If the checkpoint or corpus is missing.
        ValueError: If the split is empty or the checkpoint's bench does not
            match the cache's.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    resolved = _resolve_device(device)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    train_cfg, sim_cfg, context = _config_from_checkpoint(payload)

    corpus = load_corpus(cache_dir)
    if (corpus.sim_cfg.grid, corpus.sim_cfg.ccd_px) != (sim_cfg.grid, sim_cfg.ccd_px):
        raise ValueError(
            "checkpoint bench does not match the cache: "
            f"grid/ccd_px {sim_cfg.grid}/{sim_cfg.ccd_px} vs "
            f"{corpus.sim_cfg.grid}/{corpus.sim_cfg.ccd_px}"
        )
    indices = corpus.indices(split)
    if int(indices.numel()) == 0:
        raise ValueError(f"split {split!r} of {Path(cache_dir) / CACHE_FILE} is empty")

    context = context.to(device=resolved, dtype=torch.float32)
    model = build_unet(train_cfg, sim_cfg, device=resolved, context=context)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    mask = model.pupil_mask
    bench = _build_sim(sim_cfg, resolved)
    teacher_cfg = corpus.teacher_cfg

    target = corpus.target[indices].to(device=resolved, dtype=torch.float32)
    support = corpus.support[indices].to(device=resolved, dtype=torch.float32)
    gt_phase = corpus.phase[indices].to(device=resolved, dtype=torch.float32)
    n = int(indices.numel())

    with torch.no_grad():
        pred_phasor = model(network_input(target, context))
        pred_image = forward_image(bench, from_phasor(pred_phasor))
        teacher_image = forward_image(bench, gt_phase)
        flat_image = forward_image(bench, torch.zeros(1, sim_cfg.grid, sim_cfg.grid, device=resolved))
    if flat_image.shape[0] == 1 and n != 1:
        flat_image = flat_image.expand(n, -1, -1, -1)

    # The teacher row is recomputed rather than copied from `teacher_metrics`,
    # so every row is measured on the same bench with the same metric code.
    cached = corpus.teacher_metrics[indices, 0]
    recomputed = [
        objective_terms(teacher_image[i : i + 1], target[i : i + 1],
                        support[i : i + 1], w_ee=teacher_cfg.w_ee,
                        metric=teacher_cfg.metric, prepare=prepare)[0]
        for i in range(n)
    ]
    for i, value in enumerate(recomputed):
        reference = float(cached[i])
        measured = float(value)
        if reference > 0 and abs(measured - reference) / reference > 0.05:
            logger.warning(
                "sample {}: recomputed teacher objective {:.4f} != cached {:.4f}; "
                "the cache was made with different optics",
                i, measured, reference,
            )
            break

    kinds = corpus.kinds()
    per_sample: list[dict[str, Any]] = []
    net_rows: list[dict[str, float]] = []
    flat_rows: list[dict[str, float]] = []
    teacher_rows: list[dict[str, float]] = []
    by_kind: dict[str, dict[str, list[dict[str, float]]]] = {}

    for position in range(n):
        window = slice(position, position + 1)
        net = _row(pred_image[window], target[window], support[window],
                   w_ee=teacher_cfg.w_ee, metric=teacher_cfg.metric, prepare=prepare)
        flat = _row(flat_image[window], target[window], support[window],
                    w_ee=teacher_cfg.w_ee, metric=teacher_cfg.metric, prepare=prepare)
        teacher = _row(teacher_image[window], target[window], support[window],
                       w_ee=teacher_cfg.w_ee, metric=teacher_cfg.metric, prepare=prepare)
        corr = piston_free_phasor_corr(
            pred_phasor[window], to_phasor(gt_phase[window]), mask
        )
        net_rows.append(net)
        flat_rows.append(flat)
        teacher_rows.append(teacher)
        kind = kinds[int(indices[position])]
        bucket = by_kind.setdefault(kind, {"net": [], "flat": [], "teacher": []})
        bucket["net"].append(net)
        bucket["flat"].append(flat)
        bucket["teacher"].append(teacher)
        per_sample.append(
            {
                "index": int(indices[position]),
                "kind": kind,
                "net": net,
                "flat": flat,
                "teacher": teacher,
                "phasor_corr": corr,
            }
        )

    aggregate = {
        "flat": _mean_rows(flat_rows),
        "net": _mean_rows(net_rows),
        "teacher": _mean_rows(teacher_rows),
        "phasor_corr": sum(item["phasor_corr"] for item in per_sample) / n,
    }
    per_kind = {
        kind: {
            "n": len(buckets["net"]),
            "flat": _mean_rows(buckets["flat"]),
            "net": _mean_rows(buckets["net"]),
            "teacher": _mean_rows(buckets["teacher"]),
        }
        for kind, buckets in by_kind.items()
    }

    report = {
        "checkpoint": str(path),
        "cache": str(Path(cache_dir)),
        "split": str(split).lower(),
        "n_samples": n,
        "n_parameters": model.n_parameters(),
        "prepare": bool(prepare),
        "sim": sim_cfg.describe(),
        "teacher": teacher_cfg.to_dict(),
        "aggregate": aggregate,
        "per_kind": per_kind,
        "per_sample": per_sample,
        "columns": list(TABLE_COLUMNS),
        "notes": (
            "All rows are measured on the same aberration-free bench with the "
            "teacher's own objective (L_shape + w_ee*(1-EE)). `teacher` is "
            "recomputed from the cached phase, not copied from the cache's "
            "metrics. Cross-kind means are not a claim about any single kind: "
            "see the per-kind rows and their counts."
        ),
    }
    report = _finite(report)

    destination = Path(out) if out is not None else path.parent / EVAL_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary_rows = {
        "flat": aggregate["flat"],
        "net": aggregate["net"],
        "teacher": aggregate["teacher"],
    }
    logger.info(
        "eval {} split={} n={} params={} phasor_corr={:.4f}\n{}",
        path.name, split, n, report["n_parameters"], aggregate["phasor_corr"],
        format_table(summary_rows),
    )
    for kind, entry in per_kind.items():
        logger.info(
            "kind {:<9} n={:<4} objective flat={:.4f} net={:.4f} teacher={:.4f}",
            kind, entry["n"], entry["flat"]["objective"], entry["net"]["objective"],
            entry["teacher"]["objective"],
        )
    return report
