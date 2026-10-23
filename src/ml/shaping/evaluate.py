"""Evaluation entry point for the inverse beam-shaping model.

Run it as ``python -m ml.shaping.evaluate``. Loads a checkpoint written by
:mod:`ml.shaping.train`, scores the validation and test splits against two
baselines, and reports paired statistics **per file**.

============================  Aggregation order, and why it matters  ============================

The order is per record -> **per FILE** -> per FAMILY, and statistics are
computed on **file-level** values only.

That is not a stylistic choice. Records inside a single debug pickle are
near-duplicates -- consecutive SPGD epochs of one optimisation run -- so a
record-level test treats thousands of strongly correlated samples as
independent evidence and reports a confident, meaningless p-value.
``report/hwdataset_corpus/report.md`` §12 lists this as one of the things the
corpus structurally cannot support. Every record-level number is still written
to ``metrics.json``; it is simply never the input to a significance test.

============================  Diff convention  ============================

Every paired difference is ``baseline_metric - model_metric``, so a **positive**
number means the model is better. This is stated in the module docstring, in the
``metrics.json`` payload (``"diff_convention"``) and in the console table, because
the sign is the single easiest thing to get backwards when reading a report.

============================  What is deliberately NOT claimed  ============================

The corpus's 7 families are **perfectly collinear** with their phase
representation (``source`` and family are the same variable), so no cross-family
aggregate here is a claim about which phase representation is better. Family
metrics are reported side by side and the pooled row is labelled as such.

The corpus also has a measured noise floor: the *identical* configuration gives
a validation R^2 anywhere in 0.748-0.937 purely from changing the split seed.
Only paired differences carry information here; a single-run absolute number is
not rankable, and this module says so in its own output.

Statistics come exclusively from :mod:`ml.zernike.eval_stats`, the repository's
single authoritative implementation. Re-deriving a sign-flip test here would be
exactly the duplicate-implementation drift this repository has been bitten by
before.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from ml.hwdataset.records import MaterialiserConfig
from ml.shaping.config import IMAGE_MODES, UNKNOWN_FOV, ShapingConfig, parse_csv_tuple
from ml.shaping.dataset import build_shaping_loaders, corpus_summary, network_input
from ml.shaping.losses import _peak_normalise
from ml.shaping.models import build_phasor_unet, unit_phasor
from ml.shaping.physics import ForwardBank
from ml.zernike.eval_stats import (
    holm_bonferroni,
    min_attainable_pvalue,
    paired_comparison,
)

__all__ = [
    "DIFF_CONVENTION",
    "EvaluationResult",
    "evaluate",
    "main",
]

#: Recorded in every artefact so a reader cannot guess the sign.
DIFF_CONVENTION = "baseline_minus_model (positive means the model is better)"

#: Metric names evaluated per record, in report order.
_METRICS = ("phasor_mae", "physics_l1")

#: Reason labels used when a record cannot enter the physics metric.
_REASON_UNKNOWN_FOV = "fov_unknown"
_REASON_NO_PADDING = "fov_1944_no_calibrated_padding"


@dataclass(frozen=True)
class EvaluationResult:
    """Everything one :func:`evaluate` call produced.

    Attributes:
        records: One row per evaluated record.
        files: File-level aggregates, the input to every statistic.
        families: Per-family aggregate tables.
        stats: Paired statistics per family plus ``"pooled"``.
        exclusions: Why records were dropped from the physics metric.
        payload: The exact dict serialised to ``metrics.json``.
    """

    records: list[dict[str, Any]] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    families: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    exclusions: dict[str, int] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


def _mean(values: list[float]) -> float:
    """Mean that ignores ``None`` and NaN, returning NaN when nothing is usable.

    Args:
        values: Candidate values.

    Returns:
        The arithmetic mean, or ``float("nan")``.
    """
    usable = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(usable)) if usable else float("nan")


def _exclusion_reason(fov: Any, bank: ForwardBank) -> str | None:
    """Classify why a record cannot enter the physics metric.

    Args:
        fov: The record's ``fov_px`` (possibly the unknown sentinel).
        bank: The forward bank, consulted for its supported-FOV domain.

    Returns:
        A reason label, or ``None`` when the record is supported.
    """
    if bank.supports(fov):
        return None
    if fov is None or fov == UNKNOWN_FOV:
        return _REASON_UNKNOWN_FOV
    return _REASON_NO_PADDING


def _predict(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    grid: int,
    in_channels: int,
    bank: ForwardBank,
) -> dict[str, torch.Tensor]:
    """Run the model and the physics leg over one batch.

    Args:
        model: The network.
        batch: A collated shaping batch.
        device: Device the model lives on.
        grid: Grid side length.
        in_channels: Network input channels.
        bank: Forward bank for the physics leg.

    Returns:
        ``pred_phasor`` ``(B,2,g,g)``, ``pred_img`` ``(B,1,g,g)`` and the
        ``(B,)`` bool ``excluded`` mask.
    """
    images = network_input(batch, grid, in_channels).to(device)
    pred_phasor = unit_phasor(model(images))
    fov = batch["fov_px"]
    if torch.is_tensor(fov):
        fov = fov.to(device)
    pred_img, excluded = bank.forward(pred_phasor, fov)
    return {"pred_phasor": pred_phasor, "pred_img": pred_img, "excluded": excluded}


def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise Pearson correlation, NaN-safe on constant rows.

    Args:
        a: ``(B, N)`` predictions.
        b: ``(B, N)`` references.

    Returns:
        ``(B,)`` correlations, NaN where either row is constant.
    """
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1) * b.norm(dim=1)
    return torch.where(den > 1e-12, num / den.clamp(min=1e-12), torch.full_like(num, float("nan")))


def _collect(
    model: torch.nn.Module,
    loader: Any,
    bank: ForwardBank,
    device: torch.device,
    grid: int,
    in_channels: int,
    mean_phasor: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Score every record in ``loader`` against both baselines.

    Args:
        model: The network.
        loader: A DataLoader yielding shaping batches.
        bank: Forward bank, used identically for the model and the flat baseline.
        device: Device the model lives on.
        grid: Grid side length.
        in_channels: Network input channels.
        mean_phasor: ``(1,2,g,g)`` train-mean phasor baseline.

    Returns:
        The per-record rows and the exclusion-reason counts.
    """
    flat = torch.zeros((1, 2, grid, grid), device=device)
    flat[:, 0] = 1.0  # (cos, sin) = (1, 0): a flat, uncorrected pupil.
    flat_excluded: torch.Tensor | None = None

    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = defaultdict(int)
    offset = 0

    for batch in loader:
        size = int(batch["image"].shape[0])
        out = _predict(model, batch, device, grid, in_channels, bank)
        pred_phasor = out["pred_phasor"]
        pred_img = out["pred_img"]
        excluded = out["excluded"]

        fov = batch["fov_px"]
        fov_values = fov.tolist() if torch.is_tensor(fov) else list(fov)
        image = batch["image"].to(device=pred_img.device, dtype=pred_img.dtype)
        target_phasor = torch.cat(
            [batch["phase_cos"], batch["phase_sin"]], dim=1
        ).to(device=pred_phasor.device, dtype=pred_phasor.dtype)

        # Physics metric, model and flat baseline through the SAME bank.
        flat_img, flat_excluded = bank.forward(flat.expand(size, -1, -1, -1), fov)
        pred_norm = _peak_normalise(pred_img)
        flat_norm = _peak_normalise(flat_img)
        image_norm = _peak_normalise(image)
        per_record = (pred_norm - image_norm).abs().mean(dim=(1, 2, 3))
        flat_per_record = (flat_norm - image_norm).abs().mean(dim=(1, 2, 3))
        correlation = _pearson(pred_norm.flatten(1), image_norm.flatten(1))

        model_sup = (~excluded).cpu()
        flat_sup = (~flat_excluded).cpu()

        for index in range(size):
            fov_value = fov_values[index]
            reason = _exclusion_reason(fov_value, bank)
            supported = bool(model_sup[index])
            if reason is not None:
                reasons[reason] += 1
            rows.append(
                {
                    "index": offset + index,
                    "path": str(batch["path"][index]),
                    "family": str(batch["family"][index]),
                    "fov_px": int(fov_value),
                    "physics_supported": supported,
                    "physics_exclusion_reason": reason,
                    "phasor_mae": float(
                        (pred_phasor[index] - target_phasor[index]).abs().mean()
                    ),
                    "phasor_mae_baseline": float(
                        (mean_phasor - target_phasor[index]).abs().mean()
                    ),
                    "physics_l1": float(per_record[index]) if supported else None,
                    "physics_l1_baseline": (
                        float(flat_per_record[index]) if bool(flat_sup[index]) else None
                    ),
                    "pearson_r": (
                        float(correlation[index])
                        if supported and correlation[index] == correlation[index]
                        else None
                    ),
                }
            )
        offset += size
    return rows, dict(reasons)


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-record rows into per-FILE means.

    This is the step that makes the statistics legitimate: the unit of
    replication is the file, not the record.

    Args:
        records: Per-record rows.

    Returns:
        One aggregate row per distinct ``path``.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["path"]].append(row)

    files: list[dict[str, Any]] = []
    for path, rows in grouped.items():
        entry: dict[str, Any] = {
            "path": path,
            "family": rows[0]["family"],
            "n_records": len(rows),
        }
        for metric in _METRICS:
            entry[metric] = _mean([r[metric] for r in rows])
            entry[f"{metric}_baseline"] = _mean([r[f"{metric}_baseline"] for r in rows])
        files.append(entry)
    return files


def _family_table(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Roll file-level aggregates up into per-family means.

    Args:
        files: Per-file rows.

    Returns:
        One row per family, sorted by name.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in files:
        grouped[row["family"]].append(row)

    table: list[dict[str, Any]] = []
    for family, rows in sorted(grouped.items()):
        entry: dict[str, Any] = {"family": family, "n_files": len(rows)}
        for metric in _METRICS:
            entry[metric] = _mean([r[metric] for r in rows])
            entry[f"{metric}_baseline"] = _mean([r[f"{metric}_baseline"] for r in rows])
        table.append(entry)
    return table


def _paired_rows(files: list[dict[str, Any]], metric: str) -> list[float]:
    """Per-file paired differences for one metric.

    Args:
        files: Per-file rows.
        metric: Metric name.

    Returns:
        ``baseline - model`` for every file where both values are usable. Empty
        when nothing is usable, which is a legitimate outcome (a metric may be
        unavailable for an entire family) and is reported as such.
    """
    diffs: list[float] = []
    for row in files:
        model_value = row.get(metric)
        base_value = row.get(f"{metric}_baseline")
        if model_value is None or base_value is None:
            continue
        if math.isnan(model_value) or math.isnan(base_value):
            continue
        diffs.append(float(base_value) - float(model_value))
    return diffs


def _verdict(p_value: float, min_p: float, n_pairs: int) -> str:
    """State a significance verdict, or explicitly decline to.

    Args:
        p_value: The exact sign-flip p-value.
        min_p: The smallest p this many pairs could produce.
        n_pairs: Number of file-level pairs.

    Returns:
        A human-readable verdict.
    """
    if n_pairs < 2:
        return "n/a: fewer than two files, no inference possible"
    if min_p >= 0.05:
        return (
            f"n/a: {n_pairs} files cannot reach p<0.05 "
            f"(structural floor {min_p:.4g}); report the effect size only"
        )
    if math.isnan(p_value):
        return "n/a: p-value undefined"
    if p_value < 0.05:
        return f"significant (p={p_value:.4g} < 0.05)"
    return f"not significant (p={p_value:.4g} >= 0.05)"


def _stats_block(files: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Build the statistics block for one group of files.

    Args:
        files: Per-file rows for the group.
        label: Group label (a family name or ``"pooled"``).

    Returns:
        A dict keyed by metric, each holding the paired comparison plus an
        explicit verdict.
    """
    block: dict[str, Any] = {"label": label, "n_files": len(files), "metrics": {}}
    for metric in _METRICS:
        diffs = _paired_rows(files, metric)
        if not diffs:
            block["metrics"][metric] = {
                "n_pairs": 0,
                "verdict": "n/a: no file had both a model and a baseline value",
                "min_attainable_p": None,
                "p_sign_flip": None,
                "cohens_dz": None,
                "mean_diff": None,
            }
            continue
        summary = paired_comparison(diffs, label=f"{label}/{metric}")
        summary = dict(summary)
        summary["verdict"] = _verdict(
            float(summary["p_sign_flip"]),
            float(summary["min_attainable_p"]),
            int(summary["n_pairs"]),
        )
        summary["min_attainable_p"] = min_attainable_pvalue(int(summary["n_pairs"]))
        block["metrics"][metric] = summary
    return block


def evaluate(
    cfg: ShapingConfig,
    checkpoint: str | Path,
    *,
    device_spec: str | None = None,
) -> EvaluationResult:
    """Score a checkpoint against both baselines.

    Args:
        cfg: Configuration; supplies grid, channels and corpus roots.
        checkpoint: Path to a checkpoint written by :mod:`ml.shaping.train`.
        device_spec: Torch device string. ``None`` resolves ``cfg.device``
            through the same ``auto -> cuda if available`` rule the trainer uses.

    Returns:
        An :class:`EvaluationResult`.

    Raises:
        FileNotFoundError: If ``checkpoint`` does not exist.
        KeyError: If the checkpoint carries no ``model_state``.
    """
    target = Path(checkpoint)
    if not target.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {target}. Train one first, e.g. "
            f"`python -m ml.shaping.train --checkpoint-dir logs/shaping`."
        )
    if device_spec is None:
        device_spec = cfg.device
    if device_spec == "auto":
        device_spec = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_spec)
    payload = torch.load(target, map_location=device, weights_only=False)
    state = payload.get("model_state")
    if state is None:
        raise KeyError(f"checkpoint {target} has no 'model_state' entry")

    # The checkpoint stores the config it was trained with, so the model arity
    # is recovered from the checkpoint rather than from whatever flags the user
    # passes here -- otherwise evaluating with a default `--in-channels 2` a
    # checkpoint trained with `drop_exposure_channel` would fail to load, or
    # worse, load and feed the wrong tensor.
    model_config = (
        replace(cfg, in_channels=1) if cfg.drop_exposure_channel else cfg
    )
    model = build_phasor_unet(device=device_spec, config=model_config)
    model.load_state_dict(state)
    model.eval()

    bank = ForwardBank(
        grid=cfg.grid,
        device=device_spec,
        forward_checkpoint=cfg.forward_checkpoint,
    )

    train_loader, val_loader, test_loader = build_shaping_loaders(
        cfg.roots,
        config=MaterialiserConfig(grid=cfg.grid, image_mode=cfg.image_mode),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        limit_files=cfg.limit_files,
        index_cache=cfg.index_cache,
        use_cache=True,
        families=cfg.families,
        sources=cfg.sources,
        # Evaluation reads splits sequentially; balancing exists to shape the
        # training gradient and would only corrupt the train-mean baseline.
        balance_alpha=0.0,
    )
    in_channels = model_config.in_channels

    # Train-mean phasor baseline: the per-pixel mean of the training split's
    # unit phasors. Computed once, on the train loader only.
    accumulator = torch.zeros((2, cfg.grid, cfg.grid), dtype=torch.float64, device=device)
    samples = 0
    with torch.no_grad():
        for batch in train_loader:
            target_phasor = torch.cat(
                [batch["phase_cos"], batch["phase_sin"]], dim=1
            ).to(device)
            accumulator += unit_phasor(target_phasor).sum(dim=0).double()
            samples += int(target_phasor.shape[0])
    mean_phasor = (accumulator / max(1, samples)).float()[None]
    logger.info(
        "train-mean phasor baseline computed from {} training record(s)", samples
    )

    # Evaluation only: no autograd graph is needed here, and building one makes
    # every `float(tensor)` below warn about detaching a requires_grad value.
    all_records: list[dict[str, Any]] = []
    reasons: dict[str, int] = defaultdict(int)
    with torch.no_grad():
        for split, loader in (("val", val_loader), ("test", test_loader)):
            rows, split_reasons = _collect(
                model, loader, bank, device, cfg.grid, in_channels, mean_phasor
            )
            for row in rows:
                row["split"] = split
            all_records.extend(rows)
            for key, value in split_reasons.items():
                reasons[key] += value

    files = _aggregate(all_records)
    families = _family_table(files)

    stats: dict[str, Any] = {"pooled": _stats_block(files, "pooled")}
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in files:
        by_family[row["family"]].append(row)
    for family, rows in sorted(by_family.items()):
        stats[family] = _stats_block(rows, family)

    # Holm-Bonferroni across families, per metric, so a 7-row table cannot
    # produce a "winner" by chance.
    for metric in _METRICS:
        raw = {
            family: float(block["metrics"][metric]["p_sign_flip"])
            for family, block in stats.items()
            if family != "pooled"
            and block["metrics"][metric].get("p_sign_flip") is not None
        }
        adjusted = holm_bonferroni(raw)
        for family, value in adjusted.items():
            stats[family]["metrics"][metric]["p_holm"] = value

    payload_out: dict[str, Any] = {
        "diff_convention": DIFF_CONVENTION,
        "checkpoint": str(target),
        "checkpoint_epoch": payload.get("epoch"),
        "config": dataclasses.asdict(cfg),
        "device": device_spec,
        # Which corpus produced these numbers. Recorded because a stale index is
        # not hypothetical here: the shipped cache once described 261 pickles
        # while the directory held 165, and the rebuild changed the record count
        # by 26% and the family count by 4x. A number without this block cannot
        # be attributed to a dataset.
        "corpus": {
            "train": corpus_summary(train_loader),
            "val": corpus_summary(val_loader),
            "test": corpus_summary(test_loader),
            "preprocessing": {
                "image_mode": cfg.image_mode,
                "balance_alpha": cfg.balance_alpha,
                "max_file_repeats": cfg.max_file_repeats,
                "families": list(cfg.families) if cfg.families else None,
                "sources": list(cfg.sources) if cfg.sources else None,
                "drop_exposure_channel": cfg.drop_exposure_channel,
            },
        },
        "records": all_records,
        "files": files,
        "families": families,
        "stats": stats,
        "physics_excluded": dict(reasons),
        "noise_floor_note": (
            "The identical configuration gives validation R2 anywhere in "
            "0.748-0.937 purely by changing the split seed. Only the paired "
            "per-file differences above are informative; absolute values are "
            "not rankable."
        ),
        "family_caveat_note": (
            "The corpus's families are perfectly collinear with their phase "
            "representation, so no cross-family number here is a claim about "
            "which representation is better."
        ),
        "min_attainable_p_reminder": (
            "min_attainable_pvalue(n) = 2/2**n; a protocol with few files "
            "structurally cannot reach p<0.05."
        ),
    }
    return EvaluationResult(
        records=all_records,
        files=files,
        families=families,
        stats=stats,
        exclusions=dict(reasons),
        payload=payload_out,
    )


def _fmt(value: Any, spec: str = ".5f") -> str:
    """Format a metric for the console, degrading to ``n/a``.

    Args:
        value: The value or ``None``.
        spec: Format spec for a finite float.

    Returns:
        A display string.
    """
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(number):
        return "n/a"
    return format(number, spec)


def _print_summary(result: EvaluationResult) -> None:
    """Print the per-family console table.

    Args:
        result: The evaluation result.
    """
    logger.info("=" * 78)
    logger.info("inverse beam shaping -- evaluation")
    logger.info("diff convention: {}", DIFF_CONVENTION)
    logger.info("=" * 78)
    header = (
        f"{'family':<24}{'nfile':>6}{'phasor_mae':>14}{'baseline':>12}"
        f"{'p_phy':>10}{'min_p':>9}  verdict"
    )
    logger.info(header)
    logger.info("-" * 78)
    for family in result.families:
        name = family["family"]
        block = result.stats.get(name, {}).get("metrics", {}).get("phasor_mae", {})
        p_value = block.get("p_sign_flip")
        min_p = block.get("min_attainable_p")
        logger.info(
            "{:<24}{:>6}{:>14}{:>12}{:>10}{:>9}  {}",
            name[:24],
            family["n_files"],
            _fmt(family.get("phasor_mae")),
            _fmt(family.get("phasor_mae_baseline")),
            _fmt(p_value, ".4g") if p_value is not None else "n/a",
            _fmt(min_p, ".3g") if min_p is not None else "n/a",
            block.get("verdict", "n/a"),
        )
    pooled = result.stats.get("pooled", {}).get("metrics", {})
    for metric in _METRICS:
        block = pooled.get(metric, {})
        logger.info(
            "pooled {}: n_pairs={} mean_diff={} dz={} p={} min_p={} | {}",
            metric,
            block.get("n_pairs"),
            _fmt(block.get("mean_diff")),
            _fmt(block.get("cohens_dz"), ".3f"),
            _fmt(block.get("p_sign_flip"), ".4g"),
            _fmt(block.get("min_attainable_p"), ".3g"),
            block.get("verdict", "n/a"),
        )
    if result.exclusions:
        logger.info(
            "physics metric unavailable for {} record(s): {}",
            sum(result.exclusions.values()),
            dict(result.exclusions),
        )
    logger.info(
        "noise floor: {}",
        result.payload.get("noise_floor_note"),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser; defaults come from :class:`ShapingConfig`.

    Returns:
        The configured parser.
    """
    cfg = ShapingConfig()
    parser = argparse.ArgumentParser(
        prog="python -m ml.shaping.evaluate",
        description=(
            "Evaluate the inverse beam-shaping U-Net against per-file paired "
            "baselines."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="path to last.pt")
    parser.add_argument("--out", default=None, help="metrics.json destination")
    parser.add_argument("--grid", type=int, default=cfg.grid)
    parser.add_argument("--in-channels", type=int, default=cfg.in_channels)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--limit-files", type=int, default=cfg.limit_files)
    parser.add_argument("--checkpoint-dir", default=cfg.checkpoint_dir)
    # Read from the dataclass, NOT a literal "cpu": a hardcoded default here is
    # exactly the drift this parser is written to prevent (the `train_amp`
    # parser shipped `default=1` against a calibrated value of 10). Evaluation
    # honours the same device resolution as training, so `auto` picks CUDA when
    # it is available.
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default=cfg.device
    )
    parser.add_argument("--forward-checkpoint", default=cfg.forward_checkpoint)
    # The corpus preprocessing is part of the model's definition: evaluating an
    # `abs255`-trained checkpoint over `peak` frames measures the preprocessing
    # mismatch, not the model. These flags exist so evaluation can reproduce the
    # training run exactly.
    parser.add_argument(
        "--image-mode",
        choices=IMAGE_MODES,
        default=cfg.image_mode,
        help="frame encoding; must match the training run (abs255 default)",
    )
    parser.add_argument(
        "--families",
        type=parse_csv_tuple,
        default=cfg.families,
        help="comma-separated family allowlist; must match training",
    )
    parser.add_argument(
        "--sources",
        type=parse_csv_tuple,
        default=cfg.sources,
        help="comma-separated phase-source allowlist; must match training",
    )
    parser.add_argument(
        "--drop-exposure-channel",
        action="store_true",
        default=cfg.drop_exposure_channel,
        help="evaluate the 1-channel ablation model; must match training",
    )
    parser.add_argument(
        "--index-cache",
        default=cfg.index_cache,
        help="pin the corpus to the same JSON index snapshot used for training; "
        "a score computed over a different corpus is not comparable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` on success, ``2`` when no checkpoint is available (a usage-level
        problem, reported as a message rather than a traceback).
    """
    args = _build_parser().parse_args(argv)
    cfg = ShapingConfig(
        grid=args.grid,
        in_channels=args.in_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        limit_files=args.limit_files,
        checkpoint_dir=args.checkpoint_dir,
        forward_checkpoint=args.forward_checkpoint,
        image_mode=args.image_mode,
        families=args.families,
        sources=args.sources,
        drop_exposure_channel=args.drop_exposure_channel,
        index_cache=args.index_cache,
    )
    try:
        result = evaluate(cfg, args.checkpoint, device_spec=args.device)
    except FileNotFoundError as exc:
        logger.error("{}", exc)
        logger.error(
            "nothing to evaluate. Train first, e.g. "
            "`python -m ml.shaping.train --checkpoint-dir {}`.",
            args.checkpoint_dir,
        )
        return 2

    destination = (
        Path(args.out) if args.out else Path(cfg.checkpoint_dir) / "metrics.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.payload, indent=2), encoding="utf-8")
    _print_summary(result)
    logger.info("wrote {}", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())