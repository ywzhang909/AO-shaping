"""Repeat the objective-function comparison N times and rank by MEDIAN uniformity.

Single runs are not enough: repeating the SAME variant already gave CV 0.354 then
0.302 (see ``scripts/compare_shape_objectives.py``), i.e. the between-variant spread
sits inside the single-run variance. This script runs every objective variant
``--repeats`` times with the algorithm pinned to one fast search, then reports the
**median** (and min/max spread) of the common yardstick measured inside the fixed
target ROI (2 x spot waist, anchored at the fixed centre, spot located by argmax):

    energy = sum(I[box]) / sum(I)      light in the target (higher better)
    CV     = std(I[box]) / mean(I[box])  LOWER = more uniform
    peak   = max(I[box]) / mean(I[box])  lower = fewer hot spots

Outputs (into ``docs/slm_pib_heuristic_hw/``):
* ``objectives_repeats.csv`` - one row per (variant, repeat) plus the medians;
* ``objectives_repeats.png`` - median CV per variant with the repeat min/max as
  error bars (and the energy / peak panels);
* a ``<!-- OBJECTIVES_REPEATS_START -->`` section APPENDED to the report, holding the
  median table and naming the most uniform objective (only when the winner's spread
  does not overlap the runner-up's).

Usage::

    $env:PYTHONPATH = "src;libs"
    python scripts/repeat_shape_objectives.py --repeats 3
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.utils.image.beam_metrics import zero_order_center  # noqa: E402
from ao_shaping.utils.io.file import logger  # noqa: E402

MARK_START = "<!-- OBJECTIVES_REPEATS_START -->"
MARK_END = "<!-- OBJECTIVES_REPEATS_END -->"

OBJ_KEYS = ("init", "best", "gain", "energy", "cv", "peak")
# Per-repeat light-stability columns appended after the objective keys.
LOG_KEYS = ("exposure_ms", "frame_peak", "waist_px", "box_size_px", "center")
# Drop a round whose frame_peak deviates from the median by more than this.
DRIFT_MAX_REL_DEV = 0.25


def _load_sibling_module(name: str, filename: str):
    """Import a sibling script by file path (``scripts/`` is not a package)."""
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_comparison_module():
    """Import ``compare_shape_objectives.py`` to reuse its verified helpers."""
    return _load_sibling_module("compare_shape_objectives", "compare_shape_objectives.py")


def _load_logging_module():
    """Import ``objective_rep_logging.py`` to reuse its drift helpers."""
    return _load_sibling_module("objective_rep_logging", "objective_rep_logging.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--algorithm", default="sa", help="pinned algorithm (default: sa)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--cam-type", default="daheng", help="camera backend (default: daheng)")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--cam-size", type=int, default=320)
    parser.add_argument("--exposure-ms", type=float, default=0.0, help="0 = auto-expose")
    parser.add_argument("--target-brightness", type=float, default=180.0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--wavelength", type=int, default=1064)
    parser.add_argument("--n-max", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", default="docs/slm_pib_heuristic_hw")
    parser.add_argument(
        "--append-to", default="docs/slm_pib_heuristic_hw/report.md"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cam_type != "daheng":
        logger.warning(
            "camera backend '{}' selected - MiiCam is NOT in the optical path, only "
            "Daheng is; pass --cam-type daheng unless you really moved the bench",
            args.cam_type,
        )

    cso = _load_comparison_module()
    rep_log = _load_logging_module()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ao_shaping.optimizer.wfless.slm_zernike_pib import TARGET_BOX_WAIST_FACTOR

    exp, center, waist, _shape = cso.setup_bench(args)
    box_size = float(TARGET_BOX_WAIST_FACTOR) * waist
    aspect = 4.0 / 3.0
    logger.info(
        "fixed target box = {:.1f}px (2x waist {:.1f}px); {} repeats per variant",
        box_size,
        waist,
        args.repeats,
    )

    rows: list[dict] = []
    # Each variant opens and closes its own camera + SLM pair through the
    # optimizer's device context managers, so there is no shared device to hold.
    for idx, (slug, label, kwargs) in enumerate(cso.VARIANTS, start=1):
        for rep in range(1, args.repeats + 1):
            logger.info("=== {} rep {}/{} ===", slug, rep, args.repeats)
            try:
                res = cso.run_variant(slug, kwargs, args, exp, center, waist)
            except Exception as exc:
                logger.error(
                    "{} rep {} failed: {}: {}",
                    slug,
                    rep,
                    type(exc).__name__,
                    exc,
                )
                continue
            frame = res.get("frame")
            box_center = zero_order_center(frame) if frame is not None else center
            metrics = (
                cso._box_metrics(frame, box_center, box_size, aspect)
                if frame is not None
                else {
                    "energy": float("nan"),
                    "cv": float("nan"),
                    "peak": float("nan"),
                }
            )
            res.update(metrics)
            res["label"] = label
            res["rep"] = rep
            res["exposure_ms"] = float(exp)
            res["frame_peak"] = (
                int(np.asarray(frame).max()) if frame is not None else 0
            )
            res["waist_px"] = float(waist)
            res["box_size_px"] = float(box_size)
            res["center"] = f"{float(center[0]):.1f},{float(center[1]):.1f}"
            rows.append(res)
            logger.info(
                "{} rep {}: gain={:+.4f} energy={:.4f} CV={:.3f} peak={:.2f}",
                slug,
                rep,
                res["gain"],
                res["energy"],
                res["cv"],
                res["peak"],
            )

    if not rows:
        logger.error("no run produced a result")
        return

    # The laser drifts, so a round taken during a bright/dim spell is not
    # comparable with the others - drop the clear drift outliers before taking
    # the medians (rounds without a usable peak are kept but tagged unknown).
    kept_rows, dropped_rows = rep_log.flag_drift_reps(
        rows, key="frame_peak", max_rel_dev=DRIFT_MAX_REL_DEV
    )
    if dropped_rows:
        for r in dropped_rows:
            r["drift_outlier"] = True
        logger.warning(
            "drift outliers dropped ({} of {} rounds, frame_peak rel-dev > {:.0%}): {}",
            len(dropped_rows),
            len(rows),
            DRIFT_MAX_REL_DEV,
            [(r["slug"], r["rep"], r["frame_peak"]) for r in dropped_rows],
        )
    else:
        logger.info(
            "no drift outliers: all {} rounds within {:.0%} of the median frame_peak",
            len(rows),
            DRIFT_MAX_REL_DEV,
        )

    kept_by_variant: dict[str, list[dict]] = {}
    for r in kept_rows:
        kept_by_variant.setdefault(r["slug"], []).append(r)

    summary: list[dict] = []
    for idx, (slug, label, _kwargs) in enumerate(cso.VARIANTS, start=1):
        reps = kept_by_variant.get(slug, [])
        if not reps:
            logger.warning("{}: no drift-clean reps left - excluded from the medians", slug)
            continue
        med = {k: float(np.nanmedian([r[k] for r in reps])) for k in OBJ_KEYS}
        spread = {k: float(np.nanmax([r[k] for r in reps]) - np.nanmin([r[k] for r in reps])) for k in OBJ_KEYS}
        summary.append(
            {
                "slug": slug,
                "label": label,
                "n": len(reps),
                **med,
                **{f"{k}_spread": v for k, v in spread.items()},
            }
        )
        logger.info(
            "{}: n={} median CV={:.3f} (+/-{:.3f}) energy={:.4f} peak={:.2f} gain={:+.4f}",
            slug,
            len(reps),
            med["cv"],
            spread["cv"] / 2.0,
            med["energy"],
            med["peak"],
            med["gain"],
        )

    with (out_dir / "objectives_repeats.csv").open("w", newline="", encoding="utf8") as fh:
        fields = [
            "slug",
            "label",
            "rep",
            *OBJ_KEYS,
            *LOG_KEYS,
            "drift_unknown",
            "drift_outlier",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        writer.writerow({})
        for r in summary:
            writer.writerow(
                {
                    "slug": r["slug"],
                    "label": f"MEDIAN(n={r['n']})",
                    "rep": "",
                    **{k: r[k] for k in OBJ_KEYS},
                }
            )

    # Rank by median CV (lower = more uniform); flag when the winner is not separable.
    ranked = sorted(summary, key=lambda r: r["cv"])
    winner = ranked[0] if ranked else None
    separable = False
    if len(ranked) >= 2:
        separable = (ranked[0]["cv"] + ranked[0]["cv_spread"] / 2.0) < (
            ranked[1]["cv"] - ranked[1]["cv_spread"] / 2.0
        )

    names = [r["slug"] for r in summary]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, better in zip(axes, ("cv", "energy", "peak"), ("低更好", "高更好", "低更好")):
        vals = [r[key] for r in summary]
        err = [r[f"{key}_spread"] / 2.0 for r in summary]
        ax.bar(x, vals, 0.55, yerr=err, capsize=4, color="#4C72B0", ecolor="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{key} ({better}) — median of {args.repeats}")
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        f"目标函数对比: 重复 {args.repeats} 次取中位数 (固定 {args.algorithm.upper()}, "
        f"固定 ROI {box_size:.1f}px)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "objectives_repeats.png", dpi=150)
    plt.close(fig)

    lines = [
        MARK_START,
        f"## 附录 B: 目标函数重复实验 (固定 {args.algorithm.upper()}, {args.repeats} 次取中位数)",
        "",
        f"- 每个目标函数变体重复 **{args.repeats} 次**, 表内为**中位数** (括号内为 (max-min)/2)",
        f"- 固定 ROI: 矩形短边 **{box_size:.1f}px** (= 2 x 平场束腰 {waist:.1f}px), 锚定固定中心 {center}",
        "- 判据在框内**按 argmax 定位的真实光斑**上测得 (分母为全画幅总能量, energy 绝对值偏小属正常)",
        f"- 光漂移离群剔除: **{'none' if not dropped_rows else len(dropped_rows)}** 轮 "
        f"(frame_peak 相对中位数偏差 > {DRIFT_MAX_REL_DEV:.0%} 的轮次不参与中位数)",
        "",
        "| 目标函数 | 优化 gain (median) | 靶内 energy | **靶内 CV (越低越均匀)** | 靶内 peak | n |",
        "|---|---|---|---|---|---|",
    ]
    for r in ranked:
        lines.append(
            f"| {r['label']} | {r['gain']:+.4f} | {r['energy']:.4f} (±{r['energy_spread'] / 2:.4f}) | "
            f"**{r['cv']:.3f}** (±{r['cv_spread'] / 2:.3f}) | {r['peak']:.2f} (±{r['peak_spread'] / 2:.2f}) | "
            f"{r['n']} |"
        )
    lines += ["", "![目标函数重复实验](objectives_repeats.png)", ""]
    if winner is not None:
        lines.append(
            f"**最均匀 (median CV 最低)**: `{winner['label']}` CV = {winner['cv']:.3f} "
            f"(±{winner['cv_spread'] / 2:.3f}); "
            + (
                "且与次优的区间**不重叠 → 可判优**。"
                if separable
                else "但与次优区间**重叠 → 仍不可判优, 需更多重复或更大预算**。"
            )
        )
    lines.append(MARK_END)
    section = "\n".join(lines) + "\n"

    report = Path(args.append_to)
    if report.exists():
        text = report.read_text(encoding="utf8")
        if MARK_START in text and MARK_END in text:
            head, rest = text.split(MARK_START, 1)
            _, tail = rest.split(MARK_END, 1)
            text = head + section + tail
        else:
            text = text.rstrip() + "\n\n" + section
        report.write_text(text, encoding="utf8")
        logger.info("appended repeat study to {}", report)
    else:
        (out_dir / "objectives_repeats.md").write_text(section, encoding="utf8")
        logger.warning("report {} not found - wrote objectives_repeats.md instead", report)
    logger.info("repeat study written to {}", out_dir)


if __name__ == "__main__":
    main()
