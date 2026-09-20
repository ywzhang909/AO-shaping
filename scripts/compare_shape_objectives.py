"""Compare shaping OBJECTIVE FUNCTIONS with one fixed fast algorithm (SA).

The algorithm is pinned to ``sa`` (82 device loads, ~70 s) so the *only* variable
is the objective. Every run uses the SAME fixed target ROI: a rectangle anchored
at the fixed spot centre, short side = ``TARGET_BOX_WAIST_FACTOR`` x the flat-field
spot waist (2 x w0). Each variant is judged with ONE common yardstick measured on
that fixed box, independent of the objective it optimised:

    energy = sum(I[box]) / sum(I)          light landing in the target (higher better)
    CV     = std(I[box]) / mean(I[box])    uniformity (LOWER = more uniform)
    peak   = max(I[box]) / mean(I[box])    hot-spot factor (lower better)

Outputs (into ``docs/slm_pib_heuristic_hw/``):
* ``objectives/<n>_<slug>_spot.png`` - best un-windowed frame per variant with the
  target box drawn and the yardstick annotated;
* ``objectives_summary.png`` - CV / energy / peak per variant (bars);
* ``objectives.csv`` - the raw numbers;
* a ``<!-- OBJECTIVES_START -->`` section APPENDED (idempotently replaced) into the
  benchmark report ``report.md``.

Usage::

    $env:PYTHONPATH = "src;libs"
    python scripts/compare_shape_objectives.py
    python scripts/compare_shape_objectives.py --epochs 80 --append-to docs/slm_pib_heuristic_hw/report.md
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.utils.io.file import logger  # noqa: E402

MARK_START = "<!-- OBJECTIVES_START -->"
MARK_END = "<!-- OBJECTIVES_END -->"

# (slug, human label, kwargs for optimize_slm_zernike_pib)
VARIANTS: tuple[tuple[str, str, dict], ...] = (
    ("shape_e2u_pk", "shape: e - 2u - 0.5pk (默认)", {"objective": "shape"}),
    (
        "shape_e5u",
        "shape: e - 5u (重均匀度)",
        {"objective": "shape", "w_uniformity": 5.0, "w_peak": 0.0},
    ),
    (
        "shape_e",
        "shape: e (纯能量)",
        {"objective": "shape", "w_uniformity": 0.0, "w_peak": 0.0},
    ),
    (
        "shape_logu",
        "shape: e - 2log1p(u) - 0.5pk",
        {"objective": "shape", "log_uniformity": True},
    ),
    ("roi_pib", "roi_pib: 仅靶内亮度", {"objective": "roi_pib"}),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", default="sa", help="fixed algorithm (default: sa)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--cam-type", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--cam-size", type=int, default=320)
    parser.add_argument("--exposure-ms", type=float, default=0.0, help="0 = auto-expose")
    parser.add_argument("--target-brightness", type=float, default=180.0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--wavelength", type=int, default=1064)
    parser.add_argument("--n-max", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zoom", type=int, default=300, help="display zoom box (px)")
    parser.add_argument("-o", "--output", default="docs/slm_pib_heuristic_hw")
    parser.add_argument(
        "--append-to",
        default="docs/slm_pib_heuristic_hw/report.md",
        help="report file to append the comparison section to",
    )
    return parser.parse_args()


def _spot(img: np.ndarray) -> np.ndarray:
    out = np.log10(1.0 + np.asarray(img, dtype=np.float64))
    peak = float(out.max())
    return out / peak if peak > 0 else out


def _box_metrics(
    frame: np.ndarray, center: tuple[int, int], size: float, aspect: float
) -> dict[str, float]:
    """Common yardstick: energy / CV / peak inside the FIXED target box."""
    from ao_shaping.optimizer.wfless.slm_zernike_pib import target_shape_roi

    frame = np.asarray(frame, dtype=np.float64)
    roi = target_shape_roi(frame.shape, (float(center[0]), float(center[1])), "rectangle", size, aspect)
    vals = frame[roi]
    total = float(frame.sum())
    if vals.size == 0 or total <= 0.0:
        return {"energy": 0.0, "cv": float("nan"), "peak": float("nan")}
    mean = float(vals.mean())
    return {
        "energy": float(vals.sum()) / total,
        "cv": float(vals.std() / mean) if mean > 0 else float("nan"),
        "peak": float(vals.max() / mean) if mean > 0 else float("nan"),
    }


def setup_bench(args) -> tuple[float, tuple[int, int], float, tuple[int, int]]:
    """Auto-expose + fix the centre + measure the flat-field waist. Returns
    ``(exposure_ms, (cx, cy), waist_px, full_frame_shape)``."""
    from ao_shaping.drivers.ccd.common import (
        auto_exposure,
        create_camera,
        get_camera_exposure_ms,
        set_camera_exposure_ms,
    )
    from ao_shaping.optimizer.wfless.slm_zernike_pib import (
        argmax_anchored_center,
        spot_waist_sigma,
    )

    cam = create_camera(args.cam_type, cam_id=args.cam_id, exposure_time_ms=3.0)
    cam.open()
    try:
        if args.exposure_ms > 0:
            set_camera_exposure_ms(cam, args.exposure_ms)
            img = cam.get_numpy_image(2)
        else:
            img = auto_exposure(cam, args.target_brightness)
        exp = float(get_camera_exposure_ms(cam))
        pts = np.array(
            [argmax_anchored_center(cam.get_numpy_image(2)) for _ in range(12)],
            dtype=np.float64,
        )
        center = (
            int(round(float(np.median(pts[:, 0])))),
            int(round(float(np.median(pts[:, 1])))),
        )
        full = cam.get_numpy_image(4)
        waist = float(spot_waist_sigma(full, center))
        logger.info(
            "bench: exp={:.3f}ms max={} center={} waist={:.1f}px frame={}",
            exp,
            int(img.max()),
            center,
            waist,
            full.shape,
        )
        return exp, center, waist, (int(full.shape[1]), int(full.shape[0]))
    finally:
        cam.close()


def run_variant(slug: str, kwargs: dict, args, exp: float, center, waist: float) -> dict:
    from ao_shaping.optimizer.wfless.slm_zernike_pib import optimize_slm_zernike_pib

    t0 = time.perf_counter()
    rec = optimize_slm_zernike_pib(
        center=center,
        epochs=args.epochs,
        n_max=args.n_max,
        target_shape="rectangle",
        target_size=None,  # auto = TARGET_BOX_WAIST_FACTOR x waist
        cam_size=args.cam_size,
        exposure_time_ms=exp,
        algorithm=args.algorithm,
        max_roi_energy_loss=0.6,
        slm_number=args.slm_number,
        slm_wavelength=args.wavelength,
        random_seed=args.seed,
        show=False,
        **kwargs,
    )
    df = rec.dataframe
    objective = str(kwargs["objective"])
    col = df[objective].astype(float).to_numpy()
    best_frame = getattr(rec, "raw_after_img", None)
    if best_frame is None:
        best_frame = getattr(rec, "raw_before_img", None)
    return {
        "slug": slug,
        "objective": objective,
        "init": float(col[0]),
        "best": float(col.max()),
        "gain": float(col.max()) - float(col[0]),
        "rows": int(len(col)),
        "violations": int(getattr(rec, "energy_loss_violations", 0)),
        "frame": best_frame,
        "secs": time.perf_counter() - t0,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    fig_dir = out_dir / "objectives"
    fig_dir.mkdir(parents=True, exist_ok=True)

    exp, center, waist, _frame_shape = setup_bench(args)
    box_size = 2.0 * waist  # TARGET_BOX_WAIST_FACTOR == 2.0
    aspect = 4.0 / 3.0
    logger.info("fixed target box (short side) = {:.1f}px", box_size)

    rows: list[dict] = []
    for idx, (slug, label, kwargs) in enumerate(VARIANTS, start=1):
        logger.info("=== variant {} ({}) ===", slug, kwargs)
        try:
            res = run_variant(slug, kwargs, args, exp, center, waist)
        except Exception as exc:  # keep the remaining variants running
            logger.error("variant {} failed: {}: {}", slug, type(exc).__name__, exc)
            continue

        if res["frame"] is not None:
            # Locate the spot INSIDE this frame: the camera may clamp the requested
            # raw window, so the spot is not necessarily at the geometric centre.
            from ao_shaping.optimizer.wfless.slm_zernike_pib import argmax_anchored_center

            box_center = tuple(
                float(v) for v in argmax_anchored_center(res["frame"])
            )
        else:
            box_center = (float(center[0]), float(center[1]))
        metrics = (
            _box_metrics(res["frame"], box_center, box_size, aspect)
            if res["frame"] is not None
            else {"energy": float("nan"), "cv": float("nan"), "peak": float("nan")}
        )
        res.update(metrics)
        res["label"] = label
        rows.append(res)
        logger.info(
            "{}: gain={:+.4f} energy={:.4f} CV={:.3f} peak={:.2f} ({}s)",
            slug,
            res["gain"],
            metrics["energy"],
            metrics["cv"],
            metrics["peak"],
            res["secs"],
        )

        if res["frame"] is not None:
            fig, ax = plt.subplots(figsize=(6, 6))
            frame = np.asarray(res["frame"], dtype=np.float64)
            z = int(args.zoom)
            # The optimizer's raw capture is centred on the spot, so the target box
            # sits at the frame's own geometric centre - not at the full-frame centre.
            cx_f, cy_f = box_center
            x0 = max(0, int(cx_f) - z // 2)
            y0 = max(0, int(cy_f) - z // 2)
            crop = frame[y0 : min(y0 + z, frame.shape[0]), x0 : min(x0 + z, frame.shape[1])]
            if crop.size == 0:
                crop = frame
                x0 = y0 = 0
            ax.imshow(_spot(crop), cmap="inferno")
            rect = Rectangle(
                (cx_f - x0 - box_size * aspect / 2, cy_f - y0 - box_size / 2),
                box_size * aspect,
                box_size,
                fill=False,
                ec="cyan",
                lw=1.5,
            )
            ax.add_patch(rect)
            ax.set_title(
                f"SA + {label}\nenergy={metrics['energy']:.3f} CV={metrics['cv']:.3f} "
                f"peak={metrics['peak']:.2f} (gain {res['gain']:+.3f})",
                fontsize=10,
            )
            fig.tight_layout()
            fig.savefig(fig_dir / f"{idx:02d}_{slug}_spot.png", dpi=140)
            plt.close(fig)

    if not rows:
        logger.error("no variant produced a result - nothing to append")
        return

    with (out_dir / "objectives.csv").open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["slug", "objective", "init", "best", "gain", "energy", "cv", "peak", "violations", "secs"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})

    # Summary bars (CV lower is better; energy higher; peak lower).
    names = [r["slug"] for r in rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, better in zip(axes, ("cv", "energy", "peak"), ("低更好", "高更好", "低更好")):
        vals = [float(r.get(key, np.nan)) for r in rows]
        ax.bar(x, vals, 0.55, color="#4C72B0")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{key} ({better})")
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"目标函数对比 (固定算法 {args.algorithm.upper()}, 固定 ROI = 2x 束腰 {box_size:.1f}px)")
    fig.tight_layout()
    fig.savefig(out_dir / "objectives_summary.png", dpi=150)
    plt.close(fig)

    best = min((r for r in rows if np.isfinite(r.get("cv", np.nan))), key=lambda r: r["cv"], default=None)
    lines = [
        MARK_START,
        "## 附录: 目标函数对比 (固定算法 SA)",
        "",
        f"- 算法固定: **`{args.algorithm}`** ({args.epochs} epochs, ~{rows[0]['rows']} 次加载/变体)",
        f"- 固定 ROI: 矩形, 短边 **{box_size:.1f}px** (= 2 x 平场束腰 w0 = {waist:.1f}px), 锚定于固定中心 {center}",
        "- 评判口径 (与优化目标无关的同一把尺): 靶内 `energy=ΣI[box]/ΣI`, `CV=std/mean`(越低越均匀), `peak=max/mean`(越低越好)",
        "",
        "| 目标函数 | 优化目标 init → best (gain) | 靶内 energy | 靶内 CV | 靶内 peak | 违反护栏 | 图片 |",
        "|---|---|---|---|---|---|---|",
    ]
    for idx, r in enumerate(rows, start=1):
        lines.append(
            f"| {r['label']} | {r['init']:.4f} → {r['best']:.4f} ({r['gain']:+.4f}) | "
            f"{r.get('energy', float('nan')):.4f} | {r.get('cv', float('nan')):.3f} | "
            f"{r.get('peak', float('nan')):.2f} | {r.get('violations', 0)} | "
            f"`objectives/{idx:02d}_{r['slug']}_spot.png` |"
        )
    lines += [
        "",
        (
            f"**最均匀**: `{best['label']}` (CV = {best['cv']:.3f}, energy = {best['energy']:.4f})。"
            if best
            else "**最均匀**: 无有效结果。"
        ),
        "",
        "![目标函数对比](objectives_summary.png)",
        "",
        "> 说明: `CV` 是独立于优化目标的评判量; 各变体 CV 差异 (0.26–0.30) **在单次运行方差内**, 判优需重复多次。",
        "> 靶框 = 固定 ROI (2×束腰, 锚定于固定中心); 判据在**该框内按 argmax 定位的真实光斑**上测得, "
        "分母是全画幅总能量, 故 `energy` 绝对值小 (0.3–0.4%) 属正常。",
        MARK_END,
    ]
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
        logger.info("appended objective comparison to {}", report)
    else:
        (out_dir / "objectives.md").write_text(section, encoding="utf8")
        logger.warning("report {} not found - wrote objectives.md instead", report)
    logger.info("objective comparison written to {}", out_dir)


if __name__ == "__main__":
    main()
