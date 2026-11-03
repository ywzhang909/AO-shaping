"""Render the 2026-11-03 zero-order diagnosis figure and its artifact JSON.

Offline only: reads the finished zero-order diagnostic capture
(``data/debug/sim2f_zero_order_diag_*``), one flat capture from the archived
hardware session ``data/debug/sim2f_fov_hw_20261103_210641`` and the reference
frame of one refine run, then writes:

* ``figures/zero_order_diagnosis_20261103.png`` -- three panels: the transverse
  profile that shows the two-lobe structure of the 0-order and its 16 px
  spacing, the 30-frame estimator cloud inside one session, and the per-estimator
  repeatability bars;
* ``artifacts/zero_order_diagnosis_20261103.json`` -- every number the report
  quotes from this figure.

No hardware is touched and nothing under ``data/debug`` is written. The figure
is diagnostic evidence about where the 0-order is, not acceptance evidence for
beam shaping.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIAG_DIR = ROOT / "data" / "debug" / "sim2f_zero_order_diag_20261103_215142"
SESSION_DIR = ROOT / "data" / "debug" / "sim2f_fov_hw_20261103_210641"
REFINE_DIR = ROOT / "data" / "debug" / "sim2f_fov_refine_ring_raw16_1103_211433"
DEFAULT_OUT = ROOT / "report" / "shape_trials" / "physical_refine"
HALF = 64
PLT = {"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False}
plt.rcParams.update(PLT)


def _rel(path: Path) -> str:
    """Repository-relative POSIX path when possible, absolute otherwise."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _profile(frame: np.ndarray, cy: int, cx: int, half: int = HALF) -> np.ndarray:
    """Column sums of a ``2*half`` square around the 0-order, normalised to 1."""
    sub = np.asarray(frame, dtype=np.float64)[cy - half : cy + half, cx - half : cx + half]
    col = sub.sum(axis=0)
    return col / float(col.sum())


def _two_lobes(frame: np.ndarray, cy: int, cx: int, half: int = 32) -> list[tuple[int, int, float]]:
    """The two brightest well-separated local maxima of the 0-order in a box.

    Recipe: local maxima of the raw frame (5x5 maximum filter, value above 20
    counts) inside a ``2*half`` square, sorted by value, keeping the first that
    is more than 8 px in x from the brightest.  Two entries mean the 0-order is
    a double-lobed structure rather than a single spot.
    """
    from scipy.ndimage import maximum_filter

    sub = np.asarray(frame, dtype=np.float64)[cy - half : cy + half, cx - half : cx + half]
    mask = (sub == maximum_filter(sub, size=5)) & (sub > 20)
    rows, cols = np.nonzero(mask)
    found = [(cx - half + int(c), cy - half + int(r), float(sub[r, c])) for r, c in zip(rows, cols)]
    found.sort(key=lambda t: -t[2])
    if not found:
        return []
    best = found[0]
    partner = next((t for t in found[1:] if abs(t[0] - best[0]) > 8), None)
    return [best, partner] if partner else [best]


def _flat_frame(manifest: dict, session: Path) -> tuple[np.ndarray, str]:
    """First ``flat`` item of an archived capture session."""
    for name, item in manifest["items"].items():
        if item["kind"] == "flat":
            return np.load(session / item["frame"]), name
    raise SystemExit(f"no flat item in {session}")


def _panel_profiles(
    ax: plt.Axes,
    curves: list[tuple[str, np.ndarray, list[float]]],
) -> dict[str, object]:
    """Panel A: normalised transverse profiles with the measured lobe columns.

    ``curves`` holds ``(label, profile, lobe_offsets_px)``; the offsets are
    relative to the reference column the profiles are plotted against.
    """
    offset = np.arange(-HALF, HALF, dtype=float)
    peaks: dict[str, object] = {}
    for label, profile, lobes in curves:
        (line,) = ax.plot(offset, profile, lw=1.6, label=label)
        for x in lobes:
            ax.axvline(x, color=line.get_color(), ls=":", lw=1.0, alpha=0.8)
        peaks[label] = {
            "lobe_offsets_px": [float(v) for v in lobes],
            "profile_at_lobes": [
                float(profile[int(round(v)) + HALF]) for v in lobes if -HALF <= v < HALF
            ],
        }
    if curves and len(curves[0][2]) == 2:
        left_x, right_x = curves[0][2]
        spacing = abs(right_x - left_x)
        top = float(curves[0][1].max())
        ax.annotate(
            f"双峰间距 {spacing:.0f} px",
            xy=((left_x + right_x) / 2, top * 1.02),
            xytext=((left_x + right_x) / 2, top * 1.28),
            ha="center",
            fontsize=9,
            arrowprops={"arrowstyle": "-[", "color": "0.35", "lw": 0.8},
        )
        ax.set_ylim(0, top * 1.45)
    ax.set_xlabel("相对 0 级参考列 (px)")
    ax.set_ylabel("归一化列和")
    ax.set_title("(a) 0 级横向剖面 (点线 = 实测极大列)", fontsize=10)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.85)
    ax.grid(alpha=0.3)
    return peaks


def _panel_cloud(ax: plt.Axes, per_frame: list[dict]) -> dict[str, object]:
    """Panel B: 30-frame estimator cloud inside one session."""
    names = {
        "argmax_raw": "o",
        "argmax_blurred": "s",
        "centroid_box": "^",
        "centroid_despiked_box": "D",
    }
    stats: dict[str, object] = {}
    for name, marker in names.items():
        pts = np.array([f["estimates"][name] for f in per_frame], dtype=float)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=26,
            marker=marker,
            alpha=0.7,
            facecolors="none",
            edgecolors=f"C{list(names).index(name)}",
            label=name,
        )
        centre = pts.mean(axis=0)
        radii = np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1])
        stats[name] = {
            "mean_xy": [float(centre[0]), float(centre[1])],
            "std_x_px": float(pts[:, 0].std()),
            "std_y_px": float(pts[:, 1].std()),
            "ptp_x_px": float(np.ptp(pts[:, 0])),
            "ptp_y_px": float(np.ptp(pts[:, 1])),
            "radial_std_px": float(radii.std()),
            "radial_ptp_px": float(np.ptp(radii)),
        }
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("(b) 同一次会话内 30 帧估计散布", fontsize=10)
    ax.legend(fontsize=7, loc="center", framealpha=0.85, markerscale=1.4)
    ax.grid(alpha=0.3)
    ax.margins(0.12)
    return stats


def _panel_bars(ax: plt.Axes, summaries: dict[str, dict]) -> dict[str, float]:
    """Panel C: repeatability per estimator, from ``diag.json``."""
    order = [
        "argmax_raw",
        "argmax_despiked",
        "argmax_blurred",
        "centroid_box",
        "centroid_box_halfmax",
        "centroid_despiked_box",
    ]
    labels = [n.replace("_", "\n") for n in order]
    std = [float(summaries[n]["radial_std_px"]) for n in order]
    ptp = [float(summaries[n]["radial_ptp_px"]) for n in order]
    pos = np.arange(len(order))
    ax.bar(pos - 0.2, std, width=0.4, label="径向标准差 (px)")
    ax.bar(pos + 0.2, ptp, width=0.4, label="径向峰峰值 (px)")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("(c) 各估计量的逐帧重复性 (30 帧)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return {n: {"radial_std_px": s, "radial_ptp_px": p} for n, s, p in zip(order, std, ptp)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag-dir", type=Path, default=DIAG_DIR)
    parser.add_argument("--session-dir", type=Path, default=SESSION_DIR)
    parser.add_argument("--refine-dir", type=Path, default=REFINE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    diag = json.loads((args.diag_dir / "diag.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.session_dir / "capture_manifest.json").read_text(encoding="utf-8"))
    zero = [int(v) for v in manifest["zero_order_xy"]]
    cx, cy = zero[0], zero[1]

    flat, flat_name = _flat_frame(manifest, args.session_dir)
    diag_frame = np.load(args.diag_dir / "frames" / "representative_full.npy")
    ring_ref = np.load(args.refine_dir / "frames" / "reference.npy")
    summary = json.loads((args.refine_dir / "refine_summary.json").read_text(encoding="utf-8"))
    ring_zero = [int(v) for v in summary["zero_order_effective_xy"]]

    curves = [
        (
            f"{flat_name} 平相位 21:06 (峰值 {flat.max():.0f})",
            _profile(flat, cy, cx),
            [float(x - cx) for x, _, _ in _two_lobes(flat, cy, cx)],
        ),
        (
            f"diag 平相位 21:51 (峰值 {diag_frame.max():.0f})",
            _profile(diag_frame, cy, cx),
            [float(x - cx) for x, _, _ in _two_lobes(diag_frame, cy, cx)],
        ),
        (
            f"ring 相位参考帧 21:14 (峰值 {ring_ref.max():.0f})",
            _profile(ring_ref, int(ring_zero[1]), int(ring_zero[0])),
            [float(x - ring_zero[0]) for x, _, _ in _two_lobes(ring_ref, int(ring_zero[1]), int(ring_zero[0]))],
        ),
    ]
    lobe_evidence = {
        "flat_2106": _two_lobes(flat, cy, cx),
        "diag_2151": _two_lobes(diag_frame, cy, cx),
        "ring_ref_2114": _two_lobes(ring_ref, int(ring_zero[1]), int(ring_zero[0])),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    peaks = _panel_profiles(axes[0], curves)
    cloud = _panel_cloud(axes[1], diag["per_frame"])
    bars = _panel_bars(axes[2], diag["estimator_summaries"])
    fig.suptitle(
        "0 级定位诊断: 双峰结构、16 px 间距与逐帧重复性 (2026-11-03, 0.182 ms)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    fig_dir = args.out / "figures"
    art_dir = args.out / "artifacts"
    fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "zero_order_diagnosis_20261103.png"
    fig.savefig(fig_path, dpi=150)

    payload = {
        "schema": "sim2f_zero_order_diagnosis_figure_v1",
        "sources": {
            "diag_dir": _rel(args.diag_dir),
            "session_dir": _rel(args.session_dir),
            "flat_item": flat_name,
            "refine_dir": _rel(args.refine_dir),
            "exposure_ms": manifest.get("exposure_ms"),
        },
        "reference_column": {"x": cx, "y": cy, "ring_phase_zero_xy": ring_zero},
        "profile_peaks": peaks,
        "lobe_evidence_xy_value": lobe_evidence,
        "frame_cloud": cloud,
        "estimator_repeatability": bars,
        "from_diag_json": {
            "argmax_distinct_pixels_30_frames": diag["argmax_stability"]["n_distinct_pixels"],
            "median_abs_argmax_vs_centroid_box_px": diag[
                "median_abs_diff_argmax_vs_centroid_box_px"
            ],
            "reopen_open_to_open_spread_px": diag["reopen"]["all_opens_including_initial"][
                "summary"
            ]["radial_spread_px"],
            "peak_over_ring_median": float(
                np.median([f["peak_over_ring"] for f in diag["per_frame"]])
            ),
        },
        "figure": _rel(fig_path),
    }
    art_path = art_dir / "zero_order_diagnosis_20261103.json"
    art_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {fig_path}")
    print(f"wrote {art_path}")
    for label, info in peaks.items():
        print(
            f"  {label}: lobe offsets (px) {info['lobe_offsets_px']}, "
            f"profile {[round(v, 4) for v in info['profile_at_lobes']]}"
        )
    for name, lobes in lobe_evidence.items():
        print(f"  {name}: " + ", ".join(f"x={x} y={y} v={v:.1f}" for x, y, v in lobes))
    for name, info in bars.items():
        print(
            f"  {name:22s} radial std {info['radial_std_px']:.3f} px, "
            f"ptp {info['radial_ptp_px']:.3f} px"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
