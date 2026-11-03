"""Render CCD GIF and diagnostic charts for the 2026-11-02 sim2f hardware session.

Offline only: reads ``capture_manifest.json`` and the raw frames of a completed
session, writes figures under ``--out``. No hardware is touched. The charts are
diagnostics of the captured frames; they are not acceptance evidence.

With ``--refine-dir`` (repeatable) it additionally renders, per refine run, the
measured CCD sequence as a fixed-scale GIF, one gain row per round into
``artifacts/refine_gains_<date>.json`` and a measured-vs-predicted gain chart.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION = ROOT / "data" / "debug" / "sim2f_fov_hw_20261102"
DEFAULT_OUT = ROOT / "report" / "shape_trials" / "physical_refine"
HALF = 128
WINDOW = 256
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _load_refine_targets():
    path = ROOT / "scripts" / "ml" / "sim2f_fov_hw_refine.py"
    spec = importlib.util.spec_from_file_location("sim2f_fov_hw_refine_figs", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _crop(frame: np.ndarray, zero: tuple[int, int]) -> np.ndarray:
    x, y = zero
    return np.asarray(frame[y - HALF : y + HALF, x - HALF : x + HALF], dtype=np.float64)


def _background_removed(crop: np.ndarray) -> np.ndarray:
    body = crop - np.median(crop)
    body[body < 0] = 0.0
    return body


def _centroid(body: np.ndarray) -> tuple[float, float]:
    total = body.sum()
    if total <= 0:
        return float("nan"), float("nan")
    yy, xx = np.mgrid[0 : body.shape[0], 0 : body.shape[1]]
    return float((body * xx).sum() / total), float((body * yy).sum() / total)


def _to_uint8(crop: np.ndarray, vmax: float) -> np.ndarray:
    return (np.clip(crop / vmax, 0.0, 1.0) * 255).astype(np.uint8)


def _render_capture_gif(session: Path, manifest: dict, zero: tuple[int, int], out: Path) -> None:
    frames = []
    labels = []
    for name, item in manifest["items"].items():
        frame = np.load(session / item["frame"])
        crop = _crop(frame, zero)
        frames.append(_to_uint8(crop, 125.0))
        labels.append(f"{item['kind']} {name} ({item.get('condition', '-')})")
    images = []
    for arr, label in zip(frames, labels):
        image = Image.fromarray(arr).resize((288, 288), Image.NEAREST)
        canvas = Image.new("L", (288, 320), 20)
        canvas.paste(image, (0, 0))
        ImageDraw.Draw(canvas).text((6, 296), label[:60], fill=230)
        images.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=64))
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=180,
        loop=0,
        optimize=True,
    )


def _render_shape_gif(session: Path, manifest: dict, zero: tuple[int, int], out: Path) -> None:
    names = [n for n, v in manifest["items"].items() if v["kind"] == "shape"]
    images = []
    for name in names:
        crop = _crop(np.load(session / manifest["items"][name]["frame"]), zero)
        arr = _to_uint8(_background_removed(crop), 20.0)
        image = Image.fromarray(arr).resize((288, 288), Image.NEAREST)
        canvas = Image.new("L", (288, 320), 20)
        canvas.paste(image, (0, 0))
        ImageDraw.Draw(canvas).text((6, 296), name, fill=230)
        images.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=64))
    images[0].save(out, save_all=True, append_images=images[1:], duration=900, loop=0, optimize=True)


def _collect_stats(session: Path, manifest: dict, zero: tuple[int, int]) -> list[dict]:
    rows = []
    for name, item in manifest["items"].items():
        crop = _crop(np.load(session / item["frame"]), zero)
        body = _background_removed(crop)
        cx, cy = _centroid(body)
        rows.append(
            {
                "name": name,
                "kind": item["kind"],
                "group": item.get("group"),
                "condition": item.get("condition"),
                "peak": float(crop.max()),
                "total": float(body.sum()),
                "cx": cx - HALF,
                "cy": cy - HALF,
            }
        )
    return rows


def _render_charts(rows: list[dict], refine, out_dir: Path) -> dict:
    kinds = ["flat", "calib_screen", "probe", "shape"]
    colors = {"flat": "#4C78A8", "calib_screen": "#F58518", "probe": "#54A24B", "shape": "#E45756"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for kind in kinds:
        sub = [r for r in rows if r["kind"] == kind]
        axes[0].scatter([r["peak"] for r in sub], [r["total"] for r in sub], s=18, label=kind, color=colors[kind])
    axes[0].set_xlabel("窗口峰值 (count)")
    axes[0].set_ylabel("去背景总能量 (count)")
    axes[0].set_title("逐帧峰值与能量（按类别）")
    axes[0].legend(fontsize=8)
    flat = [r for r in rows if r["kind"] == "flat"]
    axes[1].scatter([r["cx"] for r in flat], [r["cy"] for r in flat], s=18, color=colors["flat"], label="flat")
    probe = [r for r in rows if r["kind"] == "probe"]
    axes[1].scatter([r["cx"] for r in probe], [r["cy"] for r in probe], s=18, color=colors["probe"], label="probe")
    shape = [r for r in rows if r["kind"] == "shape"]
    axes[1].scatter([r["cx"] for r in shape], [r["cy"] for r in shape], s=28, color=colors["shape"], marker="x", label="shape")
    axes[1].axhline(0, color="#888", lw=0.6)
    axes[1].axvline(0, color="#888", lw=0.6)
    axes[1].set_xlabel("质心 x - 零级 (px)")
    axes[1].set_ylabel("质心 y - 零级 (px)")
    axes[1].set_title("质心相对记录零级的偏移")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "ccd_stats_scatter.png", dpi=150)
    plt.close(fig)

    shape_names = [r["name"] for r in shape if r["name"].endswith("_r0")]
    ee_rows = []
    for name in shape_names:
        target_kind = name.removeprefix("shape_").removesuffix("_nominal_r0")
        target, support = refine._target_mask(target_kind, WINDOW)
        base = [r for r in shape if r["name"] == name][0]
        ee_rows.append((target_kind, base))
    kinds_order = [k for k, _ in ee_rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    fractions = []
    for kind in kinds_order:
        _, support = refine._target_mask(kind, WINDOW)
        fractions.append(float(support.mean()))
    ee = []
    for kind, _ in ee_rows:
        _, support = refine._target_mask(kind, WINDOW)
        values = []
        for suffix in ("r0", "r1"):
            crop = _crop(np.load(_SESSION / _MANIFEST["items"][f"shape_{kind}_nominal_{suffix}"]["frame"]), _ZERO)
            body = _background_removed(crop)
            total = body.sum()
            values.append(float(body[support].sum() / total) if total > 0 else float("nan"))
        ee.append(values)
    ee_arr = np.array(ee)
    x = np.arange(len(kinds_order))
    ax.bar(x - 0.18, ee_arr[:, 0], 0.36, label="r0", color="#4C78A8")
    ax.bar(x + 0.18, ee_arr[:, 1], 0.36, label="r1", color="#F58518")
    ax.plot(x, fractions, "k_", markersize=28, label="支持域面积占比")
    ax.set_xticks(x)
    ax.set_xticklabels(kinds_order)
    ax.set_ylabel("支持域内能量占比")
    ax.set_title("逐目标能量落入支持域（诊断，非验收）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "shape_encircled_energy.png", dpi=150)
    plt.close(fig)
    return {"shape_ee": {k: ee_arr[i].tolist() for i, k in enumerate(kinds_order)}, "support_fraction": dict(zip(kinds_order, fractions))}


REFINE_DATE = "20261103"
REFINE_CANVAS = (288, 344)
REFINE_PREFIX = "sim2f_fov_refine_ring_"
REFINE_TAG_ALIASES = {
    "raw16": "raw16",
    "eff": "effective",
    "effalign": "effalign",
    "gsringcentre": "gsseed",
}
_REFINE_TIMESTAMP_TOKEN = re.compile(r"(20\d{6}|\d{6}|\d{4})")


def _refine_tag(refine_dir: Path) -> str:
    """Short run tag for a refine directory name (``..._ring_raw16_1103_211433`` -> ``raw16``)."""
    stem = refine_dir.name
    for marker in (REFINE_PREFIX, "sim2f_fov_refine_"):
        if stem.startswith(marker):
            stem = stem[len(marker):]
            break
    tokens = [token for token in stem.split("_") if token]
    while tokens and _REFINE_TIMESTAMP_TOKEN.fullmatch(tokens[-1]):
        tokens.pop()
    base = "_".join(tokens)
    alias = REFINE_TAG_ALIASES.get(base.lower())
    if alias is not None:
        return f"{alias}_{REFINE_DATE}"
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", base).strip("_").lower() or "run"
    return f"{cleaned}_{REFINE_DATE}"


def _refine_zero(summary: dict) -> tuple[int, int]:
    request = summary.get("request") or {}
    for candidate in (
        request.get("effective_zero_order_xy"),
        summary.get("zero_order_effective_xy"),
        request.get("source_zero_order_xy"),
        summary.get("zero_order_recorded_xy"),
        summary.get("zero_order_xy"),
    ):
        if isinstance(candidate, (list, tuple)) and len(candidate) == 2:
            return int(candidate[0]), int(candidate[1])
    raise ValueError("refine summary has no zero-order xy")


def _refine_round_row(refine_dir: Path, tag: str, summary: dict, zero: tuple[int, int], round_rec: dict, has_frame: bool) -> dict:
    measured = round_rec.get("measured") or {}
    predicted = round_rec.get("predicted") or {}
    measured_ref = (measured.get("reference") or {}).get("objective")
    measured_cand = (measured.get("candidate") or {}).get("objective")
    predicted_ref = (predicted.get("reference") or {}).get("objective")
    predicted_cand = (predicted.get("candidate") or {}).get("objective")
    request = summary.get("request") or {}
    return {
        "tag": tag,
        "refine_dir": _display_path(refine_dir),
        "round_id": round_rec.get("id"),
        "accepted": bool(round_rec.get("accepted")),
        "status": round_rec.get("status"),
        "has_frame": has_frame,
        "effective_zero_order_xy": [zero[0], zero[1]],
        "measured_min_gain": request.get("measured_min_gain"),
        "measured": {
            "reference_objective": measured_ref,
            "candidate_objective": measured_cand,
            "gain": measured.get("gain"),
            "candidate_ee": (measured.get("candidate") or {}).get("ee"),
        },
        "predicted": {
            "reference_objective": predicted_ref,
            "candidate_objective": predicted_cand,
            "gain": predicted.get("gain"),
        },
    }


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _refine_notes(rows: list[dict]) -> list[str]:
    """Structural warnings readable straight off the summary, empty when the run is regular."""
    notes = []
    if len(rows) >= 2:
        measured_refs = {row["measured"]["reference_objective"] for row in rows}
        predicted_refs = {row["predicted"]["reference_objective"] for row in rows}
        if len(measured_refs) == 1 and len(predicted_refs) == 1:
            notes.append("rounds_not_chained: every round is scored against the same reference, not the previous candidate")
        candidates = {row["measured"]["candidate_objective"] for row in rows}
        if len(candidates) == 1:
            notes.append("measured_objective_constant: the measured score cannot separate the rounds")
    return notes


def _build_refine_run(refine_dir: Path, tag: str, summary: dict, zero: tuple[int, int]) -> dict:
    frames_dir = refine_dir / "frames"
    entries: list[dict] = []
    gaps: list[dict] = []
    reference = summary.get("reference") or {}
    ref_path = frames_dir / "reference.npy"
    ref_entry = {
        "id": "reference",
        "path": ref_path,
        "measured": (reference.get("measured") or {}).get("objective"),
        "predicted": (reference.get("predicted") or {}).get("objective"),
        "gain_measured": None,
        "gain_predicted": None,
        "accepted": None,
    }
    if ref_path.is_file():
        entries.append(ref_entry)
    else:
        gaps.append({"id": "reference", "path": _display_path(ref_path), "reason": "frame file missing"})

    rows = []
    for round_rec in summary.get("rounds") or []:
        round_id = round_rec.get("id")
        if not round_id:
            gaps.append({"id": None, "path": None, "reason": "round without id"})
            continue
        path = frames_dir / f"{round_id}.npy"
        has_frame = path.is_file()
        rows.append(_refine_round_row(refine_dir, tag, summary, zero, round_rec, has_frame))
        if not has_frame:
            gaps.append({"id": round_id, "path": _display_path(path), "reason": "frame file missing"})
            continue
        measured = round_rec.get("measured") or {}
        predicted = round_rec.get("predicted") or {}
        entries.append(
            {
                "id": round_id,
                "path": path,
                "measured": (measured.get("candidate") or {}).get("objective"),
                "predicted": (predicted.get("candidate") or {}).get("objective"),
                "gain_measured": measured.get("gain"),
                "gain_predicted": predicted.get("gain"),
                "accepted": bool(round_rec.get("accepted")),
            }
        )
    return {
        "tag": tag,
        "refine_dir": _display_path(refine_dir),
        "status": summary.get("status"),
        "measured_min_gain": (summary.get("request") or {}).get("measured_min_gain"),
        "zero": zero,
        "entries": entries,
        "gaps": gaps,
        "rounds": rows,
        "notes": _refine_notes(rows),
    }


def _refine_label(entry: dict) -> list[str]:
    if entry["accepted"] is None:
        head = entry["id"]
    else:
        head = f"{entry['id']} {'ACCEPT' if entry['accepted'] else 'REJECT'}"
    lines = [head[:44]]
    measured = entry["measured"]
    predicted = entry["predicted"]
    if measured is not None and predicted is not None:
        lines.append(f"meas obj {measured:.4f} | pred {predicted:.4f}")
    elif measured is not None:
        lines.append(f"meas obj {measured:.4f}")
    if entry["gain_measured"] is not None:
        lines.append(_gain_line(entry["gain_measured"], entry["gain_predicted"]))
    return lines


def _gain_line(measured_gain: float, predicted_gain: float | None) -> str:
    if predicted_gain is None:
        return f"gain meas {measured_gain:+.4f}"
    return f"gain m {measured_gain:+.4f} p {predicted_gain:+.4f}"


def _render_refine_gif(run: dict, out: Path) -> float:
    crops = [_crop(np.load(entry["path"]), run["zero"]) for entry in run["entries"]]
    vmax = max(float(crop.max()) for crop in crops)
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    caption = f"vmax={vmax:.2f}"
    images = []
    for crop, entry in zip(crops, run["entries"]):
        image = Image.fromarray(_to_uint8(crop, vmax)).resize((288, 288), Image.NEAREST)
        canvas = Image.new("L", REFINE_CANVAS, 20)
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((288 - 8 - 6 * len(caption), 6), caption, fill=235)
        for index, line in enumerate(_refine_label(entry)):
            draw.text((6, 292 + 18 * index), line, fill=230)
        images.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=64))
    images[0].save(out, save_all=True, append_images=images[1:], duration=700, loop=0, optimize=True)
    return vmax


def _render_refine_gain_chart(runs: list[dict], out: Path) -> None:
    entries: list[tuple[str, dict]] = []
    for run in runs:
        short_run = run["tag"].rsplit("_", 1)[0]
        for row in run["rounds"]:
            suffix = str(row["round_id"] or "round").split("_")[-1].lstrip("0") or "0"
            entries.append((f"{short_run} R{suffix}", row))
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    positions = np.arange(len(entries), dtype=np.float64)
    width = 0.38
    measured = np.array([float(row["measured"]["gain"] or 0.0) for _, row in entries])
    predicted = np.array([float(row["predicted"]["gain"] or 0.0) for _, row in entries])
    rejected = np.array([not row["accepted"] for _, row in entries], dtype=bool)

    measured_bars = ax.bar(positions - width / 2, measured, width, label="实测增益", color="#4C78A8")
    predicted_bars = ax.bar(positions + width / 2, predicted, width, label="预测增益", color="#F58518", alpha=0.85)
    ax.bar(
        positions[rejected] - width / 2,
        measured[rejected],
        width,
        label="实测增益（轮次被拒）",
        color="#4C78A8",
        hatch="//",
        edgecolor="#111111",
        linewidth=0.8,
    )
    for bar, value in zip(predicted_bars, predicted):
        ax.annotate(f"{value:+.3f}", (bar.get_x() + bar.get_width() / 2, value), ha="center", va="bottom", fontsize=6)
    for bar, value in zip(measured_bars, measured):
        if value >= 0:
            ax.annotate(f"{value:+.4f}", (bar.get_x() + bar.get_width() / 2, value), ha="center", va="bottom",
                        fontsize=6, rotation=90)
        else:
            ax.annotate(f"{value:+.4f}", (bar.get_x() + bar.get_width() / 2, value), ha="center", va="top",
                        fontsize=6, rotation=90)

    thresholds = sorted({float(run["measured_min_gain"]) for run in runs if run["measured_min_gain"] is not None})
    for threshold in thresholds:
        ax.axhline(threshold, color="#E45756", linestyle="--", linewidth=1.2, label=f"接受阈值 {threshold:g}")
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.set_yscale("symlog", linthresh=0.002)
    top = float(np.max(predicted)) * 2.4 if predicted.size else 1.0
    bottom = float(np.min(measured)) * 2.6 if measured.size else -0.01
    ax.set_ylim(min(bottom, -0.004), top)
    ax.set_xticks(positions)
    ax.set_xticklabels([label for label, _ in entries], fontsize=8, rotation=45, ha="right")
    ax.set_xlabel("运行 / 轮次")
    ax.set_ylabel("目标函数增益（symlog 轴）")
    ax.set_title("refine 轮次实测增益与模型预测增益（2026-11-03 机台）", pad=30)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0, 1.0, 0.10), mode="expand", borderaxespad=0.1)
    fig.text(0.01, 0.01, "斜线柱：实测增益未超过接受阈值；预测增益为模型预测，并非机台实测", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0.03, 0.05, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _merge_refine_gains(runs: list[dict], out: Path) -> None:
    payload = {"schema": "sim2f_fov_refine_gains_v1", "date": REFINE_DATE, "rows": []}
    existing: list[dict] = []
    if out.is_file():
        try:
            old = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old = None
        if isinstance(old, dict) and isinstance(old.get("rows"), list):
            existing = [row for row in old["rows"] if isinstance(row, dict)]
    tags = {run["tag"] for run in runs}
    rows = [row for row in existing if row.get("tag") not in tags]
    for run in runs:
        rows.extend(run["rounds"])
    payload["rows"] = rows
    payload["runs"] = [
        {
            "tag": run["tag"],
            "refine_dir": run["refine_dir"],
            "status": run["status"],
            "gif": f"images/ccd_refine_rounds_{run['tag']}.gif",
            "gif_frames": [entry["id"] for entry in run["entries"]],
            "vmax": run.get("vmax"),
            "gaps": run["gaps"],
            "notes": run["notes"],
        }
        for run in runs
    ]
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _render_refine_runs(refine_dirs: list[Path], out: Path) -> list[dict]:
    used: dict[str, int] = {}
    runs = []
    for refine_dir in refine_dirs:
        summary = json.loads((refine_dir / "refine_summary.json").read_text(encoding="utf-8"))
        tag = _refine_tag(refine_dir)
        seen = used.get(tag, 0)
        used[tag] = seen + 1
        if seen:
            tag = f"{tag}_r{seen + 1}"
        zero = _refine_zero(summary)
        run = _build_refine_run(refine_dir, tag, summary, zero)
        gif_path = out / "images" / f"ccd_refine_rounds_{tag}.gif"
        if run["entries"]:
            run["vmax"] = _render_refine_gif(run, gif_path)
        run["gif"] = _display_path(gif_path) if run["entries"] else None
        print(
            json.dumps(
                {
                    "tag": tag,
                    "refine_dir": run["refine_dir"],
                    "zero": list(zero),
                    "frames": [entry["id"] for entry in run["entries"]],
                    "gaps": run["gaps"],
                    "notes": run["notes"],
                    "gif": run["gif"],
                }
            )
        )
        runs.append(run)
    if runs and any(run["rounds"] for run in runs):
        _merge_refine_gains(runs, out / "artifacts" / f"refine_gains_{REFINE_DATE}.json")
        _render_refine_gain_chart(runs, out / "figures" / f"refine_measured_vs_predicted_gain_{REFINE_DATE}.png")
    return runs


_SESSION: Path
_MANIFEST: dict
_ZERO: tuple[int, int]


def main(argv: list[str] | None = None) -> int:
    global _SESSION, _MANIFEST, _ZERO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--refine-dir", type=Path, action="append", default=None)
    args = parser.parse_args(argv)
    _SESSION = args.session
    _MANIFEST = json.loads((args.session / "capture_manifest.json").read_text(encoding="utf-8"))
    _ZERO = tuple(int(v) for v in _MANIFEST["zero_order_xy"])
    fig_dir = args.out / "figures"
    img_dir = args.out / "images"
    art_dir = args.out / "artifacts"
    for path in (fig_dir, img_dir, art_dir):
        path.mkdir(parents=True, exist_ok=True)
    refine = _load_refine_targets()
    _render_capture_gif(args.session, _MANIFEST, _ZERO, img_dir / "ccd_capture_sequence.gif")
    _render_shape_gif(args.session, _MANIFEST, _ZERO, img_dir / "ccd_shape_sequence.gif")
    rows = _collect_stats(args.session, _MANIFEST, _ZERO)
    summary = _render_charts(rows, refine, fig_dir)
    summary = {"zero_order_xy": list(_ZERO), "n_items": len(rows), **summary}
    (art_dir / "figures_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.refine_dir:
        refine_dirs = [path if path.is_absolute() else ROOT / path for path in args.refine_dir]
        _render_refine_runs(refine_dirs, args.out)
    print(json.dumps({"out": str(args.out), "n_items": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
