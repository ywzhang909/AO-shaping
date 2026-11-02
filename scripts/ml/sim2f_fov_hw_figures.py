"""Render CCD GIF and diagnostic charts for the 2026-11-02 sim2f hardware session.

Offline only: reads ``capture_manifest.json`` and the raw frames of a completed
session, writes figures under ``--out``. No hardware is touched. The charts are
diagnostics of the captured frames; they are not acceptance evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


_SESSION: Path
_MANIFEST: dict
_ZERO: tuple[int, int]


def main(argv: list[str] | None = None) -> int:
    global _SESSION, _MANIFEST, _ZERO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
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
    print(json.dumps({"out": str(args.out), "n_items": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
