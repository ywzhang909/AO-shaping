"""Offline report generator for the SLM-Zernike shaping optimizer.

Reads the debug artifact bundles written by
``src/ao_shaping/optimizer/wfless/slm_zernike_shaping.py`` when ``debug=True``
(PKL + JSON sidecar + PNG) and produces:

* ``docs/slm_zernike_shaping/figures/*.png`` — per-run figures (objective
  curve, Zernike-coefficient evolution, first/best/last CCD frame);
* ``docs/slm_zernike_shaping/figures/*.gif`` — animated far-field evolution;
* ``docs/slm_zernike_shaping/report.md`` — the markdown report.

Fully offline: it never opens hardware, it only reads the saved PKL/JSON. This
follows the repo convention that report generation lives in ``scripts/`` and is
offline (see ``scripts/README.md``).

Usage:
    python scripts/generate_slm_zernike_shaping_report.py
    python scripts/generate_slm_zernike_shaping_report.py --debug-root data
    python scripts/generate_slm_zernike_shaping_report.py --max-runs 3
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: must be set before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# CJK-capable font fallbacks (per repo script convention).
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

#: ``subdir_prefix`` used by ``slm_zernike_shaping._save_debug_artifacts``.
ARTIFACT_PREFIX = "slm_zernike_shaping_"

#: Objectives the optimizer *maximises*. Mirrors the ``objective_mode``
#: derivation in ``slm_zernike_shaping.py`` (L759-762): ``pib``/``avg_radiu``/
#: ``shape``/``roi_pib``/``rms_pib`` are "max", everything else is "min"
#: (``radiu``, ``rmse``, ``rmse_out``). The JSON sidecar does not persist the
#: mode, so it is reconstructed here instead of assuming "max".
MAX_OBJECTIVES = frozenset({"pib", "avg_radiu", "shape", "roi_pib", "rms_pib"})

#: Fallback objective columns, tried in order when the payload's ``objective``
#: name is absent from the records (never invent a key — probe instead).
OBJECTIVE_CANDIDATES = (
    "pib",
    "shape",
    "roi_pib",
    "rms_pib",
    "rmse",
    "rmse_out",
    "radiu",
    "avg_radiu",
    "J",
)

#: Cross-objective metric panel — only the keys present in a bundle are drawn.
CROSS_METRIC_KEYS = ("m_shape", "m_roi_pib", "m_rms_t", "m_ee", "m_pib")

#: Metadata rows rendered from the JSON sidecar payload, in display order.
META_FIELDS: tuple[tuple[str, str], ...] = (
    ("objective", "objective 优化目标"),
    ("target_shape", "target_shape 目标形状"),
    ("target_size", "target_size 目标尺寸 (px)"),
    ("epochs", "epochs 迭代数"),
    ("algorithm", "algorithm 搜索族"),
    ("optimizer_type", "optimizer_type 梯度更新器"),
    ("delta", "delta SPGD 扰动幅度"),
    ("w_outside", "w_outside 框外能量权重"),
    ("r_bucket", "r_bucket 半径桶 (px)"),
    ("cam_type", "cam_type 相机后端"),
    ("cam_size", "cam_size 开窗 (px)"),
)


# --- artifact loading --------------------------------------------------------


def find_debug_dirs(debug_root: Path) -> list[Path]:
    """Return every shaping artifact leaf dir under ``debug_root``, newest first.

    The optimizer writes ``<debug_dir>/debug/<prefix><objective>_<ts>/<ts>/``
    (``save_recorder_debug_artifacts`` appends its own ``debug`` segment), so
    both ``<root>/debug/<prefix>*/*`` and ``<root>/<prefix>*/*`` are scanned —
    that makes ``--debug-root data`` and ``--debug-root data/debug`` both work.
    Only leaves that actually hold a ``.pkl`` are returned.
    """
    candidates: list[Path] = []
    for pattern in (f"debug/{ARTIFACT_PREFIX}*/*", f"{ARTIFACT_PREFIX}*/*"):
        candidates.extend(debug_root.glob(pattern))
    dirs = [d for d in candidates if d.is_dir() and any(d.glob("*.pkl"))]
    # de-duplicate while keeping the newest-first ordering
    seen: set[Path] = set()
    unique: list[Path] = []
    for d in sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True):
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def load_run(art_dir: Path) -> dict:
    """Load one run's PKL history (``{epoch: record}``) + JSON sidecar payload."""
    pkl = next(iter(sorted(art_dir.glob("*.pkl"))), None)
    if pkl is None:
        raise FileNotFoundError(f"no .pkl in {art_dir}")
    with open(pkl, "rb") as fh:
        data: dict = pickle.load(fh)
    epochs = sorted(int(k) for k in data.keys() if str(k).lstrip("-").isdigit())
    payload: dict = {}
    jf = next(iter(sorted(art_dir.glob("*.json"))), None)
    if jf is not None:
        try:
            payload = json.loads(jf.read_text(encoding="utf8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("unreadable JSON sidecar {}: {}", jf, exc)
    if not isinstance(payload, dict):
        payload = {}
    return {"data": data, "epochs": epochs, "payload": payload, "dir": art_dir}


# --- per-epoch series extraction --------------------------------------------


def _as_float(value: Any) -> float:
    """First element of ``value`` as float, NaN when empty/missing."""
    if value is None:
        return float("nan")
    arr = np.asarray(value, dtype=float)
    return float(arr.ravel()[0]) if arr.size else float("nan")


def scalar_series(run: dict, key: str) -> np.ndarray:
    """Per-epoch scalar for ``key`` (NaN where the record lacks the key)."""
    return np.array(
        [_as_float(run["data"].get(e, {}).get(key)) for e in run["epochs"]], dtype=float
    )


def vector_series(run: dict, key: str) -> np.ndarray:
    """Per-epoch 1-D array for ``key`` -> ``(epochs, n)``; NaN-padded."""
    rows: list[np.ndarray | None] = []
    width = 0
    for e in run["epochs"]:
        value = run["data"].get(e, {}).get(key)
        if value is None:
            rows.append(None)
            continue
        arr = np.asarray(value, dtype=float).ravel()
        rows.append(arr)
        width = max(width, arr.size)
    if width == 0:
        return np.zeros((0, 0), dtype=float)
    return np.vstack(
        [r if r is not None and r.size == width else np.full(width, np.nan) for r in rows]
    )


def image_series(run: dict, key: str = "_img") -> list[np.ndarray]:
    """Per-epoch 2-D image (e.g. ``_img``), resampled to a common shape."""
    imgs: list[np.ndarray] = []
    shape: tuple[int, ...] | None = None
    for e in run["epochs"]:
        value = run["data"].get(e, {}).get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=float)
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            arr = np.resize(arr, shape)
        imgs.append(arr)
    return imgs


def resolve_objective_key(run: dict) -> str:
    """The run's own objective column: payload ``objective`` if present."""
    records = list(run["data"].values())
    name = str(run["payload"].get("objective") or "").strip()
    if name and any(name in rec for rec in records):
        return name
    for cand in OBJECTIVE_CANDIDATES:
        if any(cand in rec for rec in records):
            return cand
    return "J"


def objective_direction(key: str) -> str:
    """``"max"`` for the bucket/shaping objectives, ``"min"`` otherwise."""
    return "max" if str(key).strip().lower() in MAX_OBJECTIVES else "min"


def objective_summary(run: dict, key: str, direction: str) -> dict[str, Any]:
    """Best (and initial) objective value plus the epoch it occurred at."""
    values = scalar_series(run, key)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return {"key": key, "direction": direction, "available": False}
    pick = np.argmin if direction == "min" else np.argmax
    idx = int(finite[pick(values[finite])])
    initial = values[0] if np.isfinite(values[0]) else float("nan")
    return {
        "key": key,
        "direction": direction,
        "available": True,
        "best_value": float(values[idx]),
        "best_epoch": int(run["epochs"][idx]),
        "initial_value": float(initial),
        "final_value": float(values[-1]),
    }


# --- figure rendering --------------------------------------------------------


def plot_objective(
    run: dict, fig_dir: Path, tag: str, obj_key: str, direction: str
) -> Path:
    """Objective-vs-epoch for the run's own column, plus the cross-metric panel."""
    x = np.asarray(run["epochs"], dtype=float)
    values = scalar_series(run, obj_key)
    summary = objective_summary(run, obj_key, direction)

    cross = [k for k in CROSS_METRIC_KEYS if np.isfinite(scalar_series(run, k)).any()]
    ncols = 2 if cross else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols, 3.8), squeeze=False)
    ax = axes[0][0]
    ax.plot(x, values, "-o", ms=3, color="tab:blue")
    if summary.get("available"):
        ax.plot(
            [summary["best_epoch"]],
            [summary["best_value"]],
            "*",
            ms=16,
            color="tab:red",
            label=f"best {summary['best_value']:.4g} @ {summary['best_epoch']}",
        )
        ax.legend(loc="best", fontsize=8)
    arrow = "minimised" if direction == "min" else "maximised"
    ax.set_title(f"Objective '{obj_key}' ({arrow})")
    ax.set_xlabel("epoch")
    ax.set_ylabel(obj_key)
    ax.grid(alpha=0.3)

    if cross:
        ax2 = axes[0][1]
        for k in cross:
            ax2.plot(x, scalar_series(run, k), "-o", ms=3, lw=1.2, label=k)
        ax2.set_title("Cross-objective metrics")
        ax2.set_xlabel("epoch")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = fig_dir / f"{tag}_objective.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_zernike(run: dict, fig_dir: Path, tag: str) -> Path | None:
    """Per-mode Zernike coefficient traces + the ``‖_c‖₂`` norm evolution."""
    c = vector_series(run, "_c")
    if c.size == 0:
        logger.warning("{}: no '_c' in bundle, skipping zernike figure", run["dir"])
        return None
    x = np.asarray(run["epochs"], dtype=float)
    n_modes = c.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    cmap = plt.get_cmap("viridis")
    for k in range(n_modes):
        axes[0].plot(x, c[:, k], "-", lw=1.3, color=cmap(k / max(1, n_modes - 1)))
    axes[0].axhline(0, color="k", lw=0.6, alpha=0.5)
    axes[0].set_title(f"Zernike coefficient trace ({n_modes} modes)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("coefficient")
    axes[0].grid(alpha=0.3)

    norms = np.linalg.norm(c, axis=1)
    axes[1].plot(x, norms, "-o", ms=3, color="tab:orange")
    axes[1].set_title("Zernike coefficient norm  norm2(_c)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("norm")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = fig_dir / f"{tag}_zernike.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_frames(
    run: dict, fig_dir: Path, tag: str, best_epoch: int | None
) -> Path | None:
    """First / best / last CCD frame (``_img``) side by side, shared intensity scale."""
    imgs = image_series(run, "_img")
    if not imgs:
        logger.warning("{}: no '_img' in bundle, skipping frame figure", run["dir"])
        return None
    epochs = run["epochs"]
    # ``image_series`` skips records without ``_img``; keep the epoch labels aligned.
    labels = [e for e in epochs if run["data"].get(e, {}).get("_img") is not None]
    idxs = [0, len(imgs) - 1]
    if best_epoch is not None and best_epoch in labels:
        idxs.append(labels.index(best_epoch))
    idxs = sorted(set(idxs))
    vmax = max(float(np.nanmax(imgs[i])) for i in idxs if imgs[i].size) or 1.0

    fig, axes = plt.subplots(1, len(idxs), figsize=(3.2 * len(idxs), 3.5), squeeze=False)
    for ax, i in zip(axes[0], idxs):
        im = ax.imshow(np.clip(imgs[i], 0, vmax), cmap="inferno", vmin=0, vmax=vmax)
        role = {0: "first", len(imgs) - 1: "last"}.get(i, "best")
        ax.set_title(f"{role} — epoch {labels[i]}", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("CCD far-field frame evolution", y=1.02)
    fig.tight_layout()
    out = fig_dir / f"{tag}_frames.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def make_gif(run: dict, gif_dir: Path, tag: str) -> Path | None:
    """GIF of the per-epoch ``_img`` frames (thinned to <= 24 frames)."""
    imgs = image_series(run, "_img")
    if not imgs:
        return None
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not available, skipping GIF for {}", run["dir"])
        return None

    if len(imgs) > 24:
        keep = np.linspace(0, len(imgs) - 1, 24).round().astype(int)
        imgs = [imgs[i] for i in keep]
    vmax = max(float(np.nanmax(f)) for f in imgs if f.size) or 1.0
    cmap = plt.get_cmap("inferno")
    frames = []
    for f in imgs:
        norm = np.clip(f, 0, vmax) / vmax
        frames.append(Image.fromarray((cmap(norm)[:, :, :3] * 255).astype(np.uint8)))
    out = gif_dir / f"{tag}_spot.gif"
    frames[0].save(
        out, save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=True
    )
    return out


# --- markdown ----------------------------------------------------------------


def rel(path: Path, out_dir: Path) -> str:
    """Path relative to the report's own directory (``figures/<name>``)."""
    try:
        return str(path.relative_to(out_dir))
    except ValueError:
        return str(path)


def build_markdown(
    runs: list[dict],
    artefacts: dict[str, dict[str, list[Path]]],
    out_dir: Path,
    out_md: Path,
) -> None:
    """Render ``report.md`` tying the figures and the per-run tables together."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# SLM-Zernike 整形优化器报告 (slm_zernike_shaping)")
    lines.append("")
    lines.append(f"**生成时间**: {now}")
    lines.append("")
    lines.append(
        "**Fully offline** — 本报告由 `scripts/generate_slm_zernike_shaping_report.py` "
        "离线生成, 仅读取 `slm_zernike_shaping` 优化器以 `debug=True` 保存的 "
        "PKL/JSON 调试产物, 不打开任何硬件。"
    )
    lines.append("")
    lines.append("## 1. 产物格式")
    lines.append("")
    lines.append("每个运行对应一个产物目录:")
    lines.append("")
    lines.append("```")
    lines.append("<debug_dir>/debug/slm_zernike_shaping_<objective>_<ts>/<ts>/")
    lines.append("  ├── slm_zernike_shaping_<objective>_<ts>_<ts>.pkl   # {epoch: record}")
    lines.append("  ├── slm_zernike_shaping_<objective>_<ts>_<ts>.json  # 配置 payload")
    lines.append("  └── slm_zernike_shaping_<objective>_<ts>_<ts>.png   # 运行期汇总图")
    lines.append("```")
    lines.append("")
    lines.append(
        "PKL 记录的键来自优化器的 `_DEBUG_SCALAR_KEYS` / `_DEBUG_IMG_KEYS` / "
        "`_DEBUG_1D_KEYS` (`_img` 远场帧, `_c` Zernike 系数向量) 以及跨目标指标 "
        "`m_shape / m_energy / m_rmse / m_roi_pib / m_pib / m_pib7 / m_rms_pib / "
        "m_rms_t / m_ee / m_brt`; 优化目标自身还有一个以目标命名的列 "
        "(如 `rmse_out` / `pib` / `shape`)。本脚本全部按键读取并以 `.get()` 兜底, "
        "缺失的图会被跳过而不是报错。"
    )
    lines.append("")

    for run, tag in zip(runs, artefacts.keys()):
        items = artefacts[tag]
        payload = run["payload"]
        obj_key = resolve_objective_key(run)
        direction = objective_direction(obj_key)
        summary = objective_summary(run, obj_key, direction)
        epochs = run["epochs"]
        lines.append(f"## 2. 运行 `{tag}` (目录 `{run['dir'].name}`)")
        lines.append("")

        # --- metadata from the JSON sidecar -------------------------------
        lines.append("### 2.1 运行配置 (JSON sidecar)")
        lines.append("")
        lines.append("| 项目 | 值 |")
        lines.append("|---|---|")
        rendered: set[str] = set()
        for field, label in META_FIELDS:
            if field in payload:
                lines.append(f"| {label} | `{payload[field]}` |")
                rendered.add(field)
        for key, value in payload.items():
            if key not in rendered:
                lines.append(f"| {key} | `{value}` |")
        if not payload:
            lines.append("| (无 JSON sidecar) | — |")
        lines.append(f"| 产物目录 | `{run['dir']}` |")
        lines.append(f"| 记录条数 | {len(epochs)} (epoch {epochs[0]}..{epochs[-1]}) |" if epochs else "| 记录条数 | 0 |")
        lines.append("")

        # --- objective summary --------------------------------------------
        lines.append("### 2.2 目标函数结果")
        lines.append("")
        if summary.get("available"):
            best_val = summary["best_value"]
            init_val = summary["initial_value"]
            final_val = summary["final_value"]
            better = (
                (direction == "min" and final_val <= init_val)
                or (direction == "max" and final_val >= init_val)
            )
            lines.append(
                f"- **最优 `{obj_key}` = {best_val:.6g}, 出现在 epoch "
                f"{summary['best_epoch']}** (该目标为**"
                f"{'最小化' if direction == 'min' else '最大化'}**)"
            )
            lines.append(
                f"- 初始值 {init_val:.6g} → 末值 {final_val:.6g} "
                f"({'改善' if better else '未改善'}); 最优相对初始 "
                f"{(init_val - best_val) if direction == 'min' else (best_val - init_val):.6g}"
            )
            # The optimizer tracks its own running best in `best_<objective>`,
            # but that key is not in `_DEBUG_SCALAR_KEYS` so it is usually absent.
            last_rec = run["data"].get(epochs[-1], {}) if epochs else {}
            tracked = last_rec.get(f"best_{obj_key}")
            if tracked is None:
                lines.append(
                    "- 优化器内部跟踪的 `best_<objective>` 未写入调试产物 "
                    "(`_DEBUG_SCALAR_KEYS` 不含该键), 故最优值由本脚本从目标列重算。"
                )
            else:
                lines.append(f"- 优化器内部跟踪的 `best_{obj_key}` = {_as_float(tracked):.6g}")
        else:
            lines.append(f"- 目标列 `{obj_key}` 无有效数值, 无法给出最优值。")
        lines.append("")

        for group in ("objective", "zernike", "frames"):
            for path in items.get(group, []):
                lines.append(f"![{path.stem}]({rel(path, out_dir)})")
                lines.append("")
        for path in items.get("gif", []):
            lines.append(f"![{path.stem}]({rel(path, out_dir)})")
            lines.append("")
        lines.append("### 2.3 解读")
        lines.append("")
        lines.append(
            f"- 左图是本次运行真正优化的目标列 `{obj_key}` "
            f"({'最小化' if direction == 'min' else '最大化'}), 红星标出全局最优 epoch; "
            "右图是同一批记录里交叉记录的全部 `m_*` 指标, 便于观察各指标是否同向变化。"
        )
        lines.append(
            "- Zernike 系数图给出逐模式演化与 `‖_c‖₂` 范数: 范数持续增长说明搜索仍在"
            "推动相位偏离初始值, 范数饱和则表示该自由度已到边界。"
        )
        lines.append(
            "- 远场帧按 first / best / last 三联对比 (统一灰度标度), 直观显示目标框内的"
            "能量分布如何随迭代变化。"
        )
        lines.append("")

    lines.append("## 3. 结论")
    lines.append("")
    lines.append(
        "1. **完全离线可复现**: 本报告只依赖 `debug=True` 写出的 PKL/JSON, 任何一次 "
        "`slm_zernike_shaping` 运行 (仿真或硬件) 的产物都能重新出报告, 无需再上硬件。"
    )
    lines.append(
        "2. **目标方向不假设**: 优化器的 `objective_mode` 决定了 `pib`/`shape`/`roi_pib`/"
        "`rms_pib`/`avg_radiu` 是最大化而 `radiu`/`rmse`/`rmse_out` 是最小化; 报告按该约定"
        "标注并计算最优 epoch, 避免把最小化目标误读为最大化。"
    )
    lines.append(
        "3. **缺键不崩**: 所有记录键均以 `.get()` 读取, 缺 `objective_keys` 的 bundle 会"
        "回退到 JSON 里的 `objective`, 再回退到候选列, 全部缺失时只跳过对应图并告警。"
    )
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


# --- cli ---------------------------------------------------------------------


def cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--debug-root",
        type=Path,
        default=ROOT / "data" / "debug",
        help="Root dir containing slm_zernike_shaping_* artifact dirs.",
    )
    ap.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="A single artifact dir (overrides the --debug-root glob).",
    )
    ap.add_argument(
        "--max-runs",
        type=int,
        default=1,
        help="How many runs (newest first) to render.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "slm_zernike_shaping",
        help="Output dir for figures/report.md.",
    )
    args = ap.parse_args()

    out_dir: Path = args.out
    fig_dir = out_dir / "figures"
    for d in (out_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.debug_dir is not None:
        dirs = [args.debug_dir]
    else:
        dirs = find_debug_dirs(args.debug_root)[: max(1, args.max_runs)]
    if not dirs:
        logger.warning(
            "no slm_zernike_shaping debug artifacts under {} — nothing to report",
            args.debug_root,
        )
        return

    runs: list[dict] = []
    artefacts: dict[str, dict[str, list[Path]]] = {}
    for i, d in enumerate(dirs):
        try:
            run = load_run(d)
        except (OSError, pickle.UnpicklingError, FileNotFoundError) as exc:
            logger.warning("skipping unreadable bundle {}: {}", d, exc)
            continue
        if not run["epochs"]:
            logger.warning("skipping empty bundle {}", d)
            continue
        runs.append(run)
        tag = f"run{len(runs) - 1}"
        obj_key = resolve_objective_key(run)
        direction = objective_direction(obj_key)
        summary = objective_summary(run, obj_key, direction)
        best_epoch = summary.get("best_epoch") if summary.get("available") else None
        logger.info(
            "run {}: {} epochs={} objective={} ({}) dir={}",
            tag,
            len(run["epochs"]),
            len(run["epochs"]),
            obj_key,
            direction,
            d,
        )

        items: dict[str, list[Path]] = {"objective": [], "zernike": [], "frames": [], "gif": []}
        try:
            items["objective"].append(plot_objective(run, fig_dir, tag, obj_key, direction))
        except (ValueError, TypeError) as exc:
            logger.warning("{}: objective figure failed: {}", d, exc)
        for group, producer in (
            ("zernike", lambda: plot_zernike(run, fig_dir, tag)),
            ("frames", lambda: plot_frames(run, fig_dir, tag, best_epoch)),
            ("gif", lambda: make_gif(run, fig_dir, tag)),
        ):
            try:
                path = producer()
            except (ValueError, TypeError, OSError) as exc:
                logger.warning("{}: {} figure failed: {}", d, group, exc)
                path = None
            if path is not None:
                items[group].append(path)
        artefacts[tag] = {k: v for k, v in items.items() if v}
        for group, paths in artefacts[tag].items():
            for path in paths:
                logger.info("  wrote [{}] {}", group, path)

    if not runs:
        logger.warning("no readable runs found — report not written")
        return

    out_md = out_dir / "report.md"
    build_markdown(runs, artefacts, out_dir, out_md)
    logger.info("report -> {}", out_md)


if __name__ == "__main__":
    cli()
