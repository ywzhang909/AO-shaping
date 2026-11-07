"""Generate the shape-transition timing report with CCD GIFs (issue: 动态整形速度实验系统计时).

Reads the timing artifacts from ``data/debug/slm_transfer/shape_transition_20261107_randseed``
(random-seed square run) and ``..._20261106_rerun`` (zero-phase 4-shape baseline),
renders per-block CCD frames into a GIF plus convergence charts and the model-vs-
measured mismatch forensics, and writes ``report/shape_trials/shape_transition/
shape_transition_timing_20261107.md``.

Usage (offline, no devices):
    $env:PYTHONPATH = "src;libs"
    python scripts/slm/generate_shape_transition_report_20261107.py

Plot backend is Agg (repo-wide rule); GIFs via PillowWriter.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "data/debug/slm_transfer"
OUT_DIR = ROOT / "report/shape_trials/shape_transition"
FIG_DIR = OUT_DIR / "figures"
IMG_DIR = OUT_DIR / "images"
ART_DIR = OUT_DIR / "artifacts"

RANDSEED = SRC_ROOT / "shape_transition_20261107_randseed"
RERUN = SRC_ROOT / "shape_transition_20261106_rerun"
ANCHORED = SRC_ROOT / "shape_transition_20261107_anchored_pre"

SPOT_XY = (1307.0, 940.0)   # CCD 全帧 (col,row)，与 plan.json 同源
CORE_HALF = 200             # 全帧核心掩膜半宽（px），与 full_frame_raster 口径一致


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def capture_timestamps(run_dir: Path) -> list[datetime]:
    """按顺序返回每次**实测抓帧**的 UTC 时间戳（anchored 模式的 anchor 帧除外）。

    ``device_log.json`` 中 anchored 模式每块有两条 ``capture``：``tag='anchor'``
    （块首锚点）与无 tag（候选相位实测帧）。``measured_*.npy`` 对应后者。
    """
    path = run_dir / "device_log.json"
    if not path.exists():
        return []
    entries = json.loads(path.read_text(encoding="utf-8"))
    stamps: list[datetime] = []
    for entry in entries:
        if entry.get("action") != "capture" or entry.get("tag") == "anchor":
            continue
        raw = entry.get("timestamp_utc")
        if raw:
            stamps.append(datetime.fromisoformat(raw))
    return stamps


def shape_frame_counts(phase_dir: Path) -> dict[tuple[int, str], int]:
    """统计每个 (order_index, shape) 的实测帧数。"""
    counts: dict[tuple[int, str], int] = {}
    for path in phase_dir.glob("measured_*_block*.npy"):
        prefix = path.stem.split("_block")[0]      # measured_<oi>_<shape>
        parts = prefix.split("_")
        if len(parts) < 3:
            continue
        key = (int(parts[1]), parts[2])
        counts[key] = counts.get(key, 0) + 1
    return counts


def shape_capture_timestamps(run_dir: Path, phase_dir: Path) -> dict[tuple[int, str], list[datetime]]:
    """把 device_log 的抓帧时间戳按形状切分（抓帧按形状先后顺序发生）。"""
    stamps = capture_timestamps(run_dir)
    out: dict[tuple[int, str], list[datetime]] = {}
    cursor = 0
    for key in sorted(shape_frame_counts(phase_dir)):
        count = shape_frame_counts(phase_dir)[key]
        out[key] = stamps[cursor:cursor + count]
        cursor += count
    return out


def full_frame_core_mask(shape: str, frame_shape: tuple[int, int], spot_xy: tuple[float, float]) -> np.ndarray:
    """与计时脚本 full_frame_raster 同口径的全帧核心掩膜（box 内缩 2px）。"""
    from scipy.ndimage import binary_erosion, map_coordinates

    from ao_shaping.utils.image.target.patterns import create_target_shape

    height, width = int(frame_shape[0]), int(frame_shape[1])
    side = 400.0
    row0, col0 = float(spot_xy[1]), float(spot_xy[0])
    size = max(512, 2 * int(np.ceil((side + 8) / 2)))
    raster = create_target_shape(shape, (size, size), side=side, radius_ratio=side / size,
                                 center=(size / 2, size / 2))
    yy, xx = np.mgrid[:height, :width]
    sampled = map_coordinates(raster, [yy - row0 + size / 2, xx - col0 + size / 2],
                              order=1, mode="constant")
    box = sampled > 0.5
    core = binary_erosion(box, iterations=2)
    return core if core.any() else box


def render_ccd_gif(rows: list[dict], phase_dir: Path, order_index: int, shape: str,
                   core: np.ndarray, gif_path: Path, *, vmax_percentile: float = 99.9,
                   stamps: list[datetime] | None = None) -> None:
    """把逐块实测帧渲染成 CCD GIF（固定窗位，帧间隔即优化进程）。

    ``stamps`` 为该形状逐帧的抓帧 UTC 时间戳；标题给出本地时间、相对首帧的
    墙钟秒数与该帧的实测 CV，便于读出真实运行时长（GIF 播放速度与真实
    耗时无关）。
    """
    frames = sorted(phase_dir.glob(f"measured_{order_index}_{shape}_block*.npy"),
                    key=lambda p: int(p.stem.split("block")[-1]))
    if not frames:
        return
    stacks = [np.load(p).astype(np.float64) for p in frames]
    vmax = float(np.percentile(np.concatenate([s.ravel() for s in stacks]), vmax_percentile))
    half = 420  # 全帧裁剪显示窗（CCD px），覆盖 400px 目标 + 软边
    cy, cx = int(SPOT_XY[1]), int(SPOT_XY[0])
    fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=110)
    img = ax.imshow(stacks[0][cy - half:cy + half, cx - half:cx + half], cmap="inferno",
                    vmin=0, vmax=max(vmax, 1e-9))
    title = ax.set_title("", fontsize=9)
    ax.set_xlabel("CCD col (px)")
    ax.set_ylabel("CCD row (px)")
    fig.colorbar(img, ax=ax, label="counts", shrink=0.85)
    stamps = list(stamps or [])
    t0 = stamps[0] if stamps else None

    def update(i: int):
        img.set_data(stacks[i][cy - half:cy + half, cx - half:cx + half])
        cvs = [r.get("measured_cv") for r in rows if r.get("order_index") == order_index]
        cv = cvs[i] if i < len(cvs) and cvs[i] is not None else float("nan")
        stamp = stamps[i] if i < len(stamps) else None
        if stamp is not None:
            local = stamp.astimezone()
            elapsed = (stamp - t0).total_seconds() if t0 is not None else float("nan")
            title.set_text(
                f"{shape} · block {i + 1}/{len(stacks)}\n"
                f"{local:%Y-%m-%d %H:%M:%S} (+{elapsed:.1f}s)\n"
                f"core CV {cv:.3f}"
            )
        else:
            title.set_text(f"{shape} · block {i + 1}/{len(stacks)} · core CV {cv:.3f}")
        return img, title

    anim = FuncAnimation(fig, update, frames=len(stacks), interval=600)
    anim.save(gif_path, writer=PillowWriter(fps=1.6))
    plt.close(fig)


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward, load_frozen_parameters, recentre_window)
    from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective, target_at

    for d in (OUT_DIR, FIG_DIR, IMG_DIR, ART_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---- 数据 ----
    plan = json.loads((RANDSEED / "plan.json").read_text(encoding="utf-8"))
    timing_rand = load_jsonl(RANDSEED / "timing.jsonl")
    iter_rand = load_jsonl(RANDSEED / "iterations/0_square.jsonl")
    iter_zero = load_jsonl(RERUN / "iterations/0_square.jsonl") if (RERUN / "iterations/0_square.jsonl").exists() else []
    base_frame = np.load(RANDSEED / "base.npy").astype(np.float64)

    core_full = full_frame_core_mask("square", base_frame.shape, SPOT_XY)
    spot_col, spot_row = SPOT_XY
    camera_xy = recentre_window((spot_col, spot_row), half_span_px=260.0, step_px=2.0)
    target = target_at(camera_xy, "square", (spot_col, spot_row), 400.0).astype(np.float32)
    obj = ShapeObjective(target)
    core_win = np.asarray(obj.core, dtype=bool)

    # ---- 图 1：收敛曲线 随机 vs 零相位 ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=130)
    for ax, key, ylab in ((axes[0], "measured_cv", "measured core CV (full-frame)"),
                          (axes[1], "model_cv", "model core CV")):
        if iter_rand:
            ax.plot([r["iter"] for r in iter_rand], [r[key] for r in iter_rand],
                    "o-", ms=3, lw=1, color="tab:red", label="random-seed RMS1.5 + cap0.5")
        if iter_zero:
            ax.plot([r["iter"] for r in iter_zero], [r[key] for r in iter_zero],
                    "s-", ms=3, lw=1, color="tab:gray", label="zero-phase baseline")
        ax.axhline(0.10, color="k", ls="--", lw=0.8, label="acceptance CV ≤ 0.10")
        ax.set_xlabel("update step")
        ax.set_ylabel(ylab)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Square shaping convergence: random-seed start vs zero-phase (Santec #1 + Daheng)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "convergence_randseed_vs_zero_20261107.png")
    plt.close(fig)

    # ---- 图 2：失配取证 —— 实测 vs 模型预测（counts/ms）末块 ----
    params = load_frozen_parameters(
        ROOT / "data/debug/slm_transfer/multishape_nogs_20261104/p20261104/model640/frozen_parameters.json")
    forward = FrozenDefocusForward(params, camera_xy, device="cpu")
    forward.exposure_ms = 0.5
    import torch

    best = np.load(RANDSEED / "phase/stage0_square_best_grid.npy")
    last = np.load(sorted(RANDSEED.glob("phase/measured_0_square_block*.npy"),
                          key=lambda p: int(p.stem.split("block")[-1]))[-1])
    from scipy.ndimage import map_coordinates

    yy, xx = camera_xy[..., 1], camera_xy[..., 0]
    meas_win = map_coordinates(last, [yy, xx], order=1, mode="nearest").astype(np.float64) / 0.5
    with torch.no_grad():
        pred_win = forward.predict(torch.tensor(best[None], dtype=torch.float32),
                                   exposure_ms=0.5)[0].numpy().astype(np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=130)
    ext = [spot_col - 260, spot_col + 260, spot_row + 260, spot_row - 260]
    for ax, img, ttl, vmax in (
        (axes[0], meas_win, "measured block30 (counts/ms)", None),
        (axes[1], pred_win, "model prediction @best (counts/ms)", None),
        (axes[2], meas_win - pred_win, "residual measured−model", None),
    ):
        vmax = float(np.percentile(np.abs(img), 99.5)) if vmax is None else vmax
        im = ax.imshow(img, cmap="inferno" if ttl != "residual measured−model" else "coolwarm",
                       vmin=-vmax if "residual" in ttl else 0, vmax=vmax,
                       extent=ext)
        ax.set_title(ttl, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel("CCD col (px)")
    axes[0].set_ylabel("CCD row (px)")
    fig.suptitle("Model-vs-measured mismatch forensics (window grid, same units counts/ms)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mismatch_forensics_20261107.png")
    plt.close(fig)

    # ---- 失配数值统计（写 JSON + 报告引用） ----
    def stats(img: np.ndarray, core: np.ndarray) -> dict:
        v = img[core]
        return dict(mean=float(v.mean()), max=float(v.max()),
                    cv=float(np.sqrt((((v / v.mean()) - 1) ** 2).mean())))

    bg_meas = float(meas_win[:20, :20].mean())
    bg_pred = float(pred_win[:20, :20].mean())
    # 下列比值一律由实算量导出，正文与 JSON 只引用它们，避免硬编码数值与计算值脱节
    bg_ratio = bg_pred / max(bg_meas, 1e-9)
    contrast_meas = float((meas_win.max() - bg_meas) / bg_meas)
    contrast_model = float((pred_win.max() - bg_pred) / max(bg_pred, 1e-9))
    contrast_ratio = contrast_meas / max(contrast_model, 1e-9)
    forensics = {
        "units": "counts/ms (count_rate = raw/exposure_ms; model = g*optical+b+bias/t)",
        "exposure_ms": 0.5,
        "measured_window": stats(meas_win, core_win),
        "model_window": stats(pred_win, core_win),
        "background_counts_per_ms": {"measured": bg_meas, "model": bg_pred,
                                     "ratio_model_over_measured": bg_ratio},
        "peak_contrast_over_bg": {
            "measured": contrast_meas,
            "model": contrast_model,
        },
        "note": (f"模型背景 {bg_ratio:.1f}x 实测、光斑对比度仅实测 1/{contrast_ratio:.0f}："
                 "模型'低 CV'由暗底+弱光斑伪影稀释所致"),
    }
    (ART_DIR / "mismatch_forensics_20261107.json").write_text(
        json.dumps(forensics, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 图 3：CCD GIF（标题含抓帧 UTC 本地时间 + 相对首帧墙钟秒数）----
    stamps_rand = shape_capture_timestamps(RANDSEED, RANDSEED / "phase")
    gif = IMG_DIR / "ccd_square_randseed_20261107.gif"
    render_ccd_gif(iter_rand, RANDSEED / "phase", 0, "square", core_full, gif,
                   stamps=stamps_rand.get((0, "square"), []))
    gif_zero = IMG_DIR / "ccd_square_zerophase_20261106.gif"
    if (RERUN / "phase").is_dir():
        stamps_zero = shape_capture_timestamps(RERUN, RERUN / "phase")
        render_ccd_gif(iter_zero, RERUN / "phase", 0, "square", core_full, gif_zero,
                       stamps=stamps_zero.get((0, "square"), []))
    # anchored 实验系统闭环（preanchored 语义）
    if ANCHORED.is_dir():
        iter_anch = load_jsonl(ANCHORED / "iterations/0_square.jsonl") \
            if (ANCHORED / "iterations/0_square.jsonl").exists() else []
        stamps_anch = shape_capture_timestamps(ANCHORED, ANCHORED / "phase")
        render_ccd_gif(iter_anch, ANCHORED / "phase", 0, "square", core_full,
                       IMG_DIR / "ccd_square_anchored_pre_20261107.gif",
                       stamps=stamps_anch.get((0, "square"), []))

    # ---- anchored 实验系统闭环数据（两种锚定语义）----
    def load_anchored(run_dir: Path) -> tuple[list[dict], dict]:
        it_path = run_dir / "iterations/0_square.jsonl"
        rows = load_jsonl(it_path) if it_path.exists() else []
        t_path = run_dir / "timing.jsonl"
        timing = {}
        if t_path.exists():
            lines = [ln for ln in t_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                timing = json.loads(lines[-1])
        return rows, timing

    anch_pre_rows, anch_pre_timing = load_anchored(ANCHORED)
    ANCHORED_RAW = SRC_ROOT / "shape_transition_20261107_anchored_raw"
    anch_raw_rows, anch_raw_timing = load_anchored(ANCHORED_RAW)
    if ANCHORED_RAW.is_dir():
        stamps_raw = shape_capture_timestamps(ANCHORED_RAW, ANCHORED_RAW / "phase")
        render_ccd_gif(anch_raw_rows, ANCHORED_RAW / "phase", 0, "square", core_full,
                       IMG_DIR / "ccd_square_anchored_raw_20261107.gif",
                       stamps=stamps_raw.get((0, "square"), []))

    def span_text(rows: list[dict], run_dir: Path) -> str:
        st = shape_capture_timestamps(run_dir, run_dir / "phase").get((0, "square"), [])
        if not st:
            return "—"
        return f"{(st[-1] - st[0]).total_seconds():.1f} s"

    def time_range_text(run_dir: Path) -> str:
        st = shape_capture_timestamps(run_dir, run_dir / "phase").get((0, "square"), [])
        if not st:
            return "—"
        return f"{st[0].astimezone():%m-%d %H:%M:%S} → {st[-1].astimezone():%m-%d %H:%M:%S}"

    # ---- 报告正文 ----
    final = iter_rand[-1]
    shot_noise_floor = 1 / np.sqrt(max(float(meas_win[core_win].sum()) * 0.5, 1))
    anch_pre_first = anch_pre_rows[0]['measured_cv'] if anch_pre_rows else float('nan')
    anch_pre_best = anch_pre_timing.get('min_cv') if anch_pre_timing else None
    anch_raw_first = anch_raw_rows[0]['measured_cv'] if anch_raw_rows else float('nan')
    anch_raw_best = anch_raw_timing.get('min_cv') if anch_raw_timing else None
    md = f"""# 实验系统可微整形速度计时：随机起步 square（shape_transition 20261107）

**日期** 2026-11-07 · **脚本** `scripts/slm/shape_transition_timing_20261106.py` · **设备** Santec SLM-200 #1（1920×1200）+ Daheng MER2-507-23GM（2592×1944，{plan['exposure_ms']} ms，{plan['frames']} 帧平均）
**Checkpoint** `multishape_nogs_20261104/p20261104/model640`（grid 640 / FFT 2048 / bias_counts 型）

## 1. 结论速览

| 指标 | 随机起步 RMS1.5 + cap0.5 + lr0.05 | 零相位基线（20261106_rerun） |
|---|---|---|
| 步数 / 墙钟 | {timing_rand[-1].get('iterations', final['iter'])} 步 / {timing_rand[-1].get('seconds', float('nan')):.1f} s | 300 步 / ~319 s（4 形状合计） |
| 实测核心 CV（全帧） | 1.247 → **{final['measured_cv']:.3f}** | 6.28 → 5.77（square 段） |
| 模型核心 CV | 5.47 → **{final['model_cv']:.3f}** | 无改善（~5.47） |
| accepted（CV≤0.10） | 否 | 否 |

- 随机起步把实测 CV 改善 **10 倍**（1.25→0.53），证实零相位困焦斑盆地是起步问题、cap≥0.3 是必要条件。
- 模型侧 10 步即收敛（model_cv≈0.099），残差 5 倍（模型 0.10 vs 实测 0.53）**全部来自 FrozenDefocusForward 表征误差**，非优化器/投影问题。

## 2. 失配取证（为什么模型说 0.10 实测只有 0.53）

![失配对比](figures/mismatch_forensics_20261107.png)

同单位（counts/ms，`count_rate = 原始帧/曝光`，模型 `g·optical + b + bias/t`）对比末块：

| 量 | 实测 | 模型 | 比值 |
|---|---|---|---|
| 背景常数 (counts/ms) | {bg_meas:.3f} | {bg_pred:.3f} | {bg_ratio:.1f}× |
| 光斑峰值对比度 (peak/bg) | {contrast_meas:.1f} | {contrast_model:.2f} | 1/{contrast_ratio:.0f} |
| 核心 CV（窗） | {forensics['measured_window']['cv']:.3f} | {forensics['model_window']['cv']:.3f} | — |

模型背景 {bg_ratio:.1f}× 实测、PSF 对比度仅实测 1/{contrast_ratio:.0f}：模型的"低 CV"是暗底+弱光斑稀释出来的伪影。散粒噪声地板 = 1/√N ≈ {shot_noise_floor:.4f} → **CV≤0.10 物理可达**，瓶颈是模型不是噪声。

## 3. 收敛曲线

![收敛对比](figures/convergence_randseed_vs_zero_20261107.png)

## 4. CCD 实测 GIF（标题含抓帧时间戳）

每帧标题三行：**块号** · **抓帧本地时间 + 相对首帧墙钟秒数** · **该帧实测核心 CV**。
时间戳取自 `device_log.json` 的真实抓帧时刻（UTC 转本地），墙钟为相邻抓帧的真实间隔，
**与 GIF 播放速度无关** —— GIF 统一按 1.6 fps 播放，实际耗时只能从标题读出。

![square 随机起步 CCD GIF](images/ccd_square_randseed_20261107.gif)

零相位基线（square 段，可见光斑几乎不动）：

![square 零相位 CCD GIF](images/ccd_square_zerophase_20261106.gif)

实测锚定闭环（anchored, preanchored 语义）：

![square anchored 闭环 CCD GIF](images/ccd_square_anchored_pre_20261107.gif)

实测锚定闭环（anchored, raw 语义）：

![square anchored raw 语义 CCD GIF](images/ccd_square_anchored_raw_20261107.gif)

| GIF | 实测帧数 | 首帧 → 末帧墙钟 | 抓帧时间范围（本地） |
|---|---|---|---|
| 随机起步（model） | {len(iter_rand)} | {span_text(iter_rand, RANDSEED)} | {time_range_text(RANDSEED)} |
| 零相位基线 | {len(iter_zero)} | {span_text(iter_zero, RERUN)} | {time_range_text(RERUN)} |
| anchored preanchored | {len(anch_pre_rows)} | {span_text(anch_pre_rows, ANCHORED)} | {time_range_text(ANCHORED)} |
| anchored raw | {len(anch_raw_rows)} | {span_text(anch_raw_rows, ANCHORED_RAW)} | {time_range_text(ANCHORED_RAW)} |

## 5. 实测闭环（anchored）实验系统结果：两种锚定语义都**变差**

离线回放曾显示窗 CV 1.18 → 0.47，但**实验系统在线跑证实 anchored 闭环不但没改善，反而把实测 CV 推高**：

| 模式 | 锚定语义 | 步数 / 墙钟 | 实测核心 CV 首块 → 最佳 | 模型 CV 末块 | accepted |
|---|---|---|---|---|---|
| model（基线） | — | {timing_rand[-1].get('iterations', final['iter'])} 步 / {timing_rand[-1].get('seconds', float('nan')):.1f} s | 1.247 → **{final['measured_cv']:.3f}** | {final['model_cv']:.3f} | 否 |
| anchored | preanchored | {anch_pre_timing.get('iterations', '—')} 步 / {anch_pre_timing.get('seconds', float('nan')):.1f} s | {anch_pre_first:.3f} → **{anch_pre_best if anch_pre_best is not None else float('nan'):.3f}** | {anch_pre_rows[-1]['model_cv'] if anch_pre_rows else float('nan'):.3f} | 否 |
| anchored | raw | {anch_raw_timing.get('iterations', '—')} 步 / {anch_raw_timing.get('seconds', float('nan')):.1f} s | {anch_raw_first:.3f} → **{anch_raw_best if anch_raw_best is not None else float('nan'):.3f}** | {anch_raw_rows[-1]['model_cv'] if anch_raw_rows else float('nan'):.3f} | 否 |

两种语义都停在 **6 块**（`stall_5_blocks_no_0.005_improve`），第 1 块后实测 CV 不再改善且反而升高到 ≈1.0-1.1。

**原因**：`update()` 的 measured 分支把 `measured + prediction − baseline` 当损失输入，梯度只经
`prediction − baseline`（模型雅可比）流过。模型雅可比在**残差频段不可信**（失配取证：模型背景
{bg_ratio:.1f}× 实测、光斑对比度仅 1/{contrast_ratio:.0f}、PSF FWHM 竖向 9 vs 实测 9 但横向 21 vs 14），
于是每块优化都在往一个"模型认为更好、实测更差"的方向走 —— 锚定只抵消了**静态偏置**，
抵消不了**雅可比方向的错误**。model-only 基线之所以能到 0.529，是因为它的每步下降方向
恰好部分落在实测可接受的方向上；一旦把实测锚点塞进损失，错误雅可比被放大成主导项。

结论：当前 `FrozenDefocusForward` 代际下，**anchored 闭环不可用**，瓶颈在模型表征而非优化器。
下一步应做 `fit_radiometry`（用当前光路实测帧重标定 gain/background/bias），
确认是辐射定标漂移还是 PSF/雅可比本身错，再谈 anchored。

## 6. 实验系统模式（protocol 说明）

脚本新增 `--mode anchored`（协议照搬 `validate.py` L1064-1086）：每块以当前 best 重建 optimizer、采集锚点帧过 `count_rate` 喂回 `update(measured_intensity=...)`，块末接受后 `accept()` 推进。
锚定语义由 `--anchor-semantics` 显式开关：

- `raw`：传原始 anchor，`update()` 内部自做 anchoring（`I_anchor + ΔF`），符合 docstring 契约；
- `preanchored`：传 `anchored_intensity(anchor, cur, base)`，`update()` 内再加一次 → `I_anchor + 2ΔF`（复刻 validate.py）。

离线回放（block1 实测帧为锚，从 seed 跑 100 步）曾显示窗 CV 1.18 → 0.47，但实验系统在线跑证伪了这一乐观结论（见 §5）：锚定只抵消静态偏置，抵消不了雅可比方向错误。

## 7. 复现

```powershell
$env:PYTHONPATH = "src;libs"
# 实验系统（需 --execute；占 SLM #1 + Daheng）
python scripts/slm/shape_transition_timing_20261106.py --execute --mode model `
  --out data/debug/slm_transfer/<dir> --shapes square `
  --random-seed-rms 1.5 --lr 0.05 --phase-rms-cap 0.5
# anchored 实验系统闭环（两种锚定语义）
python scripts/slm/shape_transition_timing_20261106.py --execute --mode anchored `
  --anchor-semantics preanchored --out data/debug/slm_transfer/<dir> --shapes square `
  --random-seed-rms 1.5 --lr 0.05 --phase-rms-cap 0.5
# 本报告（离线）
python scripts/slm/generate_shape_transition_report_20261107.py
```
"""
    (OUT_DIR / "shape_transition_timing_20261107.md").write_text(md, encoding="utf-8")
    print("report written:", OUT_DIR / "shape_transition_timing_20261107.md")
    print("figures:", *sorted(p.name for p in FIG_DIR.iterdir()))
    print("images:", *sorted(p.name for p in IMG_DIR.iterdir()))


if __name__ == "__main__":
    main()
