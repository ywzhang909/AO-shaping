"""star 块 CCD 实测远场对照分析（会话 20261107_135252）。

目的
----
确认 ``shapes=['star']`` 迁移训练会话（acceptance 启用、共 3 轮）中 star 块在
CCD 实测远场到底是五角星、方块还是弥散圆斑 —— 之前实时视图里的「方块」是否
为真实光束形状。

结论（本脚本已按此渲染，写入报告）
--------------------------------
star 块的 CCD 实测远场是**弥散圆斑（零级主导）**，既非五角星也非方块：

* 半高（50%）阈值掩膜只抠到 ~27×10 px 的亮核；
* 降到 10% 阈值才露出 ~350×375 px 的弥散光斑 —— 星形变体太弱，被零级淹没；
* 角向谐波 FFT（h2–h8）幅度噪声主导，跨帧主导谐波漂移（h7→h4→h3→h5），
  无一致 5 重 / 4 重对称 —— 属低对比散斑的几何噪声，非星形也非方形。

用法（离线，不占用任何设备）::

    python scripts/slm/star_farfield_analysis_20261107.py [--out-dir <figures 输出目录>] [--session <会话目录名>]

产物（默认写入 ``report/shape_trials/star_farfield/figures/``）:

* ``star_farfield_montage_20261107.png`` —— 理想 star/square 目标 vs 实测
  （seed / A1 × 3 轮）对照 + 角向谐波谱；
* ``star_model_linear_20261107.png`` —— model 阶段 + 线性 / 对数拉伸对照。
* ``star_model_square_artifact_20261107.png`` —— 证明 model 预测的'方块'是
  camera_sample_grids 的 in-bounds 采样窗伪影（非真实远场）。

数据来源：``data/debug/slm_transfer/20261107_135252``（star 块 CCD =
``anchored/capture_0001.npz`` = ``star_0_A1``，ABBA 实测位 A1）。
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
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from ao_shaping.algorithm.signal_processing.defocus_forward_core import (
    camera_sample_grids,
    subpixel_offsets,
)
from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
    load_frozen_parameters,
    defocus_phase,
)
from ao_shaping.gui.slm.transfer_desktop.archive_replay import load_capture
from ao_shaping.tools.slm.transfer.artifacts import resolve_recorded_path
from ao_shaping.utils.image.target.patterns import create_target_shape

DEFAULT_SESSION = "20261107_135252"
DEFAULT_OUT = ROOT / "report/shape_trials/star_farfield/figures"

#: 分析阈值相对亮度的比例（10% 比半高更能暴露低对比的星形变体）。
THRESH_FRAC = 0.10
#: 角向谐波分析范围（2..8 重）。
HARMONIC_RANGE = range(2, 9)
#: 渲染的种子/实测位标签 -> capture 序号。
PICK_LABELS = (("seed", 0), ("A1", 1))
#: 轮次目录（round0 为初始 anchored，round1/round2 为验收重训轮）。
ROUNDS = ("anchored", "anchored.round1", "anchored.round2")


def _angular_harmonics(
    raw: np.ndarray, thr_frac: float = THRESH_FRAC
) -> tuple[np.ndarray | None, dict[int, float] | None, int | None, tuple[float, float] | None]:
    """按阈值掩膜做角向半径剖面 FFT。

    Returns:
        (mask, harmonic->|FFT|, 主导谐波阶数, (cx, cy))；掩膜过小时返回
        ``(None, None, None, None)``。
    """
    raw = raw.astype(np.float64)
    bg = float(np.median(raw))
    pk = float(np.max(raw))
    if pk - bg <= 1e-6:
        return None, None, None, None
    thr = bg + thr_frac * (pk - bg)
    mask = raw >= thr
    ys, xs = np.nonzero(mask)
    if xs.size < 20:
        return None, None, None, None
    mx, my = float(xs.mean()), float(ys.mean())
    r = np.hypot(xs - mx, ys - my)
    theta = np.degrees(np.arctan2(ys - my, xs - mx)) % 360.0
    nbins = 360
    prof = np.zeros(nbins)
    idx = theta.astype(int) % nbins
    np.maximum.at(prof, idx, r)
    prof = prof - prof.mean()
    spec = np.abs(np.fft.rfft(prof))
    h = {k: float(spec[k]) for k in HARMONIC_RANGE}
    dom = max(h, key=h.get)
    return mask, h, dom, (mx, my)


def _log_stretch(raw: np.ndarray) -> np.ndarray:
    """对数拉伸到 0..1（看清暗弱变体）。"""
    vmax = float(np.percentile(raw, 99.9))
    return np.clip(np.log1p(np.clip(raw, 0, None)) / np.log1p(vmax), 0, 1)


def _linear_stretch(raw: np.ndarray) -> np.ndarray:
    vmax = float(np.percentile(raw, 99.9))
    return np.clip(raw / vmax, 0, 1)


def _mount_figure(session: Path, out: Path) -> dict[str, float]:
    """渲染主对照图（理想目标 vs 实测 seed/A1 × 3 轮 + 谐波谱）。

    Returns:
        {label: 末次迭代指标摘要} 供打印。
    """
    star_t = create_target_shape("star", 360, side=300)
    sq_t = create_target_shape("square", 360, side=300)

    fig = plt.figure(figsize=(14, 10.5))
    fig.suptitle(f"star block CCD far-field: target vs measured (session {session.name})", fontsize=13)

    n_rows = len(ROUNDS) + 1  # 顶行 = 理想目标
    grid = fig.add_gridspec(n_rows, 3, hspace=0.35, wspace=0.25)

    # 顶行: 理想目标
    axes_ideal = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    axes_ideal[0].imshow(star_t, cmap="inferno")
    axes_ideal[0].set_title("ideal star target")
    axes_ideal[1].imshow(sq_t, cmap="inferno")
    axes_ideal[1].set_title("ideal square target")
    ax_empty = fig.add_subplot(grid[0, 2])
    ax_empty.axis("off")

    results: dict[str, tuple[dict[int, float], int, float, float, str]] = {}
    for ri, rnd in enumerate(ROUNDS, start=1):
        for ci, (lab, cap_idx) in enumerate(PICK_LABELS):
            npz = session / rnd / f"capture_{cap_idx:04d}.npz"
            if not npz.exists():
                continue
            cap = load_capture(npz)
            raw = cap.raw_frame
            disp = _log_stretch(raw)
            mask, h, dom, (mx, my) = _angular_harmonics(raw)
            ax = fig.add_subplot(grid[ri, ci])
            ax.imshow(disp, cmap="inferno")
            ax.set_title(f"{rnd} {lab}  peak={cap.raw_peak:.0f} mean={cap.raw_mean:.2f}")
            ax.axis("off")
            if h is not None:
                results[(rnd, lab)] = (h, int(dom), cap.raw_peak, cap.raw_mean, f"{mx:.0f},{my:.0f}")
        # 第三列: 谐波谱
        axh = fig.add_subplot(grid[ri, 2])
        for ci, (lab, _) in enumerate(PICK_LABELS):
            if (rnd, lab) not in results:
                continue
            h, dom, _, _, _ = results[(rnd, lab)]
            ks = sorted(h)
            axh.bar(
                [f"{k}" for k in ks],
                [h[k] for k in ks],
                alpha=0.55,
                label=f"{lab}",
                color=["#e74c3c" if k == dom else "#3498db" for k in ks],
            )
        axh.set_title(f"{rnd}: angular harmonics")
        axh.set_ylabel("|FFT|")
        axh.tick_params(labelsize=7)
        axh.legend(fontsize=7)

    fig.savefig(out / "star_farfield_montage_20261107.png", dpi=110)
    plt.close(fig)
    return {f"{k[0]}/{k[1]}": v for k, v in results.items()}


def _model_figure(session: Path, out: Path) -> None:
    """渲染 model 阶段 + 线性/对数拉伸对照（排除显示伪影干扰）。"""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    fig.suptitle(f"star block CCD: model stage + stretch check ({session.name})", fontsize=12)

    def show(ax, npz, title, log=True):
        if not Path(npz).exists():
            ax.set_title(title + " (missing)")
            ax.axis("off")
            return
        cap = load_capture(npz)
        raw = cap.raw_frame
        disp = _log_stretch(raw) if log else _linear_stretch(raw)
        ax.imshow(disp, cmap="inferno")
        ax.set_title(f"{title}  peak={cap.raw_peak:.0f}")
        ax.axis("off")

    show(axes[0], session / "model.round2/capture_0001.npz", "model.r2 blk0 (log)")
    show(axes[1], session / "model.round2/capture_0002.npz", "model.r2 blk1 (log)")
    show(axes[2], session / "anchored/capture_0001.npz", "anchored A1 (linear)", log=False)
    show(axes[3], session / "anchored/capture_0001.npz", "anchored A1 (log)", log=True)
    fig.tight_layout()
    fig.savefig(out / "star_model_linear_20261107.png", dpi=110)
    plt.close(fig)


def _sampling_figure(session: Path, out: Path) -> None:
    """证明 model 预测的'方块'是 camera_sample_grids 的 in-bounds 采样窗伪影。"""
    run_dir = None
    for cand in (session, session / "anchored.round2", session / "anchored.round1", session / "anchored"):
        if (cand / "config.json").exists():
            run_dir = cand
            break
    if run_dir is None:
        return
    config = json.loads((run_dir / "config.json").read_text())
    records = json.loads((run_dir / "manifest.json").read_text())
    checkpoint = resolve_recorded_path(config["checkpoint"], relative_to=run_dir)
    params = load_frozen_parameters(checkpoint)
    recs = [r for r in records if r["label"] == "star_final"]
    if not recs:
        return
    rec = recs[0]
    with np.load(run_dir / rec["file"]) as payload:
        xy = payload["camera_xy"]
        phase = payload["model_phase"]
    fwd = FrozenDefocusForward(params, xy, device="cpu")
    fwd.exposure_ms = rec["exposure_ms"]
    grids = camera_sample_grids(
        torch, torch.as_tensor(xy, dtype=torch.float32), fwd.origin,
        fwd.inverse / fwd.axial_scale, fwd._grid_offsets,
        fft_size=params.fft_size, pitch=fwd.pitch,
    )
    g = grids[0].cpu().numpy()
    inb = (np.abs(g[..., 0]) <= 1) & (np.abs(g[..., 1]) <= 1)
    pred = fwd.predict(phase).detach().cpu().numpy()[0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "star block: model 'square' = grid_sample in-bounds window (sampling artifact)",
        fontsize=11,
    )
    axes[0].imshow(pred, cmap="magma")
    axes[0].set_title("forward.predict(phase)")
    axes[0].set_axis_off()
    axes[1].imshow(inb, cmap="gray")
    axes[1].set_title(f"grid coords in [-1,1] ({inb.mean()*100:.1f}% in-bounds)")
    axes[1].set_axis_off()
    axes[2].imshow(np.where(inb, pred, np.nan), cmap="magma")
    axes[2].set_title("pred masked to in-bounds (matches left)")
    axes[2].set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "star_model_square_artifact_20261107.png", dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=DEFAULT_SESSION, help="会话目录名（默认 20261107_135252）")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="figures 输出目录")
    args = parser.parse_args()

    session = SRC_ROOT.parent / "data/debug/slm_transfer" / args.session
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not session.is_dir():
        print(f"[star_farfield] 会话目录不存在: {session}")
        return 1

    results = _mount_figure(session, out)
    _model_figure(session, out)
    _sampling_figure(session, out)

    print(f"[star_farfield] 输出目录: {out}")
    print("[star_farfield] 角向谐波主导 (10% 阈值掩膜):")
    for key, (h, dom, peak, mean, center) in sorted(results.items()):
        spec = " ".join(f"h{k}={h[k]:.0f}" for k in sorted(h))
        print(f"  {key}: 主导 h{dom}  peak={peak:.0f} mean={mean:.2f} center=({center})  [{spec}]")
    print("[star_farfield] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
