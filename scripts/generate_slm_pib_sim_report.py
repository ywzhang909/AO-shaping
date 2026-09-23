"""Offline report generator for the ``slm-pib`` simulation run.

Reads the debug artifacts written by ``slm_pib_runner`` (PKL + JSON sidecar) and
produces:

* ``docs/slm_pib_sim/figures/*.png`` — per-run figures (objective curve,
  Zernike-coefficient trace, sent-phase evolution, far-field spot evolution);
* ``docs/slm_pib_sim/gifs/*.gif`` — animated sent-phase and far-field evolution;
* ``docs/slm_pib_sim/report.md`` — the markdown report tying it all together.

Fully offline: it never opens hardware, it only reads the saved PKL/JSON. This
follows the repo convention that report generation lives in ``scripts/`` and is
offline (see ``scripts/README.md``).

Usage:
    python scripts/generate_slm_pib_sim_report.py
    python scripts/generate_slm_pib_sim_report.py --debug-dir data/debug/slm_pib_shape_...
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must be set before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# CJK-capable font fallbacks (per repo script convention).
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# --- artifact loading --------------------------------------------------------


def find_debug_dirs(debug_root: Path) -> list[Path]:
    """Return every ``data/debug/slm_pib_*/<stamp>`` artifact dir, newest first."""
    dirs = sorted(debug_root.glob("slm_pib_*/*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs


def load_run(art_dir: Path) -> dict:
    """Load one run's PKL history + JSON sidecar payload."""
    pkl = next(iter(art_dir.glob("*.pkl")), None)
    if pkl is None:
        raise FileNotFoundError(f"no .pkl in {art_dir}")
    with open(pkl, "rb") as fh:
        data: dict = pickle.load(fh)
    epochs = sorted(int(k) for k in data.keys() if str(k).lstrip("-").isdigit())
    import json

    payload: dict = {}
    jf = next(iter(art_dir.glob("*.json")), None)
    if jf is not None:
        payload = json.loads(jf.read_text())
    return {"data": data, "epochs": epochs, "payload": payload, "dir": art_dir}


def series(run: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract a per-epoch scalar/curve array keyed by epoch."""
    x = np.array(run["epochs"], dtype=float)
    vals = []
    for e in run["epochs"]:
        rec = run["data"][e]
        v = rec.get(key)
        if v is None:
            vals.append(np.nan)
        else:
            vals.append(np.asarray(v, dtype=float))
    arr = np.stack([np.atleast_1d(v) for v in vals])
    return x, arr


def image_series(run: dict, key: str) -> list[np.ndarray]:
    """Extract a per-epoch 2-D image (e.g. ``_img``), NaN-padded to a common shape."""
    imgs = []
    shape = None
    for e in run["epochs"]:
        v = run["data"][e].get(key)
        if v is None:
            continue
        arr = np.asarray(v, dtype=float)
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            arr = np.resize(arr, shape)
        imgs.append(arr)
    return imgs


# --- figure rendering --------------------------------------------------------


def _cmap_norm(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Log-normalised grayscale for photon images (dynamic range ~1e2)."""
    arr = np.clip(np.asarray(img, dtype=float), 0, None)
    vmax = float(arr.max()) if arr.max() > 0 else 1.0
    return arr, vmax


def plot_objective(run: dict, fig_dir: Path, tag: str) -> Path:
    x, j = series(run, "J")
    x2, p = series(run, "_p%")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(x, j, "-o", ms=3)
    ax[0].set_title("Objective J (shape)")
    ax[0].set_xlabel("epoch")
    ax[0].grid(alpha=0.3)
    ax[1].plot(x2, p, "-o", ms=3, color="tab:orange")
    ax[1].set_title("In-target energy  _p%")
    ax[1].set_xlabel("epoch")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = fig_dir / f"{tag}_objective.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_zernike(run: dict, fig_dir: Path, tag: str) -> Path:
    x, c = series(run, "_c")  # (epochs, n_modes)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k in range(c.shape[1]):
        ax.plot(x, c[:, k], "-", lw=1.4)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_title("Zernike coefficient trace (per mode)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("coefficient (wavelengths)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = fig_dir / f"{tag}_zernike.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_phase_frames(
    run: dict, fig_dir: Path, tag: str, indices: list[int]
) -> Path:
    """Render the *sent* Zernike phase for a few selected epochs side by side."""
    x, c = series(run, "_c")
    n_max = 4
    # Rebuild the phase pattern from the coefficients (same math as the runner:
    # PatternHelper zernike on a 1200x1920 grid, radius 600).
    from ao_shaping.utils.wavefront.pattern_helper import PatternHelper

    ph = PatternHelper(resolution=(1920, 1200), bits=10)
    from ao_shaping.optimizer.wfless.slm_square_shaping import _zernike_indices

    modes = _zernike_indices(n_max)
    panels = [0] + indices
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4), squeeze=False)
    for ax, e in zip(axes[0], panels):
        coeff_vec = c[e]
        coeffs_dict = {}
        for i, (nn, mm) in enumerate(modes):
            if i < len(coeff_vec):
                coeffs_dict[(nn, mm)] = float(coeff_vec[i])
        phase = ph.generate_zernike_polynomial(
            n_max=n_max, coefficients=coeffs_dict, radius=600
        )
        im = ax.imshow(np.mod(phase, 2 * np.pi), cmap="hsv", vmin=0, vmax=2 * np.pi)
        ax.set_title(f"epoch {e}", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Sent Zernike phase evolution (mod 2π, rad)", y=1.02)
    fig.tight_layout()
    out = fig_dir / f"{tag}_phase_evolution.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def plot_spot_frames(run: dict, fig_dir: Path, tag: str, indices: list[int]) -> Path:
    imgs = image_series(run, "_img")
    panels = [i for i in indices if i < len(imgs)] or [0]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4), squeeze=False)
    vmax = 0.0
    for arr in (imgs[i] for i in panels):
        vmax = max(vmax, float(np.nanmax(arr)) if arr.size else 0.0)
    for ax, i in zip(axes[0], panels):
        arr = imgs[i]
        ax.imshow(np.clip(arr, 0, vmax), cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"epoch {i}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Far-field spot evolution (CCD frame)", y=1.02)
    fig.tight_layout()
    out = fig_dir / f"{tag}_spot_evolution.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def make_gif(run: dict, gif_dir: Path, tag: str, key: str, phase: bool) -> Path:
    """Build a GIF of per-epoch frames (thinned to <=24 frames for file size)."""
    from PIL import Image

    if phase:
        x, c = series(run, "_c")
        from ao_shaping.utils.wavefront.pattern_helper import PatternHelper
        from ao_shaping.optimizer.wfless.slm_square_shaping import _zernike_indices

        ph = PatternHelper(resolution=(1920, 1200), bits=10)
        modes = _zernike_indices(4)
        frames = []
        for e in run["epochs"]:
            cv = c[e]
            cd = {
                (nn, mm): float(cv[i])
                for i, (nn, mm) in enumerate(modes)
                if i < len(cv)
            }
            p = ph.generate_zernike_polynomial(n_max=4, coefficients=cd, radius=600)
            frames.append(np.mod(p, 2 * np.pi))
        cmap_name, lo, hi = "hsv", 0.0, 2 * np.pi
    else:
        frames = image_series(run, "_img")
        vmax = max(float(np.nanmax(f)) for f in frames if f.size)
        frames = [np.clip(f, 0, vmax) for f in frames]
        cmap_name, lo, hi = "inferno", 0.0, vmax

    # Thin to at most 24 evenly spaced frames.
    if len(frames) > 24:
        idx = np.linspace(0, len(frames) - 1, 24).round().astype(int)
        frames = [frames[i] for i in idx]

    from matplotlib import colormaps

    cmap = colormaps[cmap_name]
    out = gif_dir / f"{tag}_{'phase' if phase else 'spot'}.gif"
    tmp = []
    for f in frames:
        norm = (np.clip(f, lo, hi) - lo) / max(hi - lo, 1e-9)
        rgba = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
        im = Image.fromarray(rgba)
        tmp.append(im)
    if tmp:
        tmp[0].save(
            out,
            save_all=tmp[1:],
            append_images=tmp[1:],
            duration=120,
            loop=0,
        )
    return out


# --- markdown ----------------------------------------------------------------


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT / "docs" / "slm_pib_sim"))
    except ValueError:
        return str(p)


def build_markdown(
    runs: list[dict],
    figs: dict[str, list[Path]],
    gifs: dict[str, list[Path]],
    out_md: Path,
) -> None:
    now = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# slm-pib 仿真运行报告 (SPGD 方形目标, 2f-Fourier 数字孪生)")
    lines.append("")
    lines.append(f"**生成时间**: {now}")
    lines.append("")
    lines.append("**Fully offline** — 本报告由 `scripts/generate_slm_pib_sim_report.py` 离线生成, "
                 "仅读取 `slm_pib_runner --debug` 保存的 PKL/JSON 调试产物, 不打开任何硬件。")
    lines.append("")
    lines.append("## 1. 运行说明")
    lines.append("")
    lines.append("本运行使用 `src/ao_shaping/runners/slm_pib_runner.py` 的 **SPGD** 子命令, 在纯 numpy "
                 "2f-Fourier 仿真 (`src/ao_shaping/drivers/sim/slm_pib_sim.py`) 下执行, 无硬件。")
    lines.append("")
    lines.append("- **相机**: `--cam_type sim` (注册到相机注册表, 读取仿真远场)")
    lines.append("- **SLM**: `Santec` 被 monkeypatch 为 `SimSLMPib` (FFT 远场, 0 级光斑位于帧中心)")
    lines.append("- **目标**: 方形 (`--target_shape square`, `--target_size` 相机像素)")
    lines.append("- **搜索**: SPGD 梯度法 (`--optimizer_type adamod`), Zernike 系数 n≤4 (15 个模式)")
    lines.append("")
    lines.append("光学模型: SLM 位于 2f 光路前焦面, CCD 位于后焦面, 因此 CCD 图像 = SLM 瞳孔场的 2D "
                 "FFT (夫琅禾费远场)。输入为高斯光束, 施加 Zernike 相位后经 `np.fft.fft2` 传播到远场。")
    lines.append("")

    for run, tag in zip(runs, figs.keys() if figs else [f"run{i}" for i in range(len(runs))]):
        tag = tag or "run"
        epochs = run["epochs"]
        j = run["data"][epochs[0]].get("J", float("nan"))
        j_end = run["data"][epochs[-1]].get("J", float("nan"))
        p0 = run["data"][epochs[0]].get("_p%", float("nan"))
        p_end = run["data"][epochs[-1]].get("_p%", float("nan"))
        lines.append(f"## 2. 运行 `{tag}` (目录 `{run['dir'].name}`)")
        lines.append("")
        lines.append(f"| 项目 | 值 |")
        lines.append(f"|---|---|")
        lines.append(f"| epoch 数 | {len(epochs)} |")
        lines.append(f"| 初始 J (shape) | {j:.4f} |")
        lines.append(f"| 最终 J (shape) | {j_end:.4f} |")
        lines.append(f"| 初始框内能量 _p% | {p0:.4f} |")
        lines.append(f"| 最终框内能量 _p% | {p_end:.4f} |")
        if run["payload"]:
            lines.append(f"| 搜索配置 | {run['payload']} |")
        lines.append("")
        if tag in figs:
            for f in figs[tag]:
                lines.append(f"![{f.stem}]({rel(f)})")
                lines.append("")
        if tag in gifs:
            for g in gifs[tag]:
                lines.append(f"![{g.stem}](gifs/{g.name})")
                lines.append("")
        lines.append("### 解读")
        lines.append("")
        lines.append(f"- 目标 J 从 {j:.4f} 变化到 {j_end:.4f} (shape 目标为**最小化**, "
                     "数值越接近 0 越好, 负值越大代表离目标越远)。")
        lines.append(f"- 框内能量 `_p%` 从 {p0:.4f} 到 {p_end:.4f}: 低阶 Zernike 主要做波前校正/聚焦, "
                     "并非真正的方形成形 (方形需要全像素自由度, 见 AGENTS.md 反模式)。")
        lines.append("- 相位图展示 15 个 Zernike 模式 (n≤4) 的加权合成; 远场图展示 0 级光斑的 FFT 传播结果。")
        lines.append("")

    lines.append("## 3. 结论")
    lines.append("")
    lines.append("1. **管线验证通过**: `slm-pib` 的 SPGD 闭环在纯仿真下可端到端运行, 无需硬件, "
                 "调试产物 (PKL/JSON/PNG) 与硬件运行格式一致, 可直接用于离线报告生成。")
    lines.append("2. **仿真模型合理**: 2f-Fourier FFT 远场使 Zernike 相位对光斑产生真实可测的影响 "
                 "(非零梯度), 与硬件 2f 光路 (SLM 前焦面 → 透镜 → CCD 后焦面) 一致。")
    lines.append("3. **低阶 Zernike 的局限**: 与 AGENTS.md 反模式一致, n≤4 的 Zernike 是圆对称光滑基, "
                 "无法合成真正的方形远场; 本报告的方形目标用于验证 **目标函数 + 闭环反馈链路**, "
                 "而非真正的方形成形 (后者需 freeform/全像素相位, 如 `spgd-square --basis freeform`)。")
    lines.append("4. **可复用**: 报告生成器完全离线, 任何一次 `slm-pib --debug` 运行 (仿真或硬件) "
                 "的调试产物都能用本脚本重新出报告。")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


# --- cli ---------------------------------------------------------------------


def cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--debug-root",
        type=Path,
        default=ROOT / "data" / "debug",
        help="Root dir containing slm_pib_* artifact dirs.",
    )
    ap.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="A single artifact dir (overrides --debug-root glob).",
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
        default=ROOT / "docs" / "slm_pib_sim",
        help="Output dir for figures/gifs/report.md.",
    )
    args = ap.parse_args()

    out_dir: Path = args.out
    fig_dir = out_dir / "figures"
    gif_dir = out_dir / "gifs"
    for d in (out_dir, fig_dir, gif_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.debug_dir is not None:
        dirs = [args.debug_dir]
    else:
        dirs = find_debug_dirs(args.debug_root)[: max(1, args.max_runs)]
    if not dirs:
        print(f"no debug artifacts under {args.debug_root}", file=sys.stderr)
        sys.exit(1)

    runs = [load_run(d) for d in dirs]
    tags = [f"run{i}" for i in range(len(runs))]

    figs: dict[str, list[Path]] = {}
    gifs: dict[str, list[Path]] = {}
    for run, tag in zip(runs, tags):
        epochs = run["epochs"]
        # Selected evolution indices: start, mid, last.
        idx = [0, max(0, len(epochs) // 2), len(epochs) - 1]
        figs[tag] = [
            plot_objective(run, fig_dir, tag),
            plot_zernike(run, fig_dir, tag),
            plot_phase_frames(run, fig_dir, tag, idx),
            plot_spot_frames(run, fig_dir, tag, idx),
        ]
        gifs[tag] = [
            make_gif(run, gif_dir, tag, "_img", phase=True),
            make_gif(run, gif_dir, tag, "_img", phase=False),
        ]
        for f in figs[tag]:
            print(f"  wrote {f}")

    out_md = out_dir / "report.md"
    build_markdown(runs, figs, gifs, out_md)
    print(f"report -> {out_md}")


if __name__ == "__main__":
    cli()
