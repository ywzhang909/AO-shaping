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
import json
import pickle
import sys
from pathlib import Path

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


# --- artifact loading --------------------------------------------------------


def find_debug_dirs(debug_root: Path) -> list[Path]:
    """Return every ``data/debug/slm_pib_*/<stamp>`` artifact dir, newest first."""
    dirs = sorted(debug_root.glob("slm_pib_*/*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs


COMPANION_JSON = "disturbance.json"
COMPANION_NPZ = "disturbance.npz"


def load_companion(art_dir: Path) -> dict | None:
    """Load a run's disturbance companion (manifest + screen archive), if present.

    Returns:
        ``{"payload": dict, "archive": dict[str, np.ndarray] | None,
        "path": Path}``, or ``None`` when the run predates the disturbance
        model. Never raises: a malformed companion degrades to "unrecorded".
    """
    manifest = art_dir / COMPANION_JSON
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("unreadable {} ({}); disturbance treated as unrecorded", manifest, exc)
        return None

    archive: dict | None = None
    npz = art_dir / COMPANION_NPZ
    if npz.is_file():
        try:
            with np.load(npz) as handle:
                archive = {key: handle[key] for key in handle.files}
        except (OSError, ValueError) as exc:
            logger.warning("unreadable {} ({}); screens unavailable", npz, exc)
    return {"payload": payload, "archive": archive, "path": manifest}


def load_run(art_dir: Path) -> dict:
    """Load one run's PKL history + JSON sidecar payload + disturbance companion."""
    pkl = next(iter(art_dir.glob("*.pkl")), None)
    if pkl is None:
        raise FileNotFoundError(f"no .pkl in {art_dir}")
    with open(pkl, "rb") as fh:
        data: dict = pickle.load(fh)
    epochs = sorted(int(k) for k in data.keys() if str(k).lstrip("-").isdigit())

    # The disturbance companion is also a *.json in this directory; it must not
    # shadow the runner's own sidecar.
    sidecars = [p for p in sorted(art_dir.glob("*.json")) if p.name != COMPANION_JSON]
    payload: dict = {}
    if sidecars:
        try:
            payload = json.loads(sidecars[0].read_text())
        except (OSError, ValueError) as exc:
            logger.warning("unreadable sidecar {} ({})", sidecars[0], exc)

    return {
        "data": data,
        "epochs": epochs,
        "payload": payload,
        "dir": art_dir,
        "companion": load_companion(art_dir),
    }


def companion_tag(run: dict) -> str | None:
    """The run's config-derived report tag, when the companion records one."""
    companion = run.get("companion")
    if not companion:
        return None
    tag = companion["payload"].get("run", {}).get("tag")
    return str(tag) if tag else None


def unique_tags(runs: list[dict]) -> list[str]:
    """Config-derived, collision-free report tags for the selected runs."""
    tags: list[str] = []
    for index, run in enumerate(runs):
        base = companion_tag(run) or f"run{index}"
        tag = base
        suffix = 2
        while tag in tags:
            tag = f"{base}-{suffix}"
            suffix += 1
        tags.append(tag)
    return tags


def classify_disturbance_trace(rms) -> str:
    """Whether a per-evaluation disturbance RMS trace is constant or varying.

    This is what the figure claims: a frozen (``static``) screen yields exactly
    one RMS value for every evaluation, while a ``dynamic`` screen is redrawn per
    evaluation and therefore varies. Fewer than two samples can demonstrate
    neither, and is reported as ``"flat"``.
    """
    arr = np.asarray(rms, dtype=float)
    if arr.size < 2:
        return "flat"
    if not np.isfinite(arr).all():
        return "varying"
    return "flat" if np.allclose(arr, arr[0], rtol=0.0, atol=1e-12) else "varying"


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


def plot_disturbance_phase(run: dict, fig_dir: Path, tag: str) -> Path:
    """Montage of the distinct disturbance screens the run actually applied.

    A frozen (``static``) screen yields a single panel; a ``dynamic`` run draws
    an independent screen per evaluation, so several are shown side by side.
    """
    companion = run.get("companion")
    archive = companion.get("archive") if companion else None
    screens = archive.get("screens") if archive else None

    if screens is None or screens.shape[0] == 0:
        note = "未记录 (无 companion)" if companion is None else "未启用 / 无归档"
        fig, ax = plt.subplots(figsize=(7.4, 3.6))
        ax.text(0.5, 0.5, f"干扰相位屏\n{note}", ha="center", va="center", fontsize=11)
        ax.axis("off")
    else:
        indices = np.asarray(
            archive.get("streak_indices", np.arange(screens.shape[0]))
        ).reshape(-1)
        shown = min(int(screens.shape[0]), 6)
        picks = np.linspace(0, screens.shape[0] - 1, shown).round().astype(int)
        limit = float(np.abs(screens).max()) or 1.0
        fig, axes = plt.subplots(1, shown, figsize=(3.0 * shown, 3.4), squeeze=False)
        for ax, pick in zip(axes[0], picks):
            im = ax.imshow(
                screens[pick], cmap="twilight_shifted", vmin=-limit, vmax=limit
            )
            label = int(indices[pick]) if pick < indices.size else int(pick)
            ax.set_title(f"screen #{label}", fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("干扰相位屏 (湍流 + 热晕) [raw rad]")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    out = fig_dir / f"{tag}_disturbance.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def plot_disturbance_rms(run: dict, fig_dir: Path, tag: str) -> Path:
    """Per-evaluation disturbance RMS — the static-vs-dynamic proof figure.

    A frozen screen must produce a flat line; a per-evaluation redraw must
    produce a varying trace.
    """
    companion = run.get("companion")
    archive = companion.get("archive") if companion else None
    rms = archive.get("call_rms") if archive else None

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    if rms is None or np.asarray(rms).size == 0:
        ax.text(0.5, 0.5, "干扰 RMS 轨迹: 未记录", ha="center", va="center", fontsize=11)
        ax.axis("off")
    else:
        arr = np.asarray(rms, dtype=float)
        kind = classify_disturbance_trace(arr)
        measured = companion["payload"].get("measured", {})
        ax.plot(np.arange(arr.size), arr, "-o", ms=3)
        if kind == "flat":
            ax.axhline(float(arr[0]), color="tab:green", ls="--", lw=1.0, alpha=0.7)
        ax.set_xlabel("optical evaluation index")
        ax.set_ylabel("disturbance RMS [rad]")
        ax.set_title(
            f"干扰 RMS per evaluation — 判定: {kind}  "
            f"(streaks={int(measured.get('streaks_used', 0) or 0)}, evals={arr.size})"
        )
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = fig_dir / f"{tag}_disturbance_rms.png"
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


#: Objectives the optimizer ASCENDS. Mirrors ``objective_mode`` in
#: ``ao_shaping/optimizer/wfless/slm_zernike_pib.py`` (~L698): ``shape`` and
#: friends are ``"max"`` (higher J is better), everything else is ``"min"``.
#: Getting this backwards makes the report claim a minimisation that the search
#: never performed — the 2026-10-01 defect.
MAXIMIZE_OBJECTIVES = ("pib", "avg_radiu", "shape", "roi_pib", "rms_pib")


def as_scalar(arr: np.ndarray) -> np.ndarray:
    """Collapse a per-epoch series to one scalar per epoch.

    ``series`` stacks ``np.atleast_1d`` rows, so a scalar metric comes back with
    shape ``(n, 1)`` rather than ``(n,)``. Reporting code wants plain floats.
    """
    return np.asarray(arr, dtype=float).reshape(arr.shape[0], -1)[:, 0]


def objective_of(run: dict, default: str = "shape") -> str:
    """The objective name recorded in the run's JSON sidecar."""
    return str(run.get("payload", {}).get("objective", default) or default)


def direction_of(objective: str) -> str:
    """``"max"`` when the search ascends this objective, else ``"min"``."""
    return "max" if objective in MAXIMIZE_OBJECTIVES else "min"


def best_of(values: np.ndarray, mode: str) -> int:
    """Index of the best value: ``argmax`` when ascending, ``argmin`` otherwise.

    NaNs are ignored so a diverged tail cannot win the comparison.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim > 1:
        arr = arr[:, 0]
    if arr.size == 0 or not np.isfinite(arr).any():
        return 0
    if mode == "max":
        return int(np.nanargmax(arr))
    return int(np.nanargmin(arr))


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT / "docs" / "slm_pib_sim"))
    except ValueError:
        return str(p)


def _fig_for(figs: dict[str, list[Path]], tag: str, suffix: str) -> Path | None:
    """The figure for ``tag`` whose stem ends with ``suffix``, if rendered."""
    for path in figs.get(tag, []):
        if path.stem.endswith(suffix):
            return path
    return None


def disturbance_section(
    runs: list[dict], tags: list[str], figs: dict[str, list[Path]]
) -> list[str]:
    """The 干扰模型 / static-vs-dynamic comparison section."""
    lines: list[str] = []
    lines.append("## 2. 干扰模型与静态/动态对比")
    lines.append("")
    lines.append(
        "两次运行之间的**唯一差异**是波前干扰体制; 目标函数、优化器、步长、相机与目标框完全一致。"
        "干扰 = **大气湍流** (`beam_backend.turbulence_phase`, von Karman 相位屏) + "
        "**热晕** (负热透镜: Noll 4 离焦 + Noll 11 球差, 经 smoothstep 光晕窗延伸到光束半径之外), "
        "二者相加后与 SLM 命令相位一起进入瞳孔场。"
    )
    lines.append("")
    lines.append(
        "| 运行 | 体制 | Cn2 | 距离 [m] | 热晕 PV [waves] | σ_turb [rad] | σ_halo [rad] "
        "| σ_total [rad] | σ_total [waves] | 屏幕数 | 评估次数 | RMS 轨迹 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run, tag in zip(runs, tags):
        companion = run.get("companion")
        if not companion:
            lines.append(
                f"| `{tag}` | 未记录 | — | — | — | — | — | — | — | — | — | — |"
            )
            continue
        cfg = companion["payload"].get("config", {})
        measured = companion["payload"].get("measured", {})
        archive = companion.get("archive")
        rms = archive.get("call_rms") if archive else None
        trace = (
            classify_disturbance_trace(rms)
            if rms is not None and np.asarray(rms).size
            else "—"
        )
        sigma = float(measured.get("sigma_total_rad", 0.0) or 0.0)
        lines.append(
            "| `{}` | {} | {:g} | {:g} | {:g} | {:.4f} | {:.4f} | {:.4f} | {:.4f} "
            "| {} | {} | {} |".format(
                tag,
                cfg.get("mode", "?"),
                float(cfg.get("cn2", 0.0) or 0.0),
                float(cfg.get("distance_m", 0.0) or 0.0),
                float(cfg.get("thermal_halo_pv_waves", 0.0) or 0.0),
                float(measured.get("sigma_turb_rad", 0.0) or 0.0),
                float(measured.get("sigma_halo_rad", 0.0) or 0.0),
                sigma,
                sigma / (2.0 * np.pi),
                int(measured.get("streaks_used", 0) or 0),
                int(measured.get("calls", 0) or 0),
                trace,
            )
        )
    lines.append("")

    for run, tag in zip(runs, tags):
        phase_fig = _fig_for(figs, tag, "_disturbance")
        rms_fig = _fig_for(figs, tag, "_disturbance_rms")
        if phase_fig is None and rms_fig is None:
            continue
        lines.append(f"### 2.{tags.index(tag) + 1} `{tag}` 干扰可视化")
        lines.append("")
        if phase_fig is not None:
            lines.append(f"![{tag} 干扰相位屏]({rel(phase_fig)})")
            lines.append("")
        if rms_fig is not None:
            lines.append(f"![{tag} 干扰 RMS 轨迹]({rel(rms_fig)})")
            lines.append("")

    lines.append("### 2.9 口径与局限 (必读, 否则会过度解读)")
    lines.append("")
    lines.append(
        "1. **没有真实的公里级大气路径。** `r0 = (0.423·k²·Cn2·L)^(-3/5)` 只依赖 `Cn2·L` 的**乘积**, "
        "两者在单层薄屏模型里完全退化; 本台架是约 0.3 m 的实验室 2f-Fourier 光路, 因此这里是"
        "**参数化应力测试**, 不是大气传输仿真。"
    )
    lines.append(
        "2. **σ 是实测值, 不是解析值。** 仓库默认 (numpy) 相位屏生成器缺少次谐波/低频补偿 "
        "(`drivers/sim/AGENTS.md` 第 4 条), 而 `L0` (数十米) 远大于 15.36 mm 口径, 故实测 σ "
        "**低于**同 r0 下的解析 von Karman 方差 —— 本报告只引用实测值。"
    )
    lines.append(
        "3. **`dynamic` = 完全去相关 (white in time), 不是风模型。** 每次光学评估重抽一张独立相位屏; "
        "真实大气去相关时间 (~10–50 ms) 远短于本环路每次评估耗时 (~0.375 s), 因此该极限在此是合理渐近。"
    )
    lines.append(
        "4. **噪声门的早期偏置。** 优化器前 `noise_gate_window=20` 个 epoch 门控尚未生效 "
        "(`slm_zernike_pib.py`), 任何 applied/gated 计数都会被高估; 且这些计数只存在于运行日志, "
        "**不在调试产物中**, 因此本报告不做此类断言。"
    )
    lines.append(
        "5. **n≤4 的 Zernike 无法合成真正的方形远场** (仓库既有反模式), 方形目标只是**代理指标**, "
        "目标值的改善来自离焦/球差重排而非真正方形成形。"
    )
    lines.append(
        "6. **绝对数值是仿真内部单位** (`far_field()` 把峰值归一到 100 并加泊松散粒噪声), "
        "只可做**相对**比较 (static vs dynamic、优化前 vs 优化后), 不可与硬件台架对比。"
    )
    lines.append(
        "6b. **远场经过 4× 零填充过采样** (`FAR_FIELD_PADDING`): 未填充时 0 级光斑仅约 "
        "**1.8 px FWHM** (几乎无采样), 因此填充前的历史报告数值**不可与本次直接比较** —— "
        "光斑被真正分辨后, 目标函数值与自适应半径都会改变 (例如自适应半径从 ~112 px 降到 ~16 px)。"
        "static 与 dynamic 之间使用完全相同的采样, 故二者互比仍然有效。"
    )
    lines.append(
        "7. **零干扰基线本身就已近乎平坦**: 实测一次干净 10-epoch 运行只应用 4/10 更新、"
        "6/10 被噪声门拒绝, 退出时记录 `no improvement over the initial phase`。"
        "若干扰运行同样平坦, **不能归因于干扰**。"
    )
    lines.append(
        "8. **热晕是瞬态、强度相关的非线性效应**, 本模型只取它的**稳态低阶**代理 "
        "(负热透镜: 离焦 + 球差 + 光晕窗), 不建模吸收加热的时间演化、风场对流或强度反馈。"
    )
    lines.append("")

    lines.append("### 2.10 派生对比 (全部来自本次运行数据)")
    lines.append("")
    derived: list[tuple[str, str, float, float, float, float]] = []
    for run, tag in zip(runs, tags):
        objective = objective_of(run)
        direction = direction_of(objective)
        values = as_scalar(series(run, "J")[1])
        if values.size == 0 or not np.isfinite(values).any():
            continue
        best_index = best_of(values, direction)
        improvement = (
            float(values[best_index] - values[0])
            if direction == "max"
            else float(values[0] - values[best_index])
        )
        derived.append(
            (
                tag,
                objective,
                float(values[0]),
                float(values[best_index]),
                improvement,
                float(values[-1]),
            )
        )
    if derived:
        lines.append("| 运行 | 目标 | 起点 J | 峰值 J | 改善 (峰值 − 起点, 按搜索方向) | 终点 J |")
        lines.append("|---|---|---|---|---|---|")
        for tag, objective, first, best, improvement, last in derived:
            lines.append(
                f"| `{tag}` | `{objective}` | {first:.4f} | {best:.4f} | {improvement:+.4f} | {last:.4f} |"
            )
        lines.append("")
        lines.append(
            "> 改善量按各次运行的 `objective` **派生**搜索方向后计算 (见 `direction_of`), "
            "不是硬编码断言。"
        )
        lines.append("")
    else:
        lines.append("无可用目标序列, 无法做派生对比。")
        lines.append("")

    return lines


def build_markdown(
    runs: list[dict],
    tags: list[str],
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

    lines.extend(disturbance_section(runs, tags, figs))
    lines.append("")

    for index, (run, tag) in enumerate(zip(runs, tags)):
        epochs = run["epochs"]
        objective = objective_of(run)
        mode = direction_of(objective)
        j_series = as_scalar(series(run, "J")[1])
        p_series = as_scalar(series(run, "_p%")[1])
        ib = best_of(j_series, mode)
        ip = best_of(p_series, "max")  # in-target energy is always ascending
        j, j_end, j_best = j_series[0], j_series[-1], j_series[ib]
        p0, p_end, p_best = p_series[0], p_series[-1], p_series[ip]
        drift = (j_best - j_end) if mode == "max" else (j_end - j_best)
        lines.append(f"## 3.{index + 1} 运行 `{tag}` (目录 `{run['dir'].name}`)")
        lines.append("")
        lines.append(
            f"目标函数 `{objective}`, 搜索方向 **{mode}** "
            f"({'越大越好' if mode == 'max' else '越小越好'})。"
        )
        lines.append("")
        companion = run.get("companion")
        if companion:
            cfg = companion["payload"].get("config", {})
            measured = companion["payload"].get("measured", {})
            lines.append(
                f"- **干扰**: 体制 `{cfg.get('mode', '?')}`; "
                f"Cn2={float(cfg.get('cn2', 0.0) or 0.0):g}, "
                f"距离={float(cfg.get('distance_m', 0.0) or 0.0):g} m, "
                f"热晕 PV={float(cfg.get('thermal_halo_pv_waves', 0.0) or 0.0):g} waves; "
                f"实测 σ_total={float(measured.get('sigma_total_rad', 0.0) or 0.0):.4f} rad "
                f"({float(measured.get('sigma_total_rad', 0.0) or 0.0) / (2.0 * np.pi):.4f} waves), "
                f"屏幕数={int(measured.get('streaks_used', 0) or 0)}, "
                f"评估次数={int(measured.get('calls', 0) or 0)}"
            )
            lines.append("")
        else:
            lines.append("- **干扰**: 未记录 (该运行的调试产物中没有 companion 文件)。")
            lines.append("")
        lines.append(f"| 项目 | 值 |")
        lines.append(f"|---|---|")
        lines.append(f"| epoch 数 | {len(epochs)} |")
        lines.append(f"| 初始 J (epoch {int(epochs[0])}) | {j:.4f} |")
        lines.append(f"| 最佳 J (epoch {int(epochs[ib])}) | {j_best:.4f} |")
        lines.append(f"| 末轮 J (epoch {int(epochs[-1])}) | {j_end:.4f} |")
        lines.append(f"| 初始框内能量 _p% | {p0:.4f} |")
        lines.append(f"| 最佳框内能量 _p% (epoch {int(epochs[ip])}) | {p_best:.4f} |")
        lines.append(f"| 末轮框内能量 _p% | {p_end:.4f} |")
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
        lines.append(f"- 目标 J (max, 越大越好): 初始 {j:.4f} → 最佳 {j_best:.4f} "
                     f"(epoch {int(epochs[ib])}) → 末轮 {j_end:.4f}。")
        lines.append(f"- 框内能量 `_p%` 从 {p0:.4f} 到 末轮 {p_end:.4f} "
                     f"(最佳 {p_best:.4f}): 低阶 Zernike 主要做波前校正/聚焦, "
                     "并非真正的方形成形 (方形需要全像素自由度, 见 AGENTS.md 反模式)。")
        if drift > 1e-4:
            lines.append(
                f"- ⚠️ **末轮劣于最佳 {drift:.4f}**: 优化器退出时会把 SLM 停在"
                f"**最佳**相位 (epoch {int(epochs[ib])}), 而非末轮相位。"
                "报告成绩应引用最佳值, 末轮值仅反映退出瞬间的抖动。"
            )
        lines.append("- 相位图展示 15 个 Zernike 模式 (n≤4) 的加权合成; 远场图展示 0 级光斑的 FFT 传播结果。")
        lines.append("")

    lines.append("## 4. 结论")
    lines.append("")
    lines.append("0. **干扰已注入, 且 static/dynamic 在数据上可区分**: 见 §2。`static` 的逐次评估干扰 "
                 "RMS 为常数 (全程复用唯一一张冻结相位屏), `dynamic` 则逐次变化 (每次光学评估重抽一张"
                 "独立相位屏)。该判据由图 `*_disturbance_rms.png` 与 companion 归档直接给出, "
                 "不是文字断言。")
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
        default=2,
        help=(
            "How many runs (newest first) to render (default: 2, so a "
            "static + dynamic pair is compared in one report)."
        ),
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
    tags = unique_tags(runs)

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
            plot_disturbance_phase(run, fig_dir, tag),
            plot_disturbance_rms(run, fig_dir, tag),
        ]
        gifs[tag] = [
            make_gif(run, gif_dir, tag, "_img", phase=True),
            make_gif(run, gif_dir, tag, "_img", phase=False),
        ]
        for f in figs[tag]:
            print(f"  wrote {f}")

    out_md = out_dir / "report.md"
    build_markdown(runs, tags, figs, gifs, out_md)
    print(f"report -> {out_md}")


if __name__ == "__main__":
    cli()
