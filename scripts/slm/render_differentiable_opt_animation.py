"""Render the differentiable-opt **process animation** (GIF) + curve figure.

This is the *companion* renderer for
``report/slm/differentiable_opt/differentiable_opt_report.md`` (produced by
``scripts/slm/hw_differentiable_phase_opt.py``). The main report has 34
static figures but **no animation**, so the "what did the optimiser actually
do" question can only be read off numbers.

What this script does (all offline, no instrument is touched):

1. **Model side** -- re-runs the pure-model reverse-propagation loop
   (:class:`~ao_shaping.algorithm.signal_processing.differentiable_beam.DifferentiableBeamOptimizer`)
   **offline**, from the same initial phase and the same seed as the hardware
   run, and stores the model far-field every few iterations. The forward
   model / target construction are reused from the main script (loaded by path
   with :mod:`importlib`) so the two never drift apart.
   The reproduction is *verified*: the loss trace is compared against the
   ``history.csv`` the hardware run wrote, and the worst relative difference
   is recorded in ``animation_meta.json``.
2. **Hardware side** -- the measurement-anchored steps: the saved CCD frames
   are resampled onto the model far-field grid, so model frames and measured
   frames are directly comparable (same axes, same target overlay).
3. **GIF** -- target -> model far-field evolving -> measured frames, with the
   live loss / gradient norm / phase RMS on every frame and the loss & gradient
   curves advancing in a side panel (the 5 measurement anchors are marked).
4. **Curve figure** -- model-side loss and gradient norm versus iteration,
   with the 5 measurement anchors overlaid on the same axes (concatenated
   timeline: the hardware closed loop continues the *same* optimiser, so the
   anchors sit right after the model-side iterations) and the adaptive
   exposure (2.0 -> 72.0 ms) annotated.

**Degradation.** If the model side cannot be reproduced offline (no torch, no
initial phase, import failure, ...), the script falls back to a
"measured -> optimised" animation built from the saved artefacts and records
the reason in ``animation_meta.json``; the report must then say so.

Usage::

    python scripts/slm/render_differentiable_opt_animation.py
    python scripts/slm/render_differentiable_opt_animation.py \
        --run-dir data/debug/differentiable_opt/20261022_213000
    python scripts/slm/render_differentiable_opt_animation.py --force-degrade
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# `scripts.*` is imported as a package path, so the repo ROOT itself must be
# importable; `src` must stay in front of `scripts` (they both provide a
# package named `ml`).
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")  # must precede pyplot

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Main script (hardware run + model definition) -- reused read-only.
MAIN_SCRIPT = ROOT / "scripts" / "slm" / "hw_differentiable_phase_opt.py"
#: Default report series directory (same series as the main report).
DEFAULT_REPORT_DIR = ROOT / "report" / "slm" / "differentiable_opt"
#: Where the finished runs live.
DEFAULT_RUN_ROOT = ROOT / "data" / "debug" / "differentiable_opt"

#: Side of the far-field window kept for the animation (model samples,
#: centred on the DC cell). The full model grid is 1024x1024 = 2004 CCD px;
#: the interesting structure (zero order + speckle halo + target square)
#: lives within +-128 samples, so the animation shows that window.
FARFIELD_WINDOW = 256

#: Animation defaults (frames / fps / dpi).
DEFAULT_FRAMES = 40
DEFAULT_FPS = 5
DEFAULT_DPI = 88
#: A panel must stay readable in the report column -> at least this many px.
MIN_PANEL_PX = 300

#: Colours (same palette as the main report's figures).
COLOR_MODEL = "#27ae60"
COLOR_HW = "#c0392b"
COLOR_BLUE = "#2f6fb0"
COLOR_GRID = "#888888"

CJK_FONTS = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans")


def setup_cjk_font() -> None:
    """Give matplotlib a CJK font (falls back silently when none is present)."""
    matplotlib.rcParams["font.sans-serif"] = list(CJK_FONTS)
    matplotlib.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of dicts (BOM tolerant)."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a CSV (the repo convention for run-directory artefacts)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_main_module(path: Path = MAIN_SCRIPT) -> ModuleType:
    """Load the hardware-run script **by path** (it is not importable as a
    package module from ``scripts/slm`` in every layout).

    ⚠️ The module must be registered in ``sys.modules`` **before**
    ``exec_module`` -- otherwise the ``@dataclass`` inside it explodes with
    ``'NoneType' object has no attribute '__dict__'`` (the class body resolves
    ``__module__`` against ``sys.modules``, which is empty for a module that
    was never registered).
    """
    spec = importlib.util.spec_from_file_location("hw_differentiable_phase_opt", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"无法加载主脚本: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The main script pushes ROOT/"scripts" to the front of sys.path; put
    # ROOT/"src" back in front so `ml` keeps resolving to src/ml.
    src = str(ROOT / "src")
    if sys.path[0] != src:
        sys.path.insert(0, src)
    return module


def find_run_dir(explicit: str | None) -> Path:
    """Resolve the run directory (explicit path, or the newest finished run)."""
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(f"运行目录不存在: {path}")
        return path
    candidates = sorted(
        (p for p in DEFAULT_RUN_ROOT.glob("*") if (p / "run_config.json").is_file()),
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(f"{DEFAULT_RUN_ROOT} 下没有带 run_config.json 的运行目录")
    return candidates[-1]


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    """One finished hardware run + its configuration."""

    run_dir: Path
    config: dict[str, Any]
    params: dict[str, Any]
    stamp: str

    @property
    def iters(self) -> int:
        return int(self.params.get("iters", 0))

    @property
    def hw_iters(self) -> int:
        return int(self.params.get("hw_iters", 0))

    @property
    def k_mean(self) -> float:
        return float(self.config.get("geometry", {}).get("k_mean", 0.0))


def load_run(run_dir: Path) -> RunContext:
    """Read ``run_config.json`` and build the context."""
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    params = dict(config.get("params", {}))
    params.setdefault("iters", config.get("model", {}).get("iters", 0))
    params.setdefault("hw_iters", config.get("model", {}).get("hw_iters", 0))
    return RunContext(
        run_dir=run_dir,
        config=config,
        params=params,
        stamp=run_dir.name,
    )


# ---------------------------------------------------------------------------
# Model side (offline reproduction)
# ---------------------------------------------------------------------------


def build_model_inputs(
    main_mod: ModuleType,
    ctx: RunContext,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    """Rebuild (spec, source amplitude, target, target info) with the main
    script's own functions -- never a second implementation."""
    params = ctx.params
    spec = main_mod.build_model_spec(
        pupil_radius=float(params["pupil_radius"]),
        pupil_center=(float(params["pupil_center"][0]), float(params["pupil_center"][1])),
        n_pupil=int(params["model_pupil_samples"]),
        padding=int(params["model_padding"]),
        w0_px=float(params["beam_w0_px"]),
    )
    amplitude = main_mod.build_source_amplitude(spec)
    target, target_info = main_mod.build_target(
        spec,
        str(params.get("target", "square")),
        float(params.get("target_side_px", 60.0)),
        ctx.k_mean,
        delta_spot_px=float(params.get("delta_spot_px", 15.0)),
    )
    return spec, amplitude, target, target_info


def select_frame_iterations(n_iters: int, n_frames: int) -> list[int]:
    """Iteration indices to snapshot (1-based, first and last always kept)."""
    if n_iters <= 0:
        return []
    n_frames = max(1, min(int(n_frames), n_iters))
    picks = np.unique(np.round(np.linspace(1, n_iters, n_frames)).astype(int)).tolist()
    if picks[0] != 1:
        picks.insert(0, 1)
    if picks[-1] != n_iters:
        picks.append(n_iters)
    return picks


def crop_farfield(arr: np.ndarray, centre_index: int, size: int) -> np.ndarray:
    """Crop a centred window (model samples) out of a far-field grid."""
    a = np.asarray(arr, dtype=np.float64)
    size = int(min(size, a.shape[0], a.shape[1]))
    lo = int(centre_index) - size // 2
    hi = lo + size
    if lo < 0:
        hi -= lo
        lo = 0
    if hi > a.shape[0]:
        lo -= hi - a.shape[0]
        hi = a.shape[0]
    lo = max(lo, 0)
    return a[lo:hi, lo:hi]


def reproduce_model_side(
    ctx: RunContext,
    main_mod: ModuleType,
    *,
    frame_iterations: list[int],
) -> dict[str, Any]:
    """Re-run the pure-model loop offline and snapshot the far-field.

    Returns a dict with the loss/gradient trace, the per-frame far-fields and
    the **verification** against the ``history.csv`` the hardware run wrote.

    Raises:
        Exception: anything that prevents the reproduction (the caller turns
            it into the degraded path).
    """
    import torch  # optional dependency, failure is handled

    from ao_shaping.algorithm.signal_processing.differentiable_beam import (
        DifferentiableBeamOptimizer,
    )

    spec, amplitude, target, _target_info = build_model_inputs(main_mod, ctx)
    init_path = ctx.run_dir / "phase_initial.npy"
    init_phase = np.load(init_path) if init_path.is_file() else None

    recorded = str(ctx.config.get("model", {}).get("device", "") or "")
    if recorded == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    elif recorded in ("cuda", "cpu"):
        device = recorded
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    opt = DifferentiableBeamOptimizer(
        target,
        source_amplitude=amplitude,
        lr=float(ctx.params.get("lr", 0.05)),
        device=device,
        seed=int(ctx.params.get("seed", 0)),
        init_phase=init_phase,
    )

    wanted = set(int(i) for i in frame_iterations)
    trace: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    frame_iters: list[int] = []
    window = int(min(FARFIELD_WINDOW, spec.m_grid))
    n_iters = ctx.iters
    for it in range(n_iters):
        # ⚠️ **帧的远场取自本次迭代*开始时*的相位** —— 因为 ``update()`` 记录的
        # 损失与梯度范数都是在那一刻算出来的 (Adam 步之后才变相位)。若改成
        # "步后相位 + 步前损失", 帧里的数字就对不上曲线上的点 (第一帧尤其
        # 明显: 1.07e-2 → 1.05e-3 那一步)。
        phase_before = np.asarray(opt.current_phase, dtype=np.float64).copy()
        if (it + 1) in wanted:
            far = main_mod.model_far_field(amplitude, phase_before)
            frames.append(crop_farfield(far, spec.centre_index, window))
            frame_iters.append(it + 1)
        opt.update()
        grad = opt.phase_tensor.grad
        gnorm = float(grad.norm()) if grad is not None else float("nan")
        phase_after = np.asarray(opt.current_phase, dtype=np.float64)
        trace.append(
            {
                "iteration": it + 1,
                "loss": float(opt.last_loss if opt.last_loss is not None else float("nan")),
                "grad_norm": gnorm,
                "phase_rms_rad": float(np.sqrt(np.mean(phase_after**2))),
            }
        )

    check = {"history_csv": False, "compared": 0, "max_rel_loss_diff": float("nan")}
    reference = read_csv_rows(ctx.run_dir / "history.csv")
    if reference:
        diffs: list[float] = []
        for row, ref in zip(trace, reference):
            try:
                ref_loss = float(ref.get("loss", "nan"))
            except (TypeError, ValueError):
                break
            if not np.isfinite(ref_loss):
                continue
            diffs.append(abs(row["loss"] - ref_loss) / max(abs(ref_loss), 1e-30))
        if diffs:
            check = {
                "history_csv": True,
                "compared": len(diffs),
                "max_rel_loss_diff": float(max(diffs)),
            }

    # Cross-check against the saved artefacts of the hardware run: the final
    # phase and the model far-field must come out the same.
    final_phase = np.asarray(opt.current_phase, dtype=np.float64)
    phase_path = ctx.run_dir / "phase_final.npy"
    if phase_path.is_file():
        saved = np.load(phase_path).astype(np.float64)
        if saved.shape == final_phase.shape:
            check["phase_final_max_abs_diff_rad"] = float(
                np.max(np.abs(saved - final_phase))
            )
    far_path = ctx.run_dir / "model_farfield.npy"
    if far_path.is_file():
        saved_far = np.load(far_path).astype(np.float64)
        repro_far = main_mod.model_far_field(amplitude, final_phase)
        if saved_far.shape == repro_far.shape:
            peak = float(np.max(saved_far))
            check["model_farfield_max_rel_diff"] = float(
                np.max(np.abs(saved_far - repro_far)) / max(peak, 1e-30)
            )

    return {
        "trace": trace,
        "frames": frames,
        "frame_iterations": frame_iters,
        "check": check,
        "device": device,
        "spec": spec,
        "window": window,
    }


# ---------------------------------------------------------------------------
# Hardware side (measurement anchors)
# ---------------------------------------------------------------------------


def collect_hardware_frames(
    ctx: RunContext,
    main_mod: ModuleType,
    spec: Any,
) -> list[dict[str, Any]]:
    """Resample the saved CCD frames onto the model far-field grid.

    The measurement-anchored loop captured one frame per step, but only the
    last three were written to disk (``ccd_hw_iter_{002,003,004}.npz``). The
    first step is represented by ``ccd_optimized.npz`` (the model-side final
    phase on the panel -- the very phase the loop starts from). Step 2 has no
    file at all; it is skipped and the report says so.
    """
    from ao_shaping.utils.image.beam_metrics import prepare_frame

    rows = read_csv_rows(ctx.run_dir / "history_hw.csv")
    if not rows:
        return []
    centre = ctx.config.get("geometry", {}).get("center", [0.0, 0.0])
    centre_cy, centre_cx = float(centre[0]), float(centre[1])
    window = int(min(FARFIELD_WINDOW, spec.m_grid))

    # Panel <-> camera axis convention, decided by the main run.
    best = ctx.config.get("comparison", {}).get("best", {}) or {}

    def _convention(arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float64)
        if best.get("transpose"):
            out = out.T
        if best.get("flip_rows"):
            out = out[::-1, :]
        if best.get("flip_cols"):
            out = out[:, ::-1]
        return out

    out: list[dict[str, Any]] = []
    for row in rows:
        iteration = int(row.get("iteration", 0))
        candidates = [
            ctx.run_dir / f"ccd_hw_iter_{iteration - 1:03d}.npz",
            ctx.run_dir / f"ccd_hw_iter_{iteration - 1:03d}.npy",
        ]
        if iteration == 1:
            candidates.append(ctx.run_dir / "ccd_optimized.npz")
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            logger.warning("闭环第 {} 步没有对应的 CCD 文件, 跳过", iteration)
            continue
        if source.suffix == ".npy":
            frame = np.load(source)
        else:
            with np.load(source) as data:
                frame = data["frame"] if "frame" in data.files else data[list(data.files)[0]]
        measured = main_mod.resample_ccd_to_model(
            prepare_frame(np.asarray(frame, dtype=np.float64)),
            (centre_cx, centre_cy),
            ctx.k_mean,
            spec,
        )
        out.append(
            {
                "kind": "hardware",
                "iteration": iteration,
                "loss": float(row.get("loss", "nan")),
                "grad_norm": float(row.get("grad_norm", "nan")),
                "phase_rms_rad": float(row.get("phase_rms_rad", "nan")),
                "measured_peak": float(row.get("measured_peak", "nan")),
                "exposure_ms": float(row.get("exposure_ms", "nan")),
                "source": source.name,
                "farfield": _convention(measured),
                "window": window,
            }
        )
    return out


def collect_degraded_frames(ctx: RunContext, main_mod: ModuleType, spec: Any) -> list[dict[str, Any]]:
    """Fallback frames: the saved measured frames + the saved model far-field.

    Used when the model side cannot be reproduced offline. The reason must be
    reported; see the module docstring.
    """
    frames = collect_hardware_frames(ctx, main_mod, spec)
    model_path = ctx.run_dir / "model_farfield.npy"
    if model_path.is_file():
        window = int(min(FARFIELD_WINDOW, spec.m_grid))
        frames.append(
            {
                "iteration": 0,
                "loss": float("nan"),
                "grad_norm": float("nan"),
                "phase_rms_rad": float("nan"),
                "measured_peak": float("nan"),
                "exposure_ms": float("nan"),
                "source": model_path.name,
                "farfield": crop_farfield(
                    np.load(model_path), spec.centre_index, window
                ),
                "window": window,
                "kind": "model_reference",
            }
        )
    return frames


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _peak_normalised(arr: np.ndarray) -> np.ndarray:
    """Peak-normalise a far-field so model and measured frames share a scale."""
    a = np.asarray(arr, dtype=np.float64)
    peak = float(a.max()) if a.size else 0.0
    if peak <= 0:
        return np.zeros_like(a)
    return a / peak


def _target_box(half_samples: int, window: int) -> tuple[float, float, float, float]:
    """Target square in window coordinates (the window is centred on DC)."""
    centre = window / 2.0
    lo = centre - half_samples
    hi = centre + half_samples + 1
    return lo, hi, lo, hi


def _target_half_samples(ctx: RunContext, main_mod: ModuleType, spec: Any) -> int:
    """Half side of the target square, in model samples."""
    target_info = ctx.config.get("model", {}).get("target", {}) or {}
    side = int(target_info.get("side_samples", 0) or 0)
    if side:
        return max(side // 2, 1)
    kind = str(ctx.params.get("target", "square"))
    side_px = float(ctx.params.get("target_side_px", 60.0))
    s_far = spec.far_field_sample_ccd_px(ctx.k_mean) if ctx.k_mean > 0 else 1.0
    if kind == "delta":
        delta_px = float(ctx.params.get("delta_spot_px", 15.0))
        return max(int(round(delta_px / s_far / 2.0)), 1)
    return max(int(round(side_px / s_far / 2.0)), 1)


def _frame_records(
    model_frames: list[dict[str, Any]],
    hw_frames: list[dict[str, Any]],
    *,
    iters_total: int,
    hw_total: int,
) -> list[dict[str, Any]]:
    """Merge model + hardware frames into the ordered frame list."""
    records: list[dict[str, Any]] = []
    for rec in model_frames:
        records.append(
            {
                "index": len(records),
                "kind": "model",
                "iteration": rec["iteration"],
                "iteration_label": f"{rec['iteration']}/{iters_total}",
                "loss": rec["loss"],
                "grad_norm": rec["grad_norm"],
                "phase_rms_rad": rec["phase_rms_rad"],
                "measured_peak": float("nan"),
                "exposure_ms": float("nan"),
                "source": "model-repro",
                "farfield": rec["farfield"],
            }
        )
    for rec in hw_frames:
        kind = "model_reference" if rec.get("kind") == "model_reference" else "hardware"
        records.append(
            {
                "index": len(records),
                "kind": kind,
                "iteration": rec["iteration"],
                "iteration_label": f"{rec['iteration']}/{hw_total}",
                "loss": rec["loss"],
                "grad_norm": rec["grad_norm"],
                "phase_rms_rad": rec["phase_rms_rad"],
                "measured_peak": rec["measured_peak"],
                "exposure_ms": rec["exposure_ms"],
                "source": rec["source"],
                "farfield": rec["farfield"],
            }
        )
    return records


def _overlay_text(rec: dict[str, Any], *, window: int, degraded: bool) -> str:
    """The per-frame text block (the "live numbers")."""
    if rec["kind"] == "model":
        head = f"模型侧 (离线复现)  第 {rec['iteration_label']} 次迭代"
        if degraded:
            head = f"模型侧 (存档)  第 {rec['iteration_label']} 次迭代"
        return "\n".join(
            [
                head,
                f"损失 = {rec['loss']:.6e}    |梯度| = {rec['grad_norm']:.4e}",
                f"相位 RMS = {rec['phase_rms_rad']:.4f} rad",
            ]
        )
    if rec["kind"] == "model_reference":
        return "\n".join(
            [
                "模型侧存档远场 (model_farfield.npy)",
                f"来源 = {rec['source']}",
                "模型侧演化未能离线复现 (见配套报告)",
            ]
        )
    return "\n".join(
        [
            f"测量锚定闭环 第 {rec['iteration_label']} 步  ({rec['source']})",
            f"实测损失 = {rec['loss']:.6e}    |梯度| = {rec['grad_norm']:.4f}",
            f"相位 RMS = {rec['phase_rms_rad']:.4f} rad",
            f"实测峰值 = {rec['measured_peak']:.1f} counts   曝光 = {rec['exposure_ms']:.1f} ms",
        ]
    )


def render_gif(
    records: list[dict[str, Any]],
    *,
    out_path: Path,
    trace: list[dict[str, Any]],
    hw_frames: list[dict[str, Any]],
    blind_loss: float | None,
    target_half: int,
    fps: float,
    dpi: int,
    degraded: bool,
    stamp: str,
) -> dict[str, Any]:
    """Render the process animation and write the GIF."""
    from PIL import Image

    if not records:
        raise ValueError("没有可渲染的帧")

    window = int(records[0]["farfield"].shape[0])
    fig = plt.figure(figsize=(12.0, 6.6), dpi=dpi)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.7, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.045,
        right=0.985,
        top=0.90,
        bottom=0.075,
        wspace=0.22,
        hspace=0.30,
    )
    ax_img = fig.add_subplot(grid[:, 0])
    ax_loss = fig.add_subplot(grid[0, 1])
    ax_grad = fig.add_subplot(grid[1, 1])

    # -- main panel: the far-field ------------------------------------------
    normalised = [_peak_normalised(rec["farfield"]) for rec in records]
    logs = [np.log10(np.maximum(arr, 1e-6)) for arr in normalised]
    vmax = float(max(float(l.max()) for l in logs))
    vmin = float(min(float(np.percentile(l, 2.0)) for l in logs))
    if vmax - vmin < 1e-6:
        vmin, vmax = vmax - 1.0, vmax
    image = ax_img.imshow(
        logs[0],
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="nearest",
    )
    lo, hi, _, _ = _target_box(target_half, window)
    ax_img.add_patch(
        plt.Rectangle(
            (lo - 0.5, lo - 0.5),
            hi - lo + 1.0,
            hi - lo + 1.0,
            fill=False,
            edgecolor="#ffffff",
            linestyle="--",
            linewidth=1.4,
        )
    )
    ax_img.set_title(
        f"远场强度 (每帧各自峰值归一化, log10)   窗口 = 零级 ±{window // 2} 采样\n"
        "白虚线 = 目标方块 (建在零级上)",
        fontsize=10,
    )
    ax_img.set_xlabel("远场 x / 采样")
    ax_img.set_ylabel("远场 y / 采样")
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    cbar = fig.colorbar(image, ax=ax_img, fraction=0.046, pad=0.02)
    cbar.set_label("log10(I/I_max)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # -- side panels: the curves -------------------------------------------
    model_x = [row["iteration"] for row in trace]
    model_loss = [row["loss"] for row in trace]
    model_grad = [row["grad_norm"] for row in trace]
    hw_x = [len(trace) + rec["iteration"] for rec in hw_frames]
    hw_loss = [rec["loss"] for rec in hw_frames]
    hw_grad = [rec["grad_norm"] for rec in hw_frames]

    ax_loss.plot(model_x, model_loss, "-", color=COLOR_MODEL, lw=1.6, label="模型侧损失")
    if blind_loss is not None and np.isfinite(blind_loss):
        ax_loss.axhline(
            blind_loss,
            color=COLOR_HW,
            ls="--",
            lw=1.1,
            label=f"不可达下界 {blind_loss:.2e}",
        )
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("迭代 (模型侧) / 拼接位置 (锚点)", fontsize=8)
    ax_loss.set_ylabel("损失 (峰值归一 MSE)", fontsize=8)
    ax_loss.tick_params(labelsize=7)
    ax_loss.grid(alpha=0.3)
    if hw_x:
        ax_loss.plot(
            hw_x,
            hw_loss,
            "D",
            color=COLOR_HW,
            ms=5,
            label="实测锚点",
        )
    loss_marker, = ax_loss.plot([], [], "o", color=COLOR_BLUE, ms=6)
    ax_loss.legend(fontsize=7, loc="upper right")

    ax_grad.plot(model_x, model_grad, "-", color=COLOR_MODEL, lw=1.6, label="模型侧 |梯度|")
    ax_grad.set_yscale("log")
    ax_grad.set_xlabel("迭代 (模型侧) / 拼接位置 (锚点)", fontsize=8)
    ax_grad.set_ylabel("梯度范数", fontsize=8)
    ax_grad.tick_params(labelsize=7)
    ax_grad.grid(alpha=0.3)
    if hw_x:
        ax_grad.plot(hw_x, hw_grad, "D", color=COLOR_HW, ms=5, label="实测锚点")
    grad_marker, = ax_grad.plot([], [], "o", color=COLOR_BLUE, ms=6)
    ax_grad.legend(fontsize=7, loc="upper left")

    title = fig.suptitle(
        f"可微相位优化过程 ({stamp})  —  模型侧离线复现 + 测量锚定闭环",
        fontsize=12,
        y=0.985,
    )
    # The live numbers sit **inside** the far-field panel (not in the figure
    # margin) -- in the margin they collided with the suptitle.
    overlay = ax_img.text(
        0.015,
        0.985,
        "",
        transform=ax_img.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        zorder=5,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": COLOR_GRID, "pad": 4},
    )
    footer = fig.text(
        0.012,
        0.012,
        "上: 模型侧远场逐次演化 (蓝点 = 当前帧)   下: 测量锚定闭环的实测帧 "
        "(同一远场网格, 可与模型帧直接对比)",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    del title, footer  # the artists live in the figure; drop the python refs

    # -- panel size check (readability in the report column) ---------------
    fig.canvas.draw()
    panel_px = {
        name: float(ax.get_window_extent().width)
        for name, ax in (("farfield", ax_img), ("loss", ax_loss), ("grad", ax_grad))
    }
    if min(panel_px.values()) < MIN_PANEL_PX:
        raise ValueError(
            f"面板宽度不足 {MIN_PANEL_PX} px: {panel_px} —— 增大 --dpi 或 figsize"
        )

    # -- render every frame -------------------------------------------------
    images: list[Image.Image] = []
    for rec, log_arr in zip(records, logs):
        image.set_data(log_arr)
        if rec["kind"] == "model":
            marker_x = rec["iteration"]
            marker_loss = rec["loss"]
            marker_grad = rec["grad_norm"]
        elif rec["kind"] == "model_reference":
            marker_x = len(trace)
            marker_loss = float("nan")
            marker_grad = float("nan")
        else:
            marker_x = len(trace) + rec["iteration"]
            marker_loss = rec["loss"]
            marker_grad = rec["grad_norm"]
        loss_marker.set_data([marker_x], [marker_loss])
        grad_marker.set_data([marker_x], [marker_grad])
        overlay.set_text(_overlay_text(rec, window=window, degraded=degraded))
        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        images.append(Image.fromarray(buffer))
    plt.close(fig)

    # -- assemble the GIF ----------------------------------------------------
    # Dithering is switched off on purpose: Floyd-Steinberg noise ruins the
    # frame-to-frame delta compression (8.6 MB -> 2.9 MB here) and the far-field
    # is a smooth image where the banding is far less objectionable than the
    # size.
    palette = images[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    quantised = [
        img.quantize(colors=256, palette=palette, dither=Image.Dither.NONE)
        for img in images
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(fps, 1e-6)))
    quantised[0].save(
        out_path,
        save_all=True,
        append_images=quantised[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return {
        "path": str(out_path),
        "frames": len(quantised),
        "duration_ms": duration_ms,
        "duration_s": len(quantised) * duration_ms / 1000.0,
        "fps": fps,
        "dpi": dpi,
        "size_px": [int(images[0].width), int(images[0].height)],
        "panel_px": panel_px,
        "bytes": out_path.stat().st_size,
    }


def render_curves(
    *,
    out_path: Path,
    trace: list[dict[str, Any]],
    hw_frames: list[dict[str, Any]],
    blind_loss: float | None,
    exposure_flat: float,
    exposure_optimized: float,
    dpi: int,
    stamp: str,
    hw_total: int = 0,
) -> dict[str, Any]:
    """Loss / gradient norm versus iteration + the 5 measurement anchors."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.0, 10.0),
        dpi=dpi,
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.15, 0.85], "hspace": 0.28},
    )
    model_x = [row["iteration"] for row in trace]
    model_loss = [row["loss"] for row in trace]
    model_grad = [row["grad_norm"] for row in trace]
    n_model = len(trace)
    hw_x = [n_model + rec["iteration"] for rec in hw_frames]
    hw_loss = [rec["loss"] for rec in hw_frames]
    hw_grad = [rec["grad_norm"] for rec in hw_frames]
    hw_peak = [rec["measured_peak"] for rec in hw_frames]
    hw_exposure = [rec["exposure_ms"] for rec in hw_frames]

    ax = axes[0]
    ax.plot(model_x, model_loss, "-", color=COLOR_MODEL, lw=1.7, label="模型侧损失")
    ax.set_yscale("log")
    ax.set_ylabel("模型侧损失 (峰值归一 MSE)", color=COLOR_MODEL, fontsize=9)
    ax.tick_params(axis="y", labelcolor=COLOR_MODEL, labelsize=8)
    ax.grid(alpha=0.3)
    if blind_loss is not None and np.isfinite(blind_loss):
        ax.axhline(
            blind_loss,
            color=COLOR_HW,
            ls="--",
            lw=1.1,
            label=f"不可达下界 (目标块面积比) {blind_loss:.2e}",
        )
    ax.axvline(n_model, color=COLOR_BLUE, ls=":", lw=1.2)
    twin = ax.twinx()
    if hw_x:
        twin.plot(
            hw_x,
            hw_loss,
            "D-",
            color=COLOR_HW,
            ms=6,
            lw=1.4,
            label="实测锚点损失",
        )
        for x, y in zip(hw_x, hw_loss):
            twin.annotate(
                f"锚点{int(x - n_model)}",
                (x, y),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=7,
                color=COLOR_HW,
            )
    twin.set_ylabel("实测锚点损失", color=COLOR_HW, fontsize=9)
    twin.tick_params(axis="y", labelcolor=COLOR_HW, labelsize=8)
    ax.set_title(
        f"可微相位优化: 模型侧 {n_model} 次迭代 + 测量锚定闭环 "
        f"{len(hw_frames)}/{hw_total or len(hw_frames)} 步有存档帧 ({stamp})",
        fontsize=11,
    )
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=8, loc="upper left")

    # Inset: the anchors are squeezed into the last few units of a 200-wide
    # axis -- the zoom makes their (monotone) trend readable.
    if hw_frames:
        axins = ax.inset_axes([0.45, 0.44, 0.35, 0.40])
        steps = [rec["iteration"] for rec in hw_frames]
        axins.plot(steps, hw_loss, "D-", color=COLOR_HW, ms=5, lw=1.2)
        if model_loss:
            axins.axhline(
                model_loss[-1],
                color=COLOR_MODEL,
                ls="--",
                lw=1.0,
                label=f"模型侧最终 {model_loss[-1]:.3e}",
            )
            axins.legend(fontsize=6, loc="best")
        axins.set_yscale("log")
        axins.set_title("锚点放大 (闭环步)", fontsize=7)
        axins.tick_params(labelsize=6)
        axins.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(model_x, model_grad, "-", color=COLOR_MODEL, lw=1.7, label="模型侧梯度范数")
    ax.set_yscale("log")
    ax.set_ylabel("模型侧梯度范数", color=COLOR_MODEL, fontsize=9)
    ax.tick_params(axis="y", labelcolor=COLOR_MODEL, labelsize=8)
    ax.grid(alpha=0.3)
    ax.axvline(n_model, color=COLOR_BLUE, ls=":", lw=1.2)
    twin = ax.twinx()
    if hw_x:
        twin.plot(hw_x, hw_grad, "D-", color=COLOR_HW, ms=6, lw=1.4, label="实测锚点梯度范数")
    twin.set_ylabel("实测锚点梯度范数", color=COLOR_HW, fontsize=9)
    twin.tick_params(axis="y", labelcolor=COLOR_HW, labelsize=8)
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=8, loc="upper left")

    ax = axes[2]
    if hw_x:
        ax.plot(hw_x, hw_peak, "s-", color=COLOR_HW, ms=6, lw=1.4, label="实测峰值 / counts")
        twin = ax.twinx()
        twin.plot(hw_x, hw_exposure, "^--", color=COLOR_BLUE, ms=5, lw=1.2, label="曝光 / ms")
        twin.set_ylabel("曝光 / ms", color=COLOR_BLUE, fontsize=9)
        twin.tick_params(axis="y", labelcolor=COLOR_BLUE, labelsize=8)
        handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
        labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
        ax.legend(handles, labels, fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "本轮没有测量锚定记录", ha="center", va="center")
        ax.set_axis_off()
    ax.set_ylabel("实测峰值 / counts", color=COLOR_HW, fontsize=9)
    ax.tick_params(axis="y", labelcolor=COLOR_HW, labelsize=8)
    ax.grid(alpha=0.3)
    ax.annotate(
        f"曝光自适应: 平场 {exposure_flat:.1f} ms → 优化后 {exposure_optimized:.1f} ms",
        xy=(0.02, 0.86),
        xycoords="axes fraction",
        fontsize=9,
        color="#333333",
        bbox={"facecolor": "#f2f2f2", "edgecolor": COLOR_GRID, "pad": 3},
    )
    anchor_steps = [rec["iteration"] for rec in hw_frames]
    if anchor_steps:
        mapping = ", ".join(
            f"{n_model + step} = 步 {step}" for step in anchor_steps
        )
        xlabel = (
            f"迭代 (模型侧, 1..{n_model})   |   拼接位置 {mapping}"
        )
    else:
        xlabel = f"迭代 (模型侧, 1..{n_model})"
    ax.set_xlabel(xlabel, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(out_path),
        "bytes": out_path.stat().st_size,
        "model_iterations": n_model,
        "anchors": len(hw_frames),
    }


# ---------------------------------------------------------------------------
# Report link checking (the report is hand-written; its links must resolve)
# ---------------------------------------------------------------------------


def iter_markdown_links(text: str) -> list[str]:
    """Relative link targets of a markdown document (images included)."""
    import re

    out: list[str] = []
    for match in re.finditer(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", text):
        target = match.group(1).strip()
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        out.append(target)
    return out


def check_report_links(report_path: Path) -> list[str]:
    """Return the list of **broken** relative links of a report."""
    if not report_path.is_file():
        return [f"报告不存在: {report_path}"]
    text = report_path.read_text(encoding="utf-8")
    broken: list[str] = []
    for target in iter_markdown_links(text):
        candidate = (report_path.parent / target.split("#", 1)[0]).resolve()
        if not candidate.exists():
            broken.append(target)
    return broken


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def render(
    *,
    run_dir: Path,
    report_dir: Path,
    frames: int = DEFAULT_FRAMES,
    fps: float = DEFAULT_FPS,
    dpi: int = DEFAULT_DPI,
    force_degrade: bool = False,
    include_hardware: bool = True,
) -> dict[str, Any]:
    """Produce the GIF + curve figure + the run-directory data drop."""
    setup_cjk_font()
    ctx = load_run(run_dir)
    main_mod = load_main_module()

    spec, _amplitude, _target, _target_info = build_model_inputs(main_mod, ctx)
    target_half = _target_half_samples(ctx, main_mod, spec)
    blind_loss = None
    if ctx.k_mean > 0:
        blind_loss = main_mod._blind_loss_value(  # reuse, do not re-implement
            spec.far_field_sample_ccd_px(ctx.k_mean),
            float(ctx.params.get("target_side_px", 60.0)),
            int(spec.m_grid),
        )

    degraded = False
    degrade_reason = ""
    trace: list[dict[str, Any]] = []
    model_frames: list[dict[str, Any]] = []
    check: dict[str, Any] = {"history_csv": False, "compared": 0, "max_rel_loss_diff": float("nan")}
    device = ""

    if force_degrade:
        degraded = True
        degrade_reason = "命令行强制 (--force-degrade)"
    else:
        try:
            frame_iters = select_frame_iterations(ctx.iters, frames)
            repro = reproduce_model_side(ctx, main_mod, frame_iterations=frame_iters)
            trace = repro["trace"]
            check = repro["check"]
            device = repro["device"]
            window = repro["window"]
            for iteration, far in zip(repro["frame_iterations"], repro["frames"]):
                row = trace[iteration - 1]
                model_frames.append(
                    {
                        "iteration": iteration,
                        "loss": row["loss"],
                        "grad_norm": row["grad_norm"],
                        "phase_rms_rad": row["phase_rms_rad"],
                        "farfield": far,
                    }
                )
            logger.info(
                "模型侧离线复现完成: {} 帧, device={}, 与 history.csv 的最大相对偏差={:.3e}",
                len(model_frames),
                device,
                check.get("max_rel_loss_diff", float("nan")),
            )
        except Exception as exc:  # the degraded path is the contract
            degraded = True
            degrade_reason = f"{type(exc).__name__}: {exc}"
            logger.warning("模型侧离线复现失败, 走降级路径: {}", degrade_reason)
            trace = []
            model_frames = []
            window = int(min(FARFIELD_WINDOW, spec.m_grid))

    hw_frames: list[dict[str, Any]] = []
    if include_hardware:
        try:
            hw_frames = collect_hardware_frames(ctx, main_mod, spec)
        except Exception as exc:
            logger.warning("实测帧读取失败, 跳过硬件锚点: {}: {}", type(exc).__name__, exc)
            hw_frames = []

    if degraded:
        # Degraded frames: the saved measured frames + the saved model far-field.
        degraded_frames = collect_degraded_frames(ctx, main_mod, spec)
        hw_frames = degraded_frames
        if trace:
            # Keep the recorded model trace for the curve panels.
            trace = trace
        else:
            trace = [
                {
                    "iteration": int(row.get("iteration", 0)),
                    "loss": float(row.get("loss", "nan")),
                    "grad_norm": float(row.get("grad_norm", "nan")),
                    "phase_rms_rad": float(row.get("phase_rms_rad", "nan")),
                }
                for row in read_csv_rows(run_dir / "history.csv")
            ]

    records = _frame_records(
        model_frames,
        hw_frames,
        iters_total=ctx.iters,
        hw_total=ctx.hw_iters,
    )
    if not records:
        raise ValueError(f"运行目录 {run_dir} 里没有任何可渲染的帧")

    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    gif_path = fig_dir / f"{ctx.stamp}_opt_process.gif"
    curves_path = fig_dir / f"{ctx.stamp}_opt_curves.png"

    gif_info = render_gif(
        records,
        out_path=gif_path,
        trace=trace,
        hw_frames=hw_frames,
        blind_loss=blind_loss,
        target_half=target_half,
        fps=fps,
        dpi=dpi,
        degraded=degraded,
        stamp=ctx.stamp,
    )

    comparison = ctx.config.get("comparison", {}).get("best", {}) or {}
    exposure_flat = float((comparison.get("exposure_ms") or {}).get("flat", 2.0))
    exposure_optimized = float((comparison.get("exposure_ms") or {}).get("optimized", 72.0))
    curves_info = render_curves(
        out_path=curves_path,
        trace=trace,
        hw_frames=hw_frames,
        blind_loss=blind_loss,
        exposure_flat=exposure_flat,
        exposure_optimized=exposure_optimized,
        dpi=max(dpi, 120),
        stamp=ctx.stamp,
        hw_total=ctx.hw_iters,
    )

    # ---- run-directory data drop -----------------------------------------
    anim_dir = run_dir / "animation"
    anim_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = [
        {
            "frame_index": rec["index"],
            "kind": rec["kind"],
            "iteration": rec["iteration"],
            "loss": rec["loss"],
            "grad_norm": rec["grad_norm"],
            "phase_rms_rad": rec["phase_rms_rad"],
            "measured_peak": rec["measured_peak"],
            "exposure_ms": rec["exposure_ms"],
            "source": rec["source"],
        }
        for rec in records
    ]
    write_csv(
        anim_dir / "frames.csv",
        [
            "frame_index",
            "kind",
            "iteration",
            "loss",
            "grad_norm",
            "phase_rms_rad",
            "measured_peak",
            "exposure_ms",
            "source",
        ],
        frame_rows,
    )
    write_csv(
        anim_dir / "model_history.csv",
        ["iteration", "loss", "grad_norm", "phase_rms_rad"],
        trace,
    )
    if model_frames:
        np.save(
            anim_dir / "model_farfield_frames.npy",
            np.stack([rec["farfield"] for rec in model_frames]).astype(np.float32),
        )
    measured = [rec["farfield"] for rec in hw_frames if rec["kind"] == "hardware"]
    if measured:
        np.save(
            anim_dir / "measured_farfield_frames.npy",
            np.stack(measured).astype(np.float32),
        )

    meta = {
        "stamp": ctx.stamp,
        "run_dir": str(run_dir),
        "script": str(Path(__file__).relative_to(ROOT).as_posix()),
        "degraded": degraded,
        "degrade_reason": degrade_reason,
        "model_side": {
            "reproduced_offline": not degraded,
            "device": device,
            "iterations": len(trace),
            "frames": len(model_frames),
            "verification_vs_history_csv": check,
        },
        "hardware_anchors": [
            {
                "iteration": rec["iteration"],
                "loss": rec["loss"],
                "grad_norm": rec["grad_norm"],
                "phase_rms_rad": rec["phase_rms_rad"],
                "measured_peak": rec["measured_peak"],
                "exposure_ms": rec["exposure_ms"],
                "source": rec["source"],
            }
            for rec in hw_frames
        ],
        "hardware_anchor_steps_total": ctx.hw_iters,
        "hardware_anchor_steps_with_frames": [
            rec["iteration"] for rec in hw_frames if rec["kind"] == "hardware"
        ],
        "exposure_ms": {"flat": exposure_flat, "optimized": exposure_optimized},
        "farfield_window_samples": int(records[0]["farfield"].shape[0]),
        "gif": gif_info,
        "curves": curves_info,
        "data_drop": {
            "frames_csv": str(anim_dir / "frames.csv"),
            "model_history_csv": str(anim_dir / "model_history.csv"),
            "model_farfield_frames": str(anim_dir / "model_farfield_frames.npy"),
            "measured_farfield_frames": str(anim_dir / "measured_farfield_frames.npy"),
        },
    }
    (anim_dir / "animation_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def build_parser() -> argparse.ArgumentParser:
    """CLI."""
    parser = argparse.ArgumentParser(
        description="渲染可微优化的过程动画 (GIF) 与曲线图 (纯离线)",
    )
    parser.add_argument("--run-dir", default=None, help="运行目录 (默认取最新的一轮)")
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="报告子目录 (figures/ 写在这里)",
    )
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="模型侧帧数")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="GIF 帧率")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="渲染 DPI")
    parser.add_argument(
        "--force-degrade",
        action="store_true",
        help="跳过模型侧离线复现 (走降级路径, 用于验证降级契约)",
    )
    parser.add_argument(
        "--no-hardware-frames",
        action="store_true",
        help="不追加测量锚定的实测帧",
    )
    return parser


def main() -> None:
    """Entry point."""
    args = build_parser().parse_args()
    run_dir = find_run_dir(args.run_dir)
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    meta = render(
        run_dir=run_dir,
        report_dir=report_dir,
        frames=args.frames,
        fps=args.fps,
        dpi=args.dpi,
        force_degrade=args.force_degrade,
        include_hardware=not args.no_hardware_frames,
    )
    gif = meta["gif"]
    print(f"GIF: {gif['path']}")
    print(
        f"  帧数 {gif['frames']} | 时长 {gif['duration_s']:.2f} s | "
        f"尺寸 {gif['size_px'][0]}x{gif['size_px'][1]} px | "
        f"{gif['bytes'] / 1e6:.2f} MB"
    )
    print(f"  面板宽度 (px): {gif['panel_px']}")
    print(f"曲线图: {meta['curves']['path']} ({meta['curves']['bytes'] / 1e6:.2f} MB)")
    print(f"模型侧离线复现: {meta['model_side']['reproduced_offline']}")
    if meta["degraded"]:
        print(f"降级原因: {meta['degrade_reason']}")


if __name__ == "__main__":
    main()
