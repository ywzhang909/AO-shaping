"""Render the no-GS far-field evolution as two GIFs (offline).

Two animations, deliberately kept apart because they answer different questions
and must never be read as one another:

``no_gs_model_iterations_20261024.gif``
    2x2 panels (one per shape), one frame per optimizer state (seed + every
    update), drawn from ``FrozenDefocusForward.predict``. This is the **process**
    view. Its absolute level is NOT the measurement -- the pure model reads
    CV ~0.27 where the CCD reads ~0.68 -- so every panel is labelled 模型预测.

``no_gs_measured_abba_20261024.gif``
    2x2 panels, the four **archived CCD frames** A1 / B1 / B2 / A2 of the same
    block, i.e. what the camera actually saw around the accepted phase. The
    reproduced candidate is bit-for-bit equal to the archived ``B1`` phase, so
    these frames are the measurement of it (they are re-used, not re-taken).

Both animations share the exact production update loop: the frames come from
``reproduce_transfer_block_without_gs.reproduce_shape(..., capture_frames=True)``,
which calls ``gs_seed()`` zero times. Nothing here opens the SLM or the camera.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reproduce_transfer_block_without_gs import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SEED_RUN,
    DEFAULT_TARGET_RUN,
    SHAPES,
    reproduce_shape,
)

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

LABELS = {"square": "正方形", "rectangle": "长方形", "triangle": "三角形", "star": "五角星"}
STAGES = ("A1", "B1", "B2", "A2")
STAGE_TITLES = ("A1（基准 1）", "B1（候选 1）", "B2（候选 2）", "A2（基准 2）")


def _limits(frames: list[np.ndarray]) -> tuple[float, float]:
    """One fixed display scale per panel so brightness change is visible, not rescaled."""
    stack = np.stack([np.asarray(frame, dtype=np.float64) for frame in frames])
    top = float(np.percentile(stack, 99.5))
    return 0.0, top if top > 0 else 1.0


def _readout(frame: np.ndarray, objective, baseline_cv: float | None = None) -> str:
    from ao_shaping.algorithm.signal_processing.multishape_objective import image_metrics

    metrics = image_metrics(frame, objective)
    cv = metrics["cv"]
    text = f"CV {'—' if cv is None else f'{cv:.3f}'}\nEE {metrics['ee_roi']:.3f}"
    if baseline_cv is not None and cv is not None:
        text += f"\nΔCV {cv - baseline_cv:+.3f}"
    return text


def _render(frames: dict[str, list[np.ndarray]], targets: dict[str, object],
            stage_titles: tuple[str, ...], out_path: Path, *, fps: int, suptitle: str,
            footer: str | None = None, baseline_index: int | None = None) -> Path:
    from ao_shaping.algorithm.signal_processing.multishape_objective import image_metrics

    shapes = [shape for shape in SHAPES if shape in frames]
    count = len(frames[shapes[0]])
    for shape in shapes:
        if len(frames[shape]) != count:
            raise ValueError(f"{shape}: {len(frames[shape])} frames, expected {count}")
    baselines = ({shape: image_metrics(frames[shape][baseline_index], targets[shape])["cv"]
                  for shape in shapes} if baseline_index is not None else
                 dict.fromkeys(shapes))

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.6), constrained_layout=True)
    images, notes = {}, {}
    for ax, shape in zip(axes.ravel(), shapes):
        vmin, vmax = _limits(frames[shape])
        images[shape] = ax.imshow(frames[shape][0], cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(LABELS.get(shape, shape), fontsize=11)
        notes[shape] = ax.text(0.03, 0.04, "", transform=ax.transAxes, fontsize=8.5,
                               color="white", va="bottom", ha="left",
                               bbox=dict(facecolor="black", alpha=0.55, pad=2.5, lw=0))
    if footer:
        fig.supxlabel(footer, fontsize=8.5, color="0.3")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(out_path), dpi=120):
        for index in range(count):
            for shape in shapes:
                frame = np.asarray(frames[shape][index], dtype=np.float64)
                images[shape].set_data(frame)
                notes[shape].set_text(_readout(frame, targets[shape], baselines[shape]))
            fig.suptitle(f"{suptitle}\n{stage_titles[index]}", fontsize=11)
            writer.grab_frame()
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / DEFAULT_CHECKPOINT)
    parser.add_argument("--seed-run", type=Path, default=ROOT / DEFAULT_SEED_RUN)
    parser.add_argument("--target-run", type=Path, default=ROOT / DEFAULT_TARGET_RUN)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--seed-source", choices=("run", "a1"), default="run",
                        help="same meaning as in reproduce_transfer_block_without_gs.py")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--step-rms-cap", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "report/shape_trials/sim_pretrain_transfer/images",
                        help="animations live in images/, not figures/ (report/AGENTS.md §4)")
    parser.add_argument("--stamp", default="20261024")
    args = parser.parse_args()

    from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective

    model: dict[str, list[np.ndarray]] = {}
    measured: dict[str, list[np.ndarray]] = {}
    targets: dict[str, object] = {}
    for shape in SHAPES:
        result = reproduce_shape(shape=shape, block=args.block, updates=args.updates,
                                 checkpoint=args.checkpoint, seed_run=args.seed_run,
                                 target_run=args.target_run, lr=args.lr,
                                 step_rms_cap=args.step_rms_cap,
                                 seed_from_run=args.seed_source == "run",
                                 capture_frames=True)
        repro = result["phase_reproduction"]
        print(f"{shape:10s} frames={len(result['model_frames'])} "
              f"B1 phase maxΔ={repro['max_abs']:.3e} bitwise={repro['bitwise_equal']}")
        if not repro["bitwise_equal"]:
            raise SystemExit(f"{shape}: reproduced phase differs from the archive; "
                             "refusing to render frames for a phase that was not reproduced")
        model[shape] = result["model_frames"]
        measured[shape] = [result["measured_frames"][stage] for stage in STAGES]
        targets[shape] = ShapeObjective(result["target"])

    process_titles = tuple(f"update {index}" for index in range(args.updates + 1))
    process_titles = ("update 0（种子相位）",) + process_titles[1:]
    model_path = _render(
        model, targets, process_titles,
        args.out_dir / f"no_gs_model_iterations_{args.stamp}.gif",
        fps=2, suptitle=f"无 GS 复现：模型预测的远场演化（block {args.block}）",
        footer="模型预测，非实测：纯模型 CV 随更新上升、实测 CV 下降，方向相反属已知现象（报告 §3.4）；ΔCV 相对 update 0",
        baseline_index=0)
    measured_path = _render(
        measured, targets, STAGE_TITLES,
        args.out_dir / f"no_gs_measured_abba_{args.stamp}.gif",
        fps=1, suptitle=f"无 GS 复现：存档实测远场 A1→B1→B2→A2（block {args.block}）",
        footer="存档实测帧（复现相位与存档 B1 逐位相同 → 这些帧即该相位的实测）；ΔCV 相对 A1",
        baseline_index=0)
    for path in (model_path, measured_path):
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
