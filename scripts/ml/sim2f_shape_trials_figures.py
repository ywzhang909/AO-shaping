"""Render the figures of a ``sim2f_shape_trials.py`` run.

Reads the artifacts the trial driver wrote (``trials.json``, ``trials.csv``,
``calibration.*`` and the per-arm phase dumps) and draws, per condition:

* ``fig_shapes_<condition>.png`` -- one row per shape: target bitmap, then the
  far field the **device** surrogate produces under the ``flat``,
  ``net_open`` and ``net_finetune`` commands (log scale, shared normalisation
  per row so the energy redistribution is visible, not the exposure).
* ``fig_rounds.png`` -- measured objective per accepted round for every
  finetune arm, small multiples over (condition, shape).
* ``fig_ee.png`` -- encircled energy before/after, per arm, per shape.

The device benches are rebuilt from the saved calibration with the same seed
as the trial run, so the frames rendered here are the frames the loop saw.

Usage::

    python scripts/ml/sim2f_shape_trials_figures.py \
        --run report/shape_trials/mask_closed_loop
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

REPO = Path(__file__).resolve().parents[2]
for extra in (REPO / "src", REPO / "scripts" / "ml"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from ml.sim2f.calibrate import load_calibration
from ml.sim2f.losses import prepare_frame
from ml.sim2f.optics import Sim2f
from sim2f_shape_trials import CONDITIONS, build_benches, build_masks

#: Arms drawn in the shape figure, in column order.
ARMS = ("flat", "net_open", "net_finetune")


def _device_frame(bench: object, phase: np.ndarray) -> np.ndarray:
    """The prepared device far field for one command.

    Args:
        bench: The device surrogate.
        phase: ``(grid, grid)`` radians.

    Returns:
        ``(ccd_px, ccd_px)`` prepared intensity (median removed, clipped).
    """
    command = torch.as_tensor(phase, dtype=bench.dtype, device=bench.device)
    with torch.no_grad():
        frame = bench.forward(command[None], detector=False)
    return prepare_frame(frame)[0, 0].detach().cpu().numpy()


def _log_norm(frame: np.ndarray, peak: float) -> np.ndarray:
    """Log-compress a frame against a shared peak, for one row of panels.

    Args:
        frame: ``(ccd_px, ccd_px)`` non-negative intensity.
        peak: Shared normalisation (the row's maximum).

    Returns:
        ``log10(1 + frame / peak * 1e3)``, in ``[0, ~3]``.
    """
    return np.log10(1.0 + frame / max(peak, 1e-12) * 1e3)


def draw_shapes(
    condition: str,
    shapes: list[str],
    trials: dict[str, object],
    phases: Path,
    device: object,
    ccd_px: int,
    out: Path,
) -> Path:
    """One panel grid of target / flat / net_open / net_finetune far fields.

    Args:
        condition: Condition name (``"nominal"`` or ``"calibrated"``).
        shapes: Shape names, one per row.
        trials: The ``trials`` block of ``trials.json`` (for panel metrics).
        phases: Directory of the per-arm phase dumps.
        device: The condition's device surrogate.
        ccd_px: Window side, pixels.
        out: Output PNG path.

    Returns:
        The path written.
    """
    masks = build_masks(ccd_px)
    fig, axes = plt.subplots(
        len(shapes),
        1 + len(ARMS),
        figsize=(2.2 * (1 + len(ARMS)), 2.2 * len(shapes)),
        constrained_layout=True,
    )
    if len(shapes) == 1:
        axes = axes[None, :]
    for row, shape in enumerate(shapes):
        frames = [masks[shape].astype(np.float32)]
        for arm in ARMS:
            path = phases / f"{condition}_{shape}_{arm}.npy"
            frames.append(
                _device_frame(device, np.load(path))
                if path.is_file()
                else np.zeros((ccd_px, ccd_px), np.float32)
            )
        peak = max(float(frame.max()) for frame in frames[1:])
        detail = trials[f"{condition}/{shape}"]["arms"]  # type: ignore[index]
        titles = [f"target\n{shape}"]
        for arm, frame in zip(ARMS, frames[1:], strict=True):
            record = next(item for item in detail if item["arm"] == arm)  # type: ignore[union-attr]
            titles.append(
                f"{arm}\nee {record['start']['ee']:.2f} -> {record['final']['ee']:.2f}"
            )
        for col, (frame, title) in enumerate(zip(frames, titles, strict=True)):
            ax = axes[row, col]
            image = frame if col == 0 else _log_norm(frame, peak)
            ax.imshow(image, cmap="inferno", origin="lower")
            ax.set_title(title, fontsize=8)
            ax.set_xticks(())
            ax.set_yticks(())
    fig.suptitle(f"device far fields -- {condition}", fontsize=10)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def draw_rounds(trials: dict[str, object], out: Path) -> Path:
    """Measured objective per round for every finetune arm.

    Args:
        trials: The ``trials`` block of ``trials.json``.
        out: Output PNG path.

    Returns:
        The path written.
    """
    keys = sorted(trials)
    fig, axes = plt.subplots(
        2,
        (len(keys) + 1) // 2,
        figsize=(3.0 * ((len(keys) + 1) // 2), 5.4),
        constrained_layout=True,
        sharex=True,
    )
    for ax, key in zip(axes.flat, keys, strict=False):
        for arm in trials[key]["arms"]:  # type: ignore[index]
            rounds = arm["rounds"]
            if not rounds:
                continue
            xs = [0] + [item["round"] for item in rounds]
            ys = [arm["start"]["objective"]] + [item["objective"] for item in rounds]
            accepted = [item["accepted_device"] for item in rounds]
            ax.plot(xs, ys, marker="o", label=arm["arm"])
            for x, y, ok in zip(xs[1:], ys[1:], accepted, strict=True):
                if not ok:
                    ax.plot(x, y, marker="x", color="red", markersize=9)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("round")
        ax.set_ylabel("device objective")
        ax.legend(fontsize=7)
    fig.suptitle("iterative finetune, measured on the device (x = rejected round)")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def draw_ee(rows: list[dict[str, str]], out: Path) -> Path:
    """Encircled energy start -> final, per condition, shape and arm.

    Args:
        rows: The ``trials.csv`` rows.
        out: Output PNG path.

    Returns:
        The path written.
    """
    conditions = sorted({row["condition"] for row in rows})
    shapes = sorted({row["shape"] for row in rows})
    arms = [arm for arm in ARMS if any(row["arm"] == arm for row in rows)]
    width = 0.8 / (2 * len(arms))
    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=(5.4 * len(conditions), 3.6),
        constrained_layout=True,
        sharey=True,
    )
    if len(conditions) == 1:
        axes = [axes]
    for ax, condition in zip(axes, conditions, strict=True):
        xs = np.arange(len(shapes))
        for index, arm in enumerate(arms):
            start, final = [], []
            for shape in shapes:
                row = next(
                    item
                    for item in rows
                    if item["condition"] == condition
                    and item["shape"] == shape
                    and item["arm"] == arm
                )
                start.append(float(row["ee_start"]))
                final.append(float(row["ee_final"]))
            offset = (2 * index + 1 - len(arms)) * width
            ax.bar(
                xs + offset - width / 2,
                start,
                width,
                color="0.75",
                label=f"{arm} flat" if arm == "flat" else None,
            )
            ax.bar(xs + offset + width / 2, final, width, label=f"{arm} final")
        ax.set_xticks(xs, shapes)
        ax.set_title(condition)
        ax.set_ylabel("encircled energy in support")
        ax.legend(fontsize=7)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def draw_evolution(
    condition: str,
    shapes: list[str],
    trials: dict[str, object],
    phases: Path,
    device: object,
    ccd_px: int,
    out: Path,
) -> Path:
    """The finetune arm as a film strip: start frame, then every round.

    Columns are the round-0 command, then each round's measured phase, so the
    strip is the optimisation itself -- not just its endpoints.  Each row is
    log-scaled against its own peak, so what is compared across columns is
    where the energy went, not how bright the exposure was.  The target
    support is outlined in white so the eye can judge the gain directly.

    Args:
        condition: Condition name (``"nominal"`` or ``"calibrated"``).
        shapes: Shape names, one per row.
        trials: The ``trials`` block of ``trials.json``.
        phases: Directory of the per-arm phase dumps.
        device: The condition's device surrogate.
        ccd_px: Window side, pixels.
        out: Output PNG path.

    Returns:
        The path written.
    """
    masks = build_masks(ccd_px)
    arm = (
        "net_finetune"
        if any(
            item["arm"] == "net_finetune"
            for item in trials[f"{condition}/{shapes[0]}"]["arms"]
        )
        else "opt_finetune"
    )
    rounds = max(
        len(
            trials[f"{condition}/{shape}"]["arms"][
                _arm_index(trials, condition, shape, arm)
            ]["rounds"]
        )
        for shape in shapes
    )
    fig, axes = plt.subplots(
        len(shapes),
        1 + rounds,
        figsize=(2.1 * (1 + rounds), 2.1 * len(shapes)),
        constrained_layout=True,
    )
    if len(shapes) == 1:
        axes = axes[None, :]
    for row, shape in enumerate(shapes):
        record = trials[f"{condition}/{shape}"]["arms"][
            _arm_index(trials, condition, shape, arm)
        ]
        start_phase = phases / f"{condition}_{shape}_net_open.npy"
        if not start_phase.is_file():
            start_phase = phases / f"{condition}_{shape}_flat.npy"
        frames = [_device_frame(device, np.load(start_phase))]
        titles = [f"{shape}\nstart ee {record['start']['ee']:.2f}"]
        for item in record["rounds"]:
            path = phases / f"{condition}_{shape}_{arm}_r{item['round']}.npy"
            if not path.is_file():
                continue
            frames.append(_device_frame(device, np.load(path)))
            mark = "" if item["accepted_device"] else " (rejected)"
            titles.append(f"round {item['round']}{mark}\nee {item['ee']:.2f}")
        peak = max(float(frame.max()) for frame in frames)
        support = masks[shape].astype(np.float32)
        for col, frame in enumerate(frames):
            ax = axes[row, col]
            ax.imshow(_log_norm(frame, peak), cmap="inferno", origin="lower")
            ax.contour(support, levels=[0.5], colors="w", linewidths=0.6)
            ax.set_title(titles[col], fontsize=7)
            ax.set_xticks(())
            ax.set_yticks(())
    fig.suptitle(
        f"optimisation film strip -- {condition} (white contour = target support)",
        fontsize=10,
    )
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _arm_index(trials: dict[str, object], condition: str, shape: str, arm: str) -> int:
    """Index of ``arm`` in one trial's arm list.

    Args:
        trials: The ``trials`` block of ``trials.json``.
        condition: Condition name.
        shape: Shape name.
        arm: Arm name.

    Returns:
        The list index.
    """
    for index, item in enumerate(trials[f"{condition}/{shape}"]["arms"]):
        if item["arm"] == arm:
            return index
    raise KeyError(f"no arm {arm!r} in {condition}/{shape}")


def main(argv: list[str] | None = None) -> int:
    """Rebuild the benches, draw every figure.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        ``0`` on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run", default="report/shape_trials/mask_closed_loop", help="trial directory"
    )
    parser.add_argument("--seed", type=int, default=11, help="seed of the trial run")
    args = parser.parse_args(argv)

    run = Path(args.run)
    artifacts = run / "artifacts"
    summary = json.loads((artifacts / "trials.json").read_text(encoding="utf-8"))
    calibration = load_calibration(artifacts / "calibration")
    sim_config = calibration.sim_config
    ccd_px = int(sim_config.ccd_px)
    shapes = sorted({key.split("/")[1] for key in summary["trials"]})
    figures = run / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for condition in CONDITIONS:
        device, _model = build_benches(
            calibration, sim_config, condition, seed=args.seed
        )
        written.append(
            draw_shapes(
                condition.name,
                shapes,
                summary["trials"],
                artifacts / "phases",
                device,
                ccd_px,
                figures / f"fig_shapes_{condition.name}.png",
            )
        )
        written.append(
            draw_evolution(
                condition.name,
                shapes,
                summary["trials"],
                artifacts / "phases",
                device,
                ccd_px,
                figures / f"fig_evolution_{condition.name}.png",
            )
        )
    written.append(draw_rounds(summary["trials"], figures / "fig_rounds.png"))

    import csv

    with (artifacts / "trials.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    written.append(draw_ee(rows, figures / "fig_ee.png"))

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
