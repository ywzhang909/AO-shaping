"""Sweep the pupil-centre offset offline and ask whether G0-4's miss is placement.

What this answers, and what it refuses to answer
===============================================
``runs/sim2f_bench/session/`` measured 24 samples on the real 2f bench and came
out ``min ncc(measured net, twin net) = 0.4898`` against a pre-registered floor of
``0.50`` (Gate 0-4).  The session was run with ``--pupil-center-xy`` **defaulted**,
i.e. it pasted the 900 px phase square onto the panel centre ``(960, 600)`` and
assumed the illuminated pupil is there.  ``runs/sim2f_bench/README.md`` §6.1 step 2
asks the operator to confirm the illuminated pupil position; that step was skipped.

This script decides, from the **already measured** frames only, whether a different
``--pupil-center-xy`` would have cleared 0.50.  It never touches hardware and never
re-measures: the measured frames are a fixed dataset, and what is swept is the
twin's *interpretation* of where the phase map sat relative to the beam.

How a pupil-centre error can be recovered from a fixed measurement
==================================================================
Panel pixels ``x = (column, row)``; the hardware pastes the ``(128, 128)`` map into
a 900 px square centred at ``C = (960, 600)``, sampled at
``pitch = 900/128 = 7.03125`` panel px per model sample; the beam's own amplitude
is centred at ``B`` (unknown).  The entrance field is

    E(x) = A(x - B) * exp(i * net((x - C) / pitch))

Substituting ``u = x - B`` (beam-centred coordinates, the frame the twin works in)::

    E = A(u) * exp(i * net((u - (C - B)) / pitch))

so the beam sees the map **shifted by ``d = C - B``** relative to itself.  The twin
(``HwInstrument.twin``) forwards ``net`` **unshifted**, which is the ``d = 0``
assumption, i.e. ``B = C``.  Therefore the measured net frame is modelled as::

    measured  =  twin( shift(net, s*) ),      s* = d / pitch = (C - B) / pitch

and ``s*`` is recoverable by maximising the same ``ncc`` G0-4 uses.  The candidate
``--pupil-center-xy`` offset follows:  ``Delta = B - C = -s* * pitch``.

Sign convention check (the one thing that would silently invert the answer):
``scipy.ndimage.shift(f, s)`` gives ``out[i] = f[i - s]``, so ``s > 0`` moves the map
content towards higher indices.  With the beam 10 px to the *right* of the map
(``B - C = +10``) the map appears displaced by ``d = -10``, so ``s* < 0`` and
``Delta = -s* * pitch = +10`` px: the square should move right, onto the beam.  ✅

The twin side is copied exactly, not re-derived: one ``Sim2f(Sim2fConfig.calibrated())``
and ``bench.forward(tensor, detector=False)`` -- the body of ``HwInstrument.twin`` --
and the metric is ``ml.sim2f.bench._ncc``, the body of G0-4.  At ``Delta = (0, 0)``
the numbers here must reproduce ``session.json``'s per-sample ``g0_4_geometry.ncc``;
that equality is asserted, because a net phase rebuilt from the checkpoint that does
not reproduce the session would make every swept number meaningless.

Shifting is done on the **phasor** (``cos``/``sin`` shifted separately, then
``atan2``), never on the wrapped phase: a bilinear blend across an ``atan2`` branch
cut would fabricate structure.  Outside the pasted square the panel is flat
(``0`` rad), so the fill value is ``(cos, sin) = (1, 0)`` -- the same "the beam sees
flat where the square is not" the physical bench has.

Pre-registration
================
The criterion is written in
``report/sim2f_bench/twin_vs_hw_corpus/20261027/pupil_center_sweep.md`` §0 **before**
this script was first run, and it is not a parameter here: the coarse grid, the
``+-20 px`` band and the three-valued conclusion are the report's, and this file only
produces the numbers it names.  Nothing in ``src/ml/sim2f/**``,
``runs/sim2f_bench/README.md`` or ``runs/sim2f_bench/run_bench.py`` is modified --
in particular the ``0.5`` floor is read here, never written.

Usage::

    $env:PYTHONPATH="src;libs"
    .venv\\Scripts\\python.exe scripts\\ml\\calibrate_sim2f_pupil_center.py
"""

from __future__ import annotations

import argparse
from loguru import logger
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

#: ``pupil_side(450) / grid`` -- panel pixels per model sample (README §1.1).  Both
#: the 900/128 pixel-centre convention and the 899/127 endpoint-aligned embed pitch
#: are within 0.7 % of each other; a *translation* (which is what a pupil-centre
#: error is) is read at this single pitch, and 0.7 % of 120 px = 0.85 px, i.e. far
#: inside the 20 px band the criterion uses.
PITCH_PANEL_PX = 900.0 / 128.0

#: Pre-registered coarse grid, report §0.  Panel pixels, offsets from the panel centre.
COARSE_OFFSETS_PX = (-120, -100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100, 120)

#: Pre-registered extension when the optimum lands on the coarse grid's border.
WIDE_OFFSETS_PX = tuple(range(-240, 241, 20))

#: The band of the three-valued conclusion (report §0).  Not a knob.
PLACEMENT_BAND_PX = 20.0

#: The eight dihedral transforms of a square frame (the ``NCC-D4`` side evidence).
DIHEDRAL = {
    "identity": lambda a: a,
    "rot90": lambda a: np.rot90(a, 1),
    "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3),
    "flipud": lambda a: np.flipud(a),
    "fliplr": lambda a: np.fliplr(a),
    "transpose": lambda a: a.T,
    "antitranspose": lambda a: np.rot90(a.T, 2),
}


def _sha(args: argparse.Namespace) -> dict[str, Any]:
    """The run configuration, echoed into the artefacts (provenance, not gates)."""
    return {
        "session_dir": str(args.session_dir),
        "checkpoint": str(args.checkpoint),
        "corpus": str(args.corpus),
        "device": str(args.device),
        "coarse_offsets_px": list(COARSE_OFFSETS_PX),
        "wide_offsets_px": list(WIDE_OFFSETS_PX),
        "placement_band_px": PLACEMENT_BAND_PX,
        "pitch_panel_px": PITCH_PANEL_PX,
    }


def load_session(session_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Read ``session.json`` and ``frames.npz`` into ``{index: {name: frame}}``.

    Args:
        session_dir: Directory holding a completed session's artefacts.

    Returns:
        ``(payload, blocks)`` where ``blocks[index]`` holds the four measured frames.

    Raises:
        FileNotFoundError: If either artefact is missing.
    """
    payload = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    data = np.load(session_dir / "frames.npz")
    blocks: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        if not key.startswith("s"):
            continue
        stem, _, condition = key.rpartition("_")
        index = stem[1:]
        if condition not in ("a1", "b1", "b2", "a2"):
            continue
        blocks.setdefault(index, {})[condition] = np.asarray(data[key], dtype=np.float32)
    return payload, blocks


def rebuild_net_phases(
    payload: dict[str, Any], checkpoint: Path, corpus_dir: Path, device: str
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Recompute each sample's network phase exactly as :func:`bench.run_session` did.

    The session does not archive the commanded phase, so it is rebuilt from the same
    frozen checkpoint and the same corpus targets, through the same three calls
    (``_load_network`` -> ``network_input`` -> ``from_phasor``) and the same
    determinism preamble.  The rebuild is *verified*, not trusted: see
    :func:`check_reproduction`.

    Args:
        payload: ``session.json`` contents (supplies ``indices`` and ``seed``).
        checkpoint: ``model_best.pt`` used by the session.
        corpus_dir: Corpus directory (``pairs.npz``).
        device: Torch device.

    Returns:
        ``(phases, meta)`` -- ``phases[index] = (128, 128) float32`` radians, and the
        rebuild metadata for the report.
    """
    import torch

    from ml.sim2f._device import _resolve_device
    from ml.sim2f.bench import _check_geometry
    from ml.sim2f.config import Sim2fConfig
    from ml.sim2f.loop import _load_network
    from ml.sim2f.losses import from_phasor
    from ml.sim2f.model import network_input
    from ml.sim2f.train import _apply_determinism, load_corpus, seed_everything

    resolved = _resolve_device(device)
    seed_everything(int(payload["seed"]))
    _apply_determinism(True)
    corpus = load_corpus(corpus_dir)
    model, context, label_cfg, _raw = _load_network(checkpoint, resolved)
    bench_cfg = Sim2fConfig.calibrated()
    _check_geometry(label_cfg, bench_cfg)

    phases: dict[int, np.ndarray] = {}
    for index in payload["indices"]:
        target = corpus.target[index : index + 1].to(device=resolved, dtype=torch.float32)
        with torch.no_grad():
            phase = from_phasor(model(network_input(target, context)))
        phases[int(index)] = phase[0].detach().cpu().numpy().astype(np.float32)
    meta = {
        "device": str(resolved),
        "seed": int(payload["seed"]),
        "n_phases": len(phases),
        "phase_range_rad": [
            float(min(float(p.min()) for p in phases.values())),
            float(max(float(p.max()) for p in phases.values())),
        ],
    }
    return phases, meta


def _shift_phase(phase: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Translate a phase map so that ``out(u) = phase(u - (sx, sy))``.

    Done on the phasor: interpolating a wrapped ``atan2`` phase directly would
    create structure across its branch cut.  ``mode="constant"`` with
    ``(cos, sin) = (1, 0)`` means "flat (0 rad) panel where the pasted square is not",
    which is what the bench physically has outside the 900 px region.

    Args:
        phase: ``(grid, grid)`` radians, raw or wrapped (only its phasor is used).
        sx: Shift along axis 1 (panel x / column), model samples.
        sy: Shift along axis 0 (panel y / row), model samples.

    Returns:
        ``(grid, grid)`` float32 radians.
    """
    from scipy import ndimage

    cosine = ndimage.shift(
        np.cos(phase), (sy, sx), order=1, mode="constant", cval=1.0
    )
    sine = ndimage.shift(np.sin(phase), (sy, sx), order=1, mode="constant", cval=0.0)
    return np.arctan2(sine, cosine).astype(np.float32)


def twin_frames(bench: Any, phases: list[np.ndarray]) -> np.ndarray:
    """Batched twin prediction -- the body of ``HwInstrument.twin``, vectorised.

    Args:
        bench: ``Sim2f`` on ``Sim2fConfig.calibrated()``.
        phases: ``(grid, grid)`` radians per sample.

    Returns:
        ``(B, ccd_px, ccd_px)`` float arrays.
    """
    import torch

    tensor = torch.as_tensor(
        np.stack(phases), dtype=bench.dtype, device=bench.device
    )
    with torch.no_grad():
        frame = bench.forward(tensor, detector=False)
    return frame[:, 0].detach().cpu().numpy()


def measured_nets(blocks: dict[str, dict[str, np.ndarray]], indices: list[int]) -> np.ndarray:
    """``mean(B1, B2)`` per sample -- the exact left operand of G0-4."""
    return np.stack(
        [
            0.5 * (blocks[str(i)]["b1"].astype(np.float64) + blocks[str(i)]["b2"].astype(np.float64))
            for i in indices
        ]
    )


def check_reproduction(
    measured: np.ndarray,
    predicted: np.ndarray,
    indices: list[int],
    payload: dict[str, Any],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Assert the rebuild reproduces the session's own G0-4 readings.

    The whole sweep rests on "the measured net frame is ``twin(shift(net, s*))``".  That
    model is only falsifiable if ``s = 0`` reproduces the *recorded* numbers, so this is
    an equality check against ``session.json`` rather than a sanity glance.

    It also catches a real trap: the U-Net forward is **not device-portable at this
    precision**.  Rebuilding the same checkpoint's phases on CPU when the session ran on
    CUDA moves ``ncc`` by up to **1.0e-3** (measured here), i.e. 10 % of the G0-4 margin
    -- so a sweep on the wrong device would silently answer a different question.

    Args:
        measured: ``(B, ccd, ccd)`` measured net means.
        predicted: ``(B, ccd, ccd)`` twin predictions at zero shift.
        indices: Sample indices, in the same order.
        payload: ``session.json`` contents.
        tolerance: Largest allowed ``|ncc - recorded|``.

    Returns:
        A dict with the worst absolute deviation and the recorded/recomputed minima.

    Raises:
        AssertionError: If any sample's ``ncc`` differs from the recorded value by more
            than ``tolerance`` -- a rebuilt phase that does not reproduce the session
            cannot support any swept conclusion.
    """
    from ml.sim2f.bench import _ncc

    recorded = payload["gate0"]["g0_4_geometry"]["ncc"]
    values = {}
    deviation = 0.0
    for position, index in enumerate(indices):
        value = _ncc(measured[position], predicted[position])
        values[int(index)] = value
        deviation = max(deviation, abs(value - float(recorded[str(index)])))
    if deviation > float(tolerance):
        raise AssertionError(
            "rebuilt net phase does not reproduce the session's G0-4: worst |dncc| = "
            f"{deviation:.3e} > {tolerance:g}. The U-Net forward is device-dependent "
            "at this level (CPU vs CUDA moved ncc by 1e-3 in this repo's own session); "
            "rebuild on the device the session used (--device auto) instead of widening "
            "the tolerance."
        )
    return {
        "max_abs_deviation": float(deviation),
        "recorded_min_ncc": float(min(recorded.values())),
        "recomputed_min_ncc": float(min(values.values())),
        "ncc": values,
    }


def sweep(
    bench: Any,
    phases: dict[int, np.ndarray],
    indices: list[int],
    measured: np.ndarray,
    offsets_px: list[int],
    centre_xy: tuple[int, int],
) -> dict[str, Any]:
    """Evaluate ``min``/``mean`` ``ncc`` over the offset grid.

    Args:
        bench: Twin.
        phases: ``index -> (grid, grid)`` radians.
        indices: Sample order.
        measured: ``(B, ccd, ccd)`` measured net means.
        offsets_px: 1-D offsets to use on **both** axes (the grid is their product).
        centre_xy: The panel centre the offsets are relative to (echo only).

    Returns:
        ``{"offsets_px", "min_ncc", "mean_ncc", "worst_index", "per_sample"}`` with the
        first three following the grid's ``(dy, dx)`` row-major order.
    """
    from ml.sim2f.bench import _ncc

    order = [(dy, dx) for dy in offsets_px for dx in offsets_px]
    min_map = np.full((len(offsets_px), len(offsets_px)), np.nan)
    mean_map = np.full_like(min_map, np.nan)
    worst_index = np.full(min_map.shape, -1, dtype=int)
    per_sample: dict[str, list[float]] = {str(i): [] for i in indices}

    for row, dy in enumerate(offsets_px):
        for column, dx in enumerate(offsets_px):
            shifted = [
                _shift_phase(
                    phases[i], sx=-dx / PITCH_PANEL_PX, sy=-dy / PITCH_PANEL_PX
                )
                for i in indices
            ]
            predicted = twin_frames(bench, shifted)
            values = np.array(
                [_ncc(measured[p], predicted[p]) for p in range(len(indices))]
            )
            min_map[row, column] = float(values.min())
            mean_map[row, column] = float(values.mean())
            worst_index[row, column] = int(indices[int(np.argmin(values))])
            for position, index in enumerate(indices):
                per_sample[str(index)].append(float(values[position]))
    return {
        "offsets_px": list(offsets_px),
        "n_points": len(order),
        "min_ncc": min_map,
        "mean_ncc": mean_map,
        "worst_index": worst_index,
        "per_sample": per_sample,
        "centre_xy": list(centre_xy),
    }


def _best(grid: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered optimum: largest ``min_ncc``, ties broken by proximity to 0.

    Args:
        grid: A :func:`sweep` result.

    Returns:
        ``{"dx", "dy", "radius_px", "min_ncc", "mean_ncc", "worst_index"}``.
    """
    values = grid["min_ncc"]
    offsets = grid["offsets_px"]
    best = None
    for row, dy in enumerate(offsets):
        for column, dx in enumerate(offsets):
            candidate = (
                float(values[row, column]),
                -math.hypot(dx, dy),
                int(dy),
                int(dx),
                float(grid["mean_ncc"][row, column]),
                int(grid["worst_index"][row, column]),
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    assert best is not None
    return {
        "min_ncc": best[0],
        "dx": best[3],
        "dy": best[2],
        "radius_px": math.hypot(best[3], best[2]),
        "mean_ncc": best[4],
        "worst_index": best[5],
    }


def _at(grid: dict[str, Any], dx: int, dy: int) -> dict[str, float] | None:
    """The grid's readings at one offset (``None`` when the point is off the grid)."""
    offsets = grid["offsets_px"]
    if dx not in offsets or dy not in offsets:
        return None
    row, column = offsets.index(dy), offsets.index(dx)
    return {
        "min_ncc": float(grid["min_ncc"][row, column]),
        "mean_ncc": float(grid["mean_ncc"][row, column]),
        "worst_index": int(grid["worst_index"][row, column]),
    }


def dihedral_diagnostic(
    measured: np.ndarray, predicted: np.ndarray, indices: list[int]
) -> dict[str, Any]:
    """Max ``ncc`` over the 8 dihedral transforms of the twin frame (side evidence).

    G0-4 assumes the twin's far-field orientation matches the camera's.  If a
    transpose or a flip is missing, the best shift cannot reach the floor for a reason
    that is **not** pupil placement, and the three-valued conclusion would be blamed on
    the wrong thing.  This is reported next to the verdict and never changes it.

    Args:
        measured: ``(B, ccd, ccd)`` measured net means.
        predicted: ``(B, ccd, ccd)`` twin predictions (already unshifted / best-shifted).
        indices: Sample order.

    Returns:
        ``{"per_transform": [...], "best_transform", "min_over_best", "mean_over_best"}``.
    """
    from ml.sim2f.bench import _ncc

    names = list(DIHEDRAL)
    per_sample = {name: [] for name in names}
    for position in range(len(indices)):
        for name in names:
            per_sample[name].append(
                _ncc(measured[position], DIHEDRAL[name](predicted[position]))
            )
    means = {name: float(np.mean(values)) for name, values in per_sample.items()}
    best = max(means, key=lambda name: means[name])
    return {
        "per_transform": [
            {
                "transform": name,
                "min_ncc": float(np.min(per_sample[name])),
                "mean_ncc": means[name],
            }
            for name in names
        ],
        "best_transform": best,
        "min_over_best": float(np.min(per_sample[best])),
        "mean_over_best": means[best],
        "identity_min": float(np.min(per_sample["identity"])),
    }


def conclude(best: dict[str, Any], floor: float, band_px: float) -> dict[str, Any]:
    """Apply the report's pre-registered three-valued rule, verbatim.

    Args:
        best: :func:`_best` result.
        floor: G0-4's floor, read from ``session.json`` (never written here).
        band_px: The pre-registered ``+-20 px`` band.

    Returns:
        ``{"verdict", "reasons", "radius_px", "min_ncc"}`` where ``verdict`` is ``A``,
        ``B`` or ``C`` per report §0.
    """
    radius = float(best["radius_px"])
    value = float(best["min_ncc"])
    if radius <= band_px:
        verdict = "A"
        text = "placement fine: 0.4898 is the twin's fidelity ceiling; G0-4 stays FAIL, floor unchanged"
    elif value >= floor:
        verdict = "B"
        text = "pupil placement was wrong: the floor is reachable at the optimum; re-run the whole session"
    else:
        verdict = "C"
        text = "both: the optimum is far from the panel centre AND still below the floor"
    return {
        "verdict": verdict,
        "text": text,
        "radius_px": radius,
        "min_ncc": value,
        "floor": float(floor),
        "band_px": float(band_px),
    }


def render(
    grid: dict[str, Any],
    slices: dict[str, Any],
    best: dict[str, Any],
    floor: float,
    out_dir: Path,
    fine: dict[str, Any] | None = None,
) -> list[str]:
    """Draw the 2-D maps, the two 1-D slices and (when present) the fine diagnostic.

    Args:
        grid: :func:`sweep` result.
        slices: ``{"dx": {...}, "dy": {...}}`` one-dimensional readings.
        best: :func:`_best` result.
        floor: G0-4's floor (drawn as a contour, never as a decision).
        out_dir: Figure directory.
        fine: Optional fine diagnostic payload (maps + best + slices).

    Returns:
        The written paths, as strings.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    out_dir.mkdir(parents=True, exist_ok=True)
    offsets = np.array(grid["offsets_px"])
    written: list[str] = []

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for axis, key, label in zip(
        axes, ("min_ncc", "mean_ncc"), ("min ncc (G0-4 读数)", "mean ncc (诊断)")
    ):
        data = grid[key]
        image = axis.imshow(
            data,
            origin="lower",
            extent=(offsets[0], offsets[-1], offsets[0], offsets[-1]),
            cmap="viridis",
        )
        lines = axis.contour(
            offsets,
            offsets,
            data,
            levels=[floor],
            colors=["#d62728"],
            linewidths=1.6,
        )
        axis.clabel(lines, fmt=f"floor {floor:g}", fontsize=8)
        axis.plot(0, 0, marker="+", color="white", markersize=10, label="面板中心 (0,0)")
        axis.plot(
            best["dx"],
            best["dy"],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="white",
            markersize=12,
            label="最优 (min 最大)",
        )
        axis.set_xlabel("Δx / 面板像素 (相对 960,600)")
        axis.set_ylabel("Δy / 面板像素")
        axis.set_title(label)
        axis.legend(loc="upper right", fontsize=7, framealpha=0.7)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle("孪生预测随瞳面中心偏移的吻合度 (24 样本)", fontsize=10)
    figure.tight_layout()
    path = out_dir / "pupil_center_map.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(str(path))

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), sharey=True)
    for axis, key, label in zip(
        axes, ("dx", "dy"), ("沿 Δx 切片 (Δy = 0)", "沿 Δy 切片 (Δx = 0)")
    ):
        block = slices[key]
        axis.plot(block["offset_px"], block["min_ncc"], "o-", label="min ncc")
        axis.plot(block["offset_px"], block["mean_ncc"], "s--", label="mean ncc")
        axis.axhline(floor, color="#d62728", linewidth=1.2, label=f"floor {floor:g}")
        axis.axvline(0, color="grey", linewidth=0.8)
        axis.axvspan(-PLACEMENT_BAND_PX, PLACEMENT_BAND_PX, color="grey", alpha=0.12)
        axis.set_xlabel(label + " / 面板像素")
        axis.set_ylabel("ncc")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    figure.suptitle("一维切片 (灰带 = 预注册 ±20 px 判据带)", fontsize=10)
    figure.tight_layout()
    path = out_dir / "pupil_center_slices.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(str(path))

    if fine is not None:
        fine_offsets = np.array(fine["offsets_px"])
        figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
        data = np.array(fine["min_ncc_map"])
        image = axes[0].imshow(
            data,
            origin="lower",
            extent=(fine_offsets[0], fine_offsets[-1], fine_offsets[0], fine_offsets[-1]),
            cmap="viridis",
        )
        lines = axes[0].contour(
            fine_offsets,
            fine_offsets,
            data,
            levels=[floor],
            colors=["#d62728"],
            linewidths=1.6,
        )
        axes[0].clabel(lines, fmt=f"floor {floor:g}", fontsize=8)
        axes[0].plot(
            fine["best"]["dx"],
            fine["best"]["dy"],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="white",
            markersize=12,
        )
        axes[0].set_xlabel("Δx / 面板像素")
        axes[0].set_ylabel("Δy / 面板像素")
        axes[0].set_title(f"细化 min ncc (步长 {fine['step_px']} px, 形状诊断)")
        figure.colorbar(image, ax=axes[0], fraction=0.046)
        for axis, key, label in zip(
            axes[1:], ("dx", "dy"), ("细化 Δx 切片 (Δy = 0)", "细化 Δy 切片 (Δx = 0)")
        ):
            block = fine["slices"][key]
            axis.plot(block["offset_px"], block["min_ncc"], "o-", label="min ncc")
            axis.plot(block["offset_px"], block["mean_ncc"], "s--", label="mean ncc")
            axis.axhline(floor, color="#d62728", linewidth=1.2, label=f"floor {floor:g}")
            axis.axvline(0, color="grey", linewidth=0.8)
            axis.set_xlabel(label + " / 面板像素")
            axis.set_ylabel("ncc")
            axis.grid(alpha=0.3)
            axis.legend(fontsize=7)
        figure.suptitle("细化扫描 (事后形状诊断, 不重新判定 A/B/C)", fontsize=10)
        figure.tight_layout()
        path = out_dir / "pupil_center_fine.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(str(path))
    return written


def worst_tail(
    measured: np.ndarray,
    zero_prediction: np.ndarray,
    best_prediction: np.ndarray,
    indices: list[int],
) -> dict[str, Any]:
    """How much of the deficit one sample carries, at ``(0, 0)`` versus the best offset.

    ``min`` is a **single-sample** statistic: whether the floor is missed by 0.01 can be
    one sample's value, so a report that only quotes the min cannot say whether a shifted
    optimum is a placement effect or the same sample drifting.  This lists the three
    smallest values at both offsets and how many of the 24 improved.

    Args:
        measured: ``(B, ccd, ccd)`` measured net means.
        zero_prediction: Twin prediction at zero offset.
        best_prediction: Twin prediction at the pre-registered optimum.
        indices: Sample order.

    Returns:
        ``{"at_zero": {...}, "at_best": {...}, "n_improved", "n_improved_by_more_than_001"}``.
    """
    from ml.sim2f.bench import _ncc

    zero = np.array([_ncc(measured[p], zero_prediction[p]) for p in range(len(indices))])
    best = np.array([_ncc(measured[p], best_prediction[p]) for p in range(len(indices))])

    def _tail(values: np.ndarray) -> dict[str, Any]:
        order = np.argsort(values)
        return {
            "min": float(values[order[0]]),
            "second": float(values[order[1]]),
            "third": float(values[order[2]]),
            "min_index": int(indices[int(order[0])]),
            "mean": float(values.mean()),
        }

    return {
        "at_zero": _tail(zero),
        "at_best": _tail(best),
        "n_improved": int((best > zero).sum()),
        "n_improved_by_more_than_001": int(((best - zero) > 0.01).sum()),
        "per_sample_gain": {
            str(index): float(best[p] - zero[p]) for p, index in enumerate(indices)
        },
    }


def _slices(grid: dict[str, Any]) -> dict[str, Any]:
    """The two one-dimensional slices through ``(0, 0)``.

    Args:
        grid: :func:`sweep` result.

    Returns:
        ``{"dx": {...}, "dy": {...}}`` with ``offset_px`` / ``min_ncc`` / ``mean_ncc``.
    """
    offsets = grid["offsets_px"]
    zero = offsets.index(0)

    def _row(index: int) -> dict[str, Any]:
        return {
            "offset_px": list(offsets),
            "min_ncc": [float(v) for v in grid["min_ncc"][index, :]],
            "mean_ncc": [float(v) for v in grid["mean_ncc"][index, :]],
        }

    def _column(index: int) -> dict[str, Any]:
        return {
            "offset_px": list(offsets),
            "min_ncc": [float(v) for v in grid["min_ncc"][:, index]],
            "mean_ncc": [float(v) for v in grid["mean_ncc"][:, index]],
        }

    return {"dx": _row(zero), "dy": _column(zero)}


def _shape_of(curve: list[float]) -> dict[str, Any]:
    """Characterise a 1-D curve: single-peaked, plateaued or monotone.

    Args:
        curve: ``min_ncc`` or ``mean_ncc`` values along one axis.

    Returns:
        ``{"argmax_index", "peak", "n_local_maxima", "plateau_indices",
        "plateau_width"}``.  ``plateau_indices`` are the offsets that stay within 0.01 of
        the peak, so a genuinely flat ridge is visible instead of being reported as
        "the best point".
    """
    values = np.asarray(curve, dtype=float)
    peak = int(np.argmax(values))
    local_maxima = [
        index
        for index in range(len(values))
        if (index == 0 or values[index] >= values[index - 1])
        and (index == len(values) - 1 or values[index] >= values[index + 1])
        and (values[index] > values[index - 1] if index else True)
    ]
    within = np.where(values >= values[peak] - 0.01)[0]
    return {
        "argmax_index": peak,
        "peak": float(values[peak]),
        "n_local_maxima": len(local_maxima),
        "plateau_indices": within.tolist(),
        "plateau_width": int(within[-1] - within[0]) if len(within) else 0,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the sweep and write the artefacts.

    Args:
        argv: CLI arguments (defaults to ``sys.argv``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=ROOT / "runs/sim2f_bench/session",
        help="completed session artefacts (session.json + frames.npz)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs/sim2f_fair/train/model_best.pt",
        help="the checkpoint the session used",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "runs/sim2f_fair/corpus",
        help="corpus directory holding pairs.npz",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "torch device for the rebuilt network phase and the twin; 'auto' mirrors "
            "run_session's own resolution, which is required for bit-level agreement "
            "(the U-Net forward is device-dependent at the 1e-3 ncc level)"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "report/sim2f_bench/twin_vs_hw_corpus/20261027",
        help="report directory (figures/ + artifacts/ go here)",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help="use the wide +-240 px grid directly instead of the coarse one",
    )
    parser.add_argument(
        "--refine-step",
        type=int,
        default=5,
        help=(
            "post-hoc shape diagnostic: after the pre-registered coarse grid has fixed "
            "the A/B/C verdict, sweep a finer +-40 px grid at this step (0 disables). "
            "It characterises the basin and never re-decides the verdict"
        ),
    )
    args = parser.parse_args(argv)

    from loguru import logger

    from ml.sim2f._device import _resolve_device
    from ml.sim2f.config import Sim2fConfig
    from ml.sim2f.optics import Sim2f

    resolved = _resolve_device(args.device)
    payload, blocks = load_session(args.session_dir)
    indices = [int(i) for i in payload["indices"]]
    placement = payload["pupil_placement"]
    centre_xy = tuple(int(v) for v in placement["pupil_center_xy"])
    floor = float(payload["gate0"]["g0_4_geometry"]["floor"])
    recorded_min = float(payload["gate0"]["g0_4_geometry"]["min_ncc"])
    logger.info(
        "session: {} samples, recorded G0-4 min ncc = {:.4f} (floor {:g}), pupil centre "
        "{} (default={})",
        len(indices),
        recorded_min,
        floor,
        centre_xy,
        placement["pupil_center_is_default"],
    )

    phases, rebuild_meta = rebuild_net_phases(
        payload, args.checkpoint, args.corpus, str(resolved)
    )
    measured = measured_nets(blocks, indices)

    bench = Sim2f(Sim2fConfig.calibrated(), device=str(resolved))
    zero_prediction = twin_frames(bench, [phases[i] for i in indices])
    reproduction = check_reproduction(measured, zero_prediction, indices, payload)
    logger.info(
        "rebuild reproduces the session: worst |dncc| = {:.2e}, min ncc {:.4f}",
        reproduction["max_abs_deviation"],
        reproduction["recomputed_min_ncc"],
    )

    offsets = list(WIDE_OFFSETS_PX) if args.wide else list(COARSE_OFFSETS_PX)
    grid = sweep(bench, phases, indices, measured, offsets, centre_xy)
    best = _best(grid)
    logger.info(
        "coarse optimum: dx={} dy={} min={:.4f} mean={:.4f} (radius {:.1f} px); at "
        "(0,0) min={:.4f} mean={:.4f}",
        best["dx"],
        best["dy"],
        best["min_ncc"],
        best["mean_ncc"],
        best["radius_px"],
        grid["min_ncc"][offsets.index(0), offsets.index(0)],
        grid["mean_ncc"][offsets.index(0), offsets.index(0)],
    )

    # README §0 extension rule: a border optimum means the basin is not enclosed.
    extended: dict[str, Any] | None = None
    border = best["dx"] in (offsets[0], offsets[-1]) or best["dy"] in (offsets[0], offsets[-1])
    if border and not args.wide:
        logger.warning(
            "optimum ({}, {}) sits on the coarse grid's border; re-sweeping +-240 px",
            best["dx"],
            best["dy"],
        )
        extended = sweep(
            bench, phases, indices, measured, list(WIDE_OFFSETS_PX), centre_xy
        )
        extended_best = _best(extended)
        if extended_best["min_ncc"] > best["min_ncc"]:
            grid, best = extended, extended_best
            logger.info(
                "wide optimum: dx={} dy={} min={:.4f} mean={:.4f}",
                best["dx"],
                best["dy"],
                best["min_ncc"],
                best["mean_ncc"],
            )

    slices = _slices(grid)
    verdict = conclude(best, floor, PLACEMENT_BAND_PX)
    best_prediction = twin_frames(
        bench,
        [
            _shift_phase(
                phases[i],
                sx=-best["dx"] / PITCH_PANEL_PX,
                sy=-best["dy"] / PITCH_PANEL_PX,
            )
            for i in indices
        ],
    )
    dihedral_zero = dihedral_diagnostic(measured, zero_prediction, indices)
    dihedral_best = dihedral_diagnostic(measured, best_prediction, indices)
    tail = worst_tail(measured, zero_prediction, best_prediction, indices)

    # Post-hoc shape diagnostic (report §0 次判据).  The verdict above is already fixed
    # by the pre-registered coarse grid; this only says whether 0.4990 is a peak or a
    # plateau, and whether a between-grid point reaches the floor.
    fine: dict[str, Any] | None = None
    if int(args.refine_step) > 0 and not args.wide:
        step = int(args.refine_step)
        fine_offsets = list(range(-40, 41, step))
        fine_grid = sweep(
            bench, phases, indices, measured, fine_offsets, centre_xy
        )
        fine_best = _best(fine_grid)
        logger.info(
            "fine ({} px) diagnostic optimum: dx={} dy={} min={:.4f} mean={:.4f}",
            step,
            fine_best["dx"],
            fine_best["dy"],
            fine_best["min_ncc"],
            fine_best["mean_ncc"],
        )
        fine = {
            "step_px": step,
            "offsets_px": fine_offsets,
            "min_ncc_map": fine_grid["min_ncc"].tolist(),
            "mean_ncc_map": fine_grid["mean_ncc"].tolist(),
            "worst_index_map": fine_grid["worst_index"].tolist(),
            "best": fine_best,
            "reaches_floor": bool(fine_best["min_ncc"] >= floor),
            "slices": _slices(fine_grid),
        }

    # Per-sample dwelt: how far each sample's own optimum is from (0, 0), so a session
    # whose 24 samples disagree cannot be read as "one placement".
    per_sample_best: dict[str, dict[str, Any]] = {}
    for position, index in enumerate(indices):
        values = np.asarray(grid["per_sample"][str(index)])
        peak = int(np.argmax(values))
        per_sample_best[str(index)] = {
            "best_offset_index": peak,
            "ncc_at_peak": float(values[peak]),
            "ncc_at_zero": float(values[offsets.index(0) * len(offsets) + offsets.index(0)]),
            "top5": sorted(
                (
                    {
                        "dx": int(offsets[i % len(offsets)]),
                        "dy": int(offsets[i // len(offsets)]),
                        "ncc": float(values[i]),
                    }
                    for i in np.argsort(values)[-5:]
                ),
                key=lambda item: -item["ncc"],
            ),
        }

    figures = render(grid, slices, best, floor, args.out_dir / "figures", fine)
    at_zero = _at(grid, 0, 0)
    artefacts = args.out_dir / "artifacts"
    artefacts.mkdir(parents=True, exist_ok=True)
    result = {
        "config": _sha(args),
        "session": {
            "dir": str(args.session_dir),
            "n": len(indices),
            "indices": indices,
            "recorded_min_ncc": recorded_min,
            "recorded_floor": floor,
            "pupil_center_xy": list(centre_xy),
            "pupil_center_is_default": placement["pupil_center_is_default"],
            "exposure_ms": payload.get("exposure_ms"),
        },
        "rebuild": rebuild_meta,
        "reproduction": reproduction,
        "grid_extended": extended is not None,
        "grid_offsets_px": list(grid["offsets_px"]),
        "min_ncc_map": grid["min_ncc"].tolist(),
        "mean_ncc_map": grid["mean_ncc"].tolist(),
        "worst_index_map": grid["worst_index"].tolist(),
        "best": best,
        "at_zero": at_zero,
        "slices": slices,
        "slice_shape": {
            "dx_min": _shape_of(slices["dx"]["min_ncc"]),
            "dx_mean": _shape_of(slices["dx"]["mean_ncc"]),
            "dy_min": _shape_of(slices["dy"]["min_ncc"]),
            "dy_mean": _shape_of(slices["dy"]["mean_ncc"]),
        },
        "verdict": verdict,
        "worst_tail": tail,
        "dihedral": {"zero_shift": dihedral_zero, "best_shift": dihedral_best},
        "fine": fine,
        "per_sample_best": per_sample_best,
        "figures": figures,
    }
    path = artefacts / "pupil_center_sweep.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("artefacts: {} + {} figure(s)", path, len(figures))
    logger.info(json.dumps({"verdict": verdict, "best": best, "at_zero": at_zero}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
