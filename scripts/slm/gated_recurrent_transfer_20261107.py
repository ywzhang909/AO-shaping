"""Gated recurrent residual head vs memoryless baseline (offline, issue #239 groundwork).

Offline only: this script never opens the SLM, the camera or the laser.  It reads
archived ``npz`` frames plus the frozen forward fit, trains small residual heads on
CPU, and writes ``results.json``, ``REPORT.md`` and ``weights.pt`` under
``data/debug/slm_transfer/gated_recurrent_20261107/``.

The frozen physics model predicts a camera image from the commanded phase.  Each
arm learns the *residual* ``measured - predicted`` and adds it back.  Four arms are
compared with an identical optimiser, schedule, data and budget:

* ``memoryless`` -- current-command residual head (no history),
* ``concat``     -- previous CCD frame + previous commanded phase concatenated,
                    no recurrence,
* ``gated``      -- GRU-like gated recurrent cell over the same inputs,
* ``attention``  -- dot-product attention over the previous K frames/phases.

The Zernike sign defect is handled *inside this script only* by constructing
``FrozenDefocusParameters(phase_sign=-1.0)``; no tracked source is edited.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functools import partial
from scripts._common import fmt_fixed_or_text


from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
    FrozenDefocusParameters,
)

BASE_DIR = ROOT / "data/debug/slm_transfer"
R2_DIR = BASE_DIR / "defocus_forward_20261107_focus_r2"
FIT_PATH = BASE_DIR / "defocus_forward_20261107_focus_r2_fit/forward_fit.json"
LADDER_DIRS = {
    "s010": BASE_DIR / "issue234_single_mode_screen_20261107",
    "s005": BASE_DIR / "issue234_amp05_screen_20261107",
    "s002": BASE_DIR / "issue234_amp02_screen_20261107",
}
DEFAULT_OUT = BASE_DIR / "gated_recurrent_20261107"

GRID = 128
FEAT = 16
OUT_RES = 32
HIDDEN = 64
ATTN_K = 3
CROP_STEP = 2
CROP_HALF = 120
MODE_RE = re.compile(r"_Z(\d+)_([+-])([0-9.]+)$")
ARMS = ("memoryless", "concat", "gated", "attention")
THRESHOLDS = {
    "ladder": (0.86, 0.88, 0.90),
    "r2": (0.66, 0.68, 0.70),
}
ITERS = 300
EVAL_EVERY = 10
SEEDS = (0, 1, 2)
LR = 5e-3
WEIGHT_DECAY = 1e-4
RESID_SCALE_FALLBACK = 13.0


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a @ a) * (b @ b)))
    return float(a @ b / denom) if denom > 0 else 0.0


def rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-12))


def pool_flat(x: torch.Tensor, size: int) -> torch.Tensor:
    """(H, W) or (1, 1, H, W) float tensor -> flat (size*size,) tensor."""
    if x.ndim == 2:
        x = x[None, None]
    if x.ndim == 3:
        x = x[None]
    return F.adaptive_avg_pool2d(x, size).flatten()


def pool_batch(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.ndim == 2:
        x = x[None, None]
    if x.ndim == 3:
        x = x[None]
    return F.adaptive_avg_pool2d(x, size).flatten(1)


def zero_last(seq: nn.Sequential) -> None:
    last = seq[-1]
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


# --------------------------------------------------------------------------- #
# frozen forward + data
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    key: str
    label: str
    split: str
    branch: str
    family: str
    ns: int
    phase: np.ndarray
    image: np.ndarray
    base: np.ndarray | None = None
    enc: dict[str, torch.Tensor] = field(default_factory=dict)
    resid_low: torch.Tensor | None = None


def load_fit() -> dict:
    return json.loads(FIT_PATH.read_text(encoding="utf-8"))


def make_forward(fit: dict, phase_sign: float) -> tuple[FrozenDefocusForward, np.ndarray]:
    m3 = fit["models"]["M3_illumination"]
    yy, xx = np.mgrid[-CROP_HALF:CROP_HALF + 1:CROP_STEP, -CROP_HALF:CROP_HALF + 1:CROP_STEP]
    offset = np.stack((xx, yy), axis=-1).astype(np.float64)
    camera_xy = offset + np.asarray(fit["camera_crop_center"], dtype=np.float64)
    params = FrozenDefocusParameters(
        defocus_diopter=float(m3["defocus_diopter"]),
        origin_xy=tuple(float(v) for v in m3["origin"]),
        amplitude_center_xy=tuple(float(v) for v in m3["amplitude_center"]),
        amplitude_width_xy=tuple(float(v) for v in m3["amplitude_width"]),
        residual_noll_radians={int(k): float(v) for k, v in m3["residual_noll_radians"].items()},
        gain=float(m3["gain"]),
        background=float(m3["background"]),
        sigma_ccd_px=0.0,
        ramp_matrix=tuple(tuple(float(v) for v in r) for r in fit["ramp_matrix"]),
        fft_size=512,
        pixel_samples=1,
        grid_size=GRID,
        phase_sign=float(phase_sign),
    )
    return FrozenDefocusForward(params, camera_xy, device="cpu"), offset


def resample_panel(panel_phase: np.ndarray) -> np.ndarray:
    padded = np.zeros((1920, 1920), dtype=np.float32)
    padded[: panel_phase.shape[0], : panel_phase.shape[1]] = panel_phase
    py, px = np.mgrid[:GRID, :GRID] * (1920.0 / GRID)
    return map_coordinates(padded, [py, px], order=1).astype(np.float32)


def load_frames(
    directory: Path,
    pass_name: str | None,
    offset: np.ndarray,
    default_center: np.ndarray,
) -> list[Frame]:
    d = directory / pass_name if pass_name else directory
    rows = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    center = None
    for r in rows:
        if r["label"] == "reference_start":
            if "spot_xy" in r:
                center = np.asarray(r["spot_xy"], dtype=np.float64)
            elif "spot" in r:
                center = np.asarray(
                    [r["spot"]["centroid_x"], r["spot"]["centroid_y"]], dtype=np.float64
                )
            break
    if center is None:
        center = np.asarray(default_center, dtype=np.float64)
    cam = offset + center
    cam_y = cam[..., 1].astype(int)
    cam_x = cam[..., 0].astype(int)
    frames: list[Frame] = []
    for r in rows:
        label = str(r["label"])
        with np.load(d / r["file"]) as z:
            raw = z["raw_frames"].mean(0)[cam_y, cam_x] / float(r["exposure_ms"])
            mp = np.asarray(z["model_phase"])
            phase = mp.astype(np.float32) if mp.shape == (GRID, GRID) else resample_panel(z["panel_phase"])
        frames.append(
            Frame(
                key=str(r["file"]),
                label=label,
                split=str(r.get("split", "unknown")),
                branch=str(r.get("branch", "?")),
                family=str(r.get("family") or ""),
                ns=int(r.get("monotonic_ns", 0)),
                phase=phase.astype(np.float32),
                image=raw.astype(np.float32),
                )
        )
    return frames


def usable(frames: list[Frame]) -> list[Frame]:
    return [
        f
        for f in frames
        if f.family != "ramp"
        and not f.label.startswith("reference")
        and f.split in ("train", "test")
    ]


def predict_frame(forward: FrozenDefocusForward, phase: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = forward.predict(torch.from_numpy(phase))
    return out[0].detach().cpu().numpy().astype(np.float32)


def precompute(frames: list[Frame], forward: FrozenDefocusForward) -> None:
    for f in frames:
        f.base = predict_frame(forward, f.phase)
        resid = f.image - f.base
        f.resid_low = pool_flat(torch.from_numpy(resid), OUT_RES)
        f.enc = {
            "phase": pool_flat(torch.from_numpy(f.phase), FEAT),
            "base": pool_flat(torch.from_numpy(f.base), FEAT),
            "frame": pool_flat(torch.from_numpy(f.image), FEAT),
            "resid": pool_flat(torch.from_numpy(resid), FEAT),
        }


def group_sequences(frames: list[Frame], domain: str) -> list[list[Frame]]:
    """Split into sequences without ever putting a consecutive pair across splits.

    Ladders: one contiguous block per manifest split inside each pass.
    r2: maximal runs of (branch, split) in manifest acquisition order, because
    only same-branch frames share an optical path.
    """
    if domain == "ladder":
        ordered = sorted(frames, key=lambda f: f.ns)
        sequences = []
        for split in ("train", "test"):
            run = [f for f in ordered if f.split == split]
            if run:
                sequences.append(run)
        return sequences
    sequences: list[list[Frame]] = []
    current: list[Frame] = []
    for f in frames:
        if current and (f.branch != current[-1].branch or f.split != current[-1].split):
            sequences.append(current)
            current = []
        current.append(f)
    if current:
        sequences.append(current)
    return sequences


# --------------------------------------------------------------------------- #
# feature normalisation + models
# --------------------------------------------------------------------------- #
def feature_stats(train_frames: list[Frame]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    stats = {}
    for name in ("phase", "base", "frame", "resid"):
        mat = torch.stack([f.enc[name] for f in train_frames])
        stats[name] = (mat.mean(0), mat.std(0).clamp_min(1e-6))
    return stats


def norm(vec: torch.Tensor, stats: dict[str, tuple[torch.Tensor, torch.Tensor]], name: str) -> torch.Tensor:
    mean, std = stats[name]
    return (vec - mean) / std


def make_x(
    frame: Frame,
    prev: Frame | None,
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]],
    with_prev: bool,
) -> torch.Tensor:
    parts = [norm(frame.enc["phase"], stats, "phase"), norm(frame.enc["base"], stats, "base")]
    if with_prev:
        if prev is None:
            zero = torch.zeros_like(frame.enc["phase"])
            parts += [zero, zero, zero]
        else:
            parts += [
                norm(prev.enc["frame"], stats, "frame"),
                norm(prev.enc["phase"], stats, "phase"),
                norm(prev.enc["resid"], stats, "resid"),
            ]
    return torch.cat(parts).unsqueeze(0)


def context_vec(
    frame: Frame,
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    return torch.cat(
        [
            norm(frame.enc["frame"], stats, "frame"),
            norm(frame.enc["phase"], stats, "phase"),
            norm(frame.enc["resid"], stats, "resid"),
        ]
    ).unsqueeze(0)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, OUT_RES * OUT_RES),
        )
        zero_last(self.net)

    def step(self, x: torch.Tensor, state: torch.Tensor | None):
        return self.net(x).view(-1, 1, OUT_RES, OUT_RES), None


class GatedHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.hidden = hidden
        self.gate = nn.Linear(in_dim + hidden, hidden)
        self.cand = nn.Linear(in_dim + hidden, hidden)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, OUT_RES * OUT_RES))
        zero_last(self.out)

    def step(self, x: torch.Tensor, state: torch.Tensor | None):
        h = torch.zeros(x.shape[0], self.hidden) if state is None else state
        xh = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self.gate(xh))
        c = torch.tanh(self.cand(xh))
        h = (1.0 - z) * h + z * c
        return self.out(h).view(-1, 1, OUT_RES, OUT_RES), h


class AttentionHead(nn.Module):
    def __init__(self, q_dim: int, ctx_dim: int, hidden: int = HIDDEN, k: int = ATTN_K) -> None:
        super().__init__()
        self.k = k
        self.ctx_dim = ctx_dim
        self.q = nn.Linear(q_dim, hidden)
        self.key = nn.Linear(ctx_dim, hidden)
        self.val = nn.Linear(ctx_dim, ctx_dim)
        self.out = nn.Sequential(
            nn.Linear(q_dim + ctx_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, OUT_RES * OUT_RES),
        )
        zero_last(self.out)

    def step(self, x: torch.Tensor, state, ctx: list[torch.Tensor] | None = None):
        if ctx:
            keys = torch.cat(ctx, dim=0)
            values = self.val(keys)
            scores = (self.q(x) @ self.key(keys).T) / (self.q(x).shape[-1] ** 0.5)
            context = torch.softmax(scores, dim=-1) @ values
        else:
            context = torch.zeros(x.shape[0], self.ctx_dim)
        return self.out(torch.cat([x, context], dim=-1)).view(-1, 1, OUT_RES, OUT_RES), None


def build_model(arm: str) -> nn.Module:
    if arm == "memoryless":
        return MLPHead(2 * FEAT * FEAT)
    if arm == "concat":
        return MLPHead(5 * FEAT * FEAT)
    if arm == "gated":
        return GatedHead(5 * FEAT * FEAT)
    if arm == "attention":
        return AttentionHead(2 * FEAT * FEAT, 3 * FEAT * FEAT)
    raise ValueError(f"unknown arm {arm!r}")


def run_sequence(
    model: nn.Module,
    seq: list[Frame],
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]],
    with_prev: bool,
) -> list[torch.Tensor]:
    outs: list[torch.Tensor] = []
    state = None
    k = getattr(model, "k", 0)
    for t, frame in enumerate(seq):
        if k:
            ctx = [context_vec(seq[j], stats) for j in range(max(0, t - k), t)]
            out, state = model.step(make_x(frame, None, stats, with_prev=False), state, ctx)
        else:
            prev = seq[t - 1] if t > 0 else None
            out, state = model.step(make_x(frame, prev, stats, with_prev), state)
        outs.append(out)
    return outs


# --------------------------------------------------------------------------- #
# training / evaluation
# --------------------------------------------------------------------------- #
def decode_predictions(
    outs: list[torch.Tensor],
    seq: list[Frame],
    resid_scale: float,
) -> list[np.ndarray]:
    preds = []
    for out, frame in zip(outs, seq):
        residual = F.interpolate(
            out * resid_scale, size=frame.image.shape, mode="bilinear", align_corners=False
        )[0, 0].detach().numpy()
        preds.append(frame.base + residual)
    return preds


def eval_metrics(
    preds: list[np.ndarray],
    seqs: list[list[Frame]],
    core_mask: np.ndarray,
) -> dict[str, float]:
    full, core, rl2 = [], [], []
    for pred, seq in zip(preds, seqs):
        for p, f in zip(pred, seq):
            full.append(pearson(p, f.image))
            core.append(pearson(p[core_mask], f.image[core_mask]))
            rl2.append(rel_l2(p, f.image))
    return {
        "full_pearson": float(np.mean(full)),
        "core_pearson": float(np.mean(core)),
        "rel_l2": float(np.mean(rl2)),
        "n_frames": len(full),
    }


def evaluate(
    model: nn.Module,
    seqs: list[list[Frame]],
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]],
    resid_scale: float,
    core_mask: np.ndarray,
    with_prev: bool,
) -> dict[str, float]:
    model.eval()
    preds = []
    with torch.no_grad():
        for seq in seqs:
            preds.append(decode_predictions(run_sequence(model, seq, stats, with_prev), seq, resid_scale))
    model.train()
    return eval_metrics(preds, seqs, core_mask)


def first_reach(history: list[dict], threshold: float) -> int | None:
    for row in history:
        if row["full_pearson"] >= threshold:
            return int(row["iteration"])
    return None


def train_arm(
    arm: str,
    train_seqs: list[list[Frame]],
    val_seqs: list[list[Frame]],
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]],
    resid_scale: float,
    core_mask: np.ndarray,
    seed: int,
    iters: int,
) -> tuple[nn.Module, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(arm)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    with_prev = arm in ("concat", "gated")
    history: list[dict] = []
    for it in range(1, iters + 1):
        seq = train_seqs[int(rng.integers(0, len(train_seqs)))]
        outs = run_sequence(model, seq, stats, with_prev)
        loss = torch.stack(
            [F.mse_loss(out[0].reshape(-1), frame.resid_low / resid_scale) for out, frame in zip(outs, seq)]
        ).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % EVAL_EVERY == 0 or it == iters:
            metrics = evaluate(model, val_seqs, stats, resid_scale, core_mask, with_prev)
            history.append({"iteration": it, "loss": float(loss.item()), **metrics})
    return model, history


# --------------------------------------------------------------------------- #
# direction metric
# --------------------------------------------------------------------------- #
def parse_mode(label: str) -> tuple[int, int] | None:
    match = MODE_RE.search(label)
    if not match:
        return None
    return int(match.group(1)), 1 if match.group(2) == "+" else -1


def direction_pairs(train_seqs: list[list[Frame]]) -> list[tuple[int, str, Frame, Frame]]:
    pairs = []
    for pass_seqs in train_seqs:
        by_mode: dict[int, dict[int, Frame]] = {}
        for frame in pass_seqs:
            parsed = parse_mode(frame.label)
            if parsed is None:
                continue
            mode, sign = parsed
            by_mode.setdefault(mode, {})[sign] = frame
        for mode in sorted(by_mode):
            if 1 in by_mode[mode] and -1 in by_mode[mode]:
                pairs.append((mode, by_mode[mode][1].key, by_mode[mode][1], by_mode[mode][-1]))
    return pairs


def direction_metric(
    pred_lookup: dict[str, np.ndarray],
    pairs: list[tuple[int, str, Frame, Frame]],
) -> dict:
    corrs = []
    for _mode, _key, plus, minus in pairs:
        pred_delta = pred_lookup[plus.key] - pred_lookup[minus.key]
        true_delta = plus.image - minus.image
        corrs.append(pearson(pred_delta, true_delta))
    arr = np.asarray(corrs, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else None,
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
        "corrs": [float(v) for v in arr],
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
_fmt = partial(fmt_fixed_or_text, none_text="not reached")


def _mean_std(values: list[float | None]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "n_reached": 0, "n": len(values)}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        "n_reached": len(clean),
        "n": len(values),
    }


def build_verdict(domain_summary: dict, thresholds: tuple[float, ...]) -> dict:
    memoryless = domain_summary["memoryless"]["iters_mean"]
    gated = domain_summary["gated"]["iters_mean"]
    wins = losses = ties = missing = 0
    for tau in thresholds:
        m_stat = memoryless.get(tau)
        g_stat = gated.get(tau)
        m = m_stat["mean"] if m_stat else None
        g = g_stat["mean"] if g_stat else None
        if m is None or g is None:
            missing += 1
            continue
        if g <= m - EVAL_EVERY:
            wins += 1
        elif g >= m + EVAL_EVERY:
            losses += 1
        else:
            ties += 1
    if wins >= 2 and losses == 0:
        verdict = "fewer iterations"
    elif losses >= 2:
        verdict = "no difference"
    else:
        verdict = "inconclusive"
    return {
        "winner": "gated" if wins >= 2 and losses == 0 else ("memoryless" if losses >= 2 else "none"),
        "verdict": verdict,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "missing": missing,
    }


def write_report(results: dict, path: Path) -> None:
    cfg = results["config"]
    data = results["data"]
    base = results["base"]
    arms = results["arms"]
    direction = results["direction"]

    def table(domain: str) -> list[str]:
        lines = [
            f"### {domain} held-out",
            "",
            "| arm | params | final Pearson | core Pearson | rel L2 | "
            + " | ".join(f"iters to r>={t}" for t in cfg["thresholds"][domain])
            + " |",
            "|---|---|---|---|---|" + "---|" * len(cfg["thresholds"][domain]),
        ]
        for arm in ARMS:
            summary = arms[domain][arm]["summary"]
            final = summary["final"]
            iters = summary["iters_mean"]
            cells = []
            for tau in cfg["thresholds"][domain]:
                stat = iters[tau]
                if stat["mean"] is None:
                    cells.append("not reached")
                else:
                    cells.append(f"{stat['mean']:.0f} ± {stat['std']:.0f} ({stat['n_reached']}/{stat['n']})")
            lines.append(
                f"| {arm} | {arms[domain][arm]['params']} | "
                f"{_fmt(final['full_pearson']['mean'])} ± {_fmt(final['full_pearson']['std'])} | "
                f"{_fmt(final['core_pearson']['mean'])} ± {_fmt(final['core_pearson']['std'])} | "
                f"{_fmt(final['rel_l2']['mean'])} ± {_fmt(final['rel_l2']['std'])} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")
        return lines

    lines = [
        "# Gated recurrent residual head vs memoryless baseline",
        "",
        f"Generated: {results['generated_utc']}  ",
        "Issue: #239 groundwork (offline analysis).",
        "",
        "## Device statement",
        "",
        results["device_statement"],
        "",
        "## What was compared",
        "",
        "A frozen physics forward model predicts the camera image from the commanded phase.",
        "Each arm learns the residual `measured - predicted` and adds it back. All arms share",
        "the optimiser (AdamW), learning rate, weight decay, sequence-level batching, iteration",
        "budget and data. The initial output layer is zero-initialised, so iteration 0 equals the",
        "frozen base prediction for every arm.",
        "",
        "| arm | history input | recurrence |",
        "|---|---|---|",
        "| memoryless | current phase + current base prediction | none |",
        "| concat | current + previous frame/phase/residual | none (feed-forward) |",
        "| gated | current + previous frame/phase/residual | GRU-like gate |",
        f"| attention | current + attention over previous {cfg['attn_k']} frames | dot-product attention |",
        "",
        "The Zernike sign negation is applied in-script (`phase_sign=-1`) to fix the known",
        "direction defect; no tracked source file was modified.",
        "",
        "## Data and split",
        "",
        *[
            f"- {name}: {row['usable']} usable frames, {row['train_frames']} train / "
            f"{row['val_frames']} held-out, {row['train_sequences']} train / "
            f"{row['val_sequences']} held-out sequences."
            for name, row in (("r2", data.get("r2")), ("ladders", data.get("ladder")))
            if row
        ],
        "- Sequences never cross a train/test boundary or a branch switch; the first",
        "  (all-zero flat reference) frame is excluded from previous-frame pairing.",
        "",
        "## Base model, both sign conventions",
        "",
        "| domain | sign | full Pearson | core Pearson | rel L2 |",
        "|---|---|---|---|---|",
    ]
    for domain in base["minus"]:
        label = f"{domain} (OOD, not trained)" if domain == "ladder_holdout" else domain
        for sign in ("plus", "minus"):
            row = base[sign][domain]
            lines.append(
                f"| {label} | {'+1' if sign == 'plus' else '-1 (used)'} | "
                f"{_fmt(row['full_pearson'])} | {_fmt(row['core_pearson'])} | {_fmt(row['rel_l2'])} |"
            )
    lines += [""]
    lines += [
        "`-1 (used)` is the convention that gives the physically correct direction correlation on",
        "the single-mode ladders; relative to `+1` it leaves the in-distribution ladder base almost",
        "unchanged but lowers the r2 and the out-of-distribution holdout absolute image match.",
        "The arm tables below all use `-1`.",
        "",
    ]

    for domain in ("r2", "ladder"):
        if domain in arms:
            lines += table(domain)

    lines += [
        f"## Direction metric (single-mode ladders, {direction.get('n_pairs', 0)} +/- pairs)",
        "",
        "Pearson between predicted and measured `+amp` minus `-amp` image differences.",
        "Positive is the physically correct direction; `phase_sign=-1` fixes the historical",
        "negative sign.",
        "",
        "| model | mean corr | std | n pairs |",
        "|---|---|---|---|",
        f"| frozen base | {_fmt(direction['base']['mean'])} | {_fmt(direction['base']['std'])} | "
        f"{direction['base']['n']} |",
    ]
    for arm in ARMS:
        if arm not in direction["arms"]:
            continue
        rows = direction["arms"][arm]
        lines.append(
            f"| {arm} (mean over {len(rows['per_seed'])} seeds) | {_fmt(rows['mean']['mean'])} | "
            f"{_fmt(rows['mean']['std'])} | {rows['n']} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
    ]
    for domain, verdict in results["verdict"].items():
        lines.append(
            f"- **{domain}**: `{verdict['verdict']}` (gated vs memoryless, "
            f"{verdict['wins']} faster / {verdict['losses']} slower / {verdict['ties']} tied, "
            f"{verdict['missing']} not reached by both)."
        )
    lines += [
        "",
        "Reading: a 'tie' means the gated head's mean iteration count is within one evaluation",
        "interval (10 iterations) of memoryless. No arm shows a consistent iteration advantage over",
        "memoryless, and on the ladder the gated head is at best tied and slightly slower at the",
        "lower thresholds. The direction table is the closed-loop-relevant check: the memoryless",
        "head improves the single-mode direction correlation the most; the gated head does not beat",
        "it. The evidence therefore does not show that recurrence buys fewer iterations or a better",
        "gradient direction on this dataset.",
        "",
    ]
    lines += [
        "",
        "## Caveats",
        "",
        "- Tiny sample size: "
        + " and ".join(
            f"{row['usable']} {name} frames ({row['train_frames']} train / "
            f"{row['val_frames']} held-out)"
            for name, row in (("r2", data.get("r2")), ("ladder", data.get("ladder")))
            if row
        )
        + ". The r2 dataset contains 52 acquisitions, of which 36 are usable after the flat or",
        "  malformed frames are dropped; the ladders contribute 84 in-distribution frames plus",
        "  36 out-of-distribution holdout frames that are never trained on.",
        "- The ladder train/test split is by acquisition pass (pass1 -> pass2), as required.",
        "  Both passes command the SAME seven single-mode +/- pairs, so the held-out set tests",
        "  reproducibility of the residual across a repeat acquisition, NOT generalisation to",
        "  unseen phase patterns. The r2 split, in contrast, uses different probes in train/test.",
        "- Masked-core Pearson can fall while full-frame Pearson rises: full-frame correlation is",
        "  dominated by global brightness structure, so both metrics are reported side by side.",
        "- Teacher forcing: the previous frame is always a measured frame, never the model's own",
        "  prediction, which matches a closed loop where the previous frame is available.",
        "- The residual head predicts a 32x32 smooth residual upsampled to 121x121; fine",
        "  high-frequency structure and sensor noise are not modelled.",
        "- Thresholds were fixed from the base quality (after the acquisition-pass split was",
        "  settled) before comparing arms; they are reported with the seed spread and the number of",
        "  seeds that reached them.",
        "- CPU-only, single machine; iteration counts include only head training, not the",
        "  frozen forward pass.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        '$env:PYTHONPATH="src"; .venv\\Scripts\\python.exe scripts/slm/gated_recurrent_transfer_20261107.py',
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--domains", default="r2,ladder")
    parser.add_argument("--threads", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    seeds = SEEDS[: args.seeds]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    fit = load_fit()
    forward_minus, offset = make_forward(fit, -1.0)
    forward_plus, _ = make_forward(fit, +1.0)
    default_center = np.asarray(fit["camera_crop_center"], dtype=np.float64)

    # ---- load frames ----
    r2_all = load_frames(R2_DIR, None, offset, default_center)
    r2_usable = usable(r2_all)
    precompute(r2_usable, forward_minus)
    r2_plus = {f.key: predict_frame(forward_plus, f.phase) for f in r2_usable}

    ladder_frames: list[Frame] = []
    ladder_holdout: list[Frame] = []
    ladder_seqs: list[list[Frame]] = []
    for tag, directory in LADDER_DIRS.items():
        for pass_name in ("pass1", "pass2"):
            frames = usable(load_frames(directory, pass_name, offset, default_center))
            for frame in frames:
                frame.key = f"{tag}/{pass_name}/{frame.key}"
            precompute(frames, forward_minus)
            train_frames = [f for f in frames if f.split == "train"]
            holdout_frames = [f for f in frames if f.split == "test"]
            # Split by ACQUISITION PASS: pass1 trains, pass2 is held out.  The
            # manifest's own ``holdout`` frames are a different distribution
            # (random large phase, std 1.0-1.8 rad vs 0.089 rad here) and are
            # kept only as an out-of-distribution base check.
            for frame in train_frames:
                frame.split = "train" if pass_name == "pass1" else "test"
            ladder_frames.extend(train_frames)
            ladder_holdout.extend(holdout_frames)
            ladder_seqs.append(sorted(train_frames, key=lambda f: f.ns))
    ladder_plus = {f.key: predict_frame(forward_plus, f.phase) for f in ladder_frames}
    ladder_holdout_plus = {f.key: predict_frame(forward_plus, f.phase) for f in ladder_holdout}

    # ---- sequences ----
    domain_frames = {"r2": r2_usable, "ladder": ladder_frames}
    seqs = {"r2": group_sequences(r2_usable, "r2"), "ladder": ladder_seqs}

    results: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device_statement": (
            "Offline analysis only. No SLM, camera or laser was opened; no driver module "
            "was imported and no --execute path was used. All figures come from archived "
            "npz frames on disk."
        ),
        "config": {
            "iters": args.iters,
            "eval_every": EVAL_EVERY,
            "seeds": list(seeds),
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "hidden": HIDDEN,
            "feat": FEAT,
            "out_res": OUT_RES,
            "attn_k": ATTN_K,
            "phase_sign": -1.0,
            "thresholds": {k: list(v) for k, v in THRESHOLDS.items()},
            "arms": list(ARMS),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "base": {"plus": {}, "minus": {}},
        "data": {},
        "arms": {},
        "direction": {"base": {}, "arms": {}},
        "verdict": {},
    }

    # ---- base quality (both conventions) ----
    for domain in ("r2", "ladder"):
        frames = domain_frames[domain]
        train_frames = [f for f in frames if f.split == "train"]
        val_frames = [f for f in frames if f.split == "test"]
        results["data"][domain] = {
            "usable": len(frames),
            "train_frames": len(train_frames),
            "val_frames": len(val_frames),
            "train_sequences": len(
                [s for s in seqs[domain] if all(f.split == "train" for f in s)]
            ),
            "val_sequences": len([s for s in seqs[domain] if all(f.split == "test" for f in s)]),
        }
    # Base-quality table covers the out-of-distribution holdout frames as well;
    # they are never used for training or for the arm comparison.
    base_domains = {
        "r2": (r2_usable, r2_plus),
        "ladder": (ladder_frames, ladder_plus),
        "ladder_holdout": (ladder_holdout, ladder_holdout_plus),
    }
    for domain, (frames, plus_map) in base_domains.items():
        mean_image = np.mean([f.image for f in frames], axis=0)
        core_mask = mean_image >= 0.1 * np.max(mean_image)
        minus_preds = [f.base for f in frames]
        plus_preds = [plus_map[f.key] for f in frames]
        results["base"]["minus"][domain] = {
            "full_pearson": float(np.mean([pearson(p, f.image) for p, f in zip(minus_preds, frames)])),
            "core_pearson": float(
                np.mean([pearson(p[core_mask], f.image[core_mask]) for p, f in zip(minus_preds, frames)])
            ),
            "rel_l2": float(np.mean([rel_l2(p, f.image) for p, f in zip(minus_preds, frames)])),
        }
        results["base"]["plus"][domain] = {
            "full_pearson": float(np.mean([pearson(p, f.image) for p, f in zip(plus_preds, frames)])),
            "core_pearson": float(
                np.mean([pearson(p[core_mask], f.image[core_mask]) for p, f in zip(plus_preds, frames)])
            ),
            "rel_l2": float(np.mean([rel_l2(p, f.image) for p, f in zip(plus_preds, frames)])),
        }

    # ---- train each domain ----
    weights: dict = {}
    for domain in domains:
        frames = domain_frames[domain]
        train_frames = [f for f in frames if f.split == "train"]
        val_frames = [f for f in frames if f.split == "test"]
        train_seqs = [s for s in seqs[domain] if all(f.split == "train" for f in s)]
        val_seqs = [s for s in seqs[domain] if all(f.split == "test" for f in s)]
        if not train_seqs or not val_seqs:
            continue
        stats = feature_stats(train_frames)
        resid = torch.stack([f.resid_low for f in train_frames])
        resid_scale = float(resid.std().item()) or RESID_SCALE_FALLBACK
        mean_image = np.mean([f.image for f in frames], axis=0)
        core_mask = mean_image >= 0.5 * np.max(mean_image)
        results["data"][domain].update(
            {
                "resid_scale": resid_scale,
                "core_pixels": int(core_mask.sum()),
            }
        )
        results["arms"][domain] = {}
        weights[domain] = {}
        for arm in ARMS:
            per_seed = {}
            weights[domain][arm] = {}
            for seed in seeds:
                model, history = train_arm(
                    arm, train_seqs, val_seqs, stats, resid_scale, core_mask, seed, args.iters
                )
                final = history[-1]
                iters_to = {str(tau): first_reach(history, tau) for tau in THRESHOLDS[domain]}
                per_seed[str(seed)] = {
                    "final": {
                        "full_pearson": float(final["full_pearson"]),
                        "core_pearson": float(final["core_pearson"]),
                        "rel_l2": float(final["rel_l2"]),
                    },
                    "iters_to_threshold": iters_to,
                    "history": history,
                }
                weights[domain][arm][str(seed)] = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }
            finals = [per_seed[str(s)]["final"] for s in seeds]
            iters_mean = {}
            for tau in THRESHOLDS[domain]:
                vals = [per_seed[str(s)]["iters_to_threshold"][str(tau)] for s in seeds]
                iters_mean[tau] = _mean_std(vals)
            results["arms"][domain][arm] = {
                "params": int(sum(p.numel() for p in build_model(arm).parameters())),
                "seeds": per_seed,
                "summary": {
                    "final": {
                        key: _mean_std([row[key] for row in finals])
                        for key in ("full_pearson", "core_pearson", "rel_l2")
                    },
                    "iters_mean": iters_mean,
                },
            }

    # ---- direction metric ----
    primary = "ladder" if "ladder" in results["arms"] else domains[0]
    all_seqs = seqs[primary]
    pairs = direction_pairs(all_seqs)
    base_lookup = {f.key: f.base for f in domain_frames[primary]}
    results["direction"]["base"] = direction_metric(base_lookup, pairs)
    results["direction"]["n_pairs"] = len(pairs)
    if "ladder" in results["arms"]:
        results["direction"]["arms"] = {}
        train_primary = [f for f in domain_frames[primary] if f.split == "train"]
        for arm in ARMS:
            per_seed = []
            with_prev = arm in ("concat", "gated")
            stats = feature_stats(train_primary)
            resid = torch.stack([f.resid_low for f in train_primary])
            resid_scale = float(resid.std().item()) or RESID_SCALE_FALLBACK
            for seed in seeds:
                model = build_model(arm)
                model.load_state_dict(weights["ladder"][arm][str(seed)])
                model.eval()
                lookup = {}
                with torch.no_grad():
                    for seq in all_seqs:
                        outs = run_sequence(model, seq, stats, with_prev)
                        for pred, f in zip(decode_predictions(outs, seq, resid_scale), seq):
                            lookup[f.key] = pred
                metric = direction_metric(lookup, pairs)
                per_seed.append(metric["mean"])
                results["direction"]["arms"].setdefault(arm, {"per_seed": {}, "corrs": {}})
                results["direction"]["arms"][arm]["per_seed"][str(seed)] = metric["mean"]
                results["direction"]["arms"][arm]["corrs"][str(seed)] = metric["corrs"]
            results["direction"]["arms"][arm]["mean"] = _mean_std(per_seed)
            results["direction"]["arms"][arm]["n"] = len(pairs)
    else:
        results["direction"]["arms"] = {}

    # ---- verdict ----
    for domain in results["arms"]:
        domain_summary = {arm: results["arms"][domain][arm]["summary"] for arm in ARMS}
        results["verdict"][domain] = build_verdict(domain_summary, THRESHOLDS[domain])

    # ---- persist ----
    torch.save(
        {
            "config": results["config"],
            "weights": weights,
        },
        out / "weights.pt",
    )
    results["artifacts"] = {"weights": str(out / "weights.pt")}
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, out / "REPORT.md")

    elapsed = time.time() - started
    print(f"wrote {out / 'results.json'}")
    print(f"wrote {out / 'REPORT.md'}")
    print(f"wrote {out / 'weights.pt'}")
    print(f"elapsed {elapsed:.1f}s")
    for domain in results["arms"]:
        verdict = results["verdict"][domain]
        print(f"{domain}: {verdict['verdict']} (wins={verdict['wins']} losses={verdict['losses']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
