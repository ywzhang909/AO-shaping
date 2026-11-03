"""Local measured forward model and differentiable shaping, 2026-11-03.

The model predicts a full 256px CCD intensity window from smooth phase
coefficients, using central differences measured through the actual panel path.
Independent mixed commands validate the model before optimisation. Every update
is bounded relative to the accepted phase and checked by A-B-B-A captures; the
final result is checked again in five independent A-B-B-A blocks. Historical
frames are never used as acceptance evidence. Delete only after this experiment
is superseded and its report no longer needs reproduction.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ml.sim2f.bench import HwInstrument, SimInstrument
from ml.sim2f.config import Sim2fConfig
from ml.sim2f.losses import objective_terms
from ml.sim2f.optics import Sim2f
from scripts.ml.sim2f_fov_hw_refine import _target_mask


def phase_basis(grid: int, order: int) -> np.ndarray:
    """Smooth cosine modes, bounded by one, without the unobservable piston."""
    axis = (np.arange(grid, dtype=np.float64) + 0.5) / grid
    modes = [
        np.cos(np.pi * y * axis[:, None]) * np.cos(np.pi * x * axis[None, :])
        for y in range(order + 1)
        for x in range(order + 1)
        if 0 < x + y <= order
    ]
    return np.asarray(modes, dtype=np.float32)


class LocalForwardModel:
    """Measured first order intensity response with explicit phase coordinates."""

    def __init__(self, reference: np.ndarray, jacobian: np.ndarray) -> None:
        if jacobian.ndim != 3 or reference.shape != jacobian.shape[1:]:
            raise ValueError("reference and response image shapes differ")
        if not np.isfinite(reference).all() or not np.isfinite(jacobian).all():
            raise ValueError("nonfinite measured response")
        self.reference = torch.as_tensor(reference, dtype=torch.float32)
        self.jacobian = torch.as_tensor(jacobian, dtype=torch.float32)

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Predict counts; clamp negative intensities without renormalising."""
        return (
            self.reference + torch.einsum("k,khw->hw", coefficients, self.jacobian)
        ).clamp_min(0)

    def optimise(
        self, target: np.ndarray, support: np.ndarray, basis: np.ndarray, radius: float
    ) -> tuple[np.ndarray, list[float]]:
        """Differentiate the shared shaping objective through measured images."""
        coefficients = torch.zeros(len(basis), requires_grad=True)
        modes = torch.as_tensor(basis)
        wanted = torch.as_tensor(target)
        mask = torch.as_tensor(support)
        optimiser = torch.optim.Adam([coefficients], lr=0.015)
        history = []
        best_loss = float("inf")
        best = np.zeros(len(basis), dtype=np.float32)
        for _ in range(120):
            optimiser.zero_grad()
            image = self.forward(coefficients)
            loss = objective_terms(image, wanted, mask, w_ee=0.25)[0]
            loss.backward()
            value = float(loss.detach())
            history.append(value)
            if value < best_loss:
                best_loss = value
                best = coefficients.detach().numpy().copy()
            optimiser.step()
            with torch.no_grad():
                delta = torch.einsum("k,khw->hw", coefficients, modes)
                maximum = float(delta.abs().max())
                if maximum > radius:
                    coefficients.mul_(radius / maximum)
        return best, history


def metrics(
    frame: np.ndarray, target: np.ndarray, support: np.ndarray
) -> dict[str, float]:
    """Shared distribution objective and centroid as a shape safeguard."""
    image = torch.as_tensor(frame, dtype=torch.float32)
    with torch.no_grad():
        values = objective_terms(
            image, torch.as_tensor(target), torch.as_tensor(support), w_ee=0.25
        )
    signal = np.maximum(np.asarray(frame, dtype=np.float64) - np.median(frame), 0)
    yy, xx = np.indices(signal.shape)
    total = max(float(signal.sum()), 1e-12)
    center = (signal.shape[0] - 1) / 2
    return {
        **dict(zip(("objective", "l1", "ee", "ncc"), map(float, values), strict=True)),
        "centroid_error_px": float(
            np.hypot(
                (xx * signal).sum() / total - center,
                (yy * signal).sum() / total - center,
            )
        ),
        "peak_counts": float(np.max(frame)),
    }


def run(args: argparse.Namespace) -> dict:
    """Acquire response images, optimise, accept measurements, then replay."""
    torch.set_num_threads(4)
    out = args.out.resolve()
    if not args.dry_run:
        out.relative_to(ROOT / "data" / "debug")
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    (out / "phases").mkdir()
    (out / "panel_gray").mkdir()
    basis = phase_basis(128, args.order)
    np.save(out / "basis.npy", basis)
    target, support = _target_mask(args.target, 256)
    np.save(out / "target.npy", target)
    np.save(out / "support.npy", support)
    start = np.load(args.phase).astype(np.float32)
    if start.shape != (128, 128) or not np.isfinite(start).all():
        raise ValueError("starting phase must be finite 128x128 radians")
    instrument = (
        SimInstrument(
            bench=Sim2f(Sim2fConfig.calibrated(), device="cpu"), detector=False
        )
        if args.dry_run
        else HwInstrument(
            exposure_ms=args.exposure_ms,
            pupil_center_xy=tuple(args.pupil_center),
            n_sample=args.samples,
        )
    )
    summary = {
        "schema": "local_forward_shaping_v1",
        "backend": "sim" if args.dry_run else "hw",
        "utc": datetime.now(timezone.utc).isoformat(),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "model": "CCD intensity Jacobian in smooth phase coordinates; central differences",
        "rounds": [],
        "captures": [],
        "replay": [],
        "status": "running",
    }
    entry_slot = None
    current = start.copy()

    def save() -> None:
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )

    try:
        if not args.dry_run:
            from ao_shaping.drivers.slm.factory import require_drive_board_entry

            require_drive_board_entry(instrument._slm)
            entry_slot = instrument._slm.get_displayed_memory_number()
        instrument.enable_full_frame()
        summary["device"] = instrument.describe()
        # Freeze one native window for all probes, updates and replays.
        # No per-phase registration and no use of the currently displayed phase.
        flat = instrument.capture_full_frame(np.zeros_like(start))
        np.save(out / "frames" / "flat.npy", flat)
        if args.dry_run:
            cx, cy = instrument.zero_order_xy
        else:
            from scipy.ndimage import gaussian_filter

            cy, cx = np.unravel_index(gaussian_filter(flat, 2).argmax(), flat.shape)
        summary["fixed_center_xy"] = [int(cx), int(cy)]
        save()

        def capture(phase: np.ndarray, name: str) -> np.ndarray:
            frame = instrument.capture_full_frame(phase)
            crop = frame[cy - 128 : cy + 128, cx - 128 : cx + 128].copy()
            if crop.shape != (256, 256) or not np.isfinite(crop).all():
                raise ValueError("invalid native capture window")
            if not args.dry_run and (crop.max() < 10 or frame.max() >= 250):
                raise ValueError("capture has insufficient signal or saturation")
            np.save(out / "frames" / f"{name}.npy", frame)
            np.save(out / "phases" / f"{name}.npy", phase)
            if not args.dry_run:
                np.save(
                    out / "panel_gray" / f"{name}.npy", instrument.sent_grayscale(phase)
                )
            summary["captures"].append(
                {"name": name, "metrics": metrics(crop, target, support)}
            )
            save()
            return crop

        def abba(a: np.ndarray, b: np.ndarray, name: str) -> dict:
            shots = [
                capture(p, f"{name}_{label}")
                for label, p in (("a1", a), ("b1", b), ("b2", b), ("a2", a))
            ]
            a_metrics = metrics((shots[0] + shots[3]) / 2, target, support)
            b_metrics = metrics((shots[1] + shots[2]) / 2, target, support)
            individual = [metrics(f, target, support)["objective"] for f in shots]
            gain = a_metrics["objective"] - b_metrics["objective"]
            noise = max(
                abs(individual[0] - individual[3]), abs(individual[1] - individual[2])
            )
            accepted = (
                gain > max(args.min_gain, 3 * noise)
                and b_metrics["ee"] >= a_metrics["ee"]
                and b_metrics["centroid_error_px"] <= a_metrics["centroid_error_px"] + 1
            )
            return {
                "reference": a_metrics,
                "candidate": b_metrics,
                "gain": gain,
                "noise": noise,
                "accepted": bool(accepted),
            }

        capture(start, "initial")
        for r in range(args.rounds):
            prefix = f"r{r:02d}"
            before = capture(current, f"{prefix}_reference_before")
            responses = []
            for k, mode in enumerate(basis):
                plus = capture(current + args.probe * mode, f"{prefix}_p{k:02d}")
                minus = capture(current - args.probe * mode, f"{prefix}_m{k:02d}")
                responses.append((plus - minus) / (2 * args.probe))
            after = capture(current, f"{prefix}_reference_after")
            jacobian = np.asarray(responses)
            model = LocalForwardModel((before + after) / 2, jacobian)
            np.save(out / f"{prefix}_jacobian.npy", jacobian)
            coefficients, history = model.optimise(target, support, basis, args.radius)
            np.save(out / f"{prefix}_coefficients.npy", coefficients)
            delta = np.einsum("k,khw->hw", coefficients, basis)
            record = {
                "round": r,
                "history": history,
                "delta_max_rad": float(abs(delta).max()),
                "trials": [],
            }
            summary["rounds"].append(record)
            # Mixed direction not used to estimate the response model.
            heldout_delta = delta * 0.25
            validation = capture(current + heldout_delta, f"{prefix}_validation")
            prediction = model.forward(torch.as_tensor(coefficients * 0.25)).numpy()
            record["validation"] = {
                "predicted_gain": metrics((before + after) / 2, target, support)[
                    "objective"
                ]
                - metrics(prediction, target, support)["objective"],
                "measured_gain": metrics((before + after) / 2, target, support)[
                    "objective"
                ]
                - metrics(validation, target, support)["objective"],
                "ncc": float(np.corrcoef(validation.ravel(), prediction.ravel())[0, 1]),
            }
            for scale in (1.0, 0.5, 0.25):
                candidate = current + scale * delta
                trial = abba(current, candidate, f"{prefix}_s{scale}")
                trial["scale"] = scale
                record["trials"].append(trial)
                logger.info(
                    "Round {} scale {} gain {:.6f}, noise {:.6f}, accepted {}",
                    r,
                    scale,
                    trial["gain"],
                    trial["noise"],
                    trial["accepted"],
                )
                if trial["accepted"]:
                    current = candidate.copy()
                    break
            np.save(out / "accepted_phase.npy", current)
            save()
        for i in range(args.replays):
            summary["replay"].append(abba(start, current, f"replay{i:02d}"))
            save()
        capture(current, "final")
        summary["status"] = "complete"
        save()
        return summary
    finally:
        try:
            if entry_slot is not None:
                instrument._slm._display_memory(entry_slot)
                summary["restored_entry_slot"] = int(entry_slot)
        finally:
            instrument.close()
            summary["devices_closed"] = True
            save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument(
        "--target", choices=("ring", "cross", "frame", "two_spot"), default="ring"
    )
    parser.add_argument("--pupil-center", type=int, nargs=2, default=(960, 720))
    parser.add_argument("--exposure-ms", type=float, default=0.6)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--probe", type=float, default=0.25)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--min-gain", type=float, default=0.003)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
