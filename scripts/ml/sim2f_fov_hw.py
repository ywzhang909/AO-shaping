"""On-hardware small-ROI open-loop check (idea 1, bench script).

What it does, in order:

1. load a fitted bench state (``--calibration``); in ``--dry-run`` the model is
   the *nominal* twin, because the synthetic device is not the device that
   calibration describes;
2. open the camera on a **deliberately small ROI** (``--ccd``, centred on the
   0-order by :class:`~ml.sim2f.bench.HwInstrument`), so a shaped spot that is
   too large for the sensor is only partly visible;
3. probe the current optical path from those window pixels only
   (:func:`ml.sim2f.fit.probe`), yielding a residual aberration estimate;
4. build / load a commanded phase for ``--target`` (or read ``--phase``), write
   it to the SLM once, and take **one** windowed frame;
5. predict the **whole** sensor in that single open-loop shot and save it
   beside the window frame;
6. if a wide ROI is available (``--wide-ccd``, a second camera open), capture the
   true wide frame and score the prediction with
   :func:`ml.sim2f.extrapolate.extrapolation_metrics`.

``--dry-run`` replaces the SLM+camera with :class:`~ml.sim2f.bench.SimInstrument`
whose bench carries unmodelled :class:`~ml.sim2f.config.OpticsErrors`; the
window is then a crop of its full frame, so the identical code path (fit -> one
shot -> full-frame prediction -> score) runs without hardware.  A dry run is a
*plumbing* check only: the device is synthetic, so its scores say nothing about
the real bench.

The ROI is assumed to be centred on the 0-order and the model's 0-order to sit
at the frame centre, which is what makes ``CameraWindow.centered`` the right
correspondence; the script cross-correlates the measured window with the
prediction and records the peak offset so a mis-centring shows up as evidence
rather than as a silent bias.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.sim2f import fit
from ml.sim2f.calibrate import CalibratedBench, load_calibration
from ml.sim2f.config import AberrationConfig, OpticsErrors, Sim2fConfig
from ml.sim2f.extrapolate import extrapolation_metrics, frame_np
from ml.sim2f.finetune import PhaseRefiner, RefineConfig
from ml.sim2f.optics import Sim2f
from ml.sim2f.window import CameraWindow, WindowedBench

TARGETS = ("ring", "cross", "frame", "two_spot", "two_spot_wide")


def build_target(kind: str, ccd_px: int) -> np.ndarray:
    """A binary target mask for the requested family.

    Args:
        kind: One of :data:`TARGETS`.
        ccd_px: Frame size, pixels.

    Returns:
        ``(ccd_px, ccd_px)`` float32 mask, unit sum.

    Raises:
        ValueError: For an unknown ``kind``.
    """
    yy, xx = np.mgrid[0:ccd_px, 0:ccd_px].astype(np.float64)
    cx = cy = (ccd_px - 1) / 2.0
    r = np.hypot(xx - cx, yy - cy)
    out = np.zeros((ccd_px, ccd_px), dtype=np.float64)
    if kind == "ring":
        out[(r > 0.30 * ccd_px) & (r < 0.42 * ccd_px)] = 1.0
    elif kind == "cross":
        out[(np.abs(xx - cx) < 0.06 * ccd_px) & (r < 0.40 * ccd_px)] = 1.0
        out[(np.abs(yy - cy) < 0.06 * ccd_px) & (r < 0.40 * ccd_px)] = 1.0
    elif kind == "frame":
        out[
            (np.abs(np.maximum(np.abs(xx - cx), np.abs(yy - cy))) > 0.26 * ccd_px)
            & (np.abs(np.maximum(np.abs(xx - cx), np.abs(yy - cy))) < 0.36 * ccd_px)
        ] = 1.0
    elif kind in ("two_spot", "two_spot_wide"):
        # one spot at the 0-order, one deliberately at the small-ROI edge
        half = 0.5 * ccd_px
        sep = 0.42 * ccd_px if kind == "two_spot" else 0.46 * ccd_px
        for sx in (cx - sep, cx + sep):
            out[np.hypot(xx - sx, yy - cy) < 0.07 * ccd_px] = 1.0
    else:
        raise ValueError(f"unknown target {kind!r}; choose from {TARGETS}")
    total = out.sum()
    if total <= 0:
        raise ValueError(f"target {kind!r} is empty at ccd_px={ccd_px}")
    return (out / total).astype(np.float32)


@dataclass
class CaptureSource:
    """A camera view that can be read at the small ROI and, optionally, wide.

    Attributes:
        window_px: Small-ROI size, pixels.
        full_px: Full sensor size, pixels.
        label: ``"dry-run"`` or ``"hardware"``.
    """

    window_px: int
    full_px: int
    label: str

    def capture_windowed(
        self, phase: np.ndarray
    ) -> np.ndarray:  # pragma: no cover - abstract
        """One frame at the small ROI."""
        raise NotImplementedError

    def capture_full(
        self, phase: np.ndarray
    ) -> np.ndarray | None:  # pragma: no cover - abstract
        """One frame over the whole sensor, or ``None`` when unavailable."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the device."""


class DryRunSource(CaptureSource):
    """A :class:`~ml.sim2f.bench.SimInstrument` behind a crop."""

    def __init__(self, bench: Sim2f, window: CameraWindow, *, seed: int = 0) -> None:
        """Build the simulated device.

        Args:
            bench: Device bench (carries the unmodelled errors).
            window: The small ROI, in the device frame.
            seed: Aberration draw seed.
        """
        from ml.sim2f.bench import SimInstrument

        super().__init__(
            window_px=window.width, full_px=int(bench.ccd_px), label="dry-run"
        )
        self._instrument = SimInstrument(bench=bench, seed=seed)
        self.window = window

    def capture_windowed(self, phase: np.ndarray) -> np.ndarray:
        """Capture one windowed frame."""
        return np.asarray(
            self.window.crop(self._instrument.capture(phase)), dtype=np.float64
        )

    def capture_full(self, phase: np.ndarray) -> np.ndarray:
        """Capture one full frame (the dry run can always see it)."""
        return np.asarray(self._instrument.capture(phase), dtype=np.float64)

    def close(self) -> None:
        """Release the simulated instrument."""
        self._instrument.close()


class HardwareSource(CaptureSource):
    """A real SLM + camera at a small ROI, with an optional wide check."""

    def __init__(
        self,
        *,
        window_px: int,
        full_px: int,
        exposure_ms: float | None,
        wide: bool,
    ) -> None:
        """Open the devices.

        Args:
            window_px: Small-ROI size for the probe and the shot.
            full_px: Wide-ROI size used for the verification capture.
            exposure_ms: Explicit exposure, or ``None`` for the ladder.
            wide: Whether to open a second instrument for the wide check.
        """
        from ml.sim2f.bench import HwInstrument

        super().__init__(window_px=window_px, full_px=full_px, label="hardware")
        self._small = HwInstrument(ccd_px=window_px, exposure_ms=exposure_ms)
        self._wide = (
            HwInstrument(ccd_px=full_px, exposure_ms=exposure_ms) if wide else None
        )

    def capture_windowed(self, phase: np.ndarray) -> np.ndarray:
        """Capture one frame at the small ROI."""
        return np.asarray(self._small.capture(phase), dtype=np.float64)

    def capture_full(self, phase: np.ndarray) -> np.ndarray | None:
        """Capture one frame at the wide ROI when a second camera was opened."""
        if self._wide is None:
            return None
        return np.asarray(self._wide.capture(phase), dtype=np.float64)

    def close(self) -> None:
        """Close every opened device."""
        for inst in (self._small, self._wide):
            if inst is not None:
                inst.close()


def _load_model(args: argparse.Namespace) -> tuple[Any, Sim2fConfig, dict[str, Any]]:
    """Resolve the model bench.

    With ``--calibration`` the fitted state is applied to a fresh bench; in a
    dry run the model is the *nominal* twin, because the synthetic device is not
    the device that calibration describes.
    """
    if args.dry_run:
        cfg = Sim2fConfig(grid=args.grid, ccd_px=args.wide_ccd)
        return Sim2f(cfg), cfg, {"source": "nominal-twin", "dry_run": True}
    if args.calibration:
        result = load_calibration(args.calibration)
        sim = Sim2f(result.sim_config)
        bench = CalibratedBench(sim, result)
        info = {
            "source": "calibration",
            "calibration": str(args.calibration),
            "ok": bool(result.ok),
            "reason": result.reason,
        }
        return bench, result.sim_config, info
    raise SystemExit(
        "no --calibration given: run scripts/ml/calibrate_sim2f*.py first "
        "(a nominal twin cannot be applied to the device)"
    )


def main(argv: list[str] | None = None) -> int:
    """Run one small-ROI open-loop check.

    Args:
        argv: Command-line arguments (``None`` uses ``sys.argv``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="artifact directory")
    parser.add_argument(
        "--calibration", type=Path, default=None, help="calibration dir"
    )
    parser.add_argument("--phase", type=Path, default=None, help="commanded phase .npy")
    parser.add_argument("--target", choices=TARGETS, default="two_spot")
    parser.add_argument("--ccd", type=int, default=96, help="small ROI, pixels")
    parser.add_argument("--wide-ccd", type=int, default=256, help="wide ROI, pixels")
    parser.add_argument("--grid", type=int, default=128, help="model grid (dry run)")
    parser.add_argument("--fit-modes", type=int, default=6, help="free Noll modes 4..")
    parser.add_argument("--exposure-ms", type=float, default=None)
    parser.add_argument("--iters", type=int, default=150, help="phase refinement steps")
    parser.add_argument("--skip-wide", action="store_true", help="no wide verification")
    parser.add_argument(
        "--dry-run", action="store_true", help="simulated device, no hardware"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    bench, sim_cfg, model_info = _load_model(args)
    full_px = int(sim_cfg.ccd_px)
    window = CameraWindow.centered(args.ccd, full_px)
    wbench = WindowedBench(bench, window)

    if args.dry_run:
        device_cfg = Sim2fConfig(
            grid=sim_cfg.grid,
            ccd_px=full_px,
            errors=OpticsErrors(
                slm_zernike=((4, 0.35), (6, -0.28), (9, 0.22), (13, 0.18)),
                slm_roughness_sigma_rad=0.05,
                slm_roughness_seed=args.seed,
            ),
        )
        source: CaptureSource = DryRunSource(Sim2f(device_cfg), window, seed=args.seed)
    else:
        source = HardwareSource(
            window_px=args.ccd,
            full_px=args.wide_ccd,
            exposure_ms=args.exposure_ms,
            wide=not args.skip_wide,
        )

    try:
        # 1) probe the current path from the window only
        aberr_cfg = AberrationConfig(
            n_modes=args.fit_modes,
            probe_amplitude_rad=0.4,
        )

        def measure(p: Any) -> torch.Tensor:
            """Windowed camera read, as the fit machinery expects it."""
            return torch.as_tensor(source.capture_windowed(p), dtype=torch.float32)

        result = fit.probe(wbench, aberr_cfg, measure=measure)
        fitted = wbench.aberration(np.asarray(result.coeffs, dtype=np.float32)[None])
        print(
            f"[probe] ok={result.ok} modes={result.free_noll} "
            f"residual={result.residual:.4f} rank={result.rank} cond={result.condition:.1f}"
        )

        # 2) commanded phase: given, or refined against the calibrated model
        if args.phase is not None:
            phase = np.asarray(np.load(args.phase), dtype=np.float32)
            if phase.shape != (int(sim_cfg.grid), int(sim_cfg.grid)):
                raise ValueError(
                    f"phase {phase.shape} does not match grid {sim_cfg.grid}"
                )
            phase_meta = {"source": str(args.phase)}
        else:
            target = build_target(args.target, full_px)
            refiner = PhaseRefiner(bench, RefineConfig())
            init = np.zeros((int(sim_cfg.grid), int(sim_cfg.grid)), dtype=np.float32)
            refined = refiner.finetune(init, target, aberration=fitted)
            phase = np.asarray(refined.phase, dtype=np.float32)
            phase_meta = {
                "source": "refined",
                "target": args.target,
                "accepted": bool(refined.accepted),
                "objective": float(refined.objective),
            }

        # 3) one command, one windowed shot, one whole-sensor prediction
        frame_win = source.capture_windowed(phase)
        pred_full = frame_np(
            bench.forward(phase[None], aberration=fitted), normalise=False
        )
        pred_full = pred_full / max(float(pred_full.sum()), 1e-12)

        # 4) wide verification, when the hardware can give it
        frame_full = source.capture_full(phase)
        metrics: dict[str, float] | None = None
        offset: dict[str, float] | None = None
        if frame_full is not None:
            frame_full = frame_np(frame_full, normalise=False)
            frame_full = frame_full / max(float(frame_full.sum()), 1e-12)
            metrics = extrapolation_metrics(pred_full, frame_full, window).as_dict()
            offset = _peak_offset(pred_full, frame_full, window)

        artifacts = {
            "model": model_info,
            "window": {
                "center": list(window.center),
                "size": list(window.size),
                "frame": list(window.frame or (full_px, full_px)),
                "coverage": window.coverage((full_px, full_px)),
            },
            "probe": {
                "ok": bool(result.ok),
                "reason": result.reason,
                "free_noll": list(result.free_noll),
                "coeffs": [float(v) for v in result.coeffs],
                "residual": float(result.residual),
                "rank": int(result.rank),
                "condition": float(result.condition),
            },
            "phase": phase_meta,
            "mode": source.label,
            "metrics": metrics,
            "window_pred_offset_px": offset,
            "stamp": time.strftime("%Y%m%d_%H%M%S"),
        }
        np.save(out / "phase.npy", phase)
        np.save(out / "frame_window.npy", frame_win)
        np.save(out / "pred_full.npy", pred_full)
        if frame_full is not None:
            np.save(out / "frame_full.npy", frame_full)
        (out / "result.json").write_text(
            json.dumps(artifacts, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _save_figure(out / "fig_hw_check.png", frame_win, pred_full, frame_full, window)
        print(
            json.dumps(
                {k: artifacts[k] for k in ("metrics", "window_pred_offset_px")},
                indent=2,
            )
        )
        return 0 if result.ok else 2
    finally:
        source.close()


def _peak_offset(
    pred_full: np.ndarray, truth_full: np.ndarray, window: CameraWindow
) -> dict[str, float]:
    """Centroid offset between prediction and window-measured frame.

    Args:
        pred_full: Predicted whole-sensor frame.
        truth_full: Wide measured frame.
        window: The small ROI (used only for reporting its coverage).

    Returns:
        ``{"dx": ..., "dy": ...}`` in pixels, from the wide frames.
    """
    from ml.sim2f.extrapolate import moment_metrics

    p = moment_metrics(pred_full)
    t = moment_metrics(truth_full)
    return {"dx": float(p["cx"] - t["cx"]), "dy": float(p["cy"] - t["cy"])}


def _save_figure(
    path: Path,
    frame_win: np.ndarray,
    pred_full: np.ndarray,
    frame_full: np.ndarray | None,
    window: CameraWindow,
) -> None:
    """Save a compact 1x3 evidence figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:  # pragma: no cover - plotting is optional
        return
    panels = [("measured ROI", frame_win)]
    panels.append(("predicted (full, open loop)", pred_full))
    if frame_full is not None:
        panels.append(("measured (wide ROI)", frame_full))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8))
    for ax, (title, img) in zip(np.atleast_1d(axes), panels):
        vmax = float(img.max()) if float(img.max()) > 0 else 1.0
        ax.imshow(img, cmap="inferno", vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if img.shape[-1] == int(window.frame[0]):  # type: ignore[index]
            x0, y0 = window.offset()
            ax.add_patch(
                Rectangle(
                    (x0, y0), window.width, window.height, fill=False, ec="cyan", lw=1.2
                )
            )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
