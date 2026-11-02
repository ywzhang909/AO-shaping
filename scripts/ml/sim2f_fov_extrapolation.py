"""Can a window-fitted forward model predict the far field the camera cannot see?

The claim under test (idea 1 of the FOV-generalisation task): a physics-anchored
forward model, identified from a **small CCD window**, predicts the whole sensor
in one open-loop shot -- including the light outside that window -- for spots too
large and too small for the sensor to frame.

The experiment is deliberately asymmetric, because ``device is model`` would
measure nothing:

* ``device`` -- a :class:`~ml.sim2f.optics.Sim2f` carrying a static SLM
  aberration (Noll 4/6/9/14) and panel roughness;
* ``model``  -- a second, ideal :class:`~ml.sim2f.optics.Sim2f` with the same
  geometry and no errors at all.

Only the model is ever fitted, and only through :class:`~ml.sim2f.window.WindowedBench`,
so every coefficient it learns comes from window pixels.  The identification shot
is taken at **zero command** and its coefficients are frozen for every test phase,
so each ``windowed_fit`` frame is a genuine single open-loop prediction.  The true
full frame exists only for scoring, and is never given to the fit.

Three whole-sensor predictions are scored against the device truth:

* ``nominal``      -- the uncorrected ideal model;
* ``windowed_fit`` -- the model plus the aberration fitted from the window;
* ``validated``    -- ``windowed_fit`` only if one independent confirmation shot
  agrees (see ``CONFIRM_RATIO``), else ``nominal``;
* ``oracle``       -- the model plus the best *in-basis* projection of the
  device's own static phase (a labelled upper bound, never a result).

Usage::

    python scripts/ml/sim2f_fov_extrapolation.py \
        --out report/shape_trials/fov_generalization --seed 0 --grid 128 --ccd 256
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ml.sim2f.config import AberrationConfig, OpticsErrors, Sim2fConfig
from ml.sim2f.extrapolate import (
    extrapolation_metrics,
    frame_np,
    moment_metrics,
    open_loop_frame,
)
from ml.sim2f.fit import probe
from ml.sim2f.optics import Sim2f
from ml.sim2f.window import CameraWindow, WindowedBench, crop_measure

#: Prediction arms, in figure/CSV order.
ARMS = ("nominal", "windowed_fit", "validated", "oracle")

#: Confirmation gate: accept the fit when its window-frame error is at most this
#: multiple of the *nominal* model's error on the same held-out shot.  Ratio, not
#: an absolute level, so the gate needs no calibrated constant and no oracle.
CONFIRM_RATIO = 1.0

#: Gate settings reported for sensitivity (0.5x stricter, 1.5x looser).
CONFIRM_SENSITIVITY = (0.5, 1.0, 1.5)

#: Fitted coefficient vector: Noll 2 .. N_MODES + 1.
N_MODES = 10

#: Noll modes the fit may move (tip/tilt stay frozen: the target fixes them).
FREE_NOLL = tuple(range(4, 12))

#: Case groups, in report order.
GROUPS = ("random", "oversized", "tiny", "defocus")

#: Every metric :func:`extrapolation_metrics` reports.
METRICS = (
    "l1_all",
    "l1_in",
    "l1_out",
    "ncc_all",
    "ncc_in",
    "ncc_out",
    "ee_in_true",
    "ee_in_pred",
    "ee_out_err",
    "centroid_err_px",
    "d86_err_px",
    "peak_ratio",
    "energy_total_ratio",
)


# ----------------------------------------------------------------------
# Benches and the oracle
# ----------------------------------------------------------------------
def build_benches(grid: int, ccd: int) -> tuple[Sim2f, Sim2f]:
    """Build the two *different* physics objects this experiment needs."""
    device = Sim2f(
        Sim2fConfig(
            grid=grid,
            ccd_px=ccd,
            errors=OpticsErrors(
                slm_zernike=((4, 0.40), (6, -0.32), (9, 0.26), (14, 0.18)),
                slm_roughness_sigma_rad=0.06,
                slm_roughness_seed=7,
            ),
        ),
        device="cpu",
    )
    model = Sim2f(Sim2fConfig(grid=grid, ccd_px=ccd), device="cpu")
    if device is model or device.errors == model.errors:
        raise RuntimeError("device and model must not share physics")
    return device, model


def in_basis_device_coeffs(device: Sim2f, model: Sim2f) -> tuple[np.ndarray, float]:
    """Best in-basis projection of the device's static phase, and its residual.

    The fit can only ever learn Noll ``2 .. N_MODES + 1`` on the *model's* Zernike
    support, so the strongest honest oracle is the least-squares projection of the
    device's own panel phase onto that basis.  Modes above the basis (Noll 14) and
    the roughness are left in the residual: the oracle is an upper bound for this
    model class, not a perfect correction.
    """
    panel = getattr(device, "_panel_phase", None)
    if panel is None:
        return np.zeros(N_MODES), 0.0
    basis = model.zernike_basis(N_MODES + 1).detach().cpu().numpy()
    mask = model.pupil_mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(bool)
    panel_np = panel.detach().cpu().numpy()
    coefficients, *_ = np.linalg.lstsq(basis[:, mask].T, panel_np[mask], rcond=None)
    residual = panel_np[mask] - (basis[:, mask] * coefficients[:, None]).sum(axis=0)
    # Row 0 is piston, which ``Sim2f.aberration`` does not accept.
    return coefficients[1:], float(np.sqrt(np.mean(residual**2)))


# ----------------------------------------------------------------------
# Test phases
# ----------------------------------------------------------------------
def _phase_from(
    basis: np.ndarray, coefficients: dict[int, float], grid: int
) -> np.ndarray:
    """Sum ``coefficient * Z_noll`` on the model basis."""
    phase = np.zeros((grid, grid), dtype=np.float64)
    for noll, value in coefficients.items():
        phase += float(value) * basis[noll - 1]
    return phase


def build_phases(
    model: Sim2f,
    device: Sim2f,
    seed: int,
    grid: int,
    main_window: CameraWindow,
) -> tuple[list[dict[str, Any]], float]:
    """Deterministically build the >= 8 test phases and classify them.

    Returns:
        ``(records, d86_floor_px)``.  Each record carries ``case``, ``group``,
        ``phase`` (``(grid, grid)`` radians), the device truth frame and its
        moments.  ``d86_floor_px`` is the model's diffraction-limited D86 -- the
        smallest spot this geometry can form.
    """
    basis = model.zernike_basis(16).detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []

    def add(group: str, case: str, phase: np.ndarray) -> None:
        truth = frame_np(device.forward(torch.as_tensor(phase)[None]))
        moments = moment_metrics(truth)
        inside = main_window.mask((truth.shape[1], truth.shape[0])).astype(bool)
        records.append(
            {
                "group": group,
                "case": case,
                "phase": phase,
                "truth": truth,
                "d86_px": moments["d86_px"],
                "ee_in_main": float(truth[inside].sum()),
            }
        )

    # (a) generic smooth shapes: Noll 4..15, 0.2 .. 1.2 rad, random sign.
    for index in range(2):
        amplitudes = rng.uniform(0.2, 1.2, size=12)
        signs = rng.choice((-1.0, 1.0), size=12)
        coefficients = {
            noll: float(amplitude * sign)
            for noll, amplitude, sign in zip(
                range(4, 16), amplitudes, signs, strict=True
            )
        }
        add("random", f"random_{index}", _phase_from(basis, coefficients, grid))

    # (b) oversized spots: pure defocus large enough to spread past the window.
    for index, amplitude in enumerate((8.0, 10.0)):
        add("oversized", f"oversized_{index}", _phase_from(basis, {4: amplitude}, grid))

    # (c) tiny spots: the diffraction-limited floor of this geometry.
    add("tiny", "tiny_0", np.zeros((grid, grid), dtype=np.float64))
    small = {
        noll: float(value)
        for noll, value in zip(range(4, 8), rng.normal(scale=0.02, size=4), strict=True)
    }
    add("tiny", "tiny_1", _phase_from(basis, small, grid))

    # (d) defocus sweep that pushes most of the light outside the window.
    for index, amplitude in enumerate((12.0, 14.0)):
        add("defocus", f"defocus_{index}", _phase_from(basis, {4: amplitude}, grid))

    floor = moment_metrics(model.forward(torch.zeros(1, grid, grid)))["d86_px"]

    # The labels are only honest if the frames really are what they claim.
    for record in records:
        if record["group"] == "oversized":
            if not (record["d86_px"] > 90.0 and record["ee_in_main"] < 0.55):
                raise RuntimeError(
                    f"{record['case']} is not oversized: d86={record['d86_px']:.1f}, "
                    f"window energy={record['ee_in_main']:.3f}"
                )
        if record["group"] == "tiny" and record["d86_px"] > 1.1 * floor:
            raise RuntimeError(
                f"{record['case']} is not at the diffraction floor: "
                f"d86={record['d86_px']:.2f} vs floor {floor:.2f}"
            )
        if record["group"] == "defocus" and record["ee_in_main"] >= 0.35:
            raise RuntimeError(
                f"{record['case']} does not push light out of the window: "
                f"window energy={record['ee_in_main']:.3f}"
            )
    return records, floor


def build_confirmation(
    model: Sim2f, seed: int, grid: int, phases: list[dict[str, Any]]
) -> tuple[np.ndarray, dict[int, float]]:
    """One held-out phase, drawn from its own stream and never scored as a case.

    The fit is trusted only if it predicts this shot's window frame better than
    the uncorrected model does, so the shot has to be *independent* of both the
    probe bank (zero command) and the eight scored cases: same mode range as the
    random group, different seed offset, and an explicit check that it matches no
    test phase.

    Returns:
        ``(phase, coefficients)``, both radians on the model grid.

    Raises:
        RuntimeError: If 100 draws all collide with a test phase (impossible in
            practice; a guard against a silently degenerate phase bank).
    """
    basis = model.zernike_basis(16).detach().cpu().numpy()
    rng = np.random.default_rng(seed + 1009)
    for _attempt in range(100):
        amplitudes = rng.uniform(0.3, 0.9, size=12)
        signs = rng.choice((-1.0, 1.0), size=12)
        coefficients = {
            noll: float(amplitude * sign)
            for noll, amplitude, sign in zip(
                range(4, 16), amplitudes, signs, strict=True
            )
        }
        phase = _phase_from(basis, coefficients, grid)
        if all(np.abs(phase - record["phase"]).max() > 1e-6 for record in phases):
            return phase, coefficients
    raise RuntimeError("could not draw a confirmation phase distinct from the cases")


def windows_for(ccd: int) -> list[tuple[str, CameraWindow]]:
    """The main window plus the two sensitivity windows."""
    centre = ccd // 2
    offset = centre - 20
    return [
        (
            f"96x96@{centre},{offset}",
            CameraWindow(center=(centre, offset), size=(96, 96), frame=(ccd, ccd)),
        ),
        (
            f"64x64@{centre},{offset}",
            CameraWindow(center=(centre, offset), size=(64, 64), frame=(ccd, ccd)),
        ),
        (
            f"96x96@{centre},{centre}",
            CameraWindow(center=(centre, centre), size=(96, 96), frame=(ccd, ccd)),
        ),
    ]


def aberration_cfg() -> AberrationConfig:
    """The probe/fit contract: eight free modes, Noll 4..11."""
    return AberrationConfig(
        n_modes=N_MODES,
        probe_modes=FREE_NOLL,
        probe_frozen=(1,),
        fit_frozen=(1, 2, 3),
        probe_amplitude_rad=0.35,
        fit_iters=80,
        fit_lr=0.05,
    )


# ----------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------
def run(
    device: Sim2f,
    model: Sim2f,
    oracle: np.ndarray,
    phases: list[dict[str, Any]],
    windows: list[tuple[str, CameraWindow]],
    confirm_phase: np.ndarray,
    ccd: int,
    grid: int,
) -> tuple[
    list[dict[str, Any]], dict[tuple[str, str, str], np.ndarray], dict[str, Any]
]:
    """Identify the device once per window, then score every phase open loop.

    The identification shot is taken at **zero command** (``phase=None``): the
    probe bank sees the static device aberration and nothing else, and the
    coefficients are frozen for every test phase afterwards.  That is the claim
    under test -- learn the instrument from a small window, then predict unseen
    frames in one shot.  Re-identifying per phase was rejected on physics
    grounds: the free basis (Noll 4..11) overlaps the commanded test phases, so
    one window shot cannot separate commanded from static aberration and the
    fit silently absorbs the command into the model.

    ``FitResult.ok`` is not a trust gate: it reports whether the probe bank had
    rank and the fit stayed finite, not whether the coefficients are the device's.
    So each window also gets one independent confirmation shot: predict its
    window frame from the fitted aberration, measure it, and keep the fit only if
    that frame is at least as good as the uncorrected model's (``CONFIRM_RATIO``).

    Returns:
        ``(trials, frames, validation)``.  ``frames`` keeps every window's truth
        and predictions for the case figure, keyed by ``(window, case, arm)``;
        ``validation`` is the per-window confirmation record.
    """
    cfg = aberration_cfg()
    trials: list[dict[str, Any]] = []
    frames: dict[tuple[str, str, str], np.ndarray] = {}
    validation: dict[str, Any] = {}

    confirm_cmd = torch.as_tensor(confirm_phase)[None]
    confirm_truth = frame_np(device.forward(confirm_cmd))

    for window_name, window in windows:
        wrapped = WindowedBench(model, window)
        measure = crop_measure(lambda phase: device.forward(phase), window)

        started = time.perf_counter()
        result = probe(wrapped, cfg, measure=measure, phase=None)
        fit_seconds = time.perf_counter() - started
        coefficients = result.coeffs if result.ok else np.zeros(N_MODES)
        coefficient_error = float(
            np.sqrt(np.mean((coefficients[2:] - oracle[2:]) ** 2))
        )

        # How much light the identification shot itself puts in this window: the
        # one number that predicts an unidentifiable fit.
        mask = window.mask((ccd, ccd)).astype(bool)
        probe_frame = frame_np(device.forward(torch.zeros(1, grid, grid)))
        probe_ee_in = float(probe_frame[mask].sum() / probe_frame.sum())

        # Why a low identification residual does not certify the fit.  Negating
        # the pupil phase conjugates the field, which point-mirrors the far
        # field; on the centred, nearly symmetric identification shot the two
        # branches are hard to separate, and the fit's eight free modes also
        # absorb part of the out-of-basis residual (Noll 14 + roughness), so the
        # optimiser can pick the branch that explains *this* frame best.  Record
        # the opposite branch on the very shot the fit was given.
        ident_truth = probe_frame
        ident_zero = open_loop_frame(model, torch.zeros(1, grid, grid), full_ccd_px=ccd)
        ident_opposite = open_loop_frame(
            model,
            torch.zeros(1, grid, grid),
            full_ccd_px=ccd,
            aberration=wrapped.aberration((-coefficients)[None]),
        )
        ident_fitted = open_loop_frame(
            model,
            torch.zeros(1, grid, grid),
            full_ccd_px=ccd,
            aberration=wrapped.aberration(coefficients[None]),
        )
        ident_l1 = {
            "fit": float(
                extrapolation_metrics(ident_fitted, ident_truth, window).l1_in
            ),
            "opposite": float(
                extrapolation_metrics(ident_opposite, ident_truth, window).l1_in
            ),
            "zero": float(extrapolation_metrics(ident_zero, ident_truth, window).l1_in),
        }

        # Held-out confirmation: one shot, never one of the scored cases.
        confirm_nominal = open_loop_frame(model, confirm_cmd, full_ccd_px=ccd)
        confirm_fitted = open_loop_frame(
            model,
            confirm_cmd,
            full_ccd_px=ccd,
            aberration=wrapped.aberration(coefficients[None]),
        )
        confirm_oracle = open_loop_frame(
            model,
            confirm_cmd,
            full_ccd_px=ccd,
            aberration=wrapped.aberration(oracle[None]),
        )
        confirm_l1 = extrapolation_metrics(confirm_fitted, confirm_truth, window).l1_in
        confirm_l1_nominal = extrapolation_metrics(
            confirm_nominal, confirm_truth, window
        ).l1_in
        confirm_l1_oracle = extrapolation_metrics(
            confirm_oracle, confirm_truth, window
        ).l1_in
        confirm_ratio = (
            float(confirm_l1 / confirm_l1_nominal)
            if confirm_l1_nominal and np.isfinite(confirm_l1_nominal)
            else float("inf")
        )
        accepted = bool(confirm_ratio <= CONFIRM_RATIO)
        sensitivity = {
            f"{ratio:g}x": bool(confirm_ratio <= ratio) for ratio in CONFIRM_SENSITIVITY
        }

        logger.info(
            "{} identify@0: ok={} residual {:.3f}->{:.3f} rank={} "
            "coef_rmse_vs_oracle={:.3f} probe_ee_in={:.3f} confirm_l1={:.4f} "
            "(nominal {:.4f}, oracle {:.4f}) ratio={:.2f} accepted={} ({:.1f}s)",
            window_name,
            result.ok,
            result.residual_start,
            result.residual,
            result.rank,
            coefficient_error,
            probe_ee_in,
            confirm_l1,
            confirm_l1_nominal,
            confirm_l1_oracle,
            confirm_ratio,
            accepted,
            fit_seconds,
        )
        fit = {
            "ok": bool(result.ok),
            "reason": result.reason,
            "identified_at": "zero_command",
            "scope": "per_window",
            "coeffs": [float(v) for v in coefficients],
            "free_noll": list(result.free_noll),
            "residual": float(result.residual),
            "residual_start": float(result.residual_start),
            "iterations": int(result.iterations),
            "rank": int(result.rank),
            "condition": float(result.condition),
            "coefficient_rmse_vs_oracle": coefficient_error,
            "probe_ee_in": probe_ee_in,
            "confirm_l1": float(confirm_l1),
            "confirm_l1_nominal": float(confirm_l1_nominal),
            "confirm_l1_oracle": float(confirm_l1_oracle),
            "confirm_ratio": confirm_ratio,
            "confirm_accepted": accepted,
            "confirm_sensitivity": sensitivity,
            "ident_shot_l1": ident_l1,
            "seconds": float(fit_seconds),
        }
        validation[window_name] = fit

        for record in phases:
            command = torch.as_tensor(record["phase"])[None]

            fitted = open_loop_frame(
                model,
                command,
                full_ccd_px=ccd,
                aberration=wrapped.aberration(coefficients[None]),
            )
            nominal = open_loop_frame(model, command, full_ccd_px=ccd)
            predictions = {
                "nominal": nominal,
                "windowed_fit": fitted,
                # The gate's whole point: on reject the prediction is the
                # uncorrected model, i.e. what a cautious operator would use.
                "validated": fitted if accepted else nominal,
                "oracle": open_loop_frame(
                    model,
                    command,
                    full_ccd_px=ccd,
                    aberration=wrapped.aberration(oracle[None]),
                ),
            }
            frames[(window_name, record["case"], "truth")] = record["truth"]
            for arm in ARMS:
                frames[(window_name, record["case"], arm)] = predictions[arm]

            for arm in ARMS:
                metrics = extrapolation_metrics(
                    predictions[arm], record["truth"], window
                )
                trials.append(
                    {
                        "window": window_name,
                        "window_center": list(window.center),
                        "window_size": list(window.size),
                        "case": record["case"],
                        "case_group": record["group"],
                        "phase": record["case"],
                        "arm": arm,
                        # Did *this* prediction use the fitted aberration?
                        "fit_applied": arm == "windowed_fit"
                        or (arm == "validated" and accepted),
                        "confirm_l1": float(confirm_l1),
                        "confirm_ratio": confirm_ratio,
                        "metrics": metrics.as_dict(),
                        # Energy the truth puts outside: the denominator of l1_out.
                        "ee_out_true": 1.0 - metrics.ee_in_true,
                        "truth_d86_px": record["d86_px"],
                        # One identification per window: identical for every phase.
                        "fit": fit,
                    }
                )
    return trials, frames, validation


# ----------------------------------------------------------------------
# Aggregation and artifacts
# ----------------------------------------------------------------------
def aggregate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean +/- std of every metric per (window, case group, arm)."""
    buckets: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    for trial in trials:
        key = (trial["window"], trial["case_group"], trial["arm"])
        buckets.setdefault(key, []).append(trial["metrics"])
    summary: dict[str, Any] = {}
    for (window, group, arm), rows in sorted(buckets.items()):
        entry: dict[str, Any] = {"n_phases": len(rows)}
        for metric in METRICS:
            values = np.array([row[metric] for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            entry[metric] = {
                "mean": float(finite.mean()) if finite.size else None,
                "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
                "n": int(finite.size),
            }
        summary[f"{window}|{group}|{arm}"] = entry
    return summary


def _clean(value: Any) -> Any:
    """JSON-safe copy: non-finite floats become ``None``."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_artifacts(
    out: Path,
    trials: list[dict[str, Any]],
    summary: dict[str, Any],
    meta: dict[str, Any],
    commands: dict[str, str],
) -> Path:
    """Write ``artifacts/trials.json``, ``trials.csv`` and ``summary.json``."""
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    payload = {"meta": _clean(meta), "trials": _clean(trials)}
    (artifacts / "trials.json").write_text(
        json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8"
    )

    columns = (
        "window",
        "case",
        "phase",
        "arm",
        "fit_applied",
        "confirm_l1",
        "confirm_ratio",
        "l1_all",
        "l1_out",
        "l1_in",
        "ee_out_err",
        "centroid_err_px",
        "d86_err_px",
        "ncc_out",
    )
    with (artifacts / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for trial in trials:
            row: dict[str, Any] = {
                "window": trial["window"],
                "case": trial["case"],
                "phase": trial["phase"],
                "arm": trial["arm"],
                "fit_applied": trial["fit_applied"],
                "confirm_l1": f"{trial['confirm_l1']:.6g}",
                "confirm_ratio": f"{trial['confirm_ratio']:.6g}",
            }
            for metric in columns[7:]:
                value = trial["metrics"][metric]
                row[metric] = f"{value:.6g}" if math.isfinite(value) else ""
            writer.writerow(row)

    summary_payload = {
        "meta": _clean(meta),
        "commands": commands,
        "aggregate": _clean(summary),
    }
    (artifacts / "summary.json").write_text(
        json.dumps(summary_payload, indent=1, allow_nan=False), encoding="utf-8"
    )
    return artifacts


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def _draw_window(ax: Any, window: CameraWindow) -> None:
    """Mark the camera window on a full-frame panel."""
    x0, y0 = window.offset()
    ax.add_patch(
        Rectangle(
            (x0, y0),
            window.width,
            window.height,
            fill=False,
            edgecolor="cyan",
            linewidth=1.0,
        )
    )


def draw_cases(
    trials: list[dict[str, Any]],
    frames: dict[tuple[str, str, str], np.ndarray],
    window: CameraWindow,
    window_name: str,
    fit: dict[str, Any],
    out: Path,
) -> Path:
    """Truth, ``windowed_fit`` prediction and signed error for three cases.

    The window shown is the one with the lowest identification residual -- the
    only selection rule available without the oracle.  The three cases are the
    ones where that fit changed the whole-frame L1 the most.  The panel titles
    state whether the confirmation gate kept the fit, since a rejected fit is not
    a result anyone would deploy.
    """
    verdict = "fit applied" if fit["confirm_accepted"] else "FIT REJECTED"
    scored = {
        (trial["case"], trial["arm"]): trial
        for trial in trials
        if trial["window"] == window_name
    }
    deltas = []
    for case in sorted({case for _win, case, _arm in frames if _win == window_name}):
        nominal = scored[(case, "nominal")]["metrics"]["l1_all"]
        fitted = scored[(case, "windowed_fit")]["metrics"]["l1_all"]
        deltas.append((abs(nominal - fitted), case, nominal, fitted))
    chosen = sorted(deltas, reverse=True)[:3]

    fig, axes = plt.subplots(3, 3, figsize=(9.0, 8.6), constrained_layout=True)
    for row, (_delta, case, nominal, fitted) in enumerate(chosen):
        truth = frames[(window_name, case, "truth")]
        prediction = frames[(window_name, case, "windowed_fit")]
        peak = float(truth.max())
        norm = LogNorm(vmin=max(peak * 1e-4, 1e-12), vmax=peak)
        error = prediction - truth
        limit = float(np.abs(error).max())
        panels = (
            (truth, "inferno", norm, f"device truth\n{case}"),
            (
                prediction,
                "inferno",
                norm,
                f"windowed_fit prediction [{verdict}]\n"
                f"l1_all {nominal:.3f} -> {fitted:.3f}",
            ),
            (
                error / peak,
                "RdBu_r",
                {"vmin": -limit / peak, "vmax": limit / peak},
                "signed error\n(pred - truth) / peak",
            ),
        )
        for column, (image, cmap, scale, title) in enumerate(panels):
            ax = axes[row, column]
            if isinstance(scale, dict):
                ax.imshow(image, cmap=cmap, origin="lower", **scale)
            else:
                ax.imshow(image, cmap=cmap, origin="lower", norm=scale)
            ax.set_title(title, fontsize=8)
            ax.set_xticks(())
            ax.set_yticks(())
            _draw_window(ax, window)
    fig.suptitle(
        f"full-sensor prediction from the best-identified window {window_name} "
        f"(cyan): fit residual {fit['residual']:.4f}, confirmation gate "
        f"{'accepted' if fit['confirm_accepted'] else 'rejected'} "
        f"(confirm_l1 {fit['confirm_l1']:.4f} vs nominal "
        f"{fit['confirm_l1_nominal']:.4f})",
        fontsize=9,
    )
    path = out / "fig_fov_cases.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def draw_error_bars(
    trials: list[dict[str, Any]],
    windows: list[str],
    validation: dict[str, Any],
    out: Path,
) -> Path:
    """Per-window bars of ``l1_all``, ``l1_out`` and ``ee_out_err``."""
    metrics = ("l1_all", "l1_out", "ee_out_err")
    fig, axes = plt.subplots(
        len(metrics),
        len(windows),
        figsize=(3.5 * len(windows), 8.4),
        constrained_layout=True,
        sharex=True,
        squeeze=False,
    )
    width = 0.8 / len(ARMS)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(ARMS))
    positions = np.arange(len(GROUPS))
    for column, window_name in enumerate(windows):
        for row, metric in enumerate(metrics):
            ax = axes[row, column]
            for index, arm in enumerate(ARMS):
                means, stds = [], []
                for group in GROUPS:
                    values = [
                        trial["metrics"][metric]
                        for trial in trials
                        if trial["window"] == window_name
                        and trial["arm"] == arm
                        and trial["case_group"] == group
                    ]
                    finite = np.array([v for v in values if math.isfinite(v)])
                    means.append(float(finite.mean()) if finite.size else 0.0)
                    stds.append(float(finite.std(ddof=1)) if finite.size > 1 else 0.0)
                ax.bar(
                    positions + offsets[index],
                    means,
                    width=width,
                    yerr=stds,
                    capsize=2.0,
                    label=arm,
                    color=f"C{index}",
                )
            ax.set_ylabel(metric, fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            if metric != "ee_out_err":
                ax.set_yscale("log")
                ax.set_ylim(bottom=1e-3)
            if row == 0:
                verdict = (
                    "accepted" if validation[window_name]["accepted"] else "REJECTED"
                )
                ax.set_title(
                    f"{window_name}\nconfirmation gate {verdict} "
                    f"(confirm_l1 {validation[window_name]['confirm_l1']:.3f})",
                    fontsize=9,
                )
            if row == len(metrics) - 1:
                ax.set_xticks(positions)
                ax.set_xticklabels(GROUPS, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "error by case group, one column per window; validated = fitted model when "
        "the confirmation gate accepts, else nominal",
        fontsize=9,
    )
    path = out / "fig_fov_error.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="report/shape_trials/fov_generalization",
        help="artifact directory (relative paths resolve against the repo root)",
    )
    parser.add_argument("--seed", type=int, default=0, help="phase-bank seed")
    parser.add_argument("--grid", type=int, default=128, help="pupil grid side")
    parser.add_argument("--ccd", type=int, default=256, help="full sensor side, px")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the experiment and write every artifact under ``--out``."""
    args = _parser().parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    started = time.perf_counter()

    device, model = build_benches(args.grid, args.ccd)
    oracle, oracle_residual = in_basis_device_coeffs(device, model)
    windows = windows_for(args.ccd)
    phases, d86_floor = build_phases(model, device, args.seed, args.grid, windows[0][1])
    confirm_phase, confirm_coeffs = build_confirmation(
        model, args.seed, args.grid, phases
    )

    trials, frames, validation = run(
        device, model, oracle, phases, windows, confirm_phase, args.ccd, args.grid
    )
    summary = aggregate(trials)
    runtime = time.perf_counter() - started

    # The window the protocol can trust is the one whose own fit residual is
    # lowest: no oracle needed, so this is a fair selection rule.
    fits = {trial["window"]: trial["fit"] for trial in trials}
    best_name = min(fits, key=lambda name: fits[name]["residual"])
    others = [name for name in fits if name != best_name]
    oracle_free = np.asarray(oracle[2:], dtype=np.float64)
    identification: dict[str, dict[str, Any]] = {}
    for name, fit in fits.items():
        fitted = np.asarray(fit["coeffs"][2:], dtype=np.float64)
        identification[name] = {
            "fit_residual": fit["residual"],
            "coefficient_rmse_vs_oracle_rad": fit["coefficient_rmse_vs_oracle"],
            # +1: the fit found the static phase; -1: its mirror image.
            "oracle_projection_ratio": float(
                fitted @ oracle_free / float(oracle_free @ oracle_free)
            ),
            "ok": fit["ok"],
            "rank": fit["rank"],
            "condition": fit["condition"],
            "seconds": fit["seconds"],
        }
    best = identification[best_name]
    other = identification[others[0]]
    rand = {arm: summary[f"{best_name}|random|{arm}"] for arm in ARMS}

    # Accept/reject is per *identification*, so within a window it is all or
    # nothing: these counts say how many scored cases of each group the gate
    # would actually let through.
    validation_summary: dict[str, Any] = {}
    for name in fits:
        fit = fits[name]
        groups: dict[str, dict[str, int]] = {}
        for group in GROUPS:
            rows = [
                trial
                for trial in trials
                if trial["window"] == name
                and trial["case_group"] == group
                and trial["arm"] == "validated"
            ]
            groups[group] = {
                "n_cases": len(rows),
                "n_accepted": sum(1 for row in rows if row["fit_applied"]),
                "n_rejected": sum(1 for row in rows if not row["fit_applied"]),
            }
        validation_summary[name] = {
            "confirm_l1": fit["confirm_l1"],
            "confirm_l1_nominal": fit["confirm_l1_nominal"],
            "confirm_l1_oracle": fit["confirm_l1_oracle"],
            "confirm_ratio": fit["confirm_ratio"],
            "accepted": fit["confirm_accepted"],
            "sensitivity": fit["confirm_sensitivity"],
            "ident_shot_l1": fit["ident_shot_l1"],
            "probe_ee_in": fit["probe_ee_in"],
            "groups": groups,
        }
    best_val = validation_summary[best_name]
    other_val = validation_summary[others[0]]
    probe_ee = [entry["probe_ee_in"] for entry in validation_summary.values()]

    meta = {
        "grid": int(args.grid),
        "ccd_px": int(args.ccd),
        "seed": int(args.seed),
        "n_phases": len(phases),
        "case_groups": list(GROUPS),
        "arms": list(ARMS),
        "fitted_noll": list(FREE_NOLL),
        "protocol": (
            "one identification per window at zero command (the probe bank sees "
            "only the static device aberration); the coefficients are frozen for "
            "all test phases, so every windowed_fit frame is a single open-loop "
            "prediction"
        ),
        "windows": [
            {
                "name": name,
                "center": list(window.center),
                "size": list(window.size),
                "offset": list(window.offset()),
                "coverage": window.coverage((args.ccd, args.ccd)),
            }
            for name, window in windows
        ],
        "device": {
            "description": "Sim2f with static SLM aberration and roughness",
            "slm_zernike": [[4, 0.40], [6, -0.32], [9, 0.26], [14, 0.18]],
            "slm_roughness_sigma_rad": 0.06,
            "slm_roughness_seed": 7,
        },
        "model": {"description": "ideal Sim2f, same geometry, no errors"},
        "oracle": {
            "description": (
                "least-squares projection of the device panel phase onto the "
                "model's Noll 2..11 basis; a labelled upper bound, not a result"
            ),
            "coeffs_noll2_to_11": [float(v) for v in oracle],
            "projection_residual_rms_rad": oracle_residual,
        },
        "d86_floor_px": float(d86_floor),
        "runtime_s": float(runtime),
        "identification": identification,
        "best_window_by_fit_residual": best_name,
        "confirmation": {
            "description": (
                "one held-out shot, drawn from its own RNG stream and checked to "
                "match no scored case; the fit is kept only if its window-frame "
                "relative L1 on that shot is at most CONFIRM_RATIO times the "
                "uncorrected model's on the same shot"
            ),
            "coeffs_noll4_to_15": {str(k): v for k, v in confirm_coeffs.items()},
            "threshold_ratio": CONFIRM_RATIO,
            "sensitivity_ratios": list(CONFIRM_SENSITIVITY),
            "rationale": (
                "a ratio, not an absolute level, so the gate needs no calibrated "
                "constant and no oracle: it only asks whether the fitted model "
                "explains a shot it was not fitted on better than no correction"
            ),
            "windows": validation_summary,
        },
        "findings": [
            (
                "The requested 'tiny spot, D86 < 8 px' is below this geometry's "
                f"diffraction floor of {d86_floor:.2f} px (grid={args.grid}, "
                f"ccd={args.ccd}), so the tiny group uses the floor itself: the "
                "spot is still ~7x smaller than the 96 px window."
            ),
            (
                "l1_out divides by the truth energy outside the window; when the "
                "window captures ~all light (tiny group) that denominator is under "
                "1% and l1_out is unstable -- read l1_all and ee_out_err there."
            ),
            (
                "Noll 14 (0.18 rad) and the 0.06 rad roughness are outside the "
                "fitted basis; the oracle projection leaves "
                f"{oracle_residual:.3f} rad RMS unmodelled, so even the oracle is "
                "an upper bound for this model class."
            ),
            (
                "The device has no detector noise (exposure_ms=0), so the fit is "
                "an optimistic bound on what a real camera would allow."
            ),
            (
                "The fit is identified once per window at zero command and frozen. "
                "Re-identifying per phase was rejected: the free basis (Noll 4..11) "
                "overlaps the commanded test phases, so one window shot cannot "
                "separate commanded from static aberration and the fit absorbs the "
                "command, over-correcting a phase it already knows."
            ),
            (
                "Identification, not extrapolation, is the bottleneck. The "
                f"{best_name} window reaches a fit residual of {best['fit_residual']:.4f} "
                f"and a coefficient RMSE of {best['coefficient_rmse_vs_oracle_rad']:.3f} "
                f"rad against the oracle, while {others[0]} stops at "
                f"{other['coefficient_rmse_vs_oracle_rad']:.3f} rad with a projection "
                f"ratio of {other['oracle_projection_ratio']:+.2f}: a sign-flipped, "
                "over-scaled copy of the same phase. Negating a pupil phase "
                "conjugates the field, which point-mirrors the far field, so on the "
                "centred, nearly symmetric identification shot the branches are hard "
                "to tell apart -- and the fit's eight free modes also absorb part of "
                "the out-of-basis residual (Noll 14 + roughness). The consequence is "
                "that the frame prefers whichever branch the fit chose: on that same "
                f"shot the fit scores {other_val['ident_shot_l1']['fit']:.4f} window "
                f"L1 while its opposite branch scores "
                f"{other_val['ident_shot_l1']['opposite']:.4f} and zero aberration "
                f"{other_val['ident_shot_l1']['zero']:.4f}. A low residual on the "
                "identification shot therefore certifies nothing about the branch."
            ),
            (
                "FitResult.ok is not a trust gate: it reports rank and finiteness, "
                "not whether the coefficients are the device's. The confirmation "
                f"gate is the deployable test; at ratio {CONFIRM_RATIO:g} it "
                f"{'keeps' if best_val['accepted'] else 'rejects'} the {best_name} "
                f"fit (confirm_l1 {best_val['confirm_l1']:.4f} vs nominal "
                f"{best_val['confirm_l1_nominal']:.4f}, oracle floor "
                f"{best_val['confirm_l1_oracle']:.4f}) and "
                f"{'keeps' if other_val['accepted'] else 'rejects'} {others[0]} "
                f"(ratio {other_val['confirm_ratio']:.2f} vs "
                f"{CONFIRM_RATIO:g})."
            ),
            (
                (
                    "A window-only gate cannot certify the far field: the sign-flipped "
                    f"{others[0]} fit still predicts its own window frame well "
                    "enough to pass, so the sign error survives validation."
                )
                if other_val["accepted"]
                else (
                    f"The gate does catch the sign-flipped {others[0]} fit on this "
                    "shot, so it is worth more than FitResult.ok -- but it only "
                    "bounds the window error and cannot certify the far field."
                )
            ),
            (
                "The probe shot is always well lit: at zero command the windows "
                f"capture {min(probe_ee):.3f} to {max(probe_ee):.3f} of the light, "
                "so a failed identification here is not a starved probe but the "
                "gauge degeneracy above. A per-phase probe would be starved "
                "instead -- the commanded defocus throws the light out of the "
                "window -- which is why identification is done once at zero command."
            ),
            (
                "Before the gate decides anything, the fitted amplitude is biased, "
                f"and the bias costs the outside-window energy: on the random group "
                f"the raw {best_name} fit cuts l1_all "
                f"{rand['nominal']['l1_all']['mean']:.3f} -> "
                f"{rand['windowed_fit']['l1_all']['mean']:.3f} but moves ee_out_err "
                f"{rand['nominal']['ee_out_err']['mean']:.4f} -> "
                f"{rand['windowed_fit']['ee_out_err']['mean']:.4f}, while the oracle "
                f"at the true amplitude holds it at "
                f"{rand['oracle']['ee_out_err']['mean']:.4f}. The over-scale is what "
                "the frame objective trades for the unmodelled Noll 14 and "
                "roughness spread, so the window-only fit buys in-window fidelity "
                "at some cost in far-field energy."
            ),
        ],
    }
    commands = {
        "run": (
            "python scripts/ml/sim2f_fov_extrapolation.py "
            f"--out {args.out} --seed {args.seed} --grid {args.grid} --ccd {args.ccd}"
        ),
        "tests": (
            "python -m pytest tests/ao_shaping/ml/sim2f/test_window.py "
            "tests/ao_shaping/ml/sim2f/test_extrapolate.py -q"
        ),
    }
    artifacts = write_artifacts(out, trials, summary, meta, commands)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    case_figure = draw_cases(
        trials,
        frames,
        dict(windows)[best_name],
        best_name,
        fits[best_name],
        figures,
    )
    bar_figure = draw_error_bars(
        trials, [name for name, _ in windows], validation_summary, figures
    )

    print(f"wrote {artifacts / 'trials.json'}, {artifacts / 'trials.csv'}")
    print(f"wrote {artifacts / 'summary.json'}")
    print(f"wrote {case_figure}")
    print(f"wrote {bar_figure}")
    for name in fits:
        entry = identification[name]
        val = validation_summary[name]
        print(
            f"{name:15s} residual={entry['fit_residual']:.4f} "
            f"coef_rmse={entry['coefficient_rmse_vs_oracle_rad']:.3f} "
            f"oracle_ratio={entry['oracle_projection_ratio']:+.2f} "
            f"probe_ee_in={val['probe_ee_in']:.3f} "
            f"ident_shot fit/opposite/zero="
            f"{val['ident_shot_l1']['fit']:.4f}/"
            f"{val['ident_shot_l1']['opposite']:.4f}/"
            f"{val['ident_shot_l1']['zero']:.4f} "
            f"confirm_l1={val['confirm_l1']:.4f}/"
            f"{val['confirm_l1_nominal']:.4f} "
            f"gate={'accept' if val['accepted'] else 'REJECT'} "
            f"sens={val['sensitivity']}"
        )
    print(f"best window by fit residual: {best_name}")
    print(
        f"confirmation threshold ratio: {CONFIRM_RATIO:g} "
        f"(sensitivity {list(CONFIRM_SENSITIVITY)})"
    )
    for group in GROUPS:
        counts = best_val["groups"][group]
        for arm in ARMS:
            entry = summary[f"{best_name}|{group}|{arm}"]
            print(
                f"{group:10s} {arm:13s} "
                f"accept={counts['n_accepted']}/{counts['n_cases']} "
                f"l1_all={entry['l1_all']['mean']:.3f} "
                f"l1_out={entry['l1_out']['mean']:.3f} "
                f"ee_out_err={entry['ee_out_err']['mean']:.4f} "
                f"centroid_err_px={entry['centroid_err_px']['mean']:.2f}"
            )
    print(f"runtime {runtime:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
