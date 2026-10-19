#!/usr/bin/env python
"""GS-refine -> Zernike-order sweep -> SPGD uniformity experiment driver.

Runs a three-stage experiment that asks: *how much of the GS-refine freeform
phase survives truncation to a low-order Zernike basis, and can SPGD on those
coefficients recover the uniformity?*

Stage A - one GS-refine optimisation (``optimize_slm_gs_refine``) produces a
    freeform best phase and a flat baseline on the same bench.
Stage B - the best phase is least-squares fitted into Noll-order Zernike
    coefficients at several truncation orders (``fit_best_phase``).
Stage C - each coefficient vector seeds an SPGD run on the Zernike basis
    (``optimize_slm_zernike_pib``, objective ``rms_pib`` + square target).

Every run's best frame is re-scored offline on ONE frozen square mask so the
orders are comparable. The driver is hardware-ready (``--cam-type daheng`` /
``miicam``) and runs end-to-end with zero hardware (``--cam-type sim``).

The target box is probed from a flat far field and clamped to the camera
window, which sidesteps the known ``cam_size`` default bug where a target
larger than the window makes ``square_mask`` clip to the whole frame and pins
``pib`` at exactly 1.0.

Usage::

    python scripts/slm/run_gs_refine_zernike_sweep.py --cam-type sim \\
        --cam-size 256 --epochs 4 --gs-iters 20 --spgd-epochs 4 --n-max 4 6
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "libs") not in sys.path:
    sys.path.insert(0, str(ROOT / "libs"))

from ao_shaping.algorithm.signal_processing.square_shaping import (  # noqa: E402
    power_in_bucket,
    uniformity_cv,
)
from ao_shaping.optimizer.wfless.gs_refine_zernike_fit import (  # noqa: E402
    fit_best_phase,
)
from ao_shaping.optimizer.wfless.slm_gs_refine import (  # noqa: E402
    SlmGsRefineConfig,
    optimize_slm_gs_refine,
)
from ao_shaping.optimizer.wfless.slm_square_shaping import (  # noqa: E402
    square_aspect_ratio,
    square_encircled_energy,
    square_quality_score,
)
from ao_shaping.optimizer.wfless.slm_zernike_pib import (  # noqa: E402
    CAM_WINDOW_TARGET_MARGIN,
    TARGET_BOX_WAIST_FACTOR,
    ZERNIKE_CLIP,
    SlmZernikePibConfig,
    optimize_slm_zernike_pib,
)
from ao_shaping.runners.runner_common import (  # noqa: E402
    CameraParamsPib,
    ObjectiveTarget,
    SlmParamsPib,
    patch_sim_pib_shaping,
)
from ao_shaping.utils.image.beam_metrics import zero_order_center  # noqa: E402
from ao_shaping.utils.image.target.metrics import spot_waist_sigma  # noqa: E402
from ao_shaping.utils.image.targets import square_mask  # noqa: E402
from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms  # noqa: E402

MIN_TARGET_SIDE = 4.0
"""Floor for the shared target box, in camera pixels."""

DEFAULT_SIM_FAR_FIELD_PADDING = 16
"""Zero-padding factor for the simulated far-field FFT (``--cam-type sim``).

The library default is ``4`` (``slm_pib_sim.FAR_FIELD_PADDING``), which is tuned
for point-PIB work where a ~2 px diffraction-limited spot is adequate. This
experiment instead needs a *square target box that is resolvable in camera
pixels*: the box is ``TARGET_BOX_WAIST_FACTOR * w0``, so a ~1.7 px spot (padding
4) clamps to the :data:`MIN_TARGET_SIDE` floor and the whole sweep degenerates
(the objective diverges to ~-999, SPGD cannot improve, and every order reports
``after == before``). The far-field spot size in pixels is
``N * padding / (pi * beam_w0)``, so it is scale-invariant in the panel size.

The FFT cost scales as ``padding**2``, so the box size and the runtime trade
off against each other. Measured on the 1200x1920 panel (``beam_w0=400``,
``far_field_window=1024``, ``cam_size=256``):

    padding   spot w0   target side   ROI px   one ``far_field`` call
         4       1.72            3         9     0.71 s  (clamps to floor)
         8       3.44            7        49     2.23 s  (below the default)
        16       6.84           14       196     7.88 s  <-- this default
        24      10.29           21       441    16.93 s
        32      13.70           27       729    32.21 s

Padding 16 is the smallest value whose ROI (~196 pixels) is large enough for a
stable ``uniformity_cv`` under shot noise, while still costing ~3.5x padding 8.
Raise ``--sim-far-field-padding`` for a larger box at quadratic cost, or lower
it for a faster, noisier run.
"""

DEFAULT_N_MAX: tuple[int, ...] = (4, 6, 8, 11, 15)
"""Default truncation orders (15 / 28 / 45 / 78 / 136 Noll modes)."""

OBJECTIVE = "rms_pib"
"""Stage-C objective; the square target shape selects the ROI."""


# ---------------------------------------------------------------------------
# Pure helpers (importable without touching hardware)
# ---------------------------------------------------------------------------


def resolve_target_side(
    w0: float,
    cam_size: int,
    override: float = 0.0,
) -> float:
    """Resolve the shared square-target side, clamped to the camera window.

    The box is ``TARGET_BOX_WAIST_FACTOR * w0`` (the flat spot's waist
    diameter), floored at :data:`MIN_TARGET_SIDE` and capped at
    ``cam_size / CAM_WINDOW_TARGET_MARGIN`` so it always fits the window. An
    explicit ``override`` bypasses the clamp but is still validated.

    Args:
        w0: Flat-field spot waist sigma in camera pixels.
        cam_size: Camera window edge in pixels.
        override: Explicit side in pixels; ``0`` (or negative) means auto.

    Returns:
        The target side in camera pixels, strictly less than ``cam_size``.

    Raises:
        ValueError: If the resolved box is degenerate or does not fit the
            window (``side >= cam_size``).
    """
    cam_size = int(cam_size)
    if cam_size < 2:
        raise ValueError(f"cam_size must be at least 2, got {cam_size}")

    if override and override > 0:
        side = float(override)
    else:
        upper = cam_size / CAM_WINDOW_TARGET_MARGIN
        side = float(
            min(max(TARGET_BOX_WAIST_FACTOR * float(w0), MIN_TARGET_SIDE), upper)
        )

    if side >= cam_size:
        raise ValueError(
            f"target side {side:.1f}px does not fit the {cam_size}px camera "
            f"window (must be < cam_size); raise --cam-size or lower "
            f"--target-side"
        )
    if side < 1.0:
        raise ValueError(f"target side {side:.1f}px is degenerate")
    return side


def sweep_plan(n_max_list: tuple[int, ...] | list[int]) -> list[dict[str, int]]:
    """Build the per-order sweep plan, preserving the requested order.

    Args:
        n_max_list: Zernike truncation orders to sweep.

    Returns:
        One ``{"n_max": int, "n_modes": int}`` dict per order, where
        ``n_modes = calc_n_zernike_terms(n_max)``.
    """
    return [
        {"n_max": int(n_max), "n_modes": int(calc_n_zernike_terms(int(n_max)))}
        for n_max in n_max_list
    ]


def check_zernike_clip(
    max_abs_coefficient: float,
    clip: float = ZERNIKE_CLIP,
) -> str | None:
    """Return a warning if a fitted coefficient exceeds the optimizer clip.

    The SPGD loop truncates coefficients at ``ZERNIKE_CLIP`` rad, so a fit that
    exceeds it is silently altered. Returns ``None`` when the fit is safe.

    Args:
        max_abs_coefficient: ``max(|coefficients|)`` from the Zernike fit.
        clip: The optimizer's coefficient clip in radians.

    Returns:
        A human-readable warning string, or ``None`` if within the clip.
    """
    if not np.isfinite(max_abs_coefficient):
        return f"non-finite Zernike coefficient (max|c|={max_abs_coefficient})"
    if max_abs_coefficient > clip:
        return (
            f"max|Zernike coefficient| = {max_abs_coefficient:.3f} rad exceeds "
            f"the optimizer clip {clip:.1f} rad; the SPGD loop will truncate it"
        )
    return None


def build_stage_c_config(
    *,
    center: tuple[int, int],
    coefficients: np.ndarray,
    n_max: int,
    side: float,
    beam_radius_px: float,
    cam_type: str,
    cam_id: int,
    exposure_time_ms: float,
    cam_size: int,
    slm_number: int,
    epochs: int,
    delta: float,
    lr: float,
    seed: int,
    max_energy_loss: float = 0.6,
) -> SlmZernikePibConfig:
    """Assemble the Stage-C SPGD config for one truncation order.

    The fitted coefficients seed the optimizer through ``slm.init_c`` and the
    Zernike aperture is pinned to ``beam_radius_px`` so the fit and the
    re-optimisation share one basis.

    Args:
        center: 0-order centre ``(x, y)`` in camera-frame coordinates.
        coefficients: Noll-order seed coefficients (raw radians).
        n_max: Zernike truncation order for this run.
        side: Shared square-target side in camera pixels.
        beam_radius_px: Illuminated pupil radius on the panel, in pixels.
        cam_type: Camera backend (``sim`` / ``daheng`` / ``miicam``).
        cam_id: Camera device id.
        exposure_time_ms: Fixed exposure in ms (0 = device default).
        cam_size: Camera window edge in pixels.
        slm_number: SLM device number.
        epochs: SPGD iterations.
        delta: SPGD perturbation amplitude in radians.
        lr: Adam learning rate (0 = auto).
        seed: RNG seed.
        max_energy_loss: In-ROI energy-loss fraction that abandons an
            evaluation (``0`` disables the guard). The guard is what stops the
            search emptying the target box, but on a small box it can also
            reject every perturbation of an already-good fitted phase; lower it
            (or set ``0``) to let SPGD explore.

    Returns:
        A fully-specified :class:`SlmZernikePibConfig`.
    """
    camera = CameraParamsPib(
        cam_type=cam_type,
        cam_id=cam_id,
        exposure_time_ms=exposure_time_ms,
        cam_size=cam_size,
        target=ObjectiveTarget(name=OBJECTIVE, target_shape="square"),
        target_size=float(side),
        max_roi_energy_loss=float(max_energy_loss),
    )
    slm = SlmParamsPib(
        slm_number=slm_number,
        n_max=int(n_max),
        zernike_radius=float(beam_radius_px),
        init_c=np.asarray(coefficients, dtype=np.float64),
    )
    return SlmZernikePibConfig(
        center=center,
        epochs=int(epochs),
        algorithm="spgd",
        delta=float(delta),
        lr=float(lr),
        random_seed=int(seed),
        camera=camera,
        slm=slm,
    )


def score_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    center_xy: tuple[int, int],
    side: float,
) -> dict[str, float]:
    """Score one far-field frame on the shared square mask.

    Args:
        frame: 2-D camera frame.
        mask: Boolean square support (the frozen shared mask).
        center_xy: Square centre ``(x, y)`` in camera-frame coordinates.
        side: Square side in camera pixels.

    Returns:
        ``{"pib", "cv", "ee", "ar", "quality"}`` - all floats.
    """
    arr = np.asarray(frame, dtype=np.float64)
    side_i = int(round(side))
    pib = float(power_in_bucket(arr, mask))
    cv = float(uniformity_cv(arr, mask))
    ee = float(square_encircled_energy(arr, center_xy, side_i))
    ar = float(square_aspect_ratio(arr, center_xy, side_i))
    quality = float(square_quality_score(cv, ee, ar))
    return {"pib": pib, "cv": cv, "ee": ee, "ar": ar, "quality": quality}


def _metrics_identical(
    a: dict[str, float],
    b: dict[str, float],
    tol: float = 1e-9,
) -> bool:
    """Return ``True`` when two score dicts agree on every metric within ``tol``.

    Used to detect a Stage-C SPGD run that never moved off its own starting
    point: the optimizer's best row is then the epoch-0 row, so the re-scored
    ``after`` metrics are bit-for-bit the ``before`` metrics. A genuine
    improvement - however small - breaks the equality.
    """
    return all(abs(float(a[key]) - float(b[key])) <= tol for key in a)


# ---------------------------------------------------------------------------
# Flat-field probe
# ---------------------------------------------------------------------------


def configure_sim_system(seed: int, far_field_padding: int) -> None:
    """Reset the process-wide sim system with ``seed`` and the requested padding.

    ``slm_pib_sim.reset_system`` does not take a padding argument, so the padding
    is set on the returned system *before* any far field is computed (at that
    point ``_far_field`` and ``_pad_buffer`` are still ``None``, so the change is
    safe and the next ``far_field()`` uses it). The existing disturbance is
    carried over so a caller that installed one keeps it.

    Args:
        seed: RNG seed for the simulated noise model.
        far_field_padding: Zero-padding factor for the far-field FFT.
    """
    from ao_shaping.drivers.sim.slm_pib_sim import get_system, reset_system

    existing = get_system()
    system = reset_system(seed=seed, disturbance=existing.disturbance)
    system.far_field_padding = int(far_field_padding)


def probe_flat_field(
    cam_type: str,
    cam_id: int,
    exposure_time_ms: float,
    seed: int,
    cam_size: int,
    far_field_padding: int = DEFAULT_SIM_FAR_FIELD_PADDING,
) -> tuple[tuple[int, int], float, tuple[int, int]]:
    """Probe a flat-phase far field for the 0-order centre, waist and shape.

    The probe opens the *same* camera the Stage-C optimizer will open, so the
    returned frame shape and centre are in the exact geometry Stage C scores.
    In simulation the process-level system is reset with ``seed`` and configured
    with ``far_field_padding`` first, making the flat far field deterministic and
    wide enough for the target box to clear :data:`MIN_TARGET_SIDE`. The 0-order
    is located by ``argmax`` (never by geometry - the optical axis is not the
    frame centre).

    Args:
        cam_type: Camera backend (``sim`` / ``daheng`` / ``miicam``).
        cam_id: Camera device id.
        exposure_time_ms: Fixed exposure in ms (0 = device default).
        seed: RNG seed for the simulated system.
        cam_size: Camera window edge in pixels. Forwarded to ``create_camera``
            so the probe frame matches the requested window (the sim CCD
            otherwise falls back to its 250 px default).
        far_field_padding: Sim far-field FFT padding (ignored for hardware).

    Returns:
        ``(center_xy, w0_px, frame_shape)`` where ``center_xy`` is ``(x, y)``.
    """
    if cam_type == "sim":
        configure_sim_system(seed, far_field_padding)

    from ao_shaping.drivers.ccd.common import create_camera

    cam = create_camera(
        cam_type,
        cam_id=cam_id,
        exposure_time_ms=exposure_time_ms,
        cam_size=cam_size,
    )
    with cam:
        frame = np.asarray(cam.get_numpy_image(), dtype=np.float64)
    center = zero_order_center(frame)
    w0 = float(spot_waist_sigma(frame, center))
    return center, w0, (int(frame.shape[0]), int(frame.shape[1]))


def _crop_to_shape(
    frame: np.ndarray,
    center_xy: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    """Crop ``frame`` to ``shape`` centred on ``center_xy`` (no-op if equal).

    Stage A and Stage C can still open different camera windows on hardware
    (Stage A windows the CCD explicitly, Stage C relies on the backend), so the
    Stage-A baselines are re-cropped onto the Stage-C geometry before scoring.
    """
    arr = np.asarray(frame, dtype=np.float64)
    h, w = int(shape[0]), int(shape[1])
    if arr.shape == (h, w):
        return arr
    if arr.shape[0] < h or arr.shape[1] < w:
        raise ValueError(f"cannot crop frame {arr.shape} to larger shape {(h, w)}")
    cx, cy = int(center_xy[0]), int(center_xy[1])
    x0 = max(0, min(cx - w // 2, arr.shape[1] - w))
    y0 = max(0, min(cy - h // 2, arr.shape[0] - h))
    return arr[y0 : y0 + h, x0 : x0 + w]


def _patch_sim_stage_c_camera_window() -> None:
    """Make the Stage-C optimizer's camera honour ``cam_size`` for the sim backend.

    ``create_camera`` reads only ``cam_type`` / ``cam_id`` / ``exposure_time_ms``
    / ``skip_sampling`` off a params object, so ``optimize_slm_zernike_pib``'s
    ``create_camera(config.camera)`` silently drops ``cam_size`` and the sim CCD
    falls back to its 250 px default. Stage A already forwards it (via
    ``drivers.slm.factory.open_camera``), so without this the probe, Stage A and
    Stage C would score on three different windows and ``frame_shape`` would not
    match the requested ``--cam-size``. Rebinding the optimizer module's
    ``create_camera`` mirrors ``patch_sim_pib_shaping``'s rebinding of its
    ``Santec`` name; hardware backends filter ``cam_size`` out of their
    ``accepted_kwargs``, so this is a no-op for them.
    """
    import ao_shaping.optimizer.wfless.slm_zernike_pib as pib_opt

    original = pib_opt.create_camera

    def create_camera_with_window(
        camera_type: Any,
        cam_id: int | str = 0,
        exposure_time_ms: float = 20.0,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(camera_type, str):
            cam_size = getattr(camera_type, "cam_size", None)
            if cam_size is not None:
                kwargs.setdefault("cam_size", cam_size)
        return original(
            camera_type, cam_id=cam_id, exposure_time_ms=exposure_time_ms, **kwargs
        )

    pib_opt.create_camera = create_camera_with_window


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class SweepParams:
    """All knobs for one sweep run (mirrors the CLI)."""

    cam_type: str = "sim"
    cam_id: int = 0
    exposure_time_ms: float = 0.0
    cam_size: int = 256
    sim_far_field_padding: int = DEFAULT_SIM_FAR_FIELD_PADDING
    slm_number: int = 1
    beam_radius_px: float = 450.0
    epochs: int = 400
    gs_iters: int = 200
    spgd_epochs: int = 200
    n_max_list: tuple[int, ...] = DEFAULT_N_MAX
    delta: float = 0.02
    lr: float = 0.01
    seed: int = 0
    out: str = ""
    target_side: float = 0.0
    max_energy_loss: float = 0.6
    save_images: bool = False


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-serialisable values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scalar_columns(history: list[dict[str, Any]]) -> list[str]:
    """Column names whose values are scalars across every recorded row."""
    if not history:
        return []
    columns: list[str] = []
    for key in history[0]:
        if all(
            isinstance(row.get(key), (int, float, str, bool, type(None)))
            for row in history
        ):
            columns.append(key)
    return columns


def _write_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    """Write the scalar columns of a recorder history to CSV."""
    columns = _scalar_columns(history)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in history:
            handle.write(",".join(str(row.get(col, "")) for col in columns) + "\n")


def _save_frame_png(frame: np.ndarray, path: Path, title: str) -> None:
    """Save a single frame as a PNG (Agg backend, no display)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.asarray(frame, dtype=np.float64), cmap="inferno")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _flat_frame(history: list[dict[str, Any]]) -> np.ndarray:
    """Extract the flat-baseline frame from a Stage-A history."""
    for row in history:
        if row.get("stage") == "flat" and row.get("_img") is not None:
            return np.asarray(row["_img"], dtype=np.float64)
    raise RuntimeError("Stage-A history has no flat frame (record_debug off?)")


def run_sweep(params: SweepParams) -> dict[str, Any]:
    """Run the full GS-refine -> Zernike-fit -> SPGD sweep.

    Args:
        params: Fully-specified sweep parameters.

    Returns:
        The summary dict that is also written to ``<out>/summary.json``.
    """
    out_dir = (
        Path(params.out)
        if params.out
        else Path("data/debug/slm_gs_refine_zernike")
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if params.cam_type == "sim":
        patch_sim_pib_shaping(params.cam_type)
        _patch_sim_stage_c_camera_window()

    center_xy, w0, probe_shape = probe_flat_field(
        params.cam_type,
        params.cam_id,
        params.exposure_time_ms,
        params.seed,
        params.cam_size,
        params.sim_far_field_padding,
    )
    side = resolve_target_side(w0, params.cam_size, params.target_side)
    print(
        f"[probe] cam_type={params.cam_type} probe_shape={probe_shape} "
        f"center={center_xy} w0={w0:.2f}px target_side={side:.2f}px"
    )

    # --- Stage A: one GS-refine optimisation ------------------------------
    stage_a = SlmGsRefineConfig(
        epochs=params.epochs,
        seed=params.seed,
        gs_iters=params.gs_iters,
        target_side=int(round(side)),
        side_factor=1.0,
        cam_type=params.cam_type,
        cam_id=params.cam_id,
        exposure_time_ms=params.exposure_time_ms,
        cam_size=params.cam_size,
        slm_number=params.slm_number,
        beam_radius_px=params.beam_radius_px,
        record_debug=True,
    )
    rec_a = optimize_slm_gs_refine(stage_a)
    best_phase = np.asarray(rec_a.best_phase, dtype=np.float64)
    best_frame = np.asarray(rec_a.best_frame, dtype=np.float64)
    flat_frame = _flat_frame(rec_a.history)
    print(
        f"[stage A] best_stage={rec_a.best_stage} "
        f"best_score={max(r['score'] for r in rec_a.history):.4f} "
        f"evals={len(rec_a.history)}"
    )

    # --- Stage B + C: per-order fit and SPGD ------------------------------
    runs: list[dict[str, Any]] = []
    for entry in sweep_plan(params.n_max_list):
        n_max = entry["n_max"]
        fit = fit_best_phase(best_phase, n_max, params.beam_radius_px)
        clip_warning = check_zernike_clip(fit.max_abs_coefficient)
        if clip_warning:
            print(f"[warn] n_max={n_max}: {clip_warning}")

        cfg = build_stage_c_config(
            center=center_xy,
            coefficients=fit.coefficients,
            n_max=n_max,
            side=side,
            beam_radius_px=params.beam_radius_px,
            cam_type=params.cam_type,
            cam_id=params.cam_id,
            exposure_time_ms=params.exposure_time_ms,
            cam_size=params.cam_size,
            slm_number=params.slm_number,
            epochs=params.spgd_epochs,
            delta=params.delta,
            lr=params.lr,
            seed=params.seed,
            max_energy_loss=params.max_energy_loss,
        )
        rec_c = optimize_slm_zernike_pib(cfg)
        runs.append(
            {"entry": entry, "fit": fit, "clip_warning": clip_warning, "rec": rec_c}
        )

    # --- Shared frozen mask, on the Stage-C frame geometry ----------------
    ref_frame = np.asarray(runs[0]["rec"].history[0]["_img"], dtype=np.float64)
    ref_shape = (int(ref_frame.shape[0]), int(ref_frame.shape[1]))
    c_cx, c_cy = zero_order_center(ref_frame)
    mask = square_mask(ref_shape, c_cy, c_cx, int(round(side)))
    if not mask.any():
        raise ValueError(
            f"shared target square (side={side:.1f}px at ({c_cx},{c_cy})) is empty"
        )
    mask_center = (int(c_cx), int(c_cy))

    flat_scored = _crop_to_shape(flat_frame, zero_order_center(flat_frame), ref_shape)
    gs_scored = _crop_to_shape(best_frame, zero_order_center(best_frame), ref_shape)
    baselines = {
        "flat": score_frame(flat_scored, mask, mask_center, side),
        "gs_freeform": score_frame(gs_scored, mask, mask_center, side),
    }
    if params.save_images:
        _save_frame_png(flat_scored, out_dir / "baseline_flat.png", "flat")
        _save_frame_png(gs_scored, out_dir / "baseline_gs_freeform.png", "gs freeform")

    orders: list[dict[str, Any]] = []
    for run in runs:
        entry = run["entry"]
        fit = run["fit"]
        rec_c = run["rec"]
        n_max = entry["n_max"]

        before_frame = np.asarray(rec_c.history[0]["_img"], dtype=np.float64)
        # ``get_best_iter`` returns ``(best_row, (index, value))`` - not an int.
        best_row, (best_index, best_value) = rec_c.get_best_iter()
        after_frame = np.asarray(best_row["_img"], dtype=np.float64)
        before = score_frame(before_frame, mask, mask_center, side)
        after = score_frame(after_frame, mask, mask_center, side)
        # A Stage-C run that never moved off its own starting point leaves the
        # best row == the epoch-0 row, so ``after`` is bit-for-bit ``before``.
        # That is a legitimate (if disappointing) result, but it must never be
        # invisible: warn and record the flag.
        spgd_improved = not _metrics_identical(before, after)
        spgd_epochs_executed = int(
            rec_c.history[-1].get("_epoch", len(rec_c.history) - 1)
        )
        js = [float(row["J"]) for row in rec_c.history if row.get("J") is not None]
        if not spgd_improved:
            logger.warning(
                "n_max={} SPGD produced no improvement: the best metrics equal "
                "the pre-SPGD fit metrics (best_epoch={}, best_objective={:.6g}); "
                "recorded as spgd_improved=false",
                n_max,
                int(best_index),
                float(best_value),
            )
        print(
            f"n_max={n_max} modes={fit.n_modes} "
            f"fit_rms={fit.rms_residual_rad:.4f}rad "
            f"max|c|={fit.max_abs_coefficient:.3f} | "
            f"pib {before['pib']:.2f}->{after['pib']:.2f}  "
            f"cv {before['cv']:.2f}->{after['cv']:.2f}  "
            f"ee {before['ee']:.2f}->{after['ee']:.2f}  "
            f"improved={spgd_improved}"
        )

        _write_history_csv(rec_c.history, out_dir / f"spgd_nmax{n_max}.csv")
        np.save(out_dir / f"best_phase_nmax{n_max}.npy", best_phase)
        np.save(out_dir / f"fit_coefficients_nmax{n_max}.npy", fit.coefficients)
        if params.save_images:
            _save_frame_png(
                before_frame, out_dir / f"before_nmax{n_max}.png", f"n_max={n_max} fit"
            )
            _save_frame_png(
                after_frame, out_dir / f"after_nmax{n_max}.png", f"n_max={n_max} spgd"
            )

        orders.append(
            {
                "n_max": n_max,
                "n_modes": int(fit.n_modes),
                "fit_rms_rad": float(fit.rms_residual_rad),
                "max_abs_coefficient": float(fit.max_abs_coefficient),
                "clip_warning": run["clip_warning"],
                "before": before,
                "after": after,
                "best_epoch": int(best_index),
                "best_objective": float(best_value),
                "spgd_improved": bool(spgd_improved),
                "spgd_epochs_executed": spgd_epochs_executed,
                "spgd_J_min": float(min(js)) if js else None,
                "spgd_J_max": float(max(js)) if js else None,
                "spgd_J_final": float(js[-1]) if js else None,
            }
        )

    summary = {
        "config": _jsonable(asdict(params)),
        "cam_size": int(params.cam_size),
        "probe_shape": [int(probe_shape[0]), int(probe_shape[1])],
        "frame_shape": [int(ref_shape[0]), int(ref_shape[1])],
        "center_xy": [int(center_xy[0]), int(center_xy[1])],
        "mask_center_xy": [int(mask_center[0]), int(mask_center[1])],
        "w0_px": float(w0),
        "target_side": float(side),
        "beam_radius_px": float(params.beam_radius_px),
        "best_stage": str(rec_a.best_stage),
        "baselines": baselines,
        "orders": orders,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    print(f"[done] wrote {out_dir / 'summary.json'}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cam-type", default="sim", choices=["sim", "daheng", "miicam"])
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--exposure-time-ms", type=float, default=0.0)
    parser.add_argument("--cam-size", type=int, default=256)
    parser.add_argument(
        "--sim-far-field-padding",
        type=int,
        default=DEFAULT_SIM_FAR_FIELD_PADDING,
        help=(
            "Sim far-field FFT zero-padding factor (--cam-type sim only). "
            f"Default {DEFAULT_SIM_FAR_FIELD_PADDING}: the library default 4 gives "
            "a ~1.7 px spot that clamps the target box to its floor."
        ),
    )
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--beam-radius-px", type=float, default=450.0)
    parser.add_argument("--epochs", type=int, default=400, help="Stage-A SPGD epochs.")
    parser.add_argument("--gs-iters", type=int, default=200, help="GS iterations.")
    parser.add_argument(
        "--spgd-epochs", type=int, default=200, help="Stage-C SPGD epochs per order."
    )
    parser.add_argument(
        "--n-max",
        type=int,
        nargs="+",
        default=list(DEFAULT_N_MAX),
        help="Zernike truncation orders to sweep.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.02,
        help=(
            "SPGD perturbation amplitude in radians. Only honoured when "
            "--lr is non-zero: with --lr 0 the optimizer's own schedule "
            "overrides it (capped at 0.5 rad, larger than a typical fitted "
            "coefficient, which makes SPGD reject every step)."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Adam learning rate (0 = optimizer auto-schedule, which also overrides --delta).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="", help="Output directory.")
    parser.add_argument(
        "--target-side",
        type=float,
        default=0.0,
        help="Explicit target side in px (0 = auto-probe).",
    )
    parser.add_argument(
        "--max-energy-loss",
        type=float,
        default=0.6,
        help=(
            "In-ROI energy-loss fraction that abandons an SPGD evaluation "
            "(0 disables the guard). Lower it if every perturbation of an "
            "already-good fitted phase is being rejected."
        ),
    )
    parser.add_argument("--save-images", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the sweep."""
    args = _build_parser().parse_args(argv)
    params = SweepParams(
        cam_type=args.cam_type,
        cam_id=args.cam_id,
        exposure_time_ms=args.exposure_time_ms,
        cam_size=args.cam_size,
        sim_far_field_padding=args.sim_far_field_padding,
        slm_number=args.slm_number,
        beam_radius_px=args.beam_radius_px,
        epochs=args.epochs,
        gs_iters=args.gs_iters,
        spgd_epochs=args.spgd_epochs,
        n_max_list=tuple(args.n_max),
        delta=args.delta,
        lr=args.lr,
        seed=args.seed,
        out=args.out,
        target_side=args.target_side,
        max_energy_loss=args.max_energy_loss,
        save_images=args.save_images,
    )
    run_sweep(params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
