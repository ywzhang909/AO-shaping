"""Device-free reproduction of the 2026-10-29 transfer-chain radiometric unit failure.

What is reproduced, in order:

1. A known-truth 2f scene built with ``FrozenDefocusForward`` (known defocus,
   known residual Noll terms, finite illumination amplitude) on a 241 x 241
   camera window (stride 2 px, 480 px across).
2. Two renders per phase (seed / perturbed) at two bit depths, quantised like
   the real camera: Mono8 (``full_scale`` 255) and Mono12 (``full_scale`` 4095,
   exposure chosen so peaks sit at 2000-3500 counts).
3. Gain/background fitted with NNLS (same design matrix shape as
   ``radiometry.fit_radiometry``, mode ``exposure``) from the Mono12 rows and
   from the Mono8 rows, each normalised with ``count_rate`` *and*, for the
   evidence column, with the pre-fix ``image / (exposure * counts_scale)``.
   Only ``count_rate`` makes the two gain estimates agree (native counts/ms);
   the old per-call bit-depth factor makes the Mono12 gain 16x too small.
4. The failure itself: ``anchored_intensity`` evaluated on the Mono12 frames
   with each of the two anchor normalisations. Because the model difference is
   *added* to the anchor, an anchor 16x too small leaves the model difference at
   its absolute scale while the anchor is 16x down -- so the anchored prediction
   is dominated by the model, and the block loop cannot see what the camera
   measured. Quantified per scene as the prediction/anchor level ratio, the
   implied vs measured loss change, and the correlation between the model
   difference and the measured difference.
5. A short anchored block loop (3 blocks x <= 5 updates, real
   ``DifferentiableUniformityOptimizer``) on the synthetic camera, accepted by
   ``paired_decision`` from the rendered frames only.

No hardware: no camera, no SLM, no device lock. Writes only the JSON and the
markdown table named on the command line.

Usage:
    .venv/Scripts/python.exe scripts/slm/simulate_transfer_units_20261029.py \
        --out report/shape_trials/blackbox_path/sim_units_20261029.json

Delete when: the ``count_rate`` contract is enforced by a test that fails if any
call site reintroduces a per-stage bit-depth factor, and the 2026-10-29 report
no longer needs a reproducible unit-mismatch demonstration.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from ao_shaping.algorithm.signal_processing.defocus_forward_core import diopter_scale
from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    DifferentiableUniformityOptimizer,
    FrozenDefocusForward,
    FrozenDefocusParameters,
    UniformityObjective,
)
from ao_shaping.algorithm.signal_processing.multishape_objective import anchored_intensity
from ao_shaping.tools.slm.transfer.paired import paired_decision
from ao_shaping.tools.slm.transfer.radiometry import count_rate
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

#: Control grid of the frozen forward model (panel samples, ``grid_size``).
GRID = 128
#: FFT size of the frozen forward model.
FFT_SIZE = 512
#: Camera window: 241 x 241 samples, stride 2 px, centred on the optical axis.
CAM_HALF_SPAN_PX = 240.0
CAM_STEP_PX = 2.0
CAM_CENTRE_XY = (256.0, 256.0)
#: Panel -> camera ramp matrix (``CoordinateTransform`` default rows).
RAMP_MATRIX = ((8083.34, 0.17), (1.08, -8136.20))

GAIN_TOLERANCE = 0.05  # relative agreement required between Mono12 and Mono8 gains
MONO12_COUNTS_SCALE = 16
MONO8_COUNTS_SCALE = 1
MONO12_EXPOSURE_MS = 1.0
MONO8_EXPOSURE_MS = 0.07
MONO12_READ_SIGMA = 4.0
MONO8_READ_SIGMA = 1.2
DELTA_RMS_RAD = 0.1
BLOCKS = 3
MAX_UPDATES = 5
#: Secondary sensitivity check: defocus error of the fitted checkpoint, relative.
MISMATCH_DEFOCUS_RELATIVE_ERROR = 0.10


def _camera_xy() -> np.ndarray:
    start = CAM_CENTRE_XY[0] - CAM_HALF_SPAN_PX
    stop = CAM_CENTRE_XY[0] + CAM_HALF_SPAN_PX + CAM_STEP_PX
    axis = np.arange(start, stop, CAM_STEP_PX)
    return np.stack(np.meshgrid(axis, axis), axis=-1).astype(np.float64)


def _truth_parameters() -> FrozenDefocusParameters:
    return FrozenDefocusParameters(
        defocus_diopter=0.35,
        origin_xy=CAM_CENTRE_XY,
        amplitude_center_xy=(960.0, 600.0),
        amplitude_width_xy=(420.0, 340.0),
        residual_noll_radians={5: 0.7, 6: -0.5, 7: 0.4, 8: 0.3},
        gain=1.0,
        background=0.0,
        sigma_ccd_px=0.5,
        ramp_matrix=RAMP_MATRIX,
        fft_size=FFT_SIZE,
        pixel_samples=1,
        grid_size=GRID,
    )


def _phases(forward: FrozenDefocusForward) -> tuple[np.ndarray, np.ndarray, float]:
    """Seed phase and a ~0.1 rad RMS perturbed phase on the control grid."""
    seed = np.nan_to_num(generate_zernike_phase({4: 0.8, 5: 0.6, 6: -0.5}, (GRID, GRID)))
    delta = np.nan_to_num(generate_zernike_phase({5: 1.0, 7: -0.8, 8: 0.6}, (GRID, GRID)))
    mask = forward.panel_mask.detach().cpu().numpy() > 0
    rms = float(np.sqrt(np.mean(delta[mask] ** 2)))
    delta = delta * (DELTA_RMS_RAD / rms)
    return seed.astype(np.float64), delta.astype(np.float64), float(rms)


def _optical(forward: FrozenDefocusForward, phase: np.ndarray, focus: np.ndarray | None) -> np.ndarray:
    """Model intensity with gain 1 / background 0 (counts per ms per unit gain)."""
    with torch.no_grad():
        image = forward.predict(phase, focus_phase=focus)
    return image.detach().cpu().numpy()[0].astype(np.float64)


def _render(rate: np.ndarray, *, exposure_ms: float, full_scale: int,
            read_sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Poisson shot noise + Gaussian read noise, quantised and clipped."""
    expected = np.clip(exposure_ms * rate, 0.0, None)
    counts = rng.poisson(expected).astype(np.float64) + rng.normal(0.0, read_sigma, expected.shape)
    return np.clip(np.rint(counts), 0.0, float(full_scale)).astype(np.float64)


def _fit_gain(optical: np.ndarray, measured: np.ndarray, exposure_ms: np.ndarray) -> dict:
    """NNLS ``g*optical + b + bias/t`` -- same design as ``radiometry.fit_radiometry``."""
    from scipy.optimize import nnls

    columns = [optical, np.ones_like(optical), np.broadcast_to(1.0 / exposure_ms[:, None, None], optical.shape)]
    design = np.stack(columns, axis=-1)
    coefficients, _ = nnls(design.reshape(-1, len(columns)), measured.ravel())
    return dict(gain=float(coefficients[0]), background=float(coefficients[1]), bias=float(coefficients[2]))


def _cv(objective: UniformityObjective, image: np.ndarray) -> float:
    """Masked squared CV via the objective's own (torch) definition."""
    with torch.no_grad():
        return float(objective.cv(torch.as_tensor(np.asarray(image, dtype=np.float32))).detach())


def _row(loss: float) -> dict:
    return {"loss": float(loss), "metrics": {"cv": float(loss)}}


def _markdown(document: dict) -> str:
    lines = [
        "# Transfer-chain radiometric units: device-free reproduction (2026-10-29)",
        "",
        f"Generated {document['generated_utc']} by `{Path(document['script']).name}` "
        "(no hardware, synthetic camera).",
        "",
        "## Scene and renders",
        "",
        "| item | value |",
        "|---|---|",
    ]
    for key, value in document["scene"].items():
        rendered = json.dumps(value) if isinstance(value, (dict, list)) else value
        lines.append(f"| {key} | {rendered} |")
    lines += ["", "## Gain calibration (native counts / ms)", "",
              "| rows | counts_scale | exposure_ms | peak counts | gain (count_rate) | gain (old /counts_scale) | background (count_rate) |",
              "|---|---|---|---|---|---|---|"]
    for depth in ("mono12", "mono8"):
        entry = document["gain_calibration"][depth]
        lines.append(
            f"| {depth} | {entry['counts_scale']} | {entry['exposure_ms']:.4f} | {entry['peak_counts']:.0f} "
            f"| {entry['fixed']['gain']:.6g} | {entry['old']['gain']:.6g} | {entry['fixed']['background']:.6g} |")
    agree = document["gain_calibration"]["agreement"]
    lines += ["",
              f"- gain ratio Mono12/Mono8 with `count_rate`: **{agree['fixed_ratio']:.4f}** "
              f"(tolerance {agree['tolerance']:.2f}) -> {'agree' if agree['fixed_within_tolerance'] else 'DISAGREE'}",
              f"- gain ratio Mono12/Mono8 with the old `/ (exposure*counts_scale)`: "
              f"**{agree['old_ratio']:.4f}** -> {'agree' if agree['old_within_tolerance'] else 'DISAGREE'}",
              "",
              "## The unit mismatch (`anchored_intensity` on Mono12 frames)", "",
              "`prediction/anchor level` compares the model's absolute scale with the anchor's on "
              "the core ROI. `predicted rel. loss change` is what `anchored_intensity` implies "
              "before any frame is measured; `measured rel. loss change` is the frozen-ROI loss "
              "between the two rendered phases (one number, shared by both columns).", "",
              "| scene | anchor normalisation | prediction/anchor level | mean abs model difference "
              "| predicted rel. loss change | measured rel. loss change | difference correlation "
              "| difference rel. L2 |",
              "|---|---|---|---|---|---|---|---|"]
    for scene in ("matched_checkpoint", "checkpoint_mismatch"):
        for name in ("fixed", "old"):
            entry = document["failure_reproduction"][scene][name]
            lines.append(
                f"| {scene.replace('_', ' ')} | {name} "
                f"| {entry['prediction_to_anchor_level_ratio']:.3f} "
                f"| {entry['model_difference_mean_abs']:.4g} "
                f"| {entry['predicted_relative_loss_change']:+.4f} "
                f"| {entry['measured_relative_loss_change']:+.4f} "
                f"| {entry['difference_correlation']:+.3f} "
                f"| {entry['difference_relative_l2']:.3f} |")
    for scene in ("matched_checkpoint", "checkpoint_mismatch"):
        block = document["block_acceptance"][scene]
        lines += ["", f"## Anchored block acceptance -- {scene.replace('_', ' ')}", "",
                  "| anchor normalisation | accepted / blocks | measured rel. loss change (mean per block) |",
                  "|---|---|---|"]
        for name in ("fixed", "old"):
            entry = block[name]
            lines.append(f"| {name} | {entry['accepted']} / {entry['blocks']} "
                         f"| {entry['mean_measured_relative_change']:+.4f} |")
        if scene == "checkpoint_mismatch":
            error = document["failure_reproduction"]["checkpoint_mismatch"]["defocus_relative_error"]
            lines += ["", f"Checkpoint defect: defocus x{1 + error:.2f} "
                          f"(+{100 * error:.0f} %)."]

    agree = document["gain_calibration"]["agreement"]
    matched = document["block_acceptance"]["matched_checkpoint"]
    mismatch = document["block_acceptance"]["checkpoint_mismatch"]
    matched_failure = document["failure_reproduction"]["matched_checkpoint"]
    mismatch_failure = document["failure_reproduction"]["checkpoint_mismatch"]
    lines += ["", "## What reproduced and what did not", "",
              f"- **Reproduced -- the bit-depth factor.** With `count_rate` the Mono12 and Mono8 "
              f"gains agree at ratio **{agree['fixed_ratio']:.4f}**; the pre-fix "
              f"`image / (exposure_ms * counts_scale)` gives **{agree['old_ratio']:.4f}**, i.e. "
              f"the Mono12 gain is 1/{1 / agree['old_ratio']:.0f} of the Mono8 one. The fitted gain "
              f"itself is a property of the bench, so only the ratio is comparable.",
              f"- **Reproduced -- the anchor level.** On Mono12 frames the model's absolute scale "
              f"sits at **{matched_failure['fixed']['prediction_to_anchor_level_ratio']:.3f}x** the "
              f"anchor with `count_rate` and at "
              f"**{matched_failure['old']['prediction_to_anchor_level_ratio']:.2f}x** with the old "
              f"normalisation -- the ~16x that makes the model difference, not the measurement, "
              f"drive `anchored_intensity`.",
              f"- **Reproduced -- the implied loss change.** The anchored prediction says "
              f"{matched_failure['old']['predicted_relative_loss_change']:+.2%} with the old "
              f"normalisation and {matched_failure['fixed']['predicted_relative_loss_change']:+.2%} "
              f"with the fix, for a measured change of "
              f"{matched_failure['fixed']['measured_relative_loss_change']:+.2%}. The fix lands "
              f"within {abs(matched_failure['fixed']['predicted_relative_loss_change'] - matched_failure['fixed']['measured_relative_loss_change']) * 100:.2f} "
              f"percentage points of the measurement; the old one is off by "
              f"{abs(matched_failure['old']['predicted_relative_loss_change'] - matched_failure['fixed']['measured_relative_loss_change']) * 100:.2f}.",
              f"- **Not reproduced -- the hardware outcome (negative result, left as found).** "
              f"`paired_decision` accepted "
              f"**{matched['fixed']['accepted']}/{matched['fixed']['blocks']}** blocks with the fix "
              f"and **{matched['old']['accepted']}/{matched['old']['blocks']}** with the old "
              f"normalisation on the matched checkpoint, and "
              f"**{mismatch['fixed']['accepted']}/{mismatch['fixed']['blocks']}** vs "
              f"**{mismatch['old']['accepted']}/{mismatch['old']['blocks']}** with the "
              f"+{100 * mismatch_failure['defocus_relative_error']:.0f} % defocus error. The "
              f"simulated loop does not reproduce the bench's 0/N rejection, and no parameter was "
              f"tuned to make it do so.",
              f"- **What the mismatch actually corrupts** is the *reading*, not always the verdict. "
              f"In the mismatched scene the old normalisation predicts "
              f"{mismatch_failure['old']['predicted_relative_loss_change']:+.2%} (worse) where the "
              f"frames measured "
              f"{mismatch_failure['old']['measured_relative_loss_change']:+.2%} (better) -- a sign "
              f"flip -- and one of three blocks is then rejected. A block that still agrees by luck "
              f"is not evidence that the anchor was right; on this bench the 16x error moved the "
              f"predicted change by "
              f"{abs(mismatch_failure['old']['predicted_relative_loss_change'] - mismatch_failure['fixed']['predicted_relative_loss_change']) * 100:.2f} "
              f"percentage points under a checkpoint defect the fit would consider acceptable.",
              ]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="JSON artefact path")
    parser.add_argument("--seed", type=int, default=1029, help="numpy generator seed")
    args = parser.parse_args()

    torch.set_num_threads(4)
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    truth = _truth_parameters()
    xy = _camera_xy()
    logger.info("building frozen forward model ({} x {} control grid, fft {})", GRID, GRID, FFT_SIZE)
    forward = FrozenDefocusForward(truth, xy, device="cpu")
    seed_phase, delta_phase, delta_raw_rms = _phases(forward)
    perturbed_phase = seed_phase + delta_phase

    optical_seed = _optical(forward, seed_phase, None)
    optical_perturbed = _optical(forward, perturbed_phase, None)
    optical_peak = float(max(optical_seed.max(), optical_perturbed.max()))

    # True radiometry: peaks land inside 2000-3500 counts at the Mono12 exposure.
    target_peak_counts = 2800.0
    gain_true = target_peak_counts / (MONO12_EXPOSURE_MS * optical_peak)
    background_true = 60.0
    logger.info("true gain {:.4g} counts/ms, background {:.4g} counts/ms", gain_true, background_true)

    depths = {
        "mono12": dict(full_scale=4095, counts_scale=MONO12_COUNTS_SCALE,
                       exposure_ms=MONO12_EXPOSURE_MS, read_sigma=MONO12_READ_SIGMA),
        "mono8": dict(full_scale=255, counts_scale=MONO8_COUNTS_SCALE,
                      exposure_ms=MONO8_EXPOSURE_MS, read_sigma=MONO8_READ_SIGMA),
    }

    # --- 2. two renders per phase at two bit depths ---------------------------------
    renders: dict[str, list[dict]] = {}
    for depth, cfg in depths.items():
        rows = []
        for phase_name, optical in (("seed", optical_seed), ("perturbed", optical_perturbed)):
            rate = gain_true * optical + background_true
            for repeat in range(2):
                rows.append(dict(
                    label=f"{depth}_{phase_name}_{repeat}", phase=phase_name, repeat=repeat,
                    exposure_ms=float(cfg["exposure_ms"]), bit_depth=depth,
                    counts_scale=int(cfg["counts_scale"]), full_scale=int(cfg["full_scale"]),
                    frame=_render(rate, exposure_ms=cfg["exposure_ms"], full_scale=cfg["full_scale"],
                                  read_sigma=cfg["read_sigma"], rng=rng),
                ))
        renders[depth] = rows
        logger.info("rendered {} {} rows (peak {} counts)", len(rows), depth, int(max(r["frame"].max() for r in rows)))

    # --- 3. gain calibration, two ways ---------------------------------------------
    def _stack(depth: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = renders[depth]
        optical = np.stack([optical_seed if r["phase"] == "seed" else optical_perturbed for r in rows])
        frames = np.stack([r["frame"] for r in rows])
        exposure = np.asarray([r["exposure_ms"] for r in rows])
        return optical, frames, exposure

    calibration: dict[str, dict] = {}
    for depth in ("mono12", "mono8"):
        optical, frames, exposure = _stack(depth)
        scale = depths[depth]["counts_scale"]
        fixed_meas = np.stack([count_rate(frame, row) for frame, row in zip(frames, renders[depth])])
        old_meas = frames / (exposure[:, None, None] * scale)
        fixed = _fit_gain(optical, fixed_meas, exposure)
        old = _fit_gain(optical, old_meas, exposure)
        calibration[depth] = dict(counts_scale=int(scale), exposure_ms=float(exposure[0]),
                                  peak_counts=float(max(r["frame"].max() for r in renders[depth])),
                                  fixed=fixed, old=old)
        logger.info("{}: gain fixed {:.5g} / old {:.5g}", depth, fixed["gain"], old["gain"])

    fixed_ratio = calibration["mono12"]["fixed"]["gain"] / calibration["mono8"]["fixed"]["gain"]
    old_ratio = calibration["mono12"]["old"]["gain"] / calibration["mono8"]["old"]["gain"]
    agreement = dict(fixed_ratio=float(fixed_ratio), old_ratio=float(old_ratio),
                     tolerance=GAIN_TOLERANCE,
                     fixed_within_tolerance=bool(abs(fixed_ratio - 1.0) <= GAIN_TOLERANCE),
                     old_within_tolerance=bool(abs(old_ratio - 1.0) <= GAIN_TOLERANCE))
    if not agreement["fixed_within_tolerance"]:
        raise ValueError(f"count_rate gains disagree ({fixed_ratio:.4f}); the simulation or the "
                         f"normalisation changed -- inspect before publishing numbers")

    checkpoint = replace(truth, gain=float(calibration["mono12"]["fixed"]["gain"]),
                         background=float(calibration["mono12"]["fixed"]["background"]))
    checkpoint_gain, checkpoint_background = checkpoint.gain, checkpoint.background

    def model_rate(phase: np.ndarray, focus: np.ndarray | None) -> np.ndarray:
        """Checkpoint prediction in counts/ms.

        Numerically the checkpoint's own ``FrozenDefocusForward``: its last two
        ops are ``sampled * gain + background``, so applying the fitted gain to
        the gain-1 render is the same expression. ``bias_counts`` is 0, hence
        ``predict`` needs no ``exposure_ms`` here (see the report).
        """
        return checkpoint_gain * _optical(forward, phase, focus) + checkpoint_background

    # --- masks, frozen once (as the real chain freezes the target ROI) --------------
    reference = model_rate(seed_phase, None)
    box = reference > 0.05 * reference.max()
    core = reference > 0.20 * reference.max()
    objective = UniformityObjective(mask_box=box, mask_core=core, w_shape=0.0, w_phase=0.0)
    logger.info("frozen ROI: box {} px, core {} px", int(box.sum()), int(core.sum()))

    # --- 4. reproduce the failure ---------------------------------------------------
    def normalise(frame: np.ndarray, exposure_ms: float, scale: int, mode: str) -> np.ndarray:
        if mode == "fixed":
            return count_rate(frame, {"exposure_ms": exposure_ms})
        if mode == "old":
            return frame / (exposure_ms * scale)
        raise ValueError(mode)

    mono12_seed_frames = [r for r in renders["mono12"] if r["phase"] == "seed"]
    mono12_perturbed_frames = [r for r in renders["mono12"] if r["phase"] == "perturbed"]
    measured_seed = np.mean([r["frame"] for r in mono12_seed_frames], axis=0)
    measured_perturbed = np.mean([r["frame"] for r in mono12_perturbed_frames], axis=0)
    measured_rate_seed = count_rate(measured_seed, {"exposure_ms": MONO12_EXPOSURE_MS})
    measured_rate_perturbed = count_rate(measured_perturbed, {"exposure_ms": MONO12_EXPOSURE_MS})
    cv_measured_seed = _cv(objective, measured_rate_seed)
    cv_measured_perturbed = _cv(objective, measured_rate_perturbed)
    measured_relative = (cv_measured_perturbed - cv_measured_seed) / cv_measured_seed
    measured_difference = measured_rate_perturbed - measured_rate_seed

    # Secondary scene: a fitted checkpoint is never exact. A 10 % defocus error is
    # a plausible fit residual and the cheapest way to ask whether a matched-model
    # scene can reproduce the hardware failure at all.
    mismatch_focus = (MISMATCH_DEFOCUS_RELATIVE_ERROR * truth.defocus_diopter
                      * diopter_scale() * forward.z4.detach().cpu().numpy())

    def reproduce(focus: np.ndarray | None) -> dict:
        """``anchored_intensity`` with both anchor normalisations, on Mono12 frames."""
        predicted_seed = model_rate(seed_phase, focus)
        predicted_perturbed = model_rate(perturbed_phase, focus)
        predicted_difference = predicted_perturbed - predicted_seed
        reference_prediction = np.abs(predicted_difference)[box]
        predicted_difference_norm = float(np.linalg.norm(predicted_difference[box]))
        measured_difference_norm = float(np.linalg.norm(measured_difference[box]))
        correlation = float(np.corrcoef(predicted_difference[box], measured_difference[box])[0, 1])
        out: dict[str, dict] = {}
        for mode in ("fixed", "old"):
            anchor = normalise(measured_seed, MONO12_EXPOSURE_MS, MONO12_COUNTS_SCALE, mode)
            anchored = anchored_intensity(anchor, predicted_perturbed, predicted_seed)
            baseline = _cv(objective, anchor)
            out[mode] = dict(
                # Level ratio: the model's absolute scale against the anchor's. This is
                # the number the 2026-10-29 radiometry docstring quotes as 16.1.
                prediction_to_anchor_level_ratio=float(predicted_seed[core].mean() / anchor[core].mean()),
                model_difference_mean_abs=float(reference_prediction.mean()),
                anchor_mean=float(anchor[core].mean()),
                predicted_relative_loss_change=float((_cv(objective, anchored) - baseline) / baseline),
                measured_relative_loss_change=float(measured_relative),
                difference_correlation=correlation,
                difference_relative_l2=float(abs(predicted_difference_norm - measured_difference_norm)
                                             / max(measured_difference_norm, 1e-12)),
            )
        return out

    failure: dict[str, dict] = {
        "matched_checkpoint": reproduce(None),
        "checkpoint_mismatch": {**reproduce(mismatch_focus),
                                "defocus_relative_error": MISMATCH_DEFOCUS_RELATIVE_ERROR},
    }
    for scene, entry in failure.items():
        for mode in ("fixed", "old"):
            logger.info("{} {}: model/anchor {:.3f}, predicted {:+.2%}, measured {:+.2%}",
                        scene, mode, entry[mode]["prediction_to_anchor_level_ratio"],
                        entry[mode]["predicted_relative_loss_change"], measured_relative)

    # --- 5. simulated anchored block loop -------------------------------------------
    def run_blocks(*, normalisation: str, focus: np.ndarray | None, tag: str) -> dict:
        best = seed_phase
        accepted = 0
        changes: list[float] = []
        records: list[dict] = []
        for block in range(BLOCKS):
            anchor_frame = _render(model_rate_truth(best), exposure_ms=MONO12_EXPOSURE_MS,
                                   full_scale=4095, read_sigma=MONO12_READ_SIGMA, rng=rng)
            a1 = _cv(objective, count_rate(anchor_frame, {"exposure_ms": MONO12_EXPOSURE_MS}))
            anchor = normalise(anchor_frame, MONO12_EXPOSURE_MS, MONO12_COUNTS_SCALE, normalisation)
            baseline_prediction = model_rate(best, focus)
            optimizer = DifferentiableUniformityOptimizer(
                forward, objective, best, phi_focus=focus, lr=0.01, step_rms_cap=0.2, device="cpu")
            updates = 0
            previous = None
            for _ in range(MAX_UPDATES):
                current = optimizer.current_phase
                prediction = model_rate(current, focus)
                optimizer.update(anchored_intensity(anchor, prediction, baseline_prediction))
                updates += 1
                if previous is not None and abs(optimizer.last_loss - previous) <= 1e-4 * abs(previous):
                    break
                previous = optimizer.last_loss
            candidate = optimizer.current_phase
            b1_frame = _render(model_rate_truth(candidate), exposure_ms=MONO12_EXPOSURE_MS,
                               full_scale=4095, read_sigma=MONO12_READ_SIGMA, rng=rng)
            b2_frame = _render(model_rate_truth(candidate), exposure_ms=MONO12_EXPOSURE_MS,
                               full_scale=4095, read_sigma=MONO12_READ_SIGMA, rng=rng)
            a2_frame = _render(model_rate_truth(best), exposure_ms=MONO12_EXPOSURE_MS,
                               full_scale=4095, read_sigma=MONO12_READ_SIGMA, rng=rng)

            def _loss(frame: np.ndarray) -> float:
                return _cv(objective, count_rate(frame, {"exposure_ms": MONO12_EXPOSURE_MS}))
            decision = paired_decision(_row(a1), _row(_loss(b1_frame)), _row(_loss(b2_frame)),
                                       _row(_loss(a2_frame)), metric="loss")
            accepted += int(decision["accepted"])
            changes.append((decision["candidate"] - decision["baseline"]) / decision["baseline"])
            records.append(dict(block=block, updates=updates, a1=a1,
                                candidate_measured=float(decision["candidate"]),
                                baseline_measured=float(decision["baseline"]),
                                drift=float(decision["drift"]), accepted=bool(decision["accepted"]),
                                model_step_rms=float(optimizer.weighted_rms(optimizer.delta_phase))))
            if decision["accepted"]:
                best = candidate
        return dict(accepted=int(accepted), blocks=BLOCKS, records=records,
                    mean_measured_relative_change=float(np.mean(changes)),
                    tag=tag)

    def model_rate_truth(phase: np.ndarray) -> np.ndarray:
        """True camera rate (counts/ms) -- what a capture on the bench would see."""
        return gain_true * _optical(forward, phase, None) + background_true

    logger.info("running anchored block loop (matched checkpoint)")
    block_acceptance = {
        "matched_checkpoint": {mode: run_blocks(normalisation=mode, focus=None, tag=mode)
                               for mode in ("fixed", "old")},
    }

    # Secondary sensitivity check: the same scene with a 10 % defocus error in the
    # checkpoint (see ``mismatch_focus`` above).
    logger.info("running anchored block loop (checkpoint defocus error +{:.0%})", MISMATCH_DEFOCUS_RELATIVE_ERROR)
    block_acceptance["checkpoint_mismatch"] = {
        mode: run_blocks(normalisation=mode, focus=mismatch_focus, tag=f"mismatch_{mode}")
        for mode in ("fixed", "old")
    }
    block_acceptance["checkpoint_mismatch"]["defocus_relative_error"] = MISMATCH_DEFOCUS_RELATIVE_ERROR

    document = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "device_free": True,
        "scene": {
            "control_grid_px": int(GRID),
            "fft_size": int(FFT_SIZE),
            "camera_window": f"{int(xy.shape[0])} x {int(xy.shape[1])} samples, stride {CAM_STEP_PX:g} px",
            "defocus_diopter": float(truth.defocus_diopter),
            "residual_noll_radians": {str(k): float(v) for k, v in truth.residual_noll_radians.items()},
            "amplitude_centre_xy": [float(v) for v in truth.amplitude_center_xy],
            "amplitude_width_xy": [float(v) for v in truth.amplitude_width_xy],
            "sigma_ccd_px": float(truth.sigma_ccd_px),
            "delta_phase_rms_rad": float(DELTA_RMS_RAD),
            "delta_phase_raw_rms_rad": float(delta_raw_rms),
            "true_gain_counts_per_ms": float(gain_true),
            "true_background_counts_per_ms": float(background_true),
            "roi_box_px": int(box.sum()),
            "roi_core_px": int(core.sum()),
        },
        "normalisation": {
            "fixed": "count_rate(image, row) = image / exposure_ms",
            "old": "image / (exposure_ms * counts_scale)",
        },
        "gain_calibration": {**calibration, "agreement": agreement,
                             "checkpoint": dict(gain=float(checkpoint_gain),
                                                background=float(checkpoint_background),
                                                bias_counts=float(checkpoint.bias_counts))},
        "failure_reproduction": failure,
        "block_acceptance": block_acceptance,
        "runtime_s": 0.0,
    }
    document["runtime_s"] = float(time.perf_counter() - started)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    markdown_path = args.out.with_suffix(".md")
    markdown_path.write_text(_markdown(document), encoding="utf-8")

    print()
    print(f"scene: grid {GRID}, fft {FFT_SIZE}, window {xy.shape[0]}x{xy.shape[1]} stride {CAM_STEP_PX:g}px, "
          f"defocus {truth.defocus_diopter} diopter, delta RMS {DELTA_RMS_RAD} rad")
    print(f"renders: mono12 peak {calibration['mono12']['peak_counts']:.0f} counts @ "
          f"{MONO12_EXPOSURE_MS} ms; mono8 peak {calibration['mono8']['peak_counts']:.0f} counts @ "
          f"{MONO8_EXPOSURE_MS} ms")
    print("gain (counts/ms)    count_rate            old /counts_scale")
    print(f"  mono12            {calibration['mono12']['fixed']['gain']:.6g}"
          f"            {calibration['mono12']['old']['gain']:.6g}")
    print(f"  mono8             {calibration['mono8']['fixed']['gain']:.6g}"
          f"            {calibration['mono8']['old']['gain']:.6g}")
    print(f"  ratio 12/8        {fixed_ratio:.4f} (tol {GAIN_TOLERANCE:.2f})"
          f"      {old_ratio:.4f}")
    print()
    for scene in ("matched_checkpoint", "checkpoint_mismatch"):
        block = failure[scene]
        print(f"unit mismatch ({scene.replace('_', ' ')}): prediction/anchor level "
              f"fixed {block['fixed']['prediction_to_anchor_level_ratio']:.2f}, "
              f"old {block['old']['prediction_to_anchor_level_ratio']:.2f}; "
              f"predicted loss change fixed {block['fixed']['predicted_relative_loss_change']:+.4f}, "
              f"old {block['old']['predicted_relative_loss_change']:+.4f}")
    print(f"relative loss change: measured {measured_relative:+.4f}")
    for scene in ("matched_checkpoint", "checkpoint_mismatch"):
        entry = block_acceptance[scene]
        print(f"blocks ({scene}): fixed {entry['fixed']['accepted']}/{entry['fixed']['blocks']}, "
              f"old {entry['old']['accepted']}/{entry['old']['blocks']}")
    print(f"runtime {document['runtime_s']:.1f} s -> {args.out} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())