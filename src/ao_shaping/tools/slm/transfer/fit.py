"""Offline transfer of a compact physical forward model using recorded CCD data.

No device imports. Geometry uses the recorded ramp pairs. Each individual stage
fits and selects its iteration on training phases only; automatic stage selection
uses validation phases and therefore requires subsequent independent CCD checks.
Exposure division assumes a linear camera response, recorded as an assumption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates


from ao_shaping.tools.slm.transfer.capture import clock
from ao_shaping.tools.slm.transfer.device import DEVICE_CHOICES, resolve_device
from ao_shaping.tools.slm.transfer.radiometry import count_rate

from .splits import SEALED_SPLITS
from .parallel_fit import EarlyStopper


def spot(row):
    if "spot_xy" in row:
        return np.asarray(row["spot_xy"], dtype=float)
    return np.array([row["spot"][f"centroid_{axis}"] for axis in ("x", "y")])


def geometry(rows):
    """Fit full 2D ramp displacement; preserve sign and both CCD axes.

    Only ``split == "calibration"`` ramp rows may define the matrix: a
    validation/test ramp deliberately probes the held-out phase and must not
    leak into the geometry estimate.  Historical captures are all
    ``calibration``, and a row that declares no split at all (the in-memory
    capture manifest before the fitter assigns splits) keeps the historical
    behaviour of being usable, so their matrix is unchanged.
    """
    ramps = [row for row in rows if row.get("split", "calibration") == "calibration"]
    slopes = {"x": [], "y": []}
    for row in ramps:
        if not row["label"].startswith("ramp_"):
            continue
        _, axis, period, sign = row["label"].split("_")
        if int(sign) != 1:
            continue
        negative = next((other for other in ramps
                         if other["label"] == f"ramp_{axis}_{period}_-1"
                         and other.get("branch") == row.get("branch")), None)
        if negative is None:
            raise ValueError(f"missing negative ramp: {row['label']}")
        slopes[axis].append((spot(row) - spot(negative)) * float(period) / 2)
    if any(not values for values in slopes.values()):
        raise ValueError("both ramp axes required")
    matrix = np.stack([np.mean(slopes[axis], axis=0) for axis in ("x", "y")], axis=1)
    if not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 100:
        raise ValueError("degenerate camera geometry")
    spread = {axis: np.asarray(values).tolist() for axis, values in slopes.items()}
    return matrix, spread


def effective_ramp_matrix(matrix, correction):
    """Ramp matrix equivalent to a camera->panel correction ``G``.

    The fit uses ``inverse_eff = inv(matrix) @ G`` as its camera->panel map, so
    the matching panel->camera ramp matrix is
    ``inv(inverse_eff) = inv(G) @ matrix``.  Solved with numpy so the exported
    checkpoint never depends on the fitted torch tensors; ``G = I`` returns
    ``matrix`` bit-for-bit unchanged.
    """
    base = np.asarray(matrix, dtype=np.float64)
    gain = np.asarray(correction, dtype=np.float64)
    if np.array_equal(gain, np.eye(2)):
        return base
    return np.linalg.solve(gain, base)


def valid(row, full_scale=None):
    """Whether a captured row may enter the fit.

    The peak gate is a *fraction* of the row's own full scale. It used to be a
    hardcoded ``peak < 250``: correct for Mono8 and silently wrong for every
    other depth — a Mono12 capture (peaks near 4095) fails it for every row, so
    ``load_dataset`` reported "no usable rows" instead of a bit-depth problem.
    """
    scale = full_scale if full_scale is not None else row.get("full_scale", 255)
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        scale = 255.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 255.0
    try:
        peak = float(row.get("peak", scale))
    except (TypeError, ValueError):
        peak = scale
    return (row.get("settled", {}).get("settled", True)
            and row.get("exposure_status", row.get("exposure_setting", {}).get("status", "ok")) == "ok"
            and peak < 250 / 255 * scale
            and float(row["exposure_ms"]) > 0)


def load_dataset(directory, size, span, samples=None, *, train_all=False,
                 score_final_test=False):
    """Assemble training/validation pairs from a capture manifest.

    ``samples`` is an optional ``{manifest "file": payload}`` mapping — the
    capture stage's in-memory measurements (``capture.capture`` returns exactly
    that). When given, the payload comes from there instead of
    ``np.load(directory / row["file"])``; the manifest and ``config.json`` are
    still read from ``directory`` because they are small metadata that stays on
    disk either way. A payload only has to carry what the fit reads:
    ``model_phase`` and ``raw_frames``.

    The ``panel_gray`` marker keeps old on-disk datasets on their resampling
    path: a payload with neither ``panel_gray`` nor ``panel_phase`` (the
    in-memory form) is a control-grid capture by construction.

    ``train_all=False`` keeps the historical split: the returned ``train`` mask
    marks exactly the ``split == "train"`` rows. Setting ``train_all=True``
    folds the selection (validation) rows into the training set -- sealed
    ``test`` / ``final_test`` rows NEVER join the fold (#254 T2) -- which is
    what a re-fit after a failed acceptance round uses to squeeze more
    performance from the fixed data. The two-phase guard
    is skipped in that mode, since a deliberately single-fold dataset is legal.
    """
    rows = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    usable = [row for row in rows if valid(row)]
    matrix, ramp_samples = geometry(usable)
    references = [row for row in usable if row["label"] == "reference_start"]
    if not references:
        raise ValueError("valid reference_start required")
    centre = spot(references[0])
    yy, xx = np.mgrid[-span:span+1:2, -span:span+1:2]
    xy = np.stack((xx + round(centre[0]), yy + round(centre[1])), axis=-1)
    selected = [row for row in usable
                if row.get("split") in {"train", "test", "validation"}
                or (score_final_test and row.get("split") == "final_test")]
    phases, images = [], []
    for row in selected:
        payload = samples[row["file"]] if samples is not None else np.load(directory / row["file"])
        try:
            model_phase = np.asarray(payload["model_phase"])
            # New captures store exactly the control grid that was displayed.
            if model_phase.shape == (size, size) and ("panel_gray" in payload or "panel_phase" not in payload):
                phase = model_phase
            else:
                panel = np.zeros((1920, 1920), dtype=np.float32)
                panel[:1200] = payload["panel_phase"]
                gy, gx = np.mgrid[:size, :size] * (1920 / size)
                phase = map_coordinates(panel, [gy, gx], order=1)
            frame = np.asarray(payload["raw_frames"]).mean(0)
        finally:
            if hasattr(payload, "close"):
                payload.close()
        if xy[..., 0].min() < 0 or xy[..., 1].min() < 0 or xy[..., 0].max() >= frame.shape[1] or xy[..., 1].max() >= frame.shape[0]:
            raise ValueError("fixed camera window extends beyond sensor")
        images.append(count_rate(frame[xy[..., 1].astype(int), xy[..., 0].astype(int)], row))
        phases.append(phase)
    train = np.array([row["split"] == "train" for row in selected])
    if train_all:
        train = np.array([row["split"] not in SEALED_SPLITS for row in selected])
    elif not train.any() or train.all():
        raise ValueError("independent training and validation phases required")
    return selected, np.stack(phases), np.stack(images), xy, matrix, ramp_samples, train


def metrics(predictions, measured):
    results = []
    for prediction, target in zip(predictions, measured):
        cv_p = float(prediction.std() / max(float(prediction.mean()), 1e-9))
        cv_t = float(target.std() / max(float(target.mean()), 1e-9))
        p, t = prediction.ravel(), target.ravel()
        results.append(dict(correlation=float(np.corrcoef(p, t)[0, 1]),
                            cv_prediction=cv_p, cv_measured=cv_t, cv_abs_error=abs(cv_p-cv_t),
                            relative_l2=float(np.linalg.norm(p-t)/max(np.linalg.norm(t), 1e-9))))
    return results


@dataclass
class FitComponents:
    """Assembled fit state shared by :func:`fit` and the parallel worker."""
    model: "Any"
    rows: list
    phases: np.ndarray
    images: np.ndarray
    xy: np.ndarray
    matrix: np.ndarray
    ramp_samples: dict
    train: np.ndarray
    frozen: "Any"
    old: "Any"
    measured_origin: np.ndarray
    kernel: "Any"
    exposure_values: np.ndarray


def build_fit_model(dataset, checkpoint, *, span, device=None, adaptation="full",
                   amplitude_center_lr=None, radiometry="exposure", samples=None,
                   train_all=False, learn_geometry=False, learn_psf=False,
                   psf_sigma_init=None, psf_sigma_max=8.0):
    """Assemble a physical-forward fit model from a recorded dataset.

    Shared by :func:`fit` (offline) and the headless parallel worker. The model is
    warm-started from ``checkpoint`` and built fresh each call, so a monitoring
    worker can rebuild it per poll without touching this code. The caller owns
    the iteration loop.

    ``samples`` is the capture stage's in-memory ``{manifest "file": payload}``
    mapping (see :func:`load_dataset`). It must be threaded through here, not
    just through :func:`load_dataset`: the extraction of this function dropped it
    once (commit a35d5af0), and the symptom is a crash in the **fit** stage on a
    capture that succeeded — ``save_dataset=False`` means no ``sample_*.npz`` was
    ever written, so the disk fallback raises ``FileNotFoundError`` on a file that
    was never supposed to exist (2026-11-03 19:26 session, ``sample_0009.npz``).
    """
    import torch
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward, load_frozen_parameters,
    )
    from ao_shaping.algorithm.signal_processing.physical_forward_fit import PhysicalForwardFit
    from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase
    from .adaptation import configure_adaptation

    dataset = Path(dataset)
    old = load_frozen_parameters(checkpoint).effective_parameters()
    device = resolve_device(device)
        # Only the retrain loop deviates from the historical split, so the extra
    # keyword is passed *only* when it is set: a caller (or a test) that
    # replaces ``load_dataset`` keeps working unchanged.
    load_kwargs: dict[str, Any] = {"samples": samples}
    if train_all:
        load_kwargs["train_all"] = True
    rows, phases, images, xy, matrix, ramp_samples, train = load_dataset(
        dataset, old.grid_size, span, **load_kwargs)
    manifest_rows = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    reference_row = next(row for row in manifest_rows
                        if row["label"] == "reference_start" and valid(row))
    measured_origin = spot(reference_row)
    initial = replace(old, ramp_matrix=tuple(tuple(x) for x in matrix))
    frozen = FrozenDefocusForward(initial, xy, device=device)
    basis = {4: frozen.z4.detach().cpu().numpy()}
    if frozen.residual_basis is not None:
        basis.update({j: b.detach().cpu().numpy()
                      for j, b in zip(frozen.residual_modes, frozen.residual_basis)})
    # ``PhysicalForwardFit`` optimises Z5..Z11 and indexes ``basis_arrays[j]`` for
    # every one of them, so a checkpoint that declares no residual modes (an empty
    # ``residual_noll_radians``, e.g. an older M1/M3 export) or only some of them
    # would fail the fit with a KeyError. Fill the gap with the same fallback
    # ``PhysicalForwardFit`` uses when no basis is passed at all: the bare mode on
    # the control grid.
    for j in range(5, 12):
        if j not in basis:
            basis[j] = np.nan_to_num(
                generate_zernike_phase({j: 1.0}, (old.grid_size, old.grid_size)))
    # Global registration (issue #237) is carried in from the checkpoint and is not
    # fitted here, so the refit sees the same camera correspondence the frozen model did.
    # The trainable static CCD PSF is also opt-in.  When its start value is omitted
    # it reuses the checkpoint's sigma if that one is physical (> 0); a frozen
    # ``sigma_ccd_px = 0.0`` means "no blur" and is not a useful start for a
    # trainable Gaussian, so 1.0 px is used instead.
    if learn_psf and psf_sigma_init is None:
        old_sigma = float(old.sigma_ccd_px)
        initial_psf_sigma = old_sigma if old_sigma > 0 else 1.0
    else:
        initial_psf_sigma = 1.0 if psf_sigma_init is None else float(psf_sigma_init)
    model = PhysicalForwardFit(phases, images, xy, matrix, 0., train, old.amplitude_center_xy,
                              learn_amplitude=True, initial_defocus=old.defocus_diopter,
                              learn_residual=True, fft_size=old.fft_size, pixel_samples=old.pixel_samples,
                              basis_arrays=basis, batch_size=4, device=device,
                              registration_shift_ccd_px=old.registration_shift_ccd_px,
                              learn_geometry=bool(learn_geometry),
                              learn_psf=bool(learn_psf),
                              initial_psf_sigma_ccd_px=initial_psf_sigma,
                              psf_sigma_max_ccd_px=float(psf_sigma_max))
    with torch.no_grad():
        model.origin.copy_(torch.tensor(measured_origin))
        model.log_width.copy_(torch.tensor(old.amplitude_width_xy).log())
        model.residual.copy_(torch.tensor([old.residual_noll_radians.get(j, 0) for j in model.residual_modes]))
    configure_adaptation(model, adaptation, learn_geometry=bool(learn_geometry),
                         learn_psf=bool(learn_psf))
    if amplitude_center_lr is not None:
        if not np.isfinite(amplitude_center_lr) or amplitude_center_lr <= 0:
            raise ValueError("amplitude_center_lr must be finite and positive")
        for group in model.optimizer.param_groups:
            if any(parameter is model.center for parameter in group["params"]):
                group["lr"] = amplitude_center_lr
    owner = {id(model.d): "d", id(model.origin): "origin", id(model.center): "center",
             id(model.log_width): "log_width", id(model.residual): "residual"}
    if getattr(model, "psf_raw", None) is not None:
        owner[id(model.psf_raw)] = "psf_raw"
    for group in model.optimizer.param_groups:
        group.setdefault("name", owner.get(id(group["params"][0]), "unknown"))
    return FitComponents(model=model, rows=rows, phases=phases, images=images, xy=xy,
                        matrix=matrix, ramp_samples=ramp_samples, train=train, frozen=frozen,
                        old=old, measured_origin=measured_origin, kernel=frozen._blaze_kernel,
                        exposure_values=np.asarray([row["exposure_ms"] for row in rows]))


def fit_loss(model, comps, radiometry_mode="exposure", iteration=0):
    """One forward pass + loss + provenance entry. No backward / step / file write."""
    import torch
    from torch.nn.functional import conv2d
    from .radiometry import fit_radiometry

    started = clock()
    prediction = model.prediction()
    # A trainable static CCD PSF replaces ``comps.kernel`` (never both: that
    # would double-convolve).  Disabled or absent -> the frozen kernel exactly.
    psf_kernel = getattr(model, "psf_kernel", None)
    kernel = psf_kernel() if callable(psf_kernel) else None
    if kernel is None:
        kernel = comps.kernel
    if kernel is not None:
        prediction = conv2d(prediction[:, None], kernel, padding=12)[:, 0]
    optical_prediction = prediction.detach().cpu().numpy()
    _, radiometry = fit_radiometry(optical_prediction, comps.images, comps.exposure_values,
                                  comps.train, mode=radiometry_mode)
    scale, offset = radiometry["gain"], radiometry["background"]
    exposure_tensor = torch.as_tensor(comps.exposure_values[:, None, None],
                                     dtype=prediction.dtype, device=prediction.device)
    prediction = prediction * scale + offset + radiometry["bias_counts"] / exposure_tensor
    p = prediction[model.train_mask]
    target = model.images[model.train_mask]
    target_mean = target.mean((1, 2), keepdim=True).clamp_min(1e-9)
    target_cv = target.std((1, 2), correction=0) / target_mean[:, 0, 0]
    normalized = p / p.mean((1, 2), keepdim=True).clamp_min(1e-9)
    cv = p.std((1, 2), correction=0) / p.mean((1, 2)).clamp_min(1e-9)
    target_normalized = target / target_mean
    relative_mse = (((normalized - target_normalized) ** 2).mean((1, 2))
                    / target_normalized.square().mean((1, 2)).clamp_min(1e-9)).mean()
    pc, tc = p.flatten(1), target.flatten(1)
    pc, tc = pc - pc.mean(1, keepdim=True), tc - tc.mean(1, keepdim=True)
    correlation_loss = (1 - (pc * tc).sum(1) / (pc.norm(dim=1) * tc.norm(dim=1)).clamp_min(1e-9)).mean()
    loss = correlation_loss + .25 * relative_mse + .25 * ((cv - target_cv) / target_cv.clamp_min(.1)).square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite training loss; prior snapshots retained")
    sigma_getter = getattr(model, "psf_sigma_value", None)
    sigma_ccd_px = sigma_getter() if callable(sigma_getter) else None
    if sigma_ccd_px is None:
        sigma_ccd_px = comps.old.sigma_ccd_px
    params = dict(defocus_diopter=float(model.d.detach()), origin=model.origin.detach().tolist(),
                  amplitude_center=model.center.detach().tolist(),
                  amplitude_width=model.log_width.detach().exp().tolist(),
                  residual_noll_radians={str(j): float(v) for j, v in zip(model.residual_modes, model.residual.detach())},
                  gain=1., background=0., sigma_ccd_px=sigma_ccd_px)
    predicted = prediction.detach().cpu().numpy()
    params.update(radiometry)
    correction = model.geometry_value()
    entry = dict(iteration=iteration, loss=float(loss.detach()), parameters=params,
                 metrics=metrics(predicted, comps.images),
                 base_ramp_matrix=comps.matrix.tolist(),
                 learned_geometry_correction=correction.tolist(),
                 effective_ramp_matrix=effective_ramp_matrix(comps.matrix, correction).tolist(),
                 **started,
                 end_monotonic_ns=__import__("time").monotonic_ns())
    return loss, entry, predicted


def _flatten_parameters(parameters):
    """Flatten one iteration's parameter dict to ``{name: float}`` (debug-weights).

    Gives a stable, plot-friendly key set for the per-iteration weight trace:
    ``origin`` / ``amplitude_center`` / ``amplitude_width`` are lists and
    ``residual_noll_radians`` is a dict, so both are expanded to ``name_index``
    keys. Only numbers are kept.
    """
    flat = {
        "defocus_diopter": float(parameters["defocus_diopter"]),
        "gain": float(parameters.get("gain", 0.0)),
        "background": float(parameters.get("background", 0.0)),
        "sigma_ccd_px": float(parameters.get("sigma_ccd_px", 0.0)),
    }
    for key in ("origin", "amplitude_center", "amplitude_width"):
        values = parameters.get(key) or []
        for index, value in enumerate(values):
            flat[f"{key}_{index}"] = float(value)
    for mode, value in (parameters.get("residual_noll_radians") or {}).items():
        flat[f"residual_noll_{mode}"] = float(value)
    return flat


def _write_weight_trajectory(debug_dir, trajectory):
    """Write ``weights_trajectory.npz`` + ``weights_summary.json`` (debug-weights).

    ``trajectory`` is the per-iteration list of :func:`_flatten_parameters`
    dicts. The summary reports each parameter's first→last change and its
    largest single-iteration step — the direct answer to "did the weights move
    during training".
    """
    if not trajectory:
        return
    keys = sorted({key for row in trajectory for key in row})
    arrays = {
        key: np.array([row.get(key, np.nan) for row in trajectory], dtype=float)
        for key in keys
    }
    np.savez_compressed(
        debug_dir / "weights_trajectory.npz",
        iterations=np.arange(len(trajectory)),
        **arrays,
    )
    summary = {}
    for key, values in arrays.items():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        summary[key] = {
            "first": float(finite[0]),
            "last": float(finite[-1]),
            "delta": float(finite[-1] - finite[0]),
            "max_abs_step": (
                float(np.max(np.abs(np.diff(finite)))) if finite.size > 1 else 0.0
            ),
        }
    (debug_dir / "weights_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def fit(args):
    adaptation = getattr(args, "adaptation", "full")
    if adaptation == "auto":
        return fit_adaptive(args)
    if adaptation not in {"full", "alignment"}:
        raise ValueError("unknown adaptation mode")
    comps = build_fit_model(args.dataset, args.checkpoint, span=args.span,
                            device=getattr(args, "device", None),
                            adaptation=adaptation,
                            amplitude_center_lr=getattr(args, "amplitude_center_lr", None),
                            radiometry=getattr(args, "radiometry", "exposure"),
                            samples=getattr(args, "samples", None),
                            train_all=bool(getattr(args, "train_all", False)),
                            learn_geometry=bool(getattr(args, "learn_geometry", False)),
                            learn_psf=bool(getattr(args, "learn_psf", False)),
                            psf_sigma_init=getattr(args, "psf_sigma_init", None),
                            psf_sigma_max=float(getattr(args, "psf_sigma_max", 8.0)))
    import torch
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = dict(dataset=str(args.dataset.resolve()), checkpoint=str(args.checkpoint.resolve()),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      manifest_sha256=hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest(),
                      sample_source="memory" if getattr(args, "samples", None) is not None else "dataset",
                      device=comps.model._device.type, device_requested=getattr(args, "device", None),
                      iterations=args.iterations, ramp_matrix=comps.matrix.tolist(), ramp_samples=comps.ramp_samples,
                      base_ramp_matrix=comps.matrix.tolist(),
                      learned_geometry_correction=np.eye(2).tolist(),
                      effective_ramp_matrix=comps.matrix.tolist(),
                      learn_geometry=bool(getattr(args, "learn_geometry", False)),
                      learn_psf=bool(getattr(args, "learn_psf", False)),
                      psf_sigma_init=getattr(args, "psf_sigma_init", None),
                      psf_sigma_bound_ccd_px=float(getattr(args, "psf_sigma_max", 8.0)),
                      record_responses=bool(getattr(args, "record_responses", False)),
                      adaptation=adaptation,
                      amplitude_center_lr=getattr(args, "amplitude_center_lr", None),
                      radiometry=getattr(args, "radiometry", "exposure"),
                      exposure_assumption="counts/exposure_ms; linearity and illuminated offset not independently calibrated",
                      train_all=bool(getattr(args, "train_all", False)),
                      window_step_px=2, window_span_px=args.span, rows=comps.rows, **clock())
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    np.savez_compressed(args.out / "inputs.npz", phases=comps.phases, measured=comps.images,
                        camera_xy=comps.xy, train=comps.train)
    response_records = []
    if getattr(args, "record_responses", False):
        from ao_shaping.utils.image.model_response import summarize_response

        def record_response(name, value, batch_start):
            response_records.append(dict(stage=name, batch_start=batch_start,
                                         **clock(), **summarize_response(value)))

        comps.model.diagnostic_callback = record_response
    # Debug-weights: an explicit, opt-in trace of every trainable parameter per
    # iteration. Off by default and purely additive (three files under
    # ``<out>/debug_weights``); the canonical ``iterations.pkl`` stays unchanged.
    debug_weights = bool(getattr(args, "debug_weights", False))
    debug_dir = None
    weight_stream = None
    weight_trajectory: list[dict] = []
    if debug_weights:
        debug_dir = args.out / "debug_weights"
        debug_dir.mkdir(parents=True, exist_ok=True)
        weight_stream = (debug_dir / "weights_history.jsonl").open("w", encoding="utf-8")
    best_loss, history = float("inf"), []
    radiometry_mode = getattr(args, "radiometry", "exposure")
    early_stopper = EarlyStopper(
        patience=getattr(args, "patience", 8),
        min_delta=getattr(args, "min_delta", 1e-4),
    )
    stopped_at = None
    try:
        for iteration in range(args.iterations + 1):
            comps.model.optimizer.zero_grad()
            loss, entry, predicted = fit_loss(comps.model, comps, radiometry_mode, iteration)
            if iteration == 0 and entry["parameters"]["gain"] <= 0:
                raise ValueError(
                    "radiometry gain collapsed to zero at iteration 0: the checkpoint's "
                    "field does not overlap the measured window (origin/geometry mismatch). "
                    "The loss is then independent of the field, so no parameter can move; "
                    "re-anchor the geometry or refit from a fresh capture. "
                    f"(measured origin {np.round(comps.measured_origin, 1).tolist()}, "
                    f"checkpoint origin {np.round(comps.old.origin_xy, 1).tolist()})")
            if getattr(args, "record_responses", False):
                entry["responses"] = list(response_records)
            history.append(entry)
            np.savez_compressed(args.out / f"iteration_{iteration:04d}.npz", prediction=predicted)
            with (args.out / "iterations.pkl").open("wb") as stream:
                pickle.dump(history, stream, protocol=5)
            if weight_stream is not None:
                flat = _flatten_parameters(entry["parameters"])
                record = {"iteration": iteration, "loss": entry["loss"], "parameters": flat}
                if weight_trajectory:
                    previous = weight_trajectory[-1]
                    record["delta"] = {
                        key: flat[key] - previous.get(key, flat[key]) for key in flat
                    }
                weight_stream.write(json.dumps(record) + "\n")
                weight_stream.flush()
                weight_trajectory.append(flat)
            if entry["loss"] < best_loss:
                best_loss = entry["loss"]
                (args.out / "result.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
            if early_stopper.update(entry["loss"]):
                stopped_at = iteration
                print(
                    f"fit early-stop at iteration {iteration}: loss={entry['loss']:.6f}",
                    flush=True,
                )
                break
            if iteration % 10 == 0:
                print(f"{iteration}: train loss={entry['loss']:.5f}", flush=True)
            if iteration < args.iterations:
                loss.backward()
                comps.model.optimizer.step()
                with torch.no_grad():
                    comps.model.d.clamp_(-2, 2)
                    comps.model.center.clamp_(0, 1920)
                    comps.model.log_width.clamp_(float(np.log(80)), float(np.log(800)))
    finally:
        # Close the trace on every exit, including the collapsed-gain raise, so a
        # Windows run never leaves ``weights_history.jsonl`` locked. The partial
        # trajectory still lands, which is what a debugging run usually wants.
        if weight_stream is not None:
            weight_stream.close()
            _write_weight_trajectory(debug_dir, weight_trajectory)
    best = json.loads((args.out / "result.json").read_text())
    provenance["early_stopped"] = stopped_at is not None
    provenance["stopped_iteration"] = stopped_at
    # The exported geometry is the *selected* iteration's, not the last one's:
    # ``result.json`` holds the best training-loss entry, and its correction is
    # what that iteration actually predicted with.  With geometry learning off
    # this is the identity, so the historical matrix goes out unchanged.
    correction = np.asarray(best.get("learned_geometry_correction", np.eye(2)), dtype=np.float64)
    exported_matrix = effective_ramp_matrix(comps.matrix, correction)
    learned_geometry = dict(base_ramp_matrix=comps.matrix.tolist(),
                            learned_geometry_correction=correction.tolist(),
                            effective_ramp_matrix=exported_matrix.tolist())
    provenance.update(learned_geometry)
    # The exported PSF sigma is the selected iteration's: the learned value when
    # ``--learn-psf`` is on, the checkpoint's otherwise.  ``result.json`` already
    # carries it in ``parameters`` for the checkpoint writer below.
    provenance["sigma_ccd_px"] = float(best["parameters"].get("sigma_ccd_px", 0.0))
    (args.out / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    # Self-contained checkpoint, with explicit geometry rather than a stale upstream chain.
    # The fitted global registration goes out with it so a reloaded checkpoint keeps the
    # camera correspondence it was fitted with (issue #237).  ``ramp_matrix`` carries the
    # effective matrix so ``inv(ramp_matrix) == inv(base) @ G`` reloads directly.
    (args.out / "frozen_parameters.json").write_text(json.dumps(dict(
        **best["parameters"], ramp_matrix=exported_matrix.tolist(), grid_size=comps.old.grid_size,
        fft_size=comps.old.fft_size, pixel_samples=comps.old.pixel_samples,
        registration_shift_ccd_px=comps.model.registration_value), indent=2), encoding="utf-8")
    print(f"selected iteration={best['iteration']} using training loss only", flush=True)


def fit_adaptive(args):
    """Warm transfer with an explicit validation-selected baseline and fallback.

    Validation phases select the stage, so final independent acceptance remains
    the paired CCD experiment. Raw data and every candidate stay on disk.
    """
    from types import SimpleNamespace
    from .adaptation import needs_full_adaptation, validation_quality

    args.out.mkdir(parents=True, exist_ok=False)
    candidates = []

    def run_candidate(mode, checkpoint):
        options = dict(vars(args), adaptation=mode, checkpoint=checkpoint,
                       out=args.out / mode)
        if mode == "alignment":
            options["iterations"] = min(args.iterations, 30)
        fit(SimpleNamespace(**options))
        root = options["out"]
        result = json.loads((root / "result.json").read_text(encoding="utf-8"))
        rows = json.loads((root / "provenance.json").read_text(encoding="utf-8"))["rows"]
        with (root / "iterations.pkl").open("rb") as stream:
            initial = pickle.load(stream)[0]
        # Iteration zero includes fresh geometry and NNLS radiometry, but no
        # optical update. Keep it when transfer degrades validation phases.
        for label, entry in ((mode + "_initial", initial), (mode, result)):
            candidates.append(dict(stage=label, directory=str(root.resolve()),
                                   quality=validation_quality(entry, rows), entry=entry))
        return min(candidates, key=lambda item: item["quality"]["score"])

    selected = run_candidate("alignment", args.checkpoint)
    escalated = needs_full_adaptation(selected["quality"])
    if escalated:
        selected = run_candidate("full", args.checkpoint)
    root = Path(selected["directory"])
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((root / "frozen_parameters.json").read_text(encoding="utf-8"))
    checkpoint.update(selected["entry"]["parameters"])
    (args.out / "frozen_parameters.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    (args.out / "result.json").write_text(json.dumps(selected["entry"], indent=2), encoding="utf-8")
    (args.out / "provenance.json").write_text(json.dumps(provenance | dict(
        adaptation="auto", selection="validation phases; final acceptance requires paired CCD"), indent=2), encoding="utf-8")
    decision = dict(selected_stage=selected["stage"], escalated=escalated,
                    escalation_heuristic=dict(correlation_below=.85, relative_l2_above=.25),
                    candidates=[{k: v for k, v in c.items() if k != "entry"} for c in candidates],
                    **clock())
    (args.out / "adaptation.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"Automatic adaptation selected {selected['stage']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early-stop after this many iterations without an improvement larger than --min-delta",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum loss improvement (absolute) that resets the early-stop patience",
    )
    parser.add_argument("--span", type=int, default=160)
    parser.add_argument("--adaptation", choices=("auto", "alignment", "full"), default="auto")
    parser.add_argument("--amplitude-center-lr", type=float, help="Override illumination position learning rate (panel px/update)")
    parser.add_argument("--radiometry", choices=("shared", "exposure"), default="exposure")
    parser.add_argument("--learn-geometry", action="store_true",
                        help="Fit a small bounded 2x2 rotation/shear/anisotropy correction "
                             "on top of the measured ramp matrix. Off by default: the "
                             "effective matrix stays bit-identical to the measured one.")
    parser.add_argument("--learn-psf", action="store_true",
                        help="Fit one bounded static CCD Gaussian PSF (sigma in CCD px) "
                             "instead of the frozen checkpoint blur. Off by default: the "
                             "frozen kernel is used unchanged.")
    parser.add_argument("--psf-sigma-init", type=float, default=None,
                        help="Initial PSF sigma in CCD px (default: checkpoint sigma if > 0, else 1.0)")
    parser.add_argument("--psf-sigma-max", type=float, default=8.0,
                        help="Upper bound for the learned PSF sigma in CCD px")
    parser.add_argument("--record-responses", action="store_true",
                        help="Record detached physical-stage response statistics in every iteration (adds diagnostic overhead)")
    parser.add_argument("--debug-weights", action="store_true",
                        help="Debug mode: save every trainable parameter per iteration to "
                             "<out>/debug_weights (weights_history.jsonl + weights_trajectory.npz "
                             "+ weights_summary.json). Off by default; purely additive.")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto",
                        help="auto = CUDA when available else CPU; cpu pins the historical numbers")
    args = parser.parse_args()
    if args.iterations < 0 or args.span < 20 or args.span % 2:
        parser.error("iterations >= 0; span must be even and >= 20")
    fit(args)


if __name__ == "__main__":
    main()
