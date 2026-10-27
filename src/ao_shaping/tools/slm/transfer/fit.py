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
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


from ao_shaping.tools.slm.transfer.capture import clock
from ao_shaping.tools.slm.transfer.device import DEVICE_CHOICES, resolve_device


def spot(row):
    if "spot_xy" in row:
        return np.asarray(row["spot_xy"], dtype=float)
    return np.array([row["spot"][f"centroid_{axis}"] for axis in ("x", "y")])


def geometry(rows):
    """Fit full 2D ramp displacement; preserve sign and both CCD axes."""
    slopes = {"x": [], "y": []}
    for row in rows:
        if not row["label"].startswith("ramp_"):
            continue
        _, axis, period, sign = row["label"].split("_")
        if int(sign) != 1:
            continue
        negative = next((other for other in rows if other["label"] == f"ramp_{axis}_{period}_-1"
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


def valid(row):
    return (row.get("settled", {}).get("settled", True)
            and row.get("exposure_status", row.get("exposure_setting", {}).get("status", "ok")) == "ok"
            and float(row.get("peak", 255)) < 250
            and float(row["exposure_ms"]) > 0)


def load_dataset(directory, size, span, samples=None):
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
    selected = [row for row in usable if row.get("split") in {"train", "test", "validation"}]
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
        images.append(frame[xy[..., 1].astype(int), xy[..., 0].astype(int)] / row["exposure_ms"])
        phases.append(phase)
    train = np.array([row["split"] == "train" for row in selected])
    if not train.any() or train.all():
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


def fit(args):
    adaptation = getattr(args, "adaptation", "full")
    if adaptation == "auto":
        return fit_adaptive(args)
    if adaptation not in {"full", "alignment"}:
        raise ValueError("unknown adaptation mode")
    import torch
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward, load_frozen_parameters,
    )
    from ao_shaping.algorithm.signal_processing.physical_forward_fit import PhysicalForwardFit

    torch.set_num_threads(4)
    old = load_frozen_parameters(args.checkpoint).effective_parameters()
    device = resolve_device(getattr(args, "device", None))
    rows, phases, images, xy, matrix, ramp_samples, train = load_dataset(
        args.dataset, old.grid_size, args.span, samples=getattr(args, "samples", None))
    # ``load_dataset`` centres the camera window on the **measured** zero order
    # of this capture. A warm checkpoint's origin is the *previous* bench's
    # zero, and inheriting it after a realignment displaces the predicted field
    # by that difference: the NNLS radiometry then clips the gain to zero, the
    # prediction becomes a constant and every gradient vanishes, so the
    # alignment stage deadlocks instead of converging (2026-10-27: a 76 px
    # zero-order shift produced gain=0 and "nonfinite validation quality").
    # Geometry here is measured, not inherited.
    manifest_rows = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    reference_row = next(row for row in manifest_rows
                         if row["label"] == "reference_start" and valid(row))
    measured_origin = spot(reference_row)
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = dict(dataset=str(args.dataset.resolve()), checkpoint=str(args.checkpoint.resolve()),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      manifest_sha256=hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest(),
                      sample_source="memory" if getattr(args, "samples", None) is not None else "dataset",
                      device=device, device_requested=getattr(args, "device", None),
                      iterations=args.iterations, ramp_matrix=matrix.tolist(), ramp_samples=ramp_samples,
                      record_responses=bool(getattr(args, "record_responses", False)),
                      adaptation=adaptation,
                      amplitude_center_lr=getattr(args, "amplitude_center_lr", None),
                      radiometry=getattr(args, "radiometry", "exposure"),
                      exposure_assumption="counts/exposure_ms; linearity and illuminated offset not independently calibrated",
                      window_step_px=2, window_span_px=args.span, rows=rows, **clock())
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    np.savez_compressed(args.out / "inputs.npz", phases=phases, measured=images, camera_xy=xy, train=train)
    initial = replace(old, ramp_matrix=tuple(tuple(x) for x in matrix))
    frozen = FrozenDefocusForward(initial, xy, device=device)
    basis = {4: frozen.z4.detach().cpu().numpy()}
    basis.update({j: b.detach().cpu().numpy() for j, b in zip(frozen.residual_modes, frozen.residual_basis)})
    model = PhysicalForwardFit(phases, images, xy, matrix, 0., train, old.amplitude_center_xy,
                               learn_amplitude=True, initial_defocus=old.defocus_diopter,
                               learn_residual=True, fft_size=old.fft_size, pixel_samples=old.pixel_samples,
                               basis_arrays=basis, batch_size=4, device=device)
    response_records = []
    if getattr(args, "record_responses", False):
        from ao_shaping.utils.image.model_response import summarize_response

        def record_response(name, value, batch_start):
            response_records.append(dict(stage=name, batch_start=batch_start,
                                         **clock(), **summarize_response(value)))

        model.diagnostic_callback = record_response
    with torch.no_grad():
        model.origin.copy_(torch.tensor(measured_origin))
        model.log_width.copy_(torch.tensor(old.amplitude_width_xy).log())
        model.residual.copy_(torch.tensor([old.residual_noll_radians.get(j, 0) for j in model.residual_modes]))
    from .adaptation import configure_adaptation
    configure_adaptation(model, adaptation)
    center_lr = getattr(args, "amplitude_center_lr", None)
    if center_lr is not None:
        if not np.isfinite(center_lr) or center_lr <= 0:
            raise ValueError("amplitude_center_lr must be finite and positive")
        for group in model.optimizer.param_groups:
            if any(parameter is model.center for parameter in group["params"]):
                group["lr"] = center_lr
    target = model.images[model.train]
    target_mean = target.mean((1, 2), keepdim=True).clamp_min(1e-9)
    target_cv = target.std((1, 2), correction=0) / target_mean[:, 0, 0]
    best_loss, history = float("inf"), []
    kernel = frozen._blaze_kernel
    exposure_values = np.asarray([row["exposure_ms"] for row in rows])
    from .radiometry import fit_radiometry
    for iteration in range(args.iterations + 1):
        started = clock()
        response_records.clear()
        model.optimizer.zero_grad()
        prediction = model.prediction()
        if kernel is not None:
            prediction = torch.nn.functional.conv2d(prediction[:, None], kernel, padding=12)[:, 0]
        # Estimate shared radiometry on training pixels, then use that same
        # prediction in both optimization and reported metrics. Never use
        # validation pixels to choose gain or additive background.
        optical_prediction = prediction.detach().cpu().numpy()
        _, radiometry = fit_radiometry(optical_prediction, images, exposure_values, train,
                                      mode=getattr(args, "radiometry", "exposure"))
        if iteration == 0 and radiometry["gain"] <= 0:
            raise ValueError(
                "radiometry gain collapsed to zero at iteration 0: the checkpoint's "
                "field does not overlap the measured window (origin/geometry mismatch). "
                "The loss is then independent of the field, so no parameter can move; "
                "re-anchor the geometry or refit from a fresh capture. "
                f"(measured origin {np.round(measured_origin, 1).tolist()}, "
                f"checkpoint origin {np.round(old.origin_xy, 1).tolist()})")
        scale, offset = radiometry["gain"], radiometry["background"]
        exposure_tensor = torch.as_tensor(exposure_values[:, None, None], dtype=prediction.dtype, device=prediction.device)
        prediction = prediction * scale + offset + radiometry["bias_counts"]/exposure_tensor
        p = prediction[model.train]
        normalized = p / p.mean((1, 2), keepdim=True).clamp_min(1e-9)
        cv = p.std((1, 2), correction=0) / p.mean((1, 2)).clamp_min(1e-9)
        target_normalized = target / target_mean
        relative_mse = (((normalized-target_normalized)**2).mean((1, 2))
                        / target_normalized.square().mean((1, 2)).clamp_min(1e-9)).mean()
        pc, tc = p.flatten(1), target.flatten(1)
        pc, tc = pc-pc.mean(1, keepdim=True), tc-tc.mean(1, keepdim=True)
        correlation_loss = (1-(pc*tc).sum(1)/(pc.norm(dim=1)*tc.norm(dim=1)).clamp_min(1e-9)).mean()
        loss = correlation_loss + .25 * relative_mse + .25 * ((cv-target_cv)/target_cv.clamp_min(.1)).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss; prior snapshots retained")
        params = dict(defocus_diopter=float(model.d.detach()), origin=model.origin.detach().tolist(),
                      amplitude_center=model.center.detach().tolist(),
                      amplitude_width=model.log_width.detach().exp().tolist(),
                      residual_noll_radians={str(j): float(v) for j, v in zip(model.residual_modes, model.residual.detach())},
                      gain=1., background=0., sigma_ccd_px=old.sigma_ccd_px)
        predicted = prediction.detach().cpu().numpy()
        params.update(radiometry)
        entry = dict(iteration=iteration, loss=float(loss.detach()), parameters=params,
                     metrics=metrics(predicted, images), **started, end_monotonic_ns=__import__("time").monotonic_ns())
        if getattr(args, "record_responses", False):
            entry["responses"] = list(response_records)
        history.append(entry)
        np.savez_compressed(args.out / f"iteration_{iteration:04d}.npz", prediction=predicted)
        with (args.out / "iterations.pkl").open("wb") as stream:
            pickle.dump(history, stream, protocol=5)
        if entry["loss"] < best_loss:
            best_loss = entry["loss"]
            (args.out / "result.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        if iteration % 10 == 0:
            print(f"{iteration}: train loss={entry['loss']:.5f}", flush=True)
        if iteration < args.iterations:
            loss.backward()
            model.optimizer.step()
            with torch.no_grad():
                model.d.clamp_(-2, 2)
                model.center.clamp_(0, 1920)
                model.log_width.clamp_(float(np.log(80)), float(np.log(800)))
    best = json.loads((args.out / "result.json").read_text())
    # Self-contained checkpoint, with explicit geometry rather than a stale upstream chain.
    (args.out / "frozen_parameters.json").write_text(json.dumps(dict(
        **best["parameters"], ramp_matrix=matrix.tolist(), grid_size=old.grid_size,
        fft_size=old.fft_size, pixel_samples=old.pixel_samples), indent=2), encoding="utf-8")
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
    parser.add_argument("--span", type=int, default=160)
    parser.add_argument("--adaptation", choices=("auto", "alignment", "full"), default="auto")
    parser.add_argument("--amplitude-center-lr", type=float, help="Override illumination position learning rate (panel px/update)")
    parser.add_argument("--radiometry", choices=("shared", "exposure"), default="exposure")
    parser.add_argument("--record-responses", action="store_true",
                        help="Record detached physical-stage response statistics in every iteration (adds diagnostic overhead)")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto",
                        help="auto = CUDA when available else CPU; cpu pins the historical numbers")
    args = parser.parse_args()
    if args.iterations < 0 or args.span < 20 or args.span % 2:
        parser.error("iterations >= 0; span must be even and >= 20")
    fit(args)


if __name__ == "__main__":
    main()
