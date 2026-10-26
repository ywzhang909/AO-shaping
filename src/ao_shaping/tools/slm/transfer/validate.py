"""Physical-model multishape synthesis and fixed-mask paired CCD validation.

Only --execute opens devices. Each run uses a new directory. This script reads
the self-contained physical checkpoint from fit_transfer_multishape.py.

The A–B–B–A acceptance is optional (``abba``): with it off the candidate is
accepted on the optimizer's own loss history and every summary entry carries
``acceptance="model_only"``, so no artefact can imply a CCD verdict that was
never measured. An explicit ``ccd_check`` instead measures sequential A/B
captures; optional nonmonotone exploration retains a separate best phase.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np


from ao_shaping.tools.slm.transfer.capture import clock, to_panel
from ao_shaping.tools.slm.transfer.device import DEVICE_CHOICES, resolve_device


from ao_shaping.algorithm.signal_processing.multishape_objective import (
    ShapeObjective, anchored_intensity, gs_seed, image_metrics, target_at,
)
from ao_shaping.utils.image.target import POLYGON_SHAPE_CHOICES


def replay_seed(directory, shape, xy, target):
    records = json.loads((directory/"manifest.json").read_text())
    row = next(r for r in records if r["label"] == shape+"_final")
    with np.load(directory/row["file"]) as saved:
        if not np.array_equal(saved["camera_xy"], xy) or not np.array_equal(saved["target"], target):
            raise ValueError("replay grid/target differs from current fixed geometry")
        return saved["model_phase"].copy()



def model_only_decision(history) -> dict:
    """Accept a candidate on the model's own loss, with no CCD verdict measured.

    The A–B–B–A blocks exist to let the *measurement* decide. When they are
    switched off (``abba=False``) the only evidence left is the optimizer's loss
    history, so the rule is deliberately weak and explicit: accept when the last
    update improved on the previous one. With a single update there is no
    previous value, hence no evidence, hence no acceptance.

    The returned dict carries ``acceptance="model_only"`` so a summary entry can
    never be mistaken for a paired CCD acceptance, and the two loss values so an
    audit can recompute ``accepted`` without a device.
    """
    losses = [float(entry["loss"]) for entry in history]
    previous = losses[-2] if len(losses) >= 2 else None
    last = losses[-1] if losses else None
    accepted = previous is not None and last is not None and last < previous
    return dict(acceptance="model_only", accept_metric="model", accepted=bool(accepted),
                model_loss_previous=previous, model_loss_last=last)


def run(args):
    import torch
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        DifferentiableUniformityOptimizer, FrozenDefocusForward, FrozenDefocusParameters, recentre_window,
    )
    torch.set_num_threads(4)
    device = resolve_device(getattr(args, "device", None))
    d = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    if d.get("local_response_only") and args.mode != "anchored":
        raise ValueError("local-response checkpoint requires CCD anchored mode")
    params = FrozenDefocusParameters.from_dicts(d, d["ramp_matrix"], grid_size=d["grid_size"],
                                               fft_size=d["fft_size"], pixel_samples=d["pixel_samples"])
    params = params.effective_parameters()
    explore_temperature = float(getattr(args, "exploration_temperature", 0.))
    explore_budget = float(getattr(args, "exploration_budget", .25))
    if not np.isfinite([explore_temperature, explore_budget]).all() or explore_temperature < 0 or getattr(args, "exploration_window", 4) < 1 or explore_budget < 0:
        raise ValueError("nonnegative exploration temperature/budget and positive window required")
    if explore_temperature > 0 and (not getattr(args, "ccd_check", False) or getattr(args, "adaptive_exposure", False) or getattr(args, "abba", False)):
        raise ValueError("exploration requires sequential CCD checks and fixed exposure across blocks")
    bit_depth = int(getattr(args, "bit_depth", 8))
    full_scale = float(2**bit_depth - 1)
    counts_scale = float(2**(bit_depth-8))
    # Carrier displacement uses both camera axes. The target is kept away from zero order.
    shift = np.asarray(params.ramp_matrix)[:, 0]/args.period
    centre = np.asarray(args.zero_xy)+shift
    if getattr(args, "target_xy", None) is not None:
        centre = np.asarray(args.target_xy, dtype=float)
    xy = recentre_window(tuple(args.zero_xy), half_span_px=getattr(args, "window_span", 220.),
                         step_px=getattr(args, "window_step", 2.))
    forward = FrozenDefocusForward(params, xy, device=device)
    if not args.execute:
        print(json.dumps(dict(zero_xy=args.zero_xy, target_xy=centre.tolist(), shapes=args.shapes), indent=2))
        return
    prepared, seed_provenance = [], {}
    for shape in args.shapes:
        objective = ShapeObjective(target_at(xy, shape, centre, args.side),
                                   cv_only=getattr(args, "objective", "shape") == "cv",
                                   formation=getattr(args, "objective", "shape") == "formation")
        if getattr(args, "core_margin", None) is not None:
            from scipy.ndimage import binary_erosion
            cells = max(1, int(np.ceil(args.core_margin/getattr(args, "window_step", 2.))))
            objective.core = binary_erosion(objective.box, iterations=cells)
        if getattr(args, "zero_radius", 0.) > 0:
            radius = np.linalg.norm(xy - np.asarray(args.zero_xy), axis=-1)
            objective.core &= radius > args.zero_radius
            if not objective.core.any():
                raise ValueError("zero-order exclusion removes the complete target core")
        box = objective.box
        if box[0].any() or box[-1].any() or box[:, 0].any() or box[:, -1].any():
            raise ValueError("target intersects camera window boundary; adjust period/side")
        if args.seed_run:
            from .seeding import accepted_seed
            source = json.loads((args.seed_run / "config.json").read_text(encoding="utf-8"))
            if float(source["side"]) != float(args.side):
                raise ValueError("accepted seed target size differs from requested target")
            phase, seed_provenance[shape] = accepted_seed(args.seed_run, shape, forward, xy,
                objective.target, policy=getattr(args, "seed_policy", "strict"), centre=centre)
        elif getattr(args, "initial_phase", "gs") == "transport":
            if shape != "square":
                raise ValueError("transport initialization currently supports square only")
            from .geometric_seed import square_transport_phase
            phase = square_transport_phase(forward, centre, args.side)
            seed_provenance[shape] = dict(kind="transport", gs_calls=0, historical_phase=False,
                                          static_correction=True, side_ccd_px=args.side)
        elif getattr(args, "initial_phase", "gs") != "gs":
            from .cold_start import cold_phase
            phase, seed_provenance[shape] = cold_phase(forward, centre,
                kind=args.initial_phase, seed=getattr(args, "random_seed", 102626))
        else:
            phase = gs_seed(forward, centre, args.side, shape)
        if not np.isfinite(phase).all():
            raise ValueError("nonfinite seed phase")
        prepared.append((shape, objective, phase))
    from .hardware import prepare_camera_sdk
    prepare_camera_sdk(args.camera)
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms, get_camera_exposure_range
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.slm.panel.focus_probe import adapt_exposure
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    args.out.mkdir(parents=True, exist_ok=False)
    import hashlib
    abba = bool(getattr(args, "abba", False))
    (args.out/"config.json").write_text(json.dumps(dict(checkpoint=str(args.checkpoint.resolve()), zero_xy=args.zero_xy,
                                                       target_xy=centre.tolist(), shapes=args.shapes,
                                                       period=args.period, side=args.side,
                                                       window_span=getattr(args, "window_span", 220.),
                                                       window_step=getattr(args, "window_step", 2.),
                                                       zero_radius=getattr(args, "zero_radius", 0.),
                                                       meter_target=getattr(args, "meter_target", False),
                                                       bit_depth=bit_depth, full_scale=full_scale,
                                                       counts_scale=counts_scale,
                                                       core_margin=getattr(args, "core_margin", None),
                                                       checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                                       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                       gs_iterations=0 if args.seed_run or getattr(args, "initial_phase", "gs") != "gs" else 80,
                                                       initial_phase=getattr(args, "initial_phase", "gs"),
                                                       abba=abba,
                                                       acceptance="nonmonotone_ccd" if explore_temperature > 0 else ("paired_abba" if abba else ("sequential_ccd" if getattr(args, "ccd_check", False) else "model_only")),
                                                       ccd_check=getattr(args, "ccd_check", False),
                                                       exploration_temperature=explore_temperature,
                                                       exploration_budget=getattr(args, "exploration_budget", .25),
                                                       exploration_window=getattr(args, "exploration_window", 4),
                                                       exploration_seed=getattr(args, "random_seed", 102626),
                                                       fixed_exposure_ms=getattr(args, "fixed_exposure_ms", None),
                                                       device=device,
                                                       device_requested=getattr(args, "device", None),
                                                       accept_metric=getattr(args, "accept_metric", "loss"),
                                                       objective=getattr(args, "objective", "shape"),
                                                       seed_provenance=seed_provenance,
                                                       phase_lr=getattr(args, "phase_lr", .01),
                                                       phase_rms_cap=getattr(args, "phase_rms_cap", .2),
                                                       adaptive_step=getattr(args, "adaptive_step", False),
                                                       exposure_policy="per_block" if getattr(args, "adaptive_exposure", False) else "per_shape",
                                                       camera=args.camera, cam_id=args.cam_id, slm_number=args.slm_number, wavelength_nm=1064,
                                                       mode=args.mode, seed_run=str(args.seed_run) if args.seed_run else None,
                                                       blocks=args.blocks, updates=args.updates, frames=args.frames,
                                                       **clock()), indent=2), encoding="utf-8")
    records, iterations, summary = [], [], []
    camera_options = {"bit_depth": bit_depth} if args.camera == "daheng" else {}
    with create_camera(args.camera, args.cam_id, exposure_time_ms=.02, **camera_options) as cam:
        with Santec(slm_number=args.slm_number, wavelength=1064, video_mode=0) as slm:
            slm.set_phase_overlay(None)
            floor, ceiling = get_camera_exposure_range(cam)
            configured = configured_daheng_exposure_limit() if args.camera == "daheng" else 0.
            if configured > 0:
                ceiling = min(ceiling, configured)
            exposure = max(floor, .02)

            def capture(label, phase, objective):
                panel = to_panel(phase)
                gray = slm.create_phase_from_array(panel)
                slm.display_data(gray)
                time.sleep(.6)
                frames, stamps = [], []
                for _ in range(args.frames):
                    stamp = clock()
                    frame = np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False))
                    frames.append(frame.copy())
                    stamps.append(dict(**stamp, end_monotonic_ns=time.monotonic_ns(), exposure_ms=float(get_camera_exposure_ms(cam))))
                raw = np.stack(frames)
                image = raw.mean(0)
                if xy[..., 0].min() < 0 or xy[..., 1].min() < 0 or xy[..., 0].max() >= image.shape[1] or xy[..., 1].max() >= image.shape[0]:
                    raise ValueError("camera window extends beyond sensor")
                sampled = image[xy[..., 1].astype(int), xy[..., 0].astype(int)]
                per_frame = [image_metrics(frame[xy[..., 1].astype(int), xy[..., 0].astype(int)], objective) for frame in raw]
                row = dict(label=label, file=f"capture_{len(records):04d}.npz", frames=stamps,
                           metrics=image_metrics(sampled, objective), per_frame=per_frame,
                           peak=float(raw.max()), core_peak=float(sampled[objective.core].max()),
                           saturation_pixels=int((raw >= full_scale*250/255).sum()),
                           bit_depth=bit_depth, counts_scale=counts_scale, full_scale=full_scale,
                           exposure_ms=float(get_camera_exposure_ms(cam)), **clock())
                with torch.no_grad():
                    loss, _ = objective.evaluate(torch.tensor(sampled/(row["exposure_ms"]*counts_scale), dtype=torch.float32), torch.zeros_like(torch.tensor(phase)))
                row["loss"] = float(loss)
                np.savez_compressed(args.out/row["file"], raw_frames=raw, panel_gray=gray,
                                    model_phase=phase, panel_phase=panel, camera_xy=xy,
                                    target=objective.target, core=objective.core)
                records.append(row)
                with (args.out/"capture_records.pkl").open("wb") as stream:
                    pickle.dump(records, stream, protocol=5)
                (args.out/"manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
                if (sampled[objective.core].max() if getattr(args, "meter_target", False) else raw.max()) >= full_scale*250/255:
                    raise RuntimeError("saturated capture retained; stop paired comparison")
                return row

            try:
                def tune_exposure(label, phase, start_ms, objective):
                    slm.display_data(slm.create_phase_from_array(to_panel(phase)))
                    time.sleep(.6)
                    exposure_reads, exposure_stamps = [], []
                    np.save(args.out/f"{label}_exposure_phase.npy", phase)

                    def read_exposure(_frames=exposure_reads, _stamps=exposure_stamps):
                        _stamps.append(dict(**clock(), exposure_ms=float(get_camera_exposure_ms(cam))))
                        raw = np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False)).copy()
                        _frames.append(raw)
                        np.save(args.out/f"{label}_exposure_read_{len(_frames):03d}.npy", raw)
                        (args.out/f"{label}_exposure_reads.json").write_text(json.dumps(_stamps, indent=2), encoding="utf-8")
                        if getattr(args, "meter_target", False):
                            # Meter every physical pixel in the core, not just the
                            # sparse optimization grid; retain the unmasked frame.
                            from scipy.ndimage import map_coordinates
                            yy, xx = np.indices(raw.shape)
                            dx = xy[0, 1, 0] - xy[0, 0, 0]
                            dy = xy[1, 0, 1] - xy[0, 0, 1]
                            mask = map_coordinates(objective.core.astype(float),
                                [(yy-xy[0, 0, 1])/dy, (xx-xy[0, 0, 0])/dx], order=0, mode="constant") > .5
                            return np.where(mask, raw, 0)
                        return raw

                    if getattr(args, "fixed_exposure_ms", None) is not None:
                        from types import SimpleNamespace
                        if not floor <= args.fixed_exposure_ms <= ceiling:
                            raise ValueError("fixed exposure outside current session bounds")
                        cam.reset_exposure_time(args.fixed_exposure_ms)
                        for _ in range(3):
                            read_exposure()
                        setting = SimpleNamespace(status="ok")
                    else:
                        setting = adapt_exposure(cam.reset_exposure_time, read_exposure, start_ms=start_ms,
                                                 floor_ms=floor, ceiling_ms=ceiling, full_scale=full_scale, max_steps=12)
                    np.savez_compressed(args.out/f"{label}_exposure.npz", raw_frames=np.stack(exposure_reads), model_phase=phase)
                    (args.out/f"{label}_exposure.json").write_text(json.dumps(dict(frames=exposure_stamps, status=setting.status), indent=2), encoding="utf-8")
                    if setting.status != "ok":
                        raise RuntimeError("exposure did not converge")
                    forward.exposure_ms = float(get_camera_exposure_ms(cam))
                    return float(get_camera_exposure_ms(cam))

                for shape, objective, best in prepared:
                    seed_phase = best.copy()
                    exposure = tune_exposure(shape, best, exposure, objective)
                    seed_exposure = exposure
                    seed_record = capture(f"{shape}_seed", best, objective)
                    champion = best.copy()
                    policy = None
                    if explore_temperature > 0:
                        from .exploration import NonmonotoneAcceptance
                        policy = NonmonotoneAcceptance(seed_record["metrics"]["cv"], temperature=explore_temperature,
                            relative_budget=getattr(args, "exploration_budget", .25), seed=getattr(args, "random_seed", 102626))
                    phase_lr = getattr(args, "phase_lr", .01)
                    for block in range(args.blocks):
                        if policy is not None and block > 0 and block % getattr(args, "exploration_window", 4) == 0:
                            best = champion.copy()
                        if block > 0 and getattr(args, "adaptive_exposure", False):
                            exposure = tune_exposure(f"{shape}_{block}", best, exposure, objective)
                        # A1 only feeds the paired decision; "model" mode with the
                        # paired blocks switched off has no use for it, while
                        # "anchored" mode needs it as the optimizer's anchor.
                        anchor_row = (capture(f"{shape}_{block}_A1", best, objective)
                                      if abba or args.mode == "anchored" else None)
                        if args.mode == "anchored":
                            with np.load(args.out/anchor_row["file"]) as saved:
                                anchor = saved["raw_frames"].mean(0)[xy[..., 1].astype(int), xy[..., 0].astype(int)]/(anchor_row["exposure_ms"]*counts_scale)
                            with torch.no_grad():
                                anchor_prediction = forward.predict(best)[0].detach().cpu().numpy()
                        optimizer = DifferentiableUniformityOptimizer(forward, objective, best, lr=phase_lr,
                            step_rms_cap=getattr(args, "phase_rms_cap", .2), device=device)
                        for iteration in range(args.updates):
                            start = clock()
                            if args.mode == "anchored":
                                with torch.no_grad():
                                    current_prediction = forward.predict(optimizer.current_phase)[0].detach().cpu().numpy()
                                optimizer.update(anchored_intensity(anchor, current_prediction, anchor_prediction))
                            else:
                                optimizer.update()
                            iterations.append(dict(shape=shape, block=block, iteration=iteration,
                                                   phase=optimizer.current_phase.copy(), terms=optimizer.loss_history[-1],
                                                   **start, end_monotonic_ns=time.monotonic_ns()))
                            with (args.out/"iterations.pkl").open("wb") as stream:
                                pickle.dump(iterations, stream, protocol=5)
                        candidate = optimizer.current_phase
                        # Fresh optimizer per block: rejected proposals cannot advance its reference.
                        if abba:
                            b1 = capture(f"{shape}_{block}_B1", candidate, objective)
                            b2 = capture(f"{shape}_{block}_B2", candidate, objective)
                            a2 = capture(f"{shape}_{block}_A2", best, objective)
                            if block == 0:
                                # Probe the same actual phase increment with both signs.
                                # This separates a model Jacobian error from autograd errors.
                                # It needs B1/B2, so it belongs to the paired apparatus.
                                minus_phase = 2*best-candidate
                                minus = capture(f"{shape}_direction_minus", minus_phase, objective)
                                capture(f"{shape}_direction_reference", best, objective)
                                with torch.no_grad():
                                    pplus = forward.predict(candidate)[0].detach().cpu().numpy()
                                    pminus = forward.predict(minus_phase)[0].detach().cpu().numpy()
                                    lplus, _ = objective.evaluate(torch.tensor(pplus), torch.zeros_like(torch.tensor(best)))
                                    lminus, _ = objective.evaluate(torch.tensor(pminus), torch.zeros_like(torch.tensor(best)))

                                def observed(row):
                                    with np.load(args.out/row["file"]) as payload:
                                        mean = payload["raw_frames"].mean(0)
                                    return mean[xy[..., 1].astype(int), xy[..., 0].astype(int)]/(row["exposure_ms"]*counts_scale)

                                measured_difference = .5*(observed(b1)+observed(b2))-observed(minus)
                                predicted_difference = pplus-pminus
                                direction = dict(shape=shape, model_loss_plus=float(lplus), model_loss_minus=float(lminus),
                                                 ccd_loss_plus=.5*(b1["loss"]+b2["loss"]), ccd_loss_minus=minus["loss"],
                                                 difference_correlation=float(np.corrcoef(predicted_difference.ravel(), measured_difference.ravel())[0, 1]),
                                                 difference_relative_l2=float(np.linalg.norm(predicted_difference-measured_difference)
                                                                              /max(np.linalg.norm(measured_difference), 1e-9)), **clock())
                                np.savez_compressed(args.out/f"{shape}_direction_difference.npz",
                                                    predicted=predicted_difference, measured=measured_difference)
                                (args.out/f"{shape}_direction.json").write_text(json.dumps(direction, indent=2), encoding="utf-8")
                            from .paired import paired_decision
                            decision = paired_decision(anchor_row, b1, b2, a2,
                                                       metric=getattr(args, "accept_metric", "loss"))
                            entry = dict(shape=shape, block=block, **decision, **clock())
                        elif getattr(args, "ccd_check", False):
                            if anchor_row is None:
                                raise ValueError("sequential CCD check requires anchored mode")
                            b1 = capture(f"{shape}_{block}_B", candidate, objective)
                            metric = "cv" if getattr(args, "accept_metric", "loss") == "cv" else "loss"
                            av = anchor_row["metrics"][metric] if metric == "cv" else anchor_row[metric]
                            bv = b1["metrics"][metric] if metric == "cv" else b1[metric]
                            decision = dict(accepted=bool(bv < av), acceptance="sequential_ccd",
                                            accept_metric=metric, baseline=float(av), candidate=float(bv))
                            if policy is not None:
                                if metric != "cv":
                                    raise ValueError("exploration currently requires core CV acceptance")
                                decision.update(policy.update(av, bv))
                                decision["acceptance"] = "nonmonotone_ccd"
                                if decision["improved_best"]:
                                    champion = candidate.copy()
                                    np.save(args.out/f"{shape}_champion.npy", champion)
                            entry = dict(shape=shape, block=block, **decision, **clock())
                        else:
                            # No measured verdict: the model's own loss history decides,
                            # and the entry says so instead of implying a CCD acceptance.
                            decision = model_only_decision(optimizer.loss_history)
                            entry = dict(shape=shape, block=block, **decision, **clock())
                        accepted = decision["accepted"]
                        entry["phase_lr"] = phase_lr
                        if getattr(args, "adaptive_step", False):
                            phase_lr = min(getattr(args, "phase_lr", .01), phase_lr*1.1) if accepted else max(.001, phase_lr*.5)
                        entry["next_phase_lr"] = phase_lr
                        if accepted:
                            best = candidate.copy()
                            optimizer.accept()
                        summary.append(entry)
                        (args.out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                        print(json.dumps(entry), flush=True)
                    if policy is not None:
                        capture(f"{shape}_exploration_endpoint", best, objective)
                        best = champion.copy()
                        (args.out/f"{shape}_champion.json").write_text(json.dumps(dict(best_cv=policy.best,
                            final_policy="restore measured champion", **clock()), indent=2), encoding="utf-8")
                    if getattr(args, "adaptive_exposure", False):
                        capture(f"{shape}_final_adaptive", best, objective)
                        cam.reset_exposure_time(min(seed_exposure, exposure))
                        exposure = float(get_camera_exposure_ms(cam))
                        time.sleep(.3)
                        capture(f"{shape}_seed_replay", seed_phase, objective)
                    final = capture(f"{shape}_final", best, objective)
                    np.save(args.out/f"{shape}_best.npy", best)
                    print(f"{shape}: seed CV={seed_record['metrics']['cv']:.4f} final CV={final['metrics']['cv']:.4f}", flush=True)
            finally:
                slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
                time.sleep(.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="frozen_parameters.json")
    parser.add_argument("--camera", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--bit-depth", type=int, choices=(8, 12), default=8)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zero-xy", type=float, nargs=2, required=True)
    parser.add_argument("--target-xy", type=float, nargs=2, help="Freeze target centre independently of updated ramp geometry")
    parser.add_argument("--period", type=float, default=64.)
    parser.add_argument("--side", type=float, default=80.)
    parser.add_argument("--window-span", type=float, default=220., help="Camera window half-span in physical CCD pixels")
    parser.add_argument("--window-step", type=float, default=2., help="Camera sampling step in physical CCD pixels")
    parser.add_argument("--zero-radius", type=float, default=0., help="Exclude a fixed measured zero-order disk from core CV (CCD px)")
    parser.add_argument("--core-margin", type=float, help="Physical CCD pixel erosion; default is two sampling cells")
    parser.add_argument("--meter-target", action="store_true", help="Meter target core; permit saturation outside the core and retain full raw frames")
    parser.add_argument("--shapes", nargs="+", choices=POLYGON_SHAPE_CHOICES, default=list(POLYGON_SHAPE_CHOICES))
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--mode", choices=("model", "anchored"), default=None,
                        help="Accepted-phase continuation defaults to anchored; cold synthesis defaults to model")
    parser.add_argument("--seed-run", type=Path, help="Replay accepted phases; exact grid/target match required")
    parser.add_argument("--seed-policy", choices=("strict", "transfer"), default="strict")
    parser.add_argument("--initial-phase", choices=("gs", "plane", "random", "transport"), default="gs")
    parser.add_argument("--random-seed", type=int, default=102626)
    parser.add_argument("--phase-lr", type=float, default=.01)
    parser.add_argument("--phase-rms-cap", type=float, default=.2)
    parser.add_argument("--adaptive-step", action="store_true", help="Shrink learning rate after CCD rejection")
    parser.add_argument("--adaptive-exposure", action="store_true", help="Re-meter between optimization blocks; each A/B block stays at one exposure")
    parser.add_argument("--fixed-exposure-ms", type=float, help="Use a specified exposure within current session limits")
    parser.add_argument("--exploration-temperature", type=float, default=0., help="Absolute CV Metropolis temperature; zero retains strict acceptance")
    parser.add_argument("--exploration-budget", type=float, default=.25, help="Maximum relative CV degradation from retained champion")
    parser.add_argument("--exploration-window", type=int, default=4, help="Restore champion after this many exploration blocks")
    parser.add_argument("--accept-metric", choices=("loss", "cv"), default="loss")
    parser.add_argument("--objective", choices=("shape", "cv", "formation"), default="shape")
    parser.add_argument("--no-abba", dest="abba", action="store_false",
                        help="Skip the A–B–B–A blocks; accept on the model loss only (no CCD verdict)")
    parser.add_argument("--abba", dest="abba", action="store_true", help="Enable optional four-capture CCD acceptance")
    parser.add_argument("--ccd-check", action="store_true", help="Compare one same-exposure CCD candidate with its anchor, without ABBA drift cancellation")
    parser.set_defaults(abba=False)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto",
                        help="auto = CUDA when available else CPU; cpu pins the historical numbers")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.mode = args.mode or ("anchored" if args.seed_run or args.initial_phase != "gs" else "model")
    if args.seed_run and args.initial_phase != "gs":
        parser.error("--seed-run cannot be combined with a cold --initial-phase")
    if args.phase_lr <= 0 or args.phase_rms_cap <= 0 or args.period < 4 or args.side <= 0 or min(args.blocks, args.updates, args.frames) < 1:
        parser.error("invalid sampling or iteration parameters")
    run(args)


if __name__ == "__main__":
    main()
