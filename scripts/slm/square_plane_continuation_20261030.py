"""One-session plane-to-square continuation; remove after the report is final.

The first displayed shaping phase is flat (steering carrier only). Subsequent
proposals interpolate raw radians toward irradiance transport, not wrapped
phases. CCD data, not the forward prediction, decide the endpoint. No GS is used.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward, load_frozen_parameters,
)
from ao_shaping.tools.slm.transfer.capture import clock, to_panel
from ao_shaping.tools.slm.transfer.cold_start import cold_phase
from ao_shaping.tools.slm.transfer.geometric_seed import square_transport_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--feedback", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    zero = np.array([1267., 943.])
    centre = zero + [650., 0.]
    yy, xx = np.mgrid[300:1588:4, 1300:2540:4]
    xy = np.stack((xx, yy), axis=-1).astype(float)
    params = load_frozen_parameters(args.checkpoint).effective_parameters()
    forward = FrozenDefocusForward(params, xy, device="cpu")
    plane, provenance = cold_phase(forward, centre, kind="plane")
    endpoint = square_transport_phase(forward, centre, 1000.)
    direction = endpoint - plane
    w = forward.amplitude.detach().cpu().numpy() ** 2
    direction -= (direction * w).sum() / w.sum()
    rms = float(np.sqrt((direction ** 2 * w).sum() / w.sum()))
    qcorner = np.array([[x, y] for x in (-500., 500.) for y in (-500., 500.)])
    frequencies = np.linalg.solve(np.array(params.ramp_matrix),
                                  (centre + qcorner - params.origin_xy).T).T
    if np.abs(frequencies).max() >= .5 / forward.pitch:
        raise ValueError("target corners exceed control Nyquist")
    args.out.mkdir(parents=True, exist_ok=False)
    config = dict(checkpoint=str(args.checkpoint.resolve()), centre=centre.tolist(),
                  zero=zero.tolist(), side=1000., direction_rms=rms,
                  corner_frequencies=frequencies.tolist(), initial=provenance,
                  alpha=[0., .125, .25, .5, .75, 1.],
                  interpretation="plane start; measured continuation, no analytic initialization",
                  execute=args.execute)
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    np.savez_compressed(args.out / "proposals.npz", plane=plane, direction=direction,
                        camera_xy=xy)
    print(json.dumps(config), flush=True)
    if not args.execute:
        return
    from ao_shaping.tools.slm.transfer.hardware import prepare_camera_sdk
    prepare_camera_sdk("daheng")
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms
    from ao_shaping.drivers.slm.santec import Santec

    # Native-pixel aperture and one scalar background for every statistic.
    sy, sx = np.mgrid[:1944, :2592]
    box = (np.abs(sx-centre[0]) < 500) & (np.abs(sy-centre[1]) < 500)
    core = (np.abs(sx-centre[0]) < 480) & (np.abs(sy-centre[1]) < 480)
    window = (sx >= 1300) & (sx < 2540) & (sy >= 300) & (sy < 1588)
    background_mask = window & ~box
    records = []
    last_signal = None

    def measure(cam, slm, alpha, exposure, label, phase=None):
        nonlocal last_signal
        phase = plane + alpha * direction if phase is None else phase
        gray = slm.create_phase_from_array(to_panel(phase))
        slm.display_data(gray)
        cam.reset_exposure_time(exposure)
        time.sleep(.8)
        # Discard after each change to avoid queued frames from the previous phase.
        cam.get_numpy_image(n_sample=1, skip_first=False)
        raw = np.stack([np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False)).copy()
                        for _ in range(5)])
        frame = raw.mean(0)
        bg = float(np.median(frame[background_mask]))
        signal = np.maximum(frame-bg, 0)
        last_signal = signal
        blurred = gaussian_filter(signal, 8.)
        vals = signal[core]
        mean = float(vals.mean())
        fullscale = 4095.
        # Coarse coverage rejects background-only "uniform" frames. Record the
        # absolute threshold so coverage cannot be detached from its noise floor.
        # Spatial stray signal is not temporal sensor noise. Estimate from
        # repeated frames at the same phase instead of the outside-box std.
        noise = float(np.sqrt(raw[:, background_mask].var(axis=0).mean()/len(raw)))
        threshold = max(3.*noise, .2*mean)
        filled = blurred >= threshold
        overlap = float((filled & box).sum() / max((filled | box).sum(), 1))
        row = dict(label=label, alpha=alpha, exposure_ms=float(get_camera_exposure_ms(cam)),
                   background=bg, noise=noise, threshold=threshold, mean=mean,
                   cv=float(vals.std()/max(mean, 1e-9)),
                   ee=float(signal[box].sum()/max(signal.sum(), 1e-9)),
                   coverage=float(filled[core].mean()), iou=overlap,
                   core_peak=float(raw[:, core].max()),
                   clipped_core=int((raw[:, core] >= .98*fullscale).sum()),
                   clipped_full=int((raw >= .98*fullscale).sum()), **clock())
        row["file"] = f"capture_{len(records):04d}.npz"
        np.savez_compressed(args.out / row["file"], raw_frames=raw,
                            model_phase=phase, panel_gray=gray, camera_xy=xy)
        records.append(row)
        (args.out / "manifest.json").write_text(json.dumps(records, indent=2))
        print(json.dumps(row), flush=True)
        return row

    def feedback_phase(phase, signal, rate):
        """Compose measured far-field marginal CDFs with current phase slopes."""
        # The transport's static correction is unchanged by the CDF update.
        static = -params.defocus_diopter*params.diopter_scale*forward.z4.cpu().numpy()
        if forward.residual_basis is not None:
            static -= np.einsum("i,ijk->jk", forward.residual_values.cpu().numpy(),
                                forward.residual_basis.cpu().numpy())
        optical = phase-static
        matrix = np.asarray(params.ramp_matrix)
        qcentre = np.linalg.solve(matrix, centre-params.origin_xy)
        candidate = static.copy().astype(float)
        # Exclude the undiffracted spot, not the target's dim interior.
        valid = ((sx-1267.)**2+(sy-943.)**2 > 120.**2) & (sx > 1310)
        valid &= (sx < 2540) & (sy > 340) & (sy < 1545)
        weights = gaussian_filter(signal, 3.) * valid
        for axis in (0, 1):
            vector = matrix[:, axis]
            norm = np.linalg.norm(vector)
            coord = ((sx-centre[0])*vector[0]+(sy-centre[1])*vector[1])/norm
            edges = np.arange(-700., 704., 4.)
            mass, _ = np.histogram(coord, bins=edges, weights=weights)
            cdf = (np.cumsum(mass)-.5*mass)/max(mass.sum(), 1e-9)
            slope = np.gradient(optical, forward.pitch, axis=1 if axis == 0 else 0)/(2*np.pi)
            marginal_w = w.sum(axis=0 if axis == 0 else 1)
            old = (slope*w).sum(axis=0 if axis == 0 else 1)/np.maximum(marginal_w, 1e-12)
            position = (old-qcentre[axis])*norm
            mapped = np.interp(position, .5*(edges[1:]+edges[:-1]), cdf, left=0., right=1.)
            new = (1-rate)*old + rate*(qcentre[axis]+1000./norm*(mapped-.5))
            if np.max(np.abs(new)) >= .5/forward.pitch:
                raise ValueError("feedback exceeds control Nyquist")
            profile = 2*np.pi*cumulative_trapezoid(new, dx=forward.pitch, initial=0.)
            candidate += profile[None, :] if axis == 0 else profile[:, None]
        candidate -= ((candidate-plane)*w).sum()/w.sum()
        return candidate.astype(np.float32)

    with create_camera("daheng", 0, exposure_time_ms=.1, bit_depth=12) as cam:
        with Santec(slm_number=1, wavelength=1064, video_mode=0) as slm:
            slm.set_phase_overlay(None)
            try:
                if args.feedback:
                    for alpha in config["alpha"]:
                        measure(cam, slm, alpha, .5, "safe_continuation")
                    phase = endpoint.copy()
                    row = measure(cam, slm, 1., 35., "feedback_reference", phase)
                    for iteration in range(8):
                        if row["clipped_core"]:
                            raise RuntimeError("target clipped: feedback is invalid")
                        phase = feedback_phase(phase, last_signal, .65)
                        row = measure(cam, slm, 1., 35., f"feedback_{iteration+1:02d}", phase)
                    np.save(args.out / "feedback_endpoint.npy", phase)
                    # The plane can only be compared at its own safe exposure.
                    for tag, candidate in [("safe_A1", plane), ("safe_B1", phase),
                                           ("safe_B2", phase), ("safe_A2", plane)]:
                        measure(cam, slm, 1., .5, tag, candidate)
                else:
                    for alpha in config["alpha"]:
                        row = measure(cam, slm, alpha, .717, "continuation")
                        # Only increase after observing the target peak; the separate
                        # zero order is archived but never part of the target aperture.
                        if row["core_peak"] < 800:
                            exposure = min(100., .717*1600./max(row["core_peak"]-row["background"], 20.))
                            measure(cam, slm, alpha, exposure, "continuation_metered")
                    candidates = [r for r in records if r["clipped_core"] == 0]
                    best = max(candidates, key=lambda r: r["coverage"])
                    alpha = best["alpha"]
                    exp = best["exposure_ms"]
                    # ABBA at one exposure, with a repeated endpoint after the scan.
                    for tag, a in [("A1_plane", 0.), ("B1_square", alpha),
                                   ("B2_square", alpha), ("A2_plane", 0.)]:
                        measure(cam, slm, a, exp, tag)
                    np.save(args.out / "measured_endpoint.npy", plane+alpha*direction)
            finally:
                slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
                time.sleep(.3)
    (args.out / "released.json").write_text(json.dumps(clock()))


if __name__ == "__main__":
    main()
