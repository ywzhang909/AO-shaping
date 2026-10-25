"""Generate known-truth linear-camera probes for device-free transfer tests.

This matched-model test measures numerical recovery, not real-optics fidelity.
Raw simulated images, exposure, truth and deterministic phase seeds are saved.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from .capture import clock, probes


def generate_dataset(out, *, power=1.2, shift=(12., -8.), count=12, seed=102419):
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward, FrozenDefocusParameters,
    )
    if not np.isfinite([power, *shift]).all() or power <= 0:
        raise ValueError("finite positive power and finite shift required")
    out.mkdir(parents=True, exist_ok=False)
    nominal = FrozenDefocusParameters(
        defocus_diopter=.025, origin_xy=(128., 128.),
        amplitude_center_xy=(960., 600.), amplitude_width_xy=(300., 260.),
        residual_noll_radians={j: 0. for j in range(5, 12)},
        gain=1., background=0., sigma_ccd_px=0.,
        ramp_matrix=((0., 5000.), (-5000., 0.)),
        grid_size=32, fft_size=256, pixel_samples=1)
    truth = replace(nominal, amplitude_center_xy=tuple(np.asarray(nominal.amplitude_center_xy)+shift))
    yy, xx = np.mgrid[:256, :256]
    xy = np.stack((xx, yy), axis=-1)
    forward = FrozenDefocusForward(truth, xy, device="cpu")
    cases = list(probes(32, count, seed))
    optical = []
    for _, _, phase in cases:
        optical.append(forward.predict(phase).detach().numpy()[0])
    # Keep headroom even at the largest exposure and power used below.
    scale = 80. / max(float(image.max()) for image in optical)
    rows = []
    for index, ((label, split, phase), image) in enumerate(zip(cases, optical, strict=True)):
        exposure = (.5, 1., 1.5)[index % 3]
        raw = (image*scale*power*exposure).astype(np.float32)
        file = f"sample_{index:04d}.npz"
        # model_phase is authoritative; panel_gray only identifies that schema.
        np.savez_compressed(out / file, raw_frames=np.stack([raw]*3),
                            model_phase=phase, panel_gray=np.zeros((1, 1), np.uint16))
        point = np.array(truth.origin_xy)
        if label.startswith("ramp_"):
            _, axis, period, sign = label.split("_")
            point += np.asarray(truth.ramp_matrix)[:, 0 if axis == "x" else 1]*int(sign)/float(period)
        rows.append(dict(file=file, label=label, split=split, exposure_ms=exposure,
                         peak=float(raw.max()), spot_xy=point.tolist(), **clock()))
    (out / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    nominal = replace(nominal, gain=scale)
    truth = replace(truth, gain=scale*power)
    (out / "nominal.json").write_text(json.dumps(nominal.to_dict(), indent=2), encoding="utf-8")
    (out / "truth.json").write_text(json.dumps(dict(parameters=truth.to_dict(), power=power,
        shift_panel_px=list(shift), synthetic=True, noise="none; linear camera; matched propagation",
        note="Not a hardware capture; spot coordinates and ramp geometry are exact", **clock()), indent=2), encoding="utf-8")
    return out / "nominal.json"
