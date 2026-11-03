"""Measured-only square SPGD starting from flat/transport seed (2026-10-30).

The differentiable/frozen-model path has not produced a uniform 1000 px square.
This one-session script uses paired CCD evaluations (SPGD) on a *small* set of
freeform low-frequency phase cells plus an illumination-matched transport
start, so every gradient step is backed by the actual device instead of the
weak frozen model.

It is a one-off diagnostic + candidate search, not a product runner. Delete it
after the accompanying report is final.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward, load_frozen_parameters,
)
from ao_shaping.tools.slm.transfer.capture import clock, to_panel
from ao_shaping.tools.slm.transfer.cold_start import cold_phase
from ao_shaping.tools.slm.transfer.geometric_seed import square_transport_phase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", choices=("plane", "transport"), default="transport")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--delta", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--grid", type=int, default=16)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(4)

    zero = np.array([1267.0, 943.0])
    center = zero + [650.0, 0.0]
    yy, xx = np.mgrid[300:1588:4, 1300:2540:4]
    xy = np.stack((xx, yy), axis=-1).astype(float)
    params = load_frozen_parameters(args.checkpoint).effective_parameters()
    forward = FrozenDefocusForward(params, xy, device="cpu")
    transport = square_transport_phase(forward, center, 1000.0)
    plane, provenance = cold_phase(forward, center, kind="plane")
    if args.start == "plane":
        start = plane
        start_name = "plane"
    else:
        start = transport
        start_name = "transport"

    direction = transport - plane
    w = forward.amplitude.detach().cpu().numpy() ** 2
    direction -= (direction * w).sum() / w.sum()
    end_rms = float(np.sqrt((direction**2 * w).sum() / w.sum()))

    args.out.mkdir(parents=True, exist_ok=False)
    config = dict(
        checkpoint=str(args.checkpoint.resolve()),
        center=center.tolist(),
        zero=zero.tolist(),
        side=1000.0,
        start=start_name,
        epochs=int(args.epochs),
        delta=float(args.delta),
        lr=float(args.lr),
        grid=int(args.grid),
        transport_endpoint_rms=end_rms,
        initial=provenance,
        execute=bool(args.execute),
    )
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    np.savez_compressed(
        args.out / "proposals.npz", plane=plane, transport=transport,
        start=start, camera_xy=xy,
    )
    print(json.dumps(config), flush=True)
    if not args.execute:
        return

    from ao_shaping.tools.slm.transfer.hardware import prepare_camera_sdk
    prepare_camera_sdk("daheng")
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms
    from ao_shaping.drivers.slm.santec import Santec

    sy, sx = np.mgrid[:1944, :2592]
    box = (np.abs(sx - center[0]) < 500) & (np.abs(sy - center[1]) < 500)
    core = (np.abs(sx - center[0]) < 460) & (np.abs(sy - center[1]) < 460)
    inner = (np.abs(sx - center[0]) < 390) & (np.abs(sy - center[1]) < 390)
    edge = core & ~inner
    window = (sx >= 1300) & (sx < 2540) & (sy >= 300) & (sy < 1588)
    background_mask = window & ~box
    records = []
    raw_latest = None
    exposure = 35.0

    def measure(phase, label, *, extra_alpha=None, exposure_ms=None):
        nonlocal raw_latest
        if exposure_ms is None:
            exposure_ms = exposure
        gray = slm.create_phase_from_array(to_panel(phase))
        slm.display_data(gray)
        cam.reset_exposure_time(exposure_ms)
        time.sleep(0.7)
        cam.get_numpy_image(n_sample=1, skip_first=False)
        raw = np.stack([
            np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False)).copy()
            for _ in range(5)
        ])
        raw_latest = raw
        frame = raw.mean(0)
        bg = float(np.median(frame[background_mask]))
        signal = np.maximum(frame - bg, 0.0)
        vals = signal[core]
        mean = float(vals.mean())
        cv = float(vals.std() / max(mean, 1e-9))
        ee = float(signal[box].sum() / max(signal.sum(), 1e-9))
        # Use a boundary surrogacy: in-box energy could be biased by zero order,
        # so this is a steering-quality metric, never a terminal efficiency.
        pbr = float(signal[box].max() / max(np.median(signal[background_mask]) + 1e-6, 1e-9))
        row = dict(
            label=label,
            exposure_ms=float(get_camera_exposure_ms(cam)),
            bg=bg,
            mean=mean,
            cv=cv,
            ee=ee,
            pbr=pbr,
            core_peak=float(raw[:, core].max()),
            clipped_core=int((raw[:, core] >= 0.98 * 4095.0).sum()),
            clipped_full=int((raw >= 0.98 * 4095.0).sum()),
            **clock(),
        )
        row["file"] = f"capture_{len(records):04d}.npz"
        np.savez_compressed(args.out / row["file"], raw_frames=raw, model_phase=phase,
                            panel_gray=gray, camera_xy=xy)
        records.append(row)
        (args.out / "manifest.json").write_text(json.dumps(records, indent=2))
        print(json.dumps(row), flush=True)
        if row["clipped_core"]:
            raise RuntimeError("core clipped; exposures must be reduced")
        return row

    with create_camera("daheng", 0, exposure_time_ms=exposure, bit_depth=12) as cam:
        with Santec(slm_number=1, wavelength=1064, video_mode=0) as slm:
            slm.set_phase_overlay(None)
            try:
                current = start.astype(np.float64).copy()
                init_row = measure(current, "initial")
                n = args.grid
                yy_p, xx_p = np.mgrid[:n, :n]
                # Coarse low-frequency local cells; one wavelength of carrier is
                # already present, so leave target frequencies low.
                cells = np.stack([xx_p.ravel(), yy_p.ravel()], axis=-1).astype(float)
                cells = cells - (n - 1) / 2.0
                rng = np.random.default_rng(12345)
                fit_params = np.zeros(n * n, dtype=np.float64)
                # Radians-scale coarse cells; keep smooth by construction (n small).
                def phase_from_params(p):
                    cells_phase = p.reshape(n, n)
                    out = current.copy()
                    block = int(np.ceil(640 / n))
                    up = np.kron(cells_phase, np.ones((block, block), dtype=np.float64))
                    up = up[:640, :640]
                    out += up
                    return out

                best_score = -1.0
                best_params = fit_params.copy()
                obj_history = []
                for epoch in range(args.epochs):
                    disturb = rng.laplace(scale=args.delta, size=n * n)
                    pos_params = fit_params + disturb
                    neg_params = fit_params - disturb
                    pos = phase_from_params(pos_params)
                    neg = phase_from_params(neg_params)
                    rp = measure(pos, f"spgd_pos_{epoch:02d}")
                    rn = measure(neg, f"spgd_neg_{epoch:02d}")
                    # Lower CV + energy + high peak-to-background are desired.
                    def score(row):
                        return -row["cv"] * 1.0 + row["ee"] * 1.0 + row["pbr"] * 0.0
                    sp = score(rp)
                    sn = score(rn)
                    ascent = (sp - sn) * disturb
                    fit_params = np.clip(fit_params + args.lr * ascent, -1.5, 1.5)
                    obj_history.append(float(sp))
                    if float(sp) > best_score:
                        best_score = float(sp)
                        best_params = pos_params.copy()
                    print(
                        f"epoch={epoch:02d} spgd_score={sp:.4f} plus_cv={rp['cv']:.4f} "
                        f"plus_ee={rp['ee']:.4f} minus_cv={rn['cv']:.4f}",
                        flush=True,
                    )
                best_phase = phase_from_params(best_params)
                measure(best_phase, "best_by_spgd")
                np.save(args.out / "best_phase.npy", best_phase)
                # Final ABBA at the reference (start) and candidate.
                measure(current, "abba_A1")
                measure(best_phase, "abba_B1")
                measure(best_phase, "abba_B2")
                measure(current, "abba_A2")
            finally:
                slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
                time.sleep(0.3)
    (args.out / "released.json").write_text(json.dumps(clock()))


if __name__ == "__main__":
    main()
