"""Measured low-frequency SPGD from the broadened seed, with candidate ABBA."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective
from ao_shaping.tools.slm.transfer.capture import clock, to_panel

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/debug/slm_transfer/multishape_nogs_20261030"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def dct_basis(n: int, size: int) -> np.ndarray:
    x = np.arange(size) + .5
    modes = []
    for v in range(n):
        for u in range(n):
            if u == v == 0:
                continue
            mode = np.cos(np.pi * u * x[:, None] / size) * np.cos(np.pi * v * x[None, :] / size)
            modes.append(mode / np.sqrt(np.mean(mode ** 2)))
    return np.stack(modes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-until-idle", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--delta", type=float, default=.04)
    parser.add_argument("--step", type=float, default=.04)
    parser.add_argument("--grid", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20 or not 0 < args.delta <= .1 or not 0 < args.step <= .1:
        parser.error("1..20 epochs; delta and step in (0,0.1] rad")
    if not 3 <= args.grid <= 8:
        parser.error("grid must be 3..8")
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("square_protocol", ROOT / "scripts/slm/square_1000_diff_abba_20261030.py")
    protocol = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = protocol
    spec.loader.exec_module(protocol)
    seed_run = DATA / "square1000_bandlimit_live_20261030_v2"
    seed_path = seed_run / "plan.npz"
    with np.load(seed_path) as saved:
        seed = np.asarray(saved["F_blur4_full"], np.float32)
        xy = saved["camera_xy"].copy()
    basis = dct_basis(args.grid, len(seed)).astype(np.float32)
    objective = ShapeObjective(protocol.masks(xy)["box"].astype(np.float32), formation=True)
    rng = np.random.default_rng(20261030)
    # Preview uses only deterministic model-free perturbations, never a model gradient.
    preview_delta = np.tensordot(rng.choice([-1., 1.], len(basis)) * args.delta, basis, axes=1)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "plan.npz", seed=seed, basis=basis, camera_xy=xy,
                        preview_delta=preview_delta)
    write_json(args.out / "config.json", dict(
        execute=args.execute, seed_run=str(seed_run), seed_path=str(seed_path),
        seed_key="F_blur4_full", seed_sha256=hashlib.sha256(seed_path.read_bytes()).hexdigest(),
        basis="unit-RMS 2D DCT excluding piston", grid=args.grid, modes=len(basis),
        delta_rad=args.delta, trial_step_rad=args.step, epochs=args.epochs,
        optimizer="measured two-sided SPGD with Adam normalization; no forward-model Jacobian",
        acceptance="same protocol as square_1000_diff_abba_20261030.py", **clock()))
    if not args.execute:
        print(json.dumps(dict(out=str(args.out), modes=len(basis), preview_rms=float(np.sqrt(np.mean(preview_delta ** 2))))))
        return

    from ao_shaping.tools.kernels.device_mutex import DeviceLock, wait_until_idle
    verdict = wait_until_idle(args.wait_until_idle, lock=DeviceLock.default(), own_paths=[Path(__file__).resolve()])
    if not verdict.ok or verdict.lock is None:
        raise RuntimeError(verdict.headline())
    records, decisions = [], []
    try:
        from ao_shaping.tools.slm.transfer.hardware import prepare_camera_sdk
        prepare_camera_sdk("daheng")
        from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms
        from ao_shaping.drivers.slm.santec import Santec
        yy, xx = np.mgrid[:1944, :2592]
        native_masks = protocol.masks(np.stack([xx, yy], -1))
        with create_camera("daheng", 0, exposure_time_ms=.5, bit_depth=12) as cam:
            with Santec(slm_number=1, wavelength=1064, video_mode=0) as slm:
                drive_board = slm.require_drive_board()
                write_json(args.out / "device.json", dict(
                    drive_board=str(drive_board), slm_number=1, wavelength_nm=1064,
                    opened_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
                slm.set_phase_overlay(None)
                try:
                    def capture(phase, exposure, label):
                        gray = slm.create_phase_from_array(to_panel(phase))
                        slm.display_data(gray)
                        cam.reset_exposure_time(exposure)
                        time.sleep(.8)
                        for _ in range(3):
                            cam.get_numpy_image(n_sample=1, skip_first=False)
                        raw = np.stack([np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False)).copy() for _ in range(5)])
                        row = dict(label=label, exposure_ms=float(get_camera_exposure_ms(cam)),
                                   file=f"capture_{len(records):04d}.npz", **clock())
                        if raw.shape != (5, 1944, 2592) or raw.dtype != np.uint16:
                            raise RuntimeError("expected native Mono12 uint16 frames")
                        np.savez_compressed(args.out / row["file"], raw_frames=raw, model_phase=phase, panel_gray=gray, camera_xy=xy)
                        records.append(row)
                        write_json(args.out / "manifest.json", records)
                        return row, raw

                    row, raw = capture(seed, .5, "seed_safe_meter")
                    peak = float(raw[:, native_masks["box"]].max())
                    if peak < 20 or peak >= .98 * 4095:
                        raise RuntimeError("unsafe or unresolved metering")
                    exposure = min(16.87, row["exposure_ms"] * 1400 / peak)
                    row, raw = capture(seed, exposure, "seed_calibration_0")
                    bg = float(np.median(raw.mean(0)[native_masks["background"]]))
                    noise = float(np.sqrt(raw[:, native_masks["background"]].var(0, ddof=1).mean() / len(raw)))
                    mean = float(np.maximum(raw.mean(0) - bg, 0)[native_masks["core"]].mean())
                    threshold = max(3 * noise, .2 * mean)
                    write_json(args.out / "measurement_contract.json", dict(background_counts=bg,
                        coverage_threshold_counts=threshold, exposure_ms=exposure, noise_counts=noise,
                        full_scale=4095, smoothing_sigma_native_px=8))

                    def evaluate(row, raw):
                        row["metrics"] = protocol.metrics(raw, row["exposure_ms"], bg, threshold,
                                                         native_masks, objective, xy)
                        write_json(args.out / "manifest.json", records)
                        print(json.dumps(dict(label=row["label"], **row["metrics"])), flush=True)
                        return row

                    repeats = [evaluate(row, raw)]
                    for i in (1, 2):
                        row, raw = capture(seed, exposure, f"seed_calibration_{i}")
                        repeats.append(evaluate(row, raw))
                    accepted = seed.copy()
                    np.save(args.out / "accepted_phase.npy", accepted)
                    m = np.zeros(len(basis))
                    v = np.zeros(len(basis))
                    rejected = 0
                    for epoch in range(args.epochs):
                        row, raw = capture(accepted, exposure, f"epoch{epoch}_A")
                        repeats.append(evaluate(row, raw))
                        repeats = repeats[-5:]
                        sign = rng.choice([-1., 1.], len(basis))
                        perturb = np.tensordot(args.delta * sign, basis, axes=1).astype(np.float32)
                        rp, fp = capture(accepted + perturb, exposure, f"epoch{epoch}_plus")
                        evaluate(rp, fp)
                        rm, fm = capture(accepted - perturb, exposure, f"epoch{epoch}_minus")
                        evaluate(rm, fm)
                        # The old scalar loss chiefly rewarded texture smoothing. Optimize
                        # morphology as well: coverage/IoU gradients act in opposite sign.
                        def control_score(row):
                            q = row["metrics"]
                            return (q["loss"] - 20 * q["coverage"] - 10 * q["iou"]
                                    + .2 * q["cv"] + .2 * q["boundary_error"])
                        gradient = (control_score(rp) - control_score(rm)) / (2 * args.delta) * sign
                        m = .9 * m + .1 * gradient
                        v = .999 * v + .001 * gradient ** 2
                        update = (m / (1 - .9 ** (epoch + 1))) / (np.sqrt(v / (1 - .999 ** (epoch + 1))) + 1e-8)
                        step = args.step / (2 ** rejected)
                        delta = np.tensordot(-step * update, basis, axes=1).astype(np.float32)
                        candidate = accepted + delta
                        rows = []
                        for label, phase in (("A1", accepted), ("B1", candidate), ("B2", candidate), ("A2", accepted)):
                            r, f = capture(phase, exposure, f"epoch{epoch}_{label}")
                            rows.append(evaluate(r, f))
                        verdict_abba = protocol.accept_abba(rows, repeats)
                        take = bool(verdict_abba["accepted"])
                        decisions.append(dict(epoch=epoch, delta_rad=args.delta, step_rad=step,
                            spgd_loss_plus=rp["metrics"]["loss"], spgd_loss_minus=rm["metrics"]["loss"],
                            control_score_plus=control_score(rp), control_score_minus=control_score(rm),
                            gradient_rms=float(np.sqrt(np.mean(gradient ** 2))), accepted=take, verdict=verdict_abba))
                        write_json(args.out / "decisions.json", decisions)
                        if take:
                            accepted = candidate
                            rejected = 0
                            np.save(args.out / "accepted_phase.npy", accepted)
                        else:
                            rejected += 1
                        if rejected >= 2:
                            break
                finally:
                    slm.display_data(np.zeros((1200, 1920), dtype=np.uint16))
                    time.sleep(.3)
    finally:
        verdict.lock.release()
    write_json(args.out / "released.json", clock())


if __name__ == "__main__":
    main()
