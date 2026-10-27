"""Interleaved fixed-phase exposure response, with every acquired frame retained.

Requires explicit --execute. Uses a measured shape phase and a flat reference;
an illuminated affine intercept is a diagnostic, not an independently measured dark.
"""

from __future__ import annotations
from loguru import logger

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np

from ao_shaping.tools.slm.transfer.capture import clock, to_panel
from ao_shaping.utils.image.target import POLYGON_SHAPE_CHOICES

FRACTIONS = (1.0, 0.25, 1.0, 0.5, 1.0, 0.125, 1.0, 0.5, 0.25, 0.125, 1.0)


def capture(args):
    from .hardware import prepare_camera_sdk
    prepare_camera_sdk(args.camera)
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms, get_camera_exposure_range
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.slm.panel.focus_probe import adapt_exposure
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    manifest = json.loads((args.source / "manifest.json").read_text())
    record = next(r for r in manifest if r["label"] == f"{args.shape}_final")
    with np.load(args.source / record["file"]) as saved:
        shape_phase, xy, target = (
            saved["model_phase"],
            saved["camera_xy"],
            saved["target"],
        )
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = dict(
        source=str(args.source.resolve()),
        shape=args.shape,
        fractions=FRACTIONS,
        frames=args.frames,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256(
            (args.source / "manifest.json").read_bytes()
        ).hexdigest(),
        **clock(),
    )
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2))
    rows = []
    with create_camera(args.camera, args.cam_id, exposure_time_ms=0.02) as cam:
        with Santec(slm_number=args.slm_number, wavelength=1064, video_mode=0) as slm:
            slm.set_phase_overlay(None)
            try:
                floor, ceiling = get_camera_exposure_range(cam)
                limit = (
                    configured_daheng_exposure_limit()
                    if args.camera == "daheng"
                    else 0.0
                )
                if limit > 0:
                    ceiling = min(ceiling, limit)
                for phase_name, phase in (
                    ("flat", np.zeros_like(shape_phase)),
                    (args.shape, shape_phase),
                ):
                    panel = to_panel(phase)
                    gray = slm.create_phase_from_array(panel)
                    slm.display_data(gray)
                    time.sleep(0.6)
                    reads, stamps = [], []

                    def read_probe(_reads=reads, _stamps=stamps):
                        start = clock()
                        frame = np.asarray(
                            cam.get_numpy_image(n_sample=1, skip_first=False)
                        ).copy()
                        _reads.append(frame)
                        _stamps.append(
                            dict(
                                **start,
                                end_monotonic_ns=time.monotonic_ns(),
                                exposure_ms=float(get_camera_exposure_ms(cam)),
                            )
                        )
                        return frame

                    setting = adapt_exposure(
                        cam.reset_exposure_time,
                        read_probe,
                        start_ms=floor,
                        floor_ms=floor,
                        ceiling_ms=ceiling,
                        full_scale=255,
                        max_steps=12,
                    )
                    np.savez_compressed(
                        args.out / f"{phase_name}_probe.npz",
                        raw_frames=np.stack(reads),
                        panel_gray=gray,
                        panel_phase=panel,
                        model_phase=phase,
                        camera_xy=xy,
                        target=target,
                    )
                    (args.out / f"{phase_name}_probe.json").write_text(
                        json.dumps(dict(frames=stamps, status=setting.status), indent=2)
                    )
                    if setting.status != "ok":
                        raise RuntimeError(
                            "exposure search failed; all frames retained"
                        )
                    reference = float(get_camera_exposure_ms(cam))
                    for index, fraction in enumerate(FRACTIONS):
                        requested = max(floor, reference * fraction)
                        if requested > ceiling:
                            raise ValueError(
                                "exposure outside current device/environment bounds"
                            )
                        cam.reset_exposure_time(requested)
                        time.sleep(0.15)
                        frames, clocks = [], []
                        for _ in range(args.frames):
                            started = clock()
                            frames.append(
                                np.asarray(
                                    cam.get_numpy_image(n_sample=1, skip_first=False)
                                ).copy()
                            )
                            clocks.append(
                                dict(
                                    **started,
                                    end_monotonic_ns=time.monotonic_ns(),
                                    exposure_ms=float(get_camera_exposure_ms(cam)),
                                )
                            )
                        raw = np.stack(frames)
                        file = f"{phase_name}_{index:02d}.npz"
                        np.savez_compressed(args.out / file, raw_frames=raw)
                        row = dict(
                            phase=phase_name,
                            index=index,
                            fraction=fraction,
                            file=file,
                            exposure_ms=float(get_camera_exposure_ms(cam)),
                            reference_ms=reference,
                            peak=int(raw.max()),
                            frames=clocks,
                            **clock(),
                        )
                        rows.append(row)
                        (args.out / "manifest.json").write_text(
                            json.dumps(rows, indent=2)
                        )
                        with (args.out / "capture_records.pkl").open("wb") as stream:
                            pickle.dump(rows, stream, protocol=5)
                        print(json.dumps(row), flush=True)
                        if raw.max() >= 250:
                            raise RuntimeError(
                                "saturated group retained; ending exposure comparison"
                            )
            finally:
                slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
                time.sleep(0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--shape", default="square", choices=POLYGON_SHAPE_CHOICES
    )
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--camera", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.frames < 3:
        parser.error(
            "at least three frames required to inspect exposure transition frames"
        )
    if args.execute:
        capture(args)
    else:
        logger.info(
            json.dumps(
                dict(
                    phases=["flat", args.shape],
                    fractions=FRACTIONS,
                    source=str(args.source),
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
