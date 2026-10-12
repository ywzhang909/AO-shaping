"""Replay saved GS-refine phases on the bench with interleaved references."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from ao_shaping.drivers.ccd.common import create_camera
from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.optimizer.wfless.slm_gs_refine import _metrics
from ao_shaping.tools.slm.slm_panel_locate import _restore_display
from ao_shaping.utils.image.targets import square_mask
from ao_shaping.utils.image.beam_metrics import prepare_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("debug_pkl", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--exposure-ms", type=float, required=True)
    parser.add_argument("--slm-serial", required=True)
    parser.add_argument("--cam-center", default="886,706")
    parser.add_argument("--cam-size", type=int, default=250)
    parser.add_argument("--target-center", default="127,126")
    parser.add_argument("--target-side", type=int, default=100)
    args = parser.parse_args()
    if not 0 < args.exposure_ms <= 0.4:
        parser.error("Daheng exposure must be in (0, 0.4] ms")

    with args.debug_pkl.open("rb") as handle:
        records = pickle.load(handle)
    best_key = max(records, key=lambda key: records[key]["score"])
    phases = {
        "flat": np.asarray(records[0]["_phase"]),
        "gs": np.asarray(records[1]["_phase"]),
        "best": np.asarray(records[best_key]["_phase"]),
    }
    cam_xy = tuple(map(int, args.cam_center.split(",")))
    target_xy = tuple(map(int, args.target_center.split(",")))
    sequence = ("flat", "gs", "best", "gs", "flat", "best", "flat")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    with Santec(slm_number=1, wavelength=1064, video_mode=0) as slm, \
            create_camera("daheng", 0, exposure_time_ms=args.exposure_ms) as cam, \
            _restore_display(slm):
        if slm.get_serial_number() != args.slm_serial:
            raise ValueError("SLM serial changed; refusing phase replay")
        cam.reset_exposure_time(args.exposure_ms)
        if cam.exposure_time > 0.4:
            raise ValueError("camera exposure readback exceeds 0.4 ms")
        cam.reset_window(center=cam_xy, size=(args.cam_size, args.cam_size))
        protected = slm.get_displayed_memory_number()
        work_slots = [slot for slot in (126, 127, 124) if slot != protected][:2]
        for index, name in enumerate(sequence):
            slm.display_data(phases[name], memory_number=work_slots[index % 2])
            time.sleep(0.8)
            for _ in range(3):
                cam.get_numpy_image(n_sample=1)
            frame = prepare_frame(np.asarray(cam.get_numpy_image(n_sample=2), dtype=float))
            if float(np.max(frame)) >= 250:
                raise ValueError(f"CCD saturated during replay at {name}")
            mask = square_mask(frame.shape, target_xy[1], target_xy[0], args.target_side)
            score, pib, cv = _metrics(frame, mask, 0.5, 0.5)
            np.save(args.out_dir / f"{index:02d}_{name}.npy", frame)
            rows.append(dict(index=index, phase=name, score=score, pib=pib, cv=cv,
                             peak=float(np.max(frame)), total=float(np.sum(frame))))
            (args.out_dir / "measurements.json").write_text(
                json.dumps(rows, indent=2), encoding="utf-8"
            )
            print(rows[-1], flush=True)


if __name__ == "__main__":
    main()
