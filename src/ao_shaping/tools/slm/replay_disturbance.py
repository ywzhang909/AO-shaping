"""Interleaved fixed-exposure CCD replay of the four saved comparison phases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from loguru import logger

from ao_shaping.drivers.ccd.common import create_camera
from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.optimizer.wfless.slm_model_in_loop import phase_to_panel
from ao_shaping.tools.kernels.bench_kernels import capture_settled
from ao_shaping.tools.slm.compare_disturbance import _common_quality
from ao_shaping.tools.slm.panel.display_state import preserve_slm_display
from ao_shaping.utils.image.beam_metrics import prepare_frame, zero_order_center
from ao_shaping.utils.image.exposure_control import (
    DAHENG_EXPOSURE_LIMIT_ENV,
    configured_daheng_exposure_limit,
    probe_exposure_ceiling_ms,
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _phase(
    root: Path, item: dict, bias: np.ndarray, center: tuple[int, int], radius: int,
) -> np.ndarray:
    run = root / item["directory"]
    if item["algorithm"] == "gs_refine":
        panel = np.load(run / "best_phase_rad.npy", allow_pickle=False)
    else:
        grid = np.load(run / "best_phase.npy", allow_pickle=False)
        panel = phase_to_panel(grid, radius, center)
    if panel.shape != bias.shape:
        raise ValueError(f"saved phase panel shape mismatch: {run}")
    return panel if item["condition"] == "clean" else panel + bias


def replay(root: Path, exposure_ms: float, repeats: int) -> Path:
    if exposure_ms <= 0 or repeats < 1:
        raise ValueError("require positive exposure and repeats >= 1")
    env_ceiling = configured_daheng_exposure_limit()
    if env_ceiling > 0 and exposure_ms > env_ceiling:
        raise ValueError(
            f"Daheng exposure {exposure_ms} ms exceeds the {env_ceiling} ms "
            f"ceiling ({DAHENG_EXPOSURE_LIMIT_ENV})"
        )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if any(item.get("result") != "completed" for item in manifest["runs"]):
        raise ValueError("all four optimization runs must be completed")
    disturbance = json.loads((root / "disturbance.json").read_text(encoding="utf-8"))
    bias = np.load(root / disturbance["panel_file"], allow_pickle=False)
    center = tuple(int(value) for value in disturbance["center_xy"])
    radius = int(disturbance["radius_px"])
    phases = [_phase(root, item, bias, center, radius) for item in manifest["runs"]]
    output = root / "fixed_exposure_replay"
    output.mkdir(exist_ok=False)
    rows = []
    with Santec(slm_number=1, wavelength=1064) as slm, create_camera(
        "daheng", 0, exposure_time_ms=0.02
    ) as cam, preserve_slm_display(slm):
        ceiling = configured_daheng_exposure_limit()
        if not ceiling:
            estimate = probe_exposure_ceiling_ms(
                cam.reset_exposure_time,
                lambda: np.asarray(cam.get_numpy_image(n_sample=2)),
            )
            ceiling = estimate.ceiling_ms
            logger.info(
                "Daheng exposure ceiling probed from the image: {:.4f} ms", ceiling
            )
        actual = float(cam.reset_exposure_time(0.02))
        if not 0 < actual <= ceiling + 1e-9:
            raise RuntimeError(
                f"unsafe flat exposure readback {actual} ms (ceiling {ceiling} ms)"
            )
        full = capture_settled(cam, slm, np.zeros_like(bias), n_frames=4)
        full_center = zero_order_center(prepare_frame(full), refine=False)
        cam.reset_window(center=full_center, size=(256, 256))
        flat = capture_settled(cam, slm, np.zeros_like(bias), n_frames=4)
        center = zero_order_center(prepare_frame(flat), refine=False)

        def acquire(label: str, panel: np.ndarray, requested_ms: float) -> None:
            actual_ms = float(cam.reset_exposure_time(requested_ms))
            if not 0 < actual_ms <= ceiling + 1e-9:
                raise RuntimeError(
                    f"unsafe Daheng exposure readback {actual_ms} ms "
                    f"(ceiling {ceiling} ms)"
                )
            frame = capture_settled(cam, slm, panel, n_frames=4)
            raw = np.asarray(cam.get_numpy_image(n_sample=1))
            peak = max(float(np.max(frame)), float(np.max(raw)))
            if peak >= 0.9 * np.iinfo(raw.dtype).max:
                raise RuntimeError(f"saturated replay frame {label}: peak={peak}")
            clean = prepare_frame(frame)
            index = len(rows)
            filename = f"{index:03d}_{label}.npz"
            np.savez_compressed(output / filename, averaged=frame, raw=raw,
                                preprocessed=clean)
            rows.append({"label": label, "file": filename, "timestamp_utc": _utc(),
                         "exposure_ms": actual_ms, "peak": peak,
                         "quality": _common_quality(clean, 100, center),
                         "total_signal": float(clean.sum())})
            (output / "manifest.json").write_text(json.dumps({
                "source": str(root), "center_xy": list(center),
                "full_center_xy": list(full_center), "rows": rows,
            }, indent=2), encoding="utf-8")

        acquire("flat_start", np.zeros_like(bias), 0.02)
        for repeat in range(repeats):
            order = range(4) if repeat % 2 == 0 else range(3, -1, -1)
            for index in order:
                item = manifest["runs"][index]
                acquire(f"r{repeat}_{item['algorithm']}_{item['condition']}",
                        phases[index], exposure_ms)
            acquire(f"flat_{repeat}", np.zeros_like(bias), 0.02)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--exposure-ms", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(replay(args.run_dir, args.exposure_ms, args.repeats))


if __name__ == "__main__":
    main()
