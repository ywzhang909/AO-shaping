"""大目标框的多曝光 HDR 采集与指标（实验用；``--dry-run`` 可离线复算）。

为什么需要它：单个固定曝光下，大框（≥200 px）的均值落在读出噪声里，CV/EE 都是
噪声统计；短曝光保峰值、长曝光救暗区，必须融合（见 ``utils/image/hdr.py``）。

用法（设备）::

    python scripts/slm/capture_hdr_metrics.py --execute \
        --phase data/debug/defocus_differentiable/<stamp>/stage_e/phases/seed_panel.npy \
        --exposures-ms 1 4 16 --boxes 60 200 1000 --out data/debug/<...>/hdr_seed

用法（离线复算，读已有帧）::

    python scripts/slm/capture_hdr_metrics.py \
        --frames-dir <dir with frame_<exposure>ms.npy> --boxes 60 200 1000 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))

from ao_shaping.utils.image.hdr import box_metrics, hdr_fuse


def compute(frames: list[np.ndarray], exposures: list[float], boxes: list[int],
            centre_ij: tuple[int, int], *, full_scale: float, signal_floor: float | None,
            reference_ms: float) -> dict:
    fused = hdr_fuse(frames, exposures, full_scale=full_scale, reference_exposure_ms=reference_ms)
    metrics = {str(size): box_metrics(fused.radiance, centre_ij, size, signal_floor=signal_floor)
               for size in boxes}
    return dict(fused=fused, metrics=metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, help="面板分辨率相位 .npy（--execute 必需）")
    parser.add_argument("--exposures-ms", type=float, nargs="+", default=[1.0, 4.0, 16.0])
    parser.add_argument("--boxes", type=float, nargs="+", default=[60.0, 200.0, 1000.0])
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--discard", type=int, default=2)
    parser.add_argument("--settle-s", type=float, default=0.4)
    parser.add_argument("--full-scale", type=float, default=255.0, help="相机数字满量程（8bit=255, 12bit=4095）")
    parser.add_argument("--signal-floor", type=float, default=None,
                        help="目标区均值门槛（与融合图同量纲）；缺省不判 valid")
    parser.add_argument("--reference-ms", type=float, default=None, help="融合图参考曝光，缺省取最短")
    parser.add_argument("--centre", type=float, nargs=2, default=None, help="方框中心 (x, y)；缺省取首帧 argmax")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="真的开设备（默认只读/离线）")
    parser.add_argument("--frames-dir", type=Path, default=None, help="离线模式：读 frame_<ms>ms.npy")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reference = args.reference_ms if args.reference_ms is not None else min(args.exposures_ms)

    if not args.execute:
        if args.frames_dir is None:
            raise SystemExit("离线模式需要 --frames-dir（或加 --execute）")
        frames = [np.load(args.frames_dir / f"frame_{exp:g}ms.npy").astype(np.float64) for exp in args.exposures_ms]
        first = frames[0]
        centre = (int(args.centre[0]), int(args.centre[1])) if args.centre else (
            int(np.unravel_index(int(np.argmax(first)), first.shape)[1]),
            int(np.unravel_index(int(np.argmax(first)), first.shape)[0]))
        result = compute(frames, list(args.exposures_ms), [int(b) for b in args.boxes], centre,
                         full_scale=args.full_scale, signal_floor=args.signal_floor, reference_ms=reference)
        payload = dict(mode="offline", exposures_ms=list(args.exposures_ms), centre=centre,
                       coverage=result["fused"].coverage,
                       metrics=result["metrics"], recorded_utc=datetime.now(timezone.utc).isoformat())
        (args.out / "hdr_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec.driver import Santec
    from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

    phase = np.load(args.phase).astype(np.float64)
    slm = Santec(slm_number=1, video_mode=0)
    slm.open()
    slm.set_phase_overlay(None)
    captured: dict[float, np.ndarray] = {}
    try:
        gray = phase_to_slm_grayscale(phase, slm=slm)
        slm.display_data(gray)
        time.sleep(args.settle_s)
        with create_camera("daheng", 0, exposure_time_ms=float(max(args.exposures_ms))) as cam:
            for exposure in args.exposures_ms:
                cam.exposure_time = float(exposure)
                for _ in range(args.discard):
                    cam.get_numpy_image(n_sample=1)
                stack = []
                for _ in range(args.frames):
                    raw = np.asarray(cam.get_numpy_image(n_sample=1), dtype=float)
                    if raw.ndim == 1:
                        raw = raw.reshape(1944, 2592)
                    stack.append(raw.mean(0) if raw.ndim == 3 else raw)
                frame = np.mean(stack, axis=0)
                captured[float(exposure)] = frame
                np.save(args.out / f"frame_{exposure:g}ms.npy", frame.astype(np.float32))
                print(f"  exposure {exposure:g} ms: peak {frame.max():.0f} mean {frame.mean():.2f}", flush=True)
    finally:
        try:
            slm.set_phase_overlay(None)
        except Exception:
            pass
        slm.close()

    frames = [captured[float(e)] for e in args.exposures_ms]
    if args.centre:
        centre = (int(args.centre[0]), int(args.centre[1]))
    else:
        base = frames[0]
        row, col = np.unravel_index(int(np.argmax(base)), base.shape)
        centre = (int(col), int(row))
    result = compute(frames, list(args.exposures_ms), [int(b) for b in args.boxes], centre,
                     full_scale=args.full_scale, signal_floor=args.signal_floor, reference_ms=reference)
    np.save(args.out / "radiance.npy", result["fused"].radiance.astype(np.float32))
    payload = dict(mode="device", phase=str(args.phase), exposures_ms=list(args.exposures_ms),
                   centre=centre, coverage=result["fused"].coverage,
                   saturated_pixels=result["fused"].saturated_pixels,
                   invalid_pixels=result["fused"].invalid_pixels,
                   reference_ms=reference, metrics=result["metrics"],
                   recorded_utc=datetime.now(timezone.utc).isoformat())
    (args.out / "hdr_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "metrics"}, indent=2, ensure_ascii=False))
    for size, values in payload["metrics"].items():
        print(f"  {size:>5} px: mean {values['mean_core']:.4f}  fill {values['fill']:.4f}  "
              f"CV {values['cv_core']:.3f}  valid {values['valid']}")


if __name__ == "__main__":
    main()
