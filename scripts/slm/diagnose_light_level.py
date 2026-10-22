"""Measure how much light actually reaches the camera, with no SLM involvement.

When a model-in-loop run aborts early the cause is often the bench rather than the
model: a nearly uniform dim frame and a flat panel scan (all responses ~0) mean no beam
is arriving. This probe opens only the CCD, sweeps the exposure and reports peak, mean
and structure, so "no light" can be told apart from "wrong phase" before any SLM phase
is blamed.

Evidence lands in ``data/debug/light_level_probe/<stamp>/``: one npz per exposure, a
figure, and a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.drivers.ccd.common import (
    create_camera,
    get_camera_exposure_ms,
    get_camera_exposure_range,
    set_camera_exposure_ms,
)
from ao_shaping.tools.kernels.bench_kernels import (
    SLM_PANEL_H,
    SLM_PANEL_W,
    display_and_average,
    measure_spot,
)


def stats(frame: np.ndarray) -> dict:
    """Summarise a frame well enough to separate 'dim' from 'no spot'."""
    data = np.asarray(frame, dtype=np.float64)
    peak_index = np.unravel_index(int(np.argmax(data)), data.shape)
    return {
        "shape": list(data.shape),
        "dtype": str(getattr(frame, "dtype", "")),
        "min": float(data.min()),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p99": float(np.percentile(data, 99)),
        "max": float(data.max()),
        "max_position_xy": [int(peak_index[1]), int(peak_index[0])],
        "range": float(data.max() - data.min()),
        "std": float(data.std()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cam-type", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument(
        "--exposures",
        default="0.02,0.1,0.5,1,2,5,20",
        help="Comma separated exposure times in ms",
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "data" / "debug" / "light_level_probe")
    parser.add_argument(
        "--slm-number",
        type=int,
        default=0,
        help="Display a flat phase first (0 = leave the panel as it is). Without this the "
        "reading reflects whatever phase the panel retained, which cannot separate a dim "
        "source from a scattering phase.",
    )
    parser.add_argument("--slm-wavelength", type=int, default=1064)
    args = parser.parse_args(argv)

    exposures = [float(v) for v in args.exposures.split(",")]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    camera = create_camera(args.cam_type, cam_id=args.cam_id, exposure_time_ms=exposures[0])
    slm = None
    if args.slm_number:
        from ao_shaping.drivers.slm.santec import Santec

        slm = Santec(
            slm_number=args.slm_number, wavelength=args.slm_wavelength, video_mode=0
        ).open()

    records = []
    frames = {}
    try:
        with camera:
            print(
                f"camera: {args.cam_type} #{args.cam_id}  "
                f"exposure range {get_camera_exposure_range(camera)} ms"
            )
            if slm is not None:
                display_and_average(
                    camera, slm, np.zeros((SLM_PANEL_H, SLM_PANEL_W)), n_frames=2
                )
                print(f"flat phase displayed on SLM #{args.slm_number} (panel held flat for every reading)")
            for exposure in exposures:
                set_camera_exposure_ms(camera, exposure)
                effective = get_camera_exposure_ms(camera)
                frame = np.asarray(camera.get_numpy_image(n_sample=1))
                repeat = np.asarray(camera.get_numpy_image(n_sample=1))
                record = {"requested_ms": exposure, "effective_ms": float(effective), **stats(frame)}
                record["repeat_max_delta"] = float(np.abs(repeat.astype(np.float64) - frame).max())
                record["spot"] = measure_spot(frame).as_row()
                records.append(record)
                frames[exposure] = frame
                np.savez_compressed(out_dir / f"frame_{exposure:g}ms.npz", frame=frame)
                print(
                    f"  {exposure:>7.2f} ms (eff {effective:>8.4f})  peak={record['max']:>6.2f}  "
                    f"mean={record['mean']:>7.3f}  min={record['min']:>5.1f}  p99={record['p99']:>6.2f}  "
                    f"std={record['std']:>7.3f}  at {record['max_position_xy']}"
                )
                print(f"           spot: {record['spot']}")
    finally:
        if slm is not None:
            slm.close()

    brightest = max(records, key=lambda r: r["max"])
    verdict = _verdict(records, brightest)
    payload = {
        "cam_type": args.cam_type,
        "cam_id": args.cam_id,
        "readings": records,
        "brightest": {"requested_ms": brightest["requested_ms"], "max": brightest["max"]},
        "verdict": verdict,
    }
    (out_dir / "light_level.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot([r["requested_ms"] for r in records], [r["max"] for r in records], "o-", label="peak")
    axes[0].plot([r["requested_ms"] for r in records], [r["mean"] for r in records], "s--", label="mean")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("exposure (ms)")
    axes[0].set_ylabel("counts (0-255)")
    axes[0].set_title("Light level vs exposure")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].imshow(frames[brightest["requested_ms"]], cmap="inferno")
    axes[1].set_title(f"brightest frame ({brightest['requested_ms']:g} ms, peak {brightest['max']:.0f})")
    fig.tight_layout()
    figure = out_dir / "light_level.png"
    fig.savefig(figure, dpi=140)
    plt.close(fig)

    print(f"\nverdict: {verdict['state']}")
    for note in verdict["reasons"]:
        print(f"  - {note}")
    print(f"evidence: {out_dir}")
    return 0 if verdict["state"] != "no_light" else 2


def _verdict(records: list[dict], brightest: dict) -> dict:
    """Classify the bench from the exposure sweep alone."""
    reasons = []
    peak = brightest["max"]
    mean = brightest["mean"]
    span = brightest["range"]

    if peak <= 3:
        reasons.append(f"峰值在最长曝光下仍只有 {peak:.0f}/255")
        reasons.append("画面无结构: 无 0 级光斑, 说明没有光束到达相机")
        if brightest["std"] < 1.0:
            reasons.append(f"整帧标准差 {brightest['std']:.2f} —— 均匀暗场, 只剩读出噪声量级")
        return {"state": "no_light", "reasons": reasons}

    if peak < 20:
        reasons.append(f"峰值 {peak:.0f}/255 偏低, 但仍可能有结构")
    if span < 20:
        reasons.append(f"动态范围 {span:.0f} counts 很小, 光斑对比度不足")
    if brightest["std"] < 2.0 and mean > 0:
        reasons.append("整体偏平 (std 小) —— 可能全部是杂散光而非 0 级")
    if not reasons:
        reasons.append(f"峰值 {peak:.0f}/255, 动态范围 {span:.0f} counts, 有可用信号")
    state = "no_light" if peak <= 3 else ("weak" if peak < 20 else "ok")
    return {"state": state, "reasons": reasons}


if __name__ == "__main__":
    raise SystemExit(main())