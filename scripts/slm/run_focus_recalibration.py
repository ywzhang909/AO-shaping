"""光路变化后的焦点重标定：跑一次焦点扫描并写回 ``focus.json``。

用途：把 CCD 移到焦平面后，旧的 ``best_focal_mm``（约 1783 mm、增益 31×）描述的是
**旧离焦**的补偿，必须重测。本脚本：

1. 可选 ``--wait-until-idle``：先看有没有别的会话在跑设备，有就等（不抢占）；
2. 调 :func:`ao_shaping.tools.slm.panel.focus_probe.probe_session_focus`；
3. 写 ``<out>/focus.json``：同时保留探针原始证据（``probe``）与
   ``focus_preflight.load_focus_calibration`` 认识的嵌套布局（``focus`` / ``settings``）；
4. 与 ``--compare`` 给出的旧 focus.json 对比，打印变化量（焦点扫描的核心读数）。
"""

from __future__ import annotations

import argparse
from loguru import logger
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))


def wait_for_idle(budget_s: float) -> None:
    """等到没有别的硬件脚本在跑（按进程名过滤，不抢设备）。"""
    import subprocess

    pattern = ("hw_", "differentiable", "defocus", "diffraction_efficiency", "compare_sim_vs_hw",
               "capture_", "panel_locate", "focus_probe")
    deadline = time.time() + float(budget_s)
    while time.time() < deadline:
        try:
            listing = subprocess.run(["powershell", "-NoProfile", "-Command",
                                      "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                                      "Select-Object -ExpandProperty CommandLine"],
                                     capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            listing = ""
        busy = [line for line in listing.splitlines()
                if any(token in line for token in pattern)
                and "run_focus_recalibration" not in line]
        if not busy:
            return
        logger.info(f"  [wait] 设备忙（{busy[0][:80]}…），15 s 后再看")
        time.sleep(15)
    raise SystemExit("wait budget exhausted before the device became idle")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path, default=None, help="旧的 focus.json，用于对比")
    parser.add_argument("--wait-until-idle", type=float, default=0.0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--cam-type", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--exposure-ms", type=float, default=None)
    parser.add_argument("--radius-px", type=float, default=None)
    parser.add_argument("--coarse-points", type=int, default=25)
    parser.add_argument("--diopter-range", type=float, default=6.0)
    parser.add_argument("--refine-points", type=int, default=7)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--wavelength-nm", type=int, default=1064)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.wait_until_idle > 0:
        wait_for_idle(args.wait_until_idle)

    from ao_shaping.tools.slm.panel.focus_probe import probe_session_focus

    logger.info(f"[focus] 开始扫描 {datetime.now().strftime('%H:%M:%S')}")
    probe = probe_session_focus(
        slm_number=args.slm_number, cam_type=args.cam_type, cam_id=args.cam_id,
        exposure_ms=args.exposure_ms, radius_px=args.radius_px,
        coarse_points=args.coarse_points, diopter_range=args.diopter_range,
        refine_points=args.refine_points, frames=args.frames,
        wavelength_nm=args.wavelength_nm, debug_dir=args.out / "frames",
    )
    if "error" in probe:
        raise SystemExit(f"焦点扫描失败: {probe['error']}")

    best_mm = float(probe.get("best_focal_mm", float("nan")))
    best_diopter = float(probe.get("best_diopter", probe.get("best_diopter_m1", float("nan"))))
    gain = float(probe.get("gain_vs_flat", float("nan")))
    exposures = [float(p.get("exposure_ms", float("nan"))) for p in probe.get("points", [])]
    record = dict(
        recorded_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        focus=dict(best_focal_mm=best_mm, best_diopter_m1=best_diopter, gain_vs_flat=gain,
                   range_m1=[-float(args.diopter_range), float(args.diopter_range)],
                   exposure_bounds_ms=[float(np.nanmin(exposures)), float(np.nanmax(exposures))]
                   if exposures else None,
                   backend=probe.get("backend")),
        settings=dict(center_x=960.0, center_y=600.0,
                      radius=args.radius_px, wavelength=args.wavelength_nm,
                      slm_number=args.slm_number, cam_type=args.cam_type, cam_id=args.cam_id),
        probe=probe,
    )
    # settings 的照明中心由探针决定（缺省面板中心），以探针返回为准
    for key in ("centre_px", "center", "pupil_center"):
        if key in probe:
            record["settings"]["center_x"] = float(probe[key][0])
            record["settings"]["center_y"] = float(probe[key][1])
            break

    path = args.out / "focus.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info(f"[focus] best_focal_mm = {best_mm:.2f}, best_diopter = {best_diopter:+.4f} m⁻¹, "
          f"gain = {gain:.2f}×  -> {path}")

    if args.compare and args.compare.exists():
        old = json.loads(args.compare.read_text(encoding="utf-8")).get("focus", {})
        old_mm = float(old.get("best_focal_mm", float("nan")))
        old_diopter = float(old.get("best_diopter_m1", float("nan")))
        logger.info("\n[focus] 与旧标定对比（光路变化）")
        logger.info(f"  旧: best_focal_mm {old_mm:.1f}  diopter {old_diopter:+.4f}  gain {old.get('gain_vs_flat')}")
        logger.info(f"  新: best_focal_mm {best_mm:.1f}  diopter {best_diopter:+.4f}  gain {gain}")
        if np.isfinite(old_diopter) and np.isfinite(best_diopter):
            logger.info(f"  变化: {best_diopter - old_diopter:+.4f} m⁻¹ "
                  f"（绝对值 {abs(best_diopter) / max(abs(old_diopter), 1e-9):.2f}× 旧值）")


if __name__ == "__main__":
    main()



