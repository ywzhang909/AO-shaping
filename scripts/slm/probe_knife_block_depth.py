"""量化"遮挡图案"在 0 级窗口里到底残留多少光 (#235 刀口扫描的前置探针)。

为什么需要这个脚本
------------------
``scripts/slm/measure_slm_beam_center.py`` 的刀口不是真挡光板, 而是把被挡半边
换成 π 相位 checker (把光散到高阶)。2026-11-11 实测该办法失效: ``s=0`` (全画
幅 checker) 的 0 级窗口仍有满开时的 ~53% 能量, 于是刀口曲线不是单调 edge, 拟合
出的 sigma 撞上窗口上限 (960 px), #235 判定 ``reliable=false``。

在这个探针之前, "为什么 checker 不挡光" 只能靠猜 (LCOS 条纹场低通? 相位深度不
足? 图案没上屏?)。本脚本把它变成一条可复现的测量: 固定曝光, 依次显示
"平坦 / 若干 checker 周期 / 若干 blaze 周期", 比较同一 0 级窗口的功率。

判据
----
* ``retained_fraction = 遮挡图案功率 / 平坦功率``: 越小越像挡光板。
  ``>= 0.5`` 的图案**不能**用作刀口, 曲线必然不单调。
* 平坦帧必须先量 (同一曝光、同一窗口), 否则比值没有意义。
* 只报测量, 不自动挑图案 —— 选哪个由人按 ``retained_fraction`` 决定。

用法
----
``python scripts/slm/probe_knife_block_depth.py --out data/debug/knife_block_20261111``
(无 ``--execute`` 只打印计划, 不开设备)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 窗口倍数 / FWHM / 窗口功率 / 相机取帧平均 / 棋盘格弧度图案与 #235、棋盘格 LUT
# 标定同源, 一律从 checkerboard_lut 导入, 不再本地重写。
from ao_shaping.tools.slm.calibration.checkerboard_lut import (
    DEFAULT_FRAMES,
    REFERENCE_WINDOW_FACTOR,
    capture_mean,
    checkerboard_phase,
    fwhm_geometric,
    window_power,
)
from ao_shaping.utils.slm.slm_lut import random_phase_pattern
from scripts._common import parse_int_list


def _parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.replace(",", " ").split()]


def _blaze(period: int, depth_rad: float, shape: tuple[int, int]) -> np.ndarray:
    """blaze 光栅的**弧度**图案 (0..depth_rad 锯齿), 平铺整块面板。

    刻意不并入 ``slm_lut.blaze_ramp_gray`` / ``blaze_pattern``: 那两个输出取整后的
    ``uint16`` **灰度**, 本函数的自变量是相位**深度 (rad)**, 要经 ``phase_to_gray``
    的正常显示链才能上屏, 灰度直写会绕过 2π 标定, 结论不可比。
    """
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    del yy
    return (np.mod(xx, period) / period * depth_rad).astype(np.float32)


def _plan(args: argparse.Namespace) -> dict:
    return {
        "script": "scripts/slm/probe_knife_block_depth.py",
        "issue": 235,
        "plan_only": True,
        "devices": {
            "slm": args.slm_number,
            "slm_video_mode": 0,
            "camera": "daheng",
            "cam_id": args.cam_id,
            "exposure_ms": args.exposure_ms,
        },
        "patterns": {
            "flat": "phase 0 (reference)",
            "checker": parse_int_list(args.checker_cells),
            "blaze": parse_int_list(args.blaze_periods),
            "depths_rad": args.depths_rad if args.depths_rad else [args.depth_rad],
        },
        "frames_per_pattern": args.frames,
        "window_factor": REFERENCE_WINDOW_FACTOR,
        "out": str(args.out),
        "writes": ["summary.json", "frames/<pattern>.npy", "release.json"],
    }


def _run(args: argparse.Namespace) -> int:
    from ao_shaping.drivers.ccd.common import create_camera, is_saturated
    from ao_shaping.drivers.slm.factory import require_drive_board_entry
    from ao_shaping.drivers.slm.santec import Santec, SlotRotator
    from ao_shaping.tools.slm.calibration.diffraction_efficiency.core import (
        phase_to_gray,
    )

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()

    dark = None
    if args.dark_frame is not None:
        if not args.dark_frame.is_file():
            print(f"ERROR: dark frame not found: {args.dark_frame}", file=sys.stderr)
            return 2
        dark = np.asarray(np.load(args.dark_frame), dtype=np.float64)

    release = {
        "slm_flat_gray0": "not-attempted",
        "slm_closed": "not-attempted",
        "cam_closed": "not-attempted",
    }
    slm = None
    cam = None
    rows: list[dict] = []
    t0 = time.time()
    try:
        slm = Santec(slm_number=args.slm_number, video_mode=0)
        slm.open()
        # 采集入口准入: 驱动板不应答则整段测量无效 (与 measure_slm_beam_center 一致)。
        require_drive_board_entry(slm)
        cam = create_camera("daheng", args.cam_id, exposure_time_ms=args.exposure_ms)
        cam.open()

        panel_shape = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        rotator = SlotRotator(slm)

        #: 设备 2π 灰度 (1064nm → 993), 与几何标定脚本同一取法。
        max_gray = int(getattr(slm, "_max_gray", None) or slm.MAX_GRAYSCALE_VALUE)

        def show(phase: np.ndarray) -> None:
            # display_data 收的是灰度, 不是弧度 —— 传弧度会把 0..π 截成 0..3,
            # 图案变成近平坦, 遮挡探针测得 100% 残留 (2026-11-11 实测)。
            gray = phase_to_gray(phase, max_gray)
            rotator.display_data(slm, gray, float(args.settle_s))

        def measure(tag: str) -> dict:
            raw = capture_mean(cam, args.frames)
            np.save(out / "frames" / f"{tag}.npy", raw)
            if dark is not None:
                work = np.maximum(raw - dark, 0.0)
            else:
                work = raw - float(np.median(raw))
            return {"tag": tag, "raw": raw, "work": work}

        print("flat reference ...")
        show(np.zeros(panel_shape, dtype=np.float32))
        flat = measure("flat")
        cy, cx = np.unravel_index(int(np.argmax(flat["work"])), flat["work"].shape)
        cy, cx = int(cy), int(cx)
        fwhm = fwhm_geometric(flat["work"], cy, cx)
        side = int(round(REFERENCE_WINDOW_FACTOR * max(fwhm, 5.0)))
        p_flat = window_power(flat["work"], cy, cx, side)
        print(
            f"0-order at row={cy} col={cx} fwhm={fwhm:.1f}px window={side}px power={p_flat:.0f}"
        )
        if p_flat <= 0:
            print("ERROR: flat window power is zero", file=sys.stderr)
            return 2
        rows.append(
            {
                "tag": "flat",
                "kind": "flat",
                "cells": None,
                "window_power": p_flat,
                "retained_fraction": 1.0,
                "frame_max": float(flat["raw"].max()),
                "saturated": bool(is_saturated(flat["raw"])),
            }
        )

        depths = args.depths_rad if args.depths_rad else [args.depth_rad]
        specs: list[tuple[str, str, int, float]] = []
        for depth in depths:
            specs += [
                ("checker", f"checker{cells}@d{depth:.3f}", cells, depth)
                for cells in parse_int_list(args.checker_cells)
            ]
            specs += [
                ("blaze", f"blaze{period}@d{depth:.3f}", period, depth)
                for period in parse_int_list(args.blaze_periods)
            ]
        # 随机相位散斑遮挡: 0 级对相位增益不敏感 (see random_phase_pattern)。
        specs += [
            ("random", f"random_seed{seed}", seed, 0.0)
            for seed in parse_int_list(args.random_seeds)
        ]
        for kind, tag, cells, depth in specs:
            if kind == "checker":
                phase = checkerboard_phase(cells, depth, panel_shape)
            elif kind == "blaze":
                phase = _blaze(cells, depth, panel_shape)
            else:
                phase = random_phase_pattern(cells, panel_shape)
            show(phase)
            got = measure(tag)
            p = window_power(got["work"], cy, cx, side)
            rows.append(
                {
                    "tag": tag,
                    "kind": kind,
                    "cells": cells,
                    "depth_rad": depth,
                    "window_power": p,
                    "retained_fraction": p / p_flat,
                    "frame_max": float(got["raw"].max()),
                    "saturated": bool(is_saturated(got["raw"])),
                }
            )
            print(f"{tag:12s} retained={p / p_flat:5.3f} power={p:9.0f}")

        usable = [
            r["tag"]
            for r in rows
            if r["kind"] != "flat" and r["retained_fraction"] <= 0.2
        ]
        summary = {
            "script": "scripts/slm/probe_knife_block_depth.py",
            "issue": 235,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.time() - t0, 2),
            "devices": _plan(args)["devices"],
            "depth_rad": args.depth_rad,
            "max_gray_2pi": max_gray,
            "zero_order_cy_cx": [cy, cx],
            "fwhm_geometric_px": fwhm,
            "window_side_px": side,
            "window_factor": REFERENCE_WINDOW_FACTOR,
            "background_source": "dark_frame" if dark is not None else "frame_median",
            "rows": rows,
            # 只有 retained_fraction <= 0.2 的图案才可能给出单调刀口曲线。
            "usable_as_knife_edge": usable,
            "criteria": "retained_fraction <= 0.2 => 该图案可作为刀口遮挡",
        }
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps({k: summary[k] for k in ("usable_as_knife_edge",)}, indent=2))
        return 0
    finally:
        if slm is not None:
            try:
                slm.set_grayscale(0)
                release["slm_flat_gray0"] = "ok"
            except Exception as exc:  # 清理失败不改变退出码
                release["slm_flat_gray0"] = f"failed: {exc}"
            try:
                slm.close()
                release["slm_closed"] = "ok"
            except Exception as exc:  # 清理失败不改变退出码
                release["slm_closed"] = f"failed: {exc}"
        if cam is not None:
            try:
                cam.close()
                release["cam_closed"] = "ok"
            except Exception as exc:  # 清理失败不改变退出码
                release["cam_closed"] = f"failed: {exc}"
        (out / "release.json").write_text(
            json.dumps(release, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("released: " + json.dumps(release, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--exposure-ms", type=float, default=0.05)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--settle-s", type=float, default=0.3)
    parser.add_argument("--depth-rad", type=float, default=float(np.pi))
    parser.add_argument(
        "--depths-rad",
        type=_parse_float_list,
        default=None,
        help="comma list of phase depths to sweep (rad); overrides --depth-rad if given",
    )
    parser.add_argument(
        "--checker-cells",
        default="8,16,24,32,48",
        help="comma list of checker cell sizes (panel px)",
    )
    parser.add_argument(
        "--blaze-periods",
        default="32,64",
        help="comma list of blaze periods (panel px)",
    )
    parser.add_argument(
        "--random-seeds",
        default="",
        help="comma list of seeds for random-phase blocker tests (0..2π uniform)",
    )
    parser.add_argument("--dark-frame", type=Path, default=None)
    parser.add_argument(
        "--execute", action="store_true", help="open the devices and run"
    )
    args = parser.parse_args(argv)

    if not args.execute:
        print(json.dumps(_plan(args), indent=2, ensure_ascii=False))
        return 0
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
