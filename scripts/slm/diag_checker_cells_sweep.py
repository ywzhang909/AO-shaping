"""棋盘格单元尺寸扫描诊断 (判定空间覆盖 / 棋盘格几何是否为 0 级残留根因)。

问题背景 (#235 LUT 前置修正)
---------------------------
``calib_slm_lut_checkerboard.py`` 在 cells=24 下扫出 0 级残留谷值 ≈0.54 (理想应 ≈0),
反演出的 LUT 不可信。一个候选解释是**空间覆盖**:
棋盘格单元 (cells px) 相对 SLM 上光束口径太小 / 太大, 使 0 级功率未被充分抑制。

本脚本在**单次连续设备会话**内 (open → ``require_drive_board_entry`` → 采集 → close,
绝不背对背 open/close) 对多个单元尺寸跑**粗**灰度扫描, 比较各尺寸的 0 级残留
谷值 ``retained_min``:

    retained(g) = 遮挡棋盘格 0 级窗口功率 / 同曝光平坦 0 级窗口功率

若谷值随尺寸**显著变化** ⇒ 空间覆盖是根因 (据此选最优 ``--cells`` 重跑细扫描);
若谷值**基本不变** (各尺寸都在 ~0.5) ⇒ 几何不是根因, 应改查相位响应不均 / 串扰 /
背景杂散光。

只做诊断, 不反演、不产出 LUT; 每个尺寸只落盘谷值帧 + 平坦帧, 并写对比
``summary.json`` 与 ``release.json``。

判据 / 注意
-----------
* 与标定同一套采集通道 (`capture_mean` / `window_power` / `fwhm_geometric` /
  `checkerboard_gray` / `gray_scan`, 单点真源在 `ao_shaping.tools.slm.calibration.
  checkerboard_lut`, 标定 CLI 也走它)。
* 本脚本的能力已并入 `calib_slm_lut_checkerboard.py --cells-list`; 保留此入口是因为
  `report/slm/lut_checker_20261111/report.md` 的 provenance 记着它, 复现当期诊断用
  这条路径 (它不写 LUT, 只做对比)。
* 标定/诊断必须在 LUT 关闭状态下做 (棋盘格走灰度直写, 绕开驱动 LUT)。
* 每个灰度点前重测一次平坦参考 (局部比值抵消激光漂移)。
* 平坦饱和 / 窗口功率过低 ⇒ 直接报错退出 (0 级比值在饱和下失去意义)。
* 无 ``--execute`` 只打印计划, 不开设备。

用法
----
``python scripts/slm/diag_checker_cells_sweep.py --out data/debug/lut_checker_cells_20261111 --execute``
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

# 采集 / 窗口 / 棋盘格灰度的单点真源在包内 checkerboard_lut (标定 CLI 与本片共用),
# 不再按路径 import 标定脚本去取它的私有函数。
from ao_shaping.tools.slm.calibration.checkerboard_lut import (
    MIN_FLAT_POWER,
    REFERENCE_WINDOW_FACTOR,
    capture_mean,
    checkerboard_gray,
    fwhm_geometric,
    gray_scan,
    window_power,
)
from scripts._common import parse_int_list


def _plan(args: argparse.Namespace) -> dict:
    cells_list = parse_int_list(args.cells_list)
    scan = gray_scan(args.gray_step, args.max_gray_override or 1023)
    return {
        "execute": args.execute,
        "slm_number": args.slm_number,
        "camera": f"daheng:{args.cam_id}",
        "exposure_ms": args.exposure_ms,
        "frames_per_point": args.frames,
        "settle_s": args.settle_s,
        "cells_list": cells_list,
        "gray_scan_points": int(scan.size),
        "gray_first": int(scan[0]),
        "gray_last": int(scan[-1]),
        "max_gray_override": args.max_gray_override,
        "wait_until_idle_s": args.wait_until_idle,
        "writes": [
            "summary.json",
            "release.json",
            "frames/flat.npy",
            "frames/cells<cells>_valley_g<g>.npy",
        ],
    }


def _run(args: argparse.Namespace) -> int:
    from ao_shaping.drivers.ccd.common import create_camera, is_saturated
    from ao_shaping.drivers.slm.factory import require_drive_board_entry
    from ao_shaping.drivers.slm.santec import Santec, SlotRotator

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    cells_list = parse_int_list(args.cells_list)
    if not cells_list:
        print("ERROR: --cells-list empty", file=sys.stderr)
        return 2

    dark: np.ndarray | None = None
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
    device_lock = None
    t0 = time.time()
    per_cells: list[dict] = []
    try:
        if args.wait_until_idle > 0:
            from ao_shaping.tools.kernels.device_mutex import wait_until_idle

            verdict = wait_until_idle(
                args.wait_until_idle,
                slm_number=args.slm_number,
                cam_id=args.cam_id,
                cam_type="daheng",
                exposure_ms=args.exposure_ms,
            )
            if not verdict.ok:
                print(
                    "ERROR: 设备未空出 (%s) — 按提示处置后再跑" % verdict.reason,
                    file=sys.stderr,
                )
                return 2
            # 成功时 lock 已握在手里; 本会话期间持有, finally 释放。
            device_lock = verdict.lock

        slm = Santec(slm_number=args.slm_number, video_mode=0)
        slm.open()
        require_drive_board_entry(slm)
        cam = create_camera("daheng", args.cam_id, exposure_time_ms=args.exposure_ms)
        cam.open()

        panel_shape = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        rotator = SlotRotator(slm)
        device_max_gray = int(
            getattr(slm, "_max_gray", None) or slm.MAX_GRAYSCALE_VALUE
        )
        max_gray = int(args.max_gray_override or device_max_gray)

        def show(gray_img: np.ndarray) -> None:
            rotator.display_data(slm, gray_img, float(args.settle_s))

        def work_of(raw: np.ndarray) -> np.ndarray:
            if dark is not None:
                return np.maximum(raw - dark, 0.0)
            return raw - float(np.median(raw))

        # ── 平坦参考: 定 0 级中心 / 窗口 (一次, 全部尺寸共用) ──
        show(np.zeros(panel_shape, dtype=np.uint16))
        flat_raw = capture_mean(cam, args.frames)
        np.save(out / "frames" / "flat.npy", flat_raw)
        flat = work_of(flat_raw)
        cy, cx = np.unravel_index(int(np.argmax(flat)), flat.shape)
        cy, cx = int(cy), int(cx)
        fwhm = fwhm_geometric(flat, cy, cx)
        side = int(round(REFERENCE_WINDOW_FACTOR * max(fwhm, 5.0)))
        p_flat0 = window_power(flat, cy, cx, side)
        print(
            f"flat: 0-order row={cy} col={cx} fwhm={fwhm:.1f}px "
            f"window={side}px power={p_flat0:.0f} max_gray={max_gray}"
        )
        if is_saturated(flat_raw):
            print("ERROR: flat frame saturated — lower --exposure-ms.", file=sys.stderr)
            return 2
        if p_flat0 < MIN_FLAT_POWER:
            print("ERROR: flat window power too low", file=sys.stderr)
            return 2

        g_values = gray_scan(args.gray_step, max_gray)
        for cells in cells_list:
            t_cells = time.time()
            retained = np.zeros(g_values.size, dtype=float)
            valley_frame: np.ndarray | None = None
            valley_g = 0
            for i, g in enumerate(g_values):
                show(np.zeros(panel_shape, dtype=np.uint16))
                f_work = work_of(capture_mean(cam, 1))
                p_flat = window_power(f_work, cy, cx, side)

                show(checkerboard_gray(cells, int(g), panel_shape))
                c_raw = capture_mean(cam, args.frames)
                c_work = work_of(c_raw)
                p_check = window_power(c_work, cy, cx, side)

                if p_flat <= 0:
                    print(f"ERROR: flat power zero at g={g}", file=sys.stderr)
                    return 2
                retained[i] = p_check / p_flat
                if retained[i] == float(np.min(retained[: i + 1])):
                    valley_frame = c_raw
                    valley_g = int(g)

            idx = int(np.argmin(retained))
            retained_min = float(retained[idx])
            valley_g = int(g_values[idx])
            elapsed = time.time() - t_cells
            print(
                f"cells={cells:3d}: retained_min={retained_min:.4f} "
                f"@g={valley_g} ({elapsed:.1f}s)"
            )
            if valley_frame is not None:
                np.save(
                    out / "frames" / f"cells{cells}_valley_g{valley_g:04d}.npy",
                    valley_frame,
                )
            per_cells.append(
                {
                    "cells": cells,
                    "retained_min": retained_min,
                    "valley_gray": valley_g,
                    "retained_curve": [round(float(v), 6) for v in retained],
                    "elapsed_s": round(elapsed, 1),
                }
            )

        slm_serial = str(getattr(slm, "serial", "") or "")
        meta = {
            "slm_serial": slm_serial,
            "slm_number": args.slm_number,
            "camera": f"daheng:{args.cam_id}",
            "exposure_ms": args.exposure_ms,
            "frames_per_point": args.frames,
            "settle_s": args.settle_s,
            "gray_step": args.gray_step,
            "max_gray": max_gray,
            "cells_list": cells_list,
            "window": {"cy": cy, "cx": cx, "side": side},
            "fwhm_px": round(float(fwhm), 2),
            "flat_power0": round(p_flat0, 1),
        }
        summary = {
            "meta": meta,
            "gray": g_values.tolist(),
            "per_cells": per_cells,
            "elapsed_total_s": round(time.time() - t0, 1),
        }
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n{len(cells_list)} cells sizes scanned -> {out}")
        return 0
    finally:
        if slm is not None:
            try:
                slm.set_grayscale(0)
                release["slm_flat_gray0"] = "ok"
            except Exception as exc:  # 复位失败如实记录, 不改退出码
                release["slm_flat_gray0"] = f"failed: {exc}"
            try:
                slm.close()
                release["slm_closed"] = "ok"
            except Exception as exc:
                release["slm_closed"] = f"failed: {exc}"
        if cam is not None:
            try:
                cam.close()
                release["cam_closed"] = "ok"
            except Exception as exc:
                release["cam_closed"] = f"failed: {exc}"
        (out / "release.json").write_text(
            json.dumps(release, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if device_lock is not None:
            try:
                device_lock.release()
                release["device_lock_released"] = "ok"
            except Exception as exc:
                release["device_lock_released"] = f"failed: {exc}"
            # 重新写 release.json 以包含锁释放状态。
            (out / "release.json").write_text(
                json.dumps(release, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        print("released: " + json.dumps(release, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="棋盘格单元尺寸扫描诊断 (见模块 docstring)",
    )
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--exposure-ms", type=float, default=0.02)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--settle-s", type=float, default=0.3)
    parser.add_argument("--gray-step", type=int, default=32)
    parser.add_argument(
        "--cells-list", type=str, default="8 12 16 24 32", help="空格/逗号分隔"
    )
    parser.add_argument(
        "--max-gray-override",
        type=int,
        default=None,
        help="强制 2π 灰度 (默认取设备 _max_gray, 1064nm->993; 扫 0~1023 传 1023)",
    )
    parser.add_argument(
        "--dark-frame", type=Path, default=None, help="暗场 .npy (可选, 否则减中位)"
    )
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=0,
        help=">0 则先等设备空出 (秒), 不等则 0",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正开设备采集; 省略则只打印计划",
    )
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(_plan(args), indent=2, ensure_ascii=False))
        print("(plan only; 加 --execute 开设备)")
        return 0
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
