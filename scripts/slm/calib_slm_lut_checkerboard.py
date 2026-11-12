"""棋盘格 (checkerboard) 灰度→相位 LUT 标定 (#235 刀口遮挡的前置修正)。

为什么不用权威 `slm-lut` (半屏 blaze) 而另起一条采集路径
-------------------------------------------------------
`tools/slm/calibration/lut_runner.py` 的 `depth` 法用半屏**闪耀光栅** +1 级效率
按 `η=sinc²((D-2π)/2π)` 反演灰度→相位。2026-11-11 在本台实测**该法失效**: blaze
32px 周期做到 2π 深度时 0 级残留仍有 0.51 (理想应 ≈0), 条纹场被 LCOS 低通复位
平滑, 效率曲线不再跟随 sinc²。棋盘格对照 (2π→1.11, π→~0.49) 说明灰度→相位是
**可校准的亚线性**, 不是硬物理上限, 只是 blaze 这条测量通道本身被污染了。

本脚本改用**棋盘格 (交替 0 / Δ 相位)** 直接测 0 级残留比:

    retained = 遮挡棋盘格 0 级功率 / 同曝光平坦 0 级功率 = cos²(Δ/2)

`Δ` 由主动格子的**实际灰度** `g` 决定 (`show` 走 `phase_to_gray(phase, max_gray)`,
即 `g = mod(phase,2π)/(2π)·max_gray`, 灰度直写、绕开驱动 LUT)。扫描灰度 `g`,
测 `retained(g)`, 经 `invert_checkerboard_scan` 反演 `φ(g)` (分支消歧在 `retained`
的**内部最小值** = π 交叉), 再喂给**权威** `build_inverse_lut`/`save_lut` 产出
`lut.npz`/`lut_inverse.csv`, 由 `slm.load_lut(dir)` 消费 —— 与 `slm-lut` 同一份
数学与产物契约, 只换测量通道。

图案生成与灰度扫描循环不在本文件: 它们是
:mod:`ao_shaping.tools.slm.calibration.checkerboard_lut` 的原语 (与单元尺寸扫描
同一份实现, 单点真源)。本文件只做设备会话、产物落盘与 LUT 契约。

判据 / 注意
-----------
* **标定必须在 LUT 关闭状态下做** (`show` 走原始灰度直写, 若先 `load_lut` 会二次
  校正, 测到的曲线是校正后残差, 标定作废)。脚本只写 LUT, 绝不 load。
* 每个灰度点前重测一次平坦参考 (同曝光、同窗口), 用**局部**比值抵消激光漂移。
* 主动格子 2π 时 `retained≈1`、π 时 `retained≈0` → 内部最小值存在, 反演成立;
  若扫不出内部最小 (激光太弱 / 曝光饱和) 直接报错退出, 不产出伪 LUT。
* `--cells-list` 在**同一次设备会话**内扫多个单元尺寸 (原
  `diag_checker_cells_sweep.py` 的能力): 判定空间覆盖是否为 0 级残留根因。LUT
  恒由 `--cells` 那一档产出, 其余档只进 `summary.json` 的 `cells_sweep`。
* 无 `--execute` 只打印计划, **不开设备** (实验前先跑一遍确认参数)。

用法
----
`python scripts/slm/calib_slm_lut_checkerboard.py --out data/debug/lut_checker_20261111 --execute`
`python scripts/slm/calib_slm_lut_checkerboard.py --out data/debug/lut_checker_cells_20261111 --cells-list "8 12 16 24 32" --cells 24 --execute`
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

from ao_shaping.tools.slm.calibration.checkerboard_lut import (
    DEFAULT_CELLS,
    DEFAULT_FRAMES,
    MIN_FLAT_POWER,
    REFERENCE_WINDOW_FACTOR,
    CheckerScan,
    capture_mean,
    fwhm_geometric,
    gray_scan,
    scan_checker_retained,
    window_power,
)
from scripts._common import parse_int_list


def _plan(args: argparse.Namespace) -> dict:
    scan = gray_scan(args.gray_step, args.max_gray_override or 993)
    return {
        "execute": args.execute,
        "slm_number": args.slm_number,
        "camera": f"daheng:{args.cam_id}",
        "exposure_ms": args.exposure_ms,
        "frames_per_point": args.frames,
        "settle_s": args.settle_s,
        "checker_cells": args.cells,
        "cells_list": parse_int_list(args.cells_list) or [args.cells],
        "gray_scan_points": int(scan.size),
        "gray_first": int(scan[0]),
        "gray_last": int(scan[-1]),
        "max_gray_override": args.max_gray_override,
        "wait_until_idle_s": args.wait_until_idle,
        "writes": [
            "summary.json",
            "release.json",
            "frames/flat.npy",
            "frames/checker_g<gray>.npy",
            "lut/ (lut.npz, lut_forward.csv, lut_inverse.csv)",
        ],
    }


def _run(args: argparse.Namespace) -> int:
    from ao_shaping.drivers.ccd.common import create_camera, is_saturated
    from ao_shaping.drivers.slm.factory import require_drive_board_entry
    from ao_shaping.drivers.slm.santec import Santec, SlotRotator
    from ao_shaping.utils.slm.slm_lut import (
        build_inverse_lut,
        invert_checkerboard_scan,
        save_lut,
    )

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()

    cells_list = parse_int_list(args.cells_list) or [args.cells]
    if args.cells not in cells_list:
        cells_list = [args.cells, *cells_list]

    dark: np.ndarray | None = None
    if args.dark_frame is not None:
        if not args.dark_frame.is_file():
            print(f"ERROR: dark frame not found: {args.dark_frame}", file=sys.stderr)
            return 2
        dark = np.asarray(np.load(args.dark_frame), dtype=np.float64)

    def work_of(raw: np.ndarray) -> np.ndarray:
        if dark is not None:
            return np.maximum(raw - dark, 0.0)
        return raw - float(np.median(raw))

    release = {
        "slm_flat_gray0": "not-attempted",
        "slm_closed": "not-attempted",
        "cam_closed": "not-attempted",
    }
    slm = None
    cam = None
    device_lock = None
    t0 = time.time()
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
                    f"ERROR: 设备未空出 ({verdict.reason}) — 按提示处置后再跑",
                    file=sys.stderr,
                )
                return 2
            # 成功时 lock 已握在手里; 本会话期间持有, finally 释放。
            device_lock = verdict.lock

        slm = Santec(slm_number=args.slm_number, video_mode=0)
        slm.open()
        # 采集入口准入: 驱动板不应答则整段测量无效 (与探针 / #235 同一道闸门)。
        require_drive_board_entry(slm)
        cam = create_camera("daheng", args.cam_id, exposure_time_ms=args.exposure_ms)
        cam.open()

        panel_shape = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        rotator = SlotRotator(slm)

        #: 设备 2π 灰度 (1064nm -> 993); --max-gray-override 可强制覆盖。
        device_max_gray = int(
            getattr(slm, "_max_gray", None) or slm.MAX_GRAYSCALE_VALUE
        )
        max_gray = int(args.max_gray_override or device_max_gray)

        def show(gray_img: np.ndarray) -> None:
            rotator.display_data(slm, gray_img, float(args.settle_s))

        # ── 平坦参考: 定 0 级中心 / 窗口 / 曝光线性 (一次, 全部 cells 档共用) ──
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
            print(
                "ERROR: flat frame saturated — lower --exposure-ms; "
                "cos²(Δ/2) 比值在饱和下失去意义。",
                file=sys.stderr,
            )
            return 2
        if p_flat0 < MIN_FLAT_POWER:
            print("ERROR: flat window power too low to invert", file=sys.stderr)
            return 2

        # ── 灰度扫描 (可多档 cells, 同一次设备会话) ──
        g_values = gray_scan(args.gray_step, max_gray)
        scans: dict[int, CheckerScan] = {}
        for cells in cells_list:
            is_primary = cells == args.cells

            def save(
                gray: int, raw: np.ndarray, *, cells: int = cells, primary: bool = is_primary
            ) -> None:
                if primary:
                    np.save(out / "frames" / f"checker_g{gray:04d}.npy", raw)
                else:
                    np.save(
                        out / "frames" / f"cells{cells}_checker_g{gray:04d}.npy", raw
                    )

            try:
                scans[cells] = scan_checker_retained(
                    slm,
                    cam,
                    cells=cells,
                    gray_step=args.gray_step,
                    max_gray=max_gray,
                    frames=args.frames,
                    window_side=side,
                    center=(cy, cx),
                    display=show,
                    background=work_of,
                    save_frames=save if is_primary else None,
                )
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
            if not is_primary:
                print(
                    f"cells={cells:3d}: retained_min={scans[cells].retained_min:.4f} "
                    f"@g={scans[cells].valley_gray} "
                    f"(帧 {out / 'frames' / f'cells{cells}_valley_g{scans[cells].valley_gray:04d}.npy'})"
                )
                if scans[cells].valley_frame is not None:
                    np.save(
                        out / "frames"
                        / f"cells{cells}_valley_g{scans[cells].valley_gray:04d}.npy",
                        scans[cells].valley_frame,
                    )

        primary = scans[args.cells]
        retained, rows = primary.retained, primary.rows

        # ── 反演 + LUT ──
        try:
            phi = invert_checkerboard_scan(retained)
        except ValueError as exc:
            print(f"ERROR: 反演失败 (内部最小值不成立?): {exc}", file=sys.stderr)
            return 3
        t_grid, inverse_gray = build_inverse_lut(phi, g_values.astype(float), n=1024)

        meta = {
            "method": "checkerboard",
            "wavelength_nm": int(getattr(slm, "wavelength", 1064) or 1064),
            "max_gray_2pi_device": device_max_gray,
            "max_gray_used": max_gray,
            "checker_cells": args.cells,
            "gray_step": args.gray_step,
            "exposure_ms": args.exposure_ms,
            "frames_per_point": args.frames,
            "settle_s": args.settle_s,
            "slm_number": args.slm_number,
            "slm_serial": str(getattr(slm, "serial", "") or ""),
            "beam_center_0order": [cy, cx],
            "window_side": side,
            "flat_power_mean": float(np.mean(primary.flat_power)),
            "retained_min": float(np.min(retained)),
            "retained_at_first": float(retained[0]),
            "retained_at_last": float(retained[-1]),
            "branch_gray": float(g_values[int(np.argmin(retained))]),
            "n_points": int(g_values.size),
            "elapsed_s": round(time.time() - t0, 1),
        }
        lut_dir = out / "lut"
        save_lut(lut_dir, g_values.astype(float), phi, inverse_gray, meta=meta)

        summary = {
            "meta": meta,
            "gray": g_values.tolist(),
            "retained": retained.tolist(),
            "phi_rad": phi.tolist(),
            "rows": rows,
            "lut_dir": str(
                lut_dir.resolve().relative_to(ROOT.resolve())
                if lut_dir.resolve().is_relative_to(ROOT.resolve())
                else lut_dir.resolve()
            ),
            "inverse_lut_first16": inverse_gray[:16].tolist(),
            "inverse_lut_last16": inverse_gray[-16:].tolist(),
        }
        if len(cells_list) > 1:
            # 单元尺寸扫描 (原 diag_checker_cells_sweep 的结论字段): 谷值随尺寸显著
            # 变化 => 空间覆盖是根因; 各档都在 ~0.5 => 几何不是根因, 改查相位响应
            # 不均 / 串扰 / 背景杂散光。
            summary["cells_sweep"] = [
                {
                    "cells": cells,
                    "retained_min": scan.retained_min,
                    "valley_gray": scan.valley_gray,
                    "retained_curve": [round(float(v), 6) for v in scan.retained],
                }
                for cells, scan in scans.items()
            ]
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"\nphi spans {phi[0]:.3f}..{phi[-1]:.3f} rad; "
            f"retained min={np.min(retained):.3f} @g={meta['branch_gray']:.0f}; "
            f"LUT -> {lut_dir}"
        )
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
        description="棋盘格灰度→相位 LUT 标定 (见模块 docstring)",
    )
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--exposure-ms", type=float, default=0.05)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--settle-s", type=float, default=0.3)
    parser.add_argument("--gray-step", type=int, default=16)
    parser.add_argument("--cells", type=int, default=DEFAULT_CELLS)
    parser.add_argument(
        "--cells-list",
        type=str,
        default="",
        help="空格/逗号分隔的单元尺寸档 (同一次设备会话内逐档扫描); 默认只扫 --cells",
    )
    parser.add_argument(
        "--max-gray-override",
        type=int,
        default=None,
        help="强制 2π 灰度 (默认取设备 _max_gray, 1064nm->993)",
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
