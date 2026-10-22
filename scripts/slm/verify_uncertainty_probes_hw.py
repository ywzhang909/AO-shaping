"""实验试跑: commit 080e0ac9 (issue #134) 的不确定性采样探针 (R-86)。

#134 的验收写明 **"sim 通过后才讨论硬件试跑"**; sim 侧结论 (含一个 `ratio < 1`
的机制负结果) 见 ``report/shape_trials/uncertainty_probe_hw_20261016.md``。本
脚本是那次 sim 验收之后的**首次实验试跑**, 目标不是宣称整形成功, 而是回答三个
只有硬件能回答的问题:

1. **选择器在真实规模下跑不跑得起来**。它的梯度要穿过 ``far_field_tensor``
   (FFT 全图): region=64 / ``far_field_padding``=8 即 512x512, 每次上升要建
   K+1 张远场图, 每轮 12 步、每轮 3 张探针。sim 里只有 region=32/padding=4,
   且 ``uncertainty_probe_steps`` 默认 30 —— 硬件上这里给出的是**真实的单探针
   上升耗时**, 这是判断它能不能进闭环预算的唯一依据。
2. **选出的探针相位物理可下发**: 幅度带是否仍然满足 ``[0.75*spread, spread]``。
3. **同一后验下它是否真的更"不确定"**, 以及两张探针**打到真实 CCD 上的能量
   有没有可观测差别**。

**不复述 sim 的结论, 也不预设硬件会同意它。** #134 的机制在 sim 上已经测到
``ratio_matched ≈ 0.17`` (选出的探针比盲随机的预测方差**更低**)。硬件上复现
同一个符号是一个有价值的复现结果, 不是失败。

实验系统约定 (AGENTS.md / drivers/slm/AGENTS.md):

* Santec SLM **只用 memory 模式** (``video_mode=0``): DVI 模式曾观测到 ``open()``
  挂死, 且挂死后必须物理断电才恢复。
* 相位下发、槽位轮换、稳定等待**全部走既有内核**
  (:func:`~ao_shaping.tools.kernels.bench_kernels.capture_settled` /
  :func:`~ao_shaping.tools.slm.panel.phase_to_panel.phase_to_panel`)。本脚本
  不自己拼 ``create_phase_from_array``/``display_memory`` —— 那正是
  "重复实现漂移"最容易咬人的地方, 而槽位 no-op 那条红线已经踩过一次。
* 零级由全帧 ``argmax`` 定位, 不按几何中心猜 (本实验系统零级不在画面中心)。
* 固定曝光, 不自动抬到饱和; 曝光上限优先取 ``AO_DAHENG_MAX_EXPOSURE_MS``。

用法::

    python scripts/slm/verify_uncertainty_probes_hw.py
    python scripts/slm/verify_uncertainty_probes_hw.py --exposure-ms 0.05 --probe-count 2
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# 大恒 SDK (gxipy) 只随仓库分发在 libs/ 下, 不在 site-packages 里。仓库 .env
# 写了 PYTHONPATH=src;libs, 但从裸 shell / 任务调度器起进程时**不会**加载 .env,
# 于是 create_camera("daheng") 会在半途抛 "name 'gx' is not defined"。
# 这里显式补上 libs/, 让脚本不依赖调用方是否加载过 .env。
_LIBS = ROOT / "libs"
if str(_LIBS) not in sys.path:
    sys.path.insert(0, str(_LIBS))

import torch

from ao_shaping.algorithm.signal_processing.uncertainty_probe_selector import (
    UncertaintyProbeSelector,
)
from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)
from ao_shaping.drivers.ccd.common import create_camera
from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.tools.kernels.bench_kernels import capture_settled
from ao_shaping.tools.slm.panel.display_state import preserve_slm_display
from ao_shaping.utils.slm.bench_constants import SLM_PANEL_H, SLM_PANEL_W
from ao_shaping.utils.image.beam_metrics import prepare_frame, zero_order_center
from ao_shaping.utils.image.exposure_control import (
    configured_daheng_exposure_limit,
)
from ao_shaping.utils.slm.panel_geometry import phase_to_panel
from ao_shaping.optimizer.wfless.slm_model_in_loop import _probe_phase


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


#: piston/tip/tilt 从远场**强度**不可辨识, 与线上默认一致。
_FROZEN_MODES: tuple[int, ...] = (1, 2, 3)


def _n_fitted_coeffs(n_orders: int) -> int:
    """n_orders 阶 Zernike 的**可拟合**列数 = 三角数 - 冻结模式数。

    用 canonical 的 :func:`calc_n_zernike_terms`, 不自己写公式。
    """
    from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms

    return max(1, int(calc_n_zernike_terms(int(n_orders))) - len(_FROZEN_MODES))


def _u_of(selector: UncertaintyProbeSelector, batch) -> float:
    """用选择器**自己的**公共方法给一批探针打分 (单一真源, 不重写公式)。"""
    vals = []
    for probe in batch:
        leaf = torch.from_numpy(np.ascontiguousarray(probe, dtype=np.float64))
        leaf.requires_grad_(True)
        with torch.no_grad():
            vals.append(float(selector.predictive_variance(leaf)))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exposure-ms", type=float, default=0.05)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--region", type=int, default=64)
    ap.add_argument("--far-field-padding", type=int, default=8)
    ap.add_argument("--n-orders", type=int, default=10)
    ap.add_argument("--probe-spread", type=float, default=4.0)
    ap.add_argument("--probe-count", type=int, default=3)
    ap.add_argument("--uncertainty-steps", type=int, default=12)
    ap.add_argument("--uncertainty-samples", type=int, default=6)
    ap.add_argument(
        "--selector-bandlimit", type=float, default=None,
        help=("钉住选择器 bandlimit_sigma (0 = 关带限)。不指定用线上默认 1.5。"
              "sim 扫描显示该项是机制成败的关键, 实验需要复现。"),
    )
    ap.add_argument("--panel-span-px", type=float, default=900.0)
    ap.add_argument("--panel-center", default="960,600")
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # 毫秒级 + 短随机后缀: 秒级时间戳在两次快速启动下会撞名, 而
    # ``exist_ok=False`` 会让第二次直接 FileExistsError —— 曾实测踩到。
    stamp = (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
             + f"_{int.from_bytes(os.urandom(2), 'big'):04x}")
    out_dir = Path(args.out) if args.out else (
        ROOT / "data" / "debug" / "uncertainty_probes_hw" / stamp
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    px, py = (int(v) for v in args.panel_center.split(","))
    meta: dict = {
        "commit": "080e0ac9de2d7d453676d174ae4a1845acb77ee8",
        "issue": "#134 (R-86)",
        "timestamp_utc": stamp,
        "args": vars(args),
        "panel_center": [px, py],
    }
    (out_dir / "config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    _log(f"输出: {out_dir}")
    _log(f"region={args.region} padding={args.far_field_padding} "
         f"n_orders={args.n_orders} spread={args.probe_spread} "
         f"probes={args.probe_count} steps={args.uncertainty_steps} "
         f"samples={args.uncertainty_samples}")

    ceiling_env = configured_daheng_exposure_limit()
    _log(f"曝光请求 {args.exposure_ms} ms; env 上限 "
         f"{ceiling_env if ceiling_env > 0 else '未设 (将用保守默认值)'}")

    slm = Santec(slm_number=1, wavelength=1064, video_mode=0)
    opened_slm = False
    cam = None
    try:
        slm.open()
        opened_slm = True
        _log("SLM #1 已打开 (memory 模式, video_mode=0)")
        # 面板尺寸走 canonical 常量, 不问驱动: Santec 没有 get_resolution,
        # 而 :func:`phase_to_panel` 本身就按 SLM_PANEL_H/W 造输出数组, 两边
        # 必须同源, 否则 flat_panel 的形状会与后面下发的相位不一致。
        panel_shape = (int(SLM_PANEL_H), int(SLM_PANEL_W))
        _log(f"面板尺寸 {panel_shape[1]}x{panel_shape[0]} (SLM_PANEL_W x H)")

        cam = create_camera("daheng", 0, exposure_time_ms=args.exposure_ms)
        cam.open()
        _log(f"CCD #0 已打开 @ {args.exposure_ms} ms")

        with preserve_slm_display(slm):
            flat_panel = np.zeros(panel_shape, dtype=np.float64)

            # env 没设上限时用 0.6 ms 保守默认 —— 绝不自动往上抬到饱和
            ceiling = ceiling_env if ceiling_env > 0 else 0.6
            use_ms = min(args.exposure_ms, ceiling)
            actual = cam.reset_exposure_time(use_ms)
            _log(f"曝光设定 {use_ms} ms, 回读 {actual} ms, 上限 {ceiling} ms")
            meta["exposure_ms_used"] = float(actual)
            meta["exposure_ceiling_ms"] = float(ceiling)

            def grab(region_phase: np.ndarray | None, name: str) -> dict:
                """下发一张区域相位并采集 (走既有内核, 槽位/settle 由内核负责)。"""
                panel = flat_panel if region_phase is None else phase_to_panel(
                    region_phase,
                    int(round(args.panel_span_px / 2)),
                    (px, py),
                )
                raw = np.asarray(capture_settled(
                    cam, slm, panel, n_frames=args.frames, n_discard=3,
                    wait_time_s=0.5, stable_tol=0.02, max_wait_s=6.0,
                ), dtype=np.float64)
                np.save(out_dir / f"{name}_raw.npy", raw)
                if region_phase is not None:
                    np.save(out_dir / f"{name}_region.npy", region_phase)
                    np.save(out_dir / f"{name}_panel.npy", panel)
                prep = prepare_frame(raw)
                cy, cx = zero_order_center(raw)
                roi = max(8, min(raw.shape) // 8)
                energy = float(prep[max(0, cy - roi):cy + roi,
                                    max(0, cx - roi):cx + roi].sum())
                return {"raw_max": float(np.max(raw)),
                        "raw_mean": float(np.mean(raw)),
                        "zero_order_yx": [int(cy), int(cx)],
                        "roi_energy": energy}

            # ---- 平场基线 ---------------------------------------------
            flat = grab(None, "flat")
            _log(f"平场 raw_max={flat['raw_max']:.1f} "
                 f"raw_mean={flat['raw_mean']:.2f} "
                 f"零级=(y={flat['zero_order_yx'][0]},"
                 f"x={flat['zero_order_yx'][1]})")
            meta["flat"] = flat

            # ---- 正向模型 (系数全零 = 线上循环第 0 轮) ----------------
            source_amp = np.zeros((args.region, args.region), dtype=np.float64)
            yy, xx = np.mgrid[0:args.region, 0:args.region]
            c = (args.region - 1) / 2.0
            source_amp[(yy - c) ** 2 + (xx - c) ** 2 <= c ** 2] = 1.0
            opt = ZernikeCoefficientOptimizer(
                n_orders=args.n_orders,
                region=args.region,
                radius=args.region // 2,
                lr=0.05,
                device="cpu",
                dtype="float64",
                seed=None,               # 系数全零 = 线上循环第 0 轮的起点
                far_field_size=args.region * args.far_field_padding,
                frozen_modes=_FROZEN_MODES,  # piston/tip/tilt 强度不可辨识
            )
            opt.set_source_amplitude(source_amp)
            n_fit = int(np.asarray(opt.coefficients).size)
            _log(f"正向模型: n_orders={args.n_orders} n_coeffs={n_fit} "
                 f"region={args.region} radius={args.region // 2} "
                 f"far_field={args.region * args.far_field_padding} "
                 f"(源振幅=单位圆盘)")

            sel_kwargs: dict = {}
            if args.selector_bandlimit is not None:
                sel_kwargs["bandlimit_sigma"] = float(args.selector_bandlimit)
            selector = UncertaintyProbeSelector(
                opt, seed=args.seed, n_samples=args.uncertainty_samples,
                spread_cap=args.probe_spread,
                spread_floor=0.75 * args.probe_spread,
                **sel_kwargs,
            )
            _log(f"选择器 bandlimit_sigma = "
                 f"{sel_kwargs.get('bandlimit_sigma', '默认 1.5')}")
            spread_mid = 0.5 * (args.probe_spread + 0.75 * args.probe_spread)

            rows: list[dict] = []
            chosen: list[np.ndarray] = []
            for k in range(args.probe_count):
                seed = args.seed + k
                # blind 对照: 同种子探针, 归一到与选择器相同的名义幅度
                blind = _probe_phase(args.region, args.probe_spread, seed,
                                     smooth=True)
                blind_n = blind * (spread_mid / max(float(np.std(blind)), 1e-12))
                if chosen:            # 与线上 prev_probes 同义: 折进后验
                    selector.update_covariance(chosen)

                t0 = datetime.now()
                for _ in range(args.uncertainty_steps):
                    selector.update()
                ascent_s = (datetime.now() - t0).total_seconds()
                probe = selector.current_phase()
                selector.commit()
                chosen.append(probe)

                std_unc = float(np.std(probe))
                std_blind = float(np.std(blind_n))
                u_unc = _u_of(selector, [probe])
                u_blind = _u_of(selector, [blind_n])
                # 幅度归一后再比 —— sim 侧已证 U 随幅度单调, 不归一就是在比幅度
                probe_m = probe * (std_blind / max(std_unc, 1e-12))
                u_unc_m = _u_of(selector, [probe_m])

                unc_meas = grab(probe, f"probe{k}_unc")
                blind_meas = grab(blind_n, f"probe{k}_blind")

                row = {
                    "probe_index": k,
                    "seed": seed,
                    "ascent_seconds": ascent_s,
                    "std_uncertainty": std_unc,
                    "std_blind": std_blind,
                    "spread_band": [0.75 * args.probe_spread, args.probe_spread],
                    "std_in_band": bool(
                        0.75 * args.probe_spread - 1e-6 <= std_unc
                        <= args.probe_spread + 1e-6
                    ),
                    "u_uncertainty": u_unc,
                    "u_blind": u_blind,
                    "u_uncertainty_matched": u_unc_m,
                    "ratio_matched": (u_unc_m / u_blind) if u_blind > 0 else None,
                    "unc": unc_meas,
                    "blind": blind_meas,
                    "roi_energy_ratio_unc_over_blind": (
                        unc_meas["roi_energy"] / blind_meas["roi_energy"]
                        if blind_meas["roi_energy"] > 0 else None
                    ),
                }
                rows.append(row)
                _log(f"probe{k}: ascent={ascent_s:.1f}s "
                     f"std unc={std_unc:.3f} blind={std_blind:.3f} "
                     f"U unc={u_unc:.4e} blind={u_blind:.4e} "
                     f"ratio_matched={row['ratio_matched']:.3f}")
                _log(f"  CCD: unc raw_max={unc_meas['raw_max']:.1f} "
                     f"E={unc_meas['roi_energy']:.1f} | "
                     f"blind raw_max={blind_meas['raw_max']:.1f} "
                     f"E={blind_meas['roi_energy']:.1f}")

            flat_again = grab(None, "flat_after")
            _log(f"回到平场: raw_max={flat_again['raw_max']:.1f} "
                 f"E={flat_again['roi_energy']:.1f} "
                 f"(起始 E={flat['roi_energy']:.1f})")

            stds = [r["std_uncertainty"] for r in rows]
            ratios = [r["ratio_matched"] for r in rows if r["ratio_matched"]]
            ratios = [x for x in ratios if x is not None]
            summary = {
                "status": "completed",
                "n_probes": len(rows),
                "all_std_in_band": all(r["std_in_band"] for r in rows),
                "std_range": [float(min(stds)), float(max(stds))],
                "ascent_seconds_mean": float(
                    np.mean([r["ascent_seconds"] for r in rows])),
                "ratio_matched_all": ratios,
                "ratio_matched_mean": float(np.mean(ratios)) if ratios else None,
                "roi_energy_ratio_mean": float(np.mean(
                    [r["roi_energy_ratio_unc_over_blind"] for r in rows
                     if r["roi_energy_ratio_unc_over_blind"] is not None]
                )) if any(r["roi_energy_ratio_unc_over_blind"] is not None
                          for r in rows) else None,
                "flat_energy_before": flat["roi_energy"],
                "flat_energy_after": flat_again["roi_energy"],
            }
            meta["summary"] = summary
            meta["probes"] = rows
            (out_dir / "summary.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            _log("=" * 70)
            _log(f"幅度带内: {summary['all_std_in_band']} "
                 f"(std {summary['std_range'][0]:.3f}..{summary['std_range'][1]:.3f}, "
                 f"band [{0.75 * args.probe_spread:.2f}, {args.probe_spread:.2f}])")
            _log(f"单探针上升耗时 mean={summary['ascent_seconds_mean']:.1f}s "
                 f"(x{args.uncertainty_steps} steps)")
            _log(f"ratio_matched = {summary['ratio_matched_mean']} "
                 f"(逐探针 {ratios})")
            _log(f"roi 能量比 unc/blind mean="
                 f"{summary['roi_energy_ratio_mean']}")
            _log(f"summary: {out_dir / 'summary.json'}")
            _log("=" * 70)
            return 0
    except Exception as exc:
        _log(f"失败: {type(exc).__name__}: {exc}")
        meta["status"] = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        (out_dir / "summary.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    finally:
        try:
            if cam is not None:
                cam.close()
                _log("CCD 已关闭")
        except Exception as exc:
            _log(f"CCD 关闭异常: {exc}")
        try:
            if opened_slm:
                slm.close()
                _log("SLM 已关闭")
        except Exception as exc:
            _log(f"SLM 关闭异常: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())