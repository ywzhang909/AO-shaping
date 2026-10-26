"""实验配对 A/B (GitHub #139): 独立验证集相关性上, 不确定性探针是否真优于固定随机探针。

本脚本只补 #139 验收口径里**唯一没测过的那一项**: *独立验证集相关性* (#117/#119
确立的口径), 不跑整形闭环, 因此不涉及形状验收。

协议 (#139 步骤 1 的"同连接内交错"):

* 一次几何标定, 全程共用 (同连接、同曝光、同 ROI 中心);
* 每个变体开始前**重新测一遍共享验证集探针集**, 于是评分对照明漂移免疫, 且
  各变体在时间上交错, 任何漂移都落在差值里;
* 训练探针按变体的策略生成 (固定随机 / 不确定性采样), 各自独立拟合系数;
* 评分: 用拟合出的系数预测验证集探针的远场, 与实测做 Pearson 相关 —— 与循环
  里 ``holdout_correlation`` 同一个口径与同一条代码路径
  (``forward_intensity`` + ``_prediction_correlation``)。

变体:

===========  ==========================  ==========================================
变体           训练探针                     意图
===========  ==========================  ==========================================
A8           8 张固定随机 (spread 4.0)     基线 (线上默认探针预算)
A3           3 张固定随机                 基线在**同预算**下的参照
B6_s1.5      6 张不确定性, sigma=1.5       默认配置 —— 预期机制不成立
B6_s0        6 张不确定性, sigma=0         关带限 —— 本轮 sim 唯一全胜的配置
B3_s0        3 张不确定性, sigma=0         同探针预算对比 (#139 步骤 2)
B2_s0        2 张不确定性, sigma=0         同探针预算对比 (#139 步骤 2)
===========  ==========================  ==========================================

同时记录每张探针的相位标准差 (#139 首要风险: 样本塌缩 / 被推到极端)。

实验系统约定与前置: Santec #1 memory 模式; 零级按全帧 argmax 冻结; 曝光固定;
进出现场恢复显示。**几何在本次会话内现标**, 不沿用任何历史文件 —— 2026-10-15
报告已作废当时的面板入射中心, 而 #139 要求"同一几何"。

用法::

    PYTHONPATH=src;libs python scripts/slm/verify_uncertainty_probes_holdout_hw.py
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
_LIBS = ROOT / "libs"
if str(_LIBS) not in sys.path:
    sys.path.insert(0, str(_LIBS))

from ao_shaping.algorithm.signal_processing.uncertainty_probe_selector import (
    UncertaintyProbeSelector,
)
from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)
from ao_shaping.optimizer.wfless import slm_model_in_loop as mil
from ao_shaping.utils.image.beam_metrics import (
    prepare_frame,
    zero_order_center,
)

#: 共享验证集探针张数。验证集**各变体完全相同**, 所以相关性的差值就是探针策略
#: 的差值, 不掺探针实现的随机性。
HOLDOUT_COUNT = 4

#: 训练探针上的拟合预算。刻意压小: #139 问的是"同样的拟合努力下, 哪套探针
#: 换来的验证集相关性更高", 而不是各自跑到收敛。
FIT_ITERATIONS = 60


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _bandlimit_patch(sigma: float):
    """临时把选择器的 bandlimit_sigma 钉住 (0 = 关带限)。"""
    from ao_shaping.algorithm.signal_processing import (
        uncertainty_probe_selector as ups,
    )

    orig = ups.UncertaintyProbeSelector

    class _Pinned(orig):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **kw):
            kw["bandlimit_sigma"] = float(sigma)
            super().__init__(*a, **kw)

    ups.UncertaintyProbeSelector = _Pinned
    return orig


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exposure-ms", type=float, default=0.05)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--region", type=int, default=64)
    ap.add_argument("--n-orders", type=int, default=10)
    ap.add_argument("--probe-spread", type=float, default=4.0)
    ap.add_argument("--panel-span-px", type=float, default=900.0)
    ap.add_argument("--panel-center", default="960,600")
    ap.add_argument(
        "--cam-size", type=int, default=300,
        help=("CCD 开窗边长。小窗口只截到光斑核心, 几何标定缺少远场上下文"
              "时会把波束腰解成十几像素的病态值 (w0=14.1)。"),
    )
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stamp = (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
             + f"_{int.from_bytes(os.urandom(2), 'big'):04x}")
    out_dir = Path(args.out) if args.out else (
        ROOT / "data" / "debug" / "uncertainty_holdout_hw" / stamp
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    meta: dict = {
        "issue": "#139 (H-26)",
        "commit_under_test": "080e0ac9de2d7d453676d174ae4a1845acb77ee8",
        "timestamp_utc": stamp,
        "args": vars(args),
        "holdout_count": HOLDOUT_COUNT,
        "fit_iterations": FIT_ITERATIONS,
    }
    (out_dir / "config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"输出: {out_dir}")

    px, py = (int(v) for v in args.panel_center.split(","))
    config = mil.SlmModelInLoopConfig(
        cam_type="daheng", cam_id=0, slm_number=1, slm_wavelength=1064,
        cam_size=args.cam_size, exposure_time_ms=args.exposure_ms,
        max_exposure_ms=0.0,
        region=args.region, n_orders=args.n_orders,
        far_field_padding=8, panel_span_px=args.panel_span_px,
        pupil_center=(px, py),
        probe_spread=args.probe_spread,
        step_a_iterations=FIT_ITERATIONS,
        holdout_probe_count=HOLDOUT_COUNT,
        # 本脚本只做探针策略对照, 不跑接受守卫, 所以熔断放到很大
        max_rejection_streak=99,
        n_calibration_probes=8,
        dtype="float64",
    )

    bench = mil._HardwareBench(config)
    try:
        _log("SLM #1 + 大恒 CCD 已打开 (memory 模式)")
        flat = np.zeros((args.region, args.region), dtype=np.float64)
        bench.display(flat)
        raw = bench.measure()
        roi_center = zero_order_center(raw, refine=False)
        _log(f"平场 raw_max={np.max(raw):.1f} 零级 argmax (x,y)={roi_center}")

        _log("几何标定中 (本次会话现标, 不沿用历史文件)...")
        geometry, _records = mil._calibrate_geometry(config, bench)
        _log(f"geometry [{geometry.method}] far_field={geometry.far_field_size} "
             f"cam/model={geometry.camera_px_per_model_px:.4f} "
             f"corr={geometry.correlation:.4f}")
        if geometry.correlation < float(config.min_geometry_correlation):
            _log(f"!! 几何相关 {geometry.correlation:.4f} < "
                 f"{config.min_geometry_correlation} —— 本实验系统不在物理预测模型描述范围内, "
                 f"验证集相关性不可信, 中止")
            meta["status"] = "geometry_rejected"
            meta["geometry_correlation"] = float(geometry.correlation)
            (out_dir / "summary.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            return 2

        cell_scale = float(geometry.camera_px_per_model_px)   # 硬件分支
        rot = 1
        meta["geometry"] = {
            "method": geometry.method,
            "correlation": float(geometry.correlation),
            "camera_px_per_model_px": float(geometry.camera_px_per_model_px),
            "far_field_size": int(geometry.far_field_size),
            "roi_center": list(roi_center),
        }

        def to_model(measured_frame: np.ndarray) -> np.ndarray:
            return mil._to_model_grid(
                prepare_frame(measured_frame), roi_center,
                int(geometry.far_field_size), cell_scale, rot,
            )

        def capture(phase_model: np.ndarray) -> np.ndarray:
            bench.display(np.asarray(phase_model, dtype=np.float64))
            return to_model(bench.measure())

        opt_probe = ZernikeCoefficientOptimizer(
            n_orders=args.n_orders, region=args.region,
            radius=args.region // 2, lr=0.05, device="cpu", dtype="float64",
            seed=None, far_field_size=int(geometry.far_field_size),
            frozen_modes=(1, 2, 3),
        )
        amp = np.zeros((args.region, args.region), dtype=np.float64)
        yy, xx = np.mgrid[0:args.region, 0:args.region]
        c = (args.region - 1) / 2.0
        amp[(yy - c) ** 2 + (xx - c) ** 2 <= c ** 2] = 1.0
        opt_probe.set_source_amplitude(amp)
        n_coeffs = int(np.asarray(opt_probe.coefficients).size)
        _log(f"物理预测模型 n_coeffs={n_coeffs} region={args.region} "
             f"far_field={geometry.far_field_size}")

        # 共享验证集探针: 固定随机, **各变体完全相同**
        holdout_phases = [
            mil._probe_phase(args.region, args.probe_spread, 777_000 + k,
                             smooth=True)
            for k in range(HOLDOUT_COUNT)
        ]

        def score(coefficients: np.ndarray, frames) -> tuple[float, list[float]]:
            vals = [
                float(mil._prediction_correlation(
                    opt_probe.forward_intensity(coefficients, ph), meas))
                for meas, ph in frames
            ]
            return (float(np.mean(vals)) if vals else float("nan")), vals

        arms = [
            ("A8", 8, "fixed", 1.5),
            ("A3", 3, "fixed", 1.5),
            ("B6_s1.5", 6, "uncertainty", 1.5),
            ("B6_s0", 6, "uncertainty", 0.0),
            ("B3_s0", 3, "uncertainty", 0.0),
            ("B2_s0", 2, "uncertainty", 0.0),
        ]
        results: list[dict] = []
        print(f"\n{'变体':<9}{'探针':>5}{'验证集相关':>11}{'逐张':>26}"
              f"{'std(选中)':>12}{'塌缩?':>8}")
        for name, count, kind, sigma in arms:
            # 验证集**先测**, 再测训练集 —— 交错顺序, 漂移落在两个变体差值里
            holdout_frames = [(capture(ph), ph) for ph in holdout_phases]

            train_phases: list[np.ndarray] = []
            if kind == "fixed":
                train_phases = [
                    mil._probe_phase(args.region, args.probe_spread,
                                     555_000 + 10 * count + k, smooth=True)
                    for k in range(count)
                ]
            else:
                orig = _bandlimit_patch(sigma)
                try:
                    sel = UncertaintyProbeSelector(
                        opt_probe, seed=555_000, n_samples=6, ridge=1e-3,
                        spread_cap=args.probe_spread,
                        spread_floor=0.75 * args.probe_spread,
                        lr=0.3, mode="maximize",
                    )
                    for _ in range(count):
                        for _ in range(args.steps):
                            sel.update()
                        train_phases.append(sel.current_phase())
                        sel.commit()
                finally:
                    from ao_shaping.algorithm.signal_processing import (
                        uncertainty_probe_selector as ups,
                    )
                    ups.UncertaintyProbeSelector = orig

            stds = [float(np.std(p)) for p in train_phases]
            floor, cap = 0.75 * args.probe_spread, args.probe_spread
            collapsed = all(s < floor - 1e-6 or s > cap + 1e-6 for s in stds)

            train_frames = [(capture(ph), ph) for ph in train_phases]
            fitted, _loss, _it, _opt = mil._fit_aberration_at_probes(
                opt_probe, train_frames, FIT_ITERATIONS,
            )
            corr_mean, corr_each = score(np.asarray(fitted), holdout_frames)

            row = {
                "arm": name, "kind": kind, "n_probes": count,
                "bandlimit_sigma": sigma if kind == "uncertainty" else None,
                "holdout_correlation": corr_mean,
                "holdout_correlation_each": corr_each,
                "probe_std": stds,
                "collapsed": bool(collapsed),
                "spread_band": [floor, cap],
                "coeff_norm": float(np.linalg.norm(fitted)),
            }
            results.append(row)
            each = "[" + " ".join(f"{v:.3f}" for v in corr_each) + "]"
            _log(f"{name:<10}{count:>5}{corr_mean:>11.4f}{each:>26}"
                 f"{min(stds):>6.2f}-{max(stds):<5.2f}"
                 f"{'YES' if collapsed else 'no':>8}")
            np.save(out_dir / f"{name}_coeff.npy", np.asarray(fitted))
            for k, ph in enumerate(train_phases):
                np.save(out_dir / f"{name}_probe{k}.npy", ph)

        # ---- 汇总 ---------------------------------------------------------
        by = {r["arm"]: r for r in results}
        base = by["A8"]["holdout_correlation"]
        print("\n" + "=" * 84)
        print("相对基线 A8 (8 张固定随机) 的独立验证集相关性")
        print("=" * 84)
        print(f"{'变体':<9}{'探针':>5}{'验证集相关':>11}{'Δ vs A8':>11}{'每张探针换到的':>18}")
        for r in results:
            d = r["holdout_correlation"] - base
            per = d / r["n_probes"] if r["n_probes"] else float("nan")
            print(f"{r['arm']:<10}{r['n_probes']:>5}"
                  f"{r['holdout_correlation']:>11.4f}{d:>+11.4f}{per:>18.5f}")

        budget = [r for r in results if r["arm"] in ("B2_s0", "B3_s0", "B6_s0")]
        a3 = by["A3"]["holdout_correlation"]
        a8 = by["A8"]["holdout_correlation"]
        summary = {
            "baseline_A8": a8,
            "baseline_A3": a3,
            "uncertainty_beats_A8_at_same_budget": {
                r["arm"]: bool(r["holdout_correlation"] > a8) for r in budget
            },
            "budget_comparison_B_vs_A3": {
                r["arm"]: bool(r["holdout_correlation"] >= a3) for r in budget
            },
            "any_collapse": any(r["collapsed"] for r in results),
            "probe_saved_vs_A8": {
                r["arm"]: int(a8 - r["holdout_correlation"]) and
                          8 - r["n_probes"] for r in budget
            },
        }
        meta["status"] = "completed"
        meta["arms"] = results
        meta["summary"] = summary
        (out_dir / "summary.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        _log("=" * 84)
        _log(f"基线 A8={a8:.4f}  A3={a3:.4f}")
        for r in results:
            if r["arm"] != "A8":
                _log(f"{r['arm']}: {r['holdout_correlation']:.4f} "
                     f"(Δ={r['holdout_correlation'] - a8:+.4f}, "
                     f"{r['n_probes']} 张)")
        _log(f"塌缩检查: {'有变体塌缩' if summary['any_collapse'] else '全部在幅度带内'}")
        _log(f"summary: {out_dir / 'summary.json'}")
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
            bench.close()
            _log("设备已释放")
        except Exception as exc:
            _log(f"关闭异常: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())