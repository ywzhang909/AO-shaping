# -*- coding: utf-8 -*-
"""离线探针：target_lock run b 三个根因的数值定案（不碰硬件）。

对应 docs/dev/agent_notes_20261123.md 待回填的验证项：
  ① 今日零相位光斑的信号质心与 11-04 检查点 origin 的 (dx, dy)；
  ② 「auto-spot 窗口 + registration 平移」与「origin 窗口 + 无平移」在
     ``FrozenDefocusForward`` 上逐点等价（模型几何不失配）；
  ③ tilt 相位斜坡的 rad → CCD px 标度（1/5/10/20/40/80 rad 幅度扫 + wrapped 对照；≈0.628 px/(pv rad)，
      2 rad ≈ 1.3 px，否掉 2 rad≈117 px 的估计）；
  ④ 窗口口径阈值预验证：窗口图 + 窗口目标时，``calibrate_lock_thresholds``
     推出的门禁对 run b 标定帧是否放行；
  ⑤ ``FrozenDefocusForward.__init__`` 构造耗时（逐轮 rebuild 是否可行）。

输入只读 run b 落盘证据与 11-04 冻结参数（只读），输出写
``data/debug/target_lock_probe_20261123/``（工作树内，不碰原始数据）。
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent.parent
PROBE_OUT = WORKTREE / "data" / "debug" / "target_lock_probe_20261123"
RUN_B = WORKTREE / "data" / "debug" / "target_lock_20261123b"
RUN_B_CAL = RUN_B / "calibration"
FROZEN_DIR = WORKTREE / "data" / "debug" / "slm_transfer" / "20261104_192122" / "fit"

REPORT: dict[str, Any] = {"probe": "target_lock_probe_20261123", "items": {}}


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# ① 光斑信号质心与偏移
# --------------------------------------------------------------------------- #
def _load_calibration_frames() -> list[np.ndarray]:
    npz = np.load(RUN_B_CAL / "calibration_frames.npz")
    if "frames" in npz.files:
        stack = np.asarray(npz["frames"], dtype=np.float64)
        return [stack[i] for i in range(stack.shape[0])]
    return [np.asarray(npz[key], dtype=np.float64) for key in npz.files]


def item1_spot_centroid() -> dict[str, Any]:
    """run b 标定帧的光斑信号质心（掩膜阈值法）与相对 origin 的偏移。

    暗地板加权的整帧质心不是光斑位置（run b 整帧门禁质心 (1299.6, 963.7) 即
    此陷阱）；掩膜阈值法只取显著高于暗地板的像素加权。
    """
    images = _load_calibration_frames()
    calibration = json.loads((RUN_B_CAL / "calibration.json").read_text(encoding="utf-8"))
    plan = json.loads((RUN_B / "plan.json").read_text(encoding="utf-8"))
    window = plan["window"]
    origin = (1304.27, 942.63)

    per_frame: list[dict[str, Any]] = []
    centroids: list[tuple[float, float]] = []
    for idx, arr in enumerate(images):
        dark = float(np.median(arr))
        sigma = float(arr.std())
        threshold = dark + 5.0 * sigma
        mask = arr > threshold
        if not mask.any():
            per_frame.append({"frame_index": idx, "signal_pixels": 0})
            continue
        ys, xs = np.nonzero(mask)
        weights = arr[mask] - dark
        cx = float(np.sum(xs * weights) / np.sum(weights))
        cy = float(np.sum(ys * weights) / np.sum(weights))
        centroids.append((cx, cy))
        per_frame.append(
            {
                "frame_index": idx,
                "signal_pixels": int(mask.sum()),
                "threshold": float(threshold),
                "spot_centroid": [cx, cy],
                "peak": float(arr.max()),
            }
        )
    if centroids:
        mean_cx = float(np.mean([c[0] for c in centroids]))
        mean_cy = float(np.mean([c[1] for c in centroids]))
        spot_sigma = float(np.hypot(
            np.std([c[0] for c in centroids]), np.std([c[1] for c in centroids])
        ))
    else:
        raise RuntimeError("run b 标定帧中没有任何一帧检出信号（掩膜阈值法）")

    dx = mean_cx - origin[0]
    dy = mean_cy - origin[1]
    win_cx = float(window["centre"][0])
    win_cy = float(window["centre"][1])
    return {
        "origin": list(origin),
        "frame_count": len(images),
        "frame_shape": list(images[0].shape),
        "spot_centroid_mean": [mean_cx, mean_cy],
        "spot_centroid_sigma_px": spot_sigma,
        "offset_dx_dy_ccd_px": [dx, dy],
        "offset_magnitude_px": float(np.hypot(dx, dy)),
        "window": {
            "centre": [win_cx, win_cy],
            "span": int(window["span"]),
            "grid_size": int(window["grid_size"]),
            "x_range": [win_cx - window["span"], win_cx + window["span"]],
            "y_range": [win_cy - window["span"], win_cy + window["span"]],
        },
        "calibration_json_selected": {
            "sample": {
                k: calibration.get("sample", {}).get(k)
                for k in (
                    "dark_mean", "dark_sigma", "peak_counts", "total_counts",
                    "saturated_fraction", "full_scale", "pixel_count",
                )
            },
            "lock_config": {
                k: calibration.get("lock_config", {}).get(k)
                for k in (
                    "saturation_counts", "max_saturated_fraction",
                    "min_peak_counts", "min_total_counts",
                    "max_centroid_error_px",
                )
            },
            "grid_spacing_px": calibration.get("grid_spacing_px"),
            "position_tolerance_ccd_px": calibration.get("position_tolerance_ccd_px"),
        },
        "per_frame": per_frame,
    }


# --------------------------------------------------------------------------- #
# ② registration 等价性 + ⑤ 构造耗时
# --------------------------------------------------------------------------- #
def item2_registration_equivalence(spot_xy: tuple[float, float]) -> dict[str, Any]:
    """spot 窗口 + shift ≡ origin 窗口，并记录 forward 构造耗时。

    ``registration_sample_grid`` 返回 ``xy − (dx, dy)``。``analysis_grid`` 会把
    窗口中心 round 成整数像素，因此严格等价要求 shift 也是整数：
    ``dx = round(spot) − round(origin)``。分数 shift（spot − origin）会留下
    亚像素采样偏移 —— 本探针把它作为「亚像素敏感度」单独记录：预测形状对
    ~0.5 px 的采样偏移有 ~10% 量级的归一化差异（锐利 PSF 的本性），但闭环
    位置容差 5 px 下无关紧要，且模型与实测帧用同一 xy 采样、无口径错位。
    """
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
        load_frozen_parameters,
    )
    from ao_shaping.gui.slm.closed_loop_shaping import analysis_grid

    params = load_frozen_parameters(FROZEN_DIR)
    inputs = np.load(FROZEN_DIR / "inputs.npz")
    phases = np.asarray(inputs["phases"], dtype=np.float64)

    origin = (float(params.origin_xy[0]), float(params.origin_xy[1]))
    xy_origin = analysis_grid(origin, 120)
    xy_spot = analysis_grid(spot_xy, 120)
    # analysis_grid 用 np.rint：整数平移量由两处 round 差给出
    dx_int = int(np.rint(spot_xy[0]) - np.rint(origin[0]))
    dy_int = int(np.rint(spot_xy[1]) - np.rint(origin[1]))
    dx_frac = spot_xy[0] - origin[0]
    dy_frac = spot_xy[1] - origin[1]

    t0 = time.perf_counter()
    fwd_origin = FrozenDefocusForward(params, xy_origin)
    t_origin = time.perf_counter() - t0
    params_int = dataclasses.replace(
        params, registration_shift_ccd_px=(float(dy_int), float(dx_int))
    )
    t0 = time.perf_counter()
    fwd_spot_int = FrozenDefocusForward(params_int, xy_spot)
    t_spot = time.perf_counter() - t0
    params_frac = dataclasses.replace(
        params, registration_shift_ccd_px=(dy_frac, dx_frac)
    )
    fwd_spot_frac = FrozenDefocusForward(params_frac, xy_spot)
    REPORT["items"]["5_forward_build_seconds"] = {
        "origin_forward_s": round(t_origin, 3),
        "spot_forward_s": round(t_spot, 3),
        "note": (
            "FrozenDefocusForward.__init__ 单次构造耗时（首次含 CUDA 初始化，"
            "第二次为热身后量级）；逐轮 rebuild 量级参考"
        ),
    }

    test_phases: list[tuple[str, np.ndarray]] = [
        ("zero", np.zeros((params.grid_size, params.grid_size), dtype=np.float64)),
        ("inputs_phase_0", phases[0]),
        ("inputs_phase_12", phases[12]),
    ]

    def _pred(forward: FrozenDefocusForward, phase: np.ndarray) -> np.ndarray:
        return np.asarray(
            forward.predict(phase, exposure_ms=0.05).detach().cpu().numpy(),
            dtype=np.float64,
        )

    rows: list[dict[str, Any]] = []
    for label, phase in test_phases:
        base = _pred(fwd_origin, phase)
        a_norm = base / max(float(base.max()), 1e-9)
        for variant, forward in (
            ("integer_shift", fwd_spot_int),
            ("fractional_shift", fwd_spot_frac),
        ):
            other = _pred(forward, phase)
            b_norm = other / max(float(other.max()), 1e-9)
            rel = float(
                np.linalg.norm(a_norm - b_norm) / max(np.linalg.norm(a_norm), 1e-9)
            )
            rows.append(
                {
                    "phase": label,
                    "variant": variant,
                    "norm_relative_error": rel,
                    "peak_origin": float(base.max()),
                    "peak_spot": float(other.max()),
                }
            )
    int_rows = [r for r in rows if r["variant"] == "integer_shift"]
    frac_rows = [r for r in rows if r["variant"] == "fractional_shift"]
    return {
        "origin_xy": [origin[0], origin[1]],
        "spot_xy": [spot_xy[0], spot_xy[1]],
        "integer_shift_dx_dy": [dx_int, dy_int],
        "fractional_shift_dx_dy": [dx_frac, dy_frac],
        "rows": rows,
        "tolerance": 1e-4,
        "passed": all(r["norm_relative_error"] < 1e-4 for r in int_rows),
        "fractional_subpixel_note": (
            "分数 shift 相对整数 shift 多出的偏差 = 亚像素采样偏移 (~0.57 px) 的"
            "形状敏感度，非模型失配；最大值 "
            f"{max(r['norm_relative_error'] for r in frac_rows):.3e}"
        ),
    }


# --------------------------------------------------------------------------- #
# ③ tilt rad→px 标度（幅度扫 + wrapped 对照）
# --------------------------------------------------------------------------- #
#: 与 DriftBench.tilt 的注入量同口径：峰谷 rad，逐点线性张开。
TILT_SWEEP_AMPLITUDES = (1.0, 5.0, 10.0, 20.0, 40.0, 80.0)


def item3_tilt_scale(spot_xy: tuple[float, float]) -> dict[str, Any]:
    """在 spot 窗口 + 整数 registration 上实测 tilt 的 rad→CCD px 标度。

    第一版在 **origin 窗口 span-300** 上只扫 A=1 rad，且把交叉耦合分量当成了主轴
    标度（tilt-x 的主响应在 dy，tilt-y 的在 dx），报出 0.017/0.011 px/rad 的假值。
    这里改成与验收脚本完全一致的口径：

    * 窗口中心 = ``round(spot_xy)``，forward 用同一整数
      ``registration_shift_ccd_px=(dy, dx)``（与 item② 的等价性口径相同）；
    * 幅度扫 1/5/10/20/40/80 rad，逐幅度 centroid / argmax 都记；
    * 每个幅度另做 ``phase mod 2π`` 的 wrapped 对照，验证 SLM 相位回绕不改变
      注入量（相对差 ~1e-7）；
    * 最小二乘过原点拟合 ``dcent = M · A``，得到近似
      ``M = [[0, 0.62], [−0.63, 0]]`` 的交叉耦合矩阵，主轴标度约 0.628 px/(pv rad)。

    80 rad → 约 50 px，仍在 span=120 的窗口内，因此用同一个 forward/网格完成
    全部幅度，不需要像第一版那样临时把窗口开到 601 px。
    """
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
        load_frozen_parameters,
    )
    from ao_shaping.gui.slm.closed_loop_shaping import analysis_grid

    params = load_frozen_parameters(FROZEN_DIR)
    origin = (float(params.origin_xy[0]), float(params.origin_xy[1]))
    dx_int = int(np.rint(spot_xy[0]) - np.rint(origin[0]))
    dy_int = int(np.rint(spot_xy[1]) - np.rint(origin[1]))
    span = 120
    xy = analysis_grid(spot_xy, span)
    reg_params = dataclasses.replace(
        params, registration_shift_ccd_px=(float(dy_int), float(dx_int))
    )
    forward = FrozenDefocusForward(reg_params, xy)
    grid = np.asarray(xy, dtype=np.float64)
    n = params.grid_size
    axis = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    ramp_x = np.broadcast_to(axis[None, :], (n, n)).copy()
    ramp_y = np.broadcast_to(axis[:, None], (n, n)).copy()

    def _centroid_of(pred: np.ndarray) -> tuple[float, float]:
        w = np.clip(pred - float(np.median(pred)), 0.0, None)
        total = float(w.sum())
        if total <= 0.0:
            raise RuntimeError("预测强度全零，无法算质心")
        return (
            float((grid[..., 0] * w).sum() / total),
            float((grid[..., 1] * w).sum() / total),
        )

    def _argmax_xy(pred: np.ndarray) -> tuple[float, float]:
        i, j = np.unravel_index(int(np.argmax(pred)), pred.shape)
        return (float(grid[i, j, 0]), float(grid[i, j, 1]))

    def _predict(phase: np.ndarray) -> np.ndarray:
        pred = np.asarray(
            forward.predict(phase, exposure_ms=0.05).detach().cpu().numpy(),
            dtype=np.float64,
        )
        return np.squeeze(pred)

    base_pred = _predict(np.zeros((n, n), dtype=np.float64))
    base_c = _centroid_of(base_pred)
    base_am = _argmax_xy(base_pred)

    rows: list[dict[str, Any]] = []
    for amp in TILT_SWEEP_AMPLITUDES:
        for label, ramp in (("tilt-x", ramp_x), ("tilt-y", ramp_y)):
            pred = _predict(amp * ramp)
            wrapped = _predict(np.mod(amp * ramp, 2.0 * np.pi))
            cen = _centroid_of(pred)
            am = _argmax_xy(pred)
            rows.append(
                {
                    "amp": float(amp),
                    "axis": label,
                    "dcent": [cen[0] - base_c[0], cen[1] - base_c[1]],
                    "dargmax": [am[0] - base_am[0], am[1] - base_am[1]],
                    "wrap_diff": float(
                        np.linalg.norm(pred - wrapped) / max(np.linalg.norm(pred), 1e-9)
                    ),
                }
            )

    def _slope(axis_label: str, component: int) -> float:
        values = np.asarray([amp for amp in TILT_SWEEP_AMPLITUDES], dtype=np.float64)
        deltas = np.asarray(
            [r["dcent"][component] for r in rows if r["axis"] == axis_label],
            dtype=np.float64,
        )
        return float(np.dot(values, deltas) / np.dot(values, values))

    # dcent = M · [A_tilt-x, A_tilt-y]：行 = (dx, dy)，列 = (tilt-x, tilt-y)
    matrix = [[_slope("tilt-x", 0), _slope("tilt-y", 0)],
              [_slope("tilt-x", 1), _slope("tilt-y", 1)]]
    scale_x = abs(matrix[1][0])  # tilt-x 主响应在 dy
    scale_y = abs(matrix[0][1])  # tilt-y 主响应在 dx
    drift_2rad = 2.0 * max(scale_x, scale_y)

    # 把逐幅度表单独留档，便于以后重跑时直接 diff，不用从大报告里摘。
    (PROBE_OUT / "tilt_sweep.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "amplitudes_peak_valley_rad": list(TILT_SWEEP_AMPLITUDES),
        "wrapped_control": "phase mod 2π；wrap_diff 为归一化相对差",
        "window": {
            "centre": [spot_xy[0], spot_xy[1]],
            "span": span,
            "grid_size": int(grid.shape[0]),
            "registration_shift_ccd_px": [float(dy_int), float(dx_int)],
        },
        "predicted_spot_at_zero_phase": {
            "centroid": list(base_c),
            "argmax": list(base_am),
            "spot_xy": [spot_xy[0], spot_xy[1]],
            "centroid_vs_spot_px": float(
                np.hypot(base_c[0] - spot_xy[0], base_c[1] - spot_xy[1])
            ),
        },
        "rows": rows,
        # 兼容旧 main() 日志字段
        "amplitude_peak_valley_rad": 1.0,
        "px_per_peak_valley_rad": {"tilt-x": scale_x, "tilt-y": scale_y},
        "estimate_at_2rad_px": {"tilt-x": 2.0 * scale_x, "tilt-y": 2.0 * scale_y},
        "cross_coupling_matrix_dxdy_per_tiltxy": matrix,
        "run_b_injected": "2.0 rad peak-to-valley, write 4 起 20 次张开",
        "drift_2rad_vs_half_window_120px": {
            "drift_px": drift_2rad,
            "margin_px": 120.0 - drift_2rad,
            "inside_121_window": bool(drift_2rad < 120.0),
        },
    }


# --------------------------------------------------------------------------- #
# ④ 窗口口径阈值预验证
# --------------------------------------------------------------------------- #
def item4_window_thresholds(spot_xy: tuple[float, float]) -> dict[str, Any]:
    """模拟修好后的 run_calibration：统计与门禁全在窗口口径下推导。

    run b 的根因一：整帧统计推出 ``min_total_counts=30,916``（整帧 5M px 总
    量 1.55M 的 2%），而监督器看的是 121×121 窗口图（帧和 ~7.2k）→ 必判
    low_signal。这里把 run b 的 8 张标定帧 ``sample_on_grid`` 到 spot 窗口，
    窗口目标用中心矩形（模拟 --target field），门禁推导与检查都用窗口图。
    """
    from ao_shaping.gui.slm.closed_loop_shaping import (
        analysis_grid,
        grid_spacing_px,
        sample_on_grid,
        target_field,
    )
    from ao_shaping.gui.slm.target_lock import (
        CalibrationSample,
        calibrate_lock_thresholds,
        calibration_stats_from_frames,
    )
    from ao_shaping.gui.slm.target_lock import LockConfig

    images = _load_calibration_frames()
    xy = analysis_grid(spot_xy, 120)
    spacing = float(grid_spacing_px(xy))
    tgt = target_field(
        xy, centre_xy=(spot_xy[0], spot_xy[1]), half_width=20.0, half_height=20.0
    )
    window_frames = [sample_on_grid(f, xy) for f in images]

    # calibration_stats_from_frames 需要 target 与帧同形：窗口目标已是 121×121
    sample: CalibrationSample = calibration_stats_from_frames(
        window_frames, target=tgt, full_scale=255.0
    )
    cfg: LockConfig = calibrate_lock_thresholds(
        sample, position_tolerance_ccd_px=5.0, grid_spacing_px=spacing
    )

    rows: list[dict[str, Any]] = []
    reasons_union: set[str] = set()
    for idx, wf in enumerate(window_frames):
        arr = np.asarray(wf, dtype=np.float64)
        reasons: list[str] = []
        sat_frac = float(np.mean(arr >= cfg.saturation_counts))
        if sat_frac > cfg.max_saturated_fraction:
            reasons.append("overexposed")
        peak = float(arr.max())
        total = float(arr.sum())
        if peak < cfg.min_peak_counts or total < cfg.min_total_counts:
            reasons.append("low_signal")
        if float(arr.std()) <= 0.0:
            reasons.append("flat_frame")
        reasons_union.update(reasons)
        rows.append(
            {
                "frame_index": idx,
                "reasons": reasons,
                "peak": peak,
                "total": round(total, 1),
                "saturated_fraction": sat_frac,
            }
        )
    return {
        "spot_xy": [spot_xy[0], spot_xy[1]],
        "grid_spacing_px": spacing,
        "sample": {
            "dark_mean": sample.dark_mean,
            "dark_sigma": sample.dark_sigma,
            "peak_counts": sample.peak_counts,
            "total_counts": sample.total_counts,
            "saturated_fraction": sample.saturated_fraction,
            "full_scale": sample.full_scale,
            "pixel_count": sample.pixel_count,
        },
        "config": {
            "saturation_counts": cfg.saturation_counts,
            "max_saturated_fraction": cfg.max_saturated_fraction,
            "min_peak_counts": cfg.min_peak_counts,
            "min_total_counts": cfg.min_total_counts,
        },
        "threshold_total_vs_frame_total": [
            cfg.min_total_counts, float(np.mean([r["total"] for r in rows]))
        ],
        "rows": rows,
        "all_frames_pass": all(not r["reasons"] for r in rows),
        "reasons_union": sorted(reasons_union),
    }


def main() -> None:
    _log("=== target_lock 离线探针 ===")
    PROBE_OUT.mkdir(parents=True, exist_ok=True)

    _log("[1/4] ① 光斑质心与偏移 ...")
    item1 = item1_spot_centroid()
    REPORT["items"]["1_spot_centroid"] = item1
    spot = item1["spot_centroid_mean"]
    _log(
        f"  spot≈({spot[0]:.1f}, {spot[1]:.1f})  offset=(dx {item1['offset_dx_dy_ccd_px'][0]:.1f},"
        f" dy {item1['offset_dx_dy_ccd_px'][1]:.1f}) px  |offset|="
        f"{item1['offset_magnitude_px']:.1f} px"
    )
    spot_xy = (float(spot[0]), float(spot[1]))

    _log("[2/4] ② registration 等价性 + ⑤ forward 构造耗时 ...")
    item2 = item2_registration_equivalence(spot_xy)
    REPORT["items"]["2_registration_equivalence"] = item2
    for row in item2["rows"]:
        _log(
            f"  {row['phase']}: rel-err {row['norm_relative_error']:.3e}"
            f"  peak {row['peak_origin']:.1f}/{row['peak_spot']:.1f}"
        )
    _log(f"  passed={item2['passed']}  build_s="
         f"{REPORT['items']['5_forward_build_seconds']['origin_forward_s']}/"
         f"{REPORT['items']['5_forward_build_seconds']['spot_forward_s']}")

    _log("[3/4] ③ tilt rad→px 标度 ...")
    item3 = item3_tilt_scale(spot_xy)
    REPORT["items"]["3_tilt_scale"] = item3
    _log(
        f"  px/(pv rad): tilt-x {item3['px_per_peak_valley_rad']['tilt-x']:.3f},"
        f" tilt-y {item3['px_per_peak_valley_rad']['tilt-y']:.3f};"
        f" 2 rad → {item3['estimate_at_2rad_px']['tilt-x']:.2f} px"
        f"  cross-coupling [[{item3['cross_coupling_matrix_dxdy_per_tiltxy'][0][0]:.3f},"
        f" {item3['cross_coupling_matrix_dxdy_per_tiltxy'][0][1]:.3f}],"
        f" [{item3['cross_coupling_matrix_dxdy_per_tiltxy'][1][0]:.3f},"
        f" {item3['cross_coupling_matrix_dxdy_per_tiltxy'][1][1]:.3f}]];"
        f"  wrap_diff max="
        f"{max(r['wrap_diff'] for r in item3['rows']):.2e}"
    )

    _log("[4/4] ④ 窗口口径阈值预验证 ...")
    item4 = item4_window_thresholds(spot_xy)
    REPORT["items"]["4_window_thresholds"] = item4
    _log(
        f"  min_total={item4['config']['min_total_counts']:.1f} vs 窗口帧均值 "
        f"{float(np.mean([r['total'] for r in item4['rows']])):.1f};"
        f" min_peak={item4['config']['min_peak_counts']:.1f} vs 峰值 "
        f"{item4['sample']['peak_counts']:.1f}"
    )
    _log(f"  all_frames_pass={item4['all_frames_pass']}  reasons={item4['reasons_union']}")

    out_path = PROBE_OUT / "probe_report.json"
    out_path.write_text(
        json.dumps(REPORT, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(f"探针报告: {out_path}")


if __name__ == "__main__":
    main()
