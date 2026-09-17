"""SLM Zernike 模式法波前矫正 — 三阶段完整流程 (2026-09-15, 已按实测问题优化).

阶段 1 — 自动定标 + 参考波前
  1.1 自动测光束在 SLM 上的半径 (Zernike R 扫描, defocus 响应最大者)
  1.2 自动测光束中心偏移 → shift_x/shift_y (defocus tip/tilt 零点法, 细扫 5px)
  1.3 SLM **纯 0 相位** → create_default_user_ref → save_user_ref → set_ref_plane(custom=True)
  1.4 **记录并校验纯平波前平整度** (多帧 RMS/PV, 断言 < 阈值)

阶段 2 — 稳健响应矩阵 (已修正 2026-09-15 实测暴露的两个缺陷)
  对 模式 × 尺寸(R) × 幅度(±) 网格逐点:
    - `show_phase` 自动等 LCOS 像素翻转 + **100 ms 冗余**
    - **WFS 有效性门控** (`wfs_validity`): 光斑有效比不足 → 剔除该点
    - WFS 多帧平均 (中位数聚合), **全量原始读数增量落盘**
  线性度判据 (**已修正**): 响应已按单位幅度归一化 → 线性响应表现为 |resp| **恒定**,
  故用 `linearity_metrics` 的 **CV + 方向一致性 cos** (原 slope/R² 判据失效)。
  尺寸选择 (**已修正**): **先剔除不通过线性度的 (模式,R)**, 再在通过项中选响应最强,
  避免被 R≈光束半径 时 WFS 拟合崩溃产生的异常大响应误导。

阶段 3 — 闭环反向矫正 (迭代 + 增益/泄漏)
  3.1 恢复 WFS **内部参考** (`set_ref_plane(custom=False)`)
  3.2 读内部参考下的波前 Zernike 像差 w
  3.3 迭代: `c ← leak·c − gain·pinv(M)·w`, 逐轮加载并测残差
  3.4 报告每轮 RMS/PV 与最终改善

⚠️ 索引约定: WFS `get_zernike()` 返回 67 长数组, `coeff[1..66]` 有效 (index 0 未用),
排序为**顺序 m 枚举** (非标准 Noll), 详见 `slm_zernike_common`。

用法:
    python -m ao_shaping.tools.slm.slm_zernike_correction --stage all
    python -m ao_shaping.tools.slm.slm_zernike_correction --stage auto
    python -m ao_shaping.tools.slm.slm_zernike_correction --stage matrix --quick
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
from loguru import logger

from ao_shaping.drivers.slm import Santec
from ao_shaping.drivers.wfs import ThorlabWFS
from ao_shaping.tools.slm.slm_zernike_common import (
    DEFAULT_EXPOSURE_MS,
    MAX_EXPOSURE_MS,
    PANEL_H,
    PANEL_W,
    SETTLE_REDUNDANCY_S,
    WFS_ZERNIKE_ORDER,
    calibrate_center_shift,
    collect_device_info,
    diagnose_beam_radius,
    effective_cond,
    export_correction_csv,
    flat_gray,
    linearity_metrics,
    make_phase,
    measure_wavefront,
    measure_zernike,
    nm_of,
    safe_pinv,
    show_phase,
    um_to_waves,
    wfs_validity,
)
from ao_shaping.tools.slm.slm_scan_analysis import outlier_mask
from ao_shaping.utils.pattern_helper import PatternHelper


# ─────────────────────────── 阶段 1 ───────────────────────────

def setup_flat_reference(slm: Santec, wfs: ThorlabWFS, n_avg: int,
                         extra_sleep: float) -> dict[str, Any]:
    """SLM 纯 0 相位 → 用户参考, 并校验纯平波前平整度."""
    click.echo("\n[1.3/1.4] SLM 纯 0 相位 → 用户参考 + 平整度校验")
    slm.display_data(flat_gray(), wait_time_s=0.5)
    time.sleep(0.4)
    wfs.take_image(n_sample=1, dynamicNoiseCut=True)
    ok = wfs.create_default_user_ref()
    backup = wfs.save_user_ref(backup_dir=Path("data/calibration"))
    wfs.set_ref_plane(custom=True)
    click.echo(f"   create={ok}, save={backup}, use_custom_ref={wfs.use_custom_ref}")
    if not ok or not wfs.use_custom_ref:
        raise RuntimeError("用户参考创建/激活失败")

    rec = measure_wavefront(wfs, n_avg)
    if rec is None:
        raise RuntimeError("平整度测量失败")
    _, st = rec
    z = measure_zernike(wfs, n_avg, WFS_ZERNIKE_ORDER)
    zt = float(np.linalg.norm(z[2:7])) / 0.532 if z is not None else float("nan")
    click.echo(f"   纯平波前: RMS={st['rms']:.4f}λ, PV={st['diff']:.4f}λ, "
               f"|z[2..6]|={zt:.4f}λ")
    flat_ok = st["rms"] < 0.05
    click.echo(f"   [{'OK' if flat_ok else 'WARN'}] 平整度 RMS < 0.05λ → {flat_ok}")
    return {"create_ok": bool(ok), "backup_ref": str(backup),
            "use_custom_ref": bool(wfs.use_custom_ref),
            "flat_rms_lam": st["rms"], "flat_pv_lam": st["diff"],
            "flat_z_norm_lam": zt, "flat_ok": bool(flat_ok)}


# ─────────────────────────── 阶段 2 ───────────────────────────

def scan_response_matrix(slm: Santec, wfs: ThorlabWFS, ph: PatternHelper,
                         modes: list[int], radii: list[float], amps: list[float],
                         n_avg: int, extra_sleep: float, incr_path: Path,
                         order: int = WFS_ZERNIKE_ORDER,
                         outlier_factor: float = 3.0,
                         coverage_tol: int = 1
                         ) -> tuple[np.ndarray, list, dict]:
    """模式 × 尺寸 × ±幅度 网格扫描; 有效性门控 + 逐点剔除 + 统一半径 + 正确线性度判据.

    **半径一致性 (关键)**: 矩阵所有列必须由**同一半径**构建 —— 矫正相位由
    `make_phase(..., radius=r_used)` 以单一半径生成, 若标定列来自不同半径,
    Zernike 归一化不匹配会使系数被错误缩放 (R=300 标定按 R=200 加载 → 相位放大 2.25×)。
    故多半径仅用于**诊断**, 矩阵统一取"通过模式数最多"的半径 (并列取较小者, 耦合更强)。
    """
    click.echo(f"\n[2] 响应矩阵扫描: {len(modes)} 模式 × {len(radii)} 尺寸 × "
               f"{len(amps)} 幅度(±) × {n_avg} 帧")
    raw: list[dict] = []
    curves: dict[tuple[int, float], list[np.ndarray]] = {}
    amps_of: dict[tuple[int, float], list[float]] = {}
    rejected: list[dict] = []
    points: list[dict] = []          # 逐点诊断 (debug): ±对称性 / 响应 / 有效比

    total = len(modes) * len(radii) * len(amps)
    k = 0
    for midx in modes:
        nm = nm_of(midx)
        for R in radii:
            R = float(R)
            for a in amps:
                k += 1
                a_waves = a / (2 * np.pi)
                zs: dict[int, np.ndarray | None] = {}
                for sign in (+1, -1):
                    phase = make_phase(ph, {nm: sign * a}, R)
                    show_phase(slm, phase, extra_sleep)
                    val = wfs_validity(wfs)
                    if not val["ok"]:
                        logger.warning("[{}] R={:.0f} A={:.1f} sign={} 光斑有效比 "
                                       "{:.2f} → 剔除", midx, R, a, sign,
                                       val["valid_ratio"])
                        zs[sign] = None
                        continue
                    z = measure_zernike(wfs, n_avg, order)
                    zs[sign] = z
                    if z is not None:
                        raw.append({"dll_index": midx, "nm": list(nm), "radius": R,
                                    "amp_rad": sign * a, "sign": sign,
                                    "valid_ratio": val["valid_ratio"],
                                    "readout_um": z.tolist()})
                zp, zn = zs.get(+1), zs.get(-1)
                if zp is None or zn is None:
                    click.echo(f"   [{k}/{total}] [{midx}] {nm} R={R:.0f} A={a:.1f} "
                               f"无效 (剔除)")
                    continue
                diff = um_to_waves((zp[1:] - zn[1:]) / 2.0)   # µm → λ (推拉差分 = 响应)
                sym = um_to_waves((zp[1:] + zn[1:]) / 2.0)    # µm → λ (应 ≈ 基线; 大则 ± 不对称)
                col = diff / a_waves
                curves.setdefault((midx, R), []).append(col)
                amps_of.setdefault((midx, R), []).append(a)
                points.append({
                    "dll_index": midx, "nm": list(nm), "radius": R, "amp_rad": a,
                    "amp_waves": a_waves,
                    "valid_ratio_pos": next((p["valid_ratio"] for p in raw
                                             if p["dll_index"] == midx
                                             and p["radius"] == R
                                             and p["amp_rad"] == a), None),
                    "resp_norm_lam": float(np.linalg.norm(col)),
                    "diff_norm_lam": float(np.linalg.norm(diff)),
                    "sym_norm_lam": float(np.linalg.norm(sym)),
                    "snr_vs_sym": (float(np.linalg.norm(diff) / np.linalg.norm(sym))
                                   if np.linalg.norm(sym) > 0 else None),
                    "diff_lam": diff.tolist(), "sym_lam": sym.tolist(),
                })
                click.echo(f"   [{k}/{total}] [{midx:2d}] {str(nm):9s} R={R:5.0f} "
                           f"A={a:5.1f}rad  |resp|={np.linalg.norm(col):.3f}  "
                           f"[±] |z+-z-|/2={np.linalg.norm(diff):.3f} "
                           f"|z++z-|/2={np.linalg.norm(sym):.3f}")
            incr_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    # 逐点异常剔除 (幅度量级合理性): 同 (mode,R) 组内 |resp| 偏离中位数
    # > outlier_factor 倍 → 该幅度点剔除。`wfs_validity` 只查光斑有效比, 抓不到
    # "光斑正常但 LSF 拟合崩溃" (实测 |resp| 可暴涨到 10³)。
    for key, vecs in list(curves.items()):
        if len(vecs) < 2:
            continue
        norms = np.array([float(np.linalg.norm(v)) for v in vecs])
        keep = outlier_mask(norms, outlier_factor)
        if not keep.all():
            click.echo(f"   逐点剔除 [{key[0]}] R={key[1]:.0f}: 保留 {int(keep.sum())}"
                       f"/{len(vecs)}  (|resp|={np.round(norms, 3).tolist()})")
            curves[key] = [v for v, kp in zip(vecs, keep) if kp]
            amps_of[key] = [a for a, kp in zip(amps_of[key], keep) if kp]

    # 线性度 metrics (正确判据: CV + 方向一致性)
    metrics: list[dict] = []
    for (midx, R), vecs in sorted(curves.items()):
        m = linearity_metrics(vecs, amps_of[(midx, R)])
        metrics.append({"dll_index": midx, "nm": list(nm_of(midx)), "radius": R,
                        "n_amps": m["n_amps"], "norms": m["norms"],
                        "cv": m["cv"], "cos_min": m["cos_min"], "ok": m["ok"]})
        if not m["ok"]:
            rejected.append({"dll_index": midx, "radius": R, "cv": m["cv"],
                             "cos_min": m["cos_min"]})

    click.echo(f"\n   线性度: 通过 {len(metrics) - len(rejected)}/{len(metrics)} "
               f"(mode,R) 组合")
    if rejected:
        click.echo("   被拒 (CV>0.15 或 cos<0.9): " +
                   ", ".join(f"[{r['dll_index']}]R{r['radius']:.0f}"
                             f"(cv={r['cv']:.2f},cos={r['cos_min']:.2f})"
                             for r in rejected))

    # 统一矩阵半径: 在"覆盖度容差"内取**最小**半径。
    # 实测 (离线重建 raw_scan): R=300 → 13/14 列, cond=10.9, 反解降 91.7%;
    # R=400 → 14/14 列但 cond=52.3 (耦合更弱, 条件数差 5×); R=200 → 仅 7/14 列。
    # 故不能简单取"通过数最多"(会选 R=400), 应在覆盖度接近时取较小 R。
    pass_count = {float(R): sum(1 for m in metrics
                                if m["radius"] == float(R) and m["ok"])
                  for R in radii}
    best_n = max(pass_count.values()) if pass_count else 0
    cand = [R for R in sorted(pass_count) if pass_count[R] >= best_n - coverage_tol]
    r_matrix = cand[0] if cand else max(sorted(pass_count), key=lambda R: pass_count[R])
    click.echo(f"   统一矩阵半径 R={r_matrix:.0f}px "
               f"(各半径通过数: { {int(kk): v for kk, v in sorted(pass_count.items())} }, "
               f"容差 {coverage_tol})")

    n_rows = 66
    matrix = np.zeros((n_rows, len(modes)), dtype=np.float64)
    var = np.zeros_like(matrix)
    chosen: dict[int, float] = {}
    for col, midx in enumerate(modes):
        hit = next((m for m in metrics if m["dll_index"] == midx
                    and m["radius"] == r_matrix and m["ok"]), None)
        if hit is None:
            click.echo(f"   [WARN] 模式 [{midx}] 在 R={r_matrix:.0f} 未通过线性度 "
                       f"→ 该列置零")
            continue
        chosen[midx] = r_matrix
        vecs = curves[(midx, r_matrix)]
        matrix[:, col] = np.mean(np.array(vecs), axis=0)
        var[:, col] = np.var(np.array(vecs), axis=0) if len(vecs) > 1 else 0.0
    click.echo(f"   主矩阵形状 {matrix.shape}; 有效列 {len(chosen)}/{len(modes)} "
               f"(统一 R={r_matrix:.0f}px)")
    return matrix, metrics, {
        "matrix_radius": r_matrix,
        "pass_count_by_radius": {int(kk): int(v) for kk, v in pass_count.items()},
        "chosen_radius": {int(kk): float(v) for kk, v in chosen.items()},
        "rejected": rejected, "curves_n": len(curves),
        "variance": var.tolist(),
        "points": points,          # 逐点诊断 (debug 用; main 会单独落盘)
    }


# ─────────────────────────── debug 落盘 ───────────────────────────

def save_matrix_debug(dbg: Path, matrix: np.ndarray, variance: np.ndarray,
                      modes: list[int], r_matrix: float, device: dict,
                      shift: list[int] | None, flat_ref: dict | None,
                      amp_rad: float, n_avg: int, extra: dict) -> None:
    """把响应矩阵按项目标准格式落盘 (h5 + json), 供 debug 与复用.

    单位: **λ/λ** (WFS 系数已由 µm 经 `um_to_waves` 换算, 与矫正时的 `w` 一致)。
    """
    from ao_shaping.optimizer.wf.zernike_response_matrix import (
        ZernikeResponseMatrixResult,
        save_zernike_response_matrix,
    )

    dc = {
        "device": device,          # 完整 SLM/WFS 参数 (波长/2π灰度/温度/曝光/pupil/MLA...)
        "slm_serial": (device.get("slm") or {}).get("serial_number"),
        "wfs_serial": (device.get("wfs") or {}).get("serial_number"),
        "wavelength_nm": (device.get("slm") or {}).get("wavelength_nm"),
        "shift_x": shift[0] if shift else None,
        "shift_y": shift[1] if shift else None,
        "zernike_radius_px": r_matrix,
        "amplitude_rad": amp_rad,
        "amplitude_waves": amp_rad / (2 * np.pi),
        "n_averages": n_avg,
        "slm_mode_ids_dll": modes,
        "zernike_ordering": ("DLL 顺序 m 枚举 (m=-n..+n), 非标准 Noll 1976: "
                             "[5](2,0)defocus [9](3,1)coma [13](4,0)spherical"),
        "matrix_layout": "matrix[wfs_coeff_index, slm_mode_index]; 行 0..65 ↔ DLL [1..66]",
        "units": "λ/λ (WFS 系数 µm 经 um_to_waves 换算; 与矫正 w 同单位)",
        "flat_reference": flat_ref,
        "pass_count_by_radius": extra.get("pass_count_by_radius"),
        "rejected": extra.get("rejected"),
        "method": "push-pull ±A, PatternHelper + display_phase (GUI 同链路)",
    }
    res = ZernikeResponseMatrixResult(
        matrix=matrix, variance_matrix=variance, deviation_response_matrix=None,
        subaperture_mask=None, n_max=4,
        magnitude=float(amp_rad / (2 * np.pi)),
        wavelength_nm=int(device.get("wavelength_nm") or 532),
        n_averages=int(n_avg), n_cycles=1,
        timestamp=datetime.now().isoformat(),
        excluded_piston=True, excluded_tip_tilt=False, device_config=dc)
    try:
        res.pinv_matrix = safe_pinv(matrix)
        res.lstsq_matrix = res.pinv_matrix
    except np.linalg.LinAlgError as e:
        logger.warning("逆矩阵计算失败: {}", e)
    save_zernike_response_matrix(res, dbg / "matrix.h5")
    (dbg / "matrix.json").write_text(json.dumps({
        "device_config": dc,
        "matrix": matrix.tolist(),
        "variance_matrix": variance.tolist(),
        "pinv_matrix": None if res.pinv_matrix is None else res.pinv_matrix.tolist(),
        "condition_number_effective": effective_cond(matrix),
        "column_norms": [float(np.linalg.norm(matrix[:, i]))
                         for i in range(matrix.shape[1])],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(f"   [debug] 矩阵已存: {dbg / 'matrix.h5'} "
               f"(有效列 cond={effective_cond(matrix):.2f})")


# ─────────────────────────── 阶段 3 ───────────────────────────

def closed_loop_correct(slm: Santec, wfs: ThorlabWFS, ph: PatternHelper,
                        matrix: np.ndarray, modes: list[int], r_used: float,
                        n_avg: int, extra_sleep: float, n_iter: int = 3,
                        gain: float = 0.8, leak: float = 0.0,
                        limit_amp: float = 3.0,
                        debug_dir: Path | None = None,
                        save_phase: bool = False) -> dict[str, Any]:
    """恢复内部参考 → 迭代 `c ← leak·c − gain·pinv(M)·w` → 逐轮测残差.

    ``debug_dir`` 非空时逐轮落盘: ``iter{n}/w_lam.npy`` (66 系数, λ),
    ``wavefront.npy`` (波前图), ``metrics.json`` (RMS/PV/coeffs/统计);
    ``save_phase=True`` 额外存 ``phase_gray.npy`` (实际上屏 uint16 灰度图案)。
    """
    click.echo("\n[3] 闭环反向矫正验证 (迭代 + 增益/泄漏)")
    wfs.set_ref_plane(custom=False)
    click.echo(f"   恢复内部参考: use_custom_ref={wfs.use_custom_ref}")

    pinv = safe_pinv(matrix)
    c_total = np.zeros(len(modes))
    history: list[dict] = []

    slm.display_data(flat_gray(), wait_time_s=0.5)
    time.sleep(0.4)
    z0 = measure_zernike(wfs, n_avg, WFS_ZERNIKE_ORDER)
    wf0 = measure_wavefront(wfs, n_avg)
    if z0 is None or wf0 is None:
        raise RuntimeError("内部参考下基线测量失败")
    w0 = um_to_waves(z0[1:])
    # DLL [1] = piston: 它是 WFS 参考平面的整体偏置 (不可也无需矫正), 而矩阵的
    # piston 行数值很大 (平移引入) → 若保留会通过 pinv 污染整个反解, 置零。
    w0[0] = 0.0
    click.echo(f"   矫正前: RMS={wf0[1]['rms']:.4f}λ, PV={wf0[1]['diff']:.4f}λ, "
               f"|w|={np.linalg.norm(w0):.4f}λ (piston 已置零)")
    history.append({"iter": 0, "rms": wf0[1]["rms"], "pv": wf0[1]["diff"],
                    "z_norm": float(np.linalg.norm(w0)), "w": w0})
    if debug_dir is not None:
        d0 = debug_dir / "iter0_before"
        d0.mkdir(parents=True, exist_ok=True)
        np.save(d0 / "w_lam.npy", w0)
        np.save(d0 / "wavefront.npy", wf0[0])
        (d0 / "metrics.json").write_text(json.dumps(
            {"iter": 0, "rms": wf0[1]["rms"], "pv": wf0[1]["diff"],
             "z_norm": float(np.linalg.norm(w0)), "w_stats": wf0[1],
             "slm_shift": [slm.shift_x, slm.shift_y], "phase_radius": r_used,
             "note": "内部参考下的矫正前像差 (piston 置零)"},
            indent=2, ensure_ascii=False), encoding="utf-8")

    best = {"rms": wf0[1]["rms"], "coeffs": np.zeros(len(modes))}
    for it in range(1, n_iter + 1):
        w = history[-1]["w"]
        c_total = leak * c_total - gain * (pinv @ w)
        c_total = np.clip(c_total, -limit_amp, limit_amp)
        coeffs = {nm_of(m): float(c_total[i]) for i, m in enumerate(modes)
                  if abs(c_total[i]) > 1e-4}
        if not coeffs:
            click.echo(f"   [iter {it}] 反解系数全为 0, 停止")
            break
        # ⚠️ 单位: c_total 是**波长 (λ)** (来自 pinv @ w, w 为 λ), 而
        # `make_phase`/`generate_zernike_polynomial` 的系数单位是**弧度** →
        # 必须 ×2π 才能把"λ 的相位"正确加载 (曾漏此换算, 实测响应仅预测的 28%,
        # 即被缩小 2π = 6.28×)。debug 落盘的 coeffs_lam.npy 仍是 λ 单位。
        coeffs_rad = {nm: v * 2.0 * np.pi for nm, v in coeffs.items()}
        # 加载到 SLM 的 Zernike 系数必须**正负都有** (像差有正负分量, 矫正也应有);
        # 若全同号 → 矩阵符号/标定有误, 在此显式暴露。
        n_pos = int((c_total > 1e-4).sum())
        n_neg = int((c_total < -1e-4).sum())
        click.echo(f"   [iter {it}] 加载 SLM 系数: 正 {n_pos} / 负 {n_neg} / 零 "
                   f"{len(modes) - n_pos - n_neg}  |  " +
                   ", ".join(f"[{modes[i]}]{c_total[i]:+.3f}"
                             for i in range(len(modes)) if abs(c_total[i]) > 1e-3))
        phase = make_phase(ph, coeffs_rad, r_used)
        show_phase(slm, phase, extra_sleep)
        time.sleep(0.3)
        z = measure_zernike(wfs, n_avg, WFS_ZERNIKE_ORDER)
        wf = measure_wavefront(wfs, n_avg)
        if z is None or wf is None:
            click.echo(f"   [iter {it}] 测量失败, 停止")
            break
        wi = um_to_waves(z[1:])
        rms = wf[1]["rms"]
        # 模型自检: 预测 ||w_prev + M·c|| 必须与实测 ||wi|| 一致。
        # 若不符 (比值偏离 1) → 矩阵单位/符号/覆盖有问题 (实测曾因矩阵用 µm 而矫正用 λ
        # 导致系数被放大 1.88×, 此自检可直接暴露)。
        w_prev = history[-2]["w"] if len(history) >= 2 else w0
        pred_norm = float(np.linalg.norm(w_prev + matrix @ c_total))
        meas_norm = float(np.linalg.norm(wi))
        ratio = pred_norm / meas_norm if meas_norm > 0 else float("nan")
        flag = "" if 0.7 <= ratio <= 1.4 else "  <== 模型/实测不符, 检查单位/符号!"
        click.echo(f"   [iter {it}] RMS={rms:.4f}λ, PV={wf[1]['diff']:.4f}λ, "
                   f"|w|={np.linalg.norm(wi):.4f}λ  "
                   f"(改善 {100 * (1 - rms / wf0[1]['rms']):.1f}%)")
        click.echo(f"   [iter {it}] 模型自检: 预测|w|={pred_norm:.4f}λ vs "
                   f"实测{meas_norm:.4f}λ (比 {ratio:.2f}){flag}")
        if flag:
            logger.warning("模型自检失败: 预测 |w|={:.4f} vs 实测 {:.4f} (比 {:.2f})",
                           pred_norm, meas_norm, ratio)
        history.append({"iter": it, "rms": rms, "pv": wf[1]["diff"],
                        "z_norm": float(np.linalg.norm(wi)),
                        "pred_z_norm": pred_norm, "model_ratio": ratio, "w": wi})
        if rms < best["rms"]:
            best = {"rms": rms, "coeffs": c_total.copy()}
        if debug_dir is not None:
            d = debug_dir / f"iter{it}"
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "w_lam.npy", wi)
            np.save(d / "wavefront.npy", wf[0])
            np.save(d / "coeffs_lam.npy", c_total)
            (d / "metrics.json").write_text(json.dumps(
                {"iter": it, "rms": rms, "pv": wf[1]["diff"],
                 "z_norm": float(np.linalg.norm(wi)),
                 "pred_z_norm": pred_norm, "model_ratio": ratio,
                 "coeffs": {str(nm_of(m)): float(c_total[i])
                            for i, m in enumerate(modes)},
                 "w_stats": wf[1], "slm_shift": [slm.shift_x, slm.shift_y],
                 "phase_radius": r_used, "gain": gain, "leak": leak,
                 "n_modes_valid": int((np.abs(c_total) > 1e-4).sum())},
                indent=2, ensure_ascii=False), encoding="utf-8")
            if save_phase:
                np.save(d / "phase_gray.npy", slm.create_phase_from_array(phase))

    final = history[-1]
    click.echo(f"   最佳: RMS {wf0[1]['rms']:.4f} → {best['rms']:.4f}λ "
               f"({100 * (1 - best['rms'] / wf0[1]['rms']):.1f}%)")
    return {"before_rms": wf0[1]["rms"], "after_rms": best["rms"],
            "before_pv": wf0[1]["diff"], "after_pv": final["pv"],
            "before_z_norm": float(np.linalg.norm(w0)),
            "after_z_norm": final["z_norm"],
            "coeffs": {str(nm_of(m)): float(best["coeffs"][i])
                       for i, m in enumerate(modes)},
            "w_before": w0.tolist(), "w_after": final["w"].tolist(),
            "history": [{k: v for k, v in h.items() if k != "w"} for h in history],
            "n_iter": n_iter, "gain": gain, "leak": leak}


# ─────────────────────────── CLI ───────────────────────────

@click.command()
@click.option("--stage", type=click.Choice(["all", "auto", "matrix", "closed"]),
              default="all", show_default=True, help="执行阶段")
@click.option("--slm-number", type=int, default=1, show_default=True)
@click.option("--slm-wavelength", type=int, default=532, show_default=True)
@click.option("--wfs-exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS, show_default=True)
@click.option("--wfs-order", type=int, default=WFS_ZERNIKE_ORDER, show_default=True)
@click.option("--n-max", type=int, default=4, show_default=True, help="SLM 模式最大阶数")
@click.option("--n-avg", type=int, default=3, show_default=True, help="WFS 多帧平均")
@click.option("--settle-extra-s", type=float, default=SETTLE_REDUNDANCY_S,
              show_default=True, help="像素翻转估算之外的冗余等待 (s)")
@click.option("--radius-factor", default="1.5,2.0", show_default=True,
              help="扫描半径 = 光束半径 × 该列表 (默认避开 1.0×: R≈光束半径时大振幅击穿拟合)")
@click.option("--radii", default=None, help="直接指定扫描半径 px (逗号分隔, 覆盖 --radius-factor)")
@click.option("--amps", default="2,5,10", show_default=True, help="幅度 rad 列表 (各测 ±)")
@click.option("--outlier-factor", type=float, default=3.0, show_default=True,
              help="逐点异常剔除: 同组 |resp| 偏离中位数超过该倍数则剔除")
@click.option("--coverage-tol", type=int, default=1, show_default=True,
              help="统一半径选择: 覆盖度容差内取最小 R (实测 R=300 优于 R=400 条件数 5×)")
@click.option("--radius-scan", default="120,200,300,450,600", show_default=True,
              help="阶段1 半径诊断扫描列表 px")
@click.option("--n-iter", type=int, default=3, show_default=True, help="闭环迭代轮数")
@click.option("--gain", type=float, default=0.8, show_default=True, help="闭环增益")
@click.option("--leak", type=float, default=0.0, show_default=True, help="闭环泄漏因子")
@click.option("--quick", is_flag=True, default=False, help="快速模式 (单尺寸/单幅度)")
@click.option("--save-phase", is_flag=True, default=False,
              help="debug: 额外保存每轮实际上屏 uint16 灰度相位 (npy, 每张 ~4.6MB)")
@click.option("--export-correction/--no-export-correction", "export_correction",
              default=True, show_default=True,
              help="导出可复原的矫正相位 CSV (驱动 Santec.save_phase_to_csv; "
                   "文件名含 序列号/波长/shift/半径/时间 + sidecar JSON 元数据)")
@click.option("--export-dir", default="data/slm_corrections", show_default=True,
              help="矫正相位导出目录")
@click.option("-o", "--output-dir", default="data/zernike_correction", show_default=True)
def main(stage, slm_number, slm_wavelength, wfs_exposure_ms, wfs_order, n_max,
         n_avg, settle_extra_s, radius_factor, radii, amps, radius_scan,
         n_iter, gain, leak, quick, outlier_factor, coverage_tol,
         save_phase, export_correction, export_dir, output_dir) -> int:
    """SLM Zernike 模式法波前矫正 — 三阶段流程."""
    if wfs_exposure_ms > MAX_EXPOSURE_MS:
        raise click.BadParameter(f"WFS 曝光 {wfs_exposure_ms}ms > {MAX_EXPOSURE_MS}ms")
    if not 2 <= wfs_order <= 10:
        raise click.BadParameter("--wfs-order 必须在 2..10")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    incr = out / f"raw_scan_{ts}.json"
    dbg = out / f"debug_{ts}"          # 全过程 debug 数据
    dbg.mkdir(parents=True, exist_ok=True)

    modes = [i for i in range(2, (n_max + 1) * (n_max + 2) // 2 + 1)]
    r_scan = [float(v) for v in radius_scan.split(",")]
    amp_list = [float(v) for v in amps.split(",")]
    fac_list = [float(v) for v in radius_factor.split(",")]
    if quick:
        amp_list = [amp_list[0]]

    click.echo("=" * 72)
    click.echo("[SLM Zernike 模式法波前矫正] 三阶段流程 (已按实测优化)")
    click.echo("=" * 72)

    slm = Santec(slm_number=slm_number, wavelength=slm_wavelength, video_mode=0)
    wfs = ThorlabWFS(exposure_time=wfs_exposure_ms, use_custom_ref=False)
    ph = PatternHelper(resolution=(PANEL_W, PANEL_H))
    report: dict = {"stage": stage, "timestamp": ts}

    try:
        slm.open()
        wfs.open()
        exp = float(wfs.exposure_time)
        assert exp <= MAX_EXPOSURE_MS
        wl, max_gray = slm.get_wavelength_info()
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        cx, cy, dx, dy = wfs.pupil = wfs.optimize_pupil()
        click.echo(f"[OK] SLM #{slm._serial_number} {wl}nm 2π={max_gray}; "
                   f"WFS {wfs.serial_num} exp={exp:.3f}ms "
                   f"pupil=({cx:.3f},{cy:.3f})mm d=({dx:.3f},{dy:.3f})mm")
        report["device"] = collect_device_info(slm, wfs, slm_number)
        _d = report["device"]
        click.echo(f"[OK] 设备参数: SLM #{_d['slm'].get('serial_number')} "
                   f"{_d['slm'].get('wavelength_nm')}nm 2π={_d['slm'].get('two_pi_gray')}gray "
                   f"温度={_d['slm'].get('temperature_c')}°C "
                   f"版本={_d['slm'].get('version')} "
                   f"矫正={_d['slm'].get('correction_enabled')}")
        click.echo(f"[OK]            WFS #{_d['wfs'].get('serial_number')} "
                   f"曝光={_d['wfs'].get('exposure_time_ms')}ms "
                   f"pupil={_d['wfs'].get('pupil_center_mm')}mm "
                   f"d={_d['wfs'].get('pupil_diameter_mm')}mm "
                   f"MLA={_d['wfs'].get('mla_name')} "
                   f"{_d['wfs'].get('num_spots_x')}x{_d['wfs'].get('num_spots_y')}")

        r_beam = 250.0
        if stage in ("all", "auto"):
            r_beam, r_diag = diagnose_beam_radius(slm, wfs, ph, r_scan, 20.0,
                                                  n_avg, settle_extra_s)
            report["radius_diagnostic"] = r_diag
            sx, sy, sh_scan = calibrate_center_shift(slm, wfs, ph, r_beam, 20.0,
                                                     n_avg=n_avg,
                                                     extra_sleep=settle_extra_s)
            report["shift_scan"] = sh_scan
            slm.set_shift(sx, sy)
            slm.save_config()
            report["shift"] = [sx, sy]
            report["flat_reference"] = setup_flat_reference(slm, wfs, max(n_avg, 3),
                                                            settle_extra_s)

        matrix = None
        r_used = r_beam
        if stage in ("all", "matrix"):
            if radii:
                r_list = [float(v) for v in radii.split(",")]
            elif quick:
                r_list = [r_beam * fac_list[0]]
            else:
                r_list = [r_beam * f for f in fac_list]
            r_list = [float(np.clip(r, 120, 600)) for r in r_list]
            matrix, metrics, extra = scan_response_matrix(
                slm, wfs, ph, modes, r_list, amp_list, n_avg, settle_extra_s,
                incr, wfs_order, outlier_factor, coverage_tol)
            variance = np.array(extra.pop("variance", []), dtype=float)
            pts = extra.pop("points", [])
            report["metrics"] = metrics
            report.update(extra)
            # 统一半径: 矩阵与矫正相位必须同半径 (否则 Zernike 归一化不匹配)
            r_used = float(extra["matrix_radius"])
            click.echo(f"   主矩阵半径 = {r_used:.0f}px (统一, 与矫正相位一致)")
            # ---- debug 落盘: 矩阵 (标准 h5 + json) + 逐点诊断 ----
            amp_mid = float(amp_list[len(amp_list) // 2]) if amp_list else 0.0
            save_matrix_debug(dbg, matrix, variance, modes, r_used,
                              report.get("device", {}), report.get("shift"),
                              report.get("flat_reference"), amp_mid, n_avg, extra)
            (dbg / "scan_points.json").write_text(
                json.dumps(pts, ensure_ascii=False), encoding="utf-8")
            click.echo(f"   [debug] 逐点诊断: {dbg / 'scan_points.json'} "
                       f"({len(pts)} 点, 含 ±对称性/SNR)")

        if stage in ("all", "closed"):
            if matrix is None:
                click.echo("[WARN] 无矩阵, 跳过闭环矫正")
            else:
                cl = closed_loop_correct(
                    slm, wfs, ph, matrix, modes, r_used, n_avg, settle_extra_s,
                    n_iter=n_iter, gain=gain, leak=leak,
                    debug_dir=dbg / "closed_loop", save_phase=save_phase)
                report["closed_loop"] = cl

                # ---- 导出可复原的矫正相位 (驱动自带 Santec.save_phase_to_csv) ----
                if export_correction:
                    # cl["coeffs"] 单位 λ → ×2π 得弧度 (make_phase 收弧度)
                    coeffs_lam = {nm_of(m): float(cl["coeffs"].get(str(nm_of(m)), 0.0))
                                  for m in modes}
                    coeffs_rad = {nm: v * 2.0 * np.pi for nm, v in coeffs_lam.items()
                                  if abs(v) > 1e-4}
                    if not coeffs_rad:
                        click.echo("[WARN] 矫正系数全为 0, 跳过导出")
                    else:
                        phase_corr = make_phase(ph, coeffs_rad, r_used)
                        dev = report.get("device", {})
                        meta = {
                            "purpose": "Zernike 模式法波前矫正相位 (SLM)",
                            "device": dev,
                            "slm_serial": (dev.get("slm") or {}).get("serial_number"),
                            "wavelength_nm": (dev.get("slm") or {}).get("wavelength_nm"),
                            "shift_x": (dev.get("slm") or {}).get("shift_x"),
                            "shift_y": (dev.get("slm") or {}).get("shift_y"),
                            "zernike_radius_px": r_used,
                            "coefficients_lam": coeffs_lam,
                            "coefficients_rad": coeffs_rad,
                            "zernike_ordering": ("DLL 顺序 m 枚举 (m=-n..+n), 非标准 Noll: "
                                                 "[5](2,0)defocus [9](3,1)coma "
                                                 "[13](4,0)spherical"),
                            "matrix_h5": str(dbg / "matrix.h5"),
                            "closed_loop": {k: v for k, v in cl.items()
                                            if k not in ("coeffs", "w_before", "w_after")},
                            "reproduce": ("读取本 CSV 数据区为**弧度** → "
                                          "slm.display_phase(phase_rad); "
                                          "勿走 load_gray_from_csv/csv_to_phase (只接受灰度)"),
                        }
                        csv_p, json_p = export_correction_csv(
                            slm, phase_corr, meta, out_dir=export_dir, prefix="slm_corr")
                        click.echo(f"[OK] 矫正相位已导出: {csv_p}")
                        click.echo(f"[OK]   元数据 sidecar: {json_p}")
                        report["correction_export"] = {"csv": str(csv_p),
                                                       "json": str(json_p)}

        slm.display_data(flat_gray(), wait_time_s=0.5)

    except Exception as e:
        logger.exception("流程失败")
        click.echo(f"[FAIL] {type(e).__name__}: {e}")
        return 1
    finally:
        for dev in (wfs, slm):
            try:
                dev.close()
            except Exception:
                pass

    rp = out / f"report_{ts}.json"
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # ---- debug manifest: 列出全过程产物 ----
    files = sorted(str(p.relative_to(out)) for p in dbg.rglob("*") if p.is_file())
    (dbg / "manifest.json").write_text(json.dumps({
        "timestamp": ts, "stage": stage,
        "report": rp.name, "raw_scan": incr.name, "debug_dir": dbg.name,
        "files": files,
        "notes": [
            "matrix.h5 / matrix.json: 响应矩阵 (λ/λ, 行 0..65 ↔ DLL [1..66]) "
            "+ pinv + 列范数 + cond",
            "scan_points.json: 逐点 ±推拉诊断 (diff/sym 向量, SNR vs 基线, valid_ratio)",
            "closed_loop/iter0_before/: 内部参考下的矫正前像差 (w_lam.npy, wavefront.npy)",
            "closed_loop/iterN/: w_lam.npy, wavefront.npy, coeffs_lam.npy, metrics.json "
            "(含模型自检 model_ratio)",
            "--save-phase 时额外含 phase_gray.npy (实际上屏 uint16 灰度)",
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(f"\n[OK] 报告: {rp}")
    click.echo(f"[OK] 原始扫描: {incr}")
    click.echo(f"[OK] debug 目录: {dbg} ({len(files)} 个文件)")
    click.echo("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
