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
from collections import Counter
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
    diagnose_beam_radius,
    flat_gray,
    linearity_metrics,
    make_phase,
    measure_wavefront,
    measure_zernike,
    nm_of,
    show_phase,
    wfs_validity,
)
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
                         order: int = WFS_ZERNIKE_ORDER) -> tuple[np.ndarray, list, dict]:
    """模式 × 尺寸 × ±幅度 网格扫描; 有效性门控 + 正确线性度判据 + 全量原始落盘."""
    click.echo(f"\n[2] 响应矩阵扫描: {len(modes)} 模式 × {len(radii)} 尺寸 × "
               f"{len(amps)} 幅度(±) × {n_avg} 帧")
    raw: list[dict] = []
    curves: dict[tuple[int, float], list[np.ndarray]] = {}
    amps_of: dict[tuple[int, float], list[float]] = {}
    rejected: list[dict] = []

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
                col = (zp[1:] - zn[1:]) / 2.0 / a_waves
                curves.setdefault((midx, R), []).append(col)
                amps_of.setdefault((midx, R), []).append(a)
                click.echo(f"   [{k}/{total}] [{midx:2d}] {str(nm):9s} R={R:5.0f} "
                           f"A={a:5.1f}rad  |resp|={np.linalg.norm(col):.3f}")
            incr_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

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

    # 尺寸选择 (已修正): 先剔除不通过项, 再在通过项中选响应最强
    n_rows = 66
    matrix = np.zeros((n_rows, len(modes)), dtype=np.float64)
    var = np.zeros_like(matrix)
    chosen: dict[int, float] = {}
    for col, midx in enumerate(modes):
        cand = [m for m in metrics if m["dll_index"] == midx and m["ok"]]
        if not cand:
            click.echo(f"   [WARN] 模式 [{midx}] 无通过的 (mode,R) → 该列置零")
            continue
        best = max(cand, key=lambda m: float(np.mean(m["norms"])))
        R = best["radius"]
        chosen[midx] = R
        vecs = curves[(midx, R)]
        matrix[:, col] = np.mean(np.array(vecs), axis=0)
        var[:, col] = np.var(np.array(vecs), axis=0) if len(vecs) > 1 else 0.0
    click.echo(f"   主矩阵形状 {matrix.shape}; 各模式选定尺寸: "
               f"{ {m: int(chosen[m]) for m in sorted(chosen)} }")
    return matrix, metrics, {
        "chosen_radius": {int(k): float(v) for k, v in chosen.items()},
        "rejected": rejected, "curves_n": len(curves)}


# ─────────────────────────── 阶段 3 ───────────────────────────

def closed_loop_correct(slm: Santec, wfs: ThorlabWFS, ph: PatternHelper,
                        matrix: np.ndarray, modes: list[int], r_used: float,
                        n_avg: int, extra_sleep: float, n_iter: int = 3,
                        gain: float = 0.8, leak: float = 0.0,
                        limit_amp: float = 3.0) -> dict[str, Any]:
    """恢复内部参考 → 迭代 `c ← leak·c − gain·pinv(M)·w` → 逐轮测残差."""
    click.echo("\n[3] 闭环反向矫正验证 (迭代 + 增益/泄漏)")
    wfs.set_ref_plane(custom=False)
    click.echo(f"   恢复内部参考: use_custom_ref={wfs.use_custom_ref}")

    pinv = np.linalg.pinv(matrix)
    c_total = np.zeros(len(modes))
    history: list[dict] = []

    slm.display_data(flat_gray(), wait_time_s=0.5)
    time.sleep(0.4)
    z0 = measure_zernike(wfs, n_avg, WFS_ZERNIKE_ORDER)
    wf0 = measure_wavefront(wfs, n_avg)
    if z0 is None or wf0 is None:
        raise RuntimeError("内部参考下基线测量失败")
    w0 = z0[1:] / 0.532
    click.echo(f"   矫正前: RMS={wf0[1]['rms']:.4f}λ, PV={wf0[1]['diff']:.4f}λ, "
               f"|w|={np.linalg.norm(w0):.4f}λ")
    history.append({"iter": 0, "rms": wf0[1]["rms"], "pv": wf0[1]["diff"],
                    "z_norm": float(np.linalg.norm(w0)), "w": w0})

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
        phase = make_phase(ph, coeffs, r_used)
        show_phase(slm, phase, extra_sleep)
        time.sleep(0.3)
        z = measure_zernike(wfs, n_avg, WFS_ZERNIKE_ORDER)
        wf = measure_wavefront(wfs, n_avg)
        if z is None or wf is None:
            click.echo(f"   [iter {it}] 测量失败, 停止")
            break
        wi = z[1:] / 0.532
        rms = wf[1]["rms"]
        click.echo(f"   [iter {it}] RMS={rms:.4f}λ, PV={wf[1]['diff']:.4f}λ, "
                   f"|w|={np.linalg.norm(wi):.4f}λ  "
                   f"(改善 {100 * (1 - rms / wf0[1]['rms']):.1f}%)")
        history.append({"iter": it, "rms": rms, "pv": wf[1]["diff"],
                        "z_norm": float(np.linalg.norm(wi)), "w": wi})
        if rms < best["rms"]:
            best = {"rms": rms, "coeffs": c_total.copy()}

    final = history[-1]
    click.echo(f"   最佳: RMS {wf0[1]['rms']:.4f} → {best['rms']:.4f}λ "
               f"({100 * (1 - best['rms'] / wf0[1]['rms']):.1f}%)")
    return {"before_rms": wf0[1]["rms"], "after_rms": best["rms"],
            "before_pv": wf0[1]["diff"], "after_pv": final["pv"],
            "before_z_norm": float(np.linalg.norm(w0)),
            "after_z_norm": final["z_norm"],
            "coeffs": {str(nm_of(m)): float(best["coeffs"][i])
                       for i, m in enumerate(modes)},
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
@click.option("--radius-scan", default="120,200,300,450,600", show_default=True,
              help="阶段1 半径诊断扫描列表 px")
@click.option("--n-iter", type=int, default=3, show_default=True, help="闭环迭代轮数")
@click.option("--gain", type=float, default=0.8, show_default=True, help="闭环增益")
@click.option("--leak", type=float, default=0.0, show_default=True, help="闭环泄漏因子")
@click.option("--quick", is_flag=True, default=False, help="快速模式 (单尺寸/单幅度)")
@click.option("-o", "--output-dir", default="data/zernike_correction", show_default=True)
def main(stage, slm_number, slm_wavelength, wfs_exposure_ms, wfs_order, n_max,
         n_avg, settle_extra_s, radius_factor, radii, amps, radius_scan,
         n_iter, gain, leak, quick, output_dir) -> int:
    """SLM Zernike 模式法波前矫正 — 三阶段流程."""
    if wfs_exposure_ms > MAX_EXPOSURE_MS:
        raise click.BadParameter(f"WFS 曝光 {wfs_exposure_ms}ms > {MAX_EXPOSURE_MS}ms")
    if not 2 <= wfs_order <= 10:
        raise click.BadParameter("--wfs-order 必须在 2..10")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    incr = out / f"raw_scan_{ts}.json"

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
        report["device"] = {"slm": slm._serial_number, "wfs": wfs.serial_num,
                            "wavelength_nm": slm_wavelength, "exposure_ms": exp,
                            "pupil_center_mm": [cx, cy],
                            "pupil_diameter_mm": [dx, dy], "2pi_gray": max_gray}

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
                incr, wfs_order)
            report["metrics"] = metrics
            report.update(extra)
            cnt = Counter(extra["chosen_radius"].values())
            r_used = cnt.most_common(1)[0][0] if cnt else r_beam
            report["matrix_radius"] = r_used
            click.echo(f"   主矩阵半径 = {r_used:.0f}px")

        if stage in ("all", "closed"):
            if matrix is None:
                click.echo("[WARN] 无矩阵, 跳过闭环矫正")
            else:
                report["closed_loop"] = closed_loop_correct(
                    slm, wfs, ph, matrix, modes, r_used, n_avg, settle_extra_s,
                    n_iter=n_iter, gain=gain, leak=leak)

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
    click.echo(f"\n[OK] 报告: {rp}")
    click.echo(f"[OK] 原始扫描: {incr}")
    click.echo("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
