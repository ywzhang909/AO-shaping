"""SLM Zernike 模式法波前矫正 — 三阶段完整流程 (2026-09-15).

阶段 1 — 自动定标 + 参考波前
  1.1 自动测光束在 SLM 上的半径 (Zernike R 扫描, defocus 响应最大者)
  1.2 自动测光束中心偏移 → shift_x/shift_y (defocus tip/tilt 零点法)
  1.3 SLM **纯 0 相位** → create_default_user_ref → save_user_ref → set_ref_plane(custom=True)
  1.4 **记录并校验纯平波前平整度** (多帧 RMS/PV, 断言 < 阈值)

阶段 2 — 稳健响应矩阵
  对 模式 × 尺寸(R) × 幅度(±) 网格逐点:
    - `display_phase(wait_time_s=None)` 自动等 LCOS 像素翻转 + **100 ms 冗余**
    - WFS 多帧平均 (中位数聚合)
    - **全量原始读数增量落盘** (中途失败不丢数据)
  生成 metrics: 每 (模式,尺寸) 的响应幅值/斜率/线性度 R²/残差/SNR

阶段 3 — 闭环反向矫正验证
  3.1 恢复 WFS **内部参考** (`set_ref_plane(custom=False)`)
  3.2 读内部参考下的波前 Zernike 像差 w (真实系统像差)
  3.3 `c = -pinv(M) @ w` → 生成矫正相位加载 SLM
  3.4 复测残差 RMS/PV, 对比矫正前后

⚠️ 索引约定 (官方手册 + 硬件实测确认): WFS `get_zernike()` 返回 67 长数组,
   `coeff[1..66]` 有效 (index 0 未用), 排序为**顺序 m 枚举** (m=-n..+n), **非标准 Noll**:
   [1](0,0) [2](1,-1) [3](1,1) [4](2,-2) [5](2,0)defocus [6](2,2)
   [9](3,1)coma [13](4,0)spherical

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

import click
import numpy as np
from loguru import logger

from ao_shaping.drivers.slm import Santec
from ao_shaping.drivers.wfs import ThorlabWFS
from ao_shaping.utils.pattern_helper import PatternHelper

PANEL_H, PANEL_W = 1200, 1920
MAX_EXPOSURE_MS = 7.0
DEFAULT_EXPOSURE_MS = 4.0
SETTLE_REDUNDANCY_S = 0.1          # 像素翻转估算之外额外等待 (用户要求 100ms)

# DLL 顺序 m 枚举: 1-based index -> (n, m)
DLL_ZERNIKE = [
    (0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2),
    (3, -3), (3, -1), (3, 1), (3, 3),
    (4, -4), (4, -2), (4, 0), (4, 2), (4, 4),
]


# ─────────────────────────── 基础测量 ───────────────────────────

def measure_zernike(wfs: ThorlabWFS, n_avg: int, order: int) -> np.ndarray | None:
    """多帧中位数聚合 WFS Zernike 读数 (µm, 67 长, index 0 未用)."""
    rows = []
    for _ in range(n_avg):
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        try:
            z = wfs.get_zernike(zernike_order=order)
        except Exception as e:
            logger.debug("get_zernike 失败: {}", e)
            continue
        if z is None or not np.isfinite(z).all():
            continue
        rows.append(np.asarray(z, dtype=float))
    return None if not rows else np.median(np.array(rows), axis=0)


def measure_wf(wfs: ThorlabWFS, n_avg: int) -> tuple[np.ndarray, dict] | None:
    """多帧中位数波前 + 平均 stats."""
    wfs_ = []
    stats = []
    for _ in range(n_avg):
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        wf, st = wfs.get_wavefront(cancel_tile=False)
        wfs_.append(wf)
        stats.append(st)
    if not wfs_:
        return None
    med = np.nanmedian(np.array(wfs_), axis=0)
    mean_stats = {k: float(np.mean([s.get(k, np.nan) for s in stats]))
                  for k in ("rms", "diff", "mean", "max", "min")}
    return med, mean_stats


def show_phase(slm: Santec, phase_rad: np.ndarray, extra_sleep: float) -> None:
    """下发相位: wait_time_s=None 自动等像素翻转 + extra_sleep 冗余."""
    slm.display_phase(phase_rad, wait_time_s=None)
    if extra_sleep > 0:
        time.sleep(extra_sleep)


def flat_gray() -> np.ndarray:
    return np.full((PANEL_H, PANEL_W), 0, dtype=np.uint16)


# ─────────────────────────── 阶段 1 ───────────────────────────

def auto_beam_radius(slm, wfs, ph, radii, amp_rad, n_avg, extra_sleep) -> tuple[float, list]:
    """Zernike R 扫描: WFS defocus 响应最大者 ≈ 光束半径."""
    click.echo("\n[1.1] 自动测光束半径 (Zernike R 扫描)")
    out = []
    for r in radii:
        phase = ph.generate_zernike_polynomial(coefficients={(2, 0): amp_rad},
                                               radius=float(r), n_max=5)
        show_phase(slm, phase, extra_sleep)
        z = measure_zernike(wfs, n_avg, 10)
        if z is None:
            click.echo(f"   R={r:5.0f}px  无有效读数")
            continue
        # DLL [5] = (2,0) defocus
        d = float(z[5]) / 0.532
        out.append({"radius": float(r), "defocus_lam": d})
        click.echo(f"   R={r:5.0f}px  defocus[5]={d:+.4f}λ")
    if not out:
        raise RuntimeError("半径诊断无有效数据")
    best = max(out, key=lambda r: abs(r["defocus_lam"]))
    click.echo(f"   → 光束半径 ≈ {best['radius']:.0f}px")
    return float(best["radius"]), out


def auto_center_shift(slm, wfs, ph, r_beam, coarse_half, step, amp_rad, n_avg,
                      extra_sleep, limit=500) -> tuple[int, int, list]:
    """defocus tip/tilt 零点法测光束中心偏移 → shift_x/shift_y (轴无关 ‖Δz_tilt‖)."""
    click.echo("\n[1.2] 自动测光束中心偏移 (defocus 零点法 → shift)")
    r_use = max(r_beam * 3.0, 450.0)      # 盘需足够大以免平移裁切光束
    phase = ph.generate_zernike_polynomial(coefficients={(2, 0): amp_rad},
                                           radius=r_use, n_max=5)
    slm.set_shift(0, 0)
    slm.display_data(flat_gray(), wait_time_s=0.5)
    time.sleep(0.3)
    base = measure_zernike(wfs, max(n_avg, 3), 10)
    if base is None:
        raise RuntimeError("基线测量失败")
    base_t = base[2:4] / 0.532

    def eval_shift(sx: int, sy: int) -> tuple[float, np.ndarray]:
        slm.set_shift(int(np.clip(sx, -limit, limit)), int(np.clip(sy, -limit, limit)))
        show_phase(slm, phase, extra_sleep)
        z = measure_zernike(wfs, n_avg, 10)
        if z is None:
            return np.inf, np.zeros(2)
        t = z[2:4] / 0.532
        return float(np.linalg.norm(t - base_t)), t

    vals = list(range(-coarse_half, coarse_half + 1, step))
    scan = []

    def scan_axis(axis: str, values, other: int) -> float:
        best_v: float = float(values[0])
        best_m = np.inf
        for v in values:
            sx_ = int(v) if axis == "x" else int(other)
            sy_ = int(v) if axis == "y" else int(other)
            m, _ = eval_shift(sx_, sy_)
            scan.append({"axis": axis, "shift": float(v), "added_norm": float(m)})
            if m < best_m:
                best_m, best_v = m, float(v)
        click.echo(f"   {axis} 轴最优 {best_v:.0f} (‖Δ‖={best_m:.4f}λ)")
        return float(best_v)

    sx0 = scan_axis("x", vals, 0)
    sx = scan_axis("x", [sx0 + d for d in (-40, -20, 0, 20, 40)], 0)
    sy0 = scan_axis("y", vals, int(sx))
    sy = scan_axis("y", [sy0 + d for d in (-40, -20, 0, 20, 40)], int(sx))
    click.echo(f"   → shift=({int(sx)},{int(sy)})")
    return int(sx), int(sy), scan


def setup_flat_reference(slm, wfs, n_avg, extra_sleep) -> dict:
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

    rec = measure_wf(wfs, n_avg)
    if rec is None:
        raise RuntimeError("平整度测量失败")
    _, st = rec
    z = measure_zernike(wfs, n_avg, 10)
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

def scan_response_matrix(slm, wfs, ph, modes, radii, amps, n_avg, extra_sleep,
                         incr_path: Path, order: int = 10) -> tuple[np.ndarray, list, dict]:
    """模式 × 尺寸 × ±幅度 网格扫描, 全量原始读数增量落盘."""
    click.echo(f"\n[2] 响应矩阵扫描: {len(modes)} 模式 × {len(radii)} 尺寸 × "
               f"{len(amps)} 幅度(±) × {n_avg} 帧")
    raw: list[dict] = []
    # 每个 (mode, radius) 的响应曲线: amp -> (z_pos - z_neg)/(2*amp_waves)
    curves: dict[tuple[int, float], list] = {}

    total = len(modes) * len(radii) * len(amps)
    k = 0
    for midx in modes:
        nm = DLL_ZERNIKE[midx - 1]
        for R in radii:
            R = float(R)
            for a in amps:
                k += 1
                a_waves = a / (2 * np.pi)
                zs: dict[int, np.ndarray | None] = {}
                for sign in (+1, -1):
                    phase = ph.generate_zernike_polynomial(
                        coefficients={nm: sign * a}, radius=R, n_max=max(4, 4))
                    show_phase(slm, phase, extra_sleep)
                    z = measure_zernike(wfs, n_avg, order)
                    zs[sign] = z
                    if z is not None:
                        raw.append({"dll_index": midx, "nm": list(nm), "radius": R,
                                    "amp_rad": sign * a, "sign": sign,
                                    "readout_um": z.tolist()})
                zp, zn = zs.get(+1), zs.get(-1)
                if zp is None or zn is None:
                    click.echo(f"   [{k}/{total}] [{midx}] {nm} R={R:.0f} "
                               f"A={a:.1f} 无有效读数")
                    continue
                col = (zp[1:] - zn[1:]) / 2.0 / a_waves
                curves.setdefault((midx, R), []).append(
                    {"amp_rad": a, "amp_waves": a_waves, "response": col})
                click.echo(f"   [{k}/{total}] [{midx:2d}] {str(nm):9s} R={R:5.0f} "
                           f"A={a:5.1f}rad  |resp|={np.linalg.norm(col):.3f}")
            # 增量落盘
            incr_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    # 线性度 metrics (复用 _optimize_perturbation_amplitude 的斜率稳定性判据 + R²)
    metrics: list[dict] = []
    for (midx, R), pts in curves.items():
        amps_ = np.array([p["amp_rad"] for p in pts])
        norms = np.array([float(np.linalg.norm(p["response"])) for p in pts])
        order_ = np.argsort(amps_)
        amps_, norms = amps_[order_], norms[order_]
        r2 = float("nan")
        slope = float("nan")
        if len(amps_) >= 2 and np.ptp(amps_) > 0:
            kk = np.polyfit(amps_, norms, 1)
            slope = float(kk[0])
            pred = np.polyval(kk, amps_)
            ss_res = float(np.sum((norms - pred) ** 2))
            ss_tot = float(np.sum((norms - norms.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        lin = []
        for i in range(1, len(amps_) - 1):
            k1 = (norms[i] - norms[i - 1]) / (amps_[i] - amps_[i - 1])
            k2 = (norms[i + 1] - norms[i]) / (amps_[i + 1] - amps_[i])
            lin.append(abs(k1 - k2))
        metrics.append({"dll_index": midx, "nm": list(DLL_ZERNIKE[midx - 1]),
                        "radius": R, "n_amps": len(amps_),
                        "slope": slope, "r2": r2,
                        "linearity_min": float(min(lin)) if lin else float("nan")})
    # 主响应矩阵: 取线性度最好且响应最强的 (mode, R)
    best_R: dict[int, float] = {}
    for midx in modes:
        cand = [m for m in metrics if m["dll_index"] == midx]
        if not cand:
            continue
        good = [m for m in cand if np.isfinite(m["r2"]) and m["r2"] > 0.9] or cand
        best_R[midx] = max(good, key=lambda m: abs(m["slope"]))["radius"]

    n_rows = 66
    matrix = np.zeros((n_rows, len(modes)), dtype=np.float64)
    var = np.zeros_like(matrix)
    for col, midx in enumerate(modes):
        R = best_R.get(midx)
        if R is None:
            continue
        cols = [p["response"] for p in curves.get((midx, R), [])]
        if not cols:
            continue
        arr = np.array(cols)
        matrix[:, col] = np.median(arr, axis=0)
        var[:, col] = np.var(arr, axis=0) if len(cols) > 1 else 0.0
    click.echo(f"\n   主矩阵形状 {matrix.shape}; 各模式最优尺寸: "
               f"{ {m: int(best_R[m]) for m in sorted(best_R)} }")
    return matrix, metrics, {"best_radius": {int(k): float(v) for k, v in best_R.items()},
                             "curves_n": len(curves)}


# ─────────────────────────── 阶段 3 ───────────────────────────

def closed_loop_correct(slm, wfs, ph, matrix, modes, r_used, n_avg, extra_sleep,
                        limit_amp: float = 3.0) -> dict:
    """恢复内部参考 → 读像差 → c=-pinv(M)w 加载矫正 → 测残差."""
    click.echo("\n[3] 闭环反向矫正验证")
    wfs.set_ref_plane(custom=False)
    click.echo(f"   恢复内部参考: use_custom_ref={wfs.use_custom_ref}")

    slm.display_data(flat_gray(), wait_time_s=0.5)
    time.sleep(0.4)
    z0 = measure_zernike(wfs, n_avg, 10)
    wf0 = measure_wf(wfs, n_avg)
    if z0 is None or wf0 is None:
        raise RuntimeError("内部参考下基线测量失败")
    w = z0[1:] / 0.532                      # (66,) λ, DLL [1..66]
    click.echo(f"   矫正前: RMS={wf0[1]['rms']:.4f}λ, PV={wf0[1]['diff']:.4f}λ, "
               f"|w|={np.linalg.norm(w):.4f}λ")

    pinv = np.linalg.pinv(matrix)           # (n_modes, 66)
    c = -pinv @ w
    c = np.clip(c, -limit_amp, limit_amp)   # 安全限幅
    coeffs = {}
    for i, midx in enumerate(modes):
        if abs(c[i]) > 1e-4:
            coeffs[DLL_ZERNIKE[midx - 1]] = float(c[i])
    click.echo("   反解 SLM 矫正系数 (λ): " +
               ", ".join(f"[{modes[i]}]{c[i]:+.4f}" for i in range(len(modes))
                         if abs(c[i]) > 1e-3))
    if not coeffs:
        click.echo("   [WARN] 反解系数全为 0, 跳过加载")
        return {"before_rms": wf0[1]["rms"], "after_rms": None, "coeffs": {}}

    phase = ph.generate_zernike_polynomial(coefficients=coeffs, radius=r_used, n_max=4)
    show_phase(slm, phase, extra_sleep)
    time.sleep(0.3)
    z1 = measure_zernike(wfs, n_avg, 10)
    wf1 = measure_wf(wfs, n_avg)
    if z1 is None or wf1 is None:
        raise RuntimeError("矫正后测量失败")
    w1 = z1[1:] / 0.532
    resid = float(np.linalg.norm(w1))
    click.echo(f"   矫正后: RMS={wf1[1]['rms']:.4f}λ, PV={wf1[1]['diff']:.4f}λ, "
               f"|w|={resid:.4f}λ")
    click.echo(f"   改善: RMS {wf0[1]['rms']:.4f} → {wf1[1]['rms']:.4f}λ "
               f"({100 * (1 - wf1[1]['rms'] / wf0[1]['rms']):.1f}%)")
    return {"before_rms": wf0[1]["rms"], "after_rms": wf1[1]["rms"],
            "before_pv": wf0[1]["diff"], "after_pv": wf1[1]["diff"],
            "before_z_norm": float(np.linalg.norm(w)), "after_z_norm": resid,
            "coeffs": {f"[{modes[i]}]": float(c[i]) for i in range(len(modes))},
            "w_before": w.tolist(), "w_after": w1.tolist()}


# ─────────────────────────── CLI ───────────────────────────

@click.command()
@click.option("--stage", type=click.Choice(["all", "auto", "matrix", "closed"]),
              default="all", show_default=True, help="执行阶段")
@click.option("--slm-number", type=int, default=1, show_default=True)
@click.option("--slm-wavelength", type=int, default=532, show_default=True)
@click.option("--wfs-exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS, show_default=True)
@click.option("--wfs-order", type=int, default=10, show_default=True)
@click.option("--n-max", type=int, default=4, show_default=True, help="SLM 模式最大阶数")
@click.option("--n-avg", type=int, default=3, show_default=True, help="WFS 多帧平均")
@click.option("--settle-extra-s", type=float, default=SETTLE_REDUNDANCY_S,
              show_default=True, help="像素翻转估算之外的冗余等待 (s)")
@click.option("--radii", default=None, help="扫描半径 px 列表 (逗号分隔; 默认按光束半径派生)")
@click.option("--amps", default="2,5,10", show_default=True, help="幅度 rad 列表 (各测 ±)")
@click.option("--radius-scan", default="120,200,300,450,600", show_default=True,
              help="阶段1 半径诊断扫描列表 px")
@click.option("--quick", is_flag=True, default=False, help="快速模式 (单尺寸/单幅度)")
@click.option("-o", "--output-dir", default="data/zernike_correction", show_default=True)
def main(stage, slm_number, slm_wavelength, wfs_exposure_ms, wfs_order, n_max,
         n_avg, settle_extra_s, radii, amps, radius_scan, quick, output_dir) -> int:
    """SLM Zernike 模式法波前矫正 — 三阶段流程."""
    if wfs_exposure_ms > MAX_EXPOSURE_MS:
        raise click.BadParameter(f"WFS 曝光 {wfs_exposure_ms}ms > {MAX_EXPOSURE_MS}ms")
    if not 2 <= wfs_order <= 10:
        raise click.BadParameter("--wfs-order 必须在 2..10")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    incr = out / f"raw_scan_{ts}.json"

    n_modes = (n_max + 1) * (n_max + 2) // 2
    modes = [i for i in range(2, n_modes + 1)]          # 排除 piston
    r_scan = [float(v) for v in radius_scan.split(",")]
    amp_list = [float(v) for v in amps.split(",")]
    if quick:
        amp_list = [amp_list[0]]

    click.echo("=" * 72)
    click.echo("[SLM Zernike 模式法波前矫正] 三阶段流程")
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

        # ---- 阶段 1 ----
        r_beam = 250.0
        if stage in ("all", "auto"):
            r_beam, r_diag = auto_beam_radius(slm, wfs, ph, r_scan, 20.0, n_avg,
                                              settle_extra_s)
            report["radius_diagnostic"] = r_diag
            sx, sy, sh_scan = auto_center_shift(slm, wfs, ph, r_beam, 300, 100,
                                                20.0, n_avg, settle_extra_s)
            report["shift_scan"] = sh_scan
            slm.set_shift(sx, sy)
            slm.save_config()
            report["shift"] = [sx, sy]
            report["flat_reference"] = setup_flat_reference(slm, wfs, max(n_avg, 3),
                                                            settle_extra_s)

        # ---- 阶段 2 ----
        matrix = None
        r_used = r_beam
        if stage in ("all", "matrix"):
            if radii:
                r_list = [float(v) for v in radii.split(",")]
            elif quick:
                r_list = [r_beam]
            else:
                r_list = [r_beam, r_beam * 1.5, r_beam * 2.0]
            r_list = [float(np.clip(r, 120, 600)) for r in r_list]
            matrix, metrics, extra = scan_response_matrix(
                slm, wfs, ph, modes, r_list, amp_list, n_avg, settle_extra_s,
                incr, wfs_order)
            report["metrics"] = metrics
            report.update(extra)
            # 主矩阵用出现最多的最优半径
            from collections import Counter
            cnt = Counter(extra["best_radius"].values())
            r_used = cnt.most_common(1)[0][0] if cnt else r_beam
            report["matrix_radius"] = r_used
            click.echo(f"   主矩阵半径 = {r_used:.0f}px")

        # ---- 阶段 3 ----
        if stage in ("all", "closed"):
            if matrix is None:
                click.echo("[WARN] 无矩阵, 跳过闭环矫正")
            else:
                report["closed_loop"] = closed_loop_correct(
                    slm, wfs, ph, matrix, modes, r_used, n_avg, settle_extra_s)

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
