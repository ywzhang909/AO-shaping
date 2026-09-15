"""SLM + WFS 参考波前标定与倾斜线性度 (三步流程).

用户指定顺序:
  Step 1: SLM **纯平相位** → WFS 保存当前波前为参考 (`create_default_user_ref` +
          `save_user_ref` + `set_ref_plane(custom=True)`), 并**校验纯平波前平整度**
  Step 2: 还原内置参考 (`set_ref_plane(False)`) → 验证; 加载保存的参考 → 验证
  Step 3: SLM 加载不同强度 Zernike 倾斜 → 在纯平标定的参考下读出倾斜 → 线性度

判据说明
--------
- 倾斜线性度用 **zernike LSF 度量为主** (`get_zernike` 直接由 spot deviations 拟合,
  正确 pupil 下最干净); plane 拟合 (对波前图做平面拟合) 作为交叉验证 —— 其小倾斜端
  受噪声/高阶像差干扰, 实测 R² 明显低于 zernike 度量。
- **pupil 必须 `wfs.pupil = wfs.optimize_pupil()` 显式写回**: `optimize_pupil()` 只计算
  不调用 `WFS_SetPupil`; 硬编码 pupil 会让边界无效子孔径污染全孔径拟合 (实测假 tip/tilt
  达 4.6~12.8λ)。

用法:
    python -m ao_shaping.tools.slm.slm_wfs_reference
    python -m ao_shaping.tools.slm.slm_wfs_reference --tilt-amps 0.1,0.2,0.4,0.8,1.6,3.2
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
    SETTLE_REDUNDANCY_S,
    flat_gray,
    make_phase,
    measure_wavefront,
    measure_zernike,
    show_phase,
)
from ao_shaping.utils.pattern_helper import PatternHelper

PANEL_H, PANEL_W = 1200, 1920


def fit_tilt_plane(wavefront: np.ndarray) -> dict:
    """对波前图 (waves) 做平面拟合 wf = a·xx + b·yy + c.

    边界子孔径无光照时 WFS 返回 NaN → 计算前取 finite 掩膜。
    斜率单位 = waves/subaperture。
    """
    ny, nx = wavefront.shape
    yy, xx = np.meshgrid(np.arange(nx), np.arange(ny))
    mask = np.isfinite(wavefront)
    n_valid = int(mask.sum())
    if n_valid < 3:
        return {"a": np.nan, "b": np.nan, "c": np.nan, "n_valid": n_valid,
                "n_nan": int((~mask).sum()), "pv": np.nan, "rms": np.nan,
                "fit_resid_rms": np.nan}
    xx_m, yy_m, wf_m = xx[mask], yy[mask], wavefront[mask]
    A = np.column_stack([xx_m, yy_m, np.ones_like(xx_m)])
    coef, *_ = np.linalg.lstsq(A, wf_m, rcond=None)
    resid = wf_m - A @ coef
    return {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2]),
            "n_valid": n_valid, "n_nan": int((~mask).sum()),
            "pv": float(wf_m.max() - wf_m.min()),
            "rms": float(np.sqrt(np.mean(wf_m ** 2))),
            "fit_resid_rms": float(np.sqrt(np.mean(resid ** 2)))}


def measure_tilt(wfs: ThorlabWFS, n_avg: int, zernike_order: int) -> dict:
    """取图并返回 plane 拟合 + zernike tip/tilt (中位数聚合)."""
    fits, ztilts = [], []
    for _ in range(n_avg):
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        wf, stats = wfs.get_wavefront(cancel_tile=False)
        fit = fit_tilt_plane(wf)
        fit["wf_stats"] = stats
        fit["wf"] = wf
        fits.append(fit)
        try:
            z = wfs.get_zernike(zernike_order=zernike_order)
        except Exception:
            z = None
        if z is not None and np.isfinite(z[2:4]).all():
            ztilts.append(np.asarray(z[2:4], dtype=float) / 0.532)
    if not fits:
        raise RuntimeError("measure_tilt: 无有效波前")
    aggr: dict[str, Any] = {k: float(np.median([f[k] for f in fits]))
                            for k in ("a", "b", "c", "pv", "rms", "fit_resid_rms")}
    aggr["n_valid"] = int(np.median([f["n_valid"] for f in fits]))
    aggr["n_nan"] = int(np.median([f["n_nan"] for f in fits]))
    aggr["wf_stats"] = fits[-1]["wf_stats"]
    aggr["z_tilt"] = (np.median(np.array(ztilts), axis=0)
                      if ztilts else np.zeros(2))
    return aggr


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """过原点线性拟合 y = k·x, 返回 (k, R²)."""
    valid = np.isfinite(y)
    if valid.sum() < 3:
        return float("nan"), float("nan")
    k, *_ = np.linalg.lstsq(x[valid][:, None], y[valid], rcond=None)
    kk = float(k[0])
    ss_res = float(np.sum((y[valid] - kk * x[valid]) ** 2))
    ss_tot = float(np.sum((y[valid] - y[valid].mean()) ** 2))
    return kk, (1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


@click.command()
@click.option("--slm-number", type=int, default=1, show_default=True)
@click.option("--slm-wavelength", type=int, default=532, show_default=True)
@click.option("--wfs-exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS, show_default=True,
              help=f"WFS 曝光 ms (必须 ≤ {MAX_EXPOSURE_MS})")
@click.option("--wfs-order", type=int, default=10, show_default=True)
@click.option("--zernike-radius", type=float, default=600.0, show_default=True,
              help="Step3 倾斜 Zernike 半径 px (建议 ≥1.5×光束半径)")
@click.option("--tilt-amps", default="0.1,0.2,0.4,0.8,1.6,3.2", show_default=True,
              help="倾斜幅度序列 rad (Zernike (1,1) 系数)")
@click.option("--n-avg", type=int, default=5, show_default=True, help="每点 WFS 帧平均")
@click.option("--settle-extra-s", type=float, default=SETTLE_REDUNDANCY_S,
              show_default=True, help="像素翻转估算之外的冗余等待 (s)")
@click.option("--flat-rms-threshold", type=float, default=0.05, show_default=True,
              help="纯平波前 RMS 合格阈值 (λ)")
@click.option("-o", "--output", default=None, help="报告 JSON 路径")
def main(slm_number, slm_wavelength, wfs_exposure_ms, wfs_order, zernike_radius,
         tilt_amps, n_avg, settle_extra_s, flat_rms_threshold, output) -> int:
    """SLM + WFS 参考波前标定与倾斜线性度 (三步)."""
    if wfs_exposure_ms > MAX_EXPOSURE_MS:
        raise click.BadParameter(f"WFS 曝光 {wfs_exposure_ms}ms > {MAX_EXPOSURE_MS}ms")
    if not 2 <= wfs_order <= 10:
        raise click.BadParameter("--wfs-order 必须在 2..10")

    amps = [float(v) for v in tilt_amps.split(",")]
    click.echo("=" * 72)
    click.echo("[SLM+WFS 参考波前标定 + 倾斜线性度] 三步流程")
    click.echo("=" * 72)

    slm = Santec(slm_number=slm_number, wavelength=slm_wavelength, video_mode=0)
    wfs = ThorlabWFS(exposure_time=wfs_exposure_ms, use_custom_ref=False)
    ph = PatternHelper(resolution=(PANEL_W, PANEL_H))
    report: dict = {"timestamp": datetime.now().isoformat(), "tilt_amps": amps}
    all_ok = True

    try:
        slm.open()
        wfs.open()
        exp = float(wfs.exposure_time)
        assert exp <= MAX_EXPOSURE_MS
        wl, max_gray = slm.get_wavelength_info()
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        cx, cy, dx, dy = wfs.pupil = wfs.optimize_pupil()
        report["device"] = {"slm": slm._serial_number, "wfs": wfs.serial_num,
                            "wavelength_nm": slm_wavelength, "exposure_ms": exp,
                            "2pi_gray": max_gray, "pupil_center_mm": [cx, cy],
                            "pupil_diameter_mm": [dx, dy]}
        click.echo(f"[OK] SLM #{slm._serial_number} {wl}nm 2π={max_gray}; "
                   f"WFS {wfs.serial_num} exp={exp:.3f}ms")
        click.echo(f"[OK] pupil auto: center=({cx:.3f},{cy:.3f})mm "
                   f"d=({dx:.3f},{dy:.3f})mm")

        # ---------- Step 1 ----------
        click.echo("\n" + "=" * 72)
        click.echo("[STEP 1] SLM 纯平相位 → 保存为 WFS 参考波前 + 平整度校验")
        click.echo("=" * 72)
        slm.display_data(flat_gray(), wait_time_s=0.5)
        time.sleep(0.4)
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        ok = wfs.create_default_user_ref()
        backup = wfs.save_user_ref(backup_dir=Path("data/calibration"))
        wfs.set_ref_plane(custom=True)
        click.echo(f"[{'OK' if ok else 'FAIL'}] create_default_user_ref={ok}, "
                   f"use_custom_ref={wfs.use_custom_ref}")
        click.echo(f"[OK] save_user_ref → {backup}")
        if not ok or not wfs.use_custom_ref:
            raise RuntimeError("用户参考创建/激活失败")

        rec = measure_wavefront(wfs, max(n_avg, 3))
        if rec is None:
            raise RuntimeError("平整度测量失败")
        _, st = rec
        z0 = measure_zernike(wfs, max(n_avg, 3), wfs_order)
        z0n = float(np.linalg.norm(z0[2:7])) / 0.532 if z0 is not None else float("nan")
        flat_ok = st["rms"] < flat_rms_threshold
        click.echo(f"[{'OK' if flat_ok else 'WARN'}] 纯平波前 RMS={st['rms']:.4f}λ, "
                   f"PV={st['diff']:.4f}λ, |z[2..6]|={z0n:.4f}λ "
                   f"(阈值 {flat_rms_threshold}λ)")
        report["step1"] = {"create_ok": bool(ok), "backup_ref": str(backup),
                           "use_custom_ref": bool(wfs.use_custom_ref),
                           "flat_rms_lam": st["rms"], "flat_pv_lam": st["diff"],
                           "flat_z_norm_lam": z0n, "flat_ok": bool(flat_ok)}
        all_ok &= flat_ok

        fit_ref = measure_tilt(wfs, n_avg, wfs_order)
        click.echo(f"[INFO] 参考态: rms={fit_ref['rms']:.6f}λ, pv={fit_ref['pv']:.6f}λ, "
                   f"valid={fit_ref['n_valid']}/{fit_ref['n_valid'] + fit_ref['n_nan']}")

        # ---------- Step 2 ----------
        click.echo("\n" + "=" * 72)
        click.echo("[STEP 2] 还原内置参考 → 加载保存的参考 → 交替验证")
        click.echo("=" * 72)
        wfs.set_ref_plane(custom=False)
        fit_builtin = measure_tilt(wfs, n_avg, wfs_order)
        diff_b = abs(fit_builtin["rms"] - fit_ref["rms"]) * 1000
        click.echo(f"[INFO] 内置参考: rms={fit_builtin['rms']:.6f}λ "
                   f"(与自定义参考差 {diff_b:.3f} mλ)")
        click.echo(f"[{'OK' if diff_b > 10 else 'WARN'}] 内置与自定义参考"
                   f"{'差异显著 → 切换生效' if diff_b > 10 else '几乎无差异'}")

        ok_load = wfs.load_user_ref(backup) if backup else False
        wfs.set_ref_plane(custom=True)
        fit_loaded = measure_tilt(wfs, n_avg, wfs_order)
        diff_l = abs(fit_loaded["rms"] - fit_ref["rms"]) * 1000
        click.echo(f"[{'OK' if ok_load else 'FAIL'}] load_user_ref={ok_load}; "
                   f"加载后 rms={fit_loaded['rms']:.6f}λ (与保存时差 {diff_l:.3f} mλ)")
        report["step2"] = {"builtin_rms": fit_builtin["rms"],
                           "builtin_vs_custom_mLam": diff_b,
                           "load_ok": bool(ok_load),
                           "loaded_rms": fit_loaded["rms"],
                           "loaded_vs_saved_mLam": diff_l}
        all_ok &= bool(ok_load) and diff_l < 50

        # ---------- Step 3 ----------
        click.echo("\n" + "=" * 72)
        click.echo(f"[STEP 3] Zernike 倾斜扫描 {amps} rad → 线性度")
        click.echo("=" * 72)
        results = []
        for a in amps:
            phase = make_phase(ph, {(1, 1): a}, zernike_radius, n_max=5)
            show_phase(slm, phase, settle_extra_s)
            fit = measure_tilt(wfs, n_avg, wfs_order)
            zt = fit.pop("z_tilt")
            results.append({"A": a, **fit, "z_tilt": zt})
            click.echo(f"[DATA] A={a:5.2f} rad → plane a={fit['a']:+.4f} "
                       f"b={fit['b']:+.4f} λ/sub, rms={fit['rms']:.4f}λ, "
                       f"|z|={np.linalg.norm(zt):.4f}λ")

        A_arr = np.array([r["A"] for r in results])
        tilt_plane = np.array([np.hypot(r["a"], r["b"]) for r in results])
        tilt_zern = np.array([float(np.linalg.norm(r["z_tilt"])) for r in results])
        k_p, r2_p = fit_linear(A_arr, tilt_plane)
        k_z, r2_z = fit_linear(A_arr, tilt_zern)
        click.echo("\n----- 线性度 (输入 Zernike 系数 A rad → 输出倾斜) -----")
        click.echo(f"[INFO] plane   |tilt| = {k_p:.6f} × A   R²={r2_p:.6f}")
        click.echo(f"[INFO] zernike |tilt| = {k_z:.6f} × A   R²={r2_z:.6f}")
        best_r2 = max(r2_p, r2_z)
        lin_ok = bool(np.isfinite(best_r2) and best_r2 > 0.95)
        click.echo(f"[{'OK' if lin_ok else 'FAIL'}] 线性度 R²={best_r2:.4f} "
                   f"{'> 0.95' if lin_ok else '<= 0.95'}")
        report["step3"] = {"results": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                        for k, v in r.items() if k != "wf"}
                                       for r in results],
                           "k_plane": k_p, "r2_plane": r2_p,
                           "k_zernike": k_z, "r2_zernike": r2_z, "ok": lin_ok}
        all_ok &= lin_ok

        slm.display_data(flat_gray(), wait_time_s=0.5)

    except AssertionError as e:
        click.echo(f"[FAIL] 断言失败: {e}")
        all_ok = False
    except Exception as e:
        logger.exception("流程失败")
        click.echo(f"[FAIL] {type(e).__name__}: {e}")
        all_ok = False
    finally:
        for dev in (wfs, slm):
            try:
                dev.close()
            except Exception:
                pass

    report["all_ok"] = bool(all_ok)
    out = Path(output) if output else Path("data/calibration") / (
        f"slm_wfs_reference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(f"\n[OK] 报告: {out}")
    click.echo("=" * 72)
    click.echo(f"[{'ALL PASS' if all_ok else 'SOME FAILURES'}]")
    click.echo("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
