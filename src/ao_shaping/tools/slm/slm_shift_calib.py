"""SLM defocus 平移标定 (shift_x / shift_y) — WFS tip/tilt 零点法.

用途
----
确定 `Santec` 的 `shift_x`/`shift_y`: 让相位图案中心对准光束光轴。光路调整后需重标定。

原理
----
图案平移 ``(sx, sy)`` 后光束感受到 ``P(b − s + ξ)`` (``b`` = 光束光轴在 SLM 坐标中的
偏移, 未知)。对 defocus ``P = D·u²`` 有局部梯度 ``∝ 2D(b − s)`` → **WFS 读出的
tip/tilt 关于 shift 线性, 零点即 ``s = b``**(图案中心与光束对齐)。

判据用**相对纯平的"附加"倾斜** ``‖z_tilt(defocus@shift) − z_tilt(flat)‖`` —— 系统
本身有静态倾斜 (实测 flat 下 tilt ≈ −0.14λ), 绝对归零是错的判据。

关键约束 (2026-09-15 实测踩坑)
------------------------------
1. **幅度 A 必须足够大** (默认 20 rad): A=2 时响应被 ``(r_beam/R)²`` 压制到噪声级。
2. **Zernike 半径 R 必须 > 光束半径**: 实测光束在 SLM 上半径 ≈200px (≈1.6mm)。
   R=200 响应最强但一平移就裁切光束; 默认 R=600 可平移 ±400px 不裁切。
   光束尺寸可由 ``--radius-scan`` 诊断 (R 扫描取 Δdefocus 最大者)。
3. **shift 限制 ±500**: defocus 盘中心 ``(960+sx, 600+sy)`` 超出 1920×1200 面板后
   光束几乎看不到图案, 数据无效。
4. **SLM 轴 ↔ WFS 轴存在 90° 交换**: 本机实测 SLM-x → WFS Noll3 (y-tilt),
   SLM-y → WFS Noll2 (x-tip)。故判据取轴无关的 ``‖Δz_tilt‖``。

实测结果 (SLM#22030102 + WFS M01219666, 532nm): ``shift_x=106, shift_y=40``
(附加倾斜 0.854λ → 0.0245λ, 降低 97.1%)。

用法
----
    python -m ao_shaping.tools.slm.slm_shift_calib
    python -m ao_shaping.tools.slm.slm_shift_calib --defocus-a 30 --radius-scan
    python -m ao_shaping.tools.slm.slm_shift_calib --no-save      # 只测不写 config
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import click
import numpy as np
from loguru import logger

from ao_shaping.drivers.slm import Santec
from ao_shaping.drivers.wfs import ThorlabWFS
from ao_shaping.tools.slm.slm_scan_analysis import clamp_shift, parabolic_min
from ao_shaping.tools.slm.slm_zernike_common import make_phase, measure_tilt_defocus
from ao_shaping.utils.pattern_helper import PatternHelper

PANEL_H, PANEL_W = 1200, 1920
DEFAULT_CONFIG = Path("data/slm_configs/slm")
DEFAULT_OUTPUT = Path("data/calibration/defocus_shift_calib.json")

# 安全阈值
MAX_EXPOSURE_MS = 7.0
DEFAULT_EXPOSURE_MS = 4.0
DEFAULT_SHIFT_LIMIT = 500


def diagnose_radius(
    slm: Santec,
    wfs: ThorlabWFS,
    ph: PatternHelper,
    radii: tuple[float, ...],
    amplitude: float,
    n_avg: int,
) -> list[dict]:
    """扫描 Zernike 半径, 找 WFS defocus 响应最大者 (≈ 光束半径)."""
    results: list[dict] = []
    for r in radii:
        phase = make_phase(ph, {(2, 0): amplitude}, r, n_max=5)
        slm.set_shift(0, 0)
        slm.display_phase(phase, wait_time_s=0.5)
        time.sleep(0.25)
        zt, zd = measure_tilt_defocus(wfs, n_avg=n_avg)
        rec = {"radius": r, "amplitude": amplitude,
               "defocus": None if zd is None else zd}
        if zt is not None:
            rec.update({"tip": float(zt[0]), "tilt": float(zt[1])})
        results.append(rec)
        click.echo(
            f"  R={r:6.0f}px A={amplitude:5.1f} → defocus="
            f"{'n/a' if zd is None else f'{zd:+.4f}'}λ"
        )
    valid = [r for r in results if r["defocus"] is not None]
    if valid:
        best = max(valid, key=lambda r: abs(r["defocus"]))
        click.echo(f"  → 最强响应 R={best['radius']:.0f}px (≈ 光束半径)")
    return results


@click.command()
@click.option("--slm-number", type=int, default=1, show_default=True, help="SLM 设备编号")
@click.option("--slm-wavelength", type=int, default=532, show_default=True, help="SLM 波长 nm")
@click.option("--wfs-exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS,
              show_default=True, help=f"WFS 曝光 ms (必须 ≤ {MAX_EXPOSURE_MS})")
@click.option("--zernike-radius", type=float, default=600.0, show_default=True,
              help="defocus Zernike 半径 px (必须 > 光束半径, 默认 600)")
@click.option("--defocus-a", type=float, default=20.0, show_default=True,
              help="defocus 幅度 rad (太小则响应淹没在噪声中)")
@click.option("--shift-limit", type=int, default=DEFAULT_SHIFT_LIMIT, show_default=True,
              help="shift 绝对值上限 (防 defocus 盘推出面板)")
@click.option("--coarse-step", type=int, default=100, show_default=True, help="粗扫步长 px")
@click.option("--coarse-half", type=int, default=300, show_default=True, help="粗扫半宽 px")
@click.option("--fine-half", type=int, default=40, show_default=True, help="细扫半宽 px")
@click.option("--iterations", type=int, default=2, show_default=True, help="x/y 迭代轮数")
@click.option("--n-avg-scan", type=int, default=3, show_default=True, help="扫描帧平均次数")
@click.option("--n-avg-verify", type=int, default=5, show_default=True, help="校验帧平均次数")
@click.option("--improve-ratio", type=float, default=0.5, show_default=True,
              help="质量门控: 附加倾斜需降低到该比例以下才写入 config")
@click.option("--radius-scan", is_flag=True, default=False,
              help="先诊断光束半径 (扫 Zernike R), 不做标定")
@click.option("--no-save", is_flag=True, default=False,
              help="只测量, 不写入 SLM config (dry run)")
@click.option("-o", "--output", default=str(DEFAULT_OUTPUT), show_default=True,
              help="标定报告 JSON 路径")
def main(
    slm_number: int,
    slm_wavelength: int,
    wfs_exposure_ms: float,
    zernike_radius: float,
    defocus_a: float,
    shift_limit: int,
    coarse_step: int,
    coarse_half: int,
    fine_half: int,
    iterations: int,
    n_avg_scan: int,
    n_avg_verify: int,
    improve_ratio: float,
    radius_scan: bool,
    no_save: bool,
    output: str,
) -> int:
    """SLM defocus 平移标定 — WFS tip/tilt 零点法."""
    if wfs_exposure_ms > MAX_EXPOSURE_MS:
        raise click.BadParameter(
            f"WFS 曝光 {wfs_exposure_ms}ms 超过安全上限 {MAX_EXPOSURE_MS}ms"
        )
    if zernike_radius <= 0:
        raise click.BadParameter("--zernike-radius 必须 > 0")

    click.echo("=" * 72)
    click.echo("[SLM defocus shift 标定] WFS tip/tilt 零点法")
    click.echo("=" * 72)

    slm = Santec(slm_number=slm_number, wavelength=slm_wavelength, video_mode=0)
    wfs = ThorlabWFS(exposure_time=wfs_exposure_ms, use_custom_ref=False)
    ph = PatternHelper(resolution=(PANEL_W, PANEL_H))

    report: dict = {
        "zernike_radius": zernike_radius,
        "defocus_a": defocus_a,
        "method": "minimize ||z_tilt(defocus@shift) - z_tilt(flat)||",
        "evals": [],
    }
    ok = False
    sx_star, sy_star = 0.0, 0.0
    original: tuple[int, int] | None = None

    try:
        slm.open()
        wl, max_gray = slm.get_wavelength_info()
        original = (slm.shift_x, slm.shift_y)
        report["original_shift"] = list(original)
        click.echo(f"[OK] SLM open: serial={slm._serial_number}, {wl}nm, "
                   f"2π gray={max_gray}, 当前 shift={original}")

        wfs.open()
        exp = float(wfs.exposure_time)
        click.echo(f"[OK] WFS open: serial={wfs.serial_num}, exposure={exp:.4f}ms")
        assert exp <= MAX_EXPOSURE_MS, f"曝光 {exp}ms 超过 {MAX_EXPOSURE_MS}ms"

        # pupil 必须自动获取并显式写回 (optimize_pupil 只计算不设置)
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        cx, cy, dx, dy = wfs.pupil = wfs.optimize_pupil()
        click.echo(f"[OK] pupil: center=({cx:.3f},{cy:.3f})mm, "
                   f"diameter=({dx:.3f},{dy:.3f})mm")
        report["pupil"] = [cx, cy, dx, dy]

        flat = np.full((PANEL_H, PANEL_W), 0, dtype=np.uint16)

        if radius_scan:
            click.echo("\n[诊断] 扫描 Zernike 半径 → WFS defocus 响应")
            report["radius_diagnostic"] = diagnose_radius(
                slm, wfs, ph, (120.0, 200.0, 300.0, 450.0, 600.0), defocus_a, n_avg_scan
            )
            slm.display_data(flat, wait_time_s=0.5)
            _write_report(output, report)
            return 0

        # 基线: 纯平下的静态倾斜 (判据基准)
        slm.set_shift(0, 0)
        slm.display_data(flat, wait_time_s=0.5)
        time.sleep(0.3)
        base_tilt, base_def = measure_tilt_defocus(wfs, n_avg=n_avg_verify)
        if base_tilt is None:
            raise RuntimeError("基线测量失败: 无有效 zernike")
        click.echo(f"[BASE] 纯平: tip={base_tilt[0]:+.4f}λ tilt={base_tilt[1]:+.4f}λ "
                   f"(defocus={base_def:+.4f}λ) ← 判据为相对此值的附加倾斜")
        report["baseline"] = {"tip": float(base_tilt[0]), "tilt": float(base_tilt[1])}

        phase_rad = make_phase(ph, {(2, 0): defocus_a}, zernike_radius, n_max=5)
        click.echo(f"[INFO] defocus R={zernike_radius:.0f}px A={defocus_a}rad, "
                   f"range=[{phase_rad.min():.1f},{phase_rad.max():.1f}] rad")

        def evaluate(sx: int, sy: int, n_avg: int, tag: str) -> float | None:
            slm.set_shift(clamp_shift(sx, shift_limit), clamp_shift(sy, shift_limit))
            slm.display_phase(phase_rad, wait_time_s=0.5)
            time.sleep(0.25)
            zt, zd = measure_tilt_defocus(wfs, n_avg=n_avg)
            if zt is None:
                logger.warning("{} shift=({},{}) 无有效 zernike", tag, sx, sy)
                return None
            added = zt - base_tilt
            norm = float(np.linalg.norm(added))
            report["evals"].append({
                "tag": tag, "sx": int(sx), "sy": int(sy),
                "tip": float(zt[0]), "tilt": float(zt[1]),
                "added_tip": float(added[0]), "added_tilt": float(added[1]),
                "added_norm": norm, "defocus": zd,
            })
            click.echo(f"[EVAL] {tag:14s} shift=({sx:5d},{sy:5d}) "
                       f"Δtip={added[0]:+.4f} Δtilt={added[1]:+.4f} "
                       f"‖Δ‖={norm:.4f} defocus={zd:+.4f}")
            return norm

        def scan_axis(axis: str, other: int, values, tag: str):
            best_v, best_m, pts = None, np.inf, []
            for v in values:
                sx = int(v) if axis == "x" else other
                sy = int(v) if axis == "y" else other
                m = evaluate(sx, sy, n_avg_scan, tag)
                if m is None:
                    continue
                pts.append((float(v), m))
                if m < best_m:
                    best_m, best_v = m, float(v)
            return best_v, best_m, pts

        coarse = list(range(-coarse_half, coarse_half + 1, coarse_step))

        for it in range(1, iterations + 1):
            click.echo("\n" + "=" * 72)
            click.echo(f"[迭代 {it}/{iterations}]")
            click.echo("=" * 72)

            bx, bm, _ = scan_axis("x", clamp_shift(sy_star, shift_limit), coarse, f"it{it}-x-coarse")
            if bx is None:
                raise RuntimeError("X 粗扫无有效点")
            fine_x = [bx + d for d in np.linspace(-fine_half, fine_half, 5)]
            click.echo(f"[X 粗扫] 最优 sx={bx:.0f} (‖Δ‖={bm:.4f}) → 细扫 "
                       f"{[round(v) for v in fine_x]}")
            bx2, bm2, pts2 = scan_axis("x", clamp_shift(sy_star, shift_limit), fine_x,
                                       f"it{it}-x-fine")
            if bx2 is not None:
                px = parabolic_min(sorted(pts2))
                sx_star = float(clamp_shift(px if px is not None else bx2, shift_limit))
                click.echo(f"[X] → sx*={sx_star:.1f} (‖Δ‖={bm2:.4f})")

            by, bmy, _ = scan_axis("y", clamp_shift(sx_star, shift_limit), coarse, f"it{it}-y-coarse")
            if by is None:
                raise RuntimeError("Y 粗扫无有效点")
            fine_y = [by + d for d in np.linspace(-fine_half, fine_half, 5)]
            click.echo(f"[Y 粗扫] 最优 sy={by:.0f} (‖Δ‖={bmy:.4f}) → 细扫 "
                       f"{[round(v) for v in fine_y]}")
            by2, bmy2, ptsy2 = scan_axis("y", clamp_shift(sx_star, shift_limit), fine_y,
                                         f"it{it}-y-fine")
            if by2 is not None:
                py = parabolic_min(sorted(ptsy2))
                sy_star = float(clamp_shift(py if py is not None else by2, shift_limit))
                click.echo(f"[Y] → sy*={sy_star:.1f} (‖Δ‖={bmy2:.4f})")

        # 校验: 标定 shift vs (0,0)
        click.echo("\n" + "=" * 72)
        click.echo(f"[VERIFY] shift=({sx_star:.0f},{sy_star:.0f}) vs (0,0)")
        click.echo("=" * 72)
        m_star = evaluate(int(round(sx_star)), int(round(sy_star)), n_avg_verify, "verify")
        m_zero = evaluate(0, 0, n_avg_scan, "verify-zero")
        if m_star is None or m_zero is None:
            raise RuntimeError("校验测量失败")
        ratio = m_star / m_zero if m_zero > 0 else 1.0
        click.echo(f"[改善] ‖Δz_tilt‖ {m_zero:.4f}λ → {m_star:.4f}λ "
                   f"({100 * (1 - ratio):.1f}% 降低)")
        report["verify"] = {"shift": [sx_star, sy_star], "added_norm": m_star,
                            "zero_shift_added_norm": m_zero, "improvement_ratio": ratio}

        quality_ok = m_star < m_zero and ratio < improve_ratio
        report["quality_ok"] = bool(quality_ok)
        click.echo(f"[GATE] 比值={ratio:.3f} (阈值 {improve_ratio}) → "
                   f"{'通过' if quality_ok else '不通过'}")

        if quality_ok and not no_save:
            slm.set_shift(int(round(sx_star)), int(round(sy_star)))
            slm.save_config()
            click.echo(f"[OK] config 已写入: shift_x={slm.shift_x}, shift_y={slm.shift_y}")
            report["saved_shift"] = [slm.shift_x, slm.shift_y]
            ok = True
        elif quality_ok and no_save:
            click.echo(f"[INFO] --no-save: 标定通过但未写入 config "
                       f"(建议 shift=({sx_star:.0f},{sy_star:.0f}))")
            report["saved_shift"] = None
            # close() 会自动 save_config() → 必须先把 shift 还原, 否则 dry run 仍会改配置
            if original is not None:
                slm.set_shift(*original)
                click.echo(f"[INFO] --no-save: 已还原 shift={original} (防 close() 自动保存)")
            ok = True
        else:
            if original is not None:
                slm.set_shift(*original)
                slm.save_config()
                click.echo(f"[WARN] 质量门控不通过 → 恢复原始 shift={original}")
                report["saved_shift"] = list(original)

        slm.display_data(flat, wait_time_s=0.5)

    except AssertionError as e:
        click.echo(f"[FAIL] 断言失败: {e}")
    except Exception as e:
        logger.exception("标定失败")
        click.echo(f"[FAIL] {type(e).__name__}: {e}")
        # 异常路径也恢复原始 shift, 避免 close() 自动保存垃圾值
        if original is not None:
            try:
                slm.set_shift(*original)
                slm.save_config()
                click.echo(f"[INFO] 已恢复原始 shift={original}")
            except Exception as e2:
                logger.warning("恢复原始 shift 失败: {}", e2)
    finally:
        for dev, name in ((wfs, "WFS"), (slm, "SLM")):
            try:
                dev.close()
                click.echo(f"[INFO] {name} close")
            except Exception as e:
                logger.warning("{} close: {}", name, e)

    _write_report(output, report)
    click.echo("=" * 72)
    click.echo(f"[{'ALL PASS' if ok else 'SOME FAILURES'}] "
               f"final_shift=({sx_star:.0f},{sy_star:.0f})")
    click.echo("=" * 72)
    return 0 if ok else 1


def _write_report(output: str, report: dict) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(f"[INFO] 标定报告: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
