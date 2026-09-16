"""SLM + WFS 光强与 pupil 探针 — 标定前硬件状态检查.

用途
----
实机标定前确认两件事:

1. **光强**: SLM 纯平相位下 WFS 点阵光强是否足够且不饱和
   (max 强度、有效子孔径比例、均值/中位数); 曝光时间强制满足 ≤ 安全上限。
2. **pupil**: ``optimize_pupil()`` 结果**显式写回** ``wfs.pupil``
   (2026-09 教训: optimize_pupil 只计算不调用 WFS_SetPupil, 忘记写回 = 没设 pupil)。

探针输出 JSON 报告 + 终端摘要, 供标定前人工/自动门控。

用法
----
    python -m ao_shaping.tools.slm.slm_wfs_probe
    python -m ao_shaping.tools.slm.slm_wfs_probe --exposure-ms 4.0 --no-save
    
参数
----
    --slm-number      SLM 设备编号 (默认 1)
    --wavelength      SLM 工作波长 nm (默认 532)
    --mla-index       MLA 分辨率 (512/540/600/768/1280, 默认 512)
    --exposure-ms     WFS 曝光 ms (默认 4.0, 强制 ≤ 7.0)
    --no-save         不写回 pupil / 不保存报告 (dry run)
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
from loguru import logger

from ao_shaping.drivers.slm import Santec
from ao_shaping.drivers.wfs import MlaRes, ThorlabWFS

PANEL_H, PANEL_W = 1200, 1920
DEFAULT_OUTPUT = Path("data/calibration/wfs_light_pupil_probe.json")
MAX_EXPOSURE_MS = 7.0
DEFAULT_EXPOSURE_MS = 4.0


@click.command()
@click.option("--slm-number", type=int, default=1, show_default=True, help="SLM 设备编号")
@click.option("--wavelength", type=int, default=532, show_default=True, help="SLM 波长 nm")
@click.option("--mla-index", type=click.Choice(["512", "540", "600", "768", "1280"]),
              default="512", show_default=True, help="MLA 分辨率")
@click.option("--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS, show_default=True,
              help=f"WFS 曝光 ms (必须 ≤ {MAX_EXPOSURE_MS})")
@click.option("--no-save", is_flag=True, default=False, help="不写回 pupil / 不保存报告")
@click.option("-o", "--output", default=str(DEFAULT_OUTPUT), show_default=True,
              help="探针报告 JSON 路径")
def main(slm_number: int, wavelength: int, mla_index: str, exposure_ms: float,
         no_save: bool, output: str) -> int:
    """SLM 纯平 + WFS 光强/pupil 检查."""
    if exposure_ms > MAX_EXPOSURE_MS:
        raise click.BadParameter(
            f"WFS 曝光 {exposure_ms}ms 超过安全上限 {MAX_EXPOSURE_MS}ms"
        )

    click.echo("=" * 72)
    click.echo("[SLM + WFS 光强/pupil 探针]")
    click.echo("=" * 72)

    slm = Santec(slm_number=slm_number, wavelength=wavelength, video_mode=0)
    wfs = ThorlabWFS(mla_index=MlaRes.from_str(mla_index),
                     exposure_time=exposure_ms, use_custom_ref=False)

    report: dict = {"params": {"slm_number": slm_number, "wavelength": wavelength,
                               "mla_index": mla_index, "exposure_ms": exposure_ms}}
    ok = False
    try:
        slm.open()
        wl, max_gray = slm.get_wavelength_info()
        report["slm"] = {"serial": slm._serial_number, "wavelength_nm": wl,
                         "max_gray": max_gray, "shift_x": slm.shift_x,
                         "shift_y": slm.shift_y}
        click.echo(f"[OK] SLM open: serial={slm._serial_number}, {wl}nm, "
                   f"2π gray={max_gray}, shift=({slm.shift_x},{slm.shift_y})")

        wfs.open()
        exp = float(wfs.exposure_time)
        report["wfs"] = {"serial": wfs.serial_num, "exposure_time_ms": exp,
                         "mla_name": wfs.get_mla_name(),
                         "num_spots_x": int(wfs.num_spots_x),
                         "num_spots_y": int(wfs.num_spots_y)}
        click.echo(f"[OK] WFS open: serial={wfs.serial_num}, exposure={exp:.4f}ms, "
                   f"MLA={wfs.get_mla_name()} "
                   f"{wfs.num_spots_x}x{wfs.num_spots_y} spots")
        assert exp <= MAX_EXPOSURE_MS, f"曝光 {exp}ms 超过 {MAX_EXPOSURE_MS}ms"

        # SLM 纯平相位
        flat = np.full((PANEL_H, PANEL_W), 0, dtype=np.uint16)
        slm.display_data(flat, wait_time_s=0.5)

        # === 光强检查 ===
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        try:
            intensity, (cent_x, cent_y) = wfs.get_spots_statics()
        except AssertionError:
            logger.error("get_spots_statics 需要关闭 high speed mode")
            raise
        arr = np.asarray(intensity, dtype=float)
        valid = arr[np.isfinite(arr)]
        n_total = int(arr.size)
        n_valid = int(valid.size)
        valid_ratio = n_valid / n_total if n_total else 0.0
        light = {
            "n_total": n_total, "n_valid": n_valid,
            "valid_ratio": round(valid_ratio, 4),
            "max_intensity": float(np.max(valid)) if n_valid else None,
            "mean_intensity": float(np.mean(valid)) if n_valid else None,
            "median_intensity": float(np.median(valid)) if n_valid else None,
            "min_intensity": float(np.min(valid)) if n_valid else None,
        }
        report["light"] = light
        click.echo(f"[LIGHT] 有效子孔径 {n_valid}/{n_total} "
                   f"(ratio={valid_ratio:.3f}) | "
                   f"max={light['max_intensity']:.1f} "
                   f"mean={light['mean_intensity']:.1f} "
                   f"median={light['median_intensity']:.1f}")

        # === pupil 检查 ===
        cx, cy, dx, dy = wfs.optimize_pupil()
        report["pupil_optimized"] = [round(cx, 4), round(cy, 4),
                                     round(dx, 4), round(dy, 4)]
        click.echo(f"[PUPIL] optimize_pupil ⇒ center=({cx:.4f},{cy:.4f})mm "
                   f"diameter=({dx:.4f},{dy:.4f})mm")
        if not no_save:
            wfs.pupil = (cx, cy, dx, dy)  # 显式写回 (optimize_pupil 只计算不设置)
            click.echo(f"[OK] pupil 已写回: wfs.pupil = {wfs.pupil}")
        else:
            click.echo("[INFO] --no-save: pupil 未写回")

        ok = True
    except AssertionError as e:
        click.echo(f"[FAIL] 断言失败: {e}")
    except Exception as e:
        logger.exception("探针失败")
        click.echo(f"[FAIL] {type(e).__name__}: {e}")
    finally:
        for dev, name in ((wfs, "WFS"), (slm, "SLM")):
            try:
                dev.close()
                click.echo(f"[INFO] {name} close")
            except Exception as e:
                logger.warning("{} close: {}", name, e)

    if not no_save:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        click.echo(f"[INFO] 探针报告: {path}")
    click.echo("=" * 72)
    click.echo(f"[{'ALL PASS' if ok else 'FAILED'}] light_ok={report.get('light', {}).get('valid_ratio', 0) >= 0.5}")
    click.echo("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())