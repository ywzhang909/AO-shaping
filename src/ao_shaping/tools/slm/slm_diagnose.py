"""SLM 硬件自检工具 (slm-diagnose)。

在 2f Fourier 光路 (SLM 前焦面 125mm -> f=125mm 透镜 -> CCD 后焦面) 下,
对 Santec SLM-200 + MiiCam 相机做逐级硬件自检, 定位"面板不调制光"类故障。

本工具固化 2026-09 硬件诊断中确认的信息 与 成功识别方法::

  1. patches 冻结检测: 写入 flat / 全屏光栅 / 上下半屏光栅到不同内存槽, 对比各帧
     是否随图案变化。所有帧几乎相同 => 面板未被驱动 (LCOS 冻结)。
  2. 灰度调制检测: set_grayscale 扫描 0..1023, 测量 0 级桶能量是否随灰度
     周期性变化 (1064nm 下振幅耦合周期 ~993 灰度)。无周期 => 面板不调制光。
     注意: get_displayed_memory_number (SLM_Ctrl_ReadDS) 在 set_grayscale
     模式下报错码 1 是正常行为 (灰度模式无内存槽可读), 不是故障。
  3. 曝光→亮度线性检测: 曝光时间翻倍, 峰值亮度应随之增长; 曝光 ×20 而
     亮度不变 => 到达相机光强异常 (>100 倍弱于已知 ~0.02ms 临界饱和基线) 或
     面板未调制。
  4. 内存槽轮换: 连续 display_memory 到同一槽是 no-op, 必须轮换槽位。

已知约束: DVI 模式 (video_mode=1, SLM_Disp_Data) 的 open() 可能挂起,
挂起后连 memory 模式 open 也会挂, 只能物理断电重启 —— 本工具默认
只用 memory 模式 (video_mode=0), 绝不自动尝试 DVI。

用法::

    python -m ao_shaping.tools.slm.slm_diagnose [OPTIONS]

常用::

    python -m ao_shaping.tools.slm.slm_diagnose --period-ref 64 --period-test 32
    python -m ao_shaping.tools.slm.slm_diagnose --step freeze
    python -m ao_shaping.tools.slm.slm_diagnose --step modulate
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
from loguru import logger

# ── 已确认的硬件/光路事实 (2026-09 诊断固化, 勿改) ──────────────────────────

# SLM: Santec SLM-200, SLM#1 序列号 22030108, 1920x1200, 10-bit,
#      内部内存模式; @1064nm 2π 对应灰度 993 (设备动态查询, 不硬编码).
# MiiCam: 序列号 TP2408221418059418FD83E3A448D82, 2688x1520, MONO8.
# 光路: 严格 2f Fourier —— SLM 前焦面 125mm -> f=125mm 透镜 -> CCD 后焦面.
#       焦面=空间频率坐标: 半屏光栅上半/下半的 +1 级落在 CCD 同一行 (光轴行),
#       仅 x 偏移不同. 0 级 (光轴) = 帧内全局最大, 不是相机几何中心.
# 衍射偏移常数: Δx_px = 5021 / 周期 (λ·f/(d_SLM·p_cam)) -> P64=78, P32=157.
_DIFFRACTION_SCALE_PX = 5021.0

# 1064nm 振幅耦合的已知周期 (灰度) —— 调制正常时应在此周期附近出现桶能量峰.
_SLM_AMPLITUDE_PERIOD_GRAY = 993

# 已知正常曝光基线 (2026-09 实测): 2f 焦面处 ~0.02ms 曝光接近饱和.
_REFERENCE_EXPOSURE_MS = 0.02


@dataclass
class DiagnoseResult:
    """单步自检结果。"""

    ok: bool
    message: str
    metrics: dict[str, float] = field(default_factory=dict)


def _grab_frame(camera, n_sample: int = 5) -> np.ndarray:
    """取一帧平均图并转 float64。"""
    return np.asarray(
        camera.get_numpy_image(n_sample=n_sample, skip_first=True), dtype=np.float64,
    )


def peak_and_bucket(frame: np.ndarray, radius: int = 30) -> tuple[int, int, float, float]:
    """``(px, py, peak_value, bucket_sum)`` —— 0 级光斑峰值位置与桶能量。

    0 级 = 帧内全局最大 (2f 光路的光轴落点), 不是相机中心。
    """
    py, px = np.unravel_index(np.argmax(frame), frame.shape)
    y0, y1 = max(py - radius, 0), min(py + radius + 1, frame.shape[0])
    x0, x1 = max(px - radius, 0), min(px + radius + 1, frame.shape[1])
    bucket = float(frame[y0:y1, x0:x1].sum())
    return int(px), int(py), float(frame[py, px]), bucket


def frames_same(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    """两个相机帧是否几乎一致 (面板冻结判据)。"""
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=0, atol=max(1e-6, 1e-3 * float(a.max()))))

# ── 各步骤 ──────────────────────────────────────────────────────────────────


def step_freezing(
    slm,
    camera,
    period_ref: int,
    period_test: int,
    slm_wavelength: int,
    settle_s: float,
    exposure_ms: float,
) -> DiagnoseResult:
    """patches 冻结检测: 4 种图案写入不同内存槽, 对比帧是否随图案变化。

    图案: flat / 全屏 P_ref / 上半 P_test+下半 flat / 上半 flat+下半 P_test.
    面板正常 => 各帧明显不同; 全同 => LCOS 冻结 (面板未被驱动).
    """
    from ao_shaping.drivers.slm.santec_slm200 import MEMORY_MODE_INTERNAL
    from ao_shaping.utils.slm_lut import depth_pattern, stack_halves

    _, gray_for_2pi = slm.get_wavelength_info()
    w, h = slm.Panel_Res[0], slm.Panel_Res[1]
    half_h = h // 2

    flat_full = np.zeros((h, w), dtype=np.uint16)
    grat_full = depth_pattern(period_ref, gray_for_2pi, h, w)
    flat_h = np.zeros((half_h, w), dtype=np.uint16)
    test_p = depth_pattern(period_test, gray_for_2pi, half_h, w)
    test_top = stack_halves(test_p, flat_h, axis=0)
    test_bot = stack_halves(flat_h, test_p, axis=0)

    frames: dict[str, np.ndarray] = {}
    # 轮换槽位: 同一槽连续 display_memory 是 no-op (LCOS 不刷新).
    for tag, pat in [("flat", flat_full), ("grat", grat_full),
                     ("top", test_top), ("bot", test_bot)]:
        slot = 10 + len(frames)
        slm.write_phase(pat, memory_number=slot, memory_mode=MEMORY_MODE_INTERNAL)
        slm.display_memory(slot)
        time.sleep(settle_s)
        frames[tag] = _grab_frame(camera)

    px, py, peak, bucket = peak_and_bucket(frames["flat"])
    same = {t: frames_same(frames["flat"], f) for t, f in frames.items() if t != "flat"}
    n_diff = sum(1 for v in same.values() if not v)

    msg = (
        f"frames: flat max@{px},{py} peak={peak:.1f} bucket={bucket:.1f}; "
        f"pattern frames differ from flat: {n_diff}/3 "
        f"({', '.join(f'{k}={"same" if v else "diff"}' for k, v in same.items())}); "
        f"exposure={exposure_ms}ms"
    )
    ok = n_diff >= 2  # 至少 2/3 图案帧应与 flat 不同 => 面板在响应
    return DiagnoseResult(ok=ok, message=msg, metrics={"n_different": float(n_diff)})


def step_modulation(
    slm,
    camera,
    slm_wavelength: int,
    settle_s: float,
    exposure_ms: float,
) -> DiagnoseResult:
    """灰度调制检测: set_grayscale 0..1023, 测 0 级桶能量的 ~993 周期。

    1064nm 下正常面板有振幅耦合: 桶能量随灰度周期 (~993) 变化。
    无周期 => 面板不调制光 (硬件/偏振/激光故障), 与内存槽显示路径无关。
    """
    grays = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 993, 1023]
    buckets: list[float] = []
    peaks: list[float] = []
    for gs in grays:
        slm.set_grayscale(gs)
        time.sleep(settle_s)
        frame = _grab_frame(camera)
        _, _, peak, bucket = peak_and_bucket(frame)
        readback = slm.get_current_grayscale()
        if readback != gs:
            return DiagnoseResult(
                ok=False,
                message=f"set_grayscale readback mismatch: set={gs} read={readback}",
            )
        buckets.append(bucket)
        peaks.append(peak)
        logger.debug("gs={} readback={} max={:.1f} bucket={:.1f}", gs, readback, peak, bucket)

    buckets_arr = np.asarray(buckets, dtype=np.float64)
    spread = (float(buckets_arr.max()) - float(buckets_arr.min())) / (float(buckets_arr.mean()) or 1.0)
    ok = spread > 0.15  # 桶能量相对变化 >15% 才算"有调制"
    msg = (
        f"set_grayscale 0..1023: bucket range={buckets_arr.min():.1f}..{buckets_arr.max():.1f} "
        f"rel_spread={spread:.3f} peak_range={min(peaks):.1f}..{max(peaks):.1f}; "
        f"NOTE: get_displayed_memory_number error-code 1 in grayscale mode is "
        f"normal (no memory slot is being displayed)"
    )
    return DiagnoseResult(ok=ok, message=msg, metrics={"rel_spread": spread})


def step_linearity(
    slm,
    camera,
    period_ref: int,
    slm_wavelength: int,
    settle_s: float,
    base_exposure_ms: float,
) -> DiagnoseResult:
    """曝光→亮度线性检测: 曝光 x1 / x4 / x20, 峰值亮度应随之增长。

    全部曝光下亮度几乎不变 => 到达相机的光强异常 (知识库基线: ~0.02ms 应临界
    饱和) 或面板不调制。
    """
    from ao_shaping.drivers.slm.santec_slm200 import MEMORY_MODE_INTERNAL
    from ao_shaping.utils.slm_lut import depth_pattern

    _, gray_for_2pi = slm.get_wavelength_info()
    w, h = slm.Panel_Res[0], slm.Panel_Res[1]
    pat = depth_pattern(period_ref, gray_for_2pi, h, w)
    slm.write_phase(pat, memory_number=12, memory_mode=MEMORY_MODE_INTERNAL)
    slm.display_memory(12)
    time.sleep(settle_s)

    values: list[float] = []
    exposures = [base_exposure_ms, base_exposure_ms * 4, base_exposure_ms * 20]
    for exp in exposures:
        camera.reset_exposure_time(exp)
        time.sleep(settle_s)
        _, _, peak, bucket = peak_and_bucket(_grab_frame(camera))
        values.append(peak)
        logger.debug("exposure={:.3f}ms peak={:.1f}", exp, peak)

    growth = values[-1] / (values[0] or 1.0)
    ok = growth > 2.0  # x20 曝光应至少带来 x2 亮度增长
    msg = (
        f"exposure {exposures[0]:.3f}/{exposures[1]:.3f}/{exposures[2]:.3f} ms -> "
        f"peak {values[0]:.1f}/{values[1]:.1f}/{values[2]:.1f} (growth x20={growth:.2f}); "
        f"reference: ~{_REFERENCE_EXPOSURE_MS}ms was near-saturation on working setup"
    )
    return DiagnoseResult(ok=ok, message=msg, metrics={"growth_x20": growth})


# ── CLI ─────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--slm-number", type=int, default=1, help="SLM 设备编号 (默认 1)")
@click.option("--slm-wavelength", type=int, default=1064, help="SLM 工作波长 nm (默认 1064)")
@click.option("--cam-id", type=int, default=0, help="MiiCam 相机 ID (默认 0)")
@click.option("--period-ref", type=int, default=64, help="参考光栅周期 SLM px (默认 64)")
@click.option("--period-test", type=int, default=32, help="测试光栅周期 SLM px (默认 32)")
@click.option("--exposure-ms", type=float, default=2.0, help="自检曝光 ms (默认 2.0)")
@click.option("--settle-s", type=float, default=1.0, help="SLM/相机稳定等待 s (默认 1.0)")
@click.option(
    "--step",
    type=click.Choice(["all", "freeze", "modulate", "linearity"]),
    default="all",
    help="只跑某个步骤 (默认 all)",
)
@click.option("-o", "--output", default=None, help="保存诊断报告的目录 (默认不保存)")
def main(
    slm_number: int,
    slm_wavelength: int,
    cam_id: int,
    period_ref: int,
    period_test: int,
    exposure_ms: float,
    settle_s: float,
    step: str,
    output: str | None,
) -> None:
    """SLM 硬件自检: 逐级定位是否存在"面板不调制光"类故障。"""
    from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
    from ao_shaping.tools.slm.slm_lut_runner import _get_miicam_camera

    logger.info(
        "SLM self-check: slm#{} @{}nm, periods {}/{}px, camera#{} exposure {:.2f}ms "
        "(2f Fourier bench: SLM front-focus -> f=125mm lens -> CCD back-focus)",
        slm_number, slm_wavelength, period_ref, period_test, cam_id, exposure_ms,
    )

    if step in ("all", "freeze"):
        logger.warning(
            "Known constraint: DVI mode (video_mode=1) open() can hang; a hung "
            "controller then also hangs memory-mode open until physical power "
            "cycle. This tool uses memory mode (video_mode=0) only."
        )

    slm: SantecSLM200 | None = None
    camera = None
    results: dict[str, DiagnoseResult] = {}
    try:
        # 仅 memory 模式: 绝不自动进入 DVI 模式 (见 docstring 已知约束).
        slm = SantecSLM200(slm_number=slm_number, wavelength=slm_wavelength, video_mode=0)
        slm.open()
        _, gray_for_2pi = slm.get_wavelength_info()
        serial = slm.get_serial_number()
        logger.info(
            "SLM connected: serial={} panel={}x{} 2pi_gray={} (wavelength={}nm)",
            serial, slm.Panel_Res[0], slm.Panel_Res[1],
            gray_for_2pi, slm_wavelength,
        )

        camera = _get_miicam_camera(cam_id, exposure_ms)
        logger.info(
            "Camera opened: id={} exposure={:.2f}ms (frame readback on first grab)",
            cam_id, exposure_ms,
        )

        if step in ("all", "freeze"):
            results["freeze"] = step_freezing(
                slm, camera, period_ref, period_test, slm_wavelength, settle_s, exposure_ms,
            )
        if step in ("all", "modulate"):
            results["modulate"] = step_modulation(
                slm, camera, slm_wavelength, settle_s, exposure_ms,
            )
        if step in ("all", "linearity"):
            results["linearity"] = step_linearity(
                slm, camera, period_ref, slm_wavelength, settle_s, exposure_ms,
            )
    except SystemExit:
        raise
    except Exception:
        logger.exception("SLM self-check failed")
        raise
    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass
        if slm is not None:
            try:
                slm.close()
            except Exception:
                pass

    # ── 汇总 ──
    click.echo("\n=== SLM self-check summary ===")
    all_ok = True
    for name, r in results.items():
        click.echo(f"[{'PASS' if r.ok else 'FAIL'}] {name}: {r.message}")
        all_ok = all_ok and r.ok
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
            metrics_arr = np.asarray(list(r.metrics.values()), dtype=np.float64)
            keys_arr = np.asarray(list(r.metrics.keys()))
            np.savez(
                Path(output) / f"diagnose_{name}.npz",
                metrics=metrics_arr,
                metric_keys=keys_arr,
                message=np.asarray(r.message),
                ok=np.asarray(r.ok),
            )
    click.echo("")
    if not results:
        click.echo("No step executed -- nothing to report (choose --step or 'all').")
        return
    if all_ok:
        click.echo("RESULT: SLM panel responds to patterns/gray/exposure. If slm-lut "
                   "still fails, check spot geometry (2f Fourier) before hardware.")
    else:
        click.echo("RESULT: fault found. Panel does not modulate light OR light reaching "
                   "camera is abnormal. Actions: 1) power-cycle SLM controller "
                   "(DVI hang persists until physical reboot), 2) check polarization "
                   "axis vs LCOS, 3) verify beam on panel with sensor card, "
                   "4) confirm camera at back focal plane.")


if __name__ == "__main__":
    sys.exit(main())