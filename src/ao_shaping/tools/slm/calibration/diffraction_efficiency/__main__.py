"""``python -m ao_shaping.tools.slm.calibration.diffraction_efficiency`` 入口。

三个子命令:

``simulate``
    用数字孪生跑完整条链路 (图案 → 远场 → 阶次识别 → 效率 → 拟合),
    用来在没有硬件时验证分析代码, 也用来造回归数据。
``measure``
    硬件采集 (Santec SLM + Daheng CCD)。**前置条件**: 已有的光路标定
    ``.npz`` (由 ``python -m ao_shaping.tools.slm.calibration`` 生成)。
``analyze``
    对已保存的扫描 CSV 重新拟合 (离线, 不碰硬件)。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import click
import numpy as np
from loguru import logger

from ao_shaping.tools.slm.calibration.diffraction_efficiency.core import (
    BlazeSpec,
    SweepPoint,
    blaze_gray,
    diagnose,
    evaluate_fwhm_gate,
    fit_phase_response,
    run_sweep,
)
from ao_shaping.utils.cli.params import option, with_params
from ao_shaping.utils.io.cli_helpers import parse_tuple

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


@dataclass
class SimulateParams:
    """数字孪生扫描的参数。"""

    out_dir: Annotated[
        Path,
        option("--out-dir", "-o", type=click.Path(file_okay=False, path_type=Path)),
    ] = Path("data/diffraction_efficiency/sim")
    period_px: Annotated[int, option("--period-px")] = 32
    axis: Annotated[str, option("--axis", type=click.Choice(["x", "y"]))] = "x"
    n_steps: Annotated[int, option("--n-steps")] = 21
    mode: Annotated[
        str, option("--mode", type=click.Choice(["gray", "phase"]))
    ] = "gray"
    region: Annotated[int, option("--region")] = 240
    padding: Annotated[int, option("--padding")] = 8
    crop_ccd_px: Annotated[float, option("--crop-ccd-px")] = 1400.0
    pupil_radius_px: Annotated[int, option("--pupil-radius-px")] = 480
    kx: Annotated[float, option("--kx")] = 7546.4
    ky: Annotated[float, option("--ky")] = 7495.9
    rotation_deg: Annotated[float, option("--rotation-deg")] = 0.12
    panel_x_to_row: Annotated[bool, option("--panel-x-to-row/--panel-x-to-col")] = True
    depth_max_rad: Annotated[float, option("--depth-max-rad")] = TWO_PI
    gamma: Annotated[float, option("--gamma")] = 1.0
    defocus_coeff: Annotated[float, option("--defocus-coeff")] = 0.0
    peak_adu: Annotated[float, option("--peak-adu")] = 18000.0
    read_noise_adu: Annotated[float, option("--read-noise-adu")] = 25.0
    seed: Annotated[int, option("--seed")] = 0
    reference_fwhm_px: Annotated[
        float | None, option("--reference-fwhm-px", type=float)
    ] = None
    fwhm_gate_ratio: Annotated[float, option("--fwhm-gate-ratio")] = 3.0


@click.command()
@with_params(SimulateParams, kw_name="params")
def simulate(params: SimulateParams) -> None:
    """用数字孪生跑一遍完整测量链路。"""
    from ao_shaping.tools.slm.calibration.diffraction_efficiency.artifacts import (
        plot_efficiency,
        write_measurements_csv,
        write_summary_json,
    )
    from ao_shaping.tools.slm.calibration.diffraction_efficiency.simulate import (
        SimConfig,
        SimulatedBench,
    )

    config = SimConfig(
        region=int(params.region),
        padding=int(params.padding),
        crop_ccd_px=float(params.crop_ccd_px),
        pupil_radius_px=int(params.pupil_radius_px),
        kx=float(params.kx),
        ky=float(params.ky),
        rotation_deg=float(params.rotation_deg),
        panel_x_to_row=bool(params.panel_x_to_row),
        depth_max_rad=float(params.depth_max_rad),
        gamma=float(params.gamma),
        defocus_coeff=float(params.defocus_coeff),
        peak_adu=float(params.peak_adu),
        read_noise_adu=float(params.read_noise_adu),
        seed=int(params.seed),
    )
    bench = SimulatedBench(config)
    gray_levels = int(config.gray_levels)
    if params.mode == "gray":
        commands = np.linspace(0.0, float(gray_levels), int(params.n_steps))
        command_max = float(gray_levels)
    else:
        commands = np.linspace(0.0, TWO_PI, int(params.n_steps))
        command_max = TWO_PI

    def pattern_for(command: float) -> np.ndarray:
        depth = (
            TWO_PI * command / gray_levels if params.mode == "gray" else float(command)
        )
        spec = BlazeSpec(
            period_px=int(params.period_px),
            axis=str(params.axis),
            depth_rad=float(depth),
            pupil_center=(0, 0),
            pupil_radius=0,
        )
        return blaze_gray(spec, bench.panel_shape, gray_levels)

    frame_center = (bench.frame_shape[0] / 2.0, bench.frame_shape[1] / 2.0)
    separation = (config.kx if params.axis == "x" else config.ky) / float(
        params.period_px
    )

    def on_point(index: int, total: int, point: SweepPoint) -> None:
        logger.info(
            "[{}/{}] 命令={:.4g} η±1={:.4f} 间距={}",
            index,
            total,
            point.command,
            point.eta_plus_total if point.eta_plus_total is not None else float("nan"),
            point.separation_px,
        )

    # 仿真后端不需要真的"下发", 但 display/capture 必须共用同一份灰度 ——
    # core.run_sweep 的协议是 pattern_for → display → capture。
    issued: list[np.ndarray] = [np.zeros((1, 1), dtype=np.uint16)]

    def display(gray: np.ndarray) -> None:
        issued[0] = gray

    def capture() -> np.ndarray:
        return bench.capture(issued[0])

    points = run_sweep(
        pattern_for,
        display,
        capture,
        commands,
        expected_center=frame_center,
        predicted_separation_px=separation,
        on_point=on_point,
    )
    # 光斑宽度门禁: 相对在焦参照值 (默认取平相位那一点) 的倍数
    gate = evaluate_fwhm_gate(
        points,
        reference_fwhm_px=params.reference_fwhm_px,
        gate_ratio=float(params.fwhm_gate_ratio),
    )
    logger.info(
        "光斑宽度门禁: 参照 FWHM={:.1f}px 门限={:.1f}px ({}×) 越限 {}/{} 个点",
        gate.reference_fwhm_px,
        gate.limit_px,
        gate.gate_ratio,
        len(gate.violations),
        gate.evaluated_points,
    )

    try:
        fit = fit_phase_response(
            [p.command for p in points],
            [
                p.eta_plus_total if p.eta_plus_total is not None else float("nan")
                for p in points
            ],
            command_max=command_max,
            period_px=int(params.period_px),
            gate=gate,
        )
    except ValueError as exc:
        raise click.ClickException(f"拟合失败: {exc}") from exc
    diagnosis = diagnose(
        points,
        fit,
        full_scale=config.full_scale,
        gate=gate,
    )
    summary = {
        "mode": "simulate",
        "truth": {
            "depth_max_rad": config.depth_max_rad,
            "gamma": config.gamma,
        },
        "fit": {
            "depth_max_rad": fit.depth_max_rad,
            "depth_max_pi": fit.depth_max_pi,
            "gamma": fit.gamma,
            "leakage": fit.leakage,
            "residual_rms": fit.residual_rms,
        },
        "diagnosis": diagnosis,
    }
    out_dir = Path(params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_measurements_csv(out_dir / "measurements.csv", points, fit)
    json_path = write_summary_json(out_dir / "summary.json", summary)
    png_path = plot_efficiency(out_dir / "efficiency.png", points, fit)
    logger.info("产物: {} / {} / {}", csv_path, json_path, png_path)
    for note in diagnosis["notes"]:
        logger.info("诊断: {}", note)
    logger.info(
        "拟合: Δφ_max={:.3f}π (真值 {:.3f}π), γ={:.3f} (真值 {:.3f})",
        fit.depth_max_pi,
        config.depth_max_rad / math.pi,
        fit.gamma,
        config.gamma,
    )


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------


@dataclass
class MeasureParams:
    """硬件采集的参数。"""

    calibration: Annotated[
        Path,
        option(
            "--calibration",
            "-c",
            type=click.Path(dir_okay=False, path_type=Path),
        ),
    ] = Path("data/slm_geometry_calib.npz")
    out_dir: Annotated[
        Path,
        option("--out-dir", "-o", type=click.Path(file_okay=False, path_type=Path)),
    ] = Path("data/diffraction_efficiency/hw")
    slm_number: Annotated[int, option("--slm-number")] = 1
    slm_wavelength: Annotated[int, option("--slm-wavelength")] = 1064
    cam_type: Annotated[
        str, option("--cam-type", type=click.Choice(["daheng", "miicam", "sim"]))
    ] = "daheng"
    cam_id: Annotated[int, option("--cam-id")] = 0
    exposure_ms: Annotated[float, option("--exposure-ms")] = 0.4
    period_px: Annotated[int, option("--period-px")] = 16
    axis: Annotated[str, option("--axis", type=click.Choice(["x", "y"]))] = "x"
    mode: Annotated[
        str, option("--mode", type=click.Choice(["gray", "phase"]))
    ] = "gray"
    n_steps: Annotated[int, option("--n-steps")] = 21
    frames: Annotated[int, option("--frames")] = 4
    discard: Annotated[int, option("--discard")] = 3
    settle_s: Annotated[float, option("--settle-s")] = 0.4
    pupil_center: Annotated[
        tuple[float, float],
        option("--pupil-center", callback=parse_tuple),
    ] = (960.0, 600.0)
    pupil_radius: Annotated[int, option("--pupil-radius")] = 480
    defocus_focal_mm: Annotated[float, option("--defocus-focal-mm")] = float("inf")
    reference_fwhm_px: Annotated[
        float | None, option("--reference-fwhm-px", type=float)
    ] = None
    fwhm_gate_ratio: Annotated[float, option("--fwhm-gate-ratio")] = 3.0
    save_frames: Annotated[bool, option("--save-frames/--no-save-frames")] = False
    strict_geometry: Annotated[
        bool, option("--strict-geometry/--no-strict-geometry")
    ] = True


@click.command()
@with_params(MeasureParams, kw_name="params")
def measure(params: MeasureParams) -> None:
    """在硬件上采集并拟合 (需要先有光路标定)。"""
    from ao_shaping.tools.slm.calibration.diffraction_efficiency.artifacts import (
        plot_efficiency,
        write_measurements_csv,
        write_summary_json,
    )
    from ao_shaping.tools.slm.calibration.diffraction_efficiency.measure import (
        MeasurementConfig,
        run_measurement,
    )

    config = MeasurementConfig(
        calibration_path=Path(params.calibration),
        out_dir=Path(params.out_dir),
        slm_number=int(params.slm_number),
        slm_wavelength=int(params.slm_wavelength),
        cam_type=str(params.cam_type),
        cam_id=int(params.cam_id),
        exposure_ms=float(params.exposure_ms),
        period_px=int(params.period_px),
        axis=str(params.axis),
        mode=str(params.mode),
        n_steps=int(params.n_steps),
        frames=int(params.frames),
        discard=int(params.discard),
        settle_s=float(params.settle_s),
        pupil_center=(int(params.pupil_center[0]), int(params.pupil_center[1])),
        pupil_radius=int(params.pupil_radius),
        defocus_focal_mm=float(params.defocus_focal_mm),
        reference_fwhm_px=params.reference_fwhm_px,
        fwhm_gate_ratio=float(params.fwhm_gate_ratio),
        save_frames=bool(params.save_frames),
        strict_geometry=bool(params.strict_geometry),
    )
    result = run_measurement(config)
    out_dir = Path(params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_measurements_csv(
        out_dir / "measurements.csv", result.points, result.fit
    )
    json_path = write_summary_json(
        out_dir / "summary.json", {"meta": result.meta, "diagnosis": result.diagnosis}
    )
    png_path = plot_efficiency(out_dir / "efficiency.png", result.points, result.fit)
    logger.info("产物: {} / {} / {}", csv_path, json_path, png_path)
    for note in result.diagnosis.get("notes", []):
        logger.info("诊断: {}", note)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@dataclass
class AnalyzeParams:
    """离线重新拟合的参数。"""

    csv_path: Annotated[
        Path,
        option("--csv", type=click.Path(dir_okay=False, path_type=Path)),
    ] = Path("data/diffraction_efficiency/hw/measurements.csv")
    out_dir: Annotated[
        Path,
        option("--out-dir", "-o", type=click.Path(file_okay=False, path_type=Path)),
    ] = Path("data/diffraction_efficiency/analysis")
    command_max: Annotated[float, option("--command-max")] = 1023.0
    period_px: Annotated[int, option("--period-px")] = 16
    reference_fwhm_px: Annotated[
        float | None, option("--reference-fwhm-px", type=float)
    ] = None
    fwhm_gate_ratio: Annotated[float, option("--fwhm-gate-ratio")] = 3.0


@click.command()
@with_params(AnalyzeParams, kw_name="params")
def analyze(params: AnalyzeParams) -> None:
    """对已保存的扫描 CSV 重新拟合。"""
    from ao_shaping.tools.slm.calibration.diffraction_efficiency.artifacts import (
        plot_efficiency,
        write_measurements_csv,
        write_summary_json,
    )

    points = _read_points(Path(params.csv_path))
    gate = evaluate_fwhm_gate(
        points,
        reference_fwhm_px=params.reference_fwhm_px,
        gate_ratio=float(params.fwhm_gate_ratio),
    )
    logger.info(
        "光斑宽度门禁: 参照 FWHM={:.1f}px 门限={:.1f}px ({}×) 越限 {}/{} 个点",
        gate.reference_fwhm_px,
        gate.limit_px,
        gate.gate_ratio,
        len(gate.violations),
        gate.evaluated_points,
    )
    try:
        fit = fit_phase_response(
            [p.command for p in points],
            [
                p.eta_plus_total if p.eta_plus_total is not None else float("nan")
                for p in points
            ],
            command_max=float(params.command_max),
            period_px=int(params.period_px),
            gate=gate,
        )
    except ValueError as exc:
        raise click.ClickException(f"拟合失败: {exc}") from exc
    diagnosis = diagnose(points, fit, gate=gate)
    out_dir = Path(params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_measurements_csv(out_dir / "measurements.csv", points, fit)
    json_path = write_summary_json(
        out_dir / "summary.json",
        {
            "mode": "analyze",
            "source_csv": str(params.csv_path),
            "fit": {
                "depth_max_rad": fit.depth_max_rad,
                "depth_max_pi": fit.depth_max_pi,
                "gamma": fit.gamma,
                "leakage": fit.leakage,
                "residual_rms": fit.residual_rms,
            },
            "diagnosis": diagnosis,
        },
    )
    png_path = plot_efficiency(out_dir / "efficiency.png", points, fit)
    logger.info("产物: {} / {} / {}", csv_path, json_path, png_path)
    for note in diagnosis["notes"]:
        logger.info("诊断: {}", note)


def _read_points(csv_path: Path) -> list[SweepPoint]:
    """从测量 CSV 读回扫描点 (缺失值 → None)。"""
    points: list[SweepPoint] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    for row in csv.DictReader(lines):
        def _f(key: str) -> float | None:
            raw = (row.get(key) or "").strip()
            if not raw:
                return None
            try:
                value = float(raw)
            except ValueError:
                return None
            return None if math.isnan(value) else value

        center = None
        raw_center = (row.get("zero_center") or "").strip()
        if raw_center:
            try:
                cy, cx = (float(v) for v in raw_center.strip("()[]").split(","))
                center = (cy, cx)
            except ValueError:
                center = None
        points.append(
            SweepPoint(
                command=_f("command") or 0.0,
                e_zero=_f("e_zero") or 0.0,
                e_plus=_f("e_plus") or 0.0,
                e_minus=_f("e_minus") or 0.0,
                eta_zero=_f("eta_zero"),
                eta_plus_total=_f("eta_plus_total"),
                separation_px=_f("separation_px"),
                residual_px=_f("residual_px"),
                peak_adu=_f("peak_adu") or 0.0,
                fwhm_px=_f("fwhm_px"),
                ellipticity=_f("ellipticity"),
                zero_center=center,
            )
        )
    if not points:
        raise click.ClickException(f"{csv_path} 里没有可用的扫描点")
    return points


@click.group()
def main() -> None:
    """闪耀光栅衍射效率测量 (2f 实验系统)。"""


main.add_command(simulate)
main.add_command(measure)
main.add_command(analyze)


if __name__ == "__main__":
    main()
