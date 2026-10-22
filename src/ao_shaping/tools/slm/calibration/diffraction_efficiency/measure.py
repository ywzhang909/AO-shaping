"""闪耀光栅衍射效率测量的实验后端 (Santec SLM + CCD)。

分层
----
:mod:`ao_shaping.tools.slm.calibration.diffraction_efficiency.core` 是纯算法层
(图案生成 → 阶次识别 → 能量积分 → 效率解算 → 相位响应拟合), 本模块只负责
"把灰度图下到面板、把远场画面取回来", 以及两者之间的实验纪律: 光路标定前置
检查、内存槽轮换、LC 稳定等待、过曝检查。仿真后端与本后端共用 core 的同一套
代码路径 —— 仿真验证的就是实验系统要走的那条路。

两种测量模式
------------
``"gray"`` (默认) —— **器件本征表征**:
    灰度图**直接**写进 SLM 内存, 绕开驱动的 弧度→灰度 管线 (波前矫正图 /
    相位 LUT / 平移 / 底相位)。命令轴是灰度值本身, 拟合出的 Δφ_max 是器件
    在该灰度下的**真实**相位深度。代价: 这条路径上无法叠加菲涅尔离焦补偿
    (补偿是弧度量, 折回灰度必须由驱动完成, 而驱动管线正是本模式要绕开的),
    因此本模式下给了有限焦距会告警并忽略。

``"phase"`` —— **命令弧度链路核对**:
    图案以**弧度**生成 (raw 未包裹), 再经
    :func:`ao_shaping.utils.slm.phase_display.phase_to_slm_grayscale` 交给驱动
    管线 (矫正 / LUT / 平移 / mod 2π)。测的是"命令弧度 → 器件实际相位"这条
    链路, 不是器件本征。

光路标定是前置条件
------------------
零级中心与 Kx/Ky (面板像元 → CCD 像素的放大系数) 来自
:mod:`ao_shaping.tools.slm.calibration` 的几何标定产物 (``.npz``)。没有它就没有
"预期间距", 阶次分配与几何核对都无从谈起, 所以缺键直接失败, 而不是退化成
"先测了再说"。

术语
----
中文叙述遵守仓库术语表: 实验系统, 瞳面, 加性背景 (杂散光) 项。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.tools.kernels.bench_kernels import fresnel_panel, measure_spot
from ao_shaping.tools.slm.calibration.diffraction_efficiency.core import (
    BlazeSpec,
    PhaseResponseFit,
    SweepPoint,
    blaze_gray,
    blaze_phase,
    diagnose,
    evaluate_fwhm_gate,
    fit_phase_response,
    run_sweep,
)
from ao_shaping.utils.image.beam_metrics import prepare_frame
from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

__all__ = [
    "MeasurementConfig",
    "MeasurementResult",
    "run_measurement",
]

#: 零级中心相对光路标定中心的最大允许漂移 (CCD 像素)。超过即认为标定已过期。
GEOMETRY_DRIFT_TOL_PX: float = 40.0

#: 零级峰值占满量程的比例上限 (超过即过曝, 效率会被严重低估)。
SATURATION_WARN_FRACTION: float = 0.8

#: 光路标定里必须存在的键 (缺一即拒绝继续)。
REQUIRED_CALIBRATION_KEYS: tuple[str, ...] = ("center", "Kx", "Ky")


@dataclass(frozen=True)
class MeasurementConfig:
    """实验衍射效率扫描的配置。

    Attributes:
        calibration_path: 光路标定 ``.npz`` 路径 (几何标定
            :class:`~ao_shaping.tools.slm.calibration.calibration.SLMCCDCalibrator`
            的 ``save()`` 产物)。**前置条件**: 缺 ``center``/``Kx``/``Ky`` 即失败。
        out_dir: 输出目录 (仅用于存原始帧; CSV/JSON/PNG 由
            :mod:`ao_shaping.tools.slm.calibration.diffraction_efficiency.artifacts` 写)。
        slm_number: SLM 设备编号 (1-8)。
        slm_wavelength: 工作波长 (nm)。
        cam_type: 相机类型 (``create_camera`` 的注册名: ``daheng`` /
            ``miicam`` / ``sim``)。
        cam_id: 相机设备号。
        exposure_ms: 曝光时间 (ms)。
        period_px: 光栅周期 (面板像元)。
        axis: 光栅方向 (``"x"`` = 相位沿列变化, ``"y"`` = 沿行变化)。
        mode: ``"gray"`` = 器件本征表征 (绕开驱动管线); ``"phase"`` =
            命令弧度链路核对 (走驱动管线)。
        n_steps: 扫描点数 (含两端)。
        frames: 每点平均帧数。
        discard: 每点丢弃帧数 (防相机帧缓存滞后)。
        settle_s: 下发后 LC 稳定等待 (秒)。
        pupil_center: 瞳面中心 (x, y), 面板像元坐标。
        pupil_radius: 瞳面半径 (面板像元); 0 = 不加瞳面掩膜。
        defocus_focal_mm: 菲涅尔离焦补偿焦距 (mm); ``inf`` = 不补偿。
            仅 ``mode="phase"`` 生效 (见模块 docstring)。
        reference_fwhm_px: 在焦参照值 (零级光斑 FWHM, px)。None = 取本次
            扫描里平相位那一点的实测值 (即"本轮开始时的光斑宽度"); 给一个
            绝对值则跨次运行可比 (例如某天在焦时量好的参照)。
        fwhm_gate_ratio: 光斑宽度门禁倍数 (默认 3)。扫描中光斑宽度超过
            参照值这么多倍的点, 其相位深度读数被截断失真 ⇒ 从拟合里剔除。
        save_frames: 是否把每个扫描点的原始帧存成 ``.npz``。
        strict_geometry: 几何标定失配时是否直接失败 (True) 还是只告警。
    """

    calibration_path: Path
    out_dir: Path
    slm_number: int = 1
    slm_wavelength: int = 1064
    cam_type: str = "daheng"
    cam_id: int = 0
    exposure_ms: float = 0.4
    period_px: int = 16
    axis: str = "x"
    mode: str = "gray"
    n_steps: int = 21
    frames: int = 4
    discard: int = 3
    settle_s: float = 0.4
    pupil_center: tuple[int, int] = (960, 600)
    pupil_radius: int = 480
    defocus_focal_mm: float = float("inf")
    reference_fwhm_px: float | None = None
    fwhm_gate_ratio: float = 3.0
    save_frames: bool = False
    strict_geometry: bool = True

    def __post_init__(self) -> None:
        if str(self.mode).strip().lower() not in ("gray", "phase"):
            raise ValueError(
                f"mode 必须是 'gray' 或 'phase', 得到 {self.mode!r}"
            )
        if str(self.axis).strip().lower() not in ("x", "y"):
            raise ValueError(f"axis 必须是 'x' 或 'y', 得到 {self.axis!r}")
        if int(self.period_px) < 2:
            raise ValueError(f"period_px 必须 >= 2, 得到 {self.period_px}")
        if int(self.n_steps) < 3:
            raise ValueError(f"n_steps 必须 >= 3 (拟合至少需要 3 点), 得到 {self.n_steps}")
        if int(self.frames) < 1:
            raise ValueError(f"frames 必须 >= 1, 得到 {self.frames}")
        if int(self.discard) < 0:
            raise ValueError(f"discard 不能为负, 得到 {self.discard}")


@dataclass(frozen=True)
class MeasurementResult:
    """一次实验扫描的完整结果。

    Attributes:
        points: 扫描点 (逐点命令、能量、效率、几何残差)。
        fit: 相位响应拟合结果。
        diagnosis: 光路与器件体检结论 (来自 :func:`core.diagnose`)。
        meta: 可 JSON 序列化的元数据 (设备、标定、几何核对)。
    """

    points: list[SweepPoint]
    fit: PhaseResponseFit
    diagnosis: dict
    meta: dict


# ---------------------------------------------------------------------------
# 光路标定
# ---------------------------------------------------------------------------


def _load_calibration(path: Path) -> dict[str, Any]:
    """读取光路标定 ``.npz``, 校验必需键。

    Args:
        path: 标定文件路径。

    Returns:
        ``{键: 数组}``。

    Raises:
        RuntimeError: 文件不存在或缺少 ``center``/``Kx``/``Ky``。
    """
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(
            f"光路标定文件不存在: {path}。"
            "请先运行 `python -m ao_shaping.tools.slm.calibration` 生成光路标定 (.npz)。"
        )
    with np.load(path, allow_pickle=True) as data:
        calib: dict[str, Any] = {key: np.asarray(data[key]) for key in data.files}
    missing = [key for key in REQUIRED_CALIBRATION_KEYS if key not in calib]
    if missing:
        raise RuntimeError(
            f"光路标定 {path} 缺少必需键 {missing} (需要 center/Kx/Ky)。"
            "请先运行 `python -m ao_shaping.tools.slm.calibration` 重新生成光路标定。"
        )
    return calib


def _calib_center(calib: dict[str, Any]) -> tuple[float, float]:
    """从标定里取零级中心 ``(cy, cx)`` (CCD 像素)。

    几何标定的 ``center`` 来自质心矩, 顺序是 ``(行, 列)``。
    """
    center = np.asarray(calib["center"], dtype=np.float64).ravel()
    if center.size < 2:
        raise RuntimeError(f"光路标定的 center 形状异常: {calib['center']!r}")
    return float(center[0]), float(center[1])


def _k_scale(calib: dict[str, Any], axis: str) -> float:
    """取该光栅方向的放大系数 (CCD 像素 / 面板像元)。"""
    key = "Ky" if str(axis).strip().lower() == "y" else "Kx"
    value = float(np.asarray(calib[key], dtype=np.float64).ravel()[0])
    if value <= 0.0:
        raise RuntimeError(f"光路标定的 {key} 非正 ({value}), 几何标定无效")
    return value


# ---------------------------------------------------------------------------
# 画面读出
# ---------------------------------------------------------------------------


def _locate_zero_order(frame: np.ndarray) -> tuple[float, float, float]:
    """在去背景的画面里用 argmax 定位零级, 同时给出稳健峰值。

    2f 傅里叶光路上零级就是帧的全局最大 (不是画面几何中心), 所以用
    ``argmax`` 而不是质心。但原始帧带对称读出噪声, 直接 argmax 会被热像素
    骗到, 因此先做一次背景扣除 (减中位数并截断非负)。

    Args:
        frame: 相机原始帧。

    Returns:
        ``(cy, cx, peak_adu)``。

    Raises:
        RuntimeError: 画面无有效信号。
    """
    clean = prepare_frame(np.asarray(frame, dtype=np.float64))
    if clean.size == 0 or float(clean.max()) <= 0.0:
        raise RuntimeError("平场画面无有效信号 (检查曝光 / 光路 / 图案是否上屏)")
    index = int(np.argmax(clean))
    cy, cx = divmod(index, clean.shape[1])
    spot = measure_spot(frame)
    return float(cy), float(cx), float(spot.peak)


def _fmt_opt(value: Any, digits: int = 4) -> str:
    """日志用的可选数值格式化 (None → ``n/a``)。"""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_measurement(config: MeasurementConfig) -> MeasurementResult:
    """跑完一整条实验扫描: 读光路标定 → 前置检查 → 扫描 → 拟合 → 诊断。

    Args:
        config: 扫描配置。

    Returns:
        :class:`MeasurementResult`。

    Raises:
        RuntimeError: 光路标定缺失/不完整, 零级漂移超限且
            ``strict_geometry`` 为真, 或画面无有效信号。
    """
    mode = str(config.mode).strip().lower()
    axis = str(config.axis).strip().lower()

    # --- 光路标定 (前置条件) -------------------------------------------
    calib = _load_calibration(config.calibration_path)
    calib_center = _calib_center(calib)
    k_scale = _k_scale(calib, axis)
    predicted_separation = k_scale / float(config.period_px)
    logger.info(
        "光路标定 {}: Kx={:.2f} Ky={:.2f} 中心=({:.1f}, {:.1f}) → 预期间距 {:.2f}px",
        config.calibration_path,
        float(np.asarray(calib["Kx"], dtype=np.float64).ravel()[0]),
        float(np.asarray(calib["Ky"], dtype=np.float64).ravel()[0]),
        calib_center[0],
        calib_center[1],
        predicted_separation,
    )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"

    # 设备驱动在函数内惰性导入: 本模块被 import 时绝不能碰 SDK。
    from ao_shaping.drivers.ccd.common import create_camera, full_scale
    from ao_shaping.drivers.slm.santec import Santec, SlotRotator

    with (
        Santec(
            slm_number=int(config.slm_number),
            video_mode=0,  # 只用内存模式; DVI 模式的 open() 会挂起
            wavelength=int(config.slm_wavelength),
        ) as slm,
        create_camera(
            config.cam_type,
            config.cam_id,
            exposure_time_ms=float(config.exposure_ms),
        ) as camera,
    ):
        # 设备在**当前波长**下的 2π 灰度 (SLM-200 @1064nm = 993)。扫描必须
        # 停在 2π 灰度上: 再往上器件的相位绕回, 曲线非单调, 拟合出的
        # Δφ_max 会被压低成"欠调制"的假象。``MAX_GRAYSCALE_VALUE`` (1023)
        # 是硬件满量程, **不是** 2π 锁定值 (SLM-200 @532nm 是 998)。
        max_gray = int(getattr(slm, "_max_gray", None) or slm.MAX_GRAYSCALE_VALUE)
        device_two_pi_gray = getattr(slm, "_max_gray", None)
        logger.info(
            "SLM #{} 序列号={} 灰度满量程={} (设备当前波长 2π 灰度={}) 模式={}",
            config.slm_number,
            getattr(slm, "_serial_number", None),
            int(slm.MAX_GRAYSCALE_VALUE),
            max_gray,
            mode,
        )
        logger.info(
            "相机 {}#{} 曝光={}ms 周期={}px 方向={} 扫描点数={}",
            config.cam_type,
            config.cam_id,
            config.exposure_ms,
            config.period_px,
            axis,
            config.n_steps,
        )

        panel_shape = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        pupil_center = (int(config.pupil_center[0]), int(config.pupil_center[1]))
        pupil_radius = int(config.pupil_radius)

        # 离焦补偿只在 phase 模式生效 (见模块 docstring): 灰度模式绕开了驱动的
        # 弧度→灰度 管线, 补偿折不回灰度。
        use_defocus = mode == "phase" and math.isfinite(float(config.defocus_focal_mm))
        defocus_phase = (
            fresnel_panel(
                float(config.defocus_focal_mm),
                radius=pupil_radius,
                pupil_center=pupil_center,
                panel_shape=panel_shape,
                wavelength_nm=float(config.slm_wavelength),
            )
            if use_defocus
            else np.zeros(panel_shape, dtype=np.float64)
        )
        if mode == "gray" and not math.isinf(float(config.defocus_focal_mm)):
            logger.warning(
                "mode='gray' 走原始灰度直写路径 (绕开驱动的 弧度→灰度 管线), "
                "因此菲涅尔离焦补偿 (焦距 {} mm) 无法叠加 —— 补偿必须由驱动做, "
                "而驱动管线正是本模式要绕开的。需要补偿请用 mode='phase'。",
                config.defocus_focal_mm,
            )

        slot_rotator = SlotRotator(slm)
        captured: list[np.ndarray] = []

        def pattern_for(command: float) -> np.ndarray:
            """命令值 → 灰度图 (uint16)。

            ``gray`` 模式是**原始灰度直写**: 故意绕开驱动的 弧度→灰度 管线,
            所以测到的是器件本征的 灰度→相位 关系。``phase`` 模式走驱动的
            正常管线, 测的是"命令弧度 → 器件实际相位"的链路。
            """
            if mode == "gray":
                spec = BlazeSpec(
                    period_px=int(config.period_px),
                    axis=axis,
                    depth_rad=2.0 * math.pi * float(command) / float(max_gray),
                    pupil_center=pupil_center,
                    pupil_radius=pupil_radius,
                )
                return blaze_gray(spec, panel_shape, max_gray)
            spec = BlazeSpec(
                period_px=int(config.period_px),
                axis=axis,
                depth_rad=float(command),
                pupil_center=pupil_center,
                pupil_radius=pupil_radius,
            )
            phase = blaze_phase(spec, panel_shape) + defocus_phase
            return phase_to_slm_grayscale(phase, slm=slm)

        def display(gray: np.ndarray) -> None:
            """下发灰度图 (内存槽轮换) 并等待 LC 稳定。

            绝不能连续写同一个槽位: 固件对"正在显示的同一槽位"的
            ``display_memory`` 是 no-op, LCOS 面板不会刷新。
            """
            slot_rotator.display_data(slm, gray, float(config.settle_s))

        def capture() -> np.ndarray:
            """丢帧后取多帧平均。

            这里**不**用 :func:`ao_shaping.tools.kernels.bench_kernels.capture_settled`:
            那个共享核心里含一次"下发", 且下发走 ``slm.create_phase_from_array``
            (弧度→灰度 + 矫正 + LUT), 与本工具的原始灰度路径不兼容, 会在扫描
            中途把面板图案换掉。共享核心里真正可复用的部分 (去尖峰 + 定位光斑)
            由 :func:`bench_kernels.measure_spot` 提供。
            """
            for _ in range(int(config.discard)):
                camera.get_numpy_image(n_sample=1, skip_first=False)
            captured.clear()
            for _ in range(int(config.frames)):
                captured.append(
                    np.asarray(
                        camera.get_numpy_image(n_sample=1, skip_first=False),
                        dtype=np.float64,
                    )
                )
            return np.mean(np.stack(captured, axis=0), axis=0)

        def flat_pattern() -> np.ndarray:
            """平场 (零相位) 的灰度图, 走与扫描同一条下发路径。

            平场**不**经 ``create_phase_from_array``: 灰度模式下是原始灰度
            直写 (全 0), 相位模式下是零弧度走正常路径。
            """
            if mode == "gray":
                return np.zeros(panel_shape, dtype=np.uint16)
            return phase_to_slm_grayscale(
                np.zeros(panel_shape, dtype=np.float64), slm=slm
            )

        # --- 前置检查: 零级位置 + 过曝 -----------------------------------
        display(flat_pattern())
        flat_frame = capture()
        zero_cy, zero_cx, peak_adu = _locate_zero_order(flat_frame)
        detector_full_scale = float(full_scale(flat_frame))
        drift_px = math.hypot(
            zero_cy - calib_center[0], zero_cx - calib_center[1]
        )
        logger.info(
            "前置检查: 零级=({:.0f}, {:.0f}) 标定中心=({:.0f}, {:.0f}) 漂移={:.1f}px "
            "峰值={:.0f} ADU (满量程 {:.0f})",
            zero_cy,
            zero_cx,
            calib_center[0],
            calib_center[1],
            drift_px,
            peak_adu,
            detector_full_scale,
        )
        if drift_px > GEOMETRY_DRIFT_TOL_PX:
            message = (
                f"零级中心与光路标定偏离 {drift_px:.1f}px "
                f"(>{GEOMETRY_DRIFT_TOL_PX:.0f}px): 光路标定已过期 (光路动过), "
                "请重新运行 `python -m ao_shaping.tools.slm.calibration`"
            )
            if config.strict_geometry:
                raise RuntimeError(message)
            logger.warning("{} (strict_geometry=False, 继续)", message)
        if detector_full_scale > 0.0:
            saturation = peak_adu / detector_full_scale
            if saturation > SATURATION_WARN_FRACTION:
                logger.warning(
                    "零级峰值 {:.0f} ADU 达满量程的 {:.0f}% (> {:.0f}%): "
                    "过曝会让衍射效率严重低估, 请降低曝光",
                    peak_adu,
                    saturation * 100.0,
                    SATURATION_WARN_FRACTION * 100.0,
                )

        # --- 扫描 ---------------------------------------------------------
        commands = (
            np.linspace(0.0, float(max_gray), int(config.n_steps))
            if mode == "gray"
            else np.linspace(0.0, 2.0 * math.pi, int(config.n_steps))
        )

        def on_point(index: int, total: int, point: SweepPoint) -> None:
            logger.info(
                "[{}/{}] 命令={:.4g} η0={} η±1={} 间距={}px 残差={}px 峰值={:.0f}ADU",
                index,
                total,
                point.command,
                _fmt_opt(point.eta_zero),
                _fmt_opt(point.eta_plus_total),
                _fmt_opt(point.separation_px),
                _fmt_opt(point.residual_px),
                point.peak_adu,
            )
            if config.save_frames and captured:
                frames_dir.mkdir(parents=True, exist_ok=True)
                stack = np.stack(captured, axis=0)
                np.savez_compressed(
                    frames_dir / f"point_{index:03d}.npz",
                    command=np.float64(point.command),
                    frames=stack,
                    mean=np.mean(stack, axis=0),
                )

        points = run_sweep(
            pattern_for,
            display,
            capture,
            commands,
            expected_center=calib_center,
            predicted_separation_px=predicted_separation,
            on_point=on_point,
        )

    # --- 拟合与诊断 -------------------------------------------------------
    command_max = float(max_gray) if mode == "gray" else 2.0 * math.pi
    # 光斑宽度门禁: 相对**在焦参照值**的倍数。参照值默认取扫描里命令最小
    # 那一点 (平相位) 的零级 FWHM —— 也就是这条光路在本次运行开始时的光斑
    # 宽度; 也允许传入一条已知在焦的光路量出来的绝对参照值 (跨次运行可比)。
    # 越限的点 = 光斑在扫描过程中变糊, 它的相位深度读数被截断失真 ⇒ 从
    # 拟合里剔除, 判定标为不可信。
    gate = evaluate_fwhm_gate(
        points,
        reference_fwhm_px=config.reference_fwhm_px,
        gate_ratio=float(config.fwhm_gate_ratio),
    )
    logger.info(
        "光斑宽度门禁: 参照 FWHM={:.1f}px 门限={:.1f}px ({}×) 越限 {}/{} 个点",
        gate.reference_fwhm_px,
        gate.limit_px,
        gate.gate_ratio,
        len(gate.violations),
        gate.evaluated_points,
    )
    fit = fit_phase_response(
        [point.command for point in points],
        [point.eta_plus_total for point in points],
        command_max=command_max,
        period_px=int(config.period_px),
        gate=gate,
    )
    diagnosis = diagnose(
        points,
        fit,
        full_scale=detector_full_scale,
        predicted_separation_px=predicted_separation,
        gate=gate,
    )
    for note in diagnosis.get("notes", []):
        logger.info("诊断: {}", note)

    residuals = [
        point.residual_px for point in points if point.residual_px is not None
    ]
    max_residual = float(max(residuals)) if residuals else float("nan")
    meta: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "slm_number": int(config.slm_number),
        "slm_serial": getattr(slm, "_serial_number", None),
        "slm_wavelength_nm": int(config.slm_wavelength),
        "camera_type": str(config.cam_type),
        "camera_id": int(config.cam_id),
        "exposure_ms": float(config.exposure_ms),
        "camera_full_scale": float(detector_full_scale),
        "period_px": int(config.period_px),
        "axis": axis,
        "mode": mode,
        "n_steps": int(config.n_steps),
        "frames": int(config.frames),
        "discard": int(config.discard),
        "settle_s": float(config.settle_s),
        "two_pi_gray": int(max_gray),
        "hardware_max_gray": int(slm.MAX_GRAYSCALE_VALUE),
        "device_two_pi_gray": (
            int(device_two_pi_gray) if device_two_pi_gray is not None else None
        ),
        "calibration_path": str(Path(config.calibration_path)),
        "calib_Kx": float(np.asarray(calib["Kx"], dtype=np.float64).ravel()[0]),
        "calib_Ky": float(np.asarray(calib["Ky"], dtype=np.float64).ravel()[0]),
        "calib_center": [float(calib_center[0]), float(calib_center[1])],
        "predicted_separation_px": float(predicted_separation),
        "zero_order_center": [float(zero_cy), float(zero_cx)],
        "zero_order_drift_px": float(drift_px),
        "geometry_residual_px": max_residual,
        "pupil_center": [int(pupil_center[0]), int(pupil_center[1])],
        "pupil_radius": int(pupil_radius),
        "panel_shape": [int(panel_shape[0]), int(panel_shape[1])],
        "defocus_focal_mm": (
            None
            if math.isinf(float(config.defocus_focal_mm))
            else float(config.defocus_focal_mm)
        ),
        "defocus_compensated": bool(use_defocus),
        "fwhm_gate": {
            "reference_fwhm_px": float(gate.reference_fwhm_px),
            "gate_ratio": float(gate.gate_ratio),
            "limit_px": float(gate.limit_px),
            "violations": list(gate.violations),
            "max_ratio": float(gate.max_ratio),
            "evaluated_points": int(gate.evaluated_points),
            "skipped_points": int(gate.skipped_points),
            "configured_reference_fwhm_px": config.reference_fwhm_px,
        },
        "max_peak_adu": float(max(point.peak_adu for point in points)),
        "out_dir": str(out_dir),
        "save_frames": bool(config.save_frames),
    }

    return MeasurementResult(points=points, fit=fit, diagnosis=diagnosis, meta=meta)
