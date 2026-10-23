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
    在该灰度下的**真实**相位深度。菲涅尔离焦补偿在这条路径上由
    :func:`core.blaze_gray` 的 ``extra_phase_rad`` 叠加 (弧度量先加成、再经
    :func:`core.phase_to_gray` 按 ``2π = max_gray`` 包裹成灰度), 因此本模式
    **也能**在离焦的相机上测量 —— 但补偿项同样落在器件非线性的灰度→相位上,
    实际成焦效果由前置的焦面扫描自己验证 (扫不出够窄的光斑就拒绝测量)。

``"phase"`` —— **命令弧度链路核对**:
    图案以**弧度**生成 (raw 未包裹), 再经
    :func:`ao_shaping.utils.slm.phase_display.phase_to_slm_grayscale` 交给驱动
    管线 (矫正 / LUT / 平移 / mod 2π)。测的是"命令弧度 → 器件实际相位"这条
    链路, 不是器件本征。

先判在焦, 再测量
----------------
本实验系统上的 CCD 并**不**在焦面上, 离焦会把拟合出的相位深度抬高约 11%
(见 ``report/slm/diffraction_efficiency/sim_nonideality_20261022.md``), 而在
一次全量扫描里"事后"才发现离焦等于白测一轮。所以 ``focus_gate=True`` 时:

1. 先跑 :func:`ao_shaping.tools.slm.panel.focus_probe.sweep_focus` 找最佳
   离焦补偿 (上下扫 1/f), 写 ``focus_sweep.csv`` / ``focus_sweep.png``;
2. :func:`~ao_shaping.tools.slm.panel.focus_probe.judge_focus` 判最佳点的
   零级 FWHM 是否 ≤ 门限;
3. **不在焦 ⇒ 抛** :class:`FocusRefusedError`, 只留下焦面诊断产物, **一个
   测量点都不写**; 在焦 ⇒ (可选) 自动采用最佳焦距作补偿, 然后开始扫描。

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
    phase_to_gray,
    run_sweep,
)
from ao_shaping.tools.slm.panel.focus_probe import (
    FocusProbeResult,
    FocusVerdict,
    judge_focus,
    sweep_focus,
)
from ao_shaping.utils.image.beam_metrics import prepare_frame
from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

__all__ = [
    "DEFAULT_FOCUS_FWHM_MAX_PX",
    "FocusRefusedError",
    "MeasurementConfig",
    "MeasurementResult",
    "calibration_is_required",
    "run_measurement",
]

#: 零级中心相对光路标定中心的最大允许漂移 (CCD 像素)。超过即认为标定已过期。
GEOMETRY_DRIFT_TOL_PX: float = 40.0

#: 在焦判定的默认 FWHM 门限 (CCD 像素)。本实验系统实测: 在焦零级 FWHM
#: 15.2 px, 未补偿的离焦光斑 71.3 px, 所以 40 px 明确落在两者之间。
#: 出处 ``report/slm/differentiable_opt/focus_gate_verification_20261022.md``。
DEFAULT_FOCUS_FWHM_MAX_PX: float = 40.0

#: 零级峰值占满量程的比例上限 (超过即过曝, 效率会被严重低估)。
SATURATION_WARN_FRACTION: float = 0.8

#: 光路标定里必须存在的键 (缺一即拒绝继续)。
REQUIRED_CALIBRATION_KEYS: tuple[str, ...] = ("center", "Kx", "Ky")


class FocusRefusedError(RuntimeError):
    """前置焦面判定认为**不在焦**, 拒绝测量。

    抛出时只写过 ``focus_sweep.csv`` / ``focus_sweep.png`` 两份诊断产物,
    **没有**任何测量数据落盘 —— 这是刻意的: 离焦的扫描结果看着正常,
    却会把相位深度抬高约 11% (见仿真报告), 宁可空跑一轮也不要留下
    一份会被后人不加怀疑地引用的数据。
    """


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
        focus_gate: 是否在扫描前先做焦面判定 (默认 True)。
        focus_fwhm_max_px: 在焦门限 (零级 FWHM, CCD 像素)。None = 用
            :data:`DEFAULT_FOCUS_FWHM_MAX_PX`。
        focus_diopter_range: 焦面扫描的 1/f 半宽 (m^-1)。
        focus_points: 焦面扫描的粗扫点数 (≥ 3)。
        focus_refine: 找到最佳点后在其邻域内细扫的点数。
        focus_frames / focus_discard: 焦面扫描每点平均帧数 / 丢弃帧数。
        focus_auto_defocus: 判定在焦后, 若 ``defocus_focal_mm`` 是 ``inf``
            则自动采用最佳焦距作补偿。
        focus_only: 只做焦面判定, 不做扫描 (CLI ``--focus-only``)。
        command_min_gray / command_max_gray: 灰度模式下的**原始灰度**扫描区间
            (含两端)。None = ``0`` / 设备读到的 2π 灰度 (``slm._max_gray``)。
            ⚠️ 器件的 2π 灰度**不是**准的: 1064 nm 标称 993, 但真实值随温度/
            波长/LUT 漂移, 而"转移能量最多"的点正是真实 2π 处。所以找最优点
            时必须把上界放到硬件满量程 (``--gray-max 1023``), 不能停在标称值
            上 —— 否则曲线会在标称值处仍在单调上升, 最优点根本没被扫到。
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
    focus_gate: bool = True
    focus_fwhm_max_px: float | None = None
    focus_diopter_range: float = 6.0
    focus_points: int = 25
    focus_refine: int = 7
    focus_frames: int = 3
    focus_discard: int = 2
    focus_auto_defocus: bool = True
    focus_only: bool = False
    command_min_gray: int | None = None
    command_max_gray: int | None = None

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
        if self.focus_fwhm_max_px is not None and float(self.focus_fwhm_max_px) <= 0:
            raise ValueError(
                f"focus_fwhm_max_px 必须 > 0, 得到 {self.focus_fwhm_max_px}"
            )
        if int(self.focus_points) < 3:
            raise ValueError(
                f"focus_points 必须 >= 3 (抛物线顶点至少需要 3 点), 得到 {self.focus_points}"
            )
        if float(self.focus_diopter_range) <= 0:
            raise ValueError(
                f"focus_diopter_range 必须 > 0, 得到 {self.focus_diopter_range}"
            )
        lo = 0 if self.command_min_gray is None else int(self.command_min_gray)
        hi = None if self.command_max_gray is None else int(self.command_max_gray)
        if lo < 0:
            raise ValueError(f"command_min_gray 不能为负, 得到 {lo}")
        if hi is not None and hi <= lo:
            raise ValueError(
                f"command_max_gray ({hi}) 必须大于 command_min_gray ({lo})"
            )


@dataclass(frozen=True)
class MeasurementResult:
    """一次实验扫描的完整结果。

    Attributes:
        points: 扫描点 (逐点命令、能量、效率、几何残差)。``--focus-only`` 时为空。
        fit: 相位响应拟合结果。``--focus-only`` 时为 None (只做了焦面判定)。
        diagnosis: 光路与器件体检结论 (来自 :func:`core.diagnose`)。
        meta: 可 JSON 序列化的元数据 (设备、标定、几何核对、焦面判定)。
    """

    points: list[SweepPoint]
    fit: PhaseResponseFit | None
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


def calibration_is_required(config: MeasurementConfig) -> bool:
    """本次运行是否必须已有光路标定。

    ``--focus-only`` 只做焦面扫描, 不用 ``Kx``/``Ky`` (那是"±1 级落在哪"的
    问题), 所以它**不**被缺失的标定挡住 —— 否则"先判在焦"在刚重新标定之前
    根本跑不了, 而恰恰那时最需要确认焦面。其余情况一律要求标定。
    """
    if config.focus_only and not Path(config.calibration_path).exists():
        return False
    return True


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
    # ``--focus-only`` 只做焦面扫描, 不用 Kx/Ky (那是"±1 级落在哪"的问题),
    # 所以它**不**该被缺失的标定挡住 —— 否则"先判在焦"这一步在刚重新标定
    # 之前根本跑不了, 而恰恰那时最需要确认焦面。
    if not calibration_is_required(config):
        logger.warning(
            "--focus-only: 跳过光路标定前置条件 (焦面扫描不需要 Kx/Ky): {}",
            config.calibration_path,
        )
        calib: dict[str, Any] = {}
        calib_center = (float("nan"), float("nan"))
        k_scale = float("nan")
        predicted_separation = float("nan")
    else:
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

        slot_rotator = SlotRotator(slm)
        captured: list[np.ndarray] = []

        def phase_to_display(phase_rad: np.ndarray) -> np.ndarray:
            """弧度相位 → 本模式**实际下发**的灰度图。

            前置焦面扫描与正式扫描必须走**同一条**转换: 若前置用驱动管线找
            焦距、正式扫描却用原始灰度写, 两种路径的器件非线性不同, 扫出来
            的"在焦"在正式扫描里并不成立。
            """
            if mode == "gray":
                return phase_to_gray(phase_rad, max_gray)
            return phase_to_slm_grayscale(phase_rad, slm=slm)

        def display_phase(phase_rad: np.ndarray) -> None:
            slot_rotator.display_data(
                slm, phase_to_display(phase_rad), float(config.settle_s)
            )

        # --- 前置: 焦面判定 (先判在焦, 再测量) -----------------------------
        # 本实验系统的 CCD 不在焦面上, 离焦会把拟合深度抬高约 11%, 而"事后"
        # 才发现等于白测一整轮。所以扫描前先上下扫 1/f: 不合格就抛
        # :class:`FocusRefusedError`, 一个测量点都不写。
        defocus_focal_mm = float(config.defocus_focal_mm)
        focus_info: dict[str, Any] = {
            "enabled": bool(config.focus_gate),
            "fwhm_max_px": float(
                config.focus_fwhm_max_px or DEFAULT_FOCUS_FWHM_MAX_PX
            ),
        }
        if config.focus_gate:
            from ao_shaping.tools.slm.calibration.diffraction_efficiency.artifacts import (
                plot_focus_sweep,
                write_focus_sweep_csv,
            )

            fwhm_max = float(focus_info["fwhm_max_px"])
            logger.info(
                "[焦面判定] 扫描 1/f (粗扫 {} 点, 半宽 {} m⁻¹), 在焦门限 FWHM ≤ {} px",
                config.focus_points,
                config.focus_diopter_range,
                fwhm_max,
            )
            focus_result: FocusProbeResult = sweep_focus(
                apply_phase=display_phase,
                read_frame=lambda: np.asarray(
                    camera.get_numpy_image(n_sample=1, skip_first=False),
                    dtype=np.float64,
                ),
                set_exposure_ms=lambda ms: float(
                    camera.reset_exposure_time(float(ms)) or ms
                ),
                panel_shape=panel_shape,
                radius_px=float(pupil_radius),
                centre_px=(float(pupil_center[0]), float(pupil_center[1])),
                coarse_points=int(config.focus_points),
                diopter_range=float(config.focus_diopter_range),
                refine_points=int(config.focus_refine),
                start_exposure_ms=float(config.exposure_ms),
                frames=int(config.focus_frames),
                discard=int(config.focus_discard),
                wavelength_nm=float(config.slm_wavelength),
            )
            write_focus_sweep_csv(out_dir / "focus_sweep.csv", focus_result.points)
            plot_focus_sweep(
                out_dir / "focus_sweep.png",
                focus_result.points,
                fwhm_max_px=fwhm_max,
                best_diopter_m1=focus_result.best_diopter_m1,
            )
            verdict: FocusVerdict = judge_focus(focus_result, fwhm_max_px=fwhm_max)
            logger.info("[焦面判定] {}", verdict.verdict)

            focus_info.update(
                {
                    "kind": verdict.kind,
                    "verdict": verdict.verdict,
                    "reason": verdict.reason,
                    "best_fwhm_px": verdict.best_fwhm_px,
                    "best_focal_mm": verdict.best_focal_mm,
                    "best_diopter_m1": verdict.best_diopter_m1,
                    "points": len(focus_result.points),
                    "auto_defocus": bool(config.focus_auto_defocus),
                    "configured_defocus_focal_mm": (
                        None if math.isinf(defocus_focal_mm) else defocus_focal_mm
                    ),
                }
            )

            if not verdict.ok:
                raise FocusRefusedError(
                    f"焦面判定未通过: {verdict.verdict} ({verdict.reason}). "
                    f"未写出任何测量数据, 只留下 {out_dir / 'focus_sweep.csv'} / "
                    f"{out_dir / 'focus_sweep.png'} 两份诊断产物。"
                    "请检查: 光路是否在遮挡/失准; --focus-diopter-range 是否覆盖"
                    "真实焦距; --focus-fwhm-max-px 是否定得过严。"
                )

            if (
                config.focus_auto_defocus
                and math.isinf(defocus_focal_mm)
                and verdict.best_focal_mm is not None
                and math.isfinite(float(verdict.best_focal_mm))
            ):
                defocus_focal_mm = float(verdict.best_focal_mm)
                logger.info(
                    "[焦面判定] 采用最佳焦距 {} mm (1/f={:.4f} m⁻¹) 作离焦补偿",
                    defocus_focal_mm,
                    (1e3 / defocus_focal_mm if defocus_focal_mm else 0.0),
                )

            # 焦面扫描会改曝光; 正式扫描必须回到配置值, 否则各点亮度不可比。
            camera.reset_exposure_time(float(config.exposure_ms))
            focus_info["adopted_defocus_focal_mm"] = (
                None if math.isinf(defocus_focal_mm) else defocus_focal_mm
            )

            if config.focus_only:
                logger.info("[焦面判定] --focus-only: 到此为止, 不测效率")
                return MeasurementResult(
                    points=[],
                    fit=None,
                    diagnosis={},
                    meta={"focus_gate": focus_info, "out_dir": str(out_dir)},
                )

        # 离焦补偿在两种模式下都生效: phase 模式叠在弧度上走驱动管线;
        # gray 模式由 :func:`core.blaze_gray` 的 ``extra_phase_rad`` 叠在斜坡上,
        # 再经 :func:`core.phase_to_gray` 按 2π = max_gray 包裹成灰度。
        use_defocus = math.isfinite(defocus_focal_mm)
        defocus_phase = (
            fresnel_panel(
                defocus_focal_mm,
                radius=pupil_radius,
                pupil_center=pupil_center,
                panel_shape=panel_shape,
                wavelength_nm=float(config.slm_wavelength),
            )
            if use_defocus
            else np.zeros(panel_shape, dtype=np.float64)
        )

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
                return blaze_gray(
                    spec,
                    panel_shape,
                    max_gray,
                    extra_phase_rad=defocus_phase if use_defocus else None,
                )
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
            """平场 (零相位 + 离焦补偿) 的灰度图, 走与扫描**同一条**下发路径。

            ⚠️ 必须带上 ``defocus_phase``: 平场是坐标与门禁的基准 ——
            零级中心用它定位、FWHM 门禁拿它的宽度当"本轮在焦参照"。若这里
            漏掉补偿, 而扫描图案带补偿 (2026-10-23 实测): 零级中心偏 51 px
            (> 40 px 漂移容差, 直接报"标定过期"), 参照宽度变成未补偿的
            70 px 糊斑, 3 倍门限 (210 px) 等于失效。
            """
            return phase_to_display(defocus_phase)

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
        # 扫描区间的上界默认取设备读到的 2π 灰度, 但**可以也鼓励**推到硬件满量程:
        # 真实 2π 灰度不等于标称值, 停在标称值会切掉最优点 (2026-10-23 实测:
        # 到 993 时 η₊₁ 仍单调上升)。
        scan_lo = int(config.command_min_gray or 0)
        scan_hi = int(config.command_max_gray or max_gray)
        if scan_hi > int(slm.MAX_GRAYSCALE_VALUE):
            raise RuntimeError(
                f"command_max_gray={scan_hi} 超过硬件满量程 "
                f"{int(slm.MAX_GRAYSCALE_VALUE)}"
            )
        if mode == "gray":
            logger.info(
                "灰度扫描区间: {} ~ {} (标称 2π 灰度 {}, 硬件满量程 {})",
                scan_lo,
                scan_hi,
                max_gray,
                int(slm.MAX_GRAYSCALE_VALUE),
            )
        commands = (
            np.linspace(float(scan_lo), float(scan_hi), int(config.n_steps))
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
    command_max = float(scan_hi) if mode == "gray" else 2.0 * math.pi
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
        "command_min_gray": int(scan_lo),
        "command_max_gray": int(scan_hi),
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
            None if math.isinf(defocus_focal_mm) else float(defocus_focal_mm)
        ),
        "defocus_compensated": bool(use_defocus),
        "focus_gate": focus_info,
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
