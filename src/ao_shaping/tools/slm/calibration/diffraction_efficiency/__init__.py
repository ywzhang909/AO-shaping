"""闪耀光栅衍射效率测量 (2f 实验系统)。

用一维闪耀 (锯齿) 相位光栅测 SLM 的**相位调制深度**与**衍射效率**:
在 2f-Fourier 实验系统里 (SLM 在透镜前焦面, 相机在后焦面), 显示周期为 P
个面板像元的锯齿光栅, 远场出现 0 级与 ±1 级衍射斑; 一级衍射效率随相位深度
呈 ``sinc²`` 曲线, 在 Δφ = 2π 处取 1。联立零级 (Δφ = 2π 处为 0) 可唯一
定出相位深度, 再拟合幂律得到灰度→相位响应。

三层结构
--------
``core``
    纯算法层 (图案生成 / 阶次识别 / 能量积分 / 理论模型 / 拟合 / 诊断)。
    不导入任何设备驱动。
``simulate``
    数字孪生: 参数化的 2f 实验系统 (含灰度非线性、离焦、读出噪声),
    用来在没有硬件时验证整条分析链路。
``measure`` / ``artifacts``
    硬件后端 (Santec SLM + Daheng CCD) 与产物写出 (CSV / JSON / PNG)。

命令行::

    python -m ao_shaping.tools.slm.calibration.diffraction_efficiency simulate
    python -m ao_shaping.tools.slm.calibration.diffraction_efficiency measure
    python -m ao_shaping.tools.slm.calibration.diffraction_efficiency analyze

硬件前置条件
------------
``measure`` **必须**先有光路标定 (``SLMCCDCalibrator`` 保存的 ``.npz``,
含零级中心与 K = CCD 像素/面板像素)::

    python -m ao_shaping.tools.slm.calibration        # 生成光路标定

没有它就无法预测 ±1 级该落在哪里, 也就无法区分"一级衍射"与杂散光斑。

术语
----
中文叙述遵循仓库术语表: 实验系统, 瞳面 (pupil), 加性背景 (杂散光) 项。
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    # core —— 纯算法
    "BlazeSpec",
    "OrderDetectionError",
    "OrderMeasurement",
    "PhaseResponseFit",
    "Spot",
    "SweepPoint",
    "blaze_gray",
    "blaze_phase",
    "build_linearization_lut",
    "diagnose",
    "discrete_order_efficiencies",
    "evaluate_fwhm_gate",
    "fit_phase_response",
    "FwhmGate",
    "measure_orders",
    "model_ratios",
    "negative_order_efficiency",
    "order_efficiencies",
    "run_sweep",
    # simulate —— 数字孪生
    "SimConfig",
    "SimulatedBench",
    # measure —— 硬件后端
    "MeasurementConfig",
    "MeasurementResult",
    "run_measurement",
    # artifacts —— 产物写出
    "plot_efficiency",
    "write_measurements_csv",
    "write_summary_json",
]

#: 公开名 → (子模块, 属性名)。惰性解析, 避免 import 本包就拖进原生 SDK。
_LAZY: dict[str, tuple[str, str]] = {
    name: ("core", name)
    for name in (
        "BlazeSpec",
        "OrderDetectionError",
        "OrderMeasurement",
        "PhaseResponseFit",
        "Spot",
        "SweepPoint",
        "blaze_gray",
        "blaze_phase",
        "build_linearization_lut",
        "diagnose",
        "discrete_order_efficiencies",
        "evaluate_fwhm_gate",
        "fit_phase_response",
        "FwhmGate",
        "measure_orders",
        "model_ratios",
        "negative_order_efficiency",
        "order_efficiencies",
        "run_sweep",
    )
}
_LAZY.update(
    {
        "SimConfig": ("simulate", "SimConfig"),
        "SimulatedBench": ("simulate", "SimulatedBench"),
        "MeasurementConfig": ("measure", "MeasurementConfig"),
        "MeasurementResult": ("measure", "MeasurementResult"),
        "run_measurement": ("measure", "run_measurement"),
        "plot_efficiency": ("artifacts", "plot_efficiency"),
        "write_measurements_csv": ("artifacts", "write_measurements_csv"),
        "write_summary_json": ("artifacts", "write_summary_json"),
    }
)

_PREFIX = "ao_shaping.tools.slm.calibration.diffraction_efficiency"


def __getattr__(name: str) -> Any:
    """惰性解析公开 API (PEP 562)。"""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{_PREFIX}.{target[0]}")
    value = getattr(module, target[1])
    globals()[name] = value  # 缓存, 之后是普通属性查找
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
