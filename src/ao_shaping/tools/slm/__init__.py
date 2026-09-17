"""SLM 相关独立工具包。

包含:
- phase_capture.py — SLM 相位捕获 (湍流/Zernike 相位数据集采集)
- gray_response.py — SLM 灰度响应测量
- slm_diagnose.py — SLM 硬件自检 (freeze/modulate/linearity 三步证据链)
- slm_phase_response.py — SLM 相位→CCD 响应探针 (离焦/透镜用例 + 共享采集/渲染通道)
- slm_lut_runner.py — SLM 灰度→相位 LUT 校准工具 (CLI 入口, canonical LUT)
- calibration.py — SLM+CCD 装配/光束位置/几何标定 (含已废弃的 SLMLUTCalibrator)
- slm_scan_analysis.py — SLM 扫描数据分析共享助手 (纯 numpy/stdlib, 无硬件依赖)
- cartographer/   — SLM 标定综合工具 (余弦图样/Hartmann 波前重建/灰度-LUT/动态补偿)

LUT 路径约定:
- canonical (驱动可消费):  slm_lut_runner.py + utils/slm_lut.py → lut_forward.csv/
  lut_inverse.csv → Santec.load_lut(). 唯一进入 SLM 灰度↔相位补偿的 LUT。
- deprecated:  calibration.py::SLMLUTCalibrator (legacy 8-bit 自参考干涉法),
  保留供参考/单测, 已移出 CLI, 直接使用触发 DeprecationWarning。
- research:  cartographer/phase_grayscale_lut.py (WFS 实测研究 LUT, 仅供
  cartographer 闭环, 不可被 Santec.load_lut 消费)。
"""

from ao_shaping.tools.slm.slm_phase_response import build_sequence, lens_cases, defocus_cases
from ao_shaping.tools.slm.slm_scan_analysis import (
    LINEARITY_AMPS,
    analyze_linearity,
    clamp_shift,
    group_raw_scan,
    latest_match,
    outlier_mask,
    parabolic_min,
)

__all__ = [
    "build_sequence",
    "lens_cases",
    "defocus_cases",
    "LINEARITY_AMPS",
    "outlier_mask",
    "clamp_shift",
    "parabolic_min",
    "latest_match",
    "group_raw_scan",
    "analyze_linearity",
]