"""SLM 相关独立工具包。

包含:
- phase_capture.py — SLM 相位捕获 (湍流/Zernike 相位数据集采集)
- gray_response.py — SLM 灰度响应测量
- slm_diagnose.py — SLM 硬件自检 (freeze/modulate/linearity 三步证据链)
- slm_phase_response.py — SLM 相位→CCD 响应探针 (离焦/透镜用例 + 共享采集/渲染通道)
- slm_lut_runner.py — SLM 灰度→相位 LUT 校准工具 (CLI 入口)
- cartographer/   — SLM 标定综合工具 (余弦图样/Hartmann 波前重建/灰度-LUT/动态补偿)
"""

from ao_shaping.tools.slm.slm_phase_response import build_sequence, lens_cases, defocus_cases

__all__ = ["build_sequence", "lens_cases", "defocus_cases"]