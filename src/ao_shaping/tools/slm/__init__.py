"""SLM 相关独立工具包。

包含:
- phase_capture.py — SLM 相位捕获 (湍流/Zernike 相位数据集采集)
- gray_response.py — SLM 灰度响应测量
- cartographer/   — SLM 标定综合工具 (余弦图样/Hartmann 波前重建/灰度-LUT/动态补偿)
- slm_lut_runner.py — SLM 灰度→相位 LUT 校准工具 (CLI 入口)
"""