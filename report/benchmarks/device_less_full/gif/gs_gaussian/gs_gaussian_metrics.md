# Benchmark: gs × gaussian

<!-- provenance:start -->
> **生成脚本**: [`scripts/generate_beam_shaping_benchmark_report.py`](../../../../../scripts/generate_beam_shaping_benchmark_report.py)
> **复现命令**: `python scripts/generate_beam_shaping_benchmark_report.py`
> **运行环境**: 离线
> **说明**: 9 单元网格中 gs x gaussian 的逐 GIF 指标
<!-- provenance:end -->

**本轮结论**：GS 算法在 gaussian 目标上达标：实测面积接近请求值（填充率约 0.90），CV≈0.21，耗时约 0.8 s。

- requested_area: 392
- measured_area: 351  (met: True, fill_ratio: 0.895)
- uniformity_cv: 0.2104
- encircled_energy: 0.4787
- elapsed_s: 0.793
