# Benchmark: gs × square

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/generate_beam_shaping_benchmark_report.py`](../../../../../scripts/slm/generate_beam_shaping_benchmark_report.py)
> **复现命令**: `python scripts/slm/generate_beam_shaping_benchmark_report.py`
> **运行环境**: 离线
> **说明**: 9 单元网格中 gs x square 的逐 GIF 指标
<!-- provenance:end -->

**本轮结论**：GS 算法在 square 目标上达标：填充率约 0.97，CV≈0.17，围阀能量约 0.95，耗时约 0.8 s。

- requested_area: 256
- measured_area: 247  (met: True, fill_ratio: 0.965)
- uniformity_cv: 0.1687
- encircled_energy: 0.9509
- elapsed_s: 0.775
