# Benchmark: spgd-sim × circle

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/generate_beam_shaping_benchmark_report.py`](../../../../../scripts/slm/generate_beam_shaping_benchmark_report.py)
> **复现命令**: `python scripts/slm/generate_beam_shaping_benchmark_report.py`
> **运行环境**: 离线
> **说明**: 9 单元网格中 spgd-sim x circle 的逐 GIF 指标；area 已坍缩，勿据此比均匀度
<!-- provenance:end -->

**本轮结论**：spgd-sim 在 circle 目标上失败（已知失效模式，非回归）：实测面积坍缩至约 1 px，CV≈34；围阀能量 0.998 仅反映能量集中于坍缩点，不构成整形成功。

- requested_area: 1160
- measured_area: 1  (met: False, fill_ratio: 0.001)
- uniformity_cv: 34.0382
- encircled_energy: 0.9977
- elapsed_s: 0.819
