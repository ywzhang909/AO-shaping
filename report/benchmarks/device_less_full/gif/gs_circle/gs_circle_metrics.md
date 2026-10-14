# 基准测试: gs × circle

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/generate_beam_shaping_benchmark_report.py`](../../../../../scripts/slm/generate_beam_shaping_benchmark_report.py)
> **复现命令**: `python scripts/slm/generate_beam_shaping_benchmark_report.py`
> **运行环境**: 离线
> **说明**: 9 单元网格中 gs x circle 的逐 GIF 指标
<!-- provenance:end -->

**本轮结论**：GS 算法在 circle 目标上达标：实测面积 1016/1160（填充率 0.876），均匀性 CV=0.154，围阀能量 0.952，耗时 1.29 s。

- 请求面积: 1160
- 实测面积: 1016  (达标: True, 填充率: 0.876)
- 均匀性_CV: 0.1536
- 围阀能量: 0.9520
- 耗时_s: 1.293
