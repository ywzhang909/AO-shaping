# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`scripts/generate_cython_optimizer_report.py`](../../scripts/generate_cython_optimizer_report.py)
> **复现命令**: `python scripts/generate_cython_optimizer_report.py`
> **运行环境**: 离线
> **说明**: 读 src/calculators/benchmark_results.json；同目录两张 speedup 图由本脚本生成
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 0.00012024701340124011 | 2.276100218296051e-05 | 0.003312459997832775 |
| crop | 1.2308000586926937e-05 | 1.3840014580637217e-05 | 0.013903346010483802 |
| center_of_mass | 0.00010082300053909421 | 0.00010905099799856543 | 0.004794099987484515 |
| center_of_brightness | 9.294985793530942e-06 | 8.150001522153615e-06 | 0.004776346010621637 |

### CuPy availability

- No CuPy benchmark returned N/A.
