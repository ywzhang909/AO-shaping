# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`scripts/generate_backend_kernels_report.py`](../../scripts/generate_backend_kernels_report.py)
> **复现命令**: `python scripts/generate_backend_kernels_report.py`
> **运行环境**: 离线
> **说明**: NumPy/Numba/CuPy 逐函数耗时；原先挂在 tests/ 的零断言用例里写回仓库（#162）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 4.404300038004294e-05 | 1.0659000399755315e-05 | 0.0004488479997962713 |
| crop | 1.49940000846982e-05 | 1.046599994879216e-05 | 0.000792948000744218 |
| center_of_mass | 5.8899999858113005e-05 | 6.558199936989694e-05 | 0.00032243999987258576 |
| center_of_brightness | 3.735999925993383e-06 | 1.0416999721201137e-05 | 0.0003819099991233088 |

### CuPy availability

- No CuPy benchmark returned N/A.
