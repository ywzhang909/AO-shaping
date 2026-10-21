# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`scripts/generate_backend_kernels_report.py`](../../../scripts/generate_backend_kernels_report.py)
> **复现命令**: `python scripts/generate_backend_kernels_report.py`
> **运行环境**: 离线
> **说明**: NumPy/Numba/CuPy 逐函数耗时；原先挂在 tests/ 的零断言用例里写回仓库（#162）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 3.5336999862920495e-05 | 9.293000330217183e-06 | 0.001168098999914946 |
| crop | 1.2741999817080797e-05 | 8.870999881764873e-06 | 0.0005907730001490563 |
| center_of_mass | 4.585100017720834e-05 | 2.4431000056210906e-05 | 0.00022512999988975933 |
| center_of_brightness | 2.7789996238425373e-06 | 8.2230000407435e-06 | 0.00026359299969044516 |

### CuPy availability

- No CuPy benchmark returned N/A.
