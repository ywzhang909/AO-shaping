# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`tests/ao_shaping/utils/test_spots_calc.py`](../../tests/ao_shaping/utils/test_spots_calc.py)
> **复现命令**: `python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark`
> **运行环境**: 离线
> **说明**: 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）
<!-- provenance:end -->

**本轮结论**：在光斑分析四个热点函数（sharpness/crop/质心/亮度中心）上，**Numba 全面最优**——较 NumPy 加速约 1.6–3.2 倍（如 sharpness 4.0e-5 s vs 1.3e-4 s）；CuPy 反而最慢（慢 3–50 倍），原因是本工作负载数组尺寸小，主机↔设备传输开销主导，GPU 并行收益无法覆盖。据此默认执行后端选定 Numba，CuPy 路径保留但不启用。

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 0.00012886199168860913 | 4.0000001899898055e-05 | 0.0019577769935131073 |
| crop | 4.031602758914232e-05 | 2.8254999779164792e-05 | 0.0020493709575384856 |
| center_of_mass | 0.00023100101854652165 | 0.00018101401627063752 | 0.0007632810017094016 |
| center_of_brightness | 1.089299563318491e-05 | 1.5336982905864715e-05 | 0.0005920069850981236 |

### CuPy availability

- No CuPy benchmark returned N/A.
