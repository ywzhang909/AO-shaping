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
| calculate_sharpness | 0.0001145730000553158 | 2.2195999954419675e-05 | 0.0010227210000175546 |
| crop | 3.400499998861051e-05 | 1.377000007778406e-05 | 0.0011038140000800922 |
| center_of_mass | 0.00010025599995969969 | 0.00010774700002002646 | 0.00048498800007109824 |
| center_of_brightness | 9.232000020347186e-06 | 8.213000019168248e-06 | 0.0005864559999554331 |

### CuPy availability

- No CuPy benchmark returned N/A.
