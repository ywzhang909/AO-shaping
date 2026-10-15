# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`tests/ao_shaping/utils/test_spots_calc.py`](../../tests/ao_shaping/utils/test_spots_calc.py)
> **复现命令**: `python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark`
> **运行环境**: 离线
> **说明**: 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 3.5793999559246e-05 | 1.828699925681576e-05 | 0.0014877309987787158 |
| crop | 1.2060000299243257e-05 | 8.770000131335109e-06 | 0.008160628999467007 |
| center_of_mass | 4.7064999962458386e-05 | 6.000899971695617e-05 | 0.0023420710004575084 |
| center_of_brightness | 9.518999431747944e-06 | 7.893999718362466e-06 | 0.0022713920002570376 |

### CuPy availability

- No CuPy benchmark returned N/A.
