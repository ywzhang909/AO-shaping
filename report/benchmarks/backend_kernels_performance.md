# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`tests/ao_shaping/utils/test_spots_calc.py`](../../tests/ao_shaping/utils/test_spots_calc.py)
> **复现命令**: `python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark`
> **运行环境**: 离线
> **说明**: 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 0.00011679301504045725 | 2.4279009085148574e-05 | 0.0010259169922210275 |
| crop | 3.473700024187565e-05 | 1.3828009832650423e-05 | 0.0011188630131073296 |
| center_of_mass | 9.570099180564284e-05 | 0.00010323000373318791 | 0.0004789309902116656 |
| center_of_brightness | 7.282968144863844e-06 | 8.146986365318299e-06 | 0.0005686760065145791 |

### CuPy availability

- No CuPy benchmark returned N/A.
