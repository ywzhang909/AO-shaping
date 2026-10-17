# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`tests/ao_shaping/utils/test_spots_calc.py`](../../tests/ao_shaping/utils/test_spots_calc.py)
> **复现命令**: `python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark`
> **运行环境**: 离线
> **说明**: 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 3.5427999828243626e-05 | 8.817999696475453e-06 | 0.0010339809996366967 |
| crop | 1.1950000262004324e-05 | 8.445000275969505e-06 | 0.0012633769999229117 |
| center_of_mass | 4.4127999790362085e-05 | 5.2919999943696895e-05 | 0.00036234400002285836 |
| center_of_brightness | 3.130999830318615e-06 | 7.76099957874976e-06 | 0.00024648599988722706 |

### CuPy availability

- No CuPy benchmark returned N/A.
