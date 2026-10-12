# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`tests/ao_shaping/utils/test_spots_calc.py`](../../tests/ao_shaping/utils/test_spots_calc.py)
> **复现命令**: `python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark`
> **运行环境**: 离线
> **说明**: 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 4.231903702020645e-05 | 1.0521980002522468e-05 | 0.00044033802580088375 |
| crop | 1.4171008951961993e-05 | 9.794975630939006e-06 | 0.0007823950005695224 |
| center_of_mass | 5.78219722956419e-05 | 6.765399593859911e-05 | 0.00033302900847047566 |
| center_of_brightness | 3.5729911178350447e-06 | 9.134006686508656e-06 | 0.00039613496512174606 |

### CuPy availability

- No CuPy benchmark returned N/A.
