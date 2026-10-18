# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`tests/ao_shaping/utils/test_spots_calc.py`](../../tests/ao_shaping/utils/test_spots_calc.py)
> **复现命令**: `python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark`
> **运行环境**: 离线
> **说明**: 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 4.404300038004294e-05 | 1.0659000399755315e-05 | 0.0004488479997962713 |
| crop | 1.49940000846982e-05 | 1.046599994879216e-05 | 0.000792948000744218 |
| center_of_mass | 5.8899999858113005e-05 | 6.558199936989694e-05 | 0.00032243999987258576 |
| center_of_brightness | 3.735999925993383e-06 | 1.0416999721201137e-05 | 0.0003819099991233088 |

### CuPy availability

- No CuPy benchmark returned N/A.
