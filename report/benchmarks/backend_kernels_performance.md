# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`tests/ao_shaping/utils/test_spots_calc.py`](../../tests/ao_shaping/utils/test_spots_calc.py)
> **复现命令**: `python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark`
> **运行环境**: 离线
> **说明**: 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 0.00012258799513801934 | 2.2882989142090083e-05 | 0.0010401599993929266 |
| crop | 3.430400276556611e-05 | 3.206699388101697e-05 | 0.0010911049926653504 |
| center_of_mass | 9.834698867052793e-05 | 0.00011930400738492608 | 0.000495125011075288 |
| center_of_brightness | 9.345989674329758e-06 | 8.477007504552603e-06 | 0.00061670700320974 |

### CuPy availability

- No CuPy benchmark returned N/A.
