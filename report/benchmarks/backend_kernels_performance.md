# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`tests/ao_shaping/utils/test_spots_calc.py`](../../tests/ao_shaping/utils/test_spots_calc.py)
> **复现命令**: `python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark`
> **运行环境**: 离线
> **说明**: 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 0.00011879699304699898 | 2.305400092154741e-05 | 0.0015521050291135907 |
| crop | 3.541597165167332e-05 | 1.4417003840208053e-05 | 0.0016910479869693517 |
| center_of_mass | 0.00010340800043195486 | 0.00011621701065450907 | 0.0005596950184553861 |
| center_of_brightness | 9.050993248820305e-06 | 8.207024075090886e-06 | 0.0005906150303781033 |

### CuPy availability

- No CuPy benchmark returned N/A.
