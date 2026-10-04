# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`tests/ao_shaping/utils/test_spots_calc.py`](../../tests/ao_shaping/utils/test_spots_calc.py)
> **复现命令**: `python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark`
> **运行环境**: 离线
> **说明**: 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 7.845200831070542e-05 | 1.4488990418612957e-05 | 0.0033308279770426453 |
| crop | 4.092301707714796e-05 | 2.8127003461122512e-05 | 0.003319650003686547 |
| center_of_mass | 0.0002276230021379888 | 0.000253975004889071 | 0.0012149239983409643 |
| center_of_brightness | 1.679199980571866e-05 | 2.4785008281469346e-05 | 0.001027245013974607 |

### CuPy availability

- No CuPy benchmark returned N/A.
