# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`tests/ao_shaping/utils/test_spots_calc.py`](../../tests/ao_shaping/utils/test_spots_calc.py)
> **复现命令**: `python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark`
> **运行环境**: 离线
> **说明**: 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 5.9560537338256835e-05 | 1.5469337813556195e-05 | N/A |
| crop | 2.1873400546610354e-05 | 1.4499528333544731e-05 | N/A |
| center_of_mass | 5.667225457727909e-05 | 9.58214420825243e-05 | N/A |
| center_of_brightness | 3.7554465234279634e-06 | 1.5518926084041595e-05 | N/A |

### CuPy availability

- `calculate_sharpness`: No CUDA-capable GPU detected by CuPy
- `crop`: No CUDA-capable GPU detected by CuPy
- `center_of_mass`: No CUDA-capable GPU detected by CuPy
- `center_of_brightness`: No CUDA-capable GPU detected by CuPy
