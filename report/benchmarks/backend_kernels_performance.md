# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`tests/ao_shaping/utils/test_spots_calc.py`](../../tests/ao_shaping/utils/test_spots_calc.py)
> **复现命令**: `python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark`
> **运行环境**: 离线
> **说明**: 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 6.804794538766146e-05 | 1.5776422806084156e-05 | N/A |
| crop | 2.317264210432768e-05 | 1.5057781711220741e-05 | N/A |
| center_of_mass | 5.6384261697530746e-05 | 9.758416563272476e-05 | N/A |
| center_of_brightness | 4.089153371751308e-06 | 1.5498343855142594e-05 | N/A |

### CuPy availability

- `calculate_sharpness`: No CUDA-capable GPU detected by CuPy
- `crop`: No CUDA-capable GPU detected by CuPy
- `center_of_mass`: No CUDA-capable GPU detected by CuPy
- `center_of_brightness`: No CUDA-capable GPU detected by CuPy
