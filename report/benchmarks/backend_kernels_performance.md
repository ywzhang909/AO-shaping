# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`tests/ao_shaping/utils/test_spots_calc.py`](../../tests/ao_shaping/utils/test_spots_calc.py)
> **复现命令**: `python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark`
> **运行环境**: 离线
> **说明**: 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 6.368364207446575e-05 | 1.543985679745674e-05 | N/A |
| crop | 2.1850704215466976e-05 | 1.435460988432169e-05 | N/A |
| center_of_mass | 5.7972501963377e-05 | 9.531707502901553e-05 | N/A |
| center_of_brightness | 3.7407735362648964e-06 | 1.5247324481606484e-05 | N/A |

### CuPy availability

- `calculate_sharpness`: No CUDA-capable GPU detected by CuPy
- `crop`: No CUDA-capable GPU detected by CuPy
- `center_of_mass`: No CUDA-capable GPU detected by CuPy
- `center_of_brightness`: No CUDA-capable GPU detected by CuPy
