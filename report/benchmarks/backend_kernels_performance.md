# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`tests/ao_shaping/utils/test_spots_calc.py`](../../tests/ao_shaping/utils/test_spots_calc.py)
> **复现命令**: `python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark`
> **运行环境**: 离线
> **说明**: 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 3.698898246511817e-05 | 9.017996490001678e-06 | 0.0008197829988785088 |
| crop | 1.4135988894850016e-05 | 8.444003760814667e-06 | 0.001194585005287081 |
| center_of_mass | 4.763999255374074e-05 | 5.700001260265708e-05 | 0.0005048149917274713 |
| center_of_brightness | 3.2649911008775236e-06 | 8.091991767287254e-06 | 0.0002815839997492731 |

### CuPy availability

- No CuPy benchmark returned N/A.
