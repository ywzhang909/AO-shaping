# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`tests/ao_shaping/utils/test_spots_calc.py`](../../tests/ao_shaping/utils/test_spots_calc.py)
> **复现命令**: `python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark`
> **运行环境**: 离线
> **说明**: 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 0.00012362699999357574 | 3.18830000833259e-05 | 0.0014386980000745098 |
| crop | 7.030299999314593e-05 | 3.248199993322487e-05 | 0.001454660000072181 |
| center_of_mass | 0.0001526699999521952 | 0.00012860999995609746 | 0.0005045960000279592 |
| center_of_brightness | 1.3682000098924618e-05 | 1.1967000009462936e-05 | 0.0006868930001292028 |

### CuPy availability

- No CuPy benchmark returned N/A.
