# Performance Comparison

<!-- provenance:start -->
> **生成脚本**: [`tests/ao_shaping/utils/test_spots_calc.py`](../../tests/ao_shaping/utils/test_spots_calc.py)
> **复现命令**: `python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark`
> **运行环境**: 离线
> **说明**: 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）
<!-- provenance:end -->

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 4.623500164598227e-05 | 2.121700905263424e-05 | 0.0006601929944008588 |
| crop | 1.3748975470662117e-05 | 1.839001663029194e-05 | 0.0007154739741235971 |
| center_of_mass | 5.5474000982940195e-05 | 6.325903348624706e-05 | 0.0002802949957549572 |
| center_of_brightness | 3.5360082983970644e-06 | 9.280997328460217e-06 | 0.00036512698978185653 |

### CuPy availability

- No CuPy benchmark returned N/A.
