# Performance Comparison

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 4.5589220244437454e-05 | 1.3680052943527698e-05 | N/A |
| crop | 1.7009712755680085e-05 | 8.912559133023024e-06 | N/A |
| center_of_mass | 3.778138197958469e-05 | 6.637658691033721e-05 | N/A |
| center_of_brightness | 2.9151327908039092e-06 | 7.322710007429123e-06 | N/A |

### CuPy availability

- `calculate_sharpness`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `crop`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_mass`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_brightness`: CuPy is not installed in the active Python environment (No module named 'cupy')
