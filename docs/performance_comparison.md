# Performance Comparison

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 4.5132534578442576e-05 | 1.3433916028589011e-05 | N/A |
| crop | 1.695820828899741e-05 | 8.843892719596625e-06 | N/A |
| center_of_mass | 3.681745380163193e-05 | 6.787726422771812e-05 | N/A |
| center_of_brightness | 3.0338717624545095e-06 | 6.720346864312887e-06 | N/A |

### CuPy availability

- `calculate_sharpness`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `crop`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_mass`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_brightness`: CuPy is not installed in the active Python environment (No module named 'cupy')
