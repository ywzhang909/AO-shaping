# Performance Comparison

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 0.0001144760032184422 | 2.2180002415552734e-05 | N/A |
| crop | 3.4169000573456285e-05 | 1.7508002929389477e-05 | N/A |
| center_of_mass | 0.00010345099726691843 | 0.00011422299896366894 | N/A |
| center_of_brightness | 8.953000651672483e-06 | 8.405003463849425e-06 | N/A |

### CuPy availability

- `calculate_sharpness`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `crop`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_mass`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_brightness`: CuPy is not installed in the active Python environment (No module named 'cupy')
