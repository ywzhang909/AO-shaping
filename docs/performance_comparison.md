# Performance Comparison

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 4.510637605562806e-05 | 1.3406644575297832e-05 | N/A |
| crop | 1.6939127817749977e-05 | 8.951094932854176e-06 | N/A |
| center_of_mass | 3.810045076534152e-05 | 6.609636358916759e-05 | N/A |
| center_of_brightness | 2.9493751935660838e-06 | 7.400629110634327e-06 | N/A |

### CuPy availability

- `calculate_sharpness`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `crop`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_mass`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_brightness`: CuPy is not installed in the active Python environment (No module named 'cupy')
