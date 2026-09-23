# Performance Comparison

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 3.7780000129714606e-05 | 9.0710015501827e-06 | N/A |
| crop | 1.2744999257847667e-05 | 9.081000462174415e-06 | N/A |
| center_of_mass | 4.372099880129099e-05 | 5.572799942456186e-05 | N/A |
| center_of_brightness | 4.22299955971539e-06 | 7.658000104129315e-06 | N/A |

### CuPy availability

- `calculate_sharpness`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `crop`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_mass`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_brightness`: CuPy is not installed in the active Python environment (No module named 'cupy')
