# Performance Comparison

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 5.510999966645613e-05 | 1.4758000033907593e-05 | N/A |
| crop | 1.755199977196753e-05 | 1.1506000882945955e-05 | N/A |
| center_of_mass | 6.67030003387481e-05 | 0.0001001089994679205 | N/A |
| center_of_brightness | 4.104000399820507e-06 | 1.2014000385534018e-05 | N/A |

### CuPy availability

- `calculate_sharpness`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `crop`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_mass`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_brightness`: CuPy is not installed in the active Python environment (No module named 'cupy')
