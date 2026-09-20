# Performance Comparison

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 0.00022972899896558374 | 3.953699837438762e-05 | N/A |
| crop | 5.538499797694385e-05 | 1.9473001011647283e-05 | N/A |
| center_of_mass | 0.00014393399818800388 | 0.00018550599925220012 | N/A |
| center_of_brightness | 5.520000704564154e-06 | 1.2557998998090625e-05 | N/A |

### CuPy availability

- `calculate_sharpness`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `crop`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_mass`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_brightness`: CuPy is not installed in the active Python environment (No module named 'cupy')
