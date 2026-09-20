# Performance Comparison

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 7.081599906086922e-05 | 1.7454999033361672e-05 | N/A |
| crop | 1.987899886444211e-05 | 2.3668000940233468e-05 | N/A |
| center_of_mass | 0.00012590199941769242 | 9.96400008443743e-05 | N/A |
| center_of_brightness | 5.209998344071209e-06 | 1.0569001315161586e-05 | N/A |

### CuPy availability

- `calculate_sharpness`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `crop`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_mass`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_brightness`: CuPy is not installed in the active Python environment (No module named 'cupy')
