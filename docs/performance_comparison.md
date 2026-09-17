# Performance Comparison

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 0.00020053799962624908 | 3.616699919803068e-05 | N/A |
| crop | 5.5805000010877846e-05 | 2.095299947541207e-05 | N/A |
| center_of_mass | 0.00015865299967117606 | 8.152700116625056e-05 | N/A |
| center_of_brightness | 1.4515000220853835e-05 | 1.4201000158209354e-05 | N/A |

### CuPy availability

- `calculate_sharpness`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `crop`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_mass`: CuPy is not installed in the active Python environment (No module named 'cupy')
- `center_of_brightness`: CuPy is not installed in the active Python environment (No module named 'cupy')
