# Performance Comparison

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 3.5972000332549215e-05 | 8.840999798849225e-06 | 0.001433887000894174 |
| crop | 1.1764001101255417e-05 | 8.88100010342896e-06 | 0.0015755519969388842 |
| center_of_mass | 4.9458000576123594e-05 | 5.924499942921102e-05 | 0.000336853003827855 |
| center_of_brightness | 3.0049995984882116e-06 | 7.755999686196446e-06 | 0.00038838199921883645 |

### CuPy availability

- No CuPy benchmark returned N/A.
