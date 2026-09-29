# Performance Comparison

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 4.601048960466869e-05 | 1.3547039998229593e-05 | 0.000298778159703943 |
| crop | 1.67115902149817e-05 | 8.866749776643701e-06 | 0.00039355099994281773 |
| center_of_mass | 3.721394983585924e-05 | 6.581564033695031e-05 | 0.00013000994964386338 |
| center_of_brightness | 2.9124399588909e-06 | 7.415429499815218e-06 | 0.00018519860997912474 |

### CuPy availability

- No CuPy benchmark returned N/A.
