# Performance Comparison

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 0.00012996500008739532 | 2.0050000166520477e-05 | 0.0010342509986367076 |
| crop | 3.569600055925548e-05 | 1.3811002718284726e-05 | 0.0012373930017929523 |
| center_of_mass | 7.322499877773225e-05 | 0.0001246739993803203 | 0.000538187000202015 |
| center_of_brightness | 9.581999620422721e-06 | 8.17800173535943e-06 | 0.0006315150042064488 |

### CuPy availability

- No CuPy benchmark returned N/A.
