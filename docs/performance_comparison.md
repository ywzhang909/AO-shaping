# Performance Comparison

| Function | NumPy (s) | Numba (s) | CuPy (s) |
|----------|-----------|-----------|----------|
| calculate_sharpness | 3.6313002929091454e-05 | 9.936012793332338e-06 | 0.00040318699087947606 |
| crop | 1.2401989661157131e-05 | 9.032003581523896e-06 | 0.0006128289992921054 |
| center_of_mass | 4.6760996337980035e-05 | 5.391500657424331e-05 | 0.00022413900354877113 |
| center_of_brightness | 3.1459936872124673e-06 | 8.41600587591529e-06 | 0.0002956030098721385 |

### CuPy availability

- No CuPy benchmark returned N/A.
