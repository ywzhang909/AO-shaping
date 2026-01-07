import numpy as np
from numba import njit


@njit
def factorial_jit(n):
    res = 1
    for i in range(2, n+1):
        res *= i
    return res

@njit
def zernike_radial(n, m, rho):
    R = np.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        nf = factorial_jit(n - k)
        kf = factorial_jit(k)
        af = factorial_jit((n + abs(m)) // 2 - k)
        bf = factorial_jit((n - abs(m)) // 2 - k)
        c = ((-1)**k * nf) / (kf * af * bf)
        R += c * rho**(n - 2*k)
    return R

@njit
def compute_zernike(n, m, rho, theta):
    R = zernike_radial(n, abs(m), rho)
    if m > 0:
        return R * np.cos(m * theta)
    elif m < 0:
        return R * np.sin(abs(m) * theta)
    else:
        return R

class ZernikeGenerator:
    def __init__(self, zernike_indices, R, Theta, mask):
        self.zernike_indices = zernike_indices
        self.num_zernike = len(zernike_indices)
        self.zernike_modes = self.precompute_zernike_modes(R, Theta, mask)
        self.mask = mask

    def precompute_zernike_modes(self, R, Theta, mask):
        modes = np.zeros((self.num_zernike, R.shape[0], R.shape[1]))
        for i, (n, m) in enumerate(self.zernike_indices):
            Z = compute_zernike(n, m, R, Theta)
            Z *= mask
            modes[i] = Z
        return modes

    def generate_wavefront(self, coefficients):
        return np.tensordot(coefficients, self.zernike_modes, axes=(0, 0))
    
    def fit_wavefront(self, wavefront):
        """
        Fit a wavefront to Zernike coefficients using least squares.
        
        Parameters
        ----------
        wavefront : ndarray
            Input wavefront to fit, should have the same shape as zernike modes.
            
        Returns
        -------
        coefficients : ndarray
            Array of Zernike coefficients that best fit the input wavefront.
        """
        # Flatten the wavefront and zernike modes
        wavefront_flat = wavefront.flatten()
        modes_flat = self.zernike_modes.reshape(self.num_zernike, -1)
        
        # Apply mask to both wavefront and modes
        mask_flat = self.mask.flatten()
        wavefront_masked = wavefront_flat[mask_flat]
        modes_masked = modes_flat[:, mask_flat]
        
        # Solve the least squares problem: A @ coefficients = b
        # Where A is the matrix of zernike modes and b is the wavefront
        coefficients = np.linalg.lstsq(modes_masked.T, wavefront_masked, rcond=None)[0]
        
        return coefficients