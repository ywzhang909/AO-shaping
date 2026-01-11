import numpy as np
from scipy.special import factorial
import math
import sys
import os

# Add the parent directory to the path to allow relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def zernfun(n, m, r, theta, norm=False):
    """
    Zernike functions of order N and frequency M on the unit circle.
    
    Args:
        n: Order (positive integer including 0)
        m: Angular frequency (integer from -n to +n in steps of 2)
        r: Radius values between 0 and 1
        theta: Angle values
        norm: If True, return normalized Zernike functions
        
    Returns:
        z: Zernike functions evaluated at (r, theta)
    """
    # Check and prepare the inputs
    if not isinstance(n, (list, tuple, np.ndarray)):
        n = np.array([n])
    else:
        n = np.array(n)
        
    if not isinstance(m, (list, tuple, np.ndarray)):
        m = np.array([m])
    else:
        m = np.array(m)
        
    if len(n) != len(m):
        raise ValueError("N and M must be the same length.")
        
    n = n.flatten()
    m = m.flatten()
    
    if np.any((n - m) % 2 != 0):
        raise ValueError("All N and M must differ by multiples of 2 (including 0).")
        
    if np.any(np.abs(m) > n):
        raise ValueError("Each |M| must be less than or equal to its corresponding N.")
        
    if np.any(r > 1) or np.any(r < 0):
        raise ValueError("All R must be between 0 and 1.")
        
    if not isinstance(r, (list, tuple, np.ndarray)):
        r = np.array([r])
    else:
        r = np.array(r)
        
    if not isinstance(theta, (list, tuple, np.ndarray)):
        theta = np.array([theta])
    else:
        theta = np.array(theta)
        
    r = r.flatten()
    theta = theta.flatten()
    
    if len(r) != len(theta):
        raise ValueError("The number of R- and THETA-values must be equal.")
        
    # Determine the required powers of r
    m_abs = np.abs(m)
    rpowers = []
    for j in range(len(n)):
        rpowers.extend(range(m_abs[j], n[j]+1, 2))
    rpowers = list(set(rpowers))  # Unique powers
    rpowers.sort()
    
    # Pre-compute the values of r raised to the required powers
    if len(rpowers) > 0 and rpowers[0] == 0:
        rpowern = np.column_stack([r**p for p in rpowers[1:]]) if len(rpowers) > 1 else np.empty((len(r), 0))
        rpowern = np.column_stack([np.ones(len(r)), rpowern])
    elif len(rpowers) > 0:
        rpowern = np.column_stack([r**p for p in rpowers])
    else:
        rpowern = np.empty((len(r), 0))
    
    # Compute the values of the polynomials
    z = np.zeros((len(r), len(n)))
    for j in range(len(n)):
        s_vals = np.arange(0, (n[j] - m_abs[j]) // 2 + 1)
        pows = np.arange(n[j], m_abs[j] - 1, -2)
        
        for k in range(len(s_vals)-1, -1, -1):
            s_k = s_vals[k]
            # Calculate coefficient
            numerator = factorial(n[j] - s_k)
            denominator = (factorial(s_k) * 
                          factorial((n[j] - m_abs[j]) // 2 - s_k) * 
                          factorial((n[j] + m_abs[j]) // 2 - s_k))
            p = ((-1)**s_k) * numerator / denominator
            
            # Find the corresponding power index
            pow_val = pows[k]
            idx = rpowers.index(pow_val) if pow_val in rpowers else -1
            
            if idx >= 0:
                z[:, j] += p * rpowern[:, idx]
                
        # Apply normalization if requested
        if norm:
            kronecker_delta = 1 if m[j] == 0 else 0
            norm_factor = np.sqrt((2 - kronecker_delta) * (n[j] + 1) / np.pi)
            z[:, j] *= norm_factor
    
    # Compute the Zernike functions
    idx_pos = m > 0
    idx_neg = m < 0
    
    if np.any(idx_pos):
        pos_indices = np.where(idx_pos)[0]
        for idx in pos_indices:
            z[:, idx] *= np.cos(theta * m_abs[idx])
            
    if np.any(idx_neg):
        neg_indices = np.where(idx_neg)[0]
        for idx in neg_indices:
            z[:, idx] *= np.sin(theta * m_abs[idx])
            
    # Return single column if only one (n,m) pair
    if z.shape[1] == 1:
        return z[:, 0]
    else:
        return z