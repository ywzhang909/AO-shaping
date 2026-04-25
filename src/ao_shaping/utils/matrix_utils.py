"""Matrix utility functions for Zernike response matrix operations.

This module contains pure mathematical operations that don't depend on hardware.
Use this for testing without triggering hardware import chains.
"""

from __future__ import annotations

import numpy as np


def compute_pinv(matrix: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Compute SVD pseudoinverse.

    Args:
        matrix: Input matrix of shape (m, n).
        rcond: Singular value truncation threshold.

    Returns:
        Pseudoinverse matrix of shape (n, m).
    """
    return np.linalg.pinv(matrix, rcond=rcond)


def compute_lstsq(matrix: np.ndarray) -> np.ndarray:
    """Compute least-squares inverse.

    For square matrices, directly computes the inverse.
    For rectangular matrices, computes the minimum-norm solution.

    Args:
        matrix: Input matrix of shape (m, n).

    Returns:
        Least-squares inverse matrix of shape (n, m).
    """
    m, n = matrix.shape

    if m == n:
        return np.linalg.inv(matrix)
    else:
        identity = np.eye(m)
        result = np.zeros((n, m))
        for i in range(m):
            # np.linalg.lstsq returns (solution, residuals, rank, singular_values)
            solution, _, _, _ = np.linalg.lstsq(matrix, identity[:, i], rcond=None)
            result[:, i] = solution
        return result


def calc_n_zernike_terms(n_max: int) -> int:
    """Calculate number of Zernike terms up to order n_max.

    Args:
        n_max: Maximum Zernike order.

    Returns:
        Total number of Zernike terms including piston.
    """
    count = 0
    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):
            count += 1
    return count


def noll_to_index(j: int) -> int:
    """Convert Noll index to array index (0-based)."""
    return j - 1


def index_to_noll(i: int) -> int:
    """Convert array index to Noll index (1-based)."""
    return i + 1