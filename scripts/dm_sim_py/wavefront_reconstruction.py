import numpy as np
import sys
import os

# Add the parent directory to the path to allow relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mat_utils import MatUtils
from zernike import zernfun

def create_grid(max_rad=1.0, mirror_grid_size=100):
    """
    Create grid of points for wavefront reconstruction.
    
    Args:
        max_rad: Maximum radius
        mirror_grid_size: Size of the grid
        
    Returns:
        tuple: (mirror_x_iso_line, mirror_y_iso_line, mirror_x_grid, mirror_y_grid)
    """
    mirror_x_lim = [-max_rad, max_rad]
    mirror_y_lim = [-max_rad, max_rad]
    mirror_x_iso_line = np.linspace(mirror_x_lim[0], mirror_x_lim[1], mirror_grid_size)
    mirror_y_iso_line = np.linspace(mirror_y_lim[0], mirror_y_lim[1], mirror_grid_size)
    mirror_x_grid, mirror_y_grid = np.meshgrid(mirror_x_iso_line, mirror_y_iso_line)
    return mirror_x_iso_line, mirror_y_iso_line, mirror_x_grid, mirror_y_grid

def compute_zernikes(x_grid, y_grid, max_zer_level=7):
    """
    Compute Zernike polynomials up to a maximum level.
    
    Args:
        x_grid: X coordinate grid
        y_grid: Y coordinate grid
        max_zer_level: Maximum Zernike level
        
    Returns:
        tuple: (zernike_pol, circ_mask)
    """
    # Create circular mask
    circ_mask = x_grid**2 + y_grid**2 <= 1.0
    
    # Initialize Zernike polynomial array
    zernike_pol = np.full((x_grid.shape[0], x_grid.shape[1], 1), np.nan)
    
    # Calculate number of Zernike polynomials
    total_polynomials = 0
    for zer_n in range(max_zer_level + 1):
        total_polynomials += zer_n + 1
    
    # Initialize full Zernike polynomial array
    zernike_pol = np.full((x_grid.shape[0], x_grid.shape[1], total_polynomials), np.nan)
    
    pol_c = 0
    for zer_n in range(max_zer_level + 1):
        for zer_m in range(-zer_n, zer_n + 1, 2):
            # Create temporary array for this polynomial
            temp_pol = np.full((x_grid.shape[0], x_grid.shape[1]), np.nan)
            
            for j in range(x_grid.shape[1]):
                for i in range(x_grid.shape[0]):
                    if circ_mask[i, j]:
                        r = np.sqrt(x_grid[i, j]**2 + y_grid[i, j]**2)
                        phi = np.arctan2(y_grid[i, j], x_grid[i, j])
                        try:
                            temp_pol[i, j] = zernfun(zer_n, zer_m, r, phi, norm=True)
                        except Exception:
                            temp_pol[i, j] = 0
            
            zernike_pol[:, :, pol_c] = temp_pol
            pol_c += 1
    
    return zernike_pol, circ_mask

def wavefront_reconstruction():
    """
    Perform wavefront reconstruction using Zernike polynomials.
    
    Returns:
        dict: Dictionary containing reconstruction results
    """
    # Parameters
    max_rad = 1.0
    coarse_grid_size = 20
    dense_grid_size = 100
    zer_poly_max_level = 7
    
    # Coarse grid computations
    print("Computing coarse grid...")
    _, _, coarse_x_grid, coarse_y_grid = create_grid(max_rad, coarse_grid_size)
    coarse_zer_poly, circ_mask = compute_zernikes(coarse_x_grid, coarse_y_grid, zer_poly_max_level)
    
    # Stack coarse Zernike polynomials
    coarse_total_zernikes = coarse_zer_poly.shape[2]
    coarse_total_z_points = np.sum(circ_mask)
    coarse_zer_pol_mat = np.zeros((coarse_total_z_points, coarse_total_zernikes))
    coarse_circ_mask_idx_map = None
    
    for k in range(coarse_total_zernikes):
        vec, idx_map = MatUtils.matrix_to_vec_idx_map(coarse_zer_poly[:, :, k], circ_mask)
        coarse_zer_pol_mat[:, k] = vec
        if k == 0:  # Save index map from first polynomial
            coarse_circ_mask_idx_map = idx_map
    
    # Dense grid computations
    print("Computing dense grid...")
    _, _, dense_x_grid, dense_y_grid = create_grid(max_rad, dense_grid_size)
    dense_zer_poly, circ_mask = compute_zernikes(dense_x_grid, dense_y_grid, zer_poly_max_level)
    
    # Stack dense Zernike polynomials
    dense_total_zernikes = dense_zer_poly.shape[2]
    dense_total_z_points = np.sum(circ_mask)
    dense_zer_pol_mat = np.zeros((dense_total_z_points, dense_total_zernikes))
    dense_circ_mask_idx_map = None
    
    for k in range(dense_total_zernikes):
        vec, idx_map = MatUtils.matrix_to_vec_idx_map(dense_zer_poly[:, :, k], circ_mask)
        dense_zer_pol_mat[:, k] = vec
        if k == 0:  # Save index map from first polynomial
            dense_circ_mask_idx_map = idx_map
    
    # Random weights computations with coarse grid
    print("Generating random weights...")
    random_weights = (np.random.rand(coarse_total_zernikes, 1) - 0.5) * 2 * 5
    
    # Generate random vector of deformation with noise
    # Fix: Ensure dimensions match for matrix multiplication
    random_weights_flat = random_weights.flatten()
    coarse_z_rand_vec = (coarse_zer_pol_mat @ random_weights_flat) + \
                       (np.random.rand(coarse_total_z_points) - 0.5) * 2 * 0.075
    coarse_z_rand_vec = coarse_z_rand_vec.reshape(-1, 1)
    
    # Convert to random deformations with noise to matrix
    # Fix: Ensure we only use valid indices
    valid_length = min(len(coarse_z_rand_vec.flatten()), len(coarse_circ_mask_idx_map))
    coarse_z_rand = MatUtils.vec_idx_map_to_matrix(
        coarse_z_rand_vec.flatten()[:valid_length],
        coarse_circ_mask_idx_map[:valid_length],
        coarse_grid_size,
        coarse_grid_size,
        np.nan
    )
    
    # Fitting with coarse grid
    print("Fitting with coarse grid...")
    # Using pseudo-inverse for least squares solution
    ident_weights = np.linalg.pinv(coarse_zer_pol_mat.T @ coarse_zer_pol_mat) @ \
                   coarse_zer_pol_mat.T @ coarse_z_rand_vec
    
    # Plotting of the dense Zernike polynomials identified from coarse grid
    print("Generating identified vector of deformation...")
    dense_z_ident_vec = dense_zer_pol_mat @ ident_weights
    
    # Convert identified deformations to matrix
    # Fix: Ensure we only use valid indices
    valid_length = min(len(dense_z_ident_vec.flatten()), len(dense_circ_mask_idx_map))
    dense_z_ident = MatUtils.vec_idx_map_to_matrix(
        dense_z_ident_vec.flatten()[:valid_length],
        dense_circ_mask_idx_map[:valid_length],
        dense_grid_size,
        dense_grid_size,
        np.nan
    )
    
    # Calculate error
    error = np.sum(np.abs(random_weights - ident_weights)**2)
    
    return {
        'coarse_x_grid': coarse_x_grid,
        'coarse_y_grid': coarse_y_grid,
        'dense_x_grid': dense_x_grid,
        'dense_y_grid': dense_y_grid,
        'coarse_z_rand': coarse_z_rand,
        'dense_z_ident': dense_z_ident,
        'random_weights': random_weights,
        'ident_weights': ident_weights,
        'error': error,
        'coarse_circ_mask_idx_map': coarse_circ_mask_idx_map,
        'dense_circ_mask_idx_map': dense_circ_mask_idx_map
    }

if __name__ == "__main__":
    print("Performing wavefront reconstruction...")
    results = wavefront_reconstruction()
    print(f"Reconstruction error: {results['error']}")