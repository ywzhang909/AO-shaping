import numpy as np
import scipy.io as sio
import sys
import os

# Add the parent directory to the path to allow relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mat_utils import MatUtils
from zernike import zernfun

def create_hdm_matrices(inf_mat_filename="InfMat_5x5_Eff_55_Wid_12.mat"):
    """
    Create HDM matrices for deformable mirror simulation.
    
    Args:
        inf_mat_filename: Filename of the influence matrix data
        
    Returns:
        dict: Dictionary containing HDM matrices and related data
    """
    # Load saved data
    mat_data = sio.loadmat(inf_mat_filename)
    
    # Extract variables from MATLAB file
    if 'InfFuncs' in mat_data:
        inf_funcs = mat_data['InfFuncs']
    elif 'inf_funcs' in mat_data:
        inf_funcs = mat_data['inf_funcs']
    else:
        raise ValueError("Could not find influence functions in MAT file")
    
    # Parameters from CreateElectrodes
    max_rad = 1.0
    efective_surf_fact = 0.55
    elect_width = 0.12
    elect_grid = 5
    
    # Rad and surface tension for scaling
    t = 1.0
    mirror_rad = 1.0
    mirror_area = np.pi * mirror_rad**2
    elect_area = (mirror_rad * elect_width)**2
    
    # Influence functions scaled by the relation between radius and surface tension
    scaled_inf_funcs = (mirror_rad**2 / t) * inf_funcs
    
    # ZernikePol parameters
    zer_n = 2
    zer_m = -2
    zen_effect_surf_fact = 1 / 0.60
    zer_amp_fact = 50
    zer_rad_trim = 0.65
    
    # Assuming these variables are available from electrode creation
    # In practice, these would be passed as parameters or loaded from a file
    mirror_grid_size = inf_funcs.shape[0]  # Should be 100
    
    # Create coordinate grids (simplified - in practice would load from electrode data)
    mirror_x_lim = [-max_rad, max_rad]
    mirror_y_lim = [-max_rad, max_rad]
    mirror_x_iso_line = np.linspace(mirror_x_lim[0], mirror_x_lim[1], mirror_grid_size)
    mirror_y_iso_line = np.linspace(mirror_y_lim[0], mirror_y_lim[1], mirror_grid_size)
    mirror_x_grid, mirror_y_grid = np.meshgrid(mirror_x_iso_line, mirror_y_iso_line)
    
    # Creates mirror mask
    mirror_mask = mirror_x_grid**2 + mirror_y_grid**2 <= max_rad**2
    
    # Create electrode corners (simplified recreation)
    elect_corner00 = efective_surf_fact * max_rad / np.sqrt(2)
    elect_space = (efective_surf_fact * max_rad / np.sqrt(2) - elect_width +
                   efective_surf_fact * max_rad / np.sqrt(2)) / (elect_grid - 1)
    
    elect_corners = np.zeros((elect_grid**2, 4))
    for k in range(1, elect_grid + 1):  # These are the rows
        for l in range(1, elect_grid + 1):  # These are the columns
            idx = l + (k - 1) * elect_grid - 1  # Convert to 0-based indexing
            elect_corners[idx, 0] = -elect_corner00 + (l - 1) * elect_space
            elect_corners[idx, 1] = -elect_corner00 + elect_width + (l - 1) * elect_space
            elect_corners[idx, 2] = elect_corner00 - elect_width - (k - 1) * elect_space
            elect_corners[idx, 3] = elect_corner00 - (k - 1) * elect_space
    
    # Creates masks of electrodes
    elect_masks = np.zeros((mirror_grid_size, mirror_grid_size, elect_grid**2))
    
    # Compute Zernike polynomial and its mask
    zer_pol = np.full((mirror_grid_size, mirror_grid_size), np.nan)
    zer_mask = np.zeros((mirror_grid_size, mirror_grid_size))
    
    for j in range(mirror_grid_size):
        for i in range(mirror_grid_size):
            # Create electrode masks
            for k in range(len(elect_corners)):
                if (elect_corners[k, 0] <= mirror_x_grid[i, j] <= elect_corners[k, 1] and
                    elect_corners[k, 2] <= mirror_y_grid[i, j] <= elect_corners[k, 3]):
                    elect_masks[i, j, k] = 1
            
            # Compute Zernike polynomial
            r = (np.sqrt(mirror_x_grid[i, j]**2 + mirror_y_grid[i, j]**2) /
                 (efective_surf_fact * max_rad)) / zen_effect_surf_fact
            
            if mirror_mask[i, j] and r < zer_rad_trim:
                phi = np.arctan2(mirror_y_grid[i, j], mirror_x_grid[i, j])
                try:
                    zer_value = zernfun(zer_n, zer_m, r, phi, norm=True)
                    zer_pol[i, j] = zer_value
                    zer_mask[i, j] = 1
                except Exception:
                    zer_pol[i, j] = 0
                    zer_mask[i, j] = 0
    
    # Plot Zernike polynomial (scaling)
    zer_pol_max = np.nanmax(zer_pol)
    zer_pol_min = np.nanmin(zer_pol)
    zer_pol_amp = zer_pol_max - zer_pol_min
    zer_pol = 2 * (zer_pol / zer_pol_amp) * zer_amp_fact
    
    zer_pol_max = np.nanmax(zer_pol)
    zer_pol_min = np.nanmin(zer_pol)
    zer_pol_amp = zer_pol_max - zer_pol_min
    zer_pol = zer_pol - zer_pol_min
    
    zer_pol_max = np.nanmax(zer_pol)
    zer_pol_min = np.nanmin(zer_pol)
    zer_pol_amp = zer_pol_max - zer_pol_min
    
    # Everything here is computed for whole mirror surface
    mask_to_use = mirror_mask
    z_points_total = np.sum(mask_to_use)
    avg_elect_mask_vec = np.full((z_points_total, elect_grid**2), np.nan)
    elect_mask_vec = np.zeros((z_points_total, elect_grid))
    
    # Initialize arrays
    m_cal = np.full((z_points_total, elect_grid**2), np.nan)
    mirror_mask_idx_map = None
    
    for k in range(elect_grid**2):
        vec, idx_map = MatUtils.matrix_to_vec_idx_map(elect_masks[:, :, k], mask_to_use)
        elect_mask_vec[:, k] = vec
        if k == 0:  # Save the index map from the first electrode
            mirror_mask_idx_map = idx_map
        avg_elect_mask_vec[:, k] = vec / np.sum(vec)
        vec_inf, _ = MatUtils.matrix_to_vec_idx_map(scaled_inf_funcs[:, :, k], mask_to_use)
        m_cal[:, k] = vec_inf
    
    # Creates spring constants matrix K and computes HDM matrix H
    k_spring = 12
    # Calculate average number of electrode mask points
    total_elect_mask_points = np.sum(elect_masks)
    avg_elect_mask_points = total_elect_mask_points / elect_grid**2
    k_diag = np.ones(elect_grid**2) / avg_elect_mask_points
    k_matrix = (k_spring / elect_area) * np.diag(k_diag)
    
    # Compute HDM matrix
    identity_matrix = np.eye(z_points_total)
    h_bold = np.linalg.inv(identity_matrix + m_cal @ k_matrix @ avg_elect_mask_vec.T) @ m_cal
    p_inv_h_bold = np.linalg.inv(h_bold.T @ h_bold) @ h_bold.T
    
    # Everything here is computed using Zernike mask
    mask_to_use = zer_mask > 0  # Convert to boolean mask
    zer_z_points_total = np.sum(mask_to_use)
    zer_avg_elect_mask_vec = np.full((zer_z_points_total, elect_grid**2), np.nan)
    zer_elect_mask_vec = np.zeros((zer_z_points_total, elect_grid))
    zer_m_cal = np.full((zer_z_points_total, elect_grid**2), np.nan)
    
    for k in range(elect_grid**2):
        vec, idx_map = MatUtils.matrix_to_vec_idx_map(elect_masks[:, :, k], mask_to_use)
        zer_elect_mask_vec[:, k] = vec
        if np.sum(vec) > 0:  # Avoid division by zero
            zer_avg_elect_mask_vec[:, k] = vec / np.sum(vec)
        vec_inf, _ = MatUtils.matrix_to_vec_idx_map(scaled_inf_funcs[:, :, k], mask_to_use)
        zer_m_cal[:, k] = vec_inf
    
    # Creates spring constants matrix K and computes HDM matrix H for Zernike mask
    zer_h_bold = np.linalg.inv(np.eye(zer_z_points_total) + zer_m_cal @ k_matrix @ zer_avg_elect_mask_vec.T) @ zer_m_cal
    zer_p_inv_h_bold = np.linalg.inv(zer_h_bold.T @ zer_h_bold) @ zer_h_bold.T
    
    # Estimate best pressure values with MLSE for testing
    zer_pol_vec_zer_mask, _ = MatUtils.matrix_to_vec_idx_map(zer_pol, mask_to_use)
    press_ref = zer_p_inv_h_bold @ zer_pol_vec_zer_mask
    z_fit = h_bold @ press_ref
    
    z_fit_mat = MatUtils.vec_idx_map_to_matrix(z_fit, mirror_mask_idx_map, mirror_grid_size, mirror_grid_size, np.nan)
    z_fit_vec_zer_mask, _ = MatUtils.matrix_to_vec_idx_map(z_fit_mat, mask_to_use)
    
    z_fit_max = np.nanmax(z_fit)
    z_fit_min = np.nanmin(z_fit)
    z_fit_pad = 0.20
    z_fit_amp = z_fit_max - z_fit_min
    z_fit_max_abs = 100  # max(abs([ZFitMin,ZFitMax]));
    
    # Error
    max_zer_z_error = np.nanmax(zer_pol - z_fit_mat)
    
    # Mirror physical dimensions and Young's modulus
    max_z_fit = np.nanmax(z_fit_mat)
    max_press_ref = np.nanmax(press_ref)
    
    return {
        'inf_funcs': inf_funcs,
        'scaled_inf_funcs': scaled_inf_funcs,
        'elect_corners': elect_corners,
        'elect_masks': elect_masks,
        'zer_pol': zer_pol,
        'zer_mask': zer_mask,
        'mirror_mask': mirror_mask,
        'mirror_x_grid': mirror_x_grid,
        'mirror_y_grid': mirror_y_grid,
        'm_cal': m_cal,
        'h_bold': h_bold,
        'p_inv_h_bold': p_inv_h_bold,
        'zer_m_cal': zer_m_cal,
        'zer_h_bold': zer_h_bold,
        'zer_p_inv_h_bold': zer_p_inv_h_bold,
        'press_ref': press_ref,
        'z_fit': z_fit,
        'z_fit_mat': z_fit_mat,
        'z_fit_vec_zer_mask': z_fit_vec_zer_mask,
        'max_z_fit': max_z_fit,
        'max_press_ref': max_press_ref,
        'max_zer_z_error': max_zer_z_error,
        'mirror_mask_idx_map': mirror_mask_idx_map,
        'mirror_grid_size': mirror_grid_size
    }

if __name__ == "__main__":
    print("This module should be imported and used with influence matrix data.")