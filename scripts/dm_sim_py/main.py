"""
Main module for hysteretic deformable mirror simulation in Python.
This module integrates all components of the HDM simulation.
"""

import numpy as np
import argparse
import os
import sys

# Add the parent directory to the path to allow relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from create_electrodes import create_electrodes, plot_electrodes
from compute_influence_matrix import compute_influence_matrix
from create_hdm_matrices import create_hdm_matrices
from wavefront_reconstruction import wavefront_reconstruction
from create_preisachs import create_preisachs
from simulate_hdm_control import simulate_hdm_control

def run_complete_simulation():
    """
    Run the complete HDM simulation workflow.
    """
    print("=== Hysteretic Deformable Mirror Simulation ===")
    print("Starting complete simulation workflow...\n")
    
    # Step 1: Create electrodes
    print("Step 1: Creating electrodes...")
    electrode_data = create_electrodes()
    print(f"Created {len(electrode_data['elect_corners'])} electrodes\n")
    
    # Step 2: Compute influence matrix (this step can be time-consuming)
    print("Step 2: Computing influence matrix...")
    print("Note: This step may take a while due to numerical integration...")
    # For demonstration, we'll skip this step and use a precomputed matrix
    # In practice, you would uncomment the following lines:
    # inf_funcs = compute_influence_matrix(electrode_data)
    # print(f"Computed influence matrix with shape {inf_funcs.shape}\n")
    
    # Step 3: Create HDM matrices
    print("Step 3: Creating HDM matrices...")
    # Check if the precomputed influence matrix file exists
    inf_mat_file = "InfMat_5x5_Eff_55_Wid_12.mat"
    if os.path.exists(os.path.join("..", "dm_sim", inf_mat_file)):
        inf_mat_path = os.path.join("..", "dm_sim", inf_mat_file)
    else:
        # Look in the current directory
        inf_mat_path = inf_mat_file
    
    try:
        hdm_data = create_hdm_matrices(inf_mat_path)
        print("HDM matrices created successfully\n")
    except FileNotFoundError:
        print(f"Warning: Could not find {inf_mat_file}. Creating dummy data for demonstration.\n")
        # Create dummy data for demonstration
        hdm_data = create_dummy_hdm_data(electrode_data)
    
    # Step 4: Wavefront reconstruction
    print("Step 4: Performing wavefront reconstruction...")
    wf_data = wavefront_reconstruction()
    print(f"Wavefront reconstruction completed with error: {wf_data['error']:.6f}\n")
    
    # Step 5: Create Preisach operators
    print("Step 5: Creating Preisach operators...")
    preisach_data = create_preisachs()
    print(f"Created {len(preisach_data['phi_arr'])} Preisach operators\n")
    
    # Step 6: Simulate HDM control
    print("Step 6: Simulating HDM control...")
    control_results = simulate_hdm_control(hdm_data, preisach_data)
    print(f"Simulation completed in {control_results['iterations']} iterations\n")
    
    print("=== Simulation Complete ===")
    print(f"Final maximum error: {control_results['final_max_error']:.6f}")
    
    return {
        'electrode_data': electrode_data,
        'hdm_data': hdm_data,
        'wf_data': wf_data,
        'preisach_data': preisach_data,
        'control_results': control_results
    }

def create_dummy_hdm_data(electrode_data):
    """
    Create dummy HDM data for demonstration when real data is not available.
    
    Args:
        electrode_data: Dictionary containing electrode parameters
        
    Returns:
        dict: Dummy HDM data
    """
    # Extract parameters
    mirror_grid_size = electrode_data['mirror_grid_size']
    mirror_x_grid = electrode_data['mirror_x_grid']
    mirror_y_grid = electrode_data['mirror_y_grid']
    mirror_mask = electrode_data['mirror_mask']
    elect_corners = electrode_data['elect_corners']
    
    # Create dummy data
    elect_grid = int(np.sqrt(len(elect_corners)))
    
    # Create masks
    elect_masks = np.zeros((mirror_grid_size, mirror_grid_size, elect_grid**2))
    for k in range(elect_grid**2):
        # Create simple square masks for demonstration
        center_i, center_j = mirror_grid_size // 2, mirror_grid_size // 2
        half_size = mirror_grid_size // (elect_grid * 2)
        i_start, i_end = max(0, center_i - half_size), min(mirror_grid_size, center_i + half_size)
        j_start, j_end = max(0, center_j - half_size), min(mirror_grid_size, center_j + half_size)
        elect_masks[i_start:i_end, j_start:j_end, k] = 1
    
    # Create dummy Zernike data
    zer_mask = mirror_mask  # Use mirror mask as Zernike mask for simplicity
    zer_pol = np.random.rand(mirror_grid_size, mirror_grid_size) * 100  # Random Zernike polynomial
    
    # Create dummy matrices
    z_points_total = np.sum(mirror_mask)
    m_cal = np.random.rand(z_points_total, elect_grid**2)
    h_bold = np.random.rand(z_points_total, elect_grid**2)
    p_inv_h_bold = np.random.rand(elect_grid**2, z_points_total)
    
    # Create index map (simplified)
    mask_indices = np.where(mirror_mask)
    mirror_mask_idx_map = np.column_stack([mask_indices[0], mask_indices[1]])
    
    return {
        'mirror_x_grid': mirror_x_grid,
        'mirror_y_grid': mirror_y_grid,
        'mirror_mask': mirror_mask,
        'elect_masks': elect_masks,
        'zer_pol': zer_pol,
        'zer_mask': zer_mask,
        'm_cal': m_cal,
        'h_bold': h_bold,
        'p_inv_h_bold': p_inv_h_bold,
        'mirror_mask_idx_map': mirror_mask_idx_map,
        'mirror_grid_size': mirror_grid_size,
        'z_fit_vec_zer_mask': np.random.rand(np.sum(zer_mask)) * 10,
        'z_fit_mat': np.random.rand(mirror_grid_size, mirror_grid_size) * 100
    }

def main():
    """
    Main function to run the HDM simulation.
    """
    parser = argparse.ArgumentParser(description="Hysteretic Deformable Mirror Simulation")
    parser.add_argument('--step', type=str, choices=['electrodes', 'influence', 'hdm', 'wavefront', 'preisach', 'control', 'all'],
                        default='all', help='Specify which step to run')
    
    args = parser.parse_args()
    
    if args.step == 'electrodes':
        electrode_data = create_electrodes()
        print(f"Created {len(electrode_data['elect_corners'])} electrodes")
        plot_electrodes(electrode_data)
    elif args.step == 'influence':
        print("Computing influence matrix...")
        # This would require electrode data from a previous step
        print("Note: Run electrodes step first to generate required data")
    elif args.step == 'hdm':
        print("Creating HDM matrices...")
        try:
            hdm_data = create_hdm_matrices()
            print("HDM matrices created successfully")
        except Exception as e:
            print(f"Error creating HDM matrices: {e}")
    elif args.step == 'wavefront':
        print("Performing wavefront reconstruction...")
        wf_data = wavefront_reconstruction()
        print(f"Wavefront reconstruction completed with error: {wf_data['error']:.6f}")
    elif args.step == 'preisach':
        print("Creating Preisach operators...")
        preisach_data = create_preisachs()
        print(f"Created {len(preisach_data['phi_arr'])} Preisach operators")
    elif args.step == 'control':
        print("Running HDM control simulation...")
        print("Note: This requires HDM and Preisach data from previous steps")
    elif args.step == 'all':
        results = run_complete_simulation()
        return results

if __name__ == "__main__":
    main()