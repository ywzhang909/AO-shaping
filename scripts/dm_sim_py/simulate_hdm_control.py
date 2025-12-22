import numpy as np
import time

def simulate_hdm_control(hdm_data, preisach_data):
    """
    Simulate HDM control for hysteretic deformable mirror.
    
    Args:
        hdm_data: Dictionary containing HDM matrices and related data
        preisach_data: Dictionary containing Preisach operators and coupling matrix
        
    Returns:
        dict: Dictionary containing simulation results
    """
    # Extract data from hdm_data
    mirror_x_grid = hdm_data['mirror_x_grid']
    mirror_y_grid = hdm_data['mirror_y_grid']
    mirror_grid_size = hdm_data['mirror_grid_size']
    mirror_mask_idx_map = hdm_data['mirror_mask_idx_map']
    h_bold = hdm_data['h_bold']
    zer_p_inv_h_bold = hdm_data['zer_p_inv_h_bold']
    z_fit_vec_zer_mask = hdm_data['z_fit_vec_zer_mask']
    zer_mask = hdm_data['zer_mask']
    
    # Extract data from preisach_data
    phi_arr = preisach_data['phi_arr']
    e_coup = preisach_data['e_coup']
    input_min = preisach_data['input_min']
    input_max = preisach_data['input_max']
    young_mod = preisach_data['young_mod']
    
    # Initialize mirror with random remnants
    elect_grid_sq = len(phi_arr)  # Should be 25 for 5x5 grid
    
    # Apply initial inputs
    print("Initializing mirror with random remnants...")
    
    # Function to get outputs from phi array
    def get_phi_arr_outputs(phi_arr):
        phi = np.zeros(len(phi_arr))
        for i in range(len(phi_arr)):
            phi[i] = phi_arr[i].get_output()
        return phi
    
    # Function to apply input
    def apply_input(input_value, mux_ch, e_coup, force_reload=False):
        # Calculate inputs for each operator
        inputs = e_coup[:, mux_ch] * input_value
        
        # Update each operator
        for i in range(len(phi_arr)):
            phi_arr[i].update_relays(inputs[i])
            
        # Compute deflections
        press = young_mod * get_phi_arr_outputs(phi_arr)
        z_vec = h_bold @ press
        z_mat = np.full((mirror_grid_size, mirror_grid_size), np.nan)
        
        # Fill matrix using index map
        for k in range(len(z_vec)):
            i, j = int(mirror_mask_idx_map[k, 0]), int(mirror_mask_idx_map[k, 1])
            z_mat[i, j] = z_vec[k]
            
        return z_vec, z_mat, press
    
    # Function to apply input with animation (simplified)
    def apply_input_anim(input_values, mux_ch, e_coup, force_reload=False):
        z_vec, z_mat, press = None, None, None
        for input_value in input_values:
            z_vec, z_mat, press = apply_input(input_value, mux_ch, e_coup)
        return z_vec, z_mat, press
    
    # Initialize mirror
    _, _, _ = apply_input(input_min, elect_grid_sq, e_coup, True)
    _, _, _ = apply_input(input_max, elect_grid_sq, e_coup, False)
    _, _, _ = apply_input(0, elect_grid_sq, e_coup, True)
    
    # Control parameters
    kappa = 0.130
    samples_per_segment = 2
    elect_inputs = 700 * np.ones(elect_grid_sq)  # Initial value
    
    # Initialize the mirror resetting
    print("Resetting mirror...")
    voltage_input = np.linspace(0, input_max, samples_per_segment)
    _, _, _ = apply_input_anim(voltage_input, elect_grid_sq, e_coup, True)
    
    voltage_input = np.linspace(input_max, 0, samples_per_segment)
    _, _, _ = apply_input_anim(voltage_input, elect_grid_sq, e_coup)
    
    voltage_input = np.linspace(0, -800, samples_per_segment)
    _, _, _ = apply_input_anim(voltage_input, elect_grid_sq, e_coup)
    
    voltage_input = np.linspace(-800, 0, samples_per_segment)
    z_vec, z_mat, _ = apply_input_anim(voltage_input, elect_grid_sq, e_coup)
    
    # Calculate error
    z_vec_zer_mask = np.array([z_mat[int(i), int(j)] 
                              for i, j in mirror_mask_idx_map 
                              if zer_mask[int(i), int(j)] == 1])
    
    z_error = z_fit_vec_zer_mask[:len(z_vec_zer_mask)] - z_vec_zer_mask
    press_error = zer_p_inv_h_bold @ z_error
    
    print("Reset completed")
    print(f"Max Deflection Error Threshold: {np.max(np.abs(z_error)) * 0.01}")  # Placeholder
    print(f"Max Deflection Error: {np.max(np.abs(z_error))}")
    print(f"Max Pressure Error: {np.max(np.abs(press_error))}")
    
    # Control loop
    print("Starting control loop...")
    iter_count = 0
    z_error_max_thres = np.max(np.abs(z_error)) * 0.01  # 1% of current error
    
    while np.max(np.abs(z_error)) > z_error_max_thres and iter_count < 100:  # Max 100 iterations
        iter_count += 1
        print(f"Iteration {iter_count}")
        
        # Update pressure error
        press_error = np.maximum(press_error, 0)  # Set negative values to 0
        elect_inputs = elect_inputs + kappa * press_error
        
        # Sort inputs (simplified)
        sorted_inputs = elect_inputs
        idx = np.arange(len(elect_inputs))
        
        # Apply inputs to electrodes
        for i in range(len(sorted_inputs)):
            voltage_input = np.concatenate([
                np.linspace(0, sorted_inputs[i], samples_per_segment),
                np.linspace(sorted_inputs[i], 0, samples_per_segment)
            ])
            
            _, z_mat, press = apply_input_anim(voltage_input, int(idx[i]), e_coup)
            print(f"  Step {i+1}, Electrode {int(idx[i])+1}, Amplitude: {sorted_inputs[i]:.2f}")
        
        # Recalculate error
        z_vec_zer_mask = np.array([z_mat[int(i), int(j)] 
                                  for i, j in mirror_mask_idx_map 
                                  if zer_mask[int(i), int(j)] == 1])
        
        z_error = z_fit_vec_zer_mask[:len(z_vec_zer_mask)] - z_vec_zer_mask
        press_error = zer_p_inv_h_bold @ z_error
        
        print(f"  Max Deflection Error: {np.max(np.abs(z_error))}")
        print(f"  Max Pressure Error: {np.max(np.abs(press_error))}")
        
        # Update threshold
        z_error_max_thres = np.max(np.abs(z_error)) * 0.01
    
    print(f"Final Max Deflection Error: {np.max(np.abs(z_error))}")
    print(f"Total Iterations: {iter_count}")
    
    # Final error calculation
    z_fit_mat = hdm_data['z_fit_mat']
    final_error_mat = (z_fit_mat - z_mat) * zer_mask
    
    return {
        'final_z_mat': z_mat,
        'final_error_mat': final_error_mat,
        'elect_inputs': elect_inputs,
        'iterations': iter_count,
        'final_max_error': np.max(np.abs(z_error)),
        'mirror_x_grid': mirror_x_grid,
        'mirror_y_grid': mirror_y_grid
    }

if __name__ == "__main__":
    print("This module should be imported and used with HDM and Preisach data.")