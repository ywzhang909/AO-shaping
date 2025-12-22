import numpy as np

class PreisachRelayModel:
    """
    Simple implementation of Preisach relay model for hysteretic deformable mirror simulation.
    """
    def __init__(self, input_grid=None, grid_size=10):
        """
        Initialize Preisach relay model.
        
        Args:
            input_grid: Input grid range [min, max]
            grid_size: Size of the grid
        """
        if input_grid is None:
            self.input_grid = np.linspace(-1000, 1000, grid_size)
        else:
            self.input_grid = np.linspace(input_grid[0], input_grid[1], grid_size)
        self.grid_size = grid_size
        self.weight_func = None
        self.offset = 0
        # Initialize relays in off state
        self.relays = np.zeros((grid_size, grid_size))
        
    def reset_relays_off(self):
        """Reset all relays to off state."""
        self.relays = np.zeros((self.grid_size, self.grid_size))
        
    def reset_relays_on(self):
        """Reset all relays to on state."""
        self.relays = np.ones((self.grid_size, self.grid_size))
        
    def update_relays(self, input_value):
        """
        Update relay states based on input value.
        
        Args:
            input_value: Input value to update relays
        """
        # Simple implementation - turn on relays where input exceeds threshold
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                alpha = self.input_grid[i]
                beta = self.input_grid[j]
                # Relay turns on when input > alpha and off when input < beta
                if input_value >= alpha:
                    self.relays[i, j] = 1
                elif input_value <= beta:
                    self.relays[i, j] = 0
                    
    def get_output(self):
        """
        Get output of the Preisach model.
        
        Returns:
            float: Output value
        """
        if self.weight_func is not None:
            # Apply weight function if provided
            return np.sum(self.relays * self.weight_func) + self.offset
        else:
            # Simple sum of relays
            return np.sum(self.relays) + self.offset

def create_preisachs(elect_grid=5, max_press_ref=1000.0):
    """
    Create Preisach operators for hysteretic deformable mirror simulation.
    
    Args:
        elect_grid: Electrode grid size (default 5 for 5x5 grid)
        max_press_ref: Maximum reference pressure
        
    Returns:
        dict: Dictionary containing Preisach operators and coupling matrix
    """
    # Consider electrical coupling factor and parameters of base Preisach
    coupl_fac = 0.35
    
    # Create base Preisach model
    base_phi = PreisachRelayModel()
    input_min = base_phi.input_grid[0]
    input_max = base_phi.input_grid[-1]
    grid_size = base_phi.grid_size
    
    # Young's modulus calculation
    young_mod = max_press_ref / 415.0
    
    # Reset base model relays and update
    base_phi.reset_relays_on()
    base_phi.update_relays(0)
    max_remnant_press = young_mod * base_phi.get_output()
    max_ref_max_remnant_relation = max_press_ref / max_remnant_press if max_remnant_press != 0 else 0
    
    # Creates vector of Preisach operators
    phi_arr = []
    for i in range(elect_grid**2):
        phi = PreisachRelayModel([input_min, input_max], grid_size)
        phi.reset_relays_off()
        phi.weight_func = base_phi.weight_func
        phi.offset = base_phi.offset
        phi_arr.append(phi)
    
    # Creates matrix of coupling factors
    # ECoup matrix has size (ElectGrid^2, ElectGrid^2+1)
    e_coup = np.eye(elect_grid**2, elect_grid**2 + 1)
    for i in range(elect_grid**2):
        for j in range(elect_grid**2):
            if i != j:
                e_coup[i, j] = coupl_fac
    # The last column corresponds to all selected
    e_coup[:, -1] = np.ones(elect_grid**2)
    
    return {
        'phi_arr': phi_arr,
        'e_coup': e_coup,
        'input_min': input_min,
        'input_max': input_max,
        'young_mod': young_mod,
        'max_remnant_press': max_remnant_press,
        'max_ref_max_remnant_relation': max_ref_max_remnant_relation,
        'coupl_fac': coupl_fac,
        'base_phi': base_phi
    }

def get_phi_arr_outputs(phi_arr):
    """
    Get outputs from array of Preisach operators.
    
    Args:
        phi_arr: Array of Preisach operators
        
    Returns:
        numpy.ndarray: Array of outputs
    """
    y_phi = np.zeros(len(phi_arr))
    for i in range(len(phi_arr)):
        y_phi[i] = phi_arr[i].get_output()
    return y_phi

def apply_input(phi_arr, input_value, mux_ch, e_coup):
    """
    Apply input to array of Preisach operators.
    
    Args:
        phi_arr: Array of Preisach operators
        input_value: Input value
        mux_ch: Multiplexer channel
        e_coup: Coupling matrix
        
    Returns:
        tuple: (inputs, outputs)
    """
    # Calculate inputs for each operator
    inputs = e_coup[:, mux_ch] * input_value
    
    # Update each operator
    for i in range(len(phi_arr)):
        phi_arr[i].update_relays(inputs[i])
        
    # Get outputs
    outputs = get_phi_arr_outputs(phi_arr)
    return inputs, outputs

if __name__ == "__main__":
    print("Creating Preisach operators...")
    preisach_data = create_preisachs()
    print(f"Young's modulus: {preisach_data['young_mod']}")
    print(f"Number of Preisach operators: {len(preisach_data['phi_arr'])}")
    print(f"Coupling matrix shape: {preisach_data['e_coup'].shape}")