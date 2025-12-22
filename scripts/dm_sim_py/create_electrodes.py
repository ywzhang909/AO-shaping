import numpy as np
import matplotlib.pyplot as plt

def create_electrodes():
    """
    Create electrodes for the deformable mirror simulation.
    
    Returns:
        dict: Dictionary containing electrode parameters and mirror grid data
    """
    # Parameters for electrodes creation
    line_width = 1.5
    max_rad = 1.0
    efective_surf_fact = 0.55
    elect_grid = 5
    elect_width = 0.12
    elect_corner00 = efective_surf_fact * max_rad / np.sqrt(2)
    elect_space = (efective_surf_fact * max_rad / np.sqrt(2) - elect_width +
                   efective_surf_fact * max_rad / np.sqrt(2)) / (elect_grid - 1)
    
    # Array where each row is an electrode in format (x1 x2 y1 y2)
    elect_corners = np.zeros((elect_grid**2, 4))
    for k in range(1, elect_grid + 1):  # These are the rows
        for l in range(1, elect_grid + 1):  # These are the columns
            idx = l + (k - 1) * elect_grid - 1  # Convert to 0-based indexing
            elect_corners[idx, 0] = -elect_corner00 + (l - 1) * elect_space
            elect_corners[idx, 1] = -elect_corner00 + elect_width + (l - 1) * elect_space
            elect_corners[idx, 2] = elect_corner00 - elect_width - (k - 1) * elect_space
            elect_corners[idx, 3] = elect_corner00 - (k - 1) * elect_space
    
    # Parameters for computation and saving of influence functions
    mirror_grid_size = 100
    mirror_x_lim = [-max_rad, max_rad]
    mirror_y_lim = [-max_rad, max_rad]
    mirror_x_iso_line = np.linspace(mirror_x_lim[0], mirror_x_lim[1], mirror_grid_size)
    mirror_y_iso_line = np.linspace(mirror_y_lim[0], mirror_y_lim[1], mirror_grid_size)
    mirror_x_grid, mirror_y_grid = np.meshgrid(mirror_x_iso_line, mirror_y_iso_line)
    
    # Creates mirror mask
    mirror_mask = mirror_x_grid**2 + mirror_y_grid**2 <= max_rad**2
    
    return {
        'line_width': line_width,
        'max_rad': max_rad,
        'efective_surf_fact': efective_surf_fact,
        'elect_grid': elect_grid,
        'elect_width': elect_width,
        'elect_corner00': elect_corner00,
        'elect_space': elect_space,
        'elect_corners': elect_corners,
        'mirror_grid_size': mirror_grid_size,
        'mirror_x_lim': mirror_x_lim,
        'mirror_y_lim': mirror_y_lim,
        'mirror_x_iso_line': mirror_x_iso_line,
        'mirror_y_iso_line': mirror_y_iso_line,
        'mirror_x_grid': mirror_x_grid,
        'mirror_y_grid': mirror_y_grid,
        'mirror_mask': mirror_mask
    }

def plot_electrodes(electrode_data):
    """
    Plot the electrodes and mirror outline.
    
    Args:
        electrode_data: Dictionary containing electrode parameters
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot electrodes
    elect_corners = electrode_data['elect_corners']
    elect_grid = electrode_data['elect_grid']
    line_width = electrode_data['line_width']
    
    for k in range(1, elect_grid + 1):
        for l in range(1, elect_grid + 1):
            idx = l + (k - 1) * elect_grid - 1
            # Plot rectangle edges
            x_coords = [elect_corners[idx, 0], elect_corners[idx, 1], 
                        elect_corners[idx, 1], elect_corners[idx, 0], elect_corners[idx, 0]]
            y_coords = [elect_corners[idx, 3], elect_corners[idx, 3], 
                        elect_corners[idx, 4] if len(elect_corners[idx]) > 4 else elect_corners[idx, 2],
                        elect_corners[idx, 4] if len(elect_corners[idx]) > 4 else elect_corners[idx, 2], 
                        elect_corners[idx, 3]]
            ax.plot(x_coords, y_coords, 'k', linewidth=line_width)
    
    # Plot mirror outline
    max_rad = electrode_data['max_rad']
    x = np.linspace(-max_rad, max_rad, 100)
    ax.plot(x, np.sqrt(max_rad**2 - x**2), 'k', linewidth=line_width)
    ax.plot(x, -np.sqrt(max_rad**2 - x**2), 'k', linewidth=line_width)
    
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    plt.show()

if __name__ == "__main__":
    # Create electrodes
    electrode_data = create_electrodes()
    
    # Print some information
    print(f"Electrode grid size: {electrode_data['elect_grid']}x{electrode_data['elect_grid']}")
    print(f"Mirror grid size: {electrode_data['mirror_grid_size']}x{electrode_data['mirror_grid_size']}")
    print(f"Number of electrodes: {len(electrode_data['elect_corners'])}")