import numpy as np
import time
import sys
import os

# Add the parent directory to the path to allow relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from zernike import zernfun
from mat_utils import MatUtils

def get_integrand1(r, phi, rp, phip, nlim):
    """
    Compute first integrand function.
    
    Args:
        r: Radial coordinate
        phi: Angular coordinate
        rp: Radial coordinate prime
        phip: Angular coordinate prime
        nlim: Limit for summation
        
    Returns:
        int1: First integrand value
    """
    sum1 = 0.0
    for n in range(1, nlim + 1):
        term = (1/n) * ((r*rp)**n - (rp/r)**n) * np.cos(n*(phip-phi))
        sum1 += term
    # Due to the change of variables *rp is not necessary because it cancels
    # with the 1/det(dT)=1/rp
    int1 = (np.log(1/r) - sum1)
    return int1

def get_integrand2(r, phi, rp, phip, nlim):
    """
    Compute second integrand function.
    
    Args:
        r: Radial coordinate
        phi: Angular coordinate
        rp: Radial coordinate prime
        phip: Angular coordinate prime
        nlim: Limit for summation
        
    Returns:
        int2: Second integrand value
    """
    sum2 = 0.0
    for n in range(1, nlim + 1):
        term = (1/n) * ((r*rp)**n - (r/rp)**n) * np.cos(n*(phip-phi))
        sum2 += term
    # Due to the change of variables *rp is not necessary because it cancels
    # with the 1/det(dT)=1/rp
    int2 = (np.log(1/rp) - sum2)
    return int2

def get_m_term(x, y, x1, x2, y1, y2, elect_grid_size, nlim):
    """
    Compute M term for influence matrix calculation.
    
    Args:
        x, y: Point coordinates
        x1, x2, y1, y2: Electrode boundaries
        elect_grid_size: Grid size for electrode
        nlim: Limit for summation
        
    Returns:
        term: M term value
    """
    x_iso_line = np.linspace(x1, x2, elect_grid_size)
    y_iso_line = np.linspace(y1, y2, elect_grid_size)
    x_grid, y_grid = np.meshgrid(x_iso_line, y_iso_line)
    
    fun_mat = np.zeros((elect_grid_size, elect_grid_size))
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    for j in range(len(x_iso_line)):
        for i in range(len(y_iso_line)):
            rp = np.sqrt(x_grid[i, j]**2 + y_grid[i, j]**2)
            phip = np.arctan2(y_grid[i, j], x_grid[i, j])
            
            # For the change of variables to cartesian coordinates with the
            # transformation map (x,y = T(r,phi) = r*cos(phi),r*sin(phi) it
            # would be necessary to multiply funMat by 1/det(dT) with
            # 1/det([cos(phip) -rp*sin(phip);
            #        sin(phip)  rp*cos(phip)])=1/rp;
            # However the 1/rp cancels with the *rp of the integrand and
            # therefore it is omitted in the code
            if rp < r:
                fun_mat[i, j] = get_integrand1(r, phi, rp, phip, nlim)
            else:
                fun_mat[i, j] = get_integrand2(r, phi, rp, phip, nlim)
                
    term = np.sum(fun_mat)
    return term

def compute_influence_matrix(electrode_data, n_seq_lim=80, elect_grid_size=40, a=1.0, t=1.0):
    """
    Compute influence matrix for deformable mirror simulation.
    
    Args:
        electrode_data: Dictionary containing electrode parameters
        n_seq_lim: Sequence limit for computation
        elect_grid_size: Grid size for electrode
        a: Parameter a
        t: Parameter T
        
    Returns:
        inf_funcs: Influence functions
    """
    # Extract parameters from electrode_data
    mirror_grid_size = electrode_data['mirror_grid_size']
    mirror_x_grid = electrode_data['mirror_x_grid']
    mirror_y_grid = electrode_data['mirror_y_grid']
    mirror_mask = electrode_data['mirror_mask']
    elect_corners = electrode_data['elect_corners']
    
    # Initialize influence functions
    inf_funcs = np.full((mirror_grid_size, mirror_grid_size, len(elect_corners)), np.nan)
    
    print(f"Computing influence matrix for {len(elect_corners)} electrodes...")
    t_start = time.time()
    
    for k in range(len(elect_corners)):
        t_elect = time.time()
        
        # Compute influence function for each point in the grid
        for j in range(mirror_grid_size):
            for i in range(mirror_grid_size):
                # Validates that point is inside the mirror
                if mirror_mask[i, j] == 1:
                    inf_funcs[i, j, k] = get_m_term(
                        mirror_x_grid[i, j],
                        mirror_y_grid[i, j],
                        elect_corners[k, 0],
                        elect_corners[k, 1],
                        elect_corners[k, 2],
                        elect_corners[k, 3],
                        elect_grid_size,
                        n_seq_lim
                    )
        
        elapsed = time.time() - t_elect
        print(f"Electrode {k+1}/{len(elect_corners)} computed in {elapsed:.2f} seconds")
        
        # Scale the influence function
        inf_funcs[:, :, k] = (inf_funcs[:, :, k] / elect_grid_size**2) * a**2 / (2 * np.pi * t)
    
    total_elapsed = time.time() - t_start
    print(f"Total computation time: {total_elapsed:.2f} seconds")
    
    return inf_funcs

if __name__ == "__main__":
    # This would be run after creating electrodes
    print("This module should be imported and used with electrode data.")