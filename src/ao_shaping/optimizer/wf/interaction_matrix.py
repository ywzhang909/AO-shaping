# encoding=utf-8
"""
获取响应矩阵 (Interaction Matrix) for adaptive optics using slope-based method.

This module calculates the interaction matrix that maps DM actuator commands to 
wavefront sensor spot deviations. The matrix is used in closed-loop adaptive optics
control systems to determine the optimal DM shape for correcting wavefront aberrations.
"""

import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from ao_shaping.drivers.dm.NLight import NLight
from ao_shaping.drivers import Thorlab_WFS, MlaRes


def calculate_interaction_matrix(
    wfs_res: str = '1024',
    disturb_voltage: float = 1.0,
    wait_time_s: float = 0.01,
    pupil_diameter: float = 2.24,
    save_path: str = "data/interaction_matrix.txt"
) -> np.ndarray:
    """
    Calculate the interaction matrix for adaptive optics using the slope method.

    The interaction matrix maps DM actuator commands to WFS spot deviations:
    slopes = interaction_matrix @ dm_commands

    Args:
        wfs_res: WFS resolution ('512' or '768')
        disturb_voltage: Voltage perturbation applied to each DM actuator
        wait_time_s: Wait time after sending DM commands
        pupil_diameter: Pupil diameter in mm
        save_path: Path to save the interaction matrix

    Returns:
        np.ndarray: Interaction matrix of shape (2*N_spots, N_actuators)
                    where N_spots is the number of WFS spots and N_actuators is DM actuators
    """
    logger.info("Starting interaction matrix calculation...")

    # Select WFS resolution
    wfs_res_config = MlaRes.from_str(wfs_res)

    with NLight() as dm, Thorlab_WFS(wfs_res_config, use_custom_ref=False, high_speed=False, pupil_diameter=pupil_diameter) as wfs:
        num_actuators = dm.DM_Num
        logger.info(f"DM has {num_actuators} actuators")

        # Initialize DM to zero
        init_v = np.zeros(num_actuators)
        dm.send_voltages(init_v, wait_time_s)

        # Take initial measurement to get spot count
        wfs.take_image(1)
        initial_dev_x, initial_dev_y = wfs.get_spot_deviation()
        num_spots_x, num_spots_y = initial_dev_x.shape
        num_spots = num_spots_x * num_spots_y
        logger.info(f"WFS has {num_spots_x}x{num_spots_y} = {num_spots} spots")

        # Initialize interaction matrix
        # Shape: (2 * num_spots, num_actuators)
        # First num_spots rows: x-deviations
        # Last num_spots rows: y-deviations
        interaction_matrix = np.zeros((2 * num_spots, num_actuators))

        # Measure response for each actuator
        logger.info("Measuring response for each actuator...")
        for i in tqdm(range(num_actuators), desc="Actuators"):
            # Skip first actuator (usually reference)
            if i == 0:
                continue

            # Positive perturbation
            pos_v = init_v.copy()
            pos_v[i] = disturb_voltage
            dm.send_voltages(pos_v, wait_time_s)
            wfs.take_image(20)
            pos_dev_x, pos_dev_y = wfs.get_spot_deviation()

            # Negative perturbation
            neg_v = init_v.copy()
            neg_v[i] = -disturb_voltage
            dm.send_voltages(neg_v, wait_time_s)
            wfs.take_image(20)
            neg_dev_x, neg_dev_y = wfs.get_spot_deviation()

            # Calculate response (finite difference)
            # Response = (positive - negative) / (2 * disturb_voltage)
            resp_x = (pos_dev_x - neg_dev_x) / (2 * disturb_voltage)
            resp_y = (pos_dev_y - neg_dev_y) / (2 * disturb_voltage)

            # Flatten and store in matrix
            # X-deviations in first half of rows
            interaction_matrix[:num_spots, i] = resp_x.flatten()
            # Y-deviations in second half of rows
            interaction_matrix[num_spots:, i] = resp_y.flatten()

            # Reset DM
            dm.send_voltages(init_v, wait_time_s)

        # Save matrix
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(save_path, interaction_matrix)
        logger.info(f"Interaction matrix saved to {save_path}")

        return interaction_matrix


def load_interaction_matrix(path: str = "data/interaction_matrix.txt") -> np.ndarray:
    """
    Load a previously calculated interaction matrix.

    Args:
        path: Path to the saved interaction matrix

    Returns:
        np.ndarray: Interaction matrix
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Interaction matrix not found at {path}")

    return np.loadtxt(path)


def apply_interaction_matrix(
    slopes: np.ndarray,
    interaction_matrix: np.ndarray,
    regularization: float = 1e-6
) -> np.ndarray:
    """
    Apply the interaction matrix to calculate DM commands from slopes.

    This uses a pseudo-inverse with regularization to solve:
    dm_commands = pinv(interaction_matrix) @ slopes

    Args:
        slopes: Measured slopes (2*N_spots,)
        interaction_matrix: Interaction matrix (2*N_spots, N_actuators)
        regularization: Regularization parameter for pseudo-inverse

    Returns:
        np.ndarray: DM commands (N_actuators,)
    """
    # Regularized pseudo-inverse
    # Using SVD for numerical stability
    U, s, Vt = np.linalg.svd(interaction_matrix, full_matrices=False)
    s_reg = s / (s**2 + regularization**2)
    interaction_matrix_pinv = Vt.T @ np.diag(s_reg) @ U.T

    return interaction_matrix_pinv @ slopes


if __name__ == "__main__":
    # Example usage
    try:
        # Calculate and save interaction matrix
        matrix = calculate_interaction_matrix(
            wfs_res='768',
            disturb_voltage=50.0,
            wait_time_s=0.1,
            pupil_diameter=2.27,
            save_path="data/interaction_matrix.txt"
        )
        print(f"Interaction matrix shape: {matrix.shape}")

        # Load matrix
        loaded_matrix = load_interaction_matrix("data/interaction_matrix.txt")
        print(f"Loaded matrix shape: {loaded_matrix.shape}")
        print("Interaction matrix calculation completed successfully!")

    except Exception as e:
        logger.error(f"Error during interaction matrix calculation: {e}")
        raise
