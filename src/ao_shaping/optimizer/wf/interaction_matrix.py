# encoding=utf-8
"""
获取响应矩阵 (Interaction Matrix) for adaptive optics using slope-based method.

This module calculates the interaction matrix that maps DM actuator commands to 
wavefront sensor spot deviations. The matrix is used in closed-loop adaptive optics
control systems to determine the optimal DM shape for correcting wavefront aberrations.

This module also provides Zernike-based SLM response matrix measurement and 
wavefront correction using SantecSLM200 and ThorlabWFS.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

from ao_shaping.drivers.dm.NLight import NLight
from ao_shaping.drivers import Thorlab_WFS, MlaRes
from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.drivers.wfs.thorlab_wfs import WFSManager
from ao_shaping.utils.zernike_calc import generate_noll_polynomial
from ao_shaping.utils.matrix_utils import compute_pinv
from ao_shaping.utils.wfs_utils import flatten_slopes


def calculate_interaction_matrix(
    wfs_res: str = '1024',
    disturb_voltage: float = 1.0,
    wait_time_s: float = 0.01,
    pupil_diameter: float = 2.24,
    save_path: str | Path = "data/interaction_matrix.txt"
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


def load_interaction_matrix(path: str | Path = "data/interaction_matrix.txt") -> np.ndarray:
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


# =============================================================================
# Zernike-based SLM Response Matrix (for SantecSLM200 + ThorlabWFS)
# =============================================================================

# Default Zernike modes for SLM response matrix calibration
# (n, m) tuples - excluding piston (0, 0)
DEFAULT_SLM_ZERNIKE_MODES: list[tuple[int, int]] = [
    (1, -1),  # Tip
    (1, 1),   # Tilt
    (2, 0),   # Defocus
    (2, -2),  # Astigmatism 45°
    (2, 2),   # Astigmatism 0°
    (3, -1),  # Coma Y
    (3, 1),   # Coma X
    (3, -3),  # Trefoil Y
    (3, 3),   # Trefoil X
    (4, 0),   # Spherical
]


@dataclass
class ZernikeSLMResponseMatrixResult:
    """Zernike SLM → WFS slopes response matrix result.

    This dataclass stores the response matrix that maps SLM Zernike phase
    patterns to WFS spot deviations (slopes), along with calibration metadata.

    Attributes:
        matrix: Response matrix of shape (2*N_spots, N_modes)
        variance_matrix: Variance matrix of same shape as matrix
        slm_wavelength_nm: SLM operating wavelength (nm)
        wfs_resolution: WFS resolution string ('512', '768', '1024')
        pupil_diameter_mm: Pupil diameter in mm
        magnitude_rad: Phase perturbation magnitude in radians
        n_cycles: Number of positive/negative cycles
        n_averages: WFS frames averaged per measurement
        timestamp: ISO format timestamp
        pinv_matrix: Optional precomputed pseudoinverse (N_modes, 2*N_spots)
    """

    matrix: np.ndarray
    variance_matrix: np.ndarray
    slm_wavelength_nm: int
    wfs_resolution: str
    pupil_diameter_mm: float
    magnitude_rad: float
    n_cycles: int
    n_averages: int
    timestamp: str
    pinv_matrix: np.ndarray | None = None

    @property
    def n_spots(self) -> int:
        """Number of WFS spots (total)."""
        return self.matrix.shape[0] // 2

    @property
    def n_modes(self) -> int:
        """Number of SLM Zernike modes."""
        return self.matrix.shape[1]

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "slm_wavelength_nm": self.slm_wavelength_nm,
            "wfs_resolution": self.wfs_resolution,
            "pupil_diameter_mm": self.pupil_diameter_mm,
            "magnitude_rad": self.magnitude_rad,
            "n_cycles": self.n_cycles,
            "n_averages": self.n_averages,
            "timestamp": self.timestamp,
            "matrix_shape": self.matrix.shape,
            "n_spots": self.n_spots,
            "n_modes": self.n_modes,
        }

    @classmethod
    def from_dict(
        cls,
        d: dict,
        matrix: np.ndarray,
        variance_matrix: np.ndarray,
        pinv_matrix: np.ndarray | None = None,
    ) -> "ZernikeSLMResponseMatrixResult":
        """Create from dictionary and loaded arrays."""
        return cls(
            matrix=matrix,
            variance_matrix=variance_matrix,
            slm_wavelength_nm=d["slm_wavelength_nm"],
            wfs_resolution=d["wfs_resolution"],
            pupil_diameter_mm=d["pupil_diameter_mm"],
            magnitude_rad=d["magnitude_rad"],
            n_cycles=d["n_cycles"],
            n_averages=d["n_averages"],
            timestamp=d["timestamp"],
            pinv_matrix=pinv_matrix,
        )


def calculate_zernike_slm_response_matrix(
    slm: SantecSLM200,
    wfs: WFSManager,
    slm_zernike_modes: list[tuple[int, int]] | None = None,
    magnitude_rad: float = 0.5,
    n_cycles: int = 3,
    n_averages: int = 5,
    wait_time_s: float = 0.1,
    save_path: str | Path | None = "data/zernike_slm_response_matrix",
) -> ZernikeSLMResponseMatrixResult:
    """Calculate Zernike SLM → WFS slopes response matrix.

    The forward model is: g = D · a + ε
    where:
        g: slope vector from WFS (shape: (2*N_spots,))
        D: slope response matrix (shape: (2*N_spots, N_modes))
        a: Zernike coefficients vector (shape: (N_modes,))
        ε: measurement noise

    Uses positive/negative perturbation method:
        D[:, i] = (slopes_plus - slopes_minus) / (2 * magnitude_rad)

    Args:
        slm: SantecSLM200 instance (must be open, in Memory mode)
        wfs: WFSManager instance (must be initialized)
        slm_zernike_modes: List of (n, m) Zernike modes to calibrate.
            Default: [(1,-1), (1,1), (2,0), (2,-2), (2,2), (3,-1), (3,1), (3,-3), (3,3), (4,0)]
            Note: Piston (0,0) is skipped.
        magnitude_rad: Phase perturbation magnitude in radians (~0.5 rad recommended)
        n_cycles: Number of positive/negative cycles for averaging
        n_averages: WFS frames to average per measurement
        wait_time_s: Wait time after applying SLM pattern
        save_path: Path to save results (None to skip saving).
            Saves: {save_path}.matrix.npy, {save_path}.variance.npy, {save_path}.json

    Returns:
        ZernikeSLMResponseMatrixResult with response matrix and metadata
    """
    # Use default modes if not specified
    if slm_zernike_modes is None:
        slm_zernike_modes = DEFAULT_SLM_ZERNIKE_MODES.copy()

    # Get SLM panel resolution: (width, height) = (1920, 1200)
    slm_resolution = (slm.Panel_Res[0], slm.Panel_Res[1])

    # Get WFS spot count
    n_spots_x, n_spots_y = wfs.num_spots_x, wfs.num_spots_y
    n_spots = n_spots_x * n_spots_y
    n_modes = len(slm_zernike_modes)

    logger.info(
        f"Starting Zernike SLM response matrix calibration: "
        f"modes={n_modes}, n_spots={n_spots}, magnitude={magnitude_rad}rad, "
        f"cycles={n_cycles}, averages={n_averages}"
    )

    # Initialize matrices
    response_matrix = np.zeros((2 * n_spots, n_modes), dtype=np.float64)
    variance_matrix = np.zeros((2 * n_spots, n_modes), dtype=np.float64)

    # Get pupil diameter from WFS
    pupil_diameter_mm = wfs.d_x if hasattr(wfs, "d_x") and wfs.d_x else 2.24

    # Function to measure slopes with averaging
    def measure_slopes() -> np.ndarray:
        """Measure WFS slopes, averaging over n_averages frames."""
        wfs.take_image(n_sample=n_averages)
        x_dev, y_dev = wfs.get_spot_deviation()
        slopes = flatten_slopes(x_dev, y_dev)
        return slopes

    # Reset SLM to flat
    zero_phase = np.zeros(slm_resolution, dtype=np.float64)
    zero_gray = slm.create_phase_from_array(zero_phase)
    slm.write_phase(zero_gray, memory_number=1)
    slm.display_memory(1)
    time.sleep(wait_time_s * 2)  # Extra wait for SLM to stabilize

    # Measure each Zernike mode
    for mode_idx, (n, m) in enumerate(tqdm(slm_zernike_modes, desc="Zernike modes")):
        logger.debug(f"Calibrating mode ({n}, {m}) - index {mode_idx}")

        # Collect responses from multiple cycles
        cycle_responses = []

        for cycle in range(n_cycles):
            # Positive perturbation
            phase_pos = generate_noll_polynomial(n, m, slm_resolution, magnitude_rad)
            gray_pos = slm.create_phase_from_array(phase_pos)
            slm.write_phase(gray_pos, memory_number=1)
            slm.display_memory(1)
            time.sleep(wait_time_s)
            slopes_pos = measure_slopes()

            # Reset to flat briefly
            slm.write_phase(zero_gray, memory_number=1)
            slm.display_memory(1)
            time.sleep(wait_time_s)

            # Negative perturbation
            phase_neg = generate_noll_polynomial(n, m, slm_resolution, -magnitude_rad)
            gray_neg = slm.create_phase_from_array(phase_neg)
            slm.write_phase(gray_neg, memory_number=1)
            slm.display_memory(1)
            time.sleep(wait_time_s)
            slopes_neg = measure_slopes()

            # Reset to flat
            slm.write_phase(zero_gray, memory_number=1)
            slm.display_memory(1)
            time.sleep(wait_time_s)

            # Calculate response for this cycle
            response = (slopes_pos - slopes_neg) / (2 * magnitude_rad)
            cycle_responses.append(response)

        # Stack and compute mean/variance
        cycle_responses = np.array(cycle_responses)  # (n_cycles, 2*n_spots)
        response_matrix[:, mode_idx] = np.mean(cycle_responses, axis=0)
        variance_matrix[:, mode_idx] = np.var(cycle_responses, axis=0)

    # Compute pseudoinverse
    logger.info("Computing SVD pseudoinverse...")
    pinv_matrix = compute_pinv(response_matrix, rcond=1e-6)

    # Create result object
    result = ZernikeSLMResponseMatrixResult(
        matrix=response_matrix,
        variance_matrix=variance_matrix,
        slm_wavelength_nm=slm.wavelength,
        wfs_resolution=str(wfs.num_spots_x),
        pupil_diameter_mm=pupil_diameter_mm,
        magnitude_rad=magnitude_rad,
        n_cycles=n_cycles,
        n_averages=n_averages,
        timestamp=datetime.now().isoformat(),
        pinv_matrix=pinv_matrix,
    )

    logger.info(
        f"Zernike SLM response matrix calibration complete: "
        f"shape={result.matrix.shape}, mean_variance={np.mean(variance_matrix):.6f}"
    )

    # Save if path provided
    if save_path is not None:
        save_zernike_slm_response_matrix(result, save_path)

    return result


def apply_zernike_correction(
    slm: SantecSLM200,
    wfs: WFSManager,
    response_matrix: ZernikeSLMResponseMatrixResult | np.ndarray,
    slm_zernike_modes: list[tuple[int, int]] | None = None,
    regularization: float = 1e-6,
    n_averages: int = 1,
    apply_correction: bool = True,
    # PID control parameters
    Kp: float = 1.0,
    Ki: float = 0.0,
    Kd: float = 0.0,
    pid: bool = False,
    max_iterations: int = 10,
    target: np.ndarray | None = None,
    convergence_threshold: float = 1e-6,
    wait_time_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray | list]:
    """Apply Zernike correction using pseudoinverse of response matrix.

    Steps:
    1. Measure current slopes: g = WFS.get_spot_deviation()
    2. Compute Zernike coefficients: â = D⁺ · g (via SVD pseudoinverse)
    3. Apply negative correction to SLM: phase = -â · Zernike_basis

    When pid=True, runs an iterative PID control loop to converge to target.

    Args:
        slm: SantecSLM200 instance
        wfs: WFSManager instance
        response_matrix: Pre-calibrated response matrix or result object
        slm_zernike_modes: List of (n, m) Zernike modes used in calibration.
            If None, uses DEFAULT_SLM_ZERNIKE_MODES.
        regularization: Regularization parameter for pseudoinverse (λ for SVD)
        n_averages: Number of WFS frames to average for measurement
        apply_correction: If True, apply correction to SLM. If False, only compute.
        Kp: Proportional gain for PID controller (default 1.0)
        Ki: Integral gain for PID controller (default 0.0)
        Kd: Derivative gain for PID controller (default 0.0)
        pid: If True, run iterative PID control loop (default False)
        max_iterations: Maximum PID iterations (default 10)
        target: Target Zernike coefficients. If None, target is zeros.
            Shape (N_modes,). Only used when pid=True.

    Returns:
        When pid=False:
            Tuple of (estimated_zernike_coeffs, measured_slopes)
        When pid=True:
            Tuple of (final_zernike_coeffs, history_dict)
            history_dict contains: 'slopes', 'coeffs', 'corrections'
    """
    # Handle response_matrix: could be ZernikeSLMResponseMatrixResult, np.ndarray, or file path (str/Path)
    if isinstance(response_matrix, ZernikeSLMResponseMatrixResult):
        matrix = response_matrix.matrix
        pinv = response_matrix.pinv_matrix
        if slm_zernike_modes is None:
            slm_zernike_modes = DEFAULT_SLM_ZERNIKE_MODES
    elif isinstance(response_matrix, np.ndarray):
        matrix = response_matrix
        pinv = None
    else:
        # Assume it's a file path (str or Path)
        result = load_zernike_slm_response_matrix(response_matrix)
        matrix = result.matrix
        pinv = result.pinv_matrix
        if slm_zernike_modes is None:
            slm_zernike_modes = DEFAULT_SLM_ZERNIKE_MODES

    n_modes = matrix.shape[1]

    # Use default modes if not specified
    if slm_zernike_modes is None:
        slm_zernike_modes = DEFAULT_SLM_ZERNIKE_MODES[:n_modes]

    # Compute pseudoinverse if not pre-computed
    if pinv is None:
        pinv = compute_pinv(matrix)

    # Get SLM resolution for phase generation
    slm_resolution = (slm.Panel_Res[0], slm.Panel_Res[1])

    # Set target coefficients (default: zeros = fully corrected)
    if target is None:
        target = np.zeros(n_modes)
    else:
        target = np.asarray(target, dtype=np.float64)[:n_modes]

    # PID mode: iterative control loop
    if pid:
        return _apply_zernike_correction_pid(
            slm=slm,
            slm_resolution=slm_resolution,
            wfs=wfs,
            pinv=pinv,
            slm_zernike_modes=slm_zernike_modes[:n_modes],
            n_averages=n_averages,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            max_iterations=max_iterations,
            target=target,
            convergence_threshold=convergence_threshold,
            wait_time_s=wait_time_s,
        )

    # Single-shot mode (original behavior)
    return _apply_zernike_correction_single(
        slm=slm,
        slm_resolution=slm_resolution,
        wfs=wfs,
        pinv=pinv,
        slm_zernike_modes=slm_zernike_modes[:n_modes],
        n_averages=n_averages,
        apply_correction=apply_correction,
    )


def _apply_zernike_correction_single(
    slm: SantecSLM200,
    slm_resolution: tuple[int, int],
    wfs: WFSManager,
    pinv: np.ndarray,
    slm_zernike_modes: list[tuple[int, int]],
    n_averages: int,
    apply_correction: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-shot Zernike correction (original behavior)."""
    # Measure current slopes
    wfs.take_image(n_sample=n_averages)
    x_dev, y_dev = wfs.get_spot_deviation()
    g = np.concatenate([x_dev.flatten(), y_dev.flatten()])

    # Compute Zernike coefficients: a_hat = pinv @ g
    a_hat = pinv @ g

    logger.debug(f"Estimated Zernike coefficients: {a_hat[:5]}... (showing first 5)")

    if apply_correction:
        # Generate correction phase: phase = -sum(a_hat[i] * Zernike_i)
        correction_phase = np.zeros(slm_resolution, dtype=np.float64)

        for i, (n, m) in enumerate(slm_zernike_modes):
            if i >= len(a_hat):
                break
            mode_phase = generate_noll_polynomial(n, m, slm_resolution, -a_hat[i])
            # Transpose mode_phase from (height, width) to (width, height) to match slm_resolution
            correction_phase += mode_phase.T

        # Apply to SLM
        gray_corr = slm.create_phase_from_array(correction_phase)
        slm.write_phase(gray_corr, memory_number=1)
        slm.display_memory(1)
        logger.info("Zernike correction applied to SLM")

    return a_hat, g


def _apply_zernike_correction_pid(
    slm: SantecSLM200,
    slm_resolution: tuple[int, int],
    wfs: WFSManager,
    pinv: np.ndarray,
    slm_zernike_modes: list[tuple[int, int]],
    n_averages: int,
    Kp: float,
    Ki: float,
    Kd: float,
    max_iterations: int,
    target: np.ndarray,
    convergence_threshold: float,
    wait_time_s: float,
) -> tuple[np.ndarray, list]:
    """PID control loop for iterative Zernike correction.

    Returns:
        (final_coeffs, history) where history is a list of coefficient vectors (np.ndarray).
    """
    n_modes = len(slm_zernike_modes)

    # History tracking (list of coefficient vectors)
    history: list[np.ndarray] = []

    # PID state
    integral = np.zeros(n_modes, dtype=np.float64)
    prev_error = None

    # Current correction applied to SLM (starts at zero)
    current_correction = np.zeros(slm_resolution, dtype=np.float64)

    for iteration in range(max_iterations):
        # Measure current slopes
        wfs.take_image(n_sample=n_averages)
        x_dev, y_dev = wfs.get_spot_deviation()
        g = np.concatenate([x_dev.flatten(), y_dev.flatten()])

        # Compute Zernike coefficients from slopes
        a_hat = pinv @ g

        # Compute error = target - current_coeffs
        # If target is zero, we want a_hat to go to zero (fully corrected)
        error = target - a_hat

        # PID terms
        p_term = Kp * error
        integral += Ki * error
        d_term = np.zeros_like(error)
        if prev_error is not None:
            d_term = Kd * (error - prev_error)
        prev_error = error.copy()

        # PID correction
        correction_coeffs = p_term + integral + d_term

        # Update history (append coefficient vector)
        history.append(a_hat.copy())

        logger.debug(
            f"PID iteration {iteration + 1}/{max_iterations}: "
            f"error_rms={np.sqrt(np.mean(error**2)):.6f}"
        )

        # Check convergence: error RMS below threshold
        error_rms = np.sqrt(np.mean(error**2))
        if error_rms < convergence_threshold:
            logger.info(f"PID converged at iteration {iteration + 1}")
            break

        # Apply correction to SLM
        # correction_phase = +sum(correction_coeffs[i] * Zernike_i)
        # Note: slm_resolution is (width, height), but generate_noll_polynomial returns (height, width)
        # So we transpose the result to match slm_resolution
        correction_phase = np.zeros(slm_resolution, dtype=np.float64)
        for i, (n, m) in enumerate(slm_zernike_modes):
            if i >= len(correction_coeffs):
                break
            mode_phase = generate_noll_polynomial(
                n, m, slm_resolution, correction_coeffs[i]
            )
            # Transpose mode_phase from (height, width) to (width, height) to match slm_resolution
            correction_phase += mode_phase.T

        # Accumulate correction (add to existing phase pattern)
        current_correction += correction_phase

        # Apply to SLM
        gray_corr = slm.create_phase_from_array(current_correction)
        slm.write_phase(gray_corr, memory_number=1)
        slm.display_memory(1)

        # Wait time between iterations
        if wait_time_s > 0:
            time.sleep(wait_time_s)

    logger.info(f"PID loop completed: {max_iterations} iterations")

    # Return final coefficients and history (list of coeffs)
    final_coeffs = a_hat
    return final_coeffs, history


def save_zernike_slm_response_matrix(
    result: ZernikeSLMResponseMatrixResult,
    path: str | Path,
) -> None:
    """Save response matrix with JSON metadata.

    Args:
        result: ZernikeSLMResponseMatrixResult to save
        path: Base path (without extension). Saves:
            - {path}.matrix.npy: Response matrix
            - {path}.variance.npy: Variance matrix
            - {path}.pinv.npy: Pseudoinverse (if available)
            - {path}.json: Metadata
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save matrices
    np.save(path.with_suffix(".matrix.npy"), result.matrix)
    np.save(path.with_suffix(".variance.npy"), result.variance_matrix)
    if result.pinv_matrix is not None:
        np.save(path.with_suffix(".pinv.npy"), result.pinv_matrix)

    # Save metadata
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"Zernike SLM response matrix saved to {path.parent}")


def load_zernike_slm_response_matrix(
    path: str | Path,
) -> ZernikeSLMResponseMatrixResult:
    """Load response matrix from files.

    Args:
        path: Base path (without extension)

    Returns:
        ZernikeSLMResponseMatrixResult with loaded data
    """
    path = Path(path)

    # Load matrices
    matrix = np.load(path.with_suffix(".matrix.npy"))
    variance_matrix = np.load(path.with_suffix(".variance.npy"))

    # Load pseudoinverse if exists
    pinv_path = path.with_suffix(".pinv.npy")
    pinv_matrix = np.load(pinv_path) if pinv_path.exists() else None

    # Load metadata
    with open(path.with_suffix(".json"), encoding="utf-8") as f:
        metadata = json.load(f)

    return ZernikeSLMResponseMatrixResult.from_dict(
        metadata, matrix, variance_matrix, pinv_matrix
    )


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
