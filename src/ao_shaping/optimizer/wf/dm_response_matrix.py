"""DM actuator response matrix calibration module.

Measures the response of each DM actuator by applying push-pull voltage perturbations
and recording the corresponding WFS spot deviations. Builds the response matrix
that maps DM actuator commands to WFS slope measurements.

Supports:
- N-cycle push-pull measurement with M WFS averages per polarity
- Variance computation for stability metrics
- SVD pseudoinverse and least-squares inverse computation
- Per-actuator voltage optimization (linearity-based)
- Subaperture mask filtering for invalid subapertures
- HDF5 save/load with full metadata

Example:
    >>> from ao_shaping.optimizer.wf.dm_response_matrix import calibrate_dm_response_matrix
    >>> from ao_shaping.drivers.dm.NLight import NLight
    >>> from ao_shaping.drivers.wfs import ThorlabWFS
    >>>
    >>> with NLight() as dm:
    ...     with ThorlabWFS() as wfs:
    ...         result = calibrate_dm_response_matrix(dm, wfs, n_cycles=3, n_averages=5)
    ...         print(f"Matrix shape: {result.matrix.shape}")
    ...         print(f"Condition number: {result.condition_number}")
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from loguru import logger
from tqdm import tqdm

from ao_shaping.utils.matrix_utils import compute_lstsq, compute_pinv
from ao_shaping.utils.wfs_utils import flatten_slopes

if TYPE_CHECKING:
    from ao_shaping.drivers.dm.base import DM
    from ao_shaping.drivers.wfs import ThorlabWFS


# Default calibration parameters
DEFAULT_DISTURB_VOLTAGE: float = 50.0
DEFAULT_N_AVERAGES: int = 20
DEFAULT_N_CYCLES: int = 1
DEFAULT_WAIT_TIME: float = 0.1


@dataclass
class DMResponseMatrixResult:
    """DM actuator response matrix calibration result.

    Stores the response matrix mapping DM actuator voltages to WFS slope
    measurements, along with calibration metadata and optional inverse matrices.

    Attributes:
        matrix: Response matrix (n_slopes, n_actuators_valid).
        variance_matrix: Variance per element (n_slopes, n_actuators_valid).
        subaperture_mask: 2D bool array from WFS, shape (nx, ny).
        n_actuators: Total DM actuators (default 64).
        valid_actuator_indices: Indices of actuators that were measured.
        disturb_voltage: Voltage perturbation used during calibration.
        n_averages: WFS readings per measurement (M).
        n_cycles: Push-pull cycles (N).
        wait_time: Seconds after voltage apply.
        timestamp: ISO format datetime string.
        device_config: Hardware configuration snapshot dict.
        pinv_matrix: SVD pseudoinverse (n_valid, n_slopes).
        lstsq_matrix: Least-squares inverse (n_valid, n_slopes).
        amplitude_optimization: Per-actuator voltage optimization results.
    """

    matrix: np.ndarray
    variance_matrix: np.ndarray
    subaperture_mask: np.ndarray | None = None
    n_actuators: int = 64
    valid_actuator_indices: list[int] | None = None
    disturb_voltage: float = DEFAULT_DISTURB_VOLTAGE
    n_averages: int = DEFAULT_N_AVERAGES
    n_cycles: int = DEFAULT_N_CYCLES
    wait_time: float = DEFAULT_WAIT_TIME
    timestamp: str = ""
    device_config: dict | None = None
    pinv_matrix: np.ndarray | None = None
    lstsq_matrix: np.ndarray | None = None
    amplitude_optimization: dict | None = None

    @property
    def n_slopes(self) -> int:
        """Number of slope measurements (rows in matrix)."""
        return self.matrix.shape[0]

    @property
    def n_actuators_valid(self) -> int:
        """Number of valid actuators measured (columns in matrix)."""
        return self.matrix.shape[1]

    @property
    def mean_variance(self) -> float:
        """Mean of variance matrix (overall stability metric)."""
        return float(np.mean(self.variance_matrix))

    @property
    def max_variance(self) -> float:
        """Maximum variance element (worst-case stability)."""
        return float(np.max(self.variance_matrix))

    @property
    def condition_number(self) -> float | None:
        """Matrix condition number from SVD (if pseudoinverse was computed)."""
        if self.pinv_matrix is not None:
            return float(np.linalg.cond(self.matrix))
        return None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (ndarrays → lists)."""
        d = asdict(self)
        d["matrix"] = self.matrix.tolist()
        d["variance_matrix"] = self.variance_matrix.tolist()
        if self.subaperture_mask is not None:
            d["subaperture_mask"] = self.subaperture_mask.tolist()
        if self.pinv_matrix is not None:
            d["pinv_matrix"] = self.pinv_matrix.tolist()
        if self.lstsq_matrix is not None:
            d["lstsq_matrix"] = self.lstsq_matrix.tolist()
        if self.amplitude_optimization is not None:
            d["amplitude_optimization"] = self.amplitude_optimization
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DMResponseMatrixResult:
        """Create from dictionary (reconstructs ndarrays from lists)."""
        d = d.copy()
        d["matrix"] = np.array(d["matrix"])
        d["variance_matrix"] = np.array(d["variance_matrix"])
        if "subaperture_mask" in d and d["subaperture_mask"] is not None:
            d["subaperture_mask"] = np.array(d["subaperture_mask"])
        else:
            d["subaperture_mask"] = None
        if "pinv_matrix" in d and d["pinv_matrix"] is not None:
            d["pinv_matrix"] = np.array(d["pinv_matrix"])
        else:
            d["pinv_matrix"] = None
        if "lstsq_matrix" in d and d["lstsq_matrix"] is not None:
            d["lstsq_matrix"] = np.array(d["lstsq_matrix"])
        else:
            d["lstsq_matrix"] = None
        if "amplitude_optimization" in d and d["amplitude_optimization"] is not None:
            d["amplitude_optimization"] = d["amplitude_optimization"]
        else:
            d["amplitude_optimization"] = None
        return cls(**d)


def measure_actuator_response(
    dm: "DM",
    wfs: ThorlabWFS,
    actuator_idx: int,
    disturb_voltage: float,
    n_averages: int = DEFAULT_N_AVERAGES,
    n_cycles: int = DEFAULT_N_CYCLES,
    wait_time: float = DEFAULT_WAIT_TIME,
    debug_data_callback: Callable | None = None,
    cancel_tile: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure the response of a single DM actuator using push-pull method.

    For each cycle:
    1. Send +disturb_voltage to actuator_idx, others 0
    2. Wait, take n_averages WFS readings, average deviations
    3. Reset DM to zero
    4. Send -disturb_voltage to actuator_idx, others 0
    5. Wait, take n_averages WFS readings, average deviations
    6. Reset DM to zero
    7. response = (slopes_plus - slopes_minus) / (2 * disturb_voltage)

    Computes mean and variance of the response across cycles.
    Uses truncated mean when n_averages > 5 (sort, drop min/max, average rest).

    Args:
        dm: NLight DM instance (must be open).
        wfs: WFSManager instance (must be initialized).
        actuator_idx: Index of the actuator to measure.
        disturb_voltage: Voltage perturbation magnitude (positive).
        n_averages: Number of WFS readings per measurement (M).
        n_cycles: Number of push-pull cycles (N).
        wait_time: Seconds to wait after applying voltage.
        debug_data_callback: Optional callback for raw measurement data.
            Signature: callback(actuator_idx, cycle, sample, dev_x, dev_y, is_plus)
        cancel_tile: If True, remove tip/tilt from WFS deviations.

    Returns:
        tuple: (mean_slope_response, variance_slope_response, mean_dev_x, mean_dev_y)
            - mean_slope_response: Cycle-averaged slope response, flattened (n_slopes,).
            - variance_slope_response: Cycle variance of slope response, flattened (n_slopes,).
            - mean_dev_x: X-deviation response reshaped to (nx, ny).
            - mean_dev_y: Y-deviation response reshaped to (nx, ny).
    """
    total_actuators = dm.DM_NUM if hasattr(dm, "DM_NUM") else 64

    def measure_slopes_with_averaging() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Take n_averages WFS readings and return truncated-mean slopes."""
        all_dev_x = []
        all_dev_y = []
        for _ in range(n_averages):
            wfs.take_image(n_sample=1)
            dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=cancel_tile)
            all_dev_x.append(dev_x)
            all_dev_y.append(dev_y)

        dev_x_arr = np.array(all_dev_x)
        dev_y_arr = np.array(all_dev_y)

        if n_averages > 5:
            sorted_x = np.sort(dev_x_arr, axis=0)
            sorted_y = np.sort(dev_y_arr, axis=0)
            mean_dev_x = np.mean(sorted_x[1:-1], axis=0)
            mean_dev_y = np.mean(sorted_y[1:-1], axis=0)
        else:
            mean_dev_x = np.mean(dev_x_arr, axis=0)
            mean_dev_y = np.mean(dev_y_arr, axis=0)

        slopes = flatten_slopes(mean_dev_x, mean_dev_y)
        return slopes, mean_dev_x, mean_dev_y

    def apply_actuator_voltage(voltage: float) -> None:
        """Set a single actuator to voltage, all others to zero."""
        vs = np.zeros(total_actuators, dtype=np.float64)
        vs[actuator_idx] = voltage
        dm.send_voltages(vs, wait_time)

    all_responses: list[np.ndarray] = []
    all_dev_x_list: list[np.ndarray] = []
    all_dev_y_list: list[np.ndarray] = []

    for cycle in range(n_cycles):
        # --- Positive perturbation ---
        apply_actuator_voltage(+disturb_voltage)
        slopes_plus, dev_x_plus, dev_y_plus = measure_slopes_with_averaging()

        if debug_data_callback is not None:
            debug_data_callback(
                actuator_idx=actuator_idx,
                cycle=cycle,
                sample=-1,
                dev_x=dev_x_plus,
                dev_y=dev_y_plus,
                is_plus=True,
            )

        # Reset DM to zero
        dm.send_voltages(np.zeros(total_actuators, dtype=np.float64), wait_time)

        # --- Negative perturbation ---
        apply_actuator_voltage(-disturb_voltage)
        slopes_minus, dev_x_minus, dev_y_minus = measure_slopes_with_averaging()

        if debug_data_callback is not None:
            debug_data_callback(
                actuator_idx=actuator_idx,
                cycle=cycle,
                sample=-1,
                dev_x=dev_x_minus,
                dev_y=dev_y_minus,
                is_plus=False,
            )

        # Reset DM to zero
        dm.send_voltages(np.zeros(total_actuators, dtype=np.float64), wait_time)

        # Compute response for this cycle
        response = (slopes_plus - slopes_minus) / (2.0 * disturb_voltage)
        dev_x_resp = (dev_x_plus - dev_x_minus) / (2.0 * disturb_voltage)
        dev_y_resp = (dev_y_plus - dev_y_minus) / (2.0 * disturb_voltage)

        all_responses.append(response)
        all_dev_x_list.append(dev_x_resp)
        all_dev_y_list.append(dev_y_resp)

    all_responses_arr = np.array(all_responses)
    all_dev_x_arr = np.array(all_dev_x_list)
    all_dev_y_arr = np.array(all_dev_y_list)

    mean_slope_response = np.mean(all_responses_arr, axis=0)
    variance_slope_response = np.var(all_responses_arr, axis=0)

    # Mean deviation response in original (nx, ny) shape
    mean_dev_x = np.mean(all_dev_x_arr, axis=0)
    mean_dev_y = np.mean(all_dev_y_arr, axis=0)

    return mean_slope_response, variance_slope_response, mean_dev_x, mean_dev_y


def _optimize_perturbation_voltage(
    dm: "DM",
    wfs: ThorlabWFS,
    actuator_idx: int,
    test_voltages: np.ndarray,
    n_avg: int = 10,
    cancel_tile: bool = False,
) -> tuple[float, dict]:
    """Find optimal perturbation voltage for a DM actuator.

    Tests multiple voltage levels on one actuator and measures the response
    linearity (second derivative of response magnitude vs voltage). The optimal
    voltage is where the response is most linear (minimum second derivative).

    Args:
        dm: NLight DM instance.
        wfs: WFSManager instance.
        actuator_idx: Actuator index to test.
        test_voltages: Array of voltage levels to test.
        n_avg: WFS readings per measurement.
        cancel_tile: If True, remove tip/tilt from WFS deviations.

    Returns:
        tuple: (optimal_voltage, diagnostics_dict)
            - optimal_voltage: Best voltage found (float).
            - diagnostics: dict with test_voltages, responses, linearity scores,
              best_idx, optimal_voltage.
    """
    responses = []
    for v in test_voltages:
        mean_resp, _, _, _ = measure_actuator_response(
            dm=dm,
            wfs=wfs,
            actuator_idx=actuator_idx,
            disturb_voltage=v,
            n_averages=n_avg,
            n_cycles=1,
            wait_time=DEFAULT_WAIT_TIME,
            cancel_tile=cancel_tile,
        )
        resp_norm = float(np.linalg.norm(mean_resp))
        responses.append(resp_norm)

    responses = np.array(responses)

    # Compute linearity as second derivative (curvature) of response vs voltage
    linearity = []
    for i in range(1, len(test_voltages) - 1):
        k1 = (responses[i] - responses[i - 1]) / (test_voltages[i] - test_voltages[i - 1])
        k2 = (responses[i + 1] - responses[i]) / (test_voltages[i + 1] - test_voltages[i])
        linearity.append(float(abs(k1 - k2)))

    linearity = np.array(linearity)
    best_idx = int(np.argmin(linearity)) + 1
    optimal = float(test_voltages[best_idx])

    diagnostics = {
        "test_voltages": test_voltages.copy(),
        "responses": responses.copy(),
        "linearity": linearity.copy(),
        "best_idx": best_idx,
        "optimal_voltage": optimal,
    }

    logger.info(
        f"Optimized voltage for actuator {actuator_idx}: {optimal:.1f} V "
        f"(range: [{test_voltages[0]:.0f}, {test_voltages[-1]:.0f}])"
    )

    return optimal, diagnostics


def _build_device_config(
    dm: "DM",
    wfs: ThorlabWFS,
) -> dict:
    """Capture hardware configuration snapshot for metadata."""
    config: dict = {}

    # DM config
    try:
        dm_config: dict = {}
        if hasattr(dm, "DM_NUM"):
            dm_config["n_actuators"] = dm.DM_NUM
        if hasattr(dm, "V_Min") and hasattr(dm, "V_Max"):
            dm_config["voltage_min"] = dm.V_Min
            dm_config["voltage_max"] = dm.V_Max
        if hasattr(dm, "max_iter_diff"):
            dm_config["max_iter_diff"] = dm.max_iter_diff
        if hasattr(dm, "max_neibor_diff"):
            dm_config["max_neighbor_diff"] = dm.max_neibor_diff
        config["dm"] = dm_config
    except Exception as e:
        logger.debug(f"Could not capture DM config: {e}")

    # WFS config
    try:
        wfs_config: dict = {}
        if hasattr(wfs, "num_spots_x"):
            wfs_config["num_spots_x"] = wfs.num_spots_x
        if hasattr(wfs, "num_spots_y"):
            wfs_config["num_spots_y"] = wfs.num_spots_y
        if hasattr(wfs, "exposure_time"):
            wfs_config["exposure_time"] = wfs.exposure_time
        if hasattr(wfs, "mla_index"):
            wfs_config["mla_index"] = int(wfs.mla_index)
        if hasattr(wfs, "serial_num"):
            wfs_config["serial_number"] = str(wfs.serial_num)
        config["wfs"] = wfs_config
    except Exception as e:
        logger.debug(f"Could not capture WFS config: {e}")

    return config


def calibrate_dm_response_matrix(
    dm: "DM",
    wfs: ThorlabWFS,
    disturb_voltage: float | None = DEFAULT_DISTURB_VOLTAGE,
    n_cycles: int = DEFAULT_N_CYCLES,
    n_averages: int = DEFAULT_N_AVERAGES,
    wait_time: float = DEFAULT_WAIT_TIME,
    compute_inverses: bool = True,
    verbose: bool = True,
    dm_unit_mask: np.ndarray | None = None,
    cancel_tile: bool = False,
    subaperture_mask: np.ndarray | None = None,
    mask_n_avg: int = 30,
    mask_threshold_ratio: float = 0.3,
    mask_edge_clip: int = 1,
    auto_optimize_voltage: bool = True,
    optimize_n_avg: int = 10,
    debug_data_callback: Callable | None = None,
) -> DMResponseMatrixResult:
    """Calibrate DM actuator response matrix.

    Builds the response matrix by measuring each valid actuator's contribution
    to WFS spot deviations using a push-pull voltage perturbation method.

    The response matrix shape is (n_slopes, n_actuators_valid) where n_slopes
    is the number of filtered slope measurements (2 * valid subapertures if
    subaperture_mask is provided, else 2 * num_spots_x * num_spots_y).

    Args:
        dm: NLight DM instance (must be open).
        wfs: WFSManager instance (must be initialized).
        disturb_voltage: Voltage perturbation. If None/0 and auto_optimize=True,
            auto-optimizes per actuator.
        n_cycles: Push-pull cycles per actuator (N).
        n_averages: WFS readings per polarity per cycle (M).
        wait_time: Seconds after voltage apply.
        compute_inverses: If True, compute SVD pseudoinverse and least-squares inverse.
        verbose: If True, show tqdm progress bar.
        dm_unit_mask: Boolean mask of valid actuators (True = measure).
            If None, uses dm.default_dm_unit_mask. Actuator 0 is typically disabled.
        cancel_tile: If True, remove tip/tilt from WFS deviations.
        subaperture_mask: Pre-computed 2D bool mask of valid subapertures.
            If None, builds automatically via wfs.build_subaperture_mask().
        mask_n_avg: Frames to average for auto-built subaperture mask.
        mask_threshold_ratio: Intensity threshold ratio for auto-built mask.
        mask_edge_clip: Edge clip for auto-built mask.
        auto_optimize_voltage: If True and disturb_voltage is None/0, auto-optimize
            per-actuator voltage for best linearity.
        optimize_n_avg: WFS readings per voltage during optimization.
        debug_data_callback: Optional callback for raw measurement data.
            Signature depends on measure_actuator_response.

    Returns:
        DMResponseMatrixResult with response matrix, variance matrix, inverses, and metadata.
    """
    # Resolve total actuator count
    total_actuators = dm.DM_NUM if hasattr(dm, "DM_NUM") else 64

    # Resolve valid actuator indices from dm_unit_mask
    if dm_unit_mask is None:
        if hasattr(dm, "default_dm_unit_mask"):
            dm_unit_mask = dm.default_dm_unit_mask.copy()
        else:
            dm_unit_mask = np.ones(total_actuators, dtype=bool)
            dm_unit_mask[0] = 0

    dm_unit_mask = np.asarray(dm_unit_mask, dtype=bool)
    valid_indices = [int(i) for i in np.where(dm_unit_mask)[0]]
    n_valid = len(valid_indices)

    # Auto-optimize voltage if needed
    amplitude_optimization: dict | None = None
    if disturb_voltage is None or disturb_voltage == 0:
        if auto_optimize_voltage:
            logger.info("Auto-optimizing perturbation voltage for each actuator...")
            test_voltages = np.array([10, 20, 30, 50, 80, 100, 150, 200], dtype=np.float64)
            amplitude_optimization = {}
            for actuator_idx in valid_indices:
                opt_v, diagnostics = _optimize_perturbation_voltage(
                    dm=dm,
                    wfs=wfs,
                    actuator_idx=actuator_idx,
                    test_voltages=test_voltages,
                    n_avg=optimize_n_avg,
                    cancel_tile=cancel_tile,
                )
                amplitude_optimization[actuator_idx] = diagnostics
            # Use mean optimal voltage across all actuators
            disturb_voltage = float(np.mean(
                [v["optimal_voltage"] for v in amplitude_optimization.values()]
            ))
            logger.info(f"Using average optimal voltage: {disturb_voltage:.1f} V")
        else:
            disturb_voltage = DEFAULT_DISTURB_VOLTAGE

    # Build subaperture mask if not provided
    if subaperture_mask is None:
        logger.info("Building subaperture mask automatically...")
        subaperture_mask, _ = wfs.build_subaperture_mask(
            n_avg=mask_n_avg,
            threshold_ratio=mask_threshold_ratio,
            edge_clip=mask_edge_clip,
        )

    # Determine slope count
    nx = wfs.num_spots_x
    ny = wfs.num_spots_y
    n_total_slopes_unfiltered = 2 * nx * ny

    if subaperture_mask is not None:
        mask_flat = subaperture_mask.flatten()
        valid_rows = np.concatenate([mask_flat, mask_flat])
        n_slopes = int(np.sum(valid_rows))
    else:
        valid_rows = None
        n_slopes = n_total_slopes_unfiltered

    logger.info(
        f"Starting DM response matrix calibration: "
        f"total_actuators={total_actuators}, valid_actuators={n_valid}, "
        f"total_slopes={n_total_slopes_unfiltered}, filtered_slopes={n_slopes}, "
        f"voltage={disturb_voltage} V, cycles={n_cycles}, averages={n_averages}"
    )

    # Initialize response and variance matrices
    response_matrix = np.zeros((n_slopes, n_valid), dtype=np.float64)
    variance_matrix = np.zeros((n_slopes, n_valid), dtype=np.float64)

    # Initialize DM to zero and save WFS reference
    logger.info("Initializing DM to zero and saving WFS reference...")
    dm.send_voltages(np.zeros(total_actuators, dtype=np.float64), wait_time)
    wfs.save_user_ref()
    wfs.load_user_ref()
    time.sleep(wait_time)

    # Calibrate each valid actuator
    actuator_iter = valid_indices
    if verbose:
        actuator_iter = tqdm(valid_indices, desc="Actuators")

    for j, actuator_idx in enumerate(actuator_iter):
        if verbose and not isinstance(actuator_iter, tqdm):
            logger.info(f"Measuring actuator {actuator_idx + 1}/{total_actuators} "
                        f"(index {j + 1}/{n_valid})")

        mean_resp, var_resp, _, _ = measure_actuator_response(
            dm=dm,
            wfs=wfs,
            actuator_idx=actuator_idx,
            disturb_voltage=disturb_voltage,
            n_averages=n_averages,
            n_cycles=n_cycles,
            wait_time=wait_time,
            debug_data_callback=debug_data_callback,
            cancel_tile=cancel_tile,
        )

        # Apply subaperture mask filtering if available
        if valid_rows is not None:
            mean_resp = mean_resp[valid_rows]
            var_resp = var_resp[valid_rows]

        response_matrix[:, j] = mean_resp
        variance_matrix[:, j] = var_resp

        logger.debug(
            f"Actuator {actuator_idx} response RMS = "
            f"{float(np.sqrt(np.mean(mean_resp ** 2))):.6f}"
        )

    # Compute inverse matrices (optional)
    pinv_matrix = None
    lstsq_matrix = None

    if compute_inverses:
        logger.info("Computing SVD pseudoinverse...")
        pinv_matrix = compute_pinv(response_matrix)

        logger.info("Computing least-squares inverse...")
        lstsq_matrix = compute_lstsq(response_matrix)

    # Build hardware config snapshot
    device_config = _build_device_config(dm, wfs)

    result = DMResponseMatrixResult(
        matrix=response_matrix,
        variance_matrix=variance_matrix,
        subaperture_mask=subaperture_mask,
        n_actuators=total_actuators,
        valid_actuator_indices=valid_indices,
        disturb_voltage=disturb_voltage,
        n_averages=n_averages,
        n_cycles=n_cycles,
        wait_time=wait_time,
        timestamp=datetime.now().isoformat(),
        device_config=device_config,
        pinv_matrix=pinv_matrix,
        lstsq_matrix=lstsq_matrix,
        amplitude_optimization=amplitude_optimization,
    )

    logger.info(
        f"Calibration complete: matrix shape={result.matrix.shape}, "
        f"valid_actuators={n_valid}/{total_actuators}, "
        f"mean_variance={result.mean_variance:.6e}, "
        f"condition_number={result.condition_number}, "
        f"timestamp={result.timestamp}"
    )

    return result


def save_dm_response_matrix(
    result: DMResponseMatrixResult,
    path: str | Path,
    include_inverses: bool = True,
) -> None:
    """Save DM response matrix to HDF5 file with full metadata.

    HDF5 structure:
    - Datasets: matrix, variance_matrix, subaperture_mask (optional),
      pinv_matrix (optional), lstsq_matrix (optional)
    - Metadata group: stores scalar attrs and JSON-serialized fields

    Args:
        result: Calibration result to save.
        path: Save path. If suffix is not '.h5', appends '.h5'.
        include_inverses: If True, include pinv_matrix and lstsq_matrix.
    """
    path = Path(path)
    if path.suffix != ".h5":
        path = path.with_suffix(".h5")
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset("matrix", data=result.matrix)
        f.create_dataset("variance_matrix", data=result.variance_matrix)

        if result.subaperture_mask is not None:
            f.create_dataset("subaperture_mask", data=result.subaperture_mask)

        if include_inverses and result.pinv_matrix is not None:
            f.create_dataset("pinv_matrix", data=result.pinv_matrix)

        if include_inverses and result.lstsq_matrix is not None:
            f.create_dataset("lstsq_matrix", data=result.lstsq_matrix)

        meta = f.create_group("metadata")
        meta.attrs["n_actuators"] = result.n_actuators
        meta.attrs["disturb_voltage"] = result.disturb_voltage
        meta.attrs["n_averages"] = result.n_averages
        meta.attrs["n_cycles"] = result.n_cycles
        meta.attrs["wait_time"] = result.wait_time
        meta.attrs["timestamp"] = result.timestamp
        meta.attrs["mean_variance"] = result.mean_variance
        meta.attrs["max_variance"] = result.max_variance
        meta.attrs["condition_number"] = (
            result.condition_number if result.condition_number is not None else -1
        )

        if result.valid_actuator_indices is not None:
            meta.attrs["valid_actuator_indices"] = json.dumps(result.valid_actuator_indices)

        if result.device_config is not None:
            meta.attrs["device_config"] = json.dumps(result.device_config)

        if result.amplitude_optimization is not None:
            opt_grp = f.create_group("amplitude_optimization")
            for act_key, diag in result.amplitude_optimization.items():
                act_grp = opt_grp.create_group(f"actuator_{act_key}")
                act_grp.create_dataset("test_voltages", data=diag["test_voltages"])
                act_grp.create_dataset("responses", data=diag["responses"])
                act_grp.create_dataset("linearity", data=diag["linearity"])
                act_grp.attrs["best_idx"] = diag["best_idx"]
                act_grp.attrs["optimal_voltage"] = diag["optimal_voltage"]

    logger.info(f"DM response matrix saved to: {path}")


def load_dm_response_matrix(path: str | Path) -> DMResponseMatrixResult:
    """Load DM response matrix from HDF5 file.

    Args:
        path: File path (.h5 extension or base path without extension).

    Returns:
        DMResponseMatrixResult with loaded data and metadata.
    """
    path = Path(path)
    if path.suffix != ".h5":
        path = path.with_suffix(".h5")

    if not path.exists():
        raise FileNotFoundError(f"DM response matrix file not found: {path}")

    with h5py.File(path, "r") as f:
        matrix = f["matrix"][:]
        variance_matrix = f["variance_matrix"][:]

        subaperture_mask: np.ndarray | None = (
            f["subaperture_mask"][:] if "subaperture_mask" in f else None
        )
        pinv_matrix: np.ndarray | None = (
            f["pinv_matrix"][:] if "pinv_matrix" in f else None
        )
        lstsq_matrix: np.ndarray | None = (
            f["lstsq_matrix"][:] if "lstsq_matrix" in f else None
        )

        meta = f["metadata"]
        n_actuators = int(meta.attrs["n_actuators"])
        disturb_voltage = float(meta.attrs["disturb_voltage"])
        n_averages = int(meta.attrs["n_averages"])
        n_cycles = int(meta.attrs["n_cycles"])
        wait_time = float(meta.attrs["wait_time"])
        timestamp = str(meta.attrs["timestamp"])

        valid_actuator_indices: list[int] | None = None
        if "valid_actuator_indices" in meta.attrs:
            try:
                valid_actuator_indices = json.loads(meta.attrs["valid_actuator_indices"])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse valid_actuator_indices: {e}")

        device_config: dict | None = None
        if "device_config" in meta.attrs:
            try:
                device_config = json.loads(meta.attrs["device_config"])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse device_config: {e}")

        amplitude_optimization: dict | None = None
        if "amplitude_optimization" in f:
            amplitude_optimization = {}
            opt_grp = f["amplitude_optimization"]
            for act_key in opt_grp.keys():
                act_grp = opt_grp[act_key]
                act_idx = int(act_key.split("_")[1])
                amplitude_optimization[act_idx] = {
                    "test_voltages": act_grp["test_voltages"][:],
                    "responses": act_grp["responses"][:],
                    "linearity": act_grp["linearity"][:],
                    "best_idx": int(act_grp.attrs["best_idx"]),
                    "optimal_voltage": float(act_grp.attrs["optimal_voltage"]),
                }

        return DMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance_matrix,
            subaperture_mask=subaperture_mask,
            n_actuators=n_actuators,
            valid_actuator_indices=valid_actuator_indices,
            disturb_voltage=disturb_voltage,
            n_averages=n_averages,
            n_cycles=n_cycles,
            wait_time=wait_time,
            timestamp=timestamp,
            device_config=device_config,
            pinv_matrix=pinv_matrix,
            lstsq_matrix=lstsq_matrix,
            amplitude_optimization=amplitude_optimization,
        )


__all__ = [
    "DMResponseMatrixResult",
    "calibrate_dm_response_matrix",
    "save_dm_response_matrix",
    "load_dm_response_matrix",
    "measure_actuator_response",
    "DEFAULT_DISTURB_VOLTAGE",
    "DEFAULT_N_AVERAGES",
    "DEFAULT_N_CYCLES",
    "DEFAULT_WAIT_TIME",
]
