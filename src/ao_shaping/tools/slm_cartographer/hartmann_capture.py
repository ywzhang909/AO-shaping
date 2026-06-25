"""Hartmann-Shack spotfield capture and centroid tracking.

Handles image capture from Thorlabs WFS, reference spot registration,
and centroid displacement computation between reference and measurement frames.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class HartmannCaptureConfig:
    """Configuration for Hartmann-Shack capture.

    Attributes:
        n_averages: Number of frames to average for noise reduction.
        exposure_time_ms: WFS exposure time in ms (0 = auto).
        use_custom_ref: Whether to use a saved custom reference.
        ref_save_path: Optional path to save/load reference files.
        pupil_diameter_mm: Pupil diameter for WFS pupil configuration.
        pupil_center_mm: Pupil center (cx, cy) in mm.
        mla_resolution: WFS MLA resolution (e.g., "768", "512").
        high_speed: Enable high-speed mode on WFS.
        cancel_tilt: Whether to remove tip/tilt from wavefront.
        stable_samples: Number of stable samples for averaging.
        stability_threshold: Variance threshold for sample stability.
    """

    n_averages: int = 10
    exposure_time_ms: float = 0.0
    use_custom_ref: bool = True
    ref_save_path: str | None = None
    pupil_diameter_mm: float = 3.0
    pupil_center_mm: tuple[float, float] | None = None
    mla_resolution: str = "768"
    high_speed: bool = False
    cancel_tilt: bool = True
    stable_samples: int = 5
    stability_threshold: float = 0.1


@dataclass
class HartmannMeasurement:
    """Result from a single Hartmann-Shack measurement.

    Attributes:
        timestamp: ISO timestamp of measurement.
        spots_intensity: 2D array of spot intensities (num_spots_x, num_spots_y).
        centroids_x: 2D array of centroid X positions.
        centroids_y: 2D array of centroid Y positions.
        reference_centroids_x: Saved reference centroid X positions.
        reference_centroids_y: Saved reference centroid Y positions.
        displacements_x: X displacement from reference (pixels).
        displacements_y: Y displacement from reference (pixels).
        wavefront: Reconstructed wavefront (if computed).
        wavefront_stats: Statistics dictionary for wavefront.
        spotfield_image: Raw spotfield image from WFS.
        n_averages: Number of averaged frames.
        config_snapshot: Captured configuration parameters.
    """

    timestamp: str
    spots_intensity: np.ndarray
    centroids_x: np.ndarray
    centroids_y: np.ndarray
    reference_centroids_x: np.ndarray
    reference_centroids_y: np.ndarray
    displacements_x: np.ndarray
    displacements_y: np.ndarray
    wavefront: np.ndarray | None = None
    wavefront_stats: dict[str, float] = field(default_factory=dict)
    spotfield_image: np.ndarray | None = None
    n_averages: int = 10
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize measurement to a dictionary for storage."""
        d = asdict(self)
        for key in [
            "spots_intensity",
            "centroids_x",
            "centroids_y",
            "reference_centroids_x",
            "reference_centroids_y",
            "displacements_x",
            "displacements_y",
            "wavefront",
            "spotfield_image",
        ]:
            if d.get(key) is not None:
                d[key] = d[key].tolist()
        for key in ["wavefront_stats"]:
            if d.get(key) is not None:
                d[key] = {k: float(v) for k, v in d[key].items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HartmannMeasurement:
        """Deserialize measurement from a dictionary."""
        array_keys = [
            "spots_intensity",
            "centroids_x",
            "centroids_y",
            "reference_centroids_x",
            "reference_centroids_y",
            "displacements_x",
            "displacements_y",
            "wavefront",
            "spotfield_image",
        ]
        for key in array_keys:
            if d.get(key) is not None:
                d[key] = np.array(d[key])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class HartmannCapture:
    """Hartmann-Shack spotfield measurement controller.

    Wraps ThorlabWFS to provide:
    1. Reference spot registration (zero-grayscale baseline)
    2. Measurement capture with centroid tracking
    3. Displacement computation relative to reference
    4. Optional wavefront reconstruction

    Args:
        config: Capture configuration.
        wfs_device: ThorlabWFS instance (must be opened).
        storage_dir: Directory for saving reference/measurement data.
    """

    def __init__(
        self,
        config: HartmannCaptureConfig,
        wfs_device: Any,
        storage_dir: str | Path = "data/hartmann",
    ):
        self.config = config
        self._wfs = wfs_device
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._last_measurement: HartmannMeasurement | None = None
        self._reference_saved: bool = False

    def configure_wfs(self) -> None:
        """Apply configuration to WFS device."""
        assert self._wfs.is_connected(), "WFS device must be opened first"

        # Set exposure time (0 = auto)
        if self.config.exposure_time_ms > 0:
            self._wfs.exposure_time = self.config.exposure_time_ms

        # Set MLA resolution
        from ao_shaping.drivers.wfs.thorlab_wfs import MlaRes

        mla = MlaRes.from_str(self.config.mla_resolution, default=MlaRes.Res768)
        self._wfs.select_mla(mla)

        # Set pupil
        if self.config.pupil_center_mm is not None:
            cx, cy = self.config.pupil_center_mm
            dx = dy = self.config.pupil_diameter_mm
            self._wfs.pupil = (cx, cy, dx, dy)
        elif self.config.pupil_diameter_mm > 0:
            self._wfs.pupil = (
                0,
                0,
                self.config.pupil_diameter_mm,
                self.config.pupil_diameter_mm,
            )

        logger.info(
            f"WFS configured: MLA={self.config.mla_resolution}, "
            f"exposure={self.config.exposure_time_ms}ms"
        )

    def capture_reference(self, label: str = "reference") -> HartmannMeasurement:
        """Capture and save reference Hartmann spots (zero-grayscale baseline).

        Records spot positions with SLM at zero grayscale (flat/zero phase).
        This serves as the reference for subsequent displacement measurements.

        Args:
            label: Label for the reference measurement.

        Returns:
            HartmannMeasurement with saved reference centroids.
        """
        assert self._wfs.is_connected(), "WFS device must be opened first"
        logger.info(f"Capturing reference Hartmann spots: {label}")

        # Capture spots with averaging
        centroids_x, centroids_y, intensities = self._capture_centroids()

        measurement = HartmannMeasurement(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            spots_intensity=intensities,
            centroids_x=centroids_x,
            centroids_y=centroids_y,
            reference_centroids_x=centroids_x.copy(),
            reference_centroids_y=centroids_y.copy(),
            displacements_x=np.zeros_like(centroids_x),
            displacements_y=np.zeros_like(centroids_y),
            n_averages=self.config.n_averages,
            config_snapshot=self._get_config_snapshot(),
        )

        self._last_measurement = measurement
        self._reference_saved = True

        # Save reference to disk
        ref_path = self._storage_dir / f"reference_{label}.json"
        self._save_measurement(measurement, ref_path)

        if self.config.ref_save_path:
            ref_dir = Path(self.config.ref_save_path)
            ref_dir.mkdir(parents=True, exist_ok=True)
            self._save_measurement(measurement, ref_dir / f"hartmann_ref_{label}.json")

        logger.info(
            f"Reference captured: {centroids_x.shape} subapertures, saved to {ref_path}"
        )

        return measurement

    def capture_measurement(
        self,
        label: str = "measurement",
        wavefront_device: Any = None,
    ) -> HartmannMeasurement:
        """Capture measurement with current SLM pattern.

        Computes centroid displacements relative to saved reference.

        Args:
            label: Label for the measurement.
            wavefront_device: Optional WFS device for wavefront reconstruction.
                If None, uses self._wfs.

        Returns:
            HartmannMeasurement with displacements and optional wavefront.
        """
        assert self._wfs.is_connected(), "WFS device must be opened first"
        assert self._reference_saved, "Call capture_reference() first"

        logger.info(f"Capturing measurement Hartmann spots: {label}")

        centroids_x, centroids_y, intensities = self._capture_centroids()
        assert self._last_measurement is not None, "No reference measurement available"
        ref_x: np.ndarray = self._last_measurement.reference_centroids_x
        ref_y: np.ndarray = self._last_measurement.reference_centroids_y

        displacements_x = centroids_x - ref_x
        displacements_y = centroids_y - ref_y

        wavefront = None
        wavefront_stats: dict[str, float] = {}

        if wavefront_device is not None or True:
            wfs_for_wf = wavefront_device if wavefront_device else self._wfs
            try:
                wf, stats = wfs_for_wf.get_wavefront(
                    cancel_tile=self.config.cancel_tilt
                )
                wavefront = wf
                wavefront_stats = {k: float(v) for k, v in stats.items()}
                logger.debug(f"Wavefront RMS: {wavefront_stats.get('rms', 'N/A')}")
            except Exception as e:
                logger.warning(f"Wavefront reconstruction failed: {e}")

        measurement = HartmannMeasurement(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            spots_intensity=intensities,
            centroids_x=centroids_x,
            centroids_y=centroids_y,
            reference_centroids_x=ref_x.copy(),
            reference_centroids_y=ref_y.copy(),
            displacements_x=displacements_x,
            displacements_y=displacements_y,
            wavefront=wavefront,
            wavefront_stats=wavefront_stats,
            n_averages=self.config.n_averages,
            config_snapshot=self._get_config_snapshot(),
        )

        self._last_measurement = measurement

        # Save measurement
        meas_path = self._storage_dir / f"measurement_{label}.json"
        self._save_measurement(measurement, meas_path)

        logger.info(
            f"Measurement captured: mean displacement "
            f"dx={np.nanmean(displacements_x):.3f}, dy={np.nanmean(displacements_y):.3f} px"
        )

        return measurement

    def load_reference(self, ref_path: str | Path) -> HartmannMeasurement:
        """Load a previously saved reference measurement.

        Args:
            ref_path: Path to the reference JSON file.

        Returns:
            Loaded HartmannMeasurement set as current reference.
        """
        ref_path = Path(ref_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference file not found: {ref_path}")

        measurement = self._load_measurement(ref_path)
        self._last_measurement = measurement
        self._reference_saved = True

        logger.info(f"Reference loaded from {ref_path}")
        return measurement

    def get_last_measurement(self) -> HartmannMeasurement | None:
        """Get the most recent measurement."""
        return self._last_measurement

    def _capture_centroids(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Capture and average centroid positions from WFS.

        Returns:
            Tuple of (centroids_x, centroids_y, intensities), each 2D array
            of shape (num_spots_y, num_spots_x).
        """
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_int: list[np.ndarray] = []

        for i in range(self.config.n_averages):
            self._wfs.take_image(n_sample=1, dynamicNoiseCut=True)
            intensities, (cx, cy) = self._wfs.get_spots_statics()

            all_x.append(cx.T)
            all_y.append(cy.T)
            all_int.append(intensities.T)

            if i < self.config.n_averages - 1:
                time.sleep(0.05)

        centroids_x = np.mean(all_x, axis=0)
        centroids_y = np.mean(all_y, axis=0)
        intensities = np.mean(all_int, axis=0)

        return centroids_x, centroids_y, intensities

    def _get_config_snapshot(self) -> dict[str, Any]:
        """Capture current configuration as a dictionary."""
        try:
            wfs_info = self._wfs.get_hardware_info()
        except (RuntimeError, OSError, AttributeError):
            wfs_info = {}

        return {
            "n_averages": self.config.n_averages,
            "exposure_time_ms": float(self.config.exposure_time_ms),
            "mla_resolution": self.config.mla_resolution,
            "pupil_diameter_mm": float(self.config.pupil_diameter_mm),
            "num_spots_x": self._wfs.num_spots_x,
            "num_spots_y": self._wfs.num_spots_y,
            "wfs_serial": wfs_info.get("serial_number", "unknown"),
            "wfs_model": wfs_info.get("device_name", "unknown"),
        }

    def _save_measurement(self, measurement: HartmannMeasurement, path: Path) -> None:
        """Save measurement to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(measurement.to_dict(), f, indent=2, ensure_ascii=False)
        logger.debug(f"Measurement saved: {path}")

    def _load_measurement(self, path: Path) -> HartmannMeasurement:
        """Load measurement from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return HartmannMeasurement.from_dict(data)
