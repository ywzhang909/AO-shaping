"""Mock devices for testing and development.

This module provides simulated implementations of various hardware devices
for testing, development, and demonstration purposes without requiring
actual hardware.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import Device, DeviceError, DeviceState, DeviceType


class MockCameraError(DeviceError):
    """Exception raised for mock camera errors."""
    pass


class MockCamera(Device):
    """Mock camera device for testing.

    Simulates a camera with configurable resolution, noise, and capture behavior.

    Attributes:
        device_type: DeviceType.CAMERA
        manufacturer: "Mock"
        model: "Simulated Camera"

    Example:
        >>> cam = MockCamera(device_id="mock_cam_001", resolution=(1024, 1024))
        >>> with cam:
        ...     img = cam.capture()
        ...     print(f"Captured: {img.shape}")
    """

    device_type = DeviceType.CAMERA
    manufacturer = "Mock"
    model = "Simulated Camera"

    def __init__(
        self,
        device_id: str = "",
        resolution: tuple[int, int] = (1024, 1024),
        noise_level: float = 5.0,
        simulate_delay: float = 0.01,
        random_seed: int | None = None,
    ):
        """Initialize mock camera.

        Args:
            device_id: Unique device identifier.
            resolution: Image resolution (width, height).
            noise_level: Standard deviation of Gaussian noise.
            simulate_delay: Simulated capture delay in seconds.
            random_seed: Random seed for reproducible output. If None, uses random initialization.
        """
        super().__init__(device_id)

        self._resolution = resolution
        self._noise_level = noise_level
        self._simulate_delay = simulate_delay
        self._frame_counter = 0
        self._last_image: np.ndarray | None = None
        self._rng = np.random.default_rng(random_seed)

        self._register_parameters()
        self._register_capabilities()

    def _register_parameters(self) -> None:
        """Register camera-specific parameters."""
        self.register_parameter(
            "exposure_time_ms",
            default_value=20.0,
            min_value=0.1,
            max_value=10000.0,
            unit="ms",
            description="Exposure time in milliseconds",
        )
        self.register_parameter(
            "gain",
            default_value=1.0,
            min_value=1.0,
            max_value=100.0,
            unit="",
            description="Analog gain",
        )
        self.register_parameter(
            "brightness",
            default_value=128.0,
            min_value=0.0,
            max_value=255.0,
            unit="",
            description="Image brightness offset",
        )
        self.register_parameter(
            "contrast",
            default_value=1.0,
            min_value=0.1,
            max_value=10.0,
            unit="",
            description="Image contrast factor",
        )
        self.register_parameter(
            "auto_exposure",
            default_value=False,
            unit="",
            description="Enable auto exposure",
        )

    def _register_capabilities(self) -> None:
        """Register camera capabilities."""
        self.register_capability(
            "capture",
            description="Capture single image",
            return_type=np.ndarray,
        )
        self.register_capability(
            "capture_average",
            description="Capture averaged image",
            parameters=["n_frames"],
            return_type=np.ndarray,
        )
        self.register_capability(
            "get_resolution",
            description="Get current resolution",
            return_type=tuple,
        )

    def open(self) -> None:
        """Open mock camera connection."""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.05)  # Simulate connection delay
        self._frame_counter = 0
        self._set_state(DeviceState.READY)
        logger.info(f"Mock camera {self.device_id} opened")

    def close(self) -> None:
        """Close mock camera connection."""
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock camera {self.device_id} closed")

    def is_connected(self) -> bool:
        """Check if camera is connected."""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """Get mock hardware information."""
        return {
            "serial_number": f"MOCK_CAM_{self.device_id[:8]}",
            "firmware_version": "1.0.0-mock",
            "resolution": self._resolution,
            "pixel_format": "MONO8",
            "sensor_type": "MockSensor",
        }

    def _generate_image(self) -> np.ndarray:
        """Generate a synthetic image with patterns."""
        width, height = self._resolution
        brightness = self.get_parameter_value("brightness")
        contrast = self.get_parameter_value("contrast")
        gain = self.get_parameter_value("gain")

        # Create base pattern (gradient + some features)
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 4 * np.pi, height)
        xx, yy = np.meshgrid(x, y)

        # Synthetic pattern: combination of sine waves
        pattern = (
            np.sin(xx) * np.cos(yy) * 50
            + np.sin(xx * 0.5) * 30
            + np.cos(yy * 0.3) * 20
        )

        # Add some "objects" (gaussian blobs)
        for _ in range(5):
            cx, cy = self._rng.integers(0, width), self._rng.integers(0, height)
            sigma = self._rng.uniform(10, 50)
            blob = np.exp(-((xx - cx * 4 * np.pi / width) ** 2 + (yy - cy * 4 * np.pi / height) ** 2) / (2 * sigma ** 2))
            pattern += blob * 100

        # Apply brightness, contrast, gain
        image = (pattern + brightness) * contrast * gain

        # Add noise
        noise = self._rng.normal(0, self._noise_level, image.shape)
        image = image + noise

        # Clip to valid range
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def capture(self, n_samples: int = 1) -> np.ndarray:
        """Capture image from mock camera.

        Args:
            n_samples: Number of samples for averaging.

        Returns:
            Captured image as uint8 array.
        """
        if not self.is_connected():
            raise RuntimeError("Camera not connected")

        self._set_state(DeviceState.BUSY)
        try:
            time.sleep(self._simulate_delay)

            if n_samples == 1:
                img = self._generate_image()
            else:
                # Average multiple frames
                frames = [self._generate_image().astype(np.float32) for _ in range(n_samples)]
                img = np.mean(frames, axis=0).astype(np.uint8)

            self._last_image = img
            self._frame_counter += 1

            self._emit_data("image", img)
            return img
        finally:
            self._set_state(DeviceState.READY)

    def get_resolution(self) -> tuple[int, int]:
        """Get current resolution."""
        return self._resolution

    def get_twin_state(self) -> dict[str, Any]:
        """Get state for digital twin synchronization."""
        state = super().get_twin_state()
        state["hardware"] = {
            "resolution": self._resolution,
            "frame_counter": self._frame_counter,
        }
        return state


class MockSLMError(DeviceError):
    """Exception raised for mock SLM errors."""
    pass


class MockSLM(Device):
    """Mock Spatial Light Modulator (SLM) device.

    Simulates an SLM with configurable resolution and phase modulation.

    Attributes:
        device_type: DeviceType.SLM
        manufacturer: "Mock"
        model: "Simulated SLM"

    Example:
        >>> slm = MockSLM(device_id="mock_slm_001", resolution=(1920, 1080))
        >>> with slm:
        ...     phase_pattern = np.random.rand(1920, 1080) * 2 * np.pi
        ...     slm.write_phase(phase_pattern)
    """

    device_type = DeviceType.SLM
    manufacturer = "Mock"
    model = "Simulated SLM"

    def __init__(
        self,
        device_id: str = "",
        resolution: tuple[int, int] = (1920, 1080),
        bit_depth: int = 8,
    ):
        """Initialize mock SLM.

        Args:
            device_id: Unique device identifier.
            resolution: SLM resolution (width, height).
            bit_depth: Bit depth for phase representation (8 or 16).
        """
        super().__init__(device_id)

        self._resolution = resolution
        self._bit_depth = bit_depth
        self._current_pattern: np.ndarray | None = None
        self._wavelength_nm = 633.0
        self._frame_count = 0

        self._register_parameters()
        self._register_capabilities()

    def _register_parameters(self) -> None:
        """Register SLM-specific parameters."""
        self.register_parameter(
            "wavelength",
            default_value=633.0,
            min_value=300.0,
            max_value=1100.0,
            unit="nm",
            description="Operating wavelength",
        )
        self.register_parameter(
            "frame_rate",
            default_value=60.0,
            min_value=1.0,
            max_value=120.0,
            unit="Hz",
            description="Frame refresh rate",
        )
        self.register_parameter(
            "phase_range",
            default_value=2 * np.pi,
            min_value=np.pi,
            max_value=4 * np.pi,
            unit="rad",
            description="Maximum phase modulation range",
        )
        self.register_parameter(
            "gamma_correction",
            default_value=1.0,
            min_value=0.5,
            max_value=3.0,
            unit="",
            description="Gamma correction factor",
        )

    def _register_capabilities(self) -> None:
        """Register SLM capabilities."""
        self.register_capability(
            "write_phase",
            description="Write phase pattern to SLM",
            parameters=["phase_pattern"],
        )
        self.register_capability(
            "write_grayscale",
            description="Write grayscale image to SLM",
            parameters=["image"],
        )
        self.register_capability(
            "get_resolution",
            description="Get SLM resolution",
            return_type=tuple,
        )

    def open(self) -> None:
        """Open mock SLM connection."""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.1)
        self._current_pattern = np.zeros(self._resolution, dtype=np.float32)
        self._set_state(DeviceState.READY)
        logger.info(f"Mock SLM {self.device_id} opened")

    def close(self) -> None:
        """Close mock SLM connection."""
        self._current_pattern = None
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock SLM {self.device_id} closed")

    def is_connected(self) -> bool:
        """Check if SLM is connected."""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """Get mock hardware information."""
        return {
            "serial_number": f"MOCK_SLM_{self.device_id[:8]}",
            "firmware_version": "2.0.0-mock",
            "resolution": self._resolution,
            "bit_depth": self._bit_depth,
            "pixel_pitch_um": 8.0,
            "fill_factor": 0.95,
        }

    def write_phase(self, phase_pattern: np.ndarray) -> None:
        """Write phase pattern to mock SLM.

        Args:
            phase_pattern: 2D array of phase values in radians.
        """
        if not self.is_connected():
            raise RuntimeError("SLM not connected")

        if phase_pattern.shape != self._resolution:
            raise ValueError(
                f"Pattern shape {phase_pattern.shape} doesn't match SLM resolution {self._resolution}"
            )

        self._set_state(DeviceState.BUSY)
        try:
            time.sleep(0.005)  # Simulate write delay

            # Normalize and apply gamma correction
            gamma = self.get_parameter_value("gamma_correction")
            phase_range = self.get_parameter_value("phase_range")

            normalized = np.clip(phase_pattern / phase_range, 0, 1)
            if gamma != 1.0:
                normalized = np.power(normalized, 1.0 / gamma)

            # Convert to bit depth
            max_value = (1 << self._bit_depth) - 1
            self._current_pattern = (normalized * max_value).astype(np.uint16 if self._bit_depth > 8 else np.uint8)

            self._frame_count += 1
            logger.debug(f"SLM pattern written (frame {self._frame_count})")
        finally:
            self._set_state(DeviceState.READY)

    def write_grayscale(self, image: np.ndarray) -> None:
        """Write grayscale image to mock SLM.

        Args:
            image: 2D array of grayscale values (0-255).
        """
        if not self.is_connected():
            raise RuntimeError("SLM not connected")

        if image.shape != self._resolution:
            raise ValueError(
                f"Image shape {image.shape} doesn't match SLM resolution {self._resolution}"
            )

        self._set_state(DeviceState.BUSY)
        try:
            time.sleep(0.005)
            self._current_pattern = image.astype(np.uint8)
            self._frame_count += 1
        finally:
            self._set_state(DeviceState.READY)

    def get_current_pattern(self) -> np.ndarray | None:
        """Get the currently displayed pattern."""
        return self._current_pattern.copy() if self._current_pattern is not None else None

    def get_resolution(self) -> tuple[int, int]:
        """Get SLM resolution."""
        return self._resolution

    def get_twin_state(self) -> dict[str, Any]:
        """Get state for digital twin synchronization."""
        state = super().get_twin_state()
        state["hardware"] = {
            "resolution": self._resolution,
            "frame_count": self._frame_count,
            "bit_depth": self._bit_depth,
        }
        return state


class MockDMError(DeviceError):
    """Exception raised for mock DM errors."""
    pass


class MockDM(Device):
    """Mock Deformable Mirror (DM) device.

    Simulates a deformable mirror with configurable actuator count
    and voltage-to-deformation model.

    Attributes:
        device_type: DeviceType.DM
        manufacturer: "Mock"
        model: "Simulated DM"

    Example:
        >>> dm = MockDM(device_id="mock_dm_001", n_actuators=64)
        >>> with dm:
        ...     voltages = np.zeros(64)
        ...     dm.apply_voltages(voltages)
        ...     surface = dm.get_surface()
    """

    device_type = DeviceType.DM
    manufacturer = "Mock"
    model = "Simulated DM"

    def __init__(
        self,
        device_id: str = "",
        n_actuators: int = 64,
        voltage_range: tuple[float, float] = (0.0, 300.0),
    ):
        """Initialize mock DM.

        Args:
            device_id: Unique device identifier.
            n_actuators: Number of actuators.
            voltage_range: Min/max voltage range (V).
        """
        super().__init__(device_id)

        self._n_actuators = n_actuators
        self._voltage_range = voltage_range
        self._current_voltages = np.zeros(n_actuators)
        self._surface_shape = (int(np.sqrt(n_actuators)) * 10,) * 2

        # Influence matrix (actuator voltages to surface deformation)
        self._influence_matrix = self._create_influence_matrix()

        self._register_parameters()
        self._register_capabilities()

    def _register_parameters(self) -> None:
        """Register DM-specific parameters."""
        self.register_parameter(
            "voltage_limit",
            default_value=self._voltage_range[1],
            min_value=self._voltage_range[0],
            max_value=500.0,
            unit="V",
            description="Maximum actuator voltage",
        )
        self.register_parameter(
            "bias_voltage",
            default_value=150.0,
            min_value=0.0,
            max_value=300.0,
            unit="V",
            description="Actuator bias voltage",
        )
        self.register_parameter(
            "settling_time_ms",
            default_value=1.0,
            min_value=0.1,
            max_value=100.0,
            unit="ms",
            description="DM settling time after voltage change",
        )
        self.register_parameter(
            "hysteresis_factor",
            default_value=0.1,
            min_value=0.0,
            max_value=1.0,
            unit="",
            description="Hysteresis effect magnitude",
        )

    def _register_capabilities(self) -> None:
        """Register DM capabilities."""
        self.register_capability(
            "apply_voltages",
            description="Apply voltages to actuators",
            parameters=["voltages"],
        )
        self.register_capability(
            "get_surface",
            description="Get current mirror surface shape",
            return_type=np.ndarray,
        )
        self.register_capability(
            "reset",
            description="Reset all actuators to zero",
        )

    def _create_influence_matrix(self) -> np.ndarray:
        """Create influence matrix for actuator to surface mapping."""
        # Simplified model: each actuator creates a Gaussian influence
        n_grid = int(np.sqrt(self._n_actuators))
        surface_size = n_grid * 10

        act_pos = np.array([
            [(i + 0.5) / n_grid * surface_size, (j + 0.5) / n_grid * surface_size]
            for i in range(n_grid)
            for j in range(n_grid)
        ])

        x = np.linspace(0, surface_size, surface_size)
        y = np.linspace(0, surface_size, surface_size)
        xx, yy = np.meshgrid(x, y)

        influence = np.zeros((surface_size, surface_size, self._n_actuators))
        sigma = surface_size / n_grid * 0.8

        for i, (ax, ay) in enumerate(act_pos):
            influence[:, :, i] = np.exp(-((xx - ax)**2 + (yy - ay)**2) / (2 * sigma**2))

        return influence.reshape(-1, self._n_actuators)

    def open(self) -> None:
        """Open mock DM connection."""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.1)
        self._current_voltages = np.zeros(self._n_actuators)
        self._set_state(DeviceState.READY)
        logger.info(f"Mock DM {self.device_id} opened ({self._n_actuators} actuators)")

    def close(self) -> None:
        """Close mock DM connection."""
        self._current_voltages = np.zeros(self._n_actuators)
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock DM {self.device_id} closed")

    def is_connected(self) -> bool:
        """Check if DM is connected."""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """Get mock hardware information."""
        return {
            "serial_number": f"MOCK_DM_{self.device_id[:8]}",
            "firmware_version": "1.5.0-mock",
            "n_actuators": self._n_actuators,
            "actuator_pitch_um": 300.0,
            "max_voltage": self._voltage_range[1],
            "coating": "Gold",
        }

    def apply_voltages(self, voltages: np.ndarray) -> None:
        """Apply voltages to DM actuators.

        Args:
            voltages: Array of voltages for each actuator.
        """
        if not self.is_connected():
            raise RuntimeError("DM not connected")

        if len(voltages) != self._n_actuators:
            raise ValueError(
                f"Voltage array length {len(voltages)} doesn't match actuator count {self._n_actuators}"
            )

        self._set_state(DeviceState.BUSY)
        try:
            # Clip to voltage range
            v_limit = self.get_parameter_value("voltage_limit")
            voltages = np.clip(voltages, self._voltage_range[0], v_limit)

            # Simulate settling time
            settling_ms = self.get_parameter_value("settling_time_ms")
            time.sleep(settling_ms / 1000.0)

            # Apply hysteresis effect
            hysteresis = self.get_parameter_value("hysteresis_factor")
            if hysteresis > 0:
                direction = np.sign(voltages - self._current_voltages)
                voltages = voltages + direction * hysteresis * np.abs(voltages - self._current_voltages)

            self._current_voltages = voltages
            logger.debug(f"DM voltages applied: min={voltages.min():.2f}, max={voltages.max():.2f}")
        finally:
            self._set_state(DeviceState.READY)

    def get_surface(self) -> np.ndarray:
        """Get current mirror surface shape.

        Returns:
            2D array representing the mirror surface in nanometers.
        """
        if not self.is_connected():
            raise RuntimeError("DM not connected")

        # Convert voltages to surface deformation
        bias = self.get_parameter_value("bias_voltage")
        effective_voltages = self._current_voltages + bias

        # Simple model: surface = influence_matrix @ voltages
        surface_flat = self._influence_matrix @ effective_voltages
        surface = surface_flat.reshape(self._surface_shape)

        # Convert to nanometers (simplified scaling)
        surface_nm = surface * 100  # 100 nm per unit voltage effect

        return surface_nm

    def reset(self) -> None:
        """Reset all actuators to zero voltage."""
        self.apply_voltages(np.zeros(self._n_actuators))

    def get_current_voltages(self) -> np.ndarray:
        """Get current actuator voltages."""
        return self._current_voltages.copy()

    def get_twin_state(self) -> dict[str, Any]:
        """Get state for digital twin synchronization."""
        state = super().get_twin_state()
        state["hardware"] = {
            "n_actuators": self._n_actuators,
            "voltage_range": self._voltage_range,
            "current_voltages": self._current_voltages.tolist(),
        }
        return state


class MockWFSError(DeviceError):
    """Exception raised for mock WFS errors."""
    pass


class MockWFS(Device):
    """Mock Wavefront Sensor (WFS) device.

    Simulates a Shack-Hartmann wavefront sensor for wavefront measurement.

    Attributes:
        device_type: DeviceType.WFS
        manufacturer: "Mock"
        model: "Simulated WFS"

    Example:
        >>> wfs = MockWFS(device_id="mock_wfs_001", n_lenslets=32)
        >>> with wfs:
        ...     wf = wfs.measure_wavefront()
        ...     zernike = wfs.fit_zernike(wf, n_modes=15)
    """

    device_type = DeviceType.WFS
    manufacturer = "Mock"
    model = "Simulated WFS"

    def __init__(
        self,
        device_id: str = "",
        n_lenslets: int = 32,
        pupil_size_mm: float = 5.0,
        random_seed: int | None = None,
    ):
        """Initialize mock WFS.

        Args:
            device_id: Unique device identifier.
            n_lenslets: Number of lenslets per side.
            pupil_size_mm: Pupil diameter in millimeters.
            random_seed: Random seed for reproducible output. If None, uses random initialization.
        """
        super().__init__(device_id)

        self._n_lenslets = n_lenslets
        self._pupil_size_mm = pupil_size_mm
        self._spot_image: np.ndarray | None = None
        self._rng = np.random.default_rng(random_seed)

        self._register_parameters()
        self._register_capabilities()

    def _register_parameters(self) -> None:
        """Register WFS-specific parameters."""
        self.register_parameter(
            "integration_time_ms",
            default_value=10.0,
            min_value=1.0,
            max_value=1000.0,
            unit="ms",
            description="Sensor integration time",
        )
        self.register_parameter(
            "threshold",
            default_value=50.0,
            min_value=0.0,
            max_value=255.0,
            unit="",
            description="Spot detection threshold",
        )
        self.register_parameter(
            "pupil_offset_x",
            default_value=0.0,
            min_value=-5.0,
            max_value=5.0,
            unit="mm",
            description="Pupil center X offset",
        )
        self.register_parameter(
            "pupil_offset_y",
            default_value=0.0,
            min_value=-5.0,
            max_value=5.0,
            unit="mm",
            description="Pupil center Y offset",
        )

    def _register_capabilities(self) -> None:
        """Register WFS capabilities."""
        self.register_capability(
            "measure_wavefront",
            description="Measure wavefront",
            return_type=np.ndarray,
        )
        self.register_capability(
            "fit_zernike",
            description="Fit Zernike polynomials to wavefront",
            parameters=["n_modes"],
            return_type=np.ndarray,
        )
        self.register_capability(
            "get_spot_image",
            description="Get spot pattern image",
            return_type=np.ndarray,
        )

    def open(self) -> None:
        """Open mock WFS connection."""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.1)
        self._set_state(DeviceState.READY)
        logger.info(f"Mock WFS {self.device_id} opened ({self._n_lenslets}x{self._n_lenslets} lenslets)")

    def close(self) -> None:
        """Close mock WFS connection."""
        self._spot_image = None
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock WFS {self.device_id} closed")

    def is_connected(self) -> bool:
        """Check if WFS is connected."""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """Get mock hardware information."""
        return {
            "serial_number": f"MOCK_WFS_{self.device_id[:8]}",
            "firmware_version": "3.0.0-mock",
            "n_lenslets": self._n_lenslets,
            "pupil_size_mm": self._pupil_size_mm,
            "lenslet_pitch_um": 150.0,
            "focal_length_mm": 10.0,
        }

    def measure_wavefront(self) -> np.ndarray:
        """Measure wavefront.

        Returns:
            2D array of wavefront phase in radians.
        """
        if not self.is_connected():
            raise RuntimeError("WFS not connected")

        self._set_state(DeviceState.BUSY)
        try:
            time.sleep(0.01)  # Simulate measurement time

            # Generate synthetic wavefront with some aberrations
            size = self._n_lenslets * 10
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            xx, yy = np.meshgrid(x, y)

            # Simulate some Zernike aberrations
            z_defocus = 0.5 * (2 * (xx**2 + yy**2) - 1)
            z_astig = 0.3 * (xx**2 - yy**2)
            z_coma_x = 0.2 * (3 * (xx**2 + yy**2) - 2) * xx

            wavefront = z_defocus + z_astig + z_coma_x

            # Add noise
            wavefront += self._rng.normal(0, 0.05, wavefront.shape)

            return wavefront
        finally:
            self._set_state(DeviceState.READY)

    def fit_zernike(self, wavefront: np.ndarray, n_modes: int = 15) -> np.ndarray:
        """Fit Zernike polynomials to measured wavefront.

        Args:
            wavefront: Measured wavefront.
            n_modes: Number of Zernike modes to fit.

        Returns:
            Array of Zernike coefficients.
        """
        # Simplified: return random coefficients for demonstration
        return self._rng.standard_normal(n_modes) * 0.1

    def get_spot_image(self) -> np.ndarray:
        """Get spot pattern image.

        Returns:
            2D image of lenslet spots.
        """
        if not self.is_connected():
            raise RuntimeError("WFS not connected")

        # Generate synthetic spot pattern
        img_size = self._n_lenslets * 20
        image = np.zeros((img_size, img_size), dtype=np.uint8)

        # Add spots
        for i in range(self._n_lenslets):
            for j in range(self._n_lenslets):
                cx = int((i + 0.5) / self._n_lenslets * img_size)
                cy = int((j + 0.5) / self._n_lenslets * img_size)

                # Add Gaussian spot
                y, x = np.ogrid[-10:11, -10:11]
                spot = np.exp(-(x**2 + y**2) / 10) * 200

                y0, x0 = max(0, cy - 10), max(0, cx - 10)
                y1, x1 = min(img_size, cy + 11), min(img_size, cx + 11)
                sy0, sx0 = max(0, 10 - cy), max(0, 10 - cx)
                sy1, sx1 = sy0 + (y1 - y0), sx0 + (x1 - x0)

                image[y0:y1, x0:x1] += spot[sy0:sy1, sx0:sx1].astype(np.uint8)

        self._spot_image = np.clip(image, 0, 255).astype(np.uint8)
        return self._spot_image

    def get_twin_state(self) -> dict[str, Any]:
        """Get state for digital twin synchronization."""
        state = super().get_twin_state()
        state["hardware"] = {
            "n_lenslets": self._n_lenslets,
            "pupil_size_mm": self._pupil_size_mm,
        }
        return state


class MockStageError(DeviceError):
    """Exception raised for mock stage errors."""
    pass


class MockStage(Device):
    """Mock motion stage device.

    Simulates a linear translation stage with position feedback.

    Attributes:
        device_type: DeviceType.STAGE
        manufacturer: "Mock"
        model: "Simulated Stage"

    Example:
        >>> stage = MockStage(device_id="mock_stage_001", axis="X")
        >>> with stage:
        ...     stage.move_to(10.0)
        ...     pos = stage.get_position()
    """

    device_type = DeviceType.STAGE
    manufacturer = "Mock"
    model = "Simulated Stage"

    def __init__(
        self,
        device_id: str = "",
        axis: str = "X",
        travel_range: tuple[float, float] = (0.0, 100.0),
    ):
        """Initialize mock stage.

        Args:
            device_id: Unique device identifier.
            axis: Axis name (X, Y, Z, etc.).
            travel_range: Min/max travel range in mm.
        """
        super().__init__(device_id)

        self._axis = axis
        self._travel_range = travel_range
        self._current_position = travel_range[0]
        self._target_position = self._current_position
        self._is_moving = False

        self._register_parameters()
        self._register_capabilities()

    def _register_parameters(self) -> None:
        """Register stage-specific parameters."""
        self.register_parameter(
            "velocity",
            default_value=10.0,
            min_value=0.1,
            max_value=100.0,
            unit="mm/s",
            description="Stage movement velocity",
        )
        self.register_parameter(
            "acceleration",
            default_value=100.0,
            min_value=1.0,
            max_value=1000.0,
            unit="mm/s^2",
            description="Stage acceleration",
        )
        self.register_parameter(
            "backlash_compensation",
            default_value=0.01,
            min_value=0.0,
            max_value=1.0,
            unit="mm",
            description="Backlash compensation amount",
        )
        self.register_parameter(
            "home_position",
            default_value=self._travel_range[0],
            min_value=self._travel_range[0],
            max_value=self._travel_range[1],
            unit="mm",
            description="Home position",
        )

    def _register_capabilities(self) -> None:
        """Register stage capabilities."""
        self.register_capability(
            "move_to",
            description="Move to absolute position",
            parameters=["position"],
        )
        self.register_capability(
            "move_relative",
            description="Move relative to current position",
            parameters=["distance"],
        )
        self.register_capability(
            "home",
            description="Move to home position",
        )
        self.register_capability(
            "get_position",
            description="Get current position",
            return_type=float,
        )

    def open(self) -> None:
        """Open mock stage connection."""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.1)
        self._current_position = self._travel_range[0]
        self._set_state(DeviceState.READY)
        logger.info(f"Mock stage {self.device_id} opened (axis {self._axis})")

    def close(self) -> None:
        """Close mock stage connection."""
        self._is_moving = False
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock stage {self.device_id} closed")

    def is_connected(self) -> bool:
        """Check if stage is connected."""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """Get mock hardware information."""
        return {
            "serial_number": f"MOCK_STAGE_{self.device_id[:8]}",
            "firmware_version": "1.2.0-mock",
            "axis": self._axis,
            "travel_range_mm": self._travel_range,
            "resolution_um": 0.1,
            "encoder_type": "Incremental",
        }

    def move_to(self, position: float) -> None:
        """Move to absolute position.

        Args:
            position: Target position in mm.
        """
        if not self.is_connected():
            raise RuntimeError("Stage not connected")

        if not (self._travel_range[0] <= position <= self._travel_range[1]):
            raise ValueError(
                f"Position {position} mm out of range {self._travel_range}"
            )

        self._set_state(DeviceState.BUSY)
        self._is_moving = True
        try:
            velocity = self.get_parameter_value("velocity")
            distance = abs(position - self._current_position)
            move_time = distance / velocity

            # Simulate movement
            time.sleep(move_time)

            self._current_position = position
            logger.debug(f"Stage moved to {position:.3f} mm")
        finally:
            self._is_moving = False
            self._set_state(DeviceState.READY)

    def move_relative(self, distance: float) -> None:
        """Move relative to current position.

        Args:
            distance: Distance to move in mm (positive or negative).
        """
        self.move_to(self._current_position + distance)

    def home(self) -> None:
        """Move to home position."""
        home_pos = self.get_parameter_value("home_position")
        self.move_to(home_pos)

    def get_position(self) -> float:
        """Get current position in mm."""
        return self._current_position

    def is_moving(self) -> bool:
        """Check if stage is currently moving."""
        return self._is_moving

    def get_twin_state(self) -> dict[str, Any]:
        """Get state for digital twin synchronization."""
        state = super().get_twin_state()
        state["hardware"] = {
            "axis": self._axis,
            "travel_range": self._travel_range,
            "current_position": self._current_position,
            "is_moving": self._is_moving,
        }
        return state


class MockLaserError(DeviceError):
    """Exception raised for mock laser errors."""
    pass


class MockLaser(Device):
    """Mock laser device.

    Simulates a tunable laser source with power and wavelength control.

    Attributes:
        device_type: DeviceType.LASER
        manufacturer: "Mock"
        model: "Simulated Laser"

    Example:
        >>> laser = MockLaser(device_id="mock_laser_001")
        >>> with laser:
        ...     laser.set_power(10.0)
        ...     laser.set_wavelength(633.0)
        ...     laser.enable_output(True)
    """

    device_type = DeviceType.LASER
    manufacturer = "Mock"
    model = "Simulated Laser"

    def __init__(
        self,
        device_id: str = "",
        wavelength_range: tuple[float, float] = (400.0, 1100.0),
        power_range: tuple[float, float] = (0.0, 100.0),
    ):
        """Initialize mock laser.

        Args:
            device_id: Unique device identifier.
            wavelength_range: Min/max wavelength range in nm.
            power_range: Min/max power range in mW.
        """
        super().__init__(device_id)

        self._wavelength_range = wavelength_range
        self._power_range = power_range
        self._current_wavelength = wavelength_range[0]
        self._current_power = 0.0
        self._output_enabled = False
        self._temperature = 25.0

        self._register_parameters()
        self._register_capabilities()

    def _register_parameters(self) -> None:
        """Register laser-specific parameters."""
        self.register_parameter(
            "wavelength",
            default_value=self._wavelength_range[0],
            min_value=self._wavelength_range[0],
            max_value=self._wavelength_range[1],
            unit="nm",
            description="Laser wavelength",
        )
        self.register_parameter(
            "power",
            default_value=0.0,
            min_value=self._power_range[0],
            max_value=self._power_range[1],
            unit="mW",
            description="Laser output power",
        )
        self.register_parameter(
            "temperature",
            default_value=25.0,
            min_value=10.0,
            max_value=50.0,
            unit="C",
            description="Laser diode temperature",
        )
        self.register_parameter(
            "stability_mode",
            default_value=True,
            unit="",
            description="Enable power stability control",
        )

    def _register_capabilities(self) -> None:
        """Register laser capabilities."""
        self.register_capability(
            "enable_output",
            description="Enable/disable laser output",
            parameters=["enabled"],
        )
        self.register_capability(
            "get_power",
            description="Get current output power",
            return_type=float,
        )
        self.register_capability(
            "get_wavelength",
            description="Get current wavelength",
            return_type=float,
        )

    def open(self) -> None:
        """Open mock laser connection."""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.2)
        self._output_enabled = False
        self._set_state(DeviceState.READY)
        logger.info(f"Mock laser {self.device_id} opened")

    def close(self) -> None:
        """Close mock laser connection."""
        self._output_enabled = False
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock laser {self.device_id} closed")

    def is_connected(self) -> bool:
        """Check if laser is connected."""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """Get mock hardware information."""
        return {
            "serial_number": f"MOCK_LASER_{self.device_id[:8]}",
            "firmware_version": "2.1.0-mock",
            "wavelength_range_nm": self._wavelength_range,
            "power_range_mW": self._power_range,
            "beam_diameter_mm": 1.0,
            "beam_quality_m2": 1.05,
        }

    def enable_output(self, enabled: bool) -> None:
        """Enable or disable laser output.

        Args:
            enabled: True to enable output, False to disable.
        """
        if not self.is_connected():
            raise RuntimeError("Laser not connected")

        if enabled and self._current_power <= 0:
            logger.warning("Attempting to enable laser with zero power")

        self._output_enabled = enabled
        logger.info(f"Laser output {'enabled' if enabled else 'disabled'}")

    def set_power(self, power_mw: float) -> None:
        """Set laser power.

        Args:
            power_mw: Power in milliwatts.
        """
        if not self.is_connected():
            raise RuntimeError("Laser not connected")

        if not (self._power_range[0] <= power_mw <= self._power_range[1]):
            raise ValueError(
                f"Power {power_mw} mW out of range {self._power_range}"
            )

        self._set_state(DeviceState.BUSY)
        try:
            time.sleep(0.05)  # Simulate settling
            self._current_power = power_mw
            self.set_parameter_value("power", power_mw)
            logger.debug(f"Laser power set to {power_mw:.2f} mW")
        finally:
            self._set_state(DeviceState.READY)

    def set_wavelength(self, wavelength_nm: float) -> None:
        """Set laser wavelength.

        Args:
            wavelength_nm: Wavelength in nanometers.
        """
        if not self.is_connected():
            raise RuntimeError("Laser not connected")

        if not (self._wavelength_range[0] <= wavelength_nm <= self._wavelength_range[1]):
            raise ValueError(
                f"Wavelength {wavelength_nm} nm out of range {self._wavelength_range}"
            )

        self._set_state(DeviceState.BUSY)
        try:
            time.sleep(0.5)  # Simulate wavelength tuning
            self._current_wavelength = wavelength_nm
            self.set_parameter_value("wavelength", wavelength_nm)
            logger.debug(f"Laser wavelength set to {wavelength_nm:.2f} nm")
        finally:
            self._set_state(DeviceState.READY)

    def get_power(self) -> float:
        """Get current output power in mW."""
        if self._output_enabled:
            return self._current_power
        return 0.0

    def get_wavelength(self) -> float:
        """Get current wavelength in nm."""
        return self._current_wavelength

    def is_output_enabled(self) -> bool:
        """Check if laser output is enabled."""
        return self._output_enabled

    def _on_parameter_changed(self, name: str, old_value: Any, new_value: Any) -> None:
        """Handle parameter changes."""
        if name == "wavelength":
            self._current_wavelength = new_value
        elif name == "power":
            self._current_power = new_value

    def get_twin_state(self) -> dict[str, Any]:
        """Get state for digital twin synchronization."""
        state = super().get_twin_state()
        state["hardware"] = {
            "wavelength_range": self._wavelength_range,
            "power_range": self._power_range,
            "current_wavelength": self._current_wavelength,
            "current_power": self._current_power,
            "output_enabled": self._output_enabled,
        }
        return state


class MockFilterError(DeviceError):
    """Exception raised for mock filter errors."""
    pass


class MockFilter(Device):
    """Mock optical filter wheel device.

    Simulates a filter wheel with multiple filter positions.

    Attributes:
        device_type: DeviceType.FILTER
        manufacturer: "Mock"
        model: "Simulated Filter Wheel"

    Example:
        >>> filters = ["Open", "ND1", "ND2", "Red", "Green", "Blue"]
        >>> fw = MockFilter(device_id="mock_filter_001", filters=filters)
        >>> with fw:
        ...     fw.move_to_position(3)  # Move to "Red" filter
    """

    device_type = DeviceType.FILTER
    manufacturer = "Mock"
    model = "Simulated Filter Wheel"

    def __init__(
        self,
        device_id: str = "",
        filters: list[str] | None = None,
    ):
        """Initialize mock filter wheel.

        Args:
            device_id: Unique device identifier.
            filters: List of filter names. Defaults to 6-position wheel.
        """
        super().__init__(device_id)

        self._filters = filters or ["Open", "ND1", "ND2", "ND3", "Red", "Block"]
        self._n_positions = len(self._filters)
        self._current_position = 0
        self._is_moving = False

        self._register_parameters()
        self._register_capabilities()

    def _register_parameters(self) -> None:
        """Register filter-specific parameters."""
        self.register_parameter(
            "speed",
            default_value=1.0,
            min_value=0.5,
            max_value=5.0,
            unit="rev/s",
            description="Filter wheel rotation speed",
        )
        self.register_parameter(
            "settle_time_ms",
            default_value=100.0,
            min_value=10.0,
            max_value=1000.0,
            unit="ms",
            description="Settling time after move",
        )
        self.register_parameter(
            "home_on_startup",
            default_value=True,
            unit="",
            description="Auto-home on initialization",
        )

    def _register_capabilities(self) -> None:
        """Register filter capabilities."""
        self.register_capability(
            "move_to_position",
            description="Move to filter position",
            parameters=["position"],
        )
        self.register_capability(
            "move_to_filter",
            description="Move to filter by name",
            parameters=["filter_name"],
        )
        self.register_capability(
            "get_current_filter",
            description="Get current filter name",
            return_type=str,
        )

    def open(self) -> None:
        """Open mock filter connection."""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.1)

        if self.get_parameter_value("home_on_startup"):
            self._current_position = 0

        self._set_state(DeviceState.READY)
        logger.info(f"Mock filter wheel {self.device_id} opened ({self._n_positions} positions)")

    def close(self) -> None:
        """Close mock filter connection."""
        self._is_moving = False
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock filter wheel {self.device_id} closed")

    def is_connected(self) -> bool:
        """Check if filter wheel is connected."""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """Get mock hardware information."""
        return {
            "serial_number": f"MOCK_FILTER_{self.device_id[:8]}",
            "firmware_version": "1.0.0-mock",
            "n_positions": self._n_positions,
            "filters": self._filters,
            "position": self._current_position,
        }

    def move_to_position(self, position: int) -> None:
        """Move to filter position.

        Args:
            position: Filter position index (0-based).
        """
        if not self.is_connected():
            raise RuntimeError("Filter wheel not connected")

        if not (0 <= position < self._n_positions):
            raise ValueError(
                f"Position {position} out of range [0, {self._n_positions})"
            )

        self._set_state(DeviceState.BUSY)
        self._is_moving = True
        try:
            speed = self.get_parameter_value("speed")
            settle_ms = self.get_parameter_value("settle_time_ms")

            # Calculate rotation time (shortest path)
            distance = min(
                abs(position - self._current_position),
                self._n_positions - abs(position - self._current_position)
            )

            move_time = distance / (speed * self._n_positions)
            time.sleep(move_time + settle_ms / 1000.0)

            self._current_position = position
            logger.debug(f"Filter wheel moved to position {position} ({self._filters[position]})")
        finally:
            self._is_moving = False
            self._set_state(DeviceState.READY)

    def move_to_filter(self, filter_name: str) -> None:
        """Move to filter by name.

        Args:
            filter_name: Name of the filter to move to.
        """
        if filter_name not in self._filters:
            raise ValueError(f"Filter '{filter_name}' not found. Available: {self._filters}")

        position = self._filters.index(filter_name)
        self.move_to_position(position)

    def get_current_position(self) -> int:
        """Get current filter position index."""
        return self._current_position

    def get_current_filter(self) -> str:
        """Get current filter name."""
        return self._filters[self._current_position]

    def get_filter_list(self) -> list[str]:
        """Get list of available filters."""
        return self._filters.copy()

    def get_twin_state(self) -> dict[str, Any]:
        """Get state for digital twin synchronization."""
        state = super().get_twin_state()
        state["hardware"] = {
            "n_positions": self._n_positions,
            "filters": self._filters,
            "current_position": self._current_position,
            "current_filter": self.get_current_filter(),
        }
        return state


class MockADCError(DeviceError):
    """Exception raised for mock ADC errors."""
    pass


class MockADC(Device):
    """Mock NI DAQ ADC device for testing.

    Simulates analog voltage acquisition with optional noise and
    configurable baseline voltage.

    Attributes:
        device_type: DeviceType.OTHER
        manufacturer: "Mock"
        model: "Simulated ADC"

    Example:
        >>> adc = MockADC(device_id="mock_adc_001", noise_std=0.01)
        >>> with adc:
        ...     voltages = adc.read(samples=10)
        ...     mean_v = adc.read_mean()
    """

    device_type = DeviceType.OTHER
    manufacturer = "Mock"
    model = "Simulated ADC"

    def __init__(
        self,
        device_id: str = "",
        device_name: str = "Dev1",
        channel: str = "ai0",
        sample_rate: int = 5000,
        samples_per_channel: int = 10,
        base_voltage: float = 0.0,
        noise_std: float = 0.01,
        random_seed: int | None = None,
    ):
        """Initialize mock ADC.

        Args:
            device_id: Unique device identifier.
            device_name: Simulated NI DAQ device name.
            channel: Simulated analog input channel.
            sample_rate: Simulated sample rate (Hz).
            samples_per_channel: Number of samples per read.
            base_voltage: Baseline voltage output (V).
            noise_std: Standard deviation of Gaussian noise added to readings.
            random_seed: Random seed for reproducible output.
        """
        super().__init__(device_id)

        self._device_name = device_name
        self._channel = channel
        self._sample_rate = sample_rate
        self._samples_per_channel = samples_per_channel
        self._base_voltage = base_voltage
        self._noise_std = noise_std
        self._rng = np.random.default_rng(random_seed)

        self._register_parameters()

    def _register_parameters(self) -> None:
        """Register ADC-specific parameters."""
        self.register_parameter(
            "device_name",
            self._device_name,
            description="Simulated NI DAQ device name",
        )
        self.register_parameter(
            "channel",
            self._channel,
            description="Simulated analog input channel",
        )
        self.register_parameter(
            "sample_rate",
            self._sample_rate,
            min_value=1,
            max_value=1_000_000,
            unit="Hz",
            description="Simulated acquisition sample rate",
        )
        self.register_parameter(
            "samples_per_channel",
            self._samples_per_channel,
            min_value=1,
            max_value=1_000_000,
            description="Simulated samples per read",
        )
        self.register_parameter(
            "base_voltage",
            self._base_voltage,
            unit="V",
            description="Baseline voltage output",
        )
        self.register_parameter(
            "noise_std",
            self._noise_std,
            min_value=0.0,
            unit="V",
            description="Noise standard deviation",
        )

    def open(self) -> None:
        """Open mock ADC connection."""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.05)
        self._set_state(DeviceState.READY)
        logger.info(f"Mock ADC {self.device_id} opened ({self._device_name}/{self._channel})")

    def close(self) -> None:
        """Close mock ADC connection."""
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock ADC {self.device_id} closed")

    def is_connected(self) -> bool:
        """Check if ADC is connected."""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """Get mock hardware information."""
        return {
            "serial_number": f"MOCK_ADC_{self.device_id[:8]}",
            "firmware_version": "1.0.0-mock",
            "device_name": self._device_name,
            "channel": self._channel,
            "sample_rate": self._sample_rate,
            "samples_per_channel": self._samples_per_channel,
        }

    def read(self, samples: int | None = None) -> np.ndarray:
        """Read simulated voltage samples.

        Args:
            samples: Number of samples to return. Defaults to ``samples_per_channel``.

        Returns:
            1-D array of simulated voltage readings in volts.
        """
        if not self.is_connected():
            raise RuntimeError("ADC not connected")

        n = samples or self._samples_per_channel
        base = self.get_parameter_value("base_voltage")
        noise = self.get_parameter_value("noise_std")

        self._set_state(DeviceState.BUSY)
        try:
            time.sleep(0.001)  # Simulate read delay
            data = base + self._rng.normal(0, noise, n)
            return data.astype(np.float64)
        finally:
            self._set_state(DeviceState.READY)

    def read_mean(self, samples: int | None = None) -> float:
        """Read simulated voltage samples and return the mean.

        Args:
            samples: Number of samples. Defaults to ``samples_per_channel``.

        Returns:
            Mean voltage in volts.
        """
        return float(np.mean(self.read(samples=samples)))

    def get_twin_state(self) -> dict[str, Any]:
        """Get state for digital twin synchronization."""
        state = super().get_twin_state()
        state["hardware"] = {
            "device_name": self._device_name,
            "channel": self._channel,
            "sample_rate": self._sample_rate,
            "samples_per_channel": self._samples_per_channel,
            "base_voltage": self._base_voltage,
            "noise_std": self._noise_std,
        }
        return state
