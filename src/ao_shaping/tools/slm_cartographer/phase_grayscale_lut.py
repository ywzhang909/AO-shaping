"""Phase-to-grayscale LUT calibration using center cosine pattern.

Builds a lookup table by measuring the phase-grayscale response across
the dynamic range, using the center cosine pattern to minimize crosstalk.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.drivers.wfs.thorlab_wfs import ThorlabWFS, MlaRes


@dataclass
class LUTCalibrationConfig:
    """Configuration for LUT calibration.

    Attributes:
        slm_wavelength_nm: SLM operating wavelength.
        wavelength_nm: WFS measurement wavelength (should match SLM).
        grayscale_range: (min, max) grayscale values to sweep.
        step: Step size in grayscale values.
        n_averages: Number of WFS frames to average per grayscale value.
        use_center_cosine: Use center cosine pattern (paper method) if True,
            else use traditional gradient pattern.
        cosine_radius_px: Radius of the cosine pattern region (pixels).
        output_dir: Directory for saving calibration results.
        slm_number: SLM device number.
        mla_resolution: WFS MLA resolution.
    """

    slm_wavelength_nm: int = 532
    wavelength_nm: float = 532.0
    grayscale_range: tuple[int, int] = (0, 1023)
    step: int = 16
    n_averages: int = 10
    use_center_cosine: bool = True
    cosine_radius_px: float = 40.0
    output_dir: str = "data/slm_cartographer/calibration"
    slm_number: int = 1
    mla_resolution: str = "768"


@dataclass
class LUTCalibrationResult:
    """Results from LUT calibration.

    Attributes:
        grayscale_values: Array of tested grayscale values.
        measured_phases: Array of measured phases at each grayscale (radians).
        measured_phases_2pi: Array of measured phases in units of 2π.
        lut: Dictionary mapping grayscale -> phase (radians).
        cosine_pattern: Generated cosine pattern (if used).
        gradient_pattern: Generated gradient pattern (if used).
        phase_at_peak: Phase at maximum point of cosine pattern (radians).
        max_phase_2pi: Maximum measured phase in units of 2π.
        peak_grayscale: Grayscale value at cosine pattern peak.
        fit_coefficients: Polynomial fit coefficients for phase(grayscale).
        wavelength_nm: Wavelength used during calibration.
        timestamp: ISO timestamp of calibration.
        config_snapshot: Configuration used for calibration.
        measurements: Raw measurement data per grayscale step.
    """

    grayscale_values: list[int]
    measured_phases: list[float]
    measured_phases_2pi: list[float]
    lut: dict[int, float]
    cosine_pattern: np.ndarray | None = None
    gradient_pattern: np.ndarray | None = None
    phase_at_peak: float = 0.0
    max_phase_2pi: float = 0.0
    peak_grayscale: int = 0
    fit_coefficients: list[float] = field(default_factory=list)
    wavelength_nm: float = 532.0
    timestamp: str = ""
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    measurements: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ["cosine_pattern", "gradient_pattern"]:
            if d.get(key) is not None:
                d[key] = d[key].tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LUTCalibrationResult:
        for key in ["cosine_pattern", "gradient_pattern"]:
            if d.get(key) is not None:
                d[key] = np.array(d[key])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"LUT calibration result saved: {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> LUTCalibrationResult:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_phase(self, grayscale: int) -> float:
        """Get phase (radians) for a given grayscale via interpolation of LUT."""
        if not self.lut:
            return 0.0

        gs_vals = sorted(self.lut.keys())
        phases = [self.lut[g] for g in gs_vals]

        if grayscale <= gs_vals[0]:
            return phases[0]
        if grayscale >= gs_vals[-1]:
            return phases[-1]

        return float(np.interp(grayscale, gs_vals, phases))

    def get_grayscale_for_phase(self, target_phase_rad: float) -> int:
        """Get grayscale value that produces a target phase.

        Inverse lookup using linear interpolation.
        """
        if not self.lut:
            return 0

        gs_vals = sorted(self.lut.keys())
        phases = [self.lut[g] for g in gs_vals]

        if target_phase_rad <= phases[0]:
            return gs_vals[0]
        if target_phase_rad >= phases[-1]:
            return gs_vals[-1]

        gs_float = np.interp(target_phase_rad, phases, gs_vals)
        return int(np.clip(round(gs_float), gs_vals[0], gs_vals[-1]))


class PhaseGrayscaleLUT:
    """Phase-to-grayscale LUT calibration using Hartmann-Shack WFS.

    Implements the paper's calibration methodology:
    1. Generate center cosine grayscale pattern on SLM
    2. Capture Hartmann spots with WFS
    3. Reconstruct phase from centroid displacements
    4. Build LUT from measured phases vs grayscale values

    Args:
        slm: Connected SantecSLM200 instance.
        wfs: Connected ThorlabWFS instance.
        config: Calibration configuration.
        storage_dir: Directory for saving results.
    """

    def __init__(
        self,
        slm: SantecSLM200,
        wfs: ThorlabWFS,
        config: LUTCalibrationConfig | None = None,
        storage_dir: str | Path = "data/slm_cartographer/calibration",
    ):
        self.slm = slm
        self.wfs = wfs
        self.config = config if config is not None else LUTCalibrationConfig()
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        from ao_shaping.tools.slm_cartographer.cosine_pattern import (
            CosinePatternConfig,
            generate_center_cosine_pattern,
            get_pattern_peak_position,
            estimate_phase_from_pattern,
        )

        self._generate_cosine = generate_center_cosine_pattern
        self._get_peak = get_pattern_peak_position
        self._estimate_phase = estimate_phase_from_pattern
        self._CosinePatternConfig = CosinePatternConfig

    def calibrate(self, progress_callback=None) -> LUTCalibrationResult:
        """Run the full LUT calibration procedure.

        Steps:
        1. Configure WFS
        2. Capture reference (zero grayscale baseline)
        3. For each grayscale value:
           a. Display appropriate pattern
           b. Measure centroids with WFS
           c. Reconstruct phase via Fourier method
        4. Build LUT and fit polynomial

        Args:
            progress_callback: Optional callable(current, total, grayscale, message).

        Returns:
            LUTCalibrationResult with complete lookup table.
        """
        from ao_shaping.tools.slm_cartographer.hartmann_capture import (
            HartmannCapture,
            HartmannCaptureConfig,
            HartmannMeasurement,
        )
        from ao_shaping.tools.slm_cartographer.wavefront_reconstruction import (
            FourierWavefrontReconstructor,
            FourierReconstructionConfig,
        )
        from ao_shaping.tools.slm_cartographer.cosine_pattern import (
            generate_center_cosine_pattern,
            generate_traditional_gradient_pattern,
        )

        assert self.slm.is_open, "SLM must be opened"
        assert self.wfs.is_connected(), "WFS must be opened"

        logger.info("Starting LUT calibration...")

        # Configure WFS
        wfs_config = HartmannCaptureConfig(
            n_averages=self.config.n_averages,
            exposure_time_ms=0.0,
            mla_resolution=self.config.mla_resolution,
            cancel_tilt=True,
        )
        capture = HartmannCapture(
            config=wfs_config,
            wfs_device=self.wfs,
            storage_dir=str(self._storage_dir / "raw"),
        )
        capture.configure_wfs()

        # Generate cosine pattern (fixed, used for all measurements)
        pattern_config = self._CosinePatternConfig(
            center_x=self.slm.Panel_Res[0] // 2,
            center_y=self.slm.Panel_Res[1] // 2,
            radius_pixels=self.config.cosine_radius_px,
            max_phase_2pi=1.0,
        )
        cosine_pattern = generate_center_cosine_pattern(
            config=pattern_config,
            output_resolution=self.slm.Panel_Res,
        )
        peak_y, peak_x = self._get_peak(cosine_pattern)

        # Also create gradient pattern for comparison
        gradient_pattern = generate_traditional_gradient_pattern(
            config=pattern_config,
            output_resolution=self.slm.Panel_Res,
        )

        # Compute nominal phase from cosine pattern
        nominal_phase = self._estimate_phase(cosine_pattern)

        # Capture reference (zero grayscale)
        logger.info("Capturing reference (zero grayscale)...")
        self.slm.set_grayscale(0)
        time.sleep(0.3)
        ref_measurement = capture.capture_reference("lut_flat")

        # Initialize reconstructor
        num_spots = (self.wfs.num_spots_x, self.wfs.num_spots_y)
        recon_config = FourierReconstructionConfig(
            wavelength_nm=self.config.wavelength_nm,
            lenslet_pitch_mm=0.15,
        )
        reconstructor = FourierWavefrontReconstructor(
            config=recon_config,
            num_spots=num_spots,
        )

        # Sweep grayscale values
        gs_min, gs_max = self.config.grayscale_range
        grayscale_values = list(range(gs_min, gs_max + 1, self.config.step))

        measured_phases: list[float] = []
        measured_phases_2pi: list[float] = []
        all_measurements: list[dict[str, Any]] = []

        total = len(grayscale_values)

        for idx, gs in enumerate(grayscale_values):
            if progress_callback:
                progress_callback(idx, total, gs, f"Measuring grayscale {gs}")

            # Display pattern modified by current grayscale offset
            # For LUT: display cosine + uniform grayscale offset
            display_pattern = np.clip(
                cosine_pattern.astype(np.float64) + gs, 0, 1023
            ).astype(np.uint16)

            self.slm.write_phase(display_pattern, memory_number=1)
            self.slm.display_memory(1)
            time.sleep(0.2)

            try:
                measurement = capture.capture_measurement(
                    label=f"lut_gs{gs}",
                    wavefront_device=self.wfs,
                )

                if measurement.wavefront is not None:
                    # Extract phase at the peak location
                    wf = measurement.wavefront
                    py = int(np.clip(peak_y, 0, wf.shape[0] - 1))
                    px = int(np.clip(peak_x, 0, wf.shape[1] - 1))

                    phase_at_peak = float(wf[py, px])

                    # Also compute RMS over the valid pupil region
                    rms = float(np.nanstd(wf))

                    measured_phases.append(phase_at_peak)
                    measured_phases_2pi.append(phase_at_peak / (2.0 * np.pi))
                    all_measurements.append(
                        {
                            "grayscale": gs,
                            "phase_at_peak_rad": phase_at_peak,
                            "phase_at_peak_2pi": phase_at_peak / (2.0 * np.pi),
                            "wavefront_rms": rms,
                            "mean_displacement_dx": float(
                                np.nanmean(measurement.displacements_x)
                            ),
                            "mean_displacement_dy": float(
                                np.nanmean(measurement.displacements_y)
                            ),
                        }
                    )
                else:
                    measured_phases.append(0.0)
                    measured_phases_2pi.append(0.0)
                    all_measurements.append(
                        {
                            "grayscale": gs,
                            "phase_at_peak_rad": 0.0,
                            "phase_at_peak_2pi": 0.0,
                            "error": "Wavefront reconstruction failed",
                        }
                    )

            except Exception as e:
                logger.warning(f"Measurement failed at gs={gs}: {e}")
                measured_phases.append(0.0)
                measured_phases_2pi.append(0.0)
                all_measurements.append(
                    {
                        "grayscale": gs,
                        "error": str(e),
                    }
                )

        # Build LUT dictionary
        lut: dict[int, float] = {}
        for gs, phase in zip(grayscale_values, measured_phases):
            lut[gs] = phase

        # Fit polynomial: phase = f(grayscale)
        fit_coeffs: list[float] = []
        if len(measured_phases) > 3:
            try:
                coeffs = np.polyfit(grayscale_values, measured_phases, deg=3)
                fit_coeffs = coeffs.tolist()
                logger.info(
                    f"Polynomial fit: phase = {coeffs[0]:.2e}·g³ + "
                    f"{coeffs[1]:.2e}·g² + {coeffs[2]:.2e}·g + {coeffs[3]:.2e}"
                )
            except Exception as e:
                logger.warning(f"Polynomial fit failed: {e}")

        # Compute peak phase (from cosine pattern center)
        peak_phase = (
            float(nominal_phase[peak_y, peak_x])
            if peak_y < nominal_phase.shape[0] and peak_x < nominal_phase.shape[1]
            else 0.0
        )
        max_phase_2pi = (
            float(np.max(measured_phases_2pi)) if measured_phases_2pi else 0.0
        )

        result = LUTCalibrationResult(
            grayscale_values=grayscale_values,
            measured_phases=measured_phases,
            measured_phases_2pi=measured_phases_2pi,
            lut=lut,
            cosine_pattern=cosine_pattern,
            gradient_pattern=gradient_pattern,
            phase_at_peak=peak_phase,
            max_phase_2pi=max_phase_2pi,
            peak_grayscale=int(cosine_pattern[peak_y, peak_x]),
            fit_coefficients=fit_coeffs,
            wavelength_nm=self.config.wavelength_nm,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            config_snapshot=asdict(self.config),
            measurements=all_measurements,
        )

        # Save result
        result.save(
            self._storage_dir
            / f"lut_calibration_{self.config.wavelength_nm}nm_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )

        logger.info(
            f"LUT calibration complete: {total} points, "
            f"max_phase={max_phase_2pi:.3f}×2π"
        )

        return result
