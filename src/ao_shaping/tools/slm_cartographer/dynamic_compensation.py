"""Dynamic aberration compensation using measured SLM phase response.

Implements the closed-loop compensation workflow from the paper:
1. Measure dynamic aberration (flat-zero grayscale distortion)
2. Generate compensation pattern from inverse response
3. Apply to SLM and verify reduction in wavefront RMS

The center cosine pattern's peak point serves as a spatial reference
for defining the compensation region.
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
from ao_shaping.drivers.wfs.thorlab_wfs import ThorlabWFS


@dataclass
class CompensationConfig:
    """Configuration for dynamic aberration compensation.

    Attributes:
        slm_wavelength_nm: SLM operating wavelength (nm).
        n_averages: Number of frames to average per measurement.
        n_correction_iterations: Number of closed-loop correction iterations.
        convergence_threshold_rms_waves: Stop if corrected RMS below this.
        pupil_diameter_mm: WFS pupil diameter (mm).
        mla_resolution: WFS MLA resolution index (e.g., "768").
        slm_number: SLM device number (1-8).
        cosine_radius_px: Radius of cosine region (matches calibration pattern).
    """

    slm_wavelength_nm: int = 532
    n_averages: int = 10
    n_correction_iterations: int = 1
    convergence_threshold_rms_waves: float = 0.05
    pupil_diameter_mm: float = 3.0
    mla_resolution: str = "768"
    slm_number: int = 1
    cosine_radius_px: float = 40.0


@dataclass
class CompensationResult:
    """Results from aberration compensation run.

    Attributes:
        initial_wavefront_rms: RMS before correction (waves).
        final_wavefront_rms: RMS after correction (waves).
        initial_wavefront_peak_to_valley: PV before correction.
        final_wavefront_peak_to_valley: PV after correction.
        measured_aberration: Measured wavefront (phase to compensate).
        compensation_phase: Computed compensation phase map.
        compensation_grayscale: Compensation pattern in SLM grayscale values.
        iterations_used: Number of correction iterations performed.
        converged: Whether convergence threshold was reached.
        lut: Phase-grayscale LUT used for inverse mapping.
        timestamp: ISO timestamp.
        per_iteration_data: Detailed data from each correction iteration.
    """

    initial_wavefront_rms: float = 0.0
    final_wavefront_rms: float = 0.0
    initial_wavefront_peak_to_valley: float = 0.0
    final_wavefront_peak_to_valley: float = 0.0
    measured_aberration: np.ndarray | None = None
    compensation_phase: np.ndarray | None = None
    compensation_grayscale: np.ndarray | None = None
    iterations_used: int = 0
    converged: bool = False
    lut: dict[int, float] | None = None
    timestamp: str = ""
    per_iteration_data: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k, v in asdict(self).items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CompensationResult:
        for k in [
            "measured_aberration",
            "compensation_phase",
            "compensation_grayscale",
        ]:
            if d.get(k) is not None:
                d[k] = np.array(d[k])
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**kwargs)


class DynamicCompensator:
    """Dynamic aberration compensator for SLM wavefront correction.

    Implements the paper's closed-loop compensation:
    1. Measure dynamic wavefront aberration (with flat/zero grayscale)
    2. Invert the measured aberration to get compensation phase
    3. Use the phase-grayscale LUT to compute the compensation pattern
    4. Display on SLM and re-measure to verify

    The cosine pattern peak defines the central reference region for
    matching compensation geometry to the measured aberration.

    Args:
        slm: Connected SantecSLM200 instance.
        wfs: Connected ThorlabWFS instance.
        config: Compensation configuration.
        lut: Phase-grayscale LUT (grayscale -> phase in radians).
        storage_dir: Directory for saving results.
    """

    def __init__(
        self,
        slm: SantecSLM200,
        wfs: Any,
        config: CompensationConfig | None = None,
        lut: dict[int, float] | None = None,
        storage_dir: str | Path = "data/slm_cartographer/compensation",
    ):
        self.slm = slm
        self.wfs = wfs
        self.config = config if config is not None else CompensationConfig()
        self.lut = lut
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._cosine_pattern: np.ndarray | None = None
        self._peak_position: tuple[int, int] | None = None

    def set_lut(self, lut: dict[int, float]) -> None:
        self.lut = lut
        logger.info(f"LUT updated: {len(lut)} points")

    def _load_capture_dependencies(self) -> Any:
        from ao_shaping.tools.slm_cartographer.hartmann_capture import (
            HartmannCapture,
            HartmannCaptureConfig,
        )

        return HartmannCapture, HartmannCaptureConfig

    def measure_initial_aberration(self) -> np.ndarray:
        """Measure initial wavefront aberration with flat SLM (zero grayscale).

        Returns:
            Measured wavefront array (radians or waves, per WFS convention).
        """
        assert self.slm.is_open, "SLM must be opened"
        assert self.wfs.is_connected(), "WFS must be connected"

        logger.info("Measuring initial aberration (flat zero grayscale)...")

        self.slm.set_grayscale(0)
        time.sleep(0.5)

        HartmannCapture, HartmannCaptureConfig = self._load_capture_dependencies()

        wfs_config = HartmannCaptureConfig(
            n_averages=self.config.n_averages,
            exposure_time_ms=0.0,
            mla_resolution=self.config.mla_resolution,
            cancel_tilt=True,
            pupil_diameter_mm=self.config.pupil_diameter_mm,
        )
        capture = HartmannCapture(
            config=wfs_config,
            wfs_device=self.wfs,
            storage_dir=str(self._storage_dir / "raw"),
        )
        capture.configure_wfs()

        ref = capture.capture_reference("aberration_flat")
        measurement = capture.capture_measurement(
            "aberration_flat",
            wavefront_device=self.wfs,
        )

        if measurement.wavefront is not None:
            wf = measurement.wavefront
            logger.info(f"Initial aberration RMS: {float(np.nanstd(wf)):.4f}")
            return wf
        h, w = self.slm.Panel_Res[1], self.slm.Panel_Res[0]
        return np.zeros((h, w))

    def compute_compensation(
        self,
        measured_wavefront: np.ndarray,
    ) -> np.ndarray:
        """Compute compensation grayscale pattern from measured wavefront.

        Compensation phase inverts the sign of the measured aberration,
        then maps to grayscale via the inverse LUT. Only applies within
        the cosine region (defined by peak position and radius).

        Args:
            measured_wavefront: Measured wavefront from WFS.

        Returns:
            Compensation grayscale array (uint16) for SLM display.
        """
        assert self.lut is not None, "LUT must be set before compute_compensation()"
        assert self.slm.Panel_Res is not None

        logger.info("Computing compensation pattern...")

        h, w = self.slm.Panel_Res[1], self.slm.Panel_Res[0]

        if measured_wavefront.shape != (h, w):
            from scipy.ndimage import zoom

            zoom_y = h / measured_wavefront.shape[0]
            zoom_x = w / measured_wavefront.shape[1]
            measured_wavefront = zoom(measured_wavefront, (zoom_y, zoom_x), order=3)

        compensation_phase = -measured_wavefront.copy()
        compensation_phase = compensation_phase - np.nanmean(compensation_phase)

        if self._cosine_pattern is not None and self._peak_position is not None:
            peak_y, peak_x = self._peak_position
        else:
            peak_y, peak_x = h // 2, w // 2

        radius = self.config.cosine_radius_px
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        r = np.sqrt((x_coords - peak_x) ** 2 + (y_coords - peak_y) ** 2)
        window = np.where(r <= radius, (1.0 + np.cos(np.pi * r / radius)) / 2.0, 0.0)

        compensation_phase = compensation_phase * window

        gs_values = np.array(sorted(self.lut.keys()))
        phase_values = np.array([self.lut[g] for g in gs_values])

        max_ph = float(np.max(np.abs(phase_values))) if len(phase_values) else 1.0
        if max_ph > 0:
            phase_norm = np.clip(compensation_phase / (2.0 * np.pi), -1.0, 1.0)
            comp_gs = np.interp(phase_norm, phase_values / (2.0 * np.pi), gs_values)
        else:
            comp_gs = np.zeros_like(compensation_phase)

        comp_gs = np.clip(comp_gs, 0, SantecSLM200.MAX_GRAYSCALE_VALUE).astype(
            np.uint16
        )

        logger.info(
            f"Compensation grayscale range: {int(comp_gs.min())} - {int(comp_gs.max())}"
        )
        return comp_gs

    def apply_compensation(
        self,
        compensation_grayscale: np.ndarray,
        memory_slot: int = 2,
    ) -> None:
        """Display compensation pattern on SLM.

        Args:
            compensation_grayscale: 2D uint16 grayscale array.
            memory_slot: Target SLM memory slot (1-128).
        """
        assert self.slm.is_open, "SLM must be opened"
        logger.info(f"Applying compensation to SLM memory slot {memory_slot}...")
        self.slm.write_phase(compensation_grayscale, memory_number=memory_slot)
        self.slm.display_memory(memory_slot)
        time.sleep(0.3)
        logger.info("Compensation applied.")

    def verify_correction(self) -> tuple[np.ndarray, dict[str, float]]:
        """Re-measure wavefront after compensation.

        Returns:
            (corrected_wavefront, corrected_stats) tuple.
        """
        assert self.wfs.is_connected(), "WFS must be connected"
        logger.info("Verifying correction...")

        HartmannCapture, HartmannCaptureConfig = self._load_capture_dependencies()

        wfs_config = HartmannCaptureConfig(
            n_averages=self.config.n_averages,
            exposure_time_ms=0.0,
            mla_resolution=self.config.mla_resolution,
            cancel_tilt=True,
            pupil_diameter_mm=self.config.pupil_diameter_mm,
        )
        capture = HartmannCapture(
            config=wfs_config,
            wfs_device=self.wfs,
            storage_dir=str(self._storage_dir / "raw"),
        )
        capture.configure_wfs()
        measurement = capture.capture_measurement(
            "post_compensation",
            wavefront_device=self.wfs,
        )

        wf = (
            measurement.wavefront
            if measurement.wavefront is not None
            else np.zeros(self.slm.Panel_Res[::-1])
        )
        if wf is None or not np.isfinite(wf).all():
            wf = np.zeros(self.slm.Panel_Res[::-1])
        stats = measurement.wavefront_stats
        logger.info(f"Corrected RMS: {stats.get('rms', float(np.nanstd(wf)))}")
        return wf, stats

    def compensate_once(
        self,
        max_iterations: int | None = None,
    ) -> CompensationResult:
        """Run a single closed-loop compensation pass.

        Measures aberration, computes and applies compensation, verifies improvement.

        Args:
            max_iterations: Override iteration count from config.

        Returns:
            CompensationResult with all measurements and outputs.
        """
        assert self.lut is not None, "LUT must be set"

        max_iter = int(
            max_iterations
            if max_iterations is not None
            else self.config.n_correction_iterations
        )

        logger.info("Starting dynamic aberration compensation...")

        from ao_shaping.tools.slm_cartographer.cosine_pattern import (
            CosinePatternConfig,
            generate_center_cosine_pattern,
            get_pattern_peak_position,
        )

        pattern_config = CosinePatternConfig(
            center_x=self.slm.Panel_Res[0] // 2,
            center_y=self.slm.Panel_Res[1] // 2,
            radius_pixels=self.config.cosine_radius_px,
            max_phase_2pi=1.0,
        )
        self._cosine_pattern = generate_center_cosine_pattern(
            config=pattern_config,
            output_resolution=self.slm.Panel_Res,
        )
        self._peak_position = get_pattern_peak_position(self._cosine_pattern)

        initial_wf = self.measure_initial_aberration()
        initial_rms = float(np.nanstd(initial_wf))
        initial_pv = float(np.nanmax(initial_wf) - np.nanmin(initial_wf))

        per_iteration: list[dict[str, Any]] = []
        final_rms = initial_rms
        final_pv = initial_pv
        converged = False
        iterations_used = 0

        current_wf = initial_wf
        comp_gs_final: np.ndarray | None = None

        for i in range(max_iter):
            iterations_used = i + 1

            wf_before = current_wf
            rms_before = float(np.nanstd(wf_before))

            comp_gs = self.compute_compensation(wf_before)
            comp_gs_final = comp_gs
            self.apply_compensation(comp_gs, memory_slot=2)
            corrected_wf, corrected_stats = self.verify_correction()
            rms_after = corrected_stats.get("rms", float(np.nanstd(corrected_wf)))
            pv_after = float(np.nanmax(corrected_wf) - np.nanmin(corrected_wf))

            per_iteration.append(
                {
                    "iteration": i + 1,
                    "rms_before_waves": rms_before,
                    "rms_after_waves": float(rms_after),
                    "pv_before_waves": float(
                        np.nanmax(wf_before) - np.nanmin(wf_before)
                    ),
                    "pv_after_waves": pv_after,
                    "improvement_rms": rms_before - float(rms_after),
                }
            )

            current_wf = corrected_wf
            final_rms = float(rms_after)
            final_pv = pv_after

            if final_rms < self.config.convergence_threshold_rms_waves:
                converged = True
                break

            time.sleep(0.2)

        result = CompensationResult(
            initial_wavefront_rms=initial_rms,
            final_wavefront_rms=final_rms,
            initial_wavefront_peak_to_valley=initial_pv,
            final_wavefront_peak_to_valley=final_pv,
            measured_aberration=initial_wf,
            compensation_grayscale=comp_gs_final,
            iterations_used=iterations_used,
            converged=converged,
            lut=self.lut,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            per_iteration_data=per_iteration,
        )

        result_path = (
            self._storage_dir / f"compensation_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(
            f"Compensation done: RMS {initial_rms:.4f} -> {final_rms:.4f} waves, "
            f"{iterations_used} iterations, converged={converged}"
        )

        return result
