"""SLM Cartographer - Hartmann-Shack based SLM calibration and aberration correction.

This package implements the methodology from the paper "From Hartmann Spots to SLM Phase Maps:
A WFS-based calibration and dynamic aberration compensation method" using:
- Center cosine grayscale pattern for reduced pixel crosstalk
- Thorlabs WFS40-7AR wavefront sensor
- Santec SLM (HDSLM80R equivalent)
- Fourier wavefront reconstruction from centroid displacements

Usage:
    streamlit run src/ao_shaping/tools/slm/cartographer/slm_cartographer_ui.py

Or as a module:
    python -m ao_shaping.tools.slm.cartographer

NOTE: 本包的 phase_grayscale_lut 是 WFS 实测研究用 LUT, 不可被
``Santec.load_lut()`` 消费; 驱动层 canonical LUT 见 slm-lut 管线
(``tools/slm/slm_lut_runner.py`` + ``utils/slm_lut.py``)。
"""

from __future__ import annotations

from ao_shaping.tools.slm.cartographer.cosine_pattern import (
    generate_center_cosine_pattern,
    generate_traditional_gradient_pattern,
    estimate_phase_from_pattern,
    CosinePatternConfig,
)
from ao_shaping.tools.slm.cartographer.hartmann_capture import (
    HartmannCapture,
    HartmannCaptureConfig,
    HartmannMeasurement,
)
from ao_shaping.tools.slm.cartographer.wavefront_reconstruction import (
    FourierWavefrontReconstructor,
    FourierReconstructionConfig,
    reconstruct_from_displacements,
    interpolate_sparse_to_dense,
)
from ao_shaping.tools.slm.cartographer.phase_grayscale_lut import (
    PhaseGrayscaleLUT,
    LUTCalibrationConfig,
    LUTCalibrationResult,
)
from ao_shaping.tools.slm.cartographer.dynamic_compensation import (
    DynamicCompensator,
    CompensationConfig,
    CompensationResult,
)

__all__ = [
    "CosinePatternConfig",
    "generate_center_cosine_pattern",
    "generate_traditional_gradient_pattern",
    "estimate_phase_from_pattern",
    "HartmannCaptureConfig",
    "HartmannMeasurement",
    "HartmannCapture",
    "FourierReconstructionConfig",
    "FourierWavefrontReconstructor",
    "reconstruct_from_displacements",
    "interpolate_sparse_to_dense",
    "LUTCalibrationConfig",
    "LUTCalibrationResult",
    "PhaseGrayscaleLUT",
    "CompensationConfig",
    "CompensationResult",
    "DynamicCompensator",
]
