"""Physical model data structures for AO-Shaping.

This package provides type-safe containers for the fundamental optical field
quantities used throughout the AO system — phase maps, amplitude maps, and
complex electromagnetic fields — together with metadata tracking and
conversion utilities to/from OOPAO objects.

Structure:
    __init__.py   Package re-exports
    metadata.py   FieldMetadata — timestamp + source provenance
    field.py      PhaseMap, AmplitudeMap, ComplexField + OOPAO conversion

Example:
    >>> from ao_shaping.model import PhaseMap, FieldMetadata
    >>> import numpy as np
    >>> phase = np.zeros((128, 128))
    >>> pm = PhaseMap(phase=phase, pitch_size_um=8.0,
    ...               metadata=FieldMetadata(source="Santec SLM-200 #1"))
    >>> pm.shape
    (128, 128)
"""

from ao_shaping.model.metadata import FieldMetadata
from ao_shaping.model.field import (
    AmplitudeMap,
    ComplexField,
    PhaseMap,
    closest_opt_band,
    oopao_available,
)
from ao_shaping.model.quantities import (
    BeamMetrics,
    CoordinateGrid2D,
    DmCommands,
    OPDMap,
    PSFImage,
    TurbulenceParameters,
    WavefrontStatistics,
    WfsSlopes,
    ZernikeCoefficients,
)

__all__ = [
    "FieldMetadata",
    "PhaseMap",
    "AmplitudeMap",
    "ComplexField",
    "closest_opt_band",
    "oopao_available",
    "OPDMap",
    "WfsSlopes",
    "DmCommands",
    "ZernikeCoefficients",
    "CoordinateGrid2D",
    "BeamMetrics",
    "TurbulenceParameters",
    "PSFImage",
    "WavefrontStatistics",
]

__version__ = "1.0.0"
