"""Typed physical quantities used at device and simulation boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np

from ao_shaping.model.field import AmplitudeMap, PhaseMap, _validate_2d_array
from ao_shaping.model.metadata import FieldMetadata
from ao_shaping.model.spatial import centered_coordinates, resample_physical_grid


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 1-D array")
    return result


@dataclass
class OPDMap:
    """Optical path difference in metres at a specified wavelength."""

    opd: np.ndarray
    pitch_size_um: float
    wavelength_m: float
    metadata: FieldMetadata = field(default_factory=FieldMetadata)

    def __post_init__(self) -> None:
        self.opd = _validate_2d_array(self.opd, "opd", dtype=np.float64)
        self.pitch_size_um = _positive(self.pitch_size_um, "pitch_size_um")
        self.wavelength_m = _positive(self.wavelength_m, "wavelength_m")

    @property
    def shape(self) -> tuple[int, int]:
        return self.opd.shape

    def to_phase(self) -> PhaseMap:
        """Convert metres of OPD to unwrapped radians."""
        return PhaseMap(self.opd * (2 * np.pi / self.wavelength_m), self.pitch_size_um, self.metadata)

    def resample_to(self, shape: tuple[int, int], pitch_size_um: float, *, order: int = 1) -> OPDMap:
        """Interpolate metre-valued OPD on a centred physical grid."""
        opd = resample_physical_grid(self.opd, self.pitch_size_um, shape, pitch_size_um, order=order)
        return OPDMap(opd, pitch_size_um, self.wavelength_m, self.metadata)

    @classmethod
    def from_phase(cls, phase: PhaseMap, wavelength_m: float) -> OPDMap:
        """Convert an unwrapped phase map to metres of OPD."""
        return cls(phase.to_opd(wavelength_m), phase.pitch_size_um, wavelength_m, phase.metadata)

    @classmethod
    def from_oopao_source(cls, source: Any, pitch_size_um: float) -> OPDMap:
        """Read OOPAO's metre-valued OPD without an implicit wavelength fallback."""
        return cls(source.OPD, pitch_size_um, source.wavelength)


@dataclass
class WfsSlopes:
    """Paired Shack-Hartmann slopes in arcseconds or radians."""

    sx: np.ndarray
    sy: np.ndarray
    n_subaperture: int
    units: Literal["arcsec", "rad"] = "arcsec"
    valid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.sx = _vector(self.sx, "sx")
        self.sy = _vector(self.sy, "sy")
        if self.n_subaperture <= 0 or self.sx.size != self.n_subaperture**2 or self.sy.size != self.sx.size:
            raise ValueError("sx and sy must each contain n_subaperture**2 values")
        if self.units not in ("arcsec", "rad"):
            raise ValueError("units must be 'arcsec' or 'rad'")
        if self.valid_mask is not None:
            self.valid_mask = np.array(self.valid_mask, dtype=bool, copy=True)
            if self.valid_mask.shape != self.sx.shape:
                raise ValueError("valid_mask must match the slope arrays")

    @property
    def total_subapertures(self) -> int:
        return self.sx.size

    def to_flat(self) -> np.ndarray:
        """Return sx followed by sy, preserving the WFS flat-array convention."""
        return np.concatenate((self.sx, self.sy))

    def to_units(self, units: Literal["arcsec", "rad"]) -> WfsSlopes:
        """Convert both slope arrays between angular units."""
        if units not in ("arcsec", "rad"):
            raise ValueError("units must be 'arcsec' or 'rad'")
        factor = 1.0 if units == self.units else (np.pi / (180 * 3600) if units == "rad" else 180 * 3600 / np.pi)
        return WfsSlopes(self.sx * factor, self.sy * factor, self.n_subaperture, units, self.valid_mask)

    @classmethod
    def from_flat(cls, values: np.ndarray, n_subaperture: int, **kwargs: Any) -> WfsSlopes:
        """Split a flat array containing sx then sy."""
        data = _vector(values, "slopes")
        count = n_subaperture**2
        if data.size != 2 * count:
            raise ValueError("flat slopes must contain 2 * n_subaperture**2 values")
        return cls(data[:count], data[count:], n_subaperture, **kwargs)


@dataclass
class DmCommands:
    """Voltage commands with the target DM's actuator count and limits."""

    voltages: np.ndarray
    range_min: float
    range_max: float
    n_actuators: int
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        self.voltages = _vector(self.voltages, "voltages")
        if self.n_actuators <= 0 or self.voltages.size != self.n_actuators:
            raise ValueError("voltages must match n_actuators")
        if np.isnan(self.range_min) or np.isnan(self.range_max) or self.range_min >= self.range_max:
            raise ValueError("range_min must be less than range_max")

    def clipped(self) -> DmCommands:
        """Return a new command with voltages limited to the declared range."""
        return DmCommands(np.clip(self.voltages, self.range_min, self.range_max), self.range_min, self.range_max, self.n_actuators, self.timestamp)


@dataclass
class ZernikeCoefficients:
    """Zernike amplitudes with explicit one-based Noll indices and units."""

    coefficients: np.ndarray
    indices: np.ndarray
    unit: Literal["rad", "waves"] = "rad"
    metadata: FieldMetadata = field(default_factory=FieldMetadata)

    def __post_init__(self) -> None:
        self.coefficients = _vector(self.coefficients, "coefficients")
        self.indices = np.array(self.indices, copy=True)
        if self.indices.ndim != 1 or self.indices.shape != self.coefficients.shape or not np.issubdtype(self.indices.dtype, np.integer):
            raise ValueError("indices must be a 1-D integer array matching coefficients")
        if np.any(self.indices < 1) or np.unique(self.indices).size != self.indices.size:
            raise ValueError("Noll indices must be positive and unique")
        if self.unit not in ("rad", "waves"):
            raise ValueError("unit must be 'rad' or 'waves'")

    def to_radians(self) -> ZernikeCoefficients:
        """Return coefficients in radians without changing Noll ordering."""
        values = self.coefficients * (2 * np.pi if self.unit == "waves" else 1)
        return ZernikeCoefficients(values, self.indices, "rad", self.metadata)

    def to_waves(self) -> ZernikeCoefficients:
        """Return coefficients in waves without changing Noll ordering."""
        values = self.coefficients / (2 * np.pi if self.unit == "rad" else 1)
        return ZernikeCoefficients(values, self.indices, "waves", self.metadata)


@dataclass
class CoordinateGrid2D:
    """Two coordinate meshes with a declared scale and origin."""

    x: np.ndarray
    y: np.ndarray
    pitch: float
    origin: Literal["center", "corner"] = "center"
    units: Literal["um", "m", "pixels"] = "um"

    def __post_init__(self) -> None:
        self.x = _validate_2d_array(self.x, "x", dtype=np.float64)
        self.y = _validate_2d_array(self.y, "y", dtype=np.float64)
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same shape")
        self.pitch = _positive(self.pitch, "pitch")
        if self.origin not in ("center", "corner"):
            raise ValueError("origin must be 'center' or 'corner'")
        if self.units not in ("um", "m", "pixels"):
            raise ValueError("units must be 'um', 'm', or 'pixels'")

    @classmethod
    def centered(cls, shape: tuple[int, int], pitch_um: float) -> CoordinateGrid2D:
        """Build a centred grid whose x and y coordinates are micrometres."""
        x, y = centered_coordinates(shape, pitch_um)
        return cls(x, y, pitch_um, "center")

    def to_metres(self) -> CoordinateGrid2D:
        """Convert micrometre coordinates and pitch to metres."""
        if self.units == "m":
            return CoordinateGrid2D(self.x, self.y, self.pitch, self.origin, "m")
        if self.units != "um":
            raise ValueError("pixel coordinates need a physical pitch before conversion")
        return CoordinateGrid2D(self.x * 1e-6, self.y * 1e-6, self.pitch * 1e-6, self.origin, "m")


@dataclass
class BeamMetrics:
    """Far-field quality measurements; optional values require measurement."""

    pib: float | None = None
    efficiency: float | None = None
    uniformity_cv: float | None = None
    strehl: float | None = None
    zero_order_fraction: float | None = None
    fwhm_um: float | None = None

    def __post_init__(self) -> None:
        for name in ("pib", "efficiency", "strehl", "zero_order_fraction"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be a fraction in [0, 1]")
        for name in ("uniformity_cv", "fwhm_um"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass
class TurbulenceParameters:
    """Atmospheric turbulence parameters in SI units."""

    r0: float
    L0: float
    l0: float
    Cn2: float
    altitude: float | None = None
    wind_speed: float | None = None
    wind_direction: float | None = None

    def __post_init__(self) -> None:
        self.r0 = _positive(self.r0, "r0")
        self.L0 = _positive(self.L0, "L0")
        if self.l0 < 0 or not np.isfinite(self.l0) or self.l0 >= self.L0:
            raise ValueError("l0 must be non-negative and smaller than L0")
        if self.Cn2 < 0 or not np.isfinite(self.Cn2):
            raise ValueError("Cn2 must be finite and non-negative")
        for name in ("altitude", "wind_speed"):
            value = getattr(self, name)
            if value is not None and (value < 0 or not np.isfinite(value)):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.wind_direction is not None and not np.isfinite(self.wind_direction):
            raise ValueError("wind_direction must be finite")


@dataclass
class PSFImage:
    """Measured far-field intensity with camera scale and wavelength."""

    intensity: np.ndarray
    pixel_scale_um: float
    wavelength_m: float

    def __post_init__(self) -> None:
        self.intensity = _validate_2d_array(self.intensity, "intensity", dtype=np.float64, non_negative=True)
        self.pixel_scale_um = _positive(self.pixel_scale_um, "pixel_scale_um")
        self.wavelength_m = _positive(self.wavelength_m, "wavelength_m")

    @property
    def shape(self) -> tuple[int, int]:
        return self.intensity.shape

    def to_amplitude(self) -> AmplitudeMap:
        """Convert intensity samples to their square-root field amplitude."""
        return AmplitudeMap.from_intensity(
            self.intensity,
            self.pixel_scale_um,
            FieldMetadata(wavelength_m=self.wavelength_m, source="PSFImage"),
        )

    def resample_to(self, shape: tuple[int, int], pixel_scale_um: float, *, order: int = 1) -> PSFImage:
        """Interpolate intensity samples on a centred camera grid."""
        intensity = resample_physical_grid(self.intensity, self.pixel_scale_um, shape, pixel_scale_um, order=order)
        return PSFImage(np.maximum(intensity, 0), pixel_scale_um, self.wavelength_m)


@dataclass
class WavefrontStatistics:
    """Wavefront summary statistics in waves."""

    min: float
    max: float
    diff: float
    mean: float
    rms: float
    weighted_rms: float

    @classmethod
    def from_legacy(cls, values: dict[str, float]) -> WavefrontStatistics:
        """Read the legacy WFS dictionary, including its misspelled key."""
        return cls(values["min"], values["max"], values["diff"], values["mean"], values["rms"], values.get("weighted_rms", values.get("wighted_rms", values["rms"])))
