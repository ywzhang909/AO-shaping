"""Physical field data structures with OOPAO interop.

This module defines three core dataclasses — :class:`PhaseMap`,
:class:`AmplitudeMap`, and :class:`ComplexField` — representing the
fundamental optical quantities that flow through an adaptive-optics
system.  Each container couples a 2-D NumPy array with a pixel pitch
(in µm) and :class:`~ao_shaping.model.metadata.FieldMetadata` provenance.

Conversion to / from OOPAO
--------------------------
When the OOPAO library is importable (see
:func:`ao_shaping.drivers.sim._oopao_compat`), the ``to_oopao_*`` and
``from_oopao_*`` members bridge these containers and OOPAO ``Source``
objects, so that simulation and hardware pipelines can exchange field
data with the ESO/LAM optical model.

Phase is always stored and returned in **radians (unwrapped)**.  The
only ``mod 2π`` wrap lives downstream in the SLM driver; these containers
hold the raw physical phase.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from ao_shaping.model.metadata import FieldMetadata
from ao_shaping.model.spatial import resample_physical_grid

if TYPE_CHECKING:
    from ao_shaping.model.quantities import OPDMap

# ---------------------------------------------------------------------------
# Optional OOPAO backend
# ---------------------------------------------------------------------------
# Importing sim while model is still initializing creates a cycle through
# simulated DM drivers. Resolve the optional dependency only when it is used.
def _get_oopao_source() -> Any:
    try:
        from ao_shaping.drivers.sim._oopao_compat import Source
    except ImportError as exc:
        raise RuntimeError(
            "OOPAO is not available; install via `uv pip install --no-deps -e libs/OOPAO`"
        ) from exc
    return Source

__all__ = [
    "PhaseMap",
    "AmplitudeMap",
    "ComplexField",
    "oopao_available",
    "closest_opt_band",
]


def oopao_available() -> bool:
    """Return ``True`` when the OOPAO library was imported successfully."""
    try:
        _get_oopao_source()
    except (ImportError, RuntimeError):
        return False
    return True


# ---------------------------------------------------------------------------
# Wavelength → OOPAO optBand lookup
# ---------------------------------------------------------------------------
# OOPAO's ``Source.photometry`` maps optical-band strings to fixed
# (wavelength, bandwidth, zero-point-flux) triples.  We keep a sorted table
# so that callers can pick the band whose reference wavelength is closest
# to the wavelength of a measured / simulated field.
_OPTBAND_TABLE: list[tuple[float, str]] = [
    (360e-9, "U"),
    (440e-9, "B"),
    (500e-9, "V0"),
    (550e-9, "V"),
    (589e-9, "Na"),
    (600e-9, "R3"),
    (640e-9, "R"),
    (650e-9, "R2"),
    (670e-9, "R4"),
    (700e-9, "I1"),
    (750e-9, "I2"),
    (790e-9, "I"),
    (800e-9, "I3"),
    (850e-9, "I4"),
    (900e-9, "I6"),
    (950e-9, "I7"),
    (1000e-9, "I8"),
    (1064e-9, "EOS"),
    (1215e-9, "J"),
    (1550e-9, "J2"),
    (1654e-9, "H"),
    (2000e-9, "K0"),
    (2124.5e-9, "Kp"),
    (2157e-9, "Ks"),
    (2179e-9, "K"),
    (2400e-9, "K1"),
    (3547e-9, "L"),
    (4769e-9, "M"),
]


def closest_opt_band(wavelength_m: float) -> str:
    """Return the OOPAO ``optBand`` string whose reference wavelength is
    closest to *wavelength_m* (metres).

    Falls back to ``"R"`` (640 nm) if the table is empty.
    """
    wl = float(wavelength_m)
    if not _OPTBAND_TABLE:
        return "R"
    return min(_OPTBAND_TABLE, key=lambda entry: abs(entry[0] - wl))[1]


def _validate_2d_array(
    arr: np.ndarray,
    name: str,
    dtype: type | None = None,
    finite_only: bool = True,
    non_negative: bool = False,
) -> np.ndarray:
    """Validate and coerce *arr* to a 2-D NumPy array.

    Args:
        arr: Input data.
        name: Parameter name used in error messages.
        dtype: Desired dtype; ``None`` keeps the input dtype.
        finite_only: When True, reject NaN / ±Inf values.
        non_negative: When True, reject negative values.

    Returns:
        A validated ``np.ndarray``.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if dtype is not None:
        arr = arr.astype(dtype, copy=True)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {arr.ndim}-D (shape {arr.shape})")
    if finite_only and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
    if non_negative and np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative")
    return arr


# ---------------------------------------------------------------------------
# PhaseMap
# ---------------------------------------------------------------------------
@dataclass
class PhaseMap:
    """Physical-optics phase map (2-D, radians).

    Attributes:
        phase: 2-D float array of optical phase in **radians** (unwrapped).
        pitch_size_um: Pixel pitch in micrometres.
        metadata: Provenance information (timestamp, source, wavelength).
    """

    phase: np.ndarray
    pitch_size_um: float
    metadata: FieldMetadata = dc_field(default_factory=FieldMetadata)

    def __post_init__(self) -> None:
        self.phase = _validate_2d_array(self.phase, "phase", dtype=np.float64)
        if float(self.pitch_size_um) <= 0:
            raise ValueError(
                f"pitch_size_um must be positive, got {self.pitch_size_um}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape as ``(height, width)``."""
        return self.phase.shape

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of the phase array."""
        return self.phase.dtype

    @property
    def size(self) -> int:
        """Total number of pixels."""
        return self.phase.size

    # -- array-level conversions ------------------------------------------------

    def to_opd(self, wavelength_m: float) -> np.ndarray:
        """Convert phase (radians) to optical path difference (metres).

        ``OPD = φ · λ / (2π)``
        """
        return self.phase * float(wavelength_m) / (2.0 * np.pi)

    def to_opd_map(self, wavelength_m: float) -> OPDMap:
        """Return a typed metre-valued OPD map at the given wavelength."""
        from ao_shaping.model.quantities import OPDMap

        return OPDMap.from_phase(self, wavelength_m)

    def resample_to(self, shape: tuple[int, int], pitch_size_um: float, *, order: int = 1) -> PhaseMap:
        """Interpolate unwrapped radians on a centred physical grid."""
        phase = resample_physical_grid(self.phase, self.pitch_size_um, shape, pitch_size_um, order=order)
        return PhaseMap(phase, pitch_size_um, self.metadata)

    def to_oopao_opd(self, wavelength_m: float) -> np.ndarray:
        """Alias for :meth:`to_opd` — OPD array compatible with OOPAO."""
        return self.to_opd(wavelength_m)

    def to_oopao_phase(self) -> np.ndarray:
        """Return a copy of the phase array (radians) for OOPAO."""
        return self.phase.copy()

    # -- OOPAO Source conversion ------------------------------------------------

    def to_oopao_source(
        self,
        wavelength_m: float,
        opt_band: str | None = None,
        magnitude: float = 0.0,
    ) -> Any:
        """Build an OOPAO :class:`~OOPAO.Source` from this phase map.

        Args:
            wavelength_m: Wavelength in metres at which *phase* was measured.
                Required to convert radians → OPD.
            opt_band: OOPAO optical-band string (e.g. ``"EOS"`` for 1064 nm).
                If ``None``, the band whose reference wavelength is closest
                to *wavelength_m* is chosen.
            magnitude: Source magnitude passed to ``Source.__init__``.

        Returns:
            An OOPAO ``Source`` with ``OPD`` set so that
            ``source.phase`` reproduces *phase*.

        Raises:
            RuntimeError: If OOPAO is not importable.
        """
        oopao_source = _get_oopao_source()
        band = opt_band or closest_opt_band(wavelength_m)
        with contextlib.redirect_stdout(io.StringIO()):
            src: Any = oopao_source(
                optBand=band, magnitude=magnitude, display_properties=False
            )
            src.OPD = self.to_opd(wavelength_m)
        logger.debug(
            "Created OOPAO Source from PhaseMap: band={}, "
            "shape={}, wavelength={:.1f} nm",
            band,
            self.phase.shape,
            wavelength_m * 1e9,
        )
        return src

    # -- reconstruction ---------------------------------------------------------

    @classmethod
    def from_oopao_source(
        cls,
        source: Any,
        *,
        pitch_size_um: float,
        metadata: FieldMetadata | None = None,
    ) -> PhaseMap:
        """Create a :class:`PhaseMap` from an OOPAO ``Source``.

        Reads ``source.OPD`` and converts to radians at the source's own
        wavelength, so the resulting phase is physically consistent.
        """
        if source is None:
            raise ValueError("source must not be None")
        wavelength_m = float(getattr(source, "wavelength", 500e-9))
        opd = np.asarray(source.OPD, dtype=np.float64)
        phase = opd * 2.0 * np.pi / wavelength_m
        if metadata is None:
            metadata = FieldMetadata(source=f"OOPAO{getattr(source, 'type', '')}")
        return cls(phase=phase, pitch_size_um=float(pitch_size_um), metadata=metadata)

    def __repr__(self) -> str:
        meta = self.metadata
        return (
            f"PhaseMap(shape={self.phase.shape}, "
            f"pitch_size_um={self.pitch_size_um}, "
            f"source={meta.source!r})"
        )


# ---------------------------------------------------------------------------
# AmplitudeMap
# ---------------------------------------------------------------------------
@dataclass
class AmplitudeMap:
    """Physical-optics amplitude map (2-D, field amplitude |E|).

    Attributes:
        amplitude: 2-D float array of the complex field **amplitude**
            (√ of intensity), non-negative.
        pitch_size_um: Pixel pitch in micrometres.
        metadata: Provenance information.
    """

    amplitude: np.ndarray
    pitch_size_um: float
    metadata: FieldMetadata = dc_field(default_factory=FieldMetadata)

    def __post_init__(self) -> None:
        self.amplitude = _validate_2d_array(
            self.amplitude, "amplitude", dtype=np.float64, non_negative=True
        )
        if float(self.pitch_size_um) <= 0:
            raise ValueError(
                f"pitch_size_um must be positive, got {self.pitch_size_um}"
            )

    @classmethod
    def from_intensity(
        cls,
        intensity: np.ndarray,
        pitch_size_um: float,
        metadata: FieldMetadata | None = None,
    ) -> AmplitudeMap:
        """Convert non-negative intensity samples to field amplitude."""
        samples = _validate_2d_array(intensity, "intensity", dtype=np.float64, non_negative=True)
        return cls(np.sqrt(samples), pitch_size_um, metadata or FieldMetadata())

    @property
    def shape(self) -> tuple[int, int]:
        return self.amplitude.shape

    @property
    def dtype(self) -> np.dtype:
        return self.amplitude.dtype

    @property
    def size(self) -> int:
        return self.amplitude.size

    @property
    def intensity(self) -> np.ndarray:
        """Intensity map ``|E|²`` (same shape as *amplitude*)."""
        return self.amplitude**2

    @property
    def total_power(self) -> float:
        """Sum of intensity over the array (arbitrary units, not calibrated)."""
        return float(np.sum(self.intensity))

    # -- array-level conversions ------------------------------------------------

    def to_intensity(self) -> np.ndarray:
        """Return a copy of the intensity array ``|E|²``."""
        return self.intensity.copy()

    def resample_to(self, shape: tuple[int, int], pitch_size_um: float, *, order: int = 1) -> AmplitudeMap:
        """Interpolate field amplitude on a centred physical grid."""
        amplitude = resample_physical_grid(self.amplitude, self.pitch_size_um, shape, pitch_size_um, order=order)
        return AmplitudeMap(amplitude, pitch_size_um, self.metadata)

    def to_oopao_intensity(self) -> np.ndarray:
        """Alias for :meth:`to_intensity` — intensity array for OOPAO."""
        return self.to_intensity()

    def to_oopao_amplitude(self) -> np.ndarray:
        """Return a copy of the raw amplitude array."""
        return self.amplitude.copy()

    # -- OOPAO Source conversion ------------------------------------------------

    def to_oopao_source(
        self,
        wavelength_m: float,
        opt_band: str | None = None,
        magnitude: float = 0.0,
    ) -> Any:
        """Build an OOPAO :class:`~OOPAO.Source` from this amplitude map.

        Sets ``source.intensity = amplitude²`` (OOPAO's authoritative
        amplitude path — see ``Source._refresh_intensity``).

        Args:
            wavelength_m: Wavelength in metres (used to pick the closest
                ``optBand`` when *opt_band* is ``None``).
            opt_band: Explicit OOPAO optical-band string; if ``None`` the
                band closest to *wavelength_m* is chosen.
            magnitude: Source magnitude for ``Source.__init__``.

        Returns:
            An OOPAO ``Source`` with ``intensity`` set.

        Raises:
            RuntimeError: If OOPAO is not importable.
        """
        oopao_source = _get_oopao_source()
        band = opt_band or closest_opt_band(wavelength_m)
        with contextlib.redirect_stdout(io.StringIO()):
            src: Any = oopao_source(
                optBand=band, magnitude=magnitude, display_properties=False
            )
            src.intensity = self.intensity
        logger.debug(
            "Created OOPAO Source from AmplitudeMap: band={}, "
            "shape={}, wavelength={:.1f} nm",
            band,
            self.amplitude.shape,
            wavelength_m * 1e9,
        )
        return src

    # -- reconstruction ---------------------------------------------------------

    @classmethod
    def from_oopao_source(
        cls,
        source: Any,
        *,
        pitch_size_um: float,
        metadata: FieldMetadata | None = None,
    ) -> AmplitudeMap:
        """Create an :class:`AmplitudeMap` from an OOPAO ``Source``.

        Extracts ``sqrt(|source.intensity|)`` as the field amplitude.
        """
        if source is None:
            raise ValueError("source must not be None")
        intensity = np.asarray(source.intensity, dtype=np.float64)
        intensity = np.maximum(intensity, 0.0)
        amplitude = np.sqrt(intensity)
        if metadata is None:
            metadata = FieldMetadata(source=f"OOPAO{getattr(source, 'type', '')}")
        return cls(
            amplitude=amplitude,
            pitch_size_um=float(pitch_size_um),
            metadata=metadata,
        )

    def __repr__(self) -> str:
        meta = self.metadata
        return (
            f"AmplitudeMap(shape={self.amplitude.shape}, "
            f"pitch_size_um={self.pitch_size_um}, "
            f"source={meta.source!r})"
        )


# ---------------------------------------------------------------------------
# ComplexField
# ---------------------------------------------------------------------------
@dataclass
class ComplexField:
    """Full complex electromagnetic-field map ``E(x, y)``.

    Attributes:
        field: 2-D complex array ``A · exp(iφ)``.
        pitch_size_um: Pixel pitch in micrometres.
        metadata: Provenance information.
    """

    field: np.ndarray
    pitch_size_um: float
    metadata: FieldMetadata = dc_field(default_factory=FieldMetadata)

    def __post_init__(self) -> None:
        arr = self.field
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        arr = arr.astype(np.complex128, copy=True)
        if arr.ndim != 2:
            raise ValueError(f"field must be 2-D, got {arr.ndim}-D (shape {arr.shape})")
        if not np.all(np.isfinite(arr)):
            raise ValueError("field contains non-finite values (NaN or Inf)")
        self.field = arr
        if float(self.pitch_size_um) <= 0:
            raise ValueError(
                f"pitch_size_um must be positive, got {self.pitch_size_um}"
            )

    @classmethod
    def from_phase_amplitude(
        cls,
        phase: np.ndarray,
        amplitude: np.ndarray,
        *,
        pitch_size_um: float,
        metadata: FieldMetadata | None = None,
    ) -> ComplexField:
        """Construct a complex field from separate phase and amplitude arrays.

        ``E = amplitude · exp(i · phase)``
        """
        phase = _validate_2d_array(phase, "phase", dtype=np.float64)
        amplitude = _validate_2d_array(
            amplitude, "amplitude", dtype=np.float64, non_negative=True
        )
        if phase.shape != amplitude.shape:
            raise ValueError(
                f"phase {phase.shape} and amplitude {amplitude.shape} "
                "must have the same shape"
            )
        field = amplitude * np.exp(1j * phase)
        return cls(
            field=field,
            pitch_size_um=float(pitch_size_um),
            metadata=metadata or FieldMetadata(),
        )

    @classmethod
    def from_maps(cls, phase: PhaseMap, amplitude: AmplitudeMap) -> ComplexField:
        """Combine maps only when their shape and physical sampling agree."""
        if phase.shape != amplitude.shape or not np.isclose(phase.pitch_size_um, amplitude.pitch_size_um, rtol=1e-12):
            raise ValueError("phase and amplitude maps must share shape and pixel pitch")
        return cls.from_phase_amplitude(
            phase.phase, amplitude.amplitude,
            pitch_size_um=phase.pitch_size_um, metadata=phase.metadata,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.field.shape

    @property
    def dtype(self) -> np.dtype:
        return self.field.dtype

    @property
    def size(self) -> int:
        return self.field.size

    @property
    def amplitude(self) -> np.ndarray:
        """Field amplitude ``|E|`` (non-negative real array)."""
        return np.abs(self.field)

    @property
    def phase(self) -> np.ndarray:
        """Phase ``arg(E)`` in radians, range ``[-π, π]``."""
        return np.angle(self.field)

    @property
    def phase_unwrapped(self) -> np.ndarray:
        """Unwrapped phase in radians (best-effort, via project phase unwrapper)."""
        from ao_shaping.utils.wavefront.phase_unwrap import unwrap_phase

        return unwrap_phase(self.phase)

    @property
    def intensity(self) -> np.ndarray:
        """Intensity ``|E|²`` (real, non-negative)."""
        return np.abs(self.field) ** 2

    @property
    def total_power(self) -> float:
        """Sum of intensity over the array."""
        return float(np.sum(self.intensity))

    @property
    def peak_intensity(self) -> float:
        """Maximum intensity value."""
        return float(np.max(self.intensity))

    # -- array-level conversions ------------------------------------------------

    def to_phase(self, unwrap: bool = False) -> np.ndarray:
        """Return the phase array in radians.

        Args:
            unwrap: When True, return the unwrapped phase (requires SciPy).
        """
        return self.phase_unwrapped if unwrap else self.phase.copy()

    def to_amplitude(self) -> np.ndarray:
        """Return the amplitude array ``|E|``."""
        return self.amplitude.copy()

    def to_intensity(self) -> np.ndarray:
        """Return the intensity array ``|E|²``."""
        return self.intensity.copy()

    def resample_to(self, shape: tuple[int, int], pitch_size_um: float, *, order: int = 1) -> ComplexField:
        """Interpolate real and imaginary field parts in physical space."""
        values = resample_physical_grid(self.field, self.pitch_size_um, shape, pitch_size_um, order=order)
        return ComplexField(values, pitch_size_um, self.metadata)

    def to_opd(self, wavelength_m: float, unwrap: bool = False) -> np.ndarray:
        """Convert phase to optical path difference (metres).

        ``OPD = φ · λ / (2π)``
        """
        phase = self.to_phase(unwrap=unwrap)
        return phase * float(wavelength_m) / (2.0 * np.pi)

    def to_oopao_opd(self, wavelength_m: float, unwrap: bool = False) -> np.ndarray:
        """Alias for :meth:`to_opd` — OPD array for OOPAO."""
        return self.to_opd(wavelength_m, unwrap=unwrap)

    # -- OOPAO Source conversion ------------------------------------------------

    def to_oopao_source(
        self,
        wavelength_m: float,
        opt_band: str | None = None,
        magnitude: float = 0.0,
        unwrap_phase: bool = False,
    ) -> Any:
        """Build an OOPAO :class:`~OOPAO.Source` from this complex field.

        Sets both ``source.OPD`` (from the phase) and ``source.intensity``
        (from ``|E|²``), so the Source carries the full EM-field information.

        Args:
            wavelength_m: Wavelength in metres for phase→OPD conversion.
            opt_band: OOPAO optical-band string; auto-selected if ``None``.
            magnitude: Source magnitude for ``Source.__init__``.
            unwrap_phase: If True, unwrap the phase before converting to OPD.
                Recommended for smoother propagation.

        Returns:
            An OOPAO ``Source`` with ``OPD`` and ``intensity`` set.

        Raises:
            RuntimeError: If OOPAO is not importable.
        """
        oopao_source = _get_oopao_source()
        band = opt_band or closest_opt_band(wavelength_m)
        with contextlib.redirect_stdout(io.StringIO()):
            src: Any = oopao_source(
                optBand=band, magnitude=magnitude, display_properties=False
            )
            src.OPD = self.to_opd(wavelength_m, unwrap=unwrap_phase)
            src.intensity = self.intensity
        logger.debug(
            "Created OOPAO Source from ComplexField: band={}, "
            "shape={}, wavelength={:.1f} nm",
            band,
            self.field.shape,
            wavelength_m * 1e9,
        )
        return src

    # -- reconstruction ---------------------------------------------------------

    @classmethod
    def from_oopao_source(
        cls,
        source: Any,
        *,
        pitch_size_um: float,
        metadata: FieldMetadata | None = None,
    ) -> ComplexField:
        """Create a :class:`ComplexField` from an OOPAO ``Source``.

        Combines ``source.phase`` and ``source.intensity`` into a complex
        field ``E = √I · exp(iφ)``.
        """
        if source is None:
            raise ValueError("source must not be None")
        wavelength_m = float(getattr(source, "wavelength", 500e-9))
        phase = np.asarray(source.phase, dtype=np.float64)
        intensity = np.asarray(source.intensity, dtype=np.float64)
        intensity = np.maximum(intensity, 0.0)
        amplitude = np.sqrt(intensity)
        if metadata is None:
            metadata = FieldMetadata(source=f"OOPAO{getattr(source, 'type', '')}")
        return cls.from_phase_amplitude(
            phase=phase,
            amplitude=amplitude,
            pitch_size_um=float(pitch_size_um),
            metadata=metadata,
        )

    def __repr__(self) -> str:
        meta = self.metadata
        return (
            f"ComplexField(shape={self.field.shape}, "
            f"pitch_size_um={self.pitch_size_um}, "
            f"source={meta.source!r})"
        )
