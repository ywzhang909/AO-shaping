"""Metadata for physical field samples.

Provides :class:`FieldMetadata`, a lightweight dataclass that records
provenance information (when a field was captured, from which device or
simulation, and at what wavelength) for the model data structures in
:mod:`ao_shaping.model.field`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class FieldMetadata:
    """Provenance metadata for a sampled optical field.

    Attributes:
        timestamp: When the field data was captured or generated.
            Defaults to the current time when the field is created.
        source: Human-readable identifier of the origin device or
            simulation, e.g. ``"Santec SLM-200 #1"`` or
            ``"sim.digitaltwin"``.
        wavelength_m: Wavelength of the field in metres, if applicable.
            ``None`` when the wavelength is unspecified or irrelevant.
        extra: Additional free-form key/value pairs (e.g. temperature,
            humidity, DM serial number).
    """

    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    wavelength_m: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the metadata to a plain dictionary.

        ``datetime`` objects are converted to ISO-format strings so the
        result is JSON-serialisable.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "wavelength_m": self.wavelength_m,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldMetadata:
        """Reconstruct metadata from a dictionary produced by ``to_dict``.

        Unknown keys are silently ignored.
        """
        timestamp_str = data.get("timestamp")
        timestamp = (
            datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
        )
        return cls(
            timestamp=timestamp,
            source=data.get("source", "unknown"),
            wavelength_m=data.get("wavelength_m"),
            extra=dict(data.get("extra", {})),
        )
