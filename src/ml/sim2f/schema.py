"""On-disk schema of the supervised corpus -- the one place it is written down.

:mod:`ml.sim2f.synth` writes this archive and :mod:`ml.sim2f.dataset` reads it.
Both import this module instead of duplicating the key table, so a renamed or
re-shaped array fails loudly at load time rather than silently training on the
wrong thing.

The reader is duck-typed on purpose: it accepts anything with
``key -> object with .shape and .dtype`` (``numpy.lib.npyio.NpzFile``, a plain
``dict`` of arrays, a test double), so this module stays stdlib-only and can be
imported without numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CACHE_FILE",
    "CACHE_FIELDS",
    "CACHE_VERSION",
    "META_FILE",
    "SPLIT_NAMES",
    "SPLIT_TEST",
    "SPLIT_TRAIN",
    "SPLIT_VAL",
    "CacheField",
    "validate_cache",
]

#: Archive written by :func:`ml.sim2f.synth.build_pairs`.
CACHE_FILE = "pairs.npz"
#: Sidecar JSON with the three configs, the derived optics and the wall clock.
META_FILE = "meta.json"
#: Bump when a field changes meaning, not merely when it is appended.
CACHE_VERSION = 1

SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2
#: Index -> name, so a CSV/report can print a split without a second table.
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class CacheField:
    """One array of the archive.

    Attributes:
        name: Key inside ``pairs.npz``.
        shape: Shape with ``N`` for the sample axis.
        dtype: Expected numpy dtype name.
        doc: What it is, in the units the rest of the package uses.
    """

    name: str
    shape: tuple[str | int, ...]
    dtype: str
    doc: str


#: The archive, field by field.
CACHE_FIELDS: tuple[CacheField, ...] = (
    CacheField(
        "target",
        ("N", 1, "ccd_px", "ccd_px"),
        "float32",
        "Requested far-field distribution on the camera grid, unit sum.",
    ),
    CacheField(
        "support",
        ("N", 1, "ccd_px", "ccd_px"),
        "uint8",
        "Analytic support of the target (1 inside). The teacher's EE term and the "
        "shaping metrics use it; it is not a threshold on `target`.",
    ),
    CacheField(
        "phase",
        ("N", "grid", "grid"),
        "float32",
        "Teacher phase on the SLM grid, raw unwrapped radians, ABERRATION-FREE bench.",
    ),
    CacheField(
        "spot",
        ("N", 1, "ccd_px", "ccd_px"),
        "float32",
        "Ideal-bench flat-phase camera frame for the same optics (reference / plots).",
    ),
    CacheField(
        "teacher_metrics",
        ("N", 4),
        "float32",
        "Columns: final objective, L1, encircled energy, NCC -- teacher vs target.",
    ),
    CacheField("kind", ("N",), "int16", "Index into `ml.sim2f.config.TARGET_KINDS`."),
    CacheField("seed", ("N",), "int64", "Per-sample seed, so any sample is reproducible alone."),
    CacheField("split", ("N",), "int8", "0 train / 1 val / 2 test, see SPLIT_NAMES."),
)


def validate_cache(archive: object, *, grid: int, ccd_px: int) -> None:
    """Check an archive against :data:`CACHE_FIELDS`.

    Args:
        archive: Mapping-like object exposing ``key in archive`` and
            ``archive[key].shape`` / ``.dtype.name``.
        grid: Expected pupil grid side.
        ccd_px: Expected camera grid side.

    Raises:
        ValueError: If a field is missing, has the wrong shape or the wrong
            dtype, or if the sample count is inconsistent across fields.
    """
    counts: dict[str, int] = {}
    for field in CACHE_FIELDS:
        if field.name not in archive:  # type: ignore[operator]
            raise ValueError(f"{CACHE_FILE} is missing field {field.name!r}")
        array = archive[field.name]  # type: ignore[index]
        expected = tuple(
            {"N": 0, "grid": int(grid), "ccd_px": int(ccd_px)}.get(str(axis), axis)
            for axis in field.shape
        )
        got = tuple(int(dim) for dim in array.shape)
        if len(got) != len(expected) or any(
            want != 0 and want != have for want, have in zip(expected, got, strict=True)
        ):
            raise ValueError(
                f"field {field.name!r} has shape {got}, expected {expected} "
                "(0 = sample axis, any length)"
            )
        if array.dtype.name != field.dtype:
            raise ValueError(
                f"field {field.name!r} has dtype {array.dtype.name}, expected {field.dtype}"
            )
        counts[field.name] = got[0]
    if len(set(counts.values())) != 1:
        raise ValueError(f"{CACHE_FILE} fields disagree on the sample count: {counts}")
    if counts[CACHE_FIELDS[0].name] < 1:
        raise ValueError(f"{CACHE_FILE} holds no samples")
