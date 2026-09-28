"""Lean, mmap-able on-disk cache for the offline GSNet training corpus.

===========================  Why this module exists  ===========================

The debug corpus is ~9.6 GB across 26 pickles (3202 records; the largest single
pickle is 3.19 GB / 733 records), but only **two** fields per record matter for
training: ``_c`` (the Zernike coefficients) and ``_img`` (the measured far
field). Together they are ~0.20 GB, i.e. **2.0 %** of the raw bytes. The bulk is
``_phase`` (a ``1200 x 1920`` uint16 panel, ~4.6 MB on the 2192 records that
carry it) which is **never used**: the pupil phase is reconstructed from ``_c``
and was already verified bit-exact against the hardware path.

``pickle`` cannot be memory-mapped, and :class:`~ao_shaping.runners.gsnet_dataset.GSNetDebugDataset`
deserialises one whole file per LRU miss. With ``cache_size=1`` and a
:class:`~ao_shaping.runners.gsnet_dataset.FileGroupedSampler` interleaving 26
files, training therefore re-loaded multi-hundred-MB and 3.2 GB pickles
constantly: measured 3.5 samples/s (~15 min/epoch) with RSS peaking at ~2.5 GB.
No LRU tuning can fix that -- the bytes themselves are the problem. This module
stores only the two live fields, in a layout ``numpy`` *can* memory-map, so a
record costs one ``memmap`` page-in instead of a full deserialisation.

=============================  On-disk layout  ==============================

One cache directory per source pickle. For ``data/debug/slm_pib_x/y/z.pkl`` the
default layout is the sibling ``data/debug/slm_pib_x/y/.gsnet_cache/z/``::

    c_flat.npy       1-D  float64   every record's ``_c``, concatenated
    c_offsets.npy    (N+1,)  int64   record i == c_offsets[i]:c_offsets[i+1]
    img_flat.npy     1-D  uint8     every record's ``_img``, C order
    img_offsets.npy  (N+1,)  int64   same slicing contract
    img_shapes.npy   (N, 2)  int32   per-record ``(height, width)``
    keys.npy         (N,)   int64    the original pickle dict keys
    meta.json        {"source": <abs str>, "n_records": N,
                     "format_version": 1, "created": <iso str>}

Ragged ``len(_c)`` (15 / 66 / 78 are all observed) is handled by the offsets
with **no padding and no truncation**, and ``_img`` is **not** assumed to be a
constant shape: ``img_shapes`` carries the true ``(H, W)`` of every record and
each record is re-shaped from its own slice. Every ``.npy`` holds a plain
numeric dtype written by ``np.save(..., allow_pickle=False)``, so
``np.load(..., mmap_mode="r", allow_pickle=False)`` always works and object
arrays can never sneak in.

``meta.json`` is written **last**, so a build interrupted half-way leaves no
meta and the next :func:`prepare_gsnet_cache` simply rebuilds.

======================  Deviation: ``c_flat`` is float64  ======================

The original design asked for a **float32** ``c_flat``. That is incompatible
with the hard requirement that the cache be a *pure* storage optimisation:
``c_flat`` is the only copy of ``_c`` in the cache, and
:func:`~ao_shaping.runners.gsnet_offline.reconstruct_pupil_phase_rad` does
**not** quantise (``ZernikeGenerator.set_bits`` only records ``_max_val``; the
``generate_noll`` -> ``eval_grid`` path returns raw float64 radians), so a
float32 round-trip perturbs the reconstructed phase. Measured on this repo's own
synthetic corpus, float32 storage changes ``gt_phase`` in 200-208 of 4096 cells
by up to 128 float32 ULPs (~1.2e-7 rad) -- the phase would silently change.

float64 is therefore used: it is **exact** for any float32/float64/int source
coefficient, so ``(source, target, gt_phase)`` is bit-identical to the direct
pickle path, which is the whole point of the cache. The cost is negligible --
624 B/record instead of 312 B, i.e. ~2 MB instead of ~1 MB out of a ~200 MB
cache (1 %). ``_img`` is stored uint8 exactly as measured (a non-uint8 ``_img``
is rejected rather than silently requantised, for the same reason).

============================  Public entry points  ============================

* :func:`cache_dir_for` -- the on-disk location for one pickle.
* :func:`prepare_gsnet_cache` -- build/refresh every family in an index
  (idempotent; reuses a matching cache instead of rebuilding).
* :func:`load_cached_family` -- open one family, memory-mapped or eager.
* :class:`CachedFamily` -- per-record accessors (``coeffs`` / ``image`` /
  ``n_terms`` / ``keys`` / ``close``).
* :meth:`CachedFamily.as_payload` -- a ``dict``-shaped payload for
  :class:`~ao_shaping.runners.gsnet_dataset.GSNetDebugDataset`, whose records
  are lazy :class:`CachedRecord` views that resolve on first access.

Nothing here imports ``torch`` and nothing touches Zernike math: this is pure
storage. ``clear_cache()`` on the Dataset drops the *in-RAM* payload cache and
closes the mappings -- it never deletes anything from disk.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from loguru import logger

from ao_shaping.runners.gsnet_offline import RecordIndex

__all__ = [
    "CACHE_FILENAMES",
    "CACHE_FORMAT_VERSION",
    "DEFAULT_CACHE_DIR_NAME",
    "CachedFamily",
    "CachedRecord",
    "GSNetCacheError",
    "cache_dir_for",
    "close_cached_payload",
    "load_cached_family",
    "prepare_gsnet_cache",
]


#: On-disk schema version. Bumping it invalidates every existing cache dir
#: (the ``format_version`` in ``meta.json`` must match exactly to be reused).
CACHE_FORMAT_VERSION: int = 1

#: Directory name created next to each source pickle when no explicit root is
#: given. Leading dot keeps it hidden from casual ``ls``/glob views of the
#: corpus and from the ``*.pkl``-recursive index glob.
DEFAULT_CACHE_DIR_NAME: str = ".gsnet_cache"

META_FILENAME: str = "meta.json"

#: Every file a complete cache directory must contain, in write order. The
#: arrays are written first and ``meta.json`` last, so meta presence is the
#: "build finished" marker.
CACHE_FILENAMES: tuple[str, ...] = (
    "c_flat.npy",
    "c_offsets.npy",
    "img_flat.npy",
    "img_offsets.npy",
    "img_shapes.npy",
    "keys.npy",
    META_FILENAME,
)

#: Dtype contract of the stored arrays. ``c_flat`` is float64 (see the module
#: docstring: float32 would change the reconstructed phase); everything else is
#: the natural integer/uint type for its role.
C_DTYPE = np.dtype(np.float64)
IMG_DTYPE = np.dtype(np.uint8)
OFFSET_DTYPE = np.dtype(np.int64)
SHAPE_DTYPE = np.dtype(np.int32)
KEY_DTYPE = np.dtype(np.int64)


class GSNetCacheError(RuntimeError):
    """A GSNet on-disk cache is unusable (missing, stale, or corrupt).

    Raised by :func:`load_cached_family` and :func:`prepare_gsnet_cache`. The
    Dataset treats it as a *recoverable* condition -- it rebuilds the one family
    and, if the rebuild also fails, falls back to ``pickle.load`` -- so this is
    never fatal for a single bad family.
    """


# ---------------------------------------------------------------------------
# Paths / meta
# ---------------------------------------------------------------------------
def cache_dir_for(pkl_path: Path, cache_root: Path | None = None) -> Path:
    """Return the cache directory that belongs to one source pickle.

    The default is a sibling of the pickle -- ``<pkl_dir>/.gsnet_cache/<stem>/``
    -- which keeps the cache next to the data it mirrors (moving the corpus
    invalidates the cache by construction, because ``meta.json`` records the
    source's absolute path).

    Args:
        pkl_path: The source debug pickle.
        cache_root: Optional explicit root. When given, the directory is
            ``<cache_root>/<pkl stem>``; the root is used as given (relative
            paths are relative to the current working directory).

    Returns:
        The cache directory path. It is not created here.
    """
    source = Path(pkl_path)
    if cache_root is None:
        # Sibling of the pickle: <pkl_dir>/.gsnet_cache/<stem>/. Keeps the
        # cache next to the data it mirrors and makes it travel with the corpus.
        root = source.parent / DEFAULT_CACHE_DIR_NAME
    else:
        # Explicit root: <cache_root>/<stem>/, used as given.
        root = Path(cache_root)
    return root / source.stem


def _absolute(path: Path) -> str:
    """Absolute, symlink-resolved string form of ``path`` (for ``meta.json``)."""
    return str(Path(path).resolve())


def _read_meta(directory: Path) -> dict[str, Any] | None:
    """Read and parse ``meta.json``, or return ``None`` when unusable.

    Never raises: a missing file, unreadable file, non-JSON payload or
    non-object JSON all yield ``None`` so callers can treat "no cache" and
    "garbage cache" identically.
    """
    meta_path = directory / META_FILENAME
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug("Unreadable GSNet cache meta {}: {}", meta_path, exc)
        return None
    if not isinstance(payload, dict):
        logger.debug("GSNet cache meta {} is not a JSON object", meta_path)
        return None
    return payload


def _cache_is_current(source: Path, directory: Path) -> bool:
    """Whether ``directory`` is a reusable, intact cache for ``source``.

    Two layers of validation, both deliberately cheap:

    1. **Cheap invariants** -- ``meta.json`` parses, ``format_version`` matches
       this module's version, ``source`` is the same absolute pickle, and every
       expected file exists.
    2. **Structural validation** -- a full :func:`load_cached_family` pass, so a
       *truncated* ``c_flat.npy`` or a hand-edited ``meta.json`` is treated as
       corrupt instead of reusable. ``np.load(mmap_mode="r")`` only maps the
       header, and the offsets/shapes arrays are kilobytes, so this costs
       microseconds and never page-ins a record.

    Layer 2 is what makes the Dataset's "corrupt -> rebuild" fallback actually
    replace the broken family: without it a truncation would look "current" and
    :func:`prepare_gsnet_cache` would reuse the corruption.
    """
    meta = _read_meta(directory)
    if meta is None:
        return False
    if meta.get("format_version") != CACHE_FORMAT_VERSION:
        return False
    if meta.get("source") != _absolute(source):
        return False
    if not isinstance(meta.get("n_records"), int):
        return False
    if not all((directory / name).is_file() for name in CACHE_FILENAMES):
        return False

    try:
        load_cached_family(directory, mmap=True).close()
    except GSNetCacheError as exc:
        logger.debug("GSNet cache {} failed validation: {}", directory, exc)
        return False
    except OSError as exc:
        logger.debug("GSNet cache {} could not be mapped: {}", directory, exc)
        return False
    return True


# ---------------------------------------------------------------------------
# Lazy per-record view (the payload handed to the Dataset)
# ---------------------------------------------------------------------------
class _Unresolved:
    """Sentinel for a :class:`CachedRecord` entry not yet read from the cache."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "<unresolved>"


_UNRESOLVED = _Unresolved()


class CachedRecord(dict[str, Any]):
    """A ``dict``-shaped, lazily-resolved view of one cached record.

    The real ``dict`` storage holds the two keys the training path needs --
    ``"_img"`` and ``"_c"`` -- mapped to a sentinel; accessing either resolves
    it from the family's memory-mapped arrays on the spot and returns a real,
    owned array.

    Subclassing ``dict`` is what keeps the cache a *pure* storage swap: the
    Dataset's ``__getitem__`` keeps doing ``isinstance(record, dict)``,
    ``"_img" in record`` and ``record["_img"]`` exactly as it does for a
    deserialised pickle record, so the transform chain (and therefore
    ``gt_phase``) is untouched by this module.

    ``.get()`` is resolved as well. ``.values()`` / ``.items()`` / ``.copy()``
    expose the sentinel rather than the array; the training path never calls
    them, and any other payload field (``_grad``, ``_phase``, ``m_pib`` ...) is
    intentionally absent -- those are exactly the bytes this cache drops.
    """

    __slots__ = ("family", "position")

    def __init__(self, family: CachedFamily, position: int) -> None:
        """Bind this view to one record of one family.

        Args:
            family: The owning :class:`CachedFamily`.
            position: The record's 0-based position within that family.
        """
        self.family = family
        self.position = int(position)
        super().__init__({"_img": _UNRESOLVED, "_c": _UNRESOLVED})

    def __getitem__(self, key: str) -> Any:
        """Return the field, reading it from the cache on first access."""
        value = super().__getitem__(key)
        if value is _UNRESOLVED:
            return self._resolve(key)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """:meth:`dict.get`, resolving the lazy fields."""
        if key not in self:
            return default
        return self[key]

    def _resolve(self, key: str) -> Any:
        """Read one field out of the family.

        Args:
            key: ``"_img"`` or ``"_c"``; the only keys the sentinel covers.

        Returns:
            A real in-memory array: ``(H, W)`` uint8 for ``_img``, 1-D float64
            for ``_c``.

        Raises:
            KeyError: If ``key`` is not one of the two cached fields.
        """
        family = self.family
        position = self.position
        if key == "_img":
            return family.image(position)
        if key == "_c":
            return family.coeffs(position)
        raise KeyError(key)


def close_cached_payload(payload: Mapping[Any, Any]) -> None:
    """Close every family referenced by a cached payload (idempotent).

    Safe to call on a ``pickle.load`` payload: non-:class:`CachedRecord` values
    are ignored, so the Dataset can use it unconditionally in ``clear_cache``.

    Args:
        payload: A payload dict as returned by
            :meth:`CachedFamily.as_payload`.
    """
    seen: set[int] = set()
    for value in payload.values():
        if not isinstance(value, CachedRecord):
            continue
        family = value.family
        if id(family) in seen:
            continue
        seen.add(id(family))
        family.close()


# ---------------------------------------------------------------------------
# The family
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class CachedFamily:
    """One source pickle, opened as memory-mapped (or eager) ``.npy`` arrays.

    Construct it with :func:`load_cached_family`, never directly. All accessors
    copy the slice they hand out, so the returned arrays are ordinary owned
    numpy arrays: the caller may mutate or retain them freely, and closing the
    family cannot invalidate them.

    After :meth:`close` the arrays are released and every accessor raises
    :class:`GSNetCacheError`.
    """

    directory: Path
    source: str
    n_records: int
    _c_flat: np.ndarray | None
    _c_offsets: np.ndarray | None
    _img_flat: np.ndarray | None
    _img_offsets: np.ndarray | None
    _img_shapes: np.ndarray | None
    _keys: np.ndarray | None
    _key_positions: dict[int, int]

    @staticmethod
    def _require(array: np.ndarray | None, name: str) -> np.ndarray:
        """Return an array, or explain that the family was already closed."""
        if array is None:
            raise GSNetCacheError(
                "This CachedFamily has been closed; its memory maps were "
                f"released (array {name!r} is gone)"
            )
        return array

    def _index(self, position: int, name: str) -> int:
        """Validate a 0-based record position against ``n_records``."""
        index = int(position)
        if not 0 <= index < self.n_records:
            raise IndexError(
                f"record {position} is out of range for {self.directory} "
                f"({self.n_records} records)"
            )
        return index

    def keys(self) -> np.ndarray:
        """The family's record keys, as an owned ``(N,)`` int64 array."""
        return np.array(self._require(self._keys, "keys"), dtype=np.int64)

    def position_of(self, key: int) -> int:
        """Map a record key to its 0-based position in this family.

        Args:
            key: A key from the source pickle.

        Returns:
            The record's position, suitable for :meth:`image` / :meth:`coeffs`.

        Raises:
            KeyError: If the key is not in this family (e.g. the index points at
                a record the pickle does not contain).
        """
        return self._key_positions[int(key)]

    def n_terms(self, i: int) -> int:
        """Number of Zernike coefficients in record ``i`` (``len(record["_c"])``)."""
        index = self._index(i, "c_offsets")
        offsets = self._require(self._c_offsets, "c_offsets")
        return int(offsets[index + 1] - offsets[index])

    def coeffs(self, i: int) -> np.ndarray:
        """Record ``i``'s Zernike coefficients as an owned 1-D float64 array.

        Args:
            i: 0-based record position.

        Returns:
            ``(n_terms(i),)`` float64 -- the exact values stored in the source
            pickle, never padded and never truncated.

        Raises:
            IndexError: If ``i`` is out of range.
            GSNetCacheError: If the family is closed.
        """
        index = self._index(i, "c_flat")
        offsets = self._require(self._c_offsets, "c_offsets")
        flat = self._require(self._c_flat, "c_flat")
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        # np.array(...) copies: a memmap slice is only a view, and the caller
        # must not be able to reach (or outlive) the mapping through it.
        return np.array(flat[start:stop], dtype=np.float64)

    def image(self, i: int) -> np.ndarray:
        """Record ``i``'s far-field frame as an owned 2-D uint8 array.

        The shape is read from ``img_shapes``, so records of **different**
        ``(H, W)`` in the same pickle round-trip exactly.

        Args:
            i: 0-based record position.

        Returns:
            ``(height, width)`` uint8, C-contiguous, owned by the caller.

        Raises:
            IndexError: If ``i`` is out of range.
            GSNetCacheError: If the family is closed.
        """
        index = self._index(i, "img_flat")
        offsets = self._require(self._img_offsets, "img_offsets")
        flat = self._require(self._img_flat, "img_flat")
        shapes = self._require(self._img_shapes, "img_shapes")
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        height = int(shapes[index, 0])
        width = int(shapes[index, 1])
        # One copy + one reshape: the returned array owns a fresh C-order buffer
        # and never aliases the mapping.
        return np.array(flat[start:stop]).reshape(height, width)

    def as_payload(self) -> dict[int, CachedRecord]:
        """A ``dict``-shaped payload for :class:`GSNetDebugDataset`.

        Returns:
            ``{record_key: CachedRecord}`` for every record in this family, in
            ``keys`` order. The records are lazy: nothing is read from disk
            until a field is accessed.
        """
        keys = self._require(self._keys, "keys")
        return {int(key): CachedRecord(self, position) for position, key in enumerate(keys)}

    def close(self) -> None:
        """Release the memory maps. Idempotent, and safe on an eager family.

        A record handed out earlier is unaffected: every accessor copies, so no
        exported numpy view can be keeping a mapping alive. A ``BufferError``
        here (some caller kept a raw view) is logged rather than raised, and the
        reference is still dropped.
        """
        released = 0
        for name in ("_c_flat", "_c_offsets", "_img_flat", "_img_offsets", "_img_shapes", "_keys"):
            array = getattr(self, name)
            if array is None:
                continue
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                try:
                    mapping.close()
                except (BufferError, ValueError) as exc:
                    logger.debug(
                        "Could not close the memory map of {} ({}); dropping the "
                        "reference anyway",
                        self.directory,
                        exc,
                    )
                released += 1
            object.__setattr__(self, name, None)
        if released:
            logger.debug("Released {} memory maps for {}", released, self.directory)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_array(path: Path, *, mmap: bool) -> np.ndarray:
    """``np.load`` one cache array with pickling disabled.

    Args:
        path: The ``.npy`` file.
        mmap: ``True`` for ``mmap_mode="r"``, ``False`` to read eagerly.

    Returns:
        The array.

    Raises:
        GSNetCacheError: If the file is missing or unreadable as a plain
            numeric array.
    """
    try:
        return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise GSNetCacheError(f"Could not read cache array {path}: {exc}") from exc


def _expect(
    array: np.ndarray,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    path: Path,
) -> None:
    """Validate one loaded array's dtype and shape.

    Raises:
        GSNetCacheError: On any mismatch (this is the "length mismatch between
            offsets and flat array" family of corruption).
    """
    if array.dtype != dtype:
        raise GSNetCacheError(f"{path} has dtype {array.dtype}, expected {dtype}")
    if array.shape != shape:
        raise GSNetCacheError(f"{path} has shape {array.shape}, expected {shape}")


def load_cached_family(cache_dir: Path, *, mmap: bool = True) -> CachedFamily:
    """Open one cache directory as a :class:`CachedFamily`.

    With ``mmap=True`` every array is ``np.load(..., mmap_mode="r")``, so a
    record costs a page-in instead of a full file read; with ``mmap=False`` they
    are read eagerly. Either way :meth:`CachedFamily.coeffs` and
    :meth:`CachedFamily.image` return owned copies.

    The whole layout is validated up front -- dtypes, shapes, the
    ``offsets[-1] == flat.size`` contract, and ``(H, W)`` summing back to each
    record's slice length -- so a truncated or half-written cache is reported as
    corrupt *here* rather than producing a silently wrong sample.

    Args:
        cache_dir: A directory produced by :func:`prepare_gsnet_cache`.
        mmap: Memory-map the arrays (default) or load them eagerly.

    Returns:
        An open :class:`CachedFamily`.

    Raises:
        GSNetCacheError: If the directory is missing, its ``meta.json`` is
            unreadable or stale, a file is missing/corrupt, or the arrays are
            mutually inconsistent.
    """
    directory = Path(cache_dir)
    meta = _read_meta(directory)
    if meta is None:
        raise GSNetCacheError(f"No readable {META_FILENAME} in {directory}")
    version = meta.get("format_version")
    if version != CACHE_FORMAT_VERSION:
        raise GSNetCacheError(
            f"{directory} was written with format_version {version!r}, "
            f"this module speaks {CACHE_FORMAT_VERSION}"
        )
    n_records = meta.get("n_records")
    if not isinstance(n_records, int) or isinstance(n_records, bool) or n_records < 0:
        raise GSNetCacheError(f"{directory} has an invalid n_records: {n_records!r}")

    missing = [name for name in CACHE_FILENAMES if not (directory / name).is_file()]
    if missing:
        raise GSNetCacheError(f"{directory} is incomplete, missing {missing}")

    keys_path = directory / "keys.npy"
    keys = _load_array(keys_path, mmap=mmap)
    _expect(keys, dtype=KEY_DTYPE, shape=(n_records,), path=keys_path)

    c_offsets_path = directory / "c_offsets.npy"
    c_offsets = _load_array(c_offsets_path, mmap=mmap)
    _expect(
        c_offsets,
        dtype=OFFSET_DTYPE,
        shape=(n_records + 1,),
        path=c_offsets_path,
    )
    c_flat_path = directory / "c_flat.npy"
    c_flat = _load_array(c_flat_path, mmap=mmap)
    _expect(
        c_flat,
        dtype=C_DTYPE,
        shape=(int(c_offsets[n_records]),),
        path=c_flat_path,
    )
    if n_records and (int(c_offsets[0]) != 0 or bool(np.any(np.diff(c_offsets) < 0))):
        raise GSNetCacheError(f"{c_offsets_path} is not a monotonic 0-based offset table")

    img_offsets_path = directory / "img_offsets.npy"
    img_offsets = _load_array(img_offsets_path, mmap=mmap)
    _expect(
        img_offsets,
        dtype=OFFSET_DTYPE,
        shape=(n_records + 1,),
        path=img_offsets_path,
    )
    img_flat_path = directory / "img_flat.npy"
    img_flat = _load_array(img_flat_path, mmap=mmap)
    _expect(
        img_flat,
        dtype=IMG_DTYPE,
        shape=(int(img_offsets[n_records]),),
        path=img_flat_path,
    )
    if n_records and (int(img_offsets[0]) != 0 or bool(np.any(np.diff(img_offsets) < 0))):
        raise GSNetCacheError(
            f"{img_offsets_path} is not a monotonic 0-based offset table"
        )

    img_shapes_path = directory / "img_shapes.npy"
    img_shapes = _load_array(img_shapes_path, mmap=mmap)
    _expect(
        img_shapes,
        dtype=SHAPE_DTYPE,
        shape=(n_records, 2),
        path=img_shapes_path,
    )
    # The per-record (H, W) must add back up to each record's own slice: this is
    # what makes non-uniform _img shapes safe to concatenate.
    if n_records:
        spans = img_shapes[:, 0].astype(np.int64) * img_shapes[:, 1].astype(np.int64)
        expected = np.diff(np.asarray(img_offsets, dtype=np.int64))
        if not np.array_equal(spans, expected):
            raise GSNetCacheError(
                f"{img_shapes_path} disagrees with {img_offsets_path}: the "
                f"(H, W) products sum to {int(spans.sum())} but the offsets span "
                f"{int(expected.sum())} bytes"
            )

    key_positions = {int(key): position for position, key in enumerate(np.asarray(keys))}
    if len(key_positions) != n_records:
        raise GSNetCacheError(f"{keys_path} contains duplicate record keys")

    return CachedFamily(
        directory=directory,
        source=str(meta.get("source", "")),
        n_records=n_records,
        _c_flat=c_flat,
        _c_offsets=c_offsets,
        _img_flat=img_flat,
        _img_offsets=img_offsets,
        _img_shapes=img_shapes,
        _keys=keys,
        _key_positions=key_positions,
    )


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
def _coerce_coefficients(value: Any, *, source: Path, key: int) -> np.ndarray:
    """Return one record's ``_c`` as a 1-D float64 array.

    float64 is exact for every float width and for integers, so the cached value
    is bit-identical to what ``pickle.load`` would have produced.

    Raises:
        GSNetCacheError: If the value is empty or not numeric.
    """
    try:
        coeffs = np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise GSNetCacheError(
            f"Record {key} of {source} has a non-numeric '_c': {exc}"
        ) from exc
    if coeffs.size == 0:
        raise GSNetCacheError(f"Record {key} of {source} has an empty '_c'")
    return coeffs


def _coerce_image(value: Any, *, source: Path, key: int) -> np.ndarray:
    """Return one record's ``_img`` as a 2-D uint8 array.

    A non-uint8 frame is **rejected** rather than requantised: the cache must be
    byte-exact, and clipping a float frame to uint8 would silently change
    ``target``. Rejecting makes the Dataset fall back to ``pickle.load``.

    Raises:
        GSNetCacheError: If the frame is empty, not 2-D, or not uint8.
    """
    frame = np.asarray(value)
    if frame.ndim != 2 or frame.size == 0:
        raise GSNetCacheError(
            f"Record {key} of {source} has a '_img' of shape {frame.shape}; a "
            f"non-empty 2-D frame is required"
        )
    if frame.dtype != IMG_DTYPE:
        raise GSNetCacheError(
            f"Record {key} of {source} has a '{frame.dtype}' '_img'; the cache "
            f"stores uint8 verbatim and will not requantise a different dtype"
        )
    return np.ascontiguousarray(frame)


def _build_family(
    source: Path,
    directory: Path,
    *,
    log_every: int,
) -> tuple[int, int]:
    """Write the full cache directory for one source pickle.

    MEMORY: ``pickle.load`` must materialise the whole file (that is the very
    cost being amortised), so peak RAM here is one payload plus its two flat
    buffers. Nothing is retained afterwards.

    Args:
        source: The pickle to read.
        directory: Destination directory (created if absent).
        log_every: Progress-log interval in records (``<= 0`` disables it).

    Returns:
        ``(n_records, bytes_written)``.

    Raises:
        GSNetCacheError: If the source is missing, its root is not a ``dict``,
            or a record lacks a usable ``_c`` / ``_img``.
    """
    try:
        with open(source, "rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 - repo-internal debug dumps
    except FileNotFoundError as exc:
        raise GSNetCacheError(f"Debug pickle does not exist: {source}") from exc
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise GSNetCacheError(f"Could not read debug pickle {source}: {exc}") from exc

    if not isinstance(payload, dict):
        raise GSNetCacheError(
            f"{source} does not contain a dict of records "
            f"(got {type(payload).__name__})"
        )

    keys = sorted(int(k) for k in payload if isinstance(k, (int, np.integer)))
    skipped = len(payload) - len(keys)
    if skipped:
        logger.warning(
            "Skipping {} non-integer key(s) in {}; the cache addresses records "
            "by integer key only",
            skipped,
            source,
        )

    c_chunks: list[np.ndarray] = []
    img_chunks: list[np.ndarray] = []
    c_offsets = np.zeros(len(keys) + 1, dtype=OFFSET_DTYPE)
    img_offsets = np.zeros(len(keys) + 1, dtype=OFFSET_DTYPE)
    img_shapes = np.zeros((len(keys), 2), dtype=SHAPE_DTYPE)

    for position, key in enumerate(keys):
        record = payload[key]
        if not isinstance(record, dict):
            raise GSNetCacheError(
                f"Record {key} in {source} is not a dict "
                f"(got {type(record).__name__})"
            )
        for field in ("_img", "_c"):
            if field not in record:
                raise GSNetCacheError(
                    f"Record {key} in {source} has no {field!r} key; it cannot "
                    f"be cached (available keys: {sorted(record)})"
                )
        coeffs = _coerce_coefficients(record["_c"], source=source, key=key)
        frame = _coerce_image(record["_img"], source=source, key=key)

        c_chunks.append(coeffs)
        c_offsets[position + 1] = c_offsets[position] + coeffs.size
        img_chunks.append(frame.reshape(-1))
        img_offsets[position + 1] = img_offsets[position] + frame.size
        img_shapes[position, 0] = frame.shape[0]
        img_shapes[position, 1] = frame.shape[1]

        if log_every > 0 and (position + 1) % log_every == 0:
            logger.info(
                "GSNet cache {}: {}/{} records", source, position + 1, len(keys)
            )

    keys_array = np.asarray(keys, dtype=KEY_DTYPE)
    c_flat = (
        np.concatenate(c_chunks) if c_chunks else np.zeros(0, dtype=C_DTYPE)
    ).astype(C_DTYPE, copy=False)
    img_flat = (
        np.concatenate(img_chunks) if img_chunks else np.zeros(0, dtype=IMG_DTYPE)
    ).astype(IMG_DTYPE, copy=False)

    directory.mkdir(parents=True, exist_ok=True)
    # Arrays first, meta last: a build interrupted here leaves no meta.json, so
    # _cache_is_current() says "not current" and the next call rebuilds.
    arrays: tuple[tuple[str, np.ndarray], ...] = (
        ("c_flat.npy", c_flat),
        ("c_offsets.npy", c_offsets),
        ("img_flat.npy", img_flat),
        ("img_offsets.npy", img_offsets),
        ("img_shapes.npy", img_shapes),
        ("keys.npy", keys_array),
    )
    written = 0
    for name, array in arrays:
        np.save(directory / name, array, allow_pickle=False)
        written += int(array.nbytes)

    meta = {
        "source": _absolute(source),
        "n_records": len(keys),
        "format_version": CACHE_FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (directory / META_FILENAME).write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return len(keys), written


def prepare_gsnet_cache(
    index: RecordIndex,
    cache_root: Path | None = None,
    *,
    log_every: int = 200,
) -> list[Path]:
    """Build (or refresh) the lean cache for every pickle in ``index``.

    Idempotent: a cache directory whose ``meta.json`` is readable, whose
    ``format_version`` matches this module's and whose ``source`` is the same
    absolute pickle is **reused**, not rebuilt (so a second call over the real
    9.6 GB corpus is instant and opens no pickle). Anything else -- missing dir,
    stale version, moved source, truncated array -- is rebuilt.

    Records are taken from the pickle's **own** integer keys, not from the index
    entries, so the family always describes the whole file. A record that cannot
    be cached (missing/empty ``_c``, non-2-D or non-uint8 ``_img``, non-dict
    record, non-dict root) aborts that one family with :class:`GSNetCacheError`;
    the Dataset treats that as recoverable and falls back to ``pickle.load`` for
    that file.

    MEMORY: peak RAM is ONE source payload (unavoidable -- ``pickle.load``
    cannot stream) plus its flat buffers. Nothing is retained.

    Args:
        index: Record index from
            :func:`~ao_shaping.runners.gsnet_offline.build_record_index`; the
            pickles are the distinct paths of its entries, in index order.
        cache_root: Explicit cache root, or ``None`` for the default sibling
            ``.gsnet_cache`` directory next to each pickle.
        log_every: Progress-log interval in records per family. ``<= 0`` silences
            the per-file progress lines.

    Returns:
        One cache directory per distinct source pickle, in index order.

    Raises:
        TypeError: If ``index`` is not a :class:`RecordIndex`.
        GSNetCacheError: If a family cannot be built (see above).
    """
    if not isinstance(index, RecordIndex):
        raise TypeError(
            f"index must be a RecordIndex from build_record_index, "
            f"got {type(index).__name__}"
        )

    sources: list[Path] = []
    seen: set[Path] = set()
    for path, _key in index.entries:
        if path not in seen:
            seen.add(path)
            sources.append(path)

    directories: list[Path] = []
    for source in sources:
        directory = cache_dir_for(source, cache_root)
        if _cache_is_current(source, directory):
            logger.info("Reusing current GSNet cache {}", directory)
            directories.append(directory)
            continue
        n_records, written = _build_family(source, directory, log_every=log_every)
        logger.info(
            "Built GSNet cache {} ({} records, {:.1f} MiB of arrays) for {}",
            directory,
            n_records,
            written / (1024 * 1024),
            source,
        )
        directories.append(directory)
    return directories
