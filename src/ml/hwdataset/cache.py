"""On-disk cache of the derived, grid-sized arrays for the hardware dataset.

This module is the **cache** half of :mod:`ml.hwdataset`. It stores what the
materialiser produces so that an epoch reads memory-mapped ``.npy`` files instead
of re-evaluating Zernike / freeform panels and re-reading multi-GB pickles.

Why this module exists (measured on the real corpus, not hypothetical)
--------------------------------------------------------------------
A complete epoch over ``data/debug`` was timed end to end: **11,393 records in
450 s single-threaded** (179 batches of 64, 25.3 records/s). Nearly half of that
is pure, deterministic, recomputable work:

* Zernike panel reconstruction: 157 ms/record on ``slm_zernike_shaping`` (1010
  records) and 277 ms/record on ``slm_pib_online`` (192 records) -- about
  **212 s of the 450 s**, all of it :func:`~ml.hwataset.transforms.zernike_coeffs_to_panel`
  evaluating a Zernike grid over a 1200x1920 panel;
* freeform panel reconstruction: ~80 ms/record on ``slm_gsnet_square`` and
  ``sim_calib_abba`` (126 records) -- ~10 s;
* reading and unpickling 65.7 GB per epoch -- ~35 s.

Caching the derived grid-sized arrays removes all three: after a one-off build an
epoch costs a page-in per record instead of a Zernike evaluation, and peak RAM
drops from "one whole pickle, up to 3.42 GB" to "one page".

The ``slm_pib`` family (7866 records, 69% of the corpus) costs ~6.7 ms/record and
gains little on its own, but it is cached too: the build is a single pass anyway
and a partially cached corpus is far more confusing than a uniform one.

Storage layout
--------------
One directory per source pickle, by default the sibling
``<pkl_dir>/.hw_cache/<pkl_stem>/``::

    phase_cos.npy   (N, grid, grid)  float32   Re of the coherent block mean
    phase_sin.npy   (N, grid, grid)  float32   Im of the coherent block mean
    image.npy       (N, grid, grid)  float32   far-field grid, absolute /255
    exposure.npy    (N,)             float64   exposure_ms, NaN when unknown
    keys.npy        (N,)             int64     the source pickle's record keys
    fov_px.npy      (N,)             int32     source frame side, 0 when unknown
    meta.json                                       <-- written LAST

The rules follow the existing precedent in
:mod:`ml.gsnet_debug.cache`: plain numeric ``.npy`` written with
``allow_pickle=False`` so ``np.load(..., mmap_mode="r", allow_pickle=False)``
always works; ``meta.json`` written **last** as the "build finished" marker so an
interrupted build leaves no meta and is simply rebuilt; a ``format_version``
that invalidates every existing cache when bumped; and accessors that **copy**
the slice they hand out, so a caller can never reach -- or outlive -- a mapping.

Unlike that module there are no ragged offsets: the grid is baked into the cache,
so every array is fixed-shape.

``meta.json`` records ``format_version``, the absolute ``source`` path,
``n_records``, ``n_skipped``, ``grid``, and every :class:`MaterialiserConfig`
field that affects the numbers. A cache whose meta does not match the requested
configuration is **stale** and is rebuilt, because reusing it would silently
return different tensors than the direct path.

Size: 11,393 records x 3 x 64 x 64 x 4 B is about **560 MB** for the whole
corpus at ``grid=64``.

Relationship to the direct path
-------------------------------
:func:`prepare_hw_cache` calls the very same
:class:`~ml.hwdataset.records.Materialiser` the Dataset uses, and takes its
records from the very same :class:`~ml.hwdataset.index.HwCorpusIndex` -- it never
re-classifies anything. The cached arrays are therefore **bit-identical** to what
a direct read produces, which is the property the test suite pins.

The cache is not a write-only artefact: :class:`~ml.hwdataset.records.Materialiser`
reads it back by default (``use_cache=True``), and so does the Dataset through it.
Anything that would make a cached read differ from a direct one -- a different
:class:`MaterialiserConfig`, a half-written directory, a ref that was skipped when
the cache was built -- makes the reader fall back to the direct path for that
record instead of serving stale numbers.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ml.hwdataset.index import HwCorpusIndex, HwRecordRef
from ml.hwdataset.records import HwRecordError, Materialiser, MaterialiserConfig

__all__ = [
    "CACHE_FILENAMES",
    "CACHE_FORMAT_VERSION",
    "DEFAULT_CACHE_DIR_NAME",
    "HwCacheError",
    "HwCachedSource",
    "cache_dir_for",
    "cache_is_current",
    "load_hw_cache",
    "prepare_hw_cache",
]

#: On-disk schema version. Bumping it invalidates every existing cache.
#: v2 added ``fov_px.npy`` so a cache-served sample can report its source frame
#: geometry without re-opening the pickle.
CACHE_FORMAT_VERSION: int = 2

#: Directory created next to each source pickle when no explicit root is given.
#: The leading dot keeps it out of casual ``ls`` views and out of the recursive
#: ``*.pkl`` index glob.
DEFAULT_CACHE_DIR_NAME: str = ".hw_cache"

META_FILENAME: str = "meta.json"

#: Every file a complete cache directory holds, in write order. The arrays go
#: first and ``meta.json`` last, so meta presence is the "build finished" mark.
CACHE_FILENAMES: tuple[str, ...] = (
    "phase_cos.npy",
    "phase_sin.npy",
    "image.npy",
    "exposure.npy",
    "keys.npy",
    "fov_px.npy",
    META_FILENAME,
)

_PHASE_DTYPE = np.dtype(np.float32)
# float64 for exposure, not float32: the cache promises bit-identical samples, and
# float32 cannot round-trip a value like 0.1 ms (it comes back as
# 0.10000000149011612). The array is (N,) -- 8 B x 11393 records is 91 KB -- so
# exactness is free here. The synthetic corpus hid this because its exposures
# (1.0 / 2.0 / 0.5) are exactly float32-representable.
_EXPOSURE_DTYPE = np.dtype(np.float64)
_KEY_DTYPE = np.dtype(np.int64)
_FOV_DTYPE = np.dtype(np.int32)


class HwCacheError(RuntimeError):
    """A hardware-dataset cache is missing, stale, or structurally corrupt.

    Raised by :func:`load_hw_cache` and :func:`prepare_hw_cache`. The latter
    treats it as *recoverable*: one bad family is logged and skipped so a single
    corrupt pickle cannot cost the whole corpus.
    """


# ---------------------------------------------------------------------------
# Paths / meta
# ---------------------------------------------------------------------------
def cache_dir_for(pkl_path: Path, cache_root: Path | None = None) -> Path:
    """Return the cache directory belonging to one source pickle.

    The default is a sibling of the pickle -- ``<pkl_dir>/.hw_cache/<stem>/`` --
    which keeps the cache next to the data it mirrors.

    Args:
        pkl_path: The source debug pickle.
        cache_root: Optional explicit root. When given the directory is
            ``<cache_root>/<stem>``; the root is used as given.

    Returns:
        The cache directory. It is not created here.
    """
    source = Path(pkl_path)
    root = source.parent / DEFAULT_CACHE_DIR_NAME if cache_root is None else Path(cache_root)
    return root / source.stem


def _absolute(path: Path) -> str:
    """Absolute, symlink-resolved string form of ``path``, for ``meta.json``."""
    return str(Path(path).resolve())


def _config_to_meta(config: MaterialiserConfig) -> dict[str, Any]:
    """Serialise a :class:`MaterialiserConfig` into JSON-compatible fields."""
    meta = asdict(config)
    # Tuples become lists in JSON; normalise so a round-trip compares equal.
    meta["panel_center"] = list(meta["panel_center"])
    meta["panel_resolution"] = list(meta["panel_resolution"])
    return meta


def _meta_to_config(meta: dict[str, Any]) -> MaterialiserConfig:
    """Rebuild a :class:`MaterialiserConfig` from ``meta.json`` fields.

    Raises:
        KeyError: If a required field is absent.
        TypeError: If a field has the wrong type.
    """
    center = meta["panel_center"]
    resolution = meta["panel_resolution"]
    return MaterialiserConfig(
        grid=int(meta["grid"]),
        slm_max_gray=int(meta["slm_max_gray"]),
        panel_center=(int(center[0]), int(center[1])),
        panel_radius=float(meta["panel_radius"]),
        panel_resolution=(int(resolution[0]), int(resolution[1])),
        image_mode=str(meta["image_mode"]),
        zernike_radius=float(meta["zernike_radius"]),
        freeform_radius=float(meta["freeform_radius"]),
    )


def _read_meta(directory: Path) -> dict[str, Any] | None:
    """Read and parse ``meta.json``, or return ``None`` when unusable.

    Never raises: a missing file, unreadable file, non-JSON payload or non-object
    JSON all yield ``None`` so callers treat "no cache" and "garbage cache"
    identically.
    """
    meta_path = Path(directory) / META_FILENAME
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug("Unreadable hw cache meta {}: {}", meta_path, exc)
        return None
    if not isinstance(payload, dict):
        logger.debug("Hw cache meta {} is not a JSON object", meta_path)
        return None
    return payload


def _config_matches(meta: dict[str, Any], config: MaterialiserConfig) -> bool:
    """Whether ``meta`` was written with exactly ``config``."""
    try:
        return _meta_to_config(meta) == config
    except (KeyError, TypeError, ValueError, IndexError):
        return False


# ---------------------------------------------------------------------------
# The family
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class HwCachedSource:
    """One source pickle's derived arrays, memory-mapped or eager.

    Construct it with :func:`load_hw_cache`, never directly. Every accessor
    copies the slice it hands out, so the returned arrays are ordinary owned
    numpy arrays: the caller may mutate or retain them freely, and closing the
    family cannot invalidate them.

    After :meth:`close` the arrays are released and every accessor raises
    :class:`HwCacheError`.
    """

    directory: Path
    source: str
    n_records: int
    grid: int
    _phase_cos: np.ndarray | None
    _phase_sin: np.ndarray | None
    _image: np.ndarray | None
    _exposure: np.ndarray | None
    _keys: np.ndarray | None
    _fov_px: np.ndarray | None

    def _require(self, array: np.ndarray | None, name: str) -> np.ndarray:
        """Return an array, or explain that the family was already closed."""
        if array is None:
            raise HwCacheError(
                f"This HwCachedSource has been closed; its memory maps were "
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
        """The cached record keys, as an owned ``(N,)`` int64 array.

        A ``Recorder``-sourced family stores the record *position* here, since
        those payloads have no integer key of their own.
        """
        return np.array(self._require(self._keys, "keys"), dtype=np.int64)

    def position_of(self, key: int) -> int:
        """Map a stored key back to its 0-based slot in this family.

        This is the lookup a caller holding a single
        :class:`~ml.hwdataset.index.HwRecordRef` needs, because the ref carries
        no rank of its own. It is exact whenever the value passed in is the one
        :meth:`keys` stores for that ref -- ``ref.key`` for a ``dict`` payload,
        ``ref.position`` for a ``Recorder`` payload (which has no integer key) --
        because the builder writes each ref's own stored key into that ref's own
        slot. It stays exact even when ``key != position``, which happens for a
        real file in this corpus (``sim_calib_abba_20261001_163743.pkl`` stores
        keys 10000-10011 at positions 6-17) and whenever records were excluded,
        since both shift the slot without touching the stored key.

        Args:
            key: The ref's stored key, from :meth:`keys`.

        Returns:
            The ref's 0-based slot, suitable for the array accessors.

        Raises:
            KeyError: If ``key`` is not cached in this family. Callers treat this
                as "this ref is not cached" and fall back to the direct path.
        """
        keys = np.asarray(self._require(self._keys, "keys"))
        matches = np.flatnonzero(keys == int(key))
        if matches.size == 0:
            raise KeyError(f"key {key} is not cached in {self.directory}")
        return int(matches[0])

    def fov_px(self, i: int) -> int | None:
        """Record ``i``'s source frame side, or ``None`` when it was unknown."""
        index = self._index(i, "fov_px")
        value = int(self._require(self._fov_px, "fov_px")[index])
        return None if value <= 0 else value

    def phase_cos(self, i: int) -> np.ndarray:
        """Record ``i``'s ``(grid, grid)`` float32 Re component, owned."""
        return self._copy2d(self._phase_cos, i, "phase_cos")

    def phase_sin(self, i: int) -> np.ndarray:
        """Record ``i``'s ``(grid, grid)`` float32 Im component, owned."""
        return self._copy2d(self._phase_sin, i, "phase_sin")

    def image(self, i: int) -> np.ndarray:
        """Record ``i``'s ``(grid, grid)`` float32 far-field grid, owned."""
        return self._copy2d(self._image, i, "image")

    def _copy2d(self, array: np.ndarray | None, i: int, name: str) -> np.ndarray:
        """Copy one ``(grid, grid)`` record out of a mapped array.

        Args:
            array: The mapped array, or ``None`` once closed.
            i: 0-based record position.
            name: Field name for the error messages.

        Returns:
            An owned ``(grid, grid)`` float32 array.

        Raises:
            IndexError: If ``i`` is out of range.
            HwCacheError: If the family is closed.
        """
        index = self._index(i, name)
        source = self._require(array, name)
        # np.array(copy=True): a memmap row is only a view, and the caller must
        # not be able to reach (or outlive) the mapping through it.
        return np.array(source[index], dtype=np.float32)

    def exposure_ms(self, i: int) -> float | None:
        """Record ``i``'s exposure, or ``None`` when it could not be resolved."""
        index = self._index(i, "exposure")
        value = float(self._require(self._exposure, "exposure")[index])
        return None if math.isnan(value) else value

    def close(self) -> None:
        """Release the memory maps. Idempotent, and safe on an eager family.

        A record handed out earlier is unaffected: every accessor copies, so no
        exported view can be keeping a mapping alive. A ``BufferError`` here is
        logged rather than raised and the reference is still dropped.
        """
        for name in ("_phase_cos", "_phase_sin", "_image", "_exposure", "_keys", "_fov_px"):
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
            object.__setattr__(self, name, None)

    def __len__(self) -> int:
        return self.n_records


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_array(path: Path, *, mmap: bool) -> np.ndarray:
    """``np.load`` one cache array with pickling disabled.

    Raises:
        HwCacheError: If the file is missing or unreadable as a numeric array.
    """
    try:
        return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise HwCacheError(f"Could not read cache array {path}: {exc}") from exc


def _expect(array: np.ndarray, *, dtype: np.dtype, shape: tuple[int, ...], path: Path) -> None:
    """Validate one loaded array's dtype and shape.

    Raises:
        HwCacheError: On any mismatch. This is what turns a truncated cache into
            a reported error instead of a silently wrong sample.
    """
    if array.dtype != dtype:
        raise HwCacheError(f"{path} has dtype {array.dtype}, expected {dtype}")
    if array.shape != shape:
        raise HwCacheError(f"{path} has shape {array.shape}, expected {shape}")


def load_hw_cache(directory: Path, *, mmap: bool = True) -> HwCachedSource:
    """Open a cache directory as an :class:`HwCachedSource`.

    The whole layout is validated up front -- meta, dtypes and shapes -- so a
    truncated or half-written cache is reported here rather than producing a
    silently wrong sample.

    Args:
        directory: A directory produced by :func:`prepare_hw_cache`.
        mmap: Memory-map the arrays (default) or read them eagerly.

    Returns:
        An open :class:`HwCachedSource`.

    Raises:
        HwCacheError: If the directory is missing, its ``meta.json`` is unreadable
            or stale, a file is missing, or an array has the wrong dtype/shape.
    """
    directory = Path(directory)
    meta = _read_meta(directory)
    if meta is None:
        raise HwCacheError(f"No readable {META_FILENAME} in {directory}")

    version = meta.get("format_version")
    if version != CACHE_FORMAT_VERSION:
        raise HwCacheError(
            f"{directory} was written with format_version {version!r}, "
            f"this module speaks {CACHE_FORMAT_VERSION}"
        )

    n_records = meta.get("n_records")
    if not isinstance(n_records, int) or isinstance(n_records, bool) or n_records < 0:
        raise HwCacheError(f"{directory} has an invalid n_records: {n_records!r}")
    grid = meta.get("grid")
    if not isinstance(grid, int) or isinstance(grid, bool) or grid < 1:
        raise HwCacheError(f"{directory} has an invalid grid: {grid!r}")

    missing = [name for name in CACHE_FILENAMES if not (directory / name).is_file()]
    if missing:
        raise HwCacheError(f"{directory} is incomplete, missing {missing}")

    keys_path = directory / "keys.npy"
    keys = _load_array(keys_path, mmap=mmap)
    _expect(keys, dtype=_KEY_DTYPE, shape=(n_records,), path=keys_path)

    def grid_array(name: str) -> np.ndarray:
        path = directory / name
        array = _load_array(path, mmap=mmap)
        _expect(array, dtype=_PHASE_DTYPE, shape=(n_records, grid, grid), path=path)
        return array

    exposure_path = directory / "exposure.npy"
    exposure = _load_array(exposure_path, mmap=mmap)
    _expect(exposure, dtype=_EXPOSURE_DTYPE, shape=(n_records,), path=exposure_path)

    fov_path = directory / "fov_px.npy"
    fov = _load_array(fov_path, mmap=mmap)
    _expect(fov, dtype=_FOV_DTYPE, shape=(n_records,), path=fov_path)

    return HwCachedSource(
        directory=directory,
        source=str(meta.get("source", "")),
        n_records=n_records,
        grid=grid,
        _phase_cos=grid_array("phase_cos.npy"),
        _phase_sin=grid_array("phase_sin.npy"),
        _image=grid_array("image.npy"),
        _exposure=exposure,
        _keys=keys,
        _fov_px=fov,
    )


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
def cache_is_current(
    source: Path, directory: Path, config: MaterialiserConfig
) -> bool:
    """Whether ``directory`` is a reusable, intact cache for ``source`` + ``config``.

    Public entry point for a *reader* that only wants to know whether it may use
    an existing cache -- the materialiser asks this before serving a record, so
    that a cache built with a different :class:`MaterialiserConfig` (a different
    ``grid``, ``panel_radius``, ``image_mode``, ...) is ignored rather than
    silently returning differently-shaped or differently-valued tensors than the
    direct path would.

    Args:
        source: The source pickle the cache must belong to.
        directory: The cache directory to test.
        config: The configuration the caller intends to materialise with.

    Returns:
        ``True`` only if the cache is current, complete and loadable.
    """
    return _cache_is_current(Path(source), Path(directory), config)


def _cache_is_current(source: Path, directory: Path, config: MaterialiserConfig) -> bool:
    """Whether ``directory`` is a reusable, intact cache for ``source``.

    Two layers, both cheap: the meta contract (version, source path, config, file
    presence) and a full :func:`load_hw_cache` pass, so a *truncated* array is
    treated as corrupt instead of reusable. ``np.load(mmap_mode="r")`` maps only
    the header and the offsets are gone in this layout, so the structural check
    costs microseconds and never page-ins a record.
    """
    meta = _read_meta(directory)
    if meta is None:
        return False
    if meta.get("format_version") != CACHE_FORMAT_VERSION:
        return False
    if meta.get("source") != _absolute(source):
        return False
    if not _config_matches(meta, config):
        logger.debug(
            "Hw cache {} was built with a different MaterialiserConfig; rebuilding",
            directory,
        )
        return False
    if not all((directory / name).is_file() for name in CACHE_FILENAMES):
        return False
    try:
        load_hw_cache(directory, mmap=True).close()
    except HwCacheError as exc:
        logger.debug("Hw cache {} failed validation: {}", directory, exc)
        return False
    except OSError as exc:
        logger.debug("Hw cache {} could not be mapped: {}", directory, exc)
        return False
    return True


def _build_family(
    source: Path,
    records: tuple[HwRecordRef, ...],
    directory: Path,
    *,
    config: MaterialiserConfig,
    log_every: int,
) -> tuple[int, int]:
    """Write the complete cache directory for one source pickle.

    The records come from the index and are materialised by the same
    :class:`~ml.hwdataset.records.Materialiser` the Dataset uses, so the cached
    arrays are bit-identical to a direct read and the source pickle is
    deserialised exactly **once** for the whole family (the materialiser's LRU
    holds it while the group is walked).

    A record the materialiser rejects is *skipped*, not written as a hole: the
    emitted arrays are assembled from the successes only, so ``keys[i]`` always
    addresses ``phase_*[i]`` / ``image[i]``.

    MEMORY: peak RAM is the small assembled buffers (about 35 MB for the largest
    733-record family at ``grid=64``); the source payload is bounded by the
    materialiser's 1-entry LRU and is released when the family ends.

    Args:
        source: The pickle this cache mirrors.
        records: Its index entries, in the order they should be cached.
        directory: Destination directory (created if absent).
        config: The materialisation configuration; baked into the meta.
        log_every: Progress-log interval in records (``<= 0`` disables it).

    Returns:
        ``(n_records_cached, bytes_written)``.

    Raises:
        HwCacheError: If an array cannot be written to disk.
    """
    # use_cache=False is mandatory, not an optimisation: the builder's output *is*
    # the cache, so letting the materialiser read a pre-existing one would mean
    # writing a cache from itself. On Windows it is worse than pointless -- an open
    # memory map makes the target file un-overwritable, so every np.save below
    # fails with "Invalid argument" (errno 22).
    materialiser = Materialiser(config=config, cache_size=1, use_cache=False)
    cos_chunks: list[np.ndarray] = []
    sin_chunks: list[np.ndarray] = []
    image_chunks: list[np.ndarray] = []
    exposures: list[float] = []
    keys: list[int] = []
    fovs: list[int] = []
    skipped = 0

    for position, ref in enumerate(records):
        try:
            sample = materialiser.materialise(ref)
        except HwRecordError as exc:
            skipped += 1
            logger.debug("Skipping {} position {}: {}", source, ref.position, exc)
            continue
        cos_chunks.append(sample.phase_cos)
        sin_chunks.append(sample.phase_sin)
        image_chunks.append(sample.image)
        exposures.append(
            np.nan if sample.exposure_ms is None else float(sample.exposure_ms)
        )
        fovs.append(0 if sample.fov_px is None else int(sample.fov_px))
        # Recorder payloads have no integer key of their own; the position is
        # the only stable identifier there, and it is what the index addresses.
        keys.append(int(ref.key) if ref.key is not None else int(ref.position))
        if log_every > 0 and (position + 1) % log_every == 0:
            logger.info("Hw cache {}: {}/{} records", source, position + 1, len(records))

    n_records = len(cos_chunks)
    grid = config.grid
    empty2d = np.zeros((0, grid, grid), dtype=np.float32)
    phase_cos = (
        np.stack(cos_chunks).astype(_PHASE_DTYPE, copy=False)
        if cos_chunks
        else empty2d
    )
    phase_sin = (
        np.stack(sin_chunks).astype(_PHASE_DTYPE, copy=False)
        if sin_chunks
        else empty2d
    )
    image = (
        np.stack(image_chunks).astype(_PHASE_DTYPE, copy=False)
        if image_chunks
        else empty2d
    )
    exposure = np.asarray(exposures, dtype=_EXPOSURE_DTYPE)
    key_array = np.asarray(keys, dtype=_KEY_DTYPE)
    fov_array = np.asarray(fovs, dtype=_FOV_DTYPE)

    directory.mkdir(parents=True, exist_ok=True)
    # Arrays first, meta last: a build interrupted here leaves no meta.json, so
    # _cache_is_current() reports "not current" and the next call rebuilds.
    arrays: tuple[tuple[str, np.ndarray], ...] = (
        ("phase_cos.npy", phase_cos),
        ("phase_sin.npy", phase_sin),
        ("image.npy", image),
        ("exposure.npy", exposure),
        ("keys.npy", key_array),
        ("fov_px.npy", fov_array),
    )
    written = 0
    for name, array in arrays:
        try:
            np.save(directory / name, array, allow_pickle=False)
        except OSError as exc:
            raise HwCacheError(f"Could not write {directory / name}: {exc}") from exc
        written += int(array.nbytes)

    meta: dict[str, Any] = {
        "format_version": CACHE_FORMAT_VERSION,
        "source": _absolute(source),
        "n_records": n_records,
        "n_skipped": skipped,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **_config_to_meta(config),
    }
    (directory / META_FILENAME).write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return n_records, written


def prepare_hw_cache(
    index: HwCorpusIndex,
    *,
    config: MaterialiserConfig | None = None,
    cache_root: Path | None = None,
    log_every: int = 50,
    reuse: bool = True,
) -> list[Path]:
    """Build (or refresh) the cache for every distinct pickle in ``index``.

    Idempotent: a directory whose meta matches both the source pickle and the
    requested configuration is **reused**, so a second call over the real corpus
    opens no pickle at all. Anything else -- missing directory, stale version,
    moved source, changed config, truncated array -- is rebuilt.

    One family that cannot be built is logged and skipped rather than raised, so
    a single corrupt pickle costs one family instead of the whole corpus.

    MEMORY: peak RAM is one source payload (up to 3.42 GB for the largest pickle
    in this corpus) plus the assembled buffers for one family. Nothing is
    retained afterwards.

    Args:
        index: The index whose pickles to cache; its records supply the order and
            the per-record provenance.
        config: Materialisation configuration. Baked into every cache's meta, so
            changing it invalidates every existing cache. Defaults to
            ``MaterialiserConfig()``.
        cache_root: Explicit cache root, or ``None`` for the sibling
            ``.hw_cache`` directory next to each pickle.
        log_every: Progress-log interval in records per family (``<= 0`` silences
            the per-family progress lines).
        reuse: Reuse a matching cache instead of rebuilding it.

    Returns:
        One cache directory per successfully handled source pickle, in index order.

    Raises:
        TypeError: If ``index`` is not an :class:`HwCorpusIndex`.
    """
    if not isinstance(index, HwCorpusIndex):
        raise TypeError(
            f"index must be a HwCorpusIndex from build_hw_index, "
            f"got {type(index).__name__}"
        )
    if config is None:
        config = MaterialiserConfig()

    grouped: dict[Path, list[HwRecordRef]] = {}
    for ref in index.records:
        grouped.setdefault(Path(ref.path).resolve(), []).append(ref)

    directories: list[Path] = []
    for source, refs in grouped.items():
        directory = cache_dir_for(source, cache_root)
        if reuse and _cache_is_current(source, directory, config):
            logger.info("Reusing current hw cache {}", directory)
            directories.append(directory)
            continue
        try:
            n_records, written = _build_family(
                source, tuple(refs), directory, config=config, log_every=log_every
            )
        except (HwCacheError, OSError, MemoryError) as exc:
            logger.warning("Skipping uncacheable {}: {}", source, exc)
            continue
        logger.info(
            "Built hw cache {} ({} records, {:.1f} MiB) for {}",
            directory,
            n_records,
            written / (1024 * 1024),
            source,
        )
        directories.append(directory)
    return directories
