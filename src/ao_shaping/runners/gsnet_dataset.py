"""Lazy PyTorch Dataset, file-grouped sampler and DataLoader factory.

This is the **batching** half of the offline GSNet training feature. The
transform half already exists in :mod:`ao_shaping.runners.gsnet_offline`; this
module only wraps it in a :class:`torch.utils.data.Dataset` (plus a sampler and
a ``DataLoader`` factory) so the debug pickles can be streamed into
``ml.gsnet.FourierGSNet`` without ever holding the corpus in RAM.

===========================  What one sample is ===========================

``dataset[i]`` returns a **3-tuple of** ``torch.Tensor``, each ``float32`` with
shape ``(1, grid, grid)`` -- exactly the layout
:func:`ml.gsnet.train.train_gsnet` iterates::

    for source, target, gt_phase in dataloader:
        source_amp = torch.sqrt(source.clamp_min(0.0) + 1e-12)
        pred_phase = model(source, target)

=====================  Item 1: ``source`` (pupil illumination)  ==========

The SLM / source-plane intensity, produced by the **canonical** factory
:func:`ml.gsnet.dataset.make_source_intensity` -- *not* the CCD image. It does
not depend on the record, so it is built **once** in :meth:`__init__` and
re-served (as a fresh ``clone``) on every access. Its one per-sample cost is a
``(1, grid, grid)`` copy, not a Zernike evaluation.

====================  Item 2: ``target`` (measured far field)  ==========

The measured CCD frame ``record["_img"]`` passed through
:func:`~ao_shaping.runners.gsnet_offline.farfield_to_grid` (argmax-anchored
crop around the 0-order, peak-normalised to ``[0, 1]``, float32).

==================  Item 3: ``gt_phase`` (SLM pupil phase)  ============

Reconstructed from ``record["_c"]`` (Noll-order Zernike coefficients in
**radians**) in three steps::

    n_max      = infer_n_max(len(c))                    # never hardcoded
    phase_rad  = reconstruct_pupil_phase_rad(c, n_max=..., slm_width=...,
                                              slm_height=..., radius=...)
    gt_phase   = pupil_phase_to_grid(phase_rad, grid)  # exact block mean

``len(_c)`` is **not** constant across the corpus (15, 66, 78 ... are all
observed), so ``n_max`` is always inferred from the record at hand.

=========================  The lean on-disk cache  =========================

``pickle`` cannot be memory-mapped, so the LRU above can only ever amortise a
file that is already resident -- and with 26 interleaved files the corpus is
re-loaded constantly (measured: 3.5 samples/s, ~15 min/epoch, RSS peaking at
~2.5 GB). :mod:`ao_shaping.runners.gsnet_cache` therefore stores only the two
live fields per record (``_c`` and ``_img``; together 2.0 % of the 9.6 GB raw
bytes -- ``_phase`` is never used) as plain ``.npy`` arrays that
``np.load(..., mmap_mode="r")`` can map. A cached record then costs a page-in
instead of a multi-hundred-MB deserialisation.

``use_cache=True`` (the default) makes :meth:`GSNetDebugDataset._load_payload`
read from that cache when one exists; ``use_cache=False`` restores the plain
``pickle.load`` path. Either way the payload handed to :meth:`__getitem__` is a
``dict`` of record dicts, so the transform chain -- and therefore
``(source, target, gt_phase)`` -- is provably untouched: the two paths are
bit-identical (see ``tests/ao_shaping/runners/test_gsnet_cache.py``).

Three behaviours are worth stating explicitly:

* **Absent cache -> warn once, use ``pickle.load``.** Building a multi-GB cache
  is not something to do implicitly inside a ``__getitem__`` (it would stall
  the first sample for minutes, once per worker process). Call
  :func:`~ao_shaping.runners.gsnet_cache.prepare_gsnet_cache` once per corpus.
* **Corrupt cache -> warn, rebuild that one family, carry on.** A single bad
  family never raises; if the rebuild also fails, the pickle path is used.
* **A record that genuinely lacks ``_c`` / ``_img`` still raises**
  :class:`GSNetRecordError` -- that is a corrupt record, not a storage problem.

:meth:`clear_cache` drops the *in-RAM* payload cache and releases the mappings.
It never deletes anything from disk; deleting a cache is not this class's job.

===========================  Memory contract ============================

The corpus is ~5.2 GB across ~25 pickles and the largest single pickle is
~472 MB, while ``pickle.load`` cannot stream. This module therefore:

* :meth:`GSNetDebugDataset.__init__` opens **no** pickle beyond what
  :func:`~ao_shaping.runners.gsnet_offline.build_record_index` already did, and
  retains **no arrays** -- only the immutable
  :class:`~ao_shaping.runners.gsnet_offline.RecordIndex` of ``(Path, int)``
  pairs, which is itself pickle-free of large payloads.
* :meth:`GSNetDebugDataset.__getitem__` deserialises **exactly one** pickle and
  pulls one record out of it. The file payload is then referenced *only* by the
  bounded LRU cache (or by nothing at all, if ``cache_size=0`` or it is the next
  entry to be evicted) -- never by the returned tensors.
* A tiny LRU cache (:class:`collections.OrderedDict`) holds **at most**
  ``cache_size`` (default ``1``) deserialised file payloads, keyed by ``Path``,
  so consecutive samples from the same file cost a single ``pickle.load``. This
  is the *only* structure allowed to retain a payload, and it is bounded by
  construction.
* **No open file handle is ever stored on ``self``.** Every read happens inside
  a ``with open(...)`` block inside :meth:`__getitem__`, because an open handle
  is not picklable and ``DataLoader`` workers must be able to pickle the
  Dataset. :meth:`__getstate__` / :meth:`__setstate__` additionally drop the
  LRU cache on pickling, so a worker never inherits (and re-pickles, on Windows
  ``spawn``) a parent process's half-GB payload.

===========================  ``cache_size`` vs ``num_workers`` ============================

``cache_size`` is **per Dataset instance**, i.e. **per worker process**.
``num_workers=0`` is therefore the default and the recommended setting: on
Windows the default start method is ``spawn``, so every worker process
re-pickles the Dataset and, once it starts reading, holds **its own full copy**
of the corpus file it currently has open. Peak RAM is then
``num_workers * cache_size`` file payloads -- with the defaults
(``num_workers=0``, ``cache_size=1``) that is a single payload (~472 MB for the
largest pickle), and with ``num_workers=2`` it doubles to ~944 MB. Raise
``num_workers`` only when that trade is acceptable; ``cache_size=1`` per worker
is already the minimum that still gives intra-group LRU hits.

===========================  Sampler ============================

:class:`FileGroupedSampler` shuffles indices **within** each pickle-file group
and then shuffles the groups themselves, so consecutive ``__getitem__`` calls
hit the same file and the LRU cache actually pays off. A plain
``RandomSampler`` would defeat the cache and re-deserialise the 472 MB pickle
on most steps.

===========================  Malformed records ============================

A record that cannot be turned into a sample **raises**
:class:`GSNetRecordError` (a :class:`RuntimeError` subclass) -- it is **not**
silently skipped. Skipping is impossible by construction: ``dataset[i]`` is
addressed positionally through the :class:`RecordIndex`, so dropping a record
mid-epoch would silently shift the mapping between indices and files and
corrupt any ``num_samples`` arithmetic. The offending path and record key are
always included in the message. An out-of-range ``idx`` raises the standard
``IndexError`` that ``DataLoader`` relies on.
"""

from __future__ import annotations

import pickle
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from ml.gsnet.dataset import make_source_intensity
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler

from ao_shaping.runners.gsnet_cache import (
    CachedFamily,
    GSNetCacheError,
    cache_dir_for,
    close_cached_payload,
    load_cached_family,
    prepare_gsnet_cache,
)
from ao_shaping.runners.gsnet_offline import (
    RecordIndex,
    farfield_to_grid,
    infer_n_max,
    pupil_phase_to_grid,
    reconstruct_pupil_phase_rad,
)

__all__ = [
    "DEFAULT_PREFETCH_FACTOR",
    "FileGroupedSampler",
    "GSNetDebugDataset",
    "GSNetRecordError",
    "build_gsnet_dataloader",
]

#: Prefetch depth used when ``num_workers > 0``. Never passed when
#: ``num_workers == 0`` -- torch forbids a non-``None`` ``prefetch_factor``
#: without worker processes.
DEFAULT_PREFETCH_FACTOR: int = 2

#: Default source-illumination family, forwarded to
#: :func:`ml.gsnet.dataset.make_source_intensity`.
DEFAULT_SOURCE_TYPE: str = "gaussian"

#: Real SLM200 panel geometry (the corpus was recorded on this device).
DEFAULT_SLM_WIDTH: int = 1920
DEFAULT_SLM_HEIGHT: int = 1200

#: Zernike aperture radius in pixels. **Not** ``min(width, height) / 2``.
#: The corpus was recorded by the Zernike optimizers, which hard-code
#: ``ZERNIKE_APERTURE_RADIUS = 300.0``
#: (``src/ao_shaping/optimizer/wfless/slm_zernike_pib.py:235`` and
#: ``slm_zernike_shaping.py:225``) and pass it as ``radius=`` to
#: ``PatternHelper.generate_zernike_polynomial``. The beam therefore fills only
#: a 600 px-diameter disc, not the full 1200 px panel height. Reconstructing a
#: real record with 300 instead of 600 shrinks the 64x64 phase footprint from
#: nearly the whole grid to ~540/4096 cells.
DEFAULT_SLM_RADIUS: float = 300.0


class GSNetRecordError(RuntimeError):
    """A debug record could not be converted into a training sample.

    Raised (never silently skipped) when a record is missing the ``_img`` or
    ``_c`` key, when ``_c`` is empty or has a length that is not a triangular
    Zernike mode count, when the indexed key is absent from the pickle, or when
    the pickle root is not a ``dict`` of records. The originating ``Path`` and
    record key are always part of the message. An out-of-range ``idx`` raises
    the standard ``IndexError`` that ``DataLoader`` relies on.
    """


def _to_grid_tensor(array: np.ndarray) -> torch.Tensor:
    """Wrap a fresh ``(grid, grid)`` float32 array as a ``(1, grid, grid)`` tensor.

    ``np.ascontiguousarray(..., dtype=np.float32)`` copies when needed, so the
    returned tensor never aliases the caller's array (or any cached buffer) and
    is therefore safe for the caller to mutate in place.

    Args:
        array: A ``(grid, grid)`` array.

    Returns:
        A ``(1, grid, grid)`` ``torch.float32`` tensor owning its data.
    """
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return torch.from_numpy(contiguous.reshape(1, *contiguous.shape))


class GSNetDebugDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Lazy Dataset over the debug pickles indexed by :func:`build_record_index`.

    See the module docstring for the full data and memory contract. The short
    version: :meth:`__init__` touches no pickle and keeps no array, and
    :meth:`__getitem__` deserialises one file (LRU-cached, bounded by
    ``cache_size``) and returns three independent ``(1, grid, grid)``
    ``float32`` tensors.
    """

    def __init__(
        self,
        index: RecordIndex,
        *,
        grid: int = 64,
        source_type: str = DEFAULT_SOURCE_TYPE,
        slm_width: int = DEFAULT_SLM_WIDTH,
        slm_height: int = DEFAULT_SLM_HEIGHT,
        slm_radius: float = DEFAULT_SLM_RADIUS,
        cache_size: int = 1,
        cache_root: Path | None = None,
        use_cache: bool = True,
    ) -> None:
        """Validate the configuration and build the cached source illumination.

        MEMORY: no pickle is opened here. The only retained payload is the
        ``(1, grid, grid)`` source illumination, which is record-independent.

        Args:
            index: Record index from
                :func:`~ao_shaping.runners.gsnet_offline.build_record_index`.
                Only its ``(Path, int)`` tuples are retained.
            grid: Output grid side length; every returned tensor is
                ``(1, grid, grid)``.
            source_type: ``"gaussian"`` or ``"uniform"``, forwarded to
                :func:`ml.gsnet.dataset.make_source_intensity`.
            slm_width: SLM panel width used by the pupil-phase reconstruction.
            slm_height: SLM panel height used by the pupil-phase reconstruction.
            slm_radius: Zernike aperture radius in pixels. Defaults to
                :data:`DEFAULT_SLM_RADIUS` (300, the corpus device's
                ``ZERNIKE_APERTURE_RADIUS``), **not** half the panel height.
            cache_size: Maximum number of deserialised pickle payloads to keep
                in the LRU cache. ``0`` disables caching entirely; values
                above ``1`` trade RAM for fewer ``pickle.load`` calls when the
                sampler alternates between files.
            cache_root: Root directory for the lean on-disk cache built by
                :mod:`ao_shaping.runners.gsnet_cache`. ``None`` (default) uses
                the sibling ``.gsnet_cache`` directory next to each pickle.
                Only consulted when ``use_cache=True``.
            use_cache: Read records from the lean on-disk cache when one exists
                (default ``True``). ``False`` forces the direct
                ``pickle.load`` path, which yields bit-identical samples more
                slowly.

        Raises:
            TypeError: If ``index`` is not a :class:`RecordIndex`.
            ValueError: If ``grid`` is smaller than 2, if ``cache_size`` is
                negative, or if ``source_type`` is unknown (rejected by
                :func:`ml.gsnet.dataset.make_source_intensity`).
        """
        if not isinstance(index, RecordIndex):
            raise TypeError(
                f"index must be a RecordIndex from build_record_index, "
                f"got {type(index).__name__}"
            )
        if int(grid) < 2:
            raise ValueError(f"grid must be >= 2, got {grid}")
        if int(cache_size) < 0:
            raise ValueError(f"cache_size must be >= 0, got {cache_size}")

        self._entries: tuple[tuple[Path, int], ...] = tuple(index.entries)
        self._grid = int(grid)
        self._source_type = str(source_type)
        self._slm_width = int(slm_width)
        self._slm_height = int(slm_height)
        self._slm_radius = float(slm_radius)
        self._cache_size = int(cache_size)
        self._use_cache = bool(use_cache)
        self._cache_root: Path | None = (
            Path(cache_root) if cache_root is not None else None
        )
        # Sources already reported as having no cache, so the "build it first"
        # warning is emitted once per file instead of once per sample.
        self._cache_absent: set[Path] = set()
        # Bounded LRU: at most ``cache_size`` deserialised payloads, keyed by
        # Path. Deliberately an OrderedDict -- move_to_end + popitem(last=False)
        # is the O(1) LRU primitive torch itself uses.
        self._cache: OrderedDict[Path, dict[Any, Any]] = OrderedDict()

        # The pupil illumination is identical for every sample, so it is built
        # exactly once here and cloned on each access. This is the ONLY array
        # the constructor retains.
        self._source: torch.Tensor = _to_grid_tensor(
            make_source_intensity(self._grid, self._source_type)
        )
        logger.info(
            "GSNetDebugDataset: {} records from {} pickles, grid={}, "
            "source_type={}, cache_size={}, use_cache={}",
            len(self._entries),
            len({p for p, _ in self._entries}),
            self._grid,
            self._source_type,
            self._cache_size,
            self._use_cache,
        )

    # -- introspection -----------------------------------------------------
    @property
    def grid(self) -> int:
        """Output grid side length (every tensor is ``(1, grid, grid)``)."""
        return self._grid

    @property
    def entries(self) -> tuple[tuple[Path, int], ...]:
        """The retained ``(Path, record_key)`` tuples -- paths and ints only."""
        return self._entries

    @property
    def cached_files(self) -> tuple[Path, ...]:
        """Paths currently held by the LRU cache (newest last)."""
        return tuple(self._cache.keys())

    def __len__(self) -> int:
        return len(self._entries)

    # -- cache -------------------------------------------------------------
    def _load_cached_payload(self, path: Path) -> dict[Any, Any] | None:
        """Read one family from the lean on-disk cache, or return ``None``.

        ``None`` means "no usable cache" and tells the caller to use
        ``pickle.load``. That happens in exactly two cases:

        * **absent** -- the cache dir was never built. Building a multi-GB
          cache inside ``__getitem__`` would stall the first sample of every
          worker process for minutes, so absence is reported once per file and
          the pickle path is used. Call
          :func:`~ao_shaping.runners.gsnet_cache.prepare_gsnet_cache` once per
          corpus instead.
        * **unrecoverable** -- the family is corrupt *and* the rebuild also
          failed, so there is nothing left to try.

        Corrupt-but-rebuildable families never reach the caller: they are
        rebuilt in place and then used.

        Args:
            path: Source pickle the cache is derived from.

        Returns:
            ``{record_key: record_dict}`` backed by the family's memory maps,
            or ``None`` to signal the pickle fallback.
        """
        directory = cache_dir_for(path, self._cache_root)

        if not directory.is_dir():
            if path not in self._cache_absent:
                self._cache_absent.add(path)
                logger.warning(
                    "No GSNet cache for {}; reading the pickle directly. Build "
                    "one with prepare_gsnet_cache(...) to speed up training.",
                    path,
                )
            return None

        family: CachedFamily | None = None
        try:
            family = load_cached_family(directory, mmap=True)
            if family.source != str(path.resolve()):
                raise GSNetCacheError(
                    f"cache was built from {family.source}, not {path.resolve()}"
                )
        except GSNetCacheError as exc:
            logger.warning(
                "GSNet cache {} is corrupt ({}); rebuilding it.", directory, exc
            )
            if not self._rebuild_family(path, directory):
                return None
            try:
                family = load_cached_family(directory, mmap=True)
            except (GSNetCacheError, OSError) as retry_exc:
                logger.warning(
                    "GSNet cache {} is still unusable after rebuild ({}); "
                    "falling back to pickle.load.",
                    directory,
                    retry_exc,
                )
                return None

        return family.as_payload()

    def _rebuild_family(self, path: Path, directory: Path) -> bool:
        """Rebuild the single cache dir for ``path``; return whether it worked.

        Raises:
            Nothing: a failure is reported as ``False`` after a warning so a
                single bad family can never take down a training run.
        """
        # prepare_gsnet_cache keys off the *distinct source paths* in the index
        # and re-reads the pickle's own record keys, so the key below only has
        # to exist. Reusing the dataset's real key keeps the index honest.
        key = next((k for p, k in self._entries if p == path), 0)
        try:
            prepare_gsnet_cache(
                RecordIndex(entries=((path, key),)),
                self._cache_root,
                log_every=0,
            )
        except (OSError, GSNetCacheError, GSNetRecordError) as exc:
            logger.warning(
                "Could not rebuild the GSNet cache for {} ({}); "
                "falling back to pickle.load.",
                path,
                exc,
            )
            return False
        return directory.is_dir()

    def _load_payload(self, path: Path) -> dict[Any, Any]:
        """Return one pickle's records, honouring the bounded LRU.

        When ``use_cache`` is set and a usable lean cache exists, the payload
        comes from that cache (memory-mapped, no deserialisation). Otherwise it
        is produced by ``pickle.load``. Either way the caller receives the same
        ``{record_key: record_dict}`` mapping, so samples are identical.

        The file handle is opened, read and closed **inside** this method: an
        open handle must never be stored on ``self`` because it is not
        picklable and ``DataLoader`` workers pickle the Dataset.

        Args:
            path: Pickle to read.

        Returns:
            The record mapping, possibly backed by memory maps.

        Raises:
            GSNetRecordError: If the file does not exist or its root is not a
                ``dict`` of records.
        """
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached

        payload: dict[Any, Any] | None = None
        if self._use_cache:
            payload = self._load_cached_payload(path)

        if payload is None:
            try:
                with open(path, "rb") as handle:
                    payload = pickle.load(handle)  # noqa: S301 - repo-internal debug dumps
            except FileNotFoundError as exc:
                raise GSNetRecordError(
                    f"Debug pickle does not exist: {path}"
                ) from exc
            except OSError as exc:
                raise GSNetRecordError(
                    f"Could not read debug pickle {path}: {exc}"
                ) from exc
            except pickle.UnpicklingError as exc:
                raise GSNetRecordError(
                    f"Corrupt debug pickle {path}: {exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise GSNetRecordError(
                    f"{path} does not contain a dict of records "
                    f"(got {type(payload).__name__})"
                )

        if self._cache_size > 0:
            self._cache[path] = payload
            self._cache.move_to_end(path)
            while len(self._cache) > self._cache_size:
                evicted, _old = self._cache.popitem(last=False)
                close_cached_payload(_old)
                logger.debug("Evicted {} from the GSNet dataset LRU cache", evicted)
        return payload

    def clear_cache(self) -> None:
        """Drop every cached payload and release any memory maps.

        This clears the in-RAM state only. Nothing on disk is deleted: an
        on-disk cache is expensive to rebuild and is the operator's to remove
        (see :func:`~ao_shaping.runners.gsnet_cache.prepare_gsnet_cache`).
        """
        if self._cache:
            logger.debug("Clearing {} cached GSNet debug payloads", len(self._cache))
            for payload in self._cache.values():
                close_cached_payload(payload)
        self._cache.clear()

    # -- pickling (DataLoader spawn workers re-pickle the Dataset) ---------
    def __getstate__(self) -> dict[str, Any]:
        """Serialise the Dataset **without** the LRU cache.

        Required, not optional: on Windows ``spawn``, every worker re-pickles
        the parent Dataset. Shipping cached payloads would copy a
        ~472 MB dict through the pipe for every worker.

        ``_cache_absent`` goes too: it only suppresses a repeated warning
        *within one process*, so a fresh worker is expected to re-discover the
        absence. Keeping it would make the warm payload measurably larger than
        the cold one, which ``test_pickle_payload_excludes_cached_arrays``
        pins. (With a memory-mapped cache the LRU holds mappings, not arrays,
        but the pickle-safety contract is unchanged either way.)
        """
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        state["_cache_absent"] = set()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`__getstate__` with an empty cache."""
        self.__dict__.update(state)
        # Defensive: a state produced by an older/foreign serialiser may not
        # carry a usable cache at all, and the LRU must never be shared.
        self._cache = OrderedDict()
        if not isinstance(getattr(self, "_cache_absent", None), set):
            self._cache_absent = set()
        # A state predating use_cache would leave the flag undefined, which
        # would make _load_payload read an attribute that does not exist.
        self._use_cache = bool(getattr(self, "_use_cache", True))
        self._cache_root = getattr(self, "_cache_root", None)

    # -- item --------------------------------------------------------------
    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(source, target, gt_phase)``, each ``(1, grid, grid)`` float32.

        Exactly one pickle is consulted (LRU-cached, bounded by ``cache_size``).
        All three tensors are freshly allocated -- ``source`` is a ``clone`` of
        the cached illumination and the other two are built from arrays that
        were just created -- so the caller may mutate them freely.

        Args:
            idx: Positional index into the :class:`RecordIndex`. Negative
                values count from the end, as for any Python sequence.

        Returns:
            ``(source, target, gt_phase)``:
                * ``source`` -- pupil illumination from
                  :func:`ml.gsnet.dataset.make_source_intensity`, ``(1, grid,
                  grid)``;
                * ``target`` -- ``farfield_to_grid(record["_img"], grid)``;
                * ``gt_phase`` -- ``pupil_phase_to_grid`` of the ``_c``
                  reconstruction, with ``n_max = infer_n_max(len(_c))``.

        Raises:
            IndexError: If ``idx`` is out of range.
            GSNetRecordError: If the record is absent from the pickle, misses
                ``_img`` / ``_c``, or carries an unusable ``_c`` (empty, or a
                length that is not a triangular Zernike mode count).
        """
        position = int(idx)
        if position < 0:
            position += len(self._entries)
        if not 0 <= position < len(self._entries):
            raise IndexError(
                f"GSNetDebugDataset index {idx} out of range "
                f"for {len(self._entries)} records"
            )

        path, key = self._entries[position]
        payload = self._load_payload(path)

        try:
            record = payload[key]
        except KeyError as exc:
            raise GSNetRecordError(
                f"Record key {key} is missing from {path} "
                f"(available keys: {sorted(payload)[:8]}...)"
            ) from exc
        if not isinstance(record, dict):
            raise GSNetRecordError(
                f"Record {key} in {path} is not a dict "
                f"(got {type(record).__name__})"
            )
        if "_img" not in record:
            raise GSNetRecordError(
                f"Record {key} in {path} has no '_img' key; the far-field "
                f"target cannot be built (available keys: {sorted(record)})"
            )
        if "_c" not in record:
            raise GSNetRecordError(
                f"Record {key} in {path} has no '_c' key; the ground-truth "
                f"pupil phase cannot be reconstructed "
                f"(available keys: {sorted(record)})"
            )

        target = _to_grid_tensor(farfield_to_grid(record["_img"], self._grid))

        coeffs = np.asarray(record["_c"], dtype=np.float64).ravel()
        if coeffs.size == 0:
            raise GSNetRecordError(
                f"Record {key} in {path} has an empty '_c'; no Zernike mode "
                f"count can be inferred"
            )
        try:
            n_max = infer_n_max(coeffs.size)
            phase_rad = reconstruct_pupil_phase_rad(
                coeffs,
                n_max=n_max,
                slm_width=self._slm_width,
                slm_height=self._slm_height,
                radius=self._slm_radius,
            )
        except ValueError as exc:
            # infer_n_max / reconstruct_pupil_phase_rad reject a mode count that
            # is not a triangular number, or a length that disagrees with n_max.
            # Both are corrupt-record conditions, so they are re-raised as the
            # typed dataset error carrying the path and key.
            raise GSNetRecordError(
                f"Record {key} in {path} has an unusable '_c' "
                f"(len={coeffs.size}): {exc}"
            ) from exc
        gt_phase = _to_grid_tensor(pupil_phase_to_grid(phase_rad, self._grid))

        return self._source.clone(), target, gt_phase


class FileGroupedSampler(Sampler[int]):
    """Shuffle **within** each file group, then shuffle the group order.

    A plain ``RandomSampler`` interleaves indices from all 25 pickles, so
    almost every ``__getitem__`` would miss the LRU cache and re-deserialise a
    (up to ~472 MB) file. This sampler keeps every file's records contiguous
    while still randomising both the intra-file order and the visiting order of
    the files themselves.

    Reproducibility: the internal :class:`torch.Generator` is seeded once from
    ``seed`` and advanced on every :meth:`__iter__`, so consecutive epochs see
    different orders, and re-constructing the sampler with the same ``seed``
    replays the identical sequence (this is the ``DistributedSampler`` scheme).
    """

    def __init__(
        self,
        index: RecordIndex,
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        """Group the index by pickle file.

        Args:
            index: Record index from
                :func:`~ao_shaping.runners.gsnet_offline.build_record_index`.
            num_samples: Total number of indices to yield, enabling upsampling
                (``> len(index)``, whole cycles are repeated) and downsampling
                (``< len(index)``, whole groups are dropped and the last
                admitted group is trimmed) in the spirit of
                :class:`torch.utils.data.WeightedRandomSampler`. ``None``
                (default) means one pass over every record.
            seed: Seed for the internal :class:`torch.Generator`.

        Raises:
            TypeError: If ``index`` is not a :class:`RecordIndex`.
            ValueError: If ``num_samples`` is negative.
        """
        if not isinstance(index, RecordIndex):
            raise TypeError(
                f"index must be a RecordIndex from build_record_index, "
                f"got {type(index).__name__}"
            )
        if num_samples is not None and int(num_samples) < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples}")

        self._entries: tuple[tuple[Path, int], ...] = tuple(index.entries)
        self._num_samples: int | None = (
            None if num_samples is None else int(num_samples)
        )
        self._seed = int(seed)

        # file path -> ordered list of dataset positions, in first-seen order.
        groups: dict[Path, list[int]] = {}
        for position, (path, _key) in enumerate(self._entries):
            groups.setdefault(path, []).append(position)
        self._groups: list[list[int]] = list(groups.values())
        self._generator = torch.Generator()
        self._generator.manual_seed(self._seed)

    @property
    def num_samples(self) -> int | None:
        """Configured yield count, or ``None`` for one pass over the index."""
        return self._num_samples

    @property
    def num_groups(self) -> int:
        """Number of distinct pickle files in the index."""
        return len(self._groups)

    def __len__(self) -> int:
        if self._num_samples is not None:
            return self._num_samples
        return len(self._entries)

    def _shuffled_groups(self) -> list[list[int]]:
        """Shuffle inside every group, then shuffle the group order."""
        generator = self._generator
        inner: list[list[int]] = []
        for group in self._groups:
            # ``randperm`` yields *positions within* the group, so it has to be
            # used to reorder the group -- extending with the permutation itself
            # would emit 0..n-1 instead of the dataset positions.
            permutation = torch.randperm(len(group), generator=generator).tolist()
            inner.append([group[position] for position in permutation])
        # ``randperm`` over the group ids realises the inter-group shuffle.
        group_order = torch.randperm(len(inner), generator=generator).tolist()
        return [inner[group] for group in group_order]

    def _build_indices(self) -> list[int]:
        """Materialise the (contiguous-by-file) index order for one epoch."""
        total = len(self._entries)
        target = total if self._num_samples is None else self._num_samples
        if target == 0 or total == 0:
            return []

        groups = self._shuffled_groups()

        if target >= total:
            # Upsample: repeat whole cycles so every group stays contiguous.
            out: list[int] = []
            while len(out) < target:
                for group in groups:
                    out.extend(group)
                    if len(out) >= target:
                        break
            return out[:target]

        # Downsample: admit whole groups, trimming the last one to the budget.
        out = []
        for group in groups:
            remaining = target - len(out)
            if remaining <= 0:
                break
            out.extend(group[:remaining])
        return out

    def __iter__(self) -> Iterator[int]:
        return iter(self._build_indices())


def build_gsnet_dataloader(
    index: RecordIndex,
    *,
    grid: int = 64,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle: bool = True,
    source_type: str = DEFAULT_SOURCE_TYPE,
    slm_width: int = DEFAULT_SLM_WIDTH,
    slm_height: int = DEFAULT_SLM_HEIGHT,
    slm_radius: float = DEFAULT_SLM_RADIUS,
    cache_size: int = 1,
    num_samples: int | None = None,
    seed: int = 0,
    pin_memory: bool | None = None,
    cache_root: Path | None = None,
    use_cache: bool = True,
) -> DataLoader:
    """Build a ``DataLoader`` over the debug corpus.

    ``pin_memory=None`` auto-enables pinning **only** when CUDA is available
    **and** ``num_workers > 0``: torch warns (and gains nothing) when
    ``pin_memory=True`` is combined with ``num_workers=0``. Passing an explicit
    ``pin_memory=True`` with ``num_workers=0`` is honoured because the caller
    asked for it, but the auto path never does it.

    ``prefetch_factor`` is passed **only** when ``num_workers > 0`` -- torch
    raises ``ValueError`` for a non-``None`` ``prefetch_factor`` without worker
    processes.

    MEMORY: ``num_workers=0`` keeps peak RAM at ``cache_size`` file payloads.
    See the module docstring before raising it.

    Args:
        index: Record index from
            :func:`~ao_shaping.runners.gsnet_offline.build_record_index`.
        grid: Output grid side length; batches are ``(B, 1, grid, grid)``.
        batch_size: Samples per batch.
        num_workers: Worker processes. ``0`` (default, recommended) loads in the
            main process; each worker holds its own copy of the file it has
            open.
        shuffle: ``True`` (default) uses :class:`FileGroupedSampler`;
            ``False`` uses :class:`torch.utils.data.SequentialSampler`.
        source_type: ``"gaussian"`` or ``"uniform"`` pupil illumination.
        slm_width: SLM panel width for the pupil-phase reconstruction.
        slm_height: SLM panel height for the pupil-phase reconstruction.
        slm_radius: Zernike aperture radius in pixels (default 300, the corpus
            device's ``ZERNIKE_APERTURE_RADIUS``).
        cache_size: Per-Dataset LRU capacity in deserialised pickles.
        num_samples: Optional epoch length (upsample/downsample). Only honoured
            with ``shuffle=True``, since :class:`SequentialSampler` is fixed at
            ``len(dataset)``.
        seed: Seed for :class:`FileGroupedSampler`.
        pin_memory: ``None`` auto-detects, ``True``/``False`` force.
        cache_root: Root directory for the lean on-disk cache. ``None`` (default)
            uses the sibling ``.gsnet_cache`` directory next to each pickle.
            Only consulted when ``use_cache=True``.
        use_cache: Read records from the lean on-disk cache when one exists
            (default ``True``). ``False`` forces ``pickle.load``, which yields
            bit-identical samples but re-reads the whole file on every miss.

    Returns:
        A ``DataLoader`` yielding ``(source, target, gt_phase)`` batches.

    Raises:
        ValueError: If the index is empty or ``batch_size`` is not positive.
    """
    if not isinstance(index, RecordIndex):
        raise TypeError(
            f"index must be a RecordIndex from build_record_index, "
            f"got {type(index).__name__}"
        )
    if len(index.entries) == 0:
        raise ValueError(
            "Cannot build a GSNet dataloader from an empty record index; "
            "check the roots passed to build_record_index"
        )
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    dataset = GSNetDebugDataset(
        index,
        grid=grid,
        source_type=source_type,
        slm_width=slm_width,
        slm_height=slm_height,
        slm_radius=slm_radius,
        cache_size=cache_size,
        cache_root=cache_root,
        use_cache=use_cache,
    )

    workers = max(0, int(num_workers))
    if shuffle:
        sampler: Sampler[int] = FileGroupedSampler(
            index, num_samples=num_samples, seed=seed
        )
    else:
        if num_samples is not None:
            logger.warning(
                "num_samples={} is ignored because shuffle=False forces a "
                "SequentialSampler; pass shuffle=True to resample the epoch",
                num_samples,
            )
        sampler = SequentialSampler(dataset)

    if pin_memory is None:
        use_pin = bool(torch.cuda.is_available()) and workers > 0
    else:
        use_pin = bool(pin_memory)

    loader_kwargs: dict[str, Any] = {}
    if workers > 0:
        loader_kwargs["prefetch_factor"] = DEFAULT_PREFETCH_FACTOR

    logger.info(
        "GSNet dataloader: {} records, batch_size={}, num_workers={}, "
        "shuffle={}, pin_memory={}, grid={}",
        len(dataset),
        batch_size,
        workers,
        shuffle,
        use_pin,
        dataset.grid,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=workers,
        pin_memory=use_pin,
        **loader_kwargs,
    )
