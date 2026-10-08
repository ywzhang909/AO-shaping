"""PyTorch Dataset / DataLoader over the hardware debug corpus.

This module is the **consumer** half of :mod:`ml.hwdataset`: it turns the
immutable index produced by :mod:`ml.hwdataset.index` into a
``(commanded SLM phase, camera exposure) -> camera image`` regression set, which
is exactly what the user asked for:

    "请基于 data/debug 下的所有文件在 src/ml 建立一个pytorch的dataloader，
     可用读取发送的slm 的相位、相机的曝光参数作为input、
     相机的画面作为output"

============================  What one item looks like  ============================

:meth:`HwPhaseImageDataset.__getitem__` returns a **dict** (not a tuple), so
``torch.utils.data.default_collate`` needs no ``collate_fn``:

=========================  ======================================================
key                        meaning
=========================  ======================================================
``phase_cos``              ``(1, grid, grid) float32`` tensor, ``Re`` of the
                           coherent block mean of the commanded phase
``phase_sin``              ``(1, grid, grid) float32`` tensor, ``Im`` of the same
``image``                  ``(1, grid, grid) float32`` tensor, the CCD far-field
                           window on the 0-order spot, **absolute** intensity
``exposure_ms``            raw exposure in ms as a Python ``float`` (logging)
``exposure_log10``         ``(1,) float32`` **standardised** log exposure
``contrast``               ``(grid, grid) float32`` numpy, per-cell coherence
``source``                 ``PhaseSource`` **value** string, e.g. ``"panel_gray"``
``family``                 family label string, e.g. ``"slm_pib"``
``sample_idx``             resolved (non-negative) dataset position
``path``                   the source ``.pkl`` as a string
``fov_px``                 source-frame side length in camera pixels, or ``None``
=========================  ======================================================

Three of those decisions are load-bearing and were reviewed before the code was
written:

1. **Why ``(cos, sin)`` and not a phase angle** -- see
   :mod:`ml.hwdataset.transforms`. The corpus panels are wrapped to
   ``[0, 2*pi)``, so an ``arctan2`` reconstruction would carry a branch cut the
   network has to learn around.

2. **Why ``log10`` exposure and not raw ms** -- the measured corpus spans
   **0.1 ms to 80 ms**, i.e. 2.9 orders of magnitude. Fed raw, that single input
   channel saturates the first layer. The *log* compresses it; the
   standardisation ``(log10(ms) - mean) / std`` then removes the remaining
   per-corpus offset so the value is O(1). Both statistics are computed **once**
   from the index (see :attr:`HwPhaseImageDataset.exposure_log_mean`), never from
   the record being materialised -- a per-record value would be a different
   quantity for every item and would not be comparable across epochs.

3. **Why the image is NOT peak-normalised** (``image_mode="abs255"`` default).
    Exposure is a model *input*, so absolute brightness is precisely the signal
    that encodes it. Peak-normalising would delete the exposure information from
    the target while leaving it in the input -- a silent label corruption. The
    ``"peak"`` mode still exists for anyone who deliberately wants the relative
    shape. That is exactly the right trade for **cross-family** comparison:
    measured on the same seeded 252-record stratified sample, the cross-family
    brightness spread (max family-median / min family-median) is ~14.5x in
    ``abs255`` and 1.00x after ``"peak"`` -- the exposure component of the
    difference is removable, while the structural parts (frame CV, d90/FOV,
    central energy, entropy, phase spread) are already scale-invariant and no
    image processing removes them. For the full difference-source-to-processing
    mapping see ``report/hwdataset_corpus``.

=============================  The FOV caveat, surfaced  =============================

The families have **genuinely different fields of view**: a 64 px
``region=32`` window, a 250/320 px camera window, and the full 2592x1944
sensor. One output ``grid`` therefore means a *different physical angular scale*
per family. Resampling to a common physical scale was explicitly reviewed and
rejected (it fabricates pixels the measurement never had and destroys the
absolute-brightness contract above). Instead every item carries ``fov_px``, the
source frame's side length in camera pixels, so a user can **filter by family or
by ``fov_px``** before training. This is surfaced, not fixed.

=============================  Why the grouped sampler  ==============================

Each debug pickle is read **whole** (``pickle.load`` cannot stream), and the
measured corpus is 65.88 GB over 265 files with a **3.42 GB** maximum. With a
plain ``RandomSampler`` almost every ``__getitem__`` would miss the 1-entry LRU
in :class:`~ml.hwdataset.records.PayloadStore` and re-deserialise gigabytes:
measured on this corpus that is the difference between ~2 min and ~10 h per
epoch. :class:`FileGroupedSampler` therefore shuffles *within* each file group
and then shuffles the group order, which keeps the records of a file contiguous
while still decorrelating the order inside it.

=============================  Windows ``spawn`` contract  ==============================

With ``num_workers > 0`` on Windows, torch ``spawn``-s the workers and
**re-pickles the Dataset**. Two things therefore must never be pickled: the
materialiser's LRU (up to 3.42 GB of arrays) and any open file handle. See
:meth:`HwPhaseImageDataset.__getstate__`, which additionally converts the
``MappingProxyType`` sidecars -- which are **not picklable** -- to plain dicts
and rebuilds them on the other side.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import (
    DataLoader,
    Dataset,
    Sampler,
    SequentialSampler,
    Subset,
)

from ml.hwdataset.index import (
    DEFAULT_ROOTS,
    HwCorpusIndex,
    HwRecordRef,
    build_hw_index,
)
from ml.hwdataset.records import Materialiser, MaterialiserConfig

__all__ = [
    "DEFAULT_PREFETCH_FACTOR",
    "FileGroupedSampler",
    "HwPhaseImageDataset",
    "build_hw_dataloader",
    "create_hw_dataloaders",
]

#: ``DataLoader`` prefetch depth when workers are used. Torch's own default; named
#: here so the value is a single reviewable constant.
DEFAULT_PREFETCH_FACTOR: int = 2

#: Sidecar keys consulted for a camera field of view when the raw ``_img`` shape
#: is unavailable. ``region`` is the half-width of the camera window the
#: ``model_in_loop_hw_*`` families open, so its full side is ``2 * region``;
#: ``cam_size`` is the already-full window side used by the ``slm_*`` families.
_FOV_SIDECAR_REGION_KEY = "region"
_FOV_SIDECAR_CAM_SIZE_KEY = "cam_size"


def _sidecar_fov_px(sidecar: Mapping[str, Any]) -> int | None:
    """Derive a camera field-of-view side length from a sidecar, cheaply.

    This is the *fallback* used when the raw ``_img`` shape cannot be read. It
    costs two dict lookups and never touches the payload, which is why it is
    acceptable to call it per record.

    Args:
        sidecar: Merged sidecar mapping for the record's file (may be empty).

    Returns:
        Side length in camera pixels, or ``None`` when the sidecar carries no
        usable hint. ``None`` is the honest answer: an unknown FOV must not be
        fabricated, because it is the one field a user would filter on.
    """
    region = sidecar.get(_FOV_SIDECAR_REGION_KEY)
    if (
        isinstance(region, (int, float))
        and not isinstance(region, bool)
        and math.isfinite(float(region))
        and float(region) > 0.0
    ):
        return int(round(2.0 * float(region)))
    cam_size = sidecar.get(_FOV_SIDECAR_CAM_SIZE_KEY)
    if (
        isinstance(cam_size, (int, float))
        and not isinstance(cam_size, bool)
        and math.isfinite(float(cam_size))
        and float(cam_size) > 0.0
    ):
        return int(cam_size)
    return None


def _fresh_grid_tensor(array: np.ndarray) -> torch.Tensor:
    """Copy a 2-D array into a freshly allocated channel-first ``float32`` tensor.

    Freshness is mandatory, not hygiene: the arrays come out of the materialiser's
    LRU, and the documented contract is that the caller may mutate what it gets
    without corrupting the cache for the next item.

    Args:
        array: Any 2-D numeric array.

    Returns:
        A ``(1, h, w) float32`` tensor backed by its own C-contiguous buffer.
    """
    owned = np.array(array, dtype=np.float32, order="C", copy=True)
    return torch.from_numpy(owned).unsqueeze(0)


class FileGroupedSampler(Sampler[int]):
    """Shuffle WITHIN each pickle-file group, then shuffle the group order.

    This is not a nicety. Each debug pickle is read whole, so with a plain
    ``RandomSampler`` almost every ``__getitem__`` would miss the 1-entry LRU and
    re-deserialise up to 3.42 GB; measured on this corpus that is the difference
    between ~2 min and ~10 h per epoch. Keeping each file's records contiguous
    makes the LRU pay off. Reproducibility: the internal ``torch.Generator`` is
    seeded once from ``seed`` and advanced on every ``__iter__``, so consecutive
    epochs differ while re-constructing with the same ``seed`` replays the
    identical sequence.

    The emitted values are **positions into the ``records`` sequence that was
    handed to the constructor**, which makes the sampler usable directly on a
    ``torch.utils.data.Subset`` as long as the subset preserves that order (it
    does -- :func:`create_hw_dataloaders` does).
    """

    def __init__(
        self,
        records: Sequence[HwRecordRef],
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        """Group records by file and prepare the streaming RNG.

        Args:
            records: The records to sample over. Only ``record.path`` is read, so
                passing the full index costs nothing.
            num_samples: Epoch length. ``None`` (default) means one pass over
                ``records``. A larger value repeats whole cycles; a smaller one
                admits whole groups and trims the last admitted group, so a group
                is never split across the epoch boundary.
            seed: Seed for the internal ``torch.Generator``.

        Raises:
            ValueError: If ``num_samples`` is negative.
        """
        self._records = tuple(records)
        self._requested_num_samples = (
            None if num_samples is None else int(num_samples)
        )
        if self._requested_num_samples is not None and self._requested_num_samples < 0:
            raise ValueError(
                f"num_samples must be >= 0, got {self._requested_num_samples}"
            )
        self._count = (
            len(self._records)
            if self._requested_num_samples is None
            else self._requested_num_samples
        )

        # Preserve first-appearance order of the paths so that a plain (unshuffled)
        # iteration of one group is still in index order.
        grouped: dict[Path, list[int]] = {}
        for position, record in enumerate(self._records):
            grouped.setdefault(record.path, []).append(position)
        self._groups: tuple[tuple[int, ...], ...] = tuple(
            tuple(members) for members in grouped.values()
        )
        self._seed = int(seed)
        self._generator = torch.Generator()
        self._generator.manual_seed(self._seed)

    @property
    def num_samples(self) -> int | None:
        """Return the requested epoch length.

        Returns:
            The value passed to ``num_samples``, or ``None`` when the caller left
            it unset (meaning "one pass over the records"). The *effective* length
            is :meth:`__len__`.
        """
        return self._requested_num_samples

    @property
    def num_groups(self) -> int:
        """Return the number of distinct pickle files among the records.

        Returns:
            Count of distinct :attr:`HwRecordRef.path` values.
        """
        return len(self._groups)

    def __len__(self) -> int:
        """Return the effective epoch length.

        Returns:
            ``num_samples`` when given, else the number of records.
        """
        return self._count

    def _one_cycle(self) -> list[int]:
        """Build one full pass: shuffle inside every group, then the group order.

        Returns:
            All record positions exactly once, with each file's members
            contiguous.
        """
        shuffled_groups: list[list[int]] = []
        for group in self._groups:
            permutation = torch.randperm(
                len(group), generator=self._generator
            ).tolist()
            shuffled_groups.append([group[i] for i in permutation])
        group_order = torch.randperm(
            len(shuffled_groups), generator=self._generator
        ).tolist()
        cycle: list[int] = []
        for group_index in group_order:
            cycle.extend(shuffled_groups[group_index])
        return cycle

    def __iter__(self) -> Iterator[int]:
        """Yield positions for one epoch.

        Yields:
            Record positions, grouped by source file. Whole cycles are repeated
            when ``num_samples`` exceeds the record count; the final cycle is
            trimmed at a group boundary otherwise.

        Yields nothing when there are no records or ``num_samples == 0``, so an
        empty sampler never spins.
        """
        if self._count <= 0 or not self._groups:
            return iter(())
        return self._iterate()

    def _iterate(self) -> Iterator[int]:
        """Generator backing :meth:`__iter__` (kept separate so ``__iter__`` can
        short-circuit without a generator object being created at all).

        Yields:
            Record positions as described in :meth:`__iter__`.
        """
        emitted = 0
        while emitted < self._count:
            cycle = self._one_cycle()
            take = min(len(cycle), self._count - emitted)
            for position in cycle[:take]:
                yield position
            emitted += take


class HwPhaseImageDataset(Dataset[dict[str, Any]]):
    """Lazy Dataset over the debug corpus indexed by ``build_hw_index``.

    ``__init__`` opens NO pickle: it retains only the immutable index and the
    (record-independent) config, plus the derived ``exposure_log10`` statistics.
    ``__getitem__`` materialises exactly one record through
    :class:`~ml.hwdataset.records.Materialiser`.
    """

    def __init__(
        self,
        index: HwCorpusIndex,
        *,
        config: MaterialiserConfig | None = None,
        cache_size: int = 1,
        require_exposure: bool = True,
        use_cache: bool = True,
        cache_root: Path | None = None,
    ) -> None:
        """Prepare a Dataset over ``index`` without touching the disk.

        Args:
            index: Corpus index from :func:`~ml.hwdataset.index.build_hw_index`.
            config: Phase reconstruction / output geometry. ``None`` uses
                :class:`~ml.hwdataset.records.MaterialiserConfig` defaults
                (``grid=64``, ``image_mode="abs255"``).
            cache_size: Payload LRU size handed to the
                :class:`~ml.hwdataset.records.Materialiser`. ``1`` is the useful
                default given :class:`FileGroupedSampler`; larger values only pay
                off for a deliberately ordered loader.
            require_exposure: Drop records with an unknown exposure. A learning
                target needs a known exposure to be interpretable, so this
                defaults to ``True``.
            use_cache: Read from the on-disk ``.hw_cache`` directories when a
                current one exists (see
                :func:`~ml.hwdataset.cache.prepare_hw_cache`). Bit-identical to
                the direct path; defaults to ``True``, and falls back per record
                whenever a cache is absent or stale.
            cache_root: Root for the cache directories, or ``None`` for the
                sibling ``.hw_cache`` directory next to each pickle.

        Raises:
            ValueError: If the (filtered) index holds no records, or if
                ``cache_size`` is negative.
        """
        if int(cache_size) < 0:
            raise ValueError(f"cache_size must be >= 0, got {cache_size}")

        self._config = MaterialiserConfig() if config is None else config
        self._cache_size = int(cache_size)
        self._require_exposure = bool(require_exposure)
        self._use_cache = bool(use_cache)
        self._cache_root = Path(cache_root) if cache_root is not None else None
        filtered = (
            index.filter(require_exposure=True)
            if self._require_exposure
            else index.filter(require_exposure=False)
        )
        if not filtered.records:
            raise ValueError(
                "HwPhaseImageDataset: the corpus has no usable records after "
                f"filtering (families={list(filtered.families)}, "
                f"files_scanned={filtered.files_scanned}, "
                f"files_usable={filtered.files_usable}, "
                f"excluded={ {reason.value: count for reason, count in filtered.excluded.items()} }, "
                f"require_exposure={self._require_exposure}). "
                "Nothing to train on: check the roots, or pass "
                "require_exposure=False to keep records with an unknown exposure."
            )
        self._index = filtered
        self._records = filtered.records
        self._exposure_log_mean, self._exposure_log_std = self._exposure_stats()
        self._materialiser = Materialiser(
            self._config,
            cache_size=self._cache_size,
            use_cache=self._use_cache,
            cache_root=self._cache_root,
        )

        logger.info(
            "HwPhaseImageDataset ready: {} records from {} pickles ({} files scanned), "
            "grid={}, image_mode='{}', cache_size={}, require_exposure={}, "
            "use_cache={}, exposure_log mean={:.4f} std={:.4f}",
            len(self._records),
            len({record.path for record in self._records}),
            self._index.files_scanned,
            self._config.grid,
            self._config.image_mode,
            self._cache_size,
            self._require_exposure,
            self._use_cache,
            self._exposure_log_mean,
            self._exposure_log_std,
        )

    def _exposure_stats(self) -> tuple[float, float]:
        """Compute the standardised-exposure statistics from the retained index.

        Returns:
            ``(mean, std)`` of ``log10(exposure_ms)`` over the records with a
            known exposure. Fewer than two such records yields ``(0.0, 1.0)`` with
            a warning rather than a division by zero -- the standardisation then
            degenerates to the identity, which is the only safe behaviour.
        """
        known = [
            math.log10(float(record.exposure_ms))
            for record in self._records
            if record.exposure_ms is not None and float(record.exposure_ms) > 0.0
        ]
        if len(known) < 2:
            logger.warning(
                "Only {} of {} records carry a usable positive exposure; exposure "
                "log standardisation falls back to mean=0.0 std=1.0",
                len(known),
                len(self._records),
            )
            return 0.0, 1.0
        array = np.asarray(known, dtype=np.float64)
        mean = float(array.mean())
        std = float(array.std())
        if not math.isfinite(std) or std <= 0.0:
            logger.warning(
                "Every known exposure in this corpus is identical (log10={}); "
                "exposure log standardisation falls back to mean=0.0 std=1.0",
                mean,
            )
            return 0.0, 1.0
        return mean, std

    @property
    def index(self) -> HwCorpusIndex:
        """Return the (filtered) index this Dataset iterates over.

        Returns:
            The immutable :class:`~ml.hwdataset.index.HwCorpusIndex` actually in
            use, which is ``index.filter(require_exposure=...)`` when that
            changed anything.
        """
        return self._index

    @property
    def config(self) -> MaterialiserConfig:
        """Return the record-independent materialisation config.

        Returns:
            The :class:`~ml.hwdataset.records.MaterialiserConfig` in force.
        """
        return self._config

    @property
    def records(self) -> tuple[HwRecordRef, ...]:
        """Return the retained record references.

        Returns:
            The tuple of :class:`~ml.hwdataset.index.HwRecordRef` this Dataset
            maps positions onto. Holds no arrays.
        """
        return self._records

    @property
    def exposure_log_mean(self) -> float:
        """Return the corpus mean of ``log10(exposure_ms)``.

        Returns:
            The mean used to standardise :meth:`__getitem__`'s
            ``exposure_log10``. ``0.0`` when fewer than two exposures are known.
        """
        return self._exposure_log_mean

    @property
    def exposure_log_std(self) -> float:
        """Return the corpus standard deviation of ``log10(exposure_ms)``.

        Returns:
            The standard deviation used to standardise ``exposure_log10``.
            ``1.0`` when fewer than two exposures are known.
        """
        return self._exposure_log_std

    def __len__(self) -> int:
        """Return the number of usable records.

        Returns:
            ``len(self.records)``.
        """
        return len(self._records)

    def _fov_px(self, ref: HwRecordRef, sample: Any | None = None) -> int | None:
        """Return the source frame's side length in camera pixels.

        Implementation choice: the raw ``_img`` shape is read from the *sample*,
        which the materialiser already produced from the record it had in hand.
        An earlier version asked the materialiser's payload LRU instead, on the
        reasoning that ``materialise(ref)`` performs exactly one ``store.get(ref)``
        so the payload would still be resident. That reasoning breaks once the
        on-disk cache is in use: a cache hit never touches the payload, so the
        lookup would fall through to a **full re-open of the pickle** -- up to
        3.42 GB -- for every record, which is precisely the cost the cache
        exists to remove. Carrying the geometry on the sample keeps the cached and
        direct paths equally cheap.

        Args:
            ref: The record whose frame geometry is wanted.
            sample: The :class:`~ml.hwdataset.records.HwSample` just materialised
                for ``ref``, if the caller has it.

        Returns:
            The smaller of the frame's two sides in pixels, or ``None``.
        """
        if sample is not None:
            from_sample = getattr(sample, "fov_px", None)
            if from_sample is not None:
                return int(from_sample)

        payload: Any = None
        try:
            payload = self._materialiser.store.get(ref)
        except (AttributeError, KeyError) as exc:
            logger.debug(
                "PayloadStore.get unavailable or empty for {} (pos {}): {}",
                ref.path,
                ref.position,
                exc,
            )
        if isinstance(payload, Mapping):
            shape = getattr(payload.get("_img"), "shape", None)
            if isinstance(shape, tuple) and len(shape) == 2:
                return int(min(int(shape[0]), int(shape[1])))
        return _sidecar_fov_px(ref.sidecar)

    def _standardised_exposure_log(self, exposure_ms: float | None) -> float:
        """Return the standardised log exposure of one record.

        Args:
            exposure_ms: Raw exposure in ms, or ``None``.

        Returns:
            ``(log10(ms) - mean) / std`` for a known positive exposure, else
            ``0.0`` -- the neutral value of a standardised quantity.
        """
        if exposure_ms is None or not math.isfinite(float(exposure_ms)):
            return 0.0
        value = float(exposure_ms)
        if value <= 0.0:
            return 0.0
        return (math.log10(value) - self._exposure_log_mean) / self._exposure_log_std

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Materialise exactly one record.

        Args:
            idx: Dataset position. Negative indices follow Python sequence
                semantics.

        Returns:
            The sample dict documented in the module docstring. Every tensor is
            freshly allocated, C-contiguous and ``float32``, so the caller may
            mutate it.

        Raises:
            TypeError: If ``idx`` does not implement ``__index__``.
            IndexError: If ``idx`` is out of range for this Dataset.
            ml.hwdataset.records.HwRecordError: Propagated from the
                :class:`~ml.hwdataset.records.Materialiser`. A record that was
                indexed but cannot be materialised is a real failure, never a
                silent skip.
        """
        try:
            raw = idx.__index__()  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise TypeError(
                f"Dataset indices must be integers, got {type(idx).__name__}"
            ) from exc
        position = int(raw)
        total = len(self._records)
        if position < 0:
            position += total
        if not 0 <= position < total:
            raise IndexError(
                f"Dataset index {idx} is out of range for a dataset of {total} records"
            )

        ref = self._records[position]
        sample = self._materialiser.materialise(ref)
        exposure_ms = (
            None if sample.exposure_ms is None else float(sample.exposure_ms)
        )
        exposure_log = torch.tensor(
            [self._standardised_exposure_log(exposure_ms)], dtype=torch.float32
        )
        return {
            "phase_cos": _fresh_grid_tensor(sample.phase_cos),
            "phase_sin": _fresh_grid_tensor(sample.phase_sin),
            "image": _fresh_grid_tensor(sample.image),
            "exposure_ms": exposure_ms,
            "exposure_log10": exposure_log,
            "contrast": np.asarray(sample.contrast, dtype=np.float32),
            "source": ref.source.value,
            "family": ref.family,
            "sample_idx": position,
            # str, not Path: `default_collate` turns a list of strings into a
            # list of strings but has no dedicated branch for Path, and a plain
            # string is what a logging sink wants anyway.
            "path": str(ref.path),
            "fov_px": self._fov_px(ref, sample),
        }

    def clear_cache(self) -> None:
        """Drop every cached payload and derived array."""
        self._materialiser.clear()

    def __getstate__(self) -> dict[str, Any]:
        """Return a picklable state that carries no arrays and no file handle.

        Three things are deliberately excluded or rewritten:

        * ``_materialiser`` -- its LRU holds up to 3.42 GB of numpy arrays and may
          hold an open file handle. It is rebuilt in :meth:`__setstate__`.
        * every ``sidecar`` ``MappingProxyType`` -- **not picklable at all**
          (``TypeError: cannot pickle 'mappingproxy' object``), which would make
          the whole Dataset unpicklable and therefore break ``num_workers > 0``
          on Windows. They become plain dicts here.
        * ``_index`` -- rebuilt from the already-plain ``_records`` so the two can
          never drift apart.

        Returns:
            A JSON-free but pickle-safe ``dict`` of the Dataset's own state.
        """
        return {
            "_records": tuple(
                replace(record, sidecar=dict(record.sidecar)) for record in self._records
            ),
            "_excluded": dict(self._index.excluded),
            "_files_scanned": self._index.files_scanned,
            "_files_usable": self._index.files_usable,
            "_config": self._config,
            "_cache_size": self._cache_size,
            "_require_exposure": self._require_exposure,
            "_use_cache": self._use_cache,
            "_cache_root": self._cache_root,
            "_exposure_log_mean": self._exposure_log_mean,
            "_exposure_log_std": self._exposure_log_std,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`__getstate__`, rebuilding the materialiser.

        Args:
            state: The dict produced by :meth:`__getstate__`.
        """
        records = tuple(
            replace(record, sidecar=MappingProxyType(dict(record.sidecar)))
            for record in state["_records"]
        )
        self.__dict__.update(state)
        self._records = records
        self._index = HwCorpusIndex(
            records=records,
            excluded=MappingProxyType(dict(state["_excluded"])),
            files_scanned=int(state["_files_scanned"]),
            files_usable=int(state["_files_usable"]),
        )
        self._materialiser = Materialiser(
            self._config,
            cache_size=int(state["_cache_size"]),
            # .get with a default so a state pickled before these flags existed
            # still restores; a worker that lost the flag would silently fall back
            # to the slow path for the whole epoch.
            use_cache=bool(state.get("_use_cache", True)),
            cache_root=(
                Path(state["_cache_root"]) if state.get("_cache_root") is not None else None
            ),
        )


def _resolve_pin_memory(pin_memory: bool | None, num_workers: int) -> bool:
    """Decide ``pin_memory`` following the ``None`` = auto convention.

    Args:
        pin_memory: The caller's explicit choice, or ``None``.
        num_workers: Number of DataLoader workers.

    Returns:
        The resolved flag. Auto mode only enables pinning when CUDA is available
        **and** workers exist: torch warns about ``num_workers=0`` and pinning
        buys nothing there because the transfer happens on the main thread anyway.
    """
    if pin_memory is not None:
        return bool(pin_memory)
    try:
        cuda_available = bool(torch.cuda.is_available())
    except (RuntimeError, AssertionError):  # pragma: no cover - driver hiccup
        cuda_available = False
    return cuda_available and int(num_workers) > 0


def build_hw_dataloader(
    index: HwCorpusIndex,
    *,
    config: MaterialiserConfig | None = None,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle: bool = True,
    cache_size: int = 1,
    num_samples: int | None = None,
    seed: int = 0,
    pin_memory: bool | None = None,
    drop_last: bool = False,
    use_cache: bool = True,
    cache_root: str | Path | None = None,
) -> DataLoader:
    """Build a ``DataLoader`` over one corpus index.

    Args:
        index: Corpus index to iterate.
        config: Materialisation config, forwarded to
            :class:`HwPhaseImageDataset`.
        batch_size: Samples per batch.
        num_workers: Worker processes. ``> 0`` re-pickles the Dataset on Windows
            (``spawn``), which is why the pickling contract is part of the class
            contract.
        shuffle: ``True`` uses :class:`FileGroupedSampler` (the LRU-friendly
            order), ``False`` uses torch's ``SequentialSampler``.
        cache_size: Payload LRU size, forwarded to the Dataset.
        num_samples: Epoch length override for the grouped sampler.
        seed: Seed of the grouped sampler's internal generator.
        pin_memory: ``None`` auto-enables it only when CUDA is available **and**
            ``num_workers > 0`` (torch warns and gains nothing for
            ``num_workers=0``). Pass a bool to override.
        drop_last: Drop an incomplete final batch.
        use_cache: Read from the on-disk ``.hw_cache`` directories when current.
            Bit-identical to the direct path; defaults to ``True``. Build them
            once with :func:`~ml.hwdataset.cache.prepare_hw_cache` (or
            ``python -m ml.hwdataset.inspect --build-cache``) to get the speedup.
        cache_root: Root for the cache directories, or ``None`` for the sibling
            ``.hw_cache`` directory next to each pickle.

    Returns:
        A ``DataLoader`` whose ``collate_fn`` is torch's ``default_collate`` --
        the sample dict is tensors plus Python scalars/strings, which is exactly
        what it handles.
    """
    dataset = HwPhaseImageDataset(
        index,
        config=config,
        cache_size=cache_size,
        use_cache=use_cache,
        cache_root=Path(cache_root) if cache_root is not None else None,
    )
    workers = max(0, int(num_workers))
    loader_kwargs: dict[str, Any] = {}
    if shuffle:
        loader_kwargs["sampler"] = FileGroupedSampler(
            dataset.records, num_samples=num_samples, seed=seed
        )
        # torch raises "sampler option is mutually exclusive with shuffle".
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = False
    if workers > 0:
        # torch raises ValueError for a non-None prefetch_factor without workers.
        loader_kwargs["prefetch_factor"] = DEFAULT_PREFETCH_FACTOR
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=_resolve_pin_memory(pin_memory, workers),
        drop_last=drop_last,
        **loader_kwargs,
    )


def create_hw_dataloaders(
    roots: Sequence[str | Path] | str | Path = DEFAULT_ROOTS,
    *,
    config: MaterialiserConfig | None = None,
    batch_size: int = 16,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    cache_size: int = 1,
    limit_files: int | None = None,
    index_cache: str | Path | None = None,
    require_exposure: bool = True,
    use_cache: bool = True,
    cache_root: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build seeded train/val/test loaders.

    Follows the ``create_*_loaders`` convention already used by
    :func:`ml.phase.dataset.create_dataloaders` and
    :func:`ml.zernike.dataset.create_zernike_loaders`:
    ``torch.Generator().manual_seed(seed)`` + ``torch.randperm`` + ``Subset``.

    The split is by **FILE**, not by record. Records inside one file are
    near-duplicates -- consecutive SPGD epochs of the same optimisation run --
    so a record-level split would put near-identical samples on both sides of the
    train/val boundary and inflate every validation metric.

    Args:
        roots: Corpus root(s) handed to :func:`~ml.hwdataset.index.build_hw_index`.
        config: Materialisation config.
        batch_size: Samples per batch.
        train_split: Fraction of **files** for training.
        val_split: Fraction of **files** for validation. The remainder is the test
            split.
        num_workers: Worker processes per loader.
        seed: Seed of the split generator.
        cache_size: Payload LRU size.
        limit_files: Cap on the number of scanned pickles (smoke runs).
        index_cache: Optional JSON index cache path.
        require_exposure: Drop records with an unknown exposure.
        use_cache: Read from the on-disk ``.hw_cache`` directories when current.
            Bit-identical to the direct path; defaults to ``True``.
        cache_root: Root for the cache directories, or ``None`` for the sibling
            ``.hw_cache`` directory next to each pickle.

    Returns:
        ``(train_loader, val_loader, test_loader)``. The train loader is shuffled
        with a :class:`FileGroupedSampler`; val and test are sequential so their
        metrics are reproducible.

    Raises:
        ValueError: If a split fraction is out of range, if the fractions leave no
            room for a third split, if the corpus is empty, or if any of the three
            partitions would contain no record.
    """
    train_fraction = float(train_split)
    val_fraction = float(val_split)
    if not math.isfinite(train_fraction) or train_fraction <= 0.0 or train_fraction > 1.0:
        raise ValueError(
            f"train_split must be in (0, 1], got {train_split!r}"
        )
    if not math.isfinite(val_fraction) or val_fraction < 0.0 or val_fraction > 1.0:
        raise ValueError(f"val_split must be in [0, 1], got {val_split!r}")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            f"train_split ({train_split}) + val_split ({val_split}) must leave room "
            "for a non-empty test split"
        )

    index = build_hw_index(
        roots,
        limit_files=limit_files,
        index_cache=index_cache,
        progress_every=0,
    )
    dataset = HwPhaseImageDataset(
        index,
        config=config,
        cache_size=cache_size,
        require_exposure=require_exposure,
        use_cache=use_cache,
        cache_root=Path(cache_root) if cache_root is not None else None,
    )
    records = dataset.records

    # Split by FILE, in first-appearance order so the corpus order is preserved.
    paths: list[Path] = []
    positions_by_path: dict[Path, list[int]] = {}
    for position, record in enumerate(records):
        if record.path not in positions_by_path:
            paths.append(record.path)
            positions_by_path[record.path] = []
        positions_by_path[record.path].append(position)

    generator = torch.Generator().manual_seed(int(seed))
    shuffled_paths = [
        paths[i]
        for i in torch.randperm(len(paths), generator=generator).tolist()
    ]
    n_train_files = int(len(paths) * train_fraction)
    n_val_files = int(len(paths) * val_fraction)
    if n_train_files < 1 or n_val_files < 1 or n_train_files + n_val_files >= len(paths):
        raise ValueError(
            f"File-level split of {len(paths)} files with train_split={train_split} "
            f"and val_split={val_split} yields "
            f"train={n_train_files} val={n_val_files} "
            f"test={len(paths) - n_train_files - n_val_files}; "
            "every split needs at least one file"
        )

    train_paths = shuffled_paths[:n_train_files]
    val_paths = shuffled_paths[n_train_files : n_train_files + n_val_files]
    test_paths = shuffled_paths[n_train_files + n_val_files :]

    def _positions(selected: Sequence[Path]) -> list[int]:
        out: list[int] = []
        for path in selected:
            out.extend(positions_by_path[path])
        return out

    train_indices = _positions(train_paths)
    val_indices = _positions(val_paths)
    test_indices = _positions(test_paths)
    for name, indices in (
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ):
        if not indices:
            raise ValueError(
                f"The {name} split of corpus {list(paths[:3])}... contains no record; "
                f"got {len(records)} records over {len(paths)} files with "
                f"train_split={train_split} and val_split={val_split}"
            )

    logger.info(
        "File-level split: {} files -> train {} / val {} / test {} "
        "(records {} / {} / {})",
        len(paths),
        len(train_paths),
        len(val_paths),
        len(test_paths),
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )

    workers = max(0, int(num_workers))
    pin = _resolve_pin_memory(None, workers)
    worker_kwargs: dict[str, Any] = {}
    if workers > 0:
        worker_kwargs["prefetch_factor"] = DEFAULT_PREFETCH_FACTOR

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        sampler=FileGroupedSampler(
            [records[i] for i in train_indices], seed=int(seed)
        ),
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        **worker_kwargs,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        **worker_kwargs,
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        **worker_kwargs,
    )
    return train_loader, val_loader, test_loader
