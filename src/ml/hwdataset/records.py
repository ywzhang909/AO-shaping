"""One hardware record -> one concrete training sample.

This module is the **materialisation** layer of :mod:`ml.hwdataset`. It is the
single place where an :class:`~ml.hwdataset.index.HwRecordRef` -- a pointer
holding no arrays at all -- is turned into the numeric arrays a model consumes.
Both the :class:`~ml.hwdataset.dataset.HwPhaseImageDataset` (which reads the
pickles directly) and the on-disk cache builder (a later module) go through
:func:`Materialiser.materialise`, so a sample is bit-identical whichever path
produced it. That identity is what makes the cache safe to key on
:class:`MaterialiserConfig` and nothing else.

=========================  Why one module, not four  =========================

The four steps -- read the record, build the panel phase, reduce the panel to the
grid, reduce the far-field frame to the grid -- must agree to the last bit
between the cache builder and the Dataset. Every earlier version of this repo that
had two copies of such a transform drifted (see the duplicated Zernike math called
out in ``AGENTS.md``). There is exactly one implementation here, and it delegates
all array math to :mod:`ml.hwdataset.transforms`: this module contains **no**
Zernike basis, no crop arithmetic and no intensity scaling of its own.

==========================  The two payload shapes  ==========================

The measured corpus (996 files, 265 ``.pkl``, **65.88 GB**) has two payload
shapes, and both are honoured here:

* ``dict[int, dict]`` -- the overwhelming majority. Addressed by
  :attr:`~ml.hwdataset.index.HwRecordRef.key` when the key is an ``int``, and by
  :attr:`~ml.hwdataset.index.HwRecordRef.position` over the *sorted* key order
  otherwise (the index stamps ``key=None`` for a non-``int`` key, and its
  ``position`` was computed over that same sorted order -- so the two must
  agree or the wrong record would be read).
* a pickled :class:`ao_shaping.utils.io.file.Recorder` exposing
  ``.history: list[dict]`` (8 files under ``data/debug/slm_pib_online/``).
  Addressed by :attr:`~ml.hwataset.index.HwRecordRef.position` alone.

3 of the 265 pickles unpickle to ``None`` because their run aborted, and a
truncated dump raises ``EOFError``/``UnpicklingError``; both become
:class:`HwRecordError` rather than a crash, because the message is what tells an
operator *which* of the 265 files to delete.

========================  Memory: reduce before retaining  =====================

``pickle.load`` is monolithic -- it cannot stream -- so peak transient RAM is
**one payload** no matter how this module is written, and the largest pickle in
the corpus is **3.42 GB**. Unpickling itself runs at ~2 GB/s (55 MB in 0.05 s,
280 MB in 0.13 s), so the *number* of loads is what costs time, not their size.

That is why :class:`PayloadStore` exists and why its default capacity is
**1 file**: with a ``FileGroupedSampler`` each file is visited as one contiguous
group per epoch, so one resident payload turns the epoch cost from
``n_records * file_size`` into ``sum(file_size)``. Sized in *files* rather than
bytes because the granularity of reclamation is a whole file -- a byte budget
would still have to hold the largest entry.

Within one record the ordering is equally load-bearing. A ``(1200, 1920)``
``float32`` panel is **9.2 MB** and a ``uint16`` grayscale panel 4.6 MB; the
corpus holds 11393 usable records. The panel is therefore reduced to the
``grid x grid`` phasor pair **before** anything else about the record is
retained, and a single :func:`~ml.hwdataset.transforms.phase_to_grid` call
costs **6.7 ms** on real data. The panel is a local and dies at the end of the
call.

=============================  Never skip a record  ============================

:func:`Materialiser.materialise` **raises** :class:`HwRecordError`; it never
returns a placeholder and never silently drops the record. ``dataset[i]`` is
addressed positionally, so dropping record 5 of a file mid-epoch would shift
every later index and silently train on the wrong image. An exception is loud,
reproducible and names the path and position.

=============================  Units and no-ops  ==============================

``_c`` is in **radians** (Noll order for Zernike, ``grid*grid`` cell amplitudes
for freeform) and ``_phase`` for :attr:`~ml.hwdataset.index.PhaseSource.PANEL_RAD`
is already radians, wrapped to ``[0, 2pi)`` (observed panel max 6.2605).
``record["exp_t"]`` is **milliseconds**. No ``um_to_waves`` / ``*2*pi``
conversion and no ``mod 2*pi`` happens here: a generator that re-wraps its own
output duplicates the driver's single wrap point and hides the true phase
(``AGENTS.md`` red line). The Zernike ``lru_cache``d generator inside
:func:`~ao_shaping.runners.gsnet_offline.reconstruct_pupil_phase_rad` is reused
as-is, which is why no generator cache lives in this module -- only the
``radius`` / ``resolution`` arguments, taken from the config rather than
hard-coded.
"""

from __future__ import annotations

import pickle
import struct
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from ml.hwdataset.index import HwRecordRef, PhaseSource
from ml.hwdataset.transforms import (
    DEFAULT_PANEL_CENTER,
    DEFAULT_PANEL_RADIUS,
    DEFAULT_SLM_MAX_GRAY,
    farfield_frame_to_grid,
    freeform_grid_to_panel,
    grayscale_to_phase_rad,
    phase_to_grid,
    zernike_coeffs_to_panel,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "HwRecordError",
    "HwSample",
    "MaterialiserConfig",
    "PayloadStore",
]
# `Materialiser` is deliberately absent from `__all__`: the contract this module
# was written against lists exactly the four names above, and a test pins it.
# The class is still an ordinary public symbol and is imported by name (as
# `from ml.hwdataset.records import Materialiser`), which `__all__` does not
# restrict.

#: Santec SLM-200 panel resolution as ``(width, height)``, matching
#: :data:`~ml.hwdataset.transforms.DEFAULT_PANEL_CENTER` = ``(960, 600)``.
#: Only needed to reconstruct a panel for the sources whose record carries no
#: ``_phase`` array (Zernike / freeform coefficients).
DEFAULT_PANEL_RESOLUTION: tuple[int, int] = (1920, 1200)

#: Zernike aperture radius in panel pixels. This is the hard-coded
#: ``ZERNIKE_APERTURE_RADIUS`` of ``optimizer/wfless/slm_zernike_pib.py`` and
#: ``slm_zernike_shaping.py``; the sidecars of the Zernike families record
#: ``zernike_radius=480``, but the reconstruction has to pick one value and
#: this is the one the coefficients were fitted against.
_ZERNIKE_APERTURE_RADIUS: float = 300.0

#: Documented bench illumination radius on the panel, in pixels (measured by
#: ``ao_shaping.tools.slm.slm_beam_extent``: r = 450 px at ``(960, 600)``). The
#: freeform reconstruction does not currently consume it --
#: :func:`~ml.hwdataset.transforms.freeform_grid_to_panel` replicates cells
#: unmasked, because the driver and the beam do the masking -- but it is part of
#: the config and therefore of the cache key, so it stays explicit here.
_FREEFORM_BEAM_RADIUS: float = 450.0

_PHASE_KEY = "_phase"
_COEFF_KEY = "_c"
_IMAGE_KEY = "_img"
_HISTORY_ATTR = "history"

#: Scaling modes accepted by
#: :func:`~ml.hwdataset.transforms.farfield_frame_to_grid`. Mirrored here so
#: :meth:`MaterialiserConfig.validate` can name an unknown ``image_mode``
#: without calling the transform (and so the default cannot drift apart).
_IMAGE_MODES: tuple[str, ...] = ("abs255", "peak", "raw")

#: Exceptions that mean "this file is not a usable record dump". Mirrors
#: ``ml.hwdataset.index._UNREADABLE_PICKLE_ERRORS``: a corrupt or truncated
#: pickle raises a surprisingly wide set, and each is listed explicitly because
#: a bare ``except:`` is a repo red line. Duplicated rather than imported
#: because that tuple is private to ``index`` and this module's dependency on
#: ``index`` is deliberately restricted to its public API.
_LOAD_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    EOFError,
    pickle.UnpicklingError,
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    TypeError,
    ValueError,
    struct.error,
    MemoryError,
)


class HwRecordError(RuntimeError):
    """A record could not be turned into a sample.

    Raised, never silently skipped: ``dataset[i]`` is addressed positionally, so
    dropping a record mid-epoch would silently shift the index<->file mapping.
    Every message names the ``path`` and the ``position`` of the offending
    record, because a 265-file corpus gives no other way to find it.
    """


@dataclass(frozen=True)
class MaterialiserConfig:
    """Everything that affects the numeric output of a materialised sample.

    Two records materialised with equal configs must produce bit-identical
    arrays; this is what makes the on-disk cache safe to key on this dataclass
    and nothing else. Anything that only affects *where* a sample is written,
    or *how fast* it is produced, must stay out of it -- in particular the
    payload cache capacity, which is a :class:`Materialiser` constructor
    argument rather than a field here.

    Attributes:
        grid: Side length of every returned array. 64 is the working default: a
            500 px ROI at ``grid=64`` puts ~8 panel pixels in each cell.
        slm_max_gray: SLM grayscale level corresponding to 2*pi (993 on
            SLM#1 @1064nm). Only used for
            :attr:`~ml.hwdataset.index.PhaseSource.PANEL_GRAY` records; the
            device reports it at runtime and the debug dumps do not record it.
        panel_center: Illuminated-beam centre as ``(x, y)`` = ``(col, row)``.
        panel_radius: ROI half-width in panel pixels, 500 in production so the
            widest observed structure (radius ~479 px, ``slm_pib_*``) is
            contained.
        panel_resolution: Panel size as ``(width, height)``, needed only to
            reconstruct a panel for coefficient-only records.
        image_mode: ``abs255`` (default) | ``peak`` | ``raw``; see
            :func:`~ml.hwdataset.transforms.farfield_frame_to_grid`.
        zernike_radius: Zernike aperture radius in panel pixels.
        freeform_radius: Bench illumination radius in panel pixels. Currently
            not consumed by the freeform reconstruction (which replicates cells
            unmasked), but kept here so the cache key stays complete.
    """

    grid: int = 64
    slm_max_gray: int = DEFAULT_SLM_MAX_GRAY
    panel_center: tuple[int, int] = DEFAULT_PANEL_CENTER
    panel_radius: float = DEFAULT_PANEL_RADIUS
    panel_resolution: tuple[int, int] = DEFAULT_PANEL_RESOLUTION
    image_mode: str = "abs255"
    zernike_radius: float = _ZERNIKE_APERTURE_RADIUS
    freeform_radius: float = _FREEFORM_BEAM_RADIUS

    def validate(self) -> None:
        """Raise :class:`ValueError` naming the offending field, if any.

        Called by :meth:`Materialiser.__init__` so an unusable configuration can
        never reach a transform halfway through a 265-file epoch -- where the
        resulting ``ValueError`` from
        :func:`~ml.hwdataset.transforms.phase_to_grid` would name a phase array
        rather than the field the operator actually has to change.

        Raises:
            ValueError: If any field is unusable. The message always contains
                the field name (``grid``, ``slm_max_gray``, ``panel_center``,
                ``panel_radius``, ``panel_resolution``, ``image_mode``,
                ``zernike_radius``, ``freeform_radius``).
        """
        if int(self.grid) < 1:
            raise ValueError(f"grid must be >= 1, got {self.grid}")
        if int(self.slm_max_gray) < 1:
            raise ValueError(f"slm_max_gray must be >= 1, got {self.slm_max_gray}")
        if self.image_mode not in _IMAGE_MODES:
            raise ValueError(
                f"image_mode must be one of {_IMAGE_MODES}, got {self.image_mode!r}"
            )

        center = tuple(self.panel_center)
        if len(center) != 2:
            raise ValueError(
                f"panel_center must be (x, y), got {self.panel_center!r}"
            )
        for axis, value in zip("xy", center):
            if not np.isfinite(float(value)):
                raise ValueError(
                    f"panel_center {axis} must be finite, got {self.panel_center!r}"
                )
        if float(self.panel_radius) <= 0.0:
            # radius <= 0 is "no crop" in crop_panel_roi, but for a *dataset*
            # config it can only be a mistake: the pupil would occupy ~5x7 of
            # 4096 cells and the rest would be flat-field background.
            raise ValueError(
                f"panel_radius must be > 0, got {self.panel_radius}; radius <= 0 "
                f"disables cropping and is never what a dataset wants"
            )

        resolution = tuple(self.panel_resolution)
        if len(resolution) != 2:
            raise ValueError(
                f"panel_resolution must be (width, height), got "
                f"{self.panel_resolution!r}"
            )
        for axis, value in zip(("width", "height"), resolution):
            if int(value) < 1:
                raise ValueError(
                    f"panel_resolution {axis} must be >= 1, got "
                    f"{self.panel_resolution!r}"
                )

        if float(self.zernike_radius) < 1.0:
            # zernike_coeffs_to_panel requires radius >= 1; checked here so the
            # error names the config field rather than a coefficient vector.
            raise ValueError(f"zernike_radius must be >= 1, got {self.zernike_radius}")
        if float(self.freeform_radius) <= 0.0:
            raise ValueError(
                f"freeform_radius must be > 0, got {self.freeform_radius}"
            )


@dataclass(frozen=True, eq=False)
class HwSample:
    """One materialised training sample: INPUT phase + exposure, OUTPUT image.

    ``eq=False`` is deliberate. The generated ``__eq__`` would compare the numpy
    fields element-wise and then ``bool()`` the result, which raises
    ``ValueError: The truth value of an array with more than one element is
    ambiguous`` -- so the default dataclass equality is unusable here. Identity
    equality is the correct semantic for an owning array container; value
    equality is expressed with :func:`numpy.array_equal` on the fields.

    Attributes:
        phase_cos: ``(grid, grid) float32`` C-contiguous real part of the
            coherently averaged phasor, owning its data.
        phase_sin: ``(grid, grid) float32`` imaginary part, same layout.
        image: ``(grid, grid) float32`` far-field target window.
        exposure_ms: Camera exposure in milliseconds, or ``None`` when neither
            the record nor the sidecar carried one.
        contrast: ``(grid, grid) float32`` per-cell coherence
            ``hypot(phase_cos, phase_sin)`` in ``[0, 1]``. Computed from the two
            phasor channels, never stored as an independent field of the record.
        fov_px: The smaller side of the raw ``_img`` this sample came from, in
            camera pixels, or ``None`` when the frame carried no usable geometry.

            Carried on the sample rather than re-derived by the consumer: the
            record is already in hand here, so reading the shape costs nothing,
            whereas asking again later forces a re-open of the pickle -- up to
            3.42 GB -- which is exactly what the on-disk cache exists to avoid.
    """

    phase_cos: NDArray[np.float32]
    phase_sin: NDArray[np.float32]
    image: NDArray[np.float32]
    exposure_ms: float | None
    contrast: NDArray[np.float32]
    fov_px: int | None = None


def _frame_fov_px(record: Mapping[str, Any]) -> int | None:
    """Return the smaller side of the record's raw ``_img``, or ``None``.

    Args:
        record: One materialised record mapping.

    Returns:
        ``min(height, width)`` of the source frame, or ``None`` when ``_img`` is
        absent or is not a 2-D array.
    """
    shape = getattr(record.get(_IMAGE_KEY), "shape", None)
    if isinstance(shape, tuple) and len(shape) == 2:
        return int(min(int(shape[0]), int(shape[1])))
    return None


# ---------------------------------------------------------------------------
# Payload deserialisation and record addressing
# ---------------------------------------------------------------------------
def _load_payload(path: Path) -> Any:
    """Unpickle one debug dump.

    The only place in this module that opens a file. The handle is closed by the
    ``with`` block and is **never** stored on ``self``, which is what lets a
    :class:`PayloadStore` (and the Dataset holding it) be re-pickled for a
    Windows ``spawn`` worker without dragging an open file object along.

    Args:
        path: Absolute path of the ``.pkl``.

    Returns:
        Whatever ``pickle.load`` produced: usually ``dict[int, dict]``, a
        :class:`~ao_shaping.utils.io.file.Recorder`, or ``None`` for one of the
        3 aborted-run stubs in the corpus.

    Raises:
        HwRecordError: If the file is missing, unreadable, or not unpicklable.
            The message names ``path``.
    """
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)  # noqa: S301 - repo-internal debug dumps
    except _LOAD_ERRORS as exc:
        raise HwRecordError(f"Could not unpickle debug payload {path}: {exc}") from exc


def _dict_sort_key(item: tuple[Any, Any]) -> tuple[int, int, str]:
    """Total ordering over ``(key, record)`` pairs of a record ``dict``.

    Mirrors ``ml.hwdataset.index._dict_sort_key``, which is where
    :attr:`~ml.hwdataset.index.HwRecordRef.position` was computed. Duplicated
    rather than imported because that helper is private to ``index``; if the two
    ever disagreed, a record with a non-``int`` key would be read from the wrong
    position -- silently, and with a perfectly plausible phase.

    Args:
        item: One ``(key, record)`` pair.

    Returns:
        ``(rank, numeric_key, text_key)`` where ``rank`` is ``0`` for integers.
    """
    key = item[0]
    if isinstance(key, int) and not isinstance(key, bool):
        return (0, key, "")
    return (1, 0, str(key))


def _ordered_items(payload: Mapping[Any, Any]) -> list[tuple[Any, Any]]:
    """Return a ``dict`` payload's items in the index's canonical order.

    Args:
        payload: A ``dict``-like record payload.

    Returns:
        ``(key, record)`` pairs sorted by :func:`_dict_sort_key`, i.e. integer
        keys numerically first, everything else by its string form.
    """
    return sorted(payload.items(), key=_dict_sort_key)


def _extract_record(payload: Any, ref: HwRecordRef) -> Mapping[str, Any]:
    """Pull the record addressed by ``ref`` out of a deserialised payload.

    Args:
        payload: Whatever :func:`_load_payload` returned.
        ref: The reference being materialised. Its :attr:`path` and
            :attr:`position` are used verbatim in error messages.

    Returns:
        The record mapping. The payload itself is **not** returned, so a caller
        cannot accidentally retain the whole file (up to 3.42 GB).

    Raises:
        HwRecordError: If the payload is neither a ``Mapping`` nor an object
            exposing a ``history`` list, if the addressed entry is absent, or if
            the addressed entry is not a ``Mapping``.
    """
    where = f"{ref.path} position {ref.position}"

    if isinstance(payload, Mapping):
        if ref.key is not None:
            try:
                record = payload[ref.key]
            except (KeyError, IndexError, TypeError) as exc:
                raise HwRecordError(
                    f"Record {where} (key {ref.key!r}) is absent from the payload; "
                    f"the payload holds {len(payload)} entries"
                ) from exc
        else:
            # A non-int dict key: the index stamped key=None and computed
            # `position` over the sorted order, so that is what is addressed.
            items = _ordered_items(payload)
            if not 0 <= int(ref.position) < len(items):
                raise HwRecordError(
                    f"Record {where} is out of range for a payload of "
                    f"{len(items)} entries"
                )
            record = items[int(ref.position)][1]
    else:
        history = getattr(payload, _HISTORY_ATTR, None)
        if not isinstance(history, (list, tuple)):
            raise HwRecordError(
                f"Record {where} is in a payload of type {type(payload).__name__}, "
                f"which is neither a dict of records nor an object with a "
                f"{_HISTORY_ATTR!r} list"
            )
        if not 0 <= int(ref.position) < len(history):
            raise HwRecordError(
                f"Record {where} is out of range for a {len(history)}-entry "
                f"{_HISTORY_ATTR!r} list"
            )
        record = history[int(ref.position)]

    if not isinstance(record, Mapping):
        raise HwRecordError(
            f"Record {where} is a {type(record).__name__}, not a record mapping"
        )
    return record


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _require_field(record: Mapping[str, Any], ref: HwRecordRef, field: str) -> Any:
    """Fetch a mandatory record field.

    Args:
        record: One record mapping.
        ref: The reference being materialised, for the error message.
        field: Key to fetch.

    Returns:
        The stored value.

    Raises:
        HwRecordError: If ``field`` is absent.
    """
    try:
        return record[field]
    except KeyError as exc:
        raise HwRecordError(
            f"Record {ref.path} position {ref.position} has no {field!r} key"
        ) from exc


def _require_finite(values: Any, ref: HwRecordRef, field: str) -> Any:
    """Reject a non-finite record field instead of letting transforms zero it.

    :func:`~ml.hwdataset.transforms.coherent_block_mean`,
    :func:`~ml.hwdataset.transforms.grayscale_to_phase_rad` and
    :func:`~ml.hwdataset.transforms.farfield_frame_to_grid` all replace non-finite
    pixels with ``0.0`` -- correct for a stray NaN in a mostly good frame, and
    catastrophic here: a NaN phase would be silently converted into a *flat
    phase*, i.e. a fabricated "commanded nothing" training sample that the
    optimiser would then learn to trust. Reaching the transform with a non-finite
    value means something upstream is wrong, so it is raised instead.

    Args:
        values: The value read from the record. Not converted, so an integer or
            boolean array costs nothing.
        ref: The reference being materialised, for the error message.
        field: Name of the record field, quoted in the error message.

    Returns:
        ``values`` unchanged, so this can be used inline.

    Raises:
        HwRecordError: If ``values`` is a float/complex array holding a NaN or
            an infinity.
    """
    array = np.asarray(values)
    if array.dtype.kind not in ("f", "c"):
        # int/uint/bool are finite by construction.
        return values
    if not bool(np.isfinite(array).all()):
        raise HwRecordError(
            f"Record {ref.path} position {ref.position} carries non-finite "
            f"values in {field!r}; transforms would silently replace them with "
            f"zeros, fabricating a flat phase"
        )
    return values


def _own_copy(values: Any) -> NDArray[np.float32]:
    """Return a C-contiguous ``float32`` array that owns its buffer.

    Every array handed to a caller goes through this. The payload is shared
    through the LRU, so two samples taken from the same file must not share
    memory, and the caller is free to normalise in place (a very common thing
    for a Dataset collate function to do). ``np.array(..., copy=True)`` is the
    only spelling that guarantees ``flags.owndata is True``; ``ascontiguousarray``
    is not, because it returns its input unchanged when the input is already
    contiguous -- which is exactly the case for a panel that was cropped to its
    own extent.

    Args:
        values: Any array-like.

    Returns:
        A fresh ``float32`` C-contiguous array owning its data.
    """
    return np.array(values, dtype=np.float32, order="C", copy=True)


# ---------------------------------------------------------------------------
# The payload cache
# ---------------------------------------------------------------------------
class PayloadStore:
    """Bounded LRU over deserialised pickle payloads, keyed by resolved path.

    Sized in **files**, not bytes: ``pickle.load`` is monolithic so the
    granularity of reclamation is a whole file, and the largest is 3.42 GB. The
    default capacity of 1 is the minimum that still pays off under
    ``FileGroupedSampler``, where each file is visited as one contiguous group
    per epoch: with one resident payload an epoch costs ``sum(file sizes)`` at
    ~2 GB/s instead of ``n_records * file size``.

    Pickling contract (Windows ``spawn`` re-pickles the Dataset for every
    worker): :meth:`__getstate__` drops the cache and any bookkeeping, and can
    never carry a deserialised payload; :meth:`__setstate__` restores an empty
    cache. No open file handle is ever stored on ``self`` -- the read handle
    lives only inside :func:`_load_payload`'s ``with`` block.
    """

    def __init__(self, capacity: int = 1) -> None:
        """Create an empty store.

        Args:
            capacity: Maximum number of deserialised payloads to retain. ``0``
                disables retention entirely: every :meth:`get` re-reads its file,
                which is the honest setting for a memory-capped run.

        Raises:
            ValueError: If ``capacity`` is negative.
        """
        if int(capacity) < 0:
            raise ValueError(f"capacity must be >= 0, got {capacity}")
        self._capacity = int(capacity)
        self._cache: OrderedDict[Path, Any] = OrderedDict()

    @property
    def capacity(self) -> int:
        """Return the maximum number of retained payloads.

        Returns:
            The capacity this store was built with.
        """
        return self._capacity

    @property
    def cached_paths(self) -> tuple[Path, ...]:
        """Return the resident payload paths, least recently used first.

        Returns:
            The cached paths in LRU order, so ``cached_paths[-1]`` is the last
            file touched and ``cached_paths[0]`` is the next eviction candidate.
            Empty when nothing is resident.
        """
        return tuple(self._cache)

    def get(self, ref: HwRecordRef) -> Mapping[str, Any]:
        """Return the record mapping for ``ref``, loading its file if needed.

        Args:
            ref: The record to fetch. Its ``path`` is resolved before being used
                as a cache key, so two refs differing only by relative/absolute
                form or by ``..`` segments share one entry.

        Returns:
            The record mapping, addressed by ``ref.key`` for a ``dict`` payload
            and by ``ref.position`` for a ``history`` payload. The payload is
            **not** returned, so a caller cannot retain a whole file.

        Raises:
            HwRecordError: If the file is missing or unreadable, the payload is
                neither ``dict[int, dict]`` nor an object with a ``history`` list,
                or the addressed record is absent or not a ``Mapping``.
        """
        key = Path(ref.path).resolve()
        if key in self._cache:
            self._cache.move_to_end(key)
            return _extract_record(self._cache[key], ref)

        payload = _load_payload(key)
        if self._capacity > 0:
            self._cache[key] = payload
            self._cache.move_to_end(key)
            # popitem(last=False) is the LRU end; the whole payload is dropped at
            # once, which is the only reclamation granularity pickle allows.
            while len(self._cache) > self._capacity:
                evicted, _ = self._cache.popitem(last=False)
                logger.debug("Evicted debug payload {} from the payload cache", evicted)

        return _extract_record(payload, ref)

    def clear(self) -> None:
        """Drop every resident payload.

        Used to release RAM between epochs or after building the on-disk cache.
        """
        self._cache.clear()

    def __getstate__(self) -> dict[str, Any]:
        """Return the picklable state: the capacity, never the payloads.

        Returns:
            A ``dict`` holding only ``capacity``. The deserialised payloads (up
            to 3.42 GB each) are deliberately absent, so re-pickling a Dataset
            for a ``spawn`` worker cannot ship gigabytes over a pipe.
        """
        return {"capacity": int(self._capacity)}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore from a state dict, always with an empty cache.

        Args:
            state: Whatever :meth:`__getstate__` produced.
        """
        self._capacity = int(state.get("capacity", 1))
        self._cache = OrderedDict()

    def __len__(self) -> int:
        """Return the number of resident payloads.

        Returns:
            ``len(self._cache)``, which stays ``0`` when the capacity is ``0``.
        """
        return len(self._cache)

    def __repr__(self) -> str:
        """Return a debug representation naming the capacity and the cache.

        Returns:
            A single-line repr such as ``PayloadStore(capacity=1, cached=2)``.
        """
        return f"PayloadStore(capacity={self._capacity}, cached={len(self._cache)})"


# ---------------------------------------------------------------------------
# The materialiser
# ---------------------------------------------------------------------------
class Materialiser:
    """Turns :class:`HwRecordRef` objects into :class:`HwSample` arrays.

    Stateless apart from the :class:`PayloadStore` and the on-disk cache readers:
    two instances with equal :class:`MaterialiserConfig` produce equal arrays,
    which is what lets the cache builder and the Dataset share this one
    implementation.

    When ``use_cache`` is set (the default) and a cache directory built with the
    *same* configuration exists next to the source pickle, the arrays are read
    from the memory-mapped cache instead of being recomputed. The result is
    bit-identical either way -- that is what the test suite pins -- so the flag
    changes speed and peak RAM, never the numbers. Any problem reading the cache
    (absent, stale, truncated, this one ref not cached) falls back to the direct
    path for that ref.
    """

    def __init__(
        self,
        config: MaterialiserConfig | None = None,
        *,
        cache_size: int = 1,
        use_cache: bool = True,
        cache_root: Path | None = None,
    ) -> None:
        """Validate the configuration and build the payload store.

        Args:
            config: Numeric configuration. ``None`` means
                :class:`MaterialiserConfig` with every default (a 64x64 sample
                from a 1000x1000 ROI of a 1920x1200 panel).
            cache_size: Payload LRU capacity in files, and equally the number of
                cache directories kept mapped at once. Not part of
                :class:`MaterialiserConfig` on purpose: it changes speed and RAM,
                never the numbers. ``1`` is the useful default given
                :class:`~ml.hwdataset.dataset.FileGroupedSampler`.
            use_cache: Read from the on-disk cache when a current one exists.
                Defaults to ``True``; set ``False`` to force the direct path
                (which is how the cache's bit-identity is tested).
            cache_root: Root for the cache directories, or ``None`` for the
                sibling ``.hw_cache`` directory next to each pickle. Only
                consulted when ``use_cache`` is set.

        Raises:
            ValueError: Propagated from
                :meth:`MaterialiserConfig.validate`, or from
                :class:`PayloadStore` for a negative ``cache_size``.
        """
        self._config = MaterialiserConfig() if config is None else config
        self._config.validate()
        self._store = PayloadStore(capacity=int(cache_size))
        self._use_cache = bool(use_cache)
        self._cache_root = Path(cache_root) if cache_root is not None else None
        self._cache_capacity = max(1, int(cache_size))
        self._cached: OrderedDict[Path, Any] = OrderedDict()

    @property
    def config(self) -> MaterialiserConfig:
        """Return the validated numeric configuration.

        Returns:
            The :class:`MaterialiserConfig` this materialiser was built with.
        """
        return self._config

    @property
    def store(self) -> PayloadStore:
        """Return the payload LRU.

        Returns:
            The :class:`PayloadStore` backing :meth:`materialise`, exposed so a
            caller can :meth:`~PayloadStore.clear` it between epochs.
        """
        return self._store

    @property
    def use_cache(self) -> bool:
        """Whether the on-disk cache is consulted by :meth:`materialise`."""
        return self._use_cache

    @property
    def cached_pickles(self) -> tuple[Path, ...]:
        """The pickles whose cache directories are currently mapped."""
        return tuple(self._cached)

    def materialise(self, ref: HwRecordRef) -> HwSample:
        """Run the full pipeline for one record.

        Steps, in this order (the order is load-bearing):

        1. ``record = self.store.get(ref)``
        2. phase, dispatching on ``ref.source`` --
           :attr:`~ml.hwdataset.index.PhaseSource.PANEL_RAD` -> ``_phase`` is
           already radians; :attr:`~ml.hwdataset.index.PhaseSource.PANEL_GRAY` ->
           :func:`~ml.hwdataset.transforms.grayscale_to_phase_rad`;
           :attr:`~ml.hwdataset.index.PhaseSource.ZERNIKE` ->
           :func:`~ml.hwdataset.transforms.zernike_coeffs_to_panel`;
           :attr:`~ml.hwdataset.index.PhaseSource.FREEFORM` ->
           :func:`~ml.hwdataset.transforms.freeform_grid_to_panel`.
        3. ``phase_cos, phase_sin = phase_to_grid(phase, grid, center, radius)``.
           Reduce **before** holding on to anything else: the panel map is
           9.2 MB per record and there are 11393 of them.
        4. ``image = farfield_frame_to_grid(record["_img"], grid, mode)``.
        5. ``exposure_ms = ref.exposure_ms`` -- the index already resolved the
           record-then-sidecar precedence, so it is not re-derived here.
        6. ``contrast = hypot(phase_cos, phase_sin)``.

        Args:
            ref: The record to materialise.

        Returns:
            A :class:`HwSample` whose four arrays are C-contiguous ``float32``
            and own their data, so a caller may mutate them without corrupting
            the shared payload or any other sample.

        Raises:
            HwRecordError: For any unreadable/malformed record (see
                :meth:`PayloadStore.get`), for a missing ``_phase`` / ``_c`` /
                ``_img`` key, for a non-finite phase or coefficient array, for a
                Zernike ref with ``n_max is None`` or a freeform ref with
                ``freeform_grid is None``, and for a transform-level ``ValueError``
                (e.g. a coefficient length that disagrees with ``n_max``) which is
                re-raised with the path, position and original message.
        """
        cached = self._cached_sample(ref)
        if cached is not None:
            return cached

        record = self.store.get(ref)
        phase_cos, phase_sin = self.phase_grid(ref)
        image = self._target_grid(record, ref)
        contrast = _own_copy(np.hypot(phase_cos, phase_sin))

        logger.debug(
            "Materialised {} position {} ({}, grid {}): contrast_max={:.4f}",
            ref.path,
            ref.position,
            ref.source.value,
            self._config.grid,
            float(contrast.max()) if contrast.size else 0.0,
        )
        return HwSample(
            phase_cos=phase_cos,
            phase_sin=phase_sin,
            image=image,
            exposure_ms=ref.exposure_ms,
            contrast=contrast,
            fov_px=_frame_fov_px(record),
        )

    def phase_grid(
        self, ref: HwRecordRef
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Run steps 1-3 only: fetch the record, build the panel, reduce it.

        Exposed separately for callers that do not need the target -- the
        on-disk cache builder's diagnostics, and any script that wants to look at
        the commanded field without paying for a CCD frame.

        Args:
            ref: The record whose commanded phase is wanted.

        Returns:
            ``(phase_cos, phase_sin)``, each ``(grid, grid) float32``,
            C-contiguous and owning its data. The ``(1200, 1920)`` panel is
            local to this call.

        Raises:
            HwRecordError: See :meth:`materialise`.
        """
        config = self._config
        record = self.store.get(ref)
        phase = self._panel_phase_rad(record, ref)
        try:
            cos_grid, sin_grid = phase_to_grid(
                phase,
                config.grid,
                center=config.panel_center,
                radius=config.panel_radius,
            )
        except ValueError as exc:
            # Raised by crop_panel_roi / coherent_block_mean for a config that
            # does not fit this panel (an ROI that misses the panel entirely, a
            # non-2-D panel). Re-raised as a record error so the message names
            # the file rather than a bare array shape.
            raise HwRecordError(
                f"Could not reduce the phase of record {ref.path} position "
                f"{ref.position} (source={ref.source.value}, grid={config.grid}, "
                f"center={config.panel_center}, radius={config.panel_radius}): {exc}"
            ) from exc
        return _own_copy(cos_grid), _own_copy(sin_grid)

    def clear(self) -> None:
        """Drop every cached payload and unmap every cache directory."""
        self.store.clear()
        self._close_caches()

    def __getstate__(self) -> dict[str, Any]:
        """Return a pickle-safe state: no payload arrays, no memory maps.

        Both LRUs are dropped rather than carried. :class:`PayloadStore` already
        drops its arrays, and a ``numpy`` memory map is not picklable at all --
        keeping either would break ``num_workers > 0`` on Windows, where the
        Dataset is rebuilt per worker anyway.
        """
        state = dict(self.__dict__)
        state["_store"] = PayloadStore(capacity=self._store.capacity)
        state["_cached"] = OrderedDict()
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore from :meth:`__getstate__` with both caches empty."""
        self.__dict__.update(state)
        self._cached = OrderedDict()

    # -- internals ----------------------------------------------------------
    def _cached_sample(self, ref: HwRecordRef) -> HwSample | None:
        """Serve one ref from the on-disk cache, or ``None`` to fall back.

        A cache directory is accepted only if :func:`cache_is_current` confirms it
        was built for *this* pickle with *this* :class:`MaterialiserConfig`, so a
        cache left over from a different ``grid`` or ``panel_radius`` can never be
        served as if it were current.

        Args:
            ref: The record to serve.

        Returns:
            A :class:`HwSample` read from the cache, or ``None`` -- meaning "not
            cached, recompute it" -- when caching is off, the directory is absent
            or stale, the directory cannot be opened, or this ref is not in it.
        """
        if not self._use_cache:
            return None

        # Deferred: cache.py imports this module, so a module-level import here
        # would be circular. Mirrors the leaf-layer rule in AGENTS.md.
        from ml.hwdataset.cache import (
            HwCacheError,
            cache_dir_for,
            cache_is_current,
            load_hw_cache,
        )

        source = Path(ref.path).resolve()
        cached = self._cached.get(source)
        if cached is None:
            directory = cache_dir_for(Path(ref.path), self._cache_root)
            if not cache_is_current(source, directory, self._config):
                return None
            try:
                cached = load_hw_cache(directory)
            except (HwCacheError, OSError) as exc:
                logger.debug("Hw cache {} is unusable: {}", directory, exc)
                return None
            self._cached[source] = cached
            while len(self._cached) > self._cache_capacity:
                _, evicted = self._cached.popitem(last=False)
                evicted.close()
                logger.debug("Unmapped evicted hw cache for {}", evicted.directory)

        stored = int(ref.key) if ref.key is not None else int(ref.position)
        try:
            slot = cached.position_of(stored)
        except KeyError:
            # Cached under a different key set, or truncated: recompute.
            return None

        phase_cos = cached.phase_cos(slot)
        phase_sin = cached.phase_sin(slot)
        return HwSample(
            phase_cos=phase_cos,
            phase_sin=phase_sin,
            image=cached.image(slot),
            exposure_ms=cached.exposure_ms(slot),
            contrast=_own_copy(np.hypot(phase_cos, phase_sin)),
            fov_px=cached.fov_px(slot),
        )

    def _close_caches(self) -> None:
        """Unmap every open cache directory."""
        while self._cached:
            _, cached = self._cached.popitem()
            cached.close()

    def _panel_phase_rad(
        self, record: Mapping[str, Any], ref: HwRecordRef
    ) -> NDArray[np.float32]:
        """Build the full-panel phase map in radians for one record (step 2).

        A thin ``ValueError`` -> :class:`HwRecordError` wrapper around
        :meth:`_build_panel_phase_rad`, so a rejected coefficient vector (e.g.
        ``len(coeffs)`` disagreeing with ``n_max``) names the record instead of
        surfacing a bare array-shape error from deep inside a transform. The
        already-raised :class:`HwRecordError` cases pass through untouched
        because :class:`HwRecordError` derives from ``RuntimeError``, not
        ``ValueError``.

        Args:
            record: The record mapping from :meth:`PayloadStore.get`.
            ref: The reference being materialised, for messages and for the
                ``n_max`` / ``freeform_grid`` metadata.

        Returns:
            ``(height, width) float32`` unwrapped radians.

        Raises:
            HwRecordError: If a transform rejects its input.
        """
        try:
            return self._build_panel_phase_rad(record, ref)
        except ValueError as exc:
            raise HwRecordError(
                f"Could not build the panel phase of record {ref.path} position "
                f"{ref.position} (source={ref.source.value}, n_max={ref.n_max}, "
                f"freeform_grid={ref.freeform_grid}): {exc}"
            ) from exc

    def _build_panel_phase_rad(
        self, record: Mapping[str, Any], ref: HwRecordRef
    ) -> NDArray[np.float32]:
        """Dispatch on ``ref.source`` to the canonical transform (step 2).

        Args:
            record: The record mapping from :meth:`PayloadStore.get`.
            ref: The reference being materialised, for messages and for the
                ``n_max`` / ``freeform_grid`` metadata.

        Returns:
            ``(height, width) float32`` unwrapped radians. Never wrapped with
            ``mod 2*pi`` here: the driver's ``create_phase_from_array`` is the
            only wrap point in this repo.

        Raises:
            HwRecordError: For a missing/non-finite ``_phase`` or ``_c``, a
                Zernike ref with no ``n_max``, a freeform ref with no
                ``freeform_grid``, or an unknown phase source.
        """
        config = self._config
        source = ref.source

        if source is PhaseSource.PANEL_RAD:
            raw = _require_field(record, ref, _PHASE_KEY)
            _require_finite(raw, ref, _PHASE_KEY)
            return np.asarray(raw, dtype=np.float32)

        if source is PhaseSource.PANEL_GRAY:
            raw = _require_field(record, ref, _PHASE_KEY)
            _require_finite(raw, ref, _PHASE_KEY)
            return grayscale_to_phase_rad(raw, config.slm_max_gray)

        if source is PhaseSource.ZERNIKE:
            if ref.n_max is None:
                raise HwRecordError(
                    f"Record {ref.path} position {ref.position} is a Zernike "
                    f"source with n_terms={ref.n_terms} but no known n_max; the "
                    f"aperture mask and mode count cannot be reconstructed"
                )
            coeffs = _require_field(record, ref, _COEFF_KEY)
            _require_finite(coeffs, ref, _COEFF_KEY)
            return zernike_coeffs_to_panel(
                coeffs,
                n_max=int(ref.n_max),
                resolution=config.panel_resolution,
                radius=config.zernike_radius,
            )

        if source is PhaseSource.FREEFORM:
            if ref.freeform_grid is None:
                raise HwRecordError(
                    f"Record {ref.path} position {ref.position} is a freeform "
                    f"source with n_terms={ref.n_terms} but no known freeform_grid"
                )
            coeffs = _require_field(record, ref, _COEFF_KEY)
            _require_finite(coeffs, ref, _COEFF_KEY)
            return freeform_grid_to_panel(
                coeffs,
                grid=int(ref.freeform_grid),
                resolution=config.panel_resolution,
            )

        raise HwRecordError(
            f"Record {ref.path} position {ref.position} has unsupported phase "
            f"source {source!r}"
        )

    def _target_grid(
        self, record: Mapping[str, Any], ref: HwRecordRef
    ) -> NDArray[np.float32]:
        """Reduce the record's CCD frame to the target grid (step 4).

        Args:
            record: The record mapping from :meth:`PayloadStore.get`.
            ref: The reference being materialised, for the error message.

        Returns:
            ``(grid, grid) float32`` C-contiguous, owning its data.

        Raises:
            HwRecordError: If ``_img`` is absent or carries non-finite values, or
                the transform rejects the frame (not a non-empty 2-D array,
                unknown mode).
        """
        raw = _require_field(record, ref, _IMAGE_KEY)
        _require_finite(raw, ref, _IMAGE_KEY)
        try:
            window = farfield_frame_to_grid(
                raw, self._config.grid, mode=self._config.image_mode
            )
        except ValueError as exc:
            raise HwRecordError(
                f"Could not reduce the far-field frame of record {ref.path} "
                f"position {ref.position} (mode={self._config.image_mode}, "
                f"grid={self._config.grid}): {exc}"
            ) from exc
        return _own_copy(window)
