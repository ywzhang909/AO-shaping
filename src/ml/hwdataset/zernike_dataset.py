"""PyTorch Dataset over the hardware debug corpus, keyed by **Zernike coefficients**.

This is the supervised sibling of :mod:`ml.hwdataset.dataset`. Where that module
feeds a small convolutional network the *commanded SLM phase grid*, this one feeds
the **parametric** description of that phase: the Zernike coefficient vector the
bench actually wrote down in ``record["_c"]``. That is the physically meaningful
input for a model that must *predict the far field from a set of coefficients*,
because the coefficient vector is already the low-dimensional, aperture-aware,
resolution-independent description of the command -- no 64x64 pixel re-rasterising,
no panel-ROI assumption.

============================  What one item looks like  ============================

:meth:`ZernikeCoeffDataset.__getitem__` returns a **dict**, so
``torch.utils.data.default_collate`` needs no ``collate_fn``:

=========================  =====================================================
key                       meaning
=========================  =====================================================
``coeffs``                ``(calc_n_zernike_terms(n_max),) float32`` tensor,
                          **standardised** with the train-split statistics
``coeffs_raw``            same length, **raw** padded radians (no standardisation)
``image``                 ``(1, grid, grid) float32`` tensor, the CCD far-field
                          window, normalised into ``[0, 1]``
``n_terms_record``        ``len(record["_c"])`` actually present in the pickle
``n_max_record``          the record's own radial order, or ``None``
``exposure_ms``           raw exposure in ms, or ``None``
``family``                family label string, e.g. ``"slm_zernike_shaping"``
``sample_idx``            resolved (non-negative) dataset position
``path``                  the source ``.pkl`` as a string
``fov_px``                source-frame side length in camera pixels, or ``None``
=========================  =====================================================

``coeffs_raw`` is carried alongside ``coeffs`` deliberately: it is the *only*
way back to a realisable phase pattern, and re-deriving it from the standardised
tensor would need the statistics back. Two consumers with different ideas about
standardisation can therefore share one Dataset.

=========================  Four decisions worth reviewing  ==========================

1. **Padding is Noll-prefix-preserving and nothing else.**
   :func:`pad_coefficients` copies ``coeffs[:n_present]`` into the front of a
   ``calc_n_zernike_terms(n_max)`` vector and zero-fills the rest. It does **not**
   rescale, re-order, resample, or wrap, and a vector longer than ``n_max``
   allows is an error rather than a silent truncation -- truncating would drop
   the high-order modes, which are exactly the ones a shaping run was chasing.
   ``_c`` is already raw radians in Noll order, so there is deliberately **no**
   ``um_to_waves`` and no ``2*pi`` anywhere on this path.

2. **Statistics are fitted on train files only.**
   :func:`fit_coeff_stats` takes an explicit list of dataset positions, and
   :func:`create_zernike_coeff_dataloaders` calls it with the **train** positions
   *after* the file-level split. Fitting on the whole corpus is test-set leakage:
   the validation and test images would contribute to the input scaling, and any
   reported score would be optimistic by an amount nobody can bound afterwards.

3. **``image_mode="robust"`` is the default, and "sum" is never used.**
   The sibling's ``abs255`` default keeps absolute intensity because *it* feeds
   the exposure in as an input. Here the exposure is **not** a model input, so
   keeping it would only inject the per-frame detector gain (measured frame maxima
   span 255 uint8 / 239.6 float32 / 100.9 float64, a 2.5x spread before any
   physics). ``"robust"`` median-subtracts the read-noise pedestal and divides by
   the **median** of the lit pixels -- immune to hot pixels, unlike ``"peak"``.
   Dividing by total energy (``"sum"``) is never correct here and is not offered:
   it makes every prediction "almost right" in MSE while the beam itself is wrong.

4. **The on-disk ``.hw_cache`` is OFF by default here.**
   That cache stores ``phase_cos``/``phase_sin``/``image``/``exposure``, and
   **not** ``_c`` -- the coefficient vector is only in the pickle. Reading a
   cached image would therefore still need one payload lookup for ``_c``, so the
   cache buys nothing for this Dataset and would hide that dependency. Callers who
   build the cache and want the (small) win can pass ``use_cache=True``.

============================  Windows ``spawn`` contract  =============================

With ``num_workers > 0`` on Windows, torch re-pickles the Dataset per worker. Two
things therefore must never be pickled: the materialiser's LRU (up to 3.42 GB of
arrays per pickle) and any open file handle. See
:meth:`ZernikeCoeffDataset.__getstate__`, which additionally converts the
``MappingProxyType`` sidecars -- which are **not picklable at all**
(``TypeError: cannot pickle 'mappingproxy' object``) -- to plain dicts and
rebuilds them on the other side. The coefficient statistics *are* carried: they
are two ``float64`` vectors of 136 numbers, and a worker that silently lost them
would return unstandardised inputs instead of failing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, Subset

from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms
from ml.hwdataset.dataset import (
    DEFAULT_PREFETCH_FACTOR,
    FileGroupedSampler,
    _fresh_grid_tensor,
    _resolve_pin_memory,
)
from ml.hwdataset.index import (
    DEFAULT_ROOTS,
    HwCorpusIndex,
    HwRecordRef,
    PhaseSource,
    build_hw_index,
)
from ml.hwdataset.records import (
    HwRecordError,
    Materialiser,
    MaterialiserConfig,
)
from ml.hwdataset.transforms import farfield_frame_to_grid

__all__ = [
    "ZERNIKE_COEFF_SOURCES",
    "DEFAULT_N_MAX",
    "ZernikeCoeffDataset",
    "pad_coefficients",
    "fit_coeff_stats",
    "build_zernike_coeff_dataloader",
    "create_zernike_coeff_dataloaders",
]

#: The only phase source this Dataset consumes. A record qualifies when it has
#: coefficients and no panel array -- the index resolves that in
#: :func:`~ml.hwdataset.index.build_hw_index`, and a record that *also* carries
#: ``_phase`` is classified by that panel instead, so it never lands here.
ZERNIKE_COEFF_SOURCES: tuple[PhaseSource, ...] = (PhaseSource.ZERNIKE,)

#: Default maximum radial order. ``calc_n_zernike_terms(15) == 136``, so the
#: default input is 136-wide -- wide enough to hold the 78-term (``n_max=11``)
#: records the corpus actually contains, and still a small vector.
DEFAULT_N_MAX: int = 15

#: Default far-field scaling. See decision 3 in the module docstring.
DEFAULT_IMAGE_MODE: str = "robust"

#: Record key holding the Noll-ordered, raw-radian coefficient vector. Duplicated
#: from :mod:`ml.hwdataset.records` rather than imported, because that module
#: keeps the constant private and this one is a consumer, not a re-exporter.
_COEFF_KEY = "_c"
_IMAGE_KEY = "_img"


# ---------------------------------------------------------------------------
# Pure helpers: padding
# ---------------------------------------------------------------------------
def pad_coefficients(
    coeffs: Sequence[float] | np.ndarray,
    n_max: int = DEFAULT_N_MAX,
) -> np.ndarray:
    """Return ``coeffs`` zero-padded to the ``n_max`` vector length.

    The padding is **exactly** Noll-prefix-preserving: element ``i`` of the
    result is element ``i`` of the input for every ``i < len(coeffs)``, and every
    element past that is ``0.0``. Nothing is rescaled, re-ordered, wrapped or
    interpolated -- ``_c`` is already raw radians in Noll order, and converting it
    to waves or multiplying by ``2*pi`` here would silently corrupt it.

    Args:
        coeffs: The record's coefficient vector. Any 1-D sequence or array of
            finite numbers.
        n_max: Maximum radial order. The result length is
            ``calc_n_zernike_terms(n_max)``, i.e. ``(n_max + 1) * (n_max + 2) / 2``.

    Returns:
        A ``(calc_n_zernike_terms(n_max),) float64`` array owning its data.

    Raises:
        ValueError: If ``n_max`` is negative, ``coeffs`` is not 1-D, any element is
            non-finite, or ``coeffs`` is longer than ``calc_n_zernike_terms(n_max)``.
            The last case is an error rather than a truncation on purpose: dropping
            the tail would discard the high-order modes, which are the ones a
            shaping run actually drives.
    """
    order = int(n_max)
    if order < 0:
        raise ValueError(f"n_max must be >= 0, got {n_max}")
    n_terms = calc_n_zernike_terms(order)

    array = np.asarray(coeffs, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(
            f"coeffs must be a 1-D vector of Zernike coefficients, got shape "
            f"{array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"coeffs must be finite; got {int(np.size(array) - np.count_nonzero(np.isfinite(array)))} "
            f"non-finite value(s) out of {array.size}"
        )
    if array.size > n_terms:
        raise ValueError(
            f"coeffs has {array.size} entries but n_max={order} only admits "
            f"{n_terms} ({order}+1)({order}+2)/2; raise n_max to keep the tail "
            f"instead of truncating it"
        )

    padded = np.zeros(n_terms, dtype=np.float64)
    padded[: array.size] = array
    return padded


# ---------------------------------------------------------------------------
# Pure helpers: statistics
# ---------------------------------------------------------------------------
def _checked_positions(
    index: HwCorpusIndex, positions: Sequence[int]
) -> list[int]:
    """Validate and normalise a list of dataset positions.

    Args:
        index: The index the positions refer to.
        positions: Positions into ``index.records``.

    Returns:
        The positions as a list of non-negative ``int``, in the order given, with
        duplicates preserved (a caller passing a repeated position gets the
        weight it asked for).

    Raises:
        TypeError: If some position does not implement ``__index__``.
        ValueError: If ``positions`` is empty or a position is out of range. An
            empty selection is rejected here rather than silently producing an
            identity standardisation, because "fit on nothing" is always a bug in
            the caller -- almost always a split that selected no files.
    """
    total = len(index.records)
    resolved: list[int] = []
    for raw in positions:
        try:
            value = int(raw.__index__())  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise TypeError(
                f"positions must be integers, got {type(raw).__name__}"
            ) from exc
        if value < 0:
            value += total
        if not 0 <= value < total:
            raise ValueError(
                f"position {value} is out of range for an index of {total} records"
            )
        resolved.append(value)
    if not resolved:
        raise ValueError(
            "fit_coeff_stats got no positions: the standardisation must be fitted "
            "on at least one record (the TRAIN split, in the normal flow)"
        )
    return resolved


def _frame_side_px(record: Mapping[str, Any]) -> int | None:
    """Smaller side of the record's raw CCD frame, in pixels.

    The same quantity :class:`~ml.hwdataset.dataset.HwPhaseImageDataset` reports
    as ``fov_px``, recomputed here from the frame already in hand rather than
    re-read from the payload. ``None`` when the record has no usable 2-D frame --
    the honest answer, because ``fov_px`` is the one field a consumer filters on
    and fabricating it would defeat that.
    """
    raw = record.get(_IMAGE_KEY)
    if raw is None:
        return None
    shape = getattr(np.asarray(raw), "shape", ())
    if len(shape) != 2:
        return None
    return int(min(int(shape[0]), int(shape[1])))


def _record_coefficients(
    store: Any, ref: HwRecordRef, n_max: int
) -> tuple[np.ndarray, int]:
    """Read one record's ``_c`` and return it padded alongside its stored length.

    The two results differ on purpose. ``padded`` is always
    ``calc_n_zernike_terms(n_max)`` wide and is what the model consumes; ``present``
    is ``len(record["_c"])``, which is what ``n_terms_record`` reports. Returning
    only the padded vector would make ``n_terms_record`` report the *target* width
    (136) for every record, erasing the very length information a consumer needs to
    know how many modes the run actually commanded.

    Args:
        store: The materialiser's :class:`~ml.hwdataset.records.PayloadStore`, so
            the file stays resident across the records of one file group.
        ref: The record to read. Must be a Zernike-source record.
        n_max: Maximum radial order for the padding.

    Returns:
        ``(padded, present)`` where ``padded`` is the
        ``(calc_n_zernike_terms(n_max),)`` raw vector and ``present`` is the number
        of coefficients actually stored on disk.

    Raises:
        ValueError: If ``ref`` is not a :attr:`PhaseSource.ZERNIKE` record, or if
            its payload has no usable ``_c``. Both name the file and position, so
            the message points at the offending pickle rather than at a bare
            array shape.
    """
    if ref.source is not PhaseSource.ZERNIKE:
        raise ValueError(
            f"record {ref.path} position {ref.position} has source "
            f"{ref.source.value!r}, not {PhaseSource.ZERNIKE.value!r}; fit the "
            f"statistics on an index already filtered with sources="
            f"{[s.value for s in ZERNIKE_COEFF_SOURCES]}"
        )
    record = store.get(ref)
    raw = record.get(_COEFF_KEY)
    if raw is None:
        raise ValueError(
            f"record {ref.path} position {ref.position} is indexed as Zernike but "
            f"its payload has no {_COEFF_KEY!r} key"
        )
    present = int(np.asarray(raw).size)
    return pad_coefficients(raw, n_max), present


def fit_coeff_stats(
    index: HwCorpusIndex,
    positions: Sequence[int],
    *,
    n_max: int = DEFAULT_N_MAX,
    cache_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-coefficient mean and standard deviation over ``positions``.

    **Pass the TRAIN split's positions, never the whole corpus.** The returned
    vectors are applied to every split, so statistics fitted on all of it would
    let validation and test inputs contribute to their own scaling -- textbook
    test-set leakage, and the resulting validation scores are optimistic by an
    amount that cannot be corrected after the fact.

    Statistics are computed on the **padded** vectors, i.e. on exactly the arrays
    :class:`ZernikeCoeffDataset` will hand out. That matters for the corpus's
    mixed lengths (15 / 36 / 78 terms): a shorter record contributes real zeros
    to the tail indices, which is what makes the tail's mean and std describe the
    distribution the model will actually see.

    A component whose standard deviation is zero -- the tail beyond the corpus's
    longest record, and any coefficient no train file ever drove -- gets
    ``std = 1.0`` with its mean untouched, so the standardised value stays exactly
    ``0.0`` instead of becoming ``inf``/``nan``. The substitution is per
    component, not global: a zero-variance tail must not scale up the components
    that do vary.

    Args:
        index: The corpus index the positions refer to. Must already be filtered
            to :data:`ZERNIKE_COEFF_SOURCES` by the caller.
        positions: Dataset positions to fit on -- the train file positions, in the
            normal flow.
        n_max: Maximum radial order, fixing the returned vector length.
        cache_size: Payload LRU size in files. ``1`` suits a file-level split,
            where each file is read as one contiguous run.

    Returns:
        ``(mean, std)``, two ``(calc_n_zernike_terms(n_max),) float64`` arrays.

    Raises:
        TypeError: If a position is not an integer.
        ValueError: If ``positions`` is empty, a position is out of range, a
            record is not Zernike-sourced, a payload has no ``_c``, a coefficient
            vector is malformed, or ``cache_size`` is negative.
    """
    resolved = _checked_positions(index, positions)
    n_terms = calc_n_zernike_terms(int(n_max))
    # The config is irrelevant here: only ``store.get`` is used, never
    # ``materialise``. ``use_cache=False`` keeps this a plain pickle read.
    materialiser = Materialiser(cache_size=int(cache_size), use_cache=False)

    try:
        rows = np.empty((len(resolved), n_terms), dtype=np.float64)
        for row, position in enumerate(resolved):
            ref = index.records[position]
            rows[row], _present = _record_coefficients(
                materialiser.store, ref, int(n_max)
            )
    finally:
        materialiser.clear()

    if rows.shape[0] < 2:
        logger.warning(
            "Fitting Zernike coefficient statistics on {} record(s) leaves the "
            "standard deviation undefined; falling back to mean=0.0, std=1.0 "
            "(the standardisation degenerates to the identity)",
            rows.shape[0],
        )
        return np.zeros(n_terms, dtype=np.float64), np.ones(n_terms, dtype=np.float64)

    mean = rows.mean(axis=0)
    std = rows.std(axis=0)
    degenerate = ~np.isfinite(std) | (std <= 0.0)
    if bool(np.any(degenerate)):
        logger.info(
            "{} of {} Zernike coefficients are constant over the {} fitted "
            "records (n_max={} is above the corpus's longest vector, or no fitted "
            "file drives them); their std is set to 1.0 so the standardised value "
            "stays exactly 0.0",
            int(np.count_nonzero(degenerate)),
            n_terms,
            rows.shape[0],
            n_max,
        )
    std = np.where(degenerate, 1.0, std).astype(np.float64)

    logger.info(
        "Fitted Zernike coefficient statistics on {} records: n_max={} terms={} "
        "families={}",
        rows.shape[0],
        n_max,
        n_terms,
        sorted({index.records[position].family for position in resolved}),
    )
    return mean, std


# ---------------------------------------------------------------------------
# The Dataset
# ---------------------------------------------------------------------------
class ZernikeCoeffDataset(Dataset[dict[str, Any]]):
    """Lazy Dataset pairing a Zernike coefficient vector with its CCD far field.

    ``__init__`` opens NO pickle: it retains only the immutable index, the
    materialisation config, and the two small coefficient-statistic vectors.
    ``__getitem__`` materialises exactly one record through
    :class:`~ml.hwdataset.records.Materialiser` and reads its ``_c`` from the
    payload that materialiser already has resident.
    """

    def __init__(
        self,
        index: HwCorpusIndex,
        *,
        config: MaterialiserConfig | None = None,
        n_max: int = DEFAULT_N_MAX,
        coeff_mean: np.ndarray | Sequence[float] | None = None,
        coeff_std: np.ndarray | Sequence[float] | None = None,
        cache_size: int = 1,
        require_exposure: bool = True,
        use_cache: bool = False,
        cache_root: Path | None = None,
    ) -> None:
        """Prepare a Dataset over the Zernike-sourced records of ``index``.

        Args:
            index: Corpus index from :func:`~ml.hwdataset.index.build_hw_index`.
                It is filtered to :data:`ZERNIKE_COEFF_SOURCES` here, so passing
                the full corpus is correct and is what the builders do.
            config: Phase reconstruction / output geometry. ``None`` uses
                :class:`~ml.hwdataset.records.MaterialiserConfig` with
                ``image_mode="robust"``; see decision 3 in the module docstring.
            n_max: Maximum radial order. Fixes both coefficient vectors at
                ``calc_n_zernike_terms(n_max)`` entries (``136`` at the default).
            coeff_mean: Per-coefficient mean subtracted in ``__getitem__``, of
                length ``calc_n_zernike_terms(n_max)``. ``None`` means all zeros.
                Pass :func:`fit_coeff_stats`' output fitted on the **train** split.
            coeff_std: Per-coefficient divisor, same length. ``None`` means all
                ones. A non-positive or non-finite entry is replaced by ``1.0``
                with a warning rather than producing ``inf`` / ``nan`` targets.
            cache_size: Payload LRU size handed to the
                :class:`~ml.hwdataset.records.Materialiser`. ``1`` is the useful
                default given :class:`~ml.hwdataset.dataset.FileGroupedSampler`.
            require_exposure: Drop records with an unknown exposure.
            use_cache: Read from the on-disk ``.hw_cache`` directories. Defaults to
                ``False``: that cache does not hold ``_c``, so it cannot save a
                pickle read for this Dataset. It remains available for callers who
                have built it and want the partial win.
            cache_root: Root for the cache directories, or ``None`` for the
                sibling ``.hw_cache`` directory next to each pickle. Only
                consulted when ``use_cache`` is set.

        Raises:
            ValueError: If the filtered index holds no records, ``n_max`` is
                negative, ``cache_size`` is negative, or a supplied statistic has
                the wrong length or a non-finite entry.
        """
        if int(cache_size) < 0:
            raise ValueError(f"cache_size must be >= 0, got {cache_size}")
        order = int(n_max)
        if order < 0:
            raise ValueError(f"n_max must be >= 0, got {n_max}")
        n_terms = calc_n_zernike_terms(order)

        self._n_max = order
        self._n_terms = n_terms
        self._config = (
            MaterialiserConfig(image_mode=DEFAULT_IMAGE_MODE)
            if config is None
            else config
        )
        self._cache_size = int(cache_size)
        self._require_exposure = bool(require_exposure)
        self._use_cache = bool(use_cache)
        self._cache_root = Path(cache_root) if cache_root is not None else None

        filtered = index.filter(
            sources=ZERNIKE_COEFF_SOURCES, require_exposure=self._require_exposure
        )
        if not filtered.records:
            raise ValueError(
                "ZernikeCoeffDataset: no Zernike-sourced record survives filtering "
                f"(sources={[s.value for s in ZERNIKE_COEFF_SOURCES]}, "
                f"require_exposure={self._require_exposure}, "
                f"families={list(filtered.families)}, "
                f"files_scanned={filtered.files_scanned}, "
                f"files_usable={filtered.files_usable}). "
                "Nothing to train on: this corpus keeps the coefficients in "
                "record['_c'], which build_hw_index classifies as "
                f"{PhaseSource.ZERNIKE.value!r}."
            )
        self._index = filtered
        self._records = filtered.records

        self._coeff_mean = self._resolve_statistic(coeff_mean, "coeff_mean", 0.0)
        self._coeff_std = self._resolve_statistic(coeff_std, "coeff_std", 1.0)
        if coeff_mean is None and coeff_std is None:
            logger.warning(
                "ZernikeCoeffDataset was built without coeff_mean/coeff_std, so "
                "'coeffs' is the raw padded vector with no standardisation; fit "
                "them on the TRAIN split with fit_coeff_stats for a real run"
            )
        self._materialiser = Materialiser(
            self._config,
            cache_size=self._cache_size,
            use_cache=self._use_cache,
            cache_root=self._cache_root,
        )

        logger.info(
            "ZernikeCoeffDataset ready: {} records from {} pickles ({} files "
            "scanned), n_max={} terms={}, grid={}, image_mode='{}', "
            "standardised={}",
            len(self._records),
            len({record.path for record in self._records}),
            self._index.files_scanned,
            self._n_max,
            self._n_terms,
            self._config.grid,
            self._config.image_mode,
            coeff_mean is not None or coeff_std is not None,
        )

    def _resolve_statistic(
        self,
        values: np.ndarray | Sequence[float] | None,
        name: str,
        fill: float,
    ) -> np.ndarray:
        """Validate one coefficient-statistic vector into a private float64 array.

        Args:
            values: The caller's vector, or ``None`` for the constant ``fill``.
            name: Field name, used in error messages.
            fill: The constant used when ``values is None``.

        Returns:
            A ``(self._n_terms,) float64`` array owning its data.

        Raises:
            ValueError: If the length is wrong or an entry is non-finite. For
                ``coeff_std`` a non-positive entry is additionally replaced by
                ``1.0`` with a warning, so one degenerate coefficient cannot turn
                the whole input into ``inf``.
        """
        if values is None:
            return np.full(self._n_terms, float(fill), dtype=np.float64)
        array = np.array(values, dtype=np.float64, copy=True)
        if array.ndim != 1:
            raise ValueError(
                f"{name} must be a 1-D vector of {self._n_terms} entries, got shape "
                f"{array.shape}"
            )
        if array.size != self._n_terms:
            raise ValueError(
                f"{name} has {array.size} entries but n_max={self._n_max} fixes the "
                f"coefficient length at {self._n_terms}; refit it with "
                f"fit_coeff_stats(..., n_max={self._n_max})"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} must be finite; got "
                f"{int(np.size(array) - np.count_nonzero(np.isfinite(array)))} "
                f"non-finite value(s)"
            )
        if name == "coeff_std":
            nonpositive = array <= 0.0
            if bool(np.any(nonpositive)):
                logger.warning(
                    "{} of {} entries of coeff_std are <= 0; replacing them with "
                    "1.0 so the standardised coefficients stay finite",
                    int(np.count_nonzero(nonpositive)),
                    self._n_terms,
                )
                array = np.where(nonpositive, 1.0, array)
        return array

    # -- introspection ------------------------------------------------------
    @property
    def index(self) -> HwCorpusIndex:
        """Return the filtered index this Dataset iterates over.

        Returns:
            The :class:`~ml.hwdataset.index.HwCorpusIndex` actually in use, which
            is ``index.filter(sources=ZERNIKE_COEFF_SOURCES, ...)``.
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
    def n_max(self) -> int:
        """Return the maximum radial order.

        Returns:
            The ``n_max`` this Dataset was built with.
        """
        return self._n_max

    @property
    def n_terms(self) -> int:
        """Return the coefficient vector length.

        Returns:
            ``calc_n_zernike_terms(self.n_max)``, i.e. ``136`` at the default.
        """
        return self._n_terms

    @property
    def coeff_mean(self) -> np.ndarray:
        """Return the per-coefficient mean subtracted from every record.

        Returns:
            A ``(n_terms,) float64`` array; all zeros when none was supplied.
        """
        return self._coeff_mean

    @property
    def coeff_std(self) -> np.ndarray:
        """Return the per-coefficient divisor applied to every record.

        Returns:
            A ``(n_terms,) float64`` array; all ones when none was supplied.
        """
        return self._coeff_std

    def __len__(self) -> int:
        """Return the number of Zernike-sourced records.

        Returns:
            ``len(self.records)``.
        """
        return len(self._records)

    # -- item access --------------------------------------------------------
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
                :class:`~ml.hwdataset.records.Materialiser` or its payload store.
                A record that was indexed but cannot be materialised is a real
                failure, never a silent skip.
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
        # Read the payload ONCE and derive the target from `_img` directly.
        #
        # This deliberately does NOT call `Materialiser.materialise()`. That method
        # also builds the panel phase map, and for a ZERNIKE source that means
        # `zernike_coeffs_to_panel` -> `reconstruct_pupil_phase_rad` over the full
        # 1920x1200 panel -- work this Dataset has no use for, since a
        # coefficient-keyed consumer wants `_c` and the measured `_img`, never the
        # reconstructed pupil. Measured on the real corpus it dominated the cost
        # per record by ~24x (195 ms vs 8 ms) and made the GPU sit idle: GPU
        # forward+backward is 17 ms per batch of 64, so the loader, not the
        # device, set the epoch time.
        #
        # The reduction below is the SAME call `_target_grid` makes, with the same
        # arguments, so the emitted image is bit-identical to the materialise()
        # path this replaces.
        record = self._materialiser.store.get(ref)
        raw_coeffs, present = _record_coefficients(
            self._materialiser.store, ref, self._n_max
        )
        standardised = (raw_coeffs - self._coeff_mean) / self._coeff_std
        image = self._target_image(record, ref)

        return {
            "coeffs": torch.from_numpy(
                np.array(standardised, dtype=np.float32, order="C", copy=True)
            ),
            "coeffs_raw": torch.from_numpy(
                np.array(raw_coeffs, dtype=np.float32, order="C", copy=True)
            ),
            "image": _fresh_grid_tensor(image),
            # ``present``, not ``raw_coeffs.size``: the padded width is the
            # *target* and is 136 for every record, so reporting it here would
            # erase how many modes this run actually commanded.
            "n_terms_record": present,
            "n_max_record": None if ref.n_max is None else int(ref.n_max),
            # The index already resolved the record-then-sidecar precedence, so
            # this is ref.exposure_ms -- exactly what materialise() reported.
            "exposure_ms": (
                None if ref.exposure_ms is None else float(ref.exposure_ms)
            ),
            "family": ref.family,
            "sample_idx": position,
            # str, not Path: ``default_collate`` has no dedicated branch for Path.
            "path": str(ref.path),
            # Taken from the frame we already hold, so this never re-opens the
            # pickle -- the sibling's ``_fov_px`` docstring explains why that
            # lookup is fatal on a cache hit.
            "fov_px": _frame_side_px(record),
        }

    def _target_image(
        self, record: Mapping[str, Any], ref: HwRecordRef
    ) -> NDArray[np.float32]:
        """Reduce one record's CCD frame to the ``grid x grid`` target.

        The same reduction :meth:`Materialiser._target_grid` performs, reached
        through the public transform so the Dataset does not depend on a private
        method of its collaborator.

        Args:
            record: The record mapping from the payload store.
            ref: The record's reference, for error messages.

        Returns:
            ``(grid, grid) float32`` window.

        Raises:
            HwRecordError: If ``_img`` is missing, non-finite, or the transform
                rejects the frame. The message names the pickle and position.
        """
        raw = record.get(_IMAGE_KEY)
        if raw is None:
            raise HwRecordError(
                f"record {ref.path} position {ref.position} has no {_IMAGE_KEY!r} key "
                "to build the far-field target from"
            )
        frame = np.asarray(raw)
        if frame.dtype.kind == "f" and not np.isfinite(frame).all():
            raise HwRecordError(
                f"record {ref.path} position {ref.position} carries non-finite "
                f"values in {_IMAGE_KEY!r}"
            )
        try:
            return farfield_frame_to_grid(
                raw, self._config.grid, mode=self._config.image_mode
            )
        except ValueError as exc:
            raise HwRecordError(
                f"Could not reduce the far-field frame of record {ref.path} "
                f"position {ref.position} (mode={self._config.image_mode}, "
                f"grid={self._config.grid}): {exc}"
            ) from exc

    def clear_cache(self) -> None:
        """Drop every cached payload and derived array."""
        self._materialiser.clear()

    # -- pickling -----------------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        """Return a picklable state that carries no arrays and no file handle.

        Three things are deliberately excluded or rewritten:

        * ``_materialiser`` -- its LRU holds up to 3.42 GB of numpy arrays per
          pickle and may hold an open file handle. It is rebuilt in
          :meth:`__setstate__`.
        * every ``sidecar`` ``MappingProxyType`` -- **not picklable at all**
          (``TypeError: cannot pickle 'mappingproxy' object``), which would make
          the whole Dataset unpicklable and therefore break ``num_workers > 0``
          on Windows. They become plain dicts here.
        * ``_index`` -- rebuilt from the already-plain ``_records`` so the two can
          never drift apart.

        The coefficient statistics *are* carried: they are two small ``float64``
        vectors, and a worker that silently lost them would return unstandardised
        inputs instead of failing.

        Returns:
            A pickle-safe ``dict`` of the Dataset's own state.
        """
        return {
            "_records": tuple(
                replace(record, sidecar=dict(record.sidecar))
                for record in self._records
            ),
            "_excluded": dict(self._index.excluded),
            "_files_scanned": self._index.files_scanned,
            "_files_usable": self._index.files_usable,
            "_config": self._config,
            "_n_max": self._n_max,
            "_n_terms": self._n_terms,
            "_coeff_mean": self._coeff_mean,
            "_coeff_std": self._coeff_std,
            "_cache_size": self._cache_size,
            "_require_exposure": self._require_exposure,
            "_use_cache": self._use_cache,
            "_cache_root": self._cache_root,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`__getstate__`, rebuilding index and materialiser.

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
            use_cache=bool(state["_use_cache"]),
            cache_root=state["_cache_root"],
        )


# ---------------------------------------------------------------------------
# The DataLoader factories
# ---------------------------------------------------------------------------
def build_zernike_coeff_dataloader(
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
    n_max: int = DEFAULT_N_MAX,
    coeff_mean: np.ndarray | Sequence[float] | None = None,
    coeff_std: np.ndarray | Sequence[float] | None = None,
    require_exposure: bool = True,
    use_cache: bool = False,
    cache_root: Path | None = None,
) -> DataLoader:
    """Build a ``DataLoader`` over one corpus index, filtered to Zernike records.

    The Zernike filter is applied **here**, not left to the caller: every consumer
    of this function wants coefficient records, and a caller that forgot to filter
    would otherwise get a Dataset that raises deep inside ``pad_coefficients``.

    Args:
        index: Corpus index from :func:`~ml.hwdataset.index.build_hw_index`.
        config: Materialisation config, forwarded to
            :class:`ZernikeCoeffDataset`. ``None`` means
            ``image_mode="robust"``.
        batch_size: Samples per batch.
        num_workers: Worker processes. ``> 0`` re-pickles the Dataset on Windows
            (``spawn``), which is why the pickling contract is part of the class
            contract.
        shuffle: ``True`` uses :class:`~ml.hwdataset.dataset.FileGroupedSampler`
            (the LRU-friendly order), ``False`` torch's ``SequentialSampler``.
        cache_size: Payload LRU size, forwarded to the Dataset.
        num_samples: Epoch length override for the grouped sampler.
        seed: Seed of the grouped sampler's internal generator.
        pin_memory: ``None`` auto-enables it only when CUDA is available **and**
            ``num_workers > 0``. Pass a bool to override.
        drop_last: Drop an incomplete final batch.
        n_max: Maximum radial order, forwarded to the Dataset.
        coeff_mean: Per-coefficient mean, forwarded to the Dataset. ``None``
            means no standardisation -- appropriate for an exploratory loader,
            not for a reported run.
        coeff_std: Per-coefficient divisor, forwarded to the Dataset.
        require_exposure: Drop records with an unknown exposure.
        use_cache: Read from the on-disk ``.hw_cache`` directories. ``False`` by
            default, because that cache does not hold ``_c``.
        cache_root: Root for the cache directories, or ``None`` for the sibling
            ``.hw_cache`` directory next to each pickle.

    Returns:
        A ``DataLoader`` whose ``collate_fn`` is torch's ``default_collate``: the
        sample dict is tensors plus Python scalars/strings, exactly what it
        handles.
    """
    dataset = ZernikeCoeffDataset(
        index,
        config=config,
        n_max=n_max,
        coeff_mean=coeff_mean,
        coeff_std=coeff_std,
        cache_size=cache_size,
        require_exposure=require_exposure,
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


def _file_level_split(
    records: Sequence[HwRecordRef], train_split: float, val_split: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Split record positions by source file, reproducibly.

    The split is by **file**, not by record, for the same reason the sibling
    :func:`~ml.hwdataset.dataset.create_hw_dataloaders` splits that way: records
    inside one pickle are consecutive optimisation epochs of the same run, so a
    record-level split puts near-duplicates on both sides of the train/val
    boundary and inflates every validation metric.

    Files are enumerated in first-appearance order (so the corpus order is
    preserved), permuted with a seeded ``torch.Generator``, then cut.

    Args:
        records: The records to split.
        train_split: Fraction of **files** for training, in ``(0, 1]``.
        val_split: Fraction of **files** for validation, in ``[0, 1]``. The
            remainder is the test split.
        seed: Seed of the split generator.

    Returns:
        ``(train_indices, val_indices, test_indices)`` as lists of positions into
        ``records``.

    Raises:
        ValueError: If a fraction is out of range, the fractions leave no room
            for a third split, or any partition would contain no record.
    """
    train_fraction = float(train_split)
    val_fraction = float(val_split)
    if not math.isfinite(train_fraction) or train_fraction <= 0.0 or train_fraction > 1.0:
        raise ValueError(f"train_split must be in (0, 1], got {train_split!r}")
    if not math.isfinite(val_fraction) or val_fraction < 0.0 or val_fraction > 1.0:
        raise ValueError(f"val_split must be in [0, 1], got {val_split!r}")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            f"train_split ({train_split}) + val_split ({val_split}) must leave room "
            "for a non-empty test split"
        )

    paths: list[Path] = []
    positions_by_path: dict[Path, list[int]] = {}
    for position, record in enumerate(records):
        if record.path not in positions_by_path:
            paths.append(record.path)
            positions_by_path[record.path] = []
        positions_by_path[record.path].append(position)

    generator = torch.Generator().manual_seed(int(seed))
    shuffled_paths = [
        paths[i] for i in torch.randperm(len(paths), generator=generator).tolist()
    ]
    n_train_files = int(len(paths) * train_fraction)
    n_val_files = int(len(paths) * val_fraction)
    if (
        n_train_files < 1
        or n_val_files < 1
        or n_train_files + n_val_files >= len(paths)
    ):
        raise ValueError(
            f"File-level split of {len(paths)} files with train_split={train_split} "
            f"and val_split={val_split} yields train={n_train_files} "
            f"val={n_val_files} test={len(paths) - n_train_files - n_val_files}; "
            "every split needs at least one file"
        )

    def _positions(selected: Sequence[Path]) -> list[int]:
        out: list[int] = []
        for path in selected:
            out.extend(positions_by_path[path])
        return out

    train_indices = _positions(shuffled_paths[:n_train_files])
    val_indices = _positions(shuffled_paths[n_train_files : n_train_files + n_val_files])
    test_indices = _positions(shuffled_paths[n_train_files + n_val_files :])

    for name, indices in (
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ):
        if not indices:
            raise ValueError(
                f"The {name} split of {len(records)} Zernike records over "
                f"{len(paths)} files with train_split={train_split} and "
                f"val_split={val_split} contains no record"
            )
    return train_indices, val_indices, test_indices


def create_zernike_coeff_dataloaders(
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
    n_max: int = DEFAULT_N_MAX,
    require_exposure: bool = True,
    use_cache: bool = False,
    cache_root: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build seeded train/val/test loaders with leakage-free standardisation.

    Follows the same convention as the sibling
    :func:`~ml.hwdataset.dataset.create_hw_dataloaders`: build the index, cut it
    **by file** with ``torch.Generator().manual_seed(seed)`` +
    ``torch.randperm`` + ``Subset``, then shuffle the train loader with a
    :class:`~ml.hwdataset.dataset.FileGroupedSampler` and leave val and test
    sequential so their metrics are reproducible.

    The order of the two steps is load-bearing and is the point of this function:

    1. split **by file** into train / val / test;
    2. fit ``coeff_mean`` / ``coeff_std`` with :func:`fit_coeff_stats` on the
       **train positions only**;
    3. build one Dataset carrying those statistics, and wrap three ``Subset``s of
       it.

    Fitting before splitting -- or on all positions -- would let validation and
    test records contribute to the scaling applied to their own inputs. See
    :func:`fit_coeff_stats`.

    Args:
        roots: Corpus root(s) handed to :func:`~ml.hwdataset.index.build_hw_index`.
        config: Materialisation config. ``None`` means ``image_mode="robust"``.
        batch_size: Samples per batch.
        train_split: Fraction of **files** for training.
        val_split: Fraction of **files** for validation. The remainder is test.
        num_workers: Worker processes per loader.
        seed: Seed of both the split and the grouped sampler.
        cache_size: Payload LRU size.
        limit_files: Cap on the number of scanned pickles (smoke runs).
        index_cache: Optional JSON index cache path.
        n_max: Maximum radial order, fixing the coefficient vector length.
        require_exposure: Drop records with an unknown exposure.
        use_cache: Read from the on-disk ``.hw_cache`` directories. ``False`` by
            default, because that cache does not hold ``_c``.
        cache_root: Root for the cache directories, or ``None`` for the sibling
            ``.hw_cache`` directory next to each pickle.

    Returns:
        ``(train_loader, val_loader, test_loader)``.

    Raises:
        ValueError: Propagated from :func:`_file_level_split` for a bad split, and
            from :func:`fit_coeff_stats` or :class:`ZernikeCoeffDataset` when no
            Zernike record survives filtering.
    """
    index = build_hw_index(
        roots,
        limit_files=limit_files,
        index_cache=index_cache,
        progress_every=0,
    )
    filtered = index.filter(
        sources=ZERNIKE_COEFF_SOURCES, require_exposure=bool(require_exposure)
    )
    records = filtered.records
    if not records:
        raise ValueError(
            "create_zernike_coeff_dataloaders: no Zernike-sourced record in the "
            f"corpus at {roots} "
            f"(sources={[s.value for s in ZERNIKE_COEFF_SOURCES]}, "
            f"require_exposure={require_exposure}, "
            f"families={list(filtered.families)}, "
            f"files_scanned={filtered.files_scanned}, "
            f"files_usable={filtered.files_usable})"
        )

    train_indices, val_indices, test_indices = _file_level_split(
        records, train_split, val_split, seed
    )

    # Step 2: TRAIN POSITIONS ONLY. Everything below reuses these statistics.
    coeff_mean, coeff_std = fit_coeff_stats(
        filtered,
        train_indices,
        n_max=n_max,
        cache_size=cache_size,
    )

    dataset = ZernikeCoeffDataset(
        filtered,
        config=config,
        n_max=n_max,
        coeff_mean=coeff_mean,
        coeff_std=coeff_std,
        cache_size=cache_size,
        require_exposure=require_exposure,
        use_cache=use_cache,
        cache_root=Path(cache_root) if cache_root is not None else None,
    )

    logger.info(
        "Zernike file-level split: {} records over {} files -> train {} / val {} "
        "/ test {} records, fitted on the train positions only",
        len(records),
        len({record.path for record in records}),
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
