"""Corpus discovery + record indexing for the SLM-phase / CCD-image training set.

This module is the **index** half of the :mod:`ml.hwdataset` Dataset: it walks
every debug dump under ``data/debug/``, works out *where each individual record's
commanded phase and camera exposure live*, and emits an immutable
:class:`HwCorpusIndex`. This module has **no** torch import, no SLM, CCD or DM
driver, and no disk writes outside the optional JSON cache, so the index can be
built and unit-tested on a machine with nothing plugged in. The ``Dataset`` /
``DataLoader`` that consumes this index is a separate module; this one only
*locates and labels*.

.. warning::

   **Import cost is NOT torch-free — this is a pre-existing repo defect, not a
   property of this module.** This module's only cross-package import is
   ``infer_n_max`` from :mod:`ml.gsnet_debug.offline`, and *that module
   is itself torch-free* (stdlib + numpy + loguru +
   ``ao_shaping.utils.*``). The cost comes from its **parent package**:
   ``ao_shaping/runners/__init__.py`` documents a PEP 562 lazy ``__getattr__``
   (lines 72-91) but then also executes 19 eager
   ``from ao_shaping.runners.<x> import run as <x>_run`` statements (lines 14-32),
   which defeats the lazy machinery and drags in torch, wandb and every driver
   SDK.

   Measured 2026-10-02: cold ``import ml.hwdataset.index`` = **31.8 s**
   (stdlib baseline 0.1 s; ``import torch`` alone 2.0 s).

   Fix (one edit, outside this module's scope): delete the eager imports at
   ``ao_shaping/runners/__init__.py`` lines 14-32 — the ``__getattr__`` and
   ``_LAZY_RUNNERS`` map below already resolve every one of those names. Until
   then, importing :mod:`ml.hwdataset.index` cannot be made cheap from here, and
   it cannot be made to work on a machine without torch installed.

===========================  The measured corpus  ============================

All numbers below were measured on this repo's ``data/debug/`` tree; they are
quoted throughout so the heuristics below are traceable to evidence rather than
to taste:

* 996 files total, 265 of them ``*.pkl``, **65.88 GB**, largest single pickle
  **3.42 GB**, median 55 MB.
* Layout is **one to two levels deep**, e.g.
  ``data/debug/slm_pib_x/<ts>/<prefix>_<ts>_<ts>.pkl`` and
  ``data/debug/bench_stability_<ts>/<ts-ish>.pkl``. Discovery therefore **must**
  be recursive (``Path.rglob("*.pkl")``): a non-recursive ``glob("*.pkl")`` finds
  **zero** files.
* 11441 records, 11393 usable. Per-record phase sources:

  ==============  =========  ===========  =====================================
  source          records    files        evidence
  ==============  =========  ===========  =====================================
  ``PANEL_GRAY``  7866       53           ``_phase`` is ``(1200,1920) uint16``
  ``PANEL_RAD``   2199       187          ``_phase`` is ``(1200,1920) float32``
  ``ZERNIKE``     1202       18           no ``_phase``; ``len(_c)`` is triangular
  ``FREEFORM``    126        3            no ``_phase``; ``len(_c)`` is a square
  excluded        48         2            no commanded phase at all
  ==============  =========  ===========  =====================================

  The 48 excluded records are ``bench_stability`` (42, a flat-field drift probe
  that commands no phase) plus 6 of the 18 records in
  ``sim_calib_abba_20261001_163743.pkl``.
* Two payload shapes exist: ``dict[int, dict]`` (the overwhelming majority; 3
  files unpickle to ``None`` because their run aborted) and a pickled
  ``ao_shaping.utils.io.file.Recorder`` exposing ``.history: list[dict]`` (8 files
  under ``data/debug/slm_pib_online/``).

=========================  Why classification is per-record  =========================

``sim_calib_abba_20261001_163743.pkl`` is ``{NO_PHASE: 6, FREEFORM: 12}``: 6 of its
18 records carry neither ``_phase`` nor ``_c``, and 12 do. A naive index that
classifies from ``records[0]`` and stamps that label on the whole file silently
drops or mislabels real data. Every record here is therefore judged on its own
contents -- this module's most important single guarantee.

================================  Length ambiguity  ================================

When a record has no ``_phase``, ``len(record["_c"])`` is the discriminator, and
it is **ambiguous for some lengths**: 36 is both a triangular Zernike count
(``n_max=7``) and a perfect square (``6x6``). The precedence is fixed and tested:

1. sidecar ``n_max`` present **and** ``len(_c) == (n_max+1)(n_max+2)//2``
   -> :attr:`PhaseSource.ZERNIKE`;
2. a length that is a perfect square **but not** a triangular Zernike count
   -> :attr:`PhaseSource.FREEFORM` (a freeform grid is by construction
   ``grid*grid``). Unambiguous: 576, 100, ... are never triangular;
3. a length that is a triangular Zernike mode count, verified with
   :func:`~ml.gsnet_debug.offline.infer_n_max`
   -> :attr:`PhaseSource.ZERNIKE`;
4. a length that is **both** (only 36 occurs in this corpus) is resolved by
   :data:`_FREEFORM_FAMILIES`: inside one of those families the square wins,
   everywhere else the triangular count wins;
5. otherwise the record is excluded as
   :attr:`ExclusionReason.ODD_COEFFICIENT_LENGTH`.

Rule 4 exists because the ambiguity is real and family-specific. The
``slm_pib_online`` Recorder files hold ``_c`` of length 36 from the **Zernike**
``slm-pib`` optimiser, and their sidecars (``summary_*.json``) carry no
``n_max`` to settle it -- so they must resolve to ``ZERNIKE`` ``n_max=7``, not to
a 6x6 freeform grid. The freeform families are the ones that genuinely drive a
``grid*grid`` vector. Measured effect of getting this backwards: 192 of 11393
records (1.7%) would be reconstructed as the wrong basis entirely.

================================  Units  ================================

``record["exp_t"]`` is **milliseconds**. Confirmed by
:func:`ao_shaping.drivers.ccd.common.get_camera_exposure_ms`, which reads the
same quantity off the camera as ``exposure_time_ms``; observed ``exp_t`` values
are 0.1 / 0.4 / 1.2 / 1.5 / 2.0 / 80.0. ``_c`` is in **radians** (Noll order for
Zernike, ``grid*grid`` cell amplitudes for freeform) -- the same convention
:mod:`ml.gsnet_debug.offline` documents for its offline transforms.
Sidecar ``exposure_ms`` values seen in the wild are 1.1 / 3.0 and
``exposure_time_ms`` is 1.2.

============================  Layering note (deliberate)  =============================

This module imports :func:`~ml.gsnet_debug.offline.infer_n_max` from
``ao_shaping.runners`` while living under ``src/ml/``. That is a reviewed,
intentional layering choice, not an accident: ``infer_n_max`` is the exact integer
triangular inverse of the Zernike mode count and re-implementing it here would
create a second discriminant that can drift (a repo red line). There is **no
import cycle** -- ``gsnet_offline`` imports only ``ao_shaping.utils.*``, and
``ao_shaping/runners/__init__.py`` resolves its runner attributes through PEP-562
module ``__getattr__``. Cost of that reuse, measured: importing
``ml.gsnet_debug.offline`` takes ~35 s cold, because
``ao_shaping/runners/__init__.py`` eagerly imports the runner modules. Pay it once
per process.

=========================  Where ``build_record_index`` falls short  ==========================

:func:`~ml.gsnet_debug.offline.build_record_index` and its
:class:`~ml.gsnet_debug.offline.RecordIndex` were read before this
module was written, and their *contract* is deliberately mirrored (recursive
discovery, per-file skip-with-warning, JSON index cache, ``limit_files`` applied
after sorting, ``del`` before the next file). Three measured shortfalls make it
unusable as this module's backbone:

1. **It keys on ``dict[int, ...]`` only.** Its ``_load_record_keys`` raises
   ``TypeError`` on any other payload, so the 8 ``Recorder``-shaped files under
   ``data/debug/slm_pib_online/`` are skipped outright -- their history positions
   are unaddressable. Here :attr:`HwRecordRef.position` addresses both shapes and
   :attr:`HwRecordRef.key` is ``None`` for a ``Recorder`` payload.
2. **It reads no sidecar JSON.** Without ``<stem>.json`` / ``summary_*.json`` there
   is no ``n_max`` (the rule-1 disambiguator above), no ``exposure_ms``
   (``model_in_loop_hw_collect`` records carry no ``exp_t`` at all, so their
   exposure comes *only* from the sidecar), and no ``zernike_radius`` /
   ``pupil_center_panel`` needed to reconstruct a Zernike phase later.
3. **It does no per-record phase classification**, which is precisely the
   ``sim_calib_abba`` 6-vs-12 defect described above.

Re-using it for file discovery alone is not worth it either: it only exposes the
file list as a side effect of a *full* ``pickle.load`` of every file, so
borrowing it would mean reading all 65.88 GB a second time.

===========================  Memory contract  ===========================

:attr:`HwRecordRef` retains **no arrays** -- only
``(Path, int, int|None, str, PhaseSource, int, int|None, int|None, float|None,
small dict)``. ``pickle.load`` is monolithic and cannot stream, so transient peak
RAM during a scan is ONE file (up to 3.42 GB for the largest dump); that cannot
be avoided, but :func:`build_hw_index` ``del``\\ s the payload inside
:func:`_scan_file` before the next file is opened, so the high-water mark never
accumulates across the corpus.
"""

from __future__ import annotations

import json
import math
import pickle
import struct
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from loguru import logger

from ml.gsnet_debug.offline import infer_n_max

__all__ = [
    "DEFAULT_ROOTS",
    "ExclusionReason",
    "HwCorpusIndex",
    "HwRecordRef",
    "PhaseSource",
    "build_hw_index",
]

#: Default search roots: the **whole** debug tree, not the ``slm_zernike_*`` /
#: ``slm_pib_*`` subset that ``gsnet_offline.DEFAULT_ROOTS`` targets. This index
#: must see all 265 pickles (54 of which live outside that subset).
DEFAULT_ROOTS: tuple[str, ...] = ("data/debug",)

#: Family labels, matched as **longest-prefix wins in this exact order**. The order
#: is load-bearing: ``slm_pib_online`` must precede ``slm_pib`` because
#: ``"slm_pib_online".startswith("slm_pib")`` is ``True``.
_FAMILY_PREFIXES: tuple[str, ...] = (
    "model_in_loop_hw_collect",
    "model_in_loop_hw_sweep",
    "bench_stability",
    "sim_calib_abba",
    "slm_gsnet_square",
    "slm_zernike_shaping",
    "slm_pib_online",
    "slm_pib",
    "recorder_",
)

_PHASE_KEY = "_phase"
_COEFF_KEY = "_c"
_IMAGE_KEY = "_img"
_RECORDER_HISTORY_ATTR = "history"

#: Families whose ``_c`` is genuinely a freeform ``grid*grid`` control grid rather
#: than a Noll-order Zernike vector. This set is consulted ONLY to settle the one
#: length that is both a perfect square and a triangular Zernike count (36 in this
#: corpus): inside these families the square wins, everywhere else the triangular
#: count wins. See rule 4 in the module docstring.
_FREEFORM_FAMILIES: frozenset[str] = frozenset({"slm_gsnet_square", "sim_calib_abba"})

#: Exposure lookup order, record keys first (first hit wins). ``exp_t`` is ms.
_EXPOSURE_RECORD_KEYS: tuple[str, ...] = ("exp_t", "exposure_ms", "exposure_time_ms")
_EXPOSURE_SIDECAR_KEYS: tuple[str, ...] = ("exposure_ms", "exposure_time_ms")
_N_MAX_SIDECAR_KEY = "n_max"

#: Bumped whenever the on-disk cache layout changes; a mismatch forces a rescan.
_CACHE_VERSION = 2

#: Exceptions that mean "this file is not a usable record dump". A corrupt or
#: truncated pickle raises a surprisingly wide set of them, so each is listed
#: explicitly -- a bare ``except:`` is a repo red line.
_UNREADABLE_PICKLE_ERRORS: tuple[type[BaseException], ...] = (
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


class PhaseSource(Enum):
    """Where a record's commanded SLM phase lives and in what units.

    The four values are mutually exclusive and cover all 11393 usable records of
    the measured corpus. ``_phase`` wins over ``_c`` whenever both are present:
    the ``slm_pib_*`` records carry a ``(1200, 1920) uint16`` panel *and* a
    ``_c`` of length 55, which is a valid triangular Zernike count (n_max=9).
    Without the panel taking precedence those records would be mislabelled
    ``ZERNIKE`` even though ``_c`` there is the PIB optimiser's axis-bucket
    configuration, not Zernike coefficients.
    """

    PANEL_RAD = "panel_rad"
    """``record["_phase"]`` is float radians (the writer already wrapped it to ``[0, 2pi)``)."""

    PANEL_GRAY = "panel_gray"
    """``record["_phase"]`` is uint16 SLM grayscale; needs ``gray / max_gray * 2pi``."""

    ZERNIKE = "zernike"
    """Reconstruct the panel phase from ``record["_c"]`` (Noll order, radians)."""

    FREEFORM = "freeform"
    """Reconstruct the panel phase from ``record["_c"]`` (``grid*grid``, radians per cell)."""


class ExclusionReason(Enum):
    """Why a record (or a whole file) did not make it into :attr:`HwCorpusIndex.records`.

    Per-record reasons are counted once per record; :attr:`UNREADABLE_PICKLE` is
    counted once per file.
    """

    UNREADABLE_PICKLE = "unreadable_pickle"
    """The ``.pkl`` could not be opened or unpickled (truncated dump, garbage bytes)."""

    NO_RECORDS = "no_records"
    """The payload held no records at all (``None`` from an aborted run, ``{}``, or an object without ``.history``)."""

    RECORD_NOT_DICT = "record_not_dict"
    """A payload entry was not a ``dict``-like record."""

    MISSING_IMAGE = "missing_image"
    """``record["_img"]`` was absent or not a 2-D ndarray."""

    MISSING_PHASE = "missing_phase"
    """No ``_phase`` and no usable ``_c``: the run commanded nothing (e.g. the 42 ``bench_stability`` drift records, or 6 of the 18 records of ``sim_calib_abba``)."""

    ODD_COEFFICIENT_LENGTH = "odd_coefficient_length"
    """``len(record["_c"])`` is neither a triangular Zernike count nor a perfect square."""


# ---------------------------------------------------------------------------
# Records and the index itself (no arrays, ever)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HwRecordRef:
    """Immutable pointer to one usable record, holding **no arrays**.

    Attributes:
        path: The ``.pkl`` the record lives in.
        position: 0-based position of the record inside the payload (sorted for a
            ``dict`` payload, insertion order for a ``Recorder.history`` payload).
        key: The payload ``dict`` key, or ``None`` for a ``Recorder`` payload,
            which has no integer keys.
        family: Directory-derived family label from :func:`family_of`.
        source: Which of the four phase representations this record uses.
        n_terms: ``len(record["_c"])`` when present, else ``0``.
        n_max: Zernike radial order, when known (Zernike sources only).
        freeform_grid: Grid side, when :attr:`source` is
            :attr:`PhaseSource.FREEFORM`.
        exposure_ms: Resolved camera exposure in milliseconds, or ``None``.
        sidecar: Read-only view of the merged sidecar JSON (``{}`` when absent).
            One proxy is shared by every record of a file, so this costs one small
            dict per file rather than one per record.
    """

    path: Path
    position: int
    key: int | None
    family: str
    source: PhaseSource
    n_terms: int
    n_max: int | None
    freeform_grid: int | None
    exposure_ms: float | None
    sidecar: Mapping[str, Any]


@dataclass(frozen=True)
class HwCorpusIndex:
    """Immutable index over the whole hardware debug corpus.

    Attributes:
        records: Every usable record, ordered by file (sorted) then position.
        excluded: Tally of dropped records/files keyed by
            :class:`ExclusionReason`.
        files_scanned: Number of ``.pkl`` files actually opened.
        files_usable: Number of those files that contributed at least one record.
    """

    records: tuple[HwRecordRef, ...]
    excluded: Mapping[ExclusionReason, int]
    files_scanned: int
    files_usable: int

    def __len__(self) -> int:
        """Return the number of usable records.

        Returns:
            ``len(self.records)``.
        """
        return len(self.records)

    @property
    def families(self) -> tuple[str, ...]:
        """Return the sorted, unique family labels present in the index.

        Returns:
            A tuple of family names; empty when the index holds no records.
        """
        return tuple(sorted({record.family for record in self.records}))

    def counts_by_family(self) -> Mapping[str, int]:
        """Return the record count per family.

        Returns:
            A read-only mapping family -> count, only for families present.
        """
        return MappingProxyType(
            dict(sorted(Counter(record.family for record in self.records).items()))
        )

    def counts_by_source(self) -> Mapping[PhaseSource, int]:
        """Return the record count per :class:`PhaseSource`.

        Returns:
            A read-only mapping source -> count, only for sources present.
        """
        # Sort by ``.value``: plain ``Enum`` members are not orderable, so sorting
        # the Counter items directly raises TypeError as soon as two sources appear.
        counts = Counter(record.source for record in self.records)
        return MappingProxyType(
            {source: counts[source] for source in sorted(counts, key=lambda s: s.value)}
        )

    def filter(
        self,
        *,
        families: Iterable[str] | None = None,
        sources: Iterable[PhaseSource | str] | None = None,
        require_exposure: bool = True,
    ) -> HwCorpusIndex:
        """Return a new index restricted to the given families / phase sources.

        Args:
            families: Keep only records whose :attr:`HwRecordRef.family` is in
                this iterable. ``None`` keeps every family.
            sources: Keep only records whose :attr:`HwRecordRef.source` is in this
                iterable. Plain strings are accepted and resolved through
                :class:`PhaseSource`, so ``["panel_gray"]`` works. ``None`` keeps
                every source.
            require_exposure: When ``True`` (default), drop records whose
                exposure could not be resolved (``exposure_ms is None``) -- a
                learning target needs a known exposure to normalise the CCD
                signal. Pass ``False`` to keep them.

        Returns:
            A new :class:`HwCorpusIndex` whose :attr:`~HwCorpusIndex.records` are
            a filtered subsequence of ``self.records``. The
            :attr:`~HwCorpusIndex.excluded` tally and both file counters are
            carried over unchanged: they describe the *scan*, not the selection,
            and re-deriving them would need the corpus again.
        """
        wanted_families = None if families is None else {str(f) for f in families}
        wanted_sources: set[PhaseSource] | None = None
        if sources is not None:
            wanted_sources = {
                source if isinstance(source, PhaseSource) else PhaseSource(source)
                for source in sources
            }
        kept = tuple(
            record
            for record in self.records
            if (wanted_families is None or record.family in wanted_families)
            and (wanted_sources is None or record.source in wanted_sources)
            and (not require_exposure or record.exposure_ms is not None)
        )
        return HwCorpusIndex(
            records=kept,
            excluded=self.excluded,
            files_scanned=self.files_scanned,
            files_usable=self.files_usable,
        )


# ---------------------------------------------------------------------------
# Family labelling
# ---------------------------------------------------------------------------
def family_of(path: Path) -> str:
    """Map a pickle path to a family label by longest-prefix match.

    The debug tree is one to two levels deep, so both
    ``path.parent.name`` and ``path.parent.parent.name`` are tried, the former
    first (a file sitting directly in a family directory, e.g.
    ``data/debug/bench_stability_<ts>/bench_stability_<ts>.pkl``, labels itself
    from its own parent). Within each candidate name the prefixes in
    :data:`_FAMILY_PREFIXES` are tried **in order**, and the order is
    load-bearing: ``slm_pib_online`` is listed before ``slm_pib`` because
    ``"slm_pib_online".startswith("slm_pib")``.

    Unrecognised directories (e.g. ``snr_sweep_20260930``) fall back to
    ``path.parent.name``, which for a two-level dump is the timestamp directory --
    still a stable, sortable grouping key.

    Args:
        path: Path of a ``.pkl`` (its ancestors are inspected; the file need not
            exist).

    Returns:
        The matched family label, or ``path.parent.name`` when nothing matches.
    """
    # NOTE: the grandparent MUST be `path.parent.parent.name`, not `parts[-2]` --
    # `parts[-2]` is always `path.parent.name`, so it would silently collapse to a
    # single candidate and every two-level dump would fall back to the timestamp
    # directory (e.g. "20260101_000000") instead of its family.
    candidates = [path.parent.name]
    grandparent = path.parent.parent.name
    if grandparent and grandparent != candidates[0]:
        candidates.append(grandparent)
    for name in candidates:
        for prefix in _FAMILY_PREFIXES:
            if name.startswith(prefix):
                return prefix
    return path.parent.name


# ---------------------------------------------------------------------------
# Sidecar JSON
# ---------------------------------------------------------------------------
def _sidecar_candidates(pkl_path: Path) -> list[Path]:
    """List sidecar candidates for one pickle, most specific first.

    Args:
        pkl_path: The ``.pkl`` whose sidecars are wanted.

    Returns:
        ``<stem>.json`` first (this is what the CLI runners write), then every
        ``summary_*.json`` in the same directory sorted by name, then
        ``meta.json``.
    """
    candidates = [pkl_path.with_suffix(".json")]
    try:
        candidates.extend(sorted(pkl_path.parent.glob("summary_*.json")))
    except OSError as exc:  # pragma: no cover - unreadable directory
        logger.debug("Could not list summary sidecars beside {}: {}", pkl_path, exc)
    candidates.append(pkl_path.parent / "meta.json")
    return candidates


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON sidecar as a plain ``dict``, or ``{}`` on any problem.

    An unreadable, truncated or non-object sidecar must never abort a scan: it
    only costs the ``n_max`` / ``exposure_ms`` hints for that one file, and the
    record itself is still indexable.

    Args:
        path: Candidate sidecar path.

    Returns:
        The parsed object, or ``{}`` when the file is missing, unreadable, not
        valid JSON, or not a JSON object.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.debug("Sidecar {} unreadable: {}", path, exc)
        return {}
    try:
        payload = json.loads(text)
    except ValueError as exc:
        logger.debug("Sidecar {} is not valid JSON: {}", path, exc)
        return {}
    if not isinstance(payload, dict):
        logger.debug(
            "Sidecar {} is a JSON {} not an object; ignoring", path, type(payload).__name__
        )
        return {}
    return payload


def _read_sidecar(pkl_path: Path) -> Mapping[str, Any]:
    """Resolve the merged sidecar object for one pickle, first hit wins.

    Args:
        pkl_path: The ``.pkl`` whose sidecar is wanted.

    Returns:
        A read-only view of the first sidecar found among
        ``<stem>.json`` -> ``summary_*.json`` -> ``meta.json``; ``{}`` when none
        exists or none is a JSON object. The same proxy instance is returned for
        every record of one file, which is why it is shared rather than rebuilt.
    """
    for candidate in _sidecar_candidates(pkl_path):
        if not candidate.is_file():
            continue
        payload = _load_json_object(candidate)
        if payload:
            logger.debug("Using sidecar {} for {}", candidate, pkl_path)
            return MappingProxyType(payload)
        # An existing-but-empty/unusable sidecar must not shadow a better one;
        # fall through to the next candidate.
    return MappingProxyType({})


# ---------------------------------------------------------------------------
# Per-record classification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Verdict:
    """Outcome of classifying one record.

    Exactly one of :attr:`source` / :attr:`reason` is set.
    """

    source: PhaseSource | None = None
    n_terms: int = 0
    n_max: int | None = None
    freeform_grid: int | None = None
    reason: ExclusionReason | None = None


def _as_float(value: Any) -> float | None:
    """Coerce a candidate exposure to a finite ``float``, or ``None``.

    Args:
        value: Raw value read from a record or a sidecar.

    Returns:
        The value as ``float`` when it is a real (non-boolean) finite number,
        else ``None``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _sidecar_n_max(sidecar: Mapping[str, Any]) -> int | None:
    """Read ``n_max`` from a sidecar as a non-negative ``int``.

    Args:
        sidecar: Merged sidecar mapping.

    Returns:
        The radial order, or ``None`` when absent or not a usable integer.
    """
    value = sidecar.get(_N_MAX_SIDECAR_KEY)
    if value is None or isinstance(value, bool):
        return None
    try:
        order = int(value)
    except (TypeError, ValueError):
        return None
    return order if order >= 0 else None


def _is_perfect_square(value: int) -> bool:
    """Return whether a positive integer is a perfect square.

    Args:
        value: Candidate length; must be positive.

    Returns:
        ``True`` when ``value == isqrt(value) ** 2``.
    """
    if value <= 0:
        return False
    root = math.isqrt(value)
    return root * root == value


def _coefficient_length(record: Mapping[str, Any]) -> int:
    """Return ``len(record["_c"])``, or ``0`` when it is absent/unsized.

    Args:
        record: One record mapping.

    Returns:
        The coefficient vector length, ``0`` when absent, ``None``-like for a
        0-d array (``len`` raises ``TypeError``).
    """
    coeffs = record.get(_COEFF_KEY)
    if coeffs is None:
        return 0
    try:
        return int(len(coeffs))
    except TypeError:
        return 0


def _classify(record: Any, sidecar: Mapping[str, Any], family: str) -> _Verdict:
    """Decide the :class:`PhaseSource` of one record, or why it is excluded.

    The whole point of this function is that it runs **per record**. The measured
    corpus contains a file (``sim_calib_abba_20261001_163743.pkl``) that is
    ``{NO_PHASE: 6, FREEFORM: 12}``, so a per-file label would drop or mislabel
    half of it.

    Args:
        record: One payload entry. Anything that is not a ``Mapping`` is rejected
            as :attr:`ExclusionReason.RECORD_NOT_DICT`.
        sidecar: Merged sidecar mapping for the containing file (may be empty).
        family: Family label of the containing file, from :func:`family_of`. Used
            only to settle the ``len(_c) == 36`` ambiguity -- see rule 4 in the
            module docstring and :data:`_FREEFORM_FAMILIES`.

    Returns:
        A :class:`_Verdict`: either a :attr:`PhaseSource` with its
        ``n_terms`` / ``n_max`` / ``freeform_grid``, or an
        :attr:`ExclusionReason` with ``source`` left ``None``.
    """
    if not isinstance(record, Mapping):
        return _Verdict(reason=ExclusionReason.RECORD_NOT_DICT)

    # No duck-typed `np.ndarray` import on purpose: this module must not depend on
    # numpy at all, and `hasattr(image, "shape")` is exactly the duck type needed.
    image = record.get(_IMAGE_KEY)
    image_shape = getattr(image, "shape", None)
    if not isinstance(image_shape, tuple) or len(image_shape) != 2:
        return _Verdict(reason=ExclusionReason.MISSING_IMAGE)

    n_terms = _coefficient_length(record)
    phase = record.get(_PHASE_KEY)
    phase_shape = getattr(phase, "shape", None)
    if isinstance(phase_shape, tuple) and len(phase_shape) >= 1 and int(phase_shape[0]) > 0:
        # _phase present: it IS the commanded panel, whatever _c also holds.
        # uint16 grayscale vs float radians is decided by dtype; the slm_pib_*
        # corpus is uint16 and model_in_loop_hw_{collect,sweep} is float32.
        phase_dtype = getattr(phase, "dtype", None)
        # unsigned / integer / bool dtype => raw grayscale levels; floating => radians.
        # slm_pib_* is uint16, model_in_loop_hw_{collect,sweep} is float32.
        is_gray = getattr(phase_dtype, "kind", None) in ("u", "i", "b")
        return _Verdict(
            source=PhaseSource.PANEL_GRAY if is_gray else PhaseSource.PANEL_RAD,
            n_terms=n_terms,
        )

    if n_terms <= 0:
        # No panel array and no coefficients: nothing was commanded. This is the
        # bench_stability (42) and sim_calib_abba (6 of 18) case.
        return _Verdict(n_terms=0, reason=ExclusionReason.MISSING_PHASE)

    # Rule 1: the sidecar states n_max and the length agrees -> Zernike.
    order = _sidecar_n_max(sidecar)
    if order is not None and n_terms == (order + 1) * (order + 2) // 2:
        return _Verdict(source=PhaseSource.ZERNIKE, n_terms=n_terms, n_max=order)

    is_square = _is_perfect_square(n_terms)
    # Rule 3 first for a length that is unambiguously triangular (15, 78, ...).
    # infer_n_max is the authority on triangularity and is never re-derived here.
    triangular_order: int | None
    try:
        triangular_order = infer_n_max(n_terms)
    except ValueError:
        triangular_order = None

    if is_square and (triangular_order is None or family in _FREEFORM_FAMILIES):
        # Rule 2/4: square wins -- either because it is unambiguously a freeform
        # grid, or because this family genuinely drives one.
        return _Verdict(
            source=PhaseSource.FREEFORM,
            n_terms=n_terms,
            freeform_grid=math.isqrt(n_terms),
        )
    if triangular_order is not None:
        return _Verdict(
            source=PhaseSource.ZERNIKE, n_terms=n_terms, n_max=triangular_order
        )
    # Rule 5.
    return _Verdict(n_terms=n_terms, reason=ExclusionReason.ODD_COEFFICIENT_LENGTH)


def _resolve_exposure(record: Any, sidecar: Mapping[str, Any]) -> float | None:
    """Resolve one record's camera exposure in ms, first hit wins.

    Order: ``exp_t`` -> ``exposure_ms`` -> ``exposure_time_ms`` (record), then
    ``exposure_ms`` -> ``exposure_time_ms`` (sidecar), then ``None``.
    ``exp_t`` is milliseconds -- confirmed by
    :func:`ao_shaping.drivers.ccd.common.get_camera_exposure_ms`, which reads the
    same quantity off the camera as ``exposure_time_ms``.

    Args:
        record: One record mapping (non-mappings simply resolve from the sidecar).
        sidecar: Merged sidecar mapping.

    Returns:
        Exposure in milliseconds, or ``None`` when no source carries one.
    """
    mapping: Mapping[str, Any] = record if isinstance(record, Mapping) else {}
    for key in _EXPOSURE_RECORD_KEYS:
        value = _as_float(mapping.get(key))
        if value is not None:
            return value
    for key in _EXPOSURE_SIDECAR_KEYS:
        value = _as_float(sidecar.get(key))
        if value is not None:
            return value
    return None


# ---------------------------------------------------------------------------
# Payload walking
# ---------------------------------------------------------------------------
def _dict_sort_key(item: tuple[Any, Any]) -> tuple[int, int, str]:
    """Build a total ordering over ``(key, value)`` pairs of a record dict.

    Integer keys sort numerically first (that is the measured corpus shape), then
    anything else sorts by its string form, so a stray string key cannot make
    :func:`sorted` raise.

    Args:
        item: One ``(key, record)`` pair.

    Returns:
        ``(rank, numeric_key, text_key)`` where ``rank`` is ``0`` for integers.
    """
    key = item[0]
    if isinstance(key, int) and not isinstance(key, bool):
        return (0, key, "")
    return (1, 0, str(key))


def _payload_records(payload: Any) -> list[tuple[int, int | None, Any]]:
    """Flatten a pickle payload into ``(position, key, record)`` triples.

    Two shapes are supported, matching the measured corpus:

    * ``dict[int, dict]`` -- the overwhelming majority. Sorted by key so the index
      is independent of filesystem/insertion order.
    * a pickled ``Recorder`` exposing ``.history: list[dict]`` (8 files under
      ``data/debug/slm_pib_online/``) -- kept in history order, ``key`` ``None``.

    Args:
        payload: The object returned by ``pickle.load``.

    Returns:
        The triples, or ``[]`` when the payload holds no records (``None`` from an
        aborted run, an empty dict, or an object without a ``history`` list).
    """
    if isinstance(payload, dict):
        ordered = sorted(payload.items(), key=_dict_sort_key)
        records: list[tuple[int, int | None, Any]] = []
        for position, (key, value) in enumerate(ordered):
            int_key = key if isinstance(key, int) and not isinstance(key, bool) else None
            records.append((position, int_key, value))
        return records
    history = getattr(payload, _RECORDER_HISTORY_ATTR, None)
    if isinstance(history, (list, tuple)):
        return [(position, None, value) for position, value in enumerate(history)]
    return []


def _tally(tally: dict[ExclusionReason, int], reason: ExclusionReason, count: int = 1) -> None:
    """Add ``count`` to ``tally[reason]``.

    Args:
        tally: Accumulator mutated in place.
        reason: Reason to increment.
        count: Increment, ``1`` by default.
    """
    tally[reason] = tally.get(reason, 0) + int(count)


@dataclass(frozen=True)
class _FileScan:
    """Per-file scan result."""

    refs: tuple[HwRecordRef, ...]
    excluded: Mapping[ExclusionReason, int]
    unreadable: bool


def _scan_file(pkl_path: Path) -> _FileScan:
    """Unpickle one file, classify every record, then drop the payload.

    MEMORY: this is the only place a pickle is deserialised. ``pickle.load`` is
    monolithic (no streaming), so transient peak RAM is ONE file -- up to 3.42 GB
    for the largest dump in the measured corpus, which cannot be avoided. What
    *can* be avoided is accumulation, so the payload is explicitly ``del``\\ eted
    here before the caller opens the next file; the returned
    :class:`HwRecordRef` tuple keeps no array at all.

    Args:
        pkl_path: The ``.pkl`` to scan.

    Returns:
        A :class:`_FileScan`. An unreadable pickle yields
        :attr:`_FileScan.unreadable` set and a single
        :attr:`ExclusionReason.UNREADABLE_PICKLE` -- never an exception. Skipping
        a bad file is mandatory here, not defensive: one stray artifact in
        ``data/`` otherwise costs the whole 65.88 GB corpus, which has already
        silently disabled all GSNet training in this repo once.
    """
    try:
        with open(pkl_path, "rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 - repo-internal debug dumps
    except _UNREADABLE_PICKLE_ERRORS as exc:
        logger.warning("Skipping unreadable debug pickle {}: {}", pkl_path, exc)
        return _FileScan(
            refs=(),
            excluded=MappingProxyType({ExclusionReason.UNREADABLE_PICKLE: 1}),
            unreadable=True,
        )

    sidecar = _read_sidecar(pkl_path)
    family = family_of(pkl_path)
    entries = _payload_records(payload)
    tally: dict[ExclusionReason, int] = {}
    refs: list[HwRecordRef] = []
    if not entries:
        _tally(tally, ExclusionReason.NO_RECORDS)
    for position, key, record in entries:
        verdict = _classify(record, sidecar, family)
        if verdict.source is None:
            reason = verdict.reason or ExclusionReason.RECORD_NOT_DICT
            _tally(tally, reason)
            continue
        refs.append(
            HwRecordRef(
                path=pkl_path,
                position=position,
                key=key,
                family=family,
                source=verdict.source,
                n_terms=verdict.n_terms,
                n_max=verdict.n_max,
                freeform_grid=verdict.freeform_grid,
                exposure_ms=_resolve_exposure(record, sidecar),
                sidecar=sidecar,
            )
        )

    del payload  # drop every array before the next file is opened
    logger.debug(
        "Scanned {}: {} records kept, {} excluded", pkl_path, len(refs), tally or "{}"
    )
    return _FileScan(
        refs=tuple(refs), excluded=MappingProxyType(tally), unreadable=False
    )


# ---------------------------------------------------------------------------
# JSON index cache
# ---------------------------------------------------------------------------
def _normalise_roots(roots: Sequence[str | Path] | str | Path) -> list[Path]:
    """Normalise the ``roots`` argument to an ordered list of paths.

    Args:
        roots: A single root or a sequence of roots.

    Returns:
        The roots as :class:`Path` objects, in the given order.
    """
    items = [roots] if isinstance(roots, (str, Path)) else list(roots)
    return [Path(item) for item in items]


def _roots_fingerprint(roots: Sequence[str | Path] | str | Path) -> list[str]:
    """Build the order-insensitive fingerprint of ``roots`` stored in the cache.

    Args:
        roots: A single root or a sequence of roots.

    Returns:
        Sorted unique string forms of the roots. Sorting (rather than keeping the
        given order) is right because the file list is deduplicated and sorted
        anyway, so ``[A, B]`` and ``[B, A]`` describe the same corpus.
    """
    return sorted({str(root) for root in _normalise_roots(roots)})


def _index_to_payload(
    index: HwCorpusIndex,
    roots: Sequence[str | Path] | str | Path,
    limit_files: int | None,
) -> dict[str, Any]:
    """Serialise an index (plus its provenance) to plain JSON types.

    ``Path``, the :class:`PhaseSource` value, ``n_max`` and ``key`` are round-
    tripped **explicitly** -- JSON is the cache format, so nothing may rely on
    ``pickle``.

    Args:
        index: The index to serialise.
        roots: The roots it was built with, recorded so a stale cache can be
            detected.
        limit_files: The file cap it was built with (``None`` = no cap).

    Returns:
        A JSON-serialisable ``dict``.
    """
    return {
        "version": _CACHE_VERSION,
        "roots": _roots_fingerprint(roots),
        "limit_files": None if limit_files is None else int(limit_files),
        "files_scanned": int(index.files_scanned),
        "files_usable": int(index.files_usable),
        "excluded": {reason.value: int(count) for reason, count in index.excluded.items()},
        "records": [
            {
                "path": str(record.path),
                "position": int(record.position),
                "key": None if record.key is None else int(record.key),
                "family": record.family,
                "source": record.source.value,
                "n_terms": int(record.n_terms),
                "n_max": None if record.n_max is None else int(record.n_max),
                "freeform_grid": (
                    None if record.freeform_grid is None else int(record.freeform_grid)
                ),
                "exposure_ms": (
                    None if record.exposure_ms is None else float(record.exposure_ms)
                ),
                "sidecar": dict(record.sidecar),
            }
            for record in index.records
        ],
    }


def _payload_to_index(
    payload: Any, roots: Sequence[str | Path] | str | Path, limit_files: int | None
) -> HwCorpusIndex | None:
    """Rebuild an index from a cache payload, or return ``None`` if it is stale.

    A cache is rejected -- with a warning, so the caller knows it is about to pay
    for a full 65.88 GB scan -- when its ``version``, ``roots`` or
    ``limit_files`` disagree with what was requested. This is what stops a stale
    cache from silently shadowing a different corpus.

    Args:
        payload: The parsed JSON of the cache file.
        roots: The roots requested now.
        limit_files: The file cap requested now (``None`` = no cap).

    Returns:
        The reconstructed :class:`HwCorpusIndex`, or ``None`` when the payload is
        unusable or was built from different inputs.
    """
    if not isinstance(payload, dict):
        logger.warning("Ignoring index cache: payload is not a JSON object")
        return None
    if payload.get("version") != _CACHE_VERSION:
        logger.warning(
            "Ignoring index cache built by version {} (this build writes {})",
            payload.get("version"),
            _CACHE_VERSION,
        )
        return None
    if list(payload.get("roots") or []) != _roots_fingerprint(roots):
        logger.warning(
            "Ignoring stale index cache: built for roots {} but roots {} requested",
            payload.get("roots"),
            _roots_fingerprint(roots),
        )
        return None
    cached_limit = payload.get("limit_files")
    if cached_limit != limit_files:
        logger.warning(
            "Ignoring stale index cache: built with limit_files={} but {} requested",
            cached_limit,
            limit_files,
        )
        return None

    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        logger.warning("Ignoring index cache: 'records' is not a list")
        return None

    records: list[HwRecordRef] = []
    try:
        for item in raw_records:
            raw_sidecar = item.get("sidecar")
            records.append(
                HwRecordRef(
                    path=Path(str(item["path"])),
                    position=int(item["position"]),
                    key=None if item.get("key") is None else int(item["key"]),
                    family=str(item["family"]),
                    source=PhaseSource(str(item["source"])),
                    n_terms=int(item.get("n_terms") or 0),
                    n_max=None if item.get("n_max") is None else int(item["n_max"]),
                    freeform_grid=(
                        None
                        if item.get("freeform_grid") is None
                        else int(item["freeform_grid"])
                    ),
                    exposure_ms=(
                        None
                        if item.get("exposure_ms") is None
                        else float(item["exposure_ms"])
                    ),
                    sidecar=MappingProxyType(dict(raw_sidecar or {})),
                )
            )
        excluded = {
            ExclusionReason(str(reason)): int(count)
            for reason, count in dict(payload.get("excluded") or {}).items()
        }
        return HwCorpusIndex(
            records=tuple(records),
            excluded=MappingProxyType(excluded),
            files_scanned=int(payload.get("files_scanned") or 0),
            files_usable=int(payload.get("files_usable") or 0),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Ignoring malformed index cache: {}", exc)
        return None


def _read_index_cache(cache_path: Path, roots: Sequence[str | Path] | str | Path, limit_files: int | None) -> HwCorpusIndex | None:
    """Load a cached index for exactly these ``roots`` / ``limit_files``.

    Args:
        cache_path: JSON cache file.
        roots: The roots requested now.
        limit_files: The file cap requested now.

    Returns:
        The cached index, or ``None`` when the file is absent, unreadable,
        malformed, or stale.
    """
    if not cache_path.is_file():
        return None
    try:
        raw = cache_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read index cache {}: {}", cache_path, exc)
        return None
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        logger.warning("Index cache {} is not valid JSON: {}", cache_path, exc)
        return None
    return _payload_to_index(payload, roots, limit_files)


def _write_index_cache(
    cache_path: Path, index: HwCorpusIndex, roots: Sequence[str | Path] | str | Path, limit_files: int | None
) -> None:
    """Write ``index`` plus its provenance to ``cache_path`` as JSON.

    A write failure (read-only checkout, no space) is a warning, never an error:
    the freshly built index is still perfectly usable in memory.

    Args:
        cache_path: Destination JSON file.
        index: Index to persist.
        roots: The roots it was built with.
        limit_files: The cap it was built with (``None`` = no cap).
    """
    payload = _index_to_payload(index, roots, limit_files)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Could not write index cache {}: {}", cache_path, exc)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _discover_pickles(roots: Sequence[str | Path] | str | Path) -> list[Path]:
    """Collect every ``*.pkl`` under ``roots``, recursively, sorted.

    RECURSIVE discovery is mandatory: the measured tree is one to two levels deep
    (``data/debug/slm_pib_x/<ts>/<prefix>_<ts>_<ts>.pkl``), so a non-recursive
    ``glob("*.pkl")`` finds **zero** of the 265 pickles.

    Args:
        roots: A single root or a sequence of roots. A root may be a directory, a
            ``.pkl`` file, or a glob pattern.

    Returns:
        Sorted, de-duplicated absolute paths.
    """
    unique: set[Path] = set()
    for root in _normalise_roots(roots):
        if any(ch in str(root) for ch in "*?["):
            matched = sorted(Path(root).parent.glob(root.name))
        else:
            matched = [root]
        for item in matched:
            if item.is_dir():
                unique.update(candidate.resolve() for candidate in item.rglob("*.pkl") if candidate.is_file())
            elif item.is_file() and item.suffix.lower() == ".pkl":
                unique.add(item.resolve())
    return sorted(unique)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_hw_index(
    roots: Sequence[str | Path] | str | Path = DEFAULT_ROOTS,
    *,
    limit_files: int | None = None,
    index_cache: str | Path | None = None,
    progress_every: int = 10,
) -> HwCorpusIndex:
    """Scan every ``*.pkl`` under ``roots`` (recursively) and classify every record.

    MEMORY: exactly one pickle is resident at a time. ``pickle.load`` cannot
    stream, so peak transient RAM is ONE file (up to **3.42 GB**, the largest of
    the 265 measured pickles in a **65.88 GB** tree); the payload is ``del``\\ ed
    before the next file is opened, so the high-water mark never accumulates. The
    returned index holds **no arrays** -- only paths, ints, enums, floats and one
    small sidecar dict per file.

    ROBUSTNESS: a file that cannot be opened/unpickled, or that holds no records,
    is counted and skipped. A hard failure here would silently disable all
    downstream training (3 files in the measured corpus unpickle to ``None``
    because their run aborted, and a truncated dump raises
    ``EOFError``/``UnpicklingError``).

    Args:
        roots: A single root or a sequence of roots to scan, each a directory, a
            ``.pkl`` file, or a glob pattern. Defaults to
            :data:`DEFAULT_ROOTS` (``data/debug``), the whole debug tree.
        limit_files: Cap on the number of ``.pkl`` files, applied **after**
            sorting so smoke runs are deterministic. ``None`` (default) = no cap.
        index_cache: Optional JSON cache. When the file exists **and** was built
            from the same ``roots`` and ``limit_files``, the expensive scan is
            skipped entirely; otherwise the fresh index is written there.
        progress_every: Log a progress line every N files. ``<= 0`` disables it.

    Returns:
        An :class:`HwCorpusIndex`. Records are ordered by file (sorted path) then
        by position within the payload, so the index is reproducible.
    """
    cache_path = Path(index_cache) if index_cache is not None else None
    if cache_path is not None:
        cached = _read_index_cache(cache_path, roots, limit_files)
        if cached is not None:
            logger.info(
                "Loaded {} cached records from {} ({} files scanned, scan skipped)",
                len(cached.records),
                cache_path,
                cached.files_scanned,
            )
            return cached

    files = _discover_pickles(roots)
    if limit_files is not None:
        files = files[: max(0, int(limit_files))]
    if not files:
        logger.warning("No debug pickles found under roots {}", _normalise_roots(roots))

    records: list[HwRecordRef] = []
    tally: dict[ExclusionReason, int] = {}
    files_usable = 0
    unreadable = 0
    stride = max(0, int(progress_every))
    for index, path in enumerate(files, start=1):
        scan = _scan_file(path)
        records.extend(scan.refs)
        for reason, count in scan.excluded.items():
            _tally(tally, reason, count)
        if scan.unreadable:
            unreadable += 1
        if scan.refs:
            files_usable += 1
        if stride and index % stride == 0:
            logger.info(
                "Indexed {}/{} files, {} records so far", index, len(files), len(records)
            )

    hw_index = HwCorpusIndex(
        records=tuple(records),
        excluded=MappingProxyType(tally),
        files_scanned=len(files),
        files_usable=files_usable,
    )
    logger.info(
        "Indexed {} records from {} pickles ({} usable, {} unreadable); excluded {}",
        len(hw_index.records),
        hw_index.files_scanned,
        hw_index.files_usable,
        unreadable,
        {reason.value: count for reason, count in tally.items()} or "{}",
    )
    if cache_path is not None:
        _write_index_cache(cache_path, hw_index, roots, limit_files)
    return hw_index
