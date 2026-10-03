"""Hardware debug dumps as a PyTorch Dataset: SLM phase + exposure -> camera image.

This package turns every hardware debug dump under
:data:`~ml.hwdataset.index.DEFAULT_ROOTS` (``data/debug``) into a
``torch.utils.data.Dataset``. Each sample pairs **the SLM phase that was
commanded** with **the camera exposure at which it was measured**, and carries
**the measured camera frame** as the target:

======================  ==================  ==============================================
sample key              shape / dtype       meaning
======================  ==================  ==============================================
``phase_cos``           ``(1, g, g) f32``   Re of the coherent block mean of the panel
``phase_sin``           ``(1, g, g) f32``   Im of the coherent block mean
``image``               ``(1, g, g) f32``   far-field frame in ``[0, 1]``, absolute
``exposure_log10``      ``(1,) f32``        standardised ``log10`` exposure in ms
``contrast``            ``(g, g) f32``      ``hypot(phase_cos, phase_sin)``, coherence
``exposure_ms``         ``float``           the raw exposure, for logging
``source``/``family``   ``str``             phase provenance
``fov_px``              ``int | None``      source-frame side in camera pixels
======================  ==================  ==============================================

Why ``(cos, sin)`` and not ``(angle, contrast)``: the recorded phases are wrapped
to ``[0, 2*pi)``, so the panel phase must be reduced with a **coherent**
(complex-phasor) block mean -- an arithmetic mean of a wrapped phase collapses to
``~pi`` and destroys the structure. The block mean of ``exp(1j*phi)`` is returned
as its real and imaginary parts rather than as an ``arctan2``, because
``arctan2`` has a branch cut at ``+/-pi`` that a convolutional network would have
to learn around. See :mod:`ml.hwdataset.transforms`.

Two caveats are surfaced rather than hidden, because they are real properties of
this corpus rather than defects:

* **The families have different fields of view** (64 px ``region=32`` windows,
  248/250/320/400 px camera windows, and the full 1944 px sensor), so one output
  grid means a different physical angular scale per family. Every item therefore
  carries ``fov_px`` so a consumer can filter or train per family.
* **The exposure spans 0.1 ms to 80 ms**, 2.9 orders of magnitude, so it is fed as
  a standardised ``log10`` scalar rather than a raw linear one.

Module map
----------
:mod:`ml.hwdataset.index`
    Corpus discovery and per-record indexing (:func:`build_hw_index`).
:mod:`ml.hwdataset.transforms`
    Pure array-to-array transforms; no index, file or device knowledge.
:mod:`ml.hwdataset.records`
    One ``HwRecordRef`` -> :class:`HwSample` materialiser, shared by the Dataset
    and the on-disk cache so both paths are bit-identical.
:mod:`ml.hwdataset.dataset`
    The ``Dataset``, a file-grouped sampler, and the DataLoader factories.
:mod:`ml.hwdataset.cache`
    Optional memory-mappable cache of the derived grid-sized arrays.
:mod:`ml.hwdataset.inspect`
    ``python -m ml.hwdataset.inspect`` corpus statistics / smoke run.

Layering note
-------------
This package imports the pure helpers of :mod:`ao_shaping.runners.gsnet_offline`
(``infer_n_max``, ``reconstruct_pupil_phase_rad``, ``pupil_phase_to_grid``,
``farfield_to_grid``) rather than reimplementing them: this repository has been
 bitten repeatedly by duplicated Zernike and crop maths drifting between copies.
That module imports only ``ao_shaping.utils.*`` and
``ao_shaping/runners/__init__.py`` uses a PEP-562 lazy ``__getattr__``, so no
import cycle forms even though ``ao_shaping.runners.gsnet_dataset`` imports
``ml.gsnet.dataset`` in the other direction.
"""

from __future__ import annotations

from ml.hwdataset.dataset import (
    DEFAULT_PREFETCH_FACTOR,
    FileGroupedSampler,
    HwPhaseImageDataset,
    build_hw_dataloader,
    create_hw_dataloaders,
)
from ml.hwdataset.index import (
    DEFAULT_ROOTS,
    ExclusionReason,
    HwCorpusIndex,
    HwRecordRef,
    PhaseSource,
    build_hw_index,
    family_of,
)
from ml.hwdataset.records import (
    HwRecordError,
    HwSample,
    Materialiser,
    MaterialiserConfig,
    PayloadStore,
)
from ml.hwdataset.transforms import (
    DEFAULT_PANEL_CENTER,
    DEFAULT_PANEL_RADIUS,
    DEFAULT_SLM_MAX_GRAY,
    coherent_block_mean,
    crop_panel_roi,
    farfield_frame_to_grid,
    freeform_grid_to_panel,
    grayscale_to_phase_rad,
    phase_to_grid,
    zernike_coeffs_to_panel,
)

# Imported last: cache.py imports index and records, so it must see them already
# bound in this namespace.
from ml.hwdataset.cache import (
    CACHE_FORMAT_VERSION,
    HwCacheError,
    HwCachedSource,
    cache_dir_for,
    cache_is_current,
    load_hw_cache,
    prepare_hw_cache,
)

__all__ = [
    # index
    "DEFAULT_ROOTS",
    "PhaseSource",
    "ExclusionReason",
    "HwRecordRef",
    "HwCorpusIndex",
    "build_hw_index",
    "family_of",
    # transforms
    "DEFAULT_SLM_MAX_GRAY",
    "DEFAULT_PANEL_CENTER",
    "DEFAULT_PANEL_RADIUS",
    "grayscale_to_phase_rad",
    "crop_panel_roi",
    "coherent_block_mean",
    "phase_to_grid",
    "zernike_coeffs_to_panel",
    "freeform_grid_to_panel",
    "farfield_frame_to_grid",
    # records
    "HwRecordError",
    "HwSample",
    "MaterialiserConfig",
    "PayloadStore",
    "Materialiser",
    # dataset
    "DEFAULT_PREFETCH_FACTOR",
    "FileGroupedSampler",
    "HwPhaseImageDataset",
    "build_hw_dataloader",
    "create_hw_dataloaders",
    # cache
    "CACHE_FORMAT_VERSION",
    "HwCacheError",
    "HwCachedSource",
    "cache_dir_for",
    "cache_is_current",
    "load_hw_cache",
    "prepare_hw_cache",
]
