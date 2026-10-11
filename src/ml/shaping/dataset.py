"""Batch assembly for the inverse-shaping trainer.

This module adapts :mod:`ml.hwdataset`'s sample dicts into the tensors the
``ml.shaping`` network consumes. It owns four responsibilities:

1. :func:`shaping_collate` -- a ``collate_fn`` that never lets a Python
   ``None`` reach :func:`torch.utils.data.default_collate` (which raises
   ``TypeError``). ``fov_px`` is ``None`` for roughly a fifth of the corpus and
   ``exposure_ms`` can be missing too; both are replaced by the sentinels
   :data:`ml.shaping.config.UNKNOWN_FOV` / ``UNKNOWN_EXPOSURE`` so a mixed
   batch stays collatable and "unknown" stays distinguishable from a real
   value (the sentinels are negative; real exposures are strictly positive).
2. :func:`build_shaping_loaders` -- a thin wrapper over
   :func:`ml.hwdataset.dataset.create_hw_dataloaders` that swaps in
   ``shaping_collate`` while preserving the file-level split and the train
   loader's :class:`~ml.hwdataset.dataset.FileGroupedSampler` exactly.
3. :func:`network_input` -- assembles the ``(B, in_channels, g, g)`` input
   tensor.
4. :func:`phase_target` -- assembles the ``(B, 2, g, g)`` ground-truth phasor
   from ``phase_cos`` / ``phase_sin``.

Why exposure is an input, and why the image is NOT peak-normalised
------------------------------------------------------------------
The corpus's ``image`` keeps the **absolute** brightness that encodes exposure
(``image_mode="abs255"``). Exposure is an *input* covariate here -- 9 values
spanning 0.1-80 ms, i.e. 2.9 orders of magnitude -- so
:func:`network_input` feeds the raw absolute-intensity frame alongside the
standardised ``exposure_log10`` broadcast as a second channel.
Peak-normalising the image would erase exactly the information
``exposure_log10`` is meant to carry.

Why the phase target is a phasor, never an angle
------------------------------------------------
The corpus stores the pupil phase as the phasor pair ``(phase_cos,
phase_sin)`` -- the real and imaginary parts of ``exp(i * phase)``. The panels
are wrapped to ``[0, 2*pi)``, so recovering an angle from the pair would
reintroduce a branch cut the network would have to learn; :func:`phase_target`
therefore concatenates the pair as-is.

Why ``fov_px=None`` must stay unknown
-------------------------------------
``fov_px`` is ``None`` for roughly a fifth of the corpus and there is no
calibrated ``far_field_padding`` for it. Such a record contributes to the
supervised term but is excluded from the physics term (see
:mod:`ml.shaping.physics`); the sentinel keeps that distinction visible in the
batch instead of collapsing it into a fabricated value.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.utils.data import DataLoader, default_collate

from ml.hwdataset.dataset import (
    DEFAULT_ROOTS,
    FileGroupedSampler,
    create_hw_dataloaders,
)
from ml.hwdataset.index import HwCorpusIndex, HwRecordRef, build_hw_index
from ml.hwdataset.records import MaterialiserConfig
from ml.shaping.config import UNKNOWN_EXPOSURE, UNKNOWN_FOV

__all__ = [
    "FamilyBalancedGroupedSampler",
    "build_shaping_loaders",
    "corpus_summary",
    "family_file_counts",
    "network_input",
    "phase_target",
    "shaping_collate",
]


class FamilyBalancedGroupedSampler(FileGroupedSampler):
    """Per-family balanced variant of
    :class:`~ml.hwdataset.dataset.FileGroupedSampler`.

    ==============================  Why this exists  ==============================

    ``report/hwdataset_corpus/report.md`` §11 lists, as one of only four
    mitigations it judges available, "family / file-level stratified sampling and
    weighting (up-sample the small families)", because the corpus is imbalanced
    by roughly **350x** (``slm_pib`` holds 7,866 records across 53 files;
    ``slm_gsnet_square`` holds 114 across 2). That imbalance is not theoretical:
    a measured run of this module evaluated 27 files that were 18 ``slm_pib``,
    1 ``slm_pib_online`` and 1 ``slm_gsnet_square``, so any pooled number was
    effectively a ``slm_pib`` number.

    ==============================  Why whole FILES  ==============================

    Balancing is applied at the **file** level, never per record. Each debug
    pickle is read whole, and the parent class keeps one file's records
    contiguous precisely so the 1-entry payload LRU pays off -- measured as the
    difference between ~2 minutes and ~10 hours per epoch. A per-record weighted
    sampler would interleave files and destroy that. So this class draws whole
    file-groups with replacement according to a per-family weight and emits each
    one's records in one contiguous run.

    ==============================  The weight  ==============================

    ``w_f ∝ n_f ** (1 - balance_alpha)``, normalised over families, where
    ``n_f`` is the number of **files** in family ``f``. So:

    * ``alpha = 0`` -> ``w_f ∝ n_f`` -> exactly the corpus proportions (the
      parent class' behaviour);
    * ``alpha = 1`` -> ``w_f ∝ 1`` -> every family contributes the same
      number of files;
    * ``alpha = 0.5`` (default) -> the standard sqrt-inverse-frequency
      compromise.

    File counts rather than record counts are used because a file's record count
    is a poor proxy for its importance: §2 of the corpus report measures record
    counts per file spanning 4 to 733, driven by pickle size rather than by how
    much distinct optical content the file holds.

    ==============================  The honesty guard  ==============================

    ``max_file_repeats`` caps how often one file may appear in a single epoch.
    Without it, balancing a 350x-imbalanced corpus would replay the smallest
    family's one or two files hundreds of times, converting a *sampling* bias
    into an *overfitting* one. The corpus report is explicit that weighting
    mitigates imbalance and does not replace missing data; this cap is what
    keeps that true in the code rather than only in the prose.
    """

    def __init__(
        self,
        records: Sequence[HwRecordRef],
        *,
        num_samples: int | None = None,
        seed: int = 0,
        balance_alpha: float = 0.5,
        max_file_repeats: int = 8,
    ) -> None:
        """Build the sampler and precompute the per-family file weights.

        Args:
            records: The records to sample over, in the order the parent class
                emits positions into.
            num_samples: Epoch length, forwarded to the parent class.
            seed: Seed for the internal generator.
            balance_alpha: Balancing strength in ``[0, 1]``; see the class
                docstring.
            max_file_repeats: Cap on a single file's repetitions per epoch.

        Raises:
            ValueError: If ``balance_alpha`` is outside ``[0, 1]``, if
                ``max_file_repeats`` is below 1, or if a record carries no
                family label.
        """
        super().__init__(records, num_samples=num_samples, seed=seed)
        alpha = float(balance_alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"balance_alpha must be in [0, 1], got {balance_alpha!r}")
        if int(max_file_repeats) < 1:
            raise ValueError(f"max_file_repeats must be >= 1, got {max_file_repeats}")
        self.balance_alpha = alpha
        self.max_file_repeats = int(max_file_repeats)

        # Family label per file-group, taken from the group's first record: every
        # record of one pickle shares a family by construction.
        self._group_family: tuple[str, ...] = tuple(
            str(self._records[group[0]].family) for group in self._groups
        )
        self._files_by_family: dict[str, list[int]] = {}
        for group_index, family in enumerate(self._group_family):
            self._files_by_family.setdefault(family, []).append(group_index)

    @property
    def families(self) -> tuple[tuple[str, int], ...]:
        """Families present, with the number of files each contributes.

        Returns:
            A ``(family, n_files)`` tuple sorted by family name.
        """
        return tuple(
            (family, len(files))
            for family, files in sorted(self._files_by_family.items())
        )

    def family_weights(self) -> dict[str, float]:
        """Normalised per-family file weights for this epoch.

        Returns:
            ``family -> weight``, summing to 1. At ``balance_alpha == 0`` these
            are the corpus file proportions; at ``1`` they are uniform.
        """
        counts = {f: len(v) for f, v in self._files_by_family.items()}
        total = sum(counts.values())
        if total == 0:
            return {}
        raw = {
            family: (count ** (1.0 - self.balance_alpha))
            for family, count in counts.items()
        }
        norm = sum(raw.values())
        return {family: value / norm for family, value in raw.items()}

    def _one_cycle(self) -> list[int]:
        """Build one epoch: draw whole file-groups, then emit their records.

        Returns:
            Record positions with each drawn file's members contiguous. The
            cycle length is the number of drawn groups' records summed, so a
            balanced cycle is not the same length as the corpus; the parent's
            :meth:`_iterate` handles repetition and trimming against it.
        """
        if self.balance_alpha == 0.0 or not self._files_by_family:
            return super()._one_cycle()

        total_groups = len(self._groups)
        weights = self.family_weights()
        drawn: list[int] = []
        shortfall = 0

        for family, weight in weights.items():
            files = self._files_by_family[family]
            target = int(round(total_groups * weight))
            if target <= 0:
                continue
            budget = {index: self.max_file_repeats for index in files}
            picked: list[int] = []
            # Repeated uniform draws with a per-file cap. The loop terminates
            # because `budget` strictly decreases on every accepted pick.
            while len(picked) < target and any(v > 0 for v in budget.values()):
                for index in torch.randperm(
                    len(files), generator=self._generator
                ).tolist():
                    if len(picked) >= target:
                        break
                    if budget[files[index]] > 0:
                        picked.append(files[index])
                        budget[files[index]] -= 1
            if len(picked) < target:
                shortfall += target - len(picked)
            drawn.extend(picked)

        if shortfall:
            logger.warning(
                "family balancing could not place {} of {} file slots: "
                "max_file_repeats={} is too low for this family imbalance. "
                "Raising it trades sampling bias for overfitting; the corpus "
                "report is explicit that weighting cannot replace missing data.",
                shortfall,
                total_groups,
                self.max_file_repeats,
            )

        # Shuffle the group order so a family is not visited as one block, then
        # emit each group's members shuffled but contiguous.
        order = torch.randperm(len(drawn), generator=self._generator).tolist()
        cycle: list[int] = []
        for slot in order:
            group = self._groups[drawn[slot]]
            permutation = torch.randperm(
                len(group), generator=self._generator
            ).tolist()
            cycle.extend(group[i] for i in permutation)
        return cycle


def shaping_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of ``ml.hwdataset`` sample dicts into one batch dict.

    ``torch.utils.data.default_collate`` raises ``TypeError`` on a Python
    ``None`` inside a batch, and two sample fields are legitimately ``None``
    for part of the corpus: ``fov_px`` (no derivable camera window) and
    ``exposure_ms`` (no usable exposure). Both are replaced by the sentinels
    from :mod:`ml.shaping.config` before collation, so a mixed batch stays
    collatable and the downstream code can tell "unknown" apart from a real
    value.

    Args:
        samples: List of sample dicts as produced by
            :class:`~ml.hwdataset.dataset.HwPhaseImageDataset.__getitem__`.

    Returns:
        A batch dict with the same keys. Tensor fields are stacked by
        ``default_collate``; ``fov_px`` is an ``int64`` tensor with
        ``UNKNOWN_FOV`` in place of ``None``; ``exposure_ms`` is a ``float64``
        tensor with ``UNKNOWN_EXPOSURE`` in place of ``None``; string fields
        become lists of ``str``.
    """
    cleaned: list[dict[str, Any]] = []
    for sample in samples:
        item = dict(sample)
        if item.get("fov_px") is None:
            item["fov_px"] = UNKNOWN_FOV
        if item.get("exposure_ms") is None:
            item["exposure_ms"] = UNKNOWN_EXPOSURE
        cleaned.append(item)
    return default_collate(cleaned)


def _with_shaping_collate(loader: DataLoader) -> DataLoader:
    """Rebuild a ``DataLoader`` with :func:`shaping_collate` as its collator.

    The dataset, sampler, batch size, worker count, pinning and drop-last
    behaviour are carried over verbatim so the file-level split and the train
    loader's :class:`~ml.hwdataset.dataset.FileGroupedSampler` are preserved
    exactly. ``prefetch_factor`` is intentionally not carried over: torch's
    default matches the corpus loader's when workers exist, and the attribute
    is absent on worker-less loaders.

    Args:
        loader: The loader to rebuild.

    Returns:
        A new ``DataLoader`` over the same dataset/sampler with
        ``collate_fn=shaping_collate``.
    """
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=loader.sampler,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        collate_fn=shaping_collate,
    )


def family_file_counts(records: Sequence[HwRecordRef]) -> dict[str, int]:
    """Count distinct pickle files per family.

    Args:
        records: Records to summarise.

    Returns:
        ``family -> n_files``. Families with a single file are the ones that
        make balancing dangerous, so this is the diagnostic that says whether
        :class:`FamilyBalancedGroupedSampler` will be repeating a lone file.
    """
    seen: dict[str, set[Path]] = {}
    for record in records:
        seen.setdefault(str(record.family), set()).add(record.path)
    return {family: len(paths) for family, paths in sorted(seen.items())}


def corpus_summary(loader: Any) -> dict[str, Any]:
    """Describe the corpus a loader actually draws from.

    ==============================  Why this exists  ==============================

    ``report/hwdataset_corpus/report.md`` §1 opens with a warning that the
    shipped index cache is stale -- it described 265 pickles while 468 were on
    disk -- and concludes that "any conclusion drawn from that cache covers only
    those records". That is not a hypothetical here. During development the
    cache was **deleted** and ~300 pickles removed, and a run launched against
    the stale cache happily reported 11,393 records from 261 pickles; a rebuild
    of the same directory 35 seconds later produced **14,335 records from 147
    pickles across 27 families**, with two whole families gone and ``slm_pib``
    risen from 69% to 88.8% of the corpus.

    A result that does not record which corpus produced it is therefore
    unattributable. Every run and every evaluation embeds this summary.

    Args:
        loader: A shaping loader (or any ``DataLoader`` over a
            ``Subset`` of :class:`~ml.hwdataset.dataset.HwPhaseImageDataset`).

    Returns:
        A JSON-safe dict with ``n_records``, ``n_files``, ``n_families``,
        ``families`` (``family -> n_records``), ``sources`` and
        ``dominant_family_share``. Everything is read off the retained index --
        no pickle is opened and no rescan is triggered.

        ``fov_px`` is deliberately **absent**: it lives on the materialised
        :class:`~ml.hwdataset.records.HwSample`, not on
        :class:`~ml.hwdataset.index.HwRecordRef`, so reporting it here would mean
        either materialising the corpus or emitting an all-``unknown`` histogram.
        Both would be misleading, and the forward bank already reports per-FOV
        exclusions explicitly at evaluation time.
    """
    subset = getattr(loader, "dataset", None)
    inner = getattr(subset, "dataset", subset)
    # Annotated rather than left to inference: `getattr(..., ()) or ()` otherwise
    # narrows to `list[Never]` and every attribute read below is flagged.
    raw_records: Sequence[HwRecordRef] = getattr(inner, "records", ()) or ()
    records = list(raw_records)
    indices: Sequence[int] | None = getattr(subset, "indices", None)
    if indices is not None and records:
        records = [records[i] for i in indices]

    families: dict[str, int] = {}
    sources: dict[str, int] = {}
    for record in records:
        families[str(record.family)] = families.get(str(record.family), 0) + 1
        key = str(getattr(record.source, "value", record.source))
        sources[key] = sources.get(key, 0) + 1

    total = len(records)
    dominant = max(families.values()) / total if total else 0.0
    return {
        "n_records": total,
        "n_files": len({str(r.path) for r in records}),
        "n_families": len(families),
        "families": dict(sorted(families.items(), key=lambda kv: -kv[1])),
        "sources": dict(sorted(sources.items(), key=lambda kv: -kv[1])),
        "dominant_family_share": round(dominant, 4),
    }


def build_shaping_loaders(
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
    families: Sequence[str] | None = None,
    sources: Sequence[str] | None = None,
    balance_alpha: float = 0.0,
    max_file_repeats: int = 8,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build seeded train/val/test loaders that collate with :func:`shaping_collate`.

    Thin wrapper over :func:`ml.hwdataset.dataset.create_hw_dataloaders`: the
    file-level split, the train loader's
    :class:`~ml.hwdataset.dataset.FileGroupedSampler` and every other loader
    setting are produced by the corpus function unchanged; each returned
    loader is then rebuilt with ``collate_fn=shaping_collate`` so a ``None``
    ``fov_px`` / ``exposure_ms`` never reaches ``default_collate``.

    Args:
        roots: Corpus root(s) handed to
            :func:`~ml.hwdataset.index.build_hw_index`.
        config: Materialisation config.
        batch_size: Samples per batch.
        train_split: Fraction of **files** for training.
        val_split: Fraction of **files** for validation. The remainder is the
            test split.
        num_workers: Worker processes per loader.
        seed: Seed of the split generator and the grouped sampler.
        cache_size: Payload LRU size.
        limit_files: Cap on the number of scanned pickles (smoke runs).
        index_cache: Optional JSON index cache path.
        require_exposure: Drop records with an unknown exposure.
        use_cache: Read from the on-disk ``.hw_cache`` directories when
            current.
        cache_root: Root for the cache directories, or ``None`` for the
            sibling ``.hw_cache`` directory next to each pickle.
        families: Keep only these corpus families, or ``None`` for all. The
            corpus report's §11 recommends per-family handling because family
            and phase representation are perfectly collinear in this corpus.
        sources: Keep only these phase sources, or ``None`` for all.
        balance_alpha: Per-family balancing strength for the train sampler; see
            :class:`FamilyBalancedGroupedSampler`. ``0`` (the default here) keeps
            the corpus proportions and is a plain
            :class:`~ml.hwdataset.dataset.FileGroupedSampler`.
        max_file_repeats: Cap on one file's repetitions per balanced epoch.

    Returns:
        ``(train_loader, val_loader, test_loader)``, each collating with
        :func:`shaping_collate`. The train loader keeps a
        :class:`~ml.hwdataset.dataset.FileGroupedSampler` (or its
        :class:`FamilyBalancedGroupedSampler` subclass when
        ``balance_alpha > 0``); val and test stay sequential.

    Raises:
        ValueError: If a split fraction is out of range, if the fractions
            leave no room for a third split, if the corpus is empty, or if any
            of the three partitions would contain no record. A family/source
            filter that leaves too few files to fill the three splits raises from
            :func:`~ml.hwdataset.dataset.create_hw_dataloaders` -- which is
            precisely the failure that filtering *before* the split prevents.

    Note:
        Filtering happens on the **index, before the split**, via
        :meth:`~ml.hwdataset.index.HwCorpusIndex.filter`. That ordering is the
        whole point: the file-level split inside
        :func:`~ml.hwdataset.dataset.create_hw_dataloaders` is inline rather than a
        reusable helper, so the only alternative is to filter each loader
        afterwards -- which can empty a split whose files were all filtered out,
        or leave a family present in training but absent from validation.
    """
    wanted_families = None if families is None else {str(f) for f in families}
    wanted_sources = None if sources is None else {str(s) for s in sources}

    if wanted_families is None and wanted_sources is None:
        corpus = None
    else:
        scanned = build_hw_index(
            roots,
            limit_files=limit_files,
            index_cache=index_cache,
            progress_every=0,
        )
        corpus = scanned.filter(
            families=wanted_families,
            sources=wanted_sources,
        )
        logger.info(
            "corpus filter kept {}/{} records ({}/{} files) across {} families",
            len(corpus.records),
            len(scanned.records),
            len({str(r.path) for r in corpus.records}),
            scanned.files_usable,
            len(corpus.counts_by_family()),
        )

    train_loader, val_loader, test_loader = create_hw_dataloaders(
        roots,
        config=config,
        batch_size=batch_size,
        train_split=train_split,
        val_split=val_split,
        num_workers=num_workers,
        seed=seed,
        cache_size=cache_size,
        # Already applied to `corpus` when filtering; passing it again would
        # raise rather than silently scan a different corpus.
        limit_files=limit_files if corpus is None else None,
        index_cache=index_cache,
        require_exposure=require_exposure,
        use_cache=use_cache,
        cache_root=cache_root,
        index=corpus,
    )

    loaders = [
        _with_shaping_collate(train_loader),
        _with_shaping_collate(val_loader),
        _with_shaping_collate(test_loader),
    ]

    train_loader = loaders[0]

    if balance_alpha > 0.0:
        inner = getattr(train_loader.dataset, "dataset", train_loader.dataset)
        records: Sequence[HwRecordRef] = getattr(inner, "records", ()) or ()
        indices: Sequence[int] | None = getattr(
            train_loader.dataset, "indices", None
        )
        if records and indices is not None:
            loaders[0] = _replace_train_sampler(
                train_loader,
                [records[i] for i in indices],
                balance_alpha=balance_alpha,
                max_file_repeats=max_file_repeats,
                seed=seed,
            )
    return (loaders[0], loaders[1], loaders[2])



def _replace_train_sampler(
    loader: DataLoader,
    records: Sequence[HwRecordRef],
    *,
    balance_alpha: float,
    max_file_repeats: int,
    seed: int,
) -> DataLoader:
    """Swap a train loader's sampler for the family-balanced one.

    Args:
        loader: The train loader.
        records: The records its ``Subset`` covers, in subset order.
        balance_alpha: Balancing strength.
        max_file_repeats: Cap on one file's repetitions per epoch.
        seed: Sampler seed.

    Returns:
        A loader identical to ``loader`` except for its sampler.
    """
    sampler = FamilyBalancedGroupedSampler(
        records,
        seed=seed,
        balance_alpha=balance_alpha,
        max_file_repeats=max_file_repeats,
    )
    logger.info(
        "family-balanced train sampler (alpha={}): files per family {}",
        balance_alpha,
        dict(sampler.families),
    )
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=loader.num_workers,
        collate_fn=shaping_collate,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )


def network_input(
    batch: dict[str, Any], grid: int, in_channels: int = 2
) -> torch.Tensor:
    """Assemble the network's ``(B, in_channels, g, g)`` input tensor.

    The far-field ``image`` is already ``(B, 1, g, g)`` after collation. With
    the default ``in_channels=2`` the standardised ``exposure_log10``
    ``(B, 1)`` is broadcast across the grid and concatenated as a second
    channel: exposure is an *input* covariate in this corpus (9 values
    spanning 0.1-80 ms), and the image keeps the absolute brightness that
    encodes it, so the network sees both.

    Args:
        batch: A batch dict from :func:`shaping_collate`.
        grid: Side length ``g`` of the square grid.
        in_channels: ``2`` (default) = ``[image, exposure_log10 broadcast]``;
            ``1`` = the image alone.

    Returns:
        A ``(B, in_channels, g, g)`` float32 tensor.

    Raises:
        ValueError: If ``in_channels`` is not ``1`` or ``2``, or if the batch
            tensors do not match the expected shapes.
    """
    image = batch["image"]
    if image.dim() != 4 or image.shape[1] != 1:
        raise ValueError(
            f"expected image (B, 1, g, g), got shape {tuple(image.shape)}"
        )
    if image.shape[2] != grid or image.shape[3] != grid:
        raise ValueError(
            f"expected grid {grid}, got image shape {tuple(image.shape)}"
        )
    if in_channels == 1:
        return image
    if in_channels != 2:
        raise ValueError(f"in_channels must be 1 or 2, got {in_channels}")
    exposure = batch["exposure_log10"]
    if exposure.dim() != 2 or exposure.shape[1] != 1:
        raise ValueError(
            f"expected exposure_log10 (B, 1), got shape {tuple(exposure.shape)}"
        )
    exposure_grid = exposure.view(-1, 1, 1, 1).expand(-1, 1, grid, grid)
    return torch.cat([image, exposure_grid], dim=1)


def phase_target(batch: dict[str, Any]) -> torch.Tensor:
    """Assemble the ``(B, 2, g, g)`` ground-truth phasor from a batch.

    The corpus stores the pupil phase as a phasor pair ``(phase_cos,
    phase_sin)`` -- the real and imaginary parts of ``exp(i * phase)`` -- so
    the target is their channel-wise concatenation. The panels are wrapped to
    ``[0, 2*pi)``; recovering an angle from the pair would reintroduce a
    branch cut the network would have to learn, so the pair is used as-is.

    Args:
        batch: A batch dict from :func:`shaping_collate`.

    Returns:
        A ``(B, 2, g, g)`` float32 tensor, channel 0 = ``phase_cos``,
        channel 1 = ``phase_sin``.
    """
    return torch.cat([batch["phase_cos"], batch["phase_sin"]], dim=1)
