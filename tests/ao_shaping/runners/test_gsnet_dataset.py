"""Tests for :mod:`ao_shaping.runners.gsnet_dataset`.

Everything is hardware-free: the corpus is a handful of synthetic pickles in
``tmp_path``, written in the **real two-level nested layout**
(``<family>/<timestamp>/<prefix>_<timestamp>.pkl``) that
:func:`~ao_shaping.runners.gsnet_offline.build_record_index` globs
recursively. ``grid=64`` matches the production default so the shapes asserted
here are the shapes ``ml.gsnet.train.train_gsnet`` consumes.
"""

from __future__ import annotations

import io
import pickle
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from ml.gsnet.dataset import make_source_intensity
from torch.utils.data import SequentialSampler

from ao_shaping.runners.gsnet_dataset import (
    DEFAULT_PREFETCH_FACTOR,
    FileGroupedSampler,
    GSNetDebugDataset,
    GSNetRecordError,
    build_gsnet_dataloader,
)
from ao_shaping.runners.gsnet_offline import (
    RecordIndex,
    build_record_index,
    farfield_to_grid,
    infer_n_max,
    pupil_phase_to_grid,
    reconstruct_pupil_phase_rad,
)

GRID = 64
SLM_WIDTH = 1920
SLM_HEIGHT = 1200
# The corpus device's ZERNIKE_APERTURE_RADIUS
# (slm_zernike_pib.py:235 / slm_zernike_shaping.py:225) -- deliberately NOT
# min(SLM_WIDTH, SLM_HEIGHT) / 2.
SLM_RADIUS = 300.0

# Observed ``len(_c)`` values in the real corpus -> must never be hardcoded.
MODE_COUNTS = (15, 66, 78)


# ---------------------------------------------------------------------------
# Synthetic corpus helpers
# ---------------------------------------------------------------------------
def _far_field_frame(row: int, height: int = 100, width: int = 120) -> np.ndarray:
    """A deterministic far-field frame with an off-centre 0-order blob.

    The blob centre moves with ``row`` so different records get different
    ``argmax`` anchors -- exactly the case ``farfield_to_grid`` has to handle
    (the optical axis is *not* the frame centre on a 2f bench).
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cx = 30.0 + 0.7 * (row % 7)
    cy = 40.0 + 1.1 * (row % 5)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    frame = 250.0 * np.exp(-r2 / (2 * 6.0**2))
    # A dim satellite speckle so peak-normalisation is actually exercised.
    frame[10:14, 90:94] += 40.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _zernike_coefficients(n_terms: int, row: int) -> np.ndarray:
    """Deterministic Noll-order coefficients with a known leading structure.

    Index 0 (Noll 1) is piston -- a far-field no-op, but reconstructed for
    faithfulness. Index 3 (Noll 4, defocus) and index 10 (Noll 11, spherical)
    carry amplitude, so ``gt_phase`` is never an all-zero plane.
    """
    c = np.zeros(n_terms, dtype=np.float64)
    if n_terms > 3:
        c[3] = 0.8 + 0.01 * row
    if n_terms > 10:
        c[10] = -0.35 + 0.005 * row
    c[0] = 0.05 * row
    return c


def _write_pickle(
    root: Path,
    family: str,
    tag: str,
    n_records: int,
    n_terms: int,
) -> Path:
    """Write one debug-style pickle in the real two-level nested layout."""
    timestamp = "20260926_164358"
    directory = root / f"{family}_{tag}" / timestamp
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{family}_{tag}_{timestamp}.pkl"
    records: dict[int, dict[str, Any]] = {
        row: {
            "_img": _far_field_frame(row),
            "_c": _zernike_coefficients(n_terms, row),
            "_grad": np.zeros(n_terms, dtype=np.float64),
        }
        for row in range(n_records)
    }
    with open(path, "wb") as handle:
        pickle.dump(records, handle)
    return path


def _corpus(
    tmp_path: Path,
    *,
    n_terms: int = 66,
    counts: tuple[int, int] = (3, 4),
) -> RecordIndex:
    """Build a two-family nested corpus and return its ``RecordIndex``."""
    _write_pickle(tmp_path, "slm_pib", "run", counts[0], n_terms)
    _write_pickle(tmp_path, "slm_zernike_shaping", "run", counts[1], n_terms)
    return build_record_index(roots=str(tmp_path / "slm_*"))


def _dataset(index: RecordIndex, **kwargs: Any) -> GSNetDebugDataset:
    """A Dataset with the production SLM200 geometry."""
    kwargs.setdefault("grid", GRID)
    kwargs.setdefault("slm_radius", SLM_RADIUS)
    return GSNetDebugDataset(
        index,
        slm_width=SLM_WIDTH,
        slm_height=SLM_HEIGHT,
        **kwargs,
    )


def _record_of(path: Path, key: int) -> dict[str, Any]:
    """Read one record straight from disk (independent of the Dataset)."""
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return payload[key]


def _expected_gt_phase(c: np.ndarray) -> np.ndarray:
    """The documented ``_c`` -> ``gt_phase`` reconstruction, as a numpy array."""
    phase = reconstruct_pupil_phase_rad(
        np.asarray(c, dtype=np.float64),
        n_max=infer_n_max(len(c)),
        slm_width=SLM_WIDTH,
        slm_height=SLM_HEIGHT,
        radius=SLM_RADIUS,
    )
    return pupil_phase_to_grid(phase, GRID)


def _reference_source(source_type: str = "gaussian") -> torch.Tensor:
    """``make_source_intensity`` as a ``(1, grid, grid)`` tensor."""
    raw = np.ascontiguousarray(
        make_source_intensity(GRID, source_type), dtype=np.float32
    )
    return torch.from_numpy(raw)[None]


def _spy_on_pickle_load(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch ``pickle.load`` with a counter; returns the recorded call names.

    The Dataset module resolves ``pickle.load`` as an attribute at call time,
    so this patches exactly the function it uses.
    """
    calls: list[str] = []
    real_load = pickle.load

    def spy(*args: Any, **kwargs: Any) -> Any:
        handle = args[0] if args else kwargs.get("file")
        calls.append(str(getattr(handle, "name", "<no-handle>")))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(pickle, "load", spy)
    return calls


def _runs(values: list[Any]) -> int:
    """Number of maximal constant runs in ``values``."""
    return 1 + sum(1 for a, b in zip(values, values[1:]) if a != b)


# ---------------------------------------------------------------------------
# 1. shapes / dtype / finiteness / independence
# ---------------------------------------------------------------------------
class TestGetItem:
    def test_returns_three_grid_tensors(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path)
        dataset = _dataset(index)

        assert len(dataset) == 7
        for position in range(len(dataset)):
            for name, tensor in zip(
                ("source", "target", "gt_phase"), dataset[position]
            ):
                assert isinstance(tensor, torch.Tensor), name
                assert tensor.shape == (1, GRID, GRID), (name, tensor.shape)
                assert tensor.dtype == torch.float32, name
                assert torch.isfinite(tensor).all(), name

    def test_source_is_the_pupil_illumination(self, tmp_path: Path) -> None:
        """``source`` is ``make_source_intensity``, NOT the CCD image."""
        index = _corpus(tmp_path)
        dataset = _dataset(index, source_type="gaussian")

        source, _target, _gt = dataset[0]

        assert torch.equal(source, _reference_source("gaussian"))

    def test_source_is_identical_for_every_record(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path)
        dataset = _dataset(index)

        for position in range(len(dataset)):
            assert torch.equal(dataset[position][0], dataset[0][0])

    def test_uniform_source_type_is_honoured(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path)
        dataset = _dataset(index, source_type="uniform")

        assert torch.equal(dataset[0][0], _reference_source("uniform"))

    def test_tensors_are_fresh_and_unaliased(self, tmp_path: Path) -> None:
        """Mutating a returned tensor must not corrupt later reads."""
        index = _corpus(tmp_path)
        dataset = _dataset(index)

        source, target, gt_phase = dataset[0]
        source_before = source.clone()
        target_before = target.clone()
        gt_before = gt_phase.clone()
        source.add_(10.0)
        target.add_(10.0)
        gt_phase.add_(10.0)

        source2, target2, gt2 = dataset[0]
        assert torch.equal(source2, source_before)
        assert torch.equal(target2, target_before)
        assert torch.equal(gt2, gt_before)
        # The cached illumination buffer must still be pristine.
        assert torch.equal(dataset[1][0], source_before)

    def test_records_from_different_files_differ(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path)
        dataset = _dataset(index)

        _s, target_a, gt_a = dataset[0]
        _s, target_b, gt_b = dataset[-1]
        assert not torch.equal(target_a, target_b)
        assert not torch.equal(gt_a, gt_b)

    def test_negative_index_addresses_from_the_end(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path)
        dataset = _dataset(index)

        assert torch.equal(dataset[-1][1], dataset[len(dataset) - 1][1])

    @pytest.mark.parametrize("position", [7, 99, -8, -100])
    def test_out_of_range_raises_index_error(
        self, tmp_path: Path, position: int
    ) -> None:
        index = _corpus(tmp_path)
        dataset = _dataset(index)

        with pytest.raises(IndexError):
            dataset[position]


# ---------------------------------------------------------------------------
# 2. target == farfield_to_grid(record["_img"], grid)
# ---------------------------------------------------------------------------
class TestTarget:
    @pytest.mark.parametrize("position", [0, 1, 2, 3, 4, 5, 6])
    def test_target_matches_farfield_to_grid(
        self, tmp_path: Path, position: int
    ) -> None:
        index = _corpus(tmp_path)
        dataset = _dataset(index)
        path, key = index.entries[position]

        _source, target, _gt = dataset[position]
        expected = farfield_to_grid(_record_of(path, key)["_img"], GRID)

        assert target.shape == (1, GRID, GRID)
        assert torch.equal(target[0], torch.from_numpy(expected))
        assert float(target.min()) >= 0.0
        assert float(target.max()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. gt_phase == documented _c reconstruction
# ---------------------------------------------------------------------------
class TestGtPhase:
    @pytest.mark.parametrize("position", [0, 3, 6])
    def test_gt_phase_matches_reconstruction(
        self, tmp_path: Path, position: int
    ) -> None:
        index = _corpus(tmp_path)
        dataset = _dataset(index)
        path, key = index.entries[position]

        _source, _target, gt_phase = dataset[position]
        expected = _expected_gt_phase(_record_of(path, key)["_c"])

        assert gt_phase.shape == (1, GRID, GRID)
        assert torch.equal(gt_phase[0], torch.from_numpy(expected))
        # Defocus/spherical are non-zero, so a silent all-zero phase would be a
        # real failure rather than a rounding artefact.
        assert float(gt_phase.abs().max()) > 0.1

    def test_gt_phase_varies_across_records(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path)
        dataset = _dataset(index)

        first = dataset[0][2]
        second = dataset[1][2]
        assert not torch.equal(first, second)
        assert torch.isfinite(first).all() and torch.isfinite(second).all()


# ---------------------------------------------------------------------------
# 3b. aperture radius regression: 300 (ZERNIKE_APERTURE_RADIUS), NOT 600
# ---------------------------------------------------------------------------
class TestApertureRadius:
    """Pin the Zernike aperture radius so it is never re-derived as half the panel.

    The corpus was recorded with ``ZERNIKE_APERTURE_RADIUS = 300.0``; using
    ``min(w, h) / 2 = 600`` silently smears the phase across ~6x more grid cells
    and destroys the beam/aperture geometry the network is meant to learn.
    """

    @staticmethod
    def _footprint(radius: float) -> np.ndarray:
        c = np.zeros(66, dtype=np.float64)
        c[3] = 0.5   # defocus  (2, 0)
        c[10] = 0.3  # spherical (4, 0)
        phase = reconstruct_pupil_phase_rad(
            c,
            n_max=10,
            slm_width=SLM_WIDTH,
            slm_height=SLM_HEIGHT,
            radius=radius,
        )
        return pupil_phase_to_grid(phase, GRID)

    def test_default_radius_is_three_hundred(self) -> None:
        from ao_shaping.runners.gsnet_dataset import DEFAULT_SLM_RADIUS

        assert DEFAULT_SLM_RADIUS == 300.0
        assert SLM_RADIUS == 300.0

    def test_footprint_shrinks_relative_to_double_radius(self) -> None:
        small = self._footprint(300.0)
        large = self._footprint(600.0)

        n_small = int(np.count_nonzero(small))
        n_large = int(np.count_nonzero(large))
        assert n_small > 0, "a defocus+spherical phase must light some cells"
        assert n_small < n_large, (
            f"radius 300 must cover strictly fewer cells than radius 600; "
            f"got {n_small} vs {n_large}"
        )

    def test_footprint_is_centred(self) -> None:
        grid = self._footprint(300.0)
        rows = np.where(np.any(grid != 0, axis=1))[0]
        cols = np.where(np.any(grid != 0, axis=0))[0]

        assert rows.size and cols.size, "footprint must be non-empty"
        # The aperture disc is centred on the panel, so the lit span must be
        # roughly centred in the grid. Allow a couple of cells of slack for the
        # anisotropic 1200 -> 1216 ceil-padding.
        row_mid = (rows.min() + rows.max()) / 2.0
        col_mid = (cols.min() + cols.max()) / 2.0
        assert abs(row_mid - (GRID - 1) / 2.0) <= 2.0
        assert abs(col_mid - (GRID - 1) / 2.0) <= 2.0

    def test_footprint_occupies_a_minority_of_the_grid(self) -> None:
        n_small = int(np.count_nonzero(self._footprint(300.0)))
        assert n_small < GRID * GRID // 4, (
            f"radius 300 should cover well under a quarter of the grid, "
            f"got {n_small}/{GRID * GRID}"
        )

    def test_dataset_honours_overridden_radius(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path)
        wide = _dataset(index, slm_radius=600.0)[0][2]
        narrow = _dataset(index, slm_radius=300.0)[0][2]

        assert int(torch.count_nonzero(narrow)) < int(torch.count_nonzero(wide))


# ---------------------------------------------------------------------------
# 4. varying len(_c) -> n_max inferred, never hardcoded
# ---------------------------------------------------------------------------
class TestVaryingModeCount:
    @pytest.mark.parametrize("n_terms", MODE_COUNTS)
    def test_each_observed_mode_count_loads(
        self, tmp_path: Path, n_terms: int
    ) -> None:
        index = _corpus(tmp_path, n_terms=n_terms, counts=(2, 2))
        dataset = _dataset(index)

        assert infer_n_max(n_terms) in (4, 10, 11)
        for position in range(len(dataset)):
            _s, target, gt_phase = dataset[position]
            assert target.shape == (1, GRID, GRID)
            assert torch.isfinite(gt_phase).all()

    def test_three_mode_counts_in_one_corpus(self, tmp_path: Path) -> None:
        """A hardcoded ``n_max`` could not serve 15, 66 and 78 in one corpus."""
        for position, n_terms in enumerate(MODE_COUNTS):
            _write_pickle(tmp_path, "slm_pib", f"fam{position}", 2, n_terms)
        index = build_record_index(roots=str(tmp_path / "slm_*"))
        dataset = _dataset(index)

        assert len(dataset) == 6
        seen_n_terms: set[int] = set()
        for position in range(len(dataset)):
            path, key = index.entries[position]
            c = _record_of(path, key)["_c"]
            seen_n_terms.add(len(c))
            # Each record must be reconstructed with *its own* inferred n_max.
            assert torch.equal(
                dataset[position][2][0], torch.from_numpy(_expected_gt_phase(c))
            )
        assert seen_n_terms == set(MODE_COUNTS)


# ---------------------------------------------------------------------------
# 5. LRU: one pickle.load per contiguous group
# ---------------------------------------------------------------------------
class TestLruCache:
    def test_single_file_triggers_one_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        index = _corpus(tmp_path, counts=(4, 0))
        dataset = _dataset(index)
        assert len({p for p, _ in index.entries}) == 1

        calls = _spy_on_pickle_load(monkeypatch)
        for position in range(len(dataset)):
            dataset[position]

        assert len(calls) == 1, calls

    def test_cache_size_one_misses_when_alternating_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        index = _corpus(tmp_path, counts=(3, 3))
        dataset = _dataset(index, cache_size=1)
        owner = [p for p, _ in index.entries]
        assert owner[0] != owner[3], "expected two distinct file groups"

        calls = _spy_on_pickle_load(monkeypatch)
        for position in (0, 3, 1, 4, 2, 5):
            dataset[position]

        assert len(calls) == 6, calls

    def test_cache_size_two_hits_when_alternating_two_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        index = _corpus(tmp_path, counts=(3, 3))
        dataset = _dataset(index, cache_size=2)
        dataset[0]  # file A
        dataset[3]  # file B -> the cache now holds both
        assert len(dataset.cached_files) == 2

        calls = _spy_on_pickle_load(monkeypatch)
        for position in (3, 1, 4, 2, 5, 0):
            dataset[position]

        assert calls == [], "both files were already cached"

    def test_file_grouped_sampler_converts_loads_to_hits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of ``FileGroupedSampler``: one load per file."""
        for position in range(3):
            _write_pickle(tmp_path, "slm_pib", f"fam{position}", 3, 66)
        index = build_record_index(roots=str(tmp_path / "slm_*"))
        dataset = _dataset(index)

        calls = _spy_on_pickle_load(monkeypatch)
        for position in FileGroupedSampler(index, seed=4):
            dataset[position]

        assert len(calls) == 3, calls

    def test_cache_is_bounded_and_reports_paths(self, tmp_path: Path) -> None:
        for position, n_terms in enumerate(MODE_COUNTS):
            _write_pickle(tmp_path, "slm_pib", f"fam{position}", 2, n_terms)
        index = build_record_index(roots=str(tmp_path / "slm_*"))
        dataset = _dataset(index, cache_size=2)

        for position in range(len(dataset)):
            dataset[position]
            assert len(dataset.cached_files) <= 2
        assert len(dataset.cached_files) == 2

        dataset.clear_cache()
        assert dataset.cached_files == ()

    def test_cache_size_zero_disables_caching(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(2, 0))
        dataset = _dataset(index, cache_size=0)

        dataset[0]
        assert dataset.cached_files == ()
        assert dataset[1][1].shape == (1, GRID, GRID)

    def test_init_opens_no_pickle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The constructor must not deserialise anything."""
        index = _corpus(tmp_path)
        calls = _spy_on_pickle_load(monkeypatch)

        dataset = _dataset(index)

        assert calls == []
        assert dataset.cached_files == ()
        assert len(dataset) == len(index.entries)


# ---------------------------------------------------------------------------
# 6. picklability (DataLoader spawn workers re-pickle the Dataset)
# ---------------------------------------------------------------------------
class TestPickling:
    def test_round_trip_succeeds_and_drops_the_cache(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(3, 3))
        dataset = _dataset(index, cache_size=2)
        for position in range(len(dataset)):
            dataset[position]
        assert len(dataset.cached_files) == 2

        restored = pickle.loads(pickle.dumps(dataset))

        assert isinstance(restored, GSNetDebugDataset)
        assert restored.cached_files == ()
        assert len(restored) == len(dataset)
        assert restored.grid == dataset.grid
        assert restored.entries == dataset.entries

    def test_pickle_payload_excludes_cached_arrays(self, tmp_path: Path) -> None:
        """A warm cache must not inflate the serialised payload at all."""
        index = _corpus(tmp_path, counts=(3, 3))
        dataset = _dataset(index, cache_size=2)
        cold_size = len(pickle.dumps(dataset))

        for position in range(len(dataset)):
            dataset[position]
        assert len(dataset.cached_files) == 2

        assert len(pickle.dumps(dataset)) == cold_size

    def test_restored_dataset_still_serves_samples(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(2, 2))
        dataset = _dataset(index)
        expected = [dataset[i][1].clone() for i in range(len(dataset))]

        restored = pickle.loads(pickle.dumps(dataset))
        for position in range(len(restored)):
            assert torch.equal(restored[position][1], expected[position])

    def test_never_stores_an_open_handle(self, tmp_path: Path) -> None:
        """Handles are unpicklable; the cache may only hold deserialised dicts."""
        index = _corpus(tmp_path, counts=(2, 2))
        dataset = _dataset(index)
        for position in range(len(dataset)):
            dataset[position]

        for value in vars(dataset).values():
            assert not isinstance(value, io.IOBase), repr(value)
            if isinstance(value, OrderedDict):
                for key, payload in value.items():
                    assert isinstance(key, Path)
                    assert isinstance(payload, dict)
                    assert not any(
                        isinstance(item, io.IOBase) for item in payload.values()
                    )


# ---------------------------------------------------------------------------
# 7. FileGroupedSampler
# ---------------------------------------------------------------------------
class TestFileGroupedSampler:
    def test_yields_every_index_exactly_once(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(3, 4))
        sampler = FileGroupedSampler(index)

        order = list(sampler)

        assert sorted(order) == list(range(len(index.entries)))
        assert len(order) == len(index.entries)
        assert len(sampler) == len(index.entries)

    def test_groups_same_file_indices_contiguously(self, tmp_path: Path) -> None:
        for position in range(3):
            _write_pickle(tmp_path, "slm_pib", f"fam{position}", 3, 66)
        index = build_record_index(roots=str(tmp_path / "slm_*"))
        sampler = FileGroupedSampler(index, seed=3)
        assert sampler.num_groups == 3

        owner = [path for path, _ in index.entries]
        order = list(sampler)

        assert _runs([owner[i] for i in order]) == 3
        for path in set(owner):
            positions = [i for i in order if owner[i] == path]
            assert len(positions) == 3
            # Contiguous == the group's positions span one unbroken run. The
            # order *inside* the run is deliberately shuffled.
            assert max(positions) - min(positions) == len(positions) - 1

    def test_actually_shuffles(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(6, 6))
        sampler = FileGroupedSampler(index, seed=11)

        orders = [list(sampler) for _ in range(5)]

        assert any(order != orders[0] for order in orders[1:]), orders
        for order in orders:
            assert sorted(order) == list(range(len(index.entries)))

    def test_seed_reproduces_the_sequence(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(5, 5))

        first = [list(FileGroupedSampler(index, seed=7)) for _ in range(3)]
        second = [list(FileGroupedSampler(index, seed=7)) for _ in range(3)]

        assert first == second

    def test_different_seeds_differ(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(5, 5))

        assert list(FileGroupedSampler(index, seed=1)) != list(
            FileGroupedSampler(index, seed=2)
        )

    @pytest.mark.parametrize("num_samples", [1, 3, 7, 13, 40])
    def test_upsample_and_downsample_lengths(
        self, tmp_path: Path, num_samples: int
    ) -> None:
        index = _corpus(tmp_path, counts=(3, 4))
        total = len(index.entries)
        sampler = FileGroupedSampler(index, num_samples=num_samples)

        order = list(sampler)

        assert len(order) == num_samples
        assert len(sampler) == num_samples
        assert set(order) <= set(range(total))
        # Which indices survive downsampling is *not* ``range(num_samples)``:
        # whole groups are dropped in shuffled order, so the survivor set is
        # arbitrary -- but every file's survivors stay in one contiguous run
        # per emitted cycle (``ceil(num_samples / total)`` cycles).
        owner = [path for path, _ in index.entries]
        cycles = -(-num_samples // total)
        assert _runs([owner[i] for i in order]) <= sampler.num_groups * cycles
        if num_samples <= total:
            assert len(set(order)) == num_samples, "downsampling must not repeat"

    def test_downsample_keeps_groups_contiguous(self, tmp_path: Path) -> None:
        for position in range(3):
            _write_pickle(tmp_path, "slm_pib", f"fam{position}", 3, 66)
        index = build_record_index(roots=str(tmp_path / "slm_*"))
        owner = [path for path, _ in index.entries]

        for num_samples in (3, 5, 8):
            order = list(FileGroupedSampler(index, num_samples=num_samples, seed=1))
            assert _runs([owner[i] for i in order]) <= 3

    def test_upsample_repeats_whole_cycles(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(3, 3))
        total = len(index.entries)
        order = list(FileGroupedSampler(index, num_samples=3 * total, seed=2))

        assert len(order) == 3 * total
        # Each of the 3 full cycles is a complete permutation of the index.
        for cycle in range(3):
            window = order[cycle * total : (cycle + 1) * total]
            assert sorted(window) == list(range(total))

    def test_zero_samples(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(2, 2))
        sampler = FileGroupedSampler(index, num_samples=0)

        assert list(sampler) == []
        assert len(sampler) == 0

    def test_rejects_bad_arguments(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(1, 1))

        with pytest.raises(TypeError):
            FileGroupedSampler([("a", 0)])  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            FileGroupedSampler(index, num_samples=-1)


# ---------------------------------------------------------------------------
# 8. DataLoader factory
# ---------------------------------------------------------------------------
class TestBuildGsnetDataloader:
    @staticmethod
    def _shutdown_workers(loader: Any) -> None:
        """Reap DataLoader worker processes deterministically.

        With ``num_workers > 0`` on Windows the loader uses ``spawn``. If those
        children are still alive when the interpreter tears down -- which is
        what happens when the native driver stack (torch/CUDA/ctypes SDKs) has
        already been loaded by earlier tests -- the process can die with an
        access violation (0xC0000005) *after* every test already reported a
        pass, turning the run into a bogus failure. Shutting the iterator down
        explicitly keeps teardown deterministic.
        """
        iterator = getattr(loader, "_iterator", None)
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()

    def _assert_batches(self, loader: Any, expected_batches: int) -> None:
        try:
            seen = 0
            for source, target, gt_phase in loader:
                for tensor in (source, target, gt_phase):
                    assert tensor.dtype == torch.float32
                    assert tensor.ndim == 4
                    assert tensor.shape[1:] == (1, GRID, GRID)
                    assert torch.isfinite(tensor).all()
                assert source.shape[0] == target.shape[0] == gt_phase.shape[0]
                seen += 1
            assert seen == expected_batches
        finally:
            self._shutdown_workers(loader)

    def test_num_workers_zero(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(3, 3))
        loader = build_gsnet_dataloader(
            index, grid=GRID, batch_size=4, num_workers=0
        )

        assert loader.num_workers == 0
        assert loader.batch_size == 4
        # torch forbids a non-None prefetch_factor without worker processes
        assert loader.prefetch_factor is None
        self._assert_batches(loader, 2)

    def test_num_workers_two_windows_spawn(self, tmp_path: Path) -> None:
        """Windows uses ``spawn``: the Dataset is re-pickled per worker."""
        index = _corpus(tmp_path, counts=(3, 3))
        loader = build_gsnet_dataloader(
            index, grid=GRID, batch_size=2, num_workers=2, pin_memory=False
        )

        assert loader.num_workers == 2
        assert loader.prefetch_factor == DEFAULT_PREFETCH_FACTOR

        # ``spawn`` re-imports ``__main__`` inside every worker *before* it can
        # unpickle the Dataset. In a shared pytest session ``__main__`` is
        # whatever an earlier test left behind: Streamlit's
        # ``AppTest.from_function`` (see tests/ao_shaping/gui/slm/
        # test_zernike_abort_repro.py) swaps ``__main__.__file__`` for a
        # throwaway script that calls the app with undefined ``__args``, so the
        # worker dies with ``NameError`` during bootstrap and torch reports
        # "DataLoader worker exited unexpectedly". Point ``__main__`` at an
        # empty script while the workers run so this test depends only on the
        # code it exercises, never on ambient global state.
        main_module = sys.modules["__main__"]
        original_main = getattr(main_module, "__file__", None)
        stub_main = tmp_path / "_mp_main_stub.py"
        stub_main.write_text("", encoding="utf-8")
        main_module.__file__ = str(stub_main)
        try:
            self._assert_batches(loader, 3)
        finally:
            if original_main is not None:
                main_module.__file__ = original_main
            else:
                del main_module.__file__

    def test_sequential_sampler_when_shuffle_false(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(2, 2))
        loader = build_gsnet_dataloader(
            index, grid=GRID, batch_size=4, shuffle=False
        )

        assert isinstance(loader.sampler, SequentialSampler)
        assert list(loader.sampler) == list(range(len(index.entries)))
        self._assert_batches(loader, 1)

    def test_file_grouped_sampler_when_shuffle_true(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(2, 2))
        loader = build_gsnet_dataloader(
            index, grid=GRID, batch_size=2, shuffle=True, num_samples=4, seed=5
        )

        assert isinstance(loader.sampler, FileGroupedSampler)
        assert loader.sampler.num_samples == 4
        assert len(loader.sampler) == 4
        self._assert_batches(loader, 2)

    def test_pin_memory_auto_is_off_without_workers(self, tmp_path: Path) -> None:
        """Auto-pinning requires CUDA *and* workers, so it is always off here."""
        index = _corpus(tmp_path, counts=(2, 2))
        loader = build_gsnet_dataloader(index, grid=GRID, num_workers=0)

        assert loader.pin_memory is False

    def test_pin_memory_can_be_forced(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(2, 2))
        forced_off = build_gsnet_dataloader(
            index, grid=GRID, num_workers=0, pin_memory=False
        )

        assert forced_off.pin_memory is False

    def test_rejects_empty_index_and_bad_batch_size(self, tmp_path: Path) -> None:
        empty = build_record_index(roots=str(tmp_path / "nothing_here"))
        assert len(empty.entries) == 0

        with pytest.raises(ValueError):
            build_gsnet_dataloader(empty, grid=GRID)
        index = _corpus(tmp_path, counts=(1, 1))
        with pytest.raises(ValueError):
            build_gsnet_dataloader(index, grid=GRID, batch_size=0)
        with pytest.raises(TypeError):
            build_gsnet_dataloader("nope")  # type: ignore[arg-type]

    def test_dataloader_passes_geometry_through(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(1, 1))
        loader = build_gsnet_dataloader(
            index,
            grid=32,
            batch_size=2,
            source_type="uniform",
            slm_width=640,
            slm_height=480,
            slm_radius=200.0,
        )

        dataset = loader.dataset
        assert isinstance(dataset, GSNetDebugDataset)
        assert dataset.grid == 32
        for source, target, gt_phase in loader:
            assert source.shape == (2, 1, 32, 32)
            assert target.shape == (2, 1, 32, 32)
            assert gt_phase.shape == (2, 1, 32, 32)

    def test_batches_feed_train_gsnet_math(self, tmp_path: Path) -> None:
        """``source.clamp_min(0) + 1e-12`` must be square-rootable, as in train."""
        index = _corpus(tmp_path, counts=(2, 2))
        loader = build_gsnet_dataloader(
            index, grid=GRID, batch_size=4, num_workers=0
        )

        for source, target, gt_phase in loader:
            source_amp = torch.sqrt(source.clamp_min(0.0) + 1e-12)
            assert torch.isfinite(source_amp).all()
            assert float(target.min()) >= 0.0
            assert gt_phase.shape == target.shape


# ---------------------------------------------------------------------------
# 9. malformed records raise a clear typed error (never silently skipped)
# ---------------------------------------------------------------------------
class TestMalformedRecords:
    def _index_with(self, tmp_path: Path, record: dict[str, Any]) -> RecordIndex:
        directory = tmp_path / "slm_pib_bad" / "20260926_164358"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "slm_pib_bad_20260926_164358.pkl"
        with open(path, "wb") as handle:
            pickle.dump({0: record}, handle)
        return build_record_index(roots=str(tmp_path / "slm_pib_*"))

    def test_missing_img_raises_typed_error(self, tmp_path: Path) -> None:
        index = self._index_with(tmp_path, {"_c": _zernike_coefficients(15, 0)})
        dataset = _dataset(index)

        with pytest.raises(GSNetRecordError, match="_img"):
            dataset[0]

    def test_missing_c_raises_typed_error(self, tmp_path: Path) -> None:
        index = self._index_with(tmp_path, {"_img": _far_field_frame(0)})
        dataset = _dataset(index)

        with pytest.raises(GSNetRecordError, match="_c"):
            dataset[0]

    def test_empty_c_raises_typed_error(self, tmp_path: Path) -> None:
        index = self._index_with(
            tmp_path,
            {"_img": _far_field_frame(0), "_c": np.zeros(0, dtype=np.float64)},
        )
        dataset = _dataset(index)

        with pytest.raises(GSNetRecordError, match="_c"):
            dataset[0]

    def test_non_triangular_c_raises_typed_error(self, tmp_path: Path) -> None:
        """A non-triangular mode count is a corrupt record, not a caller bug."""
        index = self._index_with(
            tmp_path,
            {"_img": _far_field_frame(0), "_c": np.zeros(7, dtype=np.float64)},
        )
        dataset = _dataset(index)

        with pytest.raises(GSNetRecordError, match="triangular"):
            dataset[0]

    def test_missing_record_key_raises_typed_error(self, tmp_path: Path) -> None:
        directory = tmp_path / "slm_pib_gap" / "20260926_164358"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "slm_pib_gap_20260926_164358.pkl"
        with open(path, "wb") as handle:
            pickle.dump({5: {"_img": _far_field_frame(0), "_c": np.zeros(15)}}, handle)
        # Index a key the pickle does not contain.
        dataset = _dataset(RecordIndex(entries=((path, 99),)))

        with pytest.raises(GSNetRecordError, match="99"):
            dataset[0]

    def test_non_dict_record_raises_typed_error(self, tmp_path: Path) -> None:
        directory = tmp_path / "slm_pib_scalar" / "20260926_164358"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "slm_pib_scalar_20260926_164358.pkl"
        with open(path, "wb") as handle:
            pickle.dump({0: 42}, handle)
        dataset = _dataset(RecordIndex(entries=((path, 0),)))

        with pytest.raises(GSNetRecordError, match="not a dict"):
            dataset[0]

    def test_non_dict_root_raises_typed_error(self, tmp_path: Path) -> None:
        directory = tmp_path / "slm_pib_list" / "20260926_164358"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "slm_pib_list_20260926_164358.pkl"
        with open(path, "wb") as handle:
            pickle.dump([{"_img": _far_field_frame(0), "_c": np.zeros(15)}], handle)
        dataset = _dataset(RecordIndex(entries=((path, 0),)))

        with pytest.raises(GSNetRecordError, match="dict"):
            dataset[0]

    def test_deleted_pickle_raises_typed_error(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(1, 0))
        dataset = _dataset(index)
        dataset[0]
        dataset.clear_cache()
        index.entries[0][0].unlink()

        with pytest.raises(GSNetRecordError, match="does not exist"):
            dataset[0]


# ---------------------------------------------------------------------------
# constructor validation / memory contract
# ---------------------------------------------------------------------------
class TestConstructorValidation:
    def test_rejects_non_record_index(self) -> None:
        with pytest.raises(TypeError):
            GSNetDebugDataset([("a", 0)])  # type: ignore[arg-type]

    def test_bad_grid_and_cache_size(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(1, 1))

        with pytest.raises(ValueError):
            GSNetDebugDataset(index, grid=1)
        with pytest.raises(ValueError):
            GSNetDebugDataset(index, cache_size=-1)

    def test_unknown_source_type(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(1, 1))

        with pytest.raises(ValueError):
            GSNetDebugDataset(index, source_type="banana")

    def test_retains_paths_and_ints_only(self, tmp_path: Path) -> None:
        index = _corpus(tmp_path, counts=(2, 2))
        dataset = _dataset(index)

        assert dataset.entries == tuple(index.entries)
        for path, key in dataset.entries:
            assert isinstance(path, Path)
            assert isinstance(key, int)
        assert dataset.grid == GRID

    def test_constructor_holds_one_small_array_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``__init__`` must deserialise nothing and retain no pickle payload."""
        index = _corpus(tmp_path, counts=(4, 4))
        calls = _spy_on_pickle_load(monkeypatch)

        dataset = _dataset(index)

        assert calls == []
        arrays = [
            value
            for value in vars(dataset).values()
            if isinstance(value, (torch.Tensor, np.ndarray))
        ]
        assert len(arrays) == 1, "only the shared source illumination is retained"
        assert tuple(arrays[0].shape) == (1, GRID, GRID)
