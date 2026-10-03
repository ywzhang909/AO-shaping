"""Tests for :mod:`ml.hwdataset.cache`.

The headline property is **bit-identity** with the direct read path: the cache is
only allowed to be a storage optimisation.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ml.hwdataset.cache import (
    CACHE_FILENAMES,
    CACHE_FORMAT_VERSION,
    HwCacheError,
    HwCachedSource,
    cache_dir_for,
    load_hw_cache,
    prepare_hw_cache,
)
from ml.hwdataset.index import HwCorpusIndex, PhaseSource, build_hw_index
from ml.hwdataset.records import Materialiser, MaterialiserConfig

_STAMP = "20260101_000000"

# A small but complete panel: 60x40, centre (30, 20), so the ROI fits and every
# phase source has room to be reconstructed.
_CONFIG = MaterialiserConfig(
    grid=8,
    panel_center=(30, 20),
    panel_radius=15,
    panel_resolution=(60, 40),
    zernike_radius=10.0,
    freeform_radius=10.0,
)


def _write(directory: Path, stem: str, payload: object) -> Path:
    run = directory / _STAMP
    run.mkdir(parents=True, exist_ok=True)
    path = run / f"{stem}.pkl"
    path.write_bytes(pickle.dumps(payload))
    return path


def _frame(value: int, side: int = 16) -> np.ndarray:
    return np.full((side, side), value % 256, dtype=np.uint8)


def _panel_gray(i: int) -> dict:
    phase = np.zeros((40, 60), dtype=np.uint16)
    phase[14:27, 24:37] = 500 + i
    return {"_phase": phase, "_img": _frame(40 + i), "exp_t": 0.5 + i}


def _panel_rad(i: int) -> dict:
    phase = np.zeros((40, 60), dtype=np.float32)
    phase[14:27, 24:37] = 1.5 + 0.1 * i
    return {"_phase": phase, "_img": _frame(80 + i), "exp_t": 2.0}


def _zernike(i: int) -> dict:
    return {
        "_c": np.linspace(-0.5, 0.5, 15) + i * 0.01,
        "_img": _frame(120 + i),
        "exp_t": 3.0,
    }


def _freeform(i: int) -> dict:
    return {
        "_c": np.linspace(0.0, 1.0, 16) + i * 0.01,
        "_img": _frame(160 + i),
        "exp_t": 4.0,
    }


class _RecorderStandIn:
    """Module-level stand-in for the pickled ``Recorder`` (``.history`` list)."""

    def __init__(self, history: list[dict]) -> None:
        self.history = history


def _mixed_corpus(root: Path) -> None:
    """One file per phase source, plus a Recorder payload and a ``None`` stub."""
    _write(root / "model_in_loop_hw_collect_x", "rad", {i: _panel_rad(i) for i in range(3)})
    _write(root / "slm_pib_x", "gray", {i: _panel_gray(i) for i in range(3)})
    _write(root / "slm_zernike_shaping_x", "zern", {i: _zernike(i) for i in range(2)})
    _write(root / "slm_gsnet_square_x", "free", {i: _freeform(i) for i in range(2)})
    _write(root / "slm_pib_online", "rec", _RecorderStandIn([_zernike(0), _zernike(1)]))
    _write(root / "slm_pib_online", "none", None)


def _build(
    root: Path, config: MaterialiserConfig = _CONFIG, **kwargs
) -> tuple[HwCorpusIndex, list[Path]]:
    """Index ``root``, build every cache, and return both."""
    index = build_hw_index(root, progress_every=0)
    return index, prepare_hw_cache(index, config=config, log_every=0, **kwargs)


def _key_of(ref) -> int:
    """The identifier a cache stores for ``ref``."""
    return int(ref.key) if ref.key is not None else int(ref.position)


def _slots_by_ref(index: HwCorpusIndex) -> dict[tuple[Path, int], int]:
    """Map ``(pickle, position)`` to that record's slot in the cache.

    A cache stores one slot per *successfully materialised* ref, in index order,
    so a ref's slot is its rank among its own pickle's cached refs. That rank is
    not ``ref.position`` (excluded records shift it) and not ``ref.key`` (dict
    payloads are sorted by ``_dict_sort_key``). Both happen to coincide on the
    simple synthetic corpus, which is exactly why addressing by them slips
    through; address by rank so the test pins the real invariant.
    """
    slots: dict[tuple[Path, int], int] = {}
    counters: dict[Path, int] = {}
    for ref in index.records:
        source = Path(ref.path).resolve()
        rank = counters.get(source, 0)
        counters[source] = rank + 1
        slots[(source, int(ref.position))] = rank
    return slots


class TestBitIdentity:
    """The cache must be a pure storage optimisation."""

    def test_every_record_matches_the_direct_path_exactly(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        index, directories = _build(tmp_path)
        assert directories, "no cache directory was built"
        assert len(directories) == len({Path(r.path).resolve() for r in index.records})

        families: dict[Path, HwCachedSource] = {
            Path(d).parent.parent: load_hw_cache(d) for d in directories
        }
        materialiser = Materialiser(config=_CONFIG, cache_size=1, use_cache=False)
        slots = _slots_by_ref(index)
        try:
            for ref in index.records:
                source = Path(ref.path).resolve()
                family = families[source.parent]
                direct = materialiser.materialise(ref)
                i = slots[(source, int(ref.position))]
                # keys.npy must also let a caller find the slot from the stored key.
                assert family.position_of(_key_of(ref)) == i, ref.path
                # Exact equality, not allclose: the cache stores the direct result.
                assert np.array_equal(family.phase_cos(i), direct.phase_cos), ref.path
                assert np.array_equal(family.phase_sin(i), direct.phase_sin), ref.path
                assert np.array_equal(family.image(i), direct.image), ref.path
                assert family.exposure_ms(i) == direct.exposure_ms, ref.path
                assert family.grid == _CONFIG.grid
        finally:
            for family in families.values():
                family.close()

    def test_default_config_is_applied_when_omitted(self, tmp_path: Path) -> None:
        """``prepare_hw_cache(index)`` must build, not pass ``None`` through.

        The default has to be materialised before any helper touches ``config.grid``
        or bakes it into meta; an omitted ``config`` used to reach the helpers as
        ``None``.
        """
        _mixed_corpus(tmp_path)
        index = build_hw_index(tmp_path, progress_every=0)
        directories = prepare_hw_cache(index, log_every=0)
        assert directories, "no cache directory was built"
        meta = json.loads((directories[0] / "meta.json").read_text(encoding="utf-8"))
        assert meta["grid"] == MaterialiserConfig().grid
        for directory in directories:
            load_hw_cache(directory).close()

    def test_all_four_sources_are_covered(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        index, directories = _build(tmp_path)
        counts = index.counts_by_source()
        assert counts[PhaseSource.PANEL_RAD] == 3
        assert counts[PhaseSource.PANEL_GRAY] == 3
        # 2 explicit Zernike records + the 2 inside the Recorder payload.
        assert counts[PhaseSource.ZERNIKE] == 4
        assert counts[PhaseSource.FREEFORM] == 2
        for directory in directories:
            load_hw_cache(directory).close()


class TestLayout:
    """Files, meta and accessors."""

    def test_writes_exactly_the_declared_files(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        _, directories = _build(tmp_path)
        for directory in directories:
            assert {p.name for p in directory.iterdir()} == set(CACHE_FILENAMES)

    def test_meta_records_every_config_field(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        _, directories = _build(tmp_path)
        meta = json.loads((directories[0] / "meta.json").read_text(encoding="utf-8"))
        assert meta["format_version"] == CACHE_FORMAT_VERSION
        assert meta["grid"] == _CONFIG.grid
        assert meta["n_records"] >= 1
        assert Path(meta["source"]).is_absolute()
        # Every config-affecting field must be recorded, or a config change could
        # silently reuse an incompatible cache.
        for field in (
            "slm_max_gray",
            "panel_center",
            "panel_radius",
            "panel_resolution",
            "image_mode",
            "zernike_radius",
            "freeform_radius",
        ):
            assert field in meta, field
        source = load_hw_cache(directories[0])
        assert len(source) == meta["n_records"]
        source.close()

    def test_accessors_own_their_data(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        _, directories = _build(tmp_path)
        source = load_hw_cache(directories[0])
        try:
            first = source.phase_cos(0)
            assert first.flags.owndata is True
            assert np.array_equal(first, source.phase_cos(0))
            first[:] = 12345.0  # mutating must not corrupt the mapping
            assert not np.array_equal(first, source.phase_cos(0))
            assert source.image(0).shape == (_CONFIG.grid, _CONFIG.grid)
            assert source.keys().dtype == np.dtype(np.int64)
            assert source.position_of(int(source.keys()[0])) == 0
        finally:
            source.close()

    def test_close_releases_and_is_idempotent(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        _, directories = _build(tmp_path)
        source = load_hw_cache(directories[0])
        before = source.phase_cos(0)  # an owned copy survives the close
        source.close()
        source.close()
        for accessor in (source.phase_cos, source.phase_sin, source.image):
            with pytest.raises(HwCacheError, match="closed"):
                accessor(0)
        with pytest.raises(HwCacheError, match="closed"):
            source.keys()
        with pytest.raises(HwCacheError, match="closed"):
            source.exposure_ms(0)
        assert before.shape == (_CONFIG.grid, _CONFIG.grid)

    def test_out_of_range_and_unknown_key(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        _, directories = _build(tmp_path)
        source = load_hw_cache(directories[0])
        try:
            with pytest.raises(IndexError):
                source.phase_cos(len(source))
            with pytest.raises(IndexError):
                source.phase_cos(-1)
            with pytest.raises(KeyError):
                source.position_of(999_999)
        finally:
            source.close()

    def test_recorder_payload_keys_are_positions(self, tmp_path: Path) -> None:
        """A Recorder payload has no integer key; the position is the identity."""
        _mixed_corpus(tmp_path)
        index, directories = _build(tmp_path)
        recorder_refs = [r for r in index.records if r.key is None]
        assert recorder_refs, "the Recorder payload produced no position-addressed refs"
        # cache dir is <family>/<stamp>/.hw_cache/<stem>, so the family name is
        # three levels up from the cache directory itself.
        candidates = [
            d for d in directories if Path(d).parent.parent.parent.name == "slm_pib_online"
        ]
        assert candidates, [str(d) for d in directories]
        family = load_hw_cache(candidates[0])
        try:
            for ref in recorder_refs:
                assert family.position_of(int(ref.position)) >= 0
        finally:
            family.close()


class TestStaleness:
    """A cache must never be reused when it would change the numbers."""

    def test_changed_grid_rebuilds(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        _, directories = _build(tmp_path)
        assert load_hw_cache(directories[0]).grid == 8
        coarse = MaterialiserConfig(
            grid=4,
            panel_center=(30, 20),
            panel_radius=15,
            panel_resolution=(60, 40),
            zernike_radius=10.0,
            freeform_radius=10.0,
        )
        _, rebuilt = _build(tmp_path, config=coarse)
        source = load_hw_cache(rebuilt[0])
        try:
            assert source.grid == 4
            assert source.image(0).shape == (4, 4)
        finally:
            source.close()

    def test_changed_slm_max_gray_rebuilds(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        _, directories = _build(tmp_path)
        before = json.loads((directories[0] / "meta.json").read_text(encoding="utf-8"))
        other = MaterialiserConfig(
            grid=8,
            slm_max_gray=1023,
            panel_center=(30, 20),
            panel_radius=15,
            panel_resolution=(60, 40),
            zernike_radius=10.0,
            freeform_radius=10.0,
        )
        _, rebuilt = _build(tmp_path, config=other)
        after = json.loads((rebuilt[0] / "meta.json").read_text(encoding="utf-8"))
        assert before["slm_max_gray"] == 993
        assert after["slm_max_gray"] == 1023

    def test_stale_version_is_rejected_then_rebuilt(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        _, directories = _build(tmp_path)
        meta_path = directories[0] / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["format_version"] = CACHE_FORMAT_VERSION + 999
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(HwCacheError, match="format_version"):
            load_hw_cache(directories[0])
        _, rebuilt = _build(tmp_path)
        load_hw_cache(rebuilt[0]).close()

    def test_missing_meta_is_rejected(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        _, directories = _build(tmp_path)
        (directories[0] / "meta.json").unlink()
        with pytest.raises(HwCacheError, match="meta.json"):
            load_hw_cache(directories[0])

    def test_truncated_array_is_rejected_then_rebuilt(self, tmp_path: Path) -> None:
        """A half-written .npy must be reported, never silently reshaped."""
        _mixed_corpus(tmp_path)
        _, directories = _build(tmp_path)
        target = directories[0] / "image.npy"
        with target.open("r+b") as handle:
            handle.truncate(target.stat().st_size - 32)
        with pytest.raises(HwCacheError):
            load_hw_cache(directories[0])
        _, rebuilt = _build(tmp_path)
        load_hw_cache(rebuilt[0]).close()

    def test_reuse_does_not_rewrite(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        _, directories = _build(tmp_path)
        stamp = (directories[0] / "phase_cos.npy").stat().st_mtime_ns
        _, again = _build(tmp_path)
        assert (again[0] / "phase_cos.npy").stat().st_mtime_ns == stamp

    def test_reuse_false_rebuilds(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        _, directories = _build(tmp_path)
        stamp = (directories[0] / "phase_cos.npy").stat().st_mtime_ns
        _, again = _build(tmp_path, reuse=False)
        assert (again[0] / "phase_cos.npy").stat().st_mtime_ns != stamp


class TestResilience:
    """One bad family must not cost the corpus."""

    def test_corrupt_pickle_is_skipped(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        broken = tmp_path / "slm_pib_x" / _STAMP / "broken.pkl"
        broken.write_bytes(b"not a pickle at all")
        index, directories = _build(tmp_path)
        assert directories, "a single corrupt file cost every family"
        for directory in directories:
            load_hw_cache(directory).close()
        # The index skips the unreadable file rather than raising.
        assert index.files_scanned >= 6

    def test_empty_index_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="HwCorpusIndex"):
            prepare_hw_cache(object())  # type: ignore[arg-type]

    def test_missing_exposure_becomes_nan_then_none(self, tmp_path: Path) -> None:
        """A record whose exposure cannot be resolved stores NaN, read back None."""
        record = _panel_gray(0)
        del record["exp_t"]
        _write(tmp_path / "slm_pib_x", "gray", {0: record})
        index = build_hw_index(tmp_path, progress_every=0)
        assert len(index) == 1 and index.records[0].exposure_ms is None
        directories = prepare_hw_cache(index, config=_CONFIG, log_every=0)
        source = load_hw_cache(directories[0])
        try:
            assert len(source) == 1
            assert source.exposure_ms(0) is None
            assert np.isnan(np.load(directories[0] / "exposure.npy")[0])
        finally:
            source.close()


class TestPaths:
    """Cache directory placement."""

    def test_default_and_explicit_root(self, tmp_path: Path) -> None:
        pkl = tmp_path / "a" / "b.pkl"
        assert cache_dir_for(pkl) == tmp_path / "a" / ".hw_cache" / "b"
        assert cache_dir_for(pkl, tmp_path / "elsewhere") == tmp_path / "elsewhere" / "b"

    def test_cache_is_invisible_to_the_pickle_glob(self, tmp_path: Path) -> None:
        """A built cache must not be re-discovered as corpus data."""
        _mixed_corpus(tmp_path)
        before = len(list(tmp_path.rglob("*.pkl")))
        _build(tmp_path)
        assert len(list(tmp_path.rglob("*.pkl"))) == before
        assert any(p.name == ".hw_cache" for p in tmp_path.rglob(".hw_cache"))


def test_exposure_survives_a_round_trip_exactly(tmp_path: Path) -> None:
    """Exposure must not be narrowed on the way through the cache.

    The bit-identity test above cannot catch this: its exposures (1.0 / 2.0 / 0.5)
    are exactly representable in float32, so a float32 cache looks perfect until a
    real record with, say, 0.1 ms comes back as 0.10000000149011612. Found on the
    real corpus, where 1010 of 1010 samples disagreed on this field alone.
    """
    record = _panel_gray(0)
    record["exp_t"] = 0.1  # not exactly representable in float32
    _write(tmp_path / "slm_pib_x", "gray", {0: record})
    index = build_hw_index(tmp_path, progress_every=0)
    assert index.records[0].exposure_ms == 0.1

    directories = prepare_hw_cache(index, config=_CONFIG, log_every=0)
    served = Materialiser(config=_CONFIG, cache_size=1, use_cache=True)
    try:
        got = served.materialise(index.records[0])
        assert got.exposure_ms == 0.1, f"exposure lost precision: {got.exposure_ms!r}"
        assert repr(got.exposure_ms) == repr(0.1)
    finally:
        served.clear()
    for directory in directories:
        load_hw_cache(directory).close()


class TestCacheIsReadBack:
    """The loader must actually *use* the cache, not merely be able to build it.

    Without this the cache would be write-only decoration: the whole point is that
    an epoch stops re-evaluating Zernike panels and re-reading multi-GB pickles.
    """

    def test_cached_materialise_is_bit_identical_to_direct(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        index, directories = _build(tmp_path)
        assert directories

        direct = Materialiser(config=_CONFIG, cache_size=1, use_cache=False)
        served = Materialiser(config=_CONFIG, cache_size=1, use_cache=True)
        try:
            assert served.cached_pickles == (), "nothing should be mapped before first use"
            for ref in index.records:
                want = direct.materialise(ref)
                got = served.materialise(ref)
                assert np.array_equal(want.phase_cos, got.phase_cos), ref.path
                assert np.array_equal(want.phase_sin, got.phase_sin), ref.path
                assert np.array_equal(want.image, got.image), ref.path
                assert np.array_equal(want.contrast, got.contrast), ref.path
                assert want.exposure_ms == got.exposure_ms, ref.path
                assert want.fov_px == got.fov_px, ref.path
            # The cache was consulted, and the open mappings stay bounded by
            # cache_size (records are visited across several pickles in turn, so
            # an LRU of 1 ends with exactly one family mapped).
            assert served.cached_pickles, "the built cache was never read"
            assert len(served.cached_pickles) == 1
        finally:
            direct.clear()
            served.clear()
        assert served.cached_pickles == (), "clear() must unmap every cache"

    def test_cached_sample_owns_its_arrays(self, tmp_path: Path) -> None:
        """A served sample must not alias the memory map it came from."""
        _mixed_corpus(tmp_path)
        index, _ = _build(tmp_path)
        served = Materialiser(config=_CONFIG, cache_size=1, use_cache=True)
        try:
            first = served.materialise(index.records[0])
            cos = first.phase_cos
            assert cos.flags.owndata and cos.base is None
            # Mutating it must not corrupt the cache for the next reader.
            cos[:] = 123.0
            again = Materialiser(config=_CONFIG, cache_size=1, use_cache=True)
            try:
                assert not np.array_equal(
                    again.materialise(index.records[0]).phase_cos, cos
                )
            finally:
                again.clear()
        finally:
            served.clear()

    def test_absent_cache_falls_back_to_the_direct_path(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        index = build_hw_index(tmp_path, progress_every=0)
        served = Materialiser(config=_CONFIG, cache_size=1, use_cache=True)
        try:
            assert served.cached_pickles == ()
            sample = served.materialise(index.records[0])
            assert sample.image.shape == (_CONFIG.grid, _CONFIG.grid)
            assert served.cached_pickles == (), "no cache exists, so none may be mapped"
        finally:
            served.clear()

    def test_cache_built_with_another_config_is_not_served(self, tmp_path: Path) -> None:
        """A stale cache must be ignored, never served as if it were current.

        ``image_mode`` is the discriminator here rather than ``grid``: it changes
        the *values* while keeping the geometry valid, so a wrongly-served cache
        would hand back differently-scaled images that still look entirely
        plausible. (A ``grid`` change would also work but makes the synthetic
        panels' ROI geometrically invalid at the larger size.)
        """
        _mixed_corpus(tmp_path)
        _, built = _build(tmp_path)
        assert built

        index = build_hw_index(tmp_path, progress_every=0)
        other = replace(_CONFIG, image_mode="peak")
        assert other != _CONFIG
        served = Materialiser(config=other, cache_size=1, use_cache=True)
        direct = Materialiser(config=other, cache_size=1, use_cache=False)
        try:
            assert served.cached_pickles == (), "a mismatched cache must not be mapped"
            for ref in index.records:
                got = served.materialise(ref)
                assert got.image.shape == (other.grid, other.grid)
                assert np.array_equal(got.image, direct.materialise(ref).image)
                assert served.cached_pickles == ()
        finally:
            served.clear()
            direct.clear()

    def test_use_cache_false_never_touches_the_cache(self, tmp_path: Path) -> None:
        _mixed_corpus(tmp_path)
        index, _ = _build(tmp_path)
        served = Materialiser(config=_CONFIG, cache_size=1, use_cache=False)
        try:
            for ref in index.records:
                served.materialise(ref)
            assert served.cached_pickles == ()
        finally:
            served.clear()

    def test_materialiser_with_open_cache_is_picklable(self, tmp_path: Path) -> None:
        """Windows ``spawn`` re-pickles the Dataset per worker.

        A ``numpy`` memory map cannot be pickled at all, so the open caches must be
        dropped by ``__getstate__`` rather than carried across.
        """
        _mixed_corpus(tmp_path)
        index, _ = _build(tmp_path)
        served = Materialiser(config=_CONFIG, cache_size=1, use_cache=True)
        served.materialise(index.records[0])
        assert served.cached_pickles, "precondition: a cache is mapped"

        revived = pickle.loads(pickle.dumps(served))
        try:
            assert revived.cached_pickles == ()
            assert revived.config == served.config
            assert revived.use_cache is True
            # And it still works, by re-reading the cache lazily.
            assert revived.materialise(index.records[0]).image.shape == (
                _CONFIG.grid,
                _CONFIG.grid,
            )
        finally:
            revived.clear()
            served.clear()
