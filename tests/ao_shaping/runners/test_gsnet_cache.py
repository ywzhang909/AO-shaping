"""Tests for the lean, mmap-able GSNet corpus cache.

Hardware-free and corpus-free: every test builds a handful of synthetic pickles
in ``tmp_path`` in the real two-level nested layout
(``<family>/<timestamp>/<prefix>_<timestamp>.pkl``) that
:func:`~ao_shaping.runners.gsnet_offline.build_record_index` globs
recursively. ``grid=64`` matches the production default.

The load-bearing test is
:meth:`TestBitIdenticalTrainingTriples.test_cache_and_pickle_agree_bit_for_bit`:
it walks every record through both the cache-backed and the ``pickle.load``-
backed Dataset and asserts ``torch.equal`` on ``source``, ``target`` **and**
``gt_phase``. That is what licenses calling the cache a pure storage
optimisation.

A deliberate deviation from the original spec, pinned by
:meth:`TestOnDiskLayout.test_coefficients_are_float64_for_bit_fidelity`:
coefficients are cached as **float64**, not float32. ``pupil_phase_to_grid``
evaluates ``exp(1j * phase)`` on a 64x64 down-sample, so a float32 round trip
perturbs those sums by up to ~1.2e-07 rad (128 float32 ULPs) and changes ~5 % of
the output pixels -- which would break bit-identity on real (i.e.
non-float32-representable) hardware coefficients. float64 is the narrowest
dtype that preserves it. See
:meth:`TestOnDiskLayout.test_synthetic_coefficients_are_not_float32_representable`
for the other half of that argument: the synthetic corpus is deliberately
float32-lossy, so the regression cannot pass by accident.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from loguru import logger

from ao_shaping.runners.gsnet_cache import (
    CACHE_FILENAMES,
    CACHE_FORMAT_VERSION,
    C_DTYPE,
    DEFAULT_CACHE_DIR_NAME,
    IMG_DTYPE,
    KEY_DTYPE,
    META_FILENAME,
    OFFSET_DTYPE,
    SHAPE_DTYPE,
    CachedRecord,
    GSNetCacheError,
    cache_dir_for,
    close_cached_payload,
    load_cached_family,
    prepare_gsnet_cache,
)
from ao_shaping.runners.gsnet_dataset import GSNetDebugDataset
from ao_shaping.runners.gsnet_offline import RecordIndex

GRID = 64
SLM_WIDTH = 1920
SLM_HEIGHT = 1200
# The corpus device's ZERNIKE_APERTURE_RADIUS (slm_zernike_pib.py:235) --
# deliberately NOT min(SLM_WIDTH, SLM_HEIGHT) / 2.
SLM_RADIUS = 300.0

# Observed ``len(_c)`` values in the real corpus -> must never be hardcoded.
MODE_COUNTS = (15, 66, 78)


# ---------------------------------------------------------------------------
# Synthetic corpus helpers
# ---------------------------------------------------------------------------
def _far_field_frame(row: int, height: int = 100, width: int = 120) -> np.ndarray:
    """A deterministic far-field frame with an off-centre 0-order blob.

    The blob centre moves with ``row`` so different records get different
    ``argmax`` anchors -- the optical axis is *not* the frame centre on a 2f
    bench, and ``farfield_to_grid`` has to cope with that.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cx = 30.0 + 0.7 * (row % 7)
    cy = 40.0 + 1.1 * (row % 5)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    frame = 250.0 * np.exp(-r2 / (2 * 6.0**2))
    frame[10:14, 90:94] += 40.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _zernike_coefficients(n_terms: int, row: int) -> np.ndarray:
    """Deterministic Noll-order coefficients with a known leading structure.

    Index 3 (Noll 4, defocus) and index 10 (Noll 11, spherical) carry amplitude
    so ``gt_phase`` is never an all-zero plane.
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
    *,
    mode_counts: tuple[int, ...] = MODE_COUNTS,
    img_shapes: tuple[tuple[int, int], ...] | None = None,
) -> Path:
    """Write one debug-style pickle in the real two-level nested layout.

    Args:
        root: Directory the ``<family>/<timestamp>/`` tree is created under.
        family: Family name (also the filename prefix).
        tag: Short run tag, so several pickles can share one family.
        mode_counts: ``len(_c)`` per record, cycling if shorter than the record
            count. Mixed values exercise the ragged offset layout.
        img_shapes: ``(height, width)`` per record, cycling if needed. Defaults
            to a single uniform shape.

    Returns:
        The path of the written pickle.
    """
    n_records = max(len(mode_counts), 3)
    shapes = img_shapes or ((100, 120),)

    timestamp = "20260926_164358"
    directory = root / f"{family}_{tag}" / timestamp
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{family}_{tag}_{timestamp}.pkl"

    records: dict[int, dict[str, Any]] = {}
    for row in range(n_records):
        height, width = shapes[row % len(shapes)]
        n_terms = mode_counts[row % len(mode_counts)]
        records[row] = {
            "_img": _far_field_frame(row, height=height, width=width),
            "_c": _zernike_coefficients(n_terms, row),
            # Bulk field the cache deliberately drops; its presence must not
            # change a single cached value.
            "_phase": np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=np.uint16),
        }
    with open(path, "wb") as handle:
        pickle.dump(records, handle)
    return path


def _read_pickle(path: Path) -> dict[int, dict[str, Any]]:
    """Read a synthetic pickle straight from disk (the reference path)."""
    with open(path, "rb") as handle:
        return pickle.load(handle)  # noqa: S301 - test fixture written above


def _index_for(path: Path) -> RecordIndex:
    """A single-file :class:`RecordIndex` covering every record in ``path``."""
    return RecordIndex(entries=tuple((path, key) for key in sorted(_read_pickle(path))))


@contextmanager
def _captured_warnings() -> Iterator[list[str]]:
    """Collect ``loguru`` WARNING records emitted inside the block."""
    messages: list[str] = []

    def sink(message: Any) -> None:
        messages.append(str(message))

    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(handler_id)


def _dataset(index: RecordIndex, **kwargs: Any) -> GSNetDebugDataset:
    """A Dataset with the production corpus geometry."""
    return GSNetDebugDataset(
        index,
        grid=GRID,
        slm_width=SLM_WIDTH,
        slm_height=SLM_HEIGHT,
        slm_radius=SLM_RADIUS,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. On-disk layout
# ---------------------------------------------------------------------------
class TestOnDiskLayout:
    """The exact file set, dtypes and shapes written per source pickle."""

    def test_creates_exact_file_set_with_documented_dtypes(
        self, tmp_path: Path
    ) -> None:
        """Every documented file exists -- and nothing else does."""
        path = _write_pickle(tmp_path, "slm_pib", "layout")
        source = _read_pickle(path)
        n_records = len(source)
        n_terms = sum(len(record["_c"]) for record in source.values())
        heights = [record["_img"].shape[0] for record in source.values()]
        widths = [record["_img"].shape[1] for record in source.values()]

        (directory,) = prepare_gsnet_cache(_index_for(path))

        assert directory == cache_dir_for(path)
        assert directory.parent.name == DEFAULT_CACHE_DIR_NAME
        assert {p.name for p in directory.iterdir()} == set(CACHE_FILENAMES)

        expected_shape = {
            "c_flat.npy": (n_terms,),
            "c_offsets.npy": (n_records + 1,),
            "img_flat.npy": (sum(h * w for h, w in zip(heights, widths)),),
            "img_offsets.npy": (n_records + 1,),
            "img_shapes.npy": (n_records, 2),
            "keys.npy": (n_records,),
        }
        expected_dtype = {
            "c_flat.npy": C_DTYPE,
            "c_offsets.npy": OFFSET_DTYPE,
            "img_flat.npy": IMG_DTYPE,
            "img_offsets.npy": OFFSET_DTYPE,
            "img_shapes.npy": SHAPE_DTYPE,
            "keys.npy": KEY_DTYPE,
        }
        for name, shape in expected_shape.items():
            array = np.load(directory / name, mmap_mode="r", allow_pickle=False)
            assert array.shape == shape, name
            assert array.dtype == expected_dtype[name], name
            # 1-D flat arrays are what make offset slicing (and mmap paging)
            # work at all.
            if name.endswith("_flat.npy"):
                assert array.ndim == 1, name

    def test_meta_json_contents(self, tmp_path: Path) -> None:
        """``meta.json`` carries the source, record count, version and stamp."""
        path = _write_pickle(tmp_path, "slm_pib", "meta")
        (directory,) = prepare_gsnet_cache(_index_for(path))

        meta = json.loads((directory / META_FILENAME).read_text(encoding="utf-8"))
        assert meta["source"] == str(path.resolve())
        assert meta["n_records"] == len(_read_pickle(path))
        assert meta["format_version"] == CACHE_FORMAT_VERSION
        assert isinstance(meta["created"], str) and meta["created"]

    def test_coefficients_are_float64_for_bit_fidelity(self, tmp_path: Path) -> None:
        """Pin the float64 deviation, and the reason for it, in code.

        The spec asked for float32. float32 cannot satisfy the stronger
        bit-identity requirement, so the cache stores float64. This test is the
        executable form of that decision.
        """
        path = _write_pickle(tmp_path, "slm_pib", "f64")
        (directory,) = prepare_gsnet_cache(_index_for(path))

        cached = np.load(directory / "c_flat.npy", mmap_mode="r", allow_pickle=False)
        assert cached.dtype == np.dtype(np.float64)
        assert C_DTYPE == np.dtype(np.float64)

    def test_synthetic_coefficients_are_not_float32_representable(
        self, tmp_path: Path
    ) -> None:
        """The corpus must be float32-lossy, or the deviation is unjustified.

        If every coefficient survived a float32 round trip, a float32 cache
        would also be bit-identical here and the tests above would prove
        nothing about the real corpus.
        """
        path = _write_pickle(tmp_path, "slm_pib", "lossy")
        (directory,) = prepare_gsnet_cache(_index_for(path))

        flat = np.asarray(
            np.load(directory / "c_flat.npy", mmap_mode="r", allow_pickle=False)
        )
        round_tripped = flat.astype(np.float32).astype(np.float64)
        assert not np.array_equal(flat, round_tripped)
        assert float(np.max(np.abs(flat - round_tripped))) > 0.0


# ---------------------------------------------------------------------------
# 2. Round-trip fidelity
# ---------------------------------------------------------------------------
class TestRoundTrip:
    """Ragged coefficients and non-uniform images survive exactly."""

    def test_ragged_coefficients_survive_exactly(self, tmp_path: Path) -> None:
        """15 / 66 / 78 terms in one file, with no padding and no truncation."""
        path = _write_pickle(tmp_path, "slm_pib", "ragged")
        source = _read_pickle(path)
        (directory,) = prepare_gsnet_cache(_index_for(path))
        family = load_cached_family(directory, mmap=False)

        assert family.n_records == len(source)
        keys = sorted(source)
        assert [family.n_terms(i) for i in range(family.n_records)] == [
            len(source[key]["_c"]) for key in keys
        ]

        # The offsets must tile the flat array exactly -- no gap, no overlap.
        offsets = np.load(directory / "c_offsets.npy", allow_pickle=False)
        assert int(offsets[0]) == 0
        assert np.all(np.diff(offsets) > 0)
        assert int(offsets[-1]) == np.load(
            directory / "c_flat.npy", mmap_mode="r", allow_pickle=False
        ).size

        for i, key in enumerate(keys):
            coeffs = family.coeffs(i)
            assert coeffs.ndim == 1
            assert coeffs.size == len(source[key]["_c"])
            np.testing.assert_array_equal(coeffs, source[key]["_c"])
        family.close()

    def test_non_uniform_image_shapes_round_trip_exactly(
        self, tmp_path: Path
    ) -> None:
        """Three different ``(H, W)`` in a single source pickle."""
        shapes = ((100, 120), (73, 91), (128, 64))
        path = _write_pickle(tmp_path, "slm_pib", "shapes", img_shapes=shapes)
        source = _read_pickle(path)
        (directory,) = prepare_gsnet_cache(_index_for(path))

        stored = np.load(directory / "img_shapes.npy", allow_pickle=False)
        assert [tuple(int(v) for v in stored[i]) for i in range(len(source))] == [
            source[key]["_img"].shape for key in sorted(source)
        ]

        family = load_cached_family(directory, mmap=False)
        for i, key in enumerate(sorted(source)):
            expected = source[key]["_img"]
            image = family.image(i)
            assert image.shape == expected.shape
            assert image.dtype == IMG_DTYPE
            np.testing.assert_array_equal(image, expected)
        family.close()

    def test_keys_stay_traceable_to_the_source_pickle(self, tmp_path: Path) -> None:
        """``keys.npy`` holds the pickle's own keys."""
        path = _write_pickle(tmp_path, "slm_pib", "keys")
        source = _read_pickle(path)
        (directory,) = prepare_gsnet_cache(_index_for(path))

        family = load_cached_family(directory, mmap=False)
        assert sorted(family.keys().tolist()) == sorted(source)
        family.close()

    def test_per_record_slices_are_owned_copies(self, tmp_path: Path) -> None:
        """A slice must not keep the mapping alive, nor alias its neighbours.

        ``np.asarray(memmap[a:b])`` returns a *view* into the mapping, so a
        caller that retained it would pin the whole file in RAM. The cache
        copies instead.
        """
        path = _write_pickle(tmp_path, "slm_pib", "owned")
        (directory,) = prepare_gsnet_cache(_index_for(path))
        family = load_cached_family(directory, mmap=True)

        image = family.image(0)
        assert isinstance(image, np.ndarray)
        assert not isinstance(image, np.memmap)

        before = image.copy()
        image[:] = 0
        # A later accessor must be unaffected by the caller's mutation.
        np.testing.assert_array_equal(family.image(0), before)

        coeffs = family.coeffs(0)
        assert not isinstance(coeffs, np.memmap)
        n_terms = coeffs.size
        coeffs[:] = 0.0
        assert family.n_terms(0) == n_terms
        family.close()


# ---------------------------------------------------------------------------
# 3. Idempotency
# ---------------------------------------------------------------------------
class TestIdempotency:
    """A second ``prepare_gsnet_cache`` must not re-read the pickle."""

    def test_second_call_reuses_cache_without_pickling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No ``pickle.load`` on the second call, and byte-identical arrays."""
        path = _write_pickle(tmp_path, "slm_pib", "idem")
        index = _index_for(path)
        (first,) = prepare_gsnet_cache(index)
        snapshot = {
            name: (first / name).read_bytes()
            for name in CACHE_FILENAMES
            if name != META_FILENAME
        }

        calls = 0
        real_load = pickle.load

        def counting_load(handle: Any) -> Any:
            nonlocal calls
            calls += 1
            return real_load(handle)

        monkeypatch.setattr(pickle, "load", counting_load)
        (second,) = prepare_gsnet_cache(index)

        assert calls == 0, "the second prepare_gsnet_cache re-read the pickle"
        assert second == first
        for name, blob in snapshot.items():
            assert (second / name).read_bytes() == blob, name

    def test_stale_format_version_is_rebuilt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cache from another format version is discarded, not reused."""
        path = _write_pickle(tmp_path, "slm_pib", "ver")
        index = _index_for(path)
        (directory,) = prepare_gsnet_cache(index)

        meta_path = directory / META_FILENAME
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["format_version"] = CACHE_FORMAT_VERSION + 1
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        calls = 0
        real_load = pickle.load

        def counting_load(handle: Any) -> Any:
            nonlocal calls
            calls += 1
            return real_load(handle)

        monkeypatch.setattr(pickle, "load", counting_load)
        prepare_gsnet_cache(index)

        assert calls == 1
        reloaded = json.loads(meta_path.read_text(encoding="utf-8"))
        assert reloaded["format_version"] == CACHE_FORMAT_VERSION


# ---------------------------------------------------------------------------
# 4. load_cached_family
# ---------------------------------------------------------------------------
class TestLoadCachedFamily:
    """mmap vs eager loading, and lifecycle safety."""

    def test_mmap_and_eager_agree(self, tmp_path: Path) -> None:
        """Both load modes yield equal arrays for every record."""
        path = _write_pickle(tmp_path, "slm_pib", "mmap")
        source = _read_pickle(path)
        (directory,) = prepare_gsnet_cache(_index_for(path))

        mapped = load_cached_family(directory, mmap=True)
        eager = load_cached_family(directory, mmap=False)
        try:
            assert mapped.n_records == eager.n_records == len(source)
            np.testing.assert_array_equal(mapped.keys(), eager.keys())
            for i in range(mapped.n_records):
                np.testing.assert_array_equal(mapped.coeffs(i), eager.coeffs(i))
                np.testing.assert_array_equal(mapped.image(i), eager.image(i))
                assert mapped.n_terms(i) == eager.n_terms(i)
        finally:
            mapped.close()
            eager.close()

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        """``close()`` twice must not raise."""
        path = _write_pickle(tmp_path, "slm_pib", "close")
        (directory,) = prepare_gsnet_cache(_index_for(path))
        family = load_cached_family(directory, mmap=True)
        family.close()
        family.close()

    def test_access_after_close_raises(self, tmp_path: Path) -> None:
        """A closed family fails loudly instead of returning freed memory."""
        path = _write_pickle(tmp_path, "slm_pib", "closed")
        (directory,) = prepare_gsnet_cache(_index_for(path))
        family = load_cached_family(directory, mmap=True)
        family.close()
        with pytest.raises(GSNetCacheError):
            family.image(0)

    def test_payload_is_a_dict_of_dicts(self, tmp_path: Path) -> None:
        """``as_payload()`` keeps the LRU contract: a ``dict`` of ``dict`` s."""
        path = _write_pickle(tmp_path, "slm_pib", "payload")
        source = _read_pickle(path)
        (directory,) = prepare_gsnet_cache(_index_for(path))
        family = load_cached_family(directory, mmap=True)
        payload: dict[int, Any] = {}
        try:
            payload = family.as_payload()
            assert isinstance(payload, dict)
            assert sorted(payload) == sorted(source)
            for key, record in payload.items():
                assert isinstance(record, CachedRecord)
                assert isinstance(record, dict)
                # A dropped field must look missing, exactly like a plain dict.
                with pytest.raises(KeyError):
                    record["_grad"]
                np.testing.assert_array_equal(record["_img"], source[key]["_img"])
        finally:
            close_cached_payload(payload)
            family.close()

    def test_missing_file_set_raises(self, tmp_path: Path) -> None:
        """A family with a deleted array is reported, not silently tolerated."""
        path = _write_pickle(tmp_path, "slm_pib", "gone")
        (directory,) = prepare_gsnet_cache(_index_for(path))
        (directory / "c_flat.npy").unlink()
        with pytest.raises(GSNetCacheError):
            load_cached_family(directory)


# ---------------------------------------------------------------------------
# 5. Dataset integration
# ---------------------------------------------------------------------------
class TestBitIdenticalTrainingTriples:
    """The core promise: the cache changes storage, never the sample."""

    def test_cache_and_pickle_agree_bit_for_bit(self, tmp_path: Path) -> None:
        """``(source, target, gt_phase)`` are ``torch.equal`` across both paths."""
        path = _write_pickle(tmp_path, "slm_pib", "triple")
        index = _index_for(path)
        prepare_gsnet_cache(index)

        cached = _dataset(index, use_cache=True)
        direct = _dataset(index, use_cache=False)
        try:
            assert len(cached) == len(direct) == len(index.entries)
            for i in range(len(index.entries)):
                got = cached[i]
                want = direct[i]
                for name, left, right in zip(
                    ("source", "target", "gt_phase"), got, want, strict=True
                ):
                    assert left.shape == right.shape == (1, GRID, GRID), name
                    assert left.dtype == right.dtype == torch.float32, name
                    delta = float((left - right).abs().max())
                    assert torch.equal(left, right), (
                        f"record {i}: '{name}' differs between the cache and "
                        f"the pickle path (max |delta| = {delta:.3e})"
                    )
        finally:
            cached.clear_cache()
            direct.clear_cache()

    def test_cached_dataset_never_unpickles(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a valid cache, ``pickle.load`` is not called at all."""
        import ao_shaping.runners.gsnet_dataset as dataset_module

        path = _write_pickle(tmp_path, "slm_pib", "nopickle")
        index = _index_for(path)
        prepare_gsnet_cache(index)

        calls = 0
        real_load = pickle.load

        def counting_load(handle: Any) -> Any:
            nonlocal calls
            calls += 1
            return real_load(handle)

        monkeypatch.setattr(dataset_module.pickle, "load", counting_load)
        cached = _dataset(index, use_cache=True)
        try:
            for i in range(len(index.entries)):
                assert cached[i][0].shape == (1, GRID, GRID)
        finally:
            cached.clear_cache()

        assert calls == 0


class TestCorruptCacheFallback:
    """A bad family warns, rebuilds and carries on -- it never raises."""

    @staticmethod
    def _truncate(directory: Path) -> None:
        """Halve ``c_flat.npy`` -- the classic truncated-download failure."""
        path = directory / "c_flat.npy"
        path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])

    @staticmethod
    def _expected(index: RecordIndex) -> tuple[torch.Tensor, ...]:
        """The reference triple, straight from ``pickle.load``."""
        direct = _dataset(index, use_cache=False)
        try:
            return direct[0]
        finally:
            direct.clear_cache()

    def test_truncated_c_flat_is_rebuilt(self, tmp_path: Path) -> None:
        """Truncated flat array -> WARNING + rebuild + the correct sample."""
        path = _write_pickle(tmp_path, "slm_pib", "trunc")
        index = _index_for(path)
        (directory,) = prepare_gsnet_cache(index)
        self._truncate(directory)
        expected = self._expected(index)

        with _captured_warnings() as warnings:
            cached = _dataset(index, use_cache=True)
            try:
                actual = cached[0]
            finally:
                cached.clear_cache()

        assert any("corrupt" in message for message in warnings), warnings
        assert any("rebuild" in message for message in warnings), warnings
        for name, left, right in zip(
            ("source", "target", "gt_phase"), actual, expected, strict=True
        ):
            assert torch.equal(left, right), name

    def test_corrupt_meta_json_is_rebuilt(self, tmp_path: Path) -> None:
        """Unparseable ``meta.json`` -> WARNING + rebuild, no exception."""
        path = _write_pickle(tmp_path, "slm_pib", "badmeta")
        index = _index_for(path)
        (directory,) = prepare_gsnet_cache(index)
        (directory / META_FILENAME).write_text("{not json", encoding="utf-8")
        expected = self._expected(index)

        with _captured_warnings() as warnings:
            cached = _dataset(index, use_cache=True)
            try:
                actual = cached[0]
            finally:
                cached.clear_cache()

        assert warnings, "a corrupt cache must warn"
        for name, left, right in zip(
            ("source", "target", "gt_phase"), actual, expected, strict=True
        ):
            assert torch.equal(left, right), name

    def test_absent_cache_warns_once_and_uses_pickle(self, tmp_path: Path) -> None:
        """No cache at all -> one warning per file, then correct samples.

        Building a multi-GB cache inside ``__getitem__`` would stall the first
        sample of every worker process, so absence falls back to
        ``pickle.load`` and asks the operator to build the cache up front.
        """
        path = _write_pickle(tmp_path, "slm_pib", "absent")
        index = _index_for(path)

        direct = _dataset(index, use_cache=False)
        try:
            expected = [direct[i] for i in range(len(index.entries))]
        finally:
            direct.clear_cache()

        with _captured_warnings() as warnings:
            cached = _dataset(index, use_cache=True)
            try:
                actual = [cached[i] for i in range(len(index.entries))]
            finally:
                cached.clear_cache()

        assert len(warnings) == 1, warnings
        assert "No GSNet cache" in warnings[0]
        for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
            for name, left, right in zip(
                ("source", "target", "gt_phase"), got, want, strict=True
            ):
                assert torch.equal(left, right), f"record {i}: {name}"


class TestClearCache:
    """``clear_cache()`` is an in-RAM operation only."""

    def test_clear_cache_keeps_on_disk_cache(self, tmp_path: Path) -> None:
        """Nothing on disk is removed, and the dataset keeps working."""
        path = _write_pickle(tmp_path, "slm_pib", "clear")
        index = _index_for(path)
        (directory,) = prepare_gsnet_cache(index)

        cached = _dataset(index, use_cache=True)
        first = cached[0]
        assert directory.is_dir()
        assert cached.cached_files == (path,)

        cached.clear_cache()

        assert cached.cached_files == ()
        assert directory.is_dir(), "clear_cache() must not delete on-disk data"
        assert {p.name for p in directory.iterdir()} == set(CACHE_FILENAMES)
        # The freed mappings were genuinely released, not just forgotten.
        for name, left, right in zip(
            ("source", "target", "gt_phase"), cached[0], first, strict=True
        ):
            assert torch.equal(left, right), name

    def test_lru_eviction_stays_bounded(self, tmp_path: Path) -> None:
        """With ``cache_size=1`` only one family is ever resident."""
        first_path = _write_pickle(tmp_path, "slm_pib", "a")
        second_path = _write_pickle(tmp_path, "slm_pib", "b")
        index = RecordIndex(
            entries=_index_for(first_path).entries + _index_for(second_path).entries
        )
        prepare_gsnet_cache(index)

        cached = _dataset(index, use_cache=True, cache_size=1)
        try:
            for i in range(len(index.entries)):
                assert cached[i][0].shape == (1, GRID, GRID)
            assert len(cached.cached_files) <= 1
        finally:
            cached.clear_cache()
