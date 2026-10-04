"""Offline tests for :mod:`ml.gsnet_debug.offline`.

Everything here is hardware-free and torch-free. The synthetic fixtures are
fully deterministic (no RNG), and the real-data tests exercise the actual
``data/debug`` pickles -- including the two-level-deep layout
(``<prefix>_<tag>/<timestamp>/<prefix>_<tag>_<timestamp>.pkl``) that a
non-recursive glob would silently miss.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from ml.gsnet_debug.offline import (
    DEFAULT_ROOTS,
    RecordIndex,
    build_record_index,
    farfield_to_grid,
    infer_n_max,
    pupil_phase_to_grid,
    reconstruct_pupil_phase_rad,
)

# tests/ao_shaping/ml/gsnet_debug/<this file> -> repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
DEBUG_DIR = REPO_ROOT / "data" / "debug"
REAL_ZERNIKE_GLOB = str(DEBUG_DIR / "slm_zernike_*")
REAL_PIB_GLOB = str(DEBUG_DIR / "slm_pib_*")

# Real SLM200 panel + the aperture radius the optimizers actually use.
# ZERNIKE_APERTURE_RADIUS = 300.0 (slm_zernike_pib.py:235). This is NOT
# min(w, h) / 2 (= 600) -- that value was a bug: it made the reconstructed
# Zernike support twice the physical aperture.
SLM_WIDTH = 1920
SLM_HEIGHT = 1200
SLM_RADIUS = 300.0


def _aperture_mask(height: int, width: int, radius: float) -> np.ndarray:
    """Reference aperture mask, mirroring ``zernike_calc._build_cached_grid``.

    ``ZernikeGenerator.mask`` normalises pixel offsets by the radius and treats
    ``sqrt(xv**2 + yv**2) <= 1.0`` as inside the aperture
    (``zernike_calc.py:492-499``).
    """
    ddx = (np.arange(width) - (width - 1) / 2.0) / radius
    ddy = (np.arange(height) - (height - 1) / 2.0) / radius
    xv, yv = np.meshgrid(ddx, ddy)
    return np.sqrt(xv**2 + yv**2) <= 1.0


def _write_records(path: Path, n_records: int) -> Path:
    """Write a debug-style pickle: a plain ``dict`` keyed by ``int``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = {
        i: {
            "_img": np.zeros((4, 5), dtype=np.uint8),
            "_c": np.zeros(66, dtype=np.float64),
            "_grad": np.zeros(66, dtype=np.float64),
        }
        for i in range(n_records)
    }
    with open(path, "wb") as handle:
        pickle.dump(records, handle)
    return path


def _block_mean_reference(phase: np.ndarray, grid: int) -> np.ndarray:
    """Independent reference for ``pupil_phase_to_grid``.

    Pads symmetrically to ``grid * ceil(length / grid)`` and averages each of
    the ``grid x grid`` equal blocks.
    """
    height, width = phase.shape
    block_h = -(-height // grid)
    block_w = -(-width // grid)
    pad_h, pad_w = grid * block_h - height, grid * block_w - width
    padded = np.zeros((grid * block_h, grid * block_w), dtype=np.float64)
    padded[pad_h // 2 : pad_h // 2 + height, pad_w // 2 : pad_w // 2 + width] = phase
    return np.array(
        [
            [padded[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w].mean() for j in range(grid)]
            for i in range(grid)
        ]
    )


# ---------------------------------------------------------------------------
# infer_n_max
# ---------------------------------------------------------------------------
class TestInferNMax:
    @pytest.mark.parametrize(
        ("n_terms", "expected"),
        [(15, 4), (66, 10), (78, 11), (1, 0), (3, 1), (6, 2), (10, 3), (45, 8), (55, 9)],
    )
    def test_known_mode_counts(self, n_terms: int, expected: int) -> None:
        assert infer_n_max(n_terms) == expected

    def test_exact_for_every_order_up_to_25(self) -> None:
        """The exact-integer inverse must round-trip for all triangular counts."""
        for n_max in range(0, 26):
            n_terms = (n_max + 1) * (n_max + 2) // 2
            assert infer_n_max(n_terms) == n_max

    @pytest.mark.parametrize("n_terms", [7, 2, 8, 100, 47])
    def test_non_triangular_raises(self, n_terms: int) -> None:
        with pytest.raises(ValueError):
            infer_n_max(n_terms)

    @pytest.mark.parametrize("n_terms", [0, -1, -15])
    def test_non_positive_raises(self, n_terms: int) -> None:
        with pytest.raises(ValueError):
            infer_n_max(n_terms)


# ---------------------------------------------------------------------------
# build_record_index -- synthetic
# ---------------------------------------------------------------------------
class TestBuildRecordIndexSynthetic:
    def test_finds_two_level_nested_layout(self, tmp_path: Path) -> None:
        """The real layout is two levels deep; a non-recursive glob finds nothing."""
        deep = tmp_path / "slm_zernike_shaping_run" / "20260926_164358"
        _write_records(deep / "slm_zernike_shaping_run_20260926_164358.pkl", 3)

        index = build_record_index(roots=str(tmp_path / "slm_zernike_*"))

        assert isinstance(index, RecordIndex)
        assert len(index.entries) == 3
        assert [key for _, key in index.entries] == [0, 1, 2]

    def test_flat_layout_also_found(self, tmp_path: Path) -> None:
        _write_records(tmp_path / "flat.pkl", 2)
        index = build_record_index(roots=str(tmp_path))
        assert len(index.entries) == 2

    def test_keys_are_sorted_ints(self, tmp_path: Path) -> None:
        path = tmp_path / "shuffled.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({7: {"_c": np.zeros(3)}, 2: {"_c": np.zeros(3)}, 5: {"_c": np.zeros(3)}}, handle)

        index = build_record_index(roots=str(tmp_path))

        assert [key for _, key in index.entries] == [2, 5, 7]
        assert all(isinstance(key, int) for _, key in index.entries)

    def test_explicit_roots_sequence(self, tmp_path: Path) -> None:
        _write_records(tmp_path / "a" / "one.pkl", 2)
        _write_records(tmp_path / "b" / "two.pkl", 3)

        index = build_record_index(roots=[tmp_path / "a", tmp_path / "b"])

        assert len({p for p, _ in index.entries}) == 2
        assert len(index.entries) == 5

    def test_limit_files_caps_file_count(self, tmp_path: Path) -> None:
        for i in range(4):
            _write_records(tmp_path / f"fam_{i}" / "sub" / f"f{i}.pkl", 2)

        index = build_record_index(roots=str(tmp_path), limit_files=2)

        assert len({p for p, _ in index.entries}) == 2
        assert len(index.entries) == 4

    def test_overlapping_roots_are_deduplicated(self, tmp_path: Path) -> None:
        _write_records(tmp_path / "fam" / "sub" / "f.pkl", 3)

        index = build_record_index(roots=[tmp_path, tmp_path / "fam", str(tmp_path / "fam" / "sub")])

        assert len(index.entries) == 3

    def test_deterministic_across_calls(self, tmp_path: Path) -> None:
        for i in range(3):
            _write_records(tmp_path / f"fam_{i}" / "sub" / f"f{i}.pkl", i + 1)

        first = build_record_index(roots=str(tmp_path))
        second = build_record_index(roots=str(tmp_path))

        assert first.entries == second.entries
        assert [p.name for p, _ in first.entries] == sorted(p.name for p, _ in first.entries)

    def test_none_payload_is_skipped_not_raised(self, tmp_path: Path) -> None:
        """An aborted run's stub ``recorder_*.pkl`` (holds ``None``) is skipped.

        Regression: the real ``data/debug/slm_pib_online/20260929_085851``
        directory holds a 4-byte pickle that unpickles to ``None``. It made
        ``build_record_index`` raise ``TypeError`` and killed the whole 3202-record
        training run, because ``DEFAULT_ROOTS`` globs the entire directory tree.
        """
        _write_records(tmp_path / "good" / "records.pkl", 2)
        stub = tmp_path / "aborted" / "recorder_snr.pkl"
        stub.parent.mkdir(parents=True, exist_ok=True)
        with open(stub, "wb") as handle:
            pickle.dump(None, handle)

        index = build_record_index(roots=str(tmp_path))

        assert len(index.entries) == 2
        assert all(p.name == "records.pkl" for p, _ in index.entries)

    def test_corrupt_pickle_is_skipped_not_raised(self, tmp_path: Path) -> None:
        """A truncated dump raises ``EOFError`` and must not abort the scan."""
        _write_records(tmp_path / "good" / "records.pkl", 1)
        truncated = tmp_path / "torn" / "half.pkl"
        truncated.parent.mkdir(parents=True, exist_ok=True)
        truncated.write_bytes(b"\x80\x04\x95")

        index = build_record_index(roots=str(tmp_path))

        assert len(index.entries) == 1
        assert all(p.name == "records.pkl" for p, _ in index.entries)

    def test_all_files_skipped_yields_empty_index(self, tmp_path: Path) -> None:
        """Debris-only root degrades to an empty index, not an exception."""
        stub = tmp_path / "aborted" / "recorder.pkl"
        stub.parent.mkdir(parents=True, exist_ok=True)
        with open(stub, "wb") as handle:
            pickle.dump(None, handle)

        index = build_record_index(roots=str(tmp_path))

        assert index.entries == ()

    def test_real_data_index_survives_stub_pickles(self) -> None:
        """The live ``data/debug`` tree contains stub pickles; indexing must work.

        Guards the actual failure: any future ``data/debug`` debris (an aborted
        hardware run writes ``recorder_*.pkl``) must not disable training.
        """
        index = build_record_index(roots=[REAL_ZERNIKE_GLOB, REAL_PIB_GLOB])

        assert len(index.entries) > 0

    def test_entries_hold_paths_and_ints_only(self, tmp_path: Path) -> None:
        _write_records(tmp_path / "fam" / "sub" / "f.pkl", 2)
        index = build_record_index(roots=str(tmp_path))
        for path, key in index.entries:
            assert isinstance(path, Path)
            assert isinstance(key, int)
            assert path.is_absolute()
            assert path.suffix == ".pkl"

    def test_cache_round_trip_skips_the_scan(self, tmp_path: Path) -> None:
        source = _write_records(tmp_path / "fam" / "sub" / "f.pkl", 4)
        cache = tmp_path / "cache" / "index.json"

        fresh = build_record_index(roots=str(tmp_path), index_cache=cache)
        assert cache.is_file()

        # The JSON payload is a list of [str(path), int(key)] pairs.
        payload = json.loads(cache.read_text(encoding="utf-8"))
        assert payload == [[str(p), k] for p, k in fresh.entries]

        # Delete the source: a cache hit must not need it.
        source.unlink()
        cached = build_record_index(roots=str(tmp_path), index_cache=cache)
        assert cached.entries == fresh.entries

    def test_cache_is_written_when_absent(self, tmp_path: Path) -> None:
        _write_records(tmp_path / "fam" / "sub" / "f.pkl", 2)
        cache = tmp_path / "nested" / "index.json"

        build_record_index(roots=str(tmp_path), index_cache=cache)

        assert cache.is_file()
        assert len(json.loads(cache.read_text(encoding="utf-8"))) == 2

    def test_corrupt_cache_is_ignored_and_rebuilt(self, tmp_path: Path) -> None:
        _write_records(tmp_path / "fam" / "sub" / "f.pkl", 2)
        cache = tmp_path / "index.json"
        cache.write_text("not json at all", encoding="utf-8")

        index = build_record_index(roots=str(tmp_path), index_cache=cache)

        assert len(index.entries) == 2
        assert len(json.loads(cache.read_text(encoding="utf-8"))) == 2

    def test_empty_roots_yield_empty_index(self, tmp_path: Path) -> None:
        index = build_record_index(roots=str(tmp_path / "nothing_here"))
        assert index.entries == ()

    def test_default_roots_are_the_debug_globs(self) -> None:
        assert DEFAULT_ROOTS == ("data/debug/slm_zernike_*", "data/debug/slm_pib_*")


# ---------------------------------------------------------------------------
# build_record_index -- real data
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not DEBUG_DIR.is_dir(), reason="requires data/debug pickles")
class TestBuildRecordIndexRealData:
    def test_indexes_real_zernike_pickles(self) -> None:
        index = build_record_index(roots=REAL_ZERNIKE_GLOB)

        assert len(index.entries) > 0
        files = {p for p, _ in index.entries}
        assert len(files) > 1, "expected several debug pickles"
        for path, key in index.entries:
            assert path.is_file() and path.suffix == ".pkl"
            assert isinstance(key, int)
            # Two levels below the family dir: <prefix>_<tag>/<timestamp>/<file>.pkl
            assert len(path.parts) >= 3
            assert "slm_zernike" in path.name

    def test_limit_files_on_real_data(self) -> None:
        index = build_record_index(roots=REAL_ZERNIKE_GLOB, limit_files=1)

        files = {p for p, _ in index.entries}
        assert len(files) == 1
        assert len(index.entries) == 101, "the real dumps hold 101 records each"

    def test_explicit_roots_list_on_real_data(self) -> None:
        index = build_record_index(roots=[REAL_ZERNIKE_GLOB], limit_files=2)
        assert len({p for p, _ in index.entries}) == 2

    def test_cache_round_trip_on_real_data(self, tmp_path: Path) -> None:
        cache = tmp_path / "real_index.json"

        fresh = build_record_index(roots=REAL_ZERNIKE_GLOB, limit_files=2, index_cache=cache)
        cached = build_record_index(roots=REAL_ZERNIKE_GLOB, limit_files=2, index_cache=cache)

        assert fresh.entries == cached.entries
        assert len(fresh.entries) > 0

    def test_real_pib_records_expose_c(self) -> None:
        """The pib dumps must carry ``_c`` too, with a valid Zernike length."""
        index = build_record_index(roots=REAL_PIB_GLOB, limit_files=1)
        if not index.entries:
            pytest.skip("no data/debug/slm_pib_* pickles present")

        path, key = index.entries[0]
        with open(path, "rb") as handle:
            records = pickle.load(handle)

        record = records[key]
        assert "_c" in record, "slm_pib_* records must expose _c"
        coeffs = np.asarray(record["_c"], dtype=np.float64)
        assert coeffs.ndim == 1 and coeffs.size > 0
        # Length must be a valid Zernike mode count (no hardcoded 15/66/78).
        n_max = infer_n_max(coeffs.size)
        assert (n_max + 1) * (n_max + 2) // 2 == coeffs.size
        assert np.isfinite(coeffs).all()
        # _phase is deliberately unused by this commit.
        assert "_img" in record

    def test_real_records_expose_img(self) -> None:
        index = build_record_index(roots=REAL_ZERNIKE_GLOB, limit_files=1)
        if not index.entries:
            pytest.skip("no data/debug/slm_zernike_* pickles present")

        path, key = index.entries[0]
        with open(path, "rb") as handle:
            records = pickle.load(handle)

        img = np.asarray(records[key]["_img"])
        assert img.ndim == 2
        assert farfield_to_grid(img, 8).shape == (8, 8)


# ---------------------------------------------------------------------------
# reconstruct_pupil_phase_rad
# ---------------------------------------------------------------------------
class TestReconstructPupilPhaseRad:
    def test_shape_and_dtype(self) -> None:
        c = np.zeros(15, dtype=np.float64)
        c[0] = 1.0
        phase = reconstruct_pupil_phase_rad(
            c, n_max=4, slm_width=SLM_WIDTH, slm_height=SLM_HEIGHT, radius=SLM_RADIUS
        )
        assert phase.shape == (SLM_HEIGHT, SLM_WIDTH)
        assert phase.dtype == np.float64

    @pytest.mark.parametrize("n_terms", [15, 66, 78])
    def test_all_observed_mode_counts_reconstruct(self, n_terms: int) -> None:
        c = np.zeros(n_terms, dtype=np.float64)
        c[0] = 1.0
        phase = reconstruct_pupil_phase_rad(
            c,
            n_max=infer_n_max(n_terms),
            slm_width=SLM_WIDTH,
            slm_height=SLM_HEIGHT,
            radius=SLM_RADIUS,
        )
        assert phase.shape == (SLM_HEIGHT, SLM_WIDTH)
        assert np.isfinite(phase).all()

    def test_no_nan_anywhere(self) -> None:
        rng = np.random.default_rng(12345)
        c = rng.normal(size=66)
        phase = reconstruct_pupil_phase_rad(
            c, n_max=10, slm_width=SLM_WIDTH, slm_height=SLM_HEIGHT, radius=SLM_RADIUS
        )
        assert not np.isnan(phase).any()
        assert np.isfinite(phase).all()

    def test_exactly_zero_outside_aperture(self) -> None:
        c = np.zeros(66, dtype=np.float64)
        c[0] = 1.75
        phase = reconstruct_pupil_phase_rad(
            c, n_max=10, slm_width=SLM_WIDTH, slm_height=SLM_HEIGHT, radius=SLM_RADIUS
        )
        mask = _aperture_mask(SLM_HEIGHT, SLM_WIDTH, SLM_RADIUS)
        assert np.all(phase[~mask] == 0.0)
        assert np.all(phase[mask] != 0.0)

    def test_piston_is_constant_inside_aperture(self) -> None:
        c = np.zeros(66, dtype=np.float64)
        c[0] = 1.75
        phase = reconstruct_pupil_phase_rad(
            c, n_max=10, slm_width=SLM_WIDTH, slm_height=SLM_HEIGHT, radius=SLM_RADIUS
        )
        mask = _aperture_mask(SLM_HEIGHT, SLM_WIDTH, SLM_RADIUS)
        assert np.unique(phase[mask]).tolist() == [1.75]

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(999)
        c = rng.normal(size=15)
        first = reconstruct_pupil_phase_rad(
            c, n_max=4, slm_width=SLM_WIDTH, slm_height=SLM_HEIGHT, radius=SLM_RADIUS
        )
        second = reconstruct_pupil_phase_rad(
            c, n_max=4, slm_width=SLM_WIDTH, slm_height=SLM_HEIGHT, radius=SLM_RADIUS
        )
        assert np.array_equal(first, second)

    def test_all_zero_coefficients_return_float64(self) -> None:
        """``generate_zernike_phase`` short-circuits to uint16 zeros; we must not."""
        phase = reconstruct_pupil_phase_rad(
            np.zeros(66), n_max=10, slm_width=SLM_WIDTH, slm_height=SLM_HEIGHT, radius=SLM_RADIUS
        )
        assert phase.dtype == np.float64
        assert np.all(phase == 0.0)

    def test_length_mismatch_with_n_max_raises(self) -> None:
        with pytest.raises(ValueError):
            reconstruct_pupil_phase_rad(
                np.zeros(15), n_max=10, slm_width=SLM_WIDTH, slm_height=SLM_HEIGHT, radius=SLM_RADIUS
            )

    def test_empty_coefficients_raise(self) -> None:
        with pytest.raises(ValueError):
            reconstruct_pupil_phase_rad(
                np.zeros(0), n_max=0, slm_width=64, slm_height=64, radius=30.0
            )

    def test_small_panel(self) -> None:
        c = np.zeros(6, dtype=np.float64)
        c[1] = 0.5
        phase = reconstruct_pupil_phase_rad(c, n_max=2, slm_width=64, slm_height=64, radius=30.0)
        assert phase.shape == (64, 64)
        assert phase.dtype == np.float64
        assert np.isfinite(phase).all()


# ---------------------------------------------------------------------------
# Hardware parity: the reconstruction must equal the LIVE optimizer path
# ---------------------------------------------------------------------------
class TestHardwareParity:
    """Pin ``reconstruct_pupil_phase_rad`` to the real hardware chain.

    The offline corpus stores only ``_c``, so the phase the network trains on is
    reconstructed offline. If that reconstruction ever drifts from what the
    optimizer actually displayed, every learned phase is trained against the
    wrong forward model and the run is silently worthless.

    The authority is the live path in
    ``optimizer/wfless/slm_zernike_pib._zernike_to_phase``:

        parse_zernike_coefficients(c, n_max=n_max)
            -> PatternHelper(resolution=(SLM_WIDTH, SLM_HEIGHT), bits=10)
               .generate_zernike_polynomial(n_max=..., coefficients=..., radius=300)

    with ``radius=ZERNIKE_APERTURE_RADIUS`` (300.0,
    ``slm_zernike_pib.py:235``). Verified bit-exact on a real record
    (``max|_c| = 0.0521 rad``) -- this test keeps that guarantee from rotting.
    """

    @staticmethod
    def _hardware_phase(c: np.ndarray, n_max: int) -> np.ndarray:
        from ao_shaping.utils.wavefront.pattern_helper import PatternHelper
        from ao_shaping.utils.wavefront.zernike_utils import (
            parse_zernike_coefficients,
        )

        helper = PatternHelper(resolution=(SLM_WIDTH, SLM_HEIGHT), bits=10)
        return np.asarray(
            helper.generate_zernike_polynomial(
                n_max=n_max,
                coefficients=parse_zernike_coefficients(c, n_max=n_max),
                radius=SLM_RADIUS,
            ),
            dtype=np.float64,
        )

    @pytest.mark.parametrize("n_terms", [15, 66, 78])
    def test_bit_exact_against_hardware_path(self, n_terms: int) -> None:
        rng = np.random.default_rng(20260926)
        c = rng.normal(scale=0.05, size=n_terms)  # radian-scale, as recorded
        n_max = infer_n_max(n_terms)

        ours = reconstruct_pupil_phase_rad(
            c,
            n_max=n_max,
            slm_width=SLM_WIDTH,
            slm_height=SLM_HEIGHT,
            radius=SLM_RADIUS,
        )
        theirs = self._hardware_phase(c, n_max)

        assert ours.shape == theirs.shape == (SLM_HEIGHT, SLM_WIDTH)
        # Bit-exact, not merely close: both call the same ZernikeGenerator.
        assert np.array_equal(ours, theirs), (
            f"max |diff| = {np.abs(ours - theirs).max():.3e}"
        )

    def test_bit_exact_with_sparse_single_mode(self) -> None:
        # One dominant mode, like a real record whose only non-zero term is
        # defocus -- catches any accidental normalisation.
        c = np.zeros(66, dtype=np.float64)
        c[3] = 0.5
        ours = reconstruct_pupil_phase_rad(
            c, n_max=10, slm_width=SLM_WIDTH, slm_height=SLM_HEIGHT, radius=SLM_RADIUS
        )
        assert np.array_equal(ours, self._hardware_phase(c, 10))

    def test_zero_outside_aperture_matches_hardware(self) -> None:
        c = np.zeros(66, dtype=np.float64)
        c[3] = 0.4
        c[10] = 0.2
        ours = reconstruct_pupil_phase_rad(
            c, n_max=10, slm_width=SLM_WIDTH, slm_height=SLM_HEIGHT, radius=SLM_RADIUS
        )
        theirs = self._hardware_phase(c, 10)
        outside = ~_aperture_mask(SLM_HEIGHT, SLM_WIDTH, SLM_RADIUS)
        assert np.array_equal(ours[outside], theirs[outside])
        assert np.all(ours[outside] == 0.0)

    def test_radius_300_is_not_half_panel(self) -> None:
        """Guard the radius: 300 is the device constant, not ``min(w, h) / 2``."""
        assert SLM_RADIUS == 300.0
        assert SLM_RADIUS != min(SLM_WIDTH, SLM_HEIGHT) / 2.0

        c = np.zeros(66, dtype=np.float64)
        c[3] = 0.5
        c[10] = 0.3
        grid = pupil_phase_to_grid(
            reconstruct_pupil_phase_rad(
                c,
                n_max=10,
                slm_width=SLM_WIDTH,
                slm_height=SLM_HEIGHT,
                radius=SLM_RADIUS,
            ),
            64,
        )
        # A 600 px disc on a 1200 px panel -> a minority of the 64x64 grid.
        assert 0 < int(np.count_nonzero(grid)) < 64 * 64 // 4

    def test_generator_is_cached_across_calls(self) -> None:
        """The panel grid must be built once, not per record.

        Measured motivation: rebuilding the 1200x1920 RZern grid for every
        record dominated the pipeline.
        """
        from ml.gsnet_debug import offline as gsnet_offline

        gsnet_offline._zernike_generator.cache_clear()
        c = np.zeros(66, dtype=np.float64)
        c[3] = 0.5
        for _ in range(3):
            reconstruct_pupil_phase_rad(
                c,
                n_max=10,
                slm_width=SLM_WIDTH,
                slm_height=SLM_HEIGHT,
                radius=SLM_RADIUS,
            )
        info = gsnet_offline._zernike_generator.cache_info()
        assert info.misses == 1, f"grid rebuilt {info.misses} times for one geometry"
        assert info.hits == 2


# ---------------------------------------------------------------------------
# pupil_phase_to_grid
# ---------------------------------------------------------------------------
class TestPupilPhaseToGrid:
    @pytest.mark.parametrize(
        ("shape", "grid"),
        [((7, 9), 3), ((5, 5), 2), ((3, 7), 4), ((8, 8), 4), ((6, 10), 5), ((1, 1), 1), ((2, 3), 1), ((13, 13), 16)],
    )
    def test_exact_block_mean(self, shape: tuple[int, int], grid: int) -> None:
        rows, cols = shape
        phase = np.arange(rows * cols, dtype=np.float64).reshape(shape)
        expected = _block_mean_reference(phase, grid)
        got = pupil_phase_to_grid(phase, grid)
        assert got.shape == (grid, grid)
        assert got.dtype == np.float32
        # float32 output vs float64 reference: tolerance scaled to float32 eps.
        assert np.allclose(got, expected, rtol=1e-6, atol=1e-4)

    def test_real_panel_reduces_to_64_square(self) -> None:
        phase = np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=np.float64)
        phase[100:900, 200:1500] = 2.0
        got = pupil_phase_to_grid(phase, 64)
        assert got.shape == (64, 64)
        assert got.dtype == np.float32
        assert np.allclose(got, _block_mean_reference(phase, 64), rtol=1e-6, atol=1e-4)

    def test_symmetric_padding_1200_to_1216(self) -> None:
        """1200 rows pad to 1216 = 64 blocks of 19, 8 rows top and bottom."""
        phase = np.ones((SLM_HEIGHT, SLM_WIDTH), dtype=np.float64)
        got = pupil_phase_to_grid(phase, 64)
        # Block 0 covers padded rows 0..18, of which 8 are zero padding.
        expected_row0 = (11 * SLM_WIDTH) / (19 * SLM_WIDTH)
        assert np.isclose(float(got[0, 0]), expected_row0, rtol=1e-6)
        # The last block is diluted by the 8 trailing zero rows.
        assert np.isclose(float(got[-1, 0]), expected_row0, rtol=1e-6)
        # Every fully illuminated block row is exactly 1.0.
        assert np.allclose(got[1:-1], 1.0, rtol=0, atol=1e-6)

    def test_constant_field_is_preserved(self) -> None:
        """No padding is needed here (8 = 4x2, 12 = 4x3), so means are exact."""
        got = pupil_phase_to_grid(np.full((8, 12), -2.5), 4)
        assert got.shape == (4, 4)
        assert np.allclose(got, -2.5, rtol=0, atol=1e-6)

    def test_padding_dilutes_edge_blocks(self) -> None:
        """A padded constant field is pulled toward zero in the edge blocks.

        (5, 7) with grid=4 pads rows 5 -> 8 (1 row top, 2 bottom) and cols
        7 -> 8 (1 col right), giving 2x2 blocks.
        """
        got = pupil_phase_to_grid(np.full((5, 7), -2.5), 4)
        assert got.shape == (4, 4)
        # Block (0, 0): 1 pad row + 1 data row out of 2 => half the mean.
        assert np.isclose(float(got[0, 0]), -2.5 / 2, rtol=1e-6)
        # Block (1, 0): both rows illuminated => untouched.
        assert np.isclose(float(got[1, 0]), -2.5, rtol=1e-6)
        # Block (1, 3): 1 data col + 1 pad col out of 2 => half the mean.
        assert np.isclose(float(got[1, -1]), -2.5 / 2, rtol=1e-6)
        # Block row 3 is entirely padding.
        assert np.allclose(got[-1], 0.0, rtol=0, atol=1e-6)

    def test_exact_multiple_is_unpadded(self) -> None:
        phase = np.arange(64, dtype=np.float64).reshape(8, 8)
        got = pupil_phase_to_grid(phase, 4)
        expected = phase.reshape(4, 2, 4, 2).mean(axis=(1, 3))
        assert np.allclose(got, expected, rtol=1e-6, atol=1e-4)

    def test_integer_input_is_accepted(self) -> None:
        got = pupil_phase_to_grid(np.arange(12, dtype=np.int16).reshape(3, 4), 2)
        assert got.dtype == np.float32
        assert np.allclose(got, _block_mean_reference(np.arange(12, dtype=np.float64).reshape(3, 4), 2))

    def test_never_truncates_input(self) -> None:
        """A bright pixel in the very last row must still influence the output."""
        phase = np.zeros((5, 5), dtype=np.float64)
        phase[-1, -1] = 1.0
        got = pupil_phase_to_grid(phase, 2)
        assert float(got[-1, -1]) > 0.0

    def test_rejects_bad_shapes(self) -> None:
        with pytest.raises(ValueError):
            pupil_phase_to_grid(np.zeros(4), 2)
        with pytest.raises(ValueError):
            pupil_phase_to_grid(np.zeros((0, 4)), 2)
        with pytest.raises(ValueError):
            pupil_phase_to_grid(np.zeros((4, 4)), 0)


# ---------------------------------------------------------------------------
# farfield_to_grid
# ---------------------------------------------------------------------------
class TestFarfieldToGrid:
    def test_argmax_anchored_not_frame_centre(self) -> None:
        frame = np.zeros((40, 40), dtype=np.uint8)
        frame[10, 30] = 255  # far from the frame centre (20, 20)

        got = farfield_to_grid(frame, 16)

        assert got.shape == (16, 16)
        assert got.dtype == np.float32
        assert np.unravel_index(int(got.argmax()), got.shape) == (8, 8)
        assert float(got.max()) == 1.0
        # A frame-centre crop would have missed the peak entirely.
        assert float(got.max()) != float(got[0, 0])

    def test_peak_normalised_to_one(self) -> None:
        frame = np.zeros((20, 20), dtype=np.uint8)
        frame[7, 12] = 37
        got = farfield_to_grid(frame, 8)
        assert np.isclose(float(got.max()), 1.0)
        assert np.all(got >= 0.0) and np.all(got <= 1.0)

    def test_all_zero_frame_returns_zeros(self) -> None:
        got = farfield_to_grid(np.zeros((12, 12), dtype=np.uint8), 8)
        assert got.shape == (8, 8)
        assert np.all(got == 0.0)
        assert not np.isnan(got).any()

    def test_eps_guard_prevents_division_by_zero(self) -> None:
        frame = np.full((8, 8), 1e-18, dtype=np.float64)
        got = farfield_to_grid(frame, 4, eps=1e-12)
        assert np.all(got == 0.0)

    def test_peak_below_eps_returns_zeros(self) -> None:
        frame = np.zeros((8, 8), dtype=np.float64)
        frame[4, 4] = 1e-15
        got = farfield_to_grid(frame, 4, eps=1e-12)
        assert np.all(got == 0.0)

    def test_grid_larger_than_frame_pads_with_zeros(self) -> None:
        frame = np.zeros((5, 5), dtype=np.uint8)
        frame[2, 2] = 200
        got = farfield_to_grid(frame, 16)
        assert got.shape == (16, 16)
        assert np.isclose(float(got.max()), 1.0)

    def test_nan_input_does_not_propagate(self) -> None:
        frame = np.zeros((10, 10), dtype=np.float64)
        frame[5, 5] = 100.0
        frame[0, 0] = np.nan
        got = farfield_to_grid(frame, 6)
        assert np.isfinite(got).all()
        assert np.isclose(float(got.max()), 1.0)

    def test_edge_peak_is_clamped_and_finite(self) -> None:
        frame = np.zeros((30, 30), dtype=np.uint8)
        frame[0, 0] = 255
        got = farfield_to_grid(frame, 16)
        assert got.shape == (16, 16)
        assert np.isfinite(got).all()
        assert np.all(got >= 0.0) and np.all(got <= 1.0)

    def test_float_and_uint_inputs_agree(self) -> None:
        base = np.zeros((16, 16), dtype=np.uint8)
        base[9, 6] = 200
        base[8, 6] = 90
        assert np.allclose(farfield_to_grid(base, 8), farfield_to_grid(base.astype(np.float64), 8))

    def test_odd_and_even_grids(self) -> None:
        frame = np.zeros((20, 20), dtype=np.uint8)
        frame[6, 13] = 255
        for grid in (4, 5, 8, 9):
            got = farfield_to_grid(frame, grid)
            assert got.shape == (grid, grid)
            assert np.isclose(float(got.max()), 1.0)

    def test_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError):
            farfield_to_grid(np.zeros(8), 4)
        with pytest.raises(ValueError):
            farfield_to_grid(np.zeros((0, 4)), 4)
        with pytest.raises(ValueError):
            farfield_to_grid(np.zeros((8, 8)), 0)
