"""Unit tests for :mod:`ml.hwdataset.zernike_dataset` (coefficients -> far field).

Every fixture is a **synthetic** pickle in a pytest ``tmp_path``, so this suite never
touches the real ``data/debug`` corpus (65.88 GB, 265 files, absent in CI). Panels
are tiny -- ``(40, 60)`` instead of the real ``(1200, 1920)`` -- and every test
passes a small :class:`~ml.hwdataset.records.MaterialiserConfig` so the production
ROI (``(960, 600)`` / ``r=500``) never misses them.

The synthetic records carry ``_c`` only (no ``_phase``), which is exactly how
:func:`~ml.hwdataset.index.build_hw_index` classifies a Zernike-sourced record. The
coefficient lengths mirror the real corpus's mixed ``{15, 36, 78}`` triangular
shapes, which is what makes the padding and the tail-statistics behaviour testable.

The suite is organised around the module's four documented decisions:

* exact Noll-prefix-preserving padding (:func:`pad_coefficients`),
* train-file-only standardisation (:func:`fit_coeff_stats` and the split factory),
* ``image_mode="robust"`` by default, never total-energy,
* Windows-``spawn`` picklability that carries the statistics.

``torch`` is a hard dependency of the module under test, so it is imported at module
scope and the whole file is skipped when it is missing.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import (  # noqa: E402
    DataLoader,
    SequentialSampler,
    Subset,
)

from ao_shaping.utils.wavefront.zernike_calc import (  # noqa: E402
    calc_n_zernike_terms,
    noll_indices,
)
from ml.hwdataset.dataset import FileGroupedSampler  # noqa: E402
from ml.hwdataset.index import (  # noqa: E402
    HwCorpusIndex,
    PhaseSource,
    build_hw_index,
)
from ml.hwdataset.records import MaterialiserConfig  # noqa: E402
from ml.hwdataset.zernike_dataset import (  # noqa: E402
    DEFAULT_N_MAX,
    ZERNIKE_COEFF_SOURCES,
    ZernikeCoeffDataset,
    build_zernike_coeff_dataloader,
    create_zernike_coeff_dataloaders,
    fit_coeff_stats,
    pad_coefficients,
)

_STAMP = "20260101_000000"
_PANEL_SHAPE = (40, 60)  # (height, width), stands in for (1200, 1920)
_FRAME_SHAPE = (16, 16)
_GRID = 8
_N_TERMS = calc_n_zernike_terms(DEFAULT_N_MAX)  # 136 at the default n_max=15

# Triangular coefficient lengths, mirroring the real corpus {15, 36, 78}.
_N_MAX_SMALL = 4  # (4+1)(4+2)/2 = 15 terms
_N_MAX_MEDIUM = 7  # (7+1)(7+2)/2 = 36 terms


# ---------------------------------------------------------------------------
# Fixture writers
# ---------------------------------------------------------------------------
def _frame(scale: float = 1.0, offset: int = 0) -> np.ndarray:
    """Return a small ``uint8`` CCD frame with a controllable pedestal and peak.

    Args:
        scale: Multiplier on the ramp, so records differ in absolute brightness.
        offset: Additive pedestal, so ``"robust"`` normalisation has a median to
            subtract and ``"abs255"`` has a floor to preserve.

    Returns:
        A ``(16, 16) uint8`` array shaped like the real far-field windows.
    """
    ramp = (np.arange(_FRAME_SHAPE[0] * _FRAME_SHAPE[1], dtype=np.float64) % 251) * float(
        scale
    )
    return np.clip(ramp + offset, 0, 255).astype(np.uint8).reshape(_FRAME_SHAPE)


def _coeffs(length: int, seed: int, scale: float = 1.0) -> np.ndarray:
    """Return a reproducible ``float64`` coefficient vector of ``length`` entries.

    Args:
        length: Number of Noll terms. Use a triangular length for realistic data.
        seed: RNG seed, so a test can name a record's coefficients exactly.
        scale: Overall amplitude in radians.

    Returns:
        A ``(length,) float64`` raw-radian vector, the shape ``_c`` has on disk.
    """
    rng = np.random.default_rng(seed)
    return (rng.normal(size=length) * float(scale)).astype(np.float64)


_DEFAULT_EXPOSURE_MS = 1.0


def _config(grid: int = _GRID, image_mode: str = "robust") -> MaterialiserConfig:
    """Return a tiny-panel :class:`MaterialiserConfig`.

    Args:
        grid: Output side length.
        image_mode: Far-field scaling mode.

    Returns:
        A config whose ROI fits the ``(40, 60)`` synthetic frames.
    """
    return MaterialiserConfig(
        grid=grid,
        panel_center=(30, 20),
        panel_radius=20,
        panel_resolution=(60, 40),
        image_mode=image_mode,
    )


def _write_pkl(path: Path, payload: Any) -> Path:
    """Write ``payload`` as a pickle, creating parent directories.

    Args:
        path: Destination ``.pkl`` path.
        payload: Any picklable object.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(payload))
    return path


def _write_sidecar(pkl_path: Path, **fields: Any) -> Path:
    """Write a JSON sidecar beside ``pkl_path``.

    Args:
        pkl_path: The pickle the sidecar describes.
        **fields: JSON object fields.

    Returns:
        The written sidecar path.
    """
    sidecar = pkl_path.with_suffix(".json")
    sidecar.write_text(json.dumps(fields), encoding="utf-8")
    return sidecar


def _write_zernike_run(
    root: Path,
    stem: str,
    lengths: Sequence[int],
    *,
    seed: int = 0,
    scale: float = 1.0,
    exposures: Sequence[float | None] | None = None,
    frames: Sequence[np.ndarray] | None = None,
    with_sidecar: bool = True,
) -> Path:
    """Write one ``dict[int, dict]`` file of Zernike-sourced records.

    Each record carries ``_c`` (raw Noll radians) and ``_img`` (uint8 far field) and
    **no** ``_phase``, which is what makes ``build_hw_index`` classify it as
    :attr:`~ml.hwdataset.index.PhaseSource.ZERNIKE`.

    Args:
        root: Family directory.
        stem: File stem, which also selects the family via ``family_of``.
        lengths: One coefficient-vector length per record.
        seed: Base RNG seed; record ``i`` uses ``seed + i``.
        scale: Coefficient amplitude in radians.
        exposures: Optional per-record ``exp_t`` in ms. ``None`` (the default)
            gives every record ``_DEFAULT_EXPOSURE_MS``, because
            ``require_exposure=True`` is the production default and a record
            without any exposure is excluded by the index. Pass an explicit
            sequence of ``None`` to build a genuinely exposure-less corpus.
        frames: Optional per-record frames; defaults to :func:`_frame`.
        with_sidecar: Also write a JSON sidecar naming the family explicitly.

    Returns:
        The written ``.pkl`` path.
    """
    records: list[dict[str, Any]] = []
    for i, length in enumerate(lengths):
        record: dict[str, Any] = {
            "_c": _coeffs(length, seed + i, scale),
            "_img": _frame() if frames is None else frames[i],
        }
        if exposures is None:
            exposure = _DEFAULT_EXPOSURE_MS
        else:
            exposure = exposures[i]
        if exposure is not None:
            record["exp_t"] = float(exposure)
        records.append(record)
    payload = {i: record for i, record in enumerate(records)}
    # ``family_of`` labels a pickle by its *directory* prefix, matching the real
    # ``<family>_<timestamp>/<family>.pkl`` tree -- a flat directory would make the
    # label fall back to the timestamp and every family assertion would be vacuous.
    path = _write_pkl(root / f"{stem}_{_STAMP}" / f"{stem}.pkl", payload)
    if with_sidecar:
        _write_sidecar(path, family=stem)
    return path


def _index(root: Path) -> HwCorpusIndex:
    """Build an index over ``root`` with progress logging silenced.

    Args:
        root: Corpus root.

    Returns:
        The :class:`~ml.hwdataset.index.HwCorpusIndex` over every ``.pkl`` below.
    """
    return build_hw_index(root, progress_every=0)


def _filtered(root: Path) -> HwCorpusIndex:
    """Build an index over ``root`` and keep only the Zernike-sourced records.

    Args:
        root: Corpus root.

    Returns:
        The Zernike-filtered index, with record positions renumbered relative to
        the filtered tuple (which is what a Dataset position means).
    """
    return _index(root).filter(sources=ZERNIKE_COEFF_SOURCES)


def _std(n_terms: int = _N_TERMS) -> np.ndarray:
    """Return a valid non-degenerate ``coeff_std``.

    Args:
        n_terms: Vector length.

    Returns:
        A ``(n_terms,) float64`` vector of ones.
    """
    return np.ones(n_terms, dtype=np.float64)


# ---------------------------------------------------------------------------
# pad_coefficients
# ---------------------------------------------------------------------------
class TestPadCoefficients:
    """Exact Noll-prefix-preserving padding."""

    def test_length_matches_canonical_zernike_term_count(self) -> None:
        """The default length is the canonical 136 for ``n_max=15``."""
        assert DEFAULT_N_MAX == 15
        assert _N_TERMS == 136
        assert _N_TERMS == len(noll_indices(DEFAULT_N_MAX))

    @pytest.mark.parametrize(
        ("n_max", "expected"),
        [(0, 1), (1, 3), (4, 15), (7, 36), (11, 78), (15, 136)],
    )
    def test_length_is_triangular_in_n_max(self, n_max: int, expected: int) -> None:
        """Every ``n_max`` maps to ``(n_max+1)(n_max+2)/2`` entries."""
        assert calc_n_zernike_terms(n_max) == expected
        # The input has to fit, so feed exactly the target width -- a 3-element
        # vector is longer than ``n_max=0`` admits and is rejected by design.
        assert pad_coefficients(np.zeros(expected), n_max=n_max).shape == (expected,)

    def test_exact_length_input_is_copied_bit_for_bit(self) -> None:
        """A full-length vector is returned unchanged, with no reordering."""
        source = _coeffs(_N_TERMS, seed=7)
        padded = pad_coefficients(source)
        assert np.array_equal(padded, source)

    def test_short_input_preserves_prefix_and_zero_fills_tail(self) -> None:
        """The first ``len(coeffs)`` entries survive; the rest are exactly zero."""
        source = _coeffs(15, seed=3)
        padded = pad_coefficients(source)
        assert padded.shape == (_N_TERMS,)
        assert np.array_equal(padded[:15], source)
        assert np.array_equal(padded[15:], np.zeros(_N_TERMS - 15))

    def test_padding_does_not_rescale_wrap_or_reorder(self) -> None:
        """Values larger than ``2*pi`` and any sign survive untouched.

        A min-max or ``mod 2*pi`` normaliser would be scale-invariant or would move
        these values; the prefix contract forbids both.
        """
        source = np.array([-7.5, 0.0, 19.0, 2 * np.pi + 0.25], dtype=np.float64)
        padded = pad_coefficients(source)
        assert np.array_equal(padded[:4], source)

    def test_empty_input_pads_to_all_zeros(self) -> None:
        """A record with no coefficients still yields a full-width vector."""
        padded = pad_coefficients(np.array([], dtype=np.float64))
        assert padded.shape == (_N_TERMS,)
        assert np.array_equal(padded, np.zeros(_N_TERMS))

    def test_returns_own_float64_storage(self) -> None:
        """The result owns its data, so mutating it cannot corrupt the input."""
        source = np.ones(4, dtype=np.float64)
        padded = pad_coefficients(source)
        padded[0] = 99.0
        assert source[0] == 1.0
        assert padded.dtype == np.float64

    def test_accepts_plain_python_sequences(self) -> None:
        """A list works, not only an ndarray."""
        assert np.array_equal(
            pad_coefficients([1.0, 2.0, 3.0])[:3], np.array([1.0, 2.0, 3.0])
        )

    def test_rejects_too_long_input_rather_than_truncating(self) -> None:
        """Excess coefficients are an error, so no high-order mode is silently lost."""
        with pytest.raises(ValueError, match="raise n_max"):
            pad_coefficients(np.ones(_N_TERMS + 1))

    @pytest.mark.parametrize(
        "bad",
        [
            np.array([[1.0, 2.0], [3.0, 4.0]]),  # 2-D
            np.array([1.0, np.nan]),
            np.array([1.0, np.inf]),
        ],
    )
    def test_rejects_non_1d_or_non_finite(self, bad: np.ndarray) -> None:
        """Malformed coefficient input is rejected, never silently patched."""
        with pytest.raises(ValueError):
            pad_coefficients(bad)

    def test_rejects_negative_n_max(self) -> None:
        """A negative radial order is a caller bug."""
        with pytest.raises(ValueError, match="n_max must be"):
            pad_coefficients(np.ones(4), n_max=-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TestZernikeCoeffDataset:
    """Sample contract, filtering, and index semantics."""

    def test_filters_to_zernike_sources_only(self, tmp_path: Path) -> None:
        """A panel-sourced record in the same corpus is not silently included."""
        root = tmp_path / "mixed"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 15])
        panel_pkl = _write_pkl(
            root / _STAMP / "slm_pib_rad.pkl",
            {
                0: {
                    "_phase": np.zeros(_PANEL_SHAPE, dtype=np.float32),
                    "_img": _frame(),
                    "exp_t": 1.0,
                }
            },
        )
        _write_sidecar(panel_pkl, family="slm_pib_rad")

        dataset = ZernikeCoeffDataset(_index(root), config=_config())
        assert len(dataset) == 2
        assert {record.source for record in dataset.records} == {PhaseSource.ZERNIKE}

    def test_index_filter_matches_dataset_length(self, tmp_path: Path) -> None:
        """``len(dataset)`` equals the Zernike-filtered record count."""
        root = tmp_path / "len"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 36, 78])
        filtered = _filtered(root)
        dataset = ZernikeCoeffDataset(_index(root), config=_config())
        assert len(dataset) == len(filtered.records) == 3

    def test_sample_shapes_dtypes_and_range(self, tmp_path: Path) -> None:
        """One item has the documented keys, shapes, dtypes and image range."""
        root = tmp_path / "struct"
        _write_zernike_run(root, "slm_zernike_shaping", [15], exposures=[0.5])
        dataset = ZernikeCoeffDataset(
            _index(root),
            config=_config(),
            coeff_mean=np.zeros(_N_TERMS),
            coeff_std=_std(),
        )

        item = dataset[0]
        assert set(item) == {
            "coeffs",
            "coeffs_raw",
            "image",
            "n_terms_record",
            "n_max_record",
            "family",
            "sample_idx",
            "path",
            "fov_px",
            "exposure_ms",
        }
        assert tuple(item["coeffs"].shape) == (_N_TERMS,)
        assert tuple(item["coeffs_raw"].shape) == (_N_TERMS,)
        assert tuple(item["image"].shape) == (1, _GRID, _GRID)
        for key in ("coeffs", "coeffs_raw", "image"):
            assert item[key].dtype is torch.float32
            assert item[key].is_contiguous()
        assert float(item["image"].min()) >= 0.0
        assert float(item["image"].max()) <= 1.0
        assert item["n_terms_record"] == 15
        assert item["n_max_record"] == _N_MAX_SMALL
        assert item["family"] == "slm_zernike_shaping"
        assert item["sample_idx"] == 0
        assert isinstance(item["path"], str)
        assert item["exposure_ms"] == pytest.approx(0.5)

    @pytest.mark.parametrize("length", [15, 36, 78])
    def test_coeffs_raw_is_noll_prefix_of_the_stored_vector(
        self, tmp_path: Path, length: int
    ) -> None:
        """``coeffs_raw`` reproduces the pickle's ``_c`` exactly, then zero-fills."""
        root = tmp_path / f"raw{length}"
        source = _coeffs(length, seed=11)
        _write_pkl(
            root / _STAMP / "slm_zernike_shaping.pkl",
            {0: {"_c": source, "_img": _frame(), "exp_t": 1.0}},
        )
        _write_sidecar(root / _STAMP / "slm_zernike_shaping.pkl", family="x")

        dataset = ZernikeCoeffDataset(
            _index(root),
            config=_config(),
            coeff_std=_std(),
        )
        raw = dataset[0]["coeffs_raw"].numpy()
        assert np.array_equal(raw[:length], source.astype(np.float32))
        assert np.array_equal(raw[length:], np.zeros(_N_TERMS - length, dtype=np.float32))
        assert dataset[0]["n_terms_record"] == length

    def test_standardisation_applies_the_supplied_statistics(
        self, tmp_path: Path
    ) -> None:
        """``coeffs`` is ``(raw - mean) / std`` elementwise, not a rescale."""
        root = tmp_path / "std"
        source = _coeffs(15, seed=5)
        _write_zernike_run(root, "slm_zernike_shaping", [15], seed=5)

        mean = np.full(_N_TERMS, 0.5)
        std = np.full(_N_TERMS, 2.0)
        dataset = ZernikeCoeffDataset(
            _index(root), config=_config(), coeff_mean=mean, coeff_std=std
        )
        item = dataset[0]
        # Compare against the *padded* source: the tail is zero, so the tail of the
        # standardised vector is (0 - 0.5) / 2.0, not 0.
        padded = np.zeros(_N_TERMS, dtype=np.float32)
        padded[:15] = source
        expected = (padded - 0.5) / 2.0
        assert np.allclose(item["coeffs"].numpy(), expected, rtol=0, atol=1e-6)
        # coeffs_raw stays raw, which is the only way back to a realisable phase.
        assert np.array_equal(item["coeffs_raw"].numpy(), padded)

    def test_defaults_are_identity_standardisation(self, tmp_path: Path) -> None:
        """With no statistics supplied, ``coeffs`` equals ``coeffs_raw``."""
        root = tmp_path / "ident"
        _write_zernike_run(root, "slm_zernike_shaping", [15])
        dataset = ZernikeCoeffDataset(_index(root), config=_config())
        item = dataset[0]
        assert np.array_equal(item["coeffs"].numpy(), item["coeffs_raw"].numpy())
        assert np.array_equal(dataset.coeff_mean, np.zeros(_N_TERMS))
        assert np.array_equal(dataset.coeff_std, np.ones(_N_TERMS))

    def test_constant_tail_stays_exactly_zero(self, tmp_path: Path) -> None:
        """A degenerate std never turns the padded tail into ``inf``/``nan``.

        Every synthetic record holds 15 coefficients, so indices 15..135 are
        constant zero. If ``coeff_std`` were left at 0 there, ``(0 - 0) / 0``
        would produce non-finite inputs for exactly the coefficients this corpus
        never drives.
        """
        root = tmp_path / "tail"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 15, 15])
        index = _filtered(root)
        mean, std = fit_coeff_stats(index, [0, 1, 2])
        assert np.array_equal(mean[15:], np.zeros(_N_TERMS - 15))
        assert np.array_equal(std[15:], np.ones(_N_TERMS - 15))

        dataset = ZernikeCoeffDataset(
            index, config=_config(), coeff_mean=mean, coeff_std=std
        )
        coeffs = dataset[0]["coeffs"].numpy()
        assert np.all(np.isfinite(coeffs))
        assert np.array_equal(coeffs[15:], np.zeros(_N_TERMS - 15))

    def test_tensors_are_fresh_and_mutable(self, tmp_path: Path) -> None:
        """Two reads of one record return distinct storages, so a caller may write."""
        root = tmp_path / "fresh"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 15])
        dataset = ZernikeCoeffDataset(_index(root), config=_config())
        first = dataset[0]
        second = dataset[0]
        assert first["coeffs"].data_ptr() != second["coeffs"].data_ptr()
        assert first["image"].data_ptr() != second["image"].data_ptr()
        first["coeffs"][0] = 1234.0
        first["image"][0, 0, 0] = -5.0
        assert second["coeffs"][0] != 1234.0

    def test_negative_index_follows_sequence_semantics(self, tmp_path: Path) -> None:
        """``dataset[-1]`` is the last record, like any Python sequence."""
        root = tmp_path / "neg"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 36, 78])
        dataset = ZernikeCoeffDataset(_index(root), config=_config())
        assert dataset[-1]["n_terms_record"] == 78
        assert dataset[-1]["sample_idx"] == len(dataset) - 1

    def test_out_of_range_index_raises(self, tmp_path: Path) -> None:
        """An out-of-range position is an ``IndexError``, not a silent wrap."""
        root = tmp_path / "range"
        _write_zernike_run(root, "slm_zernike_shaping", [15])
        dataset = ZernikeCoeffDataset(_index(root), config=_config())
        with pytest.raises(IndexError, match="out of range"):
            dataset[5]

    def test_non_integer_index_raises_type_error(self, tmp_path: Path) -> None:
        """A float or string index is a ``TypeError``, as torch's Dataset contract wants."""
        root = tmp_path / "type"
        _write_zernike_run(root, "slm_zernike_shaping", [15])
        dataset = ZernikeCoeffDataset(_index(root), config=_config())
        with pytest.raises(TypeError):
            dataset["0"]  # type: ignore[index]

    def test_rejects_index_without_zernike_records(self, tmp_path: Path) -> None:
        """An empty filtered index is a clear error, not a zero-length Dataset."""
        root = tmp_path / "none"
        _write_pkl(
            root / _STAMP / "slm_pib_rad.pkl",
            {0: {"_phase": np.zeros(_PANEL_SHAPE, dtype=np.float32), "_img": _frame()}},
        )
        _write_sidecar(root / _STAMP / "slm_pib_rad.pkl", family="slm_pib_rad")
        with pytest.raises(ValueError, match="no Zernike-sourced record"):
            ZernikeCoeffDataset(_index(root), config=_config())

    def test_rejects_unknown_exposure_by_default(self, tmp_path: Path) -> None:
        """``require_exposure=True`` drops records with no resolvable exposure."""
        root = tmp_path / "expo"
        _write_zernike_run(
            root, "slm_zernike_shaping", [15], exposures=[None], with_sidecar=True
        )
        # A sidecar without exposure cannot rescue a record that has none either.
        with pytest.raises(ValueError, match="no Zernike-sourced record"):
            ZernikeCoeffDataset(_index(root), config=_config())

    def test_keeps_unknown_exposure_when_asked(self, tmp_path: Path) -> None:
        """``require_exposure=False`` keeps the record and reports ``None``."""
        root = tmp_path / "expo2"
        _write_zernike_run(root, "slm_zernike_shaping", [15], exposures=[None])
        dataset = ZernikeCoeffDataset(
            _index(root), config=_config(), require_exposure=False
        )
        assert len(dataset) == 1
        assert dataset[0]["exposure_ms"] is None

    def test_rejects_negative_cache_size(self, tmp_path: Path) -> None:
        """A negative LRU capacity is a caller bug."""
        root = tmp_path / "cache"
        _write_zernike_run(root, "slm_zernike_shaping", [15])
        with pytest.raises(ValueError, match="cache_size must be"):
            ZernikeCoeffDataset(_index(root), config=_config(), cache_size=-1)

    def test_rejects_negative_n_max(self, tmp_path: Path) -> None:
        """A negative radial order is a caller bug."""
        root = tmp_path / "nmax"
        _write_zernike_run(root, "slm_zernike_shaping", [15])
        with pytest.raises(ValueError, match="n_max must be"):
            ZernikeCoeffDataset(_index(root), config=_config(), n_max=-1)

    def test_rejects_statistic_with_wrong_length(self, tmp_path: Path) -> None:
        """A statistic whose length disagrees with ``n_max`` is rejected, not broadcast."""
        root = tmp_path / "shape"
        _write_zernike_run(root, "slm_zernike_shaping", [15])
        with pytest.raises(ValueError, match="refit it with"):
            ZernikeCoeffDataset(_index(root), config=_config(), coeff_mean=np.zeros(10))

    def test_rejects_non_finite_statistic(self, tmp_path: Path) -> None:
        """A ``nan`` in the statistics would poison every item, so it is rejected."""
        root = tmp_path / "nan"
        _write_zernike_run(root, "slm_zernike_shaping", [15])
        bad = np.full(_N_TERMS, np.nan)
        with pytest.raises(ValueError, match="must be finite"):
            ZernikeCoeffDataset(_index(root), config=_config(), coeff_mean=bad)

    def test_non_positive_std_is_replaced_to_keep_items_finite(
        self, tmp_path: Path
    ) -> None:
        """A zero or negative divisor becomes 1.0 instead of ``inf``."""
        root = tmp_path / "nonpos"
        _write_zernike_run(root, "slm_zernike_shaping", [15])
        std = np.zeros(_N_TERMS)
        std[:5] = -1.0
        dataset = ZernikeCoeffDataset(_index(root), config=_config(), coeff_std=std)
        assert np.array_equal(dataset.coeff_std, np.ones(_N_TERMS))
        assert np.all(np.isfinite(dataset[0]["coeffs"].numpy()))

    def test_shorter_n_max_still_pads_to_its_own_length(self, tmp_path: Path) -> None:
        """``n_max`` really controls the width; it is not hard-coded to 136."""
        root = tmp_path / "nmax7"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 15])
        n_max = _N_MAX_MEDIUM
        n_terms = calc_n_zernike_terms(n_max)
        dataset = ZernikeCoeffDataset(
            _index(root), config=_config(), n_max=n_max, coeff_std=np.ones(n_terms)
        )
        assert dataset.n_max == n_max
        assert dataset.n_terms == n_terms
        assert tuple(dataset[0]["coeffs"].shape) == (n_terms,)

    def test_exposes_index_config_and_records(self, tmp_path: Path) -> None:
        """The introspection properties return the filtered, live state."""
        root = tmp_path / "props"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 36])
        config = _config()
        dataset = ZernikeCoeffDataset(_index(root), config=config)
        assert isinstance(dataset.index, HwCorpusIndex)
        assert dataset.config is config
        assert len(dataset.records) == 2
        assert dataset.n_max == DEFAULT_N_MAX


# ---------------------------------------------------------------------------
# Image normalisation
# ---------------------------------------------------------------------------
class TestImageNormalisation:
    """``"robust"`` by default; never total-energy."""

    def test_default_image_mode_is_robust(self, tmp_path: Path) -> None:
        """``config=None`` means ``image_mode="robust"``, not ``"abs255"``."""
        root = tmp_path / "mode"
        _write_zernike_run(root, "slm_zernike_shaping", [15], frames=[_frame(1.0, 40)])
        dataset = ZernikeCoeffDataset(_index(root))
        assert dataset.config.image_mode == "robust"
        image = dataset[0]["image"]
        assert float(image.min()) >= 0.0
        assert float(image.max()) <= 1.0

    def test_robust_mode_rescales_brightness(self, tmp_path: Path) -> None:
        """A dim frame and a bright frame land on comparable scales.

        This is the reason ``"robust"`` is the default here: the corpus spans a 2.5x
        spread in frame maxima before any physics, and the exposure is not a model
        input, so keeping absolute intensity would inject the detector gain.
        """
        root = tmp_path / "rescale"
        _write_zernike_run(
            root,
            "slm_zernike_shaping",
            [15, 15],
            frames=[_frame(0.2, 0), _frame(1.0, 0)],
        )
        dataset = ZernikeCoeffDataset(_index(root))
        dim = dataset[0]["image"]
        bright = dataset[1]["image"]
        assert float(dim.max()) > 0.0
        assert float(bright.max()) > 0.0
        # Both are normalised, so their peaks are the same order, unlike abs255.
        assert float(bright.max()) == pytest.approx(float(dim.max()), rel=0.35)

    def test_robust_mode_subtracts_the_read_noise_pedestal(
        self, tmp_path: Path
    ) -> None:
        """An additive pedestal is removed rather than rescaled into the signal."""
        root = tmp_path / "pedestal"
        _write_zernike_run(
            root, "slm_zernike_shaping", [15], frames=[_frame(1.0, offset=30)]
        )
        dataset = ZernikeCoeffDataset(_index(root), config=_config(image_mode="robust"))
        image = dataset[0]["image"]
        # A pure pedestal offset would leave the darkest pixel above zero under a
        # max-normalised mode; median subtraction drives it to (near) zero.
        assert float(image.min()) == pytest.approx(0.0, abs=0.15)

    def test_image_is_never_total_energy_normalised(self, tmp_path: Path) -> None:
        """A dim frame does not become brighter merely by normalisation.

        The discriminating pair is a *peak-normalised* frame and an *energy-equal*
        one. Under ``"sum"`` both would carry the same total weight regardless of
        how concentrated their light is, which is the known trap: MSE and PSNR then
        look excellent while the beam is wrong.

        ``"robust"`` divides by a high percentile of the pixels above the pedestal,
        so it is deliberately gain-invariant -- a dim frame and a bright frame with
        the *same shape* normalise to the same image. So brightness alone cannot
        separate the two modes. What does separate them is that a genuinely dim
        frame (almost no positive signal above the pedestal) falls back toward
        zeros, whereas ``"sum"`` would still hand it a full-unit total.
        """
        root = tmp_path / "energy"
        dead = np.full(_FRAME_SHAPE, 3, dtype=np.uint8)
        bright = _frame(1.0, 0)
        _write_zernike_run(root, "slm_zernike_shaping", [15, 15], frames=[dead, bright])
        dataset = ZernikeCoeffDataset(
            _index(root), config=_config(image_mode="robust")
        )
        dead_sum = float(dataset[0]["image"].sum())
        bright_sum = float(dataset[1]["image"].sum())

        # A near-flat frame has no beam to normalise, so it stays dark.
        assert dead_sum == pytest.approx(0.0, abs=1e-6)
        assert bright_sum > 1.0
        # The lit frame is peak-anchored, not energy-normalised: a 16x16 frame
        # whose total is far below 1.0 proves no division by the frame sum.
        assert float(dataset[1]["image"].max()) == pytest.approx(1.0, abs=1e-3)
        assert bright_sum < _FRAME_SHAPE[0] * _FRAME_SHAPE[1]

    def test_abs255_mode_is_available_and_keeps_absolute_scale(
        self, tmp_path: Path
    ) -> None:
        """An explicit ``"abs255"`` still gives the absolute 0-255 mapping."""
        root = tmp_path / "abs255"
        _write_zernike_run(
            root, "slm_zernike_shaping", [15], frames=[_frame(1.0, 0)]
        )
        dataset = ZernikeCoeffDataset(_index(root), config=_config(image_mode="abs255"))
        image = dataset[0]["image"]
        assert float(image.min()) >= 0.0
        assert float(image.max()) <= 1.0
        # A 16x16 uint8 ramp averaging ~125/255 stays well below 1.0.
        assert float(image.max()) < 1.0


# ---------------------------------------------------------------------------
# fit_coeff_stats
# ---------------------------------------------------------------------------
class TestFitCoeffStats:
    """Train-position-only, padded, degenerate-tail-safe statistics."""

    def test_mean_and_std_match_the_padded_training_rows(self, tmp_path: Path) -> None:
        """The returned vectors equal a plain ``mean``/``std`` of the padded rows."""
        root = tmp_path / "fit"
        lengths = [15, 36, 78, 15]
        _write_zernike_run(root, "slm_zernike_shaping", lengths, seed=1)
        index = _filtered(root)

        rows = np.stack(
            [pad_coefficients(_coeffs(n, seed=1 + i)) for i, n in enumerate(lengths)]
        )
        mean, std = fit_coeff_stats(index, list(range(len(index.records))))
        assert np.allclose(mean, rows.mean(axis=0))

        # The longest vector here is 78 terms, so indices 78..135 are zero in every
        # row and their empirical std is exactly 0. ``fit_coeff_stats`` must replace
        # that with 1.0 -- dividing by 0 would turn the padded tail into inf/nan --
        # so compare against the guarded expectation, not the raw std.
        longest = max(lengths)
        assert np.allclose(std[:longest], rows.std(axis=0)[:longest])
        assert np.array_equal(std[longest:], np.ones(_N_TERMS - longest))

    def test_statistics_use_only_the_supplied_positions(self, tmp_path: Path) -> None:
        """Records outside ``positions`` do not move the mean or the std."""
        root = tmp_path / "subset"
        _write_zernike_run(
            root, "slm_zernike_shaping", [15, 15, 15, 15], seed=2, scale=1.0
        )
        index = _filtered(root)

        train_mean, train_std = fit_coeff_stats(index, [0, 1])
        full_mean, full_std = fit_coeff_stats(index, [0, 1, 2, 3])
        assert not np.allclose(train_mean, full_mean)
        assert not np.allclose(train_std, full_std)

    def test_fitting_on_one_record_falls_back_to_identity(self, tmp_path: Path) -> None:
        """A single row leaves the std undefined, so it degenerates to identity."""
        root = tmp_path / "one"
        _write_zernike_run(root, "slm_zernike_shaping", [15])
        index = _filtered(root)
        mean, std = fit_coeff_stats(index, [0])
        assert np.array_equal(mean, np.zeros(_N_TERMS))
        assert np.array_equal(std, np.ones(_N_TERMS))

    def test_zero_variance_component_gets_unit_std(self, tmp_path: Path) -> None:
        """A coefficient no record varies must not divide by zero."""
        root = tmp_path / "const"
        payload = {
            0: {"_c": np.array([1.0, 2.0, 3.0]), "_img": _frame(), "exp_t": 1.0},
            1: {"_c": np.array([1.0, 2.0, 9.0]), "_img": _frame(), "exp_t": 1.0},
        }
        path = _write_pkl(tmp_path / "const" / _STAMP / "slm_zernike_shaping.pkl", payload)
        _write_sidecar(path, family="slm_zernike_shaping")
        index = _filtered(tmp_path / "const")

        mean, std = fit_coeff_stats(index, [0, 1])
        # Index 1 is constant across the two records (2.0, 2.0).
        assert std[1] == 1.0
        assert std[2] > 1.0
        # The padded tail is constant zero for both records.
        assert np.array_equal(std[3:], np.ones(_N_TERMS - 3))

    def test_respects_custom_n_max_length(self, tmp_path: Path) -> None:
        """The returned length follows ``n_max``, not the default."""
        root = tmp_path / "custom"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 15])
        index = _filtered(root)
        n_terms = calc_n_zernike_terms(_N_MAX_MEDIUM)
        mean, std = fit_coeff_stats(index, [0, 1], n_max=_N_MAX_MEDIUM)
        assert mean.shape == std.shape == (n_terms,)

    def test_rejects_empty_positions(self, tmp_path: Path) -> None:
        """Fitting on nothing is always a caller bug, so it is rejected."""
        root = tmp_path / "empty"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 15])
        with pytest.raises(ValueError, match="no positions"):
            fit_coeff_stats(_filtered(root), [])

    def test_rejects_out_of_range_position(self, tmp_path: Path) -> None:
        """A position past the end is an error naming the index size."""
        root = tmp_path / "oob"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 15])
        with pytest.raises(ValueError, match="out of range"):
            fit_coeff_stats(_filtered(root), [0, 99])

    def test_rejects_non_zernike_position(self, tmp_path: Path) -> None:
        """Fitting on a panel-sourced record is refused with the source named."""
        root = tmp_path / "panel"
        _write_zernike_run(root, "slm_zernike_shaping", [15])
        _write_pkl(
            root / _STAMP / "slm_pib_rad.pkl",
            {
                0: {
                    "_phase": np.zeros(_PANEL_SHAPE, dtype=np.float32),
                    "_img": _frame(),
                    "exp_t": 1.0,
                }
            },
        )
        _write_sidecar(root / _STAMP / "slm_pib_rad.pkl", family="slm_pib_rad")
        index = _index(root)
        position = next(
            i for i, record in enumerate(index.records) if record.source is PhaseSource.PANEL_RAD
        )
        with pytest.raises(ValueError, match="not 'zernike'"):
            fit_coeff_stats(index, [position])

    def test_rejects_record_without_coefficients(self, tmp_path: Path) -> None:
        """An indexed-but-empty payload is a real failure, not a silent skip."""
        root = tmp_path / "noc"
        path = _write_pkl(
            tmp_path / "noc" / _STAMP / "slm_zernike_shaping.pkl",
            {0: {"_c": [0.0] * 15, "_img": _frame(), "exp_t": 1.0}},
        )
        _write_sidecar(path, family="slm_zernike_shaping")
        index = _filtered(tmp_path / "noc")
        assert index.records[0].source is PhaseSource.ZERNIKE
        # Corrupt the pickle behind the index's back, as a truncated write would.
        payload = pickle.loads(path.read_bytes())
        del payload[0]["_c"]
        path.write_bytes(pickle.dumps(payload))
        with pytest.raises(ValueError, match="no '_c' key"):
            fit_coeff_stats(index, [0])


# ---------------------------------------------------------------------------
# Pickling / Windows spawn
# ---------------------------------------------------------------------------
class TestPickling:
    """Windows ``spawn`` re-pickles the Dataset per worker."""

    def test_round_trip_preserves_items_and_statistics(
        self, tmp_path: Path
    ) -> None:
        """An unpickled Dataset returns identical items, with statistics intact."""
        root = tmp_path / "pickle"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 36, 78], seed=4)
        index = _filtered(root)
        mean, std = fit_coeff_stats(index, [0, 1, 2])
        dataset = ZernikeCoeffDataset(
            index, config=_config(), coeff_mean=mean, coeff_std=std
        )

        restored = pickle.loads(pickle.dumps(dataset))
        assert len(restored) == len(dataset)
        assert restored.n_max == dataset.n_max
        assert np.array_equal(restored.coeff_mean, dataset.coeff_mean)
        assert np.array_equal(restored.coeff_std, dataset.coeff_std)
        for i in range(len(dataset)):
            original = dataset[i]
            copy = restored[i]
            assert torch.equal(original["coeffs"], copy["coeffs"])
            assert torch.equal(original["coeffs_raw"], copy["coeffs_raw"])
            assert torch.equal(original["image"], copy["image"])
            assert original["path"] == copy["path"]

    def test_state_carries_no_materialiser_or_mappingproxy(self, tmp_path: Path) -> None:
        """The pickled state has no materialiser, so no LRU arrays or handles ride along."""
        root = tmp_path / "state"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 15])
        dataset = ZernikeCoeffDataset(_index(root), config=_config())
        dataset[0]  # populate the LRU so a leak would be visible

        state = dataset.__getstate__()
        assert "_materialiser" not in state
        assert all(
            isinstance(record.sidecar, dict) for record in state["_records"]
        )
        # The live Dataset keeps them immutable, so a consumer cannot corrupt the
        # shared index by writing through dataset.records[i].sidecar.
        assert all(
            isinstance(record.sidecar, MappingProxyType)
            for record in dataset.records
        )

    def test_worker_spawn_produces_identical_batches(self, tmp_path: Path) -> None:
        """``num_workers=2`` yields the same batches as the in-process loader.

        This is the end-to-end guarantee behind the pickling contract: on Windows
        the Dataset is re-created in each worker, so a dropped statistic or an
        unpicklable sidecar would show up here as a mismatch or an exception.
        """
        root = tmp_path / "spawn"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 15, 36, 36], seed=6)
        index = _filtered(root)
        mean, std = fit_coeff_stats(index, [0, 1, 2, 3])
        kwargs: dict[str, Any] = {
            "config": _config(),
            "coeff_mean": mean,
            "coeff_std": std,
        }
        dataset = ZernikeCoeffDataset(index, **kwargs)

        sequential = DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=0
        )
        spawned = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
        for expected, actual in zip(sequential, spawned, strict=True):
            assert torch.equal(expected["coeffs"], actual["coeffs"])
            assert torch.equal(expected["image"], actual["image"])


# ---------------------------------------------------------------------------
# The two loader factories
# ---------------------------------------------------------------------------
class TestBuildZernikeCoeffDataloader:
    """The single-loader factory."""

    def test_filters_to_zernike_sources_automatically(self, tmp_path: Path) -> None:
        """Passing the unfiltered corpus still yields only Zernike records."""
        root = tmp_path / "auto"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 15])
        _write_pkl(
            root / _STAMP / "slm_pib_rad.pkl",
            {
                0: {
                    "_phase": np.zeros(_PANEL_SHAPE, dtype=np.float32),
                    "_img": _frame(),
                    "exp_t": 1.0,
                }
            },
        )
        _write_sidecar(root / _STAMP / "slm_pib_rad.pkl", family="slm_pib_rad")

        loader = build_zernike_coeff_dataloader(
            _index(root), config=_config(), batch_size=1, shuffle=False
        )
        assert len(loader.dataset) == 2  # type: ignore[arg-type]
        assert all(
            record.source is PhaseSource.ZERNIKE
            for record in loader.dataset.records  # type: ignore[attr-defined]
        )

    def test_batches_collate_without_a_custom_collate_fn(self, tmp_path: Path) -> None:
        """``default_collate`` handles the sample dict, so no override is needed."""
        root = tmp_path / "collate"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 36, 78, 15])
        loader = build_zernike_coeff_dataloader(
            _index(root),
            config=_config(),
            batch_size=2,
            shuffle=False,
            coeff_std=_std(),
        )
        batch = next(iter(loader))
        assert tuple(batch["coeffs"].shape) == (2, _N_TERMS)
        assert tuple(batch["coeffs_raw"].shape) == (2, _N_TERMS)
        assert tuple(batch["image"].shape) == (2, 1, _GRID, _GRID)
        assert batch["coeffs"].dtype is torch.float32
        assert batch["image"].dtype is torch.float32
        assert tuple(batch["n_terms_record"].shape) == (2,)
        assert isinstance(batch["path"], list)
        assert all(isinstance(value, str) for value in batch["path"])

    def test_sequential_order_is_index_order(self, tmp_path: Path) -> None:
        """``shuffle=False`` yields records in dataset order."""
        root = tmp_path / "order"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 36, 78])
        loader = build_zernike_coeff_dataloader(
            _index(root), config=_config(), batch_size=1, shuffle=False
        )
        seen = [int(batch["n_terms_record"][0]) for batch in loader]
        assert seen == [15, 36, 78]

    def test_shuffled_order_differs_from_index_order(self, tmp_path: Path) -> None:
        """``shuffle=True`` uses the file-grouped sampler and reorders within a file."""
        root = tmp_path / "shuffle"
        _write_zernike_run(root, "slm_zernike_shaping", [15] * 12)
        loader = build_zernike_coeff_dataloader(
            _index(root), config=_config(), batch_size=12, shuffle=True, seed=3
        )
        batch = next(iter(loader))
        assert tuple(batch["coeffs"].shape) == (12, _N_TERMS)

    def test_accepts_explicit_statistics(self, tmp_path: Path) -> None:
        """Statistics supplied by the caller reach the Dataset unchanged."""
        root = tmp_path / "explicit"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 15, 15], seed=8)
        index = _filtered(root)
        mean, std = fit_coeff_stats(index, [0, 1, 2])
        loader = build_zernike_coeff_dataloader(
            index, config=_config(), batch_size=3, shuffle=False, coeff_mean=mean, coeff_std=std
        )
        assert np.array_equal(loader.dataset.coeff_mean, mean)  # type: ignore[attr-defined]

    def test_honours_a_custom_n_max(self, tmp_path: Path) -> None:
        """``n_max`` flows through the factory to the batch width."""
        root = tmp_path / "nmaxflow"
        _write_zernike_run(root, "slm_zernike_shaping", [15, 15])
        n_terms = calc_n_zernike_terms(_N_MAX_MEDIUM)
        loader = build_zernike_coeff_dataloader(
            _index(root),
            config=_config(),
            batch_size=2,
            shuffle=False,
            n_max=_N_MAX_MEDIUM,
            coeff_std=np.ones(n_terms),
        )
        batch = next(iter(loader))
        assert tuple(batch["coeffs"].shape) == (2, n_terms)


class TestCreateZernikeCoeffDataloaders:
    """The split factory and its leakage guarantee."""

    @staticmethod
    def _file_corpus(root: Path, n_files: int) -> Path:
        """Write ``n_files`` single-record files, each with its own coefficient scale.

        The per-file ``scale`` is what makes a leakage regression detectable: a
        statistic fitted on files it should not have seen differs numerically from
        one fitted on the train files alone.

        Args:
            root: Corpus root.
            n_files: How many pickles to write.

        Returns:
            The corpus root, for chaining.
        """
        for i in range(n_files):
            _write_zernike_run(
                root, f"slm_zernike_shaping_{i}", [15], seed=100 + i, scale=1.0 + i
            )
        return root

    @staticmethod
    def _subset(loader: DataLoader[Any]) -> Subset[Any]:
        """Return the ``Subset`` a split loader iterates.

        The split loaders wrap one shared Dataset in three ``Subset``s, so the
        statistics and record list live one level down.

        Args:
            loader: A loader returned by :func:`create_zernike_coeff_dataloaders`.

        Returns:
            The loader's ``Subset``.
        """
        subset = loader.dataset
        assert isinstance(subset, Subset)
        return subset

    @classmethod
    def _base_dataset(cls, loader: DataLoader[Any]) -> Any:
        """Unwrap the base ``ZernikeCoeffDataset`` from a ``Subset``.

        Args:
            loader: A loader returned by :func:`create_zernike_coeff_dataloaders`.

        Returns:
            The shared :class:`~ml.hwdataset.zernike_dataset.ZernikeCoeffDataset`.
        """
        return cls._subset(loader).dataset

    @classmethod
    def _subset_paths(cls, loader: DataLoader[Any]) -> set[str]:
        """Collect the distinct pickle paths a split covers.

        Args:
            loader: A loader returned by :func:`create_zernike_coeff_dataloaders`.

        Returns:
            The ``str`` paths of the pickles in this split.
        """
        subset = cls._subset(loader)
        base = cls._base_dataset(loader)
        return {str(base.records[i].path) for i in subset.indices}

    def test_returns_three_disjoint_loaders(self, tmp_path: Path) -> None:
        """Every record lands in exactly one split, and the split is by file.

        Uses ten files because the default ``train_split=0.8`` / ``val_split=0.1``
        floors to 8 / 1 / 1 -- three files would floor to 2 / 0 / 1 and be rejected,
        which is the sibling factory's arithmetic, so the default corpus in tests
        has to be big enough for it.
        """
        root = self._file_corpus(tmp_path / "ten", 10)
        train, val, test = create_zernike_coeff_dataloaders(
            root, config=_config(), batch_size=1
        )
        assert len(self._subset(train)) == 8
        assert len(self._subset(val)) == 1
        assert len(self._subset(test)) == 1

        splits = [self._subset_paths(loader) for loader in (train, val, test)]
        # Disjoint by file, and together they cover every file exactly once. The
        # comparison is on full paths because ``_subset_paths`` returns them.
        assert set().union(*splits) == {str(p) for p in root.rglob("*.pkl")}
        assert not splits[0] & splits[1]
        assert not splits[0] & splits[2]
        assert not splits[1] & splits[2]

    def test_split_is_reproducible_for_a_fixed_seed(self, tmp_path: Path) -> None:
        """The same seed assigns the same files to the same splits."""
        root = self._file_corpus(tmp_path / "repro", 10)
        first = create_zernike_coeff_dataloaders(root, config=_config(), batch_size=1)
        second = create_zernike_coeff_dataloaders(root, config=_config(), batch_size=1)
        for a, b in zip(first, second, strict=True):
            assert self._subset_paths(a) == self._subset_paths(b)

    def test_a_different_seed_can_move_the_boundary_files(
        self, tmp_path: Path
    ) -> None:
        """The split is a function of the seed, not of the file order."""
        root = self._file_corpus(tmp_path / "seeded", 10)
        first = create_zernike_coeff_dataloaders(
            root, config=_config(), batch_size=1, seed=0
        )
        second = create_zernike_coeff_dataloaders(
            root, config=_config(), batch_size=1, seed=7
        )
        assert self._subset_paths(first[0]) != self._subset_paths(second[0])

    def test_val_and_test_loaders_are_sequential(self, tmp_path: Path) -> None:
        """Validation and test metrics must be reproducible, so no shuffling there."""
        root = self._file_corpus(tmp_path / "seq", 10)
        train, val, test = create_zernike_coeff_dataloaders(
            root, config=_config(), batch_size=1
        )
        assert isinstance(val.sampler, SequentialSampler)
        assert isinstance(test.sampler, SequentialSampler)
        assert not isinstance(train.sampler, SequentialSampler)

    def test_statistics_are_fitted_on_train_positions_only(
        self, tmp_path: Path
    ) -> None:
        """The Dataset's statistics match a fit over the train files alone.

        This is the leakage guard. If the factory fitted on the whole corpus, the
        mean over all ten files would differ from the mean over the eight train
        files alone -- a difference large enough to catch, since every file was
        written with a different coefficient scale.
        """
        root = self._file_corpus(tmp_path / "leak", 10)
        train, _, _ = create_zernike_coeff_dataloaders(
            root, config=_config(), batch_size=1, seed=0
        )
        dataset = self._base_dataset(train)

        train_paths = self._subset_paths(train)
        full_index = _filtered(root)
        train_positions = [
            i
            for i, record in enumerate(full_index.records)
            if str(record.path) in train_paths
        ]
        expected_mean, expected_std = fit_coeff_stats(full_index, train_positions)

        assert np.allclose(dataset.coeff_mean, expected_mean)
        assert np.allclose(dataset.coeff_std, expected_std)

        # Sanity: a full-corpus fit would give a different mean, so this test can
        # actually fail if the factory ever regresses to fitting on everything.
        full_mean, _ = fit_coeff_stats(full_index, list(range(len(full_index.records))))
        assert not np.allclose(dataset.coeff_mean, full_mean)

    def test_train_loader_groups_by_file(self, tmp_path: Path) -> None:
        """The train loader samples file-by-file, which is what the LRU wants."""
        root = self._file_corpus(tmp_path / "grouped", 10)
        train, _, _ = create_zernike_coeff_dataloaders(
            root, config=_config(), batch_size=1
        )
        assert isinstance(train.sampler, FileGroupedSampler)

    def test_rejects_too_few_files_for_the_requested_split(
        self, tmp_path: Path
    ) -> None:
        """A corpus too small for the fractions fails loudly instead of silently."""
        root = self._file_corpus(tmp_path / "tiny", 3)
        with pytest.raises(ValueError, match="every split needs at least one file"):
            create_zernike_coeff_dataloaders(root, config=_config(), batch_size=1)

    def test_rejects_bad_split_fractions(self, tmp_path: Path) -> None:
        """Fractions that would leave a split empty are rejected with a reason."""
        root = self._file_corpus(tmp_path / "bad", 10)
        with pytest.raises(ValueError, match="val_split must be"):
            create_zernike_coeff_dataloaders(
                root, config=_config(), train_split=0.5, val_split=1.5
            )
        with pytest.raises(ValueError, match="leave room"):
            create_zernike_coeff_dataloaders(
                root, config=_config(), train_split=0.8, val_split=0.5
            )

    def test_rejects_corpus_without_zernike_records(self, tmp_path: Path) -> None:
        """A corpus with no Zernike record is a clear error, not an empty loader."""
        root = tmp_path / "panelonly"
        _write_pkl(
            root / _STAMP / "slm_pib_rad.pkl",
            {
                0: {
                    "_phase": np.zeros(_PANEL_SHAPE, dtype=np.float32),
                    "_img": _frame(),
                    "exp_t": 1.0,
                }
            },
        )
        _write_sidecar(root / _STAMP / "slm_pib_rad.pkl", family="slm_pib_rad")
        with pytest.raises(ValueError, match="no Zernike-sourced record"):
            create_zernike_coeff_dataloaders(root, config=_config())
