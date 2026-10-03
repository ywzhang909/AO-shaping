"""Unit tests for :mod:`ml.hwdataset.dataset` (PyTorch Dataset / DataLoader).

Every fixture is a **synthetic** pickle written into a pytest ``tmp_path``, so this
suite never touches the real ``data/debug`` corpus (65.88 GB, 265 files, absent in
CI). Panels are deliberately tiny -- ``(40, 60)`` instead of the real
``(1200, 1920)`` -- and every test therefore passes a small
:class:`~ml.hwdataset.records.MaterialiserConfig`
(``panel_center=(30, 20)``, ``panel_radius=20``, ``grid=8``) so the production ROI
(``(960, 600)`` / ``r=500``) never misses the tiny panel.

``torch`` is a hard dependency of the module under test, so it is imported at
module scope and the file is skipped wholesale when it is missing.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, Subset  # noqa: E402

from ml.hwdataset.dataset import (  # noqa: E402
    DEFAULT_PREFETCH_FACTOR,
    FileGroupedSampler,
    HwPhaseImageDataset,
    build_hw_dataloader,
    create_hw_dataloaders,
)
from ml.hwdataset.cache import prepare_hw_cache  # noqa: E402
from ml.hwdataset.index import (  # noqa: E402
    HwCorpusIndex,
    HwRecordRef,
    PhaseSource,
    build_hw_index,
)
from ml.hwdataset.records import MaterialiserConfig  # noqa: E402

_STAMP = "20260101_000000"
_PANEL_SHAPE = (40, 60)  # (height, width), stands in for (1200, 1920)
_FRAME_SHAPE = (16, 16)
_GRID = 8


# ---------------------------------------------------------------------------
# Fixture writers (module level, mirroring test_index.py)
# ---------------------------------------------------------------------------
def _panel(shape: tuple[int, int] = _PANEL_SHAPE) -> np.ndarray:
    """Return a small ``float32`` radian panel wrapped to ``[0, 2pi)``.

    Args:
        shape: Panel shape.

    Returns:
        A ``float32`` array shaped like the real panels.
    """
    flat = np.arange(shape[0] * shape[1], dtype=np.float32) % 6.2831853
    return flat.reshape(shape).astype(np.float32)


def _gray_panel(shape: tuple[int, int] = _PANEL_SHAPE) -> np.ndarray:
    """Return a small ``uint16`` SLM grayscale panel.

    Args:
        shape: Panel shape.

    Returns:
        A ``uint16`` array shaped like the real ``slm_pib_*`` panels.
    """
    flat = np.arange(shape[0] * shape[1], dtype=np.int64) % 994
    return flat.reshape(shape).astype(np.uint16)


def _frame(shape: tuple[int, int] = _FRAME_SHAPE, scale: int = 1) -> np.ndarray:
    """Return a small ``uint8`` CCD frame with a controllable peak.

    Args:
        shape: Frame shape.
        scale: Multiplier on the ramp, so two records can differ in absolute
            brightness (used to prove the image is not peak-normalised).

    Returns:
        A ``uint8`` array shaped like the real far-field windows.
    """
    ramp = (np.arange(shape[0] * shape[1], dtype=np.int64) % 251) * int(scale)
    return np.clip(ramp, 0, 255).astype(np.uint8).reshape(shape)


def _config(grid: int = _GRID, image_mode: str = "abs255") -> MaterialiserConfig:
    """Return a tiny-panel :class:`MaterialiserConfig`.

    Args:
        grid: Output side length.
        image_mode: Far-field scaling mode.

    Returns:
        A config whose ROI fits the ``(40, 60)`` synthetic panels.
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


def _write_sidecar(pkl_path: Path, name: str, **fields: Any) -> Path:
    """Write a JSON sidecar beside ``pkl_path``.

    Args:
        pkl_path: The pickle the sidecar describes.
        name: Sidecar file name (usually ``"<stem>.json"``).
        fields: JSON object fields.

    Returns:
        The written JSON path.
    """
    sidecar = pkl_path.with_name(name)
    sidecar.write_text(json.dumps(fields), encoding="utf-8")
    return sidecar


def _record(
    *,
    phase: np.ndarray | None = None,
    frame: np.ndarray | None = None,
    exposure_ms: float | None = None,
) -> dict[str, Any]:
    """Build one synthetic record.

    Args:
        phase: Optional commanded panel (float32 rad -> PANEL_RAD, uint16 ->
            PANEL_GRAY).
        frame: Optional CCD frame; defaults to :func:`_frame`.
        exposure_ms: Optional ``exp_t`` in ms.

    Returns:
        A record mapping with at least a 2-D ``_img``.
    """
    record: dict[str, Any] = {"_img": _frame() if frame is None else frame}
    if phase is not None:
        record["_phase"] = phase
    if exposure_ms is not None:
        record["exp_t"] = float(exposure_ms)
    return record


def _write_rad_run(
    root: Path,
    stem: str,
    exposures: Sequence[float | None],
    frames: Sequence[np.ndarray] | None = None,
) -> Path:
    """Write one ``dict[int, dict]`` file of PANEL_RAD records.

    Args:
        root: Family directory.
        stem: File stem.
        exposures: One ``exp_t`` per record (``None`` to omit).
        frames: Optional per-record frames; defaults to :func:`_frame`.

    Returns:
        The written ``.pkl`` path.
    """
    records = []
    for i, exposure in enumerate(exposures):
        frame = None if frames is None else frames[i]
        records.append(_record(phase=_panel(), frame=frame, exposure_ms=exposure))
    payload = {i: rec for i, rec in enumerate(records)}
    return _write_pkl(root / _STAMP / f"{stem}.pkl", payload)


def _refs_by_path(index: HwCorpusIndex) -> dict[Path, list[HwRecordRef]]:
    """Group an index's records by source file.

    Args:
        index: Corpus index.

    Returns:
        Mapping of file path -> its records, in index order.
    """
    grouped: dict[Path, list[HwRecordRef]] = {}
    for record in index.records:
        grouped.setdefault(record.path, []).append(record)
    return grouped


def test_index_and_dataset_discriminate_panel_sources(tmp_path: Path) -> None:
    """float32-radian panels and uint16 grayscale panels get distinct sources."""
    root = tmp_path / "sources"
    _write_rad_run(root, "zern_rad", [0.5, 0.5])
    gray = tmp_path / "sources" / _STAMP / "slm_pib_gray.pkl"
    _write_pkl(
        gray,
        {0: _record(phase=_gray_panel(), exposure_ms=0.5)},
    )

    index = build_hw_index(root, progress_every=0)
    grouped = _refs_by_path(index)
    assert len(grouped) == 2

    dataset = HwPhaseImageDataset(index, config=_config())
    sources = {Path(dataset[i]["path"]).name: dataset[i]["source"] for i in range(len(dataset))}
    assert sources["zern_rad.pkl"] == PhaseSource.PANEL_RAD.value
    assert sources["slm_pib_gray.pkl"] == PhaseSource.PANEL_GRAY.value
    assert PhaseSource.PANEL_RAD.value != PhaseSource.PANEL_GRAY.value

    # Both still yield well-formed, contiguous phase tensors.
    for position in range(len(dataset)):
        for key in ("phase_cos", "phase_sin"):
            tensor = dataset[position][key]
            assert tuple(tensor.shape) == (1, _GRID, _GRID)
            assert tensor.is_contiguous()


# ---------------------------------------------------------------------------
# 1-5. FileGroupedSampler
# ---------------------------------------------------------------------------
def test_sampler_keeps_file_groups_contiguous(tmp_path: Path) -> None:
    """Every file's positions must appear as one unbroken run in an epoch."""
    root = tmp_path / "grp"
    for k in range(3):
        _write_rad_run(root, f"run_{k}", [0.5, 0.5, 0.5])
    index = build_hw_index(root, progress_every=0)
    sampler = FileGroupedSampler(index.records, seed=1)
    order = list(sampler)

    assert len(order) == len(index.records)
    assert sorted(order) == list(range(len(index.records)))

    # Walk the order and assert no path is revisited after leaving it.
    path_of = {i: record.path for i, record in enumerate(index.records)}
    seen_paths: list[Path] = []
    for position in order:
        path = path_of[position]
        if not seen_paths or seen_paths[-1] != path:
            assert path not in seen_paths, "a file group was split across the epoch"
            seen_paths.append(path)


def test_sampler_shuffles_within_group_and_advances_each_epoch(tmp_path: Path) -> None:
    """Two epochs differ (generator advances) and each group's order is permuted."""
    root = tmp_path / "adv"
    _write_rad_run(root, "run_0", [0.5] * 6)
    index = build_hw_index(root, progress_every=0)
    sampler = FileGroupedSampler(index.records, seed=7)
    first = list(sampler)
    second = list(sampler)

    assert first != second, "the generator did not advance between epochs"
    # Within a single group of 6, a random permutation is very unlikely to be the
    # identity twice; assert the *set* is stable and at least one order changed.
    assert sorted(first) == sorted(second) == list(range(6))


def test_sampler_reproducible_for_same_seed(tmp_path: Path) -> None:
    """Two samplers built with the same seed replay the identical sequence."""
    root = tmp_path / "repro"
    for k in range(3):
        _write_rad_run(root, f"run_{k}", [0.5] * 4)
    index = build_hw_index(root, progress_every=0)
    a = list(FileGroupedSampler(index.records, seed=99))
    b = list(FileGroupedSampler(index.records, seed=99))
    assert a == b


def test_sampler_num_samples_trim_and_repeat(tmp_path: Path) -> None:
    """``num_samples`` shorter/longer than the record count trims/repeats cleanly."""
    root = tmp_path / "ns"
    for k in range(2):
        _write_rad_run(root, f"run_{k}", [0.5] * 4)  # 8 records total
    index = build_hw_index(root, progress_every=0)
    total = len(index.records)

    short = FileGroupedSampler(index.records, num_samples=total - 2, seed=3)
    assert len(short) == total - 2
    assert len(list(short)) == total - 2

    exact = FileGroupedSampler(index.records, num_samples=total, seed=3)
    assert len(list(exact)) == total

    long = FileGroupedSampler(index.records, num_samples=total * 2 + 3, seed=3)
    emitted = list(long)
    assert len(emitted) == total * 2 + 3

    zero = FileGroupedSampler(index.records, num_samples=0, seed=3)
    assert len(zero) == 0
    assert list(zero) == []


def test_sampler_rejects_negative_num_samples(tmp_path: Path) -> None:
    """A negative ``num_samples`` is a caller error, not silently clamped."""
    root = tmp_path / "neg"
    _write_rad_run(root, "run_0", [0.5])
    index = build_hw_index(root, progress_every=0)
    with pytest.raises(ValueError, match="num_samples"):
        FileGroupedSampler(index.records, num_samples=-1)


# ---------------------------------------------------------------------------
# 6-7. Sample structure, freshness, contiguity
# ---------------------------------------------------------------------------
def test_dataset_sample_keys_shapes_dtypes(tmp_path: Path) -> None:
    """One item carries exactly the documented keys with the documented layout."""
    root = tmp_path / "struct"
    _write_rad_run(root, "run_0", [0.5, 1.0])
    index = build_hw_index(root, progress_every=0)
    dataset = HwPhaseImageDataset(index, config=_config())

    assert len(dataset) == 2
    sample = dataset[0]
    expected = {
        "phase_cos",
        "phase_sin",
        "image",
        "exposure_ms",
        "exposure_log10",
        "contrast",
        "source",
        "family",
        "sample_idx",
        "path",
        "fov_px",
    }
    assert set(sample) == expected

    for key in ("phase_cos", "phase_sin", "image"):
        tensor = sample[key]
        assert isinstance(tensor, torch.Tensor)
        assert tuple(tensor.shape) == (1, _GRID, _GRID)
        assert tensor.dtype is torch.float32
        assert tensor.is_contiguous()

    log = sample["exposure_log10"]
    assert tuple(log.shape) == (1,)
    assert log.dtype is torch.float32

    assert isinstance(sample["contrast"], np.ndarray)
    assert sample["contrast"].shape == (_GRID, _GRID)
    assert isinstance(sample["exposure_ms"], float)
    assert sample["source"] == PhaseSource.PANEL_RAD.value
    assert isinstance(sample["path"], str) and sample["path"].endswith(".pkl")
    assert sample["sample_idx"] == 0


def test_dataset_tensors_are_fresh_contiguous_and_mutable(tmp_path: Path) -> None:
    """Two calls must not share memory, and mutating one must not affect the other."""
    root = tmp_path / "fresh"
    _write_rad_run(root, "run_0", [0.5, 0.5])
    index = build_hw_index(root, progress_every=0)
    dataset = HwPhaseImageDataset(index, config=_config(), cache_size=1)

    first = dataset[0]
    second = dataset[0]
    # Fresh buffer: distinct storage.
    assert first["image"].data_ptr() != second["image"].data_ptr()
    assert first["phase_cos"].data_ptr() != second["phase_cos"].data_ptr()

    before = second["image"].clone()
    with torch.no_grad():
        first["image"].add_(1.0)
    assert torch.equal(second["image"], before), "mutation leaked into a later sample"
    assert first["image"].is_contiguous()


# ---------------------------------------------------------------------------
# 8. Standardised log exposure
# ---------------------------------------------------------------------------
def test_exposure_log10_is_standardised(tmp_path: Path) -> None:
    """``exposure_log10 == (log10(ms) - mean) / std`` from index-level statistics."""
    root = tmp_path / "expo"
    # log10 of these exposures: -1, 0, 1 -> mean 0, population std sqrt(2/3)
    # (NumPy's default ddof=0, matching ``HwPhaseImageDataset._exposure_stats``).
    exposures = [0.1, 1.0, 10.0]
    log_std = float(np.std([-1.0, 0.0, 1.0]))
    _write_rad_run(root, "run_0", exposures)
    index = build_hw_index(root, progress_every=0)
    dataset = HwPhaseImageDataset(index, config=_config())

    assert dataset.exposure_log_mean == pytest.approx(0.0)
    assert dataset.exposure_log_std == pytest.approx(log_std)
    for position, expected in enumerate((-1.0, 0.0, 1.0)):
        sample = dataset[position]
        assert sample["exposure_ms"] == pytest.approx(exposures[position])
        value = float(sample["exposure_log10"].item())
        assert value == pytest.approx(expected / log_std, abs=1e-6)


def test_unknown_exposure_yields_zero_and_is_filtered_by_default(tmp_path: Path) -> None:
    """``require_exposure`` drops unknown-exposure records; keeping them yields 0.0."""
    root = tmp_path / "noexpo"
    _write_rad_run(root, "run_0", [1.0, None, 1.0])  # one record has no exp_t
    index = build_hw_index(root, progress_every=0)
    total_index = len(index.records)

    # Default (require_exposure=True) drops the None record.
    kept = HwPhaseImageDataset(index, config=_config())
    assert len(kept) == total_index - 1
    assert all(record.exposure_ms is not None for record in kept.records)

    # require_exposure=False keeps it, and its standardised exposure is 0.0.
    all_records = HwPhaseImageDataset(
        index, config=_config(), require_exposure=False
    )
    assert len(all_records) == total_index
    unknown = [i for i in range(total_index) if all_records[i]["exposure_ms"] is None]
    assert len(unknown) == 1
    value = float(all_records[unknown[0]]["exposure_log10"].item())
    assert value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. Empty-corpus guard
# ---------------------------------------------------------------------------
def test_dataset_rejects_empty_filtered_index(tmp_path: Path) -> None:
    """An index with no exposure-carrying records raises a guiding ``ValueError``."""
    root = tmp_path / "empty"
    _write_rad_run(root, "run_0", [None])  # single record, no exposure
    index = build_hw_index(root, progress_every=0)
    with pytest.raises(ValueError, match="no usable records"):
        HwPhaseImageDataset(index, config=_config(), require_exposure=True)


# ---------------------------------------------------------------------------
# 10. Absolute (non-peak-normalised) image
# ---------------------------------------------------------------------------
def test_image_is_absolute_not_peak_normalised(tmp_path: Path) -> None:
    """Two records whose frames differ only in brightness keep different images."""
    root = tmp_path / "abs"
    dim = _frame(scale=1)
    bright = _frame(scale=2)
    _write_rad_run(root, "run_0", [0.5, 0.5], frames=[dim, bright])
    index = build_hw_index(root, progress_every=0)
    dataset = HwPhaseImageDataset(index, config=_config())

    image_dim = dataset[0]["image"]
    image_bright = dataset[1]["image"]
    # abs255 keeps absolute intensity -> the brighter frame yields a brighter image.
    assert image_bright.sum().item() > image_dim.sum().item()
    # Both remain within [0, 1] because uint8 is scaled by /255.
    assert float(image_bright.max()) <= 1.0 + 1e-6

    # Under "peak" normalisation the two would both max to 1.0 (contrast-only).
    peak_dataset = HwPhaseImageDataset(
        index, config=_config(image_mode="peak")
    )
    assert float(peak_dataset[0]["image"].max()) == pytest.approx(1.0)
    assert float(peak_dataset[1]["image"].max()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 11. fov_px from the source frame geometry
# ---------------------------------------------------------------------------
def test_fov_px_reflects_source_frame_side(tmp_path: Path) -> None:
    """``fov_px`` is the smaller side of the raw ``_img`` that was materialised."""
    root = tmp_path / "fov"
    rect = _frame(shape=(24, 32))  # deliberately non-square
    _write_rad_run(root, "run_0", [0.5], frames=[rect])
    index = build_hw_index(root, progress_every=0)
    dataset = HwPhaseImageDataset(index, config=_config())
    sample = dataset[0]
    assert sample["fov_px"] == min(rect.shape) == 24


def test_cached_and_direct_datasets_return_identical_samples(tmp_path: Path) -> None:
    """A cache-served Dataset must be indistinguishable from a direct one.

    This is the end-to-end guarantee behind ``use_cache=True``: the whole reason
    the cache may be consulted by default is that it changes speed, never values.
    ``fov_px`` is compared too, because it is the one field that used to require
    re-opening the pickle and is now carried on the sample.
    """
    root = tmp_path / "cached_epoch"
    _write_rad_run(root, "run_0", [0.5, 1.0], frames=[_frame(shape=(24, 32))] * 2)
    _write_rad_run(root, "run_1", [2.0], frames=[_frame(shape=(16, 16))])
    index = build_hw_index(root, progress_every=0)
    config = _config()

    built = prepare_hw_cache(index, config=config, log_every=0)
    assert built, "the fixture cache was not built"

    direct = HwPhaseImageDataset(index, config=config, use_cache=False)
    served = HwPhaseImageDataset(index, config=config, use_cache=True)
    try:
        assert len(direct) == len(served) == 3
        for i in range(len(direct)):
            want = direct[i]
            got = served[i]
            for key in ("phase_cos", "phase_sin", "image", "contrast"):
                assert np.array_equal(want[key], got[key]), (i, key)
            assert want["exposure_ms"] == got["exposure_ms"]
            assert want["exposure_log10"].tolist() == got["exposure_log10"].tolist()
            assert want["fov_px"] == got["fov_px"]
            for key in ("source", "family", "sample_idx", "path"):
                assert want[key] == got[key], (i, key)
    finally:
        direct.clear_cache()
        served.clear_cache()


def test_dataset_falls_back_when_no_cache_exists(tmp_path: Path) -> None:
    """Default ``use_cache=True`` must not break a corpus with no cache built."""
    root = tmp_path / "no_cache"
    _write_rad_run(root, "run_0", [0.5, 1.0])
    index = build_hw_index(root, progress_every=0)
    dataset = HwPhaseImageDataset(index, config=_config())
    assert dataset._materialiser.use_cache is True
    grid = _config().grid
    assert tuple(dataset[0]["image"].shape) == (1, grid, grid)


def test_dataset_use_cache_survives_pickling(tmp_path: Path) -> None:
    """Windows ``spawn`` rebuilds the Dataset per worker.

    If ``__setstate__`` dropped the flag, every worker would silently fall back to
    the slow direct path for the whole epoch -- correct, but 6x slower with no
    symptom to explain it.
    """
    root = tmp_path / "spawn"
    _write_rad_run(root, "run_0", [0.5, 1.0])
    index = build_hw_index(root, progress_every=0)
    dataset = HwPhaseImageDataset(index, config=_config(), use_cache=True)
    revived = pickle.loads(pickle.dumps(dataset))
    assert revived._materialiser.use_cache is True
    grid = _config().grid
    assert tuple(revived[0]["image"].shape) == (1, grid, grid)


# ---------------------------------------------------------------------------
# 12. Index semantics
# ---------------------------------------------------------------------------
def test_negative_index_and_bounds(tmp_path: Path) -> None:
    """Negative indices wrap; out-of-range raises ``IndexError``; non-int raises."""
    root = tmp_path / "idx"
    _write_rad_run(root, "run_0", [0.5, 1.0, 2.0])
    index = build_hw_index(root, progress_every=0)
    dataset = HwPhaseImageDataset(index, config=_config())

    assert dataset[-1]["sample_idx"] == len(dataset) - 1
    assert torch.equal(dataset[-1]["image"], dataset[len(dataset) - 1]["image"])

    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]
    with pytest.raises(IndexError):
        _ = dataset[-len(dataset) - 1]

    bad_index: Any = "not-an-int"
    with pytest.raises(TypeError):
        _ = dataset[bad_index]


# ---------------------------------------------------------------------------
# 13. Windows spawn-safe pickling
# ---------------------------------------------------------------------------
def test_dataset_pickles_for_spawn_and_round_trips(tmp_path: Path) -> None:
    """The Dataset must pickle (despite ``MappingProxyType``) and rebuild itself."""
    root = tmp_path / "pickle"
    _write_rad_run(root, "run_0", [0.5, 1.0])
    _write_rad_run(root, "run_1", [2.0, 4.0])
    index = build_hw_index(root, progress_every=0)
    dataset = HwPhaseImageDataset(index, config=_config())

    # Populate the LRU so a naive pickle would try to ship a cached payload.
    _ = dataset[0]
    state = dataset.__getstate__()
    assert "_materialiser" not in state, "the materialiser/LRU must not be pickled"
    assert "_index" not in state, "the index is rebuilt, not pickled"
    for record in state["_records"]:
        assert isinstance(record.sidecar, Mapping)
        assert not isinstance(record.sidecar, type(__import__("types").MappingProxyType({})))

    # A real round trip through pickle must succeed and reproduce a sample.
    restored = pickle.loads(pickle.dumps(dataset))
    assert len(restored) == len(dataset)
    assert restored._materialiser is not None
    a = dataset[1]
    b = restored[1]
    assert torch.equal(a["image"], b["image"])
    assert torch.equal(a["phase_cos"], b["phase_cos"])
    assert a["exposure_ms"] == b["exposure_ms"]
    assert a["path"] == b["path"]


# ---------------------------------------------------------------------------
# 14-15. build_hw_dataloader
# ---------------------------------------------------------------------------
def test_build_dataloader_collates_batch_of_four_tensors(tmp_path: Path) -> None:
    """default_collate stacks the four tensors to ``(B, 1, grid, grid)`` / ``(B, 1)``."""
    root = tmp_path / "loader"
    _write_rad_run(root, "run_0", [0.5, 1.0, 2.0, 4.0])
    index = build_hw_index(root, progress_every=0)
    loader = build_hw_dataloader(index, config=_config(), batch_size=2, num_workers=0)

    batch = next(iter(loader))
    assert tuple(batch["phase_cos"].shape) == (2, 1, _GRID, _GRID)
    assert tuple(batch["phase_sin"].shape) == (2, 1, _GRID, _GRID)
    assert tuple(batch["image"].shape) == (2, 1, _GRID, _GRID)
    assert tuple(batch["exposure_log10"].shape) == (2, 1)
    assert batch["phase_cos"].dtype is torch.float32
    # Provenance collates to python lists of scalars / strings.
    assert isinstance(batch["source"], list) and len(batch["source"]) == 2
    assert isinstance(batch["path"], list) and all(p.endswith(".pkl") for p in batch["path"])
    assert isinstance(batch["fov_px"], torch.Tensor) and int(batch["fov_px"][0]) == 16


def test_build_dataloader_shuffle_selects_sampler_and_prefetch(tmp_path: Path) -> None:
    """``shuffle`` picks the grouped sampler; ``prefetch_factor`` set only w/ workers."""
    root = tmp_path / "flags"
    for k in range(2):
        _write_rad_run(root, f"run_{k}", [0.5] * 3)
    index = build_hw_index(root, progress_every=0)

    shuffled = build_hw_dataloader(index, config=_config(), batch_size=2, shuffle=True)
    assert isinstance(shuffled.sampler, FileGroupedSampler)

    sequential = build_hw_dataloader(index, config=_config(), batch_size=2, shuffle=False)
    assert not isinstance(sequential.sampler, FileGroupedSampler)

    assert DEFAULT_PREFETCH_FACTOR == 2
    # num_workers=0 path must not set prefetch_factor (torch would reject it).
    assert sequential.prefetch_factor is None or sequential.prefetch_factor == 0


# ---------------------------------------------------------------------------
# 16-17. create_hw_dataloaders: seeded, file-level split
# ---------------------------------------------------------------------------
def _four_file_corpus(tmp_path: Path) -> Path:
    """Write a four-file synthetic corpus.

    Args:
        tmp_path: pytest tmp dir.

    Returns:
        The family root containing the four ``.pkl`` files.
    """
    root = tmp_path / "split_corpus"
    for k in range(4):
        _write_rad_run(root, f"run_{k}", [0.5, 1.0, 1.5])
    return root


def _loader_paths(loader: DataLoader) -> set[Path]:
    """Return the set of source files a loader's underlying Dataset touches.

    Args:
        loader: A loader built over a ``Subset`` of ``HwPhaseImageDataset``.

    Returns:
        The distinct source ``.pkl`` paths in the loader's partition.
    """
    subset = loader.dataset
    assert isinstance(subset, Subset)
    base = subset.dataset
    assert isinstance(base, HwPhaseImageDataset)
    return {base.records[i].path for i in subset.indices}


def test_create_dataloaders_file_level_split_is_disjoint_and_seeded(
    tmp_path: Path,
) -> None:
    """The three loaders partition whole FILES, reproducibly for a fixed seed."""
    root = _four_file_corpus(tmp_path)

    tr, va, te = create_hw_dataloaders(
        root, config=_config(), batch_size=2, train_split=0.5, val_split=0.25, seed=11
    )
    tr_paths = _loader_paths(tr)
    va_paths = _loader_paths(va)
    te_paths = _loader_paths(te)

    assert tr_paths and va_paths and te_paths, "every split must be non-empty"
    assert not (tr_paths & va_paths), "train/val share a file"
    assert not (tr_paths & te_paths), "train/test share a file"
    assert not (va_paths & te_paths), "val/test share a file"
    assert len(tr_paths | va_paths | te_paths) == 4

    # The train loader is shuffled with the grouped sampler; val/test sequential.
    assert isinstance(tr.sampler, FileGroupedSampler)
    assert not isinstance(va.sampler, FileGroupedSampler)

    # Same seed -> identical assignment.
    tr2, va2, te2 = create_hw_dataloaders(
        root, config=_config(), batch_size=2, train_split=0.5, val_split=0.25, seed=11
    )
    assert _loader_paths(tr2) == tr_paths
    assert _loader_paths(va2) == va_paths
    assert _loader_paths(te2) == te_paths


def test_create_dataloaders_rejects_impossible_splits(tmp_path: Path) -> None:
    """Overlapping fractions, or too few files for three splits, raise ``ValueError``."""
    root = _four_file_corpus(tmp_path)
    with pytest.raises(ValueError, match="leave room"):
        create_hw_dataloaders(root, config=_config(), train_split=0.9, val_split=0.2)
    with pytest.raises(ValueError, match="at least one file"):
        # 4 files, 0.5 train + 0.25 val leaves test=1, but 0.9+0.05 rounding
        # starves the val split; assert the guard fires for some impossible combo.
        create_hw_dataloaders(root, config=_config(), train_split=0.5, val_split=0.0001)
