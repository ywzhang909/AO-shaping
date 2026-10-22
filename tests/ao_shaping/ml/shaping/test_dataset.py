"""Unit tests for :mod:`ml.shaping.dataset` (collate / loaders / input assembly).

Every fixture is a tiny in-memory sample dict shaped like
:class:`~ml.hwdataset.dataset.HwPhaseImageDataset.__getitem__` output, so this
suite never touches the real ``data/debug`` corpus, never opens a ``.pkl`` and
never needs hardware. ``create_hw_dataloaders`` is monkeypatched with a
recording stub so :func:`build_shaping_loaders` is exercised without a corpus.

``torch`` is a hard dependency of the module under test, so it is imported at
module scope and the file is skipped wholesale when it is missing.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, TensorDataset

import ml.shaping.dataset as shaping_module
from ml.hwdataset.dataset import FileGroupedSampler
from ml.hwdataset.index import HwRecordRef, PhaseSource
from ml.shaping.config import UNKNOWN_EXPOSURE, UNKNOWN_FOV
from ml.shaping.dataset import (
    build_shaping_loaders,
    network_input,
    phase_target,
    shaping_collate,
)

_GRID = 4


def _sample(
    *,
    fov_px: int | None = 320,
    exposure_ms: float | None = 1.5,
    exposure_log10: float = 0.0,
    grid: int = _GRID,
) -> dict[str, Any]:
    """Return a minimal in-memory sample dict matching the dataset contract."""
    rng = np.random.default_rng(0)
    return {
        "phase_cos": rng.standard_normal((1, grid, grid)).astype(np.float32),
        "phase_sin": rng.standard_normal((1, grid, grid)).astype(np.float32),
        "image": rng.standard_normal((1, grid, grid)).astype(np.float32),
        "exposure_log10": np.asarray([exposure_log10], dtype=np.float32),
        "contrast": rng.standard_normal((grid, grid)).astype(np.float32),
        "source": "panel_rad",
        "family": "slm_zernike_shaping",
        "sample_idx": 0,
        "path": "data/debug/fake.pkl",
        "fov_px": fov_px,
        "exposure_ms": exposure_ms,
    }


def test_sentinels_match_config_values() -> None:
    """The collate sentinels are the config's, by value and by type."""
    assert UNKNOWN_FOV == -1
    assert UNKNOWN_EXPOSURE == -1.0
    assert isinstance(UNKNOWN_FOV, int)
    assert isinstance(UNKNOWN_EXPOSURE, float)


def test_collate_mixed_none_fields() -> None:
    """A batch mixing None and real fov_px/exposure_ms collates with sentinels."""
    batch = shaping_collate(
        [_sample(fov_px=None, exposure_ms=None), _sample(fov_px=320, exposure_ms=1.5)]
    )
    assert batch["fov_px"].tolist() == [UNKNOWN_FOV, 320]
    assert batch["exposure_ms"].tolist() == [UNKNOWN_EXPOSURE, 1.5]


def test_collate_stacks_tensors() -> None:
    """Tensor fields are stacked by default_collate with the expected shapes."""
    batch = shaping_collate([_sample(), _sample()])
    assert batch["phase_cos"].shape == (2, 1, _GRID, _GRID)
    assert batch["phase_sin"].shape == (2, 1, _GRID, _GRID)
    assert batch["image"].shape == (2, 1, _GRID, _GRID)
    assert batch["exposure_log10"].shape == (2, 1)
    assert batch["contrast"].shape == (2, _GRID, _GRID)
    assert batch["phase_cos"].dtype == torch.float32


def test_collate_string_fields_become_lists() -> None:
    """String fields survive collation as lists of str."""
    batch = shaping_collate([_sample(), _sample()])
    assert batch["source"] == ["panel_rad", "panel_rad"]
    assert batch["family"] == ["slm_zernike_shaping", "slm_zernike_shaping"]
    assert batch["path"] == ["data/debug/fake.pkl", "data/debug/fake.pkl"]


def test_network_input_two_channels() -> None:
    """in_channels=2 concatenates the broadcast exposure channel."""
    batch = shaping_collate(
        [_sample(exposure_log10=0.5), _sample(exposure_log10=-1.0)]
    )
    out = network_input(batch, _GRID, in_channels=2)
    assert out.shape == (2, 2, _GRID, _GRID)
    assert torch.equal(out[:, 0:1], batch["image"])
    expected = batch["exposure_log10"].view(-1, 1, 1, 1).expand(-1, 1, _GRID, _GRID)
    assert torch.equal(out[:, 1:2], expected)


def test_network_input_single_channel() -> None:
    """in_channels=1 returns the image alone."""
    batch = shaping_collate([_sample()])
    out = network_input(batch, _GRID, in_channels=1)
    assert out.shape == (1, 1, _GRID, _GRID)
    assert torch.equal(out, batch["image"])


def test_phase_target_cats_phasor_channels() -> None:
    """phase_target is exactly the channel-wise cat of cos and sin."""
    batch = shaping_collate([_sample(), _sample()])
    out = phase_target(batch)
    assert out.shape == (2, 2, _GRID, _GRID)
    assert torch.equal(out[:, 0:1], batch["phase_cos"])
    assert torch.equal(out[:, 1:2], batch["phase_sin"])


def test_build_shaping_loaders_preserves_train_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rebuilt train loader keeps the exact FileGroupedSampler instance."""
    records = [
        HwRecordRef(
            path=Path("a.pkl"),
            position=0,
            key=0,
            family="slm_zernike_shaping",
            source=PhaseSource.PANEL_RAD,
            n_terms=0,
            n_max=None,
            freeform_grid=None,
            exposure_ms=1.5,
            sidecar={},
        ),
        HwRecordRef(
            path=Path("b.pkl"),
            position=0,
            key=0,
            family="slm_zernike_shaping",
            source=PhaseSource.PANEL_RAD,
            n_terms=0,
            n_max=None,
            freeform_grid=None,
            exposure_ms=1.5,
            sidecar={},
        ),
    ]
    sampler = FileGroupedSampler(records, seed=42)
    dataset = TensorDataset(torch.zeros(2, 1, _GRID, _GRID))
    train = DataLoader(dataset, batch_size=2, sampler=sampler, shuffle=False)
    val = DataLoader(dataset, batch_size=2, shuffle=False)
    test = DataLoader(dataset, batch_size=2, shuffle=False)

    def _stub(*args: Any, **kwargs: Any) -> tuple[DataLoader, DataLoader, DataLoader]:
        return train, val, test

    monkeypatch.setattr("ml.shaping.dataset.create_hw_dataloaders", _stub)
    new_train, new_val, new_test = build_shaping_loaders("data/debug")
    assert new_train.sampler is sampler
    assert new_train.dataset is dataset
    assert new_train.collate_fn is shaping_collate
    assert new_val.collate_fn is shaping_collate
    assert new_test.collate_fn is shaping_collate


def test_build_shaping_loaders_forwards_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every loader setting is forwarded to create_hw_dataloaders unchanged."""
    captured: dict[str, Any] = {}

    def _stub(*args: Any, **kwargs: Any) -> tuple[DataLoader, DataLoader, DataLoader]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        dataset = TensorDataset(torch.zeros(2, 1, _GRID, _GRID))
        return (
            DataLoader(dataset, batch_size=2, shuffle=False),
            DataLoader(dataset, batch_size=2, shuffle=False),
            DataLoader(dataset, batch_size=2, shuffle=False),
        )

    monkeypatch.setattr("ml.shaping.dataset.create_hw_dataloaders", _stub)
    build_shaping_loaders(
        "data/debug",
        batch_size=8,
        train_split=0.7,
        val_split=0.15,
        num_workers=2,
        seed=7,
        cache_size=3,
        limit_files=5,
        index_cache="data/cache.json",
        require_exposure=False,
        use_cache=False,
        cache_root="data/.hw_cache",
    )
    assert captured["args"] == ("data/debug",)
    assert captured["kwargs"]["batch_size"] == 8
    assert captured["kwargs"]["train_split"] == 0.7
    assert captured["kwargs"]["val_split"] == 0.15
    assert captured["kwargs"]["num_workers"] == 2
    assert captured["kwargs"]["seed"] == 7
    assert captured["kwargs"]["cache_size"] == 3
    assert captured["kwargs"]["limit_files"] == 5
    assert captured["kwargs"]["index_cache"] == "data/cache.json"
    assert captured["kwargs"]["require_exposure"] is False
    assert captured["kwargs"]["use_cache"] is False
    assert captured["kwargs"]["cache_root"] == "data/.hw_cache"


def test_module_never_recovers_an_angle() -> None:
    """The module must never call arctan2 / torch.angle on the phasor pair."""
    source = inspect.getsource(shaping_module)
    assert "arctan2" not in source
    assert "torch.angle" not in source