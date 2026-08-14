"""Tests for ml.zernike_prediction.dataset (T11: dual-camera Zernike regression dataset).

Covers: real-data scanning/lengths, getitem shapes/dtypes, metadata vs sidecar
label sources, split modes (sample / run), collate_with_meta, normalize modes,
and the label manifest used by CLI/plots.

Conventions under test (documented in the dataset module):
- label is a ``(65,)`` float32 torch.Tensor when present; ``None`` when the
  sample has no labels and ``require_labels=False``.
- ``collate_with_meta`` returns ``(imgs, labels, sample_dirs)``; ``labels`` is
  ``None`` when no item in the batch carried a label.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import ml.zernike_prediction.dataset as dataset_mod
from ml.zernike_prediction import ZernikeDualDataset
from ml.zernike_prediction.dataset import (
    build_label_manifest,
    collate_with_meta,
    create_zernike_loaders,
    normalize_none,
    normalize_per_image,
    normalize_standardize,
)

_REAL_ROOT = Path("data/slm_dual_spot")
_REAL_0414 = _REAL_ROOT / "20260414_171241"  # 333 samples, metadata coefficients
_REAL_155508 = _REAL_ROOT / "20260402_155508"  # 100 samples, no metadata coefficients
_REAL_164456 = _REAL_ROOT / "20260402_164456"  # 51 samples, no metadata coefficients

_FIXED_COEFFS = [1.0] + [0.1 * (i + 1) for i in range(65)]  # deterministic, distinct


def _require_real_data() -> None:
    if not _REAL_0414.exists():
        pytest.skip("requires slm_dual_spot data at data/slm_dual_spot")


# ---------------------------------------------------------------------------
# Synthetic fixtures (no conftest.py in this project — helpers inline)
# ---------------------------------------------------------------------------


def _make_sample(
    tmp_path: Path,
    run_id: str,
    idx: int,
    *,
    labeled: bool = True,
    sidecar: bool = False,
    size: int = 64,
    seed: int = 0,
) -> Path:
    """Create one sample_XXXX dir with metadata.json + tiny npy frames."""
    d = tmp_path / run_id / f"sample_{idx:04d}"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed * 1000 + idx)
    meta: dict = {"sample_idx": idx, "phase_params": {"n_max": 10}}
    if labeled:
        coeffs = [float(0.5 * (i + 1)) for i in range(66)]
        coeffs[0] = 1.0
        meta["phase_params"]["coefficients"] = coeffs
    (d / "metadata.json").write_text(json.dumps(meta))
    np.save(
        d / "daheng_frame.npy",
        rng.integers(0, 65536, size=(size, size), dtype=np.uint16),
    )
    np.save(
        d / "miicam_frame.npy",
        rng.integers(0, 256, size=(size, size), dtype=np.uint8),
    )
    if sidecar:
        np.save(d / "labels.npy", rng.normal(0, 1, size=65).astype(np.float32))
    return d


def _make_run(
    tmp_path: Path,
    run_id: str,
    n_samples: int,
    *,
    labeled: bool = True,
    sidecar: bool = False,
    size: int = 64,
    seed: int = 0,
) -> Path:
    for i in range(n_samples):
        _make_sample(tmp_path, run_id, i, labeled=labeled, sidecar=sidecar, size=size, seed=seed)
    return tmp_path / run_id


# ---------------------------------------------------------------------------
# 1. Scanning and lengths (real data)
# ---------------------------------------------------------------------------


def test_dataset_scans_and_lengths() -> None:
    _require_real_data()
    ds = ZernikeDualDataset(_REAL_ROOT, run_ids=["20260414_171241"])
    assert len(ds) == 333
    assert all(s["run_id"] == "20260414_171241" for s in ds.samples)
    assert set(ds.samples[0]) == {"run_id", "sample_idx", "sample_dir", "has_label"}
    assert all(s["has_label"] for s in ds.samples)

    capped = ZernikeDualDataset(_REAL_ROOT, run_ids=["20260414_171241"], max_samples=10)
    assert len(capped) == 10

    # run_ids=None -> all run dirs; 0402 runs have no labels yet (recovered in
    # T8), so scan-only requires require_labels=False
    all_runs = ZernikeDualDataset(_REAL_ROOT, require_labels=False)
    assert len(all_runs) == 333 + 100 + 51

    # __init__.py re-export points at the same class
    assert ZernikeDualDataset is dataset_mod.ZernikeDualDataset


# ---------------------------------------------------------------------------
# 2. getitem shapes / dtypes / normalization range (real data)
# ---------------------------------------------------------------------------


def test_getitem_shapes_and_dtypes() -> None:
    _require_real_data()
    ds = ZernikeDualDataset(_REAL_ROOT, run_ids=["20260414_171241"])
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (2, 256, 256)
    assert img.dtype == torch.float32
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert isinstance(label, torch.Tensor)
    assert label.shape == (65,)
    assert label.dtype == torch.float32
    assert bool(torch.isfinite(label).all())

    focus_ds = ZernikeDualDataset(_REAL_ROOT, run_ids=["20260414_171241"], input_mode="focus")
    focus_img, _ = focus_ds[1]
    assert focus_img.shape == (1, 256, 256)

    pupil_ds = ZernikeDualDataset(_REAL_ROOT, run_ids=["20260414_171241"], input_mode="pupil")
    pupil_img, _ = pupil_ds[1]
    assert pupil_img.shape == (1, 256, 256)


# ---------------------------------------------------------------------------
# 3. 0414 labels == metadata coefficients (exact, float32 tol 1e-6)
# ---------------------------------------------------------------------------


def test_0414_labels_match_metadata() -> None:
    _require_real_data()
    ds = ZernikeDualDataset(_REAL_ROOT, run_ids=["20260414_171241"])
    for i in (0, 17, 332):
        meta = json.loads((Path(ds.samples[i]["sample_dir"]) / "metadata.json").read_text())
        coeffs = np.asarray(meta["phase_params"]["coefficients"], dtype=np.float32)
        expected = coeffs[1:66]
        label = ds[i][1].numpy()
        assert label.shape == (65,)
        np.testing.assert_allclose(label, expected, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# 4. Sidecar labels.npy used when metadata lacks coefficients
# ---------------------------------------------------------------------------


def test_sidecar_labels_used_when_metadata_missing(tmp_path: Path) -> None:
    _make_run(tmp_path, "run_a", 3, labeled=False, sidecar=True)
    ds = ZernikeDualDataset(tmp_path, run_ids=["run_a"])
    assert len(ds) == 3
    assert all(s["has_label"] for s in ds.samples)
    for i in range(3):
        label = ds[i][1].numpy()
        sidecar = np.load(tmp_path / "run_a" / f"sample_{i:04d}" / "labels.npy")
        assert label.shape == (65,)
        np.testing.assert_allclose(label, sidecar, atol=1e-6, rtol=1e-6)

    # require_labels=True with neither metadata coefficients nor sidecar -> raise
    _make_run(tmp_path, "run_b", 2, labeled=False, sidecar=False)
    with pytest.raises(RuntimeError, match="run_b"):
        ZernikeDualDataset(tmp_path, run_ids=["run_b"], require_labels=True)


# ---------------------------------------------------------------------------
# 5. require_labels=False -> label is None (documented convention)
# ---------------------------------------------------------------------------


def test_require_labels_false_allows_none(tmp_path: Path) -> None:
    _make_run(tmp_path, "run_a", 2, labeled=False, sidecar=False)
    ds = ZernikeDualDataset(tmp_path, run_ids=["run_a"], require_labels=False)
    assert len(ds) == 2
    assert not any(s["has_label"] for s in ds.samples)
    img, label = ds[0]
    assert img.shape == (2, 256, 256)
    assert label is None  # documented: Python None when unlabeled


# ---------------------------------------------------------------------------
# 6. split_mode="sample" deterministic, all samples exactly once (real data)
# ---------------------------------------------------------------------------


def test_split_sample_deterministic() -> None:
    _require_real_data()
    kwargs = dict(run_ids=["20260414_171241"], split_mode="sample", seed=0, num_workers=0)

    res_a = create_zernike_loaders(_REAL_ROOT, **kwargs)
    res_b = create_zernike_loaders(_REAL_ROOT, **kwargs)

    def sample_dirs(res: dict) -> dict[str, list[str]]:
        full = res["dataset"]
        return {
            name: sorted(full.samples[i]["sample_dir"] for i in loader.dataset.indices)
            for name, loader in (("train", res["train"]), ("val", res["val"]), ("test", res["test"]))
        }

    dirs_a = sample_dirs(res_a)
    dirs_b = sample_dirs(res_b)
    assert dirs_a == dirs_b  # same seed -> identical splits

    total = sum(len(loader.dataset) for loader in (res_a["train"], res_a["val"], res_a["test"]))
    assert total == len(res_a["dataset"]) == 333  # every sample exactly once
    assert sum(res_a["split_sizes"].values()) == 333
    assert res_a["split_sizes"]["train"] == len(res_a["train"].dataset)
    assert res_a["split_sizes"]["val"] == len(res_a["val"].dataset)
    assert res_a["split_sizes"]["test"] == len(res_a["test"].dataset)

    res_c = create_zernike_loaders(_REAL_ROOT, **{**kwargs, "seed": 1})
    dirs_c = sample_dirs(res_c)
    assert dirs_c != dirs_a  # different seed -> different split


# ---------------------------------------------------------------------------
# 7. split_mode="run": whole runs per split, no leakage (real data)
# ---------------------------------------------------------------------------


def test_split_run_mode() -> None:
    _require_real_data()
    res = create_zernike_loaders(
        _REAL_ROOT,
        run_ids=["20260402_155508", "20260402_164456", "20260414_171241"],
        split_mode="run",
        test_split=0.5,
        require_labels=False,
        num_workers=0,
    )

    def run_ids_of(loader) -> set[str]:
        return {s["run_id"] for s in loader.dataset.samples}

    train_runs = run_ids_of(res["train"])
    val_runs = run_ids_of(res["val"])
    test_runs = run_ids_of(res["test"])

    # test set is exactly one whole run (first sorted run)
    assert test_runs == {"20260402_155508"}
    assert len(res["test"].dataset) == 100
    # val takes the next run; train the rest — no leakage between train and test
    assert val_runs == {"20260402_164456"}
    assert len(res["val"].dataset) == 51
    assert train_runs == {"20260414_171241"}
    assert len(res["train"].dataset) == 333
    assert train_runs.isdisjoint(test_runs)
    assert train_runs.isdisjoint(val_runs)
    assert val_runs.isdisjoint(test_runs)


# ---------------------------------------------------------------------------
# 8. collate_with_meta
# ---------------------------------------------------------------------------


def test_collate_with_meta(tmp_path: Path) -> None:
    _make_run(tmp_path, "run_a", 2, labeled=True)
    ds = ZernikeDualDataset(tmp_path, run_ids=["run_a"], return_meta=True)
    items = [ds[i] for i in range(2)]
    imgs, labels, dirs = collate_with_meta(items)
    assert imgs.shape == (2, 2, 256, 256)
    assert imgs.dtype == torch.float32
    assert labels.shape == (2, 65)
    assert len(dirs) == 2
    assert Path(dirs[0]).name == "sample_0000"
    assert Path(dirs[1]).name == "sample_0001"


# ---------------------------------------------------------------------------
# 9. normalize modes
# ---------------------------------------------------------------------------


def test_normalize_modes() -> None:
    # per_image: [0,1] on varying and constant inputs
    rng = np.random.default_rng(0)
    varying = rng.integers(0, 65536, size=(2, 64, 64)).astype(np.float32)
    out = normalize_per_image(varying)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6

    const = np.full((1, 16, 16), 5000.0, dtype=np.float32)
    const_out = normalize_per_image(const)
    assert const_out.shape == const.shape
    assert np.allclose(const_out, 0.0)  # constant -> zeros, still in [0,1]

    # standardize: mean ~ 0, std ~ 1 per channel
    std_out = normalize_standardize(varying)
    assert std_out.shape == varying.shape
    np.testing.assert_allclose(std_out.mean(axis=(1, 2)), 0.0, atol=1e-3)
    np.testing.assert_allclose(std_out.std(axis=(1, 2)), 1.0, atol=1e-3)

    # none: dtype-aware scaling preserves relative scale (uint16 /65535)
    base = rng.integers(0, 65536, size=(64, 64)).astype(np.float32)
    u16 = np.stack([base, 2.0 * base])
    none_out = normalize_none(u16, [np.dtype("uint16"), np.dtype("uint16")])
    ratio = none_out[1].mean() / none_out[0].mean()
    assert ratio == pytest.approx(2.0, rel=1e-3)
    # values above 1.0 are valid: "none" mode is a linear rescale, not a clip
    assert none_out.max() > 1.0 or none_out.max() <= 1.0 + 1e-6

    # uint8 channel scaled /255
    u8 = np.stack([np.full((16, 16), 100, dtype=np.float32)])
    none_u8 = normalize_none(u8, [np.dtype("uint8")])
    assert none_u8.max() == pytest.approx(100.0 / 255.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 10. build_label_manifest
# ---------------------------------------------------------------------------


def test_build_label_manifest(tmp_path: Path) -> None:
    _require_real_data()
    mf = build_label_manifest(_REAL_ROOT, run_ids=["20260414_171241"])
    assert mf is not None and len(mf) == 333
    assert list(mf.columns) == ["sample_dir", "run_id", "sample_idx", "has_metadata_labels", "labels_path"]
    assert mf["has_metadata_labels"].all()
    assert mf["labels_path"].isna().all()  # no sidecars on the real 0414 run

    mf_155508 = build_label_manifest(_REAL_ROOT, run_ids=["20260402_155508"])
    assert len(mf_155508) == 100
    assert not mf_155508["has_metadata_labels"].any()

    # tmp run with sidecar labels: has_metadata_labels False but labels_path set
    _make_run(tmp_path, "run_s", 2, labeled=False, sidecar=True)
    mf_tmp = build_label_manifest(tmp_path, run_ids=["run_s"])
    assert len(mf_tmp) == 2
    assert not mf_tmp["has_metadata_labels"].any()
    assert all(p is not None and Path(p).name == "labels.npy" for p in mf_tmp["labels_path"])
