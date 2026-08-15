"""Dual-camera PyTorch Dataset for Zernike-coefficient regression (T11).

Loads paired ``daheng_frame.npy`` (focus camera, uint16) + ``miicam_frame.npy``
(pupil camera, uint8) captures from ``data/slm_dual_spot/<run>/sample_XXXX/``
and returns image tensors + 65 non-piston Zernike coefficient targets.

Label sources (resolved at construction, metadata first):
    1. ``metadata.json`` -> ``phase_params.coefficients`` (66 floats in
       ``(n, m)`` order, index 0 = piston 1.0); the 65 non-piston entries are
       kept.
    2. Sidecar ``<sample_dir>/labels.npy`` (65,) float32 (used for the 0402
       runs whose metadata lacks coefficients).
    3. If neither exists and ``require_labels=True``, construction raises with
       the missing sample dirs listed; with ``require_labels=False`` the label
       is ``None`` (used by the predict CLI on unlabeled data).

Conventions:
    - ``__getitem__`` returns ``(img, label)`` where ``label`` is a ``(65,)``
      float32 ``torch.Tensor`` when present and Python ``None`` when the sample
      is unlabeled; with ``return_meta=True`` it returns
      ``(img, label, sample_dir)``.
    - ``collate_with_meta`` stacks images, keeps the list of sample dirs, and
      returns ``labels`` as ``None`` when no item in the batch carried a label
      (a mixed batch yields a list of tensors/None).

Resize backend: ``cv2`` (INTER_AREA) is preferred for speed; this venv does not
have cv2 installed, so the documented fallback ``torchvision.transforms.functional.resize``
(BILINEAR + ``antialias=True``) is used. Both paths downscale to float32.

Normalization modes (operate per channel, float32 before any arithmetic to
avoid uint16 overflow):
    - "per_image": ``(x - x.min()) / (x.max() - x.min() + eps)`` -> [0, 1].
    - "none": raw values scaled by dtype max (uint16 -> /65535, uint8 -> /255).
    - "standardize": ``(x - mean) / (std + eps)`` per channel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Subset

from ml.zernike_prediction.phase_gen import count_zernike_terms, non_piston_indices

try:  # preferred resize backend; falls back to torchvision when unavailable
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover - depends on env
    cv2 = None
    _HAS_CV2 = False

try:
    import torchvision.transforms.functional as _TF

    _HAS_TORCHVISION = True
except ImportError:  # pragma: no cover - depends on env
    _TF = None
    _HAS_TORCHVISION = False

_N_MAX = 10  # radial degree used by the capture pipeline
_N_TERMS = count_zernike_terms(_N_MAX)  # 66 incl. piston
_N_TARGET = len(non_piston_indices(_N_MAX))  # 65 non-piston regression targets
_NON_PISTON = non_piston_indices(_N_MAX)  # [1, 2, ..., 65]
_EPS = 1e-6

_INPUT_MODES = ("combined", "focus", "pupil", "fft", "fft_ratio")
_NORMALIZE_MODES = ("per_image", "none", "standardize")

_FFT_EPS = 1e-6

__all__ = [
    "ZernikeDualDataset",
    "build_label_manifest",
    "collate_with_meta",
    "create_zernike_loaders",
    "normalize_frames",
    "normalize_none",
    "normalize_per_image",
    "normalize_standardize",
]


# ---------------------------------------------------------------------------
# Normalization (per-channel, float32 first)
# ---------------------------------------------------------------------------


def normalize_per_image(frames: np.ndarray) -> np.ndarray:
    """Min-max scale each channel to [0, 1] (constant channels -> 0).

    Args:
        frames: ``(C, H, W)`` array, any numeric dtype.

    Returns:
        ``(C, H, W)`` float32 in ``[0, 1]``.
    """
    frames = np.asarray(frames, dtype=np.float32)
    mn = frames.min(axis=(1, 2), keepdims=True)
    mx = frames.max(axis=(1, 2), keepdims=True)
    return (frames - mn) / (mx - mn + _EPS)


def normalize_none(frames: np.ndarray, dtypes: Sequence[np.dtype]) -> np.ndarray:
    """Scale raw values by their original dtype max (uint16 -> /65535, uint8 -> /255).

    Preserves the relative scale between channels/captures.

    Args:
        frames: ``(C, H, W)`` array.
        dtypes: Original dtype per channel, length ``C``.

    Returns:
        ``(C, H, W)`` float32.
    """
    frames = np.asarray(frames, dtype=np.float32)
    scales = np.asarray([_dtype_scale(dt) for dt in dtypes], dtype=np.float32).reshape(-1, 1, 1)
    return frames / scales


def normalize_standardize(frames: np.ndarray) -> np.ndarray:
    """Per-channel ``(x - mean) / (std + eps)`` using per-image statistics.

    Args:
        frames: ``(C, H, W)`` array, any numeric dtype.

    Returns:
        ``(C, H, W)`` float32 with ~zero mean / unit std per channel.
    """
    frames = np.asarray(frames, dtype=np.float32)
    mean = frames.mean(axis=(1, 2), keepdims=True)
    std = frames.std(axis=(1, 2), keepdims=True)
    return (frames - mean) / (std + _EPS)


def normalize_frames(frames: np.ndarray, mode: str, dtypes: Sequence[np.dtype]) -> np.ndarray:
    """Dispatch to the requested normalization mode."""
    if mode == "per_image":
        return normalize_per_image(frames)
    if mode == "none":
        return normalize_none(frames, dtypes)
    if mode == "standardize":
        return normalize_standardize(frames)
    raise ValueError(f"unknown normalize mode {mode!r}; expected one of {_NORMALIZE_MODES}")


def _dtype_scale(dtype: np.dtype) -> float:
    """Numeric max used for dtype-aware raw scaling (falls back to 1.0)."""
    if np.issubdtype(dtype, np.unsignedinteger) and dtype.itemsize >= 2:
        return float(np.iinfo(dtype).max)  # uint16 -> 65535
    if np.issubdtype(dtype, np.unsignedinteger) and dtype.itemsize == 1:
        return float(np.iinfo(dtype).max)  # uint8 -> 255
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


# ---------------------------------------------------------------------------
# Frame loading / resizing
# ---------------------------------------------------------------------------


def _resize_frames(frames: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Downscale ``(C, H, W)`` float32 frames to ``target_size`` (H, W).

    cv2 INTER_AREA is preferred; the torchvision BILINEAR + antialias fallback
    is used when cv2 is not installed (the case in this venv).
    """
    th, tw = int(target_size[0]), int(target_size[1])
    if _HAS_CV2:
        out = np.empty((frames.shape[0], th, tw), dtype=np.float32)
        for c in range(frames.shape[0]):
            out[c] = cv2.resize(frames[c], (tw, th), interpolation=cv2.INTER_AREA)
        return out
    if _HAS_TORCHVISION:
        # torchvision.resize expects (N, C, d1, d2) — add a batch dim and squeeze
        t = torch.from_numpy(frames).unsqueeze(0)
        t = _TF.resize(
            t,
            (th, tw),
            interpolation=_TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        return t.squeeze(0).numpy()
    raise RuntimeError("frame resizing requires cv2 (preferred) or torchvision (fallback); neither is installed")


def _load_frames(sample_dir: str | Path, input_mode: str) -> tuple[np.ndarray, list[np.dtype]]:
    """Load ``(C, H, W)`` float32 frames and the original per-channel dtypes.

    Note: cameras have different native resolutions (daheng 1024x1280 uint16,
    miicam 1520x2688 uint8), so frames are returned UNRESIZED — stacking must
    happen after resizing to a common ``target_size`` (see ``_load_image``).
    """
    sample_dir = Path(sample_dir)
    daheng = np.load(sample_dir / "daheng_frame.npy")  # focus camera, uint16
    miicam = np.load(sample_dir / "miicam_frame.npy")  # pupil camera, uint8
    if input_mode == "combined":
        # Return as a list of frames with per-channel dtypes; caller stacks
        # after resize (native shapes differ: 1024x1280 vs 1520x2688).
        return [daheng.astype(np.float32, copy=False), miicam.astype(np.float32, copy=False)], [
            daheng.dtype,
            miicam.dtype,
        ]
    if input_mode in ("fft", "fft_ratio"):
        # Both spectra need both cameras; caller applies the FFT after resizing.
        return [daheng.astype(np.float32, copy=False), miicam.astype(np.float32, copy=False)], [
            daheng.dtype,
            miicam.dtype,
        ]
    if input_mode == "focus":
        return [daheng.astype(np.float32, copy=False)], [daheng.dtype]
    # pupil
    return [miicam.astype(np.float32, copy=False)], [miicam.dtype]


def _fft_log_magnitude(frame: np.ndarray) -> np.ndarray:
    """Log-scaled centered FFT magnitude of a 2D frame.

    ``log(1 + |fftshift(fft2(x))|)`` — preserves dynamic range of the power
    spectrum while keeping values finite. Input is a single (H, W) float32 frame.
    """
    spectrum = np.fft.fftshift(np.fft.fft2(frame))
    return np.log1p(np.abs(spectrum))


def _fft_ratio(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Log-scaled magnitude ratio of the two cameras' spectra.

    ``log(1 + |FFT(daheng)| / (|FFT(miicam)| + eps))`` — a single channel whose
    local structure captures the relative spectral content between the focus
    (daheng) and pupil (miicam) planes.
    """
    daheng_fft = np.abs(np.fft.fftshift(np.fft.fft2(frames[0])))
    miicam_fft = np.abs(np.fft.fftshift(np.fft.fft2(frames[1])))
    return np.log1p(daheng_fft / (miicam_fft + _FFT_EPS))


# ---------------------------------------------------------------------------
# Data-root scanning / label resolution
# ---------------------------------------------------------------------------


def _scan_samples(data_root: str | Path, run_ids: list[str] | None) -> list[dict[str, Any]]:
    """List run/sample dirs in deterministic (sorted) order.

    Returns a list of dicts with keys ``run_id``, ``sample_idx`` (position in
    the sorted ``sample_*`` scan), ``sample_dir``, ``metadata_path`` (or None),
    ``labels_path`` (or None).
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data_root not found: {root}")
    run_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if run_ids is not None:
        wanted = set(run_ids)
        run_dirs = [p for p in run_dirs if p.name in wanted]
        missing = sorted(wanted - {p.name for p in run_dirs})
        if missing:
            raise FileNotFoundError(f"run dir(s) not found under {root}: {missing}")
    entries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        sample_dirs = sorted(p for p in run_dir.glob("sample_*") if p.is_dir())
        for i, sd in enumerate(sample_dirs):
            meta_path = sd / "metadata.json"
            labels_path = sd / "labels.npy"
            entries.append(
                {
                    "run_id": run_dir.name,
                    "sample_idx": i,
                    "sample_dir": str(sd),
                    "metadata_path": str(meta_path) if meta_path.exists() else None,
                    "labels_path": str(labels_path) if labels_path.exists() else None,
                }
            )
    return entries


def _resolve_label(entry: dict[str, Any]) -> np.ndarray | None:
    """``(65,)`` float32 label from metadata coefficients, else the labels.npy sidecar."""
    label: np.ndarray | None = None
    if entry["metadata_path"] is not None:
        try:
            meta = json.loads(Path(entry["metadata_path"]).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"unreadable metadata.json {entry['metadata_path']}: {exc}")
            meta = {}
        phase_params = meta.get("phase_params") or {}
        coeffs = phase_params.get("coefficients")
        if coeffs is not None:
            n_max = int(phase_params.get("n_max", _N_MAX))
            arr = np.asarray(coeffs, dtype=np.float32)
            if arr.ndim == 1 and len(arr) >= len(non_piston_indices(n_max)):
                label = arr[non_piston_indices(n_max)].astype(np.float32)
                if label.shape != (_N_TARGET,):
                    logger.warning(
                        f"{entry['sample_dir']}: metadata n_max={n_max} yields {len(label)} "
                        f"non-piston terms, expected {_N_TARGET}; trying sidecar"
                    )
                    label = None
    if label is None and entry["labels_path"] is not None:
        try:
            label = np.load(entry["labels_path"]).astype(np.float32)
            if label.shape != (_N_TARGET,):
                raise ValueError(f"expected ({_N_TARGET},), got {label.shape}")
        except (OSError, ValueError) as exc:
            logger.warning(f"invalid sidecar labels {entry['labels_path']}: {exc}; ignoring")
            label = None
    return label


def _has_metadata_labels(metadata_path: str | None) -> bool:
    """True when metadata.json carries a non-empty ``phase_params.coefficients`` list."""
    if metadata_path is None:
        return False
    try:
        meta = json.loads(Path(metadata_path).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    coeffs = (meta.get("phase_params") or {}).get("coefficients")
    return isinstance(coeffs, (list, tuple, np.ndarray)) and len(coeffs) > 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ZernikeDualDataset(Dataset):
    """Dual-camera (focus + pupil) dataset with Zernike regression targets.

    Args:
        data_root: Directory containing ``<run>/sample_XXXX/`` subdirectories.
        run_ids: Restrict to these run dirs; ``None`` scans all run dirs.
        transform: Optional callable applied to the normalized image tensor.
        target_size: Resized ``(H, W)`` (default 256x256).
        input_mode: "combined" (2-channel), "focus" or "pupil" (1-channel).
        normalize: "per_image", "none", or "standardize".
        max_samples: Cap the total number of samples (first N in scan order).
        require_labels: Raise at construction listing samples without labels.
        seed: Stored for reproducibility bookkeeping (used by the loader helper
            for deterministic splitting); the scan order is always sorted.
        return_meta: Also return the sample dir from ``__getitem__``.

    Attributes:
        samples: ``[{run_id, sample_idx, sample_dir, has_label}, ...]``.
    """

    def __init__(
        self,
        data_root: str | Path,
        run_ids: list[str] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_size: tuple[int, int] = (256, 256),
        input_mode: str = "combined",
        normalize: str = "per_image",
        max_samples: int | None = None,
        require_labels: bool = True,
        seed: int = 0,
        return_meta: bool = False,
    ) -> None:
        if input_mode not in _INPUT_MODES:
            raise ValueError(f"input_mode must be one of {_INPUT_MODES}, got {input_mode!r}")
        if normalize not in _NORMALIZE_MODES:
            raise ValueError(f"normalize must be one of {_NORMALIZE_MODES}, got {normalize!r}")
        if max_samples is not None and max_samples < 0:
            raise ValueError(f"max_samples must be >= 0, got {max_samples}")

        self.data_root = Path(data_root)
        self.run_ids = sorted(run_ids) if run_ids is not None else None
        self.transform = transform
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.input_mode = input_mode
        self.normalize = normalize
        self.require_labels = require_labels
        self.seed = seed
        self.return_meta = return_meta

        raw = _scan_samples(self.data_root, run_ids)
        if max_samples is not None:
            raw = raw[:max_samples]

        self.samples: list[dict[str, Any]] = []
        self._labels: list[np.ndarray | None] = []
        missing: list[str] = []
        for entry in raw:
            label = _resolve_label(entry)
            self._labels.append(label)
            self.samples.append(
                {
                    "run_id": entry["run_id"],
                    "sample_idx": entry["sample_idx"],
                    "sample_dir": entry["sample_dir"],
                    "has_label": label is not None,
                }
            )
            if require_labels and label is None:
                missing.append(entry["sample_dir"])
        if missing:
            shown = missing[:5]
            suffix = " ..." if len(missing) > 5 else ""
            raise RuntimeError(
                f"{len(missing)} sample(s) under {self.data_root} have no labels "
                f"(no metadata coefficients, no labels.npy; require_labels=True): {shown}{suffix}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, str]:
        sample_dir = self.samples[idx]["sample_dir"]
        img = self._load_image(sample_dir)
        label = self._labels[idx]
        label_t = torch.from_numpy(label) if label is not None else None
        if self.return_meta:
            return img, label_t, sample_dir
        return img, label_t

    def _load_image(self, sample_dir: str | Path) -> torch.Tensor:
        """Load, resize and normalize the ``(C, H, W)`` float32 image tensor."""
        frames, dtypes = _load_frames(sample_dir, self.input_mode)
        resized = [_resize_frames(f, self.target_size) for f in frames]
        if self.input_mode == "fft":
            frames = np.stack([_fft_log_magnitude(f) for f in resized])
            dtypes = [np.dtype("float32")] * len(resized)
        elif self.input_mode == "fft_ratio":
            frames = _fft_ratio(resized)[np.newaxis]
            dtypes = [np.dtype("float32")]
        else:
            frames = np.stack(resized)
        frames = normalize_frames(frames, self.normalize, dtypes)
        img = torch.from_numpy(frames)
        if self.transform is not None:
            img = self.transform(img)
        return img


# ---------------------------------------------------------------------------
# Collation with metadata
# ---------------------------------------------------------------------------


def collate_with_meta(
    batch: list[tuple[torch.Tensor, torch.Tensor | None, str]],
) -> tuple[torch.Tensor, torch.Tensor | list[torch.Tensor | None] | None, list[str]]:
    """Collate items from ``return_meta=True`` datasets.

    Stacks image tensors to ``(B, C, H, W)``, keeps the list of sample dirs and
    stacks labels: ``labels`` is ``None`` when no item carried a label, a
    ``(B, 65)`` tensor when all did, and a list of tensors/None for mixed batches.
    """
    imgs, labels, sample_dirs = zip(*batch)
    img_stack = torch.stack(list(imgs))
    if all(label is None for label in labels):
        labels_out: torch.Tensor | list[torch.Tensor | None] | None = None
    elif all(label is not None for label in labels):
        labels_out = torch.stack(list(labels))
    else:
        labels_out = list(labels)
    return img_stack, labels_out, list(sample_dirs)


# ---------------------------------------------------------------------------
# Split helpers (deterministic via seed)
# ---------------------------------------------------------------------------


def _split_indices(
    n: int,
    run_ids: Sequence[str],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified-by-run random split of ``range(n)`` into (train, val, test).

    Every index lands in exactly one split. Uses sklearn ``train_test_split``
    when available (fallback: per-stratum shuffle with ``default_rng(seed)``).
    """
    idx = np.arange(n)
    run_ids_arr = np.asarray(run_ids)
    if test_split <= 0:
        test_idx = np.array([], dtype=np.int64)
        train_val_idx = idx
    else:
        train_val_idx, test_idx = _train_test_split(
            idx, test_size=test_split, stratify=run_ids_arr, random_state=seed
        )
    if val_split <= 0:
        val_idx = np.array([], dtype=np.int64)
        train_idx = train_val_idx
    else:
        val_frac = val_split / (1.0 - test_split) if test_split < 1.0 else 0.0
        train_idx, val_idx = _train_test_split(
            train_val_idx,
            test_size=val_frac,
            stratify=run_ids_arr[train_val_idx],
            random_state=seed,
        )
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64), test_idx


def _train_test_split(idx: np.ndarray, test_size: float, stratify: np.ndarray, random_state: int):
    """sklearn ``train_test_split`` with a deterministic per-stratum fallback."""
    try:
        from sklearn.model_selection import train_test_split

        return train_test_split(idx, test_size=test_size, stratify=stratify, random_state=random_state)
    except ImportError:
        pass
    except ValueError:
        logger.warning("sklearn train_test_split failed on small strata; falling back to manual split")
    return _manual_train_test_split(idx, test_size, stratify, random_state)


def _manual_train_test_split(idx: np.ndarray, test_size: float, stratify: np.ndarray, random_state: int):
    """Per-stratum shuffle split; keeps >= 1 train sample per stratum when possible."""
    rng = np.random.default_rng(random_state)
    idx = np.asarray(idx)
    stratify = np.asarray(stratify)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for grp in sorted(np.unique(stratify).tolist()):
        group = idx[stratify == grp].copy()
        rng.shuffle(group)
        n_test = min(int(round(len(group) * test_size)), len(group) - 1) if len(group) > 1 else 0
        test_parts.append(group[:n_test])
        train_parts.append(group[n_test:])
    return np.concatenate(train_parts), np.concatenate(test_parts)


def _split_runs(
    runs: list[str],
    val_split: float,
    test_split: float,
) -> tuple[list[str], list[str], list[str]]:
    """Whole-run split: first runs -> test, next -> val, rest -> train (sorted order).

    Allocates at least one run per requested split and raises an informative
    error when there are not enough runs. With the real 3 runs and
    ``test_split=0.5, val_split=0.15`` this yields test=20260402_155508,
    val=20260402_164456, train=20260414_171241 (no leakage).
    """
    n = len(runs)
    n_test = max(1, int(np.floor(n * test_split))) if test_split > 0 else 0
    n_val = max(1, int(np.floor(n * val_split))) if val_split > 0 else 0
    required = (1 if test_split > 0 else 0) + (1 if val_split > 0 else 0) + 1
    if n < required:
        raise ValueError(
            f"split_mode='run' needs at least {required} run dir(s) for "
            f"val_split={val_split}, test_split={test_split} (train must keep >= 1 run); "
            f"got {n}: {runs}"
        )
    n_test = min(n_test, n - (1 if n_val > 0 else 0) - 1)
    n_val = min(n_val, n - n_test - 1)
    test_runs = runs[:n_test]
    val_runs = runs[n_test : n_test + n_val]
    train_runs = runs[n_test + n_val :]
    return test_runs, val_runs, train_runs


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------


def create_zernike_loaders(
    data_root: str | Path,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 0,
    input_mode: str = "combined",
    target_size: tuple[int, int] = (256, 256),
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    run_ids: list[str] | None = None,
    split_mode: str = "sample",
    max_samples: int | None = None,
    num_workers: int = 2,
    *,
    normalize: str = "per_image",
    require_labels: bool = True,
    return_meta: bool = False,
) -> dict[str, Any]:
    """Build train/val/test DataLoaders over the dual-camera dataset.

    Args:
        data_root: Directory containing run/sample dirs.
        batch_size: Per-loader batch size.
        val_split / test_split: Fractions (default 0.15 / 0.15).
        seed: Determinism seed for splitting and train-shuffle generator.
        input_mode / target_size / transform / run_ids / max_samples /
        num_workers: Passed through to the dataset / DataLoader.
        split_mode: "sample" -> random per-sample split stratified by run
            (sklearn ``train_test_split``); "run" -> whole runs per split
            (test/val take entire run dirs; leakage-free evaluation).
        normalize / require_labels / return_meta: Extra dataset options
            (defaults match ``ZernikeDualDataset``).

    Returns:
        Dict with keys ``train``, ``val``, ``test`` (DataLoaders),
        ``dataset`` (full dataset in sample mode, ``None`` in run mode),
        ``split_sizes`` (per-split sample counts), ``split_mode``, ``seed``
        and ``dataset_kwargs`` (kwargs used to construct the datasets).
    """
    if split_mode not in ("sample", "run"):
        raise ValueError(f"split_mode must be 'sample' or 'run', got {split_mode!r}")
    if not (0.0 <= test_split < 1.0 and 0.0 <= val_split < 1.0 and test_split + val_split < 1.0):
        raise ValueError(
            f"invalid splits: val_split={val_split}, test_split={test_split}; "
            "need both in [0, 1) and test_split + val_split < 1"
        )

    dataset_kwargs: dict[str, Any] = {
        "data_root": str(Path(data_root)),
        "run_ids": run_ids,
        "transform": transform,
        "target_size": target_size,
        "input_mode": input_mode,
        "normalize": normalize,
        "max_samples": max_samples,
        "require_labels": require_labels,
        "seed": seed,
        "return_meta": return_meta,
    }

    if split_mode == "sample":
        full = ZernikeDualDataset(**dataset_kwargs)
        train_idx, val_idx, test_idx = _split_indices(
            len(full), [s["run_id"] for s in full.samples], val_split, test_split, seed
        )
        splits: dict[str, Dataset] = {
            "train": Subset(full, train_idx),
            "val": Subset(full, val_idx),
            "test": Subset(full, test_idx),
        }
        full_dataset: Dataset | None = full
    else:
        runs = sorted(run_ids) if run_ids is not None else sorted(
            p.name for p in Path(data_root).iterdir() if p.is_dir() and any(p.glob("sample_*"))
        )
        test_runs, val_runs, train_runs = _split_runs(runs, val_split, test_split)
        splits = {
            name: ZernikeDualDataset(**{**dataset_kwargs, "run_ids": rids})
            for name, rids in (("train", train_runs), ("val", val_runs), ("test", test_runs))
        }
        full_dataset = None

    collate_fn = collate_with_meta if return_meta else None
    loaders = {
        name: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            drop_last=False,
            generator=torch.Generator().manual_seed(seed) if name == "train" else None,
            collate_fn=collate_fn,
        )
        for name, ds in splits.items()
    }

    return {
        "train": loaders["train"],
        "val": loaders["val"],
        "test": loaders["test"],
        "dataset": full_dataset,
        "split_sizes": {name: len(ds) for name, ds in splits.items()},
        "split_mode": split_mode,
        "seed": seed,
        "dataset_kwargs": dataset_kwargs,
    }


# ---------------------------------------------------------------------------
# Label manifest (for CLI/plots)
# ---------------------------------------------------------------------------


def build_label_manifest(
    data_root: str | Path,
    run_ids: list[str] | None = None,
):
    """Build a per-sample label manifest: which samples have labels and where.

    Columns: ``sample_dir``, ``run_id``, ``sample_idx``,
    ``has_metadata_labels`` (bool), ``labels_path`` (sidecar path or None).

    Returns a ``pandas.DataFrame`` when pandas is importable (this venv has
    pandas 3.x); otherwise a dict keyed by ``sample_dir`` -> row dict. Returns
    ``None`` when the data root holds no run/sample dirs.
    """
    root = Path(data_root)
    if not root.is_dir():
        logger.warning(f"build_label_manifest: data_root not found: {root}")
        return None
    raw = _scan_samples(root, run_ids)
    if not raw:
        return None
    rows = [
        {
            "sample_dir": e["sample_dir"],
            "run_id": e["run_id"],
            "sample_idx": e["sample_idx"],
            "has_metadata_labels": _has_metadata_labels(e["metadata_path"]),
            "labels_path": e["labels_path"],
        }
        for e in raw
    ]
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:  # pragma: no cover - pandas is installed in this venv
        return {r["sample_dir"]: r for r in rows}
