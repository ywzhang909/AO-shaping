"""Hardware-free data-transformation layer for offline GSNet training.

This module is the **pure** part of a future ``gsnet-train`` command: it turns
raw beam-shaping debug pickles into the tensors a
``torch.utils.data.Dataset`` will hand to ``ml.gsnet.FourierGSNet``. It is
deliberately *not* the Dataset itself -- a later commit wraps these five
functions in a lazy ``Dataset``/``DataLoader``. Nothing here imports ``torch``,
``ml``, or any hardware driver; the whole module is importable on a machine with
no SLM, CCD or DM attached.

===========================  Data layout on disk  ===========================

The debug dumps are **two levels deep**, so every lookup must glob
*recursively*::

    data/debug/
      slm_zernike_shaping_<tag>/<timestamp>/<prefix>_<timestamp>_<timestamp>.pkl
      slm_pib_<tag>/<timestamp>/<prefix>_<timestamp>_<timestamp>.pkl

Each ``.pkl`` root is a **plain ``dict`` keyed by ``int``** (observed record
counts vary per file: 45, 101, ...), *not* a ``{"records": [...]}`` envelope.
Each record is a ``dict`` carrying at least:

===============  ==================  ==========================================
key              shape / dtype        meaning
===============  ==================  ==========================================
``_img``         ``(250, 248) uint8``  raw far-field CCD frame
``_c``           ``(nk,) float64``     Zernike coefficients, **radians**,
                                       Noll order, 1-based
``_grad``        ``(nk,) float64``     SPGD gradient (unused here)
``_phase``       ``(1200, 1920)``      *sometimes present* -- **ignored**:
                 uint16                only ``_c`` is used, uniformly across
                                       both data families
``m_pib`` etc.   float                 optimisation metrics (unused)
===============  ==================  ==========================================

``nk`` is **not** constant: 15, 66 and 78 are all observed, and it differs from
file to file. Use :func:`infer_n_max` on ``len(_c)`` at runtime -- never
hardcode a mode count, a file count or a record count.

===========================  Memory contract  ===========================

The corpus is ~5.2 GB across 25 pickles. This module therefore never holds more
than **one** pickle in RAM:

* :func:`build_record_index` returns only ``(Path, int)`` pairs -- no arrays are
  ever retained, and each dict is dropped before the next file is opened, so
  peak RSS during indexing is ONE pickle (~472 MB for the largest).
* The per-record transforms (:func:`reconstruct_pupil_phase_rad`,
  :func:`pupil_phase_to_grid`, :func:`farfield_to_grid`) are free functions over
  caller-supplied arrays; the future Dataset is responsible for loading exactly
  one record at a time.

===========================  DataLoader contract  ===========================

The target tuple is ``(source_intensity, target_intensity, gt_phase)``, each
``(1, 64, 64) float32``. This commit produces the two *measured/derived* parts:
``source_intensity`` via :func:`farfield_to_grid` and ``gt_phase`` via
:func:`reconstruct_pupil_phase_rad` + :func:`pupil_phase_to_grid`.
``target_intensity`` and the ``(1, ...)`` channel axis belong to the next
commit.
"""

from __future__ import annotations

import glob
import json
import math
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from loguru import logger

from ao_shaping.utils.image.beam_metrics import (
    clamp_center_to_frame,
    zero_order_center,
)
from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator
from ao_shaping.utils.wavefront.zernike_utils import parse_zernike_coefficients

__all__ = [
    "DEFAULT_ROOTS",
    "RecordIndex",
    "build_record_index",
    "farfield_to_grid",
    "infer_n_max",
    "pupil_phase_to_grid",
    "reconstruct_pupil_phase_rad",
]

#: Default search roots. ``slm_zernike_*`` matches ``slm_zernike_shaping_*``.
DEFAULT_ROOTS: tuple[str, ...] = (
    "data/debug/slm_zernike_*",
    "data/debug/slm_pib_*",
)


# ---------------------------------------------------------------------------
# Zernike mode-count arithmetic
# ---------------------------------------------------------------------------
def infer_n_max(n_terms: int) -> int:
    """Exact triangular inverse of the Zernike mode count.

    The number of Zernike modes up to radial order ``n`` is
    ``(n + 1) * (n + 2) / 2``. Given the length of a flat Noll-order
    coefficient vector (``len(record["_c"])``), recover ``n``.

    The inversion is done in **exact integer arithmetic** via
    :func:`math.isqrt` on ``1 + 8 * n_terms``; no float square root is taken,
    so large mode counts cannot be misrounded. Verified: 15 -> 4, 66 -> 10,
    78 -> 11.

    Args:
        n_terms: Number of Zernike modes, i.e. ``len(_c)``.

    Returns:
        Maximum radial order ``n_max``.

    Raises:
        ValueError: If ``n_terms`` is not a positive triangular number (e.g.
            7), or is zero/negative.
    """
    total = int(n_terms)
    if total < 1:
        raise ValueError(f"n_terms must be a positive integer, got {n_terms}")
    # (n + 1)(n + 2) / 2 == total  <=>  (2n + 3)^2 == 1 + 8 * total
    disc = 1 + 8 * total
    root = math.isqrt(disc)
    if root * root != disc:
        raise ValueError(
            f"n_terms={total} is not a triangular Zernike mode count "
            f"(no n_max satisfies (n_max+1)(n_max+2)/2 == {total})"
        )
    n_max = (root - 3) // 2
    if n_max < 0 or (n_max + 1) * (n_max + 2) // 2 != total:
        raise ValueError(
            f"n_terms={total} is not a valid Zernike mode count "
            f"(got n_max={n_max})"
        )
    return n_max


# ---------------------------------------------------------------------------
# Record indexing (paths + ints only -- never arrays)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RecordIndex:
    """Immutable index of every usable debug record.

    Each entry is a ``(pkl_path, record_key)`` pair. ``record_key`` indexes the
    pickle's root ``dict``. **No arrays are held** -- see the module-level
    memory contract.
    """

    entries: tuple[tuple[Path, int], ...]


def _pkl_files_under(root: str | Path) -> list[Path]:
    """Collect ``*.pkl`` files under one root, recursively.

    ``root`` may be a glob pattern (``data/debug/slm_zernike_*``), a directory
    prefix, or a ``.pkl`` file itself.

    The glob is **recursive** (``Path.rglob``): the debug dumps live at
    ``<prefix>_<timestamp>/<timestamp>/<prefix>_<timestamp>_<timestamp>.pkl``,
    i.e. two levels below the family directory, so a non-recursive
    ``glob("*.pkl")`` would silently find nothing.

    Args:
        root: Glob pattern, directory or ``.pkl`` path.

    Returns:
        Sorted list of ``.pkl`` paths (deduplicated within this root).
    """
    text = str(root)
    matched = (
        [Path(p) for p in glob.glob(text, recursive=True)]
        if glob.has_magic(text)
        else [Path(text)]
    )
    found: set[Path] = set()
    for item in matched:
        if item.is_dir():
            found.update(p for p in item.rglob("*.pkl") if p.is_file())
        elif item.is_file() and item.suffix.lower() == ".pkl":
            found.add(item)
    return sorted(found)


def _load_record_keys(path: Path) -> list[int]:
    """Load one pickle, return its sorted record keys, then free the arrays.

    MEMORY: this is the only place a pickle is deserialised, and the dict is
    dropped (``del``) before returning, so the caller's peak RAM is ONE pickle
    (~472 MB for the largest) regardless of corpus size.
    """
    with open(path, "rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - repo-internal debug dumps
    if not isinstance(payload, dict):
        raise TypeError(
            f"{path} does not contain a dict of records "
            f"(got {type(payload).__name__})"
        )
    keys = sorted(int(k) for k in payload.keys())
    del payload  # drop every array before the next file is opened
    return keys


def _read_index_cache(path: Path) -> RecordIndex | None:
    """Load a cached index, or ``None`` when absent/unreadable."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return RecordIndex(
            entries=tuple((Path(str(item[0])), int(item[1])) for item in payload)
        )
    except (OSError, ValueError, TypeError, IndexError) as exc:
        logger.warning("Ignoring unreadable index cache {}: {}", path, exc)
        return None


def _write_index_cache(path: Path, index: RecordIndex) -> None:
    """Persist an index as a JSON list of ``[str(path), int(key)]``."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [[str(p), int(k)] for p, k in index.entries]
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write index cache {}: {}", path, exc)


def build_record_index(
    roots: Sequence[str | Path] | str | Path = DEFAULT_ROOTS,
    limit_files: int | None = None,
    index_cache: str | Path | None = None,
) -> RecordIndex:
    """Index every debug record under ``roots`` as ``(path, key)`` pairs.

    Each root is globbed **recursively** for ``**/*.pkl``; paths matching more
    than one root are deduplicated and the final list is sorted, so the index is
    deterministic and independent of filesystem enumeration order.

    MEMORY: every pickle is opened once, only ``dict.keys()`` is read, and the
    dict is dropped before the next file is opened -- peak RAM is ONE pickle
    (~472 MB for the largest), never the ~5.2 GB corpus. No array ever enters
    the returned :class:`RecordIndex`.

    Args:
        roots: Glob patterns, directories or ``.pkl`` files to search. Defaults
            to :data:`DEFAULT_ROOTS` (``data/debug/slm_zernike_*`` and
            ``data/debug/slm_pib_*``).
        limit_files: Cap on the number of ``.pkl`` files, applied **after**
            sorting. ``None`` (default) means no cap.
        index_cache: Optional path to a JSON cache holding a list of
            ``[str(path), int(key)]``. When the file exists the expensive scan
            is skipped entirely; otherwise the fresh index is written there.

    Returns:
        A :class:`RecordIndex` of ``(pkl_path, record_key)`` tuples.

    Raises:
        TypeError: If a pickle root is not a ``dict`` of records.
    """
    cache_path = Path(index_cache) if index_cache is not None else None
    if cache_path is not None:
        cached = _read_index_cache(cache_path)
        if cached is not None:
            logger.info("Loaded {} cached records from {}", len(cached.entries), cache_path)
            return cached

    root_list = [roots] if isinstance(roots, (str, Path)) else list(roots)
    unique: set[Path] = set()
    for root in root_list:
        for candidate in _pkl_files_under(root):
            unique.add(candidate.resolve())
    files = sorted(unique)
    if limit_files is not None:
        files = files[: max(0, int(limit_files))]
    if not files:
        logger.warning("No debug pickles found under roots {}", root_list)

    entries: list[tuple[Path, int]] = []
    for path in files:
        for key in _load_record_keys(path):
            entries.append((path, key))

    index = RecordIndex(entries=tuple(entries))
    logger.info("Indexed {} records from {} pickles", len(index.entries), len(files))
    if cache_path is not None:
        _write_index_cache(cache_path, index)
    return index


# ---------------------------------------------------------------------------
# Per-record transforms
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _zernike_generator(
    slm_width: int, slm_height: int, radius: float, n_max: int
) -> ZernikeGenerator:
    """Return a cached :class:`ZernikeGenerator` for one panel geometry.

    Reused across records on purpose. ``zernike_utils.generate_zernike_phase``
    builds a *fresh* generator per call, which re-derives the full
    ``(height, width)`` RZern coordinate grid and aperture mask every single
    time -- on a 1200x1920 panel that dominated the offline pipeline (measured
    3.4 samples/s, i.e. ~16 min/epoch over the 3202-record corpus). The
    generator's own docstring prescribes exactly this reuse, and the README
    documents it as the canonical pattern. The Zernike math is unchanged: this
    is the same engine-layer class, only its lifetime is widened.

    Cached (not global) so the grid is built once per distinct geometry and
    released when the cache evicts. ``maxsize=4`` covers the realistic case of
    one or two geometries (e.g. 66-mode and 78-mode corpora) while bounding the
    resident grid count.

    Args:
        slm_width: SLM panel width in pixels.
        slm_height: SLM panel height in pixels.
        radius: Aperture radius in pixels.
        n_max: Maximum radial order.

    Returns:
        A generator primed with the project's 10-bit depth.
    """
    gen = ZernikeGenerator(
        resolution=(int(slm_width), int(slm_height)),
        radius=float(radius),
        n_orders=int(n_max),
    )
    gen.set_bits(10)
    return gen


def reconstruct_pupil_phase_rad(
    c: np.ndarray,
    *,
    n_max: int,
    slm_width: int,
    slm_height: int,
    radius: float,
) -> np.ndarray:
    """Reconstruct the SLM pupil phase in radians from a Noll-order ``_c``.

    **Units assumption: ``c`` is in radians.** This traces to the hardware
    path -- ``slm_zernike_pib._zernike_to_phase`` feeds ``_c`` straight into the
    radian APIs (``parse_zernike_coefficients`` then
    ``PatternHelper.generate_zernike_polynomial``), and the optimisation step
    sizes are radian-scale. No ``um_to_waves`` / ``* 2π`` conversion applies
    here; doing one would be the unit bug documented in ``AGENTS.md``.

    **Piston is included** (Noll 1 = ``(0, 0)``) and is a **far-field no-op**:
    a constant phase over the pupil shifts the global phase reference only, so
    the corresponding far-field intensity is unchanged. It is reconstructed
    anyway for faithfulness to the recorded coefficients.

    Zernike math is delegated to the canonical API
    (:func:`~ao_shaping.utils.wavefront.zernike_utils.generate_zernike_phase`,
    which applies
    :func:`~ao_shaping.utils.wavefront.zernike_utils.parse_zernike_coefficients`
    to its input at ``zernike_utils.py:231``); no Noll table or polynomial is
    hand-rolled here (repo red line). Those generators return **NaN outside the
    circular aperture** (``zernike_calc.ZernikeGenerator.mask``,
    ``zernike_calc.py:492-499``); the hardware path explicitly zeroes it
    (``pattern_helper.generate_zernike_polynomial``,
    ``utils/wavefront/pattern_helper.py:545-550``) and so do we, via
    ``np.nan_to_num(phase, nan=0.0)`` -- never propagate NaN into a training
    sample.

    Args:
        c: Flat Zernike coefficient vector in Noll order (1-based), i.e.
            ``record["_c"]``. ``len(c)`` must equal
            ``(n_max + 1) * (n_max + 2) / 2``.
        n_max: Maximum radial order (from :func:`infer_n_max`).
        slm_width: SLM panel width in pixels.
        slm_height: SLM panel height in pixels.
        radius: Aperture radius in pixels.

    Returns:
        ``(slm_height, slm_width)`` float64 array of unwrapped radians, exactly
        zero outside the aperture.

    Raises:
        ValueError: If ``c`` is not 1D, or its length disagrees with ``n_max``.
    """
    coeffs = np.asarray(c, dtype=np.float64).ravel()
    if coeffs.size == 0:
        raise ValueError("c must contain at least one Zernike coefficient")
    expected = (int(n_max) + 1) * (int(n_max) + 2) // 2
    if coeffs.size != expected:
        raise ValueError(
            f"len(c)={coeffs.size} does not match n_max={n_max} "
            f"(expected {expected} modes); derive n_max via infer_n_max(len(c))"
        )

    # The flat Noll-order vector is handed to the canonical generator as-is:
    # parse_zernike_coefficients() is applied to it first, so a separate
    # pre-parse would be pure duplication.
    gen = _zernike_generator(
        int(slm_width), int(slm_height), float(radius), int(n_max)
    )
    coeffs_dict = parse_zernike_coefficients(coeffs, n_max=int(n_max))
    if not coeffs_dict:
        return np.zeros((int(slm_height), int(slm_width)), dtype=np.float64)
    # astype first: generate_polynomial can hand back a uint16 zeros array for
    # degenerate input, which would break the float64 contract; nan_to_num then
    # reproduces the hardware zero-outside-aperture.
    return np.nan_to_num(
        np.asarray(gen.generate_polynomial(coeffs_dict), dtype=np.float64), nan=0.0
    )


def _block_axis(length: int, grid: int) -> tuple[int, int, int]:
    """Whole-block geometry for one axis.

    The output always has exactly ``grid`` cells along the axis, so the axis is
    divided into ``grid`` blocks of ``ceil(length / grid)`` pixels each. When
    ``length`` is not an exact multiple the input is zero-padded **up** to
    ``grid * ceil(length / grid)`` pixels -- nothing is ever discarded.

    Args:
        length: Axis length in pixels.
        grid: Number of output cells along the axis.

    Returns:
        ``(block_size, total_padded_length, pad_total)`` where
        ``block_size == ceil(length / grid)`` and
        ``total_padded_length == grid * block_size``.
    """
    block = max(1, -(-int(length) // int(grid)))
    total = int(grid) * block
    return block, total, total - int(length)


def pupil_phase_to_grid(phase_rad: np.ndarray, grid: int) -> np.ndarray:
    """Area-average a full-panel pupil phase down to ``(grid, grid)``.

    Uses an **exact block mean** (area average) -- ``scipy.ndimage.zoom`` is
    deliberately avoided because it applies no pre-filter and aliases badly on a
    1200x1920 -> 64x64 reduction (the sampled high-frequency Zernike structure
    folds back as noise). A block mean is the correct area integral of the
    pupil: each output cell is the mean phase of the ``(H/grid) x (W/grid)``
    input pixels it represents.

    Padding: the input is **not** truncated -- it is zero-padded symmetrically
    so the axis splits into exactly ``grid`` equal blocks of
    ``ceil(length / grid)`` pixels. For the real 1200x1920 panel this pads
    1200 -> 1216 (8 rows top, 8 rows bottom), giving 64 row-blocks of 19 px,
    while 1920 is already exact (64 col-blocks of 30 px). The aperture radius
    of 600 exactly fills the original 1200-row panel height, so the added rows
    extend the array just past the illuminated disc: nothing is discarded, and
    the only effect is a small edge dilution of the outermost block rows.

    Args:
        phase_rad: ``(H, W)`` pupil phase (radians, zero outside the aperture).
        grid: Output block count per axis.

    Returns:
        ``(grid, grid)`` float32 array of block means.

    Raises:
        ValueError: If ``phase_rad`` is not a non-empty 2D array, or ``grid``
            is not a positive integer.
    """
    phase = np.asarray(phase_rad, dtype=np.float64)
    if phase.ndim != 2 or phase.size == 0:
        raise ValueError(f"phase_rad must be a non-empty 2D array, got {phase.shape}")
    step = int(grid)
    if step < 1:
        raise ValueError(f"grid must be a positive integer, got {grid}")

    height, width = phase.shape
    block_h, total_h, pad_h = _block_axis(height, step)
    block_w, total_w, pad_w = _block_axis(width, step)

    if pad_h or pad_w:
        top = pad_h // 2
        left = pad_w // 2
        padded = np.zeros((total_h, total_w), dtype=np.float64)
        padded[top : top + height, left : left + width] = phase
    else:
        padded = phase

    blocks = padded.reshape(step, block_h, step, block_w).mean(axis=(1, 3))
    return blocks.astype(np.float32)


def _fixed_window(frame: np.ndarray, top: int, left: int, grid: int) -> np.ndarray:
    """Extract a ``grid x grid`` window, zero-padding outside the frame."""
    height, width = frame.shape
    out = np.zeros((grid, grid), dtype=np.float64)
    src_y0, src_x0 = max(top, 0), max(left, 0)
    src_y1, src_x1 = min(top + grid, height), min(left + grid, width)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[src_y0 - top : src_y1 - top, src_x0 - left : src_x1 - left] = frame[
            src_y0:src_y1, src_x0:src_x1
        ]
    return out


def farfield_to_grid(img: np.ndarray, grid: int, *, eps: float = 1e-12) -> np.ndarray:
    """Crop the 0-order spot out of a far-field frame onto a ``(grid, grid)`` grid.

    The 0-order spot is located by **global argmax**, never by assuming the
    frame centre -- on the 2f Fourier bench the optical axis (0-order = frame
    global maximum) lands at the camera centre only by luck (AGENTS.md
    documents frame centre (1344, 760) vs 0-order (1441, 705)). The locator is
    the canonical :func:`~ao_shaping.utils.image.beam_metrics.zero_order_center`
    with ``refine=False``, which is literally
    ``np.unravel_index(np.argmax(frame))`` (``beam_metrics.py:442-444``) plus
    the all-dark-frame guard that falls back to the frame centre
    (``beam_metrics.py:439-440``). The in-frame window clamp reuses
    :func:`~ao_shaping.utils.image.beam_metrics.clamp_center_to_frame`
    (``beam_metrics.py:544-570``).

    The ``resample_to_grid`` helper (``utils/image/resample.py:55-98``) is
    **not** reused because it cannot express this contract: it crops to the
    *target aspect ratio* and then bilinearly interpolates with
    ``scipy.ndimage.zoom``, so the effective field of view depends on the
    source aspect ratio instead of being a fixed ``grid x grid`` pixel window,
    and it neither peak-normalises to ``[0, 1]`` nor returns float32.

    Args:
        img: 2D far-field intensity frame (e.g. ``record["_img"]``, uint8).
        grid: Output grid size (same value for both axes).
        eps: Floor for the peak normaliser, so an all-zero frame cannot divide
            by zero.

    Returns:
        ``(grid, grid)`` float32 array in ``[0, 1]``, peak-normalised. The 0-order
        peak lands at ``(grid // 2, grid // 2)`` whenever it is at least
        ``grid // 2`` pixels from every frame edge; a degenerate all-zero frame
        returns all zeros (no NaN, no exception).

    Raises:
        ValueError: If ``img`` is not a non-empty 2D array, or ``grid`` is not a
            positive integer.
    """
    frame = np.asarray(img, dtype=np.float64)
    if frame.ndim != 2 or frame.size == 0:
        raise ValueError(f"img must be a non-empty 2D array, got {frame.shape}")
    step = int(grid)
    if step < 1:
        raise ValueError(f"grid must be a positive integer, got {grid}")

    frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    height, width = frame.shape

    # (x, y) project convention; refine=False keeps the exact argmax anchor.
    cx, cy = zero_order_center(frame, refine=False)
    cx, cy = clamp_center_to_frame((cx, cy), (height, width), min(step, height, width))

    half = step // 2
    crop = _fixed_window(frame, int(cy) - half, int(cx) - half, step)

    peak = float(crop.max())
    if peak <= eps:
        return np.zeros((step, step), dtype=np.float32)
    return (crop / peak).astype(np.float32)
