"""Shared static-aberration calibration from a batch of records.

:class:`CalibrationRecord` (in :mod:`...bench_geometry_common`) carries one
``(far field, pupil phase)`` pair; :func:`calibrate_shared_aberration` fits ONE
coefficient vector shared by the whole batch with the Zernike forward model.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)
from ao_shaping.optimizer.wfless.bench_geometry_common import CalibrationRecord
from ao_shaping.utils.image.beam_metrics import zero_order_center
from ao_shaping.utils.io import paths
from ao_shaping.utils.io.recorder_keys import (
    RECORD_COEFFS,
    RECORD_IMAGE,
    RECORD_PHASE,
)
from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms


def _to_model_units(
    optimizer: ZernikeCoefficientOptimizer, frame: np.ndarray, phase: np.ndarray
) -> np.ndarray:
    """Rescale a measured frame onto the forward model's intensity scale.

    Step A's loss is ``mean((I_model / peak_anchor - I_meas / peak_peak) ** 2)``
    with ``peak_anchor`` taken from the *measurement*, so the two sides are only
    commensurate when the measurement already shares the model's units. The
    simulated CCD does not: it re-normalises every frame to ``peak_photons``
    counts (60000 by default) while the model's own far field peaks in the
    thousands, so a raw frame is ~47x brighter than any model intensity. Feeding
    that to Step A silently changes the objective from "match the normalised
    pattern" into "maximise model brightness under the measured core", whose
    optimum is a focusing aberration rather than the true one -- the fit then
    wanders to a coefficient error of ~0.7 rad no matter how long it runs.

    The gain is recovered from the model itself: the twin is a linear detector,
    so the measured frame is the model's far field times a constant, and scaling
    by ``model_peak / measured_peak`` restores model units exactly. The scale is
    recomputed from the *current* coefficients so it tightens as the fit
    converges, and any degenerate frame (zero peak) is passed through untouched.

    Args:
        optimizer: The Step A optimizer, whose current coefficients set the scale.
        frame: Measured far field, CCD units.
        phase: The pupil phase the frame was measured through, radians.

    Returns:
        The frame rescaled to model units, or the input unchanged when either
        peak is non-positive.
    """
    measured_peak = float(np.max(frame))
    if not np.isfinite(measured_peak) or measured_peak <= 0.0:
        return frame
    model_peak = float(
        np.max(optimizer.forward_intensity(optimizer.coefficients, phase))
    )
    if not np.isfinite(model_peak) or model_peak <= 0.0:
        return frame
    return frame * (model_peak / measured_peak)


# ---------------------------------------------------------------------------
# Shared-aberration calibration
# ---------------------------------------------------------------------------


@dataclass
class FitAberrationResult:
    """Outcome of :func:`calibrate_shared_aberration`.

    Attributes:
        coefficients: The shared fitted coefficients, Noll-ordered, radians.
        loss_history: Peak-normalised MSE per Adam step across all records.
        labels: Provenance of each record consumed.
        source: Where the records came from (``"explicit"``, a dataset path, or
            ``"none"`` when no usable data was found).
        n_records: Number of records the fit consumed.
        epochs: Optimization steps granted to each record.
        iterations: Adam steps actually taken.
        record_errors: Per-record L2 error against ``record.coefficients``,
            empty when no ground truth was supplied.
    """

    coefficients: np.ndarray
    loss_history: list[float]
    labels: list[str]
    source: str
    n_records: int
    epochs: int
    iterations: int
    record_errors: list[float] = field(default_factory=list)


def _resize_grid(array: np.ndarray, grid: int) -> np.ndarray:
    """Resample a 2D array onto a ``(grid, grid)`` grid by nearest neighbour.

    Kept local rather than imported from ``ml.gsnet_debug.offline`` so the
    optimizer layer never depends on the runner layer (and never drags its
    plotting/CLI imports into an algorithm import).

    Args:
        array: Any non-empty 2D array.
        grid: Target side length.

    Returns:
        A ``(grid, grid)`` float64 array.

    Raises:
        ValueError: If ``array`` is empty or not 2D, or ``grid < 1``.
    """
    frame = np.asarray(array, dtype=np.float64)
    if frame.ndim != 2 or frame.size == 0:
        raise ValueError(f"expected a non-empty 2D array, got shape {frame.shape}")
    if int(grid) < 1:
        raise ValueError(f"grid must be a positive integer, got {grid}")
    rows = np.clip(
        np.round(np.linspace(0, frame.shape[0] - 1, grid)).astype(int),
        0,
        frame.shape[0] - 1,
    )
    cols = np.clip(
        np.round(np.linspace(0, frame.shape[1] - 1, grid)).astype(int),
        0,
        frame.shape[1] - 1,
    )
    return frame[np.ix_(rows, cols)]


def _crop_to_grid(image: np.ndarray, grid: int) -> np.ndarray:
    """Crop a far field around its global ``argmax`` onto a ``grid`` window.

    The 0-order is the frame's global maximum, never the geometric centre, so
    the window is anchored by
    :func:`~ao_shaping.utils.image.beam_metrics.zero_order_center` with
    ``refine=False``. The window is clamped inside the frame, so a frame smaller
    than ``grid`` yields a zero-padded result rather than an exception.

    Args:
        image: 2D far-field intensity frame.
        grid: Output side length.

    Returns:
        A ``(grid, grid)`` float64 window centred on the 0-order.
    """
    frame = np.nan_to_num(np.asarray(image, dtype=np.float64), nan=0.0)
    side = int(grid)
    center_x, center_y = zero_order_center(frame, refine=False)
    out = np.zeros((side, side), dtype=np.float64)
    top = center_y - side // 2
    left = center_x - side // 2
    src_y0, src_x0 = max(top, 0), max(left, 0)
    src_y1 = min(src_y0 + side, frame.shape[0])
    src_x1 = min(src_x0 + side, frame.shape[1])
    # A window whose anchor is off-centre hangs off the far edge as well as the
    # near one, so the destination span can run past ``out`` (rows 10:34 of a
    # 32-row buffer). Clamp the destination and copy only the overlapping part
    # of the source; the remainder of ``out`` stays zero padding.
    dst_y0, dst_x0 = src_y0 - top, src_x0 - left
    dst_y1 = min(dst_y0 + (src_y1 - src_y0), side)
    dst_x1 = min(dst_x0 + (src_x1 - src_x0), side)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[
        src_y0 : src_y0 + (dst_y1 - dst_y0),
        src_x0 : src_x0 + (dst_x1 - dst_x0),
    ]
    return out


def _coerce_to_record(
    item: Any, region: int, n_coeffs: int, label: str
) -> CalibrationRecord | None:
    """Coerce a loaded dataset entry into a :class:`CalibrationRecord`.

    Accepts either a mapping with ``_img`` / ``_phase`` (and optionally ``_c``)
    or an object exposing ``image`` / ``phase`` attributes.

    Args:
        item: The loaded entry.
        region: Model grid side length.
        n_coeffs: Length of the coefficient vector.
        label: Fallback provenance label.

    Returns:
        A record, or ``None`` when the entry lacks an image or a phase.
    """
    if isinstance(item, Mapping):
        image = item.get(RECORD_IMAGE, item.get("image"))
        phase = item.get(RECORD_PHASE, item.get("phase"))
        truth = item.get(RECORD_COEFFS, item.get("coefficients"))
    else:
        image = getattr(item, "image", None)
        phase = getattr(item, "phase", None)
        truth = getattr(item, "coefficients", None)
    if image is None or phase is None:
        return None
    vector: np.ndarray | None = None
    if truth is not None:
        vector = np.zeros(n_coeffs, dtype=np.float64)
        raw = np.asarray(truth, dtype=np.float64).ravel()
        keep = min(raw.size, n_coeffs)
        vector[:keep] = raw[:keep]
    return CalibrationRecord(
        image=_crop_to_grid(image, region),
        phase=_resize_grid(phase, region),
        label=label,
        coefficients=vector,
    )


def _records_from_pkl(
    path: Path, region: int, n_coeffs: int
) -> list[CalibrationRecord]:
    """Build records from a fat GSNet ``.pkl`` file.

    Args:
        path: Path to the pickle.
        region: Model grid side length.
        n_coeffs: Length of the coefficient vector.

    Returns:
        The parsed records, in file order.

    Raises:
        ValueError: If the file holds no entry with both an image and a phase.
    """
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # trusted local dataset
    entries = payload if isinstance(payload, (list, tuple)) else [payload]
    records: list[CalibrationRecord] = []
    for index, entry in enumerate(entries):
        record = _coerce_to_record(entry, region, n_coeffs, f"{path.name}#{index}")
        if record is not None:
            records.append(record)
    if not records:
        raise ValueError(f"{path} holds no entry with both an image and a phase")
    return records


def _per_record_coefficients(
    path: Path, stored: np.ndarray, n_records: int
) -> list[np.ndarray]:
    """Split a cached coefficient array into one vector per frame.

    The cache writer stores ``c_flat.npy`` as a **1-D concatenation** and keeps the record
    boundaries in ``c_offsets.npy`` so each record's term count stays exact. Indexing the flat
    array per record instead hands back a single scalar taken from somewhere inside another
    record's vector: the numbers look plausible and are attributed to the wrong frame, which is
    how a real cache read as ground truth it never contained.
    """
    if stored.ndim == 1:
        offsets_path = path / "c_offsets.npy"
        if not offsets_path.exists():
            raise ValueError(
                f"{path}/{offsets_path.name} is missing, but c_flat.npy holds {stored.size} "
                "concatenated values: the per-record boundaries cannot be recovered"
            )
        offsets = np.load(offsets_path)
        if offsets.size != n_records + 1:
            raise ValueError(
                f"{path}/{offsets_path.name} holds {offsets.size} boundaries for {n_records} "
                f"frames (expected {n_records + 1})"
            )
        if int(offsets[0]) != 0 or int(offsets[-1]) != stored.size:
            raise ValueError(
                f"{path}/{offsets_path.name} spans [{int(offsets[0])}, {int(offsets[-1])}] but "
                f"c_flat.npy holds {stored.size} values: the cache is truncated or mixed"
            )
        return [
            stored[int(offsets[index]) : int(offsets[index + 1])]
            for index in range(n_records)
        ]

    if stored.shape[0] != n_records:
        raise ValueError(
            f"{path}/c_flat.npy holds {stored.shape[0]} rows for {n_records} frames: the "
            "coefficient array and the images are not a pair"
        )
    return [stored[index] for index in range(n_records)]


def _records_from_cache(
    path: Path, region: int, n_coeffs: int
) -> list[CalibrationRecord]:
    """Build records from a lean ``.gsnet_cache`` directory.

    The lean cache stores flattened coefficients and images plus a
    ``meta.json`` but **no** pupil phase, so the phase defaults to zeros (a flat
    SLM) and the resulting fit is correspondingly weaker. Prefer the fat
    ``.pkl`` when both are present.

    Args:
        path: The cache directory.
        region: Model grid side length.
        n_coeffs: Length of the coefficient vector.

    Returns:
        The parsed records.

    Raises:
        ValueError: If the cache arrays or the metadata describing their shape
            are missing or inconsistent -- including a 1-D ``c_flat.npy`` whose
            ``c_offsets.npy`` is absent or does not span it.
    """
    coeffs = np.load(path / "c_flat.npy")
    images = np.load(path / "img_flat.npy")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    shape = tuple(int(v) for v in meta["img_shape"])
    per_record = int(np.prod(shape))
    if per_record <= 0 or images.size % per_record:
        raise ValueError(
            f"{path}/img_flat.npy holds {images.size} entries, not a multiple of "
            f"{per_record}"
        )
    # Reshape once rather than slicing per record: the on-disk layout is
    # ``(n_records, prod(img_shape))`` (the real cache is 2-D), so slicing a
    # *pixel* count off axis 0 would hand back whole rows instead of one frame.
    # ``reshape`` is order preserving, so flat, 2-D and 3-D caches all work.
    frames = images.reshape(images.size // per_record, *shape)
    flat_phase = np.zeros((region, region), dtype=np.float64)
    records: list[CalibrationRecord] = []
    rows = _per_record_coefficients(path, coeffs, len(frames))
    for index in range(len(frames)):
        vector = np.zeros(n_coeffs, dtype=np.float64)
        raw = np.asarray(rows[index], dtype=np.float64).ravel()
        keep = min(raw.size, n_coeffs)
        vector[:keep] = raw[:keep]
        records.append(
            CalibrationRecord(
                image=_crop_to_grid(frames[index], region),
                phase=flat_phase.copy(),
                label=f"{path.name}#{index}",
                coefficients=vector,
            )
        )
    if not records:
        raise ValueError(f"{path} contains no cached frames")
    return records


def _discover_dataset(
    search_root: Path, region: int, n_coeffs: int
) -> tuple[list[CalibrationRecord], str]:
    """Locate the newest usable calibration dataset under ``search_root``.

    The fat ``.pkl`` is preferred over the lean ``.gsnet_cache`` (it carries the
    pupil phase); ties on modification time prefer the ``.pkl``.

    Args:
        search_root: Directory searched recursively.
        region: Model grid side length.
        n_coeffs: Length of the coefficient vector.

    Returns:
        A tuple of the records and their source description; the list is empty
        when nothing usable was found.
    """
    if not search_root.exists():
        return [], "none"
    candidates: list[tuple[float, int, Path]] = []
    candidates.extend((p.stat().st_mtime, 0, p) for p in search_root.rglob("*.pkl"))
    candidates.extend(
        (p.stat().st_mtime, 1, p)
        for p in search_root.rglob("*.gsnet_cache")
        if p.is_dir()
    )
    if not candidates:
        return [], "none"
    _, _, newest = max(candidates)
    try:
        if newest.is_dir():
            records = _records_from_cache(newest, region, n_coeffs)
        else:
            records = _records_from_pkl(newest, region, n_coeffs)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        json.JSONDecodeError,
        pickle.UnpicklingError,
    ) as error:
        logger.warning("calibration dataset {} unusable: {}", newest, error)
        return [], "none"
    return records, str(newest)


def calibrate_shared_aberration(
    records: Sequence[CalibrationRecord] | None = None,
    *,
    search_root: Path | str | None = None,
    region: int = 64,
    n_orders: int = 10,
    epochs: int = 50,
    lr: float = 0.05,
    dtype: str = "float64",
    device: str | None = None,
    seed: int = 0,
    frozen_modes: tuple[int, ...] = (1, 2, 3),
    far_field_size: int | None = None,
    pupil_oversampling: int = 1,
) -> FitAberrationResult:
    """Fit ONE aberration shared by a whole set of records.

    A single :class:`ZernikeCoefficientOptimizer` is reused for every record so
    all records inform the same coefficient vector instead of each receiving an
    independent fit. Each record is driven by one
    :meth:`~ZernikeCoefficientOptimizer.run` call, which warm-starts from the
    coefficients accumulated so far.

    All records are fitted by **one** optimizer over consecutive
    :meth:`~ZernikeCoefficientOptimizer.update` calls, cycling through the
    records, so every record informs the same coefficient vector at every step.
    This is what makes the shared fit work. Issuing one
    :meth:`~ZernikeCoefficientOptimizer.run` per record instead optimises each
    record to its own optimum and restarts Adam in between, so the sequence
    degenerates into a chain of independent fits in which the last record wins
    and the shared vector ends up fitting none of them: on the twin at
    ``region=64``/``n_orders=10`` (66 coefficients) that per-record ``run``
    chain *raised* the loss and left the injected-mode error above 1.2 rad, and
    got worse as the budget grew.

    Cycling needs the convergence latch handled. ``update`` stops moving the
    coefficients once the plateau detector has seen ``PLATEAU_PATIENCE``
    non-improving steps, and nothing clears the latch, so a shared best-loss
    tracker across heterogeneous records trips it easily. A tripped latch is
    re-armed by rebuilding the optimizer around its *current* coefficients.
    :meth:`~ZernikeCoefficientOptimizer.reset` is deliberately not used: it
    restores the coefficients passed to ``__init__`` and would discard the whole
    fit.

    Identifiability: the fit is only well posed when the records sample

    **varied SLM phases**. A far field measured under a flat SLM phase focuses to
    a near-delta spot, so its normalized intensity barely responds to the pupil
    aberration and the coefficients are not recoverable. Datasets whose records
    all share one flat (or otherwise identical) phase will return a poor fit
    even though the optimizer behaved correctly; add records with different
    SLM phases, as the canonical recovery fixture does.

    Args:
        records: Explicit records. When ``None``, the newest dataset under
            ``search_root`` is loaded.
        search_root: Directory searched when ``records`` is ``None``. Defaults
            to ``data/debug``.
        region: Model grid side length.
        n_orders: Maximum Zernike radial order.
        epochs: Optimization steps granted to each record, so the total budget
            is ``epochs * len(records)``. Values at or above
            ``PLATEAU_PATIENCE`` may stop a record early on its own plateau.
        lr: Adam learning rate. ``0.05`` is the measured sweet spot for the
            joint cycling loop on the twin: at ``region=64``/``n_orders=10`` with
            16 records it reached a 0.19 rad full-vector error against 0.30 at
            ``0.1`` and 0.94 at ``0.01``, while ``0.2`` diverges outright (73 rad).

        dtype: Torch dtype name for the model.
        device: Torch device, or ``None`` to auto-detect.
        seed: Seed for the Step A optimizer.
        frozen_modes: Noll indices held at zero for the whole fit. Defaults to
            ``(1, 2, 3)`` because piston and tilt are unidentifiable from a
            far-field intensity; on real captures, leaving them free put 56% of
            the fitted coefficient norm into those degenerate directions without
            improving the fit at all. Pass ``()`` to fit every mode.
        far_field_size: Zero-padded FFT size for the forward model, or ``None``
            for the native ``region``-sized transform. **Must** be set when the
            records come from a real bench: the focal-plane sampling is set by
            the optical geometry, not by the model grid, so fitting on the
            native transform rescales every recovered coefficient. On this bench
            the mismatch is ~29.5x, which is larger than any coefficient being
            measured.
        pupil_oversampling: Integer factor by which :class:`ZernikeCoefficientOptimizer`
            bilinearly enlarges the pupil before the transform. The real 2f bench
            displays the panel phase by bilinear upsampling, so a model grid of
            ``region`` samples covers ``pupil_side / region`` panel pixels per
            sample; leaving this at ``1`` propagates a piecewise-constant pupil
            that the panel never shows. Measured on the GS 1000 px bench
            (``region=64``, 900 px box, 512 far field) with an independent
            bilinear-900 propagation as ground truth: ``1`` gave a far-field
            correlation of 0.066 and recovered no aberration (residual 1.92 rad
            against a 1.20 rad truth), ``14`` gave 0.9999 and 0.084 rad. Use
            ``round(pupil_side_px / region)`` whenever records come off the real
            display chain. Note that at ``factor > 1`` the model stops masking the
            pupil to its circular aperture, which is what makes the two agree --
            the panel applies the phase over the whole illuminated box.

    Returns:
        A :class:`FitAberrationResult`. When no data is available the
        coefficients are zeros and ``source`` is ``"none"`` -- the function
        degrades gracefully instead of raising.

    Raises:
        ValueError: If ``epochs < 1``, if ``records`` is given but empty, or if
            a record is missing its image or phase.
        ImportError: If PyTorch is unavailable.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if int(region) < 1:
        raise ValueError(f"region must be a positive integer, got {region}")
    n_coeffs = calc_n_zernike_terms(n_orders)

    if records is None:
        root = Path(search_root) if search_root is not None else paths.data_subdir("debug")
        found, source = _discover_dataset(root, int(region), n_coeffs)
        if not found:
            logger.warning(
                "no usable calibration dataset under {}; returning zeros", root
            )
            return FitAberrationResult(
                coefficients=np.zeros(n_coeffs, dtype=np.float64),
                loss_history=[],
                labels=[],
                source="none",
                n_records=0,
                epochs=epochs,
                iterations=0,
            )
        records = found
        logger.info("calibrating on {} record(s) from {}", len(records), source)
    else:
        source = "explicit"
        records = list(records)

    if not records:
        raise ValueError("records must contain at least one entry")

    optimizer = ZernikeCoefficientOptimizer(
        n_orders=n_orders,
        region=region,
        lr=lr,
        max_iterations=max(1, epochs),
        device=device,
        seed=seed,
        dtype=dtype,
        frozen_modes=frozen_modes,
        far_field_size=far_field_size,
        pupil_oversampling=pupil_oversampling,
    )
    frames = _csa_build_frames(records, optimizer, int(region), far_field_size)

    def rearm(coefficients: np.ndarray) -> ZernikeCoefficientOptimizer:
        """Rebuild the optimizer around ``coefficients`` to clear a tripped latch.

        ``reset`` cannot do this: it restores the coefficients handed to
        ``__init__`` rather than the current ones, so re-arming with it would
        discard the whole fit. A fresh instance keeps the progress and discards
        only the stale latch and the Adam moments that stopped moving.
        """
        return ZernikeCoefficientOptimizer(
            n_orders=n_orders,
            region=region,
            initial_coefficients=np.asarray(coefficients, dtype=np.float64),
            lr=lr,
            max_iterations=max(1, epochs),
            device=device,
            seed=seed,
            dtype=dtype,
            frozen_modes=frozen_modes,
            far_field_size=far_field_size,
            pupil_oversampling=pupil_oversampling,
        )

    # All records are driven through consecutive ``update`` calls on ONE
    # optimizer, cycling. Fitting them one after another with ``run`` instead
    # degenerates into a chain of independent fits -- each ``run`` optimises its
    # record to that record's own optimum and restarts Adam -- so the last record
    # wins and the shared vector ends up fitting none of them. ``epochs`` keeps
    # its per-record meaning: the loop runs ``epochs * len(records)`` steps.
    optimizer, history = _csa_cycle(optimizer, frames, epochs, rearm)

    coefficients = optimizer.coefficients
    errors = [
        float(np.linalg.norm(coefficients - np.asarray(record.coefficients)))
        for record in records
        if record.coefficients is not None
    ]
    logger.info(
        "shared calibration: records={} epochs={} steps={} loss {} -> {}",
        len(records),
        epochs,
        len(history),
        history[0] if history else float("nan"),
        history[-1] if history else float("nan"),
    )
    return FitAberrationResult(
        coefficients=np.array(coefficients, dtype=np.float64),
        loss_history=history,
        labels=[record.label for record in records],
        source=source,
        n_records=len(records),
        epochs=epochs,
        iterations=len(history),
        record_errors=errors,
    )


def _csa_build_frames(
    records: Sequence[CalibrationRecord],
    optimizer: ZernikeCoefficientOptimizer,
    region: int,
    far_field_size: int | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Coerce every record onto the model grids and into model units.

    Args:
        records: Records supplying the measured far fields and pupil phases.
        optimizer: Optimizer owning the forward model the frames are scaled for.
        region: Model grid side length for the pupil phase.
        far_field_size: Zero-padded FFT side length, or ``None`` for the native grid.

    Returns:
        One ``(image, phase)`` pair per record, both on the model grids.

    Raises:
        ValueError: If a record carries an empty image or phase.
    """
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for record in records:
        image = np.asarray(record.image, dtype=np.float64)
        if image.size == 0:
            raise ValueError(f"record {record.label!r} has an empty image")
        # A real-bench record must land on the model's *far-field* grid: the
        # forward model predicts ``far_field_size`` samples, so coercing the
        # measurement to ``region`` would make ``update`` resample a
        # ``region``-pixel CCD window up to ``far_field_size`` and fit a window
        # that covers ~8x less sky than the model does.
        meas_grid = int(far_field_size) if far_field_size else int(region)
        if image.shape != (meas_grid, meas_grid):
            image = _crop_to_grid(image, meas_grid)
        phase = np.asarray(record.phase, dtype=np.float64)
        if phase.size == 0:
            raise ValueError(f"record {record.label!r} has an empty phase")
        if phase.shape != (region, region):
            phase = _resize_grid(phase, region)
        # Rescaled to model units: a CCD frame peaks at ``peak_photons`` counts
        # while the model peaks in the thousands, and that mismatch would turn
        # the fit into a brightness maximisation rather than a pattern match
        # (see :func:`_to_model_units`).
        frames.append((_to_model_units(optimizer, image, phase), phase))
    return frames


def _csa_cycle(
    optimizer: ZernikeCoefficientOptimizer,
    frames: Sequence[tuple[np.ndarray, np.ndarray]],
    epochs: int,
    rearm: Callable[[np.ndarray], ZernikeCoefficientOptimizer],
) -> tuple[ZernikeCoefficientOptimizer, list[float]]:
    """Cycle one optimizer over ``frames`` for ``epochs`` steps per frame.

    Args:
        optimizer: Optimizer carrying the coefficients being fitted.
        frames: ``(image, phase)`` pairs produced by :func:`_csa_build_frames`.
        epochs: Steps granted to each frame.
        rearm: Rebuilds the optimizer around its current coefficients to clear a
            tripped convergence latch.

    Returns:
        The optimizer after the final step and the per-step loss history.
    """
    history: list[float] = []
    total = max(1, int(epochs)) * len(frames)
    count = len(frames)
    for index in range(total):
        image, phase = frames[index % count]
        optimizer.update(image, phase)
        if optimizer.converged:
            optimizer = rearm(optimizer.coefficients)
        # ``last_loss`` is populated by every ``update``; the ``None`` branch only
        # exists on the type, since the attribute is Optional-typed.
        loss = optimizer.last_loss
        if loss is not None:
            history.append(float(loss))
    return optimizer, history
