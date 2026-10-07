"""Read-side contract for ``train_amp``'s ``best_coefficients.pt``.

This module owns the **read** contract for the checkpoint that
:func:`ml.zernike.train_amp.train` writes as ``best_coefficients.pt``. The
**write** side is inline in :mod:`ml.zernike.train_amp` and is deliberately
neither imported nor modified here: reader/writer agreement is enforced by the
round-trip test driving the **real** :func:`ml.zernike.train_amp.train` (see
``tests/ao_shaping/ml/zernike/test_amp_checkpoint.py``), not by a shared key
constant. That is what makes a change to the writer's key set fail a test in
this module's suite rather than drifting silently.

The writer does **not** record ``radius``, ``center_crop``, or
``conserve_energy``, so this module never claims to verify them: they are
carried on the result as *assumed* values and are surfaced (not checked) by
:func:`check_geometry`. This is a read seam only -- it wires itself into no
training loop and imports nothing from :mod:`ao_shaping`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from loguru import logger

from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel

__all__ = [
    "UNVERIFIED_GEOMETRY_FIELDS",
    "GeometryReport",
    "TrainedForwardModel",
    "check_geometry",
    "load_trained_forward_model",
]

#: Geometry knobs the checkpoint does not record. The loader fills these with
#: fixed assumed values (the :class:`~ml.zernike.models.ZernikeAmpConfig`
#: defaults) and :func:`check_geometry` refuses to present them as verified.
UNVERIFIED_GEOMETRY_FIELDS: tuple[str, ...] = ("radius", "center_crop", "conserve_energy")


@dataclass(frozen=True)
class TrainedForwardModel:
    """A :class:`~ml.zernike.models.ZernikeAmpModel` rehydrated from a checkpoint.

    ``coefficients`` is the piston-excluded coefficient vector in raw radians
    (``K = calc_n_zernike_terms(n_max) - 1`` entries), copied *into* the model's
    own parameter without replacing it.

    ``assumed`` maps each name in :data:`UNVERIFIED_GEOMETRY_FIELDS` to the
    value the loader *supplied* because the checkpoint carries no record of it.
    It exists so that "unknown" is never silently presented as "verified equal":
    the model's behaviour depends on these knobs, yet none of them can be
    checked against the checkpoint, so a caller comparing geometry must be told
    exactly what was assumed rather than led to believe the model was fully
    reconstructed.
    """

    model: "ZernikeAmpModel"
    coefficients: np.ndarray
    n_max: int
    grid: int
    observable: str
    normalization: str
    far_field_padding: int
    assumed: Mapping[str, Any]
    source: Path
    train_config: Mapping[str, Any]


def load_trained_forward_model(path: str | Path) -> TrainedForwardModel:
    """Load a ``best_coefficients.pt`` written by ``train_amp.train``.

    Rebuilds the :class:`~ml.zernike.models.ZernikeAmpModel` from the recorded
    geometry, copies the stored coefficient vector into it, and returns a
    :class:`TrainedForwardModel` carrying the geometry, the geometry the loader
    had to *assume* (unrecorded), and the writer's ``config`` mapping.

    Args:
        path: Path to the ``.pt`` file.

    Returns:
        The rehydrated model and its metadata.

    Raises:
        ValueError: If a required key is missing, the coefficient count does not
            match the recorded ``n_max``, or a coefficient is non-finite.
        RuntimeError: If the copy into the model's parameter did not round-trip
            bit-for-bit.
    """
    source = Path(path)
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"checkpoint {source} must be a dict of tensors/metadata, got {type(ckpt).__name__}"
        )

    required = (
        "coefficients",
        "n_max",
        "grid",
        "observable",
        "normalization",
        "far_field_padding",
        "config",
    )
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise ValueError(f"checkpoint {source} is missing required key(s): {', '.join(missing)}")

    n_max = int(ckpt["n_max"])
    grid = int(ckpt["grid"])
    # These three geometry knobs are not recorded in the checkpoint, so they are
    # assumed. The dict below is the single source for the assumed value *and*
    # what is actually passed to the config, so the two can never drift apart.
    assumed: dict[str, Any] = {
        "radius": None,
        "center_crop": True,
        "conserve_energy": False,
    }
    config = ZernikeAmpConfig(
        n_max=n_max,
        grid=grid,
        observable=ckpt["observable"],
        normalization=ckpt["normalization"],
        far_field_padding=int(ckpt["far_field_padding"]),
        **assumed,
    )
    model = ZernikeAmpModel(config)

    vec = np.asarray(ckpt["coefficients"], dtype=np.float64).reshape(-1)
    if len(vec) != model.K:
        raise ValueError(
            f"checkpoint coefficient length {len(vec)} != model K={model.K} "
            f"(n_max={n_max}); piston-excluded count must match"
        )
    if not np.isfinite(vec).all():
        n_bad = int((~np.isfinite(vec)).sum())
        raise ValueError(
            f"checkpoint coefficients contain {n_bad} non-finite "
            f"value(s); refusing to load a corrupted vector (n_max={n_max}, K={model.K})"
        )

    # Copy into the model's own Parameter (do NOT reassign ``model.coefficients``).
    with torch.no_grad():
        model.coefficients.copy_(torch.as_tensor(vec, dtype=model.coefficients.dtype))
    if not np.array_equal(model.coefficients_array(), vec):
        raise RuntimeError(
            f"coefficients did not round-trip into the model for {source} "
            f"(n_max={n_max}, K={model.K})"
        )

    logger.info(
        "loaded {} from {}: n_max={} grid={} observable={} normalization={} "
        "far_field_padding={} K={}",
        type(model).__name__,
        source,
        n_max,
        grid,
        ckpt["observable"],
        ckpt["normalization"],
        int(ckpt["far_field_padding"]),
        model.K,
    )

    train_config = dict(ckpt["config"]) if isinstance(ckpt["config"], Mapping) else {}
    return TrainedForwardModel(
        model=model,
        coefficients=vec,
        n_max=n_max,
        grid=grid,
        observable=ckpt["observable"],
        normalization=ckpt["normalization"],
        far_field_padding=int(ckpt["far_field_padding"]),
        assumed=assumed,
        source=source,
        train_config=train_config,
    )


@dataclass(frozen=True)
class GeometryReport:
    """Outcome of :func:`check_geometry`: hard errors plus advisory notes."""

    ok: bool
    errors: tuple[str, ...]
    notes: tuple[str, ...]


def check_geometry(
    loaded: TrainedForwardModel,
    *,
    region: int,
    n_orders: int,
    far_field_size: int,
    assume_unverified_geometry: bool = False,
) -> GeometryReport:
    """Check a loaded forward model against a consuming loop's geometry.

    Pure -- no I/O. Compares the fields the checkpoint **does** record (``grid``,
    ``n_max``, ``far_field_padding``, ``observable``, ``normalization``) exactly
    against the loop's ``region`` / ``n_orders`` / ``far_field_size``. It **cannot**
    compare :data:`UNVERIFIED_GEOMETRY_FIELDS`, because the checkpoint stores no
    record of them; instead of silently assuming them, the checker refuses the
    comparison (an error naming them) unless ``assume_unverified_geometry`` is
    passed, in which case it records an advisory note.

    Args:
        loaded: The rehydrated model.
        region: The loop's pupil-grid side (the model's ``grid``).
        n_orders: The loop's Zernike order. The loop's vector is piston-*inclusive*
            (``calc_n_zernike_terms(n_orders)`` entries) while the checkpoint's is
            piston-*excluded*, so they only agree when ``n_orders == n_max``.
        far_field_size: The loop's far-field grid side; the effective padding is
            ``far_field_size / region``.
        assume_unverified_geometry: When ``False`` (default), the three unrecorded
            fields produce a hard error naming them; when ``True`` they are
            acknowledged in a note instead.

    Returns:
        A :class:`GeometryReport` with ``ok`` true when there are no errors.
    """
    errors: list[str] = []
    notes: list[str] = []

    if loaded.grid != region:
        errors.append(f"grid {loaded.grid} != region {region}")
    if loaded.n_max != n_orders:
        errors.append(
            f"n_max {loaded.n_max} != n_orders {n_orders} (the loop vector is "
            f"piston-inclusive, calc_n_zernike_terms(n_orders), while the checkpoint "
            f"is piston-excluded; they align only when the orders agree)"
        )
    if far_field_size % region != 0:
        errors.append(
            f"far_field_size {far_field_size} is not an integer multiple of "
            f"region {region}; effective padding is not an integer"
        )
    elif int(far_field_size / region) != loaded.far_field_padding:
        errors.append(
            f"effective padding far_field_size/region = {far_field_size}/{region} "
            f"= {int(far_field_size / region)} != far_field_padding {loaded.far_field_padding}"
        )
    if loaded.observable != "intensity":
        errors.append(
            f"observable {loaded.observable!r} != 'intensity' (the loop's forward is "
            f"hard-coded to intensity)"
        )
    if loaded.normalization != "peak":
        errors.append(f"normalization {loaded.normalization!r} != 'peak'")

    if not assume_unverified_geometry:
        errors.append(
            "unverified geometry (no record in checkpoint): "
            + ", ".join(UNVERIFIED_GEOMETRY_FIELDS)
            + f"; assumed values: {dict(loaded.assumed)}; pass "
            "assume_unverified_geometry=True to acknowledge them"
        )
    else:
        notes.append(
            "assumed (not verified) geometry: "
            + ", ".join(f"{k}={v!r}" for k, v in loaded.assumed.items())
        )

    if not errors:
        notes.append(
            f"geometry consistent: region={region} n_orders={n_orders} "
            f"far_field_size={far_field_size} K={loaded.model.K} "
            f"effective_padding={int(far_field_size / region)}"
        )

    return GeometryReport(ok=not errors, errors=tuple(errors), notes=tuple(notes))
