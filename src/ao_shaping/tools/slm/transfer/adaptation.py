"""Device-free transfer policy: separate illumination drift from aberrations."""
from __future__ import annotations

import numpy as np

from .splits import SEALED_SPLITS, SELECTION_SPLIT

__all__ = ["SEALED_SPLITS", "SELECTION_SPLIT", "configure_adaptation",
           "final_test_quality", "needs_full_adaptation", "validation_quality"]


def configure_adaptation(model, mode, learn_geometry=False, learn_psf=False):
    """Reuse measured geometry; release only parameters justified by the stage.

    ``learn_geometry`` only releases the optional bounded 2x2 geometry
    correction of :class:`PhysicalForwardFit`; it never turns that parameter on
    by itself (a model built without ``learn_geometry=True`` has none), and the
    historical two-argument call keeps every other parameter exactly as before.
    ``learn_psf`` does the same for the optional bounded static CCD PSF
    (``psf_raw``): the model must have been built with ``learn_psf=True`` for
    the parameter to exist, and this call only decides whether it trains.
    """
    if mode not in {"alignment", "full"}:
        raise ValueError("adaptation must be alignment or full")
    model.d.requires_grad_(mode == "full")
    model.log_width.requires_grad_(mode == "full")
    model.residual.requires_grad_(mode == "full")
    # Position and shared NNLS gain/background adapt in both modes.
    model.center.requires_grad_(True)
    model.origin.requires_grad_(True)
    raw = getattr(model, "geometry_raw", None)
    if raw is not None:
        raw.requires_grad_(bool(learn_geometry))
    psf_raw = getattr(model, "psf_raw", None)
    if psf_raw is not None:
        psf_raw.requires_grad_(bool(learn_psf))


def validation_quality(entry, rows):
    """Use disjoint phases for model selection, never repeated training frames.

    Only ``SELECTION_SPLIT`` rows may feed promotion (#254 T2): the old
    ``!= "train"`` filter let sealed ``test`` / ``final_test`` metrics leak into
    model selection.
    """
    selected = [m for row, m in zip(rows, entry["metrics"], strict=True)
                if row["split"] == SELECTION_SPLIT]
    if not selected:
        raise ValueError("validation phases required for automatic adaptation")
    return _aggregate_quality(selected, "validation")


def final_test_quality(entry, rows):
    """One-shot scoring of the sealed tier after candidates are frozen (#254).

    Must run exactly once per candidate, after promotion; the value never
    feeds back into selection, adaptation escalation, or checkpoint choice.
    """
    selected = [m for row, m in zip(rows, entry["metrics"], strict=True)
                if row["split"] == "final_test"]
    if not selected:
        raise ValueError("final_test phases required for sealed evaluation")
    return _aggregate_quality(selected, "final_test")


def _aggregate_quality(selected, tier: str):
    correlation = float(np.mean([m["correlation"] for m in selected]))
    relative_l2 = float(np.mean([m["relative_l2"] for m in selected]))
    cv_error = float(np.mean([m["cv_abs_error"] for m in selected]))
    score = 1 - correlation + relative_l2
    if not np.isfinite([correlation, relative_l2, cv_error, score]).all():
        raise ValueError(f"nonfinite {tier} quality")
    return dict(correlation=correlation, relative_l2=relative_l2,
                cv_abs_error=cv_error, score=score)


def needs_full_adaptation(quality):
    """Escalation heuristic, not a device-access or shaping admission gate."""
    return quality["correlation"] < .85 or quality["relative_l2"] > .25
