"""Single source of truth: objective name -> search direction.

Every optimizer and report script that decides whether to *ascend* or
*descend* an objective used to carry its own copy of the maximisation
vocabulary (``"max" if objective in ("pib", ...) else "min"``), and the
copies had already drifted:

* :mod:`ao_shaping.optimizer.wfless.pib` — ``("pib", "avg_radiu")``
* :mod:`ao_shaping.optimizer.wfless.slm_zernike_pib` —
  ``("pib", "avg_radiu", "shape", "roi_pib", "rms_pib")``
* ``scripts/slm/generate_slm_pib_sim_report.py`` — the wider tuple again,
  as a hand-maintained ``MAXIMIZE_OBJECTIVES`` mirroring the optimizer.

This module is the one place that owns the vocabulary. It has **no
imports from within ``ao_shaping``** so it can be used both from library
code and from ``scripts/`` report generators without any import-cycle
risk.

Direction semantics (higher is better):

* ``pib`` / ``avg_radiu`` / ``shape`` / ``roi_pib`` / ``rms_pib``
  — the search *ascends* (``RecordMode.MAX``).
* ``radiu`` / ``rmse`` / ``rmse_out`` / ``pearson`` — the search
  *descends* (``RecordMode.MIN``). Note ``pearson`` here means the
  *error* ``1 - corr`` as computed by the shaping metrics, which is
  smaller when the correlation is higher.
* ``quality`` (square family) — a combined score that is always
  higher-is-better (``RecordMode.MAX``); see
  :data:`ao_shaping.utils.image.target.objective.SQUARE_OBJECTIVE_CHOICES`.

The returned values are plain ``"max"`` / ``"min"`` strings so that every
existing comparison site (``mode == "max"``), the ``str``-mixin
``RecordMode`` enum, and the ``ShapingObjectiveParams.Mode.from_string``
coercion all keep working unchanged.
"""
from __future__ import annotations

from typing import Literal

__all__ = ["MAXIMIZE_OBJECTIVES", "objective_direction"]

#: Objectives the search ASCENDS (higher score is better). Single source of
#: truth shared by all optimizers and the PIB report script.
MAXIMIZE_OBJECTIVES: tuple[str, ...] = (
    "pib",
    "avg_radiu",
    "shape",
    "roi_pib",
    "rms_pib",
    "quality",
)


def objective_direction(objective: str) -> Literal["max", "min"]:
    """Search direction for ``objective``: ``"max"`` or ``"min"``.

    The comparison is case-insensitive and whitespace-tolerant (keys
    reconstructed from JSON sidecars can carry padding); in production the
    vocabulary is validated upstream (``pib.py`` / ``slm_zernike_pib.py``), so
    this only matters for report scripts. Unknown objectives default to
    ``"min"`` — mirroring the historical ``if objective in (...) else "min"``
    fallback, so a misspelled or new objective name can never silently switch
    an existing search from descending to ascending.
    """
    return "max" if str(objective).strip().lower() in MAXIMIZE_OBJECTIVES else "min"
