"""Shared SPGD gradient-sign helper.

Every perturbation-based (SPGD) optimizer in this repo measures a paired
evaluation and turns it into a gradient for an ``algorithm`` optimizer::

    pos = param + disturb   ->  J_pos
    neg = param - disturb   ->  J_neg

``Base.update()`` (Adam / AdaMOD / SGD / Muno / ...) returns a **descent** step --
the classic Adam step ``lr * m_hat / (sqrt(v_hat) + eps)`` -- so callers pair it
with a subtraction::

    param = param - optimizer.update(gradient)

The gradient handed in must therefore already point *downhill*:

    objective to MINIMISE -> gradient = +(J_pos - J_neg) * disturb
    objective to MAXIMISE -> gradient = -(J_pos - J_neg) * disturb

``(J_pos - J_neg) * disturb`` is the *ascent* estimate of the objective, so the
maximise case is simply its negation.

Writing this sign by hand is the single most error-prone line in the codebase:
it has silently inverted an objective at least twice, and both times it was only
caught on hardware --

* ``optimizer/wfless/pib.py``: a ``to_min = -1`` assigned inside a nested
  ``calc_objective`` closure was a no-op, so the PIB objective was *minimised*
  (commit 74c7d0f).
* ``optimizer/wf/rms.py``: the RMS objective was *ascended* (maximised).

Use :func:`spgd_gradient` everywhere instead of writing ``-diff * disturb``.
"""

from __future__ import annotations

import numpy as np


def spgd_gradient(pos_obj, neg_obj, disturb, *, maximize: bool) -> np.ndarray:
    """Gradient to pass to ``optimizer.update()`` when using ``param - update``.

    Args:
        pos_obj: Objective measured at ``param + disturb``.
        neg_obj: Objective measured at ``param - disturb``.
        disturb: The signed perturbation vector used for that evaluation pair.
        maximize: ``True`` if the objective must be maximised, ``False`` to
            minimise. Keyword-only on purpose: every call site must state its
            objective direction explicitly.

    Returns:
        The gradient in the descent frame expected by ``Base.update()``.

    The subtraction is performed in float64. Bucket/energy objectives are
    unsigned sums, so ``pos - neg`` wraps on underflow and ``* -1`` raises
    ``OverflowError`` for uint64 inputs if the cast is omitted.
    """
    diff = np.asarray(pos_obj, dtype=np.float64) - np.asarray(neg_obj, dtype=np.float64)
    ascent = diff * disturb
    return -ascent if maximize else ascent
