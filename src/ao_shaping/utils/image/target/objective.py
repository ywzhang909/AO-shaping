"""Shaping objective selection, adaptive ``rms_pib`` weights and the metric panel.

Part of the :mod:`ao_shaping.utils.image.target` package (split by type).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from loguru import logger

from ao_shaping.utils.image.target.metrics import (
    TargetShape,
    rmse_shape_metric,
    rms_pib_terms,
    roi_energy_loss,
    roi_pib_metric,
    shape_metric,
    shape_stage_from_energy,
)


def _update_dynamic_weights(
    state: dict,
    *,
    pib: float,
    rms: float,
    j: float,
    ee: float | None = None,
    w_ema_decay: float = 0.9,
    w_floor: float = 0.1,
    w_temperature: float = 8.0,
) -> tuple[float, float] | tuple[float, float, float]:
    """Adaptively re-weight the PIB, RMS and (optional) energy terms.

    The term that improves ``J`` more gets the higher weight ("哪个对J提升大则
    哪个权重大"). Weights are softmax-normalised EMA scores of the *positive*
    contributions of each term to the combined objective:

    * ``c_i = w_i * (term_i - prev_term_i)`` for each participating term;
    * the EMA is updated ONLY on positive contributions (improving steps);
    * if ALL contributions are ``<= 0`` the weights stay unchanged;
    * ``w_i = w_floor + (1 - n*w_floor) * softmax(T*ema_i, ...)`` for the ``n``
      participating terms (n=2 or n=3).

    When ``ee`` is ``None`` (legacy two-term objective) the behaviour is
    byte-identical to the previous ``(w_pib, w_rms)`` pair; when ``ee`` is
    given the energy-conservation term participates and a three-weight tuple
    ``(w_pib, w_rms, w_ee)`` is returned (always summing to 1).
    """
    three_term = ee is not None
    state.setdefault("w_pib", 1.0 / 3 if three_term else 0.5)
    state.setdefault("w_rms", 1.0 / 3 if three_term else 0.5)
    state.setdefault("w_ee", 1.0 / 3 if three_term else 0.0)
    state.setdefault("ema_pib", 0.0)
    state.setdefault("ema_rms", 0.0)
    state.setdefault("ema_ee", 0.0)
    state.setdefault("prev_j", None)
    state.setdefault("prev_pib", None)
    state.setdefault("prev_rms", None)
    state.setdefault("prev_ee", None)
    state.setdefault("prev_set", False)

    if not state["prev_set"]:
        # First call: record the baseline and keep the initial weights.
        state["prev_j"] = float(j)
        state["prev_pib"] = float(pib)
        state["prev_rms"] = float(rms)
        if three_term:
            state["prev_ee"] = float(ee)
        state["prev_set"] = True
        if three_term:
            return (
                float(state["w_pib"]),
                float(state["w_rms"]),
                float(state["w_ee"]),
            )
        return float(state["w_pib"]), float(state["w_rms"])

    w_pib = float(state["w_pib"])
    w_rms = float(state["w_rms"])
    c_pib = w_pib * (float(pib) - float(state["prev_pib"]))
    c_rms = w_rms * (float(rms) - float(state["prev_rms"]))
    if c_pib > 0.0:
        state["ema_pib"] = (
            w_ema_decay * float(state["ema_pib"]) + (1.0 - w_ema_decay) * c_pib
        )
    if c_rms > 0.0:
        state["ema_rms"] = (
            w_ema_decay * float(state["ema_rms"]) + (1.0 - w_ema_decay) * c_rms
        )

    c_ee = 0.0
    if three_term:
        w_ee = float(state["w_ee"])
        c_ee = w_ee * (float(ee) - float(state["prev_ee"]))
        if c_ee > 0.0:
            state["ema_ee"] = (
                w_ema_decay * float(state["ema_ee"]) + (1.0 - w_ema_decay) * c_ee
            )
        if c_pib <= 0.0 and c_rms <= 0.0 and c_ee <= 0.0:
            # No improving contribution this step: keep the current weights.
            state["prev_j"] = float(j)
            state["prev_pib"] = float(pib)
            state["prev_rms"] = float(rms)
            state["prev_ee"] = float(ee)
            return w_pib, w_rms, w_ee
    elif c_pib <= 0.0 and c_rms <= 0.0:
        # No improving contribution this step: keep the current weights.
        state["prev_j"] = float(j)
        state["prev_pib"] = float(pib)
        state["prev_rms"] = float(rms)
        return w_pib, w_rms

    def _softmax_frac(emas: list[float]) -> list[float]:
        clipped = [float(np.clip(e, -10.0, 10.0)) for e in emas]
        clipped = [0.0 if not np.isfinite(e) else e for e in clipped]
        exps = [math.exp(w_temperature * e) for e in clipped]
        total = sum(exps)
        return [e / total for e in exps]

    if three_term:
        fracs = _softmax_frac(
            [float(state["ema_pib"]), float(state["ema_rms"]), float(state["ema_ee"])]
        )
        w_pib = w_floor + (1.0 - 3.0 * w_floor) * fracs[0]
        w_rms = w_floor + (1.0 - 3.0 * w_floor) * fracs[1]
        w_ee = w_floor + (1.0 - 3.0 * w_floor) * fracs[2]
        state["w_pib"] = float(w_pib)
        state["w_rms"] = float(w_rms)
        state["w_ee"] = float(w_ee)
        state["prev_j"] = float(j)
        state["prev_pib"] = float(pib)
        state["prev_rms"] = float(rms)
        state["prev_ee"] = float(ee)
        return float(w_pib), float(w_rms), float(w_ee)

    fracs = _softmax_frac([float(state["ema_pib"]), float(state["ema_rms"])])
    w_pib = w_floor + (1.0 - 2.0 * w_floor) * fracs[0]
    w_rms = 1.0 - w_pib
    state["w_pib"] = float(w_pib)
    state["w_rms"] = float(w_rms)
    state["prev_j"] = float(j)
    state["prev_pib"] = float(pib)
    state["prev_rms"] = float(rms)
    return float(w_pib), float(w_rms)


def _resolve_init_weights(
    w_pib_init: float | None,
    w_rms_init: float | None,
    w_ee_init: float | None,
) -> tuple[float, float, float]:
    """Resolve the initial PIB/RMS/EE weights of the ``rms_pib`` objective.

    With no weight provided the (1/3, 1/3, 1/3) default is returned. When any
    weight is provided, provided terms are kept exactly and unprovided terms
    share the remaining mass equally, so the triple always sums to 1 (the
    invariant the adaptive update maintains from the first adapting step on).
    When all three are provided they are normalised to sum 1.

    Raises:
        ValueError: if the provided weights sum to more than 1 (with fewer than
            three provided) or to 0 (with all three provided).
    """
    given = (w_pib_init, w_rms_init, w_ee_init)
    if all(w is None for w in given):
        return (1.0 / 3, 1.0 / 3, 1.0 / 3)
    given_sum = float(sum(w for w in given if w is not None))
    n_missing = sum(1 for w in given if w is None)
    if n_missing > 0:
        if given_sum > 1.0 + 1e-9:
            raise ValueError(
                f"initial rms_pib weights must sum to <= 1 when not all are "
                f"provided, got {given_sum!r} "
                f"(w_pib_init={w_pib_init!r}, w_rms_init={w_rms_init!r}, "
                f"w_ee_init={w_ee_init!r})"
            )
        fill = (1.0 - given_sum) / n_missing
        out = tuple(fill if w is None else float(w) for w in given)
    else:
        if given_sum <= 0.0:
            raise ValueError(
                "all initial rms_pib weights are provided but sum to 0: "
                f"(w_pib_init={w_pib_init!r}, w_rms_init={w_rms_init!r}, "
                f"w_ee_init={w_ee_init!r})"
            )
        provided = [w for w in given if w is not None]
        out = tuple(float(w) / given_sum for w in provided)
    return (out[0], out[1], out[2])


@dataclass(frozen=True)
class ShapeScoringParams:
    """Weights handed to :func:`shape_metric` by the ``shape`` objective and panel."""

    w_uniformity: float = 3.0
    w_peak: float = 0.5
    w_displacement: float = 0.5
    log_uniformity: bool = True


@dataclass(frozen=True)
class ShapingObjectiveParams:
    """Everything :class:`ShapingObjective` needs to score a camera frame.

    Attributes:
        objective: one of :data:`SHAPING_OBJECTIVE_CHOICES`.
        mode: ``"max"`` or ``"min"`` - the direction the search ascends, which
            also selects the sign of the guard penalty.
        shape: target shape used by the ROI-based objectives.
        size: target ROI short side in pixels.
        aspect_ratio: long/short side of the target ROI.
        reference_center: window-local ``(x, y)`` of the FIXED target ROI. It
            never tracks the measured spot, so a drifting spot is penalised.
        shape_schedule: coarsen->middle->fine ``shape`` weight staging.
        scoring: ``shape_metric`` weights.
        max_roi_energy_loss: in-ROI energy-loss fraction that abandons an
            evaluation; ``0`` disables the guard.
        ideal_spot_radius: fixed radius of the baseline ``m_pib7`` metric.
        r_bucket: initial (possibly dynamically derived) bucket radius. Mutable
            through :meth:`ShapingObjective.set_bucket`.
        w_ema_decay / w_floor / w_temperature: ``rms_pib`` weight adaptation.
        w_pib_init / w_rms_init / w_ee_init: ``rms_pib`` initial weights.
    """

    objective: str
    mode: str
    shape: TargetShape
    size: float
    aspect_ratio: float
    reference_center: tuple[float, float]
    shape_schedule: bool = False
    scoring: ShapeScoringParams = field(default_factory=ShapeScoringParams)
    max_roi_energy_loss: float = 0.0
    ideal_spot_radius: int = 6
    r_bucket: float = 1.0
    w_ema_decay: float = 0.9
    w_floor: float = 0.1
    w_temperature: float = 8.0
    w_pib_init: float | None = None
    w_rms_init: float | None = None
    w_ee_init: float | None = None


@dataclass(frozen=True)
class ObjectiveResult:
    """One scored camera frame.

    Attributes:
        j: the objective value the search ascends (guard-penalised if the
            frame was abandoned).
        ratio: the secondary quantity the objective reports for logging.
        terms: ``(j, pib_term, rms_term, ee_term)`` of the last ``rms_pib``
            evaluation, ``(0.0, 0.0, 0.0, 0.0)`` for every other objective.
            Frozen so a caller can hold a *positive-perturbation* result across
            the negative-perturbation evaluation without it being overwritten.
    """

    j: float
    ratio: float
    terms: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


SHAPING_OBJECTIVE_CHOICES = (
    "pib",
    "radiu",
    "avg_radiu",
    "rmse",
    "shape",
    "roi_pib",
    "rms_pib",
)


GUARDED_OBJECTIVES = ("pib", "rmse", "shape", "roi_pib", "rms_pib")


class ShapingObjective:
    """Objective selection, the ``m_*`` metric panel and the ROI energy guard.

    One instance replaces the per-objective closure dispatch that used to live
    inside the SLM PIB search loop. It owns:

    * the seven objective families (:meth:`raw` / :meth:`__call__`);
    * the ``rms_pib`` adaptive term weights (:meth:`adapt_weights`);
    * the in-ROI energy-loss safety guard, which abandons an evaluation by
      returning a value far worse than any valid state (``j - 1e3`` in
      ``"max"`` mode, ``j + 1e3`` in ``"min"`` mode);
    * the cross-objective metric panel (:meth:`metric_panel`).

    The bucket radius is *live*: the search shrinks it mid-run through
    :meth:`set_bucket`, and both the ``pib`` objective and the panel read the
    current value.
    """

    def __init__(
        self,
        params: ShapingObjectiveParams,
        target_func: Any,
        init_img: np.ndarray,
    ) -> None:
        """Arm the objective against the initial (flat/loaded) frame.

        Args:
            params: Objective selection and scoring configuration.
            target_func: An ``ImageTargetFunc``-like object providing
                ``pib(img, radius)``, ``radius(img, energy=...)`` and
                ``avg_radius(img, moment=...)``. Required for ``pib``, ``radiu``
                and ``avg_radiu``; may be ``None`` for the ROI-only objectives.
            init_img: The initial camera frame. Its in-ROI energy arms the
                guard, its total energy is the ``rms_pib``/panel energy
                baseline, and ``rms_pib`` seeds its weight state.

        Raises:
            ValueError: if ``params.objective`` or ``params.mode`` is not one
                of the supported values, or a ``target_func``-backed objective
                was requested without a ``target_func``.
        """
        if params.objective not in SHAPING_OBJECTIVE_CHOICES:
            raise ValueError(
                f"objective must be one of {SHAPING_OBJECTIVE_CHOICES}, "
                f"got {params.objective!r}"
            )
        if params.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {params.mode!r}")
        if params.objective in ("pib", "radiu", "avg_radiu") and target_func is None:
            raise ValueError(
                f"objective {params.objective!r} needs a target_func providing "
                f"pib()/radius()/avg_radius()"
            )

        self._params = params
        self._target_func = target_func
        self._r_bucket = float(params.r_bucket)
        # Baseline window energy from the flat-phase capture: "no energy lost"
        # means sum(img) stays at this level. Both the ``rms_pib`` term and the
        # panel's ``m_ee`` are fractions of it.
        init_sum = float(np.sum(np.asarray(init_img, dtype=np.float64)))
        self._init_energy = init_sum
        self._panel_init_sum = init_sum
        self._terms: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._shape_state: dict[str, float] = {"best_energy": 0.0}
        self._violations = 0

        if params.objective == "rms_pib":
            w_pib, w_rms, w_ee = _resolve_init_weights(
                params.w_pib_init, params.w_rms_init, params.w_ee_init
            )
            # Seeded up front so the first adapting call sees the resolved
            # weights instead of the helper's own 1/3 default.
            self._rms_pib_state: dict = {
                "w_pib": w_pib,
                "w_rms": w_rms,
                "w_ee": w_ee,
            }
        else:
            self._rms_pib_state = {}

        # Reference = the in-ROI energy of the initial (flat/loaded) frame. Any
        # evaluation whose in-ROI energy dropped by more than
        # ``max_roi_energy_loss`` (fraction of that reference) is *abandoned*:
        # it is scored far worse than any valid state, so the search never adopts
        # it and the SLM is never left there. ``0`` disables the guard.
        self._guard_ref_energy: float | None = None
        if params.max_roi_energy_loss > 0.0 and params.objective in GUARDED_OBJECTIVES:
            ref = self._fixed_roi_energy(init_img)
            self._guard_ref_energy = ref if np.isfinite(ref) and ref > 0.0 else None
            if self._guard_ref_energy is None:
                logger.warning(
                    "ROI energy guard disabled: initial in-ROI energy is {:.6f}", ref
                )
            else:
                logger.info(
                    "ROI energy guard armed: reference energy {:.4f}, max loss {:.1%}",
                    self._guard_ref_energy,
                    params.max_roi_energy_loss,
                )

    def _fixed_roi_energy(self, frame: np.ndarray) -> float:
        """Light fraction inside the **FIXED** target ROI - the guard's observable.

        Deliberately NOT the running (spot-tracking) ROI used by the objective:
        that one follows the measured spot centre, so it always retains the same
        energy and could never detect a loss. This ROI stays at
        ``reference_center`` / ``size`` / ``shape``.
        """
        p = self._params
        return float(
            roi_pib_metric(frame, p.reference_center, p.shape, p.size, p.aspect_ratio)[
                0
            ]
        )

    def raw(self, img: np.ndarray) -> tuple[float, float]:
        """Score one frame without the safety guard.

        Returns:
            ``(j, ratio)`` - the objective value and the secondary quantity the
            objective reports for logging.
        """
        p = self._params
        objective = p.objective

        if objective == "pib":
            # Maximize PIB. Same convention as pib.py.
            pib, pib_ratio = self._target_func.pib(img, self._r_bucket)
            return float(pib), float(pib_ratio)

        if objective == "roi_pib":
            # Maximise the brightness inside the TARGET-SHAPED ROI: the fraction
            # of the light landing in the target rectangle (exposure-invariant
            # ratio, no uniformity/peak/drift penalties).
            score, energy = roi_pib_metric(
                img, p.reference_center, p.shape, p.size, p.aspect_ratio
            )
            return float(score), float(energy)

        if objective == "rms_pib":
            # Combined PIB + in-ROI RMS + energy-conservation objective with
            # adaptively-weighted terms:
            # J = w_pib(t)*pib_term + w_rms(t)*rms_term + w_ee(t)*ee_term.
            # The weights adapt so the term that improves J more gets the higher
            # weight (see _update_dynamic_weights). The target ROI is FIXED at
            # reference_center (never tracks the spot). ``ee_term`` = fraction
            # of the baseline (flat-phase) window energy still inside the window,
            # so a search that diffracts/scatters light out of the window (or
            # pumps it to a dark halo) is penalised even though pib/rms are
            # exposure-invariant ratios (energy-encircled constraint; see
            # AGENTS.md anti-pattern).
            w_pib = float(self._rms_pib_state["w_pib"])
            w_rms = float(self._rms_pib_state["w_rms"])
            w_ee = float(self._rms_pib_state["w_ee"])
            pib_term, rms_term = rms_pib_terms(
                img, p.reference_center, p.shape, p.size, p.aspect_ratio
            )
            frame = np.asarray(img, dtype=np.float64)
            ee_term = float(
                np.clip(
                    frame.sum() / max(self._init_energy, np.finfo(np.float64).eps),
                    0.0,
                    1.0,
                )
            )
            j = w_pib * pib_term + w_rms * rms_term + w_ee * ee_term
            self._terms = (
                float(j),
                float(pib_term),
                float(rms_term),
                float(ee_term),
            )
            return float(j), float(pib_term)

        if objective == "shape":
            # The target ROI is FIXED at reference_center - it never tracks the
            # measured spot, so a spot that drifts/scatters out of the box is
            # penalised instead of being followed (displacement weight = 0 for
            # the same reason: the energy term already captures the drift).
            stage = (
                shape_stage_from_energy(self._shape_state["best_energy"])
                if p.shape_schedule
                else None
            )
            score, energy = shape_metric(
                img,
                p.reference_center,
                p.reference_center,
                p.shape,
                p.size,
                p.aspect_ratio,
                w_uniformity=p.scoring.w_uniformity,
                w_peak=p.scoring.w_peak,
                w_displacement=p.scoring.w_displacement,
                stage=stage,
                log_uniformity=p.scoring.log_uniformity,
            )
            # Monotone: the schedule may only get stricter, never laxer.
            self._shape_state["best_energy"] = max(
                self._shape_state["best_energy"], energy
            )
            return float(score), float(energy)

        if objective == "rmse":
            # Minimise the RMSE between the frame and the uniform-intensity
            # target, both normalised to unit sum (exposure / laser-drift
            # invariant).
            rmse, energy = rmse_shape_metric(
                img, p.reference_center, p.shape, p.size, p.aspect_ratio
            )
            return float(rmse), float(energy)

        if objective == "radiu":
            return float(self._target_func.radius(img, energy=0.99)), 0.0

        # avg_radiu: maximize average radius.
        avg_r, avg_ratio = self._target_func.avg_radius(img, moment=1.0)
        return float(avg_r), float(avg_ratio)

    def __call__(self, img: np.ndarray) -> ObjectiveResult:
        """Score one frame, applying the in-ROI energy-loss safety guard.

        Returns a strongly penalised ``j`` (and the true ratio for logging) when
        the in-ROI energy loss exceeds ``max_roi_energy_loss`` - the evaluation
        is thereby abandoned.
        """
        j, ratio = self.raw(img)
        if self._guard_ref_energy is None:
            return ObjectiveResult(float(j), float(ratio), self._terms)

        loss = roi_energy_loss(self._guard_ref_energy, self._fixed_roi_energy(img))
        if loss > self._params.max_roi_energy_loss:
            self._violations += 1
            if self._violations <= 5 or self._violations % 50 == 0:
                logger.warning(
                    "ROI energy loss {:.1%} exceeds the {:.1%} limit - "
                    "evaluation abandoned (#{} violations)",
                    loss,
                    self._params.max_roi_energy_loss,
                    self._violations,
                )
            bad = j - 1e3 if self._params.mode == "max" else j + 1e3
            return ObjectiveResult(float(bad), float(ratio), self._terms)
        return ObjectiveResult(float(j), float(ratio), self._terms)

    def tracking_value(self, img: np.ndarray, res: ObjectiveResult) -> float:
        """Value the search tracks as "best" and logs under the objective name.

        For ``pib`` that is the exposure-independent bucket ratio at the fixed
        ideal radius (matches the logged column); for every other objective it
        is the value the gradient uses - e.g. the encircle radius, which must
        be MINIMISED (a ``>`` comparison would keep the worst).
        """
        if self._params.objective == "pib":
            return float(self._target_func.pib(img, self._params.ideal_spot_radius)[1])
        return float(res.j)

    def metric_panel(self, img: np.ndarray) -> dict[str, float]:
        """Cross-objective metric panel recorded on EVERY epoch.

        Evaluates all shaping objectives on the same frame with the run's actual
        configuration (weights / target shape / bucket radius), so any two runs
        can be compared on any shared ``m_*`` column regardless of which
        objective actually drove the search. The ``m_`` prefix avoids colliding
        with the objective's own row key (e.g. ``"shape"``).
        """
        p = self._params
        shape_score, energy = shape_metric(
            img,
            p.reference_center,
            p.reference_center,
            p.shape,
            p.size,
            p.aspect_ratio,
            w_uniformity=p.scoring.w_uniformity,
            w_peak=p.scoring.w_peak,
            w_displacement=p.scoring.w_displacement,
            stage=None,
            log_uniformity=p.scoring.log_uniformity,
        )
        pib_term, rms_t = rms_pib_terms(
            img, p.reference_center, p.shape, p.size, p.aspect_ratio
        )
        rmse, _ = rmse_shape_metric(
            img, p.reference_center, p.shape, p.size, p.aspect_ratio
        )
        roi_score, _roi_energy = roi_pib_metric(
            img, p.reference_center, p.shape, p.size, p.aspect_ratio
        )
        frame_sum = float(np.asarray(img, dtype=np.float64).sum())
        ee = float(
            np.clip(
                frame_sum / max(self._panel_init_sum, np.finfo(np.float64).eps),
                0.0,
                1.0,
            )
        )
        # The two bucket-ratio columns need a ``target_func``; the ROI-only
        # objectives may be built without one. Emit NaN so the panel keeps the
        # exact same ten keys (the recorder schema must not change) instead of
        # raising. Search runs always inject a ``target_func``, so this never
        # blanks a real column.
        if self._target_func is None:
            m_pib = m_pib7 = float("nan")
        else:
            m_pib = float(self._target_func.pib(img, self._r_bucket)[1])
            m_pib7 = float(self._target_func.pib(img, p.ideal_spot_radius)[1])
        return {
            "m_shape": float(shape_score),
            "m_energy": float(energy),
            "m_rmse": float(rmse),
            "m_roi_pib": float(roi_score),
            "m_pib": m_pib,
            "m_pib7": m_pib7,
            # Equal-weight (1/3 each) rms_pib score: the objective's own
            # adaptive weights vary per epoch, so the fixed-weight value is
            # the cross-run comparable form.
            "m_rms_pib": float((pib_term + rms_t + ee) / 3.0),
            "m_rms_t": float(rms_t),
            "m_ee": ee,
            "m_brt": float(np.max(img)),
        }

    def adapt_weights(self, res: ObjectiveResult) -> tuple[float, ...]:
        """Adapt the ``rms_pib`` term weights from ``res``'s terms.

        ``res`` is passed explicitly (rather than reading the instance's last
        terms) so a caller can adapt from a *held* result - the SPGD loop must
        adapt from the positive perturbation, which the negative evaluation
        would otherwise overwrite.

        Returns:
            The new ``(w_pib, w_rms, w_ee)``, or ``()`` for other objectives.
        """
        if self._params.objective != "rms_pib":
            return ()
        j, pib_t, rms_t, ee_t = res.terms
        return _update_dynamic_weights(
            self._rms_pib_state,
            pib=float(pib_t),
            rms=float(rms_t),
            ee=float(ee_t),
            j=float(j),
            w_ema_decay=self._params.w_ema_decay,
            w_floor=self._params.w_floor,
            w_temperature=self._params.w_temperature,
        )

    def set_bucket(self, r_bucket: float) -> None:
        """Update the live bucket radius used by ``pib`` and the metric panel."""
        self._r_bucket = float(r_bucket)

    @property
    def guard_violations(self) -> int:
        """Number of evaluations abandoned by the in-ROI energy guard."""
        return self._violations

    @property
    def weights(self) -> tuple[float, float, float]:
        """Current ``(w_pib, w_rms, w_ee)``; all zeros for other objectives."""
        if self._params.objective != "rms_pib":
            return (0.0, 0.0, 0.0)
        return (
            float(self._rms_pib_state.get("w_pib", 1.0 / 3)),
            float(self._rms_pib_state.get("w_rms", 1.0 / 3)),
            float(self._rms_pib_state.get("w_ee", 1.0 / 3)),
        )

    @property
    def terms(self) -> tuple[float, float, float, float]:
        """``(j, pib_term, rms_term, ee_term)`` of the last scored frame."""
        return self._terms
