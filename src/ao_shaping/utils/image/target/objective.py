"""Shaping objective selection, adaptive ``rms_pib`` weights and the metric panel.

Part of the :mod:`ao_shaping.utils.image.target` package (split by type).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from loguru import logger

from ao_shaping.utils.image.target.metrics import (
    TARGET_SHAPE_CHOICES,
    TargetShape,
    log_gradient_difference_metric,
    rmse_out_metric,
    rmse_shape_metric,
    rms_pib_terms,
    pearson_shape_metric,
    roi_energy_loss,
    roi_pib_metric,
    shape_metric,
    shape_stage_from_energy,
)
from ao_shaping.utils.image.target.patterns import create_target_shape


#: Sentinel added to (max-mode) or subtracted from (min-mode) every score of a
#: frame the in-ROI energy-loss guard abandoned. Large enough that a penalised
#: value can never win a comparison against any attainable measurement, small
#: enough to survive the CSV/JSON round-trip with full float64 precision.
GUARD_PENALTY = 1e3


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
        reference_center: window-local ``(x, y)`` SEED of the target ROI. The
            objective's ROI becomes *live* through
            :meth:`ShapingObjective.set_reference_center` - scoring rides the
            current spot so environmental beam drift is not misread as a
            coefficient-induced shaping loss (the measured 22-px drift on the
            bench is environmental; see ``docs/slm_pib``).
        shape_schedule: coarsen->middle->fine ``shape`` weight staging.
        scoring: ``shape_metric`` weights.
        max_roi_energy_loss: in-ROI energy-loss fraction that abandons an
            evaluation; ``0`` disables the guard.
        ideal_spot_radius: fixed radius of the baseline ``m_pib7`` metric.
        r_bucket: initial (possibly dynamically derived) bucket radius. Mutable
            through :meth:`ShapingObjective.set_bucket`.
        w_outside: outside-target penalty weight of the ``rmse_out`` objective
            (``J = RMSE_norm + w_outside * (1 - in_target_energy)``).
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
    w_outside: float = 1.0
    w_loggrad: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.w_loggrad) or self.w_loggrad < 0.0:
            raise ValueError(f"w_loggrad must be >= 0 and finite, got {self.w_loggrad!r}")


@dataclass(frozen=True)
class ObjectiveResult:
    """One scored camera frame.

    Attributes:
        j: the objective value the search ascends (guard-penalised if the
            frame was abandoned).
        ratio: the secondary quantity the objective reports for logging.
            **Never** guard-penalised - it is the true measured value, kept so
            offline reports can count and exclude guard-firing rows.
        tracking: the value the search compares against as "best" and writes
            back to the SLM on exit (guard-penalised exactly like ``j``).
            For ``pib`` this is the exposure-independent bucket ratio at the
            FIXED ideal radius, which lives at a different bucket radius than
            ``j`` and so cannot be recovered from it; every other objective
            tracks ``j``.
        terms: ``(j, pib_term, rms_term, ee_term)`` of the last ``rms_pib``
            evaluation, ``(0.0, 0.0, 0.0, 0.0)`` for every other objective.
            Frozen so a caller can hold a *positive-perturbation* result across
            the negative-perturbation evaluation without it being overwritten.
    """

    j: float
    ratio: float
    tracking: float
    terms: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    loggrad: float = 0.0


SHAPING_OBJECTIVE_CHOICES = (
    "pib",
    "radiu",
    "avg_radiu",
    "rmse",
    "rmse_out",
    "shape",
    "roi_pib",
    "rms_pib",
    "pearson",
)


#: Objectives that carry the in-ROI energy-loss safety guard.
#:
#: ``pearson`` **must** be here. ``1 - corr`` is computed after mean-centring, so
#: it is invariant to global intensity scale and to any additive DC offset: an
#: all-dark frame and a bright frame with the identical shape score identically.
#: Without the guard the search can drive encircled energy out of the target box
#: while the correlation keeps improving - the same failure mode already recorded
#: for the pure ``-CV`` objective in AGENTS.md (hardware: EE -> 0.002).
GUARDED_OBJECTIVES = (
    "pib",
    "rmse",
    "rmse_out",
    "shape",
    "roi_pib",
    "rms_pib",
    "pearson",
)


#: Objectives that accept an explicit ``target_shape``.
SHAPE_AWARE_OBJECTIVES = (
    "pib",
    "rmse",
    "rmse_out",
    "shape",
    "roi_pib",
    "rms_pib",
    "pearson",
)

#: Objectives for which an explicit shape only selects the ROI, so the objective
#: itself is NOT promoted to ``shape``.
ROI_ONLY_SHAPE_OBJECTIVES = ("roi_pib", "rms_pib", "rmse", "rmse_out", "pearson")

#: Objectives that score against a target ROI and therefore default to
#: ``rectangle`` when no shape is given.
DEFAULT_SHAPE_OBJECTIVES = (
    "shape",
    "roi_pib",
    "rms_pib",
    "rmse",
    "rmse_out",
    "pearson",
)


#: Objectives selectable for the *square* family (slm-gsnet / spgd-square).
#:
#: Square shaping is a separate vocabulary from :data:`SHAPING_OBJECTIVE_CHOICES`
#: because it drives a different search (freeform phase, square target box) and
#: its historical objective is the combined quality score rather than any
#: ROI-PIB/RMSE objective. Every entry is a HIGHER-IS-BETTER score; the sign
#: convention is applied once, in ``square_objective_score``.
SQUARE_OBJECTIVE_CHOICES = ("quality", "pearson")

#: Per-objective whitelist of accepted ``target_shape`` values; ``None`` means
#: the objective does not accept a literal shape.
#:
#: Derived from :data:`SHAPING_OBJECTIVE_CHOICES` x :data:`SHAPE_AWARE_OBJECTIVES`
#: so the keys are guaranteed to stay equal to the objective vocabulary — a
#: hand-written dict had already drifted once (it omitted ``rmse_out``). The
#: value is a plain tuple, so narrow vocabularies (e.g. a 3-shape benchmark
#: subset) remain expressible without a second mechanism.
#:
#: This keeps the exact invariant that :meth:`ObjectiveSpec.resolve` enforces:
#: ``OBJECTIVE_ALLOWED_SHAPES[name] is None`` if and only if ``resolve`` raises
#: for ``(name, <any shape>)``. Note this is *acceptance*, not *survival* —
#: ``pib`` accepts any shape but promotes itself to ``shape``
#: (:data:`ROI_ONLY_SHAPE_OBJECTIVES` is what decides that), so a literal shape
#: is never the ``pib`` objective's own shape.
OBJECTIVE_ALLOWED_SHAPES: dict[str, tuple[str, ...] | None] = {
    name: (tuple(TARGET_SHAPE_CHOICES) if name in SHAPE_AWARE_OBJECTIVES else None)
    for name in SHAPING_OBJECTIVE_CHOICES
}


def _quoted_choices(choices: tuple[str, ...]) -> str:
    """Render ``choices`` as ``'a', 'b', 'c'`` for error messages.

    The message is derived from the tuple itself so a newly added objective can
    never go missing from the text (the hand-written lists drifted once already,
    hiding ``rmse_out`` from both ``ValueError`` messages).
    """
    return ", ".join(repr(str(c)) for c in choices)


@dataclass(frozen=True)
class ObjectiveSpec:
    """A validated ``(objective, target_shape)`` pair with the shape nested.

    This is the single authority for the objective/target-shape pairing: the
    CLI layer, the runners and the shaping optimizer all resolve through
    :meth:`resolve` instead of re-implementing the rules.

    ``shape`` is ``None`` for the bucket/radius objectives (``pib`` / ``radiu`` /
    ``avg_radiu``) and a concrete :class:`TargetShape` for the ROI family.

    Resolution rules, applied in this order:

    1. ``target_shape``, when given, must be a known shape and the objective
       must be in :data:`SHAPE_AWARE_OBJECTIVES`.
    2. ``pib`` + a shape is promoted to the dynamic-ROI ``shape`` objective; the
       remaining shape-aware objectives (:data:`ROI_ONLY_SHAPE_OBJECTIVES`) keep
       their identity because the shape only selects the ROI there.
    3. The ROI family (:data:`DEFAULT_SHAPE_OBJECTIVES`) defaults to
       ``rectangle`` when no shape was supplied.
    4. The objective must finally be one of :data:`SHAPING_OBJECTIVE_CHOICES`.
    """

    name: str
    shape: TargetShape | None

    @classmethod
    def resolve(cls, objective: str, target_shape: str | None = None) -> ObjectiveSpec:
        """Resolve a user-supplied objective / shape pair.

        Args:
            objective: Objective name; case-insensitive.
            target_shape: Target shape name; case-insensitive, ``None`` for the
                bucket/radius objectives.

        Returns:
            The resolved spec.

        Raises:
            ValueError: if the shape is unknown, the objective cannot take a
                shape, or the objective is not supported.
        """
        name = str(objective).lower()
        shape: str | None = None
        if target_shape is not None:
            shape = str(target_shape).lower()
            if shape not in TARGET_SHAPE_CHOICES:
                raise ValueError(
                    f"target_shape must be one of {TARGET_SHAPE_CHOICES}, got {shape!r}"
                )
            if name not in SHAPE_AWARE_OBJECTIVES:
                raise ValueError(
                    "target_shape can only be used with objective="
                    f"{_quoted_choices(SHAPE_AWARE_OBJECTIVES)}, "
                    f"got objective={name!r} with target_shape={shape!r}"
                )
        if shape is not None and name not in ROI_ONLY_SHAPE_OBJECTIVES:
            # Supplying target_shape implies the dynamic-ROI shaping objective,
            # except for roi_pib/rms_pib/rmse/rmse_out where the shape only
            # selects the ROI.
            name = "shape"
        if name in DEFAULT_SHAPE_OBJECTIVES and shape is None:
            shape = "rectangle"
        if name not in SHAPING_OBJECTIVE_CHOICES:
            raise ValueError(
                f"objective must be one of "
                f"{_quoted_choices(SHAPING_OBJECTIVE_CHOICES)}, got {name!r}"
            )
        return cls(
            name=name,
            shape=cast(TargetShape, shape) if shape is not None else None,
        )



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
    current value. The target ROI centre is *live* too: the search re-locates
    it onto the current spot through :meth:`set_reference_center` before each
    evaluation, so environmental beam drift does not masquerade as a
    coefficient-induced loss (the energy guard keeps its armed baseline).
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
        # Live ROI centre, seeded from the (frozen) params. ``set_reference_center``
        # re-locates it onto the current spot before every evaluation.
        self._reference_center: tuple[float, float] = (
            float(params.reference_center[0]),
            float(params.reference_center[1]),
        )
        # Baseline window energy from the flat-phase capture: "no energy lost"
        # means sum(img) stays at this level. Both the ``rms_pib`` term and the
        # panel's ``m_ee`` are fractions of it.
        init_sum = float(np.sum(np.asarray(init_img, dtype=np.float64)))
        self._init_energy = init_sum
        self._panel_init_sum = init_sum
        self._terms: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._shape_state: dict[str, float] = {"best_energy": 0.0}
        self._violations = 0
        # Structural reference for the ``w_loggrad`` modifier: the flat-phase
        # capture, i.e. the bench's OWN natural spot profile. Referencing the
        # achieved profile rather than the requested shape is deliberate and
        # measured — a target-SHAPE reference does not work here. ``log`` of a
        # hard-edged shape (create_target_shape returns a 2-valued disc) is a
        # step function, so a gradient comparison against it scores a clean
        # Gaussian spot 0.438 and speckle 0.425, i.e. INVERTED. Against a
        # smooth profile of the achievable width the ordering is correct (clean
        # 0.000, speckle 0.084, off-size 0.279). This term therefore answers
        # "did this candidate gain structure finer than the natural profile",
        # which is exactly what the energy-in-bucket objectives are blind to.
        # Shape itself remains ``pib`` / ``shape`` / ``roi_pib``'s job.
        # Only retained when the feature is enabled, so the default path pays
        # nothing in memory.
        self._loggrad_reference: np.ndarray | None = None
        if params.w_loggrad > 0.0:
            ref = np.asarray(init_img, dtype=np.float32)
            self._loggrad_reference = np.where(np.isfinite(ref), ref, 0.0)

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
            ref = self._roi_energy(init_img)
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

    def _roi_energy(self, frame: np.ndarray) -> float:
        """Light fraction inside the target ROI - the guard's observable.

        The ROI location is the *live* ``self._reference_center`` (moved onto
        the current spot before every evaluation), so a benign environmental
        beam drift no longer collapses the ROI energy and false-fires the
        guard. A coefficient-driven loss (light scattered out of the ROI or
        pumped into a dark halo) still drops the in-ROI fraction and abandons
        the evaluation.
        """
        p = self._params
        return float(
            roi_pib_metric(
                frame, self._reference_center, p.shape, p.size, p.aspect_ratio
            )[0]
        )

    def set_reference_center(self, center: tuple[float, float]) -> None:
        """Move the target ROI onto ``center`` (window-local ``(x, y)``).

        Called by the search loop before each evaluation once the spot has
        been re-located on the freshly captured frame (``zero_order_center``).
        ``params.reference_center`` stays the *seed*; the energy guard keeps
        the baseline it was armed with, only the ROI location becomes live.
        """
        self._reference_center = (float(center[0]), float(center[1]))

    @property
    def _sign(self) -> float:
        """Polarity of the reported objective: ``-1`` when higher is better.

        Single source of truth for the penalty direction. ``_raw_base`` returns
        NATIVE polarity (higher-is-better entries); this sign is what converts a
        penalty into "uniformly bad" before it is handed to an optimiser that
        minimises.
        """
        return -1.0 if self._params.mode == "max" else 1.0

    def _get_loggrad_term(self, img: np.ndarray) -> float | None:
        """Structure term against the bench's natural spot profile, or ``None``.

        ``None`` means "skip" and must leave the objective untouched. It is
        returned when the feature is off, when no reference was captured, or
        when the candidate frame no longer matches the reference geometry (the
        window can be re-armed mid-run, e.g. after ``reset_window``) - a shape
        mismatch must degrade gracefully rather than raise inside the loop.
        """
        if self._params.w_loggrad <= 0.0:
            return 0.0
        reference = self._loggrad_reference
        if reference is None:
            return None
        frame = np.asarray(img, dtype=np.float32)
        if frame.shape != reference.shape:
            return None
        try:
            lg = float(log_gradient_difference_metric(frame, reference))
        except ValueError:
            return None
        if not np.isfinite(lg):
            return 0.0
        return lg

    def _raw_base(self, img: np.ndarray) -> tuple[float, float, float]:
        """Score one frame without the safety guard (base terms only)."""
        p = self._params
        objective = p.objective

        if objective == "pib":
            # Maximize PIB. Same convention as pib.py.
            pib, pib_ratio = self._target_func.pib(img, self._r_bucket)
            # The headline lives at the FIXED ideal radius, not the live bucket:
            # the bucket shrinks mid-run, so a value read at ``_r_bucket`` would
            # not be comparable across epochs.
            tracking = float(
                self._target_func.pib(img, p.ideal_spot_radius)[1]
            )
            return float(pib), float(pib_ratio), tracking

        if objective == "roi_pib":
            # Maximise the brightness inside the TARGET-SHAPED ROI: the fraction
            # of the light landing in the target rectangle (exposure-invariant
            # ratio, no uniformity/peak/drift penalties).
            score, energy = roi_pib_metric(
                img, self._reference_center, p.shape, p.size, p.aspect_ratio
            )
            return float(score), float(energy), float(score)

        if objective == "rms_pib":
            # Combined PIB + in-ROI RMS + energy-conservation objective with
            # adaptively-weighted terms:
            # J = w_pib(t)*pib_term + w_rms(t)*rms_term + w_ee(t)*ee_term.
            # The weights adapt so the term that improves J more gets the higher
            # weight (see _update_dynamic_weights). The target ROI rides the
            # measured spot (live ``self._reference_center``). ``ee_term`` =
            # fraction of the baseline (flat-phase) window energy still inside
            # the window, so a search that diffracts/scatters light out of the
            # window (or pumps it to a dark halo) is penalised even though
            # pib/rms are exposure-invariant ratios (energy-encircled constraint;
            # see AGENTS.md anti-pattern).
            w_pib = float(self._rms_pib_state["w_pib"])
            w_rms = float(self._rms_pib_state["w_rms"])
            w_ee = float(self._rms_pib_state["w_ee"])
            pib_term, rms_term = rms_pib_terms(
                img, self._reference_center, p.shape, p.size, p.aspect_ratio
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
            return float(j), float(pib_term), float(j)

        if objective == "shape":
            # The target ROI rides the measured spot (live
            # ``self._reference_center``), so environmental beam drift is not
            # penalised as if the coefficients had moved it (displacement weight
            # = 0 for the same reason: the energy term already captures drift).
            stage = (
                shape_stage_from_energy(self._shape_state["best_energy"])
                if p.shape_schedule
                else None
            )
            score, energy = shape_metric(
                img,
                self._reference_center,
                self._reference_center,
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
            return float(score), float(energy), float(score)

        if objective == "rmse":
            # Minimise the RMSE between the frame and the uniform-intensity
            # target, both normalised to unit sum (exposure / laser-drift
            # invariant). The target ROI rides the current spot (live centre).
            rmse, energy = rmse_shape_metric(
                img, self._reference_center, p.shape, p.size, p.aspect_ratio
            )
            return float(rmse), float(energy), float(rmse)

        if objective == "rmse_out":
            # Minimise normalised RMSE with an explicit outside-target penalty:
            # J = RMSE_norm + w_outside * (1 - in_target_energy).
            j, energy = rmse_out_metric(
                img,
                self._reference_center,
                p.shape,
                p.size,
                p.aspect_ratio,
                w_outside=p.w_outside,
            )
            return float(j), float(energy), float(j)

        if objective == "pearson":
            # Minimise 1 - Pearson between the full frame and the uniform
            # target ROI (the FourierGSNet ``shaping_loss``, ported to NumPy).
            # Correlation is taken after mean-centring, so this is invariant to
            # global scale / DC offset and therefore blind to absolute energy -
            # which is exactly why ``pearson`` is in ``GUARDED_OBJECTIVES``.
            loss, energy = pearson_shape_metric(
                img, self._reference_center, p.shape, p.size, p.aspect_ratio
            )
            return float(loss), float(energy), float(loss)

        if objective == "radiu":
            r_99 = float(self._target_func.radius(img, energy=0.99))
            return r_99, 0.0, r_99

        # avg_radiu: maximize average radius.
        avg_r, avg_ratio = self._target_func.avg_radius(img, moment=1.0)
        return float(avg_r), float(avg_ratio), float(avg_r)

    def raw(self, img: np.ndarray) -> tuple[float, float, float]:
        """Score one frame: base terms plus the opt-in log-gradient modifier.

        ``_raw_base`` returns native polarity, so a min-max penalty must be
        combined as ``j + sign * weight * lg`` - SUBTRACTED in ``max`` mode,
        ADDED in ``min`` mode. Adding it unconditionally would make every
        max-mode objective (``pib`` / ``shape`` / ``roi_pib`` / ``rms_pib`` /
        ``pearson``) score *better* when the shape gets worse.

        With ``w_loggrad == 0`` this is exactly ``_raw_base``, byte for byte.
        """
        j, ratio, tracking = self._raw_base(img)
        lg = self._get_loggrad_term(img)
        if lg is None or self._params.w_loggrad <= 0.0:
            return float(j), float(ratio), float(tracking)
        return (
            float(j + self._sign * self._params.w_loggrad * lg),
            float(ratio),
            float(tracking),
        )

    def __call__(self, img: np.ndarray) -> ObjectiveResult:
        """Score one frame, applying the in-ROI energy-loss safety guard.

        Returns a strongly penalised ``j`` (and the true ratio for logging) when
        the in-ROI energy loss exceeds ``max_roi_energy_loss`` - the evaluation
        is thereby abandoned.

        The penalty is applied to ``tracking`` **as well as** ``j``. It used to
        be applied only to ``j``: ``tracking`` was recomputed downstream straight
        from ``img``, so an abandoned frame still looked like a great candidate
        to every ``best_*`` comparison and the exit path could write a
        guard-forbidden phase back to the SLM.

        This calls ``_raw_base`` rather than ``raw`` on purpose: the modifier is
        skipped entirely on the guard-fire path, so ``GUARD_PENALTY`` dominates
        for ANY ``w_loggrad`` instead of only while ``w_loggrad < 1000``.
        """
        j, ratio, tracking = self._raw_base(img)
        lg = self._get_loggrad_term(img)
        modifier = 0.0
        if lg is not None and self._params.w_loggrad > 0.0:
            modifier = self._sign * self._params.w_loggrad * lg
        if self._guard_ref_energy is None:
            return ObjectiveResult(
                float(j + modifier), float(ratio), float(tracking),
                self._terms, float(lg or 0.0),
            )

        loss = roi_energy_loss(self._guard_ref_energy, self._roi_energy(img))
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
            return ObjectiveResult(
                float(j + self._sign * GUARD_PENALTY),
                float(ratio),
                float(tracking + self._sign * GUARD_PENALTY),
                self._terms,
                0.0,
            )
        return ObjectiveResult(
            float(j + modifier), float(ratio), float(tracking),
            self._terms, float(lg or 0.0),
        )

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
            self._reference_center,
            self._reference_center,
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
            img, self._reference_center, p.shape, p.size, p.aspect_ratio
        )
        rmse, _ = rmse_shape_metric(
            img, self._reference_center, p.shape, p.size, p.aspect_ratio
        )
        roi_score, _roi_energy = roi_pib_metric(
            img, self._reference_center, p.shape, p.size, p.aspect_ratio
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
