"""Measurement-protocol primitives: aperture-fixed metrics and repeat statistics.

Why this module exists
----------------------
The 2026-10-28 multishape review (``report/shape_trials/multishape_scale/``)
found three read-out defects that made stage-to-stage comparisons unsound, and
none of them is a physics problem:

1. **CV alone cannot tell "uniform" from "empty".** Core CV drops both when the
   far field flattens *and* when the core loses energy, so a stage that simply
   dimmed the target scores better. Every verdict therefore needs a companion
   relative encircled-energy number, and a relative one suffices: no absolute
   radiometry is required to divide in-target energy by a fixed total.

2. **The denominators drifted between stages.** ``ee_roi`` is computed against
   ``objective.box``, and that box differs between stages (dense stages erode
   the core by ``core_margin`` and sample at half the step). Two stages can
   therefore report different "relative energy" for the *same* phase. A
   cross-stage comparison needs **one aperture frozen for the whole campaign**,
   in CCD sensor pixels, independent of the stage geometry.

3. **A fixed "2 pp" noise threshold is not a measurement.** The same phase
   re-measured at the same exposure differed by 0.0-2.6 pp across the session,
   but that number came from convenience pairings, not from repeats at one
   condition. A protocol-level uncertainty must be estimated from repeated
   measurements of *one* condition and then compared against the observed
   difference.

Everything here is NumPy-only, device-free and importable without torch, so a
protocol verdict can be recomputed from archived npz files on any machine.

The aperture is addressed in **full-frame CCD coordinates**: a metric computed
through :class:`FixedAperture` is comparable across stages, scales and
sessions, which is exactly what the campaign's cross-scale comparison needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "FixedAperture",
    "aperture_metrics",
    "repeat_statistics",
    "verdict_with_uncertainty",
    "judge_pair",
    "compare_repeats",
    "shape_agreement",
    "insufficient_keys",
    "plan_reference_blocks",
    "replay_audit",
    "derive_replay_exposures",
    "SHAPE_AGREEMENT_KEYS",
    "REPLAY_PEAK_GUARD_FRACTION",
    "REPLAY_LADDER_FRACTIONS",
]

#: Metric keys a shape-agreement read-out must report. Kept as a tuple so a
#: caller cannot silently ship a verdict that omits the boundary term.
SHAPE_AGREEMENT_KEYS = ("iou", "boundary_mean_px", "boundary_p95_px", "coverage_fraction")


@dataclass(frozen=True)
class FixedAperture:
    """One target box plus its core, in **full-frame CCD sensor pixels**.

    ``centre_xy`` / ``half_side`` define the target square; ``core_margin_px``
    erodes it into the core used for uniformity; ``zero_xy`` / ``zero_radius``
    cut the zero-order disk out of the core. All fields are sensor pixels, so
    the aperture does not move when a stage changes its camera window step.

    ``exclude_zero`` defaults to True because the campaign's zero-order disk
    overlaps the 200 px target box by 5.61 % (measured, see the planning
    report's H3 erratum): a uniformity number that includes un-modulated zero
    order is not a shaping result.
    """

    centre_xy: tuple[float, float]
    half_side: float
    core_margin_px: float = 0.0
    zero_xy: tuple[float, float] | None = None
    zero_radius: float = 0.0
    exclude_zero: bool = True
    full_scale: float | None = None

    def __post_init__(self) -> None:
        if self.half_side <= 0:
            raise ValueError("half_side must be positive")
        if self.core_margin_px < 0 or self.zero_radius < 0:
            raise ValueError("core_margin_px and zero_radius must be non-negative")
        if self.exclude_zero and self.zero_radius > 0 and self.zero_xy is None:
            raise ValueError("zero_radius needs zero_xy")

    # -- masks -----------------------------------------------------------

    def box_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """The target square on a full-frame image of ``shape`` (H, W).

        The bound is inclusive, so ``half_side=40`` yields a 79x79 sample box on an
        even frame: the geometry is "distance from the centre <= half_side", which
        is what the earlier per-stage ``objective.box`` masks used. Keeping the same
        convention lets a protocol metric be compared against archived stage
        numbers without a half-pixel offset.
        """
        height, width = int(shape[0]), int(shape[1])
        yy, xx = np.indices((height, width))
        cx, cy = float(self.centre_xy[0]), float(self.centre_xy[1])
        # ``half_side`` is a half-extent in pixels: an odd-sized box of
        # side 2*h+1 would be off by one, so the bound is half-open.
        return (np.abs(xx - cx) <= self.half_side) & (np.abs(yy - cy) <= self.half_side)
    def core_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """The box eroded by ``core_margin_px``, minus the zero-order disk."""
        box = self.box_mask(shape)
        if self.core_margin_px > 0:
            from scipy.ndimage import binary_erosion

            cells = int(np.ceil(self.core_margin_px))
            if cells > 0:
                box = binary_erosion(box, iterations=cells, border_value=0)
        if self.exclude_zero and self.zero_radius > 0:
            yy, xx = np.indices(box.shape)
            distance = np.hypot(xx - float(self.zero_xy[0]), yy - float(self.zero_xy[1]))
            box = box & (distance > self.zero_radius)
        return box

    def describe(self) -> dict:
        """The aperture as plain JSON-able data, for a run's config.json."""
        return dict(centre_xy=list(self.centre_xy), half_side=float(self.half_side),
                    core_margin_px=float(self.core_margin_px),
                    zero_xy=None if self.zero_xy is None else list(self.zero_xy),
                    zero_radius=float(self.zero_radius), exclude_zero=bool(self.exclude_zero),
                    full_scale=None if self.full_scale is None else float(self.full_scale))


def _robust_background(frame: np.ndarray) -> float:
    """Frame median: the camera's own bias plus the ambient floor.

    Subtracting it is what makes the CV in
    :func:`ao_shaping.utils.image.beam_metrics.masked_cv` meaningful; a raw
    frame carries symmetric read noise (~half its pixels negative), which alone
    can drive a ratio above 1.
    """
    return float(np.median(frame))


def aperture_metrics(frame: np.ndarray, aperture: FixedAperture) -> dict:
    """Uniformity and relative-energy read-out through one frozen aperture.

    Two numbers, both required before a shaping verdict:

    * ``cv`` -- in-core coefficient of variation (std/mean) after background
      subtraction. Lower is flatter. ``None`` when the core has no signal, which
      is deliberately different from ``0.0``: "no energy" must not read as
      "perfectly uniform".
    * ``ee_rel`` -- in-box energy divided by the whole-frame energy **above the
      same background**. This is a relative quantity: it needs no radiometric
      calibration, but it *is* sensitive to how much light the whole frame
      receives, so it is only comparable between frames taken at the same
      exposure and optical-path state (callers record both).

    The returned dict also carries ``mean``, ``target_peak`` and
    ``zero_excluded_fraction`` so an auditor can see how much of the box the
    zero-order cut removed.
    """
    image = np.asarray(frame, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"expected a monochrome frame, got shape {image.shape}")
    box = aperture.box_mask(image.shape)
    core = aperture.core_mask(image.shape)
    background = _robust_background(image)
    prepared = np.clip(image - background, 0.0, None)

    box_values = prepared[box]
    core_values = prepared[core]
    mean = float(core_values.mean()) if core_values.size else 0.0
    if mean > 0 and core_values.size > 1:
        cv: float | None = float(core_values.std() / mean)
    else:
        cv = None
    total = float(prepared.sum())
    ee_rel = float(box_values.sum() / total) if total > 0 else None
    box_pixels = int(box.sum())
    excluded = float(1.0 - (core.sum() / box_pixels)) if box_pixels else None
    return dict(cv=cv, ee_rel=ee_rel, mean=mean,
                target_peak=float(box_values.max()) if box_values.size else 0.0,
                zero_excluded_fraction=excluded,
                background=background,
                saturated_fraction=_saturated_fraction(image, aperture.full_scale),
                aperture=aperture.describe())


def _saturated_fraction(image: np.ndarray, full_scale: float | None) -> float | None:
    """Share of the target box sitting at (or above) full scale.

    A clipped core reports a *lower* CV than the same field measured below
    saturation, so "CV improved" can be an artefact of the exposure. Every
    read-out therefore carries this number, and
    :func:`verdict_with_uncertainty` refuses a verdict on a clipped candidate.
    ``None`` when the caller did not say which full scale applies: an unknown
    bit depth must not read as "definitely not clipped".
    """
    if full_scale is None:
        return None
    return float(np.mean(image >= float(full_scale)))


def repeat_statistics(values: Sequence[float], *, coverage: float = 0.95) -> dict:
    """Spread of repeated measurements of **one** condition.

    The campaign used a hand-picked ``2 pp`` band as "not evidence". That band
    came from pairing *different* phases across stage boundaries, so it silently
    mixes repeatability with real stage-to-stage change. This function answers
    the narrower, honest question: how much does one unchanged condition move
    when it is measured again?

    Returns the sample standard deviation, the half-width of a two-sided
    ``coverage`` interval using Student's t (so three repeats do not pretend to
    a normal-distribution certainty), the range, and ``n``. With ``n < 2`` the
    spread is undefined and is reported as ``None`` rather than ``0.0``, because
    a single measurement says nothing about its own noise.
    """
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    n = len(finite)
    if n == 0:
        raise ValueError("no finite values")
    if n < 2:
        return dict(n=n, mean=finite[0], std=None, half_width=None, low=None, high=None,
                    range=0.0, coverage=coverage)
    array = np.asarray(finite, dtype=float)
    std = float(array.std(ddof=1))
    # Student's t: kept locally so this leaf module needs no SciPy import.
    t_value = _student_t_975(n - 1) if abs(coverage - 0.95) < 1e-9 else _student_t_cdf_inverse(
        (1.0 + coverage) / 2.0, n - 1)
    half = float(t_value * std / np.sqrt(n))
    return dict(n=n, mean=float(array.mean()), std=std, half_width=half,
                low=float(array.mean() - half), high=float(array.mean() + half),
                range=float(array.max() - array.min()), coverage=coverage)


def verdict_with_uncertainty(baseline: Mapping[str, float | None],
                             candidate: Mapping[str, float | None],
                             *, stats: Mapping[str, Mapping] | None = None,
                             improve: str = "lower",
                             min_effect: float = 0.0) -> dict:
    """Decide whether a candidate differs from its baseline by more than noise.

    A candidate is only ``supported`` when **both** the CV and the relative
    energy move the right way and the CV change clears its measured
    uncertainty: lower CV with lower ``ee_rel`` means the target was emptied,
    not uniformised (the anti-pattern this module exists to catch).

    ``min_effect`` is a domain floor on top of the statistical one -- useful when
    a change is real but operationally irrelevant. ``stats`` maps a metric key
    to a :func:`repeat_statistics` result; without it the comparison reports
    ``insufficient_repeats`` instead of inventing a threshold.

    The energy and clipping gates run **before** the statistics check on purpose:
    a clipped or energy-losing candidate is not a weak result awaiting more
    repeats, it is an invalid one, and re-measuring it would not change that.
    """
    cv_base, cv_cand = baseline.get("cv"), candidate.get("cv")
    ee_base, ee_cand = baseline.get("ee_rel"), candidate.get("ee_rel")
    result: dict = dict(baseline=dict(cv=cv_base, ee_rel=ee_base),
                        candidate=dict(cv=cv_cand, ee_rel=ee_cand))
    if cv_base is None or cv_cand is None:
        result.update(supported=False, reason="missing_cv")
        return result
    delta = float(cv_cand) - float(cv_base)
    if improve == "lower":
        better = delta < 0
    elif improve == "higher":
        better = delta > 0
    else:
        raise ValueError("improve must be 'lower' or 'higher'")
    result["delta_cv"] = delta
    result["delta_ee_rel"] = None if (ee_base is None or ee_cand is None) else float(ee_cand) - float(ee_base)
    if not better:
        result.update(supported=False, reason="wrong_direction")
        return result
    if abs(delta) < float(min_effect):
        result.update(supported=False, reason="below_min_effect")
        return result
    if ee_base is not None and ee_cand is not None and result["delta_ee_rel"] < 0:
        # A flatter core that also lost relative energy is ambiguous: the CV win
        # may be the dimming itself. Report it, never accept it as an improvement.
        result.update(supported=False, reason="energy_lost_with_cv")
        return result
    saturated = candidate.get("saturated_fraction")
    if saturated is not None and float(saturated) > 0.0:
        # Clipping lowers the measured CV on its own: a clipped candidate cannot
        # be called flatter until it is measured again below full scale.
        result.update(supported=False, reason="candidate_clipped",
                      saturated_fraction=float(saturated))
        return result
    noise = None
    if stats and "cv" in stats:
        noise = stats["cv"].get("half_width")
    if noise is None:
        result.update(supported=None, reason="insufficient_repeats", noise_half_width=None)
        return result
    result["noise_half_width"] = float(noise)
    if abs(delta) <= float(noise):
        result.update(supported=False, reason="within_repeatability")
        return result
    result.update(supported=True, reason="cv_improved_beyond_noise")
    return result


# -- Student's t (no SciPy dependency) ------------------------------------

def _student_t_975(dof: int) -> float:
    """t such that P(T <= t) = 0.975 for ``dof`` degrees of freedom."""
    return _student_t_cdf_inverse(0.975, dof)


def _student_t_cdf_inverse(p: float, dof: int) -> float:
    """Bisect the Student-t CDF. Accurate to well under 1e-4 in t.

    A closed form exists only for special cases; bisection over a bracket that
    provably contains the answer is shorter and easier to audit than a series
    expansion, and this runs once per verdict.
    """
    if not 0.5 < p < 1.0:
        raise ValueError("p must lie strictly between 0.5 and 1")
    low, high = 0.0, 1e4
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _student_t_cdf(mid, dof) < p:
            low = mid
        else:
            high = mid
        if high - low < 1e-9:
            break
    return 0.5 * (low + high)


def _student_t_cdf(t: float, dof: int) -> float:
    """CDF via the regularized incomplete beta function, by continued fraction."""
    x = dof / (dof + t * t)
    tail = 0.5 * _betainc(dof / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b), Numerical Recipes ``betacf``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    from math import exp, lgamma, log

    log_front = lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x)
    front = exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, *, iterations: int = 300,
            eps: float = 3e-12) -> float:
    """Continued fraction for the incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def shape_agreement(measured: np.ndarray, target: np.ndarray, *,
                    level: float = 0.5) -> dict:
    """Does the measured field actually *have the target's shape*?

    Core CV is shape-agnostic: a triangle and a star can score the same CV while
    one of them is a blurred disc. This compares the two binary masks directly,
    at a fraction ``level`` of the measured field's own peak so the threshold
    does not need radiometry.

    Reports IoU, mean/p95 distance from the measured boundary to the target
    boundary in samples, and how much of the target box the measured spot
    covers. Boundary distance is the term that separates "right amount of
    energy" from "right shape", so it is reported separately rather than folded
    into IoU.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    measured = np.asarray(measured, dtype=np.float64)
    target = np.asarray(target) > 0.5
    if measured.shape != target.shape:
        raise ValueError(f"measured {measured.shape} and target {target.shape} differ")
    peak = float(measured.max())
    if peak <= 0:
        return dict(iou=0.0, boundary_mean_px=None, boundary_p95_px=None,
                    coverage_fraction=0.0, threshold=0.0)
    mask = measured >= level * peak
    union = mask | target
    inter = mask & target
    iou = float(inter.sum() / union.sum()) if union.any() else 0.0
    coverage = float(inter.sum() / target.sum()) if target.any() else 0.0
    # One-pixel boundary shells: distance from each shell to the other mask.
    inner = np.ones((3, 3), bool)
    measured_shell = mask & ~binary_erosion(mask, structure=inner, border_value=0)
    target_shell = target & ~binary_erosion(target, structure=inner, border_value=0)
    if measured_shell.any() and target_shell.any():
        distances = distance_transform_edt(~target_shell)[measured_shell]
        boundary_mean = float(distances.mean())
        boundary_p95 = float(np.percentile(distances, 95))
    else:
        boundary_mean = boundary_p95 = None
    return dict(iou=iou, boundary_mean_px=boundary_mean,
                boundary_p95_px=boundary_p95, coverage_fraction=coverage,
                threshold=float(level * peak),
                missing_keys=[key for key in SHAPE_AGREEMENT_KEYS
                              if key not in ("iou", "boundary_mean_px", "boundary_p95_px",
                                             "coverage_fraction")])


def judge_pair(baseline: Mapping, candidate: Mapping, *, noise: Mapping | None = None) -> dict:
    """Compare two measured conditions, reporting a tie as a tie.

    :func:`verdict_with_uncertainty` answers "did the candidate *improve*", where
    an unchanged or slightly-worse condition is correctly a rejection. A replay
    audit asks the different question "is there a difference at all", and a strict
    improvement test would label a genuine tie as a rejection -- which reads like a
    regression to anyone scanning the table.

    So an exact or sub-threshold difference is reported ``supported=False`` with
    ``reason="no_difference"``, and everything that does move is decided by the
    improvement verdict: a real *worse* candidate keeps its ``wrong_direction``, and
    a better one is only ``supported`` when it also clears ``noise`` (the
    replicate spread of the unchanged phase). ``noise`` absent -> ``insufficient``.
    """
    base_cv = baseline.get("cv")
    cand_cv = candidate.get("cv")
    if base_cv is None or cand_cv is None:
        return dict(supported=False, reason="missing_cv", delta_cv=None)
    delta = float(cand_cv) - float(base_cv)
    half_width = None if noise is None else noise.get("half_width")
    if abs(delta) == 0.0 or (half_width is not None and abs(delta) <= float(half_width)):
        return dict(supported=False, reason="no_difference", delta_cv=delta,
                    noise_half_width=half_width)
    verdict = verdict_with_uncertainty(
        dict(cv=base_cv, ee_rel=baseline.get("ee_rel")),
        dict(cv=cand_cv, ee_rel=candidate.get("ee_rel"),
             saturated_fraction=candidate.get("saturated_fraction")),
        stats={"cv": noise} if noise is not None else None)
    return verdict


def insufficient_keys(report: Mapping) -> list[str]:
    """Which required shape-agreement keys a report still lacks."""
    return [key for key in SHAPE_AGREEMENT_KEYS if report.get(key) is None]


def compare_repeats(rows: Iterable[Mapping], *, keys: Sequence[str] = ("cv", "ee_rel")) -> dict:
    """Per-key :func:`repeat_statistics` over a list of measurement rows.

    The intended caller is a fixed reference phase captured ``n`` times at one
    exposure: the same condition, so the spread *is* the measurement noise.
    """
    rows = list(rows)
    return {key: repeat_statistics([row.get(key) for row in rows]) for key in keys}


#: A replay frame whose peak is at or above this share of full scale is withheld
#: from any cross-source comparison. Chosen from the campaign's own failures: the
#: 8-bit anchor stage clipped (peak reaching 250/255) while the 12-bit stage of
#: the *same* phase did not, and ``common_replay`` produced rows the review could
#: not use because nothing checked the peak. It is a *gate*, not a calibration:
#: below it, the frame is compared as normal.
REPLAY_PEAK_GUARD_FRACTION = 0.98


#: Where a derived replay ladder puts the *brightest* measured block, as a share
#: of full scale. The middle rung is the comparison exposure; the two outer rungs
#: exist so the comparison can be shown not to depend on exposure. All sit below
#: ``REPLAY_PEAK_GUARD_FRACTION`` by construction, so a derived ladder cannot pick
#: an exposure the audit would then withhold.
REPLAY_LADDER_FRACTIONS = (0.25, 0.5, 0.75)


def derive_replay_exposures(
    observations: Iterable[Sequence[float] | Mapping[str, float]],
    *,
    fractions: Sequence[float] = REPLAY_LADDER_FRACTIONS,
) -> list[float]:
    """Size a common replay ladder from the lane's own measured peak rate.

    A replay ladder has to be fixed **within** a lane -- a common exposure is
    what makes an A-B-B-A comparison mean anything -- but it cannot be fixed
    *across* lanes. The core peak is set by how concentrated the shaped spot is,
    so the same ladder that is comfortable at 1200 px saturates at 200 px: the
    2026-10-29 campaign aborted lane 01 with ``core saturated; raw replay
    retained`` because 125 ms drove a 200 px core to 4095/4095 while the
    1200 px lane was still clean at 225 ms.

    Args:
        observations: One ``(peak, full_scale, exposure_ms)`` per already
            measured block, in that block's own units. Carrying ``full_scale``
            makes the rate comparable across an 8-bit and a 12-bit stage, which
            the same lane mixes by design.
        fractions: Target peak shares of full scale, applied to the brightest
            block.

    Returns:
        One exposure per fraction, in ms, ascending.

    Raises:
        ValueError: If no block offers a positive peak at a positive exposure on
            a positive full scale -- then there is nothing to scale from and any
            number returned would be a guess.
    """
    rate = 0.0
    for observation in observations:
        if isinstance(observation, Mapping):
            peak, full_scale, exposure = (observation["peak"], observation["full_scale"],
                                          observation["exposure_ms"])
        else:
            peak, full_scale, exposure = observation
        if peak is None or exposure is None or full_scale is None:
            continue
        peak, full_scale, exposure = float(peak), float(full_scale), float(exposure)
        if peak <= 0 or full_scale <= 0 or exposure <= 0:
            continue
        rate = max(rate, (peak / full_scale) / exposure)
    if rate <= 0:
        raise ValueError("cannot derive a replay ladder: no block has a positive peak rate")
    return [round(float(fraction) / rate, 3) for fraction in fractions]


def replay_audit(frames: Iterable[Mapping], *, full_scale: float,
                 guard: float = REPLAY_PEAK_GUARD_FRACTION) -> dict:
    """Classify replayed CCD frames before anyone compares them.

    ``frames`` is one mapping per acquired replay frame carrying at least
    ``peak`` (max raw count), plus any identifying keys the caller wants echoed
    (``source``, ``exposure_ms``, ``file``). Returns the guard's decisions and a
    ``usable`` list, because a replay block is the one place where a single
    clipped frame silently becomes a stage comparison.

    A frame is withheld -- never silently kept -- when its peak reaches
    ``guard`` of ``full_scale``. ``None`` peaks are withheld too: "peak not
    recorded" is an unverifiable frame, and the campaign already lost one
    session to an unverifiable-but-recorded capture.
    """
    if full_scale <= 0:
        raise ValueError("full_scale must be positive")
    threshold = float(guard) * float(full_scale)
    usable, withheld = [], []
    for frame in frames:
        peak = frame.get("peak")
        if peak is None:
            withheld.append(dict(frame, reason="peak_not_recorded"))
        elif float(peak) >= threshold:
            withheld.append(dict(frame, reason="at_or_above_guard", threshold=threshold))
        else:
            usable.append(dict(frame))
    return dict(full_scale=float(full_scale), guard=float(guard), threshold=threshold,
                n=len(usable) + len(withheld), usable=usable, withheld=withheld,
                usable_fraction=(len(usable) / (len(usable) + len(withheld))
                                 if (usable or withheld) else None))


def plan_reference_blocks(conditions: Sequence[Mapping], *, block_seconds: float) -> dict:
    """Allocate the session's repeat budget and report what it can resolve.

    The analysis' step ① asks for *a* measurement-consistency protocol, and step
    ⑥ asks that any model change be validated on repeats. Both need the same
    thing: repeated measurements of one unchanged condition, taken *inside* the
    session that the comparison belongs to. This turns that into a concrete
    number instead of a habit.

    ``conditions`` are rows ``dict(name=..., block_seconds=...,
    planned_repeats=...)``; a missing per-condition ``block_seconds`` falls back
    to ``block_seconds``. The returns are the repeat counts actually affordable,
    the per-condition device time at that count, and the minimum CV difference a
    sign-flip test could resolve with ``n`` repeats of two independent
    conditions.

    The resolution figure is deliberately structural, not a promise: with ``n``
    repeats per condition the *smallest* attainable two-sided sign-flip p-value is
    ``2 / 2**n`` (see ``ml/zernike/eval_stats.min_attainable_pvalue``), so ``n=3``
    cannot reach p<0.05 by any arrangement of the data. Printing that next to the
    budget is the point: it prevents "we measured 3 times, so we tested it".
    """
    rows = [dict(row) for row in conditions]
    if not rows:
        raise ValueError("no conditions to plan")
    total_needed = sum(float(row.get("block_seconds", block_seconds)) * int(row.get("planned_repeats", 3))
                       for row in rows)
    plan = []
    for row in rows:
        seconds = float(row.get("block_seconds", block_seconds))
        repeats = max(1, int(row.get("planned_repeats", 3)))
        plan.append(dict(name=row.get("name"), repeats=repeats, block_seconds=seconds,
                         device_seconds=round(seconds * repeats, 1),
                         min_attainable_pvalue=round(_min_attainable_pvalue(repeats), 6)))
    return dict(conditions=plan, total_device_seconds=round(total_needed, 1),
                min_repeats_for_p005=_min_repeats_for_alpha(0.05),
                note=f"p<{_ALPHA} needs n>={_min_repeats_for_alpha(_ALPHA)} repeats per condition; "
                     "fewer repeats support an effect size only")


#: The significance level the repeat budget is planned against.
_ALPHA = 0.05


def _min_attainable_pvalue(repeats: int) -> float:
    """Smallest two-sided sign-flip p-value with ``repeats`` paired observations.

    ``2 / 2**n``: the sign vector and its mirror are the two extreme permutations,
    so no arrangement of the data can beat it. Defined here (rather than imported
    from ``ml/zernike/eval_stats``) because this module must stay importable from a
    device-free tool that has no ``ml`` package on its path; the value is asserted
    against that module's function in the tests.
    """
    return 2.0 / (2 ** int(repeats))


def _min_repeats_for_alpha(alpha: float) -> int:
    """Fewest repeats whose attainable p-value is at most ``alpha``."""
    repeats = 1
    while _min_attainable_pvalue(repeats) > alpha:
        repeats += 1
    return repeats


