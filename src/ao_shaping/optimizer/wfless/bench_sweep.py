"""Sweep primitives: records and the per-point derivations behind the geometry fit.

:class:`SweepRecord` is the on-disk record written by ``--stage sweep``;
:class:`SweepCalibrationError` carries the dropped-mode evidence; the helpers
here turn raw spot widths, centroids and phase-correlation shifts into tilt
slopes, a ramp focal scale and per-mode width groups.  No forward-model passes
happen in this module -- :mod:`ao_shaping.optimizer.wfless.bench_sweep_fit`
consumes these primitives to solve the geometry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from loguru import logger

from ao_shaping.optimizer.wfless.bench_geometry_common import (
    _MIN_SWEEP_POINTS,
    _has_both_signs,
)
from ao_shaping.utils.math.fitting import fit_linear_slope as _linear_slope

# Below this centroid/peak ratio the focal plane is a ring pattern, not one lobe.
_DEFOCUS_MIN_HOLLOWNESS = 0.60
# Minimum phase-correlation peak height for a tilt shift to be trusted over the
# spot centroid.
_MIN_SHIFT_CORRELATION = 0.30


@dataclass
class SweepRecord:
    """One point of a tilt / defocus calibration sweep.

    Attributes:
        label: Identifier, echoed into error messages.
        mode: ``"flat"``, ``"tilt"`` or ``"defocus"``.
        coefficient: Applied Zernike coefficient in radians.
        axis: ``"x"`` / ``"y"`` for tilt, empty otherwise.
        fwhm_px: Measured spot FWHM in camera pixels.
        centroid_x: Measured spot centroid x, camera pixels.
        centroid_y: Measured spot centroid y, camera pixels.
        peak: Measured peak, for a saturation cross-check.
        hollowness: Intensity at the centroid divided by the peak. ~1 for a
            single spot lobe, well below 1 once the focal plane breaks into rings.
            Used to keep ring points out of the defocus fit, which width cannot
            do: a -4 rad ring measured *narrower* than the -2.5 rad lobe, so a
            width threshold selects exactly the wrong points.
        shift_x: Sub-pixel pattern shift in camera px, from phase correlation
            against the flat reference. **Preferred over the centroid for tilt**:
            a tilt large enough to give signal deforms the spot, and at +/-1.5 rad
            the centroid readings jumped +/-8 px between repeats of the same
            phase, halving the fitted slope. ``nan`` when not measured.
        shift_y: As ``shift_x``, along camera y.
        shift_corr: Height of the phase-correlation peak (1.0 = perfect match).
    """

    label: str
    mode: str
    coefficient: float
    axis: str
    fwhm_px: float
    centroid_x: float
    centroid_y: float
    peak: float
    hollowness: float = 1.0
    shift_x: float = float("nan")
    shift_y: float = float("nan")
    shift_corr: float = float("nan")


class SweepCalibrationError(ValueError):
    """A sweep calibration could not be completed, with actionable context.

    The sweep route fails in two very different ways that used to look the
    same: the bench does not respond at all, and the bench does respond but
    the *measurement* is noise. The second one is the common one -- a campaign
    whose exposure window put the peak at 24-213 counts produced width series
    that bounced around at random -- and it used to end the run with a bare
    ``KeyError: 'defocus'``, which reads like a crash rather than what it is: a
    signal-to-noise verdict. This exception carries the evidence (which modes
    were dropped, why, and the raw widths) plus the two concrete next steps.

    Attributes:
        dropped: Swept modes whose width response was unusable, mapped to the
            reason (curvature <= 0, offset pinned to the scan edge, too few
            points, ...).
        usable_modes: The modes that did produce a usable width response.
        widths: The measured FWHM series per swept mode, in sweep order.
    """

    def __init__(
        self,
        message: str,
        *,
        dropped: Mapping[str, str] | None = None,
        usable_modes: Sequence[str] = (),
        widths: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        super().__init__(message)
        self.dropped: dict[str, str] = dict(dropped or {})
        self.usable_modes: tuple[str, ...] = tuple(usable_modes)
        self.widths: dict[str, list[float]] = {
            mode: [float(v) for v in values] for mode, values in (widths or {}).items()
        }


def _publish_dropped(out: dict[str, str] | None, dropped: Mapping[str, str]) -> None:
    """Copy the dropped-mode record into the caller's dict, in place.

    The dict is an out-parameter: the caller keeps ownership of it, so the
    record can be logged, serialised or written next to the geometry without
    the calibration having to know about any of that.
    """
    if out is None:
        return
    out.clear()
    out.update(dropped)


def _format_width_series(widths: Sequence[float]) -> str:
    """Format a measured width series for an error message."""
    return ", ".join(f"{v:.1f}" for v in widths)


def _dropped_summary(reason: str) -> str:
    """Compact, parse-safe summary of a dropped-mode reason.

    ``scripts/slm/generate_bench_probe_report.py`` pulls numbers out of
    ``calibration_notes`` with narrow regexes, so the summary must not put a
    number next to " rad" or in front of a ";" -- those would be read back as
    a bench offset or a curvature. The full reason stays in the ``dropped``
    record and in the error message.
    """
    if "no model response" in reason:
        return "no model response"
    if "curvature" in reason:
        return "degenerate width response (curvature <= 0)"
    if "scan edge" in reason:
        return "degenerate width response (offset at the scan edge)"
    if "sweep point" in reason or "finite positive FWHM" in reason:
        return "too few usable points"
    return "no usable width response"


def _sweep_dropped_report(
    dropped: Mapping[str, str],
    usable: Sequence[str],
    widths: Mapping[str, Sequence[float]],
) -> str:
    """The shared diagnostic block: what was dropped, why, and what survived."""
    lines: list[str] = []
    if dropped:
        lines.append("被丢弃的模式 (dropped modes):")
        for mode, reason in sorted(dropped.items()):
            series = widths.get(mode, ())
            lines.append(
                f"  - {mode}: {reason}; 原始宽度序列 (raw FWHM px) = "
                f"[{_format_width_series(series)}]"
            )
    else:
        lines.append("被丢弃的模式 (dropped modes): 无 (none)")
    lines.append(
        "仍可用的模式 (usable modes): " + (", ".join(usable) if usable else "无 (none)")
    )
    return "\n".join(lines)


def _defocus_degenerate_message(
    dropped: Mapping[str, str],
    usable: Sequence[str],
    widths: Mapping[str, Sequence[float]],
) -> str:
    """The diagnostic message for "the defocus width response is degenerate".

    Bilingual on purpose: the operator reads the Chinese half, the log/report
    greps keep the English terms, and the two concrete next steps are spelled
    out rather than left as a judgement call.
    """
    reason = dropped.get(
        "defocus",
        "离焦组的宽度响应不可用 (the defocus group produced no usable response)",
    )
    return (
        "离焦（Noll 4）的宽度响应退化 ⇒ 几何无法标定: the defocus (Noll 4) width "
        "response is degenerate, so the bench geometry cannot be calibrated -- "
        "the waist and the focal scale both come from that response.\n"
        f"  defocus: {reason}\n"
        f"{_sweep_dropped_report(dropped, usable, widths)}\n"
        "下一步 (next steps):\n"
        "  (1) 提高信噪比 (improve the SNR): 提高曝光 (exposure), 并对每一档做多帧"
        "平均再取均值。判据: 宽度的相对分辨率要 ≲ 25%, 因为 w² 拟合会把宽度的相对"
        "误差平方放大 (25% 的宽度分辨率 ⇒ w² 约 50% 的不确定度 ⇒ 曲率就拟不出来了)。\n"
        "  (2) 换用更稳健的宽度估计 (robust width estimator): 用二阶矩半径 "
        "(second-moment radius) 代替 FWHM —— FWHM 由半高位置决定, 低信噪比下会被"
        "噪声主导; 二阶矩半径用整幅光强的二阶矩, 对噪声更稳。"
    )


def _tilt_axis_slopes(tilts: Sequence[SweepRecord]) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """Per-axis tilt slope/intercept/source: phase-correlation shift preferred,
    centroid only when no usable correlation (see the sweep report rationale below).
    """
    # --- tilt -> far-field scale, per axis -----------------------------------
    # Prefer the phase-correlation shift over the centroid: a tilt big enough to
    # give a usable signal also deforms the spot, and the centroid of a
    # split/multi-lobed spot jumps by several pixels between repeats of the same
    # phase. Measured here: +/-1.5 rad gave centroid readings scattered over
    # 1020.8..1035.9 and a slope of 1.72 cam px/rad, against 3.3-3.6 from the
    # smaller single-lobed range -- a 2x error in the number the whole
    # calibration rests on.
    slopes: dict[str, float] = {}
    intercepts: dict[str, float] = {}
    sources: dict[str, str] = {}
    for axis in ("x", "y"):
        pts = [r for r in tilts if r.axis == axis]
        if len(pts) < _MIN_SWEEP_POINTS:
            continue
        coeffs = np.array([r.coefficient for r in pts], dtype=np.float64)
        use_shift = all(
            np.isfinite(r.shift_x) and np.isfinite(r.shift_y) for r in pts
        ) and all(r.shift_corr >= _MIN_SHIFT_CORRELATION for r in pts)
        if use_shift:
            sx, ix = _linear_slope(
                coeffs, np.array([r.shift_x for r in pts], dtype=np.float64)
            )
            sy, iy = _linear_slope(
                coeffs, np.array([r.shift_y for r in pts], dtype=np.float64)
            )
            sources[axis] = "phase-correlation"
        else:
            sx, ix = _linear_slope(
                coeffs, np.array([r.centroid_y for r in pts], dtype=np.float64)
            )
            sy, iy = _linear_slope(
                coeffs, np.array([r.centroid_x for r in pts], dtype=np.float64)
            )
            sources[axis] = "centroid (no usable phase correlation)"
        if np.isfinite(sx) and abs(sx) >= abs(sy):
            slopes[axis], intercepts[axis] = sx, ix
        else:
            slopes[axis], intercepts[axis] = sy, iy
    return slopes, intercepts, sources

def _ramp_focal_scale(ramps: Sequence[SweepRecord]) -> tuple[float, str]:
    """Focal-scale route from a phase-ramp sweep: the authoritative camera-px-per-model-px
    source; returns (nan, "") when no usable ramp sweep exists.
    """
    # --- focal scale ---------------------------------------------------------
    # Preferred source: a **phase ramp** sweep. A 2*pi ramp over P panel px moves
    # the spot by `S/P` camera px, measured cleanly and repeatably (2-3 repeats
    # agreeing to 0.2 px, cleanly 1/P, consistent with theory). A Zernike tilt is
    # equivalent to a ramp of period pi*R/c, so the two must agree -- but on this
    # bench they did not: the Zernike route returned 1.63 cam px/rad where the
    # ramp says 5.25, because a Zernike tilt large enough for signal deforms the
    # spot and the centroid stops tracking the peak. The ramp has no such
    # failure mode, so it wins.
    #
    # With k_tilt = S/(pi*R) and a model semi-axis of `a` model px covering the
    # same physical R = a*model_to_panel panel px,
    #     camera_px_per_model_px = k_tilt * pi * a / P_far_field
    #                              = S / (model_to_panel * far_field_size)
    ramp_scale = float("nan")
    ramp_axis = ""
    if len(ramps) >= _MIN_SWEEP_POINTS:
        for ax in ("x", "y"):
            pts = [r for r in ramps if r.axis == ax]
            if len(pts) < _MIN_SWEEP_POINTS:
                continue
            periods = np.array([r.coefficient for r in pts], dtype=np.float64)
            if np.any(periods <= 0):
                continue
            inv = 1.0 / periods
            sx, _ = _linear_slope(
                inv, np.array([r.centroid_x for r in pts], dtype=np.float64)
            )
            sy, _ = _linear_slope(
                inv, np.array([r.centroid_y for r in pts], dtype=np.float64)
            )
            s = sx if abs(sx) >= abs(sy) else sy
            if abs(s) > abs(ramp_scale) or not np.isfinite(ramp_scale):
                ramp_scale, ramp_axis = float(s), ax
    return ramp_scale, ramp_axis

def _fov_ratio_cross_check(camera_px_per_model_px: float, focal_source: str, *, a: int, far_field_size: int, aperture_m: float | None, camera_pixel_m: float | None, focal_length_m: float | None, wavelength_m: float) -> float:
    """Independent focal-scale cross-check from the model FOV over the camera FOV;
    logs the disagreement band (>25%) instead of picking a side.
    """
    # Independent cross-check, derived a completely different way: the model's
    # far-field field of view against the camera's.
    #
    #     p_fft = lambda*f / (P_far_field * model_pitch)   [m per model px]
    #     model FOV  = far_field_size * p_fft
    #     camera FOV = camera_px * camera_pixel
    #     -> camera_px_per_model_px = FOV_model / FOV_camera
    #
    # Disagreement with the ramp route beyond ~25% means one of the two is wrong,
    # and the honest response is to say so rather than pick one.
    fov_ratio = float("nan")
    if aperture_m and camera_pixel_m and focal_length_m and a > 0:
        # One model pixel spans aperture_m / a metres, since `a` model pixels
        # cover the illuminated radius. (Multiplying by model_to_panel here was a
        # bug: it double-counts the panel->model scale and inflated the pitch 3.5x,
        # which is most of why the two routes looked 124% apart.)
        model_pitch_m = aperture_m / a
        p_fft = wavelength_m * focal_length_m / (far_field_size * model_pitch_m)
        fov_ratio = (far_field_size * p_fft) / (2592.0 * camera_pixel_m)
        if np.isfinite(fov_ratio) and fov_ratio > 0:
            rel = abs(camera_px_per_model_px / fov_ratio - 1.0)
            logger.info(
                "focal scale {:.4f} cam px per model px (from {}) vs {:.4f} from "
                "the independent field-of-view ratio -- {:.0%} apart",
                camera_px_per_model_px,
                focal_source,
                fov_ratio,
                rel,
            )
            if rel > 0.25:
                logger.warning(
                    "the two focal-scale routes disagree by {:.0%} ({} vs {:.4f} "
                    "from the FOV ratio). At least one assumption is wrong -- most "
                    "likely the illuminated aperture radius, which is what both "
                    "routes scale with. Re-measure it with "
                    "`python -m ao_shaping.tools.slm.panel.beam_extent` before "
                    "trusting either number.",
                    rel,
                    focal_source,
                    fov_ratio,
                )
    return fov_ratio

def _tilt_physical_cross_check(k_tilt: float, camera_px_per_model_px: float, *, aperture_m: float | None, camera_pixel_m: float | None, focal_length_m: float | None, wavelength_m: float) -> tuple[float, float]:
    """Physical sanity band for the tilt slope against f*lambda/(pi*R*camera_pixel);
    out-of-band values are labelled untrustworthy, not hidden.
    """
    # Physical cross-check on the Zernike tilt slope, kept as a diagnostic: the
    # ramp route is authoritative, but if a Zernike sweep is present and its slope
    # is nowhere near the ramp's equivalent, say so rather than silently ignoring
    # it.

    # Physical cross-check on the tilt measurement. A unit-RMS Zernike tilt of
    # coefficient c over an aperture of radius R imposes a phase gradient 2c/R,
    # which deflects by (1/k)(2c/R) = c*lambda/(pi*R) and shifts the focal spot by
    # f times that. So the predicted slope is
    #     k_pred = f * lambda / (pi * R * camera_pixel)   [camera px per rad]
    # This is what makes the measurement falsifiable. The wide-angle runs measured
    # 1.63-1.74 cam px/rad where 5.34 was predicted -- a ratio of 0.31, which
    # back-solves to an effective aperture of 11 mm, larger than the panel's short
    # side, so those runs were corrupted (the spot had deformed and the centroid
    # stopped tracking the peak) rather than revealing a huge beam.
    if aperture_m and camera_pixel_m and focal_length_m:
        k_pred = focal_length_m * wavelength_m / (np.pi * aperture_m * camera_pixel_m)
        ratio = k_tilt / k_pred if k_pred > 0 else float("nan")
        tilt_ratio = float(ratio)
        implied_aperture_mm = (
            focal_length_m * wavelength_m / (np.pi * camera_pixel_m * k_tilt) * 1e3
            if k_tilt > 0
            else float("nan")
        )
        if not (0.4 <= ratio <= 1.6):
            logger.warning(
                "tilt slope {:.3f} cam px/rad is {:.2f}x the {:.3f} predicted for a "
                "{:.1f} mm illuminated aperture -- OUTSIDE the credible band "
                "0.4-1.6, so the focal scale derived from it ({:.4f} cam px per "
                "model px) is NOT trustworthy. It back-solves to an effective "
                "aperture of {:.1f} mm; the panel's short side is 9.6 mm, so a "
                "measurement implying more than that is corrupted, not a discovery. "
                "Re-run the tilt sweep with smaller coefficients.",
                k_tilt,
                ratio,
                k_pred,
                1e3 * aperture_m,
                camera_px_per_model_px,
                implied_aperture_mm,
            )
    else:
        tilt_ratio = float("nan")
        implied_aperture_mm = float("nan")
    return tilt_ratio, implied_aperture_mm

def _sweep_width_groups(records: Sequence[SweepRecord]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, str], dict[str, list[float]]]:
    """Per-mode width series: min-point / finite-FWHM / single-lobe (hollowness) and
    per-side monotonicity filters; unusable modes are recorded in dropped_modes.
    """
    # --- width responses: defocus + astigmatism, fitted jointly ---------------
    # Defocus alone cannot identify the waist on this bench: its width response
    # stayed 16% asymmetric after the bench's own defocus offset was removed,
    # which is astigmatism/coma the defocus sweep does not contain. Sweeping both
    # quadratic modes and fitting them together separates the two contributions.
    groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    # Modes that were swept but could not contribute a width response, with the
    # reason. This used to live only in a log line, which made "the mode does
    # not respond on this bench" indistinguishable from "the measurement was
    # noise" -- and a missing defocus group ended the run with a bare
    # KeyError. The record travels with the result and with the error instead.
    dropped_modes: dict[str, str] = {}
    raw_widths: dict[str, list[float]] = {}
    for mode in ("defocus", "astig_x", "astig_y", "coma_x", "coma_y", "spherical"):
        pts = [r for r in records if r.mode == mode]
        if pts:
            raw_widths[mode] = [float(r.fwhm_px) for r in pts]
        if len(pts) < _MIN_SWEEP_POINTS:
            if pts:
                dropped_modes[mode] = (
                    f"only {len(pts)} sweep point(s); a width response needs at "
                    "least two"
                )
            continue
        c = np.array([r.coefficient for r in pts], dtype=np.float64)
        w = np.array([r.fwhm_px for r in pts], dtype=np.float64)
        h = np.array([r.hollowness for r in pts], dtype=np.float64)
        ok = np.isfinite(w) & (w > 0) & np.isfinite(h)
        if int(ok.sum()) < _MIN_SWEEP_POINTS:
            dropped_modes[mode] = (
                f"only {int(ok.sum())} point(s) with a finite positive FWHM"
            )
            continue
        c, w, h = c[ok], w[ok], h[ok]

        # A width of `w` is only meaningful while the focal plane is a SINGLE
        # lobe. The direct measure of that is hollowness (intensity at the
        # centroid / peak): a spherical -1.2 rad point read 38.2 px wide with
        # hollowness 0.43, i.e. a ring whose "width" is a ring diameter, and
        # fitting it as a Gaussian lobe is meaningless. The constant below is
        # used *alone* here, and only after an alternative was measured and
        # rejected (the rejected alternative was never applied and was dropped
        # with the model-in-loop chain).
        single_lobe = h >= _DEFOCUS_MIN_HOLLOWNESS
        if (
            not single_lobe.all()
            and int(single_lobe.sum()) >= _MIN_SWEEP_POINTS
            and _has_both_signs(c[single_lobe])
        ):
            logger.info(
                "{} sweep: dropped {} point(s) that are not a single spot lobe "
                "(hollowness < {:.2f}): {}",
                mode,
                int((~single_lobe).sum()),
                _DEFOCUS_MIN_HOLLOWNESS,
                ", ".join(
                    f"c={cc:+.2f} w={ww:.1f} hollow={hh:.2f}"
                    for cc, ww, hh in zip(
                        c[~single_lobe], w[~single_lobe], h[~single_lobe], strict=True
                    )
                ),
            )
            c, w, h = c[single_lobe], w[single_lobe], h[single_lobe]

        # A single spot lobe under growing |coefficient| must broaden
        # monotonically. Enforce that per side and drop the violators: a real
        # sweep had a -4 rad point read *narrower* (12.5 px) than the -2.5 rad one
        # (30.9 px), and that single outlier is enough to drive the fitted
        # curvature negative. A width threshold and hollowness both fail to catch
        # that one -- the outlier's hollowness (0.80) and width both sit inside
        # the single-lobe range -- so the two filters are complementary and both
        # are needed.
        keep = np.ones(c.shape, dtype=bool)
        for side in (np.sign(c) < 0, np.sign(c) > 0):
            idx = np.flatnonzero(side)
            if idx.size == 0:
                continue
            running = -np.inf
            for j in idx[np.argsort(np.abs(c[idx]))]:
                if w[j] >= running:
                    running = w[j]
                else:
                    keep[j] = False
        if int(keep.sum()) >= 4 and _has_both_signs(c[keep]):
            dropped_points = int((~keep).sum())
            if dropped_points:
                logger.info(
                    "{} sweep: dropped {} non-monotonic width point(s); "
                    "fitting the remaining {}",
                    mode,
                    dropped_points,
                    int(keep.sum()),
                )
            c, w = c[keep], w[keep]
        groups[mode] = (c, w)
    return groups, dropped_modes, raw_widths
