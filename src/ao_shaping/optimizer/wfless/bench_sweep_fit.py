"""Fit the bench geometry from a smooth tilt / defocus sweep.

The sweep route (:func:`calibrate_bench_geometry_from_sweep`) derives the
camera/model pixel ratio from the tilt slope and the beam waist from the defocus
width response, using the Zernike forward model as the physical predictor.  It
consumes the raw primitives from :mod:`ao_shaping.optimizer.wfless.bench_sweep`
and returns :class:`~ao_shaping.optimizer.wfless.bench_geometry.BenchGeometry`
with ``method="sweep"``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from loguru import logger

from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)
from ao_shaping.optimizer.wfless.bench_geometry import BenchGeometry
from ao_shaping.optimizer.wfless.bench_geometry_common import (
    _MIN_PUPIL_SEMI_AXIS_PX,
    _MIN_SWEEP_POINTS,
    _pupil_amplitude,
    _spot_fwhm,
)
from ao_shaping.optimizer.wfless.bench_sweep import (
    SweepCalibrationError,
    SweepRecord,
    _defocus_degenerate_message,
    _dropped_summary,
    _fov_ratio_cross_check,
    _publish_dropped,
    _ramp_focal_scale,
    _sweep_dropped_report,
    _sweep_width_groups,
    _tilt_axis_slopes,
    _tilt_physical_cross_check,
)
from ao_shaping.utils.slm.bench_constants import LASER_WAVELENGTH_M

# Above this relative RMS the defocus sweep no longer constrains the waist. The
# tilt-derived scale is unaffected -- it comes from the centroid slope alone.
_MAX_DEFOCUS_FIT_RMS = 0.10


def _fit_curvature_offsets(groups: dict[str, tuple[np.ndarray, np.ndarray]], dropped_modes: dict[str, str]) -> tuple[dict[str, float], dict[str, float]]:
    """Per-mode bench offset + width^2 curvature via a scanned quadratic fit; modes whose
    curvature closes downward or whose minimum sits pinned at the scan edge are dropped.
    """
    # Per-mode bench offset: the width minimum is where the bench's own
    # aberration cancels the command, so fit it rather than assuming it sits at
    # zero (measured -0.171 rad here; assuming zero gave -1.7 rad).
    offsets: dict[str, float] = {}
    curvature_report: dict[str, float] = {}
    for mode, (c, w) in list(groups.items()):
        grid = np.linspace(c.min() - 1.0, c.max() + 1.0, 241)
        best_q: tuple[float, float, float] | None = None
        for c0 in grid:
            basis = (c - c0) ** 2
            design = np.stack([np.ones_like(basis), basis], axis=1)
            sol, *_ = np.linalg.lstsq(design, w**2, rcond=None)
            resid = design @ sol - w**2
            rms = float(np.sqrt(np.mean(resid**2)))
            if best_q is None or rms < best_q[0]:
                best_q = (rms, float(c0), float(sol[1]))
        assert best_q is not None
        rms_q, c0_q, curv_q = best_q
        # A width response that does not open upward has no physical
        # interpretation, and an offset pinned to the scan edge means the
        # minimum was never bracketed. Both happen when a mode simply does not
        # respond -- emitting a number anyway is how an astig_y "offset" of
        # +5.0 rad (the grid edge) with curvature -22 appeared. Drop the mode and
        # keep the ones that do respond.
        edge = (c0_q <= grid[0] + 1e-9) or (c0_q >= grid[-1] - 1e-9)
        if curv_q <= 0.0 or edge:
            reasons: list[str] = []
            if curv_q <= 0.0:
                reasons.append(
                    f"width^2 curvature {curv_q:+.1f} <= 0 (the width does not "
                    "open upward, so the mode has no usable width response)"
                )
            if edge:
                reasons.append(
                    f"the fitted minimum sits at coefficient {c0_q:+.3f}, pinned "
                    f"to the scan edge [{grid[0]:+.2f}, {grid[-1]:+.2f}] (the "
                    "width minimum was never bracketed)"
                )
            reason = "; ".join(reasons)
            dropped_modes[mode] = reason
            logger.warning(
                "{} width response is degenerate (curvature {:.1f}, offset {:+.3f} "
                "{}) -- dropping this mode from the fit rather than reporting a "
                "meaningless offset. Check that the mode is actually reachable on "
                "this bench: its spot widths were {}",
                mode,
                curv_q,
                c0_q,
                "at the scan edge" if edge else "",
                ", ".join(f"{v:.1f}" for v in w),
            )
            groups.pop(mode)
            continue
        offsets[mode] = c0_q
        curvature_report[mode] = curv_q
    return offsets, curvature_report

def _defocus_asymmetry(c_d: np.ndarray, w_d: np.ndarray) -> float:
    """Width asymmetry at matched |c| in the defocus response: a direct read of residual
    non-defocus aberration.
    """
    asym: list[float] = []
    for c_pos in c_d[c_d > 0]:
        neg = c_d[np.isclose(c_d, -c_pos)]
        if neg.size:
            w_p = float(w_d[c_d == c_pos][0])
            w_n = float(w_d[c_d == neg[0]][0])
            if w_p > 0 and w_n > 0:
                asym.append((w_p - w_n) / (0.5 * (w_p + w_n)))
    asymmetry = float(np.mean(asym)) if asym else float("nan")
    return asymmetry

def _fit_joint_waist(groups: dict[str, tuple[np.ndarray, np.ndarray]], dropped_modes: dict[str, str], *, opt: ZernikeCoefficientOptimizer, zeros: np.ndarray, resp_grid: np.ndarray, region: int, a: int, camera_px_per_model_px: float, waist_candidates: Sequence[float]) -> tuple[float, float, dict[str, float]] | None:
    """Joint waist fit: per-candidate model width responses on resp_grid, per-mode offsets
    by interpolation, weighted rel-RMS; None when no candidate is finite.
    """
    # Per waist: precompute each mode's model width response once, then fit the
    # per-mode offsets by interpolating against it. Evaluating the model at every
    # (waist, mode, coefficient, offset) would be ~500x more forward passes.
    best: tuple[float, float, dict[str, float]] | None = None
    for waist in waist_candidates:
        amp = _pupil_amplitude(region, float(waist), a)
        total = 0.0
        n_pts = 0
        per_mode: dict[str, float] = {}
        ok_waist = True
        for mode, (c, w) in list(groups.items()):
            nm = _SWEEP_MODE_NM.get(mode)
            if nm is None:
                # A mode without a model response cannot be fitted. Drop it
                # explicitly instead of dying with a bare KeyError -- the same
                # failure shape this whole block exists to prevent.
                logger.warning(
                    "no model width response for mode {}; dropping it from the "
                    "joint fit",
                    mode,
                )
                dropped_modes[mode] = "no model response for this mode"
                groups.pop(mode)
                continue
            curve = np.array(
                [
                    _spot_fwhm(
                        opt.forward_intensity(
                            zeros, _zernike_phase(region, a, nm, float(g)), amp
                        )
                    )
                    * camera_px_per_model_px
                    for g in resp_grid
                ],
                dtype=np.float64,
            )
            good = np.isfinite(curve) & (curve > 0)
            if good.sum() < 3:
                ok_waist = False
                break
            best_off, best_rms = 0.0, np.inf
            for c0 in resp_grid:
                pred = np.interp(c - c0, resp_grid, curve)
                okp = np.isfinite(pred) & (pred > 0)
                if int(okp.sum()) < _MIN_SWEEP_POINTS:
                    continue
                rms = float(np.sqrt(np.mean(((pred[okp] - w[okp]) / w[okp]) ** 2)))
                if rms < best_rms:
                    best_rms, best_off = rms, float(c0)
            if not np.isfinite(best_rms):
                ok_waist = False
                break
            per_mode[mode] = best_off
            total += best_rms * c.size
            n_pts += c.size
        if not ok_waist or n_pts == 0:
            continue
        joint = total / n_pts
        if best is None or joint < best[0]:
            best = (joint, float(waist), per_mode)
    return best

def _sweep_calibration_notes(*, k_tilt: float, slopes: dict[str, float], sources: dict[str, str], a: int, far_field_size: int, tilt_ratio: float, focal_source: str, camera_px_per_model_px: float, fov_ratio: float, modes_used: str, fit_rms: float, offsets: dict[str, float], curvature_report: dict[str, float], asymmetry: float, waist_determined: bool, dropped_modes: dict[str, str]) -> str:
    """One-line provenance string for bench_geometry.json: both focal routes, cross-checks,
    fitted modes, offsets, curvatures, asymmetry, waist status, dropped modes.
    """
    notes = (
        f"tilt slope {k_tilt:.3f} cam px/rad ("
        + ", ".join(f"{ax}:{v:+.3f}" for ax, v in sorted(slopes.items()))
        + f", via {sorted(set(sources.values()))}), semi-axis {a} model px, "
        f"P={far_field_size}"
        + (
            f"; physical tilt check {tilt_ratio:.2f}x predicted"
            if np.isfinite(tilt_ratio)
            else ""
        )
        + f"; focal scale from {focal_source}"
        + (
            f", FOV cross-check {fov_ratio:.3f} ({abs(camera_px_per_model_px / fov_ratio - 1.0):.0%} apart)"
            if np.isfinite(fov_ratio) and fov_ratio > 0
            else ""
        )
        + f"; modes fitted {modes_used}; joint rel-RMS {fit_rms:.1%}; bench offsets "
        + ", ".join(f"{m} {o:+.3f}" for m, o in sorted(offsets.items()))
        + " rad; curvatures "
        + ", ".join(f"{m} {v:.1f}" for m, v in sorted(curvature_report.items()))
        + f"; defocus asymmetry {asymmetry:+.0%}; waist "
        + ("fitted" if waist_determined else "NOT identifiable, flat-top default")
    )
    if dropped_modes:
        # Persisted with the geometry (bench_geometry.json) so a later reader can
        # tell "this mode did not respond" from "this mode was never swept".
        # The wording keeps the numbers out of the report parser's way.
        notes += "; dropped modes " + ", ".join(
            f"{m} ({_dropped_summary(r)})" for m, r in sorted(dropped_modes.items())
        )
    return notes


def _milsg_log_modes(curvature_report, groups, offsets):
# Per-mode calibration logger loop lifted verbatim
    for mode in sorted(offsets):
        # ``offsets`` only ever holds modes that survived, but the lookup is
        # defensive on purpose: this loop must not be the place where a dropped
        # mode turns into a KeyError.
        points = groups.get(mode)
        n_points = int(points[0].size) if points is not None else 0
        logger.info(
            "  bench {}: offset {:+.3f} rad, width^2 curvature {:.1f}, {} points",
            mode,
            offsets[mode],
            curvature_report[mode],
            n_points,
        )
    return 


def _milsg_build_notes(a, asymmetry, camera_px_per_model_px, curvature_report, dropped, dropped_modes, far_field_size, fit_rms, flats, focal_source, fov_ratio, groups, k_tilt, offsets, opt, region, slopes, sources, tilt_ratio, w_d, waist_determined, waist_model, zeros):
# Flat-field metrics and calibration notes lifted verbatim
    flat_fwhm = float(np.median([r.fwhm_px for r in flats])) if flats else float("nan")
    if not np.isfinite(flat_fwhm):
        flat_fwhm = float(np.min(w_d))
    amp = _pupil_amplitude(region, waist_model, a)
    flat_model = opt.forward_intensity(zeros, np.zeros((region, region)), amp)
    flat_model_fwhm = float(_spot_fwhm(flat_model))

    modes_used = "+".join(sorted(groups))
    notes = _sweep_calibration_notes(k_tilt=k_tilt, slopes=slopes, sources=sources,
                                    a=a, far_field_size=far_field_size,
                                    tilt_ratio=tilt_ratio, focal_source=focal_source,
                                    camera_px_per_model_px=camera_px_per_model_px,
                                    fov_ratio=fov_ratio, modes_used=modes_used,
                                    fit_rms=fit_rms, offsets=offsets,
                                    curvature_report=curvature_report,
                                    asymmetry=asymmetry,
                                    waist_determined=waist_determined,
                                    dropped_modes=dropped_modes)
    _publish_dropped(dropped, dropped_modes)
    return flat_fwhm, flat_model_fwhm, modes_used, notes


def _milsg_fit_waist(a, camera_px_per_model_px, dropped_modes, far_field_size, groups, region, waist_candidates):
# Joint waist fit lifted verbatim
    c_d, w_d = groups["defocus"]
    asymmetry = _defocus_asymmetry(c_d, w_d)

    if waist_candidates is None:
        waist_candidates = [region / float(2**k) for k in range(0, 8)]
    opt = ZernikeCoefficientOptimizer(
        n_orders=1,
        region=region,
        dtype="float64",
        device="cpu",
        far_field_size=far_field_size,
    )
    zeros = np.zeros(opt.n_coefficients)
    span = max(float(np.max(np.abs(c))) for c, _ in groups.values()) + 1.0
    resp_grid = np.linspace(-span, span, _SWEEP_RESPONSE_POINTS)

    best = _fit_joint_waist(groups, dropped_modes, opt=opt, zeros=zeros,
                         resp_grid=resp_grid, region=region, a=a,
                         camera_px_per_model_px=camera_px_per_model_px,
                         waist_candidates=waist_candidates)
    if best is None:
        raise ValueError("no waist candidate produced a finite width response")
    fit_rms, waist_fit, mode_offsets = best
    return asymmetry, fit_rms, mode_offsets, opt, w_d, waist_fit, zeros


def _milsg_focal_crosschecks(a, aperture_m, camera_pixel_m, far_field_size, focal_length_m, k_tilt, model_to_panel, ramp_axis, ramp_scale, sources, wavelength_m):
# Ramp focal scale and physical cross-checks lifted verbatim
    if np.isfinite(ramp_scale) and ramp_scale != 0.0:
        camera_px_per_model_px = abs(ramp_scale) / (
            model_to_panel * float(far_field_size)
        )
        focal_source = f"ramp sweep (S={abs(ramp_scale):.0f} cam px per 1/period, axis {ramp_axis})"
    else:
        camera_px_per_model_px = k_tilt * np.pi * a / float(far_field_size)
        focal_source = f"Zernike tilt slope {k_tilt:.2f} cam px/rad via {sorted(set(sources.values()))}"
        logger.warning(
            "no usable ramp sweep, so the focal scale comes from the Zernike tilt "
            "slope -- which has measured 3x too small on this bench when the spot "
            "deforms. Run --stage sweep with a ramp sweep included, or calibrate "
            "with `python -m ao_shaping.tools.slm.panel.tilt_probe`."
        )

    fov_ratio = _fov_ratio_cross_check(camera_px_per_model_px, focal_source,
                         a=a, far_field_size=far_field_size, aperture_m=aperture_m,
                         camera_pixel_m=camera_pixel_m, focal_length_m=focal_length_m,
                         wavelength_m=wavelength_m)

    tilt_ratio, implied_aperture_mm = _tilt_physical_cross_check(
        k_tilt, camera_px_per_model_px, aperture_m=aperture_m,
        camera_pixel_m=camera_pixel_m, focal_length_m=focal_length_m,
        wavelength_m=wavelength_m)
    return camera_px_per_model_px, focal_source, fov_ratio, tilt_ratio


def _milsg_validate_sweeps(dropped, records, region, semi_axis_model_px):
# Mode bucketing + tilt slope validation lifted verbatim
    tilts = [r for r in records if r.mode == "tilt"]
    ramps = [r for r in records if r.mode == "ramp"]
    defocuses = [r for r in records if r.mode == "defocus"]
    flats = [r for r in records if r.mode == "flat"]
    # Start from a clean out-parameter, so a record left over from an earlier
    # call can never be mistaken for this call's evidence.
    if dropped is not None:
        dropped.clear()
    if len(tilts) < _MIN_SWEEP_POINTS:
        raise ValueError(f"need >= 2 tilt points, got {len(tilts)}")
    if len(defocuses) < _MIN_SWEEP_POINTS:
        raise ValueError(f"need >= 2 defocus points, got {len(defocuses)}")

    a = int(min(semi_axis_model_px or (region // 2), region // 2))
    if a < _MIN_PUPIL_SEMI_AXIS_PX:
        raise ValueError(f"semi_axis_model_px must be >= 8, got {a}")

    slopes, intercepts, sources = _tilt_axis_slopes(tilts)
    if not slopes:
        raise ValueError("no tilt axis had two usable points")
    k_tilt = float(np.mean([abs(v) for v in slopes.values()]))
    if not np.isfinite(k_tilt) or k_tilt <= 0.0:
        raise ValueError(f"tilt slope is not usable: {slopes}")
    return a, flats, k_tilt, ramps, slopes, sources


def _milsg_require_groups(dropped_modes, groups, raw_widths):
# Empty-groups raise lifted verbatim
    if not groups:
        raise SweepCalibrationError(
            "no swept mode produced a usable width response; the spot width does "
            "not respond to any of them on this bench.\n"
            f"{_sweep_dropped_report(dropped_modes, sorted(groups), raw_widths)}\n"
            "下一步 (next steps):\n"
            "  (1) 提高信噪比 (improve the SNR): 提高曝光 (exposure), 并对每一档"
            "做多帧平均再取均值。判据: 宽度的相对分辨率要 ≲ 25%, 因为 w² 拟合会把"
            "宽度的相对误差平方放大 (25% 的宽度分辨率 ⇒ w² 约 50% 的不确定度 ⇒ "
            "曲率就拟不出来了)。\n"
            "  (2) 换用更稳健的宽度估计 (robust width estimator): 用二阶矩半径 "
            "(second-moment radius) 代替 FWHM —— FWHM 由半高位置决定, 低信噪比下"
            "会被噪声主导; 二阶矩半径用整幅光强的二阶矩, 对噪声更稳。",
            dropped=dropped_modes,
            usable_modes=tuple(sorted(groups)),
            widths=raw_widths,
        )
    return


def _milsg_require_defocus(dropped_modes, groups, raw_widths):
# Missing-defocus raise lifted verbatim
    if "defocus" not in groups:
        raise SweepCalibrationError(
            _defocus_degenerate_message(dropped_modes, sorted(groups), raw_widths),
            dropped=dropped_modes,
            usable_modes=tuple(sorted(groups)),
            widths=raw_widths,
        )
    return


def _milsg_waist_verdict(a, asymmetry, camera_px_per_model_px, fit_rms, waist_determined, waist_fit):
# Waist-determined logging and verdict lifted verbatim
    if waist_determined:
        waist_model = waist_fit
    else:
        waist_model = float(a)
        logger.warning(
            "width-response fit is poor (joint rel-RMS {:.1%}; defocus asymmetry "
            "{:+.0%}), so the illumination profile is NOT identifiable: the bench "
            "carries aberration the swept modes do not cover. Emitting the flat-top "
            "default w0={:.0f} model px instead of the best fit {:.0f}. The "
            "tilt-derived scale ({:.4f} cam px per model px) is unaffected -- it "
            "comes from the centroid slope alone. Add coma/spherical sweeps to "
            "identify the waist.",
            fit_rms,
            asymmetry,
            waist_model,
            waist_fit,
            camera_px_per_model_px,
        )
    return waist_model


def _milsg_log_geometry(a, asymmetry, camera_px_per_model_px, fit_rms, flat_fwhm, model_to_panel, modes_used, tilt_ratio, waist_determined, waist_model):
# Calibration summary log lifted verbatim
    logger.info(
        "bench geometry from sweep: aperture={} panel px, w0={:.1f} panel px, "
        "{:.4f} cam px per model px, flat FWHM {:.1f} cam px, joint fit {:.1%} "
        "over {}, asymmetry {:+.0%}, waist {}{}",
        int(round(a * model_to_panel)),
        waist_model * model_to_panel,
        camera_px_per_model_px,
        flat_fwhm,
        fit_rms,
        modes_used,
        asymmetry,
        "fitted" if waist_determined else "provisional",
        (
            f", tilt slope {tilt_ratio:.2f}x the physical prediction"
            if np.isfinite(tilt_ratio)
            else ""
        ),
    )
    return


def calibrate_bench_geometry_from_sweep(
    records: Sequence[SweepRecord],
    *,
    region: int,
    far_field_size: int,
    model_to_panel: float,
    semi_axis_model_px: int | None = None,
    waist_candidates: Sequence[float] | None = None,
    aperture_m: float | None = None,
    camera_pixel_m: float | None = 2.2e-6,
    focal_length_m: float | None = 0.125,
    wavelength_m: float = LASER_WAVELENGTH_M,
    dropped: dict[str, str] | None = None,
) -> BenchGeometry:
    """Calibrate the bench from a tilt + defocus sweep instead of speckle.

    Speckle correlation is unusable on this bench: the panel's effective phase
    resolution is far coarser than one pixel, so a random pupil phase leaves the
    far field a single tight focus instead of a speckle field, and the model has
    nothing to correlate against. A sweep of *smooth* Zernike modes does produce
    a measurable response, and two of them are enough to fix the geometry without
    assuming anything about the bench:

    **Tilt -> far-field scale.** A unit-RMS Zernike tilt of coefficient ``t``
    over an aperture of radius ``a`` imposes a phase gradient ``2t/a``, which
    deflects the beam by ``2t/(k a)`` and shifts the focal spot by
    ``f * t * lambda / (pi * a)``. The model needs
    ``camera_px_per_model_px = lambda f / (P * d_model * camera_pixel)``.
    Dividing the two cancels lambda, f, the model pitch and the camera pixel
    outright and leaves

        ``camera_px_per_model_px = k_tilt * pi * a / P``

    where ``k_tilt`` is the *measured* centroid shift in camera pixels per radian
    of tilt coefficient. That is why this measurement is the right one: it is
    immune to every geometric constant that was wrong before.

    **Defocus -> beam waist.** The spot width grows as the square root of a
    quadratic in the defocus coefficient, whose curvature depends on the
    illumination profile. The waist is fitted by matching the model's width
    response to the measured one, not by a spot-size bisection (which is noisy
    and non-monotonic).

    The two tilt axes are also compared, which doubles as a check on the
    panel/camera axis mapping -- on this bench the axes are swapped 90 degrees,
    and a run whose slopes disagree by more than a few percent says the panel
    was indexed inconsistently.

    Args:
        records: Sweep points, at least two non-collinear tilts and two
            non-zero defocuses.
        region: Model grid side length.
        far_field_size: Zero-padding size the forward model will use.
        model_to_panel: Model pixels to panel pixels (the model grid spans
            ``model_to_panel * region`` panel pixels).
        semi_axis_model_px: Aperture semi-axis in model pixels. Defaults to the
            inscribed circle, ``region // 2``.
        waist_candidates: Gaussian waists to try, in model pixels.
        aperture_m: Illuminated aperture radius in metres, used for the physical
            cross-check on the tilt slope. Pass ``None`` to skip the check.
        camera_pixel_m: Camera pixel pitch, metres, for the same check.
        focal_length_m: 2f lens focal length, metres, for the same check.
        wavelength_m: Laser wavelength, metres, for the same check.
        dropped: Optional out-parameter. On return it holds the swept modes
            whose width response was unusable, mapped to the reason
            (``"width^2 curvature -380.1 <= 0 ..."``, ``"only 1 sweep point"``,
            ...). The dict is filled in place, so the caller owns it and can
            persist it next to the geometry. It is also populated when the
            calibration raises :class:`SweepCalibrationError`.

    Returns:
        The best-fitting :class:`BenchGeometry`, with ``method="sweep"``.

    Raises:
        ValueError: If the records do not contain a usable tilt or defocus sweep.
        SweepCalibrationError: If the defocus width response is degenerate --
            a :class:`ValueError` subclass carrying the dropped modes, their
            reasons, the raw width series and the two concrete next steps
            (raise the SNR, or switch to a second-moment width estimate).
    """
    a, flats, k_tilt, ramps, slopes, sources = _milsg_validate_sweeps(dropped, records, region, semi_axis_model_px)

    ramp_scale, ramp_axis = _ramp_focal_scale(ramps)
    camera_px_per_model_px, focal_source, fov_ratio, tilt_ratio = _milsg_focal_crosschecks(a, aperture_m, camera_pixel_m, far_field_size, focal_length_m, k_tilt, model_to_panel, ramp_axis, ramp_scale, sources, wavelength_m)

    groups, dropped_modes, raw_widths = _sweep_width_groups(records)

    offsets, curvature_report = _fit_curvature_offsets(groups, dropped_modes)

    # The dropped-mode record is published before any raise, so the caller gets
    # the same evidence whether the calibration succeeded or not.
    _publish_dropped(dropped, dropped_modes)

    _milsg_require_groups(dropped_modes, groups, raw_widths)

    # Defocus is not optional: the waist and the focal scale both come from its
    # width response. When it is missing the run cannot continue -- but it must
    # not die with a bare KeyError, which reads like a crash instead of the
    # signal-to-noise verdict it actually is.
    _milsg_require_defocus(dropped_modes, groups, raw_widths)

    # Width asymmetry of the defocus response at matched |c| is a direct read of
    # residual non-defocus aberration.
    asymmetry, fit_rms, mode_offsets, opt, w_d, waist_fit, zeros = _milsg_fit_waist(a, camera_px_per_model_px, dropped_modes, far_field_size, groups, region, waist_candidates)
    for mode, off in mode_offsets.items():
        offsets[mode] = off

    # The illumination profile is only identified when the joint fit is good.
    # Otherwise the best-fit waist is an artefact of the model/bench mismatch, so
    # emit the flat-top default and say so rather than a number that looks
    # calibrated.
    waist_determined = fit_rms <= _MAX_DEFOCUS_FIT_RMS
    waist_model = _milsg_waist_verdict(a, asymmetry, camera_px_per_model_px, fit_rms, waist_determined, waist_fit)

    # --- reported spot widths ------------------------------------------------
    flat_fwhm, flat_model_fwhm, modes_used, notes = _milsg_build_notes(a, asymmetry, camera_px_per_model_px, curvature_report, dropped, dropped_modes, far_field_size, fit_rms, flats, focal_source, fov_ratio, groups, k_tilt, offsets, opt, region, slopes, sources, tilt_ratio, w_d, waist_determined, waist_model, zeros)
    _milsg_log_geometry(a, asymmetry, camera_px_per_model_px, fit_rms, flat_fwhm, model_to_panel, modes_used, tilt_ratio, waist_determined, waist_model)
    _milsg_log_modes(curvature_report, groups, offsets)
    return BenchGeometry(
        panel_disc_radius=int(round(a * model_to_panel)),
        region=region,
        beam_waist_panel_px=float(waist_model * model_to_panel),
        far_field_size=far_field_size,
        camera_px_per_model_px=float(camera_px_per_model_px),
        spot_fwhm_camera_px=flat_fwhm,
        spot_fwhm_model_px=flat_model_fwhm,
        correlation=float(max(0.0, 1.0 - fit_rms)),
        method="sweep",
        calibration_notes=notes,
    )


# Zernike modes the width sweep drives, and the grid the model response is
# precomputed on. The grid must span the measured coefficients plus a margin so
# the per-mode offset can be fitted by interpolation.
_SWEEP_MODE_NM: dict[str, tuple[int, int]] = {
    "defocus": (2, 0),
    "astig_x": (2, -2),
    "astig_y": (2, 2),
    # Coma and spherical are what make the waist identifiable. Defocus alone
    # cannot separate "wide beam" from "residual quadratic aberration", and the
    # quadratic pair is degenerate with it on this bench; the cubic/quartic pair
    # has a different width-vs-coefficient signature, so fitting them together is
    # what breaks the degeneracy.
    "coma_x": (3, -1),
    "coma_y": (3, 1),
    "spherical": (4, 0),
}
_SWEEP_RESPONSE_POINTS = 21


def _zernike_phase(
    region: int, semi_axis_model_px: int, nm: tuple[int, int], coefficient: float
) -> np.ndarray:
    """A unit-RMS Zernike mode of the given coefficient on the model grid."""
    from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

    r = int(min(int(semi_axis_model_px), region // 2))
    gen = ZernikeGenerator((region, region), radius=r, n_orders=nm[0])
    mode = np.asarray(
        gen.generate_polynomial({nm: float(coefficient)}), dtype=np.float64
    )
    if mode.shape != (region, region):  # generator may return (W, H)
        mode = mode.T
    out = np.zeros((region, region), dtype=np.float64)
    finite = np.isfinite(mode)
    out[finite] = mode[finite]
    return out
