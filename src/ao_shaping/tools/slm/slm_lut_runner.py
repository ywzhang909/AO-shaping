"""SLM gray-to-phase LUT calibration runner.

Drives a Santec SLM-200 in half-screen blazed-grating mode and a camera in
its back-focal-plane (2f) setup, scans gray depth or offset, measures the +1
diffraction efficiency of both halves in the same frame (drift-canceled ratio),
inverts to a phase LUT, and saves artifacts.

Usage::

    python -m ao_shaping.tools.slm.slm_lut_runner [OPTIONS]
"""

from __future__ import annotations

import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import click
import numpy as np
from loguru import logger

from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.utils.slm_lut import (
    build_inverse_lut,
    depth_pattern,
    invert_depth_scan,
    invert_offset_scan,
    offset_pattern,
    save_lut,
    stack_halves,
)

# ── Camera factory (lazy import, matching gs_square_runner pattern) ─────────


def _get_miicam_camera(cam_id: int, exposure_ms: float, bit_depth: int = 8):
    """Import and create MiiCam camera instance (lazy)."""
    try:
        from ao_shaping.drivers.ccd.miicam.driver import CameraStreamManager

        cam = CameraStreamManager(
            cam_id=cam_id,
            exposure_time_ms=exposure_ms,
            bit_depth=bit_depth,
        )
        cam.open()
        return cam
    except ImportError as exc:
        logger.warning("MiiCam camera unavailable: {}", exc)
        raise
    except Exception as exc:
        logger.error("MiiCam camera init failed: {}", exc)
        raise


def _get_daheng_camera(cam_id: int, exposure_ms: float):
    """Import and create Daheng camera instance (lazy)."""
    try:
        from ao_shaping.drivers.ccd.daheng import DahengCamManager

        cam = DahengCamManager(cam_id=cam_id, exposure_time_ms=exposure_ms)
        cam.open()
        return cam
    except ImportError as exc:
        logger.warning("Daheng camera unavailable: {}", exc)
        raise
    except Exception as exc:
        logger.error("Daheng camera init failed: {}", exc)
        raise


# ── Spot detection helpers ──────────────────────────────────────────────────

# λ=1064 nm, f=0.125 m, d=8 µm → p_cam ≈ 3.31 µm → scale ≈ 5021 px·period
_DIFFRACTION_SCALE_PX = 5021.0  # λ·f / (d·p_cam) in px·period


def _expected_x_offset(period: int) -> int:
    """Expected +1-order x-offset from center for a given grating period."""
    return int(round(_DIFFRACTION_SCALE_PX / period))


def _peak_in_region(
    frame: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> tuple[int, int, float] | None:
    """Peak ``(gx, gy, value)`` within a rectangular region, or ``None``."""
    region = frame[y0:y1, x0:x1]
    if region.size == 0:
        return None
    idx = np.unravel_index(np.argmax(region), region.shape)
    return int(x0 + idx[1]), int(y0 + idx[0]), float(region[idx])


def _locate_spots(
    calib_frame: np.ndarray,
    period_ref: int,
    period_test: int,
    spot_window: int,
) -> dict[str, tuple[int, int]]:
    """Locate +1-order spots for ref and test halves in calibration frame.

    The optical train is a strict 2f Fourier system (SLM at front focal
    plane, camera at back focal plane, ``f = 125 mm``).  The lens Fourier-
    transforms the SLM field, so the CAMERA coordinate is spatial FREQUENCY:
    SLM in-plane position is erased.  Both half-screen gratings are vertical
    stripes (phase varies along x), hence:

    * the 0-order, the ref +1 order and the test +1 order all land on the
      SAME camera row (the optical-axis row), separated only along x by
      ``_expected_x_offset(period)`` (e.g. 78 px for P64, 157 px for P32);
    * the optical axis is the frame GLOBAL maximum (0-order), never the
      geometric frame center — the beam axis is not pixel-perfectly centered.

    This corrects an earlier near-field assumption (ref spot on top half
    band, test on bottom half band), which is invalid for this 2f setup and
    would fail the moment the panel modulates.

    Args:
        calib_frame: 2-D camera image (float or uint).
        period_ref: Blaze period of reference half (pixels).
        period_test: Blaze period of test half (pixels).
        spot_window: Odd-sized pixel window for spot ROI.

    Returns:
        Dict with keys ``ref``, ``test``, ``center`` mapping to ``(cx, cy)``.
        Exits with ``sys.exit(1)`` if spots cannot be found.

    Raises:
        SystemExit: If the diffraction spots cannot be located.
    """
    h, w = calib_frame.shape[:2]

    x_off_ref = _expected_x_offset(period_ref)
    x_off_test = _expected_x_offset(period_test)

    # 0-order = global maximum (optical axis).  Both +1 orders share its row.
    gy, gx = np.unravel_index(np.argmax(calib_frame), calib_frame.shape)
    gx, gy = int(gx), int(gy)

    # Narrow band around the 0-order row (all orders are on the same row).
    # The x-window must be NARROW (spot_window//2) so it cannot swallow a
    # neighboring order: ref/test sit only 79 px apart and 78 px from the
    # 0-order, so a wide x-window would select the brighter neighbor and
    # then fail the distance check.  A ±20 px window around the theoretical
    # offset covers the true spot (theory is good to ~5 px) while excluding
    # adjacent orders (>58 px away).
    y_half = spot_window * 3
    x_half = spot_window // 2
    y0 = max(gy - y_half, 0)
    y1 = min(gy + y_half + 1, h)

    # A "+1 spot" must stand out twice against its own window mean AND carry
    # at least 1% of the frame maximum (the 0-order).  The second condition
    # rejects the Gaussian tail of the 0-order leaking into the window: such
    # a tail is locally "significant" vs its window but carries ~1e-9 of the
    # 0-order energy and must not be mistaken for a diffraction spot.
    frame_max = float(calib_frame.max())

    def _peak_at(xc: int) -> tuple[int, int, float] | None:
        """Peak within the row band at a narrow x-window centered at ``xc``.

        Requires the local maximum to stand out against its own window
        (``peak > 2 * window_mean``) and against the frame maximum
        (``peak > 1% * frame_max``); a flat/no-signal window therefore
        returns ``None`` instead of fabricating a spot from background.
        """
        x0 = max(xc - x_half, 0)
        x1 = min(xc + x_half + 1, w)
        sub = calib_frame[y0:y1, x0:x1]
        if sub.size == 0:
            return None
        dy, dx = np.unravel_index(np.argmax(sub), sub.shape)
        peak_val = float(sub[dy, dx])
        if peak_val <= 2.0 * float(sub.mean()) or peak_val <= 0.01 * frame_max:
            return None
        return int(x0 + dx), int(y0 + dy), peak_val

    def _side_candidates(x_off: int) -> tuple[tuple[int, int, float] | None, tuple[int, int, float] | None]:
        """Candidate +1 peaks at ``gx - x_off`` (left) and ``gx + x_off``."""
        left = _peak_at(gx - x_off)
        right = _peak_at(gx + x_off)
        return left, right

    ref_left, ref_right = _side_candidates(x_off_ref)
    test_left, test_right = _side_candidates(x_off_test)

    if ref_left is None and ref_right is None:
        logger.error(
            "Cannot locate +1-order spot for REF half (period={}, expect ±{} px). "
            "Check laser alignment and SLM camera.",
            period_ref, x_off_ref,
        )
        sys.exit(1)

    ref_l: tuple[int, int, float] = ref_left if ref_left is not None else (0, 0, -1.0)
    ref_r: tuple[int, int, float] = ref_right if ref_right is not None else (0, 0, -1.0)
    side = -1 if ref_l[2] > ref_r[2] else +1
    ref_spot = (ref_l[0], ref_l[1]) if side == -1 else (ref_r[0], ref_r[1])

    test_peak = test_left if side == -1 else test_right
    if test_peak is None:
        logger.error(
            "Cannot locate +1-order spot for TEST half (period={}, expect ±{} px). "
            "Check laser alignment and SLM camera.",
            period_test, x_off_test,
        )
        sys.exit(1)
    test_spot = (test_peak[0], test_peak[1])

    logger.info(
        "Spots located — center(0th): {}, ref: {} (period={}, side={:+d}), "
        "test: {} (period={})",
        (gx, gy), ref_spot, period_ref, side, test_spot, period_test,
    )

    return {
        "ref": ref_spot,
        "test": test_spot,
        "center": (gx, gy),
    }


# ── Measurement helpers ─────────────────────────────────────────────────────


def _joint_exposure_check(
    frame: np.ndarray,
    rois: list[tuple[tuple[int, int], int]],
    full_well: float,
    camera,
    current_exposure_ms: float,
    bright_floor: float,
    saturation_stop: float,
) -> tuple[np.ndarray, float, bool]:
    """Joint auto-exposure over all spot ROIs — keeps ref/test on ONE frame.

    The scan measures the ref/test power ratio from the same frame so laser
    drift cancels.  Auto-exposure must therefore be applied JOINTLY: if ANY roi
    saturates the exposure is halved; if the AVERAGE roi mean is underexposed
    it is doubled.  At most one adjustment round is applied per call; the
    caller re-captures as needed on the next iteration.

    Args:
        frame: Current camera frame (uint8, uint16 or float64).
        rois: List of ``((cx, cy), window)`` spot regions to guard.
        full_well: Maximum pixel value (255 for 8-bit, 65535 for 16-bit).
        camera: Camera instance with ``reset_exposure_time``.
        current_exposure_ms: Current exposure time.
        bright_floor: Minimum normalized ROI mean to avoid underexposure.
        saturation_stop: Maximum normalized ROI max to avoid saturation.

    Returns:
        ``(frame, exposure_ms, adjusted)`` — a possibly re-captured frame and
        the exposure that produced it.
    """
    h, w = frame.shape[:2]
    max_norm = 0.0
    mean_norm_sum = 0.0
    n_rois = 0

    for (cx, cy), win in rois:
        half_win = win // 2
        y0 = max(cy - half_win, 0)
        y1 = min(cy + half_win + 1, h)
        x0 = max(cx - half_win, 0)
        x1 = min(cx + half_win + 1, w)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        max_norm = max(max_norm, float(np.max(roi)) / full_well)
        mean_norm_sum += float(np.mean(roi)) / full_well
        n_rois += 1

    if n_rois == 0:
        return frame, current_exposure_ms, False

    mean_norm = mean_norm_sum / n_rois
    adjusted = False
    new_exposure = current_exposure_ms

    if max_norm > saturation_stop and current_exposure_ms > 0.01:
        new_exposure = max(current_exposure_ms / 2.0, 0.01)
        logger.info(
            "Auto-exposure: ROI max {:.3f} > saturation_stop {:.3f}, "
            "halving exposure {:.3f}→{:.3f} ms",
            max_norm, saturation_stop, current_exposure_ms, new_exposure,
        )
        adjusted = True
    elif mean_norm < bright_floor and current_exposure_ms < 10000:
        new_exposure = min(current_exposure_ms * 2.0, 10000.0)
        logger.info(
            "Auto-exposure: ROI mean {:.4f} < bright_floor {:.3f}, "
            "doubling exposure {:.3f}→{:.3f} ms",
            mean_norm, bright_floor, current_exposure_ms, new_exposure,
        )
        adjusted = True

    if adjusted:
        camera.reset_exposure_time(new_exposure)
        time.sleep(0.05)  # brief settle after exposure change
        new_frame = np.asarray(
            camera.get_numpy_image(n_sample=1, skip_first=True), dtype=np.float64,
        )
        return new_frame, new_exposure, True

    return frame, current_exposure_ms, False


def _measure_power(
    frame: np.ndarray,
    spot_center: tuple[int, int],
    spot_window: int,
) -> tuple[float, np.ndarray]:
    """Extract +1-order power from a spot ROI (no camera interaction).

    Sums the spot_window×spot_window ROI centered on *spot_center* after
    subtracting a median background estimated from the 2-pixel border ring.
    Exposure is managed OUTSIDE via ``_joint_exposure_check`` so that ref and
    test are measured from the same frame (drift-canceled ratio).

    Args:
        frame: Camera frame (uint8, uint16 or float64).
        spot_center: ``(cx, cy)`` of the tracked spot.
        spot_window: Odd-sized ROI dimension.

    Returns:
        ``(power, frame)`` — background-subtracted integrated intensity.
    """
    h, w = frame.shape[:2]
    half_win = spot_window // 2
    cx, cy = spot_center

    y0 = max(cy - half_win, 0)
    y1 = min(cy + half_win + 1, h)
    x0 = max(cx - half_win, 0)
    x1 = min(cx + half_win + 1, w)

    roi = frame[y0:y1, x0:x1].astype(np.float64)
    if roi.size == 0:
        return 0.0, frame

    # Background: median of the 2-pixel border ring
    border_mask = np.zeros_like(roi, dtype=bool)
    border_mask[:2, :] = True
    border_mask[-2:, :] = True
    border_mask[:, :2] = True
    border_mask[:, -2:] = True
    bg = float(np.median(roi[border_mask])) if border_mask.any() else 0.0

    return float(np.sum(roi - bg)), frame


def _check_spot_drift(
    spot_center: tuple[int, int],
    frame: np.ndarray,
    expected_center: tuple[int, int],
    period: int,
    spot_window: int,
) -> tuple[int, int]:
    """Re-track spot via ROI centroid if drift exceeds threshold.

    Args:
        spot_center: Current tracked spot center ``(cx, cy)``.
        frame: Current camera frame.
        expected_center: ``(cx, cy)`` from calibration frame.
        period: Grating period (for expected x-offset).
        spot_window: ROI window size.

    Returns:
        Updated spot center ``(cx, cy)``.
    """
    h, w = frame.shape[:2]
    half_win = spot_window // 2
    cx, cy = spot_center

    # Check drift against calibration position
    drift_x = abs(cx - expected_center[0])
    drift_y = abs(cy - expected_center[1])
    threshold = spot_window // 4

    if drift_x < threshold and drift_y < threshold:
        return spot_center

    logger.warning(
        "Spot drift detected ({}, {}), re-locating within expected band",
        drift_x, drift_y,
    )

    # Re-locate: search within a window around expected position
    y0 = max(expected_center[1] - spot_window * 2, 0)
    y1 = min(expected_center[1] + spot_window * 2, h)
    x0 = max(expected_center[0] - spot_window * 2, 0)
    x1 = min(expected_center[0] + spot_window * 2, w)

    band = frame[y0:y1, x0:x1].astype(np.float64)
    if band.size == 0:
        return spot_center

    idx = np.unravel_index(np.argmax(band), band.shape)
    new_cy = int(y0 + idx[0])
    new_cx = int(x0 + idx[1])
    logger.info("Spot re-located: ({}, {}) → ({}, {})", cx, cy, new_cx, new_cy)
    return (new_cx, new_cy)


# ── CLI command ──────────────────────────────────────────────────────────────


@click.command()
# Method
@click.option(
    "--method",
    type=click.Choice(["depth", "offset"], case_sensitive=False),
    default="depth",
    show_default=True,
    help="Scan method: depth=scale blaze peak gray; offset=uniform gray-offset scan.",
)
# Grating parameters
@click.option("--period-ref", default=64, type=int, help="Reference half blaze period (SLM px).")
@click.option("--period-test", default=32, type=int, help="Test half blaze period (SLM px).")
@click.option("--gray-step", default=16, type=int, help="Scan step over gray values.")
# Camera
@click.option("--exposure-ms", default=0.03, type=float, help="Initial camera exposure (ms).")
@click.option("--n-frames", default=10, type=int, help="Frames averaged per gray point.")
@click.option(
    "--camera-type",
    type=click.Choice(["miicam", "daheng"], case_sensitive=False),
    default="miicam",
    show_default=True,
    help="Camera type.",
)
@click.option("--cam-id", default=0, type=int, help="Camera device ID.")
# SLM
@click.option("--settle-time", default=0.3, type=float, help="SLM settle wait after write (s).")
@click.option("--slm-number", default=1, type=int, help="SLM device number.")
@click.option("--slm-wavelength", default=1064, type=int, help="SLM working wavelength (nm).")
# Spot detection
@click.option("--spot-window", default=41, type=int, help="Odd-sized pixel window around spot.")
# Auto-exposure thresholds
@click.option("--bright-floor", default=0.02, type=float, help="Min normalized ROI mean.")
@click.option("--saturation-stop", default=0.9, type=float, help="Max normalized ROI max.")
# Output
@click.option(
    "-o", "--output",
    type=click.Path(),
    default="data/slm_lut",
    show_default=True,
    help="Output directory for artifacts.",
)
@click.option(
    "--display/--no-display",
    default=False,
    show_default=True,
    help="Show matplotlib figures (blocking) instead of just saving PNGs.",
)
def run(
    method: str,
    period_ref: int,
    period_test: int,
    gray_step: int,
    exposure_ms: float,
    n_frames: int,
    settle_time: float,
    slm_number: int,
    slm_wavelength: int,
    camera_type: str,
    cam_id: int,
    spot_window: int,
    bright_floor: float,
    saturation_stop: float,
    output: str,
    display: bool,
) -> None:
    """SLM gray-to-phase LUT calibration.

    Drives the Santec SLM-200 in half-screen blazed-grating mode, scans gray
    depth or offset, measures +1 diffraction efficiency of both halves (drift-
    canceled ratio), inverts to a phase LUT, and saves artifacts.
    """
    output_dir = Path(output)
    run_name = datetime.now().strftime("run-%Y%m%d_%H%M%S")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    slm: SantecSLM200 | None = None
    camera = None
    final_exposure_ms = exposure_ms

    try:
        # ═══════════════════════════════════════════════════════════════════
        # 1. Open SLM
        # ═══════════════════════════════════════════════════════════════════
        logger.info("Connecting to SLM #{} (wavelength={} nm)...", slm_number, slm_wavelength)
        slm = SantecSLM200(
            slm_number=slm_number,
            wavelength=slm_wavelength,
            video_mode=0,  # memory mode
        )
        slm.open()

        # Query device wavelength info — NEVER hardcode gray_for_2pi
        wl_device, gray_for_2pi = slm.get_wavelength_info()
        logger.info(
            "SLM #{} connected — device wl={}nm, 2pi gray={}",
            slm_number, wl_device, gray_for_2pi,
        )

        # Panel dimensions: Panel_Res = (width, height) for SantecSLM200
        slm_width, slm_height = slm.Panel_Res[0], slm.Panel_Res[1]
        logger.info("SLM panel: {}x{} ({} bit)", slm_width, slm_height, slm.Gray_Scale_bits)

        # Warn if WavefrontCorrection is active
        if slm._correction.is_valid:
            logger.warning(
                "WavefrontCorrection is loaded — it will be applied on top of "
                "calibration patterns; results may be contaminated. Consider "
                "disabling correction before LUT calibration."
            )

        slm_bits = slm.Gray_Scale_bits
        max_gray = (1 << slm_bits) - 1  # e.g. 1023 for 10-bit

        # ═══════════════════════════════════════════════════════════════════
        # 2. Open camera
        # ═══════════════════════════════════════════════════════════════════
        logger.info("Opening {} camera (id={}, exposure={:.3f} ms)...", camera_type, cam_id, exposure_ms)
        if camera_type == "daheng":
            camera = _get_daheng_camera(cam_id, exposure_ms)
        else:
            camera = _get_miicam_camera(cam_id, exposure_ms)
        final_exposure_ms = exposure_ms

        # Estimate full-well based on bit depth (for saturation detection)
        cam_bit_depth = getattr(camera, "_bit_depth", 8)
        full_well = float((1 << cam_bit_depth) - 1)
        logger.info("Camera: {} bit, full_well={:.0f}", cam_bit_depth, full_well)

        # ═══════════════════════════════════════════════════════════════════
        # 3. Locate spots from CALIBRATION frame
        # ═══════════════════════════════════════════════════════════════════
        logger.info("Building calibration frame (full-depth blaze on both halves)...")

        # Half-screen patterns: ref on top, test on bottom.  Each half is
        # slm_height//2 rows so stack_halves produces the FULL panel height.
        half_h = slm_height // 2

        # Reference half: full-depth blaze at gray_for_2pi
        ref_calib = depth_pattern(period_ref, gray_for_2pi, half_h, slm_width)
        # Test half: also full-depth blaze for calibration
        test_calib = depth_pattern(period_test, gray_for_2pi, half_h, slm_width)
        calib_pattern = stack_halves(ref_calib, test_calib, axis=0)

        slm.display_data(calib_pattern, wait_time_s=settle_time)
        time.sleep(settle_time)

        calib_frame = np.asarray(
            camera.get_numpy_image(n_sample=n_frames, skip_first=True),
            dtype=np.float64,
        )
        logger.info(
            "Calibration frame captured: shape={}, max={}",
            calib_frame.shape, calib_frame.max(),
        )

        spots = _locate_spots(calib_frame, period_ref, period_test, spot_window)
        ref_center = spots["ref"]
        test_center = spots["test"]

        # Joint auto-exposure on the located spots (keeps both on ONE frame)
        calib_frame, final_exposure_ms, _ = _joint_exposure_check(
            calib_frame,
            [(ref_center, spot_window), (test_center, spot_window)],
            full_well, camera, final_exposure_ms, bright_floor, saturation_stop,
        )

        logger.info(
            "Spot centers — ref: {}, test: {}",
            ref_center, test_center,
        )

        # Save calibration frame
        np.save(run_dir / "calibration_frame.npy", calib_frame)

        # ═══════════════════════════════════════════════════════════════════
        # 4. Build scan gray values
        # ═══════════════════════════════════════════════════════════════════
        # Both methods scan the full 0..max_gray range so the efficiency peak
        # stays BRACKETED even if the true 2π gray deviates from
        # get_wavelength_info() — otherwise argmax would sit on the scan edge
        # and invert_depth_scan would mis-assign the last point to 2π.
        max_g = max_gray

        g_values = np.arange(0, max_g + 1, gray_step, dtype=int)
        # Ensure last value included
        if g_values[-1] != max_g:
            g_values = np.append(g_values, max_g)

        logger.info("Scan: method={}, {} gray points from {} to {}", method, len(g_values), g_values[0], g_values[-1])

        # ═══════════════════════════════════════════════════════════════════
        # 5. Scan loop
        # ═══════════════════════════════════════════════════════════════════
        eta = np.zeros(len(g_values), dtype=np.float64)
        p_ref_arr = np.zeros(len(g_values), dtype=np.float64)
        p_test_arr = np.zeros(len(g_values), dtype=np.float64)

        # Reference pattern (always full-depth blaze at gray_for_2pi), top half
        ref_pattern = depth_pattern(period_ref, gray_for_2pi, half_h, slm_width)

        # Per-spot expected calibration centers for drift detection
        ref_calib_center = ref_center
        test_calib_center = test_center

        for i, g in enumerate(g_values):
            if method == "depth":
                # Test half: blaze with peak_gray = g
                test_pattern = depth_pattern(period_test, int(g), half_h, slm_width)
            else:
                # Test half: offset blaze (full depth + gray_offset = g)
                test_pattern = offset_pattern(
                    period_test, gray_for_2pi, int(g), slm_bits, half_h, slm_width,
                )

            combined = stack_halves(ref_pattern, test_pattern, axis=0)
            slm.display_data(combined, wait_time_s=settle_time)

            # Capture and average
            frame = np.asarray(
                camera.get_numpy_image(n_sample=n_frames, skip_first=True),
                dtype=np.float64,
            )

            # Period-check: re-track spots if drift exceeds threshold
            ref_center = _check_spot_drift(
                ref_center, frame, ref_calib_center, period_ref, spot_window,
            )
            test_center = _check_spot_drift(
                test_center, frame, test_calib_center, period_test, spot_window,
            )

            # Joint auto-exposure once — both spots measured from the SAME frame
            frame, final_exposure_ms, _ = _joint_exposure_check(
                frame,
                [(ref_center, spot_window), (test_center, spot_window)],
                full_well, camera, final_exposure_ms, bright_floor, saturation_stop,
            )

            # Measure both powers from the same frame (drift-canceled ratio)
            p_ref, _ = _measure_power(frame, ref_center, spot_window)
            p_test, _ = _measure_power(frame, test_center, spot_window)

            # Drift-canceled ratio
            if p_ref > 0:
                eta[i] = p_test / p_ref
            else:
                eta[i] = 0.0
                logger.warning("g={}: P_ref=0, ratio undefined (set to 0)", g)

            p_ref_arr[i] = p_ref
            p_test_arr[i] = p_test

            if (i + 1) % max(1, len(g_values) // 10) == 0 or i == len(g_values) - 1:
                logger.info(
                    "[{}/{}] g={}, eta={:.4f}, P_ref={:.1f}, P_test={:.1f}, exp={:.3f}ms",
                    i + 1, len(g_values), g, eta[i], p_ref, p_test, final_exposure_ms,
                )

        logger.info("Scan complete — max eta={:.4f} at g={}", float(np.max(eta)), int(g_values[np.argmax(eta)]))

        # ═══════════════════════════════════════════════════════════════════
        # 6. Inversion + artifacts
        # ═══════════════════════════════════════════════════════════════════
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if method == "depth":
            phi = invert_depth_scan(eta)
        else:
            phi = invert_offset_scan(eta, g_values, gray_for_2pi)

        # Build inverse LUT: phase → gray
        t_grid, inverse_gray = build_inverse_lut(phi, g_values, n=1024)

        # ── Save plots ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # (a) eta vs g
        axes[0].plot(g_values, eta, "o-", markersize=3)
        axes[0].set_xlabel("Gray value")
        axes[0].set_ylabel("η (+1 efficiency ratio)")
        axes[0].set_title(f"Diffraction efficiency ({method})")
        axes[0].grid(True, alpha=0.3)

        # (b) recovered phi vs g
        axes[1].plot(g_values, phi, "s-", markersize=3)
        axes[1].set_xlabel("Gray value")
        axes[1].set_ylabel("Phase (rad)")
        axes[1].set_title("Recovered phase vs gray")
        axes[1].grid(True, alpha=0.3)

        # (c) inverse LUT: phase → gray
        axes[2].plot(t_grid, inverse_gray, "D-", markersize=2)
        axes[2].set_xlabel("Phase (rad)")
        axes[2].set_ylabel("Gray value")
        axes[2].set_title("Inverse LUT (phase → gray)")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(
            f"SLM LUT Calibration — method={method}, λ={slm_wavelength}nm, "
            f"2π gray={gray_for_2pi}",
            fontsize=12,
        )
        fig.tight_layout()
        plot_path = run_dir / "lut_calibration.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info("Calibration plot saved: {}", plot_path)
        if display:
            plt.show()
        plt.close(fig)

        # ── Save LUT artifacts via slm_lut.save_lut ──
        spot_centers = {
            "ref": list(ref_center),
            "test": list(test_center),
        }
        meta = {
            "method": method,
            "period_ref": period_ref,
            "period_test": period_test,
            "gray_step": gray_step,
            "gray_for_2pi": gray_for_2pi,
            "slm_wavelength": slm_wavelength,
            "spot_centers": spot_centers,
            "exposure_final_ms": final_exposure_ms,
            "timestamp": datetime.now().isoformat(),
        }

        lut_dir = run_dir / "lut"
        save_lut(lut_dir, g_values, phi, inverse_gray, meta=meta)
        logger.info("LUT saved to {}", lut_dir)

        # ── Save run record pkl ──
        patterns_summary = {
            "ref_period": period_ref,
            "test_period": period_test,
            "ref_pattern_shape": list(ref_pattern.shape),
        }
        record = {
            "g_values": g_values,
            "eta": eta,
            "phi": phi,
            "inverse_gray": inverse_gray,
            "t_grid": t_grid,
            "meta": meta,
            "p_ref": p_ref_arr,
            "p_test": p_test_arr,
            "patterns_summary": patterns_summary,
        }
        records_path = run_dir / "records.pkl"
        with open(records_path, "wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Run record saved: {}", records_path)

        # ═══════════════════════════════════════════════════════════════════
        # 7. Final summary
        # ═══════════════════════════════════════════════════════════════════
        peak_eta = float(np.max(eta))
        peak_g = int(g_values[np.argmax(eta)])

        # Monotonicity check of recovered phase
        phi_diff = np.diff(phi)
        n_non_mono = int(np.sum(phi_diff < 0))
        mono_pct = float(100.0 * (1.0 - n_non_mono / max(len(phi_diff), 1)))

        summary = (
            f"LUT calibration complete — method={method}\n"
            f"  max η = {peak_eta:.4f} at g = {peak_g} (gray_for_2pi={gray_for_2pi})\n"
            f"  phase monotonicity: {mono_pct:.1f}%\n"
            f"  final exposure: {final_exposure_ms:.3f} ms\n"
            f"  LUT saved to: {lut_dir}\n"
            f"  run dir: {run_dir}"
        )
        click.echo(summary)
        click.echo(
            f"\nLUT saved to {lut_dir}. "
            f"Load into SLM via: slm.load_lut('{lut_dir}')"
        )

    except Exception as exc:
        logger.error("SLM LUT runner failed: {}", exc)
        raise
    finally:
        logger.info("Cleaning up devices...")
        if slm is not None:
            try:
                slm.close()
                logger.info("SLM disconnected")
            except Exception as exc:
                logger.warning("SLM disconnect failed: {}", exc)
        if camera is not None:
            try:
                camera.close()
                logger.info("Camera disconnected")
            except Exception as exc:
                logger.warning("Camera disconnect failed: {}", exc)


def main() -> None:
    """CLI entry point with unified exception handling."""
    try:
        run(standalone_mode=True)
    except SystemExit:
        raise
    except Exception as exc:
        click.echo(f"Error: {exc}")
        logger.exception("SLM LUT runner failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
