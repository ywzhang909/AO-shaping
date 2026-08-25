"""
CCD Real-time Image Analyzer (Streamlit page)

Features:
1. Real-time CCD image capture (Daheng or MiiCam camera)
2. Background thread capture loop (r50-style daemon thread + queue pattern)
3. Automatic enclosing ellipse calculation
4. X/Y intensity cross-sections from centroid
5. Gaussian fit curves for cross-sections
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import types
from typing import Any

import numpy as np
import streamlit as st

from ao_shaping.utils.file import ROOT_DIR as PROJECT_ROOT

SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Patch miicam module before importing ccd package
if "miicam" not in sys.modules:
    sys.modules["miicam"] = types.ModuleType("miicam")

from io import BytesIO

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from loguru import logger
from PIL import Image
from scipy import ndimage
from scipy.optimize import curve_fit

from ao_shaping.drivers.ccd.daheng import DahengCamManager
from ao_shaping.utils.spots_calc import centroid

# ── Camera type import (optional) ────────────────────────────────────────────
try:
    from ao_shaping.drivers.ccd.miicam.driver import (
        CameraStreamManager as MIICamManager,
    )
except Exception:
    MIICamManager = None

# ── Constants ─────────────────────────────────────────────────────────────────
_REFRESH_INTERVAL = 0.15  # seconds – UI rerun cadence while capture loop runs


# =============================================================================
# Background thread capture loop (r50-style daemon thread pattern)
# =============================================================================


def _camera_capture_tick(
    camera: Any, img_queue: queue.Queue[Any], params: dict[str, Any]
) -> None:
    """One tick of the capture loop: grab a frame, push to queue.

    Runs inside the daemon thread.  Must NOT touch session_state.
    """
    try:
        img = camera.get_numpy_image(
            n_sample=params.get("n_sample", 1),
            skip_first=params.get("skip_first", False),
        )
        # Non-blocking: drop the oldest frame if the queue is full so the main
        # thread always sees the freshest image.
        if img_queue.full():
            try:
                img_queue.get_nowait()
            except queue.Empty:
                pass
        img_queue.put(img, block=False)
    except Exception as exc:
        try:
            img_queue.put(("error", str(exc)), block=False)
        except Exception:
            pass


def _run_capture_loop(
    camera: Any,
    img_queue: queue.Queue[Any],
    params: dict[str, Any],
    stop_event: threading.Event,
) -> None:
    """Daemon-thread entry: repeatedly call the capture tick until stopped.

    Mirrors ``r50_voltage_send.run_loop`` — never touches session_state;
    all data crosses to the main thread via *img_queue*.
    """
    try:
        while not stop_event.is_set():
            _camera_capture_tick(camera, img_queue, params)
            time.sleep(float(params.get("dt", _REFRESH_INTERVAL)))
    except Exception as exc:
        try:
            img_queue.put(("error", f"Capture loop crashed: {exc}"), block=False)
        except Exception:
            pass
    finally:
        stop_event.set()


def _start_capture_loop(
    camera: Any,
    params: dict[str, Any] | None = None,
) -> None:
    """Start (or replace) the background capture loop.

    Stops any existing loop first, creates a fresh Event + Queue, then
    launches a daemon thread.  All state lives in ``st.session_state``.
    """
    _stop_capture_loop()

    if params is None:
        params = {}
    params = dict(params)  # snapshot
    params.setdefault("dt", _REFRESH_INTERVAL)
    params.setdefault("n_sample", 1)
    params.setdefault("skip_first", False)

    ev = threading.Event()
    q: queue.Queue[Any] = queue.Queue(maxsize=3)
    st.session_state["ccd_loop_stop_event"] = ev
    st.session_state["ccd_img_queue"] = q

    thread = threading.Thread(
        target=_run_capture_loop,
        args=(camera, q, params, ev),
        daemon=True,
        name="ccd-capture-loop",
    )
    thread.start()
    st.session_state["ccd_capture_loop_running"] = True
    logger.info("Capture loop started (dt={:.3f}s)", params["dt"])


def _stop_capture_loop() -> None:
    """Request the capture loop to stop and clean up state."""
    ev = st.session_state.get("ccd_loop_stop_event")
    if ev is not None:
        ev.set()
        st.session_state["ccd_loop_stop_event"] = None
    st.session_state["ccd_img_queue"] = None
    st.session_state["ccd_capture_loop_running"] = False


def _drain_capture_feedback() -> None:
    """Main-thread consumer: drain the image queue on every rerun.

    Latest image (or error) is stored in ``st.session_state["current_image"]``.
    Called unconditionally at the top of ``main()``.
    """
    q = st.session_state.get("ccd_img_queue")
    if q is None:
        return
    latest_img: np.ndarray | None = None
    latest_error: str | None = None
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "error":
            latest_error = item[1]
        elif isinstance(item, np.ndarray):
            latest_img = item
    if latest_error:
        st.session_state["ccd_last_error"] = latest_error
    if latest_img is not None:
        # --- Analysis (runs in main thread, fast) ---
        img = latest_img
        cx, cy = centroid(img, moment=1, threshold=0.01)
        ellipse_params = calculate_enclosing_ellipse(img)
        x_profile, y_profile = get_cross_sections(img, (cx, cy))
        x = np.arange(len(x_profile))
        y_arr = np.arange(len(y_profile))
        x_popt, _ = fit_gaussian(x, x_profile)
        y_popt, _ = fit_gaussian(y_arr, y_profile)

        st.session_state["current_image"] = {
            "img": img,
            "ellipse_params": ellipse_params,
            "cx": cx,
            "cy": cy,
            "x_profile": x_profile,
            "y_profile": y_profile,
            "x_popt": x_popt,
            "y_popt": y_popt,
            "x": x,
            "y": y_arr,
            "capture_time": time.time(),
        }
        st.session_state["last_update_time"] = time.time()


# =============================================================================
# Session state initialisation
# =============================================================================


def _initialize_camera_state() -> None:
    """Initialise all camera-related session-state keys (idempotent)."""
    st.session_state.setdefault("camera", None)
    st.session_state.setdefault("camera_connected", False)
    st.session_state.setdefault("camera_id", 0)
    st.session_state.setdefault("camera_type", "Daheng")  # "Daheng" | "MiiCam"
    st.session_state.setdefault("exposure_time_ms", 50)
    st.session_state.setdefault("auto_exposure", False)
    st.session_state.setdefault("update_interval", _REFRESH_INTERVAL)
    st.session_state.setdefault("roi_size", 0)
    st.session_state.setdefault("roi_center", (0, 0))
    st.session_state.setdefault("auto_refresh", True)
    st.session_state.setdefault("last_update_time", 0)
    st.session_state.setdefault("current_image", None)
    st.session_state.setdefault("current_analysis", None)
    # Capture-loop bookkeeping
    st.session_state.setdefault("ccd_loop_stop_event", None)
    st.session_state.setdefault("ccd_img_queue", None)
    st.session_state.setdefault("ccd_capture_loop_running", False)
    st.session_state.setdefault("ccd_last_error", "")


# =============================================================================
# Analysis helpers (unchanged)
# =============================================================================


def gaussian(
    x: np.ndarray, amplitude: float, center: float, sigma: float, offset: float
) -> np.ndarray:
    """Gaussian function for curve fitting."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2)) + offset


def fit_gaussian(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray | None, float | None]:
    """Fit a Gaussian to *x*, *y* data. Returns (popt, residual) or (None, None)."""
    try:
        amplitude = np.max(y) - np.min(y)
        center = x[np.argmax(y)]
        sigma = (x.max() - x.min()) / 6
        offset = np.min(y)
        if sigma <= 0:
            sigma = 1.0
        x_min, x_max = float(x.min()), float(x.max())
        if x_max <= x_min + 0.1:
            x_max = x_min + 10.0
        popt, _pcov = curve_fit(
            gaussian,
            x,
            y,
            p0=[amplitude, center, sigma, offset],
            bounds=([0, x_min, 0.5, -np.inf], [np.inf, x_max, np.inf, np.inf]),
        )
        residual = float(np.sum((y - gaussian(x, *popt)) ** 2))
        return popt, residual
    except Exception as exc:
        logger.warning("Gaussian fitting failed: {}", exc)
        return None, None


def calculate_enclosing_ellipse(
    img: np.ndarray, threshold_ratio: float = 0.1
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Compute the enclosing ellipse of the brightest connected region."""
    threshold = threshold_ratio * np.max(img)
    binary = img > threshold
    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        h, w = img.shape
        return ((w / 2, h / 2), (w / 4, h / 4), 0)

    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    max_label = int(np.argmax(sizes)) + 1
    largest_component = labeled == max_label
    cy, cx = ndimage.center_of_mass(largest_component)

    y_coords, x_coords = np.where(largest_component)
    if len(x_coords) < 3:
        h, w = img.shape
        return ((cx, cy), (w / 4, h / 4), 0)

    x_centered = x_coords - cx
    y_centered = y_coords - cy
    m20 = float(np.sum(x_centered**2))
    m02 = float(np.sum(y_centered**2))
    m11 = float(np.sum(x_centered * y_centered))

    delta = np.sqrt((m20 - m02) ** 2 + 4 * m11**2)
    eigenvalues = [(m20 + m02 + delta) / 2, (m20 + m02 - delta) / 2]
    a = 2 * np.sqrt(max(eigenvalues))
    b = 2 * np.sqrt(min(eigenvalues))
    angle = 0.5 * np.arctan2(2 * m11, m20 - m02) if (m11 != 0 or m20 != m02) else 0.0
    return ((cx, cy), (a, b), float(np.degrees(angle)))


def draw_ellipse_on_image(
    img: np.ndarray,
    ellipse_params: tuple[tuple[float, float], tuple[float, float], float],
) -> np.ndarray:
    """Draw the enclosing ellipse on *img* and return the annotated RGB image."""
    img_normalized = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(
        np.uint8
    )
    h, w = img_normalized.shape
    fig, ax = plt.subplots(1, 1, figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img_normalized, cmap="gray")
    center, axes, angle = ellipse_params
    ellipse = mpatches.Ellipse(
        center, axes[0] * 2, axes[1] * 2, angle=angle,
        fill=False, edgecolor="red", linewidth=2,
    )
    ax.add_patch(ellipse)
    ax.plot(center[0], center[1], "r+", markersize=10, markeredgewidth=2)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    result = plt.imread(buf)
    plt.close(fig)
    if result.ndim == 2:
        result = np.stack([result] * 3, axis=-1)
    return (result * 255).astype(np.uint8)


def get_cross_sections(
    img: np.ndarray, center: tuple[float, float], width: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Extract X/Y intensity profiles through *center* averaged over *width* pixels."""
    cy, cx = int(round(center[1])), int(round(center[0]))
    h, w = img.shape
    half_width = width // 2

    x_start = max(0, cx - half_width)
    x_end = min(w, cx + half_width + 1)
    if x_end > x_start:
        x_profile = np.mean(
            img[max(0, cy - half_width) : min(h, cy + half_width + 1), x_start:x_end],
            axis=0,
        )
    else:
        x_profile = np.array([img[cy, cx]])

    y_start = max(0, cy - half_width)
    y_end = min(h, cy + half_width + 1)
    if y_end > y_start:
        y_profile = np.mean(
            img[y_start:y_end, max(0, cx - half_width) : min(w, cx + half_width + 1)],
            axis=1,
        )
    else:
        y_profile = np.array([img[cy, cx]])

    return x_profile, y_profile


# =============================================================================
# Camera helpers
# =============================================================================


def _create_camera(camera_type: str, cam_id: int, exposure_ms: float) -> Any:
    """Instantiate the selected camera driver."""
    if camera_type == "MiiCam":
        if MIICamManager is None:
            raise RuntimeError(
                "MiiCam driver is not available. "
                "Ensure the MIICAM SDK is installed (set MIICAM_SDK_PATH)."
            )
        return MIICamManager(cam_id=cam_id, exposure_time_ms=exposure_ms)
    # Default: Daheng
    return DahengCamManager(cam_id=cam_id, exposure_time_ms=exposure_ms)


def _safe_close_camera(camera: Any) -> None:
    """Safely close a camera, handling both context-manager and direct styles."""
    if camera is None:
        return
    try:
        camera.__exit__(None, None, None)  # type: ignore[union-attr]
    except AttributeError:
        if hasattr(camera, "close"):
            camera.close()  # type: ignore[union-attr]
        elif hasattr(camera, "cam") and camera.cam is not None:  # type: ignore[union-attr]
            camera.cam.stream_off()  # type: ignore[union-attr]
            camera.cam.close_device()  # type: ignore[union-attr]


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    st.set_page_config(page_title="CCD Image Analyzer", page_icon="📷", layout="wide")

    _initialize_camera_state()
    _drain_capture_feedback()  # consume latest frame from background thread

    st.title("📷 CCD Real-time Image Analyzer")
    st.markdown("Real-time CCD camera display and beam analysis")

    # ── Sidebar: Camera Settings ─────────────────────────────────────────────
    with st.sidebar:
        st.header("Camera Settings")

        # Camera type selector
        cam_type = st.selectbox(
            "Camera Type",
            options=["Daheng", "MiiCam"],
            index=0 if st.session_state.camera_type == "Daheng" else 1,
            help="Select camera driver",
            key="ccd_cam_type_select",
        )
        st.session_state.camera_type = cam_type

        st.session_state.camera_id = st.number_input(
            "Camera ID",
            min_value=0,
            value=st.session_state.camera_id,
            help="Camera device ID",
        )

        new_exposure = st.slider(
            "Exposure Time (ms)",
            min_value=1,
            max_value=1000,
            value=st.session_state.exposure_time_ms,
            help="Camera exposure time",
        )

        # Update camera exposure time if changed and camera is connected
        if new_exposure != st.session_state.exposure_time_ms:
            st.session_state.exposure_time_ms = new_exposure
            if (
                st.session_state.camera is not None
                and st.session_state.camera_connected
            ):
                try:
                    if hasattr(st.session_state.camera, "reset_exposure_time"):
                        st.session_state.camera.reset_exposure_time(new_exposure)
                    logger.info("Exposure time updated to {}ms", new_exposure)
                    st.rerun()
                except Exception as exc:
                    logger.warning("Failed to update exposure time: {}", exc)

        st.session_state.auto_exposure = st.checkbox(
            "Auto Exposure",
            value=st.session_state.auto_exposure,
            help="Auto adjust exposure",
        )

        st.session_state.update_interval = st.slider(
            "Update Interval (s)",
            min_value=0.05,
            max_value=5.0,
            value=st.session_state.update_interval,
            step=0.05,
            help="Image refresh interval (smaller = smoother live view)",
        )

        st.divider()

        # Recover stale camera object
        if (
            st.session_state.camera is not None
            and not st.session_state.camera_connected
        ):
            try:
                if (
                    hasattr(st.session_state.camera, "cam")
                    and st.session_state.camera.cam is not None
                ):
                    st.session_state.camera_connected = True
                    logger.info("Recovered existing camera connection")
            except Exception:
                st.session_state.camera = None

        # ── Connect / Disconnect ─────────────────────────────────────────────
        if not st.session_state.camera_connected:
            if st.button("Connect Camera", type="primary"):
                try:
                    cam = _create_camera(
                        st.session_state.camera_type,
                        st.session_state.camera_id,
                        st.session_state.exposure_time_ms,
                    )
                    cam.initialize()
                    st.session_state.camera = cam
                    st.session_state.camera_connected = True
                    st.success(
                        f"{st.session_state.camera_type} camera connected"
                    )
                except Exception as exc:
                    st.error(f"Connection failed: {exc}")
        else:
            if st.button("Disconnect Camera", type="secondary"):
                try:
                    _stop_capture_loop()
                    _safe_close_camera(st.session_state.camera)
                    st.session_state.camera = None
                    st.session_state.camera_connected = False
                    st.session_state.current_image = None
                    st.info("Camera disconnected")
                except Exception as exc:
                    logger.error("Disconnect failed: {}", exc)
                    _stop_capture_loop()
                    st.session_state.camera = None
                    st.session_state.camera_connected = False
                    st.error(f"Disconnect failed: {exc}")

    # ── Main area ────────────────────────────────────────────────────────────
    if st.session_state.camera_connected and st.session_state.camera is not None:
        st.subheader("Real-time Image")

        # Refresh controls
        col_auto, col_manual, col_status = st.columns([1, 1, 2])

        with col_auto:
            st.session_state.auto_refresh = st.checkbox(
                "Auto Refresh",
                value=st.session_state.auto_refresh,
                help="Automatically refresh image",
            )

        with col_manual:
            manual_refresh = st.button("🔄 Refresh Image", type="primary")

        with col_status:
            ci = st.session_state.current_image
            if ci is not None:
                capture_time = ci.get("capture_time", 0)
                if capture_time > 0:
                    from datetime import datetime
                    ts = datetime.fromtimestamp(capture_time)
                    st.success(f"✓ Live at {ts.strftime('%H:%M:%S.%f')[:-3]}")
            else:
                st.info("Click to capture")

        # Show any error from the capture loop
        last_err = st.session_state.get("ccd_last_error", "")
        if last_err:
            st.warning(f"Capture error: {last_err}")
            st.session_state["ccd_last_error"] = ""

        # ── Capture dispatch ─────────────────────────────────────────────────
        loop_running = st.session_state.ccd_capture_loop_running

        if manual_refresh:
            # Single-shot capture in the main thread
            try:
                img = st.session_state.camera.get_numpy_image(
                    n_sample=1, skip_first=False
                )
                cx, cy = centroid(img, moment=1, threshold=0.01)
                ellipse_params = calculate_enclosing_ellipse(img)
                x_profile, y_profile = get_cross_sections(img, (cx, cy))
                x = np.arange(len(x_profile))
                y_arr = np.arange(len(y_profile))
                x_popt, _ = fit_gaussian(x, x_profile)
                y_popt, _ = fit_gaussian(y_arr, y_profile)
                st.session_state.current_image = {
                    "img": img,
                    "ellipse_params": ellipse_params,
                    "cx": cx, "cy": cy,
                    "x_profile": x_profile, "y_profile": y_profile,
                    "x_popt": x_popt, "y_popt": y_popt,
                    "x": x, "y": y_arr,
                    "capture_time": time.time(),
                }
                st.session_state.last_update_time = time.time()
            except Exception as exc:
                st.error(f"Failed to get image: {exc}")

        elif st.session_state.auto_refresh and not loop_running:
            # Start the background capture loop (r50 pattern)
            _start_capture_loop(
                st.session_state.camera,
                params={"dt": st.session_state.update_interval, "n_sample": 1, "skip_first": False},
            )
            loop_running = True

        elif not st.session_state.auto_refresh and loop_running:
            # User unchecked auto-refresh → stop the loop
            _stop_capture_loop()
            loop_running = False

        # ── Display results ──────────────────────────────────────────────────
        if st.session_state.current_image is not None:
            data = st.session_state.current_image
            img = data["img"]
            ellipse_params = data["ellipse_params"]
            cx, cy = data["cx"], data["cy"]
            x_profile, y_profile = data["x_profile"], data["y_profile"]
            x_popt, y_popt = data["x_popt"], data["y_popt"]
            x, y_arr = data["x"], data["y"]

            col1, col2 = st.columns([2, 1])

            with col1:
                pil_img = Image.fromarray(img.astype("uint8"))
                st.image(
                    pil_img,
                    caption=f"Raw CCD Image ({img.shape[1]}×{img.shape[0]})",
                    width="stretch",
                )

            with col2:
                st.markdown("### 📊 Analysis Results")
                st.markdown(f"**Centroid**: ({cx:.1f}, {cy:.1f})")
                st.markdown(
                    f"**Ellipse Center**: ({ellipse_params[0][0]:.1f}, {ellipse_params[0][1]:.1f})"
                )
                st.markdown(
                    f"**Ellipse Axes**: Major={ellipse_params[1][0]:.1f}, Minor={ellipse_params[1][1]:.1f}"
                )
                st.markdown(f"**Rotation Angle**: {ellipse_params[2]:.1f}°")
                st.markdown("---")
                if x_popt is not None:
                    st.markdown("#### X-direction Gaussian Fit")
                    st.metric("Amplitude", f"{x_popt[0]:.2f}")
                    st.metric("Center", f"{x_popt[1]:.2f}")
                    st.metric("σ (Sigma)", f"{x_popt[2]:.2f}")
                if y_popt is not None:
                    st.markdown("#### Y-direction Gaussian Fit")
                    st.metric("Amplitude", f"{y_popt[0]:.2f}")
                    st.metric("Center", f"{y_popt[1]:.2f}")
                    st.metric("σ (Sigma)", f"{y_popt[2]:.2f}")

            # Intensity profile plot
            st.markdown("### 📈 Intensity Profile")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

            ax1.plot(x_profile, "b-", linewidth=1.5, label="X-profile")
            if x_popt is not None:
                x_fit = gaussian(x, *x_popt)
                ax1.plot(
                    x_fit, "r--", linewidth=1.5,
                    label=f"Gaussian Fit (σ={x_popt[2]:.2f})",
                )
            ax1.set_xlabel("X pixels")
            ax1.set_ylabel("Intensity")
            ax1.set_title("X-direction Intensity Profile")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(y_profile, "b-", linewidth=1.5, label="Y-profile")
            if y_popt is not None:
                y_fit = gaussian(y_arr, *y_popt)
                ax2.plot(
                    y_fit, "r--", linewidth=1.5,
                    label=f"Gaussian Fit (σ={y_popt[2]:.2f})",
                )
            ax2.set_xlabel("Y pixels")
            ax2.set_ylabel("Intensity")
            ax2.set_title("Y-direction Intensity Profile")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

        # ── Keep-alive: rerun while capture loop is running (r50 pattern) ───
        if loop_running:
            time.sleep(_REFRESH_INTERVAL)
            st.rerun()

    else:
        # Camera not connected
        st.info("Please connect camera in sidebar to start real-time monitoring")

        st.subheader("Available Cameras")
        try:
            cam_list = DahengCamManager.get_cam_list()
            if cam_list:
                st.write(f"Found {len(cam_list)} Daheng camera device(s)")
                for i, cam_info in enumerate(cam_list):
                    st.write(f"  - Camera {i}: {cam_info}")
            else:
                st.warning("No Daheng camera devices found")
        except Exception as exc:
            st.warning(f"Cannot get camera list: {exc}")

        if MIICamManager is not None:
            try:
                miicam_list = MIICamManager.get_cam_list()
                if miicam_list:
                    st.write(f"Found {len(miicam_list)} MiiCam device(s)")
                    for i, cam_info in enumerate(miicam_list):
                        st.write(f"  - MiiCam {i}: {cam_info}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
