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
import socketserver
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
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
_REFRESH_INTERVAL = 0.08  # seconds – UI rerun cadence while capture loop runs
_MJPEG_PORT = 0  # 0 = auto-assign available port


# =============================================================================
# MJPEG streaming server (low-latency live preview)
# =============================================================================


class _MJPEGHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves MJPEG frames from a shared queue."""

    streamer: "_MJPEGStreamer | None" = None

    def do_GET(self) -> None:
        if self.path != "/video_feed":
            self.send_error(404)
            return
        if _MJPEGHandler.streamer is None:
            self.send_error(503)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()

        try:
            while not _MJPEGHandler.streamer.stop_event.is_set():
                frame = _MJPEGHandler.streamer.frame_queue.get(timeout=1.0)
                if frame is None:
                    continue
                jpeg = _encode_jpeg(frame)
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(f"Content-Type: image/jpeg\r\n".encode())
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n".encode())
                self.wfile.write(b"\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            pass

    def log_message(self, format, *args):  # noqa: A002
        pass  # suppress server logs


class _MJPEGStreamer:
    """Lightweight MJPEG streaming server for low-latency live preview."""

    def __init__(self) -> None:
        self.frame_queue: queue.Queue[Any] = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None

    def start(self) -> int:
        """Start the MJPEG server. Returns the assigned port."""
        _MJPEGHandler.streamer = self
        self._server = HTTPServer(("127.0.0.1", _MJPEG_PORT), _MJPEGHandler)
        self._port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        logger.info("MJPEG streamer started on port {}", self._port)
        return self._port

    def stop(self) -> None:
        """Stop the MJPEG server."""
        self.stop_event.set()
        if self._server:
            try:
                self._server.shutdown()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        _MJPEGHandler.streamer = None
        logger.info("MJPEG streamer stopped")

    def put_frame(self, frame: np.ndarray | None) -> None:
        """Push a new frame to the stream (non-blocking, drops oldest)."""
        if frame is None:
            return
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.frame_queue.put(frame, block=False)
        except queue.Full:
            pass

    def _serve(self) -> None:
        if self._server:
            self._server.serve_forever(poll_interval=0.05)


_mjpeg_streamer: _MJPEGStreamer | None = None
_mjpeg_port: int | None = None


def _encode_jpeg(img: np.ndarray) -> bytes | None:
    """Encode a grayscale or RGB image to JPEG bytes."""
    try:
        import cv2  # type: ignore[import-untyped]

        if img.ndim == 2:
            success, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        else:
            success, encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 80],
            )
        if success:
            return encoded.tobytes()
    except Exception:
        pass

    # Fallback: use PIL
    try:
        if img.ndim == 2:
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()
    except Exception:
        return None


def _start_mjpeg_streamer() -> int:
    """Start the MJPEG streaming server. Returns the port number."""
    global _mjpeg_streamer, _mjpeg_port
    if _mjpeg_streamer is not None:
        return _mjpeg_port or 0
    _mjpeg_streamer = _MJPEGStreamer()
    _mjpeg_port = _mjpeg_streamer.start()
    return _mjpeg_port


def _stop_mjpeg_streamer() -> None:
    """Stop the MJPEG streaming server."""
    global _mjpeg_streamer, _mjpeg_port
    if _mjpeg_streamer is not None:
        _mjpeg_streamer.stop()
        _mjpeg_streamer = None
        _mjpeg_port = None


def _get_mjpeg_url() -> str | None:
    """Return the MJPEG stream URL, or None if not running."""
    if _mjpeg_port is not None:
        return f"http://127.0.0.1:{_mjpeg_port}/video_feed"
    return None


# =============================================================================
# Background thread capture loop (r50-style daemon thread pattern)
# =============================================================================


class _CaptureLoopResult:
    def __init__(self, analysis: dict[str, Any] | None = None, error: str | None = None):
        self.analysis = analysis
        self.error = error


def _analyze_image(img: np.ndarray) -> dict[str, Any]:
    """Analyze image and return results dict.

    Runs CPU-intensive fits off the main thread so UI reruns stay fast.
    """
    cx, cy = centroid(img, moment=1, threshold=0.01)
    ellipse_params = calculate_enclosing_ellipse(img)
    x_profile, y_profile = get_cross_sections(img, (cx, cy))
    x = np.arange(len(x_profile))
    y_arr = np.arange(len(y_profile))
    x_popt, _ = fit_gaussian(x, x_profile)
    y_popt, _ = fit_gaussian(y_arr, y_profile)
    return {
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


def _camera_capture_tick(
    camera: Any, img_queue: queue.Queue[Any], params: dict[str, Any]
) -> None:
    """One tick of the capture loop: grab a frame, analyze, push to queue.

    Also pushes raw frames to the MJPEG streamer for low-latency live preview.
    """
    try:
        img = camera.get_numpy_image(
            n_sample=params.get("n_sample", 1),
            skip_first=params.get("skip_first", False),
        )
        # Feed raw frame to MJPEG streamer (non-blocking)
        if _mjpeg_streamer is not None:
            _mjpeg_streamer.put_frame(img)
        result = _analyze_image(img)
        # Non-blocking: drop the oldest frame if the queue is full so the main
        # thread always sees the freshest image.
        if img_queue.full():
            try:
                img_queue.get_nowait()
            except queue.Empty:
                pass
        img_queue.put(result, block=False)
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

    Latest analysis result (or error) is stored in ``st.session_state["current_image"]``.
    Called unconditionally at the top of ``main()``.
    """
    q = st.session_state.get("ccd_img_queue")
    if q is None:
        return
    latest_result: dict[str, Any] | None = None
    latest_error: str | None = None
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "error":
            latest_error = item[1]
        elif isinstance(item, dict) and "img" in item:
            latest_result = item
    if latest_error:
        st.session_state["ccd_last_error"] = latest_error
    if latest_result is not None:
        st.session_state["current_image"] = latest_result
        st.session_state["last_update_time"] = time.time()
        st.session_state["ccd_frame_count"] = (
            st.session_state.get("ccd_frame_count", 0) + 1
        )


# =============================================================================
# Session state initialisation
# =============================================================================


def _initialize_camera_state() -> None:
    """Initialise all camera-related session-state keys (idempotent)."""
    st.session_state.setdefault("camera", None)
    st.session_state.setdefault("camera_connected", False)
    st.session_state.setdefault("camera_id", 0)
    st.session_state.setdefault("camera_type", "Daheng")  # "Daheng" | "MiiCam"
    st.session_state.setdefault("miicam_capture_mode", "wait")  # "wait" | "callback"
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
    st.session_state.setdefault("ccd_frame_count", 0)
    st.session_state.setdefault("ccd_fps", 0.0)
    st.session_state.setdefault("ccd_fps_last_time", time.time())
    st.session_state.setdefault("ccd_fps_last_count", 0)


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


def _discover_available_cameras() -> dict[str, list[Any]]:
    """Return a mapping of camera type to list of available devices.

    Example: {"Daheng": [<cam0>, ...], "MiiCam": [<cam0>, ...]}
    """
    result: dict[str, list[Any]] = {"Daheng": [], "MiiCam": []}

    # Daheng
    try:
        daheng_list = DahengCamManager.get_cam_list()
        if daheng_list:
            # Filter out obviously invalid entries (e.g. negative IDs from virtual devices)
            valid = [d for d in daheng_list if getattr(d, "cam_id", -1) >= 0]
            result["Daheng"] = valid
    except Exception:
        pass

    # MiiCam
    if MIICamManager is not None:
        try:
            miicam_list = MIICamManager.get_cam_list()
            if miicam_list:
                result["MiiCam"] = list(miicam_list)
        except Exception:
            pass

    return result


def _create_camera(
    camera_type: str,
    cam_id: int,
    exposure_ms: float,
    capture_mode: str = "wait",
) -> Any:
    """Instantiate the selected camera driver."""
    if camera_type == "MiiCam":
        if MIICamManager is None:
            raise RuntimeError(
                "MiiCam driver is not available. "
                "Ensure the MIICAM SDK is installed (set MIICAM_SDK_PATH)."
            )
        return MIICamManager(
            cam_id=cam_id,
            exposure_time_ms=exposure_ms,
            capture_mode=capture_mode,
        )
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


def _update_fps() -> None:
    """Calculate and store current FPS based on frame count."""
    frame_count = st.session_state.get("ccd_frame_count", 0)
    last_time = st.session_state.get("ccd_fps_last_time", time.time())
    last_count = st.session_state.get("ccd_fps_last_count", 0)

    now = time.time()
    elapsed = now - last_time
    if elapsed >= 1.0:
        fps = (frame_count - last_count) / elapsed
        st.session_state["ccd_fps"] = fps
        st.session_state["ccd_fps_last_time"] = now
        st.session_state["ccd_fps_last_count"] = frame_count


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    st.set_page_config(page_title="CCD Image Analyzer", page_icon="📷", layout="wide")

    _initialize_camera_state()
    _drain_capture_feedback()  # consume latest frame from background thread

    # Calculate FPS
    _update_fps()

    st.title("📷 CCD Real-time Image Analyzer")
    st.markdown("Real-time CCD camera display and beam analysis")

    # ── Sidebar: Camera Settings ─────────────────────────────────────────────
    with st.sidebar:
        st.header("Camera Settings")

        # Discover available cameras and auto-select type
        available_cameras = _discover_available_cameras()
        detected_types = [t for t, devices in available_cameras.items() if devices]

        if detected_types:
            # If the previously selected type is no longer available, pick the first detected
            if st.session_state.camera_type not in detected_types:
                st.session_state.camera_type = detected_types[0]
        else:
            # No cameras detected, keep default
            detected_types = ["Daheng"]

        # Camera type selector (driven by discovered cameras)
        cam_type = st.selectbox(
            "Camera Type",
            options=detected_types,
            index=detected_types.index(st.session_state.camera_type)
            if st.session_state.camera_type in detected_types
            else 0,
            help="Auto-detected camera driver",
            key="ccd_cam_type_select",
        )
        st.session_state.camera_type = cam_type

        # MiiCam-specific: capture mode selector
        if cam_type == "MiiCam" and MIICamManager is not None:
            st.session_state.miicam_capture_mode = st.selectbox(
                "MiiCam Capture Mode",
                options=["wait", "callback"],
                index=0
                if st.session_state.miicam_capture_mode == "wait"
                else 1,
                help="wait=WaitImageV3 (blocking pull), callback=StartPullModeWithCallback + Trigger",
                key="ccd_miicam_capture_mode",
            )

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
                        capture_mode=st.session_state.get("miicam_capture_mode", "wait"),
                    )
                    cam.initialize()
                    st.session_state.camera = cam
                    st.session_state.camera_connected = True
                    # Start MJPEG streamer for low-latency live preview
                    _start_mjpeg_streamer()
                    st.success(
                        f"{st.session_state.camera_type} camera connected"
                    )
                except Exception as exc:
                    st.error(f"Connection failed: {exc}")
        else:
            if st.button("Disconnect Camera", type="secondary"):
                try:
                    _stop_capture_loop()
                    _stop_mjpeg_streamer()
                    _safe_close_camera(st.session_state.camera)
                    st.session_state.camera = None
                    st.session_state.camera_connected = False
                    st.session_state.current_image = None
                    st.session_state["ccd_frame_count"] = 0
                    st.session_state["ccd_fps"] = 0.0
                    st.session_state["ccd_fps_last_time"] = time.time()
                    st.session_state["ccd_fps_last_count"] = 0
                    st.info("Camera disconnected")
                except Exception as exc:
                    logger.error("Disconnect failed: {}", exc)
                    _stop_capture_loop()
                    _stop_mjpeg_streamer()
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
            fps = st.session_state.get("ccd_fps", 0.0)
            if ci is not None:
                capture_time = ci.get("capture_time", 0)
                if capture_time > 0:
                    from datetime import datetime
                    ts = datetime.fromtimestamp(capture_time)
                    st.success(
                        f"✓ Live at {ts.strftime('%H:%M:%S.%f')[:-3]} "
                        f"({fps:.1f} FPS)"
                    )
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
                st.session_state.current_image = _analyze_image(img)
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
                # Use MJPEG stream for low-latency live preview
                mjpeg_url = _get_mjpeg_url()
                if mjpeg_url is not None:
                    st.markdown(
                        f"<img src=\"{mjpeg_url}\" "
                        f"style=\"width: 100%; height: auto;\" />",
                        unsafe_allow_html=True,
                    )
                else:
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
        available_cameras = _discover_available_cameras()
        has_any = False
        for cam_type, cam_list in available_cameras.items():
            if cam_list:
                has_any = True
                st.write(f"Found {len(cam_list)} {cam_type} camera device(s)")
                for i, cam_info in enumerate(cam_list):
                    st.write(f"  - {cam_type} {i}: {cam_info}")
        if not has_any:
            st.warning("No camera devices found. Please connect a camera.")


if __name__ == "__main__":
    main()
