"""MJPEG streaming server (low-latency live preview).

Verbatim extraction from ``ccd_analyzer`` (module-size ratchet split);
``ccd_analyzer`` re-exports the names for existing importers.
"""

from __future__ import annotations

import queue
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
_MJPEG_PORT = 0  # 0 = auto-assign available port


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
                self.wfile.write("Content-Type: image/jpeg\r\n".encode())
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n".encode())
                self.wfile.write(b"\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        # policy: an HTTP handler thread must not die over a vanished client or a partial write (the browser reconnects); queue.Empty is the 1 s frame timeout, OSError is every remaining socket/write failure (broken pipe and reset are matched above)
        except (queue.Empty, OSError):
            pass

    def log_message(self, format, *args):
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
            # policy: teardown is best effort (the socket may already be closed); the stdlib shutdown() only flips a flag and waits on an event, so the failures it can surface are a half-torn-down server object (AttributeError) or a closed socket (OSError); raising out of stop() would leave the streamer thread unreaped
            except (OSError, AttributeError):
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


def _encode_jpeg(img: np.ndarray) -> bytes | None:
    """Encode a grayscale or RGB image to JPEG bytes."""
    try:
        import cv2  # type: ignore[import-untyped]

        if img.ndim == 2:
            success, encoded = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
        else:
            success, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 80],
            )
        if success:
            return encoded.tobytes()
    # policy: first of two encoders: on failure fall through to the Pillow path below instead of reporting an error (cv2.error derives from Exception directly, no stable base class, and this machine cannot import cv2 to confirm the taxonomy)
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
    # policy: preview encoding is best effort -- the MJPEG handler skips the frame when this returns None; Pillow raises OSError/ValueError/TypeError from fromarray + save
    except (OSError, ValueError, TypeError):
        return None
