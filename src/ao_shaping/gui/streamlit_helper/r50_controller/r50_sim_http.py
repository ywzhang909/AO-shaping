"""HTTP interface for the R50 simulated controllers.

The simulated controllers (``SimulatedR50Controller`` / ``SimulatedMicroDM``)
are in-memory twins of the real TCP R50Power hardware. To make the simulation
observable and drivable without the Streamlit UI — and to let tests and
scripts verify it with any HTTP client — each simulated controller can expose
a tiny JSON API bound to ``127.0.0.1``.

Only the Python standard library is used (``http.server`` + ``urllib``), so no
new dependencies are introduced and the module stays importable inside the
spawned service process.

API (JSON):

    GET  /health        -> {"ok": true, "service": "r50-sim", "ip": ..., "port": ...}
    GET  /status        -> {"opened": bool, "relay_on": bool, "connected": bool,
                            "ip": str, "port": int, "voltages": [50 floats]}
    POST /relay         {"on": bool}                            -> {"ok": true, "relay_on": bool}
    POST /voltage       {"channel": int, "voltage": float}      -> {"ok": true, "applied": true}
    POST /voltage       {"voltages": [float, ... 50]}           -> {"ok": true, "applied": 50}

Client helpers use ``urllib.request`` so any script/test can query a running
simulation without installing ``requests``.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from loguru import logger

# One live listener per bound port. When a new controller starts on a port
# that already has a running server (e.g. an earlier instance of the same sim
# IP left one behind), the listener is reused and serves the newest controller,
# so 18000+ip_suffix stays a stable endpoint for the whole process.
_PORT_SERVERS: dict[int, "_SimHttpServer"] = {}


class SimHttpServer:
    """Small JSON HTTP server wrapping one simulated controller.

    Serves on a background daemon thread so it never blocks the asyncio loop
    of the control service. ``close()`` on the wrapped controller stops it.
    """

    def __init__(self, ctrl: Any, port: int = 0) -> None:
        self.ctrl = ctrl
        self.port = int(port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._server is not None

    def start(self) -> int:
        """Bind and start serving on a daemon thread; returns the bound port.

        When a live listener already exists for the requested port, it is
        reused and serves this controller (stable endpoint per sim IP). If the
        port is occupied by a foreign process, falls back to an OS-assigned
        free port so the simulation stays usable.
        """
        if self._server is not None:
            return self.port
        existing = _PORT_SERVERS.get(self.port)
        if existing is not None:
            existing.ctrl = self.ctrl
            logger.info(
                f"[SIM {self._ip_label()}] reusing HTTP API on port {self.port} "
                f"(serving latest controller)"
            )
            return self.port
        try:
            self._server = _SimHttpServer(("127.0.0.1", self.port), self.ctrl)
        except OSError as exc:
            logger.warning(f"[SIM {self._ip_label()}] port {self.port} busy ({exc}), using ephemeral port")
            self._server = _SimHttpServer(("127.0.0.1", 0), self.ctrl)
        self.port = int(self._server.server_address[1])
        _PORT_SERVERS[self.port] = self._server
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name=f"sim-http-{self.port}",
        )
        self._thread.start()
        logger.info(f"[SIM {self._ip_label()}] HTTP API ready: http://127.0.0.1:{self.port}/status")
        return self.port

    def stop(self) -> None:
        """Shut the server down and release the listening socket.

        A reused listener is left running (it serves newer controllers); only
        the owning server is actually shut down.
        """
        if self._server is None:
            return
        if _PORT_SERVERS.get(self.port) is self._server:
            _PORT_SERVERS.pop(self.port, None)
        try:
            self._server.shutdown()
            self._server.server_close()
        except OSError:
            pass
        self._server = None
        self._thread = None
        logger.info(f"[SIM {self._ip_label()}] HTTP API stopped")

    def _ip_label(self) -> str:
        return str(getattr(self.ctrl, "ip", "?"))


class _SimHttpServer(ThreadingHTTPServer):
    """ThreadingHTTPServer carrying a reference to the wrapped controller."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], ctrl: Any) -> None:
        self.ctrl = ctrl
        super().__init__(server_address, _SimRequestHandler)


class _SimRequestHandler(BaseHTTPRequestHandler):
    """JSON request handler for the simulated controller API."""

    server: _SimHttpServer  # type: ignore[assignment] — set by ThreadingHTTPServer

    # quiet the default stderr logging of BaseHTTPRequestHandler
    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug(f"[SIM-HTTP {self.server.server_address[1]}] {fmt % args}")

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        ctrl = self.server.ctrl
        path = urllib.parse.urlparse(self.path).path
        if path == "/health":
            self._send_json(
                {
                    "ok": True,
                    "service": "r50-sim",
                    "ip": str(getattr(ctrl, "ip", "?")),
                    "port": int(getattr(ctrl, "port", 0)),
                }
            )
            return
        if path == "/status":
            voltages = []
            readback = getattr(ctrl, "readback", None)
            if callable(readback):
                raw: Any = readback()
                voltages = [float(v) for v in raw]
            self._send_json(
                {
                    "opened": bool(getattr(ctrl, "_opened", False)),
                    "relay_on": bool(getattr(ctrl, "_relay_on", False)),
                    "connected": bool(ctrl.is_connected() if callable(getattr(ctrl, "is_connected", None)) else True),
                    "ip": str(getattr(ctrl, "ip", "?")),
                    "port": int(getattr(ctrl, "port", 0)),
                    "voltages": voltages,
                }
            )
            return
        self._send_json({"ok": False, "error": f"unknown path: {path}"}, status=404)

    def do_POST(self) -> None:
        ctrl = self.server.ctrl
        path = urllib.parse.urlparse(self.path).path
        try:
            payload = self._read_json()
        except (ValueError, json.JSONDecodeError):
            self._send_json({"ok": False, "error": "invalid JSON body"}, status=400)
            return
        if path == "/relay":
            on = bool(payload.get("on", False))
            fn = getattr(ctrl, "set_relay", None)
            if fn is None:
                self._send_json({"ok": False, "error": "controller has no set_relay"}, status=400)
                return
            result = fn(on)
            ok = True if result is None else bool(result)
            self._send_json({"ok": ok, "relay_on": bool(getattr(ctrl, "_relay_on", on))}, status=200 if ok else 500)
            return
        if path == "/voltage":
            if "voltages" in payload:
                fn = getattr(ctrl, "set_all_voltage_array", None)
                if fn is None:
                    self._send_json({"ok": False, "error": "controller has no set_all_voltage_array"}, status=400)
                    return
                ok = bool(fn([float(v) for v in payload["voltages"]]))
                self._send_json({"ok": ok, "applied": len(payload["voltages"]) if ok else 0}, status=200 if ok else 500)
                return
            if "channel" in payload and "voltage" in payload:
                fn = getattr(ctrl, "set_channel_voltage", None)
                if fn is None:
                    self._send_json({"ok": False, "error": "controller has no set_channel_voltage"}, status=400)
                    return
                ok = bool(fn(int(payload["channel"]), float(payload["voltage"])))
                self._send_json({"ok": ok, "applied": ok}, status=200 if ok else 500)
                return
            self._send_json({"ok": False, "error": "need 'voltages' or 'channel'+'voltage'"}, status=400)
            return
        self._send_json({"ok": False, "error": f"unknown path: {path}"}, status=404)


# =============================================================================
# Client helpers (stdlib urllib — usable from any script/test)
# =============================================================================


def sim_http_request(port: int, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 2.0) -> dict[str, Any]:
    """Issue a JSON request to a simulation HTTP endpoint; returns parsed JSON."""
    url = f"http://127.0.0.1:{int(port)}{path}"
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — localhost-only sim endpoint
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"ok": False, "error": f"HTTP {exc.code}: {body}"}


def get_sim_status(port: int, timeout: float = 2.0) -> dict[str, Any]:
    """Convenience: GET /status of a simulation controller."""
    return sim_http_request(port, "GET", "/status", timeout=timeout)
