"""Multi-port TCP mock server for testing MicroDM multi-controller coordination.

Simulates multiple R50Power controllers, each listening on its own TCP port.
Captures all received bytes with timestamps so tests can verify:

1. **Correctness** — that voltage commands are routed to the correct controller
   with the correct channel mapping and byte encoding.
2. **Timeliness** — that commands to all controllers are sent promptly and
   within acceptable latency bounds.

Each controller is a separate ``socket`` server thread. The server records
every received packet with a monotonic timestamp, so tests can analyze
ordering and inter-arrival times.

Example usage::

    server = MultiPortServer(n_controllers=3)
    server.start()
    # ... connect MicroDM to server.ports ...
    server.stop()
    records = server.get_records()
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class ReceivedRecord:
    """A single received packet on one controller port."""

    controller_index: int
    data: bytes
    timestamp: float  # monotonic seconds
    peer_info: str


class ControllerPort:
    """A single TCP server simulating one R50Power controller."""

    def __init__(self, controller_index: int, host: str = "127.0.0.1", port: int = 0):
        self.controller_index = controller_index
        self.host = host
        self.port = port
        self._server_socket: socket.socket | None = None
        self._records: list[ReceivedRecord] = []
        self._is_running = False
        self._listen_thread: threading.Thread | None = None
        self._client_sockets: list[socket.socket] = []

    def start(self) -> None:
        """Start listening on this controller's port."""
        if self._is_running:
            return
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)
        if self.port == 0:
            self.port = self._server_socket.getsockname()[1]
        self._is_running = True
        self._listen_thread = threading.Thread(
            target=self._listen_loop, daemon=True
        )
        self._listen_thread.start()

    def stop(self) -> None:
        """Stop listening and close all client connections."""
        self._is_running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None
        for sock in self._client_sockets:
            try:
                sock.close()
            except Exception:
                pass
        self._client_sockets.clear()
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=1)

    def _listen_loop(self) -> None:
        """Accept connections and spawn client handler threads."""
        while self._is_running:
            try:
                assert self._server_socket is not None
                self._server_socket.settimeout(1.0)
                client_socket, address = self._server_socket.accept()
                if not self._is_running:
                    break
                peer_info = f"{address[0]}:{address[1]}"
                self._client_sockets.append(client_socket)
                t = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, peer_info),
                    daemon=True,
                )
                t.start()
            except socket.timeout:
                continue
            except Exception:
                if self._is_running:
                    break

    def _handle_client(self, client_socket: socket.socket, peer_info: str) -> None:
        """Read data from a client and record it with a timestamp."""
        try:
            while self._is_running:
                data = client_socket.recv(4096)
                if not data:
                    break
                self._records.append(
                    ReceivedRecord(
                        controller_index=self.controller_index,
                        data=data,
                        timestamp=time.monotonic(),
                        peer_info=peer_info,
                    )
                )
        except Exception:
            pass
        finally:
            try:
                client_socket.close()
            except Exception:
                pass
            if client_socket in self._client_sockets:
                self._client_sockets.remove(client_socket)

    def get_records(self) -> list[ReceivedRecord]:
        """Return a copy of the records received on this port."""
        return self._records.copy()

    def clear_records(self) -> None:
        """Clear all records on this port."""
        self._records.clear()


class MultiPortServer:
    """A multi-port TCP server simulating several R50Power controllers.

    Each controller listens on its own port. The server records every
    received packet with a monotonic timestamp so tests can verify
    correctness and timeliness of multi-controller joint control.
    """

    def __init__(self, n_controllers: int = 2, host: str = "127.0.0.1"):
        self.n_controllers = n_controllers
        self.host = host
        self.ports: list[ControllerPort] = []
        self._started = False

    def start(self) -> None:
        """Start all controller ports."""
        if self._started:
            return
        self.ports = [
            ControllerPort(controller_index=i, host=self.host) for i in range(self.n_controllers)
        ]
        for p in self.ports:
            p.start()
        self._started = True

    def stop(self) -> None:
        """Stop all controller ports."""
        for p in self.ports:
            p.stop()
        self._started = False

    def get_records(self) -> list[ReceivedRecord]:
        """Return all records across all controllers, sorted by timestamp."""
        all_records: list[ReceivedRecord] = []
        for p in self.ports:
            all_records.extend(p.get_records())
        all_records.sort(key=lambda r: r.timestamp)
        return all_records

    def get_records_for_controller(self, index: int) -> list[ReceivedRecord]:
        """Return records for a specific controller index."""
        if 0 <= index < len(self.ports):
            return self.ports[index].get_records()
        return []

    def clear_records(self) -> None:
        """Clear all records on all controllers."""
        for p in self.ports:
            p.clear_records()

    def get_ports(self) -> list[int]:
        """Return the actual port numbers for each controller."""
        return [p.port for p in self.ports]

    def get_ips(self) -> list[str]:
        """Return the IP addresses for each controller (host)."""
        return [self.host for _ in self.ports]