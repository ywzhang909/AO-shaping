"""Simulation HTTP API tests.

Simulated controllers expose a localhost JSON API (stdlib http.server) so the
simulation can be verified with any HTTP client. These tests exercise the
server through the urllib client helpers against a real bound socket.
"""

from __future__ import annotations

import socket

import numpy as np
import pytest

from ao_shaping.gui.streamlit_helper.r50_controller.r50_channel_select import (
    CFG,
    SINGLE_CHANNELS,
)
from ao_shaping.gui.streamlit_helper.r50_controller.r50_connection import (
    SimulatedMicroDM,
    SimulatedR50Controller,
)
from ao_shaping.gui.streamlit_helper.r50_controller.r50_sim_http import (
    get_sim_status,
    sim_http_request,
)


def _open_ctrl(ip: str = "192.168.0.101", port: int | None = None) -> SimulatedR50Controller:
    ctrl = SimulatedR50Controller(101, ip, port or CFG.DEFAULT_PORT)
    ctrl.open()
    return ctrl


def _serve(ctrl: SimulatedR50Controller, port: int | None = None) -> int:
    http_port = ctrl.start_http_server(port)
    assert sim_http_request(http_port, "GET", "/health")["ok"] is True
    return http_port


class TestHttpStatus:
    def test_status_reports_initial_state(self) -> None:
        ctrl = _open_ctrl()
        http_port = _serve(ctrl)
        try:
            status = get_sim_status(http_port)
            assert status["opened"] is True
            assert status["connected"] is True
            assert status["relay_on"] is False
            assert status["ip"] == "192.168.0.101"
            assert len(status["voltages"]) == SINGLE_CHANNELS
            assert all(v == 0.0 for v in status["voltages"])
        finally:
            ctrl.close()

    def test_status_reflects_voltage_and_relay(self) -> None:
        ctrl = _open_ctrl()
        http_port = _serve(ctrl)
        try:
            assert sim_http_request(http_port, "POST", "/relay", {"on": True})["ok"] is True
            assert sim_http_request(http_port, "POST", "/voltage", {"channel": 4, "voltage": 12.5})["ok"] is True
            status = get_sim_status(http_port)
            assert status["relay_on"] is True
            assert status["voltages"][4] == pytest.approx(12.5)
        finally:
            ctrl.close()


class TestHttpVoltage:
    def test_post_channel_voltage(self) -> None:
        ctrl = _open_ctrl()
        http_port = _serve(ctrl)
        try:
            resp = sim_http_request(http_port, "POST", "/voltage", {"channel": 0, "voltage": -5.0})
            assert resp["ok"] is True
            assert ctrl.readback()[0] == pytest.approx(-5.0)
        finally:
            ctrl.close()

    def test_post_full_array(self) -> None:
        ctrl = _open_ctrl()
        http_port = _serve(ctrl)
        try:
            volts = [float(i) for i in range(SINGLE_CHANNELS)]
            resp = sim_http_request(http_port, "POST", "/voltage", {"voltages": volts})
            assert resp["ok"] is True
            assert resp["applied"] == SINGLE_CHANNELS
            assert np.allclose(ctrl.readback(), np.arange(SINGLE_CHANNELS, dtype=np.float64))
        finally:
            ctrl.close()

    def test_post_invalid_payload_rejected(self) -> None:
        ctrl = _open_ctrl()
        http_port = _serve(ctrl)
        try:
            resp = sim_http_request(http_port, "POST", "/voltage", {"bogus": 1})
            assert resp["ok"] is False
        finally:
            ctrl.close()


class TestHttpRelayAndRoutes:
    def test_relay_on_off(self) -> None:
        ctrl = _open_ctrl()
        http_port = _serve(ctrl)
        try:
            assert sim_http_request(http_port, "POST", "/relay", {"on": True})["relay_on"] is True
            assert sim_http_request(http_port, "POST", "/relay", {"on": False})["relay_on"] is False
            assert ctrl._relay_on is False
        finally:
            ctrl.close()

    def test_unknown_path_returns_404(self) -> None:
        ctrl = _open_ctrl()
        http_port = _serve(ctrl)
        try:
            resp = sim_http_request(http_port, "GET", "/nope")
            assert resp["ok"] is False
        finally:
            ctrl.close()


class TestHttpPortSelection:
    def test_default_port_derived_from_ip_suffix(self) -> None:
        ctrl = _open_ctrl(ip="192.168.0.123")
        http_port = _serve(ctrl)
        try:
            assert http_port == 18000 + 123
        finally:
            ctrl.close()

    def test_explicit_port_respected(self) -> None:
        ctrl = _open_ctrl(ip="192.168.0.101")
        http_port = _serve(ctrl, port=19001)
        try:
            assert http_port == 19001
        finally:
            ctrl.close()

    def test_falls_back_to_ephemeral_when_derived_port_busy(self) -> None:
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", 18105))
        blocker.listen(1)
        try:
            ctrl = _open_ctrl(ip="192.168.0.105")
            http_port = _serve(ctrl)
            try:
                assert http_port != 18105
                assert get_sim_status(http_port)["opened"] is True
            finally:
                ctrl.close()
        finally:
            blocker.close()

    def test_stop_http_server_releases_port(self) -> None:
        ctrl = _open_ctrl(ip="192.168.0.102")
        http_port = ctrl.start_http_server()
        assert http_port == 18102
        ctrl.stop_http_server()
        assert ctrl.http_port is None
        # rebinding the same port must succeed after stop
        ctrl2 = _open_ctrl(ip="192.168.0.102")
        try:
            assert ctrl2.start_http_server() == 18102
        finally:
            ctrl2.close()


class TestSimulatedMicroDMHttp:
    def test_http_ports_per_ip(self) -> None:
        dm = SimulatedMicroDM([101, 102])
        dm.open()
        try:
            ports = dm.start_http_servers()
            assert set(ports) == {101, 102}
            assert ports[101] == 18101
            assert get_sim_status(ports[102])["ip"] == "192.168.0.102"
        finally:
            dm.close()

    def test_close_stops_servers(self) -> None:
        dm = SimulatedMicroDM([103])
        dm.open()
        dm.start_http_servers()
        assert dm.http_ports == {103: 18103}
        dm.close()
        assert dm.http_ports == {}
