"""R50Controller compatibility with ``BaseDM`` and its software readback echo.

The R50Power protocol has **no** readback/query command. ``readback()`` (and
``get_actuator_positions()``) therefore return the last voltage frame the
driver successfully handed to the wire, never a hardware measurement. These
tests pin both halves of that contract: the echo value, and the fact that
reading it emits no frame of its own.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from ao_shaping.drivers.dm import BaseDM
from ao_shaping.drivers.dm.micro import (
    CMD_SET_ALL_VOLTAGE_BY_ARR,
    FOOTER,
    HEADER,
    MAX_CHANNELS,
    MicroDMConnectionError,
    R50Controller,
    VOLTAGE_MAX,
    VOLTAGE_MIN,
    voltages_to_payload,
)

CONTROLLER_IP = "192.168.0.108"
CONTROLLER_PORT = 10108


class TestBaseDmContract:
    """The single-controller client implements ``BaseDM`` directly."""

    def test_is_base_dm(self) -> None:
        assert issubclass(R50Controller, BaseDM)

    def test_channel_model_and_voltage_range(self) -> None:
        assert MAX_CHANNELS == 50
        assert R50Controller.DM_NUM == MAX_CHANNELS
        assert R50Controller.V_Min == VOLTAGE_MIN
        assert R50Controller.V_Max == VOLTAGE_MAX

    def test_is_connected_is_a_method(self) -> None:
        controller = R50Controller(
            controller_id=1, ip=CONTROLLER_IP, port=CONTROLLER_PORT
        )
        assert callable(R50Controller.is_connected)
        assert controller.is_connected() is False


class TestSoftwareReadback:
    """``readback()`` echoes successfully applied commands only."""

    @pytest.fixture
    def controller(self) -> R50Controller:
        ctrl = R50Controller(
            controller_id=1, ip=CONTROLLER_IP, port=CONTROLLER_PORT
        )
        ctrl._socket = Mock()  # fake the TCP connection
        return ctrl

    def test_echoes_last_successful_public_array_setter(
        self, controller: R50Controller
    ) -> None:
        values = np.arange(MAX_CHANNELS, dtype=float)

        assert controller.set_all_voltage_array(values) is True

        echo = controller.readback()
        assert isinstance(echo, np.ndarray)
        assert echo.shape == (MAX_CHANNELS,)
        np.testing.assert_allclose(echo, values)

    def test_echoes_last_successful_public_scalar_setter(
        self, controller: R50Controller
    ) -> None:
        mock_sock = controller._socket
        assert controller.set_all_channel_voltage(12.5) is True
        sends_after_apply = mock_sock.sendall.call_count

        np.testing.assert_allclose(
            controller.readback(), np.full(MAX_CHANNELS, 12.5)
        )
        assert mock_sock.sendall.call_count == sends_after_apply, (
            "readback() must only return the software echo, not send a query"
        )

    def test_echoes_last_successful_single_channel_setter(
        self, controller: R50Controller
    ) -> None:
        values = np.zeros(MAX_CHANNELS)
        channel = 7

        assert controller.set_channel_voltage(channel, 12.5) is True
        values[channel] = 12.5
        np.testing.assert_allclose(controller.readback(), values)

    def test_starts_as_zeros_before_any_send(self) -> None:
        controller = R50Controller(
            controller_id=1, ip=CONTROLLER_IP, port=CONTROLLER_PORT
        )
        np.testing.assert_allclose(
            controller.readback(), np.zeros(MAX_CHANNELS)
        )

    def test_readback_puts_no_frame_on_the_wire(
        self, controller: R50Controller
    ) -> None:
        mock_sock = controller._socket
        controller._apply_voltages(np.full(MAX_CHANNELS, 3.0))
        sends_after_apply = mock_sock.sendall.call_count

        echo = controller.readback()

        assert mock_sock.sendall.call_count == sends_after_apply, (
            "readback() must not emit a query frame: R50Power has no "
            "readback command"
        )
        np.testing.assert_allclose(echo, np.full(MAX_CHANNELS, 3.0))

    def test_readback_returns_a_copy(self, controller: R50Controller) -> None:
        controller._apply_voltages(np.full(MAX_CHANNELS, 1.0))

        echo = controller.readback()
        echo[0] = 99.0

        assert controller.readback()[0] == pytest.approx(1.0)

    def test_failed_send_keeps_previous_echo(
        self, controller: R50Controller
    ) -> None:
        controller._apply_voltages(np.full(MAX_CHANNELS, 5.0))

        controller._socket.sendall.side_effect = OSError("cable unplugged")
        with pytest.raises(MicroDMConnectionError):
            controller._apply_voltages(np.full(MAX_CHANNELS, 10.0))

        np.testing.assert_allclose(
            controller.readback(), np.full(MAX_CHANNELS, 5.0)
        )

    def test_get_actuator_positions_matches_readback(
        self, controller: R50Controller
    ) -> None:
        controller._apply_voltages(np.full(MAX_CHANNELS, 2.0))

        np.testing.assert_allclose(
            controller.get_actuator_positions(), controller.readback()
        )


class TestExactProtocolPreserved:
    """The refactor must not alter the 0x09 frame bytes on the wire."""

    def test_set_all_voltage_array_frame_is_unchanged(self) -> None:
        controller = R50Controller(
            controller_id=1, ip=CONTROLLER_IP, port=CONTROLLER_PORT
        )
        mock_sock = Mock()
        controller._socket = mock_sock
        values = [10.0] * MAX_CHANNELS

        assert controller.set_all_voltage_array(values) is True

        expected = (
            HEADER
            + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR])
            + voltages_to_payload(values)
            + FOOTER
        )
        mock_sock.sendall.assert_called_once_with(expected)
