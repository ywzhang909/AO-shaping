"""Unit tests for R50Controller class.

Tests the low-level TCP controller for R50Power devices.
"""

from __future__ import annotations

import asyncio
import socket as socket_module
from unittest.mock import Mock, patch

import pytest

from ao_shaping.drivers.dm.MicroDM import (
    CMD_RELAY_OFF,
    CMD_RELAY_ON,
    CMD_SET_ALL_CHANNEL_VOLTAGE,
    CMD_SET_ALL_VOLTAGE_BY_ARR,
    CMD_SET_CHANNEL_VOLTAGE,
    FOOTER,
    HEADER,
    MAX_CHANNELS,
    R50Controller,
    voltages_to_payload,
)

# Import the port listener for protocol verification tests
from tests.ao_shaping.drivers.dm.port_listener import PortListener

PORT_LISTENER_AVAILABLE = True


CONTROLLER_IP = "127.0.0.1"
CONTROLLER_PORT = 10101


class TestR50Controller:
    """Tests for the R50Controller TCP client."""

    @pytest.fixture
    def controller(self):
        """Create an R50Controller instance for testing."""
        return R50Controller(controller_id=1, ip=CONTROLLER_IP, port=CONTROLLER_PORT)
    
    def test_sends(self, controller: R50Controller):
        assert controller.open()
        assert controller.set_relay(True)
        assert controller.set_all_channel_voltage(20)
        assert controller.set_all_voltage_array(list(range(1, 51)))
        assert controller.set_channel_voltage(1, 20)
        assert controller.set_relay(False)
        
    async def test_sends_async(self, controller: R50Controller):
        assert controller.open()
        assert await controller.set_all_voltage_array_async([30.0]*50)
        
    def test_initialization(self, controller):
        """Test controller initialization."""
        assert controller.controller_id == 1
        assert controller.ip == CONTROLLER_IP
        assert controller.port == CONTROLLER_PORT
        assert controller._timeout == 10.0  # DEFAULT_TIMEOUT
        assert controller._socket is None

    def test_is_connected_property(self, controller):
        """Test the is_connected property."""
        assert controller.is_connected is False
        
        controller._socket = Mock()
        assert controller.is_connected is True
        
        controller._socket = None
        assert controller.is_connected is False

    def test_open_success(self, controller):
        """Test successful connection."""
        with patch('socket.socket') as mock_socket_cls:
            mock_sock_instance = Mock()
            mock_socket_cls.return_value = mock_sock_instance
            
            result = controller.open()
            
            assert result is True
            assert controller._socket == mock_sock_instance
            mock_socket_cls.assert_called_once_with(
                socket_module.AF_INET, socket_module.SOCK_STREAM
            )
            mock_sock_instance.settimeout.assert_called_once_with(10.0)
            mock_sock_instance.connect.assert_called_once_with(("127.0.0.1", 10101))

    def test_open_timeout(self, controller):
        """Test connection timeout."""
        with patch('socket.socket') as mock_socket_cls:
            mock_sock_instance = Mock()
            mock_sock_instance.connect.side_effect = socket_module.timeout("timed out")
            mock_socket_cls.return_value = mock_sock_instance
            
            result = controller.open()
            
            assert result is False
            assert controller._socket is None

    def test_open_os_error(self, controller):
        """Test connection OS error."""
        with patch('socket.socket') as mock_socket_cls:
            mock_sock_instance = Mock()
            mock_sock_instance.connect.side_effect = OSError("Network error")
            mock_socket_cls.return_value = mock_sock_instance
            
            result = controller.open()
            
            assert result is False
            assert controller._socket is None

    def test_close(self, controller):
        """Test disconnection."""
        # Set up connected state
        mock_sock = Mock()
        controller._socket = mock_sock
        
        controller.close()
        
        assert controller._socket is None
        mock_sock.close.assert_called_once()

    def test_close_not_connected(self, controller):
        """Test disconnection when not connected."""
        controller._socket = None
        
        # Should not raise any exception
        controller.close()
        
        assert controller._socket is None

    def test_send_command_success(self, controller):
        """Test successful command sending."""
        mock_sock = Mock()
        controller._socket = mock_sock
        
        test_data = b'\xAA\xBB\x04\x00\x00\x11\xCC\xDD'
        result = controller.send_command(test_data)
        
        assert result is True
        mock_sock.sendall.assert_called_once_with(test_data)

    def test_send_command_not_connected(self, controller):
        """Test command sending when not connected."""
        result = controller.send_command(b'test')
        
        assert result is False

    def test_send_command_os_error(self, controller):
        """Test command sending with OS error."""
        mock_sock = Mock()
        mock_sock.sendall.side_effect = OSError("Send failed")
        controller._socket = mock_sock
        
        result = controller.send_command(b'test')
        
        assert result is False
        assert controller._socket is None  # Should be marked as disconnected

    def test_set_all_channel_voltage_success(self, controller):
        """Test setting all channels to same voltage."""
        mock_sock = Mock()
        controller._socket = mock_sock
        
        # Test with 0V
        result = controller.set_all_channel_voltage(0.0)
        
        assert result is True
        payload = voltages_to_payload(0.0)
        expected_cmd = HEADER + bytes([CMD_SET_ALL_CHANNEL_VOLTAGE, payload[0], payload[1]]) + FOOTER
        mock_sock.sendall.assert_called_once_with(expected_cmd)

    def test_set_relay_off(self, controller):
        """Test setting relay to OFF."""
        mock_sock = Mock()
        controller._socket = mock_sock
        
        result = controller.set_relay(False)
        
        assert result is True
        expected_cmd = HEADER + bytes([CMD_RELAY_OFF]) + FOOTER
        mock_sock.sendall.assert_called_once_with(expected_cmd)


@pytest.mark.asyncio
class TestR50ControllerSendCommandAsync:
    """Tests for the send_command_async method (the only async method on R50Controller)."""

    @pytest.fixture
    def controller(self):
        """Create an R50Controller instance for testing."""
        return R50Controller(controller_id=1, ip=CONTROLLER_IP, port=CONTROLLER_PORT)

    async def test_send_command_async_success(self, controller):
        """Test successful async command sending."""
        mock_sock = Mock()
        controller._socket = mock_sock

        test_data = b'\xAA\xBB\x04\x00\x00\x11\xCC\xDD'
        result = await controller.send_command_async(test_data)

        assert result is True
        mock_sock.sendall.assert_called_once_with(test_data)

    async def test_send_command_async_not_connected(self, controller):
        """Test async command sending when not connected."""
        result = await controller.send_command_async(b'test')

        assert result is False

    async def test_send_command_async_os_error(self, controller):
        """Test async command sending with OS error."""
        mock_sock = Mock()
        mock_sock.sendall.side_effect = OSError("Send failed")
        controller._socket = mock_sock

        result = await controller.send_command_async(b'test')

        assert result is False
        assert controller._socket is None  # Should be marked as disconnected

    async def test_send_command_async_returns_coroutine(self, controller):
        """Verify send_command_async returns an awaitable coroutine."""
        mock_sock = Mock()
        controller._socket = mock_sock

        test_data = b'\xAA\xBB\xCC\xDD'
        coro = controller.send_command_async(test_data)

        # Should be a coroutine (not yet awaited)
        assert asyncio.iscoroutine(coro), "send_command_async should return a coroutine"

        result = await coro
        assert result is True
        mock_sock.sendall.assert_called_once_with(test_data)


@pytest.mark.skipif(not PORT_LISTENER_AVAILABLE, reason="PortListener not available")
@pytest.mark.asyncio
class TestR50ControllerProtocol:
    """Tests for R50Controller protocol compliance using a real port listener."""
    
    async def _create_port_listener(self):
        """Create and start a port listener."""
        listener = PortListener()
        await listener.start()
        return listener
    
    async def _create_controller(self, port_listener):
        """Create an R50Controller connected to the port listener."""
        controller = R50Controller(
            controller_id=1,
            ip="127.0.0.1",
            port=port_listener.port,
            timeout=1.0
        )
        
        # Connect the controller (sync call)
        connected = controller.open()
        assert connected, "Failed to connect to port listener"

        return controller

    @pytest.mark.asyncio
    async def test_set_all_channel_voltage_protocol(self):
        """Test that set_all_channel_voltage sends correct protocol bytes."""
        # Create and start port listener
        port_listener = await self._create_port_listener()

        try:
            # Create controller pointing to our listener
            controller = await self._create_controller(port_listener)

            try:
                # Send command to set all channels to 0V
                result = controller.set_all_channel_voltage(0.0)
                assert result is True, "Failed to send command"

                # Give it a moment to transmit
                await asyncio.sleep(0.1)

                # Check what was captured
                commands = port_listener.get_captured_commands()
                assert len(commands) >= 1, f"Expected at least 1 command, got {len(commands)}"

                # Verify the command format
                cmd_data = commands[0].data
                assert cmd_data.startswith(HEADER), f"Missing header: {cmd_data.hex()}"
                assert cmd_data.endswith(FOOTER), f"Missing footer: {cmd_data.hex()}"
                # HEADER(2) + CMD(1) + HV(1) + LV(1) + FOOTER(2) = 7 bytes
                assert len(cmd_data) == 7, f"Expected 7 bytes, got {len(cmd_data)}: {cmd_data.hex()}"

                # Check command byte (should be 0x08 for SET_ALL_CHANNEL_VOLTAGE)
                assert cmd_data[2] == CMD_SET_ALL_CHANNEL_VOLTAGE, \
                    f"Expected command 0x08, got 0x{cmd_data[2]:02x}"

                # Check voltage bytes for 0V (should be 0x16, 0xD1)
                payload = voltages_to_payload(0.0)
                assert cmd_data[3] == payload[0], f"Expected high byte 0x{payload[0]:02x}, got 0x{cmd_data[3]:02x}"
                assert cmd_data[4] == payload[1], f"Expected low byte 0x{payload[1]:02x}, got 0x{cmd_data[4]:02x}"

            finally:
                # Cleanup controller (sync call)
                controller.close()
        finally:
            # Cleanup port listener
            await port_listener.stop()

    @pytest.mark.asyncio
    async def test_set_channel_voltage_protocol(self):
        """Test that set_channel_voltage sends correct protocol bytes."""
        # Create and start port listener
        port_listener = await self._create_port_listener()

        try:
            # Create controller pointing to our listener
            controller = await self._create_controller(port_listener)

            try:
                # Send command to set channel 0 to 5.0V
                result = controller.set_channel_voltage(0, 5.0)
                assert result is True, "Failed to send command"

                # Give it a moment to transmit
                await asyncio.sleep(0.1)

                # Check what was captured
                commands = port_listener.get_captured_commands()
                assert len(commands) >= 1, f"Expected at least 1 command, got {len(commands)}"

                # Verify the command format
                cmd_data = commands[0].data
                assert cmd_data.startswith(HEADER), f"Missing header: {cmd_data.hex()}"
                assert cmd_data.endswith(FOOTER), f"Missing footer: {cmd_data.hex()}"
                assert len(cmd_data) == 8, f"Expected 8 bytes, got {len(cmd_data)}: {cmd_data.hex()}"

                # Check command byte (should be 0x04 for SET_CHANNEL_VOLTAGE)
                assert cmd_data[2] == CMD_SET_CHANNEL_VOLTAGE, \
                    f"Expected command 0x04, got 0x{cmd_data[2]:02x}"

                # Check channel byte (should be 0 for channel 0)
                assert cmd_data[3] == 0, f"Expected channel 0, got {cmd_data[3]}"

                # Check voltage bytes for 5.0V
                payload = voltages_to_payload(5.0)
                assert cmd_data[4] == payload[0], f"Expected high byte 0x{payload[0]:02x}, got 0x{cmd_data[4]:02x}"
                assert cmd_data[5] == payload[1], f"Expected low byte 0x{payload[1]:02x}, got 0x{cmd_data[5]:02x}"

            finally:
                # Cleanup controller (sync call)
                controller.close()
        finally:
            # Cleanup port listener
            await port_listener.stop()

    @pytest.mark.asyncio
    async def test_set_all_voltage_array_protocol(self):
        """Test that set_all_voltage_array sends correct protocol bytes."""
        # Create and start port listener
        port_listener = await self._create_port_listener()

        try:
            # Create controller pointing to our listener
            controller = await self._create_controller(port_listener)

            try:
                # Send command to set all channels to 0V using array
                voltages = [0.0] * MAX_CHANNELS
                result = controller.set_all_voltage_array(voltages)
                assert result is True, "Failed to send command"

                # Give it a moment to transmit
                await asyncio.sleep(0.1)

                # Check what was captured
                commands = port_listener.get_captured_commands()
                assert len(commands) >= 1, f"Expected at least 1 command, got {len(commands)}"

                # Verify the command format
                cmd_data = commands[0].data
                assert cmd_data.startswith(HEADER), f"Missing header: {cmd_data.hex()}"
                assert cmd_data.endswith(FOOTER), f"Missing footer: {cmd_data.hex()}"

                # Check command byte (should be 0x09 for SET_ALL_VOLTAGE_BY_ARR)
                assert cmd_data[2] == CMD_SET_ALL_VOLTAGE_BY_ARR, \
                    f"Expected command 0x09, got 0x{cmd_data[2]:02x}"

                # Should have: header(2) + command(1) + 50*(hv, lv) + footer(2) = 105 bytes
                expected_length = 2 + 1 + (MAX_CHANNELS * 2) + 2
                assert len(cmd_data) == expected_length, \
                    f"Expected {expected_length} bytes, got {len(cmd_data)}: {cmd_data.hex()}"

                # Verify each pair of bytes corresponds to 0V
                payload = voltages_to_payload(0.0)
                for i in range(MAX_CHANNELS):
                    byte_index = 3 + (i * 2)
                    assert cmd_data[byte_index] == payload[0], \
                        f"Byte pair {i}: expected high byte 0x{payload[0]:02x}, got 0x{cmd_data[byte_index]:02x}"
                    assert cmd_data[byte_index + 1] == payload[1], \
                        f"Byte pair {i}: expected low byte 0x{payload[1]:02x}, got 0x{cmd_data[byte_index + 1]:02x}"

            finally:
                # Cleanup controller (sync call)
                controller.close()
        finally:
            # Cleanup port listener
            await port_listener.stop()

    @pytest.mark.asyncio
    async def test_send_command_async_protocol(self):
        """Test that send_command_async sends correct protocol bytes."""
        # Create and start port listener
        port_listener = await self._create_port_listener()

        try:
            # Create controller pointing to our listener
            controller = await self._create_controller(port_listener)

            try:
                # Build a valid command: set all channels to 0V
                payload = voltages_to_payload(0.0)
                cmd = HEADER + bytes([CMD_SET_ALL_CHANNEL_VOLTAGE, payload[0], payload[1]]) + FOOTER

                # Send via the async method
                result = await controller.send_command_async(cmd)
                assert result is True, "Failed to send command via send_command_async"

                # Give it a moment to transmit
                await asyncio.sleep(0.1)

                # Check what was captured
                commands = port_listener.get_captured_commands()
                assert len(commands) >= 1, f"Expected at least 1 command, got {len(commands)}"

                # Verify the exact raw bytes match what we sent
                cmd_data = commands[0].data
                assert cmd_data == cmd, (
                    f"Sent bytes differ from captured:\n"
                    f"  sent:     {cmd.hex()}\n"
                    f"  captured: {cmd_data.hex()}"
                )

                # Verify protocol compliance
                assert cmd_data.startswith(HEADER), f"Missing header: {cmd_data.hex()}"
                assert cmd_data.endswith(FOOTER), f"Missing footer: {cmd_data.hex()}"
                assert len(cmd_data) == 7, f"Expected 7 bytes, got {len(cmd_data)}: {cmd_data.hex()}"
                assert cmd_data[2] == CMD_SET_ALL_CHANNEL_VOLTAGE, \
                    f"Expected command 0x08, got 0x{cmd_data[2]:02x}"
                assert cmd_data[3] == payload[0], f"Expected high byte 0x{payload[0]:02x}, got 0x{cmd_data[3]:02x}"
                assert cmd_data[4] == payload[1], f"Expected low byte 0x{payload[1]:02x}, got 0x{cmd_data[4]:02x}"

            finally:
                # Cleanup controller (sync call)
                controller.close()
        finally:
            # Cleanup port listener
            await port_listener.stop()