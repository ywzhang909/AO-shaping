"""Unit tests for R50Controller class.

Tests the low-level async TCP controller for R50Power devices.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch

from ao_shaping.drivers.dm.MicroDM import (
    R50Controller,
    MAX_CHANNELS,
    CMD_SET_CHANNEL_VOLTAGE,
    CMD_SET_ALL_CHANNEL_VOLTAGE,
    CMD_SET_ALL_VOLTAGE_BY_ARR,
    CMD_RELAY_ON,
    CMD_RELAY_OFF,
    HEADER,
    FOOTER,
    voltage_to_bytes_clipped,
)

# Import the port listener for protocol verification tests
from tests.ao_shaping.drivers.dm.port_listener import PortListener
PORT_LISTENER_AVAILABLE = True


CONTROLLER_IP = "127.0.0.1"
CONTROLLER_PORT = 10101


class TestR50Controller:
    """Tests for the R50Controller async TCP client."""

    @pytest.fixture
    def controller(self):
        """Create an R50Controller instance for testing."""
        return R50Controller(controller_id=1, ip=CONTROLLER_IP, port=CONTROLLER_PORT)

    def test_initialization(self, controller):
        """Test controller initialization."""
        assert controller.controller_id == 1
        assert controller.ip == CONTROLLER_IP
        assert controller.port == CONTROLLER_PORT
        assert controller._timeout == 10.0  # DEFAULT_TIMEOUT
        assert controller._reader is None
        assert controller._writer is None
        assert controller._connected is False

    def test_is_connected_property(self, controller):
        """Test the is_connected property."""
        assert controller.is_connected is False
        
        controller._connected = True
        controller._writer = Mock()
        assert controller.is_connected is True
        
        controller._writer = None
        assert controller.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_success(self, controller):
        """Test successful connection."""
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()
        
        with patch('asyncio.open_connection', return_value=(mock_reader, mock_writer)) as mock_open:
            result = await controller.connect()
            
            assert result is True
            assert controller._connected is True
            assert controller._reader == mock_reader
            assert controller._writer == mock_writer
            mock_open.assert_called_once_with("127.0.0.1", 10101)

    @pytest.mark.asyncio
    async def test_connect_timeout(self, controller):
        """Test connection timeout."""
        with patch('asyncio.open_connection', side_effect=asyncio.TimeoutError()):
            result = await controller.connect()
            
            assert result is False
            assert controller._connected is False
            assert controller._reader is None
            assert controller._writer is None

    @pytest.mark.asyncio
    async def test_connect_os_error(self, controller):
        """Test connection OS error."""
        with patch('asyncio.open_connection', side_effect=OSError("Network error")):
            result = await controller.connect()
            
            assert result is False
            assert controller._connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, controller):
        """Test disconnection."""
        # Set up connected state
        controller._connected = True
        mock_writer = AsyncMock()
        controller._writer = mock_writer
        controller._reader = AsyncMock()
        
        await controller.disconnect()
        
        assert controller._connected is False
        assert controller._writer is None
        assert controller._reader is None
        mock_writer.close.assert_called_once()
        mock_writer.wait_closed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, controller):
        """Test disconnection when not connected."""
        controller._connected = False
        controller._writer = None
        controller._reader = None
        
        # Should not raise any exception
        await controller.disconnect()
        
        assert controller._connected is False

    @pytest.mark.asyncio
    async def test_send_command_success(self, controller):
        """Test successful command sending."""
        controller._connected = True
        mock_writer = AsyncMock()
        controller._writer = mock_writer
        
        test_data = b'\xAA\xBB\x04\x00\x00\x11\xCC\xDD'
        result = await controller.send_command(test_data)
        
        assert result is True
        mock_writer.write.assert_called_once_with(test_data)
        mock_writer.drain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_command_not_connected(self, controller):
        """Test command sending when not connected."""
        controller._connected = False
        
        result = await controller.send_command(b'test')
        
        assert result is False

    @pytest.mark.asyncio
    async def test_send_command_os_error(self, controller):
        """Test command sending with OS error."""
        controller._connected = True
        mock_writer = AsyncMock()
        mock_writer.drain.side_effect = OSError("Send failed")
        controller._writer = mock_writer
        
        result = await controller.send_command(b'test')
        
        assert result is False
        assert controller._connected is False  # Should be marked as disconnected

    @pytest.mark.asyncio
    async def test_set_all_channel_voltage_success(self, controller):
        """Test setting all channels to same voltage."""
        controller._connected = True
        mock_writer = AsyncMock()
        controller._writer = mock_writer
        
        # Test with 0V
        result = await controller.set_all_channel_voltage(0.0)
        
        assert result is True
        hv, lv = voltage_to_bytes_clipped(0.0)
        expected_cmd = HEADER + bytes([CMD_SET_ALL_CHANNEL_VOLTAGE, hv, lv]) + FOOTER
        mock_writer.write.assert_called_once_with(expected_cmd)
        mock_writer.drain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_set_relay_off(self, controller):
        """Test setting relay to OFF."""
        controller._connected = True
        mock_writer = AsyncMock()
        controller._writer = mock_writer
        
        result = await controller.set_relay(False)
        
        assert result is True
        expected_cmd = HEADER + bytes([CMD_RELAY_OFF]) + FOOTER
        mock_writer.write.assert_called_once_with(expected_cmd)
        mock_writer.drain.assert_awaited_once()


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
        
        # Connect the controller
        connected = await controller.connect()
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
                result = await controller.set_all_channel_voltage(0.0)
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
                
                # Check voltage bytes for 0V (should be 0x16, 0xD1 based on voltage_to_bytes_clipped)
                hv, lv = voltage_to_bytes_clipped(0.0)
                assert cmd_data[3] == hv, f"Expected high byte 0x{hv:02x}, got 0x{cmd_data[3]:02x}"
                assert cmd_data[4] == lv, f"Expected low byte 0x{lv:02x}, got 0x{cmd_data[4]:02x}"
                
            finally:
                # Cleanup controller
                await controller.disconnect()
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
                result = await controller.set_channel_voltage(0, 5.0)
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
                hv, lv = voltage_to_bytes_clipped(5.0)
                assert cmd_data[4] == hv, f"Expected high byte 0x{hv:02x}, got 0x{cmd_data[4]:02x}"
                assert cmd_data[5] == lv, f"Expected low byte 0x{lv:02x}, got 0x{cmd_data[5]:02x}"
                
            finally:
                # Cleanup controller
                await controller.disconnect()
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
                result = await controller.set_all_voltage_array(voltages)
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
                hv, lv = voltage_to_bytes_clipped(0.0)
                for i in range(MAX_CHANNELS):
                    byte_index = 3 + (i * 2)
                    assert cmd_data[byte_index] == hv, \
                        f"Byte pair {i}: expected high byte 0x{hv:02x}, got 0x{cmd_data[byte_index]:02x}"
                    assert cmd_data[byte_index + 1] == lv, \
                        f"Byte pair {i}: expected low byte 0x{lv:02x}, got 0x{cmd_data[byte_index + 1]:02x}"
                
            finally:
                # Cleanup controller
                await controller.disconnect()
        finally:
            # Cleanup port listener
            await port_listener.stop()