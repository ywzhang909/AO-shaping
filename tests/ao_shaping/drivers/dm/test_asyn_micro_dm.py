"""Unit tests for AsyncMicroDM driver.

Tests VoltageConverter LUT, SendResult dataclass, AsyncR50Controller
(async TCP client), and AsyncMicroDM (multi-controller DM driver).

All tests use mocks — no real TCP connections or hardware required.
"""

from __future__ import annotations

import asyncio
import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from ao_shaping.drivers.dm.MicroDM import voltages_to_payload

# Import from the module under test.
# NOTE: WiringMap is re-exported from asyn_micro_dm for convenience,
# but we import the canonical class from MicroDM for assertion.
from ao_shaping.drivers.dm.asyn_micro_dm import (
    AsyncMicroDM,
    AsyncR50Controller,
    SendResult,
    VoltageConverter,
)


# =============================================================================
# VoltageConverter
# =============================================================================


class TestAsyncVoltageConverter:
    """Validate that the LUT produces byte pairs identical to voltages_to_payload."""

    @pytest.fixture()
    def converter(self) -> VoltageConverter:
        return VoltageConverter()

    def test_voltage_minus_20(self, converter: VoltageConverter):
        """At -20 V the LUT should give (0, 0), matching voltages_to_payload."""
        ref = voltages_to_payload(-20.0)
        hi, lo = converter.convert_single(-20.0)
        assert hi == ref[0] == 0
        assert lo == ref[1] == 0

    def test_voltage_zero(self, converter: VoltageConverter):
        """At 0 V the LUT should match voltages_to_payload(0.0)."""
        ref = voltages_to_payload(0.0)
        hi, lo = converter.convert_single(0.0)
        assert hi == ref[0] == 22
        assert lo == ref[1] == 209

    def test_voltage_120(self, converter: VoltageConverter):
        """At 120 V the LUT should match voltages_to_payload(120.0)."""
        ref = voltages_to_payload(120.0)
        hi, lo = converter.convert_single(120.0)
        assert hi == ref[0] == 159
        assert lo == ref[1] == 182

    def test_voltage_clipped_low(self, converter: VoltageConverter):
        """Values below -20 V should clip to -20 V output."""
        ref = voltages_to_payload(-30.0)
        hi, lo = converter.convert_single(-30.0)
        assert hi == ref[0] == 0
        assert lo == ref[1] == 0

    def test_voltage_clipped_high(self, converter: VoltageConverter):
        """Values above 120 V should clip to 120 V output."""
        ref = voltages_to_payload(150.0)
        hi, lo = converter.convert_single(150.0)
        assert hi == ref[0] == 159
        assert lo == ref[1] == 182

    def test_fill_buffer_matches_reference(self, converter: VoltageConverter):
        """fill_buffer for 50 channels should produce the same bytes as voltages_to_payload."""
        voltages = np.linspace(-20.0, 120.0, 50)
        buf = bytearray(100)
        converter.fill_buffer(voltages, buf)

        ref = voltages_to_payload(voltages)
        assert bytes(buf) == ref


# =============================================================================
# SendResult
# =============================================================================


class TestSendResult:
    """Tests for the frozen SendResult dataclass."""

    def test_frozen_dataclass(self):
        """SendResult instances should be immutable (frozen=True)."""
        r = SendResult(success=True)
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.success = False  # type: ignore[misc]

    def test_latency_us_default(self):
        """Default latency_us should be 0."""
        r = SendResult(success=True)
        assert r.latency_us == 0.0

    def test_all_fields(self):
        """All fields should store the provided values."""
        r = SendResult(success=False, error="drain_timeout", latency_us=123.4)
        assert r.success is False
        assert r.error == "drain_timeout"
        assert r.latency_us == 123.4


# =============================================================================
# AsyncR50Controller
# =============================================================================


@pytest.fixture()
def controller() -> AsyncR50Controller:
    """Create a controller with an unconnected state."""
    return AsyncR50Controller(controller_id=1, ip="192.168.0.101", port=10101)


def _make_mock_writer() -> MagicMock:
    """Create a mock asyncio.StreamWriter."""
    writer = MagicMock()
    writer.is_closing.return_value = False
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return writer


class TestAsyncR50Controller:
    """Tests for AsyncR50Controller with mocked asyncio streams."""

    @pytest.mark.asyncio
    async def test_connect_success(self, controller: AsyncR50Controller):
        """Successful connect should set is_connected = True."""
        mock_reader = MagicMock()
        mock_writer = _make_mock_writer()

        with patch("ao_shaping.drivers.dm.asyn_micro_dm.asyncio.open_connection",
                    new_callable=AsyncMock, return_value=(mock_reader, mock_writer)):
            result = await controller.connect()

        assert result is True
        assert controller.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self, controller: AsyncR50Controller):
        """Failed connect (exception) should return False."""
        with patch("ao_shaping.drivers.dm.asyn_micro_dm.asyncio.open_connection",
                    new_callable=AsyncMock, side_effect=OSError("refused")):
            result = await controller.connect()

        assert result is False
        assert controller.is_connected is False

    @pytest.mark.asyncio
    async def test_send_voltages_success(self, controller: AsyncR50Controller):
        """send_voltages on a connected controller should return success=True."""
        controller._writer = _make_mock_writer()
        voltages = np.zeros(50)

        result = await controller.send_voltages(voltages)

        assert result.success is True
        assert result.error is None
        assert result.latency_us > 0
        controller._writer.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_voltages_not_connected(self, controller: AsyncR50Controller):
        """send_voltages when not connected should return error='not_connected'."""
        result = await controller.send_voltages(np.zeros(50))

        assert result.success is False
        assert result.error == "not_connected"

    @pytest.mark.asyncio
    async def test_send_voltages_drain_timeout(self, controller: AsyncR50Controller):
        """A drain() TimeoutError should produce error='drain_timeout'."""
        writer = _make_mock_writer()
        writer.drain = AsyncMock(side_effect=asyncio.TimeoutError)
        controller._writer = writer

        result = await controller.send_voltages(np.zeros(50))

        assert result.success is False
        assert result.error == "drain_timeout"

    @pytest.mark.asyncio
    async def test_disconnect(self, controller: AsyncR50Controller):
        """disconnect should close the writer and clear state."""
        writer = _make_mock_writer()
        controller._writer = writer
        controller._reader = MagicMock()

        await controller.disconnect()

        writer.close.assert_called_once()
        writer.wait_closed.assert_awaited_once()
        assert controller._writer is None
        assert controller._reader is None
        assert controller.is_connected is False


# =============================================================================
# AsyncMicroDM
# =============================================================================


@pytest.fixture()
def mock_dm() -> AsyncMicroDM:
    """Create an AsyncMicroDM with mocked controllers (no network)."""
    dm = AsyncMicroDM(ips=["192.168.0.101", "192.168.0.102", "192.168.0.103"])
    # Replace real controllers with mocks
    dm._controllers = []
    for i in range(3):
        ctrl = AsyncMock(spec=AsyncR50Controller)
        ctrl.controller_id = i + 1
        ctrl.is_connected = True
        ctrl.send_voltages = AsyncMock(return_value=SendResult(success=True))
        ctrl.connect = AsyncMock(return_value=True)
        ctrl.disconnect = AsyncMock()
        ctrl.send_relay = AsyncMock(return_value=SendResult(success=True))
        dm._controllers.append(ctrl)
    return dm


class TestAsyncMicroDM:
    """Tests for the AsyncMicroDM driver with mock controllers."""

    def test_init_default(self):
        """Default construction should set DM_Num, V_Min, V_Max correctly."""
        dm = AsyncMicroDM()
        assert dm.DM_Num == 39 * 39
        assert dm.V_Min == -20.0
        assert dm.V_Max == 120.0
        assert len(dm._controllers) == 1
        assert dm._controllers[0].ip == "192.168.0.101"

    def test_init_multiple_ips(self):
        """Passing multiple IPs should create multiple controllers."""
        dm = AsyncMicroDM(ips=["10.0.0.1", "10.0.0.2"])
        assert len(dm._controllers) == 2

    @pytest.mark.asyncio
    async def test_connect_all_success(self, mock_dm: AsyncMicroDM):
        """connect_all should return all-success when every controller connects."""
        results = await mock_dm.connect_all()

        assert all(results.values())
        for ctrl in mock_dm._controllers:
            ctrl.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_partial_failure(self, mock_dm: AsyncMicroDM):
        """connect_all should report partial failures."""
        mock_dm._controllers[1].connect = AsyncMock(return_value=False)

        results = await mock_dm.connect_all()

        assert results[1] is True
        assert results[2] is False
        assert results[3] is True

    @pytest.mark.asyncio
    async def test_send_frame(self, mock_dm: AsyncMicroDM):
        """send_frame should dispatch voltages to all controllers."""
        vs = np.linspace(-10.0, 80.0, 1521)
        results = await mock_dm.send_frame(vs)

        assert len(results) == 3
        for r in results:
            assert r.success is True
        # Voltages should be stored
        np.testing.assert_array_equal(mock_dm._last_voltages, vs)

    @pytest.mark.asyncio
    async def test_send_frame_uses_last_voltages(self, mock_dm: AsyncMicroDM):
        """send_frame(None) should re-send the stored _last_voltages."""
        mock_dm._last_voltages = np.full(1521, 42.0)
        results = await mock_dm.send_frame()

        assert len(results) == 3
        for r in results:
            assert r.success is True

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_dm: AsyncMicroDM):
        """shutdown should home voltages, relay off, then disconnect."""
        mock_dm._open = True
        await mock_dm.shutdown(home_voltage=0.0)

        for ctrl in mock_dm._controllers:
            ctrl.send_voltages.assert_awaited_once()
            ctrl.send_relay.assert_awaited_once_with(False)
            ctrl.disconnect.assert_awaited_once()

        assert mock_dm._open is False
        np.testing.assert_array_equal(mock_dm._last_voltages, np.zeros(1521))

    def test_sync_open(self, mock_dm: AsyncMicroDM):
        """sync open() should bridge to async and set _open = True."""
        with patch.object(mock_dm, "_run_async") as mock_run:
            mock_run.return_value = None
            mock_dm.open()

            mock_run.assert_called_once()
            # After _async_open completes, _open should be True
            mock_dm._open = True  # Simulate what _async_open does
            assert mock_dm._open is True

    def test_sync_close(self, mock_dm: AsyncMicroDM):
        """sync close() should bridge to async and clear _open."""
        mock_dm._open = True
        with patch.object(mock_dm, "_run_async") as mock_run:
            mock_run.return_value = None
            mock_dm.close()

            mock_run.assert_called_once()
            mock_dm._open = False
            assert mock_dm._open is False

    def test_apply_voltages(self, mock_dm: AsyncMicroDM):
        """_apply_voltages should clip and store voltages."""
        vs = np.array([-30.0, 0.0, 60.0, 150.0] + [0.0] * (1521 - 4))
        result = mock_dm._apply_voltages(vs)

        # -30 → -20 (clipped), 150 → 120 (clipped)
        assert result[0] == -20.0
        assert result[1] == 0.0
        assert result[2] == 60.0
        assert result[3] == 120.0
        np.testing.assert_array_equal(result, mock_dm._last_voltages)

    def test_get_actuator_positions(self, mock_dm: AsyncMicroDM):
        """get_actuator_positions should return a copy of _last_voltages."""
        mock_dm._last_voltages = np.full(1521, 50.0)
        pos = mock_dm.get_actuator_positions()

        np.testing.assert_array_equal(pos, np.full(1521, 50.0))
        # Returned array should be a copy (not same object)
        pos[0] = 999.0
        assert mock_dm._last_voltages[0] == 50.0

    def test_transform(self, mock_dm: AsyncMicroDM):
        """transform should map [-1, 1] → [V_Min, V_Max] linearly."""
        cmd = np.array([-1.0, 0.0, 1.0])
        result = mock_dm.transform(cmd)

        expected = np.array([-20.0, 50.0, 120.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_clipping(self, mock_dm: AsyncMicroDM):
        """transform should clip input outside [-1, 1]."""
        cmd = np.array([-5.0, 5.0])
        result = mock_dm.transform(cmd)

        expected = np.array([-20.0, 120.0])
        np.testing.assert_array_equal(result, expected)

    def test_context_manager_sync(self, mock_dm: AsyncMicroDM):
        """__enter__ / __exit__ should call open/close via _run_async."""
        with patch.object(mock_dm, "_run_async") as mock_run:
            mock_run.return_value = None
            with mock_dm:
                pass

            # open + close = 2 calls
            assert mock_run.call_count == 2

    def test_is_connected(self, mock_dm: AsyncMicroDM):
        """is_connected should reflect _open and controller status."""
        mock_dm._open = False
        assert mock_dm.is_connected() is False

        mock_dm._open = True
        assert mock_dm.is_connected() is True


# =============================================================================
# WiringMap Re-export
# =============================================================================


class TestWiringMapReuse:
    """Verify WiringMap is importable from asyn_micro_dm (re-exported from MicroDM)."""

    def test_import_wiring_map(self):
        """WiringMap should be importable from asyn_micro_dm."""
        from ao_shaping.drivers.dm.asyn_micro_dm import WiringMap as WMFromAsync
        from ao_shaping.drivers.dm.MicroDM import WiringMap as WMFromMicro

        assert WMFromAsync is WMFromMicro


# =============================================================================
# Registry Integration
# =============================================================================


class TestRegistry:
    """Verify the @register_dm decorator and create_dm factory work."""

    def test_register_dm(self):
        """AsyncMicroDM should be registered under 'asyn_micro'."""
        from ao_shaping.drivers.dm._registry import get_dm_registry

        registry = get_dm_registry()
        assert registry.has_type("asyn_micro")
        assert registry.get_class("asyn_micro") is AsyncMicroDM

    def test_create_dm_factory(self):
        """create_dm('asyn_micro', ips=[...]) should return an AsyncMicroDM."""
        from ao_shaping.drivers.dm._registry import create_dm

        dm = create_dm("asyn_micro", ips=["192.168.0.101"])
        assert isinstance(dm, AsyncMicroDM)
        assert len(dm._controllers) == 1
