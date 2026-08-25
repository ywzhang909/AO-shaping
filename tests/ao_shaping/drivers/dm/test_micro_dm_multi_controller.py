"""Tests for MicroDM multi-controller joint control correctness and timeliness.

These tests use a multi-port TCP mock server to simulate multiple R50Power
controllers. The server records every received packet with a monotonic
timestamp, allowing verification of:

1. **Correctness** — voltage commands are routed to the correct controller
   with the correct channel mapping and byte encoding.
2. **Timeliness** — commands to all controllers are sent promptly and
   within acceptable latency bounds.

The mock server is started on localhost with random ports, and the MicroDM
driver is pointed at those ports via the ``ips`` parameter.
"""

from __future__ import annotations

import time

import numpy as np
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
    MicroDM,
    MicroDMVoltageError,
    voltages_to_payload,
)
from tests.ao_shaping.drivers.dm.multi_port_server import MultiPortServer

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def multi_server():
    """Create and start a multi-port TCP mock server."""
    server = MultiPortServer(n_controllers=2)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def micro_dm(multi_server):
    """Create a MicroDM instance connected to the mock server."""
    dm = MicroDM(
        ips=multi_server.get_ips(),
        use_wiring_map=False,
        safety_mode=False,
        timeout=1.0,
    )
    # Override ports to point to the mock server ports
    for ctrl, port in zip(dm._controllers, multi_server.get_ports()):
        ctrl.port = port
    dm.open()
    yield dm
    dm.close()


# =============================================================================
# Helper functions for parsing received records
# =============================================================================


def _parse_voltage_array(data: bytes) -> np.ndarray:
    """Parse a 0x09 command payload into a voltage array.

    Args:
        data: Raw bytes of the command (header + cmd + payload + footer).

    Returns:
        Array of 50 voltages decoded from the payload.
    """
    assert data.startswith(HEADER), f"Missing header: {data.hex()}"
    assert data.endswith(FOOTER), f"Missing footer: {data.hex()}"
    assert data[2] == CMD_SET_ALL_VOLTAGE_BY_ARR, f"Not a 0x09 command: {data.hex()}"

    payload = data[3:-2]  # Skip header(2) + cmd(1) + footer(2)
    assert len(payload) == MAX_CHANNELS * 2, f"Unexpected payload length: {len(payload)}"

    voltages = np.zeros(MAX_CHANNELS)
    for i in range(MAX_CHANNELS):
        high = payload[i * 2]
        low = payload[i * 2 + 1]
        raw = high * 256 + low
        # Inverse of voltages_to_payload: value = (v + 20) / 20 / 3.4 / 3.3 * 65535
        # v = raw / 65535 * 20 * 3.4 * 3.3 - 20
        voltages[i] = raw / 65535.0 * 20.0 * 3.4 * 3.3 - 20.0
    return voltages


def _parse_single_voltage(data: bytes) -> float:
    """Parse a 0x08 command payload into a single voltage value."""
    assert data.startswith(HEADER), f"Missing header: {data.hex()}"
    assert data.endswith(FOOTER), f"Missing footer: {data.hex()}"
    assert data[2] == CMD_SET_ALL_CHANNEL_VOLTAGE, f"Not a 0x08 command: {data.hex()}"

    high = data[3]
    low = data[4]
    raw = high * 256 + low
    return raw / 65535.0 * 20.0 * 3.4 * 3.3 - 20.0